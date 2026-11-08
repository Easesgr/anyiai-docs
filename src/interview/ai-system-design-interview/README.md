---
title: AI 系统设计怎么答？面试思路与核心考点
icon: mdi:server
tag:
- 系统设计面试
- 面试题
- 系统设计
description: 梳理AI系统设计面试的口述回答框架，从输入输出延迟、概率生成与成本约束切入，掌握五步解题法与核心设计清单。
---

![AI 系统设计面试指南全景](/images/content-rebuild/4d0e7fccc8f1667b/cover.png)

在技术面试中遇到“如何设计一个基于大模型的系统”时，可先用一段简短结构化的语言破题：

> “设计 AI 系统与传统分布式系统的核心区别在于，除了常规的并发与高可用，系统还要在应用层兜底大模型带来的三大物理约束：**长延迟与非对称吞吐**、**输出的概率性与格式漂移**，以及**按 Token 计算的显存与调用成本**。因此设计重点不是写提示词，而是围绕这三点建立确定性的工程架构。”

以一个典型的**企业内部知识问答与工单处理助手**为例，系统既要从海量内部文档中检索事实作答，又要根据用户要求调用工单系统执行动作。下面以此贯穿全流程，拆解面试回答的推进步骤与底层机制。

---

## 核心考点：大模型带来的三项系统约束

传统微服务系统的接口响应通常在几十到数百毫秒，后端服务主要关注无状态扩容、网络 I/O 与数据库连接池。引入大语言模型（LLM）后，系统行为发生了显著变化：

1. **吞吐与延迟的不对称性**：
   大模型推理分为两个阶段：**输入处理阶段（Prefill）**和**逐字生成阶段（Decode）**。
   - Prefill 阶段把所有输入 Token 并行送入显卡计算，受算力限制，首字延迟（TTFT，Time to First Token）通常在数百毫秒到数秒。
   - Decode 阶段每生成一个新 Token 都要读写一次之前所有 Token 的键值缓存（KV Cache），属于显存带宽受限操作，生成每个字的时间（TPOT，Time per Output Token）相对固定。生成数百字可能持续数秒甚至数十秒。
   - 这意味着服务必须采用流式传输（如 Server-Sent Events, SSE），网关层需要管理长连接生命周期，不能照搬短请求的超时与连接池配置。
2. **输出的概率波动与格式漂移**：
   大模型基于词表概率采样，即使设置较低的采样温度（Temperature），也无法保证 100% 遵守 JSON 结构或枚举值要求。系统必须由程序逻辑完成结构校验、语法修复和兜底分支。
3. **上下文长度与成本约束**：
   输入与输出均按 Token 计费，且输入上下文过长会导致 KV Cache 显存暴涨。系统必须在无损关键信息的前提下控制注入模型的上下文长度。

---

## 面试解题五步法

回答 AI 系统设计题时，建议遵循五步结构化展开：

![AI 系统设计面试解题五步法](/images/content-rebuild/4d0e7fccc8f1667b/fig-1.png)

### 第一步：需求澄清与规格估算
首先向面试官确认系统的功能边界与性能指标，并根据 Token 换算系统吞吐。

**假设场景与输入参数**：
- 峰值并发请求：$QPS = 50$；
- 用户平均提问长度：$200\ \text{Tokens}$；
- 检索增强（RAG）召回上下文：拼接 3 篇文档片段，共约 $1500\ \text{Tokens}$；
- 系统单次期望回复长度：平均 $500\ \text{Tokens}$。

**规格推算（示例）**：
- **输入峰值吞吐（Prefill 压力）**：
  $$\text{输入吞吐} = 50 \times (200 + 1500) = 85,000\ \text{Tokens/s} \approx 5,100,000\ \text{TPM (Tokens Per Minute)}$$
- **输出峰值吞吐（Decode 压力）**：
  $$\text{输出吞吐} = 50 \times 500 = 25,000\ \text{Tokens/s} \approx 1,500,000\ \text{TPM}$$

![AI 系统规格估算与参数推导示意](/images/content-rebuild/4d0e7fccc8f1667b/fig-3.png)

这一估算能够直接指导后续的资源选型：
1. 如果调用外部商业 API，必须提前向供应商申请对应 TPM 的并发配额，并配置客户端排队限流。
2. 网关需按并发连接数规划长连接容量。若单请求生成速率约 25~50 Tokens/s，回复 500 Tokens 需持续 10~20 秒；按利特尔法则计算，QPS 为 50 时在途活跃长连接数约为 $50 \times (10 \sim 20) = 500 \sim 1000$ 个，网关需配置对应规格的流式连接池，并关闭下游代理（如 Nginx）的默认响应缓冲。

### 第二步：高层架构与核心数据流
按职责将系统划分为四层：

1. **接入与网关层**：处理鉴权、请求限流、SSE 长连接保持与多模型动态路由。
2. **编排与检索层**：
   - **意图分流器**：判断用户请求是闲聊、知识检索还是工单操作。
   - **检索模块**：向量数据库（稠密检索）与 BM25（稀疏全文检索）做混合召回，再经重排模型（Reranker）压缩至 1500 Tokens 以内。
   - **上下文拼装**：将系统提示词、对话历史、检索结果装配入最终 Prompt。
3. **模型执行层**：商业 API 统一适配网关或私有推理服务集群（如部署 vLLM 的服务）。
4. **存储与状态层**：会话状态库（Redis）、长效知识库（向量库与文档库）、审计日志与追踪数据库。

### 第三步：核心模块深潜
选择题目中的关键链路展开。以“知识检索与工单执行”为例：
- **文档切分与混合检索**：不能仅做固定长度切分，应保留段落语义边界并记录元数据（如租户 ID、部门权限）；采用“全文检索 + 向量检索”混合召回，防止专有名词或缩写在单一向量检索中漏召回，再通过重排模型挑选得分最高的片段。
- **工具调用的可靠性**：大模型决定调用“创建工单”工具时，网关不直接信任其生成的参数，而是经过 JSON Schema 校验。若缺字段，由程序提示模型补全，并在调用外部真实 API 前做权限校验与幂等键检查。

### 第四步：容错、降级与安全边界
- **重试与熔断**：当下游模型返回 429（限流）或 504（超时）时，网关采用指数退避重试；若主模型集群不可用，快速降级至备用轻量模型，或直接返回预置提示。
- **格式校验兜底**：模型输出未通过结构化校验时，最多追加一次提示请求模型修复；若仍失败，由代码回退至保守的纯文本展示或人工处理分支。
- **数据隔离与输入过滤**：在向量检索阶段利用元数据（Metadata Filter）强制带上用户权限标识，杜绝越权访问；在输入端检测提示词注入，输出端建立敏感词与格式过滤器。

### 第五步：权衡与指标验收
清晰交代设计中的妥协与指标平衡：
- **成本与质量权衡**：意图分类采用轻量小模型降低成本，复杂综合回答才路由至大模型；
- **延迟与精度权衡**：重排模型选用轻量交叉熵模型，平衡召回精度与首字延迟；
- **核心监控指标**：首字延迟 P99、生成速率（Tokens/s）、接口 429 报错率、工具调用有效执行率。

---

## 基础设施与业务场景的分层依赖

面试中的各种业务题目不是零散孤立的，它们共享同一套底层基础设施：

![AI 系统模块依赖拓扑与分层结构](/images/content-rebuild/4d0e7fccc8f1667b/fig-2.png)

- **底座支撑组件**：多模型网关、上下文管理、熔断降级、权限隔离与追踪审计，是所有 AI 应用的通用底座。
- **上层业务场景**：企业问答、智能客服、内容生成、面试助手等场景，本质上是依据不同业务约束，对底座组件进行差异化装配。

---

## 专题题目目录索引与推荐学习顺序

建议按照“基础设施设计 $\rightarrow$ 可靠性与安全 $\rightarrow$ 综合业务场景”的顺序展开复习：

### 第一阶段：技术选型与基础设施设计
- [/interview/ai-system-design-interview/rag-agent-workflow.md](/interview/ai-system-design-interview/rag-agent-workflow.md)：厘清 Prompt、RAG、工作流与 Agent 的选型边界。
- [/interview/ai-system-design-interview/multi-model-gateway.md](/interview/ai-system-design-interview/multi-model-gateway.md)：掌握多模型路由、负载均衡与流式响应聚合。
- [/interview/ai-system-design-interview/session-context-management.md](/interview/ai-system-design-interview/session-context-management.md)：设计高吞吐长对话的滑动窗口与状态摘要压缩。

### 第二阶段：生产级容错、安全与可观测性
- [/interview/ai-system-design-interview/failure-handling.md](/interview/ai-system-design-interview/failure-handling.md)：模型超时、限流、输出格式漂移的兜底重试与降级。
- [/interview/ai-system-design-interview/security-privacy.md](/interview/ai-system-design-interview/security-privacy.md)：提示词注入防御、多租户向量隔离与隐私脱敏。
- [/interview/ai-system-design-interview/logging-monitoring.md](/interview/ai-system-design-interview/logging-monitoring.md)：全链路 Trace、Token 计费核算与线上效果持续评估。

### 第三阶段：高频业务系统端到端实战
- [/interview/ai-system-design-interview/agent-tool-platform.md](/interview/ai-system-design-interview/agent-tool-platform.md)：工具沙箱、鉴权机制与 OpenAPI 动态注册中心。
- [/interview/ai-system-design-interview/knowledge-base-qa.md](/interview/ai-system-design-interview/knowledge-base-qa.md)：生产级 RAG 架构、文档清洗、混合检索与重排。
- [/interview/ai-system-design-interview/multi-tenant-platform.md](/interview/ai-system-design-interview/multi-tenant-platform.md)：大型企业多组织架构下的配额分配与知识库逻辑隔离。
- [/interview/ai-system-design-interview/smart-customer-service.md](/interview/ai-system-design-interview/smart-customer-service.md)：混合意图识别、转人工仲裁与高可用工单协同。
- [/interview/ai-system-design-interview/content-generation-platform.md](/interview/ai-system-design-interview/content-generation-platform.md)：异步批量作业调度、多步骤审核与模板组装流水线。
- [/interview/ai-system-design-interview/ai-interview-assistant.md](/interview/ai-system-design-interview/ai-interview-assistant.md)：确定性状态机驱动的多轮考查、动态探针与评分校准。

若希望深入理解架构演进与前沿理论，可同步参考架构主专栏 [/architecture/README.md](/architecture/README.md)。

---

## 常见面试沟通陷阱与高频追问

### 常见沟通陷阱
1. **一上来就提模型微调**：
   面试官询问“如何提高回答专业度”时，若直接回答“拿业务数据做全量或 LoRA 微调”，通常会被追问知识更新成本。微调适合固化特定输出风格或复杂格式，不适合频繁变动的事实知识，事实类需求应优先考虑 RAG。
2. **把大模型当确定性代码调用**：
   在设计方案中默认模型总能输出合法的 JSON，未设计格式校验失败、字段缺失或截断时的重试与降级逻辑。
3. **按普通 HTTP 短请求设计网关**：
   忽略了流式返回（SSE）的长连接特性。大模型逐字生成可能耗时数十秒，若网关代理层（如 Nginx、API Gateway）开启了缓冲或超时时间设得过短（如 5 秒），会导致连接中断或客户端无法实时获取首字。

### 高频反向追问要点
- **追问 1：当下游模型响应延迟骤增到 30 秒，网关如何避免请求堆积导致服务雪崩？**
  - **要点**：必须在网关层设置动态超时与并发信号量隔离；对非核心请求开启熔断器，快速返回降级文案；利用异步任务队列处理耗时较长的离线任务。（详见 [/interview/ai-system-design-interview/failure-handling.md](/interview/ai-system-design-interview/failure-handling.md)）
- **追问 2：多租户企业知识库中，普通员工提问是否可能越权看到管理层的敏感文档？**
  - **要点**：纯向量相似度检索不具备权限判断能力。权限隔离必须前置，即在向向量数据库发起查询时，把租户 ID 和用户角色组作为硬过滤条件（Metadata Filter），检索召回范围仅限于已授权切片。（详见 [/interview/ai-system-design-interview/security-privacy.md](/interview/ai-system-design-interview/security-privacy.md)）
- **追问 3：系统上线后，如何持续评估回答质量？**
  - **要点**：建立分层评估体系。实时层监控基础指标（延迟、报错率、用户点赞/点踩率）；离线层抽取真实交互记录，构建金标准测试集，使用大模型结合规则打分评估上下文相关度与忠实度，替代全量人工抽检。（详见 [/interview/ai-system-design-interview/logging-monitoring.md](/interview/ai-system-design-interview/logging-monitoring.md)）
