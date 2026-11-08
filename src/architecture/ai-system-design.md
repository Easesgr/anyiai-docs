---
title: 大模型应用系统架构设计：从单次调用到生产级系统
icon: mdi:chart-box-outline
description: 解析大模型应用系统的四层分层架构、三路数据与控制流、三级状态管理分治，以及应对模型推理挂起的全链路超时与降级工程实践。
---

![大模型应用系统架构全景概览](/images/content-rebuild/2e48fcfea2547905/cover.png)

在将大型语言模型（LLM）接入生产业务时，不少团队最开始习惯沿用传统的 Web 架构：前端发起 HTTP 请求，后端业务服务同步查询数据库，调用大模型 SDK 获得字符串，最后组装成 JSON 返回前端。然而，一旦遇到真实用户的并发请求或复杂任务，系统往往迅速暴露问题：业务线程池被长请求耗尽、网关在 60 秒时触发超时熔断、模型账单飞速增长，甚至因模型偶发输出格式错误导致整条链路崩溃。

大模型推理与传统的微服务调用不同：它的单次响应耗时通常在数秒到数十秒之间，输出采用自回归概率生成，单次计算成本与输入输出的 Token 数量紧密绑定。构建稳定的 AI 系统，需要针对这些特征重构系统的分层职责、数据流、状态存储与失败防御链路。

为了直观展示整套架构的运转方式，本文以一个具体的**企业差旅与报销助手**为例贯穿始终：员工在前端输入“我上周在上海出差住了两晚酒店，一共 1400 元，帮我查查是否超标，没超标就提交报销申请”。该任务既包含流式文本回复，又包含内部政策检索、ERP 工具调用，以及需要人工确认的持久化状态推进。

---

## 生产级 AI 系统与传统 Web 架构的核心差异

传统 Web 应用通常面向确定性模式（Schema）和确定性延迟进行设计。典型数据库读写或下游微服务调用一般在数十至数百毫秒内完成，输出结构固定。AI 系统则表现出截然不同的工程特征：

1. **从一次性响应转为持续流式交付**：模型生成 500 个汉字可能需要 3 到 8 秒甚至更久。若采用传统同步等待，应用网关与后端的 I/O 线程会在整个生成周期内保持阻塞。生产系统需要端到端支持非阻塞流式协议（如 Server-Sent Events, SSE），并分别观测首字延迟（Time to First Token, TTFT）与字间生成耗时（Time Per Output Token, TPOT）。
2. **输出结果存在概率波动与结构漂移**：大模型生成文本依赖概率采样，即使将采样温度（Temperature）设为 0，底层浮点计算或微小提示词变动仍可能导致生成内容偏离预期格式。业务系统不能直接假设输出一定是合法的 JSON，必须在消费端配置模式校验、重试或字段提取纠错机制。
3. **计算成本动态化与服务端资源敏感**：传统接口的开销相对固定，而大模型的耗时与费用直接取决于输入提示词（Prompt Tokens）与输出文本（Completion Tokens）的长度。并发请求增多时，服务端显存中的 KV 缓存（KV Cache）会快速增长，导致排队延迟激增。系统入口必须配备细粒度的请求预算限制、频次限制与并发配额。

![传统Web同步请求与AI系统长时流式响应对比](/images/content-rebuild/2e48fcfea2547905/fig-1.png)

在上述差旅助手的场景中，如果员工发起的报销审查需要调用外部模型进行 10 秒的多步思考，系统不能让客户端处于白屏加载状态，而应建立长连接，在网关收到模型首个 Token 时立即推送到前端展示，维持交互反馈。

<ReadMoreLock />

---

## 分层架构设计：交互、编排、网关与存储

为了解耦业务逻辑与底层模型接入的易变性，生产系统通常划分为四层结构：**用户交互层**、**业务编排层**、**模型网关层**与**存储层**。

![大模型应用系统的四层拓扑结构](/images/content-rebuild/2e48fcfea2547905/fig-2.png)

### 1. 用户交互层（Interaction Layer）
负责前端渲染、长连接维持与状态回显：
* **长连接协议**：基于 SSE 或 WebSocket 接收增量 Token 并实时渲染 Markdown。
* **主动打断机制**：当用户在报销助手生成中途点击“停止生成”或关闭页面时，前端立即向后端发送中断信号（Client Abort），后端联动断开上游模型调用，阻断后续无用生成以减少 Token 浪费。需要注意，中断只能截断后续开销，供应商对已完成的输入 Prefill 与已吐出的增量 Token 仍会照常计费，无法全额释放已消耗配额。
* **乐观更新与重连**：网络偶发抖动时，前端利用最后收到的消息序号进行增量重连，防止对话内容被截断。

### 2. 业务编排层（Orchestration Layer）
负责组装具体业务的执行图（Workflow Graph）与状态流转，是大模型系统的大脑：
* **上下文装配**：针对报销助手，编排层负责将员工的角色身份、近期待办，以及知识库中检索出的《差旅住宿报销标准（上海地区上限 700 元/天）》拼装成系统提示词。
* **工具调用调度**：当模型判断可以提交报销时，编排层负责拦截模型生成的函数调用参数，调用内部 ERP 系统的报销接口，并将执行成功的结果回传给模型继续总结。编排与状态流转的基础模式可参考 [/fundamentals/llm-basics/app-system-design.md](/fundamentals/llm-basics/app-system-design.md)。

### 3. 模型网关层（Model Gateway Layer）
作为模型调用的统一基础设施接入层，承担多租户与流量治理职责：
* **协议标准化**：将不同厂商（如 OpenAI、Anthropic）以及本地自建推理服务（如 vLLM、Ollama）的输入输出统一转化为标准化格式，使上层编排代码不绑定特定供应商。
* **配额与计费计量**：按部门或用户记录 Token 消耗，提供统一的密钥轮换、熔断限流和敏感词拦截。关于网关的具体设计，可参考 [/architecture/model-gateway.md](/architecture/model-gateway.md) 与 [/architecture/multi-model-routing.md](/architecture/multi-model-routing.md)。

### 4. 存储层（Storage Layer）
单一大模型应用通常需要多种存储介质协同分工：
* **关系型数据库（RDBMS）**：持久化保存审批单据、员工权限数据、计费结算账单以及完整的审计日志。
* **键值与状态存储（Redis / Checkpoint 数据库）**：存储活跃会话的最近几轮对话上下文，以及长任务执行图的快照状态（State Checkpoint）。
* **外部检索库（向量数据库与倒排索引库）**：保存文档切片后的向量（Embedding）与全文索引，支持知识库混合检索。

---

## 数据流与控制流划分：三路通路分离

AI 系统内部根据交互时延与执行模式的要求，将数据与控制通路划分为三类：**同步流式通路**、**异步排队通路**与**外部工具调用通路**。

![AI系统内同步流式、异步排队与外部工具通路的流向分解](/images/content-rebuild/2e48fcfea2547905/fig-3.png)

### 1. 同步流式通路（Low-Latency Interactive Path）
适用于员工与差旅助手的即时问答（例如咨询“去北京和上海出差的住宿补贴分别是多少”）：
* 编排层组装提示词后，通过网关向模型发起流式请求。
* 模型服务每吐出若干 Token，网关将其封装为 SSE 数据块无缓冲转发给编排层，编排层即刻推向用户浏览器，确保首字延迟可控。
* 该通路中严禁插入大文件解析或高延迟重型运算。详细推流设计可参阅 [/architecture/streaming.md](/architecture/streaming.md)。

### 2. 异步排队通路（High-Throughput Async Path）
适用于耗时数分钟的任务，例如让助手批量扫描过去半年的 50 张发票 PDF，完成 OCR、验真与合规稽核：
* 此类任务如果使用 HTTP 长连接，极易遭遇客户端网络重置或反向代理（如 Nginx 的 60 秒限流）中断。
* 编排层接收任务后生成全局唯一的 `task_id`，将任务推入分布式消息队列（如 Redis Streams、RabbitMQ），并向客户端返回 `202 Accepted`。
* 后台 Worker 从队列拉取任务并发执行，每完成一部分进度将中间结果持久化至检查点存储。客户端通过长轮询或系统通知接收终态。更多关于离线任务的系统设计可见 [/architecture/async-tasks.md](/architecture/async-tasks.md) 与 [/architecture/long-task-execution.md](/architecture/long-task-execution.md)。

### 3. 外部工具调用通路（Tool Execution Loop）
当差旅助手确认员工住宿“两晚共 1400 元未超标（标准每日上限 700 元）”后，模型会产生结构化的工具调用请求（Tool Call）：
* 模型输出中包含调用函数 `submit_expense_claim(amount=1400, city='Shanghai', nights=2)`。
* 编排层截断该输出，暂停直接回复用户，先进行安全鉴权与参数校验（参见 [/architecture/security-permission.md](/architecture/security-permission.md)）。
* 编排层调用后台真实微服务完成数据写入，再将调用结果构造成工具角色消息（`ToolMessage`）塞回上下文，驱动模型生成最终对账反馈（关于工具平台接入可参考 [/interview/ai-coding-interview/tool-design.md](/interview/ai-coding-interview/tool-design.md) 和 [/interview/ai-system-design-interview/agent-tool-platform.md](/interview/ai-system-design-interview/agent-tool-platform.md)）。

---

## 状态管理分级：瞬态、持久化检查点与外部索引

在分布式部署中，系统必须明确区分状态的生存周期，避免将不必要的数据硬编码进上下文或随意堆放在服务内存中。

![AI 系统状态分级与存储拓扑](/images/content-rebuild/2e48fcfea2547905/fig-4.png)

### 三级状态划分对照表

| 状态级别 | 典型数据内容 | 承载媒介 | 存活周期 | 恢复与淘汰机制 |
| :--- | :--- | :--- | :--- | :--- |
| **瞬态上下文 (Transient)** | SSE 响应流句柄、未发送的 Token 缓冲区、请求局部变量 | 应用服务内存 | 单次 HTTP 请求期间 | 连接断开即释放，无需持久化或跨节点同步 |
| **持久化检查点 (Checkpoints)** | 多轮会话历史、任务图执行节点游标、待人工确认的审批快照 | Redis / 关系数据库 | 会话存活期（数小时至数月） | 强一致写入，服务崩溃重启后根据检查点原地重放恢复 |
| **外部认知索引 (External Index)** | 企业差旅制度切片、知识库文档 Embedding、全局倒排词表 | 向量检索库 / 搜索引擎 | 长期系统资产 | 离线异步同步，检索失败可走降级检索或人工提示 |

### 检查点在人机协同（Human-in-the-Loop）中的应用
在报销场景中，即便模型判断未超标，系统规则也往往要求“提交前必须由员工点击确认”。此时编排层不能在内存里无休止地挂起协程，而应将当前的上下文、工具调用参数以及待决状态保存为检查点，中断图的执行。

当员工在前端完成二次确认点击后，请求携带 `checkpoint_id` 到达任意一个无状态业务容器，容器从持久化存储中读取快照并恢复执行，继续调用 ERP 提交报销单。更多关于上下文窗口裁剪与记忆架构的细节，可查阅 [/architecture/context-management.md](/architecture/context-management.md)、[/frontier/agent-memory-system/memory-design.md](/frontier/agent-memory-system/memory-design.md) 以及 [/frontier/agent-memory-system/what-is-memory.md](/frontier/agent-memory-system/what-is-memory.md)。

---

## 跨模块异常传递与兜底设计：应对模型挂起

大模型系统最脆弱的地方在于外部模型供应商的网络抖动或推理过载。假设模型服务因突发排队发生 30 秒静默挂起——既不报错也不吐出任何 Token：如果业务网关使用阻塞等待，数秒之内系统的工作线程池就会被耗尽，导致全站不可用。

![模型供应商挂起30秒场景下的级联超时与降级时序](/images/content-rebuild/2e48fcfea2547905/fig-5.png)

生产环境需要部署三层防护机制：

### 1. 全异步非阻塞调用
业务编排与网关服务之间必须使用异步事件驱动机制（例如 Python 的 `asyncio`、Go 的 Goroutine 或 Java 虚拟线程）。当底层模型等待响应时，系统仅消耗少量协程上下文内存，而不会霸占物理线程，保证服务器仍能处理其他请求。

### 2. 双重超时控制（首字超时与字间超时）
* **首字超时（TTFT Timeout）**：例如设定为 8 秒。连接建立后若 8 秒内未收到供应商返回的第一个 Chunk，立即主动终止连接（Abort），不再无谓等待。
* **字间超时（TPOT / Chunk Timeout）**：例如设定为 3 秒。生成过程中若停顿超过 3 秒，判定为传输中断，及时阻断死连接。

### 3. 降级代码实现示意

在流式交互中，降级处理必须严格区分“首字前超时”与“生成中途中断”：首字到达前未向客户端推送任何业务字符，可安全无缝切换至备用模型；若已开始向客户端推流，直接拼接备用模型从头生成的内容会导致文本错乱与重复，应追加截断提示并平稳闭合连接。

以下 Python 异步示例展示了编排层与网关交互时的双重超时监控与状态感知降级逻辑：

```python
# 标注：本代码为跨模块超时拦截与降级逻辑的实现示意
import asyncio
from typing import AsyncGenerator

class ModelTimeoutError(Exception):
    """模型首字或字间生成超时异常"""
    pass

async def fetch_primary_model_stream(
    prompt: str, 
    first_token_timeout: float = 8.0,
    chunk_timeout: float = 3.0
) -> AsyncGenerator[str, None]:
    """调用主模型，配置首字与字间独立超时保护"""
    # 模拟从底层网关获取流式通道
    gateway_channel = await get_gateway_stream(prompt)
    is_first_token = True
    
    while True:
        current_limit = first_token_timeout if is_first_token else chunk_timeout
        try:
            # 在规定时间内等待下一个 Token Chunk
            chunk = await asyncio.wait_for(gateway_channel.get(), timeout=current_limit)
        except asyncio.TimeoutError:
            raise ModelTimeoutError(f"模型流式传输中断，超时限制: {current_limit}s")
            
        if chunk is None:  # 流传输正常结束
            break
            
        is_first_token = False
        yield chunk

async def safe_orchestration_flow(prompt: str) -> AsyncGenerator[str, None]:
    """业务编排层：包含状态感知的流式主备切换与安全降级"""
    yielded_any = False
    try:
        # 优先使用主力模型
        async for token in fetch_primary_model_stream(prompt, first_token_timeout=8.0):
            yielded_any = True
            yield token
    except (ModelTimeoutError, ConnectionError):
        if not yielded_any:
            # 首字尚未发出，客户端无感知，可安全切换至备用模型从头生成
            yield "[系统提示：主推理通道负载较高，正在切换至备用节点]\n\n"
            try:
                async for token in fetch_backup_model_stream(prompt, timeout=6.0):
                    yield token
            except Exception:
                # 备用模型亦不可用时的终极兜底
                yield "当前差旅审核排队人数较多，您的报销单已自动转入后台异步队列，稍后将在待办中提醒您。"
        else:
            # 已经有局部内容推给前端，不可直接拼入备用模型的全文生成，应平稳闭合流
            yield "\n\n[系统提示：网络连接波动，流式生成中断。已生成内容已保存，您可点击重试或转入后台处理]"
```

在高可用微服务架构中，有关熔断器（Circuit Breaker）与多供应商动态切换的更多细节，可查阅 [/architecture/high-availability.md](/architecture/high-availability.md)。

---

## 可观测性与上线验收指标

传统的 HTTP 状态码 200 和整体响应耗时已经无法完全反映 AI 系统的真实健康状况。一次持续 15 秒但正常输出了 2000 字的请求是健康的，而一次持续 15 秒但首字等待就占了 14 秒的请求则极度损耗用户体验。

![AI应用系统全链路Trace与可观测性验收看板](/images/content-rebuild/2e48fcfea2547905/fig-6.png)

系统上线前，需要打通统一 Trace ID 的分布式追踪，并建立以下维度的可观测看板：

1. **首字延迟（TTFT）**：衡量用户发起请求到屏幕开始出现第一个字符的耗时。生产系统建议监控 P50、P90 与 P99 分位数。
2. **字间延迟（TPOT）**：衡量输出过程中的打字速度。若 TPOT 波动剧烈，通常说明推理服务负载偏高或网络链路发生拥塞。
3. **流式异常中断率（Stream Abort Rate）**：监控在未收到结束标志前，由于网络超时、前端关闭或网关异常而强行终止的会话比例。
4. **Token 开销核算**：按业务模块、Prompt 与 Completion 维度分别统计每日消耗的 Token 数，结合价格模型计算实时成本（详见 [/architecture/token-cost.md](/architecture/token-cost.md)）。
5. **编排图节点执行耗时**：将知识检索、工具调用（如 ERP 查询）与模型推理拆分成独立的 Trace Span，便于精准定位系统瓶颈。更详细的指标落地与输出质量评测可参考 [/architecture/logging-trace.md](/architecture/logging-trace.md) 与 [/architecture/ai-evaluation.md](/architecture/ai-evaluation.md)。

---

## 架构自测题

* **测试题 1**：当外部模型供应商发生推理节点挂起、长达 30 秒无任何响应时，业务网关和编排层应分别依靠哪些机制在 8 秒内主动截断并释放资源，而不是等待整整 30 秒？
* **测试题 2**：在需要用户人工核验（如确认报销金额）的人机协同流程中，为何不能直接使用应用内存挂起等待？应如何利用持久化检查点完成状态解耦与断点恢复？

下一篇我们将展开解析 [模型网关：协议归一、多租户流控与模型路由](/architecture/model-gateway.md)，探讨如何统一异构模型标准并实现多租户配额管理。
