---
title: AI 应用如何做日志、监控、成本统计和效果评估？
icon: mdi:chart-box-outline
tag:
- 系统设计面试
- 日志监控
- 成本统计
description: 梳理大模型应用的可观测性体系：包含 OpenTelemetry 链路追踪、流式性能监控指标、Token 成本核算以及双轨在线质量评估。
---

## 面试标准口述模板

在系统设计面试中被问到“如何设计大模型应用的日志、监控、成本统计与效果评估体系”时，可以用以下逻辑直接口述：

> “监控大模型系统不能只看传统微服务的 CPU、内存和 HTTP 状态码，需要围绕大模型运行特性建立四个维度的支撑：
>
> 1. **调用链追踪**：基于 OpenTelemetry 规范，把提示词渲染、向量检索、模型流式生成和工具调用串成树状 Span 结构，用 Trace ID 贯穿多轮检索与工具执行链路。
> 2. **流式性能指标**：重点采集首字延迟（TTFT）、单 Token 解码延迟、每秒生成 Token 数（TPS）以及上下文缓存命中率，把报错按限流、超时、超长和安全拦截分类。
> 3. **异步成本核算**：网关在流式响应结束时提取精确的 Token 用量，异步投递到消息队列，根据模型单价与租户阶梯价格计算账单，实现预算监控与实时告警。
> 4. **在线效果评估**：采用双轨制——全量采集用户的隐式行为（如重新生成、点踩、复制），配合小比例异步抽样的裁判大模型（LLM-as-a-Judge）评估相关性与幻觉，既不阻塞主调用，又能监控质量漂移。
> 5. **合规存储**：全量输入输出在记录前完成敏感个人信息（PII）脱敏，冷热数据分层保存，兼顾实时排障与审计要求。”

![AI可观测性大屏：串联追踪、指标、成本与在线评估](/images/content-rebuild/22cc78c1f9c8d6a8/cover.png)

---

## 贯穿示例：智能售后客服助手的请求旅程

为了让整套体系具体化，我们以一个典型的“电商售后客服助手”为例：

用户提问：*“我昨天买的吸尘器还没发货，能帮我催一下或者退款吗？”*

系统处理这一请求通常需要多步协作：
1. **检索历史与知识**：把用户问题转成向量，去知识库检索退款政策；同时根据用户 ID 查订单系统。
2. **调用大模型决策**：组装包含提示词、退款规则和订单信息的上下文，调用模型生成回复或触发工具调用。
3. **执行工具**：模型决定调用 `urge_shipment(order_id="100293")` 催发货。
4. **流式输出**：模型根据工具返回结果，通过流式响应向客户端输出最终解答。

如果没有任何追踪，这一连串操作在传统日志里只是一条耗时 3.5 秒的 HTTP 200 记录。排查问题时，你无法知道究竟是向量检索慢、模型排队、还是订单查询接口超时。

---

## 分布式调用追踪：基于 OpenTelemetry 的 Span 建模

在分布式追踪中：
- **Trace** 表示一个请求从头到尾的完整调用过程，拥有全局唯一的 `Trace ID`。
- **Span** 表示调用过程中的某一个具体操作单元（例如一次数据库查询、一次模型推理），拥有 `Span ID` 和 `Parent Span ID`。

大模型系统把单次交互拆解为树状的嵌套 Span：

```text
[Trace ID: 4bf92f3577b34da6a3ce929d0e0e4736]
└── Span: customer_service_agent (根操作，总耗时 3.2s)
    ├── Span: prompt_template_render (耗时 2ms)
    ├── Span: rag_retrieval (向量库召回退款规则，耗时 45ms，top_k=3)
    ├── Span: llm_reasoning (模型决策调用工具，耗时 820ms，Prompt Tokens=1250, Completion Tokens=32)
    ├── Span: tool_execution (调用催发货接口，耗时 180ms)
    └── Span: llm_final_generation (模型流式生成最终回复，耗时 2153ms，TTFT=310ms, Completion Tokens=210)
```

### 标准语义属性定义

为了在 Jaeger、Prometheus、ClickHouse 等分析后端中进行聚合分析，每个与大模型相关的 Span 需要记录约定的语义属性字段：

```json
{
  "gen_ai.system": "openai",
  "gen_ai.request.model": "gpt-4o",
  "gen_ai.request.temperature": 0.3,
  "gen_ai.request.max_tokens": 1024,
  "gen_ai.usage.prompt_tokens": 1480,
  "gen_ai.usage.completion_tokens": 242,
  "gen_ai.usage.total_tokens": 1722,
  "gen_ai.response.finish_reasons": ["stop"],
  "gen_ai.prompt_cache.hit": true,
  "gen_ai.prompt_cache.cached_tokens": 1024
}
```

当助手执行外部沙箱或者微服务工具时（参见[/interview/ai-system-design-interview/agent-tool-platform.md](/interview/ai-system-design-interview/agent-tool-platform.md)），网关或调用框架需在 HTTP 请求头中注入 `traceparent`，保证上下文透传到外部系统，调用链不会中断。

![基于OpenTelemetry的AI调用链Span嵌套模型](/images/content-rebuild/22cc78c1f9c8d6a8/fig-1.png)

---

## 核心指标采集矩阵：从传统吞吐到流式度量

绝大多数交互式大模型应用采用流式传输（如 Server-Sent Events，SSE）。传统 API 监控的“整包响应延迟”无法真实反映用户的体感等待。

### 关键度量指标定义

1. **首字延迟（Time To First Token, TTFT）**：
   - 定义：从用户发出请求到客户端收到第一个内容字符的时间。
   - 耗时构成：网络排队 + 路由网关处理 + 提示词预处理 + 模型的预填充计算（Prefill）。在 Prefill 阶段，模型必须一次性并行计算整个输入 Prompt 的注意力矩阵，因此 Prompt 越长，TTFT 越长。
2. **单 Token 生成延迟与解码吞吐（TPS）**：
   - 模型吐出第一个字后，进入自回归解码（Decode）阶段。每次生成一个 Token 并通过 SSE 推送一次。
   - 假设从首字到达至输出结束共生成 $N_{\text{completion}}$ 个 Token，该阶段总耗时为 $T_{\text{decode\_total}}$，则平均解码速率为：
     $$\text{TPS} = \frac{N_{\text{completion}} - 1}{T_{\text{decode\_total}}}$$
     每个 Token 的平均输出间隔为 $T_{\text{per\_token}} = \frac{1}{\text{TPS}}$。
3. **上下文缓存（Prompt Cache）命中率**：
   - 在服务端开启上下文缓存时，系统 Prompt 和长期不变的召回知识无需每次重复进行全量 Prefill：
     $$\text{Cache Hit Rate} = \frac{\text{Tokens}_{\text{cached}}}{\text{Tokens}_{\text{prompt}}}$$
   - 缓存命中率直接降低 TTFT 并节省输入 Token 费用。
4. **错误分类统计**：
   - 429（模型厂商限流）：表明触发并发或 RPM/TPM 上限，需在网关层做重试与分流（参见[/interview/ai-system-design-interview/multi-model-gateway.md](/interview/ai-system-design-interview/multi-model-gateway.md)）。
   - 504 / 超时：推理集群过载或网络中断。
   - Context Window Exceeded：输入上下文超出模型最大支持窗口，需触发截断策略。
   - 内容审核拦截（Content Moderation Filtered）：触发安全护栏或提供商合规策略，配合[/interview/ai-system-design-interview/failure-handling.md](/interview/ai-system-design-interview/failure-handling.md)做友好兜底。

![流式推理下的指标采集时序与分类图](/images/content-rebuild/22cc78c1f9c8d6a8/fig-2.png)

---

## Token 与成本实时计量流水线

大模型 API 的调用计费是真金白银的显性成本。对于多租户或多业务线系统，必须做到精确到每个请求的用量归属与预算告警。

### 异步无锁计量架构

如果在每一次 SSE 推送字符时都去操作数据库，极高的读写频率会拖垮系统。标准做法是解耦流式传输与计费：

1. **网关流式捕获**：网关代理与模型提供商的长连接。当捕获到最后一个 SSE 数据块（通常带有完整的 `usage` 字段）或连接正常结束时，网关提取出本次请求的实际输入与输出 Token 数量。
2. **投递消息队列**：网关组装轻量事件，异步投递到 Kafka 的 `llm-token-metering` 主题，包含字段：`trace_id`、`tenant_id`、`model`、`prompt_tokens`、`completion_tokens`、`cached_tokens`。
3. **流批分流核算**：
   - **实时处理（如 Redis / Flink）**：维护租户当前的当月/当日累计消耗金额，与租户配额（Quota）比对。一旦超过预警阈值，发送邮件或飞书告警；一旦超过硬限额，网关拒绝后续新请求。
   - **数仓落盘（如 ClickHouse）**：批量入库供后续多维财务报表查询、部门分摊和调用回溯。

### 成本计算模型与对比算例

设客服助手每天承接 10,000 次咨询：
- 系统提示词与固定企业知识库前缀：1,200 Tokens。
- 用户新输入提问：300 Tokens（总输入 1,500 Tokens）。
- 模型平均输出：300 Tokens。
- 基础价格假设：输入单价 $P_{\text{prompt}} = 0.002 \text{ 美元} / 1,000 \text{ Tokens}$；输出单价 $P_{\text{comp}} = 0.006 \text{ 美元} / 1,000 \text{ Tokens}$。
- 缓存优惠假设：命中 Prompt Cache 的 Token 享受 $50\%$ 价格折扣（即 $0.001 \text{ 美元} / 1,000 \text{ Tokens}$）。

#### 场景 A：无缓存机制（全量重新计算 Prefill）
$$\text{Cost}_{\text{daily}} = 10,000 \times \left( \frac{1,500}{1,000} \times 0.002 + \frac{300}{1,000} \times 0.006 \right) = 10,000 \times (0.003 + 0.0018) = 48.00 \text{ 美元/天}$$

#### 场景 B：前缀缓存命中率 80%（前 1,200 Tokens 中有 80% 命中缓存）
- 未命中缓存的输入部分（包含用户 300 新 Token + 未命中前缀 $1,200 \times 20\% = 240$ Token，共 540 Token）：
  $$10,000 \times \left( \frac{540}{1,000} \times 0.002 \right) = 10.80 \text{ 美元}$$
- 命中缓存的前缀部分（$1,200 \times 80\% = 960$ Token）：
  $$10,000 \times \left( \frac{960}{1,000} \times 0.001 \right) = 9.60 \text{ 美元}$$
- 输出生成部分：
  $$10,000 \times \left( \frac{300}{1,000} \times 0.006 \right) = 18.00 \text{ 美元}$$
- 场景 B 每日总成本：$10.80 + 9.60 + 18.00 = 38.40 \text{ 美元/天}$。相比场景 A 每日节省开销 $20\%$。

![实时成本核算与预算风控流水线架构图](/images/content-rebuild/22cc78c1f9c8d6a8/fig-3.png)

<ReadMoreLock />

---

## 在线质量度量：隐式行为与异步 LLM-as-a-Judge

传统 API 只要返回 HTTP 200 就代表调用成功，但大模型返回 200 可能完全是一句事实胡扯或无关废话。因此，必须建立质量度量机制。

### 1. 用户隐式行为度量（低成本全量覆盖）

让用户主动在界面上点赞、点踩的比例在生产环境中通常低于 $1\%$。更可靠的方法是收集用户的操作行为：

- **负向行为**：
  - 点击“重新生成”（Regenerate）。
  - 快速关闭或跳出（在内容生成完 3 秒内关闭对话框）。
  - 撤回或立刻修改上一轮提示词（说明模型没理解意思）。
- **正向行为**：
  - 一键复制内容（Copy）。
  - 对话轮数自然深入（基于该回答继续提问，而不是重开新会话）。
  - 停留阅读时间与生成文本长度呈合理正比。

前端通过打点把行为与该轮会话的 `Trace ID` 绑定，后端聚合成“满意度参考指数”，作为版本迭代的效果大盘。

### 2. 异步 LLM-as-a-Judge 抽检（高精度采样审计）

“大模型做裁判”（LLM-as-a-Judge）是指使用性能更强或经过专门微调的评估模型，对实际生产产生的问答对进行自动评分。

由于裁判模型本身的调用也是耗时且昂贵的，生产上**绝不能把它放在用户请求的主链路里同步拦截**，而应采用**异步分层抽检**：

- **采样规则**：
  - 正常流量随机抽样 $1\% \sim 3\%$ 作为基线监控；
  - 触发了点踩、多次重试或被安全模块标记的疑似坏用例（Bad Case），$100\%$ 自动送入评估队列。
- **核心评估维度**：
  - **相关性（Relevance）**：回答是否准确回应了用户的提问要求？
  - **真实性与依据忠实度（Faithfulness）**：针对 RAG 场景，回答中的事实陈述是否能从召回的上下文中找到依据，是否存在凭空捏造（幻觉）？
  - **安全性与规范遵循（Safety / Format）**：是否违背预设的角色设定，是否泄露系统提示词？
- **规避裁判模型的认知偏差**：
  - **位置偏差（Position Bias）**：如果是两个版本的 A/B 对比打分，裁判模型容易偏向排在前面的选项。需在评估时随机调换顺序，或正反评两次取平均分。
  - **长度偏好（Verbosity Bias）**：大模型往往认为字数多的回答更完整。提示词中必须明确要求：“重点评估事实是否准确、废话是否精简，不要仅仅因为篇幅长而给高分”。

![线上质量评估双轨制：隐式反馈与异步LLM-as-a-Judge抽检](/images/content-rebuild/22cc78c1f9c8d6a8/fig-4.png)

---

## 数据合规与审计日志存储设计

大模型日志中直接包含用户输入的明文对话，一旦发生数据泄漏会引发合规风险（参见[/interview/ai-system-design-interview/security-privacy.md](/interview/ai-system-design-interview/security-privacy.md)）。

### PII 敏感信息脱敏

在日志写入持久化介质前，必须通过脱敏流水线：
- **规则正则匹配**：快速识别并打码中国大陆手机号（如 `138****0000`）、身份证号、邮箱地址、API Key（如以 `sk-` 开头的秘钥串）。
- **轻量 NER 命名实体识别**：对人名、企业内部代号等非固定格式的敏感词进行占位符替换。

### 存储生命周期分层

全量对话日志体积巨大，需采用分层存储策略平衡查询性能与硬件成本：

| 存储层级 | 存储介质 | 保留时间 | 核心用途 | 访问权限控制 |
| :--- | :--- | :--- | :--- | :--- |
| **热存储层** | Elasticsearch / OpenSearch | 7 天 | 线上实时故障排查、基于 Trace ID 查明细、排查最近报错 | 仅限授权值班工程师，只读脱敏数据 |
| **温存储层** | ClickHouse / 云数仓 | 30 ~ 90 天 | Token 成本归属分摊、多版本 A/B 质量统计、业务趋势报表 | 业务分析师与产品负责人 |
| **冷存储层** | 对象存储（S3 / OSS）分块压缩 | 180 ~ 365 天 | 法规合规审计、离线安全回溯、作为后续模型微调的清洗候选池 | 极高权限离线审批使用，全盘加密存储 |

结合会话管理机制（参见[/interview/ai-system-design-interview/session-context-management.md](/interview/ai-system-design-interview/session-context-management.md)），在多租户系统下，日志存储需包含 `tenant_id`，以支持租户注销时按租户维度一键物理清除历史数据。

![合规审计日志脱敏分层存储与检索生命周期](/images/content-rebuild/22cc78c1f9c8d6a8/fig-5.png)

---

## 面试考点演练：线上回答相关性暴跌根因排查

### 题目场景

> **面试官**：“某在线知识库问答系统（RAG），监控大屏显示昨天下午 14:00 开始，用户点踩率从 $1.5\%$ 飙升到了 $11\%$，异步 LLM-as-a-Judge 评估的相关性得分从 4.7 跌到了 3.2。但是整个服务的 HTTP 5xx 错误率没有增加，QPS 也一切正常。你该如何设置告警，并沿着可观测链路快速定位根因？”

### 排查步骤

1. **看输入与输出 Token 长度分布**：
   - 查看 14:00 之后的监控，如果 Prompt 的平均 Token 长度出现断崖式下跌，说明上游向量检索召回（Retrieval）可能完全没有召回内容，LLM 在缺少上下文资料的情况下凭空回答。
2. **检查向量检索 Span 的相似度分数**：
   - 检索链路的 Span 中通常会记录召回文档的相似度得分（Score）。如果相似度得分突然全线跌破阈值，说明检索侧发生异常（参见[/interview/ai-system-design-interview/knowledge-base-qa.md](/interview/ai-system-design-interview/knowledge-base-qa.md)）。
3. **核查变更事件记录**：
   - 检查下午 14:00 是否有发布记录：
     - **知识库切分逻辑变更**：是否新导入了一批长文档且切分损坏？
     - **Embedding 模型版本漂移**：最常见的事故之一是在配置中热更新了查询侧的 Embedding 模型，而向量数据库中存储的仍然是老模型生成的向量，导致空间距离彻底失真。
     - **Prompt 模板发布**：上游提示词发布是否破坏了格式占位符，导致用户提问没有被正确拼入上下文？
4. **调取典型 Bad Case Trace 验证**：
   - 过滤出打分低于 3 分的请求 `Trace ID`，打开对应的 Span 树：
     - 展开 `rag_retrieval` Span，确认其返回的文档与用户提问是否相关；
     - 展开 `llm_generation` Span，检查实际送入模型的最终 Prompt，一眼即可确认是检索失效还是模型提示词破坏。

### 告警规则设计

- **复合指标告警**：在 15 分钟滑动窗口内，当 `Relevance_Score < 3.5` 的比例超过 $8\%$ 且有效评估样本数 $N \ge 30$ 时，触发 P2 业务质量告警。
- **依赖抑制**：如果同时伴随向量数据库集群宕机或网络连通性报错，优先触发底层基础设施 P0 告警，抑制上层质量劣化告警，避免告警风暴。

![质量劣化故障排查决策树与多维排障拓扑](/images/content-rebuild/22cc78c1f9c8d6a8/fig-6.png)

---

## 深度追问与常见误区

### 追问 1：在流式生成中，用户中途直接关闭了网页，指标和成本应该如何记录？

**解答**：
1. 当用户关闭网页，网关检测到 TCP 连接断开（或捕获到 `ClientClosedRequest` 事件）。
2. **中止下游推理**：网关必须立刻向下游大模型接口发出 Cancel / Abort 信号。因为如果下游还在继续计算 Decode，GPU 算力将被白白浪费。
3. **指标分流记录**：该请求的 Trace 状态标记为 `CANCELLED`。记录两组 Token 指标：`tokens_sent_to_client`（实际已推送给用户的字数）和 `tokens_billed_by_provider`（服务商已计算生成的字数）。
4. **计费依据**：以提供商实际返回的账单为准扣减租户额度；同时在业务大屏上关注“中途取消率”，如果该指标异常偏高，通常说明首字延迟太长或模型回答开头废话过多，导致用户失去耐心。

### 追问 2：为什么直接用“GPU 利用率”来监控大模型推理服务的健康度是不可靠的？

**解答**：
传统监控中常看 `nvidia-smi` 的 GPU Utilization，它只代表在采样周期内是否有计算核心在跑，无法代表计算效率：
- **显存带宽受限（Memory-Bound）**：大模型解码（Decode）阶段是典型的内存搬运密集型任务。显存与计算核心之间频繁交换 KV Cache，GPU 利用率可能显示 $90\%$，但算力利用率（MFU）可能极低。
- **无效等待与阻塞**：当推理引擎的 KV Cache 显存耗尽发生页交换（Swap），或者在处理长文本 Prefill 时陷入等待队列，GPU 计算核心可能看似在持续运转，但整个系统的 TPS（每秒输出字数）却可能断崖式下跌。
- **合理指标**：大模型推理服务应监控专用引擎指标，如：**KV Cache 显存占用率**、**等待请求队列长度（Queue Size）**、**Prefill 批次大小**和**系统实际输出 TPS**。

### 常见设计误区

- **误区一：把日志入库代码写在主请求路径里**。大模型调用的首字和生成延迟本就敏感，如果在同一个协程或线程里同步往 Elasticsearch 或 MySQL 写全量日志，一旦存储系统出现 I/O 抖动，会瞬间导致网关连接池耗尽、新请求全部排队。日志必须通过内存环形缓冲（Channel）异步批量投递。
- **误区二：在主请求链路上同步挂载 LLM-as-a-Judge 做质检**。为了防止模型胡说八道，在把回复推给用户前，先调一个大模型“审阅”一遍。这会导致系统的端到端延迟翻倍、API 成本翻倍。在实际架构中，在线阶段只允许运行轻量级的敏感词或分类模型进行硬性拦截，大模型裁判仅用于**异步旁路分析与事后质量评估**。
- **误区三：单次复杂请求只打一条单层日志**。Agent 多轮调用、多次工具执行、多次意图判断如果全打在一条日志里，只记录一个总耗时，一旦出错完全看不出是哪一次工具调用返回了脏数据。必须采用标准的嵌套 Span 树（参见[/architecture/logging-trace.md](/architecture/logging-trace.md)）。
