---
title: 大模型系统如何做日志与分布式链路追踪？端到端可观测性实践
icon: mdi:chart-line
description: 基于OpenTelemetry解析大模型系统链路追踪，涵盖W3C上下文传播、流式TTFT与ITL指标、双轨日志脱敏及排障下钻。
---

![封面：分布式调用链与可观测性看板全景](/images/content-rebuild/00423fb95e91547f/cover.png)

在传统微服务系统中，一次 HTTP 请求的成败通常由状态码（如 `200 OK` 或 `500 Internal Server Error`）、响应耗时和异常堆栈决定。但在包含大语言模型（LLM）、检索增强生成（RAG）和智能体（Agent）的 AI 系统中，传统观测方式经常遇到盲区：

- 模型接口返回了 HTTP `200`，输出的内容却可能是格式畸变的 JSON、死循环的工具调用参数，甚至是答非所问的幻觉；
- 一次用户问答总耗时长达 3 秒，单看 Nginx 日志无法判断瓶颈究竟是在知识库向量检索、外部 API 排队、模型的提示词预填充（Prefill），还是自回归解码（Decode）过程；
- 每次请求消耗的 Input Token 和 Output Token 直接对应账单成本，如果链路追踪没有捕获这些元数据，后续就无法完成细粒度的租户成本分摊。

**结论是：AI 系统的可观测性不能只监控网络连通性，而必须将调用链（Trace）、离散指标（Metrics）与上下文日志（Logs）深度绑定，贯穿网关、检索、工具执行和长连接模型流式推理的全生命周期。**

为了清晰说明这套机制，本文将以一个典型的“企业技术支持智能助手”为贯穿示例：用户发起问题后，网关首先完成鉴权并创建链路标识，编排层检索知识库文档，随后驱动大模型判断是否需要调用工单查询工具，最后以流式输出的方式向客户端推送 101 个 Token 的回答。

---

## 1. AI 系统的时延与状态特殊性

为什么不能直接沿用旧有的微服务监控？根本原因在于生成式大模型在计算过程和状态表达上有独特的工程特征：

1. **多阶段瀑布时延结构**：传统 RPC 调用耗时通常是单一计算或 IO 块，耗时多在几十毫秒以内。而 AI 请求是一个长达数秒的多阶段串并行瀑布：向量召回与重排、多轮工具交互往返、模型 Prefill（提示词预填充）首包等待，以及由显存带宽受限的长尾逐字流式解码。
2. **弱状态码绑定与内容非确定性**：HTTP `200` 仅代表传输层正常结束，模型输出的语义失败（如未能解析出有效函数名、内容安全合规拦截）需要从返回的文本结构或特定终止原因（Finish Reason）中提取。
3. **离散 Token 消耗与计费归因**：每次请求的输入与输出长度动态变化。若追踪链路未能准确记录 Token 开销，就无法进行租户限额（Rate Limit）与财务核算（详见 [Token 成本控制](/architecture/token-cost.md)）。
4. **黑盒推理服务的队列与状态不可控**：无论对接公网 API 还是自建推理引擎（如 vLLM），推理端都有独立的并发 Batch 调度、KV 缓存复用逻辑与排队队列。客户端必须细分外部网络传输与模型实际计算的时延差异。

![AI 系统瀑布流延迟与传统 RPC 调用的结构差异对比](/images/content-rebuild/00423fb95e91547f/fig-1.png)

如上图所示，传统微服务调用链呈现规整、均一的短耗时块；而 AI 系统不仅链路长，且绝大部分时间消耗在推理引擎的 Prefill 首包等待与长尾解码流中。

---

## 2. 基于 W3C TraceContext 的跨边界链路传播

为了在客户端、[模型网关](/architecture/model-gateway.md)、Python 编排工作流、向量数据库与模型推理服务之间串联起完整的调用拓扑，系统应当遵循 W3C TraceContext 规范，使用 `traceparent` 请求头传递上下文：

```text
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
             │  └────────────── trace-id ──────────────┘ └─ parent-id ─┘ └flags
             version (00)
```

在 AI 应用落地中，链路上下文必须跨越三类特殊边界：

### 跨网关与编排服务传递
客户端请求首先进入网关，网关生成全局唯一的 `trace-id` 并开启 Root Span。网关将请求路由至应用编排服务（如基于 LangGraph 构建的 Agent 运行时）时，将 `traceparent` 注入 HTTP Header。编排服务从中提取（Extract）该上下文并作为其父级 Span，确保后续所有检索与模型交互挂载在同一链路下。

### 跨异步任务队列传递
在复杂 Agent 离线规划、长文档向量切片或批量评测场景下，任务通常会推入队列（如 Redis、RabbitMQ 或 Kafka，机制详见 [异步任务](/architecture/async-tasks.md)）。由于跨越了线程与进程生命周期，生产者必须将当前 `traceparent` 序列化并写入消息的属性字典（Metadata/Headers）中；消费者提取该元数据并将其绑定为子 Span 的父上下文，避免链路在异步中继处断裂。

### 跨长连接流式响应传递
在通过 Server-Sent Events（SSE）向前端流式吐字时（机制详见 [流式响应](/architecture/streaming.md)），连接会持续数秒至数十秒。Span 的生命周期必须从请求发起一直延续到接收到流式终止符（如 `data: [DONE]`）。

由于 HTTP 响应头在首个 Chunk 发送时已经固化，追踪上下文除了在握手响应头中返回外，还可以在首个 SSE 事件（如 `:handshake` 注释包或首包 JSON）中携带 `trace_id`。如果用户中途关闭页面导致连接中断，Span 必须正确捕获客户端断开（`ClientDisconnected`）事件，记录已生成的 Partial Token 数量并正常闭合 Span。

![跨网关、异步队列与流式响应的 W3C TraceContext 传播机制](/images/content-rebuild/00423fb95e91547f/fig-2.png)

<ReadMoreLock />

---

## 3. Span 粒度划分与 OpenTelemetry GenAI 语义元数据

链路划分过粗，排障时只能看到黑盒；划分过细，则会产生大量的网络与存储遥测开销。根据 OpenTelemetry GenAI 语义约定，推荐将链路划分为清晰的树状结构：

1. **Root Span（`api_gateway.chat_request`）**：网关入口，记录客户端 IP、租户 ID、端到端耗时与鉴权状态。
2. **编排层 Span（`agent.execution` / `rag.pipeline`）**：记录工作流循环轮数、状态跃迁与最终回答状态。
3. **组件层 Span（`retrieval.vector_search` / `tool.execution`）**：记录外部系统的具体动作。例如检索阶段记录 Top-K 分数与召回分块数；工具调用记录工具名称（`tool_name`）、输入参数及退出码。
4. **模型层 Span（`gen_ai.chat_completion`）**：直连模型推理服务或其反向代理，记录模型核心元数据。

![OpenTelemetry 规范下的 AI 系统 Span 树状层级与元数据挂载](/images/content-rebuild/00423fb95e91547f/fig-3.png)

在模型层 Span 上，应按照规范挂载标准化的元数据属性（Attributes），以便在监控大盘中聚合与过滤：

| 属性键名 | 类型 | 示例值 | 说明 |
| :--- | :--- | :--- | :--- |
| `gen_ai.system` | string | `"vllm"` / `"openai"` | 推理后端系统标识 |
| `gen_ai.request.model` | string | `"claude-3-5-sonnet"` | 请求指定使用的模型名称 |
| `gen_ai.response.model` | string | `"claude-3-5-sonnet-20241022"` | 实际返回结果的模型具体版本 |
| `gen_ai.request.temperature` | double | `0.7` | 采样温度参数 |
| `gen_ai.usage.input_tokens` | int | `1250` | 提示词消耗的 Token 数 |
| `gen_ai.usage.output_tokens` | int | `100` | 生成消耗的 Token 数 |
| `gen_ai.response.finish_reasons` | string[] | `["stop"]` | 终止原因（stop, length, tool_calls） |

以下为一段轻量级 Python 示例，演示如何在流式生成过程中使用 OpenTelemetry SDK 捕获事件并计算指标：

```python
import time
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("ai.system.tracer")

def execute_llm_stream(prompt: str, model_name: str, stream_iterator):
    # 启动符合语义规范的模型调用 Span
    with tracer.start_as_current_span("gen_ai.chat_completion") as span:
        span.set_attribute("gen_ai.system", "vllm")
        span.set_attribute("gen_ai.request.model", model_name)
        span.set_attribute("gen_ai.request.temperature", 0.7)
        
        t_start = time.perf_counter()
        t_first_token = None
        output_chunks = []
        
        try:
            for chunk in stream_iterator:
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                    # 记录首包时间（TTFT）为 Span 事件
                    ttft_ms = (t_first_token - t_start) * 1000
                    span.add_event("first_token_received", {"ttft_ms": ttft_ms})
                
                output_chunks.append(chunk.text)
            
            t_end = time.perf_counter()
            
            # 计算 Token 统计量（生产中优先提取自推理引擎返回的 usage 数据包）
            total_output_tokens = len(output_chunks)
            span.set_attribute("gen_ai.usage.output_tokens", total_output_tokens)
            span.set_attribute("gen_ai.response.finish_reasons", ["stop"])
            
            if total_output_tokens > 1 and t_first_token is not None:
                # 计算平均逐 Token 延迟（ITL）
                itl_ms = ((t_end - t_first_token) / (total_output_tokens - 1)) * 1000
                span.set_attribute("gen_ai.latency.itl_ms", itl_ms)
                
            span.set_status(Status(StatusCode.OK))
            return "".join(output_chunks)
            
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
```

---

## 4. 敏感数据审计与动态脱敏

AI 系统的日志可能包含两类敏感信息：一是用户输入的个人身份信息（PII，如身份证、手机号、银行卡号）；二是上下文注入的企业专有文档或接口密钥。若未经处理直接写入公共 APM 存储，会带来严重的安全与合规风险。

为了兼顾线上排障与安全合规，架构设计应采用**双轨日志存储与动态脱敏机制**（与 [安全与权限](/architecture/security-permission.md) 体系保持一致）：

![排障存储与合规审计存储的双轨日志流向](/images/content-rebuild/00423fb95e91547f/fig-4.png)

### 双轨分流机制
1. **排障热存储链路（Diagnostic Pipeline）**：
   - **定位**：供开发与运维人员排查即时故障、观察耗时瀑布。
   - **脱敏策略**：在 Collector 导出之前，由脱敏模块对 API Key、手机号等正则匹配并替换为 `[REDACTED]`。长提示词进行模板化剥离，仅记录 Prompt 模板 ID 与变量长度，不直接落盘完整私有文档内容。
   - **保留期**：7 至 14 天，保存在 Elasticsearch、ClickHouse 或 Loki 等支持快速检索的引擎中。
2. **合规审计冷存储链路（Audit Pipeline）**：
   - **定位**：应对合规审查、争议回溯与离线微调评测。
   - **安全策略**：保留全量原始输入输出，但必须直接写入防篡改（WORM）对象存储；落盘前通过密钥管理服务（KMS）进行加密；数据解密查询需走专门的审批与只读日志审计流程。
   - **保留期**：180 天至数年，定期转入深冷归档。

若脱敏中间件在热链路中发生异常或超时，**降级策略必须是默认全量遮蔽或阻断输出**，严禁未脱敏文本流向排障热存储。

---

## 5. 核心延迟指标计算与排障下钻

在流式问答场景下，“总耗时”（Total Duration）无法反映系统瓶颈所在。必须将其解构为针对大模型推理特征的两个核心指标：

![流式生成场景下的 TTFT 与 ITL 延迟计算模型](/images/content-rebuild/00423fb95e91547f/fig-5.png)

### 延迟指标定义与计算

1. **首包延迟（TTFT, Time to First Token）**：
   从客户端发送请求到收到模型返回的第一个有效 Token 字符的耗时。该指标反映了网络链路传输、网关路由、知识库检索以及模型端的 **提示词预填充（Prefill）阶段** 耗时：
   $$TTFT = t_{\text{first\_token}} - t_{\text{request\_start}}$$
   如果 TTFT 很高而后续吐字极快，说明瓶颈在向量检索、外部工具调用超时，或是因为 Prompt 极长导致 Prefill 计算排队。

2. **Token 间延迟（ITL, Inter-Token Latency）**：
   在流式传输过程中，连续生成的两个 Token 之间的到达时间差。工程统计上通常取其平均值：
   $$ITL = \frac{t_{\text{last\_token}} - t_{\text{first\_token}}}{N_{\text{output\_tokens}} - 1}$$
   模型的自回归解码属于显存带宽密集型计算。若 ITL 偏高，通常反映出推理实例显卡并发负载过高、KV 缓存带宽吃紧或动态批处理队列拥堵。

### 贯穿算例：流式问答耗时分解

回到开篇的技术支持 Agent 示例。系统记录了某次请求的关键时间戳节点（以下数值仅为教学示意）：
- $T_0 = 0\text{ ms}$：网关收到用户请求，生成 `trace_id`。
- $T_1 = 30\text{ ms}$：网关路由完成，编排服务发起知识库混合检索。
- $T_2 = 230\text{ ms}$：检索结束，拼接 Prompt 模板并发送至模型推理服务。
- $T_3 = 680\text{ ms}$：编排服务收到推理引擎返回的第一个流式 Token 数据块。
- $T_4 = 2680\text{ ms}$：流式传输结束，收到最后一个 Token 及终止符，共统计生成 $101$ 个 Token。

指标推导计算：
1. 知识库检索耗时：$T_2 - T_1 = 200\text{ ms}$；
2. 模型端 Prefill 及排队耗时：$T_3 - T_2 = 450\text{ ms}$；
3. 端到端首包延迟：$TTFT = T_3 - T_0 = 680\text{ ms}$；
4. 模型自回归解码总耗时：$T_4 - T_3 = 2000\text{ ms}$；
5. 平均逐 Token 延迟：
   $$ITL = \frac{2000\text{ ms}}{101 - 1} = \frac{2000\text{ ms}}{100} = 20\text{ ms/token}$$
   对应的流式生成吞吐为：$\frac{1}{0.020\text{ s}} = 50\text{ tokens/s}$。

**算例结论**：如果用户反馈该次交互耗时过长（总耗时 $2680\text{ ms}$），排障人员通过 Trace 瀑布图可以明确获知：系统在前 $680\text{ ms}$ 就已经让用户看到了第一个字（体感响应良好），总体耗时长的主要原因在于模型输出了多达 101 个 Token，解码持续了 2 秒。此时优化重点应当是通过提示词收敛输出长度，而不是花费精力去重写向量检索逻辑。

![生产级调用链异常定位与下钻排障流程](/images/content-rebuild/00423fb95e91547f/fig-6.png)

### 异常排查下钻路径
当大盘触发错误告警时，排障人员应按上图流程在追踪系统中进行下钻：
1. **过滤 `status_code == ERROR`**：首先筛选出带有异常标记的 Trace 集合。
2. **定位异常 Span 层级**：
   - 若错误落在 `tool.execution`：检查 `exception.message`，确认是外部 API 超时还是大模型生成的 JSON 参数不满足函数 Schema（如缺失必要字段）。
   - 若错误落在 `gen_ai.chat_completion`：检查 HTTP 返回码。若为 `429` 说明触发了提供商速率限制，应排查重试退避与多网关分流配置（详见 [高可用设计](/architecture/high-availability.md)）；若为 `400 Context Window Exceeded`，说明前置 RAG 检索注入了过长文档切片（参见 [上下文管理](/architecture/context-management.md)）。
   - 若全链路状态均为 `OK` 但用户反馈回答内容违规或存在逻辑硬伤：提取该 `trace_id` 对应的 Prompt 输入与模型输出样本，送入离线测试集进行回归复现（详见 [AI 评测体系](/architecture/ai-evaluation.md)）。

---

## 6. 生产落地可观测性验收清单

在落地 AI 系统的链路追踪与日志体系时，各环节的输入规格、状态流转、降级策略与可观测验收标准如下：

| 模块 / 环节 | 输入规格 | 状态流转 | 失败路径与降级 | 可观测验收标准 |
| :--- | :--- | :--- | :--- | :--- |
| **上下文传播** | HTTP/gRPC Header、MQ 消息元数据 | `traceparent` 网关注入 $\to$ 编排提取 $\to$ 队列元数据中继 $\to$ 绑定流式子 Span | 上游丢失上下文时生成新 `trace_id` 并打上 `is_orphaned=true` 标签告警 | 链路不发生断链，跨服务与异步任务均能通过单条 `trace_id` 完整串联 |
| **指标度量** | 逐 Token 时间戳、Token 计数器 | 首包到达记录 TTFT 事件 $\to$ 累加 Output Tokens $\to$ 结束计算 ITL | 客户端提前断开时捕获 `ClientDisconnected`，记录 Partial Tokens 并正常闭合 Span | 每一个成功的模型 Span 均包含正确的 TTFT、平均 ITL 与 Token 数量属性 |
| **动态脱敏** | 原始输入 Prompt 与模型生成文本 | 识别敏感正则 $\to$ 掩码遮蔽 $\to$ 写入排障热存；原始报文流入 KMS 加密冷存 | 脱敏引擎超时或异常时**默认全量掩码**，严防明文外泄 | 热存储抽检无明文 API Key 与用户 PII，冷存储支持基于合规审批的解密复盘 |
| **排障下钻** | 告警事件、Trace 过滤查询 | 监控报警 $\to$ 按 `trace_id` 下钻 $\to$ 定位故障组件（检索/工具/模型） | 遥测 Collector 突发拥塞时启用本地尾部采样，优先保留异常与高延迟 Trace | 能通过 Trace 瀑布图直接区分出网络故障、工具格式错误与模型限流等不同根因 |

关于系统可观测指标与端到端评测、追溯机制的更多内容，可继续参阅 [AI 应用如何做日志、监控、成本统计和效果评估？](/interview/ai-system-design-interview/logging-monitoring.md) 以及 [如何让 AI 回答有来源、有依据、可追溯？](/interview/high-freq-scenarios/answer-traceability.md)。
