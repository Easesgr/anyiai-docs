---
title: 大模型部署需要关注哪些核心指标？TTFT、TPOT、吞吐量与 SLO 体系
icon: mdi:chart-line
tag:
- 模型部署面试
- 延迟
- 吞吐
description: 系统梳理大模型推理核心指标：TTFT、TPOT、ITL、Token 吞吐量与 Goodput，结合全链路算例与两类业务 SLO 给出面试口述与工程度量方案。
---

![大模型推理核心指标体系全貌](/images/content-rebuild/d077a72d770ad877/cover.png)

在传统微服务中，API 的性能通常使用 QPS（每秒请求数）和平均响应时间（RT）即可清晰刻画。但在大语言模型（LLM）推理场景中，由于自回归生成的逐字输出特性以及输入、输出长度的高度动态性，单纯沿用传统指标会导致性能评估严重失真。

评估大语言模型在线服务，需要建立一套覆盖首字响应、生成流速、系统容量与履约质量的完整指标体系。本文聚焦服务层与客户端视角，拆解核心指标的采集口径、测量物理锚点以及生产环境中的 SLO（服务等级目标）设计。

---

### 口述答案：大模型在线服务核心指标体系四要素

在面试中被问及“如何量化评估大模型部署服务的性能”时，可以按以下框架结构化作答：

> **核心口述回答：**
>
> 评估大模型推理服务不能使用单一延迟指标，需要拆解为**四个核心要素**：
>
> 1. **TTFT（Time To First Token，首字延迟）**：衡量从客户端发出请求到收到第一个 Token 的耗时，反映系统调度排队时间与 Prefill（预填充）阶段的计算性能。
> 2. **TPOT（Time Per Output Token）与 ITL（Inter-Token Latency）**：衡量后续 Token 的生成流速与平稳度。TPOT 是生成阶段每个 Token 的平均耗时；ITL 是流式返回时相邻两个 Token 的瞬时间隔。
> 3. **吞吐量双口径（RPS 与 Tokens/s）及 Goodput**：系统容量不能只看 RPS，因为长短文本消耗算力悬殊；必须以每秒处理和生成的 Token 数（Tokens/s）作为吞吐基准。在生产环境中，还要进一步统计 **Goodput（有效吞吐量）**，即在满足预设 SLO 约束的前提下所交付的有效 Token 吞吐。
> 4. **分位数与尾延迟（P90/P99）**：大模型服务对长尾极为敏感，受变长请求、KV Cache 分页换入与连续批处理调度的影响，均值容易掩盖卡顿。因此需要针对交互型（关注低 TTFT 与稳定 ITL）与批处理型（关注高吞吐量）场景，分别设定 P90、P99 的分位数治理目标。

---

### 阶段划分：Prefill 首字延迟与 Decode 生成速度的计算特征

自回归大模型推理在执行上分为两个阶段：**Prefill（预填充）阶段**与 **Decode（解码生成）阶段**。指标的分离直接对应了不同的计算特征。

![Prefill与Decode阶段指标对应机制](/images/content-rebuild/d077a72d770ad877/fig-1.png)

#### 1. Prefill 阶段与 TTFT

Prefill 阶段负责将用户输入的整段 Prompt 进行并行计算，生成所有输入 Token 的键值向量并写入缓存。在典型服务批量与中长提示词场景下，整段 Token 并行计算带来较高的算术强度，通常表现为**计算受限（Compute-Bound）**，但具体瓶颈仍受批大小与硬件算力带宽比制约。

**TTFT 的构成**包含两部分：

$$\text{TTFT} = t_{\text{queue}} + t_{\text{prefill}}$$

- $t_{\text{queue}}$：请求在推理引擎（如 vLLM 调度器）等待被装配进批处理批次的时间。
- $t_{\text{prefill}}$：模型完整执行一次前向计算以处理全部 Prompt Token 的硬件耗时。

线上 TTFT 出现异常尖刺，通常意味着请求队列积压严重，或者当前批次被包含超长上下文的 Prefill 计算阻塞。

#### 2. Decode 阶段与 TPOT / ITL

在输出第一个 Token 之后，模型转入自回归 Decode 阶段。每次前向计算仅接收上一步生成的单个 Token，加载模型权重及历史 [KV Cache](/interview/model-deployment-interview/kv-cache-deploy.md) 进行 Attention 计算。在常见服务并发批次下，单步算术强度较低，通常表现为**显存带宽受限（Memory-Bound）**（在超大批次下亦可能向计算受限过渡）。

- **TPOT（Time Per Output Token）**：Decode 阶段每个输出 Token 的平均耗时。若请求一共生成 $N$ 个输出 Token（$N > 1$），端到端总耗时为 $T_{\text{total}}$，则：

$$\text{TPOT} = \frac{T_{\text{total}} - \text{TTFT}}{N - 1}$$

- **ITL（Inter-Token Latency，Token 间延迟）**：流式传输（Streaming）中，第 $i$ 个 Token 与第 $i-1$ 个 Token 之间的瞬时间隔：

$$\text{ITL}_i = t_{\text{token}_i} - t_{\text{token}_{i-1}} \quad (i \ge 2)$$

TPOT 反映宏观平均生成速率，而 ITL 能够捕捉单步推理因调度打断、内存重分配或并发抢占导致的瞬间停顿。

<ReadMoreLock />

---

### 吞吐口径辨析：RPS、Tokens/s 与 Goodput

传统 Web 开发习惯使用 QPS/RPS（Requests Per Second）作为容量衡量单位。但在大模型服务中，直接使用 RPS 会掩盖真实的负载状况。

![吞吐口径与Goodput过滤机制对比](/images/content-rebuild/d077a72d770ad877/fig-2.png)

#### 1. 为什么 RPS 在大模型服务中容易失真？

考虑两个完全相同的 RPS=10 的服务：
- 场景 A：每个请求输入 50 Tokens，输出 50 Tokens。
- 场景 B：每个请求输入 4000 Tokens，输出 1000 Tokens。

场景 B 所需的显存带宽与计算浮点数是场景 A 的数十倍。因此，**脱离序列长度的 RPS 缺乏横向可比性**。

#### 2. Token 生成吞吐（Tokens/s）的口径界定

推理引擎通常上报两种 Token 吞吐：
- **Prefill/Prompt Throughput（输入 Tokens/s）**：系统每秒处理的 Prompt Token 总数。
- **Decode/Generation Throughput（生成 Tokens/s）**：系统每秒生成的 Output Token 总数。

在评估服务集群的实际承载能力时，**Decode 生成吞吐是核心评估指标**。因为 Decode 阶段每生成一个 Token 都需要加载一次模型权重与对应请求的 KV Cache，它直接决定了系统支撑并发用户会话的上限。

#### 3. 为什么必须关注 Goodput？

将引擎的批大小（Batch Size）持续放大，硬件计算单元的利用率会提高，总输出 Tokens/s 也会持续攀升。但批处理扩大会拉长每个请求的排队时间与单步步进耗时，造成严重的延迟违约。

**Goodput（有效吞吐量）**是指**在满足业务预设 SLO 约束（例如 TTFT < 1.0s 且 TPOT < 50ms）的前提下，单位时间内系统产出的有效 Token 数量**：

$$\text{Goodput} = \frac{\sum_{k \in \mathcal{S}_{\text{SLO}}} N_{\text{tokens}, k}}{\Delta T}$$

其中 $\mathcal{S}_{\text{SLO}}$ 代表完全满足延迟与正确性约束的请求集合。超过 SLO 阈值交付的 Token，在商业服务中通常无法带来良好的用户体验，属于低效甚至无效损耗。

---

### 端到端观测点：客户端感知延迟 vs 推理引擎耗时

监控大模型服务时，不能只看推理引擎输出的内部指标。请求从发出到最终呈现给用户，涉及多个中间链路。

![全链路监控打点与耗时构成分解](/images/content-rebuild/d077a72d770ad877/fig-3.png)

#### 1. 关键监控锚点分布

一条采用 SSE（Server-Sent Events）流式协议的 LLM 请求，在端到端链路上有三个典型打点层级：

- **Client 层（客户端/前端）**：
  - 发送请求时间戳 $T_{\text{c\_send}}$。
  - 接收到第一个 SSE 数据帧的时间戳 $T_{\text{c\_first}}$。
  - 接收到最后一个结束帧的时间戳 $T_{\text{c\_end}}$。
- **Gateway 层（网关/反向代理）**：
  - 负责协议解析、鉴权、全局[限流降级](/interview/model-deployment-interview/rate-limiting.md)以及反向代理转发。
- **Engine 层（推理引擎，如 vLLM）**：
  - 请求进入等待队列的时间戳 $T_{\text{e\_enqueue}}$。
  - 调度器分配资源并启动 Prefill 计算的时间戳 $T_{\text{e\_prefill\_start}}$。
  - 第一个 Token 计算完成的时间戳 $T_{\text{e\_first\_token}}$。
  - 生成终止条件触发的时间戳 $T_{\text{e\_finish}}$。

#### 2. 测量差异与隐形损耗

两端观测到的数据差异揭示了底层基础设施的运行瓶颈：

$$\text{TTFT}_{\text{client}} - \text{TTFT}_{\text{engine}} = \text{RTT}_{\text{network}} + t_{\text{gateway}} + t_{\text{sse\_buffer}}$$

- **网关缓冲问题**：若 API 网关开启了 HTTP 响应缓冲（Response Buffering）且未配置流式直通，首个 Token 的 SSE Chunk 会被网关拦截积攒，导致客户端感知的 TTFT 严重劣化，甚至退化为非流式响应。
- **序列化与调度损耗**：高并发下 Python 引擎层的 HTTP Server（如 FastAPI / Uvicorn）如果存在事件循环阻塞，请求在进入 Engine 核心调度器之前就会产生排队延迟。

关于线上日志与监控指标的埋点收集实践，可在 [日志监控、调用统计与成本控制](/interview/model-deployment-interview/monitoring-cost.md) 中展开对照。

---

### 逐步算例：单请求链路拆解与指标计算

为了直观理解各指标的数值关联，以下设定一组实验参数进行逐步推导。

![单请求流式链路耗时推进算例](/images/content-rebuild/d077a72d770ad877/fig-4.png)

#### 1. 实验场景设定（示意数据）

设某次流式补全请求的参数与打点数据如下：
- 输入 Prompt 长度：$L_{\text{in}} = 512$ Tokens
- 实际生成长度：$L_{\text{out}} = 128$ Tokens
- 客户端与服务端单向网络传输耗时（单向时延）：$t_{\text{net}} = 30$ ms
- 网关处理耗时：$t_{\text{gw}} = 10$ ms
- 引擎内部调度排队：$t_{\text{queue}} = 80$ ms
- Prefill 执行耗时：$t_{\text{prefill}} = 120$ ms
- Decode 阶段执行 127 步前向，每步耗时稳定在 $25$ ms
- 业务约定的 SLO 阈值：$\text{TTFT} \le 300$ ms，$\text{TPOT} \le 35$ ms

#### 2. 指标逐步推导过程

**步骤一：计算推理引擎侧 TTFT**

$$\text{TTFT}_{\text{engine}} = t_{\text{queue}} + t_{\text{prefill}} = 80 + 120 = 200\text{ ms}$$

**步骤二：计算客户端感知 TTFT**

请求需先经历上行网络传输与网关处理，引擎算完第 1 个 Token 后，经下行网络传回客户端：

$$\text{TTFT}_{\text{client}} = t_{\text{net(上行)}} + t_{\text{gw}} + \text{TTFT}_{\text{engine}} + t_{\text{net(下行)}} = 30 + 10 + 200 + 30 = 270\text{ ms}$$

**步骤三：计算 Decode 阶段总耗时与 TPOT**

生成后续 $128 - 1 = 127$ 个 Token，引擎单步耗时 25 ms：

$$t_{\text{decode\_total}} = 127 \times 25\text{ ms} = 3175\text{ ms}$$

$$\text{TPOT} = \frac{t_{\text{decode\_total}}}{L_{\text{out}} - 1} = \frac{3175}{127} = 25\text{ ms/token}$$

换算为瞬时生成速率即为：$1000 / 25 = 40$ Tokens/s。

**步骤四：端到端总体耗时（E2E Latency）**

$$\text{Latency}_{\text{E2E}} = \text{TTFT}_{\text{client}} + t_{\text{decode\_total}} = 270 + 3175 = 3445\text{ ms}$$

**步骤五：SLO 履约评估与 Goodput 归属**

- 客户端 $\text{TTFT} = 270\text{ ms} \le 300\text{ ms}$（履约达标）。
- 实际 $\text{TPOT} = 25\text{ ms} \le 35\text{ ms}$（履约达标）。

该请求满足全部预设 SLO 目标，其输出的 128 个 Token 计入系统的有效交付产出（Goodput）。若系统在 1 秒内稳定完成 10 个此类请求，则有效生成吞吐为 $1280$ Tokens/s。

---

### SLO 体系设计：交互型与批处理型的分位数约束

不同业务形态对大模型服务的核心诉求差异显著，脱离具体业务场景套用统一的 SLO 规范容易导致资源配置失衡。

![交互型与批处理型业务SLO权衡矩阵](/images/content-rebuild/d077a72d770ad877/fig-5.png)

#### 1. 典型业务形态指标对照

| 业务分类 | 代表场景 | 核心优先级指标 | 典型容忍阈值（P90/P99） | 架构权衡取向 |
| :--- | :--- | :--- | :--- | :--- |
| **实时交互型** | 智能客服、在线 Copilot、对话 Agent | **TTFT**（首字响应）<br>**ITL**（打字流畅度） | TTFT P90 < 800ms<br>ITL P95 < 40ms（约 25 tok/s） | 限制批大小上限，优先降低排队等待时间 |
| **异步处理型** | 长文档摘要、离线数据生成、代码批量分析 | **Tokens/s**（总吞吐）<br>**成功率** | 整体耗时 P90 < 120s<br>失败重试率 < 0.1% | 允许请求在队列中充分积压以组装大批次，允许 TTFT 达到秒级 |
| **极短敏感型** | 语义网关分类、意图识别路由 | **E2E Latency**（端到端） | E2E P99 < 150ms | 限制生成 Token 上限（通常 ≤ 10 tokens），专注 Prefill 与单步决策 |

#### 2. 人类阅读速度与 ITL 设定依据

交互型业务的 ITL 目标并非越快越好。人类阅读正常文本的速度通常在每秒 15 到 25 个 Token（对应 ITL 约为 40ms ~ 65ms）。

- 如果系统 $\text{ITL} \le 40\text{ ms}$，用户体验表现为平滑的流式输出；
- 如果系统平均 ITL 虽快，但 **ITL P99 出现大于 500ms 的明显波动**，用户会清晰感受到界面停顿。

因此，在交互业务中，治理 **ITL 的波动与 P99 尾延迟**比单纯压缩平均 TPOT 更加重要。

---

### 面试高频追问与常见认知误区

![吞吐提升伴随延迟劣化的排队机制曲线](/images/content-rebuild/d077a72d770ad877/fig-6.png)

#### 追问 1：为什么高并发场景下吞吐量（Tokens/s）上升时，客户端感知的 TTFT P99 会急剧恶化？

> **面试口述要点：**
>
> 这是因为现代大模型推理普遍采用 [连续批处理（Continuous Batching）](/interview/model-deployment-interview/continuous-batching.md)。
>
> 当并发请求涌入时，调度器会尽可能将新请求装入当前推理批次。这引发了两个主要资源竞争：
>
> 1. **算力时间片抢占（Prefill-Decode 干扰）**：新请求的 Prefill 计算量较大（计算受限）。如果系统未采用 Chunked Prefill（分块预填充）技术，新请求的长 Prefill 会占用该轮前向计算时间，导致批次内既有请求的单步生成被推迟数百毫秒，同时后续新请求在等待队列中积压，放大了 $t_{\text{queue}}$。
> 2. **显存池容量竞争**：并发增大会快速消耗 KV Cache 显存。当显存分配接近上限水位线时，调度器必须暂缓调度新的 Prefill 请求进显存，或者触发 KV Cache 换入换出（Swap）。新请求在队列中被迫等待可用显存块，导致 TTFT 的 P99 尾延迟出现非线性上升。

#### 常见认知误区辨析

- **误区一：“系统平均 TPOT 很低，说明生成体验一定平稳。”**
  *反驳*：平均值会掩盖严重的偶发停顿。自回归生成中哪怕单次 Token 因为垃圾回收、CUDA Kernel 初始化调度或长 Prefill 插入产生 800ms 的 ITL 尖刺，只要其他 100 个 Token 耗时 20ms，平均 TPOT 依然表现良好（约 27ms）。但用户在界面上会经历一次明显的卡顿，因此必须考察 ITL 的高分位数（P95/P99）。

- **误区二：“TPOT 持续上升是因为用户输入的提示词（Prompt）太长。”**
  *反驳*：概念混淆。输入的 Prompt 长度主要决定 Prefill 阶段的计算量，直接承受百毫秒至秒级大开销的是 **TTFT**。进入 Decode 阶段后，单步耗时（TPOT）在短文本下主要取决于并发批大小（Batch Size）与模型权重的显存搬运；尽管长上下文累积的 KV Cache 搬运开销也会使单步延迟有所增加，但 Prompt 长度暴增的核心影响阶段在于首字预填充，不应将 TTFT 的暴增与单步流速 TPOT 混为一谈。

- **误区三：“吞吐量（Tokens/s）越高，服务的工程架构就越优秀。”**
  *反驳*：脱离了 SLO 约束的吞吐量缺乏工程意义。如果通过把 Batch Size 强行打满让系统生成吞吐量翻倍，但同时将 TTFT 从 500ms 推高至 10 秒、TPOT 推高至 150ms，导致大部分请求违反了交互型业务的超时标准，那么其 Goodput 实质上跌落到了极低水平，伴随着算力浪费与用户体验恶化。

关于硬件显存与带宽如何定量制约上述指标的具体推导，可继续深入阅读 [首 Token 延迟与平均生成速度的硬件影响因果](/interview/model-deployment-interview/latency-throughput.md) 与 [vLLM 与 PagedAttention 原理](/interview/model-deployment-interview/vllm-paged-attention.md)。
