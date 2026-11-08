---
title: 部署大模型时如何做日志监控、调用统计和成本控制？
icon: mdi:chart-box-outline
tag:
- 模型部署面试
- 监控
- 成本控制
description: 解析大模型生产监控的三层指标体系、GPU利用率假饱和陷阱、分布式流式追踪以及Prompt与Completion非对称Token成本核算模型。
---

在将大语言模型（LLM）推向生产环境时，工程团队常面临两个关键挑战：服务变慢时难以快速定位是显卡算力饱和、KV Cache 耗尽还是网络流式回包阻塞；在月末结算账单时，无法准确拆解每个业务方消耗的 Token 边际成本。要解决这些问题，必须建立贯穿硬件层、推理引擎层与业务调用层的全链路可观测体系。

为了让这些指标和核算方法更具体，我们以一个贯穿全程的典型场景为例：某企业知识库与智能客服系统在单机 8 卡节点上部署了一个开源 70B 模型。在晚高峰时，业务端反馈“首字响应变慢”，财务端发现“输出 Token 账单远高于预期”，而监控大盘上的 `GPU-Util` 却长期显示为 99%。我们将围绕这个案例，逐层排查指标陷阱、梳理调用链路，并建立精细化的成本核算模型。

---

## 1. 口述答案：生产环境三层监控与成本核算方法

面试中遇到“大模型上线后如何做监控与成本控制”时，可直接口述以下结构化答案：

> **口述回答**：
> 在生产环境中，我们构建了“硬件层、引擎层、业务层”三层监控体系，配合基于工作负载特征的 Token 边际成本模型实现精细化治理：
> 1. **硬件层**：绕过 `nvidia-smi` 表面 GPU-Util 的监控盲区，通过 DCGM 采集 `SM_ACTIVE` 和 `PIPE_TENSOR_ACTIVE` 指标，真实衡量 Tensor Core 的张量计算占空比，结合显存带宽和 PCIe/NVLink 吞吐判定物理瓶颈。
> 2. **引擎层**：基于 Prometheus 采集推理引擎的核心调度状态。重点监控调度队列等待数（`num_requests_waiting`）、GPU KV Cache 使用率（`gpu_cache_usage_factor`）以及发生抢占时的显存换出（`num_requests_swapped`），同时分离监控 Prefill 与 Decode 两个阶段的吞吐与延迟。
> 3. **业务层**：基于 OpenTelemetry 注入 TraceID，串联网关、调度器、推理批处理与流式 SSE 回包。在日志中记录脱敏后的会话元数据，精确分离统计 Prompt Token 与 Completion Token，建立结合 TTFT（首字延迟）和 TPOT（生成速率）的 SLO 告警。
> 4. **成本核算**：将实例折旧与机房网络总拥有成本（TCO）折算为机器时间单价，结合 Prefill 算力受限（Compute-bound）与 Decode 带宽受限（Memory-bound）的物理特性，建立非对称的 Token 边际成本公式；通过闲时动态缩容、Spot 实例分流容灾与 Multi-LoRA 共享底座实现综合降本。

![三层可观测体系与成本核算框架图](/images/content-rebuild/99a2f5d8a6342759/cover.png)

上图展示了从底层裸金属 GPU 到顶层业务网关的完整可观测分层。阅读时请注意：每一层指标不是孤立的数据孤岛，当业务层告警 TTFT 突增时，排查链路需反向穿透到引擎层的调度等待队列与硬件层的张量核心占空比。

---

## 2. 硬件指标监控陷阱：GPU-Util 100% 并不等价于算力拉满

在传统深度学习模型训练或 CPU 服务监控中，CPU/GPU 利用率（Utilization）通常能直接反映机器负荷。然而在大模型自回归推理场景下，直接依赖 `DCGM_FI_DEV_GPU_UTIL`（即 `nvidia-smi` 输出的 GPU-Util）会导致严重的容量误判。

### 2.1 统计采样机制导致的假饱和

根据 NVIDIA 官方驱动设计，`GPU-Util` 的统计定义为：**在采样时间窗口内，GPU 上至少有一个活动 Kernel 在执行的时间比例**。

这意味着，如果一个 Kernel 仅分配了极少量的线程，只占用了 GPU 整体流式多处理器（SM）中不到 2% 的计算单元，只要该 Kernel 在采样窗口内持续处于运行状态，驱动上报的 `GPU-Util` 就是 100%。

在大模型自回归解码（Decode，逐字生成）阶段，模型每次仅基于前文生成一个 Token。由于批处理大小（Batch Size）受到显存限制或者流量较小，矩阵向量乘法（GEMV）无法提供足够的并行度。此时大量的 GPU 核心处于空闲等待状态，算力并未饱和，但因为显存在低速搬运、Kernel 持续占线，监控面板上的 `GPU-Util` 会持续显示为 99% 或 100%。

![GPU执行时间占空比与Tensor Core实际饱和度对比](/images/content-rebuild/99a2f5d8a6342759/fig-1.png)

上图对比了两个看似同为 100% 利用率的节点内部状态。左侧在自回归单请求阶段，虽然时间轴被占满，但内部 SM 运算单元大半处于等待内存拉取状态；右侧在批量 Prefill 阶段，大量张量指令同时灌满 Tensor Core。

<ReadMoreLock />

### 2.2 DCGM 深层指标与算力饱和度判定

为了准确反映算力真实负荷，生产监控建议通过 NVIDIA Data Center GPU Manager（DCGM）采集以下底层 Profile 指标：

| 指标名 | 采集物理对象 | 生产监控含义 | 预警基线参考 |
| :--- | :--- | :--- | :--- |
| `DCGM_FI_DEV_GPU_UTIL` | 时间切片覆盖率 | 仅代表 GPU 处于非休眠状态的时间比例，不能反映计算量饱和度 | 不作为扩容唯一标准 |
| `DCGM_FI_PROF_SM_ACTIVE` | SM 活跃周期比 | 至少有 1 个 Warp 处于活跃执行状态的流式多处理器比例 | 评估算力调度是否有大范围闲置 |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor Core 管道执行率 | 张量核心实际执行浮点矩阵运算周期的占比 | < 15% 说明严重受限于访存，> 70% 说明计算饱和 |
| `DCGM_FI_PROF_DRAM_ACTIVE` | 显存控制器活跃比 | GPU 访问 HBM/GDDR 显存带宽的占空比 | Decode 密集型服务通常该值极高 |

当排查性能瓶颈时，应当联合分析：
- **Compute-bound（算力受限）**：`PIPE_TENSOR_ACTIVE` 处于高位，通常发生在长文本大 Batch 的 Prefill 阶段。
- **Memory-bound（访存受限）**：`DRAM_ACTIVE` 极高，而 `PIPE_TENSOR_ACTIVE` 处于低位，通常发生在小 Batch 的 Decode 阶段，需要结合 [/interview/model-deployment-interview/continuous-batching.md](/interview/model-deployment-interview/continuous-batching.md) 中介绍的批处理策略来提升单步吞吐。

---

## 3. 推理引擎核心暴露指标：Prometheus 格式深度解析

推理引擎（以 [vLLM 官方文档](https://docs.vllm.ai/en/latest/) 为代表）通过 HTTP 接口暴露标准的 Prometheus Metrics。监控引擎层指标的核心目的，是掌握内部请求排队状态与显存缓存池的健康度。

### 3.1 核心指标与排队流转

生产监控大盘建议包含以下维度的指标群：

```promql
# 1. 队列调度健康度：排队请求数（持续大于 0 意味着需要水平扩容）
vllm:num_requests_waiting{model="qwen-72b"}

# 2. 正在并发执行的请求数与被换出到内存的请求数
vllm:num_requests_running{model="qwen-72b"}
vllm:num_requests_swapped{model="qwen-72b"}

# 3. 显存缓存占用率（反映 PagedAttention 显存池的饱和度）
vllm:gpu_cache_usage_factor{model="qwen-72b"}
```

![推理引擎内部队列调度与指标暴露链路](/images/content-rebuild/99a2f5d8a6342759/fig-2.png)

上图展示了从外部请求进入推理服务，到内部三个状态队列流转所暴露的指标拓扑。阅读提示：特别注意红色的 `Swapped` 队列，当显存完全耗尽时，引擎被迫将正在推理的 KV Block 换出到主机内存（Host Memory），这会导致严重的延迟劣化。

### 3.2 阶段性能拆解：Prefill 与 Decode

大模型推理包含两个阶段：Prefill（处理输入 Prompt）与 Decode（自回归逐字生成）。关于首字延迟与生成速率的基础概念已在 [/interview/model-deployment-interview/latency-throughput.md](/interview/model-deployment-interview/latency-throughput.md) 中阐明。在监控中，严禁用单一的“接口总耗时”来度量服务质量：

1. **首字延迟（TTFT，Time to First Token）**：
   通过直方图 `vllm:time_to_first_token_seconds_bucket` 追踪。TTFT 包含两部分：在 `waiting` 队列中的排队等待时间 + 整个 Prompt 的 Prefill 计算时间。若 TTFT 上涨而硬件计算未饱和，应首先检查排队深度与长前缀解析开销。
2. **单字生成耗时（TPOT / ITL）**：
   通过直方图 `vllm:time_per_output_token_seconds_bucket`（Inter-Token Latency）追踪。度量模型自回归每生成一个 Token 的时间间隔。该指标直接反映了流式阅读的流畅度，主要受当前 Batch 中正在运行的请求数以及显存带宽制约。
3. **前缀缓存命中率（Prefix Caching Ratio）**：
   通过 `vllm:gpu_prefix_cache_hit_rate` 采集。对于多轮对话或固定 System Prompt 的场景，KV Cache 命中的 Token 不需要重新执行矩阵乘法计算，能大幅降低 Prefill 耗时。

---

## 4. 调用日志与分布式追踪：基于 TraceID 的端到端流式观测

大模型服务多采用 Server-Sent Events（SSE）长连接流式吐字，传统的“请求-响应”单点日志无法精确定位“首包等待很久、吐字中途卡顿”等细粒度故障。必须建立基于分布式链路追踪（Distributed Tracing）与审计日志的排查链路。

### 4.1 流式调用生命周期分解

一次流式调用的分布式链路通常跨越网关、调度与引擎，需要在 Trace Context 中记录关键时间戳事件（Span Events）：

![基于TraceID的流式调用时序与Span分解](/images/content-rebuild/99a2f5d8a6342759/fig-3.png)

上图详细标定了一个流式请求从客户端触发到最终连接关闭的 Span 结构。排查故障时，根据时序切片即可瞬间定界问题来源：
- **$T_1 - T_0$**：网络传输与 API 网关鉴权/限流耗时，机制可参考 [/interview/model-deployment-interview/rate-limiting.md](/interview/model-deployment-interview/rate-limiting.md)。
- **$T_2 - T_1$**：引擎调度排队耗时。若此区间大，说明当前实例并发能力不足。
- **$T_3 - T_2$**：首包计算（Prefill）与首个 Chunk 发送耗时。
- **$T_4 - T_3$**：自回归解码（Decode）循环生成耗时。

### 4.2 审计日志结构化设计与敏感信息治理

调用日志必须满足财务核算、性能复盘与数据合规三重要求。日志建议采用如下结构化 JSON 格式：

```json
{
  "timestamp": "2026-04-09T10:15:30.124Z",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "service_name": "llm-gateway",
  "tenant_id": "dept-ai-search",
  "model_name": "qwen-72b-instruct",
  "tokens": {
    "prompt_tokens": 1248,
    "completion_tokens": 256,
    "cached_tokens": 1024,
    "total_tokens": 1504
  },
  "timings": {
    "queue_time_ms": 12.5,
    "ttft_ms": 148.2,
    "total_duration_ms": 2450.8
  },
  "client_ip": "10.244.3.18",
  "status_code": 200,
  "finish_reason": "stop"
}
```

> **合规警示**：生产环境下严禁无脱敏全量打印用户 Prompt 与生成的完整文本。如需留存用于评估或微调，必须在网关层通过规则匹配或本地实体识别模型实施个人身份信息（PII，如身份证、电话号码、API 密钥）的掩码脱敏（Masking），脱敏后转存至受权限保护的冷存储，线上检索系统仅保留元数据。

---

## 5. 成本度量模型：单位 Token 边际成本核算公式

在公有云或自建算力中心部署大模型，不能仅凭粗略的“每张卡多少钱”来评估投入。需要建立总拥有成本（TCO）模型，精确核算到每个业务方调用的单位 Token 边际成本。

### 5.1 TCO 构成与机器时钟成本

节点的单位时间总拥有成本 $C_{\text{node\_hour}}$ 由以下部分组成：

$$C_{\text{node\_hour}} = \frac{C_{\text{hw\_amortization}} + C_{\text{facility\_power}} + C_{\text{network}} + C_{\text{ops}}}{T_{\text{lifecycle\_hours}}} + C_{\text{cloud\_license}}$$

- $C_{\text{hw\_amortization}}$：服务器硬件及 GPU 卡的折旧总额（通常按 3 年或 5 年直线折旧）。
- $C_{\text{facility\_power}}$：机柜电力（PUE 修正系数 $\times$ 实际功耗）与制冷开销。
- $C_{\text{cloud\_license}}$：若是公有云 GPU 实例，则直接为云厂商小时牌价。

### 5.2 Token 成本核算算例：Prefill 与 Decode 的非对称性

公有云大模型 API 的单价由云供应商根据运营成本与市场策略综合公布，并非单凭底层硬件物理机制决定。但在底层运行特征上，Prefill 通常较容易通过批量并行打满算力核心，而自回归 Decode 阶段逐字生成受显存读取带宽限制，二者的机器占用开销存在客观差异，这也为私有部署时的多租户内部结算提供了核算参考。

我们通过一个满足相同基准延迟 SLO 与批处理假设的算例来说明这种硬件开销差异：

假设我们租用了一台 8 卡 GPU 推理节点，综合成本（包含算力、电力、网络）为 $C = 16$ 美元/小时（示意值）。在该节点部署 70B 模型，对比在基准连续批处理负载下的整机理论吞吐：
- **场景 A（Prefill 阶段）**：长提示词矩阵乘法（GEMM）易于并行饱和张量算力（Compute-bound），假设在基准工作负载下该 8 卡节点整机处理吞吐为 4,000 Prompt Tokens/s。
- **场景 B（Decode 阶段）**：自回归逐字生成受限于显存读取带宽（Memory-bound），在维持流式首字与交互延迟 SLO 的连续批处理并发下，假设该 8 卡节点整机持续生成吞吐为 1,000 Completion Tokens/s。

![Prefill与Decode计算耗时与Token经济成本非对称对比](/images/content-rebuild/99a2f5d8a6342759/fig-4.png)

上图直观展示了硬件执行特征对机器时间占用的参考影响。我们以此推算每百万（1M）Token 的硬件时间分摊成本：

1. **每小时整机能处理的 Prompt Token 上限**：
   $$\text{Capacity}_{\text{prefill}} = 4000 \times 3600 = 14,400,000 \text{ tokens/hr}$$
   $$\text{Cost}_{\text{per 1M Prompt}} = \frac{\$16}{14.4} \approx \$1.11$$

2. **每小时整机能生成的 Completion Token 上限**：
   $$\text{Capacity}_{\text{decode}} = 1000 \times 3600 = 3,600,000 \text{ tokens/hr}$$
   $$\text{Cost}_{\text{per 1M Completion}} = \frac{\$16}{3.6} \approx \$4.44$$

从计算过程与访存开销来看，在基准并发下**生成 1 个 Completion Token 所占用的 GPU 显存搬运与时钟时间，约为并发处理 1 个 Prompt Token 的 4 倍**（即约 $\$4.44 / \$1.11$）。公有云最终定价由供应商发布，但企业在内部做多租户成本分摊核算时，必须依据实测的吞吐差异将 Prompt Token 与 Completion Token 分开加权计费：

$$\text{Cost}_{\text{request}} = \alpha \cdot \text{Tokens}_{\text{prompt}} + \beta \cdot \text{Tokens}_{\text{completion}}$$

其中权重比 $\beta / \alpha$ 应依据引擎在目标模型下的实测吞吐比进行校准。关于应用层的详细预算控制与提示词裁剪，可延伸阅读 [/architecture/token-cost.md](/architecture/token-cost.md) 与 [/llm-100/deployment/llm-cost-optimization.md](/llm-100/deployment/llm-cost-optimization.md)。

---

## 6. 降本增效实战路径：动态装卸、弹性伸缩与 Spot 实例

单纯依赖业务端裁剪 Prompt 无法从根本上解决显卡空载消耗问题。工程落地的降本通常结合基础设施与调度层面的技术手段。

### 6.1 Multi-LoRA 共享底座架构

当企业内部存在多个特定业务微调模型（如客服助手、合同审查、代码审核）时，为每个微调模型单独分配独立的 GPU 节点会导致严重的资源闲置与高昂成本。

- **方案对比**：传统的独立部署需要 $N \times 8$ 张 GPU；而采用 Multi-LoRA 架构时，多副本仅需常驻同一套 Base 模型权重。
- **工作机制**：底层推理引擎（如 vLLM 支持的 Multi-LoRA 特性）在 Batch 计算 GEMM 时，底座权重共享，针对不同请求动态叠加微小的 LoRA A/B 矩阵分支。每个 LoRA 适配器仅占用几十到几百兆显存，显存占用与冷热装载机制详见 [/llm-100/finetuning/qlora-cost-reduction.md](/llm-100/finetuning/qlora-cost-reduction.md)。

### 6.2 基于队列深度的弹性缩容（Scale-to-Zero）

传统基于 CPU/GPU 利用率的 K8s HPA 扩缩容在 LLM 场景下反应迟钝。生产环境可接入 KEDA（Kubernetes Event-driven Autoscaling），订阅 Prometheus 中的 `vllm:num_requests_waiting` 指标：

1. **波谷缩容**：当等待队列持续为 0 且超过设定的闲置阈值（如 15 分钟），逐步释放计算 Pod，对于低频长尾模型甚至缩容至 0 副本。
2. **冷启动加速防线**：Scale-to-Zero 的主要瓶颈是数十 GB 权重加载耗时较长。落地时可采用本地 NVMe SSD 缓存权重、使用 `mmap` 进行内存映射，配合容器镜像预装依赖，将冷启动加载控制在较短时间内。

![弹性扩缩容与Spot抢占容灾机制](/images/content-rebuild/99a2f5d8a6342759/fig-5.png)

上图展示了利用 Spot 实例与常驻按需实例混合组网的容灾流向。阅读提示：核心在于网关层的主动健康探测与优雅排空（Drain）。

### 6.3 竞价（Spot）实例容灾架构

公有云的 Spot（竞价）实例价格通常低于按需实例，但面临被云厂商回收中断的风险。

在推理部署中利用 Spot 实例的容灾处理流程如下：
- **无状态 Prefill 卸载**：在 Prefill/Decode 物理分离部署架构下，将高算力消耗、短周期的 Prefill 节点放置在 Spot 实例池；单次中断只需由网关层快速重试重发，不影响用户长会话状态。
- **中断预警拦截（Termination Notice）**：公有云在回收 Spot 实例前通常会发出提前通知（如 AWS 的 2 分钟预警）。监控 DaemonSet 捕获该事件后，立刻向服务网关发送反注册信号，进入 Drain 状态：**拒绝新请求接入，等待当前正在生成的自回归请求结束**。
- **按需保底与混合混部**：生产集群必须维持最低限度的按需（On-demand）实例保障基础服务，流量波峰则向 Spot 实例弹性外溢。

---

## 7. 面试追问、高频陷阱与横向对比

### 7.1 面试深度追问

#### 追问 1：如果线上监控发现某业务的 TTFT 突然飙升，但 TPOT（每字生成速度）完全正常，你会沿着什么逻辑排查？

**回答思路**：
1. **首查网关与队列**：检查 `vllm:num_requests_waiting`。如果排队深度急剧上升，说明并发请求超过了引擎的单批吞吐能力，请求卡在队列中导致 TTFT 整体后移。
2. **再查 Prompt 长度与缓存命中**：检查输入请求的 Prompt Tokens 统计与 `gpu_prefix_cache_hit_rate`。若业务方突然灌入长文档或 Prompt 模板改动导致 Prefix Cache 失效，Prefill 计算时间会呈线性或超线性增长。
3. **排查动态换出（Swapping）**：观察 `vllm:num_requests_swapped` 是否大于 0。如果显存耗尽，即使单个请求解码正常，调度器在调度新请求时也必须执行昂贵的 Host-to-Device 显存搬运，导致首包时间极大拉长。

#### 追问 2：为什么不能简单地用传统微服务的 P99 响应耗时（Latency）来定义大模型的可用性 SLA？

**回答思路**：
大模型服务单次请求的总响应时间（$T_{\text{total}} = \text{TTFT} + \text{TPOT} \times \text{Tokens}_{\text{output}}$）严重依赖于用户期望生成的最大 Token 数量。生成 10 个字与生成 2,000 个字在同一系统上的物理耗时必然存在数十倍差异。直接用总请求耗时作为 SLA 违背了输入变量受控原则。

生产环境应采用双轨制 SLO：
- **交互响应 SLO**：度量 TTFT P95（例如要求 $< 500\text{ms}$），保障用户界面的首屏交互感知。
- **生成速率 SLO**：度量 ITL（Inter-Token Latency）或每秒 Token 数（例如要求平均生成速率 $> 30\text{ tokens/s}$），保障流式阅读的连续性。

![大模型部署面试核心故障排查决策树](/images/content-rebuild/99a2f5d8a6342759/fig-6.png)

上图系统化总结了面试中面临高频告警时的标准定位决策路径。遇到具体告警时，先看队列再查显存，最后联动底层 DCGM 指标，能条理清晰地展示工程排查思路。

### 7.2 生产常见错误认识

在面试答辩或架构设计中，必须规避以下常见误区：

- ❌ **错误说法一：“看到 GPU-Util 长期在 95% 以上，说明模型推理卡已经充分发挥了全部算力性能。”**
  *修正*：在大模型 Decode 阶段，即使一个批处理很小的受限 GEMV Kernel 在跑，驱动层采样也会报出 100% 的 GPU-Util。必须结合 `PIPE_TENSOR_ACTIVE` 才能确认张量单元是否真正饱和；Decode 往往受限于显存带宽而非计算核心。
- ❌ **错误说法二：“为了避免显存溢出，只要通过流式请求按需分配，就永远不会发生 OOM。”**
  *修正*：自回归过程中 KV Cache 是随生成长度动态累加的。如果引擎调度器未采用物理分页管理（如 PagedAttention），并发请求在达到最大长度时仍可能引发动态显存耗尽，触发服务崩溃或灾难性的显存换出（Swap）。具体显存预估与保障机制需参考 [/interview/model-deployment-interview/memory-estimation.md](/interview/model-deployment-interview/memory-estimation.md)。
- ❌ **错误说法三：“Spot 实例随时可能中断，只能用于无状态的离线任务，绝对不能部署高可用的在线 LLM 推理。”**
  *修正*：通过网关的主动健康拦截、Prefill 阶段的无状态重试以及结合按需实例的混合调度，Spot 实例完全可以安全承载大量弹性算力，在保底 SLA 不受影响的前提下降低综合 TCO。

---

## 8. 本章小结与部署知识链条

生产可观测性与成本工程是大模型走向商业化落地的最后一道防线。它不是简单的搭几个仪表盘，而是要求工程师深刻理解 Transformer 架构在 Prefill 与 Decode 阶段的计算与访存瓶颈，并将其映射到硬件指标、引擎调度与财务账单上。

掌握了可观测性与成本核算之后，你已经完成了大模型部署从底层机制到顶层运维的完整学习：
- 回顾指标与基础延迟：[/interview/model-deployment-interview/core-metrics.md](/interview/model-deployment-interview/core-metrics.md)、[/interview/model-deployment-interview/latency-throughput.md](/interview/model-deployment-interview/latency-throughput.md)
- 进阶调度与显存机制：[/interview/model-deployment-interview/vllm-paged-attention.md](/interview/model-deployment-interview/vllm-paged-attention.md)、[/interview/model-deployment-interview/continuous-batching.md](/interview/model-deployment-interview/continuous-batching.md)
- 容灾限流与综合优化：[/interview/model-deployment-interview/rate-limiting.md](/interview/model-deployment-interview/rate-limiting.md)、[/interview/model-deployment-interview/optimization.md](/interview/model-deployment-interview/optimization.md)
- 返回模块总览：[/interview/model-deployment-interview/README.md](/interview/model-deployment-interview/README.md)
