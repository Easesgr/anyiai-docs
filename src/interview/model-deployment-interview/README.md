---
title: 大模型推理部署面试题：从性能瓶颈到显存调度与故障排查
icon: mdi:cloud-outline
tag:
- 模型部署面试
- 面试题
- 模型部署
description: 系统整理大模型部署面试考点：涵盖延迟吞吐指标、Prefill与Decode硬件瓶颈、PagedAttention显存管理、连续批处理、70B模型算力测算与生产排障。
---

大模型推理部署面试不仅考察模型结构本身，更重点考察候选人对硬件执行瓶颈（计算受限 vs 访存受限）、显存动态占用（模型权重、KV Cache、激活值）、迭代级调度逻辑与服务可用性治理的工程掌握程度。

本篇作为面试复习导航，先给出一套可直接口述的核心结论与贯穿例子，再梳理端到端请求流向、12 篇核心考点导航、高频系统设计题答题步骤，以及生产常见故障的排查路径。

![大模型部署面试体系全景图](/images/content-rebuild/e3b16eb878aee4f7/cover.png)

---

## 1. 口述核心答案：大模型推理部署与传统服务有何不同？

如果面试官要求在两分钟内讲清大模型推理服务的核心瓶颈与架构特点，可以口述以下重点：

> 传统 Web 服务是无状态的 CPU/IO 密集型任务，扩容依赖无状态副本复制；大模型在线推理是有状态的 GPU 资源密集型任务，单个请求具有多轮自回归生成的长周期生命周期。
>
> 大模型推理在计算上有两个完全不同的阶段：
> 1. **Prefill（预填充阶段）**：将输入的全部 Prompt 一次性计算出首个输出 Token，属于计算受限（Compute-bound）的大矩阵乘法（GEMM），决定首字延迟（TTFT）。
> 2. **Decode（自回归生成阶段）**：每次只输入前一步生成的单个 Token，结合之前保存的全部 Key 和 Value 张量（KV Cache）计算下一步，属于显存带宽受限（Memory-bandwidth-bound）的矩阵向量乘法（GEMV），决定每个 Token 的生成速度（TPOT）。
>
> 系统设计的两大核心矛盾是：
> - **显存碎片与并发容量矛盾**：动态增长的 KV Cache 极易造成显存碎片，现代推理框架通过页式显存分配（如 PagedAttention）消除碎片并支持跨请求前缀共享。
> - **不同请求长度不一带来的算力浪费矛盾**：传统静态批处理会导致提前结束的请求等待最长请求，产生大量计算空泡；现代框架采用以每步生成为粒度的连续批处理（Continuous Batching）和分块预填充（Chunked Prefill），动态插入新请求并复用算力。

---

## 2. 贯穿实例与端到端系统数据流

为了将各项技术串联起来，本文全程以一个典型生产目标为例：**在单台 4 卡 GPU 服务器上，部署开源 70B 级别大模型，稳定支撑 200 个并发用户的实时问答**。

当 200 个并发请求陆续到达时，现代推理引擎（如 vLLM、TensorRT-LLM）内部由以下五个阶段流转完成处理：

1. **网关接入与并发准入**：客户端发起 HTTP 流式请求，API 网关负责鉴权并检查活跃并发槽位。如果引擎显存接近饱和，网关在接入层实施限流排队，防止过载请求直接击垮显存。
2. **等待队列与迭代级组批**：请求进入引擎调度器的等待队列（Waiting Queue）。调度器以单个迭代步（Iteration Step）为调度周期，向物理显存块管理器（Block Manager）申请显存配额，将符合条件的请求推进至运行队列（Running Queue），实现连续批处理。
3. **Prefill 计算与写缓存**：请求首次被执行，其 Prompt 中的所有 Token 一次性并行输入模型计算。计算产出第 1 个输出 Token，同时将该请求各层生成的 Key/Value 向量写入逻辑块映射的非连续物理显存块中。
4. **自回归 Decode 循环迭代**：首字生成后，请求转入 Decode 阶段。此后每轮迭代只输入前一步生成的 1 个 Token，从物理显存池中按需读取历史 KV Cache 进行计算。调度器随着生成长度增加为该请求追加物理块，直至输出结束标记符（EOS）或达到最大长度。
5. **显存回收与流式推送**：生成结束后，调度器通知显存管理器释放该请求占用的物理块，并断开流式连接；增量 Token 通过 Server-Sent Events（SSE）以打字机形式推给客户端。

![推理引擎全链路数据流与关键交互全景](/images/content-rebuild/e3b16eb878aee4f7/fig-1.png)

---

## 3. 知识图谱与 12 篇核心考点导航

大模型推理优化的各技术模块不是孤立的，而是围绕“指标定义 -> 显存管理 -> 批处理调度 -> 分布式与服务治理”层层递进：

### 第一部分：性能指标与硬件瓶颈分析
- [/interview/model-deployment-interview/core-metrics.md](/interview/model-deployment-interview/core-metrics.md)
  **面试考察点**：首字延迟（TTFT）、每输出 Token 延迟（TPOT）、Token 间延迟（ITL）与系统吞吐量（Tokens/s）的计算公式、测试方法与服务等级协议（SLA）制定逻辑。
- [/interview/model-deployment-interview/latency-throughput.md](/interview/model-deployment-interview/latency-throughput.md)
  **面试考察点**：为什么首 Token 依赖 GPU 的峰值浮点算力（TFLOPS），而后序生成依赖显存带宽（GB/s）；Roofline 模型如何量化算力受限与访存受限的分界线。
- [/interview/model-deployment-interview/memory-estimation.md](/interview/model-deployment-interview/memory-estimation.md)
  **面试考察点**：7B、13B、70B 模型在不同精度（FP16/INT8/FP8）下的静态权重开销、激活值缓冲与动态 KV Cache 显存占用估算方法。

### 第二部分：推理引擎内核技术
- [/interview/model-deployment-interview/kv-cache-deploy.md](/interview/model-deployment-interview/kv-cache-deploy.md)
  **面试考察点**：自回归解码中 KV Cache 显存占用随长度累加的计算公式；多头注意力（MHA）、分组查询注意力（GQA）和多头潜在注意力（MLA）在显存占用上的具体差异。
- [/interview/model-deployment-interview/vllm-paged-attention.md](/interview/model-deployment-interview/vllm-paged-attention.md)
  **面试考察点**：PagedAttention 的逻辑块与物理块映射表设计；如何借鉴虚拟内存分页消除显存内部与外部碎片；并行采样与前缀共享的写时复制（Copy-on-Write）原理。
- [/interview/model-deployment-interview/continuous-batching.md](/interview/model-deployment-interview/continuous-batching.md)
  **面试考察点**：静态批处理中不同长度请求造成的计算气泡成因；基于迭代级调度（Iteration-level Scheduling）动态插入新请求和退出结束请求的调度逻辑。

### 第三部分：分布式切分与模型压缩
- [/interview/model-deployment-interview/quantization.md](/interview/model-deployment-interview/quantization.md)
  **面试考察点**：仅权重（Weight-only，如 AWQ、GPTQ）与权重激活联合量化（Weight-Activation，如 SmoothQuant、FP8）的精度损失与真实加速比权衡；KV Cache 量化对长文本显存的缩减效果。
- [/interview/model-deployment-interview/parallel-strategies.md](/interview/model-deployment-interview/parallel-strategies.md)
  **面试考察点**：张量并行（TP）、流水线并行（PP）与数据并行（DP）的通信开销对比；为什么在线推理优先选择单机多卡张量并行配合 NVLink 高速互联，而不建议跨节点走低速以太网做 TP。

### 第四部分：生产服务化治理与运维
- [/interview/model-deployment-interview/optimization.md](/interview/model-deployment-interview/optimization.md)
  **面试考察点**：分块预填充（Chunked Prefill）如何平抑 Decode 延迟抖动；投机采样（Speculative Decoding）的加速比与接受率依赖；自动前缀缓存（Automatic Prefix Caching）在多轮对话中的命中条件与收益。
- [/interview/model-deployment-interview/openai-compatible-api.md](/interview/model-deployment-interview/openai-compatible-api.md)
  **面试考察点**：SSE 流式协议在推理网关中的边界处理；多模态输入与工具调用（Tool Calling）的流式返回解析；网关代理对流式响应的缓冲踩坑点（如反向代理默认开启 Buffer 导致首字延迟暴增）。
- [/interview/model-deployment-interview/rate-limiting.md](/interview/model-deployment-interview/rate-limiting.md)
  **面试考察点**：并发槽位与 Token 速率双重限流策略；显存水位预警下的熔断机制与请求驱逐（Preemption，重算 vs 换出）策略。
- [/interview/model-deployment-interview/monitoring-cost.md](/interview/model-deployment-interview/monitoring-cost.md)
  **面试考察点**：`nvidia-smi` 采样的 GPU 利用率（GPU-Util）虚高的物理原因；DCGM 采样张量核心活跃度（Tensor Active）；Prefill-Decode 分离部署架构与单 Token 综合运行成本核算。

---

## 4. 面试系统设计四步法与 70B 案例实战

面对大模型系统设计与部署类追问，建议采用以下结构化逻辑作答：

![面试结构化答题四步法框架](/images/content-rebuild/e3b16eb878aee4f7/fig-2.png)

### 4.1 四步答题框架
1. **结论前置**：直接给出推荐的部署架构、硬件卡型、并行切分方式和引擎选型。
2. **瓶颈分析**：从算力墙与显存墙出发，分析该模型在 Prefill 和 Decode 两个阶段的资源开销瓶颈。
3. **定量测算**：给出显存占用（模型静态参数、动态 KV Cache、激活值）、带宽消耗与并发容量的具体计算过程。
4. **工程取舍与降级**：分析长文本突发、网络通信延迟等边界情况，并给出降级、熔断与保活方案。

### 4.2 案例实战：单台 4 卡服务器部署 70B 模型支撑 200 并发

**第 1 步：结论前置**
推荐方案：采用 **4 卡 H100 80GB SXM** 服务器（若在 A100 服务器上则对应采用 INT8 量化或 AWQ/GPTQ 方案），模型采用 **FP8 量化**（权重与激活均量化为 8 位，Hopper 架构原生支持），组建 **TP=4（张量并行度为 4）** 实例；使用 **vLLM** 引擎，开启 **PagedAttention** 与 **Chunked Prefill**，并将 KV Cache 精度设置为 FP8。

**第 2 步：瓶颈分析**
- 70B 参数模型在 FP16 下仅静态权重就需要约 140 GB 显存，单张 80GB 卡无法装载，必须采用张量并行将参数均匀切分到 4 张卡上。
- 在承载 200 个并发自回归生成时，系统处于重度访存受限状态，显存的主要消耗是随并发数和上下文长度线性累加的 KV Cache。如果模型未采用 GQA 或未开启分页管理，显存将迅速耗尽导致 OOM（显存溢出）。

**第 3 步：定量显存测算**
设定以下基准参数（符合主流 70B 开源模型典型配置，如 Llama-3-70B）：
- 模型层数 $L = 80$
- 注意力头数分为两类：查询头 $Q = 64$，键值头采用分组查询注意力（GQA），键值头数 $KV = 8$
- 每个注意力的头维度 $d = 128$
- 假设 200 个并发请求的平均上下文长度为 $N = 2048\text{ tokens}$
- 采用 FP8 精度（每个数值 1 字节）存储 KV Cache 与权重

**静态权重显存**：
$$70\text{B 参数} \times 1\text{ 字节} \approx 70\text{ GB}$$
切分到 4 张 GPU 后，每张卡承担静态权重：
$$\frac{70\text{ GB}}{4} \approx 17.5\text{ GB}$$

**单 Token 的 KV Cache 显存开销**：
每个 Token 在所有层产生的 Key 和 Value 向量总量为：
$$\text{KV}_{\text{token}} = 2 \times L \times KV \times d \times \text{Bytes}$$
$$\text{KV}_{\text{token}} = 2 \times 80 \times 8 \times 128 \times 1\text{ 字节} = 163,840\text{ 字节} \approx 160\text{ KB}$$

**200 并发动态 KV Cache 总量**：
单请求 2048 长度需要的 KV Cache 为：
$$2048 \times 160\text{ KB} \approx 320\text{ MB}$$
200 并发全局所需的 KV Cache 显存为：
$$200 \times 320\text{ MB} = 64,000\text{ MB} \approx 62.5\text{ GiB（约 64 GB）}$$
在 TP=4 配置下，8 个 KV 头被切分到 4 张 GPU（每张 GPU 处理 2 个头），因此单张卡分配到的动态 KV Cache 池为：
$$\frac{64\text{ GB}}{4} = 16\text{ GB}$$

**单卡总显存核算**：
- 静态权重：$17.5\text{ GB}$
- 动态 KV Cache 池：$16\text{ GB}$
- 临时激活值与 CUDA 运行时开销：约 $5\text{ GB}$
- **单卡总占用**：$17.5 + 16 + 5 = 38.5\text{ GB}$

对于 80GB 显存的 GPU，实际占用约 38.5 GB，剩余 40+ GB 可作为突发长文本与请求积压的安全缓冲，显存余量充足。

**第 4 步：工程取舍与降级机制**
- **卡间通信带宽要求**：TP=4 在每个 Transformer 层的前向传播中需要两次跨卡 All-Reduce 通信。因此 4 张 GPU 之间必须具备 **NVLink（卡间双向带宽至少 600 GB/s）** 高速总线。若在普通 PCIe 4.0 插槽上运行 TP=4，通信延迟将成为主导瓶颈，算力将严重闲置。
- **平抑字间延迟抖动（Chunked Prefill）**：200 个并发正在 Decode 时，如果突然进入一个包含 8000 Token 的长 Prompt，未切块的 Prefill 会独占 GPU 算力数秒，导致其他 200 个请求的字间延迟（ITL）严重卡顿。必须开启 Chunked Prefill 并配置 `max_num_batched_tokens`（例如设为 2048），强制将长 Prefill 切块并与 Decode 混合组批执行，保障在线生成平稳输出。
- **显存超限降级**：当瞬时突发请求导致显存物理块使用率突破 95% 时，网关层立即对新增请求返回 HTTP 429 熔断；引擎调度器对积压请求触发驱逐机制（优先重算或挂起低优先级请求），防止直接发生 CUDA OOM 导致服务进程崩溃。

---

## 5. 生产高频故障诊断排查清单

面试官常通过真实的线上异常场景考察候选人的故障定位能力。以下总结五个最高频的排查链路：

![生产高频故障诊断排查树](/images/content-rebuild/e3b16eb878aee4f7/fig-3.png)

### 故障 1：引擎突发 CUDA OOM 崩溃
- **原因定位**：
  1. **区分发生阶段与机理**：在采用预分配显存池的引擎中，Decode 阶段物理块耗尽只会触发请求排队或调度抢占（Swap/重算），不会向底层 CUDA 分配器动态申请新显存；真正的 CUDA OOM 多发生在 Prefill 阶段（超长 Prompt 或批次过大导致前向激活值峰值突破预留）或采样阶段（大词表 Logits 等临时张量超限）。
  2. **检查显存预留比例**：vLLM 等引擎通过 `gpu_memory_utilization` 参数（默认 0.9）在启动时预占显存池。若系统内有多个进程竞争显存，或框架未给 PyTorch 运行时留足显存，就会发生不可捕获的 OOM。
- **解决方案**：
  - 将 `gpu_memory_utilization` 调整至 0.88~0.92，预留 8~10% 的显存缓冲。
  - 在网关层严格配置最大请求上下文上限（Max Model Len）与并发连接硬限，在显存耗尽前执行 HTTP 429 快速失败。

### 故障 2：首字延迟（TTFT）P99 出现长尾突刺
- **原因定位**：
  1. **等待队列积压**：检查调度器排队日志。当请求到达速率超过引擎吞吐极限时，请求在等待队列中滞留，导致首字延迟主要被排队时间消耗。
  2. **长 Prefill 阻塞**：没有开启分块预填充，导致单个长文本输入计算霸占 GPU 张量核心数秒。
  3. **显存换出（Swap）发生**：引擎物理块不足时，触发了将显存块置换到 CPU 宿主机内存的操作。PCIe 带宽（单向约 32~64 GB/s）远低于 GPU 显存带宽（1.5~3.3 TB/s），换入换出引发极高延迟。
- **解决方案**：
  - 开启 `chunked_prefill`，限制单批次最大预填充 Token 数量。
  - 调高显存容量配额，关闭 Swap 机制，改用直接重新计算（Recompute）或在网关层做过载丢弃。

### 故障 3：nvidia-smi 显示 GPU-Util 达 100%，但每秒生成吞吐很低
- **原因定位**：
  1. **指标统计口径误区**：`nvidia-smi` 采样的 GPU-Util 仅代表过去一秒内 GPU 上有 Kernel 运行的时间比例，不代表计算单元的饱和程度。在 Decode 阶段，GPU 执行的是显存带宽受限的向量乘法，计算核心大部分时间在等待数据从显存搬运到寄存器，计算单元真实利用率可能不到 15%。
  2. **批处理大小过小**：并发请求数过低，GPU 显存带宽未被有效利用，每次只为极少量的 Token 读取一次全量模型权重。
  3. **TP 通信链路带宽受阻**：在未接入 NVLink 的普通 PCIe 机器上配置了 TP=4 或 TP=8，GPU 核心大部分时间在等待跨卡通信同步，但 Kernel 依然处于活跃状态。
- **解决方案**：
  - 使用 DCGM 监控 `PIPE_TENSOR_ACTIVE` 指标，观察张量核心的真实利用率。
  - 确认跨卡通信走的是 NVLink 总线；若硬件无 NVLink，应将并行方式调整为流水线并行（PP）或单卡独立实例。

### 故障 4：长文本请求进入后，所有请求生成速度骤降
- **原因定位**：
  1. **平方复杂度计算开销**：自注意力计算量随 Prompt 长度平方级增长。未优化内核时，长文本 Prefill 会严重消耗算力。
  2. **物理显存块迅速耗尽**：长文本快速占满显存块，导致运行中的其他 Decode 任务无法分配到新块，迫使调度器触发换出或重算逻辑。
- **解决方案**：
  - 部署内核级优化算子（如 FlashAttention-2、FlashInfer），降低 Attention 显存读写瓶颈。
  - 针对带有固定系统提示词（System Prompt）的多轮对话，开启自动前缀缓存（Automatic Prefix Caching），避免对重复前缀做二次 Prefill 计算。

### 故障 5：高并发下出现显存死锁与重算风暴
- **原因定位**：
  1. 当可用物理显存块耗尽，调度器为了继续执行，必须中止某些 Decode 请求并释放其 KV Cache。
  2. 如果中止策略选择“释放缓存并在下一周期重新执行 Prefill”，当被中止的请求重新入队时，它所引发的新 Prefill 会再次抢占算力并消耗显存，导致系统中多个请求在“被抢占 -> 从头重新 Prefill -> 显存再次耗尽 -> 再次被抢占”之间无限循环，生成吞吐归零。
- **解决方案**：
  - 设置调度器水位安全边界（Watermark），当可用块低于 5% 时暂停接入新请求，保留显存给已在运行的请求直到完成。
  - 引入基于请求存活时间的优先级策略，保护即将生成完成的长请求，优先让快结束的请求释放显存。

---

## 6. 跨专栏关联阅读指引

大模型推理服务是系统交付的承重层，掌握单机内核优化与显存调度后，建议结合以下专栏继续进阶：

- 前往 [/architecture/README.md](/architecture/README.md) 了解大模型 API 网关流式路由、分布式负载均衡与生产级多租户隔离架构；
- 前往 [/frontier/context-engineering/README.md](/frontier/context-engineering/README.md) 探索长上下文管理策略、上下文压缩与提示词结构对显存占用的直接影响；
- 前往 [/frontier/agentic-rag/README.md](/frontier/agentic-rag/README.md) 评估多步 Agent 循环与密集检索增强下，高频多次调用对推理首字延迟与服务成本的复合压力。
