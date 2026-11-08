---
title: 大模型推理的首 Token 延迟和平均生成速度分别受什么影响？
icon: mdi:timer-outline
tag:
- 模型部署面试
- 首Token延迟
- 生成速度
description: 从 Prefill 与 Decode 两阶段的算子形态（GEMM 与 GEMV）出发，结合 Roofline 模型深入拆解大模型推理中计算受限与访存受限的成因与优化策略。
---

![首 Token 延迟与生成吞吐的硬件瓶颈拆解图](/images/content-rebuild/d61a5db96877b054/cover.png)

在评估大模型线上服务性能时，常会遇到两个看似矛盾的现象：输入提示词（Prompt）增加一倍，首字吐出时间（Time to First Token, TTFT）成倍变慢，但随后每个字弹出的时间间隔（Inter-Token Latency, ITL）几乎不变；单请求生成时 GPU 利用率看似不高，但当并发请求增加时，系统每秒吐出的总 Token 数反而翻了数倍。

这些现象的根源在于：大语言模型（Decoder-only 架构）的推理并非单一均匀过程，而是清晰地划分为 **Prefill（预填充）** 与 **Decode（自回归解码）** 两个物理特征完全不同的阶段。本文用统一的计算访存比与硬件模型，拆解两阶段的瓶颈差异与工程解法。

---

## 1. 面试口述答案：60 秒极简应答

面试中被问到“首 Token 延迟和平均生成速度分别受什么影响、为什么”时，可按以下结构作答：

> 大模型推理包含两个执行阶段，两者的计算形态和硬件瓶颈完全不同：
>
> 1. **首 Token 延迟（TTFT）** 主要取决于 **Prefill 阶段**。该阶段将用户的输入提示词整段并行输入网络，核心算子是矩阵乘矩阵（**GEMM**）。由于提示词内所有 Token 的权重被共同复用，算术强度（计算量/访存量）较高，通常处于 GPU 的 **计算受限（Compute-bound）** 状态。决定首字延迟的核心因素是 **Prompt 长度**、**模型参数量** 和 GPU 的 **Tensor Core 浮点峰值算力（TFLOPs）**。
> 2. **单 Token 生成速度（ITL）与并发吞吐** 主要取决于 **Decode 阶段**。该阶段受制于自回归机制，必须逐字串行生成，每步只能输入 1 个 Token，核心算子退化为矩阵乘向量（**GEMV**）。每生成 1 个 Token，GPU 都必须把全量模型权重从显存读取到片上计算单元一次，算术强度极低，处于严重的 **显存带宽受限（Memory-bandwidth-bound）** 状态。决定生成速度的核心因素是 **显存带宽（GB/s）**、**模型显存占用** 以及 **Batch Size 摊薄效应**。
> 3. **优化方向截然不同**：降低首字延迟需要提升并行算力、使用注意力显存分块（如 FlashAttention）或做张量并行；提升生成速度与吞吐则需提高显存带宽利用率（如权重与 KV 量化）、采用多查询注意力（GQA）减少 KV 搬运，并通过连续批处理（Continuous Batching）增大并发来摊薄权重读取。

![Prefill 与 Decode 两阶段数据流动与算子形态对比](/images/content-rebuild/d61a5db96877b054/fig-1.png)

<ReadMoreLock />

---

## 2. 贯穿案例：长入短出 vs 短入长出的耗时差异

为了直观理解两阶段的独立性，设想我们在一张显存带宽为 $2\text{ TB/s}$、FP16 密集体算力为 $312\text{ TFLOPs}$ 的 GPU 上部署一个 8B 参数模型（FP16 权重约占用 16 GB 显存）。考虑两个工作量不同的典型场景：

*   **场景 A（文档摘要，长入短出）**：输入 Prompt 为 4000 tokens，输出回答为 50 tokens。
*   **场景 B（代码编写，短入长出）**：输入 Prompt 为 50 tokens，输出代码为 4000 tokens。

在**场景 A**中，Prefill 需要一次性对 4000 个 Token 进行多层前向传播，由于序列较长，注意力与投影计算量较大，用户需要等待数百毫秒才能看到第一个字；但一旦首字吐出，后续只生成 50 次，输出瞬间完成。此时系统的主要体验瓶颈在于首 Token 延迟。

在**场景 B**中，Prefill 仅处理 50 个 Token，首字几乎在几十毫秒内立刻响应；但随后的 Decode 阶段需要逐字串行循环 4000 次，每次循环都必须将 16 GB 权重在显存总线上完整读取一次。总耗时被数千次串行显存搬运牢牢卡住。此时系统的主要瓶颈在于 Decode 吞吐与生成延迟。

两者的差异由算子底层的计算访存比决定，下面我们给出数学推导。

---

## 3. Prefill 阶段机制：并行处理与 GEMM 的计算受限特征

Prefill 阶段的目标是处理用户输入的全部已知序列，生成首个输出 Token，并把整段 Prompt 对应的 Key 和 Value 向量缓存起来，供后续生成使用。关于键值缓存的数据组织，可参考 [KV Cache 在模型部署中有什么作用？](kv-cache-deploy.md)。

### 算术强度推导

衡量一个算子受限于算力还是受限于访存，业内通用的指标是 **算术强度（Arithmetic Intensity, $I$）**：

$$I = \frac{\text{浮点运算量 (FLOPs)}}{\text{显存访存量 (Bytes)}}$$

设输入 Batch Size 为 $B$，Prompt 序列长度为 $S_{in}$，隐藏层维度为 $d_{model}$，前向全连接投影权重为 $W \in \mathbb{R}^{d_{model} \times d_{out}}$。

在 Prefill 阶段，由于 Prompt 的所有 Token 序列在时间上同时已知，前向计算可写为通用矩阵乘法（GEMM）：

$$Y = X W, \quad X \in \mathbb{R}^{(B \cdot S_{in}) \times d_{model}}, \quad W \in \mathbb{R}^{d_{model} \times d_{out}}$$

在 FP16 精度（每个浮点数占用 2 字节）下：
*   **浮点运算量**：$2 \times (B \cdot S_{in}) \times d_{model} \times d_{out}$ FLOPs；
*   **权重搬运量**：权重矩阵仅需从显存（HBM）加载至片上缓存（SRAM）一次，大小为 $2 \times d_{model} \times d_{out}$ Bytes；
*   **激活搬运量**：输入与输出激活张量进出显存，大小为 $2 \times (B \cdot S_{in}) \times (d_{model} + d_{out})$ Bytes。

当输入总 Token 数量 $B \cdot S_{in}$ 较大时（例如单请求 $B=1, S_{in}=2048$），权重矩阵在片上 SRAM 中被数千个 Token 的向量多次复用，权重搬运开销被大幅摊薄。若忽略相对较小的激活数据搬运，投影层的理论算术强度近似为：

$$I_{prefill} \approx \frac{2 \cdot (B \cdot S_{in}) \cdot d_{model} \cdot d_{out}}{2 \cdot d_{model} \cdot d_{out}} = B \cdot S_{in} \quad (\text{FLOPs/Byte})$$

当 $B \cdot S_{in} = 2048$ 时，算术强度约为 $2048\text{ FLOPs/Byte}$。这一数值远高于主流数据中心 GPU 的硬件分界点，因此 GPU 的 Tensor Core 处于饱满运行状态，此阶段表现为典型的 **计算受限（Compute-bound）**。

![Prefill 阶段 GEMM 算子执行与数据复用机制](/images/content-rebuild/d61a5db96877b054/fig-2.png)

---

## 4. Decode 阶段机制：逐字自回归与 GEMV 的访存受限特征

模型生成后续内容时，下一个 Token 依赖于前一个生成的 Token，无法预知未来。因此，在第 $t$ 步生成时，输入模型前向网络的只有前一步生成的 **1 个 Token**。

### 算子退化为 GEMV

在单请求（$B=1$）情况下，输入向量形态退化为 $[1, d_{model}]$。线性变换算子退化为矩阵乘向量（GEMV）：

$$y_t = x_t W, \quad x_t \in \mathbb{R}^{1 \times d_{model}}, \quad W \in \mathbb{R}^{d_{model} \times d_{out}}$$

核算单步生成的算术强度：
*   **浮点运算量**：$2 \times 1 \times d_{model} \times d_{out}$ FLOPs；
*   **权重搬运量**：$2 \times d_{model} \times d_{out}$ Bytes；
*   **算术强度**：

$$I_{decode} \approx \frac{2 \cdot d_{model} \cdot d_{out}}{2 \cdot d_{model} \cdot d_{out}} = 1 \quad (\text{FLOPs/Byte})$$

算术强度仅为 $1\text{ FLOPs/Byte}$。这意味着：**仅仅为了对这 1 个 Token 做一次前向投影，GPU 必须将庞大的模型权重在显存与计算核心之间完整倒腾一遍。**

仍以 8B 参数模型（FP16 权重约 16 GB）为例：每生成 1 个 Token，就必须从显存总线搬运约 16 GB 权重。而这 1 个 Token 在全连接投影中的浮点计算量仅约 $2 \times 8 \times 10^9 = 16\text{ GFLOPs}$。在算力达数百 TFLOPs 的现代 GPU 上，计算核心只需几十微秒就能算完，其余大部分时间都在等待总线传输数据，计算核心严重空闲。这就是 Decode 阶段处于 **显存带宽受限（Memory-bandwidth-bound）** 的根本物理原因。

![Decode 阶段 GEMV 算子显存搬运与计算核心闲置现象](/images/content-rebuild/d61a5db96877b054/fig-3.png)

---

## 5. 瓶颈量化分析：Roofline 模型与临界 Batch 计算

**Roofline 模型** 是评估算子执行效率的标准工具。它描述了系统可达到的最大性能上限受算力与带宽的双重制约：

$$P_{attainable} = \min\left(P_{peak},\; I \times BW_{mem}\right)$$

其中：
*   $P_{peak}$：加速卡理论峰值浮点算力（FLOP/s）；
*   $BW_{mem}$：加速卡物理显存带宽（Bytes/s）；
*   $I$：算子的算术强度（FLOPs/Byte）。

### 硬件临界拐点（Knee Point）

算力上限线与带宽上限线相交的点称为 **临界算术强度（$I^*$）**：

$$I^* = \frac{P_{peak}}{BW_{mem}}$$

以 NVIDIA A100 SXM4 (80GB) 为例：
*   FP16 Tensor Core 密集体理论峰值算力 $P_{peak} \approx 312\text{ TFLOPs} = 312 \times 10^{12}\text{ FLOP/s}$；
*   HBM2e 物理显存带宽 $BW_{mem} \approx 2039\text{ GB/s} = 2.039 \times 10^{12}\text{ Bytes/s}$；
*   临界算术强度：

$$I^* = \frac{312 \times 10^{12}}{2.039 \times 10^{12}} \approx 153\text{ FLOPs/Byte}$$

*   **当 $I < 153$ 时**：落在左侧倾斜区间，性能受限于显存带宽；
*   **当 $I \ge 153$ 时**：落在右侧平坦区间，性能受限于计算核心算力峰值。

![Roofline 模型下 Prefill 与 Decode 的工作点及拐点位置](/images/content-rebuild/d61a5db96877b054/fig-4.png)

### Decode 阶段的临界 Batch Size 推导

单请求（$B=1$）解码时，$I_{decode} \approx 1$，远远落在带宽受限区的极左端。如果通过并发批处理将 Batch Size 增大至 $B$：
*   权重仍读取 1 次（大小为 $2P$ 字节）；
*   计算量放大为 $2P \cdot B$ FLOPs；
*   投影权重部分的算术强度随并发线性增加为 $I(B) \approx B\text{ FLOPs/Byte}$。

若不考虑 KV Cache 读取，仅就模型权重读取而言，要让 Decode 阶段越过拐点、跑满 Tensor Core 算力，理论上需要的批大小为：

$$B_{critical} \approx I^* = \frac{P_{peak}}{BW_{mem}} \approx 153$$

在实际工程中，除了读取权重，每个 Token 解码还必须从显存中读取历史累积的 KV Cache。随着上下文变长，KV 显存读取量增加，分母变大，会使实际的算术强度拐点进一步向更大的 Batch Size 偏移。这也是推理框架广泛引入 [连续批处理 Continuous Batching](continuous-batching.md) 的动力：增大批大小能有效摊薄权重搬运成本，在单字时延增加可控的前提下成倍提升系统整体吞吐。

---

## 6. 输入输出长度对时延曲线的非对称影响

总端到端响应耗时可分解为：

$$T_{total} = T_{prefill}(S_{in}) + \sum_{t=1}^{S_{out}} T_{decode}(S_{in} + t)$$

这种结构导致输入与输出长度对整体耗时产生明显的 **非对称影响**：

| 对比场景 | 输入长度 $S_{in}$ | 输出长度 $S_{out}$ | Prefill 耗时特征 | 单步 Decode 耗时特征 | 用户感知的耗时形态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **场景 A（长入短出）** | 4000 tokens | 50 tokens | 较高（GEMM 计算量大） | 极短（仅迭代 50 次） | 首字出现明显较慢，但后续内容瞬间完成。 |
| **场景 B（短入长出）** | 50 tokens | 4000 tokens | 极短（几十毫秒完成首字） | 漫长（受显存带宽限制迭代 4000 次） | 首字立刻出现，但整段内容生成需要持续等待。 |

当我们将 Prompt 从 1000 tokens 扩充到 2000 tokens 时：
1. **首字延迟（TTFT）**：前向投影计算量增加一倍，自注意力算术计算量随序列长度二次方增长（即便使用 FlashAttention 避免了中间大矩阵在显存上的物化，其算术复杂度依然是 $O(S^2)$）。因此 TTFT 会明显上涨。
2. **单 Token 解码延迟（ITL）**：单步 Decode 耗时取决于 $\frac{\text{权重数据量} + \text{KV Cache数据量}}{BW_{mem}}$。对于 8B 模型，权重恒定为 16 GB；Prompt 增加 1000 tokens 所增加的 KV Cache 仅几十兆字节，在权重搬运面前占比极小。因此后续每个 Token 的弹出间隔几乎不会发生肉眼可见的变化。

![输入输出序列长度对总响应时序与耗时曲线的非对称影响](/images/content-rebuild/d61a5db96877b054/fig-5.png)

---

## 7. 架构演进与面试深入追问

### 追问 1：混布 Prefill 与 Decode 有什么缺陷？

在传统推理引擎中，Prefill 和 Decode 共享同一组 GPU：
*   **时延抖动（HoL 阻塞）**：当有新请求携带超长 Prompt 到达时，Prefill 计算会瞬间占满 GPU 的 Tensor Core 达数百毫秒。同实例上正在逐字流式返回的 Decode 请求会被迫挂起等待，导致流式输出出现明显卡顿，劣化 P99 级别的单字生成延迟（ITL）。
*   **硬件错配**：Prefill 需要极高的算力密度（算力强的卡），Decode 需要极高的显存带宽和容量（带宽大、容量多的卡）。混布使得单机硬件难以兼顾两者性价比。

虽然 **Chunked Prefill（分块预填充）** 能够把长 Prompt 切成细粒度块穿插到 Decode 中平抑抖动，但切片太小会导致 GEMM 无法吃满 GPU 核心分块，在一定程度上牺牲了 Prefill 阶段的算力利用率。

### 追问 2：什么是 Prefill-Decode 分离（PD 分离）架构？代价是什么？

为避免混布干扰，工业级大规模服务发展出了 **PD 分离架构**：
1. **节点池解耦**：将集群划分为 Prefill 节点组与 Decode 节点组；
2. **流程拆分**：Prefill 节点并行处理长文本 Prompt 并生成第 1 个 Token；随后将该请求生成的 **KV Cache 通过跨节点网络发送到 Decode 节点**；Decode 节点接收后接管后续逐字自回归生成。

![Prefill-Decode 分离架构与跨节点 KV Cache RDMA 传输流程](/images/content-rebuild/d61a5db96877b054/fig-6.png)

**核心权衡（Trade-offs）**：
*   **网络传输开销**：KV Cache 的跨节点搬运耗时必须小于分离带来的调度收益。以一个采用 GQA、32 层、隐藏维度 4096、8 对 KV 头的 8B 模型为例（每头维度 $128$），单个 Token 在 FP16 精度下的 KV Cache 约为：

$$2 \times 32 \times (8 \times 128) \times 2\text{ Bytes} = 131,072\text{ Bytes} = 128\text{ KB}$$

一段 4000 tokens 的上下文，KV Cache 数据量约为 $500\text{ MB}$。如果依靠普通的千兆或万兆以太网传输，网络延迟会完全抵消架构收益。因此 PD 分离**高度依赖 RDMA（RoCEv2 或 InfiniBand）高速集群网络**。
*   **系统控制面复杂度**：涉及跨节点的显存生命周期同步、调度路由和网络重试机制，工程复杂度显著高于单体推理引擎。关于多卡分布式部署的切分方式，可参考 [Tensor Parallel、Pipeline Parallel 和 Data Parallel 有什么区别？](parallel-strategies.md)。

---

## 8. 常见高频概念陷阱辨析

### 误区 1：认为模型量化仅能节省显存容量，不影响生成速度

**辨析**：量化的收益必须分阶段评估。
*   在 **Prefill 阶段**，系统处于计算受限状态。若采用 INT8/INT4 权重并在片上动态反量化为 FP16 计算，由于反量化算子引入额外开销，推理延迟不仅提升不明显，有时甚至可能微弱劣化。
*   在 **Decode 阶段**，瓶颈是显存总线带宽。将 FP16 权重压缩为 INT8 或 INT4，能直接使每次循环从显存搬运的数据量减少一半或四分之三，**单步 Decode 耗时几乎与权重大小成比例下降，能直接成倍提高单请求生成速度和吞吐**。具体量化实现与精度权衡可参阅 [模型量化是什么？](quantization.md)。

### 误区 2：认为 FlashAttention 能同样大幅优化 Decode 阶段的每步时延

**辨析**：FlashAttention 核心解决的是长序列自注意力计算中 $S \times S$ 中间注意力得分矩阵的片外显存读写瓶颈。在 **Prefill 阶段**，序列长度为 $S_{in}$，注意力矩阵尺寸为 $S_{in} \times S_{in}$，消除物化显存的优化收益极大。而在 **单步 Decode 阶段**，每次仅输入 1 个 Query，注意力矩阵退化为 $1 \times S$ 的向量，原本就不存在高维庞大矩阵物化到显存的瓶颈。Decode 阶段的主要访存开销是从显存读取全部历史 KV 向量。因此，Decode 阶段的提速主要依靠减少 KV 存储总量的 GQA 架构和消除内存碎片的 [PagedAttention](vllm-paged-attention.md)。

---

## 9. 总结与指标对比速查

| 评估维度 | Prefill 阶段（首 Token 生成） | Decode 阶段（后续逐字生成） |
| :--- | :--- | :--- |
| **对应观测指标** | 首 Token 延迟（TTFT） | 单 Token 延迟（ITL）、生成吞吐（Tokens/s） |
| **核心前向算子** | 通用矩阵乘法（GEMM） | 矩阵乘向量（GEMV） |
| **硬件受限类型** | **计算受限（Compute-bound）** | **显存带宽受限（Memory-bandwidth-bound）** |
| **单请求算术强度** | 较高（通常大于数百 FLOPs/Byte） | 极低（单并发时约等于 $1\text{ FLOPs/Byte}$） |
| **硬件决定因素** | Tensor Core 算力峰值（TFLOPs） | 显存带宽（GB/s）、显存容量 |
| **核心优化方案** | FlashAttention、张量并行、Chunked Prefill | 连续批处理、权重/KV量化、PagedAttention、GQA |

理解了两阶段底层硬件瓶颈的物理成因后，建议继续深入阅读推理引擎的显存管理与调度机制：
*   [KV Cache 在模型部署中有什么作用？](kv-cache-deploy.md)：理解自回归模型如何用空间换时间消除重复计算；
*   [vLLM 是什么？PagedAttention 为什么能提升推理吞吐？](vllm-paged-attention.md)：深入学习显存分页管理如何减少内部碎片并支撑大规模并发 Decode。
