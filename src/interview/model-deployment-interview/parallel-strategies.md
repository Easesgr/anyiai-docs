---
title: Tensor Parallel、Pipeline Parallel 和 Data Parallel 有什么区别？大模型推理并行策略的选择
icon: mdi:source-branch
tag:
- 模型部署面试
- 并行策略
- 分布式
description: 用 70B 模型部署贯穿对比 TP、PP 和 DP 的切分轴向、通信机制与硬件依赖，给出可口述的面试答案与工程选型依据。
---

以部署一个 70B 参数的开源大模型（如 Llama-3-70B）为例：在半精度（FP16/BF16）下，仅模型权重就需要约 140 GB 显存。单张 80 GB 的 A100 或 H100 显存根本放不下，更不用说运行时还要为上下文分配 KV Cache。这时必须使用多张 GPU 协同工作。

大模型推理与分布式训练不同：推理没有反向传播与优化器状态更新，核心目标是在保证显存放得下的前提下，平衡首 Token 延迟（TTFT）、逐 Token 生成延迟（ITL）与系统吞吐量。常用的并行方式有张量并行（Tensor Parallelism，TP）、流水线并行（Pipeline Parallelism，PP）和数据并行（Data Parallelism，DP）。

![分布式推理并行策略概览](/images/content-rebuild/5d50c09d88bbcc79/cover.png)

---

## 面试口述答案：三者的核心区别与选型规则

如果面试官问：“Tensor Parallel、Pipeline Parallel 和 Data Parallel 有什么区别？推理时该怎么选？”，可以按以下逻辑口述：

> **核心答案**：  
> 三者的切分轴向、通信频率与网络依赖各不相同：
> 1. **Tensor Parallel（TP，张量并行）**：在算子内部切分矩阵（如 Attention 和 MLP）。它把计算和显存均匀分摊到多卡，能**直接降低单请求生成延迟**。但每个 Transformer 层需要 2 次 All-Reduce 同步，通信非常频繁，因此**强依赖机内高带宽 NVLink**，通常只在单机 8 卡以内使用。
> 2. **Pipeline Parallel（PP，流水线并行）**：按网络层数纵向切分模型，不同卡负责不同的层，相邻卡之间只传递激活值（P2P 通信）。它对网络带宽要求较低，适合**跨物理节点承载超大模型**；但在自回归生成阶段存在**流水线气泡**，单请求时硬件利用率较低。
> 3. **Data Parallel（DP，数据并行）**：每组卡保存一份完整的模型副本，不同卡处理不同的请求批次。推理阶段副本之间**没有任何通信**，配合 [连续批处理（Continuous Batching）](/interview/model-deployment-interview/continuous-batching.md) 与 [PagedAttention](/interview/model-deployment-interview/vllm-paged-attention.md)，是**提升系统吞吐量（Throughput）**的首选方案。
> 
> **工程选型准则**：优先在单机内用 TP（如单机 8 卡 TP=8 跑 70B）；如果并发吞吐需求增加，就横向增加 DP 副本；只有当模型大到单机显存完全装不下（如 405B 模型），才在跨机节点间使用 PP + TP 的混合并行。

---

## 张量并行（TP）：矩阵如何切分与何时通信

主流推理框架（如 vLLM、TensorRT-LLM）中的张量并行普遍遵循 Megatron-LM 的切分方式。目标是：**在切分矩阵的同时，把跨卡集合通信（All-Reduce）次数降到最低，并让激活函数能独立在各卡本地完成计算**。

![Self-Attention 的列切分与行切分机制](/images/content-rebuild/5d50c09d88bbcc79/fig-1.png)

### 1. 自注意力模块（Self-Attention）的切分

设隐藏层维度为 $H$，输入张量为 $X$，并行 GPU 数量为 $N$：

1. **QKV 投影采用列切分（Column Parallel）**：  
   将权重矩阵沿列方向等分为 $N$ 份：$W_Q = [W_Q^1, \dots, W_Q^N]$，$W_K$ 和 $W_V$ 同样处理。  
   各卡输入相同的激活值 $X$，分别计算局部的注意力头：
   $$Q_i = X W_Q^i, \quad K_i = X W_K^i, \quad V_i = X W_V^i$$
   因为注意力头之间是各自独立的，GPU $i$ 无需任何通信即可在本地算完 Softmax 并输出局部上下文向量 $Z_i$。同时，每张卡只需要保存自己所负责头部的 KV Cache，显存占用自然降低为 $\frac{1}{N}$。

2. **输出投影（$W_O$）采用行切分（Row Parallel）**：  
   将输出矩阵 $W_O$ 沿行方向等分为 $N$ 份：
   $$W_O = \begin{bmatrix} W_O^1 \\ \vdots \\ W_O^N \end{bmatrix}$$
   各卡直接在本地将局部上下文 $Z_i$ 与本地权重 $W_O^i$ 相乘：$Y_i = Z_i W_O^i$。  
   由分块矩阵乘法原理可知，整体注意力输出为：
   $$Y = Z W_O = \sum_{i=1}^N Z_i W_O^i = \sum_{i=1}^N Y_i$$
   此时只需要在所有卡之间执行一次 **All-Reduce（求和）**，每张卡就能得到完整的注意力输出张量 $Y$。

<ReadMoreLock />

### 2. MLP 模块的切分

![MLP 层的列切分与行切分机制](/images/content-rebuild/5d50c09d88bbcc79/fig-2.png)

两层 MLP 包含升维矩阵 $W_1$ 和降维矩阵 $W_2$：

1. **第一层 $W_1$ 采用列切分**：各卡计算 $H_i = \text{Activation}(X W_1^i)$。因为 GELU 或 SiLU 是按元素独立的非线性激活函数，各卡在本地即可独立完成非线性变换，不需要跨卡同步。
2. **第二层 $W_2$ 采用行切分**：各卡在本地完成乘法 $O_i = H_i W_2^i$。
3. **通信同步**：最后执行一次 **All-Reduce（求和）**，将各卡的输出累加：$O = \sum_{i=1}^N O_i$。

### 3. 通信开销总结

在标准 Transformer 层中：
* **通信次数**：每个 Layer 包含 1 次 Attention 和 1 次 MLP，因此正好需要 **2 次 All-Reduce**。
* **整除约束**：注意力头数必须能被 TP 整除。以 GQA（分组查询注意力）为例，如果 KV 头数是 8，TP 的值就只能选 1、2、4 或 8。如果设为 16，部分卡将分不到 KV 头，需要做额外的广播复制。

关于训练阶段更详细的参数切分推导，可延伸阅读 [数据并行、模型并行、流水线并行分别是什么？](/llm-100/training/parallel-strategies.md)。

---

## 流水线并行（PP）：层间切分、通信特征与气泡问题

当多张卡安装在不同机器上，或者单机总显存仍不够放模型时，就需要使用流水线并行。

![Pipeline Parallelism 推理气泡与微批处理特征](/images/content-rebuild/5d50c09d88bbcc79/fig-3.png)

### 1. 通信模式与拓扑特征

流水线并行按照网络层数纵向拆分。假设模型有 80 层，设置 PP=4：
* Stage 0（GPU 0）：负责第 1 到 20 层及输入 Embedding。
* Stage 1（GPU 1）：负责第 21 到 40 层。
* Stage 2（GPU 2）：负责第 41 到 60 层。
* Stage 3（GPU 3）：负责第 61 到 80 层及最后的 LM Head。

**通信特征**：
* 通信不是全局集合通信，而是**相邻阶段的点对点（P2P）通信**。
* 上一个 Stage 算完后，把中间的隐藏状态（激活值张量）发送给下一个 Stage。
* 跨节点通信仅发生在阶段交界处，通信量只与批次大小及隐藏维度相关，对网络带宽的敏感度远低于 TP，因此适合跨物理节点连接。

### 2. 自回归解码中的流水线气泡（Bubble）

在自回归逐 Token 生成（Decode）阶段，模型必须拿到上一步生成的 Token 才能计算下一步。对于单请求或单 Batch：
1. GPU 0 计算完第 1 到 20 层后将激活值传给 GPU 1，随后 GPU 0 陷入等待；
2. 数据依次穿过 GPU 1、GPU 2；
3. GPU 3 算完并采样出下一个 Token 后，必须把该 Token 传回 GPU 0 才能开启下一轮计算。

在阶段数 $P=4$ 且只有单请求时，任何时刻只有 1 张卡在计算，其余 3 张卡都处于空闲等待状态，空闲气泡占比达到 $\frac{P-1}{P} = 75\%$。PP 无法像 TP 那样通过算力切分来缩短单请求计算延迟，单 Token 解码延迟（ITL）不仅无法降低，还会额外累加跨阶段 P2P 通信耗时，且硬件计算资源浪费严重。具体时延指标定义可参见 [大模型推理中的首 Token 延迟和平均生成速度分别受什么影响？](/interview/model-deployment-interview/latency-throughput.md)。

可以通过将多个并发请求拆分成多个 Micro-batch 交错送入流水线来填补空闲，但这需要系统维持极高的并发请求量，并且各阶段需要同时维护多路请求的 KV Cache，调度与显存管理更为复杂。

因此在实际推理中，**能用单机 TP 解决的，尽量不引入 PP**；PP 主要用于容纳超大规模模型的保底方案。

---

## 数据并行（DP）：无通信开销的横向扩容

当单机实例已经能跑起模型且延迟符合要求，但请求吞吐量（QPS）不断增加时，应优先采用数据并行。

![Data Parallelism 独立副本与调度网关架构](/images/content-rebuild/5d50c09d88bbcc79/fig-4.png)

### 1. 推理 DP 与训练 DP 的区别

在训练中，DP 在每步反向传播都需要对梯度执行 All-Reduce。而在推理中：
* **零卡间通信**：推理阶段模型权重是只读的，没有反向传播，没有梯度同步。
* **实例完全独立**：每个 DP 副本维护完整的模型参数和独立的上下文显存池。
* **请求解耦**：前端负载均衡网关将不同的请求分发给各个副本，副本之间互不干扰。

### 2. 配合 PagedAttention 的扩展模式

在现代推理引擎（如 vLLM）中：
1. 每个 DP 实例独立管理自己的 PagedAttention 显存池，具体显存计算可参考 [KV Cache 在模型部署中的显存空间](/interview/model-deployment-interview/kv-cache-deploy.md)。
2. 网关依据各副本的实时队列长度、剩余 KV 显存块等指标动态调度流量。
3. 因为副本间无通信开销，DP 的吞吐量增长与副本数量几乎呈线性关系，代价是每个副本都需要消耗一套独立的权重显存。

---

## 硬件互联约束：为什么 TP 绝不建议跨机？

在架构设计中，有时会有人提出“既然单机 8 卡算力有限，为什么不用 2 台机器跑 TP=16 来继续降低延迟？”。在实际部署中，跨机 TP 往往会导致性能严重下滑甚至负优化。

![机内 NVLink 与跨机以太网在 TP 下的延迟断崖对比](/images/content-rebuild/5d50c09d88bbcc79/fig-5.png)

### 1. 互联带宽与传输时延差异

* **机内 NVLink**：NVIDIA HGX 架构中，卡间双向带宽可达 900 GB/s，内部 NVSwitch 硬件延迟通常在微秒级别。
* **跨机网络**：双节点之间若使用 400 Gbps（约 50 GB/s）的 RDMA 网络，有效带宽也仅有机内 NVLink 的几分之一；若走普通以太网，带宽差距在几十倍以上。
* **网络基础时延（RTT）**：跨节点数据需要经过 PCIe、网卡驱动、网络交换机再到对端，小包通信的基础往返时延（RTT）比机内总线高出 1 到 2 个数量级。

### 2. 算例验证：Decode 阶段的小包通信瓶颈

以 Llama-3-70B（80 层 Transformer，隐藏维度 $H=8192$）在批大小 $B=1$ 时生成单个 Token 为例：

* **通信频率**：每层 2 次 All-Reduce，单步生成需要进行：
  $$80 \times 2 = 160 \text{ 次 All-Reduce}$$
* **单次传输数据量**：
  张量大小为 $1 \times 1 \times 8192$。在 FP16 精度下：
  $$8192 \times 2 \text{ 字节} = 16 \text{ KB}$$

这是一个**高频次、极小数据包**的场景。决定小包通信耗时的主要因素不是网络带宽极限，而是**每次调用的固定网络延迟与协议栈开销**：

1. **单机 NVLink 环境下（TP=8）**：  
   单次 16 KB 的 All-Reduce 借助硬件 NVSwitch 和 CUDA Graph 优化，延迟通常在 $2 \sim 4\ \mu\text{s}$。  
   160 次通信的总耗时约为 $160 \times 3\ \mu\text{s} \approx 0.48\text{ ms}$。与单步模型计算耗时（约数十毫秒）相比，通信开销很小，TP 能够显著加速。
2. **跨机网络环境下（跨机 TP=16）**：  
   跨机小包通信受网卡及交换机转发延迟影响，单次往往在数十微秒（如 $30 \sim 50\ \mu\text{s}$ 甚至更高）。  
   160 次累加的通信延迟可达：
   $$160 \times 40\ \mu\text{s} = 6.4\text{ ms}$$
   通信耗时甚至会超过单步矩阵计算的时间，GPU 算力长期处于跨机通信等待中，逐 Token 延迟大幅增加。

**结论**：TP 的切分上限应以单机内物理高速互联（如 NVLink）的卡数为界（常见为 TP=2、4、8），不宜跨越物理机箱。

---

## 拓展并行概念：Context Parallel 与 Expert Parallel

随着长文本和 MoE（混合专家）架构的普及，TP/PP/DP 衍生出了补充模式。

![Context Parallelism 与 Expert Parallelism 切分机制对比](/images/content-rebuild/5d50c09d88bbcc79/fig-6.png)

### 1. Context Parallelism（CP，上下文并行）
* **面对的问题**：在 128k 或更大上下文的 Prefill 阶段，序列长度 $S$ 产生的中间激活值和 KV Cache 极大，单卡无法容纳。
* **切分方式**：沿**序列长度轴**切分数据，各卡处理序列中的一个片段。
* **通信机制**：在注意力计算中，通过环形传递（Ring Attention）或 All-to-All 方式轮转传递 $K$ 和 $V$ 块，使长序列的计算和传输互相重叠。

### 2. Expert Parallelism（EP，专家并行）
* **面对的问题**：在 MoE 模型（如 Mixtral、DeepSeek 系列）中，虽然参数量极大，但每个 Token 只激活少数几个专家（Top-K）。如果用 TP 切分所有专家，专家矩阵会切得过碎，降低 GPU 计算效率。
* **切分方式**：将不同的完整专家分配到不同 GPU 上。
* **通信机制**：在专家层前后执行 **All-to-All** 通信，根据 Router 的分配把 Token 发送到对应卡上计算，算完后再通信回收合并。

关于各策略在复杂部署系统中的并发吞吐链路调优，可参考 [如何优化大模型服务的并发、吞吐和响应速度？](/interview/model-deployment-interview/optimization.md)。

---

## 常见面试追问

### 追问 1：为什么 GQA 模型的 TP 度数必须能整除 KV 头数？
在 Megatron-LM 的列切分规则中，每个 GPU 分到独立的注意力头并在本地完成 Attention。在 GQA 架构下，KV 头数远小于 Query 头数（例如 Query 头数为 64，KV 头数仅为 8）。每张 GPU 必须分到整数个完整的 KV 头，因此 TP 的度数必须能整除 KV 头数。若强行设 TP=16，会导致部分卡分不到 KV 头，需要跨卡广播复制数据，破坏并行独立性。

### 追问 2：为什么流水线并行（PP）的第一级和最后一级显存占用更高？
* **Stage 0（第一级）**：额外承载了完整的词表 Embedding 矩阵（在大词表中参数量可达数 GB）。
* **Stage P-1（最后一级）**：额外承载了最终的 LayerNorm、LM Head 输出矩阵，以及生成 Logits 和采样（Sampling）所需的计算显存。
因此在工程中，往往会给首尾两级分配略少数量的 Transformer 层（例如中间级放 10 层，首尾级放 8 层），以保持多卡显存均衡。

### 追问 3：在显存充足的情况下，想提高系统 QPS，应该调大 TP 还是增加 DP 副本？
**优先增加 DP 副本**。增大 TP 只能降低单请求计算延迟，但每层都有 2 次 All-Reduce 通信同步开销，加速比不是 100% 线性的；而 DP 副本间在推理时没有任何通信，吞吐量随副本数增加保持线性增长。只要单实例的延迟符合业务要求，提升系统吞吐的最优策略是部署更多 DP 副本。

显存测算与部署规模的详细推导，可参考 [如何估算一个大模型部署需要多少显存？](/interview/model-deployment-interview/memory-estimation.md)。
