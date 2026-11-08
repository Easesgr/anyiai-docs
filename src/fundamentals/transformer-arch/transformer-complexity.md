---
title: Transformer 的计算复杂度为什么是瓶颈？
icon: mdi:speedometer
tag:
- Transformer架构
- 计算复杂度
- 性能优化
description: 拆解 Transformer 在 Prefill 与 Decode 阶段的浮点计算量与显存占用，理清长序列下二次方复杂度与访存带宽瓶颈的成因。
---

自注意力机制（Self-Attention）让模型具备了跨越任意距离捕捉上下文依赖的能力。但在实际工程中，当输入文本从几千 Token 增长到几万、几十万甚至上百万时，模型推理速度会急剧变慢，训练显存也极易耗尽。

长序列带来的开销增长，核心在于两部分限制：**算力墙（FLOPs 随序列长度呈二次方膨胀）**与**显存墙（中间激活值与 KV Cache 对显存容量和带宽的挤压）**。要看清这个瓶颈，我们需要把大模型的运行过程拆解到具体的矩阵乘法与显存搬运中。

![讲解员在坐标系前对比展示长序列下的计算量二次方陡峭曲线与显存红线](/images/content-rebuild/29e2bbdde1c6452f/cover.png)

---

## 一、 运行大模型的两个阶段：Prefill 与 Decode

分析计算复杂度前，首先需要区分 Decoder-only 大语言模型在推理时的两种不同状态：**Prefill（预填充）**与 **Decode（逐字解码）**。两者的输入规模不同，硬件瓶颈也截然不同。

1. **Prefill 阶段（处理用户提示词）**：
   - **输入**：用户输入的 Prompt，包含 $N$ 个 Token。这 $N$ 个 Token 是一次性全部送入模型的。
   - **计算特征**：由于提示词已知，所有位置可以并行完成矩阵乘法，一次性算出这 $N$ 个位置的 Query、Key 和 Value，并计算完整的因果注意力矩阵。
   - **硬件瓶颈**：这是典型的**计算密集型（Compute-bound）**任务，GPU 的张量计算核心全速运转，浮点计算量大小直接决定处理时长。
2. **Decode 阶段（逐个生成新 Token）**：
   - **输入**：模型每次只根据当前所有历史 Token 预测下一个新 Token。每一步的输入序列长度仅为 $1$。
   - **计算特征**：新生成的 Token 产生对应的 Query 向量（形状为 $1 \times d$），去和之前所有历史 Token 的 Key 进行内积。为了避免每一步都重新计算历史 Token 的 Key 和 Value，系统会在显存中保留之前所有步计算出的结果，这被称为 **KV Cache**。
   - **硬件瓶颈**：在单序列或中小并发批次下，每生成一个新 Token，矩阵乘法退化为向量-矩阵乘法，计算强度极低，GPU 必须频繁把前面成千上万个 Token 的 KV Cache 从片外显存（HBM）读取到片上缓存（SRAM）。因此，Decode 在典型部署中常表现为**显存带宽密集型（Memory-bandwidth bound）**任务（当并发批次极大时算力利用率才会明显回升）。

![Prefill阶段的矩阵全量并行计算与Decode阶段逐Token增量计算的时序机制对比](/images/content-rebuild/29e2bbdde1c6452f/fig-1.png)

---

## 二、 单层注意力浮点计算量（FLOPs）推导

在浮点运算计量中，一次乘加操作（Multiply-Accumulate, $a \times b + c$）包含一次浮点乘法与一次浮点加法，业界通常约定记为 **2 个 FLOPs**。因此，形状为 $(m \times k)$ 的矩阵与 $(k \times n)$ 的矩阵相乘，计算量记为 $2mkn$ FLOPs。

定义符号：
- $N$：输入序列长度（Token 数量）
- $d$：模型的隐藏层特征维度（$d_{\text{model}}$）
- $h$：注意力头的数量
- $d_k$：每个注意力头的向量维度，满足 $h \times d_k = d$

单层标准 Self-Attention 的前向计算步骤如下：

```text
输入矩阵 X (N, d)
  │
  ├──> 1. QKV 线性投影 ──> Q, K, V 形状各为 (N, d)
  │
  ├──> 2. 注意力打分乘法: QK^T ──> (h, N, N)
  │
  ├──> 3. 标量缩放与 Softmax 归一化 ──> Attention Scores (h, N, N)
  │
  ├──> 4. 加权求和乘法: Scores × V ──> (N, d)
  │
  └──> 5. 输出投影 Wo ──> 输出特征 (N, d)
```

以 Prefill 阶段的全量计算为例，单层注意力的浮点计算量拆解如下：

### 1. QKV 线性投影
输入特征 $X \in \mathbb{R}^{N \times d}$ 分别乘以三个权重矩阵 $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$：
- 单个矩阵乘法尺寸为 $(N \times d) \times (d \times d)$，计算量为 $2 \times N \times d \times d = 2Nd^2$ FLOPs。
- 三个投影矩阵总计算量：
  $$\text{FLOPs}_{\text{proj}} = 3 \times 2Nd^2 = 6Nd^2$$

### 2. 注意力矩阵计算（$Q K^T$）
将 $Q$ 和 $K$ 拆分成 $h$ 个头，每个头的查询和键为 $Q_i, K_i \in \mathbb{R}^{N \times d_k}$。每个头计算打分矩阵 $S_i = Q_i K_i^T \in \mathbb{R}^{N \times N}$：
- 单个头的乘法尺寸为 $(N \times d_k) \times (d_k \times N)$，计算量为 $2 \times N \times d_k \times N = 2N^2 d_k$ FLOPs。
- $h$ 个头相加的总计算量：
  $$\text{FLOPs}_{QK^T} = h \times (2N^2 d_k) = 2N^2 (h \cdot d_k) = 2N^2 d$$

### 3. Softmax 归一化与缩放
对 $h$ 个大小为 $N \times N$ 的打分矩阵进行除以 $\sqrt{d_k}$ 并按行做 Softmax。这属于逐元素与行级规约运算，总运算量量级为 $O(h N^2)$。由于不含维度因子 $d$，在整体 FLOPs 统计中远小于矩阵乘法项，因此主要算力通常由矩阵乘法主导。

### 4. 值向量加权求和（$\text{Scores} \times V$）
每个头的注意力分数 $A_i \in \mathbb{R}^{N \times N}$ 与对应的值矩阵 $V_i \in \mathbb{R}^{N \times d_k}$ 相乘，得到输出 $O_i \in \mathbb{R}^{N \times d_k}$：
- 单个头的乘法尺寸为 $(N \times N) \times (N \times d_k)$，计算量为 $2 \times N \times N \times d_k = 2N^2 d_k$ FLOPs。
- $h$ 个头相加的总计算量：
  $$\text{FLOPs}_{AV} = h \times (2N^2 d_k) = 2N^2 d$$

### 5. 输出投影（Output Projection）
拼接后的多头结果 $\in \mathbb{R}^{N \times d}$ 乘以权重矩阵 $W_O \in \mathbb{R}^{d \times d}$ 还原为隐层向量：
- 矩阵乘法尺寸为 $(N \times d) \times (d \times d)$，计算量：
  $$\text{FLOPs}_{\text{out}} = 2Nd^2$$

将上述矩阵乘法项相加，单层 Self-Attention 的主要浮点计算量为：
$$\text{FLOPs}_{\text{Attention}} = (6Nd^2 + 2Nd^2) + (2N^2 d + 2N^2 d) = 8Nd^2 + 4N^2 d$$

![Self-Attention各子步骤输入输出张量维度与浮点计算量FLOPs逐步拆解](/images/content-rebuild/29e2bbdde1c6452f/fig-2.png)

公式清楚地分成了两项：
- **线性项 $8Nd^2$**：来自输入和输出的线性投影，与序列长度 $N$ 成线性比例，与模型维度平方 $d^2$ 成正比。
- **二次项 $4N^2 d$**：来自 Token 之间的两两计算（$QK^T$ 和 $AV$），与序列长度的平方 $N^2$ 成正比。

正是这项 $4N^2 d$，决定了长文本场景下的计算代价。

<ReadMoreLock />

---

## 三、 注意力与 FFN 的开销对比：临界点在哪？

很多初学者会产生误区：“既然注意力包含 $O(N^2)$ 计算，那它在任何时候都是模型中计算耗时最多的部分。”

实际上并非如此。在序列较短时，前馈神经网络（FFN）的计算开销远大于注意力。

### 1. FFN 的计算量
以标准 Transformer 结构为例，FFN 由两层全连接组成，中间维度通常扩展为 $d_{\text{ff}} = 4d$：
1. 升维层：$X W_1$，维度从 $(N \times d)$ 乘到 $(d \times 4d)$，计算量为 $2 \times N \times d \times 4d = 8Nd^2$ FLOPs。
2. 降维层：$H W_2$，维度从 $(N \times 4d)$ 乘到 $(4d \times d)$，计算量为 $2 \times N \times 4d \times d = 8Nd^2$ FLOPs。

两层相加，标准 FFN 的浮点计算量为：
$$\text{FLOPs}_{\text{FFN}} = 8Nd^2 + 8Nd^2 = 16Nd^2$$

（若采用现代大模型常用的 SwiGLU 结构，通常包含 3 个权重矩阵，中间维度设为 $\frac{8}{3}d$，总计算量同样约为 $3 \times 2 \times N \times d \times (\frac{8}{3}d) = 16Nd^2$。）

### 2. 算力开销的交叉临界点
在单层 Transformer 中，各计算项可以归结为两类：
- **与 $d^2$ 成正比的线性项**（线性投影与 FFN 矩阵乘法）：
  $$\text{FLOPs}_{\text{linear}} = 8Nd^2 (\text{投影}) + 16Nd^2 (\text{FFN}) = 24Nd^2$$
- **与 $N^2$ 成正比的二次项**（注意力打分与值汇聚）：
  $$\text{FLOPs}_{\text{quadratic}} = 4N^2 d$$

如果对比整个 Transformer 层中二次项与线性项的总量，令 $4N^2 d = 24Nd^2$，解得：
$$N = 6d$$

如果只单独对比 FFN 与注意力的两两交互项（$4N^2 d = 16Nd^2$），解得：
$$N = 4d$$

![不同序列长度下Attention二次方开销与FFN线性开销的临界交叉点演变](/images/content-rebuild/29e2bbdde1c6452f/fig-3.png)

这一比较指出了真实的工程现象：
1. **短序列场景（$N < 4d$）**：
   以隐层维度 $d = 4096$ 的模型为例，临界点 $4d = 16,384$。如果输入长度只有 $2048$ 或 $4096$，序列长度远未达到临界点。此时整个模型计算量的 80% 以上都花在 FFN 和投影矩阵中。在这个区间内，优化注意力算力带来的全流程收益并不显著。
2. **长序列场景（$N \gg 6d$）**：
   当上下文长度扩展到 $32k$、$64k$ 甚至 $128k$ 时，$N$ 迅速超越 $6d$。由于二次项以 $N^2$ 的速度上升，注意力的计算量会快速反超 FFN，成为消耗计算时间的主要瓶颈。

---

## 四、 显存瓶颈：训练与推理的机制差异

算力决定了运算需要多少时间，而显存容量决定了程序能不能跑起来。在实际运行中，显存往往比算力更早遇到物理硬上限。训练显存与推理显存的占用机制完全不同：

### 1. 训练期：反向传播与中间激活值
训练模型时，显存由四部分组成：静态模型参数、优化器状态（如 AdamW 的动量）、梯度，以及前向传播保留的**中间激活值（Activation Memory）**。

在标准反向传播中，为了求出参数的梯度，必须在显存中保留前向传播计算出的中间张量。对于注意力机制，单层就需要保存完整的注意力矩阵：
$$A = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times h \times N \times N}$$

假设批大小 $B=1$，头数 $h=32$，数据精度为 16 位浮点数（FP16/BF16，每元素占 2 字节）：
- 当 $N = 2,048$ 时：
  单层显存为 $1 \times 32 \times 2048 \times 2048 \times 2 \text{ 字节} = 268,435,456 \text{ 字节} = 256 \text{ MiB}$。
- 当 $N = 32,768$（长度扩大 16 倍）时：
  由于显存与 $N^2$ 成正比，单层占用将扩大 $16^2 = 256$ 倍：
  $$256 \text{ MiB} \times 256 = 65,536 \text{ MiB} = 64 \text{ GiB}$$

对于一个 32 层的模型，光是把这部分注意力矩阵保留在显存中就需要 $32 \times 64 \text{ GiB} = 2048 \text{ GiB}$ 显存。没有任何单张 GPU 可以承受，程序会直接报错 `CUDA Out of Memory`。

![训练期间全量保存N×N注意力图导致的显存激活值爆炸与OOM报错场景](/images/content-rebuild/29e2bbdde1c6452f/fig-4.png)

### 2. 推理期：KV Cache 的线性增长与并发压力
推理时不需要反向传播，因此**不需要显式存储 $N \times N$ 的激活矩阵**。

但推理有其特有的显存消耗源：**KV Cache**。在自回归生成时，模型必须保留前面生成的所有 Token 的 Key 和 Value 向量。

每个 Token 在每一层模型中需要存储：
- Key 向量：$d$ 个数值
- Value 向量：$d$ 个数值

以半精度（2 字节）计算，模型层数为 $L$，并发请求数（Batch Size）为 $B$，序列长度达到 $N$ 时，KV Cache 占用的显存为：
$$\text{Memory}_{\text{KVCache}} = 2 \times (d + d) \times L \times B \times N = 4 B L d N \text{ 字节}$$

对于单次请求，KV Cache 随着序列长度 $N$ 呈**线性增长（$O(N)$）**，并不包含二次方项。但当多并发同时访问时，这个开销会被快速放大。

**算例：**
以 70B 模型为例（$L = 80, d = 8192$），采用标准多头注意力（MHA）：
- 每个 Token 产生的 KV Cache 显存：
  $$4 \times 80 \times 8192 = 2,621,440 \text{ 字节} = 2.5 \text{ MiB}$$
- 当服务 $B = 8$ 个并发用户、每个请求上下文达到 $N = 32,768$ 时：
  $$\text{总显存} = 2.5 \text{ MiB} \times 8 \times 32,768 = 655,360 \text{ MiB} = 640 \text{ GiB}$$

仅保存推理过程中的 KV Cache，就需要 8 张 80GB 的 GPU。这尚未计算模型参数本身的显存占用。

![推理期KV Cache随自回归步数线性累加占用的显存空间与多并发挤占模型](/images/content-rebuild/29e2bbdde1c6452f/fig-5.png)

---

## 五、 物理现实：算力墙与显存带宽墙

长上下文在实际部署中遇到的核心矛盾，可以归纳为两个物理限制：

### 1. 首字延迟（TTFT）与算力墙
在 Prefill 阶段，算力开销中包含了 $4N^2 d$。序列长度翻 16 倍，这部分计算量膨胀 256 倍。

面对几十万 Token 的输入提示词，哪怕 GPU 显存放得下，计算核心也需要更长的物理时间才能完成这组矩阵运算。这直接导致首字返回时间（Time to First Token, TTFT）显著延长。

### 2. 生成吞吐与显存带宽墙
在自回归 Decode 阶段，生成瓶颈从“矩阵计算”转向了“数据读取”。

GPU 计算核心的处理能力极高，但片外显存（HBM）向片上计算单元（SRAM）传输数据的总线带宽是有限的。计算强度（每搬运 1 字节显存所能执行的浮点运算次数）决定了硬件能否被充分利用。

- 在单请求自回归 Decode 阶段，每步仅输入 1 个 Token，计算强度极低。
- 算力核心做完微秒级的矩阵乘法后，必须等待下一次庞大的 KV Cache 重新从显存读入缓存。
- 序列越长，每次生成新 Token 需要从显存中搬运的 KV Cache 越多，Token 之间的生成间隔（Inter-Token Latency, ITL）就越长。

![上下文窗口扩展时算力受限与带宽瓶颈导致的推理延迟激增与失效权衡](/images/content-rebuild/29e2bbdde1c6452f/fig-6.png)

---

## 六、 总结与工程改进方向

Transformer 用无递归的全连接注意力，换取了全局感受野与训练并行化能力，但这一设计在序列延长时带来了严密的算力与显存代价：

| 维度 | 短序列（$N < 4d$） | 长序列（$N \gg 6d$） | 物理限制类型 |
| :--- | :--- | :--- | :--- |
| **Prefill 阶段算力** | FFN 与投影矩阵占主导（$O(Nd^2)$） | 注意力打分与求和占主导（$O(N^2d)$） | 计算密集（Compute-bound） |
| **训练期显存** | 静态权重与优化器状态占主导 | $N \times N$ 注意力激活图主导 | 容量瓶颈（易触发 OOM） |
| **Decode 阶段吞吐** | 计算与访存相对均衡 | 历史 KV Cache 显存搬运耗时主导 | 显存带宽受限（Memory-bandwidth bound） |

针对这些瓶颈，后续技术路线发展出了明确的优化方向：
- **针对训练期激活值显存爆炸**：开发了 **FlashAttention** 等算法。通过利用 GPU 片上 SRAM 进行分块计算与在线重计算，避免把完整的 $N \times N$ 中间矩阵写回慢速片外显存。这显著降低了显存读写延迟和空间占用（注意：它并不改变标准注意力本身的理论 FLOPs 复杂度）。
- **针对推理期 KV Cache 显存与访存压力**：采用 **Multi-Query Attention (MQA)** 或 **Grouped-Query Attention (GQA)**，让多个 Query 头共享同一组 Key/Value 头，直接将 KV Cache 显存和带宽搬运量减少数倍到数十倍；同时配合 **PagedAttention** 技术解决显存碎片化问题。

理解了 $O(N^2 d)$ 与 $O(N d^2)$ 的消长规律以及显存瓶颈的成因，就能更准确地评估长文本模型的推理成本与架构选型。
