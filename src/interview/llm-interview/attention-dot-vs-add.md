---
title: Transformer 计算注意力时用点乘还是加法？为什么？
icon: mdi:multiplication
tag:
- Transformer
- Attention
- 点积
description: 解析 Transformer 选择缩放点积注意力的原因，对比加法注意力的张量广播与硬件执行差异，推导缩放因子对梯度的稳定作用与面试要点。
---

在深度学习面试中，“Transformer 为什么选择缩放点积注意力（Scaled Dot-Product Attention）而不是加法注意力（Additive Attention）”是一道高频考题。这道题既考察对注意力计算过程的掌握，也考察对现代 GPU 硬件特性与训练稳定性的理解。

![Transformer 缩放点积注意力与加法注意力对比封面](/images/content-rebuild/9afa961f846bdeae/cover.png)

---

## 面试速答（30 秒口述）

面试时可以按以下两点简明作答：

> 1. **计算速度与显存开销**：理论上加法注意力与点积注意力的渐近计算复杂度相同（均为 $\mathcal{O}(n \cdot m \cdot d)$）。但点积注意力是标准的矩阵乘法（GEMM），能充分调用 GPU 的矩阵计算单元（如 Tensor Core），显存复用率高；加法注意力需要把 Query 和 Key 广播扩展为三维张量相加，并执行逐元素非线性激活（如 $\tanh$），显存读写量大，硬件执行速度显著落后。
> 2. **数值稳定性与缩放因子**：未经缩放的点乘在向量维度 $d_k$ 较大时，内积方差会随维度线性放大，容易导致 Softmax 进入梯度极小的平坦区；Transformer 引入了除以 $\sqrt{d_k}$ 的缩放因子，将方差拉回 1，既保留了矩阵乘法的高速度，又解决了大维度下训练不稳定的问题。

---

## 贯穿算例：加法与点积到底怎么算？

为了直观看到两种注意力的张量变化，我们先用一组极小的数据走一遍完整的计算流程。

假设输入如下：
- 序列长度：Query 数量 $n = 2$，Key 数量 $m = 2$；
- 向量维度：$d_k = d_a = 2$；
- 假定线性投影已经完成，Query 矩阵 $Q$ 与 Key 矩阵 $K$ 分别为：
  $$Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad K = \begin{pmatrix} 1 & 1 \\ 0 & 2 \end{pmatrix}$$

### 1. 加法注意力怎么算？

加法注意力（源自 Bahdanau 等人在 2014 年提出的 Seq2Seq 模型）的打分公式为：

$$\text{score}_{\text{add}}(q, k) = v_a^\top \tanh(W_q q + W_k k)$$

为了展示纯加法聚合过程，我们假设线性投影已包含在 $Q$ 和 $K$ 中，权重向量设为 $v_a = [1, 1]^\top$：

1. **广播扩展**：$Q$ 形状为 $(2, 2)$，$K$ 形状为 $(2, 2)$。两者不能直接相加，需要把 $Q$ 扩展为 $(2, 1, 2)$，把 $K$ 扩展为 $(1, 2, 2)$。
2. **三维张量相加**：在显存中相加生成形状为 $(2, 2, 2)$ 的三维中间张量：
   - 位置 $(0, 0)$：$[1, 0] + [1, 1] = [2, 1]$
   - 位置 $(0, 1)$：$[1, 0] + [0, 2] = [1, 2]$
   - 位置 $(1, 0)$：$[0, 1] + [1, 1] = [1, 2]$
   - 位置 $(1, 1)$：$[0, 1] + [0, 2] = [0, 3]$
3. **逐元素计算 $\tanh$**：对该三维张量内的 8 个浮点数逐个计算双曲正切。
4. **向量点乘降维**：将激活后的张量与向量 $v_a$ 进行内积，缩减回形状为 $(2, 2)$ 的二维得分矩阵。

![加法注意力张量计算与广播数据流](/images/content-rebuild/9afa961f846bdeae/fig-1.png)

### 2. 缩放点积注意力怎么算？

Transformer 采用的点积注意力省去了广播张量与非线性映射，直接进行矩阵乘法：

$$\text{score}_{\text{dot}}(Q, K) = \frac{Q K^\top}{\sqrt{d_k}}$$

代入具体数值：

1. **矩阵乘法**：
   $$Q K^\top = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 1 & 2 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 1 & 2 \end{pmatrix}$$
   直接得到形状为 $(2, 2)$ 的二维矩阵，没有生成任何三维中间张量。
2. **标量缩放**：维度 $d_k = 2$，缩放因子为 $\sqrt{2} \approx 1.414$：
   $$\frac{Q K^\top}{\sqrt{2}} = \begin{pmatrix} 1/1.414 & 0/1.414 \\ 1/1.414 & 2/1.414 \end{pmatrix} \approx \begin{pmatrix} 0.707 & 0 \\ 0.707 & 1.414 \end{pmatrix}$$
3. **Softmax 归一化**：按行计算 Softmax，直接得到注意力权重。

![点积注意力矩阵乘法数据流](/images/content-rebuild/9afa961f846bdeae/fig-2.png)

### 计算步骤直接对比

| 比较项 | 加法注意力 | 缩放点积注意力 |
| :--- | :--- | :--- |
| **中间数据形态** | 必须在显存中广播展开为 $(n, m, d)$ 三维张量 | 直接通过 GEMM 产出 $(n, m)$ 二维矩阵 |
| **非线性操作** | 对三维张量每个元素执行 $\tanh$ 计算 | 无内部非线性，仅最后做一次 Softmax |
| **降维方式** | 三维张量与权重向量 $v_a$ 点乘降维 | 矩阵乘法原生完成内积求和 |
| **缩放机制** | 依赖 $\tanh$ 自身有界性限制数值 | 显式除以标量 $\sqrt{d_k}$ 修正方差 |

<ReadMoreLock />

---

## 硬件执行：为什么点积在 GPU 上更快？

加法注意力只有加法和激活，直觉上似乎比点积乘法更轻量。但在现代 GPU 架构下，决定耗时的关键往往不是总算术次数，而是**访存瓶颈（Memory Bandwidth Bound）**与**矩阵计算单元利用率**。

![GPU 硬件 GEMM 与逐元素访存瓶颈对比](/images/content-rebuild/9afa961f846bdeae/fig-3.png)

### 1. 算术强度与显存带宽

算法的实际运行速度受“算术强度 = 计算量 (FLOPs) / 访存量 (Bytes)”约束：

- **加法注意力属于访存密集型（Memory-Bound）**：广播加法与 $\tanh$ 是逐元素（Element-wise）操作。GPU 需要先从全局显存（HBM）读取 Query 和 Key，写入膨胀后的三维中间张量 $(n, m, d_a)$，再读入中间张量送入非线性单元计算 $\tanh$，最后写回。每次运算伴随着大量的数据搬运，计算核心经常处于“等待数据从显存载入”的饥饿状态。
- **点积注意力属于计算密集型（Compute-Bound）**：$Q K^\top$ 是标准的稠密通用矩阵乘法（GEMM）。在计算矩阵乘法时，GPU 可以将数据分块载入片上高速共享内存（SRAM），同一批数据在片上反复复用参与多次乘加运算，显存读写开销显著降低。

### 2. 专用矩阵加速单元（Tensor Core）

现代深度学习硬件（如 NVIDIA GPU）内置了专门用于矩阵乘加的硬件加速单元（Tensor Core）：

- Tensor Core 原生支持单指令多数据的高速矩阵乘累加（MMA 指令）。在半精度（FP16/BF16）下，Tensor Core 的计算吞吐量通常是普通矢量 CUDA Core 的数倍；
- 点积注意力由高度优化的底库（如 cuBLAS、CUTLASS）直接调用 Tensor Core 执行；
- 加法注意力中的 $\tanh$ 属于特殊函数，必须由专门的特殊函数单元（SFU）或常规矢量单元逐个计算，无法利用矩阵硬件单元。

### 3. 中间激活显存占用

加法注意力的中间张量尺寸为 $\mathcal{O}(n \cdot m \cdot d)$。当序列长度 $n, m$ 达到上千时，存储三维张量会占用数倍于注意力矩阵的显存。而点积注意力直接输出 $(n \cdot m)$ 矩阵，省去了中间特征维度的显存驻留。

---

## 为什么必须除以 $\sqrt{d_k}$？维度放大与梯度饱和

点积在硬件执行上具有优势，但在数学上有一个天然缺陷：**点积结果的数值方差会随着向量维度 $d_k$ 线性增长**。如果不对其进行约束，大模型在训练初期就会崩溃。

![缩放因子方差修正与 Softmax 饱和机制](/images/content-rebuild/9afa961f846bdeae/fig-4.png)

### 1. 方差放大推导

我们来推导为什么点积方差会膨胀。

**假设前提**：假定向量 $q$ 与 $k$ 的各个分量 $q_i, k_i$ 是相互独立、均值为 0、方差为 1 的随机变量（在模型初始化阶段，这与常见参数初始化假设一致）：

$$\mathbb{E}[q_i] = 0, \quad \text{Var}(q_i) = 1$$
$$\mathbb{E}[k_i] = 0, \quad \text{Var}(k_i) = 1$$

两者的点积结果为 $z = q^\top k = \sum_{i=1}^{d_k} q_i k_i$：

1. 每个分量乘积的期望：
   $$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$
2. 每个分量乘积的方差：
   $$\text{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] - 0 = 1 \times 1 = 1$$
3. $d_k$ 个独立分量累加的总方差：
   $$\text{Var}(z) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

点积 $z$ 的方差为 $d_k$，标准差为 $\sqrt{d_k}$。当向量维度从较小的 64 提升到 128 甚至更高时，点积数值的绝对值范围会被成倍拉大。

### 2. Softmax 饱和与梯度消失

Softmax 计算公式为：

$$S_i = \frac{e^{z_i}}{\sum_{j=1}^m e^{z_j}}$$

其偏导数为：

$$\frac{\partial S_i}{\partial z_j} = S_i (\delta_{ij} - S_j)$$

当输入向量的分量方差极大时，绝对值较大的那一项经过指数运算 $e^{z_i}$ 之后，会占据绝对主导地位，导致 Softmax 的输出极度逼近 One-Hot 分布（某一位置接近 1，其余接近 0）：
- 当 $S_i \approx 1$ 时，$\delta_{ii} - S_i \approx 1 - 1 = 0$；
- 当 $S_i \approx 0$ 时，$S_i (\delta_{ij} - S_j) \approx 0 \times \dots = 0$。

这会导致反向传播时的偏导数几乎全为 0，模型进入**Softmax 梯度饱和区**，权重参数无法正常更新。

### 3. 缩放因子的修正效果

除以 $\sqrt{d_k}$ 之后：

$$z_{\text{scaled}} = \frac{z}{\sqrt{d_k}}$$
$$\text{Var}(z_{\text{scaled}}) = \text{Var}\left(\frac{z}{\sqrt{d_k}}\right) = \frac{1}{d_k} \text{Var}(z) = \frac{d_k}{d_k} = 1$$

无论隐藏层维度 $d_k$ 设为多大，缩放后的得分方差始终被规范在 1 附近。Softmax 的输入被保持在具有平缓梯度的灵敏区间内，保证了梯度能够在深层网络中稳定回传。

---

## 面试追问与边界考查

![面试高频追问与极限维度表现对比](/images/content-rebuild/9afa961f846bdeae/fig-5.png)

### 追问 1：如果完全不缩放，加法注意力和点积注意力谁更稳定？

**答案**：加法注意力更稳定。

**原因**：加法注意力公式为 $v_a^\top \tanh(W_q q + W_k k)$。中间的 $\tanh$ 激活函数输出严格限制在 $(-1, 1)$ 区间内，相当于自然设置了数值截断屏障。即使维度升高，进入最终内积的值也是有界的，数值不会无限发散。原始论文《Attention Is All You Need》在第 3.2.1 节明确指出：在小维度下两种注意力的表现相近，但在大维度下加法注意力明显优于未经缩放的点积注意力；正是因为引入了 $\frac{1}{\sqrt{d_k}}$ 缩放，点积注意力才在大维度下兼顾了高速度与高稳定性。

### 追问 2：在多头注意力（MHA）下，两者的实现复杂度差异有多大？

![多头注意力拼接与批处理效率对比](/images/content-rebuild/9afa961f846bdeae/fig-6.png)

**答案**：点积注意力能直接打包为批处理矩阵乘法（Batched GEMM），而加法注意力工程实现极其冗长。

**原因**：在多头注意力机制中，点积注意力的所有头可以通过一次线性映射投影为 $(B, n, H, d_k)$，转置合并为 $(B \times H, n, d_k)$ 后，只需调用单次批处理矩阵乘法（Batched GEMM Kernel）即可同时计算全部注意力头。而加法注意力若每个头有独立的非线性映射与偏置，会导致多组循环计算和频繁的 Kernel 启动调用，难以实现高吞吐的多头合并并行。

### 追问 3：类似 FlashAttention 这类融合算子技术，能让加法注意力追平点积吗？

**答案**：不能从根本上追平。

**原因**：FlashAttention 的核心收益是利用分块（Tiling）算法减少注意力矩阵从 HBM 到片上 SRAM 的读写次数，但它底层依赖的依然是 Tensor Core 的密集矩阵乘法单元。加法注意力包含逐元素加法和 $\tanh$ 激活，即便通过算子融合避免写回显存，片上寄存器开销大、缺乏专用矩阵计算单元硬件加速的问题依然存在，无法在硬件执行效率上反超点积矩阵乘。

---

## 常见误区与避坑指南

- ❌ **误区 1：“加法注意力的渐近时间复杂度比点积注意力高一个阶”**  
  **纠偏**：从大 $\mathcal{O}$ 阶数来看，加法与点积注意力的理论计算复杂度都是 $\mathcal{O}(n \cdot m \cdot d)$。点积的优势在于**硬件实际运行时间（Wall-clock Time）**极短，属于常数项和硬件适配带来的巨大工程优势，而非渐进复杂度存在阶数差距。
- ❌ **误区 2：“除以 $\sqrt{d_k}$ 限制了注意力权重变得尖锐”**  
  **纠偏**：缩放因子是在初始化假设下避免方差膨胀引发的初态饱和。在模型训练过程中，当某些 Key 与 Query 真正高度相关时，其内积值依然会很大，注意力权重依然能够学出高度集中的尖锐分布。
- ❌ **误区 3：“点积注意力完全不存在显存瓶颈”**  
  **纠偏**：点积虽然避免了 $(n, m, d)$ 的三维中间特征张量，但其最终生成的 $(n, m)$ 注意力矩阵在长序列下显存开销依然呈平方级增长（这也是需要后续 FlashAttention 等显存优化技术的原因）。

---

## 知识关联与延伸阅读

1. **前置机制**：
   - [Self-Attention 中的 K 和 Q 是用来做什么的？](/interview/llm-interview/self-attention-qk.md)
   - [Self-Attention 中 Q、K、V 分别是什么意思？](/interview/llm-interview/attention-qkv.md)
2. **进阶对比**：
   - [为什么 Transformer 采用多头注意力机制？](/interview/llm-interview/why-multi-head-attention.md)
   - [Multi-Head Attention、MQA 和 GQA 有什么区别？](/interview/llm-interview/mha-mqa-gqa.md)
   - [Attention 机制到底在算什么？](/fundamentals/dl-basics/attention-mechanism.md)
