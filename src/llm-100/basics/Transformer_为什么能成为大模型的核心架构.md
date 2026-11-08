---
title: Transformer 为什么能成为大模型的核心架构？
icon: mdi:book-open-page-variant-outline
date: 2026-07-16
tag:
- 大模型100讲
- 基础篇
description: 通过具体语句算例，解析 Transformer 在训练并行度、长程信息通路与 GPU 矩阵乘法上的核心优势，并讲清推理阶段与计算复杂度的真实边界。
---

在深度学习序列建模的演进历程中，主流模型经历了从循环神经网络（RNN）到卷积神经网络（CNN），再到 Transformer 的转变。Transformer 能够取代此前的序列模型并成为今天大语言模型（LLM）的基础骨干，核心原因在于两个工程特性：**训练阶段的全序列高并行度**与**任意两个位置之间的单步直接信息交互**。

本节我们将围绕一个具体的 8 词自然语言处理例子，对比 RNN、CNN 与 Transformer 的计算过程，解释为什么这种架构能充分释放现代 GPU 算力，以及它的计算边界在哪里。

![Transformer 架构演进：并行计算突破与长程信息通路封面图](/images/content-rebuild/a3ca58602f4d0ac1/cover.png)

---

## 贯穿算例：跨越 7 步的长程依赖

为了直观对比不同架构的信息传递机制，假设系统需要处理如下包含 8 个词的输入序列：

```text
位置索引：  0      1      2      3      4      5      6      7
Token 序列：银行   账户   里的   余额   变动   已经   实时   提醒
```

目标是让处于序列末尾（位置 7）的“提醒”，准确结合序列开头（位置 0）的“银行”建立上下文表征。

---

## RNN 的瓶颈：串行递推依赖与梯度衰减

循环神经网络（RNN，包括 LSTM 与 GRU）的基本设计假设是时序递推：每个时间步的隐藏状态 $h_t$，由当前输入的向量 $x_t$ 与上一个时间步的隐藏状态 $h_{t-1}$ 共同计算得出：

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

![RNN 时序依赖与串行训练瓶颈分解](/images/content-rebuild/a3ca58602f4d0ac1/fig-1.png)

### 1. 训练阶段的串行阻塞
在监督训练或语言建模训练时，虽然整句文本已经完整提供（即采用教师强制 Teacher Forcing），但在 RNN 的数学定义下，要计算第 7 步的隐藏状态 $h_7$，计算图必须先依次求出 $h_0, h_1, h_2, \dots, h_6$。这意味着：
- 序列维度存在强依赖链，计算深度正比于序列长度 $n$；
- 尽管可以在批次（Batch）维度并发多个句子，但在单句序列长度方向上无法展开并行计算；
- GPU 的大规模并行流式多处理器（SM）经常因为等待前一时间步的输出而处于低利用率状态，算术强度极低。

### 2. 远距离信息传递衰减
在反向传播通过时间（BPTT）过程中，误差梯度从 $h_7$ 传递至 $h_0$ 时，需要经过连续 7 次雅可比矩阵的链式连乘：

$$\frac{\partial h_7}{\partial h_0} = \prod_{k=1}^7 \frac{\partial h_k}{\partial h_{k-1}}$$

当特征值偏离 1 时，容易引发梯度消失或梯度爆炸。尽管 LSTM 和 GRU 引入了门控机制和加性状态通道以减缓这一问题，但将全部历史信息强制压缩进单个固定维度的向量 $h_t$ 中，在长序列建模时依然容易丢失上下文细节。关于梯度连乘推导的详细数学过程，可参阅 [/fundamentals/dl-basics/gradient-vanishing-exploding.md](/fundamentals/dl-basics/gradient-vanishing-exploding.md)。

---

## CNN 的折中：局部感受野与层级堆叠

为了打破 RNN 在序列维度上的串行计算锁，早期研究（如 ByteNet、ConvS2S）尝试使用一维卷积神经网络处理序列。卷积核在文本序列上滑动，整条序列的所有位置可以同时进行前向卷积，实现了训练时的时间维度并行。

但 CNN 面临的是**感受野（Receptive Field）受限**问题。

![CNN 序列建模中的局部感受野与层级堆叠金字塔](/images/content-rebuild/a3ca58602f4d0ac1/fig-2.png)

设一维卷积核大小为 $k=3$：
- 第 1 层卷积：输出节点的感受野仅为相邻的 3 个词（例如覆盖位置 [5, 6, 7]）；
- 第 2 层卷积：感受野扩张到 5 个词（覆盖位置 [3, 4, 5, 6, 7]）；
- 第 3 层卷积：感受野扩张到 7 个词；
- 必须堆叠到第 4 层卷积，位置 7 的特征才能感知到位置 0 的信息。

若两个词相距 $n$，标准卷积需要堆叠 $O(n/k)$ 层；若采用空洞卷积（Dilated Convolution），感受野随深度呈指数增长，最少也需要堆叠 $O(\log_k n)$ 层。信息在跨层传递中伴随多重非线性变换与加权平均，远距离交互路径依然较长。关于卷积感受野的推导细节可参考 [/fundamentals/dl-basics/cnn.md](/fundamentals/dl-basics/cnn.md)。

<ReadMoreLock />

---

## Transformer 的自注意力拓扑：单步直接交互

Transformer 完全舍弃了循环递推与局部滑动卷积，改用自注意力（Self-Attention）机制。

![Transformer 全连通拓扑图与 O(1) 交互通路对比](/images/content-rebuild/a3ca58602f4d0ac1/fig-3.png)

### 1. 任意两点的交互路径为 1 步
在贯穿例子中，模型首先通过三个投影矩阵将输入序列映射为 Query（查询）、Key（键）、Value（值）三个矩阵：$Q, K, V \in \mathbb{R}^{n \times d_k}$。

计算注意力打分矩阵：

$$S = \frac{Q K^T}{\sqrt{d_k}}, \quad P = \mathrm{softmax}(S)$$

在注意力矩阵中，第 7 行第 0 列的分数直接由向量内积给出：

$$S_{7,0} = \frac{q_7 \cdot k_0^T}{\sqrt{d_k}}$$

接着经过 Softmax 归一化后，位置 7 的输出直接包含了 $v_0$ 的加权分量。无论序列长度是 8 还是 8192，任意两个位置之间的关联权重均在同一次矩阵内积中计算出来，两点间信息交互的路径长度在逻辑上恒为 $O(1)$。

### 2. 三种架构在序列建模上的指标对比

设序列长度为 $n$，表征维度为 $d$，卷积核尺寸为 $k$：

| 架构体系 | 单层时间计算复杂度 | 训练时串行操作数 | 任意两点间最大信息路径长度 |
| :--- | :--- | :--- | :--- |
| **循环神经网络 (RNN)** | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| **一维标准卷积 (CNN)** | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(n/k)$ |
| **空洞卷积 (Dilated CNN)** | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| **自注意力机制 (Self-Attention)** | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |

注：自注意力单层计算复杂度中包含 $O(n^2)$ 项，这是由于需要计算全序列两两成对的内积矩阵所致。

---

## 硬件契合度：利用 GEMM 跑满张量核心

现代深度学习加速硬件（如 GPU Tensor Core 或 TPU）的处理强项是**通用矩阵乘法（GEMM, General Matrix Multiply）**。

![GPU 张量核心高吞吐 GEMM 运算场景分解](/images/content-rebuild/a3ca58602f4d0ac1/fig-4.png)

### 1. RNN 的访存受限（Memory-Bound）问题
在 RNN 中，每推进一个时间步，硬件必须从高带宽显存（HBM）中读取参数矩阵 $W_{hh}$，并与当前单个向量 $h_{t-1}$ 相乘。计算与访存的比例（算术强度）极低，计算单元在纳秒级完成运算后，大部分时间处于等待从显存中搬运权重的状态，受制于显存带宽限制。

### 2. Transformer 的计算密集（Compute-Bound）优势
Transformer 将输入整条序列拼成维度为 $(B \times n, d)$ 的矩阵（$B$ 为批次大小）：
- $Q, K, V$ 投影是两个大稠密矩阵的乘法；
- 注意力打分 $Q K^T$ 是 $(B, n, d)$ 与 $(B, d, n)$ 的批量矩阵乘法；
- 前馈网络（FFN）包含大尺度的线性映射（通常从 $d$ 扩展到 $4d$ 再收缩回 $d$）。

权重矩阵加载到片上高速缓存（SRAM/Cache）后，可以与成千上万个 Token 的向量连续进行乘加运算，算术强度（FLOPs/Byte）高，能充分发挥张量计算核心的峰值吞吐。

> **注意区分：训练/Prefill 与自回归 Decode**
> - **训练阶段与推理 Prefill 阶段**：整个序列已知，Transformer 能通过因果掩码（Causal Mask）实现整句全矩阵并行计算，属于计算密集型。
> - **自回归推理 Decode 阶段**：大模型生成下一个词时，每一步仅输入上一步生成的 1 个 Token，依赖 KV 缓存进行拼接。因此 Decode 阶段仍然是逐步串行的，且通常受限于显存读取带宽（Memory-Bound），并非全序列整段并行生成。

---

## 深层训练保障：残差连接与层归一化

仅靠矩阵并行和全连通结构，如果网络无法稳定堆叠到数十层乃至上百层，模型依然无法具备强大的上下文能力。Transformer 结合了残差连接与 LayerNorm。

![残差连接与 LayerNorm 构成的梯度高速通道](/images/content-rebuild/a3ca58602f4d0ac1/fig-5.png)

### 1. 残差连接（Residual Connection）的梯度直通
残差连接在子层外部建立了一条恒等映射分支：
- **原始 Post-LN 结构**（Vaswani et al., 2017）：$x_{l+1} = \mathrm{LayerNorm}(x_l + \mathrm{Sublayer}(x_l))$。这种结构在未经精细学习率预热（Warmup）时，反向传播经过外层 LayerNorm 容易发生数值不稳定。
- **现代主流 Pre-LN / Pre-RMSNorm 结构**：$x_{l+1} = x_l + \mathrm{Sublayer}(\mathrm{Norm}(x_l))$。
在 Pre-LN 结构下，主干路径的反向导数包含恒等项：

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_{l+1}} + \frac{\partial \mathcal{L}}{\partial x_{l+1}} \cdot \frac{\partial \mathrm{Sublayer}(\mathrm{Norm}(x_l))}{\partial x_l}$$

这保证了梯度信号能够无阻碍地从深层直接回传至浅层，防止多层非线性堆叠导致的梯度弥散。深入分析可参考 [/fundamentals/dl-basics/gradient-vanishing-exploding.md](/fundamentals/dl-basics/gradient-vanishing-exploding.md)。

### 2. 为什么选择 LayerNorm 而非 BatchNorm？
- **Batch Normalization (BatchNorm)** 沿 Batch 维度对不同样本的同一特征进行统计。但在自然语言中，不同句子长度差异大，Padding 填充符会干扰统计量；同时小批量微批次（Micro-batch）训练时统计量波动剧烈。
- **Layer Normalization (LayerNorm)** 沿**单个样本内部的特征隐藏维度 $d$** 独立计算均值与方差：

$$\mu = \frac{1}{d} \sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$$

$$\mathrm{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

它与序列长度解耦，与批次大小无关，确保了变长序列训练中特征分布的尺度稳定。关于各种归一化机制的对比可参阅 [/fundamentals/dl-basics/normalization.md](/fundamentals/dl-basics/normalization.md)。

---

## 架构的代价与边界：$O(n^2)$ 计算瓶颈

全连通机制的代价，是每次计算注意力打分时必须构建一张大小为 $n \times n$ 的关联矩阵。

![长序列下的注意力矩阵面积与显存消耗边界](/images/content-rebuild/a3ca58602f4d0ac1/fig-6.png)

当序列长度 $n$ 持续增加时：
- 序列长度扩大 4 倍（如从 2,048 到 8,192），注意力打分的计算量扩大 $4^2 = 16$ 倍；
- **FlashAttention 的作用与边界**：FlashAttention 算法通过 GPU 片上高速共享内存（SRAM）的分块在线计算，避免了在显存（HBM）中实体化存储完整的 $n \times n$ 中间矩阵，将显存占用从 $O(n^2)$ 降低到 $O(n)$，并大幅减少了显存读写瓶颈；但需要注意，全自注意力的**算术浮点计算量（FLOPs）仍然是 $O(n^2)$**，并没有改变其二次方计算复杂度的本质。

这也是学术界与工业界后续持续探索滑动窗口注意力（Sliding Window Attention）、稀疏注意力及线性状态空间模型（如 Mamba）的原因所在。

---

## 学习验收与可检验练习

完成本节后，请通过以下两道题目检验对核心机制的掌握：

1. **并行性检验**：
   已知一个训练样本包含 4,096 个 Token。如果采用标准 LSTM 训练，在前向传播中该样本至少需要等待多少次时序递推计算？如果采用标准因果 Transformer 训练，由于使用了因果注意力掩码（Causal Mask），该样本在序列维度上需要等待多少次串行操作？

   *检验答案提示*：LSTM 在序列维度必须执行 4,096 次串行递推；因果 Transformer 借助下三角掩码，利用单次下三角矩阵乘法即可一次性并行算出全部 4,096 个位置的损失，序列维度的串行操作数为 1（不考虑层间堆叠）。

2. **推理与训练差异检验**：
   为什么说“Transformer 彻底消除了串行计算”在推理生成阶段是不成立的？

   *检验答案提示*：Transformer 在自回归解码（Decode）生成文本时，第 $t$ 个 Token 的生成依赖上一个 Token 的预测结果作为输入，必须按时间步逐个生成，因而生成阶段依然是串行的，且单步推理受显存带宽与 KV 缓存读取限制。
