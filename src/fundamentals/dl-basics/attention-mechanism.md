---
title: 注意力机制是什么？如何实现动态关联与检索？
icon: mdi:focus-field
tag:
- 深度学习
- Attention
- 注意力机制
description: 介绍注意力机制如何解决定长压缩的信息瓶颈，拆解 QKV 角色、缩放点积四步计算、方差缩放推导及计算复杂度边界。
---

注意力机制（Attention Mechanism）是一种根据当前目标动态分配权重并聚合特征的计算方法。在处理序列数据时，它不把长输入强行压缩为单一固定向量，而是让模型在每个时间步按需计算与其他位置的相关程度，再把对应的内容加权提取出来。

以句子“小明把钥匙递给小李，因为他要出门”为例：当模型计算代词“他”的特征时，需要确定它究竟与前面的谁相关。注意力机制通过计算“他”与“小明”“小李”“钥匙”等各个词的匹配得分，动态赋予关键实体更高的权重，从而按需提取对应实体的特征，而不是把整句话的所有信息无差别地混在一起。

在现代深度学习中，从神经机器翻译到大语言模型，注意力机制已经成为构建长程依赖的核心组件。在此之前，时序建模主要依赖循环神经网络（可参考[/fundamentals/dl-basics/rnn-lstm-gru.md](/fundamentals/dl-basics/rnn-lstm-gru.md)）。要理解注意力机制的必要性，我们需要先看循环模型在处理长序列时遇到的瓶颈。

![注意力机制原理封面图](/images/content-rebuild/edfcb36d4e56ff3d/cover.png)

---

### 1. 传统 Encoder-Decoder 架构的信息压缩瓶颈

在注意力机制出现之前，基于序列到序列（Seq2Seq）的经典范式通常由两部分组成：编码器（Encoder）与解码器（Decoder）。

假设输入序列为 $X = (x_1, x_2, \dots, x_T)$，编码器通过循环单元逐步更新隐状态：

$$h_t = f(x_t, h_{t-1})$$

当处理完最后一个输入词 $x_T$ 后，编码器输出一个固定维度的上下文向量（Context Vector）$c$。在基础实现中，$c$ 通常直接取最后一个时间步的隐状态 $h_T$（或由其经过线性变换得到）：

$$c = h_T$$

解码器在生成每一个目标词 $y_i$ 时，都必须以该固定向量 $c$ 作为输入序列的全局表示：

$$s_i = g(y_{i-1}, s_{i-1}, c)$$

其中 $s_i$ 是解码器在第 $i$ 步的隐藏状态。

这种结构存在明显的**定长信息压缩瓶颈**：无论输入序列包含 5 个词还是 100 个词，编码器都试图将整句话的所有语义信息压缩进一个固定维度的向量 $c$（例如 512 维）。在循环传递过程中，随着输入长度 $T$ 的增加，前面位置（如 $x_1, x_2$）的信息经过多次矩阵乘法和非线性激活后，逐渐被后续信息覆盖和稀释。这导致解码器在生成长序列后半部分时，难以精准还原前文的关键细节。

![传统 Seq2Seq 定长上下文向量导致的信息挤压瓶颈](/images/content-rebuild/edfcb36d4e56ff3d/fig-1.png)

<ReadMoreLock />

---

### 2. 动态按需检索：保留所有隐状态

为了打破单一固定向量的容量上限，Dzmitry Bahdanau 等人在机器翻译任务中提出了注意力机制。其核心思路是：**编码器不再只输出最后一个隐状态，而是完整保留序列中每一个位置的隐状态 $(h_1, h_2, \dots, h_T)$；解码器在生成当前词时，动态决定应该关注编码器的哪些部分。**

具体而言，在解码器生成第 $i$ 个词时，系统会计算解码器当前状态与输入序列中每一个 $h_j$ 的相关性得分，经归一化后得到一组权重 $\alpha_{i,j}$：

$$c_i = \sum_{j=1}^{T} \alpha_{i,j} h_j$$

此时，上下文向量 $c_i$ 变成了随解码步骤 $i$ 动态变化的向量：
- 当翻译句首词时，权重集中在源语言的前半段；
- 当翻译句尾词时，权重转移到源语言的后半段。

这种按需加权求和的方式，避开了把全部信息强行塞入单个固定向量的约束，让解码器能够直接提取任意输入位置的原始隐状态。

![从固定上下文向量到动态按需检索的加权生成示意](/images/content-rebuild/edfcb36d4e56ff3d/fig-2.png)

---

### 3. Query、Key、Value 的分工与定义

在 Transformer 及其衍生架构中，注意力机制被抽象为一种**软寻址检索过程**。这个过程由三个核心角色构成：**Query（查询）、Key（键）与 Value（值）**。

- **Query（查询向量，$Q$）**：代表“当前的检索需求”。例如，解码器在当前步骤需要寻找动作的主体，它发出的 Query 包含“寻找施动者”的匹配特征。
- **Key（键向量，$K$）**：代表“候选对象的检索标签”。输入序列中的每个词都会生成一个 Key，用于与 Query 进行匹配打分。
- **Value（值向量，$V$）**：代表“候选对象实际承载的内容特征”。一旦某个位置的 Key 与 Query 匹配度高，该位置的 Value 就会在最终输出中占更大的权重。

普通数据库中的字典查询是**硬匹配**：Key 相等就返回 Value，不相等就返回空。而注意力机制是**软寻址（Soft Addressing）**：Query 与每一个 Key 都计算连续的相似度得分，并归一化为概率分布，最终输出所有 Value 的加权平均和。

在具体实现中，输入向量序列 $X \in \mathbb{R}^{T \times d_{model}}$ 并不直接充当 Q、K、V，而是分别通过三个可学习的线性变换矩阵投影得到：

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

其中 $W^Q \in \mathbb{R}^{d_{model} \times d_k}$、$W^K \in \mathbb{R}^{d_{model} \times d_k}$、$W^V \in \mathbb{R}^{d_{model} \times d_v}$。

> **维度约束说明**：Query 矩阵形状为 $(T_q, d_k)$，Key 矩阵形状为 $(T_k, d_k)$，Value 矩阵形状为 $(T_k, d_v)$。Query 与 Key 的特征维度必须一致（均为 $d_k$），否则无法计算点积相似度；而 Value 的特征维度 $d_v$ 可以根据任务需要与 $d_k$ 相同或不同。注意力的最终输出维度由 $d_v$ 决定。

![Query、Key、Value 的角色分工与线性映射结构](/images/content-rebuild/edfcb36d4e56ff3d/fig-3.png)

---

### 4. 缩放点积注意力的计算步骤

缩放点积注意力（Scaled Dot-Product Attention）的标准计算公式如下：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

整个运算由四个连续步骤组成：

1. **点积打分（Dot-Product Scoring）**：
   计算相似度矩阵 $S = Q K^T$，结果维度为 $(T_q, T_k)$。其中元素 $S_{ij} = q_i \cdot k_j^T$ 表示第 $i$ 个 Query 向量与第 $j$ 个 Key 向量的内积。内积越大，表明两个向量的方向越接近、语义关联度越高。
2. **尺度缩放（Scaling）**：
   将打分矩阵的每个元素除以缩放因子 $\sqrt{d_k}$，得到缩放打分矩阵 $\hat{S} = \frac{S}{\sqrt{d_k}}$。缩放的目的在于控制数值方差，防止进入 Softmax 梯度饱和区。
3. **概率归一化（Softmax Normalization）**：
   沿 Key 所在的序列维度（即矩阵的每一行）执行 Softmax 激活函数（参见[/fundamentals/dl-basics/activation-functions.md](/fundamentals/dl-basics/activation-functions.md)），得到注意力权重矩阵 $A = \text{softmax}(\hat{S})$，维度为 $(T_q, T_k)$。每一行元素均为非负数且和为 1：
   $$A_{ij} = \frac{\exp(\hat{S}_{ij})}{\sum_{l=1}^{T_k} \exp(\hat{S}_{il})}$$
4. **特征加权聚合（Weighted Aggregation）**：
   用权重矩阵乘以内容矩阵：$O = A V$，结果维度为 $(T_q, d_v)$。输出的每一行都是所有 Value 向量按照对应注意力概率加权求和后的新表征。

![缩放点积注意力的四步数据流与张量变换流程](/images/content-rebuild/edfcb36d4e56ff3d/fig-4.png)

#### PyTorch 最小验证代码

以下代码使用 PyTorch 展示上述四个步骤中的张量变换逻辑：

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)
seq_len = 3
d_k = 4
d_v = 2

# 模拟单个 Batch 的 Query, Key, Value 张量 (Batch=1, SeqLen=3, Dim)
Q = torch.randn(1, seq_len, d_k)
K = torch.randn(1, seq_len, d_k)
V = torch.randn(1, seq_len, d_v)

# 步骤 1: 计算点积打分 (1, 3, 4) @ (1, 4, 3) -> (1, 3, 3)
scores = torch.matmul(Q, K.transpose(-2, -1))

# 步骤 2: 尺度缩放除以 sqrt(d_k)
scale_factor = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
scaled_scores = scores / scale_factor

# 步骤 3: Softmax 归一化，在 Key 的序列维度求和为 1
attention_weights = F.softmax(scaled_scores, dim=-1)

# 步骤 4: 加权聚合 Value (1, 3, 3) @ (1, 3, 2) -> (1, 3, 2)
output = torch.matmul(attention_weights, V)

print("注意力权重矩阵 (每行和为 1):\n", attention_weights)
print("输出张量形状 (由 d_v 决定):", output.shape)
```

运行上述代码可以看到，最终输出的特征维度完全由 $d_v$ 决定，而中间权重矩阵的大小只取决于 Query 和 Key 的序列长度。

---

### 5. 为什么除以 $\sqrt{d_k}$：方差控制与梯度特性

许多初学者会有疑问：既然点积本身就能衡量相关性，为什么还要在 Softmax 前除以 $\sqrt{d_k}$？这可以通过统计学方差分析和反向传播梯度特性来解释。

#### 5.1 假设条件下的点积方差推导

假设 Query 向量 $q \in \mathbb{R}^{d_k}$ 与 Key 向量 $k \in \mathbb{R}^{d_k}$ 的各个分量相互独立，且均满足均值为 0、方差为 1 的分布：

$$\mathbb{E}[q_i] = 0, \quad \mathrm{Var}(q_i) = 1, \quad \mathbb{E}[k_i] = 0, \quad \mathrm{Var}(k_i) = 1$$

两者的点积为分量乘积之和：

$$S = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

根据独立变量乘积性质，每一项的均值与方差为：

$$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$
$$\mathrm{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] - 0 = 1 \times 1 = 1$$

由于各分量相互独立，和的方差等于各项方差之和：

$$\mathrm{Var}(S) = \sum_{i=1}^{d_k} \mathrm{Var}(q_i k_i) = d_k$$

该推导表明：**在分量独立标准化的理论假设下，点积的方差与维度 $d_k$ 呈线性关系，其标准差随 $\sqrt{d_k}$ 递增。** 当特征维度较小（如 $d_k = 4$）时，标准差为 2；当特征维度扩大到 64 或 128 时，标准差达到 8 或 11.3。点积的数值波动区间随维度增大而显著扩散。

#### 5.2 Softmax 饱和与梯度减弱

设 Softmax 函数的输入向量为 $z$，输出向量为 $y$，其中 $y_i = \text{softmax}(z)_i$。对输入标量 $z_j$ 求偏导的结果为：

$$\frac{\partial y_i}{\partial z_j} = y_i (\delta_{ij} - y_j)$$

其中 $\delta_{ij}$ 为克罗内克符号（当 $i=j$ 时为 1，其余为 0）。

当输入向量的分量方差极大时，某个位置的值会显著大于其他位置。经过指数函数 $\exp(\cdot)$ 计算后，最大值在分母中占据绝对优势，导致该位置的输出概率接近 1，其余位置的输出概率接近 0：

- 当 $i = j$ 时，$\frac{\partial y_i}{\partial z_i} = y_i (1 - y_i) \approx 1 \times (1 - 1) = 0$
- 当 $i \neq j$ 时，$\frac{\partial y_i}{\partial z_j} = - y_i y_j \approx - 1 \times 0 = 0$

这时，Softmax 的导数在所有位置均接近于 0，进入梯度饱和区。当下游梯度通过反向传播（参见[/fundamentals/dl-basics/forward-backward-propagation.md](/fundamentals/dl-basics/forward-backward-propagation.md)与[/fundamentals/dl-basics/gradient-vanishing-exploding.md](/fundamentals/dl-basics/gradient-vanishing-exploding.md)）传回时，乘以此处的极小导数会导致上游参数（如投影矩阵 $W^Q, W^K$）的梯度大幅衰减，增加训练优化的难度。

将打分除以 $\sqrt{d_k}$ 后，点积的方差在理想假设下重新被缩放到 $\frac{d_k}{(\sqrt{d_k})^2} = 1$，使 Softmax 的输入保持在导数响应较为敏感的平缓区间内。

![Softmax 梯度饱和与缩放因子稳定方差对比分析](/images/content-rebuild/edfcb36d4e56ff3d/fig-5.png)

---

### 6. 计算复杂度与建模路径对比

不同架构处理序列长程依赖时的交互路径和计算代价存在本质差异：

| 对比维度 | 循环网络（RNN / LSTM） | 卷积网络（1D CNN） | 全自注意力（Self-Attention） |
| :--- | :--- | :--- | :--- |
| **任意两位置交互路径长** | $O(T)$ 步（沿时间步逐步累积） | $O(\log_k T)$ 层（依赖网络层数堆叠） | **$O(1)$ 步（矩阵乘法直接交互）** |
| **全序列交互算术复杂度** | $O(T \cdot d^2)$ | $O(k \cdot T \cdot d^2)$ | **$O(T^2 \cdot d)$** |
| **序列并行能力** | 弱（前后时间步存在递归依赖） | 强（沿时间维度卷积并行） | **强（所有位置矩阵操作同时进行）** |

在循环网络中，第 1 个词若要对第 100 个词产生影响，必须按顺序流经中间 99 个时间步的递归传递。而在全自注意力机制中，第 1 个词与第 100 个词直接在 $Q K^T$ 的矩阵乘法中发生交互，路径长度仅为 1 步。

然而，这种一步直达的交互能力需要付出算术计算与显存访问的代价：

1. **全序列计算代价**：在模型训练或推理的首字预填充（Prefill）阶段，全注意力需要计算完整的 $T \times T$ 打分矩阵。这意味着浮点运算量（FLOPs）和存储中间权重的理论显存开销随序列长度 $T$ 呈二次方增长（$O(T^2)$）。
2. **单步解码代价**：在文本生成（Decode）阶段，模型每次只生成一个新 Token。如果维护了历史 Key 和 Value 的缓存（KV Cache），当前新 Token 只需要与历史 $T$ 个位置计算一次点积，该单步计算的算术复杂度为 $O(T \cdot d)$，并非每一步都重新进行 $O(T^2)$ 运算。
3. **工程优化边界**：FlashAttention 等算子级优化通过分块加载和在线 Softmax 算法，显著减少了显卡高带宽内存（HBM）的中间读写次数，将显存占用降低到线性级别，但并没有改变全注意力在算术运算量上固有的 $O(T^2)$ 特性。

![时序建模路径与计算复杂度对比拓扑图](/images/content-rebuild/edfcb36d4e56ff3d/fig-6.png)

---

### 总结

注意力机制通过引入 Query、Key、Value 的分工，将特征检索与内容提取解耦，利用点积软寻址打破了定长上下文向量的信息瓶颈。除以 $\sqrt{d_k}$ 的缩放操作从数值上控制了点积方差，避免了 Softmax 陷入梯度饱和区。

单组注意力在同一个特征子空间内往往只能关注一种特定的关联模式。为了同时建模多种句法与语义关系，Transformer 将注意力扩展为多头注意力（Multi-Head Attention），并结合残差连接与层归一化构建深度网络。具体机制将在下一篇[/fundamentals/dl-basics/transformer.md](/fundamentals/dl-basics/transformer.md)中详细讲解。
