---
title: Transformer 为什么需要位置编码？位置信息是如何注入的
icon: mdi:map-marker-outline
tag:
- Transformer架构
- 位置编码
- 序列建模
description: 通过具体例子推导自注意力的置换等变性，解析正弦余弦位置编码的波长设计、线性平移性质与相加注入机制。
---

在前面的课程[/fundamentals/transformer-arch/multi-head-attention.md](/fundamentals/transformer-arch/multi-head-attention.md)中，我们学习了多头注意力机制如何通过矩阵投影同时捕获多种词语关联。但在使用自注意力时，会遇到一个天然的结构特征：**它本身不包含词语的前后顺序信息**。

如果我们把一段话里的词语顺序完全打乱，自注意力机制所计算出的特征依然只会在打乱后的槽位里保持原样，无法识别语法上的先后关系。本文通过一个贯穿例子说明自注意力的“置换等变性”，推导原始 Transformer（Vaswani et al., 2017）采用的正弦余弦绝对位置编码机制，并解析位置向量通过“相加”注入模型的代数与几何原因。

![位置编码整体机制封面图](/images/content-rebuild/338e3dd618cd6a49/cover.png)

---

## 1. 核心困境：没有位置信息会发生什么？

我们先来看一个贯穿全文的具体例子：
- 句子 A：`["张三", "打", "李四"]`
- 句子 B：`["李四", "打", "张三"]`

在中文常识中，这两个句子的意思完全相反。但在没有位置编码的标准自注意力机制中：
1. 计算“张三”和“打”的相关性，只取决于“张三”和“打”本身的词嵌入向量；
2. 计算“打”和“李四”的相关性，也只取决于它们的词向量；
3. 无论“张三”排在第一位还是第三位，它与其它词的两两点积得分都是完全相同的一组数值。

其直接结果是：模型为句子 A 中的“打”计算出来的上下文表示向量，与句子 B 中的“打”计算出来的向量完全相同；句子 A 里的“张三”所得到的表征，也和句子 B 里的“张三”一模一样。模型根本分不清谁是施事者、谁是受事者。

这种数学性质称为**置换等变性（Permutation Equivariance）**。

![自注意力置换等变性矩阵变换流程图](/images/content-rebuild/338e3dd618cd6a49/fig-1.png)

---

## 2. 数学推导：自注意力的置换等变性

为什么自注意力无法感知顺序？我们用置换矩阵进行严格推导。

设输入序列的嵌入矩阵为 $X \in \mathbb{R}^{n \times d}$，包含 $n$ 个 Token，每个 Token 维度为 $d$。线性投影后的查询、键、值矩阵分别为 $Q = XW_Q$、$K = XW_K$、$V = XW_V$。

自注意力的计算公式为：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

我们对输入的 $n$ 个 Token 任意调换顺序。任何离散换序操作都可以表示为一个置换矩阵（Permutation Matrix）$P \in \mathbb{R}^{n \times n}$。置换矩阵的每行和每列只有一个元素为 1，其余全为 0，且满足正交性质 $P^T P = P P^T = I$。换序后的新输入矩阵为 $X_{\text{perm}} = PX$。

我们把打乱后的输入代入计算链路：

1. **投影矩阵**：
   $$Q_{\text{perm}} = (PX)W_Q = P(XW_Q) = PQ$$
   $$K_{\text{perm}} = (PX)W_K = PK, \quad V_{\text{perm}} = (PX)W_V = PV$$

2. **注意力打分**：
   $$S_{\text{perm}} = \frac{Q_{\text{perm}} K_{\text{perm}}^T}{\sqrt{d_k}} = \frac{(PQ)(PK)^T}{\sqrt{d_k}} = \frac{PQK^T P^T}{\sqrt{d_k}} = P \left(\frac{QK^T}{\sqrt{d_k}}\right) P^T = PSP^T$$
   左乘 $P$ 交换了得分矩阵的行，右乘 $P^T$ 交换了得分矩阵的列。

3. **按行执行 Softmax 归一化**：
   Softmax 是对每一行内部的所有列做指数归一化。右乘 $P^T$ 只是改变了行内求和项的累加顺序，而加法满足交换律，分母的分值总和不变。因此：
   $$\text{softmax}(PSP^T) = P \,\text{softmax}(S)\, P^T$$

4. **与 Value 矩阵加权求和**：
   $$\text{Attention}(Q_{\text{perm}}, K_{\text{perm}}, V_{\text{perm}}) = \left( P \,\text{softmax}(S)\, P^T \right) (PV) = P \,\text{softmax}(S)\, (P^T P) V = P \cdot \text{Attention}(Q, K, V)$$

残差连接、LayerNorm 以及逐位置前馈网络（Position-wise Feed-Forward Network）都是针对每个位置独立计算的，同样满足 $f(PX) = P f(X)$。

<ReadMoreLock />

这说明：**输入顺序被打乱，输出只是在位置槽位上做同样的打乱，向量数值本身没有因语序变化而产生任何新的语境信息。**

![张三打李四与李四打张三在无位置编码下的失败对比图](/images/content-rebuild/338e3dd618cd6a49/fig-2.png)

自然语言强依赖语序。如果要让模型区分前后顺序，必须在输入端显式引入与位置相关的信息，打破置换对称性。

---

## 3. 正弦与余弦位置编码：波长与频率设计

在《Attention Is All You Need》中，Vaswani 等人提出了无需训练参数的正弦与余弦绝对位置编码（Sinusoidal Positional Encoding）。

设输入序列的绝对索引为 $pos \in [0, N-1]$，向量隐藏维度为 $d_{\text{model}}$。在位置 $pos$ 处，位置向量 $PE_{pos} \in \mathbb{R}^{d_{\text{model}}}$ 的各个通道定义如下：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$

其中 $i \in [0, \frac{d_{\text{model}}}{2}-1]$ 是通道对的索引。

![正弦与余弦波长随维度几何级数衰减设计机制图](/images/content-rebuild/338e3dd618cd6a49/fig-3.png)

这个公式的设计直觉与机械表上的“秒针、分针、时针”类似：通过多组不同快慢周期的指针，共同组合出一个精确的时间读数：

1. **角频率随维度指数降低**：
   每个通道对 $(2i, 2i+1)$ 对应的角频率为：
   $$\omega_i = \frac{1}{10000^{2i / d_{\text{model}}}}$$
   当 $i=0$ 时，$\omega_0 = 1$；当 $i$ 增加到通道末端时，频率降至约 $\frac{1}{10000}$。

2. **波长跨度覆盖短程与长程**：
   三角函数的周期（即波长）为 $\lambda_i = \frac{2\pi}{\omega_i} = 2\pi \cdot 10000^{2i / d_{\text{model}}}$：
   - **低维度分量（$i=0$）**：波长 $\lambda_0 = 2\pi \approx 6.28$ 个 Token。波形变化密集，相邻位置数值变化剧烈，专门用于刻画局部的相邻相对位移；
   - **高维度分量（$i$ 接近末端）**：波长长达数万个 Token。在常见上下文长度内，数值变化平缓，主要用于提供全局位置基底。

这样，每一个位置 $pos$ 都在 $d_{\text{model}}$ 维空间中对应一个全局唯一的坐标向量，且各分量数值严格落在 $[-1, 1]$ 之间。

---

## 4. 为什么三角函数能表达相对距离？

语言理解不仅关心绝对位置，更关心词与词之间的**相对距离**（例如动词往往修饰其前方或后方紧邻的宾语）。

正余弦编码的一个重要代数性质是：**任意偏移 $k$ 个步长后的位置向量 $PE_{pos+k}$，可以通过一个只与 $k$ 有关、与 $pos$ 无关的固定线性变换由 $PE_{pos}$ 计算得到。**

![二维正交旋转矩阵推导几何示意图](/images/content-rebuild/338e3dd618cd6a49/fig-4.png)

### 4.1 数学推导

考虑第 $i$ 组二维子空间向量：
$$u_i(pos) = \begin{bmatrix} \sin(\omega_i pos) \\ \cos(\omega_i pos) \end{bmatrix}$$

当位置偏移到 $pos+k$ 时，根据三角和角公式：
$$\sin(\omega_i (pos + k)) = \sin(\omega_i pos)\cos(\omega_i k) + \cos(\omega_i pos)\sin(\omega_i k)$$
$$\cos(\omega_i (pos + k)) = \cos(\omega_i pos)\cos(\omega_i k) - \sin(\omega_i pos)\sin(\omega_i k)$$

写成矩阵形式：
$$\begin{bmatrix} \sin(\omega_i (pos + k)) \\ \cos(\omega_i (pos + k)) \end{bmatrix} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix} \begin{bmatrix} \sin(\omega_i pos) \\ \cos(\omega_i pos) \end{bmatrix}$$

定义二维旋转矩阵 $R_i(k)$：
$$R_i(k) = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix}$$

该矩阵的元素只取决于频率 $\omega_i$ 和相对偏移量 $k$，完全不依赖初始位置 $pos$。在全维度空间上，这对应一个分块对角矩阵 $M(k)$：
$$PE_{pos+k} = M(k) \cdot PE_{pos}$$

### 4.2 对注意力的直接效果

当注意力机制计算位置 $pos$ 与位置 $pos+k$ 的位置向量内积时：
$$u_i(pos)^T u_i(pos+k) = \sin(\omega_i pos)\sin(\omega_i (pos+k)) + \cos(\omega_i pos)\cos(\omega_i (pos+k)) = \cos(\omega_i k)$$

在二维子空间上，两个位置向量的内积直接化简为 $\cos(\omega_i k)$，只与相对位移 $k$ 相关。这让线性投影矩阵能够方便地通过位置分量感知相对距离。

---

## 5. 注入方式：为什么直接相加而非拼接？

在实现中，位置编码并不是拼接到词嵌入的末尾（Concatenation），而是直接逐元素相加（Element-wise Addition）：
$$z_{pos} = x_{pos} + PE_{pos}$$

开发者常问：把语义向量和位置向量直接加在一起，难道不会破坏词原有的语义表达吗？

![相加与拼接在空间投影与交互维度的对比图](/images/content-rebuild/338e3dd618cd6a49/fig-5.png)

从计算成本和向量空间特性来看，这种设计有其考量：

### 5.1 维度与计算开销
如果采用拼接，输入维度将扩大到 $d + d_{pos}$。这意味着后续所有的投影矩阵（$W_Q, W_K, W_V$、输出投影 $W_O$ 以及前馈网络矩阵）的参数量和矩阵乘法计算量都会对应增大。直接相加保持了输入维度 $d_{\text{model}}$ 不变，无需扩充模型参数。

### 5.2 高维空间的子空间容量
在高维向量空间中（例如 $d_{\text{model}} = 512$ 或更高），随机向量两两之间几乎保持近似正交。词嵌入往往分布在整个高维空间的某些低维子空间中，其余维度空间具有足够的容量容纳位置信号。在模型训练过程中，投影矩阵可以通过反向传播自发学习将语义方向与位置方向区分开来。

此外，在 Transformer 原论文的实现中，词嵌入在与位置编码相加前会先乘以 $\sqrt{d_{\text{model}}}$（512 维时约为 22.6）。这是因为在与输出层共享权重时，词嵌入初始化方差通常为 $1/d_{\text{model}}$，向量模长仅约为 1；乘以 $\sqrt{d_{\text{model}}}$ 可将其整体模长提升至与位置编码相当的尺度（512 维位置向量模长约为 16），防止固定的位置信号在训练初期压制词语义特征。

### 5.3 形成四项注意力交互
将相加后的向量 $z_m = x_m + pe_m$ 与 $z_n = x_n + pe_n$ 投影为 Query 和 Key，其点积得分为：
$$q_m k_n^T = (x_m + pe_m) W_Q W_K^T (x_n + pe_n)^T$$

展开后形成四项：
$$q_m k_n^T = \underbrace{x_m W_Q W_K^T x_n^T}_{\text{词-词内容关联}} + \underbrace{x_m W_Q W_K^T pe_n^T}_{\text{内容-位置偏差}} + \underbrace{pe_m W_Q W_K^T x_n^T}_{\text{位置-内容偏差}} + \underbrace{pe_m W_Q W_K^T pe_n^T}_{\text{位置-位置偏差}}$$

1. **第 1 项**：传统的语义内容交互；
2. **第 2、3 项**：特定词汇在特定位置的偏好（例如句首更常出现连词或名词）；
3. **第 4 项**：纯位置之间的偏好，直接建模序列的距离分布。

通过相加操作，模型无需扩充向量维度，就能让注意力得分自动包含这四类信息交互。

---

## 6. 代码验证：置换等变性与对称性打破

我们通过一个最小 Python 脚本验证两个事实：
1. 无位置编码时，打乱输入会导致输出以相同方式打乱，等变误差在浮点精度内为 0；
2. 注入位置编码后，置换等变性被打破，模型能够区分词语所在的具体位置。

```python
import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def self_attention(X, W_q, W_k, W_v):
    d_k = W_q.shape[1]
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    scores = (Q @ K.T) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    return weights @ V

def get_sinusoidal_pe(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    # 计算 1 / 10000^(2i / d_model)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

# 固定随机种子
np.random.seed(42)
seq_len, d_model = 4, 8
X = np.random.randn(seq_len, d_model)
W_q = np.random.randn(d_model, d_model)
W_k = np.random.randn(d_model, d_model)
W_v = np.random.randn(d_model, d_model)

# 构造置换矩阵 P (交换索引 1 与索引 3 对应行)
P = np.eye(seq_len)[[0, 3, 2, 1]]
X_perm = P @ X

# 1. 验证无位置编码下的置换等变性: Attention(PX) == P @ Attention(X)
out_orig = self_attention(X, W_q, W_k, W_v)
out_perm = self_attention(X_perm, W_q, W_k, W_v)
perm_error = np.max(np.abs(out_perm - (P @ out_orig)))
print(f"无位置编码置换等变误差 (应趋近于 0): {perm_error:.2e}")

# 2. 注入位置编码打破对称性
PE = get_sinusoidal_pe(seq_len, d_model)
out_with_pe = self_attention(X + PE, W_q, W_k, W_v)
out_perm_with_pe = self_attention(X_perm + PE, W_q, W_k, W_v)
broken_diff = np.max(np.abs(out_perm_with_pe - (P @ out_with_pe)))
print(f"注入位置编码后对称性被打破差异量: {broken_diff:.4f}")
```

运行结果中，无编码状态下的置换等变误差为 `0.00e+00`，说明没有位置信息时自注意力完全无法识别词序变化；加入位置编码后，`broken_diff` 远大于 0，模型成功感知到了每个词在当前句子中的绝对位置。

![位置编码在完整网络中的注入拓扑图](/images/content-rebuild/338e3dd618cd6a49/fig-6.png)

---

## 7. 机制边界与现代演进

正弦余弦绝对位置编码结构简单且无需可学习参数，但在大模型实践中存在外推局限：
1. **绝对坐标外推困难**：虽然理论上 $PE_{pos+k} = M(k)PE_{pos}$ 成立，但在堆叠多层非线性激活函数与注意力计算后，模型很难单纯依赖权重恢复未见过的长距离位置模式。如果推理长度超出预训练长度，性能通常明显下降；
2. **相对位置编码与 RoPE**：正因如此，后续架构逐步发展出了基于注意力打分偏置的相对位置编码（如 T5、ALiBi），以及现代大语言模型（如 LLaMA）广泛使用的旋转位置编码（RoPE）。RoPE 通过直接在 $Q, K$ 上施加复数旋转变换，将相对位置信息显式保存在点积中。

关于旋转位置编码（RoPE）的具体推导，可参考面试进阶专题[/interview/llm-interview/positional-encoding.md](/interview/llm-interview/positional-encoding.md)。

下一篇我们将探讨架构分支：[/fundamentals/transformer-arch/encoder-decoder-variants.md](/fundamentals/transformer-arch/encoder-decoder-variants.md)，对比 Encoder-only、Decoder-only 和 Encoder-Decoder 结构在结合因果掩码与位置信号时的设计差异。
