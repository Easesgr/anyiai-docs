---
title: 位置编码为什么重要？RoPE 和 ALiBi 分别解决了什么？
icon: mdi:book-open-page-variant-outline
date: 2026-07-16
tag:
- 大模型100讲
- 基础篇
description: 解析自注意力的置换等变性问题，详解绝对位置编码、RoPE 旋转位置编码与 ALiBi 线性偏置的数学推导、代码实现与外推边界。
---

在自然语言中，词语出现的先后顺序直接决定了整句话的语义。上一讲我们讨论了 [Self-Attention 和 Multi-Head Attention 有什么区别](/llm-100/basics/Self-Attention_和_Multi-Head_Attention_有什么区别.md)，了解了查询（Query）、键（Key）和值（Value）如何通过点积计算权重。然而，标准的自注意力机制在数学上对输入顺序完全不敏感。如果把输入词打乱，模型算出来的注意力输出也只是跟着被打乱，没有任何一个参数能分辨谁先谁后。为了让模型感知语序和词距，必须显式引入**位置编码（Positional Encoding）**。

从最初 Transformer 采用的绝对位置编码，到 LLaMA 等主流模型普及的 RoPE，再到 BLOOM 采用的 ALiBi，位置编码经历了一场从“给每个词打物理坐标”到“在注意力计算中表达相对距离”的技术演进。

![讲解员在黑板前对比绝对坐标、旋转位置编码与注意力线性偏置](/images/content-rebuild/a7b93b99f66ce028/cover.png)

---

## 1. 贯穿示例与核心结论

为了直观理解位置编码的作用，我们看一个贯穿全文的具体例子：

* **句子 A**：`小明 打了 小红`
* **句子 B**：`小红 打了 小明`

这两个句子包含完全相同的词汇集合。如果模型没有位置编码：
1. `打` 对 `小明` 和 `小红` 的注意力权重完全取决于词本身的词向量，模型分不清谁是主语、谁是宾语。
2. 进一步考虑长文本：在第 5000 个词出现代词 `他` 时，它距离第 4995 个词的实体更近，距离第 5 个词的实体极远。模型需要感知两者相隔的“相对距离”，才能建立正确的指代消解。

### 核心结论速览
* **为什么必须要有位置编码**：自注意力计算具备**置换等变性（Permutation Equivariance）**。打乱输入，输出只是原样打乱，模型无法感知时序先后。
* **绝对位置编码（Sinusoidal / Learned PE）**：直接把一个表示绝对坐标的向量加到输入词向量上。缺点是一旦推理长度超出训练时的最大长度，模型要么没有对应参数，要么因为分布漂移导致性能急剧下降，几乎无法直接外推。
* **RoPE（Rotary Position Embedding，旋转位置编码）**：不给词向量做加法，而是在计算注意力前，将 Query 和 Key 向量在二维子空间中逆时针旋转一定角度。两向量的点积结果自然只取决于**旋转夹角差（相对距离）**，兼顾了相对位置建模与高维计算效率。
* **ALiBi（Attention with Linear Biases，带线性偏置的注意力）**：彻底不修改输入和投影向量，而在计算出的注意力分数（Logits）上直接减去一个与距离成正比的惩罚项。它具备极佳的免微调长文本外推能力，但由于强行压制远距离注意力，在需要超长程精确定位的检索任务中表达力受限。

---

## 2. 注意力机制为什么是“位置盲”的？

自注意力机制的标准公式为：

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q = XW_Q, K = XW_K, V = XW_V$，$X \in \mathbb{R}^{N \times d}$ 是由 $N$ 个 Token 嵌入向量拼成的输入矩阵。

假设我们用一个置换矩阵 $P \in \mathbb{R}^{N \times N}$ 打乱输入的行次序（$P$ 是正交矩阵，每行每列仅有一个 1，其余全为 0，满足 $P^T P = I$）。令打乱后的输入为 $\tilde{X} = PX$，此时新的查询、键、值矩阵变为 $\tilde{Q} = PQ, \tilde{K} = PK, \tilde{V} = PV$。

代入计算：

$$\begin{aligned} \mathrm{Attention}(\tilde{Q}, \tilde{K}, \tilde{V}) &= \mathrm{softmax}\left(\frac{(PQ)(PK)^T}{\sqrt{d_k}}\right)(PV) \\ &= \mathrm{softmax}\left(\frac{P Q K^T P^T}{\sqrt{d_k}}\right)(PV) \\ &= P \cdot \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) P^T (PV) \\ &= P \cdot \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V \\ &= P \cdot \mathrm{Attention}(Q, K, V) \end{aligned}$$

![注意力置换不变性示意图：打乱输入序列后自注意力权重的对应重排现象](/images/content-rebuild/a7b93b99f66ce028/fig-1.png)

数学推导清晰表明：输入矩阵做了一次行重排 $P$，输出矩阵仅仅是跟着做了一次完全相同的行重排 $P$。

这意味着每个词在更新表征时，聚合周围词的权重完全由词向量的内容点积决定。即使在因果解码器中加入了下三角因果掩码（Causal Mask），使得模型看不到未来的词，但是在已经可见的所有历史词中，模型仍然无法区分“距离当前词 1 个位置的词”和“距离当前词 100 个位置的词”。因此，必须向模型注入次序信息。

---

## 3. 绝对位置编码：给每个位置贴门牌号

早期最直观的做法是在模型输入层，给序列中的每个绝对物理坐标 $pos \in [0, N-1]$ 分配一个位置向量 $p_{pos} \in \mathbb{R}^d$，并直接与词嵌入向量 $e_{pos}$ 逐元素相加：

$$x_{pos} = e_{pos} + p_{pos}$$

![正弦绝对位置编码的生成与加法注入过程](/images/content-rebuild/a7b93b99f66ce028/fig-2.png)

绝对位置编码主要有两种经典实现：

### 3.1 可学习绝对位置嵌入（Learned Absolute PE）
在 BERT、GPT-2 等模型中，模型维护一个形状为 $[L_{max}, d]$ 的可学习参数矩阵。第 0 个位置查第 0 行，第 1 个位置查第 1 行，随模型预训练一起梯度更新。
* **优点**：简单直接，参数可自由适配训练数据中的位置统计偏置。
* **缺陷**：最大长度被固定为 $L_{max}$。如果模型预训练长度是 2048，推理遇到第 2049 个 Token 时，根本没有对应的参数行可供索引，无法直接处理超长文本。

### 3.2 正弦位置编码（Sinusoidal Positional Encoding）
在 Vaswani et al. (2017) 的原始 Transformer 中，作者使用预先计算好的不同波长正余弦函数生成位置向量：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

其中 $pos$ 为绝对坐标，$i \in [0, d/2 - 1]$ 为特征维度的配对索引。频率 $\omega_i = \frac{1}{10000^{2i/d}}$ 随维度增加呈几何级数递减，低维分量对应高频短周期波，高维分量对应低频长周期波。

该设计具备一个数学性质：对于任意固定位移 $k$，位置 $pos+k$ 的编码向量可以通过位置 $pos$ 的编码向量经过一个线性旋转矩阵直接计算得出：

$$\begin{bmatrix} \sin(\omega_i (pos + k)) \\ \cos(\omega_i (pos + k)) \end{bmatrix} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix} \begin{bmatrix} \sin(\omega_i pos) \\ \cos(\omega_i pos) \end{bmatrix}$$

### 为什么正弦编码依然无法顺畅外推？
虽然公式可以为任意大的 $pos$ 计算出数值，但在实际模型中：
1. **加法操作破坏纯粹性**：位置向量 $p_m$ 在输入层直接加到词向量 $e_m$ 上。经过多层投影矩阵 $W_Q, W_K$ 后，两词的点积展开为：
   $$q_m k_n^T = (e_m + p_m) W_Q W_K^T (e_n + p_n)^T = e_m W_Q W_K^T e_n^T + e_m W_Q W_K^T p_n^T + p_m W_Q W_K^T e_n^T + p_m W_Q W_K^T p_n^T$$
   展开式中存在词向量与位置向量交叉相乘的杂项（$e_m W_Q W_K^T p_n^T$ 等），相对位移关系无法直接保留在最终注意力分数中。
2. **深层非线性破坏结构**：经过多层 Transformer 的非线性激活（Feed-Forward Network）与归一化（LayerNorm）后，浅层注入的位置数学关系早已变形。
3. **高频分量数值未见**：模型在高层从未学习过超出预训练窗口的绝对数值分布，直接输入长文本会导致注意力分布紊乱，困惑度快速恶化。

<ReadMoreLock />

---

## 4. RoPE 旋转位置编码：用复数旋转表达相对位置

为了克服绝对坐标的局限，研究转向了**相对位置编码**。在语言理解中，两个词之间的语法依从关系主要取决于它们相隔多远（相对距离 $m - n$），而不是它们位于文章的第几百个词。

旋转位置编码（Rotary Position Embedding, RoPE，Su et al., 2021）是当前主流大模型（LLaMA、Qwen、DeepSeek 等）最通用的位置编码方案。

### 4.1 设计目标
RoPE 不在输入嵌入上做加法，而是直接作用于投影后的 Query 向量 $q_m$（位置 $m$）和 Key 向量 $k_n$（位置 $n$）。我们希望找到一个变换函数 $f(x, pos)$，使得经过变换后的两向量内积**仅包含相对位置差 $(m - n)$**：

$$\langle f(q, m), f(k, n) \rangle = g(q, k, m - n)$$

### 4.2 二维平面下的旋转推导
以特征维度 $d = 2$ 为例。在平面直角坐标系中，实数向量 $x = [x_0, x_1]^T$ 可以映射为二维复数 $z_x = x_0 + i x_1$。

若定义变换 $f(x, m)$ 为将该复数在复平面上逆时针旋转角度 $m\theta$：

$$f(x, m) = z_x \cdot e^{i m \theta} = (x_0 + i x_1)(\cos(m\theta) + i \sin(m\theta))$$

写回实数矩阵形式，即为二维正交旋转矩阵：

$$f(x, m) = R_m x = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \end{bmatrix}$$

现在计算位置 $m$ 的 Query 和位置 $n$ 的 Key 之间的内积：

$$\langle f(q, m), f(k, n) \rangle = (R_m q)^T (R_n k) = q^T R_m^T R_n k$$

根据正交矩阵的性质，$R_m^T = R_{-m}$，且旋转矩阵相乘等于角度相加：

$$R_m^T R_n = R_{-m} R_n = R_{n-m} = R_{-(m-n)}$$

此时点积结果变为：

$$\langle f(q, m), f(k, n) \rangle = q^T R_{-(m-n)} k$$

![RoPE 旋转位置编码在二维平面中的复数向量旋转与相对角差内积](/images/content-rebuild/a7b93b99f66ce028/fig-3.png)

推导结果非常直观：**绝对位置 $m$ 和 $n$ 在内积中完全消除，最终计算值仅由相对距离差 $(m - n)$ 决定**。

### 4.3 扩展到高维空间
对于注意力头维度 $d$（$d$ 为偶数），RoPE 将这 $d$ 维切分成 $d/2$ 个互不重叠的二维子空间。第 $i$ 个子空间（$i \in [0, d/2 - 1]$）分配一个独立的基频 $\theta_i$：

$$\theta_i = 10000^{-2i/d}$$

![RoPE 高维分块对角旋转计算流](/images/content-rebuild/a7b93b99f66ce028/fig-4.png)

高维旋转矩阵 $R_{\Theta, m}^d$ 是一个分块对角矩阵：

$$R_{\Theta, m}^d = \begin{bmatrix} \cos(m\theta_0) & -\sin(m\theta_0) & 0 & 0 & \dots \\ \sin(m\theta_0) & \cos(m\theta_0) & 0 & 0 & \dots \\ 0 & 0 & \cos(m\theta_1) & -\sin(m\theta_1) & \dots \\ 0 & 0 & \sin(m\theta_1) & \cos(m\theta_1) & \dots \\ \vdots & \vdots & \vdots & \vdots & \ddots \end{bmatrix}$$

### 4.4 工程实现与性质
1. **保范数性**：由于旋转矩阵是正交矩阵（$R^T R = I$），旋转操作完全不改变向量的欧几里得范数（L2 范数），不会引入数值放大或缩小。
2. **无需构造稀疏矩阵**：在显卡实现时，不需要真正开辟大矩阵做矩阵乘法。可以利用向量逐元素相乘（Hadamard 积 $\odot$）并行计算：
   $$R_{\Theta, m}^d x = x \odot \cos(m\Theta) + \tilde{x} \odot \sin(m\Theta)$$
   其中 $\tilde{x} = [-x_1, x_0, -x_3, x_2, \dots, -x_{d-1}, x_{d-2}]$。

---

## 5. ALiBi：用线性衰减偏置实现零样本外推

虽然 RoPE 实现了相对位置内积，但如果在训练时模型只见过最大 4096 长度的旋转角度，当推理输入达到 16384 时，低频分量（对应高维切片）的旋转角度超出了预训练见过的最大角度区间（分布外 OOD），直接外推仍会出现注意力发散。

Press et al. (2021) 提出了 **ALiBi（Attention with Linear Biases，带线性偏置的注意力）**。它采取了一条完全不同的思路：**不在向量中添加任何位置信号，而是在注意力打分矩阵上直接减去一个与相对距离成正比的惩罚项**。

### 5.1 机制定义
在因果注意力机制中，位置 $i$ 的 Query 与位置 $j$ 的 Key（$i \ge j$）之间的未归一化注意力得分计算如下：

$$\text{Score}_{i, j} = \frac{q_i k_j^T}{\sqrt{d_k}} - m \cdot (i - j)$$

其中 $(i - j)$ 是两个 Token 之间的相对距离，$m$ 是为各个注意力头预先设定的正数斜率。

![ALiBi 线性偏置在不同注意力头上的斜率惩罚衰减矩阵对比](/images/content-rebuild/a7b93b99f66ce028/fig-5.png)

### 5.2 斜率 $m$ 的设定规则
斜率 $m$ 不需要通过反向传播学习，而是在训练前按几何级数固定分配。对于包含 $H$ 个头的多头注意力，第 $h$ 个头（$h \in [1, H]$）的斜率为：

$$m_h = 2^{-\frac{8h}{H}}$$

以 8 个注意力头为例，斜率集合为 $\left[\frac{1}{2^1}, \frac{1}{2^2}, \dots, \frac{1}{2^8}\right] = \left[\frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}, \frac{1}{64}, \frac{1}{128}, \frac{1}{256}\right]$：
* **大斜率头（如 $m=0.5$）**：惩罚剧烈，稍微远离的词得分迅速变低，迫使该头专注于局部上下文。
* **小斜率头（如 $m=1/256$）**：惩罚非常微弱，允许该头在数千个 Token 的长范围内建立全局依赖。

### 5.3 为什么 ALiBi 能天然外推？
当模型在训练时只见过 2048 个词，而推理长度增加到 8192 时：
新出现的远端 Token 其相对距离 $(i - j)$ 很大，对应的偏置 $-m \cdot (i - j)$ 是一个绝对值很大的负数。经过 Softmax 后，这些未经训练见过的远端噪声 Token 权重被自然压缩至接近 0，不会破坏模型已学到的局部注意力分布，因此困惑度（PPL）表现非常平稳。

---

## 6. 代码机制实验：验证 RoPE 与 ALiBi

下面用一段简明的 Python（NumPy）代码，分别验证 RoPE 的相对内积性质以及 ALiBi 的偏置矩阵构造。

```python
import numpy as np

def apply_rope_2d(x: np.ndarray, pos: int, theta: float = 0.01) -> np.ndarray:
    """对二维向量在位置 pos 处应用正交旋转"""
    angle = pos * theta
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # 2x2 旋转矩阵
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])
    return R @ x

# 1. 验证 RoPE 的相对距离内积恒等性
q = np.array([1.0, 0.5])
k = np.array([0.8, -0.2])

# 场景 A: m = 5, n = 2 (相对距离 delta = 3)
q_rot_A = apply_rope_2d(q, pos=5)
k_rot_A = apply_rope_2d(k, pos=2)
dot_A = np.dot(q_rot_A, k_rot_A)

# 场景 B: m = 105, n = 102 (相对距离同样 delta = 3)
q_rot_B = apply_rope_2d(q, pos=105)
k_rot_B = apply_rope_2d(k, pos=102)
dot_B = np.dot(q_rot_B, k_rot_B)

print(f"RoPE 验证 (m-n=3):")
print(f"  位置对 (5, 2)   的点积: {dot_A:.6f}")
print(f"  位置对 (105, 102) 的点积: {dot_B:.6f}")
assert np.isclose(dot_A, dot_B), "RoPE 内积未能保持相对距离不变性！"

# 2. 构造 ALiBi 的因果偏置矩阵
def get_alibi_bias(seq_len: int, num_heads: int) -> np.ndarray:
    """生成形状为 [num_heads, seq_len, seq_len] 的因果线性偏置矩阵"""
    # 计算每个头的斜率
    slopes = np.array([2 ** (-8 * (h + 1) / num_heads) for h in range(num_heads)])
    
    # 生成距离矩阵 (i - j)
    idx = np.arange(seq_len)
    distance = idx[:, None] - idx[None, :]  # 下三角为正数，对角线为 0
    
    # 广播相乘并添加因果下三角约束（仅保留 i >= j 的历史相对距离）
    alibi_bias = -slopes[:, None, None] * distance[None, :, :]
    # 对于未来位置（j > i），通常叠加因果掩码 -inf，这里仅展示偏置本身
    return alibi_bias

biases = get_alibi_bias(seq_len=4, num_heads=2)
print("\nALiBi 偏置矩阵（头 0，大斜率局部关注）：")
print(np.round(biases[0], 2))
print("\nALiBi 偏置矩阵（头 1，小斜率长程关注）：")
print(np.round(biases[1], 2))
```

运行上述代码可以直观看到：
1. 无论绝对坐标处于第 5 步还是第 105 步，只要 Query 与 Key 的相对间隔相同，RoPE 处理后的点积结果严格一致。
2. ALiBi 在下三角区域随相对距离增加线性减少得分，斜率较大的头距离惩罚更重，斜率较小的头惩罚更轻。

---

## 7. 方案对比与工业界工程选型

![绝对位置编码与外推方案在超长上下文推理下的困惑度爆炸曲线](/images/content-rebuild/a7b93b99f66ce028/fig-6.png)

下表汇总了主流位置编码的技术方案与工程特性：

| 维度 | 可学习绝对位置 (Learned PE) | 正弦位置编码 (Sinusoidal) | 旋转位置编码 (RoPE) | 线性偏置注意力 (ALiBi) |
| :--- | :--- | :--- | :--- | :--- |
| **作用入口** | 词嵌入加法（输入层） | 词嵌入加法（输入层） | $Q, K$ 投影后（点积前） | 注意力分数矩阵（Logits 减法） |
| **额外参数量** | $L_{max} \times d$ | 0 | 0 | 0 |
| **编码性质** | 纯绝对物理坐标 | 绝对坐标（含理论移位性） | 相对距离内积表达 | 相对距离单调线性惩罚 |
| **直接外推能力** | 无（越界报错） | 极弱（分布漂移） | 较弱（未插值前低频分量分布外漂移） | **强**（无需插值直接长距离推理） |
| **长文本扩展手段** | 只能从头增加参数微调 | 难以直接平滑扩展 | **NTK-Aware 插值、YaRN 等** | 天然支持（但受限于结构衰减） |
| **代表模型** | BERT, GPT-2 | Transformer 原作, DETR | **LLaMA, Qwen, DeepSeek** | BLOOM, MPT |

### 为什么主流开源大模型大多选用 RoPE 而非 ALiBi？
虽然 ALiBi 具备出色的零样本长文本外推能力，但目前以 LLaMA、Qwen、DeepSeek 为代表的绝大多数开源大模型普遍采用了 RoPE。这背后的核心工程权衡在于：

1. **ALiBi 的“单调惩罚”制约了长程精准检索**：ALiBi 的线性偏置是硬性施加的，不能被模型动态撤销。在普通的语言建模（预测下一个词）中，这种距离衰减很有效；但在需要“在 64K 文本中精确定位某个变量定义”的**大海捞针（Needle In A Haystack）**任务中，远端关键信息的注意力分数被线性偏置压得极低，难以被模型精准聚焦。
2. **RoPE + 频率插值的上限更高**：RoPE 保持了查询与键之间完整的交互表达自由度。虽然 RoPE 在没有微调的情况下直接外推长文本效果有限，但工程上可以通过 **NTK-aware 插值**、**Dynamic YaRN** 等方法调整旋转基频，在极少量的长文本微调代价下，将 4K 训练的模型平滑拉伸至 32K、128K 乃至 1M 上下文，同时保持优异的长文本检索召回率。关于 RoPE 在 LLaMA 中的具体工程实现，可参考 [了解 LLaMA 中的旋转位置编码吗？](/interview/llm-interview/llama-rope.md)。

---

## 总结与课后检验

* **为什么需要位置编码**：自注意力矩阵具备置换等变性，纯点积计算无法感知序列的物理先后与相对间隔。
* **绝对位置编码**（Learned PE / Sinusoidal）将位置直接加在输入词向量上，深层网络会打碎相对位置关系，难以处理超出训练长度的序列。
* **RoPE** 利用正交分块旋转矩阵，使得 Query 和 Key 的点积仅由相对距离 $(m - n)$ 决定，不改变向量范数，结合频率插值具备极佳的长文本扩展能力。
* **ALiBi** 直接在注意力分数上按相对距离添加线性惩罚，零样本外推表现好，但单调衰减限制了长距离精确检索能力。

### 课后检验
1. **检验题目**：假设模型维度 $d_k=64$，基频定义为 $\theta_i = 10000^{-2i/d_k}$。当特征切片索引 $i=0$ 时，相邻两个位置（$\Delta pos = 1$）带来的旋转角度是多少？当 $i=31$（最高维切片）时，相邻两个位置的旋转角度又是多少？这说明了高维和低维在关注距离上有何区别？
2. **思考实践**：运行本文第 6 节的代码，尝试在 ALiBi 中将某个 Token 放在距离当前词 4096 的位置，计算其注意力 logits 惩罚值（对于斜率 $m=0.5$ 和 $m=1/256$ 两种情况）。对比两种头分别在多少距离时会产生超过 10.0 的 logit 压制？

完成本讲后，读者可进入下一篇课程学习架构级选型：[Decoder-only、Encoder-only、Encoder-Decoder 架构怎么区分？](/llm-100/basics/Decoder-only_Encoder-only_Encoder-Decoder_架构怎么区分.md)。

### 基础索引与前后关系
* 预备知识：[Self-Attention 和 Multi-Head Attention 有什么区别？](/llm-100/basics/Self-Attention_和_Multi-Head_Attention_有什么区别.md)
* 架构延伸：[Decoder-only、Encoder-only、Encoder-Decoder 架构怎么区分？](/llm-100/basics/Decoder-only_Encoder-only_Encoder-Decoder_架构怎么区分.md)
* 模型演化：[GPT、BERT、T5 的核心差异是什么？](/llm-100/basics/GPT_BERT_T5_的核心差异是什么.md)
* 专题面试：[了解 LLaMA 中的旋转位置编码吗？](/interview/llm-interview/llama-rope.md)
* 本组主目录：[基础篇](/llm-100/basics/README.md)
