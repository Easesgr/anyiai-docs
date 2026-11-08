---
title: LLaMA 中的旋转位置编码（RoPE）是怎么工作的？
icon: mdi:rotate-3d-variant
tag:
- LLaMA
- RoPE
- 位置编码
description: 详解 RoPE 的口述答案、复数旋转推导、向量化实现、长程衰减机制与上下文扩展方案。
---

面试官问到“LLaMA 为什么使用 RoPE（旋转位置编码）”时，通常想考察四个递进的问题：
1. 能不能用最通俗的语言讲清为什么“旋转操作”能让绝对位置变成相对位置？
2. 能否在二维复平面上推导它的点积公式？
3. 在工程实现中，几百维的高维向量是如何不用大矩阵乘法、直接用逐元素操作高效算出来的？
4. 当文本长度超过训练长度时，为什么直接推理会崩溃，位置插值（PI）和 YaRN 又是怎么解决的？

![旋转位置编码核心机制与上下文扩展知识框架图](/images/content-rebuild/9d4010aacd832940/cover.png)

## 面试 30 秒口述答案

**核心结论**：  
RoPE（Rotary Position Embedding）通过将 Query 和 Key 向量旋转一个与其绝对位置成正比的角度，使得两者的点乘内积天然只依赖于相对距离 $(m - n)$。

**怎么做到的**：
- **注入位置**：RoPE 不在模型输入层直接做词向量相加，而是在每一层计算注意力前，对线性变换后的 $Q$ 和 $K$ 施加二维分块的正交旋转变换。
- **兼具两端优势**：它保留了绝对位置编码的优势（每个 Token 的 Key 算完后可以直接写入 KV Cache，不需要随新 Token 输入而更新历史向量），同时兼具相对位置编码的表达能力（注意力权重只受相对距离影响，且随距离增加呈现总体衰减趋势）。

---

## 贯穿例子：为什么“转圈”能抵消绝对位置？

想象两个指针向量 $q$ 和 $k$。如果它们都在原点：
- 词 $A$ 位于序列的第 $2$ 个位置，我们把向量 $q$ 逆时针旋转 $2\theta$ 角度；
- 词 $B$ 位于序列的第 $5$ 个位置，我们把向量 $k$ 逆时针旋转 $5\theta$ 角度。

当这两个向量做点积时，几何上点积等于模长乘积乘以两者的**夹角余弦**。
此时它们的夹角是 $5\theta - 2\theta = 3\theta$。

如果文本整体后移：词 $A$ 移动到第 $102$ 个位置（旋转 $102\theta$），词 $B$ 移动到第 $105$ 个位置（旋转 $105\theta$）。
它们做点乘时，夹角依然是 $105\theta - 102\theta = 3\theta$。

![二维复平面上 Query 与 Key 旋转及相对内积夹角几何示意图](/images/content-rebuild/9d4010aacd832940/fig-1.png)

绝对位置数字 $102$ 和 $105$ 虽然很大，但通过向量相角相减，在点乘结果中被直接抵消，留下的正是相对位移差 $3$。

---

## 原理推导：从内积目标到二维旋转矩阵

在自注意力计算中，位置 $m$ 的 Query 向量 $q_m$ 与位置 $n$ 的 Key 向量 $k_n$ 计算内积：
$$\text{Score} = \langle f(q, m), f(k, n) \rangle$$

我们希望设计一个函数 $f$，输入是单向量和绝对位置，输出后的内积却只与相对位置差 $(m - n)$ 相关：
$$\langle f(q, m), f(k, n) \rangle = g(q, k, m - n)$$

### 1. 二维复平面上的解

把二维实数向量 $[x_1, x_2]^T$ 对应到复数平面：$x = x_1 + i x_2$。
设赋予位置 $m$ 的操作为乘以一个单位旋转复数 $e^{i m \theta}$（根据欧拉公式，$e^{i m \theta} = \cos(m\theta) + i \sin(m\theta)$）：
$$f(q, m) = q \cdot e^{i m \theta}$$
$$f(k, n) = k \cdot e^{i n \theta}$$

在复数表示下，两个二维实向量的欧氏内积等价于第一个复数与第二个复数共轭之积的实部：
$$\langle f(q, m), f(k, n) \rangle = \text{Re}\left( f(q, m) \cdot f(k, n)^* \right)$$
其中 $*$ 表示共轭复数（虚部变号）。代入公式计算：
$$\text{Re}\left( (q \cdot e^{i m \theta}) \cdot (k \cdot e^{i n \theta})^* \right) = \text{Re}\left( q k^* \cdot e^{i m \theta} \cdot e^{-i n \theta} \right) = \text{Re}\left( q k^* \cdot e^{i (m - n) \theta} \right)$$

指数部分合并后，$m$ 与 $n$ 变成了 $(m - n)$。这意味着点积结果完全由相对位移差决定。

<ReadMoreLock />

### 2. 写成实数旋转矩阵

复数乘法在实数向量空间中对应标准二维旋转矩阵：
$$(x_1 + i x_2)(\cos(m\theta) + i \sin(m\theta)) = (x_1 \cos(m\theta) - x_2 \sin(m\theta)) + i (x_1 \sin(m\theta) + x_2 \cos(m\theta))$$

写成矩阵与向量相乘形式：
$$f(x, m) = R_{\theta, m} x = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

因为旋转矩阵是正交矩阵，满足转置等于逆矩阵 $R_{\theta, m}^T = R_{\theta, -m}$，两向量内积为：
$$(R_{\theta, m} q)^T (R_{\theta, n} k) = q^T R_{\theta, m}^T R_{\theta, n} k = q^T R_{\theta, n - m} k$$
这就是相对位置在实数矩阵下的直接表达。

---

## 工程实现：高维向量如何高效旋转？

在真实大模型中，每个注意力头的维度 $d$ 通常为 64 或 128（例如 LLaMA-7B 的注意力头维度 $d = 128$），并不是二维。

### 分块正交变换

RoPE 并没有构造一个巨大的 $128 \times 128$ 密集稠密矩阵，而是把 128 维向量拆分成 64 个两两一组的二维子空间，每一组分配一个独立的基频 $\theta_i$：
$$R_{\Theta, m} = \begin{pmatrix} R_{\theta_1, m} & & & 0 \\ & R_{\theta_2, m} & & \\ & & \ddots & \\ 0 & & & R_{\theta_{d/2}, m} \end{pmatrix}$$

第 $i$ 组的基频设定沿用了 Transformer 正弦编码的设计：
$$\theta_i = b^{-2(i - 1)/d}, \quad i \in \{1, 2, \dots, d/2\}$$
在 LLaMA 中，底数 $b$ 初始默认值为 $10000$。

![RoPE 高维分块正交变换与向量元素重排计算流](/images/content-rebuild/9d4010aacd832940/fig-2.png)

### 逐元素向量化替代稀疏矩阵乘法

如果在 GPU 上真正实例化这些块对角矩阵再做矩阵乘法，显存和带宽开销很大。工程实现利用三角函数的分配律，直接化简为逐元素向量操作：
$$R_{\Theta, m} x = x \odot \cos(m\Theta) + \tilde{x} \odot \sin(m\Theta)$$

其中 $\tilde{x}$ 是将向量分组成对后取负与换位的结果。例如：
- 若按相邻两两配对，$\tilde{x} = [-x_2, x_1, -x_4, x_3, \dots]$；
- 在 Hugging Face Transformers 实现中，为了内存读写连续，通常把前半段和后半段配对：$\tilde{x} = [-x_{d/2:}, x_{:d/2}]$。

**PyTorch 最小可运行验证代码**：

```python
import torch

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将向量后半部分取负，并与前半部分互换"""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    q, k 形状: [batch_size, seq_len, num_heads, head_dim]
    cos, sin 形状: [1, seq_len, 1, head_dim]
    """
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

# 验证维度与计算
seq_len, head_dim = 4, 8
positions = torch.arange(seq_len).unsqueeze(-1)  # [seq_len, 1]
# 频率计算
theta = 10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim)  # [head_dim / 2]
freqs = torch.outer(positions.squeeze(1), theta)  # [seq_len, head_dim / 2]
emb = torch.cat((freqs, freqs), dim=-1)  # 拼成 [seq_len, head_dim]
cos = emb.cos()[None, :, None, :]
sin = emb.sin()[None, :, None, :]

q = torch.randn(1, seq_len, 1, head_dim)
k = torch.randn(1, seq_len, 1, head_dim)
q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)

# 验证模长保持不变（旋转前后模长一致）
assert torch.allclose(torch.norm(q, dim=-1), torch.norm(q_out, dim=-1), atol=1e-5)
```

---

## 距离衰减特性：为什么越远的词关注度越低？

许多位置编码需要人工添加距离衰减惩罚（如 ALiBi），而 RoPE 天然具备长程衰减的几何特性。

整个高维注意力内积等于所有二维子空间内积的叠加：
$$\langle R_{\Theta, m} q, R_{\Theta, n} k \rangle = \text{Re} \sum_{i=1}^{d/2} (q^{(i)} {k^{(i)}}^*) e^{i(m - n)\theta_i}$$

![不同频率维度在序列距离增加时的注意力衰减曲线对比](/images/content-rebuild/9d4010aacd832940/fig-3.png)

这种衰减源于多组不同频率波形的相互干涉：
1. **高频分量（$\theta_i$ 较大）**：波长短，相对距离稍微增加就会快速正负振荡。多维求和时由于相位杂乱相互抵消，均值迅速收敛至零，负责区分紧挨着的近邻 Token。
2. **低频分量（$\theta_i$ 极小）**：波长长，旋转速度极慢，能够在较长跨度内保持较平稳的内积响应，负责跨句的长程语义关联。
3. **综合衰减**：由级数求和性质（类似于 Abel 变换与狄利克雷核），多组不同频率的正弦余弦叠加后，整体包络线随相对距离 $|m - n|$ 增加而呈现衰减趋势。模型无须人工规则截断，就能对临近上下文保持更高的敏锐度。

---

## 长文本扩展：直接外推为什么会失效？

当模型在 2048 长度下训练完成，直接输入 4096 长度的文本时（称为**直接外推**），模型的困惑度（Perplexity）通常会急剧上升而输出胡言乱语。

**失效原因**：  
当 $m > 2048$ 时，尤其是低频维度的旋转角度 $m\theta$ 超过了模型在训练期见过的最大角度。注意力分数矩阵中的数值分布完全脱离了预训练的数值范围，Softmax 无法形成有意义的注意力概率分布。

![直接外推、位置插值（PI）与 YaRN 动态缩放的波形分布对比](/images/content-rebuild/9d4010aacd832940/fig-4.png)

为了将上下文扩展到更长窗口，主流有两种解决方案：

### 1. 位置插值（Position Interpolation, PI）

设要将长度扩展 $s$ 倍（如从 2048 扩展到 4096，则缩放因子 $s = 2$）。  
位置插值直接将推理时的位置序号缩小：
$$m' = \frac{m}{s}$$
- **原理**：把 0 到 4096 的位置强制压缩进预训练见过的 $[0, 2048]$ 坐标区间，相角不再越界。
- **代价**：所有维度的频率被等比例拉长。高频维度的波长也被放大了 $s$ 倍，导致模型区分相邻两个连续 Token 的位置分辨率下降，通常需要几百步微调让模型适应这种压缩。

### 2. YaRN（分频自适应插值）

YaRN 改进了位置插值的“一刀切”压缩，按维度频率进行分类处理：
- **高频维度不插值**：波长远小于预训练窗口的维度，在预训练中已经完整旋转过许多个周期，具有最关键的局部细节辨识能力，因此**保持原有频率不变**（缩放因子为 1）。
- **低频维度全量插值**：波长远大于窗口的维度，模型未见过完整周期，按线性位置插值缩小 $s$ 倍。
- **中频维度过渡**：在两端阈值之间使用光滑权重混合插值。
- **注意力熵校准**：序列变长后，Softmax 分布容易散开。YaRN 乘以一个温度修正因子以恢复原本的注意集中度。

---

## 面试追问与常见误区

![传统绝对位置编码与 RoPE 在 Transformer 架构中的注入位置流向对比](/images/content-rebuild/9d4010aacd832940/fig-5.png)

### 追问 1：RoPE 能不能加在 Transformer 最底层的输入词嵌入上？

**答案**：不能。必须加在每一层线性投影之后的 Query 和 Key 上。

**原因解释**：
1. **网络层变换会破坏相对旋转关系**：如果加在输入层，向量随后要经过多层网络（包括线性投影矩阵 $W_Q, W_K$ 以及前馈网络非线性激活函数 SwiGLU）。通用矩阵投影和非线性变换均无法保持旋转矩阵的正交内积不变性，相角会被打乱，无法在后续注意力内积中推导出 $(m - n)$ 相对位移。
2. **Value 向量无需旋转**：RoPE 的目的是在 Query 和 Key 计算相关度时引入相对距离感知；Value 向量代表 Token 本身的语义特征，直接参与加权求和，不需要叠加位置旋转变换。

![自回归推理时 RoPE 在 Prefill 与 Decode 阶段对 KV Cache 的作用方式](/images/content-rebuild/9d4010aacd832940/fig-6.png)

### 追问 2：RoPE 对 KV Cache 推理有什么影响？会改变向量模长吗？

**答案**：
1. **保持模长不变**：旋转矩阵是正交矩阵（$R^T R = I$），正交变换具有保范性，即 $\|R_{\Theta, m} x\|_2 = \|x\|_2$。它只改变向量朝向，绝不改变向量模长，数值稳定性好。
2. **对 KV Cache 完全兼容**：在生成第 $m$ 个 Token 时，当前 Token 对应的 Key 向量按位置 $m$ 旋转一次后直接存入 KV Cache。历史缓存中第 $n$ 个位置的 Key 已经包含旋转，不需要随步数增加重新更新，单步推理依然保持增量复杂度。

### 常见错误说法辨析

| 常见错误说法 | 真实机制纠正 |
| :--- | :--- |
| “RoPE 是相对位置编码，所以它抛弃了绝对位置。” | **错误**。RoPE 本身是将绝对位置序号 $m$ 作用在单个向量上；只是由于正交旋转的内积性质，点积展开后自然等价于相对位置差。它在实现上是绝对编码，在效果上是相对编码。 |
| “RoPE 计算涉及大量复数矩阵，训练推理开销很大。” | **错误**。通过向量切半翻转（`rotate_half`）和逐元素广播乘加，完全用实数算子实现，无须构造稀疏矩阵或调用复数运算库。 |
| “RoPE 具有自然衰减性，所以模型天然支持任意长度零样本外推。” | **错误**。超出预训练最大位置后，未见过的相角差会导致注意力分布异常（分布外泛化失败）。长文本必须配合位置插值（PI）、YaRN 等方法，并进行针对性的上下文长文本微调。 |
