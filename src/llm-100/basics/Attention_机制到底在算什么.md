---
title: Attention 机制到底在算什么？Q、K、V 矩阵计算与缩放原理
icon: mdi:book-open-page-variant-outline
date: 2026-07-16
tag:
- 大模型100讲
- 基础篇
description: 通过具体文字例子与矩阵算例，推导自注意力机制中 Q、K、V 投影、点积相似度、除以根号 dk 缩放及加权聚合的完整计算过程。
---

在上一讲[/llm-100/basics/Transformer_为什么能成为大模型的核心架构.md](/llm-100/basics/Transformer_为什么能成为大模型的核心架构.md)中，我们看到 Transformer 依靠自注意力机制实现了全序列位置的两两直接交互。本讲的目标是把这个交互过程拆解为具体的矩阵步骤，讲清 Query、Key、Value 是如何计算的，为什么点积结果必须除以 $\sqrt{d_k}$，并通过单步数值算例与代码完成检验。

先看一个直观例子：处理句子“吃 苹果”与“苹果 手机”。在这两个短语中，“苹果”的含义完全不同：
- 遇到“吃”时，模型需要找出动作的对象，应该把注意力放在“苹果（水果）”上；
- 遇到“手机”时，模型需要识别品牌或限定语，应该把注意力放在“苹果（公司）”上。

自注意力机制（Scaled Dot-Product Attention）的核心目标，就是让序列中的每个词，根据上下文其他词的信息来动态更新自己的向量表示。这个更新过程通过一套固定的矩阵计算完成：
$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V
$$

![单头点积自注意力计算流程全景图](/images/content-rebuild/5e17f06b5a485ecd/cover.png)

---

## 1. Q、K、V 的角色分工与线性变换

输入模型的一段文本首先被切分成 Token，每个 Token 对应一个初始向量。自注意力层不会直接拿初始向量去互相做乘法，而是先把它们投射为三种不同的角色：

- **Query（查询向量，$Q$）**：代表当前 Token “想寻找什么特征”。例如当处理“吃”时，它的 Query 倾向于寻找可食用的宾语。
- **Key（键向量，$K$）**：代表当前 Token “能提供什么特征”。例如“苹果”的 Key 包含“水果、数码产品”等维度的描述，供其他词来匹配。
- **Value（内容向量，$V$）**：代表当前 Token 承载的“实际信息内容”。一旦匹配度很高，模型就会把这部分信息提取出去。

这种结构类似于一个连续的加权检索系统：Query 与每一个 Key 比较，算出一个介于 0 到 1 之间的匹配分数，最后把所有 Token 的 Value 按照分数加权混合起来。

### 从输入矩阵到 Q、K、V
设输入序列包含 $L$ 个 Token，每个 Token 的向量维度为 $D$。整个输入构成矩阵 $X \in \mathbb{R}^{L \times D}$。

模型通过三个可学习的参数权重矩阵进行线性投影：
- $W^Q \in \mathbb{R}^{D \times d_k}$
- $W^K \in \mathbb{R}^{D \times d_k}$
- $W^V \in \mathbb{R}^{D \times d_v}$

计算公式如下：
$$
Q = X W^Q \quad (Q \in \mathbb{R}^{L \times d_k})
$$
$$
K = X W^K \quad (K \in \mathbb{R}^{L \times d_k})
$$
$$
V = X W^V \quad (V \in \mathbb{R}^{L \times d_v})
$$

在标准单头实现中，通常设定 $d_k = d_v$。通过这三个参数矩阵，同一个 Token 在作为“提问者”（$Q$）、“被检索者”（$K$）以及“信息提供者”（$V$）时，展现出不同的向量表征。

![Q、K、V 矩阵线性投影过程](/images/content-rebuild/5e17f06b5a485ecd/fig-1.png)

<ReadMoreLock />

---

## 2. 点积相似度矩阵：Token 之间的相关性计算

得到 $Q$ 与 $K$ 之后，需要衡量序列中任意两个 Token 之间的关联程度。自注意力采用向量内积（点积）来量化这种关联。

### 点积的几何意义
两个向量 $q$ 和 $k$ 的点积为：
$$
q \cdot k = \sum_{m=1}^{d_k} q_m k_m
$$
若两个向量在多个维度上符号相同且模长较大，点积为较大的正数；若两向量正交，点积为 0；若方向相反，点积为负数。因此，点积可以直接反映两个向量在表征空间中的方向一致性。

### 矩阵并行计算
为了同时算完序列中所有词对之间的关系，直接将 $Q$ 与 $K$ 的转置 $K^T$ 相乘，得到注意力得分矩阵 $S$：
$$
S = Q K^T \quad (S \in \mathbb{R}^{L \times L})
$$
其中第 $i$ 行第 $j$ 列的值 $S_{ij} = q_i \cdot k_j$，表示第 $i$ 个 Token 的 Query 与第 $j$ 个 Token 的 Key 的未缩放相似度。这是一个 $L \times L$ 的方阵，覆盖了序列内全部 $L^2$ 个两两配对。

![点积相似度矩阵生成计算图](/images/content-rebuild/5e17f06b5a485ecd/fig-2.png)

---

## 3. 缩放因子 $\frac{1}{\sqrt{d_k}}$ 的数学推导

拿到矩阵 $S$ 后，标准公式并不直接算 Softmax，而是先除以 $\sqrt{d_k}$。这步操作是为了控制方差，防止数值随向量维度增加而过度扩散。

### 方差推导假设
假设向量的分量满足以下统计假设：
1. 查询向量的分量 $q_m$ 与键向量的分量 $k_m$ 是相互独立的随机变量；
2. 各分量的均值为 0，方差为 1，即 $\mathbb{E}[q_m] = \mathbb{E}[k_m] = 0$，$\mathrm{Var}(q_m) = \mathrm{Var}(k_m) = 1$。

对于单项乘积 $Z_m = q_m k_m$：
$$
\mathbb{E}[Z_m] = \mathbb{E}[q_m] \mathbb{E}[k_m] = 0
$$
$$
\mathrm{Var}(Z_m) = \mathbb{E}[(q_m k_m)^2] - (\mathbb{E}[q_m k_m])^2 = \mathbb{E}[q_m^2] \mathbb{E}[k_m^2] - 0 = 1 \times 1 = 1
$$

点积得分 $S_{ij}$ 是 $d_k$ 个独立变量的和：
$$
S_{ij} = \sum_{m=1}^{d_k} Z_m
$$
根据独立随机变量求和的方差性质：
$$
\mathbb{E}[S_{ij}] = 0, \quad \mathrm{Var}(S_{ij}) = \sum_{m=1}^{d_k} \mathrm{Var}(Z_m) = d_k \times 1 = d_k
$$

**结论**：点积的方差与维度 $d_k$ 成正比，标准差为 $\sqrt{d_k}$。
- 当 $d_k = 64$ 时，标准差为 8；
- 当 $d_k = 128$ 时，标准差约为 11.31。

维度越大，点积算出的最大值与最小值差距就越悬殊。引入缩放因子后：
$$
\mathrm{Var}\left( \frac{S_{ij}}{\sqrt{d_k}} \right) = \frac{1}{(\sqrt{d_k})^2} \mathrm{Var}(S_{ij}) = \frac{1}{d_k} \times d_k = 1
$$
除以 $\sqrt{d_k}$ 能在理论假设下把点积方差重置为 1，避免分数因特征维度变宽而剧烈发散。

![缩放因子对点积分布方差的影响机制](/images/content-rebuild/5e17f06b5a485ecd/fig-3.png)

---

## 4. Softmax 归一化与梯度饱和

缩放后的分数通过 Softmax 转化为概率分布：
$$
A = \mathrm{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) \quad (A \in \mathbb{R}^{L \times L})
$$
Softmax 按行独立计算。对于第 $i$ 行，第 $j$ 列的注意力权重为：
$$
A_{ij} = \frac{\exp\left( \frac{S_{ij}}{\sqrt{d_k}} \right)}{\sum_{t=1}^L \exp\left( \frac{S_{it}}{\sqrt{d_k}} \right)}
$$
计算后的权重矩阵满足：
1. 每个元素大于 0：$A_{ij} > 0$
2. 每行之和为 1：$\sum_{j=1}^L A_{ij} = 1$

### 为什么不做缩放会导致梯度消失？
Softmax 函数对输入的偏导数为：
$$
\frac{\partial A_i}{\partial z_j} = A_i (\delta_{ij} - A_j)
$$
其中 $\delta_{ij}$ 为克罗内克符号（$i=j$ 时为 1，其余为 0）。

如果不进行缩放，当 $d_k$ 很大时，点积数值的极大差距经过 $\exp(\cdot)$ 放大后，Softmax 会输出一个极尖锐的分布——最大值位置接近 1，其余位置极接近 0：
- 对于最大值位置：导数约为 $1 \times (1 - 1) = 0$；
- 对于其他位置：导数约为 $0 \times (0 - 0) = 0$。

此时 Softmax 处在导数极小的饱和区，反向传播时传回的梯度几乎衰减为 0，导致注意力权重以及前端投影参数无法正常更新。关于激活函数饱和的通性分析可阅读[/fundamentals/dl-basics/activation-functions.md](/fundamentals/dl-basics/activation-functions.md)与[/fundamentals/dl-basics/gradient-vanishing-exploding.md](/fundamentals/dl-basics/gradient-vanishing-exploding.md)。

![Softmax 梯度饱和与反向传播截断对比](/images/content-rebuild/5e17f06b5a485ecd/fig-4.png)

---

## 5. 加权求和：生成融合上下文的新向量

有了归一化权重矩阵 $A$，最后一步是用它对 Value 矩阵进行加权求和：
$$
Z = A V \quad (Z \in \mathbb{R}^{L \times d_v})
$$
对于第 $i$ 个 Token，其更新后的向量表示 $z_i \in \mathbb{R}^{d_v}$ 为：
$$
z_i = \sum_{j=1}^L A_{ij} v_j
$$

因为 $\sum_{j=1}^L A_{ij} = 1$ 且 $A_{ij} > 0$，$z_i$ 是全序列所有 Token 内容向量 $v_j$ 的凸组合：
- 如果“吃”对“苹果”的注意力权重 $A_{吃, 苹果} = 0.85$；
- 那么“吃”更新后的向量 $z_{吃}$ 中将包含 85% 来自“苹果”的内容向量 $v_{苹果}$；
- 经过这一层，原本独立的词向量就融合了上下文的语法与语义信息。

![注意力权重对 Value 矩阵的线性加权重组](/images/content-rebuild/5e17f06b5a485ecd/fig-5.png)

---

## 6. 手工算例与 PyTorch 实验

为了验证上述步骤，我们设定一个极简算例：序列长度 $L=2$，维度 $d_k=d_v=2$。

### 单步数值演算
假设投影后的张量为（数值为教学示意）：
$$
Q = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}, \quad K = \begin{bmatrix} 1.0 & 1.0 \\ 0.0 & 1.0 \end{bmatrix}, \quad V = \begin{bmatrix} 2.0 & 0.0 \\ 1.0 & 3.0 \end{bmatrix}
$$

1. **转置 $K$**：
   $$K^T = \begin{bmatrix} 1.0 & 0.0 \\ 1.0 & 1.0 \end{bmatrix}$$
2. **计算未缩放得分 $S = Q K^T$**：
   $$S = \begin{bmatrix} 1.0 \times 1.0 + 0.0 \times 1.0 & 1.0 \times 0.0 + 0.0 \times 1.0 \\ 0.0 \times 1.0 + 1.0 \times 1.0 & 0.0 \times 0.0 + 1.0 \times 1.0 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 \\ 1.0 & 1.0 \end{bmatrix}$$
3. **缩放得分（$\sqrt{d_k} = \sqrt{2} \approx 1.414$）**：
   $$\frac{S}{\sqrt{2}} \approx \begin{bmatrix} 0.707 & 0.000 \\ 0.707 & 0.707 \end{bmatrix}$$
4. **逐行计算 Softmax 权重 $A$**：
   - 第一行：$\exp(0.707) \approx 2.028$, $\exp(0) = 1.000$，和为 $3.028$。归一化得 $[0.670, 0.330]$
   - 第二行：两值相同，归一化得 $[0.500, 0.500]$
   $$A = \begin{bmatrix} 0.670 & 0.330 \\ 0.500 & 0.500 \end{bmatrix}$$
5. **计算输出 $Z = A V$**：
   - 第一行：$0.670 \times [2.0, 0.0] + 0.330 \times [1.0, 3.0] = [1.340 + 0.330, 0.000 + 0.990] = [1.670, 0.990]$
   - 第二行：$0.500 \times [2.0, 0.0] + 0.500 \times [1.0, 3.0] = [1.000 + 0.500, 0.000 + 1.500] = [1.500, 1.500]$
   $$Z = \begin{bmatrix} 1.670 & 0.990 \\ 1.500 & 1.500 \end{bmatrix}$$

### PyTorch 梯度对比实验
以下代码运行缩放与不缩放两种模式，观察注意力的熵（分布平缓度）和传回到 $Q$ 的梯度平均模长：

```python
import torch
import torch.nn.functional as F

def run_attention_experiment():
    # 设定较大维度，模拟大模型单头的典型特征维度
    L, d_k = 4, 128
    torch.manual_seed(42)
    
    Q = torch.randn(L, d_k, requires_grad=True)
    K = torch.randn(L, d_k, requires_grad=True)
    V = torch.randn(L, d_k, requires_grad=True)
    
    # 1. 正常缩放模式
    scores_scaled = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attn_scaled = F.softmax(scores_scaled, dim=-1)
    out_scaled = torch.matmul(attn_scaled, V)
    out_scaled.sum().backward()
    grad_scaled = Q.grad.abs().mean().item()
    
    # 清理梯度
    Q.grad.zero_()
    
    # 2. 故意省略缩放模式
    scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))
    attn_unscaled = F.softmax(scores_unscaled, dim=-1)
    out_unscaled = torch.matmul(attn_unscaled, V)
    out_unscaled.sum().backward()
    grad_unscaled = Q.grad.abs().mean().item()
    
    entropy_scaled = (-attn_scaled * torch.log(attn_scaled + 1e-9)).sum(-1).mean().item()
    entropy_unscaled = (-attn_unscaled * torch.log(attn_unscaled + 1e-9)).sum(-1).mean().item()
    
    print(f"缩放模式注意力熵: {entropy_scaled:.4f} | Q 平均梯度模长: {grad_scaled:.6f}")
    print(f"未缩放注意力熵:   {entropy_unscaled:.4f} | Q 平均梯度模长: {grad_unscaled:.6f}")

run_attention_experiment()
```

运行结果会显示：未缩放时注意力熵显著更低（概率集中在单一 Token），反向传给 $Q$ 的梯度平均模长也显著减小。

![单头自注意力张量流动与维度校验汇总](/images/content-rebuild/5e17f06b5a485ecd/fig-6.png)

---

## 7. 适用条件与局限

使用自注意力机制时需要注意以下三个特性：

1. **置换不变性（Permutation Invariance）**：单纯的矩阵点积运算不包含位置信息。如果在输入中打乱 Token 的顺序，计算出来的得分矩阵只是发生对应的行列重排，模型本身无法分辨词语的前后顺序。因此必须引入[位置编码](/llm-100/basics/位置编码为什么重要_RoPE_ALiBi_分别解决什么问题.md)来注入次序。
2. **计算与显存开销**：生成矩阵 $S = Q K^T$ 需要进行 $O(L^2 \cdot d_k)$ 次乘加运算，且注意力方阵需要 $O(L^2)$ 显存空间。序列长度翻倍，注意力方阵的显存消耗翻 4 倍。在 Prefill 阶段需要一次性计算全序列注意力，而在逐字生成（Decode）阶段则主要依赖 KV Cache 避免重复计算历史 Key 和 Value。
3. **单头表征的单一性**：单头注意力在每一层只产生一个 $L \times L$ 的注意力矩阵，意味着每个词在同一层只能聚焦一种关系（例如语法依存或实体指代）。为了同时关注不同角度的语义联系，Transformer 采用了多头注意力机制，这将在下一讲[/llm-100/basics/Self-Attention_和_Multi-Head_Attention_有什么区别.md](/llm-100/basics/Self-Attention_和_Multi-Head_Attention_有什么区别.md)中展开讨论。

---

## 课后练习

1. 设 $Q = \begin{bmatrix} 2.0 & 0.0 \end{bmatrix}$，$K = \begin{bmatrix} 0.0 & 2.0 \\ 2.0 & 0.0 \end{bmatrix}$，$V = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}$，特征维度 $d_k = 2$。
   - 请手工计算 $Q K^T$、缩放后数值、Softmax 归一化权重及最终输出向量 $Z$。
2. 在 PyTorch 实验代码中，将序列长度 $L$ 改为 16，特征维度 $d_k$ 改为 512，观察未缩放时的梯度模长衰减比例与 $d_k=128$ 时有何不同。
