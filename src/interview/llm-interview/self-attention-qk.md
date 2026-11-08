---
title: 自注意力中 Q 和 K 为什么要做点积？为什么要除以根号 d_k？
icon: mdi:key
tag:
- Self-Attention
- QKV
- Transformer
description: 从向量几何意义、独立变量方差推导到 Softmax 饱和区梯度消失，解析自注意力机制中 QK 点积与缩放因子的数学逻辑与工程原因。
---

在自注意力机制（Self-Attention）的面试中，面试官通常会从基本概念切入，最终落脚在两个连环追问上：**为什么衡量查询与键的相似度要用点积？为什么点积之后必须除以 $\sqrt{d_k}$？**

要答清楚这两个问题，需要讲清向量点积的几何含义、随机变量求和的方差累加过程，以及 Softmax 函数在输入极差过大时的梯度表现。

![QK点积计算与根号d缩放因子在自注意力机制中的数学平衡](/images/content-rebuild/b6d2192c188cf1b3/cover.png)

---

## 面试 30 秒口述答案

面试时可以按以下四个层次作答：

1. **点积的作用**：查询向量 $q$ 与键向量 $k$ 做点积，能同时衡量两者的**方向对齐程度**（余弦夹角）与**特征重要性**（模长大小）。在工程上，点积对应标准的矩阵乘法（GEMM），在 GPU 上执行效率极高。
2. **方差线性膨胀**：假设 $q$ 与 $k$ 的各个分量独立同分布，均值为 0，方差为 1。它们内积求和共有 $d_k$ 项相加。根据独立随机变量求和的方差性质，内积结果的方差会线性放大到 $d_k$，标准差膨胀为 $\sqrt{d_k}$。
3. **Softmax 饱和与梯度消失**：若不除以 $\sqrt{d_k}$，随着维度 $d_k$ 增大，点积数值的离散程度（极差）迅速拉大。输入 Softmax 后，最大项概率趋近于 1，其余项趋近于 0，分布近似 One-hot。此时 Softmax 的导数极小，反向传播时注意力参数拿不到更新梯度，导致模型训练受阻。
4. **缩放因子的归一化效果**：点积除以 $\sqrt{d_k}$ 后，得分的方差被重新拉回 1，使输入数值落在 Softmax 导数敏感的有效区间，兼顾了注意力区分度与训练梯度流动。

![自注意力机制中QK点积方差膨胀与除以根号d的缩放控制数据流](/images/content-rebuild/b6d2192c188cf1b3/fig-1.png)

---

## 贯穿例子：从三个词的匹配直观理解点积与缩放

为了直观看到计算过程，假设隐层维度 $d_k = 4$。序列中有三个词：当前词 $q$（查询）以及两个候选上下文词 $k_1, k_2$（键）。

假设向量取值如下：
- 查询向量：$q = [1.0, 1.0, -1.0, 0.0]$
- 候选词 1 的键：$k_1 = [1.0, 1.0, -1.0, 0.0]$（与 $q$ 完全同向，语义高度相关）
- 候选词 2 的键：$k_2 = [0.0, -1.0, 0.0, 1.0]$（与 $q$ 夹角大于 90 度，负相关）

计算点积得分：
- $z_1 = q \cdot k_1 = 1\times 1 + 1\times 1 + (-1)\times (-1) + 0\times 0 = 3.0$
- $z_2 = q \cdot k_2 = 1\times 0 + 1\times (-1) + (-1)\times 0 + 0\times 1 = -1.0$

两者的点积差值为 $3.0 - (-1.0) = 4.0$。当维度 $d_k$ 只有 4 时，数值差别并不算极端。

但如果维度扩大到主流大模型的单头尺寸（例如 $d_k = 64$ 或 $d_k = 128$），在类似的分量分布下，累加项由 4 项变成 64 项或 128 项，两个点积的差值很容易达到 15 到 30 以上。

- 当差值为 4 时：$\frac{e^3}{e^{-1}} = e^4 \approx 54.6$，Softmax 仍给 $k_2$ 分配了一定注意力权重；
- 当差值放大到 20 时：$e^{20} \approx 4.85 \times 10^8$，$k_2$ 的权重接近绝对的 0，模型失去了在多个词之间平衡分配权重的能力。

除以 $\sqrt{d_k}$，正是为了让 64 维甚至 128 维下的内积数值尺度，重新回到 1 维到 4 维这种温和的离散范围之内。

---

## 为什么相似度选用点积

自注意力机制将匹配方式设计为向量内积（Dot-product）：

$$z = q \cdot k = \sum_{i=1}^{d_k} q_i k_i = \|q\| \|k\| \cos\theta$$

选用点积主要出于几何表达和工程效率两个层面的考虑：

1. **兼顾方向夹角与模长权重**：
   - $\cos\theta$ 代表方向匹配程度。夹角越小（越接近 0 度），$\cos\theta$ 越接近 1，表示两者语义越契合。
   - 向量模长 $\|q\|$ 与 $\|k\|$ 代表特征的显著程度。未归一化的点积允许模型通过学习更大的向量范数，赋予某些关键 Token（例如句首特殊标记或核心实体）天然更高的全局关注度，保留了表达自由度。
2. **硬件矩阵乘法友好**：
   - 序列中所有位置的 Query 与 Key 计算可以写成矩阵相乘 $Q K^T$。
   - 现代 GPU 与专用 AI 芯片对密集矩阵乘法（GEMM）拥有高度优化的硬件计算单元（如 Tensor Core），吞吐量远高于逐元素计算或非线性多层感知机网络。

关于 $Q$、$K$、$V$ 在注意力机制中的分工细节，可参考 [/interview/llm-interview/attention-qkv.md](/interview/llm-interview/attention-qkv.md)；关于点积注意力与加法注意力的详细工程权衡，可参考 [/interview/llm-interview/attention-dot-vs-add.md](/interview/llm-interview/attention-dot-vs-add.md)。

![高维空间中查询向量Q与键向量K的点积几何投影与模长关系](/images/content-rebuild/b6d2192c188cf1b3/fig-2.png)

<ReadMoreLock />

---

## 方差放大推导：为什么内积方差是 d_k

很多初学者容易记混概念，误以为“点积的方差是 $\sqrt{d_k}$”。实际上，**方差是 $d_k$，标准差才是 $\sqrt{d_k}$**。下面给出详细统计推导。

### 假设条件
设查询向量 $q \in \mathbb{R}^{d_k}$ 和键向量 $k \in \mathbb{R}^{d_k}$。在模型参数随机初始化或经过层归一化后，假设 $q$ 与 $k$ 的每个分量 $q_i, k_i$ 满足均值为 0、方差为 1 的独立同分布（i.i.d.）：

$$\mathbb{E}[q_i] = 0, \quad \mathrm{Var}(q_i) = 1$$
$$\mathbb{E}[k_i] = 0, \quad \mathrm{Var}(k_i) = 1$$

且假设 $q$ 与 $k$ 的各分量之间相互独立。

### 单项乘积的均值与方差
令单项随机变量 $Z_i = q_i k_i$。

由于 $q_i$ 与 $k_i$ 独立，单项乘积的数学期望为：

$$\mathbb{E}[Z_i] = \mathbb{E}[q_i k_i] = \mathbb{E}[q_i] \mathbb{E}[k_i] = 0 \times 0 = 0$$

根据方差计算公式 $\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$，先求 $Z_i$ 的二阶原点矩：

$$\mathbb{E}[Z_i^2] = \mathbb{E}[(q_i k_i)^2] = \mathbb{E}[q_i^2 k_i^2]$$

由独立性可知，$\mathbb{E}[q_i^2 k_i^2] = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2]$。
因为 $\mathbb{E}[q_i] = 0$，所以 $\mathbb{E}[q_i^2] = \mathrm{Var}(q_i) + (\mathbb{E}[q_i])^2 = 1 + 0 = 1$。同理 $\mathbb{E}[k_i^2] = 1$。代入得：

$$\mathbb{E}[Z_i^2] = 1 \times 1 = 1$$
$$\mathrm{Var}(Z_i) = \mathbb{E}[Z_i^2] - (\mathbb{E}[Z_i])^2 = 1 - 0 = 1$$

结论：**两个均值为 0、方差为 1 的独立随机变量相乘，所得乘积的均值仍为 0，方差仍为 1。**

### 点积累加的总方差
点积标量 $z = q \cdot k = \sum_{i=1}^{d_k} Z_i$。

因为向量各分量相互独立，各分量乘积 $Z_i$ 彼此独立。根据独立随机变量和的方差性质：

$$\mathbb{E}[z] = \sum_{i=1}^{d_k} \mathbb{E}[Z_i] = 0$$
$$\mathrm{Var}(z) = \mathrm{Var}\left( \sum_{i=1}^{d_k} Z_i \right) = \sum_{i=1}^{d_k} \mathrm{Var}(Z_i) = \sum_{i=1}^{d_k} 1 = d_k$$

这就证明了：**未缩放的点积结果 $z$，其均值为 0，方差为 $d_k$。其标准差 $\sigma_z = \sqrt{\mathrm{Var}(z)} = \sqrt{d_k}$。**

需要说明的是，真实神经网络训练过程中，各特征通道并不完全独立，均值和方差也会动态变化；上述推导是基于初始化和理想分布的标准理论分析，它准确指出了“维度相加导致方差按项数线性累加”的数学因果。

![独立分量点积相加导致总方差线性累积至dk的推导流程](/images/content-rebuild/b6d2192c188cf1b3/fig-3.png)

---

## Softmax 的饱和区灾难：为什么大方差会导致训练受阻

当单头维度 $d_k = 64$ 时，点积的标准差达到 $\sqrt{64} = 8$；当 $d_k = 128$ 时，标准差达到约 $11.3$。
这意味着输入给 Softmax 的不同位置分数很容易相差 20 到 30 以上。这种大方差输入会给反向传播带来致命问题。

### 输出分布退化
对于查询 $q$，其对所有上下文位置的键计算出的得分向量为 $\mathbf{z} = [z_1, z_2, \dots, z_n]$，经过 Softmax 得到注意力概率：

$$S_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}$$

当输入向量的分散程度极大时，得分最高的分量 $z_{\max}$ 与次高分量 $z_{\text{second}}$ 的差距如果达到 16：

$$\frac{e^{z_{\max}}}{e^{z_{\text{second}}}} = e^{z_{\max} - z_{\text{second}}} = e^{16} \approx 8.88 \times 10^6$$

此时 Softmax 输出分布会退化为一个极端的近乎 One-hot 的分布：

$$S_{\max} \approx 1, \quad \forall j \neq \max, \, S_j \approx 0$$

模型失去了对整个上下文按权重进行综合汇聚（Soft Attention）的能力，退化为只抽取单点信息。

### 反向传播梯度消失
Softmax 函数相对于输入 logits $z_j$ 的偏导数（即雅可比矩阵）计算公式为：

$$\frac{\partial S_i}{\partial z_j} = \begin{cases} S_i(1 - S_i), & \text{if } i = j \\ -S_i S_j, & \text{if } i \neq j \end{cases}$$

当注意力分布退化为近似 One-hot 时，分别分析其偏导数值：

1. **对于概率接近 1 的最大位置 $i = \max$**：$S_i \approx 1$，对角线导数 $\frac{\partial S_i}{\partial z_i} = S_i (1 - S_i) \approx 1 \times (1 - 1) = 0$。
2. **对于其余概率接近 0 的位置 $j \neq \max$**：$S_j \approx 0$，导数 $\frac{\partial S_j}{\partial z_j} = S_j (1 - S_j) \approx 0 \times 1 = 0$。
3. **对于非对角线项 $i \neq j$**：包含乘积项 $-S_i S_j$。由于其他位置的 $S_j \approx 0$，乘积项恒趋于 0。

无论是对角线还是非对角线，雅可比矩阵中的所有元素全部趋于 0。

根据反向传播链式法则，下游损失函数 $\mathcal{L}$ 对点积未激活输入 $z_j$ 的梯度为：

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_{i} \frac{\partial \mathcal{L}}{\partial S_i} \frac{\partial S_i}{\partial z_j} \approx \sum_{i} \frac{\partial \mathcal{L}}{\partial S_i} \times 0 = 0$$

这意味着梯度在 Softmax 处被完全衰减，无法向后传导给计算 $Q$ 和 $K$ 的线性投影权重矩阵（$W_Q$ 和 $W_K$），导致注意力层权重无法有效更新。

![Softmax函数导数随Logits差异增大陷入两端平坦死区的机制](/images/content-rebuild/b6d2192c188cf1b3/fig-4.png)

---

## 缩放因子的数学证明与实验对比

为了消除维度 $d_k$ 对方差的干扰，Transformer 引入了 Scaled Dot-Product 操作：

$$\text{Score}(q, k) = \frac{q \cdot k}{\sqrt{d_k}}$$

### 方差归一化证明
令缩放后的变量为 $z' = \frac{z}{\sqrt{d_k}}$。根据方差的数乘性质 $\mathrm{Var}(aX) = a^2 \mathrm{Var}(X)$：

$$\mathrm{Var}(z') = \mathrm{Var}\left( \frac{z}{\sqrt{d_k}} \right) = \left( \frac{1}{\sqrt{d_k}} \right)^2 \mathrm{Var}(z) = \frac{1}{d_k} \cdot d_k = 1$$

无论 $d_k$ 是 64、128 还是更大，除以 $\sqrt{d_k}$ 后，点积得分的均值依然是 0，方差被恒定归一化回 1。得分落在 Softmax 导数敏感的区间内，保证反向传播能够传出健康的梯度。

### 验证实验代码
以下代码使用 PyTorch 模拟 1 个 Query 对 10 个 Key 的注意力过程。为了公平对比，两组实验使用相同的随机初始化向量，并以“优化特定位置（位置 0）的注意力权重”为相同损失目标，测量回传给 Query 的梯度范数：

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)
d_k = 64
seq_len = 10

q_base = torch.randn(1, d_k)
k_base = torch.randn(seq_len, d_k)

# 对照组 1：未缩放点积 (Unscaled)
q_unscaled = q_base.clone().detach().requires_grad_(True)
scores_unscaled = torch.matmul(q_unscaled, k_base.T)  # 理论方差约为 64
probs_unscaled = F.softmax(scores_unscaled, dim=-1)
# 目标：对位置 0 的概率进行反向求导
loss_unscaled = probs_unscaled[0, 0]
loss_unscaled.backward()
grad_norm_unscaled = q_unscaled.grad.norm().item()

# 对照组 2：缩放点积 (Scaled by sqrt(d_k))
q_scaled = q_base.clone().detach().requires_grad_(True)
scores_scaled = torch.matmul(q_scaled, k_base.T) / (d_k ** 0.5)  # 理论方差归一化为 1
probs_scaled = F.softmax(scores_scaled, dim=-1)
loss_scaled = probs_scaled[0, 0]
loss_scaled.backward()
grad_norm_scaled = q_scaled.grad.norm().item()

print(f"未缩放得分方差: {scores_unscaled.var().item():.3f} (理论值: {d_k})")
print(f"缩放后得分方差: {scores_scaled.var().item():.3f} (理论值: 1.0)")
print(f"未缩放反向梯度范数: {grad_norm_unscaled:.6e}")
print(f"缩放后反向梯度范数: {grad_norm_scaled:.6e}")
```

运行结果中，未缩放情况下的反向梯度范数通常比缩放后小几个数量级。除以 $\sqrt{d_k}$ 显著防止了梯度在反向传播初期迅速衰减。

![未缩放与缩放后Softmax概率分布形态与数值稳定性对比](/images/content-rebuild/b6d2192c188cf1b3/fig-5.png)

---

## 面试追问与常见认知误区

### 追问 1：如果维度很小（如 $d_k = 4$），除以 $\sqrt{d_k}$ 还重要吗？
**回答要点**：
- 当 $d_k$ 较小时，方差本身只有 4，标准差为 2，点积数值的极差有限，Softmax 较难进入两端平坦的饱和区。
- 在原始论文《Attention Is All You Need》中，作者明确提到：对于较小的 $d_k$，未缩放的点积注意力和加法注意力机制表现相当；但**随着 $d_k$ 增大，不进行缩放的点积注意力表现明显恶化**。因此在维度较小时，缩放因子的收益不显著，但在大维度下则是必需的稳定器。

### 追问 2：既然点积模长会导致数值波动，为什么不直接用余弦相似度？
**回答要点**：
- **保留表达自由度**：余弦相似度强行将向量除以 $L_2$ 范数，仅保留空间夹角。但在自然语言中，不同词汇的信息量和确定性不同（例如高频虚词与核心语义实词）。保留模长信息，模型就拥有了通过参数调整“基底权重强度”的能力。
- **计算吞吐差异**：余弦相似度需要对每个向量在线计算范数并执行除法，破坏了连续大矩阵相乘在硬件上的密集计算吞吐；而矩阵点乘后除以一个全局静态标量 $\sqrt{d_k}$，能够无缝融合进底层算子流水线，计算效率高得多。

### 追问 3：输入之前已经有 LayerNorm 了，为什么点积方差还会爆炸？
**回答要点**：
- **规范化的维度不同**：LayerNorm 约束的是单个 Token 向量在自身特征维度（通道维度）上的均值和方差，确保输入给投影矩阵时单个向量数值稳定。
- **累加发生在内积环节**：即使每个分量方差严格为 1，只要在计算内积时把 $d_k$ 个独立项累加在一起，累加后的**标量结果**其方差必然随着项数增加而线性膨胀到 $d_k$。LayerNorm 无法跨越内积求和步骤来约束后续标量的统计量。关于 LayerNorm 的具体机制可查阅 [/interview/llm-interview/layer-normalization.md](/interview/llm-interview/layer-normalization.md)。

### 常见错误说法纠偏
- **误区一**：“除以 $\sqrt{d_k}$ 是为了把数值限制在 0 到 1 之间。”
  - *纠正*：点积数值有正有负，除以常数只改变方差尺度，不可能把所有结果限制在 0 到 1，也没必要限制在 0 到 1。负责输出 0 到 1 概率的是后续的 Softmax。
- **误区二**：“如果不除以 $\sqrt{d_k}$，前向传播就会直接发生 NaN 溢出。”
  - *纠正*：现代深度学习框架中的 Softmax 普遍采用了减去最大值的数值稳定实现（$\frac{e^{z_i - z_{\max}}}{\sum e^{z_j - z_{\max}}}$），前向计算通常不会溢出报 NaN；未缩放的核心问题是反向传播梯度消失和注意力单热退化。
- **误区三**：“点积的方差是 $\sqrt{d_k}$。”
  - *纠正*：独立分量相乘求和后，方差是 $d_k$，标准差才是 $\sqrt{d_k}$。除以标准差方能将方差归一化回 1。

![面试常见追问对比矩阵：缩放点积、余弦相似度与小维度边界](/images/content-rebuild/b6d2192c188cf1b3/fig-6.png)

---

## 关联阅读
- 自注意力中 Q、K、V 的定义与角色分工：[/interview/llm-interview/attention-qkv.md](/interview/llm-interview/attention-qkv.md)
- 点积注意力与加法注意力的计算性能对比：[/interview/llm-interview/attention-dot-vs-add.md](/interview/llm-interview/attention-dot-vs-add.md)
- 多头注意力机制的拆分原理与投影意义：[/interview/llm-interview/why-multi-head-attention.md](/interview/llm-interview/why-multi-head-attention.md)
