---
title: Transformer 中的注意力遮蔽（Attention Masking）是如何工作的？
icon: mdi:eye-off
tag:
- Transformer
- Attention Masking
- 注意力
description: 深入拆解 Padding Mask 与 Causal Mask 的加法偏置机理、Softmax 归一化性质、反向梯度截断以及 FP16 下
  Safe Softmax 的边界陷阱。
---

在模型面试中被问到“Transformer 中的 Attention Mask 是如何工作的”时，仅回答“把未来的词遮住”通常只能拿到及格分。面试官更看重的是：为什么要在 Softmax 之前做加法而不是事后乘零、反向传播时被遮蔽位置的梯度如何变化、半精度浮点数下为什么容易出现 NaN，以及训练、Prefill 与 Decode 阶段的掩码形状有何不同。

![注意力遮蔽机制概览](/images/content-rebuild/0a9a03cf1923849c/cover.png)

---

## 30 秒口述结论

> **面试口述模版：**
>
> Transformer 的注意力遮蔽（Attention Masking）是在原始点积打分矩阵输入 Softmax 之前，**向无效交互位置叠加一个负无穷大偏置（Additive Mask）**。
>
> 核心分为两类：
> 1. **Padding Mask**：用于批处理（Batching）对齐变长文本，掩蔽 `<pad>` 填充字符，防止无意义的占位符参与注意力权重的分配与语义聚合；
> 2. **Causal Mask（因果掩码）**：在自回归解码中构建下三角矩阵，将位置 $j > i$ 的未来权重置零，防止信息泄露，确保当前 Token 只能看到历史上下文。
>
> 之所以采用 Softmax 前加负无穷而不是 Softmax 后将概率乘零，有两个原因：第一，保持概率归一化，让剩余有效位置自动分摊 100% 的概率总和，防止特征模长萎缩；第二，在反向传播中，Softmax 的雅可比导数外层包含前向概率因子 $P_j$，当 $P_j = 0$ 时该位置的反向梯度严格为 0，前序权重不会受到无效词污染。
>
> 在工程实现中，不能盲目硬编码 `-1e9`。在 FP16 下其最小有限负数为 -65504，绝对值过大会向负无穷方向溢出（Overflow to `-inf`）；若整行全被 Mask，Safe Softmax 会出现 $(-\infty) - (-\infty) = \text{NaN}$，且仅在 Softmax 之后做 `nan_to_num` 无法普遍保证反向梯度有限，需在 Softmax 前识别全遮蔽行赋予安全数值，并在输出显式置零。

---

## 贯穿示例：从变长批处理看数据流动

为了直观理解遮蔽过程，我们通过一个具体的批处理示例来追踪数据流动。

假设批次大小为 2，最大对齐长度为 3：
* 样本 1：`["我", "爱", "AI"]`（有效长度 3）
* 样本 2：`["好", "的", "<pad>"]`（有效长度 2，末尾补 `<pad>`）

如果这是自回归模型，在计算样本 2 的第 2 个字（“的”）时，它能看哪些字？
* 不能看 `<pad>`：因为 `<pad>` 是补齐长度的占位符，没有实际语义，属于 **Padding Mask** 负责的范围；
* 不能看时间步在它之后的词：如果有第 3 个真实词，也不能提前看，这属于 **Causal Mask** 负责的范围。

因此，样本 2 的第二个 Token 最终只能和第 1 个 Token（“好”）以及它自身（“的”）交互。

---

## Padding Mask：处理变长文本 Batch 对齐的无效填充

为了充分利用 GPU 矩阵乘法的并发吞吐能力，深度学习框架必须将多个样本组成批次并行计算。真实自然语言句子的长度不一致，较短的句子必须在末尾填充特定占位符（如 `<pad>`）对齐到批次最大长度。

![Padding Mask 消除批处理填充干扰](/images/content-rebuild/0a9a03cf1923849c/fig-1.png)

### 为什么必须遮蔽 Padding Token？

输入序列中的 `<pad>` 向量经过线性映射层（参见 [Self-Attention 中 Q、K、V 分别是什么意思？](/interview/llm-interview/attention-qkv.md)）后，同样会生成非零的 $Q_{\text{pad}}, K_{\text{pad}}, V_{\text{pad}}$ 向量：
1. **Key 污染（被有效 Token 检索）**：正常 Token 的 Query 会与 $K_{\text{pad}}$ 点积并产生分值。若不加遮蔽，无意义的 Padding 项会在 Softmax 中分走概率权重，削弱真实上下文之间的交互强度；
2. **Query 污染（主动检索其他 Token）**：Padding 位置的 Query 会对整个序列聚合生成表示，并在残差连接和 Layer Normalization 之后混入后续网络计算，浪费显存与算力。

### 张量计算与广播机制

假设批次大小为 $B$，序列长度为 $S$，注意力头数为 $H$。输入序列的 Padding 掩码为二维布尔张量：

$$\text{mask}_{\text{pad}} \in \{0, 1\}^{B \times S}$$

其中有效 Token 标记为 1，`<pad>` 标记为 0。原始点积打分矩阵为 $S_{\text{raw}} = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{B \times H \times S \times S}$。我们将 Padding Mask 转换为加法偏置，通过重塑（Reshape）为 $[B, 1, 1, S]$ 形状，利用张量广播（Broadcasting）机制叠加到打分矩阵上：

$$M_{\text{pad}}(b, 1, 1, j) = \begin{cases} 0, & \text{若第 } j \text{ 个 Token 为有效字符} \\ -\infty, & \text{若第 } j \text{ 个 Token 为 } \langle\text{pad}\rangle \end{cases}$$

$$S_{\text{masked}} = S_{\text{raw}} + M_{\text{pad}}$$

无论当前 Query 处于哪一行，只要它尝试交互的目标 Key（列索引 $j$）属于 padding，对应得分就会加上 $-\infty$。

<ReadMoreLock />

---

## Causal Mask：构建下三角矩阵防止未来信息泄露

在自回归语言模型（参考 [为什么主流大模型采用 Decoder-only 架构？](/interview/llm-interview/decoder-only.md)）或标准 Transformer 解码器中，模型基于前文预测下一个 Token：

$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t \mid x_{<t})$$

如果第 $t$ 个位置在计算时能够“看到”第 $t+1$ 个及之后的 Token，模型就会学会抄袭目标标签，导致自回归训练目标失效。

![Causal Mask 下三角因果结构](/images/content-rebuild/0a9a03cf1923849c/fig-2.png)

### 为什么 Transformer 训练时可以并行？

循环神经网络（RNN）依靠时间步串行迭代实现因果约束，无法高效并行计算。Transformer 采用 **Teacher Forcing**：在训练阶段，整个目标序列被一次性输入网络。

为了兼顾“并行训练效率”与“因果时序单向性”，Transformer 使用了 **因果掩码（Causal Mask）**。它将打分矩阵中所有位于主对角线上方的未来位置全部置为 $-\infty$。

对于长度为 $S$ 的序列，因果掩码矩阵定义为一个下三角矩阵：

$$M_{\text{causal}}(i, j) = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}$$

当序列长度 $S = 4$ 时，加法掩码的形式为：

$$M_{\text{causal}} = \begin{bmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

* 位置 1 的 Query 只能与位置 1 的 Key 交互；
* 位置 2 的 Query 能与位置 1、2 的 Key 交互；
* 位置 $i$ 无法与任何索引 $j > i$ 的位置产生交互权重。

在同时存在 Padding 和因果约束的场景下，最终的掩码矩阵是二者的组合：

$$M_{\text{combined}} = M_{\text{causal}} + M_{\text{pad}}$$

---

## 运算机理：为什么是加法掩码而非事后概率置零？

面试中的高频追问：“**为什么必须在 Softmax 之前对打分做加法偏置，而不能在 Softmax 计算完后直接把无效位置的概率乘 0？**”

![掩码在注意力张量计算数据流中的具体注入节点](/images/content-rebuild/0a9a03cf1923849c/fig-3.png)

### 1. 保持概率分布的归一化性质

Softmax 函数将任意实数打分向量映射为一个总和严格为 1 的离散概率分布：

$$\text{Softmax}(z)_i = \frac{\exp(z_i)}{\sum_{k} \exp(z_k)}$$

如果在 Softmax 计算完成后强行将遮蔽位置的概率置 0：
1. 整个行的概率和将小于 1（即 $\sum_{j} P_{ij} < 1$），破坏了“加权平均凸组合”的数学定义；
2. 剩余有效位置的相对概率没有得到重新放大，导致输出向量 $\sum_j P_{ij} V_j$ 的数值模长发生非预期衰减。

而在 Softmax 之前令无效位置的输入为 $-\infty$：

$$\exp(-\infty) = 0$$

分母中的求和项会自动剔除这些无效项，**剩余有效位置的指数项自动平分完整的 100% 概率总和**。

### 2. 梯度的数学截断

对于注意力打分矩阵的某一行，设 $P_i = \text{Softmax}(S_i)$。Softmax 的雅可比矩阵（Jacobian）为：

$$\frac{\partial P_i}{\partial S_j} = P_i (\delta_{ij} - P_j)$$

其中当 $i = j$ 时 $\delta_{ij} = 1$，否则为 $0$。根据链式法则，损失函数 $\mathcal{L}$ 对打分值 $S_j$ 的反向传播梯度为：

$$\frac{\partial \mathcal{L}}{\partial S_j} = \sum_k \frac{\partial \mathcal{L}}{\partial P_k} \frac{\partial P_k}{\partial S_j} = P_j \left( \frac{\partial \mathcal{L}}{\partial P_j} - \sum_k \frac{\partial \mathcal{L}}{\partial P_k} P_k \right)$$

![Softmax 反向传播时被 mask 位置梯度归零的数学推导示意](/images/content-rebuild/0a9a03cf1923849c/fig-4.png)

只要该行存在至少一个有效位置使分母正常，被遮蔽位置 $j$ 的前向概率 $P_j = 0$。根据上式，最外层直接乘上了 $P_j$：

$$\frac{\partial \mathcal{L}}{\partial S_j} = 0 \cdot \left( \dots \right) = 0$$

反向梯度为 0 意味着，被遮蔽的无效交互项不会给前序投影矩阵（$W_Q, W_K$）贡献任何注意力分数路径的梯度更新，截断了无效交互对模型参数的污染。

---

## 数值计算陷阱：为什么不能直接填 `-1e9`？

在理论数学中可以写 $-\infty$，但在工程实现（特别是半精度 FP16 训练与推理）中，数值范围非常有限。

![半精度浮点下数值溢出与 Safe Softmax 异常分析对比](/images/content-rebuild/0a9a03cf1923849c/fig-5.png)

### 1. FP16 的动态范围限制

标准 IEEE 754 半精度浮点数（FP16）包含 5 位指数和 10 位尾数：
* 最大有限正数为 $65504$
* 最小有限负数为 $-65504$

如果在 FP16 张量中直接写入 `-1e9`，该数值超出了 FP16 所能表示的最小负数范围，会发生向负无穷方向的数值溢出（Overflow to `-inf`）。在某些特定的硬件编译器或指令集优化中，过大负数的越界隐式转换容易诱发未定义行为。

### 2. Safe Softmax 中的 NaN 崩溃陷阱

深度学习框架为了防止 $\exp(x)$ 在大数值下溢出，普遍采用 **Safe Softmax** 算法：

$$\text{Softmax}(x)_i = \frac{\exp(x_i - \max(x))}{\sum_k \exp(x_k - \max(x))}$$

思考一个边界情况：**在批处理中，若某一个样本在当前头下的某一行全是 Padding（整行全被遮蔽）**：

* **情况 A：使用严格的 `-inf`**：
  1. 该行所有元素均为 $-\infty$；
  2. 计算最大值得到 $\max(x) = -\infty$；
  3. 执行减法时计算 $(-\infty) - (-\infty)$，这在浮点规范中是未定义操作，**直接产生 NaN**；
  4. 随后反向传播将 NaN 扩散到模型权重，导致训练崩溃。
* **情况 B：使用有限极小值（如 `torch.finfo(dtype).min`）**：
  1. 整行所有值均为 $-65504$；
  2. 最大值 $\max(x) = -65504$；
  3. 减去最大值后每个元素都变为 $0$，计算 $\exp(0) = 1$；
  4. 归一化后，**整行输出了均匀分布 $1/S$**。模型会对无效的 Padding 向量计算等权重平均，依然引入了污染。

### 工程实现代码参考

若仅在 Softmax 之后调用 `torch.nan_to_num`，无法普遍保证反向梯度的数值稳定性（Softmax 反向算子输入整行 `-inf` 时容易计算出未定义梯度）。规范做法是在 Softmax 前识别全遮蔽行并填充安全数值，计算 Softmax 后再显式置零：

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention_with_mask(Q, K, V, attn_mask=None):
    """
    参数说明:
    Q, K, V: [B, H, S, d_k]
    attn_mask: [B, 1, S_q, S_k], True 表示保留，False 表示遮蔽
    """
    d_k = Q.size(-1)
    # 1. 计算原始打分 [B, H, S_q, S_k]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # 2. 注入加法掩码与全 Mask 行防护
    if attn_mask is not None:
        # 识别整行全被遮蔽的位置，避免 Safe Softmax 出现 (-inf) - (-inf) = NaN
        all_masked_rows = ~attn_mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~attn_mask, -float('inf'))
        # 将全被遮蔽的行临时替换为 0.0，确保进入 Softmax 的输入在数值上有限
        scores = scores.masked_fill(all_masked_rows, 0.0)
        
        # 3. Softmax 概率计算
        probs = F.softmax(scores, dim=-1)
        # 显式将原本全遮蔽行的注意力权重清零
        probs = probs.masked_fill(all_masked_rows, 0.0)
    else:
        probs = F.softmax(scores, dim=-1)
    
    # 4. 加权求和
    return torch.matmul(probs, V)
```

---

## 面试追问：Prefill 阶段与 Decode 阶段的 Mask 有何不同？

在大模型推理中，服务通常分为两个阶段：处理提示词的 **Prefill 阶段** 和逐字生成的 **Decode 阶段**。两阶段在 Attention Mask 的使用上有显著区别。

![推理阶段 Prefill 与 Decode 的 Mask 形状与机制对比](/images/content-rebuild/0a9a03cf1923849c/fig-6.png)

### 阶段一：Prefill 阶段（全 Prompt 并行处理）

* **输入状态**：模型一次性接收用户的全部 Prompt（长度为 $S$）；
* **张量形状**：Query 长度为 $S$，Key/Value 长度为 $S$；
* **Mask 行为**：$Q K^T$ 产生 $S \times S$ 的二维打分矩阵。为了与训练阶段的行为保持一致，必须施加标准的 $S \times S$ 下三角因果掩码。第 1 个 Prompt Token 不能看第 2 个 Prompt Token。

### 阶段二：Decode 阶段（自回归单步推演）

* **输入状态**：在生成第 $t$ 个 Token 时，前 $t-1$ 个历史 Token 的 Key 和 Value 已经缓存在 [KV Cache](/interview/llm-interview/kv-cache.md) 中。当前输入仅为最新生成的单个 Token（长度为 1）；
* **张量形状**：Query 长度为 $1$（张量形状 $[B, H, 1, d]$），而 Key/Value 从缓存中拼接取出，包含历史所有内容，长度为 $t$（张量形状 $[B, H, t, d]$）；
* **Mask 行为**：单 Token 自回归时打分矩阵形状为 $1 \times t$。对于标准因果关系，**此时不需要下三角矩阵**。

> **深度解析：为什么单步 Decode 不需要下三角因果掩码？**
>
> 因为未来的时序已由**物理生成顺序天然约束**。当前步只有最新 Token 的 Query，它检索历史已缓存的 $t$ 个 Token。未来的第 $t+1$ 个 Token 在物理显存中尚不存在，没有对应的 Key/Value 参与运算，不存在下三角窥探未来的数学可能。
>
> 但需要注意，**无未来 KV 不等于 Decode 阶段不需要任何 Mask**：若批次中存在变长 Padding、启用了滑动窗口注意力（Sliding Window，需遮蔽超出窗口的历史 KV）、多序列打包，或是投机解码（Speculative Decoding 一次并行验证多个 Candidate Token），Decode 阶段仍需对应的一维 Padding Mask 或局部树状因果掩码。

---

## 常见面试误区与避坑指南

### 误区 1：认为 BERT 和 GPT 使用相同的 Attention Mask
* **纠偏**：BERT 属于 Encoder-only 架构，采用双向注意力机制，内部**仅使用 Padding Mask**，不使用任何因果掩码；GPT 属于 Decoder-only 架构，必须叠加 **Causal Mask** 保证自回归时序。

### 误区 2：以为 Decode 阶段每步都在动态维护一个膨胀的下三角方阵
* **纠偏**：这是对 KV Cache 机制的混淆。下三角方阵只出现在训练阶段和 Prefill 阶段（$S \times S$）。在单 Token Decode 阶段，单步 Query 长度为 1，打分矩阵是 $1 \times t$ 的行向量，无需维护膨胀的下三角方阵；但若存在 Padding、滑动窗口或投机验证多 Token，依然需要传入对应的一维或局部掩码。

### 误区 3：认为被 Mask 掉的位置在反向传播中依然会更新投影矩阵
* **纠偏**：只要该行存在至少一个有效位置，被打上掩码的位置输出概率严格为 0，根据 Softmax 雅可比矩阵特性，损失对该打分项的偏导严格为 0，不会通过该交互路径回传梯度更新投影矩阵 $W_Q$ 和 $W_K$。但需注意，这仅切断了该位置作为被关注目标（Key）时的交互梯度；若要彻底防止 Padding Token 本身（作为 Query 及其残差输出）更新网络权重，还需在上层配合整行掩码置零或损失函数忽略（如 `ignore_index`）机制。

---

## 延伸思考与关联知识

在掌握了注意力遮蔽机制后，建议进一步横向串联以下核心模块：
* 想理解现代大语言模型为何全盘转向 Decoder-only 架构，可阅读 [为什么主流大模型采用 Decoder-only 架构？](/interview/llm-interview/decoder-only.md)；
* 关于 Encoder 与 Decoder 在多头注意力机制上的异同，详见 [Decoder 与 Encoder 阶段多头注意力的差异](/interview/llm-interview/decoder-encoder-mha-diff.md)；
* 理解 Decode 阶段打分矩阵降为 $1 \times t$ 的核心缓存机制，参阅 [KV Cache 原理解析](/interview/llm-interview/kv-cache.md)。
