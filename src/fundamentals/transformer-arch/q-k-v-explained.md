---
title: Self-Attention 中的 Q、K、V 是什么？从查询匹配到加权聚合
icon: mdi:alphabetical-variant
tag:
- Transformer架构
- Self-Attention
- QKV
description: 用直观例子拆解自注意力中的 Query、Key、Value。剖析线性投影打破对称性的数学逻辑，给出完整手算算例与 PyTorch 实现。
---

在构建 Transformer 模型的过程中，最核心的操作是自注意力机制（Self-Attention）。在阅读相关论文或代码时，你一定会频繁看到三个术语：**Query（查询向量，Q）**、**Key（键向量，K）** 和 **Value（值向量，V）**。

这三个向量究竟代表什么？为什么同一个词要拆成三份不同的向量？它们又是如何计算出上下文信息的？

本文通过一个贯穿例句、具体算例和精简代码，为你拆解 Q、K、V 的计算过程与设计意图。

![Self-Attention 中 Q、K、V 投影与匹配机制总览](/images/content-rebuild/4a413940b8ae71dd/cover.png)

---

## 1. 概念建立：从字典检索到连续向量加权

要理解 Q、K、V，最直观的类比是传统编程中的**键值对字典（Key-Value Store）**。

假设我们有一个简单的 Python 字典：

```python
# 键值对数据库：Key 是图书类别，Value 是具体的书名
book_db = {
    "python_basics": "《Python编程从入门到实践》",
    "deep_learning": "《深度学习》（花书）",
    "history": "《全球通史》"
}

# 此时发起一次查询（Query）
query = "deep_learning"
result = book_db.get(query, None)  # 返回：《深度学习》（花书）
```

在传统字典查找中，匹配逻辑是**离散且非黑即白**的（硬匹配）：
- 当 Query 与某个 Key 完全相等时，返回对应的 Value；
- 如果 Query 与所有 Key 都不相等，或者 Query 同时涉及两个类别（比如“用 Python 搞深度学习”），传统查找就无法直接返回两者的平滑组合。

![传统精确字典查找与连续向量空间注意力聚合机制对比](/images/content-rebuild/4a413940b8ae71dd/fig-1.png)

Transformer 的自注意力机制将这一过程搬到了**连续的向量空间**：

1. **Query（Q）**：表示“当前位置正在寻找什么样的上下文信息”。
2. **Key（K）**：表示“当前位置能提供什么样的特征供别人检索”。
3. **Value（V）**：表示“当前位置真正包含的信息内容”。

在连续向量空间中，Query 与每一个 Key 计算**向量内积（点积）**来衡量相关度。相关度经过 Softmax 归一化为一组权重，最后对所有 Value 做**加权平均**：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

这种检索方式是连续且可导的：模型不需要在多个词之间做“非此即彼”的硬选择，而是可以同时吸收多个词的部分信息。

<ReadMoreLock />

---

## 2. 为什么同一个词需要拆成 Q、K、V 三个向量？

在进入注意力层之前，序列中的每个词已经由嵌入层（Embedding）转换成了一个稠密向量 $x$。很多初学者会产生一个疑问：**为什么不直接计算词向量之间的内积 $x_i \cdot x_j$，然后对 $x$ 本身做加权平均？为什么非要引入三个线性变换矩阵 $W_Q, W_K, W_V$？**

这主要解决两个核心问题：**打破点积对称性**与**解耦词的不同职责**。

![输入 Embedding 经过三个独立投影矩阵解耦为 Q、K、V 空间](/images/content-rebuild/4a413940b8ae71dd/fig-2.png)

### 2.1 打破点积的对称性
向量点积满足交换律，即：

$$x_i \cdot x_j = x_j \cdot x_i$$

如果直接用原始输入向量做点积打分，词 $i$ 对词 $j$ 的注意力得分，就必定等于词 $j$ 对词 $i$ 的注意力得分。注意力矩阵天然就会成为一个对称矩阵。

但在自然语言中，词与词之间的语法和语义依存关系通常是**非对称**的。

以句子 **“小猫 追逐 蝴蝶”** 为例：
- 当模型处理谓语动词 **“追逐”** 时，为了确定动作的发出者和承受者，它需要高度关注主语 **“小猫”** 和宾语 **“蝴蝶”**；
- 但当模型处理名词 **“小猫”** 时，它可能更关注修饰它的定语（比如“黑色的小猫”），并不需要以同等强度的注意力回看谓语动词 **“追逐”**。

引入两个不同的参数矩阵 $W_Q$ 和 $W_K$ 后：

$$q_i = x_i W_Q, \quad k_j = x_j W_K$$

它们之间的注意力打分为：

$$q_i \cdot k_j^T = x_i (W_Q W_K^T) x_j^T$$

因为矩阵乘积 $W_Q W_K^T$ 通常不是对称矩阵，所以 $q_i \cdot k_j^T \neq q_j \cdot k_i^T$。这样就打破了强制对称性，让定向依存关系的建模成为可能。

### 2.2 解耦特征的表达职责
同一个词在句子中扮演的角色是多样的：
- 作为**查询者**（Query）时，它关心的是“我缺什么信息，我需要去注意谁”；
- 作为**被查者**（Key）时，它关心的是“我的特征标签是什么，别人凭什么选我”；
- 作为**内容提供者**（Value）时，它关心的则是“一旦被选中，我应该把什么信息传递给下游”。

通过三个独立的线性映射：

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

模型可以在不同的子空间中学习这三种不同的特征表达，避免了单一向量同时承担多重冲突任务时的表达瓶颈。

---

## 3. 注意力的计算步骤与因果掩码

明确了 Q、K、V 的角色后，我们梳理完整的单头自注意力计算流程。设输入序列长度为 $L$，词向量维度为 $d$，投影后的向量维度为 $d_k$（$V$ 的维度为 $d_v$，通常取 $d_v = d_k$）。

### 第一步：线性投影
将输入矩阵 $X \in \mathbb{R}^{L \times d}$ 分别乘以三个可学习的权重矩阵，得到：
- $Q = X W_Q \in \mathbb{R}^{L \times d_k}$
- $K = X W_K \in \mathbb{R}^{L \times d_k}$
- $V = X W_V \in \mathbb{R}^{L \times d_v}$

### 第二步：计算注意力得分与缩放
将查询矩阵 $Q$ 与键矩阵的转置 $K^T$ 相乘，得到序列中所有词两两之间的未归一化打分矩阵 $S$：

$$S = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{L \times L}$$

矩阵中第 $i$ 行第 $j$ 列的元素 $S_{i, j}$ 表示第 $i$ 个词对第 $j$ 个词的匹配得分。除以 $\sqrt{d_k}$ 是为了防止向量维度较大时点积方差过大，导致 Softmax 进入饱和区出现梯度消失。

### 第三步：施加掩码（可选，区分编码与生成）
注意力机制本身支持双向可见。但在自回归语言生成任务（如 GPT 这类 Decoder-only 架构）中，模型生成当前词时不能提前看到未来的词。因此需要在执行 Softmax 之前施加**因果掩码（Causal Mask）**：

![点积打分与因果掩码（Causal Mask）在矩阵层面的操作流程](/images/content-rebuild/4a413940b8ae71dd/fig-3.png)

对于掩码矩阵 $M$，若 $j > i$（未来位置），则设为 $-\infty$；其余位置设为 $0$：

$$\tilde{S} = S + M$$

在普通的双向编码任务（如 BERT 或 Transformer Encoder）中，无需添加该因果掩码。

### 第四步：Softmax 归一化
对打分矩阵的每一行分别计算 Softmax，得到权重矩阵 $A$：

$$A = \text{Softmax}(\tilde{S}) \in \mathbb{R}^{L \times L}$$

矩阵每一行元素均为非负数且和为 1，代表对应词对其他词的关注权重分布。

### 第五步：对 Value 加权求和
将权重矩阵 $A$ 与值矩阵 $V$ 相乘，得到融合了上下文信息的最终输出 $O$：

$$O = A V \in \mathbb{R}^{L \times d_v}$$

输出矩阵 $O$ 的每一行，就是对应位置吸纳全序列相关信息后生成的新表示向量。

---

## 4. 逐步算例：手算一次注意力加权

为了直观展示上述张量运算，我们构造一个极简算例。

假设输入序列包含 $L=3$ 个词，特征维度 $d_k=2$。为了把焦点集中在矩阵交互本身，我们设定一组具体的 $Q, K, V$ 数值，并假设缩放因子 $\sqrt{d_k}=1$ 进行手工验算。

设定矩阵如下：

$$Q = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix}, \quad K = \begin{bmatrix} 1.0 & 1.0 \\ 0.0 & 1.0 \\ 2.0 & 0.0 \end{bmatrix}, \quad V = \begin{bmatrix} 10.0 & 0.0 \\ 0.0 & 10.0 \\ 5.0 & 5.0 \end{bmatrix}$$

### 步骤 1：计算打分矩阵 $S = Q K^T$

首先转置 $K$：

$$K^T = \begin{bmatrix} 1.0 & 0.0 & 2.0 \\ 1.0 & 1.0 & 0.0 \end{bmatrix}$$

计算矩阵乘积 $S$（形状为 $3 \times 3$）：

$$S = \begin{bmatrix}
1\times1 + 0\times1 & 1\times0 + 0\times1 & 1\times2 + 0\times0 \\
0\times1 + 2\times1 & 0\times0 + 2\times1 & 0\times2 + 2\times0 \\
1\times1 + 1\times1 & 1\times0 + 1\times1 & 1\times2 + 1\times0
\end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 & 2.0 \\ 2.0 & 2.0 & 0.0 \\ 2.0 & 1.0 & 2.0 \end{bmatrix}$$

### 步骤 2：施加因果掩码（以自回归解码为例）

若处于自回归生成场景，将右上角未来位置置为 $-\infty$：

$$\tilde{S} = \begin{bmatrix}
1.0 & -\infty & -\infty \\
2.0 & 2.0 & -\infty \\
2.0 & 1.0 & 2.0
\end{bmatrix}$$

### 步骤 3：逐行 Softmax 归一化

- **第 1 行（位置 1 只能看位置 1）**：
  $$A_{1, :} = [\text{Softmax}(1.0), 0, 0] = [1.0, 0.0, 0.0]$$
- **第 2 行（位置 2 可看位置 1 和 2）**：
  指数项为 $e^2, e^2$，两者相等：
  $$A_{2, :} = [0.5, 0.5, 0.0]$$
- **第 3 行（位置 3 可看位置 1、2、3）**：
  指数项为 $e^2 \approx 7.389$、$e^1 \approx 2.718$、$e^2 \approx 7.389$，分母总和约为 $17.496$：
  $$A_{3, 1} = \frac{7.389}{17.496} \approx 0.422, \quad A_{3, 2} = \frac{2.718}{17.496} \approx 0.155, \quad A_{3, 3} = \frac{7.389}{17.496} \approx 0.422$$
  $$A_{3, :} \approx [0.422, 0.155, 0.422]$$

整理后的权重矩阵 $A$ 约为：

$$A \approx \begin{bmatrix} 1.0 & 0.0 & 0.0 \\ 0.5 & 0.5 & 0.0 \\ 0.422 & 0.155 & 0.422 \end{bmatrix}$$

![Softmax 归一化注意力权重与 Value 矩阵加权求和示意图](/images/content-rebuild/4a413940b8ae71dd/fig-4.png)

### 步骤 4：对 Value 矩阵加权聚合

计算最终输出 $O = A V$：

- **第 1 个位置的输出**：
  $$o_1 = 1.0 \times [10, 0] = [10.0, 0.0]$$
- **第 2 个位置的输出**：
  $$o_2 = 0.5 \times [10, 0] + 0.5 \times [0, 10] = [5.0, 5.0]$$
- **第 3 个位置的输出**：
  $$o_3 \approx 0.422 \times [10, 0] + 0.155 \times [0, 10] + 0.422 \times [5, 5] = [4.22 + 2.11, 1.55 + 2.11] = [6.33, 3.66]$$

从结果可以看到：第 1 个位置只保留了自己的信息；第 2 个位置均匀融合了前两项的信息；第 3 个位置则按照 $0.422 : 0.155 : 0.422$ 的比例，将前三个词的 Value 向量混合成了一个新的上下文表示。

---

## 5. 自注意力与交叉注意力的对比

在标准 Transformer 架构中，注意力机制主要有两种形式：**自注意力（Self-Attention）** 与 **交叉注意力（Cross-Attention）**。两者的内部计算逻辑完全一致，差别主要在于 **Q、K、V 的来源**。

![Self-Attention 与 Cross-Attention 中 Q、K、V 来源与交互意图对比](/images/content-rebuild/4a413940b8ae71dd/fig-5.png)

| 对比维度 | 自注意力（Self-Attention） | 交叉注意力（Cross-Attention） |
| :--- | :--- | :--- | 
| **Query（Q）来源** | 来自当前序列自身 | 来自解码器（Decoder）当前层的隐藏状态 |
| **Key（K）来源** | 来自当前序列自身 | 来自编码器（Encoder）最终输出的全局状态 |
| **Value（V）来源** | 来自当前序列自身 | 来自编码器（Encoder）最终输出的全局状态 |
| **计算目标** | 计算同一个句子内部词与词之间的依存关系 | 让生成端（如目标语言）根据已有内容检索源端（如源语言）的对应内容 |
| **序列长度** | Q、K、V 序列长度相同（均为 $L$） | Q 的长度为目标序列长度 $L_{\text{tgt}}$，K 与 V 的长度为源序列长度 $L_{\text{src}}$ |
| **打分矩阵尺寸** | $L \times L$ 方阵 | $L_{\text{tgt}} \times L_{\text{src}}$ 矩形矩阵 |

关于 Transformer 整体架构以及编码器、解码器的组织形式，可以参考《[Transformer 相比 RNN 和 CNN 有什么优势？](/fundamentals/transformer-arch/transformer-vs-rnn-cnn.md)》与《[Encoder-only、Decoder-only 和 Encoder-Decoder 架构有什么区别？](/fundamentals/transformer-arch/encoder-decoder-variants.md)》。

---

## 6. 最小可验证代码：单头注意力实现

下面给出一个包含因果掩码选项的 PyTorch 实现，便于你在本地直接运行和验证张量形状：

```python
import torch
import torch.nn as nn
import math

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.d_k = d_k
        # 定义三个独立的线性映射
        self.w_q = nn.Linear(d_model, d_k, bias=False)
        self.w_k = nn.Linear(d_model, d_k, bias=False)
        self.w_v = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x: torch.Tensor, is_causal: bool = False):
        """
        参数:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            is_causal: 是否施加因果掩码（仅在自回归生成时开启）
        返回:
            output: 上下文聚合输出，形状为 (batch_size, seq_len, d_k)
            weights: 注意力权重矩阵，形状为 (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # 1. 线性投影得到 Q, K, V -> (batch_size, seq_len, d_k)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 2. 计算点积得分并除以 sqrt(d_k) -> (batch_size, seq_len, seq_len)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_k)

        # 3. 施加因果掩码（右上角置为 -inf）
        if is_causal:
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
            scores = scores + mask

        # 4. Softmax 归一化得到注意力权重
        weights = torch.softmax(scores, dim=-1)

        # 5. 加权聚合 Value -> (batch_size, seq_len, d_k)
        output = torch.bmm(weights, v)
        return output, weights


# 本地验证示例
if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size, seq_len, d_model, d_k = 1, 4, 8, 8
    mock_x = torch.randn(batch_size, seq_len, d_model)

    attn_layer = SingleHeadAttention(d_model=d_model, d_k=d_k)
    out, weights = attn_layer(mock_x, is_causal=True)

    print("输入维度:", mock_x.shape)          # [1, 4, 8]
    print("输出维度:", out.shape)             # [1, 4, 8]
    print("注意力权重矩阵尺寸:", weights.shape)  # [1, 4, 4]
    print("第 1 个词的注意力分布（只能看自己）:\n", weights[0, 0].detach().round(decimals=3))
    print("最后 1 个词的注意力分布（可看所有历史）:\n", weights[0, -1].detach().round(decimals=3))
```

![自注意力实现中的掩码泄漏排查与热力图对比](/images/content-rebuild/4a413940b8ae71dd/fig-6.png)

---

## 7. 总结

回顾 Q、K、V 的核心机制：

1. **角色分工**：Query 是寻索线索，Key 是匹配索引，Value 是承载的实质内容。三者通过独立的权重矩阵投影得到。
2. **打破对称性**：不同的 $W_Q$ 和 $W_K$ 使得 $Q K^T$ 成为非对称矩阵，允许词 $A$ 关注词 $B$ 的程度与词 $B$ 关注词 $A$ 的程度互不相等。
3. **软检索聚合**：计算相似度经过 Softmax 转为概率分布，再对 Value 进行加权线性求和，完成上下文信息的融合。

在上述点积计算中，我们除以了缩放常数 $\sqrt{d_k}$。这个除法看似简单，却是保证训练初期梯度平稳的关键设计。在下一篇《[为什么 Self-Attention 要除以根号 dk？](/fundamentals/transformer-arch/why-divide-sqrt-dk.md)》中，我们将从方差推导与 Softmax 导数性质出发，深入解析这一设计的数学依据。
