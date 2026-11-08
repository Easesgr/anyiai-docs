---
title: Embedding 是什么？大模型如何把文本变成向量
icon: mdi:book-open-page-variant-outline
date: 2026-07-16
tag:
- 大模型100讲
- 基础篇
description: Embedding 层是一张可学习的矩阵查找表，将离散 Token ID 映射为稠密连续向量。本文详解其查表机制、显存开销、梯度更新原理与工程实践。
---

文本在计算机底层的旅程，始于分词后的离散整数序列。在上一讲[/llm-100/basics/Token_是什么_为什么大模型不是按字理解世界.md]中，我们了解到分词器会将文本切分为一个个 Token，并赋予其唯一的整数 ID（例如句子“新电影真好看”被切分为 `[464, 5975, 318]`）。

然而，神经网络是由连续的浮点数加权、乘法与激活函数构建的系统。如果直接把编号送入网络，模型就会把这些整数当作连续标量进行运算。例如系统可能会认为 $5976$ 是 $2988$ 的两倍，但 Token 编号的大小次序与语言本身的词义没有任何逻辑关联。

**Embedding（嵌入）层的核心任务，就是将这些离散的符号 ID 转换成固定长度的稠密浮点数向量。**

以三个词汇“猫”、“狗”和“挖掘机”为例：在输入模型前，它们仅是三个数字标签；经过训练良好的 Embedding 映射后，“猫”和“狗”的向量在方向上彼此贴近，而“挖掘机”的向量则与它们相距较远。通过这种转换，离散的词汇获得了可计算的几何距离。

![从离散 Token ID 到连续几何向量](/images/content-rebuild/053a03aff9b1da45/cover.png)

在进入 Transformer 的注意力层之前，**输入端的 Embedding 层本质上就是一张静态查找表**。本文将拆解这张查找表的存储结构、硬件计算方式、梯度更新机制，以及它与后续上下文向量的边界。

---

## 为什么不能直接用 One-Hot 编码？

在传统机器学习中，处理类别特征最直接的方法是 One-Hot（独热）编码。如果词表大小为 $V$（例如现代大模型常见的 32,000 到 128,000），每个 Token 都可以写成一个 $V$ 维的向量：只在自身 ID 对应的索引位置置 1，其余位置全为 0。

这种简单表示在自然语言处理中面临两个无法逾越的工程与数学瓶颈：

### 1. 几何正交导致无法表达语义相关性
在欧氏空间中，任意两个不同 Token 的 One-Hot 向量 $x_i, x_j \in \mathbb{R}^V$（其中 $i \neq j$）之间是严格正交的：

$$
x_i \cdot x_j = 0
$$

我们来计算两者的余弦相似度：

$$
\cos(x_i, x_j) = \frac{x_i \cdot x_j}{\|x_i\|_2 \|x_j\|_2} = \frac{0}{1 \times 1} = 0
$$

再计算两者的欧氏距离：

$$
\|x_i - x_j\|_2 = \sqrt{\sum_{k=1}^V (x_{i,k} - x_{j,k})^2} = \sqrt{(1-0)^2 + (0-1)^2} = \sqrt{2}
$$

无论词表 $V$ 扩充到多大规模，**任意两个不同 Token 的 One-Hot 向量欧氏距离永远等于 $\sqrt{2}$，余弦相似度永远等于 0**。在 One-Hot 空间里，“猫”与“狗”的距离，和“猫”与“挖掘机”的距离完全一样。这种表示切断了所有先验关联，模型无法通过距离判断词义的亲疏。

![One-Hot 几何正交性与稀疏浪费](/images/content-rebuild/053a03aff9b1da45/fig-1.png)

### 2. 极端稀疏带来的显存与计算浪费
如果要将 $V$ 维的 One-Hot 向量转换到模型的隐藏层维度 $D$（如 4096），需要做一次矩阵乘法：

$$
e = x_{\text{one-hot}} W
$$

其中 $x_{\text{one-hot}} \in \mathbb{R}^{1 \times V}$，权重矩阵 $W \in \mathbb{R}^{V \times D}$。在这个乘法中，$x$ 向量里有 $V - 1$ 个元素都是 0。整个计算充斥着无意义的 $0 \times W_{k, d}$ 运算，却需要显存分配一个长达数万维度的稀疏张量，造成了严重的显存读写与计算开销。

需要澄清的是：**稀疏存储带来的资源浪费并不等同于高维空间的“维度灾难”**。这里的核心瓶颈在于内存带宽浪费与语义几何距离的完全缺失。

---

## Embedding 层的结构：形状为 `[vocab_size, hidden_dim]` 的查找表

在数学定义上，Embedding 层是一个由可学习浮点数构成的权重矩阵：

$$
W_E \in \mathbb{R}^{V \times D}
$$

- $V$（Vocab Size）：词表容量，即矩阵的行数；
- $D$（Hidden Dimension）：特征维度，即矩阵的列数。

当我们用 One-Hot 向量 $x_i$ 乘以矩阵 $W_E$ 时：

$$
e_i = x_i W_E = [0, \dots, 0, \underbrace{1}_{\text{第 } i \text{ 位}}, 0, \dots, 0] \begin{bmatrix} W_{0,0} & W_{0,1} & \dots & W_{0, D-1} \\ \vdots & \vdots & \ddots & \vdots \\ W_{i,0} & W_{i,1} & \dots & W_{i, D-1} \\ \vdots & \vdots & \ddots & \vdots \\ W_{V-1,0} & W_{V-1,1} & \dots & W_{V-1, D-1} \end{bmatrix} = W_{E}[i, :]
$$

这一矩阵乘法的计算结果，在数学上严格等于**权重矩阵 $W_E$ 的第 $i$ 行**。

![Embedding 查表矩阵结构与索引切片数据流](/images/content-rebuild/053a03aff9b1da45/fig-2.png)

### 工程实现：从矩阵乘法转为显存切片
既然数学结果必然是提取第 $i$ 行，工程实现便不再需要构造 $V$ 维的稀疏向量，也不需要调用通用矩阵乘法（GEMM）算子。

深度学习框架（例如 PyTorch 的 `nn.Embedding`）在底层将其实现为**查找表（Lookup Table）**：直接接收整数索引 `input_ids`，计算内存偏移量，在连续显存块中将第 $i$ 行数据切片拷贝出来。虽然通过首地址偏移定位行索引的时间复杂度为 $O(1)$，但读取并搬运该行向量涉及 $D$ 个浮点数（单 Token 搬运复杂度为 $O(D)$，序列总搬运量为 $O(T \times D)$）。这种切片拷贝避免了稀疏矩阵乘法中无意义的零值运算。

<ReadMoreLock />

### 参数量与显存测算实例
以常见开源大模型的配置为例：
- 词表大小 $V = 128{,}000$
- 隐藏层维度 $D = 4096$

Embedding 层包含的参数量为：
$$
128{,}000 \times 4096 = 524{,}288{,}000 \text{ 个参数（约 } 5.24 \text{ 亿）}
$$

该参数矩阵在不同精度格式下的静态显存占用如下：
- **FP32（单精度，4 字节/参数）**：$5.24 \times 10^8 \times 4 \text{ B} \approx 2.097 \text{ GB}$
- **FP16 / BF16（半精度，2 字节/参数）**：$5.24 \times 10^8 \times 2 \text{ B} \approx 1.048 \text{ GB}$

这意味着在模型尚未执行任何注意力计算前，仅将输入层权重常驻于显存中，就需要消耗约 1GB 至 2GB 的空间。词表扩充能提升分词压缩率，但也会带来显存的固定成本，这是模型架构设计中必须权衡的因素。关于分词算法的更多细节，可参考[/llm-100/basics/BPE_WordPiece_SentencePiece_分词有什么区别.md]。

---

## 前向查表与反向更新过程

静态的查找表是如何跟随模型训练学到有意义的特征表示的？这需要从前向传播与反向传播的数据流来观察。

### 1. 前向传播（Forward Pass）
假设输入一个批次（Batch）包含 $B$ 条序列，每条序列长度为 $T$（Sequence Length），输入数据是一个二维整数张量：

$$
X \in \mathbb{N}^{B \times T}, \quad X_{b, t} \in \{0, 1, \dots, V-1\}
$$

经过 Embedding 层按行切片提取后，输出为三维浮点张量：

$$
E = \text{Embedding}(X) \in \mathbb{R}^{B \times T \times D}
$$

该张量随后与位置编码相加，作为后续 Transformer Block 的输入。关于模型整体前向结构的拆解，可参见[/llm-100/basics/Transformer_为什么能成为大模型的核心架构.md]。

### 2. 反向传播与梯度累加（Backward Pass）
训练过程中，下游网络输出预测并计算损失 $\mathcal{L}$。误差通过链式法则逐层向前传递，抵达输入 Embedding 层，产生针对输出张量的梯度：

$$
\nabla_E \mathcal{L} = \frac{\partial \mathcal{L}}{\partial E} \in \mathbb{R}^{B \times T \times D}
$$

系统需要据此计算权重矩阵 $W_E$ 的更新梯度 $\nabla_{W_E} \mathcal{L} \in \mathbb{R}^{V \times D}$。该过程具有两个特点：

1. **未出现的行在输入端直接梯度为零**：在输入 Embedding 独立且无输出权重共享（Weight Tying）的前提下，一个批次实际命中的 Token ID 数量最多为 $B \times T$。未被该批次使用的绝大多数行，从输入端前向计算传回的损失梯度保持为 0。需注意两点例外：若模型开启了输入与输出层权重共享，输出头计算全词表 Logits 时会向所有词表行反传梯度；若优化器开启了解耦权重衰减（如 AdamW 的 Weight Decay），即使某行梯度为 0，其参数在优化步中仍会发生衰减更新。
2. **重复出现的 Token 梯度累加（Scatter Add）**：如果某个 Token ID（如常见字或常用标点）在批次内出现了多次，根据多元复合函数的求导链式法则，所有引用该行的位置梯度必须进行加和：

$$
\nabla W_E[i, :] = \sum_{(b, t) \; \text{且} \; X_{b, t} = i} \nabla E[b, t, :]
$$

![反向传播中梯度的 Scatter Add 累加机制](/images/content-rebuild/053a03aff9b1da45/fig-3.png)

在底层实现上，深度学习框架通常使用原子加操作（Atomic Add）将梯度累加到对应行，供优化器（如 AdamW）更新权重参数。更底层的反向数值传递逻辑可参考[/fundamentals/dl-basics/forward-backward-propagation.md]。

### 实验验证：查表等价性与梯度累加行为
通过以下最小可执行代码，可以验证查表与矩阵乘法的数值一致性，并观察多位置引用同一 Token 时的梯度聚合：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

vocab_size = 10
hidden_dim = 4

# 实例化 Embedding 层
emb = nn.Embedding(vocab_size, hidden_dim)

# 构造输入序列：Token ID 2 出现了两次
input_ids = torch.tensor([2, 5, 2], dtype=torch.long)

# 1. 前向查表切片
out_lookup = emb(input_ids)  # 形状: [3, 4]

# 2. 用 One-Hot 矩阵乘法模拟相同的前向计算
one_hot = F.one_hot(input_ids, num_classes=vocab_size).float()  # 形状: [3, 10]
out_matmul = torch.matmul(one_hot, emb.weight)                 # 形状: [3, 4]

# 验证查表与矩阵乘法的数值完全一致
assert torch.allclose(out_lookup, out_matmul), "查表与矩阵乘法输出不一致！"

# 3. 模拟反向传播
loss = out_lookup.sum()
loss.backward()

print("Embedding 梯度形状:", emb.weight.grad.shape)
print("未出现在批次中的 Token (ID 0) 的梯度行:\n", emb.weight.grad[0])
print("仅出现 1 次的 Token (ID 5) 的梯度行:\n", emb.weight.grad[5])
print("出现了 2 次的 Token (ID 2) 的梯度行（发生加和）:\n", emb.weight.grad[2])
```

运行结果验证：未命中的 ID 0 对应行梯度全为 0；ID 5 的梯度为全 1.0；而被两次引用的 ID 2 其梯度累加为 2.0。

---

## 概念辨析：静态输入嵌入、Word2Vec 与上下文表征

在工程与学术讨论中，“Embedding”一词经常出现在不同层级，容易产生混淆。我们可以根据生成时机与上下文依赖性将其归纳为三类：

![静态输入嵌入、传统静态词向量与上下文动态表征对比](/images/content-rebuild/053a03aff9b1da45/fig-4.png)

| 比较维度 | 传统词向量（如 Word2Vec / GloVe） | 大模型输入嵌入（Input Embedding） | Transformer 上下文表征（Contextual Representation） |
| :--- | :--- | :--- | :--- |
| **基本单元** | 通常为整词（Word） | 分词器切出的子词（Subword Token ID） | 序列中的 Token 隐状态（Hidden States） |
| **是否依赖上下文** | **不依赖**。每个词在全词表中对应一个固定向量 | **不依赖**。同一个 Token ID 查出的行向量固定相同 | **高度依赖**。经过注意力机制计算，融入了全局上下文 |
| **多义词处理** | 无法消除歧义，“苹果”的水果含义与公司含义被压缩在同一向量中 | 无法消除歧义，输入端只要 Token ID 相同，查出的切片就一致 | 能够消歧。注意力层根据“吃”或“发布会”动态混合语义，生成不同表征 |
| **训练方式** | 在浅层共现任务中独立训练，通常离线冻结使用 | 在大模型端到端训练中从随机初始化开始，与全网参数共同更新 | 由深层注意力网络在前向计算中逐层生成 |
| **所在网络位置** | 独立于复杂模型外部的静态词典 | 位于 Transformer 模型的最底层输入端 | 位于各 Transformer Block 内部及输出端 |

现代大模型之所以不直接使用训练好的 Word2Vec 作为输入，原因在于：
1. **词表分词粒度不同**：Word2Vec 依赖整词切分，遇到未登录词（OOV）表现脆弱；大模型采用子词算法（如 BPE），词表结构完全不同。
2. **端到端联合训练收益**：将 Embedding 层置于端到端反向传播链路中，使其特征分布能够直接配合后续数十层注意力的表示需求，收敛自由度更高。

---

## 向量相似度与几何特性

当 Token 转化为连续向量后，计算机就可以通过向量夹角或模长计算其相关性。

### 1. 度量方式：点积与余弦相似度
在语义匹配、检索增强生成（RAG）及模型内部计算中，常用的度量包括点积（Dot Product）与余弦相似度（Cosine Similarity）：

$$
\text{DotProduct}(u, v) = u \cdot v = \sum_{d=1}^D u_d v_d
$$

$$
\text{CosineSimilarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \frac{\sum_{d=1}^D u_d v_d}{\sqrt{\sum_{d=1}^D u_d^2} \sqrt{\sum_{d=1}^D v_d^2}}
$$

![高维单位超球面与余弦夹角几何示意](/images/content-rebuild/053a03aff9b1da45/fig-5.png)

点积计算同时受**向量夹角**和**向量模长（Norm）**的影响。在训练过程中，高频 Token 往往更容易累积较大的参数范数。如果不希望模长干扰方向判定，通常使用余弦相似度。

**如果对向量进行 $L_2$ 归一化（使每个向量的模长满足 $\|u\|_2 = 1$），余弦相似度在数学上便等同于点积**：

$$
\cos(u, v) = \frac{u_{\text{norm}} \cdot v_{\text{norm}}}{1 \times 1} = u_{\text{norm}} \cdot v_{\text{norm}}
$$

在工程检索系统中，预先对向量库执行归一化后直接使用点积，可以免去实时求平方根与除法的计算开销。

参考官方开源库 `SentenceTransformers` 的语义相似度（STS）标准用法：

```python
from sentence_transformers import SentenceTransformer

# 加载开源句嵌入模型
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences1 = [
    "The new movie is awesome",
    "The cat sits outside",
]
sentences2 = [
    "The new movie is so great",
    "The dog plays in the garden",
]

# 计算句向量并度量余弦相似度
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)
similarities = model.similarity(embeddings1, embeddings2)

for i, s1 in enumerate(sentences1):
    for j, s2 in enumerate(sentences2):
        print(f"{s1} <-> {s2} : {similarities[i][j]:.4f}")
```

运行输出表明，“The new movie is awesome”与“The new movie is so great”的余弦相似度接近 0.9，而跨话题句子的相似度显著降低。

### 2. 向量聚类是梯度的优化结果
为什么经过训练的 Embedding 能让“猫”与“狗”在几何空间中靠近？

从优化过程来看：如果“猫”和“狗”在大量语料中频繁出现在相似的上下文语境中（例如前文常有“可爱的”、“它跳上了沙发”），在预测后续词计算交叉熵损失时，传导至输入层第“猫”行与第“狗”行的梯度向量在数值和方向上高度一致。

经过成千上万步的参数优化，这两个行向量被反向传播逐渐调整到相似的方向上。空间聚类不是神秘的涌现现象，而是共现语境在梯度驱动下的参数联合优化结果。

### 3. 常见认知边界

![多义词初始无上下文与向量算术局部性警示](/images/content-rebuild/053a03aff9b1da45/fig-6.png)

#### 边界一：多义词在输入查表阶段无法自动消歧
在经过 Transformer 自注意力机制计算之前，输入 Embedding 仅仅是静态查找。句子“吃一个**苹果**”与“**苹果**发布了新款电脑”中，如果分词后的 Token ID 均为 `1234`，那么查出的初始向量在数值上**完全相同**。消除歧义是后续自注意力层动态融合周围词汇特征后的结果，不可混淆输入查找与注意力计算的职能。关于上下文融合机制，可参考[/llm-100/basics/Attention_机制到底在算什么.md]。

#### 边界二：“向量线性算术”是局部经验现象而非数学公理
经典科普常引用“国王 - 男人 + 女人 = 女王”。在真实系统中：
1. 该现象仅在特定模型、特定共现词对与归一化条件下局部成立，属于经验观察；
2. 不能推广为“一切自然语言逻辑都可以通过高维向量简单加减精确求解”；
3. 现代大模型的隐层表示高度复杂纠缠，并不存在人类直观可解释的单一概念坐标轴。

---

## 课堂动手练习（可检验）

### 目标
编写一个轻量脚本，验证向量归一化后“点积”与“余弦相似度”的严格等价性，并对比两者在大批量计算下的速度差异。

### 验证步骤
运行以下代码，检查输出断言是否通过：

```python
import torch
import torch.nn.functional as F
import time

# 1. 构造随机向量测试集 (1000 个查询向量，10000 个候选向量，隐藏维度 768)
num_queries = 1000
num_docs = 10000
dim = 768

q = torch.randn(num_queries, dim)
d = torch.randn(num_docs, dim)

# 2. 计算标准余弦相似度（每次实时计算模长与除法）
t0 = time.time()
q_norm = torch.norm(q, p=2, dim=1, keepdim=True)
d_norm = torch.norm(d, p=2, dim=1, keepdim=True)
cos_sim = torch.mm(q / q_norm, (d / d_norm).t())
t_cosine = time.time() - t0

# 3. 模拟工程场景：候选向量库离线预归一化，查询向量在线归一化后直接使用点积（矩阵乘法）
d_unit = F.normalize(d, p=2, dim=1)

t1 = time.time()
q_unit = F.normalize(q, p=2, dim=1)
dot_sim = torch.mm(q_unit, d_unit.t())
t_dot = time.time() - t1

# 验证数值完全一致
assert torch.allclose(cos_sim, dot_sim, atol=1e-5), "数值等价性校验失败！"
print(f"校验成功！余弦相似度耗时: {t_cosine:.4f}s, 预归一化点积耗时: {t_dot:.4f}s")
```

### 延伸阅读与前置衔接
- 离散符号生成：[/llm-100/basics/Token_是什么_为什么大模型不是按字理解世界.md]
- 输入向量的下游骨架：[/llm-100/basics/Transformer_为什么能成为大模型的核心架构.md]
- 上下文混合的真实场所：[/llm-100/basics/Attention_机制到底在算什么.md]
- 反向传播数值机制：[/fundamentals/dl-basics/forward-backward-propagation.md]
