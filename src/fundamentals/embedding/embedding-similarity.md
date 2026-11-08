---
title: Embedding 是如何计算语义相似度的？余弦相似度、点积与欧氏距离的计算机制
icon: mdi:approximately-equal
tag:
- Embedding
- 余弦相似度
- 向量空间
description: 详解余弦相似度、点积与欧氏距离的计算公式、归一化等价推导、Sentence Transformers 调用方式与语义判定边界。
---

在上一讲[《Embedding 到底是什么？》](/fundamentals/embedding/what-is-embedding.md)中，我们了解到文本嵌入模型能将离散的文字转换成一串固定长度的浮点数向量。

但得到向量后，计算机是如何判断两句话语义是否接近的？

以一个具体例子切入：
* 句子 A：`新上映的电影非常棒`
* 句子 B：`新电影太好看了`
* 句子 C：`这只猫坐在门外`

模型将三句话分别编码为 384 维向量 $u, v, w$。计算机并不能直接阅读文字，它判定 $u$ 与 $v$ 相似、而 $u$ 与 $w$ 不相似的唯一依据，就是**对这几组向量做数值计算**。

衡量向量接近程度的核心度量主要有三种：**余弦相似度（Cosine Similarity）**、**点积（Dot Product）**与**欧氏距离（Euclidean Distance）**。本文将解释这三种度量的数学计算过程、在模长归一化下的等价转换关系、官方库的代码调用规范，以及向量距离在真实语义理解中的失效边界。

![向量度量核心机制概览](/images/content-rebuild/417c4e531b73b888/cover.png)

---

## 三种几何度量的数学定义与计算方式

设有两个维度同为 $d$ 的实数向量 $u, v \in \mathbb{R}^d$，向量分量分别记为 $u = (u_1, u_2, \dots, u_d)$ 与 $v = (v_1, v_2, \dots, v_d)$。

### 1. 欧氏距离（Euclidean Distance）

欧氏距离是两点在多维坐标系中的直线几何距离，定义为各维度坐标差值平方和的算术平方根：

$$d(u, v) = \|u - v\|_2 = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}$$

* **取值范围**：$[0, +\infty)$。
* **含义**：数值越小代表两点在空间中靠得越近；两向量完全相同时，距离为 0。
* **特点**：欧氏距离直接受向量绝对长度（模长）影响。如果两段文本表达相同主题，但因为输入长度或池化方式导致生成的向量模长差异很大，两者的欧氏距离会被拉大。

### 2. 点积（Dot Product / 内积）

点积是两个向量对应位置分量相乘后的代数累加和：

$$u \cdot v = \sum_{i=1}^d u_i v_i$$

根据解析几何定义，点积也可以写为向量模长与夹角余弦的乘积：$u \cdot v = \|u\|_2 \|v\|_2 \cos(\theta)$，其中 $\|u\|_2 = \sqrt{\sum_{i=1}^d u_i^2}$ 表示向量 $u$ 的模长（$L_2$ 范数），$\theta$ 为两向量的夹角。

* **取值范围**：$(-\infty, +\infty)$。
* **含义**：点积同时包含了“方向一致程度”与“两向量自身的模长”。方向相同且模长越大，点积越高；方向相反时，点积为负。
* **特点**：在现代硬件矩阵计算（BLAS、Tensor Core）中，点积只需要乘加操作（FMA），没有开根号与除法，吞吐性能极高。

### 3. 余弦相似度（Cosine Similarity）

余弦相似度计算的是两个向量夹角 $\theta$ 的余弦值，公式在点积的基础上除以了两者的模长：

$$\cos(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$

* **取值范围**：$[-1, 1]$。
* **含义**：只衡量两个向量的朝向是否一致，完全消除了向量长度带来的偏差：
  * $\cos(u, v) = 1$：夹角为 $0^\circ$，方向完全相同；
  * $\cos(u, v) = 0$：夹角为 $90^\circ$，向量正交无相关性；
  * $\cos(u, v) = -1$：夹角为 $180^\circ$，方向完全相反。
* **特点**：即使文本长短引起了向量模长的缩放，余弦值只看方向，因此是未归一化文本向量最常用的直观指标。

![三种向量相似度与距离度量几何含义对比](/images/content-rebuild/417c4e531b73b888/fig-1.png)

---

<ReadMoreLock />

## 模长归一化：三者在排序上的数学等价性

在工程检索中，我们频繁听到“向量归一化后点积等于余弦相似度”。这是怎么推导出来的？

$L_2$ 归一化是指将任意非零向量 $u$ 除以其自身模长，映射到半径为 1 的单位超球面上：

$$\hat{u} = \frac{u}{\|u\|_2}$$

归一化后的新向量其模长恒等于 1，即 $\|\hat{u}\|_2 = 1$。当所有向量都完成 $L_2$ 归一化后，三种度量存在严格的代数对应关系。

### 1. 余弦相似度转化为点积

当 $\|u\|_2 = 1$ 且 $\|v\|_2 = 1$ 时，代入余弦相似度定义式：

$$\cos(u, v) = \frac{u \cdot v}{1 \times 1} = u \cdot v$$

对于单位向量，**计算内积即等价于计算余弦相似度**。内积省去了每次查询时动态计算模长与分母除法的开销，能显著加快大规模匹配速度。

### 2. 欧氏距离与余弦相似度的直接换算

展开两个单位向量差值的欧氏距离平方 $\|u - v\|_2^2$：

$$
\begin{aligned}
\|u - v\|_2^2 &= (u - v) \cdot (u - v) \\
&= u \cdot u - 2(u \cdot v) + v \cdot v \\
&= \|u\|_2^2 + \|v\|_2^2 - 2(u \cdot v)
\end{aligned}
$$

将 $\|u\|_2 = 1$、$\|v\|_2 = 1$ 和 $u \cdot v = \cos(u, v)$ 带入：

$$\|u - v\|_2^2 = 1 + 1 - 2\cos(u, v) = 2 - 2\cos(u, v)$$

对两边开平方根，得到欧氏距离与余弦相似度的单调函数关系：

$$\|u - v\|_2 = \sqrt{2 - 2\cos(u, v)}$$

$$\cos(u, v) = 1 - \frac{1}{2}\|u - v\|_2^2$$

分析这个公式的取值变化：
* 当方向完全相同时（$\cos = 1$），$\|u - v\|_2 = \sqrt{2 - 2(1)} = 0$；
* 当两向量垂直正交时（$\cos = 0$），$\|u - v\|_2 = \sqrt{2} \approx 1.414$；
* 当两向量完全相反时（$\cos = -1$），$\|u - v\|_2 = \sqrt{2 - (-2)} = 2$。

![模长归一化下超球面欧氏距离与余弦相似度换算推导](/images/content-rebuild/417c4e531b73b888/fig-2.png)

因为函数 $f(x) = \sqrt{2 - 2x}$ 在区间 $[-1, 1]$ 上严格单调递减，所以在做相似内容检索时：
* **按余弦相似度从大到小排序**
* **按点积从大到小排序**
* **按欧氏距离从小到大排序**

这三种排序方式产出的结果列表在数学上是完全相同的。因此向量数据库在底层构建索引时，通常优先选用内积（Inner Product）模式。

---

## Sentence Transformers 中的计算 API 与代码验证

在官方库 `sentence-transformers` 中，计算文本相似度主要依托 `model.similarity` 与 `model.similarity_pairwise` 两个方法。

### 1. 全对相似度矩阵：`model.similarity`

用于计算两组列表之间的笛卡尔积相似度。如果输入 $M$ 个查询句子与 $N$ 个候选句子，其编码张量形状分别为 $(M, D)$ 与 $(N, D)$，返回的将是 $(M, N)$ 的二维矩阵。

### 2. 逐对单项打分：`model.similarity_pairwise`

用于评估两组一一对应的句子对（如评测数据集）。两组输入向量张量形状必须均为 $(N, D)$，返回的是形状为 $(N,)$ 的一维张量。

以下为官方调用示例及归一化等价关系的实际代码验证：

```python
import torch
from sentence_transformers import SentenceTransformer, SimilarityFunction

# 1. 加载官方通用模型
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. 准备两组句子
sentences1 = [
    "The new movie is awesome",
    "The cat sits outside",
]
sentences2 = [
    "The dog plays in the garden",
    "The new movie is so great",
    "A woman watches TV",
]

# 编码获取 Embedding 向量
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

# 3. 计算全对相似度矩阵 (形状: 2 x 3)
similarities = model.similarity(embeddings1, embeddings2)

print("全对相似度矩阵:")
for i, s1 in enumerate(sentences1):
    print(s1)
    for j, s2 in enumerate(sentences2):
        print(f"  - {s2:<30} : {similarities[i][j]:.4f}")

# 4. 验证 L2 归一化后 Dot Product 与 Cosine 的等价性
pair_a = model.encode(["气候变暖导致干旱", "TCP网络传输控制"], convert_to_tensor=True)
pair_b = model.encode(["温室效应造成降水减少", "网络通信握手协议"], convert_to_tensor=True)

# 执行 L2 归一化
norm_a = torch.nn.functional.normalize(pair_a, p=2, dim=1)
norm_b = torch.nn.functional.normalize(pair_b, p=2, dim=1)

# 计算余弦相似度与点积
cos_scores = torch.cosine_similarity(norm_a, norm_b)
dot_scores = torch.sum(norm_a * norm_b, dim=1)

print("\n归一化后余弦得分:", cos_scores)
print("归一化后点积得分:", dot_scores)
print("两者最大绝对误差:", torch.max(torch.abs(cos_scores - dot_scores)).item())
```

如果模型在导出时结构末端带有 `Normalize` 模块，其输出的向量本身就已经满足 $\|u\|_2 = 1$。此时在初始化模型时将相似度函数指定为点积即可：

```python
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", 
    similarity_fn_name=SimilarityFunction.DOT_PRODUCT
)
```

这样可以在计算相似度时直接调用矩阵乘法，省去重复的模长归一化计算。

![Sentence Transformers 全对矩阵与逐对相似度维度分解](/images/content-rebuild/417c4e531b73b888/fig-3.png)

---

## 向量空间中“语义相似”的失效边界

虽然数学公式能精确算出两组数值的夹角或距离，但开发者不能把“数学距离接近”简单等同于“人类逻辑同义”。

受对比学习目标与语言模型统计特征的影响，基于向量的相似度判定存在以下三个典型边界：

### 1. 语境共现导致的反义词聚集

嵌入模型在预训练时依据的是分布假说（具有相似上下文环境的词句享有相近表征）。然而，反义词往往出现在完全一致的句式结构中。

例如：
* `这支股票明天大概率上涨`
* `这支股票明天大概率下跌`

两句话除了“上涨”与“下跌”互为反义外，上下文词汇和语法结构完全相同。如果训练时没有加入针对反义语义的硬负例（Hard Negatives），两者的余弦相似度往往高达 0.85 以上。向量空间判定它们接近，是因为它们都属于“股市行情预测”这一话题语境，无法识别二者在经济逻辑上的对立。

### 2. 否定词钝化（Negation Blindness）

在未经专门定向微调的双塔模型中，否定词（如“不”、“没”、“禁止”）只占全句中 1~2 个 Token 的位置。经过多层 Transformer 注意力与全句均值池化（Mean Pooling）后，否定词所产生的数值影响很容易被整句其他实词稀释。

例如，“我可以接受这个条件”与“我不能接受这个条件”，两者的向量方向极其贴近。在向量检索阶段，否定句很可能被判定为高度匹配。

### 3. 高维空间的各向异性（Anisotropy）

未经过针对性校准的语言模型嵌入空间常呈现各向异性分布：所有文本向量被压缩在单位球面上一个狭窄的高维圆锥体内，而非均匀散开。这导致即使两句毫无关系的文本（如“量子纠缠现象”与“番茄炒蛋先放盐”），计算出的余弦相似度也常常处于 0.4 到 0.6 之间。绝对分值偏高并不代表文本之间具有实质关联。

![高维空间中反义词共现与否定词钝化分布陷阱](/images/content-rebuild/417c4e531b73b888/fig-4.png)

---

## 工程落地中的阈值选择与两阶段方案

在将相似度度量接入业务系统（如知识库检索、对话拦截、意图识别）时，直接拍脑袋写死一个固定的相似度阈值（例如 `if similarity > 0.8: pass`）容易引发线上异常。

### 1. 绝对分值不可跨模型通用

不同模型由于架构、训练损失（如 MultipleNegativesRankingLoss 与 CosineSimilarityLoss）和输出尺度的不同，其得分基线完全不同：
* 模型 A 在无关句子上的平均余弦相似度可能在 0.5 左右，正样本在 0.85 以上；
* 模型 B 经对比学习微调后，负样本相似度被有效拉低到 0.0 上下，正样本在 0.7 左右；
* 若使用未经归一化的负欧氏距离或曼哈顿距离，输出分值是负无穷到 0 的绝对数值。

因此，脱离了具体模型和具体评测集，任何单个相似度数值都不具备可移植性。

![不同模型相似度分布基线差异与阈值失效对比](/images/content-rebuild/417c4e531b73b888/fig-5.png)

### 2. 推荐的系统工程实践

1. **入库强制归一化**：在数据存入向量数据库前，由客户端或嵌入流水线统一执行 $L_2$ 归一化。底层索引统一配置为内积（IP），以获取硬件矩阵乘法加速。
2. **基于验证集统计选定阈值**：在业务场景中收集同义句、相关句和负样本句对，绘制准确率-召回率（Precision-Recall）曲线，依据业务对于误召回（Bad Case）与漏召回的容忍程度，选取曲线转折点对应的相似度分位点作为阈值。
3. **双塔粗筛 + 交叉注意力重排**：向量检索擅长快速从百万级候选库中圈定 Top-K 候选（双塔独立编码），但对反义词与否定词不够敏感。在核心业务流程中，应在向量检索后接入重排模型（Reranker，将 Query 与 Document 拼接后共同输入交叉注意力层），由重排模型裁决细粒度的逻辑因果与真假匹配。

![向量度量工程选型与两阶段检索流水线](/images/content-rebuild/417c4e531b73b888/fig-6.png)

掌握了向量相似度的计算方式与几何局限后，下一步我们将分析这些度量如何在实际的知识问答架构中与全文检索协同工作：请继续阅读[《Embedding 在 RAG 中有什么作用？》](/fundamentals/embedding/embedding-in-rag.md)。如果需要对比评估不同嵌入模型在公开基准上的表现差异，请参考[《文本 Embedding 模型怎么选？》](/fundamentals/embedding/embedding-model-selection.md)。
