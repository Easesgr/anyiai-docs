---
title: Embedding 是什么？文本向量学习指南与知识地图
icon: mdi:vector-square
tag:
- Embedding
- 向量检索
- 大模型基础
description: 以具体例句和官方代码切入，解释文本 Embedding 的表示、度量与工程检索机制，梳理模块六篇专题文章的学习路径。
---

![文本 Embedding 模块导读与学习路径全景](/images/content-rebuild/c542d4535bdab392/cover.png)

在计算机程序中，文本最初只是一串离散的字符编码。计算机无法直接判断“这部电影太棒了”与“这部新片真精彩”在含义上有多接近，因为它们的字面用词完全不同。

**Embedding（嵌入，也常直接称为“向量”）**的核心作用，就是把这些离散的文本符号映射到连续的数值向量空间中。映射完成后，两段文本语义是否相似，就可以通过计算两个向量在几何空间中的夹角或距离来评估。

本模块是「安逸Ai学习站·基础教材」的一部分，旨在为开发者梳理从“离散文本表示”到“几何度量计算”，再到“生产环境向量检索与重排”的技术脉络。

---

## 一个直观的句子相似度例子

为了直观理解 Embedding 的工作方式，我们可以看一组实际文本比对。假设有以下三句话：

1. 句子 A：`The new movie is awesome`（这部新电影太棒了）
2. 句子 B：`The new movie is so great`（这部新电影太赞了）
3. 句子 C：`A woman watches TV`（一位女士在看电视）

如果采用传统的 One-Hot（独热）编码，每个单词对应一个独立维度。在 One-Hot 视角下，`awesome` 和 `great` 是两个互不相关的正交向量，它们之间的夹角是 90 度，欧氏距离恒为 $\sqrt{2}$，余弦相似度恒为 0。这种表示无法体现词义的相近。

而借助现代句向量模型（例如开源的 `sentence-transformers/all-MiniLM-L6-v2`），我们可以把每个句子编码为一个 384 维的稠密浮点数向量。计算余弦相似度后，结果如下：

- 句子 A 与 句子 B 的相似度：**0.8939**（语义高度相近）
- 句子 A 与 句子 C 的相似度：**-0.0502**（语义毫无关联）

模型并未硬编码任何同义词词典，而是在大量文本训练中，学习到了“awesome”和“great”经常出现在类似的上下文语境中，因此给它们分配了空间方向相近的向量。这就是连续表示的核心机制。

---

## 学习路线：从表示、度量到检索工程

在工程落地中，使用 Embedding 通常涉及三个层级的问题：

![从连续表示到度量计算与工程落地的演进层级](/images/content-rebuild/c542d4535bdab392/fig-1.png)

### 1. 向量表示层（怎么把文本变成向量）
文本首先被切分为 Token（词元），模型根据上下文将离散 Token 转化为一组高维连续浮点数，再通过池化（如均值池化 Mean Pooling）生成代表整句的定长向量。我们需要理解词向量（如 Word2Vec）与上下文句向量（如 BERT、Sentence Transformers）的差异，以及定长向量在压缩长文本时可能面临的容量边界。

### 2. 度量计算层（怎么比较两个向量）
拿到向量后，如何定义“相似”？常见的度量指标有余弦相似度（Cosine Similarity）、点积（Dot Product）和欧氏距离（Euclidean Distance）：
- 当向量进行了 L2 归一化（模长为 1）时，**点积与余弦相似度在数学上完全等价**。
- 在实际计算中，若模型输出端已经内置了归一化模块，直接计算点积比重复进行余弦归一化更节省算力。

### 3. 检索工程层（海量数据下如何快速查找）
当数据库中有上百万个文本块时，逐一进行向量计算（暴力搜索）延迟过高。此时需要引入专门的向量数据库或近似最近邻（ANN）索引算法（如 HNSW、IVF）。

同时需要明确：Embedding 擅长抓取跨用词的“语义泛化”匹配，但在精确的产品型号、身份证号、特定错误码匹配上，往往不如基于字面统计的倒排索引（如 BM25）。因此，在生产检索系统（如 RAG）中，通常采用“稠密向量检索 + 稀疏关键词检索”的多路召回，再配合 Cross-Encoder 重排器（Reranker）纠偏。

---

## 模块文章地图与依赖关系

本模块包含 6 篇彼此承接的专题文章，建议按以下脉络阅读：

![Embedding 模块六篇文章内容地图与依赖关系](/images/content-rebuild/c542d4535bdab392/fig-2.png)

1. [/fundamentals/embedding/what-is-embedding.md](/fundamentals/embedding/what-is-embedding.md)  
   **核心议题**：从独热编码的局限推导稠密向量的必要性，解释词向量与句向量的生成机制。  
   **解决问题**：建立对连续空间、特征维度与文本压缩的基本认识。

2. [/fundamentals/embedding/embedding-similarity.md](/fundamentals/embedding/embedding-similarity.md)  
   **核心议题**：余弦相似度、点积与欧氏距离的几何含义、等价条件与计算开销。  
   **解决问题**：搞清不同度量公式的适用场景，以及为什么归一化后点积运算速度更快。

3. [/fundamentals/embedding/embedding-in-rag.md](/fundamentals/embedding/embedding-in-rag.md)  
   **核心议题**：文本向量在 RAG（检索增强生成）全流程中的定位，包括切块、入库、召回与注入大模型。  
   **解决问题**：辨析检索质量对大模型生成结果的制约关系，明确向量初筛的边界。

4. [/fundamentals/embedding/sparse-dense-vectors.md](/fundamentals/embedding/sparse-dense-vectors.md)  
   **核心议题**：稠密向量与稀疏向量（BM25、SPLADE）的机制对比与互补方案。  
   **解决问题**：解决长尾专有名词、代码符号在纯向量检索下容易漏召回的问题，掌握混合检索实现路径。

5. [/fundamentals/embedding/embedding-model-selection.md](/fundamentals/embedding/embedding-model-selection.md)  
   **核心议题**：如何根据语言支持、上下文窗口截断、吞吐速度与显存消耗选拔模型。  
   **解决问题**：脱离单纯的榜单分数（如 MTEB），建立结合自身硬件条件与业务场景的选型标准。

6. [/fundamentals/embedding/embedding-optimization.md](/fundamentals/embedding/embedding-optimization.md)  
   **核心议题**：切分粒度调优、Cross-Encoder 重排、难负例挖掘与领域微调。  
   **解决问题**：在生产召回率不足或排序颠倒时，给出具体的排查步骤与调优方案。

---

## 官方规范验证：在 Python 中运行你的第一个向量比对

在现代开发中，基于开源库 [Sentence Transformers](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html) 可以用几行代码完成句向量的生成与比对。

![基于 Sentence Transformers、Qdrant 与 LlamaIndex 的分层技术流水线](/images/content-rebuild/c542d4535bdab392/fig-3.png)

以下是官方给出的标准验证代码：

```python
from sentence_transformers import SentenceTransformer, SimilarityFunction

# 1. 加载轻量级英文预训练模型，并显式指定默认相似度函数为余弦相似度
# 注：该模型适合英文；若测试中文任务，建议选用多语言或中文模型（如 BAAI/bge-small-zh-v1.5）
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    similarity_fn_name=SimilarityFunction.COSINE
)

# 2. 准备两组待比对的句子
sentences1 = [
    "The new movie is awesome",
    "The cat sits outside",
    "A man is playing guitar",
]
sentences2 = [
    "The dog plays in the garden",
    "The new movie is so great",
    "A woman watches TV",
]

# 3. 将文本编码为稠密向量（返回 numpy 数组或 PyTorch Tensor）
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

# 4. 计算成对的余弦相似度矩阵（输出形状为 3x3）
similarities = model.similarity(embeddings1, embeddings2)

# 5. 打印两两比对结果
for i, s1 in enumerate(sentences1):
    print(f"基准句: {s1}")
    for j, s2 in enumerate(sentences2):
        print(f"  - 候选句: {s2:<30} | 相似度: {similarities[i][j]:.4f}")
```

运行上述代码后，官方文档记录的标准输出矩阵如下：

```text
基准句: The new movie is awesome
  - The dog plays in the garden    | 相似度: 0.0543
  - The new movie is so great      | 相似度: 0.8939
  - A woman watches TV             | 相似度: -0.0502
基准句: The cat sits outside
  - The dog plays in the garden    | 相似度: 0.2838
  - The new movie is so great      | 相似度: -0.0029
  - A woman watches TV             | 相似度: 0.1310
基准句: A man is playing guitar
  - The dog plays in the garden    | 相似度: 0.2277
  - The new movie is so great      | 相似度: -0.0136
  - A woman watches TV             | 相似度: -0.0327
```

从结果中可以清楚看到两个规律：
1. `The new movie is awesome` 与 `The new movie is so great` 虽然用词不同，但得到了 `0.8939` 的高分；
2. `The cat sits outside` 与 `The dog plays in the garden` 虽然不是完全同义，但因为都涉及“动物在户外活动”，得分达到了 `0.2838`，明显高于与看电视、看电影等无关话题的分数。

这就是 Embedding 捕捉文本语义关系的基本方式。建议先阅读第一讲 [/fundamentals/embedding/what-is-embedding.md](/fundamentals/embedding/what-is-embedding.md)，从离散符号与连续空间的几何原理开始系统的学习。
