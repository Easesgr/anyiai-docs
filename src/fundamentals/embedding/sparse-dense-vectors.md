---
title: 稀疏向量、稠密向量与多向量检索有什么区别？原理与混合检索实践
icon: mdi:chart-bubble
tag:
- Embedding
- 稀疏向量
- 稠密向量
description: 对比稀疏向量、稠密向量与多向量的数学原理、计算开销与检索表现，结合 Qdrant 演示混合检索与 RRF 融合实践。
---

在搭建检索增强生成（RAG）或企业级搜索系统时，最常遇到的问题就是：为什么用户明明搜了具体的错误码，系统却召回了一堆看似相关但毫无帮助的泛化文档？或者反过来，用户换了一种口语说法，系统就什么都搜不出来？

这些问题的根源，往往在于底层文本向量的表征方式。稀疏向量、稠密向量和多向量并不是同一套机制的简单参数变体，它们在几何结构、信息压缩程度以及计算复杂度上存在根本差异。

在阅读本文前，如果你对向量概念或距离度量还不熟悉，可以先回顾 [/fundamentals/embedding/what-is-embedding.md](/fundamentals/embedding/what-is-embedding.md) 与 [/fundamentals/embedding/embedding-similarity.md](/fundamentals/embedding/embedding-similarity.md)。本文将以一个具体的工程排障例子贯穿始终，对比这三种向量方案的工作原理、适用边界，以及如何在向量数据库中做混合检索。

![稀疏向量、稠密向量与多向量检索的核心机制概览](/images/content-rebuild/56c0fd5fd6a79c88/cover.png)

---

### 贯穿场景：一次具体的排障搜索

为了直观理解这三种向量的差异，我们设定一个具体的检索任务：

* **用户查询（Query）**：`"RTX 4090 显卡报错 0x80070057"`
* **知识库候选文档**：
  * **文档 A**：《图形工作站驱动安装指南》（包含“RTX 4090”、“显卡驱动”，但没有任何具体报错代码）。
  * **文档 B**：《Windows 更新失败问题汇总》（包含精确的错误代码“0x80070057”和通用注册表修复方法，但完全未提显卡）。
  * **文档 C**：《高端 GPU 运行时崩溃及异常排查》（包含“显卡崩溃”、“驱动故障”，并在附录中列出“错误代码 0x80070057：显存参数越界”）。

理想情况下，检索系统应当将**文档 C**排在最前面。我们来看不同的向量表征方式分别会如何处理这一场景。

---

### 稀疏向量：基于词表的离散权重映射

稀疏向量（Sparse Vector）的每个维度直接对应词表中的一个离散词项（Token）。如果一个分词词表包含 30,000 个词，那么稀疏向量就是一个 30,000 维的向量。

因为任何一段普通文本实际包含的词汇数量通常只有几十到几百个，所以这个高维向量中绝大部分维度的数值都是 0，只有文本中出现过的词对应的槽位才会有非零值。

#### 1. 传统统计稀疏表示：BM25

在传统的 BM25 算法中，稀疏向量各个非零位置的数值是由统计公式计算出的词项权重。查询 $Q$ 与文档 $D$ 的相关性得分，等于命中词项权重的线性和：

$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

其中 $f(q_i, D)$ 为词频，$\text{IDF}(q_i)$ 为逆文档频率，$|D| / \text{avgdl}$ 用于惩罚过长文档，$k_1$ 与 $b$ 是经验超参数。

* **在我们的例子中**：`"0x80070057"` 属于罕见字符串，其 IDF 值极高。BM25 会迅速命中包含该字符串的文档 B 和文档 C，但由于词表完全基于字面匹配，如果用户把“显卡报错”换成“GPU 崩溃”，BM25 就会因为字面不重合而丢分。这就是**词汇鸿沟（Vocabulary Mismatch）**。

#### 2. 神经学习型稀疏表示：SPLADE

为了保留稀疏向量的高效倒排索引特性，同时解决同义词识别问题，研究者提出了学习型稀疏模型（如 SPLADE）。

SPLADE 使用类似 BERT 的掩码语言建模（Masked Language Model, MLM）预测头。对于输入文本中的每个位置，模型不仅预测当前词本身，还会预测与其语义关联的整个词表上的概率分布。随后，模型在序列所有位置上取各词维度的最大激活值，并加上稀疏正则化惩罚：

$$w_j = \max_{t \in T} \log\left(1 + \text{ReLU}(W_j h_t + b_j)\right)$$

这种机制被称为**词汇扩展（Term Expansion）**。当文档中出现“显卡”时，SPLADE 会自动在“GPU”、“显示适配器”、“驱动”等未直接出现的词维度上激活微小的非零权重。这样既能利用倒排索引实现毫秒级检索，又能召回同义词。

![稀疏向量机制：传统统计与神经扩展的词表映射对比](/images/content-rebuild/56c0fd5fd6a79c88/fig-1.png)

---

### 稠密向量：全局语义压缩与细节模糊

与稀疏向量相反，稠密向量（Dense Embedding）将整段文本映射为一个固定长度的连续向量空间（例如 768、1024 或 1536 维）。在这个向量中，几乎所有分量都是非零的浮点数。

#### 1. 编码与池化机制（Pooling）

常见的双塔句子编码器（如基于 Sentence-BERT 架构）先通过多层 Transformer 计算所有 Token 的上下文表示序列 $[\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_L]$，然后通过均值池化（Mean Pooling）将其压缩为一个单一向量：

$$\mathbf{v} = \frac{1}{L} \sum_{i=1}^L \mathbf{h}_i$$

根据 Sentence Transformers 官方实现规范，如果模型末尾包含归一化层（Normalize），向量的模长会被约束为 1。此时计算两个向量的余弦相似度等价于直接计算点积：

```python
from sentence_transformers import SentenceTransformer, SimilarityFunction

# 加载预训练模型并指定使用点积度量（对归一化向量等价于余弦相似度，避免重复归一化开销）
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.similarity_fn_name = SimilarityFunction.DOT_PRODUCT

sentences = [
    "RTX 4090 显卡报错 0x80070057",
    "高端 GPU 运行时崩溃及异常排查",
    "Windows 更新失败问题汇总"
]
embeddings = model.encode(sentences, normalize_embeddings=True)
similarities = model.similarity(embeddings, embeddings)
```

<ReadMoreLock />

#### 2. 稠密向量的盲区：信息瓶颈与细节平滑

均值池化将长文本中所有 Token 的信息无差别地投影至有限维度中。在常规语义匹配下，这能很好地捕捉“显卡报错”与“GPU 崩溃”的同义关系。但在涉及精确标识符时，会表现出明显盲区：

1. **专有编号与错误码被平滑**：像 `0x80070057` 或 `RTX-4090-D` 这种专有名词，在 BPE 分词时会被切成离散碎片（如 `0x`、`8007`、`0057`）。经过全序列均值池化后，这些碎片的特征权重被整篇文本中的普通词稀释。模型可以识别出“这属于计算机系统报错”，但难以区分 `0x80070057` 与 `0x80070002` 的差异，从而给文档 B 甚至不相干的报错文档打出较高的相似度分值。
2. **多约束排斥能力弱**：例如查询包含否定条件“不带独立显卡的轻薄本”，自注意力和全局池化容易把“独立显卡”和“轻薄本”的概念混合在一起，反而召回包含大量独立显卡配置的近邻结果。

![稠密向量机制：信息压缩过程与精确符号特征丢失](/images/content-rebuild/56c0fd5fd6a79c88/fig-2.png)

---

### 多向量检索：ColBERT 与晚期交互机制

为了兼顾稠密向量的语义理解与精确词项的对齐能力，研究者提出了**多向量晚期交互（Late Interaction）**方案，其典型代表是 ColBERT。

#### 1. MaxSim 算子原理

ColBERT 放弃了全局均值池化，而是为输入序列的每一个 Token 保留一个独立的低维稠密向量（通常投影压缩至 128 维）：

* 查询 $Q$ 编码为矩阵 $\mathbf{E}_Q \in \mathbb{R}^{|Q| \times 128}$，包含 Token 向量 $[\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_{|Q|}]$。
* 文档 $D$ 编码为矩阵 $\mathbf{E}_D \in \mathbb{R}^{|D| \times 128}$，包含 Token 向量 $[\mathbf{d}_1, \mathbf{d}_2, \dots, \mathbf{d}_{|D|}]$。

计算匹配得分时，系统不再计算两个单向量的余弦夹角，而是使用 **MaxSim** 算子：针对查询中的每一个 Token $\mathbf{q}_i$，在文档所有 Token 中寻找与其点积最大的那一个（最佳对齐项），最后将所有查询词的最大得分相加：

$$\text{Score}(Q, D) = \sum_{i=1}^{|Q|} \max_{j \in \{1, \dots, |D|\}} \left( \mathbf{q}_i \cdot \mathbf{d}_j \right)$$

![多向量机制：ColBERT 矩阵匹配与 MaxSim 计算过程](/images/content-rebuild/56c0fd5fd6a79c88/fig-3.png)

#### 2. MaxSim 数值算例

假设简化后的查询包含 2 个 Token：`["显卡", "0x80070057"]`。文档包含 3 个 Token：`["GPU", "0x80070057", "驱动"]`。两两 Token 向量的点积矩阵如下（数值为示意点积）：

| 查询 Token | 文档 Token: GPU | 文档 Token: 0x80070057 | 文档 Token: 驱动 | 查询词行最大值（Max） |
| :--- | :--- | :--- | :--- | :--- |
| **显卡** | **0.86** | 0.05 | 0.42 | **0.86**（命中同义词 GPU） |
| **0x80070057** | 0.03 | **0.98** | 0.01 | **0.98**（精确命中代码） |

最终相关性得分为 $0.86 + 0.98 = 1.84$。  
在这一机制下，`"0x80070057"` 依靠独立的 Token 向量直接锁定了文档中的对应实体，而 `"显卡"` 依靠上下文语义锁定了 `"GPU"`。两项特征各司其职，互不冲淡。

#### 3. 三种向量形态的工程权衡

| 特性维度 | 稀疏向量（BM25 / SPLADE） | 单稠密向量（Dense Embedding） | 多向量（ColBERT） |
| :--- | :--- | :--- | :--- |
| **存储开销（每文档）** | 极低（BM25）至中等（SPLADE 倒排表） | 极低（如 768 维 float32 约 3 KB） | 较高（未量化时 200 Token $\times$ 128 维约 100 KB；量化后约 4~8 KB） |
| **检索延迟** | 毫秒级（WAND、倒排索引剪枝） | 毫秒级（HNSW、IVF-PQ 近似近邻） | 相对较高（需两阶段聚类候选过滤 + 聚合评分） |
| **精确符号匹配** | 极强（严格字面或词表约束扩展） | 较弱（容易在池化中被上下文平滑） | 强（保留单个 Token 的精确对齐能力） |
| **泛化语义理解** | 弱（BM25）至中等（SPLADE） | 强（捕获全局抽象语义） | 极强（细粒度语义与短语匹配兼顾） |
| **架构定位** | 海量初筛召回 | 通用语义召回 | 高精度召回或二阶段重排序（Reranker） |

![三种向量方案在延迟、存储与表现力维度的工程权衡](/images/content-rebuild/56c0fd5fd6a79c88/fig-4.png)

---

### 混合检索：稀疏与稠密的协同

在生产工程中，多向量检索由于存储开销和计算复杂度较高，往往部署于精排环节；而第一阶段的大规模召回，业界最成熟且高性价比的方案是**混合检索（Hybrid Search）**——同时运行稀疏检索与稠密检索分支，最后做结果融合。

#### 1. 为什么双路协同能互补？

* **防止语义漂移（Semantic Drift）**：用户输入具体型号（如“RTX 4090 显卡报错 0x80070057”），稠密分支可能因“4090”与“3080”、“显卡”与“主板”的余弦距离相近而召回其他泛化维修文档；稀疏分支凭借精确词项匹配把目标文档直接拉回前列。
* **消除词汇鸿沟（Vocabulary Mismatch）**：用户输入“显卡玩游戏跳出闪退”，但技术文档里写的是“图形渲染管线崩溃转储”，稀疏分支字面命中数为 0，此时稠密分支提供语义保底。

#### 2. 结果融合算法

双路召回的候选列表需要合并为一个统一的排序。主流方式有两种：

1. **倒数排名融合（Reciprocal Rank Fusion, RRF）**：  
   RRF 不依赖各分支的原始分数大小，仅依据文档在各分支中的排位名次计算融合分：
   $$\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
   其中 $M$ 为检索分支（如 `["dense", "sparse"]`），$r_m(d)$ 是文档 $d$ 在该分支中的排名位置（从 1 开始），$k$ 是平滑常数（通常默认取 60）。RRF 的好处在于不用担心 BM25 的无界分数与余弦相似度的有界分数不在同一尺度的问题。

2. **加权线性融合（Score Fusion）**：  
   如果将稀疏分数（如 BM25 或 SPLADE）经过 Min-Max 归一化或 Sigmoid 映射到 $[0, 1]$ 区间，可与同样在 $[0, 1]$ 区间的稠密相似度按权重相加：
   $$S_{\text{final}}(d) = \alpha \cdot S_{\text{dense}}(d) + (1 - \alpha) \cdot S_{\text{sparse}}(d)$$

![混合检索融合策略：RRF 排名融合与分数线性加权流程](/images/content-rebuild/56c0fd5fd6a79c88/fig-5.png)

---

### Qdrant 混合检索实践与调优建议

现代向量数据库（以 Qdrant 为例）允许在一个 Collection 中同时存储命名稠密向量与稀疏向量，并在单次查询接口内完成双路并发检索与 RRF 融合。

#### 1. 原生双路并发与 RRF 代码示例

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(location=":memory:")
collection_name = "hybrid_kb"

# 1. 创建包含 dense 与 sparse 双向量通道的集合
client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "dense": models.VectorParams(size=768, distance=models.Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(index=models.SparseIndexParams())
    }
)

# 2. 写入示例数据：包含稠密特征与离散词项索引
client.upsert(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id=1,
            vector={
                "dense": [0.01] * 768,  # 示意：实际由稠密 Embedding 模型输出
                "sparse": models.SparseVector(
                    indices=[101, 8890, 2045],  # 词表中分词对应的 ID（如：4090, 0x80070057 等）
                    values=[1.2, 3.5, 0.8]      # SPLADE 预测权重或 BM25 权重
                )
            },
            payload={"title": "高端 GPU 运行时崩溃及异常排查（附 0x80070057 说明）"}
        )
    ]
)

# 3. 使用 prefetch 执行双路召回并使用原生 RRF 算法融合
query_dense = [0.012] * 768
query_sparse = models.SparseVector(indices=[8890, 2045], values=[3.2, 0.9])

results = client.query_points(
    collection_name=collection_name,
    prefetch=[
        models.Prefetch(query=query_dense, using="dense", limit=20),
        models.Prefetch(query=query_sparse, using="sparse", limit=20),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10
)

for hit in results.points:
    print(f"Hit ID: {hit.id}, Payload: {hit.payload}")
```

#### 2. 调优策略与避坑事项

在生产部署混合检索时，需要关注以下两点：

1. **不可直接将原始 BM25 分数与稠密余弦分数相加**：稠密余弦值被约束在 $[-1, 1]$ 之间，而 BM25 得分取决于词频和文档长度，无理论上界。如果直接相加，未归一化的稀疏分将完全覆盖稠密分支。如果选择加权融合，通常需要先通过 Sigmoid 映射或分数分位数对齐等方式将各路结果归一至相近尺度（若使用批内 Min-Max 归一化，需注意在单次查询无强相关结果时可能放大低分噪声）；若不想维护分数对齐管道，优先使用 RRF。
2. **以评估指标驱动权重设置**：不要凭空设定“稠密 0.5、稀疏 0.5”。在专业知识库（如代码搜索、法律文书、零件型号）中，专有名词的判别权重往往更高；在日常闲聊与客服意图识别中，稠密语义的权重更高。应当划分独立的验证集与测试集（例如分别包含带真实标注的查询），以 MRR@10 或 NDCG@10 为优化指标，在验证集上进行超参数网格搜索，最后在测试集上验证泛化效果，防止因在测试集上调优导致数据泄漏与过拟合。

![Qdrant 混合检索工程架构与基于指标评测的权重调优闭环](/images/content-rebuild/56c0fd5fd6a79c88/fig-6.png)

关于文本 Embedding 模型的选择策略，可参考 [/fundamentals/embedding/embedding-model-selection.md](/fundamentals/embedding/embedding-model-selection.md)；若需进一步了解检索召回指标的评测方法与调优路径，见 [/fundamentals/embedding/embedding-optimization.md](/fundamentals/embedding/embedding-optimization.md)。
