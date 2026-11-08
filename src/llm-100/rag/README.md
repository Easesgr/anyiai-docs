---
title: RAG 检索增强怎么做？从切分、混合检索到重排与评估
icon: mdi:database
description: 介绍 RAG 系统的四阶段技术流程：文档切分、索引构建、多路召回重排与量化评估，包含加权 RRF 算例与可检验代码。
---

![RAG 检索增强模块整体知识拓扑与进阶学习路线](/images/content-rebuild/a755482531c14dbb/cover.png)

### 为什么需要 RAG：从一个具体查询说起

大语言模型的权重在预训练和微调完成后便固定下来。当遇到私有文档、频繁变动的配置规范或精准技术代码时，模型无法仅靠自身参数稳定答对。

例如，用户询问：“如何在配置文件中设置 `storage.wal.dir` 参数，并将预写日志的最大保留大小限制为 10GB？”

如果直接让模型作答，模型可能会依据通用命名习惯编造出类似 `wal_storage_path` 或 `max_wal_size` 的虚构键名。检索增强生成（Retrieval-Augmented Generation，RAG）的做法是：先从企业的技术手册中检索出包含 `storage.wal.dir` 的真实文档片段，再把这些片段与用户的问题拼接成提示词提交给模型。模型以检索到的片段作为依据生成答复，降低凭空编造的概率。

但要把这套流程稳定跑在实际应用中，不能只做一次简单的文本搜索。分块太粗会导致无关信息淹没模型注意力；分块太细会导致上下文含义断裂；只用语义向量检索容易遗漏特定的英文参数名，而只用关键词检索又无法理解相近语意的同义表述。本模块将围绕这些实际问题，梳理从非结构化文档到可用问答系统的完整实现过程。

---

### RAG 流水线的四个阶段

一个完整的 RAG 系统通常分为四个数据处理与交互阶段：

![RAG 四阶段技术流水线与双向反馈拓扑](/images/content-rebuild/a755482531c14dbb/fig-1.png)

1. **数据准备与分块（Chunking）**：清洗 Markdown、PDF 或 HTML 文本，按照语义边界（如标题、段落、代码块）或固定长度将长文切分成文本块。切分的目标是在保持单块语意相对完整的前提下，控制单块长度，使其符合检索粒度需求。
2. **表征嵌入与索引存储（Indexing）**：利用文本嵌入模型（Embedding Model）把文本块转换为高维稠密向量，写入向量数据库（构建 HNSW 或 IVF 索引）；同时对文本提取关键词，构建 BM25 倒排索引，并记录文件路径、所属模块等元数据以备属性过滤。
3. **多路召回与二次精排（Retrieval & Rerank）**：用户发起查询时，系统分别执行稠密向量检索与 BM25 稀疏检索，通过倒排排名融合（RRF）等算法合并候选结果；随后用交叉编码器（Cross-Encoder）重排模型对候选块进行深度语义匹配评分，截取相关性最高的前若干篇放入模型提示词中。
4. **失效归因与评测优化（Evaluation）**：在线下测试集上，分别衡量检索阶段的召回表现（如 Hit Rate、MRR）与生成阶段的保真度（Faithfulness，即回答是否均可由检索上下文支撑），依据指标反推分块、索引或检索策略的短板。

---

### 十讲课程目录与进阶规划

本模块分为十篇教程，建议按流水线的数据流动顺序依次学习：

| 序号 | 课程文档 | 学习目标 | 对应产出 |
| :--- | :--- | :--- | :--- |
| 01 | [/llm-100/rag/what-is-rag.md](/llm-100/rag/what-is-rag.md) | 理解外部知识注入上下文的工作方式与核心链路 | 运行最小端到端检索生成示例 |
| 02 | [/llm-100/rag/rag-vs-finetuning.md](/llm-100/rag/rag-vs-finetuning.md) | 区分 RAG 与模型微调（SFT）的适用条件与成本结构 | 架构选型技术决策对比表 |
| 03 | [/llm-100/rag/chunk-strategy.md](/llm-100/rag/chunk-strategy.md) | 掌握固定长度、滑动窗口与基于结构的分块策略 | 分块长度与重叠率测试对比用例 |
| 04 | [/llm-100/rag/vector-db-in-rag.md](/llm-100/rag/vector-db-in-rag.md) | 了解 HNSW/IVF 索引原理与元数据过滤的过滤时机 | 带标量过滤的向量库建库与查询配置 |
| 05 | [/llm-100/rag/embedding-selection.md](/llm-100/rag/embedding-selection.md) | 掌握向量嵌入模型选型考量、维度影响与微调评估 | 嵌入相似度分布分析脚本 |
| 06 | [/llm-100/rag/bm25-in-rag.md](/llm-100/rag/bm25-in-rag.md) | 掌握 BM25 词频与逆文档频计算逻辑及其适用场景 | 纯 Python 实现的 BM25 算例 |
| 07 | [/llm-100/rag/retrieval-methods.md](/llm-100/rag/retrieval-methods.md) | 掌握密集检索与稀疏检索的多路召回与融合逻辑 | 基于多路 Prefetch 的混合检索查询构造 |
| 08 | [/llm-100/rag/rerank-importance.md](/llm-100/rag/rerank-importance.md) | 区分双塔模型（Bi-Encoder）与交叉编码器（Cross-Encoder） | 引入重排后的检索排序评估对比 |
| 09 | [/llm-100/rag/rag-wrong-answers.md](/llm-100/rag/rag-wrong-answers.md) | 排查检索召回不全、排序靠后、提示词截断与模型幻觉 | 4 维度排障核对表与排错日志设计 |
| 10 | [/llm-100/rag/optimize-rag.md](/llm-100/rag/optimize-rag.md) | 掌握查询改写（HyDE、多查询扩展）与分层检索方案 | 端到端检索生成评测流程 |

---

### 混合检索实战：加权 RRF 融合机制

当系统同时运行稠密向量检索与 BM25 检索时，两者的原始得分量纲不同（稠密检索一般是余弦相似度或内积，BM25 是基于词频的正实数）。不能直接把两个分值相加，常用倒排排名融合（Reciprocal Rank Fusion，RRF）依据文档在各路检索中的名次进行合并。

![加权倒排排名融合（RRF）打分与重排计算机制](/images/content-rebuild/a755482531c14dbb/fig-2.png)

#### 计算公式

对于候选文档 $d$，若它在稠密检索中的名次为 $r_{dense}(d)$，在稀疏检索中的名次为 $r_{sparse}(d)$（名次从 1 开始计），平滑常数通常取 $k = 60$：

$$RRF\_Score(d) = w_{dense} \cdot \frac{1}{k + r_{dense}(d)} + w_{sparse} \cdot \frac{1}{k + r_{sparse}(d)}$$

如果某篇文档在某一分支未被召回，则该分支对应的项不计入或按极大名次处理。

#### 逐步推导算例

在包含专有配置参数的场景中，关键词匹配通常比向量模糊匹配更可靠，因此对稀疏检索赋予更高的相对权重，设 $w_{dense} = 1.0$，$w_{sparse} = 1.5$，$k = 60$。

检索系统召回了 3 篇候选文档，各自在两个分支中的排名如下：
- 文档 $d_1$：语义相近但缺少精确配置项，稠密排名第 1，稀疏排名第 3；
- 文档 $d_2$：包含精准参数 `storage.wal.dir`，稠密排名第 2，稀疏排名第 1；
- 文档 $d_3$：常规说明文档，稠密排名第 3，稀疏排名第 2。

依次代入公式计算分值：

$$
\begin{aligned}
Score(d_1) &= 1.0 \times \frac{1}{60 + 1} + 1.5 \times \frac{1}{60 + 3} \approx 0.016393 + 0.023810 = 0.040203 \\
Score(d_2) &= 1.0 \times \frac{1}{60 + 2} + 1.5 \times \frac{1}{60 + 1} \approx 0.016129 + 0.024590 = 0.040719 \\
Score(d_3) &= 1.0 \times \frac{1}{60 + 3} + 1.5 \times \frac{1}{60 + 2} \approx 0.015873 + 0.024194 = 0.040067
\end{aligned}
$$

计算结果为：$Score(d_2) > Score(d_1) > Score(d_3)$。虽然 $d_1$ 在单路向量检索中排名第一，但综合两路排名并叠加稀疏分支的更高权重后，$d_2$ 在最终排序中位居首位，保证了带有精确配置项的文档优先送入模型。

---

### 可检验练习：验证加权 RRF 对排名的影响

你可以运行以下纯 Python 代码，观察不同权重设置如何改变最终文档的排布：

```python
def compute_weighted_rrf(rank_lists: dict, weights: dict, k: int = 60) -> list:
    """
    rank_lists: 字典，键为检索分支名，值为按排序从前到后排列的文档ID列表
    weights: 字典，键为检索分支名，值为权重浮点数
    k: 平滑常数，默认 60
    """
    scores = {}
    for branch_name, doc_ids in rank_lists.items():
        w = weights.get(branch_name, 1.0)
        for rank, doc_id in enumerate(doc_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w * (1.0 / (k + rank))
    
    # 按得分从高到低排序
    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs

# 模拟两路召回结果
retrieval_results = {
    "dense": ["d1", "d2", "d3"],
    "sparse": ["d2", "d3", "d1"]
}

# 实验 1：等权重 (1.0, 1.0)
res_equal = compute_weighted_rrf(retrieval_results, {"dense": 1.0, "sparse": 1.0})
print("等权重结果:", [(doc, round(score, 6)) for doc, score in res_equal])

# 实验 2：偏向稀疏 (1.0, 1.5)
res_weighted = compute_weighted_rrf(retrieval_results, {"dense": 1.0, "sparse": 1.5})
print("加权稀疏结果:", [(doc, round(score, 6)) for doc, score in res_weighted])
```

运行上述代码可验证：在等权重时，$d_2$ 凭借两路均处前列的综合名次（第 2 与第 1）已略微领先在稀疏路排至第 3 的 $d_1$；而为精确关键词检索配置更高的稀疏权重后，$d_2$ 相比 $d_1$ 的领先分差进一步扩大，使包含专有配置项的文档排序优势更加稳固。

---

### 工程基线验收清单

在将 RAG 服务交付给上层业务前，应准备带有标准答案的评测集，核对三项基本指标：

![RAG 工程上线验收标准与进阶演进路线](/images/content-rebuild/a755482531c14dbb/fig-3.png)

1. **前 K 位命中率（Hit Rate@K）**：在评测集中，标准答案所在的文档块出现在前 K 个召回结果里的样本比例。如果 Hit Rate@5 低于预设阈值，说明大部分错误源于检索漏召回，需检查分块粒度或混合检索配置。
2. **平均倒数排名（MRR@K）**：衡量第一篇相关文档排名的倒数均值。如果标准文档虽然被召回但总是排在第 4、第 5 位，模型可能因为上下文靠后而忽视关键信息，这时需要引入或优化重排模型（Reranker）。
3. **上下文保真度（Faithfulness）**：通过规则检测或评测模型判断生成的句子是否有对应的检索证据支持。保真度低说明模型在证据不足时出现了幻觉，需要调整提示词中的约束规则或提高检索相关性截断阈值。

#### 进阶方向衔接

掌握本模块的十讲基础后，如果你的系统需要应对多租户权限隔离、高并发缓存与网关路由，请参考架构进阶专栏 [/architecture/README.md](/architecture/README.md)；若需要大模型根据提问自主规划检索次数、按需调用多个外部工具或自检结果，可继续学习智能体检索篇 [/frontier/agentic-rag/README.md](/frontier/agentic-rag/README.md) 与上下文工程篇 [/frontier/context-engineering/README.md](/frontier/context-engineering/README.md)。
