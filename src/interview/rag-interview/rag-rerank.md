---
title: 什么是 RAG 中的 Rerank？具体需要怎么做？
icon: mdi:sort
tag:
  - RAG
  - Rerank
  - 重排序
---

# 什么是 RAG 中的 Rerank？具体需要怎么做？

这道题考的是 RAG 系统中文档检索的精排环节。面试官想看你能不能把"粗筛到精筛"这个链路讲清楚，知道什么时候该用 Rerank、用什么方案、怎么调参。

我从四个方面来讲：Rerank 是什么、为什么需要它、三种主流方案怎么选、 工程实践怎么落地。

---
## Rerank 是什么——RAG 漏斗的最后一步

Rerank 中文叫"重排序"，干的事情很简单：对初筛出来的候选文档重新打分、重新排序。

想象一下你们公司招聘：
- HR 先用关键词快速筛简历，筛出 100 份
- 技术 leader 认真看这 100 份，选出 5 份进入面试

RAG 系统也是这个思路：

1. **文档库** → 全量待检索文档
2. **向量检索（粗筛）** → 用 Bi-Encoder 快速召回 Top-20~100
3. **Rerank（精筛）** → 用更精准的模型重新排序，取 Top-3~5
4. **LLM 生成** → 把精筛后的文档喂给大模型回答问题

漏斗越往下，文档越少，但质量越高。Rerank 就是这个漏斗最底部的精细过滤器。

---
<ReadMoreLock />

## 为什么需要 Rerank——Bi-Encoder 的局限

这里有个关键问题：向量检索（Bi-Encoder）不是挺好吗，为什么还要 Rerank？

因为 Bi-Encoder 有个天生缺陷——**把文档压缩成了一个向量**。

### Bi-Encoder 的工作方式

Query 和 Document 分别独立编码成向量，然后算余弦相似度。

你搜"Python 内存溢出"，可能出现这些文档：
- ✅ Python 内存溢出原因及排查
- ✅ Python 内存管理机制详解
- ❌ Java 内存溢出与 Python 的关系
- ❌ Python 性能优化指南（只提到"溢出"一次）

向量模型可能把第二和第三篇都判为高相关，因为它们在向量空间里和 Query 比较接近。问题出在哪？**独立编码没有让 Query 和 Document 真正"对话"**。

### Cross-Encoder 怎么解决

Cross-Encoder 的做法是：把 Query 和 Document **拼在一起**，一起过模型，做全注意力交互。

模型能看到 Query 的每个词和 Document 的每个词之间的关系。它能精准判断：
- "Python 内存溢出"和"Java 内存溢出"虽然语义相关，但 Query 明确说的是 Python
- "Python 内存管理"虽然有"管理"但没提"溢出"，可能比"Java相关"更相关

类比一下：
- **Bi-Encoder** 像只看书的封面和标题就决定买不买
- **Cross-Encoder** 像把书翻到目录，对着目录问自己"这书讲的是不是我要的"

Cross-Encoder 慢，但准。它不需要在海量文档里大海捞针，只用对 Top-20~100 做精排，开销完全可接受。

---
## 三大技术方案——怎么选

### 方案一：Cross-Encoder（主流推荐）

这是目前最常用的方案。速度、精度、成本三方面最平衡。

**原理**：Query + Document 拼接 → 过 Transformer → 输出相关性分数

**中文推荐模型**：
- **BGE-Reranker**（BAAI出品，中文效果好）
- **bce-reranker-base**（效果相近，可对比实测）

**英文/多语言**：
- Cohere Rerank（商业 API，效果好但收费）
- Jina Reranker（性价比不错）
- cross-encoder/ms-marco 系列（开源老牌）


# 使用示例（LlamaIndex）
from llama_index.postprocessor.cohere_rerank import CohereRerank

node_postprocessor = CohereRerank(
    api_key=cohere_api_key,
    top_n=5,  # Rerank 后取 Top-5
    model="rerank-multilingual-v2.0"
)


### 方案二：LLM 重排序（精度最高，成本最高）

让大模型直接判断文档相关性，精度确实更高。

代表方案是 **RankGPT**，但实测下来：
- **GPT-4** 效果明显
- GPT-3.5 勉强能用
- 其他开源模型效果一般

问题是贵。每个 Query 要让 LLM 评分 20~50 个文档，Token 消耗不小。适合对精度要求极高、愿意花钱的场景。


# 伪代码逻辑
for doc in candidate_docs:
    prompt = f"Query: {query}\nDoc: {doc}\n判断是否相关，返回 yes/no"
    response = gpt4(prompt)
    score = 1 if "yes" in response else 0


### 方案三：多特征融合（轻量级备选）

不依赖单独的 Rerank 模型，综合多个信号打分：
- BM25 分数
- 文档新鲜度
- 原始向量相似度
- 关键词命中

这种方案快，但天花板低——多个弱信号加起来还是弱信号。适合资源极度受限、或者作为 ensemble 的一部分。

### 怎么选

| 场景 | 推荐方案 |
|------|----------|
| 中文业务系统，常规精度要求 | BGE-Reranker / bce-reranker |
| 英文，多语言 | Cohere Rerank / Jina |
| 追求最高精度，不差钱 | GPT-4 + RankGPT |
| 资源极度受限 | 多特征融合 |
| 想快速验证 | 先用开源 Cross-Encoder 跑 demo |

---

## 工程实践——候选集大小与动态阈值

### 候选集大小怎么定

初检索取多少个文档进 Rerank？

这是个典型的 **召回率 vs 延迟** 的权衡：

- **取 Top-20**：快，但可能漏掉相关文档
- **取 Top-50**：平衡，大多数场景用这个
- **取 Top-100**：召回率高，但 Rerank 延迟明显增加

建议：先用 **Top-20~50**，跑一段时间后看命中率曲线，找到拐点。

有个简单方法：在验证集上跑不同 Top-K，算 **MRR@K**（平均倒数排名），选曲线开始平缓的地方作为最优 K。

### 分数阈值要设

Cross-Encoder 输出的分数一般在 0~1 之间。

**低于 0.3 说明库里可能真的没有相关文档**，这时候应该：
- 友好地告知用户"暂时没找到相关信息"
- 而不是硬把不相关的内容塞给 LLM 生成

后者会导致 LLM 胡编，专业点叫"幻觉"。

### 框架集成

LlamaIndex 和 LangChain 都支持直接插入 Rerank 节点：

python
# LlamaIndex 完整示例
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.postprocessor import SentenceTransformerRerank

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 构建索引
index = VectorStoreIndex.from_documents(documents)

# 配置 Rerank
rerank = SentenceTransformerRerank(top_n=5, model="BAAI/bge-reranker-base")

# 查询时带上 Rerank
query_engine = index.as_query_engine(
    similarity_top_k=30,  # 初检 Top-30
    node_postprocessors=[rerank]  # Rerank 精筛到 Top-5
)

response = query_engine.query("Python 内存溢出怎么处理")


python
# LangChain  equivalent
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

compressor = CrossEncoderReranker(
    model_name="BAAI/bge-reranker-base", 
    top_n=5
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)


---

## 面试怎么答

### 基础版（能过的回答）

> Rerank 是 RAG 检索链路中的精筛环节，对向量检索召回的 Top-20~100 个候选文档做重新排序，取 Top-3~5 送入 LLM。
>
> 之所以需要它，是因为向量检索用的 Bi-Encoder 把文档压缩成向量，会丢失细粒度语义。Cross-Encoder 让 Query 和 Document 一起编码，能精确判断它们之间的相关性。
>
> 主流方案是 Cross-Encoder，中文场景推荐 BGE-Reranker 或 bce-reranker，英文可以用 Cohere Rerank。工程上，初检取 Top-30 左右进 Rerank，Rerank 后取 Top-5；Cross-Encoder 分数低于 0.3 说明没有相关文档，这时候不要硬塞给 LLM。

### 加分版（让面试官眼前一亮）

> 在基础版的基础上，可以补充几点：
>
> **方案选型**：Cross-Encoder 是精度和成本的平衡点；LLM 重排（RankGPT）精度最高但只有 GPT-4 效果明显，适合不差钱的场景；多特征融合快但天花板低。
>
> **工程细节**：Top-K 不是拍脑袋定的，是用验证集跑 MRR/NDCG 曲线找拐点。Rerank 分数要设阈值，低于阈值直接返回"未找到"而不是产生幻觉。
>
> **模型选型**：中文必测 BGE-Reranker 和 bce-reranker，实测对比；多语言用 Cohere 或 Jina；资源受限再看 sentence-transformers 轻量模型。

---

## 一句话总结

Rerank 就是用更精准的模型对初筛文档做精排，主流用 Cross-Encoder，核心是平衡精度、速度和成本。
