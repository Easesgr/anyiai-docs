---
title: 为什么 RAG 常用 BM25 和向量检索做混合召回？
icon: mdi:call-merge
tag:
- RAG面试
- BM25
- 混合召回
description: 从词汇匹配与语义泛化的互补原理出发，拆解 BM25 与稠密向量混合检索的适用场景、融合打分与工程并发实现。
---

在检索增强生成（RAG）系统中，候选文档召回通常会同时运行 BM25 关键词检索和稠密向量检索两路方案。面试中，如果只回答“两者可以互补”往往不够深入，需要进一步说明：两类检索各自基于什么假设、在哪些具体场景下会漏检或错检，以及工程上如何并发拉取并合并候选集。

在阅读本文前，建议了解 [检索方式对比](/interview/rag-interview/retrieval-methods.md) 中的检索范式划分；后续进阶内容可参考 [混合检索实现与融合算法](/interview/rag-interview/rag-hybrid-retrieval.md) 和 [Rerank 精排机制](/interview/rag-interview/rerank.md)。

![BM25 与稠密向量双路混合检索核心机制](/images/content-rebuild/bbf24d3bd241153a/cover.png)

---

## 1. 核心口述答案

如果面试官提问“为什么 RAG 不单独使用向量检索，而是常用 BM25 + 向量检索的混合召回”，可以按以下逻辑回答：

> **面试口述要点**：
>
> 1. **两者的核心假设不同**：BM25 基于离散词频与逆文档频率（TF-IDF 思想）进行精确字词匹配；稠密向量检索通过神经网络将文本编码为固定长度向量，基于空间几何距离（如余弦相似度）计算语义接近程度。
> 2. **向量检索的盲区**：当查询包含专有名词、错误码、零件型号或冷门人名时，分词器容易将词切碎（Subword Tokenization），且自注意力权重往往被句中常见词主导，导致语义向量无法精准锁定目标编号，出现误召回。
> 3. **BM25 的盲区**：当用户提问存在同义词替换、口语化表达或意图抽象时，查询词与知识库文档的字面重合度很低，BM25 会因为“词汇不匹配（Vocabulary Mismatch）”导致漏检。
> 4. **双路协同方案**：在初筛阶段并行调用 BM25 和向量检索，各取 Top-K 候选；再使用倒数排名融合（RRF）或分数归一化算法合并，交由下游重排模型（Reranker）筛选，兼顾精准匹配与语义理解。

![语义空间与词法倒排索引的互补覆盖域](/images/content-rebuild/bbf24d3bd241153a/fig-1.png)

*上图展示了向量语义空间与 BM25 词法索引在不同查询类型下的覆盖分布。正文接下来拆解两种机制的数学与表征因果。*

---

## 2. 贯穿案例：数据库错误排查

为了直观理解两者的差异，我们以一个技术支持场景为例：

- **文档 A**：《Oracle 数据库常见 SQL 执行报错诊断》（包含 `ORA-00904: invalid identifier`，即列名无效）；
- **文档 B**：《Oracle 权限与表对象排查手册》（包含 `ORA-00942: table or view does not exist`，即表或视图不存在）；
- **文档 C**：《MySQL 1146 Table doesn't exist 排查思路》。

假设用户输入两个不同的查询：

- **查询 1（具体代码）**：`Oracle ORA-00942 报错排查`
- **查询 2（口语化表达）**：`数据库提示找不到表怎么处理`

不同检索方式在两个查询下的表现差异明显：

| 查询类型 | BM25 表现 | 稠密向量表现 | 混合检索表现 |
| :--- | :--- | :--- | :--- |
| **查询 1**：`Oracle ORA-00942 报错排查` | **命中 文档 B**。`ORA-00942` 作为罕见词，逆文档频率（IDF）极高，精准命中。 | **容易偏向 文档 A 或 C**。模型将注意力集中在“Oracle/报错/排查”等高频词上，`00942` 与 `00904` 向量距离接近，区分度不足。 | **准确命中 文档 B**。BM25 将文档 B 排在前列，合并后提升至最终候选前列。 |
| **查询 2**：`数据库提示找不到表怎么处理` | **容易漏检 文档 B**。文档 B 原文主要记录“table or view does not exist”，中文几乎没有“找不到表”字样，词频分极低。 | **命中 文档 B 和 C**。向量模型理解“找不到表”与“table does not exist”的同义关系。 | **准确命中 文档 B**。向量路补足同义改写召回，避免漏检。 |

<ReadMoreLock />

![固定维度向量的压缩瓶颈与高频主题压制](/images/content-rebuild/bbf24d3bd241153a/fig-2.png)

*上图呈现了变长文本向固定维度向量映射时的信息压缩过程：全局语义通常被保留，但长文本中局部低频特征在均值汇聚后容易被整句上下文稀释。*

---

## 3. 两种检索机制的原理对比

### 3.1 稠密向量检索：解决词汇不匹配

稠密向量检索通常借助双编码器（Bi-Encoder）模型，分别将查询 $q$ 和文档块 $d$ 映射到连续向量空间中：

$$
s(q, d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\|\mathbf{v}_q\| \|\mathbf{v}_d\|}
$$

- **优势**：经过大量文本对训练的嵌入模型能够学习词汇与语境的隐含关联，识别“调优”与“优化”、“延迟高”与“响应慢”等同义表达，甚至跨语言对齐。
- **局限**：模型通常采用子词分词器（如 WordPiece 或 Byte-level BPE）。罕见的错误码 `ORA-00942` 会被切散为多个子片段（如 `['OR', '##A', '-', '00', '##94', '##2']`）。在 Transformer 层的自注意力汇聚和池化操作后，这些细粒度子词特征容易被整句的常见词语义稀释，导致对精确符号不够敏感。

### 3.2 BM25 检索：基于概率统计的精准制导

BM25 是一种改进的词法倒排索引打分算法。给定查询 $Q$ 包含的词元 $q_1, q_2, \dots, q_n$，文档 $D$ 的打分公式为：

$$
\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}
$$

其中逆文档频率 $\text{IDF}(q_i)$ 计算为：

$$
\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)
$$

公式包含三个关键调控因子：
1. **逆文档频率（IDF）**：$N$ 为总文档数，$n(q_i)$ 为包含该词的文档数。若词元极其罕见（如错误码、型号），$n(q_i)$ 极小，IDF 值急剧上升，对总分起决定性作用。
2. **词频饱和度（$k_1$）**：$f(q_i, D)$ 为词频。当词频持续增加时，公式分子分母同步增长，使边际得分增长迅速放缓，避免单篇文档堆砌关键词作弊。通常取值范围为 $1.2 \sim 2.0$。
3. **文档长度惩罚（$b$）**：$|D| / \text{avgdl}$ 衡量当前文档长度与语料平均长度的比例。参数 $b$（通常取 0.75）调节惩罚力度，防止超长文档因覆盖词多而占优。

![BM25 词频饱和度与 IDF 惩罚因果曲线](/images/content-rebuild/bbf24d3bd241153a/fig-3.png)

*上图展示了 BM25 的两大核心性质：罕见词的超高 IDF 权重带来强判别力，而词频饱和机制防止局部刷词作弊。*

![单一向量检索中错误码发生语义漂移的失效全过程](/images/content-rebuild/bbf24d3bd241153a/fig-4.png)

*上图展示了分词切分、向量降维汇聚与近邻搜索各阶段，专有错误码如何逐步被高频上下文同化并导致失效。*

---

## 4. 候选集融合：为什么不能直接把两路分数相加？

在双路召回完成后，必须将两组候选列表合并。初学者常犯的错误是直接按权重线性相加：

$$
\text{Score}_{\text{final}} = \alpha \cdot S_{\text{dense}} + (1 - \alpha) \cdot S_{\text{bm25}} \quad \text{（不推荐直接使用）}
$$

这种做法在工程上通常不可靠，原因在于：
1. **数值范围不一致**：余弦相似度取值在 $[-1, 1]$ 之间（归一化后通常在 $[0, 1]$）；而 BM25 得分是非负实数，理论上无上界，短文档可能得 5 分，长匹配文档可能达数十甚至上百。
2. **查询间方差极大**：BM25 得分随着查询词数量累加。3 个词的查询与 10 个词的查询得分尺度差异悬殊，固定的 $\alpha$ 权重无法跨查询通用。

### 推荐方案：倒数排名融合（RRF）

倒数排名融合（Reciprocal Rank Fusion, RRF）直接利用文档在各个检索分支中的**相对排名**计算融合分，不再依赖原始分数的具体数值：

$$
\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}
$$

- $M$ 为检索通路集合（如 $\{\text{dense}, \text{bm25}\}$）；
- $r_m(d)$ 为文档 $d$ 在分支 $m$ 中的排名位置（从 1 开始）；
- $k$ 为常数平滑项（通常取 60），防止排名前列的文档权重过高，抑制极端值干扰。

若文档在 BM25 排第 1，在向量路排第 10，其 RRF 分数为：

$$
\text{Score} = \frac{1}{60 + 1} + \frac{1}{60 + 10} = \frac{1}{61} + \frac{1}{70} \approx 0.01639 + 0.01428 = 0.03067
$$

这样，在任意一路排名靠前、或在两路均排名前列的文档都能获得较高总分。

---

## 5. 工程实现：异步并发拉取与解耦

混合召回如果使用串行调用，总时延为 $T_{\text{dense}} + T_{\text{bm25}}$，会直接增加首字生成耗时。线上通常采用**异步并行拉取 + 超时保护**。

![双路召回并行调度与超时熔断降级架构](/images/content-rebuild/bbf24d3bd241153a/fig-5.png)

*上图展示了检索网关分发、并发执行、超时控制与候选集合并的时序全景。*

### 5.1 异步并发检索代码示例

以下 Python 示例展示了使用 `asyncio` 并行执行双路召回并做超时保护的逻辑：

```python
import asyncio
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class HybridRetrievalGateway:
    def __init__(self, dense_client, bm25_client, timeout_ms: float = 350.0):
        self.dense_client = dense_client
        self.bm25_client = bm25_client
        self.timeout = timeout_ms / 1000.0

    async def _fetch_dense(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        try:
            return await self.dense_client.search_async(query=query, limit=top_k)
        except Exception as err:
            logger.error(f"Dense retrieval error: {err}")
            return []

    async def _fetch_bm25(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        try:
            return await self.bm25_client.search_async(query=query, limit=top_k)
        except Exception as err:
            logger.error(f"BM25 retrieval error: {err}")
            return []

    async def retrieve(
        self, query: str, dense_k: int = 50, bm25_k: int = 50
    ) -> Dict[str, List[Dict[str, Any]]]:
        # 同时启动两个异步任务
        dense_task = asyncio.create_task(self._fetch_dense(query, dense_k))
        bm25_task = asyncio.create_task(self._fetch_bm25(query, bm25_k))

        done, pending = await asyncio.wait(
            [dense_task, bm25_task],
            timeout=self.timeout,
            return_when=asyncio.ALL_COMPLETED
        )

        # 超时未完成的子任务主动取消，避免资源悬挂
        for task in pending:
            task.cancel()
            logger.warning("Sub-retrieval task timed out and was cancelled.")

        dense_res = dense_task.result() if dense_task in done and not dense_task.exception() else []
        bm25_res = bm25_task.result() if bm25_task in done and not bm25_task.exception() else []

        return {
            "dense_candidates": dense_res,
            "bm25_candidates": bm25_res
        }
```

### 5.2 离线索引构建的解耦设计

在数据入库侧，BM25 倒排索引构建与向量计算的资源开销差异显著：

- **BM25 构建**：主要消耗 CPU 与磁盘 I/O，通过分词直接向倒排表（Posting List）写入记录，处理速度通常较快。
- **向量索引构建**：需要调用嵌入模型推理（通常消耗 GPU 或专用推理芯片），生成向量后再插入向量索引（如 HNSW 图），计算耗时较长。

因此，在离线入库流水线中，通常借助消息队列将写入任务解耦为两个独立消费组，分别维护全文检索与向量存储，避免向量推理堵塞倒排索引更新。详情可参见 [向量库增量更新方案](/interview/rag-interview/vector-db-incremental.md)。

![离线数据接入层双路解耦消费队列架构](/images/content-rebuild/bbf24d3bd241153a/fig-6.png)

*上图展示了基于消息队列解耦的离线流水线：CPU 快速构建倒排索引，GPU 批量构建向量索引。*

---

## 6. 面试深度追问与常见误区

### 追问 1：现代向量数据库原生支持混合检索吗？

**回答**：
支持。很多现代向量引擎（如 Qdrant、Milvus 等）已支持在引擎内完成多路召回：
- 例如 Qdrant（v1.10+）提供了 `prefetch` 查询参数，可以在单次请求中同时提交基于文本 Payload 的全文检索和基于稠密向量的近邻搜索；
- 数据库在服务端并发拉取两路结果，并使用内置的 RRF 或分数分布归一化算法（如 DBSF）合并后返回 Top-K；
- 相比于应用网关自行拉取，引擎内融合能减少网关与存储之间的多次网络往返传输。

### 追问 2：召回阶段候选集（Top-K）是不是取得越大越好？

**回答**：
不是。扩大候选集（例如从 50 放大到 500）虽然可能提升召回上限，但存在两项副作用：
1. **增加精排计算延迟**：下游若接有交叉编码器（Cross-Encoder）等重排模型，计算开销与候选文档数呈线性增长，候选集过大会拉长端到端时延；
2. **引入上下文噪声**：过多的低相关度文档进入最终阶段，会稀释有效信息密度，增加大模型产生事实错误的几率。具体的调试策略可参考 [RAG 检索质量排查指南](/interview/rag-interview/rag-troubleshooting.md)。

### 常见认知误区

- **误区一：对 Embedding 模型进行领域微调后，就可以完全放弃 BM25**。
  **纠偏**：微调能改善领域词汇的语义分布，但很难覆盖所有未见符号（如临时工单流水号、动态告警代码）。BM25 基于精确字符匹配，依然是未登录词（OOV）的稳定保底手段。
- **误区二：混合检索必然导致系统时延翻倍**。
  **纠偏**：在网关层采用异步并行调度后，两路查询并发运行，检索耗时主要取决于耗时较长的那一路（通常是向量推理与检索），而不是两者相加。
- **误区三：BM25 与向量相似度可以随意指定固定权重相加**。
  **纠偏**：两者的打分尺度与分布差异极大，BM25 是无界分数且与查询长度相关。未经分位数或 Min-Max 归一化直接线性相加，打分极易被 BM25 主导，通常建议首选无参数的 RRF 融合。
