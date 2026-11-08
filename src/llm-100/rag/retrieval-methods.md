---
title: 相似度检索、关键词检索与混合检索有什么区别？从两路召回到RRF融合
icon: mdi:database
tag:
- RAG
- 大模型100讲
- 检索增强
description: 对比关键词检索与向量相似度检索的适用场景，解析分数线性加权的失效原因，掌握倒数排名融合（RRF）与加权 RRF 的工程实现。
---

在检索增强生成（RAG）系统中，我们经常面临两难选择：是使用基于词频统计的关键词检索（如 BM25），还是使用基于向量相似度的稠密检索（Dense Retrieval）？

答案通常不是“二选一”，而是将两者结合。我们在 [BM25 为什么还能在 RAG 中发挥作用？](/llm-100/rag/bm25-in-rag.md) 中分析了词法倒排索引在精确匹配上的作用，在 [Embedding 模型如何选择？](/llm-100/rag/embedding-selection.md) 中也探讨了稠密向量对语义泛化的捕捉能力。本讲将这两条路径合流，解析如何利用倒数排名融合（Reciprocal Rank Fusion, RRF）搭建稳定的混合检索流程，为后续的 [Rerank 重排序为什么重要？](/llm-100/rag/rerank-importance.md) 准备高质量候选集。

![从稠密检索到混合检索的架构概览](/images/content-rebuild/d8e4467f5543373f/cover.png)

上图展示了混合检索的核心流程：用户查询被同时送入关键词检索与向量检索两个独立分支，各自筛选候选结果后，通过融合层合并为统一列表。理解混合检索的前提，是认清单一检索方式在实际场景中的表现差异。

## 一个贯穿例子：两类检索的优势与短板

以运维排障知识库为例，考察以下两个真实查询：

- **查询 A（口语化泛化描述）**：“显卡不认设备怎么办”
- **查询 B（精准专有名词）**：“A100-SXM4-80GB 故障排查”

针对**查询 A**：
知识库文档通常写作“GPU 故障排查：当遇到显卡掉卡或 CUDA 驱动未加载时，查看 dmesg 日志”。
- **关键词检索（BM25）**：依赖词项命中。如果文档中没有出现“不认”或“设备”，词频匹配分数极低甚至无法召回。
- **相似度检索（Dense）**：Embedding 模型能够将“显卡不认设备”映射到与“掉卡”、“驱动未加载”相近的向量空间区域，准确召回相关文档。

针对**查询 B**：
“A100-SXM4-80GB”是一个具体的硬件型号，与“A100-PCIE-40GB”或“H100-SXM5-80GB”仅有几个字符之差，但属于完全不同的硬件规格。
- **相似度检索（Dense）**：Embedding 模型在压缩长文本语义时，往往无法对专有型号保持严格的字符敏感性，可能将宽泛的“深度学习训练集群供电与机柜散热要求”排在首位，导致排序偏差。
- **关键词检索（BM25）**：利用逆文档频率（IDF），“A100-SXM4-80GB”这类稀有词项会获得极高的打分权重，能够迅速命中对应型号的文档。

![Dense 语义泛化与 Sparse 精确匹配的召回互补对比](/images/content-rebuild/d8e4467f5543373f/fig-1.png)

如图所示，相似度检索负责语义泛化，关键词检索负责精确匹配。将二者结合的混合检索（Hybrid Search），目标就是通过多路召回让两条路径并行工作，降低第一阶段的漏召率。

## 为什么不能直接将两路分数相加？

获取两路候选后，最容易想到的合并方式是加权求和：

$$Score(d) = \alpha \cdot Score_{dense}(d) + (1 - \alpha) \cdot Score_{bm25}(d)$$

但在实际工程中，直接相加通常会导致排序失真，原因在于两者的数值尺度存在根本差异。

<ReadMoreLock />

余弦相似度的理论取值区间是 $[-1, 1]$，在文本检索中候选集的得分通常落在 $[0.4, 0.95]$ 的窄带内；而 BM25 得分是一个无固定上界的非负实数 $[0, +\infty)$，其大小取决于查询长度、文档命中词项的 IDF 以及文档长度。一个包含多个生僻专有名词的长查询，BM25 最高分可能达到 40 分以上；而一个常见短词查询，最高分可能只有 3 分左右。

![线性归一化在极值漂移下的失效机制](/images/content-rebuild/d8e4467f5543373f/fig-2.png)

针对量纲冲突，常见的折中方案是对两路分数做 Min-Max 归一化：

$$\hat{S} = \frac{S - S_{min}}{S_{max} - S_{min}}$$

但如图所示，Min-Max 归一化极易受到长尾极值的影响。若某一候选文档偶然命中了某些罕见高频词，导致其 BM25 得分异常偏高，$S_{max}$ 会被单点拉伸，使得其他正常相关文档的归一化得分被压缩到接近 0。此外，由于不同 Query 触发的分数分布动态漂移，固定参数 $\alpha$ 无法在所有输入类型上保持均衡。

## 倒数排名融合（RRF）：基于位置的无标度合并

为了摆脱不可靠的绝对分数值，倒数排名融合（Reciprocal Rank Fusion, RRF）放弃了原始得分，仅使用候选文档在各分支中的排序序号（Rank）：

$$RRF\_Score(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

其中 $M$ 是检索分支集合，$r_m(d)$ 是文档 $d$ 在检索器 $m$ 返回列表中的排位名次（从 1 开始计），$k$ 是平滑常数（通常取经验值 $60$）。

![RRF 倒数排名与平滑常数 k 的打分曲线](/images/content-rebuild/d8e4467f5543373f/fig-3.png)

倒数函数赋予靠前的排名更高的分值。引入常数 $k=60$ 的主要作用是平滑衰减梯度：
- 如果不加平滑常数（$k=0$），第 1 名得分是 $1.0$，第 2 名得分是 $0.5$，前两名分差高达 $0.5$；
- 加入 $k=60$ 后，第 1 名得分是 $1/61 \approx 0.01639$，第 2 名得分是 $1/62 \approx 0.01613$，分差平滑缩小为 $0.00026$。

这保证了单个分支内头部排名的分差不会过于悬殊，同时对“在多路检索中均出现在前列”的文档给予明显的累加优势。以下表为例，考察两篇文档在 $k=60$ 下的融合过程：

| 文档编号 | Dense 分支排名 | BM25 分支排名 | Dense 得分分量 | BM25 得分分量 | RRF 最终得分 | 最终排名 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Doc-A | 第 2 名 | 第 5 名 | $1 / (60 + 2) \approx 0.01613$ | $1 / (60 + 5) \approx 0.01538$ | **0.03151** | **第 1 名** |
| Doc-B | 未进入前 100 | 第 1 名 | 0.0 | $1 / (60 + 1) \approx 0.01639$ | 0.01639 | 第 2 名 |

Doc-B 仅在 BM25 取得单项第 1 名；而 Doc-A 虽然在两路中都不是第 1，但都在靠前位置（第 2 与第 5），两路得分叠加后总分反超 Doc-B。RRF 仅依赖排位，不需要校准跨检索器的绝对分数。

## 加权 RRF（Weighted RRF）：调整分支偏置

标准 RRF 假设各路检索器的置信度完全对等。但在实际业务中，不同场景的可靠性不同。例如在代码库检索中，函数名的词法匹配更重要；而在开放客服对话中，用户口语化表达占比更高，稠密向量分支往往更可靠。

在 Qdrant 等现代检索系统中，加权 RRF 允许为不同预取分支配置权重系数：

$$Weighted\_RRF\_Score(d) = \sum_{m \in M} w_m \cdot \frac{1}{k + r_m(d)}$$

![加权 RRF 在多检索分支上的权重分配与排位映射](/images/content-rebuild/d8e4467f5543373f/fig-4.png)

设置稠密分支权重 $w_{dense} = 3.0$、词法分支权重 $w_{sparse} = 1.0$ 时，若采用上述分子得分加权，稠密分支每项得分贡献被放大 3 倍。在带入 $k=60$ 时，高权重分支第 3 名分量为 $3.0 / (60 + 3) \approx 0.0476$，显著高于低权重分支第 1 名的 $1.0 / (60 + 1) \approx 0.0164$，容易使高权重路整体压制低权重路。

相比之下，Qdrant 等系统内置的加权 RRF 采用位次尺度加权：通过缩放排位步长，使得权重 3.0 分支的第 3 名与权重 1.0 分支的第 1 名得分对齐，且两路结果不重叠时按 3:1 比例交织候选。工程落地时需区分应用层“得分加权”与存储引擎“位次加权”的行为差异。

## 预取与归约架构（Prefetch & Reduce）

在底层存储引擎（如 Qdrant 或 Elasticsearch）内部，混合检索通常采用“并发预取 + 内存归约”两阶段时序：

![多阶段 Prefetch 架构与二次合并归约执行时序](/images/content-rebuild/d8e4467f5543373f/fig-5.png)

流程如下：
1. **并发预取（Parallel Prefetch）**：协调节点接收查询请求后，在内存中将 Query 生成分词项送给倒排索引，同时调用向量接口生成 Embedding 送入 HNSW 图索引。两组查询并发检索，各取出 Top-$N$（例如 $N=100$）个候选。
2. **归约融合（Merge & Reduce）**：协调节点收集两路共计 $2N$ 个候选项，执行去重，根据文档在各自列表中的位置套用加权 RRF 公式计算得分，截断并输出 Top-$K$（例如 $K=20$）个结果。
3. **下游衔接**：精简后的候选集可送入 Cross-Encoder 模型进行精准重排，或直接拼入 Prompt 供模型参考。

## 完整可运行代码：混合检索机制模拟验证

下面是一个包含简易 BM25、模拟向量检索与加权 RRF 的 Python 脚本，可直接运行验证：

```python
import math
from collections import Counter
from typing import List, Dict, Tuple

# 1. 模拟运维知识库文档
DOCUMENTS = [
    "NVIDIA A100-SXM4-80GB 采用 Ampere 架构，专为深度学习加速设计。",
    "服务器无法开机排查指南：检查电源接线、主板自检指示灯以及内存条接触状态。",
    "高并发系统中的缓存雪崩解决方案：设置随机过期时间与引入双写一致性保障。",
    "GPU 故障排查：当遇到显卡掉卡或 CUDA 驱动未加载时，查看 dmesg 日志。",
    "深度学习训练集群的供电与机柜散热要求，探讨高算力硬件能耗指标。"
]

def tokenize(text: str) -> List[str]:
    # 简易分词：将标点替换为空格后按空格拆分（生产环境建议使用专业分词工具）
    clean_text = text.replace("：", " ").replace("，", " ").replace("、", " ").replace(" ", " ")
    return [w.lower() for w in clean_text.split() if w]

# 2. 简易 BM25 检索器
class SimpleBM25:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.doc_tokens = [tokenize(doc) for doc in corpus]
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in self.doc_tokens) / self.corpus_size
        self.k1 = 1.5
        self.b = 0.75
        
        # 统计文档频率
        self.doc_freqs: Dict[str, int] = Counter()
        for doc in self.doc_tokens:
            for word in set(doc):
                self.doc_freqs[word] += 1
        # 计算标准 BM25 的 IDF 分数
        self.idf = {
            word: math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            for word, freq in self.doc_freqs.items()
        }

    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        query_tokens = tokenize(query)
        scores = []
        for idx, doc in enumerate(self.doc_tokens):
            score = 0.0
            doc_len = len(doc)
            doc_counter = Counter(doc)
            for token in query_tokens:
                if token in doc_counter:
                    freq = doc_counter[token]
                    idf = self.idf.get(token, 0.0)
                    # 词频归一化公式
                    tf_norm = (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl)))
                    score += idf * tf_norm
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# 3. 模拟向量检索器（模拟语义泛化对型号的轻微漂移）
class MockDenseRetriever:
    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        if "A100-SXM4-80GB" in query:
            # 稠密模型将宏观能耗文档 Doc 4 误判为与算力集群语义高度相近
            sim_scores = [(4, 0.81), (0, 0.80), (3, 0.71), (2, 0.30), (1, 0.15)]
        else:
            sim_scores = [(idx, 0.5) for idx in range(len(DOCUMENTS))]
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        return sim_scores[:top_k]

# 4. 加权 RRF 融合实现
def rrf_fusion(
    rank_lists: List[Tuple[List[Tuple[int, float]], float]], 
    k: int = 60, 
    top_k: int = 3
) -> List[Tuple[int, float]]:
    doc_rrf_scores: Dict[int, float] = {}
    for results, weight in rank_lists:
        for rank, (doc_id, _) in enumerate(results, start=1):
            contribution = weight * (1.0 / (k + rank))
            doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + contribution
    
    sorted_docs = sorted(doc_rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_k]

# 5. 执行对比
bm25_engine = SimpleBM25(DOCUMENTS)
dense_engine = MockDenseRetriever()

query = "A100-SXM4-80GB 故障排查"
print(f"测试查询: '{query}'")

bm25_hits = bm25_engine.search(query, top_k=3)
dense_hits = dense_engine.search(query, top_k=3)
hybrid_hits = rrf_fusion([(dense_hits, 1.0), (bm25_hits, 1.0)], k=60, top_k=3)

print("\n[BM25 检索结果]")
for rank, (idx, score) in enumerate(bm25_hits, 1):
    print(f"第 {rank} 名: Doc {idx} (Score: {score:.4f}) -> {DOCUMENTS[idx]}")

print("\n[Dense 检索结果]")
for rank, (idx, score) in enumerate(dense_hits, 1):
    print(f"第 {rank} 名: Doc {idx} (Score: {score:.4f}) -> {DOCUMENTS[idx]}")

print("\n[Hybrid RRF 检索结果]")
for rank, (idx, score) in enumerate(hybrid_hits, 1):
    print(f"第 {rank} 名: Doc {idx} (RRF Score: {score:.6f}) -> {DOCUMENTS[idx]}")
```

![消融实验中三路检索方案在两类典型 Query 下的召回准确率对比](/images/content-rebuild/d8e4467f5543373f/fig-6.png)

运行上述代码，各检索路径的输出表现如下：
- 在输入包含特定型号“A100-SXM4-80GB”的精确词法查询时，BM25 凭借倒排索引准确定位到 Doc 0 并赋予最高分，而 Dense 单路受语义泛化影响，将概括性文章 Doc 4 排在首位。
- 经过混合 RRF 融合后，Doc 0 同时获得 BM25 的第 1 名与 Dense 的第 2 名加权支撑，最终以最高总分稳定在融合列表首位。
- 混合检索阻断了单一检索器的误判，提升了候选集进入重排阶段时的召回质量。

关于落地中如何与向量库配合，可参阅 [向量数据库在 RAG 中起什么作用？](/llm-100/rag/vector-db-in-rag.md)；若需在候选集生成后进一步做深度交叉打分，请继续学习下一讲 [Rerank 重排序为什么重要？](/llm-100/rag/rerank-importance.md)。面试相关讨论可参考 [为什么 RAG 中常用 BM25 + 向量检索的混合召回？](/interview/rag-interview/hybrid-retrieval.md)。

## 检验理解的练习

1. **参数敏感性验证**：在上述代码中，将平滑常数 $k$ 分别修改为 $k=1$ 与 $k=100$，观察 Doc 0、Doc 3 与 Doc 4 的 RRF 得分差距如何变化。平滑常数 $k$ 过大或过小时，融合结果分别会更偏向头部单项冠军，还是更偏向多路均衡者？
2. **权重调节实验**：在代码中将权重配置调整为 `dense_weight=5.0` 与 `bm25_weight=1.0`，观察加权后排序的变化，思考在专有词汇密集型文档库中盲目提高向量权重会带来什么风险。
