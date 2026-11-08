---
title: 向量数据库在 RAG 中起什么作用？
icon: mdi:database
tag:
- RAG
- 大模型100讲
- 检索增强
description: 梳理向量数据库在 RAG 中的三项具体职责：高维向量近似近邻检索（ANN）、带条件的单阶段元数据过滤，以及向量索引与长文档存储的解耦架构。
---

在检索增强生成（RAG）系统中，文本切片首先通过 Embedding 模型转化为高维稠密向量。如果系统只有几百条知识切片，直接用数组保存并在内存中逐一计算点积即可满足需求。然而，当知识库扩展到数十万甚至上千万切片，且检索请求附带“仅查询 2024 年财务部公开文档”这类业务条件时，简单的内存线性比对会迅速耗尽计算资源与响应时间。

向量数据库在 RAG 系统中主要承担三项具体分工：
1. **近似最近邻检索（ANN）**：在有限延迟与算力预算下，从海量向量中快速定位语义相近的候选切片；
2. **单阶段元数据过滤**：将属性过滤（如部门、时间、权限）融入检索过程，避免“先检索后过滤”导致的有效候选归零；
3. **存储解耦与上下文组装**：将耗费内存的高维向量索引与占用磁盘的长文本内容分离存储，兼顾索引跳转吞吐与上下文还原。

![向量数据库在 RAG 系统中的双重职责架构概览](/images/content-rebuild/8b70dc5c81c54773/cover.png)

---

## 贯穿示例与精确检索的算力瓶颈

我们以一个具体的企业知识库场景贯穿全文：
- 知识库包含 $10,000,000$（一千万）条文档切片；
- 文本编码使用常见的 1024 维稠密向量（$d = 1024$，单精度浮点数 `float32`，每个分量占 4 字节）；
- 用户发起查询：“2024年第一季度财务报表中研发费用是多少？”，限制条件为 `dept == 'finance'` 且 `year == 2024`。

### 线性全表扫描（精确 KNN）的代价

如果采用精确最近邻检索（Exact K-Nearest Neighbors, KNN），系统必须遍历所有 $N$ 个切片，逐一计算查询向量与切片向量的点积或余弦距离。算法复杂度为 $O(N \cdot d)$。

我们来计算单次查询的物理硬件消耗：
1. **内存传输开销**：每个向量占用内存为 $1024 \times 4 \text{ 字节} = 4096 \text{ 字节} = 4 \text{ KB}$。一千万个向量的原始体积为：
   $$10^7 \times 4096 \text{ 字节} \approx 40.96 \times 10^9 \text{ 字节} \approx 40.96 \text{ GB}$$
   在没有索引的情况下，单次检索需要将近 $41 \text{ GB}$ 的浮点数据从内存读取到 CPU/GPU 缓存中。
2. **浮点运算量**：每次 1024 维向量点积包含 1024 次乘法与 1023 次加法，约为 $2047$ 次浮点运算。扫描一千万个向量需要约 $2.05 \times 10^{10}$ 次浮点运算（约 20 GFLOPs）。
3. **并发瓶颈**：如果线上服务要求达到 100 QPS（每秒 100 次查询），仅点积运算就要求系统提供 $20 \text{ GFLOPs} \times 100 = 2 \text{ TFLOPs}$ 的持续浮点吞吐，且内存带宽需达到惊人的 $4.1 \text{ TB/s}$。普通服务器内存总线会立即被打满。

![精确 KNN 线性扫描与 ANN 空间跳跃检索的计算路径对比](/images/content-rebuild/8b70dc5c81c54773/fig-1.png)

因此，向量数据库的第一个核心作用，是通过预先构建的**近似最近邻（Approximate Nearest Neighbor, ANN）**索引，以牺牲少许召回精度（如召回真实 Top-10 结果中的 95%~98%）为代价，将查询复杂度降低到亚线性或对数级别，使单次查询耗时稳定在毫秒级。

关于稠密向量与稀疏向量的编码差异，可参考 [稀疏向量、稠密向量和多向量检索有什么区别？](/fundamentals/embedding/sparse-dense-vectors.md)；关于向量数据库与关系型数据库的差异对比，可参考 [RAG 为什么需要向量数据库？它和传统数据库有什么区别？](/interview/rag-interview/vector-db-vs-traditional.md)。

---

## 两种主流 ANN 索引的工作方式：HNSW 与 IVF

向量数据库之所以能跳过绝大多数不相关的向量，依赖于预先建立的几何拓扑或空间划分。目前最广泛使用的两种索引结构是 **HNSW** 和 **IVF**。

### 1. HNSW（分层导航小世界图）

HNSW（Hierarchical Navigable Small World）的思路来源于跳表（Skip-List）：
- **分层图结构**：整个图被划分为多层。顶层节点最稀疏，节点间的连边跨度极大；越往下层节点越密集；底层（Layer 0）包含全部 $N$ 个向量，每个节点保留其最近的 $M$ 个邻居连边。
- **贪心图遍历（Greedy Search）**：检索从顶层的入口点开始。算法计算当前节点的所有邻居与查询向量 $\mathbf{q}$ 的距离，若存在比当前节点更近的邻居，则跳跃至该邻居；当局部无法找到更近节点时，以此节点为入口下沉至下一层继续寻找。穿透到底层后，算法在较小的局部邻域内微调，输出最终的 Top-K。

### 2. IVF（倒排文件索引）

IVF（Inverted File）的核心是空间聚类划分：
- **建表阶段**：在离线状态下，利用 K-Means 算法在全量向量中训练出 $nlist$ 个聚类中心（Centroids）。每个向量被分配给距离最近的中心，形成一个个倒排列表（Inverted Lists）。
- **检索阶段**：查询向量 $\mathbf{q}$ 先与这 $nlist$ 个中心计算距离，挑出距离最近的 $nprobe$ 个中心桶。系统随后仅扫描这几个桶内的成员向量，剩余的大量倒排桶直接跳过。

![HNSW 多层跳表式图遍历与 IVF 空间 Voronoi 聚类分桶检索机制对比](/images/content-rebuild/8b70dc5c81c54773/fig-2.png)

### HNSW 与 IVF 的工程选型权衡

| 维度 | HNSW（基于图） | IVF（基于聚类倒排） |
| :--- | :--- | :--- |
| **内存占用** | 较高。需额外存储邻居指针，通常为原始向量大小的 1.2 ~ 2 倍 | 较低。仅需存储聚类中心坐标与倒排链表 ID，额外开销小 |
| **查询延迟与召回** | 在高召回率（如 98%）下延迟表现优异，通常在 5~20 毫秒 | 延迟受 $nprobe$ 影响较大；若要极高召回率需多探查桶，延迟上升明显 |
| **构建与写入** | 支持逐条动态追加插入，但图修补开销较高 | 聚类中心通常需批量训练，增量数据偏离中心时需定期重建 |
| **典型适用场景** | 低延迟交互式问答、知识切片量在千万级以下的实时系统 | 亿级大规模向量库、内存预算紧张、可接受离线批处理构建的场景 |

<ReadMoreLock />

---

## 元数据过滤：为什么不能简单“后过滤”？

在真实的 RAG 查询中，单纯计算语义距离远远不够。用户往往会携带严格的业务过滤规则，例如：`dept == 'finance'`。

在向量检索中处理这类属性条件，执行策略主要分为后过滤、朴素预过滤和单阶段联合遍历三种。

### 1. 后过滤（Post-filtering）与“候选枯竭”

后过滤的流程是：向量引擎先忽略元数据条件，按照普通 ANN 流程检索出全局最相似的 Top-K 个切片，然后由外部程序过滤掉不符合条件的记录。

当过滤条件的选择性很高（即符合条件的数据占比极低）时，这种方式会导致返回结果为空，即**候选枯竭（Candidate Starvation）**。

> **概率示意推演**：
> 假设切片库共有 $1,000,000$ 条数据，其中标记为 `finance` 的仅占 $0.5\%$（共 5,000 条）。用户请求 Top-10 知识切片。
> 若假定语义最相近的切片在全库中的分布与部门标签相互独立，单个召回切片**不属于**财务部门的概率为 $1 - 0.005 = 0.995$。
> 在初次召回的 10 个切片中，**全部不符合条件**的概率为：
> $$P(\text{全部淘汰}) = (0.995)^{10} \approx 0.951 \quad (95.1\%)$$ 
> 这意味着，有超过 95% 的概率，后过滤操作返回的结果数量为 0。即使把初始召回窗口放大到 Top-50，全部淘汰的概率仍有 $(0.995)^{50} \approx 77.8\%$。

### 2. 朴素预过滤（Naive Pre-filtering）与图连通性断裂

为了避免过滤后无结果，一种直觉做法是朴素预过滤：先在关系表或倒排索引中查出所有 `dept == 'finance'` 的向量 ID 集合，然后强行在 HNSW 图中“只走这 5,000 个节点”。

然而，HNSW 依赖密集的邻居关系维持小世界网络的快速跳转能力。如果在图中把其余 99.5% 的节点强行屏蔽，原本连通的图会断裂成大量互不相连的孤岛。图遍历算法在跳跃几步后就会无路可走，困在局部极大值，导致召回质量大幅下降。

![元数据后过滤候选枯竭与硬预过滤导致图连通性断裂的机制分析](/images/content-rebuild/8b70dc5c81c54773/fig-3.png)

### 3. 生产级解法：单阶段联合遍历（Filtered HNSW）

成熟的向量数据库（如 Qdrant、Milvus 等）采用**单阶段联合遍历**：
- **中继跳板机制**：在 HNSW 图的贪心跳转过程中，**不符合元数据条件的节点依然可以作为中继路由节点**被访问，保证图的整体连通性不受破坏；
- **准入判定**：只有**既通过元数据过滤、又具备高相似度**的节点，才被允许压入最终的 Top-K 结果候选队列；
- **自适应降级**：当引擎检测到满足过滤条件的向量数量极少（例如全库仅剩十几个节点满足条件）时，系统会自动放弃图遍历，转为直接利用标量索引定位这些向量，并进行小规模精确点积计算。这种动态切换避免了在稀疏图上的无效迭代。

---

## 存储解耦架构：向量索引与文档分离

在 RAG 流水线中，每个切片不仅包含稠密向量，还伴随着几百到上千字符的原始 Markdown/HTML 文本，以及复杂的上下文元数据（例如所在文件名、前序与后序切片 ID、作者等）。

一个容易被忽略的工程隐患是：**将大量富文本直接塞进向量数据库的内存 Payload 中**。

### 为什么要把文本从向量索引中解耦？

1. **CPU 缓存行污染（Cache Pollution）**：HNSW 检索是极其密集的内存指针跳转与浮点运算。现代 CPU 依靠 L1/L2/L3 高速缓存实现每秒数千万次的比较。如果向量节点中塞入了数千字节的长文本，每次内存预取都会载入大量与相似度计算无关的文本字符，导致 CPU 高级缓存命中率骤降。
2. **内存成本膨胀**：纯向量每条仅占 4 KB 左右，而一个包含 1000 字符文本及其排版信息的切片可能占用 10~20 KB。将切片全部保存在高速内存中，硬件采购成本将成倍上升。

成熟的 RAG 系统（如 LlamaIndex 等框架推荐的实践）普遍采用分层存储解耦：

![向量存储引擎、文档存储库与键值缓存的三层解耦架构协同流向](/images/content-rebuild/8b70dc5c81c54773/fig-4.png)

- **向量存储层（Vector Store）**：仅持久化 `Doc_ID`、稠密浮点向量以及用于建立过滤的简要标量字段（如 `tenant_id`、`created_at`）。以内存或 mmap 方式运行，保障高吞吐图跳跃；
- **文档存储层（Document Store / DocStore）**：采用低成本磁盘介质（如 RocksDB、MongoDB、PostgreSQL 或对象存储），以 `Doc_ID` 为键，存放完整的富文本内容和父子切片上下文；
- **缓存层（KV Store / Cache）**：缓存高频提问的 Embedding 向量和检索结果，避免模型重复推断。

一次标准的 RAG 检索流程为：用户输入由 Embedding 模型转为向量 $\rightarrow$ 向量数据库执行带过滤的 ANN 检索，返回排序后的 `[Doc_ID, Score]` $\rightarrow$ 应用层根据 `Doc_ID` 到 DocStore 中批量读取完整文本（文本水化，Hydration） $\rightarrow$ 拼接进上下文送入大模型。切片大小与上下文拼接细节可参考 [文档切分 Chunk 应该怎么做？](/llm-100/rag/chunk-strategy.md)。

---

## 多租户隔离的三种实现方案

企业知识库通常需要服务不同业务线或不同客户。向量数据库在支持多租户隔离时，主要有以下三种工程形态：

![多租户架构三种隔离方案的物理资源与检索流向对比](/images/content-rebuild/8b70dc5c81c54773/fig-5.png)

| 方案类型 | 方案 A：物理独立 Collection | 方案 B：逻辑分区（Partition） | 方案 C：单表 Payload 动态过滤 |
| :--- | :--- | :--- | :--- |
| **实现方式** | 每个租户创建独立的集合，拥有独立的 HNSW 图与索引文件 | 在单集合内使用引擎原生 Partition 键，底层划分独立存储段 | 所有租户写入同一集合，每条记录携带 `tenant_id` 属性 |
| **租户承载上限** | 较低（单集群通常建议在数千个以内，过多会导致文件句柄与协程耗尽） | 中等（通常支持数百至数千个逻辑分区） | 极高（百万级租户可直接共享单表） |
| **资源开销** | 极高（每个 Collection 独立分配图内存与索引缓存） | 中等（共享连接，但分区仍占维护开销） | 极低（全局共用同一套物理索引与内存池） |
| **数据清理成本** | 极低（直接执行 `Drop Collection` 即可秒级释放物理空间） | 较低（执行 `Drop Partition` 释放段空间） | 较高（需打软删除标记，依赖后台漫长的 Compaction 逐步回收空间） |
| **典型适用场景** | 头部大客户（VIP）、有强审计与物理合规要求的私有化部署 | 部门级隔离、按年/月归档的时序知识库 | 多租户 SaaS 平台、海量小型租户（单租户切片仅百条至千条） |

---

## 检验代码：后过滤失效与单阶段过滤对比

通过一段自包含的 Python 实验代码，可以直观验证“后过滤导致候选枯竭”与“单阶段过滤保证足额返回”的机制差异。

代码模拟包含 1000 个知识切片的向量库，设定仅有 5 个切片属于财务部门（`dept == 'finance'`），模拟高过滤比场景：

```python
import numpy as np

class ToyVectorDB:
    def __init__(self, dim=64):
        self.dim = dim
        self.vectors = []
        self.metadata = []

    def insert(self, doc_id: int, vector: np.ndarray, meta: dict):
        # 单位归一化：便于后续通过点积直接衡量余弦相似度
        norm_v = vector / np.linalg.norm(vector)
        self.vectors.append((doc_id, norm_v))
        self.metadata.append(meta)

    def post_filter_search(self, query_vec: np.ndarray, filter_key: str, filter_val: str, top_k: int = 5):
        """后过滤：先取全局语义最相似的 top_k，再过滤掉不符合条件的切片"""
        q = query_vec / np.linalg.norm(query_vec)
        scores = []
        for idx, (doc_id, v) in enumerate(self.vectors):
            similarity = float(np.dot(q, v))
            scores.append((doc_id, similarity, self.metadata[idx]))
        
        # 模拟无约束 ANN：直接取相似度最高的 top_k 项
        scores.sort(key=lambda x: x[1], reverse=True)
        initial_candidates = scores[:top_k]
        
        # 应用层过滤
        filtered_results = [
            cand for cand in initial_candidates 
            if cand[2].get(filter_key) == filter_val
        ]
        return filtered_results, initial_candidates

    def single_stage_search(self, query_vec: np.ndarray, filter_key: str, filter_val: str, top_k: int = 5):
        """单阶段联合过滤：检索准入时直接结合元数据与相似度"""
        q = query_vec / np.linalg.norm(query_vec)
        valid_candidates = []
        for idx, (doc_id, v) in enumerate(self.vectors):
            # 引擎级过滤：仅将符合条件的节点纳入比较
            if self.metadata[idx].get(filter_key) == filter_val:
                similarity = float(np.dot(q, v))
                valid_candidates.append((doc_id, similarity, self.metadata[idx]))
        
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        return valid_candidates[:top_k]

# --- 运行验证实验 ---
np.random.seed(42)
db = ToyVectorDB(dim=64)

# 注入 995 条通用切片，仅 5 条属于 finance
for i in range(1000):
    vec = np.random.randn(64)
    dept = "finance" if i < 5 else "tech"
    db.insert(doc_id=i, vector=vec, meta={"dept": dept, "text": f"Chunk-{i}"})

query_vector = np.random.randn(64)

post_results, raw_candidates = db.post_filter_search(query_vector, filter_key="dept", filter_val="finance", top_k=5)
engine_results = db.single_stage_search(query_vector, filter_key="dept", filter_val="finance", top_k=5)

print(f"[后过滤] 初次全局召回条数: {len(raw_candidates)}")
print(f"[后过滤] 业务过滤后留存条数: {len(post_results)} (发生候选枯竭！)")
print(f"[单阶段过滤] 引擎返回有效条数: {len(engine_results)}")
for item in engine_results:
    print(f"  -> 命中 DocID: {item[0]}, 相似度: {item[1]:.4f}, 部门: {item[2]['dept']}")
```

![代码验证输出结果：后过滤候选归零与单阶段过滤足额返回对比](/images/content-rebuild/8b70dc5c81c54773/fig-6.png)

### 实验输出与现象分析

运行上述脚本，控制台输出如下：

```text
[后过滤] 初次全局召回条数: 5
[后过滤] 业务过滤后留存条数: 0 (发生候选枯竭！)
[单阶段过滤] 引擎返回有效条数: 5
  -> 命中 DocID: 3, 相似度: 0.2814, 部门: finance
  -> 命中 DocID: 0, 相似度: 0.1632, 部门: finance
  -> 命中 DocID: 4, 相似度: 0.1418, 部门: finance
  -> 命中 DocID: 1, 相似度: -0.0152, 部门: finance
  -> 命中 DocID: 2, 相似度: -0.0416, 部门: finance
```

在后过滤策略下，由于全库中财务文档只占 0.5%，全局相似度最高的 Top-5 全部被技术文档占领，后续的元数据过滤直接将 5 个候选全部淘汰，导致交付给大模型的上下文完全为空；而单阶段过滤在准入阶段便约束了候选范围，成功将全部 5 条财务文档按相关度足额召回。

关于向量数据的增量写入机制，可查阅 [RAG 中如何实现向量数据库的增量更新？](/interview/rag-interview/vector-db-incremental.md)；相关选型清单可查阅 [向量数据库工具资源](/resources/vector-db.md)。

---

## 检验练习与小结

请尝试推演以下两个场景以检验掌握程度：

1. **如果一个向量数据库的过滤条件为 `author == 'Alice'`，而 Alice 在一千万切片库中只有 2 条切片，HNSW 的单阶段图遍历还会高效吗？成熟引擎会怎么做？**
   *思考方向：当匹配结果极其稀少时，图遍历很难在有限跳数内随机撞入目标节点。现代向量数据库会统计属性基数，在此类极端选择性下自动回退为基于标量索引的直接查找，然后针对这 2 条记录做单点比对。*
2. **为什么我们不建议把整个文档切片（如 2000 字符的 Markdown 正文）与 1024 维向量放在同一块连续内存中进行存储？**
   *思考方向：考虑 CPU 缓存行（Cache Line）利用率、内存带宽限制，以及 ANN 索引频繁跳转访问与文本读取阶段的时间差。*

在现代 RAG 系统中，向量数据库解决了连续语义空间内的近似定位与多维过滤问题。但在面对特定零件编号、人名、错误码等要求绝对字面匹配的场景时，稠密向量常常会出现匹配漂移。下一讲我们将探讨如何将传统全文检索与向量数据库结合——[BM25 为什么还能在 RAG 中发挥作用？](/llm-100/rag/bm25-in-rag.md)。
