---
title: GraphRAG 是什么？知识图谱如何解决传统向量检索的断裂与漏查
icon: mdi:graph-outline
tag:
- GraphRAG
- 知识图谱
- 向量检索
- RAG
description: 解释 GraphRAG 如何结合知识图谱与文本切块，解决传统向量检索在跨文档多跳推理与全景总结中的上下文断裂问题，介绍局部与全局检索的双轨机制及适用边界。
---

检索增强生成（RAG）的主流实现依赖于文本分块、稠密向量嵌入与余弦相似度检索。这种方案在事实定位明确的“单点查询”中效率很高，但在面对跨文档多跳推理与全景主题归纳时，往往暴露出上下文割裂与召回不全的问题。GraphRAG 通过引入显式知识图谱与分层社区摘要，重构了知识的索引与检索流程。

![GraphRAG 核心范式：从离散向量点阵到结构化知识图谱拓扑的演进](/images/content-rebuild/55ae7abfb750ea15/cover.png)

上图概括了知识表示范式的跃迁：左侧离散的向量空间依赖点对点几何距离，右侧的图结构则通过明确的实体节点与关系边建立知识拓扑。成年卡通讲解员站在中间，对比两种模式在连接关联时的机制差异。

---

## 一个贯穿全文的供应链审计案例

为了直观理解传统向量检索在什么情况下会漏查，我们设定一个贯穿本文的实际场景。某科技公司有一份长达数十页的内部系统审计报告，其中散落着以下几条事实：

* **切块 1（系统架构章节）**：“核心交易网关完全部署在华东一号数据中心。”
* **切块 2（机房与基建章节）**：“华东一号数据中心的双路外部供电全部接入江北变电站。”
* **切块 3（近期运维计划章节）**：“江北变电站将在下周六进行年度停电检修。”
* **切块 4~50（其他各章节）**：分散记录了存储集群网络丢包、鉴权服务单机部署、日志冷备未加密等十余项独立的可靠性隐患。

我们将围绕两个典型问题展开分析：
1. **细节多跳问题**：“下周六江北变电站检修，会影响到公司的核心交易网关吗？”
2. **全局归纳问题**：“这份审计报告涉及的全部基础设施中，存在哪些潜在的单点故障？”

---

## 传统文本分块与向量检索的假设与断裂

标准稠密向量检索（Dense Retrieval）的工作流程通常包含三步：将原始文本按固定字符数或标点切分成文本块（Chunks）；通过向量嵌入模型将每个分块转换为连续空间中的稠密向量；在检索时计算用户提问（Query）与各切块向量的余弦相似度，挑选最相似的 Top-K 个切块交给大语言模型。

关于向量嵌入的数值表示与几何意义，读者可参考 [/fundamentals/embedding/what-is-embedding.md](/fundamentals/embedding/what-is-embedding.md)。这种做法在工程上得以广泛应用，依赖于两个实用假设：

1. **局部信息完备假设**：回答问题所需的关键事实，集中在单篇或少数几个孤立的连续分块中。
2. **显式语义邻近假设**：用户的提问用词与包含答案的切块，在向量空间中具有足够高的余弦相似度，能通过几何近邻算法被检索召回。

![传统向量切块与相似度检索的机械截断问题分析](/images/content-rebuild/55ae7abfb750ea15/fig-1.png)

上图展示了长文本在经过机械分块后出现的上下文断裂现象：讲解员使用手持放大镜指出，代词、前置条件与核心实体被滑动窗口生硬切分在不同块中，导致孤立分块丧失独立语义。读者在审视分块策略时，需要重点观察句法依存是否跨越了切分边界。

当遇到逻辑复杂、跨越长文本的资料时，这两个假设往往会被打破：

滑动窗口分块容易截断句法与代词指代。例如切块 2 写着“该数据中心的外部供电……”，如果指代词“该数据中心”的主语留在切块 1，切块 2 就失去了独立完整的语义。更关键的是，向量余弦相似度度量的是整段文本在语义空间中的统计相近程度，它并不记录“A 依赖 B，B 连接 C”这样的结构化关系，无法沿着实体之间的逻辑链路传递推理。

---

## 向量相似度在两类任务中的失效分析

完全依赖切块余弦相似度的 Top-K 检索，会在以下两类典型场景中出现系统性漏查：

### 1. 跨文档多跳推理的链路中断

在上述案例中，用户提问：“下周六江北变电站检修，会影响到公司的核心交易网关吗？”

* 提问中包含的实体是**江北变电站**与**核心交易网关**。
* 关键桥梁是**华东一号数据中心**，但用户提问中压根没有提到这个词。
* 切块 2（数据中心与变电站）和切块 1（交易网关与数据中心）分别包含一部分事实。

此时，纯向量检索计算提问与所有切块的余弦相似度。切块 3 包含“江北变电站”和“检修”，向量相似度很高，会被召回；切块 1 包含“核心交易网关”，相似度也较高，可能被召回。但是，起着承上启下桥梁作用的切块 2 仅写了“数据中心接入变电站”，既没有提到“交易网关”，也没有提到“检修”，与提问的全局向量相似度偏低，极容易排在 Top-K 之外。

丢失了切块 2 这座逻辑桥梁，大语言模型即使具备很强的逻辑推理能力，也无法知道核心交易网关和江北变电站之间存在实际供电依赖，最终只能回答“报告未提及检修对交易网关有直接影响”。

### 2. 全局主题归纳的采样偏差

当用户提出宏观问题：“这份审计报告涉及的全部基础设施中，存在哪些潜在的单点故障？”

报告中并不存在一个直接写明“全公司单点故障清单”的现成分块。相关的风险事实分散在切块 1（网关无跨机房容灾）、切块 2（单变电站供电）、切块 15（单机鉴权）等几十个互不相连的章节中。

纯向量检索在面对“潜在的单点故障”这类宏观抽象提问时，倾向于召回包含“单点故障概述”、“基础设施高可用原则”等泛化词汇的总结性段落。由于 Top-K 只能挑选少数几个分块，系统只能采样到两三个局部的案例，或者拿到一段空泛的方法论，无法把散落在几十个切块中的具体隐患完整罗列出来。

<ReadMoreLock />

![局部语义命中在多跳推理与全局归纳中的失效对比图](/images/content-rebuild/55ae7abfb750ea15/fig-2.png)

上图对比了向量检索在两类复杂任务中的失效机制：左侧展示了缺少“中间桥梁分块”导致多跳断裂；右侧展示了面对宏观总结时，Top-K 采样陷入局部孤岛造成的视野遮蔽。讲解员通过红叉警示与断开的链路符号，说明纯余弦相似度为何难以覆盖跨文本结构线索。

---

## 知识图谱表示：实体、关系与原始文本映射

GraphRAG 并不是用图数据库完全取代文本切块，而是在文档切块之上叠加了一层显式的知识图谱（Knowledge Graph）索引。关于实体抽取与社区生成的离线构建管线，读者可参考 [/frontier/graphrag/entity-extraction.md](/frontier/graphrag/entity-extraction.md)。其数据结构主要由四部分构成：

1. **实体节点（Entity Nodes）**：文本中具有独立含义的概念、组织、系统或事件。例如 `核心交易网关`、`华东一号数据中心`、`江北变电站`。每个实体节点包含标准名称、实体类别以及由 LLM 归纳生成的实体描述。
2. **关系边（Relationship Edges）**：连接两个实体节点的有向或无向语义边。例如 `(核心交易网关) -[DEPLOYED_AT]-> (华东一号数据中心)`、`(华东一号数据中心) -[POWERED_BY]-> (江北变电站)`。每条边附带具体的关系陈述与权重。
3. **实体断言与协变量（Claims / Covariates）**：关于实体的特定状态、时间范围或结论记录，例如“检修时间：下周六全天”。
4. **源文本指针（Text Unit Mappings）**：每个实体节点和关系边，都保留指向其来源文本分块（Text Chunks）的 ID。大语言模型不仅能读到结构化的“实体-关系”三元组，还可以调取原始文本片段验证具体细节。

![知识图谱表示结构：实体、关系、属性与源切块双向映射图](/images/content-rebuild/55ae7abfb750ea15/fig-3.png)

上图阐述了 GraphRAG 的多层数据组织形式：最下层是文档切块，中层是抽取的实体节点与带箭头的关系边，最上层则汇聚为属性元数据与指针映射。讲解员正在连接实体节点与底层切块，表明图拓扑具备双向追溯到原始语料文本的确定性。

有了这套图结构，信息不再受限于 512 或 1024 个 Token 的滑动窗口。即使用户提问中缺少中间名词，系统也可以沿着边找到被隐藏的连接节点。

---

## 局部检索与全局检索的双轨架构

针对“深入某项细节”和“概括全局大局”两类完全不同的提问形态，GraphRAG 将检索过程解耦为两条执行路径：局部检索（Local Search）与全局检索（Global Search）。

| 对比维度 | 局部检索（Local Search） | 全局检索（Global Search） |
| :--- | :--- | :--- |
| **适用问题** | 针对明确主体、特定实体细节、因果关联的问题 | 针对全集宏观主题、总体趋势、全景总结的问题 |
| **核心流程** | 提问识别实体 -> 匹配种子节点 -> K-Hop 图邻居扩展 | 层次化图社区划分（Leiden） -> 社区摘要 -> Map-Reduce 汇总 |
| **上下文构成** | 种子实体、邻居实体、关系边描述、映射的原始切块 | 预先生成的分层社区摘要（Community Summaries） |
| **调用耗时与成本** | 单次调用，消耗上下文 Token 较少，响应快 | 需分批处理多个社区摘要，多次调用 LLM，响应耗时较长 |

### 1. 局部检索（Local Search）的执行机制

局部检索用于解答针对具体实体的深挖问题。以“江北变电站检修是否影响交易网关”为例，其执行步骤如下：

1. **提取种子实体**：分析用户提问，提取出 `江北变电站` 和 `核心交易网关` 两个关键实体，在图节点库中精准定位。
2. **K-Hop 子图扩展**：从 `江北变电站` 出发，沿着关系边向外延伸 1 到 2 步（Hop）。第一步找到邻居 `华东一号数据中心`，第二步沿着该数据中心找到 `核心交易网关`。此时，整条依赖链路 `(核心交易网关) -> (华东一号数据中心) -> (江北变电站)` 被完整激活。
3. **组装上下文**：系统提取这条路径上的实体描述、关系边描述（“华东一号双路市电均由江北变电站供给”），并通过源文本指针拉取相关的原始切块 1、2、3。
4. **大模型生成**：大模型接收到连贯的上下文链路，便能准确指出：核心交易网关部署在华东一号，而该机房完全依赖江北变电站供电，因此变电站停电检修会导致交易网关面临断电服务不可用的重大风险。

![Local Search 检索流与子图遍历提取展开机制](/images/content-rebuild/55ae7abfb750ea15/fig-4.png)

上图展示了 Local Search 的运行时闭环：提问进入后先定位种子实体，接着沿关系边向外激活一度和二度邻居节点，提取关联的边属性与源切块，最终聚合进 Prompt。讲解员在图上追踪展开路径，突出这一过程面向点查与深钻任务的高精确定位能力。

### 2. 全局检索（Global Search）的执行机制

全局检索用于回答“整份报告发现了哪些单点故障”这类全景宏观问题。它不依赖单点子图扩展，而是通过“层次化社区发现 + Map-Reduce 归纳”实现：

1. **离线社区检测（Community Detection）**：在图谱构建阶段，算法（如 Leiden 图聚类算法）根据节点之间连线的密集程度，把整张图划分为若干个聚类社区。比如“机房供电与网络拓扑”为一个社区，“核心业务与网关中间件”为一个社区，“认证与权限”为另一个社区。社区可以分层，顶层较为宽泛，底层更具体。
2. **离线生成社区摘要（Community Summaries）**：为每个聚类社区调用一次 LLM，生成一份结构化的《社区摘要报告》，总结该社区包含的核心实体、主要关联关系以及暴露出的主要问题。
3. **在线 Map 阶段**：用户提出宏观问题时，系统不检索底层细碎切块，而是将提问分发给各个社区的摘要报告。让大模型针对每个社区摘要独立评估相关性并打分，提取出该社区范围内的局部答案（如“机房供电社区识别出单路变电站风险”、“认证社区识别下单机部署风险”）。
4. **在线 Reduce 阶段**：系统过滤掉无关或低评分的社区，将各社区产生的局部答案按重要性聚合，再次调用 LLM 综合归纳，最终输出一份条理分明的全局单点故障清单。

![Global Search 层次化社区划分与 Map-Reduce 聚合检索机制](/images/content-rebuild/55ae7abfb750ea15/fig-5.png)

上图解析了 Global Search 的两段式结构：左半部分是基于拓扑密度划分的树状多层社区并预生成摘要，右半部分展示了 Query 广播到各社区独立打分生成中间答案，最后聚合（Reduce）输出全局结论的完整数据流。讲解员站在分流汇总节点旁，明确指出了它与传统切块抽样的因果区别。

---

## 混合路由模式与代码实现

在实际工程中，图检索和向量检索并不是非此即彼的对立关系。如果用户提问只是查找一个孤立事实（例如“华东一号机房的准确地址在哪里”），纯向量检索速度快、开销小；如果是复杂多跳或宏观归纳，才需要图谱机制介入。

常见的混合路由系统会根据用户提问特征进行分发：

```python
# 教学示意代码：演示根据提问类型分流 Local 与 Global 检索路径

class MinimalGraphRAGRouter:
    def __init__(self, entities: dict, community_summaries: list, vector_index):
        # entities 存储节点结构：实体名 -> {type, description, neighbors, chunks}
        self.entities = entities
        # community_summaries 存储预生成的社区报告列表: [{community_id, level, summary}]
        self.community_summaries = community_summaries
        # vector_index 模拟稠密向量检索组件
        self.vector_index = vector_index

    def retrieve(self, query: str, query_type: str = "auto") -> dict:
        """
        输入: query (用户提问), query_type (local 点查 / global 宏观 / auto 自动判断)
        输出: 组装好的检索证据上下文
        """
        if query_type == "auto":
            query_type = self._classify_intent(query)

        if query_type == "global":
            return self._global_search_map_reduce(query)
        else:
            return self._local_search_subgraph(query)

    def _classify_intent(self, query: str) -> str:
        # 基于关键词或轻量分类模型判断问题是宏观全景还是具体实体细节
        macro_indicators = ["总结", "全景", "整体", "潜在风险", "主要隐患", "单点故障"]
        if any(ind in query for ind in macro_indicators):
            return "global"
        return "local"

    def _local_search_subgraph(self, query: str, k_hop: int = 1) -> dict:
        # 1. 利用向量相似度定位提问中的种子实体（Entity Linking）
        seed_names = self.vector_index.search_entities(query, top_k=2)
        visited_entities = set(seed_names)

        # 2. 沿着图谱向外扩展指定步数（K-Hop）
        current_frontier = list(seed_names)
        for _ in range(k_hop):
            next_frontier = []
            for name in current_frontier:
                node = self.entities.get(name, {})
                for neighbor in node.get("neighbors", []):
                    if neighbor not in visited_entities:
                        visited_entities.add(neighbor)
                        next_frontier.append(neighbor)
            current_frontier = next_frontier

        # 3. 收集遍历到的所有实体关联的原始文本切块
        collected_chunks = []
        for name in visited_entities:
            node = self.entities.get(name, {})
            collected_chunks.extend(node.get("chunks", []))

        return {
            "mode": "local",
            "matched_entities": list(visited_entities),
            "evidence_chunks": list(set(collected_chunks))
        }

    def _global_search_map_reduce(self, query: str) -> dict:
        # 直接遍历预先生成的顶层社区摘要（不扫描细碎切块）
        selected_summaries = [
            c["summary"] for c in self.community_summaries if c.get("level") == 1
        ]
        return {
            "mode": "global",
            "target_community_count": len(selected_summaries),
            "community_summaries": selected_summaries,
            "strategy": "Map-Reduce across community reports"
        }
```

在上述代码逻辑中，对于局部问题，向量检索承担了“寻找图入口种子（Entity Linking）”的作用，找到入口后再通过图结构进行 K-Hop 扩散；对于全局问题，系统避开了细粒度分块，直接把提问广播给预先计算好的社区摘要。

![向量检索与图遍历协同工作的混合路由模式全景架构图](/images/content-rebuild/55ae7abfb750ea15/fig-6.png)

上图清晰刻画了系统在运行时的多路分发机制：提问经过意图判定后，分别流向基于向量链接的 Local 子图检索管线，或基于分层摘要的 Global Map-Reduce 管线，最后通过证据融合层交由 LLM 生成答案。讲解员在控制网关处执行策略切换，展示了双引擎互补集成的拓扑全貌。

需要注意，图谱路由与向量数据库底层的混合检索是两个不同层面的概念。向量数据库本身的混合检索通常指稠密向量与稀疏关键词向量的多路打分融合，开发者可参阅 Qdrant 官方提供的 [Qdrant hybrid queries 规范文档](https://qdrant.tech/documentation/concepts/hybrid-queries/)；而对于检索、索引与合成的通用流水线架构，可参考 LlamaIndex 官方的 [LlamaIndex RAG concepts 架构解析](https://docs.llamaindex.ai/en/stable/getting_started/concepts/)。

---

## 适用边界与常见误区

在使用 GraphRAG 时，需要明确其适用的业务边界：

* **误区一：只要用了图数据库存储数据，就等于实现了 GraphRAG**。图数据库仅提供了实体与边的持久化存储能力（例如执行 Cypher 检索）。GraphRAG 的核心在于离线阶段利用大语言模型抽取非结构化文本的实体、关系，并通过社区检测算法生成分层摘要；在线阶段通过子图激活或 Map-Reduce 归纳来支持推理。没有这套 LLM 抽取与社区抽象管线，单纯的图数据库并不能自动解决长文本的宏观总结问题。
* **误区二：GraphRAG 在所有任务上都优于传统向量 RAG**。GraphRAG 的构建成本非常高。离线阶段需要调用大量 LLM 调用来抽取实体、关系并生成各级社区摘要，建立索引的 Token 开销和耗时通常是传统向量切块的数十倍以上。对于诸如“查询某 API 的入参格式”等直接的事实性单点查询，传统向量检索或关键词检索速度更快、成本更低、命中率足够高，盲目引入图谱只会徒增系统延迟与复杂度。

关于在具体业务中如何权衡数据规模、关系密度与计算成本，读者可进一步参考 [/frontier/graphrag/graphrag-scenarios.md](/frontier/graphrag/graphrag-scenarios.md)，关于完整的技术模块设计请查阅 [/frontier/graphrag/README.md](/frontier/graphrag/README.md)。
