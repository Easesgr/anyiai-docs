---
title: GraphRAG 如何从文档中抽取实体、关系并生成社区摘要？
icon: mdi:account-network
tag:
- GraphRAG
- 实体抽取
- 关系提取
- 社区摘要
description: 拆解 GraphRAG 离线索引构建全流程：从文本切块、实体关系抽取、消歧合并，到 Leiden 社区发现、分层摘要生成及断点续跑工程设计。
---

在[上一讲 GraphRAG 基础原理](/frontier/graphrag/what-is-graphrag.md)中，我们了解了知识图谱与分层摘要如何帮助模型回答跨文档的全局概括性问题。但这些图结构和多级摘要并不是在大模型推理时实时生成的，而是来自一套离线索引构建流水线。

构建 GraphRAG 索引的核心任务，是把非结构化的文本集合，转换为一张具有清晰实体与关系的知识网络，再利用图算法自底向上生成多层级的社区摘要。本文以一套“电商微服务履约系统演进手册”为贯穿示例，拆解从原始文本切块到最终分层摘要的完整数据流、图算法逻辑、开销估算与工程容错实现。

![GraphRAG 离线索引构建全流程概览](/images/content-rebuild/3ad4789fcb87ac4d/cover.png)

---

## 文本切块与结构化实体关系抽取

索引构建的第一步是将长文档拆分为文本块（Chunk）。切块粒度直接影响信息抽取的质量：
- **切块过小（如 100~200 Token）**：跨句依赖和长距离实体关系被硬性切断，模型难以捕获完整的系统交互因果；
- **切块过大（如 3000 Token 以上）**：模型在单次提示中需要处理的信息过多，容易遗漏局部细节和次要实体。

在微软开源的 GraphRAG 默认配置中，通常选用 600 到 1200 Token 的块大小，并设置 100 Token 左右的滑动重叠（Overlap），减少实体在切块边界被截断的概率。

![文本切块与多轮实体关系抽取流向](/images/content-rebuild/3ad4789fcb87ac4d/fig-1.png)

### 抽取提示词与数据格式
切块完成后，程序调用大语言模型从每个文本块中抽取实体（Entity）与关系（Relationship）。为了让抽取结果能够被程序稳定解析，提示词中会限定输出为结构化 JSON 格式：

```json
{
  "entities": [
    {
      "name": "履约调度服务",
      "type": "系统服务",
      "description": "负责接收订单中台的消息，进行履约单拆分与仓库路由"
    },
    {
      "name": "库存中台",
      "type": "数据中心",
      "description": "维护全渠道物理库存与虚拟库存的实时扣减状态"
    }
  ],
  "relationships": [
    {
      "source": "履约调度服务",
      "target": "库存中台",
      "relation_type": "调用",
      "description": "在分单成功后发起预占库存 RPC 请求",
      "weight": 8
    }
  ]
}
```

### 多轮挖掘（Gleaning）机制
单次提示词抽取往往倾向于提取文档中最显眼的核心名词，容易漏掉背景信息中的隐含依赖。为了提升抽取的完整度，GraphRAG 引入了多轮挖掘机制：
1. **首轮抽取**：模型根据基础提示词提取明显的实体与关系；
2. **检查与追问**：程序将前一轮提取出的实体列表追加到上下文后，发起追问指令（例如：“请检查上述切片与已有抽取结果，是否仍有遗漏的中间件组件、状态转移链路或上下游依赖？若有，请按相同格式输出补充项；若无，请返回空列表”）；
3. **循环终止**：追问循环通常执行 1 到 2 次，直至模型输出空列表，或达到预设的最大追问轮次。

多轮挖掘能够补充边缘实体与深层关系，但每一轮追问都对应一次额外的完整模型 API 调用，这会使抽取阶段的 Token 开销成倍增长（例如进行 1 轮追问即意味着抽取调用量与 Token 消耗翻倍）。

<ReadMoreLock />

---

## 实体消歧、别名合并与属性清洗

如果直接把每个切片抽取出的实体与关系写入图存储，图谱会出现严重的碎片化。在我们的电商文档中，作者在第 3 节写“调度引擎”，在第 15 节写“履约调度服务”，在第 40 节简写为“调度中心”。它们在现实中指向同一个架构组件，但在文本抽取结果里是三个互相独立的节点。

![实体消歧与别名合并前后图拓扑对比](/images/content-rebuild/3ad4789fcb87ac4d/fig-2.png)

### 实体对齐三步清洗流
为了让图谱正确反映真实的实体关系，抽取出的节点与边需要经过以下对齐步骤：

1. **规则归一化**：
   - 去除空格与标点，统一英文字符大小写；
   - 加载领域词表进行确定性映射（例如强制将 `k8s` 映射为 `Kubernetes`）。
2. **候选分块与相似度匹配（Blocking & Matching）**：
   - 若对全图 $N$ 个实体进行两两语义比对，计算量为 $O(N^2)$，当实体量达到数万时开销过大；
   - 工程中通常采用分块过滤（Blocking）：仅在实体类型相同、首字母/前缀一致或局部敏感哈希（LSH）相同的候选集内部，计算实体名称与描述文本的向量余弦相似度。
3. **模型裁决与属性合并**：
   - 当两者的向量相似度高于设定阈值（例如 0.88），且规则无法直接裁定时，将两个实体的名称、类型、描述及来源切片上下文提交给小型判别模型进行确认；
   - 确认属于同一对象后，保留规范名称为主节点，将重叠边重新定向到主节点上；同时把各个切片中的实体描述拼接，提示模型归纳成一段简明的综合描述。

若不做消歧合并，关键组件会被拆散为多个低连接度的孤立节点，不仅破坏拓扑连通性，也会导致后续的社区发现算法无法把真正相关的模块归拢到同一个簇中。

---

## 层次化图社区发现：Leiden 聚类逻辑

经过消歧去重后，我们得到了一张包含数千个节点与数万条边的实体关系图。若要在回答宏观问题时遍历整张图，依然会超出大模型的上下文容量。GraphRAG 的方案是：**使用图聚类算法，将紧密关联的节点自底向上聚合为多层级的“社区”（Community），将图检索转变为多级摘要检索**。

GraphRAG 选用 **Leiden 聚类算法** 作为社区划分工具，而不是传统的 Louvain 算法。

![Leiden 算法聚类与局部细化步骤机制](/images/content-rebuild/3ad4789fcb87ac4d/fig-3.png)

### 为什么选择 Leiden 替代 Louvain？
Louvain 算法通过贪心迭代来最大化图的模块度（Modularity $Q$）：
$$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$
其中 $A_{ij}$ 为节点 $i$ 与节点 $j$ 之间的边权重，$k_i$ 与 $k_j$ 为两节点的度数，$m$ 为全图总边权，$\delta(c_i, c_j)$ 表示两节点是否被划入同一社区。

Louvain 存在一个结构性问题：在快速合并社区的过程中，容易将两个在拓扑上互不相连、但各自密集的节点团划入同一个社区编号，产生内部不连通的子图。Leiden 算法在此基础上增加了**局部细化（Refinement）阶段**：
1. **快速节点移动**：与 Louvain 类似，尝试把节点移动到能提升质量指标的邻居社区；
2. **社区局部细化**：在每个初步形成的社区内部，只允许节点与该社区内部紧密相连的核心子群结合，未紧密连通的节点会被拆出，保证划分出的子社区内部具备良好的连通性；
3. **图聚合**：将细化后的社区整体作为一个“超级节点”聚合，重新计算超级节点间的连接权重，并在聚合图上递归执行上述步骤。

### 层级树（Hierarchy Levels）控制
通过调整分辨率参数（Resolution Parameter $\gamma$），算法可以控制社区划分的粒度：
- **底层社区（Level 0）**：分辨率较高，社区数量多且范围较小，通常聚合具体组件或单个功能模块（例如“分单引擎”与“路由规则配置器”）；
- **高层社区（Level 1、Level 2 等）**：分辨率较低，社区由下层社区聚拢而成，对应宏观业务领域（例如整个“履约中枢域”或“全域供应链平台”）。

这棵自底向上的社区层级树，就是后续生成层级摘要的组织骨架。

---

## 自底向上的多层级社区摘要生成机制

得到社区层级树后，系统为树中的每一个社区节点生成一份自然语言文本摘要（Community Summary）。这一步骤采用自底向上的 Map-Reduce 逻辑。

![自底向上多层级社区摘要生成机制](/images/content-rebuild/3ad4789fcb87ac4d/fig-4.png)

### 为什么需要分层生成摘要？
如果把所有抽取的实体和关系一次性拼接为长文本交给模型，模型在面对高度离散的三元组时容易出现关键信息遗漏；更重要的是，当用户查询“履约系统在灰度切流时发生过什么异常”这类中观问题时，全图综述粒度太粗缺少细节，而单独的三元组又无法拼凑出全局因果。分层摘要能够兼顾不同层级的提问需求。

### 分层数据流拆解
摘要生成从最底层的叶子社区开始，逐级向上传递：

1. **叶子社区摘要（Level 0）**：
   - **输入**：该社区内包含的实体清单、实体详细描述，以及社区内部的关联边与其关系说明；
   - **生成要求**：指示模型归纳该模块的主要职责、核心流转流程以及已知的边界约束，输出包含标题、内容提要与关键事件的结构化文档；
   - **存储**：写入 Level 0 社区表，并为其生成文本向量嵌入，供后续向量检索使用。
2. **高层社区摘要（Level 1 及以上）**：
   - **输入**：当前社区所管辖的所有子社区摘要报告，外加跨越子社区边界的“桥接边”（Bridge Edges）；
   - **生成要求**：以子社区报告为素材，归纳高层领域之间的交互架构与主干流程，忽略单个服务内部的具体函数或配置细节；
   - **存储**：作为粗粒度视图存入上一级社区索引中。

通过这种自底向上的提炼，底层摘要保留具体实现与故障细节，高层摘要提供业务脉络与架构全貌。

---

## 索引构建的开销模型与断点续跑工程设计

构建 GraphRAG 索引需要大量的模型推理调用。在面对数百篇技术文档（百万 Token 量级）时，模型调用通常达到数千次。在实际项目中，需要提前建立开销估算模型，并通过持久化检查点来保证流水线具备容错重试能力。

![开销模型拆解与断点续跑架构设计](/images/content-rebuild/3ad4789fcb87ac4d/fig-5.png)

### Token 消耗估算模型
设原始文档被切分为 $N$ 个分块，平均每个分块包含 $T_{\text{chunk}}$ 个 Token，抽取提示词模板为 $T_{\text{prompt}}$ Token，单次抽取平均输出 $T_{\text{ext}}$ Token，挖掘追问轮数为 $G$（通常设为 1）。Leiden 聚类在各层级共生成 $M$ 个社区，单个社区平均输入上下文为 $T_{\text{comm\_in}}$ Token，社区摘要平均输出为 $T_{\text{comm\_out}}$ Token。

全流程消耗的输入与输出 Token 估算公式如下：
$$T_{\text{input}} \approx N \times (1 + G) \times (T_{\text{chunk}} + T_{\text{prompt}}) + \sum_{i=1}^{M} T_{\text{comm\_in}}^{(i)}$$
$$T_{\text{output}} \approx N \times (1 + G) \times T_{\text{ext}} + \sum_{i=1}^{M} T_{\text{comm\_out}}^{(i)}$$

**算例对比（仅作量化估算示意，不代表生产绝对开销）**：
假定一份包含 150 个分块的系统架构文档（每块约 1000 Token），抽取提示词 500 Token，抽取输出 400 Token，追问轮数 $G=1$；聚类后产生各级社区共 40 个，平均每个社区输入 2500 Token，输出摘要 600 Token。
- **传统向量索引**：只需调用 150 次 Embedding 模型，总输入约 15 万 Token，无生成输出；
- **GraphRAG 索引**：
  - 抽取阶段：调用 $150 \times 2 = 300$ 次模型，输入约 $150 \times 2 \times 1500 = 45$ 万 Token，输出约 $150 \times 2 \times 400 = 12$ 万 Token；
  - 摘要阶段：调用 40 次模型，输入约 $40 \times 2500 = 10$ 万 Token，输出约 $40 \times 600 = 2.4$ 万 Token；
  - 阶段总计调用 340 次大语言模型，消耗约 55 万输入 Token 与 14.4 万输出 Token（暂未计入消歧裁决开销）。

GraphRAG 索引是一次性离线重计算，如果在运行中途因为网络超时、格式解析失败或接口限流而中断，未做持久化的脚本会导致前面的 Token 消耗全部作废。

### 断点续跑与状态检查点设计
在工程落地时，索引构建流水线通常按阶段拆解，每个阶段完成后将中间结果以 Parquet 或 SQLite 文件落地到磁盘：

![Pipeline 阶段状态机与恢复逻辑](/images/content-rebuild/3ad4789fcb87ac4d/fig-6.png)

```python
# 伪代码：带有状态持久化检查的 GraphRAG 索引流水线编排
class GraphRAGIndexPipeline:
    def __init__(self, workspace_dir: str):
        self.storage = LocalParquetStorage(workspace_dir)

    def run(self, raw_documents):
        # 阶段 1：文档切块
        if not self.storage.exists("chunks.parquet"):
            chunks = split_documents(raw_documents, chunk_size=1200, overlap=100)
            self.storage.write("chunks.parquet", chunks)
        chunks = self.storage.read("chunks.parquet")

        # 阶段 2：实体与关系抽取 (支持基于切片 ID 的增量重试)
        if not self.storage.exists("raw_graph.parquet"):
            raw_graph = extract_entities_and_relations(chunks, max_gleanings=1)
            self.storage.write("raw_graph.parquet", raw_graph)
        raw_graph = self.storage.read("raw_graph.parquet")

        # 阶段 3：实体对齐与消歧清洗
        if not self.storage.exists("normalized_graph.parquet"):
            clean_graph = deduplicate_and_merge(raw_graph)
            self.storage.write("normalized_graph.parquet", clean_graph)
        clean_graph = self.storage.read("normalized_graph.parquet")

        # 阶段 4：Leiden 社区发现
        if not self.storage.exists("communities.parquet"):
            communities = run_leiden_clustering(clean_graph, resolutions=[1.0, 0.5])
            self.storage.write("communities.parquet", communities)
        communities = self.storage.read("communities.parquet")

        # 阶段 5：自底向上多级摘要生成
        if not self.storage.exists("community_summaries.parquet"):
            summaries = generate_hierarchical_summaries(communities, clean_graph)
            self.storage.write("community_summaries.parquet", summaries)
            
        return "索引构建完成"
```

在这类流水线中，常见的容错机制包括：
- **分块级重试**：抽取过程中如果某一个分块返回的 JSON 无法解析，程序仅重试该分块对应的请求，达到最大重试次数后记录失败日志并略过，不阻断整体流程；
- **幂等写与原子重命名**：中间文件使用临时文件名写入，写入并校验完整后再重命名为目标文件，避免因中途强退产生半写入的损坏文件；
- **调用预算阈值**：在启动大规模文本构建前，先运行切块统计，根据公式预估预期的模型调用开销，超出业务设定的预算限额时暂停并通知人工确认。

完成了实体图与社区摘要的构建后，知识网络就具备了支持多跳检索与全局宏观分析的索引基础。在下一讲[GraphRAG 业务落地场景与选型决策](/frontier/graphrag/graphrag-scenarios.md)中，我们将讨论如何衡量这套构建成本，并在业务场景中选定适合的 RAG 形态。
