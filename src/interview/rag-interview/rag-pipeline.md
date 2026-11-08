---
title: RAG 的完整工作流程是什么？包含哪些阶段与核心模块？
icon: mdi:pipe
tag:
- RAG面试
- RAG流程
- 文档处理
description: 系统梳理检索增强生成（RAG）从离线入库、在线检索到在线生成的完整技术流程、阶段契约、算例与面试高频追问。
---

![RAG 系统的三阶段核心流程看板与流向](/images/content-rebuild/e030652f199d92aa/cover.png)

在检索增强生成（Retrieval-Augmented Generation, RAG）系统中，许多工程问题源于各模块职责边界不清——例如离线切片不当导致在线检索漏召回，或是检索结果未加清洗校验便直接送入大模型，导致生成幻觉或首字延迟激增。将 RAG 拆解为离线入库、在线检索与在线生成三个阶段，并建立明确的数据契约，是理解和设计 RAG 系统的基础。

在学习本篇前，建议先阅读 [/interview/rag-interview/what-is-rag.md](/interview/rag-interview/what-is-rag.md) 了解 RAG 的基础定义与它解决大模型知识时效性、幻觉问题的基本背景。

为便于理解，本文使用一个贯穿全篇的具体场景：**企业运维知识库系统**。知识库包含《网关部署手册》和《集群监控规范》等技术文档，线上用户常提出诸如“生产网关超时时间如何配置？心跳异常怎么判定？”的问题。

---

### 1. 核心口述答案：入库、检索、生成三阶段划分

如果在面试中被问到“请介绍一下 RAG 的完整流程和核心模块”，建议用以下结构化语言进行口述回答：

> **面试口述模板**：
> “RAG 的工作链路在工程上通常划分为一条离线链路和两条在线处理阶段，共包含三个核心步骤：
> 1. **离线入库期（Ingestion）**：负责知识的提取、分块与向量化。系统输入原始多格式文档，经过文本解析与版面提取、清洗降噪、切分分块（Chunking），送入嵌入模型生成稠密向量，最后把文档文本、元数据与向量索引分别写入存储引擎和向量数据库。
> 2. **在线检索期（Retrieval）**：负责根据用户输入召回最相关的候选片段。在线接收用户 Query 后，可选执行意图识别或多轮改写，再经编码后在向量库中做近似近邻检索（ANN），通常配合 BM25 关键词检索做混合召回，并使用 RRF 等算法融合结果，最后交由重排模型（Reranker）精排，输出 Top-K 个高质量相关片段。
> 3. **在线生成期（Generation）**：负责基于召回证据进行受控推理与回答。系统根据 Token 预算对召回内容做筛选或截断，拼接入提示词模板输入大语言模型。大模型在 Prefill 阶段处理上下文，并在自回归 Decode 阶段流式输出结果，最终通过引用标记映射回原始文档，实现可溯源的回答。
> 各阶段之间通过结构化的数据契约连接，设置相似度阈值和长度控制，防止下游模块出现静默截断或事实幻觉。”

![离线入库期全貌流程：解析、清洗、切片与持久化](/images/content-rebuild/e030652f199d92aa/fig-1.png)

---

### 2. 离线入库期（Ingestion）：解析、清洗、切分与持久化

离线入库期的主要目标是将格式各异的非结构化原始文件，转换成便于检索和定位的结构化片段。

#### 2.1 文档解析与清洗
在贯穿例子中，《网关部署手册》可能是带有双栏排版、配置表格和代码块的 PDF 文件。解析环节需重点处理三类问题：
- **版面还原（Layout Analysis）**：识别阅读顺序，剔除页眉、页脚和页码等干扰字符，避免正文被无序插入的标题打断。解析细节可参考 [/interview/rag-interview/rag-multiformat-parsing.md](/interview/rag-interview/rag-multiformat-parsing.md)。
- **表格结构化**：表格若被简单压平为单行文本，行列对应关系会丢失。通常需解析为 Markdown 或 HTML 表格，保留行列语义。
- **代码与公式保护**：防止配置示例（如 YAML、JSON 片段）被误清洗或因转义字符损坏。

#### 2.2 文本切块（Chunking）
解析后的长文本不能整篇存储，必须切分成大小适中的片段（Chunk）。切分策略通常包括：
- **基于文档结构的层级切分**：优先在 Markdown 的一级、二级标题处断开，保证逻辑主题独立。
- **固定长度与重叠窗口（Overlap）**：当单个章节过长时，按设定的 Token 上限进行滑动切分，保留一部分重叠区间（如 10%~20%），防止关键短语或主谓关系刚好落在切分点而被硬性切断。具体切分策略请参阅 [/interview/rag-interview/rag-chunk-size-overlap.md](/interview/rag-interview/rag-chunk-size-overlap.md) 与 [/interview/rag-interview/chunk-strategy.md](/interview/rag-interview/chunk-strategy.md)。

#### 2.3 向量化与存储映射
切分出的每个片段通过 Embedding 模型（如 `bge-large-zh-v1.5`）编码为一个高维稠密浮点向量，原理可参阅 [/interview/rag-interview/rag-embedding.md](/interview/rag-interview/rag-embedding.md)。向量库（如 Qdrant、Milvus，详见 [/interview/rag-interview/vector-database.md](/interview/rag-interview/vector-database.md)）不仅存储向量，还需存储载荷数据（Payload）和元数据，以便后续检索与权限过滤。

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ChunkRecord(BaseModel):
    chunk_id: str = Field(..., description="全局唯一分块ID，通常为 doc_id + 分块序号")
    doc_id: str = Field(..., description="所属原始文档ID")
    text_content: str = Field(..., description="切分后的文本正文")
    embedding: List[float] = Field(..., description="稠密向量数组")
    token_count: int = Field(..., description="当前分块包含的 Token 数量")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="业务元数据，包括所属部门、权限标签、更新时间、标题路径等"
    )
```

<ReadMoreLock />

![切片粒度对向量语义表达与检索上限的影响机制对比](/images/content-rebuild/e030652f199d92aa/fig-2.png)

---

### 3. 在线检索期（Retrieval）：查询编码、混合召回与重排精排

在线检索期接收用户的实时提问，核心目标是在毫秒级时间内，从离线存储的海量分块中筛选出最具相关性的少量候选片段。

#### 3.1 查询预处理与编码
用户提问经常包含指代代词或口语化表述。例如用户追问“那它的超时时间呢？”，系统需要结合多轮对话上下文改写为完整的“生产网关配置中的超时时间是多少秒？”（改写机制详见 [/interview/rag-interview/query-rewrite.md](/interview/rag-interview/query-rewrite.md)）。
改写后的 Query 送入与入库阶段相同的 Embedding 模型生成向量。若模型区分非对称检索的前缀，需按规范附加对应的查询提示（如 `query: `）。

#### 3.2 混合召回与 RRF 融合
在实际生产中，纯稠密向量检索擅长捕捉语义相近的自然语言表达，但在面对诸如错误码 `ERR_GW_504`、配置键名 `timeout_ms` 等精确字符匹配时容易出现偏置；而 BM25 等传统稀疏关键词检索擅长精确匹配，却缺乏同义词泛化能力。因此，通常采用双路混合召回（详见 [/interview/rag-interview/hybrid-retrieval.md](/interview/rag-interview/hybrid-retrieval.md)）。

两路检索产出的得分量纲与分布不同（余弦相似度数学取值在 $[-1, 1]$ 之间，而 BM25 则是非负且无固定上限的离散相关性打分），不能直接简单相加。通常使用加权互惠排名融合（Weighted Reciprocal Rank Fusion, RRF）通过名次进行打分折算：

$$RRF\_Score(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$

- $M$：召回通路集合，如 $\{\text{Dense}, \text{BM25}\}$。
- $r_m(d)$：文档 $d$ 在通路 $m$ 中的排名位置（1-based，即第 1 名为 1）。
- $k$：平滑常数，防止排名前列的文档得分出现过大断层，行业实践中常取 $k=60$。
- $w_m$：通路相对权重，默认可设为 1.0。

**RRF 简单算例**：
假设设平滑常数 $k=60$，两路权重 $w=1.0$。
- 文档 A 在 Dense 检索中排第 1 名，在 BM25 中排第 3 名：
  $$RRF(A) = \frac{1.0}{60 + 1} + \frac{1.0}{60 + 3} = \frac{1}{61} + \frac{1}{63} \approx 0.01639 + 0.01587 = 0.03226$$
- 文档 B 仅在 BM25 中排第 1 名，Dense 未召回该文档（分数为 0）：
  $$RRF(B) = 0 + \frac{1.0}{60 + 1} = \frac{1}{61} \approx 0.01639$$
文档 A 在两路中均有较好排名，融合后总分明显高于仅在单路命中第 1 的文档 B。

#### 3.3 重排模型（Reranker）精排
双塔 Embedding 模型分别对 Query 和 Passage 独立编码，无法在编码期捕捉两者的细粒度交互词特征。混合召回初筛出的 Top-30 至 Top-50 个候选片段，通常交由基于 Cross-Encoder 架构的重排模型（如 BGE-Reranker）打分（详见 [/interview/rag-interview/rerank.md](/interview/rag-interview/rerank.md) 与 [/interview/rag-interview/rag-topk.md](/interview/rag-interview/rag-topk.md)）。重排模型将 Query 与 Passage 拼接后共同输入模型计算全局交叉注意力，输出更准确的相关性得分，最终选出 Top-3 至 Top-5 个片段供大模型使用。

![在线检索期双路召回、RRF融合与重排模型精排机制](/images/content-rebuild/e030652f199d92aa/fig-3.png)

---

### 4. 在线生成期（Generation）：Token 预算规划、Prompt 组装与流式响应

生成阶段的职责不是简单拼接所有召回片段，而是进行受控的上下文管理和事实归因。

#### 4.1 Token 预算与上下文管理
大模型的自注意力计算在 Prefill 阶段需对整个输入序列做并行前向计算。未做稀疏优化的全局自注意力计算复杂度为 $O(N^2)$，序列越长，Prefill 耗时与显存激活占用就越高，会拉长首字响应时间（TTFT）。此外，过长的上下文还可能引发模型对中间证据关注度下降的现象（Lost in the Middle）。

系统在拼装 Prompt 前必须明确预算分配，从模型上下文总窗口长度（$Max_{total}$）中扣除各项固定开销：

$$Budget_{context} = Max_{total} - Tokens_{system} - Tokens_{query} - Tokens_{reserve\_output}$$

其中 $Max_{total}$ 为模型支持的最大上下文总 Token 数，$Tokens_{reserve\_output}$ 为预留给模型生成回答的最大输出长度。如果重排后的候选片段总长度超出 $Budget_{context}$，需按重排打分从低到高淘汰低分片段，或调用压缩模块抽取核心句，绝不能放任超长或任由底层库物理截断。

#### 4.2 结构化 Prompt 注入
将筛选后的高置信度片段注入提示词，并设立清晰的防幻觉约束，参考 [/interview/rag-interview/rag-prompt-design.md](/interview/rag-interview/rag-prompt-design.md)：

```markdown
你是一个严谨的运维技术助理。请严格根据以下提供的参考资料回答问题。
如果参考资料中没有提及，请明确回复“根据当前资料无法确认”，严禁凭空臆测。

### 参考资料：
[文档 1] (来源: gateway-deploy.md)
生产环境部署网关时，需配置 timeout_ms 为 3000，重试次数设为 2。

[文档 2] (来源: cluster-monitor.md)
所有工作节点的心跳间隔默认为 5 秒，连续超过 3 次未响应标记为下线。

### 用户问题：
生产网关的超时时间应该如何配置，心跳异常如何判定？

### 回答规范：
答案必须标注引用依据，例如 [文档 X]。
```

#### 4.3 流式生成与溯源引用
推理引擎在 Decode 阶段利用 KV Cache 逐字生成，通过 Server-Sent Events (SSE) 协议向前端推送数据流。模型输出形如 `[文档 1]` 的引用标记后，系统依据后处理逻辑将标记与原始 `ChunkRecord` 中的 `doc_id`、文件名、标题路径对应起来，为用户提供可点击验证的原文章节跳转（详见 [/interview/rag-interview/rag-source-citation.md](/interview/rag-interview/rag-source-citation.md)）。

![检索到生成的契约衔接：Token 预算划分与证据注入](/images/content-rebuild/e030652f199d92aa/fig-4.png)

---

### 5. 阶段交界的工程契约与防御校验

RAG 线上系统的很多偶发故障，本质是上下游数据契约不一致所致。各阶段必须在交界处设置防御式校验：

| 交界位置 | 传递数据 | 核心约束与防御校验 | 契约违背时的系统后果 |
| :--- | :--- | :--- | :--- |
| **入库 $\rightarrow$ 检索** | 向量数据库索引与 Payload | 1. 向量维度与 Embedding 模型输出严格一致。<br>2. 向量归一化状态与距离度量方式（Cosine/IP/L2）匹配。<br>3. 过滤字段已在数据库建立 Payload 索引。 | 数据库抛出向量维度不匹配错误、排序颠倒或失真召回不相关噪声、元数据未建索引导致全表过滤慢查超时。 |
| **检索 $\rightarrow$ 生成** | Top-K 候选片段集合 | 1. 文本内容不可为空。<br>2. 单条记录包含归一化相关性打分，支持阈值过滤。<br>3. 候选片段总 Token 数受上下文预算严格约束。 | 超长上下文被底层库静默截断丢失结尾信息、低相关性噪声引发模型幻觉。 |

#### 检索至生成的契约数据结构示例
```python
class RetrievedCandidate(BaseModel):
    chunk_id: str
    text: str
    score: float = Field(..., ge=0.0, le=1.0, description="重排模型输出的归一化相关性得分")
    source_file: str
    section_title: str | None = None

class GenerationPayload(BaseModel):
    query: str
    candidates: List[RetrievedCandidate]
    is_fallback: bool = False  # 当最高分数低于系统设定阈值（如 0.35）时置为 True，触发拒答
```

![在线生成期：KV Cache 预填充、流式生成与引用归因机制](/images/content-rebuild/e030652f199d92aa/fig-5.png)

---

### 6. 面试追问与常见认知误区

#### 追问 1：切片大小（Chunk Size）过大或过小，对检索和生成各有什么具体影响？
- **切片过大（如 2000 Tokens）**：单个向量承载的信息过多，多主题混合使得向量表示趋于全局平均，针对细粒度配置或事实的近邻打分降低，容易在向量检索阶段发生漏召回；召回后单块占用过多上下文预算，导致能送入的候选片段数量减少，增加无关信息对大模型推理的干扰。
- **切片过小（如 30 Tokens）**：句子往往被物理切断，失去代词指代的主语或条件从句的前置假设。虽然纯向量匹配得分可能很高，但送入生成阶段后，大模型因缺乏上下文无法还原事实逻辑，无法输出有效回答。

#### 追问 2：如果在检索阶段召回了 20 个片段，直接全部塞进提示词会带来哪些问题？
- **首字延迟（TTFT）显著增加**：Prefill 阶段针对输入 Token 一次性做并行自注意力计算，属于计算密集型过程，长上下文直接导致计算量（FLOPs）急剧上升并增大 KV Cache 与激活值的显存占用，从而显著拉长首字生成耗时。
- **事实注意力分散（Lost in the Middle）**：模型在长上下文中对放置于中间位置的证据敏感度往往低于开头和结尾，容易忽略关键事实。
- **计费与并发承载力下降**：绝大多数商业模型 API 按输入 Token 计费，输入膨胀直接导致调用成本成倍上升；私有化部署时，过长的上下文也会占用更多 KV Cache 显存，显著降低服务的最大并发吞吐量。

#### 常见错误认知纠偏
- **错误认知 1**：“只要向量数据库返回了 Top-K，就能保证找齐相关答案。”
  - *纠正*：向量检索依赖嵌入模型的空间分布映射。若涉及罕见专业缩写、特定流水号或否定语义表达，单靠向量检索容易漏召回或召回错位。工程上需配合 BM25 混合检索与 Cross-Encoder 重排才能提高 Top-K 的命中精度。
- **错误认知 2**：“入库时设置重叠窗口（Overlap）就能完全解决切片语义断裂问题。”
  - *纠正*：重叠窗口主要用于解决跨边界短语、短从句被截断的问题。如果文章核心逻辑跨越了几个完整段落，或存在跨多页的代词指代，简单的 Overlap 无法建立长距离关联，需要结合层级索引、父子分块（Parent-Child Retriever）或文档摘要等方案来补充全局语境。

![RAG 阶段契约断裂与典型失败排查图示](/images/content-rebuild/e030652f199d92aa/fig-6.png)

在掌握了 RAG 系统的整体流程划分后，关于不同格式文档的解析要点可参考 [/interview/rag-interview/rag-multiformat-parsing.md](/interview/rag-interview/rag-multiformat-parsing.md)，分块切分大小与重叠窗口的具体取舍可进一步参考 [/interview/rag-interview/rag-chunk-size-overlap.md](/interview/rag-interview/rag-chunk-size-overlap.md) 与 [/interview/rag-interview/chunk-strategy.md](/interview/rag-interview/chunk-strategy.md)。
