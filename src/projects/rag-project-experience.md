---
title: 如何设计面向技术面试的知识库问答系统？PrepRAG 架构与业务设计
description: 解析面向技术面试的知识库问答系统 PrepRAG 的架构分层、混合检索与父子分块模型，给出了端到端流程与验收指标。
---

面试技术准备往往面临知识点分散、概念相似易混淆、技术演进时效性强三项现实阻碍。通用大语言模型直接回答面试问题时，容易对专有名词产生幻觉，或在面对对比类问题时给出泛化的定性描述，难以满足高标准技术面试对实现细节与边界条件的考查要求。

例如，当面试官提问：“在单卡 24GB 显存限制下，微调 7B 模型时 LoRA 与 QLoRA 的核心差异是什么？显存开销各由哪些部分构成？”如果系统直接将整段提问送入通用稠密向量检索，返回的往往是一篇通篇讲解 LoRA 矩阵分解的文档，导致回答中关于 QLoRA 的 NF4 量化、双重量化（Double Quantization）与分页优化器（Paged Optimizers）的推导完全缺失。

本篇聚焦 PrepRAG（AI 面试知识库问答系统）的宏观设计，说明系统业务抽象、架构分层、技术选型权衡与数据模型设计，为后续工程实现建立清晰的基础。

![PrepRAG 整体系统架构与业务流转全景](/images/content-rebuild/0edb8adc17ed079c/cover.png)

---

## 1. 业务痛点与需求边界

PrepRAG 的知识源来自结构化技术文档体系，覆盖深度学习原理、分布式训练、量化推理与工程落地等类目。相较于开放域闲聊或泛企业内部制度问答，技术面试知识库场景具有三项明确的业务特征：

1. **概念相似导致检索偏科**：技术面试高频出现横向对比考题，例如“LoRA 与 QLoRA 的区别”或“DPO 与 PPO 的损失函数推导对比”。两类技术在语义向量空间中的表征距离极近。如果检索仅依赖单一稠密向量召回，Top-K 候选集容易全篇被某一篇篇幅较长的技术长文占据，导致大模型获取的对比上下文信息残缺。
2. **专业术语与代码符号要求字面精准**：面试题高度依赖精确术语，如 `vLLM` 的 PagedAttention 算子、FlashAttention 的分块计算、AdamW 优化器超参数等。若文本被切分器机械按固定 Token 长度截断，关键字与代码标识符可能被切散，导致语义检索失真。
3. **多模态图表信息断层**：技术文档包含大量流程图、网络拓扑图与对比表格。在静态 Markdown 中，图片通常仅以 `![alt](url)` 形式存在，若不进行多模态内容提取与文本化关联，图片所承载的步骤和架构关系在文本检索阶段将处于盲区。

![面试知识库场景下的概念混淆与信息检索断层痛点](/images/content-rebuild/0edb8adc17ed079c/fig-1.png)

根据上述业务痛点，PrepRAG 将系统目标定义为：在结构化 Markdown 文档与技术题库之上，提供可溯源、低幻觉、支持概念对比的多轮面试问答辅助服务。

---

## 2. 核心系统指标与验收约束

为确保工程设计的可验证性，系统围绕时延、召回能力与稳定性制定验收指标：

*   **端到端时延（P90）**：在单卡或标准云主机部署条件下，首字流式输出时延（TTFT, Time to First Token）控制在 $800\text{ ms}$ 以内；在 Top-5 检索上下文拼接量约 $3000\text{ tokens}$ 条件下，完整回答生成时延（P90）控制在 $4.0\text{ s}$ 以内。
*   **检索召回指标**：在构建的 200 道典型面试基准评测集（包含 60 道对比题、80 道概念题、60 道推导实现题）上，综合检索 Hit@5 目标达到 $85\%$ 以上。
*   **并发与降级约束**：开发与测试环境下支持 10 QPS 稳定响应；在向量检索服务发生瞬时超时（阈值设为 $1.5\text{ s}$）时，系统需具备关键词退避检索降级通道，直接返回 BM25 结果并给出降级日志，避免阻塞问答流水线。

---

## 3. 总体系统架构设计

PrepRAG 采用解耦的三层架构：离线数据管道（Ingestion Pipeline）、在线检索计算引擎（Retrieval & Serving Engine）以及交互网关层。

<ReadMoreLock />

![PrepRAG 系统分层架构与核心数据管道流向](/images/content-rebuild/0edb8adc17ed079c/fig-2.png)

### 3.1 离线数据构建管道
离线管道负责将原始技术仓库解析为可检索的高质量单元：
*   **文档结构提取器**：解析 Markdown 的 Frontmatter 元数据（分类、标签、知识域），利用 AST（抽象语法树）识别 H1/H2/H3 标题层级，保留小节所属的标题上下文路径。
*   **图文增强处理器**：提取 Markdown 中的图片标签与 Alt 说明，调用多模态模型离线生成结构化视觉描述（提取图中的步骤关系、表格字段与组件连接），将描述以结构化文本形式拼装到图表所属的切片中。
*   **父子分块切分（Parent-Child Indexing）**：采用两级切分策略。小块（Child Chunk，约 256 tokens）保留紧凑的局部特征，用于向量匹配与关键词索引；大块（Parent Chunk，通常为包含完整上下文的章节，约 1024 tokens 左右）用于装配最终输入给大模型的提示词。

### 3.2 在线检索计算引擎
*   **查询解析器（Query Parser）**：负责多轮对话的指代消解与检索词扩充，将用户的简短追问还原为具备完整检索线索的独立 Query。
*   **混合检索器（Hybrid Retriever）**：并行触发稠密向量检索（Dense Search）与稀疏关键词检索（BM25），通过倒数排名融合算法（Reciprocal Rank Fusion, RRF）合并双路候选集。
*   **重排器（Cross-Encoder Reranker）**：对混合检索返回的前 $K=20$ 个候选分块执行查询与文本块的联合打分，输出最终的 Top-5 强相关切片。

### 3.3 交互网关与应用层
基于 FastAPI 构建异步 Web 服务，提供 Server-Sent Events (SSE) 协议，实现流式输出、引用来源标注与关联文档锚点展示。

---

## 4. 技术选型决策与因果论证

工程选型需要立足具体的项目规模与维护成本：

![技术选型权衡：存储与检索机制的多维对比](/images/content-rebuild/0edb8adc17ed079c/fig-3.png)

### 4.1 向量存储选型：自建轻量级存储 vs 云端全托管服务
*   **选型结论**：采用轻量级自建方案（如 Qdrant 或 ChromaDB 独立实例），不接入云端全托管向量数据库。
*   **决策原因**：技术面试知识库通常在数千篇文档规模，切分后的子块数量处于 $10^4 \sim 10^5$ 量级。单节点部署的轻量向量库内存占用通常低于 $1\text{ GB}$，通过本地网络或进程内通信即可满足检索要求，避免了跨公网调用延迟与额外的服务账单。
*   **边界代价**：单节点部署缺少开箱即用的跨可用区容灾，但对于知识库构建类场景，数据源可从 Git 仓库随时重建，本地持久化挂载卷已能满足容灾需求。

### 4.2 检索范式选型：Dense 向量检索 vs 混合检索（Dense + BM25）
*   **选型结论**：采用“稠密向量检索 + BM25 稀疏检索 + RRF 融合”的混合检索方案。
*   **决策原因**：稠密向量检索擅长匹配近义表达与抽象意图，但面对缩写术语、版本号、代码变量名（如 `bge-m3`、`fp16`、`q_proj`）时，容易因嵌入空间的平滑特性而产生语义偏移；BM25 基于精确词频匹配，对关键专有名词命中具有高确定性。
*   **RRF 融合机制与算例**：
    不同检索通路的得分尺度不同（向量余弦相似度在 $[-1, 1]$ 之间，BM25 得分为非负实数且无固定上限），直接加权打分需要复杂的归一化调优。RRF 仅根据排位计算融合得分，打分公式为：
    $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
    其中 $M$ 为检索通路集合，$r_m(d)$ 为文档 $d$ 在第 $m$ 路检索中的排位（从 1 开始计），常数 $k$ 通常取 60。

    **融合算例演示**：假设常数 $k=60$，候选池中有文档 A 与文档 B：
    *   文档 A 语义相关但术语匹配弱：Dense 排名第 1，BM25 排名第 20。
        $$RRF(A) = \frac{1}{60 + 1} + \frac{1}{60 + 20} = \frac{1}{61} + \frac{1}{80} \approx 0.01639 + 0.01250 = 0.02889$$
    *   文档 B 兼具语义与精确术语：Dense 排名第 3，BM25 排名第 2。
        $$RRF(B) = \frac{1}{60 + 3} + \frac{1}{60 + 2} = \frac{1}{63} + \frac{1}{62} \approx 0.01587 + 0.01613 = 0.03200$$
    计算结果显示，$RRF(B) > RRF(A)$。虽然文档 A 在向量检索中位列第一，但文档 B 在两路检索中均位于前列，综合排位更高，符合对比类问题的召回诉求。

---

## 5. 数据模型设计

知识切分如果破坏了面试题的原生结构，大模型将无法获得完整的上下文推导。系统在数据结构层对切片与文档进行解耦建模：

![PrepRAG 核心数据模型与父子分块关联结构](/images/content-rebuild/0edb8adc17ed079c/fig-4.png)

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class KnowledgeChunk(BaseModel):
    """最小检索单元（子块，用于向量匹配与BM25索引）"""
    chunk_id: str = Field(description="切片全局唯一标识")
    parent_id: str = Field(description="所属父文档章节的唯一标识")
    content: str = Field(description="子块文本内容，控制在256 tokens左右")
    category: str = Field(description="一级分类，例如 '分布式训练'")
    sub_category: str = Field(description="二级分类，例如 '显存优化与量化'")
    tags: List[str] = Field(default_factory=list, description="关键词标签，例如 ['LoRA', 'QLoRA', 'NF4']")
    has_image: bool = Field(default=False, description="当前块是否关联图表增强描述")
    image_caption: Optional[str] = Field(default=None, description="离线提取的图表结构化文本描述")

class ParentDocument(BaseModel):
    """上下文装配单元（父块，用于组装输入提示词）"""
    doc_id: str = Field(description="完整章节或面试题唯一标识")
    title: str = Field(description="文章主标题或考题名称")
    full_content: str = Field(description="未截断的小节完整内容，包含完整推导与代码片段")
    source_file: str = Field(description="Markdown 源文件相对路径")
    child_chunk_ids: List[str] = Field(default_factory=list, description="包含的所有子块ID列表")
```

在检索阶段，系统利用 `KnowledgeChunk.content` 进行高精度的语义与关键词匹配；在组装 LLM 输入时，系统根据命中子块的 `parent_id` 抓取对应的 `ParentDocument.full_content`，避免了小切片导致上下文信息碎片化的问题。

---

## 6. 端到端工作流时序流转

当用户发起一轮面试技术提问时，流水线各阶段的执行逻辑如下：

![用户端到端查询与流水线处理时序图](/images/content-rebuild/0edb8adc17ed079c/fig-5.png)

1. **请求入参解析**：网关接收包含 `session_id` 和用户提问的请求体，执行参数校验与流控检查。
2. **多轮指代消解**：若存在多轮上下文，调用轻量模型将模糊追问还原为完整检索语句。例如用户追问：“它的反向传播怎么更新？”，结合上文改写为“QLoRA 微调中适配层权重的反向传播求导与梯度更新机制”。
3. **并行双路检索**：
   *   **Dense 检索**：调用文本嵌入模型生成向量，在向量数据库中检索余弦距离最近的前 20 个子块。
   *   **Sparse 检索**：对改写后的 Query 进行分词，在 BM25 索引中检索匹配度最高的前 20 个子块。
4. **RRF 融合与重排**：通过 RRF 算法合并两路结果，选取前 20 个候选块；使用 Cross-Encoder 重排模型对查询与各候选子块进行联合打分，截取前 5 个强相关子块。
5. **父上下文去重与拼接**：提取这 5 个子块的 `parent_id`，查询对应的 `ParentDocument` 并按章节顺序去重合并，填充至系统提示词模板中。
6. **流式生成与溯源输出**：大模型执行自回归推理，通过 SSE 协议将生成的 Token 流式推送到前端，并在回答末尾附上引用的源文件路径与图表定位标识。

---

## 7. 演进路径与可观测性规划

系统研发划分为三个明确的构建阶段，并配合细粒度的链路打点以便排查问题：

![项目演进里程碑与可观测性监控大盘规划](/images/content-rebuild/0edb8adc17ed079c/fig-6.png)

### 7.1 三阶段演进路径
*   **阶段一：基础验证原型（Baseline RAG）**
    *   实现 Markdown 结构化切分，基于向量数据库和单路 Dense 检索打通基础问答链路。
    *   目标：验证“文档切分-向量写入-语义检索-大模型回答”的基础链路通畅性。
*   **阶段二：混合检索与多模态扩展（Advanced Retrieval）**
    *   引入 BM25 实现双路召回与 RRF 融合，引入多模态描述提取 Markdown 拓扑图与对比表内容。
    *   接入父子分块结构与 Cross-Encoder 重排，改善专有名词漏检与对比题上下文断裂情况。
*   **阶段三：链路加固与质量评估（Production Readiness）**
    *   接入端到端 Trace 追踪与打分监控，实现低置信度退避机制（当重排最高得分低于预设阈值时，明确提示“当前知识库缺少充分证据”，避免模型强行猜测）。

### 7.2 链路可观测性设计
每个请求均会生成唯一的 `trace_id`，记录各阶段执行耗时与中间结果元数据：

```json
{
  "trace_id": "trace_20260329_092011_a4f8",
  "latency_ms": {
    "query_rewrite": 210,
    "vector_retrieval": 45,
    "bm25_retrieval": 12,
    "rrf_merge": 3,
    "reranking": 180,
    "llm_ttft": 320,
    "llm_total": 2150
  },
  "retrieval_stats": {
    "dense_candidates_count": 20,
    "sparse_candidates_count": 20,
    "rerank_top1_score": 0.892,
    "rerank_top5_score": 0.641
  },
  "fallback_triggered": false
}
```

当问答结果出现偏离时，研发人员可通过日志直接定位问题发生在“召回未命中”、“重排置信度过低”还是“大模型未严格遵循上下文”。

---

## 关联阅读与下一阶段

*   基础概念：阅读 [/projects/ai-knowledge-base.md](/projects/ai-knowledge-base.md) 了解知识库基础构建与切分方法。
*   工程落地：进入 [/projects/rag-implementation.md](/projects/rag-implementation.md) 查看 PrepRAG 的切分器、多模态提取脚本与 FastAPI 流式接口实现。
*   技术复盘：进入 [/projects/rag-interview-qa.md](/projects/rag-interview-qa.md) 查看 PrepRAG 在技术求职中的架构问答表达与追问解析。
*   上下文工程：参考 [/interview/ai-coding-interview/project-context.md](/interview/ai-coding-interview/project-context.md) 了解代码与结构化上下文组装方式。
