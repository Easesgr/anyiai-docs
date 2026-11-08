---
title: Agentic RAG 是什么？与传统 RAG 有何区别？
icon: mdi:database-search
tag:
- Agent面试
- Agentic RAG
- 检索增强
description: 详解 Agentic RAG 与传统静态检索流水线的区别，通过财报对比实例解析查询拆解、多源路由、证据评估与自纠错机制，附工程开销算例与面试答题点。
---

![Agentic RAG 架构与动态检索决策封面图](/images/content-rebuild/f700fd985c5331b9/cover.png)

在面试中被问到“Agentic RAG 与传统 RAG 有什么区别”时，回答如果只停留在“Agentic RAG 引入了大模型做思考和规划”，往往很难体现工程深度。面试官真正考察的，是候选人是否清楚传统检索生成流水线的失效边界，以及如何用带有状态反馈的循环控制流解决多跳、模糊与多数据源检索问题。

本讲结合 [Agent 的核心流程](/interview/agent-interview/agent-loop.md) 与 [Tool Calling 机制](/interview/agent-interview/tool-calling.md)，拆解 Agentic RAG 的控制流设计、评估自纠错实现、混合路由逻辑与延迟成本边界。

---

## 面试标准口述：传统静态 RAG 与 Agentic RAG 的区别

面试官提问：“请用两分钟概括传统静态 RAG 与 Agentic RAG 的根本区别，并说明后者的核心运行机制。”

### 口述核心回答（建议耗时 60-90 秒）

> “两者的核心区别在于**流程控制方式不同**。
>
> 传统静态 RAG 是**确定性的单向流水线**。无论用户提问简单还是复杂，系统都遵循固定步骤：把输入 Query 向量化 $\rightarrow$ 单次向量检索召回 Top-K 文档 $\rightarrow$ 拼接进 Prompt $\rightarrow$ 大模型生成回答。这种管道假定单次检索一定能召回完整证据。面对多实体比较或复杂推理问题时，只要单次检索漏掉关键信息，生成阶段就会直接产生事实缺陷或幻觉。
>
> Agentic RAG 则将检索系统重构成**由模型驱动工具调用的状态机循环**。系统把向量库、全文索引、知识图谱等检索能力封装成工具，让大模型充当流程调度器：
> 1. **查询分析与拆解**：面对复杂问题，先将主查询分解为若干子问题；
> 2. **自适应路由**：根据子问题类型，自主决定调用哪类数据源工具；
> 3. **证据评估与自纠错**：检查召回内容是否足够回答问题，证据不足时自动改写查询词重试；
> 4. **条件停止**：只有收集到充足证据，或触发最大重试次数等硬边界时，才退出循环生成最终答案。
>
> 简单来说，传统 RAG 是‘单向直达’的固定链路，Agentic RAG 是‘检索—评估—补充纠错’的自适应闭环。”

![传统静态 RAG 与 Agentic RAG 架构与控制拓扑对比](/images/content-rebuild/f700fd985c5331b9/fig-1.png)

如图所示，传统流水线无论检索质量高低，都直接进入生成步骤；Agentic RAG 则将检索作为受控工具，在评估反馈的驱动下决定是继续检索还是输出结果。

<ReadMoreLock />

---

## 贯穿场景：为什么单次检索会失效？

我们以一个具体问题贯穿整篇分析：

> **业务提问**：“对比苹果与微软 2023 财年的云计算业务收入增速，并结合各自研发投入占比分析其战略侧重。”

如果用传统单向 RAG 处理该问题：
1. 系统把整段长句子一次性转成稠密向量；
2. 在向量库中查找语义最接近的 5 个文档切片；
3. 召回结果大概率会集中在谈论“两家公司云计算”的概括性新闻或某一份财报的开头，很难在单次 Top-K 列表中同时精准覆盖四份分散的报告细节：苹果 2023 财报研发费用、苹果服务与云业务披露口径、微软 2023 财报智能云收入增速、微软整体研发支出。
4. 大模型在输入证据残缺的情况下被迫作答，极易出现把某一方数据张冠李戴，或直接回答“未找到相关信息”。

Agentic RAG 的解决思路是将一次性检索泛化为多阶段动态决策流：

```
                    [用户原始复杂查询]
                            │
                            ▼
                  [1. 查询拆解与规划]
                   ├── 意图判别与重写
                   └── 子问题分解 (Sub-queries)
                            │
                            ▼
┌─────────────────► [2. 动态路由执行] ◄─────────────────┐
│                  ├── 工具与数据源分发                   │
│                  └── 参数化检索调用 (Tool Call)        │
│                           │                            │
│                           ▼                            │
│                 [3. 证据抽取与评估]                     │
│                  ├── 文档相关性判别                    │
│                  └── 证据充分性推断                    │
│                           │                            │
│                 是否充分且无事实冲突？                  │
│                 /                     \                │
│             (否)                       (是)            │
│              /                           \             │
│     [重写查询词 / 补齐缺口]         [4. 综合答案生成]   │
└─── (触发下一轮检索循环)                   │             │
                                            ▼            │
                                       [输出响应]         │
                                                         │
              (循环达到 max_retries 上限强制降级) ─────────┘
```

### 1. 查询拆解与重写（Decomposition & Rewriting）
模型接收原始复杂提问后，识别出这是一个依赖多方数据的复合任务，将其拆解为独立执行的子问题：
- 子查询 A：“苹果公司 2023 财年研发支出及占总营收比例”
- 子查询 B：“微软 2023 财年 Intelligent Cloud 业务营收增速”
- 子查询 C：“微软 2023 财年总研发投入及占比”

若某些子任务需要分步依赖，规划器可构建有向无环图逐步求解。关于复杂规划机制可参阅 [任务规划与步骤拆解机制](/interview/agent-interview/agent-planning.md) 与 [复杂查询规划专题](/frontier/agentic-rag/complex-query-planning.md)。

### 2. 多源路由（Routing）
系统根据子任务属性匹配最合适的工具，例如调用财报结构化数据库（SQL 工具）查询研发与营收确切数值，调用文本向量库检索战略描述段落。

### 3. 自适应停止机制（Adaptive Stopping）
检索循环的退出条件不能单纯靠运气，常见有两种控制策略：
- **模型内部决策退出**：大模型在多轮思考中发现证据已完整，不再发起检索 Tool Call，直接进入内容生成。
- **独立判决节点校验（Deterministic Evaluator）**：由轻量级判决模型或带结构化输出校验的代码节点，评估已检索片段对各子问题的覆盖率。当信息覆盖完整，或当前循环轮数达到设定的上限 `max_iterations` 时，强制停止检索。

![Agentic RAG 动态决策流与自适应停止判断](/images/content-rebuild/f700fd985c5331b9/fig-2.png)

上图展示了从子问题分解到自适应停止的判定分流：模型依据每一步环境反馈（Observation）更新状态，决定是继续补充检索还是汇合输出。

---

## 检索评估与自纠错机制

在传统 RAG 中，如果检索器返回了错误或不相干的文本切片，模型很容易受偏置内容影响产生误导。在 Agentic RAG 中，评估器（Grader）与改写重试构成系统的自纠错机制。

### 最小自纠错状态机实现（Python 逻辑示意）

以下伪代码展示如何利用显式状态机管理检索、文档评估与查询改写的流转过程：

```python
from typing import List, Dict, Any

class AgenticRAGState:
    def __init__(self, query: str):
        self.original_query: str = query
        self.current_search_query: str = query
        self.documents: List[str] = []
        self.iterations: int = 0
        self.max_iterations: int = 3
        self.is_sufficient: bool = False
        self.missing_aspect: str = ""
        self.final_answer: str = ""

def retrieve_node(state: AgenticRAGState, retriever_tool) -> AgenticRAGState:
    state.iterations += 1
    # 调用检索工具拉取文档块
    raw_docs = retriever_tool.invoke({"query": state.current_search_query})
    state.documents.extend(raw_docs)
    return state

def grade_documents_node(state: AgenticRAGState, grader_llm) -> AgenticRAGState:
    # 评估当前所有文档是否足够回答原始问题
    prompt = f"""分析以下文档是否足以回答问题：'{state.original_query}'。
已检索文档列表：{state.documents}
请以 JSON 格式输出：{{"sufficient": true/false, "missing_aspect": "缺少的具体信息描述"}}"""
    
    eval_result = grader_llm.invoke_structured(prompt)
    state.is_sufficient = eval_result.get("sufficient", False)
    state.missing_aspect = eval_result.get("missing_aspect", "")
    return state

def rewrite_query_node(state: AgenticRAGState, rewriter_llm) -> AgenticRAGState:
    # 针对缺失的信息盲区生成更精准的后续检索词
    prompt = f"""原始目标：{state.original_query}
当前信息缺口：{state.missing_aspect}
请生成一条针对该缺口的具体搜索词："""
    state.current_search_query = rewriter_llm.invoke(prompt).strip()
    return state

# 路由条件判断函数
def decide_next_step(state: AgenticRAGState) -> str:
    if state.is_sufficient:
        return "generate_final_answer"
    if state.iterations >= state.max_iterations:
        return "generate_with_fallback"  # 达到重试预算上限，降级生成
    return "rewrite_and_retry"
```

### 常见纠错策略
1. **假设性文档嵌入（HyDE）**：当用户查询极短、缺乏上下文关键词时，先由大模型生成一份假想答案，用假想答案的向量去匹配真实文档。
2. **后退一步提问（Step-back Prompting）**：当细节维度的检索连续落空时，将检索目标提升一层到概念或行业定义，先检索背景框架，再向细分指标收敛。
3. **文档清洗与重排（Pruning & Rerank）**：Grader 节点过滤掉被判定为噪声的文档块，防止无关上下文挤占模型工作窗口。关于上下文预算与管理的细节，可参考 [上下文管理为什么重要](/interview/agent-interview/context-management.md)。

![检索评估与自纠错反馈循环](/images/content-rebuild/f700fd985c5331b9/fig-3.png)

如图所示，`grade_documents_node` 充当中间检查点。当证据完整性不足时，系统不会强行调用生成器，而是触发改写节点生成新查询词，重新执行检索。

---

## 混合数据源路由：将向量库与数据库抽象为工具

在真实业务架构中，单一向量索引很难覆盖所有查询形态：
- **稠密向量数据库（Dense Vector Store）**：擅长匹配自然语言语义、模糊意图，但处理专有名词、流水号、精确财务数字时容易漂移；
- **倒排索引（BM25 / Sparse Vector）**：擅长特定关键字、代码、型号精确命中，但不理解语义变体；
- **知识图谱（Knowledge Graph / Cypher 查询）**：擅长多跳实体关联与股权层级穿透，但文本关系抽取成本高；
- **外部网络搜索（Web Search API）**：补充最新动态与突发事实。

在混合检索层面，现代检索系统（如 Qdrant）常采用加权倒排互惠排名（Weighted RRF）融合不同检索器的结果：

$$RRF\_Score(d) = \sum_{m \in M} w_m \cdot \frac{1}{k + r_m(d)}$$

其中 $M$ 为检索器集合，$r_m(d)$ 为文档切片 $d$ 在第 $m$ 个检索器结果中的排位位次（通常从 1 开始），$k$ 是防止分母过小的平滑常量（通常取 60），$w_m$ 为对应检索器的权重。而在 Agentic RAG 中，**路由决策可以在大模型工具调度层直接发生**。

### 工具化定义与调用

系统通过将不同检索器声明为标准 JSON Schema（可参考 [Tool Calling 机制](/interview/agent-interview/tool-calling.md) 以及 [MCP 协议概念](/interview/agent-interview/mcp.md)），大模型可按需派发结构化检索请求：

```json
{
  "tool": "financial_sql_query",
  "parameters": {
    "company": "Apple",
    "fiscal_year": 2023,
    "metrics": ["total_revenue", "rd_expenses"]
  }
}
```

在面对复杂问题时，模型可在第一轮并行发起调用：一边调用 `financial_sql_query` 查询数值，一边调用 `vector_store_search` 检索年报中管理层讨论的战略文本，再由后续节点做异构证据聚合。多步协作可参考 [多步检索与答案校验](/frontier/agentic-rag/multi-step-retrieval.md)。

![混合数据源统一工具化抽象与路由架构](/images/content-rebuild/f700fd985c5331b9/fig-4.png)

图中展示了模型利用统一定义的工具接口作为总控，依据子查询特点分流至密集向量库、知识图谱与外部搜索工具，并汇总异构证据。

---

## 成本与性能边界：召回质量与延迟的权衡

评估技术方案不能脱离工程代价。Agentic RAG 的灵活性，实质上是以**调用轮次增加、端到端延迟拉长、Token 消耗放大**为代价换取对复杂任务的解答能力。

### 性能与消耗对比

| 维度 | 传统静态 RAG | Agentic RAG（假定 $N$ 轮迭代） |
| :--- | :--- | :--- |
| **LLM 调用次数** | 固定 1 次（终答生成） | $1 + N \times k$ 次（包含路由、评估、改写、终答） |
| **端到端延迟（Latency）** | 向量检索 + 单次模型推理（通常 $\le 1.5$ 秒） | 多轮工具交互 + 评估 + 改写 + 终答（通常 $3.5 \sim 15$ 秒） |
| **输入 Token 累计** | 单次发送：系统提示 + 检索切片 | 梯级累加：工具定义 Schema + 多轮推理记录 + 多批次切片 |
| **执行路径确定性** | 确定性流水线，仅受采样随机性影响 | 动态状态流转，可能受工具误选或评估偏差扰动 |
| **复杂多跳问题表现** | 极易因单次召回证据缺失而答错 | 通过多轮补充检索显著提升证据完整性 |

### 相同任务下的成本算例对比

以“对比两家公司跨年度多维度财报数据”为例进行示意性测算。设定单次大模型基础推理耗时（如路由、评估或上下文较少的静态生成）约为 800ms，多上下文复杂终答综合生成约为 1200ms，单次工具检索耗时 100ms：

- **方案 A：静态 RAG**
  - 执行 1 次单向检索（100ms） + 1 次终答生成（800ms）。
  - **总耗时**：$\approx 900\text{ ms}$。
  - **Token 消耗**：约 2,000 Token。
  - **表现**：单次检索漏掉了微软研发投入占比，最终回答残缺不全。

- **方案 B：Agentic RAG（假定迭代 2 轮退出）**
  - 步骤 1：主模型规划并并行调用 2 次检索工具（模型路由 800ms + 并行检索 100ms = 900ms）；
  - 步骤 2：评估节点发现缺少微软研发数据，触发改写并补充检索（评估改写 800ms + 检索 100ms = 900ms）；
  - 步骤 3：证据齐全，综合所有文档生成终答（终答生成 1200ms）。
  - **总耗时**：$900 + 900 + 1200 = 3000\text{ ms}$。
  - **Token 消耗**：多轮历史轨迹和两批切片累加，约 8,500 Token。

**权衡结论**：Agentic RAG 用约 3.3 倍的延迟和约 4.25 倍的 Token 成本，换取了多实体复杂任务的正确解答率。如果业务场景是在线高并发客服（SLA 严格限制在 1.5 秒内），采用 Agentic RAG 会直接突破超时阈值；而对于深度研报分析或离线工单诊断，这种延迟和成本换取正确率的权衡则是合理的。

![召回质量提升与延迟及 Token 开销权衡示意](/images/content-rebuild/f700fd985c5331b9/fig-5.png)

图中展示了典型的收益边际递减关系：随着检索迭代轮次增加，证据完整度提升逐渐饱和，而延迟与 Token 开销呈持续增长，系统设计需要找到合理的重试折中点。

---

## 面试深度追问与常见误区

### 追问一：如何避免系统在自纠错中陷入“检索-评估不满足-改写重搜”的无限死循环？

死循环通常由于评估节点标准过严，或知识库中根本不存在目标数据（超出数据分布，Out-of-Distribution）导致。必须由外围确定性代码实现多重防护：
1. **硬步数计数器（Max Iterations Cap）**：状态机必须配置全局循环上限（如 `max_iterations = 3`）。一旦计数超限，立即切断循环流转，进入降级策略。
2. **语义停滞判定（Semantic Stagnation Detection）**：计算连续两轮改写 Query 的向量相似度，或比对检索召回 Document ID 的重叠比例。若两轮查询语义高度相近，说明重写已无法产生增量信息，强制中断循环。
3. **诚实降级机制（Graceful Fallback）**：达到上限退出时，系统明确告知用户当前基于有限证据的推导结论，并标注未查证的具体维度，而不是强行编造或反复死扛。关于工程异常恢复的具体模式，可参考 [Agent 执行失败时如何恢复](/interview/agent-interview/agent-error-handling.md)。

### 追问二：Agentic RAG 的效果该怎么评测？还能只看传统 Ragas 指标吗？

传统 RAG 评测主要聚焦在静态三元组 `(Query, Context, Answer)`，通过 Context Precision、Faithfulness 等指标评估。在 Agentic RAG 中，评测需要拆解为链路各节点的断言与整体指标结合：
1. **路由选择准确率（Routing Accuracy）**：针对已知问题，测试模型是否准确匹配了正确的数据源工具；
2. **改写增益度（Query Rewrite Gain）**：对比改写后的查询词相比原始查询在候选集命中（Recall@K）上的净提升；
3. **步骤收敛效率（Step Efficiency）**：记录解决问题所需的平均交互轮数，轮数异常偏高说明规划或改写策略存在抖动；
4. **抗噪能力（Noise Robustness）**：在上下文注入干扰文档，检测 Grader 节点的辨别剔除准确率。

![Agentic RAG 面试核心考点排查与错误模式梳理](/images/content-rebuild/f700fd985c5331b9/fig-6.png)

上图归纳了系统在长链条运行中需要设置的防线：包含最大步数熔断、停滞检测和分段指标监控。

### 常见错误观念澄清

- ❌ **误区 1：“Agentic RAG 一定比传统 RAG 更好，应该全面替代。”**
  - 正确理解：简单事实查询（如“内部考勤打卡时间”）单次检索召回率极高，强行上 Agentic RAG 只会带来数秒的无谓延迟与更高的错误概率。
- ❌ **误区 2：“只要在 Prompt 里写明‘如果找不到就多找几次’，就是 Agentic RAG。”**
  - 正确理解：单纯依靠提示词让模型自行反复决定缺乏可靠的退出保障。工程落地通常需要配合显式状态控制（无论是原生代码状态循环还是工作流运行时）、结构化 Tool Schema 以及最大步数、停滞检测等确定性硬分支代码做熔断兜底。
- ❌ **误区 3：“Agentic RAG 能完全消除模型幻觉。”**
  - 正确理解：外部检索能提供可靠事实依据，但模型对证据的理解、提取与综合仍由概率采样决定，评估节点本身也有误判概率，无法做到数学意义上的绝对零幻觉。

---

## 总结

1. **控制流转变**：传统 RAG 是静态、单向的线性流水线；Agentic RAG 是由大模型作为调度器、按需调用检索工具的状态机闭环。
2. **四个关键环节**：查询拆解规划、多源动态路由、文档相关性评估、带预算上限的自适应纠错与停止。
3. **工程权衡法则**：Agentic RAG 解决的是“单次向量匹配无法处理的复杂多跳问题”，代价是端到端延迟与 Token 开销的倍数放大，架构选型需严格对照 SLA 与业务 ROI。
