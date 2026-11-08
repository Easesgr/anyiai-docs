---
title: Agentic RAG 是什么？它与传统 RAG 最大的区别在哪里
icon: mdi:database
tag:
- Agentic RAG
- RAG
- AI Agent
- 检索增强
description: Agentic RAG 将检索从单向静态流水线改造为模型自主调用的工具循环。本文用多跳查询实例拆解其执行机制、上下文开销与选型边界。
---

先看一个具体的业务查询：

> **用户提问**：“我司在 2023 年采购了哪个版本的向量数据库？该版本支持的最大向量维度是多少？”

如果将这句话直接输入给传统的检索增强生成（RAG）系统，系统会将其转换为向量，并在知识库中执行一次相似度检索。然而在企业内部，“采购合同”（记录采购了 Qdrant v1.4.0）与“技术手册”（记录该版本的最大向量维度限制）往往存放在完全不同的文档中。单次向量检索很难同时兼顾两个语义完全不同的事实，最终检索到的上下文往往只包含采购记录或只包含产品手册，导致大模型因为信息不全而答非所问。

**Agentic RAG 与传统 RAG 的核心区别，不在于底层的向量数据库或切片算法，而在于谁来控制检索的执行过程：**

* **传统 RAG 是静态代码驱动的单向管道**：查什么、查几次、去哪里查，完全由开发者提前写死在代码中。流程只能从检索单向流向生成，中间无法根据检索质量自动调整。
* **Agentic RAG 是模型驱动的探索循环**：检索组件被封装为大模型可以调用的工具。由 [大模型（LLM）](/fundamentals/llm-basics/what-is-llm.md) 依据当前获得的信息，在循环中自主决定“下一步是继续查其他关键词、换一个数据库，还是直接给出答案”。

![Agentic RAG 架构封面图：传统线性管道与模型驱动控制环的演进对比](/images/content-rebuild/065cf69a4a0c0f5b/cover.png)

---

## 传统 RAG 的单向流水线局限

基础 RAG（Naive RAG）以及很多高级 RAG（Advanced RAG）系统，在程序结构上都是**单向执行流水线**。

标准流程通常包含四个串行步骤：接收查询 $\to$ 文本向量化与 Top-K 召回 $\to$ 重排（Rerank） $\to$ 上下文拼接并调用模型生成答复。这种架构在处理单一、明确的事实检索时高效且稳定，但在复杂场景下会面临三个明显的工程瓶颈：

1. **查询表达与知识分散的断裂**：复杂任务往往需要分步推理。用户的一句提问可能隐式依赖多个前置事实（例如必须先查明“购买了哪款产品”，才能去查“该产品的技术参数”）。单向系统使用原始提问直接做向量相似度计算，很容易被提问中字面占比更大的词带偏。
2. **中间过程缺乏回溯重试机制**：流水线的每个节点都是单向向下传递的。如果初次检索召回的内容质量较差或完全不相关，重排模块只能在不相关的文档中挑选相对较好的一批，最终迫使大模型在缺少关键上下文的前提下强行回答，容易引发漏答或幻觉。
3. **对初次召回率的高度敏感**：最终生成的准确率严重受制于检索阶段是否成功命中了目标切片。如果切分时上下文丢失，或向量相似度没有排进前几名，后续生成步骤就失去了事实依托。

![传统 RAG 单向拓扑中的不可逆执行断裂与召回瓶颈](/images/content-rebuild/065cf69a4a0c0f5b/fig-1.png)

<ReadMoreLock />

---

## 机制解析：检索作为工具与控制反转

Agentic RAG 的核心变化在于**控制反转**（Inversion of Control）：不再由主流程代码决定何时检索，而是把检索能力下沉为外部工具，由大模型根据对话状态动态发起调用。

在实现上，系统将向量库检索、关键词全文检索、SQL 数据库查询等接口封装为标准的函数签名（Tool Specification）暴露给模型：

```python
# 检索工具定义示例（伪代码，展示接口约定）
def search_company_erp(query: str) -> str:
    """查询企业 ERP 系统中的采购合同、订单流水与采购审批记录。"""
    ...

def search_tech_docs(query: str, version: str = "") -> str:
    """查询技术架构文档、软件产品规格说明书以及版本限制说明。"""
    ...
```

在大模型的多轮调用循环中，状态不断追加累加：

$$\text{State}_t = \{ \text{Query},\, (\text{Action}_1, \text{Observation}_1),\, \dots,\, (\text{Action}_{t-1}, \text{Observation}_{t-1}) \}$$

在第 $t$ 步，模型接收包含历史行动与环境返回结果的完整上下文 $\text{State}_t$，输出下一步决策 $\text{Step}_t$：

* **继续调用工具**：输出工具名称与具体参数（例如发现还没查到版本号，继续调用合同检索）；
* **终止循环并作答**：模型判断已收集到的信息足以回答用户问题，输出最终自然语言文本。

![Agentic RAG 的工具封装与观察反馈循环拓扑](/images/content-rebuild/065cf69a4a0c0f5b/fig-2.png)

在这种模式下，检索变成了可以反复触发、多源互补的探索过程。如果第一次检索返回的内容不相关，模型可以在下一步修改关键词重新检索；如果一个问题需要跨越不同的存储系统，模型可以分步骤先后调用不同的专用接口。

---

## 概念辨析：确定性工作流 vs 模型自主循环

在实践中，许多开发者常把“带有条件分支或意图路由的 RAG”误认为是 Agentic RAG。参考 Anthropic 在《Building effective agents》中的架构区分，我们应当明确划分**工作流（Workflow）**与**自主智能体（Agent）**：

| 维度 | 预定义工作流（Workflow / Advanced RAG） | 模型自主循环（Agentic RAG） |
| :--- | :--- | :--- |
| **流程控制权** | **预定义代码逻辑**：编排拓扑、分支条件与执行规则由代码显式控制 | **大模型自身**：何时调用工具、调用几次由模型根据上下文动态推演 |
| **执行路径** | 代码编排拓扑：程序预先定义状态转移与分支流向，执行范围受代码约束 | 动态状态机：运行时根据中间检索结果自适应推演下一步动作 |
| **循环与退出** | 代码显式控制：支持分支或评估优化循环，退出条件由显式规则或计次阈值决定 | 模型自适应退出：由模型在循环中判断信息是否完备并主动发出终止指令 |
| **失败重试策略** | 由工程师在代码中显式编写 Fallback 降级规则或重试策略 | 模型能够看到工具执行的错误反馈，并在下一轮尝试调整关键词重试 |
| **审计与复现** | 拓扑可审计：执行逻辑与分支规则明确（但若节点内含模型推理或外部检索，输出仍可能波动） | 动态推演：相同输入在多轮调用中可能因模型采样产生不同调用轨迹 |

下面两段简化代码清晰展示了两者的实现差异：

```python
# 1. 确定性工作流：代码掌控分支与检索时机（即使引入了 LLM 做意图识别）
def workflow_rag(user_query: str) -> str:
    # 模型仅用作静态分类器
    intent = classify_intent_with_llm(user_query)
    
    if intent == "contract":
        docs = search_contract_db(user_query)
    elif intent == "technical":
        docs = search_tech_docs(user_query)
    else:
        docs = search_general_wiki(user_query)
        
    # 单次拼接后输出，无法根据 docs 的内容决定是否补充检索
    return generate_answer(user_query, docs)


# 2. Agentic RAG：模型掌控循环，自主决定查什么、查几次、何时停
def agentic_rag(user_query: str, tools: list, max_steps: int = 5) -> str:
    messages = [{"role": "user", "content": user_query}]
    
    for step in range(max_steps):
        # 模型自主决定是返回最终答案，还是输出 tool_calls 指令
        response = call_llm(messages=messages, tools=tools)
        
        if not response.tool_calls:
            return response.content  # 模型判断信息充分，主动终止
            
        for tool_call in response.tool_calls:
            tool_result = execute_tool(tool_call.name, tool_call.arguments)
            # 将工具返回结果回填至上下文，作为下一次思考的输入
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})
            
    return "已达到最大检索步数限制，未能完全解决问题。"
```

单纯在代码中加入 `if-else` 分支路由或意图分类，构成的是由代码固定流转路径的工作流系统；只有当系统将检索控制权下沉为工具，由大模型在循环中根据中间返回的观测结果动态调整后续检索行为时，系统才具备典型的 Agentic 特征。

![确定性工作流的静态分支与自主 Agent 动态循环的机制对比](/images/content-rebuild/065cf69a4a0c0f5b/fig-3.png)

---

## 贯穿案例追踪：多跳查询的状态推演

回到开头的多跳问题，我们通过具体的状态演进展现实时执行过程：

* **输入**：“我司在 2023 年采购了哪个版本的向量数据库？该版本支持的最大向量维度是多少？”
* **可用工具**：`search_company_erp`（查合同流水）与 `search_tech_docs`（查产品手册）。

### 执行链路分解

```text
[第 0 步：初始化输入]
上下文：用户问题
模型判断：这是一个包含依赖关系的复合问题。首要任务是查出 2023 年采购的具体产品和版本。
发出动作：调用 search_company_erp(query="2023 采购 向量数据库 合同")

[第 1 步：接收工具结果]
工具返回：“合同编号 PO-202308-01：采购 Qdrant Enterprise Edition v1.4.0 商业许可。”
模型判断：第一个事实已落实（产品为 Qdrant，版本为 v1.4.0）。下一步需要查询该版本支持的最大向量维度。
发出动作：调用 search_tech_docs(query="maximum vector dimension limit", version="v1.4.0")

[第 2 步：接收工具结果]
工具返回：“Qdrant v1.4.0 在默认配置下，单个向量支持的最大维度为 65,536 维。”
模型判断：两个子问题的信息均已完备，无需进一步调用工具。
发出动作：组织语言，向用户输出最终回答。

[第 3 步：生成答复]
输出：“我司在 2023 年 8 月采购的向量数据库为 Qdrant Enterprise Edition v1.4.0。该版本单向量支持的最大维度为 65,536 维。”
```

![多跳查询在工具循环中的两阶段状态演进与上下文累积示意](/images/content-rebuild/065cf69a4a0c0f5b/fig-4.png)

在单向 RAG 中，如果用原始问题直接计算向量匹配，向量检索很可能被“最大向量维度”这种高频技术词主导，直接返回通用的技术规格，却完全漏掉了采购记录。Agentic 循环将大任务解耦为按需依赖的连续子动作，从机制上解决了多源跨文档的信息拼装问题。

关于多步查询拆解与查询改写的具体实现模式，可参考专题后续内容：[/frontier/agentic-rag/multi-step-retrieval.md](/frontier/agentic-rag/multi-step-retrieval.md) 与 [/frontier/agentic-rag/complex-query-planning.md](/frontier/agentic-rag/complex-query-planning.md)。

---

## 工程代价与失效面分析

Agentic RAG 虽然灵活性更高，但在工程落地时伴随着显著的资源消耗与控制风险。它不是默认的通用解法，上线前必须评估以下工程代价：

### 1. 串行延迟累加

传统 RAG 的执行时间几乎由固定的流水线决定：

$$T_{\text{Traditional}} = T_{\text{embed}} + T_{\text{search}} + T_{\text{rerank}} + T_{\text{llm}}$$

端到端延迟通常在 1 秒以内。

而在 Agentic RAG 中，每一次工具调用都构成一次完整的网络往返与模型推理周期。完成一个需要 $K$ 轮交互的任务，其总耗时为：

$$T_{\text{Agent}} = \sum_{k=1}^{K} \left( T_{\text{prefill}}^{(k)} + T_{\text{tool\_gen}}^{(k)} + T_{\text{exec}}^{(k)} \right) + T_{\text{final}}$$

由于每一轮的输入都依赖上一轮工具执行的结果，各轮之间无法并行。如果一次多跳问答需要 3 轮交互，每次模型调用耗时 1.5 秒，整体响应时间将直接上升至 5 秒以上，在要求亚秒级交互的在线对话场景中难以接受。

### 2. [Token 成本](/fundamentals/token-and-tokenizer/what-is-token.md) 阶梯式膨胀

在 Agentic 循环中，每一轮的历史操作与工具返回的大段文本都会被累加进 Prompt 中：

* 第 1 轮模型输入：基础指令 + 原始提问；
* 第 2 轮模型输入：基础指令 + 原始提问 + 工具 1 调用参数 + **工具 1 返回的全部文档片段**；
* 第 3 轮模型输入：基础指令 + 原始提问 + 工具 1 记录 + 工具 1 文档 + 工具 2 记录 + **工具 2 返回的全部文档片段**。

单步输入的上下文长度随着步数线性增加。在未启用或未命中前缀缓存（Prompt Caching）时，整个会话所处理的累计 Prefill Token 呈等差数列阶梯式累加（$\sum_{i=1}^{K} i \approx O(K^2)$）。即使服务支持前缀缓存，多轮交互中持续累积的历史检索文档也会显著增加显存占用与计费负担，处理一个复杂问题消耗的资源往往是单向 RAG 的数倍到数十倍。

### 3. 不可控发散与死循环

当遇到知识库中完全未收录的问题时，若传统 RAG 配置了相似度截断规则，可在相似度过低时直接拒答；而模型驱动的 Agent 则可能陷入无意义的试探：

* **换词重试死循环**：模型在检索无果后，不断微调关键词重复调用检索工具，直到耗尽程序设定的最大步数上限；
* **噪声误导与语义漂移**：如果前置某一步检索到了具有同名歧义的噪声实体，模型可能会基于错误线索继续向下追踪，越查越偏，最终基于拼凑的无关上下文生成严重偏离事实的虚假答复。

![Agentic RAG 系统运行时的延迟级联与发散失效分析](/images/content-rebuild/065cf69a4a0c0f5b/fig-5.png)

因此，在实际工程实现中，**不能依赖模型自觉终止**，必须在宿主程序层增加强制的安全约束：设置单次调用的最大步数（`max_steps`）、请求总超时时间、工具单次返回内容的长度截断，以及针对重复查询参数的幂等与死循环拦截检测。

---

## 技术选型判定矩阵

在实际架构选型中，应当根据业务的具体特征，自底向上评估所需的控制复杂度：

| 选型考量维度 | 传统 Naive / Advanced RAG | 确定性工作流（Workflow DAG） | 模型驱动 Agentic RAG |
| :--- | :--- | :--- | :--- |
| **适用业务形态** | FAQ 知识问答、单点产品手册查询、在线客服即时建议 | 结构化政务审批分流、表单自动校验填报、跨部门固定路由 | 深度行业研报分析、未知系统故障根因排查、多层数据调查 |
| **查询复杂度** | 语义直白，单跳事实即可回答 | 意图清晰，分支明确但流程固定 | 不确定性强，必须根据中间线索发现未知条件 |
| **端到端延迟预算** | 极严格（< 1.5 秒） | 中等（1.5 秒 - 3 秒） | 宽松（> 5 秒，甚至支持后台异步任务） |
| **可审计性与确定性** | 高，处理流水线固定，单次检索与生成节点易于独立排查 | 较高，执行拓扑与分支规则由代码固定，节点埋点清晰（但涉及模型推理时输出仍受采样影响） | 较低，模型自主决定调用步骤与参数，每次推演的链路轨迹可能存在波动 |
| **推荐技术栈** | 混合检索（向量 + BM25）+ 交叉编码重排 | LangGraph（固定 DAG）/ 原生代码状态机 | LangGraph（动态状态循环）/ LlamaIndex 智能体组件 |

![架构选型四维雷达图与工程决策边界](/images/content-rebuild/065cf69a4a0c0f5b/fig-6.png)

### 落地建议

1. **首选简单架构**：在绝大多数业务场景中，优先通过优化文档切分粒度、引入混合检索以及配置重排模型来解决检索质量问题，不要过早引入动态循环。
2. **以确定性工作流作为过渡**：如果业务逻辑涉及明确的多分支（如根据用户身份或咨询类型分发至不同库），优先使用硬编码代码或确定性工作流图（DAG）实现分支路由，保持系统时延与行为的可控性。
3. **审慎使用自主循环**：仅在面对分析型长任务、多跳关联调查、且业务对响应延迟有较高容忍度（或采用异步任务推送模式）时，才建议采用 Agentic RAG。

---

## 专题导航与延伸阅读

* **下一步进阶**：掌握基本区别后，如何具体实现检索结果反思与查询重写？请阅读第二讲：[/frontier/agentic-rag/multi-step-retrieval.md](/frontier/agentic-rag/multi-step-retrieval.md)。
* **宏观编排拓展**：面对多种异构工具时，如何进行子任务分解与规划？请参考第三讲：[/frontier/agentic-rag/complex-query-planning.md](/frontier/agentic-rag/complex-query-planning.md)。
* **专题主页**：返回栏目全景与知识体系：[/frontier/agentic-rag/README.md](/frontier/agentic-rag/README.md)。
* **面试口述准备**：查看本主题的面试核心答题要点与追问剖析：[/interview/agent-interview/agentic-rag.md](/interview/agent-interview/agentic-rag.md)。
