---
title: RAG、Agent、Workflow 在系统设计中分别适合解决什么问题？
icon: mdi:compare-horizontal
tag:
- 系统设计面试
- RAG
- Agent
- Workflow
description: 系统梳理 RAG、Workflow 与 Agent 的核心差异、适用边界与复合架构设计，提供面试口述框架、工程实现代码与指标观测体系。
---

在系统设计面试中，“这个场景该选 RAG、Workflow 还是 Agent”是考察频率极高的问题。很多候选人容易把三者看作简单的“初级、中级、高级”递进关系，误以为 Agent 必然比 Workflow 更先进，或者认为接入大模型就必须构建 Agent。面试官考察的核心，并非单纯记忆框架名词，而是你能否结合业务约束——**业务规则确定性、合规审计要求、端到端响应延迟与调用成本**——给出清晰的技术权衡依据。

![架构选型总览：Workflow、RAG 与 Agent 在控制流主导权与自主性轴线上的技术定位](/images/content-rebuild/bd21b2f18e8a0abe/cover.png)

---

## 面试标准回答：控制流主导权、状态转移与资源开销

如果在面试前两分钟需要给出简洁透彻的结论，可以这样组织表达：

> “RAG、Workflow 与 Agent 的核心区别在于**控制流的主导权归属**与**状态转移的确定性**。
>
> 1. **Workflow** 的控制流由**确定性代码（有向无环图 DAG 或状态机）**主导。节点内部可以调用大模型处理非结构化文本，但节点之间的跳转条件、异常分支和重试逻辑完全由硬编码规则控制。系统行为可预测、可单步审计，适合业务规则严密、有明确标准作业程序（SOP）的场景。
> 2. **RAG** 是**定向外挂的事实检索与上下文注入管道**。它的核心职责是解决模型参数知识陈旧和私有数据缺乏的问题。控制流通常是线性的‘检索-拼接-生成’。它本身是一个只读的信息获取方案，不具备主动改变外部环境状态的能力。
> 3. **Agent** 的控制流由**大模型自主驱动**。模型在‘思考-行动-观察’循环中动态决定下一步调用什么工具、传入什么参数，以及何时终止。Agent 具备很强的探索灵活性，但代价是多轮调用的延迟叠加、Token 消耗增加以及不可完全杜绝的决策偏差。
>
> 在实际系统设计中，最稳妥的工程实践是以 Workflow 作为主干骨架约束业务边界，在确定性信息获取节点挂载 RAG，仅在需要开放式探索或长尾排查的局部子图中引入受限步数的 Agent。”

![三类架构控制流主导权对比：硬编码状态机、单向事实检索注入与模型自省决策循环](/images/content-rebuild/bd21b2f18e8a0abe/fig-1.png)

为了在面试中更全面地横向对比三者，可整理出如下技术特征维度：

| 维度 | Workflow（工作流） | 基础 RAG（检索增强） | Agent（自主智能体） |
| :--- | :--- | :--- | :--- |
| **控制流主导者** | 预定义代码 / 规则状态机 | 固定检索与生成流水线 | 大模型运行时的推理规划 |
| **状态转移确定性** | 规则受控（跳转逻辑与候选分支由代码显式预定义） | 拓扑固定（检索与提示词拼接路径预定义） | 动态开放（由模型每次输出动态决定下一步） |
| **端到端延迟（P99）** | 稳定受控（取决于串并联节点耗时） | 相对稳定（检索耗时 + 单次或两次模型调用） | 波动大且长尾严重（依赖多轮工具交互循环） |
| **Token / 计算成本** | 随固定步骤线性增长，预算可控 | 受检索切片长度影响，单次请求开销可测 | 随探索轮数显著增加，可能产生多余调用 |
| **审计与调试定位** | 容易（每个节点的输入输出均可记录重放） | 容易（检索召回切片与生成结果直接对应） | 复杂（执行路径动态生成，复现排查成本高） |
| **典型业务场景** | 订单审批、退货退款 SOP、合同抽取 | 内部制度问答、技术文档检索、产品手册查询 | 复杂日志排查、代码动态调试、未知数据探查 |

<ReadMoreLock />

---

## 贯穿实例：电商售后系统的三种任务形态

为了清晰理解三者的适用场景，以一个“电商售后处理系统”为例：

1. **场景 A：买家询问“大家电退货的运费由谁承担？退货时效是几天？”**
   - **选择 RAG**：这是典型的非结构化事实问答。系统通过混合检索在售后政策库中召回对应条款，拼入提示词，让模型据此整理回答。流程单向只读，不需要修改数据库或触发退款。
2. **场景 B：买家提交“商品破损退款申请，要求退款 200 元”**
   - **选择 Workflow**：涉及资金变动与风控，绝不能交由模型自由发挥。系统由硬编码代码控制：先调接口读取订单状态；若订单处于“已签收且在 7 天内”，且退款金额小于阈值，则自动流转至商家退款接口；若金额超出阈值，分支走向人工审核节点。大模型在其中只负责提取买家留言里的损坏原因并打标，不决定资金流向。
3. **场景 C：物流异常长尾排查“包裹显示已被某驿站签收，但买家表示未收到，物流轨迹在三天前停滞于某个中转仓”**
   - **选择 Agent**：排查路径在运行前无法预知。系统需要给模型提供“查快递底单”、“查中转仓监控日志”、“调驿站入库记录”、“发短信核实”等工具。模型先查中转仓状态，若发现滞留，再查该网点的异常报备接口；根据返回结果决定下一步动作。每一步查什么，完全取决于上一步返回了什么。

---

## RAG 适用场景与边界：只读事实检索与生成

### 核心机制与收益

RAG（检索增强生成）的基本运行范式是“检索器检索上下文 + 提示词注入 + 单次生成”。在离线阶段，系统将业务文档切分成文本块（Chunk），通过嵌入模型转为向量并存入向量数据库，或同时构建 BM25 倒排索引；在线阶段，用户 Query 经改写后执行检索，通过加权倒排融合（如 Reciprocal Rank Fusion, RRF）等方式综合打分召回相关片段，再送给模型合成最终回答。关于多源知识检索的设计细节，可参考 [/interview/ai-system-design-interview/knowledge-base-qa.md](/interview/ai-system-design-interview/knowledge-base-qa.md)。

它的核心工程价值是**将事实数据从模型参数中解耦**：企业无需承担高昂的预训练或频繁微调成本，只要更新文档库，大模型就能基于最新事实作答；并且回答中的关键内容可以附带源文档切片来源，方便人类复核。

![RAG 检索增强生成流水线与失效边界：切分、混合检索与单向合成中的断裂点](/images/content-rebuild/bd21b2f18e8a0abe/fig-2.png)

### 边界与局限

在面试中需要准确指出 RAG 的设计边界：
1. **只读性与无副作用**：标准 RAG 是只读查询系统，不能修改业务系统状态。如果任务包含“调接口发货”或“修改库存”，RAG 无法独立完成。
2. **检索质量决定生成上限**：如果召回的 Top-K 文档包含过时信息、矛盾数据或噪声，大模型会在有偏差的上下文上给出看似合理的错误推论（基于受污染上下文的幻觉）。
3. **多跳逻辑推理困难**：如果一个问题需要先查到实体 A 的属性 X，再以 X 为参数去检索实体 B 才能得出结论，单轮检索往往无法一步召回所有必要事实。此时往往需要引入多阶段查询规划或受限的 Agentic RAG（详见 [/frontier/agentic-rag/complex-query-planning.md](/frontier/agentic-rag/complex-query-planning.md)）。

---

## Workflow 适用场景与边界：固定拓扑、状态可控与合规审计

### 核心机制与收益

根据主流工程实践与编排框架（如 LangGraph、LlamaIndex Workflows）的定义，Workflow 是**通过代码或状态机预定义任务执行拓扑与路由规则的编排系统**。其执行路径通常表现为有向无环图（DAG）或带条件跳转的有限状态机。

在 Workflow 中，大语言模型通常充当“语义转换算子”或“结构化解析器”。例如：节点 A 负责调用模型将混乱的用户反馈解析为符合 JSON 规范的工单对象；节点 B 则是纯代码条件判断：`if ticket.category == 'refund' and ticket.amount > 500`，根据计算结果跳转到不同下游节点。

![Workflow 确定性状态机与可审计执行流：明确的分支路由、状态持久化与审计记录](/images/content-rebuild/bd21b2f18e8a0abe/fig-3.png)

Workflow 的核心收益包括：
- **确定性服务等级（SLA）**：分支路径和最多调用的模型步数在代码中写明，耗时与成本可预测，P99 延迟波动范围较小。
- **完备的审计与复现能力**：每个节点的入参、出参和状态变更都可写入数据库日志。金融风控、医疗合规等领域需要向监管团队解释“系统在某一步因什么条件判定拒绝”，Workflow 能完整重现当时的决策链路。
- **故障隔离与局部重试**：某个 LLM 节点解析 JSON 失败或超时，系统只需局部重试该节点，或走预先设定的降级规则，无需中断整个业务流。

### 边界与局限

Workflow 的痛点在于**无法应对规则分支组合爆炸**。如果业务场景没有固定的标准处理路径，用户输入意图发散，或者排查流程中的中间状态多达上百种且无法预先枚举，硬编码的工作流就必须编写极其繁琐的分支条件，导致代码维护成本急剧升高。

---

## Agent 适用场景与边界：动态探索、多轮试错与失控风险

### 核心机制与收益

Agent 的核心是**将控制流交给大模型的推理循环**。系统通常遵循 ReAct（Reasoning + Acting）模式：
1. 模型接收任务目标、上下文和当前环境的工具调用返回结果（Observation）；
2. 模型内部生成思考过程（Thought），并决定调用哪一个工具以及传入什么参数（Action）；
3. 宿主程序执行该工具并将结果返回给上下文；
4. 模型根据工具返回继续判断任务是否完成，未完成则继续迭代。

工具平台的鉴权与沙箱隔离设计可参考 [/interview/ai-system-design-interview/agent-tool-platform.md](/interview/ai-system-design-interview/agent-tool-platform.md)。

![Agent 推理-行动-观察动态循环与长尾失控风险：步数增加导致的延迟级联与死循环](/images/content-rebuild/bd21b2f18e8a0abe/fig-4.png)

Agent 适合解决**探索路径未知、强依赖多轮动态反馈**的问题。例如自动化运维排查，工程师面对线上报警通常无法预设固定流程，必须先看监控面板，发现 CPU 飙高后再登录服务器查看慢查询日志，再根据日志定位具体的 SQL 语句。这种根据返回动态推导下一步的动作，正是 Agent 适用的场景。

### 生产落地中的风险点

在面试中不仅要讲 Agent 的优势，更要主动指出它在生产环境中的工程风险：
- **延迟与成本线性叠加**：设大模型单次前向推理与工具执行的串行耗时为 $T$，若 Agent 自主循环迭代了 $N$ 轮，端到端延迟约为 $N \times T$。当 $N=5$ 时，总延迟常达到 10~20 秒，在线实时交互体验极差。
- **死循环与状态震荡**：当某个工具报错或返回非预期数据时，模型容易陷入“报错 -> 重新传入相同参数重试 -> 再次报错”的循环，在有限步数内耗尽 Token 预算。
- **不可逆操作的安全隐患**：让大模型自主决定写操作（如数据库删除、资金转账、批量发送邮件）存在误操作或受到 Prompt Injection 攻击的风险。工程上通常禁止开放式 Agent 直接执行不可逆的高危写入动作。

---

## 复合架构实践：以 Workflow 为骨架，挂载 RAG 与受限 Agent

在实际企业级系统设计中，极少出现脱离工程框架的“纯自主 Agent”，主流方案是**复合层级架构**：**以 Workflow 状态机为全局骨架保证业务可控性，在需要事实依据的节点嵌入 RAG，在局部不确定的探索节点嵌入带最大步数熔断的受限 Agent。**

![生产级复合架构模式：以 Workflow 状态机为全局骨架，内嵌 RAG 检索子节点与受限 Agent 沙箱](/images/content-rebuild/bd21b2f18e8a0abe/fig-5.png)

### 状态图实现示例

以下以类似 LangGraph 的状态图伪代码，展示一个电商客服的复合系统：主干是确定性的路由状态机，事实类问题走 RAG，资金与订单查询走固定业务代码，异常排查走受限 Agent。

```python
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END

# 1. 显式定义全局状态结构，便于持久化与全链路审计
class SupportState(TypedDict):
    user_id: str
    query: str
    intent: Literal["faq", "order_query", "investigation", "unknown"]
    rag_evidence: list[str]
    agent_steps: int
    final_response: str

# 2. 意图路由节点（属于 Workflow 确定性骨架）
def router_node(state: SupportState) -> dict:
    q = state["query"].lower()
    if any(k in q for k in ["运费", "时效", "退换规则"]):
        return {"intent": "faq"}
    elif any(k in q for k in ["查订单", "改地址", "物流状态"]):
        return {"intent": "order_query"}
    else:
        return {"intent": "investigation"}

# 3. RAG 知识检索节点：单向生成，无环境修改副作用
def rag_node(state: SupportState) -> dict:
    # 模拟检索模块：混合检索召回文档切片
    docs = ["《平台大家电售后细则》：签收 7 天内无理由退货，非质量问题买家承担运费。"]
    # 调用大模型基于召回切片组织语言
    reply = f"根据平台售后政策：{docs[0]}"
    return {"rag_evidence": docs, "final_response": reply}

# 4. 确定性业务节点：涉及用户核心数据，完全由业务代码控制
def order_service_node(state: SupportState) -> dict:
    # 真实场景中调用内部 RPC 接口
    return {"final_response": "您的订单已由顺丰发货，预计今日 18:00 送达。"}

# 5. 受限 Agent 节点：允许探索排查，但必须配置最大步数硬熔断
def bounded_agent_node(state: SupportState) -> dict:
    current_step = state.get("agent_steps", 0)
    MAX_STEPS = 3  # 硬性预算上限，防止死循环
    
    if current_step >= MAX_STEPS:
        return {"final_response": "该问题链路较长，已为您自动转接资深专员跟进。"}
    
    # 模拟 Agent 多步分析：递增步数计数器
    return {
        "agent_steps": current_step + 1,
        "final_response": "已排查中转站滞留记录，发现受暴雨影响，已协调优先发车。"
    }

# 6. 编排全局状态图
builder = StateGraph(SupportState)
builder.add_node("router", router_node)
builder.add_node("rag_node", rag_node)
builder.add_node("order_node", order_service_node)
builder.add_node("agent_node", bounded_agent_node)

builder.add_edge(START, "router")

def route_branch(state: SupportState) -> str:
    if state["intent"] == "faq":
        return "rag_node"
    elif state["intent"] == "order_query":
        return "order_node"
    return "agent_node"

builder.add_conditional_edges("router", route_branch)
builder.add_edge("rag_node", END)
builder.add_edge("order_node", END)
builder.add_edge("agent_node", END)

app = builder.compile()
```

这种设计将安全与成本控制放在外层：敏感操作留在确定性业务逻辑中；只读问答走 RAG 管道；非结构化长尾排查交给受限步数的 Agent 沙箱。

---

## 面试深度追问与应对策略

![架构演进决策树与质量评测关卡：从纯 Prompt 到 RAG、Workflow 与 Agent 的分流判断指标看板](/images/content-rebuild/bd21b2f18e8a0abe/fig-6.png)

关于日志监控与链路追踪的具体方案，可参考 [/interview/ai-system-design-interview/logging-monitoring.md](/interview/ai-system-design-interview/logging-monitoring.md)。

### 追问 1：大模型上下文窗口已经达到 1M 甚至 2M Token，为什么还要做 RAG，而不是把所有文档直接塞给模型？

**应对要点**：
1. **首字延迟（TTFT）与计算成本**：单次请求输入 1M Token 会带来高昂的 Prefill 计算耗时（常达数十秒），商业 API 亦按输入 Token 线性计费。若请求并发量较大，全量喂入长文本会导致端到端延迟与 Token 账单大幅上升。
2. **注意力随长度稀释（Lost in the Middle 现象）**：实测表明，当上下文极长时，大模型对位于中段区域细节的检索精度会明显下降，整体召回效果通常不如通过精确切片检索后注入核心上下文。
3. **动态权限与数据更新**：企业内部文档常伴随按行、按部门的权限过滤（Row-level Security）。RAG 可以在向量检索阶段利用元数据（Metadata Filter）快速过滤无权限切片；如果直接将整份文档塞入模型，很难在不重新处理提示词的情况下做到实时、细粒度的权限隔离。

### 追问 2：什么时候需要从基础单轮 RAG 演进到 Agentic RAG？

**应对要点**：
- **业务痛点**：当评测集包含大量跨文档的多跳问题（例如：“对比 A 部门与 B 部门去年差旅报销标准的区别”），单次检索无论取 Top-3 还是 Top-10 均无法同时精准覆盖两个实体的条件。
- **演进策略**：引入查询分解（Query Decomposition）与自检反思（Self-Reflection）机制。让模型先拆解出“A 部门标准”和“B 部门标准”两条子查询，分别执行检索；若检索回来的内容不足以回答，允许模型自动调整关键词再搜一次。注意：此过程必须设定最大重试次数，防止无限循环。

### 追问 3：生产上线这三类系统，分别应该关注哪些核心监控指标？

**应对要点**：
- **RAG 验收指标**：
  - 检索阶段：Hit Rate@K（命中文档比率）、MRR（平均倒数排名）。
  - 生成阶段：忠实度（Faithfulness，评估回答是否违背召回切片）、回答相关性（Answer Relevance）。
- **Workflow 验收指标**：
  - 节点异常率：LLM 结构化提取失败率（Schema Validation Error Rate）。
  - 性能与业务：全链路 P95 / P99 延迟稳定性、各业务分支路由命中比例。
- **Agent 验收指标**：
  - 工具调用准确率：选错工具的比例、参数解析错误率。
  - 资源与收敛：平均任务执行步数、步数达到上限触发熔断的比例（Max Steps Breached Rate）、端到端任务成功率。

---

## 常见误区与避坑指南

| 常见错误说法 | 错误原因与技术事实 | 推荐的严谨表述 |
| :--- | :--- | :--- |
| “Agent 比 Workflow 更高级，是未来系统演进的必然方向。” | 选型取决于业务约束与合规审计需求。强监管或涉及资产变动的核心场景要求流程每一步都有据可查且跳转逻辑严格可控，代码主导的 Workflow 能提供明确的分支边界与完整的审计日志。 | “Workflow 强调确定性与可控性，Agent 强调开放环境下的探索能力。工程上通常以 Workflow 为主干骨架，受控引入 Agent。” |
| “RAG 能保证 100% 消除大模型的所有事实幻觉。” | RAG 会受到检索切片断章取义、多篇文档内容冲突、检索未命中等问题影响。一旦注入了受污染的上下文，模型仍会产生错误推论。 | “RAG 能显著缓解无据可依的幻觉，并提供可溯源的参考来源，但仍需搭配过滤策略与答案校验机制降低错误概率。” |
| “给 Agent 配备了外部工具，就天然能保证任务执行的可靠性。” | 工具执行需要程序侧提供重试机制、超时熔断、参数校验与幂等保证。模型无法自主解决网络丢包或下游服务不可用的问题。 | “Agent 的工具调用必须在代码层实现幂等性检查、输入校验与事务保护，防止模型在循环重试中重复提交非幂等请求。” |
