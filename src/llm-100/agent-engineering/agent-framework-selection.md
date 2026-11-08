---
title: LangChain、LangGraph、CrewAI 怎么选？从机制差异到场景落地
icon: mdi:book-open-page-variant
description: 从电商退款贯穿示例出发，剖析三大框架的控制流、状态管理与容错机制差异，给出明确选型依据与可验证代码。
---

评估智能体（Agent）框架时，关键不在于“几行代码能跑通 Demo”，而在于业务逻辑变复杂时，系统能不能稳定执行、中间状态能否保存、是否允许人工介入，以及能否精准控制模型何时被调用。

本讲通过一个“电商售后退款审核”的贯穿案例，拆解 LangChain、LangGraph 与 CrewAI 的设计差异，厘清它们的机制边界，最后提供一套包含断点续跑和人工介入的可验证代码。

![Agent 框架选型全局视角与核心定位对比](/images/content-rebuild/6c56d83b8934b92d/cover.png)

### 贯穿业务场景：电商售后退款审核
假设我们要开发一个售后退款处理系统，业务规则如下：
1. 输入：订单号、退款金额、用户填写的申请理由。
2. 规则前置校验：如果退款金额超过 500 元，强制要求人工客服复核；若金额不超过 500 元，进入系统自动审核。
3. 非结构化语义判定：自动审核阶段调用大模型，分析用户理由是否包含恶意欺诈或质量争议。
4. 执行动作：审核通过则调用退款 API，被驳回则向用户发送解释说明。
5. 容错与合规要求：人工复核可能需要数小时甚至几天，系统必须能够暂停、持久化保存状态，并在人工审批完成后恢复执行。

在接下来的分析中，我们将看这三大框架分别如何处理这一场景。

## 框架选型的三大工程指标

在选择框架前，我们需要建立客观的评估维度：

![Agent 编排框架的核心工程评估维度](/images/content-rebuild/6c56d83b8934b92d/fig-1.png)

### 1. 控制流透明度（Control Flow Transparency）
控制流是指系统下一步该执行什么操作。常见的控制方式有两种：
* **模型自主循环（Model-Driven Loop）**：由大模型在每一轮推理中通过工具调用（Tool Call）自行决定下一步是继续查数据还是结束。这种方式最灵活，但执行顺序是概率性的，外部代码很难对它的执行路径做硬性约束。
* **代码显式图编排（Graph-Based Orchestration）**：由工程师通过节点（Node）和条件边（Conditional Edge）用代码显式声明拓扑与跳转规则。大模型可以在指定节点中负责内容分析、结构化提取或子循环决策，而关键路由与风险检查点由普通条件语句（如 Python 的 `if/else`）刚性控制，防止模型自主越权跳过审批节点。需要注意，代码控制边能保证路由拓扑的确定性，但外部网络、依赖状态和模型节点内部生成仍具备不确定性。

### 2. 状态持久化与断点续跑（State Persistence）
长时间运行的任务难免会遇到网络中断、服务重启或等待外部审批：
* 状态是存储在显式的数据结构中，还是散落在对象的私有属性里？
* 框架是否支持将每一步的状态保存到数据库（如 PostgreSQL 或 Redis）？
* 系统崩溃后，能否直接从上一个成功的步骤继续，而不需要重新消耗 Token 从第一步跑起？

### 3. 容错与人工介入（Human-in-the-loop）
在涉及资金与合规的关键流程中，系统必须能在特定节点主动挂起（Interrupt），释放计算资源，等待人工核验。人工可以修改上下文参数（例如调整退款金额），确认后再唤醒执行。

<ReadMoreLock />

## LangChain 的定位：高阶组件库与单 Agent 工具循环

LangChain 的核心定位是 **LLM 应用的通用连接器与组件库**。它统一封装了各大模型提供商的接口、提示词模板、文档加载器、向量检索器以及工具定义规范。

![LangChain 工具调用循环与底层黑盒流转机制](/images/content-rebuild/6c56d83b8934b92d/fig-2.png)

### 运行机制：黑盒工具循环
在 LangChain 中，典型的 Agent 结构由大模型与工具列表构成。例如通过早期的 `AgentExecutor` 或封装好的 `create_tool_calling_agent`：

```python
# LangChain 工具调用循环示意（高阶封装）
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是售后客服，可查询物流和订单状态。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 框架接管循环：调用模型 -> 解析 Tool Calls -> 运行 Tool -> 结果拼接回上下文 -> 循环直到模型停止
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "订单 ORD-2026 申请退款 600 元，帮我审核"})
```

### 机制边界与场景适用
* **退款场景表现**：如果要求“金额大于 500 元必须人工审批”，虽然可以在退款工具内部编写硬编码校验逻辑拦截超额请求，但在单一循环中仍必须先经历一轮大模型推理才能触发工具校验，无法在模型调用前用纯代码直接短路分流；更重要的是，当需要挂起等待数小时甚至数天的人工审批时，单一工具循环缺乏原生的步级持久化检查点与图级中断恢复能力，若要强行插入外部人工介入与状态保存，往往需要编写大量侵入式回调，系统维护成本极高。
* **适用场景**：适合**单次任务处理、轻量问答、通用检索增强（RAG）或工具调用较为自由的单体辅助助手**。它提供了丰富的周边生态集成，但在复杂受控流程中，LangChain 官方更推荐将其降维为工具和模型包装器，将控制流交由 LangGraph 处理。

## LangGraph 的定位：基于计算图的状态编排运行时

LangGraph 是 LangChain 团队开源的**底层状态编排运行时**。它的核心架构借鉴了 Google 的 Pregel 图计算模型与状态机理念。

需要明确的是：**使用 LangGraph 并不强制依赖 LangChain**。它的节点就是普通的 Python 函数，可以直接接收原生字典并返回更新字典。

![LangGraph 基于 Pregel 模型的超步与持久化状态机机制](/images/content-rebuild/6c56d83b8934b92d/fig-3.png)

### 运行机制：状态、超步与检查点
1. **状态（State）**：通过 `TypedDict` 或 Pydantic 定义的共享数据结构。所有节点都对这个状态进行读取和增量更新。
2. **超步（Superstep）**：图的每个执行周期称为一个超步。在当前超步中运行节点，更新状态后，通过边决定下一个超步要激活的节点。
3. **混合确定性与概率性（Mix Deterministic & Agentic Steps）**：可以在图中将纯手写 Python 规则（确定性节点）与 LLM 判定（非确定性节点）串联在一起。路由分支由代码决定，杜绝模型越权。
4. **持久化检查点（Checkpointer）**：图在每个超步结束后，都会自动把最新状态写入存储介质。当需要人工介入时，可在指定节点配置中断（如 `interrupt_before` 或 `interrupt()`）挂起流程；外部系统恢复执行时（如向图传入更新状态并调用 `invoke(None)` 或使用 Command 恢复），可以直接从检查点继续往下走。

### 机制验证：用 LangGraph 实现退款审核流程

以下是一个完整可运行的最小实例，演示如何结合“确定性规则路由”、“LLM 判定节点”与“持久化检查点”：

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. 显式定义状态模式
class RefundState(TypedDict):
    order_id: str
    amount: float
    reason: str
    risk_level: str
    decision: str

# 2. 确定性业务规则节点：纯 Python 逻辑，不消耗 Token
def rule_check_node(state: RefundState) -> dict:
    # 业务硬规则：大于 500 元标为高风险
    if state["amount"] > 500.0:
        return {"risk_level": "HIGH"}
    return {"risk_level": "LOW"}

# 3. 模型决策节点：仅在需要分析非结构化理由时调用 LLM
def llm_evaluate_node(state: RefundState) -> dict:
    # 模拟大模型提取理由语义（实际场景可接入任意 LLM API）
    # 模型输出受限于当前节点的职责范围
    if "欺诈" in state["reason"]:
        return {"decision": "REJECTED"}
    return {"decision": "APPROVED"}

# 4. 人工审核准备节点
def human_review_node(state: RefundState) -> dict:
    # 标记当前等待人工处理
    return {"decision": "PENDING_HUMAN"}

# 5. 条件路由逻辑：用纯代码拦截高风险路径
def route_risk(state: RefundState) -> Literal["human_review_node", "llm_evaluate_node"]:
    if state["risk_level"] == "HIGH":
        return "human_review_node"
    return "llm_evaluate_node"

# 6. 构建图拓扑
workflow = StateGraph(RefundState)
workflow.add_node("rule_check_node", rule_check_node)
workflow.add_node("llm_evaluate_node", llm_evaluate_node)
workflow.add_node("human_review_node", human_review_node)

workflow.add_edge(START, "rule_check_node")
workflow.add_conditional_edges("rule_check_node", route_risk)
workflow.add_edge("llm_evaluate_node", END)
workflow.add_edge("human_review_node", END)

# 配置检查点存储器（支持状态持久化）
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

### 机制边界与场景适用
* **退款场景表现**：金额 > 500 元走 `route_risk` 强制流入人工节点，没有任何模型能够越权跳过这一步。状态全部落盘，流程可控性极高。
* **适用场景**：**企业核心业务流程、包含刚性业务规则的审批流、长周期异步任务、需要步级排查与断点恢复的复杂系统**。缺点是开发前期需要显式设计状态模式和节点拓扑，样板代码比单纯写 Prompt 多。

## CrewAI 的定位：基于角色与目标的拟人化协作抽象

CrewAI 采用了截然不同的**拟人化组织管理抽象**。它把智能体系统模拟为一家由不同岗位员工组成的工作团队。

![CrewAI 团队协作模型与任务委托循环机制](/images/content-rebuild/6c56d83b8934b92d/fig-4.png)

### 运行机制：角色、目标与任务委派
CrewAI 的基本抽象单元是：
* **Agent**：定义 `Role`（角色名称，如“合规专员”）、`Goal`（工作目标）和 `Backstory`（背景经历，会被编译进 System Prompt 中）。
* **Task**：具体的任务描述（Description）与预期交付格式（Expected Output），并绑定给具体的 Agent。
* **Crew**：由多个 Agent 和 Task 组成的班组，提供串行（Sequential）或层级（Hierarchical）执行流程。在层级模式下，框架会使用一个 Manager 节点充当领导，负责把任务在各成员之间分发。

### 机制边界与场景适用
* **退款场景表现**：在 CrewAI 中处理退款时，通常需要设定一个“客服初审员”角色和一个“法务复核员”角色，让二者通过语言交互来委派任务。但在需要严格数字比对（金额 > 500）时，纯粹通过 Prompt 驱动的角色委派可能出现不稳定性：角色之间容易产生多轮客套对话，消耗额外 Token；如果两个角色的 Prompt 产生语义歧义，还可能陷入来回推诿的循环。
* **适用场景**：**发散型调研、长文撰写、多角度头脑风暴、竞品分析报告**。在这类任务中，产出结果没有唯一的严格标准，拟人化分工能够充分发挥多提示词协同的创造力。

## 架构特性对比

为了直观展示三者的工程差异，我们将核心维度整理如下：

| 评估维度 | LangChain (Agent 模块) | LangGraph | CrewAI |
| :--- | :--- | :--- | :--- | 
| **核心抽象隐喻** | 工具调用管道 / 辅助执行器 | 有向图 / 状态机运行时 | 拟人化协作团队 (角色/目标/任务) |
| **控制流主导权** | 模型自主决策（Prompt 引导） | **代码显式定义**（节点与条件边） | 模型自主委派（角色 Prompt 引导） |
| **状态管理模型** | 隐式，多保存在消息列表与组件内存中 | **显式强类型状态**（TypedDict / Pydantic） | 基于任务输出文本逐步追加上下文 |
| **确定性规则混合** | 较弱，工具调用与循环紧耦合 | **极强**，节点可为任意原生 Python 函数 | 较弱，主要依赖自然语言指令约束 |
| **步级持久化与恢复** | 需自行在外层实现状态快照 | **原生提供 Checkpointer**，支持步级持久化 | 主要依赖简单的缓存机制，缺乏图级快照 |
| **人工介入挂起** | 需借助回调机制进行中断控制 | **原生支持中断（interrupt）与继续执行** | 支持任务级人工确认，但控制粒度较粗 |
| **典型失效表现** | 陷入反复调用同一工具的循环 | 开发者自定义的状态更新冲突或图分支遗漏 | 角色之间来回推诿、上下文无限膨胀 |

![三大框架在控制力与开发心智模型上的对比拓扑](/images/content-rebuild/6c56d83b8934b92d/fig-5.png)

## 场景落地选择指南

在工程实践中，建议遵循“先简单后复杂、按需演进”的原则：

![企业级 Agent 技术选型演进决策树](/images/content-rebuild/6c56d83b8934b92d/fig-6.png)

### 选型落地建议
1. **首选原生 SDK 或 LangChain 标准组件**：
   * 如果需求只是单轮对话、根据用户输入调用 1~2 个外部 API 获取天气或订单号，用官方 SDK 配合几行简单的逻辑代码即可完成。此时引入图编排或多智能体框架不仅无法带来收益，反而增加部署与学习成本。
2. **发散型创意与综合调研选用 CrewAI**：
   * 如果需求是“输入一个产品名称，由搜索员抓取行业资料、分析师对比优缺点、文案编辑输出排版报告”，选用 CrewAI。这种基于多角色的协作能快速产出发散性成果。
3. **严谨业务流与长周期任务选用 LangGraph**：
   * 如果需求涉及支付、风控、合同审批等有明确规则红线的业务，或者任务跨越数小时且需要人工审批，选用 LangGraph（或自研类似状态机系统，参见 [/llm-100/agent-engineering/agent-state-machine.md](/llm-100/agent-engineering/agent-state-machine.md)）。通过代码显式约束跳转逻辑，才能提供稳定的执行轨迹（参见 [/llm-100/agent-engineering/agent-execution-tracing.md](/llm-100/agent-engineering/agent-execution-tracing.md)）。

深入掌握状态图的节点设计与中断恢复，请继续阅读下一讲：[/llm-100/agent-engineering/langgraph-controlled-workflow.md](/llm-100/agent-engineering/langgraph-controlled-workflow.md)；在面试中回答多智能体架构对比时，可参考 [/interview/agent-interview/agent-frameworks.md](/interview/agent-interview/agent-frameworks.md)；开源生态最新项目索引见 [/resources/agent-framework.md](/resources/agent-framework.md)。

---

## 进阶练习：实现带状态中断与人工审批的退款流程

### 练习目标
使用 LangGraph 构建一个带有“持久化中断”与“人工更新”的退款状态图，验证当系统遇到高风险退款时，能够安全挂起，并在人工修改决策后继续完成执行。

### 练习步骤
1. 在上文的 LangGraph 示例中，为图编译配置中断规则：在进入人工审核或离开人工审核时设置断点。
2. 使用特定的 `thread_id` 触发一次金额为 800 元的高风险退款任务，观察图是否停留在人工审核阶段。
3. 人工将状态中的 `decision` 字段修改为 `"APPROVED_BY_MANAGER"`，再次调用执行，验证图顺利运行至 `END`。

### 参考验证代码
```python
# 验证环境：langgraph >= 0.2.0
config = {"configurable": {"thread_id": "refund_task_001"}}

# 1. 触发高风险退款（金额 800 > 500）
input_data = {
    "order_id": "ORD-999",
    "amount": 800.0,
    "reason": "商品尺码不合适",
    "risk_level": "",
    "decision": ""
}

# 编译时指定在 human_review_node 前挂起
app_with_interrupt = workflow.compile(
    checkpointer=memory,
    interrupt_before=["human_review_node"]
)

# 执行直到遇到断点
app_with_interrupt.invoke(input_data, config=config)

# 检查当前状态（应停在断点处）
current_state = app_with_interrupt.get_state(config)
print("当前中断节点前的状态:", current_state.values)
assert current_state.next == ("human_review_node",), "未在人工节点前正确挂起！"

# 2. 模拟人工介入：更新决策状态
app_with_interrupt.update_state(
    config,
    {"decision": "APPROVED_BY_MANAGER"},
    as_node="human_review_node"
)

# 3. 唤醒图继续执行直到结束
final_state = app_with_interrupt.invoke(None, config=config)
print("最终执行结果:", final_state["decision"])
assert final_state["decision"] == "APPROVED_BY_MANAGER", "状态恢复执行不符合预期！"
print("验证通过：成功实现确定性路由、状态持久化挂起与人工热修改。")
```
