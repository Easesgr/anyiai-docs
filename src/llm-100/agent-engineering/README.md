---
title: Agent 工程怎么做？从工作流到状态编排的系统实践
icon: mdi:cog-outline
description: 厘清确定性工作流与动态智能体的适用边界，掌握混合状态图编排方式，梳理十讲学习路线，并通过可运行代码完成环境验收。
---

![Agent 工程模块导读与全局拓扑](/images/content-rebuild/d9467ed3c22f7f52/cover.png)

在构建大语言模型应用时，很多开发者最先尝试的方式是写一段复杂的提示词，或者用链式调用把几段提示词串联起来。这种做法在处理简单问答时很有效，但面对真实的业务系统，很快就会遇到瓶颈。

例如一个电商的“售后退款与仲裁系统”：用户可能输入“订单 1002 收到时屏幕碎了，要求全额退款并补偿优惠券”。这个任务无法靠一次模型调用可靠完成，因为系统需要依次完成多项工作：从数据库拉取订单明细、校验物流收货时间、评估破损图片凭据、检查用户信用分，并在涉及高额赔付时转交人工审核。

如果仅依赖一段无状态的提示词，模型可能会跳过规则直接承诺赔付，或者在多次调用中丢失之前的上下文。要解决长执行周期、外部系统交互与异常重试问题，系统必须从“无状态的提示词调用”转向“具备明确状态流转与持久化能力的软件系统”。这就是 Agent 工程要解决的核心问题。

---

## 工作流与智能体的工程适用边界

在规划系统架构前，首先要分清两种常见的复合系统模式：**工作流（Workflows）**与**智能体（Agents）**。

根据 Anthropic 在系统设计中的实践划分，两者的核心区别在于**运行时的控制权归属于代码还是归属于大模型**。

![工作流与智能体的控制流与决策拓扑对比](/images/content-rebuild/d9467ed3c22f7f52/fig-1.png)

### 1. 工作流（Workflows）
在工作流模式中，大语言模型与工具通过预先编写的代码路径（Predefined Code Paths）进行编排。执行流由代码逻辑主导，大语言模型作为流程中具体步骤的处理组件（例如提取用户意图、整理文本摘要）。

* **控制流特征**：流向哪个分支、在什么条件下重试或何时终止，由代码规则或状态机预先定义。需要注意，代码控制分支不等于整个流程输出 100% 确定，内部的模型输出、外部依赖状态与网络波动仍会带来动态变化。
* **优势与代价**：执行路径可预测，延迟相对更低，便于单元测试与日志追踪；但在面对非预期的用户输入或长尾需求时，缺乏动态调整的能力。
* **适用场景**：规则明确、业务合规要求高的任务。例如提取发票字段、对固定模板文本做合规质检、固定的两阶段信息检索。

### 2. 智能体（Agents）
在智能体模式中，代码为模型提供一组可用工具（如查询接口、计算器、检索器）和一个目标，由模型在运行循环中自主决定下一步行动。

* **控制流特征**：模型根据当前的对话历史和中间执行结果，自主决定是继续调用工具、更换工具，还是给出最终回复。
* **优势与代价**：能灵活应对复杂、未知的长尾问题；但多轮循环会显著增加 API 调用延迟与成本，且存在偶发性的死循环、工具参数幻觉或偏离目标的风险。
* **适用场景**：无法预先穷举所有执行分支的任务。例如开放式的代码排错、长程多信源调研、需要反复尝试不同解题路径的数据分析。

### 3. 生产落地的混合形态：状态图编排
在实际生产中，极少有团队会构建完全不受控的纯自主智能体。主流的工程做法是将两者结合：**用确定性的状态图构建系统主干，仅在需要模型推理的局部节点引入动态决策循环**。

以售后退款为例：
1. **确定性入口（代码执行）**：提取订单号，调用后端鉴权接口，检查订单状态。若已过售后有效期，直接返回预定义提示，无需调用模型。
2. **模型分析节点（动态 Agent）**：如果用户申诉原因复杂，将聊天记录与商品退换政策送入模型，让模型调取质检工具分析破损照片并给出责任归属建议。
3. **确定性出口（代码兜底与审批）**：模型判断可以退款后，程序检查退款金额。低于阈值自动调用支付网关（通过幂等键防止重复打款）；高于阈值则挂起流程，等待人工审批（人在回路）。

通过状态图（如 LangGraph 等框架所实现的机制），系统既获得了处理长尾语义的灵活性，又守住了业务底线。

---

## 模块内容进阶全景导航

本模块共包含 10 讲内容，沿着“架构选型 $\to$ 认知规划 $\to$ 工具与环境 $\to$ 稳定性与工程化”四层梯度推进：

![Agent 工程模块十讲知识递进拓扑图](/images/content-rebuild/d9467ed3c22f7f52/fig-2.png)

### 1. 架构选型与状态编排
* [LangChain、LangGraph、CrewAI 分别适合什么场景？](agent-framework-selection.md)：对比主流框架在控制粒度、架构复杂度与多智能体协作上的设计权衡。
* [LangGraph 为什么适合构建可控 Agent 流程？](langgraph-controlled-workflow.md)：解析图编排运行机制、环形状态流动支持与跨轮次会话持久化（Persistence）。
* [Agent 状态机怎么设计？](agent-state-machine.md)：学习如何显式定义全局状态（State Schema）、保持节点函数的无副作用纯函数规范，以及设计状态快照回退。

### 2. 认知架构与任务规划
* [Planner、Executor、Reflection 分别是什么？](planner-executor-reflection.md)：厘清规划拆解、工具执行与结果反思三大分工模块的协同逻辑与常见死循环陷阱。
* [Agent 如何进行任务拆解？](agent-task-decomposition.md)：拆解复合目标为多阶段有向依赖树，处理上下文污染与子任务隔离。

### 3. 工具接入、防护与环境隔离
* [Agent 如何接入搜索、数据库和本地文件？](agent-tool-integration.md)：解析工具描述规范（Tool Schema）、参数类型校验与外部环境集成模式。
* [Agent 如何处理工具调用失败？](agent-tool-failure.md)：面对参数格式错误、服务超时与非预期返回值的逐级重试与降级恢复策略。
* [Agent 如何做权限控制和安全隔离？](agent-permissions-isolation.md)：实现代码沙箱执行、只读数据源视图与高危动作的人在回路（Human-in-the-loop）拦截机制。

### 4. 生产可观测性与高可用交付
* [Agent 如何记录执行轨迹和中间状态？](agent-execution-tracing.md)：基于标准跟踪格式记录节点耗时、中间变量快照与工具调用全轨迹。
* [面试高频：Agent 项目如何保证稳定性？](agent-reliability.md)：掌握多轮调用死循环熔断、接口幂等性保障以及离线回归评测方案。

如需扩展了解系统层面的流水线与提示词上下文设计，可配合阅读：[AI 工程架构](/architecture/README.md)、[AI Workflow](/frontier/ai-workflow/README.md) 与 [上下文工程](/frontier/context-engineering/README.md)。

---

## 实验与运行环境检验

为了动手实践后续章节的代码，建议在本地准备独立的 Python 虚拟环境（建议 Python 3.10 及以上版本）。

![本地开发与可观测验证环境运行示意](/images/content-rebuild/d9467ed3c22f7f52/fig-3.png)

### 1. 依赖安装
```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows 用户执行: .venv\Scripts\activate

# 2. 安装图编排与核心依赖库
pip install -U langgraph langchain-core pydantic
```

### 2. 状态机最小可验证示例
通过下面这段无需大模型 API 密钥的代码，你可以快速验证本地安装是否正常，并体会“显式定义状态数据结构、节点更新字典、条件路由分发”的核心流程：

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

# 1. 定义全局状态契约：每个节点接收该状态并返回增量更新
class OrderTicketState(TypedDict):
    order_id: str
    order_amount: float
    reason: str
    audit_route: str
    process_log: list[str]

# 2. 编写处理节点（遵循接收 state 并返回新字典的规范）
def triage_node(state: OrderTicketState) -> dict:
    # 模拟对工单分流：高额订单或复杂破损走人工仲裁，其余走自动化规则
    if state["order_amount"] > 500.0 or "碎" in state["reason"]:
        route = "agent_arbitration"
    else:
        route = "auto_refund"
    return {
        "audit_route": route,
        "process_log": state["process_log"] + [f"分流判定完成: 分流至 {route}"]
    }

def auto_refund_node(state: OrderTicketState) -> dict:
    return {
        "process_log": state["process_log"] + [f"订单 {state['order_id']} 自动化小额退款完成"]
    }

def agent_arbitration_node(state: OrderTicketState) -> dict:
    # 模拟进入局部智能体决策或人工复审环节
    return {
        "process_log": state["process_log"] + [f"订单 {state['order_id']} 已转入专家仲裁流"]
    }

# 3. 条件边路由函数：根据状态中的值决定下一个节点
def route_decision(state: OrderTicketState) -> Literal["auto_refund_node", "agent_arbitration_node"]:
    if state["audit_route"] == "auto_refund":
        return "auto_refund_node"
    return "agent_arbitration_node"

# 4. 构建并编译状态图
workflow = StateGraph(OrderTicketState)
workflow.add_node("triage_node", triage_node)
workflow.add_node("auto_refund_node", auto_refund_node)
workflow.add_node("agent_arbitration_node", agent_arbitration_node)

workflow.add_edge(START, "triage_node")
workflow.add_conditional_edges("triage_node", route_decision)
workflow.add_edge("auto_refund_node", END)
workflow.add_edge("agent_arbitration_node", END)

app = workflow.compile()

# 5. 验收测试
# 测试用例 A：普通小额未破损订单 -> 走自动化规则
res_a = app.invoke({
    "order_id": "A001",
    "order_amount": 49.9,
    "reason": "尺码不合适",
    "audit_route": "",
    "process_log": []
})
assert res_a["audit_route"] == "auto_refund"
assert "自动化小额退款完成" in res_a["process_log"][1]

# 测试用例 B：破损争议订单 -> 走仲裁流程
res_b = app.invoke({
    "order_id": "B002",
    "order_amount": 899.0,
    "reason": "商品屏幕碎了",
    "audit_route": "",
    "process_log": []
})
assert res_b["audit_route"] == "agent_arbitration"
assert "专家仲裁流" in res_b["process_log"][1]

print("恭喜！LangGraph 状态图基础环境验证通过，工单路由断言全部正确。")
```

---

## 课后动手检验
完成本讲后，请尝试在本地运行上述代码，并思考：
1. 如果要在 `auto_refund_node` 增加对外部支付接口的调用，需要引入什么机制防止用户网络卡顿重复点击导致的“重复退款”？
2. 为什么在状态图中推荐将节点编写为只返回增量字典的纯函数，而不是直接修改传入的全局变量？

这些问题将在后续的 [状态机设计](agent-state-machine.md) 与 [系统稳定性保障](agent-reliability.md) 中逐一剖析解决。
