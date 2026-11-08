---
title: LangGraph 为什么适合构建可控 Agent 流程？如何混合确定性代码与模型决策
icon: mdi:book-open-page-variant
description: 解析纯自主 Agent 在业务场景中的发散风险，讲解 LangGraph 状态图、节点与条件边机制，通过退款与咨询贯穿案例掌握受控流程设计与静态编译校验。
---

在上一讲[《LangChain、LangGraph、CrewAI 分别适合什么场景？》](/llm-100/agent-engineering/agent-framework-selection.md)中，我们梳理了主流 Agent 框架的技术定位，明确了 LangGraph 作为底层状态图运行时，在处理高可靠长流程时的定位。

本讲的学习目标是：理解大语言模型（LLM）纯自主循环在真实业务中的失控风险，掌握 LangGraph 的核心图元素（State、Node、Edge、Conditional Edge），并亲手实现一个“确定性代码守门、模型负责推理提取”的受控客服退款流程。

![基于 LangGraph 构建混合控制流的受控 Agent 封面](/images/content-rebuild/1b1859ef54bfc6c9/cover.png)

---

### 贯穿场景：为什么严肃业务不能全交给模型自主循环？

以电商平台的“退款与常见问题（FAQ）咨询”为例。如果采用完全自主的 ReAct 循环，外部系统将所有工具（如查询知识库、风控核验、退款接口）交给大模型，控制流完全由模型每一步生成的推理文本和工具调用参数驱动。

在理想情况下，模型能识别意图并按步骤完成任务；但在实际应用中，纯概率模型主导的无约束循环存在三类典型问题：

1. **步骤不可控与循环震荡**：面对模糊输入或非预期的工具返回，模型可能在多个工具之间反复横跳，或者陷入无效的自我猜测。这不仅浪费 Token 和增加延迟，还可能消耗完上下文窗口导致任务失败。
2. **关键风控无法兜底**：如果由模型自主判断“用户是否符合退款条件”以及“调用退款接口转账多少元”，一旦遭遇提示注入（Prompt Injection）或模型幻觉，安全防线可能失效，造成未授权放款。
3. **业务状态不可审计与恢复**：纯自主循环的状态隐藏在上下文对话历史中，外部调度器难以量化评估当前任务进展到哪一步。如果调用发生超时，调度器无法确定当前动作是否已经产生了外部副作用（如扣款、发信），从而难以实施幂等重试。

Anthropic 在《Building effective agents》中将这类系统区分为两类：
- **工作流（Workflows）**：通过确定性代码路径编排模型与工具，具有高可预测性。
- **自主 Agent（Agents）**：由模型在运行时自主决定后续步骤与工具调用。

在金融、支付、风控等高敏感业务中，更稳妥的做法是构建**受控 Agent 流程**：使用固定的拓扑结构编排任务主干，用常规代码执行权限校验、输入清洗和事务操作；仅在需要自然语言理解、信息抽取或灵活草拟回复的局部节点接入模型。

![纯自主 Agent 与受控 Agent 图工作流的控制力对比](/images/content-rebuild/1b1859ef54bfc6c9/fig-1.png)

---

### LangGraph 核心图元素：StateGraph、Node 与 Edge

LangGraph 的运行时受 Pregel 消息传递模型启发，其核心思路是将任务交互建模为**带有共享状态的有向图（Directed Graph）**。在编写具体流程前，需要掌握以下核心概念：

- **状态模式（State Schema）**：节点之间流转的数据契约。通常使用 Python 的 `TypedDict` 或 Pydantic 定义。每个节点都可以读取当前状态的快照，并返回包含部分字段更新的字典。
- **状态图构建器（StateGraph）**：用于装配节点、边以及绑定状态契约的容器对象。在调用编译前，它负责维护静态拓扑结构。
- **计算节点（Node）**：图中的执行单元，签名通常为 `Callable[[State], dict]` 的普通 Python 函数或协程。节点既可以是一个包装好的 LLM 调用，也可以是一段纯代码逻辑（如正则过滤、数据库读写）。
- **普通边（Edge）**：连接两个节点的单向连线。源节点执行完成后，控制流无条件传递给目标节点。
- **虚拟起点与终点（START / END）**：LangGraph 内置的图边界常量。`START` 代表流程入口，接收外部初始输入；`END` 代表图终结，执行流到达该标记后任务结束。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. 定义状态契约
class SimpleState(TypedDict):
    input_query: str
    cleansed_query: str
    response: str

# 2. 声明节点逻辑
def clean_input_node(state: SimpleState) -> dict:
    # 确定性代码处理：规范输入格式
    return {"cleansed_query": state["input_query"].strip()}

def mock_model_node(state: SimpleState) -> dict:
    # 模型决策或内容生成
    return {"response": f"已处理: {state['cleansed_query']}"}

# 3. 组装拓扑图
builder = StateGraph(SimpleState)
builder.add_node("clean_input", clean_input_node)
builder.add_node("mock_model", mock_model_node)

# 4. 连接确定性边
builder.add_edge(START, "clean_input")
builder.add_edge("clean_input", "mock_model")
builder.add_edge("mock_model", END)
```

![LangGraph 核心图元素与状态流转拓扑结构](/images/content-rebuild/1b1859ef54bfc6c9/fig-2.png)

---

### 确定性代码节点与 LLM 推理节点的配合原则

在一个受控工作流中，节点不能无差别地全部接入模型，而应根据可靠性要求进行明确分工：

#### 确定性代码节点的职责
确定性节点由常规程序代码构成，执行固定的分支判断与数据转换，不受模型采样随机性影响。主要承担：
- **前置防护与输入规范化**：过滤异常控制字符、绑定会话租户 ID 与时间戳、执行硬性敏感词拦截。
- **权限与风控拦截**：判断当前用户是否具备执行操作的权限凭证（通过程序逻辑与数据库比对），不能交给模型自主推理决定。关于权限隔离细节，可参阅[《Agent 如何做权限控制和安全隔离？》](/llm-100/agent-engineering/agent-permissions-isolation.md)。
- **副作用操作与事务提交**：资金扣减、修改主库状态等具有副作用的操作，必须由代码节点接收模型提取的结构化参数后，进行二次规则核验，再执行实际接口调用。

#### LLM 推理节点的职责
模型推理节点聚焦于非结构化文本的理解与生成：
- 自由自然语言意图分类与槽位信息抽取。
- 结合召回的知识库上下文撰写友好的咨询回复。
- 在给定的候选工具列表中，建议下一步调用的工具名及初步参数草稿。

通过让代码节点充当“守门员”，系统可以保证：**即使模型生成了不合规的参数或受到提示词扰动，底线业务规则依然会通过 Python 代码进行拦截。**

![确定性代码守门节点与 LLM 推理节点的职责划分对比](/images/content-rebuild/1b1859ef54bfc6c9/fig-3.png)

---

### 条件边（Conditional Edge）：动态路由与重试回路

如果图中的边全为普通边，流程将退化为线性管道。为了应对动态分支，LangGraph 提供了**条件边（Conditional Edge）**。

条件边通过 `StateGraph.add_conditional_edges(source, path, path_map)` 声明，包含三个要素：
1. **源节点（source）**：触发条件判断的起始节点。
2. **路由函数（path）**：一个接收当前状态并返回路由标识符（如字符串）的 Python 函数。该函数只读读取状态，不产生状态更新。
3. **路由映射表（path_map）**：字典结构，将路由函数返回的标识符显式映射到具体的下游 Node 名称或 `END`。显式声明映射表有助于在编译期验证下游节点是否存在。

```python
def route_by_validation(state: SimpleState) -> str:
    # 根据状态字段决定下游去向
    if not state.get("cleansed_query"):
        return "invalid"
    return "valid"

# 挂载条件边
builder.add_conditional_edges(
    "clean_input",
    route_by_validation,
    {
        "valid": "mock_model",
        "invalid": END
    }
)
```

条件边不仅支持多路分流，也是构建**重试回路（Retry Loop）**与**失败降级**的核心机制。关于工具异常的重试设计，可参考[《Agent 如何处理工具调用失败？》](/llm-100/agent-engineering/agent-tool-failure.md)。

![LangGraph 条件边多分支动态路由机制](/images/content-rebuild/1b1859ef54bfc6c9/fig-4.png)

---

### 实战示例：受控订单退款与咨询流程

下面构建一个受控客户服务 Agent 流程：
- 普通咨询：路由至知识库回复节点。
- 退款申请：由模型解析意图与金额，但后续必须流转给确定性代码节点进行额度校验（单笔上限 500 元）。
- 异常处理：金额非法或超出限额时，若未达重试上限则进入修复提示节点提示修改并递增计数；若达到重试上限，则流转至人工兜底节点。

```python
from typing import TypedDict, Literal, Optional
import re
from langgraph.graph import StateGraph, START, END

# 1. 定义状态模式契约
class SupportState(TypedDict):
    user_input: str
    intent: Literal["faq", "refund", "unknown"]
    refund_amount: Optional[float]
    is_authorized: bool
    retry_count: int
    final_output: str

# 2. 节点逻辑实现
def input_sanitizer_node(state: SupportState) -> dict:
    """确定性代码节点：输入规范化与重试计数初始化"""
    raw = state["user_input"].strip()
    # 基础输入清洗示例
    sanitized = raw.replace("ignore previous instructions", "[FILTERED]")
    return {"user_input": sanitized, "retry_count": state.get("retry_count", 0)}

def mock_model_intent_node(state: SupportState) -> dict:
    """模型推理节点：意图解析与参数提取（此处使用规则模拟模型推理产出）"""
    text = state["user_input"]
    if "退款" in text:
        amounts = re.findall(r"\d+", text)
        amount = float(amounts[0]) if amounts else -1.0
        return {"intent": "refund", "refund_amount": amount}
    return {"intent": "faq", "refund_amount": None}

def business_validator_node(state: SupportState) -> dict:
    """确定性代码节点：业务规则与风控硬性校验"""
    amount = state.get("refund_amount", 0)
    # 规则：金额必须大于 0 且不超过 500 元
    is_valid = (amount is not None) and (0 < amount <= 500.0)
    return {"is_authorized": is_valid}

def model_fix_node(state: SupportState) -> dict:
    """模型修复节点：提示重新确认金额并递增重试计数"""
    current_retry = state.get("retry_count", 0) + 1
    return {
        "retry_count": current_retry,
        "final_output": f"退款金额解析失败或超出单笔限额（500元）。当前尝试次数：{current_retry}"
    }

def fallback_node(state: SupportState) -> dict:
    """确定性代码节点：超过重试上限，流转至人工兜底"""
    return {"final_output": "退款核验多次未通过，已为您转接人工客服处理。"}

def execution_node(state: SupportState) -> dict:
    """确定性代码节点：调用外部支付接口"""
    return {"final_output": f"退款成功！受理金额：{state['refund_amount']} 元。"}

def faq_response_node(state: SupportState) -> dict:
    """模型回复节点：知识库问答生成"""
    return {"final_output": "这是关于售后保障期限的常见问题答复。"}

# 3. 路由函数定义
def route_after_intent(state: SupportState) -> str:
    if state["intent"] == "refund":
        return "to_validator"
    return "to_faq"

def route_after_validation(state: SupportState) -> str:
    if state["is_authorized"]:
        return "to_execute"
    if state.get("retry_count", 0) < 2:
        return "to_fix"
    return "to_fallback"

# 4. 图结构组装
workflow = StateGraph(SupportState)

workflow.add_node("sanitizer", input_sanitizer_node)
workflow.add_node("intent_classifier", mock_model_intent_node)
workflow.add_node("validator", business_validator_node)
workflow.add_node("fix_node", model_fix_node)
workflow.add_node("fallback", fallback_node)
workflow.add_node("executor", execution_node)
workflow.add_node("faq_responder", faq_response_node)

# 连接边
workflow.add_edge(START, "sanitizer")
workflow.add_edge("sanitizer", "intent_classifier")

# 条件分流
workflow.add_conditional_edges(
    "intent_classifier",
    route_after_intent,
    {
        "to_validator": "validator",
        "to_faq": "faq_responder"
    }
)

workflow.add_conditional_edges(
    "validator",
    route_after_validation,
    {
        "to_execute": "executor",
        "to_fix": "fix_node",
        "to_fallback": "fallback"
    }
)

workflow.add_edge("fix_node", END)
workflow.add_edge("fallback", END)
workflow.add_edge("executor", END)
workflow.add_edge("faq_responder", END)

# 编译为可执行图
app = workflow.compile()
```

运行验证测试用例：

```python
# 校验用例 1：合法退款申请
res1 = app.invoke({"user_input": "我需要申请退款 120 元", "retry_count": 0})
print(res1["final_output"])
# 预期输出: 退款成功！受理金额：120.0 元。

# 校验用例 2：越界非法退款申请（超出 500 元限额，未达重试上限）
res2 = app.invoke({"user_input": "退款 8000 元", "retry_count": 0})
print(res2["final_output"])
# 预期输出: 退款金额解析失败或超出单笔限额（500元）。当前尝试次数：1

# 校验用例 3：达到重试上限，流转至人工兜底节点
res3 = app.invoke({"user_input": "退款 8000 元", "retry_count": 2})
print(res3["final_output"])
# 预期输出: 退款核验多次未通过，已为您转接人工客服处理。
```

![受控客服退款流程实例的状态流转与节点执行细节](/images/content-rebuild/1b1859ef54bfc6c9/fig-5.png)

---

### 编译与校验（compile）：静态拓扑检查与运行时步数限制

在代码末尾调用 `workflow.compile()` 时，LangGraph 会将声明的拓扑结构转化为一个可执行对象（`CompiledGraph`）。理解编译阶段做与不做什么，有助于规避流程设计缺陷：

1. **静态拓扑完整性检查**：
   - **入口边存在性**：检查是否显式配置了从 `START` 出发的边。缺少入口边会在编译期抛出异常。
   - **节点引用有效性**：遍历所有普通边与条件边的 `path_map`，核对引用的节点名称是否已通过 `add_node` 注册。若存在拼写错误或悬空边，编译期会主动报错中断。
2. **有环图的停机问题与递归限制**：
   - 本实战示例构建的是单轮状态流转的有向无环图（DAG），通过业务状态字段 `retry_count` 控制多轮会话中的退款重试上限。
   - 若将修复节点（`fix_node`）通过条件边直接连回 `intent_classifier` 形成图内自主重试闭环，图结构将演进为**有向有环图（Cycles）**。
   - 由于动态路由取决于运行时的模型输出与状态数据，**编译期无法静态证明有环图是否一定会收敛到 `END`**。为了防止逻辑死循环耗尽 Token 与计算资源，LangGraph 在执行配置中内置了 `recursion_limit` 参数（默认阈值为 25）。在一次 `invoke` 执行中若节点流转步数（Hop）超过该上限，引擎会抛出 `GraphRecursionError` 强制中断任务。
3. **运行时观察与轨迹流式输出**：
   - 编译后的图支持 `.stream(initial_state)` 接口。调度层可以按节点流式接收每次状态更新的增量字典，从而记录细粒度的执行轨迹。这部分实现机制将在[《Agent 如何记录执行轨迹和中间状态？》](/llm-100/agent-engineering/agent-execution-tracing.md)中进一步讲解。

![LangGraph 编译生命周期、静态拓扑校验与运行时递归限制机制](/images/content-rebuild/1b1859ef54bfc6c9/fig-6.png)

---

### 进阶动手练习与验收标准

为检验对状态图编排与条件边的掌握，请完成以下练习：

- **练习目标**：在上方客服退款工作流中，为 `business_validator_node` 增加会员等级区分规则。
- **具体规则**：
  1. 在 `SupportState` 中增加一个字段 `user_level: Literal["standard", "vip"]`。
  2. 业务风控规则修改为：普通用户（`standard`）单笔退款上限为 500 元；VIP 用户（`vip`）单笔退款上限提升至 2000 元。
  3. 调整测试用例，验证当输入为 `{"user_input": "退款 1200 元", "user_level": "vip", "retry_count": 0}` 时，能够成功进入 `executor` 节点并输出成功提示；而当 `user_level` 为 `standard` 时被正确拦截并流转至 `fix_node`。

---

### 总结与后验指引

受控 Agent 架构的核心原则，是**接受大语言模型的概率波动特征，并用确定性的图拓扑对不确定性进行边界约束**。通过 LangGraph，复杂的文本提示词试探被解构为具体的节点与有向边：常规规则由代码节点防守，意图理解由模型节点担当，分支流转由条件边保障。

在拓扑结构搭建完成之后，另一个工程关键在于：节点之间如何安全地传递状态？遇到并发更新或多轮历史追加时如何避免状态相互覆盖？下一讲[《Agent 状态机怎么设计？》](/llm-100/agent-engineering/agent-state-machine.md)将详细解析 LangGraph 的状态模式（State Schema）、增量更新规约（Reducer）以及检查点持久化机制。
