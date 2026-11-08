---
title: 什么是 LangChain 中的 Agent？它和 Chain 有什么区别？
icon: mdi:graph
tag:
- LangChain
- Agent
- Chain
description: 从控制权归属、执行图结构和贯穿实例对比 LangChain 中 Chain 与 Agent 的区别，拆解面试回答策略与混合架构设计。
---

在技术面试中聊到 LangChain 的 Chain 与 Agent 时，很多开发者容易给出“Agent 会调工具、Chain 不会调工具”或“Agent 比 Chain 更高级”这类模糊表述。实际上，工具调用只是表面功能，两者的关键差异在于**谁掌握下一步执行的决定权**，以及**程序底层的执行图形态是有向无环图还是循环图**。

理解这个差异不仅能帮你在面试中准确作答，还能让你在实际业务选型时，清楚什么场景该用流水线，什么场景该交给模型自主决策。

![Chain 与 Agent 架构差异对比封面](/images/content-rebuild/183eda2b5018348f/cover.png)

---

## 60 秒口述答案：面试中怎么回答

如果在面试中被问到两者的区别，可以用这段话清晰表述：

> “LangChain 中 Chain 与 Agent 的核心区别在于**执行控制权的归属**与**计算图的拓扑形态**：
>
> * **Chain 是由代码预设的固定流水线（有向无环图 DAG）**：步骤顺序、分支条件和退出时机完全由程序员在代码中写死。LLM 在其中只是流水线中受代码编排的数据处理节点，不能决定程序下一步去调用哪个节点。
> * **Agent 是由大模型驱动的动态循环状态机（Cyclic Graph）**：程序将‘下一步做什么’的决定权交给了 LLM。模型通过观察环境和工具返回的结果，动态决定是继续调用工具、更换参数，还是结束循环输出最终答案。
>
> Chain 的优势是确定性强、延迟和 Token 开销可控、排查简单；Agent 的优势是能自主应对多变与未知步骤的任务，但伴随非确定性、高延迟和死循环风险。在生产环境中，通常推荐使用 LangGraph 这类状态图框架，用确定性的代码工作流去包裹局部的 Agent 循环。”

![静态流水线与动态决策智能体在执行模式上的直观对比](/images/content-rebuild/183eda2b5018348f/fig-1.png)

---

## 一个贯穿例子：处理“用户查询延误订单并补偿”

为了直观看到差异，我们看同一个业务需求在两种架构下的实现方式：**“用户查询订单状态，如果延误超过 3 天，就自动发放补偿优惠券并告知用户。”**

### 方式 A：用 Chain 实现（固定流水线）

代码中明确写好执行路径：
1. 第一步：运行代码调 API 查询订单状态；
2. 第二步：代码逻辑判断 `if delay_days > 3`，若满足则调用发券函数；
3. 第三步：把订单状态和发券结果拼成 Prompt，交给 LLM 润色为一段自然语言发给用户。

整个过程中，每一步先走哪个函数、何时调用 LLM、何时结束，全部由宿主代码逻辑控制。LLM 没有任何调度程序的权力。

### 方式 B：用 Agent 实现（模型自主循环）

程序向 LLM 注册了两个工具：`query_order` 和 `issue_coupon`，并把用户的初始问题输入模型：
1. **第一轮推理**：LLM 分析需求，决定调用 `query_order(order_id="123")`；
2. **执行并反馈**：宿主程序执行该工具，返回结果“订单已延误 4 天”；
3. **第二轮推理**：LLM 读到返回结果，根据内置要求判断满足补偿条件，决定调用 `issue_coupon(user_id="U456")`；
4. **执行并反馈**：宿主程序执行发券，返回结果“发放成功”；
5. **第三轮推理**：LLM 看到发券成功，判定全部任务已完成，输出最终回复文本，不再产生工具调用，循环退出。

在这个例子中，程序并不知道模型究竟会调用几次工具，也不知道它会不会去调发券接口。**决定走哪条路、调哪个接口、走几轮结束，全由 LLM 输出的指令决定。**

---

## 机制对比：静态有向无环图与动态循环图

从底层计算图的结构来看，两者代表了两种截然不同的执行范式。

### 1. Chain：单向流动的有向无环图（DAG）

在 LangChain 中，通过 LCEL（LangChain 表达语言）组装的 Chain 无论多长，数据都沿着预设的边单向流动：

```python
# 典型的确定性顺序 Chain
chain = prompt_template | llm | output_parser
```

* **节点（Node）**：执行具体的 Python 函数、Prompt 格式化或模型请求。
* **边（Edge）**：严格单向无环。即使代码中写了条件分支（例如 `RunnableBranch`），分支规则也是由程序员写在代码里的布尔函数。分支走向在输入数据到达时就已经由固定规则算出，无法在运行中由模型自主掉头。

### 2. Agent：带观察反馈的循环图（Loop / Cycle）

Agent 的执行流是一个带有环境反馈的持续循环。单次执行可以抽象为状态的推进过程：

$$\text{当前状态} \xrightarrow{\text{LLM 推理}} \text{决定动作 (Action)} \xrightarrow{\text{运行工具}} \text{得到反馈 (Observation)} \xrightarrow{\text{更新状态}} \text{下一轮推理}$$

![Chain 的单向流水线与 Agent 的循环图拓扑对比](/images/content-rebuild/183eda2b5018348f/fig-2.png)

模型在每个步骤产出的动作会直接改变程序流向：如果输出包含 `tool_calls`，程序跳向工具执行节点；工具节点把执行结果写回对话消息后，无条件连回模型节点再次推理；直到模型认为信息足够、不再生成工具调用时，流向才跳向终止节点（END）。这种**循环往复的回路**是 Chain 结构所不具备的。

<ReadMoreLock />

---

## 控制权归属：代码驱动 vs 模型驱动

面试官常会深挖：“如果我在 Chain 里面写了 10 个 if-else 分支，它算不算 Agent？”

答案很明确：**不算**。决定两者界限的不是分支有多复杂，也不是其中是否包含模型分类，而是**执行流是由预设的代码路径严格限定，还是由模型在环境反馈循环中自主驱动下一步动作与退出时机**。

| 维度 | Chain（流水线） | Agent（智能体） |
| :--- | :--- | :--- |
| **控制权主体** | 工程师编写的静态宿主代码 | 大语言模型在推理时生成的 Token |
| **执行路径** | 确定：代码逻辑预先编排完成 | 动态：由模型根据运行时环境逐步推演 |
| **图结构** | 有向无环图（DAG） | 包含条件跳转与反馈的有环图（Cycle） |
| **退出条件** | 遍历完所有预定义步骤即结束 | 模型自主输出最终文本，或触发代码硬性熔断 |
| **典型适用场景** | 步骤固定、流程规范、强合规的业务任务 | 开放式排查、多工具试错、需要动态规划的任务 |

![状态转移控制权的归属机制对比](/images/content-rebuild/183eda2b5018348f/fig-3.png)

### 从核心调度代码看控制权

我们通过伪代码直接看两者的控制流差别：

**代码驱动（Chain 路由模式）**：
```python
def route_step(state: dict) -> str:
    # 即使模型做分类，后续去哪里也是由 Python 代码白纸黑字严格限定的
    intent = state.get("intent")
    if intent == "refund":
        return "refund_node"
    return "default_node"
```

**模型驱动（Agent 循环分发模式）**：
```python
# Agent 的核心循环逻辑
while step_count < max_steps:
    response = llm.invoke(messages, tools=tools_schema)
    messages.append(response)
    
    # 程序自己不知道下一步去哪，完全依据模型返回中是否有 tool_calls 决定
    if not response.tool_calls:
        return response.content  # 模型决定结束，输出文本
        
    for tool_call in response.tool_calls:
        result = execute_tool(tool_call.name, tool_call.args)
        messages.append(result)
    step_count += 1
```

在 Agent 中，宿主代码蜕变为一个“执行沙箱与调度器”，真正指挥程序指针的是模型在每一步生成的参数结构。

![同一个具体任务在 Chain 与 Agent 下的不同状态轨迹演变](/images/content-rebuild/183eda2b5018348f/fig-4.png)

---

## 排障与可观测性对比

控制权的让渡直接改变了系统的调试复杂度。

### 1. Chain 的排障：单步隔离与确定性重现
Chain 的每个节点输入与输出契约是明确的。如果第 3 步出现解析异常，工程师只要拿到第 2 步的输出作为入参，在本地编写单元测试就能稳定复现，定位是 Prompt 编写问题还是代码逻辑错误。每次请求消耗的 Token 和网络耗时也基本保持在稳定区间内。

### 2. Agent 的排障：非确定性轨迹与状态漂移
Agent 由于每一步都依赖模型生成，排查故障通常要复杂得多：
* **路径漂移**：相同的输入在模型温度不为 0 或提示词微调后，可能在第 2 步选择完全不同的工具；
* **工具死循环**：当工具返回未预料的报错信息时，模型可能无法理解该错误，不断用相同的错误参数重复调用同一工具，陷入死循环；
* **上下文污染**：多轮失败调用的报错日志堆积在上下文（Context Window）中，越积越多，导致模型推理质量大幅下降。

![Chain 的单点故障与 Agent 的死循环失效路径分析](/images/content-rebuild/183eda2b5018348f/fig-5.png)

因此，在构建 Agent 时，**绝对不能完全信任模型的自主停止能力**，必须在宿主程序层增加兜底机制：
1. 强制设定最大迭代次数 `max_steps`；
2. 强制设定整次请求超时时间；
3. 通过全链路 Trace 工具（如 LangSmith 等）监控完整的 Thought-Action-Observation 轨迹。关于 Agent 出错后的系统捕获策略，可以进一步参考 [Agent 执行失败时如何恢复？](agent-error-handling.md)。

---

## 工程演进趋势：为什么生产系统倾向于用 LangGraph 做混合架构

在早期的 LangChain 实践中，开发者常陷入两个极端：要么用纯 Chain 导致遇到稍微偏离预设的分支就直接抛错崩溃；要么用全自主 Agent 导致成本不可控、调用超时且行为难以通过安全审计。

Anthropic 在工程指南 *Building effective agents* 中给出了明确经验：**高可靠性的落地往往不是完全放权的 Agent，而是用确定性的代码工作流（Workflows）编排局部灵活的 Agent 决策**。

这也是 LangChain 团队推出 LangGraph 的核心原因。LangGraph 允许在同一个状态图（StateGraph）中将确定性节点和动态 Agent 循环无缝混合：

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(WorkflowState)

# 1. 确定性前置节点（Chain 模式）：参数清洗、鉴权、权限检查（严格代码控制）
builder.add_node("auth_and_sanitize", deterministic_security_check)
builder.add_node("fetch_user_profile", deterministic_db_query)

# 2. 局部 Agent 循环：只在需要发散分析或排查的长尾环节交给模型决策
builder.add_node("agentic_troubleshoot", llm_tool_loop_node)

# 3. 确定性后置节点（Chain 模式）：合规风控审查、输出格式序列化（严格代码控制）
builder.add_node("compliance_guard", deterministic_compliance_check)

# 编排执行路径
builder.add_edge(START, "auth_and_sanitize")
builder.add_edge("auth_and_sanitize", "fetch_user_profile")
builder.add_edge("fetch_user_profile", "agentic_troubleshoot")
builder.add_edge("agentic_troubleshoot", "compliance_guard")
builder.add_edge("compliance_guard", END)
```

![LangGraph 混合编排架构：确定性节点与 Agentic 循环的共存](/images/content-rebuild/183eda2b5018348f/fig-6.png)

这种混合设计把安全底线（如权限判定、合规审核）用纯代码锁死，仅在需要自主查询和多步尝试的局部环节启用 Agent 循环，既保留了灵活性，又守住了生产系统的稳定性底线。关于各类多智能体框架的具体选型，可以查阅 [CrewAI、LangGraph 和 AutoGen 的核心差异是什么？](agent-frameworks.md)。

---

## 常见追问与避坑清单

### 追问 1：在什么场景下，把 Chain 重构成 Agent 反而是反模式？

* **回答要点**：业务流程明确收敛、步骤固定、对响应延迟和准确率要求极其严苛的场景（例如金融开户表单提取、固定格式报表生成）。在这些场景中，流程本来就能用几行代码清晰编排，强行引入 Agent 会白白增加多轮 LLM 交互延迟与调用成本，而且会引入模型生成无效参数或选错工具的不确定性风险。

### 追问 2：LCEL 中的 RunnableBranch 算不算 Agent 的轻量形态？

* **回答要点**：不算。`RunnableBranch` 本质上只是 DAG 里的静态条件路由，它的路由依据在每个节点执行后由程序代码判定，整个图依旧无环且单向流动。Agent 的核心特征是**带有环境观察反馈的循环迭代（Loop）**，必须包含“执行工具 -> 结果返回给模型 -> 模型根据结果决定下一跳”的闭环反馈结构。

### 容易说错的两个误区

* 误区 1：**“Chain 只能调一次 LLM，Agent 能调多次”**。
  纠正：错误。Chain 可以级联多个 LLM 节点，比如经典的 Map-Reduce 长文档摘要 Chain，会发起多次 LLM 调用，但每次调用的顺序在代码中是固定编排好的，不代表它是 Agent。
* 误区 2：**“Agent 必须有长短期记忆，Chain 只能无状态”**。
  纠正：错误。Chain 也可以挂载 `ConversationBufferMemory` 维护上下文状态；记忆机制是系统组件，不能作为区分 Chain 与 Agent 的根本边界。
