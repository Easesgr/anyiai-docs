---
title: Agent 状态机怎么设计？状态结构、增量更新与断点恢复指南
icon: mdi:book-open-page-variant
description: 详解 Agent 状态机设计：从 TypedDict 结构、add_messages 规约机制到 Pregel 超步并发与断点恢复。
---

在开发具备多步推理和工具调用的 Agent 时，很多开发者最先遇到的并不是提示词写得好不好，而是流程跑着跑着就失控了：

- 服务进程偶尔重启或超时，前面执行了五六步的大模型调用和工具结果全部丢失，只能从头重来；
- 涉及退款、转账或删除数据库等高风险动作，需要等待人工审批，但普通的函数调用无法直接挂起并在几小时后原样恢复；
- 多个分支并发执行时，两个工具同时修改同一个变量，造成状态覆盖冲突。

解决这些工程问题的标准方案是**状态机（State Machine）**。在上一讲 [/llm-100/agent-engineering/langgraph-controlled-workflow.md](/llm-100/agent-engineering/langgraph-controlled-workflow.md) 中，我们讨论了为什么确定性流程图是构建可靠 Agent 的骨架。本讲我们将以一个“退款审批与执行 Agent”为贯穿案例，深入解析状态机运行时的核心机制：状态结构定义（Schema）、增量更新规约（Reducer）、基于 Pregel 模型的超步同步，以及断点持久化与人工介入机制。

![Agent 状态机核心机制全景图](/images/content-rebuild/9cefdf4c75cd9cca/cover.png)

---

### 状态在 Agent 架构中的角色：全局数据源

在经典的状态机理论与分布式运行时（参考 [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph/overview)）中，**状态（State）**是跨节点通信的单一数据源。

在普通的嵌套函数调用中，节点 A 必须显式将参数传给节点 B；当流程中出现条件判断、循环或并行分支时，函数入参和出参的依赖关系会迅速变得混乱。状态机架构通过把“数据管理”与“逻辑执行”分离，改变了这种依赖：

1. **数据流动模式**：图中的每一个节点都是一个执行单元，统一接收当前状态的只读快照 $S_t$。节点只负责业务计算或模型交互，执行结束后返回状态的增量更新（通常称为补丁或 $\Delta S$）。
2. **节点通信解耦**：节点之间不直接传参。风控节点无需直接调用审批节点，它只需把风控评估结果写入状态，图的控制流会依据当前状态的值决定下一步唤醒哪一个节点。
3. **临时变量与图状态分离**：局部计算的临时变量只存在于节点函数内部；需要跨步骤保留、参与路径跳转或在故障后恢复的数据，必须显式存放在图状态中。

![状态作为单一事实来源的数据流动拓扑](/images/content-rebuild/9cefdf4c75cd9cca/fig-1.png)

如图所示，状态管理器持有最新的全局状态快照，节点读取快照后并行计算，计算完毕后将增量更新（Delta）提交给规约器（Reducer）进行合并，生成新快照后再驱动下一步的条件路由。

---

### 状态结构定义：字段隔离与契约设计

状态结构（State Schema）定义了 Agent 在整个生命周期中允许记录的数据字典与类型。在 Python 中，通常使用 `typing.TypedDict` 进行定义。

在设计状态时，建议将**面向模型的对话历史**与**面向业务逻辑的结构化字段**清晰分开：

```python
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class RefundAgentState(TypedDict):
    # 1. 对话通信列表：用于大模型多轮上下文交互
    messages: Annotated[list[AnyMessage], add_messages]
    
    # 2. 业务流转字段：供条件路由与节点分支直接判断
    order_id: str
    refund_amount: float
    risk_level: str  # "LOW" | "HIGH"
    
    # 3. 流程控制与人工审批状态
    requires_approval: bool
    approval_status: Optional[str]  # None | "APPROVED" | "REJECTED"
```

#### 为什么需要区分对话消息与业务字段？
- **防止上下文膨胀**：如果把内部控制变量（如审批状态、重试次数）都序列化成自然语言追加到 `messages` 对话历史中，会线性消耗大模型的上下文窗口，提高调用开销。
- **保障路由可靠性**：图的条件判断（Conditional Edge）应当只依赖确定性的布尔值或枚举字符串（例如 `requires_approval == True`），避免每次做流程跳转都依赖大模型解析自然语言意图。

<ReadMoreLock />

---

### 增量更新与规约函数（Reducer）

在状态机中，节点执行完毕后返回的不是修改后的完整状态，而是一个局部的更新字典。运行时通过**规约函数（Reducer）**来决定如何把这个局部更新合并进现有状态。

状态更新分为两种基本行为：

1. **默认覆盖更新**：未指定 Reducer 的字段，运行时默认采用字典覆盖语义。例如当前状态中 `risk_level` 为 `"LOW"`，节点返回 `{"risk_level": "HIGH"}`，旧值就会被直接替换。
2. **增量规约更新**：对于历史记录或对话消息，如果采用覆盖，每次返回单条消息就会把之前的历史全部抹除。此时必须使用 `typing.Annotated` 挂载 Reducer 函数。

#### `add_messages` 规约器的行为规则
在 LangGraph 中，`add_messages` 并非单纯的 Python 列表拼接（`list.extend`），它内置了消息合并逻辑：
- **新增追加**：若新消息没有指定 `id`，或该 `id` 在现有列表中不存在，则直接追加到列表末尾。
- **按 ID 原地更新**：若新消息的 `id` 与已有列表中某条消息相同，则会覆盖该 ID 对应的旧消息。这在流式输出、工具结果回填或人工纠错修改上一条消息时非常实用。
- **删除消息**：如果传入特殊的 `RemoveMessage(id=...)` 对象，规约器会在状态中移除对应 ID 的消息。

```python
# 常见错误：未挂载 Reducer 导致上下文丢失
class BrokenState(TypedDict):
    messages: list[AnyMessage]  # 缺少 Annotated[..., add_messages]

def llm_node(state: BrokenState):
    # 错误：这里返回新消息后，现有状态里的所有历史对话都将被直接冲掉
    return {"messages": [AIMessage(content="风控扫描通过")]}
```

![Reducer 覆盖更新与追加更新语义对比](/images/content-rebuild/9cefdf4c75cd9cca/fig-2.png)

上图展示了两种更新方式的区别：左侧没有配置 Reducer 的字段在接收到补丁后执行全量替换；右侧挂载了 `add_messages` 的字段则会保留历史，并支持通过 ID 索引进行幂等追加或修改。

---

### Pregel 超步模型：并发节点与状态收敛

LangGraph 的底层调度基于 Google 的 **Pregel 超步（Superstep）计算模型**。状态机并不是任意跳转的，而是通过一系列离散的时钟周期（超步）协同推进：

$$\text{超步}(t) = \text{并发执行} \longrightarrow \text{同步屏障} \longrightarrow \text{状态收敛规约} \longrightarrow \text{条件边判定}$$

1. **并发执行阶段**：在超步 $t$ 中，所有被激活的节点并发运行。此时所有并行节点读取到的都是超步 $t-1$ 结束时的只读状态快照，各节点之间互不影响。
2. **同步屏障（Barrier）**：系统等待该超步内所有激活的节点计算完毕。如果有任意节点发生未捕获异常，流程会在此中断。
3. **状态收敛规约**：收集所有并发节点提交的局部更新补丁 $\Delta S_1, \Delta S_2, \dots$，依照字段预设的 Reducer 合并到全局状态中。注意：如果多个并行节点试图向同一个没有 Reducer 的字段同时写入更新（不论写入的值是否相同），运行时会抛出写入冲突异常（`InvalidUpdateError`）。
4. **条件边判定**：根据收敛完成后的全局状态，计算出下一步需要激活的节点集合，作为超步 $t+1$ 的输入。

![Pregel 超步同步屏障与状态收敛机制](/images/content-rebuild/9cefdf4c75cd9cca/fig-3.png)

如上图所示，节点 1 和节点 2 在同一超步内并行运行，到达同步屏障后将增量补丁提交给规约器收敛成新快照，随后触发条件边激活下一个超步中的节点 3。

---

### 检查点持久化：状态快照与断点恢复

在实际生产中，为了应对网络波动、服务重启或跨时段异步等待，需要引入**检查点（Checkpointer）**机制。

#### 检查点数据模型
每次超步收敛完成后，Checkpointer 会对当前状态进行序列化并持久化存储（例如写入 SQLite、PostgreSQL 或 Redis）。定位检查点的核心元数据包含：
- `thread_id`：线程标识符，代表一个独立的会话任务或业务工单。
- `checkpoint_id`：时间戳或单调递增的版本号，用于精确定位某一次超步的状态快照。

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

checkpointer = MemorySaver()
builder = StateGraph(RefundAgentState)
# ... 添加节点与边 ...
graph = builder.compile(checkpointer=checkpointer)

# 运行时必须指定 thread_id
config = {"configurable": {"thread_id": "refund-order-98721"}}
graph.invoke({"refund_amount": 6000.0}, config=config)
```

#### 故障恢复与历史状态提取
如果程序中途异常崩溃，只需传入相同的 `thread_id` 并以 `None` 作为输入再次调用 `graph.invoke(None, config=config)`，Checkpointer 会自动加载该会话下最新保存的 `checkpoint_id`，从尚未完成的超步继续向下执行，无需重跑已经完成的前置节点。

此外，也可以通过指定过去的 `checkpoint_id` 读取历史状态快照，用于复盘调试或从某一节点重新分支执行。

![Checkpointer 状态快照与基于 thread_id 的恢复](/images/content-rebuild/9cefdf4c75cd9cca/fig-4.png)

---

### 人工介入：使用 interrupt 挂起与 Command 恢复

根据 [Anthropic 针对有效 Agent 的架构建议](https://www.anthropic.com/engineering/building-effective-agents)，涉及写操作、大额资金或高风险操作时，应当引入人机协同（Human-in-the-loop, HITL）机制。

状态机配合检查点，能够通过 `interrupt()` 原语实现优雅的挂起等待与唤醒恢复：

```python
from langgraph.types import interrupt

def human_review_node(state: RefundAgentState):
    # 1. 构造需要人工审批的载荷信息
    review_payload = {
        "task": "退款人工复核",
        "amount": state["refund_amount"],
        "order_id": state["order_id"]
    }
    
    # 2. 触发中断：运行时在此处暂停，将当前超步状态落盘并挂起流程
    review_result = interrupt(review_payload)
    
    # 3. 恢复执行后，review_result 将获取到外部传入的审批数据
    if review_result.get("action") == "APPROVE":
        return {"approval_status": "APPROVED"}
    return {"approval_status": "REJECTED"}
```

#### 人工审批交互过程
1. **执行并挂起**：调用 `graph.invoke(..., config)`，当流程流转到 `human_review_node` 时，`interrupt()` 挂起执行，当前上下文安全写入存储。
2. **查询挂起载荷**：前端或 Web 服务调用 `graph.get_state(config)`，读取到当前的挂起状态以及 `review_payload`，展示在审批后台界面。
3. **注入审批结果并恢复**：审批人员点击“通过”或“驳回”，后台通过 `Command(resume=...)` 发送恢复指令：

```python
from langgraph.types import Command

resume_cmd = Command(resume={"action": "APPROVE", "operator": "admin_tom"})
graph.invoke(resume_cmd, config=config)
```

![人工介入中断、审查修改与恢复时序图](/images/content-rebuild/9cefdf4c75cd9cca/fig-5.png)

---

### 完整可验证示例：退款审批工作流

下面是一个自包含、可直接运行的退款状态机脚本，演示了状态定义、超步持久化与人工挂起恢复的完整过程：

```python
import uuid
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# 1. 状态结构定义
class OrderWorkflowState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    order_status: str
    risk_level: str

# 2. 节点逻辑
def risk_assessment_node(state: OrderWorkflowState):
    return {
        "risk_level": "HIGH",
        "messages": [AIMessage(content="风控扫描完成：检测到大额交易，进入人工审批池。")]
    }

def human_review_node(state: OrderWorkflowState):
    review_signal = interrupt({
        "prompt": "请审查此订单并决定是否放行",
        "risk": state["risk_level"]
    })
    if review_signal.get("action") == "APPROVE":
        return {"order_status": "APPROVED"}
    return {"order_status": "REJECTED"}

def execution_node(state: OrderWorkflowState):
    status = state["order_status"]
    return {"messages": [AIMessage(content=f"订单最终处理完毕，状态为: {status}")]}

# 3. 构造状态机图
builder = StateGraph(OrderWorkflowState)
builder.add_node("risk_assessment", risk_assessment_node)
builder.add_node("human_review", human_review_node)
builder.add_node("execution", execution_node)

builder.add_edge(START, "risk_assessment")
builder.add_edge("risk_assessment", "human_review")
builder.add_edge("human_review", "execution")
builder.add_edge("execution", END)

# 绑定持久化存储
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 4. 运行验证
thread_config = {"configurable": {"thread_id": f"order-run-{uuid.uuid4().hex[:6]}"}}

# 阶段一：触发运行，流程将在 human_review 处暂停
print("=== 第一阶段：启动工作流 ===")
initial_input = {"messages": [HumanMessage(content="用户请求退款 ¥10,000")]}
for event in graph.stream(initial_input, config=thread_config):
    print("事件产生:", event)

# 查看当前挂起状态
current_state = graph.get_state(thread_config)
print("\n当前流程挂起节点:", current_state.next)
print("中断捕获的载荷:", current_state.tasks[0].interrupts[0].value)

# 阶段二：模拟人工审批放行并唤醒
print("\n=== 第二阶段：人工通过审批并恢复 ===")
resume_cmd = Command(resume={"action": "APPROVE", "reviewer": "auditor_01"})
for event in graph.stream(resume_cmd, config=thread_config):
    print("恢复后事件:", event)

# 验证最终执行结果
final_state = graph.get_state(thread_config)
print("\n订单最终状态:", final_state.values["order_status"])
print("完整历史消息条数:", len(final_state.values["messages"]))
```

#### 运行输出验证
执行上述脚本后，终端输出如下。可以看到流程在触发 `human_review` 后成功挂起，人工注入命令后唤醒并完整保留了全部消息历史：

```text
=== 第一阶段：启动工作流 ===
事件产生: {'risk_assessment': {'messages': [AIMessage(content='风控扫描完成...')], 'risk_level': 'HIGH'}}

当前流程挂起节点: ('human_review',)
中断捕获的载荷: {'prompt': '请审查此订单并决定是否放行', 'risk': 'HIGH'}

=== 第二阶段：人工通过审批并恢复 ===
恢复后事件: {'human_review': {'order_status': 'APPROVED'}}
恢复后事件: {'execution': {'messages': [AIMessage(content='订单最终处理完毕，状态为: APPROVED')]}}

订单最终状态: APPROVED
完整历史消息条数: 3
```

---

### 常见工程陷阱与设计边界

在将状态机应用到复杂系统时，需要防范以下三类典型问题：

![状态机设计中的典型并发冲突与序列化开销陷阱](/images/content-rebuild/9cefdf4c75cd9cca/fig-6.png)

1. **大对象直接塞进状态导致序列化开销过大**：
   - **原因**：Checkpointer 在每个超步结束时都会做一次完整的状态快照序列化。如果直接将几十兆的 PDF 二进制内容、超长网页原始 HTML 或大型向量矩阵放入状态，每一步保存都需要消耗大量 CPU 与 I/O 时间。
   - **解法**：状态中只保存对象的元数据引用（如 S3 文件路径、文档 ID），大文件通过外部存储挂载。
2. **并发分支写冲突（Write-Conflict）**：
   - **原因**：在同一个超步中，如果节点 A 和节点 B 并行执行，且同时修改未指定 Reducer 的普通标量字段，状态收敛阶段无法判定覆盖顺序，会直接报错崩溃。
   - **解法**：对于任何可能被并发写入的字段，必须声明合并规则（如使用列表规约追加，或拆分为不同命名字段）。
3. **循环超步死锁（Infinite Loop）**：
   - **原因**：如果条件分支依赖的模型判断反复无法满足退出条件，图会在多个节点之间循环空转，直到耗尽调用限额。
   - **解法**：在状态中增加 `step_count: int` 计数器字段，并在条件边上强制配置最大跳转限制（或者使用框架自带的 `recursion_limit`）。

---

### 课后实践练习

为了检验你对状态机设计和断点恢复的掌握程度，建议完成以下两个实验：

1. **测试人工驳回路径**：修改示例中第二阶段的 `resume_cmd`，将 `action` 改为 `"REJECT"`，验证执行节点输出的订单状态是否正确变为 `REJECTED`。
2. **测试未配置 Reducer 的数据丢失**：将 `OrderWorkflowState` 中的 `messages` 类型从 `Annotated[list[AnyMessage], add_messages]` 改为普通的 `list[AnyMessage]`，观察重新运行脚本时，最终历史消息条数是否从 3 条降为 1 条。

掌握状态机的结构与断点机制后，我们就具备了构建可靠长链路应用的数据基础。下一讲我们将讨论高阶协作模式 [/llm-100/agent-engineering/planner-executor-reflection.md](/llm-100/agent-engineering/planner-executor-reflection.md)，学习如何基于结构化状态构建 Planner、Executor 与 Reflection 的协同机制；对于工具调用失败时的降级方案，可参考 [/llm-100/agent-engineering/agent-tool-failure.md](/llm-100/agent-engineering/agent-tool-failure.md)；生产环境的系统级稳定性保障策略可阅读 [/llm-100/agent-engineering/agent-reliability.md](/llm-100/agent-engineering/agent-reliability.md)。
