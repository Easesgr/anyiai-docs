---
title: Agent 框架选型：从自建循环到状态机运行时
icon: mdi:cpu-64-bit
tag:
- Agent
- 工作流
- 多智能体
description: 解析 Agent 框架的调度机制与适用场景，对比自建 ReAct 循环、LangGraph 状态机与多智能体框架的权衡，提供选型指南与生产防护策略。
---

构建大语言模型（LLM）应用时，很多团队的第一反应是引入功能全面的 Agent 框架。然而在实际工程中，框架本身并不增强模型的推理能力；它的核心职责是**为不确定的模型推理补充确定性的执行保障**——包括参数校验、状态管理、执行超时控制、异常重试以及人工接入。

以一个常见的“电商售后退款助手”为例：
- 如果完全用确定性代码编写：面对“上周买的鞋子小了，可以退钱吗”这类自然语言，规则代码难以穷举用户的表达方式；
- 如果完全交由大模型自主处理：模型可能会直接调用退款接口，甚至在没有校验订单状态或用户权限的情况下执行资金划转，且容易因工具调用格式错误陷入死循环；
- 合理的架构设计是：由 LLM 负责理解用户意图并提取结构化参数，而由确定性代码负责身份鉴权、校验退款资格、写入数据库，并在涉及高额退款时主动暂停并通知人工审批。

Anthropic 在工程实践总结《Building effective agents》中，将这类系统划分为两大模式：
- **工作流（Workflows）**：LLM 与工具的执行路径由确定性代码预先编排，适用于步骤可拆解、流程相对固定的业务；
- **自主智能体（Agents）**：LLM 动态决定下一步执行何种操作、调用哪些工具以及何时终止，具备更高的灵活性，但伴随着执行路径不可控与发散的风险。

编排框架与运行时的价值，正是在工作流的确定性与智能体的动态决策之间建立平衡。

![Agent 编排架构全景图展示了从提示词硬编码、轻量循环到图状态机与多智能体拓扑的演进关系](/images/content-rebuild/c31ff19e0b86bbf2/cover.png)

---

## 基于有向图和状态机的编排框架（LangGraph 等架构设计）

以 [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph/overview) 为代表的状态图框架，受到分布式批处理模型 Pregel 的启发，将 Agent 的执行过程抽象为**有向图（Directed Graph）与状态机**。这种设计替代了早期基于链式管道（Linear Chains）的固定调用流程，支持了更复杂的循环重试和条件分支。

![有向图状态机架构图，展示了从 START 节点经计算节点、状态归约到条件分支判断的数据流动过程](/images/content-rebuild/c31ff19e0b86bbf2/fig-1.png)

### 状态、节点与条件边如何配合

在状态机架构中，整个系统由以下四个基本构件驱动：

1. **共享状态（State）**：一个定义明确的数据契约（通常采用 TypedDict 或 Pydantic 定义）。计算图中的所有节点均只从状态中读取所需字段，并返回增量更新字典。
2. **更新归约器（Reducers）**：定义状态更新时的合并规则。例如，对普通状态字段可直接进行值覆盖；而对对话消息列表，则通过累加归约器（如 LangGraph 内置的 `add_messages`）将新消息按 ID 追加或合并到历史列表中，避免状态字段被意外覆盖。
3. **节点（Node）**：独立的 Python 函数。节点既可以是纯确定性业务代码（如调用数据库查询订单、校验用户身份），也可以是一次模型推理步骤。两者在图调度中享有平等的执行地位。
4. **条件边（Conditional Edge）**：一个基于当前状态返回分支名称的路由函数。例如：若模型返回了工具调用参数，路由函数返回 `"execute_tool"`；若模型已生成最终文本回复，路由函数返回 `END` 终结执行。

下面是一个使用 LangGraph 组织“模型判断 - 工具执行”循环的状态图实现：

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    retry_count: int

def call_model(state: AgentState):
    # 模拟模型输出：实际场景中此处调用大模型 API
    # 模型可输出直接文本回复，或输出携带工具调用指令的结构化消息
    return {"messages": [{"role": "assistant", "content": "Tool request"}]}

def router(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
    # 根据模型输出内容判断下一步流向
    if "Tool request" in content:
        return "execute_tool"
    return END

def execute_tool(state: AgentState):
    # 确定性工具执行节点，执行完毕后将工具结果回填到状态中
    return {
        "messages": [{"role": "tool", "content": "Tool executed successfully"}],
        "retry_count": state.get("retry_count", 0) + 1
    }

# 构建计算图
builder = StateGraph(AgentState)
builder.add_node("call_model", call_model)
builder.add_node("execute_tool", execute_tool)

builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model", 
    router, 
    {"execute_tool": "execute_tool", END: END}
)
builder.add_edge("execute_tool", "call_model")

graph = builder.compile()
```

**工程收益**：与仅靠提示词让模型“自行规划下一步”相比，状态机将分支跳转的主导权交给了确定性的路由函数。即使模型输出了预期之外的噪声内容，路由逻辑也能依据格式校验将其引向重试或降级节点，确保整个流程不会偏离预设边界。

---

## 多角色交互与协作型智能体框架（CrewAI、AutoGen 等适用场景）

多智能体框架（如 [CrewAI 文档](https://docs.crewai.com/) 与 [Microsoft AutoGen 文档](https://microsoft.github.io/autogen/)）将应用拆分为多个具备独立角色（Role）、目标（Goal）与工具集合的模型实例，通过相互通信完成较复杂的协作任务。

![多智能体通信拓扑与上下文膨胀机制图，对比了网状广播下的二次方 Token 增长与分层移交模式的开销差异](/images/content-rebuild/c31ff19e0b86bbf2/fig-2.png)

### 协作拓扑与常见模式

- **层级管理模式（Hierarchical）**：由一个主管（Manager）Agent 接收用户总目标并负责任务分解，分派给下属的研究员、编码员等专业 Agent，最后由主管汇总验证。这种结构适合可自顶向下逐层分解的任务。
- **生成者-验证者模式（Generator-Critic）**：一个 Agent 负责起草业务方案或代码，另一个 Agent 仅配备只读验证工具与验收标准进行审查。两者上下文相互隔离，避免单个模型陷入自我确认偏差。
- **广播与群聊模式（Group Chat / Broadcast）**：多个 Agent 围绕同一个会话流发言，每轮发言对所有人可见。多用于发散性的方案讨论与头脑风暴场景。

### 潜在代价：Token 消耗膨胀与注意力稀释

引入多个 Agent 并非无代价的质量提升，其首要工程开销来自通信上下文的增长：

假设有 $N$ 个 Agent 在未经剪枝的广播通信模式下协作，任务经历 $K$ 轮发言，每轮发言平均产生 $\bar{L}$ 个 Token。在该特定全量广播机制假设下（即每轮 1 个 Agent 发言后，其余 $N-1$ 个 Agent 均需独立调用模型处理全量历史以更新状态或评估接话）：
- 第 $k$ 轮广播时，累积的历史上下文长度约为：
  $$\text{Context Size}(k) \approx (k-1) \cdot \bar{L}$$
- 该轮中其余 $N-1$ 个 Agent 处理该广播时，均将该全量历史作为输入上下文发起推理。若不做上下文剪枝，整个任务在 $K$ 轮广播中的总输入 Token 开销约为：
  $$\text{Total Input Tokens} \approx \sum_{k=1}^K (N-1) \cdot (k-1) \bar{L} = \frac{(N-1) K(K-1)}{2} \bar{L} \in \mathcal{O}(N \cdot K^2)$$
（注：若机制为严格单线交替发言且每轮仅当值 Agent 调用模型，则 $K$ 步的总输入开销为 $\frac{K(K-1)}{2} \bar{L} \in \mathcal{O}(K^2)$；若由 $N$ 个 Agent 各轮流发言 $M$ 轮，即 $K = N \cdot M$，总输入开销则为 $\mathcal{O}(N^2 M^2)$。）

**注意力稀释（Context Dilution）**：除了 API 费用随轮次呈二次方上升外，多 Agent 内部频繁的寒暄、冗长中间过程和无效确认会迅速挤占有限的有效上下文空间。核心任务提示词在上下文中的比重被稀释后，模型发生幻觉或格式漂移的概率会随之增加。因此，在没有严格工具隔离与通信修剪机制时，不建议盲目将单个顺序工作流拆散为多角色对话。

---

## 单代理极简运行时与工具驱动模式（轻量自建 ReAct 循环 vs 复杂框架集成）

在很多生产场景中，并不需要引入庞大的第三方框架。Anthropic 工程团队的实践表明，多数工业落地初期往往使用简单、透明的基础循环代码即可满足要求。

![轻量自建 ReAct 循环架构图，拆解了模型调用、工具格式验证、本地执行与步数硬中断的闭环流程](/images/content-rebuild/c31ff19e0b86bbf2/fig-3.png)

### 最小化自建 ReAct 运行时的优势

一个具备基础防御能力的单 Agent ReAct 循环，其核心逻辑可以通过清晰的确定性代码完整实现：

```python
import json
from dataclasses import dataclass
from typing import Callable, Dict, Any

@dataclass
class Tool:
    name: str
    func: Callable
    schema: dict

def robust_agent_loop(
    client: Any, 
    tools: Dict[str, Tool],
    initial_prompt: str,
    max_steps: int = 8
) -> str:
    messages = [{"role": "user", "content": initial_prompt}]
    
    for step in range(max_steps):
        # 1. 发起结构化模型调用
        response = client.chat.completions.create(
            model="model-name",
            messages=messages,
            tools=[t.schema for t in tools.values()]
        )
        message = response.choices[0].message
        messages.append(message)
        
        # 2. 终止条件检查：如果模型没有生成工具调用请求，直接返回文本答复
        if not message.tool_calls:
            return message.content or ""
            
        # 3. 确定性执行工具调用并做异常捕获
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
                tool_obj = tools.get(tool_name)
                if not tool_obj:
                    raise ValueError(f"工具 {tool_name} 未注册")
                observation = tool_obj.func(**tool_args)
            except Exception as e:
                # 捕获异常并将错误原因回填给模型，允许模型在下一步自我纠错
                observation = f"执行错误: {str(e)}"
                
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(observation, ensure_ascii=False)
            })
            
    return "错误: 达到最大步数限制，任务未完成。"
```

### 选型权衡：自建循环 vs 框架集成

- **自建轻量循环**：无框架黑盒依赖，调用栈扁平，上下文与错误处理完全可控。如果业务逻辑属于单线程短流程、无需跨进程持久化恢复，自建循环是更直接、可维护性更高的选择。
- **采用框架运行时**：当业务涉及**跨服务会话中断恢复**、**人工异步审批挂起**、**并行执行分支汇聚**或**分支状态回滚**时，手写代码需要自行维护状态快照与调度引擎，此时引入 LangGraph 等专业图运行时更符合开发效率。

---

## 状态持久化、执行回滚、人工介入（Human-in-the-loop）支持能力对比

在金融扣款、敏感数据变更或外部系统写操作等关键场景中，Agent 不能在无监督状态下全自动执行。生产级运行时的分水岭在于**对状态快照（Checkpoint）与执行中断（Interrupt）的原生支持**。

![检查点持久化与人工介入时序图，展示了状态序列化存盘、系统挂起等待审批以及带状态恢复执行的步骤](/images/content-rebuild/c31ff19e0b86bbf2/fig-4.png)

### 运行时状态治理机制

1. **检查点机制（Checkpointing）**：调度系统在每个计算超步（Superstep）执行完成后，将当前状态序列化写入持久化存储（如 PostgreSQL、SQLite 或 Redis）。当运行实例异常宕机或进程重启时，通过会话唯一标识（`thread_id`）即可重构执行图并从最近的检查点继续运行。
2. **人工介入（Human-in-the-Loop, HITL）**：运行时支持在敏感节点前配置断点挂起（如声明在 `process_refund` 节点前中断）。调度引擎在流转到该节点时主动暂停执行，将当前建议参数持久化存盘，等待人工审计人员通过管理后台批准或修改参数。审批完成后，系统唤醒图并传入外部更新恢复运行。
3. **执行回滚与时间旅行（Time-Travel）**：排查故障或人工纠错时，开发者可以通过读取历史检查点快照，基于特定时间步分叉（Fork）出新的执行分支，就地修正错误的状态数据后重新启动执行，而无需从头重新调用前置的所有大模型步骤。

### 主流框架能力对比

| 框架 / SDK | 核心编排抽象 | 状态持久化机制 | 人工介入（HITL）支持 | 生产监控集成 | 官方入口与适用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LangGraph** | 有向状态图、Pregel 模型 | 原生支持（PostgreSQL / SQLite 检查点存储） | 原生支持（节点前后断点中断、状态查看与就地修改） | 深度集成 LangSmith，支持 OpenTelemetry 导出 | [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph/overview)<br>推荐：高可靠长任务、复杂状态机与分支流程 |
| **OpenAI Agents SDK** | 轻量 Agent 抽象、Handoff 移交 | 依赖应用层自行维护会话状态 | 支持 Guardrails 拦截与审批挂起 | 内置原生 Tracing 链路追踪 | [OpenAI Agents SDK 仓库](https://openai.github.io/openai-agents-python/)<br>推荐：以 OpenAI 模型生态为主的轻量多 Agent 协作 |
| **Microsoft AutoGen** | 对话驱动、事件驱动架构 | 基于消息总线与会话快照存储 | 支持消息拦截与人工参与应答模式 | 支持 OpenTelemetry 链路导出 | [AutoGen 官方文档](https://microsoft.github.io/autogen/)<br>推荐：多角色交互协作研究、原型探索 |
| **CrewAI** | 团队、角色、顺序任务编排 | 支持任务状态留存到 SQLite 等存储 | 支持任务级人工审核开关 | 集成主流可观测性平台 | [CrewAI 官方文档](https://docs.crewai.com/)<br>推荐：角色职责分明的协同研究原型与初创落地 |
| **Semantic Kernel** | 插件（Plugin）、管道、连接器 | 依赖企业存储框架接入自定义上下文 | 需在函数过滤器（Filters）中自行拦截 | 深度集成 Azure Monitor 与 Application Insights | [Semantic Kernel 官方文档](https://learn.microsoft.com/semantic-kernel/)<br>推荐：.NET / C# 技术栈与微软企业级服务集成 |

---

## 框架调试困难度、死循环风险与生产环境可观测性落地考量

Agent 系统在生产环境中最常见的异常，往往不是模型本身拒绝响应，而是**陷入静默死循环或工具振荡**：模型因工具入参校验失败反复重试同一错误格式、在两个语义相近的工具之间来回跳转，或者在自我修正步骤中持续推翻之前的结论。

![生产安全防线与可观测性监控图，呈现了递归熔断、费用阈值和链路追踪瀑布流的协同机制](/images/content-rebuild/c31ff19e0b86bbf2/fig-5.png)

### 生产防御工程规范

1. **最大步数硬上限（Max Recursion Limit）**：无论流程逻辑设计多完善，建议在调度层配置明确的最大步数限制（Recursion Limit，依据业务调用链深度合理配置，常见为 8~25 步）。一旦达到上限，系统立即中断循环并返回兜底提示，防止异常发散时无限调用消耗资源。
2. **调用预算与费用熔断器（Budget Circuit Breaker）**：在系统全局维护单次任务累积消耗的 Token 与成本计数器。当单次任务消耗达到预警阈值时，直接触发中断挂起，禁止无休止调用外部计费接口。
3. **工具幂等性与失败降级**：所有涉及写操作的外部工具接口必须强制要求传入幂等键（Idempotency Key），防止因重试导致重复下单或扣款。当单一工具在单任务内连续抛错达到指定次数（如 3 次），系统应主动向模型提示该工具暂时不可用，并引导流程转入降级处理或人工客服。

### 可观测性（Observability）建设要求

排查 Agent 系统问题时，仅查看终端最终输出无法还原失败原因。通过接入 LangSmith、OpenTelemetry 或自建日志链路，系统应记录每个节点的完整元数据：
- **状态差分（State Diff）**：记录每一执行节点进入与离开时的状态变更，快速定位是哪个节点写入了异常变量；
- **工具入参 Payload 与原始返回值**：排查是否存在工具返回大量冗余 HTML/JSON 挤占上下文窗口的情况；
- **端到端耗时瀑布图**：区分网络传输等待、模型首字生成（TTFT）、解码生成以及本地业务代码执行的时间占比。

---

## 选型决策树与工程落地检查清单

![Agent 运行时技术选型决策树，根据任务确定性、状态复杂度与审计需求引导开发者选择适配的架构方案](/images/content-rebuild/c31ff19e0b86bbf2/fig-6.png)

### 架构决策路径

在启动开发前，可依据任务确定性与状态复杂度进行评估：
1. **输入与业务链路完全确定？** $\to$ 采用常规业务代码或确定性工作流，无需引入大模型决策 Agent。
2. **仅需一次信息检索并组织语言回答？** $\to$ 采用标准检索增强生成架构（参见 [RAG 框架与选型边界](/resources/rag-framework.md)）。
3. **单线顺序调用几个工具，无需跨会话暂停？** $\to$ 采用自建轻量 ReAct 循环，保持架构简单、排查透明。
4. **存在多分支循环、步骤回环、持久化检查点与人工审批中断？** $\to$ 选用 LangGraph 等基于有向图与状态机机制的运行时。
5. **涉及多个独立业务部门间的权限隔离与协作分派？** $\to$ 选用 CrewAI 或 AutoGen 等多智能体协作框架，但必须建立严格的上下文剪枝规则，避免通信膨胀。

---

## 关联主题与延伸阅读

- **检索增强架构**：深入理解文档索引与生成式问答链路，参见 [RAG 框架与选型边界](/resources/rag-framework.md) 与 [Agentic RAG 复杂查询规划](/frontier/agentic-rag/complex-query-planning.md)。
- **通信与工具协议**：了解跨环境标准工具调用接口实现，参见 [MCP Server 规范与实践](/resources/mcp-server.md)。
- **智能体上下文与记忆**：探究状态机之上的持久记忆存储架构，参见 [Agent 记忆系统设计](/frontier/agent-memory-system/memory-design.md) 与 [Agent Skills 封装规范](/frontier/agent-skills/skill-encapsulation.md)。
