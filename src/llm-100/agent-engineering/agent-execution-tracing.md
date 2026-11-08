---
title: Agent 如何记录执行轨迹与中间状态？State 快照、Span 追踪与本地调试实战
icon: mdi:book-open-page-variant
description: 详解 Agent 执行轨迹六要素、OpenTelemetry 链路映射、LangSmith 状态差分与 Checkpoint 离线单步重放机制。
---

在开发基于多轮对话和工具调用的 Agent 时，开发者最常碰到的问题是：单次提示词测试表现良好，但在串联多个工具后，最终输出却偏离了预期。如果只看控制台打印的最终文本或错误日志，排查往往无从下手——你很难判断到底是第二步的工具返回了不符合预期的字段，第三步的模型推理误解了上下文，还是图状态中的关键变量被后续节点意外覆盖。

在[构建可控 Agent 流程](/llm-100/agent-engineering/langgraph-controlled-workflow.md)与[状态机架构](/llm-100/agent-engineering/agent-state-machine.md)中，系统不再是简单的输入到输出映射，而是一个包含多轮状态转移、动态分支与外部交互的执行流。本文以一个电商售后处理 Agent 为贯穿案例，讲解如何通过记录执行轨迹（Trajectory）、划分 OpenTelemetry 链路（Span）、对比状态快照（State Snapshot），以及利用持久化检查点（Checkpoint）在本地单步复现并修复故障。

![Agent 执行轨迹与可观测性全貌](/images/content-rebuild/e42b9eee5ac2b782/cover.png)

---

### 1. 传统日志为何难以排查 Agent 故障？

以处理“用户申请退款 ORD-9921”为例，Agent 正常需要执行以下步骤：
1. 提取订单编号 `ORD-9921`；
2. 调用 `query_order` 查询物流与签收时间；
3. 结合“7 天无理由退货”规则判断是否符合退款条件；
4. 调用 `apply_refund` 写入退款申请；
5. 向用户生成确认答复。

如果最终用户收到了“您的订单不符合退款条件”的答复，但后台查询显示该订单昨天才签收，传统排障手段通常会遇到以下阻碍：

根据 Anthropic 在 [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) 中的定义，Agentic 系统的特点在于模型能够动态决定自身的过程与工具使用。这种灵活性带来了三项独特的排障难题：

1. **分支由模型动态决定**：传统的服务路由由代码中的 `if-else` 或配置规则控制，而 Agent 的下一步调用取决于模型对上一轮输出的理解。微小的提示词调整或检索内容变化，都可能让模型转向不同的工具调用链。
2. **输出合规但语义偏移**：各层 API 返回的状态码都是 200，中间节点的 JSON 结构也符合 Schema，但模型可能将物流状态中的 `SHIPPED`（已发货）误判为 `EXPIRED`（已逾期）。这类静默错误无法通过异常拦截捕获。
3. **状态转移缺乏记录**：Agent 的每次决策不仅依赖上一轮输出，还依赖上下文中的共享状态字典。如果只看 HTTP 报文，无法还原当前决策时刻上下文中的历史对话被截断到了什么程度、中间变量存留了什么内容。

![传统微服务链路与 Agent 动态分支对比](/images/content-rebuild/e42b9eee5ac2b782/fig-1.png)

因此，要排查 Agent 的运行问题，核心在于记录并解析完整的**执行轨迹（Trajectory）**。

---

### 2. 执行轨迹的核心元数据结构

执行轨迹不是杂乱的纯文本日志，而是在离散时间步 $t \in [0, N]$ 内对模型思考、操作与状态演化的结构化记录。一个能够支撑调试与归因的执行轨迹步骤通常包含以下六类元数据：

| 元数据维度 | 核心字段 | 工程作用与排查价值 |
| :--- | :--- | :--- |
| **Input** | `system_prompt`, `user_messages`, `context` | 还原模型在当前步骤看到的完整 Prompt，判断是否存在指令被长上下文稀释或污染 |
| **Thought** | `reasoning_content`, `scratchpad` | 捕获模型的推理逻辑，确认模型为何决定调用某个工具或拒绝请求 |
| **Tool Call** | `tool_name`, `arguments`, `call_id` | 记录模型生成的工具调用结构体，核验参数类型与业务入参规范 |
| **Tool Result** | `status`, `output_payload`, `error` | 记录外部系统（API、数据库）的原始返回，检查是否存在字段缺失或下游截断 |
| **State Snapshot** | `state_id`, `state_diff`, `variables` | 当前步骤执行完成后的全局状态快照，明确共享变量在时间线上的赋值变化 |
| **Telemetry** | `latency_ms`, `tokens`, `model` | 统计成本与耗时，精确区分首字生成延迟（TTFT）与外部网络 I/O 阻塞 |

以下是贯穿案例中第 2 步（查询订单物流）执行时的结构化轨迹示例：

```json
{
  "step": 2,
  "trace_id": "trace-ord-7a8f",
  "checkpoint_id": "chk-node-fetch-order-002",
  "node_name": "fetch_order_node",
  "input": {
    "messages": [{"role": "user", "content": "帮我处理 ORD-9921 的退款"}]
  },
  "thought": "用户需要退款，我需要先查验订单 ORD-9921 的发货与签收状态以核实退款资质。",
  "tool_call": {
    "name": "query_order",
    "arguments": {"order_id": "ORD-9921"},
    "call_id": "call_query_9921"
  },
  "tool_result": {
    "status": "success",
    "data": {"order_id": "ORD-9921", "status": "DELIVERED", "delivered_at": "2026-03-28T10:00:00Z"}
  },
  "state_snapshot": {
    "order_info": {"order_id": "ORD-9921", "status": "DELIVERED"},
    "refund_eligible": null,
    "current_step": "EVALUATE_POLICY"
  },
  "metrics": {
    "prompt_tokens": 512,
    "completion_tokens": 42,
    "duration_ms": 620
  }
}
```

![执行轨迹核心元数据结构解析](/images/content-rebuild/e42b9eee5ac2b782/fig-2.png)

只要轨迹中完整保留了这些字段，定位问题就会变得非常直接：
- 如果 `tool_result` 正确返回了签收时间，但在后续评估节点中 `state_snapshot.refund_eligible` 被错误写成了 `false`，说明是后续规则判定或解析提示词出错；
- 如果 `tool_call.arguments` 提取的单号变成了空值，说明是模型参数抽取步骤失败，无需盲目排查后端接口。

<ReadMoreLock />

---

### 3. OpenTelemetry 映射：将节点与工具调用分解为 Span

为了将 Agent 的执行轨迹接入企业现有的链路监控系统（如 Jaeger、Prometheus、DataDog），通常采用 **OpenTelemetry (OTel)** 标准进行链路埋点。

在 OTel 的语义体系中，Agent 的各个运行环节可对应为不同层级的 Span：
1. **Root Span**：代表一次完整的 Agent 执行过程（如 `app.invoke`），记录全链路耗时与最终状态；
2. **Node Span**：代表状态机中的具体计算节点（如 `fetch_order_node`、`evaluate_node`）；
3. **Leaf Span**：代表节点发起的底层原子操作，包括调用大模型 API（LLM Span）或请求外部接口（Tool Span）。

![OpenTelemetry Span 层级映射示意](/images/content-rebuild/e42b9eee5ac2b782/fig-3.png)

下面展示基于 OpenTelemetry SDK 封装工具执行节点的实现方式：

```python
import json
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("agent.tracer")

def call_external_order_api(order_id: str) -> dict:
    # 模拟外部物流接口调用
    return {"order_id": order_id, "status": "DELIVERED", "sign_days": 1}

def execute_tool_node(state: dict, tool_name: str, tool_args: dict) -> dict:
    # 开启当前节点的 Child Span
    with tracer.start_as_current_span(f"tool:{tool_name}") as span:
        # 记录关键业务与调试属性
        span.set_attribute("agent.step.node", "fetch_order_node")
        span.set_attribute("gen_ai.tool.name", tool_name)
        span.set_attribute("gen_ai.tool.arguments", json.dumps(tool_args))
        
        try:
            if tool_name == "query_order":
                result = call_external_order_api(tool_args.get("order_id", ""))
            else:
                raise ValueError(f"未知工具: {tool_name}")
                
            span.set_attribute("gen_ai.tool.result_status", "success")
            span.set_status(Status(StatusCode.OK))
            # 将工具结果写回状态
            return {"order_info": result}
        except Exception as exc:
            # 工具异常记录堆栈，用于后续容错或重试
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise exc
```

配合[工具调用失败处理](/llm-100/agent-engineering/agent-tool-failure.md)机制，通过 Span 中的参数和异常属性，可以在监控看板中按工具名称快速统计各 API 的错误率与调用耗时分布。

---

### 4. LangSmith 可观测机制：状态流转与差分追踪

通用 APM 工具擅长记录请求耗时和错误码，但通常无法解析 Agent 内部的状态字典演进与提示词模板变化。在 LangGraph 体系中，[LangSmith](https://docs.langchain.com/oss/python/langgraph/overview) 提供了图感知追踪能力。

#### 4.1 环境变量配置

无需修改核心业务逻辑代码，通过注入环境变量即可接入追踪：

```bash
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="lsv2_pt_..."
export LANGSMITH_PROJECT="order-agent-prod"
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
```

#### 4.2 运行期状态差分（State Diff）

在 LangGraph 中，状态由各节点返回的增量字典驱动更新。当节点发生故障或变量被覆盖时，通过对比前后快照可以迅速排查：

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    user_input: str
    order_info: dict
    refund_approved: bool

def query_order_node(state: AgentState):
    # 查询接口并更新 order_info
    return {"order_info": {"order_id": "ORD-9921", "status": "DELIVERED"}}

def policy_check_node(state: AgentState):
    # 根据订单信息做出业务判断
    is_delivered = state.get("order_info", {}).get("status") == "DELIVERED"
    return {"refund_approved": is_delivered}

builder = StateGraph(AgentState)
builder.add_node("query_order", query_order_node)
builder.add_node("policy_check", policy_check_node)
builder.add_edge(START, "query_order")
builder.add_edge("query_order", "policy_check")
builder.add_edge("policy_check", END)

app = builder.compile()
result = app.invoke({"user_input": "申请退款 ORD-9921"})
```

![LangSmith 可观测视图与状态审计](/images/content-rebuild/e42b9eee5ac2b782/fig-4.png)

在可视化控制台中，开发者可以查看到：
- **调用树视图**：清晰展示 `START -> query_order -> policy_check -> END` 的执行顺序；
- **State Diff 视窗**：绿色高亮展示 `query_order` 新增的 `order_info`，以及 `policy_check` 新增的 `refund_approved: true`；
- **Token 与耗时统计**：将提示词输入 Token、补全 Token 与节点耗时分别归类，便于针对性做性能优化。

---

### 5. Checkpoint 检查点机制与离线单步重放

在线上排查多步骤故障时，直接从第 1 步重新发起完整请求通常存在两个问题：
1. **成本与耗时**：前置步骤的 LLM 调用和外部网络请求需要重复计费与等待；
2. **执行不确定性**：大模型采样存在随机性，重新请求可能走出不同的调用路径，难以稳定复现第 4 步的 Bug。

LangGraph 的持久化检查点（Checkpointer，如 `SqliteSaver` 或 `PostgresSaver`）在每个节点执行结束后，都会将当前的 State 快照序列化落库。通过绑定 `thread_id` 与 `checkpoint_id`，开发者可以在本地直接加载历史故障快照。

#### 5.1 离线读取与重放实现

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 1. 挂载持久化检查点数据库（支持本地 SQLite 或生产 Postgres 备份文件）
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    debug_app = builder.compile(checkpointer=checkpointer)
    
    # 2. 定位故障会话与特定执行节点
    config = {
        "configurable": {
            "thread_id": "session-refund-001",
            "checkpoint_id": "chk-node-fetch-order-002"
        }
    }
    
    # 3. 提取当刻的历史状态快照
    snapshot = debug_app.get_state(config)
    print("当前快照中的状态值:", snapshot.values)
    print("下一个待执行节点:", snapshot.next)
    
    # 4. 在本地修改完业务逻辑后，直接基于该快照继续执行下一步，无需重跑前序逻辑
    # 传入 None 表示沿用当前状态快照继续执行
    replayed_result = debug_app.invoke(None, config=config)
    print("重放执行结果:", replayed_result)
```

![本地检查点重放与分支调试示意](/images/content-rebuild/e42b9eee5ac2b782/fig-5.png)

**使用边界与注意事项**：
- Checkpoint 精准还原的是**保存在检查点中的图内部状态快照**，并不包含也不会自动回滚外部数据库或第三方 API 的外部状态。如果重放节点包含外部写操作，排查时需注意幂等性与测试环境隔离。
- 代码控制的分支与固定状态输入，并不等于整个工作流具备 100% 确定性；网络抖动、并发竞争、外部依赖状态以及模型的采样波动都会引入变化。若需在本地对节点逻辑进行确定性单步断言，除固化模型输入与随机种子（或 Mock 模型响应）外，还需对外部 I/O 与工具调用实施打桩与隔离。

---

### 6. 从生产 Trace 到回归测试集

可观测性不仅用于事后定位，更是持续维护系统质量的依据。通过将线上的典型错误 Trace 转化为回归测试用例，可以有效防止提示词修改或模型升级引发功能退化。

![轨迹驱动的持续优化飞轮](/images/content-rebuild/e42b9eee5ac2b782/fig-6.png)

具体落地流程分为三个环节：

1. **捕获与脱敏**：筛选线上触发告警（如重试超限、异常退出、用户差评）的 Trace，移除手机号、密钥等个人隐私与敏感信息。
2. **构建黄金基准（Golden Trajectory）**：将输入提取为测试用例，将正确的工具调用序列（例如必须先调 `query_order`，再调 `apply_refund`）和预期的关键状态键值保存为断言基准。
3. **自动化回归**：在 CI 流程中批量执行测试集，断言：
   - 工具选择准确率（是否选中了预期工具）；
   - 参数抽取正确率（参数是否符合类型与业务定义）；
   - 最终状态合规性（业务标记是否准确写入）。

关于系统长周期运行的稳定性防御策略，可进一步参考 [Agent 项目的生产稳定性](/llm-100/agent-engineering/agent-reliability.md)。

---

### 7. 本讲小结与检验练习

#### 学习目标复盘
- 掌握 Agent 执行轨迹（Input、Thought、Tool Call、Tool Result、State Snapshot、Telemetry）的核心组成与排查作用；
- 能够将 Agent 的图节点与工具请求拆解为 OpenTelemetry 的树状 Span 结构；
- 掌握利用 Checkpointer 提取历史状态快照并在本地单步重放的调试方法。

#### 可检验练习
在本地创建一段 Python 脚本，完成以下验证任务：
1. 使用 `TypedDict` 定义一个包含 `step_count: int` 和 `logs: list` 的 `DemoState`；
2. 构建包含两个节点的图：`node_a` 将 `step_count` 加 1，`node_b` 抛出 `RuntimeError("模拟逻辑异常")`；
3. 接入 `MemorySaver` 作为 checkpointer，执行图并捕获异常；
4. 调用 `app.get_state`，打印出异常发生前最后一处成功的 `checkpoint_id` 与 `step_count` 数值，确认状态是否完整保存。
