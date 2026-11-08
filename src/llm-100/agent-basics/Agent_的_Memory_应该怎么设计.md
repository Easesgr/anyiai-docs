---
title: Agent 的工作记忆如何设计？长任务中的上下文与状态流转
icon: mdi:cpu-64-bit
date: 2026-07-16
tag:
- 大模型100讲
- Agent
description: 解析 Agent 运行期工作记忆的设计方法，剖析消息配对修剪、状态机解耦、检查点持久化与人机协同的落地机制与边界。
---

大语言模型每次调用都是独立的无状态请求。我们常说的 Agent **工作记忆（Working Memory）**，并不是模型内部持久保存的记忆区，而是宿主程序在每次请求前，根据任务进展收集并动态装配到上下文窗口（Context Window）中的一段信息。

如果仅使用一个 Python 列表不断追加消息（`messages.append(...)`），随着任务轮数增加，系统很快会遇到三个具体问题：

1. **首字延迟与调用成本上升**：Prefill 阶段的计算量与输入 Token 长度直接相关，每次调用都全量附带历史，费用和响应时间都会显著增长。
2. **中间日志挤占窗口**：工具返回的成百上千行代码、控制台日志或 JSON 数据迅速占满上下文空间。
3. **有效注意力衰减**：在长报文中，模型关注特定约束和早期目标的能力会出现经验性下降，容易忽略初始限制或偏离主线。

本讲聚焦于单任务执行期内的**工作状态编排与上下文治理**。跨越独立会话的长期存储将在下一讲 [/llm-100/agent-basics/短期记忆_长期记忆_向量记忆分别是什么.md](/llm-100/agent-basics/短期记忆_长期记忆_向量记忆分别是什么.md) 中展开。

![Agent 状态与工作记忆设计封面](/images/content-rebuild/8a079eaa1ba492ad/cover.png)

---

## 一个贯穿例子：自动化代码修复 Agent

为了直观理解状态流转，我们来看一个贯穿全程的工程场景：**缺陷自动修复 Agent（Bug-Fix Agent）**。

用户指派任务：“修复 issue #102 的除以零异常”。Agent 的典型工作流程包括：

1. 读取报错日志并定位代码文件；
2. 运行本地测试命令（工具输出 500 行终端日志）；
3. 修改目标代码文件；
4. 再次运行测试，若失败则重试（最多重试 3 次）；
5. 测试全部通过后，准备推送代码并提交 Pull Request（需要人工确认）。

如果把所有步骤的对话、上千行测试输出、重试过程全塞在一个扁平的消息列表里，Agent 很容易在第 3 轮时数不清自己到底重试了几次，或者因为满屏的终端报错而丢失最初的业务需求。

---

## 1. 交互流水与结构化状态分离

要保障执行稳定，首先需要区分两类性质完全不同的数据：

* **交互流水（Messages Stream）**：用户输入的自然语言、模型的推导步骤、工具调用的请求与返回报文。这是模型用来理解语境和进行单步推理的素材。
* **结构化状态（Working State）**：当前任务的目标标识、已执行的重试次数、已经定位的文件列表、单测是否通过的布尔值等。这是宿主程序进行流程控制的依据。

![状态管理与上下文膨胀对比](/images/content-rebuild/8a079eaa1ba492ad/fig-1.png)

如果将循环终止条件完全交给模型（例如在 Prompt 中写“如果尝试了 3 次就停止”），模型在长上下文中往往无法稳定计数。关于循环退出的风险成因，可参考 [/llm-100/agent-basics/Agent_为什么容易陷入死循环.md](/llm-100/agent-basics/Agent_为什么容易陷入死循环.md)。

合理的做法是在宿主程序中维护一个强类型的结构化字典或对象：由代码记录 `retry_count += 1`；当检测到 `retry_count >= 3` 时，程序直接切断循环走向失败处理。**模型负责推理分析，程序负责流程控制与状态约束。**

---

## 2. 消息列表治理：配对感知与观测截断

在很多简单实现中，开发者习惯用滑动窗口（例如只截取最后 10 条消息 `messages[-10:]`）来控制长度。在带工具调用的 Agent 场景中，这种做法极易触发接口报错。

OpenAI、Anthropic 等模型服务对消息 Schema 有明确校验：如果一条 `assistant` 消息中包含了 `tool_calls`（或 `tool_use`），其后必须紧跟对应 `tool_call_id` 的 `tool` 返回消息。如果截断窗口恰好切在两者之间，服务端会直接拒绝请求并返回 400 校验错误。

![消息修剪机制](/images/content-rebuild/8a079eaa1ba492ad/fig-2.png)

针对消息列表，需要执行更精细的治理策略：

### 2.1 锚定关键消息（Pinning）
系统的全局规则（System Prompt）以及用户最初的任务描述（Initial User Message）必须始终保留在列表首部，不参与常规窗口滚动，确保核心要求不被冲淡。

### 2.2 成对感知修剪（Pair-Aware Pruning）
当触发消息淘汰时，修剪逻辑必须将 `tool_calls` 消息及其对应的 `tool` 结果消息作为一个原子单元整体移除或降级，防止产生单边悬空的孤立调用。

### 2.3 观测结果替换（Observation Truncation）
在贯穿例子中，运行单测后控制台吐出了 500 行日志。模型在随后的一轮中已经提炼出关键结论：“第 42 行发生 ZeroDivisionError”。

此时，早期那 500 行原始文本在后续的逻辑中已经失去了原始细节的必要性。宿主程序可以将该 Tool 消息的内容替换为紧凑的占位标记：

```text
[pytest 执行完毕：发现 1 处失败，第 42 行 ZeroDivisionError。原始 500 行日志已归档]
```

这样既保留了因果链条，又释放了成百上千个无用 Token。

<ReadMoreLock />

---

## 3. 基于状态机（StateGraph）的有状态编排

当任务从简单的单循环演进为包含多步骤、动态分支与异常重试的复杂流程时，单纯依靠 Python 的 `while True` 很难清晰界定执行边界。[Anthropic building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 提出，可靠的系统应区分“预定义代码编排的工作流（Workflows）”与“模型自主决策的动态循环（Agents）”。[LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) 则通过图状态机（StateGraph）提供了将二者结合的实现载体。

![结构化状态机设计](/images/content-rebuild/8a079eaa1ba492ad/fig-3.png)

在状态机中，核心组件包括：

1. **状态模式（State Schema）**：在整个图节点间传递的共享数据结构；
2. **节点（Nodes）**：独立的 Python 函数，接收当前 State，执行确定性逻辑或调用大模型，返回需要更新的状态键值对；
3. **边（Edges）**：连接节点的路径，包括无条件的确定性边和根据状态字段判断走向的条件边（Conditional Edges）。

下面是根据 Bug-Fix Agent 场景编写的最小状态图示例：

```python
from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 1. 定义状态结构：消息流与业务变量分离
class BugFixState(TypedDict):
    # add_messages 会在节点返回新消息时进行增量追加，而非覆盖全列表
    messages: Annotated[Sequence[dict], add_messages]
    retry_count: int
    test_passed: bool

# 2. 定义节点逻辑
def run_tests(state: BugFixState) -> dict:
    # 运行本地测试（此处用模拟逻辑表示）
    # 真实场景中通过 subprocess 调用 pytest
    success = False  # 假设本次测试未通过
    # 由工作流代码节点直接执行的测试结果，作为系统环境反馈（user 角色）追加
    # 避免无对应 tool_call_id 的孤立 tool 消息触发大模型 API 校验 400 错误
    return {
        "messages": [{"role": "user", "content": "单测反馈: pytest 失败: ZeroDivisionError at line 42"}],
        "test_passed": success,
        "retry_count": state.get("retry_count", 0) + 1
    }

def fix_code(state: BugFixState) -> dict:
    # 调用大模型生成修复补丁
    return {
        "messages": [{"role": "assistant", "content": "已根据报错修改 math_utils.py 第 42 行"}]
    }

def check_test_result(state: BugFixState) -> str:
    # 依据状态中的强类型字段进行确定性路由
    if state.get("test_passed", False):
        return "success"
    if state.get("retry_count", 0) >= 3:
        return "exceeded_limit"
    return "retry"

# 3. 构造状态图
workflow = StateGraph(BugFixState)
workflow.add_node("run_tests", run_tests)
workflow.add_node("fix_code", fix_code)

workflow.add_edge(START, "fix_code")
workflow.add_edge("fix_code", "run_tests")

workflow.add_conditional_edges(
    "run_tests",
    check_test_result,
    {
        "retry": "fix_code",
        "success": END,
        "exceeded_limit": END
    }
)

app = workflow.compile()
```

在这个实现中，循环退出逻辑完全由 `check_test_result` 判定，不再依赖大模型输出“我认输了”才停机，杜绝了无限重试引发的账单失控。

---

## 4. 检查点机制（Checkpoints）：持久化与断点续跑

对于需要多步执行的长任务，内存中的变量是极其脆弱的。模型服务限流（Rate Limit）、网络波动或服务重启都会导致执行轨迹完全丢失。

**检查点（Checkpointing）** 解决了执行状态的持久化问题。在状态图的每个执行步结束时，持久化组件将当前 State 的快照以及元数据（线程标识 `thread_id`、版本号、父节点指针）写入存储介质（如 SQLite、PostgreSQL 或 Redis）。

![检查点机制与断点续跑](/images/content-rebuild/8a079eaa1ba492ad/fig-4.png)

检查点机制为工程落地提供了具体支撑：

* **故障恢复**：任务在第 4 步因为网络超时崩溃，系统不需要重新执行前 3 步昂贵的推理与文件定位，直接从最后一次写入的快照加载状态即可继续。
* **多会话隔离**：通过参数 `thread_id` 将并发用户的任务状态严格隔离在不同的存储分区，后端无状态 API 实例可以安全地并发处理不同任务。
* **历史回放与排查**：可以通过指定的历史检查点版本，查看模型在哪一步产生了不合预期的决策，便于对比验证。

---

## 5. 人机协同（Human-in-the-loop）：断点挂起与状态注入

在 Bug-Fix 场景中，当测试全部通过后，Agent 准备执行 `git push` 并提交合并请求。对于这类具有外部影响的操作，通常不允许完全由模型静默执行。

在 Web 服务架构中，人机协同不能简单地依靠代码中的阻塞式 `input()`，因为人工审核可能需要几分钟甚至数小时，长连接和工作线程会被长时间耗尽。

![Human-in-the-loop 机制](/images/content-rebuild/8a079eaa1ba492ad/fig-5.png)

依托状态机与检查点机制，人机协同的流转可以拆解为非阻塞模式：

1. **设置中断（Interrupt）**：在敏感节点（例如 `submit_pr`）前配置静态或动态断点。状态机流转到该节点前，自动持久化当前状态并退出运行，释放计算资源。
2. **外部审核**：审核人员登录管理控制台，根据 `thread_id` 查询当前上下文，查看准备提交的代码变更（Diff）。
3. **状态修改（State Update）**：如果审核人员发现提交信息不规范，可以在前端表单修改 PR 标题与描述，通过 API 更新检查点中的状态变量。
4. **恢复执行（Resume）**：审核人员点击“批准”按钮，后端以相同的 `thread_id` 重新唤起执行图，系统从断点处加载被人工校正过的新状态，继续推进剩余节点。

---

## 6. Token 预算管理与动态摘要

当任务经历多轮交互，修剪单次观测结果后仍然逼近上下文窗口时，需要引入**动态摘要（Summarization）**。

摘要并非无损操作。代码行号、文件精确路径、特定报错代码等细节极易在概括中丢失。因此必须明确分工：**关键细节存储在结构化状态中，概括性语境保留在摘要中**。

![Token 预算分配与动态摘要](/images/content-rebuild/8a079eaa1ba492ad/fig-6.png)

### 6.1 预算分区参考
在设计单次请求报文时，可以按功能对 Token 预算进行区间切分。以下为一个以 8,000 Token 为总上限的规划示例：

* **固定系统指令（System & Tools）**：预留约 1,500 Token，包含固定规范与工具声明；
* **早期压缩摘要（Summary）**：预留约 2,000 Token，存放早期历史的阶段性总结；
* **近期交互消息（Recent Messages）**：预留约 2,500 Token，保留最后 3~4 轮未压缩的原始对话，保障即时语境清晰；
* **生成预留空间（Response Budget）**：预留约 2,000 Token，避免模型输出因撞墙而被强行截断。

### 6.2 摘要触发方式
* **轮数触发（Step-based）**：每执行固定步数（如 5 轮）调用一次小型模型进行总结。实现简单，但如果中间步骤交互文本很短，会产生不必要的调用开销。
* **水位线触发（Watermark-based）**：每次请求前计算当前列表的总 Token 占用。当总数超过安全阈值（例如预算上限的 80%）时，才触发异步节点，将窗口前段的超期消息合并为一条摘要，释放空间。

---

## 7. 可检验练习：实现一个感知配对的安全修剪函数

为了检验对消息治理的掌握，请编写并测试一个纯 Python 消息修剪函数 `prune_messages`。

### 目标要求
1. 输入参数包含消息列表 `messages`（每项为包含 `role`, `content`，可选 `tool_calls`, `tool_call_id` 的字典）以及保留近期消息条数的整数 `keep_last_n`。
2. **保护机制**：保留第一条 `system` 消息和第一条 `user` 消息。
3. **配对完整性**：对于超出保留窗口的早期消息，如果一条 `assistant` 消息包含 `tool_calls`，必须与其对应的 `tool` 结果消息同时移除，严禁留下孤立的 `tool_calls` 或悬空的 `tool` 响应。
4. **长文本替换**：在被保留的 `tool` 消息中，如果 `content` 长度超过 100 字符，自动替换为前 30 字符加上 `...[已截断]`。

### 验证代码框架

```python
def prune_messages(messages: list[dict], keep_last_n: int) -> list[dict]:
    # 请在此处实现你的修剪逻辑
    pass

# --- 验收断言 ---
test_history = [
    {"role": "system", "content": "你是代码助手"},
    {"role": "user", "content": "修复 issue #102"},
    {"role": "assistant", "content": "定位中", "tool_calls": [{"id": "call_1", "name": "read_log"}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "log: " + "x" * 150},
    {"role": "assistant", "content": "正在分析分析结果..."},
    {"role": "user", "content": "请尽快修改"}
]

result = prune_messages(test_history, keep_last_n=2)
# 验收要求：
# 1. result[0] 必须是 system 消息
# 2. result[1] 必须是 user 的 '修复 issue #102'
# 3. 如果保留了 tool 消息且原长度超 100 字符，其 content 应截断为前 30 字符并以 '...[已截断]' 结尾
# 4. 如果移除了 call_1，对应的 tool 消息也必须一并移除，反之亦然
```

---

## 总结与延伸

设计稳健的 Agent 工作记忆，关键在于主动接管执行上下文的控制权：

1. **结构化状态与非结构化流水解耦**：推理靠消息，控制靠变量；
2. **消息列表成对治理**：避免粗暴切片导致孤立 Tool 消息触发 API 校验失败；
3. **基于状态机与检查点构建底座**：借助 LangGraph 等状态框架，让 Agent 具备可持久化、可中断、支持人工审核的能力。

在掌握了单个任务周期内的状态与上下文治理后，当需要跨越多个独立会话记住用户偏好，或从大量知识库中召回上下文时，就需要引入分层存储与向量检索。这些机制将在下一讲 [/llm-100/agent-basics/短期记忆_长期记忆_向量记忆分别是什么.md](/llm-100/agent-basics/短期记忆_长期记忆_向量记忆分别是什么.md) 中展开讨论。
