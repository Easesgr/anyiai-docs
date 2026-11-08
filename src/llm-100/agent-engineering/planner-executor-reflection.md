---
title: Planner、Executor 与 Reflection 分别负责什么？如何协同构建多步 Agent？
icon: mdi:book-open-page-variant
description: 梳理 Planner、Executor 与 Reflection 的职责边界、数据契约与状态流转机制，解决长链路中的目标偏离与反思死循环。
---

在构建多步骤 Agent 时，许多开发者常遇到两类典型问题：一是随着工具调用次数增加，模型逐渐偏离最初目标，在冗长的 API 响应中迷失；二是当某个工具报错或返回空数据时，模型容易反复微调无意义的参数，陷入原地死循环。

在上一讲[《Agent 状态机怎么设计？》](/llm-100/agent-engineering/agent-state-machine.md)中，我们了解了如何用状态、条件边和转移逻辑管理 Agent 行为。当面对较为复杂的长链路任务时，将大模型单一的调用拆分为 **Planner（规划器）**、**Executor（执行器）** 与 **Reflection（反思器/评估器）** 三个协同角色，是目前工程中提升系统稳定性的常见解法。

![Planner-Executor-Reflection 认知闭环整体架构图](/images/content-rebuild/153ed277296d4913/cover.png)

---

### 一个贯穿任务：排查线上服务异常

为了直观理解这三个角色的分工，我们以一个具体的运维排查任务贯穿全文：
> **任务目标**：“排查订单服务 payment-service 过去 1 小时响应延迟突增的原因，并给出分析摘要。”

如果直接让一个大模型在单次会话里自主调用各种监控、日志与数据库接口，上下文往往会迅速被大量无用堆栈、JSON 数据和网络报错填满。而在分工架构中，这个目标会被分配给三个不同职责的节点有序处理：

1. **Planner**：把目标拆分为结构化步骤（查询 APM 监控指标 -> 拉取对应时段错误日志 -> 提取 Top 3 异常错误类型），明确前后依赖与验收标准。
2. **Executor**：每次只认领一个子步骤，带着精简的上下文调用对应工具（例如调用 Prometheus 查询指标），返回局部结果。
3. **Reflection**：比对当前执行结果与既定标准。如果数据符合预期，推进下一步；如果发现工具报错“日志索引不存在”，分析原因并决定局部重试或退回重规划。

---

### 为什么单纯的单步循环容易出现瓶颈？

在常见的 ReAct（Reasoning + Acting）模式中，大模型在同一个上下文中交替执行“思考-行动-观察”。这种模式适合交互轮数少、依赖明确的短任务。但当步骤增多时，扁平单上下文容易遇到以下困难：

1. **目标偏离**：工具返回的原始数据（如长篇 API 返回值、网络响应头）不断堆入上下文。模型的注意力分布会被最近追加的细节牵引，逐渐忘记顶层任务目标与约束。
2. **局部死胡同**：当某一工具持续返回错误时，模型仅依据最近的上下文，倾向于在同一个工具上反复修改参数重试，难以站在全局视角判断“该接口当前不可用，应该换备用方案”。

![单步 ReAct 与解耦闭环架构的上下文演进对比](/images/content-rebuild/153ed277296d4913/fig-1.png)

解决思路是将**全局规划**、**单步执行**与**质量验收**拆分成独立的节点或提示词，通过受控的全局状态传递数据，保持每个节点的上下文清爽聚焦。

---

### Planner 模块：目标分解与任务契约

Planner 的核心职责是：**接收全局目标与已知环境信息，产出带依赖关系和验收标准的结构化任务图**。

Planner 通常不直接调用底层业务工具，因此它的上下文不会被底层的原始调试日志污染。它的输出不是自由格式的 Markdown 清单，而是强类型的任务数据契约：

```json
{
  "plan_id": "plan_payment_latency_01",
  "tasks": [
    {
      "id": "task_1",
      "title": "获取过去1小时 payment-service 的 P99 延迟与 QPS 曲线",
      "tool_hint": "query_prometheus",
      "depends_on": [],
      "acceptance_criteria": "返回包含 p99 延迟与 qps 时间序列的数据点，确认延迟突增的时间点"
    },
    {
      "id": "task_2",
      "title": "抓取异常时间段的错误日志样本",
      "tool_hint": "query_loki_logs",
      "depends_on": ["task_1"],
      "acceptance_criteria": "获取延迟突增窗口内日志级别为 ERROR 的堆栈样本，至少包含1条有效堆栈"
    },
    {
      "id": "task_3",
      "title": "聚合异常根因并产出诊断摘要",
      "tool_hint": "summarize_findings",
      "depends_on": ["task_1", "task_2"],
      "acceptance_criteria": "结合延迟突增时间和错误堆栈，给出明确的问题排查结论"
    }
  ]
}
```

关键字段说明：
* `depends_on`：标记任务的前置依赖，构成任务的有向无环图（DAG），调度器据此决定哪些任务可以并行或按序推进。
* `acceptance_criteria`：验收标准，它直接作为后续 Reflection 节点的核验依据。

![Planner 目标分解与任务 DAG 产出机制](/images/content-rebuild/153ed277296d4913/fig-2.png)

关于如何设计具体的拆解提示词与依赖策略，在下一讲[《Agent 如何进行任务拆解？》](/llm-100/agent-engineering/agent-task-decomposition.md)中会进一步说明。

---

### Executor 模块：上下文隔离与原子执行

Executor 只处理调度器分派给它的**单个当前任务**。它的设计原则是“上下文卫生（Context Hygiene）”：只给它完成当前任务所必需的信息，防止漫长的历史信息干扰工具调用。

输入到 Executor 的上下文通常只包含三部分：
1. 当前子任务的目标与 `acceptance_criteria`；
2. 该任务依赖的前置任务产出摘要（而不是上游全部原始对话）；
3. 本次执行所允许使用的工具声明。

```python
def build_executor_context(current_task: dict, state: dict) -> dict:
    # 仅提取该任务声明依赖的前置输出摘要
    dependency_summaries = {
        dep_id: state["task_results"][dep_id]["summary"]
        for dep_id in current_task.get("depends_on", [])
        if dep_id in state["task_results"]
    }
    return {
        "task_title": current_task["title"],
        "acceptance_criteria": current_task["acceptance_criteria"],
        "upstream_summaries": dependency_summaries
    }
```

Executor 接收到输入后，调用对应工具获取结果。无论工具返回的原始响应多么冗长，Executor 的终点是将结果整理为结构化的局部产出（包括结论摘要和关键字段），再写回状态机。

![Executor 隔离上下文并调用工具产出局部结果](/images/content-rebuild/153ed277296d4913/fig-3.png)

当执行过程中遇到网络超时或工具接口报错时，状态机在节点内的重试与回退策略可参考[《Agent 如何处理工具调用失败？》](/llm-100/agent-engineering/agent-tool-failure.md)。

---

### Reflection 模块：基于标准的比对与归因

如果由 Executor 自身来评估“任务是否完成”，模型往往存在自洽偏误：即使某条字段为空或报错未解决，模型在自我评判时也经常生成“任务已成功完成”的乐观判断。

因此，在关键步骤上，**Reflection（反思/评估器）应作为独立节点与独立提示词运行**。Reflection 节点的输入是一组对照材料：
* **标准面**：Planner 定义的 `acceptance_criteria`。
* **事实面**：Executor 提交的实际产出。

Reflection 节点输出判断结论与结构化归因：

```json
{
  "task_id": "task_2",
  "status": "REJECTED",
  "reason": "未拉取到突增时间段（14:30-14:40）的日志，查询区间错误地指定为了当前时间",
  "root_cause": "QUERY_TIME_RANGE_MISMATCH",
  "suggested_remedy": "参考上游 task_1 发现的突增起始点 14:30:00，将查询时间窗设为 14:25 至 14:45"
}
```

通过将“执行”与“评判”解耦，系统能够以客观的验收契约过滤掉不合格的执行输出。未通过的任务将带着归因建议返回调度，避免无效产出传递到下游。

![Reflection 节点基于契约的比对与归因流程](/images/content-rebuild/153ed277296d4913/fig-4.png)

---

### 状态图拓扑流转与代码实现

根据 [Anthropic 关于构建高效 Agent 的架构指引](https://www.anthropic.com/engineering/building-effective-agents) 以及 [LangGraph 核心设计理念](https://docs.langchain.com/oss/python/langgraph/overview)，将确定性状态控制（代码流程分支）与大模型的动态推理能力结合，是提高 Agent 可靠性的常见做法。

三者在状态机中的典型流转拓扑如下：

1. **Planner 节点**：生成任务列表与依赖关系。
2. **调度路由**：取出下一个就绪（前置依赖均已满足）的任务。
3. **Executor 节点**：组装局部上下文，调用工具并生成产物。
4. **Reflection 节点**：对照标准进行核验。
5. **条件分支流转**：
   * **验收通过**：记录产物，推进到下一个任务。所有任务完成后结束流程。
   * **局部失败，可重试**：将反思建议作为输入，返回 Executor 重新执行当前任务。
   * **前置假设失效**：如工具反馈底层指标库已下线，触发 Re-plan，交由 Planner 调整后续任务路径。

![Plan-Execute-Reflect-Replan 状态机完整流转拓扑](/images/content-rebuild/153ed277296d4913/fig-5.png)

下面是一段自包含、可直接运行的 Python 状态机流转逻辑：

```python
from typing import Dict, List, Literal, TypedDict

class Task(TypedDict):
    id: str
    title: str
    criteria: str

class WorkflowState(TypedDict):
    goal: str
    tasks: List[Task]
    current_task_idx: int
    task_results: Dict[str, dict]
    reflection_feedback: str
    retry_count: int
    status: Literal["planning", "executing", "reflecting", "finished", "aborted"]

def planner_node(state: WorkflowState) -> WorkflowState:
    print(f"[Planner] 正在分解目标: {state['goal']}")
    state["tasks"] = [
        {"id": "t1", "title": "查询 Prometheus P99 延迟", "criteria": "包含有效时间点和延迟数值"},
        {"id": "t2", "title": "提取异常时间段日志", "criteria": "包含至少一条有效堆栈记录"}
    ]
    state["current_task_idx"] = 0
    state["status"] = "executing"
    return state

def executor_node(state: WorkflowState) -> WorkflowState:
    task = state["tasks"][state["current_task_idx"]]
    print(f"[Executor] 正在执行任务 {task['id']}: {task['title']}")
    
    # 模拟工具调用：若有反思反馈则针对性修正，否则初次模拟返回不合规产出
    if state["reflection_feedback"]:
        print(f"  -> 依据反思建议优化参数: {state['reflection_feedback']}")
        if task["id"] == "t1":
            raw_output = "2026-03-30 14:35:00 P99=2350ms, QPS=1200"
        else:
            raw_output = "2026-03-30 14:35:12 ERROR [payment] NullPointerException at OrderService.java:42"
    else:
        raw_output = "连接成功，但未解析到有效字段"

    state["task_results"][task["id"]] = {"summary": raw_output}
    state["status"] = "reflecting"
    return state

def reflection_node(state: WorkflowState) -> WorkflowState:
    task = state["tasks"][state["current_task_idx"]]
    result_summary = state["task_results"][task["id"]]["summary"]
    print(f"[Reflection] 正在核验任务 {task['id']} 的产出是否符合: {task['criteria']}")
    
    # 依据任务的具体验收标准进行核验示意
    if task["id"] == "t1":
        is_passed = "P99=" in result_summary
        failure_hint = "产出缺少关键指标字段，请检查查询语句与时间窗配置"
    else:
        is_passed = "ERROR" in result_summary or "Exception" in result_summary
        failure_hint = "日志中未检索到有效错误堆栈，请调整过滤条件或时间窗"
    
    if is_passed:
        print("  -> 验收通过！")
        state["reflection_feedback"] = ""
        state["retry_count"] = 0
        state["current_task_idx"] += 1
        if state["current_task_idx"] >= len(state["tasks"]):
            state["status"] = "finished"
        else:
            state["status"] = "executing"
    else:
        print("  -> 验收未通过，记录归因建议。")
        state["retry_count"] += 1
        if state["retry_count"] >= 2:
            print("  -> 重试次数超限，流程中止。")
            state["status"] = "aborted"
        else:
            state["reflection_feedback"] = failure_hint
            state["status"] = "executing"
    return state

# 运行模拟测试
if __name__ == "__main__":
    state: WorkflowState = {
        "goal": "排查 payment-service 响应延迟突增",
        "tasks": [],
        "current_task_idx": 0,
        "task_results": {},
        "reflection_feedback": "",
        "retry_count": 0,
        "status": "planning"
    }
    
    while state["status"] not in ["finished", "aborted"]:
        if state["status"] == "planning":
            state = planner_node(state)
        elif state["status"] == "executing":
            state = executor_node(state)
        elif state["status"] == "reflecting":
            state = reflection_node(state)
    
    print(f"\n工作流执行完毕，最终状态: {state['status']}")
```

---

### 反思死循环与工程防御手段

引入反思机制后，系统中最需要警惕的故障形态是**反思死循环（Reflection Death Loop）**：Executor 的产物不符合要求，Reflection 驳回并提出修改意见；Executor 再次调用，但模型仍然生成同质化的错误输出，导致两个节点间不断来回振荡，直到耗尽调用配额。

针对这种问题，需要在状态机中配置确定性的防御措施：

1. **设置重试与重规划上限（Circuit Breaker）**：
   - 单个任务的重试次数建议设定明确阈值（根据具体任务开销配置，常见如 2 到 3 次），避免因模型同质化输出导致死锁。
   - 全局重规划（Re-plan）轮次同样建议配置熔断阈值，超出后及时退出或转入人工接管，防止产生失控账单与无效循环。
2. **逐步升级提示词与策略（Escalation Prompting）**：
   - **初次失败**：仅在提示词中附加 Reflection 的错误说明，允许模型调整参数；
   - **连续失败**：收窄工具参数的选择范围，或切换为确定性更强的降级脚本；必要时切换推理能力更强的模型进行分析。
3. **人工在环断点（Human-in-the-loop Interrupt）**：
   - 当自动重试耗尽时，状态机将状态切换为挂起状态（如 `awaiting_human_input`），并通过通知渠道告警。人工介入查看上下文、修改参数或手动补充必要数据后，再恢复流程。

![反思死循环的演化路径与三重工程防御墙](/images/content-rebuild/153ed277296d4913/fig-6.png)

关于状态与执行轨迹的全局监控方案，读者可参考[《Agent 如何记录执行轨迹和中间状态？》](/llm-100/agent-engineering/agent-execution-tracing.md)。

---

### 动手练习

请修改上述 Python 示例中的 `executor_node` 与 `reflection_node`，完成以下验证：
1. 将初始 `retry_count` 限制放宽到 3 次。
2. 在 `executor_node` 中模拟第二轮重试依然失败、第三轮才成功的场景。
3. 观察控制台输出，确认工作流是否按预期在第三次重试时通过 Reflection，并正确过渡到下一个任务。

---

### 知识复盘与自测

* **问题 1：为什么不建议让 Executor 节点直接自行判断任务是否合格？**
  * *自测比对*：单节点自我核验容易受自洽偏误影响，模型倾向于认为自己的输出正确；将执行与评估拆分为独立提示词与上下文，能依据客观验收标准过滤不合规产物。
* **问题 2：如果 Reflection 反馈某个底层数据源已经彻底下线，应该触发局部重试还是触发 Planner 重规划？**
  * *自测比对*：底层资源失效属于前提假设破裂，在局部微调参数无法解决，应触发重规划（Re-plan），由 Planner 调整后续任务路径或启用替代数据源。
