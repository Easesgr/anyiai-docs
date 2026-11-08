---
title: Agent 如何拆解复杂任务？静态工作流、动态规划与子任务分发
icon: mdi:book-open-page-variant
description: 分析长链条任务单体执行的上下文干扰与误差传递，详解基于状态图的静态工作流、运行时动态依赖规划与主控工作者模式，提供上下文隔离与调度实战方案。
---

在构建大语言模型 Agent 时，开发者常会发现：当面对包含多步操作、外部工具调用与格式约束的复合任务时，如果试图把所有要求写在一个超长 Prompt 里一次性生成，模型极易遗漏关键条件、出现幻觉或者陷入格式错误。

在[上一讲探讨的 Planner-Executor-Reflection 架构](/llm-100/agent-engineering/planner-executor-reflection.md)中，我们梳理了“规划—执行—反思”的运行回路。本讲进一步解决工程落地的核心问题：**如何将粗粒度的复杂目标，拆解为边界清晰、可验证、可并发的子任务，并在多节点协作中保持上下文纯净**。

本文通过一个贯穿示例进行说明：构建一个**代码仓库安全与依赖审查助手**。该助手需要接收仓库地址，并行完成依赖项脆弱性分析与代码静态扫描，最后合并输出合规审计报告。

![Agent 任务拆解与分发全景图](/images/content-rebuild/6c5d78cbaa157535/cover.png)

---

### 单体 Prompt 执行复杂任务的瓶颈

在把任务交给大模型自回归生成时，单次请求的上下文窗口承载的内容越多，系统的可控性通常越弱。这种脆弱性主要源于两个具体工程原因：

#### 1. 上下文语义干扰与注意力衰减
如果在一个 Prompt 中同时注入系统设定、5 个以上工具的 JSON Schema、多篇背景参考文档以及分步骤生成的格式规范，模型在注意力分配上面临显著压力。尤其当输入过长时，位于提示词中间位置的约束条件容易被忽略（即常见的“中间遗忘”现象）。例如，模型可能会记住最终的输出格式，却遗漏了“禁止使用非 LTS 版本依赖”的中间约束规则。

#### 2. 误差级联效应（Error Cascading）
如果将多个存在先后依赖的子任务串联在同一个会话上下文中顺序执行，前置步骤产生的细微偏差会被作为确定的事实写入后续步骤的提示词中，导致错误被放大。

假定一个任务链路包含 5 个按顺序依赖的环节，各步骤在相互独立且无上下文污染的理想假设下，单步执行成功率均为 $p = 0.95$。那么整体串联完成的联合成功率为：

$$P_{total} = p^5 = 0.95^5 \approx 0.774$$

> **注**：该算例基于各步骤相互独立的简明假设，用于直观说明乘法效应。在实际工程中，步骤之间往往存在条件依赖；前置步骤一旦出现微小幻觉，后续步骤的实际成功率通常会加速劣化，端到端可用性往往低于独立相乘的计算结果。

长任务拆解的核心价值，在于通过**切断直接上下文依赖**并在关键步骤之间建立**校验与拦截机制**，将全局不确定性限制在局部节点内。

![复杂任务单体执行的注意力分散与误差级联效应示意图](/images/content-rebuild/6c5d78cbaa157535/fig-1.png)

<ReadMoreLock />

---

### 静态拓扑拆解：基于业务规则的有向无环图（DAG）

根据 Anthropic 在 *Building effective agents* 中的分类，智能体系统可以划分为**工作流（Workflows）**与**自主智能体（Agents）**。其中，静态拓扑编排属于典型的工作流模式：**执行路径、依赖关系与并发分支在编译期完全由代码或配置文件预先定义，模型仅负责完成单个节点内部的理解、生成或条件判断**。

#### 1. 静态拓扑的核心特征
* **确定性依赖**：子任务的先后逻辑由开发者预先用代码固化。例如在代码审查助手中，“拉取仓库代码”必须先执行，而“扫描依赖清单”和“运行静态语法分析”彼此独立，可以直接配置为并发节点。
* **分支路由规则化**：条件跳转（如“依赖文件是否存在”、“测试覆盖率是否达标”）通过宿主语言的代码逻辑或状态转移条件处理，而不是交由大模型自由决定流程。

![基于有向无环图（DAG）的静态任务并发与汇聚拓扑结构图](/images/content-rebuild/6c5d78cbaa157535/fig-2.png)

#### 2. 基于 LangGraph 的静态并发实现
在 [LangGraph 受控工作流](/llm-100/agent-engineering/langgraph-controlled-workflow.md) 与 [Agent 状态机设计](/llm-100/agent-engineering/agent-state-machine.md) 中，通过定义全局状态和分支汇聚边，即可实现确定性的并发拆解。以下是代码审查场景中扇出（Fan-out）与汇聚（Fan-in）的实现示例：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. 定义全局状态结构，不同并发节点向各自独立的字段写入数据
class AuditState(TypedDict):
    repo_url: str
    repo_path: str
    dep_vulnerabilities: list[str]
    ast_issues: list[str]
    final_report: str

# 2. 定义独立的执行节点
def clone_repo(state: AuditState) -> dict:
    # 确定性代码操作：拉取代码并返回本地路径
    return {"repo_path": f"/tmp/{state['repo_url'].split('/')[-1]}"}

def audit_dependencies(state: AuditState) -> dict:
    # 子任务 A：依赖项漏洞分析（可调用依赖检查工具或专用模型）
    return {"dep_vulnerabilities": ["CVE-2024-1234: 存在高危原型污染漏洞"]}

def audit_code_ast(state: AuditState) -> dict:
    # 子任务 B：代码静态结构扫描
    return {"ast_issues": ["src/auth.py:28 发现硬编码 JWT 密钥"]}

def generate_report(state: AuditState) -> dict:
    # 汇聚节点：从状态中提取两个并发分支的产物，交给模型合成最终建议
    issues_summary = (
        f"依赖问题：{state['dep_vulnerabilities']}\n"
        f"代码缺陷：{state['ast_issues']}"
    )
    return {"final_report": f"安全审查报告：\n{issues_summary}"}

# 3. 编排静态 DAG 拓扑
builder = StateGraph(AuditState)
builder.add_node("clone_repo", clone_repo)
builder.add_node("audit_dependencies", audit_dependencies)
builder.add_node("audit_code_ast", audit_code_ast)
builder.add_node("generate_report", generate_report)

# 定义执行边
builder.add_edge(START, "clone_repo")

# 静态并发扇出：clone 完成后同时唤起两个独立扫描任务
builder.add_edge("clone_repo", "audit_dependencies")
builder.add_edge("clone_repo", "audit_code_ast")

# 汇聚扇入：两个并发扫描完成后，统一流入报告合成节点
builder.add_edge("audit_dependencies", "generate_report")
builder.add_edge("audit_code_ast", "generate_report")
builder.add_edge("generate_report", END)

graph = builder.compile()
```

这种静态 DAG 架构的优势在于**拓扑结构与调度规则确定、无额外的规划推理耗时与模型调用成本**。尽管节点内的网络请求与模型输出仍具不确定性，但当某个步骤失败（例如代码拉取超时）时，宿主程序可对该节点直接发起重试，而无需重新运行前序或同级任务。

---

### 动态拓扑拆解：模型运行时的子任务规划与调度

当面对无法预先枚举步骤的开放式场景（例如“调查线上某个微服务的异常堆栈并修复潜在缺陷”，开发者事先不知道需要看哪些日志文件、查哪个数据库表），静态写死分支便难以应对。此时需要由大模型充当“规划器”，在运行时动态生成子任务依赖图。

#### 1. 结构化规划输出规范
为了让调度引擎能够执行大模型的规划方案，不能让模型输出自由文本，必须强制其输出带有唯一标识与依赖关系的结构化数据格式：

```json
{
  "plan_id": "audit_plan_001",
  "tasks": [
    {
      "id": "task_1",
      "action": "clone_repo",
      "parameters": {"url": "https://github.com/example/repo"},
      "dependencies": []
    },
    {
      "id": "task_2",
      "action": "audit_dependencies",
      "parameters": {"manifest": "requirements.txt"},
      "dependencies": ["task_1"]
    },
    {
      "id": "task_3",
      "action": "scan_python_files",
      "parameters": {"directory": "./src"},
      "dependencies": ["task_1"]
    },
    {
      "id": "task_4",
      "action": "generate_report",
      "parameters": {},
      "dependencies": ["task_2", "task_3"]
    }
  ]
}
```

![大模型在运行时动态生成子任务依赖并进行拓扑排序调度流程](/images/content-rebuild/6c5d78cbaa157535/fig-3.png)

#### 2. 调度执行校验与安全防御
程序接收到模型生成的 JSON 规划后，必须经过校验器验证才能进入执行调度：
1. **成环检测（Cycle Detection）**：模型生成的依赖列表可能存在死循环（例如 A 依赖 B，B 依赖 A）。调度程序必须使用拓扑排序（如 Kahn 算法）检验其是否为合法的 DAG。一旦检测到环路，应直接打回重试并附带具体的错误信息。
2. **入度计算与并发推进**：在拓扑排序中，初始入度为 0 的任务构成首个并发批次。当某个任务执行完毕，调度器将其产物记入结果表，并将后续依赖该任务的节点的入度减 1。新产生的入度为 0 的节点立刻进入下一轮并发池。
3. **拆解规模限制（Guardrails）**：为防止模型生成不可控的任务深度，调度层需设置硬约束（如单次任务总数上限 $N \le 10$，依赖深度上限 $Depth \le 4$）。超过上限则触发截断或降级策略。

---

### 主控与工作者模式（Orchestrator-Workers）

在动态任务规划落地中，**Orchestrator-Workers（主控-工作者）模式**是隔离复杂度与职责的通用架构。

![主控与工作者模式（Orchestrator-Workers）的派发与汇总数据链路](/images/content-rebuild/6c5d78cbaa157535/fig-4.png)

#### 1. 角色权责划分
* **Orchestrator（主控智能体）**：
  * 负责理解用户的原始输入，拆解生成子任务规划，并维护整个任务的状态字典。
  * 决定当前批次派发哪些子任务，以及任务失败后的重试或重新规划策略。
  * **通常不直接挂载大量底层业务工具**，避免工具定义过多降低规划质量。
* **Worker（工作者智能体）**：
  * 接收主控智能体打包下发的具体任务指令与精简参数。
  * **仅搭载当前子任务必需的专用工具**（例如代码审计 Worker 只赋予静态文件读取权限，不暴露外部网络访问工具）。
  * 输出符合特定 Schema 规范的产物并返回给主控，不与其他 Worker 发生直接交互。

#### 2. 调度推进的伪代码实现
以下伪代码展示了 Orchestrator 驱动 Worker 派发与汇总的基本逻辑，该机制在工程上与系统的[异步任务调度机制](/architecture/async-tasks.md)和[长任务执行设计](/architecture/long-task-execution.md)紧密配合：

```python
# 伪代码：主控调度器与 Worker 派发机制示意
class TaskPlan:
    def validate_acyclic(self) -> bool: ...
    def is_completed(self) -> bool: ...
    def get_ready_tasks(self, completed_ids: set) -> list: ...
    def mark_completed(self, task_id: str): ...

class Orchestrator:
    def __init__(self, planner_llm, workers_map: dict):
        self.planner = planner_llm
        self.workers = workers_map

    def run(self, user_prompt: str) -> str:
        # 1. 规划子任务拓扑并校验
        plan = self.planner.generate_plan(user_prompt)
        if not plan.validate_acyclic():
            raise ValueError("生成的任务图存在循环依赖，拒绝执行")

        completed_results = {}

        # 2. 循环调度入度为 0 的任务
        while not plan.is_completed():
            ready_tasks = plan.get_ready_tasks(completed_ids=set(completed_results.keys()))
            if not ready_tasks:
                raise RuntimeError("依赖阻塞且未完成全部任务，流程中断")

            # 并发派发执行批次
            batch_outputs = self._dispatch_parallel(ready_tasks, completed_results)
            for task_id, output in batch_outputs.items():
                completed_results[task_id] = output
                plan.mark_completed(task_id)

        # 3. 最终由主控完成结果整合
        return self.planner.synthesize(user_prompt, completed_results)

    def _dispatch_parallel(self, tasks: list, completed_results: dict) -> dict:
        results = {}
        for task in tasks:
            worker = self.workers[task.worker_type]
            # 仅投影该任务声明依赖的前序产物，不传入全局全量结果
            task_context = {dep: completed_results[dep] for dep in task.dependencies if dep in completed_results}
            results[task.id] = worker.execute(task.payload, task_context)
        return results
```

---

### 子智能体的上下文隔离：避免全量历史广播

在设计多 Agent 协作时，容易出现一种直觉实现：将主会话中的所有上下文消息（Full Conversation History）完整复制给每一个子 Agent。这种做法在复杂场景下存在明显的工程缺陷：

1. **噪声扩散与提示词干扰**：执行“代码安全检查”的 Worker，如果能看到前面数千 Token 关于商业背景和技术讨论的聊天记录，模型的注意力极易被分散，导致代码分析的严格程度下降。
2. **Token 开销膨胀**：如果每个 Worker 都在重复承载全局长历史，多次并发调用会导致输入 Token 成本成倍攀升，首字延迟（TTFT）随之显著增加。
3. **安全风险传递**：某一个 Worker 如果从外部网页读取了含有潜在 Prompt 注入的代码片段，若直接广播到全局历史，可能影响后续其他 Worker 的判定逻辑。

![子智能体全量历史广播污染与上下文投影隔离机制对比图](/images/content-rebuild/6c5d78cbaa157535/fig-5.png)

#### 上下文投影（Context Projection）
工程上推荐采用**上下文投影（Context Projection）**策略：为每个 Worker 建立独立的、短暂的会话上下文。Worker 只能接收完成本任务所必须的参数输入，其执行完成后的输出也必须通过结构化 Schema 验证，再写回主控的全局状态：

```python
def project_worker_input(task_spec: dict, global_artifacts: dict) -> list[dict]:
    """
    为特定 Worker 构建独立的、极简的提示词列表
    """
    system_message = {
        "role": "system",
        "content": (
            f"你是一个专用于执行 {task_spec['type']} 任务的分析组件。"
            f"请仅依据输入参数分析并返回结构化结果，不要输出无关推论。"
        )
    }
    
    # 仅提取该任务所声明依赖的产物，杜绝全局无关信息
    dependent_data = {
        dep_id: global_artifacts[dep_id]
        for dep_id in task_spec.get("depends_on", [])
        if dep_id in global_artifacts
    }
    
    user_message = {
        "role": "user",
        "content": str({
            "instruction": task_spec["instruction"],
            "context_data": dependent_data,
            "expected_format": task_spec["schema"]
        })
    }
    
    return [system_message, user_message]
```

通过该机制，Worker 节点之间完全解耦。它们仅通过主控定义的数据契约传递计算产物，从而切断了长上下文带来的错误传导链路。

---

### 静态拓扑与动态规划的工程选型

在实际系统搭建中，应根据业务确定性与自由度要求选择合适的拆解机制：

| 评估维度 | 静态 DAG 拓扑拆解 | 动态运行时自发拆解 |
| :--- | :--- | :--- |
| **系统确定性** | 高：路径在上线前已被完全固化，执行分支可被单测覆盖 | 中低：依赖模型实时规划，存在生成畸形图或死循环的可能 |
| **响应延迟** | 低：无需额外规划推理轮次，就绪节点可立即并发推进 | 较高：必须等待模型完成规划并经拓扑排序检验后方可启动 |
| **调试与排障成本** | 低：每个节点输入输出契约明确，易于断点重放和定位 | 较高：子任务数量和依赖随用户输入动态变化，问题复现成本高 |
| **未知场景适应力** | 低：无法处理代码中未枚举的业务路径，需人工扩展代码 | 高：能够根据开放式目标自主分析、拆解并探索未知路径 |
| **推荐应用场景** | 规范审批流、固定数据提取、标准化运维巡检脚本 | 开放式调研问卷分析、未知代码库缺陷诊断、自主探索系统 |

![静态拓扑编排与动态自发拆解的多维工程权衡矩阵雷达图](/images/content-rebuild/6c5d78cbaa157535/fig-6.png)

#### 生产推荐方案：外层静态骨架 + 局部动态自主
在生产系统中，通常不建议构建“从输入到输出完全放手由模型自行规划”的纯动态系统，主流做法是采用分层结合的架构：
* **外层管线保持静态**：核心主干（例如：用户意图分类 $\rightarrow$ 参数鉴权 $\rightarrow$ 核心求解 $\rightarrow$ 合规安全审查 $\rightarrow$ 落库持久化）使用静态 DAG 固化，确保流程处于宿主程序的严格控制之下。
* **局部节点内嵌动态**：在“核心求解”节点内部，针对开放场景挂载动态 Orchestrator-Workers 子图，赋予其在局部工具集与最大步数约束下的自发拆解能力。若子图执行超时或规划异常，外层静态状态机能够捕获并切入兜底规则。

在接下来的章节中，我们将进一步探讨[外部物理工具的挂载与通信协议](/llm-100/agent-engineering/agent-tool-integration.md)，详细解析这些被拆解出来的具体子任务如何安全、受控地与外部真实世界完成交互。

---

### 课后实践练习

**目标**：编写一个轻量任务依赖校验器与并发批次调度器（不需要调用模型 API，专注于图拓扑调度算法验证）。

**待测试数据**：
```python
raw_tasks = [
    {"id": "A", "dependencies": []},
    {"id": "B", "dependencies": ["A"]},
    {"id": "C", "dependencies": ["A"]},
    {"id": "D", "dependencies": ["B", "C"]},
]
```

**检验标准**：
1. 实现成环检测：传入包含循环依赖的数据（例如新增 `{"id": "E", "dependencies": ["D"]}` 且让 `A` 依赖 `E`），算法需抛出异常并指出依赖成环。
2. 实现并发批次划分：对上述合法输入 `raw_tasks` 进行拓扑分层调度，输出的执行批次应为：
   * 第 1 批并发：`['A']`
   * 第 2 批并发：`['B', 'C']`
   * 第 3 批并发：`['D']`
