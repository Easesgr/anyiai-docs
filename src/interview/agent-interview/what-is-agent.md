---
title: AI Agent 是什么？它和工作流有什么区别？
icon: mdi:cpu-64-bit
tag:
- Agent面试
- Agent基础
- ChatBot
description: 从控制流归属、工程代价与选型边界出发，解析 AI Agent 与工作流的核心区别，附面试口述答案与选型决策指南。
---

![AI Agent 与 Workflows 的核心辨析封面](/images/content-rebuild/7f54fad55edd3e4e/cover.png)

在技术面试中，“什么是 AI Agent？它和工作流（Workflows）或传统脚本有什么区别？”是一个高频考题。许多回答容易陷入堆砌“自主思考、自我进化、全自动闭环”等抽象词汇的误区。工程面试的核心不在于背诵科幻定义，而在于讲清**控制流（Control Flow）的主导权在谁手里**、**系统的确定性与延迟成本如何权衡**，以及**在具体业务中如何做技术选型**。

---

## 面试标准口述：1 分钟答出技术分界

面试时被问到“如何定义 AI Agent”时，建议先给出一句工程维度的定性结论，再对比控制流，控制在 1 分钟左右：

> **口述参考答案**：
> **AI Agent 是一种由大语言模型（LLM）动态主导执行步骤与工具调用的计算系统。**
> 它和传统工作流（Workflows）最根本的区别在于**控制流的归属权**：
> - **工作流**是**“代码主导流程”**：所有的分支跳转、执行步骤和重试逻辑都由工程师在代码中预先写死，大模型只是其中的一个计算节点，负责局部的文本处理或特征提取。
> - **AI Agent**是**“模型主导流程”**：代码只提供运行循环与可用工具列表，由大模型根据当前目标和环境反馈，自主决定下一步是调用工具、调整方向还是直接给出最终结论。

![传统自动化控制流 vs Agent 动态控制流对比](/images/content-rebuild/7f54fad55edd3e4e/fig-1.png)

---

## 贯穿示例：排查线上服务器磁盘告警

为了直观理解这两种架构的区别，以一个常见的研发运维场景为例：**系统收到“某台服务器根目录磁盘占用率达 95%”的报警，需要定位原因并尝试恢复**。

### 方式 A：预定义工作流（代码主导）
工程师预先写好流程逻辑：
1. 步骤 1：固定执行命令 `df -h`。
2. 步骤 2：代码用正则解析挂载点，进入 `if-else` 判断：
   - 如果 `/var/log` 占比最高，执行日志归档清理脚本；
   - 如果 `/tmp` 占比最高，清理临时文件；
   - 如果是未知目录，调用大模型分析排查建议，并发送告警给运维人员。

在这个过程中，**每一步往哪里走完全由代码逻辑约束**。即使最后一步调用了大模型，模型也无法改变执行链条的拓扑结构。

### 方式 B：自主智能体（模型主导）
系统只给大模型提供目标“排查磁盘占用过高并尝试安全恢复”，以及一组只读与受控的工具接口（如 `execute_shell_command`、`send_alert`）：
1. **第 1 轮**：模型判断需要看系统状态，主动调用 `execute_shell_command(cmd="df -h")`。
2. **观察反馈**：工具返回输出，显示 `/var/lib/docker` 目录占用了 80GB。
3. **第 2 轮**：模型看到 Docker 目录异常，自主决定下一步排查容器镜像，调用 `execute_shell_command(cmd="docker system df")`。
4. **观察反馈**：工具返回显示有 50GB 未使用的悬空镜像（dangling images）。
5. **第 3 轮**：模型决定调用安全清理工具执行清理，随后再次调用 `df -h` 检查是否生效。
6. **最终输出**：确认磁盘使用率降至 40% 后，生成总结报告退出。

在 Agent 模式下，**没有一条代码预先写明“先查 df 再查 docker”**。大模型根据每一步工具返回的真实环境数据，动态决定下一步动作。

---

## Anthropic 的架构划分：Workflows 与 Agents

根据 [Anthropic 发布的 Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)，广义上结合大模型完成任务的系统统称为 **Agentic Systems**。Anthropic 在架构上把它们分为两类：**Workflows（工作流）** 与 **Agents（智能体）**。

<ReadMoreLock />

![Anthropic 两分法：预定义 Workflows 架构 vs 循环 Agents 架构](/images/content-rebuild/7f54fad55edd3e4e/fig-2.png)

### 1. 常见工作流模式（代码驱动）
在工作流中，开发者提前在代码中编排执行路径：
- **链式（Prompt Chaining）**：将任务拆解为线性的固定先后步骤，前一步的输出作为下一步的输入，中间可插入校验规则（Gate）。
- **路由（Routing）**：由分类器（规则或分类小模型）识别输入意图，将请求分发给指定的分支或提示词。
- **并行化（Parallelization）**：将大任务拆分为子任务并发请求大模型（分块并发），或同一输入并发多路调用比对结果（投票过滤）。
- **编排器-工作者（Orchestrator-Workers）**：中心模型在预设范围内拆解出子任务列表，各个子任务由专有代码或模型并行执行，结果由外部代码负责收集汇总。
- **评估-优化循环（Evaluator-Optimizer）**：一个模型负责生成内容，另一个模型根据明确标准打分反馈，代码循环驱动直到达标或达到最大迭代次数。

### 2. 智能体模式（模型驱动）
Agent 模式将主控权交给模型。典型的执行范式如经典的 [ReAct 原理](/interview/agent-interview/react-principle.md)，模型在单个运行步骤中交替进行思考（Thought）、行动（Action）与观察（Observation）。

```python
# 伪代码：Workflows 与 Agent 控制流的差异

# 模式 A：工作流（代码决定下一步走哪里）
def workflow_disk_check(host_ip: str) -> str:
    df_result = run_ssh_command(host_ip, "df -h")
    target_dir = parse_heavy_mount(df_result)
    
    # 代码硬编码决策分支
    if target_dir == "/var/log":
        return run_clean_log(host_ip)
    elif target_dir == "/var/lib/docker":
        return run_docker_prune(host_ip)
    else:
        # 规则无法处理时，调用 LLM 提供辅助分析
        return call_llm(f"分析该磁盘输出并给出建议：{df_result}")

# 模式 B：Agent（模型在循环中决定下一步）
def agent_disk_troubleshoot(host_ip: str, tools: list) -> str:
    messages = [{"role": "user", "content": f"排查主机 {host_ip} 磁盘满告警并解决"}]
    
    while True:
        # 模型自主决定：是发起工具调用，还是输出排查完成的最终回复
        decision = call_llm_with_tools(messages, tools)
        if decision.is_final_answer:
            return decision.answer
        
        # 宿主系统执行模型申请的工具并回填观察结果
        tool_output = execute_tool(decision.tool_name, decision.tool_args)
        messages.append({"role": "assistant", "content": decision.raw_response})
        messages.append({"role": "tool", "content": str(tool_output)})
```

[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) 等现代编排框架的核心设计，正是支持在同一个状态图（StateGraph）中混合编排确定性的代码节点与由模型控制的 Agentic 节点，兼顾可控性与灵活性。

---

## 工程权衡：延迟、调用成本与错误累积

自主性并不是免费的，它直接伴随着系统不确定性与执行成本的上升。在系统设计中，引入 Agent 需要权衡三项关键工程代价：

![Agent 延迟、Token 成本膨胀与错误累积机制分析](/images/content-rebuild/7f54fad55edd3e4e/fig-3.png)

### 1. 响应延迟累加
在工作流中，许多互不依赖的步骤可以并发执行；而 Agent 依赖“根据上一步观察结果做下一步决策”，各轮之间存在强串行依赖。如果一个 Agent 完成任务需要 6 轮迭代，每轮包含一次大模型推理（Prefill + Decode）和一次外部工具 IO，端到端延迟通常在数十秒甚至数分钟，不适用于对实时性要求高的在线接口。

### 2. 上下文膨胀与计算开销
在运行循环中，历史上的每一次思考、工具调用参数与返回内容（Observation）都会追加到对话历史中。模型在第 $n$ 步做决策时，需要将前面累积的历史上下文全部传入。在未命中前缀缓存（Prompt Cache）的情况下，模型需要对完整历史重新进行前向计算（Prefill 阶段）；即使服务商支持前缀缓存，输入的累积 Prompt Token 总量与内存占用也会随步数持续攀升。关于上下文的截断与压缩策略，可参考 [Agent 上下文管理机制](/interview/agent-interview/context-management.md)。

### 3. 多步复合错误率
在多步骤任务中，模型哪怕某一步决策失误（如选错工具、参数格式错误），都可能导致后续推理基于错误的假设展开。

为量化说明长链条的脆弱性，**假设各步骤相对独立，且每一步成功率均为 $p$（$0 < p < 1$）**，则连续完成 $n$ 步的整体成功率示意公式为：

$$P_{\text{success}} = p^n$$

> **注**：在真实场景中，模型具备一定的重试和纠错能力，步骤之间不满足严格的统计独立性；此处公式仅作为教学示意，用于直观展示长执行链中微小单步误差的复合放大效应。

假设单步准确率 $p = 0.95$：
- 执行 3 步的成功率约为 $0.95^3 \approx 85.7\%$
- 执行 8 步的成功率降至 $0.95^8 \approx 66.3\%$
- 执行 15 步的成功率跌至 $0.95^{15} \approx 46.3\%$

如果缺乏兜底规则或最大步数熔断，深层循环容易陷入无休止的无效尝试或偏离原始目标。针对异常保护策略，详见 [Agent 执行失败时如何恢复？](/interview/agent-interview/agent-error-handling.md)。

---

## 技术选型决策：什么时候用单次调用、工作流或 Agent？

系统设计面试中，给出清晰的权衡决策树比空谈概念更能体现工程经验。选型建议遵循**“如无必要，勿增实体”**原则：

![模型调用、工作流与 Agent 的三级选型决策流程](/images/content-rebuild/7f54fad55edd3e4e/fig-4.png)

### 三级选型对照表

| 评估维度 | 单次模型调用（Single Call） | 预定义工作流（Workflows） | 自主智能体（Agents） |
| :--- | :--- | :--- | :--- |
| **任务路径确定性** | 确定（单次输入单次输出） | 路径固定或条件分支可穷举 | 探索性强，依赖环境动态反馈 |
| **状态控制** | 无状态或单次上下文交互 | 状态转移由代码严格约束 | 由模型在循环中动态更新决策 |
| **平均延迟与开销** | 极低（1 次往返） | 中等（确定步骤或受控并发） | 较高（多轮串行推理与累积上下文） |
| **错误排查与审计** | 最简单，直接复现 Prompt | 容易，步骤节点有明确日志 | 较难，模型每次探索路径可能不同 |
| **典型场景** | 摘要提取、情感分类、单轮答疑 | 合同审查流水线、数据入库清洗 | 开放式代码调试、复杂网络排障 |

### 选型自检路径
1. **能否通过单次调用解决？**
   通过完善系统提示词、增加 Few-shot 示例或标准 RAG 能否稳定达成目标？如果能，不要引入多步编排。
2. **执行流程是否可拆解为明确规则？**
   如果业务有标准作业程序（SOP，例如“先校验参数 -> 查数据库 -> 规则对比 -> 调用模板输出”），应选择 **Workflows**，用代码保证确定性与低延迟。
3. **是否面临未知或动态反馈的开放问题？**
   只有当任务无法预先枚举执行分支、强依赖环境返回结果反复调整试探时，才选择 **Agent**。

---

## 高频追问：长期记忆和规划模块是 Agent 的必要组件吗？

面试官常会追问：*“如果一个系统既没有接入向量数据库做跨会话记忆，也没有写显式的任务分解规划器（Planner），它还能被称为 Agent 吗？”*

![Agent 极简内核与外围增强模块的解耦关系](/images/content-rebuild/7f54fad55edd3e4e/fig-5.png)

### 追问回答要点

1. **核心判定**：
   **长期记忆和显式规划器不是 Agent 的必要条件，它们是处理长周期任务时的增强模块。**
2. **Agent 的核心判定与最小闭环**：
   一个系统被称为 Agent 的核心判定在于：**模型能否根据环境反馈自主闭环控制执行流**。最简单的 ReAct Agent 仅需要：
   - 当前上下文和工具定义（Prompt）；
   - 基础的循环运行时（Runtime Loop）；
   - 支持工具调用的 LLM 与工具执行器。
   它仅依靠**工作记忆（Working Memory，即当前 Context Window 中的短期对话历史）**与局部的单步决策，就已经构成完整的 Agent 闭环。
3. **增强模块的作用定位**：
   - **显式规划模块（Planning）**：如 Plan-and-Solve 模式，先生成大纲再执行，主要用于缓解长链条任务中的注意力漂移和目标丢失。详见 [Agent 如何进行任务规划？](/interview/agent-interview/agent-planning.md) 与 [复杂查询规划设计](/frontier/agentic-rag/complex-query-planning.md)。
   - **长期记忆模块（Long-term Memory）**：利用向量库或外部 KV 存储历史信息，主要用于跨 Session 维护上下文偏好或沉淀历史经验，但并不参与单次任务的动态控制流判定。详见 [AI Agent 记忆系统解析](/frontier/agent-memory-system/what-is-memory.md) 与 [Agent 中的 Memory 有哪些类型？](/interview/agent-interview/agent-memory.md)。

---

## 面试避坑指南

![面试答辩中的常见伪概念与技术边界辨析](/images/content-rebuild/7f54fad55edd3e4e/fig-6.png)

在面试中讨论 Agent 时，避免以下三处常见误区：

| 常见错误说法 | 问题所在 | 严谨表达 |
| :--- | :--- | :--- |
| “Agent 能在执行中实时自我学习、优化自身权重。” | 混淆了“模型微调”与“上下文更新”。生产中模型权重是冻结的，所谓“学习”只是把执行结果或反思写回上下文或外部存储。 | “Agent 通过将执行反馈写入 Working Memory 或外部数据库调整后续推理策略，模型本身的权重保持不变。” |
| “Agent 比工作流更先进，未来所有工作流都会被 Agent 取代。” | 忽略了商业系统对确定性、延迟、成本与合规审计的要求。 | “生产架构更倾向于‘Agentic Workflow’：用确定性代码构建骨架以保证可控，仅在特定复杂决策节点引入局部的 Agent 子图。” |
| “只要接入了 Function Calling / API 就是 Agent。” | 混淆了“工具调用能力”与“智能体系统”。如果只是单次提取参数调一次 API，它依然只是增强版的单次模型调用。 | “只有当模型能够根据 API 的执行结果多轮迭代、自主纠错并决定后续分支时，才形成了 Agent 闭环。”具体机制参考 [Tool Calling 与 Function Calling 的区别](/interview/agent-interview/tool-calling.md)。 |

理解了控制流与系统权衡之后，在实际开发框架中，这两类模式是如何落地的？下一篇我们将分析 LangChain 中的具体实现：[什么是 LangChain 中的 Agent？它和 Chain 有什么区别？](/interview/agent-interview/langchain-agent-vs-chain.md)。
