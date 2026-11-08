---
title: Agent 是什么？它与工作流有什么区别？
icon: mdi:cpu-64-bit
description: 梳理 Agent 与工作流的架构边界，拆解受控循环状态机、工具调用时序与执行熔断机制。
---

如果给大语言模型（LLM）一个输入，它返回一段文本，这只是一次无状态的文本补全。但如果让模型根据当前目标，自己决定先查询数据库、根据返回的数据计算指标，再决定是否报警，这就构成了具备环境交互能力的代理系统。

我们将通过一个“排查并清理服务器高磁盘占用”的贯穿任务，说明从普通模型调用到工作流（Workflow），再到自主代理（Agent）的演进过程与工程实现方式。

![Agent 基础篇核心知识架构全景](/images/content-rebuild/ab329503476a47fd/cover.png)

---

## 学习目标与能力分级

本模块将 Agent 开发涉及的知识拆解为四个可逐步验证的层级：

- **L1 运行机制**：理解只生成文本的对话与产生副作用（Side Effect）的工具调用之间的不同。明确模型并不直接“运行”代码，而是输出结构化的调用指令，由外部宿主系统执行。
- **L2 协议解析**：掌握模型工具调用的请求与响应格式，理解如何通过模式描述（如 JSON Schema）约束模型参数，以及模型上下文协议（Model Context Protocol，MCP）如何在不同客户端与服务间统一工具暴露标准。
- **L3 控制与边界**：理解推理与行动交替执行的循环过程，知道如何设置步数上限（Max Steps）和异常捕获机制，防止模型在遇到未知错误时反复重试导致无限循环。
- **L4 混合编排**：依据业务的确定性要求，合理划分工作流代码与模型自主决策的边界，在状态图框架（如 LangGraph）中实现断点检查与人工审核（Human-in-the-loop）。

---

## 架构区分：预定义工作流与自主代理

Anthropic 在系统构建指南中指出，引入大语言模型的系统可以统称为 Agentic Systems，但在控制权归属上存在两种主要形态：

1. **工作流（Workflows）**：执行路径由预定义代码路由编排。例如：固定先调用模型分类工单，再根据分类结果走预设的 Python `if-else` 分支，最后调用工具发邮件。模型的输出影响数据流向，但不改变程序的调用结构。需注意代码分支不等于整个工作流完全确定，节点内的模型输出、网络环境与外部状态仍可能波动。
2. **自主代理（Agents）**：执行步骤由模型在运行时动态决定。外部代码提供一个运行循环（Loop）和一组可用工具，模型根据每一步获得的环境反馈（Observation），自主决定下一步是继续调用工具还是结束任务。

![从单轮补全、确定性工作流到自主代理的架构控制权对比](/images/content-rebuild/ab329503476a47fd/fig-1.png)

在生产系统中，很少直接使用完全无约束的代理。常见的做法是混合编排：主干流程使用预定义工作流代码约束核心业务流转，降低偏离风险；在分类不确定、搜索策略需要动态调整的局部节点引入代理。

---

## 模块内容导引

本模块分为五个阶段，各篇之间存在直接的依赖关系：

| 学习阶段 | 讲义路径与主题 | 解决的具体问题 |
| :--- | :--- | :--- |
| **第一阶段：基础概念** | [/llm-100/agent-basics/Agent_是什么_它和普通_ChatBot_有什么区别.md](/llm-100/agent-basics/Agent_是什么_它和普通_ChatBot_有什么区别.md)<br>[/llm-100/agent-basics/Agent_的核心组成_模型_记忆_工具和规划.md](/llm-100/agent-basics/Agent_的核心组成_模型_记忆_工具和规划.md) | 区分无状态补全与有副作用操作；拆解模型、记忆、工具与规划的基本分工。 |
| **第二阶段：工具调用与协议** | [/llm-100/agent-basics/Tool_Calling_是什么_大模型如何调用工具.md](/llm-100/agent-basics/Tool_Calling_是什么_大模型如何调用工具.md)<br>[/llm-100/agent-basics/Function_Calling_和_MCP_有什么区别.md](/llm-100/agent-basics/Function_Calling_和_MCP_有什么区别.md) | 掌握模型返回参数的格式化校验；理解私有 API 与标准化 MCP 服务端协议的集成差异。 |
| **第三阶段：控制循环与容错** | [/llm-100/agent-basics/ReAct_框架是什么_为什么适合_Agent.md](/llm-100/agent-basics/ReAct_框架是什么_为什么适合_Agent.md)<br>[/llm-100/agent-basics/Agent_为什么容易陷入死循环.md](/llm-100/agent-basics/Agent_为什么容易陷入死循环.md) | 追踪思考-行动-观察步骤；分析模型反复生成相同调用导致死循环的成因与拦截手段。 |
| **第四阶段：上下文与状态管理** | [/llm-100/agent-basics/Agent_的_Memory_应该怎么设计.md](/llm-100/agent-basics/Agent_的_Memory_应该怎么设计.md)<br>[/llm-100/agent-basics/短期记忆_长期记忆_向量记忆分别是什么.md](/llm-100/agent-basics/短期记忆_长期记忆_向量记忆分别是什么.md) | 管理长轮次调用的上下文窗口；区分单次运行内的短期状态与跨会话的外部持久化存储。 |
| **第五阶段：多智能体与落地工程** | [/llm-100/agent-basics/多_Agent_协作到底解决什么问题.md](/llm-100/agent-basics/多_Agent_协作到底解决什么问题.md)<br>[/llm-100/agent-basics/面试高频_如何设计一个可落地的_Agent_系统.md](/llm-100/agent-basics/面试高频_如何设计一个可落地的_Agent_系统.md) | 评估多模型协作的调用成本与错误级联风险；掌握生产环境的人机审核挂起与状态追踪。 |

![Agent 基础篇十讲学习进阶拓扑图](/images/content-rebuild/ab329503476a47fd/fig-2.png)

建议先掌握工具调用协议与控制循环，再研究记忆与状态管理。因为状态本质上是在每一次模型决策与工具执行后更新的数据结构；如果对执行循环没有清晰认知，记忆设计很容易沦为单纯的文本追加。

---

## 验证代码：最小受控状态机

以下 Python 代码实现了一个不依赖任何第三方代理框架的受控执行循环。它展示了一个排查磁盘报警的任务中，模型生成指令、系统派发工具、捕获执行结果以及步数截断的完整流程：

```python
from typing import Callable, Dict, Any

class ControlledAgent:
    def __init__(self, tools: Dict[str, Callable], max_steps: int = 4):
        self.tools = tools
        self.max_steps = max_steps
        self.state: Dict[str, Any] = {
            "history": [],
            "completed": False,
            "result": None
        }

    def mock_llm_step(self, observation: str) -> dict:
        """模拟大模型根据上一步的执行结果给出下一步动作"""
        if "未初始化" in observation:
            return {"action": "call_tool", "tool_name": "check_disk", "args": {"path": "/data"}}
        elif "占用率 92%" in observation:
            return {"action": "call_tool", "tool_name": "clean_cache", "args": {"path": "/data/tmp"}}
        elif "清理完成" in observation:
            return {"action": "finish", "output": "磁盘已成功清理，恢复安全运行水位"}
        return {"action": "finish", "output": "无法识别当前状态，人工介入"}

    def run(self, task: str) -> Dict[str, Any]:
        self.state["history"].append(f"Task: {task}")
        observation = "系统未初始化，准备开始排查"
        step = 0

        while step < self.max_steps:
            step += 1
            # 1. 模型决策（Agentic 步骤）
            decision = self.mock_llm_step(observation)
            
            if decision["action"] == "finish":
                self.state["completed"] = True
                self.state["result"] = decision["output"]
                break

            # 2. 外部确定性分发与执行（Deterministic 步骤）
            tool_name = decision.get("tool_name", "")
            tool_fn = self.tools.get(tool_name)
            if not tool_fn:
                observation = f"Error: 未知工具 {tool_name}"
            else:
                try:
                    observation = str(tool_fn(**decision.get("args", {})))
                except Exception as err:
                    observation = f"Tool Execution Error: {str(err)}"

            self.state["history"].append(
                f"Step {step}: Action={tool_name}, Observation={observation}"
            )

        if not self.state["completed"]:
            self.state["result"] = "Error: 超过最大允许步数，触发系统熔断"
            
        return self.state

# 验证用例：模拟工具环境
tools_env = {
    "check_disk": lambda path: f"{path} 磁盘占用率 92%（超过警戒线 85%）",
    "clean_cache": lambda path: f"{path} 临时文件清理完成，释放 12GB 空间"
}

agent = ControlledAgent(tools=tools_env, max_steps=4)
final_state = agent.run("检查服务器 /data 盘空间并清理")

print("执行完成:", final_state["completed"])
print("最终产物:", final_state["result"])
print("状态记录:")
for log in final_state["history"]:
    print("  ", log)
```

![带有步数限制与状态检查点的受控 Agent 运行机制](/images/content-rebuild/ab329503476a47fd/fig-3.png)

---

## 检验练习与常见误区

读者可以尝试修改上面的代码，验证以下三种典型故障场景并设计对应的防护逻辑：

1. **工具报错处理**：让 `clean_cache` 抛出 `PermissionError("权限不足")`，观察模拟模型是否会陷入重复调用同一工具的死循环，并在宿主层添加连续失败重试阈值。
2. **步数熔断验证**：将 `max_steps` 设为 1，确认系统是否能正确中断并把未完成的状态信息写回持久化存储。
3. **高危操作断点（Human-in-the-loop）**：在执行清理操作（如 `clean_cache`）之前向命令行打印提示，要求必须由人工输入 `yes` 后再调用真实函数，否则挂起任务。

针对更复杂的状态图编排与工程设计，可进一步阅读后续进阶主题：[/frontier/ai-workflow/README.md](/frontier/ai-workflow/README.md)、[/frontier/agent-memory-system/README.md](/frontier/agent-memory-system/README.md) 以及架构设计模块 [/architecture/README.md](/architecture/README.md)。
