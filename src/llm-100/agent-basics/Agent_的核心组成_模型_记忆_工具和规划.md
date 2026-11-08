---
title: Agent 由哪些部分组成？模型、状态、工具与规划的协作机制
icon: mdi:cpu-64-bit
date: 2026-07-16
tag:
- 大模型100讲
- Agent
description: 拆解 Agent 系统的四大组成部分：模型基座、工作状态、工具接口与规划策略，厘清宿主程序驱动的控制循环与工程边界。
---

在上一讲[《Agent 是什么？它和普通 ChatBot 有什么区别？》](/llm-100/agent-basics/Agent_是什么_它和普通_ChatBot_有什么区别.md)中，我们明确了 Agent 与普通对话机器人的核心差异：ChatBot 的交互终点是输出一段文本，而 Agent 的终点是借助外部环境完成一项具体任务。

很多开发者在初次设计 Agent 时，往往容易陷入组件堆砌的误区，以为必须同时引入大模型、向量数据库、多级分解器和几十个外部接口才算完整。但在实际工程中，过度复杂的组件耦合不仅会带来高昂的调用延迟，还会大幅增加系统排障的难度。

一个健壮的 Agent 到底由哪些部分构成？可以用一个结论概括：**Agent 并不是一个活在模型内部的独立生命体，而是由宿主程序（Host Runtime）驱动的状态机。它由模型基座负责判断、工作状态记录历史、工具提供操作能力、规划策略决定执行路径。**

为了看清它们如何协作，本文将以一个具体的工程任务贯穿始终：**运维巡检 Agent 自动排查测试服务器磁盘占用过高的问题**。

![Agent 系统的四大积木与系统定位](/images/content-rebuild/5b6e415abe80ad5c/cover.png)

---

## 一、Agent 系统的协作中枢：宿主驱动循环

要理解 Agent 的运转机制，首先要打破对“模型自主运行”的想象。大语言模型本身是一个无状态的被动数学函数：输入一段 Token 序列，计算输出下一个 Token 的概率分布。模型自身既不能定时自动唤醒，也无法向操作系统发起任何网络请求或磁盘读写。

真正维系 Agent 持续行动的，是开发者编写的**宿主程序（Host Runtime / 控制循环）**。宿主程序负责维护会话状态、组装提示词、发起模型推断、拦截工具请求并执行真实代码，最后将结果送回下一轮模型推断中。

![Agent 系统核心拓扑与宿主驱动循环](/images/content-rebuild/5b6e415abe80ad5c/fig-1.png)

在“排查磁盘占用”的例子中，宿主驱动的完整循环分为四步：

1. **组装初始上下文**：宿主程序接收用户指令“检查测试机磁盘并清理大文件”，结合当前可用的工具清单（如 `check_disk`、`find_large_files`），打包成消息列表发送给大模型。
2. **模型推断意图**：大模型读取上下文后，判断当前缺少磁盘数据，输出一段结构化的工具调用请求：`{"name": "check_disk", "args": {"mount_point": "/"}}`。
3. **宿主执行拦截**：宿主程序拦截该请求，验证参数无误后，在真实的测试机或沙盒中执行 `df -h /` 命令，获得真实输出 `"/dev/vda1 95% full"`。
4. **状态写回与迭代**：宿主将该执行结果封装为一条 `role: tool` 消息追加到历史中，再次请求模型。模型看到磁盘已满 95%，接着决定发起下一个工具调用 `find_large_files`。这个循环会一直持续，直到模型确认任务完成并输出文本汇报。

这个控制循环说明：**工具不是大模型直接调用的，而是宿主程序代为执行的；系统状态也不是大模型自动记住的，而是宿主程序按轮次组装提交的。**

<ReadMoreLock />

---

## 二、模型基座：从文本写手到结构化决策引擎

在普通的问答场景下，模型主要负责生成通顺的长篇自然语言。但在 Agent 体系中，模型基座的核心职责转变为**意图决策器**与**参数提取器**。

![模型基座从文本生成到结构化决策引擎的演进](/images/content-rebuild/5b6e415abe80ad5c/fig-2.png)

作为 Agent 大脑的基础模型，通常需要重点具备三项核心工程能力：

1. **严格的指令遵从（Instruction Following）**：在面对冗长工具返回信息和多轮提示时，能够严格遵守既定限制。例如要求“仅允许在 `/data/temp` 目录下查找文件”，模型不能越权去根目录扫描。
2. **可靠的结构化输出（Structured Output）**：能够按照特定 JSON Schema 输出函数名与参数。如果模型偶尔漏掉闭合括号，或者将整型参数输出成字符串，宿主程序的反序列化逻辑就会报错中断。
3. **依据错误反馈调整策略**：当宿主返回“权限不足”或“目录不存在”等报错信息时，模型能够理解失败原因并尝试备选方案，而不是机械地重复同一条错误调用。

在后续的[《Tool Calling 是什么？大模型如何调用工具？》](/llm-100/agent-basics/Tool_Calling_是什么_大模型如何调用工具.md)与[《Function Calling 和 MCP 有什么区别？》](/llm-100/agent-basics/Function_Calling_和_MCP_有什么区别.md)中，我们将深入解析主流厂商是如何在模型底层通过特殊标记（Special Tokens）来保障结构化输出稳定性的。

---

## 三、工具（Tools）：定义接口规范与安全沙盒

工具是 Agent 突破静态参数限制、获取外部实时信息并改变环境状态的桥梁。

在代码实现中，**工具不等于一段让模型直接运行的黑盒脚本，而是一份暴露给模型的“接口说明书”与宿主内部“执行函数”的结合体**。大模型只负责“根据说明书填写表单”，实际的动作完全由宿主程序代劳。

![工具调用的执行屏障与宿主沙盒机制](/images/content-rebuild/5b6e415abe80ad5c/fig-3.png)

一个规范的工具实现包含三个层次：

- **接口声明（Schema）**：告诉模型该工具有什么作用、需要哪些参数、各参数的数据类型以及是否必填。例如声明 `delete_file(file_path: str, force: bool)`。
- **宿主执行体（Handler）**：宿主程序中真实的业务逻辑，如 Python 脚本中的 `os.remove()` 或发送 HTTP POST 请求。
- **权限控制与沙盒隔离（Sandbox & Permissions）**：只读操作（如查询磁盘、读取日志）可以直接放行；而具有破坏性或写操作的工具（如删除文件、重启容器），必须在宿主中设计拦截确认逻辑（Human-in-the-Loop），甚至必须在只读容器沙盒中运行。

**工程边界提醒**：不要指望在提示词里写一句“请勿删除系统文件”来保障安全。提示词约束无法从根本上防御注入攻击与偶发幻觉，真正的安全防护只能在宿主程序的参数校验和系统权限层面做硬性拦截。

---

## 四、工作状态与记忆（Memory）：执行状态与跨会话沉淀

关于 Agent 的“记忆”，业界存在较多模糊的讨论。在工程落地中，必须把**单次运行的工作状态（Working State）**与**跨会话的长期存储（Long-term Memory）**严格区分开来。

![工作状态维护与可选跨会话记忆的数据流向](/images/content-rebuild/5b6e415abe80ad5c/fig-4.png)

### 1. 工作状态（短期上下文）：系统的必需品
工作状态指当前排查任务正在发生时的消息列表（Message Queue）。它包含：
- 用户发起的目标（User Message）；
- 模型历次推断和工具调用请求（Assistant Message）；
- 宿主执行工具后回填的观察内容（Tool Message）。

大模型在两次 API 调用之间没有任何记忆残留。宿主程序在第 3 轮推断时，必须把前两轮的所有消息完整发给模型，模型才能知道“上一轮找出了哪个大文件”。如果工作状态丢失，Agent 就无法连续推进任务。

这也带来了**上下文窗口预算**问题：随着工具返回结果的累加（例如一次文件扫描返回了数百行文本），上下文会快速消耗。宿主程序通常需要结合截断策略或摘要策略，在保留核心状态的同时防止上下文溢出。

### 2. 长期存储（可选外置记忆）：按需增强项
长期存储通常借助持久化数据库（关系型数据库、Key-Value 库或向量数据库）实现。它记录跨任务、跨日期的持久信息，例如“该服务器上周曾经因为 `/tmp/cache` 溢出而报警”或“运维人员张三偏好通过钉钉接收告警”。

如果任务是自包含的一次性排查，完全不需要挂载向量数据库。相关架构设计与选型分析，我们将在[《Agent 的 Memory 应该怎么设计？》](/llm-100/agent-basics/Agent_的_Memory_应该怎么设计.md)与[《短期记忆、长期记忆、向量记忆分别是什么？》](/llm-100/agent-basics/短期记忆_长期记忆_向量记忆分别是什么.md)中展开说明。

---

## 五、规划策略（Planning）：单步反应与预先分解

规划能力决定了 Agent 面对复杂目标时如何编排具体步骤。工程中主流的规划实现分为两类：

![隐式单步决策与显式任务分解的执行路径对比](/images/content-rebuild/5b6e415abe80ad5c/fig-5.png)

### 1. 反应式单步决策（Reactive / Step-by-Step）
以经典的 [ReAct 框架](/llm-100/agent-basics/ReAct_框架是什么_为什么适合_Agent.md) 为代表。模型不提前规划全套详细流程，而是“看一眼当前状态，决定下一步做什么”。
- **工作方式**：先查磁盘 -> 拿到 95% -> 决定查大文件 -> 发现日志目录很大 -> 决定汇报给用户。每一步根据上一步的反馈动态调整。
- **适用条件**：外部环境变化大、依赖运行时返回结果分支多的场景。
- **风险**：视野较短，缺乏全局约束时容易在循环状态中打转。具体排查方案可参考[《Agent 为什么容易陷入死循环？》](/llm-100/agent-basics/Agent_为什么容易陷入死循环.md)。

### 2. 显式任务分解（Plan-and-Solve）
系统引入一个专门的规划阶段：面对目标，模型首先输出一个有序任务清单（例如 `[1. 检查磁盘占用, 2. 扫描前三名大文件, 3. 询问用户是否清理, 4. 执行清理]`），随后按清单逐条执行。
- **工作方式**：预先拆解静态步骤，再交给执行模块按序调度。
- **适用条件**：业务路径固定、前后依赖清晰的任务。
- **风险**：容错能力较低。如果步骤 1 发现磁盘占用仅为 20%（无需排查），固定的步骤清单可能会继续盲目向下执行，必须额外补充重规划（Re-planning）机制。

Anthropic 在《Building effective agents》工程指南中提出了一条务实建议：**在构建系统时应优先寻找最简方案，从单次模型调用或通过预定义代码路径编排的工作流（Workflows）开始，以保证任务的可预测性与一致性；只有在确实需要高灵活性与模型动态主导流程的复杂场景下，才建议转向由模型自主驱动的 Agent 模式。**

---

## 六、手写一个最小控制循环

为了直观展示四大模块在代码中的交互细节，下面编写一个不依赖三方 Agent 框架的纯 Python 控制循环。

代码模拟巡检 Agent 处理“测试服务器磁盘占用过高”的过程：

```python
import json

# 1. 工具集合：声明与实际执行函数
TOOLS = {
    "check_disk": {
        "description": "查看指定挂载点的磁盘使用率",
        "parameters": {"mount_point": "str"},
        "handler": lambda mount_point: json.dumps({"mount": mount_point, "used_percent": 92})
    },
    "find_large_files": {
        "description": "查找指定目录下占用空间最大的文件",
        "parameters": {"directory": "str", "limit": "int"},
        "handler": lambda directory, limit: json.dumps([
            {"path": f"{directory}/app_debug.log", "size_gb": 14.5}
        ])
    }
}

# 2. 模拟大模型的结构化决策（用分支逻辑模拟模型推断）
def mock_llm_inference(messages: list) -> dict:
    last_msg = messages[-1]
    
    # 阶段 1：初始输入，模型决定检查磁盘
    if last_msg["role"] == "user":
        return {
            "role": "assistant",
            "tool_call": {"name": "check_disk", "args": {"mount_point": "/"}}
        }
    
    # 阶段 2：收到 check_disk 结果，发现使用率 92%，决定扫描大文件
    if last_msg["role"] == "tool" and last_msg["tool_name"] == "check_disk":
        disk_data = json.loads(last_msg["content"])
        if disk_data["used_percent"] > 80:
            return {
                "role": "assistant",
                "tool_call": {"name": "find_large_files", "args": {"directory": "/var/log", "limit": 1}}
            }
        return {"role": "assistant", "content": "磁盘使用率正常，无需处理。"}
    
    # 阶段 3：收到扫描结果，输出最终报告
    if last_msg["role"] == "tool" and last_msg["tool_name"] == "find_large_files":
        files = json.loads(last_msg["content"])
        target = files[0]
        return {
            "role": "assistant",
            "content": f"排查完成：根目录使用率已达 92%。定位到主要大文件为 {target['path']}（{target['size_gb']} GB），建议运维人员评估后清理。"
        }
    
    return {"role": "assistant", "content": "无法识别当前状态。"}

# 3. 宿主驱动控制循环
def run_inspection_agent(goal: str, max_steps: int = 5):
    # 初始化工作状态（上下文列表）
    working_state = [{"role": "user", "content": goal}]
    
    for step in range(1, max_steps + 1):
        print(f"\n=== 第 {step} 轮推断 ===")
        
        # 步骤 A：大模型根据当前状态进行推断
        response = mock_llm_inference(working_state)
        working_state.append(response)
        
        # 步骤 B：检查是否给出最终文本回答
        if "tool_call" not in response:
            print(f"Agent 最终答复: {response['content']}")
            return response["content"]
        
        # 步骤 C：宿主程序拦截并执行工具调用
        call_info = response["tool_call"]
        name = call_info["name"]
        args = call_info["args"]
        print(f"模型申请调用工具: {name}，参数: {args}")
        
        if name in TOOLS:
            observation = TOOLS[name]["handler"](**args)
        else:
            observation = json.dumps({"error": f"未授权的工具: {name}"})
        
        print(f"宿主执行结果回填: {observation}")
        
        # 步骤 D：将执行结果组装为标准 Tool 消息追加到工作状态
        working_state.append({
            "role": "tool",
            "tool_name": name,
            "content": observation
        })
    
    raise TimeoutError("Agent 在最大步数内未收敛")

if __name__ == "__main__":
    run_inspection_agent("请排查测试服务器磁盘占用过高问题")
```

运行上述脚本，可以在控制台清晰观察到三轮状态演进：
1. 用户目标存入工作状态；
2. 模型返回 `check_disk` 调用意向，宿主执行并回填 92%；
3. 模型感知到数值超标，返回 `find_large_files` 调用意向，宿主执行并回填日志文件信息；
4. 模型综合全部状态，收敛并输出最终自然语言报告。

无论是使用 LangGraph、AutoGPT 还是自研框架，底层运转的本质都是这一套以状态列表为核心的宿主循环。

---

## 七、常见误区与工程选型边界

在规划系统架构时，了解“不需要什么”往往比“引入什么”更重要。

![最小可用 Agent 与过度设计架构的组件边界对比](/images/content-rebuild/5b6e415abe80ad5c/fig-6.png)

### 误区 1：没有向量数据库的系统就不是 Agent
**事实分析**：错误。向量数据库只是长期持久化存储的一种实现形式，主要用于相似度检索。如果任务是短程自包含的（如代码编辑、日志排查、SQL 查询），Agent 只需要依靠当前工作状态与本地工具即可高水准完成。是否属于 Agent，取决于是否具备**基于环境反馈推进任务的决策循环**，而不是挂载了何种数据库。

### 误区 2：规划器拆解越深越智能
**事实分析**：不一定。多层嵌套的任务分解在面对真实网络波动和动态环境时极其脆弱。提示词层层传递会带来语义损耗，前置步骤稍有偏差就会引发后续连环崩溃。在实际工程中，**使用硬编码代码（Deterministic Code）把控业务主干，在关键节点让大模型进行单步决策（Agentic Step）**，其稳定性和可维护性远高于全交给大模型自行生成的多级计划。

### 误区 3：Agent 具备完全自主的自愈能力
**事实分析**：模型的自我修正建立在环境返回了明确错误原因的前提下（例如抛出包含行号的 SyntaxError）。如果外部 API 静默超时或返回空数据，模型极易出现自我确证偏差，进而陷入无效重试的死循环。生产系统必须在宿主程序中设定**最大步数限制（Max Steps）**、**超时熔断**与**人工兜底介入**机制。

---

## 动手练习：验证与加固控制循环

**学习目标**：掌握宿主程序对工具调用异常的处理逻辑，验证防死循环熔断机制。

**练习任务**：
基于第六节的代码，修改 `run_inspection_agent` 和 `mock_llm_inference`，完成以下两个工程防御改动：
1. **模拟工具异常处理**：在 `TOOLS` 中增加一个名为 `clean_log_file` 的工具，执行逻辑为直接抛出 `PermissionError("Permission denied: /var/log/app_debug.log")`。修改宿主循环的“步骤 C”，使用 `try-except` 捕获异常，并将结构化的错误信息（如 `json.dumps({"status": "error", "message": str(e)})`）回填给模型。
2. **验证最大步数保护**：让模型在收到该错误后持续发起重试请求。观察宿主控制循环在达到 `max_steps=3` 时是否能如期抛出 `TimeoutError` 中断程序，防止 API 费用失控。

**验收标准**：
- 程序不会因为 Python 原生异常直接闪退；
- 运行记录中能看到工作状态正确记录了错误观察消息；
- 达到预设最大步数时触发熔断，控制台打印出防护提示。

---

## 本讲小结与延伸阅读

梳理 Agent 的构成可以归纳为三句话：
1. **最小必需组件**：基础模型决策 + 宿主控制循环 + 工具接口说明 + 上下文工作状态。
2. **外部增强组件**：向量数据库（跨轮记忆）与显式分解器（规划器）属于特定长链路场景的增强选项，按需接入即可。
3. **核心工程防线**：大模型只负责填写参数，宿主程序负责物理执行与权限拦截。系统的健壮性取决于宿主程序的边界校验与熔断能力。

下一讲我们将进入工具交互协议的底层细节：[《Tool Calling 是什么？大模型如何调用工具？》](/llm-100/agent-basics/Tool_Calling_是什么_大模型如何调用工具.md)。
