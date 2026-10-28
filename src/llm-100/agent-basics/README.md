---
title: Agent 基础篇
icon: mdi:cpu-64-bit
---

![](/images/agent-basics/agent-basics_cover.png)

# Agent 基础篇：这个合集能让你学到什么？

你有没有想过，为什么 ChatGPT 只能聊天，而 Agent 能帮你订机票、查数据库、写代码、操作文件？

区别在于：ChatGPT 是一个"只会说话"的语言模型，而 Agent 是一个"能动手干活"的智能体。它有记忆、有工具、有规划能力，能在真实环境中执行任务。

但 Agent 不是简单地把大模型和工具拼在一起。记忆怎么设计？工具怎么调用？为什么 Agent 容易陷入死循环？多 Agent 协作到底解决什么问题？这些才是 Agent 的核心设计问题。

这个系列共 10 篇文章，从 Agent 的本质出发，覆盖核心架构、ReAct 框架、Tool Calling、记忆系统和多 Agent 协作。

---

## 十篇文章，带你从概念到设计

**核心概念篇（前三篇）**

先搞清楚 Agent 和 ChatBot 的本质区别。然后理解 Agent 的四大核心组成：模型、记忆、工具和规划。最后深入 ReAct 框架，理解"思考-行动-观察"循环为什么有效。

1. [Agent 是什么？它和普通 ChatBot 有什么区别？](/llm-100/agent-basics/Agent_是什么_它和普通_ChatBot_有什么区别.html) - Agent 的本质
2. [Agent 的核心组成：模型、记忆、工具和规划](/llm-100/agent-basics/Agent_的核心组成_模型_记忆_工具和规划.html) - 四大核心模块
3. [ReAct 框架是什么？为什么适合 Agent？](/llm-100/agent-basics/ReAct_框架是什么_为什么适合_Agent.html) - 推理与行动的结合

**工具调用篇（中间三篇）**

Agent 的核心能力之一是调用外部工具。Tool Calling 是什么？Function Calling 和 MCP 有什么区别？这些决定了 Agent 能"做什么"。

4. [Tool Calling 是什么？大模型如何调用工具？](/llm-100/agent-basics/Tool_Calling_是什么_大模型如何调用工具.html) - 工具调用机制
5. [Function Calling 和 MCP 有什么区别？](/llm-100/agent-basics/Function_Calling_和_MCP_有什么区别.html) - 两种工具调用范式
6. [Agent 的 Memory 应该怎么设计？](/llm-100/agent-basics/Agent_的_Memory_应该怎么设计.html) - 记忆系统设计

**记忆与协作篇（接着两篇）**

Agent 的另一个核心能力是记忆。短期记忆、长期记忆、向量记忆分别是什么？为什么 Agent 容易陷入死循环？这些是 Agent 工程化的关键问题。

7. [短期记忆、长期记忆、向量记忆分别是什么？](/llm-100/agent-basics/短期记忆_长期记忆_向量记忆分别是什么.html) - 三种记忆类型
8. [Agent 为什么容易陷入死循环？](/llm-100/agent-basics/Agent_为什么容易陷入死循环.html) - 循环问题与对策

**综合实战篇（最后两篇）**

多 Agent 协作到底解决什么问题？如何设计一个可落地的 Agent 系统？这是 Agent 从理论到实践的最后一步。

9. [多 Agent 协作到底解决什么问题？](/llm-100/agent-basics/多_Agent_协作到底解决什么问题.html) - 从单体到协作
10. [面试高频：如何设计一个可落地的 Agent 系统？](/llm-100/agent-basics/面试高频_如何设计一个可落地的_Agent_系统.html) - 综合实战

---

[← 返回大模型100讲](/llm-100/)
