---
title: ReAct 是什么？
icon: mdi:brain
tag:
  - Agent面试
  - ReAct
  - 思考行动
---

![](/images/agent-interview/react_cover.png)

# ReAct 是什么？Thought、Action、Observation 分别代表什么？

这道面试题考的是 **ReAct 推理框架**，是大模型 Agent 开发中最核心的推理范式之一。简单说，ReAct 就是让大模型在解决问题时"边想边做、边做边想"，通过 Thought（思考）、Action（行动）、Observation（观察）三个步骤的循环迭代，最终完成任务。

---

## 板块1：ReAct 解决的核心问题——为什么需要"推理+行动"

先说背景。早期大模型处理复杂任务，走的是两条路：

**纯推理路线**（CoT思维链）：模型只管想，从头到尾靠内部推理得出答案。问题在哪？模型"闭门造车"，不跟外部世界交互，遇到不知道的信息就瞎编，幻觉问题严重。

**纯行动路线**（WebGPT、SayCan）：模型直接调用工具查资料、执行操作。问题在哪？模型"只干不想"，拿到什么查什么，没有规划，走一步算一步，经常做无用功。

ReAct 的出现，就是为了解决这两条路的缺陷。它的核心思想很简单：**交替进行推理和行动，让思考指导行动，让行动验证思考**。

![三列对比图：纯推理路线只思考不行动、纯行动路线只行动不思考、ReAct路线边思考边行动](/images/agent-interview/20260608_213352_002624_01.png)

你可以把大模型想象成一个需要完成复杂任务的员工：

- 纯推理型员工 → 纸上谈兵，方案写得漂亮但不一定能用
- 纯行动型员工 → 蛮干硬拼，累死累活可能方向就错了
- ReAct 型员工 → 先想后做、边做边想，遇到问题随时调整

---
<ReadMoreLock />

## 板块2：三元组各司其职——Thought、Action、Observation 的定义

ReAct 的核心就是 **Thought → Action → Observation** 三个环节不断循环。咱们逐个说：

### Thought（思考）

这是模型的"内心独白"。Thought 负责：
- 分解任务：把大目标拆成小步骤
- 分析进度：判断当前做到哪一步了
- 决定策略：下一步该怎么做

**关键点**：Thought 只在模型内部流转，不会发到外部工具。它是纯推理过程。

### Action（行动）

这是模型的"外部接口"。Thought 想清楚之后，Action 负责：
- 调用搜索引擎查资料
- 调用 API 获取信息
- 执行某个具体操作

**关键点**：Action 是模型与外部世界的交互通道，它会把指令发送给外部工具。

### Observation（观察）

这是外部世界的"真实反馈"。Action 执行完后，Observation 返回：
- 搜索结果
- API 返回的数据
- 操作的实际结果

**关键点**：Observation 保证模型是基于真实信息推理，而不是凭空想象。

![三元组关系示意图，Thought内心独白指向Action外部行动，Action结果返回为Observation真实反馈，三者形成闭环](/images/agent-interview/20260608_213346_821568_01.png)

用一个场景来理解：医生看病。

- **Thought**：医生思考"这个症状可能是什么病，需要做哪些检查"
- **Action**：医生开出检查单，让病人去做检查
- **Observation**：检查报告出来了，这是真实的客观数据

然后医生再根据检查结果继续思考、继续开检查或开药，这就是一个完整的循环。

---

## 板块3：循环工作机制——ReAct 如何迭代完成任务

单个 Thought-Action-Observation 只是一个基本单元。ReAct 的真正威力在于**循环迭代**。

整个流程是这样的：

1. 模型根据当前状态输出 **Thought**
2. Thought 决定需要执行什么 **Action**
3. Action 调用外部工具，返回 **Observation**
4. Observation 作为下一轮的输入，触发新的 **Thought**
5. 循环往复，直到模型认为"可以输出答案了"，输出 **Final Answer**

![ReAct循环流程图，展示Thought→Action→Observation→新Thought...→Final Answer的多轮迭代过程](/images/agent-interview/20260608_213344_125696_01.png)

这就好比用导航开车：

- 纯推理模式：出发前规划好全程路线，然后闭着眼睛开
- ReAct 模式：导航每过一段路就播报"你当前位置如何、前方怎么走"，实时调整

每一步的 Observation 都在纠正模型的推理，让后续思考越来越准确。这也是 ReAct 能显著**减少幻觉**的原因——你的每一步推理都有真实数据做锚点。

在工程实现上，**LangChain 的 AgentExecutor** 就是 ReAct 的典型实现。它维护一个循环，不断调用 agent 生成 Thought 和 Action，执行后获取 Observation，直到 agent 输出 Final Answer。

---

## 板块4：优势、局限与实际应用

### 优势

- **可解释性强**：你能看到模型每一步在想什么、做了什么、结果是什么
- **减少幻觉**：每步推理都基于真实的 Observation，不再凭空编造
- **泛化性好**：同一套框架，换个工具就能处理不同任务

### 局限

- **计算开销大**：多轮循环意味着多次 LLM 调用，成本高、延迟高
- **错误传播**：某一轮推理错了，后面的结论可能全歪
- **依赖模型能力**：模型本身不够强，Thought 的质量就不行，整个链条都会受影响

![优劣势对比图，左侧优势（可解释、减少幻觉、泛化）、右侧局限（开销、错误传播、能力依赖）](/images/agent-interview/20260608_213414_869440_01.png)

### 应用场景

ReAct 是当前主流 Agent 框架的底层推理模式：
- **LangChain Agent**：用 AgentExecutor 实现 ReAct 循环
- **LlamaIndex Agent**：同样基于 ReAct 范式
- **AutoGPT 等项目**：本质就是 ReAct 的工程实现

这把"瑞士军刀"很好用，但也要看任务场景——简单的问答用 CoT 就够了，复杂的工具调用任务才需要 ReAct。

---

## 面试怎么答

### 基础版

ReAct 全称是 Reasoning and Acting，是一种让大模型"边推理边行动"的推理框架。它通过 Thought-Action-Observation 三个环节的循环迭代来处理复杂任务：Thought 是模型的内心独白，负责分解任务和规划策略；Action 是外部接口，调用工具获取信息；Observation 是真实反馈，保证推理基于事实数据。这个循环持续进行，直到模型输出最终答案。LangChain 的 AgentExecutor 就是 ReAct 的典型实现。

### 加分版

相比纯推理的 CoT，ReAct 的核心优势在于引入了真实世界的反馈，能有效减少幻觉。相比纯行动的 WebGPT，ReAct 通过 Thought 保持了推理的可控性和方向感。ReAct 也有不少改进变体，比如 Reflexion（加入自我反思）、Plan-and-Execute（先规划再执行）。在实际开发中，ReAct 是 LangChain、LlamaIndex 等主流 Agent 框架的底层推理模式，适用于需要多步工具调用的复杂场景。

---

## 一句话总结

ReAct 就是让大模型通过"想清楚 → 做一下 → 看结果 → 再想清楚"的循环，在推理和行动交替中完成复杂任务。