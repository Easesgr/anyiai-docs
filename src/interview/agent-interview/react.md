---
title: ReAct 是什么？模型如何交替推理与调用工具？
icon: mdi:brain
tag:
- Agent面试
- ReAct
- 思考行动
description: 通过具体案例拆解 ReAct 中 Thought、Action 与 Observation 的交替循环机制，梳理宿主控制流程、面试口述模板与常见追问。
---

在面试中被问及“什么是 ReAct”时，很多回答容易停留在“模型边想边做”这类抽象概念上，缺乏具体的工程输入、宿主拦截逻辑以及与纯思维链的对比依据。理解 ReAct，需要看清 Thought、Action、Observation 三者在上下文中的流转方式，以及宿主程序如何截断模型输出并回传工具结果。

![ReAct 范式的循环交互机制](/images/content-rebuild/c95259cab1f0cba9/cover.png)

---

## 面试速答：30 秒核心口述模板

> **面试官提问**：请简要介绍一下 Agent 中的 ReAct 模式，它的工作流程是怎样的？

**建议口述回答**：

ReAct 是 **Reasoning（推理）与 Acting（行动）** 协同交替的运行模式，最早由 Yao 等人在 2022 年提出。

它的核心是在模型推理和外部环境之间建立一个循环：
1. **Thought（思考）**：模型根据当前任务和上下文，输出一段推理文本，说明当前状态、分解目标并确定下一步计划；
2. **Action（行动）**：模型生成结构化的工具调用指令（如工具名和入参）；宿主程序拦截该指令，并在外部环境中实际执行工具；
3. **Observation（观察）**：宿主程序把工具返回的结果作为客观事实，追加到上下文中重新输入给模型。

三者交替循环，直到模型判定已获取全部所需信息，输出 **Final Answer** 结束任务。

相比只靠内部参数记忆的 CoT（思维链），ReAct 引入外部工具的返回结果，能够及时阻断虚构事实的滚雪球效应；相比不经思考直接调接口的纯 Acting 模式，ReAct 的推理过程提升了多步任务调用的准确率与可排查性。

---

## 一、Thought、Action、Observation 的流转机制

ReAct 并没有修改 Transformer 的模型权重或内部架构，而是一种依托上下文拼接和程序调度的模式。其核心是由宿主程序（Host）控制的大模型自回归生成循环。

我们在 [/interview/agent-interview/agent-loop.md](/interview/agent-interview/agent-loop.md) 中分析过 Agent 运行循环，ReAct 就是该循环的一种经典实现。

![Thought、Action、Observation 交互与状态迭代流程](/images/content-rebuild/c95259cab1f0cba9/fig-1.png)

### 一次完整的状态流转过程

1. **构建输入序列 ($S_t$)**：上下文包含用户原始问题、历史步骤中的推理文本、执行过的工具指令以及工具返回的文本：
   $$S_t = \{Q, \text{Thought}_1, \text{Action}_1, \text{Obs}_1, \dots, \text{Thought}_{t-1}, \text{Action}_{t-1}, \text{Obs}_{t-1}\}$$
2. **推理生成 ($	ext{Thought}_t$)**：模型接收 $S_t$，生成一段分析文本，归纳已收集到的信息并确定下一步调用意图。
3. **动作生成 ($	ext{Action}_t$)**：在思考文本之后，模型生成约定的工具调用指令（例如 `search["查询词"]` 或特定的 JSON 格式）。
4. **输出截断与工具执行**：宿主程序通过预先设定的停止标记（如遇到 `Observation:` 文本）拦截模型输出，暂停生成，解析出工具名称和入参，并在外部真实环境中运行工具代码。
5. **回填环境观察 ($\text{Obs}_t$)**：宿主程序将工具执行结果格式化为 `Observation: [返回内容]`，拼接在上下文末尾，形成 $S_{t+1}$，再次请求模型继续生成。
6. **终止分支**：模型在某一轮思考后判断已有足够依据回答问题，跳过工具调用，直接输出 `Final Answer: [最终回答]`，退出循环。

---

## 二、为什么不单独使用思维链或单独调用工具？

ReAct 的提出是为了解决早期大模型应用中两条路线的各自缺陷：**仅推理（Reasoning-only，如思维链 CoT）**与**仅行动（Acting-only，如直接根据输入映射工具 API）**。

![CoT、Acting 与 ReAct 在状态依赖任务中的表现对比](/images/content-rebuild/c95259cab1f0cba9/fig-2.png)

### 1. 仅依赖思维链（CoT）的缺陷
思维链通过逐步展开推理步骤激发模型的内在逻辑分解能力，但计算过程完全发生在固定的模型参数内：
* **无法感知外部变化**：面对模型训练截止期之后的新闻、企业私有数据库或随时变动的环境状态，模型无法仅凭参数给出准确事实；
* **错误不断累积**：由于自回归是以前序 Token 为条件进行概率采样，一旦前几步出现事实性错误，该错误就会作为已知事实输入后续计算，导致整条推理链顺着错误假设推导下去。

### 2. 仅依赖工具调用（Acting）的缺陷
模型不写出中间思考，直接输出工具调用指令并接收返回：
* **意图理解与参数组装耦合**：模型必须在单次生成中同时完成长目标规划和入参提取。当问题需要多步拆解时，模型容易直接使用过于宽泛的搜索词，或者陷入重复调用的死循环；
* **缺乏信息整合阶段**：工具返回多段文本后，若没有中间思考步骤来进行筛选和提炼，模型容易漏掉关键条件，导致最终输出质量下降。

<ReadMoreLock />

---

## 三、贯穿案例：推理与行动如何互相配合

在 ReAct 运行中，**推理引导工具调用，外部观察修正后续推理**。

![Observation 反馈修正错误推理假设的因果链条](/images/content-rebuild/c95259cab1f0cba9/fig-3.png)

### 真实两步推演示例

设用户提问：“2024 年巴黎奥运会男子 100 米自由泳冠军打破了谁保持的世界纪录？”

```text
Question: 2024 年巴黎奥运会男子 100 米自由泳冠军打破了谁保持的世界纪录？

Thought 1: 我需要先查明 2024 年巴黎奥运会男子 100 米自由泳的冠军是谁。
Action 1: search["2024 巴黎奥运会 男子100米自由泳 冠军"]
Observation 1: 潘展乐在 2024 年巴黎奥运会男子 100 米自由泳决赛中以 46 秒 40 夺冠并打破世界纪录。

Thought 2: 冠军是潘展乐，他在该场比赛中打破了世界纪录。我需要查询在该场比赛之前，该项目的世界纪录由谁保持。
Action 2: search["潘展乐 46.40 之前 男子100米自由泳 世界纪录保持者"]
Observation 2: 潘展乐在巴黎奥运会打破的是自己此前在 2024 年 2 月多哈世锦赛创造的 46 秒 80 的世界纪录。

Thought 3: 经过核实，打破的是潘展乐本人此前保持的纪录。信息已经完整，可以直接作答。
Final Answer: 潘展乐打破了他本人此前在 2024 年多哈世锦赛上创造的 46 秒 80 的世界纪录。
```

在这个过程中：
1. **Thought 1 提炼目标**：将复杂问题拆分为第一步“找冠军”。
2. **Observation 1 提供新事实**：明确了冠军是潘展乐，同时带出“打破了世界纪录”的新信息。
3. **Thought 2 纠正方向**：不再盲目搜索其他人，而是精准检索“该纪录此前是谁保持的”。
4. **Observation 2 阻断主观猜想**：如果单凭预训练记忆推测，模型可能会误以为是上一代名将保持的纪录；工具返回的事实直接纠正了这一步，确保了结论准确。

---

## 四、ReAct 的 Prompt 结构与宿主控制实现

在原生工具调用（Function Calling）普及前，ReAct 通常借助特定的提示词模板和文本截断机制在应用层落地。

![ReAct 文本提示词模板与 Stop 标记控制时序](/images/content-rebuild/c95259cab1f0cba9/fig-4.png)

### 1. 经典文本 Prompt 模板结构
标准的 ReAct System Prompt 包含工具说明、少样本格式范例（Few-shot）和运行约定：

```text
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:
```

### 2. 宿主程序的三项关键控制
* **设置停止字符串（Stop Sequence）**：在支持该参数的接口中，宿主通常配置 `stop=["\nObservation:", "Observation:"]`，在模型生成到观察前缀时由服务端中止输出。需要注意，停止字符串依赖接口支持，且若模型产生格式变体或在动作前提前伪造内容，停止字符串无法完全防范，宿主仍需配合正则解析与截断控制；
* **输出文本解析**：宿主通过正则匹配提取 `Action` 与 `Action Input`。如果格式不合法（如工具名不存在），宿主把解析报错拼装在 `Observation` 中（例如 `Observation: Tool not found. Please choose from [...]`），让模型重新组织输入；
* **现代原生调用的承接**：现代大模型（如通过 OpenAI 或 Anthropic 工具调用协议）将 Action 封装成了结构化的 JSON 字段。关于底层协议的具体实现，见 [/interview/agent-interview/tool-calling.md](/interview/agent-interview/tool-calling.md)。尽管交互格式从纯文本演变为结构化字段，推演、调用、回填、再推演的 ReAct 运行逻辑依然一致。

---

## 五、面试常见误区与深层追问

![外显生成的 Thought 与模型内部计算表示的本质区别](/images/content-rebuild/c95259cab1f0cba9/fig-5.png)

### 误区 1：模型输出的“Thought”是否等同于它内部真实的物理思考过程？

**澄清**：**不等同。外显的 Thought 只是模型自回归采样生成的自然语言文本，不是模型内部注意力或特征流动的物理读数。**

1. **事后合理化倾向**：模型生成思考文本的本质依然是计算下一个 Token 的条件概率。在部分实验中发现，模型在较浅层可能已经形成了走向某种动作的隐层状态，输出的“Thought”只是用自然语言把这个倾向包装成看起来合乎逻辑的理由；
2. **逻辑与行动脱节**：外显的思考内容有时可能与紧随其后的行动矛盾（例如 Thought 里写着“此方法不可行”，但紧接着 Action 仍调用了该方法）。不能把 Thought 视为绝对可靠的内部控制寄存器，它主要的作用是为后续 Token 采样提供有价值的上下文先验。

### 误区 2：ReAct 是否总是优于单次调用或纯思维链？

**澄清**：**不是。ReAct 伴随着显式的延迟和调用成本。**
* **Token 开销与延迟成倍增加**：ReAct 每前进一步都需要发起一次模型请求。随着上下文把所有的历史 Observation 都拼装进来，上下文长度线性增长，Prefill 与生成 Decode 的耗时都会显著叠加；
* **并非所有场景都需要工具**：对于封闭域内的代码编写、翻译、纯文本总结或基础常识推理，直接使用思维链更加经济且快速，强行引入 ReAct 反而增加了网络超时和解析失败的风险。

### 常见追问及应对口径

> **追问 1：如果模型陷入了“Thought -> Action -> Observation”的死循环，如何防范？**

**应对思路**：
必须由外部宿主程序施加确定性规则，而不能依赖模型自觉：
1. **硬性最大步数限制（Max Iterations）**：在运行循环中设置步数上限（例如最多 5 轮），达到上限立刻中断工具调用，强制要求模型汇总已有线索直接生成答案；
2. **工具调用历史去重**：宿主记录历史 `(Action, Action Input)` 哈希值。如果检测到模型连续两次使用完全相同的入参调用同一个工具，直接在新的 Observation 中注入告警提示；
3. **容错与降级机制**：工具连续报错时触发熔断，退化为无工具回答并告知用户依赖服务受限。具体异常控制见 [/interview/agent-interview/agent-error-handling.md](/interview/agent-interview/agent-error-handling.md)。

> **追问 2：ReAct 和 Plan-and-Solve（规划-执行）的区别是什么？**

**应对思路**：
* **决策粒度**：ReAct 属于**单步动态规划**。它每拿到一次工具返回，才临时决定下一步做什么，适应动态多变、需要实时探索的环境，但缺乏对全局任务的长程把控；
* **Plan-and-Solve**：属于**前置全局规划**。它在开始前先生成完整的任务计划列表，再依次分配给执行模块。前者偏向局部动态调整，后者偏向复杂任务的结构化拆解。关于规划方案的对比详见 [/interview/agent-interview/agent-planning.md](/interview/agent-interview/agent-planning.md)。

![ReAct 在面试考察知识图谱中的定位与延伸方向](/images/content-rebuild/c95259cab1f0cba9/fig-6.png)

---

## 关联导航

* **运行环境**：了解 ReAct 所依赖的宿主调度机制 [/interview/agent-interview/agent-loop.md](/interview/agent-interview/agent-loop.md)
* **源码实践**：从状态机与代码实现角度查看执行流程 [/interview/agent-interview/react-principle.md](/interview/agent-interview/react-principle.md)
* **协议演进**：从文本解析升级到结构化调用 [/interview/agent-interview/tool-calling.md](/interview/agent-interview/tool-calling.md)
* **长程决策**：单步贪心不足以覆盖全局时的规划方案 [/interview/agent-interview/agent-planning.md](/interview/agent-interview/agent-planning.md)
* **错误治理**：工具异常与循环调用的工程防御 [/interview/agent-interview/agent-error-handling.md](/interview/agent-interview/agent-error-handling.md)
* **备战索引**：查阅完整 Agent 常见面试考点 [/interview/agent-interview/README.md](/interview/agent-interview/README.md)
