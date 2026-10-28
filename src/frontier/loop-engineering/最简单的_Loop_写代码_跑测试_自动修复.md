---
title: 最简单的 Loop：写代码、跑测试、自动修复
icon: mdi:refresh
date: 2026-07-16
tag:
  - Loop Engineering
  - Agent
---

# 最简单的 Loop：写代码、跑测试、自动修复

> 一句话核心：Loop Engineering 是一种让 AI Agent 自主运行"写代码 → 跑测试 → 自动修复"闭环循环的工程方法论，核心在于设计目标驱动的自主系统而非手动提示 AI。

---
## 核心概念（术语表）

- **Loop Engineering（循环工程）**：2026 年 6 月由 Google 工程师 Addy Osmani 正式命名的工程学科，设计让 Agent 自己提示自己的循环系统，核心是"规划 → 执行 → 验证 → 修复 → 持续运行"
- **/loop 命令**：Claude Code 的内置 Bundled Skill，支持定时任务与自主迭代，能让 AI 自动执行多轮任务循环
- **Agent Loop（智能体循环）**：目标驱动的自主闭环系统，输入目标、状态、记忆、验证机制，输出持续运行的结果
- **/run 命令**：Claude Code 中启动应用并观察的命令，可在修改后验证代码是否正确运行
- **/verify 命令**：构建或运行来验证改动的命令，属于 Bundled Skill，与 /run 配合使用
- **Bundled Skill（捆绑技能）**：Claude Code 中以 / 开头的技能集合如 /loop、/batch、/run、/verify，基于 Prompt 工作流，调用时加载特定 Markdown 指令集
- **Harness Engineering（编排工程）**：AI 工程第三阶段，关注如何调用能力，通过上下文 + API + 工具链执行任务
- **Closed-loop System（闭环系统）**：能够根据输出状态自动调整输入的系统，是 Loop Engineering 的核心目标

---
<ReadMoreLock />

## 历史背景 / 来源

Loop Engineering 的概念在 2026 年 6 月正式形成并迅速传播。

2026 年 6 月初，Anthropic Claude Code 负责人 Boris Cherny 在公开演讲中提出："I don't prompt Claude anymore. I have loops running. They're the ones prompting Claude and figuring out what to do. My job is to write loops."（我不再直接提示 Claude 了，我有一套 Loop 在运行，它们负责提示 Claude 并决定下一步做什么，我的工作是编写 Loop。）

几天后，2026 年 6 月 7 日，开发者 Peter Steinberger（开源 AI Agent 项目 OpenClaw 的创作者，GitHub 历史上获星最快的新仓库）发推："You shouldn't be prompting coding agents anymore. You should be designing loops that prompt your agents."（你不应该再手动提示 AI 编程助手了，你应该设计让 Agent 自己提示自己的 Loop。）

随后 Google 工程师 Addy Osmani 在 Substack 发表长文，将这一实践正式命名并系统化为 Loop Engineering，使其成为独立的工程学科。

---
## 工作原理 / 核心机制

### 整体思路

Loop Engineering 的核心是构建一个目标驱动的自主闭环系统，AI Agent 自主完成"规划 → 执行 → 验证 → 修复 → 持续运行"的循环，人从"提问者"转变为"规则制定者"。

### 输入 / 输出

**输入**：目标（Goal）、当前状态（State）、记忆（Memory）、验证机制（Verification）

**输出**：持续运行的结果、自主修复后的代码、通过验证的任务完成

### 核心步骤详解

**第一步：规划（Plan）**

Claude Code 分析当前任务，理解目标意图，分解为可执行的子任务。当用户输入 "/loop" 命令时，Claude Code 加载特定 Markdown 指令集到上下文，启动子代理（Sub-agents）执行多步工作流。

**第二步：执行（Execute）**

使用 /run 命令启动应用并观察，Claude Code 执行代码编写、文件修改等操作。此阶段 Bundled Skill 调动工具链完成任务，调用频率由调度方案控制。

**第三步：验证（Verify）**

使用 /verify 命令构建或运行来验证改动是否正确。Claude Code 自动运行测试、检查输出、对比预期结果。验证失败则触发修复流程。

**第四步：修复（Fix）**

当验证失败时，Loop 自动进入修复阶段，Claude Code 根据错误信息分析问题根因，进行代码修改，再次执行验证循环。

**第五步：持续运行（Loop）**

三种调度方案控制循环频率：固定间隔（如每 30 分钟）、条件触发（如测试失败时）、持续运行（直到手动停止）。两种工作模式：监控模式（观察并报告）和修复模式（发现问题后主动修复）。

---

## 关键知识点

- Loop Engineering 是 AI 工程演进的第四阶段，从 Prompt → Context → Harness → Loop 逐步演进
- Claude Code 命令来源分两层：硬编码 Commands（如 /clear、/help）和 Bundled Skills（如 /loop、/run、/verify）
- /loop 命令本质是基于 Prompt 的能力，调用时加载 Markdown 指令集调动子代理执行多步工作流
- AI 编程工具经历三代演进：第一代自动补全、第二代对话式、第三代 Agent 自主循环
- 第三代工具瓶颈从"如何提问"变为"如何设计让 Loop 可靠运行的系统"
- 工程师核心竞争力从"会写提示词"变为"会设计 Loop"
- Claude Code 三种调度方案：固定间隔、条件触发、持续运行
- Claude Code 两种工作模式：监控模式（观察并报告）和修复模式（发现问题后主动修复）
- /run 命令用于启动应用并观察，可在修改后验证代码是否正确运行
- /verify 命令属于 Bundled Skill，专门用于构建或运行来验证改动
- /run-skill-generator 可记录项目的启动方式，便于后续自动化执行
- Loop Engineering 的典型场景：Claude Code、AI 编程、自动运营、AI 员工
- 人的角色从"提问者"变为"规则制定者"，核心是设计闭环而非手动交互

---

## 应用场景

- **Claude Code 定时任务**：开发者设定每 30 分钟运行一次测试套件，Loop 自动执行、发现问题、尝试修复，节省 70% 的人工检查时间
- **持续集成验证**：代码提交后自动触发 /verify，失败时 /loop 启动修复循环，直到所有测试通过才停止
- **AI 编程助手**：用户描述需求后，Agent 自主规划任务、编写代码、运行测试、修复 Bug，循环直到需求满足
- **OpenClaw 项目**：Peter Steinberger 设计的开源 Agent 项目，实现让 Agent 自己提示自己的 Loop 系统
- **Claude Code 接入国内模型**：通过 CC Switch 工具接入 GLM5.1、DeepSeek V4 等国内模型，保持 Loop 运行能力

---

## 常见误区 / 踩坑

- ❌ **误区 1**：很多人以为 /loop 只是定时执行任务的命令
  ✅ **正解**：/loop 是完整的自主迭代系统，支持监控模式和修复模式，可根据验证结果自动调整行为
- ❌ **误区 2**：认为 Loop Engineering 只是更好的 Prompt Engineering
  ✅ **正解**：Prompt Engineering 是"教 AI 怎么做"，Loop Engineering 是"设计一个系统让 AI 自己持续做"，两者本质不同
- ❌ **误区 3**：把 /run 和 /verify 当作两个独立命令使用
  ✅ **正解**：两者是配合关系，/run 启动应用观察，/verify 验证改动，通常组合使用形成完整的验证循环
- ❌ **误区 4**：认为 Loop 运行后不需要人参与
  ✅ **正解**：人仍是"规则制定者"，需要设计验证机制、设定停止条件、审核修复结果
- ❌ **误区 5**：忽视 /debug 命令的作用
  ✅ **正解**：Claude Code 自己出问题时应先跑 /debug，这是解决 CLI 问题的首选命令

---

## 性能 / 复杂度

- **时间复杂度**：O(n × k)，n 为循环次数，k 为每次循环内操作数，最坏情况可能无限循环
- **空间复杂度**：O(m)，m 为记忆存储大小，与上下文窗口和任务复杂度相关

**与替代方案对比**：

| 方案 | 适用场景 | 效率 | 人工介入 |
|------|---------|------|---------|
| 手动 Prompt | 单次任务、快速验证 | 低 | 高 |
| 固定脚本 | 重复性任务 | 中 | 中 |
| Loop Engineering | 复杂闭环任务、持续交付 | 高 | 低 |

- **临界点**：当任务需要超过 3 次手动干预时，Loop Engineering 的效率优势开始显现

---

## 与相关概念的区别

- **vs Prompt Engineering**：
  - Prompt Engineering 关注"怎么问问题"，Loop Engineering 关注"怎么让 AI 持续做事"
  - Prompt Engineering 输入是 Prompt/指令，Loop Engineering 输入是目标、状态、记忆、验证机制
  - Prompt Engineering 人的角色是"提问者"，Loop Engineering 人的角色是"规则制定者"
  - 怎么选：一次性任务用 Prompt Engineering，持续性任务用 Loop Engineering

- **vs Harness Engineering**：
  - Harness Engineering 关注"如何调用能力"，Loop Engineering 关注"如何持续完成目标"
  - Harness 是静态的编排，Loop 是动态的循环
  - 怎么选：单次工作流用 Harness，持续迭代用 Loop

- **vs Context Engineering**：
  - Context Engineering 关注"给 AI 什么信息"，Loop Engineering 关注"让 AI 持续产出结果"
  - Context 是输入的组织，Loop 是输出的循环
  - 怎么选：信息理解用 Context，目标实现用 Loop

---

## 进阶 / 面试加分项

- **最新进展**：2026 年 Claude Code 推出了 /code-review ultra 云端深度审查功能，可与 /loop 组合实现自动代码审查循环
- **业界争议**：Loop 何时该停止仍未有定论，过度循环可能导致资源浪费，但停止过早可能遗漏问题
- **送给候选人**：Prompt Engineering 决定 AI 能做什么，Loop Engineering 决定 AI 能持续做什么——后者才是 2026 年工程师的核心竞争力


## 面试如何回答


### 🟢 什么是 Loop Engineering？它和 Prompt Engineering 有什么区别？

**回答要点**：

Loop Engineering 是 2026 年 6 月由 Google 工程师 Addy Osmani 正式命名的工程学科，核心是设计让 AI Agent 自主运行的循环系统。它与 Prompt Engineering 的本质区别在于：Prompt Engineering 是「教 AI 怎么做」，关注怎么问问题、输入 Prompt/指令；Loop Engineering 是「设计一个系统让 AI 自己持续做」，关注如何持续完成目标、输入目标、状态、记忆、验证机制。打个比方，Prompt Engineering 像是一次性教练，Loop Engineering 像是一套自动训练系统。从人的角色看，Prompt Engineering 需要人作为「提问者」，Loop Engineering 需要人作为「规则制定者」。适用场景上，一次性任务用 Prompt Engineering，复杂闭环任务用 Loop Engineering。


### 🟡 Claude Code 的 /loop 命令是如何工作的？有哪些调度方案和工作模式？

**回答要点**：

/loop 是 Claude Code 的 Bundled Skill，本质是基于 Prompt 的能力，调用时 Claude 会载入特定的 Markdown 指令集到上下文，然后调动子代理执行多步工作流。/loop 解决了什么问题？它让 AI 能够自主迭代，用户设定目标后，AI 自动完成规划、执行、验证、修复的循环。调度方案有三种：固定间隔（如每 30 分钟自动运行）、条件触发（如测试失败时启动修复）、持续运行（直到手动停止）。工作模式分两种：监控模式（观察并报告问题）和修复模式（发现问题后主动修复）。实际使用时，用户输入 /loop 后接任务描述，Claude Code 自动分析目标、分解任务、循环执行直到达成目标或达到停止条件。


### 🟡 /run 和 /verify 命令在 Loop 循环中扮演什么角色？它们如何配合使用？

**回答要点**：

/run 和 /verify 是 Claude Code 中形成完整验证循环的两个关键命令。/run 用于启动应用并观察，可指定启动命令（如 npm run dev）、监控特定端口、观察日志输出。当用户修改代码后，/run 可以验证应用是否能正常启动。/verify 是 Bundled Skill，专门用于构建或运行来验证改动是否正确，常配合测试套件使用。两者配合形成「执行-验证」闭环：/run 启动应用，/verify 运行测试或构建，如果验证失败，/loop 启动修复循环，再次执行 /run 和 /verify，直到所有检查通过。Claude Code 还提供 /run-skill-generator 来记录项目的启动方式，便于后续自动化执行。在 CI/CD 场景中，这个组合可以实现提交代码后自动验证的闭环流程。


### 🟢 AI 编程工具经历了哪三代演进？第三代工具的核心瓶颈是什么？

**回答要点**：

AI 编程工具经历三代演进：第一代是自动补全，代表工具是 GitHub Copilot 早期版本，工作方式是补全当前行或函数，人类主导所有决策，瓶颈是只能辅助不能自主。第二代是对话式，代表工具是 ChatGPT 和 Claude.ai，工作方式是问一句答一句，人类手动推进每一步，瓶颈是人类成为瓶颈，速度受限于打字速度。第三代是 Agent 自主循环，代表工具是 Claude Code 和 OpenAI Codex Agent，工作方式是 Agent 自主规划、执行、验证，循环迭代直到完成。第三代工具的核心瓶颈从「如何提问」变成了「如何设计让 Loop 可靠运行的系统」。这意味着工程师的核心竞争力从「会写提示词」变成了「会设计 Loop」，这也是 Loop Engineering 概念在 2026 年迅速流行的原因。


### 🔴 在设计一个可靠的 Loop 系统时，需要考虑哪些关键要素？

**回答要点**：

设计可靠的 Loop 系统需要考虑六大要素：第一，明确的目标定义（Goal），需要具体可量化，不能模糊；第二，状态追踪（State），记录 Loop 当前执行到哪一步、结果如何；第三，记忆机制（Memory），让 Agent 知道历史执行情况，避免重复犯错；第四，验证机制（Verification），定义什么叫「成功」，通过 /verify 自动检查；第五，停止条件（Stop Condition），防止无限循环，比如最大迭代次数、时间限制、成功率阈值；第六，修复策略（Fix Strategy），验证失败后如何处理，是重试、回滚还是人工介入。实际项目中，建议先用 /loop 跑一个简单任务观察行为，设定合理的停止条件（如最多循环 5 次），每次循环后用 /verify 验证结果，失败时人工检查后再决定是否继续。


### 🟡 Claude Code 命令体系中，Commands 和 Bundled Skills 有什么区别？

**回答要点**：

Claude Code 中 / 开头的命令来源分两层。第一层是硬编码的 Commands，逻辑写死在 CLI 代码里，直接与终端交互，通常不需要额外 Prompt 工作流，包括 /clear、/compact、/model、/usage、/help、/review、/diff、/context、/permissions 等。这些命令是 Claude Code 内置的原生能力。第二层是 Bundled Skills，本质是基于 Prompt 的能力，调用时 Claude 会载入特定的 Markdown 指令集到上下文，然后调动子代理执行多步工作流，包括 /simplify、/code-review、/loop、/batch、/debug、/run、/verify、/claude-api 等。简单理解，Commands 是「肌肉反射」，Bundled Skills 是「复杂动作」。使用时注意：本地 diff 正确性审查优先用 /code-review，深度云端审查用 /code-review ultra。另外 /review 是内置 PR review 命令，不是 Bundled Skill，需要区分。


### 🟡 Loop Engineering 为什么在 2026 年 6 月突然流行？背后的驱动因素是什么？

**回答要点**：

Loop Engineering 在 2026 年 6 月突然流行的核心驱动因素是：Prompt Engineering 已经成熟，但瓶颈转移了。两句话引爆了社区：Boris Cherny（Claude Code 负责人）说「我不再直接提示 Claude 了，我有一套 Loop 在运行，它们负责提示 Claude 并决定下一步做什么」；Peter Steinberger（OpenClaw 创作者）说「你不应该再手动提示 AI 编程助手了，你应该设计让 Agent 自己提示自己的 Loop」。这两句话精准描述了一种趋势：提示词写得好不好已经不再是瓶颈，瓶颈在于为 Agent 设计的整套运行系统。Claude Code 等第三代 Agent 工具的出现让「让 AI 持续做事」成为可能，而不仅仅是「问一次答一次」。随后 Addy Osmani 系统整理成 Loop Engineering 学科，使其从经验上升为可教授的工程方法论。


### 🟡 在实际项目中，如何用 Claude Code 实现「写代码、跑测试、自动修复」的最简 Loop？

**回答要点**：

实现最简 Loop 的步骤如下：第一步，用 /run 启动应用并观察，比如 /run npm run dev；第二步，用 /verify 运行测试套件验证当前状态，比如 /verify npm test；第三步，如果测试失败，输入 /loop 启动修复循环，描述任务如「修复所有失败的测试，循环直到全部通过或达到 5 次迭代」；第四步，Claude Code 自动分析错误、修改代码、运行测试，循环执行直到满足停止条件。这个最简 Loop 体现了 Loop Engineering 的核心：用户从「提问者」变为「规则制定者」，设定目标后让 AI 自主执行。Claude Code 会调用子代理、加载特定指令集、自动规划修复策略。进阶用法可以配合 /code-review 进行代码审查循环，配合 /batch 并行执行多个任务循环。在 CI 场景中，可以用 claude -p 非交互模式在脚本里运行这个 Loop。

