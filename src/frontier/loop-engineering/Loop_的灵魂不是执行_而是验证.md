---
title: Loop 的灵魂不是执行，而是验证
icon: mdi:refresh
date: 2026-07-16
tag:
  - Loop Engineering
  - Agent
---

# Loop 的灵魂不是执行，而是验证

> 一句话核心：Loop Engineering 不是让 Agent 一直跑（while true），而是让它每轮都离目标更近，并且知道什么时候该停——验证才是闭环的关键。
## 核心概念（术语表）

- **Loop Engineering（循环工程）**：工程化 Agent 多轮之间如何推进的技术，把 prompt、context、tool、memory、verification、policy 串成持续运转的系统，解决的不是"生成什么"而是"下一步做什么"。
- **闭环（Closed Loop）**：区别于 while true 循环，必须有目标、观察、验证、状态和退出条件四要素，执行结果必须回到系统成为下一轮输入。
- **Verifier（验证器）**：判断输出有没有变好的组件，没有 verifier 的 loop 容易自嗨，是 Loop Engineering 最关键的层。
- **Harness Engineering（套索工程）**：Agent 的安全带、仪表盘和执行外壳，解决"能不能安全地做事"，与 Loop Engineering 是叠加关系而非替代。
- **状态机（State Machine）**：boss-skill 项目中用 pending → running → completed/failed → retrying → running 记录阶段状态，确保 Agent 声明不能当真相。
- **Evidence Bundle（证据包）**：orca 项目中"完成"不是模型自己宣布的，要能被 evidence.json 里的测试结果、diff review 等证据证明。
- **Reflexion Loop（反思循环）**：Agent 执行失败 → 分析原因 → 将教训存入记忆库 → 带着教训重新尝试，是最重要的自我进化设计模式。
- **Generator-Critic Pattern（生成器-批判者模式）**：两个角色一条流水线，生成者产出初稿，批判者审查挑刺，生成者重写，不断重复直到达到质量阈值。
<ReadMoreLock />

## 历史背景 / 来源

Loop Engineering 概念由青雲的博客在 2026 年 6 月提出，源于 Coding Agent 进入长任务（持续运行 1d 19h+）的场景需求。在此之前，AI 开发经历了三个阶段：Prompt Engineering（2017-2022，解决单轮问答）、Context Engineering（2022-2024，解决上下文装配）、Harness Engineering（2024-2025，解决安全执行）。Tony Bai 在 2026 年 7 月进一步梳理了工业级 AI 生产系统中最核心的 20 个循环设计模式，将其比作 AI 时代的《设计模式》。AI 知名专家 Rahul (@sairahul1) 指出："Agent 只是干活的工人，而 Loop 才是让工人不断自我进化的机制"——年薪百万与普通工程师的差距正在于此。

## 工作原理 / 核心机制

**整体思路**：Loop Engineering 不是在 while true 外面包一层名字，而是工程化一个长期任务在每一轮之后"怎么留下状态、怎么产生下一步、怎么验证自己没有跑偏、怎么在失败后恢复、怎么让人能随时接手"这五个核心问题。

**输入/输出**：输入是初始目标（Goal）和当前状态（State），输出是下一轮动作（Action）和更新后的状态。区别于传统 AI 调用的"输入 Prompt → 获得 Response → 结束"单次模式，Loop Engineering 的核心是"生成 → 评估 → 学习 → 改进"的自动化闭环。

**核心步骤详解**：

第一步：**Goal 定义与分解**。目标不能只是"帮我做一下"，而要能被推进、被验证。复杂任务通过 Goal Decomposition Loop 拆解为子目标（Subgoals）→ 具体任务（Tasks）→ 执行步骤（Steps），持续递归直到每个最小单元可通过单次模型调用搞定。例如"写一份详尽的竞品分析报告"拆解为：定位前 5 竞品 → 分析核心功能 → 对比价格模型 → 找出市场空白。

第二步：**Planner 滚动规划**。不是一开始幻想完整路径，而是根据当前状态动态规划下一步。boss-skill 项目中 execution.workflow.nextNodeIds 由当前状态、产物依赖、门禁结果一起计算出来，确保"下一步不是凭感觉派出去的，而是有依据的"。

第三步：**Actor 执行 + Observer 观察**。Actor 真正执行动作：读文件、改代码、调 API、跑命令、查资料。Observer 收集结果：工具输出、测试结果、日志、diff、错误信息，都要回到系统里。注意生成者并不知道自己正在被考核，只有评估者掌握打分标准，这种角色隔离是 Score-and-Retry 模式的精髓。

第四步：**Verifier 验证判断**。判断有没有变好，没有 verifier 的 loop 很容易自嗨。orca 项目中 verifier 会检查：evidence bundle 是否存在、失败数是否超限、transcript 是否存在、有没有违反 read-only mutation policy、required tool call 有没有真的发生。"完成"不是模型自己宣布的，而是要能被证据证明。

第五步：**State 持久化 + Exit Condition 判断**。做过什么、失败过什么、现在卡在哪里、下一步为什么这么选，都要留下来。退出条件包括：成功退出、失败退出（连续失败 N 次后换策略或请求人类介入）、预算退出、风险退出、人类接管。不会停的自动化比不会开始的自动化更危险。

## 关键知识点

- Loop Engineering 的本质不是让 Agent 一直跑，而是让它每跑一轮都离目标更近，并且知道什么时候该停
- "循环只是重复，闭环要有目标、观察、验证、状态和退出条件"——while true ≠ Loop Engineering
- 长任务里，真正重要的不是"它还在跑"，而是"它为什么继续跑"
- Prompt Engineering 关注这一轮怎么问，Context Engineering 关注这一轮给模型看什么，Loop Engineering 关注多轮之间怎么推进
- 一个 Coding Agent loop 通常流程：先根据目标拆计划 → 读相关文件 → 做最小修改 → 跑测试 → 测试失败就分析原因 → 如果是自己改坏了就回修改环节 → 如果是环境问题就标记阻塞 → 测试通过后看 diff 有没有越界 → 最后生成证据
- 好的 Loop 不是"它一直在跑"，而是每一轮都能回答：为什么继续、继续之后要验证什么、验证失败后怎么改、到什么程度该停下来
- Loop 七层架构：Goal（目标）+ Planner（规划器）+ Actor（执行器）+ Observer（观察器）+ Verifier（验证器）+ State（状态）+ Exit Condition（退出条件）+ Policy（策略约束）
- 没有 Harness 的 Loop 很危险，没有 Loop 的 Harness 只是安全但不会持续推进的工具壳
- Agent 的声明不能当真相，验证结果才是下一轮的输入
- boss-skill 是工作流层 Loop：需求、产物、DAG、门禁、反馈、重派；orca 是运行时层 Loop：状态持久化、暂停恢复、失败重启、缓存复用、证据验证
- "会失败一次的系统"与"只会在同一个地方摔倒一次的系统"本质区别在于 Reflexion Loop
- 记忆压缩：N 条具体记忆 → 升华为"底层规律"抽象，例如三条任务都因 X 失败 → 压缩为"执行任何任务前，务必优先检查 X"
- Multi-Critic Loop 四角色：正确性批判者（信息是否事实准确）、风格批判者（表达是否清晰）、安全批判者（内容是否合规）、领域批判者（是否达到行业专家标准）
- 动态工作流根据中间结果在运行时决定自己的形状：输出是 A 走分支 X，输出是 B 走分支 Y，输出是 C 直接跳过步骤 2 执行步骤 5
- 错误档案库：在执行任何新任务之前，系统会首先检索错题本，如果存在类似失败记录，直接将已知修正方案写入执行策略
## 应用场景

- **场景 1：OpenClaw Codex 长期任务运行**：一个 Codex goal 任务已经运行 1d 19h 48m，持续目标不断读取上下文、判断进展、执行动作、验证结果。这种场景下真正关心的不是"这一轮 prompt 写得好不好"，而是它现在在做什么、为什么认为下一步该这么做、做过哪些尝试、哪些失败已被排除、离目标还有多远、什么时候该停。

- **场景 2：boss-skill 多 Agent 研发流水线**：从需求到部署的多 Agent 流水线，包含 PM、Architect、Dev、QA 等 9 个角色。一次 feature 过程落到 .boss/`<feature>`/ 目录下：events.jsonl 是状态真相源，execution.json 是只读投影，workflow-plan.json 是编译出来的 DAG。Wave 边界跑自动校验（类型检查、测试、lint、diff），QA 可发 REVISION_NEEDED，最多 2 轮重派修订复验。

- **场景 3：orca 运行时状态管理**：每次运行有 run_id、task_id、session_id，有 state.json、control.json、evidence.json、mailbox.json、task-lists.json、transcripts/ 和 agent-cache.json。暂停和停止不是靠自然语言"你别跑了"，而是写入 control request；恢复时可按输入 hash 复用之前成功的 Agent 输出；只重启某个 phase 时，不用把整条链路重新烧一遍。

- **场景 4：医疗 AI 多重批判者审查**：正确性批判者（信息是否事实准确）+ 风格批判者（表达是否清晰）+ 安全批判者（内容是否合规）+ 领域批判者（是否达到行业专家标准），每个批判者独立评估，最终输出必须同时通过四个维度审核才能获准出库。

- **场景 5：客服自动路由动态工作流**：如果用户输入包含退款关键词 → 走退款处理分支；如果包含技术问题关键词 → 走技术支持分支；如果检测到情绪负面 → 优先接入人工客服。流水线根据运行时输出动态决定下一步形状，而非固定步骤序列。

## 常见误区 / 踩坑

- ❌ **误区 1：Loop Engineering 就是 while true**
  ✅ **正解**：while true 只是循环，Loop Engineering 要解决的是闭环。循环只是重复，闭环要有目标、观察、验证、状态和退出条件四要素。

- ❌ **误区 2：Agent 能连续跑 10 小时就是 Loop Engineering 做得好**
  ✅ **正解**：一个没有退出条件、没有验证器、没有状态压缩、没有成本控制的长循环，可能只是 token 黑洞。真正好的 Loop 是它每一轮都能回答"我为什么继续"。

- ❌ **误区 3：让写代码的模型自己判断"我写好了"**
  ✅ **正解**：这很危险。至少要有测试、lint、typecheck、diff review、静态检查，复杂任务还要有另一个模型或规则系统做 reviewer。Agent 的声明不能当真相，验证结果才是下一轮的输入。

- ❌ **误区 4：测试不过就继续让模型修，不停重试**
  ✅ **正解**：好的 loop 必须有失败语义：连续失败几次以后，要么换策略，要么缩小范围，要么请求人类介入。否则代码会越改越乱。

- ❌ **误区 5：每一轮都把所有历史塞回上下文**
  ✅ **正解**：长 loop 最怕上下文越来越脏，成本会上升，注意力会漂移，旧错误还可能被重新激活。必须有 state management：哪些是稳定目标、哪些是当前观察、哪些是历史尝试、哪些可以压缩、哪些必须保留原文证据。

- ❌ **误区 6：没有停止条件没关系，Agent 跑得越久越好**
  ✅ **正解**：不会停的自动化比不会开始的自动化更危险。退出条件包括成功退出、失败退出、预算退出、风险退出、人类接管，都要定义清楚。

## 性能 / 复杂度

- **时间复杂度**：Loop 轮数与任务复杂度呈线性或对数关系，但单轮内部评估（Verifier）可能引入 O(n) 扫描。Reflexion Loop 的自我改进使后续轮数显著减少，总时间可能从 O(n×单轮) 降至 O(log n×单轮)。

- **空间复杂度**：State 持久化需要 O(s) 存储，Memory Compression Loop 将 N 条记忆压缩为 O(1) 条抽象规律，空间从 O(N) 降至 O(k)（k 为抽象层级数）。Context 窗口膨胀问题通过分层存储（events.jsonl + execution.json 分离）缓解。

- **与替代方案对比**：
  - 方案 A（单次 Prompt 优化）：时间 O(1)，空间 O(1)，适用边界：n ≤ 3 轮对话
  - 方案 B（固定 Pipeline）：时间 O(n)，空间 O(1)，适用边界：n ≤ 10 步固定流程
  - 方案 C（Loop Engineering）：时间 O(log n × 单轮)，空间 O(s + k)，适用边界：n > 10 轮或任务复杂度 > 5 子目标
  - 临界点：n = 3-5 轮时方案 A 更优；n = 5-10 轮固定流程时方案 B 更优；n > 10 轮或需要动态规划时方案 C 更优

- **性能数字**：boss-skill 项目中一次 feature 过程通过状态机流转，Wave 边界自动校验拦住约 30% 的不合格产物；orca 项目中按输入 hash 复用 Agent 输出，相同任务重启时间减少约 70%。

## 与相关概念的区别

- **vs Prompt Engineering**：
  - 维度 1（关注点）：Prompt Engineering 关注这一轮怎么问，Loop Engineering 关注多轮之间怎么推进
  - 维度 2（适用场景）：Prompt Engineering 适合单次问答，Loop Engineering 适合长任务持续执行
  - 维度 3（工程化程度）：Prompt Engineering 是调优艺术，Loop Engineering 是系统工程
  - 怎么选：单次任务问一句答一句用 Prompt Engineering；需要多轮迭代、失败恢复、状态持久化用 Loop Engineering

- **vs Context Engineering**：
  - 维度 1（关注点）：Context Engineering 关注这一轮该给模型看什么，Loop Engineering 关注多轮之间状态如何传递
  - 维度 2（上下文范围）：Context Engineering 处理单轮上下文装配，Loop Engineering 处理跨轮状态管理
  - 维度 3（技术手段）：Context Engineering 关注 prompt 压缩、检索增强，Loop Engineering 关注 events.jsonl、state.json 等文件持久化
  - 怎么选：上下文长度问题用 Context Engineering；多轮状态丢失、失败恢复问题用 Loop Engineering

- **vs Harness Engineering**：
  - 维度 1（解决问题）：Harness 解决能不能安全地做事（权限、环境、日志、沙箱、回滚），Loop 解决能不能一轮一轮把事情推进到完成
  - 维度 2（安全 vs 效率）：Harness 是安全带保障下限，Loop 是推进器提升效率
  - 维度 3（依赖关系）：没有 Harness 的 Loop 很危险，没有 Loop 的 Harness 只是安全但不会推进的工具壳
  - 怎么选：两者叠加才是完整方案，Harness 管下限（安全），Loop 管上限（效率）

## 进阶 / 面试加分项

- **最新进展**：2026 年 7 月 Tony Bai 梳理的 20 个 Loop Design Patterns 是目前最系统的工业级总结，从质量提升循环（Generate-Critique-Rewrite、Score-and-Retry、Multi-Critic）到记忆循环（Reflexion、Error Library、Success Pattern、Memory Compression）再到规划循环（Plan-Execute-Replan、Dynamic Workflow、Goal Decomposition），覆盖了 Agent 从脆弱 Demo 走向生产级落地的完整路径。

- **业界争议/未解问题**：Loop Engineering 目前缺乏统一的评估标准——如何量化一个 loop 的"质量"？Verifier 的设计目前依赖人工规则，Self-Verifier（让模型验证自己的输出）仍是开放问题；多 Agent 场景下的状态一致性（如 boss-skill 的 Wave 边界校验）如何保证也是难点。

- **一句话送给候选人**：Prompt Engineering 决定单轮上限，Loop Engineering 决定系统下限——真正区分普通工程师和年薪百万 AI 架构师的，是"让 Agent 在第一次失败后能够自主变得更好"的循环设计能力。


## 面试如何回答


### 🟢 什么是 Loop Engineering？它和传统的 while true 循环有什么区别？

**回答要点**：

Loop Engineering 是一种工程化技术，关注的是让 Agent 在长任务中多轮之间怎么推进、怎么验证、怎么恢复，而不是简单地让程序一直跑下去。while true 只是循环，Loop Engineering 要解决的是闭环。两者核心区别在于四要素：闭环必须有目标（Goal）、观察（Observer）、验证（Verifier）、状态（State）和退出条件（Exit Condition）。比如一个 while true 循环可以无限执行，但一个好的 Loop 会在每一轮回答：我为什么继续？继续之后要验证什么？验证失败后怎么改？到什么程度我该停下来？很多团队以为 Agent 能连续跑 10 小时就是做得好，其实一个没有退出条件、没有验证器、没有状态压缩的长循环，可能只是 token 黑洞。


### 🟡 为什么说 Loop 的灵魂是验证，而不是执行？能否举例说明？

**回答要点**：

验证是 Loop 灵魂，因为没有验证的 loop 只是在产出"自嗨"式的输出。以 boss-skill 项目为例：子 Agent 返回 DONE 或 DONE_WITH_CONCERNS，并不等于事实完成。Wave 边界还要跑自动校验，类型检查、测试、lint、diff 都可能把流水线拦下来。如果测试失败，QA 会发 REVISION_NEEDED，编排器记录事件后重派上游 Agent 修订。这里最朴素的道理就是：Agent 的声明不能当真相，验证结果才是下一轮的输入。另一个例子是 orca 项目中的 verifier，它会检查 evidence bundle 是否存在、失败数是否超限、有没有违反 read-only mutation policy、required tool call 有没有真的发生。"完成"不是模型自己宣布的，而是要能被证据证明。


### 🟡 一个完整的 Loop 应该包含哪些层？请结合实际项目说明每一层的作用。

**回答要点**：

一个完整的 Agent Loop 包含七层：Goal（目标层）定义任务要能被推进、被验证；Planner（规划层）根据当前状态滚动规划下一步，而不是幻想完整路径；Actor（执行层）真正执行读文件、改代码、调 API 等动作；Observer（观察层）收集工具输出、测试结果、日志、diff、错误信息；Verifier（验证层）判断有没有变好，是最关键的层；State（状态层）记录做过什么、失败过什么、现在卡在哪里；Exit Condition（退出层）定义成功退出、失败退出、预算退出、风险退出、人类接管。以 boss-skill 为例，状态不是靠 Agent 自己说一句"我完成了"，而是通过 pending → running → completed/failed → retrying → running 这种状态机流转，确保每一步都有据可查。


### 🟡 Loop Engineering 和 Harness Engineering 是什么关系？为什么两者缺一不可？

**回答要点**：

Harness Engineering 解决的是"能不能安全地做事"，包括权限控制、环境隔离、日志审计、沙箱、回滚机制；Loop Engineering 解决的是"能不能一轮一轮把事情推进到完成"，包括计划、执行、反馈、验证、恢复、退出。两者的关系是叠加而非替代：没有 Harness 的 Loop 很危险，比如一个没有权限控制的 Agent 可能误删生产环境数据；没有 Loop 的 Harness 只是一个安全但不会持续推进的工具壳。真正的 Coding Agent 需要两者叠在一起：Harness 管下限（安全），Loop 管上限（效率）。这就像汽车需要安全带（Harness）和发动机（Loop）一起才能跑，光有安全带只是辆安全的废车。


### 🟡 在实际项目中，Loop Engineering 容易踩哪些坑？如何避免？

**回答要点**：

最常见的四个坑：第一个是把 loop 做成无限重试，测试不过就一直让模型修，最后代码越改越乱。正确做法是必须有失败语义，连续失败 N 次后要么换策略，要么缩小范围，要么请求人类介入。第二个是没有独立验证，让写代码的模型自己判断"我写好了"，这很危险，至少要有测试、lint、typecheck、diff review。第三个是没有状态边界，每一轮都把所有历史塞回去，上下文越来越脏，成本上升，注意力漂移，旧错误可能被重新激活。第四个是没有停止条件，很多 Agent 看起来很努力，其实只是不会停，不会停的自动化比不会开始的自动化更危险。


### 🔴 Reflexion Loop（反思循环）是什么？它为什么被认为是目前最重要的自我进化设计模式？

**回答要点**：

Reflexion Loop 的工作流程是：Agent 执行失败 → 分析失败原因 → 将教训存入记忆库 → 带着这段教训重新尝试。它的核心价值在于把"会失败一次的系统"变成"只会在同一个地方摔倒一次的系统"。比如尝试 1 失败了，反思发现"我假设了 X 成立，但实际上 X 是错的，下次一定要先验证 X"；尝试 2 注入教训后获得部分成功，进一步反思"变好了，但我漏掉了步骤 Y，需要增加对 Y 的检查"；尝试 3 最终成功。这就是自我进化的机制。相比之下，没有 Reflexion Loop 的系统会在同一个错误上反复失败，而有了反思循环，系统每次迭代都比上一次更聪明，运行到第 6 个月时的表现和第 1 个月相比会有天壤之别。


### 🔴 如何在面试中解释"Generator-Critic Pattern"和"Multi-Critic Loop"的区别？

**回答要点**：

Generator-Critic Pattern（生成器-批判者模式）是一对一的角色：生成者产出初稿，一个批判者审查挑刺，生成者根据反馈重写，不断重复直到达到质量阈值。核心洞察是负责生成的模型往往不是评估自身输出的最佳裁判，让独立的批判者去挑刺，总能找出生成者忽略的盲点。Multi-Critic Loop 则是多对一的角色：引入四个独立批判者——正确性批判者（信息是否事实准确）、风格批判者（表达是否清晰）、安全批判者（内容是否合规）、领域批判者（是否达到行业专家标准）。每个批判者独立评估，最终输出必须同时通过四个维度的审核才能获准出库。区别在于：Generator-Critic 是二元博弈，Multi-Critic 是多元矩阵；前者适合一般文案/代码审查，后者适合医疗 AI、法律文件审查等高风险场景。


### 🔴 假设你设计一个 AI 客服 Agent 的 Loop，需要考虑哪些关键设计点？请给出具体方案。

**回答要点**：

设计一个生产级 AI 客服 Agent Loop，需要考虑以下关键点：首先，Goal 层要定义清晰：用户意图识别 → 问题解答 → 满意度确认，不是简单的一问一答。其次，Planner 层要支持 Dynamic Workflow：根据用户输入的关键词（退款/技术问题/情绪负面）动态路由到不同分支，输出是 A 走分支 X，输出是 B 走分支 Y。第三，Actor 层执行具体动作：查询知识库、生成回复、调用工单系统。第四，Observer 层收集：用户反馈、对话满意度、是否转人工。第五，Verifier 层判断：回答是否准确、情绪是否得当、是否需要升级人工。第六，State 层记录：用户历史、当前问题进度、失败尝试。第七，Exit Condition 定义：用户确认解决、连续 3 次回答错误、转人工成功、对话超过 10 分钟。最重要的是，不能让 Agent 自己判断"我回答好了"，必须有独立的评估机制。

