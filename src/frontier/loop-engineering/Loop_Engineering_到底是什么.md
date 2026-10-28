---
title: Loop Engineering 到底是什么？
icon: mdi:refresh
date: 2026-07-16
tag:
  - Loop Engineering
  - Agent
---

# Loop Engineering 到底是什么？

> 一句话核心：Loop Engineering 是围绕 Agent 设计可持续运行的反馈循环，解决的是"多轮之间怎么推进"而非"单轮 prompt 怎么写"，让 Agent 在明确目标、工具、上下文、验证信号和停止条件下反复行动，直到任务完成、失败或需要人工接管。
## 核心概念（术语表）

- **Loop Engineering**：围绕 Agent 设计可持续运行的反馈循环，让 Agent 一轮一轮把事情做成，核心是多轮之间的推进机制
- **Inner Loop（内层循环）**：Agent 自己每一轮"推理、行动、观察"的循环，由 ReAct 模式驱动，停止条件是"不再需要工具"
- **Outer Loop（外层循环）**：调度系统或人写的流程，负责唤醒 Agent、分配任务、验证结果、记录状态，停止条件是"达成目标、超预算、失败转人工"
- **Harness Engineering**：Agent 的安全带、仪表盘和执行外壳，解决"能不能安全地做事"，包含沙箱、权限、审计、回滚
- **Context Engineering**：每一轮该给 Agent 看什么，解决上下文怎么装配，让验证结果成为下一轮的输入
- **Verification（验证层）**：判断有没有变好的一层，没有 verifier 的 loop 很容易自嗨，必须有独立验证
- **State Persistence（状态持久化）**：长任务不能只活在上下文窗口里，必须记录做过什么、失败过什么、现在卡在哪里
- **Exit Condition（退出条件）**：成功退出、失败退出、预算退出、风险退出、人类接管，都要定义清楚
- **Closed Loop（闭环）**：不是 while true 的循环，而是要有目标、观察、验证、状态和退出条件，循环只是重复，闭环要有目标
<ReadMoreLock />

## 历史背景 / 来源

Loop Engineering 这个词大约在 2026 年 6 月上旬开始热起来。Addy Osmani 在 2026 年 6 月 7 日写了篇 Loop Engineering，随后 AI 圈开始反复讨论"让系统去提示 Agent"。但它并不是全新概念，而是 Agent Loop、Workflow Graph、Context Engineering、Skills、MCP、CI、测试验证这些已有东西的重新包装和系统化整合。Claude Code、Codex 里的 /loop、/goal、Automations、Skills、Sub-agent、工作区隔离、MCP/Connector，解决的都是同一类问题：别让 Agent 只停在一轮回答里，给它边界，让它继续干活。

## 工作原理 / 核心机制

Loop Engineering 的整体思路是：不只是盯着"下一句 Prompt 怎么写"，而是设计一个可持续运行的反馈系统，让 Agent 在长任务里一轮一轮把事情做成。

输入是：一个明确的目标（如"全部测试通过"、"CI green"、"覆盖率到 80%"），可用的工具集，初始上下文，当前状态。

输出是：完成的任务产物、验证证据、执行日志、状态快照，或者失败报告和人工接管请求。

**核心机制分三层展开**：

**第一步：Goal 定义层**。目标不能只是"帮我做一下"，而要能被推进、被验证。比如"修复这个 bug"不如"修改该函数后，测试用例 T-123 从 FAIL 变 PASS，lint 无 warning"来得具体。目标的颗粒度决定 loop 能不能收敛。

**第二步：Executor + Observer 双轨**。Actor（执行器）负责真正执行动作：读文件、改代码、调 API、跑命令、查资料。Observer（观察器）负责观察结果：工具输出、测试结果、日志、diff、错误信息，都要回到系统里。这两者必须分开，不能让写代码的模型自己判断"我写好了"。

**第三步：Verifier + State + Policy 三保险**。Verifier（验证器）判断有没有变好，是闭环的关键。State（状态）记录做过什么、失败过什么、现在卡在哪里、下一步为什么这么选，都要留下来。Policy（策略约束）管权限、成本、模型路由、重试次数、敏感操作审批。一个典型的 Coding Agent loop 不是"生成代码 -> 再生成代码 -> 再生成代码"，而是：先根据目标拆计划 -> 读相关文件 -> 做最小修改 -> 跑测试 -> 测试失败就分析失败原因 -> 如果是自己改坏了就回到修改环节，如果是环境问题就标记阻塞 -> 测试通过后看 diff 有没有越界 -> 最后生成证据。

## 关键知识点

- Loop Engineering 关注的是"多轮之间怎么推进"，不是"单轮 prompt 怎么写"
- 闭环 = 目标 + 观察 + 验证 + 状态 + 退出条件，循环只是重复，闭环要有目标
- 内层循环是 Agent 自己每轮的"推理、行动、观察"，外层循环是调度系统负责"唤醒、分配、验证、记录"
- 一个已运行 1d 19h 48m 的 Codex goal 任务，真正重要的是"它为什么继续跑"而不是"它还在跑"
- 没有 verifier 的 loop 很容易自嗨，Agent 的声明不能当真相，验证结果才是下一轮的输入
- 长任务不能只活在上下文窗口里，必须有状态持久化（events.jsonl、execution.json、workflow-plan.json）
- Harness 管"能不能安全地做事"，Loop 管"能不能一轮一轮把事情推进到完成"
- Context Engineering 解决"每一轮该给 Agent 看什么"，因为上下文越来越脏会导致成本上升和注意力漂移
- 不会停的自动化比不会开始的自动化更危险，必须定义成功退出、失败退出、预算退出、风险退出、人类接管
- boss-skill 做的是工作流层的 loop（需求、产物、DAG、门禁、反馈、重派），orca 做的是运行时层的 loop（状态持久化、暂停恢复、失败重启、缓存复用、证据验证）
- 子 Agent 返回 DONE 或 DONE_WITH_CONCERNS 不等于事实完成，Wave 边界还要跑自动校验
- 代码 Agent 真能连续读文件、改代码、跑命令、处理 PR 之后，不能只盯着"下一句 Prompt 怎么写"
- Loop Engineering 的本质是：让 Agent 每跑一轮都离目标更近，并且知道什么时候该停
## 应用场景

- **场景 1（Codex goal 长任务）**：用户启动一个持续目标任务，Agent 不断读取上下文、判断进展、执行动作、验证结果，遇到问题继续调整，任务可能持续 1d 19h+。解决的问题是：人不需要守在对话框前，Agent 被叫醒后自己读材料、跑命令、写状态，卡住再把问题抛回来，节省 token 和人工成本。

- **场景 2（boss-skill 多 Agent 流水线）**：一个"从需求到部署"的多 Agent 研发流水线，有 PM、Architect、Dev、QA 等角色，通过 events.jsonl（状态真相源）、execution.json（只读投影）、workflow-plan.json（编译出来的 DAG）管理状态。解决的问题是：让 9 个 Agent 不乱跑，下一个 Agent 不是凭感觉派出去的，而是由当前状态、产物依赖、门禁结果一起算出来的。

- **场景 3（orca 运行时级控制）**：底层运行时支持 run/list/show/stop/pause/resume/clone/restart-failed/restart-phase，每次运行有 run_id、task_id、session_id，有 state.json、control.json、evidence.json、mailbox.json、transcripts/。解决的问题是：长任务的状态、控制、缓存、证据和 transcript 都能落下来，暂停和停止不是靠自然语言，而是写入 control request。

- **场景 4（CI 触发 Loop）**：CI 失败后自动触发 Agent 读取日志、定位问题、尝试修复、再次跑测试。解决的问题是：传统 CI 失败需要人工介入，Loop 可以让 Agent 先尝试自动修复，减少人工轮次。

- **场景 5（PR 评论 Loop）**：PR 创建后 Agent 读取代码 diff、运行 lint 和测试、生成 review 意见，如果测试失败则自动分析原因并尝试修复。解决的问题是：让 code review 的反馈循环自动化，Agent 在 CI green 之前持续尝试改进。

## 常见误区 / 踩坑

- ❌ **误区 1：Loop Engineering 就是 while true**
  ✅ **正解**：while true 只是循环，Loop Engineering 要解决的是闭环。循环只是重复，闭环要有目标、观察、验证、状态和退出条件。没有退出条件的循环可能只是 token 黑洞。

- ❌ **误区 2：Agent 能连续跑 10 小时就是 Loop 做得好**
  ✅ **正解**：真正好的 Loop 不是"它一直在跑"，而是它每一轮都能回答"我为什么继续、继续之后要验证什么、验证失败以后我怎么改、到什么程度我应该停下来"。没有 verifier、没有状态压缩、没有成本控制的长循环可能只是浪费 token。

- ❌ **误区 3：把 Agent 自评当验收**
  ✅ **正解**：让写代码的模型自己判断"我写好了"非常危险。至少要有测试、lint、typecheck、diff review、静态检查，复杂任务还要有另一个模型或规则系统做 reviewer。Agent 的声明不能当真相，验证结果才是下一轮的输入。

- ❌ **误区 4：loop 做成无限重试**
  ✅ **正解**：测试不过就继续让模型修，修不过就再修，再不过还修，这种 loop 最后会把代码越改越乱。好的 loop 必须有失败语义：连续失败几次以后，要么换策略，要么缩小范围，要么请求人类介入。

- ❌ **误区 5：没有状态边界**
  ✅ **正解**：长 loop 最怕上下文越来越脏。每一轮都把所有历史塞回去，成本会上升，注意力会漂移，旧错误还可能被重新激活。必须有 state management：哪些是稳定目标，哪些是当前观察，哪些是历史尝试，哪些可以压缩，哪些必须保留原文证据。

- ❌ **误区 6：没有停止条件**
  ✅ **正解**：很多 Agent 看起来很努力，其实只是不会停。不会停的自动化比不会开始的自动化更危险。必须定义：成功退出（达成目标）、失败退出（连续失败 N 次）、预算退出（token 耗尽）、风险退出（触发安全规则）、人类接管（遇到未知情况）。

## 性能 / 复杂度

- **时间成本**：每次 loop 轮次 = LLM 调用时间 + 工具执行时间 + 验证时间 + 状态读写时间。总成本与轮次线性相关，但每轮 token 消耗会随上下文增长而增加（如果不做状态压缩）。
- **空间成本**：状态持久化需要存储 execution.json、events.jsonl、transcripts/ 等文件。典型规模：单个长任务可能产生 10-50MB 的运行时文件。
- **与替代方案对比**：
  - 方案 A（纯 Prompt Engineering）：时间 O(1) 单轮调用，空间 O(1)，适合简单查询但无法处理长任务
  - 方案 B（本方案 Loop Engineering）：时间 O(k) k 轮循环，空间 O(k) 随轮次增长，适合长任务但有成本上限
  - 临界点：任务需要超过 3-5 轮交互时，Loop Engineering 的总体成本（人工 + token）开始优于纯 Prompt Engineering

## 与相关概念的区别

- **vs Prompt Engineering**：
  - 维度 1（关注点）：Prompt Engineering 关注的是"这一轮怎么问"，Loop Engineering 关注的是"多轮之间怎么推进"
  - 维度 2（适用场景）：Prompt Engineering 适合简单查询和单轮任务，Loop Engineering 适合需要持续执行的长任务
  - 维度 3（工程量）：Prompt Engineering 只需要优化 prompt 本身，Loop Engineering 需要设计完整的触发、验证、状态、退出系统
  - 怎么选：简单问答用 Prompt Engineering，长任务自动化用 Loop Engineering

- **vs Context Engineering**：
  - 维度 1（粒度）：Context Engineering 解决"这一轮该给模型看什么"，Loop Engineering 解决"这一轮做完之后下一轮怎么推进"
  - 维度 2（层次）：Context Engineering 是 Loop 的输入层，Loop Engineering 是横跨多轮的编排层
  - 维度 3（目标）：Context Engineering 优化单轮信息密度，Loop Engineering 优化多轮状态连续性
  - 怎么选：先做好 Context Engineering，再构建 Loop Engineering，两者是正交关系

- **vs Harness Engineering**：
  - 维度 1（解决的问题）：Harness 解决"能不能安全地做事"，Loop 解决"能不能一轮一轮把事情推进到完成"
  - 维度 2（组件类型）：Harness 包含沙箱、权限、审计、回滚，Loop 包含 Planner、Actor、Observer、Verifier
  - 维度 3（关系）：没有 Harness 的 Loop 很危险，没有 Loop 的 Harness 只是安全但不会持续推进的工具壳
  - 怎么选：两者必须叠加使用，Harness 管边界安全，Loop 管任务推进

## 进阶 / 面试加分项

- **最新进展**：2026 年 6 月 Addy Osmani 首次系统阐述 Loop Engineering 后，Coding Agent 领域竞争点已从"模型能力"转向"谁能让 Agent 在长流程里不迷路、让失败变成下一轮改进信号、让人随时接管、让整个过程可观察可审计可恢复"。

- **业界争议**：Loop Engineering 是否只是新瓶装旧酒？观点是：新鲜的是名字和能力方向（持续执行），组件（Agent Loop、Workflow、Context、Harness、Skills、MCP）确实早已存在，关键在于系统化整合和边界设计。

- **金句**：Prompt Engineering 关注的是这一轮怎么问，Context Engineering 关注的是这一轮该给模型看什么，Harness Engineering 关注的是模型在真实环境里怎么安全执行，而 Loop Engineering 关注的是多轮之间怎么推进——它把 prompt、context、tool、memory、verification、policy 串成一个持续运转的系统。


## 面试如何回答


### 🟢 用一句话解释什么是 Loop Engineering，它和 Prompt Engineering 的核心区别是什么？

**回答要点**：

Loop Engineering 是围绕 Agent 设计可持续运行的反馈循环，解决的是"多轮之间怎么推进"的问题。Prompt Engineering 关注的是"这一轮怎么问"，Loop Engineering 关注的是"多轮之间怎么推进"。比如你让 Agent 修复一个 bug，Prompt Engineering 关心的是这个修复指令怎么写，而 Loop Engineering 关心的是：Agent 改完代码之后，下一步是跑测试还是看 diff？测试失败了怎么办？连续失败几次要换策略还是人工介入？这就是本质区别：Prompt Engineering 是单轮优化，Loop Engineering 是多轮系统设计。


### 🟡 一个完整的 Loop Engineering 系统应该包含哪些核心组件？请列举并说明每个组件的作用。

**回答要点**：

一个完整的 Loop 包含 8 个核心组件：Goal（目标）定义要能被推进和验证；Planner（下一步选择）根据当前状态滚动规划而非幻想完整路径；Actor（执行器）真正执行读文件、改代码、跑命令等动作；Observer（观察器）收集工具输出、测试结果、日志、diff；Verifier（验证器）判断有没有变好，这是最关键的一层；State（状态）记录做过什么、失败过什么、现在卡在哪里；Exit Condition（退出条件）定义成功退出、失败退出、预算退出、风险退出、人类接管；Policy（策略约束）管权限、成本、模型路由、重试次数。缺少任何一个组件，loop 都会出问题：没有 Verifier 容易自嗨，没有 Exit Condition 会无限跑，没有 State 就无法在失败后恢复。


### 🟡 Loop Engineering 和 Harness Engineering 是什么关系？为什么说两者必须叠加使用？

**回答要点**：

Harness 是 Agent 的安全带、仪表盘和执行外壳，解决"能不能安全地做事"，包含沙箱、权限、审计、回滚。Loop 是 Agent 的任务推进机制，解决"能不能一轮一轮把事情推进到完成"，包含 Planner、Actor、Observer、Verifier。两者不是替代关系，而是正交叠加：没有 Harness 的 Loop 很危险，比如 Agent 可能误删生产环境数据；没有 Loop 的 Harness 只是一个安全但不会持续推进的工具壳。真正的 Coding Agent 需要两者叠在一起：Harness 管权限、环境、日志、沙箱、回滚，Loop 管计划、执行、反馈、验证、恢复、退出。类比一下：Harness 像是汽车的刹车和安全气囊，Loop 像是油门和方向盘，缺一不可。


### 🟡 在 Loop Engineering 中，为什么说"Agent 的声明不能当真相"？如何设计有效的验证机制？

**回答要点**：

因为 Agent 自己判断"我写好了"存在严重的利益冲突：它需要完成任务就会倾向于高估自己。必须设计独立的验证层。具体做法：至少要有测试（单元测试、集成测试）、lint（代码风格检查）、typecheck（类型检查）、diff review（变更审查），复杂任务还要有另一个模型或规则系统做 reviewer。以 boss-skill 为例：子 Agent 返回 DONE 或 DONE_WITH_CONCERNS 不等于事实完成，Wave 边界还要跑自动校验，QA 如果认为代码有问题可以发 REVISION_NEEDED，编排器会记录事件、重派上游 Agent 修订、再让当前 Agent 复验，而且最多 2 轮。验证结果才是下一轮的输入，这个原则必须坚守。


### 🟡 Loop Engineering 中最容易踩的坑有哪些？请至少说出 4 个并说明如何避免。

**回答要点**：

第一个坑是做成无限重试：测试不过就继续修，修不过还修，最后代码越改越乱。正确做法是定义失败语义：连续失败 2-3 次后，要么换策略，要么缩小范围，要么请求人工介入。第二个坑是没有独立验证：让写代码的 Agent 自己判断完成与否。必须分离 Executor 和 Verifier，用测试/lint/diff 作为客观证据。第三个坑是没有状态边界：每轮把所有历史塞回去，导致上下文越来越脏、成本上升、注意力漂移。必须有状态压缩策略，区分稳定目标、当前观察、历史尝试。第四个坑是没有停止条件：Agent 只会跑不会停。必须定义成功退出（达成目标）、失败退出（连续失败 N 次）、预算退出（token 耗尽）、风险退出（触发安全规则）、人类接管（遇到未知情况）。


### 🔴 你如何理解 Loop Engineering 的本质？为什么说它不是新概念？

**回答要点**：

Loop Engineering 的本质是"让 Agent 每跑一轮都离目标更近，并且知道什么时候该停"。它不是全新概念，而是老概念的重新包装和系统化整合：Agent Loop/ReAct（内层循环早就存在）、Workflow/Graph/Loop（可控回边早就有）、Context Engineering（每轮给 Agent 看什么早就有人研究）、Harness Engineering（模型外面的执行环境）、Skills（把重复解释的经验写下来）、MCP（让 Loop 能接触真实工具）。Addy Osmani 在 2026 年 6 月 7 日首次系统阐述，但 Claude Code、Codex 的 /loop、/goal、Automations 等功能早就实现了类似能力。新鲜的是名字和能力方向（从单次执行走向持续执行），组件确实早就存在，关键在于系统化整合和工程边界设计。


### 🔴 如果让你设计一个代码审查的 Loop Engineering 系统，请描述核心架构和数据流。

**回答要点**：

一个代码审查 Loop 的核心架构分为触发层、执行层、验证层、决策层。触发层由 PR 创建或更新事件唤醒 Loop，目标定义为"PR 通过审查，CI green，无重大风险"。执行层：Actor 读取 diff、运行 lint 和测试、分析代码质量规则。Observer 收集测试结果、lint 输出、复杂度报告、安全扫描结果。验证层：Verifier 对比测试通过率、lint warning 数量、代码覆盖率等指标与阈值。决策层：根据验证结果决定：测试全过且 lint clean 则标记通过并通知作者；测试失败则分析原因后重派作者修复（最多 2 轮）；遇到安全风险或未知规则则转人工审查。状态层记录审查历史、失败原因、修复轮次，确保可以断点恢复。这个 Loop 的关键是把"代码审查"从单次判断变成多轮闭环，让 Agent 持续优化直到满足质量门禁。


### 🔴 Loop Engineering 和传统的工作流引擎（如 Airflow）有什么本质区别？什么时候该用哪个？

**回答要点**：

传统工作流引擎（如 Airflow）是确定性任务编排，节点是固定的，节点之间的流转是预设好的，失败重试是节点级别的。Loop Engineering 是 AI 原生的不确定任务推进，节点本身由 LLM 动态决定下一步做什么，节点之间的流转根据观察结果动态调整。以代码审查为例：传统工作流是"固定流程：lint -> test -> security scan -> approve"，每个步骤的输入输出是确定的。Loop Engineering 是"动态流程：Agent 根据当前状态决定下一步是修复 lint 还是继续 test，验证失败后决定重试还是换策略"。选择标准：如果是确定性任务（ETL、报表、定时同步）用传统工作流引擎；如果是需要 AI 判断的不确定任务（代码生成、故障排查、需求分析）用 Loop Engineering。实际生产中两者可以叠加：用工作流引擎调度多个 Loop，每个 Loop 内部用 AI 动态决策。

