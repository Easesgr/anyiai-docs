---
title: Agent Skills 是什么？如何规范组织与按需调度复杂任务？
icon: mdi:cpu-64-bit
description: Agent Skills 是一套基于文件系统组织智能体能力的开放规范。本文系统讲解其渐进式加载机制、适用边界、专题学习路径及最小工程验证方法。
---

![Agent Skills 专题学习导引封面](/images/content-rebuild/bfaa96e3923b3e6c/cover.png)

设想你在构建一个企业内部的代码与运维智能体。它既要能查询内部工单、审查 Python 代码，又要能排查 Kubernetes 集群故障、导出 PDF 报表。如果把每一个任务的操作手册（SOP）、参数定义、边界约束和示例代码全量写入模型的系统提示词（System Prompt），很快就会遇到两个现实瓶颈：

1. **首 Token 延迟与费用上升**：不论用户提问只是简单的“你好”还是复杂的“集群崩溃排查”，每次调用都必须把数万 Token 的规则全量发送给大模型，首 Token 计算时间（TTFT）与调用成本随之大幅增加。
2. **规则干扰与检索衰减**：长上下文中包含过多互不相干的操作指南时，模型更容易忽略针对当前任务的关键约束，甚至在不同业务规则之间产生混淆。

**Agent Skills 规范**正是为了解决这类能力组织与调度问题而演进出的开放文件约定（如 Agent Skills 官方规范）。它不是新的模型结构，也不是独立的执行运行时，而是一套**基于文件系统的目录与元数据标准**。通过将操作说明、专用脚本和参考资料组织为规范化文件包，智能体宿主程序可以在会话中按需检索并加载具体能力。

本专题旨在为开发者提供清晰、可验证的 Agent Skills 知识体系，帮助你理解它的工作原理、编写方法与适用边界。

## 渐进式加载如何平衡上下文与能力深度

Agent Skills 解决上下文瓶颈的核心机制是**渐进式披露（Progressive Disclosure）**。宿主程序与大模型配合，将能力的加载拆分为三个阶段：

<ReadMoreLock />

![渐进式上下文加载机制三层结构](/images/content-rebuild/bfaa96e3923b3e6c/fig-1.png)

1. **元数据常驻（Metadata Tier）**：在智能体启动或初始化会话时，宿主程序仅将每个技能的元数据（主要为 `SKILL.md` 中的 `name` 与 `description`）注入上下文，每个技能通常仅占几十到一百 Token。此时模型知道系统具备哪些能力，以及各自适用于什么场景。
2. **指令按需加载（Instruction Tier）**：当用户输入的任务明确匹配某项技能的描述时，宿主程序（或由模型通过文件读取工具）才将该技能的完整 `SKILL.md` 正文读入当前对话上下文。官方规范建议将主指令控制在 5000 Token 以内，确保指引明确且不挤占过多工作空间。
3. **资源与脚本按需调取（Resource Tier）**：技能目录中的辅助文件（如 `references/` 下的细分领域字典、`scripts/` 下的可执行脚本、`assets/` 下的模板文件），只有在模型执行具体步骤确实需要时才被按需读取或调用，不提前占用上下文。

### 机制边界与工程权衡

必须明确，**渐进式加载是一种工程上的预算权衡，并非消除上下文限制的绝对保证**：

* **语义匹配存在概率偏差**：宿主或模型依靠 `description` 判断是否激活技能。如果描述过于简略或关键词重叠，可能导致技能误激活或漏激活。
* **多技能累积仍会撑爆上下文**：在一个长会话中，如果任务连续触发了 5 个不同技能，每个技能正文和执行日志相加，上下文依然会迅速膨胀，必须配合会话截断或状态持久化机制。
* **本身不提供安全沙箱**：Skill 目录下的脚本不能直接赋予执行权限，高危操作（如删除数据库、修改系统配置）必须由宿主运行时实施拦截、权限隔离或人工介入确认（Human-in-the-loop）。

## 本专题的学习路径与模块递进

为了帮助开发者建立连贯的技术认知，本专题按“认知定位 → 编写标准 → 工程落地”三个层级展开：

![Agent Skills 专题学习路径与知识递进图](/images/content-rebuild/bfaa96e3923b3e6c/fig-2.png)

1. **第一篇：概念与定位**  
   [Agent Skills 是什么？它和普通 Prompt、工具调用有什么区别？](what-is-agent-skills.md)  
   理清 System Prompt、Tool Calling、MCP（模型上下文协议）与 Agent Skills 之间的职责边界。解释为什么 Skill 不是单纯的函数调用，而是一套包含意图引导与多步规程的复合组织形式。
2. **第二篇：规程与结构**  
   [一个可复用的 Agent Skill 应该包含哪些内容？说明、输入、步骤和边界怎么写？](reusable-skill.md)  
   逐字段解析官方规范的 YAML Frontmatter（如 `name`、`description`、`compatibility`）以及 Markdown 正文的推荐结构，掌握输入验证、容错规程与清晰边界的编写范式。
3. **第三篇：封装与实战**  
   [如何把常用任务封装成 Skills，让 Agent 更稳定地完成复杂工作？](skill-encapsulation.md)  
   结合具体工程场景，展示如何将确定性计算下沉到脚本（Script）、将意图理解留给模型、并结合宿主框架（如 LangGraph 等状态机或 CLI 环境）完成具备人工审批、持久化状态的高可靠封装。

如需深入了解底层大语言模型的上下文机制与系统架构，可在学习本专题前后参考 [大模型基础](/fundamentals/llm-basics/README.md) 与 [AI 工程架构](/architecture/README.md)。

## 智能体宿主环境的职责分工

Agent Skills 是一组静态文件，**它自身不具备代码执行或流程调度的驱动能力**。要让技能生效，必须依托支持该规范的宿主环境（例如 Claude Code，或基于 LangChain、LangGraph 自行实现的 Agent 运行时）。理解两者分工是工程落地的关键：

* **大模型（LLM）的角色**：根据系统上下文中的元数据理解用户意图；在需要时产出读取 `SKILL.md` 或执行某段脚本的结构化指令（Tool Call）；按照技能正文给出的步骤分步推理并组织返回结果。
* **宿主程序（Host Runtime）的角色**：在系统启动时扫描技能目录并提取元数据；拦截模型的工具调用请求，完成物理磁盘文件的读取；在沙箱环境中运行脚本并回传输出结果；负责持久化会话状态与拦截未授权操作。

## 最小可验证 Skill：编写与双向验收

学习本专题的第一步，是在本地环境编写一个符合规范的最小技能，并通过“格式校验”与“行为触发”完成闭环验证。

![最小可验证 Skill 的目录与检验流程](/images/content-rebuild/bfaa96e3923b3e6c/fig-3.png)

### 1. 目录与文件编写

新建一个名为 `csv-line-counter` 的文件夹（必须与技能名称完全一致），并在其中创建 `SKILL.md` 文件：

```markdown
---
name: csv-line-counter
description: Counts the total number of lines in a user-provided CSV file. Use when the user asks for CSV row counts or basic line statistics.
---

## Instructions
1. Check if the provided file path ends with `.csv` and exists on disk.
2. Run a line count command or use a simple script to count rows.
3. Return the total line count and explicitly note whether the header row is included in the count.
```

编写时需严格遵循规范约束：
* `name` 长度为 1-64 字符，仅允许小写英文字母、数字和连字符（`-`），不能以连字符开头或结尾，不能包含连续连字符（如 `--`），且必须与父目录名一致。
* `description` 长度为 1-1024 字符，必须同时写清技能做什么以及何时使用。

### 2. 验收两步法

#### 第一步：静态格式校验
使用规范参考工具校验文件头格式是否合规：

```bash
# 使用 Agent Skills 官方校验工具进行静态格式检查
skills-ref validate ./csv-line-counter
```

如果元数据格式、字符数或命名规则有误，工具会明确指出具体的字段错误；校验通过则输出成功提示。

#### 第二步：动态触发测试
在支持技能发现的宿主环境中，提供两类测试指令，观察模型的决策路径：

* **正向命中测试**：“请帮我看一下 `data/orders.csv` 一共有多少行。”  
  *验收指标*：模型能匹配到 `csv-line-counter` 的元数据，宿主加载 `SKILL.md`，模型按规程要求确认文件并区分表头做出回复。
* **负向未命中测试**：“请把这段 Python 代码重构成 Go 语言版本。”  
  *验收指标*：模型直接响应重构请求，不会无故读取或调用 `csv-line-counter` 的正文，上下文保持纯净。

通过上述步骤，你就建立起了一个规范化、可验证的 Agent Skill 原型，可以继续深入阅读后续章节探索更复杂能力的封装模式。
