---
title: "一个可复用的 Agent Skill 应该包含哪些内容？说明、输入、步骤和边界怎么写？"
icon: mdi:file-document-outline
tag:
  - Agent Skills
  - 可复用设计
  - Skill 编写
  - 最佳实践
---

![COVER](/images/agent-skills/reusable-skill-cover.png)

# 一个可复用的 Agent Skill 应该包含哪些内容？说明、输入、步骤和边界怎么写？

面试里问到 Agent Skill 的设计规范，本质在考你两件事：**Agent 怎么找到正确的 Skill**、**Skill 怎么给 Agent 合适的指令**。这不光是背概念，得真正理解 Skill 的结构和加载机制，才能写出可复用的 Skill。

---

## Skill 是什么——像个「工具使用说明书」

一个 Skill 本质上是一个**目录**，里面装着一堆相关的东西。

最核心的是 `SKILL.md` 这个文件，它由两部分组成：

- **YAML Frontmatter**：放在文件顶头的元数据，比如 name、description、examples
- **Markdown Body**：指令正文，就是告诉 Agent 怎么执行这个任务

除了 SKILL.md，通常还会有配套的目录：

- `scripts/`：放代码脚本
- `references/`：放参考资料
- `assets/`：放图片、模板这类资源

整个结构就像你买电器附带的「快速上手指南」——告诉你这个工具是干嘛的、怎么用、有啥注意事项。

![Skill 目录结构图，展示 SKILL.md、scripts/、references/、assets/ 的层级关系](/images/agent-skills/20260716_194420_146944_01.png)

---


<ReadMoreLock />




## description 是灵魂——写得不好，Agent 根本不会用它

description 太重要了，它是 **Agent 匹配 Skill 的唯一依据**。上限 1024 字符，不能超过。

一个好的 description 要包含三点：

**1. 触发词**——什么情况下该用这个 Skill？
**2. 排除条件**——什么情况下不该用？
**3. 领域关键词**——让 Agent 精准识别

Anthropic 官方有个建议：**写主动一点**。因为 Claude 本身有「触发不够」的倾向，你得明确告诉它「遇到这类问题就用我」。

比如你写一个「代码审查 Skill」，不能光写「代码审查」，得写清楚触发场景、适用范围。

这就像写搜索引擎的关键词广告，既要让目标用户搜到，还得过滤掉不该看到的人。

![description 三要素结构图，展示触发词、排除条件、领域关键词的位置](/images/agent-skills/20260716_194419_669248_01.png)

---

## 渐进式披露——别一次性把内容全塞给 Agent

这是 Skill 设计里**最关键**的决策。

渐进式披露的意思是：Skill 内容不是一次性全给 Agent，而是分三个层级加载。

**Level 1（Metadata）**：所有 Skills 一起加载，只加载 name + description。这是最轻量的，Agent 根据这些决定用哪个 Skill。

**Level 2（Instructions）**：某个 Skill 被激活时，加载完整 SKILL.md 正文。这是主体内容。

**Level 3（Resources）**：指令中引用 scripts/、references/、assets/ 时，按需加载。比如你在正文里写「请参考 scripts/deploy.sh」，这时候才加载那个脚本。

类比一下，就像点外卖：先看菜单分类（Level 1），再点进某个菜看详情（Level 2），最后看食材来源（Level 3）。你不会一开始就把所有信息砸给用户。

有个实操建议：**超过 300 行就该考虑拆分了**。不是硬性规定，但超过这个量，Agent 的上下文负担会变重。

![渐进式披露三层加载流程图，展示 Level 1/2/3 的触发时机和数据量](/images/agent-skills/20260716_194423_402240_01.png)

---

## 14 个设计模式——根据场景灵活选用

Skill 设计有 14 个成熟模式，分成五大类。

### 发现与选择
- **激活元数据模式**：让 name 和 description 精准表达 Skill 的能力
- **排除条款模式**：这比正向触发更重要。明确告诉 Agent「什么情况下不要用我」，能大幅减少误触发

### 上下文经济
- **上下文预算模式**：选一个说法就一直用，别来回换词，增加 token 消耗
- **渐进式披露模式**：刚讲过，不再赘述

### 指令校准
- **控制调优模式**：决定用文本指令（开放型）、伪代码（灵活流程）还是精确脚本（高风险操作）
- **解释原因模式**：MUST、ALWAYS 这类词容易让 Agent 过度遵循，有反效果，该重构就重构

### 工作流控制
- **执行清单模式**：超过三步的任务，用清单模式让 Agent 按顺序执行
- **自纠正循环模式**：设置重试上限，避免 Agent 在错误方向上死循环

### 可执行代码
- **实用工具包模式**：提供可直接调用的代码片段
- **自主校准模式**：allowed-tools 是预批准，不是硬限制。Agent 觉得有必要，可以申请额外的工具

这 14 个模式不是都要用上，而是**根据任务类型选合适的工具**。就像工具箱，不需要每次都全带上，捡有用的拿。

![14 个设计模式分类图，按五大类展示各模式的位置关系](/images/agent-skills/20260716_194455_951744_01.png)

---

## 面试怎么答

**基础版**（能过的回答）：

一个可复用的 Agent Skill 包含 SKILL.md 和配套资源。SKILL.md 由 YAML Frontmatter 和 Markdown Body 组成。description 是 Agent 匹配 Skill 的唯一依据，要包含触发词、排除条件和领域关键词。内容采用渐进式披露：Level 1 加载 name + description，Level 2 加载完整指令，Level 3 按需加载资源。常用设计模式包括排除条款、执行清单、解释原因等，根据场景选用。

**加分版**（让面试官眼前一亮）：

description 和 examples 加起来最多 1536 字符，要注意精简。超过 300 行就该拆分 Skill，避免上下文爆炸。严格指令的代价是把失败方式从「做错」变成「做不了」，所以要根据任务自由度选择指令形式：开放型任务用文本指令，高风险操作用精确脚本，中间用伪代码。allowed-tools 本质是「预批准」而非「硬限制」，Agent 仍有自主决策空间。

---

## 一句话总结

Agent Skill 的核心是让 Agent **精准找到** 合适的 Skill，并**恰到好处地**获取指令——description 决定找不找得到，设计模式决定给得恰不恰当。

![Agent Skill 设计核心要点总结图](/images/agent-skills/20260716_194440_971584_01.png)