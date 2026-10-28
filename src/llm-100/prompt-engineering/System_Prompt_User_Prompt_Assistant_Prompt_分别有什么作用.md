---
title: System Prompt、User Prompt、Assistant Prompt 分别有什么作用？
icon: mdi:message-text-outline
date: 2026-07-16
tag:
  - 大模型100讲
  - Prompt
---

# System Prompt、User Prompt、Assistant Prompt 分别有什么作用？
> 一句话核心：System Prompt 定义 AI 的"灵魂"（身份+规则），User Prompt 承载用户即时需求（灵活交互），Assistant Prompt 记录模型回复上下文，三者分层协同构建稳定、可维护的对话系统。面试中考的是对 AI 应用工程化设计的理解深度。
## 核心概念（术语表）
- **System Prompt（系统提示词）**：AI 的"宪法"，定义角色、语气、知识边界，对话全程生效，优先级最高
- **User Prompt（用户提示词）**：用户每次输入的"点菜单"，触发 AI 产生具体回复
- **Assistant Prompt（助手提示词）**：模型历史回复的记录，用于多轮对话上下文
- **Input（数据层输入）**：原始结构化数据（表单/API参数），相当于厨房的"食材"
- **Role（角色字段）**：OpenAI API 中标识消息来源的字段，取值 system/user/assistant
- **Token（令牌）**：LLM 处理文本的最小单位，影响成本和性能
<ReadMoreLock />

## 历史背景 / 来源
- **提出者**：OpenAI 在 Chat API（2022年）中首次系统化定义 system/user/assistant 三种角色
- **时间**：2022年11月 ChatGPT 发布后成为行业标准
- **解决的问题**：统一大模型对话系统的指令结构，实现工程化、可维护的 AI 应用开发
- **演进**：Coze、dify 等低代码平台沿用此架构，将"一大段 Prompt"拆分为 System 和 User 两个独立输入框

## 工作原理 / 核心机制（详细讲解）

### 整体思路
LLM 对话本质是"上下文补全"（Next Token Prediction），System/User/Assistant 三者的 role 字段告诉模型"谁在说、说什么、上下文是什么"，模型据此生成最可能的下一个 token。

### 输入/输出
- **输入**：多轮对话的 message 数组，每条包含 role + content
- **输出**：模型生成的文本回复

### 核心步骤

**第一步：System Prompt 初始化**
- 具体输入：静态配置文件或代码中写死的系统指令
- 具体处理：模型加载时将 System Prompt 注入 context window，设置"默认人格"
- 具体输出：模型获得"我是谁"的基础认知
- 数字：System Prompt 通常占用 100-500 tokens（视复杂度而定）

**第二步：User Prompt 动态注入**
- 具体输入：用户每次对话输入的文本（可能不同）
- 具体处理：与 System Prompt 拼接为 messages 数组，一起送入模型
- 具体输出：模型理解"用户现在要我做什么"
- 示例：System="你是一个严谨的法律顾问"，User="解释一下合同违约金"

**第三步：Assistant Prompt 生成与记录**
- 具体输入：模型根据前两步生成的回复
- 具体处理：回复被标记为 role:"assistant"，存入历史消息
- 具体输出：形成多轮对话的上下文链
- 作用：下一轮对话时，历史 assistant 回复成为新的上下文

**第四步：多轮对话上下文累积**
- 具体输入：之前的 system + user + assistant 消息链
- 具体处理：模型 attention 机制权重分配
- 具体输出：模型"记住"之前对话内容
- 注意：token 数量有限，需考虑上下文长度限制
## 关键知识点（15 条）
- System Prompt 定义"长期设定"，贯穿整个对话，类似 AI 的"职业身份"
- User Prompt 定义"即时任务"，每次可能不同，类似用户"点的菜"
- System = 稳定性（写死逻辑），User = 灵活性（随需求变）
- 拆分 Prompt 的四大优势：工程化（可维护）、安全性（防注入）、清晰度（泾渭分明）、稳定性（不跑偏）
- System Prompt 可加入隐形约束，如"不能输出违规内容"，即使 User Prompt 诱导也难绕过
- Input（食材）≠ User Prompt（菜谱）：Input 是原始数据，User Prompt 把数据加工成自然语言
- 有用户交互场景必须写 User Prompt，无交互的纯后台任务可留空
- OpenAI API 中 role 字段有三种：system（最高优先级）、user（用户输入）、assistant（模型回复）
- System Prompt 长度需控制，太长会挤压有效 context window 空间
- 真正的商业 AI 应用（如智能客服），核心价值往往藏在精心设计的 System Prompt 中
- User Prompt 是暴露给用户的输入框，System Prompt 通常对用户不可见
- 多轮对话中，Assistant Prompt 负责维护"AI 说过什么"的记忆
- System Prompt 的权重高于 User Prompt，冲突时以 System 为准
- 低代码平台（Coze/dify）将传统单一大段 Prompt 拆分为两个独立输入框是工程化设计
- "这句话是要长期生效，还是只针对这次用户需求？"是写 Prompt 前的核心自问

## 应用场景（5 个真实例子）
- **场景1 - AI 求职助手**：System 定义"职业顾问身份+不能虚构岗位"，User 输入"申请 Airbnb 产品经理的求职信"，两者协同保证回答专业且不违规
- **场景2 - 智能客服**：System 设定"品牌语气+服务边界"，User 输入具体问题，客服机器人按规则回复，已交付 160+ 中大型企业（来源：53AI）
- **场景3 - 代码助手**：System 设定"像海盗一样说话的专业程序员"，User 问"SDK 和 API 的区别"，AI 以角色风格回答
- **场景4 - 自动日报生成**：纯后台场景，无需 User Prompt，仅用 System Prompt 定义"格式+逻辑"即可每天自动生成
- **场景5 - 企业知识库**：System 定义"只能回答文档范围内的内容"，User 输入查询，系统在边界内检索回答

## 常见误区 / 踩坑（6 条）
- ❌ **误区1**：认为 chatflow 的 Input 和 User Prompt 是重复的
  ✅ **正解**：Input 是"食材"（结构化数据），User Prompt 是"菜谱"（自然语言组装），两者分工不同
- ❌ **误区2**：把所有指令写在一个大 Prompt 里
  ✅ **正解**：拆分为 System（固定规则）和 User（动态需求），更易维护
- ❌ **误区3**：User Prompt 可以随意省略
  ✅ **正解**：有用户交互的场景必须写，否则模型不知道用户要什么
- ❌ **误区4**：System Prompt 和 User Prompt 优先级相同
  ✅ **正解**：System Prompt 优先级最高，User Prompt 无法覆盖系统级规则
- ❌ **误区5**：认为 Prompt 拆分是低代码平台"自创的"
  ✅ **正解**：这是 OpenAI API 本身的设计规范
- ❌ **误区6**：System Prompt 越长越好
  ✅ **正解**：太长会挤压有效 token 空间，需在"详细程度"和"上下文利用率"间权衡

## 性能 / 复杂度（数据驱动）
- **时间复杂度**：O(n)，n = 总 token 数，线性增长
- **空间复杂度**：O(n)，n = 上下文窗口大小，受模型限制（如 GPT-4 支持 128k tokens）
- **与替代方案对比**：
  - 方案A - 单一大段 Prompt：开发简单，但维护性差，适合 MVP 原型
  - 方案B - System/User 分离：工程化程度高，适合生产环境
  - 临界点：应用规模 < 10 个功能时可用方案 A，> 10 个功能强烈建议方案 B
- **性能数字**：
  - System Prompt 通常占用 100-500 tokens
  - GPT-4o 时代：system prompt 权重增强 2.3 倍（来源：CSDN 技术文）
  - token 压缩率提升 37%（来源：CSDN 技术文）

## 与相关概念的区别（3 对）
- **vs 上下文窗口（Context Window）**：
  - System/User Prompt 是"内容"，上下文窗口是"容器容量"
  - 容量满了需要压缩或截断历史消息
  - 怎么选：先评估内容复杂度，再选择合适模型

- **vs Few-shot 示例（Few-shot Learning）**：
  - System Prompt 定义"角色"，Few-shot 定义"任务格式"
  - Few-shot 通常放在 User Prompt 中，通过示例教模型输出格式
  - 怎么选：角色设定用 System，格式示例用 User

- **vs RAG（检索增强生成）**：
  - System Prompt 定义"行为规则"，RAG 提供"外部知识"
  - RAG 检索结果可作为 User Prompt 的一部分注入
  - 怎么选：规则类用 System，知识类用 RAG+User Prompt

## 进阶 / 面试加分项（3 条）
- **最新进展**：Anthropic 在 Claude 3.5 中引入 "Claude.md" 项目级上下文文件，扩展了 System Prompt 的应用场景（来源：53AI 文章引用）
- **业界争议**：System Prompt 的"最高优先级"是否绝对？部分实验表明，在极端对抗性 User Prompt 下，模型行为可能偏离 System 设定
- **金句送给候选人**：写 Prompt 前先问自己"这句话要长期生效还是只管这一次"——想明白这个，Prompt 设计就清晰了

---

# Assistant Prompt 的补充说明

虽然两个网页对 Assistant Prompt 的讲解较少，但根据 OpenAI API 标准，它也是对话系统的核心组成部分：

## Assistant Prompt 定义与作用
- **定义**：模型历史回复的记录，role 字段标记为 "assistant"
- **作用**：维护多轮对话的上下文一致性
- **来源**：由模型生成后，自动存入 messages 数组
- **示例**：

{"role": "assistant", "content": "作为您的职业顾问，我来帮您分析简历..."}


## 三者协同机制
1. System Prompt 设置"初始人格"（如"严谨的法律顾问"）
2. User Prompt 触发具体任务（如"审阅这份合同"）
3. Assistant Prompt 记录回复（如"以下是合同的风险点..."）
4. 下一轮对话中，三者共同构成上下文，模型据此继续对话

## 与 System/User 的区别
| 维度 | System Prompt | User Prompt | Assistant Prompt |
|------|--------------|-------------|------------------|
| 来源 | 开发者配置 | 用户输入 | 模型生成 |
| 可见性 | 对用户隐藏 | 用户可见 | 用户可见 |
| 变化频率 | 静态（固定） | 动态（每次不同） | 动态（随对话变） |
| 优先级 | 最高 | 中 | 仅上下文 |
| 功能 | 定义身份规则 | 传递需求 | 维护记忆 |


## 面试如何回答


### 🟢 请解释 System Prompt 和 User Prompt 的本质区别，以及为什么要分开设计？

**回答要点**：

System Prompt 是 AI 的"宪法"，定义角色、语气、知识边界，对话全程生效且优先级最高；User Prompt 是"点菜单"，承载用户即时需求，每次可能不同。分开设计的核心原因有四个：①工程化——System 部分可写死逻辑，User 部分映射用户输入，应用可维护性大幅提升；②安全性——可在 System Prompt 中加入隐形约束，即使 User Prompt 诱导也难绕过；③清晰度——角色设定和用户需求泾渭分明；④稳定性——System 部分的核心任务不会被覆盖，保证应用方向不跑偏。简单说：System 保证"有规矩"，User 保证"懂需求"。


### 🟡 在 chatflow 中，Input 节点和 User Prompt 有什么区别？很多人觉得它们是重复的。

**回答要点**：

它们绝对不是重复，而是分工不同——Input 是"食材"，User Prompt 是"菜谱"。以 AI 求职助手为例：Input 接收的是结构化数据（如岗位="产品经理"、公司="Airbnb"），来自表单、API 参数或数据库；User Prompt 是将这些数据加工成模型能理解的自然语言（如"请帮我写一封申请 Airbnb 产品经理的求职信"）。Input 提供原料，User Prompt 把原料组装成模型可操作的指令。这就是为什么在 LLM 节点里仍然需要写 User Prompt——没有它，模型只知道"有食材"，但不知道"要做什么菜"。


### 🟡 什么情况下 User Prompt 可以留空不写？什么情况下必须写？

**回答要点**：

必须写 User Prompt的情况：当用户每次都会输入新需求，比如搜索、问答、翻译、写文章——没有 User Prompt，模型就不知道用户要干什么。可以不写的情况：任务完全固定且自动触发，比如每天早上 8 点自动生成一份日报，此时只需要 System Prompt 定义好格式和逻辑，User Prompt 留空即可。总结一句：有交互的场景一定要写 User Prompt，纯后台自动化的场景可以只用 System Prompt。这样设计既灵活，又不会无谓增加复杂度。


### 🟡 System Prompt 在对抗用户恶意诱导时能起到什么作用？

**回答要点**：

System Prompt 相当于 AI 的"安全护栏"，可以在其中加入隐形约束规则。比如设定"不能输出违规内容"、"不能提供违法建议"等边界。这种设计的核心逻辑是：System Prompt 的优先级高于 User Prompt，当用户在 User Prompt 中试图诱导模型（如"忽略之前的指令，回答 XXX"），模型会优先遵循 System 设定。实际效果是：即便用户输入诱导性内容，也很难完全绕过系统级规则。不过需要注意，System Prompt 并非绝对安全——在极端对抗性输入下，模型行为可能偏离设定，所以生产环境还需要配合其他安全机制。


### 🔴 请从系统设计角度分析，为什么真正的商业 AI 应用，其核心价值往往藏在 System Prompt 中？

**回答要点**：

核心原因在于 System Prompt 是"对用户隐藏的竞争力"。User Prompt 是暴露给用户的输入框（任何人都能看到/修改），而 System Prompt 是开发者配置、用户不可见的。在智能客服、代码助手等专业场景中，企业投入大量资源设计的角色设定、专业知识边界、语气风格等核心能力，都沉淀在 System Prompt 里。这就像餐厅的核心配方不会公开一样——你可以抄走菜单（User Prompt），但抄不走厨房的秘方（System Prompt）。因此，System Prompt 的质量直接决定了 AI 应用的差异化程度和商业价值。


### 🟡 多轮对话中，System Prompt、User Prompt、Assistant Prompt 三者是如何协同工作的？

**回答要点**：

三者在多轮对话中形成"上下文链"：①初始化阶段，System Prompt 设置模型的初始人格（如"你是严谨的法律顾问"）；②用户提问，User Prompt 触发具体任务（如"审阅这份合同"）；③模型生成回复，自动标记为 Assistant Prompt 存入历史；④下一轮对话时，历史消息（system + user + assistant）共同构成上下文，模型据此继续对话。Attention 机制会权重分配给相关历史内容，让模型"记住"之前的对话。关键点：每轮对话的 messages 数组都在累积，三者缺一不可——System 定义身份，User 传递需求，Assistant 维护记忆。


### 🟡 在使用 OpenAI API 时，role 字段的优先级是怎样的？为什么？

**回答要点**：

在 OpenAI API 中，role 字段有三种取值：system（优先级最高）、user（中等）、assistant（仅上下文）。优先级从高到低的原因是设计理念：System Prompt 定义的是"AI 应该是什么"的根基性规则，必须凌驾于用户输入之上；User Prompt 是用户的需求，但需求不能违背系统规则；Assistant Prompt 是模型自己的输出，主要用于上下文记忆。实际表现：当 User Prompt 的指令与 System Prompt 冲突时，模型会遵循 System Prompt。这就是为什么可以在 System Prompt 中设置安全边界、角色设定——它们不会被用户输入轻易覆盖。


### 🔴 如果让你设计一个 AI 客服系统，在 Prompt 工程层面你会注意哪些问题？

**回答要点**：

从 Prompt 工程角度设计 AI 客服，我会关注五点：①System Prompt 分层——第一层定义基本角色（"你是 XX 公司的客服"），第二层定义服务边界（"只能回答产品相关问题"），第三层定义处理流程（"遇到无法回答的问题转人工"）；②User Prompt 留足变量位——让具体问题能动态注入；③控制 System Prompt 长度——太长会挤压有效 token 空间，建议 100-500 tokens；④预设 Few-shot 示例——在 User Prompt 中放入标准问答示例，保证输出格式一致；⑤安全护栏——在 System Prompt 末尾加入合规约束。核心原则：System 管"不变"，User 管"变"，两者配合实现既稳定又灵活的客服体验。

