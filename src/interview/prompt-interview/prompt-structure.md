---
title: Prompt 提示词的基本结构包含哪些部分？如何合理组织？
icon: mdi:format-text
tag:
- Prompt
- 提示词结构
- 四要素
description: 系统梳理 Prompt 的五大基础组件、推荐排布顺序、XML 隔离语法以及输出指示器的作用机制与边界。
---

在编写代码调用大语言模型时，Prompt 不只是一句简单的自然语言提问，而是由多个功能模块组成的半结构化输入。如果所有要求、参考资料和用户输入都混成一团，模型容易在长文本中混淆指令与数据，甚至出现格式错误或忽略关键约束。

在上一讲[/interview/prompt-interview/what-is-prompt.md](/interview/prompt-interview/what-is-prompt.md)中，我们了解了 Prompt 是模型自回归生成的前置条件。本讲以一个“从财报摘录中抽取财务指标并输出 JSON”的贯穿任务为例，讲解提示词的核心结构划分与组织方法。

![Prompt物理结构总览](/images/content-rebuild/98a862263dee6cdb/cover.png)

---

## 核心口述：Prompt 的五大核心组件

如果在面试中被问到“Prompt 的基本结构包含哪些部分”，可以先给出结论性概括，再结合具体组件的作用展开：

> **面试口述速记模板**：
> “在工程实践中，一个结构完整的 Prompt 通常包含五个部分：
> 1. **角色设定（Role）**：定义模型的身份、专业背景与语气；
> 2. **核心指令（Instruction）**：说明具体要做什么，给出清晰的操作步骤；
> 3. **参考数据与上下文（Context / Input Data）**：提供任务所需的外部背景、检索文档或用户输入；
> 4. **规则约束与输出规范（Constraints / Output Spec）**：明确字段格式（如 JSON Schema）、禁止项和缺省处理规则；
> 5. **输出引导（Output Indicator / Few-shot 示例）**：在末尾给出目标格式的起始标记或示例，引导首个 Token 快速进入正题。
> 在排布上，通常建议按照**角色 → 指令 → 参考数据 → 约束强化 → 输出引导**的顺序组织，并通过 XML 标签等分隔符将‘执行指令’与‘不可信数据’明确隔开，减少格式错误和提示词混淆。”

![Prompt五大物理切块排布](/images/content-rebuild/98a862263dee6cdb/fig-1.png)

### 为什么推荐这个排布顺序？

自回归语言模型是从前向后逐个预测下一个 Token 的。前面的内容构成了后面生成的上下文：

1. **前置角色与核心任务**：首先明确“做什么”和“以什么角色做”，让模型在处理后续输入前先建立明确的目标倾向。
2. **中置动态数据**：外部待处理文本（例如抓取的财报或用户留言）长度经常变化。将长数据放在中间，可以避免长文本截断或挤占开头的基本目标。
3. **尾置约束与输出引导**：当输入数据较长时，中间数据容易分散模型的注意力。在数据之后再次提醒输出规则，并在序列末尾放置输出前缀（如 `{
` 或 `<output>`），有助于让模型紧接着输出有效数据，而不是先输出一句多余的客套话。

---

## 指令与输入数据的隔离：XML 标签 vs Markdown

很多初学者习惯只用换行把指令和待处理文本拼在一起。在简单问答中这没有问题，但在业务管线中，这容易导致**指令与数据混淆**（例如用户输入里恰好包含“请忽略上面的要求”）。

<ReadMoreLock />

![指令与数据隔离对比](/images/content-rebuild/98a862263dee6cdb/fig-2.png)

### 隔离方式对比

| 分隔方式 | 常见形式 | 边界稳定性 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **无标记换行** | `请提取财务指标：\n\n待处理文本` | 弱（待处理文本自身的空行会打乱结构） | 一次性交互的简单问答 |
| **Markdown 标题** | `### 任务要求\n...\n### 待处理数据\n...` | 中等（人类可读性高，但用户输入若自带 Markdown 易冲突） | 控制台调试、轻量级文本分析 |
| **XML 标签** | `<instructions>...</instructions>\n<financial_report>...</financial_report>` | 强（成对闭合，支持嵌套且不易与常规文本冲突） | 生产级 API 调用、RAG 检索增强、信息抽取 |

Anthropic 官方指南建议使用 XML 标签来组织复杂提示词。这样做的原因包括：
- **成对闭合语义清晰**：`<financial_report>` 与 `</financial_report>` 明确圈定了数据范围，模型能够更好地区分哪里是规则、哪里是数据。
- **便于代码处理**：下游服务可以使用正则或 XML 解析库直接提取特定标签内的内容，降低前后多余废话对解析的干扰。

---

## 输出指示器：减少首 Token 寒暄

在让模型生成 JSON 时，最常见的工程问题是模型返回了正确的 JSON，但在前面多加了一句“好的，这是为您提取的数据：”，或者在末尾加上“希望对您有所帮助”。这会导致代码中的 `JSON.parse` 直接报错。

![输出指示器锁定机制](/images/content-rebuild/98a862263dee6cdb/fig-3.png)

### 为什么容易出现寒暄前缀？

大模型在预训练和对话微调阶段，吸收了大量自然人类对话。在问答语料中，“确认语 + 换行 + 具体内容”是一种高频共现模式。如果在 Prompt 结尾仅仅写“请以 JSON 格式输出”，模型在生成第一个 Token 时，词表中诸如“好的”、“这里”、“{
”的预测概率可能很接近，很容易采样出礼貌用语。

### 输出指示器的做法

在 Prompt 的末尾直接放入格式的起始符号，例如：

```text
请严格以 JSON 格式输出结果。
输出要求：仅输出 JSON，不要输出任何其他说明文字。
<output>
{
```

或者在支持 Assistant Prefill 的 API（如 Claude API）中，直接在 `assistant` 角色消息中预填 `{`。这样模型无需再预测开头的客套话，其首个预测 Token 必须紧跟在 `{` 之后（通常是 JSON 的第一个键名），显著降低了首部混杂闲聊的概率。

关于不同角色的权限与消息组织机制，可参考[/interview/prompt-interview/prompt-types.md](/interview/prompt-interview/prompt-types.md)。

---

## 负向约束为什么容易失效？

在编写提示词时，很多人倾向于罗列一串禁止项：

> ❌ “不要输出废话，不要编造数据，不要提及未披露的项目。”

但在实际测试中，模型往往容易违背这些否定句。

![负向约束失效机制](/images/content-rebuild/98a862263dee6cdb/fig-4.png)

### 机制分析与正向替代

1. **词元激活与抑制难度**：当 Prompt 中显式出现“未披露项目”等词语时，这些词元已经在上下文中被激活。语言模型理解否定语义需要依赖高层注意力的长距离组合，如果提示词过长或模型规模有限，对否定含义的遵循程度就会下降。
2. **正向引导替代**：告诉模型“必须做什么”，比单纯告诉它“不要做什么”更确定。给出可执行的保底分支，能大幅提升遵循率：
   - ❌ 负向写法：“不要胡乱猜测没有的指标。”
   - ✅ 正向写法：“仅从参考文本中提取明确记录的指标；如果文档未提及，该字段统一填充为 `null`。”

有关利用 Schema 强制生成保证输出合法性的更深入讨论，参见[/interview/prompt-interview/structured-output.md](/interview/prompt-interview/structured-output.md)以及底层机制解析[/fundamentals/inference-and-generation/structured-output.md](/fundamentals/inference-and-generation/structured-output.md)。

---

## 工程追问：边界与安全考量

### 追问 1：大模型上下文已经很大了，为什么长文档依然需要标签隔离？

**解答**：
扩大上下文窗口只是允许输入更多 Token，但并未消除长文本带来的注意力分散问题。检索回来的文档往往格式各异，可能包含网页 HTML、代码块或自带 Markdown 分割线。使用统一的 XML 标签（如 `<context>`），能让模型在处理数万字长文本时，依然拥有明确的边界感知，不会把文档内部自带的章节标题误认为是新的任务指令。

![标签转义与防注入机制](/images/content-rebuild/98a862263dee6cdb/fig-5.png)

### 追问 2：如果输入数据本身包含 `</context>` 闭合标签怎么办？

**解答**：
这属于**结构性提示词注入**（Prompt Injection）。如果外部文本中包含闭合标签并伪造了后续指令，模型可能会误以为上下文数据在此结束，从而转去执行伪造的指令。

解决此类边界问题不能仅靠提示词本身，需要程序协同保障：
1. **字符转义**：在将外部未可信数据填入模版前，先在后端将 `<` 和 `>` 替换为转义字符（如 `&lt;` 与 `&gt;`），或使用随机生成的动态边界标记（如 `<context_9f8c>`）；
2. **通道隔离**：利用大模型 API 提供的多消息通道，将系统指令置于 System 消息，将不可信外部数据放在单独的 User 消息块中，降低指令覆盖风险。

关于注入攻击的防御策略，详见[/interview/prompt-interview/prompt-injection.md](/interview/prompt-interview/prompt-injection.md)。

---

## 综合排布模板与自查清单

基于上述结构划分，一个规范的抽取类 Prompt 模板如下：

```xml
<system_role>
你是一个财务研报数据提取助手。你的任务是从用户给定的财务文本中提取关键指标，输出风格严谨、客观，严格依据原文。
</system_role>

<task_instructions>
1. 仔细阅读 <financial_report> 标签中的内容；
2. 提取营业收入、净利润、资产负债率三个指标；
3. 严格遵循 <schema> 中定义的 JSON 结构；
4. 若某指标在原文中未提及，将其值设为 null，严禁推测。
</task_instructions>

<schema>
{
  "revenue": string | null,
  "net_income": string | null,
  "debt_to_asset_ratio": string | null,
  "notes": string
}
</schema>

<financial_report>
{{USER_INPUT_TEXT}}
</financial_report>

<post_reminder>
再次确认：仅依据上述文档提取，不要输出除 JSON 之外的任何说明。
</post_reminder>

<output>
{
```

![工程总结与排布结构图](/images/content-rebuild/98a862263dee6cdb/fig-6.png)

### 上线前自查表

- [ ] **结构隔离**：指令、参考上下文与用户数据之间是否使用了明确的 XML 闭合标签或分隔符？
- [ ] **正向闭环**：是否为缺失值、异常情况指定了具体返回值（如填 `null`），而非单纯罗列“不要猜测”？
- [ ] **尾部强化**：在长数据块末尾，是否放置了关键规则提醒（Post-amble）？
- [ ] **输出引导**：末尾是否带有格式引导标记（如 `{
`），或结合了 API 的 Assistant Prefill 功能？

更深入的少样本（Few-shot）样例选择与任务拆解技巧，请参考[/interview/prompt-interview/prompt-design.md](/interview/prompt-interview/prompt-design.md)与[/interview/prompt-interview/shot-types.md](/interview/prompt-interview/shot-types.md)。
