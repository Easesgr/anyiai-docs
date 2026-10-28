---
title: 如何让大模型稳定输出 JSON？
icon: mdi:message-text-outline
date: 2026-07-16
tag:
  - 大模型100讲
  - Prompt
---

# 如何让大模型稳定输出 JSON？

> 一句话核心：让大语言模型（LLM）稳定输出合法JSON格式，是AI应用落地的关键技术，涉及提示工程、解码约束、后处理修复等多层策略，面试中常考察实际工程能力。
## 核心概念（术语表）

- **JSON mode**：OpenAI/Claude等模型提供的原生JSON输出支持，通过API参数指定"response_format":{"type":"json_object"}
- **Prompt Engineering（提示工程）**：通过设计精准的Prompt引导模型输出特定JSON结构，是最直接且低成本的方案
- **Few-shot Learning（少样本学习）**：在Prompt中提供完整示例，让模型模仿输出格式
- **JSON Schema**：JSON的结构定义规范，支持枚举、类型约束、必填字段等复杂规则
- **Temperature**：控制模型输出随机性的参数，值越低输出越稳定
- **BNF（巴科斯-诺尔范式）**：用于解析和修复无效JSON的语法表示方法
- **Structured Output（结构化输出）**：AI原生开发中的关键技术，让模型输出程序可直接解析的格式
<ReadMoreLock />

## 历史背景 / 来源

- 2023-2024年：大模型应用开发兴起，结构化输出成为刚需
- OpenAI在GPT-4发布后引入JSON mode（2023年）
- Anthropic的Claude几乎同期支持JSON/XML结构化输出
- 解决的问题：模型"说废话"、输出格式不稳定、程序无法解析

## 工作原理 / 核心机制

### 整体思路
通过"提示约束 + 模型原生支持 + 后处理修复"三层机制，确保模型输出的JSON合法且结构正确。

### 输入/输出
- **输入**：用户query + system prompt + 可选的few-shot示例 + 可选的JSON schema定义
- **输出**：符合指定格式的JSON对象/数组

### 核心步骤详解

**第一步：提示约束（Prompt约束）**
- 在system prompt中明确"Do NOT include anything other than a JSON object"
- 提供完整示例："Your output should look like this: {...}"
- 指定输出格式为JSON，禁止markdown代码块

**第二步：启用JSON mode（模型原生支持）**
- OpenAI API：设置 response_format={"type":"json_object"}
- Claude API：设置 output={"format":{"type":"json_object"}}
- 注意：OpenAI要求prompt中必须包含"JSON"字样才能激活

**第三步：JSON Schema定义（复杂结构）**
- 定义type、properties、required、enum等约束
- 不需要提供示例，通过schema自动引导输出
- 适合强类型语言的后续处理

**第四步：后处理修复（容错机制）**
- 使用json_repair等库修复不合法JSON
- 原理：BNF语法解析 + 启发式规则（补全括号、添加引号等）
- 适用于模型"抽风"的兜底处理
## 关键知识点

1. 最简单有效的方式：在system prompt中以示例要求输出格式
2. OpenAI的JSON mode只支持JSON object，不支持纯数组，需套一层items
3. Claude支持JSON和XML两种结构化格式
4. Few-shot比单纯示例更稳定，需要在user/assistant消息中都提供示例
5. JSON Schema支持枚举（enum）约束，如"genre":{"enum":["SCI-FI","NON-SCI-FI"]}
6. 降低Temperature（如设为0.3-0.5）可显著提升输出稳定性
7. 代码层面必须做好解析失败的容错和重试机制
8. 使用JSON Schema不需要给例子，避免示例带偏模型
9. 复杂结构（多种类型混合、嵌套层级深）时，Schema比示例更可靠
10. JSON修复工具基于BNF语法，修复策略包括补全括号、引号、空格
11. TypeScript结构体与JSON Schema实现类似效果
12. 实际工程中建议"提示工程优先 → 不稳定再加JSON mode → 复杂结构用Schema"

## 应用场景

- **场景1：API响应生成**：某电商系统用LLM生成商品详情页，后端解析JSON直接渲染卡片，响应时间`<500ms`
- **场景2：测试用例生成**：角色定位为"测试用例生成API"，禁止输出解释性文字，直接返回纯JSON
- **场景3：数据导出**：用户输入分析需求，LLM输出固定格式JSON，前端展示表格/图表
- **场景4：配置文件生成**：AI原生应用开发中，强制返回指定格式的配置JSON
- **场景5：推书推荐系统**：输入主题，输出包含name/author/reason/year_of_publish的JSON数组

## 常见误区 / 踩坑

- ❌ **误区1**：以为启用JSON mode就100%合法
  ✅ **正解**：模型仍可能"抽风"，必须配合后处理修复和重试机制
- ❌ **误区2**：OpenAI的JSON mode不需要prompt中提"JSON"
  ✅ **正解**：必须包含"JSON"字样，否则会生成失败
- ❌ **误区3**：JSON mode可以输出纯数组
  ✅ **正解**：OpenAI的JSON mode只支持object，需套一层items
- ❌ **误区4**：示例越详细越好
  ✅ **正解**：示例不恰当会带偏模型，复杂结构用Schema替代
- ❌ **误区5**：Temperature越低越好
  ✅ **正解**：过低会导致输出缺乏多样性，建议0.3-0.7区间调试
- ❌ **误区6**：不需要做容错
  ✅ **正解**：必须处理解析失败场景，可重试或降级处理

## 性能 / 复杂度

- **Token消耗**：示例 < few-shot < JSON Schema（复杂度递增）
- **稳定性**：示例约70% < JSON mode约85% < JSON Schema + JSON mode约95%
- **解析成功率**：配合json_repair可提升至99%以上
- **与替代方案对比**：
  - 正则匹配：速度快但无法处理嵌套结构，适用简单字段提取
  - JSON Schema：稳定性最佳但token消耗高，适用复杂业务场景
  - 纯Prompt：成本低但不稳定，适用简单场景快速验证

## 与相关概念的区别

- **vs 正则表达式提取**：
  - 正则：速度快O(n)，但无法处理嵌套和复杂类型
  - JSON输出：稳定性高，支持嵌套结构，但需要模型配合
  - 怎么选：简单字段提取用正则，结构化数据用JSON输出

- **vs XML格式输出**：
  - XML：Claude原生支持，可读性好但token消耗高
  - JSON：更简洁，生态工具丰富，适合程序处理
  - 怎么选：Web API用JSON，文档类用XML

- **vs 纯文本 + 后处理**：
  - 纯文本：模型更容易"说废话"，解析成功率低
  - JSON输出：结构强制约束，解析成功率高
  - 怎么选：程序调用场景必须JSON输出

## 进阶 / 面试加分项

- **最新进展**：OpenAI的Structured Outputs API（2024年）可100%保证输出符合schema
- **业界争议**：JSON mode vs JSON Schema哪个更好？实际上取决于场景复杂度
- **金句**："让模型稳定输出JSON不是单点技术，而是提示工程 + 模型能力 + 后处理的三层配合"


## 面试如何回答


### 🟢 如何让大模型稳定输出合法的JSON格式？

**回答要点**：

让大模型稳定输出JSON需要"三层配合"：第一层是Prompt约束，在system中明确"只输出JSON，禁止其他内容"；第二层是启用JSON mode，如OpenAI设置response_format={"type":"json_object"}，Claude设置output format；第三层是后处理修复，使用json_repair库修复不合法JSON。实际推荐顺序是：先尝试Prompt+示例（约70%稳定），不稳定再加JSON mode（提升到85%），复杂结构用JSON Schema（可达95%+），最后加后处理兜底（99%）。面试时要强调不能只依赖单一方案。


### 🟡 OpenAI的JSON mode有什么限制？为什么有时候启用了还是输出乱码？

**回答要点**：

OpenAI JSON mode有两个关键限制：第一，必须在prompt中包含"JSON"字样才能激活，否则会生成失败；第二，只支持JSON object，不支持纯数组，如果需要数组必须套一层items包装。启用后仍可能输出乱码的原因有两个：一是模型"抽风"是概率事件，不是100%可靠；二是prompt中没有提供足够的格式约束。解决方案是配合few-shot示例和JSON Schema定义，并做好后处理修复。面试时要用"概率"而非"绝对"来描述JSON mode的可靠性。


### 🟡 JSON Schema相比Few-shot示例有什么优势？什么时候该用Schema？

**回答要点**：

JSON Schema的优势在于：第一，不依赖示例，避免示例不恰当带偏模型；第二，支持强类型约束（enum、type、required），程序解析更安全；第三，适合复杂结构（多层嵌套、混合类型），而Few-shot在结构复杂时容易失败。建议在以下场景用Schema：需要枚举限制（如只能返回"SCI-FI"或"NON-SCI-FI"）、字段类型必须严格（如year_of_publish必须是number）、结构超过3层嵌套。Token消耗会比示例高，但稳定性显著提升。简单场景用Few-shot即可。


### 🟡 在工程实践中，如何处理模型输出不稳定的问题？

**回答要点**：

工程实践中有三个关键处理：第一，Prompt层面，从简单示例开始逐步升级到JSON Schema，不稳定再叠加；第二，参数层面，适当降低Temperature（建议0.3-0.5），既保证稳定性又有一定多样性；第三，代码层面，必须做容错处理——先用json.loads()解析，失败后调用json_repair修复，若仍失败则重试（建议3次），超过重试次数后降级返回默认结构或记录错误日志。面试时要体现"防御性编程"思维，不能假设模型100%输出正确。


### 🔴 Few-shot和JSON Schema可以一起用吗？有什么注意事项？

**回答要点**：

Few-shot和JSON Schema可以叠加使用，但通常建议二选一。原因是：如果同时提供示例和Schema，模型可能过度依赖示例而忽略Schema约束。正确用法是：简单字段结构用Few-shot（成本低），复杂结构用JSON Schema（稳定性高），极少数需要枚举+示例的混合场景才叠加。注意事项：JSON Schema中的required字段是强制约束，而示例可能不完整，所以Schema的优先级高于示例。面试时要理解"约束来源"的概念——Schema是机器可读的强约束，示例是自然语言的软引导。


### 🟡 除了JSON，还有哪些结构化输出格式？各有什么优劣？

**回答要点**：

主流结构化输出格式有三种：第一是JSON，最简洁、工具生态成熟、程序解析方便，但需要模型配合才能稳定；第二是XML，Claude原生支持，可读性好（适合调试），但token消耗比JSON高30%左右；第三是CSV/KV格式，适合简单表结构，但嵌套支持差。实际选型建议：Web API场景用JSON（生态最好），Claude应用可用XML（原生支持），内部工具用KV（解析最快）。面试时要体现对"格式即契约"的理解，不同场景需要不同格式。


### 🟡 Temperature对JSON输出有什么影响？如何调参？

**回答要点**：

Temperature控制输出随机性，值越低输出越稳定、越可预测。对于JSON输出场景，建议设置在0.3-0.5之间：过低（如0.1）虽然稳定但输出缺乏多样性，相同query可能返回完全相同的结果；过高（如1.0）会导致输出格式随机，容易出现非法JSON。实际调参建议：先用默认值0.7测试结构稳定性，如果频繁出现格式错误，逐步降到0.5甚至0.3；如果是确定性场景（如配置生成），可以降到0.1-0.2。面试时可以提到这是"稳定性与多样性的trade-off"。


### 🔴 如果模型输出的JSON完全无法解析，修复策略是什么？

**回答要点**：

当JSON完全非法时，修复策略基于BNF语法解析和启发式规则：第一步，检测语法错误类型（缺少引号、括号不匹配、逗号位置错误等）；第二步，针对性修复——未闭合的对象/数组自动补全、字符串缺少引号则添加、尾部逗号删除、非法字符过滤；第三步，重新解析验证。常用工具有json_repair（Python）和json-repair（JS）。注意：修复不是万能的，如果模型输出与预期结构完全偏离（如返回了一句话而非JSON），修复无效，需要调整Prompt或重试。面试时要强调"修复是兜底，不是主力"。

