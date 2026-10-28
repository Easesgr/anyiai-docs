---
title: Tool Calling 和 Function Calling 有什么区别？
icon: mdi:wrench-outline
tag:
  - Agent面试
  - Tool Calling
  - Function Calling
---

# Tool Calling 和 Function Calling 有什么区别？Agent 为什么需要工具调用？

![](/images/agent-interview/agentic-rag_cover.png)

这是一道深度理解题，考的是你对 AI Agent 工具调用体系的完整认知。很多人会把 Tool Calling 和 Function Calling 混为一谈，但实际上它们解决的是不同层次的问题。搞懂这两个概念，你不仅能答好这道题，对整个 Agent 架构的理解都会提升一个档次。

---
## 1. 先厘清概念：Tool Calling 和 Function Calling 到底是什么？

先说重点：**Function Calling 是大模型的内置能力，Tool Calling 是外部工具的实现机制**。业界经常混用这两个词，但严格来说，它们的含义是不同的。

**Function Calling** 解决的是"大模型怎么可靠地调用工具"这个问题。传统方式下，大模型生成的是自然语言，比如它会输出"我去查一下北京现在的天气"。这种输出不稳定，有时候模型会说"我帮你查到了"，实际上根本没查。Function Calling 就是为了解决这个问题——它让模型生成结构化的 JSON，而不是自由文本。

这个 JSON 长这样：

```json
{
  "name": "get_weather",
  "arguments": {
    "location": "北京",
    "unit": "celsius"
  }
}
```

清晰、可执行、不会产生幻觉。这就是 Function Calling 的核心价值。

**Tool Calling** 则是指具体的工具实现。比如你在代码里写了一个 `get_weather` 函数，这就是一个 Tool。Tool Calling 就是"调用这个 Tool 的过程"。

用人话来类比：**Function Calling 像是一份标准的快递单格式，Tool Calling 像是具体的快递公司和配送员**。快递单格式是统一的，但执行配送的是各个具体的快递公司。

![Function Calling工作流程图，展示"定义工具→转JSON描述→提示词告知LLM→生成调用指令→外部执行"的完整链路](/images/agent-interview/20260608_214037_959616_01.png)

---
<ReadMoreLock />

## 2. MCP 协议：打破厂商壁垒的通用标准

搞清楚了基础概念，我们来看一个更重要的东西——**MCP（Model Context Protocol）**。

为什么这个协议很重要？因为现在各家大模型的 Function Calling 实现是**私有**的：

- **OpenAI** 用 `tools` 参数
- **Claude** 用 `tools` 字段
- **Google** 可能又是另一套

这就像手机厂商各自搞自己的充电线，苹果用 Lightning，安卓用 Type-C，互相不兼容。你写了一套工具，想让不同模型都能用，就得为每个模型写一套适配代码。

**MCP 就是这个领域的"USB-C"协议**。它是一个开放标准，定义了：

- 工具怎么描述
- 请求怎么传递
- 响应怎么返回

有了 MCP，任何兼容 MCP 的模型都能调用任何 MCP 工具。就像蓝牙协议让任何手机都能连接任何蓝牙耳机一样。

用一个生活场景帮你理解：**Function Calling 像手机自带的语音助手，只能控制本机预装的应用**；**MCP 像万能遥控器，能控制家里所有品牌的电器**。

![Function Calling与MCP多维度对比表，展示兼容性、热插拔、权限控制、跨设备调用、开发复杂度、标准化程度六个维度的差异](/images/agent-interview/20260608_214034_967360_01.png)

![MCP架构简图，展示Host（用户交互应用）、Client（1:1连接MCP Server）、Server（暴露Tools/Resources/Prompts）的三层架构](/images/agent-interview/20260608_214031_295552_01.png)

---

## 3. 为什么 Agent 非得调用工具？三大硬核原因

好，现在说这道题的第二部分：Agent 为什么需要工具调用？

你可能会想，大模型不是已经很强大了吗？为什么还要调用外部工具？答案很简单：**大模型有三个天然的局限性**，不借助工具它就只是个"会说话的图书馆"。

### 原因一：知识有截止日期

大模型的训练数据有截止日期。ChatGPT-4 的知识可能停在 2023 年，Claude 的知识可能有 2024 年的。但世界是实时变化的——股票价格、天气状况、今天发生的新闻，这些模型都不知道。

工具调用让 Agent 能够实时获取最新信息。就像考试时遇到超纲题，你得翻书查资料，而不是凭印象瞎猜。

### 原因二：推理可能出错

大模型会产生幻觉——一本正经地胡说八道。它可能自信地告诉你"2024 年奥运会在巴黎举办"，但你让它算个简单的数学题，它可能都会出错。

通过调用外部工具，Agent 可以获得精确的计算结果。查数据库、做数学计算、验证事实——这些交给专业的工具来完成，模型只负责推理和决策。

### 原因三：能力有边界

大模型训练时没学过的东西，它就真的不会。查天气？调 API？操作文件？这些都需要工具来扩展 Agent 的能力边界。

就像一个学霸，理论知识很扎实，但你让他去银行办个业务，他得查攻略；让他写代码，他得调用 IDE。工具就是 Agent 的"手脚"。

![Agent工具调用三大原因示意图，用图示展示"知识滞后"、"逻辑不足"、"能力受限"三个痛点及解决方案](/images/agent-interview/20260608_214053_786752_01.png)

---

## 4. 未来趋势：Function Calling 与 MCP 将走向融合

很多同学以为 Function Calling 和 MCP 是竞争关系，要选一个。实际上，**它们解决的是不同层次的问题，未来会走向协同**。

打个比方：**Function Calling 负责"意图解析"，MCP 负责"协议传输"**。

就像你去餐厅吃饭：服务员理解你想吃什么（意图解析），然后把订单传给厨房（协议传输），最后厨房做好菜端给你（工具执行）。这三个环节缺一不可。

具体来说：

- **Function Calling** 让模型理解"用户想查天气"，生成结构化的调用指令
- **MCP** 把这个指令标准化地传递给任何兼容的工具提供者
- **工具执行层** 真正完成查询天气的任务

OpenAI 等厂商已经开始支持通过 adapter 把 Function Call 转为 MCP 请求。这意味着什么？意味着**协议标准化会加速生态繁荣**——就像 HTTP 和 REST API 的关系，标准化之后，整个互联网的 API 调用都变得简单了。

![Function Calling与MCP协同模式图，展示分层架构"意图解析-协议传输-工具执行"](/images/agent-interview/20260608_214051_502848_01.png)

---

## 面试怎么答

**基础版**：

Function Calling 是大模型调用外部工具的内置能力，它让模型生成结构化的 JSON 调用指令，而不是不稳定的自然语言。Tool Calling 和 Function Calling 在实际中常被混用，但严格来说 Function Calling 是"怎么调用"的方法论，Tool Calling 是具体的工具实现。Agent 需要工具调用主要有两个原因：一是获取实时信息，弥补训练数据的时效性缺陷；二是扩展能力边界，完成模型本身不会的任务。MCP 是一个跨模型的开放协议，让不同厂商的模型都能调用统一的工具。

**加分版**：

从原理层面看，Function Calling 生成 JSON 而不是自然语言，核心目的是保证稳定性和可执行性。不同厂商的实现方式不同——OpenAI 用 tool 参数，Claude 用 tools 字段，各有各的私有标准。MCP 解决的就是这个互通性问题，它像蓝牙协议一样，让任何兼容 MCP 的模型都能调用任何 MCP 工具。Agent 调用工具的必要性可以从三个角度看：知识时效性（训练数据有截止日期）、逻辑推理补强（避免幻觉和计算错误）、能力边界扩展（完成训练未覆盖的任务）。未来两者会走向融合，形成"意图解析-协议传输-工具执行"的分层架构，Function Calling 负责理解该调用什么，MCP 负责标准化传递指令。

---

## 一句话总结

Function Calling 是大模型的内置能力负责"理解该调用什么"，MCP 是标准化协议负责"怎么规范地传递调用"，两者协同让 Agent 突破知识边界、避免幻觉、扩展能力。