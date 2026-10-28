---
title: "MCP Server、MCP Client 和 Tool 分别是什么？一次完整调用流程是怎样的？"
icon: mdi:server-network
tag:
  - MCP 协议
  - MCP Server
  - MCP Client
  - 调用流程
---

# MCP Server、MCP Client 和 Tool 分别是什么？一次完整调用流程是怎样的？

![COVER](/images/mcp-protocol/mcp-server-client-tool_cover.png)

这道题考的是 MCP 协议最核心的架构设计。你要是连 Server、Client、Tool 这三个组件都说不清楚，那 MCP 你就是还没入门。

## 先理解 MCP 是什么——AI 世界的"USB 接口"

大模型有个致命短板：它只会训练时学到的数据，没法实时查天气、搜新闻、读文件。

怎么办？

MCP（Model Context Protocol）就是干这个的。它是 AI 模型连接外部工具的标准化协议，让大模型能调用各种真实世界的功能。

打个比方。USB 接口出现之前，鼠标、键盘、U盘各用各的接口，乱七八糟。MCP 就像一个统一的 USB 接口——不管你接什么设备，插上就能用。

对 AI 来说也一样。不管是查数据库、调 API、读文件，只要接入 MCP，就能统一管理。

![USB接口连接电脑与外部设备，类比MCP连接LLM与工具](/images/mcp-protocol/20260716_192156_073664_01.png)

<ReadMoreLock />


## 三个核心组件——Server、Client、Tool 各司其职

MCP 架构里就三个关键角色，你搞明白它们的关系，就搞懂了 MCP 一半。

**MCP Server** 是远端服务器，负责暴露各种工具和资源。它就相当于一个工具仓库管理员，你问它"你有啥工具"，它就给你列表；你说"帮我执行这个工具"，它就干活。

**MCP Client** 是你应用这边的组件，负责连接 Server、发现工具、发起调用。你可以让它列出所有可用工具，也可以让它执行具体某个工具。

**Tool** 就是具体的功能单元。每个 Tool 有个名称，有一组参数定义。你调用 Tool 时，就是告诉 Client："帮我调用 name=xxx 的工具，参数是 yyy"。

再打个餐厅的比方，帮助你记忆：

- Server = 后厨，负责实际做菜
- Client = 服务员，负责接收你的点单、传递菜单、把菜端回来
- Tool = 菜单上的具体菜品

这样对应，是不是清晰多了？

![三角色分工示意图，展示Server/Client/Tool的位置和职责关系](/images/mcp-protocol/20260716_192151_824192_01.png)

## 一次完整调用流程——六步走

光知道组件不够，还得知道它们怎么配合干活的。

**第一步**，用户提问。你问大模型："今天北京天气怎么样？"

**第二步**，LLM 推理判断。大模型一看，这问题训练数据里没有，得查外部工具。它回复说"我需要调用天气工具"。

**第三步**，MCP Client 发起调用。Client 调用 `call_tool("get_weather", {"city": "北京"})`，把请求发到 Server。

**第四步**，MCP Server 执行工具。Server 收到请求，调用真实天气 API，获取北京今天的温度、湿度、天气状况。

**第五步**，Server 返回结果。返回的是个 `content` 数组，里面装着工具的执行结果。

**第六步**，LLM 整合生成回答。大模型拿到天气数据，组织成自然语言告诉你："今天北京晴，25度，适合出门。"

整个流程你可以理解成点外卖：你下单（提问）→ 外卖小哥接单（Client 调用）→ 商家做饭（Server 执行）→ 小哥配送（结果返回）→ 你吃饭（LLM 生成回答）。

![六步调用流程图，从用户提问到LLM生成回答的完整链路](/images/mcp-protocol/20260716_192157_185088_01.png)

## 面试怎么答

**基础版**（能过的回答）：

MCP 是连接 AI 模型与外部工具的标准化协议。它的架构包含三个核心组件：MCP Server 是远端服务器，暴露各种工具并执行实际功能，通过 list_tools 和 call_tool 这两个核心方法提供服务；MCP Client 是应用端组件，负责连接 Server、发现可用工具、发起调用请求；Tool 是具体的功能单元，有名称和参数定义，调用时通过工具名和参数字典指定。一次完整调用流程是：用户提问 → LLM 判断需要外部工具 → Client 调用 call_tool → Server 执行工具并返回结果 → LLM 整合结果生成最终回答。

**加分版**（让面试官眼前一亮）：

在基础回答之上，你还可以补充几点。第一，MCP 的核心价值是"一次接入，处处使用"——只要实现了 MCP 协议，任何 AI 应用都能调用这些工具。第二，除了 Client-Server，MCP 还有个 MCP Host 角色，负责协调整个流程。第三，MCP 采用 SSE 长连接机制，Server 可以主动推送消息给 Client。第四，工具发现是通过 list_tools() 动态获取可用工具列表，应用可以随时知道现在能调哪些工具。第五，和 Function Calling 的区别要清楚——Function Calling 是模型内置能力，MCP 是外部协议；Function Calling 调用的是本地函数，MCP 调用的是远端服务器上的工具。

## 一句话总结

MCP 通过 Client-Server-Tool 的分层架构，让大模型能统一调用各种外部工具，核心就是"问→判断→调用→执行→返回→回答"这个六步流程。