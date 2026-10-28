---
title: "如何用 MCP 把本地文件、数据库、浏览器和第三方 API 接入 AI 应用？"
icon: mdi:connection
tag:
  - MCP 协议
  - 系统集成
  - 本地工具
  - API 接入
---

![COVER](/images/mcp-protocol/mcp-integration_cover.png)

# 如何用 MCP 把本地文件、数据库、浏览器和第三方 API 接入 AI 应用？

这道题考的是**MCP（Model Context Protocol）协议**——Anthropic 在 2024 年 11 月开源的 AI 应用连接标准。核心要回答清楚：MCP 是什么、怎么通信、怎么集成、为什么比 Function Calling 强。

---

## 板块1：MCP 协议是什么——AI 应用的"USB 接口"

先说个痛点。以前 AI 想读写本地文件、查数据库、控制浏览器，要一个个写集成代码，换个模型可能还要重写。MCP 就是来解决这个的。

**MCP 的本质是个标准化协议**。它把 AI 与外部资源的交互，抽象成三类原语：

- **Resources（资源）**：只读数据，比如本地文件、数据库表结构
- **Tools（工具）**：可执行操作，比如读文件、查 SQL、搜索网页
- **Prompts（提示模板）**：预定义好的 prompt，方便复用

类比一下：就像 USB 接口让电脑能连各种外设，不管你插 U 盘还是移动硬盘，电脑不需要认识这个设备，只要接口标准对就行。MCP 就是 AI 的"USB 接口"。

**核心架构分四层**：

Hosts 是应用入口（比如 Claude Desktop、Cursor），下面连接多个 Clients，每个 Client 1:1 连接一个 Server，Server 才是真正对接外部资源的程序。

![MCP四层架构图，展示Hosts→Clients→Servers→Resources的层级关系](/images/mcp-protocol/20260716_192815_056832_01.png)

---

<ReadMoreLock />


## 板块2：两种通信模式——stdio 和 SSE 怎么选

MCP 的 Server 可以跑在本地，也可以跑在远程，这就涉及到两种通信模式的选择。

**stdio（标准输入输出）**：适合本地工具。通过子进程管道传递 JSON-RPC 消息，Server 作为 AI 应用的子进程运行。典型场景是文件系统操作、浏览器控制。

**SSE + HTTP**：适合云端服务。建立 HTTP 持久连接，Server 运行在远程机器上。典型场景是第三方 API（高德地图、天气接口）、远程数据库。

类比一下：stdio 像同一台电脑上用管道传纸条，SSE 像两个人通过网络发消息。

![stdio与SSE通信模式对比图，左侧stdio展示子进程管道通信，右侧SSE展示HTTP持久连接](/images/mcp-protocol/20260716_192811_548352_01.png)

选哪个？看场景。本地文件、浏览器用 stdio，性能好、不需要网络；远程 API、云端数据库用 SSE，不用在本地部署服务。

---

## 板块3：实战接入流程——MultiServerMCPClient 集成四步走

光说不练假把式，讲讲具体怎么集成。

**第一步：初始化配置**

```python
from mcp_client import MultiServerMCPClient

client = MultiServerMCPClient({
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
    },
    "database": {
        "url": "http://localhost:5000/sse"
    }
})
```

并发连接多个 Server，自动获取它们的工具列表。

**第二步：工具发现**

连接后，所有工具都注册到 Client 上。注意每个 Server 的工具会自动加前缀避免冲突，比如 `filesystem_read_file`、`database_query`。

**第三步：Agent 调用循环**

```python
while True:
    response = agent()
    if tool_calls := response.tool_calls:
        for call in tool_calls:
            # 返回值可能嵌套，检查 result 字段
            result = call.execute()
            if "result" in result:
                yield result["result"]
```

重点：返回值的结构可能不一致，高德地图返回的是 `{data: {...}}`，Neo4j 返回的是 `{results: [...]}`，要习惯处理这种情况。

**第四步：关闭连接**

```python
client.close()
```

别忘了调用 close()，否则会有资源泄露。

![MCP集成流程图，展示初始化→Server连接→工具发现→Agent调用→结果处理→close()的完整闭环](/images/mcp-protocol/20260716_192813_073920_01.png)

---

## 板块4：为什么选 MCP 而不是 Function Calling

Function Calling 是各 AI 厂商自己实现的功能调用方案，各家不通用。你在 OpenAI 写的 Function Calling，换到 Claude 可能要重写。

MCP 是协议层标准。一旦你的应用接入了 MCP 协议，200+ 第三方 Server 直接拿来用，不用一个个集成。

**类比**：Function Calling 像每个手机厂商自己造充电线，苹果用 Lightning、华为用 Type-C，你得买一堆线。MCP 像统一成 Type-C 标准，一根线全搞定。

![MCP与Function Calling对比表，展示适用范围、标准化程度、生态丰富度、通信方式四个维度](/images/mcp-protocol/20260716_192834_944960_01.png)

从开发者视角看，**最大变化是角色转变**：从"写工具"变成"选工具"。以前要自己开发文件读取、数据库查询的代码；现在从生态里选合适的 Server，配一配就能用。

---

## 板块5：真实场景举例——文件、数据库、浏览器、API 怎么接

讲讲具体怎么接入这四类资源：

**本地文件**：用 `@modelcontextprotocol/server-filesystem`，stdio 通信方式。适合 AI 读取项目代码、配置文件。

**数据库**：Neo4j 图数据库用 `mcp-neo4j`，时序数据用 `mcp-tinybird`，通常走 SSE 通信。云端数据库用这种方式最合适。

**浏览器**：用 Playwright MCP，stdio 启动一个 Playwright 进程。适合自动化测试、网页数据抓取。

**第三方 API**：高德地图用 `mcp-server-amap`，SSE 连接到高德的服务。返回格式注意处理，不是标准 MCP 格式。

![MCP Server生态图，展示官方/第三方/社区Server分类及典型代表](/images/mcp-protocol/20260716_192830_953408_01.png)

---

## 面试怎么答

**基础版**（100-150字）：

> MCP 是 Anthropic 开源的标准化协议，把 AI 与外部系统交互抽象为 Resources、Tools、Prompts 三类原语。核心架构是 Hosts→Clients→Servers→Resources 四层。通信模式分两种：stdio 适合本地工具（文件、浏览器），SSE+HTTP 适合云端服务（远程 API、数据库）。用 MultiServerMCPClient 统一管理多个 Server，工具自动加前缀区分来源，返回值需检查 result 字段，做参数校验。

**加分版**（加这些细节）：

> 补充几点工程实践：初始化时并发连接多个 Server，用 Promise.all 加速；工具调用后返回值格式可能不一致（如高德返回 `{data}`，Neo4j 返回 `{results}`），要做好兼容处理；Server 不再使用时必须调用 close() 避免资源泄露。MCP 的本质是范式转变——从"写工具"变成"选工具"，开发者角色从实现者变成编排者。

---

## 一句话总结

MCP 就是 AI 应用的"USB 接口"——用标准化协议把本地文件、数据库、浏览器、第三方 API 统一接入，开发者从"造轮子"变成"选轮子"。