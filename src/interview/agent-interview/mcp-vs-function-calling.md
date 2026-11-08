---
title: MCP 与 Function Calling 的区别是什么？
icon: mdi:compare
tag:
- MCP
- Function Calling
- Agent
description: 梳理模型工具调用能力与上下文通信协议的定位差异、协作链路、进程隔离机制与工程选型边界。
---

在智能体（Agent）开发与架构面试中，面试官经常会问：“既然大模型原生就支持 Function Calling，为什么社区还要搞一个 MCP？它们两个到底有什么区别？”

回答这个问题的关键，在于分清**“模型输出结构化参数的能力”**与**“跨系统交换上下文与执行调用的应用层通信协议”**。两者不是互相排斥的替代关系，而是上下游的协同组件。

![MCP 与 Function Calling 的技术层次关系概览](/images/content-rebuild/99484172f6ec51de/cover.png)

---

## 面试简洁口述：能力与协议的层次划分

如果需要在面试中用 1 到 2 分钟清晰作答，可以按以下三层结构口述：

1. **定位不同**：  
   **Function Calling 是大语言模型提供的 API 能力**。模型根据用户输入和预先给定的参数定义（JSON Schema），在回答中生成符合格式的结构化调用参数（例如 OpenAI 的 `tool_calls`）。它负责输出“调用什么函数、传什么参数”的意图决策，但模型本身不运行外部代码。  
   **MCP（Model Context Protocol，模型上下文协议）是一套开放的应用层通信协议**。它定义了宿主程序（Host/Client）与外部工具服务（Server）之间如何发现能力、读取资源和传递调用。它负责外部系统如何用统一的规范向 AI 应用提供工具与数据。
2. **职责分工**：  
   Function Calling 解决的是“模型与宿主程序之间如何表达调用参数”；MCP 解决的是“宿主程序与外部服务之间如何标准化连接与通信”。
3. **协作流程**：  
   宿主程序连接 MCP Server 拿到工具定义，将其转换为模型所需的 JSON Schema；模型通过 Function Calling 返回参数；宿主程序再通过 MCP 协议将调用发送给外部服务执行，最后把执行结果回传给模型。MCP 负责后端的工具与上下文接入，Function Calling 负责前端的模型意图解析。

![技术层次划分：决策层与协议传输层的协同关系](/images/content-rebuild/99484172f6ec51de/fig-1.png)

---

## 贯穿例子：查询订单并发送通知

为了看清两者在系统中的实际分工，我们以一个具体的客服场景为例：用户询问“帮我查一下订单 10086 的发货状态，并把物流短信发给用户手机”。

在这个任务中，系统需要连接内部订单数据库和短信网关：

1. **工具发现（MCP 负责）**：宿主程序启动时，通过 MCP 协议向订单服务和短信服务发送 `tools/list` 请求，获取到 `query_order` 和 `send_sms` 的参数结构说明。
2. **提供 Schema 给模型**：宿主程序把这两个工具的结构定义装配进 Prompt，发给模型 API。
3. **生成调用参数（Function Calling 负责）**：模型理解用户需求，命中 `query_order` 工具，并以 JSON 格式输出 `{"order_id": "10086"}`。
4. **执行工具调用（MCP 负责）**：宿主程序捕获这个输出，组织一条标准 MCP `tools/call` 请求，通过进程管道（stdio）或 Streamable HTTP 发送给订单 MCP Server。订单服务查库后返回物流数据。
5. **第二轮调用**：宿主程序将物流数据追加到上下文再次请求模型，模型通过 Function Calling 生成 `send_sms` 的调用参数；宿主程序再次通过 MCP 协议把短信发送指令转发给短信 MCP Server。

从这个链路可以看出，模型负责看懂上下文并填对参数，MCP 负责跨进程、跨网络把这些参数送到具体服务手中。

---

## 集成成本对比：点对点胶水代码 vs 统一协议标准

在引入 MCP 之前，基于原生 [Tool Calling 与 Function Calling](/interview/agent-interview/tool-calling.md) 开发系统时，通常面临多客户端对接多系统的集成成本。

假设团队维护了 $M$ 个不同的应用运行时（例如基于 Python 的后台自动化脚本、基于 Node.js 的桌面端助手、Web 聊天服务），需要连接 $N$ 个外部系统（如 GitHub、PostgreSQL、内部知识库、Slack）：

1. **点对点编写胶水代码**：  
   对于每个外部系统，开发者需要在不同的应用中分别编写 SDK 包装、认证逻辑、异常重试，并手动维护对应供应商（如 OpenAI、Anthropic、DeepSeek）所需的工具描述字段。不同模型供应商的字段定义存在微调，详见 [多模型 Tool Schema 适配](/interview/agent-interview/tool-schema-compatibility.md)。这导致每个客户端都要适配 $N$ 个服务，总维护成本随着服务增加而线性累加。
2. **接口变动引发连锁修改**：  
   如果某个外部工具的参数定义发生调整，所有调用该工具的客户端应用都需要同步修改源码并重新发版。

<ReadMoreLock />

![点对点粘合与协议化集成的复杂度对比](/images/content-rebuild/99484172f6ec51de/fig-2.png)

引入 [MCP 协议概述](/interview/agent-interview/mcp.md) 后，整个工程采用类似客户端-服务端的协议分工：

- **服务端单点实现**：数据源维护方按照 MCP 规范实现一个通用的 MCP Server，支持 `stdio` 本地进程传输或 `Streamable HTTP` 网络传输，并在协议中声明自己支持的工具列表与参数结构。
- **客户端协议复用**：所有支持 MCP 的客户端（无论是自研框架、Claude Desktop 还是支持 MCP 的编辑器），都可以通过标准的握手与发现机制动态获取这些能力，不再需要为每个服务单独编写专用客户端适配层。

*注意：这种协议化收益的前提是外部服务均提供了标准的 MCP 包装。如果目标系统没有 MCP Server，依然需要先为其编写一层服务适配代码。*

---

## 上下文能力范围：单一工具执行 vs 上下文三要素

部分开发者把 MCP 简单理解为“远程函数调用”，这忽略了 MCP 在上下文管理上的整体设计。

原生 Function Calling 的设计目标是单一的：**它仅针对工具（Tools）执行**。模型输出结构化参数，宿主程序执行操作并将结果返回给模型，驱动智能体的工作循环。具体执行循环机制可参考 [大模型如何调用工具](/fundamentals/llm-basics/function-calling.md)。

而按照 [MCP 核心架构组件](/interview/agent-interview/mcp-architecture.md) 的标准，MCP 将外部系统提供的内容抽象为三种交互范式：

| 交互范式 | 原生 Function Calling | MCP 协议规范 | 实际应用场景 |
| :--- | :--- | :--- | :--- |
| **工具执行（Tools）** | 原生支持。由模型发起决策，参数遵循 JSON Schema。 | 协议支持。通过 `tools/list` 与 `tools/call` 进行 RPC 通信。 | 执行带有副作用的操作（如插入数据库记录、发送邮件、调用外部 API）。 |
| **只读资源（Resources）** | **无协议级抽象**。需要开发者手动读取数据并拼入上下文，或包装成只读工具让模型主动调用。 | 协议支持。类比只读文件或 GET 请求，通过 URI（如 `file:///logs/app.log`、`postgres://orders`）暴露只读数据。 | 宿主程序无需消耗模型的推理轮次，可以直接将日志、系统配置、项目文件作为被动上下文装配给对话。 |
| **提示词模板（Prompts）** | **无协议支持**。开发者在应用层代码或配置文件中硬编码提示词。 | 协议支持。Server 端可直接向 Client 暴露预设的提示词模版与操作指令。 | 工具维护者可以把经过验证的提示词和调用指引随服务一同发布，客户端动态读取即可使用。 |
| **状态订阅与反向采样** | 无。模型与 API 之间为单次请求-响应模型。 | 协议支持。Client 可订阅 Server 资源变动通知；Server 也支持反向请求 Client 借助大模型进行采样（Sampling）。 | 资源订阅适用于配置与日志流更新通知；Sampling 适用于让外部服务借助宿主模型能力完成辅助推理。 |

![Function Calling 与 MCP 上下文能力范围对比](/images/content-rebuild/99484172f6ec51de/fig-3.png)

---

## 运行时与进程模型：进程内调用 vs 跨进程沙箱

除了协议定义，两者在系统运行时的部署结构和容错边界上也有明确差别。

### 1. 原生 Function Calling 的典型运行结构（进程内执行）

在很多轻量 Agent 框架中，大模型返回调用参数后，宿主程序直接在当前主进程内运行对应的 Python 或 Node.js 函数：

- **故障蔓延风险**：如果工具函数发生内存泄漏、执行死循环阻塞或触发未捕获的严重异常，可能导致整个应用服务崩溃退出。
- **依赖环境耦合**：如果工具 A 需要特定版本的系统类库，工具 B 需要冲突的版本，在同一个运行时环境下容易产生依赖冲突。
- **安全隔离受限**：本地执行的函数通常具有与宿主应用完全相同的系统权限（如文件读写、环境变量访问）。

### 2. MCP 的运行结构（跨进程 / 独立服务）

MCP 采用客户端/服务端架构，两者处于不同的执行环境：

- **故障隔离**：MCP Server 可以作为独立的本地子进程运行（通过 `stdio` 管道连接），也可以作为远程独立服务运行（通过 Streamable HTTP 连接）。执行机制分析见 [MCP 工作流程与执行时序](/interview/agent-interview/mcp-workflow.md)。单个 Server 如果发生故障中断，宿主程序只会收到连接异常或超时错误，主服务依然能正常运转。
- **环境隔离**：不同的 MCP Server 可以拥有完全独立的虚拟环境、运行时环境甚至 Docker 容器。
- **安全可控**：宿主程序在拉起 MCP 子进程或连接远程服务时，可以通过操作系统机制配置只读权限、隔离工作目录或网络出口策略。

*注意：MCP 协议本身只负责定义通信格式，并不会自动为代码构建沙箱。运行时的文件只读挂载、系统调用限制和网络访问控制，必须由宿主平台或容器环境显式配置与实施。*

![进程内函数内联与 MCP 跨进程隔离拓扑对比](/images/content-rebuild/99484172f6ec51de/fig-4.png)

---

## 链路实现：MCP 工具如何转换为 Function Calling

为了直观展示两者的连接方式，下面给出 Python 客户端中将 MCP Server 的工具转换为大模型 Function Calling 参数的核心链路代码：

```python
# 逻辑示例：展示 MCP 客户端如何把 Server 工具暴露给模型并完成调用执行
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

openai_client = OpenAI()

async def main():
    # 1. 配置子进程传输参数，启动本地 MCP Server
    server_params = StdioServerParameters(
        command="python",
        args=["order_mcp_server.py"]
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 2. 协议握手初始化
            await session.initialize()

            # 3. 工具发现：向 MCP Server 请求工具列表
            mcp_tools = await session.list_tools()

            # 4. 格式映射：把 MCP 工具 Schema 转换为 OpenAI Function Calling 格式
            openai_tools = []
            for tool in mcp_tools.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })

            # 5. 请求模型：由模型根据用户意图生成调用参数
            messages = [{"role": "user", "content": "查询订单 10086 的物流进度"}]
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto"
            )
            choice_msg = response.choices[0].message

            # 6. 执行转发：如果模型生成了工具调用，通过 MCP 执行并回填结果
            if choice_msg.tool_calls:
                for call in choice_msg.tool_calls:
                    tool_name = call.function.name
                    tool_args = json.loads(call.function.arguments)

                    # 向 MCP Server 发起跨进程调用
                    result = await session.call_tool(tool_name, tool_args)
                    print(f"MCP Server 执行结果: {result.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

代码清晰反映了两者职责边界：  
- `session.list_tools()` 和 `session.call_tool()` 是 **MCP 客户端协议** 的标准操作；  
- `openai_client.chat.completions.create(..., tools=...)` 与 `choice_msg.tool_calls` 是 **大模型 Function Calling 能力** 的消费过程。

![从工具发现到参数执行的转换调用时序图](/images/content-rebuild/99484172f6ec51de/fig-5.png)

---

## 工程选型：何时使用 MCP，何时坚持原生实现？

在架构设计面试中，不能一味推崇新技术，解释清楚技术方案的成本与边界才是衡量工程成熟度的依据。

### 适合引入 MCP 的场景
1. **多端复用工具库**：团队同时存在 Web 助手、命令行工具和 IDE 插件，需要共享统一的企业内部服务连接器。
2. **工具动态管理与生态集成**：系统需要支持用户自主添加第三方集成（如各类公有云、代码托管平台的预制连接器），或在不改动主应用代码的前提下插拔工具。
3. **强隔离与独立部署**：工具执行需要安装庞大的特定系统库，或者需要执行用户提交的代码，必须放在独立的容器或微服务中运行。

### 坚持使用原生 Function Calling 的场景
1. **本地轻量级计算与延迟敏感操作**：应用仅需要执行几个纯内存计算函数（如本地文本拼接、日期计算或轻量缓存读取）。MCP 的 JSON-RPC 序列化与跨进程传输会引入几毫秒到几十毫秒的额外耗时，原生直接调用开销更低。
2. **验证期原型（MVP）**：项目处于概念验证阶段，只需通过 10 行 Python 代码定义本地函数即可跑通验证流程。此时引入 MCP 会增加子进程管理、异步事件循环和协议调试的额外复杂度。
3. **强依赖宿主内部状态**：如果工具需要频繁直接操作宿主内存中的 UI 状态机或共享变量，跨进程通过协议调用会增加状态同步开销。

![工程选型决策路径图](/images/content-rebuild/99484172f6ec51de/fig-6.png)

---

## 面试追问与避坑解答

### 追问 1：既然 MCP 这么规范，它会取代模型提供商原生的 Function Calling 吗？

**答题要点**：  
不会取代，二者属于不同层次。只要模型与宿主程序之间依然通过文本或结构化流通信，系统就需要依赖大模型自身的 Function Calling 能力来解析意图、生成参数。MCP 是宿主应用连接外部系统的总线，它的工具必须先喂给模型的 Function Calling 接口，由模型决策后再返回给 MCP 执行。两者是协同关系。

### 追问 2：MCP 工具与 Agent Skills 有什么区别？

**答题要点**：  
Agent Skills 通常是指向模型提供完成特定业务任务的指导文档（例如包含任务目标、执行步骤、规则约束与可选脚本的说明，如 `SKILL.md`），指导模型和宿主如何协作完成复杂任务。而 MCP 工具是跨进程提供底层执行能力的接口协议。关于两者的协作方式，可参考 [MCP 与 Skills 的架构配合](/interview/agent-interview/mcp-vs-skills.md)。一个 Skill 可以给出多步规划指导，模型或宿主在执行对应步骤时，再按需调用底层 MCP 工具完成具体的数据读写。

### 常见错误说法纠偏

- **错误说法 1**：“Function Calling 只能调本地函数，MCP 才能调远程接口。”  
  *纠正*：原生 Function Calling 本质是模型生成结构化参数，宿主程序拿到参数后完全可以用 HTTP 请求调用远程系统；MCP 协议本身既支持网络上的 Streamable HTTP 远程传输，也支持本地单机的 `stdio` 标准输入输出进程管道。
- **错误说法 2**：“用了 MCP，模型就不会填错工具参数了。”  
  *纠正*：MCP 负责传递 JSON Schema 和调用结果，参数生成的准确率取决于模型自身的推理能力、提示词质量以及 Schema 的描述清晰度，与中间传输协议无因果关系。
- **错误说法 3**：“MCP 协议是专为 HTTP 设计的 RESTful 规范。”  
  *纠正*：MCP 采用的是基于 JSON-RPC 2.0 的双向消息交互规范，传输层核心支持本地进程间通道（`stdio`）以及支持流式传输的 `Streamable HTTP`，旧版使用的 HTTP+SSE 是历史兼容方案。
