---
title: MCP Server、MCP Client 和 Tool 分别是什么？一次调用流程如何跑通？
icon: mdi:server-network
tag:
- MCP 协议
- MCP Server
- MCP Client
- 调用流程
description: 拆解 MCP 架构中 Host、Client、Server 与 Tool 的分工，对比 stdio 与 Streamable HTTP 通信机制，梳理握手、发现、调用时序与容错方案。
---

在大模型应用开发中，很多开发者容易把模型自身的工具生成能力与外部通信协议混淆。在 [MCP 协议到底是什么？为什么说它是 AI Agent 连接外部工具的标准接口？](/frontier/mcp-protocol/what-is-mcp.md) 中，我们了解了 MCP 作为上下文交换协议的定位。要真正将其落地到工程实现，需要搞清楚系统里的几个核心实体分别负责什么、底层通过什么通道收发消息，以及一次完整的调用过程包含哪些状态流转。

先给出一个具体结论：**MCP 本质上是在应用程序（Host）与上下文提供者（Server）之间建立的一套 JSON-RPC 2.0 通信约定。Host 通过实例化独立的 Client 与每个 Server 保持通信；Tool 是 Server 内部挂载的一段具体函数逻辑；模型负责根据 Tool 的描述生成调用参数，而 Host 和 Client 负责把指令发给 Server 执行并将结果回传给模型。**

以开发者在 VS Code 中提问“帮我查一下订单 ORD-20260401-8899 的物流状态”为例：VS Code 作为宿主，调用底层的订单系统 Server 查库，再把物流信息拼进上下文交给模型生成回答。下面我们围绕这个贯穿示例，逐步拆解 MCP 的实体职责、传输机制、协议握手与调用流程。

![MCP 核心实体与通信时序总览图](/images/content-rebuild/f001e65082364abe/cover.png)

---

## 四大核心实体的分工与关系

MCP 遵循客户端-服务端（Client-Server）架构，但在实际工程中，官方明确区分了宿主环境（Host）与通信客户端（Client）。四个核心实体的分工如下：

1. **MCP Host（宿主系统）**：直接面对终端用户或业务工作流的应用层外壳，例如 Claude Desktop、VS Code 或自定义的 Agent 编排应用。Host 负责调用大模型、维护多轮对话历史、执行安全鉴权（如弹窗询问用户是否允许执行危险操作），并管理底下的多个 MCP Client 实例。
2. **MCP Client（协议客户端）**：由 Host 实例化的轻量协议适配对象。Host 每连接一个 MCP Server，就会创建一个独立的 Client 实例。Client 负责维护底层的网络或进程通道、解析 JSON-RPC 2.0 报文，并将 Server 暴露出的能力汇总汇报给 Host。**Client 本身没有推理能力，也不决定什么时候调工具，它只负责转发指令与接收结果。**
3. **MCP Server（协议服务端）**：独立运行的轻量程序，负责对外暴露三类核心上下文：工具（Tools）、资源（Resources）和提示模板（Prompts）。Server 不直接与大模型对话，只响应 Client 发来的结构化请求并执行对应逻辑。
4. **Tool（工具单元）**：Server 内部注册的具体执行函数或 API 封装。每个 Tool 会通过 JSON Schema 声明自己的输入参数格式。Tool 本身不关心底层通信，只接收解包后的参数并输出结果。

![Host、Client、Server 与 Tool 架构拓扑与交互关系](/images/content-rebuild/f001e65082364abe/fig-1.png)

在代码实现中，Host 不会直接与 Server 进程进行底层套接字通信。以 VS Code 为例，当它同时连接本地文件系统 Server 和远程 Sentry 监控 Server 时，VS Code 内部会分别实例化两个平行的 MCP Client 对象。Client 与 Server 之间维持 1:1 的独立通道，两个通道彼此隔离，互不干扰。

关于大模型如何根据提示词输出工具名称与参数，可以参考 [Tool Calling 是什么？大模型如何调用工具？](/llm-100/agent-basics/Tool_Calling_是什么_大模型如何调用工具.md)。

<ReadMoreLock />

---

## 两种传输通道：stdio 与 Streamable HTTP

MCP 的协议报文遵循 JSON-RPC 2.0 规范，其传输层是解耦的。官方标准规范目前提供两种主流的传输实现：**本地进程管道（stdio）** 与 **远程网络流（Streamable HTTP）**。

![本地进程管道 stdio 与网络 Streamable HTTP 通信传输原理对比](/images/content-rebuild/f001e65082364abe/fig-2.png)

### 1. 本地进程通道：stdio

stdio 适用于本地桌面应用或同机容器环境。Client 负责拉起 Server 对应的子进程，并通过标准 I/O 管道进行双向通信：
- **Client -> Server**：Client 将单行 JSON-RPC 请求写入子进程的 `stdin`，并以换行符（`\n`）作为消息边界。
- **Server -> Client**：Server 将响应报文写入自身的 `stdout`，同样以换行符定界返回。
- **日志隔离要求**：Server 内部的业务调试日志必须输出到 `stderr`，绝对不能写入 `stdout`。因为 Client 会把 `stdout` 中的每一行都当成 JSON-RPC 报文解析，混入普通日志会导致 JSON 解析抛错从而中断连接。

stdio 的优点在于无需占用网络端口，依赖操作系统自带的进程权限进行隔离，非常轻量。但它的生命周期与宿主绑定，通常只供单个 Client 独占使用。

### 2. 远程网络通道：Streamable HTTP

当 Server 部署在云端容器、共享集群或跨主机环境时，使用 Streamable HTTP。早期 MCP 曾采用过基于 Server-Sent Events 的 HTTP+SSE 双通道设计（Client 发 POST，Server 经 SSE 返回）。最新标准已规范化为 **Streamable HTTP**：
- **单向请求与增量流**：客户端向指定端点发起标准的 HTTP POST 请求，消息体为 JSON-RPC 报文。服务端可返回常规的单次响应（HTTP 200 JSON），也可以使用分块传输（Chunked Transfer Encoding）保持连接，按流式回传事件或增量数据。
- **连接无需永久保活**：无状态请求下，HTTP 连接在处理完毕后即可关闭，只需在请求头中携带会话标识；只有在需要服务端主动推送变更时，才保持流式通道。这让单个远程 MCP Server 能够轻松支持多个 Client 并发连接。

| 特性 | stdio 传输 | Streamable HTTP 传输 |
| :--- | :--- | :--- |
| **适用场景** | 本地桌面端、同机子进程 | 远程服务、跨网络、云端微服务 |
| **连接载体** | 操作系统匿名管道（Pipe） | 标准 HTTP 连接（HTTP/1.1 或 HTTP/2） |
| **并发模型** | 1:1（一个 Server 子进程服务一个 Client） | 1:N（一个远端服务承接多个客户端连接） |
| **消息定界** | 换行符（`\n` 单行 JSON） | HTTP Headers、Content-Length 或分块数据流 |
| **日志与排查** | stdout 跑协议，stderr 跑日志 | HTTP 响应跑协议，服务端集中收集日志 |

---

## 连接建立过程：握手与能力协商

在典型的有状态连接（如本地 stdio 模式或有状态网络会话）中，双方在收发具体业务数据之前，通常先完成版本校验与能力协商。标准生命周期由三次报文交互组成（而在支持无状态的请求中，元数据可随请求携带，独立发现协商是可选且可缓存的）。

![MCP 协议连接初始化握手与能力协商时序图](/images/content-rebuild/f001e65082364abe/fig-3.png)

### 步骤 1：Client 发送初始化请求（initialize）

在标准生命周期流程中，Client 打开传输通道后首先发出 `initialize` 方法。在完成该握手前，Server 通常会拒绝或忽略后续的业务请求。

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "roots": { "listChanged": true },
      "sampling": {}
    },
    "clientInfo": {
      "name": "VSCode-MCP-Host",
      "version": "1.0.0"
    }
  }
}
```

Client 会在 `capabilities` 中声明自己支持哪些宿主特性，例如是否支持工作区目录变更通知（`roots`），或是否允许 Server 反向请求大模型进行采样推理（`sampling`）。

### 步骤 2：Server 确认并返回支持的能力（initialize 响应）

Server 检查协议版本。若版本兼容，Server 返回确认消息，并声明自己具备的能力：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": { "listChanged": true },
      "resources": { "subscribe": false }
    },
    "serverInfo": {
      "name": "OrderQueryServer",
      "version": "0.5.2"
    }
  }
}
```

如果响应体里包含 `capabilities.tools`，代表 Server 能够提供工具功能；如果该字段包含 `listChanged: true`，说明该 Server 支持在工具发生变动时主动发通知通知客户端刷新。

### 步骤 3：Client 发送就绪通知（initialized）

Client 收到并确认 Server 的能力参数后，发送一个通知：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

注意该报文没有 `id` 字段，属于无需回复的单向通知（Notification）。发送完毕后，通信状态正式转为“就绪”。

---

## 工具发现与动态变更机制

握手完成后，Host 需要向 Server 查询当前可用的工具列表，并把这些工具的信息挂载到大模型的提示词中。

![tools/list 报文发现与 listChanged 动态通知刷新机制](/images/content-rebuild/f001e65082364abe/fig-4.png)

### 1. 查询可用工具（tools/list）

Client 向 Server 发送 `tools/list` 请求：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
```

Server 返回当前可供调用的全部工具定义：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "query_order_status",
        "description": "根据订单编号查询物流状态与配送轨迹",
        "inputSchema": {
          "type": "object",
          "properties": {
            "order_id": {
              "type": "string",
              "description": "订单ID，格式形如 ORD-YYYYMMDD-XXXX"
            }
          },
          "required": ["order_id"],
          "additionalProperties": false
        }
      }
    ]
  }
}
```

每个 Tool 都包含名称、功能说明和 `inputSchema`（基于 JSON Schema 规范）。Host 收集到这些信息后，会将其转换成各大模型 API 所要求的格式注入上下文。不同厂商对 Schema 的支持差异，可参考 [不同 LLM Provider 对 Tool Schema 的支持不完全一致，如何处理差异？](/interview/agent-interview/tool-schema-compatibility.md)。

### 2. 运行时动态更新（list_changed）

当 Server 上的插件热重载或权限切换导致可用工具改变时，若此前已协商支持变更通知，Server 会主动发送一条单向通知：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/tools/list_changed"
}
```

Client 接收后，会重新发起 `tools/list` 拉取最新列表并同步给 Host，Host 即可在下一轮对话中无感使用新的工具集。

---

## 一次完整的端到端调用过程

回到最初的订单查询场景，用户在界面输入：“帮我查一下订单 ORD-20260401-8899 的状态”。一次完整的交互流程包含以下步骤：

![端到端工具调用时序闭环与宿主拦截流](/images/content-rebuild/f001e65082364abe/fig-5.png)

1. **用户提问与模型推理**：Host 将用户输入与之前获取的工具 Schema 一并提交给大模型。大模型判断该问题需要查询订单，输出结构化的工具调用决策：`query_order_status(order_id="ORD-20260401-8899")`。
2. **Host 策略检查与安全拦截**：
   - **模型输出不可盲信**：大模型生成的参数可能存在幻觉，或者可能调用修改数据、删除文件的危险指令。Host 作为安全边界，有权在此步骤进行拦截检查。
   - 对于只读的查询工具，Host 可直接放行；若遇到支付、删除等写操作，Host 可以暂停调用，弹窗让用户手动授权确认后再继续。
3. **Client 派发调用请求（tools/call）**：用户确认或策略放行后，Host 找到管理该 Server 的 Client 实例，发送执行请求：
   ```json
   {
     "jsonrpc": "2.0",
     "id": 101,
     "method": "tools/call",
     "params": {
       "name": "query_order_status",
       "arguments": {
         "order_id": "ORD-20260401-8899"
       }
     }
   }
   ```
4. **Server 执行与返回**：Server 拿到参数后调用内部对应的 Python 或 TypeScript 函数，完成数据查询并返回响应：
   ```json
   {
     "jsonrpc": "2.0",
     "id": 101,
     "result": {
       "content": [
         {
           "type": "text",
           "text": "{\"status\": \"SHIPPED\", \"courier\": \"SF\", \"tracking_no\": \"SF1029384756\"}"
         }
       ],
       "isError": false
     }
   }
   ```
5. **上下文回填与最终生成**：Client 把返回内容解包并交还 Host。Host 将执行结果作为工具消息（Tool Message）追加到对话历史中，再次请求大模型。大模型据此生成自然语言回答：“您的订单已发货，顺丰单号为 SF1029384756。”

工程落地中关于网络抖动和容错回退机制，可以参阅 [Agent 如何处理工具调用失败？](/llm-100/agent-engineering/agent-tool-failure.md)。

---

## 异常与失败处理机制

在生产环境中，网络故障、非法参数或子进程异常退出都是常见情况。MCP 协议从两个层面定义了错误模型：**协议级错误** 与 **业务级错误**。

![异常处理机制：协议错误码、业务错误与进程退出检测容错](/images/content-rebuild/f001e65082364abe/fig-6.png)

### 1. 协议级错误（JSON-RPC 顶层 error）

当请求方法不存在、报文语法错误或输入格式不合法时，Server 会返回带有顶层 `error` 对象的报文，不返回 `result` 字段：

```json
{
  "jsonrpc": "2.0",
  "id": 102,
  "error": {
    "code": -32602,
    "message": "Invalid params: Missing required field 'order_id'"
  }
}
```

常见的规范错误码包括：
- `-32700`（Parse error）：JSON 格式错误，Server 无法解析。
- `-32600`（Invalid Request）：报文缺少必需的 JSON-RPC 字段。
- `-32601`（Method not found）：请求的 JSON-RPC 协议方法未在 Server 注册或不支持（如 Server 未实现某项 MCP 规范方法；调用具体 Tool 不存在不属于该错误）。
- `-32602`（Invalid params）：传入的参数不符合 inputSchema 的要求。
- `-32603`（Internal error）：Server 运行时底层出现未捕获的严重异常。

### 2. 业务级执行错误（isError 标志）

如果参数格式正确，但业务执行时出现预期内的失败（例如“查询的订单号不存在”或外部 API 返回 404），**Server 不应返回顶层 JSON-RPC 错误，而应当返回正常的 `result` 报文，并将 `isError` 设为 `true`**：

```json
{
  "jsonrpc": "2.0",
  "id": 103,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Order ORD-99999999-0000 not found."
      }
    ],
    "isError": true
  }
}
```

**为什么这么设计？**
因为业务层面的失败本身也是有价值的上下文。大模型看到“订单未找到”后，能够自然地引导用户：“系统里未查到该单号，请检查是否输入有误”，而不是让客户端直接抛错崩溃。

### 3. stdio 进程异常退出检测

在 stdio 运行模式下，Server 子进程可能因内存溢出（OOM）或未处理的 Panic 崩溃退出。Client 依赖底层操作系统机制进行检测与资源清理：
1. **管道关闭与 EOF**：子进程崩溃时，其 stdout 管道会自动关闭。Client 在读取数据流时会收到文件结束符（EOF）。如果此时 Client 正在向 stdin 写入数据，则会收到管道破裂信号（如 Linux 的 `EPIPE` 异常）。
2. **进程退出监听**：Client 监听子进程的退出事件，捕获退出状态码与异常信号。
3. **挂起请求清理与降级**：Client 必须立即将所有正在等待响应的请求（Pending Promises）标记为失败并抛出断开连接异常，避免 Host 界面一直转圈。同时通知 Host 更新 UI 状态，并根据策略决定是否重新拉起 Server 重新执行初始化握手。

系统设计层面的更多落地参考，可以查看 [如何设计一个 Agent 工具调用平台？](/interview/ai-system-design-interview/agent-tool-platform.md)。而在下一篇 [如何用 MCP 把本地文件、数据库、浏览器和第三方 API 接入 AI 应用？](/frontier/mcp-protocol/mcp-integration.md) 中，我们将通过实际代码演示如何在具体工程中实现这些接口。
