---
title: MCP 的工作流程是什么？从握手到工具调用的完整时序
icon: mdi:workflow
tag:
- MCP
- 工作流程
- 协议
description: 详解 MCP 基于 JSON-RPC 2.0 的四阶段时序：初始化握手协商、动态能力发现、工具调用与进度流转、双层异常与取消机制。
---

在面试中被问到 Model Context Protocol（MCP）时，不少开发者只能回答出“Host、Client、Server 三者通过 JSON-RPC 通信”，一旦面试官深入追问“Client 如何知道 Server 开放了哪些能力”、“长耗时任务如何通知前端更新进度”、“SQL 执行报错该返回协议错误还是正常响应”，回答往往缺少明确的消息格式与时序状态依据。

MCP 是一个建立在 JSON-RPC 2.0 规范之上的应用层协议。它通过固定的消息交互阶段，解决了 AI 客户端与外部数据源、执行环境之间的握手协商、能力暴露与异步调用问题。

![MCP 会话全生命周期交互时序与消息分类概览](/images/content-rebuild/05e353ee409c0f41/cover.png)

---

## 60 秒口述答案：MCP 会话的四个核心阶段

面试时回答“MCP 的端到端交互流程”，可以直接按以下四个阶段组织口述：

> “MCP 的完整生命周期分为四个阶段，全程基于 JSON-RPC 2.0 规范通信：
>
> 1. **初始化握手（Handshake）**：Client 发送 `initialize` 请求携带自身支持的功能；Server 响应自身协议版本与能力字典（如 tools、resources、prompts）；Client 回复 `notifications/initialized` 单向通知。在 Server 收到此通知前，不得处理任何业务请求。
> 2. **能力与资源发现（Discovery）**：Client 按需发送 `tools/list` 或 `resources/list` 获取元数据与参数 Schema。当 Server 端运行时工具发生增减，会通过 `notifications/tools/list_changed` 主动广播，由 Client 自主决定是否重新拉取。
> 3. **工具调用与进度流转（Execution）**：大模型决定调用工具后，Client 发送 `tools/call`，可在请求元数据中携带 `progressToken`；Server 执行耗时任务期间可回传多条 `notifications/progress` 通知，最后返回统一格式的执行结果。
> 4. **异常与取消管理（Error & Cancellation）**：通信或方法解析失败返回标准 JSON-RPC Error；工具内部运行失败则在 Result 中标记 `isError: true`，便于大模型阅读错误并修正输入；超时或用户中断时，Client 可发送 `notifications/cancelled` 终止任务。”

为让流程更具体，下文以一个典型场景贯穿全篇：Host 客户端（如桌面分析应用）连接一个本地数据服务 `analytics-server`（提供销售数据分析与 SQL 查询）。前置架构角色可参考 [/interview/agent-interview/mcp-architecture.md](/interview/agent-interview/mcp-architecture.md)，传输通道细节可参考 [/interview/agent-interview/mcp-modes.md](/interview/agent-interview/mcp-modes.md)。

<ReadMoreLock />

---

## 阶段一：Initialize 握手与 Capabilities 双向协商

MCP 客户端在与服务端建立传输通道（如本地 stdio 进程管道或远程 Streamable HTTP 连接）后，不能直接发送业务请求。双方必须先完成对称的能力协商，以确定协议版本和各自支持的扩展特性。

![MCP 初始化握手与 Capabilities 双向协商时序流程](/images/content-rebuild/05e353ee409c0f41/fig-1.png)

### 1. 握手三步时序

```text
Client                                             Server
  |                                                  |
  | ----- (Request: id=1) initialize --------------> |
  | <---- (Response: id=1) protocolVersion + Caps -- |
  |                                                  |
  | ----- (Notification) notifications/initialized ->|  (握手完成，进入就绪态)
  |                                                  |
  | ===== 后续方可发起 tools/list、resources/read === |
```

1. **Client 发送 `initialize` 请求**：声明客户端所支持的 MCP 协议版本（如 `2024-11-05`）、自身客户端信息（`clientInfo`）与支持的能力字典（`capabilities`，例如是否支持文件根路径变更监听 roots、是否支持反向模型采样 sampling）。
2. **Server 返回 `InitializeResult` 响应**：返回服务端最终确认的协议版本、服务端标识（`serverInfo`）以及服务端开放的能力（`capabilities`，例如 tools、resources、prompts、logging）。
3. **Client 发送 `notifications/initialized` 通知**：这是单向通知（不含 `id` 字段），向服务端确认客户端已就绪。在收到该通知之前，Server 必须拒绝处理除了初始化之外的业务请求。

### 2. 交互报文样例

**Client -> Server（`initialize` 请求）**：
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
      "name": "AnalyticsHost",
      "version": "1.0.0"
    }
  }
}
```

**Server -> Client（响应协商结果）**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": { "listChanged": true },
      "logging": {}
    },
    "serverInfo": {
      "name": "analytics-server",
      "version": "0.4.0"
    }
  }
}
```

**Client -> Server（确认握手完成）**：
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

若 Server 在 `capabilities` 中未声明 `tools` 字段，说明该服务端不提供工具执行能力，Client 后续便不会尝试发送 `tools/list`。

---

## 阶段二：能力与资源发现（tools/list）及动态推送

握手完成后，Client 需要获取服务端公开的工具定义与参数约束，将其注入到大模型的上下文提示词或工具声明中。

![工具与资源列表发现及动态推送刷新流程](/images/content-rebuild/05e353ee409c0f41/fig-2.png)

### 1. 列表拉取与 Schema 解析

Client 发送 `tools/list` 请求。若工具数量较多，协议支持游标（Cursor）分页机制：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
```

Server 返回当前可用的工具数组。每个工具必须包含名称 `name`、用途描述 `description` 以及用于参数校验的 `inputSchema`（遵循 JSON Schema Draft-07 规范）：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "query_sales_data",
        "description": "执行只读 SQL 查询，拉取指定区域与年份的销售汇总记录",
        "inputSchema": {
          "type": "object",
          "properties": {
            "sql": { "type": "string", "description": "只读 SQL 语句" }
          },
          "required": ["sql"]
        }
      }
    ],
    "nextCursor": null
  }
}
```

### 2. 动态能力变更通知（List Changed）

在微服务或插件化运行环境中，服务端的可用工具可能随权限或配置变更（如动态挂载了新数仓）。MCP 采用服务端单向通知避免客户端轮询：

1. Server 检测到工具集发生变化；
2. Server 主动向 Client 发送 `notifications/tools/list_changed` 通知（无 `id` 字段）；
3. Client 接收到通知后，在适当时机重新触发 `tools/list` 请求拉取最新定义。

类似机制也存在于资源管理中，例如 `notifications/resources/list_changed`。

---

## 阶段三：工具调用时序（tools/call）与进度上报

当宿主的大模型在推理循环中决定使用某个工具时，Client 解析模型输出并组装为 `tools/call` 请求。关于模型如何做出调用决策，可参考 [/interview/agent-interview/tool-calling.md](/interview/agent-interview/tool-calling.md) 与 [/interview/agent-interview/mcp-vs-function-calling.md](/interview/agent-interview/mcp-vs-function-calling.md)。

![tools/call 完整调用时序与进度反馈流](/images/content-rebuild/05e353ee409c0f41/fig-3.png)

### 1. 进度令牌（Progress Token）机制

对于需要数秒甚至数分钟的长任务（如扫描海量分表或执行多维聚合），单次请求如果长时间没有数据流动，容易触发客户端网络超时，用户也无法得知任务进展。MCP 允许 Client 在请求元数据中传递 `progressToken`。

**Client 发送调用请求**：
```json
{
  "jsonrpc": "2.0",
  "id": 101,
  "method": "tools/call",
  "params": {
    "name": "query_sales_data",
    "arguments": {
      "sql": "SELECT region, SUM(amount) FROM sales_2024 GROUP BY region;"
    },
    "_meta": {
      "progressToken": "query-tok-5501"
    }
  }
}
```

**Server 执行期间分步推送进度**：
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/progress",
  "params": {
    "progressToken": "query-tok-5501",
    "progress": 40,
    "total": 100
  }
}
```

进度消息属于单向通知，Client 无需逐条回执，收到后可以直接在前端界面更新进度指示条。

### 2. 工具调用的正常响应

Server 执行结束后，返回与请求 `id` 对应的结果包。结果内容统一存放在 `content` 数组中，可包含纯文本、嵌入图片或内联资源，并通过 `isError: false` 表明业务成功：

```json
{
  "jsonrpc": "2.0",
  "id": 101,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "查询成功：华东区 1200 万元，华北区 850 万元。"
      }
    ],
    "isError": false
  }
}
```

---

## 阶段四：Request 与 Notification 的语义边界

在编写或排查 MCP 客户端及服务端时，混淆 Request 与 Notification 是常见的协议级错误。

![Request 消息与 Notification 消息结构对比](/images/content-rebuild/05e353ee409c0f41/fig-4.png)

| 对比维度 | JSON-RPC Request | JSON-RPC Notification |
| :--- | :--- | :--- |
| **标识符（`id`）** | **必须包含**（数值或字符串），用于严格配对返回包 | **严禁包含**；若包含 `id`，对端会强制作为 Request 解析 |
| **回执行为** | 对端必须返回 `result` 或 `error` 响应 | 单向发送（Fire-and-forget），对端严禁返回任何响应报文 |
| **传输方向** | 支持双向（Client 请求 Server 工具；Server 也可通过 sampling 请求 Client 提供大模型推理） | 支持双向（如进度推送、工具列表变更通知、任务取消通知） |
| **常见方法名** | `initialize`, `tools/list`, `tools/call`, `resources/read` | `notifications/initialized`, `notifications/progress`, `notifications/cancelled` |

根据协议健壮性规范：当任何一方收到未知的 **Notification** 时，**必须直接忽略**，不得回传错误；如果收到未知的 **Request**，则必须返回标准错误。

---

## 异常处理：区分“协议级错误”与“工具业务级错误”

面试中的高频考察点在于：工具执行异常与协议通信异常在 MCP 中应如何区分与返回？

![协议层错误与工具业务层错误在 MCP 中的处理分支判定](/images/content-rebuild/05e353ee409c0f41/fig-5.png)

### 1. 协议级错误（JSON-RPC Error）

当消息解析失败、方法不存在、或基础 JSON 结构畸形时，Server 必须返回带有顶层 `error` 对象的报文，此时响应中不得包含 `result` 字段。

| 错误码 | 错误名称 | 触发条件 |
| :--- | :--- | :--- |
| **-32700** | Parse error | Server 收到无法反序列化的非法 JSON 字节流 |
| **-32600** | Invalid Request | 报文缺失 `jsonrpc`、`method` 或基础字段格式错误 |
| **-32601** | Method not found | 请求了 Server 未声明的方法（例如在不支持 tools 的 Server 上调用 `tools/call`） |
| **-32602** | Invalid params | 请求参数未通过基本 Schema 校验或参数结构缺失 |
| **-32603** | Internal error | 服务端协议栈或运行时内部未捕获的严重故障 |

**协议级错误报文示例**：
```json
{
  "jsonrpc": "2.0",
  "id": 102,
  "error": {
    "code": -32601,
    "message": "Method not found: tools/unknown_method"
  }
}
```

### 2. 工具业务级错误（Tool Execution Failure）

如果客户端发送的 JSON-RPC 格式完全合法，`query_sales_data` 也成功被路由执行，但在执行 SQL 时发生了数据库报错（如 `Table 'sales_2024' doesn't exist`），此时**不能**返回 `-32603` 协议错误，而是要返回包含 `isError: true` 的标准 Result：

```json
{
  "jsonrpc": "2.0",
  "id": 103,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "SQL execution error: Table 'sales_2024' doesn't exist. Did you mean 'sales_2023'?"
      }
    ],
    "isError": true
  }
}
```

**机制原因**：
若返回 `-32603` 协议级错误，Client 端的协议通信层会将其作为通信异常抛出中断，宿主应用直接终止流程。而将业务错误封装在 `result` 中并标记 `isError: true`，错误文本能作为上下文完整传递回大模型，让大模型在后续推理步骤中读取具体的错误原因，修正 SQL 语句或查询参数后重新尝试调用。相关工程处理可参考 [/interview/agent-interview/agent-error-handling.md](/interview/agent-interview/agent-error-handling.md)。

### 3. 超时与取消时序

![请求超时与任务取消机制时序](/images/content-rebuild/05e353ee409c0f41/fig-6.png)

当工具运行耗时超过 Client 设置的阈值（如等待 60 秒未收到任何结果或进度通知），或者终端用户主动点击了“停止生成”按钮时，容错时序如下：

1. Client 内部定时器触发超时，或捕获到用户中断事件；
2. Client 向 Server 发送 `notifications/cancelled` 单向通知，携带超时的 `requestId` 与取消原因：
   ```json
   {
     "jsonrpc": "2.0",
     "method": "notifications/cancelled",
     "params": {
       "requestId": 101,
       "reason": "Client timeout after 60s"
     }
   }
   ```
3. Client 注销该 `requestId` 对应的等待回调（Promise/Future）；
4. Server 收到取消通知后，主动终止正在执行的后台子任务（如中断数据库查询或终止子进程）；若 Server 随后仍送回了延迟响应，Client 协议层静默丢弃该响应，避免数据串扰。

---

## 面试追问与技术边界

### 追问 1：在 `notifications/initialized` 发出前，Client 可以直接调用 `tools/call` 吗？

> **回答要点**：不可以。MCP 规范明确要求，在收到 `notifications/initialized` 之前，Server 不能处理除初始化外的任何请求。此时双方的能力字典与版本仍在协商中，如果在握手未确认前发送 `tools/call`，合规的 Server 必须拒绝执行并返回 `-32600 Invalid Request`。

### 追问 2：进度通知（`notifications/progress`）如何防止过密推送影响性能？

> **回答要点**：
> 1. **按需开启**：只有 Client 在调用参数的 `_meta` 中显式传入了 `progressToken`，Server 才会上报进度。若未提供 Token，Server 即使有进度逻辑也不得发送通知；
> 2. **服务端限流与采样**：长耗时任务在实现时，应对进度通知做节流控制（例如按时间间隔或关键进度节点推送），避免在快速循环中瞬间向管道灌入大量消息，导致缓冲区堆积。

### 追问 3：如果 Server 发送了 Client 无法识别的通知，Client 应该报错还是忽略？

> **回答要点**：必须忽略。根据 JSON-RPC 2.0 与 MCP 规范，Notification 属于单向消息，严禁对其返回错误响应；对端无法处理未知通知时应静默丢弃。只有当收到无法识别的 Request 时，才返回 `-32601 Method not found`。
