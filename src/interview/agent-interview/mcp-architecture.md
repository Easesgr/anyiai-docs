---
title: MCP 架构包含哪些核心组件？各自职责是什么？
icon: mdi:cube-outline
tag:
- MCP
- 架构
- 组件
description: 解析 MCP 的 Host、Client 与 Server 三层组件分工、一对多连接拓扑、初始化握手与跨进程隔离机制，附面试口述答案与追问梳理。
---

在面试中讨论 MCP（Model Context Protocol）时，面试官通常不满足于“它是一个连接工具的协议”这种泛化回答，而是考查**系统角色的职责划分、通信与连接关系，以及进程隔离带来的安全与稳定性考虑**。理解 Host、Client 与 Server 的边界，是排查多工具调用异常与设计 Agent 系统的基础。

---

## 面试标准口述：Host、Client、Server 的角色划分与连接结构

当面试官提出“MCP 架构包含哪些核心组件？各自职责是什么？”时，建议按以下结构作答：

> **面试口述速记**：  
> MCP 架构由 **Host（宿主环境）**、**Client（协议客户端）** 和 **Server（能力服务端）** 三个核心组件构成：  
> 1. **Host** 是面向用户和调配大模型的应用主程序（例如 Claude Desktop 或集成开发环境），负责接收用户请求、组装上下文提示词、调度大模型决策、渲染界面以及处理人机确认（Human-in-the-Loop）授权。  
> 2. **Client** 是驻留在 Host 进程内部的协议适配模块，与具体的 Server 保持 **1:1** 的通信通道，负责连接生命周期维护、协议初始化与能力协商（Capabilities Negotiation），以及 JSON-RPC 消息的编解码与分发。  
> 3. **Server** 是提供具体能力与数据的独立进程，负责封装本地或远程资源，通过标准协议向 Client 暴露 **Tools（工具）**、**Resources（资源）** 与 **Prompts（提示词模板）**。  
> 
> 系统的连接结构为：**1 个 Host 内部管理 N 个 Client，每个 Client 独立连接 1 个 Server**。组件间通常采用跨进程通信（如本地 `stdio` 或基于网络的 `Streamable HTTP`），从而隔离运行环境并避免单个服务异常导致整个主程序崩溃。

![MCP 三层核心组件与其依赖拓扑图](/images/content-rebuild/d8f3394e91ebee0f/cover.png)

为了直观理解这三个角色的协作，可以设想一个典型场景：用户在桌面助手（Host）中输入“分析本地 sales.db 中的退款记录，并在 GitHub 上提交一份分析报告”。Host 内部会分别运行对接本地 SQLite 的 Client A，以及对接代码平台的 Client B。Client A 与 SQLite Server 通信获取数据，Client B 与 GitHub Server 通信创建 Issue，整个过程由 Host 完成调度。

---

## MCP Host 职责：模型调度、上下文装配与人机授权

**Host 是整个 AI 应用的主控中心**，面向终端用户和底层大模型。

Host 在运行过程中承担以下核心工作：

1. **大模型决策与上下文编排**：Host 接收用户的自然语言输入，将各个 Client 汇报的可用工具与资源列表整理并注入上下文提示词中，提交给大模型。当大模型返回工具调用请求时，Host 解析请求并定位到对应的工具与 Client。
2. **多 Client 实例聚合与调度**：Host 统一管理多个 Client 实例的创建、配置加载与连接状态。当任务涉及多个外部数据源时，Host 负责把不同 Server 提供的能力聚合在同一对话上下文中，并在模型触发调用时代为分发。
3. **信任根与人机确认（Human-in-the-Loop）**：**图形界面展示与敏感操作拦截完全由 Host 负责**。Server 只负责在工具定义中声明参数和说明，无权直接在操作系统中弹出窗口打断用户。当大模型发出具有破坏性影响的操作（如删除表或覆盖文件）时，Host 负责拦截调用并在界面弹出审核框，由用户点击确认后才允许底层执行。

![Host 在工具调用过程中的人机确认与安全拦截时序](/images/content-rebuild/d8f3394e91ebee0f/fig-1.png)

将 Host 作为系统的信任根是保障安全的前提：大模型的生成结果具有概率性，外部 Server 可能是第三方编写的代码。只有掌握用户界面的 Host 具备拦截并核验执行动作的能力。

<ReadMoreLock />

---

## MCP Client 职责：连接维护、能力协商与消息路由

在很多初学者的理解中，Client 容易被误认为是独立部署的网关服务。实际上，**MCP Client 通常是 Host 程序内部的一个 SDK 模块实例**，充当 Host 与外部 Server 之间的协议适配驱动。

Client 的具体职责包括：

### 1. 通信通道管理与数据序列化
Client 负责维护与单个 Server 的通信连接。在本地环境下，Client 通常使用进程管道（`stdio`）拉起并管理子进程；在远程环境下，则通过网络流式协议（如 `Streamable HTTP`）与远端服务通信。Client 负责将 Host 内部的调用请求序列化为 JSON-RPC 2.0 报文，并反序列化 Server 返回的结果。

### 2. 初始化与能力协商（Capabilities Negotiation）
建立通信连接后，Client 与 Server 必须通过握手流程协商双方支持的功能范围：
- Client 发送 `initialize` 请求，携带支持的协议版本以及自身能力（例如是否支持根目录变更通知 `roots`、是否支持模型反向采样 `sampling`）。
- Server 回复其支持的协议版本和能力清单（如是否提供 `tools`、`resources`、`prompts` 或 `logging`）。
- Client 发送 `notifications/initialized` 通知，确认握手完成，正式进入运行状态。

```json
// Client 发送的 initialize 请求示例（教学示意）
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
      "name": "Dev-Host-Client",
      "version": "1.0.0"
    }
  }
}
```

### 3. 双向请求中继与状态对应
MCP 支持双向通信：不仅 Client 可以调用 Server 的 `tools/call`，开启了采样能力的 Server 也可以向 Client 发送 `sampling/createMessage` 请求，借助 Host 侧的大模型来协助完成子任务（例如让模型帮助分析一段文本的结构）。Client 负责跟踪每次请求的 `id`，将异步响应与最初发起的请求正确对应起来。

![Client 在初始化阶段与 Server 协商能力集合的交互机制](/images/content-rebuild/d8f3394e91ebee0f/fig-2.png)

---

## MCP Server 职责：功能实现、上下文提供与数据暴露

**MCP Server 是外部能力的具体提供方**。它专注于执行具体的输入输出逻辑，并通过协议暴露以下三类标准原语：

- **Tools（工具）**：可供模型调用的操作函数，通常伴随状态改变或外部 I/O。例如执行数据库读写、抓取网页内容、调用外部 Webhook。每次调用通过 `tools/call` 触发，返回执行结果。
- **Resources（资源）**：只读的数据源上下文，类似于带有统一标识符的文件或接口。Server 通过 URI（如 `file:///repo/README.md` 或 `db://schema/orders`）暴露内容，并支持向 Client 发送更新通知。
- **Prompts（提示词模板）**：由 Server 预置的结构化提示词模版。例如一个专门辅助 SQL 排错的 Server 可以提供名为 `analyze-slow-query` 的模板，方便 Host 快速加载标准排查逻辑。

![Server 暴露的 Tools、Resources 与 Prompts 三大能力维度结构](/images/content-rebuild/d8f3394e91ebee0f/fig-3.png)

Server 应当专注于业务处理。当工具执行发生错误（如 SQL 语法错误或文件不存在）时，Server 应捕获异常并返回符合规范的 JSON-RPC 错误对象，而不是直接让整个进程异常终止。

---

## 组件连接关系：单 Host 管理多 Server 的分发机制

理解 MCP 的系统连接需要厘清数量对应关系：

- **Host 与 Client 是 1:N 关系**：一个 Host 应用内部可以同时持有多个 Client 实例，分别用于管理文件读写、代码仓库操作或数据库连接。
- **Client 与 Server 是 1:1 关系**：每个 Client 实例只绑定一个特定的 Server 通道。MCP 规范要求每个连接通道的状态独立，单个 Client 不能在同一物理连接中混用多个 Server 的状态。

![单 Host 管理多 Client 并连接独立 Server 的一对多拓扑关系](/images/content-rebuild/d8f3394e91ebee0f/fig-4.png)

### 工具同名冲突的处理
当 Host 同时接入了两个均包含同名工具 `search` 的 Server（例如一个本地文件搜索 Server 和一个网络搜索 Server）时，Host 负责对工具进行区分和路由：
1. **前缀命名空间（Namespacing）**：Host 在向大模型注册工具时，为工具名称加上 Server 别名前缀，转换成 `fs_search` 和 `web_search`。
2. **调用路由分发**：当大模型输出调用 `fs_search` 时，Host 识别出前缀属于本地文件服务，去除前缀后将原始的 `search` 请求路由给对应的 Client 实例。

这样各个 Server 开发者只需关心自身内部的函数命名，不需要相互协调名称唯一性。

---

## 为什么 MCP 优先采用跨进程而不是进程内调用？

很多传统的插件系统习惯以模块形式直接导入（`import`）到宿主进程中。而 MCP 推荐通过跨进程（子进程或独立网络服务）运行 Server。这种设计主要是为了权衡以下系统工程考量：

| 考量维度 | 进程内库调用（In-Process） | MCP 跨进程架构（Out-of-Process） |
| :--- | :--- | :--- |
| **故障隔离** | 第三方代码发生未捕获异常、内存耗尽或段错误时，可能直接导致 Host 主程序崩溃退出。 | Server 进程崩溃仅导致当前通信连接断开，Host 主界面依然正常运行，可提示用户并尝试重启 Server。 |
| **技术栈解耦** | Host 与插件代码通常受限于同一种编程语言，或需要引入复杂的跨语言二进制桥接。 | 只要满足 JSON-RPC 协议标准即可通信。Host 无论采用何种技术栈构建，Server 都可以按需使用 Python、Go、Node.js 等独立实现。 |
| **权限与沙箱控制** | 插件与 Host 共享内存和系统环境，可能读取到敏感的环境变量或密钥。 | 可通过操作系统的沙箱机制、容器或受限用户身份运行 Server，按需授予其指定目录的访问权限。 |
| **资源开销** | 内存共享，无 IPC 序列化与上下文切换开销，执行延迟极低。 | 存在跨进程数据序列化与通信延迟，处理大规模高频数据流时需考虑传输成本。 |

![进程内直接打包与 MCP 跨进程架构在故障隔离与安全沙箱上的对比](/images/content-rebuild/d8f3394e91ebee0f/fig-5.png)

跨进程设计通过轻微的序列化与通信开销，换取了系统稳定性与多语言接入灵活性，使得本地开发者可以方便地通过配置文件启动和替换不同能力的工具服务。

---

## 面试追问与常见认知误区

### 追问 1：如果 Server 执行工具陷入死循环或长时间未响应，系统如何处理？
**回答思路**：  
Client 在发出请求时通常会挂载请求级定时器（Timeout）。若在规定时间内未收到 Server 的 JSON-RPC 响应：
1. Client 判定请求超时，向 Host 汇报超时错误；
2. Host 捕获错误后根据部署方式处理：对本地子进程 Server 可发送系统信号（如 `SIGTERM` / `SIGKILL`）终止挂起进程，对远程 Server 则中断请求或关闭传输通道，并提示用户是否重试；
3. Host 将“工具调用超时”作为工具执行观察结果（Observation）写入模型会话上下文，大模型可据此调整执行策略或向用户说明情况。

### 追问 2：为什么不让 MCP Server 直接连接大模型，而要通过 Host 进行中转？
**回答思路**：  
让外部 Server 直接调用模型会破坏统一上下文和安全边界：
1. **全局上下文断裂**：一个复杂任务往往需要多个工具协同。如果各 Server 分别与大模型通信，难以维持统一的任务记忆和执行历史；
2. **安全与授权缺失**：Server 往往是第三方编写的代码。没有 Host 作为人机交互界面，系统无法在危险调用发生时拦截并要求用户授权，容易引发越权与安全隐患。

![面试高频认知误区分析与组件归属对照图](/images/content-rebuild/d8f3394e91ebee0f/fig-6.png)

### 高频认知误区澄清

- **误区一：把 MCP Client 当作独立运行的网关程序**  
  *纠正*：Client 是嵌入在 Host 程序中的协议连接模块。每接入一个 Server，Host 就会在自身内存中初始化一个对应的 Client 实例，无需额外启动中间件。
- **误区二：误认为权限审核弹窗由 Server 弹出**  
  *纠正*：Server 通常运行在无界面环境中，不承担 UI 职责。弹窗审查和权限确认始终属于 Host 的职责范畴。
- **误区三：认为 MCP 会取代大模型内部的 Tool Calling 机制**  
  *纠正*：MCP 不替代模型层面的工具调用推理能力，而是定义了标准协议，用来规范模型外部如何发现工具、描述接口和传输调用参数。

---

## 知识自测与验收

检查对 MCP 架构核心组件的掌握程度，可以尝试回答以下两个问题：

1. 在 MCP 规范中，Client 与 Server 进行连接初始化的完整握手流程由哪几个步骤组成？
2. 为什么 Host 适合充当整个系统的“信任根”，而 Server 不适合？

如果能清晰说明 Client 发送 `initialize` 请求、Server 响应能力、Client 发送 `initialized` 通知的握手流程，并指出 Host 作为人机交互载体对高危调用的审查作用，就说明已经掌握了 MCP 核心组件的运行逻辑。
