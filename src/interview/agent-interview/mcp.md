---
title: MCP 是什么？它解决了大模型连接外部工具的什么问题？
icon: mdi:power-plug-outline
tag:
- Agent面试
- MCP
- 工具调用
description: 面试中如何清晰口述 MCP？通过贯穿案例解析协议解耦原理、Prompts/Resources/Tools 三大基语与技术边界。
---

在大模型开发中，让模型调用外部工具扩展能力，我们通常最先接触到的是模型厂商的原生工具调用机制（可参考 [/interview/agent-interview/tool-calling.md](/interview/agent-interview/tool-calling.md)）。但在真实业务场景中，当团队需要将多个不同的 AI 应用同时接入多个外部数据系统时，点对点的私有代码往往会导致严重的维护负担。Anthropic 在 2024 年底开源的 MCP（Model Context Protocol，模型上下文协议）就是为了统一这一通信层制定的开放标准。

![MCP协议核心定位与大模型生态连接架构图](/images/content-rebuild/b61d3ff958c59dc2/cover.png)

上图展示了讲解员在白板前梳理大模型连接生态，明确 MCP 是一套通信协议规范，连接不同的模型宿主应用与外部工具系统。理解 MCP，需要跳出“它只是一个工具调用库”的视角，从接口解耦和系统标准化来认识它。

## 1. 面试标准口述：MCP 的定义与生态定位

在面试中被问及“什么是 MCP”时，回答切忌长篇大论背诵文档，建议采用“定义 + 痛点 + 架构三要素 + 边界”的结构作答：

> **面试口述标准答案：**
> “MCP（Model Context Protocol）是由 Anthropic 发起并开源的**应用层开放通信协议**。它的目标是为大模型宿主应用（Host）与外部数据源、工具服务（Server）之间建立标准化的通信通道。
>
> 它主要解决的是**多客户端接入多数据源时的 $N \times M$ 适配成本**。在没有统一协议前，每个 AI 工具都要为每个外部系统写专属插件；引入 MCP 后，双方通过统一的 JSON-RPC 2.0 规范通信，将接口适配复杂度收敛为 $N + M$。
>
> 协议定义了三种核心能力实体：供模型决策调用的工具（Tools）、供应用只读加载的外部数据（Resources），以及预置指令模板（Prompts）。
>
> 需要说明的是，MCP 本身是一套**通信协议规范**，并不负责模型的底层推理，也不是 Agent 的业务流程编排引擎。”

这一回答既给出了定义，又界定了工程边界，便于面试官展开后续追问。

## 2. 贯穿案例：没有统一标准时的 $N \times M$ 维护难题

为了直观理解 MCP 解决的问题，我们设定一个贯穿全文的实际工程场景：

假设团队有 3 个 AI 交互端（Host）：
1. **Claude Desktop**（桌面分析客户端）
2. **Cursor / IDE 插件**（开发者的代码助手）
3. **内部自研运维 Web 端**（基于开源模型搭建的运维助手）

同时，这些端都需要访问企业内部的 3 个数据服务（Server）：
1. **Git 仓库服务**（用于拉取代码与分支差异）
2. **Jira 工单系统**（用于查询与更新需求进度）
3. **只读审计日志服务**（用于排错与安全审计）

![N个模型接入M个工具导致的网状耦合与维护困境](/images/content-rebuild/b61d3ff958c59dc2/fig-1.png)

上图展示了在没有统一规范时，左侧客户端与右侧数据源之间直接直连的网状连接形态。

在点对点私有对接模式下，系统会面临以下三个具体的工程问题：

* **代码重复编写**：每个客户端都需要单独针对 Git、Jira、日志系统编写请求逻辑、鉴权处理和返回解析。3 个应用接入 3 个服务，一共要开发维护 $3 \times 3 = 9$ 套胶水代码。如果是 5 个客户端接入 8 个服务，就会膨胀为 40 套。
* **变更连锁反应**：如果 Jira 升级了 API 鉴权机制或修改了字段结构，开发人员必须在 3 个客户端的代码库中分别查找、修改并发布一次，测试与回归成本成倍增加。
* **数据格式不兼容**：不同厂商模型对工具参数（Tool Schema）的定义细节常有差异（详见 [/interview/agent-interview/tool-schema-compatibility.md](/interview/agent-interview/tool-schema-compatibility.md)）。开发者往往需要在各个端内手动转换参数格式，容易引入兼容性错误。

<ReadMoreLock />

## 3. MCP 的解耦方式：客户端与服务端的标准化连接

MCP 借鉴了软件工程中“语言服务器协议（LSP）”的设计思路。LSP 让 VS Code 等编辑器只需实现一套通用客户端，就能通过标准协议支持 Python、Go、Rust 等不同语言的分析器。MCP 也是如此，它在 AI 应用和外部数据之间建立了一层通用协议。

![MCP解耦架构：从N乘M收敛为N加M的标准星型拓扑](/images/content-rebuild/b61d3ff958c59dc2/fig-2.png)

上图展示了引入 MCP 之后的拓扑结构：各个客户端作为 MCP Host 挂载 MCP Client，各个外部系统分别封装为独立的 MCP Server，双方通过 JSON-RPC 2.0 协议交互。在理想情况下，接口适配工作量从原本的乘积关系变成了加和关系：3 个客户端各自实现一套通用的 MCP 客户端模块，3 个外部系统各自封装为一个 MCP Server，总共只需维护 $3 + 3 = 6$ 项适配工作。

### 系统角色划分

在 MCP 规范中，通信角色分为三个部分（详细机制见 [/interview/agent-interview/mcp-architecture.md](/interview/agent-interview/mcp-architecture.md)）：

1. **MCP Host（宿主应用）**：面向用户的顶层应用，例如 Claude Desktop、Cursor 或企业内部 Agent。Host 负责调用大模型、向用户展示界面，并在必要时发起安全审批。
2. **MCP Client（客户端模块）**：运行在 Host 内部的协议适配组件。每个 Client 与一个对应的 Server 建立并维持一条独立连接，负责消息的序列化、发送与协议解析。
3. **MCP Server（服务提供方）**：独立的进程或轻量网络服务。它包装底层系统（如 Git、Jira），将自身拥有的能力通过 MCP 协议暴露给外部。

### 通信载荷示意

Client 与 Server 之间默认通过 JSON-RPC 2.0 格式交换数据。例如，当客户端想要了解当前连接的 Jira Server 提供了哪些工具时，会发送 `tools/list` 请求：

```json
// 客户端发出的工具查询请求
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

服务端收到后，返回符合协议约定的工具描述列表：

```json
// 服务端返回的标准格式响应
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "query_jira_issue",
        "description": "根据工单 Key 查询工单标题、经办人与状态",
        "inputSchema": {
          "type": "object",
          "properties": {
            "issue_key": {
              "type": "string",
              "description": "工单唯一标识，例如 PROJ-1024"
            }
          },
          "required": ["issue_key"]
        }
      }
    ]
  }
}
```

在这种模式下，Host 只要能解析标准 JSON-RPC 格式，就能直接读取 Jira Server 暴露的工具列表，并把工具结构传递给大模型，不再需要为 Jira 单独手写一套专用接入代码。

## 4. 三大核心能力实体：Prompts、Resources 与 Tools

在很多初学者的印象中，MCP 往往被当作“工具调用协议”。但在官方规范中，MCP 服务向外部暴露的能力实体分为三类：**Resources（资源）**、**Prompts（提示模板）** 和 **Tools（工具）**。

![MCP三大核心开放能力基语对比：Prompts、Resources与Tools](/images/content-rebuild/b61d3ff958c59dc2/fig-3.png)

上图对比了 Prompts、Resources、Tools 三者的触发主体与数据属性区别。我们结合前面的贯穿案例来看它们的分工：

| 能力类型 | 主导触发方 | 状态属性与副作用 | 定位与表达形式 | 在贯穿案例中的典型用途 |
| :--- | :--- | :--- | :--- | :--- |
| **Resources（资源）** | 应用程序/模型被动读取 | **只读、无副作用**。类似只读文件，支持文本和二进制，支持监听变化 | 使用 URI 寻址，如 `git://repo/diff/main` 或 `file:///var/log/audit.log` | 读取当前代码分支差异、直接加载审计日志文本，供模型作为背景资料阅读 |
| **Prompts（提示模板）** | 用户交互主动选择 | **结构化指令模板**。提供预设的上下文引导与参数填槽 | 包含名称、描述与传入参数列表 | “代码审查向导”、“Jira 周报生成器”等预置角色与提示词模板 |
| **Tools（工具）** | 模型自回归推理决定调用 | **可读写、通常伴随系统状态改变（副作用）** | 包含函数名、功能描述与 JSON Schema 输入参数 | 在 Jira 中创建缺陷工单、在 Git 仓库创建 release tag、重启远程容器 |

区分这三者能带来明确的工程收益：

1. **被动数据与主动行为分离**：读取只读日志不需要封装成一个“查询日志工具”让模型猜测何时调用，Host 可以直接通过 URI 将 Resource 读取后注入上下文，减少了模型因推理失误而漏调工具的概率。
2. **复用提示词资产**：服务端可以将经过充分验证的 Prompt 模板直接随服务一同分发，团队成员只要连接该 MCP Server，就能在客户端下拉菜单中选用对应的流程模板。

## 5. 边界划分：MCP 是协议标准，不是推理算法

在技术面试中，面试官经常会考察候选人对“技术分层”的理解。需要明确区分 MCP 与模型推理、应用编排框架之间的定位边界。

![MCP作为传输协议层与模型推理算法及Agent循环的边界划分图](/images/content-rebuild/b61d3ff958c59dc2/fig-4.png)

上图清晰标出了系统的架构与职责边界：Host 内部承载业务应用与 Agent 决策循环，并通过标准 MCP 应用层协议与独立的 MCP Server 进程交互，Server 进一步包装底层外部系统。具体有三条边界：

1. **MCP 不改变模型的内在推理能力**：
   MCP 只是一套格式化通信规范。如果选用的模型本身理解力不足，面对复杂的业务需求，即便接入了高质量的 MCP Server，模型依然可能填错参数或选错工具。驱动多轮调用的推理循环仍由 ReAct 等算法主导（详见 [/interview/agent-interview/agent-loop.md](/interview/agent-interview/agent-loop.md) 与 [/interview/agent-interview/react.md](/interview/agent-interview/react.md)）。
2. **MCP 不替代工作流与 Agent 编排**：
   MCP 规定了“如何把请求发给 Server 并取回内容”，但并不决定“内容取回后走哪条分支、失败后按何种策略重试、何时让多 Agent 协作”。这些逻辑属于 LangGraph 等编排框架负责的应用层范畴（参考 [/interview/agent-interview/agent-frameworks.md](/interview/agent-interview/agent-frameworks.md)）。
3. **MCP 不自动解决上下文超限问题**：
   如果一个 MCP Server 暴露了包含上百个工具的清单，或者返回了兆字节级别的日志 Resource，未经筛选直接塞给模型，依然会耗尽模型的上下文窗口。上下文的管理与裁剪需要 Host 端结合业务策略主动处理（详见 [/interview/agent-interview/context-management.md](/interview/agent-interview/context-management.md)）。

## 6. 面试高频追问与回答要点

![面试官与候选人围绕MCP协议深度追问的技术演练场景](/images/content-rebuild/b61d3ff958c59dc2/fig-5.png)

上图模拟了技术面试场景中，围绕 MCP 与原生工具调用差异、上下文溢出防护、与 Agent Skills 区别展开的典型推演。

### 追问 1：原生 Function Calling 已经可用，为什么还需要 MCP？

*参考资料与延伸：[/interview/agent-interview/mcp-vs-function-calling.md](/interview/agent-interview/mcp-vs-function-calling.md)*

* **核心差异**：Function Calling 是模型厂商针对自家 API 提供的单点输出特性（把自然语言转换为 JSON 参数）；执行函数的动作通常需要开发者在应用代码中自己写硬编码调用。
* **MCP 带来的提升**：MCP 是一套完整的跨进程/跨网络通信标准。它不仅包含工具调用，还包含统一的资源发现、提示词分发、会话生命周期以及数据通道管理。外部工具被解耦到独立的 Server 进程中，Host 无论底层切换哪家模型，只要支持 MCP 协议，就能无缝挂载这些外部工具。

### 追问 2：连接的 MCP Server 暴露的数据量过大时，如何避免挤爆上下文？

* **回答要点**：
  1. **元数据与内容按需分离**：初次连接握手时，Host 只调用 `resources/list` 或 `tools/list` 获取轻量的名称、描述与 URI 格式，不加载真实数据内容。
  2. **按需延迟读取（Lazy Loading）**：只有当模型推理确实需要某份数据时，Host 才通过 `resources/read` 传入对应 URI 去拉取内容。
  3. **分页机制限制**：MCP 规范支持游标分页（Pagination），对于大列表数据通过 `cursor` 分批拉取，防止一次 RPC 返回海量文本撑爆 Token。

### 追问 3：MCP 与 Agent Skills（如 SKILL.md 模式）有什么区别？

*参考资料与延伸：[/interview/agent-interview/mcp-vs-skills.md](/interview/agent-interview/mcp-vs-skills.md) 与 [/frontier/agent-skills/what-is-agent-skills.md](/frontier/agent-skills/what-is-agent-skills.md)*

* **回答要点**：Skills 主要是以文档、参考指引和可选脚本组成的可复用经验包，核心是告诉 Agent“完成某项任务的具体步骤和规则”；MCP 则是标准化的底层通信协议，负责“打通应用与外部服务的数据通道”。两者可以结合使用：一个负责规划复杂流程的 Skill，底层可以直接调用 MCP Server 暴露出的具体工具执行操作。

## 7. 常见认知误区与规避建议

![MCP集成中的三大典型认知误区与反模式剖析](/images/content-rebuild/b61d3ff958c59dc2/fig-6.png)

上图总结了开发与面试交流中常见的三个误区及其防范策略。

### 误区 1：把后端所有的 API 接口直接原样转为 MCP Tools

* **现象与问题**：直接把内部 Swagger/OpenAPI 的上百个接口全部无差别暴露给 MCP Client。结果导致模型单次请求携带大量工具描述，浪费大量上下文 Token，且由于参数相似，模型极易调用错误接口。
* **规避建议**：按业务领域划分 Server，或者在 Host 端设置意图路由，每次请求只向模型动态注入与当前用户目标最相关的少量候选工具。

### 误区 2：以为使用了 MCP 就能自动保证系统安全

* **现象与问题**：认为 MCP 自带安全防护，在 Server 端编写直接执行操作系统 shell 脚本的工具，且 Host 端允许模型静默调用，这在遭遇提示词注入（Prompt Injection）时可能导致越权操作。
* **规避建议**：MCP 是协议标准，安全需依赖应用层实现。敏感写入操作必须在 Host 端配置“人在回路”（Human-in-the-loop）确认机制，Server 端自身也需要进行严格的输入参数校验与权限最小化控制（详见 [/interview/agent-interview/mcp-security.md](/interview/agent-interview/mcp-security.md)）。

### 误区 3：在 MCP Server 内部编写复杂的业务循环和多步编排

* **现象与问题**：在 Server 端硬编码多轮逻辑判断与分支跳转，导致 Server 变得笨重且难以在其他业务中复用。
* **规避建议**：Server 应尽量保持无状态和职责单一，专注于把底层能力规范化暴露；多轮循环、重试与复杂决策交给 Host 端或 LangGraph 等编排系统管理。

---

**建议进阶阅读：**
* 深入理解 MCP 的系统角色与拓扑结构：[/interview/agent-interview/mcp-architecture.md](/interview/agent-interview/mcp-architecture.md)
* 掌握 MCP 支持的传输通道与底层区别：[/interview/agent-interview/mcp-modes.md](/interview/agent-interview/mcp-modes.md)
* 梳理完整的通信时序与握手流程：[/interview/agent-interview/mcp-workflow.md](/interview/agent-interview/mcp-workflow.md)
