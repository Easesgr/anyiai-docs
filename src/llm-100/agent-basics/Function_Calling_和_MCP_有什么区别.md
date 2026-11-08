---
title: Function Calling 与 MCP 有什么区别？从单模型工具调用到上下文协议
icon: mdi:cpu-64-bit
date: 2026-07-16
tag:
- 大模型100讲
- Agent
description: 理清 Function Calling 与 MCP 的本质区别：模型结构化生成能力与应用层解耦协议的分工、三层架构与工程选型标准。
---

在构建大模型应用时，许多开发者容易将 **Function Calling（函数调用）** 与 **Model Context Protocol（MCP，模型上下文协议）** 混为一谈，甚至误认为 MCP 是一种新型的大模型微调技术或 Function Calling 的替代品。

实际上，这两者处于完全不同的技术分层：
* **Function Calling** 是**大模型自身的能力**：模型接收提示词和工具参数模式（JSON Schema），在补全时输出符合格式的结构化文本（如函数名和实参 JSON），但模型本身并不执行任何代码。
* **MCP** 是一套**应用层通信协议**：它基于 JSON-RPC 2.0，规范了运行模型的宿主应用（Host/Client）与提供数据或工具的服务端（Server）之间如何建立连接、发现能力、读取资源与发起执行。

![Function Calling 与 MCP 架构概念总览](/images/content-rebuild/2f28ce2e674f36ae/cover.png)

为了直观理解两者的分工，我们以一个贯穿全文的场景为例：假设团队正在开发一个“数据库运维助手”，它需要具备两项能力——查询某个订单的支付状态，以及在必要时重启只读从库的同步服务。

---

## 原生 Function Calling 的工作机制与瓶颈

在只有单个应用和少数几个固定工具时，开发者通常直接使用模型供应商提供的原生 Function Calling。

### 1. 原生调用的生命周期
原生 Function Calling 的执行链路完全运行在业务进程内部：

```python
# 原生 Function Calling 伪代码：业务进程直接承载一切
import json
from openai import OpenAI

client = OpenAI()

tools = [{
    "type": "function",
    "function": {
        "name": "query_order",
        "description": "依据订单号查询支付状态",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"]
        }
    }
}]

# 1. 宿主向大模型发送对话与工具定义
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "帮我查一下订单 ord_1024 的状态"}],
    tools=tools
)

# 2. 模型返回结构化调用指令（模型未执行代码，仅做文本生成）
tool_call = response.choices[0].message.tool_calls[0]
if tool_call.function.name == "query_order":
    # 3. 宿主应用必须自行在本地环境编写执行逻辑
    args = json.loads(tool_call.function.arguments)
    result = db_query_order(args["order_id"])
    
    # 4. 宿主将执行结果重新打包传回模型，生成最终回复
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "帮我查一下订单 ord_1024 的状态"},
            response.choices[0].message,
            {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}
        ]
    )
```

在这一流程中，模型的作用仅限于理解用户意图并产出 `tool_call` 参数对象；真正的外部系统访问、凭证管理和安全隔离全部由宿主代码硬编码处理。

### 2. $N \times M$ 点对点集成的扩展困境
当系统复杂度上升时，这种紧耦合的实现会暴露出三个核心问题：

![N×M 点对点工具绑定与 MCP 标准化解耦对比](/images/content-rebuild/2f28ce2e674f36ae/fig-1.png)

1. **多客户端复用困难**：假设运维团队除了自研的 Web Agent 之外，日常还使用 Cursor 编写脚本、使用 Claude Desktop 分析日志。如果想让这三个客户端都能调用“数据库运维工具”，就必须在三个不同的客户端环境下分别适配一遍接口和运行环境。
2. **外部服务协议与生命周期各异**：不同外部服务暴露能力的协议五花八门（本地 SDK、RESTful API、gRPC 或 CLI 等），认证鉴权和连接生命周期各不相同，宿主必须为每个外部服务编写专用的连接与转译代码，导致维护成本随服务数量剧增。
3. **上下文窗口膨胀**：所有工具的 JSON Schema 都必须在请求时全量塞进对话的系统提示中。如果一个系统挂载了 30 个内部 API，即使当前对话只用到了 1 个，其余 29 个工具的描述依然会占用每轮对话的输入 Token。

---

## MCP 的定位：解耦客户端与外部能力的协议标准

为了解决客户端与能力提供方的耦合，Anthropic 于 2024 年末主导开源了 **Model Context Protocol（MCP）**。

它的设计思想借鉴了现代开发工具中的 **Language Server Protocol（LSP，语言服务协议）**：
* 在 LSP 出现前，每款编辑器（VS Code、Sublime、Neovim）要支持 Python 补全，都需要各自写一套解析插件（$N \times M$ 复杂度）。
* LSP 制定了通信规范后，Python 官方只需提供一个 Python 语言服务器，各类支持 LSP 的编辑器即可直接挂载（$N + M$ 复杂度）。

<ReadMoreLock />

MCP 在大模型与上下文系统之间建立了类似的标准化协议层。它并不改变大模型内部的注意力机制，也不替代 Function Calling，而是规定了宿主与外部服务之间如何用通用的 JSON-RPC 2.0 报文进行发现与调度。

---

## MCP 架构的三层角色划分

MCP 规范严格定义了系统中三个角色的职责边界：

![MCP 核心三层架构角色与职责切分](/images/content-rebuild/2f28ce2e674f36ae/fig-2.png)

### 1. MCP Host（宿主应用）
Host 是直接与最终用户交互、并拥有模型决策循环的应用主体。典型例子包括 Claude Desktop、Cursor、或基于 LangGraph 构建的企业自动化应用。
* **核心职责**：管理全局状态、组装给 LLM 的最终提示词、调用模型进行推理，并在模型要求调用外部能力时，发起人工审批（Human-in-the-loop）或安全审计。
* **注意点**：Host 自身通常不直接实现底层系统协议，而是通过创建并管理内部的 MCP Client 实例来与外部交互。

### 2. MCP Client（协议客户端）
Client 运行在 Host 进程内部，是 Host 与特定 MCP Server 通信的标准化连接代理。
* **核心职责**：管理底层的会话生命周期（握手建立、协议版本协商、异常重连），将 Host 的业务请求包装为规范的 JSON-RPC 2.0 消息，并将 Server 返回的响应还原为 Host 可消费的数据对象。
* **连接关系**：通常一个 Client 实例对应一个 Server 实例（1:1 连接）。若 Host 需要接入 3 个独立数据源，它会在内部维护 3 个 Client 实例。

### 3. MCP Server（协议服务端）
Server 是一个轻量级、独立的程序或服务，专门向外暴露特定系统的功能与数据。
* **核心职责**：持有特定系统（如本地文件系统、PostgreSQL、GitLab API）的访问凭证与执行逻辑，遵循 MCP 规范向 Client 声明自身具备的能力，并在收到 `tools/call` 请求时实际执行底层动作。
* **边界限制**：MCP Server **绝不直接向大模型发起对话补全**。它只被动接收 Client 的指令并返回计算或查询结果。

### 端到端调用时序流转

当用户发起提问时，Host、Client、Server 与 LLM 的交互顺序如下：

![MCP 架构下工具发现与执行的时序交互图](/images/content-rebuild/2f28ce2e674f36ae/fig-3.png)

1. **握手与发现**：MCP Client 启动后向 Server 发起能力发现请求（如 `tools/list`）。Server 返回其支持的所有函数声明与参数 Schema。
2. **格式映射与模型输入**：Host 获取这些定义后，按当前接入模型（如 Claude 或 GPT）的原生 API 格式将其转换为对应的 Function Calling 参数，连同用户问题提交给大模型。
3. **模型生成调用决策**：模型根据输入输出结构化 JSON，表明希望调用 `query_order(order_id="ord_1024")`。
4. **协议调用中转**：Host 截获模型的 `tool_calls`，交由对应的 MCP Client 发送 JSON-RPC 2.0 请求：
   ```json
   {
     "jsonrpc": "2.0",
     "method": "tools/call",
     "params": {
       "name": "query_order",
       "arguments": {"order_id": "ord_1024"}
     },
     "id": 1
   }
   ```
5. **Server 执行与回传**：MCP Server 执行真实的数据库查询，将 JSON 结果返回给 Client。
6. **聚合生成回复**：Host 将结果组装为观察值（Observation）再次提交给大模型，模型最终生成自然的回复文本。由于整个过程中 MCP Server 运行在独立进程或远程容器中，即便工具执行崩溃，也不会破坏 Host 的主进程上下文。

---

## 传输机制：标准 stdio 与 Streamable HTTP

为了兼顾本地与远程等不同部署架构，MCP 规范定义了标准传输通道（Transports）。协议本身不内置进程沙箱或系统级权限隔离，而是由具体的传输通道与宿主环境承载通信。

![MCP 传输机制演进：stdio 进程通信与 Streamable HTTP 远程调用](/images/content-rebuild/2f28ce2e674f36ae/fig-4.png)

### 1. 标准 stdio（标准输入/输出）
* **机制**：Host 以子进程方式启动 MCP Server，双方通过操作系统原生的 `stdin` 与 `stdout` 管道按行传递 JSON 文本。
* **适用场景**：本地桌面级集成（如本地代码分析、读取用户本地日志文件、SQLite 数据库操作）。
* **特点**：无需开放网络端口，生命周期与 Host 子进程绑定，规避了网络监听与网络层篡改风险，配置简单；但协议本身不限制子进程的系统访问权限，高危命令仍需宿主提供沙箱防护。

### 2. Streamable HTTP（标准网络通信）
* **机制**：基于现代 HTTP 协议，在规范的统一端点上实现双向数据交互和流式输出。
* **历史演进**：在早期 MCP 方案中，远程通信依赖非对称的双向通道——Client 通过 HTTP GET 建立 Server-Sent Events（SSE）长连接用于接收服务端下行数据，而上行请求则通过独立的 HTTP POST 发送。在最新规范演进中，**Streamable HTTP** 整合了这一交互模式，提供了更符合标准 HTTP 网关、代理与负载均衡器习惯的流式通道。
* **适用场景**：企业级远程微服务、部署在 K8s 集群中的统一工具网关、需要多租户共享访问的外部 API 服务。

---

## MCP 的三大原语：不仅是工具

很多人误将 MCP 等同于“标准化的工具调用框架”。实际上，**Tools 仅仅是 MCP 定义的三大上下文原语之一**：

![MCP 三大核心原语：Tools、Resources 与 Prompts 协同关系](/images/content-rebuild/2f28ce2e674f36ae/fig-5.png)

| 协议原语 | 控制权归属 | 核心概念与用途 | 在我们的运维 Agent 场景中 | 呈现给模型的方式 |
| :--- | :--- | :--- | :--- | :--- |
| **Tools**（工具） | 模型自主决策触发 | 具有动态计算能力或外部副作用的操作 | 执行 `restart_slave_db(node_id="node_2")` 重启从库 | 转换为模型的 `tools` 参数，生成参数 JSON |
| **Resources**（资源） | 宿主或用户按需读取 | 通过 URI 定位的只读数据上下文（类似于文件抽象） | 通过 `db://schema/orders` 读取最新的数据字典与表定义 | 作为静态上下文段落（Context）直接注入 Prompt |
| **Prompts**（提示模板）| 用户引导与主动调用 | 服务端预设的高频交互模版与工作流规范 | 服务端提供的“故障诊断排查标准作业流程（SOP）” | 转换为对话的初始系统提示或多轮指导范式 |

这三大原语的分工非常清晰：
* **Prompts** 帮助宿主标准化工作流的初始结构；
* **Resources** 让宿主可以按需挂载只读的静态与半静态背景信息，而无需把所有说明都当成可执行函数；
* **Tools** 则负责在需要改变外部世界或获取动态变量时执行具体动作。

---

## 工程选型权衡：直连还是上 MCP？

架构设计不存在银弹。引入 MCP 意味着增加了一层 JSON-RPC 序列化、进程通信或网络往返的开销，同时也引入了更多的维护组件。

![工程选型决策路径：直连 Function Calling 与接入 MCP 的权衡](/images/content-rebuild/2f28ce2e674f36ae/fig-6.png)

### 何时推荐原生 Function Calling 直连？
1. **工具少且逻辑专用**：应用内只有 1~3 个简单的专用函数（例如将用户自然语言解析成特定的内部查询参数），且这些代码不需要给任何外部工具复用。
2. **极端延迟敏感**：调用链路上每一毫秒都至关重要，希望省去额外的进程间管道读写或网络 JSON-RPC 包装损耗。
3. **单体自研系统**：整个业务在单一后端代码库内闭环，没有跨桌面客户端（如 Cursor、Claude Desktop）挂载调试的需求。

### 何时必须或强烈建议采用 MCP？
1. **跨客户端与跨平台生态复用**：同一个工具集（例如企业的 Git 仓库操作、内部知识库检索、云资源启停）需要同时服务于工程师的 IDE、业务部门的聊天客户端以及后端的自动化 Agent 流。
2. **团队职责与权限隔离**：基础设施或数据工程团队独立维护 MCP Server 并管控内网数据库账号密码；AI 智能体开发团队只持有 MCP Client 协议接口，无需接触底层凭据，实现权限与代码解耦。
3. **模块化沙箱防护**：执行某些具有潜在高风险的脚本（如 Bash 命令行执行）时，需要将其放入隔离的 Docker 容器中作为独立 MCP Server 运行，防止宿主主程序受到威胁。

---

## 学习检验与实战练习

### 核心概念自检
1. **模型会执行代码吗？** 不会。Function Calling 只是大模型输出结构化参数的生成过程。
2. **MCP 会替代大模型的推理吗？** 不会。MCP 只是 Host、Client 与 Server 之间传递声明与执行结果的协议通道，模型依然负责意图识别与参数生成。
3. **MCP Server 可以直接调用 OpenAI API 生成回复吗？** 规范中 MCP Server 只向 Client 提供上下文与工具响应，不管理大模型对话主循环，对话主循环由 Host 负责驱动。

### 动手实验：验证一个最小 JSON-RPC 2.0 工具响应
**目标**：理解 MCP 底层的数据封包格式，验证协议的独立性。

假设你的 MCP Server 接收到了客户端的 `tools/call` 请求：
```json
{
  "jsonrpc": "2.0",
  "id": "req_001",
  "method": "tools/call",
  "params": {
    "name": "query_order",
    "arguments": {"order_id": "ord_1024"}
  }
}
```

**检验任务**：
编写一段 Python 代码，根据规范要求返回标准响应。如果数据库返回结果 `{"status": "PAID", "amount": 99.0}`，标准 MCP 响应体格式应该包含哪些字段？

```python
# 验收参考实现
import json

def handle_mcp_call(request_json_str: str) -> str:
    request = json.loads(request_json_str)
    req_id = request.get("id")
    params = request.get("params", {})
    tool_name = params.get("name")
    args = params.get("arguments", {})
    
    if tool_name == "query_order":
        order_id = args.get("order_id")
        # 模拟业务查询
        data = {"order_id": order_id, "status": "PAID", "amount": 99.0}
        
        # 符合 MCP 规范的工具返回必须包含 content 数组（通常为 text 类型）
        response = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(data, ensure_ascii=False)
                    }
                ],
                "isError": False
            }
        }
        return json.dumps(response, ensure_ascii=False)
    
    raise ValueError("Tool not found")

# 验证运行
print(handle_mcp_call('{"jsonrpc": "2.0", "id": "req_001", "method": "tools/call", "params": {"name": "query_order", "arguments": {"order_id": "ord_1024"}}}'))
```
