---
title: Agent 的核心运行流程是什么？
icon: mdi:sync
tag:
- Agent面试
- Agent流程
- 规划
description: Agent 核心是宿主程序控制的有状态循环（感知、决策、执行、反馈）。模型只做下一步决策，宿主负责调用工具、维护状态并把控安全终止。
---

在面试中被问到“Agent 的运行机制”时，很多人容易背诵 ReAct 论文里的“思考-行动-观察（Thought-Action-Observation）”，或者以为大语言模型（LLM）自己在后台默默循环。但在工程实现中，**模型本身只是无状态的文本/Token 预测函数，真正的循环引擎是外部宿主程序（Host Runtime）**。

宿主程序负责读取外部状态、调用大模型、解析返回的动作、实际执行工具调用，再把结果灌回上下文。这一过程循环往复，直到达成目标或被安全拦截。

![Agent 核心运行循环机制概览](/images/content-rebuild/3dc667dfa99a6744/cover.png)

---

## 面试标准口述：感知、决策、执行与反馈

面试时可以用 30 秒给出逻辑清晰的口述回答：

> “Agent 的底层流程是一个由**宿主程序驱动的有状态控制循环**，每轮循环包含四个核心步骤：
>
> 1. **感知（Perception）**：宿主程序组装当前状态，包括用户目标、历史对话、环境变量和上一步工具的执行结果。
> 2. **决策（Decision）**：宿主将组装好的上下文发送给大模型。模型根据上下文推理，决定是直接输出最终答复，还是发起结构化的工具调用请求（Tool Calling）。
> 3. **执行（Execution）**：如果模型决定调用工具，宿主程序负责拦截参数、鉴权并真正调用外部 API、数据库或代码沙箱获取物理副作用。
> 4. **反馈（Feedback）**：宿主程序捕获工具的返回值或报错，格式化为新的上下文消息写回会话历史，驱动进入下一轮决策。
>
> 这个循环会持续运转，直到**业务达成返回最终答复**，或者触发**步数用尽、Token 耗尽、超时与安全拦截**等硬性退出条件。”

---

## 贯穿案例：一个线上日志排查任务

为了看清真实的数据流，我们设定一个贯穿全篇的具体场景：

> **用户指令**：“线上集群报 500 错误，帮我查看最新错误日志，并排查是哪个微服务出的问题。”

宿主程序向大模型注册了两个工具：
1. `fetch_cluster_logs(service_name: str, limit: int)`：拉取指定服务的日志。
2. `list_services()`：列出集群所有运行中的服务名称。

数据在四阶段中如何流转：

![Agent 核心循环的状态机抽象与数据流向](/images/content-rebuild/3dc667dfa99a6744/fig-1.png)

1. **感知（组装上下文）**：
   宿主将系统提示词（包含工具声明）、用户指令“线上集群报 500 错误……”以及初始空历史拼接成 Prompt。
2. **决策（模型输出）**：
   模型发现自己不知道集群有哪些服务，于是返回一段带有工具调用的结构化数据：`tool_calls: [list_services()]`。
3. **执行（宿主执行物理操作）**：
   模型本身不连服务器，宿主读取到 `list_services()` 请求，发起真实网络请求，查到结果为 `["auth-service", "order-service", "payment-service"]`。
4. **反馈（状态回写）**：
   宿主将服务列表封装成一条 `role: "tool"` 的消息，追加到历史对话末尾。此时状态更新完毕，控制流回到“决策”阶段。
5. **第二轮决策与执行**：
   模型看到服务列表后，再次发起 `fetch_cluster_logs(service_name="order-service", limit=50)`。宿主执行并返回错误堆栈 `DB Connection Timeout`。
6. **第三轮决策（终止退出）**：
   模型拿到日志发现根本原因是订单服务数据库超时，不再请求新工具，直接输出自然语言分析：“排查完成，是 order-service 的数据库连接超时导致了 500 报错……”。宿主检测到没有后续工具调用，流程正常结束。

---

## 最小宿主循环实现

以下是一个用原生 Python 实现的宿主控制循环，直观展示了宿主程序如何代理物理调用并更新会话状态：

```python
import json
from typing import Any, Callable, Dict, List

class AgentHostLoop:
    def __init__(
        self,
        model_client: Any,
        tools: Dict[str, Callable],
        tool_schemas: List[Dict],
        max_steps: int = 10
    ):
        self.client = model_client
        self.tools = tools
        self.tool_schemas = tool_schemas
        self.max_steps = max_steps

    def run(self, user_query: str) -> str:
        # 1. 感知阶段：初始状态
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "你是一个排查运维问题的助手。请合理使用工具解决问题。"},
            {"role": "user", "content": user_query}
        ]
        
        step = 0
        while step < self.max_steps:
            step += 1
            
            # 2. 决策阶段：模型根据当前上下文产出决策
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tool_schemas
            )
            choice = response.choices[0].message
            # 把模型的输出（包含思考内容或 tool_calls 意图）存入上下文
            messages.append(choice)
            
            # 正常终止：模型未发起工具调用，说明给出了最终答复
            if not choice.tool_calls:
                return choice.content or ""
            
            # 3. 执行阶段：宿主程序代替模型执行外部真实代码
            for tool_call in choice.tool_calls:
                func_name = tool_call.function.name
                call_id = tool_call.id
                
                if func_name not in self.tools:
                    result_str = f"Error: Tool '{func_name}' 不存在。"
                else:
                    try:
                        args = json.loads(tool_call.function.arguments)
                        result_str = str(self.tools[func_name](**args))
                    except Exception as e:
                        # 捕获异常作为观察反馈，不直接崩溃退出
                        result_str = f"执行工具 {func_name} 报错: {str(e)}"
                
                # 4. 反馈阶段：将执行结果构造成 tool 消息写回上下文
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": result_str
                })
        
        # 超出最大步数限制的保护退出
        return "任务未完成：已达最大迭代步骤上限。"
```

更深入的上下文装配与状态管理机制，可以参考 [Agent 上下文管理机制](/interview/agent-interview/context-management.md)。

---

## 循环的三类终止出口

控制循环最怕“停不下来”。在实际工程中，循环必须具备确定性的出口通道：

![三类运行循环终止条件判断树](/images/content-rebuild/3dc667dfa99a6744/fig-2.png)

### 1. 业务正常终止
* **自然语言返回**：模型在最后一步不再产出 `tool_calls`，只返回纯文本内容，宿主判定任务结束并向用户交付答复。
* **显式结束标记（Stop Function）**：在部分受限场景或代码生成场景中，为了防止模型喋喋不休，会要求模型显式调用 `finish(answer=...)`。一旦宿主捕获到该终止工具，立即结束循环。

### 2. 资源耗尽与系统硬约束
* **最大步数限制（Max Steps）**：宿主配置硬性步数计数器（例如常规任务设为 10～20 步）。超过步数仍未得出结论，强制截断并告警。
* **上下文长度告警（Context Limits）**：随着工具返回内容的堆积，历史消息 Token 总量可能逼近模型的上下文窗口上限。此时若未做滑动窗口或摘要裁剪，宿主需主动中断，避免底层 API 报 `400 Bad Request`。
* **绝对超时时间（Wall-clock Timeout）**：工具请求（如等待远程接口或复杂 SQL）可能阻塞，宿主必须用异步超时（例如 `asyncio.wait_for`）进行外层兜底。

### 3. 运行时异常与主动熔断
* **安全护栏拦截**：输出合规检查或越权审查模块发现异常输入/输出，宿主安全网关强制切断执行。
* **不可恢复的底层错误**：当工具返回不可重试的致命状态（如账号欠费、HTTP 401 权限失效），重试无意义，宿主应主动抛出错误中断循环。关于容错策略，可参考 [Agent 错误恢复与重试](/interview/agent-interview/agent-error-handling.md)。

<ReadMoreLock />

---

## 生产中的死循环与振荡防护

在真实环境中，模型可能会因为反馈不够直观或概率采样波动而陷入死循环。宿主程序必须具备拦截与干预能力。

![死循环检测与宿主干预机制示意](/images/content-rebuild/3dc667dfa99a6744/fig-3.png)

### 常见死循环类型
1. **参数原地打转（Tool Stuttering）**：模型调用工具 `get_user(id=10)` 失败，下一轮没有换参数，继续调用 `get_user(id=10)`，连续多次完全相同。
2. **多工具交替振荡（A-B-A-B Oscillation）**：模型在两个工具之间来回摇摆。比如先调用搜索工具，觉得信息太长；再去调用精简提取工具，发现缺少字段；又回过头去搜索，陷入循环。

### 宿主介入机制

宿主程序可以通过维护**动作签名队列（Action History）**来主动打破僵局：

1. **计算动作哈希**：宿主将每次模型请求的“工具名 + 规范化排序后的参数 JSON”做 SHA-256 签名。
2. **重复频次监控**：若检测到完全相同的签名在最近几步中重复出现（且前次未产生新有效数据），宿主**短路拦截真实调用**。
3. **反思注入（Intervention Prompt）**：宿主直接构造一条包含警告提示的 Tool 消息（携带对应 `tool_call_id`）回填入上下文：“*系统提示：检测到连续使用相同参数发起调用且未产生新进展，本次物理执行已被拦截。请分析原因并更换参数、尝试其他工具，或向用户说明无法完成。*”
4. **熔断退出**：如果注入提示词后模型依然重复无效动作，宿主强制退出循环。

```python
# 签名检测示意
import hashlib

def get_action_fingerprint(tool_name: str, args_dict: dict) -> str:
    # 规范化 key 排序以避免 JSON 字符串序列化差异
    canonical_json = json.dumps(args_dict, sort_keys=True)
    return hashlib.sha256(f"{tool_name}:{canonical_json}".encode()).hexdigest()
```

### 写操作的幂等性保证
特别注意：**重试与死循环干预依赖程序层面的幂等性设计，而不是提示词**。对于转账、发邮件、修改数据库等具有外部副作用的写操作，调用参数必须附带唯一的**幂等键（Idempotency Key）**。即使 Agent 控制流重复执行了该工具，底层服务也能通过幂等键识别为重复请求并予以去重。

---

## 执行时序推演：正常循环 vs 故障拦截

通过时序对比，可以清晰看到健康循环与受控异常中断的区别：

![健康循环与故障循环的时序对比](/images/content-rebuild/3dc667dfa99a6744/fig-4.png)

### 场景对比

* **正常自愈循环**：
  * 步骤 1：模型尝试调用 `query_order(date="today")`。
  * 宿主返回：`[]`（空数据，提示可能未结算）。
  * 步骤 2：模型自主改变策略，调用 `query_order(date="yesterday")`，成功拿到订单数据。
  * 步骤 3：模型整理数据，输出最终结论，循环平稳结束。
* **故障拦截循环**：
  * 步骤 1：模型调用 `delete_server(id="srv-01")`，工具抛出权限错误 `HTTP 403 Forbidden`。
  * 步骤 2：模型未调整逻辑，再次发起相同的 `delete_server(id="srv-01")` 请求。
  * 宿主拦截：检测到重复调用且前序已报错，宿主拦截网络请求，注入合成警告信息：“权限不足，继续重试将无进展”。
  * 步骤 3：模型接收警告，放弃无效调用，返回用户：“由于权限不足（HTTP 403），无法删除服务器”。流程安全截断。

---

## 面试追问深度剖析

### 追问一：如何区分正常的长耗时任务与死循环？

面试官常问：*“如果一个编程 Agent 跑了 15 步还没停，你怎么判断它是正在复杂推演，还是卡在死循环里了？”*

单纯看“步数”不能解决问题。步数设太小会长任务夭折，设太大又无法抑制 Token 浪费。正确做法是结合**外部状态变化**与**信息增益**：

![长耗时探索与停滞死循环的指标对比](/images/content-rebuild/3dc667dfa99a6744/fig-5.png)

| 维度 | 卡死的死循环（Deadlock） | 正常的复杂长探索（Valid Exploration） |
| :--- | :--- | :--- |
| **工具参数多样性** | 参数固定或在 2 个状态间往复震荡 | 参数呈单向演进（如搜索词逐步精准、文件读取行号下移） |
| **环境状态变化** | 外部文件、数据库记录等物理状态完全未发生改变 | 单元测试通过率提升、生成了新中间文件、代码行数改变 |
| **返回信息差异率** | 工具返回的报错堆栈或输出内容字节哈希完全一致 | 每次工具调用带回了全新的实体、代码段或环境变量 |

**工程实现做法**：宿主内部维护一个“**连续无进展计数器（No-Progress Counter）**”。只要某一步带来了新的状态变更或增量信息，计数器清零；如果连续 3 到 5 步信息增益为零（例如连续报同样的错误），则判定为停滞，立即熔断。

### 追问二：Agent 与普通工作流（Workflow）的根本区别是什么？

根据 Anthropic 官方文章《Building effective AI agents》的界定：
* **工作流（Workflows）**：由**预先编写的代码路径**编排 LLM 与工具。流转逻辑（如提示词链、分类路由、并行聚合或带条件的优化循环）由外部代码定义，LLM 负责完成局部处理、评估或分类，但不自主决定整体拓扑与流向。需要注意，代码编排分支不等于运行结果绝对确定，模型输出与外部系统状态仍会波动。
* **智能体（Agents）**：由 **LLM 动态驱动自身的执行流**。模型根据上下文与环境反馈，在运行期自主决定调用哪些工具、何时转向、何时终止任务，宿主程序负责提供运行环境与边界约束。

在真实业务中，通常不会走极端，而是使用像 [LangGraph 架构](/interview/agent-interview/agent-frameworks.md) 这样的框架，将手写规则或可预测的工程步骤与局部的 Agent 自主决策步骤混合编排，兼顾可控性与灵活性。

---

## 常见面试误区辨析

![宿主运行环境与模型计算职责划分](/images/content-rebuild/3dc667dfa99a6744/fig-6.png)

* **误区一：“Agent 循环是由大模型自己驱动的”**
  * *澄清*：模型本身是一组无状态的矩阵运算，输入一组 Token，返回下一个 Token 的概率分布。模型既没有系统定时器，也没有网络套接字。循环驱动、环境状态保持、网络 I/O 调度全部由外部**宿主程序**实现。
* **误区二：“ReAct 模式就是 Agent 的全部”**
  * *澄清*：ReAct 只是早期通过纯文本 Prompt 引导模型在一段文字中混合输出“Thought”与“Action”的技巧。现代 Agent 普遍使用标准协议层的原生 [Tool Calling 机制](/interview/agent-interview/tool-calling.md)（如 OpenAI 协议或 [MCP 服务](/interview/agent-interview/mcp.md)）。循环是控制论层面的抽象，不依赖于 ReAct 这一特定提示工程形式。
* **误区三：“反思（Reflection）只是上下文压缩”**
  * *澄清*：上下文压缩是为了应对长对话超出模型输入上限而做的工程裁剪；反思则是让模型（或专门的评审节点）针对前序失败轨迹进行评判并给出修正建议的认知策略。两者的目的与实现方案完全不同。

掌握宿主控制循环的执行流程、边界处理与死循环拦截，是设计高可靠、可上线业务 Agent 的关键基础。进阶规划与工具接入可参阅 [Agent 任务规划机制](/interview/agent-interview/agent-planning.md) 与 [工具调用兼容性设计](/interview/agent-interview/tool-schema-compatibility.md)。
