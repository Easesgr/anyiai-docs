---
title: Tool Calling 是什么？大模型如何调用工具？
icon: mdi:cpu-64-bit
date: 2026-07-16
tag:
- 大模型100讲
- Agent
description: 大模型无法直接联网或运行代码。Tool Calling 是模型在提示驱动下输出结构化参数，由外部宿主程序代为执行并将结果回传的协作机制。
---

大语言模型本身既不能直接连网，也不能读写你的本地磁盘。如果在对话中问它“北京现在的室外气温是多少”，或者“用户 usr_10023 的账户余额够不够支付 299 元打折后的价格”，模型在不借助外部系统的情况下，只能基于训练期权重生成概率推测，甚至编造虚假数字。

在上一讲[/llm-100/agent-basics/Agent_的核心组成_模型_记忆_工具和规划.md](/llm-100/agent-basics/Agent_的核心组成_模型_记忆_工具和规划.md)中，我们梳理了智能体由模型、记忆、工具和规划四部分组成。本讲我们将聚焦其中最核心的操作环节——工具调用（Tool Calling / Function Calling）。

![工具调用全流程架构图](/images/content-rebuild/dd919fc17a17dae3/cover.png)

先给出一个核心结论：**大模型在工具调用中只负责生成格式化的文本符号，真正的代码运行、网络请求和数据库读写，全部由包裹模型的外部应用程序（称为宿主程序）代为执行。**

下面我们通过一个贯穿全程的例子来看这一过程是如何跑通的：
> 用户提问：“帮我查一下用户 usr_10023 现有的账户余额，顺便算一下原价 299 元打八五折后的价格。”

---

### 运行边界：模型只生成文本，执行由宿主完成

关于工具调用，最容易产生的一个误解是：模型像操作系统内核一样，自己开辟了线程、创建了 TCP Socket、给第三方接口发送了请求。

从实际运行机制看，大模型只是一个基于自回归计算下一个 Token 条件概率的张量计算程序：

$$P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

模型的输入是一串文本 Token，输出也是一串文本 Token。在运行模型的 GPU 或服务器内存之外，根本不存在直连外部网络或数据库的硬件线路。

![模型与执行器物理隔离图](/images/content-rebuild/dd919fc17a17dae3/fig-1.png)

整个调用过程可以分为两个明确的角色：
1. **大语言模型（LLM）**：分析用户意图，判断当前问题需要调用哪个工具，并按照约定格式提取并组织参数字符串。
2. **宿主环境（Host Application）**：开发者编写的外部 Python/Node.js 等应用程序。宿主负责向模型提供可用工具清单、截获模型输出的调用请求、执行具体的本地代码或外部 API，最后把运行结果转化成文本格式送回模型。

---

### 契约定义：如何告诉模型有哪些工具可用

宿主程序必须在向大模型发起请求时，显式传入一份“工具说明书”。目前主流 API（如 OpenAI、Anthropic、DeepSeek 等）普遍采用基于 JSON Schema 的规范。

针对我们前面的例子，宿主程序会向模型注册两个工具声明：

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_user_balance",
      "description": "通过用户ID查询账户当前的可用现金余额",
      "parameters": {
        "type": "object",
        "properties": {
          "user_id": {
            "type": "string",
            "description": "系统的唯一用户标识符，例如 usr_10023"
          }
        },
        "required": ["user_id"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "calculate_discount",
      "description": "计算商品的折后价格",
      "parameters": {
        "type": "object",
        "properties": {
          "price": {
            "type": "number",
            "description": "商品原价"
          },
          "discount_rate": {
            "type": "number",
            "description": "折扣率，取值范围在 0 到 1 之间，例如 0.85 代表八五折"
          }
        },
        "required": ["price", "discount_rate"]
      }
    }
  }
]
```

在底层实现中，API 网关或推理框架会把这段 JSON 结构注入到当前提示词中，或者拼接到专门的系统字段内。

![工具声明转换示意图](/images/content-rebuild/dd919fc17a17dae3/fig-2.png)

模型判断是否调用工具，高度依赖工具名称、参数定义以及 `description` 里的自然语言说明。如果描述含糊不清或与实际场景脱节，模型就容易在解码阶段产生偏差，出现“该调不调”或“选错工具”的现象。

<ReadMoreLock />

---

### 意图判断与生成：模型如何决定调用工具

当带上了工具声明的上下文进入自回归解码时，模型会根据当前语义上下文进行多选一：是直接生成普通的回复语句，还是生成一条结构化的调用指令？

支持 Tool Calling 的模型在后训练（Post-training / SFT）阶段，学习了大量带有特殊标记（Special Tokens）的结构化数据。当模型认为需要调用工具时，其生成的输出不再是面向人类的自然语言，而是一个包含参数的载荷结构：

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_user_001",
      "type": "function",
      "function": {
        "name": "get_user_balance",
        "arguments": "{\"user_id\": \"usr_10023\"}"
      }
    }
  ]
}
```

这里需要注意一个经常被忽略的工程细节：**模型输出的 `arguments` 在传输协议中只是一个被转义的字符串（String），并不是已经完成反序列化的原生对象**。

![模型决策与生成时序图](/images/content-rebuild/dd919fc17a17dae3/fig-3.png)

当模型生成完调用指令后，推理服务返回的终止状态（`finish_reason`）会被标记为 `tool_calls`。此时，生成过程暂时中止，控制权交还给外部宿主程序。

---

### 宿主拦截与执行：反序列化、校验与权限拦截

宿主程序检测到 `finish_reason == "tool_calls"` 后，绝对不能直接无脑把模型参数送入执行引擎，必须经过三道防线：

1. **语法反序列化**：把 `arguments` 字符串解析成字典对象。即便模型正常结束生成，输出的文本也可能包含未转义字符或格式残缺，直接反序列化会导致解析库抛出语法错误（若因达到 Token 上限中途截断，服务端通常会返回 `finish_reason == "length"`，宿主应先行拦截）。
2. **契约校验**：使用类型检查工具（如 Python 的 Pydantic）校验字段类型和取值范围。比如 `discount_rate` 是否在 `0` 到 `1` 之间，是否缺少了必填字段。
3. **权限与安全拦截**：防范提示词注入（Prompt Injection）导致的危险操作。提示词注入并不等同于系统越狱，但攻击者可能诱导模型在参数里填入诸如 `path: "../../etc/passwd"` 或越权的 `user_id`。这类访问控制必须在宿主代码中通过白名单和沙箱硬性拦截，不能寄希望于模型自觉遵守规则。

![宿主拦截与校验流程图](/images/content-rebuild/dd919fc17a17dae3/fig-4.png)

校验通过后，宿主程序根据 `name` 字段找到真实的本地函数或外部接口并发起执行。

---

### 回传与多调用对齐：tool_call_id 的作用

宿主环境运行完代码后，拿到执行结果（比如查询到账户余额为 500 元），必须把它包装成约定格式的消息追加到对话历史中。

在通用规范中，回传消息具有以下固定属性：
- `role`: `"tool"`
- `tool_call_id`: 必须与前面模型生成的 `id`（例如 `call_user_001`）严格匹配。
- `content`: 工具执行结果的字符串表示（如 `'{"balance": 500.0, "currency": "CNY"}'`）。

**为什么必须携带 `tool_call_id`？**
模型支持并行调用多个工具（Parallel Tool Calling）。例如模型在同一轮中同时发起了查询余额和计算折扣两个请求：

$$\text{call\_1: get\_user\_balance}, \quad \text{call\_2: calculate\_discount}$$

宿主并发发起请求时，外部接口返回的顺序往往是不确定的。模型正是依赖消息中的 `tool_call_id`，将返回的数据与之前的每个请求精准关联起来。

![上下文回传拼接闭环图](/images/content-rebuild/dd919fc17a17dae3/fig-5.png)

当宿主把所有的工具结果拼回消息历史后，会再次向大模型发起第二次推理请求。此时模型的上下文里同时包含了“原始问题 + 模型的调用指令 + 工具的真实输出”。模型阅读到这组事实后，才能组织出最终回答：“用户 usr_10023 当前余额为 500 元；原价 299 元打八五折后为 254.15 元，余额充足，可以完成支付。”

---

### 失败处理：工具报错时的应对策略

在实际生产中，网络超时、参数不合法、接口崩溃是常见现象。系统不能因为单次工具报错就直接崩溃。

工具执行阶段的常见故障可以分为两类：

| 故障分类 | 常见原因 | 宿主处理方式 |
| :--- | :--- | :--- |
| **模型生成端错误** | 输出的 JSON 缺少闭合括号、漏掉必填参数、参数类型错误（如把数字传成了字符串）。 | **错误回传重试**：将解析器给出的具体校验报错包装为 `role: tool` 的内容回传给模型。模型看到报错后，通常会在下一轮重新生成正确的参数。 |
| **外部系统执行错误** | 接口超时、第三方数据库连接失败、目标资源不存在。 | **降级与兜底**：向模型回传错误状态，让模型向用户解释当前某项服务暂不可用；或者由宿主程序触发备用接口。 |

![异常重试与自愈时序图](/images/content-rebuild/dd919fc17a17dae3/fig-6.png)

需要说明的是，**模型重试并不保证 100% 成功纠错**。在极少数情况下，模型可能会反复输出错误的参数。因此，宿主程序必须在代码层面设立最大重试次数（例如最多重试 2 次），超过阈值时必须执行中断或熔断逻辑，避免死循环消耗大量 Token。

---

### 最小可运行实验：手动实现一个工具调用循环

下面我们用一段纯原生 Python 代码模拟整个调用生命周期，无需安装 LangChain 或复杂 Agent 框架：

```python
import json

# 1. 宿主环境中的真实业务函数
def calculate_discount(price: float, discount_rate: float) -> str:
    if not (0 < discount_rate <= 1):
        return json.dumps({"error": "折扣率必须在 (0, 1] 之间"})
    final_price = round(price * discount_rate, 2)
    return json.dumps({"final_price": final_price})

# 2. 模拟大模型在第 1 轮返回的调用指令（模拟实际 API 返回）
mock_model_response = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_discount_99",
            "type": "function",
            "function": {
                "name": "calculate_discount",
                "arguments": '{"price": 299.0, "discount_rate": 0.85}'
            }
        }
    ]
}

# 3. 宿主对话上下文容器
messages = [
    {"role": "user", "content": "原价 299 元打八五折是多少钱？"},
    mock_model_response
]

# 4. 宿主执行器：拦截、反序列化、校验并分发执行
def host_dispatch(tool_call: dict) -> dict:
    call_id = tool_call["id"]
    func_name = tool_call["function"]["name"]
    raw_args = tool_call["function"]["arguments"]
    
    # 语法反序列化
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError as e:
        return {"tool_call_id": call_id, "content": json.dumps({"error": f"JSON格式错误: {str(e)}"})}
    
    # 校验并执行
    if func_name == "calculate_discount":
        try:
            price = float(args["price"])
            discount_rate = float(args["discount_rate"])
            result_str = calculate_discount(price, discount_rate)
            return {"tool_call_id": call_id, "content": result_str}
        except (KeyError, ValueError, TypeError) as e:
            return {"tool_call_id": call_id, "content": json.dumps({"error": f"参数错误: {str(e)}"})}
    
    return {"tool_call_id": call_id, "content": json.dumps({"error": "未知工具"})}

# 5. 宿主执行工具并将结果组装为 role: tool 消息追加回上下文
call_item = mock_model_response["tool_calls"][0]
execution_result = host_dispatch(call_item)

messages.append({
    "role": "tool",
    "tool_call_id": execution_result["tool_call_id"],
    "content": execution_result["content"]
})

# 查看组装完成的上下文，此时即可发起第 2 次模型 API 调用
print(json.dumps(messages, ensure_ascii=False, indent=2))
```

---

### 动手练习

为了检验你对宿主拦截与错误处理的理解，请完成以下练习：

1. 修改上述代码中的 `arguments` 为不合法的数据：`'{"price": 299.0, "discount_rate": 1.5}'`。
2. 观察 `host_dispatch` 产生的输出。宿主程序回传了什么格式的错误信息？
3. 思考并在代码中增加一个计数器：如果连续收到两次带有 `error` 字段的工具回传，宿主程序应当如何直接给出兜底文案，而不再将错误送给模型陷入死循环？

---

### 下一步

掌握了单次 Tool Calling 之后，你会发现很多真实系统面临两个扩展问题：
1. 如果工具运行在远程服务器或外部独立进程中，如何用标准化协议连接它们？请看下一讲[/llm-100/agent-basics/Function_Calling_和_MCP_有什么区别.md](/llm-100/agent-basics/Function_Calling_和_MCP_有什么区别.md)。
2. 如果一个复杂任务需要“思考一次、查一次、再思考、再查”，如何组织多步连续的工具调用？请继续学习[/llm-100/agent-basics/ReAct_框架是什么_为什么适合_Agent.md](/llm-100/agent-basics/ReAct_框架是什么_为什么适合_Agent.md)。
