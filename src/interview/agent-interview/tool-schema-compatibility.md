---
title: 如何处理不同 LLM Provider 的 Tool Schema 差异？
icon: mdi:puzzle-star
tag:
- Tool Schema
- LLM Provider
- 适配
description: 梳理 OpenAI、Claude、Gemini 在工具定义、响应解析与结果回传上的协议差异，讲解基于中间抽象层的双向适配与边缘降级方案。
---

在开发支持多模型切换的 Agent 系统时，工程团队常遇到一个现实问题：业务侧只定义了一套工具逻辑（例如查询天气或执行 SQL），但接入的模型可能来自 OpenAI、Anthropic Claude、Google Gemini，或者通过 vLLM、Ollama 部署的开源模型。这些平台在工具定义（Schema 格式）、模型响应（调用指令组织形式）以及结果回传（执行结果提交方式）上，协议规范各不相同。

理解这些差异并设计统一的适配层，是让同一套业务工具在不同模型间平滑迁移的基础。

![异构模型工具定义与调用流程适配全景](/images/content-rebuild/2df9306e7fed893c/cover.png)

---

## 面试高频口述回答

如果面试官询问：“**不同的 LLM Provider 对 Tool Schema 的支持不完全一致，你会怎么处理这种差异？**”可以按以下要点口述：

> 处理异构 Provider 的 Tool Schema 差异，核心是**建立规范化的内部中间表示（Canonical Schema），在通信边界做双向映射与兼容清洗**：
>
> 1. **统一输入定义**：业务层使用标准的 JSON Schema（通常基于 Pydantic 导出）统一声明工具名、功能描述和入参。通过 Provider 适配器进行下行转换：OpenAI 需要套一层 `{"type": "function", "function": {...}}`；Anthropic 使用根部平铺的 `input_schema`；Gemini 则必须包装进 `function_declarations` 数组，且仅支持严格的类型子集。
> 2. **统一响应解析**：不同模型吐出的工具调用格式不同。OpenAI 返回未反序列化的 JSON 字符串（`arguments`）；Anthropic 返回已解析的字典（`input`）；Gemini 返回 `args` 字典。适配器需要统一反序列化并提取为标准的 `ToolCall(id, name, args)` 对象给上层业务执行。
> 3. **统一结果回传**：执行工具后的报文结构也有差异。OpenAI 要求以 `role="tool"` 并带上 `tool_call_id` 回传；Anthropic 要求在 `role="user"` 的 `content` 中使用 `type="tool_result"` 块回传；Gemini 则在 `role="function"` 中使用 `functionResponse`。
> 4. **边缘清洗与降级容错**：针对 Gemini 等对空 `required: []` 报错的端点做 Schema 字段修剪；面对不支持 `$ref` 或复杂联合类型的端点做 Schema 扁平化；对完全没有原生工具调用能力的轻量开源模型，降级为在 System Prompt 中注入格式要求，并通过正则提取 JSON 参数。

---

## 贯穿示例：天气查询工具的定义差异

我们以一个统一的示例展开：定义一个工具 `get_weather`，接收两个参数：必填的城市名称 `city`（字符串）和可选的温度单位 `unit`（枚举值 `celsius` 或 `fahrenheit`）。

大模型本身不直接执行代码，它接收的是用文本描述的工具规范，推导出调用意图后生成对应的结构化参数。不同厂商在 API 协议层设计的封装外壳存在明显差异。

![主流 Provider 工具定义顶层结构对比](/images/content-rebuild/2df9306e7fed893c/fig-1.png)

### 1. OpenAI 规范（Chat Completions API）

OpenAI 规范引入较早，采用嵌套结构。外层通过 `type: "function"` 声明类型，具体元数据挂在 `function` 字段下：

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "城市名称"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度单位"}
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

### 2. Anthropic 规范（Messages API）

Anthropic 去掉了冗余的 `function` 嵌套层级，工具名和描述平铺在根对象上，参数模式字段命名为 `input_schema`：

```json
{
  "tools": [
    {
      "name": "get_weather",
      "description": "获取指定城市的天气信息",
      "input_schema": {
        "type": "object",
        "properties": {
          "city": {"type": "string", "description": "城市名称"},
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度单位"}
        },
        "required": ["city"]
      }
    }
  ]
}
```

### 3. Google Gemini 规范（generateContent API）

Gemini 的请求报文中，`tools` 列表内包含一个或多个 `function_declarations` 对象。参数类型在底层映射到 Protocol Buffers 枚举，因此通常要求类型大写（如 `STRING`、`OBJECT`）：

```json
{
  "tools": [
    {
      "function_declarations": [
        {
          "name": "get_weather",
          "description": "获取指定城市的天气信息",
          "parameters": {
            "type": "OBJECT",
            "properties": {
              "city": {"type": "STRING", "description": "城市名称"},
              "unit": {"type": "STRING", "enum": ["celsius", "fahrenheit"], "description": "温度单位"}
            },
            "required": ["city"]
          }
        }
      ]
    }
  ]
}
```

### 核心差异对比表

| 维度 | OpenAI API | Anthropic API | Google Gemini API |
| :--- | :--- | :--- | :--- |
| **工具列表容器** | `tools: list` | `tools: list` | `tools: [ { function_declarations: list } ]` |
| **类型标识字段** | 必须包含 `type: "function"` | 无顶层 `type` | 声明于 `function_declarations` 内 |
| **参数 Schema 键名** | `parameters` | `input_schema` | `parameters` |
| **类型命名习惯** | 小写字符串（`"string"`、`"object"`） | 小写字符串（`"string"`、`"object"`） | 推荐或兼容大写枚举（`"STRING"`、`"OBJECT"`） |
| **高级关键字兼容度** | 支持大部分 JSON Schema Draft 7 关键字 | 支持 Draft 7（不支持 `$ref` 跨节点引用） | 仅支持受限 OpenAPI 3.0 子集，不支持 `$ref`、`pattern` 等 |

<ReadMoreLock />

---

## 响应解析与结果回传的协议差异

当模型判定需要调用 `get_weather` 时，各家返回的报文结构以及后续接收执行结果的方式各不相同。如果业务代码直接绑定某种特定格式，切换模型时整套循环逻辑都会中断。

![模型响应解析与工具结果回传时序链路](/images/content-rebuild/2df9306e7fed893c/fig-2.png)

### 1. 模型输出的解析差异

- **OpenAI**：
  返回内容挂在 `message.tool_calls` 列表中。每个调用项包含全局唯一字符串 `id`（例如 `call_123`）、`type: "function"`，以及包含 `name` 和 `arguments` 的字典。  
  **重点差异**：`arguments` 是一个未反序列化的 **JSON 字符串**（例如 `"{\"city\": \"Beijing\"}"`）。宿主程序必须执行 `json.loads` 才能拿到具体参数。若模型中途被截断，该字符串可能是非法的。
- **Anthropic**：
  工具调用与文本响应作为 `content` 数组中的内容块（Block）并列返回。工具调用的块类型为 `type: "tool_use"`，包含 `id`（例如 `toolu_456`）、`name` 以及 `input`。  
  **重点差异**：`input` 字段在返回时已经是**解析好的结构化字典**，无需手动再次执行 JSON 反序列化。
- **Gemini**：
  调用项挂在 `candidates[0].content.parts` 中，字段名为 `functionCall`，包含 `name` 与 `args` 字典对象。早期的部分端点中不自带显式的唯一 `call_id`，多轮并发调用时需要宿主自行维护关联映射。

### 2. 执行结果的回传差异

宿主程序执行完本地函数（如查询天气得到 `{"temperature": 22, "condition": "sunny"}`）后，必须按特定消息格式将结果喂回模型上下文，才能让模型生成后续回复（参考 [Agent 运行循环](/interview/agent-interview/agent-loop.md)）。

- **OpenAI 回传格式**：  
  消息角色为单独的 `role: "tool"`，必须提供与发起调用时严格一致的 `tool_call_id`：
  ```json
  {
    "role": "tool",
    "tool_call_id": "call_123",
    "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"
  }
  ```
- **Anthropic 回传格式**：  
  必须放在 `role: "user"` 的 `content` 数组中，作为一个类型为 `tool_result` 的内容块，通过 `tool_use_id` 进行关联：
  ```json
  {
    "role": "user",
    "content": [
      {
        "type": "tool_result",
        "tool_use_id": "toolu_456",
        "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"
      }
    ]
  }
  ```
- **Gemini 回传格式**：  
  在 `role: "function"` 或 `role: "user"` 的 parts 中，使用 `functionResponse` 对象，通过 `name` 和 `response` 字段传递结果字典。

---

## 统一中间抽象层设计

为了让应用代码与底层 Provider 解耦，必须在应用层建立统一的中间数据结构（Canonical Representation）。无论底层对接什么模型，上层业务只与该中间结构交互。

![基于中间抽象的枢纽转换架构](/images/content-rebuild/2df9306e7fed893c/fig-3.png)

### 1. 中间数据结构定义

在 Python 中，通常基于 Pydantic 定义统一的工具声明、调用指令与执行结果对象：

```python
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class CanonicalToolDefinition(BaseModel):
    """统一工具定义：上层只维护这一份规范"""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})

class CanonicalToolCall(BaseModel):
    """统一工具调用指令：屏蔽不同模型的原始响应差异"""
    call_id: str
    name: str
    arguments: Dict[str, Any]  # 统一为反序列化后的原生字典

class CanonicalToolResult(BaseModel):
    """统一工具执行结果：用于后续回传构造"""
    call_id: str
    name: str
    output: str                # 执行输出的文本或序列化字符串
    is_error: bool = False
```

### 2. 双向转换器实现

适配器模块（`ToolAdapter`）负责在统一结构与特定 Provider 之间双向翻译：下发工具时做**下行转换**，模型吐出结果时做**上行解析**，提交执行结果时做**回传封装**。

![ToolAdapter 数据流与转换处理机](/images/content-rebuild/2df9306e7fed893c/fig-4.png)

以下是最小化但功能完整的适配器实现示例：

```python
import json
from typing import List, Dict, Any

class OpenAIAdapter:
    @staticmethod
    def to_provider_tools(tools: List[CanonicalToolDefinition]) -> List[Dict[str, Any]]:
        """下行：转为 OpenAI tools 列表"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }
            }
            for t in tools
        ]

    @staticmethod
    def parse_response_tool_calls(response_message: Dict[str, Any]) -> List[CanonicalToolCall]:
        """上行：解析 OpenAI tool_calls"""
        raw_calls = response_message.get("tool_calls") or []
        results = []
        for item in raw_calls:
            if item.get("type") == "function":
                fn = item["function"]
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {"__raw_invalid_json": raw_args}
                results.append(CanonicalToolCall(
                    call_id=item["id"],
                    name=fn["name"],
                    arguments=args
                ))
        return results

    @staticmethod
    def to_provider_tool_result(result: CanonicalToolResult) -> Dict[str, Any]:
        """回传：构造 OpenAI tool 结果消息"""
        return {
            "role": "tool",
            "tool_call_id": result.call_id,
            "content": result.output
        }

class AnthropicAdapter:
    @staticmethod
    def to_provider_tools(tools: List[CanonicalToolDefinition]) -> List[Dict[str, Any]]:
        """下行：转为 Anthropic tools 列表"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters
            }
            for t in tools
        ]

    @staticmethod
    def parse_response_tool_calls(response_content: List[Dict[str, Any]]) -> List[CanonicalToolCall]:
        """上行：解析 Anthropic content blocks"""
        results = []
        for block in response_content:
            if block.get("type") == "tool_use":
                results.append(CanonicalToolCall(
                    call_id=block["id"],
                    name=block["name"],
                    arguments=block.get("input", {})  # 原生已是字典
                ))
        return results

    @staticmethod
    def to_provider_tool_result(result: CanonicalToolResult) -> Dict[str, Any]:
        """回传：构造 Anthropic user 消息（包含 tool_result 块）"""
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": result.call_id,
                    "content": result.output,
                    "is_error": result.is_error
                }
            ]
        }
```

---

## 边缘情况与容错降级

实际调用中，各平台校验器的宽容度差异经常引发接口报错。适配层必须在序列化前对 Schema 做边界清洗。

![边缘错误拦截与降级执行流](/images/content-rebuild/2df9306e7fed893c/fig-5.png)

### 1. `required` 字段的空列表校验

- **现象**：当工具没有任何必填参数时，标准的 JSON Schema 会生成 `"required": []` 或省略该字段。部分 Gemini 端点在收到 `"required": []` 时，会直接抛出 `INVALID_ARGUMENT`；而部分开源小模型在完全没有 `required` 键时，容易把所有参数都当成可选并不填任何入参。
- **处理方式**：在向 Gemini 发送前，适配器需检查 `required` 字段，若长度为 0 则直接从字典中剔除该键；而向 OpenAI 或 Claude 发送时，保留合法的 JSON Schema 结构。

### 2. 复杂类型（Union、`$ref`、`anyOf`）的降维扁平化

- **现象**：通过 Pydantic 自动生成的 Schema 常带有 `anyOf`（如可选字段导出的 `anyOf: [{type: string}, {type: null}]`）或本地结构引用（`$defs`、`$ref`）。Gemini 的 API 对这类高级关键字支持不完全，容易报错拒绝解析。
- **处理方式**：进入适配层前进行 Schema 预处理：
  1. 将带 `null` 的联合类型降维为基础类型，并在参数的 `description` 中补充“该参数可为空”的说明；
  2. 解析并内联展开（Inline Expand）所有 `$ref` 引用，避免模型端点解析器在多层嵌套路径中迷路。

### 3. 不支持 Tool Calling 的开源模型降级方案

部分本地部署的小参数模型未经过函数调用格式微调，API 接口不接收 `tools` 字段。此时需启用基于 Prompt 的降级策略：

1. **提示词注入 Schema**：适配器将 `CanonicalToolDefinition` 格式化为 JSON 字符串，拼接到 System Prompt 尾部，并明确规定唯一的输出格式：
   ```text
   你可以使用以下工具：
   [{"name": "get_weather", "description": "...", "parameters": {...}}]
   如果需要调用工具，请务必只输出以下格式的 JSON 代码块：
   ```json
   {"name": "工具名", "arguments": {"参数": "值"}}
   ```
   ```
2. **输出提取与回填**：模型返回后，适配器通过正则匹配提取该 JSON 代码块，反序列化为参数字典，并合成一个 `CanonicalToolCall(call_id="fallback_001", ...)` 传给业务层。业务层执行完成后，再将结果按纯文本形式拼进对话上下文。这样无需修改上层业务流程。

---

## 面试追问与架构选型

![API 网关转发与客户端适配器的分工权衡](/images/content-rebuild/2df9306e7fed893c/fig-6.png)

### 追问 1：如果大模型生成的入参 JSON 字符串发生截断或格式损坏，适配器该如何处理？

**回答要点**：
1. **本地容错修复**：在 Python 端捕获 `json.JSONDecodeError`，先通过 `jsonrepair` 等容错库进行修复（如自动补全末尾缺失的双引号和花括号）。
2. **携带报错重试纠错**：若修补依然失败，不要让整个系统直接抛异常中断。将错误包装为失败结果（`is_error=True`），写明：`"Error: Invalid JSON parameter. Details: <错误详情>"`。将该错误信息以工具结果的形式喂回模型上下文，引导模型在下一轮对话中重新输出正确的参数（参考 [Agent 错误恢复与重试](/interview/agent-interview/agent-error-handling.md)）。

### 追问 2：市面上已有 LiteLLM、One-API 等统一代理网关，为什么业务代码里还需要在应用层维护适配器？

**回答要点**：
1. **网关解决的是协议转发，应用层解决的是状态机管理**：中转网关通常只做 HTTP 报文层面的字段对齐（把请求映射为 OpenAI 兼容接口），但对多轮 Agent 循环中的复杂状态流向（例如工具调用中途插入人工审批、分支路由）缺乏上下文知晓。
2. **厂商专有特性容易丢失**：透明转发网关往往采用“最大公约数”策略，可能过滤掉特定厂商的事件流。例如 Anthropic 的思维链输出（Thinking Block）与工具调用块的交织顺序，经过标准化代理转换后可能被扁平化或丢弃；Gemini 的安全评分（SafetyRatings）也容易被忽略。
3. **内存级动态裁剪**：在应用层维护适配逻辑，可以根据实时上下文长度（Context Window）动态过滤工具、按用户权限即时下线某些敏感工具，便于做精细化的安全审计与端到端链路追踪。

---

## 常见误区

- **误区一：混淆 Schema 适配与 MCP（Model Context Protocol）**  
  Schema 适配解决的是**在同一个进程内将工具元数据翻译成不同 LLM 厂商的 API 报文格式**；而 [MCP 规范](/interview/agent-interview/mcp.md) 解决的是**客户端与外部独立服务之间跨进程、跨机器发现与调用工具的通信标准**。MCP Server 暴露出来的工具，在接入特定模型时依然需要通过本篇讨论的适配器转换并注入给 LLM。
- **误区二：误以为所有 Provider 返回的参数都是字典**  
  OpenAI 规范原生返回的是未解析的 `string`，而 Claude 返回的是解析好的结构化字典。如果适配器漏掉类型分支判断，直接调用 `.get()` 或盲目调用 `json.loads()`，会导致运行时抛出 `AttributeError` 或 `TypeError`。
- **误区三：认为只要符合 JSON Schema 规范，任何模型都能正确调用**  
  模型的调用效果高度依赖工具描述的质量。过深的嵌套层级、缺失字段 `description` 说明、过于复杂的枚举命名，都极易引发模型参数错填。在适配层做 Schema 转换时，应尽量保持参数结构扁平，并在描述中提供明确的取值样例。
