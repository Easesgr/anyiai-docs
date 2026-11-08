---
title: Agent 如何接入搜索、数据库和本地文件？
icon: mdi:book-open-page-variant
description: 拆解大模型工具调用的执行机制，讲解搜索引擎清洗、SQL只读防护与文件沙箱隔离的落地实现。
---

大语言模型本身只是离散 Token 的概率生成器。模型既没有网络套接字（Socket）来请求网页，也没有数据库驱动去执行 SQL，更无法直接操作操作系统的文件描述符。

当用户提出“查一下订单 ORD-20260330 的退款状态；若没有记录则检索知识库退款规则，并将摘要保存到本地 reports/refund.txt”时，模型并不会直接运行代码。**工具调用的实际机制是：模型在上下文中生成符合格式约定的调用意图，外部宿主程序（Host Runtime）拦截该意图并代为执行真实代码，最后将运行结果封装后回填给模型。**

![Agent 工具集成的架构与协议交互全景](/images/content-rebuild/8668457d16206ae9/cover.png)

---

## 工具调用的执行机制

整个工具调用过程由“模型预测”与“宿主执行”交替驱动。一次完整的调用包含五个阶段：

![工具调用生命周期的序列拆解](/images/content-rebuild/8668457d16206ae9/fig-1.png)

1. **元数据注册**：宿主程序向模型发起推理请求时，将可用工具的名称、功能说明和参数规范（JSON Schema）组装并随请求传入。
2. **调用意图生成**：模型判断是否需要使用工具。若需要，模型生成包含目标工具名与参数的结构化内容，并在生成完成时输出特定的终止状态（例如 OpenAI 协议中的 `finish_reason: "tool_calls"`）。
3. **参数校验与拦截**：宿主程序拦截该响应，解析参数字符串，并使用校验器检查字段类型、必填项与取值约束。
4. **宿主执行副作用**：宿主程序在受控环境中调用具体的 Python 函数、网络 API、SQL 客户端或本地文件系统接口，并捕获执行结果或运行时异常。
5. **观察值回填**：宿主程序将执行结果构造成工具角色消息（例如 `role: "tool"` 或 `type: "tool_result"`），携带对应的调用 ID 追加到会话上下文中，再次触发模型生成下一轮推理或最终回复。

**模型负责决定“调什么”和“传什么参数”，所有对外部环境产生影响的操作（副作用）必须由宿主程序承载并校验。**如果宿主程序缺少参数验证与权限控制，模型生成的错误或恶意参数就会直接穿透到底层系统。

---

## 定义工具元数据：Schema 与描述边界

模型能否准确提取参数并命中正确的工具，取决于注册工具时提供的元数据质量。元数据包括接口技术定义与自然语言描述两个部分。

![工具元数据的结构定义与语义边界对比](/images/content-rebuild/8668457d16206ae9/fig-2.png)

### 1. JSON Schema 技术规范

参数定义应遵循标准 JSON Schema，显式声明类型（`type`）、必填字段（`required`）、正则模式（`pattern`）或取值枚举（`enum`）：

```json
{
  "name": "query_order_status",
  "description": "根据订单号查询订单当前的履约状态与物流信息。仅用于查询已有订单，不用于退款或修改订单。",
  "parameters": {
    "type": "object",
    "properties": {
      "order_id": {
        "type": "string",
        "pattern": "^ORD-[0-9]{8}$",
        "description": "标准订单号，以 ORD- 开头后接8位数字，例如 ORD-20260330"
      },
      "include_history": {
        "type": "boolean",
        "description": "是否返回历史状态变更记录，默认为 false"
      }
    },
    "required": ["order_id"]
  }
}
```

<ReadMoreLock />

### 2. 描述（description）的边界要求

工具描述并非单纯的代码注释，而是参与模型注意力计算的提示文本。编写工具描述时需要注意三点：

* **明确功能边界**：不仅写明工具能做什么，还要写明不能做什么（如“仅用于查询已有订单，不用于退款或修改订单”），降低语义相近工具之间的路由冲突。
* **标明格式与单位**：时间参数写清格式（如 `YYYY-MM-DD` 或秒级时间戳），金额和数值标明单位（如人民币“元”或“分”）。
* **提供合法范例**：在描述中给出具体可参考的格式范例，帮助自回归生成在解码时命中正确模式。

关于多厂商在 Schema 支持上的细节差异，可参考[《不同的 LLM Provider 对 Tool Schema 的支持不完全一致，你会怎么处理这种差异？》](/interview/agent-interview/tool-schema-compatibility.md)。

---

## 接入搜索引擎：清洗、截断与注入隔离

接入搜索引擎常见的问题是直接将搜索引擎 API 返回的原始 HTML 或未清洗的全文 JSON 拼入提示词。这不仅会迅速挤占上下文窗口，还可能引入网页中的间接提示注入（Indirect Prompt Injection，即网页内容中包含诱导模型忽略原有指令的恶意文本）。

![搜索引擎接入与数据流水线过滤](/images/content-rebuild/8668457d16206ae9/fig-3.png)

接入外部搜索应在宿主端建立数据过滤流程：

1. **提取结构化字段**：仅提取每个搜索结果的核心字段，包括标题（Title）、来源链接（URL）以及搜索引擎提供的正文摘要片段（Snippet）。
2. **内容降噪**：若需要进一步抓取网页落地页，需剥离 `<script>`、`<style>`、导航条和页脚代码，转换为纯文本或紧凑 Markdown。
3. **Top-K 截断与长度控制**：通常仅保留相关度最高的 3 到 5 条结果。在宿主端计算 Token 数量，对单条内容做字符截断，防止突发长文本挤占预算。
4. **数据边界隔离**：将外部检索内容包裹在独立标签中（如 `<search_results>`），并在系统提示词中声明标签内部属于未经验证的外部参考资料，降低外部文本劫持系统逻辑的风险。

过滤后的回填格式示例：

```markdown
<search_results query="电商退款规则">
[1] 标题: 售后退款退货政策说明
    链接: https://example.com/help/refund
    摘要: 用户在签收商品后7天内可发起无理由退货。定制作业与生鲜商品不支持该条款...
[2] 标题: 订单异常处理指南
    链接: https://example.com/help/order-issues
    摘要: 若订单处于未发货状态，用户可直接申请全额退款，系统将在1个工作日内原路退回...
</search_results>
```

---

## 接入数据库：只读账号、Schema 裁剪与语法拦截

如果直接让大模型生成任意 SQL 并执行，存在写操作破坏数据或慢查询耗尽连接池的风险。宿主程序必须在不同层级建立防御机制。

![数据库只读网关与安全拦截机制](/images/content-rebuild/8668457d16206ae9/fig-4.png)

### 1. 账号与运行环境隔离
* **独立只读权限**：供 Agent 查询的数据库账号必须使用只读角色配置（如只授予 `SELECT` 权限），撤销 `INSERT`、`UPDATE`、`DELETE`、`DROP`、`ALTER` 等一切写入与表结构变更权限。
* **只读从库路由**：将分析类和查询类请求路由到独立的只读副本（Read Replica），避免大查询锁定主库核心业务表。
* **连接超时限制**：在数据库连接层强制设置较短的单语句超时（如 PostgreSQL 的 `statement_timeout = 3000`，即 3 秒），阻断无索引全表扫描引发的资源耗尽。

### 2. 动态 Schema 裁剪
若数据库包含数十甚至上百张数据表，直接将完整 DDL 传入上下文会导致 Token 消耗过大，且容易干扰模型的判断。宿主应先做表名检索，仅将与当前提问相关的 3 到 5 张表的结构信息（表名、关键列名、数据类型及必要注释）注入上下文，并主动过滤敏感列（如密码散列值、个人身份敏感信息）。

### 3. AST 静态语法解析
仅用正则表达式匹配关键字容易被注释、换行符或特殊字符绕过。宿主端在执行生成的 SQL 前，应使用抽象语法树（AST）解析库（如 `sqlglot`）进行静态结构检查：
* **单语句限制**：检查 AST 顶层节点，确保只包含单一语句，拒绝分号拼接的多语句输入。
* **操作类型验证**：严格验证顶级 AST 节点是否为 `Select` 查询，拦截管理命令或结构变更指令。
* **分页改写**：若模型生成的查询未包含分页限制，解析器在顶层 AST 语法树节点中注入 Limit 限制并重新序列化为 SQL，而非直接在文本末尾拼接字符串，防止查询返回几万行数据占满宿主内存或破坏复杂查询语法。

---

## 接入本地文件系统：沙箱路径与原子操作

在代码生成与本地任务处理场景中，Agent 经常需要读写本地文件。如果缺乏路径约束，模型可能生成 `../../../../etc/passwd` 之类的路径进行越权访问。

![文件系统沙箱路径校验机制](/images/content-rebuild/8668457d16206ae9/fig-5.png)

### 1. 路径规范化与从属判断
所有文件操作必须限制在指定的沙箱目录内。防御路径遍历攻击的关键是**在路径规范化（Canonicalization）之后再进行从属判断**。

注意安全陷阱：**不能简单地使用字符串前缀匹配判断路径合法性**。例如沙箱根目录为 `/workspace/sandbox`，若攻击者传入 `../sandbox_secret/pass.txt`，计算后的路径为 `/workspace/sandbox_secret/pass.txt`，如果用 `str(target).startswith(str(SANDBOX_ROOT))` 判断，该路径仍然以 `/workspace/sandbox` 开头，导致访问到同级敏感目录。

在 Python 中，应使用 `Path.resolve()` 解析符号链接与相对路径，并调用 `is_relative_to` 验证边界：

```python
from pathlib import Path

SANDBOX_ROOT = Path("/app/workspace/sandbox").resolve()

def resolve_safe_path(user_input_path: str) -> Path:
    # 1. 拼接并解析所有 .. 与软链接，得到实际绝对物理路径
    target_path = (SANDBOX_ROOT / user_input_path).resolve()
    
    # 2. 严格校验目标路径是否完全位于沙箱根目录下
    if not target_path.is_relative_to(SANDBOX_ROOT):
        raise PermissionError(f"非法访问：路径超出沙箱范围 -> {user_input_path}")
    
    return target_path
```

### 2. 细粒度接口封装
不要直接向模型暴露没有边界的原始文件句柄，应提供封装好的操作工具：
* `read_file_chunk(path, start_line, max_lines)`：支持分页读取，例如单次限制最多读取 200 行，避免读取大日志文件时撑爆上下文。
* `write_file(path, content)`：限制单次写入内容的字节上限（如 1MB），并在写入前自动递归创建父目录。
* `list_directory(path, depth)`：列出目录内容时强制设置遍历深度与最大返回条数。

系统级隔离与权限控制的深度设计，可参考[《Agent 如何做权限控制和安全隔离？》](/llm-100/agent-engineering/agent-permissions-isolation.md)。

---

## 宿主执行器（Tool Runner）设计与并发调度

现代模型通常支持并行工具调用（Parallel Tool Calling），即单次推理可以输出多个调用请求（例如同时查天气与查汇率）。宿主程序需要调度执行这些请求、控制超时、统一捕获异常并把结果回传给模型。

![宿主执行器 Tool Runner 调度与反馈消息组装](/images/content-rebuild/8668457d16206ae9/fig-6.png)

工具执行器还要注意兼容同步阻塞函数与异步协程：如果注册的工具是同步的 CPU 计算或普通阻塞网络请求，需将其派发到线程池执行，避免锁死全局事件循环。

下面是一个包含 Schema 校验、超时控制与异步派发的执行器代码：

```python
import asyncio
import inspect
import json
from typing import Any, Callable, Dict, List
from pydantic import BaseModel, Field, ValidationError

# 1. 定义工具参数结构
class OrderQueryInput(BaseModel):
    order_id: str = Field(..., pattern=r"^ORD-[0-9]{8}$", description="标准订单号")

# 2. 定义具体工具函数（异步网络请求）
async def query_order(order_id: str) -> Dict[str, Any]:
    await asyncio.sleep(0.05)  # 模拟网络 IO
    if order_id == "ORD-20260330":
        return {"order_id": order_id, "status": "shipped", "carrier": "SF-Express"}
    return {"order_id": order_id, "status": "not_found"}

# 3. 宿主执行器实现
class ToolRunner:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, type[BaseModel]] = {}

    def register_tool(self, name: str, schema: type[BaseModel], func: Callable):
        self._tools[name] = func
        self._schemas[name] = schema

    async def execute_single_call(self, tool_call: Dict[str, Any], timeout: float = 3.0) -> Dict[str, Any]:
        call_id = tool_call.get("id", "unknown_id")
        func_meta = tool_call.get("function", {})
        tool_name = func_meta.get("name")
        arguments_raw = func_meta.get("arguments", "{}")

        # 路由校验
        if tool_name not in self._tools:
            return self._build_tool_message(call_id, f"Error: 工具 '{tool_name}' 不存在。")

        # Schema 强校验
        try:
            raw_dict = json.loads(arguments_raw) if isinstance(arguments_raw, str) else arguments_raw
            validated_args = self._schemas[tool_name].model_validate(raw_dict)
        except (json.JSONDecodeError, ValidationError) as err:
            return self._build_tool_message(call_id, f"Error: 参数格式错误 - {str(err)}")

        # 执行与超时控制（区分异步与同步）
        try:
            func = self._tools[tool_name]
            args = validated_args.model_dump()
            
            if inspect.iscoroutinefunction(func):
                coro = func(**args)
            else:
                coro = asyncio.to_thread(func, **args)

            result = await asyncio.wait_for(coro, timeout=timeout)
            serialized_output = json.dumps(result, ensure_ascii=False)
        except asyncio.TimeoutError:
            serialized_output = f"Error: 工具 '{tool_name}' 执行超时（超过 {timeout} 秒）。"
        except Exception as e:
            serialized_output = f"Error: 执行失败 - {str(e)}"

        return self._build_tool_message(call_id, serialized_output)

    def _build_tool_message(self, call_id: str, content: str) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": content
        }

    async def run_parallel(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tasks = [self.execute_single_call(call) for call in tool_calls]
        return await asyncio.gather(*tasks)

# 验证示例
if __name__ == "__main__":
    runner = ToolRunner()
    runner.register_tool("query_order", OrderQueryInput, query_order)

    mock_tool_calls = [
        {
            "id": "call_001",
            "type": "function",
            "function": {"name": "query_order", "arguments": '{"order_id": "ORD-20260330"}'}
        },
        {
            "id": "call_002",
            "type": "function",
            "function": {"name": "query_order", "arguments": '{"order_id": "INVALID_ID"}'}
        }
    ]

    results = asyncio.run(runner.run_parallel(mock_tool_calls))
    for msg in results:
        print(msg)
```

运行上述代码输出结果如下：

```text
{'role': 'tool', 'tool_call_id': 'call_001', 'content': '{"order_id": "ORD-20260330", "status": "shipped", "carrier": "SF-Express"}'}
{'role': 'tool', 'tool_call_id': 'call_002', 'content': 'Error: 参数格式错误 - 1 validation error for OrderQueryInput\norder_id\n  String should match pattern \'^ORD-[0-9]{8}$\' [type=string_pattern_mismatch, input_value=\'INVALID_ID\', input_type=str]'}
```

当执行器将包含错误信息的消息回传给模型后，模型即可在下一个思考步骤根据错误反馈修正参数，或调整后续执行路径。

关于跨进程或跨语言的通用工具集成协议，可参考[《MCP Server、MCP Client 和 Tool 分别是什么？一次完整调用流程是怎样的？》](/frontier/mcp-protocol/mcp-server-client-tool.md)。

---

## 可检验练习

为验证你对沙箱路径校验与工具调用的理解，请完成以下练习：

**任务目标**：编写一个名为 `safe_read_lines` 的 Python 工具函数并接入上面的 `ToolRunner`，满足以下验收条件：

1. **入参模式**：包含 `file_path: str`、`start_line: int`（默认为 1，必须 $\ge 1$）、`limit: int`（默认为 50，区间为 $[1, 100]$）。
2. **安全约束**：沙箱目录设为系统临时目录下的 `/tmp/agent_sandbox`（或当前目录的 `./sandbox`）。若传入路径通过 `..` 跳出该目录，函数需抛出或返回明显的权限拦截错误。
3. **验收测试**：
   * 在沙箱内创建一个包含 10 行文本的测试文件 `test.txt`。
   * 测试 1：读取 `file_path="test.txt"`, `start_line=2`, `limit=3`，应准确返回第 2 到第 4 行内容。
   * 测试 2：传入 `file_path="../../etc/passwd"`，验证执行器回传的消息中必须包含错误提示，且宿主程序不崩溃。

---

## 总结与延伸

接入外部资源是 Agent 从对话玩具走向工程实用工具的必经之路。设计工具接入层时应遵循以下原则：

1. **执行权与推理权分离**：模型只负责输出意图，宿主程序负责参数校验与真实执行。
2. **Schema 与描述并重**：Schema 约束参数结构，文本描述界定工具的适用边界与取值格式。
3. **资源分级防御**：搜索做降噪与截断，数据库做只读从库与 AST 分页限制，文件系统做路径解析与范围约束。
4. **执行器异常透明化**：通过 `tool` 角色消息将参数错误与超时信息如实回传，为模型的重试与纠错提供依据。

在生产环境中，外部接口不仅会报错，还可能遭遇网络抖动或模型反复生成错误参数。关于如何设计重试策略与状态回滚，请阅读下一讲[《Agent 如何处理工具调用失败？》](/llm-100/agent-engineering/agent-tool-failure.md)。
