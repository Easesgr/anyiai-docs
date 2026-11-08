---
title: Agent 如何处理工具调用失败？格式自愈、重试与降级方案
icon: mdi:book-open-page-variant
description: 详解 Agent 工具调用中的四类常见故障，掌握参数校验自愈、带抖动的指数退避重试、业务空结果语义引导及熔断降级设计，含可检验练习代码。
---

**学习目标**：掌握 Agent 工具执行中四类典型故障（Schema 不匹配、网络超时、业务鉴权拒绝、语义空结果）的判定边界，能使用 Pydantic 实现带轮数上限的参数自愈机制，并编写带抖动的重试与确定性降级路由。

在由大语言模型（LLM）驱动的 Agent 系统中，模型本身并不直接运行外部代码。模型的作用是根据上下文输出结构化的函数名与参数，宿主程序解析这些内容后发起真实调用，再把执行结果包装为工具消息（Tool Message）写回上下文。

**核心结论**：工具调用失败不能全部交给模型去重新思考，也不能一律抛出异常中断任务。处理失败应遵循分层原则：
1. **格式与字段错误**：由宿主程序提取校验器返回的具名错误，回传给模型纠错（自愈）。
2. **网络瞬时超时**：由宿主程序在底层直接退避重试，不浪费模型 Token。
3. **业务拒绝与空结果**：将底层机器码翻译为带有行动建议的自然语言提示，引导模型换方向推理。
4. **重试与自愈耗尽**：打破模型循环，由预设代码分支切入只读缓存或人工介入，保证系统受控退出。

本文使用一个**机票查询与预订 Agent** 贯穿全文。该 Agent 包含两个工具：只读的机票检索 `query_flights` 与产生状态变更的订票扣款 `book_flight`。

承接上一讲 [/llm-100/agent-engineering/agent-tool-integration.md](/llm-100/agent-engineering/agent-tool-integration.md) 中工具注册与 Schema 定义的基础，本文聚焦执行阶段的容错。权限与隔离可参考 [/llm-100/agent-engineering/agent-permissions-isolation.md](/llm-100/agent-engineering/agent-permissions-isolation.md)，链路追踪可参考 [/llm-100/agent-engineering/agent-execution-tracing.md](/llm-100/agent-engineering/agent-execution-tracing.md)。

![Agent 工具调用容错机制全景概览](/images/content-rebuild/47eb3b6f58697288/cover.png)

---

### 工具调用的四类典型故障

工具失败的成因不同，处理层级也不同。如果把所有错误都原封不动打包给大模型，轻则浪费调用费用，重则引发死循环。

![工具调用四大故障类型与应对策略分流图](/images/content-rebuild/47eb3b6f58697288/fig-1.png)

这四类故障的表现与第一响应机制如下：

| 故障类型 | 贯穿案例中的典型表现 | 发生位置 | 第一响应机制 |
| :--- | :--- | :--- | :--- |
| **1. Schema 校验失败** | 模型将 `max_price` 传为 `"两千元"`，或漏传 `destination` | 本地宿主运行时 | 捕获 Pydantic 错误，构造结构化报错消息回传模型让其修正 |
| **2. 物理网络与网关超时** | 航司查询接口在 3 秒内未返回，或网关报 HTTP 504 | 网络传输 / 网关层 | 宿主程序执行指数退避加随机抖动重试，不通知模型 |
| **3. 业务与鉴权硬拒绝** | 订票扣款时返回 `HTTP 403 凭证过期` 或 `402 余额不足` | 远端业务系统 | 阻断重试，转为清晰业务说明通知模型或中断流程 |
| **4. 语义空结果与软异常** | 接口正常返回 200，但数据列表为空 `[]` | 业务逻辑层 | 注入语义提示，明确告知“无结果不等于接口坏了”，引导放宽条件 |

---

### 参数校验失败自愈：Pydantic 反馈机制

当模型生成的参数无法反序列化为目标对象时，直接让系统崩溃或只回复 `"Error: invalid arguments"` 都是不可取的。模型需要明确知道**哪个字段在什么位置出了什么问题**，才能一次性纠正。

![Pydantic 参数校验失败即时自愈闭环流程图](/images/content-rebuild/47eb3b6f58697288/fig-2.png)

#### 错误信息的提取与格式化

使用 Pydantic 进行类型校验时，其抛出的 `ValidationError` 包含了字段路径（`loc`）、错误信息（`msg`）和实际输入值（`input`）。宿主程序提取这些字段，组装成结构化文本：

```python
import json
from typing import Any, Dict, Tuple
from pydantic import BaseModel, Field, ValidationError

class FlightQueryArgs(BaseModel):
    origin: str = Field(..., min_length=3, max_length=3, description="出发机场三字码，如 PEK")
    destination: str = Field(..., min_length=3, max_length=3, description="到达机场三字码，如 SHA")
    max_price: int = Field(..., gt=0, description="预算上限，必须为正整数（单位：元）")

def validate_tool_args(raw_args_json: str) -> Tuple[bool, Dict[str, Any]]:
    # 阶段 1：JSON 语法校验
    try:
        parsed = json.loads(raw_args_json)
    except json.JSONDecodeError as err:
        return False, {
            "error_type": "JSONDecodeError",
            "message": f"参数无法解析为合法 JSON，报错位置：行 {err.lineno} 列 {err.colno}。请仅输出合法的 JSON 键值对。"
        }

    # 阶段 2：Pydantic Schema 校验
    if not isinstance(parsed, dict):
        return False, {
            "error_type": "TypeError",
            "message": "参数解析结果必须为 JSON 对象（键值对映射），不能为顶层列表或基本数据类型。"
        }

    try:
        validated = FlightQueryArgs(**parsed)
        return True, validated.model_dump()
    except ValidationError as err:
        details = []
        for item in err.errors():
            field_path = " -> ".join(str(loc) for loc in item["loc"])
            details.append(f"- 字段 '{field_path}': {item['msg']} (接收到的值: {item.get('input')})")
        
        error_payload = {
            "error_type": "ValidationError",
            "message": "参数字段校验失败，请依据以下要求修改后重新调用：\n" + "\n".join(details)
        }
        return False, error_payload
```

#### 限制自愈轮数

模型可能在复杂嵌套场景中反复生成格式不达标的参数。如果放任自流，会耗尽上下文和预算。必须在宿主状态中记录参数修正尝试次数：

```python
# 状态记录伪代码
state["repair_attempts"] = state.get("repair_attempts", 0) + 1
if state["repair_attempts"] > 2:
    # 超过 2 次自愈失败，切入备用分支，不再交由模型
    return route_to_fallback(state, reason="参数校验连续失败超限")
```

---

### 网络超时与重试：硬超时、指数退避与防抖

网络丢包、远端网关瞬时拥塞等故障无法通过大模型重新思考来解决。大模型改变不了物理连接状态。这类问题必须由宿主调用层在本地消化。

![带有随机抖动的指数退避重试时序机制图](/images/content-rebuild/47eb3b6f58697288/fig-3.png)

#### 超时时间配置

调用外部服务必须严格分离连接超时（Connect Timeout）与读取超时（Read Timeout）。连接超时通常设为 1 到 2 秒，读取超时根据业务特性限制在 3 到 5 秒，防止任务线程被外部卡死。

```python
import asyncio
import httpx

async def fetch_flight_api(url: str, params: dict, timeout_sec: float = 3.0) -> dict:
    # connect 限制握手耗时，read 限制网关传输耗时
    limits = httpx.Timeout(timeout_sec, connect=1.0)
    async with httpx.AsyncClient(timeout=limits) as client:
        # 使用 asyncio.wait_for 双重保底，防止底层连接池挂起
        response = await asyncio.wait_for(
            client.get(url, params=params),
            timeout=timeout_sec + 0.5
        )
        response.raise_for_status()
        return response.json()
```

#### 指数退避与随机抖动（Full Jitter）

多实例 Agent 并发运行时，外部服务短时不可用若采用固定间隔重试，会导致大量请求在同一秒钟同步重发，造成“惊群效应”（Thundering Herd Problem）。

标准重试策略采用带有全随机抖动的指数退避算法：

$$T_{\text{wait}} = \text{uniform}\left(0, \min(T_{\text{max}}, T_{\text{base}} \times 2^{\text{retry}})\right)$$

其中 $T_{\text{base}}$ 是初始基准时间（例如 0.5 秒），$T_{\text{max}}$ 是单次等待上限（例如 4 秒），$\text{retry}$ 是已重试次数（从 0 开始计数）。每次计算出上限区间后，在 $0$ 到该上限之间随机取值，把并发请求在时间线上打散。

#### 幂等性约束（Idempotency Guard）

**只有无副作用的操作才允许自动重试**：
- **只读操作（如 `query_flights`）**：反复查询不会修改数据状态，可安全重试。
- **写操作（如 `book_flight` 扣款与锁座）**：如果外部请求已经到达银行但网关在返回时发生读取超时，盲目重试会导致二次扣款。在未接入客户端幂等令牌（Idempotency Key）前，写操作超时必须直接挂起，不得触发自动重试程序。

---

### 业务异常与空结果的语义转化

当外部工具返回业务错误（如用户未认证）或合法的空集合（如当天没有机票）时，不能只把底层返回代码贴给模型，否则容易诱发模型胡乱猜测甚至捏造数据。

![业务空结果与纠错提示的上下文构造对比演示图](/images/content-rebuild/47eb3b6f58697288/fig-4.png)

#### 区分物理错误与事实否定

在检索类场景中，必须向模型讲明数据状态：
1. **接口故障**：`"[查询中断] 航司航班接口通信异常，并非没有航班，请勿向用户断言无票，建议告知用户系统维护稍后重试。"`
2. **空结果**：`"[查询完成] 已成功检索航司数据库，当前预算（2000元）内无符合 PEK->SHA 的直飞航班。建议操作：询问用户是否提高预算至 2500 元，或考虑转机。"`

#### 业务报错的语义包装

- **生硬报错（差）**：`HTTP 400: {"err_code": "ERR_DATE_PAST"}`
- **语义包装（优）**：`"工具调用失败：出发日期不能早于当前系统时间（2025-05-10）。请检查用户输入的出行日期并重新推理。"`

通过把状态码转化为明确的行动线索，模型就能直接产出向用户追问日期的有效回复，避免进入无效尝试。类似排查逻辑亦见于检索增强系统 [/interview/high-freq-scenarios/rag-retrieval-failure.md](/interview/high-freq-scenarios/rag-retrieval-failure.md)。

---

### 熔断降级：重试耗尽后的确定性流转

当参数自愈达到上限或外部服务重试耗尽后，必须通过工作流的状态图进行确定性熔断，不能继续把控制权留给模型。

![重试耗尽触发降级路由的分支状态流转图](/images/content-rebuild/47eb3b6f58697288/fig-5.png)

#### 条件路由设计（基于 LangGraph 状态图模式）

在状态机中，工具执行节点（Tool Node）输出后，通过条件边（Conditional Edge）判断是否跳出循环：

```python
from typing import Literal

def tool_execution_router(state: dict) -> Literal["agent", "degrade_cache", "human_review"]:
    status = state.get("last_tool_status")
    failures = state.get("consecutive_failures", 0)
    is_write_operation = state.get("is_write_action", False)

    # 执行成功，返回模型继续下一步规划
    if status == "success":
        return "agent"

    # 连续失败达 3 次，强制熔断，不再调用模型
    if failures >= 3:
        if is_write_operation:
            # 涉及资金与状态变更，转人工复核
            return "human_review"
        # 只读查询，转本地历史缓存降级节点
        return "degrade_cache"

    # 未超限且可重试，交回给模型处理错误反馈
    return "agent"
```

#### 确定性降级节点的行为

进入 `degrade_cache` 节点后，由手写代码执行固定动作，彻底绕过大模型推理：
1. **读取缓存快照**：读取 2 小时前的历史航线均价，并在文本开头附带系统声明：“航司实时网络受阻，以下为历史参考票价。”
2. **安全退出话术**：向用户输出受控提示：“抱歉，机票查询接口暂时无法响应，已保留您的需求，请 5 分钟后再试。”

关于系统级故障的整体设计可参考 [/interview/ai-system-design-interview/failure-handling.md](/interview/ai-system-design-interview/failure-handling.md) 与 [/interview/high-freq-scenarios/agent-failure.md](/interview/high-freq-scenarios/agent-failure.md)。

---

### 全流程审计：双轨数据分离

在容错设计中，不能直接把整个 Python 堆栈（Traceback）和几十行网络日志塞进模型的输入窗口，这不仅会挤占 Context Window，还极易导致小参数量模型注意力分散，复读报错文本。

![工具调用全生命周期的双轨日志与审计追踪拓扑图](/images/content-rebuild/47eb3b6f58697288/fig-6.png)

工程实践采用**双轨分离**：
1. **模型轨（In-Context Track）**：只向模型提供清洗后的业务信息与字段修改提示。
2. **审计轨（Telemetry Track）**：向 APM / 日志系统写入包含物理延迟、HTTP 原始状态码、已重试次数和 Trace ID 的完整追踪数据。

```python
import time
import logging

audit_logger = logging.getLogger("agent.audit")

def safe_execute_tool(tool_fn, tool_name: str, call_id: str, args: dict) -> dict:
    t0 = time.perf_counter()
    audit_record = {
        "trace_id": call_id,
        "tool_name": tool_name,
        "tool_args": args,
        "duration_ms": 0.0,
        "success": False,
        "error_type": None
    }
    try:
        result = tool_fn(**args)
        audit_record["success"] = True
        return {"model_content": str(result), "raw_data": result}
    except Exception as exc:
        audit_record["error_type"] = type(exc).__name__
        audit_record["raw_message"] = str(exc)
        # 运维审计轨：记录全部堆栈
        audit_logger.error("工具调用异常", extra=audit_record, exc_info=True)
        # 模型上下文轨：只给精简摘要
        return {
            "model_content": f"工具执行遇到错误（{type(exc).__name__}），请调整参数或告知用户稍后重试。",
            "raw_data": None
        }
    finally:
        audit_record["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        audit_logger.info("调用指标打点", extra=audit_record)
```

---

### 可检验练习

为检验你对参数校验自愈与重试截断的掌握情况，请完成以下练习。

#### 练习目标
编写一个容错调度控制器函数 `simulate_tool_loop`，模拟 Agent 对 `FlightQueryArgs` 工具的调用过程，要求满足：
1. 当参数格式不合法时，能解析出校验错误并累加 `repair_count`；
2. 当 `repair_count` 达到 2 次且依然失败时，强制熔断，返回预设降级信息，且不再发起后续尝试；
3. 一旦参数校验通过，返回执行成功。

#### 待测环境与测试用例代码

你可以把以下代码保存为 `exercise_tool_fault.py` 并直接运行验证：

```python
from typing import List, Dict, Any, Tuple
import json
from pydantic import BaseModel, Field, ValidationError

class FlightQueryArgs(BaseModel):
    origin: str = Field(..., min_length=3, max_length=3)
    destination: str = Field(..., min_length=3, max_length=3)
    max_price: int = Field(..., gt=0)

def simulate_tool_loop(mock_model_outputs: List[str]) -> Tuple[str, int]:
    """
    参数: mock_model_outputs 是模型在一轮会话中依次给出的参数字符串列表
    返回: (最终返回给用户的内容, 实际修正尝试次数)
    """
    repair_count = 0
    max_repairs = 2

    for raw_args in mock_model_outputs:
        try:
            data = json.loads(raw_args)
            valid_obj = FlightQueryArgs(**data)
            # 校验成功，正常返回
            return f"查询成功: {valid_obj.origin}->{valid_obj.destination}", repair_count
        except (json.JSONDecodeError, ValidationError):
            repair_count += 1
            if repair_count >= max_repairs:
                # 熔断降级
                return "触发降级: 连续参数校验错误超限，转入人工客服", repair_count

    return "调用未完成", repair_count

# --- 自动化验收断言 ---
if __name__ == "__main__":
    # 测试用例 1：第一次错误（max_price 是负数），第二次修正成功
    seq_1 = [
        '{"origin": "PEK", "destination": "SHA", "max_price": -500}',
        '{"origin": "PEK", "destination": "SHA", "max_price": 1200}'
    ]
    res_1, count_1 = simulate_tool_loop(seq_1)
    assert count_1 == 1, f"用例 1 重试计数错误，当前为 {count_1}"
    assert "查询成功" in res_1, f"用例 1 应成功，实际输出: {res_1}"

    # 测试用例 2：连续输出错误参数（机场代码长度不符），达到 2 次后必须熔断
    seq_2 = [
        '{"origin": "BEIJING", "destination": "SHA", "max_price": 1000}',
        '{"origin": "PEK", "destination": "SHANGHAI", "max_price": 1000}',
        '{"origin": "PEK", "destination": "SHA", "max_price": 1000}'  # 不应执行到这步
    ]
    res_2, count_2 = simulate_tool_loop(seq_2)
    assert count_2 == 2, f"用例 2 熔断计数错误，当前为 {count_2}"
    assert "触发降级" in res_2, f"用例 2 未触发降级，实际输出: {res_2}"

    print("所有容错测试用例校验通过！")
```
