---
title: ReAct 是什么？说说它的运行原理与工程实现
icon: mdi:thought-bubble
tag:
- ReAct
- 推理
- Agent
description: 解释 ReAct 核心范式、执行引擎状态循环、文本解析截断与工具异常保护机制，并对比原生 Tool Calling 的实现差异。
---

![ReAct 执行引擎实现与故障处理封面图](/images/content-rebuild/e6c572d2e213d757/cover.png)

面试官问到“ReAct 是什么？说说它的原理”时，很多候选人只背诵“Reasoning + Acting（思考加行动）的循环交替”，这只是论文的概念描述。面试官通常是在考察你是否真正写过或调试过 Agent：面对输出不稳定的模型，宿主程序如何把一段文本拆成调用参数？工具超时或报错时系统怎么处理？模型陷入重复调用的死循环时怎么打断？

关于 ReAct 与其他提示策略的基础对比，可参考章节 [/interview/agent-interview/react.md](/interview/agent-interview/react.md)。本文从运行时状态机、文本解析截断、异常防御与死循环控制四个工程环节展开说明。

---

## 面试核心口答模板

> **面试官提问**：请阐述 ReAct 的工作原理。如果要你实现一个 ReAct 执行引擎，核心模块与异常机制该如何设计？
>
> **口述回答范式**：
> 1. **核心定义**：ReAct 是一种让大语言模型交替进行“推理（Thought）”与“行动（Action）”的协同范式。模型面对复杂任务不是单次吐出最终答案，而是在每一步先写出当前的思考意图，决定调用哪个工具并给出参数，由宿主环境执行后把结果作为观察（Observation）灌回上下文，再驱动下一轮思考，直到得出最终答复。
> 2. **执行引擎架构**：大模型本身是无状态的，引擎本质上是一个运行在宿主进程中的**状态循环程序**。核心包含四个部分：
>    * **状态容器**：记录用户输入、历史工具调用记录（Scratchpad）与步数计数；
>    * **截断解析器**：结合停止词与宿主解析截断，阻止模型伪造后续观察值，并提取工具名与入参；
>    * **安全执行器**：负责参数校验、设置执行超时，并将底层代码报错转换为可读文本反馈；
>    * **上下文拼装器**：将真实的执行结果追加到提示词尾部，驱动下一轮迭代。
> 3. **核心防御机制**：
>    * **防伪造**：在接口支持时配置停止词辅助分段，并在宿主解析端做截断兜底，防止模型自回归编造环境反馈；
>    * **防崩溃**：工具抛错不能直接让主程序挂掉，必须将错误信息转为 Observation 喂回给模型，给模型修正参数的机会；
>    * **防死循环**：记录动作签名（工具名 + 参数哈希），当检测到连续重复调用或达到最大步数时，强制切断工具权限并提示模型给出最终答复。

---

<ReadMoreLock />

## 贯穿示例：ReAct 的完整执行流

为了直观理解这套机制，我们来看一个具体任务：**“查询订单 10023 的状态，若已签收则给用户发一条确认短信。”**

执行引擎与大模型的实际交互轨迹如下：

1. **第 1 轮**：
   * **Prompt 输入**：包含任务描述、可用工具说明以及历史对话。
   * **模型生成**：
     ```text
     Thought: 我需要先获取订单 10023 的详细信息，查看其是否已签收。
     Action: get_order_detail
     Action Input: {"order_id": "10023"}
     ```
   * **引擎拦截与执行**：遇到停止词中断生成。引擎调用数据库接口，返回结果。
   * **追加 Observation**：`Observation: {"status": "delivered", "phone": "13800000000", "sign_time": "14:30"}`

2. **第 2 轮**：
   * **Prompt 输入**：上一轮完整轨迹 + 新增的 Observation。
   * **模型生成**：
     ```text
     Thought: 订单状态为 delivered（已签收），手机号为 13800000000，我需要调用短信接口发送确认信息。
     Action: send_sms
     Action Input: {"phone": "13800000000", "message": "您的订单 10023 已于 14:30 签收。"}
     ```
   * **引擎拦截与执行**：调用短信网关，返回结果。
   * **追加 Observation**：`Observation: {"code": 0, "msg": "success"}`

3. **第 3 轮**：
   * **模型生成**：
     ```text
     Thought: 短信已成功发送，任务全部完成。
     Final Answer: 订单 10023 已于 14:30 顺利签收，且已向用户手机号 13800000000 发送了确认短信。
     ```

引擎检测到 `Final Answer:` 关键字，循环结束，向最终用户返回文本。

---

## 状态机与状态对象设计

大模型每次调用 API 都是一次独立的 HTTP 无状态请求。要实现多轮“思考-行动-观察”，必须由宿主进程维护状态对象。

![ReAct 引擎状态机与输入输出流动](/images/content-rebuild/e6c572d2e213d757/fig-1.png)

在参考 [LangGraph 架构概览](https://docs.langchain.com/oss/python/langgraph/overview) 的状态管理方式时，一个典型的状态容器可以用如下 Python 数据结构表示：

```python
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class AgentAction:
    tool: str
    tool_input: Dict[str, Any]
    thought: str

@dataclass
class AgentStep:
    action: AgentAction
    observation: str

@dataclass
class AgentState:
    input_query: str
    intermediate_steps: List[AgentStep] = field(default_factory=list)
    current_step: int = 0
    max_steps: int = 5
    final_answer: Optional[str] = None
    is_finished: bool = False
```

每次循环开始，引擎把 `input_query` 和 `intermediate_steps` 渲染进提示词；模型推理后得到 `AgentAction`；引擎执行外部工具得到结果，打包成 `AgentStep` 追加到列表；直到 `is_finished` 为 True 或达到 `max_steps`。关于生命周期运转细节，可结合 [/interview/agent-interview/agent-loop.md](/interview/agent-interview/agent-loop.md) 进一步理解。

---

## 文本输出解析与停止词机制

在没有原生 Tool Calling 功能的模型或传统纯文本 ReAct 模板中，系统依靠自然语言约定格式：

```text
Thought: 思考内容
Action: 工具名称
Action Input: JSON参数
Observation: 观察内容
```

此时引擎面临两个工程风险：**模型脑补虚假反馈** 与 **参数提取失败**。

![解析器拦截伪造 Observation 与正则抽取流程](/images/content-rebuild/e6c572d2e213d757/fig-2.png)

### 1. 为什么必须配置停止词（Stop Sequences）？
大模型本质是自回归文本续写器。如果上下文里有很多完整的示例，模型在输出完 `Action Input` 之后，极大概率会顺着生成：
```text
Observation: 接口调用成功，返回数据为...
```
这种伪造的观察值完全是模型凭空想象的幻觉。因此，在接口支持的前提下，通常会在请求参数中传入停止词辅助分段：
```python
stop=["\nObservation:", "\nObservation"]
```
当大模型解码出该标记时，推理服务端会终止生成并交回控制权。但需注意，stop 字符串主要帮助文本分段，并非所有模型接口都支持自定义停止词，且模型一旦出现大小写或格式变体仍可能漏截断。因此工程上必须由停止词与宿主解析器的截断逻辑做双重防御，不能仅靠单一配置保证完全阻断伪造。

### 2. 宽容型正则解析器
模型生成的 JSON 经常会包含多余的换行、Markdown 代码块标记（如 ` ```json `）或中文字符。解析器必须做容错处理：

```python
import re
import json
from typing import Tuple, Optional

ACTION_PATTERN = re.compile(
    r"Thought\s*:(.*?)\nAction\s*:(.*?)\nAction Input\s*:(.*)", 
    re.DOTALL
)
FINAL_PATTERN = re.compile(r"Final Answer\s*:(.*)", re.DOTALL)

def parse_react_output(llm_output: str) -> Tuple[Optional[AgentAction], Optional[str]]:
    # 优先检测是否已达到最终答复
    if "Final Answer:" in llm_output:
        match = FINAL_PATTERN.search(llm_output)
        if match:
            return None, match.group(1).strip()
            
    match = ACTION_PATTERN.search(llm_output)
    if not match:
        raise ValueError(f"模型输出未遵循 ReAct 格式: {llm_output}")
        
    thought = match.group(1).strip()
    tool_name = match.group(2).strip()
    raw_input = match.group(3).strip()
    
    # 清理包裹在外的 Markdown 代码块
    if raw_input.startswith("```"):
        raw_input = re.sub(r"^```(?:json)?\n|\n```$", "", raw_input.strip(), flags=re.IGNORECASE)
        
    try:
        parsed_input = json.loads(raw_input)
    except json.JSONDecodeError:
        # 若无法解析为 JSON，则退化为单文本参数字典
        parsed_input = {"query": raw_input}
        
    return AgentAction(tool=tool_name, tool_input=parsed_input, thought=thought), None
```

---

## 工具执行的安全防护与异常捕获

从模型输出中抽取到工具名和参数后，引擎将调用真实世界的外部代码。外部环境存在网络超时、服务崩溃、入参非法等诸多不稳定因素。

![执行阶段捕获与异常降级分支](/images/content-rebuild/e6c572d2e213d757/fig-3.png)

根据 [Anthropic 关于构建有效 Agent 的技术指南](https://www.anthropic.com/engineering/building-effective-agents)，外部工具抛错时，**绝不能让异常直接向上击穿并中断主进程**，而是应当把异常内容转化为一段文本，作为 `Observation` 送回给模型：

```python
import asyncio
from pydantic import ValidationError

async def execute_tool_safely(
    tool_registry: Dict[str, Any], 
    action: AgentAction, 
    timeout_seconds: float = 10.0
) -> str:
    tool = tool_registry.get(action.tool)
    if not tool:
        return f"ToolExecutionError: 工具 '{action.tool}' 不存在。可用工具列表: {list(tool_registry.keys())}"
        
    try:
        # 使用 Pydantic 等库进行入参契约校验
        validated_args = tool.args_schema(**action.tool_input)
        
        # 增加超时控制，防止某个网络调用无限期挂起进程
        result = await asyncio.wait_for(
            tool.run(validated_args),
            timeout=timeout_seconds
        )
        return str(result)
        
    except ValidationError as val_err:
        # 参数名传错或类型不对，把具体字段错误提示给模型
        return f"ToolInputValidationError: 参数格式错误，请根据错误修正入参: {val_err.errors()}"
    except asyncio.TimeoutError:
        return f"ToolTimeoutError: 工具 '{action.tool}' 执行超时（超过 {timeout_seconds} 秒），请尝试更换参数或稍后重试。"
    except Exception as err:
        # 捕获未知异常，屏蔽底层敏感堆栈，返回错误简述
        return f"ToolRuntimeError: 执行发生内部错误 ({type(err).__name__}): {str(err)}"
```

当模型把参数名写错（例如把 `order_id` 拼写成 `id`）时，`ValidationError` 会作为 Observation 返回，模型在下一轮思考中看到这条提示，就能自行修正入参。关于系统级重试与恢复的完整设计，可参阅 [/interview/agent-interview/agent-error-handling.md](/interview/agent-interview/agent-error-handling.md)。

---

## 结果清洗与上下文窗口防护

有些工具返回的数据量极为庞大（例如搜索引擎抓取回来的整张网页 HTML，或者数据库查询返回的 1000 行记录）。如果直接拼入上下文，会瞬间占满输入窗口，导致后续推理成本飙升甚至触发上下文超限报错。

![观察值数据清洗与上下文窗口防护](/images/content-rebuild/e6c572d2e213d757/fig-4.png)

宿主引擎在将 Observation 追加至状态列表前，必须做截断与清洗：

```python
MAX_OBSERVATION_CHARS = 2000

def sanitize_observation(raw_obs: str, max_chars: int = MAX_OBSERVATION_CHARS) -> str:
    if not raw_obs:
        return "Observation: (工具已成功执行，但返回结果为空)"
        
    # 压缩连续空行和冗余空白字符
    cleaned = re.sub(r"\n\s*\n", "\n", raw_obs.strip())
    
    if len(cleaned) <= max_chars:
        return f"Observation: {cleaned}"
        
    # 超过限制时，保留开头和结尾信息，中间打上裁剪标记
    half = max_chars // 2
    head = cleaned[:half]
    tail = cleaned[-half:]
    cut_count = len(cleaned) - max_chars
    return f"Observation: {head}\n... [系统提示：内容过长，中间已截断 {cut_count} 字符] ...\n{tail}"
```

结合清洗后的内容，引擎将轨迹按轮次组织后送入下一轮生成。更多关于上下文压缩与滑动窗口管理的细节，可参考 [/interview/agent-interview/context-management.md](/interview/agent-interview/context-management.md)。

---

## 死循环检测与分级干预

ReAct 在实际运行中最常见的失败现象是**重复调用死循环（Action Loop）**：模型在遭遇挫折或工具返回空结果时，用完全相同的参数反复调用同一个工具，白白耗尽 Token 配额。

![死循环检测、动作哈希与状态纠偏](/images/content-rebuild/e6c572d2e213d757/fig-5.png)

### 1. 动作签名哈希检测

执行引擎可通过记录每次 Action 的签名哈希来判定是否发生死循环：

```python
import hashlib

def compute_action_signature(action: AgentAction) -> str:
    # 对字典键进行排序，确保参数顺序不影响序列化结果
    serialized_input = json.dumps(action.tool_input, sort_keys=True)
    raw_sig = f"{action.tool}:{serialized_input}"
    return hashlib.md5(raw_sig.encode()).hexdigest()

class LoopDetector:
    def __init__(self, threshold: int = 3):
        self.history: List[str] = []
        self.threshold = threshold
        
    def is_looping(self, action: AgentAction) -> bool:
        sig = compute_action_signature(action)
        self.history.append(sig)
        
        # 连续 threshold 次签名完全一致，判定为死循环
        if len(self.history) >= self.threshold:
            recent = self.history[-self.threshold:]
            if len(set(recent)) == 1:
                return True
        return False
```

### 2. 三级干预策略

当检测到异常状态时，引擎按照严重程度分级应对：

| 级别 | 触发条件 | 引擎应对措施 | 注入提示范例 |
| :--- | :--- | :--- | :--- |
| **Level 1：单次错误** | 单次工具抛出异常或参数非法 | 捕获错误，包装为 Observation 喂回，允许模型自我调整 | `ToolInputValidationError: 缺少必填字段 'order_id'` |
| **Level 2：重复死循环** | 连续 3 次调用相同的工具与参数 | 强行改写 Observation，警告其动作无效，必须改变策略 | `SystemNotice: 你已连续多次使用相同参数调用该工具且未获得新信息。请停止重复调用，改用其他工具或直接基于已知信息回答。` |
| **Level 3：步数超限** | 当前步数达到 `max_steps` | 剥夺所有工具调用权限，只保留生成文本通道，强行要求产出结论 | `SystemNotice: 工具调用配额已用尽。请立即根据目前掌握的所有 Observation 输出 Final Answer。` |

---

## 纯文本 ReAct 与原生 Tool Calling 差异

早期 ReAct 主要靠 Prompt 约定文本格式并配合正则截断，而现代主流大模型（如 OpenAI、Claude 等）提供了原生 Tool Calling 协议支持。

![文本约定的 ReAct 与原生 Tool Calling 执行对比](/images/content-rebuild/e6c572d2e213d757/fig-6.png)

两者的核心差异主要在实现载体上：

* **通信协议与字段**：纯文本 ReAct 将思考、动作名、参数全部挤在 `content` 纯文本中，容易因格式偏差解析失败；原生 Tool Calling 在 API 响应中提供专属的 `tool_calls` 结构化字段，包含确定性的工具名和 JSON 字符串，并为每一次调用分配唯一的 `tool_call_id`。
* **多工具并发能力**：文本 ReAct 单轮很难同时发起多个动作；原生 Tool Calling 支持模型在单轮响应中返回多个工具调用项，引擎可以用 `asyncio.gather` 并发执行，再把结果统一作为 `tool` 角色的多条消息回传，大幅降低执行耗时。
* **思考过程可见性**：文本 ReAct 强依赖在文本中显式输出 `Thought:`；而在原生 Tool Calling 中，有些模型支持独立的思考字段（如 Claude 的 `thinking` 块），有些模型则直接输出调用参数，业务层需要根据具体模型特点决定是否需要额外显式引导模型进行 CoT 推理。

关于原生 Tool Calling 协议的详细数据结构与流式解析机制，可参阅 [/interview/agent-interview/tool-calling.md](/interview/agent-interview/tool-calling.md)。

---

## 面试追问与边界考查

### 追问 1：如果工具执行时间很长（如执行长时间的复杂 SQL 或大数据导出），引擎该如何设计？
**口述要点**：
不能使用同步阻塞逻辑。执行器必须引入异步超时机制（如 Python 的 `asyncio.wait_for`）。工程上通常拆分为两类设计：
1. **两级超时机制**：软超时（如 5 秒）触发进度消息上报给用户，硬超时（如 30 秒）强行中断协程并把超时错误作为 Observation 返回模型；
2. **异步轮询模式**：对于真正的长任务，工具设计为“提交任务并返回 Task ID”，后续模型通过循环调用“查询任务状态”接口，或通过引擎层面的人机协同（Human-in-the-loop）将状态持久化入库，等外部任务完成后唤醒 Agent 恢复执行。

### 追问 2：达到最大步数后模型依然拒绝输出 Final Answer，系统怎么兜底？
**口述要点**：
达到 `max_steps` 阈值后，引擎应进入强收敛模式：
1. 从 Prompt 中剔除工具列表，防止模型再次生成 `Action:`；
2. 在 Prompt 尾部追加强引导提示词（如 `“请基于以上已收集的所有信息，直接输出最终总结回答：”`）；
3. 若模型输出仍不合规或继续报错，系统触发降级兜底方案，向用户返回预设的安全文案（如 `“抱歉，经过多次尝试未能完整获取所需数据，请稍后重试或联系客服”`），并向日志系统打上链路失败埋点。

### 追问 3：多用户并发请求时，如何保证 ReAct 状态机的数据安全？
**口述要点**：
模型 API 交互是无状态的，但引擎维护的 `AgentState` 有状态。在并发场景下，同一个用户会话必须加分布式锁（如基于 Redis 的 Session Lock），防止多个请求同时向同一个状态对象的 `intermediate_steps` 追加数据产生竞态写入。状态演进推荐采用不可变数据流更新，每一轮迭代生成新的状态快照并持久化到数据库（如 LangGraph 的 Checkpointer 机制），以便崩溃后按节点恢复。

---

## 常见答题陷阱与误区

* **误区 1：认为只要 Prompt 写得足够详细，就不需要设置停止词（Stop Words）**。
  * *辨析*：自回归大模型具有续写惯性。即使 Prompt 中强调“不要自己编写 Observation”，在缺少停止词或解析截断时，模型依然容易在生成完 Action 后顺手编造出 Observation，破坏环境反馈的真实性。配置停止词可以有效帮助文本分段，但因模型接口支持情况各异且可能出现格式变体，工程上仍需配合宿主解析器的截断逻辑进行双重防御。
* **误区 2：把外部工具报错直接 raise 抛出并中断任务**。
  * *辨析*：Agent 的核心优势之一就是具备根据反馈调整策略的纠错能力。工具报错通常代表入参有误或网络抖动，把错误作为观察值提供给模型，模型通常能自主修正并重新发起调用。只有不可恢复的系统级故障才应中断。
* **误区 3：认为 ReAct 与原生 Tool Calling 是对立互斥的技术**。
  * *辨析*：ReAct 是一种“推理-行动-观察”的**算法与认知循环思想（Paradigm）**，而 Tool Calling 是模型厂商为了支持结构化调用提供的**底层通信协议（Protocol）**。现代 Agent 框架普遍是在原生 Tool Calling 的协议基础上，实现 ReAct 的闭环运转。
