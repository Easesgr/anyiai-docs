---
title: Agent 执行失败时如何恢复？运行时容错与降级设计
icon: mdi:alert-circle-outline
tag:
- Agent面试
- 错误处理
- 重试
description: 系统梳理 Agent 运行时的四类核心故障、异常清洗回灌、受限解码与提示重试、后备降级及断路器熔断机制，提供面试口答与工程实现。
---

在以大语言模型为核心的代理（Agent）系统中，执行路径不再由确定性的静态代码预先写死，而是由模型在循环中根据上一轮的观察结果动态决策。这种动态性带来了处理开放任务的灵活性，但只要外部 API 超时、模型输出畸变 JSON，或者陷入重复调用的死循环，整个流程就会中断。

当 Agent 执行失败时，系统既不能直接崩溃报错，也不能无限制盲目重试。构建高可用的 Agent，核心在于建立分层的故障处理链路：**捕获并清洗底层异常、向模型回灌具备纠错线索的观察值、利用结构化校验与受限解码收敛格式、在外部不可用时执行降级阶梯，并在发生死循环时依靠断路器果断熔断。**

![Agent 运行时容错与自愈闭环封面图](/images/content-rebuild/2e2eca7220310f3d/cover.png)

---

## 一、面试标准口述：Agent 运行时的四类故障与分级应对

面试中被问及“Agent 执行失败时如何恢复”时，建议先给出一套清晰的故障分类，再按层级展开对应的工程防护：

> **面试口述速记**：  
> “Agent 运行时故障主要分为四类：  
> 1. **协议与格式错误**：模型输出的 JSON 截断、未闭合或混入多余的 Markdown 标记；  
> 2. **参数与约束错误**：输出符合 JSON 语法，但字段缺失、类型不符或超出业务合法范围；  
> 3. **工具与环境异常**：外部 API 网络超时、接口鉴权失效或数据库不可用；  
> 4. **推理停滞与死循环**：模型在遭遇失败后缺乏调整能力，反复使用相同参数调用相同工具，耗尽 Token 预算。  
> 
> 恢复策略对应四个层次：  
> 对格式错误，在解码层使用**受限解码（Constrained Decoding）**，或在后置校验失败后追加包含具体错误字段的**纠错提示（Retry Prompt）**；  
> 对参数和环境错误，不能原样回灌敏感崩溃栈，而是通过**异常网关清洗脱敏**后作为 Observation 回传，引导模型自主修正；  
> 对外部持续不可用，触发**后备工具替换、只读数据降级或人工介入（Human-in-the-loop）**；  
> 最后在调度引擎外挂载**断路器（Circuit Breaker）**，依据步数阈值和动作哈希检测拦截死循环。”

![Agent 运行时的四类核心故障分类与发生阶段](/images/content-rebuild/2e2eca7220310f3d/fig-1.png)

如上图所示，在 [Agent 核心执行循环](/interview/agent-interview/agent-loop.md) 中，格式错误与参数错误发生在模型生成阶段；工具异常发生在环境执行阶段；而连续多轮未能推进目标时，则在状态流转中累积为死循环。

### 贯穿场景：财务数据对账 Agent
为了具体理解这四类故障，我们设定一个具体场景：**“财务对账 Agent”**。  
用户输入指令：“统计上月华东区退款金额超过 1 万元的订单，生成报表并邮件发送给主管。”  
执行过程中，Agent 需要按顺序执行三个动作：
1. 调用数据库查询工具 `query_sql`；
2. 调用数据汇总与绘图工具 `generate_report`；
3. 调用邮件发送工具 `send_email`。

在接下来的各个环节中，我们将观察该 Agent 遇到各类故障时系统如何拦截并恢复。

---

## 二、异常清洗与观察值回灌：为什么不能直传 Traceback？

在传统后端开发中，遇到网络超时或 500 错误通常采用指数退避重试（Exponential Backoff）。但在 Agent 场景中，如果报错原因是模型编写的 SQL 语句少写了一个逗号，或者把订单号参数传成了中文描述，原样重试无论多少次都必然失败。系统需要将失败信息转化为下一步的提示输入，促使模型改变参数。

但如果直接将底层环境的完整报错调用栈（Traceback）原样塞回给模型，会引发三个严重问题：
1. **安全与隐私泄露**：调用栈通常包含数据库内网 IP、连接池密码、数据表结构甚至服务器本地绝对路径。
2. **提示词注入与指令混淆**：若异常信息中包含外部未过滤的用户输入或网页抓取数据，攻击者可能在错误信息中伪造指令覆盖系统提示。
3. **有效注意力稀释**：数百行的调用栈会挤占上下文窗口，严重干扰注意力机制对最初任务目标和有效线索的提取。

因此，系统必须在工具层与模型层之间架设**异常清洗网关**，将底层异常过滤并格式化为符合 [Tool Calling](/interview/agent-interview/tool-calling.md) 协议的自然语言 Observation。

![异常堆栈清洗与反馈闭环架构](/images/content-rebuild/2e2eca7220310f3d/fig-2.png)

### 工程实现：异常脱敏与语义提炼

下面的 Python 代码演示了异常网关如何拦截内部错误、脱敏敏感信息并输出结构化指导：

```python
import re
from typing import Dict, Any, Callable

class ToolExceptionSanitizer:
    """清洗工具异常信息，避免暴露底层结构并提取关键纠错线索"""
    
    SENSITIVE_PATTERNS = [
        (r"(password|token|key|secret)\s*=\s*['\"][^'\"]+['\"]", r"\1=***"),
        (r"/Users/[^/]+(/.*)", r"/path/to\1"),
        (r"/home/[^/]+(/.*)", r"/path/to\1"),
        (r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b", "[INTERNAL_HOST]"),
    ]

    @classmethod
    def sanitize_error(cls, tool_name: str, raw_exception: Exception) -> str:
        raw_msg = str(raw_exception)
        cleaned_msg = raw_msg
        for pattern, repl in cls.SENSITIVE_PATTERNS:
            cleaned_msg = re.sub(pattern, repl, cleaned_msg, flags=re.IGNORECASE)
        
        lower_msg = cleaned_msg.lower()
        # 识别常见数据库错误并给出具有纠错导向的提示
        if "column" in lower_msg and "does not exist" in lower_msg:
            return (
                f"Tool '{tool_name}' failed: Column not found. "
                "Please verify table schema using describe_table before re-generating SQL."
            )
        elif "syntax error" in lower_msg:
            return (
                f"Tool '{tool_name}' failed: SQL syntax error near unexpected token. "
                "Please review SQL standard grammar and retry."
            )
        elif "timeout" in lower_msg:
            return (
                f"Tool '{tool_name}' failed: Request timed out. "
                "The query scope may be too large. Add date/region filters to reduce data volume."
            )
        
        # 默认只截取首行核心描述，丢弃深层堆栈
        first_line = cleaned_msg.split("\n")[0][:180]
        return f"Tool '{tool_name}' execution error: {first_line}. Please adjust input arguments."

def execute_tool_safely(tool_name: str, tool_func: Callable, **params) -> Dict[str, Any]:
    """安全执行工具并规范化输出格式"""
    try:
        result = tool_func(**params)
        return {
            "role": "tool",
            "tool_name": tool_name,
            "status": "success",
            "content": str(result)
        }
    except Exception as e:
        safe_feedback = ToolExceptionSanitizer.sanitize_error(tool_name, e)
        return {
            "role": "tool",
            "tool_name": tool_name,
            "status": "error",
            "content": safe_feedback
        }
```

以贯穿场景为例：当 Agent 执行 `query_sql(query="SELECT refund_amt FROM orders WHERE region='East'")` 时，数据库抛出 `psycopg2.errors.UndefinedColumn: column "refund_amt" does not exist`。网关不会将数据库堆栈甩回，而是输出：“`Tool 'query_sql' failed: Column not found. Please verify table schema using describe_table before re-generating SQL.`”。Agent 收到该信息后，即可主动调用 `describe_table` 检查字段，从而自主纠正字段名。

<ReadMoreLock />

---

## 三、结构自愈：受限解码与纠错提示（Retry Prompt）

当模型生成的工具调用无法被系统解析时，常见的两类错误是：JSON 缺失结尾括号等**格式畸变**，以及参数字段缺少必填项或类型错误的**校验失败**。针对这两类结构失效，工程上有两种主要解决方案：

### 1. 受限解码（Constrained Decoding / Grammars）
如果开发者使用的是可控推理引擎（如 vLLM、llama.cpp、Outlines 或支持结构化输出的 API），可在模型生成 Token 的过程中直接约束输出。

**机制**：推理引擎将目标工具的 JSON Schema 转换为有限状态自动机（FSM）或语法文法（EBNF）。在每一步生成下一个 Token 时，引擎检查当前生成的前缀状态，将词表中所有会导致 JSON 语法损坏或字段类型冲突的非法 Token 的 Logits 强制掩码为 $-\infty$。

*   **优势**：在单次生成中确保输出符合预设的语法与类型规范，不会出现缺少括号或整数写成字母的情况。
*   **边界**：受限解码只能保证语法结构和基础类型合规，无法保证参数在具体业务层面的合法性（例如日期范围前后颠倒、订单号不存在）；同时维护复杂文法状态机会占用额外的推理计算与调度开销。

### 2. 校验反馈重试（Retry Prompt Loop）
若调用的是黑盒商业模型 API，或者错误属于深层业务语义校验失败（例如 Pydantic 校验抛出 `ValidationError`），则需要依赖宿主环境的校验重试流程。

**机制**：宿主程序拦截反序列化异常或 Pydantic 错误信息，将出错的具体字段路径、预期类型以及模型上一步给出的错误参数打包成一条临时系统消息，再次请求模型：

```text
[System Feedback]:
Your previous call to 'generate_report' failed schema validation:
- Field 'chart_type': Value '3D_PIE' is not a valid enum. Allowed choices: ['line', 'bar', 'pie'].
- Field 'threshold': Missing required field.
Please regenerate the tool arguments strictly adhering to the schema.
```

![受限解码与补救提示对比示意图](/images/content-rebuild/2e2eca7220310f3d/fig-3.png)

如上图所示，受限解码作用于 Token 解码步进内部，依靠状态机过滤非法候选词；而纠错提示重试作用于模型输出完成后的验证层，通过将结构化错误回传给模型发起下一轮对话修正。

---

## 四、降级阶梯与状态流转

当底层服务发生宕机、配额超限（HTTP 429）或网络中断等硬故障时，要求模型反复调整参数重试只会浪费资源。系统需要配置预设的**降级阶梯（Fallback Ladder）**。

![降级阶梯与状态机流转示意图](/images/content-rebuild/2e2eca7220310f3d/fig-4.png)

### 1. 备用工具切换（Fallback Tool Substitution）
对于核心能力，可在工具注册表中绑定备用方案。  
在我们的财务对账例子中，如果默认的高并发数据库只读节点发生超时（Timeout），系统底层可自动将调用无缝切换至备用离线数仓接口，并在 Observation 中附带数据新鲜度提示。这样无需修改模型的上层逻辑，即可完成透明切换。

### 2. 功能退化通知（Graceful Degradation）
如果该类别的所有可用工具全部失效（例如邮件发送服务网关彻底宕机），调度引擎应主动拦截该工具的后续调用，并向对话上下文注入明确指示：
```text
[Runtime System Notice]:
The 'send_email' service is currently unreachable and all fallback channels failed.
Do NOT attempt to call 'send_email' again.
Please provide the final calculation results directly to the user in text format, and explicitly state that the email could not be delivered.
```
此时 Agent 停止尝试调用发信工具，转而将对账统计结果直接在对话框中输出，避免整个任务无功而返。

### 3. 人工介入与状态挂起（Human-in-the-loop）
对于涉及外部副作用不可逆的高风险操作（如退款打款、删除数据库表），或者重试达到阈值仍无法恢复时，执行器应主动触发中断（Interrupt），将任务状态持久化并转入人工审批队列。

主流编排框架（如 LangGraph）提供的持久化检查点（Checkpointer）机制，允许系统在节点执行前后保存完整的状态快照。人工审核人员修正参数或核准后，系统从检查点恢复（Resume）并继续向下流转。

---

## 五、防止死循环：运行时代理解除与断路器设计

在由模型自主决策的循环中，最容易失控的问题是“状态震荡死循环”：模型尝试调用 A，报错；模型由于注意力机制对上下文中错误前缀的偏好，在下一轮稍作无关改动甚至原封不动地再次调用 A，导致同样的报错再次出现，直到耗尽最大上下文长度或预算。

为了防范此类故障，必须在执行循环外部挂载**断路器（Circuit Breaker）**，从三个维度进行硬性防护。

![断路器状态机与检测拦截机制](/images/content-rebuild/2e2eca7220310f3d/fig-5.png)

### 断路器的三个监控维度：
1. **最大迭代步数（Max Iteration Steps）**：限制单次任务调用的总轮次（例如不超过 10 步），防止无限制消耗。
2. **连续错误计数器（Consecutive Failure Counter）**：当连续遭遇工具执行失败达到阈值（例如 3 次），暂停盲目调用，强制切换到降级或人工通道。
3. **动作指纹哈希（Action Fingerprint Hashing）**：将调用的工具名与其参数键值对进行排序并序列化计算哈希。如果在最近几步的滑动窗口内出现了完全相同的指纹，判定模型陷入无意义重复，直接熔断。

### 工程实现：Agent 断路器

```python
import hashlib
import json
from typing import Dict, Any, List

class AgentCircuitBreaker:
    """用于检测 Agent 执行循环中的死循环与高频连续失败"""
    
    def __init__(self, max_steps: int = 10, max_consecutive_errors: int = 3, window_size: int = 3):
        self.max_steps = max_steps
        self.max_consecutive_errors = max_consecutive_errors
        self.window_size = window_size
        
        self.current_step = 0
        self.consecutive_errors = 0
        self.action_history: List[str] = []
        self.is_tripped = False

    def _compute_action_hash(self, tool_name: str, args: Dict[str, Any]) -> str:
        # 参数键排序后序列化，确保即使字典无序也能得到稳定哈希
        serialized = f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def pre_tool_check(self, tool_name: str, args: Dict[str, Any]) -> None:
        """在工具实际调用前进行熔断判定"""
        if self.is_tripped:
            raise RuntimeError("Circuit breaker is OPEN. Execution halted.")

        self.current_step += 1
        if self.current_step > self.max_steps:
            self.is_tripped = True
            raise RuntimeError(f"Circuit Breaker tripped: Maximum allowed steps ({self.max_steps}) exceeded.")

        action_hash = self._compute_action_hash(tool_name, args)
        recent_actions = self.action_history[-self.window_size:]
        if action_hash in recent_actions:
            self.is_tripped = True
            raise RuntimeError(
                f"Circuit Breaker tripped: Duplicate action detected for '{tool_name}' within last "
                f"{self.window_size} steps. System detected a repetitive failure loop."
            )
        
        self.action_history.append(action_hash)

    def post_tool_record(self, is_error: bool) -> None:
        """在工具执行完成后更新失败计数"""
        if is_error:
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.max_consecutive_errors:
                self.is_tripped = True
                raise RuntimeError(
                    f"Circuit Breaker tripped: Consecutive error limit ({self.max_consecutive_errors}) reached."
                )
        else:
            self.consecutive_errors = 0
```

---

## 六、面试深入追问与常见边界

### 追问 1：为什么不能把原始数据库 Traceback 直接扔给 LLM？
**回答要点**：
*   **安全信息泄露**：调用栈会暴露数据库内网拓扑、表名结构、密码字段或敏感数据行。
*   **提示词注入风险**：未清洗的异常文本可能夹带恶意内容，干扰或覆盖主任务指令。
*   **注意力与上下文消耗**：长文本调用栈挤占上下文，降低模型对核心任务目标的关注度，容易引发幻觉。

![Traceback 原始暴露与净化反馈对比](/images/content-rebuild/2e2eca7220310f3d/fig-6.png)

如上图所示，直接回灌未经清洗的日志不仅带来安全隐患，还会导致注意力偏离；而通过网关脱敏提炼后，只保留关键错误原因与修复建议，能让模型高效调整参数。

### 追问 2：通过纠错提示（Retry Prompt）让模型纠错，在概率机制上有什么局限？
**回答要点**：  
大语言模型基于自回归生成。当一个错误的参数已经生成并保存在历史上下文中时，自注意力机制会对该错误前缀分配权重。在没有外部强干预的情况下，模型容易出现“自证实偏误”（Confirmation Bias），即倾向于顺着上一次的思路给出几乎一致的回答。

因此工程上需要配合其他手段：  
1. **适度提升采样温度（Temperature）** 或切换采样分支以跳出局部最优；  
2. **剪枝无效历史**：在重新尝试前，从上下文中剔除过去多轮尝试失败的中间交互，仅保留原始意图与最精简的报错提示；  
3. **强制规则兜底**：多次提示重试失败后，切换至确定性的代码规则或人工处理，不能完全押注概率模型自发收敛。

### 追问 3：断路器触发熔断后，Agent 的上下文状态应该如何处理？
**回答要点**：  
熔断发生后，直接丢弃任务或粗暴报错会导致极差的用户体验。标准工程处理包括：
*   **持久化执行快照**：将当前的会话历史、已成功步骤的输出、熔断原因写入数据库或持久化检查点（Checkpointer）。
*   **梳理并返回阶段性成果**：梳理此前已经成功执行的子任务。例如在对账场景中，告知用户：“已完成华东区退款金额统计与图表生成，但因邮件服务不可用且达到最大重试次数，未能发送邮件。请在此处下载报表文件。”
*   **挂载人工干预接口**：为该任务生成唯一跟踪 ID，方便人工排查或在修复外部接口后一键重放继续。
