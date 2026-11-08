---
title: 如何让大模型稳定输出结构化数据？Prompt 设计与约束解码
icon: mdi:code-json
tag:
- Prompt面试
- 结构化输出
- JSON
description: 梳理大模型结构化输出的两种路径：Prompt软引导与引擎级约束解码，结合日志提取实例解析Schema定义、XML权衡及宿主端校验降级方案。
---

在大模型应用开发中，模型生成的下游对象往往是后端微服务、自动化脚本或数据库。如果模型输出的文本无法被稳定解析为结构化数据（如合法的 JSON 对象），整个调用链路就会中断。如何保证大模型输出符合格式规范，是提示词工程与模型应用架构中的高频考点。

学习本篇前，建议已掌握 [Prompt 设计基础](/interview/prompt-interview/prompt-design.md)。如需探究底层推理架构实现，可参考 [结构化输出和 Function Calling 是什么？](/fundamentals/inference-and-generation/structured-output.md)。

![结构化输出技术全景：从 Prompt 软引导到约束解码与工程校验](/images/content-rebuild/13725640a73c3f17/cover.png)

---

## 核心面试口述：结构化输出的两种实现路径

面试中被问及“如何保证模型稳定输出 JSON 等结构化数据”时，建议先给出明确的两种路径分类，再说明生产环境中的配合策略：

> **核心口述回答：**
> “保证大模型结构化输出主要有两种路径：
> 1. **提示词软引导（Soft Constraint）**：在上下文提示词中注入强类型定义（如 TypeScript 接口或 JSON Schema）、声明枚举常量与缺省值处理规则，并给出少量高质量示例（Few-shot）。这种方式兼容所有文本大模型，但由于自回归生成的概率特性，在长文本、深层嵌套或遇到输出截断时，依然可能出现字段缺失、括号未闭合或多余解释文字。
> 2. **推理引擎级约束解码（Grammar-based Constrained Decoding / Hard Constraint）**：例如各大模型厂商提供的结构化输出功能（Structured Outputs），或开源推理引擎（如 vLLM、llama.cpp、Outlines）。该机制在模型自回归采样的每一步，通过词法有限状态机（FSM）或语法树，把词表中不符合 Schema 语法的候选 Token 的 Logit 设为 $-\infty$（施加掩码），确保每一步采样出来的 Token 都符合指定的文法规则。
> 
> 在生产架构中，只要底层模型接口支持，**优先采用约束解码保证语法严格合规**；若接口不支持或使用纯文本流式输出，则使用‘规范提示词 + 宿主端截断探测 + 语法微修复（如 json-repair）+ 校验带错重试’的防御方案。”

![纯 Prompt 软引导与约束解码在 Token 采样阶段的区别](/images/content-rebuild/13725640a73c3f17/fig-1.png)

软引导依赖模型自身在概率分布中选出符合规则的 Token；约束解码则在 Softmax 采样前对非法候选进行掩码拦截，改变了最终采样的 Token 候选池。

<ReadMoreLock />

---

## 贯穿示例：系统告警日志提取

为了直观理解这两种路径及相关边界，本文使用一个统一的贯穿示例：**从非结构化的服务器运维日志中提取结构化事件数据**。

待处理的日志原文：
```text
[2025-05-12 14:23:01] ERROR [db_pool] Connection pool exhausted on node-01, node-02. 
HTTP status 504 returned to client. Retrying in 5s.
```

期望提取出的数据对象需满足如下结构：
- `event_name`：事件名称，字符串。
- `severity`：严重级别，只能是 `"INFO"`、`"WARNING"`、`"CRITICAL"` 之一。
- `affected_hosts`：受影响的主机名列表，字符串数组。
- `error_code`：错误代码，整数；若未明确提及则为 `null`。

---

## 提示词软引导：四项核心设计规范

当底层推理引擎或接口不支持强制约束解码时，结构化输出的稳定性高度依赖提示词的设计质量。编写高容错的结构化提示词需遵循以下四项规范：

### 1. 契约表达：优先使用紧凑的类型定义
用松散的自然语言描述（如“请输出包含事件、级别、机器列表的 JSON”）容易引起字段名漂移（例如把 `affected_hosts` 写成 `hosts` 或 `servers`）。
相比冗长的完整 JSON Schema，TypeScript 接口定义对大模型语义更清晰且节省 Token：

```typescript
// 推荐：语义明确、紧凑，且主流大模型在代码语料中充分学习过此类结构
interface SystemAlertEvent {
  /** 事件名称，简明英文大写下划线标识，如 DB_CONNECTION_TIMEOUT */
  event_name: string;
  /** 告警严重级别 */
  severity: "INFO" | "WARNING" | "CRITICAL";
  /** 受影响的主机名数组，未提及则返回空数组 [] */
  affected_hosts: string[];
  /** 提取到的错误码数值，未提及或无明确数字时填 null */
  error_code: number | null;
}
```

### 2. 缺省值与空值约定（Null Safety）
如果原文没有提及某个字段，模型容易出现两种错误行为：随意编造一个数字填入，或者直接省略该 Key，导致后端解析报 `KeyError`。在提示词中必须明确要求：
> “必须输出 Schema 中定义的所有字段。如果输入文本未提及对应事实，必须填充约定的缺省值（如 null 或空数组 []），严禁漏输出任何键。”

### 3. 枚举收敛与防发散
对于分类字段，严禁让模型自由发挥，必须列举完整的闭合取值集合：
> “`severity` 字段必须且仅能从 ["INFO", "WARNING", "CRITICAL"] 中选取一个，严禁自创同义词（例如严禁输出 'FATAL' 或 'URGENT'）。”

### 4. 字段职责单一明确
避免定义过于宽泛的大字段（例如 `details: string`）。字段职责越单一明确，模型越不容易掺杂无关内容。每个字段应具备简单的判定条件说明（如“若无具体错误码填 null”）。

![生产级结构化 Prompt 核心规范与字段约束设计](/images/content-rebuild/13725640a73c3f17/fig-2.png)

---

## JSON 与 XML 标签的工程权衡

在以 Claude 系列模型为主，或任务包含较多长文本推理提取的场景中，许多团队会在 JSON 与 XML 标签之间做权衡。

![JSON 与 XML 在长文本抽取与思维链场景下的工程权衡](/images/content-rebuild/13725640a73c3f17/fig-3.png)

两者的对比与选型考量如下：

| 维度 | JSON 输出 | XML 标签包裹输出 |
| :--- | :--- | :--- |
| **解析生态** | 现代后端原生支持（Pydantic、Jackson 等），反序列化方便 | 需要使用正则匹配或 XML 解析器提取指定标签内容 |
| **未转义字符容错** | **低**。若字符串内容中包含未转义的双引号（`"`）或原始换行，会导致 JSON 解析器抛错 | **高**。标签内部可直接容纳任意自然语言文本与换行，无引号转义冲突 |
| **思维链协同** | 较难协同。若强制要求首个字符输出 `{`，模型无法在生成 JSON 前输出推理草稿 | **天然协同**。可轻松划分为 `<thinking>...</thinking>` 与 `<event>...</event>` |
| **截断恢复** | 缺少结尾闭合括号 `}` 时，标准 JSON 解析器直接报错 | 可以用正则快速提取已经闭合的前置完整标签段落 |
| **适用场景** | 纯数据返回、直接对接业务微服务、启用了约束解码的场景 | 复杂文本提取、需要模型先思考后回答的场景 |

大模型是自回归生成的：$P(y_t \mid y_{<t}, x)$。如果任务难度较高，且强制模型输出的首个 Token 必须是 `{`，模型就无法利用自然语言生成过程作为“思考暂存区”，容易直接在字段中输出推导不充分的内容。此时使用 XML 标签将“推理过程”与“最终数据”分开，是一种常见且实用的设计。思维链机制详见 [Chain-of-Thought 原理解析](/interview/prompt-interview/chain-of-thought.md)。

---

## 宿主端防护流程：截断、轻量修复与带错重试

模型生成的文本属于外部不确定输入。即使模型接口配置了 JSON 模式，宿主端程序依然需要构建防护流程：

![宿主端鲁棒性闭环：截断探测、轻量修复与带错重试流程](/images/content-rebuild/13725640a73c3f17/fig-4.png)

### 1. 截断检测（Truncation Detection）
在纯提示词模式下，长文本生成失败的最常见原因是**触碰了 `max_tokens` 上限**。一旦达到最大生成长度，推理会被物理终止，输出停留在一个未闭合的状态：
```json
{"event_name": "DB_CONNECTION_TIMEOUT", "severity": "CRITICAL", "affected_hosts": ["node-01", "no
```
此时模型 API 返回的元数据中，`finish_reason` 通常是 `"length"` 而非正常结束的 `"stop"`。如果检测到 `finish_reason == "length"`，无需直接进入反序列化，应判定为截断异常并触发续写或降级逻辑。

### 2. 文本清洗与轻量语法修复
模型输出有时会附带 Markdown 代码围栏（如 ````json ... ````）或前置客套话（如“好的，这是提取结果：”）。
宿主端应先剥离 Markdown 代码围栏与前置客套话，定位到首个 `{`（或 `[`）起始位置。对于结尾缺少反括号、有多余末尾逗号（Trailing Comma）等微小语法瑕疵，切忌盲目截取到最后一个现存的 `}`（以免将内层嵌套之后的未闭合字段截断丢弃），而应在进入严格校验前，使用容错工具（如 Python 的 `json-repair`）尝试补全与修复。

### 3. 带错反馈重试（Reflection Retry）
当校验失败时，直接无脑重新发起完全相同的请求往往命中率不高。更有效的做法是记录校验失败的具体原因（例如类型校验器抛出的异常信息），将**上一次模型生成的错误文本**与**错误描述**作为补充上下文发送给模型：
> “你上一轮返回的 JSON 无法通过校验。错误原因：`error_code 必须为整数或 null，你输入了 'none'`。请修改该字段并重新输出完整合法的 JSON。”

---

## 代码实战：结合 Pydantic 的解析与防护方案

以下是一个完整的 Python 处理示例，演示清洗围栏、容错修复到强类型校验的完整流程：

```python
import json
import re
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ValidationError
import json_repair

# 1. 定义期望的业务数据结构
class SystemAlertEvent(BaseModel):
    event_name: str = Field(description="事件名称")
    severity: Literal["INFO", "WARNING", "CRITICAL"] = Field(description="严重级别")
    affected_hosts: List[str] = Field(default_factory=list, description="受影响主机名列表")
    error_code: Optional[int] = Field(None, description="错误码数值，无则为 null")

# 2. 模拟大模型输出的瑕疵文本（夹带 Markdown 围栏且结尾缺少花括号）
raw_llm_response = """
好的，已完成日志分析，提取结果如下：
```json
{
  "event_name": "DB_CONNECTION_TIMEOUT",
  "severity": "CRITICAL",
  "affected_hosts": ["node-01", "node-02"],
  "error_code": 504
"""  # 模拟由于截断或失误丢失了闭合的反花括号

# 3. 宿主端容错解析函数
def parse_alert_event(llm_output: str, model_cls: type[BaseModel]) -> BaseModel:
    # 步骤 A：剥离 Markdown 代码围栏（兼容末尾截断未闭合的情况）
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)(?:```|$)"
    match = re.search(fence_pattern, llm_output)
    candidate_text = match.group(1).strip() if match else llm_output.strip()
    
    # 步骤 B：尝试标准 JSON 解析，失败后尝试自动修复
    try:
        parsed_dict = json.loads(candidate_text)
    except json.JSONDecodeError:
        # 使用 json_repair 工具修补缺失括号或逗号
        repaired_text = json_repair.repair_json(candidate_text)
        parsed_dict = json.loads(repaired_text)
        
    # 步骤 C：执行 Pydantic 强类型与枚举校验
    return model_cls.model_validate(parsed_dict)

# 运行验证
try:
    event = parse_alert_event(raw_llm_response, SystemAlertEvent)
    print("解析成功：", event)
    print("输出标准 JSON：", event.model_dump_json())
except (ValidationError, Exception) as err:
    print("校验彻底失败，准备带错重试：", err)
```

![Pydantic 与 JSON Schema 驱动的类型安全防护链路](/images/content-rebuild/13725640a73c3f17/fig-5.png)

该方案把大模型输出视为“不可信输入”，通过代码建立防线，保证进入核心业务系统的数据类型完全符合预期。

---

## 高频面试追问与技术边界

### 追问 1：基于语法约束的解码（Grammar-based Decoding）会有额外开销吗？
**口述要点：**
约束解码并非零成本，其开销主要体现在三个方面：
1. **初始化编译开销**：引擎在接收到 JSON Schema 时，需要将其预编译为有限状态机（FSM）或语法解析表。若 Schema 嵌套过深、包含大量复杂的正则表达式或递归结构，编译阶段会有明显的 CPU 计算和内存占用；
2. **解码步骤的词表过滤开销**：在模型自回归生成每一个 Token 时，引擎需要根据当前状态机状态判定哪些 Token 是合法的，并将非法 Token 的 Logit 置为 $-\infty$。虽然现代引擎（如 XGrammar）利用位图预计算加速了这一过程，但在超高并发或超大词表（如 128k 词表）场景下，依然会增加单步解码的时延；
3. **语义表达扭曲风险**：约束解码只保证文法（Syntax）合法，无法保证内容（Semantics）真实。如果前面的推导有误，强制约束可能会迫使模型在受限的 Token 集合中选择概率次优甚至荒谬的词汇来强行满足语法要求。

![语法约束解码的性能开销与状态机边界分析](/images/content-rebuild/13725640a73c3f17/fig-6.png)

### 追问 2：思考链（CoT）与严格 JSON 输出在底层机制上有什么冲突？如何解决？
**口述要点：**
冲突在于自回归生成的依赖关系：如果限制输出的第一字符必须是 `{`，模型就无法在生成最终结果前进行自然语言草稿推理，对于复杂归纳、代码分析或多步计算类任务，容易造成字段内容准确度下降。

**常见的工程解法：**
1. **Schema 预留思考字段**：在 JSON Schema 的首个字段声明 `"thinking": {"type": "string"}`，引导模型在填写业务数据前，先在 `thinking` 字段内输出推理过程；
2. **两阶段架构或 XML 标签解耦**：第一阶段允许模型自由输出推理内容，在末尾使用 `<result>{...}</result>` 标签包裹纯数据；或者第一阶段做纯文本推理，第二阶段调用小模型专门负责将推理结果提取为结构化 JSON。

--- 

## 常见面试答题误区

1. **混淆 JSON Mode 与 Structured Outputs**：很多候选人误以为只要在 API 请求中加入 `response_format={"type": "json_object"}` 就能保证字段完全正确。实际上，JSON Mode 只约束输出是合法的 JSON 语法，根本不校验里面有哪些 Key、类型是否正确；只有基于 JSON Schema 的约束解码（如 Structured Outputs）才能保证输出与定义的字段结构一致。
2. **忽略 Token 截断导致的物理性语法损毁**：在回答“如何保证 JSON 解析不报错”时，如果只讨论提示词怎么写，而忽略了 `max_tokens` 导致的物理截断，说明缺乏生产环境的排障经验。
3. **迷信单次提示词的“绝对要求”**：试图在提示词中使用“绝对不能错”、“必须遵守”等情绪化词汇来彻底杜绝格式错误。大模型输出属于概率采样，提示词只能降低出错概率；宿主端的类型校验、容错修复和防御性重试才是系统稳定运行的底线。深入掌握 Prompt 失效原因可继续参考 [Prompt 为什么会失效？](/interview/prompt-interview/prompt-failure.md) 与 [工具调用 Tool Calling 实现](/interview/prompt-interview/tool-calling-spring-ai.md)。
