---
title: 如何让大模型稳定输出 JSON？从提示词约束到受限解码
icon: mdi:message-text-outline
date: 2026-07-16
tag:
- 大模型100讲
- Prompt
description: 分析大模型输出 JSON 格式破损的原因，讲解提示词约束、API 结构化参数、受限解码与应用层容错的完整处理方案。
---

在大模型与下游程序对接时，JSON 是最常见的数据交换格式。但在实际调用中，模型经常返回带有 Markdown 标记、末尾多出逗号或字段名拼写不一致的文本，导致下游代码在解析时报错崩溃。

要解决这个问题，需要根据业务场景选择处理方案：在提示词中明确类型约束、调用服务商提供的结构化输出参数、在推理引擎中启用受限解码，或在应用层加入容错修复。

![稳定输出 JSON 的工程分层全景图](/images/content-rebuild/96e9ee43838eff6f/cover.png)

上图展示了保障 JSON 输出稳定性的四层防御体系：从最上层的提示词诱导，到接口层的参数约束，再到底层的受限解码，最后由后置解析校验兜底。理解这些层级需要首先拆解格式破损的具体根因。

---

### 一、贯穿示例与常见格式破损原因

以一个典型的用户信息提取任务为例。假设输入文本为：

> “张三今年 28 岁，精通 Python 和 Go。”

期望得到的 JSON 数据为：

```json
{
  "name": "张三",
  "age": 28,
  "tags": ["Python", "Go"]
}
```

如果直接在提示词中写“请输出 JSON”，程序调用 `json.loads()` 时经常遇到以下四种异常：

1. **解释性前缀与 Markdown 代码块**：模型在预训练与对话对齐语料中学习了大量自然语言问答模式，倾向于在开头添加“好的，这是提取的数据：”，或者用 ` ```json ... ``` ` 包裹内容。下游解析器如果直接读取字符串，会触发词法解析错误。
2. **尾部悬空逗号（Trailing Comma）**：在生成最后一个元素时（如 `["Python", "Go",]`），自回归采样很容易预测出逗号 `,`。虽然部分编程语言的对象字面量允许尾部逗号，但在严格的 JSON 标准（RFC 8259）中，逗号后面必须跟有效的值，否则属于语法错误。
3. **键名漂移（Key Drift）**：模型在生成字段名时，可能输出 `userId`、`username` 代替预期的 `name`，导致下游强类型校验找不到对应字段。
4. **单双引号与字符转义问题**：模型受 Python 字典等代码语料影响输出单引号 `'name'`，或者字符串字段内部的双引号没有添加转义符 `\"`，导致引号闭合混乱。

![常见 JSON 破坏模式示例](/images/content-rebuild/96e9ee43838eff6f/fig-1.png)

上图归纳了自回归解码过程中最常见的四种语法破坏表现。仅依靠提示词声明“禁止输出解释性文字”，无法在概率分布上完全杜绝低概率异常 Token 的采样。

<ReadMoreLock />

---

### 二、提示词优化：TypeScript 接口与前缀诱导

当调用不支持结构化参数的第三方接口时，可以通过在提示词中提供清晰约束来降低出错概率。

#### 1. 使用 TypeScript 接口定义数据结构
相比自然语言描述，TypeScript 接口表达字段名、数据类型和可选状态时语义更加严谨，能够更有效地激活模型在预训练中学习到的类型模式：

```typescript
// 在系统提示词中定义目标格式契约
interface UserExtractionResult {
  /** 用户姓名，未提及填 null */
  name: string | null;
  /** 年龄，正整数，未提及填 null */
  age: number | null;
  /** 技能标签列表，无标签时输出空数组 [] */
  tags: string[];
}
```

关于系统提示词与输入上下文的分块编排，可以参考 [/llm-100/prompt-engineering/结构化_Prompt_应该怎么写.md](/llm-100/prompt-engineering/结构化_Prompt_应该怎么写.md)。

#### 2. Few-shot 样例与 Assistant 前缀诱导
给出 1 到 2 个输入与合法 JSON 的对照样例，可以明确格式边界。在支持补全前缀的接口（如 Anthropic 的 Claude API 或部分兼容接口）中，可以在 `assistant` 消息的开头预先填入 `{`：

```python
# 伪代码：在 assistant 角色开头预填字符
messages = [
    {"role": "system", "content": "你是一个结构化提取器，仅输出符合 UserExtractionResult 规范的纯 JSON。"},
    {"role": "user", "content": "张三今年 28 岁，精通 Python 和 Go。"},
    {"role": "assistant", "content": "{\n  \"name\": "}  # 预先填入起始大括号与首个键名
]
```

当生成起点已被固定为 JSON 对象的内部结构时，模型就不会在开头输出客套话或 Markdown 标记。需要注意，接口通常仅返回模型在预填内容之后继续生成的文本，下游在调用解析器前必须将预填前缀与返回文本拼接完整。关于消息角色的生效机制，可以参考 [/llm-100/prompt-engineering/System_Prompt_User_Prompt_Assistant_Prompt_分别有什么作用.md](/llm-100/prompt-engineering/System_Prompt_User_Prompt_Assistant_Prompt_分别有什么作用.md)。

![Prompt 优化与前缀诱导流程](/images/content-rebuild/96e9ee43838eff6f/fig-2.png)

上图展示了自然语言松散提示与前缀诱导技术在生成起点上的对比。通过人为固定起始 Token，截断了模型输出问候语与 Markdown 标记的采样分支。

---

### 三、平台级机制：JSON Mode 与 Structured Outputs

主流模型 API 提供了结构化控制参数，常见的方式有两种：

| 特性 | 普通 JSON Mode | Structured Outputs (JSON Schema) |
| :--- | :--- | :--- |
| **启用方式** | `response_format={"type": "json_object"}` | `response_format={"type": "json_schema", "strict": true, ...}` |
| **语法保证** | 保证输出符合 JSON 语法规范 | 严格匹配给定的 JSON Schema 结构与字段类型 |
| **必需字段** | 无法保证 Schema 中定义的字段必然出现 | 必须包含所有声明在 `required` 中的字段 |
| **额外键名** | 可能输出未预期的键 | 禁止生成 Schema 之外的冗余键 |
| **实现方式** | 提示词注入结合通用 JSON 语法采样检查 | 将 Schema 转为下推自动机，在推理时进行受限解码 |

普通 JSON Mode 的作用是确保输出为合法的 JSON 文本，但并不约束字段的具体名称和数据类型。例如字段 `age` 仍可能被生成为字符串 `"28"`，或者丢失 `tags` 字段。

启用 `strict: true` 的 Structured Outputs 时，推理服务会将传入的 JSON Schema 编译为语法状态机，在模型生成每个 Token 时直接限制候选词集合。

![平台级结构化输出机制演进](/images/content-rebuild/96e9ee43838eff6f/fig-3.png)

上图展示了普通 JSON Mode 与严格 Structured Outputs 的处理分工。前者主要约束字符外壳，后者通过文法映射同时锁死字段骨架与类型。

---

### 四、受限解码原理：基于文法状态机的 Logits 屏蔽

如果在本地或私有服务器上部署开源模型（例如使用 vLLM、SGLang、llama.cpp、Outlines），可以直接在推理引擎层启用受限解码（Constrained Decoding）。

#### 1. 工作原理
在标准自回归生成中，时间步 $t$ 生成下一个 Token $y_t$ 时，模型首先根据上文计算整个词表 $\mathcal{V}$ 对应的未归一化得分（Logits）向量 $\mathbf{z}_t \in \mathbb{R}^{|\mathcal{V}|}$，再通过 Softmax 转换为采样概率：

$$P(y_t = v_i \mid y_{<t}) = \frac{\exp(z_{t, i})}{\sum_{j \in \mathcal{V}} \exp(z_{t, j})}$$

受限解码在 Softmax 之前引入了一个文法状态机（例如对应 JSON Schema 的确定性下推自动机）。执行步骤如下：

1. 状态机根据当前已经生成的 Token 序列 $y_{<t}$，计算出当前语法状态下允许接收的字符集合。
2. 推理引擎检查词表中的每个 Token，筛选出能够合法推进状态转移的 Token 子集 $\mathcal{V}_{\text{valid}} \subseteq \mathcal{V}$。
3. 对所有非法 Token 执行 **Logit 屏蔽（Logit Masking）**，将其 Logit 得分强制重置为负无穷大：

$$z_{t, i}' = \begin{cases} z_{t, i}, & \text{若 } v_i \in \mathcal{V}_{\text{valid}} \\ -\infty, & \text{若 } v_i \notin \mathcal{V}_{\text{valid}} \end{cases}$$

4. 在被掩蔽的向量 $\mathbf{z}'_t$ 上执行 Softmax。非法 Token 的采样概率变为 0，从而使生成的字符序列在词法和语法上完全符合要求。

```python
# 逻辑示意：在每个生成步根据合法 Token 集合过滤 Logits
def apply_grammar_mask(logits: list[float], valid_token_ids: set[int]) -> list[float]:
    masked_logits = []
    for token_id, score in enumerate(logits):
        if token_id in valid_token_ids:
            masked_logits.append(score)
        else:
            masked_logits.append(float("-inf"))
    return masked_logits
```

受限解码避免了生成完成后反复重试的延迟开销，但状态机维护和合法 Token 掩蔽计算会带来轻微的每步推理开销。针对复杂文法，需要对词表索引建立前缀树（Trie）进行加速。

![文法状态机驱动的 Logits 屏蔽机制](/images/content-rebuild/96e9ee43838eff6f/fig-4.png)

上图展示了自动机状态转移与 Token 采样的同步过程：当状态机迁移至键名闭合边界时（键名内容已匹配完成，下一个 Token 必须为闭合引号），词表中除双引号 `"` 之外的所有非法 Token 均被掩蔽为 $-\infty$。

---

### 五、应用层容错处理：解析器修复与重试反馈

如果使用的 API 不支持严格受限解码，或者生成过程因网络超时、上下文长度达到上限而被截断，则需要在应用层加入防御性解析与重试机制。

```
[模型原始响应]
       │
       ▼
 尝试标准 json.loads() ──────(成功)─────► 进入业务逻辑
       │
     (失败)
       ▼
 容错解析器 (如 json_repair 修复未闭合括号/单引号)
       │
       ├─────(修复成功)───► Pydantic 模型校验
       │                         │
       ▼                       (通过)───► 进入业务逻辑
 提取 ValidationError 错误原因    │
       │                       (失败)
       ▼                         │
 组装带错误信息的 Prompt，发起重试 ◄┘
```

#### 1. 容错解析器（json_repair）
当文本因为 Token 耗尽被截断时，末尾往往缺少闭合的大括号或引号。容错解析工具（如 Python 社区的 `json_repair`）通过维护符号栈，检测未闭合状态并在末尾补全必要的括号，同时剥离 Markdown 围栏：

```python
import json
from json_repair import repair_json

# 模拟一段带有 Markdown 且末尾截断的响应
raw_output = '```json\n{"name": "张三", "age": 28, "tags": ["Python",' 

# 自动修剪围栏并补全闭合符号
repaired_str = repair_json(raw_output)
parsed_data = json.loads(repaired_str)
# 解析结果: {'name': '张三', 'age': 28, 'tags': ['Python']}
```

![容错解析器状态栈修复过程](/images/content-rebuild/96e9ee43838eff6f/fig-5.png)

上图展示了解析器通过维护符号栈检测未闭合状态，并在序列末端自动压入缺失字符的过程。

#### 2. Pydantic 校验与带错重试
即使 JSON 语法有效，字段值也可能违反业务规则（例如 `age` 小于 0，或缺少必填字段）。使用 Pydantic 可以对数据进行类型与范围校验。当校验失败时，将具体的错误描述反馈给模型并进行有限次数的重试：

```python
from pydantic import BaseModel, Field, ValidationError

class UserExtractionResult(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)
    tags: list[str]

try:
    # 假设模型输出了非法年龄值 {"name": "张三", "age": -5, "tags": []}
    user = UserExtractionResult.model_validate(parsed_data)
except ValidationError as err:
    # 提取机器可读的错误信息，组织新的 Prompt 重试
    error_details = err.errors()
    print(f"校验失败，原因：{error_details}")
```

关于重试与验证环节的控制设计，可参考 [/frontier/loop-engineering/Loop_的灵魂不是执行_而是验证.md](/frontier/loop-engineering/Loop_的灵魂不是执行_而是验证.md)。

![生产级容错兜底架构闭环](/images/content-rebuild/96e9ee43838eff6f/fig-6.png)

上图展示了从原始文本接收、语法容错修复、类型强校验到带错重试的完整闭环，确保下游系统不受异常输入崩溃。

---

### 六、技术方案选型建议

在实际工程落地时，可以根据部署环境选择对应的方案组合：

1. **商业 API 服务**：优先使用官方提供的 Structured Outputs（如设置 `strict: true` 的 JSON Schema 参数）。该方式省去了应用层编写脆弱正则或语法修补的开销，但在面对生成长度超限截断或网络异常时，仍需在应用层保留截断判断与请求重试机制。
2. **私有化与开源模型部署**：在推理框架层（如 vLLM 或 llama.cpp）配置文法约束（BNF 或 Schema）。在生成过程中进行 Logit 掩蔽，从源头防止非法字符产生。
3. **缺少结构化参数的通用接口**：采用 TypeScript 接口定义 + Assistant 角色前缀诱导，并在应用层配合 `json_repair` 与 Pydantic 进行后置校验与带错重试。

---

### 七、课后检验练习

1. **验证前缀诱导效果**：使用任意支持消息预填充的 API，对比在 `assistant` 消息中留空与预填 `{\n  "` 时，模型输出是否还包含 Markdown 围栏或问候语。
2. **编写容错处理流水线**：编写一段 Python 测试代码，传入一段故意破坏的字符串（例如单引号字典、未闭合的数组），先尝试使用 `json.loads` 解析，捕获异常后调用 `repair_json` 修复并由 Pydantic 进行类型校验，观察修复情况。
