---
title: 上下文长度、Token 预算与 API 计费到底怎么算？
icon: mdi:calculator-variant-outline
tag:
- Token与分词
- 上下文长度
- 费用计算
description: 系统拆解大模型上下文窗口的物理边界、Prefill与Decode非对称计费机理，并提供对话截断算法与预算预检方案。
---

在调用大语言模型构建应用时，开发者经常会遇到两类典型问题：一是程序抛出 `400 Bad Request: context_length_exceeded`，或者模型输出在句子中途突兀中断，返回 `finish_reason: "length"`；二是月末账单上的费用往往超出最初按“字符数乘单价”估算出的数值。

理解这些现象，需要理清大语言模型运行时的物理约束：**模型始终在严格的离散 Token 预算下执行前向推断**。上下文窗口限定了模型单次处理与生成的最大 Token 总数，决定了请求是否合法；同时在典型服务场景下，硬件处理输入（Prompt）与输出（Completion）时的并发吞吐和资源开销特点不同，也是供应商制定非对称计费策略的重要参考依据。

本文从上下文窗口的底层约束出发，拆解模型计费公式，建立可落地的上下文配额划分策略，并提供基于离散 Token 计数的成对截断算法与前置校验规范。

![上下文预算管理与费用测算全景图，展示窗口硬约束、动态配额与计费链路](/images/content-rebuild/2ccb784431ae5a5c/cover.png)
*图 1：大模型上下文预算由系统指令、多轮历史、检索证据与生成缓冲共同组成，受底层最大位置编码刚性约束，并由输入输出非对称计费模型计量。*

---

## 上下文窗口的物理约束：Prompt 与 Completion 的总和边界

大语言模型的上下文窗口（Context Window），是指模型在单次自回归前向计算中能够支持的最大序列长度 $L_{max}$。这里的基本计量单位是分词后的离散 Token，而不是自然语言的字符或汉字。

### 为什么上下文长度存在硬性上限？

上下文长度不是云服务商随意设定的业务额度，而是由模型结构与硬件物理特性共同决定的：
1. **位置编码（Positional Encoding）**：无论是早期的绝对位置编码，还是主流的 RoPE（旋转位置编码），在预训练时都会设定一个基准上下文长度。虽然可以通过位置插值、YaRN 等外推方法扩展，但任意一个部署完毕的模型服务实例，其单次推理能够分配的最大位置索引范围是确定的。
2. **显存与 KV Cache（键值缓存）**：在标准自注意力计算中，序列长度为 $L$ 时，Attention 矩阵的基础显存与计算开销对应 $O(L^2)$ 复杂度。在逐字自回归解码阶段，系统必须在显存中常驻每一层、每个注意力头的 Key 和 Value 张量（KV Cache）。硬件显存容量决定了并发服务下能够容纳的最大 Token 总长度。

### 核心约束等式与请求校验

在单次推理会话中，请求必须满足以下基本不等式：

$$L_{prompt} + L_{completion} \le L_{max}$$

其中：
* $L_{prompt}$ 是送入模型的输入 Token 总数，包含 System Prompt、开发者提示词、RAG 检索证据、历史对话上下文以及当前轮次的用户提问。
* $L_{completion}$ 是模型在自回归解码中实际生成的输出 Token 总数。
* $L_{max}$ 是模型实例支持的物理上限（如 4,096、8,192、32,768 或 128k）。

在发起 API 请求时，开发者通常会传递 `max_tokens` 参数（部分 SDK 命名为 `max_completion_tokens`），用来限制模型生成的**最大输出 Token 上限**。

![上下文窗口的物理硬约束结构图，展示输入序列与输出预留占满最大位置编码](/images/content-rebuild/2ccb784431ae5a5c/fig-1.png)
*图 2：总上下文标尺被 Prompt 与预留输出共同占用。当两者的请求总和超出模型支持的物理最大位置编码上限时，推理请求将被底层立即阻断。*

这就带来了开发中最常见的报错：

> **场景推演**：若一个模型的物理上限 $L_{max} = 4,096$，当前 Prompt 经分词后实际消耗 3,500 Token，开发者在请求体中设置 `max_tokens = 1,000`，发起调用时会发生什么？

在主流 API 规范下，**该请求在进入解码阶段前就会被网关拒绝**。API 网关会检查输入与预留空间的总和：$L_{prompt} + \text{max\_tokens} = 3,500 + 1,000 = 4,500 > 4,096$。由于网关无法保证在不突破上下文上限的前提下满足用户对 1,000 Token 输出容量的承诺，服务端会返回 HTTP 400 错误：

```json
{
  "error": {
    "message": "This model's maximum context length is 4096 tokens. However, you requested 4500 tokens (3500 in the messages, 1000 in the completion). Please reduce the length of the messages or completion!",
    "type": "invalid_request_error",
    "code": "context_length_exceeded"
  }
}
```

若部分宽容网关自动将 `max_tokens` 收缩至剩余可支配空间（$4,096 - 3,500 = 596$ Token），只要模型在生成第 596 个 Token 时仍未输出自然结束符（如 `<|endoftext|>` 或 `<|im_end|>`），生成过程就会被强行终止，返回 `finish_reason = "length"`，造成输出内容被阶段性截断。

关于 Token 的离散本质与切分原理，可参考基础教材 [/fundamentals/token-and-tokenizer/what-is-token.md](/fundamentals/token-and-tokenizer/what-is-token.md)。

---

## API 计费规则：输入输出不对称与多轮累加

大模型 API 通常按照 Token 数量计费（如“每 100 万 Token（1M Tokens）”或“每 1,000 Token（1k Tokens）”）。单次调用的总费用公式如下：

$$\text{Cost}_{total} = L_{input} \times P_{input} + L_{output} \times P_{output}$$

其中 $P_{input}$ 为输入单价，$P_{output}$ 为输出单价。

### 为什么商业 API 的输出单价通常显著高于输入单价？

API 的具体定价标准由各云服务商公布，但在主流供应商的价目表中，输出 Token 单价通常是输入单价的数倍。这种定价差异与大模型服务在两个阶段的典型硬件吞吐特征密切相关：

1. **Prefill（预填充）阶段处理输入**：输入 Prompt 是一次性送入模型的，在大块矩阵运算下，GPU 更容易利用张量核心（Tensor Cores）进行并发计算。在常见服务场景下，该阶段通常更接近**算力受限（Compute-Bound）**，吞吐效率相对较高，摊薄了单个输入 Token 的设备时间。
2. **Decode（解码）阶段处理输出**：大模型生成输出采用逐步自回归机制。每生成 1 个新 Token，模型就需要执行一轮前向计算，并在常规批大小下将此前所有 Token 的 KV Cache 从显存（HBM）读入片上高速缓存（SRAM）。由于单步算术计算强度（FLOPs/Byte）较低，硬件往往处于**显存带宽受限（Memory Bandwidth-Bound）**状态，整体生成吞吐远低于预填充阶段，单 Token 占用的硬件服务时间与资源开销明显更高。

![Prefill阶段与Decode阶段的计算特征对比图，解释API输入与输出定价不对称原因](/images/content-rebuild/2ccb784431ae5a5c/fig-2.png)
*图 3：Prefill 阶段利用矩阵并发吞吐处理全部输入，Decode 阶段受制于自回归逐字读写 KV Cache 的显存带宽瓶颈，使得输出阶段的单 Token 耗时通常更高，成为服务商制定非对称定价的重要成本动因。*

### 多轮对话的 Token 二次方增长

在无状态的 HTTP API 接口下，模型服务端不会替客户端长期保存会话上下文。为了维持多轮对话的连贯性，**每一次交互都必须将此前所有的历史对话重新打包发送给 API**。

假设施加的系统提示词为 $S$ 个 Token，用户每轮提问均为 $U$ 个 Token，模型每轮回答均为 $A$ 个 Token：

* **第 1 轮**：输入 $S + U$，输出 $A$
* **第 2 轮**：输入 $S + U + A + U$，输出 $A$
* **第 3 轮**：输入 $S + 2(U + A) + U$，输出 $A$
* **第 $k$ 轮**：输入 $L_{input}(k) = S + k \cdot U + (k - 1) \cdot A$

如果连续进行 $N$ 轮对话，累计送入的**总输入 Token 量**为等差数列求和：

$$\sum_{k=1}^{N} L_{input}(k) = N \cdot S + U \cdot \frac{N(N + 1)}{2} + A \cdot \frac{N(N - 1)}{2}$$

输入 Token 总量与轮数 $N$ 呈现**二次方关系（$O(N^2)$）**。在多轮对话中，即使单次提问并不长，反复重传的历史记录也会在数轮之后占据输入消耗的绝大部分，大幅推高计费成本。

<ReadMoreLock />

---

## 动态上下文预算分配：配额划分与安全预留

为了防止运行时上下文溢出并确保输出完整，系统在拼装请求内容时需要对上下文窗口进行分层规划：

$$
L_{max} \ge L_{fixed} + L_{reserved} + L_{dynamic} + L_{elastic}
$$

各组成部分的作用与淘汰优先级划分如下：

| 配额区间 | 构成内容 | 调整优先级 | 策略说明 |
| :--- | :--- | :--- | :--- |
| **固定区间 ($L_{fixed}$)** | System Prompt、模型角色设定、核心约束与输出格式说明 | 最高 | **严格保留**。编写时需精简表述，剔除冗余修饰词。 |
| **工具声明区 ($L_{tools}$)** | Function/Tool Schema 描述（JSON 格式参数定义） | 高 | **按需挂载**。根据当前任务路由仅载入对应工具，避免全量堆叠。 |
| **安全生成区 ($L_{reserved}$)** | 为模型生成预留的缓冲空间（即 `max_tokens` 参数） | 高 | **硬性预留**。预留不足会造成半句截断（`length`），预留过大会挤压输入空间。 |
| **当前输入区 ($L_{dynamic}$)** | 当前轮次用户的直接提问、上传的即时日志或文本 | 中 | **前置校验**。若当前单次输入自身超出可用上限，在网关层予以拦截并提示用户。 |
| **弹性区间 ($L_{elastic}$)** | RAG 检索召回的文档切片、多轮历史对话记录 | 低 | **动态淘汰**。空间不足时，按相关性淘汰 RAG 切片，按轮次滑动淘汰旧对话。 |

![上下文预算动态分配堆叠图，展示固定配额、弹性配额与保底生成的层级](/images/content-rebuild/2ccb784431ae5a5c/fig-3.png)
*图 4：垂直层级结构明确划分了预算分配优先级，核心指令与输出缓冲享有刚性保障，中间的多轮历史与外部检索证据在空间吃紧时执行弹性淘汰。*

### 预算分配具体算例

以物理窗口 $L_{max} = 8,192$ 的模型构建工单助手为例：
1. **安全生成预留 ($L_{reserved}$)**：根据预期业务输出长度，设定 `max_tokens = 1,024`。此时输入总预算上限被锁定为 $8,192 - 1,024 = 7,168$ Token。
2. **核心指令与工具 ($L_{fixed} + L_{tools}$)**：经精确分词，System Prompt 占用 420 Token，工具描述 Schema 占用 580 Token，合计固定占用 1,000 Token。
3. **剩余可用输入配额**：$7,168 - 1,000 = 6,168$ Token。
4. **当前用户提问 ($L_{dynamic}$)**：用户本轮发送了一段工单文本，分词计数为 1,168 Token。
5. **弹性预算空间 ($L_{elastic}$)**：留给“RAG 检索切片”与“多轮历史”的总额为 $6,168 - 1,168 = 5,000$ Token。

若本轮检索召回了 4 篇相关文档共消耗 3,200 Token，则留给多轮对话历史的空间仅剩 $5,000 - 3,200 = 1,800$ Token。此时裁剪程序必须运行，将历史消息压缩在 1,800 Token 以内。

关于全局 Token 成本控制体系，可参考进阶架构篇 [/architecture/token-cost.md](/architecture/token-cost.md)；对于代理循环调用导致 Token 异常激增的防御，请参考专题分析 [/frontier/loop-engineering/Loop_Engineering_的坑_无限循环_烧_Token_越改越差.md](/frontier/loop-engineering/Loop_Engineering_的坑_无限循环_烧_Token_越改越差.md)。

---

## 历史上下文截断算法：为什么字符数估算法不可靠？

在编写上下文管理逻辑时，常见的简易做法是按字符数粗略折算（例如假设“英文 4 字符折算 1 Token，中文 1 汉字折算 1 Token”）。在生产代码中，**直接依据字符数截断容易引发意料之外的溢出或截断过度**。

### 字符估算的失效机理

分词器（Tokenizer）采用基于统计数据的子词切分算法（如 BPE、WordPiece、Unigram，可查阅 [Hugging Face tokenizers 官方文档](https://huggingface.co/docs/transformers/tokenizer_summary)）。文本类型不同，字符到 Token 的折算比差异极大：

1. **生僻字与多字节 UTF-8 编码**：常用汉字通常在词表中占 1 个独立 Token，而低频字或专业术语在字节级 BPE（Byte-level BPE）中会被拆分为 2 到 3 个字节 Token。关于中文分词的详细机制，可阅读 [/fundamentals/token-and-tokenizer/chinese-tokenization.md](/fundamentals/token-and-tokenizer/chinese-tokenization.md)。
2. **代码缩进与空白符**：连续空格、制表符（`	`）、换行符（`
`）在不同分词器中的合并规则不一，有时单个空白字符就是一个 Token，导致 Token 数量明显多于常规文本。
3. **标点与特殊符号**：JSON 结构中的引号、括号、转义符号以及 Emoji 可能会显著增加切分后的 Token 数量。

若按字符长度估算，在遇到分词膨胀率高的样本时，真实 Token 数量很容易突破安全裕度，导致 API 报错；而在膨胀率低的纯英文样本中，又会过早裁剪，丢弃原本可以容纳的上下文。

![字符估算法与离散Token分词截断算法对比示意图，展示字符计数失效案例](/images/content-rebuild/2ccb784431ae5a5c/fig-4.png)
*图 5：字符估算策略无法感知离散子词的真实切分规则，易在生僻字、排版缩进和特殊代码中引发意外溢出；离散 Token 裁剪能确保物理边界严格收敛。*

### 基于成对逻辑的历史截断实现

在标准的 Chat 接口中，消息以 `{"role": "user", ...}` 和 `{"role": "assistant", ...}` 的形式交互组织。编写截断逻辑时需遵守两项原则：
* **保留 System Prompt**：索引为 0 的系统设定始终保留。
* **成对淘汰（User + Assistant）**：不宜只保留 Assistant 回答而裁掉对应的 User 提问，否则上下文将缺乏语义前提；裁剪应从最早的完整问答对开始向前淘汰。

以下是基于离散 Token 计数的截断参考实现：

```python
from typing import List, Dict
import tiktoken

def truncate_chat_history(
    messages: List[Dict[str, str]],
    max_history_tokens: int,
    model_name: str = "gpt-4o"
) -> List[Dict[str, str]]:
    """
    按照离散 Token 预算裁剪多轮对话历史。
    保留开头的 System Prompt，按 (User, Assistant) 成对从旧到新依次淘汰。
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # 统计单条消息的精确 Token 消耗（包含角色标记与边界协议开销）
    def count_message_tokens(msg: Dict[str, str]) -> int:
        # 每条结构化消息通常包含 3-4 个协议控制 Token
        return len(encoding.encode(msg["content"])) + 4

    if not messages:
        return []

    system_message = None
    dialogue_messages = []

    # 提取不可淘汰的 System Message
    if messages[0]["role"] == "system":
        system_message = messages[0]
        dialogue_messages = messages[1:]
    else:
        dialogue_messages = messages

    system_tokens = count_message_tokens(system_message) if system_message else 0
    if system_tokens > max_history_tokens:
        raise ValueError("System Prompt 自身 Token 消耗已超出分配的历史总预算")

    available_tokens = max_history_tokens - system_tokens
    
    preserved_messages: List[Dict[str, str]] = []
    accumulated_tokens = 0

    # 从最新消息向前逆序扫描，优先保留最近的对话
    i = len(dialogue_messages) - 1
    while i >= 0:
        current_turn_tokens = 0
        turn_pair = []

        # 识别成对的 (User, Assistant) 交互
        if i > 0 and dialogue_messages[i]["role"] == "assistant" and dialogue_messages[i-1]["role"] == "user":
            turn_pair = [dialogue_messages[i-1], dialogue_messages[i]]
            current_turn_tokens = count_message_tokens(dialogue_messages[i-1]) + count_message_tokens(dialogue_messages[i])
            step = 2
        else:
            # 单条孤立消息（如开局首个 User 提问）
            turn_pair = [dialogue_messages[i]]
            current_turn_tokens = count_message_tokens(dialogue_messages[i])
            step = 1

        if accumulated_tokens + current_turn_tokens <= available_tokens:
            preserved_messages = turn_pair + preserved_messages
            accumulated_tokens += current_turn_tokens
            i -= step
        else:
            # 预算不足以容纳更早的完整交互，终止回溯
            break

    # 组装最终结果
    final_messages = []
    if system_message:
        final_messages.append(system_message)
    final_messages.extend(preserved_messages)
    
    return final_messages
```

---

## 离线预检流程：避免请求到达远端后报错

在调用大模型 API 前，在应用本地执行离线分词预检，可以有效避免由 `context_length_exceeded` 引发的网络重试与耗时损失。

### 离线预检流程设计

![离线本地预校验与在线网络调用防护管道流程图](/images/content-rebuild/2ccb784431ae5a5c/fig-5.png)
*图 6：本地分词预校验作为前置网关，在发起物理请求前拦截超标载荷，执行针对性裁剪或告警，阻断下游无效网络消耗与报错。*

一个完整的 Token 预检流程通常包含四个步骤：

1. **分词器对齐（Tokenizer Alignment）**：必须加载与目标模型一致的分词器。例如调用 OpenAI 模型使用 `tiktoken`，调用开源 Qwen、Llama 系列模型使用 Hugging Face `tokenizers` 或通过 Transformers 库载入对应的 `AutoTokenizer`。关于分词器的实现机理，请参考 [/fundamentals/token-and-tokenizer/how-tokenizer-works.md](/fundamentals/token-and-tokenizer/how-tokenizer-works.md)。
2. **协议标记补偿（Protocol Overhead）**：Chat 模型在格式化多轮角色时会注入结构化标记（如 `<|im_start|>user
...<|im_end|>`）。计算输入 Token 时，需为每条消息叠加固定的协议开销（前文代码中设为 $+4$ 偏置），并为末尾引导符留出余量（通常为 $+2$ 到 $+3$）。
3. **安全裕度（Safety Margin）**：在本地计算得到的 $L_{prompt}$ 基础上，建议预留 16 到 32 个 Token 的安全浮动缓冲 $\Delta$，用于吸收不同 SDK 版本在处理特殊控制符时的微小差异。
4. **分流与熔断**：
   * **弹性修剪**：若仅是多轮历史或 RAG 证据超限，调用滑动窗口裁剪算法剔除旧数据；
   * **前端拦截**：若单次用户输入加上核心 System Prompt 本身就已超过 $L_{max} - L_{reserved}$，应在本地直接拒绝该请求，向用户提示精简输入，不向后端网络发起无效调用。

---

## 响应状态判定与指标监控

即使经过了前置配额管理，在实际生成过程中仍需根据服务端返回的状态元数据进行相应处理。

![模型生成终止原因判定与异常降级状态机图](/images/content-rebuild/2ccb784431ae5a5c/fig-6.png)
*图 7：推理服务响应后的状态分流机制，通过检测 finish_reason 区分正常自然收敛与越界截断，触发下游补偿流程。*

### 1. `finish_reason` 的含义与处理

API 响应体中的 `choices[0].finish_reason` 给出了生成的终止原因：
* `"stop"`：模型输出了结束符（EOS Token）或自然完成生成，属于正常状态。
* `"length"`：生成的 Token 总数触碰了 `max_tokens` 上限，或请求总长触及了物理窗口边界 $L_{max}$。此时输出文本不完整。应用检测到此信号时，可选择自动拼接提示词（“请继续”）发起追问，或通知用户缩小单次任务范围。
* `"tool_calls"` 或 `"function_call"`：模型生成了符合 Schema 的结构化工具调用参数，等待客户端执行外部函数。

### 2. 核心监控指标

在服务网关层，可以通过以下量化指标评估上下文使用状况：
* **Prompt 饱和度（Saturation）**：计算方式为 $\frac{L_{prompt}}{L_{max}}$。若系统在日常使用中饱和度持续高于 80%，表明当前模型的上下文容量偏紧，需要考虑切分任务、改进入库分块策略或选用更长窗口的模型。
* **截断发生率（Length Finish Rate）**：统计响应中 `finish_reason == "length"` 的调用比例。该值过高通常说明预留的 `max_tokens` 偏小，或者提示词未能有效引导模型输出凝练的答案。
* **单会话平均消耗（Tokens per Session）**：统计单次完整交互中输入和输出 Token 的实际消耗，用于分析长上下文下的成本分布。

长文本场景下的具体上下文管理方案，可查阅 [/llm-100/app-system/context-budget.md](/llm-100/app-system/context-budget.md)；分词与上下文在工程面试中的高频问题，请参考 [/interview/llm-interview/token-basics.md](/interview/llm-interview/token-basics.md) 与 [/interview/llm-interview/tokenization.md](/interview/llm-interview/tokenization.md)。

关于分词机制对大语言模型逻辑能力产生的深层影响，将在下一篇展开说明：[/fundamentals/token-and-tokenizer/tokenization-impact.md](/fundamentals/token-and-tokenizer/tokenization-impact.md)。
