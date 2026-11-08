---
title: 大模型为什么会复读、早停与偏离要求？自回归生成常见问题解析
icon: mdi:alert-circle-outline
tag:
- 推理与生成
- 幻觉
- 安全兜底
description: 解析大语言模型自回归生成中的退化重复、意外早停、长度截断、目标漂移与事实幻觉，掌握 finish_reason 排查与惩罚参数配置。
---

在掌握了 [/fundamentals/inference-and-generation/inference-pipeline.md](/fundamentals/inference-and-generation/inference-pipeline.md) 的推理流程与 [/fundamentals/inference-and-generation/temperature-topk-topp.md](/fundamentals/inference-and-generation/temperature-topk-topp.md) 的采样策略之后，开发者在调用模型时常会遇到一些反常现象：提示词看似清晰、参数看似合理，但模型输出仍然可能出现严重异常。

大模型在生成文本时采用自回归（Autoregressive）机制：模型每次根据已有的上下文预测下一个 Token 的概率分布，采样出一个词后，将该词追加到上下文末尾，再继续预测下一步。**模型在单次生成中没有全局重构历史的能力，任何一步出现的微小偏差，都会变成下一步计算时既成的上下文事实。**

以一个常见的后台任务为例：假设我们要求模型根据几条零散的运维日志，整理一份“用户系统故障排查报告，严格用 Markdown 表格整理，包含模块名、错误码、修复建议三列”。在实际运行中，任务可能会以各种不同形式偏离预期：

- 模型连续输出“检查网关配置……检查网关配置……检查网关配置……”陷入死循环；
- 只返回了一句“收到，已整理。”便提前结束；
- 表格画到一半突然中断，留下残缺的文本；
- 前两百字严格按表格输出，写到后文逐渐抛弃表格格式，甚至开始以客服口吻闲聊；
- 流畅地编造出了一个系统中根本不存在的微服务名和配置项。

本文围绕自回归解码过程，拆解退化重复、意外早停与截断、长程目标漂移以及事实幻觉等常见生成问题，帮助开发者建立基于运行机制的排查思路。

![大模型自回归生成常见缺陷全貌封面图](/images/content-rebuild/5e432fd425946c6b/cover.png)

---

## 退化重复：自强化的正反馈循环

退化重复是指模型在解码过程中不断输出相同的词、短语甚至整段句子，例如反复输出“因此我们需要注意，注意，注意……”或机械式地循环输出同一段代码。

根据自回归条件概率公式：

$$P(X) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

在预测第 $t$ 步的 Token 时，输入包含前 $t-1$ 步的所有历史，其中也包括模型在前面自己生成的词。

![自回归解码中退化死循环的正反馈形成机制](/images/content-rebuild/5e432fd425946c6b/fig-1.png)

### 重复现象是如何自强化的？

假设在第 $k$ 步，由于局部的句法惯性或采样随机性，模型输出了某个词组 $S$（例如“需要注意”）。

此时，上下文变成了 $[\dots, S]$。在随后的解码中，如果上下文再次接续输出了 $S$，上下文中便连续出现了两个相邻的相同模式 $[\dots, S, S]$。

这种结构会引发注意力计算的正反馈：
1. **局部模式匹配增强**：在自注意力（Self-Attention）计算中，第二个 $S$ 的 Query 向量在检索前面的上下文时，会与第一个 $S$ 的 Key 向量计算出较高的注意力分数。
2. **接续概率被动放大**：在预训练数据中，重复出现的词组后面往往紧跟着特定连词或该词组本身的延伸。模型根据统计规律判断“重复上一次出现的模式”具有较高的局部合理性，使得紧随其后的候选词概率进一步集中在与 $S$ 相关的模式上。
3. **低随机性放大死循环**：在贪婪搜索（Greedy Search，即 Temperature = 0）或极小 Top-k 截断下，解码器每一步都机械选择当前概率最高的词。一旦重复模式在局部取得微弱优势，模型便很难通过随机采样跳出这一循环，从而出现连续的机械复读。

---

## 重复惩罚算法：Presence 与 Frequency Penalty

为了打破这种自强化循环，主流推理框架（如 vLLM）和主流 API 服务在执行 Softmax 之前，提供了对已出现 Token 扣减 Logit 的惩罚机制。

![Presence Penalty 与 Frequency Penalty 对 Logits 惩罚的数学差异对比](/images/content-rebuild/5e432fd425946c6b/fig-2.png)

以 OpenAI API 等服务采用的加法惩罚为例，未归一化的原始 Logit $z_i$ 按如下规则调整为 $z'_i$：

$$z'_i = z_i - \left( \alpha_{\text{freq}} \cdot c_i + \alpha_{\text{pres}} \cdot \mathbb{I}(c_i > 0) \right)$$

其中：
- $c_i$ 表示 Token $i$ 在当前已生成回复中出现的累计次数；
- $\mathbb{I}(c_i > 0)$ 为指示函数，只要 Token $i$ 至少出现过一次，取值为 1，否则为 0；
- $\alpha_{\text{freq}}$ 为 **Frequency Penalty（频率惩罚系数）**；
- $\alpha_{\text{pres}}$ 为 **Presence Penalty（存在惩罚系数）**。

> **注**：在部分开源推理框架（如 Hugging Face Transformers）中，还存在乘法形式的 `repetition_penalty`。加法惩罚是目前商业 API 中最常见的配置形式。

### 两种参数的作用差异与选型

1. **Presence Penalty（存在惩罚）**：
   - **机制**：一次性阶跃扣减。无论某个 Token 出现过 1 次还是 20 次，只要在回复中出现过，其 Logit 就会被扣除固定的 $\alpha_{\text{pres}}$。
   - **适用目标**：促使模型在生成时尝试未出现过的新词汇与话题，适合需要拓展思路的头脑风暴或创意写作。
2. **Frequency Penalty（频率惩罚）**：
   - **机制**：随出现次数线性累积扣减。Token 每重复出现一次，被扣减的分值就增加 $\alpha_{\text{freq}}$。
   - **适用目标**：抑制局部高频复读和口癖。某个短语重复次数越多，其后续候选概率被压制得越低，从而给其他替代词胜出的机会。

### 惩罚参数过大的副作用

惩罚并非越大越好，典型 API 的推荐范围通常在 0.1 到 0.5 之间。

- 如果把 `presence_penalty` 设得过高（例如 > 1.5），模型会极力避免使用任何已经出现过的词汇。这会导致代词（“它”、“这”）、常见介词和当前话题的核心实体名词都被抑制，模型被迫去翻找同义词或冷僻词，甚至造成句式断裂。
- 如果把 `frequency_penalty` 设得过高，在编写代码、SQL 或生成 JSON 等结构化文本时，模型会因拒绝重复使用花括号 `{}`、引号 `"` 或关键词（如 `return`、`SELECT`）而直接破坏语法结构。

### 惩罚计算代码示意

以下 Python 示例展示了加法惩罚如何改变 Logits 和最终概率分布：

```python
import math

def softmax(logits):
    max_val = max(logits.values())
    exps = {k: math.exp(v - max_val) for k, v in logits.items()}
    total = sum(exps.values())
    return {k: round(v / total, 4) for k, v in exps.items()}

# 模拟当前步模型的原始 Logits
raw_logits = {"网关": 4.2, "配置": 3.5, "异常": 2.8, "超时": 2.1, "建议": 2.0}

# 假设当前已生成的文本中词频统计如下
token_counts = {"网关": 3, "配置": 1, "异常": 0, "超时": 0, "建议": 0}

def apply_penalties(logits, counts, alpha_freq=0.0, alpha_pres=0.0):
    adjusted = {}
    for token, z in logits.items():
        c = counts.get(token, 0)
        has_appeared = 1 if c > 0 else 0
        z_prime = z - (alpha_freq * c + alpha_pres * has_appeared)
        adjusted[token] = z_prime
    return adjusted

print("原始概率分布:", softmax(raw_logits))
# 未经惩罚时，“网关”概率约 50.6%

# 施加适度惩罚
penalized = apply_penalties(raw_logits, token_counts, alpha_freq=0.8, alpha_pres=0.5)
print("适度惩罚后:", softmax(penalized))
# “网关”的高频优势被削减，“异常”和“超时”获得竞争机会

# 施加过高惩罚
over_penalized = apply_penalties(raw_logits, token_counts, alpha_freq=2.5, alpha_pres=2.0)
print("过度惩罚后:", softmax(over_penalized))
# 出现过的词概率几乎归零，原本不相关的词被强行拉高
```

惩罚机制通过在 Logit 层面引入负向偏移，人工降低高频已出现词的采样竞争力，从而给多样化文本腾出空间。

<ReadMoreLock />

---

## 意外早停与截断：识别 finish_reason

在接口调用中，有时模型还没写完就突然停下，有时只输出了两三个字便草草收尾。这两种情况的底层诱因完全不同，排查时的第一步是查看响应包元数据中的 `finish_reason`。

![生成终止原因的排障分支决策树](/images/content-rebuild/5e432fd425946c6b/fig-3.png)

### 1. 意外早停（finish_reason = 'stop'）

当 `finish_reason` 为 `stop` 时，意味着本次生成是由模型主动输出的结束符（EOS Token，如 `<|endoftext|>`、`</s>`）或命中开发者设定的停止词（Stop Sequences）而正常退出的。

如果内容明显未完成却返回了 `stop`，通常有以下三类诱因：

- **指令冲突或缺乏引导**：如果提示词中包含了互相矛盾的限制，或者在低温度下模型对后续词的置信度普遍极低，模型有可能将生成结束符判定为当前最稳妥的路径。
- **微调数据中的长度偏差**：如果模型在指令微调（SFT）阶段接触了大量短平快的问答语料（例如直接给答案而不展开说明），模型便容易在回答出核心词句后立刻偏向输出结束符。
- **停止词列表配置误伤**：在调用 API 时，如果传入了宽泛的停止词（例如将换行符 `\n` 或常见逗号句号设为了 `stop`），模型只要一排版或断句就会触发强制终止。

### 2. 异常截断（finish_reason = 'length'）

当 `finish_reason` 为 `length` 时，表明模型**本身还打算继续生成，但碰到了最大 Token 数量的硬限制**。

在推理框架中，显存开销和单次计算预算受总序列长度限制（参见 [/fundamentals/inference-and-generation/kv-cache.md](/fundamentals/inference-and-generation/kv-cache.md) 与 [/fundamentals/inference-and-generation/context-length.md](/fundamentals/inference-and-generation/context-length.md)）。推理运行时维护着一个计数约束：

$$\text{已生成 Token 数量} \ge \text{max\_tokens}$$

一旦生成的 Token 数达到预设的 `max_tokens`（或系统上下文窗口满额），循环强行中断。截断不仅会导致句子断在半空中，在 [/fundamentals/inference-and-generation/structured-output.md](/fundamentals/inference-and-generation/structured-output.md) 场景下，还会导致 JSON 缺少闭合括号 `}` 或 Markdown 表格缺少底线，引发下游程序的解析异常。

---

## 目标漂移：长距离解码中的指令淡化

在撰写长篇报告或多步骤推理时，模型常出现前密后疏、格式崩塌的现象：开头严格按照提示词规范输出 Markdown 表格，但写到数千字之后，格式逐渐瓦解，甚至丢弃了所有预设的角色和约束。这种现象称为**目标漂移（Goal Drift）**。

![长距离解码过程中的注意力分散与近邻偏置机制](/images/content-rebuild/5e432fd425946c6b/fig-4.png)

目标漂移并不是模型物理丢弃了开头的输入，而是长序列自注意力计算中，初始指令对当前步解码的影响力受到了客观削弱：

1. **长程检索干扰与候选竞争**：Softmax 在数学上具备指数放大极值的特性，只要匹配内积足够大，序列增长并不必然摊薄特定关键 Token 的权重。然而在长程自回归生成中，随着中间文本大幅累积，KV 缓存中引入了大量局部语义相关的内容；如果当前解码步的 Query 无法对开头的初始指令保持显著压倒性的匹配高分，初始指令就容易在众多中间键值向量的竞争中受到检索干扰，导致模型对初始约束的关注度出现波动或分散。
2. **近邻偏置与位置影响**：在主流预训练语料和因果掩码的共同作用下，语言生成具有显著的局部连续性：下一个词往往与刚写完的几个词相关度最高。在长序列中，模型受紧邻前文的风格与叙述惯性驱动较强，对数千步之前的全局提示词逐渐钝化。
3. **局部违规的滚雪球效应**：一旦长距离生成中局部出现了一次轻微的格式瑕疵（例如漏写了一个表格分隔符 `|`），这个错误就会成为后续生成的新上下文。自回归机制会继续顺着这个有缺陷的局部模式往下推断，导致格式进一步瓦解。

关于长文本在系统工程层面的性能考量与优化，可参考 [/frontier/long-context-app/long-context-pitfalls.md](/frontier/long-context-app/long-context-pitfalls.md)。

---

## 幻觉的生成层机制：知识混淆与连贯性逼近

幻觉（Hallucination）通常指模型生成了与客观事实不符或与输入材料冲突的陈述。除了外部信息缺失外，自回归模型本身的概率目标也是诱发幻觉的重要因素。

![生成层内生性幻觉的两大因果机制对比](/images/content-rebuild/5e432fd425946c6b/fig-5.png)

### 1. 参数关联知识混淆

大模型并不像关系型数据库那样以独立表项存储知识，而是将大量统计关联隐式编码在全连接网络（如 MLP 层）的权重中。如果两个实体在预训练文本中频繁出现在极为相似的上下文语境里（例如同一技术领域的两位开源作者，或两个命名相近的系统接口），模型在激活对应表征时可能产生混淆。

如果当前步采样出现轻微抖动，挑中了同类相近但事实错误的实体，一旦该错误词被写入已生成历史，后续的自回归解码就会基于这个错误的实体继续往下推演，形成一段前后看似自洽但主语完全错误的陈述。

### 2. 连贯性优先于事实性

大语言模型的优化目标通常是最大化文本序列的自回归对数似然度：

$$\mathcal{L} = -\sum_{t} \log P(x_t \mid x_{<t}; \theta)$$

这一目标鼓励模型预测出在语法、修辞和上下文过渡上最自然的下一个词，但模型内部并没有一个专门验证“现实世界客观真伪”的独立判定逻辑。

当提问涉及一个模型未曾见过的罕见概念或虚构名称时，模型虽然缺乏对该事实的参数记忆，却能识别出提问的句式结构。为了维持文本生成的流畅度与合理节奏，解码器会倾向于调用最常见的行文模板，顺畅地拼凑出看似严密的术语、日期、模块名甚至引用链接。

**大语言模型在本质上是在生成“在统计上最顺理成章的后续表达”，并不天然对事实真实性做出保证。**

---

## 生成问题排查对照表

在工程实践中，排查与定位生成异常可以参考以下处置清单：

![大模型生成异常现象与参数调优排障矩阵](/images/content-rebuild/5e432fd425946c6b/fig-6.png)

| 异常表现 | 优先排查项 | 可能的诱发原因 | 建议处置策略 |
| :--- | :--- | :--- | :--- |
| **退化死循环**<br>（反复输出相同短语） | 耗尽 Token 预算，`finish_reason = 'length'` | 采样策略过窄（如贪婪搜索），局部模式正反馈导致概率集中在重复词上。 | 1. 将 `frequency_penalty` 调至 0.2 - 0.5；<br>2. 适度提高 `temperature`（例如从 0 提至 0.7）并配置 `top_p`（0.8 - 0.95）。 |
| **半路截断**<br>（JSON 不完整、段落中断） | `finish_reason = 'length'` | 客户端配置的 `max_tokens` 阈值低于实际生成所需的 Token 量。 | 1. 调大调用请求中的 `max_tokens`；<br>2. 若生成体量巨大，改用多阶段生成或流式输出机制，参见 [/fundamentals/inference-and-generation/streaming.md](/fundamentals/inference-and-generation/streaming.md)。 |
| **意外早停**<br>（仅回几个字即退出） | `finish_reason = 'stop'` | 命中错误配置的停止词；或提示词指令冲突诱发 EOS 高置信度。 | 1. 检查 API 的 `stop` 参数，确认未误传换行或通用标点；<br>2. 优化 Prompt，明确拆解回答步骤，引导起手式内容。 |
| **长程目标漂移**<br>（后文格式完全失效） | 文本长度达数千字，前后结构严重不对称 | 局部近邻偏置和长程自注意力分散，弱化了开头的全局指令权重。 | 1. 将单次长文生成任务拆解为多步管道任务；<br>2. 结合结构化输出模式（如 JSON Schema 校验），参见 [/fundamentals/inference-and-generation/structured-output.md](/fundamentals/inference-and-generation/structured-output.md)。 |
| **事实性幻觉**<br>（语言流畅但内容虚构） | 关键参数、实体名或外部引用查无此据 | 预训练知识混淆，或模型在缺乏事实支撑时优先选择统计上通顺的模板。 | 1. 关键事实引入检索增强（RAG）或工具调用查询；<br>2. 降低 `temperature`，并在提示词中明确授权“信息不足时直接承认未知”。 |

了解这些问题的生成机制后，开发者可以将模糊的“模型输出不稳定”转化为明确的参数微调和工程架构设计，为后续评估指标与质量监控（如 [/llm-100/evaluation/generation-quality-metrics.md](/llm-100/evaluation/generation-quality-metrics.md)）打下清晰的基础。
