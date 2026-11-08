---
title: Temperature、Top-P、Top-K 分别控制什么？
icon: mdi:message-text-outline
date: 2026-07-16
tag:
- 大模型100讲
- Prompt
description: 拆解大模型文本生成的采样参数：从未归一化分值到概率分布，理清温度缩放、Top-K 截断与 Top-P 核采样的计算顺序与工程调优。
---

在大语言模型生成文本时，模型每预测一个词（Token），神经网络最后一层都会输出一组与词表等长的打分向量——未归一化对数概率（Logits）。最终屏幕上输出哪一个词，不仅取决于模型预训练学到的权重，还取决于解码阶段的**采样策略（Sampling Strategy）**。

许多开发者在调用大模型 API 时，把 `temperature`、`top_p` 和 `top_k` 当作三个可以随意叠加的“发散度滑块”。但实际上，它们在大模型推理引擎中处于严格的先后执行管线中。若不理解背后的计算方式，调参时往往会发现参数相互抵消，或者让输出陷入重复、机械甚至乱码。

本讲的学习目标是：
1. 搞清 Logits 如何通过 Softmax 转化为概率分布；
2. 理解 Temperature、Top-K、Top-P 各自的作用对象、生效顺序与数学边界；
3. 掌握不同任务场景（如代码生成、结构化抽取、创意写作）下的调参策略，并通过可运行的代码完成验证。

![采样参数整体转换管线](/images/content-rebuild/be5b603061a67d94/cover.png)

---

## 贯穿示例：模型如何决定下一个词？

为了直观说明所有参数的作用，我们设定一个具体的贯穿任务：模型正在续写句子“今天天气真不错，我们一起去[ ? ]”。

假设词表极其简化，仅有 5 个候选 Token，神经网络最后一层输出的原始 Logits 分值如下：

| 候选 Token | 原始 Logit 分值 $z_i$ |
| :--- | :--- |
| **A: 公园** | 4.0 |
| **B: 散步** | 3.0 |
| **C: 图书馆** | 2.0 |
| **D: 火星** | 1.0 |
| **E: 香蕉** | 0.0 |

Logits 是实数范围的分值，不能直接当作概率使用。必须通过 Softmax 函数将其转换为非负且总和为 1 的概率分布：

$$P(w_i) = \frac{\exp(z_i)}{\sum_{j=1}^{|V|} \exp(z_j)}$$

![Logits 到 Softmax 映射过程](/images/content-rebuild/be5b603061a67d94/fig-1.png)

代入原始分值计算 Softmax：
- $\exp(4.0) \approx 54.60$
- $\exp(3.0) \approx 20.09$
- $\exp(2.0) \approx 7.39$
- $\exp(1.0) \approx 2.72$
- $\exp(0.0) = 1.0$
- 分母总和 $S \approx 54.60 + 20.09 + 7.39 + 2.72 + 1.0 = 85.80$

各词的基准概率为：
- $P(\text{公园}) = 54.60 / 85.80 \approx 63.64\%$
- $P(\text{散步}) = 20.09 / 85.80 \approx 23.41\%$
- $P(\text{图书馆}) = 7.39 / 85.80 \approx 8.61\%$
- $P(\text{火星}) = 2.72 / 85.80 \approx 3.17\%$
- $P(\text{香蕉}) = 1.00 / 85.80 \approx 1.17\%$

如果直接在全词表上按概率抽样，虽然“公园”和“散步”占了 87% 的概率，但荒谬的“香蕉”依然有 1.17% 的概率被抽中；在长文本生成中，连续推演成百上千步后，抽中低概率荒谬词的概率就会累积变大。相反，如果完全放弃抽样，永远只取概率最大的第一名（即贪心解码 Greedy Decoding），输出虽然绝对稳定，但在长文本中容易陷入词句自我重复的死循环。因此，我们需要通过参数来调控这个分布。

---

## Temperature：缩放分值，改变概率分布的平坦度

Temperature（温度参数 $T$）通过一个标量缩放因子，在计算 Softmax 之前对所有 Logits 进行全局等比缩放：

$$P_T(w_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^{|V|} \exp(z_j / T)}$$

<ReadMoreLock />

参数 $T$ 的取值通常在 $(0, 2]$ 之间（注：不同模型 API 取值范围不同，OpenAI API 支持 $[0, 2]$，而 Anthropic Claude API 上限为 $1.0$，传入大于 $1.0$ 会报错）：

1. **$T = 1.0$**：原始基准状态，不改变模型的打分比例。
2. **$0 < T < 1.0$（低温区）**：Logits 除以小于 1 的数，各分量之间的绝对分差被成倍拉大。例如当 $T = 0.5$ 时，A 与 B 的分差从 1.0 变成了 2.0。经过 $\exp$ 运算后，最高分词的概率急剧上升，长尾低分词的概率被压缩到趋近于 0。
3. **$T > 1.0$（高温区）**：Logits 除以大于 1 的数，各分量之间的差距被缩小。经过 $\exp$ 运算后，原本悬殊的概率差异被抹平，分布整体趋向均匀。

![Temperature 对分布形态与信息熵的控制](/images/content-rebuild/be5b603061a67d94/fig-2.png)

### 极限行为与工程注意点
- **当 $T \to 0^+$ 时**：最高分词的概率趋近于 1，其余词趋近于 0，等价于贪心解码。在实际 API（如 OpenAI 或 Anthropic）中，传入 `temperature=0` 时，推理引擎底层通常不会真做除法运算（防止浮点除零），而是直接短路切换为 `argmax` 算子选择最大值。
- **当 $T \to +\infty$ 时**：所有 $z_i / T \to 0$，$\exp(0) = 1$，所有词的概率都趋向于 $1/|V|$，退化为均匀随机抽样。
- **关键认知**：$T$ 改变的是概率分布的集中程度（信息熵），**它完全不改变候选词之间的大小先后次序**。只要 $T > 0$，原始分值最高的词在经过温度缩放后依然是第一名。

---

## Top-K：按名次硬截断长尾

单纯降低温度可以压低低分词的概率，但无法彻底把它们从候选池中剔除。在真实大模型包含 10 万词表的场景下，长尾词汇加起来仍然可能带来意外波动。

**Top-K 截断（Fan et al., 2018）** 的逻辑非常直接：**只保留概率排名前 $K$ 个的词，将其余词彻底剔除（将其 Logit 设为 $-\infty$ 或概率置零），再对保留的 $K$ 个词重新归一化。**

例如在我们的贯穿示例中，若设定 $K = 3$：
- 排序前 3 名是：A（公园）、B（散步）、C（图书馆）；
- 排名在 3 之后的 D（火星）和 E（香蕉）被直接剔除；
- 仅保留前 3 项重新计算概率归一化。

![Top-K 静态截断在不同分布下的局限](/images/content-rebuild/be5b603061a67d94/fig-3.png)

### Top-K 的核心局限
Top-K 的最大问题在于 **$K$ 是一个静态固定的整数，无法感知当前上下文的确定性差异**：
- **分布陡峭场景**：当模型预测常识或固定搭配时（例如“人工智能的英文缩写是 [ ? ]”），第一名“AI”的概率可能高达 98%。此时如果固守 $K = 40$，剩下的 39 个词累计概率不足 2%，强行把它们留在候选池中只会增加抽中无意义符号的风险。
- **分布平缓场景**：当模型面临发散任务时（例如“请列举一个描写春天的形容词 [ ? ]”），前 50 个形容词合情合理、概率极为接近。如果固定 $K = 5$，就会机械切断后 45 个同样优质的候选词。

因此，静态的 Top-K 适合作为防止系统踩坑的底层防护网，但难以独立完成自适应调节。

---

## Top-P（核采样）：按累积概率动态自适应截断

为了克服 Top-K 静态候选数量的不足，Holtzman 等人在 2019 年提出了 **Top-P（Nucleus Sampling，核采样）**。

Top-P 不再限定保留“几个词”，而是设定一个累积概率阈值 $p \in (0, 1]$：
1. 将候选词按概率从大到小降序排列；
2. 从高到低依次累加概率；
3. **一旦累加总和首次大于或等于 $p$，停止吸纳，仅保留当前这个最小集合（称为核集合 Nucleus）**；
4. 集合之外的其余词彻底剔除，并对核集合内部重新做归一化。

![Top-P 动态自适应截断机制](/images/content-rebuild/be5b603061a67d94/fig-4.png)

### 为什么 Top-P 具备自适应弹性？
沿用贯穿示例的直觉对比：
- **在高度确信的上下文中**：如果第一名候选词的自身概率达到了 92%，当我们设置 $p = 0.9$ 时，累加第一个词就已经跨过了 0.9 门槛，核集合仅包含这 1 个词，候选集自动收缩为贪心状态。
- **在高度开放的上下文中**：如果概率前 10 个词每个都只有约 9% 的概率，累积到 0.9 需要吸纳全部 10 个词，候选集自动扩张为 10 个词，充分保留自然语言的多样性。

> **调优直觉提示**：$p$ 越小，过滤越激进（候选词池越小、越确定）；$p$ 越接近 1.0，截断越少；$p = 1.0$ 时表示完全关闭核截断。

---

## 联合执行管线与完整数值算例

在真实推理框架（如 vLLM、Hugging Face Transformers 等）中，这几个参数不是并列执行的，而是组成了一条级联流水线。

### 1. 执行时序与前后依赖

标准推理执行流程如下：
$$\text{原始 Logits } \mathbf{z} \xrightarrow{/\, T} \text{温度缩放分值 } \mathbf{z}' \xrightarrow{\text{Top-K 截断}} \mathbf{z}'' \xrightarrow{\text{Softmax 概率}} \mathbf{P} \xrightarrow{\text{Top-P 截断}} \mathbf{P}' \xrightarrow{\text{重新归一化}} \mathbf{P}'' \xrightarrow{\text{多项采样}} \text{Token}$$

![Logits 联合采样执行管线时序](/images/content-rebuild/be5b603061a67d94/fig-5.png)

注意 **Temperature 与 Top-P 之间严格的前后因果关系**：
- Top-K 仅依赖候选词的分值排序次序。由于 $T > 0$ 时缩放是单调变换，不改变相对顺序，因此 Top-K 在温度缩放前或缩放后执行在数学结果上是一致的。
- **Top-P 则严重依赖温度缩放后的具体数值**。如果调高温度，概率分布被抹平，累积和达到 $p$ 所需吸纳的词数会急剧增加；如果降低温度，概率迅速向第一名集中，达到 $p$ 所需的词数会骤减。因此，**Top-P 必须在 Temperature 缩放并转换为 Softmax 概率之后执行**。

### 2. 逐步推演算例

以贯穿示例的 5 个 Token 为例：
- 原始 Logits：$\mathbf{z} = [\text{公园}: 4.0, \; \text{散步}: 3.0, \; \text{图书馆}: 2.0, \; \text{火星}: 1.0, \; \text{香蕉}: 0.0]$
- 设定参数：$T = 0.5$, $\text{Top-K} = 3$, $\text{Top-P} = 0.85$。

#### 第一步：Temperature 缩放（除以 0.5，相当于乘以 2）
$$\mathbf{z}' = [\text{公园}: 8.0, \; \text{散步}: 6.0, \; \text{图书馆}: 4.0, \; \text{火星}: 2.0, \; \text{香蕉}: 0.0]$$

#### 第二步：Top-K 截断（保留前 3 名，其余置为 $-\infty$）
$$\mathbf{z}'' = [\text{公园}: 8.0, \; \text{散步}: 6.0, \; \text{图书馆}: 4.0, \; \text{火星}: -\infty, \; \text{香蕉}: -\infty]$$

#### 第三步：计算 Softmax 概率
- $\exp(8.0) \approx 2980.96$
- $\exp(6.0) \approx 403.43$
- $\exp(4.0) \approx 54.60$
- 其余项 $\exp(-\infty) = 0$
- 分母 $S = 2980.96 + 403.43 + 54.60 = 3438.99$

归一化概率：
- $P(\text{公园}) = 2980.96 / 3438.99 \approx 0.8668$ (86.68%)
- $P(\text{散步}) = 403.43 / 3438.99 \approx 0.1173$ (11.73%)
- $P(\text{图书馆}) = 54.60 / 3438.99 \approx 0.0159$ (1.59%)
- $P(\text{火星}) = P(\text{香蕉}) = 0$

#### 第四步：Top-P 截断（阈值 $p = 0.85$）
从高到低累加：
- 考察第 1 个词“公园”：自身概率为 $0.8668$。
- 因为 $0.8668 \ge 0.85$，在第 1 个词就已经满足阈值要求，截断停止！
- 核候选集最终只收缩为 1 个词：$\{\text{公园}\}$。
- 重新归一化后，“公园”的抽中概率为 $100\%$。

这个算例清楚表明：即使 Top-K 保留了 3 个词，但低温度（$T=0.5$）将首位的优势大幅拉大，随后的 Top-P 能够自适应地把后面两个词进一步剥离，确保输出高度确定。

---

## 验证实现：可运行的采样演示代码

下面的 Python 代码通过 PyTorch 实现了上述完整的 Logits 处理管线。开发者可以在本地运行，观察各阶段张量数值的变化：

```python
import torch
import torch.nn.functional as F

def sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0
) -> int:
    """
    联合采样函数演示
    :param logits: 一维张量，未归一化分值，形状为 [vocab_size]
    :param temperature: 温度缩放因子
    :param top_k: 截断名次，0 表示不启用
    :param top_p: 核采样累积概率阈值，1.0 表示不启用
    """
    # 1. 应对贪心极端情况（防止浮点除零）
    if temperature <= 1e-4:
        return int(torch.argmax(logits))

    # 2. Temperature 缩放
    scaled_logits = logits.clone() / temperature

    # 3. Top-K 截断：将第 K 个之后的分值置为负无穷
    if top_k > 0:
        top_k = min(top_k, scaled_logits.size(-1))
        k_th_value = torch.topk(scaled_logits, top_k)[0][-1]
        scaled_logits[scaled_logits < k_th_value] = float('-inf')

    # 4. Top-P (核采样) 截断
    if 0.0 < top_p < 1.0:
        # 降序排序以计算累积概率
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 找出累积概率超过 top_p 的多余 Token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 向右移一位，保证刚好超过阈值的第一个 Token 仍被保留
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        # 映射回原始张量维度，将被剔除项置为负无穷
        indices_to_remove = torch.zeros_like(scaled_logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=0, index=sorted_indices, src=sorted_indices_to_remove)
        scaled_logits[indices_to_remove] = float('-inf')

    # 5. 计算最终的归一化概率分布
    final_probs = F.softmax(scaled_logits, dim=-1)

    # 6. 多项分布随机采样（若只剩1个非零项则确定性输出）
    sampled_token = torch.multinomial(final_probs, num_samples=1)
    return int(sampled_token)

if __name__ == "__main__":
    # 测试用 5 个词的分值
    test_logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
    
    # 运行与前文算例相同的参数
    selected_idx = sample_logits(test_logits, temperature=0.5, top_k=3, top_p=0.85)
    words = ["公园", "散步", "图书馆", "火星", "香蕉"]
    print(f"抽样选中的 Token: {words[selected_idx]} (索引: {selected_idx})")
    # 预期输出: 公园 (索引: 0)
```

---

## 典型场景调参策略与对照

理解了参数之间的依赖逻辑后，在实际工程调优时就不应盲目三个参数一起乱调。

主流工程实践的经验法则是：**固定 Top-P（例如 0.85 ~ 0.9）切断长尾噪声；仅用 Temperature 作为调节输出确定性与多样性的主旋钮；Top-K 设为默认值（如 40 ~ 50）或直接设为 0（不启用）。**需要注意供应商接口差异：OpenAI 官方建议在 Temperature 和 Top-P 之间二选一调节，其公有 API 未开放 Top-K 参数；Anthropic Claude API 则支持三者联调，但其 Temperature 上限为 1.0。

![不同任务类型参数配置雷达对比](/images/content-rebuild/be5b603061a67d94/fig-6.png)

| 业务场景 | 推荐 Temperature | 推荐 Top-P | 推荐 Top-K | 设定理由与边界条件 |
| :--- | :--- | :--- | :--- | :--- |
| **代码生成 / 自动化脚本** | `0.0 ~ 0.2` | `0.9` | 默认 / 关闭 | 语法规则严密，变量名与逻辑分支需要前后严格一致，需要低熵抑制低概率非法标记。 |
| **格式化抽取 (JSON/SQL)** | `0.0` (贪心) | `1.0` | 默认 / 关闭 | 字段格式与 Schema 有刚性约束，直接使用确定性解码抑制幻觉。相关策略见 [如何让大模型稳定输出 JSON](/llm-100/prompt-engineering/如何让大模型稳定输出_JSON.md)。 |
| **事实问答 / RAG 总结** | `0.1 ~ 0.3` | `0.85` | 40 ~ 50 | 依赖检索出的参考上下文，低温确保事实复述严密，Top-P 切掉罕见联想词。 |
| **通用多轮人机对话** | `0.6 ~ 0.7` | `0.9` | 40 ~ 50 | 兼顾语义准确与语言表达的自然度，避免回答像复读机一样死板，同时截断偏门词。 |
| **创意文案 / 故事润色** | `0.8 ~ 1.0` | `0.95` | 默认 / 关闭 | 需要丰富的修辞和多样的词汇组合，提高分布平滑度，允许更多合理候选词进入采样池。 |
| **开放头脑风暴** | `1.0 ~ 1.2` | `0.98` | 100 | 刻意保留部分低概率词以产生跨领域联想；注意温度超过 1.3 时容易引入语病（若使用 Claude API 则因上限限制最高设为 1.0）。 |

> **调优警示**：低温度可以减少随机性，但**无法纠正模型已有的事实性错误**。如果模型的预训练知识本身就认为某错误结论得分最高，那么无论怎么调低温度，模型都会更加确定地输出错误答案。

---

## 可检验练习

为了检验你对联合采样过程的理解，请完成以下两项练习：

### 练习 1：笔算验证
仍采用本文贯穿示例的原始 Logits：`[公园: 4.0, 散步: 3.0, 图书馆: 2.0, 火星: 1.0, 香蕉: 0.0]`。
- 设定新参数组合：$T = 2.0$, $\text{Top-K} = 0$ (不启用), $\text{Top-P} = 0.60$。
- 请计算各词经 $T=2.0$ 缩放后的 Softmax 概率；
- 按照 Top-P 规则累加，写出最终进入核候选集的词汇列表。

<details>
<summary>点击查看练习 1 参考答案</summary>

1. 缩放后 Logits（除以 2.0）：`[2.0, 1.5, 1.0, 0.5, 0.0]`。
2. 计算指数项：
   - $\exp(2.0) \approx 7.389$
   - $\exp(1.5) \approx 4.482$
   - $\exp(1.0) \approx 2.718$
   - $\exp(0.5) \approx 1.649$
   - $\exp(0.0) = 1.000$
   - 总和 $S \approx 17.238$。
3. 计算各词概率：
   - 公园: $7.389 / 17.238 \approx 42.87\%$
   - 散步: $4.482 / 17.238 \approx 26.00\%$
   - 图书馆: $2.718 / 17.238 \approx 15.77\%$
   - 火星: $1.649 / 17.238 \approx 9.57\%$
   - 香蕉: $1.000 / 17.238 \approx 5.80\%$
4. 按降序累加概率：
   - 累加第 1 项（公园）：$42.87\% < 60\%$
   - 累加第 2 项（散步）：$42.87\% + 26.00\% = 68.87\% \ge 60\%$
5. 满足阈值，截断停止。最终核候选集包含：`["公园", "散步"]`。
</details>

### 练习 2：代码实测
将上述练习 1 的数值输入到本文提供的 Python 演示代码中，把循环采样执行 1000 次，统计“公园”与“散步”各自被抽中的频次比率，验证它是否接近两者的相对理论概率比例（约 $42.87 : 26.00 \approx 1.65 : 1$）。

---

## 课程导航与延伸学习

- **上一讲**：[Prompt 为什么能影响大模型输出？](/llm-100/prompt-engineering/Prompt_为什么能影响大模型输出.md) —— 从注意力机制与自回归预填充理解模型输出的根本成因。
- **下一讲**：[为什么同一个 Prompt 每次输出不一样？](/llm-100/prompt-engineering/为什么同一个_Prompt_每次输出不一样.md) —— 深入底层浮点非结合律、并发调度与采样噪声带来的非确定性根源。
- **体系目录**：[Prompt 工程实战指南](/llm-100/prompt-engineering/README.md) —— 查看全套 10 讲进阶路线图。
