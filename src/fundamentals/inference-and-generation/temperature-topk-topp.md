---
title: Temperature、Top-k 与 Top-p 是什么？大模型采样参数工作原理
icon: mdi:thermometer
tag:
- 推理与生成
- Temperature
- 采样策略
description: 系统讲解大模型生成阶段的采样参数原理，拆解 Safe Softmax、Temperature、Top-k 与 Top-p 的数学机制、逐步算例与
  PyTorch 实现。
---

大模型在完成前向推理后，输出层给出的并不是现成的文字，而是词表中每一个 Token 对应的一个未经归一化的实数得分，称为 **Logits**。关于前向推理如何产出这些得分，可以参考[大模型推理是什么？](/fundamentals/inference-and-generation/inference-pipeline.md)。

为了决定下一个生成的词，推理引擎需要把这些得分变成概率，并按规则挑选候选词。这就是采样参数发挥作用的阶段。在实际使用中，我们常常见到 **Temperature（温度）**、**Top-k** 和 **Top-p（核采样）** 三个配置项。它们并不是互斥的独立开关，而是在同一条采样处理流水线上按先后顺序配合工作的调控工具。

为了建立直观理解，本文使用一个具体的例子贯穿始终：假设模型输入为“今天的天气真”，当前词表简化为 4 个候选词，模型输出的原始得分分别为：

- $w_1 = \text{“好”}$，Logit = 10.0
- $w_2 = \text{“不错”}$，Logit = 8.0
- $w_3 = \text{“冷”}$，Logit = 5.0
- $w_4 = \text{“香蕉”}$，Logit = 1.0（语境无关的长尾干扰词）

接下来，我们将逐步观察各个参数如何改变这组得分，并最终决定输出哪一个词。

![采样参数控制看板](/images/content-rebuild/39cce7bc2500f48a/cover.png)

上图展示了模型词表从 Logits 计算到最终生成单个 Token 的完整参数流水线。讲解员正在中央控制面板前梳理整个变换过程：未归一化得分经过数值稳定 Softmax 得到基准分布，通过 Temperature 调整概率峰度，经由 Top-k 与 Top-p 逐步截断不可靠的长尾 Token，最终输入采样算子。正文将按照这一标准流程逐一展开。

---

## 从得分到概率：Softmax 计算与 Safe Softmax 数值防溢出

大模型最后一层投影矩阵的输出维度等于词表大小 $|V|$（例如 LLaMA 词表约为 32,000，Qwen 词表约为 151,936）。设输出向量为 $z = [z_1, z_2, \dots, z_{|V|}] \in \mathbb{R}^{|V|}$，其中的实数即为 Logits。

要将任意实数范围的得分转换为所有元素非负且总和为 1 的离散概率分布，标准的数学变换是 **Softmax 函数**：

$$P(w_i) = \frac{\exp(z_i)}{\sum_{j=1}^{|V|} \exp(z_j)}$$

### 浮点上溢风险与平移不变性

在底层工程计算中，直接对 Logits 求指数容易引发计算故障。在半精度浮点（FP16）表示下，能表示的最大正实数约为 65504。当某个 Logit 满足 $z_i > \ln(65504) \approx 11.089$ 时，$\exp(z_i)$ 就会超出 FP16 上限并溢出为无穷大（`inf`），进而导致分母求和后出现 `inf / inf`，最终产生不可计算的 `NaN`（Not a Number）。

为了避免溢出，主流推理框架均利用 Softmax 的**平移不变性**进行工程优化：

$$\frac{\exp(z_i)}{\sum_j \exp(z_j)} = \frac{\exp(z_i - c)}{\sum_j \exp(z_j - c)}, \quad \text{其中取 } c = \max_j(z_j)$$

通过将整个向量减去当前的最大值 $z_{\max}$，所有调整后的得分均满足 $z_i - z_{\max} \le 0$。此时指数项 $\exp(z_i - z_{\max}) \in (0, 1]$，最大值正好等于 1，完全杜绝了正向上溢出问题。这种实现方式被称为 **Safe Softmax**。

![Safe Softmax 数值稳定性机制](/images/content-rebuild/39cce7bc2500f48a/fig-1.png)

上图演示了原始 Logits 在浮点计算中面临的溢出陷阱与 Safe Softmax 的平移对齐机制。左侧显示当 Logits 出现较大正数时，FP16 指数运算迅速冲破 65504 上限导致 NaN 崩溃；右侧讲解员将整个得分向量统一步长减去最大值 $z_{\max}$，将指数输入安全映射在负数区间内，确保归一化顺利进行。

---

## Temperature：缩放得分差值以控制分布平坦度

在将 Logits 输入 Softmax 函数之前，推理引擎引入了一个正标量参数 $T$（Temperature，$T > 0$）：

$$P(w_i; T) = \frac{\exp(z_i / T)}{\sum_{j=1}^{|V|} \exp(z_j / T)}$$

Temperature 的作用在于**线性缩放不同候选词之间的分值差**。假设词 $a$ 和词 $b$ 的得分差值为 $\Delta z = z_a - z_b$，除以 $T$ 之后，两者的概率比值为：

$$\frac{P(w_a)}{P(w_b)} = \exp\left(\frac{z_a - z_b}{T}\right) = \exp\left(\frac{\Delta z}{T}\right)$$

当 $T$ 很小时，原有的微小差值会被除数放大，指数运算后概率优势被急剧拉大；当 $T$ 很大时，分值差被压缩变小，指数运算后各候选词的概率趋于接近。

<ReadMoreLock />

### 典型取值与极限状态

1. **$T = 1.0$（基准状态）**：不改变得分差距，忠实保留模型原始预训练学习到的分布形态。
2. **$T \to 0$（趋于贪婪解码，Greedy Search）**：
   - 对于最高分候选词之外的任意词，其相对差值 $(z_i - z_{\max}) / T \to -\infty$。
   - 指数变换后，除最高分项外其余各项的取值均趋近于 0。
   - 最终最高分词的选中概率趋近于 100%，完全消除采样的随机性。
3. **$T \to \infty$（趋向均匀分布，Uniform Distribution）**：
   - 任意得分除以极大数后 $z_i / T \to 0$。
   - $\exp(0) = 1$，所有候选词的概率均趋近于 $1 / |V|$。此时模型失去了对上下文逻辑的判断力，生成表现为词表中随机抽词的杂乱文本。

![Temperature 对概率分布的几何拉伸](/images/content-rebuild/39cce7bc2500f48a/fig-2.png)

上图描绘了同一组输入在不同温度参数下的概率形态。讲解员指引对比三条离散曲线：低温 $T=0.2$ 时分布如尖刺拔地而起，绝大部分概率压向第一名；常温 $T=1.0$ 保留原始语义分布；高温 $T=2.0$ 曲线被强行摊平，低频词获得大量采样机会。图文对应解释了温度作为除数在指数放大中的杠杆作用。

### 算例验证：温度对候选词概率的影响

回到贯穿算例，针对四个候选词 $[10.0, 8.0, 5.0, 1.0]$，分别代入不同温度进行计算：

| 候选词 | 原始 Logit ($z_i$) | $T = 1.0$ 概率 | $T = 0.5$ 概率 | $T = 2.0$ 概率 |
| :--- | :--- | :--- | :--- | :--- |
| $w_1$（好） | 10.0 | **87.56%** | **98.20%** | **68.44%** |
| $w_2$（不错） | 8.0 | 11.85% | 1.80% | 25.18% |
| $w_3$（冷） | 5.0 | 0.59% | 0.00% | 5.62% |
| $w_4$（香蕉） | 1.0 | 0.01% | 0.00% | 0.76% |

数据清晰表明：
- 在 $T=0.5$ 的低温下，“好”与“不错”之间 2.0 的得分差被放大为 $(10-8)/0.5 = 4.0$，$\exp(4.0) \approx 54.6$，使得“好”的选中几率提升至 98.20%，低分词基本没有机会被抽中。
- 在 $T=2.0$ 的高温下，分差被折半为 1.0，$\exp(1.0) \approx 2.72$，语义毫不相关的干扰词“香蕉”由原本的万分之一概率攀升至 0.76%，在长文本生成中抽中偏离主题词汇的概率明显增大。更多面试中针对选值的总结可参考[面试问答：Temperature 与 Top-P 选值](/interview/llm-interview/sampling-temperature-topp.md)。

---

## Top-k：按固定数量截断候选池

即使把温度设为 1.0，现代大模型的词表通常包含几万到十几万个 Token。即使极度不合逻辑的词（如上述的“香蕉”）单次概率只有 $0.01\%$，在数千步的长文本自回归生成中，依然有一定概率命中长尾噪声，破坏整段生成的逻辑连贯性。

**Top-k 采样**提供了一种简单直接的防御机制：**按概率从大到小排序，只保留前 $k$ 个候选 Token，将其余所有位置的得分强制设为负无穷大（$-\infty$）**，然后再进行采样。

$$
\text{Top-k}(z_i) = 
\begin{cases} 
z_i, & \text{若 } z_i \ge z_{(k)} \\
-\infty, & \text{若 } z_i < z_{(k)}
\end{cases}
$$

其中 $z_{(k)}$ 是降序排列后的第 $k$ 个得分。被设为 $-\infty$ 的 Token 经过 Softmax 后概率为 0，彻底被排除在采样候选池之外。

![Top-k 固定截断机制与边界风险](/images/content-rebuild/39cce7bc2500f48a/fig-3.png)

上图展示了 Top-k 静态门槛的切分方式。讲解员立于概率柱状图前，一道标有 $K=5$ 的机械闸门垂直切下，前 5 位候选词保留原色，第 6 位以后的长尾柱状全部变为灰色并标记负无穷淘汰符号。图下侧直观揭示了当第 6、7 位概率与第 5 位极度接近时，静态截断带来的硬性信息损失。

### Top-k 的局限性

Top-k 的核心短板在于其**固定阈值无法适应模型置信度的动态变化**：

1. **高置信度语境（确定性高）**：当输入“法国的首都是”时，正确答案“巴黎”的置信度可能超过 99%。如果固定配置 $k=50$，系统就会保留 49 个错误或低质的词进入候选池，平白增加了随机抽取到杂乱词汇的概率。
2. **低置信度语境（开放发散）**：当输入“我最喜欢的运动是”时，“足球、篮球、羽毛球、跑步、游泳、网球”等词的概率均较为接近。若此时固定 $k=3$，就会粗暴截断同样完全合理的大量候选词，导致生成失去多样性。

> **概念区分**：模型文本生成中的 Top-k 作用于**单步词表概率截断**；而检索增强生成中的 Top-K 作用于**从向量数据库检索返回的相关文档块数量**，两者属于完全不同的阶段。关于检索场景的设定可参考 [RAG 检索中的 Top-K 是什么意思？](/interview/rag-interview/rag-topk.md)。

---

## Top-p（核采样）：根据累积概率自适应收缩与扩张

为了解决 Top-k 缺乏弹性的缺陷，Holtzman 等人在 2019 年提出了 **Top-p 采样（又称 Nucleus Sampling，核采样）**。

Top-p 不固定选中的候选词个数，而是设定一个累积概率阈值 $p$（通常取值在 0.8 到 0.95 之间）。它的规则是：**将候选词按概率从大到小排列并依次累加，保留使累积概率首次达到或超过阈值 $p$ 的最小候选词集合**，将其余尾部词全部过滤掉。

$$\sum_{w_i \in V^{(p)}} P(w_i) \ge p$$

![Top-p 核采样自适应候选池机制](/images/content-rebuild/39cce7bc2500f48a/fig-4.png)

上图对比了 Top-p 在置信度差异场景下的自适应展开。左侧高置信度面板中，前 2 个 Token 累积概率即突破 0.9，候选池迅速收拢为 2；右侧低置信度面板中，因概率平缓分散，讲解员拉动蓝色累积虚线，足足包含了 7 个 Token 才达到 0.9 阈值，候选池自动扩张。清晰展示了 Nucleus 采样“随熵而动”的自适应特征。

### 对比算例：Top-p 如何自适应变化？

假设我们固定设置 $p = 0.90$：

- **场景一：模型高度确信（分布集中）**
  - 候选词概率为：$w_1 (0.88), w_2 (0.05), w_3 (0.04), w_4 (0.03)$
  - 累加第 1 项：$0.88 < 0.90$
  - 累加第 2 项：$0.88 + 0.05 = 0.93 \ge 0.90$（达到阈值，截断停止）
  - **最终候选池仅保留 2 个词**。尾部大量低质量候选词被安全切除。

- **场景二：开放探索语境（分布分散）**
  - 候选词概率为：$w_1 (0.25), w_2 (0.22), w_3 (0.18), w_4 (0.15), w_5 (0.12), w_6 (0.08)$
  - 累加计算：$0.25 + 0.22 + 0.18 + 0.15 = 0.80 < 0.90$
  - 加入第 5 项：$0.80 + 0.12 = 0.92 \ge 0.90$（达到阈值，截断停止）
  - **最终候选池自动扩大到 5 个词**。

可以看到，Top-p 的边界会随着模型预测的不确定性自动伸缩：模型确定时候选池自动变小，模型不确定时候选池自动扩容。

---

## 组合采样流水线：执行顺序与代码实现

在实际的推理引擎中，Temperature、Top-k 与 Top-p 通常会配合使用。这就涉及到一个关键的工程问题：**它们应该按什么顺序执行？**

标准的执行流水线必须遵循严格的先后顺序：

1. **Logits 输入**：获取模型前向传播输出的未归一化得分向量 $z$；
2. **Temperature 缩放**：执行 $z' = z / T$。必须在最初一步完成，因为温度会重塑整个分布的曲率；
3. **Top-k 截断**：在缩放后的得分上找出前 $k$ 名之外的所有位置，填充为 $-\infty$；
4. **Top-p 截断**：对保留下来的得分计算 Softmax 并按降序累加，将累积概率超过 $p$ 的多余尾部置为 $-\infty$；
5. **重新归一化与采样**：对剩余未被掩码的得分执行 Safe Softmax，并调用随机类别采样算子抽选最终的 Token。

如果颠倒顺序（例如先做 Top-p 截断再调 Temperature），截断时依据的就是未受温度控制的分布，Top-p 动态调整候选池大小的能力就会失去预期的联动效果。

![推理采样流水线执行阶段](/images/content-rebuild/39cce7bc2500f48a/fig-5.png)

上图拆解了推理引擎内部从 Logits 到采样输出的完整五步流水线。从左到右依次为 Logits 输入、除以 T 缩放、Top-k 掩码过滤、Top-p 累积掩码过滤，到最终由 Safe Softmax 重新归一化后掷骰子采样。讲解员在关键节点标注了张量数值的状态转移与负无穷掩码标记。

### 完整的 PyTorch 实现代码

以下为依照标准流水线编写的采样函数：

```python
import torch
import torch.nn.functional as F

def sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    标准文本生成采样流水线：
    Temperature 缩放 -> Top-k 过滤 -> Top-p 过滤 -> 重新归一化离散采样
    输入 logits 形状: [batch_size, vocab_size]
    """
    # 1. 贪婪解码特判：若温度极低或为0，直接取最大得分对应的 Token
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. Temperature 缩放
    logits = logits / temperature

    # 3. Top-k 截断：仅保留排名前 k 的得分
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # 找出第 k 大的值作为过滤门槛
        kth_val = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(logits < kth_val, float("-inf"))

    # 4. Top-p 截断：基于当前得分计算累积概率
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 标记累积概率超过 top_p 的位置
        # 右移一位以确保刚好使累积和超过 top_p 的第一个 Token 依然被保留
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # 将排序维度的掩码还原散布到原 logits 的对应下标位置
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # 5. 重新归一化并进行多项式分布采样
    final_probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(final_probs, num_samples=1)
    return next_token
```

代码中通过把无效候选位置填充为 `float('-inf')`，避免了动态截断张量造成的显存频繁重分配，保证了整个批次计算能够高效完成矢量化运算。

---

## 实用配置参考与常见认知误区

在日常调用大模型 API 或搭建推理服务时，针对不同的业务目标，参数配置有明确的侧重点：

| 应用场景 | Temperature | Top-k | Top-p | 参数设计考量 |
| :--- | :--- | :--- | :--- | :--- |
| **代码生成 / SQL 解析 / JSON 输出** | 0.0 ~ 0.2 | 1 ~ 10 | 0.80 ~ 0.85 | 语法结构严苛，需要最大程度消除随机性，锁定置信度最高的分支。如需强制结构保证，可参考 [结构化输出和 Function Calling 是什么？](/fundamentals/inference-and-generation/structured-output.md)。 |
| **事实问答 / 知识库问答** | 0.3 ~ 0.5 | 20 ~ 40 | 0.85 ~ 0.90 | 允许句式上有自然的微小差异，同时压制低概率幻觉词。配合 [大模型生成常见问题](/fundamentals/inference-and-generation/generation-pitfalls.md) 排查。 |
| **头脑风暴 / 故事创作** | 0.7 ~ 1.0 | 40 ~ 100 | 0.92 ~ 0.95 | 适当摊平分布，扩大核候选池边界，提升词汇多样性与句意新颖度。 |

![参数配置误区与输出异常对比](/images/content-rebuild/39cce7bc2500f48a/fig-6.png)

上图对比了参数配置失误所引发的异常表象。讲解员剖析两处典型故障：左侧由于温度过高导致模型词频失序、语法解构，输出不可读的乱码字符；右侧由于温度过低或过度截断，在没有频率惩罚的干预下陷入单调死循环短语。图下方列举了排查和调参纠偏方案。

### 常见误区辨析

1. **误区一：Top-k 和 Top-p 同时启用是冗余的**  
   **事实**：两者同时启用是成熟工程系统的常见做法。Top-k 提供了一个**硬性上限**（例如无论分布多平缓，候选词绝不超过 50 个），能够防止极端情况下计算和显存开销过大；Top-p 则负责在上限之内，根据当前语义确信度做二次动态过滤。
2. **误区二：调高 Temperature 可以消除模型的幻觉**  
   **事实**：模型的幻觉本质上来源于训练知识遗忘、注意力检索失真或上下文引导偏差（详见 [大模型生成常见问题](/fundamentals/inference-and-generation/generation-pitfalls.md)）。调高温度只会让原本概率很低的杂乱词汇更容易被抽中，不仅无法纠正错误事实，反而会加剧语句语法混乱和毫无意义的词汇堆叠。
3. **误区三：遇到句子重复复读，只要调高温度就行**  
   **事实**：复读通常是自回归生成中局部注意力偏置导致的自我强化。微调温度无法从根源解决该问题；更直接有效的办法是在推理时引入**重复惩罚（Repetition Penalty）**或**频率惩罚（Frequency Penalty）**，直接扣减历史已出现 Token 的 Logit 分值。

---

## 总结与延伸阅读

- **Softmax 与数值稳定**：Logits 必须经由减去最大值的 Safe Softmax 运算，防止浮点计算溢出导致输出崩溃。
- **Temperature（温度）**：作为缩放分母作用于未归一化得分。$T < 1$ 放大尖峰优势，$T > 1$ 摊平差距，其本质是控制概率分布的全局信息熵。
- **Top-k**：硬性保留排名前 $k$ 的候选词，计算边界清晰但缺乏对语义置信度的自适应弹性。
- **Top-p（核采样）**：以累积概率阈值为动态边界，在高确信语境自动收紧候选池、在开放探索语境自适应扩容。
- **组合顺序**：标准流水线始终遵循 `Temperature 缩放 -> Top-k 掩码 -> Top-p 掩码 -> 重新归一化采样`。

在生成完成当前单个 Token 之后，推理引擎需要将其追加到序列中，并进入下一个自回归迭代步。为了避免每次生成新 Token 时对历史输入进行重复前向矩阵乘法，系统引入了关键的显存与计算重用机制，下一篇请阅读 [KV Cache 是什么？](/fundamentals/inference-and-generation/kv-cache.md)。

想查看本栏目全部进阶架构内容，可访问 [推理与生成导航页](/fundamentals/inference-and-generation/README.md)。
