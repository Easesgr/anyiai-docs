---
title: 什么是上下文学习（ICL）？Few-shot 原理与示例挑选策略
icon: mdi:counter
tag:
- Few-shot
- Zero-shot
- One-shot
description: 解释大模型上下文学习（ICL）的前向计算机制，分析示例顺序敏感、近因偏置与标签失衡成因，提供基于 MMR 的动态少样本选择方案与面试口述框架。
---

在技术面试中，大模型的少样本提示（Few-shot Prompting）经常被拿来与传统机器学习的微调（Fine-Tuning）或少样本学习（Few-shot Learning）做对比。理解大模型的上下文学习（In-Context Learning，简称 ICL），核心是分清**推理阶段的前向传播计算**与**训练阶段的反向传播参数更新**。

如果你需要先了解零样本（Zero-shot）、单样本（One-shot）与少样本的场景差异，可以先参考 [/interview/prompt-interview/shot-types.md](/interview/prompt-interview/shot-types.md)。本文通过一个贯穿始终的“金融工单分类”例子，解释 ICL 的工作过程、常见失效原因以及工程中挑选示例的算法。

![Few-shot 上下文学习原理全景](/images/content-rebuild/2b28ee5fcb7b1107/cover.png)

---

## 面试速答：什么是上下文学习（ICL）？

> **建议在 60 秒内这样口述回答：**
>
> “大模型中的上下文学习（In-Context Learning）是指：在**不修改模型权重参数**、**没有任何梯度更新**的情况下，仅通过前向推理，在输入提示词（Prompt）中放置几个示范样本，让模型理解当前任务的目标、格式约束与分类映射，从而对新输入完成预测。
>
> 它和传统模型微调的差异在于：微调改变的是持久化的网络权重，而 ICL 改变的是推理过程中注意力机制计算出的键值对缓存（KV Cache）和上下文激活值。
>
> 因为 ICL 完全依赖自回归注意力，它很容易受到**示例摆放顺序**（近因偏置）、**正负样本比例**（先验偏置）以及**表面特征**（如句子长短或标点符号）的干扰。在工程落地时，通常需要保持类别平衡，或使用 MMR（最大边际相关）等算法动态选择多样化示例。”

![上下文学习与微调机制对比](/images/content-rebuild/2b28ee5fcb7b1107/fig-1.png)

---

## 贯穿案例：金融工单意图识别

为了直观理解，我们来看一个金融客服意图分类任务。目标是将用户的输入分类到三个类别之一：`[账单咨询, 额度调整, 欺诈申诉]`。

如果我们构造一个 3-shot 的上下文提示词：

```text
请对以下金融工单的意图进行分类，只输出类别名称。

工单：我上个月的账单明细怎么看不懂？
类别：账单咨询

工单：我的信用卡额度太低了，想提高到五万。
类别：额度调整

工单：我的卡在国外被盗刷了一笔消费，赶紧帮我冻结！
类别：欺诈申诉

工单：昨晚收到扣款短信，但我根本没买东西，这是怎么回事？
类别：
```

模型在末尾直接输出 `欺诈申诉`。在这个过程中，没有任何代码在训练模型，也没有任何参数被改写，模型仅读取了前面的文字就完成了任务。这个机制是如何在前向计算中发生的？

---

## 原理分析：前向注意力如何捕获任务映射

在标准自回归 Transformer 中，对于输入序列 $X = [x_1, x_2, \dots, x_N]$，每层多头注意力通过线性映射计算查询向量 $Q$、键向量 $K$ 和值向量 $V$：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

当我们向 Prompt 传入 $k$ 个示例 $(x^{(1)}, y^{(1)}), \dots, (x^{(k)}, y^{(k)})$，再接上测试样本 $x^{(\text{test})}$ 时，计算过程如下：

1. **示例信息的编码与缓存**：每个示例的输入与标签通过自注意力层后，提取出语义特征并保存在对应的 Key 与 Value 张量中。
2. **测试样本的前向检索**：在因果注意力掩码下，待预测的测试样本 $x^{(\text{test})}$ 的 Query 向量在前向推理时，会与前面所有示例的 Key 向量计算内积，得到注意力权重。这意味着模型在计算新样本时，会以加权求和的形式聚合前面示例的 Value 表征。
3. **归纳头与模式复制**：在可解释性研究中，研究者观察到预训练 Transformer 中存在一类特定的注意力头（Induction Heads）。这类头能够执行类似 `[A][B] ... [A] -> [B]` 的模式匹配与信息搬运。部分理论研究也将该前向计算过程类比为在隐层激活状态中执行隐式元优化。

因此，模型并不需要改动持久化参数，而是在前向计算生成的 KV 缓存中，把示例作为条件约束，从而调整下一个词元的条件概率分布 $P(y^{(\text{test})} \mid x^{(\text{test})}, \text{Examples})$。

<ReadMoreLock />

---

## 常见失效现象一：示例顺序敏感与近因偏置

在评测中常常会发现：**仅仅调换 Few-shot 中几个示例的先后次序，分类准确率就可能产生大幅波动**。这主要源于两点：

![示例顺序与近因偏置机制](/images/content-rebuild/2b28ee5fcb7b1107/fig-2.png)

### 1. 因果掩码下的信息累积不对称
自回归模型采用因果掩码。序列靠后的示例可以看到前面所有示例的内容，因此其深层表征融合了前序上下文；而排在最开头的第一个示例只能看到它自己。这使得每个示例在上下文中的表征所包含的背景信息量并不均等。

### 2. 近因偏置（Recency Bias）
在因果注意力与位置编码的作用下，待预测的测试样本在位置上紧邻**最后一个示例**。在不少经验观察中，模型对物理距离更近的词元往往会分配更多的注意力权重。如果最后一个示例刚好是 `欺诈申诉`，模型在输出新预测时，该类别的预测概率就更容易被拉高。

#### 示例对比测试
在工单任务中，如果我们用模糊文本作为测试输入：`"昨天账户被扣了200元。"`（既可能是账单疑问，也可能是盗刷）：

- **排列 A（末尾为账单咨询）**：`[额度调整, 欺诈申诉, 账单咨询] -> 测试样本`
  模型大概率倾向于将模糊样本归类为 `账单咨询`。
- **排列 B（末尾为欺诈申诉）**：`[额度调整, 账单咨询, 欺诈申诉] -> 测试样本`
  模型生成 `欺诈申诉` 的对数几率（Logits）会被明显抬高。

---

## 常见失效现象二：标签失衡与虚假相关性

Few-shot 并非给出的样例越多越好。低质量或分布不当的样本会导致模型推断产生偏置：

![标签分布扭曲与虚假相关性](/images/content-rebuild/2b28ee5fcb7b1107/fig-3.png)

### 1. 多数类标签偏置（Majority Label Bias）
如果开发者在 Few-shot 中挑选了 4 个“账单咨询”和 1 个“额度调整”，模型在计算词元概率时，会把上下文中的频率当成先验概率 $P(Y)$。即便输入是一个倾向于额度调整的样本，模型也可能受高频示例的影响误判为账单咨询。因此在少样本设置下，**各类别示例数量应当保持平衡**。

### 2. 格式与表面特征的虚假相关性
大语言模型对表面格式十分敏感。如果在示例中：
- 所有的“欺诈申诉”工单都写得很长，且末尾都带感叹号 `！`；
- 所有的“账单咨询”工单都是极短的句子，且末尾是问号 `？`；

模型可能学习到“长句+感叹号 = 欺诈申诉”，而不是去理解文本中的盗刷语义。当遇到一条写得很长并带有感叹号的咨询类工单时，模型很容易误判。

---

## 动态 Few-shot：基于 MMR 解决多样性坍缩

在真实业务中，静态硬编码 3 到 5 个固定示例往往无法覆盖长尾意图。工程上通常采用**动态少样本选择**：针对用户的当前输入，从离线样本库中实时检索出最匹配的 $K$ 个示例拼入 Prompt。

但这里有一个常见问题：**只根据向量余弦相似度做 Top-K 检索，容易导致“多样性坍缩”**。

![动态检索的多样性崩溃与 MMR 算法](/images/content-rebuild/2b28ee5fcb7b1107/fig-4.png)

如果用户提问：“我想把分期还款的期数从12期改成24期，怎么办理？”纯相似度检索出来的 3 条示例可能全是关于“申请账单分期”的相近问句，缺乏展示“提前结清分期”或“取消分期”等相近边界情况的反例。这不仅白白占用了上下文窗口，也没有给模型提供足够的辨析边界。

工程上常用**最大边际相关算法（Maximal Marginal Relevance, MMR）**来平衡示例的相关度与多样性：

$$\text{MMR}(q, D, S) = \arg\max_{d_i \in D \setminus S} \left[ \lambda \cdot \text{Sim}(d_i, q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}(d_i, d_j) \right]$$

其中 $q$ 是当前用户输入，$D$ 是候选示例库，$S$ 是已选出的示例集合。参数 $\lambda \in [0, 1]$ 用于调节权重：$\lambda$ 越大越偏向相似度，$\lambda$ 越小越强调多样性。

![动态 Few-shot 检索执行流程](/images/content-rebuild/2b28ee5fcb7b1107/fig-5.png)

### 动态示例挑选代码实现

以下 Python 代码演示了如何使用 MMR 算法从候选池中选出兼顾相关性与多样性的少样本：

```python
import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量之间的余弦相似度。"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def select_dynamic_few_shots(
    query_vec: np.ndarray,
    candidate_vecs: list[np.ndarray],
    candidate_samples: list[dict],
    top_k: int = 3,
    lambda_mult: float = 0.5
) -> list[dict]:
    """
    使用 MMR 挑选兼顾语义相关性与多样性的示例集合。
    lambda_mult: 1.0 为纯相似度 Top-K，0.0 为完全强调多样性。
    """
    if len(candidate_samples) <= top_k:
        return candidate_samples

    selected_indices: list[int] = []
    unselected_indices = list(range(len(candidate_samples)))

    # 1. 预先计算所有候选样本与当前 query 的相似度
    query_sims = [cosine_similarity(query_vec, v) for v in candidate_vecs]

    # 2. 首选与 query 最相似的样本
    first_idx = int(np.argmax(query_sims))
    selected_indices.append(first_idx)
    unselected_indices.remove(first_idx)

    # 3. 循环选择剩余样本，综合考量相似度与已有样本的重复度
    while len(selected_indices) < top_k and unselected_indices:
        best_score = -float("inf")
        best_idx = -1

        for c_idx in unselected_indices:
            relevance = query_sims[c_idx]
            # 计算候选样本与已选集合中各样本的最大相似度（代表冗余度）
            redundancy = max(
                cosine_similarity(candidate_vecs[c_idx], candidate_vecs[s_idx])
                for s_idx in selected_indices
            )
            mmr_score = lambda_mult * relevance - (1.0 - lambda_mult) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = c_idx

        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)

    return [candidate_samples[i] for i in selected_indices]
```

---

## 面试高频追问

### 追问 1：如果 Few-shot 示例里的标签全标错了，模型准确率会归零吗？
**回答**：不会归零，在很多大模型上准确率甚至下降得有限。
- **原因**：学术研究（如 Min 等人《Rethinking the Role of Demonstrations》）发现，Few-shot 示例主要向模型传递了四个维度的信息：
  1. **输入文本的分布**（输入句子长什么样、包含哪些术语）；
  2. **任务的标签空间**（输出允许有哪些类别选项）；
  3. **输出的格式结构**（比如键值对格式、前后包裹的分隔符）；
  4. **输入与标签之间的真实映射关系**。
- 大模型在预训练阶段已经掌握了大量的通用分类常识。输入中的错误标签主要破坏了第 4 项（输入到标签的映射），但保留了前 3 项对任务形式与范围的提示。因此，除非任务是完全人工合成的符号加密（模型无法调用预训练知识），否则打错标签不会导致推理准确率直接归零。

### 追问 2：长上下文普及后，直接放 100 个示例一定比放 5 个示例更好吗？
**回答**：不一定，存在明显的边际效用递减与性能代价。
- **收益递减**：在常见文本分类或抽取任务中，示例数量从 0 增加到 3~5 个时准确率提升最显著，继续增加到几十个时，边际提升会迅速收敛。
- **性能开销**：上下文长度大幅增长会显著拉长首字生成延迟（TTFT），且每次调用的输入 Token 成本和显存开销会成倍增加。更多长提示词失效机理可参考 [/interview/prompt-interview/prompt-failure.md](/interview/prompt-interview/prompt-failure.md)。
- **建议**：优先保证 3 到 8 个高质量、类别平衡且经过多样性筛选的示例。只有在涉及复杂格式规范（参见 [/interview/prompt-interview/structured-output.md](/interview/prompt-interview/structured-output.md)）或垂直冷门领域语法时，才考虑通过长示例库覆盖。

---

## 工程验收：构建可靠的评测流程

交付包含 Few-shot 的系统时，应进行系统性评测，防范线上偶发故障：

![Few-shot 稳定性与验证工程工作流](/images/content-rebuild/2b28ee5fcb7b1107/fig-6.png)

### 1. 严格防止示例与测试集数据穿越
构建动态示例库时，必须使用文本哈希与语义重合度检测，将示例候选池与离线评测集严格隔离。严禁将测试集里的同义改写样本放入示例池中，避免评测指标虚高。

### 2. 排列顺序随机洗牌测试（Permutation Test）
选定 $K$ 个示例后，对它们的排列顺序做洗牌遍历（例如 4 个示例共有 $4! = 24$ 种排列），在固定测试集上观察准确率波动。
- **验收准则**：不同顺序下的准确率极差（最大值减最小值）应控制在合理范围（如 5%~8% 以内）。如果波动过大，应通过格式标准化约束，或在线上固定经测试最稳定的顺序。

### 3. 基线对照表参考
在上线评审时，建议提供如下维度的消融对比记录（下表数据为教学参考示意，非特定生产环境实测值，实际指标因模型选型、硬件环境及业务数据分布而异）：

| 方案配置 | 评测准确率（示意） | 格式合规率（示意） | P95 首字延迟（示意） | 单次请求输入 Token（示意） |
| :--- | :--- | :--- | :--- | :--- |
| Zero-shot 基线 | 68.2% | 89.4% | ~280ms | 180 tokens |
| 静态 4-shot（未校准顺序） | 74.5%（波动 ±6.2%） | 98.1% | ~460ms | 620 tokens |
| 静态 4-shot（固定优选顺序） | 78.1%（波动 ±1.1%） | 99.2% | ~460ms | 620 tokens |
| 动态检索 4-shot（MMR 兼顾多样性）| 84.3% | 99.8% | ~520ms（含向量检索） | 640 tokens |

如果任务属于多步骤的逻辑推导，仅给输入输出的 Few-shot 可能不够，建议配合思维链方法，参考 [/interview/prompt-interview/chain-of-thought.md](/interview/prompt-interview/chain-of-thought.md)；如需评估是否需要转为模型参数微调，请查阅 [/interview/prompt-interview/prompt-vs-finetuning.md](/interview/prompt-interview/prompt-vs-finetuning.md)。如需系统化备战相关岗位面试，请访问 [/learning-path/ai-interview-prep.md](/learning-path/ai-interview-prep.md) 与 [/interview/prompt-interview/README.md](/interview/prompt-interview/README.md)。
