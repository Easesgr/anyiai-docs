---
title: 大模型预训练目标是什么？因果语言建模、交叉熵与困惑度
icon: mdi:target
tag:
- 大模型面试
- 预训练
- NTP
description: 解析因果语言建模（CLM）原理、最大似然与交叉熵等价推导、Teacher Forcing并行机制、与掩码建模（MLM）的差异及困惑度（PPL）评测边界。
---

![大模型预训练目标函数全景图：自回归因果语言建模、交叉熵损失与困惑度标尺](/images/content-rebuild/dfe729bdd8bb9154/cover.png)

在学习大语言模型的架构与训练流程时，首先需要理解模型学习的驱动机制——**预训练目标（Pre-training Objective）**。无论是 GPT、LLaMA 还是 DeepSeek，生成式语言模型的核心能力，都建立在对预训练目标函数的优化之上。

本文通过一个贯穿始终的文本示例，推导因果语言模型（Causal Language Modeling, CLM）的损失函数，解释它与多分类交叉熵的等价关系，对比掩码语言模型（Masked Language Modeling, MLM）的机制差异，并分析评测指标困惑度（Perplexity, PPL）的物理含义与对比边界。

---

## 面试速记：30秒口述回答

如果面试官询问：“主流大模型的预训练目标是什么？损失函数与评测指标是如何定义的？”，可以按以下三点回答：

1. **预训练目标**：主流生成式大模型采用**因果语言建模（Causal LM）**，即“下一个词预测”（Next Token Prediction）。基于概率链式法则，模型根据当前已知的所有上文 Token，预测下一个位置出现各个词的条件概率分布。
2. **损失函数**：目标是最小化整个序列的**负对数似然损失（Negative Log-Likelihood, NLL）**。在具体工程中，输出层在词表维度上做 Softmax 多分类，单步损失等价于预测概率与真实标签 One-hot 向量的**交叉熵损失（Cross-Entropy Loss）**。训练时利用 **Teacher Forcing** 和因果注意力掩码，实现全序列各位置损失的并行计算。
3. **评测指标**：困惑度（Perplexity, PPL）衡量模型对测试文本的预测不确定性，数值上等于平均交叉熵损失的自然指数 $\text{PPL} = \exp(\mathcal{L})$。它的直观物理含义是模型预测下一个词时的**有效候选分支数**。PPL 越低，模型的预测越确定。不同分词器之间的 PPL 不能直接横向对比。

---

## 贯穿示例与因果语言模型（CLM）损失推导

假设输入句子为“今天天气真好”，分词后得到 4 个 Token 的离散序列：

$$X = (x_1, x_2, x_3, x_4) = (\text{“今天”}, \text{“天气”}, \text{“真”}, \text{“好”})$$

因果语言建模的核心任务，是估计该序列在真实语言分布中的联合概率 $P(X)$。由于词汇组合空间极大，直接估计联合分布在计算上不可行。因此，根据条件概率的链式法则，联合概率被分解为按时序展开的单向条件概率乘积：

$$P(X) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_1, x_2) \cdot P(x_4 \mid x_1, x_2, x_3) = \prod_{t=1}^T P(x_t \mid x_{<t})$$

其中 $x_{<t} = (x_1, \dots, x_{t-1})$ 表示时步 $t$ 之前的上文前缀。

![自回归语言模型（CLM）预测时序与训练期 Teacher Forcing 并行机制](/images/content-rebuild/dfe729bdd8bb9154/fig-1.png)

### 1. 最大似然估计与负对数似然（NLL）

给定海量无标注语料库 $\mathcal{D} = \{X^{(1)}, X^{(2)}, \dots, X^{(N)}\}$，训练参数化模型 $\theta$ 的目标是最大化整个语料库的生成似然（Maximum Likelihood Estimation, MLE）：

$$\theta^* = \arg\max_\theta \prod_{X \in \mathcal{D}} P(X; \theta)$$

由于多个小数连续相乘容易引发数值下溢，通常对目标取自然对数，转化为对数似然。为了将其转化为标准的经验风险最小化（Loss 越小越好），在前面加上负号，得到**负对数似然损失（Negative Log-Likelihood, NLL）**：

$$\mathcal{L}_{\text{NLL}}(\theta) = - \sum_{X \in \mathcal{D}} \log P(X; \theta) = - \sum_{X \in \mathcal{D}} \sum_{t=1}^T \log P(x_t \mid x_{<t}; \theta)$$

对单条长度为 $T$ 的样本取序列长度的平均值，得到单样本 CLM 损失：

$$\mathcal{L}_{\text{CLM}}(\theta) = - \frac{1}{T} \sum_{t=1}^T \log P(x_t \mid x_{<t}; \theta)$$

### 2. 负对数似然与多分类交叉熵的等价推导

在模型输出端，Transformer 最后一层的隐藏向量 $h_t \in \mathbb{R}^{d}$ 经过线性层映射到词表大小 $V$，输出未归一化的分数向量（Logits）$z_t \in \mathbb{R}^V$。设词表为 $\mathcal{V} = \{1, 2, \dots, V\}$，模型在位置 $t$ 预测下一个词是词表中第 $k$ 个词的概率由 Softmax 函数给出：

$$\hat{y}_{t, k} = P(x_t = k \mid x_{<t}; \theta) = \frac{\exp(z_{t, k})}{\sum_{j=1}^V \exp(z_{t, j})}$$

在监督训练视角下，时步 $t$ 真实的下一个词 $x_t$ 对应一个长度为 $V$ 的 One-hot 标签向量 $y_t$：

$$y_{t, k} = \begin{cases} 1, & \text{当 } k = x_t \\ 0, & \text{当 } k \neq x_t \end{cases}$$

![Logits 经过 Softmax 转化为概率分布并与 One-hot 标签计算多分类交叉熵](/images/content-rebuild/dfe729bdd8bb9154/fig-2.png)

真实标签分布 $y_t$ 与模型预测分布 $\hat{y}_t$ 之间的多分类交叉熵定义为：

$$\mathcal{H}(y_t, \hat{y}_t) = - \sum_{k=1}^V y_{t, k} \log \hat{y}_{t, k}$$

因为 $y_t$ 仅在真实词对应索引 $k = x_t$ 处为 1，其余位置均为 0，求和式中仅剩下一项：

$$\mathcal{H}(y_t, \hat{y}_t) = - \log \hat{y}_{t, x_t} = - \log P(x_t \mid x_{<t}; \theta)$$

将整个序列所有位置的交叉熵取平均值：

$$\mathcal{L}_{\text{CE}} = \frac{1}{T} \sum_{t=1}^T \mathcal{H}(y_t, \hat{y}_t) = - \frac{1}{T} \sum_{t=1}^T \log P(x_t \mid x_{<t}; \theta) = \mathcal{L}_{\text{CLM}}(\theta)$$

**结论**：在离散符号预测设定下，因果语言模型的最大似然估计（MLE）在数学上与各时步的多分类交叉熵损失严格等价。

### 3. 工程实现：Teacher Forcing 与序列错位切片（Shift）

在预训练阶段，模型不需要逐步串行生成，而是采用 **Teacher Forcing** 策略：直接将真实文本输入模型，依靠因果注意力掩码（下三角矩阵）遮挡未来位置，防止模型偷看右侧内容（具体机制见 [/interview/llm-interview/attention-masking.md](/interview/llm-interview/attention-masking.md)）。

这意味着长为 $T$ 的序列可以在一次前向传播中并行算完所有时步的 Logits。为了计算交叉熵，需要将 Logits 序列与标签序列进行**向右错位切片**：

* 输入序列：`["今天", "天气", "真", "好"]`，长度为 4。
* 模型在位置 0（“今天”）输出的 Logit 负责预测下一个词“天气”；位置 1 输出预测“真”；位置 2 输出预测“好”。
* 最后一个位置 3（“好”）的 Logit 之后没有下一个文本词作为标签（除非显式拼入句子结束标记 `[EOS]`），因此需要切除最后一个 Logit；同理，第一个词“今天”没有上文输入预测它，因此需要切除第一个 Label。

标准 PyTorch 实现如下：

```python
import torch
import torch.nn.functional as F

def compute_causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    计算因果语言模型（CLM）的交叉熵损失
    
    参数:
        logits: 模型输出的未归一化分数，形状为 (batch_size, seq_len, vocab_size)
        input_ids: 输入序列的 token 索引，形状为 (batch_size, seq_len)
    返回:
        平均交叉熵标量损失
    """
    # 错位操作：剔除最后一个时步的 logit，因为它没有对应的下一个真实 token
    shift_logits = logits[..., :-1, :].contiguous()
    
    # 错位操作：剔除第一个时步的 label，因为没有输入前缀来预测它
    shift_labels = input_ids[..., 1:].contiguous()
    
    # 展平 batch 与序列维度，送入 PyTorch 交叉熵计算
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100  # 用于屏蔽 padding 位置
    )
    return loss

# 验证示例
if __name__ == "__main__":
    B, T, V = 2, 4, 100  # 批大小2，序列长度4，词表大小100
    dummy_logits = torch.randn(B, T, V)
    dummy_input_ids = torch.randint(0, V, (B, T))
    
    loss_val = compute_causal_lm_loss(dummy_logits, dummy_input_ids)
    print(f"计算所得平均交叉熵损失值: {loss_val.item():.4f}")
```

<ReadMoreLock />

---

## 掩码语言模型（MLM）对比：为什么现代大模型普遍选用 CLM？

在 GPT 确立自回归范式之前，以 BERT 为代表的**掩码语言模型（Masked Language Modeling, MLM）**曾是自然语言处理的核心预训练方式。

![因果自回归模型（CLM）与掩码语言模型（MLM）的注意力拓扑与目标机制对比](/images/content-rebuild/dfe729bdd8bb9154/fig-3.png)

### 1. 机制差异对比

MLM 不按时序展开条件概率，而是随机选取输入中约 15% 的 Token 替换为 `[MASK]` 占位符、随机词或保持不变，让模型利用双向上下文重构这些被遮蔽的词：

$$\mathcal{L}_{\text{MLM}}(\theta) = - \sum_{i \in M} \log P(x_i \mid \tilde{X}; \theta)$$

| 维度 | 因果语言模型（CLM / NTP） | 掩码语言模型（MLM） |
| :--- | :--- | :--- |
| **代表模型** | GPT、LLaMA、DeepSeek、Qwen | BERT、RoBERTa、DeBERTa |
| **注意力可见范围** | 下三角因果掩码，仅能查看左侧历史 Token | 全局双向注意力，可同时查看左右上下文 |
| **每样本有效监督比例** | 高（序列中几乎所有位置均计算预测损失） | 较低（通常仅对约 15% 的遮蔽位置计算损失） |
| **自回归生成适配度** | 原生适配逐词生成，训练与推理行为一致 | 难以直接高效用于自由生成，通常需多次双向重采样 |
| **核心擅长任务** | 开放式文本生成、代码补全、长上下文续写 | 句级分类、实体识别、语义相似度匹配等判别理解任务 |

### 2. MLM 难以直接用于自回归生成的工程原因

现代大语言模型全面转向 Decoder-only 与 CLM 目标，而非双向 MLM，核心原因包含以下三点：

1. **多位置独立性假设缺陷**：MLM 在单次前向中预测多个 `[MASK]` 时，形式上假设这些被掩盖的位置在给定未掩盖上下文后**相互条件独立**：
   $$P(x_i, x_j \mid \tilde{X}) \approx P(x_i \mid \tilde{X}) P(x_j \mid \tilde{X})$$
   若对短语“人工智能”中的两个字同时掩盖为“人工[MASK][MASK]”，模型可能独立输出高概率词“智能”或“合成”，但如果独立采样组合，可能会产出“智成”这种不通顺的片段。
2. **训练与生成的分布不一致（Discrepancy）**：MLM 训练时必须依赖预先插入的 `[MASK]` 符号；而在自由开放生成场景中，模型需要从提示词开始一步步生成未知的后续文本，真实生成环境中不存在现成的 `[MASK]` 标记。
3. **推理时无法复用 KV 缓存**：在自回归生成中，由于 CLM 采用因果单向注意力，前面已生成的历史 Token 的 Key 和 Value 向量在后续生成中保持不变，可以直接缓存（详见 [/interview/llm-interview/kv-cache.md](/interview/llm-interview/kv-cache.md)）。如果强行用双向 MLM 做自回归生成，每次新增一个词，整段历史的上下文表示都会因双向注意力而改变，KV 缓存彻底失效，生成单条长序列的时间开销将急剧增加。

关于架构选型的更深层讨论，可参考 [/interview/llm-interview/decoder-only.md](/interview/llm-interview/decoder-only.md)。

---

## 评测标尺困惑度（Perplexity）：物理含义与使用边界

在模型训练与验证中，损失函数数值通常与具体实现细节绑定，而**困惑度（Perplexity, 简称 PPL）**则是语言建模中常用的归一化指标。

![困惑度（Perplexity）的物理直观：有效候选词分支度示意](/images/content-rebuild/dfe729bdd8bb9154/fig-4.png)

### 1. 困惑度与交叉熵的数学转换

对于长度为 $T$ 的测试序列 $X = (x_1, x_2, \dots, x_T)$，困惑度定义为测试集序列联合概率倒数的几何平均数：

$$\text{PPL}(X) = \left( \frac{1}{P(x_1, x_2, \dots, x_T)} \right)^{\frac{1}{T}} = \left( \prod_{t=1}^T \frac{1}{P(x_t \mid x_{<t})} \right)^{\frac{1}{T}}$$

两边取自然对数：

$$\ln \text{PPL}(X) = - \frac{1}{T} \sum_{t=1}^T \ln P(x_t \mid x_{<t}) = \mathcal{L}_{\text{CE}}$$

再取自然底数指数幂：

$$\text{PPL}(X) = \exp(\mathcal{L}_{\text{CE}})$$

> **底数说明**：若信息论推导中使用以 2 为底的对数 $\log_2$（单位为比特 bit），则 $\text{PPL} = 2^{\mathcal{L}}$；在 PyTorch 与 Hugging Face 的标准实现中，`CrossEntropyLoss` 默认以自然常数 $e$ 为底（单位为奈特 nat），因此换算公式为 $\text{PPL} = \exp(\text{Loss})$。

### 2. 算例与直观物理含义：有效候选分支数

假设某模型在测试集上的平均交叉熵损失为 $\mathcal{L} = 2.3026$：

$$\text{PPL} = \exp(2.3026) \approx 10.0$$

**物理含义**：PPL 等于 10，代表模型在预测下一个词时，面临的不确定性相当于**在 10 个等概率出现的备选词中随机猜一个（Effective Branching Factor）**。

* **极端无信息猜测**：若模型未学到任何有效规律，在大小为 $V$ 的词表上输出完全均匀的概率 $P = 1/V$：
  $$\mathcal{L} = -\ln(1/V) = \ln V \implies \text{PPL} = \exp(\ln V) = V$$
  此时有效候选分支数等于整个词表大小。
* **绝对确定且预测正确**：若模型在每个位置都能以概率 1 命中真实目标词：
  $$\mathcal{L} = -\ln(1) = 0 \implies \text{PPL} = \exp(0) = 1$$
  此时分支数为 1，代表模型预测完全确定。

### 3. PPL 横向对比的严格边界：分词器依赖性

![分词器切分粒度对 Token 数量与困惑度计算的影响与横向对比边界](/images/content-rebuild/dfe729bdd8bb9154/fig-5.png)

在工程评测中，**不能脱离分词器直接对比两个不同模型的 PPL 数值**：

1. **切分粒度影响单步条件难度**：假设模型 A 采用细粒度分词（如字节级），将单词 `Understanding` 切分为 4 个子词 Token；模型 B 采用大词表粗粒度分词，直接保留为 1 个完整 Token。模型 A 在预测后续子词时，由于常见字母搭配规律极其固定，单步候选空间狭窄，计算出的单步交叉熵往往较小，进而得出较低的 PPL；但模型 A 需要连续预测 4 步才能完成该单词的输出。
2. **跨模型公平对比标准**：当两个模型的分词器不同时，必须将指标归一化到相同的字符或字节基准上，例如计算**每字节比特数（Bits Per Byte, BPB）**：
   $$\text{BPB} = \frac{\mathcal{L}_{\text{CE}} \times \text{总 Token 数}}{\ln(2) \times \text{文本总字节数}}$$
   关于分词器的切分规则与常见词表设计，可参考 [/interview/llm-interview/token-basics.md](/interview/llm-interview/token-basics.md)。

---

## 面试高频追问：为什么预训练时 PPL 平稳下降，下游任务表现却可能剧烈波动？

在模型预训练监控中，开发者常会发现：验证集上的 Loss 和 PPL 一直在平滑单调下降，但在定期运行下游基准评测（如 MMLU 问答、GSM8K 数学、HumanEval 代码）时，准确率得分却出现剧烈震荡，甚至在某些训练阶段发生局部回退。

![验证集 PPL 持续平稳下降与下游多步推理基准非单调波动的解耦机制解析](/images/content-rebuild/dfe729bdd8bb9154/fig-6.png)

回答这一问题时，可以从以下四个机制层面解释：

### 1. 全局算术平均与关键决策 Token 的非对称重要性

交叉熵是对序列中所有 Token 的无差别算术平均。在一篇数千词的语料中，绝大多数 Token 是语法虚词、标点符号和常见过渡短语（如“的”、“在”、“is”、“the”）。
* 模型只需要将这些高频背景词的预测置信度微幅提升，整体 Loss 就会平稳降低，带动 PPL 持续下降。
* 但在下游推理任务中，决定答案正确与否的，往往只是序列末尾极少数的关键决策 Token（例如选择题中的选项标记 `A/B/C/D`，或代码中的逻辑运算符）。
* 如果模型在优化过程中过多拟合了背景词分布，而关键决策词的预测概率在候选阈值边缘出现微小扰动（例如从 0.51 跌至 0.49），平均 PPL 依旧在改善，但贪婪解码（Greedy Decoding）输出的正确率会直接从 1 跌至 0。

### 2. 连续概率指标与离散 0/1 判定的门限非线性效应

预训练优化的是连续可导的概率分布，而下游评估多数采用离散的硬性指标（如 Pass@1、Accuracy）。
* **教学概率示意**：在需要多步推理的任务中（例如解一道数学题需要连续推导 $K$ 步），若在教学简化模型中将多步生成展开为条件概率链条，最终答案正确的概率取决于各关键步骤依次成功的条件概率连乘：$P_{\text{final}} = \prod_{k=1}^K P(\text{step}_k \mid \text{step}_{<k})$（若在更简化的假设下视各关键步相对独立，则近似为各单步正确率的连乘）。
* 即使各单步条件概率都在稳步微增，在整条链条连乘未越过选对正确答案的判定阈值之前，整道题的离散判分依然为 0，通过率曲线在前期呈现平直停滞；一旦关键单步概率跨过临界点，整体准确率便会呈现阶跃式上升。这种离散判定带来的门限效应，会导致下游任务评测表现与验证集连续平滑下降的 PPL 曲线产生脱节（现象解析可参考 [/interview/llm-interview/emergent-abilities.md](/interview/llm-interview/emergent-abilities.md)）。

### 3. Teacher Forcing 与自由自回归解码的曝光偏差（Exposure Bias）

* **PPL 评测环境**：计算验证集 PPL 时，模型处于严格的 Teacher Forcing 模式。预测每个 Token 时，输入给模型的全部是人类撰写的高质量真实上文，模型从不面对自己产生的错误。
* **下游生成环境**：模型在测试下游任务时处于开环自回归生成模式。一旦前序步骤生成了轻微偏差的词，后续上下文就包含了模型自己的错误输出，进而导致误差沿时序累积放大。
* 因此，Teacher Forcing 下测出的低 PPL，无法完全等价于开环自由生成时的长程抗漂移能力。

### 4. 续写语料分布与指令测试格式的领域差异（Domain Mismatch）

预训练语料绝大多数是自然网页、文档和书籍，模型学习到的是自然的文本续写能力；而下游评测基准普遍采用特定的结构化 Prompt 模板（如 Few-shot 问答、`问题:... 答案:...`）。
* 在预训练的中早期，模型往往尚未形成对特定评测 Prompt 格式的稳定遵循，可能会输出无关的复述或续写，导致正则抽取答案失败并判为零分。
* 这一格式对齐的缺口通常依赖后续的指令微调（SFT）来系统弥补，具体开发阶段可参考 [/interview/llm-interview/training-pipeline.md](/interview/llm-interview/training-pipeline.md)。

---

## 常见面试误区辨析

| 常见错误观点 | 机制原因辨析 |
| :--- | :--- |
| **“Transformer 预训练时也是像推理一样逐词串行生成的”** | 混淆了训练与推理机制。预训练借助 Teacher Forcing 机制，通过将整段真实文本一次性输入并配合因果掩码矩阵，各位置在前向传播中是完全并行计算的，一次即可得出全序列各位置的预测损失。 |
| **“困惑度 PPL 越低越好，所以 PPL 最低的模型代码与数学能力一定最强”** | PPL 衡量的是对语料文本分布的整体拟合程度。如果两个模型的分词器不同，PPL 数值不具备可比性；且平均 PPL 会被大量背景词稀释，无法完全反映关键推理步骤的离散准确率。 |
| **“自回归模型各步输出概率和为 1，因此所有生成序列的概率加起来必然严格等于 1”** | 混淆了“单步词表概率分布”与“变长序列概率空间”。单步输出经过 Softmax，在词表维度上的概率和为 1；而整个变长序列空间的总概率和是否为 1，取决于自回归生成是否能在有限步内以概率 1 生成终止符 `[EOS]`（在概率论中称为语言模型的紧致性 Tightness 条件）。 |
| **“BERT 的 MLM 比 GPT 的 CLM 更好，因为 BERT 拥有双向上下文视野”** | 脱离任务类型谈优劣。双向视野对分类、判别和句子表征非常有效，但因其独立性假设和 `[MASK]` 机制，无法高效适配开放式长文本自回归生成，且无法复用 KV 缓存。 |

---

## 总结与延伸阅读

因果语言模型以因果掩码和最大似然估计为基础，将语言的理解与生成统一在“预测下一个 Token”的形式下。多分类交叉熵提供了工程可导的损失计算方式，而困惑度则提供了直观的加权分支度量标尺。

理解预训练目标之后，建议进一步阅读以下相关主题：

* 因果注意力掩码的具体矩阵运算与实现：[/interview/llm-interview/attention-masking.md](/interview/llm-interview/attention-masking.md)
* 现代大模型全面采用 Decoder-only 架构的原因：[/interview/llm-interview/decoder-only.md](/interview/llm-interview/decoder-only.md)
* 推理加速的核心技术 KV 缓存机制：[/interview/llm-interview/kv-cache.md](/interview/llm-interview/kv-cache.md)
* 预训练后进入指令微调（SFT）与人类偏好对齐的生命周期：[/interview/llm-interview/training-pipeline.md](/interview/llm-interview/training-pipeline.md)
