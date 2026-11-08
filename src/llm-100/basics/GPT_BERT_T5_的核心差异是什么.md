---
title: GPT、BERT 与 T5 的核心差异是什么？
icon: mdi:book-open-page-variant-outline
date: 2026-07-16
tag:
- 大模型100讲
- 基础篇
description: 对比 GPT、BERT 与 T5 在架构、注意力掩码、预训练目标与微调映射上的机制差异，解释为什么 Decoder-only 成为主流及各自适用边界。
---

在了解了 Transformer 的基本模块（可参考上一讲 [/llm-100/basics/Decoder-only_Encoder-only_Encoder-Decoder_架构怎么区分.md](/llm-100/basics/Decoder-only_Encoder-only_Encoder-Decoder_架构怎么区分.md)）后，很多人会产生一个疑问：同样都基于自注意力计算（详见 [/llm-100/basics/Attention_机制到底在算什么.md](/llm-100/basics/Attention_机制到底在算什么.md)），为什么 GPT、BERT 和 T5 的表现形态和使用方式完全不同？

核心结论是：**决定模型表征特性和下游任务形态的，不是自注意力公式本身，而是模型架构骨架（Encoder 还是 Decoder）、注意力掩码矩阵的可见方向，以及预训练阶段设计的损失函数目标**。

为了让对比保持直观，本文用一个包含 5 个词的具体中文句子作为贯穿算例：

> 示例序列：`["深度", "学习", "改变", "机器", "视觉"]`（简记为 $[w_1, w_2, w_3, w_4, w_5]$）

![三大经典范式概览](/images/content-rebuild/5e78e5be599611d8/cover.png)

---

## 1. GPT 范式：纯解码器与自回归因果语言模型

GPT 采用纯解码器（Decoder-only）结构。它的核心逻辑是**单向可见**与**按顺序预测下一个词（自回归）**。

### 注意力掩码矩阵：因果下三角

在标准自注意力计算中，每个位置都要与所有位置计算相关性：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

为了防止模型在预测第 $t$ 个词时提前看到第 $t+1$ 个词的信息，GPT 引入了一个下三角掩码矩阵 $M$：

$$M_{i,j} = \begin{cases} 0 & i \ge j \\ -\infty & i < j \end{cases}$$

当 $M_{i,j} = -\infty$ 时，Softmax 之后对应的注意力权重严格为 0。也就是说，词 $w_3$（“改变”）只能关注到自身以及前面的 $w_1$（“深度”）、$w_2$（“学习”），无法看到后面的 $w_4$（“机器”）和 $w_5$（“视觉”）。

### 预训练目标：因果语言建模（Causal LM）

给定序列 $X = (w_1, w_2, \dots, w_n)$，GPT 的预训练目标是最小化负对数似然损失：

$$\mathcal{L}_{\text{CLM}}(\theta) = -\sum_{i=1}^n \log P(w_i \mid w_1, w_2, \dots, w_{i-1}; \theta)$$

在我们的 5 词示例中，模型在一次前向传播中同时进行多步预测：
- 输入 $w_1$（深度），预测 $w_2$（学习）；
- 结合 $w_1, w_2$，预测 $w_3$（改变）；
- 结合 $w_1, w_2, w_3$，预测 $w_4$（机器）；
- 结合 $w_1, w_2, w_3, w_4$，预测 $w_5$（视觉）。

每个预测位置都会计算交叉熵损失并回传梯度，序列中几乎所有位置都在参与训练。

![GPT 预训练机制与因果掩码](/images/content-rebuild/5e78e5be599611d8/fig-1.png)

### 早期微调映射方式（以 GPT-1 为例）

在 GPT-3 兴起“提示词（Prompt）与上下文学习”之前，GPT-1 和 GPT-2 微调判别任务仍然需要微调模型参数。

例如做文本分类时，在句子末尾追加一个特殊的终止符 `[Extract]`：

$$\text{输入：}[w_1, w_2, w_3, w_4, w_5, \text{[Extract]}]$$

由于因果掩码的存在，只有最后一个标记 `[Extract]` 汇聚了前文所有词的上下文信息。微调时，取出该位置顶层的输出向量 $h_{\text{extract}}$，外接一个线性层计算分类概率。

<ReadMoreLock />

---

## 2. BERT 范式：纯编码器与双向掩码语言模型

BERT 采用纯编码器（Encoder-only）结构。它的核心逻辑是**全向双向可见**与**完形填空**。

### 注意力掩码矩阵：全向双向互见

BERT 的注意力掩码矩阵除填充位（Padding）外全为 0：

$$M_{i,j} = 0 \quad (\forall i, j)$$

序列中任何一个词都可以同时看到它左边和右边的所有词。例如计算“改变”的表征时，它不仅能结合左边的“深度”、“学习”，也能结合右边的“机器”、“视觉”。这种双向上下文表征对抽取式任务（如命名实体识别、文本相似度度量）非常有利。

但全向双向掩码也直接封死了像 GPT 那样自回归生成文本的能力：如果在双向注意力下直接做“预测下一个词”，模型会直接通过未来位置看到答案，导致训练退化为简单的恒等复制。

### 预训练目标：掩码语言模型（MLM）与下句预测（NSP）

为了在双向结构下训练模型，BERT 采用了类似完形填空的掩码语言模型：

1. 在输入序列中随机选取 15% 的词作为预测目标。
2. 对这 15% 选中的词执行 **80/10/10 替换策略**：
   - 80% 的概率替换为特殊标记 `[MASK]`；
   - 10% 的概率替换为一个随机词；
   - 10% 的概率保留原词不变。

假设在贯穿示例中，第 3 个词 $w_3$（“改变”）被选中，且进入 80% 分支被替换为 `[MASK]`：

$$\text{模型输入：}[\text{[CLS]}, \text{深度}, \text{学习}, \text{[MASK]}, \text{机器}, \text{视觉}, \text{[SEP]}]$$

模型只在对应的 `[MASK]` 标记位置（即拼接 `[CLS]` 后的第 4 个位置 / 索引 3）计算预测原词“改变”的交叉熵损失，其余未被选中的位置不计算该项损失：

$$\mathcal{L}_{\text{MLM}}(\theta) = -\sum_{i \in \mathcal{M}} \log P(w_i \mid \tilde{X}; \theta)$$

此外，BERT 还设计了下一句预测（Next Sentence Prediction, NSP）二分类目标，判断拼接在 `[SEP]` 两侧的句子 A 与句子 B 是否是连续上下文（在后续的 RoBERTa 等工作中，NSP 往往被移除或替换）。

![BERT 掩码机制与分布差异](/images/content-rebuild/5e78e5be599611d8/fig-2.png)

### 为什么需要 80/10/10 策略？

这是为了缓解**预训练与微调阶段的输入分布不一致（Discrepancy）**：

在真实的下游任务（如分类、标注）中，文本里根本没有 `[MASK]` 这个标记。如果预训练阶段只在有 `[MASK]` 的时候才强迫模型去预测和校准语义，模型在下游微调面对纯正常文本时，特征提取能力就会受到干扰。

- **10% 随机词替换**：迫使模型不能盲目相信输入词本身，必须结合上下文检验语义一致性；
- **10% 保持原词不变**：让模型在面对真实正常词时，依然保留对当前词特征的有效编码能力。

---

## 3. T5 范式：编码器-解码器与片段破坏重建

Google 提出的 T5（Text-to-Text Transfer Transformer）重新回到了标准 Transformer 的编码器-解码器（Encoder-Decoder）双结构。它的目标是将所有自然语言任务统一为**输入文本 $\rightarrow$ 输出文本（Text-to-Text）**。

### 混合注意力结构

T5 将两套不同的注意力掩码组合在一个模型中：
- **Encoder 端**：使用全向双向注意力，完整阅读输入内容，不设因果掩码。
- **Decoder 端**：包含自注意力层与交叉注意力层。自注意力使用下三角因果掩码，只能看已生成的输出历史；交叉注意力（Cross-Attention）则以 Decoder 状态为 Query，完整关注 Encoder 输出的 Key 和 Value。

### 预训练目标：片段破坏（Span Corruption）

与 BERT 逐词打掩码不同，T5 采用连续片段破坏策略：
1. 同样选中输入文本中约 15% 的词，但把连续的词合并为一个片段（Span，平均长度约为 3 个词）。
2. 每个被破坏的片段用唯一的哨兵标记（Sentinel Token，如 `<extra_id_0>`, `<extra_id_1>` 等）替代。
3. **Decoder 的生成目标不是还原整句话，而只生成被破坏的片段及其对应的哨兵标记**。

回到 5 词示例：假设我们将中间的连续片段 `["改变", "机器"]`（即 $w_3, w_4$）选为破坏目标：

- **Encoder 输入**：`深度 学习 <extra_id_0> 视觉`
- **Decoder 目标**：`<extra_id_0> 改变 机器 <extra_id_1>`

与 BART 等直接重构整句的 Denoising Sequence-to-Sequence 模型相比，T5 的 Decoder 只需要解码较短的目标片段，有效降低了预训练期间 Decoder 端的自回归计算开销。

![T5 片段破坏与 Text-to-Text 统一机制](/images/content-rebuild/5e78e5be599611d8/fig-3.png)

### Text-to-Text 任务统一映射

T5 不为特定下游任务增加独立初始化的线性分类头，而是将所有任务通过任务前缀格式化为纯文本序列输入与输出：

- **文本分类任务**：输入 `classification: 深度学习改变机器视觉` $\rightarrow$ 解码器生成文本：`科技`。
- **机器翻译任务**：输入 `translate Chinese to English: 深度学习改变机器视觉` $\rightarrow$ 解码器生成文本：`Deep learning transforms computer vision.`。
- **文本相似度（数值回归）**：输入 `stsb sentence1: ... sentence2: ...` $\rightarrow$ 解码器直接生成浮点数字符串：`3.8`。

---

## 4. 三大范式的输入、掩码与损失计算逐算例对比

我们将示例句子 `["深度", "学习", "改变", "机器", "视觉"]`（简写为 $w_1 \sim w_5$）放入统一表格，对比三者在输入数据格式、掩码形式与损失计算上的具体区别：

| 对比维度 | GPT (Decoder-only) | BERT (Encoder-only) | T5 (Encoder-Decoder) |
| :--- | :--- | :--- | :--- |
| **模型输入形式** | $[w_1, w_2, w_3, w_4, w_5]$ | $[\text{[CLS]}, w_1, w_2, \text{[MASK]}, w_4, w_5, \text{[SEP]}]$ | **Encoder**: $[w_1, w_2, \langle e_0 \rangle, w_5]$<br>**Decoder**: $[\langle \text{pad} \rangle, \langle e_0 \rangle, w_3, w_4]$（训练输入，带前置起始标记） |
| **自注意力掩码** | 严格下三角矩阵（仅可见左侧与当前） | 全 0 矩阵（非 padding 区域全双向互通） | **Encoder**: 全双向<br>**Decoder**: 下三角 + Cross Attention |
| **预训练预测目标** | 对齐右移的序列：$[w_2, w_3, w_4, w_5, \text{[EOS]}]$ | 仅在被掩码的第 4 个位置预测原词 $w_3$ | Decoder 自回归生成目标：$[\langle e_0 \rangle, w_3, w_4, \langle e_1 \rangle]$ |
| **预训练单步损失贡献** | 全序列几乎 100% 的有效 token 参与计算 | 仅被选中的约 15% 掩码位置参与计算 | 仅 Decoder 端生成的片段内容及哨兵标记计算损失 |
| **单步表征计算流** | $h_t = f(w_1, \dots, w_t)$ | $h_t = f(w_1, \dots, w_L)$（全双向融合） | $h_{\text{enc}} = f(\text{全输入})$<br>$h_{\text{dec}, t} = f(y_{<t}, h_{\text{enc}})$ |

![同一算例下的掩码矩阵与预测目标对比](/images/content-rebuild/5e78e5be599611d8/fig-4.png)

### 下游微调代码逻辑对比

在代码实现上，三者适配下游任务（以句子分类为例）的方式存在明显分流：

```python
# 伪代码：展示三大模型在分类任务微调时的前向传播区别

# 1. BERT：依赖句子首位的 [CLS] 向量，外接线性分类头
def bert_finetune_step(input_ids, attention_mask, labels):
    outputs = bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
    # 取 [CLS]（索引为 0）的顶层隐藏状态
    cls_vector = outputs.last_hidden_state[:, 0, :]  # 形状: (batch, hidden_dim)
    logits = linear_classifier(cls_vector)           # 形状: (batch, num_classes)
    loss = cross_entropy(logits, labels)
    return loss

# 2. 早期 GPT-1：在末尾追加 [Extract]，取该位置向量外接线性分类头
def gpt1_finetune_step(input_ids, labels):
    # 序列格式已经处理为: [w1, w2, ..., wn, Extract]
    outputs = gpt_decoder(input_ids=input_ids)       # 内部自动应用因果掩码
    extract_vector = outputs.last_hidden_state[:, -1, :]  # 取末尾 Extract 处向量
    logits = linear_classifier(extract_vector)
    loss = cross_entropy(logits, labels)
    return loss

# 3. T5：无新增分类头，将分类标签作为纯文本进行自回归训练
def t5_finetune_step(encoder_input_ids, target_label_ids):
    # target_label_ids 是目标文字（如类别词 "科技"）对应的 token ids
    outputs = t5_model(input_ids=encoder_input_ids, labels=target_label_ids)
    # 直接使用 Decoder 生成文本的交叉熵损失
    loss = outputs.loss
    return loss
```

![微调结构与任务适配模式对比](/images/content-rebuild/5e78e5be599611d8/fig-5.png)

---

## 5. 架构演进与适用边界：为什么 Decoder-only 成为通用大模型主流？

在 2018 到 2020 年间，BERT 占据了理解与判别任务的主流，T5 成为序列转换基准的核心模型，而 GPT 坚持在自回归生成方向推进。但随着模型参数达到千亿级别，通用大语言模型几乎全线收敛到纯 Decoder-only 架构。这一演进的核心工程原因如下：

### 1. 统一交互界面与 In-Context Learning
BERT 依赖任务专用的输出头（Classification Head），无法直接通过人类自然对话完成新任务。而因果自回归模型从一开始就在做“根据上文预测后文”。当参数量扩展后，无论输入是问答、翻译、推理步骤还是多轮对话，在计算图上都是统一的自回归续写（Prompt + In-Context Learning），无需为每个任务重新初始化或微调特定结构。

### 2. 推理状态缓存（KV Cache）的工程复杂度
Decoder-only 在自回归生成时只需维护一套单向的自注意力 KV Cache（存储 $K$ 和 $V$ 向量避免重复计算）。而 Encoder-Decoder 架构在生成长文本时，必须先对输入执行一次完整的双向编码并保存为静态的 Cross-Attention KV Cache，随后 Decoder 还要维护自身的 Causal Self-Attention KV Cache。在长上下文多轮对话场景中，双重状态维护和通信开销更加复杂。

### 3. 无监督预训练的损失监督密度
在预训练的前向计算中，BERT 的 MLM 仅在约 15% 的掩码位置计算交叉熵损失，T5 也仅对抽取的破坏片段计算解码损失；而 Decoder-only 因果模型除第 1 个 token 外，序列中每一个有效 token 都在预测下一个 token 并计算交叉熵损失。虽然双向模型中所有 token 都会通过注意力机制接收反向梯度并更新参数，但因果自回归模型在单步前向传播中提供了更密集的监督信号，在超大规模无标签预训练中的数据利用吞吐更具优势。

![架构选型与任务适配决策边界](/images/content-rebuild/5e78e5be599611d8/fig-6.png)

### 现代工程中的适用边界

纯 Decoder-only 成为通用大模型主流，并不意味着 Encoder-only 和 Encoder-Decoder 失去了存在价值：

- **Encoder-only（BERT / RoBERTa / BGE）**：
  在**文本嵌入向量生成（Dense Retrieval / Embedding）**、**高 QPS 抽取式语义分类与命名实体识别（NER）**中仍然是主力。它们无需自回归逐步解码，一次前向推理即可输出完整序列的双向向量，在延迟敏感和纯判别式系统中推理吞吐和部署成本明显优于生成式大模型。
- **Encoder-Decoder（T5 / BART / mT5）**：
  在**受控条件转换任务**（例如长文档特定摘要、结构化数据转换为固定格式说明）中，Encoder 的全向注意力能更严谨地锁定原文约束，Decoder 专注于目标生成，依然具备独特的架构优势。

---

## 可检验练习

### 练习 1：掩码与微调机制验证
**题目**：在 BERT 的预训练中，如果将“80/10/10”策略简化为“100% 直接替换为 `[MASK]`”，模型在预训练时能正常收敛吗？在微调用于真实分类任务时可能会发生什么问题？

**检验标准**：
- 答出预训练时模型依然能够收敛，因为完形填空损失能够正常回传；
- 明确指出微调阶段真实文本中没有 `[MASK]` 标记，导致预训练学到的特定激活模式在微调时缺失，造成特征提取效果下降（输入分布不一致）。

### 练习 2：推理缓存开销分析
**题目**：假设有一段 1000 词的输入提示词，需要生成 100 个词的回答。请对比 Decoder-only 模型与 Encoder-Decoder 模型在解码第 50 个词时，各自 KV Cache 的存储对象有何区别？

**检验标准**：
- **Decoder-only**：KV Cache 存储前 1000 个提示词加上已生成的 49 个输出词，共 1049 个 token 的自注意力 Key 与 Value。
- **Encoder-Decoder**：需要存储两份缓存——一份是 Encoder 处理 1000 个提示词后产生的 Cross-Attention Key 与 Value（固定长度 1000，生成过程中保持静态）；另一份是 Decoder 自身前 49 个词的自注意力 Key 与 Value。两者的状态机维护路径不同。

---

## 关联索引

- 架构类型基础：[/llm-100/basics/Decoder-only_Encoder-only_Encoder-Decoder_架构怎么区分.md](/llm-100/basics/Decoder-only_Encoder-only_Encoder-Decoder_架构怎么区分.md)
- 注意力计算底层：[/llm-100/basics/Attention_机制到底在算什么.md](/llm-100/basics/Attention_机制到底在算什么.md)
- Transformer 整体骨架：[/llm-100/basics/Transformer_为什么能成为大模型的核心架构.md](/llm-100/basics/Transformer_为什么能成为大模型的核心架构.md)
