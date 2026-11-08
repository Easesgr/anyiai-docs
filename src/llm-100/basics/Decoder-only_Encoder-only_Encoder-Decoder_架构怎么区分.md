---
title: Decoder-only、Encoder-only 与 Encoder-Decoder 架构怎么区分？
icon: mdi:book-open-page-variant-outline
date: 2026-07-16
tag:
- 大模型100讲
- 基础篇
description: 详解 Transformer 三大架构谱系在注意力掩码、KV 缓存、数据流向与计算开销上的核心差异与选型依据。
---

同样基于 2017 年提出的 Transformer 结构，为什么 BERT 只能用来做理解和打标签，原版 Transformer 和 T5 擅长翻译，而今天的 GPT、LLaMA、Qwen 却几乎清一色采用纯解码器（Decoder-only）架构？

这三种形态的核心区别并不神秘，主要体现在三点：**注意力掩码（Mask）控制了哪些词互相可见**、**不同网络模块之间的数据流向**，以及**推理时能否高效复用中间计算缓存（KV Cache）**。

我们先用一个贯穿全文的具体例子建立直观认识。假设输入文本是：`["机器", "学习", "让"]`。
- **Encoder-only**：计算第 1 个词“机器”时，可以直接看到后面的“学习”和“让”，擅长整合全句上下文，输出整句的特征表示。
- **Decoder-only**：计算“机器”时只能看到它自己，计算“学习”时只能看到“机器”和“学习”。它严格阻止词看到未来内容，擅长根据上文一步步预测下一个词。
- **Encoder-Decoder**：把整句`["机器", "学习", "让"]`完整输入左侧的编码器，允许双向互看并提取全局表示；右侧的解码器一边看着已经翻译出来的英文词（单向不可看未来），一边通过跨注意力机制（Cross-Attention）查看编码器提取的全局表示，逐步生成目标译文。

![三大 Transformer 架构谱系全景](/images/content-rebuild/453cae1de79c079d/cover.png)

---

## 上下文可见性：双向掩码与因果掩码

在自注意力计算中，核心公式为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

其中 $M$ 是加在点积得分矩阵上的加性掩码矩阵（Attention Mask）。**$M$ 的取值规则直接决定了模型中每个位置能看到哪些上下文。**

![双向全连通掩码与因果下三角掩码计算对比](/images/content-rebuild/453cae1de79c079d/fig-1.png)

### 1. 双向掩码（Bidirectional Mask）

在 Encoder 模块中，所有有效 Token 之间互相对齐可见。对于非填充（Padding）位置，掩码取值为 0：

$$M_{i,j} = 0 \quad (\forall i, j)$$

此时位置 $i$ 的 Query 可以与序列中任意位置 $j$ 的 Key 计算点积。在 Softmax 归一化时，分母包含序列中所有 Token 的指数项。每个位置都能同时观察到前文和后文。

### 2. 因果掩码（Causal Mask）

在自回归生成模块中，为了防止模型在预测下一个词时提前看到后面的词，掩码被设计为下三角矩阵：

$$M_{i,j} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}$$

当 $j > i$（即目标位置在当前位置之后）时，$M_{i,j} = -\infty$。在 Softmax 计算中，$e^{-\infty} = 0$，使得位置 $i$ 对未来位置 $j$ 的注意力权重被严格置零。当前位置在计算输出时，无法获取任何来自后方 Token 的信息。

<ReadMoreLock />

---

## 逐步算例：3 个 Token 下的注意力权重对比

假设输入序列为 3 个 Token：`["大", "模", "型"]`。假设模型某一层计算出的未加掩码点积得分 $S = \frac{Q K^T}{\sqrt{d_k}}$ 如下：

$$S = \begin{bmatrix} 2.0 & 1.0 & 0.5 \\ 1.0 & 3.0 & 2.0 \\ 0.5 & 1.5 & 2.5 \end{bmatrix}$$

### 双向掩码（Encoder 模式）

由于 $M$ 为全 0 矩阵，直接对每一行计算 Softmax：

以第 1 行（Token "大"）为例：
- 计算指数项：$e^{2.0} \approx 7.389,\; e^{1.0} \approx 2.718,\; e^{0.5} \approx 1.649$
- 指数和：$7.389 + 2.718 + 1.649 = 11.756$
- 权重归一化：$[7.389/11.756,\; 2.718/11.756,\; 1.649/11.756] \approx [0.63, 0.23, 0.14]$

**结果分析**：Token "大" 在第一层就同时分配了注意力到其后方的 "模"（权重 0.23）和 "型"（权重 0.14）。

### 因果掩码（Decoder 模式）

对上三角位置填入 $-\infty$：

$$S + M = \begin{bmatrix} 2.0 & -\infty & -\infty \\ 1.0 & 3.0 & -\infty \\ 0.5 & 1.5 & 2.5 \end{bmatrix}$$

再对每一行执行 Softmax 归一化：
- 第 1 行：$\text{softmax}([2.0, -\infty, -\infty]) = [1.00, 0.00, 0.00]$
- 第 2 行：$\text{softmax}([1.0, 3.0, -\infty]) \approx [0.12, 0.88, 0.00]$
- 第 3 行：$\text{softmax}([0.5, 1.5, 2.5]) \approx [0.09, 0.25, 0.66]$

**结果分析**：Token "大" 只能看到它自己（权重 1.00）；Token "模" 只能看到前两项；只有末位的 "型" 能综合看到前文所有词。这就是自回归生成所需的信息可见性限制。

---

## Encoder-only 架构：全向表征与为什么不适合生成

Encoder-only 架构由多层双向自注意力模块与前馈网络（FFN）串联组成。代表模型包括 BERT、RoBERTa、DeBERTa 等。

![Encoder-only 数据流与双向语义特征提取](/images/content-rebuild/453cae1de79c079d/fig-2.png)

### 工作机制

1. **双向语义抽取**：所有 Token 并行进入计算，每一层的每个位置都聚合了整段输入的完整上下文。
2. **输出表征向量**：输出序列长度与输入完全相等。在分类或判别任务中，通常取句首标记（如 `[CLS]`）的向量接入分类头，或者对每个 Token 的输出向量接入序列标注头。

### 为什么 Encoder-only 不适合自回归生成？

如果强行用 Encoder 进行自回归逐字生成：每预测出一个新 Token $x_{t+1}$ 并拼接回输入序列后，因为缺少因果掩码的隔离，这个新 Token 会立刻参与到前序所有词 $x_1 \dots x_t$ 的双向注意力计算中。

这会导致两个后果：
1. 历史所有位置的中间状态全部被新词更新，先前计算好的 Key 和 Value 无法保持静态复用（即无法使用标准追加式的 KV Cache）。
2. 每生成一个新词，都必须把包含历史和新词的完整序列从头到尾重新跑一遍前向传播。若输出长度为 $N$，累积的总自注意力计算量将从有缓存时的 $\mathcal{O}(N^2)$ 增加到 $\mathcal{O}(N^3)$。

因此，纯 Encoder 结构适合用于不需要逐词生成的判别式任务，如文本分类、实体识别以及向量检索模型（Embedding 提取，可参考 [/llm-100/basics/Embedding_是什么_大模型如何把文字变成向量.md](/llm-100/basics/Embedding_是什么_大模型如何把文字变成向量.md)）。

---

## Decoder-only 架构：因果掩码与 KV Cache 的工程配合

Decoder-only 架构移除了编码器和跨注意力模块，由多层带有因果掩码的自注意力层与 FFN 组成。现代主流生成式大语言模型（如 GPT 系列、LLaMA、Qwen）几乎全部采用这一形态。

![Decoder-only 自回归单步生成与 KV Cache 单向追加机制](/images/content-rebuild/453cae1de79c079d/fig-3.png)

### 运行时的两个阶段：Prefill 与 Decode

Decoder-only 的推理在工程上明确分为两个计算特征不同的阶段：

1. **预填充阶段（Prefill）**：
   - 输入为长度为 $N$ 的用户提示词（Prompt）。
   - 虽然带有因果掩码，但在矩阵计算上，整个 Prompt 是并行送入 GPU 进行一次前向传播的。
   - 在常规提示词长度与批大小下，通常呈现计算密集型（Compute-bound）特征，张量计算单元（如 Tensor Core）利用率较高（若提示词极短或显卡算力未饱和则可能受访存制约）。
   - 前向过程中计算出的历史 Token 的 Key 和 Value 被存入显存，构成初始的 KV Cache。

2. **单步解码阶段（Decode）**：
   - 每次只将上一步刚生成的**单个 Token**（输入维度为 $1 \times d$）传入模型。
   - 当前 Token 投影生成当前的 Query $Q_{\text{curr}}$、Key $K_{\text{curr}}$、Value $V_{\text{curr}}$。
   - $K_{\text{curr}}$ 和 $V_{\text{curr}}$ 直接追加到显存中已有的 KV Cache 尾部。
   - $Q_{\text{curr}}$ 与包含全部历史的缓存进行点积注意力计算：
     $$\text{Output} = \text{softmax}\left(\frac{Q_{\text{curr}} [K_{\text{history}}; K_{\text{curr}}]^T}{\sqrt{d_k}}\right) [V_{\text{history}}; V_{\text{curr}}]$$
   - 模型输出下一个 Token 的概率分布，采样后继续循环。
   - 该阶段每次只算一个 Token，计算量小但需要频繁从显存读取大容量的 KV 缓存，在小批量推理时通常属于显存带宽受限（Memory-bound）操作（当并发批处理极大时算力占用显著上升）。

### 为什么因果掩码是 KV Cache 能够工作的数学前提？

在因果掩码下，第 $t$ 个词的隐状态仅由第 $1 \dots t$ 个词决定，与未来时刻出现的第 $t+1$ 个词无关。因此，之前算好的 $K_1 \dots K_t$ 和 $V_1 \dots V_t$ 在后续步骤中数值始终固定不变，不需要重新计算。这使得单步解码的计算时间只与序列长度成线性关系，而不是重新跑全序列。

---

## Encoder-Decoder 架构：跨注意力连接双流

标准 Encoder-Decoder 架构包含两个子网络，典型代表为 2017 年的原始 Transformer 以及后来的 T5 和 BART。

![Encoder-Decoder 架构的双流交互与 Cross-Attention 算子](/images/content-rebuild/453cae1de79c079d/fig-4.png)

### 数据流拆解

- **Encoder 侧**：接收源端输入（如中文原句），执行**双向自注意力**，输出全局表示矩阵 $H_{\text{enc}} \in \mathbb{R}^{S \times d}$（$S$ 为源序列长度）。
- **Decoder 侧**：接收目标端已生成的 Token（如已翻译出的英文片段）。Decoder 内部的每个 Transformer 层包含两道注意力计算：
  1. **带掩码的自注意力（Masked Self-Attention）**：仅在目标端序列内部计算因果注意力，防止看到未来的译文。
  2. **跨注意力（Cross-Attention）**：打通源端与目标端的信息连接。

### Cross-Attention 的输入来源

理解跨注意力时，最关键的是看清 $Q$、$K$、$V$ 分别来自哪里：

设 Decoder 当前自注意力层的输出为 $H_{\text{dec}} \in \mathbb{R}^{T \times d}$（$T$ 为目标序列长度），Encoder 最后一层的输出为 $H_{\text{enc}} \in \mathbb{R}^{S \times d}$：

$$Q = H_{\text{dec}} W_Q, \quad K = H_{\text{enc}} W_K, \quad V = H_{\text{enc}} W_V$$

$$\text{CrossAttention}(H_{\text{dec}}, H_{\text{enc}}) = \text{softmax}\left(\frac{(H_{\text{dec}} W_Q) (H_{\text{enc}} W_K)^T}{\sqrt{d_k}}\right) (H_{\text{enc}} W_V)$$

- **Query 来自 Decoder**：表示“当前解码步骤正在寻找哪些语义”；
- **Key 和 Value 来自 Encoder**：表示“源端输入提供了哪些全局信息可供检索”。

### 推理时的缓存状态

在解码生成阶段：
1. **Encoder 仅执行一次前向计算**。其输出映射得到的 Cross-Attention Key 和 Value 在整个解码周期内保持不变，缓存一次即可常驻显存。
2. **Decoder 端的自注意力仍需动态追加维护**（Self KV Cache）。
3. 因此，Encoder-Decoder 架构推理时显存中需同时保存静态的 Cross KV Cache 与动态递增的 Self KV Cache 两套缓存。

---

## 为什么现代超大模型普遍收敛于 Decoder-only？

在参数量较小（如 1B 以下）的专用序列映射任务（如机器翻译、文本摘要）中，Encoder-Decoder 架构往往能凭借双向源端编码取得优秀的参数表现。但随着参数量扩大到百亿甚至千亿级别，业界几乎全线转向了 Decoder-only 架构。

这一演变并非单一维度的性能优势，而是由数据吞吐、任务表达和工程调度三方面共同驱动的权衡结果：

![三大架构在计算预算与显存负载调度维度的对比矩阵](/images/content-rebuild/453cae1de79c079d/fig-5.png)

### 1. 预训练数据吞吐效率

- **Encoder-only**：常用掩码语言建模（如 BERT 的完形填空），每个批次只在被掩码的约 15% 的 Token 上计算损失和梯度，训练信号利用率较低。
- **Encoder-Decoder**：需要将文本人为切分成“受损输入”与“恢复目标”（如 T5 的 Span Corruption），切分策略带有强人为先验，切分后的片段较短，难以直接无损吞吐长篇连续文本。
- **Decoder-only**：采用纯粹的自回归下一个 Token 预测（Next-Token Prediction）。输入文本中的**每一个 Token 都是后一个 Token 的预测目标**。海量无标注语料无需任何结构改造，直接切分为 Token 流即可开展预训练。

### 2. 统一任务格式与上下文学习（In-Context Learning）

在 Decoder-only 架构中，系统提示词（Prompt）、示例（Few-shot）和回答（Completion）共处同一段扁平的序列空间内。模型在因果注意力机制下，天然能够结合上文示例进行续写式推理，多轮对话只需不断向后追加文本。而在 Encoder-Decoder 架构中，长提示词和短回答需要在编码器与解码器之间明确划分界限，在处理长上下文多轮对话时工程复杂度显著增加。

### 3. 推理显存调度与分布式并行

- **KV Cache 结构单一**：Decoder-only 架构只需维护单一维度的动态追加 KV Cache。像 PagedAttention 这类显存池化管理技术，可以对单一类型的缓存块进行高效分配与回收，无需为 Encoder 静态缓存和 Decoder 动态缓存设计双重非对称内存池。
- **并行切分均匀**：Decoder-only 整个网络由结构完全相同的 Transformer Block 纵向堆叠而成。在采用流水线并行（Pipeline Parallelism）或张量并行（Tensor Parallelism）切分模型时，各个 GPU 之间的计算量和显存占用完全对齐，更容易降低流水线等待时间。

---

## 架构选型对照与工程决策

在实际工程开发中，应依据输入输出模式与推理延迟要求进行理性选型，避免一律盲目套用超大模型：

| 对比维度 | Encoder-only | Decoder-only | Encoder-Decoder |
| :--- | :--- | :--- | :--- |
| **代表模型** | BERT, RoBERTa, BGE-Embedding | LLaMA, Qwen, DeepSeek, GPT-4 | 原版 Transformer, T5, BART |
| **主要注意力掩码** | 全双向掩码 ($M = 0$) | 严格下三角因果掩码 | 编码器双向，解码器因果 + Cross |
| **输入输出形式** | 输入序列，输出等长表征向量 | 输入文本前缀，自回归逐词生成 | 输入源序列，自回归生成目标序列 |
| **推理 KV 缓存** | 无需（也无法直接使用追加缓存） | 单一动态追加 KV Cache | 静态 Cross Cache + 动态 Self Cache |
| **计算与瓶颈特征** | 一次性前向计算，通常计算密集 | Prefill 偏计算密集，Decode 常见显存带宽受限（受批大小影响） | 编码段偏计算密集，解码段常见带宽受限 |
| **典型适用场景** | 文本分类、命名实体识别、语义检索向量提取 | 通用问答、多轮对话、代码生成、复杂推理 | 固定语言对机器翻译、特定格式重写与摘要 |

![Transformer 架构演进与技术选型决策树](/images/content-rebuild/453cae1de79c079d/fig-6.png)

---

## 可检验练习

为了检验你对三种架构可见性与计算机制的掌握程度，请尝试完成以下两道推演题：

### 练习 1：手算因果掩码作用结果

假设输入包含 2 个 Token：`["你", "好"]`，当前层未经缩放的点积得分矩阵为：

$$S = \begin{bmatrix} 3.0 & 2.0 \\ 1.0 & 4.0 \end{bmatrix}$$

1. 写出 Decoder-only 因果掩码矩阵 $M$ 的具体形式。
2. 计算加上掩码后的第 1 行与第 2 行的 Softmax 权重分布（保留两位小数）。

<details>
<summary>点击查看参考答案与解析</summary>

1. **因果掩码矩阵**：
   $$M = \begin{bmatrix} 0 & -\infty \\ 0 & 0 \end{bmatrix}$$

2. **权重计算**：
   - 第 1 行：$S + M = [3.0, -\infty]$。
     $$\text{softmax}([3.0, -\infty]) = [1.00, 0.00]$$
   - 第 2 行：$S + M = [1.0, 4.0]$。
     $$e^{1.0} \approx 2.72,\quad e^{4.0} \approx 54.60,\quad \text{和} \approx 57.32$$
     $$\text{softmax}([1.0, 4.0]) = \left[ \frac{2.72}{57.32}, \frac{54.60}{57.32} \right] \approx [0.05, 0.95]$$

**结论核对**：第 1 个 Token 对后方 Token 的注意力权重严格为 0；第 2 个 Token 同时看到前文（0.05）和自身（0.95）。
</details>

### 练习 2：Cross-Attention 形状推导

在 Encoder-Decoder 架构中，假设源序列长度为 $S = 128$，在训练阶段（Teacher Forcing 模式）输入的目标序列长度为 $T = 32$，隐藏层维度 $d = 512$。此时 Decoder 正在计算第 3 层的 Cross-Attention：
1. Query、Key、Value 的矩阵形状分别是什么？
2. 注意力得分矩阵 $Q K^T$ 的维度是什么？这一步需要加下三角因果掩码吗？为什么？

<details>
<summary>点击查看参考答案与解析</summary>

1. **矩阵形状**：
   - **训练并行阶段**：$Q$ 来自 Decoder 端，形状为 $(32, 512)$；$K$ 与 $V$ 来自 Encoder 最终输出，形状均为 $(128, 512)$。（注：若在自回归单步推理解码且启用 KV Cache 场景下，当前步仅输入 1 个 Token，$Q$ 的形状则为 $(1, 512)$）。

2. **得分矩阵与掩码**：
   - 训练阶段 $Q K^T$ 的形状为 $(32, 128)$（单步推理时为 $(1, 128)$）。
   - **不需要加因果掩码**。因为源序列（128 个 Token）是已经输入完毕的完整文本，当前解码器生成的每一个目标词都有权且必须观察源端的所有信息。因此 Cross-Attention 仅在源端有 Padding 补齐时屏蔽填充位置，而不施加下三角因果截断。
</details>

---

## 知识延伸与下节导引

三种架构的掩码设计与连接机制，直接决定了模型在训练时的学习目标。BERT 选择了双向掩码语言建模（MLM），GPT 选择了自回归因果语言建模，而 T5 选择了文本片段破坏恢复（Span Corruption）。

在下一讲 [/llm-100/basics/GPT_BERT_T5_的核心差异是什么.md](/llm-100/basics/GPT_BERT_T5_的核心差异是什么.md) 中，我们将基于本讲建立的掩码与数据流视角，进一步拆解这三款标杆模型的预训练目标差异以及它们在微调阶段的具体表现。
