---
title: Encoder-only、Decoder-only 和 Encoder-Decoder 架构有什么区别？
icon: mdi:source-branch
tag:
- Transformer架构
- BERT
- GPT
- T5
description: 通过一个具体任务对比 Transformer 三大变体的注意力掩码构造、张量流向、推理 KV Cache 差异及工程选型依据。
---

![架构拓扑演化封面图：展示三种变体结构的注意力连接形态](/images/content-rebuild/df0f8eb5702da85f/cover.png)

在完成 [Multi-Head Attention 机制](/fundamentals/transformer-arch/multi-head-attention.md) 与 [位置编码原理](/fundamentals/transformer-arch/positional-encoding.md) 的学习后，我们已经熟悉了缩放点积注意力的核心计算形式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

原始论文《Attention Is All You Need》提出的是一个包含编码器和解码器的双塔模型，用于机器翻译。然而在后续发展中，社区逐渐分化出三类主流架构：**Encoder-only（纯编码器）**、**Decoder-only（纯解码器）** 以及 **Encoder-Decoder（编码器-解码器）**。

决定它们差异的核心不是堆叠层数的多少，而是两件事：**注意力掩码矩阵 $M$ 如何控制信息可见性**，以及**张量在网络各模块间的流动路径**。

---

## 一个贯穿例子：处理同一文本时的三种视角

假设我们有一句文本：`今天 天气 很好`，分词后包含 3 个 Token。

1. **场景 A：情感分类（序列理解）**
   * 目标是判断整句话是正面还是负面。输入序列在计算前已经完整已知。模型应该在处理 `天气` 时同时参考前后的 `今天` 和 `很好`。
   * **对应架构**：**Encoder-only**。所有位置彼此可见，无时序约束，单次前向传播输出特征。
2. **场景 B：句子续写（自由文本生成）**
   * 目标是接着往下写出 `，适合散步`。在模型预测第 4 个词时，它只能依据前 3 个词。在训练时，即使整个句子都放在显存里，也必须强行屏蔽未来词，防止模型“抄近道”。
   * **对应架构**：**Decoder-only**。使用严格的因果掩码，模型只能单向关注自身及历史 Token。
3. **场景 C：机器翻译（非对称序列转换）**
   * 目标是将英文 `Good weather today` 翻译为中文 `今天天气很好`。英文输入句子是完整已知的，需要全局双向理解；中文输出句子则是逐词生成的，生成时既需要关注已经生成的中文前文，又需要随时回查英文源句。
   * **对应架构**：**Encoder-Decoder**。编码器双向处理英文，解码器单向自回归生成中文，并通过交叉注意力查阅编码器的输出。

![三种拓扑结构全景对比图：展示双向全连接、因果单向与双塔交叉结构](/images/content-rebuild/df0f8eb5702da85f/fig-1.png)

下表汇总了三者的结构特征与计算模式：

| 架构类型 | 包含子模块 | 自注意力掩码形态 | 是否包含 Cross-Attention | 输入与输出时序关系 | 典型代表 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Encoder-only** | 仅 Encoder 堆叠 | 全可见（无因果限制） | 否 | 输入已知，单步前向输出表征 | BERT, RoBERTa |
| **Decoder-only** | 仅 Decoder 堆叠 | 因果下三角（屏蔽未来） | 否 | 自回归，逐 Token 向后预测 | GPT 系列, LLaMA |
| **Encoder-Decoder** | Encoder + Decoder | Encoder 全可见；Decoder 因果 | 是（Decoder 查询 Encoder） | 源序列全局编码，目标端逐步解码 | 原始 Transformer, T5, BART |

<ReadMoreLock />

---

## Encoder 结构：双向自注意力与全可见掩码

在纯编码器架构中，所有层的自注意力都允许全向交互。给定序列输入 $X \in \mathbb{R}^{N \times d}$，其线性投影得到：

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

在没有批处理填充（Padding）的理想情况下，掩码矩阵 $M$ 中的元素全部为 0：

$$M_{i,j} = 0 \quad (\forall i, j \in [1, N])$$

![Encoder 双向自注意力与全可见掩码机制图](/images/content-rebuild/df0f8eb5702da85f/fig-2.png)

### 信息流动与计算特征

* **双向上下文整合**：在计算第 $i$ 个位置的输出向量时，它的 Query 向量与所有位置（$j \in [1, N]$）的 Key 计算相似度。`Token_1` 可以直接关注 `Token_3`，`Token_3` 也可以直接关注 `Token_1`。
* **Padding Mask 的作用**：当多条不同长度的句子打包为一个批次（Batch）计算时，短句后方会补齐占位的 `<pad>` 标记。为了防止有效词将注意力权重分给无意义的占位符，工程上会在掩码矩阵的对应列填入极大负数（如 $-10^4$ 或 $-\infty$），使得经过 Softmax 后的权重精确为 0。
* **计算效率**：单次前向传播即可计算出所有 Token 的高阶表征，适合文本分类、特征抽取和密集检索向量生成。

---

## Decoder 结构：因果掩码防止未来信息泄露

自回归文本生成的基本规则是：预测第 $t$ 个词时，模型只能依赖前 $t-1$ 个词。然而，如果每次只输入前 $t-1$ 个词，训练时序列长度为 $N$ 的句子就需要执行 $N$ 次前向传播，效率极低。

为了在训练阶段能够像 Encoder 一样**一次性并行输入整条句子**，Decoder 引入了**因果掩码（Causal Mask）**。它在数学上表现为一个上三角被屏蔽的矩阵。

![Decoder 因果掩码矩阵相加与 Softmax 截断过程](/images/content-rebuild/df0f8eb5702da85f/fig-3.png)

### 因果掩码的计算过程

设输入序列长度为 $N$，注意力未归一化得分矩阵为 $S = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{N \times N}$。因果掩码矩阵 $M \in \mathbb{R}^{N \times N}$ 定义为：

$$M_{i,j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

以序列长度 $N=4$ 为例，掩码矩阵的具体取值为：

$$M = \begin{pmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

将其直接加到得分矩阵上：$S_{\text{masked}} = S + M$。在执行 Softmax 归一化时：

$$A_{i,j} = \frac{\exp(S_{i,j} + M_{i,j})}{\sum_{k=1}^N \exp(S_{i,k} + M_{i,k})}$$

当 $j > i$（未来位置）时，$S_{i,j} + M_{i,j} = -\infty$，此时 $\exp(-\infty) = 0$。最终得到的注意力权重矩阵呈现标准的下三角形式：

$$A = \begin{pmatrix} 1.0 & 0 & 0 & 0 \\ a_{2,1} & a_{2,2} & 0 & 0 \\ a_{3,1} & a_{3,2} & a_{3,3} & 0 \\ a_{4,1} & a_{4,2} & a_{4,3} & a_{4,4} \end{pmatrix}$$

借助因果掩码，模型在训练时可以利用 **Teacher Forcing** 一次性计算所有位置的下一个词预测损失；在推理时则按步串行吐字。

---

## Encoder-Decoder 连接机制：Cross-Attention 的张量流动

Encoder-Decoder 架构由两组子网络组成。源序列经过 Encoder 得到最终表征 $H_{\text{enc}}$；目标序列则输入 Decoder 逐步生成。连接这两个子网络的枢纽是 **交叉注意力（Cross-Attention）**。

在标准 Encoder-Decoder 的解码层中，包含两个串联的注意力计算：
1. **Masked Causal Self-Attention**：解码端已生成 Token 之间的自注意力，使用因果掩码防止偷看后文。
2. **Cross-Attention**：将解码端当前生成的语义作为查询，去编码端提取对应的输入信息。

![Cross-Attention 张量投影与跨序列交互拓扑图](/images/content-rebuild/df0f8eb5702da85f/fig-4.png)

### Query、Key、Value 的来源与张量维度

在 Cross-Attention 中，张量的输入来源被解耦：

* **Query（$Q$）**：来自 Decoder 前一层对目标序列的隐层输出，形状为 $(L_{\text{target}}, d)$。
* **Key（$K$）与 Value（$V$）**：来自 Encoder 顶层对源序列的最终输出 $H_{\text{enc}}$，形状均为 $(L_{\text{source}}, d)$。

点积计算得到的注意力矩阵形状为：

$$S_{\text{cross}} = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{L_{\text{target}} \times L_{\text{source}}}$$

这个矩阵的行数对应当前已解码的目标词数，列数对应输入源句的词数。它表示**每一个正在生成的目标词，应该在源句的各个词上分配多少权重**。

需要特别指出的是，Cross-Attention **没有因果掩码限制**。每一个生成步都可以查阅源句的全部 Token（仅需屏蔽源句本身的 Padding 占位符）。

---

## 推理阶段的状态维护：KV Cache 的行为差异

在自回归推理（逐词生成）阶段，模型每生成一个新 Token，都需要将其拼接到历史序列中参与下一步计算。由于自注意力中历史 Token 的线性投影在后续计算中数值不变，为了避免 $O(N^2)$ 的重复矩阵相乘，系统会引入 **KV Cache（键值缓存）**。

不同架构在推理时对 KV Cache 的维护方式截然不同：

![不同架构在推理阶段的 KV Cache 行为与显存占用对比图](/images/content-rebuild/df0f8eb5702da85f/fig-5.png)

### 1. Encoder-only
无需自回归生成。输入整段文本后，执行一次前向计算即完成，**不需要维护 KV Cache**。

### 2. Decoder-only
* 在第 1 步 Prefill（提示词预填充）阶段，计算出 Prompt 中所有 Token 的 Key 和 Value 并存入缓存。
* 后续每生成 1 个 Token（Decode 阶段），仅计算当前最新 Token 的 $Q, K, V$。
* 将新的 $K, V$ 拼接进缓存中。自注意力 KV Cache 的长度随生成步数逐步累加（步长递增 1）。

### 3. Encoder-Decoder
* **Encoder 端**：在推理开始时仅执行一次编码计算，输出 $H_{\text{enc}}$ 后即结束任务。
* **Cross-Attention KV Cache**：由 $H_{\text{enc}}$ 投影生成的 Key 和 Value 在整个解码过程中保持不变。因此只需在第 1 步计算一次，其大小固定为 $L_{\text{source}}$，**后续解码步全程静态复用**。
* **Decoder Self-Attention KV Cache**：与 Decoder-only 类似，记录目标端已生成的 Token，随着生成步数动态递增。

---

## 工程选型依据：为什么现代 LLM 倾向于 Decoder-only？

在模型规模较小（如百兆参数量级）时，三大架构各自在特定任务上表现突出：

![任务形态与架构选型决策流及权衡谱系图](/images/content-rebuild/df0f8eb5702da85f/fig-6.png)

* **Encoder-only**：在文本分类、语义匹配和重排等判别任务中，由于具备全局双向注意力且无自回归开销，延迟最低且表征能力强。
* **Encoder-Decoder**：在机器翻译、文本摘要、语音识别（如 Whisper）等非对称转换任务中，输入与输出天然异构，双塔解耦提供了良好的条件对齐能力。
* **Decoder-only**：在开放式生成、对话和通用推理中展现出极佳的扩展性。

### 现代通用大型语言模型为何大多收敛于 Decoder-only？

当模型参数规模上升至数十亿到数千亿（如 7B、70B 级别）时，工程实践全面转向 Decoder-only 架构，主要原因包含以下三点：

1. **系统实现与并行计算更统一**：Decoder-only 只有单一类型的注意力模块，没有 Encoder 与 Decoder 之间的跨层数据依赖。在进行张量并行（Tensor Parallelism）与流水线并行（Pipeline Parallelism）时，通信模式规整，不存在双塔之间由于长度不对称造成的计算气泡。
2. **显存管理与调度开销更低**：在推理服务框架（如 vLLM）中，核心挑战之一是管理不规则的 KV Cache 显存碎片。Encoder-Decoder 需要同时管理固定长度的 Cross KV Cache 与动态递增的 Self KV Cache；而 Decoder-only 仅需维护一条连续增长的键值链，大幅简化了 PagedAttention 等内存池化机制的实现。
3. **训练任务与上下文学习（In-context Learning）的泛化统一**：在 Decoder-only 框架下，无论是问答、分类还是翻译，都可以统一表述为“提示词前缀 + 下文预测”。模型在预训练时学习的通用自回归目标可以直接迁移到零样本或少样本提示（Few-shot Prompting）中，无需对输入端和输出端做专门的架构切分。

关于 BERT 与 GPT 在预训练任务、表征能力与工程特化上的进一步对比，请参考下一讲：[BERT 和 GPT 的核心区别是什么？](/fundamentals/transformer-arch/bert-vs-gpt.md)。

---

## 读者自测

1. 请写出序列长度为 3 时的因果掩码矩阵（Causal Mask），并说明当得分矩阵加上该掩码后，未来位置在 Softmax 后的计算结果为什么是严格的 0。
2. 在 Encoder-Decoder 架构中，Cross-Attention 的 Query、Key、Value 分别来自哪里？为什么 Cross-Attention 的 Key 和 Value 在推理生成阶段可以做静态缓存？
