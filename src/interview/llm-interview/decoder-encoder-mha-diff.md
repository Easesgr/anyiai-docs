---
title: Decoder 的自注意力和 Encoder 的自注意力有什么区别？
icon: mdi:compare
tag:
- Transformer
- 多头注意力
- Encoder-Decoder
description: 对比 Transformer 中 Encoder 自注意力、Decoder 掩码自注意力与交叉注意力的输入来源、因果掩码、矩阵形状与 KV
  Cache 行为差异。
---

在标准 Transformer（序列到序列 / Seq2Seq）架构中，一共存在三种不同的多头注意力结构：
1. **编码器自注意力（Encoder Self-Attention）**
2. **解码器掩码自注意力（Decoder Masked Self-Attention）**
3. **解码器交叉注意力（Cross-Attention，又称 Encoder-Decoder Attention）**

它们虽然都采用“缩放点积注意力”这一数学公式，但在输入来源、掩码控制、计算矩阵形状以及推理时的缓存行为上完全不同。

![Transformer 三种注意力模块全景对比](/images/content-rebuild/6de08d840c3f936e/cover.png)

---

## 30 秒口述答案

面试时回答该问题，可以从三个核心差异切入：

1. **Query、Key、Value 的来源不同**：
   - **Encoder 自注意力**：$Q, K, V$ 全部来自编码器上一层的输出；
   - **Decoder 掩码自注意力**：$Q, K, V$ 全部来自解码器上一层的输出（属于目标序列的前文）；
   - **Decoder 交叉注意力**：$Q$ 来自解码器当前层的自注意力子层，而 $K$ 和 $V$ 来自编码器最后一层的输出。
2. **时序掩码不同**：
   - Encoder 自注意力是**双向可见**的，任何位置都能看到全句的所有 Token；
   - Decoder 掩码自注意力加入了**因果掩码（Causal Mask）**，每个位置只能看到自己和它之前的 Token，未来位置的注意力被强制置为 0；
   - Decoder 交叉注意力**没有因果掩码**，解码器的每个位置都能看到源序列的全部 Token（仅需遮蔽源序列的填充符 `<pad>`）。
3. **注意力矩阵形状与功能不同**：
   - Encoder 自注意力的权重矩阵形状为 $L_{enc} \times L_{enc}$，负责提取源语言的全局上下文特征；
   - Decoder 掩码自注意力的权重矩阵为 $L_{dec} \times L_{dec}$ 的下三角方阵，负责在自回归生成时建模目标序列历史；
   - 交叉注意力的权重矩阵通常是长方形（$L_{dec} \times L_{enc}$），负责让目标语言在生成时对齐源语言中的对应信息。

### 贯穿算例：英译中任务

假设我们要将源英文句子 `["I", "love", "cats"]`（源序列长度 $L_{enc} = 3$）翻译为中文 `["<bos>", "我", "爱", "猫"]`（目标序列长度 $L_{dec} = 4$）：

| 特征维度 | Encoder 自注意力 | Decoder 掩码自注意力 | Decoder 交叉注意力 (Cross-Attention) |
| :--- | :--- | :--- | :--- |
| **Query ($Q$) 来源** | 编码器前一层输出 | 解码器前一层输出 | 解码器自注意力子层输出 |
| **Key/Value ($K, V$) 来源** | 编码器前一层输出 | 解码器前一层输出 | 编码器最终隐状态输出 |
| **本例矩阵形状** | $3 \times 3$ 方阵 | $4 \times 4$ 下三角方阵 | $4 \times 3$ 长方形矩阵 |
| **时序因果掩码** | 无 | 有（下三角掩码） | 无（仅需源端 Padding 掩码） |
| **在例子中的作用** | 让 "love" 同时参考 "I" 和 "cats" | 生成 "爱" 时只能参考已生成前文 `<bos>` 与 "我"，看不到未生成的 "爱" 与 "猫" | 生成 "爱" 时跨序列检索英文词 "love" |

---

## Encoder 自注意力：双向全局上下文

编码器的核心目标是理解输入序列的完整含义。因此，Encoder 自注意力必须让每一个 Token 都能够与整个句子的其他所有 Token 产生交互。

![Encoder 自注意力数据流与全局拓扑](/images/content-rebuild/6de08d840c3f936e/fig-1.png)

### 机制说明与计算过程

设编码器第 $l$ 层的输入为矩阵 $H^{(l-1)}_{enc} \in \mathbb{R}^{L_{enc} \times d_{model}}$。通过三个可训练的参数矩阵映射得到：

$$
Q = H^{(l-1)}_{enc} W_Q^{(l)}, \quad K = H^{(l-1)}_{enc} W_K^{(l)}, \quad V = H^{(l-1)}_{enc} W_V^{(l)}
$$

单头缩放点积注意力的计算公式为：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M_{pad}\right) V
$$

其中得分矩阵 $S = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{L_{enc} \times L_{enc}}$。在该子层中，除了用来屏蔽批处理填充符的 $M_{pad}$（把 `<pad>` 位置加上 $-\infty$），没有施加任何时序限制。

这意味着每个输入词不仅能看到左侧的内容，也能看到右侧的内容。例如句子 "The bank of the river" 中，"bank" 可以同时关注到右侧的 "river"，从而消除歧义。关于 Padding Mask 的具体逻辑，可参考 [/interview/llm-interview/attention-masking.md](/interview/llm-interview/attention-masking.md)。

<ReadMoreLock />

### 细节确认：注意力矩阵为什么不是对称矩阵？

有些初学者会认为，既然 $Q$ 和 $K$ 都是从同一个输入矩阵 $H$ 变换出来的，那么计算出的注意力得分矩阵 $S$ 是不是对称矩阵（即 $S_{i, j} = S_{j, i}$）？

答案是：**不对称**。

![自注意力矩阵非对称性机理解析](/images/content-rebuild/6de08d840c3f936e/fig-2.png)

原因可以从两方面理解：
1. **数学原因**：$W_Q$ 和 $W_K$ 是两组完全独立初始化的参数矩阵。位置 $i$ 对位置 $j$ 的得分为 $q_i k_j^T = h_i W_Q W_K^T h_j^T$，而位置 $j$ 对位置 $i$ 的得分为 $q_j k_i^T = h_j W_Q W_K^T h_i^T = h_i W_K W_Q^T h_j^T$。因为矩阵乘积 $W_Q W_K^T \neq W_K W_Q^T$，两者的转置不相等，所以点积结果天然不对称。
2. **语义原因**：语言中的语法依存是有方向的。例如动词 "love" 可能会强烈关注它的宾语 "cats"，但宾语 "cats" 的表征重点并不一定同等关注动词。非对称性使模型能够表达这种有向依赖关系。更详细的 $Q, K, V$ 映射分析，可阅读 [/interview/llm-interview/attention-qkv.md](/interview/llm-interview/attention-qkv.md) 与 [/interview/llm-interview/self-attention-qk.md](/interview/llm-interview/self-attention-qk.md)。

---

## Decoder 掩码自注意力：防止偷看未来的因果约束

解码器在推理阶段是自回归生成的：必须先产生第 1 个词，再基于第 1 个词预测第 2 个词，逐词向后推进。在模型训练阶段，为了利用 GPU 并行计算，我们通常使用 **Teacher Forcing**，把完整的目标序列一次性喂给解码器。这时就必须通过掩码阻止当前位置“偷看”后面的真实词。

![Decoder 掩码自注意力数据流与因果下三角结构](/images/content-rebuild/6de08d840c3f936e/fig-3.png)

### 因果掩码如何生效

设解码器第 $l$ 层的输入为 $H^{(l-1)}_{dec} \in \mathbb{R}^{L_{dec} \times d_{model}}$。它的 $Q, K, V$ 同样全部通过线性投影来自解码器自身：

$$
Q = H^{(l-1)}_{dec} W_Q^{(l)}, \quad K = H^{(l-1)}_{dec} W_K^{(l)}, \quad V = H^{(l-1)}_{dec} W_V^{(l)}
$$

在执行 Softmax 之前，需要叠加一个因果掩码矩阵 $M_{causal} \in \mathbb{R}^{L_{dec} \times L_{dec}}$：

$$
M_{causal}[i, j] = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}
$$

计算注意力权重矩阵：

$$
A = \mathrm{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M_{causal}\right)
$$

对于所有 $j > i$（代表未来时间步），得分会被加上 $-\infty$。在 Softmax 运算中，$e^{-\infty} = 0$，这就使得权重矩阵的上三角部分严格为 0。第 $i$ 步的输出只由前 $i$ 步的信息线性组合而成，保证了自回归训练与推理的一致性。

---

## Decoder 交叉注意力（Cross-Attention）：跨序列信息检索

交叉注意力是连接 Encoder 和 Decoder 的中枢。在标准的 Seq2Seq 架构中，它负责让解码器在生成每个目标词时，回过头从源句子中查找相关的词汇和语义。

![Cross-Attention 的 Q/K/V 异构拓扑与矩阵形状](/images/content-rebuild/6de08d840c3f936e/fig-4.png)

### 输入来源与矩阵维度

在解码器的每一层中，交叉注意力位于掩码自注意力子层和前馈网络（FFN）之间：
1. **Query ($Q$)**：来自**当前解码器层中掩码自注意力子层**的输出，记为 $H^{(l)}_{dec, mid} \in \mathbb{R}^{L_{dec} \times d_{model}}$；
2. **Key ($K$) 与 Value ($V$)**：来自**编码器最后一层（第 $N$ 层）**的输出，记为 $H^{(N)}_{enc} \in \mathbb{R}^{L_{enc} \times d_{model}}$。

计算投影并计算注意力得分：

$$
Q = H^{(l)}_{dec, mid} W_Q^{(l)} \in \mathbb{R}^{L_{dec} \times d_k}
$$
$$
K = H^{(N)}_{enc} W_K^{(l)} \in \mathbb{R}^{L_{enc} \times d_k}, \quad V = H^{(N)}_{enc} W_V^{(l)} \in \mathbb{R}^{L_{enc} \times d_v}
$$
$$
S_{cross} = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{L_{dec} \times L_{enc}}
$$

**关键特点**：
- **通常是非方阵**：注意力得分矩阵 $S_{cross}$ 的尺寸是 $L_{dec} \times L_{enc}$。以贯穿例子为例，中文目标句有 4 个 Token，英文源句有 3 个 Token，矩阵形状就是 $4 \times 3$。
- **无因果掩码**：解码器在翻译任何一个目标词时，都可以完整阅读英文原文，因此不需要下三角因果掩码（只需要针对英文源句填充符的 Padding Mask）。
- **输出长度等于 $L_{dec}$**：Softmax 沿着 $L_{enc}$ 维度归一化，再与形状为 $(L_{enc}, d_v)$ 的 Value 矩阵相乘，加权求和后的输出形状为 $(L_{dec}, d_v)$，保证了解码器能够继续沿着目标序列的步长传递隐状态。

---

## 面试高频追问

### 追问 1：交叉注意力中，为什么不能把 Q 设为 Encoder 输出、K/V 设为 Decoder 输出？

如果将来源颠倒，在维度和因果逻辑上都会出现冲突：

![来源颠倒导致的架构坍塌实验分析](/images/content-rebuild/6de08d840c3f936e/fig-5.png)

1. **输出序列长度被锁死**：
   - 注意力计算后输出张量的序列长度恒等于 Query 的序列长度。
   - 如果 $Q$ 来自 Encoder（长度为 $L_{enc}$），$K, V$ 来自 Decoder（长度为 $L_{dec}$），那么加权求和后的输出形状为 $(L_{enc}, d_v)$。
   - 这样输出长度就被固定为源序列长度 $L_{enc}$。然而，机器翻译或摘要任务中，目标句子的长度是动态且未知的（比如英文翻中文，词数通常不同），解码器无法自主决定何时结束并输出 `<eos>`。
2. **提问与检索的主被动逻辑倒置**：
   - Query 代表“提出检索需求的主体”，Key/Value 代表“被检索的参考知识库”。
   - 在解码生成时，是解码器当前正在生成的词需要去原文中寻找依据（例如当前要翻动词，解码器去源句中检索对应的英文动词）。
   - 如果颠倒过来，变成了静态固定的源句不断向尚未生成完毕的动态历史去查询，因果逻辑混乱。

### 追问 2：现代 Decoder-only 大模型（如 GPT、LLaMA）中还有 Cross-Attention 吗？

在主流的现代生成式大语言模型中，**已经没有独立的 Cross-Attention 模块**。更多关于架构选择的讨论可参考 [/interview/llm-interview/decoder-only.md](/interview/llm-interview/decoder-only.md) 与 [/fundamentals/transformer-arch/encoder-decoder-variants.md](/fundamentals/transformer-arch/encoder-decoder-variants.md)。

![Decoder-only 拼接序列与 Seq2Seq 交叉注意力交互对比](/images/content-rebuild/6de08d840c3f936e/fig-6.png)

**替代方案与机制**：
- **统一序列拼接**：Decoder-only 架构将 Prompt（输入）与 Completion（生成部分）拼接成同一个一维序列：`[Prompt Tokens, Target Tokens]`。
- **因果/前缀掩码统一处理**：整个模型只保留一个自注意力层。在序列内部，Prompt 内部的 Token 可以采用双向掩码互相可见（或统一使用因果掩码），Target 部分采用因果掩码只能看到前面的 Token。
- **工程优势**：省去了独立的 Encoder 和 Cross-Attention 结构，让整个模型结构规整；自回归推理时，所有上下文产生的 Key 和 Value 统一放入同一套 KV Cache 中，简化了显存调度和通信管理。关于显存加速机制可参考 [/interview/llm-interview/kv-cache.md](/interview/llm-interview/kv-cache.md)。

### 追问 3：在自回归推理（Decode）阶段，三种注意力的 KV Cache 行为有何不同？

在自回归逐词生成阶段（单步推理）：
1. **Encoder 自注意力**：在推理开始时（Prefill 阶段）对源序列一次性计算完毕，在后续逐词解码循环中不再参与计算，**不涉及动态更新**。
2. **Decoder 掩码自注意力**：维护动态递增的 **KV Cache**。每生成一个新 Token，只计算该 Token 对应的单行 $Q, K, V$。新算出的 $K, V$ 追加到当前层的缓存末尾，使缓存长度逐步累加。
3. **Decoder 交叉注意力**：其 $K, V$ 全部来自 Encoder 最后一层的输出。因为源序列在整个生成过程中是固定的，所以交叉注意力的 $K$ 和 $V$ 在 Prefill 阶段由 Encoder 输出计算完毕后即**固化为只读张量**。在后续每一个生成步中，解码器只需计算当前步的单个 $Q$，然后直接与固定的静态 Cross-Attention $K, V$ 做检索计算。

---

## 常见面试误区与避坑指南

1. **误区一：认为自注意力的得分矩阵是对称的**
   - *纠错*：投影矩阵 $W_Q$ 和 $W_K$ 是非同构的独立矩阵，$W_Q W_K^T$ 不对称，因此 $S_{i, j} \neq S_{j, i}$。
2. **误区二：误认为 Cross-Attention 的注意力矩阵是方阵**
   - *纠错*：矩阵尺寸是 $(L_{dec}, L_{enc})$，仅当源文本与目标文本长度刚好相同时才是方阵，一般情况下是长方形。
3. **误区三：误认为 Cross-Attention 不需要任何 Mask**
   - *纠错*：它不需要因果 Mask，但依然需要源序列的 **Key Padding Mask**，否则解码器会把注意力权重分配给源端用来对齐 Batch 的 `<pad>` 标记。
4. **误区四：混淆 Cross-Attention 的 Q 来源**
   - *纠错*：Cross-Attention 的 $Q$ 来自 Decoder 内部刚算完的掩码自注意力层输出，而不是直接来自目标端的原始 Embedding，更不是来自 Encoder。

---

## 读者自测检查点

完成阅读后，可尝试在不翻看正文的情况下回答：

1. **在 Seq2Seq 模型的 Cross-Attention 中，注意力矩阵的形状是 $(L_{dec}, L_{dec})$ 还是 $(L_{dec}, L_{enc})$？输出张量的序列长度由谁决定？**
2. **在自回归推理（逐词生成）过程中，Decoder 自注意力和 Cross-Attention 的 KV Cache 管理方式有什么本质区别？**

相关基础延伸阅读：
- [/interview/llm-interview/attention-masking.md](/interview/llm-interview/attention-masking.md)（注意力掩码的实现细节与边界）
- [/interview/llm-interview/why-multi-head-attention.md](/interview/llm-interview/why-multi-head-attention.md)（多头注意力机制解析）
- [/interview/llm-interview/transformer-seq2seq.md](/interview/llm-interview/transformer-seq2seq.md)（Seq2Seq 完整映射流程）
