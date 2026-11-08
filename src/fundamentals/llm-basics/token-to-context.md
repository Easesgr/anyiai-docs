---
title: 大模型如何表示文本：从 Token、Embedding 到上下文向量
icon: mdi:text-box-outline
tag:
- 大模型基础
- Token
- 上下文表示
description: 解析大语言模型将自然语言转换为高维向量的计算过程，详解分词、嵌入查找表、位置编码与自注意力机制的协同原理与工程边界。
---

在上一讲[/fundamentals/llm-basics/what-is-llm.md](/fundamentals/llm-basics/what-is-llm.md)中，我们了解到大语言模型依靠计算条件概率来逐字生成文本。然而，计算机芯片与神经网络只能进行浮点数矩阵运算，无法直接“阅读”字符串。

一段自然语言从输入模型到最终被用于预测，经历了一条明确的转换通路：

1. **分词与编号（Tokenization）**：分词器将连续的文本切分成离散的子词单元（Token），并映射为词表中的整数 ID。
2. **静态查表与位置注入（Embedding & Position）**：模型根据 ID 从嵌入矩阵中提取出对应的初始向量（Token Embedding），并补充序列位置信息。
3. **动态语境融合（Self-Attention）**：静态向量进入 Transformer 堆叠层，自注意力机制在不同词元之间分配权重，将没有上下文的静态向量重塑为融入整句语境的上下文向量（Contextual Representation）。

以“苹果”一词为例：它在分词后得到一个固定 ID；在静态词向量表中对应一组固定的浮点数；但当它出现在“苹果发布了新款手机”时，自注意力机制会把它与“手机”、“发布”动态关联，最终输出带有“科技公司”属性的上下文向量。

![从离散文本到上下文表示的全局数据流水线](/images/content-rebuild/6947865c1f61cd21/cover.png)

上图展示了这一完整的数据流动通路：输入字符串首先被切分为离散的 Token ID，再转化为静态向量，最终通过注意力计算演变为上下文向量。理解这条通路中的每一步机制、约束与工程代价，是掌握大模型推理与上下文长度限制的物理前提。

---

## 文本离散化：分词器切分原理与词表映射

大模型处理文本的第一步，是把连续的字符序列切分为离散的符号单元（Token）。

### 为什么需要子词分词（Subword Tokenization）？

在自然语言处理早期，文本切分通常走两个极端：
* **词级切分（Word-level）**：依靠空格或词典切分单词。这种方案在英语等屈折语中会导致词表急剧膨胀（如 walk、walking、walked 都要单独占位）；一旦遇到未登录词（Out-Of-Vocabulary, OOV），只能替换为 `<unk>` 标记，丢失原始语义。
* **字符级切分（Character-level）**：直接切为单个字母或单个汉字。虽然词表很小（基础 ASCII 仅 128 个），完全没有 OOV 问题，但代价是输入序列长度成倍增加。单个字符本身携带的信息量极低，而自注意力机制的计算开销与序列长度呈二次方增长，过长的序列会浪费大量算力。

现代主流大模型普遍采用折中的**子词切分算法（Subword Tokenization）**，如字节对编码（Byte Pair Encoding, BPE）和 WordPiece。它们让常见的高频词保持完整独立，让低频词或形态变体拆解为更小的有意义片段（例如将 `unhappiness` 拆分为 `un`、`happi`、`ness`）。

![子词切分算法与词表映射过程](/images/content-rebuild/6947865c1f61cd21/fig-1.png)

上图解析了 BPE 算法从字符级逐步合并为子词的过程。以 Hugging Face Tokenizers 中的标准 BPE 为例，其构建词表的训练过程完全基于频次统计：

1. **准备基础词表**：统计语料中所有词频，将单词拆解为单个字符序列，并在词尾添加边界标记。
2. **统计相邻对频次**：遍历语料，统计所有连续出现的相邻符号对（Pair）的共现次数。
3. **合并最高频对**：找出出现频次最高的一组相邻对，合并为一个新的子词单元，加入词表。
4. **迭代更新**：在语料中将该相邻对替换为新子词，重复统计与合并，直到词表大小达到预设目标 $V$。

```python
# BPE 统计合并的核心逻辑演示（简化教学代码）
from collections import Counter

corpus = {
    ("l", "o", "w"): 5,
    ("l", "o", "w", "e", "r"): 2,
    ("n", "e", "w", "e", "s", "t"): 6,
    ("w", "i", "d", "e", "s", "t"): 3,
}

def get_stats(corpus):
    pairs = Counter()
    for token_seq, freq in corpus.items():
        for i in range(len(token_seq) - 1):
            pairs[token_seq[i], token_seq[i+1]] += freq
    return pairs

# 统计当前最高频相邻对
pairs = get_stats(corpus)
best_pair = max(pairs, key=pairs.get)
print(f"最高频合并对: {best_pair}, 出现频次: {pairs[best_pair]}")
# 输出: 最高频合并对: ('e', 's'), 出现频次: 9
```

词表构建完成后，分词器就获得了一张固定大小的**词表（Vocabulary）**与有序的**合并规则表（Merge Rules）**。实际推理时，分词器根据合并规则匹配文本，并输出对应的整数索引（Token ID）。

为了彻底解决未登录词问题，GPT-2、LLaMA、Qwen 等模型引入了 **Byte-level BPE**：算法的基础单元不是 Unicode 字符，而是计算机底层的 256 个基础字节（Bytes）。任何字符串在 UTF-8 编码下都可以拆分为字节，因此 Byte-level BPE 能够无损编码任何语言与符号组合。

---

## Token 的工程边界与中文膨胀

在工程实践中，必须分清 Token 的定义：**Token 是特定分词器离散切分出来的单元，它既不等于自然语言中的一个词，也不等于汉字里的一个单字。**

1. **前导空格敏感**：多数分词器（如 GPT-4 的 `cl100k_base` 或 LLaMA 分词器）会将单词前方的空格与单词本身合并为一个 Token。例如 `"apple"` 与 `" apple"` 在词表中属于两个完全不同的 ID。拼接提示词时多出或遗漏空格，会直接改变模型接收到的初始向量。
2. **大小写差异**：`"The"`、`"the"` 和 `"THE"` 会被编码为不同的 Token ID。大写组合如果出现频率较低，往往会被碎化切分成多个短字符。
3. **特殊控制符**：对话系统中的控制标记（如 `<|im_start|>`、`<s>`、`</s>`）在词表中作为独立的特殊 Token 存在，拥有独立的固定 ID。

<ReadMoreLock />

![中英文在不同词表下的切分与膨胀对比](/images/content-rebuild/6947865c1f61cd21/fig-2.png)

上图对比了同一段语义在中英文环境下的切分粒度差异。这里引出了应用开发中常见的**中文 Token 膨胀现象**：

在标准 UTF-8 编码中，一个常见汉字占用 3 个字节。早期的开源模型（如 LLaMA 1 代）训练语料绝大部分是英文，词表规模较小（约 32,000），且未收录常见的中文词组或单字。
当这类分词器处理中文时：
1. 汉字无法命中词表中的完整条目；
2. 分词器退化为 Byte-level 模式，把汉字的 3 个字节拆开切分为 3 个独立的 Token；
3. 结果导致**输入 1 个汉字，在模型内部占用 3 个 Token**。

这种膨胀对系统性能有直接影响：
* **输入长度成倍增加**：表达相同的业务信息，中文所需的 Token 数量是英文的 2 到 3 倍。
* **计算与显存开销增长**：注意力机制的计算开销与序列长度 $L$ 存在二次方关系。Token 数量翻倍，意味着预填充（Prefill）阶段的注意力计算矩阵翻 4 倍。
* **有效上下文空间被压缩**：模型固定的物理上下文窗口（如 4K 或 8K）能容纳的中文实际文本量大幅缩减。关于上下文承载力的详细讨论见[/fundamentals/llm-basics/context-window.md](/fundamentals/llm-basics/context-window.md)与[/architecture/token-cost.md](/architecture/token-cost.md)。

现代针对多语言或中文优化的模型（例如 Qwen 系列），通过将词表扩充到 150,000 以上，收录了大量单字、常见词语及成语，使得 1 个汉字的平均 Token 数降至 0.6 到 1.0 左右，大幅缓解了膨胀问题。

---

## 静态 Embedding 查找表与位置信息注入

分词完成后，输入的文本变成了一组离散的整数数组。长度为 $T$ 的文本序列可以表示为：

$$\mathbf{x}_{\text{id}} = [t_1, t_2, \dots, t_T], \quad t_i \in \{0, 1, \dots, V-1\}$$

其中 $V$ 为词表容量（Vocabulary Size）。

### 查表操作的本质：从 One-Hot 到稠密向量

如果直接将每个整数用 One-Hot 独热编码向量 $\mathbf{e}_i \in \{0, 1\}^V$ 表示，向量维度将高达数万甚至数十万。更重要的是，任意两个不同类别 One-Hot 向量在几何上都是完全正交的：

* 任意两个不同 Token 之间的欧氏距离恒为 $\sqrt{2}$：$\|\mathbf{e}_a - \mathbf{e}_b\|_2 = \sqrt{1^2 + (-1)^2} = \sqrt{2}$
* 任意两个不同 Token 之间的余弦相似度恒为 0：$\cos(\mathbf{e}_a, \mathbf{e}_b) = 0$

在 One-Hot 空间中，“猫”和“狗”的距离，与“猫”和“推土机”的距离完全一样，模型无法从中感知任何语义关联。

![静态 Embedding 查找与位置编码注入机制](/images/content-rebuild/6947865c1f61cd21/fig-3.png)

上图展示了模型如何通过查找表将离散 ID 映射为连续高维向量。模型内部维护着一个可学习的静态词嵌入矩阵：

$$\mathbf{W}_e \in \mathbb{R}^{V \times d_{\text{model}}}$$

其中 $d_{\text{model}}$ 是隐藏层维度（如 4096）。当输入某个 Token ID $t_i$ 时，查表操作（Embedding Lookup）在数学上等价于用 One-Hot 向量左乘词嵌入矩阵：

$$\mathbf{x}_i = \mathbf{e}_{t_i} \mathbf{W}_e = \mathbf{W}_e[t_i, :]$$

在代码实现中，计算框架直接按行索引取出矩阵的第 $t_i$ 行切片。这行长度为 $d_{\text{model}}$ 的连续浮点数向量，就是该 Token 的**静态词向量（Static Token Embedding）**。

### 为什么必须注入位置编码（Positional Encoding）？

拿到静态向量序列后，还不能直接进行注意力计算。因为基础的自注意力计算公式是**置换不变（Permutation Invariant）**的：注意力机制通过点积计算两两元素之间的相似度，加权求和的操作对元素次序不敏感。

如果不添加位置信息，句子“张三 借了 李四 100元”与“李四 借了 张三 100元”，在输入端的表示集合完全相同，模型无法区分谁是借款人、谁是债权人。

为了打破这种对称性，必须为序列中的每个位置注入位置编码（Positional Encoding, PE）：
* **可学习 / 正弦绝对位置编码（Absolute PE）**：在静态词向量上直接按元素逐项相加一个位置向量 $\mathbf{p}_i$：
  $$\mathbf{h}_i^{(0)} = \mathbf{x}_i + \mathbf{p}_i$$
* **旋转位置编码（Rotary Position Embedding, RoPE）**：现代主流大模型（如 LLaMA、Qwen）更广泛的选择。它不在最外层直接与输入相加，而是在每一层计算自注意力时，通过正交旋转矩阵对 Query 和 Key 向量施加与绝对位置成正比的旋转变换，使得两个向量的点积结果直接携带它们的**相对距离** $(m - n)$。

---

## Self-Attention 计算机制：Query、Key、Value 如何构建动态表示

经过 Embedding 查表和位置编码后，我们得到了初始隐藏层表示序列 $\mathbf{H}^{(0)} \in \mathbb{R}^{T \times d}$。但这时的向量仍然是**孤立的**。

词表中代表“苹果”的那个 Token ID，无论是在句子“今天买了两斤苹果”中，还是在“苹果发布了最新款手机”中，通过查表拿到的初始向量 $\mathbf{x}_i$ 是完全一样的。静态向量本身无法直接表达多义词在具体语境下的含义。

自注意力机制（Self-Attention）的作用，就是让序列中的每个 Token 能够根据整句话的上下文，自适应地从其他 Token 处提取相关信息，进而把静态向量动态更新为包含深层语境的上下文表示。

![Self-Attention 中 Q、K、V 的计算机制流向](/images/content-rebuild/6947865c1f61cd21/fig-4.png)

上图展现了注意力计算的核心数据通路。对于当前层的输入向量 $\mathbf{h}_i$，模型通过三个不同的线性投影矩阵，分别将其映射为三个角色向量：

$$\mathbf{q}_i = \mathbf{h}_i \mathbf{W}_Q, \quad \mathbf{k}_i = \mathbf{h}_i \mathbf{W}_K, \quad \mathbf{v}_i = \mathbf{h}_i \mathbf{W}_V$$

这三个向量分工明确：
* **Query（查询向量，$\mathbf{q}_i$）**：代表当前 Token **需要寻找什么样的语境**。例如，“苹果”需要寻找周围是否存在“吃/水果”或“手机/发布”等关联信息。
* **Key（键向量，$\mathbf{k}_j$）**：代表其他 Token **对外暴露的特征标识**，供其他词元检索匹配。
* **Value（值向量，$\mathbf{v}_j$）**：代表其他 Token **实际承载的具体语义内容**。一旦匹配度高，这些内容就会被加权汇聚过来。

注意力机制的标准计算矩阵公式如下：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

### 最小数值算例验证

为了直观展示计算过程，我们设定一个简化的教学示例：序列包含 3 个 Token，投影维度 $d_k = 2$。
假设当前某个注意力头计算得到的 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$ 矩阵如下（数值仅供教学说明）：

$$\mathbf{Q} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix}, \quad \mathbf{K} = \begin{bmatrix} 1.0 & 0.0 \\ 1.0 & 1.0 \\ 0.0 & 2.0 \end{bmatrix}, \quad \mathbf{V} = \begin{bmatrix} 2.0 & 1.0 \\ 0.0 & 3.0 \\ 1.0 & 0.0 \end{bmatrix}$$

**第一步：计算点积相似度矩阵 $\mathbf{Q}\mathbf{K}^T$**

$$\mathbf{S} = \mathbf{Q}\mathbf{K}^T = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1.0 & 1.0 & 0.0 \\ 0.0 & 1.0 & 2.0 \end{bmatrix} = \begin{bmatrix} 1.0 & 1.0 & 0.0 \\ 0.0 & 2.0 & 4.0 \\ 1.0 & 2.0 & 2.0 \end{bmatrix}$$

**第二步：缩放与归一化（除以 $\sqrt{d_k} = \sqrt{2} \approx 1.414$）**

除以 $\sqrt{d_k}$ 的原因在于：当向量维度很高时，点积结果的方差会线性放大，导致 Softmax 函数进入梯度极小的平坦区，引起训练梯度消失。
缩放后矩阵约为：

$$\frac{\mathbf{S}}{\sqrt{2}} = \begin{bmatrix} 0.707 & 0.707 & 0.000 \\ 0.000 & 1.414 & 2.828 \\ 0.707 & 1.414 & 1.414 \end{bmatrix}$$

**第三步：按行执行 Softmax 得到注意力权重矩阵 $\mathbf{A}$**
以第 2 个 Token（第二行）为例：
* 未归一化指数值：$e^0 = 1.0,\ e^{1.414} \approx 4.113,\ e^{2.828} \approx 16.919$
* 分母求和：$1.0 + 4.113 + 16.919 = 22.032$
* 第二行归一化权重：$[0.045, 0.187, 0.768]$

**第四步：加权聚合 $\mathbf{V}$ 向量**
该 Token 经过注意力聚合后的新输出向量为：

$$\mathbf{z}_2 = 0.045 \times \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix} + 0.187 \times \begin{bmatrix} 0.0 \\ 3.0 \end{bmatrix} + 0.768 \times \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.090 + 0 + 0.768 \\ 0.045 + 0.561 + 0 \end{bmatrix} = \begin{bmatrix} 0.858 \\ 0.606 \end{bmatrix}$$

计算结果表明，第 2 个 Token 吸收了来自第 3 个 Token 约 76.8% 的语义特征。在自回归生成时，还需要施加因果掩码（Causal Mask），将未来时刻的注意力分数置为 $-\infty$，确保当前 Token 只能从前文吸收信息。

![多义词在自注意力交互中的消歧演变](/images/content-rebuild/6947865c1f61cd21/fig-5.png)

上图展示了这种计算对于多义词消歧的实际效果：
当“苹果”在句子“苹果发布了最新款手机”中出现时，“苹果”的 Query 与后方“手机”、“发布”的 Key 计算出极高的点积相关度。在加权聚合 Value 时，“苹果”吸收了大量数码产品的特征，其向量在向量空间中更靠近科技相关词；反之，若处于“买了两斤红苹果”中，它则大幅吸收“吃”、“斤”的特征，向量在空间中更靠近水果类词汇。

---

## 静态词向量与深层上下文向量的对比

在学习大模型时，初学者容易混淆词嵌入（Token Embedding）与上下文表示（Contextual Representation / Hidden States）。两者的数学性质与用途有明确区别：

| 维度 | 静态词向量（Static Embedding） | 深层上下文向量（Hidden States） |
| :--- | :--- | :--- |
| **产生位置** | 模型最底层的查找表（Lookup Table） | Transformer 堆叠层（尤其是中高层）内部 |
| **语境依赖** | **无语境**。给定一个 Token ID，输出向量固定不变 | **强语境**。同一个 Token 在不同句子里会得到完全不同的向量 |
| **信息内容** | 仅包含该词元在预训练中学习到的统计先验 | 融合了上下文逻辑、指代关系与消歧后的实际语义 |
| **多义词处理** | 无法区分。不同语义被混杂在同一组参数中 | 自然区分。通过自注意力机制根据前后词动态投影更新 |
| **典型用途** | 输入投影、简单词表检索 | 下一个 Token 预测、句向量表征（需 Pooling）、语义判别 |

早期自然语言处理技术（如 Word2Vec、GloVe）本质上只停留在静态词向量阶段。尽管可以建立类似 $\vec{v}_{\text{King}} - \vec{v}_{\text{Man}} + \vec{v}_{\text{Woman}} \approx \vec{v}_{\text{Queen}}$ 的向量运算关系，但由于缺乏自注意力机制对局部序列的重构能力，静态向量在面对复杂的长文本与多样化句法时表现受限。

![静态词向量与深层上下文表示的退化与演进对比](/images/content-rebuild/6947865c1f61cd21/fig-6.png)

上图对比了静态嵌入在面对歧义和复杂长语境时的退化表现，以及多层 Transformer 如何逐步建立上下文深度。

需要指出的是，深层上下文向量也有其计算边界。随着输入序列长度增加，注意力分布往往会被大量背景噪音稀释，或者由于模型的位置偏差而更倾向于关注首尾信息，导致中间关键语义被平滑甚至淹没。这正是后续[/frontier/context-engineering/what-is-context-eng.md](/frontier/context-engineering/what-is-context-eng.md)与[/fundamentals/llm-basics/pretraining.md](/fundamentals/llm-basics/pretraining.md)中需要深入讨论的上下文工程与表示衰减问题。

---

## 机制总结

在采用绝对位置编码的经典结构中，从文本到上下文表示的输入通路可表示为：

$$\text{原始文本} \xrightarrow{\text{Tokenizer}} \{t_i\} \xrightarrow{\text{Embedding}} \mathbf{x}_i \xrightarrow{+\text{PE}} \mathbf{h}_i^{(0)} \xrightarrow{\text{Transformer Layers}} \mathbf{h}_i^{(L)}$$

若采用 RoPE 等现代旋转位置编码，初始隐藏层输入 $\mathbf{h}_i^{(0)}$ 直接对应词嵌入 $\mathbf{x}_i$，位置序关系则在各 Transformer 层的自注意力阶段动态旋转注入给 Query 与 Key。

* **分词器（Tokenizer）**把连续的自然语言离散化为紧凑的有界整数 ID；
* **词嵌入与位置注入（Embedding & Position）**：词嵌入表将整数 ID 映射为稠密浮点向量，并通过输入端相加（绝对位置编码）或在注意力层旋转投影（RoPE）引入序列位置关系；
* **自注意力（Self-Attention）**利用 $Q, K, V$ 矩阵计算，把静态独立的词向量重塑为融入整段语境的动态上下文向量。

当模型在最后一层获得了最终的上下文向量 $\mathbf{h}_T^{(L)}$ 后，便可以通过输出线性层将其投影回词表大小的概率分布上，从而预测下一个 Token。但在实际工程中，这个输入序列到底能拉多长？序列变长后计算显存会面临怎样的物理限制？这些问题将在下一讲[/fundamentals/llm-basics/context-window.md](/fundamentals/llm-basics/context-window.md)中展开详细分析。
