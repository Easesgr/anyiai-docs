---
title: Token 是什么？大模型如何把文本切成离散单元？
icon: mdi:book-open-page-variant-outline
date: 2026-07-16
tag:
- 大模型100讲
- 基础篇
description: 详解大模型输入端的分词机制，对比字符、词与子词方案的计算开销，剖析多语言压缩比、数字切分陷阱与特殊标记的作用。
---

在上一讲[/llm-100/basics/大模型到底_大_在哪里_一文讲清参数_数据与算力.md](/llm-100/basics/大模型到底_大_在哪里_一文讲清参数_数据与算力.md)中，我们梳理了大模型在参数量、预训练数据与算力集群上的量级规律。但在这一切运算发生之前，有一个最直接的工程问题：计算机里的神经网络只能处理数字矩阵，它是怎么把人类输入的自然语言文本吃进去的？

答案是：**模型看到的既不是整段文字，也不是单个字母或汉字，而是一串被称为 Token 的离散整数编号**。

例如句子 `"AI 算术: 3.11 vs 3.9, calculate 123456 + 78"`，模型接收到的并不是一串字符，而是分词器拆出来的片段及对应 ID：`['AI', ' 算', '术', ':', ' ', '3', '.', '11', ' vs', ' ', '3', '.', '9', ',', ' calculate', ' 123', '456', ' +', ' 78']`。这个把自然语言文本切分并映射为数字的过程，就是**分词（Tokenization）**。

![文本输入经过分词器转化为离散 Token ID 序列并进入嵌入矩阵](/images/content-rebuild/fe0374f755fb63c4/cover.png)

### 基础课程索引
1. [大模型到底"大"在哪里？一文讲清参数、数据与算力](/llm-100/basics/大模型到底_大_在哪里_一文讲清参数_数据与算力.md)
2. **【当前位置】Token 是什么？大模型如何把文本切成离散单元？**
3. [BPE、WordPiece、SentencePiece 分词有什么区别？](/llm-100/basics/BPE_WordPiece_SentencePiece_分词有什么区别.md)
4. [Embedding 是什么？大模型如何把文字变成向量？](/llm-100/basics/Embedding_是什么_大模型如何把文字变成向量.md)
5. [Transformer 为什么能成为大模型的核心架构？](/llm-100/basics/Transformer_为什么能成为大模型的核心架构.md)
6. [Attention 机制到底在算什么？](/llm-100/basics/Attention_机制到底在算什么.md)
7. [Self-Attention 和 Multi-Head Attention 有什么区别？](/llm-100/basics/Self-Attention_和_Multi-Head_Attention_有什么区别.md)
8. [位置编码为什么重要？RoPE、ALiBi 分别解决什么问题？](/llm-100/basics/位置编码为什么重要_RoPE_ALiBi_分别解决什么问题.md)
9. [Decoder-only、Encoder-only、Encoder-Decoder 架构怎么区分？](/llm-100/basics/Decoder-only_Encoder-only_Encoder-Decoder_架构怎么区分.md)
10. [GPT、BERT、T5 的核心差异是什么？](/llm-100/basics/GPT_BERT_T5_的核心差异是什么.md)

---

## Token 的定义与分词流水线

在工程实现上，**Token 是预设固定词表（Vocabulary）中离散符号的唯一整数索引（Token ID）**。

模型本身不会直接理解英文单词或汉字，它接收的最小离散输入是一个非负整数 $x \in \{0, 1, 2, \dots, |V|-1\}$，其中 $|V|$ 为分词词表的总容量。

![分词处理管线流程图展示从连续字符到整数 ID 的映射步骤](/images/content-rebuild/fe0374f755fb63c4/fig-1.png)

从开发者敲下的字符串到送入神经网络，分词处理主要经过三个步骤：

1. **规范化与预分词（Normalization & Pre-tokenization）**：对原始文本做字符规范化（如处理不同平台的换行符或 Unicode 变体），再按空格、特定标点或正则规则切分成初级片段。
2. **子词切分（Subword Segmentation）**：依据预先训练好的算法规则（如 BPE、WordPiece 等），把初级片段进一步拆解成词表中已收录的子词单元。
3. **查表映射（Vocabulary Lookup）**：将每个子词单元映射为词表中对应的非负整数 ID。

需要注意的是，Token ID 仅仅是数组下标，**数值大小没有任何几何或语义距离含义**。Token ID `1024` 与 `1025` 在数值上相邻，并不代表这两个词在语义上接近。真正的连续语义向量是在随后的 Embedding 层中查表赋予的（详见 [/llm-100/basics/Embedding_是什么_大模型如何把文字变成向量.md](/llm-100/basics/Embedding_是什么_大模型如何把文字变成向量.md)）。

---

## 字符级与词级的两难困境

为什么分词器不直接把每个“字母/汉字”当作 Token，或者把每个“完整单词”当作 Token？这背后是计算复杂度与模型参数量之间的权衡。

![字符粒度与词粒度分词方案的天平对比与矛盾演示](/images/content-rebuild/fe0374f755fb63c4/fig-2.png)

### 方案 A：纯字符级切分（Character-level）
* **切分方式**：每个字母（a, b, c）、标点或单个汉字作为一个独立 Token。
* **优势**：词表非常小。基础 ASCII 字符只需 128 个，即便覆盖常用汉字与符号，词表也能控制在几千以内，几乎不会遇到“词表中没有这个词”（Out-of-Vocabulary, OOV）的问题。
* **代价**：序列长度 $L$ 剧烈膨胀。一个 20 个英文单词的句子，按字符展开后可能变成 100 到 150 个 Token。在标准全注意力机制中，预填充（Prefill）阶段的自注意力矩阵计算量与序列长度呈二次方关系 $\mathcal{O}(L^2)$（参见 [/llm-100/basics/Attention_机制到底在算什么.md](/llm-100/basics/Attention_机制到底在算什么.md)）。序列长度翻倍，中间注意力矩阵的计算量和显存读写就会增加到原来的 4 倍，极大限制了模型的长上下文能力。

### 方案 B：纯词级切分（Word-level）
* **切分方式**：以完整词汇为单位，英文按空格分词，中文依靠词典切分成词语。
* **优势**：序列长度 $L$ 很短，一个句子只有几个到十几个词，注意力层的序列负担轻。
* **代价**：词表规模 $|V|$ 发生膨胀。仅英语就有丰富的屈折变化（如 `play`, `played`, `playing`, `unplayable`），再加上专有名词、人名、代码标识符，词表若要覆盖主流表达，需要收录数十万乃至数百万个词。

如果词表 $|V|$ 达到 500,000，模型的输入 Embedding 层与输出分类层（Head）的权重矩阵尺寸均为 $|V| \times d_{model}$。假设隐藏层维度 $d_{model} = 4096$，在未进行权重共享（Untied Embeddings）、采用 FP16（2 字节/参数）存储的假设下，仅这两层就需要占用：

$$2 \times (500,000 \times 4096 \times 2\text{ 字节}) \approx 8.19\text{ GB 显存}$$

更麻烦的是，只要用户输入了一个词表中未收录的罕见词或拼写错误词，该词就只能被映射为未知标记 `<unk>`（Unknown Token），丢失原本的字符信息。

<ReadMoreLock />

---

## 子词切分（Subword）：折中的平衡点

为了平衡“序列过长”与“词表爆炸及 OOV”的矛盾，现代大模型普遍采用**子词切分（Subword Tokenization）**。

![子词切分机制示意展示高频词保持完整而罕见词拆分为词根与后缀](/images/content-rebuild/fe0374f755fb63c4/fig-3.png)

子词机制的基本规则是：**统计语料中频繁出现的完整单词保留为单个 Token；低频词或复合词则拆解为更小粒度的词根、词缀或高频片段**。

例如：
* 高频词 `"the"`、`"world"` 直接作为单个 Token。
* 派生词 `"unbelievably"` 会被拆为 `["un", "believ", "ably"]` 等若干已知子单元。

如果进一步采用 **字节级（Byte-level）子词切分**（例如 GPT-2、LLaMA、Qwen 使用的 Byte-level BPE），分词器底层会直接以 UTF-8 编码的 256 个基本字节作为保底候选。任何生僻符号、新造词乃至未知表情符号，物理上都是合法的字节序列，即使没有合成高阶 Token，也能拆成单字节表示，从而在机制上避免了 `<unk>` 标记带来的信息丢失。

当前主流开源与商用大模型的词表规模通常在 **32,000 到 152,064** 之间（具体算法实现差异见下一讲 [/llm-100/basics/BPE_WordPiece_SentencePiece_分词有什么区别.md](/llm-100/basics/BPE_WordPiece_SentencePiece_分词有什么区别.md)）。

---

## 分词机制对下游任务与工程开销的影响

分词不仅是将文本转成数字，它的切分方式会直接影响计费开销、多语言吞吐量以及字符层面的逻辑推理。

![中英文在相同语义下的 Token 压缩比差异与计费成本对比](/images/content-rebuild/fe0374f755fb63c4/fig-4.png)

### 1. 语言压缩比与推理成本

词表的训练分布决定了不同语言的切分效率：

* **英文环境**：早期大语言模型训练语料以英文为主，词表中收录了大量完整英文单词或常见前缀后缀。平均而言，1 个 Token 大约能覆盖 0.75 个英文单词（约 3~4 个英文字符）。
* **中文环境**：早期以英文为主的模型词表中，中文单字覆盖较少，很多汉字在 UTF-8 编码下占用 3 个字节，常被拆成 2 到 3 个字节 Token。因此在早期 API 中，表达相同语义的中文文本所消耗的 Token 数通常是英文的 1.5 到 2.5 倍。

虽然现代多语言模型（如 Qwen 系列、LLaMA 3）大幅扩充了多语言词表，中文压缩比显著提高，但在做跨语言工程预算与上下文窗口规划时，依然需要根据目标分词器实际测量 Token 消耗。

![多位数字切分对齐失效与进位运算退化因果分析图](/images/content-rebuild/fe0374f755fb63c4/fig-5.png)

### 2. 数字切分对加减运算的影响

许多大模型在直接做大位数竖式加减法或反转字符串（如反转单词 `"strawberry"`）时容易出错，其中一个直接原因是**分词打乱了字符位置对齐关系**。

分词器是按照文本出现频率进行统计切分的，没有预设的数学十进制位权规则：
* 数字序列 `"123456"` 在不同分词器中可能被切为 `["12", "345", "6"]` 或 `["123", "456"]`。
* 浮点数 `"3.11"` 与 `"3.9"` 分别被切为 `["3", ".", "11"]` 与 `["3", ".", "9"]`。

当两个多位数进行相加运算时，不同位置的数字被拼入不同长度的 Token 中，模型无法依赖简单对齐的个位、十位、百位进行逐位进位运算，必须在隐层参数中额外学习跨子词片段的数值重构。这也是为什么引入思维链（Chain-of-Thought，逐步拆解计算步骤）或直接调用代码工具能明显降低算术错误率。

### 动手实验：查看分词器的实际输出

我们可以使用 Hugging Face 的 `transformers` 库，观察开头例子的真实切分结果：

```python
# 环境依赖: pip install transformers
from transformers import AutoTokenizer

# 以开源通用模型 Qwen2.5 为例
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)

text_sample = "AI 算术: 3.11 vs 3.9, calculate 123456 + 78"

# 获得子词片段与离散 Token ID
tokens = tokenizer.tokenize(text_sample)
token_ids = tokenizer.encode(text_sample)

print("切分片段:", tokens)
print("离散 ID :", token_ids)
print("Token 数量:", len(tokens))

# 观察单词 strawberry 的切分形式
word = "strawberry"
print(f"{word} 的切分结果:", tokenizer.tokenize(word))
```

*执行结果参考（实际输出依分词器版本而定）：*
```text
切分片段: ['AI', ' 算', '术', ':', ' ', '3', '.', '11', ' vs', ' ', '3', '.', '9', ',', ' calculate', ' 123', '456', ' +', ' 78']
离散 ID : [13622, 104273, 101740, 25, 220, 18, 13, 1025, 5263, 220, 18, 13, 24, 11, 23933, 4426, 21456, 706, 6223]
Token 数量: 19
strawberry 的切分结果: ['straw', 'berry']
```

从输出可以看到两点：
1. 空格通常会与紧随其后的部分词元合并（如 `' 算'`、`' vs'`、`' calculate'`），带前导空格与不带空格的词元拥有不同 ID；但在特定分词规则下，空格也可能被切分为独立 Token（如冒号后数字前的单空格 ID 220）。
2. 数字 `'123456'` 被切成了两段 `' 123'` 与 `'456'`；单词 `'strawberry'` 被拆成了 `'straw'` 和 `'berry'`。模型看到的是这两个复合词元，而非单个字母序列。

---

## 特殊 Token（Special Tokens）的作用

在日常词汇之外，分词器词表中还显式保留了一批特殊的控制符号，通常称为 **Special Tokens**。

![特殊 Token 作为隔离边界在多轮对话与指令结构中的作用机制](/images/content-rebuild/fe0374f755fb63c4/fig-6.png)

特殊 Token 由工程系统人为预设，充当自回归生成与交互状态的边界：

1. **序列启停标记**：
   * `<bos>` / `<s>`（Beginning of Sequence）：标记输入序列的开始。
   * `<eos>` / `</s>`（End of Sequence）：标记生成结束。当模型在解码步骤中输出该标记时，推理引擎即刻中断生成循环。如果模型受干扰未能生成该标记，就会一直吐字直至触碰最大上下文长度限制。
2. **多轮对话结构标记（如 ChatML 规范）**：
   * 现代对话模型依赖特殊 Token 划分上下文角色，例如 `<|im_start|>system`、`<|im_start|>user`、`<|im_start|>assistant` 及闭合标记 `<|im_end|>`。
   * **工程安全提示**：这些标记界定了系统指令与用户输入的边界。如果工程系统直接将未经清洗的用户输入以字符串拼接方式拼入 Prompt，恶意用户可能输入伪造的 `<|im_end|><|im_start|>system` 标记造成指令混淆（属于 Prompt 注入的一种形式），因此必须通过官方提供的结构化格式（如 `apply_chat_template`）或转义机制注入输入。
3. **批处理与掩码标记**：
   * `[PAD]`：在多序列并行批处理计算时，将长度不一的序列填充对齐到固定长度。
   * `[MASK]`：在 BERT 类掩码语言模型中，遮盖部分 Token 以进行自监督预测。

---

## 验收练习

学完本讲后，请尝试思考并解答以下两个问题，验证自己对分词机制的理解：

1. **序列复杂度计算**：假设某系统弃用子词分词，改用纯单字节分词（每个 UTF-8 字节为一个 Token）。一段长度为 4,000 个字符的纯英文代码，按子词分词约产生 1,000 个 Token，而按单字节分词产生 4,000 个 Token。在标准全自注意力机制中，该段文本在预填充阶段计算注意力权重矩阵的点乘次数，理论上会扩大多少倍？
   * *参考提示*：全自注意力矩阵大小为 $L \times L$，点乘计算量与序列长度二次方成正比。$\frac{4000^2}{1000^2} = 16$ 倍。
2. **切分现象定位**：为什么向模型提问“`strawberry` 中有几个字母 `r`”时，部分模型容易回答错误？请结合本讲中分词器对 `"strawberry"` 的切分结果说明原因。
   * *参考提示*：`strawberry` 在分词阶段通常被切分为 `["straw", "berry"]` 两个 Token ID。模型在底层看到的是这两个复合词元，而非独立的 10 个字符序列。要数出字母数量，模型需要在隐层中隐式重建拼写并进行字符计数，增加了推断难度。

---

## 下一讲预告

了解了 Token 是什么以及它为何存在后，下一个工程问题是：**词表里的这几万个子词到底是用什么算法从无到有统计出来的？**

在下一讲[/llm-100/basics/BPE_WordPiece_SentencePiece_分词有什么区别.md](/llm-100/basics/BPE_WordPiece_SentencePiece_分词有什么区别.md)中，我们将逐一拆解 BPE（字节对编码）、WordPiece 与 SentencePiece 的合并规则与适用场景。
