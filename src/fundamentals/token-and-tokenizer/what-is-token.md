---
title: Token 是什么？大模型如何把文本变成数字输入
icon: mdi:cube-outline
tag:
- Token与分词
- Token
- 大模型基础
description: 解析大模型输入端 Token 与 Token ID 的映射机制，阐明子词切分必要性、前导空格、特殊标记及流式解码断裂处理。
---

大模型本质上是一个巨大的浮点数计算网络，它的每一层矩阵乘法、注意力加权都在处理数字向量，完全无法直接读取由自然语言字符构成的原始字符串。在模型眼里，无论中文、英文还是代码，都必须先被切分成一个个离散片段，并转换成固定的整数编号，这个最小片段就是 **Token**（词元），对应的整数编号称为 **Token ID**。

理解 Token 的切分方式、整型张量表示以及特殊控制标记，是掌握模型上下文窗口长度、API 计费成本以及 Prompt 编写规范的基础。分词算法的具体实现与训练过程可参考 [/fundamentals/token-and-tokenizer/how-tokenizer-works.md](/fundamentals/token-and-tokenizer/how-tokenizer-works.md)；中文字词的分词效率与压缩比细节可参考 [/fundamentals/token-and-tokenizer/chinese-tokenization.md](/fundamentals/token-and-tokenizer/chinese-tokenization.md)。

![离散输入单元：从变长字符流到定长整型 Token ID 序列的投影映射](/images/content-rebuild/e0198bce55e9708f/cover.png)

---

## 为什么不直接用字符或完整单词？

计算机里的文本是由字符编码（如 UTF-8）构成的字节流。如果直接把文本送进神经网络，工程上通常有两种最直观的设想，但它们都存在明显缺陷：

1. **按单个字符（或单字节）切分**：
   * **做法**：英文字符以 26 个字母加标点为单元，或者直接以 256 个基础字节（0~255）作为单元。
   * **问题**：词表非常小，且永远不会出现无法识别的未知字符（Out-of-Vocabulary, OOV）。但是，文本序列会变得极长。例如一个 1000 汉字的段落，在 UTF-8 下占用 3000 个字节，序列长度直接达到 3000。现代大语言模型的全注意力计算量与显存占用在序列维度具有平方级增长的特性，过长的序列不仅大幅增加推理显存和计算耗时，而且单个字符承载的信息过于稀疏，模型需要跨越大量步长才能拼合出完整的概念。
2. **按完整单词切分**：
   * **做法**：英文以空格为界切成完整单词（如 `apple`, `learning`），中文使用分词字典切成词语。
   * **问题**：自然语言中的词汇量呈开放式增长。随着时态、单复数、派生词、新专有名词以及错别字的出现，词表会迅速膨胀到几百万甚至上千万。词表过大会导致模型最前方的嵌入矩阵（Embedding）和最后的分类输出层（LM Head）显存占用不可接受；更严重的是，一旦遇到词表外的新词，模型只能将其替换为 `<unk>`（Unknown）符号，直接丢失原始信息。

为了平衡**词表大小**与**序列长度**，主流大语言模型统一采用了**子词切分（Subword Tokenization）**方案。

子词切分通过统计语料中的字符搭配规律，将高频出现的完整词保持为单个 Token（例如英文常用词 `the`、中文常用词 `学习`），而将低频词拆分成常见的前缀、词根或后缀（例如 `unbelievable` 拆成 `un`、`believ`、`able`）。极端罕见的字符则退化为字节序列表示。这样既把词表控制在合理规模（常见范围为 32,000 到 150,000 之间），又保证了任何文本都能被切分表达，不发生截断丢失。

![文本不定长字节流与连续稠密浮点计算矩阵之间的离散映射断层](/images/content-rebuild/e0198bce55e9708f/fig-1.png)

---

## Token 与 Token ID：从静态词表到整数张量

在代码实现中，**Token** 和 **Token ID** 是两个紧密关联但类型不同的概念：

* **Token（词元）**：分词器在词表中定义的字符串片段，例如 `"hello"`、`" learning"`、`"Ġworld"`。
* **Token ID**：该字符串片段在静态词表数组中所对应的唯一整型行号（非负整数）。

假设某个模型的词表 $\mathcal{V}$ 大小为 $V = |\mathcal{V}|$，每个词项对应一个在 $[0, V - 1]$ 范围内的整数索引：

$$\mathcal{V} = \{s_0: 0, s_1: 1, s_2: 2, \dots, s_{V-1}: V-1\}$$

当一段文本被分词器解析后，它变成了一个由整型数值构成的向量：

$$\mathbf{x} = [t_1, t_2, \dots, t_L], \quad t_i \in \{0, 1, \dots, V-1\}$$

<ReadMoreLock />

这里需要明确：**Token ID 本身没有任何连续空间上的几何距离或大小含义**。ID 为 `1024` 的 Token 和 ID 为 `1025` 的 Token，仅仅是词表里的两个不同编号，并不代表它们语义相近。

当输入文本以批次（Batch）形式提供给深度学习框架时，数据被规整为一个二维张量：

* **张量形状（Shape）**：`[Batch_Size, Sequence_Length]`（记为 $[B, L]$）。
* **数据类型（Dtype）**：64 位有符号整数 `torch.int64`（或 `torch.long`），部分经过优化的推理系统也支持 `int32`。

在模型的最前端，这个二维整数张量传入形状为 $[V, d_{\text{model}}]$ 的静态 Embedding 矩阵中执行查表操作，依照行号取出每一行对应的连续浮点向量，输出形状为 $[B, L, d_{\text{model}}]$ 的连续隐藏状态张量。在此步骤之前，模型输入完全由离散整数构成。

![静态词表双向映射结构与输入张量进入 Embedding 矩阵前的维度变换](/images/content-rebuild/e0198bce55e9708f/fig-2.png)

---

## 字符、词与 Token 的切分边界

在实际工程使用中，Token 的切分方式经常与开发者的直觉产生出入。

### 英文的前导空格处理

现代主流分词器（如 Byte-level BPE 和基于 SentencePiece 的实现）并不把空格作为丢弃式分隔符，而是把空格作为普通字符内容包含在 Token 内部。

在 GPT 系列分词器中，空格通常映射为特殊可视字符 `Ġ`；在 Llama、Gemma 等模型采用的 SentencePiece 规范中，空格通常用下划线符号 `_`（` `）表示。

这种设计导致：**同一个单词在句首和句中会被切分成不同的 Token**。

* 句首的 `"Apple"`（前面无空格）：匹配词表中无空格前缀的条目 `"Apple"`，对应 Token ID 为 `35890`。
* 句中的 `" Apple"`（前面有空格）：匹配词表中带有前导空格的条目 `"ĠApple"`，对应 Token ID 为 `25055`。

在 Prompt 拼接中，如果开发者手动处理字符串时不慎多加或删除了单词间的空格，模型接收到的 Token ID 序列会发生改变，直接影响后续的注意力表征。关于拼接上下文时的格式控制，可参考 [/frontier/context-engineering/what-is-context-eng.md](/frontier/context-engineering/what-is-context-eng.md)。

### 中文的 Token 粒度差异

中文书写连续且无空格。早期以英文为核心的模型（例如词表大小约为 50,000 的 GPT-2），词表中极少收录常用汉字，导致大部分中文汉字被拆分为 3 个独立的 UTF-8 字节 Token（即输入 1 个汉字消耗 3 个 Token）。

现代针对多语言与中文优化的模型（如 Qwen 系列，词表扩大至 150,000 以上），不仅收录了绝大多数常用单字，还收录了高频双字词和行业术语（如“人工智能”、“模型”）。常见词汇可以作为单个 Token 编码，显著减少了中文文本的 Token 消耗。

| 输入片段 | 语言 | 切分结果（Token 字符串） | Token 数量 | 机制说明 |
| :--- | :--- | :--- | :--- | :--- |
| `"Learning"` (句首) | 英文 | `["Learning"]` | 1 | 句首无空格，首字母大写单词元 |
| `" learning"` (句中) | 英文 | `["Ġlearning"]` | 1 | 句中带前导空格，匹配含空格前缀的独立词元 |
| `"tokenization"` | 英文 | `["token", "ization"]` | 2 | 复合派生词被拆分为词根与后缀 |
| `"深度学习"` | 中文 | `["深度", "学习"]` | 2 | 在扩充词表中分别匹配双字词 |
| `"砼"` (生僻汉字) | 中文 | `["砼"]` 或 `[byte_0, byte_1, byte_2]` | 1 或 3 | 词表已收录则为 1 个 Token；未收录则退化为 3 个单字节 Token |

![英文前导空格感知与中文字词粒度差异的切分对比](/images/content-rebuild/e0198bce55e9708f/fig-3.png)

---

## 特殊 Token（Special Tokens）的作用

除了代表自然语言文字的普通 Token，分词器中还包含一组承担控制功能的特殊标记，它们同样占用词表位置并拥有固定的 Token ID。

### 1. 序列边界标记：`<bos>` 与 `<eos>`

* **`<bos>` / `<s>` (Beginning of Sequence)**：序列起始符号，通常放置在输入最前端，用于通知模型自此开始构建语义状态。
* **`<eos>` / `</s>` (End of Sequence)**：序列结束符号。自回归生成模型以循环逐步预测下一个 Token。在每一步采样中，当模型预测出的 Token ID 等于 `<eos>` 的 ID 时，推理服务便判定本次生成结束并退出循环。如果该符号丢失或配置错误，模型将无法停止，直到打满 `max_tokens` 阈值并输出重复语句。

### 2. 批处理补齐：`<pad>` 与 Attention Mask

当多条不同长度的文本放在同一个 Batch 中送入模型并行计算时，必须将它们对齐到相同的序列长度 $L_{\max}$。较短的序列需要使用 `<pad>` 符号填充补齐（在并行训练或特征提取时常用末尾右侧填充，而在 Decoder-only 模型进行批处理自回归生成时，工程上通常要求在序列开头进行左侧填充，以保证新生成的 Token 正确紧随有效上下文）。

为避免模型对填充标记进行无效的自注意力计算，系统会配套生成一个布尔型或二值整型的 **Attention Mask**：

$$\text{Mask}_{i} = \begin{cases} 1, & \text{有效文本位置} \\ 0, & \text{填充位置（Pad）} \end{cases}$$

在底层计算自注意力权重时，掩码为 0 的位置会被加上一个极大的负数偏置（通常为 $-10000.0$ 或 $-\infty$），经过 Softmax 后权重归零，防止填充位置干扰有效文本的输出。

### 3. 多轮对话角色标记

在多轮对话模型中，系统通过特殊标记（如 ChatML 规范）构建角色边界：

```text
<|im_start|>system
你是一个代码助手。<|im_end|>
<|im_start|>user
请解释什么是 Token。<|im_end|>
<|im_start|>assistant
```

这些标记作为结构分隔符，帮助模型在自注意力中准确识别哪部分是系统设定、哪部分是用户提问。如果在处理不可信外部输入时没有对这类特殊标记进行转义，攻击者就可能通过注入这些文本片段来伪造对话角色。

![特殊 Token 在序列边界标识、批处理填充掩码及对话结构隔离中的系统功能](/images/content-rebuild/e0198bce55e9708f/fig-4.png)

---

## 编码、解码与流式输出中的截断处理

分词器是一个具备双向映射能力的流水线系统：

1. **编码（Encode）**：原始字符串 $\to$ 规范化（Normalization） $\to$ 预分词（Pre-tokenization） $\to$ 子词匹配（Model Match） $\to$ 添加特殊标记（Post-Processing） $\to$ 输出整型 ID 列表。
2. **解码（Decode）**：整型 ID 列表 $\to$ 查表还原为字节/子词 $\to$ 拼接字节流 $\to$ UTF-8 解码还原为自然文本。

```
原始文本 ──► [ 正规化与预分词 ] ──► [ 子词切分匹配 ] ──► Token ID 整数序列 (送入模型)
                                                               │
完整文本 ◄── [ 字节拼接还原 UTF-8 ] ◄── [ 词表反向查表 ] ◄──────┘
```

### Byte-level BPE 的可逆性

由于 Byte-level BPE 将 256 个基础单字节直接作为词表的基底，任何文本即使完全由生僻字符或不可见二进制符号构成，也能退化为单字节 Token 进行表示，系统无需使用 `<unk>` 丢弃未知字符。在分词器未配置有损规范化（如强行转小写、去除特殊符号）且编解码时排除额外控制标记的前提下，编码与解码过程是**无损可逆**的：

$$\text{Decode}(\text{Encode}(\text{Text})) \equiv \text{Text}$$

### 流式输出时的中文乱码问题

在流式生成（Streaming Response）中，服务端每生成一个 Token 就需要解码并推送给客户端。在 UTF-8 编码下，一个标准中文字符占用 3 个字节。

某些模型在生成中文时，可能会将一个汉字的 3 个字节拆分到两个连续的 Token 中（例如前一个 Token 携带前 2 个字节，后一个 Token 携带第 3 个字节）。如果服务端对收到的每个 Token 直接独立调用 `decode("utf-8")`，前一个 Token 就会因字节不完整而解码失败：在默认严格模式下会直接抛出 `UnicodeDecodeError` 异常，在替换模式下则会输出形如 ``（Unicode 替换字符 `\ufffd`）的乱码字符。

因此，流式解码器必须维护一个**增量字节缓冲区（Incremental Decoder）**：当遇到不完整的 UTF-8 字节序列时，先将其缓存在内存中；直到后续 Token 补齐剩余字节后，再统一解码为正确的文字对外输出。

![完整分词编码与解码生命周期管线及流式输出下字节断裂处理机制](/images/content-rebuild/e0198bce55e9708f/fig-5.png)

---

## 代码验证：Token ID 映射与张量规格

以下代码使用官方 `transformers` 库验证前导空格对 Token ID 的影响，并展示输入模型前张量的维度与数据类型：

```python
from transformers import AutoTokenizer
import torch

# 以 GPT-2 分词器为例（采用 Byte-level BPE）
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # 显式设置填充符号用于批处理演示

text_a = "Apple"
text_b = " Apple"  # 前置一个空格

# 1. 验证前导空格对 Token ID 的影响
tokens_a = tokenizer.tokenize(text_a)
tokens_b = tokenizer.tokenize(text_b)
ids_a = tokenizer.encode(text_a)
ids_b = tokenizer.encode(text_b)

print(f"句首无空格: 文本='{text_a}' -> Token={tokens_a} -> ID={ids_a}")
print(f"句中带空格: 文本='{text_b}' -> Token={tokens_b} -> ID={ids_b}")
# 运行输出：
# 句首无空格: 文本='Apple' -> Token=['Apple'] -> ID=[35890]
# 句中带空格: 文本=' Apple' -> Token=['ĠApple'] -> ID=[25055]

# 2. 验证多条文本批处理时的张量规格
batch_texts = ["Hello world!", "What is a token?"]
encoded_inputs = tokenizer(
    batch_texts,
    padding=True,          # 自动补齐为等长
    return_tensors="pt"    # 返回 PyTorch 张量
)

input_ids = encoded_inputs["input_ids"]
attention_mask = encoded_inputs["attention_mask"]

print("\n--- 张量物理规格验证 ---")
print(f"input_ids 形状 (Shape): {input_ids.shape}")        # 输出: torch.Size([2, 5])
print(f"input_ids 数据类型 (Dtype): {input_ids.dtype}")    # 输出: torch.int64
print(f"input_ids 数值内容:\n{input_ids}")
print(f"attention_mask 数值内容:\n{attention_mask}")
```

该实验印证了两个事实：
1. 同样的字符在句首和句中会因前导空格被映射为完全不同的 Token ID。
2. 送入模型的输入是形状为 `[Batch_Size, Sequence_Length]` 的 `torch.int64` 整型张量，配合 `attention_mask` 协同完成批处理对齐。

![读者自测实验流程与核心张量形状状态对照图](/images/content-rebuild/e0198bce55e9708f/fig-6.png)

---

## 读者自测与核心概念核对

为检验对 Token 机制的理解，可核对以下两个问题：

* **问题一**：为什么英文单词开头的空格会直接改变分词后的 Token ID？
  * **机制回答**：现代分词器（如 Byte-level BPE 和 SentencePiece）不丢弃空格，而是把空格作为字符内容编入 Token 内部（如标记为 `Ġ` 或 ` `）。句首的单词没有空格，匹配基础单词条目；句中的单词带有前导空格，匹配词表中包含空格前缀的独立条目，因此在词表中对应不同的索引。
* **问题二**：词表大小为 32,000 的模型在接收单条长度为 20 的文本时，送入 Embedding 层之前的张量形状和数据类型分别是什么？
  * **机制回答**：单条文本的 Batch Size 为 1，序列长度为 20，因此张量形状为 `[1, 20]`。数据类型必须为整型（通常是 `torch.int64`），作为行号索引去访问形状为 `[32000, d_model]` 的 Embedding 查找表。

---

## 学习路径导引

理解了 Token 和 Token ID 构成的整型输入后，可以沿着以下路径继续深入：

* **分词算法演进**：阅读 [/fundamentals/token-and-tokenizer/how-tokenizer-works.md](/fundamentals/token-and-tokenizer/how-tokenizer-works.md)，掌握 BPE、Unigram 与 WordPiece 三种主流子词算法的学习过程与切分细节。
* **中文分词调优**：阅读 [/fundamentals/token-and-tokenizer/chinese-tokenization.md](/fundamentals/token-and-tokenizer/chinese-tokenization.md)，了解多语言大模型扩充中文词表的实践与压缩比优化。
* **工程成本与显存**：关于不同序列长度下的显存占用与计算开销，请参考 [/fundamentals/token-and-tokenizer/token-budget.md](/fundamentals/token-and-tokenizer/token-budget.md) 与 [/architecture/token-cost.md](/architecture/token-cost.md)。
