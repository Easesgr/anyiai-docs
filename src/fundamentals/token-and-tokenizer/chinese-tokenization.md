---
title: 大模型中文分词为什么特殊？从 UTF-8 编码到词表大小的权衡
icon: mdi:translate
tag:
- Token与分词
- 中文分词
- 多语言
description: 解析中文在 UTF-8 下的三字节编码机制、早期大模型中的字节碎片化现象、现代模型的词表扩充方案，以及显存与生成延迟的工程权衡。
---

在上一篇[《Tokenizer 是怎么工作的？》](/fundamentals/token-and-tokenizer/how-tokenizer-works.md)中，我们介绍了现代分词器（Tokenizer）的核心算法：为了消除未知字符（OOV），主流大模型普遍采用 Byte-level 的子词切分算法（如 Byte-level BPE）。这种设计保证了任何合法的 Unicode 字符都能被底层的 256 个基础字节拼出。

但这套体系在处理中文时遇到了特殊的工程挑战：在早期以英文为主的模型中，处理一段中文所消耗的 Token 数量往往是同等信息量英文的 2 到 3 倍。为什么汉字会面临严重的“Token 膨胀”？现代模型将词表从 3.2 万扩展到 15 万之后带来了哪些收益，又付出了怎样的显存与延迟代价？我们通过一个具体的汉字切分过程展开解析。

![中文分词机制与词表权衡概览](/images/content-rebuild/01935f4abe3522a7/cover.png)

---

## UTF-8 编码结构与汉字表示

计算机底层的文本流最终都表现为二进制字节。现代操作系统和网络通信中最普遍采用的字符编码标准是 **UTF-8**。UTF-8 是一种变长编码方案，针对不同区间的 Unicode 码位分配不同长度的字节：

* **单字节（1 Byte）**：`0xxxxxxx`，覆盖标准 ASCII 字符集（英文字母、数字和常用英文标点，码位 `U+0000` 到 `U+007F`）。
* **双字节（2 Bytes）**：`110xxxxx 10xxxxxx`，覆盖拉丁文补充、希腊文、阿拉伯文等部分符号（码位 `U+0080` 到 `U+07FF`）。
* **三字节（3 Bytes）**：`1110xxxx 10xxxxxx 10xxxxxx`，覆盖绝大部分常用汉字所在的 CJK 统一表意文字区块（码位 `U+4E00` 到 `U+9FFF`）。
* **四字节（4 Bytes）**：`11110xxx 10xxxxxx 10xxxxxx 10xxxxxx`，覆盖生僻古籍字、部分数学符号以及常见的 Emoji 表情。

以汉字“算”为例，其 Unicode 码位为十六进制的 `U+7B97`，二进制表示为 `0111 1011 1001 0111`。将这 16 位二进制按规则填入三字节的 UTF-8 模板：

1. 第一个字节填入前 4 位（`0111`）：`1110` + `0111` $\to$ 二进制 `11100111`（十六进制 `0xE7`）
2. 第二个字节填入中间 6 位（`101110`）：`10` + `101110` $\to$ 二进制 `10101110`（十六进制 `0xAE`）
3. 第三个字节填入后 6 位（`010111`）：`10` + `010111` $\to$ 二进制 `10010111`（十六进制 `0x97`）

$$\text{汉字“算”} \xrightarrow{\text{UTF-8}} [\texttt{0xE7}, \texttt{0xAE}, \texttt{0x97}]$$

![汉字在 UTF-8 变长编码中的三字节结构分解](/images/content-rebuild/01935f4abe3522a7/fig-1.png)

在以 256 个基础字节作为基底的 Byte-level BPE 词表中，任何文本首先被解构成基础字节序列。如果分词器的词表里没有收录“算”这个字，也没有收录对应的字节合并规则，那么这个汉字就会退化为 3 个独立的单字节 Token：`0xE7`、`0xAE`、`0x97`。

---

## 字节碎片化及其工程代价

在 GPT-2 和初代 LLaMA 等早期模型中，训练语料库中英文占比通常在 90% 以上，中文语料极为稀少。这种语料分布直接影响了 BPE 算法的构建过程：

1. **英文占满合并名额**：BPE 算法在统计相邻字节对共现频次时，高频出现的全部是英文字母组合、常见词根和空格（例如 `th`、`in`、`the`）。
2. **中文缺乏共现统计**：中文文本占比极低，导致汉字拆出的 3 个字节很难在统计计数中达到合并阈值。
3. **字节碎片化（Byte Fragmentation）**：大量常用汉字未被收录进词表，分词时只能回退为单字节或双字节碎片。

以包含 5 个汉字的短语“中文大模型”为例，其在 UTF-8 编码下占用 15 个字节：

```text
输入文本: 中文大模型
UTF-8 字节序列 (15 字节):
[0xE4, 0xB8, 0xAD] (中)
[0xE6, 0x96, 0x87] (文)
[0xE5, 0xA4, 0xA7] (大)
[0xE6, 0xA8, 0xA1] (模)
[0xE5, 0x9E, 0x8B] (型)

早期英文分词器切分结果:
[<0xE4>, <0xB8>, <0xAD>, <0xE6>, <0x96>, <0x87>, ...]
Token 消耗: 12 ~ 15 个 Token
```

![字节碎片化导致的 Token 序列膨胀与计算浪费](/images/content-rebuild/01935f4abe3522a7/fig-2.png)

字节碎片化对大语言模型系统的推理效率构成了直接负面影响：

* **有效上下文容量折损**：若模型的最大上下文长度为 4096 Token，在英文语境下约能表达 3000 个英文单词；而当中文每个汉字平均消耗 2.5 个 Token 时，4096 Token 只能容纳约 1600 个汉字，实际容纳的文本信息大幅萎缩。
* **自回归解码延迟显著上升**：Transformer 生成文本以 Token 为步长自回归进行（每前向传播一次计算出一个 Token）。如果模型生成一个“算”字需要串行预测 3 次单字节 Token，其解码端到端延迟就比单字单 Token 模型高出数倍。
* **KV Cache 显存消耗增加**：注意力机制的键值缓存（KV Cache）大小直接与 Token 序列长度 $L$ 成正比。序列长度膨胀使推理服务器能并发承载的请求数（Batch Size）显著下降。

<ReadMoreLock />

---

## 现代多语言模型的词表扩充方案

为缓解碎片化，现代开源与闭源模型（如 LLaMA-3、Qwen、DeepSeek 等）普遍采取了**扩充分词器词表容量**的策略。

通过在分词器构建阶段引入充足的高质量中文文本，BPE 算法在统计时能够顺畅完成多级合并：

1. **单字收敛**：常用汉字对应的 3 个字节（如 `0xE7 + 0xAE` $\to$ 临时节点；临时节点 `+ 0x97` $\to$ “算”）优先合并为单个独立 Token。
2. **高频词合并**：常见的双字词（如“模型”、“技术”）、四字成语乃至专有名词，由于共同出现的频次极高，在后续轮次中被进一步合并为单 Token。

```text
未优化状态（早期英文模型）:
"人工智能" -> [0xE4, 0xBA, 0xBA, 0xE5, 0xB7, 0xA5, 0xE6, 0x99, 0xBA, 0xE8, 0x83, 0xBD] (12 个 Token)

完成单字合并（基础多语言支持）:
"人工智能" -> ["人", "工", "智", "能"] (4 个 Token)

完成多字词合并（现代大词表模型）:
"人工智能" -> ["人工智能"] (1 个 Token) 或 ["人工", "智能"] (2 个 Token)
```

![词表扩充前后从单字节到多字复合 Token 的合并层次变化](/images/content-rebuild/01935f4abe3522a7/fig-3.png)

随着词表规模的逐步扩张，主流模型的中文分词效率演进如下：

| 模型系列 | 分词基底实现 | 词表规模 ($V$) | 常见单字独立消耗 | 常见双字词独立消耗 |
| :--- | :--- | :--- | :--- | :--- |
| LLaMA-1 / LLaMA-2 | SentencePiece (BPE) | 32,000 | 1.5 ~ 2.8 Token | 3.0 ~ 5.0 Token |
| LLaMA-3 | Tiktoken (BPE) | 128,256 | 1.0 ~ 1.2 Token | 1.0 ~ 2.0 Token |
| Qwen-2.5 | HuggingFace Tokenizers (BPE) | 151,936 | 约 1.0 Token | 1.0 ~ 1.5 Token |
| GPT-4o (`o200k_base`) | Tiktoken (BPE) | 200,000 | 约 1.0 Token | 1.0 ~ 1.5 Token |

*注：单汉字独立输入切分时产物至少为 1 个 Token（现代大词表已收录绝大多数常用单字，独立输入消耗为 1.0 Token）。只有在包含多字词合并的连续语料中，按总 Token 数除以字符数计算的“字符单耗（Tokens/字）”才会因多字合并而低于 1.0。*

---

## 文本压缩比的量化测算

在工程评估中，分词效率通常使用**压缩比**或**字符单耗**来度量。为了避免对比指标口径混乱，需统一两种表达方式的定义：

$$\text{字符单耗（Tokens per Character）} = \frac{\text{Token 序列长度}}{\text{输入字符数量}}$$
$$\text{文本压缩比（Characters per Token）} = \frac{\text{输入字符数量}}{\text{Token 序列长度}}$$

在中文工程评估中，习惯使用“字符单耗”（即每个汉字平均消耗多少个 Token）。单耗越接近或低于 1.0，说明分词器针对该语言的表示效率越高。

以下 Python 脚本展示了如何使用 Hugging Face 的 `transformers` 库分析同一句中文在不同分词器下的单耗差异：

```python
# 运行前准备：pip install transformers
from transformers import AutoTokenizer

def evaluate_chinese_efficiency(text: str, tokenizer_name: str):
    # 加载预训练分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # 获取切分后的 Token 序列与 ID 序列
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    char_count = len(text)
    token_count = len(token_ids)
    tokens_per_char = token_count / char_count if char_count > 0 else 0
    
    print(f"分词器: {tokenizer_name}")
    print(f"测试文本: {text}")
    print(f"字符数: {char_count} | Token数: {token_count} | 字符单耗: {tokens_per_char:.3f} tokens/char")
    print(f"切分细节: {tokens}")
    print("-" * 50)

# 示例测试文本（共 19 个字符）：
# "深度学习与大语言模型技术的快速发展。"
# 在早期 32k 词表（如 hf-internal-testing/llama-tokenizer）下，通常产生 30 个以上的 Token（单耗 > 1.5）
# 在现代 150k 词表（如 Qwen/Qwen2.5-7B）下，通常仅产生 9 到 12 个 Token（单耗约 0.5 ~ 0.6）
```

![不同词表规模下中英文文本压缩率与上下文有效负载对比](/images/content-rebuild/01935f4abe3522a7/fig-4.png)

提升压缩比直接带来三项工程收益：

1. **首字延迟（TTFT）降低**：在 Prefill 阶段，序列长度 $N$ 变短，自注意力计算的计算复杂度 $\mathcal{O}(N^2)$ 带来计算量显著降低。
2. **系统并发吞吐提升**：KV Cache 占用随 Token 数量线性减少，单卡在相同显存下可容纳更大的 Batch Size。
3. **有效上下文装填能力翻倍**：在 RAG（检索增强生成）场景下，检索返回的文本块（Chunk）能以更少的 Token 容纳更多知识，缓解了模型窗口上限带来的截断压力。

---

## 词表扩充的代价：显存与计算瓶颈

扩大词表虽然可以显著压缩序列长度，但词表大小（$V$）无法无限制增加。在线性代数层面，扩充词表直接对模型的参数规模和单步解码延迟施加了硬件约束。

### 1. 词表矩阵的显存开销

大模型中有两个张量的大小与词表规模 $V$ 呈线性关系：

* **输入嵌入层（Input Embedding Matrix）**：形状为 $[V, d_{\text{model}}]$，将输入的 Token ID 映射为隐层向量。
* **输出投影层（Unembedding Layer / `lm_head`）**：形状为 $[d_{\text{model}}, V]$，将最后一层的隐层向量映射为词表上的 Logits。

若模型**未开启权重共享（Weight Tying）**，两层独立存在，总参数量计算为：

$$\text{Params}_{\text{vocab}} = 2 \times V \times d_{\text{model}}$$

以典型 7B/8B 规模模型为例，假设隐层维度 $d_{\text{model}} = 4096$，模型权重以 BF16 格式存储（每个参数占 2 字节）：

* **词表规模 $V = 32,000$（早期分词器）**：
  $$\text{参数量} = 2 \times 32000 \times 4096 \approx 2.62 \times 10^8\text{ (约 2.62 亿参数)}$$
  $$\text{静态显存} = 2.62 \times 10^8 \times 2\text{ bytes} \approx 524\text{ MB}$$

* **词表规模 $V = 150,000$（现代多语言分词器）**：
  $$\text{参数量} = 2 \times 150000 \times 4096 \approx 1.23 \times 10^9\text{ (约 12.3 亿参数)}$$
  $$\text{静态显存} = 1.23 \times 10^9 \times 2\text{ bytes} \approx 2.46\text{ GB}$$

* **假设极端扩充至 $V = 1,000,000$**：
  $$\text{参数量} = 2 \times 1000000 \times 4096 \approx 8.19 \times 10^9\text{ (约 81.9 亿参数)}$$
  $$\text{静态显存} = 8.19 \times 10^9 \times 2\text{ bytes} \approx 16.38\text{ GB}$$

*注：部分模型架构采用了权重共享（Weight Tying），即输入嵌入层与输出投影层共享同一个权重矩阵（参数量减半为 $V \times d_{\text{model}}$）。但即便开启共享，$V=100\text{万}$ 时的单个矩阵参数量也将达到 41 亿，在 7B 模型中占据一半以上的体量。*

![输入 Embedding 与输出投影层随词表扩大时的显存扩张与矩阵计算瓶颈](/images/content-rebuild/01935f4abe3522a7/fig-5.png)

### 2. 输出层矩阵计算与访存带宽瓶颈

在自回归解码阶段（Decode），每生成一个 Token，模型都需要通过最后一层的隐藏状态 $\mathbf{h} \in \mathbb{R}^{1 \times d_{\text{model}}}$ 计算在全词表上的分类对数几率：

$$\mathbf{z} = \mathbf{h} \mathbf{W}_{\text{head}} \quad (\mathbf{W}_{\text{head}} \in \mathbb{R}^{d_{\text{model}} \times V})$$
$$\mathbf{p} = \text{Softmax}(\mathbf{z})$$

在 Batch Size 较小时，Decode 阶段的矩阵向量乘法属于典型的**访存受限（Memory-bound）**操作。GPU 计算单元大部分时间处于等待显存读取权重的状态。词表 $V$ 越大，每生成一个 Token 就必须从显存中搬运越庞大的 $\mathbf{W}_{\text{head}}$ 矩阵，同时 Softmax 归一化操作的规约开销也随之增大，导致单步生成延迟升高。

### 3. 长尾词汇的训练不充分

若盲目把专业领域罕见词、长成语全部加入词表，这些 Token 在实际训练语料中的出现频次极低。在未开启权重共享且输入序列未命中该词时，其输入嵌入行缺乏有效的前向激活与数据梯度更新；而即便开启权重共享（输出层负类参与 Softmax 计算回传负梯度）或存在优化器权重衰减，由于缺乏足够丰富多样的共现上下文，其向量难以收敛到高质量的语义流形，容易导致表征退化。

![词表扩充在序列长度收益与矩阵计算开销之间的工程平衡曲线](/images/content-rebuild/01935f4abe3522a7/fig-6.png)

综合考虑序列压缩率、静态显存占用与 Decode 访存延迟，当前主流多语言大模型普遍将词表大小权衡在 **10 万至 15 万**（部分如 GPT-4o 推进至 20 万）。这一规模既覆盖了绝大多数汉字单字与高频词汇（字符单耗控制在 0.6 到 0.9 左右），又将词表显存与输出层计算开销约束在合理范围内。

---

## 知识自测与延伸

完成阅读后，可通过以下问题检验对中文分词机制与系统权衡的理解：

1. 某模型分词器未收录汉字“算”（UTF-8 编码为 `0xE7 0xAE 0x97`），且其词表中没有对应的相邻字节合并规则。该汉字输入时会被切为几个 Token？在没有 Speculative Decoding 等加速技术的前提下，自回归生成该汉字需要几步解码？
2. 为什么工业界模型极少把词表容量扩充到 100 万以上？请从参数显存开销与单步自回归延迟两个角度简要说明限制机制。

**参考解答提示：**
* 第 1 题：输入会被切分为 3 个单字节 Token（`0xE7`、`0xAE`、`0x97`）；解码生成该字时，模型需要进行 3 步自回归迭代依次预测这 3 个字节。
* 第 2 题：参数显存维度，未开启权重共享时，百亿级参数将集中于词表矩阵（如 7B 隐层 $d=4096$ 时仅两层矩阵就占约 16 GB 显存）；延迟维度，Decode 阶段是访存受限操作，每步预测都要从显存完整加载巨大的输出投影矩阵，词表过大会导致单步延迟显著上升。

在掌握了分词编码机制后，如何根据不同模型的字符单耗科学规划 Prompt 长度与预算控制？欢迎继续阅读下一篇：[《上下文长度、Token 预算和接口费用到底怎么算？》](/fundamentals/token-and-tokenizer/token-budget.md)。若想了解分词切分边界如何影响下游 RAG 检索的关键词召回，可参考进阶篇[《Tokenization 对 Prompt、RAG 和长文本处理有什么影响？》](/fundamentals/token-and-tokenizer/tokenization-impact.md)。
