---
title: BPE、WordPiece 与 SentencePiece 分词有什么区别？大模型分词算法机制拆解
icon: mdi:book-open-page-variant-outline
date: 2026-07-16
tag:
- 大模型100讲
- 基础篇
description: 对比BPE、WordPiece与SentencePiece的词表构建准则、推断切分方式与未登录词处理机制，附可运行代码练习。
---

在上一讲[/llm-100/basics/Token_是什么_为什么大模型不是按字理解世界.md](/llm-100/basics/Token_是什么_为什么大模型不是按字理解世界.md)中，我们明确了大模型处理语言的最小计算单元是 Token。完全基于词（Word）的分词方式会导致词表无限膨胀且无法处理未登录词（Out-Of-Vocabulary, OOV），而纯粹基于字符（Character）的分词又会导致序列过长、单步计算携带的信息密度过低。

子词切分（Subword Tokenization）是目前的主流平衡方案。同样是子词切分，不同模型采用了不同的算法与实现：GPT 系列采用 BPE 及其字节衍生版（Byte-level BPE），BERT 采用 WordPiece，而 T5 与 LLaMA 则依赖 SentencePiece 框架。这三类方案在词表构建准则、推断切分方式以及多语言适应性上存在明确差异。

![分词算法全景对比封面图](/images/content-rebuild/c553975c23e16bba/cover.png)

### 结论速览与贯穿算例

为了直观理解这三种算法的区别，我们先看两项核心维度的对比结论：
1. **构建词表的合并标准不同**：BPE 每次选择语料中**共现频次最高**的相邻单元；WordPiece 选择能**最大化互信息增益（提升语言模型整体似然度）**的相邻单元；SentencePiece 则是一个框架，不发明单一数学准则，内部支持 BPE 和基于概率剪枝的 Unigram 算法。
2. **对空格与原始输入的假设不同**：BPE 与 WordPiece 原生依赖预分词（把空格作为天然分割符，切分不能跨词），对无空格语言需要外部工具，且分词过程不可逆；SentencePiece 把文本视作不间断的字符流，将空格替换为可见的元符号（如 ` `），实现无损可逆分词。

下文我们将统一使用由 `"low"`（5次）、`"lower"`（2次）、`"newest"`（6次）、`"widest"`（3次）组成的教学语料，并以未见词 `"lowest"` 的切分为贯穿例子，逐步拆解它们的运行机制。

---

### 1. BPE：基于相邻符号对频次的贪心合并

字节对编码（Byte-Pair Encoding, BPE）最初用于通用数据压缩，后被引入作为子词词表构建方案。其运行机制是：**自底向上，从基础字符集合出发，迭代合并语料中共现频次最高的相邻符号对**。

#### 训练阶段：构建合并规则表

给定示例语料及词频：
* `"low": 5`
* `"lower": 2`
* `"newest": 6`
* `"widest": 3`

构建流程分为四个步骤：
1. **字符拆解与边界标记**：将单词切分成单字符，在词尾增加结束符 `</w>` 以保留词边界。初始基础词表 $V$ 仅包含出现的独立字符：
   $$V = \{l, o, w, e, r, n, s, t, i, d, </w>\}$$
2. **统计相邻符号对频次**：
   - `("l", "o")`：出现在 `"low"`（5次）和 `"lower"`（2次），频次为 $5 + 2 = 7$
   - `("w", "e")`：出现在 `"lower"`（2次）和 `"newest"`（6次），频次为 $2 + 6 = 8$
   - `("e", "s")`：出现在 `"newest"`（6次）和 `"widest"`（3次），频次为 $6 + 3 = 9$
   - `("s", "t")`：出现在 `"newest"`（6次）和 `"widest"`（3次），频次为 $6 + 3 = 9$
3. **执行最高频合并**：当前最高频次为 9。假设在并列项中优先选取 `("e", "s")`。系统将所有相邻的 `"e"` 和 `"s"` 替换为新符号 `"es"`，并将 `"es"` 加入词表 $V$。同时在全局规则表中追加第 1 条规则：`("e", "s") -> "es"`。
4. **迭代统计与更新**：更新语料后继续统计。此时相邻对 `("es", "t")` 的频次为 $6 + 3 = 9$，成为新的最高频对，合并为 `"est"`，记录第 2 条规则：`("es", "t") -> "est"`。重复该过程，直到词表达到设定大小或候选频次低于阈值。

![BPE 迭代合并机制流程](/images/content-rebuild/c553975c23e16bba/fig-1.png)

#### 推断阶段：应用有序规则切分文本

训练完成后，BPE 会生成一张**严格按合并先后顺序排列的规则表**。当处理未见词 `"lowest"` 时：
1. 拆解为初始字符序列：`['l', 'o', 'w', 'e', 's', 't', '</w>']`。
2. 遍历规则表。当匹配到第 1 条规则 `("e", "s") -> "es"` 时，序列变为 `['l', 'o', 'w', 'es', 't', '</w>']`。
3. 继续匹配到第 2 条规则 `("es", "t") -> "est"`，序列变为 `['l', 'o', 'w', 'est', '</w>']`。
4. 匹配到后续规则 `("est", "</w>") -> "est</w>"`，序列变为 `['l', 'o', 'w', 'est</w>']`。
5. 继续匹配后续规则 `("l", "o") -> "lo"` 与 `("lo", "w") -> "low"`，序列合并为 `['low', 'est</w>']`。
6. 没有规则可再命中时，输出最终切分结果：`['low', 'est</w>']`。

BPE 推断的核心特征在于：切分由有序的静态规则表驱动，不需要在推理阶段维护概率图或计算最优路径。

---

### 2. WordPiece：基于互信息得分的子词选择

WordPiece 与 BPE 同样采取自底向上的合并路径，但两者的合并选取准则不同。BPE 只看符号对的绝对共现频次，而 WordPiece 的目标是**最大化训练语料在单字语言模型下的对数似然度**。

#### 合并依据：对高频孤立字符的惩罚

如果单纯根据频次合并，语料中极高频出现的独立字符（例如定冠词中的单个字母、高频虚词）容易与相邻词缀频繁组合，即使它们之间缺乏紧密的语义关联。在单字独立（Unigram）假设下，将符号对 $(x, y)$ 合并为新符号所带来的语言模型对数似然增益，近似等价于两者的点互信息（PMI）。WordPiece 在工程实现中常将候选合并评估简化为互信息打分公式：

$$\text{Score}(x, y) = \frac{\text{count}(xy)}{\text{count}(x) \times \text{count}(y)}$$

其中 $\text{count}(xy)$ 是组合符号在语料中连续出现的频次，$\text{count}(x)$ 与 $\text{count}(y)$ 是各自独立出现的总频次。

![WordPiece 与 BPE 合并依据对比](/images/content-rebuild/c553975c23e16bba/fig-2.png)

该评分具有两重效果：
- **抑制虚假高频**：若 $x$ 本身是全语料的高频单字（$\text{count}(x)$ 极大），分母的放大效应会拉低该候选对的得分。
- **保留强关联词根词缀**：只有当 $x$ 和 $y$ 几乎总是成对出现、很少单独出现时，得分才会显著升高。这促使算法优先提炼语法和语义上关联更强的构词成分（如 `"un"` 与 `"happy"`）。

#### 推断策略：最长优先匹配（MaxMatch）

在实现中（如 BERT 分词器），WordPiece 为非词首子词增加了显式的前缀标记（通常为 `##`），例如 `["play", "##ing"]`。

推断阶段并不依赖规则表的迭代扫描，而是针对预分词后的单词采取**从左至右的贪心最长匹配（MaxMatch）**：
1. 从单词首字符开始，在词表中查找能够匹配的最长前缀子串。
2. 命中后截断前缀，对剩余部分继续在词表中寻找带有 `##` 前缀的最长子串。
3. 若剩余某段子串在词表中完全找不到任何匹配项，整词退化标记为 `[UNK]`。

<ReadMoreLock />

---

### 3. 预分词依赖：空格假设在多语言中的局限

早期的 BPE 与 WordPiece 实现均建立在一项前提之上：**预分词（Pre-tokenization）**。分词器在统计字符频次和构建词表前，必须先利用正则表达式或空白字符将原始文本切分成离散的“词”。

#### 空格假设的破裂与连锁问题

以英文为代表的印欧语系天然以空格作为词边界，分词器只要规定“合并不能跨越空格”，就能把统计复杂度约束在单个词内部。

但在更广泛的语言场景下，这一假设会带来两个工程问题：
1. **无空格语言切分困难**：中文、日文、泰文等语言的书写系统中没有自然空格。若要在这些语言上运行传统 BPE，必须预先挂载外部切词组件（如中文 Jieba、日文 MeCab）。外部工具的切分错误会被直接固化进子词词表中，无法与模型联合优化。
2. **不可逆性（Lossy Pre-tokenization）**：在预切分阶段，如果将多个连续空格、制表符或换行符归一化或丢弃，解码阶段就无法还原原始文本的字符级排版（例如代码缩进或多空格排版）。

![预分词在无空格语言中的失效与歧义](/images/content-rebuild/c553975c23e16bba/fig-3.png)

---

### 4. SentencePiece：统一字符流与可逆重构

为了消除对特定语言预分词规则的依赖，SentencePiece 被提出作为端到端的开源分词框架。需要明确的是：**SentencePiece 本身不是某一种单一的切分数学公式，而是一套分词工程框架**。它内部集成了 BPE 算法和 Unigram 语言模型算法（Unigram 采用从大候选集合出发、基于损失剪枝的自顶向下策略）。

SentencePiece 处理无空格文本与多语言混合文本的核心机制包含两个部分：

#### 1. 空格转义为可见元符号（Meta Symbol）
SentencePiece 不预先将句子拆分成词，而是将整篇输入视为连续的**原生字符流**。它引入特殊的转义字符（默认使用 Unicode 下划线 ` `，U+2581）显式替换文本中的空格。

例如原始输入：
```text
"Hello World!"
```
进入算法前被转换为：
```text
" Hello World!"
```
空格在此被赋予与其他字母同等的统计地位。BPE 或 Unigram 可以直接统计跨越原空格位置的频次，从而学出带有前置空格标记的子词 Token（如 `" World"`）。

![SentencePiece 可逆切分机制](/images/content-rebuild/c553975c23e16bba/fig-4.png)

#### 2. 无损可逆重构（Lossless Tokenization）
由于所有排版信息（换行、空格、标点）均被 1:1 保留并映射到 Token 序列中，解码过程无需依赖特定语言的标点复原逻辑，直接执行字符串拼接并还原下划线即可完全恢复原始排版：

$$\text{Detokenize}(\text{Tokens}) = \text{Replace}(\text{Join}(\text{Tokens}), " ", " ")$$

这一特性使 SentencePiece 成为跨语言模型（如 mT5、LLaMA、BLOOM）的基础分词组件。

---

### 5. Byte-level BPE 与字节回退：消除未登录词

虽然 SentencePiece 解决了空格假设问题，但分词系统依然受制于**基础字符集的大小**。Unicode 字符空间极其庞大（涵盖数万个汉字、生僻符号与 Emoji）。若将语料中所有出现过的 Unicode 字符全部塞进基础词表，词表底座往往在合并前就已经膨胀到数万量级。

为了兼顾词表体积并**避免产生未登录词标记（`<unk>`）**，工程上发展出两条路线：

#### 路线 A：Byte-level BPE（BBPE）
GPT-2 及其后续模型采用了 Byte-level BPE。该方案直接在 **UTF-8 编码的原始字节序列**上构建和应用 BPE：
1. 基础词表大小在初始时刻固定为 **256**（对应单字节的全部取值范围 `0x00` 到 `0xFF`）。
2. 字符无论来自哪个语种，均转换为 1 到 4 个字节。例如汉字 `"你"` 的 UTF-8 编码为 3 个字节：`0xE4 0xBD 0xA0`。
3. BPE 合并在字节序列上运行。高频汉字的三字节组合会逐步被合并成单个 Token；生僻字若未被合并，则保留为 3 个独立的单字节 Token。

由于基础词表已经穷举了全部 256 个字节取值，**系统可以表示任意合法的字节流，不再需要 `<unk>` 标记**。

![Byte-level BPE 与字节回退流程](/images/content-rebuild/c553975c23e16bba/fig-5.png)

#### 路线 B：SentencePiece 的字节回退（Byte Fallback）
SentencePiece 采取折中设计：基础词表主要覆盖语料中绝大部分高频 Unicode 字符。当遇到极罕见的字符或特殊 Emoji 时，系统并不输出 `<unk>`，而是动态将其拆解为 UTF-8 字节 Token（如 `<0xE4><0xBD><0xA0>`）。LLaMA 系列分词器即使用了这一机制。

#### 必须注意的边界代价：序列长度膨胀
字节级方案避免了 `<unk>`，但存在序列长度的代价。中文、日文在 UTF-8 编码下通常每个字符占用 3 个字节。如果在预训练语料中某语言占比较低、分词器没有为其分配足够的合并配额，该语言文本就会被打碎成大量单字节 Token。这会导致**相同语义下该语言的 Token 数量达到英文的 2 到 3 倍**，从而挤占模型的有效上下文窗口，并成倍增加自回归推理的计算量。

---

### 6. 横向对比与选型参考

下表汇总了四种典型分词配置的关键维度对照：

| 算法 / 框架 | 初始单元 | 词表构建准则 | 推断切分方式 | 空格与边界表示 | OOV 处理能力 | 典型应用模型 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BPE** | 字符 | 相邻符号对共现频次最大化 | 静态合并规则表优先级匹配 | 依赖预分词切除；无显式前缀或使用词尾符 `</w>` | 基础字符未覆盖时退化为 `<unk>` | GPT-1、RoBERTa |
| **WordPiece** | 字符 | 互信息得分（似然增益）最大化 | 贪心最长匹配优先（MaxMatch） | 依赖预分词切除；非词首子词加 `##` 前缀 | 匹配失败则全词退化为 `[UNK]` | BERT、DistilBERT |
| **SentencePiece** | 字符 / 字节 | 支持 BPE 频次合并或 Unigram 似然剪枝 | 规则匹配（BPE）或 Viterbi 最优路径（Unigram） | 将空格转义为独立元符号 ` ` | 支持字节回退（Byte Fallback）消除 `<unk>` | T5、LLaMA 系列、BLOOM |
| **Byte-level BPE** | 字节（0-255） | 相邻字节对共现频次最大化 | 字节级静态规则表优先级匹配 | 字节重映射为可见字符，空格随字节编码 | 初始底座覆盖 256 完整字节，天然无 `<unk>` | GPT-2/3/4、Tiktoken |

![分词方案工业选型决策图](/images/content-rebuild/c553975c23e16bba/fig-6.png)

---

### 7. 可验证练习：实现 BPE 的训练与切分推断

为了检验对 BPE 机制的理解，我们编写一个包含**训练合并**与**推断切分**完整流程的 Python 脚本。你可以直接运行该代码，观察合并规则如何生成，并验证新词 `"lowest"` 是如何被切分的：

```python
import re
from collections import defaultdict

# 1. 训练数据：模拟小型语料及词频统计
train_corpus = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3
}

def get_stats(vocab):
    """统计当前语料中所有相邻符号对的共现频次"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, v_in):
    """在语料中将选定的相邻对合并为新子词"""
    v_out = {}
    bigram = re.escape(' '.join(pair))
    # 仅匹配独立完整的两个符号
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    replacement = ''.join(pair)
    for word in v_in:
        w_out = pattern.sub(replacement, word)
        v_out[w_out] = v_in[word]
    return v_out

# 2. 训练：迭代执行 6 次合并，记录有序规则表
num_merges = 6
vocab = train_corpus.copy()
merge_rules = []

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    # 选取频次最高的相邻对
    best_pair = max(pairs, key=pairs.get)
    vocab = merge_vocab(best_pair, vocab)
    merge_rules.append(best_pair)
    print(f"合并 #{i+1}: {best_pair} (频次: {pairs[best_pair]})")

# 3. 推断：使用学到的规则切分新词
def tokenize_bpe(word, rules):
    """根据有序合并规则表对输入单词进行 BPE 切分"""
    # 初始状态：字符序列并添加边界符
    symbols = list(word) + ["</w>"]
    
    for pair in rules:
        i = 0
        new_symbols = []
        while i < len(symbols):
            # 如果遇到连续两个符号匹配规则，则执行合并
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i+1] == pair[1]:
                new_symbols.append(pair[0] + pair[1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols

# 验证新词切分
test_word = "lowest"
tokens = tokenize_bpe(test_word, merge_rules)
print(f"\n新词 '{test_word}' 切分结果: {tokens}")
```

#### 思考与验证
运行上述代码后，你会发现：
1. 词对 `('e', 's')` 和 `('es', 't')` 先后被合并；
2. 测试未见词 `lowest` 最终会被稳定切分为 `['low', 'est</w>']`；
3. 如果将测试词改为完全未见字符构成的词（例如 `"cat"`），输出将无法合并，保留初始单字符并可能包含未登录字符。这正是现代模型进一步采用 Byte-level 方案的原因。
