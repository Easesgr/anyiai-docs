---
title: BM25 为什么还能在 RAG 中发挥作用？词频饱和度、长度归一化与专名检索机制
icon: mdi:database
tag:
- RAG
- 大模型100讲
- 检索增强
description: 解析 BM25 算法在 RAG 中的互补价值，掌握词频饱和度与长度归一化机制，通过 Python 代码实现排障错误码精准检索。
---

在构建检索增强生成（RAG）系统时，许多开发者倾向于完全依赖基于神经网络的稠密向量检索（Dense Retrieval）。然而，一旦业务进入排障日志、接口配置、专有名词、设备型号等领域，纯向量检索经常把相似但错误的切片排在前面。

本讲是进阶课程的第一篇，我们将完成以下学习目标：
1. 解释为什么连续向量检索在精确专名、错误码等离散符号匹配上存在召回局限；
2. 掌握 BM25 算法的三个核心数学因子：逆文档频率（IDF）、词频饱和度（$k_1$）与长度归一化（$b$）；
3. 通过一个网关排障场景的贯穿实例与纯 Python 代码，可验证地对比 BM25 与向量检索的行为差异，并完成调参练习。

![BM25 算法机制与 RAG 精确检索概览](/images/content-rebuild/fa98fbbd7df6c04c/cover.png)

在探讨具体检索公式之前，关于检索组件在大模型系统中的标准上下游定位，可以参考基础篇中的 [RAG 是什么？为什么大模型需要外部知识库？](/llm-100/rag/what-is-rag.md)。

## 贯穿场景：排障日志中的错误码混淆

为了直观理解检索问题，我们以微服务网关排障作为贯穿全篇的例子。假设知识库中存储了以下四条运维手册切片：

- **文档 0**：`System error log: ERR_AUTH_EXPIRED_401 happened during user token validation on gateway`（Token 过期）
- **文档 1**：`System error log: ERR_AUTH_FORBIDDEN_403 happened during user permission validation on gateway`（权限拒绝）
- **文档 2**：`System error log: ERR_AUTH_TIMEOUT_504 happened during remote auth center communication on gateway`（鉴权超时）
- **文档 3**：`Network failure log: ERR_CONNECTION_REFUSED happened during socket connection on db pool`（数据库连接异常）

此时，运维工程师输入具体排障查询：`ERR_AUTH_EXPIRED_401`。

工程师的诉求非常精确：只需要包含该错误码的文档 0，绝不需要讨论 403 或 504 的文档。

### 稠密语义向量的边界

稠密向量模型（如常见的中英文双塔 Embedding 模型）的工作方式，是将输入文本通过注意力机制和池化层，映射到一个固定维度的连续向量空间（例如 768 维或 1024 维）。连续向量擅长捕捉语义层面的近义表达，比如能将“手机”与“移动电话”映射到相近的位置。

但在面对特定错误码时，这种连续语义泛化反而可能成为干扰：
- 文档 0、1、2 在自然语言上下文上高度重合，都描述了“网关身份鉴权过程中发生的系统错误日志”。
- 模型的自注意力层在均值或注意力池化后，整段文本的宏观语义占据了主导地位。错误码中的微小字符变化（`EXPIRED_401` 与 `FORBIDDEN_403`）在连续空间中仅仅表现为微小扰动。
- 结果是三篇文档与查询的余弦相似度可能都在 0.84 至 0.88 之间，差距极小。如果用户在查询中顺手加入了自然语言描述，例如“怎么排查网关认证权限相关的 ERR_AUTH_EXPIRED_401 报错”，受到“权限”一词的拉动，向量检索甚至可能把包含 403 权限拒绝的文档 1 排在最前面。

![语义空间近邻泛化与离散符号精确匹配的召回差异](/images/content-rebuild/fa98fbbd7df6c04c/fig-1.png)

BM25（Best Matching 25）属于基于离散词项匹配的统计检索算法。它直接依赖倒排索引，寻找包含特定字符 Token 的文档。在上述错误码场景下，只要未登录词被合理分词，BM25 会将命中 `ERR_AUTH_EXPIRED_401` 的文档赋予高分，而完全未包含该 Token 的文档得分为 0，从而提供清晰的判别边界。

<ReadMoreLock />

## BM25 评分公式：三要素拆解

BM25 源于概率检索框架，用于评估查询 $Q$ 与候选文档 $D$ 之间的匹配相关性。给定查询词集合 $Q = \{q_1, q_2, \dots, q_m\}$ 与文档 $D$，BM25 打分公式为：

$$\text{Score}(D, Q) = \sum_{i=1}^{m} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

公式中的变量定义如下：
- $q_i$：查询拆分后的第 $i$ 个词项（Term）；
- $f(q_i, D)$：词项 $q_i$ 在当前文档 $D$ 中的词频（Term Frequency, TF）；
- $|D|$：文档 $D$ 的词项总数（文档长度）；
- $\text{avgdl}$：语料库中所有文档的平均词项数（Average Document Length）；
- $k_1$：词频饱和度调节参数（正实数，工程中通常取 $1.2 \sim 2.0$）；
- $b$：文档长度归一化惩罚参数（实数，取值范围在 $[0, 1]$，通常取 $0.75$）；
- $\text{IDF}(q_i)$：逆文档频率，度量词项在整个语料库中的信息量。

![BM25 评分公式三要素组合机制](/images/content-rebuild/fa98fbbd7df6c04c/fig-2.png)

### 1. 逆文档频率（IDF）：度量词项的信息判别力

在搜索引擎和全文检索库（如 Lucene）中，为了防止文档频率较高时对数项出现负数，IDF 通常采用平滑计算公式：

$$\text{IDF}(q_i) = \ln \left( 1 + \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} \right)$$

其中 $N$ 是语料库文档总数，$n(q_i)$ 是包含词项 $q_i$ 的文档数量：
- 如果一个词在全部文档中频繁出现（例如“系统”、“说明”），$n(q_i) \approx N$，分式接近 0，IDF 趋近于 0，该词对最终打分贡献极弱；
- 如果一个词非常稀有（例如特定错误码 `ERR_AUTH_EXPIRED_401`），只有 1 篇文档包含它，$n(q_i)=1$，IDF 值就会显著升高，成为主导最终评分的关键项。

### 2. 词频饱和度（$k_1$）：控制词频增长的边际收益

早期传统的 TF-IDF 算法中，词频贡献大多是线性的。线性增长存在一个缺陷：如果一篇文章恶意重复堆叠某个关键词 100 次，其得分就会是出现 1 次的 100 倍，从而扭曲相关性。

BM25 通过有理分式将词频贡献限制在有界区间内。为了看清其变化规律，我们假设文档长度恰好等于平均长度（即 $|D| = \text{avgdl}$），此时长度惩罚项为 1，词频部分简化为：

$$S_{\text{TF}}(f) = \frac{f \cdot (k_1 + 1)}{f + k_1}$$

当词频 $f \to \infty$ 时，取极限：

$$\lim_{f \to \infty} \frac{f \cdot (k_1 + 1)}{f + k_1} = k_1 + 1$$

这说明单词项的得分增益上限被严格截断在 $k_1 + 1$ 倍的 IDF 权重内。

![线性 TF 与 BM25 饱和曲线在不同 k1 取值下的对比](/images/content-rebuild/fa98fbbd7df6c04c/fig-3.png)

参数 $k_1$ 的作用，是调节词频增长逼近饱和上限的速率：
- 取 $k_1 = 1.2$ 时：
  - 当 $f = 1$ 时，$S_{\text{TF}}(1) = \frac{1 \times 2.2}{1 + 1.2} = 1.0000$；
  - 当 $f = 2$ 时，$S_{\text{TF}}(2) = \frac{2 \times 2.2}{2 + 1.2} = 1.3750$；
  - 当 $f = 5$ 时，$S_{\text{TF}}(5) = \frac{5 \times 2.2}{5 + 1.2} \approx 1.7742$；
  - 当 $f = 20$ 时，$S_{\text{TF}}(20) = \frac{20 \times 2.2}{20 + 1.2} \approx 2.0755$；
  - 当 $f = 100$ 时，$S_{\text{TF}}(100) = \frac{100 \times 2.2}{100 + 1.2} \approx 2.1739$。

词频从 1 次增长到 2 次，增益提升了 37.5%；但从 20 次增长到 100 次（暴增 5 倍），增益仅微调了 4.7%。这种明显的边际递减效应，使得关键词重复堆砌失去了不合理的竞争优势。

### 3. 长度归一化（$b$）：平衡长文档与短文档的篇幅差异

长文档在统计词频时天然占便宜：一篇一万字的长手册，即便只是随笔提到一次某技术词，其命中概率也高于仅有一百字的问题诊断卡片。长文档可能来自两种情况：
1. **冗长型**：作者语言啰嗦，翻来覆去讲同一件事，实际信息密度低；
2. **多主题型**：文档本身覆盖了多个子业务领域，信息量真实充实。

BM25 引入分母调节项来衡量长度影响：

$$K = 1 - b + b \cdot \frac{|D|}{\text{avgdl}}$$

![文档长度归一化惩罚机制与分母调节项运作机理](/images/content-rebuild/fa98fbbd7df6c04c/fig-4.png)

参数 $b$ 控制对长文档的惩罚力度：
- **当 $b = 1$ 时**：$K = \frac{|D|}{\text{avgdl}}$，系统采用严格惩罚。若文档长度是平均长度的 2 倍，词频项的分母就会大幅扩大，相当于假定长文档只是纯粹的冗长累赘；
- **当 $b = 0$ 时**：$K = 1$，完全关闭长度惩罚，直接依据词频与饱和度打分；
- **推荐默认值 $b = 0.75$**：在冗长惩罚与多主题容忍之间取得折中。

在 RAG 切块策略中，参数 $b$ 的调校与切块方式紧密相关：
- 如果采用固定大小的滑动窗口分块（如每块切固定 512 个 Token），所有 Chunk 长度十分接近（$|D| \approx \text{avgdl}$），此时参数 $b$ 的影响较小；
- 如果按照 Markdown 标题、函数体或自然段落切块，切片长度分布差异很大（有的段落仅有 50 字，有的章节包含 2000 字），保留并调优 $b$（通常保持在 $0.75 \sim 0.85$）可以防止超大章节由于偶然多命中一两个词而挤占排序前排。关于切分策略的具体分析，可参阅 [文档切分 Chunk 应该怎么做？](/llm-100/rag/chunk-strategy.md)。

## 中文分词对 BM25 召回的约束

BM25 算法本身并不关心语义，只依赖离散词项的统计。英文文本天然以空格分隔单词，但中文句子是连续字符流，必须先通过分词器（Tokenizer）切分成词。因此，中文 BM25 的检索质量高度受制于分词阶段。

在工程落地中，分词往往面临两类常见问题：

### 1. 粒度切分不匹配
如果索引库构建时采用细粒度分词（例如将“高并发网关”切分为 `["高并发", "网关"]`），而检索请求使用粗粒度分词（切分为单个词项 `["高并发网关"]`），倒排索引中若没有收录复合词，就会导致漏召。

### 2. 未登录词与连字符截断
工程运维中的错误码或配置项通常包含连字符或下划线，例如 `BGP_EVPN_ROUTE_FAIL_0x89`。如果分词规则将下划线作为停用符号强行剔除，将其拆成 `["BGP", "EVPN", "ROUTE", "FAIL", "0x89"]`，检索时如果只输入单个片段，就容易引入大量无关的噪音匹配。

![中文分词器粒度切分与未登录词检索分叉路径](/images/content-rebuild/fa98fbbd7df6c04c/fig-5.png)

在面向特定专业场景（如医药配方、运维排障、法律条文）时，需要补充领域专业词典，并在分词管线中保护数字、英文连字符与特定符号编码。

## 代码实战：构建 BM25 检索器并验证排障场景

下面我们通过一个纯 Python 实现的 BM25 类，跑通贯穿例子中的网关排障场景，验证其打分行为。

![BM25 与向量相似度在精确错误码场景的检索对照实验](/images/content-rebuild/fa98fbbd7df6c04c/fig-6.png)

```python
import math
from collections import Counter

class SimpleBM25:
    def __init__(self, corpus, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        # 示例中使用空格切词；中文环境建议替换为 jieba 等分词库
        self.doc_tokens = [doc.split() for doc in corpus]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / self.corpus_size
        self.doc_freqs = []
        self.idf = {}
        self._build_index()

    def _build_index(self):
        df = Counter()
        for tokens in self.doc_tokens:
            frequencies = Counter(tokens)
            self.doc_freqs.append(frequencies)
            for term in frequencies.keys():
                df[term] += 1

        # 采用 Lucene 平滑 IDF 公式
        for term, freq in df.items():
            self.idf[term] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def retrieve(self, query):
        query_terms = query.split()
        results = []
        for idx, freqs in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_lens[idx]
            # 计算长度归一化分母项
            len_norm = 1.0 - self.b + self.b * (doc_len / self.avgdl)

            for term in query_terms:
                if term not in freqs:
                    continue
                tf = freqs[term]
                idf = self.idf.get(term, 0.0)
                # BM25 词频饱和与长度惩罚
                tf_component = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
                score += idf * tf_component
            results.append((idx, score))
        # 按得分从高到低排序
        return sorted(results, key=lambda x: x[1], reverse=True)


# 语料库：四条排障文档
corpus = [
    "System error log: ERR_AUTH_EXPIRED_401 happened during user token validation on gateway",
    "System error log: ERR_AUTH_FORBIDDEN_403 happened during user permission validation on gateway",
    "System error log: ERR_AUTH_TIMEOUT_504 happened during remote auth center communication on gateway",
    "Network failure log: ERR_CONNECTION_REFUSED happened during socket connection on db pool"
]

bm25 = SimpleBM25(corpus, k1=1.2, b=0.75)
query = "ERR_AUTH_EXPIRED_401"
scores = bm25.retrieve(query)

print(f"查询项: '{query}'")
for idx, score in scores:
    print(f"文档 {idx} 得分: {score:.4f} | 内容: {corpus[idx]}")
```

### 运算结果与机制分析

运行上述代码，在控制台可以得到如下可验证结果：

```text
查询项: 'ERR_AUTH_EXPIRED_401'
文档 0 得分: 1.2150 | 内容: System error log: ERR_AUTH_EXPIRED_401 happened during user token validation on gateway
文档 1 得分: 0.0000 | 内容: System error log: ERR_AUTH_FORBIDDEN_403 happened during user permission validation on gateway
文档 2 得分: 0.0000 | 内容: System error log: ERR_AUTH_TIMEOUT_504 happened during remote auth center communication on gateway
文档 3 得分: 0.0000 | 内容: Network failure log: ERR_CONNECTION_REFUSED happened during socket connection on db pool
```

我们来核对具体计算过程：
1. 语料库总文档数 $N = 4$。包含 `ERR_AUTH_EXPIRED_401` 的文档数 $n = 1$；
2. 计算该词项的 IDF：
   $$\text{IDF} = \ln\left(1 + \frac{4 - 1 + 0.5}{1 + 0.5}\right) = \ln\left(1 + \frac{3.5}{1.5}\right) = \ln\left(\frac{10}{3}\right) \approx 1.2040$$
3. 四篇文档长度分别为 11、11、12、11，平均长度 $\text{avgdl} = 45 / 4 = 11.25$；
4. 文档 0 包含 11 个词，$|D| / \text{avgdl} = 11 / 11.25 \approx 0.9778$；
5. 分母调节项：$K = 1 - 0.75 + 0.75 \times 0.9778 = 0.9833$；
6. 词频 $f = 1$，饱和项计算：
   $$\frac{1 \times (1.2 + 1)}{1 + 1.2 \times 0.9833} = \frac{2.2}{2.1800} \approx 1.0092$$
7. 最终得分：代码以浮点数连续计算得 $1.2040 \times 1.009174 \approx 1.2150$（若直接按保留四位小数的中间截断值相乘为 $1.2040 \times 1.0092 \approx 1.2151$）。

其余三篇文档由于完全不包含该词项，得分为精确的 `0.0000`。在精确匹配上，BM25 展现了极高的信噪比。

而在纯向量检索中，由于文档 0、1、2 大部分词汇重叠，通用向量模型给三者计算出的余弦相似度可能普遍在 0.85 左右，缺乏清晰的排他性。这就是 BM25 在工业生产级 RAG 检索链路中始终不可或缺的技术原因。

关于向量模型的选择与表现边界，可以进一步参考 [Embedding 模型如何选择？](/llm-100/rag/embedding-selection.md)；关于存储与检索的系统落地，详见 [向量数据库在 RAG 中起什么作用？](/llm-100/rag/vector-db-in-rag.md)。

## 课后动手练习

请使用上面的 Python 代码完成以下两个实验，以加深对参数影响的体会：

1. **调整长度惩罚参数 $b$**：
   向语料库末尾追加一篇极长的文档 4：`"System error log: ERR_AUTH_EXPIRED_401 " + "other irrelevant info " * 100`。保持 $k_1 = 1.2$，分别测试 $b = 0.0$ 与 $b = 1.0$ 时，文档 4 与文档 0 的得分比例变化，观察长度惩罚何时生效。
2. **测试未登录词分词影响**：
   将查询改为带下划线的小写短语 `err_auth_expired_401`。如果直接使用 `.split()` 分词，大小写不匹配会导致得分为 0。尝试在检索前加入文本清洗管线（`.lower()`）以及正则表达式切分，体会分词预处理对离散检索的核心影响。

当掌握了 BM25 独立工作的机制后，下一步我们将探讨如何将 BM25 关键词检索与稠密向量语义检索融合为一条混合检索管线，详细方案将在 [相似度检索、关键词检索、混合检索有什么区别？](/llm-100/rag/retrieval-methods.md) 中展开讲解。
