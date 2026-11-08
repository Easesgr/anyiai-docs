---
title: Embedding 模型如何选择？从向量表示、维度成本到业务评测
icon: mdi:database
tag:
- RAG
- 大模型100讲
- 检索增强
description: 系统梳理文本嵌入模型的池化机制、非对称检索指令、维度成本折衷、套娃表示学习与业务评测方案，指导工程选型。
---

在检索增强生成（RAG）系统中，文本切分确定了知识单元的物理边界（参考 [文档切分 Chunk 应该怎么做？](chunk-strategy.md)），而文本嵌入（Text Embedding）模型则决定了这些文本块在连续向量空间中的相对位置。如果嵌入模型无法有效区分业务查询与参考语料的语义关联，后续送入 [向量数据库](vector-db-in-rag.md) 的近邻搜索就会从底层召回无关或易混淆的片段。

考虑一个典型的银行客服问答场景：
- **用户提问（Query）**：“审核通过后，怎么把授信额度提现到借记卡？”
- **目标段落（正样本）**：“审核通过后，登录手机银行进入首页，点击‘额度提现’并输入提现金额与短信验证码，资金通常会在几个工作日内转入绑定的借记卡。”
- **干扰段落（难负样本）**：“授信额度是平台根据信用资质评定的最高可借款限额，授信成功并不代表资金已经自动转入借记卡。”

目标段落和干扰段落包含大量重合词汇（如“审核通过”、“授信额度”、“借记卡”）。如果嵌入模型仅依据字面词频或通用百科浅层特征进行映射，干扰段落的相似度得分可能反超目标段落，导致检索失败。选型嵌入模型不仅是看公开排行榜的高低，还要理清其池化机制、非对称指令、维度成本以及在业务测试集上的实际辨识边界。

![文本嵌入模型特征空间与向量映射封面图](/images/content-rebuild/f85b6e98394de826/cover.png)

---

## 序列隐层如何变成单个向量：Pooling 策略与归一化

静态词向量（如 Word2Vec）为每个单词分配固定向量，无法动态体现多义词在不同上下文中的变化。现代 Dense Embedding 模型多基于 Transformer 架构（双向编码器如 BERT/RoBERTa，或因果解码器如微调后的 Qwen/Mistral），将包含 $L$ 个 Token 的文本转换为一组隐层向量：

$$\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_L] \in \mathbb{R}^{L \times d}$$

其中 $d$ 为隐层维度。由于不同文本的长度 $L$ 各不相同，但在做相似度计算（如点积或余弦相似度）时需要统一长度，必须通过“池化（Pooling）”将矩阵 $\mathbf{H}$ 压缩为固定长度的句向量 $\mathbf{e} \in \mathbb{R}^d$。

![三种主流 Pooling 策略的机制与张量流转分解](/images/content-rebuild/f85b6e98394de826/fig-1.png)

### 1. CLS Pooling
在输入序列最前部放置特殊标记 `[CLS]`，其对应的隐层向量为 $\mathbf{h}_1$。如果模型在训练阶段（如监督对比学习）显式将优化目标绑定在 `[CLS]` 上，即可直接提取首位置向量：

$$\mathbf{e}_{cls} = \mathbf{h}_1$$

需要注意：如果模型预训练仅使用了掩码语言模型任务（MLM）且没有经过句子级对比微调，直接取 `[CLS]` 往往偏向局部语言建模特征，其语义检索效果通常不如均值池化。

### 2. Mean Pooling（均值池化）
对输入中所有有效 Token 的隐层向量计算算术平均值。设 $m_i \in \{0, 1\}$ 为 Attention Mask（真实文本为 1，补齐 Padding 为 0）：

$$\mathbf{e}_{mean} = \frac{\sum_{i=1}^L m_i \mathbf{h}_i}{\sum_{i=1}^L m_i}$$

Mean Pooling 综合了经过多层自注意力机制交互后的全局状态，受单个异常 Token 的影响较小，是 Sentence-Transformers 体系中多数 BERT 类双向编码器模型的默认方案。

### 3. Last Token Pooling（尾 Token 池化）
在使用因果语言模型（Causal Decoder-only）微调生成的嵌入模型中，由于下三角注意力掩码的存在，前部 Token 无法看到后续 Token，只有序列末尾最后一个有效 Token 能够汇聚整个输入序列的上下文状态：

$$\mathbf{e}_{last} = \mathbf{h}_{\text{last\_valid\_idx}}$$

如果在此类模型上误用 Mean Pooling，前部视野受限的 Token 向量会被赋予相同权重，导致整句语义表达受损。

### L2 归一化与余弦相似度
句向量输出时通常会进行 L2 归一化，即将其长度缩放为 1：

$$\hat{\mathbf{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2}$$

对两个已经完成 L2 归一化的向量 $\hat{\mathbf{u}}$ 与 $\hat{\mathbf{v}}$，余弦相似度直接等同于向量点积：

$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \hat{\mathbf{u}} \cdot \hat{\mathbf{v}}$$

归一化后在线检索只需做点积相乘，不必每次重复计算模长除法，降低了向量检索阶段的浮点开销。

---

## 对称检索与非对称检索：指令前缀的作用

输入文本对的长度与句式结构直接影响两端文本在向量空间中的相对距离。

![对称检索与非对称检索特征空间与指令前缀作用对比](/images/content-rebuild/f85b6e98394de826/fig-2.png)

### 对称检索（Symmetric Retrieval）
Query 与 Document 在长度、语法结构与信息密度上基本对等。例如：
- 客户服务问答库中的相似问去重（“修改登录密码在哪里”与“怎样重置账号密码”）。
- 语义相似度判别（STS）与文章去重。

两端句子角色一致，通常共享同一套编码方式，无需为输入做额外角色区分。

### 非对称检索（Asymmetric Retrieval）
这是问答系统与标准 RAG 的常见形态，Query 与 Document 存在明显结构差异：
- **Query**：通常较短（约 5~30 个 Token），多为疑问句或关键词，语义聚焦但表达精简（如“退款一般几天能到账？”）。
- **Document**：通常较长（约 200~800 个 Token），为说明性陈述句或结构化条款，包含流程细节与边界条件。

如果直接将两者送入未经特殊引导的同一编码器，模型容易因文本长度和表层语法差异形成聚集偏置：短问句聚在一起，长说明文档聚在一起，导致问句与能回答它的文档之间向量距离过大。

<ReadMoreLock />

### 指令前缀（Instruction Prefix）的实际作用
为了在单塔共享参数的前提下让模型区分不同角色，E5、BGE 等模型引入了显式前缀指令：

- **文档侧（Passage）**：输入格式通常为 `passage: {doc_text}` 或保留原始文本。
- **查询侧（Query）**：加入提示词前缀，例如：
  - E5 模型规范：`query: {query_text}`
  - BGE 模型规范：`为这个句子生成表示以用于检索相关文章：{query_text}`

自注意力机制会让前缀指令中的 Token 参与整个查询序列的注意力权重计算，从而促使编码器在空间中把短问句的表征偏向相关答案段落所在的分布区。

**选型注意**：前缀规则必须严格遵循模型作者的训练规范。如果模型本身没有基于前缀训练，强行添加前缀反而会引入文本噪声；反之，若选用了需要前缀的模型而在查询时遗漏了前缀，召回率往往会出现明显下滑。

---

## 向量维度的工程成本：存储、索引与计算延迟

并非维度越高的模型就越适合上线。向量维度 $d$ 的增加会成比例带来存储带宽、内存占用与计算量的上升。

![向量维度对显存占用与计算开销的量化影响分析](/images/content-rebuild/f85b6e98394de826/fig-3.png)

### 1. 裸向量内存占用量化
假设企业知识库切分出 $N = 1,000,000$（一百万）个 Chunk，以标准 32 位单精度浮点数（Float32，每个数字占 4 字节）存储，裸向量占用的内存为：

$$\text{Memory}_{raw} = N \times d \times 4 \text{ 字节}$$

不同维度在不同数据规模下的裸内存开销如下（按 $1\text{ GiB} = 1024^3\text{ 字节}$ 计算）：

| 向量维度 $d$ | 典型模型代表 | 100 万条裸向量体积 | 1000 万条裸向量体积 |
| :--- | :--- | :--- | :--- |
| 384 | all-MiniLM-L6-v2 | 1.43 GiB | 14.31 GiB |
| 768 | BGE-base | 2.86 GiB | 28.61 GiB |
| 1024 | BGE-large, BGE-M3 | 3.81 GiB | 38.15 GiB |
| 1536 | text-embedding-3-small | 5.72 GiB | 57.22 GiB |
| 3072 | text-embedding-3-large | 11.44 GiB | 114.44 GiB |
| 4096 | 基于开源 7B 模型适配的嵌入 | 15.26 GiB | 152.59 GiB |

### 2. 近似最近邻（ANN）图索引开销
在工程实践中，为了支持低延迟检索，通常会使用 HNSW 图索引（详见 [向量数据库在 RAG 中起什么作用？](vector-db-in-rag.md)）。

HNSW 不仅要在内存中驻留上述原始向量，还要为每个节点维护多层的邻居连边指针列表。在典型配置（如邻居数 $M=16 \sim 32$）下，HNSW 结构本身的额外开销常为原始向量体积的 1.0 到 2.0 倍。若选型 3072 维向量存储一千万条数据，完整的 HNSW 内存索引往往需要消耗超过 200 GiB 的高速物理内存。

### 3. 计算延迟与内存带宽限制
向量点积包含 $d$ 次乘法与 $d-1$ 次加法，浮点计算量约为 $2d$ 次运算。大规模检索本征上对内存带宽高度敏感。当维度从 768 扩大到 3072 时，每次距离比较所需要从总线加载到 CPU/GPU 缓存的数据量增加到了 4 倍。高并发查询下，维度的线性增长会显著推高服务器的 P99 延迟。

---

## Matryoshka 俄罗斯套娃表示学习（MRL）与两阶段分层检索

传统嵌入模型的向量维度在训练后无法直接任意截断。如果直接把 1024 维切断为前 256 维，向量原本的距离关系会被打破，检索性能通常大幅衰减。

Matryoshka Representation Learning（MRL）改善了这一限制。

![Matryoshka 套娃表示多级损失与分级检索机制](/images/content-rebuild/f85b6e98394de826/fig-4.png)

### 训练思路：嵌套多尺度损失
MRL 在训练对比损失时，预先设定多个截断长度（如 $\{64, 128, 256, 512, 1024\}$）。模型产出的向量在不同前缀切片后，均各自做 L2 归一化并计算对比损失，最终把各截断维度的损失求和回传：

$$\mathcal{L}_{MRL} = \sum_{m \in \mathcal{M}} \lambda_m \mathcal{L}_{contrastive}\left( \hat{\mathbf{e}}_{1:m} \right)$$

通过这种多尺度约束，模型被引导将最重要的核心区分特征集中到靠前的维度（如前 256 维），靠后的维度则用于容纳更为细微的残差细节。

### 两阶段检索流水线设计
借助 MRL 特性，系统可以用单套模型实现兼顾内存与精度的两阶段检索：

1. **入库建索引**：将向量截取前 256 维建立全内存 HNSW 图索引，将完整 1024 维向量存入磁盘或量化二级存储。
2. **查询阶段**：
   - **第一阶段（粗筛）**：Query 向量同样截取前 256 维，在内存索引中快速检索出 Top-100 候选 Chunk。
   - **第二阶段（精细打分）**：对这 100 个候选块读取完整的 1024 维向量，做局部的精确点积重排，选出 Top-5 进入后续链路（也可接入 [Rerank 重排序为什么重要？](rerank-importance.md) 进一步优化）。

这种分阶段方式大幅减轻了图索引对内存的占用压力，而在大量通用任务上的命中率损失往往能控制在极小范围内。

---

## MTEB 基准的参考价值与局限

HuggingFace 维护的 MTEB（Massive Text Embedding Benchmark）是评估嵌入模型表现的主流公开基准。在参考该榜单选型时，需注意以下区分：

![MTEB 综合指标与垂直领域业务数据偏差对比分析](/images/content-rebuild/f85b6e98394de826/fig-5.png)

### 1. 任务子榜单分化
MTEB 综合得分融合了分类（Classification）、聚类（Clustering）、语义文本相似度（STS）、检索（Retrieval）、重排（Reranking）等多种任务。某些模型在分类或 STS 上得分很高，但在非对称检索任务上的表现相对平庸。**为 RAG 选型时，应重点筛选 `Retrieval` 或 `Retrieval (zh)` 子榜单**，不要直接依据全局综合总分做决策。

### 2. 评测集与真实业务语料的分布差异
MTEB 的检索基准以维基百科、通用网页搜索问答和新闻语料为主：
- **文本形态差异**：基准中文档长度通常在 100~300 个 Token 之间，且行文工整。如果业务知识库切分段落较长，或混杂了 Markdown 表格、日志堆栈与树状层级，通用榜单靠前的模型表现可能会出现下滑。
- **专业术语稀疏**：医药化学、法律合规、工业制造与金融信贷等专业词汇在通用语料中出现频率低。通用模型可能无法准确区分字面相似但含义对立的行业术语。

---

## 动手验证：构建业务评测集与对比实验

选型嵌入模型最可靠的方法是构建一个包含正样本与“难负样本（Hard Negatives）”的场景基准集，并在相同硬件环境下对比候选模型的区分余量（Margin）。

![基于业务术语区分度的嵌入对比测试流水线](/images/content-rebuild/f85b6e98394de826/fig-6.png)

### 核心考量：相似度区分边界（Margin）
如果模型给正样本打 0.82 分，而给看似相似实则不符的难负样本打 0.80 分，二者的区分边界（Margin $= 0.82 - 0.80 = 0.02$）过窄，在千万级语料检索时极易被噪声淹没。一个更适合业务的模型应当拉大正负样本的得分差距。

下面是使用官方 `sentence_transformers` 库执行选型对比的完整示例：

```python
import numpy as np
from sentence_transformers import SentenceTransformer, SimilarityFunction

# 候选模型列表：按实际机器配置替换为本地路径或 HuggingFace 模型名
CANDIDATE_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    # 生产对比可增加: "BAAI/bge-base-zh-v1.5", "intfloat/multilingual-e5-base"
]

# 构建包含难负样本的业务测试样例集
EVAL_CASES = [
    {
        "case_name": "信贷业务：授信额度提现 vs 额度定义",
        "query": "审核通过后，怎么把授信额度提现到借记卡？",
        "positive": "审核通过后，登录手机银行进入首页，点击‘额度提现’并输入提现金额与短信验证码，资金通常会在几个工作日内转入绑定的借记卡。",
        "hard_negative": "授信额度是平台根据信用资质评定的最高可借款限额，授信成功并不代表资金已经自动转入借记卡。",
    }
]

for model_name in CANDIDATE_MODELS:
    print(f"\n====== 评估模型: {model_name} ======")
    # 初始化模型并指定余弦相似度
    model = SentenceTransformer(model_name, similarity_fn_name=SimilarityFunction.COSINE)
    
    for case in EVAL_CASES:
        query = case["query"]
        pos_doc = case["positive"]
        neg_doc = case["hard_negative"]
        
        # 编码向量并进行 L2 归一化（注意：需按各模型规范配置 query 前缀）
        embeddings = model.encode([query, pos_doc, neg_doc], normalize_embeddings=True)
        q_emb, pos_emb, neg_emb = embeddings[0], embeddings[1], embeddings[2]
        
        # 归一化后向量点积等价于余弦相似度
        sim_pos = float(np.dot(q_emb, pos_emb))
        sim_neg = float(np.dot(q_emb, neg_emb))
        margin = sim_pos - sim_neg
        
        print(f"测试用例: {case['case_name']}")
        print(f"  - 正样本相似度 (Sim+): {sim_pos:.4f}")
        print(f"  - 难负样本相似度 (Sim-): {sim_neg:.4f}")
        print(f"  - 区分余量 (Margin): {margin:.4f}")
        
        if margin > 0.15:
            print("  -> 评价: 正向区分度明确")
        elif margin > 0:
            print("  -> 评价: 具备正向偏置，但分值间隔偏小，易受语料噪声干扰")
        else:
            print("  -> 评价: 发生排序倒置，难负样本得分高于正样本，不可用")
```

### 检索基准指标评估
当构建了包含数十至数百条带标准答案的检索评测集（每条 Query 关联其标注的 Chunk ID）时，可计算下列检索指标进行多维度比较：

1. **Recall@K（召回率）**：检索返回的前 $K$ 个候选中命中的相关 Chunk 数量，占该 Query 标注的相关 Chunk 总数的比例（若每条 Query 仅对应单个正样本，则等价于前 $K$ 项命中目标正样本的查询比例）。
2. **MRR@K（平均倒数排名）**：关注首个相关文档出现在返回列表中的位次（若首个正确答案排第 1 名得分为 1，排第 2 名得分为 0.5）。
3. **NDCG@K（归一化折损累计增益）**：在测试用例文档具备多级相关度评分时，衡量高相关度文档排在前列的程度。

---

## 关联阅读与选型路径

选定 Dense Embedding 模型后，检索系统的构建还涉及以下环节：
- 向量生成后如何组织索引与查询？请阅读 [向量数据库在 RAG 中起什么作用？](vector-db-in-rag.md)。
- 对精确数字、专业工单编码容易漏检时如何弥补？请阅读 [BM25 为什么还能在 RAG 中发挥作用？](bm25-in-rag.md)。
- 结合词面检索与向量检索的方案，请参考 [相似度检索、关键词检索、混合检索有什么区别？](retrieval-methods.md)。
- 仅靠 Embedding 双塔初筛难以捕捉复杂长文交互，请阅读 [Rerank 重排序为什么重要？](rerank-importance.md)。
