---
title: Embedding 在 RAG 中如何工作？双塔模型、计算解耦与工程调优
icon: mdi:vector-polyline
tag:
- RAG
- Embedding
- 向量化
description: 解析 RAG 中 Embedding 的双塔工作机制，对比与 Cross-Encoder 的计算开销，讲解指令前缀作用、动态分桶批处理与对比学习微调工程实践。
---

在检索增强生成（RAG）系统中，文本向量化（Embedding）负责将非结构化文本映射为计算机能够计算相似度的稠密向量。在面试和实际工程落地中，向量化不仅是调用模型接口生成数组，更涉及双塔架构的计算解耦、推理批处理调度、指令前缀引导以及领域微调等具体问题。

理解 Embedding 模型在底层的计算形态与时延边界，是构建高召回、低延迟检索链路的基础。

![RAG 向量化双塔架构全景](/images/content-rebuild/6653794c0bb2bb57/cover.png)

---

## 1. 核心口述答案：双塔向量编码的工作原理与生产考量

面试中被问到“RAG 中 Embedding 是如何工作的？生产环境中需要考虑哪些关键问题？”时，可以按以下三点作答：

> **核心口述结构**：
>
> 1. **架构选择**：RAG 召回阶段通常采用**双塔模型（Bi-Encoder）**架构，将查询和文档分别编码为同维度的稠密向量。它的核心优势在于**离线与在线计算解耦**：海量文档在入库阶段提前离线计算向量并写入向量库；在线查询到达时，仅需对单条查询执行一次模型前向推理，随后在向量空间中通过近似最近邻（ANN）索引完成对数级或亚线性复杂度的相似度检索。
> 2. **前缀机制**：现代稠密检索模型（如 BGE、E5）普遍采用**指令前缀（Instruction Prefix）**。查询侧常添加 `query: `，文档侧添加 `passage: `，通过注意力层注入任务先验，平衡问句（短、疑问句、无答案）与文档（长、陈述句、含细节）之间的非对称特征。
> 3. **工程优化**：离线入库关注**吞吐量**，通过按文本长度排序并**动态分桶批处理（Length-grouped Batching）**减少无效 Padding，结合 FP16 推理降低显存消耗；在线检索关注**P99 延迟**，通过前置 L2 归一化将余弦计算简化为向量内积；在垂直专有领域，结合难负样本（Hard Negatives）进行对比学习微调，缓解生僻词语义漂移。

<ReadMoreLock />

---

## 2. Bi-Encoder 结构：离线计算 Document 向量与在线实时 Query 编码

在自然语言处理中，评估文本对匹配度通常有两种架构：**双塔编码器（Bi-Encoder）**与**交叉编码器（Cross-Encoder）**。在 [/interview/rag-interview/embedding-in-rag.md](/interview/rag-interview/embedding-in-rag.md) 中我们讨论了向量基础，而 RAG 召回阶段之所以普遍采用 Bi-Encoder，是由计算复杂度决定的。

![Bi-Encoder 与 Cross-Encoder 架构与计算复杂度对比](/images/content-rebuild/6653794c0bb2bb57/fig-1.png)

### 2.1 召回阶段选用 Bi-Encoder 的原因

- **Cross-Encoder 的开销**：将 Query 与 Document 拼接为单个序列 `[CLS] Query [SEP] Document [SEP]` 送入 Transformer。Query 的每个 Token 与 Document 的每个 Token 在每一层注意力机制中全量交互，匹配精度高。但如果有 $N$ 个候选文档，在线检索必须对这 $N$ 个组合分别执行一次 Transformer 前向推理，计算开销与候选数 $N$ 成正比。如果知识库包含数十万个切片，单次查询的在线计算耗时将无法承受，因此 Cross-Encoder 仅适合在召回后的少量候选集上做重排（详见 [/interview/rag-interview/rerank.md](/interview/rag-interview/rerank.md)）。
- **Bi-Encoder 的解耦**：Query 与 Document 分别通过独立的编码通道（通常共享模型权重）。文档向量的生成与查询完全无关：
  $$\mathbf{v}_d = \text{Pool}(\text{Transformer}(D))$$
  海量切片可以在离线批处理阶段预先计算完毕，存入 [/interview/rag-interview/vector-database.md](/interview/rag-interview/vector-database.md) 并构建索引。在线阶段，系统只对用户 Query 进行一次前向推理：
  $$\mathbf{v}_q = \text{Pool}(\text{Transformer}(Q))$$
  两者之间的语义匹配转化为两个向量的几何相似度计算：$S(q, d) = \langle \mathbf{v}_q, \mathbf{v}_d \rangle$。配合向量数据库的 HNSW 或 IVF 等索引，在线检索开销大幅降低至亚线性水平。

![RAG 向量编码时序流：离线批量与在线实时处理链路](/images/content-rebuild/6653794c0bb2bb57/fig-2.png)

### 2.2 离线建库与在线检索的处理时序

整个向量化系统由两条独立链路组成：

1. **离线处理链路（Ingestion Pipeline）**：长文本通过切分策略（参考 [/interview/rag-interview/rag-chunk-strategies.md](/interview/rag-interview/rag-chunk-strategies.md) 与 [/interview/rag-interview/rag-chunk-size-overlap.md](/interview/rag-interview/rag-chunk-size-overlap.md)）划分为若干文本段落；批处理调度器聚合切片，批量调用 Embedding 模型前向计算；经由池化层（通常为 Mean Pooling）输出固定维度的向量，经过 L2 归一化后写入向量库。
2. **在线查询链路（Serving Pipeline）**：网关接收用户查询；调用 Embedding 推理服务生成 Query 向量；在向量库中执行相似度度量，检索 Top-K 相关文本片段；将命中内容作为上下文装配进大模型提示词。相似度度量相关细节可查阅 [/interview/rag-interview/rag-similarity-algorithms.md](/interview/rag-interview/rag-similarity-algorithms.md)。

下面给出基于官方 `sentence-transformers` 库的实现，演示离线与在线解耦及归一化后的点积计算：

```python
import torch
from sentence_transformers import SentenceTransformer, SimilarityFunction

# 初始化双塔嵌入模型，指定相似度度量为内积
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    similarity_fn_name=SimilarityFunction.DOT_PRODUCT
)

# 模拟离线阶段：处理知识库切片（Passages）
corpus_documents = [
    "Kubernetes Pod 遇到 OOMKilled 错误通常是由于容器内存超出 limit 配额。",
    "Python 垃圾回收机制采用引用计数为主，标记清除与分代回收为辅。",
    "微服务架构下配置中心通过长轮询实现配置的实时动态刷新。"
]
# 离线批量编码并进行 L2 归一化（normalize_embeddings=True 确保模长为 1.0）
doc_embeddings = model.encode(corpus_documents, normalize_embeddings=True, convert_to_tensor=True)

# 模拟在线阶段：处理单条实时请求（Query）
user_query = "容器被系统杀掉报 OOM 是什么原因？"
query_embedding = model.encode([user_query], normalize_embeddings=True, convert_to_tensor=True)

# 检索阶段：矩阵内积运算 (1, 384) x (3, 384)^T -> (1, 3)
similarity_scores = torch.matmul(query_embedding, doc_embeddings.T)

# 取出相似度最高的切片索引
top_idx = torch.argmax(similarity_scores, dim=-1).item()
print(f"命中文档索引: {top_idx}")
print(f"命中文档内容: {corpus_documents[top_idx]}")
print(f"匹配得分: {similarity_scores[0][top_idx]:.4f}")
```

---

## 3. 指令前缀（Instruction Prefix）机制：query: 与 passage: 的作用

在使用近两年的稠密向量模型（如 BGE、E5、Instructor）时，官方文档通常要求：检索提问前拼接 `query: ` 或指令描述，而在被检索文档前拼接 `passage: `。如果在编码时不添加前缀，检索召回率往往会出现明显下滑。

![指令前缀在自注意力机制中的导向作用](/images/content-rebuild/6653794c0bb2bb57/fig-3.png)

### 3.1 对称任务与非对称检索的区别

传统的文本相似度任务（STS, Semantic Textual Similarity）属于**对称任务**：
- 句子 A：“今天天气晴朗。”
- 句子 B：“户外阳光明媚。”
两个句子结构平行、长度相当、语义对称，模型抽取两个句子的主题特征并在向量空间中比较角度即可。

而 RAG 检索属于**非对称任务（Asymmetric Retrieval）**：
- 用户提问（Query）：“MySQL 分页查询越往后越慢怎么解决？”（短句、疑问语气、提出问题）
- 目标文档（Passage）：“在深分页场景中，可通过子查询延迟关联或基于自增主键建立游标查找来规避回表开销……”（长段落、陈述语气、给出方案）

若在此类场景下采用无前缀的对称编码，模型倾向于提取表层主题词，容易把“MySQL 分页查询为什么慢”与“另一个同样讨论 MySQL 慢的用户提问”拉得更近，反而拉远了它与真正提供解法的答案段落之间的距离。

### 3.2 自注意力层中的引导机制

前缀在 Transformer 中的作用机制如下：
1. **前置 Token 引导自注意力**：前缀 Token 处于输入序列最前端。在自注意力计算中，后续正文的每个 Token 都会与前缀 Token 产生注意力权重交互。模型在训练中学会了利用前缀来切换表征目标——面对带 `query: ` 的序列，表征的重点是“这段文字在寻找什么事实”；面对带 `passage: ` 的序列，表征的重点是“这段文字能提供什么事实”。
2. **空间映射对齐**：前缀充当软标签（Soft Indicator），使共享权重的编码器能够将两类不同语式的文本分别投影到提问子空间和事实子空间，并通过对比损失函数拉近匹配对之间的几何距离。

> **工程建议**：入库切片在数据库中持久化时，应保持原貌，不建议把 `passage: ` 前缀直接写入文本存储。前缀应仅在调用 Embedding 模型编码的内存流水线中临时拼接，避免污染后续送给生成大模型的上下文。

---

## 4. 吞吐与延迟平衡：动态分桶、量化加速与显存占用

在生产环境中部署 Embedding 推理服务时，离线建库与在线服务关注的指标并不相同：

- **离线建库**：核心指标是**吞吐量（Throughput，单位时间处理的文档数或 Token 数）**，目标是充分利用显卡算力，缩短入库总耗时。
- **在线检索**：核心指标是**延迟（Latency，尤其是 P99 耗时）**，目标是使单次查询编码稳定在数十毫秒以内，保证交互流畅。

![动态长度分桶与固定填充算力浪费对比](/images/content-rebuild/6653794c0bb2bb57/fig-4.png)

### 4.1 动态批处理与长度分桶（Length-grouped Batching）

在离线批量处理文档时，常见的不当做法是将批次内的所有切片统一填充（Padding）到模型最大序列长度（如 512）。

Transformer 自注意力的矩阵乘法计算量与序列长度呈平方增长。若一个批次中绝大多数文本仅有 100 Token，仅因一条 512 Token 的文本就让整个批次补齐到 512，GPU 将把大量算力浪费在无意义的 `[PAD]` 占位符上。

**按长度分桶打包**是解决这一问题的常规工程手段：
1. 统计当前批待处理文本的实际 Token 长度。
2. 按照长度进行升序或降序排列。
3. 将长度接近的文本划分到同一个批次中（例如 64~128 长度的文本打包为 Batch A，400~512 长度的文本打包为 Batch B）。
4. 每个批次内部仅填充到该批次的实际最大长度。

通过长度分桶，可以明显减少填充 Token 占比，在固定显存预算下有效提升离线吞吐量。

### 4.2 推理显存与 batch_size 估算

在推理模式（Evaluation Mode）下，模型不需要保存优化器状态和反向传播梯度。此时显存占用主要来自两部分：
1. **模型权重显存**：
   $$M_{\text{weights}} = P \times \text{每参数字节数}$$
   以 BGE-large（约 3.3 亿参数）为例：
   - FP32 精度：$330 \times 10^6 \times 4 \approx 1.32\text{ GB}$
   - FP16/BF16 精度：$330 \times 10^6 \times 2 \approx 0.66\text{ GB}$
2. **前向激活值显存（Activation Memory）**：
   与批次大小 $B$、序列最大长度 $L$、注意力头数及层数相关。长序列在自注意力计算时会生成 $(B, \text{heads}, L, L)$ 维度的中间张量。

两项常见的优化手段：
- **FP16 / INT8 推理量化**：使用 ONNX Runtime 或 TensorRT 将模型静态图导出为半精度或整型格式，在精度基本无损的前提下降低显存占用并提升吞吐。
- **Matryoshka 嵌套表示（MRL）截断**：对于在训练中支持 MRL 的模型，可以直接截取输出向量的前缀维度（例如将 1024 维截断至 256 维或 512 维）。这能按比例缩减向量存储体积并降低距离计算开销，支持先利用短向量粗筛、再用长向量精排的阶梯检索。

---

## 5. 领域 Embedding 微调：对比学习与难负样本挖掘

通用预训练 Embedding 模型在面对私有领域（如医疗病例、金融财报、企业错误日志）时，常因专业术语未被充分覆盖而出现表征泛化不足。针对领域语料进行微调是提升特定场景召回精度的有效手段。

![对比学习微调机制：锚点、正样本与难负样本](/images/content-rebuild/6653794c0bb2bb57/fig-5.png)

### 5.1 对比学习损失函数：MultipleNegativesRankingLoss

双塔模型微调中广泛采用基于 **InfoNCE** 的多重负样本排序损失（MultipleNegativesRankingLoss, MNRL）。

给定数据批次中的样本对 $(q_i, p_i^+)$，其中 $q_i$ 是查询，$p_i^+$ 为正相关文档。在一个大小为 $B$ 的批次中，除了正样本 $p_i^+$ 外，其他样本对应的正样本 $p_j^+ (j \neq i)$ 会被作为 $q_i$ 的**批内随机负样本（In-batch Negatives）**。

损失函数表达为：
$$\mathcal{L}_i = -\log \frac{\exp(\text{sim}(q_i, p_i^+) / \tau)}{\sum_{j=1}^B \exp(\text{sim}(q_i, p_j^+) / \tau)}$$
其中 $\tau$ 为温度系数，$\text{sim}(\cdot)$ 为向量余弦相似度。

### 5.2 难负样本（Hard Negatives）的作用

仅靠批内随机负样本训练，模型通常只能学会区分“主题完全不同”的内容（例如区分“数据库”与“烹饪”）。而在实际 RAG 检索中，排序失误往往发生在“主题高度相关，但细节不匹配”的样本之间。

因此，在训练三元组 $(q_i, p_i^+, p_{i, 1}^-)$ 中引入**难负样本**十分关键。获取难负样本的常用途径包括：
1. **BM25 检索挖掘**：利用关键词匹配算法（参考 [/interview/rag-interview/hybrid-retrieval.md](/interview/rag-interview/hybrid-retrieval.md)）检索语料，提取排名靠前但实际不含答案的文本作为负样本。
2. **通用向量模型筛选**：使用已有稠密检索模型召回得分高但标签为不相关的切片。
3. **保留通用测试集评估**：在微调过程中监控领域指标的同时，保留通用语义评测数据进行回归验证，防止过拟合导致通用语义辨识能力下降。

---

## 6. 面试追问与工程排查清单

![线上向量工程排查与防御拓扑](/images/content-rebuild/6653794c0bb2bb57/fig-6.png)

### 深度追问 1：既然余弦相似度需要除以向量模长，为什么许多向量库推荐直接配置内积（Dot Product）？

**答案解析**：
余弦相似度的定义式为：
$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
在检索遍历海量向量时，若每次实时计算模长并进行浮点除法，会产生额外计算开销。

工程上的标准实践是在**向量生成阶段做 L2 归一化**：
$$\mathbf{u} \leftarrow \frac{\mathbf{u}}{\|\mathbf{u}\|_2}, \quad \mathbf{v} \leftarrow \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$
此时两向量模长均为 1，余弦相似度在数值上严格等价于内积：
$$\cos(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v}$$
向量数据库底层只需执行乘加指令（FMA），无需重复计算范数和除法，既保持了方向匹配的特性，又提升了检索吞吐量。

### 深度追问 2：支持长文本窗口（如 8k）的 Embedding 模型，能否直接替代切 chunk？

**答案解析**：
不能直接替代，原因主要在于**信息稀释**和**下游装配粒度**：
1. **平均池化导致信息稀释**：即使模型能够接收 8k Token，在经过 Mean Pooling 压缩成单根定长向量（如 1024 维）时，整篇长文的多种主题和细节信息会被均值化，针对微小技术细节的特定查询在向量空间中的区分度会明显减弱。
2. **上下文利用效率**：RAG 召回的文本片段最终要拼入大模型上下文（参考 [/interview/rag-interview/rag-prompt-design.md](/interview/rag-interview/rag-prompt-design.md)）。若单次命中返回整篇 8k 文本，不仅占用大模型的输入窗口和计算开销，还可能引发长文本注意力分散的问题。合理的段落切分依然是保证检索精准度和控制成本的重要手段。

### 常见工程排查清单

| 常见问题行为 | 表现与影响 | 应对方案 |
| :--- | :--- | :--- |
| **查询端遗漏前缀指令** | 在需要前缀的模型（如 BGE/E5）上，检索召回率产生明显滑落。 | 在网关或客户端封装固定逻辑，区分 Query 与 Passage 格式自动拼接。 |
| **在线与离线归一化不一致** | 离线做了 L2 归一化而在线未做，导致点积打分不再等价于余弦相似度，破坏排序。 | 在客户端调用层统一开启 `normalize_embeddings=True`，保证向量模长归一。 |
| **固定最大长度填充批处理** | 离线入库时大量显存消耗在 Padding 上，导致吞吐量偏低。 | 采用按文本长度分桶（Length-grouped Batching），相似长度分同一批次处理。 |
| **微调缺乏难负样本** | 模型仅能区分宽泛大类，在相近概念的细分文档上容易选错。 | 结合 BM25 或初始向量模型召回排名高但不相关的切片，作为难负样本参与训练。 |
