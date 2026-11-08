---
title: Rerank 重排序为什么重要？Cross-Encoder 机制与两阶段检索设计
icon: mdi:database
tag:
- RAG
- 大模型100讲
- 检索增强
description: 解析双塔向量检索的信息压缩局限，讲解 Cross-Encoder 全交叉注意力机制与两阶段检索算力权衡，给出可验证代码与工程降级策略。
---

在检索增强生成（RAG）系统中，单纯依赖向量余弦相似度或关键词匹配，经常会把“表面词汇重合度高、但无法解答问题”的段落排在最前。上一讲在[/llm-100/rag/retrieval-methods.md](/llm-100/rag/retrieval-methods.md)中讨论了向量检索与 BM25 的混合检索，但混合检索返回的粗排列表依然受制于浅层评分融合。要为大语言模型（LLM）提供上下文噪声低、论据严密的参考片段，必须在召回之后引入重排序（Rerank）环节。

先给出核心结论：
1. **双编码器（Bi-Encoder）**为了支持数百万文档的离线向量建库与毫秒级近邻搜索，必须将整段文本压缩为单一定长向量，在编码时查询词与文档词彼此完全不可见，因而难以捕捉细粒度的语义对齐与逻辑否定。
2. **交叉编码器（Cross-Encoder）**将“查询+候选文档”拼接后同时输入 Transformer，每一层的自注意力让每个词都能相互参照，从而精准判断逻辑条件与上下文依存。
3. Cross-Encoder 无法离线提前固化文档特征，单次推理复杂度随拼接序列长度呈平方级扩展，且计算开销随候选文档数量线性累加。若对全库逐一计算耗时极高，因此工程上普遍采用**两阶段检索架构**：第一阶段粗排快速挑出几十到上百条候选，第二阶段精排对候选进行高精度重排，兼顾吞吐量与排序质量。

![两阶段检索漏斗机制](/images/content-rebuild/9ec0bea48bcd3c28/cover.png)

上图展示了从海量文档库到最终模型上下文的两阶段过滤漏斗。本讲围绕一个具体的工程排错场景，拆解 Rerank 机制的技术实现、计算瓶颈与降级策略。

## 一个贯穿例子：为什么相似度第一的文档不可用？

假设系统运维知识库中包含以下三篇技术文档，用户提问：
> **查询（Query）**：“如何在 Kubernetes 集群中配置跨节点 Pod 通信的 MTU？”

知识库召回池中有如下三篇候选文本：
- **文档 A（集群初始化教程）**：“Kubernetes 集群搭建指南：详细介绍如何使用 kubeadm 初始化 Master 节点及加入 Worker 节点，配置集群网络插件 Flannel 基础网络与 CIDR 划分……”
- **文档 B（底层硬件指南）**：“数据中心网络交换机配置：以太网接口标准 MTU 为 1500 字节，若需要支持巨型帧（Jumbo Frames），需在核心交换机端口全局开启 9000 字节配置……”
- **文档 C（网络插件调优）**：“Calico CNI 网络调优：当跨节点跨网段通信因外层网络隧道封装产生丢包时，需在 Felix 节点配置中手动指定 vethMTU 或开启 MTU 自动检测……”

如果采用纯向量检索：
- 文档 A 密集出现了“Kubernetes”、“集群”、“节点”、“网络插件”等词，在通用 Embedding 空间中与查询向量的余弦相似度极高，往往排在第 1 位，但整篇文档根本没有说明 MTU 具体该如何配置；
- 文档 B 命中“MTU”、“网络配置”，但其讨论对象是物理交换机硬件，与“Pod 通信”场景脱节；
- 文档 C 准确给出了 CNI 插件解决跨节点 Pod MTU 的方法，但因专有名词“Calico”、“Felix”、“vethMTU”权重分散，向量余弦相似度反而落在第 3 位甚至更靠后。

如果直接截取 Top-2 喂给大语言模型，模型拿到的上下文充斥着无关的集群搭建步骤与交换机配置，极易引发幻觉或答非所问。关于生成错误归因的深入剖析，可参考后续内容[/llm-100/rag/rag-wrong-answers.md](/llm-100/rag/rag-wrong-answers.md)。

Rerank 的核心使命，就是通过更深度的交叉注意力把真正相关的文档 C 提至首位，将缺乏有效解法的文档 A 与 B 降序排除。

![两阶段检索流水线架构](/images/content-rebuild/9ec0bea48bcd3c28/fig-1.png)

上图展示了两阶段检索的端到端数据流：
1. **召回阶段（Retrieval / First-Stage）**：追求覆盖率与吞吐。通过向量数据库近似最近邻搜索（ANN）或倒排索引（BM25），在数十毫秒内从全量 $N$ 条切片中筛选出候选集合 Top-$K$（通常 $K \in [50, 100]$）。
2. **精排阶段（Rerank / Second-Stage）**：追求精准次序。使用交叉注意力模型对 $K$ 个候选对逐一打分，截取 Top-$M$（通常 $M \in [3, 5]$）送入大语言模型的上下文窗口。

<ReadMoreLock />

## Bi-Encoder 独立编码的物理局限

召回阶段广泛使用的 Dense Embedding 模型属于**双编码器（Bi-Encoder）**架构。

![Bi-Encoder 独立编码与向量池化瓶颈](/images/content-rebuild/9ec0bea48bcd3c28/fig-2.png)

如上图所示，在 Bi-Encoder 中，查询（Query）和文档（Document）分别输入到独立的编码器网络中：

$$\mathbf{u} = \mathrm{Pool}(\mathrm{Transformer}_Q(q))$$
$$\mathbf{v} = \mathrm{Pool}(\mathrm{Transformer}_D(d))$$

其中 $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ 分别是经由均值池化（Mean Pooling）或提取 `[CLS]` 位置向量生成的定长向量。两者的相关性依赖向量点积或余弦相似度：

$$S_{\text{Bi}}(q, d) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

这种架构带来了显著的工程优势：所有文档的嵌入向量 $\mathbf{v}_i$ 可以在录入时**离线预先计算**并固化到向量数据库中。在线查询时，仅需计算一次 Query 嵌入 $\mathbf{u}$，随后在向量索引中执行近邻搜索。关于向量数据库的索引机制可参考[/llm-100/rag/vector-db-in-rag.md](/llm-100/rag/vector-db-in-rag.md)。

然而，这种设计天然存在两重信息损耗：
1. **无跨文本注意力（No Cross-Attention）**：在 Transformer 的各隐藏层中，Query 的词与 Document 的词完全不可见。自注意力运算 $A = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ 仅在单侧序列内部进行。
2. **定长向量压缩损失**：将包含数百个词的文档片段压缩为 768 或 1024 维的单一向量时，局部语义细节与约束条件会被全局均值稀释。面对包含限定词或否定词的查询（例如“非物理网卡的 MTU 配置”），单向量表征很难有效隔离修饰关系，极易被高频共现词带偏。

## Cross-Encoder 架构与全交叉注意力计算

重排序模型采用**交叉编码器（Cross-Encoder）**架构，在输入层直接建立词级别的细粒度关联。

![Cross-Encoder 全自注意力交互矩阵](/images/content-rebuild/9ec0bea48bcd3c28/fig-3.png)

如上图所示，Cross-Encoder 在输入阶段就将查询与候选文档拼接为单一文本序列：

$$\mathbf{x} = [\texttt{[CLS]}, q_1, q_2, \dots, q_{L_q}, \texttt{[SEP]}, d_1, d_2, \dots, d_{L_d}, \texttt{[SEP]}]$$

该拼接序列被统一输入 Transformer 编码器。在从底层到顶层的每一个自注意力层中，Query 的每个词元（Token）都能直接与 Document 的每个词元计算相关性权重：

$$\mathbf{H}^{(l)} = \mathrm{softmax}\left(\frac{\mathbf{H}^{(l-1)}\mathbf{W}_Q (\mathbf{H}^{(l-1)}\mathbf{W}_K)^T}{\sqrt{d_k}}\right) \mathbf{H}^{(l-1)}\mathbf{W}_V$$

在此机制下：
1. **跨序列直接对齐**：查询中的“Pod 通信”可以直接在自注意力矩阵中对准文档 C 中的“Felix / vethMTU”，同时降低文档 B 中与“交换机”相关的权重，解决双塔模型无法跨序列参照的盲区。
2. **单标量直接输出**：在最后一层，提取序列起始标记 `[CLS]` 对应的隐藏层向量 $\mathbf{h}_{\texttt{[CLS]}}$，通过线性层映射输出标量分数：

$$S_{\text{Cross}}(q, d) = \mathbf{w}^T \mathbf{h}_{\texttt{[CLS]}} + b$$

该分数直接反映了在全文本交互条件下，文档 $d$ 是否能作为回答查询 $q$ 的有效依据。

## 计算复杂度与两阶段权衡

既然 Cross-Encoder 判定更准，能否在整个库中直接使用它进行检索？答案是否定的，原因在于在线计算开销。

![Bi-Encoder 与 Cross-Encoder 计算复杂度与耗时对比](/images/content-rebuild/9ec0bea48bcd3c28/fig-4.png)

上图对比了两种架构在不同数据处理规模下的计算特性。我们可以从计算复杂度来拆解其本质差异：

### 1. 向量召回（Bi-Encoder + 向量索引）
假设库容为 $N$，向量维度为 $d_{\text{emb}}$：
- **在线前向推理**：仅需对 Query 计算 1 次 Transformer 编码，序列长度为 $L_q$。
- **索引检索复杂度**：使用基于图的 ANN 索引（如 HNSW）时，搜索耗时约为 $O(\log N \cdot d_{\text{emb}})$。
- **在线耗时**：当 $N = 10^6$（百万级切片）时，查询编码加索引检索通常可在 10~25 毫秒内完成。

### 2. 重排匹配（Cross-Encoder）
设拼接后的输入序列总长度为 $L = L_q + L_d$，需要打分的候选集数量为 $K$：
- **在线前向推理**：对于每一个候选对 $(q, d_i)$，自注意力层的浮点运算量与序列长度的平方成正比。计算复杂度为 $O(K \cdot L^2 \cdot d_{\text{model}})$。
- **无法预先缓存**：由于每次用户提问 $q$ 都是动态输入的，拼接后的序列 $[q; d_i]$ 每次都不同，因此文档端无法提前生成任何离线嵌入进行复用，**所有 $K$ 次前向传播必须在线即时计算**。

### 算例对比（基于相同文档库）
假设知识库规模 $N = 1,000,000$，平均查询长度 $L_q = 32$，文档片段长度 $L_d = 480$，拼接长度 $L = 512$：
- **方案 A（全库直接运行 Cross-Encoder）**：需对全库执行 $10^6$ 次 512-Token 的 Transformer 序列前向传播。即便按批大小（Batch Size）为 64 在现代 GPU 上并发计算，单次查询也需数分钟甚至更久，在线问答完全无法承受。
- **方案 B（两阶段检索流水线）**：
  1. 召回阶段：Bi-Encoder 在 $N=10^6$ 库中检索 Top-100 候选，耗时约 $15\,\text{ms}$。
  2. 精排阶段：Cross-Encoder 仅需对 $K=100$ 个候选对进行批量推理，在典型 GPU 上耗时约为 $60\sim 120\,\text{ms}$。
  3. 总耗时控制在 $150\,\text{ms}$ 以内，在保留高吞吐的前提下实现了接近单塔模型的精排效果。

## 折中探索：后期交互（ColBERT 机制）

在“完全独立双塔”与“完全拼接单塔”之间，以 ColBERT 为代表的**后期交互（Late Interaction）**机制提供了一种折中思路。

![ColBERT 后期交互与 MaxSim 矩阵计算](/images/content-rebuild/9ec0bea48bcd3c28/fig-5.png)

如上图所示，ColBERT 的设计特点如下：
1. **分别编码**：Query 与 Document 依然分别输入编码器，允许文档端在离线阶段预先计算特征。
2. **保留词级多向量（Multi-Vector）**：文档不再被压缩为单一向量，而是保留每一个 Token 位置降维投影后的词级向量列表：
   $$\mathbf{E}_q \in \mathbb{R}^{L_q \times d_v}, \quad \mathbf{E}_d \in \mathbb{R}^{L_d \times d_v} \quad (d_v \text{ 通常降维至 } 32\sim 128)$$
3. **MaxSim 算子打分**：在线检索时，直接计算 Query 与 Document 词向量之间的点积相似度矩阵，对 Query 中的每个词，找出 Document 中相似度最高的值，最后求和：

$$S_{\text{Late}}(q, d) = \sum_{i=1}^{L_q} \max_{j=1}^{L_d} \left( \mathbf{E}_{q, i} \cdot \mathbf{E}_{d, j}^T \right)$$

MaxSim 算子避免了深度自注意力层的二次方耗时，兼顾了部分细粒度词对齐能力。其主要代价是**索引存储体积成倍膨胀**（由单条定长向量变为数百条词向量）。在多数注重通用落地与工程维护的系统中，成熟的“Dense/BM25 召回 + Cross-Encoder 精排”两阶段方案依然是主流选择。

## 验证实验：重排前后的候选序列变化

为了直观检验 Cross-Encoder 对干扰项的过滤效果，我们基于开源重排模型（如 `BAAI/bge-reranker-base`）构建一个可运行的打分对比实验。

### 实验代码

```python
import time
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def run_rerank_experiment(
    query: str, 
    candidates: List[Dict[str, str]], 
    model_name: str = "BAAI/bge-reranker-base"
) -> Tuple[List[Dict[str, str]], float]:
    """
    对召回的候选文档执行 Cross-Encoder 重排序。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # 构建成对输入序列 [CLS] Query [SEP] Doc [SEP]
    pairs = [[query, doc["text"]] for doc in candidates]

    start_time = time.perf_counter()
    
    inputs = tokenizer(
        pairs, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # 获取分类 Logits，bge-reranker 单标量输出即为相关性打分
        scores = outputs.logits.view(-1).float().cpu().tolist()

    latency_ms = (time.perf_counter() - start_time) * 1000

    for doc, score in zip(candidates, scores):
        doc["rerank_score"] = score

    # 按 Cross-Encoder 分数降序排列
    reranked_docs = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked_docs, latency_ms

# 贯穿例子中的候选数据（模拟第一阶段召回输出）
query_sample = "如何在 Kubernetes 集群中配置跨节点 Pod 通信的 MTU？"
simulated_candidates = [
    {
        "id": "doc_A", 
        "text": "Kubernetes 集群搭建指南：详细介绍如何使用 kubeadm 初始化 Master 节点及加入 Worker 节点，配置集群网络插件 Flannel 基础网络与 CIDR 划分。", 
        "initial_rank": 1
    },
    {
        "id": "doc_B", 
        "text": "数据中心网络交换机配置：以太网接口标准 MTU 为 1500 字节，若需要支持巨型帧需在核心交换机端口全局开启 9000 字节配置。", 
        "initial_rank": 2
    },
    {
        "id": "doc_C", 
        "text": "Calico CNI 网络调优：当跨节点跨网段通信因外层网络隧道封装产生丢包时，需在 Felix 节点配置中手动指定 vethMTU 或开启 MTU 自动检测。", 
        "initial_rank": 3
    },
    {
        "id": "doc_D", 
        "text": "Kubernetes Pod 资源配置规范：详细讲解 requests 与 limits 字段对 CPU 与内存调度的影响。", 
        "initial_rank": 4
    }
]

if __name__ == "__main__":
    try:
        results, elapsed = run_rerank_experiment(query_sample, simulated_candidates)
        print(f"Rerank 处理耗时: {elapsed:.2f} ms\n")
        print("--- 原始召回顺序 vs 重排后顺序 ---")
        for rank, doc in enumerate(results, 1):
            print(f"新排名: {rank} | 原召回排名: {doc['initial_rank']} | 原始ID: {doc['id']} | 分数: {doc['rerank_score']:.4f}")
            print(f"文本摘要: {doc['text'][:45]}...\n")
    except Exception as e:
        print(f"环境依赖提示: 请先安装 transformers 与 torch，并确保模型权重可拉取。错误信息: {e}")
```

### 实验输出与对比分析（贯穿样例示意）

运行上述逻辑，在当前样本集下的打分特征如下表所示：

| 文档编号 | 关键内容 | 初始召回排名（双塔向量） | 重排后排名（Cross-Encoder） | 排序变化原因 |
| :--- | :--- | :--- | :--- | :--- |
| **`doc_C`** | Calico CNI 配置 vethMTU | 第 3 位 | **第 1 位** | 模型通过全自注意力捕获“Pod 通信”与“Calico CNI”及“vethMTU”的对应关系 |
| **`doc_A`** | kubeadm 初始化集群步骤 | 第 1 位 | **第 2 位** | 包含集群网络背景但未涉及 MTU 参数配置，排在正解之后 |
| **`doc_B`** | 物理交换机配置巨型帧 | 第 2 位 | **第 3 位** | 虽提及 MTU，但模型识别出其属于硬件交换机配置，脱离 Pod 通信语境 |
| **`doc_D`** | Pod CPU/内存资源配额 | 第 4 位 | **第 4 位** | 完全无关的主题在两阶段评分中均居末尾，直接淘汰 |

通过跨词注意力，原本被埋在第 3 位的正解 `doc_C` 跃升至第一位，而原本凭借关键词热度排在前面的 `doc_A` 与 `doc_B` 被有效后置。这证明重排阶段能显著减轻大模型生成时被噪声文档干扰的问题。

## 生产落地中的边界与降级机制

在将 Rerank 服务接入线上生产链路时，必须明确其运行边界与容错策略：

![Rerank 生产落地中的边界与降级状态机](/images/content-rebuild/9ec0bea48bcd3c28/fig-6.png)

上图展示了重排链路的高可用状态流转：
1. **动态候选池截断（Dynamic Batch Truncation）**：高并发访问时，若 GPU 显存或队列耗时达到预警水位，调度层应动态将候选集 $K$ 从 100 临时降至 30 或 50，优先保障 P99 响应延迟稳定。
2. **严格超时与熔断回退（Timeout & Fallback）**：Rerank 模型通常作为独立的微服务部署，必须设置硬性超时阈值（如 $120\sim 150\,\text{ms}$）。一旦重排服务超时或发生网络中断，链路应自动降级，直接取第一阶段召回（如 BM25 与向量混合检索的 RRF 分数）截取前 $M$ 篇文档送入生成模型，确保主交互链路不中断。
3. **打分阈值校准**：Cross-Encoder 输出的原始 Logits 未经过归一化，其数值范围随不同模型、不同预训练目标差异极大（可能分布在 $[-10, 10]$ 或更大范围）。不可直接设定一个通用的固定阈值（例如“必须大于 0.8 才保留”），需基于测试集的打分分布拟合 Sigmoid 概率或分位数判定。

系统化的检索评测指标（如 MRR、NDCG、Precision@K）与优化路径，可参阅后续的进阶总结[/llm-100/rag/optimize-rag.md](/llm-100/rag/optimize-rag.md)。有关技术面试中的高频设问，亦可对照[/interview/rag-interview/rag-rerank.md](/interview/rag-interview/rag-rerank.md)与[/interview/rag-interview/rerank.md](/interview/rag-interview/rerank.md)。

## 练习与检验

为检验对本讲内容的掌握程度，读者可以结合实验代码完成以下验证：
1. **限定词敏感度测试**：在测试用例中加入一个负向查询（如“非 CNI 插件层面的主机网络 MTU 配置”），对比纯向量召回顺序与 Rerank 顺序的差异，观察 Cross-Encoder 是否能够识别“非 CNI”这一否定修饰条件。
2. **规模与长度开销测试**：将候选文档列表扩增至 $K=20, 50, 100, 200$，观察模型总耗时随候选数 $K$ 的近线性增长；同时尝试调整文档截断长度（如 128、256、512），验证耗时随单序列长度 $L$ 的增长趋势。
