---
title: RAG 是什么？为什么大模型需要外部知识库？
icon: mdi:database
tag:
- RAG
- 大模型100讲
- 检索增强
description: 用运维实战案例讲透 RAG 核心机制：对比参数记忆与外部索引，拆解加载检索五步流程，提供纯 Python 可运行验证代码与工程练习。
---

**学习目标**：
- 搞清楚大模型为什么不能仅靠参数记住所有事实
- 理解 RAG 的五个阶段（加载、切分、索引、检索、合成）
- 理解提示词上下文注入的机制与局限
- 运行并修改一个纯 Python 实现的内存检索验证程序

假设你正在维护一套微服务系统，刚在内网上线了一份配置变更说明。此时一名工程师向技术助手提问：

> **问**：“服务 A 在什么超时条件下会触发熔断？降级行为是什么？”

大语言模型（LLM）本身是在公开互联网语料上预训练完成的。它的权重在部署时已经冻结，根本不可能知道你昨晚在内部系统里做了什么更新。如果直接让模型回答，它只能根据通用的公网经验去“猜”，甚至编造出一个看似合理的超时时间和降级策略。

**检索增强生成（Retrieval-Augmented Generation，简称 RAG）** 的思路非常直接：不要指望模型把所有事实都背在参数里，而是在它回答之前，先从你的外部知识库中检索出相关的文档片段，把这些片段当成参考资料塞进提示词，再让模型根据参考资料总结出答案。

![RAG 架构整体运行全貌](/images/content-rebuild/1ba02e09ddd23a13/cover.png)

上图展示了 RAG 架构的双轨工作流：左轨将异构数据离散化为可检索单元并构建非参数索引，右轨在推理阶段通过向量相似度计算抓取上下文，组装提示词后交由冻结的模型解码生成答案。

---

## 为什么大模型不能仅靠自身权重记住所有事实？

大模型在训练时，目标是根据前文预测下一个单词（Token）：

$$\mathcal{L}_{\text{LM}}(\theta) = - \sum_{t=1}^{T} \log P_\theta(w_t \mid w_{<t})$$

在这个过程中，语料里的事实被转化成复杂的网络权重（主要是前馈网络层和注意力层的浮点数）。这种依靠模型权重保存信息的方式，称为**参数化记忆（Parametric Memory）**。

依赖参数化记忆来维护业务知识，在工程上会遇到三个实际困难：

1. **知识有时效截断**：模型的记忆停留在训练数据收集的那一刻。权重一旦冻结，现实世界的任何新变化都不会自动同步进去。
2. **增量更新代价大**：如果为了更新某几条业务规则而重新微调（Fine-Tuning）模型，不仅算力成本高，而且连续的梯度更新容易破坏模型原有的输出稳定性，造成灾难性遗忘。微调更适合让模型学习特定的输出风格、指令格式或专业任务逻辑，不适合充当高频更新的事实数据库。关于两者的对比，可参考 [/llm-100/rag/rag-vs-finetuning.md](/llm-100/rag/rag-vs-finetuning.md)。
3. **无法精确追溯与即时删除**：模型权重是高度交织的连续数值，无法像传统数据库那样通过 `WHERE id = ?` 精确查询某条记录的来源，更无法执行可靠的单条数据删除或修改校验。

![参数化黑盒与外部非参数索引对比](/images/content-rebuild/1ba02e09ddd23a13/fig-1.png)

上图对比了参数化记忆与外部索引的机制差异：参数化记忆将知识纠缠在全网稠密浮点权重矩阵中，难以精确定位与单独修改；外部非参数索引则是解耦的独立文档集合，支持毫秒级的增删改查（CRUD）与事实追踪。

---

## RAG 核心处理管道：加载、切分、索引、检索与合成

为了解决上述问题，RAG 把知识存储从模型内部抽离出来，交给**外部索引（External Index）**管理。一个典型的 RAG 系统通常包含以下五个阶段：

```
[原始文档] 
    │
    ▼  (1) 加载 (Loading)
[统一 Document 对象]
    │
    ▼  (2) 切分 (Chunking)
[文本块 (Chunk / Node)]
    │
    ▼  (3) 索引 (Indexing: 计算向量并存入向量库)
[向量索引 / 倒排索引]
    │
    │ ◄─── 用户提问 (Query)
    ▼  (4) 检索 (Retrieving: 计算相似度，取 Top-k 片段)
[候选证据片段]
    │
    ▼  (5) 合成 (Synthesizing: 拼装提示词并交给 LLM 生成答案)
[最终自然语言答案]
```

1. **加载（Loading）**：解析 PDF、Markdown、Word、网页或数据库导出记录，提取出纯文本并附带来源文件、章节等元数据。不同文件格式的解析考量可参考 [/frontier/document-parsing/what-is-file-parsing.md](/frontier/document-parsing/what-is-file-parsing.md)。
2. **切分（Chunking）**：长文档不能直接塞进检索库，需要拆成相对独立、长度适中的文本块（Chunk）。切块过大会导致语义重点被稀释，切块过小又会切断上下文关联。具体切分策略见 [/llm-100/rag/chunk-strategy.md](/llm-100/rag/chunk-strategy.md)。
3. **索引（Indexing）**：利用文本嵌入模型（Embedding Model）把每个文本块转化为固定维度的密集实数向量（Dense Vector），写入支持近似最近邻搜索（ANN）的向量数据库或倒排索引。Embedding 的原理与选型依据见 [/llm-100/rag/embedding-selection.md](/llm-100/rag/embedding-selection.md) 与 [/llm-100/rag/vector-db-in-rag.md](/llm-100/rag/vector-db-in-rag.md)。
4. **检索（Retrieving）**：用户提问时，系统将问题用**相同的 Embedding 模型**转化为向量，在向量库中计算距离（如余弦相似度），找出语义最接近的前 $k$ 个文本块。
5. **合成（Synthesizing）**：把检索出来的文本块拼进预设的提示词模板中，交给 LLM 阅读并生成最终回答。

![RAG 核心管道数据流与状态转移](/images/content-rebuild/1ba02e09ddd23a13/fig-2.png)

上图展示了文档从非结构化原始状态经过五阶流水线转换的全过程。注意：在离线索引阶段与在线检索阶段，Embedding 模型必须保持参数一致性，否则向量空间将失去数学可比性。

<ReadMoreLock />

---

## 上下文注入机制：提示词模板如何组装 Query 与文档片段

大模型本身并不会主动连接数据库去查资料，它在推理时只是读取输入序列。RAG 所谓的“增强”，本质是在推理之前，通过**上下文注入（Context Injection）**把外部证据放进模型的输入文本里。

一个典型的提示词模板结构如下：

```text
[系统指令]
你是一个严格基于提供的事实回答问题的技术助手。请遵循以下规则：
1. 仅依据 [参考上下文] 中明确提及的内容回答，不要引入外部推测；
2. 如果 [参考上下文] 的信息不足以解答，请直接说明“依据不足”，严禁编造；
3. 引用证据时，标注文档对应的编号。

[参考上下文]
--- 文档 [1] (来源: service_arch.md, 块ID: 102) ---
服务 A 在请求超时达到 3000 毫秒时自动触发熔断，并向监控中心上报 ALERT_TIMEOUT 事件。

--- 文档 [2] (来源: service_arch.md, 块ID: 103) ---
服务 A 在进入熔断状态后，后续所有读请求将直接走本地缓存降级，持续周期为 60 秒。

[用户提问]
服务 A 在什么条件下会触发熔断？降级行为是什么？

[生成答案]
```

从 Transformer 的计算原理来看，模型在处理这段提示词时，输入的每个 Token 会通过自注意力机制互相计算关联度：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

在生成回答时，新生成的 Token 会跨越位置区间，对 `[参考上下文]` 区域里的事实 Token 产生较高的注意力权重。通过给模型输入明确的参考文本，我们将预测下一个 Token 的概率分布从无约束的纯先验输出，收敛为受上下文制约的条件概率输出：

$$P(w_t \mid w_{<t}, \text{参考上下文}) \neq P(w_t \mid w_{<t})$$

原本在通用预训练语料中出现概率较低的内部私有规则（如“3000毫秒”“走本地缓存降级”），因为出现在了输入上下文里，生成概率得到显著提升。更多进阶上下文处理见 [/frontier/context-engineering/what-is-context-eng.md](/frontier/context-engineering/what-is-context-eng.md)。

![Prompt 上下文注入装配与注意力交互示意](/images/content-rebuild/1ba02e09ddd23a13/fig-3.png)

上图拆解了输入上下文的装配物理过程：系统指令规范了模型的置信度阈值与格式，检索召回的片段注入中间区域，底部的生成 Token 产生注意力权重，跨越上下文区间精准寻址。

---

## 为什么引入检索不能 100% 消除幻觉？

工程中常见的一个误区是认为“接了 RAG 就绝对不会胡说”。RAG 显著降低了错误率，但并不能在理论上保证 100% 消除幻觉。失效通常发生在两个阶段：

### 1. 检索阶段失效（找错了或漏掉了）
- **语义相似不等于逻辑相符**：Embedding 衡量的是全局语义方向。对于包含双重否定、复杂时序先后、专有错误码（如 `ERR_504_TIMEOUT`）的查询，单纯依靠稠密向量检索容易找出一堆主题接近、但具体条件完全不匹配的片段。
- **召回缺失与阈值拦截**：如果向量相似度阈值设得过高，检索结果可能为空；如果 $k$ 设得过小，包含关键证据的文本块可能排在第 $k+1$ 位被截断。一旦没有检索到有效证据，模型就只能退化为依靠自身预训练参数去编造。

### 2. 生成阶段失效（给对了但读错了）
- **知识冲突（Knowledge Conflict）**：当外部文档的内容与模型预训练时见过成千上万次的高频知识相冲突时，模型的内在参数先验与外部上下文注意力会产生竞争。如果提示词约束不够强，模型有时会忽略注入的上下文，执意输出它更习惯的错误答案。排查与纠正此现象的策略见 [/llm-100/rag/rag-wrong-answers.md](/llm-100/rag/rag-wrong-answers.md)。
- **中间位置注意力衰减（Lost in the Middle）**：长文本注意力往往偏向开头和结尾。如果关键片段被夹在一大堆弱相关的候选文档中间，模型可能无法充分提取关键信息。

![检索与生成两阶段幻觉来源分解](/images/content-rebuild/1ba02e09ddd23a13/fig-4.png)

上图将系统失效解构为两阶段关卡：第一阶段为检索漏斗过滤后的信息偏差；第二阶段为模型在强参数先验与上下文注意力之间的概率博弈。任何一个环节出现破损，最终生成均可能产生事实幻觉。

![知识冲突时的解码分叉机制](/images/content-rebuild/1ba02e09ddd23a13/fig-5.png)

上图揭示了解码阶段的底层机制冲突：当外部证据与模型内在先验产生矛盾时，自注意力聚合的外部上下文特征与全网深层参数对高频模式的内在偏好在词表投影（Logits）阶段产生竞争，最终生成的 Token 取决于两者的概率分布博弈。

---

## 最小验证代码：用纯 Python 验证 RAG 流程

以下是一段不依赖任何第三方 AI 框架的纯 Python 验证脚本。为了让你直接看清数据流动，这里用固定词表词频来模拟向量嵌入（生产环境必须换成预训练 Embedding 模型），完整走一遍索引、相似度计算与提示词组装。

```python
import math
from typing import List, Dict, Tuple

# 1. 原始知识文档库
raw_documents = [
    {"doc_id": 1, "text": "服务A在请求超时达到3000毫秒时自动触发熔断，并向监控中心上报ALERT_TIMEOUT事件。"},
    {"doc_id": 2, "text": "服务B支持通过动态权重调整流量，权重范围为0到100，默认生效值为50。"},
    {"doc_id": 3, "text": "服务A在进入熔断状态后，后续所有读请求将直接走本地缓存降级，持续周期为60秒。"},
]

# 2. 模拟 Embedding 过程（基于词表词频 + L2 归一化）
VOCABULARY = ["服务A", "服务B", "熔断", "超时", "权重", "缓存", "3000毫秒", "监控"]

def mock_embedding(text: str) -> List[float]:
    # 统计词频构建原始向量
    vec = [float(text.count(term)) for term in VOCABULARY]
    norm = math.sqrt(sum(x ** 2 for x in vec))
    if norm == 0:
        return [0.0] * len(VOCABULARY)
    # L2 归一化后，两个向量的点积直接等于余弦相似度
    return [x / norm for x in vec]

# 3. 离线构建索引
index_store: List[Dict] = []
for doc in raw_documents:
    index_store.append({
        "doc_id": doc["doc_id"],
        "text": doc["text"],
        "embedding": mock_embedding(doc["text"])
    })

# 4. 在线检索：计算余弦相似度并排序
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))

def retrieve(query: str, top_k: int = 2) -> List[Tuple[Dict, float]]:
    query_vec = mock_embedding(query)
    scored_candidates = []
    for item in index_store:
        score = cosine_similarity(query_vec, item["embedding"])
        scored_candidates.append((item, score))
    # 按相似度降序
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return scored_candidates[:top_k]

# 5. 上下文合成：组装提示词
def assemble_prompt(query: str, retrieved_docs: List[Tuple[Dict, float]]) -> str:
    context_segments = []
    for i, (doc, score) in enumerate(retrieved_docs, 1):
        context_segments.append(f"[文档 {i} | 相似度: {score:.4f}]\n{doc['text']}")
    
    context_str = "\n\n".join(context_segments)
    
    prompt = f"""[系统指令]
你是一个客观的技术助理。请严格根据下方 [参考上下文] 的事实回答问题。如果信息不足，回答“依据不足”。

[参考上下文]
{context_str}

[用户提问]
{query}

[生成答案]"""
    return prompt

if __name__ == "__main__":
    user_query = "服务A在什么超时条件下会触发熔断？"
    hits = retrieve(user_query, top_k=2)
    final_prompt = assemble_prompt(user_query, hits)
    
    print("=== 检索阶段输出 ===")
    for doc, score in hits:
        print(f"ID: {doc['doc_id']} | 相似度分数: {score:.4f} | 文本: {doc['text']}")
    
    print("\n=== 上下文注入后的最终提示词 ===")
    print(final_prompt)
```

### 运算结果与验证

我们可以手工推导上述向量计算：
- 查询 `"服务A在什么超时条件下会触发熔断？"` 命中了 `服务A`、`熔断`、`超时` 三个词，未归一化向量模长为 $\sqrt{1^2+1^2+1^2} = \sqrt{3}$。
- 文档 1 命中了 `服务A`、`熔断`、`超时`、`3000毫秒`、`监控` 共五个词，模长为 $\sqrt{5}$。两者点积为 $\frac{1\times 1 + 1\times 1 + 1\times 1}{\sqrt{3}\times\sqrt{5}} = \frac{3}{\sqrt{15}} \approx 0.7746$。
- 文档 3 命中了 `服务A`、`熔断`、`缓存` 三个词，模长为 $\sqrt{3}$。点积为 $\frac{1\times 1 + 1\times 1}{\sqrt{3}\times\sqrt{3}} = \frac{2}{3} \approx 0.6667$。
- 文档 2 只包含服务 B 的词，点积为 0。

运行上述代码，控制台会输出：

```text
=== 检索阶段输出 ===
ID: 1 | 相似度分数: 0.7746 | 文本: 服务A在请求超时达到3000毫秒时自动触发熔断，并向监控中心上报ALERT_TIMEOUT事件。
ID: 3 | 相似度分数: 0.6667 | 文本: 服务A在进入熔断状态后，后续所有读请求将直接走本地缓存降级，持续周期为60秒。

=== 上下文注入后的最终提示词 ===
[系统指令]
你是一个客观的技术助理。请严格根据下方 [参考上下文] 的事实回答问题。如果信息不足，回答“依据不足”。

[参考上下文]
[文档 1 | 相似度: 0.7746]
服务A在请求超时达到3000毫秒时自动触发熔断，并向监控中心上报ALERT_TIMEOUT事件。

[文档 2 | 相似度: 0.6667]
服务A在进入熔断状态后，后续所有读请求将直接走本地缓存降级，持续周期为60秒。

[用户提问]
服务A在什么超时条件下会触发熔断？

[生成答案]
```

后续无论是采用关键词加向量的混合检索（见 [/llm-100/rag/retrieval-methods.md](/llm-100/rag/retrieval-methods.md)），还是加上重排模型重新打分（见 [/llm-100/rag/rerank-importance.md](/llm-100/rag/rerank-importance.md)），本质都是在优化这个管道各阶段的准确率与抗噪能力。

![最小验证代码运行流水线与控制台输入输出](/images/content-rebuild/1ba02e09ddd23a13/fig-6.png)

上图梳理了前文最小验证脚本的数据处理闭环：文本库经向量化后沉淀至内存，通过 Query 点积运算完成排序与打分截断，最终拼装为符合规范的提示词送入生成阶段。

---

## 长上下文普及了，为什么还要用 RAG？

现在很多模型已经支持 128k 甚至 1M Token 的长上下文窗口。既然能把整本手册直接塞进模型，为什么还需要分块与检索？

在真实工程场景中，两者有明确的成本和性能分界：

1. **首字延迟与算力开销**：在处理输入文本时（Prefill 阶段），标准自注意力计算必须让每个 Token 与前文所有 Token 进行点积，计算量随序列长度呈现二次方 $O(N^2)$ 增长。如果有数百万 Token 的知识库，每次提问都要重新计算全量注意力，首字生成时间（TTFT）和显存占用会非常明显；而基于向量库的检索时间通常在毫秒级，检索完成后只需要给模型喂入数千 Token 的精准片段。
2. **调用成本**：主流商业大模型 API 均按输入 Token 量计费。每次用户提问都发送几十万 Token，调用成本会极其昂贵；RAG 每次仅注入过滤后的少量片段，单次请求成本可以降低数个数量级。
3. **信息定位精度**：实验证明，当输入文档极其冗长时，模型即使拥有足够长的窗口，也容易出现关键信息遗漏的情况。详细权衡见 [/frontier/long-context-app/what-is-long-context.md](/frontier/long-context-app/what-is-long-context.md)。

因此，超长上下文更适合**单篇长报告分析、代码库全局依赖理解或多轮复杂长会话**；而大规模动态知识库、高频更新内容与企业多租户数据，依然以 **RAG** 为主要架构方案。

---

## 动手练习

请复制上文的 Python 验证脚本到本地环境运行，并完成以下两个实验：

1. **支持服务 B 查询**：在 `raw_documents` 中添加一条针对服务 B 降级策略的新文档，调用 `retrieve("服务B的默认权重是多少？", top_k=1)`，观察能否正确召回文档 2 并验证得分计算。
2. **观察词表覆盖限制**：使用一个不在 `VOCABULARY` 中的词提问（例如 `"服务A的错误日志在哪里？"`），观察检索得分的变化。思考并记录：在生产环境中，为什么必须使用真正的预训练 Dense Embedding 模型而非固定词表词频？
