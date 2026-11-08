---
title: 什么是 RAG 中的 Rerank？主流模型选型与工程落地怎么做？
icon: mdi:sort
tag:
- RAG
- Rerank
- 重排序
description: 详解 RAG 中 Rerank 重排的作用机制、Cross-Encoder 与 ColBERT 晚期交互的选型对比、ONNX 加速实战及超时降级策略。
---

在检索增强生成（RAG）系统中，从海量知识库粗排召回到送入大语言模型（LLM）生成之间，重排（Rerank）是决定上下文质量与端到端响应耗时的关键环节。粗排阶段的向量双塔模型为了实现数百万级文档的毫秒级检索，牺牲了查询（Query）与文档（Document）之间的细粒度交互；重排模块则负责在缩小后的候选集（通常为 20~50 个候选文本块）上恢复这种交互，修正排序偏差。

理解 Rerank 工程落地，核心在于权衡**排序精度**、**计算耗时**与**系统稳定性**。本文围绕主流模型选型（BGE-Reranker、Cohere）、多向量晚期交互（ColBERT）、推理加速实操以及高并发下的超时熔断降级展开说明。

![重排模块在 RAG 链路中的工程定位与选型流向](/images/content-rebuild/76bb185de848aee0/cover.png)

---

## 1. 核心口述答案：重排模块的作用与工程落地规范

在面试中被问到“什么是 RAG 中的 Rerank？生产环境具体怎么做？”时，建议按“定义与解决的问题 -> 选型矩阵 -> 耗时预算 -> 异常降级”四个层次作答：

> **核心口述回答：**
>
> Rerank（重排）是 RAG 检索链路中的二阶段精排机制。向量双塔粗排模型分别独立编码 Query 和 Document，无法捕捉词与词之间的修饰、限定和否定等跨上下文交互，容易把主题相似但未解答问题的噪声排在前面。Rerank 模块接收粗排召回的少量候选集，通过更充分的跨上下文计算，重新计算相关性得分并截取 Top-K 送入大模型。
>
> 在工程落地中，主要遵循三条实践原则：
>
> 1. **模型选型按场景划分**：
>    * **高精度私有化部署**：选用 Cross-Encoder 架构（如 `bge-reranker-large` 或支持多语言长文本的 `bge-reranker-v2-m3`）。单批次打分精度高，但计算复杂度随输入序列长度呈二次方增长；
>    * **免运维云端 API**：选用 Cohere Rerank 等托管服务，语义理解和多语言排序质量高，但需要评估网络往返时延（RTT）及数据出境合规；
>    * **高吞吐低延迟精排**：选用以 ColBERT 为代表的多向量晚期交互（Late Interaction）模型。文档词向量可在入库时预先计算并持久化存储，在线仅做轻量级 MaxSim 向量内积汇总，吞吐量明显高于 Cross-Encoder。
> 2. **严格控制延迟预算**：以交互式问答整体 P99 延迟控制在 500ms 内的典型场景为例，粗排混合检索通常分配 30~50ms，大模型首字生成（TTFT）预留 200~300ms，留给重排模块的硬性时间窗口仅有 **50~100ms**。因此候选集大小通常收敛在 **20~40 个候选块**，最终重排截取前 3~5 个喂给大模型。
> 3. **具备超时降级与门禁机制**：重排服务设置硬超时（如 80ms）。超时或节点异常时，降级旁路立即生效，回退使用粗排阶段的加权分数（如混合检索中的加权 RRF）直接截断输出；若重排最高分低于安全阈值，直接触发拒答，防止低质量上下文诱发模型幻觉。

![在线 RAG 系统端到端延迟预算切分与重排时间窗口](/images/content-rebuild/76bb185de848aee0/fig-1.png)

---

## 2. 为什么需要 Rerank：从贯穿示例看双塔的局限

为了直观理解 Rerank 的必要性，我们以一个具体的查询为例：

> **用户查询**：“Python 中如何避免多线程下的全局解释器锁（GIL）阻塞？”

知识库粗排召回了以下三个候选文本块：
* **文本块 A**：“深入理解 Python 全局解释器锁：GIL 诞生于 1990 年代，是为了解决多线程访问共享内存的线程安全问题……”（包含大量“Python”、“全局解释器锁”、“多线程”关键词，主题高度吻合，但全文都在讲历史和原理，没有回答“如何避免阻塞”）；
* **文本块 B**：“规避 Python GIL 阻塞的工程方案：对于 CPU 密集型任务，应使用 `multiprocessing` 替代多线程；或者将耗时计算封装在 C/C++ 扩展中，在执行密集计算前调用 `Py_BEGIN_ALLOW_THREADS` 显式释放 GIL……”（精准回答了“如何避免阻塞”）；
* **文本块 C**：“Java 并发编程：深入解析 synchronized 与 ReentrantLock 避免线程阻塞机制……”（包含大量“并发”、“多线程”、“锁”、“阻塞”，但语言主体不匹配）。

### 双塔模型（Bi-Encoder）为什么容易误判？

在双塔检索中，Query 和 Document 分别输入相互独立的 Encoder（编码器），各自压缩成一个固定维度的稠密向量 $\vec{q}$ 和 $\vec{d}$，最末端只计算余弦相似度或点积：

$$\text{Score}_{\text{Bi}} = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}$$

在将几百个 Token 压缩为一个向量的过程中，文本块 A 中高频出现的领域专有名词会让整体语义向量严重偏向“Python GIL 概念”。此时双塔余弦打分可能出现：

$$\text{Score}(A) > \text{Score}(B)$$

文本块 A 尽管没有提供解决方案，却因为关键词密度被排在最前；而真正给出方案的文本块 B 可能被排在第十名甚至更靠后。

### Cross-Encoder 如何纠正排序？

Cross-Encoder 将 Query 和 Document 拼接成同一个序列直接送入 Transformer：

$$\text{Input} = [\text{CLS}] + Q + [\text{SEP}] + D + [\text{SEP}]$$

在模型的每一层 Transformer 编码层中，Query 中的每一个 Token 都可以与 Document 中的每一个 Token 进行双向全注意力（All-to-All Attention）交互。模型能够清楚感知到 Query 中的核心动词“避免”与“阻塞”，与文本块 B 中的“规避方案”、“显式释放 GIL”之间存在强逻辑修饰与条件约束关系，从而准确给文本块 B 打出最高分。

![Cross-Encoder 与 Bi-Encoder 的内部注意力交互机制对比](/images/content-rebuild/76bb185de848aee0/fig-2.png)

---

## 3. 主流重排方案对比：Cross-Encoder 与 ColBERT 晚期交互

在实际生产中，选择重排模型需要在精确度与计算吞吐之间权衡。目前业界主要有两类主流技术路线：全交互 Cross-Encoder 与多向量晚期交互（Late Interaction）。

### 方案一：Cross-Encoder（如 BGE-Reranker 系列）

北京智源人工智能研究院（BAAI）开源的 BGE-Reranker 系列是私有化部署的常见基准：
* **bge-reranker-base / large**：基于 RoBERTa/XLM-RoBERTa 架构微调，large 版本参数量约为 5.6 亿。在中英文实体匹配、否定关系与细粒度语义区分上表现稳定；
* **bge-reranker-v2-m3**：支持跨语言交互，支持长达 8192 Token 的上下文长度，适合长文档分块及多语种检索场景。

**计算特性**：输入序列长度为 $L = L_q + L_d$。自注意力计算复杂度为 $\mathcal{O}(L^2)$。若对 $K$ 个候选文本块打分，需要执行 $K$ 次前向传播（可组 Batch 并行），无法利用预计算，计算开销较大。

### 方案二：ColBERT 晚期交互（Late Interaction）

当并发请求上升、候选集需要扩大至 100~200 个文本块时，Cross-Encoder 往往会导致 GPU 算力瓶颈。ColBERT（Contextualized Late Interaction over BERT）提供了一种折中解法：

1. **离线处理文档**：文本块 $D$ 在建索引阶段通过 Transformer 编码，保留**每个 Token 的低维嵌入向量**（如降维至 128 维），得到矩阵 $E_d \in \mathbb{R}^{|D| \times 128}$，预先存储在向量数据库或内存中；
2. **在线编码查询**：用户 Query 输入相同的编码器，生成 Token 向量矩阵 $E_q \in \mathbb{R}^{|Q| \times 128}$；
3. **MaxSim 算子打分**：不进行昂贵的跨文本 Transformer 全注意力层计算，而是直接计算 Query Token 与 Document Token 的余弦相似度矩阵，对每个 Query Token 取其在整个文档中的最大相似度，最后相加：

$$\text{Score}_{\text{ColBERT}}(Q, D) = \sum_{i=1}^{|Q|} \max_{j=1}^{|D|} \left( E_{q, i} \cdot E_{d, j}^\top \right)$$

![ColBERT 晚期交互 MaxSim 相似度矩阵运算流程](/images/content-rebuild/76bb185de848aee0/fig-3.png)

### 架构与工程指标权衡矩阵

| 特性维度 | Bi-Encoder（常规粗排向量） | ColBERT（晚期交互） | Cross-Encoder（经典精排） |
| :--- | :--- | :--- | :--- |
| **交互深度** | 仅在最末端执行标量内积 | 向量表征完成后做词级 MaxSim 匹配 | 输入层拼接，逐层执行全自注意力 |
| **打分计算复杂度** | $\mathcal{O}(d_v)$（$d_v$ 为向量维度） | $\mathcal{O}(|Q| \cdot |D| \cdot d_v)$ 轻量矩阵乘法 | $\mathcal{O}((L_q + L_d)^2)$ 逐层自注意力 |
| **文档离线预计算** | 支持（每文档压缩为 1 个向量） | 支持（每文档持久化 $|D|$ 个词向量） | **不支持**（依赖 Query 动态拼接） |
| **存储与内存占用** | 低（单文档 768~1024 Float32） | 较高（需结合量化机制压缩存储） | 0（无需为精排持久化特征向量） |
| **相对排序精度** | 基准水平 | 接近 Cross-Encoder | 行业质量上限基准 |

现代向量引擎（如 Qdrant 的多阶段查询接口）已支持该模式：先利用常规密集向量检索粗选候选集，随后在内存中调用存储的多向量执行 MaxSim 细排，兼顾了吞吐与细粒度词法交互。

<ReadMoreLock />

---

## 4. 重排推理加速实战：ONNX Runtime 与批处理

直接使用原生 Hugging Face `pipeline` 进行 Python 单条循环推理，通常难以满足生产环境的并发与延迟要求。在自建 Cross-Encoder 服务时，常规的工程优化手段包括算子融合（ONNX Runtime / TensorRT）、半精度（FP16）以及动态 Batching。

![重排模型推理加速技术栈与优化路径](/images/content-rebuild/76bb185de848aee0/fig-4.png)

### 可运行加速示例：基于 ONNX Runtime 的 Cross-Encoder 打分服务

以下代码展示了一个基于 ONNX Runtime 的 Cross-Encoder 批处理打分实现。它包含了动态张量封装、批大小切分以及分类头分数的标准化映射：

```python
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort

class OptimizedReranker:
    def __init__(self, model_onnx_path: str, tokenizer_path: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        
        # 配置 ONNX Runtime 推理会话，开启所有图级优化
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 优先指定 CUDA 硬件加速，CPU 兜底
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_onnx_path, sess_options, providers=providers)

    def compute_scores(self, query: str, documents: list[str], batch_size: int = 32) -> list[float]:
        all_scores = []

        # 批处理执行，避免单批次过大导致显存溢出
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_queries = [query] * len(batch_docs)
            inputs = self.tokenizer(
                text=batch_queries,
                text_pair=batch_docs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np"
            )
            
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            if "token_type_ids" in inputs:
                ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

            # 执行前向推演
            ort_outputs = self.session.run(None, ort_inputs)
            logits = ort_outputs[0]
            
            # 归一化得分处理：
            # 1. 若输出为单 Logit（如 bge-reranker-large 原生输出），经由 Sigmoid 映射到 [0, 1]
            # 2. 若输出为二分类 Logits (batch, 2)，通过 Softmax 取正类 (index=1) 的概率
            if logits.ndim == 1 or logits.shape[-1] == 1:
                flat_logits = logits.flatten()
                scores = (1 / (1 + np.exp(-flat_logits))).tolist()
            else:
                exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                scores = softmax_probs[:, 1].tolist()
                
            all_scores.extend(scores)
            
        return all_scores

# 验证调用示例
if __name__ == "__main__":
    # 运行前提：已通过 optimum-cli 将模型导出为 onnx 格式
    # reranker = OptimizedReranker("bge-reranker-large.onnx", "BAAI/bge-reranker-large")
    # query = "Python 中如何避免多线程下的全局解释器锁阻塞？"
    # docs = [
    #     "深入理解 Python 全局解释器锁：GIL 诞生于 1990 年代...",
    #     "规避 Python GIL 阻塞的工程方案：CPU 密集任务使用 multiprocessing..."
    # ]
    # scores = reranker.compute_scores(query, docs)
    # print(scores)
    pass
```

---

## 5. 生产稳定性：超时熔断、保底降级与低分截断

在分布式微服务架构中，重排服务极易因为长文本切片过密或突发并发流量产生排队。若重排服务阻塞导致上游网关超时报错，整个 RAG 问答链路就会中断。因此，系统必须建立明确的熔断保底与门禁逻辑。

### 熔断与保底回退（Fallback）

1. **时间预算硬截止（Hard Timeout）**：网关在向下游重排微服务发起请求时，设定超时门限（如 80ms）。
2. **保底回退到粗排排序**：若重排微服务调用超时或报错，降级逻辑立即被触发。系统直接取粗排阶段的融合分数进行排序截断。在向量检索与 BM25 的[混合检索](/interview/rag-interview/rag-hybrid-retrieval.md)架构中，可以直接使用粗排计算好的倒数排名融合（RRF）分数：

$$\text{RRF Score}(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$

其中 $r_m(d)$ 为文档在检索通路 $m$ 中的名次，$k$ 为平滑常数（通常设为 60），$w_m$ 为各路权重。直接依据粗排分数截取 Top-3~5 传递给大模型，保证链路不因重排单点故障而中断。

![重排超时微服务熔断与粗排分数保底回退时序图](/images/content-rebuild/76bb185de848aee0/fig-5.png)

### 分数安全门禁（Score Threshold Gating）

重排模块除了重整名次，还承担着**相关性门禁**的职责。对于没有相关文档支撑的查询，如果不做拦截直接喂给大语言模型，极易引发虚假信息生成（即 [RAG 幻觉](/interview/rag-interview/rag-hallucination.md)）。

与双塔向量检索基于余弦相似度容易因高维向量聚集而产生虚高分数不同，Cross-Encoder 经过 Sigmoid 映射后的概率分布对于语义关联有更强的判别力：
* **低分截断拦截**：如果在经过 Rerank 打分后，候选集中所有文本块的最高得分依然低于安全阈值（例如经验值低于 0.25），则判定候选文档均无法支撑回答；
* **防御性阻断**：系统阻断将低质文本拼入 Prompt，直接返回预设的兜底回复（例如：“在当前知识库中未检索到能回答该问题的准确资料”）。

![重排边界条件：分数失真与长候选集队列阻塞分析](/images/content-rebuild/76bb185de848aee0/fig-6.png)

---

## 6. 面试追问与常见认知误区

### 追问 1：为什么不能把初筛候选集扩大到 500 个，让 Rerank 充分打分以追求极致的召回率？

**回答要点**：
这属于忽略了**计算边际收益**与**端到端延迟预算**的误区：
1. **耗时与算力成本剧增**：Cross-Encoder 的推理计算量随候选集大小 $K$ 呈线性增长，若候选集从 30 增加到 500，计算量扩大了十余倍。即便是使用 GPU 批处理，多批次推演也会将毫秒级耗时拖慢至秒级，直接击穿在线问答系统的交互 SLA；
2. **边际召回收益迅速收敛**：在向量检索与 BM25 混合粗排调优合理的前提下，大部分有效上下文已落入前 30~50 名。排在 100 名开外的文本块，引入背景噪音的概率远高于召回有效答案的概率；
3. **排队雪崩风险**：单次请求长时间占用 GPU 计算单元，在面对并发访问时会导致推理队列迅速堆积，引发整个检索链路超时雪崩。

### 追问 2：ColBERT 算不算纯粹的双塔模型？它与经典双塔的核心差异在哪里？

**回答要点**：
ColBERT **不属于**经典的单向量双塔模型，而属于**多向量晚期交互（Multi-Vector Late Interaction）**架构。它们的核心差异体现在表征粒度与交互逻辑上：
* **经典双塔**：通过池化层将整个序列无差别压缩进单个定长稠密向量（Single Embedding），序列中的细微约束、否定和短语映射在压缩中被平均化稀释；
* **ColBERT**：保留了序列中每个 Token 的独立向量表征，在打分时通过 MaxSim 算子显式执行词与词之间的软对齐匹配。它既保留了文档特征可在入库阶段预计算的性能优势，又保留了词级别的细粒度语义交叉，其排序表现明显优于传统单一向量双塔。

### 常见错误认知辨析

* ❌ **错误说法**：“重排模型打分直接看 Logit 绝对大小就行，各个模型阈值通用。”
  * **正解**：不同底座模型未归一化的 Logit 极值区间各不相同，且受输入序列长度偏置影响。生产中必须根据具体模型的输出结构做标准化（如 Sigmoid），并在业务离线评测集上标定绝对过滤阈值，避免误杀有效文档或放行无关噪声。
* ❌ **错误说法**：“引入了顶级的 Cross-Encoder 重排模型，粗排阶段就可以废弃 BM25 了。”
  * **正解**：重排无法对粗排没有召回的文档打分。面对版本号、报错错误码（如 `ORA-00942`）、特定人名或专有设备单号时，纯向量检索可能无法有效将其选入前 30 名候选集。粗排阶段保持“密集向量 + BM25 稀疏检索”的多样性，是后续重排模型发挥价值的必要前提。
