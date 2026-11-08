---
title: 生产级 RAG 系统的完整流程与容灾降级怎么做？
icon: mdi:file-tree-outline
tag:
- RAG
- 完整流程
- 检索增强
description: 从状态机流转、超时预算切分、向量检索回退到 BM25、重排降级至加权 RRF 及相关度门禁，详解生产 RAG 的高可用工程设计。
---

在面试中被问及“RAG 系统的完整流程”时，只回答线性的“切分-入库-检索-重排-组装-生成”通常只能证明了解基础 demo。面试官真正考察的，是系统面对**外部依赖抖动与不可用**时的工程防御能力：当向量数据库查询超时、重排模型显存排队、无相关文档召回或大模型限流时，系统如何避免请求雪崩并保证返回质量？

在上一篇 [/interview/rag-interview/rag-complete-workflow.md](/interview/rag-interview/rag-complete-workflow.md) 中，我们梳理了 RAG 的基本流程。本文将视角切换到生产运行环境，结合具体的企业知识库查询示例，分析 RAG 的有限状态机流转、超时预算分配与降级分支设计。

![生产级 RAG 状态机与容灾降级全景图](/images/content-rebuild/e305f4a930e45f7a/cover.png)

---

## 1. 核心口述答案：状态流转与三条降级分支

面试官提问：“请描述生产环境下 RAG 系统的运行时流转；如果外部依赖发生高延迟或故障，你会如何设计降级？”

建议口述结构如下：

> “生产环境下的 RAG 流程不应写成单向无分支的线性脚本，而应作为具备明确状态流转与兜底分支的状态机来运行。主干流包含四个核心阶段：**查询预处理、混合检索、重排过滤、上下文生成**。
>
> 为保障系统 SLA 与接口可用性，我们在三个关键节点设计了超时控制与降级逻辑：
> 1. **检索层降级**：通过并发执行向量检索（Dense）与关键词检索（BM25）。若向量数据库超时（如超过 150ms 预算），系统取消向量任务，降级仅使用已返回的 BM25 词法结果，保障基础召回；
> 2. **重排层降级**：Cross-Encoder 重排服务若发生 HTTP 5xx 或处理超时，系统绕过深度模型计算，退回至基于排序位置的无参数加权互惠排名融合（Weighted RRF），用微秒级内存排序替代百毫秒级的模型推理；
> 3. **相关度门禁拦截**：候选文档打分若低于预设门槛，直接流转至确定性拒答或人工客服分支，阻止低质、无关文档进入 Prompt，从输入端遏制模型幻觉。
>
> 全链路引入超时预算（Timeout Budget）传递，通过 context deadline 逐级扣减，防止单点耗时拖垮整个链路。”

![核心口述体系：主干正常流转与三条容灾逃逸分支](/images/content-rebuild/e305f4a930e45f7a/fig-1.png)

---

## 2. 运行时状态机：请求全生命周期与超时预算

我们以企业 HR 知识库中的真实查询为例：“**2024 年企业年金单位缴费比例是多少？**”。

如果将检索和模型调用串联在无异常捕获的脚本中，一旦下游向量库执行分片重新均衡或网络丢包，整个 HTTP 请求就会挂起，最终导致网关 504 错误。生产系统的执行引擎通常定义如下状态转移表：

| 状态名称 | 前置条件 | 正常迁移动作 | 异常/超时迁移 | 贯穿示例（年金缴费比例）处理 |
| :--- | :--- | :--- | :--- | :--- |
| `STATE_VALIDATION` | 收到客户端请求 | 参数与安全检查通过 $\rightarrow$ `STATE_REWRITE` | 参数非法 $\rightarrow$ 直接返回 400 终态 | 验证 Query 非空、符合字符长度要求 |
| `STATE_REWRITE` | 请求合法 | 改写成功 $\rightarrow$ `STATE_RETRIEVAL` | 改写超时 $\rightarrow$ 降级使用原始 Query 并行检索 | 补全实体上下文，提取关键词“企业年金 单位缴费 比例” |
| `STATE_RETRIEVAL` | Query 已就绪 | 检索完成 $\rightarrow$ `STATE_RERANK` | 向量库熔断 $\rightarrow$ 仅使用 BM25 词法召回结果；双路均超时 $\rightarrow$ 查静态缓存 | 向量库查语义嵌入，BM25 查精确词；若向量库超时，保留 BM25 结果 |
| `STATE_RERANK` | 粗排候选集返回 | 计算重排得分 $\rightarrow$ `STATE_GATE` | 重排超时/宕机 $\rightarrow$ 退化至 RRF 粗排截断 Top-K | Cross-Encoder 打分；若服务超 300ms，按 RRF 顺序截取前 5 篇 |
| `STATE_GATE` | 候选集已排序 | 最高分 $\ge$ 阈值 $\rightarrow$ `STATE_ASSEMBLY` | 最高分 $<$ 阈值 $\rightarrow$ `STATE_REFUSAL` | 检查前几篇文档是否包含年金相关政策，低于阈值直接拒答 |
| `STATE_ASSEMBLY` | 检索文档过审 | 拼装提示词 $\rightarrow$ `STATE_STREAMING` | 上下文超长 $\rightarrow$ 动态滑动窗口截断后再组装 | 将年金管理办法相关条款组装进 Prompt，预留生成空间 |
| `STATE_STREAMING` | Prompt 就绪 | 首字流式吐出 $\rightarrow$ `STATE_COMPLETED` | 模型限流/断流 $\rightarrow$ 切备用渠道或返回标准兜底文案 | 逐字推流生成答案，监听用户断开事件 |

<ReadMoreLock />

![运行时状态转移图（FSM）：状态节点、转移事件与终态](/images/content-rebuild/e305f4a930e45f7a/fig-2.png)

### 链路超时预算（Timeout Budget）的传播机制

在分布式链路中，各阶段共享全局超时预算。假设网关对外承诺的端到端 SLA 上限为 $2000\text{ ms}$：

$$\text{Budget}_{\text{Total}} = T_{\text{Rewrite}} + T_{\text{Retrieval}} + T_{\text{Rerank}} + T_{\text{LLM\_TTFT}} + T_{\text{Buffer}}$$

当请求到达网关时生成全局 TraceID 与 Deadline 时间戳：
1. 若 `STATE_REWRITE`（如调用小参数量模型进行扩展，见 [/interview/rag-interview/query-rewrite.md](/interview/rag-interview/query-rewrite.md)）消耗了 $300\text{ ms}$，传递给检索层的可用预算自动扣减为剩余值；
2. 为检索层设定的单阶段超时上限为 $150\text{ ms}$。一旦超过此阈值，非阻塞客户端立即抛出超时信号并取消挂起的请求；
3. 这样可防止上游各阶段将延迟挤压给最后的大模型生成阶段，确保留给大模型首字返回（TTFT）的时间始终在安全预算内。

---

## 3. 向量库检索超时与服务降级：回退到关键词索引

在基础教学中，检索常被简化为仅查询向量数据库（原理见 [/interview/rag-interview/vector-database.md](/interview/rag-interview/vector-database.md)）。但在生产环境中，向量数据库（如基于 HNSW 索引的节点）在大规模写入、内存压缩或节点垃圾回收时，P99 延迟可能出现抖动。

![检索层超时熔断与双路降级拓扑](/images/content-rebuild/e305f4a930e45f7a/fig-3.png)

### 双路检索与熔断判定

生产系统通常采用 [/interview/rag-interview/rag-hybrid-retrieval.md](/interview/rag-interview/rag-hybrid-retrieval.md) 讨论的双路并行召回架构。主路为稠密向量检索（Dense Retrieval），辅路为基于倒排索引的词法检索（BM25）：

1. **双路并发请求**：同时向向量服务与 BM25 服务发起异步调用，统一设置 $150\text{ ms}$ 的硬超时。
2. **向量超时降级**：若向量检索超时未返回，系统断开其任务，直接使用已返回的 BM25 词法结果；对于包含专有名词的查询（如“企业年金”），BM25 往往能准确命中包含该专有名词的政策条例文档。
3. **全路失败兜底**：若向量与 BM25 服务均不可用，系统状态机转移至兜底分支，查询预置的高频问答缓存（如 Redis 中的精确匹配问答对）；若未命中缓存，则转入确定性拒答，不继续向下游调用。

```python
import asyncio
from typing import List, Dict, Any

async def hybrid_retrieval_with_fallback(
    query: str, 
    vector_client, 
    bm25_client, 
    timeout_ms: int = 150
) -> Dict[str, Any]:
    """
    并发执行稠密向量检索与稀疏 BM25 检索，包含超时截断与降级处理
    """
    timeout_sec = timeout_ms / 1000.0
    
    dense_task = asyncio.create_task(vector_client.search(query, top_k=20))
    sparse_task = asyncio.create_task(bm25_client.search(query, top_k=20))
    
    dense_results: List[Dict[str, Any]] = []
    sparse_results: List[Dict[str, Any]] = []
    
    # 并发等待双路结果，受统一超时预算控制
    done, pending = await asyncio.wait(
        [dense_task, sparse_task],
        timeout=timeout_sec
    )
    for task in pending:
        task.cancel()
    
    if dense_task in done and not dense_task.cancelled():
        try:
            dense_results = dense_task.result()
        except Exception:
            dense_results = []
    
    if sparse_task in done and not sparse_task.cancelled():
        try:
            sparse_results = sparse_task.result()
        except Exception:
            sparse_results = []
    
    # 根据各路召回情况确定输出状态
    if dense_results and sparse_results:
        return {"status": "NORMAL_HYBRID", "dense": dense_results, "sparse": sparse_results}
    elif sparse_results:
        return {"status": "DEGRADED_SPARSE_ONLY", "dense": [], "sparse": sparse_results}
    elif dense_results:
        return {"status": "DEGRADED_DENSE_ONLY", "dense": dense_results, "sparse": []}
    else:
        return {"status": "RETRIEVAL_EXHAUSTED_FALLBACK", "dense": [], "sparse": []}
```

---

## 4. 重排模型不可用时的排序退化：从 Cross-Encoder 到加权 RRF

重排（Rerank）服务（详细原理见 [/interview/rag-interview/rag-rerank.md](/interview/rag-interview/rag-rerank.md)）多基于 Cross-Encoder 架构。由于需要将 `Query` 与 `Document` 拼接后输入模型做全注意力交互，计算复杂度为序列长度的平方级。重排服务一旦并发积压或发生显存溢出（OOM），往往是整条链路中最容易超时的环节。

![重排模型不可用时的拓扑退化机制](/images/content-rebuild/e305f4a930e45f7a/fig-4.png)

### 降级方案：切换至加权互惠排名融合（Weighted RRF）

当 `STATE_RERANK` 检测到 Cross-Encoder 接口返回 5xx 错误、连接池耗尽或处理耗时超过预算（如 $300\text{ ms}$）时，系统应跳过深度模型计算，切换至无参数的加权互惠排名融合（Weighted RRF，参考 [/interview/rag-interview/hybrid-retrieval.md](/interview/rag-interview/hybrid-retrieval.md)）：

$$RRF\_Score(d \in D) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$

其中：
- $M$ 表示当前可用的召回通路集合（例如 Dense 路与 Sparse 路）；
- $w_m$ 是通路 $m$ 的权重（例如当测试集上向量检索质量更高时可设 $w_{\text{dense}}=1.5, w_{\text{sparse}}=1.0$）；
- $r_m(d)$ 是文档 $d$ 在该通路排序结果中的名次（从 $1$ 开始计数）；
- $k$ 是平滑常数（在信息检索实践中通常取 $60$），用于避免前几名排位分数差过大。

**算例对照（假设 $k=60$）：**
- 文档 A（《企业年金实施办法》）在 Dense 路排第 1 名，在 Sparse 路排第 4 名：
  $$RRF(A) = \frac{1.5}{60 + 1} + \frac{1.0}{60 + 4} = \frac{1.5}{61} + \frac{1.0}{64} \approx 0.02459 + 0.01563 = 0.04022$$
- 文档 B（《年金方案备案指南》）在 Dense 路排第 8 名，在 Sparse 路排第 1 名：
  $$RRF(B) = \frac{1.5}{60 + 8} + \frac{1.0}{60 + 1} = \frac{1.5}{68} + \frac{1.0}{61} \approx 0.02206 + 0.01639 = 0.03845$$

依据加权 RRF 得分排序，文档 A 排在文档 B 前面。计算过程只涉及简单的排位倒数求和与数组排序，耗时通常在 $1\text{ ms}$ 以内。虽然相比 Cross-Encoder 损失了部分深层语义交互收益，但在重排节点异常时能快速产出候选集，保住下游生成链路。

---

## 5. 相关度门禁与拒答拦截：阻断低质上下文

很多初学者系统无论检索到什么文档，都会直接拼装进 Prompt 送给模型。这种做法在遇到知识库未收录的问题时极易导致严重幻觉（参见 [/interview/rag-interview/rag-hallucination.md](/interview/rag-interview/rag-hallucination.md)）。

![相似度阈值判定与拒答/兜底分支分流机制](/images/content-rebuild/e305f4a930e45f7a/fig-5.png)

### 为什么不能仅靠 Prompt 里的“不知道就说不知道”？

在 Prompt 中加入“如果参考资料中没有提及，请回答不知道”，并不能完全杜绝幻觉：
1. **成本浪费**：无论模型是否拒答，拼装进入 Prompt 的低质文档都已产生网络传输与 Prefill 阶段的 Token 计算开销；
2. **模型注意力偏置**：大语言模型具有顺应提问的倾向，若上下文包含哪怕弱相关的干扰词句，模型也可能捕获到片面线索并推测编造事实。

### 门禁逻辑实现

在 `STATE_GATE` 阶段应设置两道拦截标准：
- **最高分数硬门禁**：若正常经过 Cross-Encoder 重排，输出的归一化相关度分数最高值仍低于安全阈值（例如经过 Sigmoid 映射后的相关分 $< 0.40$）；或在重排降级场景下，向量相似度最高值低于硬阈值（如余弦相似度 $< 0.60$），系统判定知识库内无对应事实基础。
- **分流动作**：
  - **确定性合规拒答**：状态机立即终止向大模型流转，直接返回配置好的静态应答（如：“知识库未检索到关于 2024 年企业年金缴费比例的有效制度，请核对提问或咨询人力资源部相关专员”）。端到端耗时在检索完成后即刻结束，不产生大模型生成调用成本；
  - **开放域兜底（可选配置）**：若业务允许开放生成，系统需明确剥离“企业内部知识库”的认证标识，并在回复前追加声明：“以下内容来自模型公开知识，非公司正式制度认证”，避免误导业务决策。

---

## 6. 生产可观测性与验收指标

讨论生产架构时，不能停留在“系统会降级”的定性描述，需要结合具体的工程监控指标（评估方法可参考 [/interview/rag-interview/rag-evaluation.md](/interview/rag-interview/rag-evaluation.md)）：

![生产级可观测指标看板](/images/content-rebuild/e305f4a930e45f7a/fig-6.png)

### 关键指标定义与基线设定

1. **端到端 P99 延迟分位线**：
   - **目标示例**：在正常状态下，从用户发起请求到首字流式输出（TTFT）的 P99 延迟控制在 $1500\text{ ms} \sim 1800\text{ ms}$ 区间；
   - **验收行为**：当对向量数据库施加压力导致其超时率升高时，由于断路器和降级逻辑生效，端到端 P99 延迟应收敛在降级路径上限（如 BM25 兜底路径下 P99 应保持在 $800\text{ ms}$ 以内），不得无限排队累积。
2. **降级触发率（Fallback Trigger Rate）**：
   $$\text{Fallback Rate} = \frac{\text{降级请求数}}{\text{总请求数}} \times 100\%$$
   - **稳态监控**：在正常网络与负载下，降级触发率通常应低于 $0.1\%$。若该指标短时间内上升至 $2\%$ 以上，说明向量检索或重排服务存在网络抖动、分片异常或硬件资源瓶颈。
3. **拒答精确率（Refusal Precision）**：
   - 在评测集中准备一定比例知识库范围外的测试样本。系统在 `STATE_GATE` 阶段应将这些样本精准拦截。若发生拒答，必须确认其确实属于未收录或弱相关问题，而非正常语义误杀。
4. **降级召回衰减度（Recall@K Drop Rate）**：
   - 评估重排服务不可用、退化为 RRF 排序时，Top-5 召回中相关文档的衰减幅度。通常要求 RRF 降级路径下的相关文档召回率相较正常重排路径的衰减控制在可接受范围（如不超过 $10\% \sim 15\%$），以验证粗排与融合算法的底线承载能力。

---

## 7. 面试深度追问与常见误区

### 追问 1：如果大模型生成阶段客户端中途断开连接，系统如何处理？

**解答**：
流式传输通常基于 HTTP chunked 或 SSE（Server-Sent Events）。网关服务应监听客户端连接关闭信号：
- 一旦检测到客户端断开（如用户关闭了标签页），网关应通过 `context.cancel()` 向上游大模型推理引擎发送中断信号；
- 推理后端收到信号后停止该请求后续 token 的生成计算，并释放对应的 KV Cache 显存资源，防止算力空转浪费；
- **资源与计费边界**：中断只能阻止后续未生成 Token 的计算开销。对于中断前已经完成的 Prefill 输入阶段以及已经流式吐出的 Token，底层推理服务或第三方 API 通常仍会正常计量计费，实际消耗的算力与 Token 配额不会无条件全额释放或返还。

### 追问 2：为什么有些团队不用向量相似度，而优先用重排分数做门禁拦截？

**解答**：
双塔向量模型（Bi-Encoder）将 Query 与 Document 独立编码为定长向量，捕捉的是粗粒度的全局语义空间相似度，对“不缴费”、“全额缴费”、“缴费比例”等局部否定词或细微修饰词不够敏感。而 Cross-Encoder 重排模型对 Query 和 Document 进行完整注意力交互，更能捕获细粒度的逻辑匹配关系。因此，使用 Cross-Encoder 的输出分数做门禁拦截，其误杀率和漏杀率通常显著低于双塔向量余弦相似度。

### 常见误区辨析

- **误区一**：“向量库挂掉时，直接让大模型根据自带的通用预训练知识来回答即可。”
  - **辨析**：企业私有知识库（如内部政策、技术方案、财务报表）通常从未出现在通用预训练数据中。此时让模型自由发挥，极易产生具有欺骗性的幻觉。正确的做法是降级为关键词检索；若检索全部失效，应直接合规拒答并提供人工支持入口。
- **误区二**：“重排模型太耗时，高可用系统为了保证低延迟绝对不能引入重排。”
  - **辨析**：重排能显著改善进入生成阶段的文档相关性。在工程上，可以通过“控制候选集规模（如仅重排 Top-15）+ 设置硬超时（如 250ms）+ 超时退化至 RRF”的方式来平衡效果与可用性，而非直接放弃重排。
