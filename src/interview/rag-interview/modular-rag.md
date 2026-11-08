---
title: 什么是 Modular RAG？如何通过解耦算子与动态编排实现灵活检索？
icon: mdi:puzzle
tag:
- Modular RAG
- RAG
- 模块化
description: 解析 Modular RAG 的核心思路：将固定检索流水线拆解为改写、检索、重排等独立算子，通过状态总线与动态路由、循环迭代编排应对复杂多跳问答。
---

在构建检索增强生成（RAG）系统时，最基础的实现通常是一条固定的线性流水线：用户输入问题 $\rightarrow$ 计算向量并检索文档 $\rightarrow$ 拼接提示词输入大模型（参考 [RAG 的完整工作流程是怎样的？核心步骤有哪些？](/interview/rag-interview/rag-complete-workflow.md)）。即使在进阶版本中加入了检索前改写与检索后重排序（参考 [Advanced RAG](/interview/rag-interview/advanced-rag.md)），整套流程依然是按固定顺序串行执行的。

然而，当面对真实的业务场景时，这种“一条路走到黑”的固定流水线很快就会遇到瓶颈。

例如用户提问：**“我们上季度华东区的实际销售额是多少？这是否达到了 Q3 战略会议纪要里设定的目标？”**

如果用固定流水线处理：
1. 直接拿整句话去向量数据库检索，往往只能搜出一些包含“华东区”或“销售目标”的零散文档，很难准确获取数据库中精确的“实际销售额数字”；
2. 用户的单次提问实际上包含了两个动作：先查结构化销售数据（报表/数据库），再查非结构化的会议纪要（文档库），并将两者比对。

为了灵活应对这种多数据源、多意图和多跳推理的场景，**Modular RAG（模块化 RAG）** 成为主流架构。它将固定的检索生成链路拆解为可插拔的独立算子，并引入编排机制，根据输入动态调度不同的处理路径。

![Modular RAG 架构全景展示](/images/content-rebuild/d5dc4a4329212bcf/cover.png)

---

## 1. 核心口述答案：面试中如何清晰回答

在面试中被问到“什么是 Modular RAG”时，可以按照以下层次组织回答：

> **口述答题参考**：  
> “Modular RAG 是在简单线性流水线受限后演进出的架构模式，其核心思想是**算子解耦**与**动态编排**。
> 
> 传统 RAG 将改写、检索、重排、生成等步骤硬编码为固定的顺序链条；而 Modular RAG 把每个步骤封装成独立、具备明确输入输出契约的算子（如 Rewrite、Search、Rerank、Memory）。
> 
> 在算子解耦的基础上，系统通过编排引擎（状态机或有向无环图）实现动态流转：根据用户意图进行**条件路由**（例如闲聊直接回复、数据统计查 SQL、政策咨询查文档），或者支持**循环迭代检索**（先查第一跳实体，再根据结果生成下一跳查询）。
> 
> 这样做的好处是各算子可以独立升级、测试和按需伸缩（例如高计算开销的 Rerank 独立部署在 GPU 上），并且能灵活组合出应对复杂业务场景的检索拓扑。”

![从单体流水线到模块化架构的演进](/images/content-rebuild/d5dc4a4329212bcf/fig-1.png)

---

## 2. 核心独立算子拆解：标准化契约与职责边界

在模块化架构中，算子（Operator）不依赖具体的上下文环境，而是通过统一的数据结构进行通信。常见算子包括：

![四大核心算子的功能与接口契约](/images/content-rebuild/d5dc4a4329212bcf/fig-2.png)

### 2.1 Search 算子（检索召回）
* **职责**：根据给定的检索词或条件，从底层索引中召回候选文档。
* **输入**：标准化的 Query 文本、元数据过滤条件（如部门、时间范围）、目标召回数量 $K$。
* **底层动作**：调用向量数据库执行稠密向量检索（参见 [向量数据库主要解决什么问题？](/interview/rag-interview/vector-database.md)），或调用搜索引擎执行 BM25 稀疏检索，亦或是两者结合的 [混合检索](/interview/rag-interview/hybrid-retrieval.md)。
* **输出**：候选文档列表 `List[CandidateChunk]`，包含文档内容、元数据及粗排初步得分。

### 2.2 Rerank 算子（语义重排）
* **职责**：对多路召回或粗排筛出的候选文档进行精细语义打分并排序。
* **输入**：原始 Query、待排序的候选文档列表 `List[CandidateChunk]`、最终保留数量 $N$。
* **底层动作**：通常调用 Cross-Encoder 模型将 Query 与每篇文档拼接后计算深度语义相关性得分（原理解析见 [什么是 RAG 中的 Rerank？](/interview/rag-interview/rag-rerank.md)）。
* **输出**：有序的文档列表 `List[ScoredChunk]`，包含归一化后的重排得分。

### 2.3 Rewrite 算子（查询改写与解构）
* **职责**：消除自然语言表达的模糊性，适配下游检索通道。
* **输入**：当前轮自然语言 Query、历史对话记录。
* **底层动作**：补全指代内容、扩展同义词（参见 [Query Rewrite 是什么？](/interview/rag-interview/query-rewrite.md)），或者将多意图复杂问题拆分为多个并行的子问题（Sub-queries）。
* **输出**：改写后的优化 Query，或分解后的子查询列表 `List[SubQuery]`。

### 2.4 Memory 算子（上下文与会话状态）
* **职责**：维护对话历史并向编排流程提供会话上下文。
* **输入**：Session ID、当前轮输入内容。
* **底层动作**：维护多轮交互滑动窗口、提取用户长期偏好或生成阶段性对话摘要，控制送入模型的上下文长度，减缓超出模型上下文窗口的风险。
* **输出**：注入到状态上下文中的会话约束与历史摘要。

<ReadMoreLock />

---

## 3. 动态流转编排：从固定顺序到条件图与迭代循环

算子独立后，系统的执行路径不再由代码中的固定调用链锁定，而是由编排引擎根据数据状态动态决定。

![动态路由与分支流转机制](/images/content-rebuild/d5dc4a4329212bcf/fig-3.png)

### 3.1 条件分支路由（Routing）
系统并不需要对所有输入都走一遍耗时巨大的重排与知识检索。路由算子依据输入特征将流量导向对应子链路：
* **直接生成分支**：针对日常寒暄或常识问答，无需调用检索直接由 LLM 生成；
* **结构化数据分支**：遇到统计、汇总类问题（如“华东区销售额”），路由到 Text-to-SQL 或内部报表 API；
* **非结构化文档分支**：遇到规章制度、方案说明类问题，路由到传统的“混合检索 + 重排”链路。

### 3.2 循环迭代检索（Iterative & Multi-hop Retrieval）
针对无法一次检索出全量信息的多跳推理问题，系统采用“检索 $\rightarrow$ 评估 $\rightarrow$ 补充检索”的闭环状态机。

以开头的案例为例：“华东区实际销售额是否达到了 Q3 战略会议纪要设定的目标？”系统需要：
1. 第一步：先调用数据接口查得实际销售额为“1.2 亿元”；
2. 第二步：将“华东区 Q3 销售额 1.2 亿元 达标 战略目标”组合为新 Query，在会议纪要文档库中检索；
3. 第三步：判定信息是否完整，若完整则进入生成算子，否则继续补充检索。

```python
# 循环迭代检索状态机逻辑（示意伪代码）
def iterative_rag_flow(initial_query: str, max_steps: int = 3) -> str:
    current_query = initial_query
    accumulated_context = []
    step = 0
    
    while step < max_steps:
        # 1. 召回与重排候选知识
        raw_chunks = search_operator.execute(query=current_query, top_k=6)
        scored_chunks = rerank_operator.execute(query=current_query, candidates=raw_chunks)
        accumulated_context.extend(scored_chunks)
        
        # 2. 评估当前已获取的上下文是否足以回答用户初始提问
        eval_result = judge_operator.evaluate(
            target_query=initial_query, 
            context=accumulated_context
        )
        if eval_result.is_sufficient:
            # 信息完备，流向最终生成节点
            return generate_operator.execute(query=initial_query, context=accumulated_context)
        
        # 3. 提取缺失信息槽位，生成下一跳子查询
        current_query = rewrite_operator.generate_next_hop_query(
            target_query=initial_query, 
            missing_info=eval_result.missing_reason
        )
        step += 1
        
    # 达到安全步数上限熔断退出，使用现有上下文进行兜底生成，并标注不确定性
    return generate_operator.execute(
        query=initial_query, 
        context=accumulated_context, 
        is_fallback=True
    )
```

![循环迭代检索与终止判定的状态机](/images/content-rebuild/d5dc4a4329212bcf/fig-4.png)

### 3.3 终止判定与安全熔断
循环检索如果缺乏边界控制，极易引发死循环或多次调用大模型产生高额账单。工业实践中必须设置三道防线：
1. **最大步数硬上限**（如上例中的 `max_steps = 3`）；
2. **单次信息增益阈值**：如果新一轮检索引入的内容与上一轮高度重合，判决算子判定收益不足，立即中止循环；
3. **全局超时控制**：整个 RAG 任务设置总耗时上限（如 3000ms），超时直接截断并基于已有材料降级输出。

---

## 4. 状态总线：算子间解耦的数据契约

若每个算子直接调用下游算子，系统会退化为网状调用关系。Modular RAG 通常依靠**统一状态上下文（Context State）**贯穿整个流转过程。

![上下文状态总线的数据契约流动](/images/content-rebuild/d5dc4a4329212bcf/fig-5.png)

### 4.1 核心状态数据结构示例
```json
{
  "trace_id": "trace-rag-20240501-001",
  "session_id": "sess-9021",
  "current_step": "POST_RERANK",
  "query_state": {
    "raw_query": "华东区实际销售额是否达到了 Q3 战略会议设定的目标？",
    "active_sub_queries": ["华东区 2023 Q3 销售考核目标 战略纪要"]
  },
  "retrieval_records": [
    {
      "chunk_id": "doc-meeting-q3-chunk4",
      "content": "华东区第三季度销售目标确定为 1.1 亿元整...",
      "source": "Q3_Strategy_Meeting.pdf",
      "score": 0.92
    }
  ],
  "intermediate_scratchpad": {
    "extracted_facts": {
      "actual_sales": "1.2 亿元 (来自财务报表接口)"
    }
  },
  "budget": {
    "max_retries": 2,
    "current_step": 1,
    "timeout_deadline_ms": 1714550000
  }
}
```

### 4.2 状态驱动的工程价值
* **全链路可观测性**：每次状态变更作为日志或事件沉淀，排查回答异常时，可以清晰定位是 Rewrite 算子改偏了、Search 算子没召回，还是 Rerank 算子排错了顺序。
* **无副作用与可重试**：算子尽量设计为无副作用函数，接收状态快照，产出计算结果写入上下文。某一节点发生网络异常时，可在该节点安全重试，无需全链路重头执行。

---

## 5. 单体流水线与模块化架构的选型权衡

模块化虽然灵活，但并不适合所有场景。拆分微服务与增加编排层必然带来工程开销：

| 维度 | 单体流水线（Linear Pipeline） | 模块化架构（Modular RAG） |
| :--- | :--- | :--- |
| **系统形态** | 进程内单体调用，顺序串行 | 算子独立服务化 / 有向图编排引擎 |
| **扩展与伸缩** | 无法针对局部热点伸缩，GPU/CPU 混布 | 算子独立伸缩（如 GPU Rerank 独立扩缩容） |
| **分支与灵活性** | 依靠代码内硬编码 `if-else`，难以扩展复杂分支 | 支持动态路由、并发汇聚与循环迭代 |
| **网络与调用开销** | 极低（进程内对象引用传递） | 若拆分为微服务独立部署，存在网络往返与序列化开销（每次 RPC 增加数毫秒至数十毫秒） |
| **维护与排错** | 逻辑紧凑，单机断点易复现 | 链路较长，高度依赖分布式追踪（Tracing）系统 |
| **适用场景** | 内部原型验证、单一问答场景、逻辑简单的 FAQ | 多数据源协同、多跳复杂推理、具备一定规模的企业级平台 |

![单体架构与模块化微服务架构的生产部署对比](/images/content-rebuild/d5dc4a4329212bcf/fig-6.png)

---

## 6. 面试追问与高频避坑

### 追问 1：将算子拆解为分布式服务后，网络开销增加了端到端耗时，工程上如何优化？
**考核点**：对分布式微服务落地在低延迟场景下的工程优化积累。
* **高性能 RPC 协议**：算子内部通信避免使用臃肿的 JSON-over-HTTP，改用基于 HTTP/2 的 gRPC 或基于 Protobuf 的序列化协议，降低传输体积与编解码时间；
* **并发分发与分支预取**：对于分解出的多个子查询，使用线程池或协程并发向底层索引发起请求，最后通过归并算子聚合，掩盖单次查询耗时；
* **旁路缓存（Cache）与快速通道**：在满足数据时效要求与权限隔离的前提下，对高频且语义判定等价的请求复用缓存直接返回；简单问题经由路由算子跳过改写和重排，走低延迟直通分支。

### 追问 2：如果路由算子（Router）意图识别失误，导致后续全部失败，如何做防御？
**考核点**：复杂流程中的容灾兜底与降级设计。
* **置信度模糊多路召回**：当路由分类器的概率分布较为平均（例如判断为 Text-to-SQL 的置信度与文档检索的置信度接近）时，编排引擎不武断二选一，而是**并发触发双路查询**，由下游精排算子统一筛选，避免前端单点判断失误截断数据源；
* **异常降级与主干回退**：专用通道（如 Text-to-SQL 算子）解析失败或语法执行报错时，捕获异常并将状态流转至通用文档检索主干分支，输出通用知识并提示用户数据统计异常，防止直接抛出 500 错误。

### 常见错误认知清单
* **误区一：把模块化等同于“写了几个 Python 类”。**  
  *纠偏*：模块化的关键是**契约解耦与动态图调度**。如果虽然拆分了类，但仍然在业务主函数中硬编码了固定的 `A() -> B() -> C()` 顺序，逻辑分支依然耦合在业务层，系统并未获得动态编排与弹性扩缩容的能力。
* **误区二：循环检索比单次检索效果好，所以默认都开循环检索。**  
  *纠偏*：循环迭代检索每多走一步，都会带来额外的 LLM 调用与检索延迟，端到端耗时可能从 500ms 飙升至数秒。实际工程中只有经判别确认信息存在多跳依赖或缺失时才触发，并必须严格配置最大步数熔断。

---

## 7. 读者自测与实战验收

1. 在面对包含“结构化统计”和“非结构化制度文档”的双重提问时，Modular RAG 的路由算子与编排引擎通常如何协作处理？
2. 为什么在设计循环迭代检索（Iterative Retrieval）时，必须配置最大步数和信息增益退出机制？
