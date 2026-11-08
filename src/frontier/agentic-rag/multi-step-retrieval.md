---
title: 为什么单次检索经常不够用？RAG 如何做查询改写、补充检索与验证
icon: mdi:magnify-expand
tag:
- Agentic RAG
- 多步检索
- 查询改写
- 答案验证
description: 单次检索易因术语差异或指代缺失而失效。本文通过具体用例，讲解证据评估、查询改写、增量合并及事实校验的循环检索机制与防死循环设计。
---

传统的检索增强生成（RAG）流水线通常是一次性的：用户输入问题，系统转为向量进行检索，取 Top-K 文档拼接成上下文，交由大语言模型生成答案。在上一讲[/frontier/agentic-rag/what-is-agentic-rag.md](/frontier/agentic-rag/what-is-agentic-rag.md)中，我们区分了线性 RAG 与带状态控制的系统：引入 Agent 机制的核心目的，不是为了套用复杂的外部框架，而是在检索与生成之间加入**状态记录、证据评估与重试控制**。

线性流水线预设了一个前提：**单次召回的文档块一定包含回答问题所需的完整事实**。但在实际业务中，用户提问往往不够规范，知识库的文档切分也各不相同，单次检索经常出现遗漏。如果系统不做检查就直接生成，模型只能基于残缺信息强行推理，从而产生看似合理实则错误的猜测。

我们来看一个贯穿全文的具体例子：用户在运维支持系统提问：“**升级到 v2.4 之后接口报 403 怎么排查？**”
- 系统初次检索召回了两篇文档：《通用 403 权限排查手册》和《v2.4 版本发布说明》；
- 《通用 403 权限排查手册》讲解的是账号权限与 IP 白名单，完全没提 v2.4 的变化；
- 《v2.4 版本发布说明》只写了“安全加固：网关层启用双向认证 mTLS”，没有出现“403”字样。

如果只做一次检索，模型拿到这两篇文档，要么回答“请检查 IP 白名单”（错误建议），要么直接回答“文档中未提及”。但在循环检索流程中，系统会先检查证据是否充足，发现缺少“v2.4 与 403 报错之间的关联因果”，接着自动把查询改写为“v2.4 网关 mTLS 证书 403 报错”，触发二次检索拿到《mTLS 客户端证书 SAN 扩展缺失报错排查》，最终拼出准确答案并完成校验。

本文聚焦于**围绕单一明确问题展开的检索、判定、改写与补充流程**。跨文档的复杂子任务拆解将在后续的[/frontier/agentic-rag/complex-query-planning.md](/frontier/agentic-rag/complex-query-planning.md)中讨论。

![循环检索机制全景图](/images/content-rebuild/375b8b80741ff721/cover.png)

---

## 单次检索的三种典型失效场景

单次检索拿不到关键证据，通常不是数据库离线，而是用户输入与知识库表达之间存在逻辑差距。最常见的有以下三类场景：

### 1. 词表不匹配（Vocabulary Mismatch）
用户提问习惯使用日常口语或现象描述，而知识库通常使用标准术语、缩写或代码符号。例如，用户提问“数据库写入太慢怎么自动切备机”，而核心文档的标题是“心跳超时与 failover 仲裁阈值配置”。
- 稠密向量检索（Dense Retrieval）会将多词语义压缩为固定维度向量，容易被泛化的“数据库性能调优”文档吸引，反而忽略特定的切换配置；
- 稀疏关键词检索（如 BM25）则因为没有字符重叠而直接漏检。关于不同检索机制的权衡，可参考[/interview/rag-interview/retrieval-methods.md](/interview/rag-interview/retrieval-methods.md)。

### 2. 上下文缺失与指代模糊（Context Ambiguity）
用户的单句输入通常缺少全局约束。例如提问“升级后报错 403 怎么排查？”，句中没有说明是哪个业务组件、什么协议、从哪个版本升上来。检索器在索引库中搜索“403”，会同时召回对象存储权限、网页代理、网关认证等数十篇互不相干的文档，有效信息被噪声淹没。

### 3. 弱区分度文档块干扰（Weak-discriminative Distraction）
知识库文档在切分时常以固定长度或段落截断。很多召回的文档块仅包含功能背景、配置项名称列表或法律声明，字面与提问高度重合，但缺少核心的因果细节（如“默认值”、“先决条件”或“已知异常”）。模型拿到这类文档，往往陷入“看似查到了，其实没法下结论”的状态。

![单步检索失败的三类典型场景对比](/images/content-rebuild/375b8b80741ff721/fig-1.png)

| 失败场景 | 输入特征 | 召回结果特征 | 传统单步 RAG 表现 | 循环系统应对机制 |
| :--- | :--- | :--- | :--- | :--- |
| **词表不匹配** | 口语化、高层抽象、使用别名 | 语义泛化、缺少具体参数的文档 | 输出空泛建议，无法定位问题 | 识别未命中术语，自动补齐同义词与技术代号 |
| **上下文缺失** | 缺少环境、组件、版本前置信息 | 跨组件杂乱文档混合召回 | 拼凑出无关模块的错误答案 | 判定缺失约束，改写查询提取核心实体 |
| **弱区分度干扰** | 包含提问里的关键词 | 背景套话居多，缺乏具体因果 | 强行根据无依据片段做猜测 | 判定证据不充分，定向补充核心排查事实 |

---

## 检索证据充分性判定：如何判断“查够了没有”？

在把检索结果交给模型写出最终答案前，需要设立一道**判定节点**。如果当前拿到的文档足以回答问题，则进入生成环节；如果关键证据残缺，则提取出缺失的信息点，交由下游重试。

判定证据是否充足，不能仅让大模型输出简单的“好”或“不好”，而需要约定结构化的输入与输出格式（Contract），从三个具体维度进行审查：

1. **实体覆盖度（Entity Coverage）**：提问涉及的主体、动作、环境版本是否在文档中都有明确描述？
2. **因果链条完整度（Causal Completeness）**：若用户问的是“排查步骤”或“配置原因”，文档是否给出了推导逻辑，而不仅仅是一句结论？
3. **信息一致性（Evidence Consistency）**：召回的多篇文档之间是否存在相互冲突的版本说明或参数定义？

判定节点的结构化输出示例如下：

```json
{
  "is_sufficient": false,
  "confidence_score": 0.45,
  "covered_aspects": ["确认了 403 报错属于网关鉴权阶段"],
  "missing_aspects": ["缺少 v2.4 版本下 mTLS 证书校验规则与客户端配置要求"],
  "decision": "REWRITE_AND_SUPPLEMENT"
}
```

![证据充分性判定的结构化处理流程](/images/content-rebuild/375b8b80741ff721/fig-2.png)

通过强制模型输出缺失清单（`missing_aspects`），下游的改写节点才能明确知道“接下来该去搜什么”，避免盲目换词或无目的地循环。

<ReadMoreLock />

---

## 查询改写：根据已知信息生成差异化检索词

当判定节点给出 `decision: REWRITE_AND_SUPPLEMENT` 时，系统不能用原来的句子去重新检索。底层索引库与检索算法在输入不变时会返回完全一致的结果，二次检索必须具备**差异性（Variance）**。

查询改写通常接收三项输入：
- **原始问题（Original Query）**：锚定用户的核心诉求；
- **已检索上下文摘要（Retrieved Context Summary）**：说明哪些事实已经查明，避免重复检索同质文档；
- **缺失维度列表（Missing Aspects）**：明确本次补充检索必须攻克的目标。

![查询改写机制与差异化检索词生成](/images/content-rebuild/375b8b80741ff721/fig-3.png)

在工程落地中，改写通常生成两种检索输入：
1. **面向关键词与稀疏索引的布尔表达式**：将口语转化为精准的技术术语。例如把“升级后 403”改写为 `"v2.4" AND "mTLS" AND ("client certificate" OR "SAN")`。在混合检索场景下，这一步可以补足稠密向量对生僻专有名词不敏感的短板，混合检索的配置细节可参考[/interview/rag-interview/hybrid-retrieval.md](/interview/rag-interview/hybrid-retrieval.md)。
2. **面向稠密向量的假设性文档或描述句（HyDE）**：由模型根据缺失维度，生成一段符合技术手册风格的短陈述，再对该陈述计算向量进行检索，缩短口语提问与书面知识库之间的表征距离。

---

## 补充检索与状态合并：去重、打分与窗口截断

二次检索会带回一批新的文档块。此时不能直接覆盖原有的检索结果，因为初次检索可能已经拿到了部分有用信息，完全覆盖会导致信息丢失。

系统需要维护一个状态容器，对新增的文档执行**去重、合并与窗口截断**：

```
   初次检索文档池 D_0
         │
         ├── 过滤完全重复的文档块 (按 hash(chunk_content))
         ▼
   增量候选集 D_delta (二次检索召回)
         │
         ▼
   合并证据池 D_merged = 去重(D_0 ∪ D_delta)
         │
         ▼
   重排模型 (Cross-Encoder Reranker) 统一打分
         │
         ▼
   窗口裁剪: 取最终得分最高的前 Top-K 片段输入生成器
```

![增量文档合并、去重与重排窗口控制](/images/content-rebuild/375b8b80741ff721/fig-4.png)

在状态维护中，最关键的限制是**控制上下文长度**。如果每一轮检索都无上限地向后追加，不仅会导致 API 费用和响应耗时线性增加，还会让模型在过长的上下文中出现“中间遗忘”（Lost in the Middle）现象。因此，状态中必须设置文档数量或 Token 上限，并在每轮通过重排模型淘汰得分较低的边缘片段。

---

## 答案事实校验：验证推论并过滤未支持内容

当证据集最终判定充足并生成候选答案后，大语言模型仍有可能借助其预训练参数中的外部记忆，填补文档中没有提到的细节。为了保证企业问答的严谨性，需要对生成的文本进行**反向事实核验（Attribution & Entailment）**：

1. **命题切分**：将候选答案拆分为独立的陈述句命题集合 $\{S_1, S_2, \dots, S_n\}$；
2. **自然语言蕴含推理（NLI）**：针对每一个命题 $S_i$，在最终证据池中寻找能够支撑它的文档片段；
3. **状态标记**：
   - **Supported（有据可查）**：文档中有明确证据支持；
   - **Contradictory（与事实相悖）**：与文档描述存在直接矛盾；
   - **Unsupported（无证据推断）**：文档未提及，属于模型自行发挥。

![答案事实校验与逐句归因图](/images/content-rebuild/375b8b80741ff721/fig-5.png)

一旦检测到 `Unsupported` 或 `Contradictory` 的句子，系统会执行校正逻辑：要求模型剔除未经证实的推论，或在句子后明确标注“当前文档未给出明确依据”。如果核心推论全部无依据，则返回“信息不足，无法确认”。

---

## 防御性工程设计：防止死循环与控制运行开销

循环检索让系统具备了重试与自我修正的能力，但也带来了工程风险：**如果模型始终判定证据不足，改写后又搜不到新内容，系统容易陷入死循环，耗尽 Token 额度并引发接口超时（HTTP 504）**。

为了保证系统在生产环境中的确定性，必须在控制流代码中加入四道防御机制：

1. **最大重试步数硬上限（Max Steps Bound）**：无论模型判断是否充足，循环步数达到预设值（通常建议为 2 次重试）必须无条件退出循环。
2. **查询振荡检测（Query Oscillation Detection）**：比对当前改写词与历史检索词的相似度（如字符 Jaccard 相似度或向量余弦值）。若相似度过高，说明模型已无法提出新的检索角度，立即中止重试。
3. **增量收益递减检测（Information Gain Decay）**：若新一轮检索经过去重后，没有引入任何新文档，或者重排最高分没有高于历史记录，说明已触及当前知识库的检索边界，停止重试。
4. **兜底降级响应（Fallback Strategy）**：当触发上限跳出循环时，系统绝不抛出未经处理的异常，而是基于当前已累积的所有文档生成受限回答，并附带说明“已达到最大重试次数，以下内容根据现有部分文档整理，缺失：...”，提示人工介入。关于检索失败的常见排查手段，可参阅[/interview/high-freq-scenarios/rag-retrieval-failure.md](/interview/high-freq-scenarios/rag-retrieval-failure.md)。

![循环检索状态机熔断流转图](/images/content-rebuild/375b8b80741ff721/fig-6.png)

下面给出一个完整的最小可运行原型代码，演示了带有步数限制、去重合并与充分性判定的循环检索状态机：

```python
from typing import List, Dict, Any

# 模拟静态文档库
MOCK_KNOWLEDGE_BASE = [
    {"id": "doc_01", "text": "用户升级至平台 v2.4 之后，网关将强制开启双向认证机制。"},
    {"id": "doc_02", "text": "通用 403 权限排查：请检查账号权限配置、IP 白名单以及网络访问策略。"},
    {"id": "doc_03", "text": "若客户端证书缺少 SAN（Subject Alternative Name）扩展，网关鉴权模块将直接返回 ERR_403_CERT_INVALID 错误码。"}
]

def mock_bm25_search(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """模拟简易检索器：基于关键词匹配计算得分"""
    scores = []
    q_words = set(query.lower().split())
    for doc in MOCK_KNOWLEDGE_BASE:
        text = doc["text"].lower()
        score = sum(1.0 for w in q_words if w in text)
        scores.append((score, doc))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scores[:top_k] if item[0] > 0]

def evaluate_evidence(query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """判定节点：检查召回内容是否覆盖排查核心要素"""
    text_block = " ".join([d["text"] for d in retrieved_docs])
    has_version = "v2.4" in text_block or "升级" in text_block
    has_cause = "SAN" in text_block or "mTLS" in text_block
    
    if has_version and has_cause:
        return {"is_sufficient": True, "missing_aspects": []}
    elif has_version and not has_cause:
        return {"is_sufficient": False, "missing_aspects": ["mTLS 证书 403 错误原因与 SAN 扩展配置"]}
    else:
        return {"is_sufficient": False, "missing_aspects": ["v2.4 版本网关更新说明"]}

def rewrite_query(original_query: str, missing_aspects: List[str]) -> str:
    """根据缺失维度组合成差异化新查询词"""
    aspects_str = " ".join(missing_aspects)
    return f"{original_query} {aspects_str}"

def run_loop_rag(original_query: str, max_steps: int = 2) -> Dict[str, Any]:
    # 运行状态初始化
    state: Dict[str, Any] = {
        "step": 0,
        "current_query": original_query,
        "query_history": set(),
        "accumulated_docs": {},
        "terminated_by": ""
    }
    
    while state["step"] < max_steps:
        curr_q = state["current_query"]
        print(f"[Step {state['step']}] 执行检索，当前 Query: '{curr_q}'")
        
        # 振荡检测：若查询词完全重复则跳出
        if curr_q in state["query_history"]:
            state["terminated_by"] = "OSCILLATION_DETECTED"
            break
        state["query_history"].add(curr_q)
        
        # 执行检索并执行 ID 去重合并
        new_docs = mock_bm25_search(curr_q)
        added_count = 0
        for doc in new_docs:
            if doc["id"] not in state["accumulated_docs"]:
                state["accumulated_docs"][doc["id"]] = doc
                added_count += 1
                
        # 增量收益熔断：若未捞出新文档且已非首轮，则停止重试
        if added_count == 0 and state["step"] > 0:
            state["terminated_by"] = "NO_NEW_DOCS_FUSED"
            break
            
        # 证据充分性评估
        eval_result = evaluate_evidence(original_query, list(state["accumulated_docs"].values()))
        if eval_result["is_sufficient"]:
            state["terminated_by"] = "SUFFICIENT_EVIDENCE"
            break
            
        # 改写查询并递增步数
        state["current_query"] = rewrite_query(original_query, eval_result["missing_aspects"])
        state["step"] += 1
        
    if not state["terminated_by"]:
        state["terminated_by"] = "MAX_STEPS_REACHED"
        
    return state

# 运行测试
if __name__ == "__main__":
    result = run_loop_rag("升级 v2.4 接口 403 排查")
    print(f"\n执行结束。退出原因: {result['terminated_by']}，最终证据库文档数: {len(result['accumulated_docs'])}")
    for doc_id, doc in result["accumulated_docs"].items():
        print(f"  - [{doc_id}]: {doc['text']}")
```

在这段代码中：
1. **Step 0**：初始查询仅命中了包含“升级”与通用权限排查的文档（doc_01、doc_02），判定节点识别出缺失具体的错误因果（缺少 SAN 扩展信息）；
2. **Step 1**：查询被补充改写，加入了“mTLS 证书 403 错误原因与 SAN 扩展配置”，精准召回了包含错误码和排查方案的核心文档（doc_03）；
3. 判定节点检测到关键因果链条已补齐，系统以 `SUFFICIENT_EVIDENCE` 正常跳出并输出，避免了盲目的无效检索。
