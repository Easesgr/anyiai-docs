---
title: AI 高频场景题怎么答？大模型业务与故障排查的分析框架
icon: mdi:star-outline
tag:
- 高频场景题
- 面试题
- 实战场景
description: 提供面试中回答大模型业务故障与场景题的四步分析法，涵盖分层排查链路、工程容错手段及14个核心场景索引。
---

大模型落地相关的面试中，面试官经常提出开放式业务问题，例如“用户反馈问答助手回答不准确怎么办”或“知识库检索不到文档如何排查”。这类题目考察的不是死记硬背提示词技巧，而是候选人能否理清系统链路、自顶向下定位根因、制定工程兜底方案，并给出量化验收指标。

![AI高频业务与故障排查场景全景导读](/images/content-rebuild/8a6ef94900230e76/cover.png)

---

## 口述回答模板：四步分析法

在面试中被问及系统故障或业务设计题时，建议用 1 到 2 分钟按以下四个步骤展开：

1. **澄清现象与约束**：先向面试官确认故障范围。是偶发长尾个案还是系统性批次劣化？是事实性错误、格式不符合要求，还是接口超时？确认调用量、延迟与成本预算。
2. **沿链路分层排查**：按照数据流向逐层追溯调用日志（Trace），从输入解析、文档切分与检索、上下文组装、模型解码参数，一路排查到外部工具调用，确定问题发生在哪个节点。
3. **工程兜底与降级**：模型输出具有随机性，网络和外部接口也有抖动。工程上不能假设模型永远正确，必须配套规则拦截、缓存命中、模型降级或结构化模板兜底。
4. **量化评测与线上监控**：说明上线前后如何验收，例如构建标准测试集（Golden Dataset）评测准确率与召回率，线上监控端到端延迟（P99）、首字时间（TTFT）与用户点踩率。

---

## 贯穿案例：用户反馈“知识库助手答非所问”

我们以一个典型场景为例：业务上线了一个基于检索增强生成（RAG）的内部政策问答助手，某天客服反馈“询问报销额度时，模型回答了出差申请流程，或者给出了错误的金额”。

如果直接回答“改写提示词让模型更听话”，往往会被判定为缺乏真实排查经验。正确的排查过程是顺着数据流自上而下逐层核实：

![场景题四步漏斗应答结构](/images/content-rebuild/8a6ef94900230e76/fig-1.png)

### 步骤 1：澄清现象与复现问题
拿到用户提问的完整 Request ID，提取当时系统记录的 Trace 日志，确认：
* 用户输入的原始文本是什么？是否有错别字或指代不清（例如只输入了“这个能报多少”）？
* 知识库中到底有没有对应的最新报销政策文件？文档是否已被更新或下架？
* 是偶发错误，还是某类特定提问（如包含表格、多币种转换）普遍出错？

### 步骤 2：沿调用链路逐层定位根因
大模型系统由多个组件串联而成，每个阶段都可能引入误差：
1. **输入意图与安全检查**：检查输入是否命中了安全过滤规则，或者用户意图被分类模型错误分流。
2. **文档召回与切分**：检查向量检索和关键词检索返回的切片（Chunk）。如果得分都很低，说明检索阶段就已经漏掉了关键文档；如果切片虽然召回了，但关键表格被切分截断在两个块之间，属于解析切分损坏了信息完整性。
3. **上下文组装（Prompt 组装）**：检查注入模型上下文的最终提示词。如果检索召回了 20 个切片，但因为上下文窗口长度限制，真正包含答案的第 3 个切片被截断丢弃，或者排在提示词最中间触发了位置偏置（中间迷失现象）导致模型未能有效提取，问题就出在上下文排版。
4. **模型推理与生成**：如果上下文里有完整的正文且位置合理，但模型依然算错金额，说明基座模型理解复杂表格或计算能力不足，或者是生成采样参数（如 Temperature 过高）导致输出偏离。
5. **外部工具与后处理**：如果调用了报销系统查询 API，检查传参是否正确、API 返回是否超时或报错。

### 步骤 3：设计工程兜底方案
在不能立即换大模型的前提下，可以通过工程手段阻断错误输出：
* **相似问缓存**：对无权限差异、版本明确的高频通用政策问题，在限定有效期与置信阈值的前提下，使用经人工审核的标准答案做缓存兜底，降低模型随机波动与调用开销。
* **拒答机制**：当检索出来的最高切片相似度低于阈值时，直接提示“未检索到相关政策，请联系人工财务”，避免模型基于预训练常识胡乱编造。
* **格式化约束与校验**：针对金额提取，要求模型输出 JSON 并使用代码做正则与范围校验，不符合范围则触发告警或重试。

### 步骤 4：建立评测集与监控指标
* 抽取 100 条真实高频报销问答作为离线基准数据集（Golden Dataset）。
* 每次修改切分策略或提示词后，自动运行评测：检索召回率（Recall@K）、答案忠实度（Faithfulness，即回答是否严格源自上下文）、端到端首字时延（TTFT）。

```python
# 故障排查与复盘归因的标准分层抽象伪代码
def diagnose_issue(trace: RequestTrace) -> FailureStage:
    # 1. 检查输入侧：安全攻击或意图模糊
    if trace.query_detector.is_injected() or trace.query_detector.is_too_ambiguous():
        return FailureStage.INPUT_OR_INTENT
    
    # 2. 检查检索侧：得分过低或空召回
    if not trace.retrieval_chunks or max(c.score for c in trace.retrieval_chunks) < SIM_THRESHOLD:
        return FailureStage.RETRIEVAL_FAILURE
    
    # 3. 检查上下文：基准证据是否被截断（用于离线复盘或带有标注基准 trace.ground_truth 的排查）
    if getattr(trace, "ground_truth", None) and not is_evidence_in_context(trace.context, trace.ground_truth):
        return FailureStage.CONTEXT_TRUNCATION
        
    # 4. 检查外部工具侧：工具执行异常应先于最终生成结果校验
    if trace.tool_calls and any(call.status == "failed" for call in trace.tool_calls):
        return FailureStage.TOOL_EXECUTION_ERROR
    
    # 5. 检查生成侧：输出与参考证据是否矛盾
    if not trace.evaluator.verify_faithfulness(trace.response, trace.context):
        return FailureStage.MODEL_HALLUCINATION
        
    return FailureStage.UNKNOWN_ANOMALY
```

---

## 高频场景全景分类与索引

为了系统应对面试中的各类场景题，本模块将 14 个高频问题划分为五大方向：

![大模型高频实战场景拓扑分类](/images/content-rebuild/8a6ef94900230e76/fig-2.png)

### 1. 输出质量与排查定位
当系统生成非预期结果时，如何通过日志建立归因链路并实现回答可追溯：
* [/interview/high-freq-scenarios/inaccurate-answer.md](/interview/high-freq-scenarios/inaccurate-answer.md)：用户反馈回答不准确，如何建立端到端排查闭环？
* [/interview/high-freq-scenarios/rag-retrieval-failure.md](/interview/high-freq-scenarios/rag-retrieval-failure.md)：RAG 检索阶段发生漏召回与语义断层的定位步骤。
* [/interview/high-freq-scenarios/rag-answer-wrong.md](/interview/high-freq-scenarios/rag-answer-wrong.md)：上下文包含正确文档但模型依然答错的原因与解法。
* [/interview/high-freq-scenarios/answer-traceability.md](/interview/high-freq-scenarios/answer-traceability.md)：让模型回答具备引用来源与原文依据的工程设计。

### 2. 知识库与数据预处理
现实业务文档格式繁复，非结构化数据的处理质量往往决定了系统的上限：
* [/interview/high-freq-scenarios/doc-cleaning.md](/interview/high-freq-scenarios/doc-cleaning.md)：海量杂乱文档的清洗、切分与元数据抽取入库规范。
* [/interview/high-freq-scenarios/rag-multiformat.md](/interview/high-freq-scenarios/rag-multiformat.md)：混杂表格、图片和代码块的复杂文档如何解析与召回？

### 3. 人机交互与智能体容错
当面对模糊意图或外部调用异常时，智能体系统应具备的控制流与交互策略：
* [/interview/high-freq-scenarios/ambiguous-query.md](/interview/high-freq-scenarios/ambiguous-query.md)：面对歧义和模糊提问，何时直接回答、何时主动反问澄清？
* [/interview/high-freq-scenarios/agent-failure.md](/interview/high-freq-scenarios/agent-failure.md)：Agent 工具调用失败、死循环与幻觉参数的防御机制。

### 4. 生产可用性、安全与成本控制
脱离原型验证进入生产上线必须面对的高可用指标与防线：
* [/interview/high-freq-scenarios/timeout-degradation.md](/interview/high-freq-scenarios/timeout-degradation.md)：模型网关突发延迟抖动与接口超时的高可用降级链路。
* [/interview/high-freq-scenarios/cost-optimization.md](/interview/high-freq-scenarios/cost-optimization.md)：高频调用场景下的 Token 降本与缓存策略。
* [/interview/high-freq-scenarios/prompt-injection-defense.md](/interview/high-freq-scenarios/prompt-injection-defense.md)：直接注入与跨上下文间接注入的安全纵深防御。

### 5. 业务系统集成与持续迭代
将模型与企业存量微服务打通，并保证交付节奏与数据飞轮运转：
* [/interview/high-freq-scenarios/mvp-approach.md](/interview/high-freq-scenarios/mvp-approach.md)：在资源受限场景下，如何两周内构建高质量业务 MVP？
* [/interview/high-freq-scenarios/business-integration.md](/interview/high-freq-scenarios/business-integration.md)：将大模型无缝接入已有微服务架构的数据流与鉴权实践。
* [/interview/high-freq-scenarios/feedback-optimization.md](/interview/high-freq-scenarios/feedback-optimization.md)：线上正负反馈收集、自动归因与影子测试迭代体系。

---

## 复习建议与阶段规划

场景类面试题涉及调用链的多个环节，死记硬背容易混淆。建议按照由浅入深的顺序分阶段复习：

![高频业务场景题进阶复习依赖路线](/images/content-rebuild/8a6ef94900230e76/fig-3.png)

1. **前置知识核对**：在深入场景前，先了解 [AI 工程架构](/architecture/README.md) 的网关规范，以及 [上下文工程](/frontier/context-engineering/README.md) 中关于窗口限制和信息偏置的基础机制。
2. **第一阶段：夯实 RAG 质量闭环**（第 1 ~ 6 篇）。先掌握单次提问下的文档清洗、切片召回、长上下文组装与引用溯源，建立端到端 Trace 的排查直觉。
3. **第二阶段：掌握动态决策与工程容错**（第 7 ~ 11 篇）。学习如何应对 Agent 循环死锁、外部服务超时、调用成本陡增以及提示词注入等真实生产风险。
4. **第三阶段：架构集成与持续迭代**（第 12 ~ 14 篇）。从具体技术点抽离出来，理解 MVP 剪裁策略、微服务鉴权打通，以及如何利用线上用户反馈建立优化闭环。

---

## 常见追问与避坑提示

### 追问 1：如果面试官直接问“AI 系统很慢，怎么优化？”，应该怎么开口？

> **回答思路**：
> 不要一开口就列举具体工具（如“上 vLLM”或“加 Redis 缓存”）。正确的做法是先**厘清瓶颈指标**：
> “请问慢指的是首字延迟（TTFT）高，还是后续文本生成速率（Tokens Per Second）低？当前系统的并发吞吐是多少？链路中是否包含耗时较大的向量检索、网页抓取或多跳外部 API 调用？”
> 确认瓶颈到底是在网络 IO、检索召回、还是模型的自回归解码阶段后，再针对性给出流式传输、语义缓存、推测解码或异步并行的具体优化手段。

### 追问 2：为什么说“换用更大的模型”通常不是排查业务故障的合格回答？

> **回答思路**：
> 换用更大参数的模型固然可能提升理解泛化能力，但在实际工程落地中存在明显的副作用：
> 1. **无法弥补信息缺失**：如果业务私有文档或实时数据本身没有正确召回，再大的基座模型也无法凭空得到正确数据，反而容易产生更有说服力的事实性幻觉；
> 2. **恶化时延与成本**：参数量提升通常伴随推理延迟增加和调用成本上升，容易打破业务系统的 SLA 延迟与预算上限；
> 3. **掩盖工程链路缺陷**：切分错误、编码乱码、提示词组装截断等属于明确的工程 Bug，应当在工程层修复，依靠大模型容错治标不治本。

### 常见错误认知（避坑清单）

*   **把优化手段当成排查步骤**：被问到故障时直接回答“重做微调”或“换大模型”，跳过了查看日志、定位哪一环出错的基本排查过程。
*   **把模型当成系统唯一瓶颈**：大模型系统里包含文档解析器、向量数据库、重排模型、网络网关等多个组件。PDF 布局解析错乱或数据库超时同样会导致最终回答错误。
*   **承诺绝对确定性**：在回答大模型方案时使用“百分之百消除幻觉”、“彻底杜绝超时”等绝对化词汇。大模型推理具有概率性，网络环境也存在抖动，工程设计的核心是在概率和不确定性下建立防御与兜底手段。
