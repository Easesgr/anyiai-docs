---
title: 大模型应用如何做日志、监控与可观测性？
icon: mdi:book-open-page-variant
description: 从传统APM的局限出发，拆解Trace树状追踪、TTFT与TPOT流式指标，并结合Ragas实验评估，建立线上异常排查与回归评测链路。
---

用户向企业内部知识助手提问：“生产库只读副本延迟超过 5 秒，应该排查哪些指标？”系统返回了 HTTP 200，但界面上白屏卡顿了 4 秒才吐出第一个字，最终输出了一篇看似完整、却把关键监控参数名完全拼错的回答。

在传统 Web 系统中，接口响应码为 200，耗时 4 秒多，服务器 CPU 和内存都在安全阈值内，监控面板会亮起绿灯。但在大语言模型（LLM）应用中，这已经是一次包含“高首字延迟”和“事实幻觉”的双重质量故障。系统返回 200，仅代表网络连通且推理服务没有崩溃，并不代表输出内容可用。

在完成[内容安全护栏与审核过滤](/llm-100/evaluation/content-safety-guardrails.md)和[生成质量离线指标选型](/llm-100/evaluation/generation-quality-metrics.md)后，我们需要把度量能力扩展到在线运行阶段，明确三个核心问题：**请求卡在哪一步？为什么会给出错误答案？每次调用花了多少钱？**

![大模型可观测性全景架构](/images/content-rebuild/2e8c4c7712e1ae34/cover.png)

---

## 传统监控为什么不够用？

传统应用性能监控（APM）主要监测确定性系统。它的监控对象包括 HTTP 状态码、接口响应时长、数据库慢查询、微服务调用链路以及服务器的 CPU、内存、网络 I/O。在传统系统中，输入与输出通常遵循强类型的字段定义，接口的行为具有高度可预测性。

当系统接入大语言模型后，传统 APM 面临以下四个维度的局限：

1. **状态码掩盖语义质量问题**：模型即使返回 HTTP 200，内容仍可能存在事实幻觉、安全拒答、格式截断或逻辑自相矛盾。系统没有崩溃，但业务结果已经失效。
2. **复合调用链路难以单点定位**：一个看似简单的提问，后台通常包含“Query 重写 $\rightarrow$ 知识库检索 $\rightarrow$ 重排 $\rightarrow$ Prompt 组装 $\rightarrow$ 模型生成 $\rightarrow$ 工具调用”等多步操作。回答错误往往由链路前段引入的噪声或工具参数解析错误引发，传统单层监控无法自动归因。
3. **流式传输使单一耗时指标失效**：大模型推断通常采用 Server-Sent Events（SSE）流式传输。用户体验不仅取决于“总耗时”，更取决于“多久吐出第一个字”以及“吐字是否顺畅”。单一的接口总响应时间无法反映实际交互体验。
4. **Token 消耗直接关联财务成本**：每次调用的输入与输出 Token 数量直接对应计费与显存占用。长提示词与多轮会话累积的历史会导致单次调用开销成倍上升，必须按业务维度归集成本。

![传统APM与LLM可观测性对比](/images/content-rebuild/2e8c4c7712e1ae34/fig-1.png)

如上图所示，传统 APM 集中在基础设施和协议层；大模型可观测性则需要向下拆解流式推断的各阶段时延，向上覆盖文本语义质量与离散 Token 的开销计量。

---

## 链路追踪：Session、Trace 与 Span 的树状拓扑

为了排查复杂的复合调用问题，业界通常沿用 OpenTelemetry 规范并对其扩展，采用 `Session`、`Trace` 和 `Span` 构成的树状层级结构。

*   **Session（会话）**：记录用户在某个任务周期内的连续交互（例如包含 5 轮问答的排障过程），主要用于观察多轮对话中上下文累积对时延和质量的影响。
*   **Trace（请求链路）**：记录单次用户请求引发的完整执行过程。用户每次点击发送，系统分配一个全局唯一的 `trace_id`。
*   **Span（调用跨度）**：Trace 内部的具体执行单元，记录明确的开始时间、结束时间、状态、入参、出参和元数据标签。

在大模型应用中，Span 依据执行组件的不同通常划分为四种类型：

1. **Workflow / Chain Span**：整体编排或分支路由节点，记录业务逻辑分支的选择与编排总耗时。
2. **Retrieval Span**：检索系统执行节点，记录检索 Query、向量库召回的 Document Chunks 列表，以及重排（Rerank）前后的文本块得分。
3. **LLM Span**：模型推理节点，记录发给模型的完整 Prompt（包含系统提示词、少样本示例与上下文）、模型名称、采样参数（如 temperature）、生成的完整文本、结束原因（finish_reason）以及具体的 Token 用量。
4. **Tool Span**：工具或函数执行节点，记录模型发起的函数调用请求、解析出的入参 JSON、外部接口执行耗时与返回的原始数据。

```json
{
  "trace_id": "trace-789a-4bc1",
  "span_id": "span-llm-01",
  "parent_id": "span-agent-root",
  "name": "llm_reasoning_step_1",
  "attributes": {
    "llm.model_name": "qwen2.5-72b-instruct",
    "llm.input_tokens": 1420,
    "llm.output_tokens": 85,
    "llm.temperature": 0.2,
    "llm.finish_reason": "tool_calls",
    "llm.tool_call_name": "query_database_metrics"
  },
  "start_time": 1735689600120,
  "end_time": 1735689601450
}
```

![Trace树状调用拓扑结构](/images/content-rebuild/2e8c4c7712e1ae34/fig-2.png)

如上图所示，当一次智能体请求出现延迟时，通过展开 Trace 树中的子 Span，排查人员能立刻看清时间花在哪里：究竟是向量检索耗时了 1.5 秒，外部运维 API 阻塞了 2 秒，还是模型自身推理耗时过长。

---

## 运行监控核心指标：TTFT、TPOT 与成本度量

流式推断使得“响应总时间”这一单一指标失去解释力。两个同样耗时 3 秒返回的请求，请求 A 在 0.3 秒内开始吐字，用户可以边读边等待；请求 B 白屏等待 2.8 秒后瞬间吐出全部文字。在实际体验中，请求 A 的体感流畅度远高于请求 B。因此，必须拆解流式生成过程。

大模型推理包含两个阶段：
*   **Prefill（预填充）阶段**：模型并行处理全部输入 Prompt 的 Token，计算并构建 KV Cache。输入文本越长，计算耗时越长。
*   **Decode（自回归解码）阶段**：模型逐个生成输出 Token。每生成一个 Token 都需要加载一次模型权重并读取更新 KV Cache，该阶段主要受限于显卡显存带宽。

### 1. 首字延迟（Time To First Token, TTFT）

TTFT 定义为从客户端发出请求时刻 $t_0$ 到接收到第一个流式 Token $t_{\text{first}}$ 的时间间隔：

$$\text{TTFT} = t_{\text{first}} - t_0$$

TTFT 涵盖了网络传输、服务排队、向量检索、Prompt 拼接以及模型的 Prefill 计算耗时。如果系统输入的参考文档过长，Prefill 阶段的计算量增加，TTFT 会明显上升。

### 2. 单字生成耗时（Time Per Output Token, TPOT）

TPOT 代表首字输出后，后续每个 Token 的平均生成时间间隔：

$$\text{TPOT} = \frac{t_{\text{end}} - t_{\text{first}}}{N_{\text{completion}} - 1} \quad (N_{\text{completion}} > 1)$$

其中 $t_{\text{end}}$ 为最后一个 Token 生成完成的时间，$N_{\text{completion}}$ 为生成的总 Token 数量。TPOT 反映了模型在 Decode 阶段的吐字速度。如果并发请求过多挤占了 GPU 显存带宽，TPOT 会明显拉长，导致界面吐字卡顿。

整个请求的总响应时间（Latency）可表示为：

$$\text{Latency} = t_{\text{end}} - t_0 = \text{TTFT} + (N_{\text{completion}} - 1) \times \text{TPOT}$$

### 3. Token 成本核算

通过统计各 Span 中的 Token 数量，可以计算单次调用的实际资金成本：

$$\text{Cost} = N_{\text{prompt}} \times P_{\text{prompt}} + N_{\text{completion}} \times P_{\text{completion}}$$

式中 $P_{\text{prompt}}$ 与 $P_{\text{completion}}$ 分别代表输入与输出 Token 的单价。结合 Span 中的租户或业务线标签，可以把模型开销精确分摊到具体业务模块中。

![流式生成时序指标分解](/images/content-rebuild/2e8c4c7712e1ae34/fig-3.png)

上图展示了从请求发起至完成的时间轴切分。Prefill 阶段耗时决定了首字到达的速度，随后的 Decode 阶段则决定了后续字符流出的平滑度。

---

## 基于 Ragas 的评估实践：从主观抽检到系统化实验

在不少开发团队中，Prompt 或检索策略的调整往往依赖“主观体感抽检”——工程师在网页端手工测试两三个样例，感觉顺畅便直接发布上线。这种做法极易带来质量回退：针对某一个特殊用例修改 Prompt，往往会在未测试的其他场景中导致理解偏差。

依据 [Ragas](https://docs.ragas.io/en/stable/) 官方指南的实验驱动（Experiments-first）理念，工程团队应当将每一次检索参数调整、Prompt 模板修改或模型替换视为一次受控实验。通过在固定的基准测试集上自动化运行指标度量，对比修改前后的得分变化，确保每次上线收益可验证。

![从体感测试到实验驱动的闭环体系](/images/content-rebuild/2e8c4c7712e1ae34/fig-4.png)

上图展示了从单点手工验证迁移至版本化测试集与自动化评估的流程，通过量化指标保障系统改动的可靠性。

---

## Ragas 核心指标：解耦检索与生成两端故障

当系统给出错误答案时，必须进行因果解耦：**究竟是检索引擎没有召回正确资料，还是检索到了正确资料但模型自己产生了幻觉？**

在[RAG 评测指标基准](/fundamentals/evaluation-metrics/rag-evaluation.md)与[AI 整体评测体系](/architecture/ai-evaluation.md)中，度量维度被明确解耦为检索端和生成端两个独立方向：

### 1. 检索端核心指标

*   **上下文相关性（Context Relevance）**：衡量检索返回的文本切片中，有用内容占总内容的比例。如果检索召回了 5 个长段落，但其中 80% 都是不相干的背景介绍，该指标会显著降低，提示切分粒度过粗或过滤不严。
*   **上下文精准率（Context Precision）**：评估真正有价值的参考切片是否排在检索结果的最前面。模型在长上下文中存在关注头部与尾部的倾向，相关文档排位越靠前，模型越容易正确提炼。

### 2. 生成端核心指标

*   **回答忠实度（Faithfulness）**：衡量回答内容能否完全从检索上下文推导出来，是检测事实幻觉的核心指标。
    计算逻辑首先利用评估模型把生成的回答拆解为一组独立的原子事实陈述（Claims），随后逐一核对每项陈述是否能被检索上下文支持：

$$\text{Faithfulness} = \frac{\text{可由检索上下文支撑的陈述数}}{\text{回答中拆解出的陈述总数}}$$

*   **回答相关性（Answer Relevance）**：衡量回答是否切中用户的提问意图。即使内容完全忠实于上下文，如果答非所问（例如用户询问排查命令，模型却回答了产品历史），该指标依然很低。

![RAG评估四边形与故障归因](/images/content-rebuild/2e8c4c7712e1ae34/fig-5.png)

如上图所示，指标表现直接对应系统模块的调优方向：低 Context Precision 指向检索引擎的语义匹配或重排算法；低 Faithfulness 则提示需要增强 Prompt 约束或降低采样随机性。

### 自动化评估代码实践

以下 Python 代码演示了使用 Ragas 计算 `faithfulness` 与 `answer_relevancy` 的典型流程：

```python
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# 准备符合评估规范的样本数据
eval_data = {
    "question": [
        "生产库只读副本延迟升高，主要排查哪些指标？"
    ],
    "contexts": [[
        "只读副本延迟升高时，应重点排查主库写负载（QPS/TPS）、网络延迟（RTT）、"
        "只读副本的回放线程状态，以及大事务执行时长（如长事务阻塞了日志同步）。"
    ]],
    "answer": [
        "主要排查主库的写负载、主备之间的网络 RTT、只读副本日志回放线程状态，以及是否有长事务阻塞。"
    ]
}

# 转换为 Dataset 对象
dataset = Dataset.from_dict(eval_data)

# 执行评估实验（需配置评测模型支持，如环境变量 OPENAI_API_KEY）
results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy
    ]
)

# 输出量化报表
print("Ragas 评测结果:")
print(results.to_pandas())
```

---

## 线上数据回流：从线上告警到回归测试集

可观测性体系不仅用于发现线上故障，更负责把线上的失败案例转化为持续回归的基准数据集。

系统可以在交互界面为用户提供点赞 👍 / 点踩 👎 按钮与问题反馈标签，并记录异常退出、重复提问等隐式负向信号。

![在线反馈驱动离线回归测试集闭环](/images/content-rebuild/2e8c4c7712e1ae34/fig-6.png)

如上图所示，线上运行数据通过以下步骤完成回归沉淀：

1. **捕获异常与负反馈 Trace**：当用户点踩，或系统旁路轻量检测发现 Faithfulness 过低时，自动捕获对应的完整 Trace 记录。
2. **敏感信息脱敏**：通过正则与专用模型过滤输入输出中的姓名、密码、手机号和密钥等隐私数据（PII）。
3. **标注沉淀为黄金用例**：对清洗后的样本进行人工复核，补齐标准答案（Ground Truth），沉淀进黄金回归测试集（Golden Dataset）。
4. **CI 门禁与发布校验**：在后续调整 Prompt、更换模型或优化检索算法时，流水线自动在黄金测试集上运行评测。只有当平均指标不降低且 P99 TTFT 符合要求时，改动才允许上线部署。

---

## 课后自测练习

完成本讲学习后，请在本地环境或纸笔上完成以下自测：

1. **时延拆解计算**：某流式请求从客户端发送到首字显示耗时 1.2 秒，整个请求共生成 151 个 Token，在 4.2 秒时完全结束。请分别计算该请求的 TTFT 与 TPOT，并简述如果要改善用户的首字等待体验，应重点优化哪个阶段？
2. **故障归因判断**：在知识库问答排查中，发现某个请求生成的回答完全脱离了参考文档，属于事实编造；但检索 Span 记录显示，知识库确实召回了包含正确答案的切片。请结合 Ragas 指标，指出此时 Faithfulness 和 Context Precision 各自的表现，并说明应优先调整哪一部分系统逻辑？

在下一讲[RAG、微调与 Agent 如何组合落地](/llm-100/evaluation/rag-finetuning-agent.md)中，我们将把本讲建立的度量与可观测性体系作为准绳，深入探讨不同技术路线的选择与组合设计。
