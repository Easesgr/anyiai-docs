---
title: 如何评估一个 RAG 系统？指标体系与自动化评测
icon: mdi:chart-line
tag:
- RAG面试
- RAG评估
- Hit@K
description: 详解 RAG 评测三元组（Context Relevance、Faithfulness、Answer Relevance）的计算原理、算例推导、基准集构建与
  CI/CD 质量卡点。
---

![RAG 系统科学评测体系封面](/images/content-rebuild/8ecfc95e6a300295/cover.png)

在优化检索增强生成（RAG）系统时，开发者经常遇到一个典型问题：调整了分块大小（Chunk Size）、更换了嵌入（Embedding）模型，或者重写了系统提示词（Prompt）之后，整体生成质量到底提升了还是下降了？如果仅靠人工随手抽查两三个问题，既无法捕捉偶发错误，也无法在改动代码时及时发现能力退化。

评估 RAG 系统不能只看最终答案表面上是否通顺，而需要把检索模块和生成模块拆开分别度量。上一讲我们分析了 [RAG 为什么仍然会产生幻觉](/interview/rag-interview/rag-hallucination.md)，本篇将围绕行业通用的评估方法，建立一套可解释、可复现的自动化评测流程。

---

## 一、面试口述答案：RAG 评测三元组

如果在面试中被问到“如何系统性地评估一个 RAG 系统”，可以采用如下结构作答：

> 评估 RAG 系统不能只依赖传统的文本重叠指标（如 BLEU 或 ROUGE），因为这类指标无法区分语义改写与事实性错误，也无法定位问题究竟出在检索环节还是生成环节。
>
> 目前通用的评测方案采用**评测三元组（RAG Triad）**，将系统拆解为三个实体：**用户提问（Query）**、**检索上下文（Context）** 与 **最终回答（Answer）**，并在它们之间计算三项核心指标：
>
> 1. **上下文相关度（Context Relevance）**：度量 Query 到 Context 的信噪比，即检索返回的文本块中有多少内容与用户问题真正相关，过滤冗余噪声。
> 2. **忠实度（Faithfulness）**：度量 Context 到 Answer 的事实支撑程度，即回答中包含的每一项事实陈述是否都能在检索上下文中找到依据，这是量化幻觉的关键指标。
> 3. **答案相关度（Answer Relevance）**：度量 Query 到 Answer 的切题程度，即回答是否完整回应了用户提问的目标，惩罚答非所问或废话连篇。
>
> 如果业务具备标注的标准答案（Ground Truth），还可以补充度量**上下文召回率（Context Recall）**与**语义相似度（Semantic Similarity）**。在工程实践中，通常借助大模型作为裁判（LLM-as-a-Judge），通过 Ragas 或 TruLens 等工具搭建自动化评测脚本，并将其集成到 CI/CD 流程中，作为版本发布的质量卡点。

![RAG 评测三元组核心结构](/images/content-rebuild/8ecfc95e6a300295/fig-1.png)

如上图所示，RAG 评测三元组将端到端流程切分为两个环节：检索环节的输入是 Query、输出是 Context；生成环节的输入是 Query 与 Context、输出是 Answer。三项指标分别约束这两个环节，从而实现故障归因。

---

## 二、检索模块指标：上下文相关度与召回率

检索模块决定了输入给大模型的原材料上限。检索模块评估通常分为“无需标注答案的信噪比度量”与“依赖标注答案的完备性度量”。

### 1. 上下文相关度（Context Relevance）

上下文相关度关注检索返回的内容纯度。如果检索返回了 4 个片段，共计 1000 字，但其中只有 200 字与问题相关，其余全是免责声明或无关介绍，相关度就较低。过多的无关上下文不仅浪费上下文窗口（Context Window），还会干扰大模型对关键信息的提取。

在没有标准答案的情况下，评测工具常借助大模型作为裁判（LLM-as-a-Judge），将上下文拆解为句子，并判断每句话是否对解答问题有价值：

$$\text{Context Relevance} = \frac{\text{检索内容中与提问相关的句子数}}{\text{检索内容中的总句子数}}$$

**算例演示：**
* **Query**：“企业版年费支持按月退订吗？”
* **检索返回 Context（按标点拆分为 4 个独立句子）**：
  1. “企业版年费套餐按自然年签约并一次性付款。”
  2. “签约 7 天内支持全额退款，超过 7 天不支持按月折算退订。”
  3. “所有企业用户享受 7×24 小时专属技术支持。”
  4. “系统支持信用卡、对公转账及支付宝付款。”
* **判决模型研判**：句子 1 和句子 2 提供了退订规则与计费方式，与提问直接相关；句子 3 和句子 4 分别介绍服务与支付渠道，与退订问题无关。
* **计算结果**：$\text{Context Relevance} = \frac{2}{4} = 0.50$。

注：在部分工具（如 Ragas 的 Context Precision）中，该指标需要参考标准答案（Ground Truth），并引入位置加权惩罚把相关文档排在后面的情况；而纯无标注场景下的 Context Relevance 则聚焦度量检索上下文整体的信噪比。

### 2. 上下文召回率（Context Recall）

与相关度关注“纯度”不同，上下文召回率关注“回答问题所需的全部事实，检索模块是否找全了”。该指标需要基准测试集中提供人工标注的标准答案（Ground Truth）。

$$\text{Context Recall} = \frac{\text{标准答案中可由检索上下文支撑的事实陈述数}}{\text{标准答案中的总事实陈述数}}$$

![检索相关度与上下文召回率对比机制](/images/content-rebuild/8ecfc95e6a300295/fig-2.png)

如上图所示，上下文相关度相当于**查准（Precision）**，关注剔除噪声；上下文召回率相当于**查全（Recall）**，关注避免漏检。结合我们在 [什么是 RAG 中的 Rerank](/interview/rag-interview/rag-rerank.md) 中讨论过的重排机制，Rerank 能够重新调整候选顺序并截断低分段落，从而改善上下文信噪比。

### 3. 经典搜索指标（Hit@K, MRR, NDCG）与语义评测指标的边界

在优化向量数据库索引或 BM25 算法时，常用 Hit@K、MRR（平均倒数排名）和 NDCG（归一化折损累计增益）。它们依赖于“文档 ID”级别的标注。

* **经典搜索指标**：适合离线验证检索算法、向量模型或分块参数。只要指定的目标文档排在 Top-K，即可得分。但它无法判断文档内部具体哪句话有用。
* **RAG 语义指标**：直接在文本和事实粒度上度量，关注大模型实际读到的语义片段能否支撑最终回答。

两者互为补充：经典指标用于评估召回算子本身，语义指标用于评估检索输出对下游生成的实际支撑质量。

<ReadMoreLock />

---

## 三、生成模块指标：忠实度与答案相关度

生成模块关注两个问题：第一，模型有没有依据给定的上下文作答（不胡编）；第二，模型有没有回答用户的提问（不跑题）。

### 1. 忠实度（Faithfulness）：量化幻觉

忠实度衡量回答中的事实陈述是否均来源于检索上下文。如果模型根据训练阶段学到的先验知识补全了上下文里没有的信息，即使这些信息在现实中客观正确，也会因为脱离上下文而被判定为未受支持。

Faithfulness 的标准评测流程如下：

1. **事实陈述拆解（Claim Extraction）**：要求判决模型将生成的 Answer 拆解成若干个独立的原子陈述（Statements），每个陈述只包含一个单一事实。
2. **陈述核对（Statement Verification）**：判决模型对照检索上下文 Context，逐一判断每个原子陈述是否能被上下文逻辑推导或明确支持（支持记 1，不支持记 0）。
3. **计算占比**：

$$\text{Faithfulness} = \frac{\text{能被上下文支持的事实陈述数}}{\text{回答中拆解出的事实陈述总数}}$$

![忠实度计算流程分解](/images/content-rebuild/8ecfc95e6a300295/fig-3.png)

上图展示了忠实度的执行过程。以一个具体示例说明：

* **检索 Context**：“系统构建工具已升级至 3.2 版本，编译速度提升了 40%，当前版本不支持跨平台交叉编译。”
* **模型生成 Answer**：“构建工具目前已升级到 3.2 版本，编译速度提升了 40%，并且支持跨平台交叉编译。”
* **拆解原子陈述**：
  * 陈述 1：“构建工具目前为 3.2 版本。” $\rightarrow$ Context 明确支持（得分 1）。
  * 陈述 2：“编译速度相比旧版本提升了 40%。” $\rightarrow$ Context 明确支持（得分 1）。
  * 陈述 3：“工具支持跨平台交叉编译。” $\rightarrow$ Context 明确否定（得分 0）。
* **最终得分**：$\text{Faithfulness} = \frac{2}{3} \approx 0.67$。

通过将长文本拆解为离散陈述，评测系统避免了对整体回答的主观打分，将幻觉量化为可比对的百分比。

### 2. 答案相关度（Answer Relevance）

答案相关度评估最终生成内容是否切题，主要用于惩罚答非所问、过度客套或遗漏关键约束的问题。该指标不依赖标准答案，也不验证事实真伪。

为了减少大模型直接给“相关度”打分的主观偏差，主流方法采用**反向提问机制（Reverse Query Generation）**：

1. 判决模型根据生成的 Answer，反向推测出 $N$ 个用户最可能提出的原始问题（$Q'_1, Q'_2, \dots, Q'_N$）。
2. 计算反推问题 $Q'_i$ 与用户真实提问 $Q$ 之间的嵌入向量余弦相似度。
3. 对 $N$ 次相似度计算均值：

$$\text{Answer Relevance} = \frac{1}{N} \sum_{i=1}^{N} \cos(E(Q), E(Q'_i))$$

如果模型给出的答案抓住了核心关键点，反向推导出的问题与原问题语义非常接近；如果答案通篇是无关套话，反向推测出的问题就会与原问题产生较大语义偏移，余弦相似度随之降低。

---

## 四、主流评测框架选型对比

在工程实践中，开发者一般无需从零实现拆解与打分逻辑，已有成熟的开源框架提供现成组件：

| 对比维度 | Ragas | TruLens | LlamaIndex 评测模块 |
| :--- | :--- | :--- | :--- |
| **核心定位** | 专注于 RAG/Agent 离线评测与合成测试集生成 | 聚焦全链路可观测性、实时护栏（Guardrails）与反馈函数 | 原生集成于 LlamaIndex 引擎的轻量评测组件 |
| **无标注评测支持** | 支持 Faithfulness、Answer Relevancy（Context Precision/Recall 则需参考答案） | 基于 Feedback Functions 计算 RAG Triad 指标 | 提供 `FaithfulnessEvaluator`、`RelevancyEvaluator` 等基础组件 |
| **追踪与可视化** | 侧重输出 DataFrame，配合数据科学工具或 CI 报表展示 | 内置 TruLens Dashboard，可下钻单次请求的检索与调用明细 | 依赖外部 Tracing 工具（如 OpenInference、Phoenix）集成 |
| **测试集生成能力** | 内置 Testset Generator，支持基于知识库合成多跳问答数据 | 偏向运行时指标记录，离线合成数据功能较弱 | 提供基础的 `LabelledRagDataset` 生成工具 |
| **典型适用场景** | 离线算法选型对比、版本发布 CI/CD 质量卡点 | 线上服务实时监控、调用链追踪、异常请求告警 | 深度使用 LlamaIndex 框架的项目内部单元测试 |

![主流评测框架选型与功能侧重对比](/images/content-rebuild/8ecfc95e6a300295/fig-4.png)

实际工程选型中，这三者并非互斥：团队常使用 **Ragas** 在发布前进行批量离线基准测试与自动化回归，而在线上运行时借助 **TruLens** 或其他链路追踪工具进行持续抽样观测。

---

## 五、黄金评测集（Ground Truth Dataset）构建方案

自动化评测的准确性高度依赖测试集的质量。完全依赖人工编写数百道带标准答案的问答集成本较高，而完全依赖模型生成又容易产生分布偏离。

工程中通常采用**合成生成、人工校验、负样本补充**的四步流程：

![黄金评测集构建四步法流程](/images/content-rebuild/8ecfc95e6a300295/fig-5.png)

1. **知识切片抽样**：从清洗后的知识库中抽取具有代表性的文档块（Chunks），覆盖技术规范、政策条款、常见问答（FAQ）等不同业务类型，确保数据分布均衡。
2. **合成多复杂度问题**：利用大模型基于抽取的文本块自动生成问题与参考答案。可分为三种形态：
   * **直接问题**：答案单次出现，仅需单个文本块即可回答。
   * **推理问题**：答案需要结合同一文档块内的多处条件进行逻辑推导。
   * **多跳问题（Multi-Context）**：需要将两个甚至多个相关文档块结合起来才能完整作答，专门测试多路召回能力。
3. **人工校验与修正**：由业务人员或工程师对合成样本进行抽检和清洗，修正事实错误、删去表意不清的问题，并核准标准答案。
4. **注入硬负样本与越界提问**：加入故意超出知识库范围的提问（Out-of-Domain），以及表述相近但答案不存在的诱导性问题，用于检验系统的拒答与防御机制。

---

## 六、自动化回归测试与 CI/CD 质量卡点

为了避免修改 Prompt 或调整 Embedding 参数时引入退化，评测应作为自动化测试集成在开发流程中。

以下是基于 Python 与 Ragas 构建的自动化回归测试示例脚本：

```python
import pytest
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# 待评测的基准测试样本（实际工程中从 json 或 parquet 文件加载）
eval_samples = {
    "question": [
        "知识库支持哪些文件格式的直接解析？",
        "企业版授权过期后数据会保留多久？"
    ],
    "contexts": [
        ["系统支持 PDF、Word、Markdown 和 TXT 格式的文本提取，暂不支持直接解析扫描件表格。"],
        ["企业版授权过期后，数据将保留 30 天宽限期供导出。30 天后系统将执行物理清理，不可恢复。"]
    ],
    "answer": [
        "系统支持解析 PDF、Word、Markdown 以及 TXT 文本文件，但目前暂不支持直接解析扫描件中的表格。",
        "授权过期后会有 30 天的保留期供备份，超过 30 天后数据会被物理清除。"
    ],
    "ground_truth": [
        "支持 PDF、Word、Markdown 和 TXT 格式。扫描件表格暂不支持。",
        "过期后保留 30 天，超时将被物理清理。"
    ]
}

def test_rag_quality_gate():
    """CI/CD 自动化质量回归门禁"""
    dataset = Dataset.from_dict(eval_samples)
    
    # 执行自动化评测计算
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    )
    
    # 计算各项指标均值
    scores = results.to_pandas().mean(numeric_only=True)
    print("\n--- RAG 自动化评测结果 ---")
    print(scores)
    
    # 设置质量阻断卡点阈值（根据业务历史基线确定）
    FAITHFULNESS_THRESHOLD = 0.85
    ANSWER_RELEVANCE_THRESHOLD = 0.80
    CONTEXT_RECALL_THRESHOLD = 0.80
    
    assert scores["faithfulness"] >= FAITHFULNESS_THRESHOLD, \
        f"忠实度未达标: {scores['faithfulness']:.2f} < {FAITHFULNESS_THRESHOLD}"
    assert scores["answer_relevancy"] >= ANSWER_RELEVANCE_THRESHOLD, \
        f"答案相关度未达标: {scores['answer_relevancy']:.2f} < {ANSWER_RELEVANCE_THRESHOLD}"
    assert scores["context_recall"] >= CONTEXT_RECALL_THRESHOLD, \
        f"上下文召回率未达标: {scores['context_recall']:.2f} < {CONTEXT_RECALL_THRESHOLD}"
```

![CI/CD 质量门禁流水线](/images/content-rebuild/8ecfc95e6a300295/fig-6.png)

在代码提交（Pull Request）触发流水线后，测试环境调用当前版本的 RAG 链路生成回答，评测脚本对输出计算得分。一旦忠实度或相关度低于预设阈值，CI/CD 流水线自动报错并阻止合并，避免未验证的修改污染生产环境。

---

## 七、面试追问与常见错误

### 追问 1：如果 Context Relevance 很高，但 Faithfulness 很低，通常是哪个环节的问题？

**答案解析**：
这说明**检索环节给出的材料信噪比高且有效，问题出在下游的大模型生成环节**。主要排查方向包括：
1. **提示词约束过弱**：Prompt 中未明确声明“仅基于所给上下文回答，未提及的内容明确声明不知道”，导致模型倾向于用预训练先验知识自由发挥。
2. **采样温度（Temperature）设置偏高**：在事实型问答任务中，过高的采样温度会放大随机采样概率，导致产生偏离上下文的事实虚构。建议将温度降低（如设为 0 到 0.2）。
3. **模型遵循能力不足**：选用了推理能力或长文本遵循能力偏弱的模型，在面对限制条件时无法严格执行。
4. **多片段上下文之间存在冲突**：检索返回的多个文本块虽然都与主题相关，但它们内部的陈述存在版本不一致或相互矛盾，模型在合成时产生混乱。

### 追问 2：LLM 作为裁判（LLM-as-a-Judge）存在评分波动和偏见，工程上如何提升可靠性？

**答案解析**：
大模型判官可能存在自偏好（对自身生成的文本打分偏高）、位置偏见（关注靠前或靠后的内容）或长度偏见。工程上通常采用以下手段抑制偏差：
1. **选用高稳定性推理模型作为裁判**：评估模型一般选用遵循度与推理能力更高的模型，避免裁判模型自身出现格式解析错误。
2. **强制结构化输出与思维链（CoT）**：要求裁判模型在给出分数前，先输出证据比对理由与陈述拆解结果，并采用 JSON Schema 或 Pydantic 约束输出格式，减少主观评分发散。
3. **评测打分锚定（Few-Shot Calibration）**：在评测 Prompt 中提供标准的满分与低分判定示例，明确打分边界。
4. **小规模人工标定对齐**：定期抽样人工复核 50～100 条样本，计算判决模型与人工打分的一致性系数（如 Spearman 相关系数），确保自动打分趋势与人工标准保持一致。

### 常见错误辨析

* **误区 1**：“评测 RAG 系统只需计算生成文本和参考答案的 BLEU / ROUGE 即可。”
  * **纠偏**：BLEU 和 ROUGE 基于字词重叠（n-gram）。如果模型用同义词改写了答案，语义完全正确但字面重叠度低，BLEU 得分会严重偏低；反之，若生成了字面类似但颠倒了关键数字或否定词的幻觉句子，BLEU 反而得分很高。此外，BLEU 无法区分是检索漏召回还是生成发生了幻觉。
* **误区 2**：“没有人工标注的标准答案，就完全无法建立量化评测体系。”
  * **纠偏**：RAG 评测三元组中的核心指标（Context Relevance、Faithfulness、Answer Relevance）**均不需要标注的标准答案**。只要记录用户 Query、检索得到的 Context 和模型输出的 Answer，即可启动自动化评估。
* **误区 3**：“Hit Rate 达到 95% 就代表检索端没有问题。”
  * **纠偏**：Hit Rate（命中率）仅代表 Top-K 结果中包含至少一条相关文档，它既不能度量回答所需的事实是否全部被召回（Context Recall），也无法反映召回的片段里是否掺杂了 90% 的无关噪声（Context Relevance）。
