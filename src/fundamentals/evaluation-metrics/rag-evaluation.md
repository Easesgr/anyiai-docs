---
title: RAG 系统怎么评测？检索与生成的分阶段量化方法
icon: mdi:database-check
tag:
- 评测指标
- RAG评测
- 检索命中率
description: 评测 RAG 系统不能只看最终回答。本文拆解检索与生成的两阶段评估体系，详解 Hit Rate、MRR、NDCG、忠实度与相关性指标的计算与诊断。
---

评估一个普通的文本生成任务时，我们通常拿模型输出与参考答案做字词重合度对比（例如 [生成任务怎么评估？](/fundamentals/evaluation-metrics/generation-metrics.md) 中的 ROUGE 或 BLEU）。但检索增强生成（RAG）不同：最终回答的好坏，既取决于知识库里有没有搜出相关资料，也取决于模型有没有根据资料准确提炼。

如果用户反馈“回答不准确”，单纯从最终答案做黑盒测试，往往查不出根因：到底是检索器漏掉了核心依据，还是检索出了一堆噪音分散了模型注意力，亦或是模型忽略了参考资料自行胡编？

要让 RAG 系统的调优有据可循，基本做法是**将检索阶段与生成阶段切开**，以检索出来的上下文为分界线，分别量化两个阶段的表现。

我们以一个具体场景贯穿全文：
> 用户提问：“员工离职后，企业版账号内的个人项目数据能否保留？”
> 知识库基准事实（Ground Truth）：“企业版账号在离职后立即停用，但所属项目数据由管理员保留 30 天，30 天后自动清空。”

![RAG 检索与生成两阶段解耦评估全景](/images/content-rebuild/d74e4ec118c2fff6/cover.png)

---

## RAG 系统的四类典型失败路径

一个标准 RAG 的信息流为：用户输入（Query） $\to$ 检索器（Retriever） $\to$ 检索上下文（Context） $\to$ 生成模型（Generator） $\to$ 最终回答（Answer）。在这一链路中，信息可能在四个环节出现偏差：

![RAG 系统四类失效路径因果分析](/images/content-rebuild/d74e4ec118c2fff6/fig-1.png)

结合前面的贯穿例子，这四种情况分别表现为：

1. **未检索到相关内容（Retrieval Miss）**：检索器只查到了个人版账号政策，根本没查到企业版的 30 天保留规则。生成模型拿不到事实输入，要么拒答，要么基于自身预训练知识瞎猜。
2. **检索混入过多噪声（Retrieval Noise）**：检索器虽然查到了该条款，但同时塞进了“企业报销政策”、“考勤制度”等 5 篇无关文档。关键信息被淹没，模型注意力被分散，漏掉了核心条件。
3. **模型忽略上下文（Context Ignored）**：检索器精准命中了该条款且排在第 1 位，但模型受预训练记忆影响，脱口而出：“数据永久保留”。模型没有遵从给定的上下文。
4. **模型外推产生幻觉（Extrapolation Hallucination）**：检索到的切片只写了“数据保留 30 天”，模型却擅自补充：“30 天内可联系 HR 申请无偿导出为本地压缩包”。上下文并未提及无偿导出，模型过度推断导致失真（幻觉成因可参考 [幻觉是什么？为什么大模型会胡说？](/llm-100/evaluation/hallucination-causes.md)）。

显然，仅对最终回答打分，无法区分上述四种病因。我们需要两套指标：一套测检索器，一套测生成模型。

---

## 检索阶段指标：Hit Rate@K、MRR@K 与 NDCG@K

检索阶段的评价目标是：**关键信息不仅要搜出来，还应尽量排在最前面**。因为在实际调用中，排在最前面的切片更容易被大模型的注意力机制优先读取。

![检索阶段核心排序指标评估机制](/images/content-rebuild/d74e4ec118c2fff6/fig-2.png)

### 1. 命中率（Hit Rate@K）

Hit Rate@K 衡量在前 $K$ 个检索结果中，是否至少包含一条真实标注的参考切片（Ground Truth）：

$$\text{Hit Rate@K} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(\text{Top-}K_i \cap \text{GT}_i \neq \emptyset)$$

其中 $\mathbb{I}$ 为指示函数，命中为 1，未命中为 0。该指标只关注“有没有找到”，不关心正例文档排在第 1 位还是第 $K$ 位。

### 2. 平均倒数排名（Mean Reciprocal Rank, MRR@K）

MRR@K 衡量**第一个相关文档出现的靠前程度**。如果第一个相关文档出现在第 $r$ 位，其倒数得分为 $1/r$；若前 $K$ 个结果中均未命中，则得分为 0：

$$\text{MRR@K} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

首位命中得 1.0，第 2 位命中得 0.5，第 5 位命中得 0.2。MRR 侧重考核单点关键证据能否推到最前排。

### 3. 归一化折损累计增益（NDCG@K）

当回答一个问题需要多条参考切片共同补充，且不同切片的相关度有层级区分时（例如：0 分不相关，1 分补充信息，2 分核心结论），NDCG@K 能全面评估多文档排序质量。

先计算折损累计增益（DCG@K）：

$$\text{DCG@K} = \sum_{i=1}^K \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

其中 $rel_i$ 为位置 $i$ 处切片的相关度得分，分母 $\log_2(i + 1)$ 是位置惩罚项，越靠后增益衰减越厉害。

将真实结果的 DCG 除以理想排序下的最大可能增益（IDCG@K），即可得到归一化得分：

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

### 逐步算例验证

假设检索器在 $K=3$ 时返回了 3 个切片，相关度标注如下：
- 排名第 1：$rel_1 = 0$（不相关，考勤政策）
- 排名第 2：$rel_2 = 2$（核心依据，数据保留 30 天条款）
- 排名第 3：$rel_3 = 1$（补充依据，账号注销补充流程）

各项指标计算过程：
1. **Hit Rate@3**：前 3 位中命中了相关切片，$\text{Hit Rate@3} = 1.0$。
2. **MRR@3**：第一个相关切片出现在第 2 位，倒数排名为 $1/2 = 0.5$。
3. **NDCG@3**：
   - 实际 DCG@3：
     $$\text{DCG@3} = \frac{2^0 - 1}{\log_2(2)} + \frac{2^2 - 1}{\log_2(3)} + \frac{2^1 - 1}{\log_2(4)} = 0 + \frac{3}{1.585} + \frac{1}{2.0} \approx 1.893 + 0.5 = 2.393$$
   - 理想排序应为降序 $[2, 1, 0]$，计算 IDCG@3：
     $$\text{IDCG@3} = \frac{2^2 - 1}{\log_2(2)} + \frac{2^1 - 1}{\log_2(3)} + \frac{2^0 - 1}{\log_2(4)} = 3.0 + \frac{1}{1.585} + 0 \approx 3.0 + 0.631 = 3.631$$
   - 归一化得分：
     $$\text{NDCG@3} = \frac{2.393}{3.631} \approx 0.659$$

虽然命中率是满分 1.0，但核心文档由于排在第 2 位，导致 MRR 跌至 0.5，NDCG 仅为 0.659。这反映出首位的无关文档给生成端带来了潜在干扰。

<ReadMoreLock />

---

## 生成阶段指标：忠实度与回答相关性

生成阶段评测的核心是：**模型是否忠实提取了给入的参考资料，以及是否切中用户的问题**。

在 Ragas 等评测实践中，通常将生成模块解耦为两个指标：**忠实度（Faithfulness）** 与 **回答相关性（Answer Relevance）**。

![忠实度与回答相关性评估逻辑对比](/images/content-rebuild/d74e4ec118c2fff6/fig-3.png)

### 1. 忠实度（Faithfulness）：事实陈述是否有据可依

忠实度评估模型输出的每一句事实陈述，能否在检索出来的上下文中找到依据：

- **第一步（陈述拆解）**：将生成模型的长篇回答拆解为一组独立的命题原子句：$A \to \{s_1, s_2, \dots, s_n\}$。
- **第二步（依据校验）**：检查每一项命题 $s_i$ 是否能被检索上下文 $C$ 逻辑支撑。

$$\text{Faithfulness} = \frac{|\{s_i \in A \mid C \vdash s_i\}|}{|A|}$$

**适用条件与边界**：
- **只验遵从，不验真理**：如果知识库切片本身写错了（例如误写成保留 60 天），模型也输出 60 天，其忠实度依然是 1.0。忠实度度量的是“是否照着资料说”，而非“是否符合真实世界事实”。
- **不代表信息完整**：如果模型只回复“根据文档无法确定”，只要上下文中确实没写明确答案，其断言集合成立，忠实度同样可能得满分。

### 2. 回答相关性（Answer Relevance）：输出是否直面问题

回答相关性考察模型是否直接解答了用户疑惑，是否顾左右而言他或夹杂大量套话。

常用的一种自动评测机制是**反向问题推导（Reverse Question Generation）**：
1. 仅根据模型生成的回答 $A$，让评测模型推断用户可能提了什么问题，生成 $m$ 个推测问题 $Q^{\text{gen}}_j$（通常 $m=3$）。
2. 计算推测问题与真实提问 $Q$ 之间的语义向量余弦相似度均值：

$$\text{Answer Relevance} = \frac{1}{m} \sum_{j=1}^m \cos(\mathbf{e}(Q), \mathbf{e}(Q^{\text{gen}}_j))$$

如果模型回答直击要害，根据回答反推出来的问题与原问题语义一致；若回答充斥着与提问无关的免责声明或套话，反推的问题将偏离原意，相关度得分显著降低。

---

## 检索与生成协同指标：上下文召回率与上下文精确率

除了单独评估检索排序和生成质量，评估体系还需要以**标注的标准答案（Ground Truth, $A_{\text{gt}}$）**为参照，度量两阶段的信息衔接效率。

![RAG 四元组协同因果归因矩阵](/images/content-rebuild/d74e4ec118c2fff6/fig-4.png)

### 1. 上下文召回率（Context Recall）

衡量标准答案中包含的事实信息，有多少比例被检索出来的上下文 $C$ 成功覆盖：

$$\text{Context Recall} = \frac{|\{s \in A_{\text{gt}} \mid C \vdash s\}|}{|A_{\text{gt}}|}$$

如果上下文召回率过低，说明检索阶段就漏掉了核心信息。下游模型无论推理能力多强，也无法基于给定的上下文输出完整答复。

### 2. 上下文精确率（Context Precision）

衡量召回列表中的有效切片是否集中排在顶部。它利用平均精确率（Average Precision）思想，对相关切片的位置进行加权：

$$\text{Context Precision@K} = \frac{\sum_{k=1}^K \left( \text{Precision@k} \times v_k \right)}{\sum_{k=1}^K v_k}$$

其中 $v_k \in \{0, 1\}$ 表示第 $k$ 个切片是否对推导标准答案提供了实质帮助，$\text{Precision@k}$ 为前 $k$ 个切片中有效切片的比例。如果召回了 5 个切片，唯一有用的切片排在第 5 位，即便它被找出来了，前 4 位的无用切片也会将上下文精确率拉低。

| 指标低下的表现 | 问题环节 | 常见原因 | 调优切入点 |
| :--- | :--- | :--- | :--- |
| **低 Context Recall** | 检索召回 | 向量相似度未匹配、分块过碎导致语义断裂、召回条数 $K$ 过小 | 调整分块长度与重叠率、增大初筛 Top-K、接入关键词检索（BM25） |
| **低 Context Precision** | 检索重排 | 关键词重合度高但语义无关的文档排在前面 | 引入重排模型（Reranker）、按相关度分数设置截断阈值 |
| **低 Faithfulness** | 生成阶段 | 模型更倾向于依赖自身预训练记忆、提示词约束过弱 | 优化提示词强化上下文约束、降低采样温度参数、更换理解力更强的模型 |
| **低 Answer Relevance** | 生成阶段 | 模型过度拒答、输出冗余客套话、意图未完全对齐 | 补充 Few-Shot 示例、微调生成提示词的输出格式要求 |

---

## 评测数据集工程：LabelledRagDataset 的构成

运行上述指标，需要准备标准测试集（如 LlamaIndex 中定义的 `LabelledRagDataset`）。一条完整的评测样本通常包含四个核心字段：
- `query`：用户提问。
- `reference_contexts`：能够支撑答案的标准文本切片集合。
- `reference_answer`：经过校验的标准答案（Ground Truth）。
- `metadata`：场景标签、切片 ID、难度分类等元数据。

![LabelledRagDataset 构建工程管线](/images/content-rebuild/d74e4ec118c2fff6/fig-5.png)

### 自动化合成与真实日志回流

准备评测集通常有两条互补路径：

```python
# 伪代码：从文档切片自动化生成问答样本的逻辑示意
from dataclasses import dataclass
from typing import List

@dataclass
class RagEvalSample:
    query: str
    reference_contexts: List[str]
    reference_answer: str

def build_synthetic_sample(document_chunk: str, synthesizer_llm) -> RagEvalSample:
    """
    从单个文档切片生成评测三元组
    适用条件：系统冷启动初期，缺乏线上真实访问日志
    """
    prompt = f"请仔细阅读以下文档，生成一个具体的问题，并给出依据文档的标准回答：\n{document_chunk}"
    response = synthesizer_llm.generate(prompt)
    return RagEvalSample(
        query=response.question,
        reference_contexts=[document_chunk],
        reference_answer=response.ground_truth
    )
```

- **自动化合成（Synthesizer）**：利用大模型遍历知识库切片自动出题。优点是冷启动速度快、覆盖面广；缺点是模型生成的提问往往语句完整、用词与原文高度契合，缺乏真实用户提问常见的口语化、拼写笔误和模糊表达。
- **生产日志采样（Production Logs）**：从线上真实交互日志中抽取提问，结合用户点踩反馈进行人工校验。这种样本量相对较小，但更能反映真实的业务分布与长尾提问。

---

## 混合检索调参：根据评测反馈校准融合权重

建立评测集后，对 RAG 系统的优化就能转变为有量化指标支撑的实验。以**混合检索（Dense 稠密向量检索 + BM25 稀疏检索）**为例：

Dense 擅长理解语义泛化，但在专有缩写、型号、错误码上容易漂移；BM25 擅长精确字符匹配，但缺少语义泛化能力。

在融合两者时，不能简单将它们的分数按比例相加：

$$S_{\text{final}} = 0.5 \cdot S_{\text{dense}} + 0.5 \cdot S_{\text{bm25}}$$

**原因在于**：Dense 余弦相似度通常分布在有界区间内（数学定义域为 $[-1, 1]$，部分向量库或框架裁剪/归一化至 $[0, 1]$），而 BM25 得分取决于词频和逆文档频率，是一个无理论上界的正实数。两者的量纲和分数分布不同，简单的绝对数值相加容易让某一通道完全主导排序结果。

![基于评测指标反馈的混合检索参数校准闭环](/images/content-rebuild/d74e4ec118c2fff6/fig-6.png)

### 基于排名的加权融合（Weighted RRF）

一种更稳妥的融合方式是倒数排名融合（Reciprocal Rank Fusion, RRF）。它只依据各通道给出的相对名次进行计算（例如 Qdrant 支持的加权 RRF 接口）：

$$RRF\_Score(d) = \sum_{m \in \{dense, bm25\}} w_m \cdot \frac{1}{k + \text{rank}_m(d)}$$

其中 $w_m$ 是该检索通道的权重，$k$ 是平滑常数（常规默认取 60），$\text{rank}_m(d)$ 是文档 $d$ 在通道 $m$ 中的排序位次（从 1 开始）。

通过评测集调优权重的步骤如下：
1. **划分验证集与测试集**：挑选覆盖专有名词查询与抽象语义查询的样本，严格划分为用于超参数搜索的开发/验证集（Dev/Validation Set）与独立的测试集（Test Set），防止数据泄漏。
2. **网格搜索权重**：固定平滑项 $k=60$，在验证集上以 0.1 为步长调整 $w_{\text{dense}}$ 与 $w_{\text{bm25}}$。
3. **跟踪检索指标**：在验证集上运行各组权重配置，统计 Hit Rate@5 与 NDCG@5 的得分曲线，选定表现最佳的权重配比。
4. **测试集泛化验证与重排接入**：在独立测试集上评估选定权重组合的泛化性能；在此基础上评估接入 Cross-Encoder 重排器（Reranker）带来的 NDCG 增益，并以此决定最终送入生成模型的 Top-K 截断点。

借助这套流程，检索参数调整不再依靠主观经验，而是依据在验证集调优、测试集无偏验证的 NDCG 和命中率数据得出结论。

---

## 推荐阅读与延伸

- 前置基础：[AI 模型评测到底评什么？](/fundamentals/evaluation-metrics/what-is-evaluation.md) — 了解离线评测的基本逻辑、测试集划分与过拟合风险。
- 前置基础：[生成任务怎么评估？](/fundamentals/evaluation-metrics/generation-metrics.md) — 掌握无检索约束下生成模型的传统 NLP 指标与 LLM 评审范式。
- 后续深入：[AI 应用怎么评测？](/fundamentals/evaluation-metrics/app-evaluation.md) — 了解带工具调用、多步骤状态机与 Agent 应用的系统评测思路。
- 面试演练：[如何评估一个 RAG 系统？](/interview/rag-interview/rag-evaluation.md) — 从系统架构设计与面试答辩视角，复盘两阶段评测的口述框架与追问防线。
