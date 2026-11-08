---
title: 文本生成质量如何评估？BLEU、ROUGE、大模型裁判与人工评估选型
icon: mdi:book-open-page-variant
description: 系统讲解自然语言生成评估方法：词法匹配（BLEU/ROUGE）、语义向量对齐（BERTScore）、大模型裁判（LLM-as-a-judge）与人工一致性检验，附可检验练习。
---

评估大语言模型的生成质量，是验证 Prompt 优化、RAG 检索增强以及模型微调效果的核心环节。与分类、回归等确定性任务不同，自由文本生成存在“一题多解”的特点——同一语义可以用完全不同的句式表达，而微小的字符变动却可能彻底颠倒事实。

在上一讲[/llm-100/evaluation/llm-evaluation-benchmarks.md](/llm-100/evaluation/llm-evaluation-benchmarks.md)中，我们梳理了标准化基准测试集。但在业务实践中，我们面对的往往是非标准化的开放文本。为了建立直观认知，我们使用一个贯穿全篇的医疗咨询场景用例：

*   **用户提问**：`高血压患者可以吃布洛芬退烧吗？`
*   **医生标准参考（Reference）**：`布洛芬属于非甾体抗炎药，可能会引起血压升高或减弱降压药疗效，高血压患者应在医生指导下谨慎使用。`
*   **候选回答 A（同义改写）**：`该药为抗炎止痛类药物，可能导致血压波动并影响降压药作用，患者须遵医嘱谨慎服用。`
*   **候选回答 B（致命反义）**：`布洛芬不属于非甾体抗炎药，不会引起血压升高或减弱降压药疗效，高血压患者可以放心日常使用。`
*   **候选回答 C（冗长废话）**：`高血压是常见的心血管慢性疾病，常见降压药物有五大类……（中间展开400字科普）……因此吃布洛芬前建议问问医生。`

理想的评估体系应当给回答 A 高分，给回答 B 极低分（严重事实错误），并识别回答 C 的冗余。然而，不同评估工具在面对这些样本时会给出截然相反的结论。

![文本生成质量度量体系全景](/images/content-rebuild/f1888a4c0c12f983/cover.png)

---

### 1. 基于字词重叠的词法匹配指标：BLEU 与 ROUGE

词法匹配指标的核心思想，是统计模型候选生成序列（Candidate）与参考答案（Reference）之间的局部字符片段（n-gram）重叠比例。其计算过程完全确定、执行耗时通常在毫秒级，且无需额外算力成本。

#### 1.1 BLEU：精确率导向与简短惩罚

BLEU（Bilingual Evaluation Understudy）最初针对机器翻译场景设计，核心关注**精确率（Precision）**：候选文本生成的词语中，有多少比例包含在参考文本中。

为了避免模型通过重复单个高频安全词（如连续生成“的 的 的”）刷高精确率，BLEU 采用**截断词频统计（Modified n-gram Precision）**：某个 n-gram 在候选文本中计入的匹配次数，不能超过其在单篇参考文本中出现的最大频次。

设候选文本中 $n$-gram 出现频次为 $\text{Count}(n\text{-gram})$，在参考文本中最高频次为 $\text{Max\_RefCount}(n\text{-gram})$，第 $n$ 阶截断精确率 $p_n$ 为：

$$p_n = \frac{\sum_{n\text{-gram} \in C} \min(\text{Count}(n\text{-gram}), \text{Max\_RefCount}(n\text{-gram}))}{\sum_{n\text{-gram} \in C} \text{Count}(n\text{-gram})}$$

如果仅看精确率，模型只输出 1 个命中词就能拿到满分。因此 BLEU 引入了**简短惩罚因子（Brevity Penalty, BP）**。设候选文本长度为 $c$，参考文本长度为 $r$：

$$BP = \begin{cases} 1 & \text{若 } c > r \\ \exp\left(1 - \frac{r}{c}\right) & \text{若 } c \le r \end{cases}$$

最终的 BLEU-N 是各阶精确率的几何加权平均乘以惩罚因子（通常取 $N=4$，权重 $w_n = 1/4$）：

$$\text{BLEU} = BP \cdot \exp\left( \sum_{n=1}^{N} w_n \ln p_n \right)$$

#### 1.2 ROUGE：摘要任务导向的召回指标

ROUGE（Recall-Oriented Understudy for Gisting Evaluation）常用于文本摘要评估，核心关注**召回率（Recall）**：参考答案中的核心信息，有多少被模型生成的内容覆盖。

*   **ROUGE-N**：统计参考文本在候选文本中的 $n$-gram 召回率。
*   **ROUGE-L**：基于最长公共子序列（Longest Common Subsequence, LCS）计算匹配度。LCS 允许匹配词之间存在间隔，不强制要求连续硬拼，保留了宏观语序关系。设参考文本长度为 $m$，候选文本长度为 $n$，二者最长公共子序列长度为 $\text{LCS}(\text{Ref}, \text{Cand})$：
    $$R_{\text{LCS}} = \frac{\text{LCS}(\text{Ref}, \text{Cand})}{m}, \quad P_{\text{LCS}} = \frac{\text{LCS}(\text{Ref}, \text{Cand})}{n}$$
    $$\text{ROUGE-L} = \frac{(1 + \beta^2) R_{\text{LCS}} P_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 P_{\text{LCS}}}$$
    工程中通常取 $\beta=1$，即召回率与精确率的调和平均值 $F_1$。

![BLEU与ROUGE词法匹配机制对比](/images/content-rebuild/f1888a4c0c12f983/fig-1.png)

#### 1.3 算例演练与词法指标的失效场景

我们对比两个极简序列：
*   **参考答案**：`the cat is on the mat`（长度 $r=6$）
*   **候选文本**：`the cat sat on the mat`（长度 $c=6$）

1.  **1-gram 计算**：候选词为 `the`(2次)、`cat`(1次)、`sat`(1次)、`on`(1次)、`mat`(1次)。参考词中 `the` 最多出现 2 次，其余词除 `sat`（0次）外均出现 1 次。截断匹配数为 $2 + 1 + 0 + 1 + 1 = 5$，因此 1-gram 精度 $p_1 = 5 / 6 \approx 0.833$。
2.  **2-gram 计算**：候选 2-gram 包括 `the cat`、`cat sat`、`sat on`、`on the`、`the mat`（共 5 个）。匹配参考答案的有 `the cat`、`on the`、`the mat`（共 3 个），因此 $p_2 = 3 / 5 = 0.600$。
3.  **ROUGE-L 计算**：最长公共子序列为 `the cat on the mat`，长度为 5。召回率与精确率均为 $5/6$，故 ROUGE-L 为 0.833。

**贯穿用例测试**：
将前面的回答 A 与回答 B 代入词法指标计算：
*   **回答 A（同义改写）**：由于替换了“非甾体抗炎药 $\to$ 抗炎止痛类药物”、“减弱降压药疗效 $\to$ 影响降压药作用”，BLEU 与 ROUGE-L 得分大幅跌落。
*   **回答 B（致命反义）**：全句仅增加了“不”、“不会”和“放心日常”，其余 80% 以上的 n-gram 与参考答案完全重合，BLEU 和 ROUGE 均给出极高分！

这表明：**词法匹配指标仅度量表面字符串重叠，无法感知同义表达，更无法识别事实矛盾。** 更多生成指标分类细节可参考[/fundamentals/evaluation-metrics/generation-metrics.md](/fundamentals/evaluation-metrics/generation-metrics.md)。

---

### 2. 基于表示匹配的语义相似度度量：BERTScore

为解决同义替换无法命中的问题，BERTScore 引入预训练 Transformer 模型（如 RoBERTa、DeBERTa）提取上下文词向量，并在稠密向量空间中计算语义对齐度。

#### 2.1 贪婪匹配（Greedy Matching）对齐机制

设参考文本词元序列为 $x = \langle x_1, \dots, x_k \rangle$，候选文本词元序列为 $\hat{x} = \langle \hat{x}_1, \dots, \hat{x}_l \rangle$：
1.  通过编码器提取上下文嵌入向量，并做 $L_2$ 归一化：$\mathbf{x}_i = \frac{\mathbf{e}(x_i)}{\|\mathbf{e}(x_i)\|}$，$\mathbf{\hat{x}}_j = \frac{\mathbf{e}(\hat{x}_j)}{\|\mathbf{e}(\hat{x}_j)\|}$。
2.  计算所有词元对之间的余弦相似度矩阵 $\mathbf{S}_{i,j} = \mathbf{x}_i^\top \mathbf{\hat{x}}_j$。
3.  **非对称贪婪匹配**：为候选句中的每个词元，在参考句中寻找余弦相似度最高的词元做配对（计算精确率）；为参考句中的每个词元，在候选句中寻找相似度最高的词元做配对（计算召回率）。

BERTScore 的召回率、精确率与 $F_1$ 值公式为：

$$R_{\text{BERT}} = \frac{1}{|x|} \sum_{x_i \in x} \max_{\hat{x}_j \in \hat{x}} \mathbf{x}_i^\top \mathbf{\hat{x}}_j, \quad P_{\text{BERT}} = \frac{1}{|\hat{x}|} \sum_{\hat{x}_j \in \hat{x}} \max_{x_i \in x} \mathbf{x}_i^\top \mathbf{\hat{x}}_j$$

$$F_{\text{BERT}} = 2 \cdot \frac{P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}$$

![BERTScore贪婪匹配机制与向量对齐](/images/content-rebuild/f1888a4c0c12f983/fig-2.png)

#### 2.2 逆文档频率（IDF）加权

自然语言中，“的”、“在”、“了”等停用词在向量空间中往往具有普遍较高的相似度，但承载的核心语义很少。BERTScore 支持利用预先统计的 IDF 权重进行加权平均：

$$R_{\text{BERT}} = \frac{\sum_{x_i \in x} \text{idf}(x_i) \max_{\hat{x}_j \in \hat{x}} \mathbf{x}_i^\top \mathbf{\hat{x}}_j}{\sum_{x_i \in x} \text{idf}(x_i)}$$

#### 2.3 边界与局限

BERTScore 成功解决了回答 A 中“抗炎止痛”与“非甾体抗炎”的语义对齐。然而面对回答 B 时，预训练模型提取的高层向量由于各向异性（Anisotropy）现象，不相关词对的余弦相似度基底普遍集中在 0.6~0.7 之间；且自注意力对孤立否定词（如“不属于”）引发的逻辑反转不够敏感，导致回答 B 的 BERTScore 依然处于较高区间，无法作为事实合规的阻断门禁。

<ReadMoreLock />

---

### 3. 大模型作为裁判（LLM-as-a-judge）评测范式

面对回答 C 这种开放生成，或缺乏唯一标准参考答案的场景，词法匹配与嵌入相似度均失去基准。LLM-as-a-judge 范式通过提示词，驱动推理能力较强的模型对候选输出进行定性与定量裁判。

#### 3.1 单模型打分与两两对比评测

LLM 裁判主要有两种评测组织形式：

| 对比维度 | 单模型绝对打分（Direct Scoring） | 两两对比评测（Pairwise Comparison） |
| :--- | :--- | :--- |
| **输入组成** | 待测回答 + 评估量表（Rubric） | 模型 A 回答 + 模型 B 回答 + 评测提示词 |
| **输出格式** | 离散分值（如 1~5 分）及对应评语 | 胜者判定（A 胜 / B 胜 / 平局）及对比推演 |
| **计算开销** | $O(M)$，与待测模型数量 $M$ 呈线性关系 | $O(M^2)$ 或采用瑞士轮/Elo 采样对比 |
| **判别灵敏度** | 较低；模型打分容易趋于保守（集中在 3~4 分） | 较高；在质量接近时更容易迫使裁判作出细微优劣区分 |
| **常见应用** | 持续集成日常质量监控、异常熔断拦截 | 大版本发布选型、核心模型水平对比 |

![LLM-as-a-judge两种评估架构对比](/images/content-rebuild/f1888a4c0c12f983/fig-3.png)

#### 3.2 评分提示词编写规范

编写裁判 Prompt 时，如果只写“请给以下回答打分（1-5分）”，输出结果将充满随机性。构建高可靠裁判需遵循三项原则（相关上下文规范可参考[/frontier/context-engineering/high-quality-context.md](/frontier/context-engineering/high-quality-context.md)）：

1.  **明确每个分阶的操作定义（Rubric）**：分值必须与具体缺陷绑定，而非主观印象。
2.  **强制思维链先行（CoT-first）**：必须要求模型在输出最终分值前，先在独立标签内完成证据核对，防止自回归先生成了分数再强行圆场。
3.  **输出结构化结果**：统一使用 JSON 输出，包含理由与打分。

```markdown
[系统角色]
你是一名严谨的问答质量评审员。请依据评分细则对候选回答进行检验。

[评分细则]
- 5分：完全符合参考事实，无多余虚构，逻辑清晰直接。
- 3分：核心结论正确，但存在未证实的推论或明显冗余信息。
- 1分：包含与参考事实矛盾的致命错误，或答非所问。

[评测输入]
- 用户提问：{query}
- 参考依据：{reference}
- 待测回答：{candidate_answer}

[执行流程]
1. 在 <analysis> 标签内列举待测回答包含的每项事实声明，逐一核对是否与参考依据矛盾。
2. 在 <verdict> 标签内输出 JSON：{"score": 整数分值, "reason": "核验原因"}。
```

---

### 4. 裁判模型的固有偏置与工程校准

LLM 充当裁判并非绝对中立的度量衡。由于自回归生成机制和对齐阶段的人类偏好先验，裁判模型存在三类常见的系统性偏置。

![裁判大模型偏置成因与校准机制](/images/content-rebuild/f1888a4c0c12f983/fig-4.png)

#### 4.1 位置偏置（Position Bias）与双向交换校准

*   **现象**：在两两对比中，裁判模型倾向于选择排在前面的候选回答（位置 1），或者因长文本衰减倾向于选择排在末尾的回答（位置 2）。
*   **双向交换校准（Swap Calibration）**：
    设待测回答对为 $(A, B)$。系统执行两次独立的推理判定：
    *   第一次测试输入顺序为 $(A, B)$，记录胜者 $W_1$。
    *   第二次调换输入顺序为 $(B, A)$，记录胜者 $W_2$。
    *   **判定规则**：
        *   若 $W_1 = A$ 且 $W_2 = A$，判定模型 $A$ 胜；
        *   若 $W_1 = B$ 且 $W_2 = B$，判定模型 $B$ 胜；
        *   若两次判定产生冲突（例如两次都是排在前面的位置获胜），则判定为平局（Tie）或视作无效波动样本予以剔除。

#### 4.2 长度偏置（Verbosity Bias）与长度约束

*   **现象**：面对贯穿用例中的回答 C，大模型裁判容易倾向于给出更高评分。由于人类反馈对结构工整、段落丰富的文本具有天然奖励倾向，裁判模型常把“篇幅长、带 Markdown 排版”误判为“论述详实”。
*   **校准方案**：
    1.  **量表中明确扣分项**：在 Prompt 中增加“若回答包含无必要引申或废话套话，扣 1~2 分”。
    2.  **长度配对采样**：在两两对比评估集中，严格控制测试样本对的词元长度差在一定阈值内（如比例差距不超过 20%），隔离字数带来的影响。

#### 4.3 自模型偏好（Self-Enhancement Bias）

*   **现象**：特定家族微调的模型充当裁判时，对于同一模型家族生成的文本往往给出统计显著的偏高分数，因为二者的表达风格和高频词汇空间高度吻合。
*   **校准方案**：在发布验收等高敏感阶段，采用多裁判交叉评审（如组合不同架构、不同研发机构的开源和闭源权威模型），综合多数投票决定。

---

### 5. 人工评估设计与标注一致性检验

人工评估（Human Evaluation）是校准自动化指标和决定核心版本上线的黄金基准。但如果缺乏统一量表和质量检验，人工标注容易沦为缺乏一致性的主观感官判断。

#### 5.1 正交属性量表设计

人工评估应当避免只打一个笼统的“综合分”，建议将任务拆分为相互独立的四个维度：

*   **事实忠实度（Faithfulness）**：是否存在捏造事实、颠倒因果或逻辑违背（无幻觉）。
*   **指令遵循度（Instruction Following）**：是否落实了字数限制、排版格式或角色身份设定。
*   **信息完整度（Completeness）**：是否完整回答了问题涉及的所有子疑问。
*   **表达流畅度（Fluency）**：是否存在病句、错别字或前后矛盾语病。

#### 5.2 标注一致性量化：Fleiss' Kappa（$\kappa$）

当多个标注者对同一批文本样本进行离散分类标注时，必须剔除偶然碰巧一致的情况。Fleiss' Kappa 适用于固定数量标注员在无序名义分类（Nominal scale，如“合格/不合格/存疑”）任务中的一致性检验；若任务采用 1~5 分等具备强弱顺序的等级量表（Ordinal scale），则应采用加权 Kappa（Weighted Kappa）或组内相关系数（ICC），避免将等级间距视作无序离散类别而丢失距离信息。

设待评估样本总数为 $N$，每个样本由 $n$ 位标注者独立打分，评分选项共 $k$ 个离散类别：
1.  设 $n_{ij}$ 为把第 $i$ 个样本归入第 $j$ 类的标注者数量。计算第 $i$ 个样本的标注者一致度比例 $P_i$：
    $$P_i = \frac{1}{n(n-1)} \sum_{j=1}^{k} n_{ij}(n_{ij} - 1)$$
2.  计算所有样本的平均实际一致度 $\bar{P} = \frac{1}{N} \sum_{i=1}^{N} P_i$。
3.  计算类别 $j$ 在所有标注中所占的边际概率 $p_j$ 及随机预期一致度 $\bar{P}_e$：
    $$p_j = \frac{1}{Nn} \sum_{i=1}^{N} n_{ij}, \quad \bar{P}_e = \sum_{j=1}^{k} p_j^2$$
4.  计算最终 Fleiss' Kappa 系数：
    $$\kappa = \frac{\bar{P} - \bar{P}_e}{1 - \bar{P}_e}$$

![人工标注一致性检验与Fleiss Kappa计算](/images/content-rebuild/f1888a4c0c12f983/fig-5.png)

| $\kappa$ 数值区间 | 一致性程度解释 | 建议工程动作 |
| :--- | :--- | :--- |
| $< 0.40$ | 较差（Poor） | 量表定义模糊或培训不足，需重新校准评估标准后再标注 |
| $0.41 \sim 0.60$ | 中等（Moderate） | 存在边界模糊样本，需在量表中增加边界案例示例（Few-shot） |
| $0.61 \sim 0.80$ | 良好（Substantial） | 标注质量可靠，可用于模型迭代评估 |
| $> 0.80$ | 极高（Almost Perfect） | 标注高度一致，可固化为基准金标测试集 |

---

### 6. 场景选型矩阵与分层评测流水线

在实际工程中，单一度量工具无法兼顾计算成本与评测质量。应根据业务任务的容忍度与迭代周期，构建分层的混合评估流水线。

![工业级多场景评估指标组合矩阵](/images/content-rebuild/f1888a4c0c12f983/fig-6.png)

#### 6.1 四类典型场景指标选型

| 业务任务场景 | 首选核心指标 | 辅助与校验指标 | 选型原理解释 |
| :--- | :--- | :--- | :--- |
| **机器翻译** | COMET / BLEU-4 | BERTScore | 词元位置相对固定，BLEU 负责低成本底线拦截，COMET/BERTScore 校验语义对齐 |
| **文本摘要** | ROUGE-L / BERTScore | LLM 忠实度检测 | ROUGE-L 校验关键实体覆盖，BERTScore 允许近义归纳，LLM 裁判过滤虚构事实 |
| **RAG 检索问答** | 忠实度（Faithfulness） / 答案相关度（Answer Relevance） | 词法 Exact Match | 核心是回答能否被检索文档支撑，推荐使用 [Ragas 框架](https://docs.ragas.io/en/stable/)，详解可查阅[/fundamentals/evaluation-metrics/rag-evaluation.md](/fundamentals/evaluation-metrics/rag-evaluation.md) |
| **开放创意与对话** | 两两对比 LLM 裁判 | 分层人工抽检 | 缺乏标准答案，词法重叠指标失效，依赖双向校准的大模型裁判初筛与人工终审 |

#### 6.2 CI/CD 三层评估漏斗

在研发流水线中，不能对每次代码提交都调用昂贵的大模型裁判。推荐构建三层分流漏斗：

1.  **第一层：轻量门禁（代码 Commit 触发）**：运行百级别标准化测试集，使用 BLEU、ROUGE 或关键词匹配，在几秒内拦截因 Prompt 改动导致的格式崩塌、关键字段缺失等明显回归。
2.  **第二层：离线裁判抽检（合并请求 PR / 每日构建触发）**：抽取 100~300 个典型样本，运行 BERTScore 以及经过双向交换校准的 LLM 裁判，统计整体胜率与事实忠实度，拦截逻辑退化。
3.  **第三层：发布准出大盘（版本正式上线前）**：从线上真实日志抽取脱敏数据，交由标注员基于多维量表进行双盲盲测，在确认 $\kappa > 0.65$ 后作为最终上线准出依据。

关于模型在自由文本中产生事实违背的根因，可进一步阅读[/llm-100/evaluation/hallucination-causes.md](/llm-100/evaluation/hallucination-causes.md)。

---

### 7. 可检验练习

#### 练习目标
通过手动与简易代码计算，亲身体会词法匹配指标与人工一致性检验的内部计算过程。

#### 练习 1：手算 BLEU-1 与简短惩罚因子
*   **参考答案**：`red apple on the tree`（长度 $r=5$）
*   **待测回答**：`red red apple`（长度 $c=3$）
*   **任务**：
    1.  计算待测回答的截断 1-gram 精度 $p_1$。
    2.  计算简短惩罚因子 $BP$。
    3.  计算 BLEU-1 得分（$BP \times p_1$）。

#### 练习 2：小规模标注一致性验算
设 3 位标注员（$n=3$）对 3 个文本生成样本（$N=3$）是否合格进行标注（$k=2$，“合格”记为类别 1，“不合格”记为类别 2）：
*   样本 1：3 人均判“合格”（$n_{11}=3, n_{12}=0$）
*   样本 2：2 人判“合格”，1 人判“不合格”（$n_{21}=2, n_{22}=1$）
*   样本 3：3 人均判“不合格”（$n_{31}=0, n_{32}=3$）
*   **任务**：依次计算各样本的 $P_i$、整体 $\bar{P}$、边际概率 $p_1, p_2$、预期一致度 $\bar{P}_e$ 及 Fleiss' Kappa 系数 $\kappa$。

<details>
<summary>点击查看参考答案与解析</summary>

**练习 1 答案**：
1.  待测词频：`red` 出现 2 次，`apple` 出现 1 次。参考答案中 `red` 最多出现 1 次，`apple` 最多出现 1 次。因此截断分子为 $\min(2, 1) + \min(1, 1) = 1 + 1 = 2$。总词数为 3，截断精度 $p_1 = 2/3 \approx 0.667$。
2.  因 $c = 3 \le r = 5$，$BP = \exp(1 - 5/3) = \exp(-2/3) \approx 0.513$。
3.  $\text{BLEU-1} = 0.513 \times 0.667 \approx 0.342$。

**练习 2 答案**：
1.  计算单样本一致度 $P_i = \frac{1}{n(n-1)} \sum n_{ij}(n_{ij}-1) = \frac{1}{6} \sum n_{ij}(n_{ij}-1)$：
    *   样本 1：$\frac{1}{6} [3 \times 2 + 0] = 1.0$
    *   样本 2：$\frac{1}{6} [2 \times 1 + 1 \times 0] = \frac{2}{6} \approx 0.333$
    *   样本 3：$\frac{1}{6} [0 + 3 \times 2] = 1.0$
    *   平均值 $\bar{P} = \frac{1.0 + 0.333 + 1.0}{3} = \frac{2.333}{3} \approx 0.778$
2.  计算边际概率：
    *   类别 1（合格）总票数 $= 3 + 2 + 0 = 5$。总标注次数 $= 3 \times 3 = 9$。$p_1 = 5/9 \approx 0.556$。
    *   类别 2（不合格）总票数 $= 0 + 1 + 3 = 4$。$p_2 = 4/9 \approx 0.444$。
3.  计算预期一致度：$\bar{P}_e = p_1^2 + p_2^2 = (5/9)^2 + (4/9)^2 = \frac{25 + 16}{81} = \frac{41}{81} \approx 0.506$。
4.  计算 Kappa：代入精确分数 $\kappa = \frac{\bar{P} - \bar{P}_e}{1 - \bar{P}_e} = \frac{22/81}{40/81} = \frac{22}{40} = 0.55$（若使用上述保留三位小数代入：$\frac{0.778 - 0.506}{1 - 0.506} = \frac{0.272}{0.494} \approx 0.551$）。
    结论：一致性处于中等水平（Moderate），说明样本 2 属于边缘案例，需补充打标细则明确判定标准。
</details>
