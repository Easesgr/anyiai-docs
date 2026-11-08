---
title: 大模型应用怎么做自动化评测？从核心指标到CI/CD门禁
icon: mdi:flask-outline
description: 介绍大模型应用评测体系的搭建方法，涵盖评测集构建、LLM-as-a-Judge裁判偏差校准、忠实度与召回率等关键指标计算，以及CI/CD分层门禁与线上回流实践。
---

大模型工程落地中，系统变更不仅包括代码逻辑的修改，还频繁涉及提示词微调、检索超参数调整（如检索候选数量 Top-K、重排阈值）、模型版本更迭以及底层知识库更新。如果缺乏客观、可复现的度量工具，开发团队往往陷入“修好一个 Case、改坏三个场景”的困境。

以一个典型的“企业 Kubernetes 运维助手”为例：当团队把系统提示词调整得更客气，或者把向量检索召回篇数从 3 篇提高到 5 篇后，经常出现原有“Pod CrashLoopBackOff 排查步骤”的回答被不相关的网络插件配置挤占，甚至生成了不存在的 kubectl 参数。在上一讲 [/architecture/logging-trace.md](/architecture/logging-trace.md) 中，我们构建了追踪调用链、定位通信延迟的 Trace 观测能力；日志与 Trace 解决的是运行时链路“通不通”与“快不快”的问题，而评测体系解决的是生成内容“好不好”与“对不对”的质量度量问题。

![大模型应用评测体系全景](/images/content-rebuild/d9846430d8cb799a/cover.png)

*图说明：本图展示大模型应用评测体系的全景链路，读者应沿左侧测试集输入、中间 LLM-as-a-Judge 自动化度量、右侧 CI/CD 门禁与在线回流的流向建立全局认知。*

---

## 1. 评测体系的三层维度

大模型系统评测不能混淆通用基础能力与垂直业务成效。工程上通常将评测对象拆解为自底向上的三层维度：

1. **基础模型层（Foundation Model Capability）**：评测通用语言理解、指令遵从、代码编写、数学推理及上下文窗口吞吐能力。这部分主要参考标准化基准测试（如 MMLU、GSM8K、HumanEval），在模型选型与基底升级时作为能力基线，参见 [/fundamentals/evaluation-metrics/llm-benchmarks.md](/fundamentals/evaluation-metrics/llm-benchmarks.md)。
2. **应用系统层（RAG / Agent Quality）**：评测检索与生成的协同效果，重点度量文本检索相关性、上下文利用效率、工具调用参数准确率以及内容真实度。这是应用开发团队最主要的把控层，通用基准无法替代。更详细的基础概念可阅读 [/fundamentals/evaluation-metrics/rag-evaluation.md](/fundamentals/evaluation-metrics/rag-evaluation.md)。
3. **业务交付层（Business Impact）**：评测最终用户对交付结果的接受程度，指标包括任务达成率、运维工单排障耗时、用户采纳率（Acceptance Rate）及安全合规达标率。

![评测体系三层维度与指标映射](/images/content-rebuild/d9846430d8cb799a/fig-1.png)

*图说明：本图展示基础模型、应用系统、业务交付三层评测体系及其对应的关键度量指标，读者应注意下层能力是上层效果的必要条件而非充分条件。*

在日常架构维护中，下层能力是上层的必要条件但非充分条件。即使选用了评测榜单第一的基础大模型，如果知识库检索切片出现偏差或系统提示词发生冲突，应用系统依然会给出错误命令。

---

## 2. 评测数据集构建与维护

缺乏高质量测试集时，任何打分算法都会退化为无效计算。工程测试集通常由三个互补的部分构成：

### 2.1 黄金测试集（Golden Dataset）
黄金测试集是由业务领域专家、标注人员与测试工程师共同精细化构建的高可信度基准库。每个样本通常包含五个确定字段：
- `question`：用户输入的原生意图文本，如“Pod 状态显示 CrashLoopBackOff 怎么定位？”；
- `contexts`（可选）：经人工核验过的标准支持文档片段（如 Kubernetes 官方排障手册特定章节）；
- `ground_truth`：人工编写的标准参考答案或关键事实命题列表；
- `constraints`：输出限制条件（如必须输出 bash 命令块、禁止提供需要 root 权限的高危指令）；
- `category`：用例归属类别（如 Pod 调度故障、网络路由异常、多跳排障推理、越权拒答测试）。

黄金集规模通常在 200 到 1000 条之间，要求标签准确度极高，作为生产版本发布的硬性准入门槛。

### 2.2 合成测试用例（Synthetic Data）
利用高能力模型对企业知识库未覆盖的长尾切片进行逆向提问生成。典型步骤包括：
1. 从知识切片（Chunk）中提取核心实体与故障因果事实；
2. 提示大模型生成针对该切片的正向提问、反事实虚假提问与需要多篇文档比对的复合提问；
3. 通过规则过滤器（如重复度计算、长度限制、语法校验）剔除低质样本。

合成数据集的优势在于能够低成本扩充规模至数万条，用于压力测试与鲁棒性扫描，但不能直接替代人工黄金集作为最终准入标准。

### 2.3 线上 Bad Case 回流管道
真实用户流量是评测集保持更新的来源。当线上产生显式负反馈（如点踩）、会话异常中断或安全拦截时，系统捕获完整的上下文与 Trace ID，经脱敏清洗后写入候选池。经人工专家二次确认与修正后，样本正式合并入黄金回归库，防止历史缺陷二次复发。

<ReadMoreLock />

![测试集构建与回流管道架构](/images/content-rebuild/d9846430d8cb799a/fig-2.png)

*图说明：本图展示人工黄金集、合成测试集与线上 Bad Case 经脱敏过滤并汇总至版本回归库的完整管道流向。*

---

## 3. 自动化评测方法（LLM-as-a-Judge）与偏差缓解

人工评估质量高但无法满足每次代码提交的敏捷验证；传统字符串匹配指标（BLEU、ROUGE）对于语义发散的开放生成任务相关性极低。使用具备强推理能力的大模型作为裁判（LLM-as-a-Judge）已成为自动化评测的常用解法，参见 [/fundamentals/evaluation-metrics/generation-metrics.md](/fundamentals/evaluation-metrics/generation-metrics.md)。

### 3.1 裁判提示词设计原则
裁判模型极易受到提示词引导性影响。评测提示词必须满足三个条件：
1. **离散分级锚定（Graded Rubric）**：避免让模型给出宽泛的 1-10 分，应使用带有明确判决锚点的 1-5 分制或分类标签（完全符合、部分符合、不符合），并列出每个档位的具体判定标准。
2. **先推导后打分（Chain-of-Thought First）**：强制要求模型在输出最终得分前，逐条陈述事实比对推导过程。若模型先给出分数，推理过程往往沦为后验合理化补充。
3. **单向结构化输出**：要求以严格 JSON 格式返回评分、理由与涉及的命题索引，以便程序稳定解析。

### 3.2 裁判模型系统性偏差与工程消除策略
大模型作为裁判存在四类显著偏差：

| 偏差类型 | 机制表现 | 工程缓解策略 |
| :--- | :--- | :--- |
| **位置偏见 (Position Bias)** | 在成对比较（Pairwise）任务中，模型倾向于判定排在前面或后面的候选更优。 | **双向翻转对比（Swap Position）**：每次评测执行两次判定（A在前与B在前），只有两次均胜出或均打平才采信，结果冲突判定为平局或分歧。 |
| **长度偏见 (Verbosity Bias)** | 裁判模型倾向于认为字数更多、排版更丰富、使用复杂修辞的回答更专业。 | **归一化约束**：在提示词中明确指令“不要因为答案更长而加分”；对于事实度量，将打分拆分为命题匹配数量比率而非整体主观打分。 |
| **同族模型偏好 (Self-enhancement Bias)** | 模型倾向于给同系列、同体系生成的文本打出更高分数。 | **异构模型交叉裁决**：避免使用待测模型的同系列模型单独裁决，改用第三方独立模型，或由双模型交叉仲裁。 |
| **打分压缩偏见 (Middle Bias)** | 模型在连续评分时倾向于扎堆给出 4 分或中庸分，区分度不足。 | **比较替代绝对打分**：在评测 Prompt 中使用带基准样本的 Pairwise 相对比对，或者直接采用二值化决策（Pass/Fail）。 |

### 3.3 评分一致性校准（Cohen's Kappa）
在正式将 LLM-as-a-Judge 接入流水线前，必须通过统计检验验证其与人工专家打分的一致性。使用 Cohen's Kappa 系数 $\kappa$ 评估二分类或离散标签的吻合程度：

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

其中 $P_o$ 为两者实际吻合的样本比例（观测一致率），$P_e$ 为两者仅靠偶然概率达成一致的理论期望率：

$$P_e = \sum_{k} P(\text{rater}_1 = k) \cdot P(\text{rater}_2 = k)$$

$\kappa$ 值的解释准则通常为：小于 0.4 为弱一致，0.4 至 0.75 为中等一致，大于 0.75 为强一致。若 $\kappa < 0.6$，说明评测提示词锚点不清晰或模型能力不足，不能作为自动化门禁指标。

以下 Python 脚本演示如何计算专家标注与 LLM 判定之间的 Cohen's Kappa：

```python
def calculate_cohen_kappa(human_labels: list[int], llm_labels: list[int]) -> float:
    """计算人类专家标注与 LLM 判定的一致性系数 (仅使用 Python 标准库)"""
    assert len(human_labels) == len(llm_labels), "样本数量必须一致"
    total = len(human_labels)
    if total == 0:
        return 0.0

    # 获取所有可能类别
    categories = sorted(list(set(human_labels + llm_labels)))
    
    # 1. 计算观测一致率 Po
    agree_count = sum(1 for h, l in zip(human_labels, llm_labels) if h == l)
    p_o = agree_count / total
    
    # 2. 计算偶然一致率 Pe
    p_e = 0.0
    for c in categories:
        p_human = sum(1 for h in human_labels if h == c) / total
        p_llm = sum(1 for l in llm_labels if l == c) / total
        p_e += p_human * p_llm
        
    # 3. 计算 Kappa
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)

# 验证算例：50 条样本判定（1 为合格，0 为不合格）
# 专家标注：40 条合格，10 条不合格
expert_eval = [1]*40 + [0]*10
# LLM 判定：前 40 条中判定 35 条合格、5 条不合格；后 10 条中判定 4 条合格、6 条不合格
model_eval  = [1]*35 + [0]*5 + [1]*4 + [0]*6  # 共有 9 处不一致

kappa = calculate_cohen_kappa(expert_eval, model_eval)
print(f"观测一致率 Po: {(35+6)/50:.4f}, Kappa 系数: {kappa:.4f}")
# 运行输出：观测一致率 Po: 0.8200, Kappa 系数: 0.4578
```

算例显示，即使观察一致率高达 82%，由于样本分布中合格用例占比较大（80%），扣除随机猜测后的实际 Kappa 系数仅为 0.4578，说明判定标准的区分度仍然偏弱。关于分类指标的统计学背景可参阅 [/fundamentals/ml-basics/evaluation-classification.md](/fundamentals/ml-basics/evaluation-classification.md)。

![LLM-as-a-Judge 翻转与校准机制](/images/content-rebuild/d9846430d8cb799a/fig-3.png)

*图说明：本图展示 Pairwise 评测中通过正反位置翻转（A/B Position Swap）与结构化 Rubric 约束消除判定偏见的控制机制。*

---

## 4. 关键指标度量定义

针对典型的 RAG 与 Agent 系统，评测重点分布在四个独立正交的指标上：

```
    [ Ground Truth ]                    [ User Query ]
           |                                  |
           | (Context Recall)                 | (Answer Relevance)
           v                                  v
  [ Retrieved Context ] --(Faithfulness)--> [ Generated Answer ]
                                              |
                                              | (Format Compliance)
                                              v
                                  [ Validated Output Schema ]
```

### 4.1 忠实度（Faithfulness）
度量生成答案是否完全由检索到的上下文支撑，用于量化幻觉率。
- **计算方法**：
  1. 让语言模型将生成答案拆解为原子事实命题集合 $S = \{s_1, s_2, \dots, s_n\}$；
  2. 针对检索出的上下文文本 $C$，让裁判模型逐一验证命题 $s_i$ 是否能由 $C$ 逻辑推导得出：
  $$\text{Faithfulness} = \frac{|\{s \in S \mid C \models s\}|}{|S|}$$
- **运维助手算例**：
  - 模型回答包含 3 个命题：“Pod 内存超限会触发 OOMKilled 状态($s_1$)，可以通过 kubectl describe pod 查看退出码 137($s_2$)，并应执行 kubectl edit limits 直接在线修改配额($s_3$)”；
  - 检索文档仅说明了前两项排障依据，并未提及可以在线修改 limits（甚至该操作在实际环境受限）；
  - 此时支撑命题为 2 个，$\text{Faithfulness} = 2 / 3 \approx 0.67$。

### 4.2 答案相关度（Answer Relevance）
度量生成答案是否切中用户问题，不论检索文档是否齐全。
- **计算方法**：
  根据生成的答案，利用大模型逆向生成 $M$ 个潜在问题 $\{q'_1, q'_2, \dots, q'_M\}$，计算逆向问题向量与原生用户问题向量 $q$ 的平均余弦相似度：
  $$\text{Answer Relevance} = \frac{1}{M} \sum_{j=1}^{M} \cos(E(q), E(q'_j))$$
  该指标的有效性依赖于嵌入模型（Embedding）的表征质量与逆向生成能力。如果运维助手在面对排障问题时给出了格式工整但答非所问的安全免责声明，逆向生成的问题将偏向法律免责，导致余弦相似度显著降低。

### 4.3 上下文召回覆盖率（Context Recall）
度量检索模块是否成功召回回答所需的核心事实。
- **计算方法**：
  以人工标注的黄金答案（Ground Truth）为基准，拆解为标准事实命题集合 $G = \{g_1, g_2, \dots, g_k\}$。检查检索召回切片 $C$ 覆盖了其中多少命题：
  $$\text{Context Recall} = \frac{|\{g \in G \mid C \models g\}|}{|G|}$$

### 4.4 格式遵从率（Format Compliance）
针对 Agent 与结构化输出场景，度量输出内容对 JSON Schema、XML 标签或工具入参规约的遵从性。度量指标属于确定性二值检验：
$$\text{Format Compliance} = \frac{\text{解析成功且符合 Schema 的调用数}}{\text{总调用请求数}}$$

![RAG 四元因果评估拓扑图](/images/content-rebuild/d9846430d8cb799a/fig-4.png)

*图说明：本图展示 Query、Retrieved Context、Generated Answer 与 Ground Truth 四个实体之间的语义蕴含、实体覆盖与相似度度量关系拓扑。*

---

## 5. CI/CD 评测门禁与持续回归

将评测脚本直接嵌入 GitLab CI 或 GitHub Actions 流水线，是防止提示词劣化与代码破坏的关键。但每次流水线全量运行大型评测集会导致外部 API 开销过高与构建超时，成本控制策略参见 [/architecture/token-cost.md](/architecture/token-cost.md)。

### 5.1 分层测试策略（Tiered Evaluation）
为了兼顾反馈速度与评估深度，工程上将回归集分级：
- **Tier 1 冒烟门禁（Smoke Test）**：包含 30-50 个涵盖核心运维功能的黄金用例与高频边缘 Case。代码分支每次发起 PR 时触发，使用轻量判决模型，运行时间通常控制在数分钟内。门禁阻断标准通常为：格式遵从率达到设定阈值（如要求 100% 符合 JSON Schema），忠实度不低于基准线。
- **Tier 2 全量回归（Full Regression）**：包含 500+ 个全量黄金集用例。在合并到主分支或夜间定时构建（Nightly Build）时触发，支持双向翻转对比与异构裁判，生成版本对比检验报告。

### 5.2 输入哈希与评测缓存
对于流水线中未修改的组件，不必重复调用评测模型。但评测流水线包含“待测应用生成回答”与“裁判模型判决得分”两个环节，评测缓存必须同时绑定待测流水线配置与裁判配置。计算 SHA-256 哈希值作为缓存 Key：
$$\text{Key} = \text{Hash}(\text{Pipeline\_Config} + \text{KB\_Version} + \text{Judge\_Config} + \text{Sample\_ID})$$
其中 $\text{Pipeline\_Config}$ 包含提示词模板、待测模型版本、检索超参数（Top-K、重排阈值）与工具代码逻辑；$\text{KB\_Version}$ 为知识库快照版本；$\text{Judge\_Config}$ 包含裁判模型与打分量表（Rubric）哈希。只有当待测流水线依赖项与裁判标准均未变更时才能复用历史打分；检索调优或知识库更新均会使缓存失效。

```python
# CI/CD 自动化门禁逻辑示例（Python）
def run_ci_eval_pipeline(branch: str):
    test_suite = "tier1_smoke.json" if branch.startswith("feature/") else "tier2_golden.json"
    dataset = load_dataset(test_suite)
    
    # 锁定待测流水线完整配置（含 Prompt、模型、检索参数 Top-K 与知识库快照哈希）
    pipeline_config = {
        "prompt_hash": hash_text(current_prompt_template),
        "model_id": current_model_id,
        "rag_params": {"top_k": current_top_k, "rerank_threshold": current_rerank_th},
        "kb_version": get_knowledge_base_version(),
    }
    # 锁定裁判模型与评测量表配置
    judge_config = {
        "judge_model_id": judge_model_id,
        "rubric_hash": hash_text(judge_rubric_template),
    }
    
    results = []
    for sample in dataset:
        cache_key = compute_eval_cache_key(sample["id"], pipeline_config, judge_config)
        if cache_exists(cache_key):
            results.append(get_cache(cache_key))
        else:
            output = execute_llm_pipeline(sample["question"])
            score = llm_as_a_judge(sample, output)
            save_cache(cache_key, score)
            results.append(score)
            
    # 门禁阻断逻辑
    metrics = aggregate_metrics(results)
    if metrics["format_compliance"] < 1.0:
        raise SystemExit("CI 拦截：格式遵从率低于 100%")
    if metrics["faithfulness_avg"] < 0.85:
        raise SystemExit(f"CI 拦截：平均忠实度 {metrics['faithfulness_avg']:.2f} 低于阈值 0.85")
    print("评测通过，允许构建打包")
```

![CI/CD 评测流水线门禁架构](/images/content-rebuild/d9846430d8cb799a/fig-5.png)

*图说明：本图展示从代码提交、哈希缓存分流、轻量 Tier 1 阻断到全量 Tier 2 生成质检报告的持续集成流程。*

---

## 6. 在线真实流量抽检与反馈回路

离线指标合格并不能覆盖所有生产未知异常。系统上线后，需要通过在线反馈回路维持持续监控。

### 6.1 显式与隐式用户反馈
1. **显式反馈（Explicit Signals）**：用户主动点击的赞、踩、复制按钮或提交的纠错反馈。踩反馈应附带结构化标签（如命令执行报错、建议内容过时、答非所问）。
2. **隐式反馈（Implicit Signals）**：通过交互日志挖掘的行为特征，包括：
   - **短时间改写重问**：用户在 60 秒内改写并重新提交相似问题，通常代表初次生成未能满足排障需求；
   - **命令复制**：生成代码块被用户完整复制，作为初步采纳信号；
   - **会话快速流失**：在多轮交互中迅速关闭窗口且无后续行为，可能指向负向体验。

### 6.2 抽检过滤与专家二次标注
线上全量请求调用裁判模型成本过高，系统通常采用低比例随机采样（如 1%~5%）或针对边缘特征进行条件过滤：
- 当用户标记为踩、模型响应长度异常短、或安全网关触发拦截告警时，请求记录全量投递到复核队列；
- 离线复核台将用户会话 Trace、检索切片与生成结果并排呈现给运维专家；
- 专家标注出具体出错环节（检索漏召回、模型忽略上下文、还是指令冲突），生成修正补丁并将脱敏后的 Case 打包入库，用于扩充回归测试集。

![在线流量抽检与专家二次标注闭环](/images/content-rebuild/d9846430d8cb799a/fig-6.png)

*图说明：本图展示在线流量经过显隐式指标采集、条件抽检池、人工复核台排查，最终回流更新黄金评测库的完整演进闭环。*

通过将离线门禁、自动化判定与在线长尾捕获协同连接，大模型工程系统才能在频繁的版本变更中保持指标受控与功能稳定。
