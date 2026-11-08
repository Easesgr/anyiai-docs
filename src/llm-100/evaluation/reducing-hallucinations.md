---
title: 如何减少大模型幻觉？从输入约束、受限解码到后置核验
icon: mdi:book-open-page-variant
description: 系统讲解工程化抑制大模型幻觉的实用方案，涵盖提示词认知边界、结构化检索注入、受限解码与基于 NLI 的后置一致性校验。
---

在上一讲[幻觉是什么？为什么大模型会胡说？](/llm-100/evaluation/hallucination-causes.md)中，我们了解了大模型幻觉的来源：自回归的预测目标追求词元序列的概率合理性，而非事实真伪；预训练对海量语料进行了有损压缩；注意力在超长上下文中可能被稀释。这意味着仅靠扩大模型参数量或寄希望于单一提示词，无法彻底根除幻觉。

在实际软件工程中，减少幻觉的实用思路是：**将不可控的自由文本生成，拆解为带有证据输入、过程约束与结果校验的受控工程流水线**。通过在输入前、推理中和生成后设置三道检查点，我们可以大幅降低错误信息传递给终端用户的概率。

本讲将以一个**企业财务问答系统**（如查询“某公司 2023 年第一季度研发费用与营收占比”）为例，系统拆解工程中行之有效的三阶段防幻觉方案。

![幻觉治理全景防御流水线](/images/content-rebuild/5064a229ed164b10/cover.png)

---

## 幻觉治理的三道工程防线

大模型系统防幻觉通常分为三个相互衔接的阶段：

1. **前置输入约束**：在组装 Prompt 时，通过提供结构化参考文档（RAG）、建立明确的拒答触发条件以及隔离外部输入，让模型生成内容时有据可循。
2. **推理采样控制**：在模型自回归生成 Token 时，调整采样参数（如降低 Temperature）来抑制低概率词的随机跳跃，或使用有限状态机进行受限解码，保证输出严格符合格式定义。
3. **后置核验拦截**：文本生成后，通过自动化程序拆解句子并进行一致性核验（如自然语言推理 NLI），对缺乏上下文支撑的内容做降级、重试或拦截。

![幻觉防御三层流水线架构](/images/content-rebuild/5064a229ed164b10/fig-1.png)

如上图所示，输入层负责注入有效证据，推理层负责压制无关发散，输出层负责核对事实是否与证据吻合。三层协同工作，各司其职。

---

## 输入端防御：边界提示与结构化注入

提示词工程防幻觉不是写“请千万不要胡说”，而是给模型建立清晰的行为边界与输入隔离机制。

### 1. 显式认知边界与拒答出口
自回归模型在遇到知识盲区时，由于补全句子的倾向，很容易顺着语义语法编造看似合理的虚构事实。如果不在提示词中提供合法的“不知道”路径，模型就会被迫编造。

在企业财务问答场景中，可以采用如下约束：

```text
【核心约束】
1. 你的回答必须严格基于【参考资料】中列出的原文事实。不得根据预训练知识推测资料中未提及的数值。
2. 如果【参考资料】中没有直接提及问题的明确答案，必须直接回复“根据现有资料无法确认”，不要进行推测。
3. 如果参考资料之间存在数据冲突，必须如实分别列出各文档的表述，禁止私自合并或平均数据。
```

### 2. 思维链拆解：抽取、推理与结论隔离
当用户提出复合问题（例如“2023 年 Q1 研发费用是多少？研发费用率相比去年同期变化了多少个百分点？”）时，如果要求模型单步直接给出答案，注意力很容易在复杂长句中偏离，导致数字张冠李戴。

工程实践中，建议将回答过程拆解为可观测的三阶段链条：

![抽取-推理-回答三阶段提示链对比结构](/images/content-rebuild/5064a229ed164b10/fig-2.png)

* **第一步：原文摘录**。要求模型先把参考资料中与研发费用、营收相关的原始句子完整摘抄出来。
* **第二步：步骤推导**。展示计算过程：$\text{研发费用率} = \frac{\text{研发费用}}{\text{营业收入}}$，并列出去年同期的基准数字。
* **第三步：精炼结论**。仅根据前两步得出的结果输出最终答复。

这种分步拆解的核心作用，在于强制让模型在生成推理和结论前，先在自己的上文（Context）中生成具体的证据片段，引导后续生成过程中的注意力始终落在真实数据上。

### 3. 多源检索上下文隔离与冲突处理
在 RAG（检索增强生成）场景中，检索到的片段可能来自不同年份的文档或不同部门的报告。如果直接拼接成纯文本，模型容易将不同时间段的数据混淆。

建议采用 XML 标签包裹检索片段，并显式标注元数据（如发布时间、权威等级）：

![带冲突标记的多源检索注入与证据隔离格式](/images/content-rebuild/5064a229ed164b10/fig-3.png)

```xml
<context>
  <doc id="doc_01" date="2023-04-20" source="官方一季度财报">
    2023年第一季度研发支出为 4,200 万元人民币。
  </doc>
  <doc id="doc_02" date="2023-05-15" source="券商研报估算">
    预计一季度研发支出约 4,500 万元。
  </doc>
</context>
```

提示词中补充规则：“当不同文档的数据不一致时，优先采用官方报告的数据，并在回答中说明研报估算情况。”这样可以防止模型在冲突信息面前随机选择或合成错误事实。

---

## 推理端控制：超参调节与受限解码

在模型生成 Token 阶段，我们可以通过控制采样概率分布和限制词表候选空间，来约束模型的输出行为。

### 1. 采样超参数调节的真实作用与边界
模型在每个生成步输出词表对应的未归一化对数概率（Logits $z_i$），通过 Softmax 和温度系数（Temperature $T$）转化为采样概率：

$$P(w_i | w_{<i}) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

* **降低 Temperature（例如设为 0 或接近 0）**：概率分布会变得更加陡峭，模型倾向于选择当前概率最高的词（贪心解码），从而抹平低概率的长尾 Token，大幅减少因随机采样导致的无端联想。
* **必须注意的边界**：降低温度只能解决“随机性带来的自由发挥”。如果模型自身的参数记忆本来就是错误的，或者 Prompt 输入的上下文本身含有错误信息，贪心解码依然会输出错误结果。调低温度并不是消除幻觉的充分条件。

### 2. 受限解码（Constrained Decoding）
在要求模型输出结构化数据（如抽取 JSON 格式的指标数据）时，模型可能会捏造不存在的字段名或输出非法的格式。

受限解码工具（如 Outlines、vLLM 的 Guided Decoding、llama.cpp 的 GBNF）通过在推理时引入**有限状态机（FSM）**来解决这一问题：

![Logits Mask受限解码状态机过滤机制](/images/content-rebuild/5064a229ed164b10/fig-4.png)

在每一步预测下一个 Token 时，FSM 根据预先定义的 JSON Schema 或正则表达式，判断当前哪些 Token 是语法合规的。对于不合规的 Token，程序直接在进入 Softmax 前将其 Logit 值强制修改为 $-\infty$（Logits Masking）。

* **适用范围**：受限解码能够保证输出格式和字段完全遵循预设规范（杜绝“格式幻觉”）。
* **局限性**：受限解码无法验证字段内容的真实性。例如它能保证输出 `"research_expense": <float>`，但填入的浮点数到底是真实财报的 4200 还是编造的 9999，受限解码无法判定。

---

## 推理时自洽性检验（Self-Consistency）

在需要多步数学计算或因果推导的复杂场景中，单次采样可能会在某一步推导中算错，导致最终结论偏差。自洽性检验（Self-Consistency）通过多次独立路径采样来提升可靠性。

![多路径采样收敛与发散投票机制](/images/content-rebuild/5064a229ed164b10/fig-5.png)

### 工作机制
1. **适度采样**：将 Temperature 设在适中水平（如 $0.5 \sim 0.7$），允许模型在推理表达上有适度探索。
2. **并行生成 $K$ 条路径**：对同一个问题和提示词，独立生成 $K$ 条带有推导过程的解答（例如 $K=5$）。
3. **提取结论并多数投票**：解析每条路径末尾的最终答案，选择出现频次最高的答案作为最终输出。

其经验假设在于：基于正确逻辑推导的路径，虽然推导措辞各有不同，但通常会汇聚到相同的最终数值；而幻觉编造的推理过程通常呈现高度发散和随机性，难以形成多数票。

### 成本权衡与局限
* **计算与延迟开销**：生成 $K$ 条路径意味着消耗的 Token 算力成本增加约 $K$ 倍。若采用并发或批量推理，端到端延迟取决于最长单条路径与系统并发能力，而串行生成则会导致响应耗时线性成倍增加。工程中一般采用早停策略：先并发请求 3 条路径，若答案完全一致则直接返回；存在分歧时再追加采样。
* **先验偏差局限**：如果模型存在系统性的常识偏见，或者参考文档本身存在强误导信息，多条路径可能会收敛到同一个错误答案上。因此多数投票并不能完全等同于事实核实。

---

## 后置自动化核验：基于 NLI 的一致性校验

文本生成完毕后，如何不依赖人工审查，通过程序自动判断回答是否忠实于参考文档？在当前的评估与质检实践中（例如 Ragas 的 Faithfulness 忠实度评估框架），自然语言推理（Natural Language Inference, NLI）是常用的自动化方案。

![原子事实抽取与NLI蕴含检验失败拦截流程](/images/content-rebuild/5064a229ed164b10/fig-6.png)

### 检验流水线设计

整个校验流程分为三个明确步骤：

1. **原子命题拆解**：调用轻量模型或规则，将模型生成的长篇回答拆解为多个不可再分的独立事实陈述（Atomic Claims）。
2. **逻辑关系判定**：以检索到的参考文档为前提（Premise），以每个原子命题为假设（Hypothesis），通过 NLI 模型（或特定微调模型）判断两者的逻辑关系：
   * **蕴含（Entailment）**：该陈述能够被参考文档充分证明。
   * **中立（Neutral）**：参考文档未提及该陈述。这说明陈述可能来自模型自身记忆或推测，缺乏证据支撑。
   * **矛盾（Contradiction）**：该陈述与参考文档内容直接冲突，属于明确的事实性幻觉。
3. **规则裁定与熔断**：汇总所有命题的判断结果，计算忠实度得分：
   $$\text{忠实度} = \frac{\text{被蕴含的命题数}}{\text{总命题数}}$$

### 伪代码实现

```python
from typing import List, Dict

def verify_answer_faithfulness(answer: str, context: str) -> Dict:
    """
    核验生成的回答是否忠实于上下文证据
    """
    # 1. 拆解原子命题
    claims: List[str] = extract_atomic_claims(answer)
    if not claims:
        return {"passed": True, "score": 1.0, "details": []}
    
    verification_results = []
    for claim in claims:
        # 2. 调用 NLI 判别器（如预训练的 Cross-Encoder 模型）
        # 输出标签: 'entailment', 'neutral', 'contradiction'
        relation, confidence = run_nli_inference(premise=context, hypothesis=claim)
        verification_results.append({
            "claim": claim,
            "relation": relation,
            "confidence": confidence
        })
    
    # 3. 统计指标并作出工程决策
    entailed_count = sum(1 for r in verification_results if r["relation"] == "entailment")
    contradiction_count = sum(1 for r in verification_results if r["relation"] == "contradiction")
    
    faithfulness_score = entailed_count / len(claims)
    
    # 判定规则：存在矛盾直接判定失败；若缺乏足够证据（中立过多导致得分低）也触发拦截
    has_conflict = contradiction_count > 0
    passed = (faithfulness_score >= 0.8) and (not has_conflict)
    
    return {
        "passed": passed,
        "score": faithfulness_score,
        "has_contradiction": has_conflict,
        "details": verification_results
    }
```

### 中立（Neutral）结果的处理策略
遇到 Neutral（未提及）时，并不意味着该陈述在客观世界中一定是假话，而是说明**系统无法从当前提供的参考文档中为它背书**。在严谨的问答场景中，系统可以采取以下动作：
* **提示降级**：在展示回答时，对缺乏证据的句子标注“未在参考文档中找到直接出处，请核验”。
* **触发重搜**：将 Neutral 命题重新作为查询词输入检索模块，寻找补充证据。
* **拦截重试**：要求模型重新生成，明确要求剔除该无依据的陈述。

关于大模型评估与质检指标的更多进阶知识，可参考[大模型如何评估？常见 Benchmark 有哪些？](/llm-100/evaluation/llm-evaluation-benchmarks.md)与[生成质量如何评价？BLEU、ROUGE、人工评估怎么选？](/llm-100/evaluation/generation-quality-metrics.md)。

---

## 可检验练习

### 练习 1
在构建 RAG 提示词时，为什么建议将外部检索文档放入具有唯一标识的 XML 标签（如 `<context id="doc_1">`）中，而不是直接作为普通段落拼在 Prompt 末尾？

**参考分析**：
1. 结构化标签可以明确区分系统指令、检索文档和用户问题，避免用户输入或检索文档中的文字与系统指令混淆（防范间接提示注入）。
2. 唯一的 `id` 允许模型在生成回答时附带显式引用标记（如 `[doc_1]`），便于后置程序对特定引用段落做自动化比对与溯源。

### 练习 2
在基于 NLI 的后置一致性校验中，如果模型生成了句子“公司总部位于上海”，而参考文档中只写了“公司在上海建立了大型研发中心”，NLI 模型判别该句子为 Neutral。此时是否应该立即向用户报错系统崩溃？应如何合理处置？

**参考分析**：
不应该报错系统崩溃。Neutral 仅代表“参考文档无法证明该陈述”，并不代表陈述客观错误。
合理的工程处置方案包括：
* 若系统为严格问答模式（如审计合规问答），系统可以自动过滤该句，或将回答降级为“根据资料，公司在上海设有大型研发中心，但未提及总部位置”。
* 若系统具备多轮检索能力，可将“公司总部所在地”作为补充 Query 再次触发检索，核查是否存在其他未被召回的文档支持该命题。

---

## 学习检验与下一讲预告

完成本讲学习后，你可以检验自己是否掌握了以下要点：
1. 能够区分提示词约束、受限解码和自洽性验证各自的技术边界与解决对象。
2. 清楚为什么将温度设为 0 无法解决事实记忆错误与输入错误。
3. 理解基于原子命题拆解与 NLI 分类的自动化核验流水线设计原理。

在下一讲中，我们将探讨大模型在工程化落地中的安全风险与攻防对抗：[大模型安全问题有哪些？提示注入、越狱与数据投毒](/llm-100/evaluation/llm-security-risks.md)。

### 本模块索引
* [/llm-100/evaluation/llm-evaluation-benchmarks.md](/llm-100/evaluation/llm-evaluation-benchmarks.md) | 大模型如何评估？常见 Benchmark 有哪些？
* [/llm-100/evaluation/generation-quality-metrics.md](/llm-100/evaluation/generation-quality-metrics.md) | 生成质量如何评价？BLEU、ROUGE、人工评估怎么选？
* [/llm-100/evaluation/hallucination-causes.md](/llm-100/evaluation/hallucination-causes.md) | 幻觉是什么？为什么大模型会胡说？
* [/llm-100/evaluation/reducing-hallucinations.md](/llm-100/evaluation/reducing-hallucinations.md) | 如何减少大模型幻觉？
* [/llm-100/evaluation/llm-security-risks.md](/llm-100/evaluation/llm-security-risks.md) | 大模型安全问题有哪些？
* [/llm-100/evaluation/content-safety-guardrails.md](/llm-100/evaluation/content-safety-guardrails.md) | 内容审核和安全护栏如何设计？
* [/llm-100/evaluation/llm-observability.md](/llm-100/evaluation/llm-observability.md) | 大模型应用如何做日志、监控和可观测性？
* [/llm-100/evaluation/rag-finetuning-agent.md](/llm-100/evaluation/rag-finetuning-agent.md) | 面试高频：RAG、微调、Agent 如何组合落地？
* [/llm-100/evaluation/llm-platform-design.md](/llm-100/evaluation/llm-platform-design.md) | 面试高频：从零设计一个大模型应用平台
* [/llm-100/evaluation/learning-roadmap.md](/llm-100/evaluation/learning-roadmap.md) | 大模型学习路线总结：从基础到 Agent 工程落地
* [/llm-100/evaluation/README.md](/llm-100/evaluation/README.md) | 评估、安全与面试综合篇
