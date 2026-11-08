---
title: Prompt 是什么？如何设计提示词让大模型稳定输出？
icon: mdi:message-text-outline
description: 本文介绍提示词影响模型输出的生成机制，梳理原理认知、结构化设计与参数评测的三阶段学习路径，并提供评测驱动的迭代方法与可检验练习。
---

![Prompt 工程篇全景导引封面](/images/content-rebuild/46c5d5b44513ce05/cover.png)

### 贯穿任务：从模糊自然语言到确定结构

假设我们要开发一个客服系统，自动把用户的报修反馈提取为工单结构：

*   **用户输入**：“昨天系统更新后登录页一直白屏，重试三次都不行，影响我们整个部门打卡，麻烦赶紧处理！”
*   **目标输出**：`{"category": "login_error", "priority": "P0", "reason": "更新后登录白屏阻断打卡"}`

如果只给模型发送一句“请提取工单信息”，模型可能会返回一段带有客套话的自然语言，或者擅自将字段名改成 `issue_type`，甚至漏掉优先级评估。

大语言模型本质上是一个基于已输入文本预测下一个词元（Token）条件概率分布的自回归系统：

$$P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

模型并没有自主行动的意志。输入给模型的初始文本序列（Prompt），直接决定了它在预测下一个 Token 时的上下文依据。**Prompt 工程**并非碰运气的玄学试错，而是通过明确的指令规范、上下文边界划分、样本示范与推理步骤引导，使模型生成的文本稳定落在业务预期的格式与语义范围内。

---

### 机制链条：提示词如何影响模型的生成行为

从输入自然语言指令到程序稳定解析数据，中间经历了明确的计算流程：

![提示工程自底向上的因果约束链路图](/images/content-rebuild/46c5d5b44513ce05/fig-1.png)

1. **前缀上下文输入**：输入文本经过词元化（Tokenization）切分成离散的 Token 序列。Prompt 中的角色设定、背景上下文和任务目标，共同构成了初始自注意力计算的键（Key）和值（Value）。
2. **注意力计算与引导**：在因果自注意力机制下，每一个新生成的 Token 都可以关注到 Prompt 中所有的历史 Token。清晰的格式约束（如“仅输出 JSON”）会显著提高词表中符合 JSON 语法符号（如 `{`、`"`）的预测概率（Logits）。
3. **采样空间调控**：模型在词表上计算出未归一化的对数概率后，经过 Temperature、Top-p 等采样参数进行缩放和截断，决定最终选择哪一个候选 Token。较低的 Temperature 会让高概率词的选出概率更集中，减少格式漂移。
4. **输出契约验证**：提示工程的目标是让文本可以被程序可靠消费。由于纯自然语言提示无法 100% 杜绝语法错误，生产环境通常结合结构化模版、JSON Schema 约束解码或程序后置断言来确保契约有效。

下面是一段最小可验证的契约检验代码，用于检查模型输出是否符合下游程序预期的 JSON 规范：

```python
import json

def validate_llm_json_contract(raw_output: str, required_keys: set) -> dict:
    """
    验证大模型输出是否满足预期的 JSON 契约约束
    """
    try:
        # 1. 尝试解析 JSON 数据结构
        parsed = json.loads(raw_output.strip())
    except json.JSONDecodeError as err:
        return {"valid": False, "error": f"JSONDecodeError: {str(err)}", "data": None}
    
    # 2. 验证预期的字段完整性（需确保根节点为字典类型）
    if not isinstance(parsed, dict):
        return {"valid": False, "error": f"TypeError: Expected dict, got {type(parsed).__name__}", "data": parsed}
        
    missing_keys = required_keys - set(parsed.keys())
    if missing_keys:
        return {"valid": False, "error": f"MissingKeys: {list(missing_keys)}", "data": parsed}
        
    return {"valid": True, "error": None, "data": parsed}

# 验证示例：检查模型返回的文本是否包含完整字段
mock_model_response = '{"category": "login_error", "priority": "P0", "reason": "更新后登录白屏阻断打卡"}'
contract_keys = {"category", "priority", "reason"}

result = validate_llm_json_contract(mock_model_response, contract_keys)
assert result["valid"] is True
print("契约验证通过，解析结果：", result["data"])
```

---

### 学习路径：三阶段知识体系

掌握提示工程需要由浅入深，跨越原理理解、结构编写与工程控制三个阶段：

![提示工程三阶段进阶路径图](/images/content-rebuild/46c5d5b44513ce05/fig-2.png)

*   **阶段一：机制理解（原理层）**：理解自回归生成机制如何依赖前缀上下文；掌握 System、User、Assistant 等对话角色标记（如 ChatML 模板）的用途与角色指令层级差异；对比零样本提示（Zero-shot）、少样本示范（Few-shot）与思维链（Chain-of-Thought, CoT）在推理复杂度上的适用条件。
*   **阶段二：结构化实操（技术层）**：学习利用 Markdown 区块与 XML 标签清晰划分任务指令与用户数据；掌握稳定输出 JSON 的模版技巧；严格区分越狱（Bypass Guardrails，绕过安全限制）与提示注入（Prompt Injection，外部不可信数据篡改指令）的原理与防御方案。
*   **阶段三：系统级控制（工程层）**：理解 Temperature 与 Top-p 等采样参数对生成多样性的控制范围；分析服务并发、批处理与浮点计算对输出一致性的影响；掌握基于测试集的评测驱动迭代流程。

---

### 章节目录与依赖导引

本模块包含 10 篇进阶教程，按顺序建立提示词工程的实践能力：

| 课时顺序 | 章节名称与链接 | 核心学习目标 | 推荐前置阅读 |
| :--- | :--- | :--- | :--- | 
| 01 | [Prompt 为什么能影响大模型输出？](/llm-100/prompt-engineering/Prompt_为什么能影响大模型输出.md) | 掌握自回归预测、注意力权重分配与上下文学习机制 | 深度学习基础 |
| 02 | [System Prompt、User Prompt、Assistant Prompt 分别有什么作用？](/llm-100/prompt-engineering/System_Prompt_User_Prompt_Assistant_Prompt_分别有什么作用.md) | 掌握 ChatML 等模板的特殊 Token 标记与角色指令层级 | 01 讲 |
| 03 | [Zero-shot、Few-shot、Chain-of-Thought 有什么区别？](/llm-100/prompt-engineering/Zero-shot_Few-shot_Chain-of-Thought_有什么区别.md) | 对比样本示范、单步问答与显式推理链的适用场景 | 01 讲 |
| 04 | [CoT 思维链提示为什么有效？](/llm-100/prompt-engineering/CoT_思维链提示为什么有效.md) | 从计算量扩充与自回归语义解耦解释 CoT 机制 | 03 讲 |
| 05 | [结构化 Prompt 应该怎么写？](/llm-100/prompt-engineering/结构化_Prompt_应该怎么写.md) | 掌握 XML 标签、Markdown 边界排版与角色分工模版 | 02 讲 |
| 06 | [如何让大模型稳定输出 JSON？](/llm-100/prompt-engineering/如何让大模型稳定输出_JSON.md) | 掌握 Schema 约束、Few-shot 强化与工程降级兜底 | 05 讲 |
| 07 | [Prompt 注入攻击是什么？如何防范？](/llm-100/prompt-engineering/Prompt_注入攻击是什么_如何防范.md) | 识别直接越狱与间接注入，实施输入分流与沙箱隔离 | 05 讲 |
| 08 | [Temperature、Top-p、Top-k 分别控制什么？](/llm-100/prompt-engineering/Temperature_Top-p_Top-k_分别控制什么.md) | 理解采样截断、缩放对概率分布平滑度的数学影响 | 01 讲 |
| 09 | [为什么同一个 Prompt 每次输出不一样？](/llm-100/prompt-engineering/为什么同一个_Prompt_每次输出不一样.md) | 分析浮点非结合律、并行批处理与非确定性采样成因 | 08 讲 |
| 10 | [面试高频：如何设计一个高质量 Prompt？](/llm-100/prompt-engineering/面试高频_如何设计一个高质量_Prompt.md) | 掌握评测集构建、定点归因迭代与成本收益边界考量 | 01 ~ 09 讲 |

---

### 评测驱动：如何科学迭代 Prompt

根据 Anthropic 官方的最佳实践指南，提示工程的第一步并非盲目修改提示词，而是先明确成功指标并建立评测集。

![提示工程测试驱动与评测迭代闭环图](/images/content-rebuild/46c5d5b44513ce05/fig-3.png)

如果缺乏测试集，修复一个边界问题往往会导致另一个常见场景失效。规范的评测迭代包含五个环节：

1. **构建基准集（Golden Dataset）**：收集 30~100 个典型输入，包含常规案例、容易混淆的边界案例（如描述模糊的报修）以及包含指令的对抗样本。
2. **确定量化指标**：对抽取和分类任务，计算字段完全匹配率（Exact Match）与 JSON 解析合法率；对开放生成任务，建立规则评分或使用评测模型评分。
3. **运行基线测试（Baseline）**：运行初版 Prompt，统计指标得分、平均消耗的 Token 数与接口响应延迟。
4. **归因与定点修订**：抽取失败案例分析原因。如果是字段缺失，补充 Few-shot 样例；如果是理解偏差，细化判断规则。避免为了单个失败样本大幅重写整体架构。
5. **全量回归验证**：运行修改后的 Prompt，确保全量评测集的准确率提升，且核心高频用例没有发生负向回退。

---

### 进阶练习：设计结构化提示词

**练习目标**：改进以下存在缺陷的原始提示词，使其稳定提取工单信息并输出合法 JSON。

**待优化提示词（原始）**：
```text
请帮我分析用户的客服反馈，提取出分类、优先级和原因。
```

**优化要求**：
1. 使用 XML 标签（如 `<rules>`、`<input>`）明确拆分规则说明与待处理数据。
2. 规定 `priority` 只能在 `["P0", "P1", "P2"]` 中选取，并给出明确判定标准（影响团队核心流程为 P0，影响个人操作为 P1，一般体验建议为 P2）。
3. 限制只返回标准 JSON 字符串，不得输出多余的解释文字。

**参考改写方案**：
```text
你是一个工单分类助手。请阅读 <input> 中的用户反馈，按照 <rules> 提取结构化信息，并严格输出为 JSON 格式。

<rules>
1. 字段定义：
   - category: 故障类别字符串（例如 login_error, performance, feature_request）
   - priority: 优先级，仅限 "P0"、"P1"、"P2"。阻断核心业务或团队打卡为 P0；单人功能受阻为 P1；体验建议或咨询为 P2。
   - reason: 简要原因说明（20字以内）。
2. 输出要求：只输出合法的 JSON 字符串，不要包含 markdown 标记或任何前导、后置说明文字。
</rules>

<input>
昨天系统更新后登录页一直白屏，重试三次都不行，影响我们整个部门打卡，麻烦赶紧处理！
</input>
```
