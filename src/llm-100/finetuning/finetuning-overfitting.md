---
title: 大模型微调为什么会过拟合与遗忘？如何检测和缓解
icon: mdi:tune-vertical
tag:
- 微调与对齐
- 大模型100讲
- RLHF
description: 剖析大模型指令微调中的模板坍塌、实体死记与灾难性遗忘成因，掌握多维监控指标与防范方案。
---

在监督指令微调（SFT）大语言模型时，很多开发者会遇到这类情况：训练损失和验证损失持续下降，但在实际使用时，模型要么只会机械套用固定的客套句式，要么原本具备的代码与逻辑推导能力大幅衰退。

这种现象并不仅是传统机器学习中常见的样本拟合过度，还包含自回归生成模型特有的表面模板坍塌、实体死记硬背与灾难性遗忘。本文通过贯穿实际场景，梳理退化原因、监控指标与缓解方案。

![大模型微调退化机理与流形偏移全景](/images/content-rebuild/d3e8f7547d558720/cover.png)

---

## 贯穿场景：一个财务报销助手的“练废”过程

为了直观理解这一现象，我们以研发一个“企业财务报销与差旅助手”为例：

团队收集了 3,000 条内部制度问答，对开源基座模型进行全参数微调。为了让回答规范，训练样本的输出格式高度统一，开头全部是“您好！根据公司最新财务制度，为您解答如下：”，结尾全部带有标准免责声明。微调 5 个 epoch 后，测试人员发现了三个典型故障：

1. **只会说套话（模板坍塌）**：输入任何问题，哪怕是日常打招呼，模型都吐出一大段“根据公司最新财务制度...”，即便核心解答内容不完整，形式也必须凑满。
2. **换个问法就报错（实体死记与抗扰动差）**：问“飞机票报销标准是多少？”，模型能准确输出“经济舱实报实销”；但改成“出差坐飞机能报头等舱吗？”，模型却开始胡编或者输出断裂的制度编号。
3. **通用能力丢失（灾难性遗忘）**：让模型写一个简单的 Python 脚本汇总报销单金额，基座模型原本能轻松写出正确代码，现在却只能写出残缺、充满语法错误的片段。

上述三个现象，分别对应微调中的表面模板坍塌、实体过度拟合与灾难性遗忘。

---

## 为什么会过拟合：模板坍塌与实体过度拟合

传统分类模型的过拟合通常表现为“训练集准确率上升、测试集准确率下降”（可参考 [过拟合和欠拟合的基础定义](/fundamentals/ml-basics/overfitting-underfitting.md)）。但在以自回归预测为核心的大语言模型中，参数量通常达数十亿以上，而垂直领域的指令微调数据往往只有几千到几万条。模型表现出的过拟合有其独特的形态。

![表面模板坍塌与实体信息死记硬背机制对比](/images/content-rebuild/d3e8f7547d558720/fig-1.png)

### 1. 表面模板坍塌

在指令微调中，输入 Prompt 部分通常通过标签掩码忽略，仅对目标回复的 $T$ 个 token 计算自回归交叉熵损失，工程实现通常取 token 维度的平均值：

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \log P(y_t \mid x, y_{<t})$$

如果训练集中大量样本的前缀和结尾句式高度重复（例如每条回答都以固定的 20 个字客套话开头），优化器能以极快速度将这部分低熵 token 的损失降到接近零，从而显著拉低整体平均损失。此时总损失曲线迅速下行，掩盖了核心推理 token 的拟合不足，模型学到的主要是低熵的格式固定搭配。

一旦推理时用户给出的 prompt 风格偏离训练集，模型为了强行贴合学到的局部套路，会把概率集中在那些固定短语上，导致实际有用的推理信息变少、输出车轱辘话，词汇多样性（如 Distinct-N 指标）骤降。

### 2. 实体信息的死记硬背

微调主要用于调整模型的任务形式、行文风格和指令遵循模式，并不适合作为录入具体事实知识的主要手段（参见 [SFT 监督微调流程](/llm-100/finetuning/sft-instruction-tuning.md)）。

当我们将包含大量特定参数、制度细节、编号明细的样本直接送入微调时，模型倾向于直接调整前馈网络（MLP）中的权重来强行记住这些具体的 token 关联。这会带来两个隐患：
* **抗扰动能力差**：输入中的语序微调或同义词替换可能直接改变模型的激活路径，导致生成错误数字。
* **产生虚假绑定**：例如训练集中某个特定职级的差旅总与某一固定报销额度一同出现，模型会把二者强行关联，在处理其他不同场景时输出错误推断。

事实性、经常更新的知识，更适合通过外挂知识库检索（RAG）或工具调用注入上下文，而非强行通过微调写入模型参数。

---

## 为什么通用能力会退化：灾难性遗忘与约束代价

除了在特定垂直任务上拟合过度，微调最棘手的副作用是破坏基座原本的通用推理和多任务能力。

<ReadMoreLock />

### 1. 灾难性遗忘的梯度解释

基座模型在预训练阶段通过海量代码、数学推导与多语言文本，在参数空间中收敛到了一个兼顾多种能力的区域。在垂直任务微调时，损失函数 $\mathcal{L}_{\text{domain}}(\theta)$ 仅依据垂直领域的分布计算：

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_{\text{domain}}(\theta_t)$$

如果垂直任务的损失梯度 $\nabla_\theta \mathcal{L}_{\text{domain}}$，与保持通用能力所需的损失梯度 $\nabla_\theta \mathcal{L}_{\text{general}}$ 之间内积为负（即两任务梯度方向夹角大于 $90^\circ$）：

$$\nabla_\theta \mathcal{L}_{\text{domain}} \cdot \nabla_\theta \mathcal{L}_{\text{general}} < 0$$

由于参数更新沿垂直任务负梯度方向移动（$\Delta \theta = -\eta \nabla_\theta \mathcal{L}_{\text{domain}}$），通用任务损失的一阶变化量满足 $\Delta \mathcal{L}_{\text{general}} \approx \nabla_\theta \mathcal{L}_{\text{general}} \cdot \Delta \theta = -\eta (\nabla_\theta \mathcal{L}_{\text{domain}} \cdot \nabla_\theta \mathcal{L}_{\text{general}}) > 0$。这意味着垂直领域的每次参数更新都在推高通用任务损失，破坏原本维系代码编写、逻辑推演等通用能力的权重连接。在连续微调数百步后，通用能力就会出现明显衰退。

### 2. 强格式约束下的发散能力萎缩

为了让模型严格输出某种格式（例如纯 JSON 或特定表格），训练时往往对偏离格式的候选 token 施加强烈的负反馈。这种做法会强行压扁模型在解码时的预测概率分布，使条件熵大幅降低。其代价是削弱了模型的灵活联想和多角度分析能力，在面对较为开放或复杂的综合问题时，生成内容变得死板单调。

---

## 训练阶段的多维监控指标

不能仅凭领域内验证集（ID Validation Loss）是否下降来判断微调质量，因为领域验证集往往包含与训练集类似的句式风格和标注倾向。

![监控体系剪刀差与多维预警指标大屏](/images/content-rebuild/d3e8f7547d558720/fig-2.png)

### 1. 分布外验证集损失（OOD Validation Loss）
在准备数据时，单独切分出 10%~20% 来源不同、表述风格多样或由真实用户自由提问构成的分布外数据作为 OOD 验证集。如果训练集 Loss 和同分布 ID Loss 持续下降，而 OOD Loss 在某个 step 开始反弹上升，这就是模型开始死记句式和浅层特征的明确信号。

### 2. Token 预测条件熵（Predictive Entropy）
用于监测模型的预测分布是否过早塌缩。在固定的一组开放提示词输入下，计算模型在每个生成位置输出概率分布的香农熵：

$$H(P) = - \sum_{v \in V} P(v) \log_2 P(v)$$

如果微调过程中开放提示词的平均预测熵发生断崖式下跌，且生成的 Distinct-2（不重复二元词组比例）同步萎缩，说明模型输出正在走向僵化。

### 3. 影子基准跟踪（Shadow Benchmarks）
在训练循环中（如每隔 100~200 个 step），挂载轻量级的评估子集（例如 50 道 GSM8K 数学题或少量代码补全题）执行零样本推理。一旦通用得分出现明显跌幅，即可及早发出告警。

---

## 诊断代码：预测熵与多样性统计

以下 Python 脚本展示了如何基于模型输出的原始对数几率（logits）计算平均预测条件熵，并统计生成文本的 Distinct-N 比例：

![评估监控代码的数据流向与阈值判定机制](/images/content-rebuild/d3e8f7547d558720/fig-3.png)

```python
import math
from typing import List
import torch
import torch.nn.functional as F

def compute_predictive_entropy(logits: torch.Tensor) -> float:
    """
    计算给定 logits 下的平均预测条件熵（以比特为单位）。
    logits 维度: [batch_size, seq_len, vocab_size]
    """
    # 计算每个 token 上的概率分布 P
    probs = F.softmax(logits, dim=-1)
    log2_probs = F.log_softmax(logits, dim=-1) / math.log(2)
    
    # 熵 H = - sum(P * log2(P))
    entropy_per_token = -torch.sum(probs * log2_probs, dim=-1)
    return entropy_per_token.mean().item()

def compute_distinct_n(token_ids_list: List[List[int]], n: int = 2) -> float:
    """
    计算生成的序列中不重复 N-gram 的比例。
    token_ids_list: 多个样本生成的 token id 列表
    """
    total_ngrams = 0
    unique_ngrams = set()
    
    for tokens in token_ids_list:
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            unique_ngrams.add(ngram)
            total_ngrams += 1
            
    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams

# 运行示意：对比发散预测与单一 token 塌缩分布
vocab_size = 32000
seq_len = 16
batch_size = 2

# 场景 1：相对健康的概率分布
healthy_logits = torch.randn(batch_size, seq_len, vocab_size) * 2.0
entropy_healthy = compute_predictive_entropy(healthy_logits)

# 场景 2：发生模板坍塌（某个 token 概率极高）
collapsed_logits = torch.randn(batch_size, seq_len, vocab_size)
collapsed_logits[:, :, 101] += 30.0  # 人为放大单一 token 的输出
entropy_collapsed = compute_predictive_entropy(collapsed_logits)

print(f"健康分布下的预测熵: {entropy_healthy:.3f} bits")
print(f"坍塌分布下的预测熵: {entropy_collapsed:.3f} bits")
```

在微调过程中，多指标联合观测呈现出的典型轨迹如下表所示（数值为教学演示示意）：

| 训练 Step | 领域训练 Loss | 领域同分布验证 Loss | 分布外验证 Loss | 开放 Prompt 预测熵 | Distinct-2 多样性 | 通用推理基准得分 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 (微调前) | 2.85 | 2.89 | 2.95 | 8.42 bits | 0.88 | 100% |
| 200 | 1.12 | 1.18 | 1.25 | 7.95 bits | 0.85 | 98.0% |
| 500 (适宜状态) | 0.65 | 0.72 | 0.81 | 7.30 bits | 0.81 | 96.5% |
| 800 | 0.38 | 0.58 | 1.15 | 3.12 bits | 0.42 | 82.0% |
| 1200 | 0.15 | 0.79 | 2.30 | 1.05 bits | 0.18 | 51.0% |

由表可见，500 step 之后虽然领域训练 Loss 仍在下降，但分布外损失开始反弹，预测熵与词组多样性快速下降，通用能力受损。如果仅盯着训练 Loss，就会误把已严重过拟合的模型上线。

---

## 缓解手段：数据回放共训

从数据源头平衡梯度更新，是缓解灾难性遗忘的有效手段。最常用的工程做法是经验回放（Replay Co-training）。

![数据混合配比与双向梯度约束机制](/images/content-rebuild/d3e8f7547d558720/fig-4.png)

### 经验回放配置方法

在构建垂直领域微调数据时，按 5%~15% 的比例混合通识指令数据（如清洗过的通用多轮对话、基础代码题或逻辑题）：

$$\mathcal{L}_{\text{joint}}(\theta) = \alpha \mathcal{L}_{\text{domain}}(\theta) + (1 - \alpha) \mathcal{L}_{\text{general}}(\theta)$$

通识样本在反向传播时会提供针对通用能力的梯度分量。即使垂直任务产生较强偏移，通识梯度也会在一定程度上制约参数过度偏离原有的能力空间。关于微调样本的具体构造规范，可参考 [微调数据构造方法论](/llm-100/finetuning/finetuning-data-construction.md)。

---

## 缓解手段：LoRA 约束与复合早停

除了数据层面的调整，在模型参数更新方式和训练停止时机上也可以加入防护措施。

![LoRA参数更新约束与复合早停判定状态机](/images/content-rebuild/d3e8f7547d558720/fig-5.png)

### 1. 利用 LoRA 限制更新自由度

全参数微调会修改模型的所有权重，在垂直样本量较少时极易发生剧烈参数漂移。采用 [LoRA 高效微调](/llm-100/finetuning/lora-efficient-finetuning.md) 可以在结构上降低这种风险：

$$\Delta W = \frac{\alpha}{r} B A$$

* **主权重完全冻结**：基座模型的权重 $W_0$ 保持不变，基础语言理解和预训练知识得以保全在主干中。
* **低秩维度限制**：低秩旁路（例如 $r=8$ 或 $r=16$）将参数更新限制在更低维的子空间内，降低了模型将微调样本全部死记下来的自由度。
* **模块适配选型**：若任务更看重逻辑和推理能力的保持，可优先仅对注意力层（`q_proj`, `v_proj`）挂载 LoRA；若需吸收较多新领域术语，再考虑开放 MLP 层，同时适度增加回放数据比例。

### 2. 复合早停工程准则

训练不能只设置固定的轮数，也不建议只监测同分布验证 Loss。可结合分布外验证表现与通用任务底线设置综合评分：

$$S_{\text{val}} = \mathcal{L}_{\text{OOD-val}} + \lambda \cdot \max\bigl(0, \text{Baseline}_{\text{general}} - \text{Score}_{\text{general}}\bigr)$$

当 $S_{\text{val}}$ 连续 2~3 个检查点评估未见改善或出现持续恶化时，及时触发早停，并回滚到历史综合表现最佳的检查点权重。

---

## 适用边界与权衡思考

不同业务目标对过拟合与风格固化的容忍度不同，不应一概而论：

![业务目标与微调约束权衡象限图](/images/content-rebuild/d3e8f7547d558720/fig-6.png)

| 应用类型 | 核心要求 | 对风格固化的容忍度 | 推荐策略组合 |
| :--- | :--- | :--- | :--- |
| **结构化输出 / API 调用** | 严格遵循 JSON Schema 或 DSL 语法，不能自由发挥 | **较高**：允许格式高度收敛 | 纯领域数据微调，重点约束语法合规，无需引入过多通用回放 |
| **垂直专业助手 (如报销/医疗)** | 准确使用专业术语，且保持正常的常识与问答逻辑 | **较低**：防止模板坍塌与逻辑遗忘 | LoRA 微调 + 5%~10% 通用数据回放 + 监控 OOD 验证 Loss |
| **开放创意 / 通用助手** | 强调思考发散度、多角度分析与语言表达丰富度 | **极低**：严格防止输出单调退化 | 较大秩 LoRA 或全参数微调 + 充足多样化通用指令 + 监控预测熵与多样性 |

微调过程的监控保障了训练方向不发生严重跑偏。训练完成后，还必须在独立的测试集上进行端到端的质量检验，相关测试设计可进一步参考 [微调模型评估与验收](/llm-100/finetuning/evaluate-finetuned-model.md)。

---

## 练习与思考

1. **代码实操**：运行上文的 `compute_predictive_entropy` 与 `compute_distinct_n` 脚本。假设将序列长度 `seq_len` 提升至 100，生成一段完全由同一个 token 重复构成的序列（如 `[42] * 100`），计算其 Distinct-2 的得分，并解释为什么该指标可以反映生成退化。
2. **情境分析**：某团队在微调一个法律合同审查助手时，发现微调后模型在合同术语上的判断准确率达标，但面对用户的日常多轮追问时，每次回复都会在开头复读相同的法条声明。请根据本文内容，给出两项具体的改进措施。
