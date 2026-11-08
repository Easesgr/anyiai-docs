---
title: 预训练、SFT 与偏好对齐分别解决什么问题？
icon: mdi:arrow-decision-outline
tag:
- 大模型面试
- 训练流程
- SFT
description: 拆解大模型三阶段训练链路的目标、损失掩码实现、数据差异与能力边界，解答面试高频追问与选型误区。
---

在大模型技术面试中，“大模型的三阶段训练链路分别解决什么问题”是一道必考题。面试官考察的核心不在于背诵名词，而在于候选人能否讲清楚每个阶段的优化目标、数据形式、损失计算范围，以及它们各自的边界。

![大模型三阶段训练链路全景流程](/images/content-rebuild/61c0cacfe2ace0a1/cover.png)

---

## 30秒口述结论

可以用一个具体的运行例子来串联三者的分工：如果向模型输入提示词 `请用 Python 写一个快速排序算法，并说明复杂度。`

1. **预训练（Pre-training）解决“知识与语言规律压缩”问题**：模型在万亿级离散 Token 构成的文本上做自回归因果预测，把语法、代码结构、通用常识压缩进参数。面对上述输入，基座模型具备写出 Python 代码的概率分布，但它本质是缺乏助手角色意识的通用文本续写器，很可能把输入当成考卷的一部分，接着续写出 `请用 C++ 写一个归并排序算法`。
2. **监督微调（SFT）解决“指令遵循与交互协议适配”问题**：使用数万至数十万条高质量问答对，通过在输入提示词（Prompt）部分屏蔽损失、仅对回答（Response）计算交叉熵的方式，训练模型理解“用户提问时，自己应当作为助手给出解答”。此时输入上述问题，模型会停下无端续写，直接输出快速排序代码与复杂度分析。
3. **偏好对齐（Alignment）解决“输出行为边界与价值取向”问题**：利用对比数据（胜出回答 vs 落败回答），通过 RLHF 或 DPO 等算法调整策略分布。模型学会选择更安全、更准确、更有帮助的回答（例如给出边界检查、说明递归栈溢出风险；面对恶意诱导时则拒绝作恶）。

简要概括：**预训练建立基础表征与通用知识，SFT 建立交互格式与任务遵循，偏好对齐限定安全边界与回答质量偏好**。

---

## 预训练：全序列因果建模与参数初始化

大模型的大部分语言能力与常识都建立在预训练阶段。这一阶段使用海量未标注文本（如网页、开源代码、学术论文、数字化图书），无需人工编写问答对。

![预训练阶段自回归语言建模机制与全序列交叉熵计算流程](/images/content-rebuild/61c0cacfe2ace0a1/fig-1.png)

### 1. 优化目标与损失计算

因果语言模型（Causal LM）预训练采用标准自回归任务：给定序列 $X = (x_1, x_2, \dots, x_T)$，模型在每个位置 $t$ 根据前文 $x_{<t}$ 预测当前 Token $x_t$。详细推导见 [/interview/llm-interview/pretraining-objective.md](/interview/llm-interview/pretraining-objective.md)。

其损失函数是全序列上的负对数似然损失：

$$\mathcal{L}_{\text{PT}}(\theta) = - \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

在代码工程实现中，将输入序列错开一个位置（Shift by one）作为标签，全序列的所有位置都会产生预测误差并参与反向传播。

### 2. 算力消耗与输出形态

- **算力开销集中**：预训练通常消耗模型全生命周期 95% 以上的计算量（FLOPs），需要在数千至数万张加速卡构成的集群上训练数周至数月。
- **基座模型（Base Model）的局限**：模型输出的是统计学上的自然语言续写概率。它不具备“助手角色意识”，无法可靠区分“这是用户给我的指令”还是“这是一篇待续写的未完稿件”。

<ReadMoreLock />

---

## 监督微调（SFT）：回答区掩码与格式激活

基座模型具备丰富的表征，但需要学习现代对话界面的行为范式。监督微调（Supervised Fine-Tuning）负责将文本续写能力转换为指令问答能力，其机制见 [/fundamentals/training-and-finetuning/sft.md](/fundamentals/training-and-finetuning/sft.md)。

![SFT与预训练的损失掩码机制对比](/images/content-rebuild/61c0cacfe2ace0a1/fig-2.png)

### 1. 核心技术实现：Prompt 损失掩码（Loss Masking）

SFT 的一条训练样本由 `Prompt`（用户输入）和 `Response`（助手回答）拼接而成。如果模型对 `Prompt` 部分也计算损失并更新梯度，模型会把参数容量消耗在记忆“用户提问常用的口吻”上，甚至影响对复杂任务的泛化能力。

因此，SFT 的标准做法是对 `Prompt` 位置的标签设置掩码，通常将标签值填为 `-100`。在 PyTorch 中，`CrossEntropyLoss` 默认忽略索引为 `-100` 的目标：

```python
import torch
import torch.nn as nn

# 假设 batch_size=1，序列长度为 5 个预测步，词表大小为 10
# 输入序列共 6 个 token: [User, :, Hi, Assistant, Hello, !]
# 其中 Prompt 为前 4 个 token，Response 为后 2 个 token
logits = torch.randn(1, 5, 10, requires_grad=True)

# 对应错位后的目标 token（Shifted targets）
# 前 3 个预测步的目标落在 Prompt 内部（预测 :, Hi, Assistant），打上 -100 掩码
# 后 2 个预测步的目标落在 Response 区域内（预测 Hello, !），保留真实 token id
labels = torch.tensor([[-100, -100, -100, 4, 5]])

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
loss = loss_fn(logits.view(-1, 10), labels.view(-1))
loss.backward()

# 此时只有 Response 对应的输出 logits 产生了反向传播梯度
assert logits.grad[:, :3, :].abs().sum() == 0
assert logits.grad[:, 3:, :].abs().sum() > 0
```

该实现可参考 Hugging Face 的 [Causal Language Modeling 官方文档](https://huggingface.co/docs/transformers/tasks/language_modeling)。

### 2. 表面对齐假说（LIMA）与数据质量

Meta 在 LIMA 论文中提出了表面对齐假说（Surface Alignment Hypothesis）：
- 模型几乎所有的推理能力与世界知识都在预训练阶段沉淀在参数中。
- 微调的主要作用是唤醒并规范模型与用户交互时的风格、排版与语调。
- **工程含义**：在常规问答和交互场景下，数百到数千条经过严格质检、覆盖多样领域的高质量微调样本，效果往往胜过数十万条格式嘈杂、充满机器模板句的低质数据。数据清洗与构建细节见 [/fundamentals/training-and-finetuning/finetuning-data.md](/fundamentals/training-and-finetuning/finetuning-data.md)。

---

## 偏好对齐：对比反馈与行为边界

SFT 训练的模型本质上是高质量示范样本的最大似然拟合器，这会留下两类典型问题：
1. **谄媚与盲从（Sycophancy）**：当用户在提问中包含明显的逻辑谬误时，SFT 模型往往顺着用户的错误假设继续推理，而不是指出错误。
2. **安全越狱脆弱性**：当面临带有诱导性的危险请求（如制造攻击脚本、伪造身份证明）时，简单的模仿训练无法让模型在“服从指令”与“守住安全红线”之间形成稳定的权衡。

![偏好对齐中对比样本对模型行为分布的塑造机制](/images/content-rebuild/61c0cacfe2ace0a1/fig-3.png)

### 1. 对齐数据与算法路径

偏好对齐不再使用单条示范 `<Prompt, Response>`，而是使用针对同一个提示词 $x$ 的成对对比样本 $(x, y_w, y_l)$，其中 $y_w$ 是优选回答（Winner），$y_l$ 是落败回答（Loser）。

主要技术路径包括：
- **RLHF（如基于 PPO）**：先利用对比样本训练一个打分奖励模型（Reward Model），再通过 PPO 强化学习算法让策略模型在线生成回答，以奖励分数最大化为目标，同时引入相对于初始模型的 KL 散度惩罚防止策略漂移。推导见 [/interview/llm-interview/rlhf-pipeline.md](/interview/llm-interview/rlhf-pipeline.md)。
- **直接偏好优化（DPO）**：基于 Bradley-Terry 偏好模型，通过代换推导出策略模型输出概率比值与隐式奖励的关系，省去独立奖励模型训练与在线强化学习采样，直接在离线数据上计算交叉熵损失。对比见 [/interview/llm-interview/rlhf-dpo.md](/interview/llm-interview/rlhf-dpo.md)。
- **单阶段对齐（如 ORPO）**：在 SFT 损失基础上附加胜负几率惩罚比率，尝试在单次微调中兼顾格式遵循与偏好排序。见 [/interview/llm-interview/orpo-alignment.md](/interview/llm-interview/orpo-alignment.md)。

---

## 边界辨析：为什么不能用 SFT 来更新企业业务知识？

很多开发者在面对“让大模型掌握企业最新制度规范”时，会直觉性地选择收集新制度问答跑 SFT。这是典型的方案选型错位。

![SFT强行注入事实知识导致的参数遗忘与幻觉发散](/images/content-rebuild/61c0cacfe2ace0a1/fig-4.png)

### 1. 神经网络参数不是关系型数据库

大模型权重由连续的高维矩阵构成，存储的是分布式参数表征，不具备精准键值寻址的特性：
1. **灾难性遗忘（Catastrophic Forgetting）**：强行通过过拟合微调让模型强记数百条专有事实，通常会干扰相邻权重神经元的表征分布，导致模型的基础逻辑、常识或编码能力下降。详见 [/fundamentals/training-and-finetuning/finetuning-pitfalls.md](/fundamentals/training-and-finetuning/finetuning-pitfalls.md)。
2. **幻觉与虚假确信**：参数化记忆缺少明确的上下文引用机制。当外部细则调整时，模型容易把旧记忆与新提示词拼接成看似言之凿凿的虚假陈述。
3. **维护成本与时延**：规章、政策和动态库存若频繁变动，反复执行模型微调并在发布前完成全量基准回归，在工程和算力上极不经济。

### 2. 知识落地的正确选型路径

- **实时动态事实、规章制度、要求可查源头**：使用 **RAG（检索增强生成）**。把数据保存在外置向量库或检索系统中，通过检索将原文作为上下文注入 Prompt，模型只负责基于给定上下文进行总结归纳。
- **结构化输出要求、专有交互流程、固定业务格式**：使用 **SFT**。例如要求模型必须按照严格的 JSON 模式返回结构化字段。
- **特定垂类巨量语料（全量行业代码、海量专业医学论文）**：使用 **增量预训练（Continual Pre-training）**，在已有底座上继续进行自回归无监督学习以建立领域词表与上下文表征。
完整选型对比见 [/fundamentals/training-and-finetuning/prompt-rag-finetuning.md](/fundamentals/training-and-finetuning/prompt-rag-finetuning.md)。

---

## 面试高频追问与极限推演

### 追问 1：如果跳过 SFT，直接拿基座模型做 RLHF 会怎样？

这是考察强化学习探索空间与奖励信号稳定性的经典追问。

**分析与回答**：
在工程实践中极难收敛，通常会导致训练失败，主要原因有三点：

![跳过SFT直接进行强化学习导致探索发散与崩溃](/images/content-rebuild/61c0cacfe2ace0a1/fig-5.png)

1. **探索动作空间过大，有效奖励稀疏**：基座模型采样生成的候选序列是无约束续写（可能继续写出新题目、生成乱码、直接输出 HTML 源码或跳入其他语言）。在极度广袤的离散序列空间中，随机采样命中“符合人类指令意图回答”的概率接近于零。
2. **奖励模型（Reward Model）面临严重分布外（OOD）失效**：奖励模型是在“正常问答回答对”上训练出来的。当基座模型采样出大量格式紊乱的续写文本时，奖励模型在未覆盖的数据分布上极易给出荒谬的打分，错误的高分信号会导致策略崩溃。
3. **KL 散度约束机制失衡**：RLHF 依赖参考模型（Reference Model）计算 KL 散度以防止策略剧烈漂移。如果起始策略本身就处于发散状态，策略在更新早期就会陷入局部模式坍塌。

因此，SFT 是强化学习对齐所必需的**冷启动（Warm Start）前置步骤**，其核心价值是把模型的采样空间强行收束进“指令遵循”的有效子集内。

### 追问 2：SFT 训练时如果不对 Prompt 做掩码，全序列计算损失会有什么后果？

**分析与回答**：
1. **梯度更新目标偏离**：模型会分摊大量梯度去拟合用户的提问句式（如“帮我写一下”、“请问怎么理解”），而这些句式往往高度多变且缺乏逻辑深度。
2. **推理阶段容易答非所问**：当模型遇到未见过的提问句式时，可能会将该句式视作模仿对象，在生成回答中重新输出类似的提问句式，降低对话准确性。
3. **破坏条件概率分布建模**：SFT 的优化目标是建模条件生成概率 $P(\text{Response} \mid \text{Prompt})$。若对 Prompt 计算自回归损失，会强迫模型分配容量去拟合用户的提问句式与前缀，导致生成时交互角色边界模糊，容易诱发自问自答或循环复读提示词。

### 追问 3：在技术架构设计中，如何评估三阶段的投入产出？

![三阶段工业落地与选型决策树](/images/content-rebuild/61c0cacfe2ace0a1/fig-6.png)

在面试总结阶段，可以通过下表收敛三者的技术特征：

| 维度 | 预训练（Pre-training） | 监督微调（SFT） | 偏好对齐（Alignment） |
| :--- | :--- | :--- | :--- |
| **输入数据形态** | 海量无标注纯文本（万亿级 Tokens） | 成对示范数据 `<Prompt, Response>` | 成对偏好三元组 `(Prompt, Winner, Loser)` |
| **损失掩码机制** | 全序列自回归错位预测 | Prompt 设置掩码（-100），仅对 Response 算损失 | 基于对比样本的隐式/显式奖励差值优化 |
| **相对计算开销** | 极大（占总算力 95% 以上） | 较小（通常 < 5%） | 极小（通常 < 1%） |
| **核心解决问题** | 语言底层表征、通用知识沉淀 | 指令遵循、对话交互协议适配 | 行为边界锁定、安全拒答、幻觉抑制 |
| **不可替代性** | 提供下游任务所需的泛化基础 | 将无序续写收缩至有效问答子空间 | 解决单向模仿示范无法学到的多目标权衡 |
