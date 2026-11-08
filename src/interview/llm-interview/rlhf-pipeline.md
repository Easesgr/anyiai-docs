---
title: RLHF 训练流程：SFT、奖励模型（RM）与 PPO 的协作机制
icon: mdi:school
tag:
- RLHF
- SFT
- PPO
- Reward Model
description: 详解大模型 RLHF 三阶段流程：SFT 基线、成对偏好训练 RM、以及 PPO 四网络协同机制与显存调度。
---

## 30 秒口述结论

在面试中被问到“RLHF 的完整训练流程”，可以用以下结构分三步口述：

1. **指令微调（SFT）**：在高质量问答对上做有监督因果语言建模，让预训练基座掌握“按提示词回答问题”的基础格式与指令遵循能力，把自由续写收敛到问答任务的搜索空间内。
2. **奖励模型训练（Reward Model, RM）**：给定同一个提示词和多个候选回答，由标注员标注胜负对；利用 Bradley-Terry 偏好模型计算成对排序损失，训练一个输入完整文本、输出标量分数的打分器，用于量化人类主观偏好。
3. **强化学习对齐（PPO）**：由 **Actor（策略网络）** 生成回答，**RM（奖励模型）** 给出终局分数，**Reference Model（参考模型）** 计算单步 KL 散度防止模型偏离初始语言先验（防范 Reward Hacking），**Critic（价值网络）** 评估状态价值并配合 GAE 计算优势函数，最后通过 PPO-Clip 目标稳定更新 Actor 与 Critic。

![RLHF 三阶段全景与模型状态流转](/images/content-rebuild/a04f8f4d7e5f11f5/cover.png)

---

## 贯穿示例：模型是如何被调教的

为了直观理解这三个阶段的分工，我们以一个具体的指令为例：

* **输入提示词 $x$**：“请用通俗的语言解释什么是光合作用。”
* **预训练基座状态**：输入该指令后，基座模型通常会继续续写：“是初中生物的重要知识点，以下是相关的考试真题……”。它并不知道人类需要它“扮演回答者”。
* **阶段一（SFT）处理**：给模型学习人工编写的标准回复 $y$：“光合作用就像植物的厨房，植物利用阳光作为能源，把吸收的水和二氧化碳做成养分，并释放出氧气。”模型由此学会了直接回答问题。
* **阶段二（RM）处理**：针对该问题，模型生成两个回答：$y_w$（解释通俗易懂）和 $y_l$（仅回复“这是生物化学反应，去翻教材吧”）。标注员标出 $y_w \succ y_l$，RM 学会给 $y_w$ 打高分、给 $y_l$ 打低分。
* **阶段三（PPO）处理**：Actor 尝试生成新回答。如果 Actor 试图通过投机取巧（例如在结尾不断堆砌无意义的客套话“祝您生活愉快，非常感谢您的提问！”）来骗取 RM 高分，Reference 模型产生的 KL 散度惩罚就会拉低综合奖励，迫使 Actor 在保持语言自然的前提下最大化回答质量。

---

## 第一阶段：SFT 建立指令遵循基线

预训练大模型输出的是下一个 Token 的条件概率分布。直接在预训练基座上执行强化学习几乎不可行：因为策略网络在庞大的词表序列空间中随机探索，生成符合人类对话逻辑回答的概率极低，会导致强化学习在前期长期采样不到正向奖励。

在整体流程中（可参考 [预训练、SFT、偏好对齐全解析](/interview/llm-interview/training-pipeline.md)），SFT 的职责是**缩小策略网络的搜索空间并建立对话交互基准**：

* **输入与损失**：输入序列为提示词 $x$ 与标准回答 $y$，损失函数仅在回答文本 $y$ 的 Token 位置计算标准交叉熵损失：

$$\mathcal{L}_{\text{SFT}}(\theta) = - \sum_{t=1}^{|y|} \log P_\theta(y_t \mid x, y_{<t})$$

* **阶段局限**：SFT 属于行为克隆（Behavioral Cloning）。它只能拟合标注员写出的单一 Token 序列分布，模型无法知道不同回答之间的质量梯度，也无法区分“标点符号微调”与“产生事实幻觉”在惩罚力度上的重要性差异。

![三阶段数据产物与模型继承拓扑](/images/content-rebuild/a04f8f4d7e5f11f5/fig-1.png)

---

## 第二阶段：奖励模型（RM）与偏好建模

为了在强化学习阶段替代昂贵、低速的人工在线打分，需要先训练一个可微分的打分器——奖励模型（Reward Model）。

### 1. 为什么采用成对比较（Pairwise）而不是绝对打分？

在实际标注中，通常不要求标注员对单条回答给出 1 到 10 分的绝对标量，而是展示同一个提示词 $x$ 对应的两个生成回答 $(y_w, y_l)$，标注员仅需选择 $y_w$ 优于 $y_l$（记作 $y_w \succ y_l$）。原因在于：

1. **标尺主观性与偏移**：不同标注员对“7 分”与“8 分”的标准不同；同一标注员在疲劳或不同日期下，打分标准也会漂移。
2. **相对比较一致性高**：成对盲测不需要校准绝对分值，不同标注员对于“回答 A 是否比回答 B 表达更清晰”的判断一致性远高于绝对数值。

### 2. Bradley-Terry 数学建模与损失函数

奖励模型 $r_\psi(x, y)$ 的结构通常是将 SFT 模型的词表分类头（LM Head）替换为一个输出维度为 1 的线性层（Value Head），输入为完整的上下文 $(x, y)$，输出为一个标量得分。

基于经典的 **Bradley-Terry (BT) 偏好模型**，人类认为回答 $y_w$ 优于 $y_l$ 的概率，建模为两个回答标量分数的差值经过 Sigmoid 函数映射：

$$P(y_w \succ y_l \mid x) = \sigma\big(r_\psi(x, y_w) - r_\psi(x, y_l)\big) = \frac{1}{1 + e^{-\left(r_\psi(x, y_w) - r_\psi(x, y_l)\right)}}$$

针对偏好数据集 $\mathcal{D} = \{(x, y_w, y_l)\}$，奖励模型的训练目标是最小化负对数似然：

$$\mathcal{L}_{\text{RM}}(\psi) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\big(r_\psi(x, y_w) - r_\psi(x, y_l)\big) \right]$$

![成对偏好与 Bradley-Terry 奖励建模](/images/content-rebuild/a04f8f4d7e5f11f5/fig-2.png)

<ReadMoreLock />

### 3. 损失函数梯度行为分析

令分差 $\Delta r = r_\psi(x, y_w) - r_\psi(x, y_l)$，损失函数对两项打分的偏导数为：

$$\frac{\partial \mathcal{L}_{\text{RM}}}{\partial r_\psi(x, y_w)} = - \big(1 - \sigma(\Delta r)\big) = - \sigma(-\Delta r)$$

$$\frac{\partial \mathcal{L}_{\text{RM}}}{\partial r_\psi(x, y_l)} = \sigma(-\Delta r)$$

从导数变化可以看出其自动调节机制：

* **当模型已经预测正确且分差很大时**（$r_\psi(x, y_w) \gg r_\psi(x, y_l)$），$\Delta r \to +\infty$，梯度 $\sigma(-\Delta r) \to 0$。模型不会对已经拉开分差的样本继续过度推大数值，能有效抑制过拟合。
* **当模型打分颠倒时**（$r_\psi(x, y_w) < r_\psi(x, y_l)$），$\Delta r < 0$，梯度模长接近 1。这会施加明显的梯度推力：同步提高胜出样本得分、压低失败样本得分。

---

## 第三阶段：PPO 阶段的四网络协同运作

进入 PPO（近端策略优化）阶段，系统运行时通常需要调度四个网络实例。理解各网络的状态与交互是理清该阶段的关键：

| 模型角色 | 初始权重来源 | 训练期状态 | 职责与计算内容 |
| :--- | :--- | :--- | :--- |
| **Actor（策略网络 $\pi_\theta$）** | SFT 模型权重 | **参数更新** | 接收 Prompt $x$，自回归采样生成 Token 序列 $y$，通过梯度更新最大化最终期望回报 |
| **Reference Model（参考网络 $\pi_{\text{ref}}$）** | SFT 模型快照 | **完全冻结** | 与 Actor 输入相同前缀，计算参考概率分布，提供 KL 散度惩罚锚点 |
| **Reward Model（奖励网络 $r_\psi$）** | 阶段二训练完成的 RM | **完全冻结** | 输入生成完的完整序列 $(x, y)$，输出环境奖励标量分 $R_{\text{env}}$ |
| **Critic（价值网络 $V_\phi$）** | RM 权重或 SFT 权重初始化 | **参数更新** | 预测每个 Token 状态的前向累计收益期望，配合 GAE 计算优势函数 $\hat{A}_t$ |

![PPO 强化学习四网络数据流与交互闭环](/images/content-rebuild/a04f8f4d7e5f11f5/fig-3.png)

### 单轮训练的执行步骤

1. **采样阶段（Rollout）**：从 Prompt 集中批量抽取 $x$，**Actor** 以自回归方式采样生成完整的回答 $y = (y_1, y_2, \dots, y_T)$，并保存生成过程中的动作对数概率 $\log \pi_\theta(y_t \mid x, y_{<t})$。
2. **打分与奖励分配阶段（Evaluation）**：
   * 序列完整生成后送入 **Reward Model**，获得终局标量奖励 $R_{\text{env}} = r_\psi(x, y)$。
   * 完整序列同时送入 **Reference Model**，获得基准对数概率 $\log \pi_{\text{ref}}(y_t \mid x, y_{<t})$。
   * 计算每一步的 Token 级复合奖励 $R_t$（在结尾处叠加 $R_{\text{env}}$ 与各位置的即时 KL 散度惩罚）。
3. **价值与优势估计阶段（Estimation）**：
   * **Critic** 计算每个时间步的当前状态价值 $V_\phi(s_t)$。
   * 使用广义优势估计（GAE, Generalized Advantage Estimation）聚合多步即时奖励与 Critic 预估值，输出每个 Token 位置的优势值 $\hat{A}_t$。
4. **参数更新阶段（Optimization）**：
   * 利用 $\hat{A}_t$ 和裁剪重要性采样率计算 PPO-Clip 损失，反向传播更新 **Actor**。
   * 通过均方误差损失（MSE）拉近 $V_\phi(s_t)$ 与回报目标值（TD Target），反向传播更新 **Critic**。

---

## KL 散度惩罚与 Reward Hacking

### 1. 什么是 Reward Hacking？

古德哈特定律指出：“当一个指标变成目标时，它就不再是一个好指标。”

奖励模型只是基于有限偏好数据拟合出的近似判别器，在偏离训练分布的样本上存在盲区与评分漏洞。若 Actor 在训练中缺乏约束，它会迅速找到能够触发 RM 虚高分数的特定模式（例如：频繁堆叠“权威”“专家”等诱导词、无意义地拉长回答、谄媚提问者等），但实际文本可读性与逻辑性严重退化。这种现象称为 **Reward Hacking（奖励作祟 / 策略作弊）**。

![KL 散度约束对抗 Reward Hacking 的有效边界](/images/content-rebuild/a04f8f4d7e5f11f5/fig-4.png)

### 2. Token 级复合奖励设计

为了将 Actor 约束在合理的初始语言分布内，PPO 在强化学习奖励中引入了 Actor 与 Reference Model 之间的 KL 散度惩罚。

在生成序列的每个 Token 位置 $t$，其实际奖励定义为：

$$R_t = - \beta \cdot \left( \log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{\text{ref}}(y_t \mid x, y_{<t}) \right) + \mathbb{I}(t = T) \cdot r_\psi(x, y)$$

其中：
* $\beta$ 为 KL 惩罚系数，用于调节多样性探索与基准分布约束的平衡。
* $\mathbb{I}(t = T)$ 为示性函数，表示仅在序列最后一个 Token（终止符）处，叠加 RM 给出的整句环境奖励。
* 若 Actor 在某一位置大幅修改概率使得 $\pi_\theta \gg \pi_{\text{ref}}$，对数比值为正，$-\beta(\cdot)$ 会产生惩罚扣分；这确保模型不会为了迎合 RM 的单一偏好而破坏通顺度和语法。

---

## PPO 目标函数与更新边界

大模型训练成本高昂，无法使用采样一次即丢弃样本的基础策略梯度（REINFORCE）。PPO 引入重要性采样（Importance Sampling），使得 Actor 可以基于同一批采样的文本数据进行多次小步梯度更新。

![PPO-Clip 目标函数的截断梯度调节机制](/images/content-rebuild/a04f8f4d7e5f11f5/fig-5.png)

### 1. 重要性采样比率

定义当前策略与采样该样本时的旧策略之间的概率比率：

$$r_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t \mid x, y_{<t})}$$

在生成完刚开始训练的第 1 步时，$r_t(\theta) = 1$。

### 2. PPO-Clip 截断目标

为了避免策略网络单次更新步长过大导致语言模型崩溃，PPO 将概率比率裁剪在 $[1-\epsilon, 1+\epsilon]$（工程通常设 $\epsilon = 0.2$）之间：

$$\mathcal{L}_{\text{CLIP}}(\theta) = - \hat{\mathbb{E}}_t \left[ \min\Big( r_t(\theta) \hat{A}_t, \; \text{clip}\big(r_t(\theta), 1-\epsilon, 1+\epsilon\big) \hat{A}_t \Big) \right]$$

截断对梯度的调节机制如下：

* **正优势（$\hat{A}_t > 0$）**：说明该 Token 的动作表现优于基准，优化方向希望提高 $r_t(\theta)$。当 $r_t(\theta) > 1 + \epsilon$ 时，截断项生效，$\min$ 函数取值为固定常数 $(1+\epsilon)\hat{A}_t$，对参数 $\theta$ 的梯度变为 0。这防止模型因单次好运采样而过度更新。
* **负优势（$\hat{A}_t < 0$）**：说明该 Token 动作劣于基准，优化方向希望压低 $r_t(\theta)$。当 $r_t(\theta) < 1 - \epsilon$ 时，截断项生效，梯度同样被截断为 0，防止参数单步调整剧烈导致模型破坏。

Critic 网络则通过均方误差损失迫使其对状态价值的估计逼近实际折现回报：

$$\mathcal{L}_{\text{VF}}(\phi) = \frac{1}{2} \hat{\mathbb{E}}_t \left[ \big( V_\phi(s_t) - V_t^{\text{target}} \big)^2 \right]$$

---

## 工程挑战：四网络显存与流水线调度

在数十亿乃至数百亿参数模型上跑完整 PPO，主要瓶颈在系统调度与显存分配：

![PPO 四网络分布式显存与调度拓扑](/images/content-rebuild/a04f8f4d7e5f11f5/fig-6.png)

1. **显存压力差异**：
   * **训练网络（Actor 和 Critic）**：不仅需要保存模型参数，还需要保存前向激活值、反向传播梯度，以及 AdamW 优化器状态（一阶动量和二阶动量通常占用权重 2 倍以上的显存）。
   * **推理网络（Reference 和 RM）**：参数完全冻结，仅需前向推理，显存占用相对固定。
2. **算力形态切换瓶颈**：
   * **Rollout 生成阶段**：属于自回归串行解码，受限于显存带宽（Memory-bound）。
   * **Training 更新阶段**：多步前向与反向传播属于高度并行的矩阵乘法计算，受限于算力吞吐（Compute-bound）。
   * 频繁交替这两类任务会导致 GPU 计算集群在阶段切换时产生大面积空等（计算气泡）。
3. **工程优化方案**：
   * **权重共享与 LoRA**：Actor 与 Reference 共用底层冻结权重，仅在 Actor 上挂载 LoRA 适配器训练；
   * **混合部署与错峰调度**：将 RM 与 Critic 安排在同一组物理 GPU 节点上错峰推理与训练，或通过参数卸载（Offload）减少同时驻留显存的实例数量。
   * 这种显存与调度的工程复杂度，正是后续社区青睐不需要训练 RM 与 Critic 的离线算法（如 [DPO 原理与对比](/fundamentals/training-and-finetuning/rlhf-dpo.md) 以及 [面试 DPO 解析](/interview/llm-interview/rlhf-dpo.md)）的核心原因。

---

## 面试追问与高频辨析

### 追问 1：训练 PPO 时，为什么不能省去 Reference 模型？

**答题要点**：
* **锚定语言基础能力**：RM 只是偏好分数的判别器，并不具备维持语法正确和语言通顺的约束。如果去掉 Reference 模型，Actor 在纯粹追求高分的过程中，很快会产生 Reward Hacking（生成无意义的高分词串）。
* **维持置信更新域**：Reference 模型保存了 SFT 阶段的先验知识，KL 散度充当了软约束，让策略只在可信的近邻空间内探索，避免策略漂移导致模型退化。

### 追问 2：为什么奖励模型不直接做成回归模型（打 1 到 5 分）？

**答题要点**：
* **消除主观尺度偏差**：不同标注员对绝对分数的尺度敏感度不同（严格者给 3 分的质量，宽松者可能给 5 分），直接用回归损失会导致数据方差过大、难以收敛。
* **成对排序具有尺度鲁棒性**：人类判断“回答 A 比回答 B 更好”的共识度远高于独立给分。通过 Bradley-Terry 建模，只优化两者的相对胜率差，自然规避了绝对量纲的标定误差。

### 常见认知错误

* **错误说法 1**：“PPO-Clip 超出截断区间后，损失函数变成了 0。”
  * **纠正**：超出截断区间后，损失函数的值是一个常数项（即截断边界值乘以优势函数），因而该项对模型参数的**梯度（导数）为 0**，并不是损失值本身归零。
* **错误说法 2**：“RLHF 就是 PPO 算法的代名词。”
  * **纠正**：RLHF 是包含“指令数据准备 -> 奖励模型训练 -> 强化学习对齐”的完整范式框架；PPO 仅是实现第三阶段强化学习更新的一种具体算法，强化学习阶段也可采用 GRPO、ReMax 等算法。此外，偏好对齐领域还发展出了跳过显式奖励模型与在线强化学习循环的直接偏好优化算法（如 DPO、KTO 等）。
* **错误说法 3**：“Actor 损失中的 KL 散度是两个模型输出文本向量的直接差值。”
  * **纠正**：KL 散度是基于 Token 级别条件概率分布对数差 $\log \pi_\theta(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t)$ 计算的相对熵，用于度量两个概率分布的距离，而非向量欧氏距离。
