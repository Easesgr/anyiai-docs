---
title: RLHF 和 DPO 有什么区别？大模型偏好对齐机制与工程权衡
icon: mdi:scale-balance
tag:
- 训练与微调
- RLHF
- DPO
- 偏好对齐
description: 解析 RLHF 与 DPO 在优化目标、奖励建模与工程调度上的异同，结合数学推导与最小算例，理清离线拟合与在线探索的适用边界。
---

![偏好对齐技术全景对比封面图](/images/content-rebuild/93e9006334df3302/cover.png)

在完成预训练与监督微调（SFT）之后，大语言模型已经能够理解指令并生成语法通顺的文本。但如果用户提问：“如何保证充足的睡眠？”，训练数据中可能同时存在两种回答：

- **回答 A（优选）**：“建立固定的作息习惯，睡前一小时减少使用荧光屏幕...”
- **回答 B（劣选）**：“白天拼命灌浓茶硬撑，晚上自然累瘫秒睡...”

单纯依赖监督微调的语言模型无法天然识别哪一种回答更符合人类的真实价值与安全期望。为了让模型在面对开放式问题时倾向于输出回答 A 而抑制回答 B，模型训练需要进入**偏好对齐（Preference Alignment）**阶段。

目前主流的对齐方案主要有两条路线：一条是基于强化学习的 **RLHF（基于人类反馈的强化学习，通常采用 PPO 算法）**；另一条是不需要显式训练奖励模型的 **DPO（Direct Preference Optimization，直接偏好优化）**。本文将从具体机制出发，解释为什么需要偏好对齐、两种方法背后的数学原理，以及它们在工程实践中的核心权衡。

---

## 1. 为什么只有 SFT 还不够？成对偏好的由来

监督微调（SFT）的核心优化目标是自回归交叉熵损失。对于训练集中的目标回答 $y = (y_1, y_2, \dots, y_T)$，模型通过最大化每一步的条件概率来更新参数：

$$\mathcal{L}_{\text{SFT}}(\theta) = - \sum_{t=1}^T \log P_\theta(y_t \mid x, y_{<t})$$

这种训练方式在处理偏好问题时存在三个实际局限：

1. **缺乏负反馈表达能力**：交叉熵损失的作用是“提高样本的生成概率”。如果把劣质回答 $y_l$ 放进训练集，哪怕标注了提示词“不要模仿”，模型前向计算时仍然在学习生成这些词的概率分布，无法教模型“什么绝对不能说”。
2. **多模态分布的均值平摊**：对于开放式问题，高质量的回答可能有成百上千种不同的行文风格。强行让单一参数模型在 token 级别去完全拟合单一人工标注，容易把概率质量分散，导致模型输出变得平庸或产生模棱两可的废话。
3. **暴露偏差（Exposure Bias）**：SFT 训练时采用 Teacher Forcing 模式（每一步预测时，模型输入的都是真实的黄金前文）。而在实际推理时，模型依赖自回归采样（上一步自己生成的 token 成为下一步的输入）。一旦早期 token 产生轻微偏离，误差就会逐步累积，导致整段输出脱轨。

![SFT 与偏好对齐的数据与优化目标差异](/images/content-rebuild/93e9006334df3302/fig-1.png)

为了解决这些问题，偏好对齐改变了数据组织方式。我们不再只给模型喂单条“标准答案”，而是针对同一个提示词 $x$，构造一对包含胜出项与落选项的样本 $(x, y_w, y_l)$。其中 $y_w$（Chosen）表示更优选的回答，$y_l$（Rejected）表示相对劣质的回答。对齐的目标，是拉大模型生成 $y_w$ 与 $y_l$ 之间的概率差距。

<ReadMoreLock />

---

## 2. 典型 RLHF-PPO 的三阶段机制

经典 RLHF 通常采用 OpenAI 在 InstructGPT 中确立的三阶段流程，将文本生成建模为一个马尔可夫决策过程：已生成的文本是当前状态，下一个预测的 token 是动作。

![典型 RLHF-PPO 三阶段流程拆解](/images/content-rebuild/93e9006334df3302/fig-2.png)

### 阶段一：收集成对偏好数据

人工评估一段文字的绝对分数（例如给一篇文章打 1 到 10 分）主观性强、方差极大。因此，实际标注通常采用二元相对比较：给定提示词 $x$，让模型生成两个候选响应 $y_1, y_2$，由标注员指出哪个更好（记作 $y_w \succ y_l$）。

根据 Bradley-Terry (BT) 偏好模型，两个输出之间的相对偏好概率可以建模为未知的潜在奖励之差的 Sigmoid 映射：

$$P(y_w \succ y_l \mid x) = \sigma\left(r^*(x, y_w) - r^*(x, y_l)\right) = \frac{1}{1 + e^{-(r^*(x, y_w) - r^*(x, y_l))}}$$

### 阶段二：训练独立的奖励模型（Reward Model）

既然无法直接获取真实的人类效用函数 $r^*$，我们可以用一个神经网络 $r_\phi$ 来拟合它。奖励模型通常由 SFT 后的模型去掉顶层分类头、换成一个标量回归头构成。它接收输入序列 $(x, y)$，输出一个标量分数。

训练奖励模型的目标是最大化二元分类的对数似然：

$$\mathcal{L}_{\text{RM}}(\phi) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right) \right]$$

训练完成后，奖励模型就充当了一个自动化的“打分裁判”。

### 阶段三：使用 PPO 算法更新策略模型

以 SFT 模型作为初始策略模型 $\pi_\theta$。为了防止模型在追求高奖励的过程中“钻空子”（即奖励黑客行为 Reward Hacking：生成胡言乱语但能骗取奖励模型高分的序列），优化目标中加入了与未更新的参考模型 $\pi_{\text{ref}}$ 之间的 KL 散度约束：

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \left[ r_\phi(x, y) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)) \right]$$

其中 $\beta$ 是控制约束力度的系数。在 token 级别展开时，每一步生成的即时奖励被设计为：

$$R(s_t, a_t) = \begin{cases} - \beta \log \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)} & (t < T) \\[8pt] r_\phi(x, y) - \beta \log \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)} & (t = T) \end{cases}$$

PPO（近端策略优化）算法根据当前状态计算广义优势估计（GAE），利用策略比率截断机制对策略模型参数 $\theta$ 进行平滑更新。

---

## 3. RLHF-PPO 的工程瓶颈：多模型协作与在线采样

在标准 RLHF-PPO 的训练节点上，系统通常需要协调四个模型角色：

| 模型角色 | 符号 | 参数状态 | 主要职责 |
| :--- | :--- | :--- | :--- |
| **策略模型 (Actor)** | $\pi_\theta$ | 训练中 (频繁反向传播) | 接收提示词自回归采样生成文本，更新梯度 |
| **价值模型 (Critic)** | $V_\psi$ | 训练中 (频繁反向传播) | 估计每个 token 的状态价值，用于计算优势函数 |
| **奖励模型 (Reward)** | $r_\phi$ | 冻结 (只读) | 对 Actor 完整生成的回答进行打分 |
| **参考模型 (Reference)** | $\pi_{\text{ref}}$ | 冻结 (只读) | 提供原始分布基准，用于计算 KL 散度惩罚 |

![RLHF 训练期四模型交互拓扑图](/images/content-rebuild/93e9006334df3302/fig-3.png)

这套架构带来了显著的系统复杂度：

1. **显存开销**：训练中的 Actor 和 Critic 模型不仅要存权重，还要存储 AdamW 优化器状态（每个参数约占用 8 字节）、梯度和前向激活值；Reward 和 Reference 即使冻结，也需要占用基础模型权重的只读显存。虽然在工程实现中，可以通过参数高效微调（如 LoRA）让 Actor 和 Reference 共享一套基座权重（关闭 LoRA adapter 即可作为 Reference 使用），或者让 Critic 和 Reward 复用部分底层特征，但多模型并行的显存调度开销依然显著。
2. **在线生成（Rollout）耗时占比极高**：PPO 属于在线强化学习（On-Policy），训练循环必须先由当前的 Actor 模型针对一批输入提示词完整生成回答序列。自回归解码是一个串行的内存带宽密集型操作，吞吐量远低于并行前向计算。Rollout 阶段通常会占据整个 PPO 训练时钟周期的 60% 至 80%。
3. **训练稳定性敏感**：价值模型 Critic 的估计误差如果出现偏差，会导致策略模型的优势函数方差剧烈波动，引发策略崩溃。

---

## 4. DPO 数学原理：省去显式奖励模型

由于 RLHF-PPO 的多模型流水线非常笨重，Rafael Rafailov 等人在 2023 年提出了直接偏好优化（DPO）。DPO 的核心数学推导证明了：**在带 KL 散度约束的强化学习框架下，最优策略与潜在奖励函数之间存在一一对应的代数关系。我们可以用策略模型本身的概率比值反解出隐式奖励，从而直接跳过奖励模型和强化学习过程。**

![DPO 数学置换机制推导流程图](/images/content-rebuild/93e9006334df3302/fig-4.png)

### 严谨推导过程

回顾在给定输入 $x$ 时，强化学习的目标函数：

$$\max_\pi \mathbb{E}_{y \sim \pi(\cdot \mid x)} [r(x, y)] - \beta \mathbb{D}_{\text{KL}}(\pi(y \mid x) \parallel \pi_{\text{ref}}(y \mid x))$$

展开期望与 KL 散度积分：

$$\sum_y \pi(y \mid x) r(x, y) - \beta \sum_y \pi(y \mid x) \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = - \beta \sum_y \pi(y \mid x) \left[ \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - \frac{1}{\beta} r(x, y) \right]$$

定义配分函数 $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$。利用配分函数的恒等变换，括号内的对数项可以改写为：

$$\log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - \frac{1}{\beta} r(x, y) = \log \frac{\pi(y \mid x)}{\frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)} - \log Z(x)$$

将改写后的项代回目标函数中，注意到 $\sum_y \pi(y \mid x) = 1$：

$$\max_\pi - \beta \mathbb{D}_{\text{KL}}\left( \pi(y \mid x) \;\parallel\; \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right) \right) + \beta \log Z(x)$$

因为 KL 散度 $\mathbb{D}_{\text{KL}}(P \parallel Q) \ge 0$，且当且仅当 $P = Q$ 时散度取最小值 0。因此，当策略分布等于以下解析表达式时，目标函数取得全局最优解 $\pi^*(y \mid x)$：

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$

两边取对数并移项，可以直接得到真实奖励函数 $r(x, y)$ 的代数表达：

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

现在将这一表达代入 Bradley-Terry 偏好概率公式中。计算胜出回答 $y_w$ 与落败回答 $y_l$ 的奖励差值：

$$r(x, y_w) - r(x, y_l) = \left( \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \log Z(x) \right)$$

由于配分函数 $Z(x)$ 仅依赖于输入提示词 $x$，与具体的生成回答 $y$ 无关，**两项相减时 $\beta \log Z(x)$ 被完全抵消**：

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$$

用参数化模型 $\pi_\theta$ 替代最优策略 $\pi^*$，并在静态偏好数据集 $\mathcal{D}$ 上最小化负对数似然，就得到了 DPO 的最终损失函数：

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

这个公式意味着：**语言模型自身就能兼任策略模型和奖励模型**。我们不需要单独训练一个奖励网络，也不需要在训练循环中进行在线生成采样，只需通过前向计算这四个序列的对数概率，就能直接完成偏好对齐。

---

## 5. DPO 损失函数的动力学分析与算例

为了直观观察 DPO 是如何调整模型权重的，我们对模型参数 $\theta$ 计算梯度：

$$\nabla_\theta \mathcal{L}_{\text{DPO}}(\theta) = - \beta \; \underbrace{\sigma\left(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\right)}_{\text{动态权重系数 } w(x, y_w, y_l)} \left[ \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right]$$

其中，我们把 $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ 视为策略模型当前的隐式奖励。

![DPO 损失函数梯度权重动态响应曲线](/images/content-rebuild/93e9006334df3302/fig-5.png)

该梯度由两部分组成：

1. **梯度更新方向**：方括号内的 $\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)$ 明确指出，参数更新会增大优选回答 $y_w$ 的全序列生成概率，同时压制劣选回答 $y_l$ 的概率。
2. **动态调节系数**：前面的标量权重 $w = \sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))$ 充当自动调节器。如果模型出现严重误判（即模型赋予劣选回答的隐式得分高于优选回答，$\hat{r}_\theta(x, y_l) > \hat{r}_\theta(x, y_w)$），权重 $w$ 趋近于 1，产生最大梯度进行纠偏；当模型已经大幅拉开两者差距时，权重趋近于 0，自动减弱更新力度，避免过度拟合。

### 逐步最小验证算例

设定超参数 $\beta = 0.1$。输入提示词 $x$ 仍为“如何保证睡眠质量？”：
- 优选回答 $y_w$：“建立固定的作息习惯...”
- 劣选回答 $y_l$：“白天拼命灌浓茶硬撑...”

假设在整个序列上累加得到的条件对数似然（数值为机制示意）如下表所示：

| 指标 | 优选回答 $y_w$ | 劣选回答 $y_l$ |
| :--- | :--- | :--- |
| 冻结参考模型 $\log \pi_{\text{ref}}(y \mid x)$ | $-12.0$ | $-14.0$ |
| 初始策略模型（状态 A）$\log \pi_{\theta_A}(y \mid x)$ | $-12.0$ | $-14.0$ |
| 微调后策略模型（状态 B）$\log \pi_{\theta_B}(y \mid x)$ | $-11.0$ | $-15.0$ |

#### 步骤 1：验证初始状态 A（未微调时，$\pi_\theta = \pi_{\text{ref}}$）

计算各自的隐式奖励：

$$\hat{r}_{\theta_A}(x, y_w) = 0.1 \times (-12.0 - (-12.0)) = 0.0$$

$$\hat{r}_{\theta_A}(x, y_l) = 0.1 \times (-14.0 - (-14.0)) = 0.0$$

隐式奖励差为 $0.0 - 0.0 = 0$。带入 DPO 损失公式：

$$\mathcal{L}_{\text{DPO}}(\theta_A) = - \log \sigma(0) = - \log(0.5) = \ln(2) \approx 0.6931$$

当策略模型与参考模型完全一致时，DPO 的初始 Loss 精确等于 $\ln 2 \approx 0.693$。在实际工程调试中，如果刚开始训练时的首个 Step Loss 显著偏离 0.693，通常说明数据对数概率计算或模型初始化存在异常。

#### 步骤 2：验证优化后状态 B（优选概率上升，劣选概率下降）

在状态 B 下，模型生成 $y_w$ 的对数概率由 $-12.0$ 提升至 $-11.0$（概率扩大为原先的 $e^1 \approx 2.72$ 倍），生成 $y_l$ 的对数概率由 $-14.0$ 压缩至 $-15.0$：

$$\hat{r}_{\theta_B}(x, y_w) = 0.1 \times (-11.0 - (-12.0)) = +0.1$$

$$\hat{r}_{\theta_B}(x, y_l) = 0.1 \times (-15.0 - (-14.0)) = -0.1$$

两者的差值为 $\hat{r}_{\theta_B}(x, y_w) - \hat{r}_{\theta_B}(x, y_l) = 0.1 - (-0.1) = 0.2$。代入损失函数：

$$\mathcal{L}_{\text{DPO}}(\theta_B) = - \log \sigma(0.2) = - \log\left( \frac{1}{1 + e^{-0.2}} \right) \approx - \log(0.5498) \approx 0.5982$$

动态梯度权重从初始的 $\sigma(0) = 0.5$ 降为：

$$w = \sigma(-0.2) = 1 - 0.5498 \approx 0.4502$$

这直观说明了模型是如何随着对数概率差别的拉大，逐步平滑降低损失并减弱纠偏力度的。

---

## 6. 为什么不能丢弃参考模型（Reference Model）？

在 DPO 的实现中，虽然不需要独立的奖励模型，但内存中依然需要保留一份基座概率基准 $\pi_{\text{ref}}$。在 Hugging Face TRL 的 `DPOTrainer` 典型代码中，如果使用 LoRA 进行轻量微调，系统会在前向计算时通过临时停用 adapter 来获取 $\pi_{\text{ref}}$ 的输出：

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

# 配置 LoRA：传入 peft_config 后，DPOTrainer 会复用同一份基座权重，
# 计算参考模型分布时自动在上下文中临时停用 LoRA adapter，无需占用双份基座显存
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,  # 调节 KL 散度约束强度的关键超参数
)

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-7B-Instruct",
    args=training_args,
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
    peft_config=peft_config,
)
trainer.train()
```

如果为了极端省显存，把参考模型完全去掉（等价于令 $\pi_{\text{ref}}(y \mid x) = 1$），或者将约束强度 $\beta \to 0$，会导致明显的生成质量退化：

1. **概率作弊与语法崩溃**：没有参考模型作为针脚锚定，优化目标退化为无约束的 $\log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x)$。要拉大这个差距，模型最轻松的策略不是把 $y_w$ 写得更好，而是通过梯度更新彻底破坏 $y_l$ 中出现的常用虚词、标点符号的概率。这会导致模型的基础语法表达能力迅速失常。
2. **长度偏差（Length Bias）失控**：序列对数似然是各个 token 条件对数概率的连加。在自回归模型中，更长的句子由于相加项更多，累加值通常更负。参考模型的作用是在相同文本长度上提供一个基准线，抵消由于文本长短带来的数值偏差。

超参数 $\beta$ 决定了模型被允许偏离初始模型的程度：$\beta$ 设置过大，模型几乎不更新偏好；$\beta$ 设置过小，模型容易迅速产生分布漂移。

---

## 7. RLHF 与 DPO 的现实权衡：离线拟合 vs 在线探索

虽然 DPO 在工程上手门槛和计算资源需求上具备压倒性优势，但在复杂推理任务（例如数学推理、多步代码执行）以及前沿主力基座的对齐中，基于在线强化学习的方法（包括 PPO 以及后续衍生出的 RLOO、GRPO 等）依然保持着重要地位。

![离线拟合与在线探索的状态空间分布差异](/images/content-rebuild/93e9006334df3302/fig-6.png)

### 核心差异对比

| 维度 | RLHF-PPO | DPO (Direct Preference Optimization) |
| :--- | :--- | :--- |
| **算法形式** | 在线强化学习（On-Policy） | 离线对比学习（Offline） |
| **在线生成开销** | 极高（训练循环中持续串行自回归解码） | 无（仅需对静态数据进行并行前向打分） |
| **显存占用** | 高（需维护策略、价值、奖励与参考模型） | 低（仅需策略模型及参考分布，支持 LoRA 动态复用） |
| **数据分布探索** | 动态探索（策略模型自主生成新轨迹） | 静态受限（严格局限于离线数据集中的两项回答） |
| **训练稳定性** | 容易受超参数和优势函数方差波动影响 | 类似普通监督训练，收敛过程更平稳 |

### 离线拟合的局限与在线探索的价值

- **DPO 的主要瓶颈是分布偏移（Out-Of-Distribution, OOD）**：DPO 使用的数据对 $(y_w, y_l)$ 是在历史策略模型下预先采样好的静态文本。随着 DPO 训练的推进，策略模型 $\pi_\theta$ 自身的生成分布已经改变。模型在实际推理时可能会生成一些历史偏好数据中从未见过的表达。由于离线训练期间无法对这些新生成的表达进行重新打分，模型可能会对某些偶发错误赋予虚高的隐式奖励，产生胡言乱语或循环复读。
- **RLHF 的核心价值是在线探索能力**：RLHF 在训练循环中，由当前的策略模型针对提示词实时生成回答（Rollout），再交由奖励模型动态打分。这种在线循环让模型有机会探索超出原始数据集的新解题路径。即使模型采样出了未在训练集中出现的推理逻辑，只要结果正确，就能获得及时的正向奖励反馈。这也是为什么在追求强推理能力的大模型对齐中，在线强化学习依然是不可或缺的探索手段。

---

## 关联索引与进阶阅读

理解偏好对齐有助于理清模型在监督微调之后如何向人类意图靠拢。建议结合以下章节建立完整的训练链路认知：

- **前置基础**：了解完整训练流水线中 SFT 与对齐的衔接逻辑，请参阅 [大模型训练流程是什么？](/fundamentals/training-and-finetuning/training-pipeline.md) 与 [SFT 监督微调是什么？](/fundamentals/training-and-finetuning/sft.md)。
- **微调实现配合**：了解如何在显存受限条件下通过 LoRA 实现低成本对齐，请参阅 [LoRA 和 QLoRA 是什么？](/fundamentals/training-and-finetuning/lora-qlora.md)。
- **数据清洗与排错**：深入了解偏好对中常见的数据标注噪音与训练调优技巧，请参阅 [微调数据怎么构造？](/fundamentals/training-and-finetuning/finetuning-data.md) 与 [微调常见问题有哪些？](/fundamentals/training-and-finetuning/finetuning-pitfalls.md)。
