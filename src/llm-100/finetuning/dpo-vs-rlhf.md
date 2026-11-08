---
title: DPO 是什么？它与传统 RLHF 有何区别与工程权衡
icon: mdi:tune-vertical
tag:
- 微调与对齐
- 大模型100讲
- RLHF
description: 对比直接偏好优化（DPO）与传统 PPO-RLHF 的数学原理、系统架构与训练差异，掌握离线偏好对齐的工程落地与局限。
---

大语言模型经过有监督微调（SFT）后，虽然学会了回答问题的句式与基本知识，但往往仍会出现输出冗长、事实幻觉或不合人类意图的内容。在上一讲 [/llm-100/finetuning/why-rlhf.md](/llm-100/finetuning/why-rlhf.md) 中，我们讨论了如何通过人类反馈来对齐模型偏好。

在实际工程中，最知名的对齐方法是基于近端策略优化（PPO）的经典 RLHF 链路。但经典 RLHF 架构复杂、显存开销极大且调优困难。斯坦福大学在 2023 年提出的**直接偏好优化（Direct Preference Optimization, DPO）**，通过数学推导直接将奖励函数替换为策略模型本身的概率比，完全省去了独立的奖励模型与在线采样环节。本讲将用一个贯穿例子，拆解 DPO 的工作原理、损失函数计算、数据流组织以及选型边界。

![DPO 与传统 RLHF 对齐演进封面](/images/content-rebuild/7de63c41ed7da633/cover.png)

---

## 一个贯穿例子：从偏好标注到模型更新

为了直观理解偏好对齐在做什么，我们假定模型收到如下输入：

- **提示词 $x$**：“请用一句话解释什么是 API。”
- **偏好回答（赢家 $y_w$）**：“API 是软件之间互相通信的一套接口规范，类似于餐馆中服务员传递顾客与后厨之间的点单信息。”（表达精炼、比喻恰当）
- **不合偏好回答（输家 $y_l$）**：“API 是 Application Programming Interface 的缩写，在计算机软件工程中被广泛应用，有很多系统都在使用它，非常重要。”（空洞废话，未解释机制）

对齐的目标是：**在保持模型基础语言能力不退化的前提下，调高模型生成 $y_w$ 的概率，同时调低生成 $y_l$ 的概率。**

实现这一目标，传统 RLHF 与 DPO 采取了完全不同的工程路径：
- **传统 RLHF**：先训练一个专门的“打分裁判”（奖励模型），然后在训练循环中让生成模型不断吐出新文本，裁判打分，再用强化学习算法（如 PPO）更新模型参数。
- **DPO**：不训练单独的裁判，直接在离线收集好的 $(x, y_w, y_l)$ 数据上，通过特殊的分类交叉熵损失，直接拉大赢家与输家的对数概率差。

---

## 传统 PPO-RLHF 的工程瓶颈：四模型常驻与在线采样

在标准实现的 PPO-RLHF 训练中，训练集群的显存中通常需要同时支持四个模型：

1. **策略模型（Actor $\pi_\theta$）**：当前正在更新参数的语言模型，负责自回归生成文本并根据优势函数计算策略梯度。
2. **价值模型（Critic $V_\phi$）**：输出标量分数的网络，用于估计当前状态的期望累计收益，协助计算广义优势估计（GAE），需要持续更新梯度。
3. **参考模型（Reference Model $\pi_{\mathrm{ref}}$）**：参数完全冻结的初始 SFT 模型。在计算每个 token 的损失时，它提供基准对数概率，用于施加 KL 散度惩罚，防止模型为了迎合奖励而丧失基本的语言连贯性。
4. **奖励模型（Reward Model $r_\psi$）**：参数完全冻结的打分模型，负责对生成的完整回复给出质量标量。

![传统 PPO-RLHF 四模型在线交互拓扑](/images/content-rebuild/7de63c41ed7da633/fig-1.png)

上图展示了传统 PPO-RLHF 的模型拓扑。这种设计在实际工程落地时存在两个突出瓶颈：

- **显存占用门槛高**：在典型的 16 位浮点（如 FP16 或 BF16）混合精度 AdamW 训练中，每个可训练参数通常需要约 16 字节显存（含模型权重 2 字节、梯度 2 字节、优化器一阶与二阶动量 8 字节、FP32 主权重备份 4 字节）。Actor 与 Critic 两个模型都在持续反向传播，再加上常驻显存用于前向推理的 Reference 与 Reward 模型，导致单机显存压力极大。
- **在线生成（Rollout）耗时占比极高**：PPO 是在线（On-policy）算法。为了得到当前步的强化学习优势值，必须由当前的策略模型 $\pi_\theta$ 针对 Prompt 自回归逐 token 生成一段完整文本。由于大模型解码受显存带宽限制，整个训练时钟周期中往往有 60%～80% 的时间在等待自回归生成。

<ReadMoreLock />

---

## DPO 推导：如何把奖励函数代换掉？

DPO 的核心洞见在于：**在带 KL 散度约束的强化学习数学目标下，最优策略与真实奖励函数之间存在严格的闭式解析解。因此，奖励函数可以直接用“当前策略与参考策略的概率比”表示。**

这意味着我们不再需要一个独立的网络去拟合奖励，而是把语言模型自身当作隐式奖励模型。

![DPO 数学推导与配分函数相消机制图](/images/content-rebuild/7de63c41ed7da633/fig-2.png)

其推导逻辑可以分为三步：

### 步骤 1：RL 约束目标的闭式解展开
给定输入 Prompt $x$，经典的对齐目标是最大化奖励，同时惩罚与参考模型之间的 KL 散度：
$$
\max_{\pi} \mathbb{E}_{x} \left[ \mathbb{E}_{y \sim \pi(\cdot|x)} [r(x,y)] - \beta D_{\mathrm{KL}}(\pi(\cdot|x) \parallel \pi_{\mathrm{ref}}(\cdot|x)) \right]
$$
其中 $\beta$ 是控制偏离惩罚力度的超参数。展开求和项后，可以构造一个包含配分函数 $Z(x)$ 的形式：
$$
Z(x) = \sum_y \pi_{\mathrm{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x,y) \right)
$$
经过相对熵代换，当策略与理论最优分布一致时目标取最大值，由此得到最优策略 $\pi^*$ 的显式解：
$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x,y) \right)
$$

### 步骤 2：用策略反解奖励函数
对上述最优策略公式两边取对数并整理，可以把未知的奖励 $r(x,y)$ 表达为：
$$
r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta \log Z(x)
$$

### 步骤 3：在偏好概率中相消配分函数
根据 Bradley-Terry (BT) 人类偏好模型，在赢家回答 $y_w$ 与输家回答 $y_l$ 之间，人类偏好 $y_w$ 的概率由两者的奖励差决定：
$$
P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))
$$
将步骤 2 反解出的奖励代入两者的差值项：
$$
r(x, y_w) - r(x, y_l) = \left[ \beta \log \frac{\pi^*(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} + \beta \log Z(x) \right] - \left[ \beta \log \frac{\pi^*(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} + \beta \log Z(x) \right]
$$
由于赢家和输家来自同一个输入 $x$，**配分函数项 $\beta \log Z(x)$ 被完全相消抵消**！奖励差纯粹由两个模型在这两个序列上的对数概率比决定：
$$
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}
$$

---

## DPO 损失函数结构与步进算例

用参数化模型 $\pi_\theta$ 替代理论最优策略 $\pi^*$，并套入负对数似然形式，便得到了 DPO 的最终损失函数：
$$
\mathcal{L}_{\mathrm{DPO}}(\theta; \pi_{\mathrm{ref}}) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} \right) \right]
$$

![隐式奖励差与损失梯度权重的动态抑制机制](/images/content-rebuild/7de63c41ed7da633/fig-3.png)

### 梯度拆解与动力学
我们定义模型当前的隐式奖励为 $\hat{r}_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}$，奖励差值为 $\Delta = \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$。对参数 $\theta$ 求导后，梯度为：
$$
\nabla_\theta \mathcal{L}_{\mathrm{DPO}}(\theta) = - \beta \cdot \sigma(-\Delta) \left[ \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right]
$$

该梯度包含两个核心因子：
1. **更新方向项** $[\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)]$：推动模型调高赢家序列各 token 的生成几率，同时同等幅度调低输家序列的生成几率。
2. **自适应调节标量 $\sigma(-\Delta)$**：
   - 当模型产生错误打分（即当前模型认为输家比赢家更好，$\Delta < 0$）时，$\sigma(-\Delta) \to 1$，提供最大的梯度更新力度；
   - 当模型已经正确且显著地拉开差距（$\Delta \gg 0$）时，$\sigma(-\Delta) \to 0$，梯度权重自动归零，避免在已经学好的样本上过度惩罚。

### 步进数值验证（示意算例）
设超参数 $\beta = 0.1$。输入前文贯穿例子中的 Prompt，两个回答的累加序列对数概率（假设值）如下表：

| 序列 | 参考模型对数概率 $\log \pi_{\mathrm{ref}}$ | 当前策略模型对数概率 $\log \pi_\theta$ | 隐式奖励 $\hat{r} = \beta (\log \pi_\theta - \log \pi_{\mathrm{ref}})$ |
| :--- | :--- | :--- | :--- |
| 赢家 $y_w$ | $-10.0$ | $-9.0$ | $0.1 \times (-9.0 - (-10.0)) = +0.10$ |
| 输家 $y_l$ | $-8.0$ | $-8.5$ | $0.1 \times (-8.5 - (-8.0)) = -0.05$ |

1. **计算隐式奖励差**：
   $$\Delta = 0.10 - (-0.05) = 0.15$$
2. **计算 Sigmoid 胜率预测**：
   $$\sigma(0.15) = \frac{1}{1 + e^{-0.15}} \approx \frac{1}{1 + 0.8607} \approx 0.5374$$
3. **计算当前样本标量损失**：
   $$\mathcal{L}_{\mathrm{DPO}} = -\log(0.5374) \approx 0.6210$$
4. **反向传播缩放权重**：
   $$\sigma(-\Delta) = 1 - 0.5374 = 0.4626$$
   模型将以 $0.1 \times 0.4626 = 0.04626$ 的有效比例系数，沿增加赢家似然、减少输家似然的方向更新权重。

---

## 数据组织与工程实现：离线批处理

在数据组织方式上，DPO 不需要强化学习维护的在线轨迹缓冲区（Trajectory Buffer），而是回归到纯粹的离线批处理。其基本数据单元为三元组：$(x, y_{\mathrm{chosen}}, y_{\mathrm{rejected}})$，参见 [/fundamentals/training-and-finetuning/finetuning-data.md](/fundamentals/training-and-finetuning/finetuning-data.md)。

![DPO 离线批处理拼接与对齐计算流程](/images/content-rebuild/7de63c41ed7da633/fig-4.png)

### 批处理与 Label Masking
一个批次大小为 $N$ 的偏好数据，在训练时的具体计算流如下：
1. **输入拼接**：将 Prompt 与 Chosen 拼成序列 $A$，Prompt 与 Rejected 拼成序列 $B$。一个大小为 $N$ 的数据批次，在输入模型时对应为 $2N$ 条序列。
2. **标签掩码（Label Masking）**：计算自回归交叉熵时，Prompt 对应位置的 token 标签设置为 `-100`，使反向传播只计算回答部分（Chosen / Rejected）对应 token 的对数概率之和。
3. **参考模型概率复用**：参考模型 $\pi_{\mathrm{ref}}$ 在训练全程保持参数冻结。可以在训练启动前对离线数据预先推理一遍参考概率并持久化，或者在每个训练步通过 `torch.no_grad()` 做一次前向传播，从而省去为参考模型保留梯度与优化器状态的显存开销。

### 最小可验证实现（PyTorch 核心逻辑）

以下代码展示了 DPO 损失的核心计算逻辑（符合 Hugging Face TRL 的算法规范）：

```python
import torch
import torch.nn.functional as F

def compute_dpo_loss(
    policy_chosen_logps: torch.FloatTensor,   # 形状: (batch_size,), 整个回答段 token 对数概率总和
    policy_rejected_logps: torch.FloatTensor, # 形状: (batch_size,), 整个回答段 token 对数概率总和
    reference_chosen_logps: torch.FloatTensor,# 形状: (batch_size,)
    reference_rejected_logps: torch.FloatTensor,# 形状: (batch_size,)
    beta: float = 0.1,
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    根据策略模型与参考模型在赢家/输家上的序列对数概率总和计算 DPO 标量损失。
    """
    # 策略模型与参考模型在 Chosen 与 Rejected 上的对数概率比值之差
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    
    logits = pi_logratios - ref_logratios
    
    # DPO 标量损失：-log(sigmoid(beta * logits))
    losses = -F.logsigmoid(beta * logits)
    
    # 隐式奖励监控（无梯度）
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
        
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

# 验证示例
if __name__ == "__main__":
    # 模拟两个样本在回答段的对数概率和
    p_win = torch.tensor([-9.0, -12.5], requires_grad=True)
    p_lose = torch.tensor([-8.5, -11.0], requires_grad=True)
    ref_win = torch.tensor([-10.0, -13.0])
    ref_lose = torch.tensor([-8.0, -10.5])
    
    loss, r_win, r_lose = compute_dpo_loss(p_win, p_lose, ref_win, ref_lose, beta=0.1)
    loss.backward()
    
    print(f"DPO 损失: {loss.item():.4f}")
    print(f"赢家隐式均值奖励: {r_win.item():.4f}, 输家隐式均值奖励: {r_lose.item():.4f}")
    print(f"梯度存在性: win={p_win.grad is not None}, lose={p_lose.grad is not None}")
```

---

## 工程权衡与现实边界：DPO 并非万灵丹

虽然 DPO 省去了独立奖励模型和在线采样，使对齐训练的显存与吞吐表现接近普通 SFT，但在实际业务中它也面临明显的边界制约：

![离线与在线对齐分布偏移对比](/images/content-rebuild/7de63c41ed7da633/fig-5.png)

### 1. 离线分布偏移（Distribution Drift）
- **在线强化学习（PPO）的动态闭环**：PPO 让模型实时生成文本。如果模型在训练过程中开始产生新的重复口癖或逻辑漏洞，奖励模型会立即在下一迭代对这些新生成样本给以低分惩罚，将模型拉回正常轨道。
- **DPO 的离线视界局限**：DPO 的样本对 $(y_w, y_l)$ 是提前写死在数据集中的。训练过程中，策略模型 $\pi_\theta$ **从不自主生成文本**。当损失函数压低 $y_l$ 的概率、提升 $y_w$ 的概率时，模型可能会把概率质量转移到离线数据集中从未出现过的异常文本模式中（例如特定无意义标点或死循环），造成真实线上推理时的生成退化。

### 2. 长度偏差（Length Bias）与数据噪声
在构建偏好对时，人工或大模型打分员往往更容易给篇幅较长、结构详尽的回答打高分。DPO 会忠实地学习这一统计特征。由于 DPO 计算隐式奖励时累加序列各 token 的对数几率差而未对序列长度做归一化，若策略模型在某些词上稍有正向偏离，越长的句子累积的奖励差就越大，使模型极易学到“只要把句子写长，就能拿到更高隐式奖励”的捷径。

另外，如果偏好数据集中存在标注冲突或噪声标签，DPO 的离线拟合会强行拟合错误偏好，而不像 PPO 那样通过奖励模型的平滑打分具备一定缓冲空间。

### 3. 超参数 $\beta$ 过于敏感
超参数 $\beta$ 既是隐式奖励的尺度因子，也是对偏离参考模型的惩罚系数：
- 若 $\beta$ 设得过大，隐式奖励差被放大，Sigmoid 项容易迅速饱和，导致反向传播梯度过小，模型学不动；
- 若 $\beta$ 设得过小，参考模型的锚定约束变弱，模型容易在离线偏好样本上过拟合，甚至出现灾难性遗忘。

### 偏好对齐方案选型参考

![偏好对齐工程选型决策路径](/images/content-rebuild/7de63c41ed7da633/fig-6.png)

| 维度 | 经典 PPO-RLHF | 直接偏好优化（DPO） |
| :--- | :--- | :--- |
| **运行时模型数** | 典型涉及 4 个角色（可通过共享骨干或卸载复用显存） | 1～2 个（Policy 训练；若预计算参考对数概率则仅需 1 个） |
| **在线推理采样** | 必需（Rollout 阶段耗时占比大） | 无（完全基于静态离线样本） |
| **硬件显存门槛** | 极高（多卡分布式或复杂卸载） | 适中（与标准 SFT 规模接近） |
| **抗生成退化能力** | 较强（在线探索具备自纠偏能力） | 较弱（易受离线数据分布偏移影响） |
| **适合团队与场景** | 算力充裕、有完整在线评测闭环的基座模型团队 | 领域垂直微调、显存受限、追求快速交付的工程团队 |

---

## 验收练习与可检验核对

完成本讲学习后，请尝试回答以下问题以检验掌握程度：

1. **梯度调节验证**：假设某个 Prompt 的赢家回答与输家回答差异极大，当前策略模型已经为其赋予了极大的隐式奖励差（$\Delta = \hat{r}_\theta(y_w) - \hat{r}_\theta(y_l) = 15.0$）。此时 DPO 梯度的缩放标量 $\sigma(-\Delta)$ 会接近多少？这对已经掌握得很好的样本起到了什么保护作用？
2. **离线退化排查**：某团队使用 DPO 训练了一个营销文案模型，离线验证集准确率达标，但上线推理时发现模型频繁生成超长排版和口水词。根据本讲分析，造成这种现象的主要机制可能是什么？在不改换算法的前提下，可以从数据端采取什么具体措施？

在下一讲 [/llm-100/finetuning/finetuning-overfitting.md](/llm-100/finetuning/finetuning-overfitting.md) 中，我们将深入剖析模型在微调与对齐阶段最容易遭遇的病态问题——微调过拟合，探讨其数学特征与工程缓解方案。
