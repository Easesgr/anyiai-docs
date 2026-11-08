---
title: RLHF 和 DPO 有什么区别？原理、推导与工程对比
icon: mdi:scale-balance
tag:
- 大模型面试
- RLHF
- DPO
description: 对比经典 RLHF（PPO）与 DPO 的数学推导、系统架构与适用边界，解释配分函数相消机制与离线学习局限。
---

在大模型对齐算法面试中，“DPO 与经典 RLHF 有什么区别”是一个兼具数学推导与工程落地的高频考点。考官不仅关注“DPO 不需要独立奖励模型”这层表象，更看重候选人能否讲清：**带 KL 约束的强化学习最优解如何反解出奖励函数、配分函数为何能在偏好概率差值中精确抵消，以及从在线探索切换到离线学习带来了哪些工程收益与泛化代价**。

![DPO 机制与工程平替概览](/images/content-rebuild/f4cccb9faea70f5b/cover.png)

---

## 1. 30 秒口述结论：DPO 与经典 RLHF 的核心差异

在面试中被问及两者区别时，可按“机制原理、工程架构、适用边界”三层展开口述：

1. **机制区别（显式拟合 vs 隐式代换）**：
   - **经典 RLHF（以 PPO 为代表）**：分为三阶段（SFT $\rightarrow$ 奖励模型训练 $\rightarrow$ PPO 在线强化学习）。PPO 阶段由策略网络在线自回归生成文本，奖励模型给出打分，配合价值网络（Critic）估计优势函数，并在参考模型的 KL 散度约束下更新策略。
   - **DPO（直接偏好优化）**：利用数学推导，证明了在 KL 约束强化学习目标的最优解下，**奖励函数可以由当前策略与参考策略的对数几率比显式表示**。将其代入 Bradley-Terry 偏好模型后，配分函数在两式作差时被精确抵消，从而跳过了独立的奖励模型训练与在线生成环节，将对齐转化为针对静态偏好数据对的二分类交叉熵优化。
2. **工程区别（在线多模型调度 vs 离线双模型前向）**：
   - **经典 PPO 架构**：训练时需常驻策略（Actor）、价值（Critic）、奖励（RM）、参考（Ref）四个模型实例。在线自回归采样（Rollout）耗时长、显存与通信开销大，且容易出现训练抖动与数值崩溃。
   - **DPO 架构**：仅需保留当前策略模型和冻结的参考模型，且无需在线采样，只需对静态数据集中的回复计算前向对数似然（Log-Likelihood）。显存压力小，计算吞吐更高，训练过程与普通有监督微调（SFT）类似。
3. **适用边界（在线探索上限 vs 离线分布偏移）**：
   - DPO 属于离线（Off-policy）优化，训练数据是静态的。当策略更新脱离偏好数据覆盖的分布范围时，容易发生分布偏移（Distribution Shift）与流利度下降，且存在偏好更长文本的长度偏置。
   - 在通用对话与风格对齐场景中，DPO 凭借高工程效率被广泛应用；但在数学推理、代码生成等**具备明确客观判据（Ground Truth）、依赖多步在线探索**的任务中，支持动态探索与验证器反馈的在线强化学习（如 PPO、GRPO）依然保持性能上限优势。

---

## 2. 贯穿示例：偏好对齐到底在对齐什么？

为了直观理解两者的机制差异，以一个具体的代码指令问答为例：

- **输入提示词 $x$**：“请用 Python 实现一个快速排序函数，并说明其最坏时间复杂度。”
- **偏好回答 $y_w$（胜出项）**：代码实现完整，边界条件（空列表或单元素列表）处理正确，明确指出当基准选取不当时最坏时间复杂度为 $O(n^2)$。
- **落选回答 $y_l$（被拒绝项）**：递归函数缺少终止条件导致死循环，且错误地声称最坏时间复杂度始终为 $O(n \log n)$。

在**经典 RLHF** 中：
1. 标注员标注出 $y_w \succ y_l$（偏好 $y_w$ 胜过 $y_l$）；
2. 训练一个独立的奖励模型 $r(x, y)$，使其对 $(x, y_w)$ 的打分高于 $(x, y_l)$；
3. 在 PPO 阶段，策略模型接收 $x$，**实时自回归生成一段全新的回复 $y'$**，奖励模型为 $y'$ 打分，PPO 依靠该标量奖励指导策略参数更新。

在 **DPO** 中：
1. 同样拿到离线三元组 $(x, y_w, y_l)$；
2. 模型不需要在线生成新文本 $y'$，而是直接计算策略模型 $\pi_\theta$ 和参考模型 $\pi_{\text{ref}}$ 在既有文本 $y_w$ 和 $y_l$ 上的 token 对数概率；
3. 通过损失函数拉大 $y_w$ 相对于参考模型的概率优势，同时压低 $y_l$ 的概率优势。

---

## 3. 数学推导：从 KL 约束强化学习到 DPO 目标

DPO 并非经验性的启发式损失，而是从经典强化学习优化目标严格推导而来的闭式解代换。

经典 RLHF 的策略优化阶段目标是：在提示词分布 $\mathcal{D}$ 上，最大化奖励模型期望得分，同时通过反向 KL 散度约束策略 $\pi(y|x)$ 不能偏离初始参考策略 $\pi_{\text{ref}}(y|x)$ 过远（详见 [/interview/llm-interview/rlhf-pipeline.md](/interview/llm-interview/rlhf-pipeline.md)）：

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathbb{E}_{y \sim \pi(\cdot|x)} [r(x, y)] - \beta \mathbb{D}_{\text{KL}}(\pi(\cdot|x) \parallel \pi_{\text{ref}}(\cdot|x)) \right]$$

其中 $\beta > 0$ 是控制 KL 散度惩罚力度的超参数。

![从带 KL 约束的强化学习推导策略闭式解](/images/content-rebuild/f4cccb9faea70f5b/fig-1.png)

针对固定的输入提示词 $x$，将期望写为对离散序列 $y$ 的求和形式：

$$\max_{\pi} \sum_{y} \pi(y|x) r(x, y) - \beta \sum_{y} \pi(y|x) \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$$

将最大化目标取相反数等价转化为最小化问题，并提取正系数 $\beta > 0$（省略正系数不改变最优策略解）：

$$\min_{\pi} \sum_{y} \pi(y|x) \left[ \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta} r(x, y) \right] = \min_{\pi} \sum_{y} \pi(y|x) \log \left( \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)} \right)$$

为了将对数内部的分母构造为一个合法的概率分布，定义**配分函数（Partition Function）** $Z(x)$：

$$Z(x) = \sum_{y} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$

此时构造辅助分布 $\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)$。显然 $\sum_y \pi^*(y|x) = 1$，它是一个合法的概率分布。将 $\pi^*(y|x)$ 带回目标函数：

$$\min_{\pi} \sum_{y} \pi(y|x) \log \left( \frac{\pi(y|x)}{Z(x) \pi^*(y|x)} \right) = \min_{\pi} \left[ \sum_{y} \pi(y|x) \log \frac{\pi(y|x)}{\pi^*(y|x)} - \log Z(x) \sum_y \pi(y|x) \right]$$

由于 $\sum_y \pi(y|x) = 1$，且配分函数 $Z(x)$ 仅由输入 $x$ 决定、与待优化的策略 $\pi$ 无关，因此最小化目标转化为：

$$\min_{\pi} \left[ \mathbb{D}_{\text{KL}}(\pi(\cdot|x) \parallel \pi^*(\cdot|x)) - \log Z(x) \right]$$

根据 KL 散度的非负性，当且仅当 $\pi(y|x) = \pi^*(y|x)$ 时，KL 散度取得极小值 $0$。因此，带 KL 约束的强化学习最优策略具备**唯一的闭式解析解（Closed-form Solution）**：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$

对等式两边取自然对数并移项，可以直接用最优策略反解出奖励函数 $r(x, y)$：

$$\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta} r(x, y) - \log Z(x)$$

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

此时，原本黑盒的奖励函数 $r(x, y)$ 被严格表达为最优策略与参考策略的对数几率比，外加一个只和 $x$ 相关的配分函数项。

<ReadMoreLock />

---

## 4. 消除配分函数：DPO 损失函数的落地

在经典人类偏好建模中，通常采用 Bradley-Terry (BT) 模型。对于同一提示词 $x$，人类偏好 $y_w$ 超过 $y_l$ 的概率由两者的奖励差决定：

$$p(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l)) = \frac{1}{1 + \exp\left(-(r(x, y_w) - r(x, y_l))\right)}$$

其中 $\sigma(t)$ 为标准 Sigmoid 函数。

![隐式奖励差值消除配分函数的过程](/images/content-rebuild/f4cccb9faea70f5b/fig-2.png)

将反解出的奖励闭式解代入奖励差值项 $(r(x, y_w) - r(x, y_l))$：

$$r(x, y_w) - r(x, y_l) = \left( \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} + \beta \log Z(x) \right)$$

关键性质在此显现：**配分函数 $Z(x)$ 仅取决于输入提示词 $x$，与生成回复 $y_w$ 或 $y_l$ 均无关联。因此在两式相减时，$\beta \log Z(x) - \beta \log Z(x) = 0$ 被精确相消**：

$$r(x, y_w) - r(x, y_l) = \beta \left( \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)$$

在训练时，使用待优化的参数化策略模型 $\pi_\theta$ 逼近最优策略 $\pi^*$。在离线偏好数据集 $\mathcal{D} = \{(x, y_w, y_l)\}$ 上，通过极大似然估计（即最小化负对数似然），便得到了 **DPO 的最终损失函数**：

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

### 梯度拆解与自适应加权机制

对模型参数 $\theta$ 求梯度，可以看清 DPO 的具体更新行为：

$$\nabla_\theta \mathcal{L}_{\text{DPO}}(\theta) = - \beta \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)) \left( \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right) \right]$$

其中定义隐式奖励 $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$。由于 Sigmoid 导数具有性质 $1 - \sigma(u) = \sigma(-u)$，所以求导后的系数项正是 $\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))$。

![DPO 梯度的动态加权调节机制](/images/content-rebuild/f4cccb9faea70f5b/fig-3.png)

该梯度反映了两层更新机制：
1. **更新方向**：由 $(\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x))$ 决定，即同时增大胜出项 $y_w$ 的生成对数似然，压低落选项 $y_l$ 的生成对数似然。
2. **更新幅度**：标量系数 $\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))$ 充当自适应误差权重。当模型判断错误（当前隐式奖励 $\hat{r}_\theta(x, y_l) > \hat{r}_\theta(x, y_w)$）时，该标量趋近于 $1$，施加极大的修正梯度；而当模型已经能够准确区分两者的优劣（$\hat{r}_\theta(x, y_w) \gg \hat{r}_\theta(x, y_l)$）时，系数迅速衰减至接近 $0$，自动抑制对已有正确知识的无效过拟合。

---

## 5. 工程实现与架构对比

经典 RLHF（PPO）由于引入了强化学习采样循环，在工程实现上面临较大的显存开销与长链路调度挑战；而 DPO 将问题收敛为有监督分类形式，工程复杂度明显降低。

![PPO 与 DPO 架构部署及显存模型实体对比](/images/content-rebuild/f4cccb9faea70f5b/fig-4.png)

| 维度 | 经典 PPO 流程 | 直接偏好优化（DPO） |
| :--- | :--- | :--- |
| **常驻模型实例** | 4 个（Actor 策略、Critic 价值、Reward 奖励、Reference 参考） | 2 个（Actor 策略、Reference 参考） |
| **显存与参数占用** | Critic 通常需要与 Actor 类似规模的显存与独立优化器状态；四模型调度显存开销大 | Reference 模型参数冻结，使用 LoRA 微调时甚至可共享同一基座底模 |
| **计算瓶颈环节** | **在线自回归 Rollout**（Decode 阶段串行、访存密集、吞吐受限） | 仅计算离线文本的前向 Log-Likelihood（Prefill 批处理计算密集），**无在线生成** |
| **数值稳定性** | 易受 GAE 优势估计方差、Critic 拟合误差及价值崩塌影响，调参超参数极多 | 损失函数为二分类交叉熵，训练平稳度与常规 SFT 相似 |
| **流水线阶段** | 需先训练 RM，验证其精度后再启动 PPO，流程涉及多阶段串行 | 获得标注偏好数据后，直接在 SFT 检查点上微调，流程阶段被压缩 |

在 PyTorch 中，DPO 核心损失函数的代码实现非常紧凑：

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,    # [batch_size]，策略网络在偏好回答 yw 上的 token 对数概率之和
    policy_rejected_logps: torch.FloatTensor,  # [batch_size]，策略网络在落选回答 yl 上的 token 对数概率之和
    ref_chosen_logps: torch.FloatTensor,       # [batch_size]，参考模型在偏好回答 yw 上的 token 对数概率之和
    ref_rejected_logps: torch.FloatTensor,     # [batch_size]，参考模型在落选回答 yl 上的 token 对数概率之和
    beta: float = 0.1,
) -> torch.FloatTensor:
    """
    计算 DPO 标量损失。
    输入均为张量，表示针对整个回复序列按 token 累加后的 log-prob。
    """
    # 计算当前策略对偏好对的对数几率差
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    # 计算参考模型对偏好对的对数几率差
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    logits = pi_logratios - ref_logratios
    
    # DPO 目标是最大化 log(sigmoid(beta * logits))，最小化其相反数
    losses = -F.logsigmoid(beta * logits)
    return losses.mean()
```

---

## 6. DPO 的适用条件与固有局限

尽管 DPO 大幅降低了工程实现门槛，但将其简单视为“全面取代 PPO”并不准确。从数学假设到工程落地，开发者需注意以下三项固有边界：

![DPO 离线分布偏移与未见样本泛化盲区](/images/content-rebuild/f4cccb9faea70f5b/fig-5.png)

### 1. 离线学习引发的分布偏移（Distribution Shift）

DPO 本质上是**离线（Off-policy）学习**。其偏好数据对 $(x, y_w, y_l)$ 通常是在 SFT 阶段甚至更早由旧模型采样生成的。

随着策略网络 $\pi_\theta$ 在训练过程中持续更新，它在真实推理时采样输出的空间分布会逐渐偏离离线静态数据集。由于 DPO 缺乏在线探索和即时奖励机制，策略网络可能会漂移到离线数据从未覆盖的高概率区域（未见样本分布，OOD）。在这些区域，模型的生成行为缺乏有效约束，可能出现语句重复、幻觉骤增或语法退化。

### 2. 文本长度偏置（Length Bias）

整个回复序列的对数似然是逐 token 条件概率的累加：

$$\log \pi(y|x) = \sum_{t=1}^{T} \log \pi(y_t | x, y_{<t})$$

序列长度 $T$ 越长，对数似然通常越呈负向绝对值累积。如果偏好数据集在构建时存在系统性偏见（例如偏好回答 $y_w$ 普遍比落选回答 $y_l$ 详尽或冗长），DPO 模型容易把“长度”作为区分胜负的隐式特征，倾向于通过增加生成长度来获取虚高的隐式奖励；另一方面，为了拉大差值，模型有时并不是在提高胜出项的合理性，而是在将落选回答中某些普通词元的条件概率不合理地压向负无穷。

### 3. 数据噪声无法获得平滑泛化

在经典 RLHF 中，独立训练的奖励模型是一个回归/分类网络，能够在连续参数空间中对多轮标注数据的冲突起到一定的平滑与泛化作用。而在 DPO 中，生成模型自身直接拟合离线偏好标签。如果标注数据中存在互相矛盾的人类偏好（例如标注员 A 偏好极简回答，标注员 B 偏好细致拆解），DPO 的策略模型容易在相互冲突的梯度方向间拉扯，进而损伤基座模型的常识与泛化能力。

---

## 7. 面试高频追问与实战解答

![在线探索与离线 DPO 在推理任务中的差异对比](/images/content-rebuild/f4cccb9faea70f5b/fig-6.png)

### 追问 1：DPO 能否完全取代 PPO？在数学推导和代码任务中两者有何差异？

**口述要点**：**不能完全取代**。
- 在开放域闲聊、文本润色、通用问答及安全性对齐等以“风格偏好”或“主观规范”为主的任务中，DPO 凭借训练稳定和工程简单，已成为主流方案。
- 但在数学计算（如 GSM8K、MATH）与代码合成等**具备明确客观判据（Ground Truth）、依赖多步复杂推导链**的场景下，支持在线探索的强化学习（如 PPO、RLOO、GRPO）依然表现更优。
  - **在线 RL**：模型在训练时自主采样生成新路径，配合单元测试、编译器或规则验证器（Rule-based Verifier）给出客观真值奖励。即使模型生成了一条离线训练集未收录的全新解题逻辑，只要验证正确，该探索路径便能获得强化；
  - **离线 DPO**：仅能在固定的正负样本对之间拉大似然差，无法在离线数据覆盖之外进行主动试错，难以探索并学习复杂的解题链条。

### 追问 2：超参数 $\beta$ 的物理含义是什么？调大或调小对模型有什么具体影响？

**口述要点**：$\beta$ 来自 KL 散度惩罚系数，对应**正则化温度**：
- **$\beta$ 设置过大**：对模型偏离参考模型 $\pi_{\text{ref}}$ 的惩罚极重，隐式奖励数值被缩放放大，策略更新非常保守，模型学习人类偏好信号的速度过慢，对齐效果不明显；
- **$\beta$ 设置过小**：KL 惩罚较弱，模型被允许大幅偏离基座模型的初始分布，容易激进地单向压低落选样本的似然概率，引发过拟合、语言退化（如无意义符号循环、输出乱码）或指令遵循能力丢失。
- **工程实践**：在大多数 7B 到 70B 模型训练中，$\beta$ 通常根据微调目标和学习率在 $0.01$ 到 $0.2$ 之间调试。

### 追问 3：为什么 DPO 训练后经常出现模型过度敏感、拒答率增加的现象？如何缓解？

**口述要点**：这主要由偏好数据构建偏置和离线似然机制共同引发：
- **成因**：在构建安全性偏好数据时，胜出回答 $y_w$ 往往是标准模板化的拒绝用语（如“抱歉，作为一个人工智能助手，我无法提供该内容……”），其文本结构高度同质；而落选回答 $y_l$ 通常包含详细的实质性文本。由于模板回复特征非常明显，DPO 会快速学习到“遇到可能存在敏感词的提示词直接输出拒绝模板”这一捷径，导致安全判定边界被过度放宽，甚至对无害的日常询问频繁拒答。
- **缓解方案**：
  1. 丰富安全样本的多样性，改写固定模板，增加安全答复并附带建设性建议的样本比例；
  2. 在损失函数中混入适量的通用有监督微调（SFT）正则化损失，约束模型在通用非安全任务上的生成概率分布；
  3. 调整安全样本在整体偏好数据集中的配比，防止单一目标反客为主。

---

## 8. 学习检验与自测清单

完成本篇学习后，可尝试在草稿纸上回答以下自测问题：

- [ ] 能否不看推导，独立写出从 $\min_\pi \mathbb{D}_{\text{KL}}(\pi \parallel \pi^*)$ 到 $r(x, y)$ 闭式解的变形过程？
- [ ] 配分函数 $Z(x)$ 是因为什么原因在带入 Bradley-Terry 偏好概率公式后相消的？这一相消成立的数学前提是什么？
- [ ] 当模型在离线数据上训练至过拟合时，梯度中的自适应权重系数 $\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))$ 会发生什么变化？

如需进一步探究把有监督指令微调与偏好对齐合并为单阶段优化的演进思路，可阅读后续章节：[/interview/llm-interview/orpo-alignment.md](/interview/llm-interview/orpo-alignment.md)；如需复习对齐在整个大语言模型全流程训练中的全局位置，可参考 [/interview/llm-interview/training-pipeline.md](/interview/llm-interview/training-pipeline.md)。
