---
title: 反向传播与 AdamW 在大模型训练中如何更新参数？
icon: mdi:tune-vertical
tag:
- 训练原理
- 大模型100讲
- 预训练
description: 详解大模型训练中的链式法则反向求导、残差梯度直通、全局梯度裁剪，以及 AdamW 解耦权重衰减与学习率调度的具体计算过程。
---

在上一讲[/llm-100/training/loss-calculation.md](/llm-100/training/loss-calculation.md)中，我们推导了自回归语言模型如何通过交叉熵损失（Cross-Entropy Loss）度量预测分布与真实下一个 Token 的差异，并输出一个标量损失值 $\mathcal{L}$。

得到标量损失后，训练的核心目标是调整模型内部的权重矩阵，使损失逐步下降。在大模型预训练体系中，这一过程主要由两个步骤配合完成：**反向传播（Backpropagation）**利用微积分的链式法则，计算出损失关于每个可学习参数的梯度；**优化器（以 AdamW 为主流）**则结合梯度、历史动量和学习率调度曲线，计算出各参数具体的更新幅度。

本讲通过一个贯穿前向与反向的张量算例，梳理 Transformer 中梯度的反传路径、梯度裁剪的几何控制，以及 AdamW 为什么要把权重衰减从自适应梯度中解耦出来。

![讲解员在工作台前解析反向传播链式法则与AdamW优化器更新循环](/images/content-rebuild/ad9265ae499a0b8f/cover.png)

---

### 前向激活缓存与反向求导计算流

大模型训练的前向传播不仅是为了输出预测概率和损失，还在显存中留存了反向求导所必需的中间变量（激活值，Activation）。

#### 线性层的正反向计算

以 Transformer 中最常见的线性变换（如注意力投影矩阵、前馈网络 FFN）为例：

$$
Y = XW + b
$$

设：
* 输入激活矩阵 $X \in \mathbb{R}^{B \times d_{in}}$（$B$ 为当前批次的有效 Token 总数，$d_{in}$ 为输入维度）；
* 权重参数矩阵 $W \in \mathbb{R}^{d_{in} \times d_{out}}$，偏置向量 $b \in \mathbb{R}^{1 \times d_{out}}$；
* 输出矩阵 $Y \in \mathbb{R}^{B \times d_{out}}$。

在反向传播阶段，假设计算图上一层已经传回损失 $\mathcal{L}$ 关于输出 $Y$ 的梯度 $\frac{\partial \mathcal{L}}{\partial Y} \in \mathbb{R}^{B \times d_{out}}$。依据矩阵求导的链式法则，损失关于权重 $W$、偏置 $b$ 以及前序输入 $X$ 的梯度分别为：

$$
\frac{\partial \mathcal{L}}{\partial W} = X^T \frac{\partial \mathcal{L}}{\partial Y}, \quad \frac{\partial \mathcal{L}}{\partial b} = \sum_{i=1}^B \left(\frac{\partial \mathcal{L}}{\partial Y}\right)_{i,:}, \quad \frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} W^T
$$

从公式中可以看到明确的数据依赖：
1. **更新当前层权重**：计算 $\frac{\partial \mathcal{L}}{\partial W}$ 必须使用前向传播时传入该层的输入激活值 $X$。
2. **将梯度回传给更浅层**：计算 $\frac{\partial \mathcal{L}}{\partial X}$ 必须使用当前层的权重矩阵 $W$。

这也是大模型训练显存消耗远高于纯推理的原因：推理时只需要保留当前层的中间结果，算完即可丢弃；而训练时，每个 Transformer 层的自注意力矩阵、线性层输入以及激活函数前后的张量，都必须作为激活缓存留在显存中，直到反向传播完成对该层的梯度计算后才能释放。

![讲解员演示前向激活值存储与反向梯度反向流动的链式法则图解](/images/content-rebuild/ad9265ae499a0b8f/fig-1.png)

#### 残差连接的梯度直通机制

现代 Transformer 通常包含几十甚至上百层网络结构。如果仅采用普通网络层的串联，反向梯度将由一系列雅可比矩阵（Jacobian Matrix）连续相乘构成，极易随层数加深而剧烈衰减。残差连接改变了梯度的传递形式。对于任意子层（如注意力子层或前馈子层）：

$$
Y = X + F(X)
$$

反向传播对输入 $X$ 求偏导时，算子展开为：

$$
\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} \cdot \left( I + \frac{\partial F(X)}{\partial X} \right) = \frac{\partial \mathcal{L}}{\partial Y} + \frac{\partial \mathcal{L}}{\partial Y} \frac{\partial F(X)}{\partial X}
$$

式中 $I$ 为单位矩阵。第一项 $\frac{\partial \mathcal{L}}{\partial Y}$ 表明，深层传回的梯度信号包含一条不需要经过子层非线性变换 $F(X)$ 衰减的直通路径。这种加法流动结构在很大程度上缓解了深层网络的梯度消失问题。

---

### 梯度数值不稳定与全局梯度裁剪

在数千亿参数规模的深层网络中，各层权重的微小扰动与长上下文注意力分布的极端值，仍可能导致梯度数值出现剧烈起伏。

* **梯度消失**：底层参数梯度趋近于 0，更新停滞。现代架构通常结合 Pre-LayerNorm 与规范初始化（如截断正态分布）来抑制该问题。
* **梯度爆炸**：训练初期或遇到异常样本时，反向传播经过连续大矩阵乘法，梯度数值可能瞬间暴增。这会导致参数迈出过大的更新步长，落入损失曲面较差的区域，甚至产生浮点数溢出变为 `NaN`。

<ReadMoreLock />

#### 基于全局范数的梯度裁剪（Gradient Clipping）

为防止单步梯度暴增破坏已有权重，工业界主流做法是在反向传播完成、优化器更新参数前，执行全局梯度裁剪。

具体计算步骤如下：
1. 将模型全部待更新参数的梯度张量展平，计算联合组成的全局 $L_2$ 范数：
   $$
   \|g\|_2 = \sqrt{\sum_{i=1}^{P} \sum_{j} (g_{i,j})^2}
   $$
   其中 $P$ 为参数张量总数，$g_{i,j}$ 为第 $i$ 个参数张量中的标量梯度元素。
2. 设定裁剪阈值 $\tau$（预训练中常设为 $1.0$）。
3. 当全局范数 $\|g\|_2 > \tau$ 时，对所有参数的梯度进行等比例缩小；若未超过阈值，则保持原状：
   $$
   g_i \leftarrow g_i \times \frac{\tau}{\max(\tau, \|g\|_2)}
   $$

这一操作的几何特征在于**保持方向、等比缩放**：它并没有改变高维空间中各参数梯度合成的向量方向，仅在模长超出安全阈值时将其截断到 $\tau$，从而兼顾了梯度指向与数值稳定性。

![讲解员对比全局梯度范数超出阈值前后的向量模长收缩机制](/images/content-rebuild/ad9265ae499a0b8f/fig-2.png)

---

### AdamW 优化器的状态递推与自适应步长

经典随机梯度下降（SGD）在 Transformer 训练中收敛较慢。由于不同参数所在的曲率差异极大，统一的学习率很容易在陡峭方向反复震荡，而在平缓方向推进缓慢。

AdamW（Adaptive Moment Estimation with Decoupled Weight Decay）为每个参数维护独立的状态统计量：一阶动量估计（平滑梯度的历史方向）与二阶方差估计（记录各参数梯度的变动幅度）。

#### 递推计算公式

在训练步 $t$，设某个参数对应的标量梯度为 $g_t$。AdamW 的完整计算流程如下：

**1. 一阶动量更新（指数移动平均 EMA）**
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$
超参数 $\beta_1$ 常取 $0.9$。$m_t$ 平滑了批次采样带来的随机波动，起到物理动量的作用。

**2. 二阶方差更新（平方梯度的 EMA）**
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$
超参数 $\beta_2$ 常见设置为 $0.95$ 或 $0.999$（大模型预训练常取 $0.95$）。$v_t$ 记录了该参数梯度的离散程度。

**3. 冷启动偏差修正**
训练开始时状态张量初始化为零（$m_0 = 0, v_0 = 0$）。在初始几步中，移动平均值会严重向 0 偏移。展开递推式：
$$
m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i
$$
若假设梯度的期望保持稳定，对其两端取期望，利用等比数列求和公式 $\sum_{i=1}^t \beta_1^{t-i} = \frac{1 - \beta_1^t}{1 - \beta_1}$，可得 $\mathbb{E}[m_t] = (1 - \beta_1^t) \mathbb{E}[g_t]$。因此，为获得无偏估计，需除以修正系数：
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$
随着步数 $t$ 增大，$\beta_1^t$ 与 $\beta_2^t$ 迅速衰减到 0，分母趋近于 1，修正项自然淡出。

**4. 自适应更新方向**
经过归一化的自适应梯度方向为：
$$
u_t = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$
其中 $\epsilon$ 是微小常数（如 $10^{-8}$ 或 $10^{-6}$），用于防止除以零。对于历史梯度模长较大的参数，分母大，步长被压制；对于历史梯度较小的参数，分母小，步长得到相对补偿。

![讲解员拆解AdamW中一阶动量平滑、二阶方差均方根与初始步偏差修正](/images/content-rebuild/ad9265ae499a0b8f/fig-3.png)

---

### 解耦权重衰减与传统 L2 正则化的计算差异

权重衰减（Weight Decay）用于抑制权重过度膨胀以降低过拟合风险。但在自适应学习率优化器下，传统的“L2 正则化”与真正的“权重衰减”产生了计算分歧。

#### 传统 L2 正则化在 Adam 中的失真机理

经典理论中，L2 正则化通过在损失函数中增加参数模长惩罚项：
$$
\mathcal{L}_{total}(\theta) = \mathcal{L}_{task}(\theta) + \frac{\lambda}{2} \|\theta\|^2
$$
对参数求导后，进入优化器的综合梯度为：
$$
\tilde{g}_t = \nabla_\theta \mathcal{L}_{task}(\theta_{t-1}) + \lambda \theta_{t-1} = g_t + \lambda \theta_{t-1}
$$

在标准 SGD 中，将 $\tilde{g}_t$ 代入后得到：
$$
\theta_t = \theta_{t-1} - \eta \tilde{g}_t = (1 - \eta \lambda) \theta_{t-1} - \eta g_t
$$
每个参数都按比例 $(1 - \eta \lambda)$ 发生线性衰减，这与直接对权重做模长惩罚的直觉完全一致。

然而，当直接把包含 $\lambda \theta$ 的混合梯度 $\tilde{g}_t$ 输入 Adam 时，二阶方差统计量变为：
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (g_t + \lambda \theta_{t-1})^2
$$
更新步变为：
$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{\tilde{m}}_t}{\sqrt{\hat{\tilde{v}}_t} + \epsilon}
$$
此时，原本用来惩罚大权重的 $\lambda \theta_{t-1}$ 被除以了自适应分母 $\sqrt{\hat{\tilde{v}}_t}$：
* **大梯度参数**：历史梯度大，分母 $\sqrt{\hat{\tilde{v}}_t}$ 很大，导致该参数受到的权重衰减惩罚被大幅缩小；
* **小梯度参数**：历史梯度接近 0，分母很小，受到的权重衰减惩罚反而被剧烈放大。
这导致惩罚力度与参数更新活跃度倒挂，背离了正则化的初衷。

#### AdamW 的解耦实现

AdamW 将权重衰减与梯度自适应更新完全解耦开来。优化器的状态估计只依据纯任务梯度 $g_t$，权重衰减项则独立地以线性系数作用于参数本身：

$$
\theta_t = \theta_{t-1} - \eta_t \lambda \theta_{t-1} - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

如此一来，所有参数都按与历史梯度大小无关的相对速率 $\eta_t \lambda$ 进行模长收缩，保证了正则化效果在所有参数间的一致性。

![讲解员通过分流路径对比经典L2正则化与AdamW解耦权重衰减的计算路径](/images/content-rebuild/ad9265ae499a0b8f/fig-4.png)

#### 简化数值算例对比

为了直观展示两者的差异，我们设定一个稳态简化算例：设偏差修正项已趋近于 1，忽略动量阻尼（即单步直接观察更新方向），学习率 $\eta = 0.1$，衰减系数 $\lambda = 0.1$。当前有两个权重值相同的参数 $W_A = 2.0$ 和 $W_B = 2.0$：
* 参数 A 梯度大：$g_A = 10.0$；
* 参数 B 梯度小：$g_B = 0.01$。

| 算法机制 | 参数 A ($g_A = 10.0, W_A=2.0$) 的衰减部分贡献 | 参数 B ($g_B = 0.01, W_B=2.0$) 的衰减部分贡献 | 衰减分配特征 |
| :--- | :--- | :--- | :--- |
| **Adam + L2 正则** | 混合梯度 $10.0 + 0.2 = 10.2$，分母约为 $10.2$。衰减贡献约 $\frac{0.1 \times 0.2}{10.2} \approx \mathbf{0.00196}$ | 混合梯度 $0.01 + 0.2 = 0.21$，分母约为 $0.21$。衰减贡献约 $\frac{0.1 \times 0.2}{0.21} \approx \mathbf{0.09524}$ | **失真**：小梯度参数受到的惩罚反而比大梯度参数大近 50 倍 |
| **AdamW 解耦衰减** | 独立衰减项 $\eta \lambda W_A = 0.1 \times 0.1 \times 2.0 = \mathbf{0.02}$ | 独立衰减项 $\eta \lambda W_B = 0.1 \times 0.1 \times 2.0 = \mathbf{0.02}$ | **一致**：所有参数严格以相同的比例系数进行模长收缩 |

---

### 学习率调度：预热（Warmup）与余弦退火（Cosine Decay）

在大模型训练中，一般不采用固定学习率，而是配合动态学习率调度策略。最广泛使用的是**线性预热配合余弦退火**。

![讲解员结合学习率时序图解析Warmup预热与Cosine退火衰减曲线](/images/content-rebuild/ad9265ae499a0b8f/fig-5.png)

#### 为什么需要预热（Warmup）阶段？

在训练刚开始的阶段，模型所有参数均处于随机初始化状态附近，预测误差很大，输出梯度波动剧烈。更重要的是，AdamW 的二阶方差估计 $v_t$ 需要一定的步数来积累历史梯度的二阶统计量。

如果在最初几步就直接使用峰值学习率 $\eta_{max}$：
1. 二阶方差 $v_t$ 尚未形成可靠估计，部分维度的分母可能异常小，乘上大学习率容易导致单步更新量过大，破坏合理的参数分布；
2. 早期梯度方向随机性强，过大的更新步长容易使训练发散。

线性预热在训练前 $T_{warmup}$ 步（通常占总步数的 $1\% \sim 3\%$）内，将学习率从接近 0 线性拉升至峰值 $\eta_{max}$：

$$
\eta_t = \eta_{max} \times \frac{t}{T_{warmup}}, \quad t \le T_{warmup}
$$

这一阶段为一阶动量与二阶方差提供了积累统计量的缓冲期，保证训练平稳启动。

#### 余弦退火（Cosine Decay）阶段

在达到峰值后，学习率按照余弦函数逐渐下降至预设的最小学习率 $\eta_{min}$（通常取 $\eta_{max} \times 0.1$）：

$$
\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min}) \left( 1 + \cos\left( \frac{t - T_{warmup}}{T_{total} - T_{warmup}} \pi \right) \right), \quad t > T_{warmup}
$$

余弦衰减在训练中期的下降坡度相对平缓，维持了模型探索参数空间的能力；在训练尾期，学习率以极小斜率缓慢逼近 $\eta_{min}$，有助于参数稳定收敛在平缓的低损失区域，提升泛化表现。

---

### 动手实验：AdamW 与梯度裁剪的实现验证

以下脚本实现了一个可运行的前向计算、反向求导、全局梯度裁剪以及 AdamW 参数更新过程。我们可以清晰追踪裁剪前后的范数变化，以及解耦权重衰减的执行逻辑：

```python
import torch
import math

def manual_adamw_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.1
):
    """
    单参数张量的 AdamW 更新逻辑
    """
    # 1. 独立执行解耦权重衰减: param = param * (1 - lr * weight_decay)
    param.data.mul_(1.0 - lr * weight_decay)

    # 2. 一阶动量更新: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

    # 3. 二阶方差更新: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    # 4. 偏差修正计算
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step

    # 5. 计算自适应步长分母
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
    step_size = lr / bias_correction1

    # 6. 自适应更新权重
    param.data.addcdiv_(exp_avg, denom, value=-step_size)

# --- 实验验证 ---
torch.manual_seed(42)

# 构建两组可训练权重
w1 = torch.tensor([[2.0, -1.0], [1.5, 3.0]], requires_grad=True)
w2 = torch.tensor([[0.5], [-2.0]], requires_grad=True)

# 模拟前向计算与标量损失
x = torch.tensor([[1.0, 2.0]])
h = torch.matmul(x, w1)
loss = torch.sum(torch.matmul(h, w2) ** 2)

# 1. 执行反向传播，生成参数梯度
loss.backward()

# 2. 计算并执行全局梯度裁剪 (阈值 tau = 1.0)
params = [w1, w2]
grads = [p.grad for p in params]
global_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
clip_threshold = 1.0

print(f"[裁剪前] 全局梯度范数: {global_norm.item():.4f}")
if global_norm > clip_threshold:
    scale = clip_threshold / (global_norm + 1e-6)
    for g in grads:
        g.mul_(scale)

clipped_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
print(f"[裁剪后] 全局梯度范数: {clipped_norm.item():.4f}")

# 3. 初始化 AdamW 动量缓存并执行单步更新
m1 = torch.zeros_like(w1)
v1 = torch.zeros_like(w1)

print(f"[更新前] w1[0,0] 权重值: {w1[0, 0].item():.4f}")
manual_adamw_step(w1, w1.grad, m1, v1, step=1, lr=0.01, weight_decay=0.05)
print(f"[更新后] w1[0,0] 权重值: {w1[0, 0].item():.4f}")
```

运行上述代码后可以观察到两点验收结果：
1. 裁剪前范数超出 1.0 时，所有参数的梯度被等比例按系数缩放，裁剪后的全局范数严格被限制在阈值范围内；
2. 在 `manual_adamw_step` 内部，解耦权重衰减直接按 `1.0 - lr * weight_decay` 的比例压缩权重当前值，动量项再在此基础上执行平滑更新。

![讲解员展示从梯度反向传播到AdamW参数更新的单步工程验收闭环](/images/content-rebuild/ad9265ae499a0b8f/fig-6.png)

---

### 总结与后向延伸

大模型训练由链式法则反向求导与优化器协同推进：
1. **前向与反向**：前向传播缓存的输入张量是反向矩阵求导的必要依赖，残差连接提供了不经子层衰减的梯度直流通路；
2. **数值稳定**：全局梯度裁剪在更新前抑制梯度爆炸，并在多维空间中保持更新向量的方向不变；
3. **自适应更新**：AdamW 通过一阶动量减少震荡、二阶方差均衡各坐标轴步长、偏差修正消除冷启动滞后，并借由解耦权重衰减确保正则化不被方差分母扭曲；
4. **学习率调度**：预热阶段给动量和方差估计留出建立缓冲的时间，余弦退火阶段则引导参数平滑收敛。

理解了梯度生成与优化器单步更新后，一个工程问题自然浮现：既然已经有了稳定且高效的参数更新算法，为什么训练基座模型依然需要万亿级别的海量 Token 数据？在下一讲[/llm-100/training/why-massive-data.md](/llm-100/training/why-massive-data.md)中，我们将从参数容量、泛化边界与 Scaling Law 的角度展开探讨。
