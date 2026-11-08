---
title: 梯度消失与梯度爆炸：深层网络训练困难的成因与缓解方案
icon: mdi:alert-outline
tag:
- 深度学习
- 梯度问题
- 模型训练
description: 解析反向传播链式法则导致的梯度消失与爆炸机制，详解激活函数、权重初始化、残差连接与梯度裁剪等缓解方案与适用条件。
---

在深度学习早期，多层感知机的堆叠层数通常很难超过 4 到 5 层。一旦网络进一步加深，模型训练往往会陷入两种困境：靠近输入端的浅层参数几乎完全停止更新；或者训练过程中的数值突然变成 `NaN`（非数字）或 `Inf`（无穷大），导致模型彻底崩溃。这两种现象分别对应**梯度消失（Gradient Vanishing）**与**梯度爆炸（Gradient Exploding）**。

这并非代码实现错误，而是反向传播中链式求导连续相乘的直接结果。理解这两个问题及其解决方案，是掌握现代神经网络结构设计的核心起点。

![深层神经网络反向传播中梯度消失与梯度爆炸现象全景概览](/images/content-rebuild/48a40db24e0c0a73/cover.png)

---

## 贯穿例子：连乘效应的直观感受

在进入复杂的矩阵推导前，我们可以先看一个最直观的标量算例。

假设有一个极度简化的 30 层线性网络，输入为 $x$，输出为 $y$。前向计算只有单纯的乘法：

$$
y = w_{30} \times w_{29} \times \dots \times w_1 \times x
$$

根据微积分链式法则，损失 $\mathcal{L}$ 对第一层权重 $w_1$ 的导数包含了后续所有权重的乘积：

$$
\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left( \prod_{k=2}^{30} w_k \right) \cdot x
$$

此时考察这 29 次连乘：
- **情况 A（连乘项略小于 1）**：若每层权重均为 $0.8$，$0.8^{29} \approx 0.00155$。若权重为 $0.5$，$0.5^{29} \approx 1.86 \times 10^{-9}$。梯度衰减到了极小数值，第一层权重几乎得不到更新，这就是**梯度消失**。
- **情况 B（连乘项略大于 1）**：若每层权重均为 $1.2$，$1.2^{29} \approx 197.8$。若权重为 $1.5$，$1.5^{29} \approx 1.28 \times 10^5$。梯度被急剧放大，导致浅层参数剧烈跳动甚至溢出，这就是**梯度爆炸**。

当层数扩大到几十层甚至上百层时，任何偏离 1.0 的乘数都会在指数级放大下迅速走向两个极端。

---

## 机制解析：反向传播中的雅可比矩阵连乘

在多维全连接神经网络中，标量乘法变成了矩阵相乘。我们可以回顾[前向传播和反向传播](/fundamentals/dl-basics/forward-backward-propagation.md)的计算流程。

设网络共有 $L$ 层。对于第 $l$ 层（$l \in \{1, 2, \dots, L\}$），前向传播计算为：

$$
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}
$$
$$
a^{[l]} = \sigma(z^{[l]})
$$

其中 $a^{[0]} = x$ 为输入向量，$W^{[l]}$ 为权重矩阵，$b^{[l]}$ 为偏置，$\sigma(\cdot)$ 为激活函数，$z^{[l]}$ 为线性变换后的净输入。

定义第 $l$ 层的误差项为损失 $\mathcal{L}$ 对净输入 $z^{[l]}$ 的梯度：$\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial z^{[l]}}$。根据链式法则，相邻两层之间的误差传递公式为：

$$
\delta^{[l]} = \left( \frac{\partial z^{[l+1]}}{\partial z^{[l]}} \right)^T \delta^{[l+1]}
$$

其中，$\frac{\partial z^{[l+1]}}{\partial z^{[l]}}$ 为雅可比矩阵（Jacobian Matrix）。因为 $z^{[l+1]} = W^{[l+1]} \sigma(z^{[l]}) + b^{[l+1]}$，所以：

$$
\frac{\partial z^{[l+1]}}{\partial z^{[l]}} = W^{[l+1]} \operatorname{diag}\left( \sigma'(z^{[l]}) \right)
$$

将其转置后代入，得到反向传播递推公式：

$$
\delta^{[l]} = \operatorname{diag}\left( \sigma'(z^{[l]}) \right) (W^{[l+1]})^T \delta^{[l+1]}
$$

将该递推式从最顶层输出层 $L$ 一路展开至最浅层（第 1 层）：

$$
\delta^{[1]} = \left[ \prod_{k=1}^{L-1} \operatorname{diag}\left( \sigma'(z^{[k]}) \right) (W^{[k+1]})^T \right] \delta^{[L]}
$$

第 1 层的权重梯度为：

$$
\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \delta^{[1]} (a^{[0]})^T
$$

![反向传播中雅可比矩阵沿网络层数连续连乘的链式拓扑图](/images/content-rebuild/48a40db24e0c0a73/fig-1.png)

浅层梯度中包含了一个长达 $L-1$ 项的矩阵连乘链。连乘项由两个因素共同构成：**激活函数的导数矩阵**与**层间权重矩阵的转置**。

---

## 梯度消失：饱和激活与小权重的双重作用

当反向传播经过的矩阵连乘项尺度持续小于 1 时，浅层误差项 $\delta^{[1]}$ 的模长将以指数级速度萎缩至接近 0。

### 1. 饱和激活函数的导数限制

早期神经网络主要采用 Sigmoid 或 Tanh 激活函数（相关推导可参考[激活函数选型分析](/fundamentals/dl-basics/activation-functions.md)）。

Sigmoid 函数定义为 $\sigma(z) = \frac{1}{1 + e^{-z}}$，其导数形式为：

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

当 $z = 0$ 时，$\sigma'(z)$ 取得最大值 $0.25$；当 $|z|$ 增大时，导数迅速趋近于 0。这意味着对角矩阵中的每一个分量都有上界限制：

$$
0 < \sigma'(z) \le 0.25
$$

假设各层权重矩阵经过缩放后未放大信号，仅考虑激活函数的导数连乘：在 30 层的网络中，浅层接收到的衰减因子最大仅为 $0.25^{29} \approx 3.47 \times 10^{-18}$。虽然该数值尚未触及 FP32 的绝对表示下限（最小正正规数约为 $1.18 \times 10^{-38}$），但当常规学习率（如 $10^{-2}$）与其相乘后，单步更新量降至 $10^{-20}$ 量级；在 FP32 的有效尾数精度（机器精度约为 $1.19 \times 10^{-7}$）下，将其与量级通常在 $0.1 \sim 1.0$ 的权重相加时，更新量会被浮点舍入完全忽略（即浮点吸收），导致浅层参数数值实质停滞。若在半精度 FP16（最小次正规数约为 $5.96 \times 10^{-8}$）下，该量级更会直接下溢为绝对的 0。

### 2. 权重矩阵的奇异值抑制

即使激活函数不存在导数饱和区（例如导数恒为 1 的线性激活），若权重矩阵的最大奇异值持续小于 1，向量在连续通过矩阵变换后，模长仍将以几何级数递减并趋近于零。

两者叠加，直接导致深层网络只有最后几层能有效学到特征，而负责提取底层原始特征的前几层权重基本保持初始随机状态。

![Sigmoid导数上界与权重衰减导致浅层梯度趋零的衰减示意图](/images/content-rebuild/48a40db24e0c0a73/fig-2.png)

<ReadMoreLock />

---

## 梯度爆炸：大矩阵尺度与数值溢出

与梯度消失相反，如果矩阵连乘项的有效放大倍数持续大于 1，反向传播的梯度模长会呈指数级增长。

当每层权重矩阵的特征值模长或奇异值大于 1，误差向量在经过数十层连续放大后，浅层梯度模长可能达到难以估量的天文数字。

### 梯度爆炸带来的典型故障

1. **参数大幅震荡，错过最优解**：在梯度下降更新公式 $\theta_{t+1} = \theta_t - \eta g_t$ 中，巨大的梯度 $g_t$ 会使得参数单步位移极大，瞬间跳出当前的损失盆地，落入平坦区或损失极高的区域。
2. **浮点溢出（`Inf` / `NaN`）**：
   - FP32 可表示的最大正数值约为 $3.4 \times 10^{38}$。
   - FP16（半精度）的最大正数值仅为 $65504$。
   - 梯度连乘一旦超出硬件表示范围就会溢出为 `Inf`。随后的矩阵运算中若出现 `Inf - Inf` 或 `0 * Inf`，计算结果就会变成 `NaN`（非数字），导致整组参数报废且不可恢复。

![梯度模长指数级爆炸引发参数越过最优解及数值溢出的失效分析](/images/content-rebuild/48a40db24e0c0a73/fig-3.png)

---

## 缓解方案一：采用非饱和激活函数

解决梯度消失最直接的方法，是将具有饱和性质的激活函数替换为在正半轴导数不衰减的函数：

- **ReLU（Rectified Linear Unit）**：$f(z) = \max(0, z)$。在 $z > 0$ 时，其导数恒为 1。由于连乘项中的激活导数不再是小于 0.25 的衰减因子，反向传播可以有效穿透多层。
- **适用条件与边界**：当 $z < 0$ 时，ReLU 的导数恒为 0。若某个神经元输入长时间处于负半轴，该神经元将永久失去梯度（称为 Dying ReLU 现象）。在现代网络中，通常采用 LeakyReLU、ELU 或平滑变体 GELU 来缓解完全截断问题。

---

## 缓解方案二：科学的权重初始化（Xavier 与 He 初始化）

非饱和激活函数移除了导数衰减因子，但权重矩阵 $W^{[l]}$ 本身的尺度仍然可能引发方差发散或衰减。合理的初始化目标是：**在前向传播中维持各层激活值的方差稳定，在反向传播中维持各层梯度的方差稳定**。

### 1. Xavier / Glorot 初始化（适用于 Tanh、Sigmoid 等激活）

假设输入分量独立且均值为 0，权重独立且均值为 0。对于单神经元前向计算 $z = \sum_{j=1}^{n_{in}} W_j x_j$：

$$
\operatorname{Var}(z) = n_{in} \operatorname{Var}(W) \operatorname{Var}(x)
$$

为了让输出方差保持等于输入方差（$\operatorname{Var}(z) = \operatorname{Var}(x)$），需要让前向权重方差满足 $\operatorname{Var}(W) = \frac{1}{n_{in}}$。

同理，考察反向传播梯度流，为了避免梯度跨层传播时方差缩放，需要反向满足 $\operatorname{Var}(W) = \frac{1}{n_{out}}$。综合考虑二者，Xavier 初始化采用调和平均折中：

$$
\operatorname{Var}(W) = \frac{2}{n_{in} + n_{out}}
$$

### 2. He / Kaiming 初始化（适用于 ReLU 系列激活）

ReLU 将负半轴全部置 0。在对称分布假设下，前向传播平均会屏蔽掉一半的神经元输出，导致输出信号方差减半：

$$
\operatorname{Var}(a) = \frac{1}{2} \operatorname{Var}(z)
$$

若直接使用 Xavier 初始化，经过 20 层 ReLU 堆叠后，方差会衰减为原来的 $(0.5)^{20} \approx 10^{-6}$。为了补偿这一折半效应，何恺明等人推导出权重方差需要翻倍：

$$
\operatorname{Var}(W) = \frac{2}{n_{in}}
$$

采用高斯分布采样时，权重初始标准差应设为 $\sqrt{\frac{2}{n_{in}}}$。

![Xavier与He初始化方差保持机制与激活值分布维持对比图](/images/content-rebuild/48a40db24e0c0a73/fig-4.png)

以下脚本直观展示了在 20 层 ReLU 前向传播中，不合理的方差衰减与 He 初始化对方差的维持效果：

```python
import torch

torch.manual_seed(42)
layers = 20
dim = 256
x = torch.randn(100, dim)  # 输入方差约为 1.0

# 实验 1: 权重方差过小（未做补偿）
x_small = x.clone()
for _ in range(layers):
    w = torch.randn(dim, dim) * 0.01
    x_small = torch.relu(x_small @ w)
print(f"过小初始化 20 层后输出方差: {x_small.var().item():.6e}")

# 实验 2: He 初始化（维持方差稳定）
x_he = x.clone()
for _ in range(layers):
    w = torch.randn(dim, dim) * (2.0 / dim) ** 0.5
    x_he = torch.relu(x_he @ w)
print(f"He 初始化 20 层后输出方差:    {x_he.var().item():.4f}")
```

运行结果中，实验 1 的方差会彻底坍缩到 0，而实验 2 的输出方差能维持在 1.0 附近稳定振荡。

---

## 缓解方案三：残差连接——用加法直通打破乘法级联

初始化只在训练刚开始起作用。随着网络迭代推进，权重参数会发生偏移，单纯依靠初始化无法保障上百层深度的梯度传递。ResNet（残差网络）提出的**残差连接（Residual Connection）**从根本上改变了梯度的流动方式。

一个基础残差块的前向计算为：

$$
x_{l+1} = x_l + \mathcal{F}(x_l, \mathcal{W}_l)
$$

当连续多个残差块的跳跃分支都保持纯恒等映射时，从第 $l$ 层到第 $L$ 层可以展开为递归累加：

$$
x_L = x_l + \sum_{k=l}^{L-1} \mathcal{F}(x_k, \mathcal{W}_k)
$$

对该式求偏导，损失 $\mathcal{L}$ 对浅层特征 $x_l$ 的梯度为：

$$
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left( I + \frac{\partial}{\partial x_l} \sum_{k=l}^{L-1} \mathcal{F}(x_k, \mathcal{W}_k) \right)
$$

展开后即为：

$$
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} + \frac{\partial \mathcal{L}}{\partial x_L} \left( \frac{\partial}{\partial x_l} \sum_{k=l}^{L-1} \mathcal{F}(x_k, \mathcal{W}_k) \right)
$$

### 残差梯度流动的数学特性

1. **加法保底项**：公式第一项直接是 $\frac{\partial \mathcal{L}}{\partial x_L}$，它由单位矩阵 $I$ 保证。即便第二项权重连乘因为饱和或小权重而衰减至 0，深层的误差信号依然能够通过单位通道毫无损耗地直接加到浅层。
2. **打破全乘积约束**：浅层梯度完全消失的条件变成了括号内的矩阵导数恰好等于 $-I$。在随机初始化的连续参数空间中，这种概率几乎为零。

这一加法拓扑不仅是 ResNet 成功的关键，也是后续 [Transformer](/fundamentals/dl-basics/transformer.md) 等深层结构的核心组件。

![残差块结构中主干路径与恒等映射分流通路的反向传播梯度流图](/images/content-rebuild/48a40db24e0c0a73/fig-5.png)

---

## 缓解方案四：梯度范数裁剪——工程防爆安全气囊

针对梯度爆炸，架构设计和初始化能降低其概率，但在长序列模型（如 RNN、Transformer）训练过程中，异常数据或离群点仍可能瞬间激发出极大的局部梯度。工程上最稳健的防御手段是**梯度裁剪（Gradient Clipping）**。

### 1. 全局范数裁剪（Norm Clipping，主流方式）

将模型所有参数的梯度展平拼接为一个总向量 $\mathbf{g}$，计算其 $L_2$ 范数。若范数超过预设阈值 $C$，则对整体梯度进行等比例缩小：

$$
\tilde{\mathbf{g}} = \begin{cases}
\mathbf{g}, & \text{若 } \|\mathbf{g}\|_2 \le C \\[6pt]
\dfrac{C}{\|\mathbf{g}\|_2} \mathbf{g}, & \text{若 } \|\mathbf{g}\|_2 > C
\end{cases}
$$

范数裁剪的特性是**仅缩放梯度模长，不改变梯度的空间方向**。它保留了梯度指明的最优下降角度，仅仅限制了参数更新的最大单步位移。

### 2. 标量逐元素截断（Value Clipping）

直接限定梯度的最大最小值：$\tilde{g}_i = \max(v_{\min}, \min(g_i, v_{\max}))$。这种方式实现简单，但因为不同分量被独立截断，会改变整体梯度向量的方向角，可能干扰优化器的正常寻优路径。

PyTorch 官方内置了全局范数裁剪，通常在 `loss.backward()` 与 `optimizer.step()` 之间调用：

```python
import torch

# 假设已完成模型前向与损失计算
optimizer.zero_grad()
loss.backward()

# 计算所有参数梯度的全局 L2 范数，若超过 1.0 则等比缩放
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

![梯度范数裁剪前后向量等比例缩放保留更新方向几何示意图](/images/content-rebuild/48a40db24e0c0a73/fig-6.png)

---

## 技术方案综合对比与适用边界

深层网络训练的稳定性依赖多项技术的配合。下表归纳了各方案的核心机制与边界：

| 技术手段 | 主要应对现象 | 工作原理 | 适用边界与局限 |
| :--- | :--- | :--- | :--- |
| **非饱和激活（ReLU/GELU）** | 梯度消失 | 正半轴导数为 1，消除了导数连乘缩小的机制 | 负半轴导数为 0 可能导致神经元坏死，需搭配平滑变体 |
| **权重初始化（He/Xavier）** | 消失与爆炸 | 依据输入输出维度调整初始权重方差，保持信号传输稳定 | 仅在训练初始阶段有效，参数迭代后分布会产生偏移 |
| **残差连接（ResNet）** | 梯度消失 | 在连乘主干旁并联一条单位映射跳跃路径，提供梯度加法通路 | 要求输入与输出张量空间尺寸一致；改变了网络拓扑结构 |
| **梯度范数裁剪** | 梯度爆炸 | 当全局梯度模长超过阈值时进行等比例收缩 | 属于被动防护机制，无法解决梯度消失，阈值需经验配置 |
| **归一化层（BN / LN）** | 协变量偏移 | 强制层间输入保持固定的均值与方差（详见[归一化技术对比](/fundamentals/dl-basics/normalization.md)） | 引入额外计算开销，对 batch size 或序列维度有特定依赖 |

理解了梯度在计算图中的传播与控制机制后，如何在此基础上让模型平稳、快速地找到最优解？我们将在下一篇探讨[优化器算法机制](/fundamentals/dl-basics/optimizers.md)。
