---
title: LayerNorm 和 RMSNorm 有什么区别？为什么大模型更偏爱 Pre-LN？
icon: mdi:chart-line
tag:
- 大模型面试
- LayerNorm
- RMSNorm
description: 解析 LayerNorm 与 RMSNorm 的数学简化与显存访存差异，对比 Pre-LN 与 Post-LN 的反向传播梯度特性及深层稀释权衡，给出面试口述与推导。
---

![LayerNorm、Pre-LN/Post-LN 拓扑与 RMSNorm 优化机制导读封面](/images/content-rebuild/41336da0546f82c6/cover.png)

在现代大语言模型的网络结构中，归一化层（Normalization）如何计算、放在哪里，直接决定了模型能不能稳健训练，以及硬件执行时能跑多快。从早期 Transformer 的 Post-LN 演进到当下的 Pre-LN，再从标准 LayerNorm 普遍切换为 RMSNorm，背后的原因可以概括为两点：**让反向传播的梯度在深层网络中畅通传递**，以及**减少显存搬运以提升计算速度**。

## 30秒口述结论

在面试中被问到“LayerNorm 与 RMSNorm 有什么区别？为什么现代大模型普遍采用 Pre-LN 结构？”时，可以按以下三点回答：

1. **RMSNorm 的优化核心**：RMSNorm 移除了 LayerNorm 中的“减均值（去中心化）”步骤以及可学习平移偏置 $\beta$，仅基于均方根（RMS）进行缩放。研究表明，稳定梯度的关键在于特征的“缩放不变性”而非平移不变性。去掉均值计算后，算子少了一次全局规约访存，显存读写更少、速度更快，同时在大模型上的收敛效果与 LayerNorm 相当。
2. **Pre-LN 与 Post-LN 的结构区别**：Post-LN 将归一化放在残差相加之后（$x_{l+1} = \text{LN}(x_l + \mathcal{F}(x_l))$），残差主干会被归一化层阻断；Pre-LN 将归一化放在子层变换（Attention 或 FFN）的分支内部（$x_{l+1} = x_l + \mathcal{F}(\text{LN}(x_l))$），残差连接构成了贯穿全网的加法主干。
3. **训练稳定性的差异**：反向传播时，Pre-LN 的残差主干包含一条直通的恒等项（$+ I$），梯度可以直接无阻碍地传回底层，因此对初始学习率和 Warmup 的敏感度显著降低；而 Post-LN 反向梯度必须连续穿透每一层的归一化导数矩阵，容易在网络初始化初期导致深层与浅层梯度极度失衡，极易发散。

![30秒核心结论：RMSNorm 算子精简与 Pre-LN 梯度直通路径](/images/content-rebuild/41336da0546f82c6/fig-1.png)

---

## Pre-LN 与 Post-LN 的结构对比

在 Transformer 的标准 Block 中，每个子层都包含两部分：核心变换函数 $\mathcal{F}(x)$（多头注意力 MHA 或前馈网络 FFN）与残差连接。

关于归一化层与注意力机制的基础原理，可先参考 [Transformer 为什么采用 Layer Normalization 而不是 Batch Normalization](/interview/llm-interview/layer-normalization.md) 与 [为什么 Transformer 采用多头注意力机制？](/interview/llm-interview/why-multi-head-attention.md)。

### 1. Post-LN：残差后归一化

原始 Transformer（Vaswani et al., 2017）使用的是 Post-LN 结构。计算公式为：

$$
x_{l+1} = \text{LN}\big(x_l + \mathcal{F}_l(x_l)\big)
$$

输入向量 $x_l$ 输入子层计算得到增量，两者相加后，**必须先通过归一化层处理**，其输出才能作为下一层的输入 $x_{l+1}$。

### 2. Pre-LN：残差前归一化

现代开源自回归模型（如 LLaMA、GPT-2/3、Qwen）绝大多数采用 Pre-LN 结构。计算公式为：

$$
x_{l+1} = x_l + \mathcal{F}_l\big(\text{LN}(x_l)\big)
$$

输入向量 $x_l$ 先经过归一化，再送入子层 $\mathcal{F}_l$ 计算。子层输出的结果直接加到未经归一化的主干输入 $x_l$ 上。主干连接始终是纯粹的累加路径。

![Pre-LN 与 Post-LN 数据流拓扑对比](/images/content-rebuild/41336da0546f82c6/fig-2.png)

<ReadMoreLock />

---

## 梯度反向传播推导：为什么 Pre-LN 训练更稳定？

位置的微调为何对数十层乃至上百层的模型训练稳定性产生决定性影响？我们可以通过反向传播的链式法则来观察。

### 1. Post-LN 的链式相乘问题

设中间相加变量为 $z_l = x_l + \mathcal{F}_l(x_l)$，则 Post-LN 的状态传递为 $x_{l+1} = \text{LN}(z_l)$。当损失函数 $\mathcal{L}$ 对 $x_l$ 求导时：

$$
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_{l+1}} \cdot \frac{\partial \text{LN}(z_l)}{\partial z_l} \cdot \left( I + \frac{\partial \mathcal{F}_l(x_l)}{\partial x_l} \right)
$$

若跨越 $L$ 层回传到底层输入 $x_0$：

$$
\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \prod_{l=0}^{L-1} \left[ \frac{\partial \text{LN}(z_l)}{\partial z_l} \cdot \left( I + \frac{\partial \mathcal{F}_l(x_l)}{\partial x_l} \right) \right]
$$

* **问题根源**：梯度每向后回传一层，都必须连续乘以该层的归一化雅可比矩阵 $\frac{\partial \text{LN}(z_l)}{\partial z_l}$。在模型刚初始化时，各层参数随机，子层输出叠加使得深层特征方差增大，进而导致归一化导数分母尺度剧烈膨胀。
* **工程后果**：深层参数的梯度通常偏大，而越往浅层回传，梯度方差受连乘影响严重恶化。如果缺少极其谨慎的学习率预热（Warmup）与梯度截断，训练在最初几十步就会出现数值上溢（NaN）或参数剧烈振荡。

### 2. Pre-LN 的恒等直通路径

Pre-LN 的前向计算可以展开为跨层的直接累加：

$$
x_L = x_0 + \sum_{l=0}^{L-1} \mathcal{F}_l\big(\text{LN}(x_l)\big)
$$

对任意浅层状态 $x_l$ 求导时，直接应用求导加法法则：

$$
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \left( I + \sum_{j=l}^{L-1} \frac{\partial \mathcal{F}_j(\text{LN}(x_j))}{\partial x_l} \right)
$$

* **稳定性根源**：展开项中包含一个恒等矩阵 $I$。无论网络堆叠到多少层，最深处的损失梯度 $\frac{\partial \mathcal{L}}{\partial x_L}$ 都可以沿着残差主干直接、完整地注入到浅层 $x_l$。
* **工程后果**：浅层和深层的参数梯度模长保持在相对均衡的范围，训练初始阶段对学习率波动不敏感，使得超深 Transformer 能够稳定起步。

![反向传播路径对比：Post-LN 乘积链阻碍 vs Pre-LN 恒等直通](/images/content-rebuild/41336da0546f82c6/fig-3.png)

---

## RMSNorm 算子推导：为什么去中心化可以省略？

当 Pre-LN 确立为结构基础后，工程师们开始着手精简归一化层本身的计算开销。标准 LayerNorm 的计算如下：

$$
y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i
$$

其中均值与方差定义为：

$$
\mu = \frac{1}{d} \sum_{j=1}^d x_j, \quad \sigma^2 = \frac{1}{d} \sum_{j=1}^d (x_j - \mu)^2
$$

标准 LayerNorm 包含两个核心动作：
1. **去中心化（Zero-mean）**：减去输入向量的均值 $\mu$。
2. **缩放归一化（Scaling）**：除以标准差 $\sqrt{\sigma^2 + \epsilon}$。

### 1. 均值计算的冗余性

Zhang & Sennrich (2019) 在提出 RMSNorm 时分析发现，LayerNorm 之所以能稳定前向特征和反向梯度，核心在于其具备**缩放不变性（Scaling Invariance）**：对于任意缩放标量 $\alpha > 0$，满足：

$$
\text{Norm}(\alpha x) = \text{Norm}(x)
$$

缩放不变性约束了特征向量的尺度，防止在前向传播中逐层发散。然而，“减去均值 $\mu$”带来的平移不变性对梯度的平稳流动贡献极小。

### 2. RMSNorm 计算公式

RMSNorm（Root Mean Square Layer Normalization）直接去掉了“减均值”步骤，也不再保留可学习的偏置参数 $\beta$，仅用均方根（RMS）进行尺度规范：

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{j=1}^d x_j^2 + \epsilon}
$$

$$
\bar{x}_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i
$$

其中 $\gamma \in \mathbb{R}^d$ 是保留的可学习缩放向量，$\epsilon$ 为防止除以零的微小常数。

![LayerNorm 与 RMSNorm 硬件执行访存与步骤对比](/images/content-rebuild/41336da0546f82c6/fig-4.png)

### 3. 硬件执行与显存带宽分析

在现代 GPU 上执行 Transformer 时，归一化算子属于典型的**内存受限型（Memory-bound）任务**。它的瓶颈不是浮点计算次数（FLOPs），而是高带宽显存（HBM）与片上缓存（SRAM）之间的数据搬运：

* **标准 LayerNorm**：传统实现需要两次规约（Reduction）操作。第一遍遍历求和计算均值 $\mu$；拿到 $\mu$ 后，第二遍遍历求 $(x_i - \mu)^2$ 的累加和以计算方差 $\sigma^2$。这会产生中间结果的读写和线程同步开销。
* **RMSNorm**：仅需一次规约操作即可完成平方和累加，紧接着求开方完成特征缩放。中间规约步骤减少，同时省去了偏置参数 $\beta$ 及其反向梯度的显存存储。

在优化良好的 GPU Kernel（如 Flash-Attention 库或 Triton 实现）中，RMSNorm 算子执行耗时相比标准 LayerNorm 可降低约 10% 到 50%，在千亿模型的多层叠加下，能直接转化为训练和推理吞吐的提升。

---

## 逐步算例对比：LayerNorm vs RMSNorm

为直观理解差异，我们设隐藏维度 $d = 4$，输入向量为：

$$
x = [2.0, 4.0, 6.0, 8.0], \quad \epsilon = 10^{-5}
$$

假设缩放参数初始化为 $\gamma = [1.0, 1.0, 1.0, 1.0]$，偏置初始化为 $\beta = [0.0, 0.0, 0.0, 0.0]$。

### 1. 标准 LayerNorm 计算步骤

1. **计算均值**：
   $$\mu = \frac{2.0 + 4.0 + 6.0 + 8.0}{4} = 5.0$$
2. **计算中心化向量**：
   $$x - \mu = [2.0 - 5.0, 4.0 - 5.0, 6.0 - 5.0, 8.0 - 5.0] = [-3.0, -1.0, 1.0, 3.0]$$
3. **计算方差**：
   $$\sigma^2 = \frac{(-3.0)^2 + (-1.0)^2 + 1.0^2 + 3.0^2}{4} = \frac{9 + 1 + 1 + 9}{4} = 5.0$$
4. **标准化并缩放**：
   $$
   y_i = \frac{x_i - \mu}{\sqrt{5.0 + 10^{-5}}} \approx \frac{[-3.0, -1.0, 1.0, 3.0]}{2.23607} \approx [-1.3416, -0.4472, 0.4472, 1.3416]
   $$

### 2. RMSNorm 计算步骤

1. **计算平方和与均方根 RMS**：
   $$\sum_{j=1}^4 x_j^2 = 2.0^2 + 4.0^2 + 6.0^2 + 8.0^2 = 4 + 16 + 36 + 64 = 120$$
   $$\text{RMS}(x) = \sqrt{\frac{120}{4} + 10^{-5}} = \sqrt{30.00001} \approx 5.47723$$
2. **缩放输出**：
   $$
   \bar{x}_i = \frac{x_i}{\text{RMS}(x)} \approx \frac{[2.0, 4.0, 6.0, 8.0]}{5.47723} \approx [0.3651, 0.7303, 1.0954, 1.4606]
   $$

从结果对比可以看出：RMSNorm 保持了输入向量的正值特性与相对比例，完全省略了差值计算，公式更简洁，计算链路更短。

![离散数值向量的归一化计算对比白板推导](/images/content-rebuild/41336da0546f82c6/fig-5.png)

---

## 原理解释：残差主干模长累加与深层贡献稀释

**追问**：“Pre-LN 既然保证了反向传播的数值稳定性，那为什么早期文献常提到，在充分调优并配合精心设计的学习率预热后，Post-LN 的模型收敛性能上限有时会略好于 Pre-LN？”

### 残差主干的模长增长与子层贡献稀释（Sub-layer Dilution）

这一现象的因果逻辑如下：

1. **Pre-LN 的模长增长**：在前向传播中，$x_L = x_0 + \sum_{l=0}^{L-1} \mathcal{F}_l(\text{LN}(x_l))$。由于每一层子层的输出直接加在主干上，残差主干向量 $x_l$ 的模长 $\|x_l\|$ 会随着网络层数 $l$ 加深而单调增长（理论上模长大致按 $\mathcal{O}(\sqrt{l})$ 尺度扩展）。
2. **子层输出的相对贡献衰减**：在进入子层分支前，输入向量必须先经过归一化，方差被重置为 1。因此每个子层分支计算出的增量输出向量 $\mathcal{F}_l$，其模长规模是相对恒定的。
3. **深层退化**：当网络极深（例如 64 层以上）时，浅层子层的增量在主干中所占比例尚可，而深层子层的增量相对于已经膨胀的残差主干模长，占比越来越低，逐渐退化为“微小扰动”：
   $$
   \frac{\|\mathcal{F}_l(\text{LN}(x_l))\|}{\|x_l\|} \to 0 \quad (l \to L)
   $$
   这导致深层模块的学习能力没有被充分激活，造成了网络深度的容量浪费。
4. **Post-LN 的平权更新**：在 Post-LN 中，每一层残差相加后都必须强制归一化（方差重置为 1），每一层的输出在最终表示中所占的相对比重保持一致，不存在深层被主干稀释的问题。因此，**在能够通过 DeepNorm 或精细学习率控制彻底解决 Post-LN 初始梯度不稳定的前提下**，Post-LN 往往能展现出略高的表示容量与收敛表现。

**工程权衡选择**：在超大规模模型的实际训练中，成千上万张 GPU 同时运行，训练发散带来的机器和时间成本不可承受。因此，“确定性保证训练不崩溃”远比“追求理论上限稍高但极易中途发散”重要。业界最终统一选择了以 Pre-LN + RMSNorm 为代表的高稳定性方案。

![深层子层贡献稀释问题机制解析](/images/content-rebuild/41336da0546f82c6/fig-6.png)

---

## 常见误区与辨析

| 常见错误认知 | 实际机制澄清 |
| :--- | :--- |
| **误区 1**：RMSNorm 去除均值操作是为了节省模型参数。 | RMSNorm 每个维度仅省去了一个标量偏置 $\beta$，参数量差异几乎可以忽略。其核心收益是**减少显存规约次数与访存量**，从而加快 Kernel 计算速度。 |
| **误区 2**：Pre-LN 解决了数值稳定问题，所以大模型训练不再需要学习率 Warmup。 | 虽然 Pre-LN 极大地改善了梯度的传递，但在训练刚刚开始时，注意力矩阵的权重分布尚未收敛，Adam 优化器的二阶动量统计量也不准确。因此，适度的 Warmup 依然是保障早期平稳收敛的标准工程实践。 |
| **误区 3**：Post-LN 训练困难是因为前向激活值数值溢出。 | 核心问题在**反向传播梯度**：由于归一化雅可比矩阵的连续级联相乘，未预热的网络中深层参数梯度极易剧烈震荡，浅层参数梯度尺度失衡，破坏了初始权重。 |
