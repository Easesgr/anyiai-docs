---
title: Transformer 中的残差连接如何解决梯度消失与网络退化？
icon: mdi:call-split
tag:
- Transformer
- 残差连接
- 梯度消失
description: 面试标准回答：从全导数公式推导残差直通项，对比普通网络与LSTM衰减机制，解析恒等扰动如何解决退化，详解Pre-LN与初始化方差对直通梯度的影响。
---

## 面试速答

面试中如果被问到“Transformer 为什么要用残差连接？它能缓解梯度消失吗？”，可以按以下结构口述回答：

1. **核心结论**：可以。残差连接（Residual Connection）把前向计算重构为 $x_{l+1} = x_l + \mathcal{F}(x_l)$。当对输入求全导数时，反向传播公式中会自然分解出一项常数单位矩阵 $I$。
2. **解决梯度消失**：在无残差的普通网络中，反向传播是多层雅可比矩阵的连续相乘，层数一深梯度就会呈指数级衰减；残差连接将连乘变成了“加法”结构，哪怕非线性子层 $\mathcal{F}$ 的梯度趋近于 0，顶层误差依然可以通过单位矩阵 $I$ 这条捷径无损传回底层。
3. **解决网络退化**：深层网络难以训练的另一个原因是普通权重层很难逼近恒等映射。残差连接把学习目标从拟合完整目标 $H(x)$ 变成了拟合残差扰动 $\mathcal{F}(x) \to 0$。优化器天然容易把参数压向 0，这让深层网络在最差情况下也能无损退化为浅层模型，保证性能不随深度增加而变差。

![Transformer 残差连接梯度机制封面](/images/content-rebuild/1e1f3431f7b71f9a/cover.png)

---

## 贯穿例子：一个 6 层子网络中的梯度传递

为了直观理解加法直通的作用，我们来看一个具体算例。

假设网络中有 6 个连续子层（例如堆叠的注意力层或前馈网络层），每个子层的输入和输出维度均为 $d = 512$。假定在当前的训练状态下，由于激活函数饱和或权重偏小，每个子层局部的导数谱半径平均只有 $0.2$：

- **在没有残差连接的普通网络中**：误差信号跨越这 6 个子层需要经历 6 次矩阵连乘。传到底层时，梯度的缩放倍率约为：
  $$0.2^6 = 0.000064$$
  误差信号衰减了接近一万倍，底层的参数几乎接收不到有效的更新信号。
- **在包含残差连接的网络中**：反向传播的梯度表达式形如：
  $$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_{l+6}} \left( I + \sum_{i=l}^{l+5} \frac{\partial \mathcal{F}_i}{\partial x_l} \right)$$
  即便右侧各子层对底层输入的全导数之和非常小（例如接近 0），括号内依然保留了一个恒定的单位矩阵 $I$（模长增益为 1）。这为反向梯度提供了不经权重连乘衰减的直通基底，有效避免了误差信号在跨层传递中呈指数级衰减殆尽。

---

## 数学推导：从跨层展开到全导数方程

要在面试中严谨证明残差的作用，需要从跨越多个网络层的全局状态切入展开推导。

在标准纯净残差假设下（跳跃路径上不经过归一化或缩放），设第 $l$ 层的输入状态为 $x_l$。子层变换（如 [多头注意力机制](/fundamentals/transformer-arch/multi-head-attention.md) 或 [前馈网络 FFN](/interview/llm-interview/ffn-swiglu.md)）记作 $\mathcal{F}(x_l, \mathcal{W}_l)$。单层的前向关系为：

$$x_{l+1} = x_l + \mathcal{F}(x_l, \mathcal{W}_l)$$

将这个递推式自底向上连续展开，第 $l+2$ 层的状态为：

$$x_{l+2} = x_{l+1} + \mathcal{F}(x_{l+1}, \mathcal{W}_{l+1}) = x_l + \mathcal{F}(x_l, \mathcal{W}_l) + \mathcal{F}(x_{l+1}, \mathcal{W}_{l+1})$$

依此类推，任意更深层 $L$（其中 $L > l$）的激活状态可以直接写成底层状态 $x_l$ 与中间所有子层残差输出的累加和：

$$x_L = x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i, \mathcal{W}_i)$$

![残差连接的前向递归与跨层展开图](/images/content-rebuild/1e1f3431f7b71f9a/fig-1.png)

现在根据微积分链式法则，设模型最终损失为 $\mathcal{L}$，反向传播到达第 $L$ 层输出的误差梯度记为 $\frac{\partial \mathcal{L}}{\partial x_L}$。损失函数对底层状态 $x_l$ 的全导数为：

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \frac{\partial x_L}{\partial x_l}$$

对展开后的 $x_L$ 关于 $x_l$ 求偏导：

$$\frac{\partial x_L}{\partial x_l} = \frac{\partial}{\partial x_l} \left( x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i, \mathcal{W}_i) \right) = I + \sum_{i=l}^{L-1} \frac{\partial \mathcal{F}(x_i, \mathcal{W}_i)}{\partial x_l}$$

代回全导数公式，得到核心梯度传播方程：

$$\frac{\partial \mathcal{L}}{\partial x_l} = \underbrace{\frac{\partial \mathcal{L}}{\partial x_L}}_{\text{恒等直通项}} + \underbrace{\frac{\partial \mathcal{L}}{\partial x_L} \sum_{i=l}^{L-1} \frac{\partial \mathcal{F}(x_i, \mathcal{W}_i)}{\partial x_l}}_{\text{变换求和项}}$$

这个展开式清晰展现了残差连接的运作方式：反向传播信号被拆分为两条路径，左侧是没有任何权重参与的无损直通项，右侧是中间各层变换的加权组合项。

---

## 机制对比：打破链式乘法与对比 LSTM 门控

### 为什么单位矩阵 I 打破了连乘衰减？

对比没有跳跃连接的普通串联网络（Plain Network）：

$$x_{l+1} = \sigma(W_l x_l)$$

跨越 $K = L - l$ 个层级时，由链式法则可得雅可比矩阵的纯连乘：

$$\frac{\partial x_L}{\partial x_l} = \prod_{i=l}^{L-1} \frac{\partial x_{i+1}}{\partial x_i} = \prod_{i=l}^{L-1} \left( \operatorname{diag}(\sigma'(W_i x_i)) \cdot W_i^T \right)$$

当各层雅可比矩阵的特征值模长小于 1 时，随着层数 $K$ 线性增加，其连乘结果会以几何级数收缩趋向于 0 矩阵：

$$\lim_{K \to \infty} \left\| \prod_{i=l}^{l+K-1} J_i \right\| \to 0$$

![普通网络连乘衰减与残差直通路径对比](/images/content-rebuild/1e1f3431f7b71f9a/fig-2.png)

而在残差网络中，梯度变为加法结构：

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left( I + \sum_{i=l}^{L-1} \frac{\partial \mathcal{F}_i}{\partial x_l} \right)$$

其中求和项是中间各残差分支输出关于底层状态 $x_l$ 的全偏导（包含跨层高阶复合项，而非孤立单层雅可比矩阵的简单累加）。除非整个求和项的雅可比矩阵恰好精确等于 $-I$（在连续参数空间和随机初始化下发生的测度接近 0），否则括号项整体不会收缩归零。只要顶层存在有效梯度 $\frac{\partial \mathcal{L}}{\partial x_L}$，底层就能稳固接收到直通信号。

### 残差连接与 LSTM 门控通道的差异

在面试中，考官经常将残差连接与循环神经网络中的抗衰减机制做对比。需要区分两者的实现假设：

在 [LSTM 缓解长程梯度衰减的机制](/interview/llm-interview/lstm-gradient-vanishing.md) 中，其核心是维持细胞状态（Cell State）的流动：

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

LSTM 对前一时刻细胞状态的导数为：

$$\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t)$$

![残差直通与 LSTM 门控通道机制对比](/images/content-rebuild/1e1f3431f7b71f9a/fig-3.png)

两者的关键差异在于：
- **确定性 vs 参数依赖**：残差连接的直通系数是数学上恒定的单位矩阵 $I$（或标量 1），不依赖任何参数；LSTM 的遗忘门 $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ 是动态数据驱动的，网络必须通过学习让门控激活接近 1 才能维持记忆。
- **乘法衰减风险**：LSTM 跨越 $T$ 个时间步的梯度传递依然是标量连乘 $\prod_{\tau=1}^T f_\tau$。如果遗忘门在序列展开中长期处于 0.8 或 0.9，经过数十步后仍会出现明显的衰减；残差连接在网络深度方向上是纯加法累加。
- **优化维度**：LSTM 针对的是序列长度轴（Time Step）的梯度衰减，而 Transformer 的残差连接针对的是网络层数轴（Depth）的梯度阻碍。

<ReadMoreLock />

---

## 解决网络退化：从全量映射到残差逼近

在残差结构出现前，深度学习面临“网络退化（Degradation Problem）”：排除过拟合影响后（因为训练集误差同样升高），简单加深网络层数反而会导致模型的训练误差和测试误差双双恶化。

从理论上看，如果在已收敛的浅层网络上再堆叠若干层，只要这些新增层学会“恒等映射”（即输出等于输入 $H(x) = x$），深层网络的训练误差就绝不应该比浅层网络更高。但在实际训练中，常规优化算法（如 SGD 或 AdamW）很难让普通多层网络学出恒等函数。

![网络退化现象与残差恒等学习动力学对比](/images/content-rebuild/1e1f3431f7b71f9a/fig-4.png)

### 为什么普通前馈层很难学出恒等映射？

考虑一个常规前馈子层：$H(x) = \operatorname{GELU}(W_2 \operatorname{GELU}(W_1 x + b_1) + b_2)$。
1. 要让这个非线性复合函数对任意连续向量 $x$ 满足 $H(x) \equiv x$，参数矩阵必须满足极度严苛的条件：$W_2 W_1 = I$、偏置必须精确归零，且所有激活值必须严格落在激活函数的线性区。
2. 深度学习的通用初始化策略（如 Xavier、Kaiming 初始化）通常均值为 0，且优化过程中配合有权重衰减（Weight Decay / L2 正则）。优化器的自然动力学倾向于将参数拉向 0，而不是推向特定的非零单位矩阵。

残差连接把网络目标重构为：

$$H(x) = x + \mathcal{F}(x)$$

此时，学习恒等映射的条件变成了：

$$\mathcal{F}(x) = 0$$

让一个子层输出全 0 是非常简单的：只需将最外层线性变换的投影权重初始化为 0，或者在权重衰减的作用下让参数收缩到 0，该层就会自然变成无害的恒等跳过通道。这赋予了深层网络“性能保底”的能力。

---

## 工程实现与边界：LayerNorm 摆放与方差累积

在纯数学公式中，残差项是纯净的 $x_{l+1} = x_l + \mathcal{F}(x_l)$。然而在实际的 [Transformer 架构](/fundamentals/dl-basics/transformer.md) 中，残差分支必须配合归一化层工作，其摆放位置直接决定了主干直通机制是否有效。

### Post-LN 对纯净直通的缩放抑制

原始 Vaswani 论文采用的是 Post-LN 结构：

$$x_{l+1} = \operatorname{LayerNorm}(x_l + \mathcal{F}(x_l))$$

此时，反向传播穿透第 $l+1$ 层时必须经过该层的 LayerNorm：

$$\frac{\partial x_{l+1}}{\partial x_l} = \frac{\partial \operatorname{LayerNorm}(z_l)}{\partial z_l} \cdot \left( I + \frac{\partial \mathcal{F}(x_l)}{\partial x_l} \right), \quad z_l = x_l + \mathcal{F}(x_l)$$

![Post-LN 与 Pre-LN 对残差直通梯度的物理阻断效应](/images/content-rebuild/1e1f3431f7b71f9a/fig-5.png)

在 Post-LN 架构中，LayerNorm 位于残差相加之后的主干路径上。LayerNorm 内部包含除以标准差 $\sigma_l$ 的操作。当网络层数加深时，主干上的激活方差逐层累加膨胀，导致反传到底层的梯度幅度被严重除小缩放。因此原版 Transformer 在没有 Warmup 学习率预热策略时极难收敛，且在无技巧的情况下很难稳定扩展到 30 层以上。

### Pre-LN 的无损直通修复

现代主流大语言模型（如 LLaMA、GPT 系列）普遍采用 Pre-LN 结构：

$$x_{l+1} = x_l + \mathcal{F}(\operatorname{LayerNorm}(x_l))$$

在 Pre-LN 结构中，LayerNorm 被移入了非线性分支 $\mathcal{F}$ 内部，主干跳跃连接重新变回了严格的恒等加法：

$$x_L = x_0 + \sum_{i=0}^{L-1} \mathcal{F}(\operatorname{LayerNorm}(x_i))$$

此时 $\frac{\partial x_L}{\partial x_0}$ 重新具备了纯净的 $I + \sum$ 形式，即使模型深度达到数百层，梯度也能稳定传回嵌入层。关于 LayerNorm 及其演进变体在计算流上的更多细节，可参考 [LayerNorm 与 RMSNorm 原理解析](/interview/llm-interview/layernorm-rmsnorm.md)。

### 初始权重过大导致的方差累加

即便使用 Pre-LN，残差结构也存在固有风险。在假设各残差分支的输出与主干特征互不相关（协方差近似为 0）的简化分析前提下，前向方差可近似表示为线性累加：

$$\operatorname{Var}(x_L) \approx \operatorname{Var}(x_0) + \sum_{i=0}^{L-1} \operatorname{Var}(\mathcal{F}(\operatorname{LN}(x_i)))$$

如果每一层残差分支输出的方差没有做初始化收缩约束，主干的数值方差就会随着深度不断膨胀。

![残差分支初始权重过大导致的直通湮灭与方差失控](/images/content-rebuild/1e1f3431f7b71f9a/fig-6.png)

如果子层初始权重设置过大，会引发两个后果：
1. **直通信号被噪声淹没**：深层输出被各子层杂乱的随机初始化输出淹没，恒等直通项失去了作为稳定基底的意义。
2. **数值发散**：方差膨胀导致末层分类头或 Softmax 的输入数值极大，进而导致 Softmax 输出饱和、梯度过小或出现 NaN。

因此在实际训练大模型时，通常会采用权重缩放策略（例如将残差分支最后一层线性层的权重缩小为 $\frac{1}{\sqrt{2N}}$ 初始化，或使用 DeepNorm 缩放因子），确保初期 $\mathcal{F}(x) \approx 0$，让网络在训练初始阶段先表现为一条干净的恒等直通通道。

---

## 面试追问与常见认知误区

### 追问 1：如果彻底拿掉 Transformer 中的所有残差连接，还能加深到 32 层收敛吗？

**口述要点**：
无法正常收敛。32 层的标准 Transformer 包含 64 个复杂的串联子层（32 个自注意力层与 32 个 FFN 层）。
- 自注意力机制内部的 Softmax 本质是加权凸组合，单层雅可比矩阵天生具有收缩性；FFN 内部包含非线性激活函数。64 个子层连续连乘，梯度在回传几层后就会缩减至数值下溢。
- 缺乏恒等保底映射，深层 Transformer 的损失曲面会变得极度破碎（Non-smooth Landscape），训练在最初几步就会发生梯度消失或数值发散。

### 追问 2：残差连接是否把深层 Transformer 变成了一种集成模型？

**口述要点**：
在分析视角下有一定道理（参考 Veit 等人针对残差网络的集成分析）。
展开多层累加式：$x_L = x_0 + \mathcal{F}_0 + \mathcal{F}_1 + \dots + \mathcal{F}_{L-1}$。从展开图上看，输入到输出实际存在 $2^L$ 条潜在的前向计算路径组合。
在训练好的模型中，如果随机丢弃中间某一层的残差分支，模型的输出性能通常只是平缓下滑，而不会像普通串联网络那样直接彻底瘫痪。这表明模型在某种程度上学会了“在主干特征上逐步叠加增量更新”的解耦计算模式。

### 常见认知误区

- **误区 1：“有了残差连接，底层梯度就绝对不可能为 0。”**
  - **反驳**：这是不严谨的。如果顶层损失对输出的初始导数 $\frac{\partial \mathcal{L}}{\partial x_L}$ 本身因为分类头饱和（如 Cross-Entropy 配合极度置信的 Softmax）而接近 0，那么即便通过单位阵直通，乘出来的底层梯度依然接近 0。残差连接解决的是**跨层传递过程中的指数级缩放**，并不意味着能无视顶层任务梯度的归零。
- **误区 2：“残差连接既然能保证梯度流动，那模型就不需要归一化（LayerNorm）了。”**
  - **反驳**：残差连接只负责反向梯度的加法传递，但前向加法 $x + \mathcal{F}(x)$ 会导致激活值的方差逐层累积增大。如果没有 LayerNorm 重新校准各层的特征分布，输入到后续注意力点积的数值就会严重膨胀，导致注意力权重极度极端化和数值溢出。残差连接保证反向梯度流通，LayerNorm 保证前向激活尺度稳定，两者协同配合才能训练深层网络。
