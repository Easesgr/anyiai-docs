---
title: Transformer 为什么采用 Layer Normalization 而非 Batch Normalization？
icon: mdi:layers
tag:
- Transformer
- LayerNorm
- BatchNorm
description: 解析 Transformer 选用 Layer Normalization 的核心原因：通过张量统计轴向对比、变长序列 Padding 干扰以及自回归推理一致性，厘清与
  Batch Normalization 的机制差异。
---

在自然语言处理与大模型相关的技术面试中，“为什么 Transformer 选择 Layer Normalization（层归一化，简称 LN）而不是卷积神经网络中常见的 Batch Normalization（批归一化，简称 BN）”是一道高频考察模型底层机制的经典试题。回答这道题的关键，在于厘清**张量统计缩减的轴向**与**序列建模任务特性**之间的匹配关系。

在上一篇 [/interview/llm-interview/transformer-vs-llm.md](/interview/llm-interview/transformer-vs-llm.md) 中，我们梳理了标准 Transformer 架构与现代自回归大模型的演进脉络。本篇将聚焦于归一化层，从数据维度、变长序列填充和推理一致性三个方面分析其选型原因。

一个典型的贯穿例子是：假设训练批次中有两句话，短句“你好”（实际长度为 2），长句“今天天气真好适合出去运动”（实际长度为 12）。为了拼成规整矩阵，短句末尾必须填充 10 个 `<pad>` 标记。如果采用跨样本统计的 BN，这 10 个填充项会直接参与均值和方差计算；而如果采用 LN，每个词的向量只在自己的特征维度内部完成计算，完全不受填充标记干扰。

![Transformer 归一化选型核心机制概览](/images/content-rebuild/fadd370120c1a9ba/cover.png)

---

## 1. 面试口述答案

面试中遇到该问题时，可按以下三个核心点作答：

> **口述作答参考：**
> 
> Transformer 选择 Layer Normalization 而非 Batch Normalization，主要原因有三点：
> 
> 1. **避免变长序列的 Padding 统计污染**：自然语言句子长度通常不一致，同批次内短句子必须填充 `<pad>` 占位符。BN 是跨样本、跨时间步联合统计，填充的无效数值会拉偏全局均值和方差，导致有效词的特征表示随批次内最长句子的变化而剧烈波动；LN 是在单个 Token 的特征维度内独立计算，完全不受序列填充的影响。
> 2. **适应自回归生成与动态 Batch 推理**：大语言模型推理时通常采用自回归逐步解码，线上请求的 Batch Size 动态变化，极端情况下单样本（$\text{Batch Size}=1$）生成。BN 依赖全局累积的滑动均值与方差，在小批次或动态长文本下极易发生分布偏移；LN 即算即用，不需要维护全局统计量。
> 3. **计算解耦与工程实现一致**：LN 在训练和推理阶段使用完全相同的数学公式，样本之间互不依赖，更适合长序列分布式训练中的并行加速。

---

## 2. 统计轴向对比：沿批次轴 vs 沿特征隐藏轴

理解两者的差异，首先需要从三维张量的几何切片入手。在基于 Transformer 的序列模型中，隐藏层的输入张量形状通常表示为三维矩阵：

$$
X \in \mathbb{R}^{B \times L \times D}
$$

其中 $B$ 为批次大小（Batch Size），$L$ 为序列长度（Sequence Length），$D$ 为特征隐藏维度（Hidden Dimension，即论文中的 $d_{\text{model}}$）。任意元素 $X_{b, l, d}$ 表示第 $b$ 个样本在第 $l$ 个时间步的第 $d$ 个隐藏特征值。

![BN 与 LN 在三维张量上的统计轴向几何对比](/images/content-rebuild/fadd370120c1a9ba/fig-1.png)

### Batch Normalization 的统计轴向

标准 BN 最初针对图像设计（输入形状多为 $[B, C, H, W]$），其假设同通道内所有像素服从同一分布。若将类似方式应用到三维序列张量上，最常见的实现是沿着批次轴 $B$ 与序列轴 $L$ 联合统计，仅保留特征维度 $D$：

$$
\mu_d = \frac{1}{B \times L} \sum_{b=1}^{B} \sum_{l=1}^{L} X_{b, l, d}
$$

$$
\sigma_d^2 = \frac{1}{B \times L} \sum_{b=1}^{B} \sum_{l=1}^{L} (X_{b, l, d} - \mu_d)^2
$$

$$
\text{BN}(X_{b, l, d}) = \frac{X_{b, l, d} - \mu_d}{\sqrt{\sigma_d^2 + \epsilon}} \cdot \gamma_d + \beta_d
$$

这种统计方式建立在一个先验假设上：不同样本在所有时间步上的第 $d$ 维特征均来自同一分布。但在自然语言中，不同位置的词（如主语、介词、标点）语法功能与特征分布差异极大，将其强行拉通统计在语义上并不合理。

### Layer Normalization 的统计轴向

Layer Normalization 放弃了跨样本统计。在 Transformer 的标准实现中，LN 针对每一个样本在每一个时间步的 Token 向量，独立计算其在特征维度 $D$ 上的均值与方差：

$$
\mu_{b, l} = \frac{1}{D} \sum_{d=1}^{D} X_{b, l, d}
$$

$$
\sigma_{b, l}^2 = \frac{1}{D} \sum_{d=1}^{D} (X_{b, l, d} - \mu_{b, l})^2
$$

$$
\text{LN}(X_{b, l, d}) = \frac{X_{b, l, d} - \mu_{b, l}}{\sqrt{\sigma_{b, l}^2 + \epsilon}} \cdot \gamma_d + \beta_d
$$

此时，均值 $\mu_{b, l}$ 和方差 $\sigma_{b, l}^2$ 的形状为 $[B, L, 1]$。每个位置的 Token 向量独立调整尺度。可学习参数增益 $\gamma$ 和偏置 $\beta$ 维度为 $[D]$，用于恢复网络所需的表达能力。

关于归一化层在深度学习中的演进脉络，可交叉参考 [/fundamentals/dl-basics/normalization.md](/fundamentals/dl-basics/normalization.md)。

<ReadMoreLock />

---

## 3. 序列变长与 Padding 污染：算例对比

在 NLP 批处理计算中，为了让硬件高效执行矩阵乘法（GEMM），短序列通常需要用特殊的填充标记（如 `<pad>`）补齐到当前批次的最大长度。

![Padding 对 Batch Normalization 统计量的拉偏破坏](/images/content-rebuild/fadd370120c1a9ba/fig-2.png)

我们可以通过一个简化算例观察补零操作对统计量的影响：

设批次大小 $B=2$，最大对齐长度 $L=4$，为便于计算取单一特征通道 $D=1$。
*   **样本 1**：实际有效长度为 2，有效特征值为 $[2.0, 4.0]$，末尾填充两个 0 占位：`[2.0, 4.0, 0.0, 0.0]`；
*   **样本 2**：实际有效长度为 4，特征值为 `[1.0, 3.0, 2.0, 4.0]`，无须填充。

### BN 的统计结果受 Padding 干扰

若按 BN 跨批次与序列联合求和，8 个数值全部参与计算：

$$
\mu_{\text{BN}} = \frac{(2.0 + 4.0 + 0.0 + 0.0) + (1.0 + 3.0 + 2.0 + 4.0)}{8} = \frac{16.0}{8} = 2.0
$$

若仅统计 6 个真实有效的语义特征，其真实均值为：

$$
\mu_{\text{True}} = \frac{(2.0 + 4.0) + (1.0 + 3.0 + 2.0 + 4.0)}{6} = \frac{16.0}{6} \approx 2.67
$$

填充的零值将均值拉低了约 $25\%$。这意味着一个样本的归一化结果，会受到同批次内其他样本长度的直接影响：
*   若样本 1 与长度为 512 的长文本编入同一批次，样本 1 需填充 510 个零，其特征均值将被严重拉向 0；
*   若样本 1 与同样长度为 2 的短文本编入同一批次，则不需要填充零，均值恢复为 3.0。

这种由组批（Batching）策略引入的数据分布抖动，会直接干扰梯度的稳定更新。

### LN 的计算完全解耦

对于 **Layer Normalization**，统计是在各 Token 自身的隐藏特征维度 $D$（通常为 768、4096 等）上进行。样本 1 中有效位置的计算仅依赖该位置自身的向量分量，与序列末尾填充了多少个 `<pad>`、批次中有无超长文本没有任何数学关联，有效特征的分布得以保持稳定。

---

## 4. 自回归解码与推理一致性

除了序列填充问题，Transformer 广泛用于自回归生成（Autoregressive Decoding）任务，这一场景对归一化算子的推理特性提出了明确要求。

![自回归推理阶段单 Token 动态生成下两者的适应性对比](/images/content-rebuild/fadd370120c1a9ba/fig-3.png)

### 1. 动态 Batch 与单样本推理
在实际生产服务中，为了降低首字延迟并提高显存利用率，系统通常采用动态批处理（Continuous Batching）。在请求较少或单卡运行时，Batch Size 经常为 1；而在启用 KV Cache 的自回归解码步中，每次送入模型的序列长度 $L=1$。
*   **BN 的机制局限**：在训练阶段，BN 依赖当前批次内多个样本计算方差。若在训练中传入 $B=1$ 且 $L=1$，单个标量无法计算无偏样本方差。在推理阶段，BN 必须依赖预先累积的全局滑动均值（Running Mean）和滑动方差（Running Variance）。然而，自回归生成的文本分布具有长尾特性，离线累积的全局统计量与特定 Prompt 下的局部激活分布容易出现偏差。
*   **LN 的即算即用**：LN 在单个 Token 向量内部计算均值和方差。即使 $B=1$ 且 $L=1$，只要隐藏维度 $D$ 足够大（如常见模型的 $D=4096$），单个向量内部即可完成统计归一化，不需要在推理阶段维护和切换全局状态。

### 2. 训练与推理模式一致性
BN 在工程上区分训练模式（使用批次统计量并更新滑动累积值）与评估模式（固定使用滑动累积量）。模式切换偶尔会因滑动统计量估计不足引发推理效果异常。

LN 的训练与推理逻辑完全一致，消除了双模式运行时的状态同步开销。

---

## 5. 代码验证：PyTorch 维度与统计行为分析

下面的 PyTorch 代码验证了 BN 与 LN 在张量统计轴向上的实际区别：

```python
import torch
import torch.nn as nn

# 设定张量维度：Batch_size=2, Seq_len=3, Hidden_dim=4
B, L, D = 2, 3, 4
torch.manual_seed(42)

# 模拟输入张量：第 1 个样本长度为 2，第 3 位为填充 0
x = torch.randn(B, L, D)
x[0, 2, :] = 0.0  # 注入 Padding 标记

print("输入张量形状 [B, L, D]:", x.shape)

# 1. 验证 LayerNorm 的统计维度
ln = nn.LayerNorm(normalized_shape=D, elementwise_affine=False)
x_ln = ln(x)
# 手动计算 LN：在最后一个维度 (dim=-1) 统计均值和方差
manual_ln_mean = x.mean(dim=-1, keepdim=True)
manual_ln_var = x.var(dim=-1, keepdim=True, unbiased=False)
manual_x_ln = (x - manual_ln_mean) / torch.sqrt(manual_ln_var + 1e-5)

print("LN 算子与手动实现的最大误差:", torch.max(torch.abs(x_ln - manual_x_ln)).item())
print("LN 均值张量形状:", manual_ln_mean.shape)  # 输出为 [2, 3, 1]

# 2. 验证 BatchNorm1d 的统计维度（BatchNorm1d 默认输入形状为 [B, C, L]）
x_permuted = x.transpose(1, 2)  # 转换为 [B, D, L]
bn = nn.BatchNorm1d(num_features=D, affine=False, momentum=0.1)
bn.train()
x_bn = bn(x_permuted).transpose(1, 2)

# 手动计算 BN：在 Batch 轴和 Seq 轴 (dim=(0, 2)) 联合统计
manual_bn_mean = x_permuted.mean(dim=(0, 2), keepdim=True).transpose(1, 2)
print("BN 均值张量形状:", manual_bn_mean.shape)  # 输出为 [1, 1, 4]
```

![代码验证实验执行流程与统计量维度输出解析](/images/content-rebuild/fadd370120c1a9ba/fig-4.png)

从运行结果可以明确看出：
*   **LN 的求和轴为 `dim=-1`**：均值张量形状为 `[B, L, 1]`，每个 Token 保持独立；
*   **BN 的求和轴跨越批次和序列维度**：均值张量形状被压缩为 `[1, 1, D]`，同批次内所有样本以及 Padding 的内容均被算入统计量中。

---

## 6. 面试追问与架构边界

### 追问 1：ViT 图像切分 Patch 长度固定且无 Padding，为什么依然采用 LN？

在 Vision Transformer（ViT）中，图像被切分成固定数量的 Patch（如 $16 \times 16$），序列长度 $L$ 固定，并不存在文本中的变长 Padding 现象。但 ViT 依然沿用了 LN。

![ViT 处理 Patch 序列并选用 LN 的数学与注意力机制因果](/images/content-rebuild/fadd370120c1a9ba/fig-5.png)

主要原因包括：
1.  **自注意力点积的尺度稳定性**：自注意力计算依赖 Query 与 Key 的点积：
    $$
    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$
    LN 约束了每个 Token 隐藏向量在各维度上的方差，有助于抑制极端模长对 Softmax 分布平缓程度的扰动。
2.  **小 Batch 训练的稳定性**：ViT 模型参数量较大，在高分辨率或大模型预训练时，单卡批次大小往往受到显存限制而较小。此时 BN 的批次统计量方差较大，而 LN 不受批次大小影响。

关于 ViT 结构更详细的分析，可参考 [/interview/llm-interview/vit-vision-transformer.md](/interview/llm-interview/vit-vision-transformer.md)。

### 追问 2：能否通过加 Mask 剔除 Padding 来改进 BN？

一种常见的思路是：通过 Attention Mask 过滤掉 Padding 标记，仅统计有效 Token 的均值和方差。

这种设计在理论上可以缓解数值污染，但在工程实现上存在明显短板：
*   **硬件执行效率低**：带有动态掩码的归一化计算需要根据每个样本的有效长度动态调整除数，造成内存访存不规整，无法充分发挥现代 GPU 对规整张量块的并行计算吞吐。
*   **未解决单样本与自回归推理问题**：即便训练期加入了 Masked BN，推理期单样本逐字解码时依然面临全局统计量难以准确匹配动态文本分布的根本矛盾。

---

## 7. 常见误区排雷

![面试常见伪机制与误解排雷总结](/images/content-rebuild/fadd370120c1a9ba/fig-6.png)

在面试作答时，需要注意避免以下三个典型概念误区：

*   **误区 1：声称“Transformer 不用 BN 是因为 BN 显存开销更大”**
    *   *真实情况*：归一化层的参数量仅为增益 $\gamma$ 和偏置 $\beta$（通道维度 $D$ 级别），在整个模型参数中占比极小。Transformer 的显存压力主要来自自注意力矩阵激活值、全连接层中间状态以及推理时的 KV Cache，与 BN 或 LN 本身的参数显存无关。
*   **误区 2：将归一化的主要作用归结于“消除了内部协变量偏移（ICS）”**
    *   *真实情况*：现代深度学习理论研究表明，归一化方法的核心收益在于平滑了优化过程中的损失曲面（Loss Landscape），使梯度流动更稳定，允许使用更大的学习率加速收敛，而非单纯消除了特征分布偏移。
*   **误区 3：认为 LN 只能在一维特征上计算**
    *   *真实情况*：在 PyTorch 的 `nn.LayerNorm(normalized_shape)` 实现中，`normalized_shape` 支持传入元组，对张量末尾的多个连续维度联合做归一化。在 Transformer 中归一化最后一维 $D$，是针对 Token 向量表示所选择的标准设计实践，并非算子本身的数学限制。

---

## 总结与延伸对比

| 对比维度 | Batch Normalization (BN) | Layer Normalization (LN) |
| :--- | :--- | :--- |
| **统计轴向** | 跨批次与序列维度（常用 $\text{dim}=(0, 1)$ 联合求和） | 沿单 Token 隐藏特征维度（$\text{dim}=-1$） |
| **样本解耦程度** | 样本间存在统计耦合，单样本输出受同批其他样本影响 | 样本间完全解耦，每个 Token 独立计算 |
| **变长序列适应** | 较差，Padding 占位符会直接干扰统计均值和方差 | 良好，归一化在特征轴进行，不受序列长度填充影响 |
| **批次大小依赖** | 强依赖，小批次下统计量抖动明显 | 不依赖批次大小，支持单样本（$B=1$）推理 |
| **训练推理一致性** | 需维护滑动平均统计量，训练与推理计算逻辑不同 | 训练与推理公式完全一致，即算即用 |
| **典型应用** | 传统 CNN、规整图像分类网络 | Transformer、自回归大语言模型、RNN 序列任务 |

在现代大模型演进中，为了进一步简化计算并提升吞吐，许多模型（如 LLaMA、Mistral）使用去除了均值中心化操作的 RMSNorm 替代了标准 LN。关于 RMSNorm 的数学简化与加速原理，可继续阅读 [/interview/llm-interview/layernorm-rmsnorm.md](/interview/llm-interview/layernorm-rmsnorm.md)。
