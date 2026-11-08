---
title: 视觉 Token 是什么？图像如何变成 Transformer 能读的序列
icon: mdi:eye-outline
tag:
- 多模态基础
- 视觉Token
- 图像编码
description: 介绍视觉 Token 的生成机制：从图像网格切片、展平映射、步长卷积等价实现到位置编码，厘清连续向量与离散文本 Token 的区别。
---

Transformer 最初是为处理一维文本序列设计的。文本可以通过分词切分成词元，再通过查表得到向量。然而，数码照片是二维的像素网格，包含高度、宽度和红绿蓝三个颜色通道。要让 Transformer 理解图像，就必须把二维连续的像素矩阵转换成一维的特征向量序列，这些特征向量就是**视觉 Token（Visual Token）**。

具体来说，主流模型（如 Vision Transformer 与 CLIP）的实现方式是：**把整张图像切成多个固定大小的网格方块（Patch），再把每个小方块内的像素展平并通过线性变换映射为一个固定维度的向量**。接下来，我们以一张常见的 $224 \times 224$ 像素、包含 3 个颜色通道（RGB）的彩色照片为例，拆解这一转换的完整过程与工程实现细节。

![视觉 Token 化全景图](/images/content-rebuild/5b8758eac98a46c9/cover.png)

---

## 为什么不能把每个像素直接当成一个 Token？

文本模型将文本切成离散的 Token，例如在 [/interview/llm-interview/tokenization.md](/interview/llm-interview/tokenization.md) 中介绍的 BPE 算法，每个词或子词对应词表中的一个数字编号。

面对一张尺寸为 $(H, W, C)$ 的图像（高 $H$、宽 $W$、通道数 $C$），最朴素的想法是把每个像素点当作一个序列输入。但这种做法在实际工程中存在两个严重瓶颈：

1. **序列过长导致显存与算力吃紧**：对于一张并不算大的 $224 \times 224$ 图像，总像素数达到 $224 \times 224 = 50,176$。标准自注意力机制中，注意力权重的计算复杂度与显存占用随序列长度 $N$ 呈二次方增长（$\mathcal{O}(N^2)$）。处理长度超 5 万的序列，单层自注意力中 $N \times N$ 的注意力矩阵就需要存储约 $2.52 \times 10^9$ 个数值。仅单精度浮点数（float32）就需占用约 10 GB 显存，半精度（fp16）也需要约 5 GB，普通显卡很难支撑深层网络的多层堆叠运算。
2. **单像素的语义密度极低**：在自然图像中，相邻像素之间具有极强的色彩连续性与统计冗余。单个像素（例如一个 RGB 值为 `[120, 125, 122]` 的点）无法代表任何具体的物体部件或纹理，其高频噪点还会分散自注意力对宏观轮廓和语义特征的捕捉。

因此，模型需要将局部的像素区域打包合并，在降低序列长度的同时提取局部特征。

![像素与序列的计算开销冲突对比](/images/content-rebuild/5b8758eac98a46c9/fig-1.png)

---

## 网格切片（Patching）：把整图划分为固定小方块

Vision Transformer（ViT）采用的切片方案非常直观：将图像划分为网格状、互不重叠的正方形小图像块，每个图像块被称为一个 **Patch**。

设输入图像大小为 $(H, W, C)$，设定的切片边长为 $P$（常见的设定为 $16$ 或 $14$ 像素）。
切片后的横向与纵向网格数分别为：

$$N_H = \frac{H}{P}, \quad N_W = \frac{W}{P}$$

整张图像切分出的 Patch 总数为：

$$N = N_H \times N_W = \frac{H \cdot W}{P^2}$$

以我们的贯穿例子来看：
- 输入尺寸：$224 \times 224 \times 3$
- 设定切片大小：$P = 16$
- 纵横切片数：$N_H = 224 / 16 = 14$，$N_W = 224 / 16 = 14$
- Patch 总数：$N = 14 \times 14 = 196$ 个小方块
- 单个 Patch 的原始像素体积：$16 \times 16 \times 3 = 768$ 个数值

通过这一步切分，原本包含 50,176 个像素点的二维图像，被重组为包含 196 个小图像块的序列集合，输入序列的长度从 50,176 大幅下降到了 196。

<ReadMoreLock />

![图像网格切片与展平流程](/images/content-rebuild/5b8758eac98a46c9/fig-2.png)

---

## 从像素块到特征向量：线性映射与卷积的等价性

切分出的每个小方块包含 $P \times P \times C$（例如 768）个像素值。为了适配 Transformer 模型的隐层维度 $D$，需要把每个展开的切片向量 $\mathbf{x}_p$ 投影为一个长度为 $D$ 的特征向量：

$$\mathbf{z}_i = \mathbf{x}_{p, i} \mathbf{W} + \mathbf{b}$$

其中 $\mathbf{W} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 是可训练的投影权重矩阵，$\mathbf{b} \in \mathbb{R}^D$ 是偏置向量。得到的 $\mathbf{z}_i \in \mathbb{R}^D$ 即为该切片对应的**视觉 Token 嵌入（Patch Embedding）**。

### 卷积操作的等价实现
在深度学习框架中，显式地把图像用循环或切片展开成矩阵不仅耗费内存，而且访存效率较低。在工程实现中，通常使用一个步长等于核尺寸的 **二维卷积（Conv2d）** 来一次性完成切片与线性映射：
- 卷积核尺寸（`kernel_size`）：$(P, P)$
- 步长（`stride`）：$(P, P)$
- 输入通道数（`in_channels`）：$C$
- 输出通道数（`out_channels`）：$D$

因为卷积步长刚好等于卷积核尺寸，卷积核在整张图上滑动时互不重叠，每次刚好覆盖一个 Patch 的范围；卷积核中每个输出通道对应的权重，正好相当于线性投影矩阵 $\mathbf{W}$ 中一列展开后的参数。

下面的代码演示了显式展开后做线性投影与使用步长卷积在数值上的严格一致性：

```python
import torch
import torch.nn as nn

# 设定参数：批大小 1，RGB 3 通道，分辨率 224x224，Patch 16x16，向量维度 768
B, C, H, W = 1, 3, 224, 224
P = 16
D = 768

x = torch.randn(B, C, H, W)

# 方法一：利用步长等于核尺寸的 2D 卷积一步完成
conv_proj = nn.Conv2d(in_channels=C, out_channels=D, kernel_size=P, stride=P)
conv_out = conv_proj(x)  # 输出形状: [1, 768, 14, 14]
conv_tokens = conv_out.flatten(2).transpose(1, 2)  # 转换为序列形状: [1, 196, 768]

# 方法二：显式展开每个 Patch 并通过全连接层计算
linear_proj = nn.Linear(P * P * C, D)
with torch.no_grad():
    # 将卷积核参数 [D, C, P, P] 展平并复制给全连接权重 [D, P*P*C]
    linear_proj.weight.copy_(conv_proj.weight.reshape(D, -1))
    linear_proj.bias.copy_(conv_proj.bias)

# 使用 unfold 展开切片
unfold = nn.Unfold(kernel_size=P, stride=P)
x_unfolded = unfold(x).transpose(1, 2)  # 展开后形状: [1, 196, 768]
linear_tokens = linear_proj(x_unfolded)  # 投影后形状: [1, 196, 768]

# 验证两种方法的数值差异
max_diff = torch.max(torch.abs(conv_tokens - linear_tokens)).item()
print(f"两种实现的数值差异: {max_diff:.8e}")  # 输出通常在 1e-6 到 1e-7 以内（浮点误差范围）
```

这种卷积实现在 PyTorch 与 Hugging Face 的代码库中被广泛使用，既精简了代码，又充分利用了 GPU 的张量计算单元。

![线性映射与步长卷积的等价机制](/images/content-rebuild/5b8758eac98a46c9/fig-3.png)

---

## 空间结构恢复：位置编码的注入

自注意力机制本身是具有“置换等变性（Permutation Equivariance）”的——如果不显式注入位置信息，把输入的 196 个 Patch 打乱顺序，Transformer 计算出的自注意力权重与输出特征仅仅跟随发生相同的行与列位置置换，数值保持对应一致。这意味着自注意力本身无法感知各个切片在二维平面中的几何拓扑关系（例如猫耳朵在上方、猫爪在下方）。如果缺少位置信息，图像的全局空间结构就会丢失。

为了让模型区分每个切片来自图像的哪个区域，必须在进入注意力层前注入位置编码：

$$\mathbf{Z}_0 = [\mathbf{z}_1, \mathbf{z}_2, \dots, \mathbf{z}_N] + \mathbf{E}_{pos}$$

其中 $\mathbf{E}_{pos} \in \mathbb{R}^{N \times D}$ 是位置编码矩阵。

常用的位置编码方案有两种：

1. **一维可学习位置编码（1D Learnable Position Embeddings）**：
   将 14 行 14 列的网格按照从左到右、从上到下的顺序编成编号 0 到 195。模型为每个编号初始化一个 $D$ 维的向量，并在预训练过程中跟随模型参数一同优化更新。实验表明，经过训练后，模型自发学到了二维空间关系：空间上相邻的切片，其位置向量之间的余弦相似度显著高于相距较远的切片。
2. **二维插值（Bicubic Interpolation）**：
   当推理时的图像分辨率发生变化（例如从预训练的 $224 \times 224$ 变为 $448 \times 448$）时，生成的 Patch 数量从 196 变成 784。此时原有的 196 个位置向量无法直接按元素相加。常规做法是把原位置编码还原成 $14 \times 14$ 的空间网格，利用双三次插值重采样为 $28 \times 28$ 的网格，再展平加到新的切片序列上。这种方式允许视觉模型在一定范围内适应不同尺寸的图像输入。

![位置编码注入与空间拓扑恢复机制](/images/content-rebuild/5b8758eac98a46c9/fig-4.png)

---

## 分辨率与 Token 数量的换算关系

在自然语言模型中，Token 数量由文本长短自适应决定；而在标准视觉模型中，Token 数量完全由**输入图像的物理分辨率**与**切片尺寸（Patch Size）**严格决定：

$$N = \left\lfloor \frac{H}{P} \right\rfloor \times \left\lfloor \frac{W}{P} \right\rfloor$$

如果模型像 BERT 一样引入了额外的全局分类标记（`[CLS]` Token），则送入 Transformer 的总序列长度为 $N + 1$。

以下是常见视觉编码器配置的 Token 换算对照（不计 `[CLS]` 标记）：

| 模型典型配置 | 输入分辨率 ($H \times W$) | 切片尺寸 ($P \times P$) | 视觉 Token 数量 ($N$) | 相较 224 (Patch 14) 的序列倍率 |
| :--- | :--- | :--- | :--- | :--- |
| ViT-B/16 (CLIP 基础款) | $224 \times 224$ | $16 \times 16$ | 196 | $0.77\times$ |
| ViT-L/14 (CLIP 进阶款) | $224 \times 224$ | $14 \times 14$ | 256 | $1.00\times$（基准） |
| ViT-L/14@336px | $336 \times 336$ | $14 \times 14$ | 576 | $2.25\times$ |
| 现代多模态常用高清输入 | $448 \times 448$ | $14 \times 14$ | 1024 | $4.00\times$ |
| 细粒度文档/OCR 常用切片 | $896 \times 896$ | $14 \times 14$ | 4096 | $16.00\times$ |

### 分辨率变化对计算开销的影响
当切片尺寸固定时（例如保持 $P=14$），图像边长如果翻倍（从 224 提升到 448），视觉 Token 的数量会扩大为原来的 $2^2 = 4$ 倍（从 256 增加到 1024）。这一变化带来两个影响：
1. **视觉编码器自身的注意力计算开销上升**：全注意力的矩阵大小随 $N$ 平方增长，Token 数量扩大 4 倍意味着注意力矩阵的运算量和内存占用上升约 $4^2 = 16$ 倍。
2. **多模态大模型（VLM）的上下文压力**：在多模态大模型（如 [/fundamentals/multimodal-basics/multimodal-how.md](/fundamentals/multimodal-basics/multimodal-how.md) 所述架构）中，这 1024 个视觉 Token 会全部进入后续大语言模型的上下文窗口。这不仅挤占了可容纳的用户文本长度，还会显著增加首字响应时间（TTFT）。关于此类推理上下文的开销控制，可参考 [/architecture/token-cost.md](/architecture/token-cost.md)。

![分辨率提升引发 Token 数量与显存二次方增长关系](/images/content-rebuild/5b8758eac98a46c9/fig-5.png)

---

## 概念厘清：连续向量与离散词表索引的区别

“视觉 Token”这个名称容易让人联想到文本处理中的 Token。尽管它们在输入 Transformer 时都表现为一个一维向量序列，但两者的底层机制截然不同：

| 对比维度 | 文本 Token（Text Token） | 主流视觉 Token（Visual Patch Token） |
| :--- | :--- | :--- |
| **离散与连续** | **严格离散**：映射为预定义词表中一个具体的整数 ID（如编号 `7532`） | **连续取值**：每个 Token 都是高维实数空间 $\mathbb{R}^D$ 中的浮点数向量 |
| **生成途径** | 查表（Embedding Lookup）：通过整数 ID 从词表嵌入矩阵中直接提取对应行 | 前向投影：通过卷积层或全连接层对局部切片的实际像素值进行数值计算 |
| **词表限制** | 存在固定的词表大小（如 32,000 或 128,000），词表外字符可能被截断或转为未知标记 | **没有预设词表**：任意连续图像切片均能完成线性投影，不存在“未登录图像” |
| **语义粒度** | 单个 Token 往往对应一个完整的词根或单词，具备独立的语法与语义信息 | 单个 Patch 往往只包含部分纹理或局部色块，语义高度依赖全图注意力的聚合 |

> **注**：在部分生成式模型中，也存在使用离散量化（如 VQ-VAE、VQGAN）将图像块映射为离散 Codebook 索引的做法。但在当前主流的判别式视觉模型（如 ViT、CLIP）及主流多模态大语言模型（如 LLaVA 等）中，视觉 Token 均指代**切片投影后的连续实数向量序列**。

![连续视觉特征向量与离散文本 Token 的机制对比](/images/content-rebuild/5b8758eac98a46c9/fig-6.png)

---

## 思考与检验

1. 如果输入一张分辨率为 $448 \times 448$ 的单通道灰度图（$C=1$），采用 $14 \times 14$ 的 Patch 大小，在不添加额外特殊标记的情况下，会生成多少个视觉 Token？在投影前，每个小切片展平后的原始特征维度是多少？
2. 为什么主流多模态模型可以直接把图像切片得到的连续向量与文本词向量拼接在一起送进同一个 Transformer，而不需要先把视觉切片强行映射成词表中的整数 ID？
