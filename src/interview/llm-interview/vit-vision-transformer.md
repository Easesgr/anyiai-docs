---
title: ViT 是如何工作的？Vision Transformer 结构与图像序列化原理解析
icon: mdi:image
tag:
- ViT
- Vision Transformer
- 图像识别
description: 详解 ViT 将图像切块为 Patch 并转化为向量序列的过程，分析 CLS 标记、位置编码、Pre-LN 架构及与 CNN 归纳偏置的核心差异。
---

![Vision Transformer 整体架构与序列化流程总览](/images/content-rebuild/24a62555dc58ee25/cover.png)

在自然语言处理中，Transformer 将文本切分为 Token 序列，并通过自注意力机制建立长程依赖。在计算机视觉领域，图像长期被视为规则的二维连续像素网格，主流模型一直由卷积神经网络（CNN）主导。Google 在 2020 年提出的 Vision Transformer（ViT）打破了这一局面：**ViT 不依赖任何深层卷积网络，直接把二维图像切割为固定大小的图像块（Patches），展平后通过线性映射转化为一维向量序列，完整复用了标准的 Transformer Encoder 架构**。

理解 ViT 的关键在于两点：一是二维像素矩阵如何转化为一维离散向量序列；二是放弃了卷积内置的结构先验（归纳偏置）后，模型在不同数据规模下的表现为何会发生反转。前置基础可参考 [Transformer 和 LLM 有哪些区别](/interview/llm-interview/transformer-vs-llm.md) 以及 [Transformer 相比 RNN 和 CNN 有什么优势？](/fundamentals/transformer-arch/transformer-vs-rnn-cnn.md)。

---

## 30 秒口述结论

面试中被问到“请介绍 ViT 的基本结构和运行机制”时，可以按照以下三个层次口述：

> 1. **图像序列化**：ViT 将输入图像切分为互不重叠的固定网格块（例如 $16 \times 16$ 像素）。每个小块展平后经过可学习的线性映射，变成固定维度的向量（Patch Embedding），等价于 NLP 里的 Token 向量。
> 2. **前向传递过程**：在 Patch 向量序列最前面拼入一个可学习的 `[CLS]` 标记向量，作为全图信息的汇总通道；随后与可学习的 1D 位置编码逐元素相加，注入空间位置信息；最终送入标准的 Pre-LN Transformer Encoder 堆叠计算，取最后输出的 `[CLS]` 对应特征接入分类头。
> 3. **与 CNN 的关键差异**：CNN 具备强**归纳偏置**（局部连接与平移等变性），小数据上训练不容易过拟合，但浅层无法直接获得全局视野；ViT 没有这些硬编码约束，单层自注意力即可跨越任意两个图像块计算相关性。在小数据集（如 ImageNet-1K 原始标准训练）上 ViT 容易过拟合，但在数千万至数亿级别的大规模数据集预训练下，ViT 的表现上限与扩展潜力超过了传统 CNN。

---

## 图像 Token 化：从 (C, H, W) 到 (N, P²·C) 的切块与投影

![图像切块与线性投影过程](/images/content-rebuild/24a62555dc58ee25/fig-1.png)

Transformer 的自注意力层要求输入为二维张量 $(N, D)$，其中 $N$ 是序列长度，$D$ 是特征隐藏层维度。对于一张三通道图像 $x \in \mathbb{R}^{H \times W \times C}$，如果直接将每个独立像素作为一个 Token，分辨率为 $224 \times 224$ 时序列长度 $N = 224 \times 224 = 50,176$。自注意力矩阵计算的显存与计算复杂度与序列长度呈二次方关系 $O(N^2)$（参见 [Transformer 的计算复杂度为什么是瓶颈？](/fundamentals/transformer-arch/transformer-complexity.md)），单层运算就会耗尽显存。

ViT 采用非重叠切块方案（Patch Partition）解决了这个序列膨胀问题。

### 1. 切块与展平计算规则

设输入图像的高度为 $H$，宽度为 $W$，通道数为 $C$（如 RGB 图像 $C=3$）。预先设定正方形图像块的边长为 $P$（常见设定为 $16$ 或 $14$）。

* **切块数量 $N$**：
  $$N = \frac{H \cdot W}{P^2} = \left(\frac{H}{P}\right) \times \left(\frac{W}{P}\right)$$
* **单块物理像素数**：每个 Patch 包含 $P \times P \times C$ 个原始数值。
* **展平为向量**：将每个 Patch 展平成一维向量，得到序列矩阵 $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$。

### 2. 线性投影（Linear Projection）

每个切块拉直后的维度为 $P^2 \cdot C$。为了匹配 Transformer Encoder 内部固定的特征维度 $D$，ViT 使用一个可学习的线性映射矩阵 $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$，将每个 Patch 映射为目标向量：

$$z_{\text{patches}} = x_p E \quad \in \mathbb{R}^{N \times D}$$

### 3. 具体算例与工程实现

以处理 $224 \times 224$ 分辨率的 RGB 图像、取 $P=16$、模型隐层维度 $D=768$ 为例：
* 输入形状：$(C, H, W) = (3, 224, 224)$
* 序列长度 $N$：$(224 / 16) \times (224 / 16) = 14 \times 14 = 196$
* 单块展平维度 $P^2 \cdot C$：$16 \times 16 \times 3 = 768$
* 线性投影层权重尺寸为 $768 \times 768$
* 得到的 Patch Embedding 矩阵尺寸为 $(196, 768)$

在代码实现中，不需要显式写循环切片，而是利用一个步长等于卷积核大小的二维卷积层（Conv2d）等价完成：

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        # 当 stride == kernel_size 时，卷积窗口互不重叠，正好完成切块与线性映射
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # 输入 x 形状: (B, C, H, W)，如 (B, 3, 224, 224)
        x = self.proj(x)       # 输出形状: (B, D, H/P, W/P) -> (B, 768, 14, 14)
        x = x.flatten(2)       # 展平后两维: (B, D, N) -> (B, 768, 196)
        x = x.transpose(1, 2)  # 调整维度顺序: (B, N, D) -> (B, 196, 768)
        return x
```

<ReadMoreLock />

---

## [CLS] 标记与 1D 位置编码的角色机制

![CLS Token 与位置编码结构](/images/content-rebuild/24a62555dc58ee25/fig-2.png)

得到 $(N, D)$ 矩阵后，还缺少两个关键要素：全局特征聚合的固定载体，以及图像块的空间相对关系。

### 1. 为什么需要可学习的 [CLS] 标记？

ViT 沿用了 NLP 中 BERT 的设计思路（参考 [BERT 和 GPT 的核心区别是什么？](/fundamentals/transformer-arch/bert-vs-gpt.md)），在 Patch Embedding 序列的首位拼接一个可学习的特殊向量 $x_{\text{class}} \in \mathbb{R}^{1 \times D}$。输入序列长度从 $N$ 变为 $N + 1$：

$$z_0 = \left[ x_{\text{class}}; x_p^1 E; x_p^2 E; \dots; x_p^N E \right] \in \mathbb{R}^{(N + 1) \times D}$$

* **为什么采用 [CLS] 标记而不是全局平均池化（GAP）？**
  * 沿用 BERT 规范：设计首要考虑是最大化复用标准 Transformer Encoder 架构，使视觉任务在形式上与 NLP 序列分类保持一致。
  * 解耦的聚合载体：`[CLS]` 标记初始值与任何特定 Patch 空间区域解耦，在深层交互中由注意力权重自主从各图像块汇集语义，流程规范稳定。
  * 性能事实：ViT 论文消融实验表明，在配合适配的学习率时，GAP 与 `[CLS]` 的分类效果并无明显差距，选用 `[CLS]` 属于架构一致性取舍，而非 GAP 存在梯度平均导致无法训练的缺陷。

### 2. 1D 可学习位置编码（Positional Encoding）

自注意力机制核心是点积与加权求和（参见 [Self-Attention 中 Q、K、V 分别是什么意思？](/interview/llm-interview/attention-qkv.md)），对序列元素具有**置换等变性（Permutation Equivariance）**。若打乱 Patch 序列顺序，输出仅随之以完全相同的次序置换，模型本身将输入视为无序集合，无法识别图像块在二维网格中的几何相对位置。

ViT 使用形状为 $(N+1, D)$ 的一维可学习参数 $E_{\text{pos}}$，通过**逐元素相加**为输入补充空间坐标：

$$z_{\text{input}} = z_0 + E_{\text{pos}}$$

在原始论文实验中，作者对比了 1D 可学习位置编码、2D 显式网格位置编码以及正弦位置编码。实验表明，1D 编码与 2D 编码的最终表现几乎无差异。经过训练，模型的位置向量在余弦相似度上自发呈现出空间规律：每个位置向量与它在几何网格中相邻的上下左右位置相似度最高，同列、同行的向量也表现出明显的相关性。

---

## Transformer Encoder 内部数据流向

![Transformer Encoder 内部数据流向](/images/content-rebuild/24a62555dc58ee25/fig-3.png)

完成位置编码叠加后，张量 $z_{\text{input}} \in \mathbb{R}^{(N+1) \times D}$ 进入堆叠的 Transformer Encoder。ViT 采用的是 **Pre-LN** 架构（在进入自注意力或 MLP 子层之前先执行层归一化，详见 [LayerNorm 和 RMSNorm 有什么区别？](/interview/llm-interview/layernorm-rmsnorm.md)），包含 $L$ 个连续的 Block。

每个 Block 的计算公式为：

$$z'_\ell = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}, \quad \ell = 1 \dots L$$
$$z_\ell = \text{MLP}(\text{LN}(z'_\ell)) + z'_\ell, \quad \ell = 1 \dots L$$

其中：
* $\text{MSA}(\cdot)$ 是多头自注意力机制（Multi-Head Self-Attention，细节见 [Multi-Head Attention 为什么有效？](/fundamentals/transformer-arch/multi-head-attention.md)）。
* $\text{MLP}(\cdot)$ 由两层全连接层构成，中间隐藏层维度放大到 $4D$，激活函数采用 GELU。
* 经过 $L$ 层计算后，取出第 0 个位置的向量，经过 LayerNorm 后传入单层线性分类头：
  $$y = \text{LinearHead}(\text{LN}(z_L^0))$$

---

## CNN 与 ViT 的先验差异：归纳偏置的取舍

![CNN 强归纳偏置与 ViT 弱归纳偏置对比](/images/content-rebuild/24a62555dc58ee25/fig-4.png)

在面试中，CNN 与 ViT 的差异常被重点考察。两者的核心分野在于**归纳偏置（Inductive Bias）**——即网络架构为推断未知数据而提前硬编码的先验假设。

| 对比维度 | 卷积神经网络（CNN，如 ResNet） | Vision Transformer（ViT） |
| :--- | :--- | :--- |
| **核心计算单元** | 局部卷积核（Sliding Window） | 全局自注意力（Global Self-Attention） |
| **局部相关性（Locality）** | **强先验**：硬编码假设相邻像素具有关联，浅层只能看局部窗口 | **无硬编码**：第一层注意力即可跨任意 Patch 进行交互 |
| **平移等变性（Equivariance）**| **强先验**：卷积核权重全局共享，图像平移导致特征图等位平移 | **弱先验**：仅在 MLP 层处理各 Token 时具备平移等变性 |
| **有效感受野** | 随网络层数加深逐渐扩大（自底向上） | 理论上第一层就具备全图交互能力（全图到全图） |
| **对数据量的敏感度** | 依赖先验约束，小样本下收敛快、抗过拟合 | 缺乏硬编码先验，依赖更充分的数据自主学习模式 |

### 1. 局部相关性（Locality）
CNN 假定图像中关键的视觉模式（如边缘、纹理）都由相邻像素构成，因此卷积核通常限制在 $3 \times 3$ 或 $5 \times 5$。这种硬编码结构阻隔了浅层单元直接感知全图上下文，必须堆叠大量卷积层与下采样，才能让深层单元覆盖全局。

而在 ViT 中，自注意力矩阵直接计算序列中任意两个 Patch 之间的内积。哪怕在第一层网络中，位于左上角和右下角的 Patch 只要权重允许就能直接交换信息，结构上没有任何限制在局部窗口内的硬性约束。

### 2. 平移等变性（Translation Equivariance）
标准卷积具有平移等变性：如果输入图像中的目标向右平移 10 个像素，在没有边缘截断的前提下，卷积特征图的响应峰值也精确地向右平移 10 个位置。当结合全局池化时，CNN 能较为自然地处理图像中物体的位移。

ViT 输入中叠加上了绝对坐标绑定的 1D 位置编码，自注意力计算依赖绝对位置索引带来的特征区分。如果物体平移到另一批 Patch 对应的位置，输入向量发生了彻底改变，网络内部并不具备结构赋予的平移等变性，必须依赖多样化的训练样本自行拟合出平移不变的泛化能力。

---

## 高频追问与实战边界

![数据规模与模型准确率的 Scaling 交叉曲线](/images/content-rebuild/24a62555dc58ee25/fig-5.png)

### 追问 1：ViT 在 ImageNet-1K 上从头训练不如 ResNet，但在超大数据集上大幅反超，原因是什么？

在 ViT 原始论文中有一组典型结果：在仅包含约 130 万张图像的 ImageNet-1K 上采用标准配置从头训练时，ViT-Base 的分类准确率低于同等体量的 ResNet；但在 ImageNet-21k（1400 万张）或 JFT-300M（3 亿张私有数据集）上完成预训练后再微调时，ViT 的表现显著反超 ResNet。

**原因解释**：
1. **小数据阶段（CNN 占优）**：CNN 固有的局部性与平移等变性先验极大缩小了模型的假设空间，起到了天然正则化的作用，使得模型在没有海量样本覆盖各种视角与位移时依然能稳定泛化。而 ViT 缺乏这些先验，必须依靠训练样本从零学起“相邻像素构成物体”这类常识，在样本量不足时容易对训练集特征产生过拟合。
2. **大数据阶段（ViT 占优）**：硬编码的局部先验在数据规模极大时会成为性能瓶颈。局部卷积限制了复杂长程上下文的直接建模能力；而弱先验的 ViT 拥有更大的模型自由度，在海量数据供给下，自注意力能够灵活捕获任意尺度的全局关系，表现出更好的 Scaling 扩展规律。
3. **工程补充**：需要注意，后续的研究（如 DeiT）表明，通过引入更强的数据增强（CutMix、Mixup）和模型正则化策略，ViT 即使只在 ImageNet-1K 上从头训练也能取得与优秀 CNN 相当的精度。原论文的结论主要建立在基础训练配置对比之上。

### 追问 2：当微调阶段的分辨率高于预训练分辨率时，ViT 如何处理？

为了捕获更多图像细节，微调时通常会使用更高的分辨率（例如从预训练的 $224 \times 224$ 调整到微调的 $384 \times 384$）。

![位置编码 2D 双三次插值机制](/images/content-rebuild/24a62555dc58ee25/fig-6.png)

* **尺寸冲突**：保持 Patch 边长 $P=16$ 不变，当分辨率提升至 $384 \times 384$ 时，Patch 数量从 $(224/16)^2 = 196$ 增加至 $(384/16)^2 = 576$。然而预训练学到的位置编码张量 $E_{\text{pos}}$ 长度仅为 197（包含 `[CLS]`），无法直接相加。
* **处理方案**：采用 **2D 双三次插值（2D Bicubic Interpolation）**。
  1. 剔除序列开头的 `[CLS]` 对应位置向量，该向量不参与几何插值。
  2. 将剩余的 196 个一维位置向量还原为其在原始网格中的二维分布 $(14, 14, 768)$。
  3. 通过双三次插值算法，将二维网格上采样为目标网格尺寸 $(24, 24, 768)$。
  4. 将插值后的特征重新展平为 $(576, 768)$，并将原有的 `[CLS]` 位置向量重新拼回最前端，构成适配新分辨率的 $(577, 768)$ 位置编码矩阵。
* **落地要求**：插值操作虽然解决了维度对齐问题，但改变了预训练中学习到的绝对坐标数值。因此在分辨率变更后，必须进行少量 Epoch 的低学习率微调以校准位置编码。

### 追问 3：原生 ViT 为什么不适合直接用于目标检测和语义分割？

1. **缺乏原生多尺度金字塔**：目标检测与分割需要感知不同尺度的物体。传统 CNN 伴随多次下采样，天然拥有 $1/4, 1/8, 1/16, 1/32$ 的多尺度特征图金字塔（FPN）。原生 ViT 每一层保持恒定的序列长度 $N$，输出的是单一分辨率的列向量集合，缺少细粒度几何特征。
2. **高分辨率计算代价高**：检测与分割输入图像通常在 $1000 \times 1000$ 级别。若直接沿用 $16 \times 16$ 切块，序列长度将达到数千，$O(N^2)$ 的自注意力计算会让显存开销剧增。
3. **演进方向**：后续工作（如 Swin Transformer）引入了分层合并与基于移动窗口的局部注意力，在保留 Transformer 特性的同时找回了多尺度金字塔结构与线性的计算开销。

---

## 面试避坑指南

1. **不要混淆“序列长度”与“单块特征维度”**：
   * $224 \times 224$ 的 RGB 图像按 $16 \times 16$ 切块，切出来的块数是 $N = (224/16)^2 = 196$（加上 CLS 之后序列长度为 197）；而单块展平后的像素数是 $16 \times 16 \times 3 = 768$。不要把 Token 数量与向量维度混为一谈。
2. **自注意力机制不等于天然具备平移不变性**：
   * 自注意力本身对序列元素表现为置换等变性，不仅无法天然带来视觉任务所需的平移不变性，甚至无法感知网格相对顺序。叠加了绑定在绝对网格上的 1D 位置编码后，目标在图像中移动时，对应 Token 叠合的位置向量会彻底改变，平移泛化能力完全依赖后续的大量数据拟合。
3. **不要断言“ViT 模型中绝对不允许出现卷积”**：
   * ViT 原论文专门测试了 Hybrid（混合）模式，即先用 ResNet 提取浅层特征图，再切块输入 Transformer。此外在工程实现中，最标准、最高效的 Patch Embedding 实现方式就是一层二维卷积层。

---

## 推荐阅读与延伸练习

* 了解多模态图文对齐中基于 Transformer 的经典结构：[ViLT 模型是如何将 Transformer 应用于图像识别任务的](/interview/llm-interview/vilt-vision-transformer.md)
* 深入理解自注意力内部投影与缩放机制：[Multi-Head Attention 为什么有效？](/fundamentals/transformer-arch/multi-head-attention.md)
* 了解位置编码从绝对到相对/旋转编码的演化路径：[Transformer 的位置编码是怎样的？](/interview/llm-interview/transformer-positional-encoding.md)
