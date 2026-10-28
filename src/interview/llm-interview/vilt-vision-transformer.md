---
title: ViLT 模型是如何将 Transformer 应用于图像识别任务的
icon: mdi:image-multiple
tag:
  - ViLT
  - Vision Transformer
  - 图像识别
---

这篇文章考的是 ViT 和 ViLT 如何用 Transformer 的思路来处理图像。说白了，就是 CNN 统治视觉领域这么多年，Transformer 是怎么挤进来的，它凭什么能行。面试官想看你对视觉 Transformer 核心设计思路的理解。

我从四个方面来讲：图像怎么变成 token、位置编码怎么处理、ViLT 做了哪些创新、最后怎么组织答案。

---
## 一、把图像当句子来处理

先说 ViT（Vision Transformer）的基本思路。

CNN 处理图像靠的是卷积核在图像上滑动，提取局部特征。Transformer 靠的是自注意力，看的是全局关系。那怎么把图像塞进 Transformer？

**核心做法：把图像切成小块（patch），每个 patch 当成一个"词"。**

拿 224×224 的图像举例：

- 按 16×16 大小切分
- 一行 14 个 patch
- 一共 14×14 = 196 个 patch

每个 patch 先展平成一维向量（比如 16×16×3 = 768 维），然后通过一个线性投影层压缩成固定维度（比如 768 维）。

这 196 个向量就相当于 NLP 里的 196 个词 token。Transformer 读这 196 个向量，跟读一个句子没什么区别。

类比一下：就像把一张马赛克拼图拆成 196 块，每块用一个颜色向量来表示。Transformer 就能像读句子一样"读"这张图。

这个做法为什么重要？因为它让图像数据和 NLP 数据在表示形式上统一了。**之前 CNN 的那套局部特征提取逻辑，在 Transformer 这里被"图像分块 + 线性投影"替代了。**

---
<ReadMoreLock />

## 二、位置编码——Transformer 需要知道像素在哪

NLP 里，词的位置很重要。"我爱你"和"你爱我"意思完全相反。Transformer 本身不关注顺序，所以需要额外加位置信息。

图像也一样。patch 1 和 patch 2 谁在左谁在上，Transformer 不知道。

**解决方案：可学习的 1D 位置编码。**

具体做法是为每个 patch 的向量加上一个同等维度的位置向量。这个位置向量是模型学出来的，它能表达"这个 patch 在图像的哪个位置"。

加上位置编码后，196 个 patch 向量变成了带有位置信息的新向量，进入 Transformer Encoder。

还有一个细节：[CLS] token。

这是从 BERT 借鉴来的做法。在输入序列最前面加一个特殊的 [CLS] token，它的输出向量用来做最终分类。

为什么不用所有 patch 的输出做分类？因为 [CLS] 经过多层自注意力的交互，汇聚了全局信息，更适合做整体判断。

**总结这个板块：位置编码解决"在哪"的问题，[CLS] token 解决"输出什么"的问题。**

---

## 三、ViLT 的创新——统一模态，告别 CNN

讲完 ViT，再看 ViLT（Vision-and-Language Transformer）。

ViLT 是把视觉和语言统一到 Transformer 架构里的尝试。它的目标是用一个模型同时处理图像和文本。

**传统视觉-语言预训练（VLP）的做法：**

- 用 CNN 提取图像特征
- 用 Transformer 处理文本
- 两个模态的交互靠后面的 cross-attention

问题在哪？CNN 提取图像特征的计算量非常大，往往占了整个模型的大部分参数和算力。

**ViLT 的核心创新：用线性投影替代 CNN。**

图像不通过 CNN 了，直接通过一个简单的线性层做 patch embedding，文本通过 word embedding，两边拼成一个序列送进 Transformer。

这样做有几个好处：

1. **速度提升 10-60 倍**。省掉了 CNN 那个大计算量。
2. **参数量大幅减少**。CNN 那套卷积层不用了。
3. **无需区域监督**。传统方法需要 bounding box 标注，ViLT 不需要。
4. **真正的统一架构**。图像和文本在同一个表示空间里，用同一个 Transformer 处理。

类比一下：

- 传统方法像"专业翻译器（CNN）+ 通用翻译（Transformer）"，各管各的。
- ViLT 就是"一个翻译器搞定一切"，视觉和语言模态平等处理。

**ViLT 的意义在于：它证明了视觉特征提取不需要复杂的 CNN，一个简单的线性投影加 Transformer 就够了。**

---

## 四、面试怎么答

### 基础版（直接背）

ViT 将图像划分为 16×16 的 patch，每个 patch 展平后通过线性投影变成 token，224×224 图像得到 196 个 patch token。为每个 token 添加可学习的 1D 位置编码，前面加一个 [CLS] token，整组输入标准 Transformer Encoder，[CLS] 的输出向量用于分类任务。

ViLT 在此基础上更进一步，用线性投影替代 CNN 提取视觉特征，图像和文本都通过 embedding 直接送入 Transformer，实现真正的多模态统一处理。这种设计大幅降低了计算量，无需区域监督，参数效率更高。

### 加分版（加深度）

这道题的核心是理解 CNN 的局部卷积和 Transformer 的全局自注意力在图像处理上的差异。ViT 的关键洞察是把图像切成离散的 patch，每个 patch 独立 embedding，这样就绕过了 CNN 的局部感受野限制，用自注意力来建模全局关系。

ViLT 的意义更在于解决了一个效率问题：传统 VLP 方案里，CNN 提取视觉特征的计算量远超过模态交互的部分，导致视觉端成了瓶颈。ViLT 用简单的线性投影替代 CNN，让视觉和语言模态在同一个 Transformer 里平等交互，这是架构层面的简化，也是范式层面的突破。

往深了说，这反映了 Transformer 统一视觉和语言建模的大趋势：NLP 那边 BERT、GPT 已经验证了 Transformer 的能力，CV 这边 ViT、ViLT 正在把这个能力迁移过来。

---

## 一句话总结

**ViT 把图像切成 patch 做 tokenization，ViLT 用线性投影替代 CNN 实现多模态统一，核心都是让 Transformer 用处理序列的方式处理图像。**
