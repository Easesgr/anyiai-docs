---
title: Transformer 为什么采用 Layer Normalization 而不是 Batch Normalization
icon: mdi:layers
tag:
  - Transformer
  - LayerNorm
  - BatchNorm
---

# Transformer 为什么采用 Layer Normalization 而不是 Batch Normalization

这道题考的是归一化方法在不同场景下的适用性，以及 Transformer 为什么选择了 LN 而非 BN。核心能力是理解**归一化方向的本质差异**如何影响模型对输入数据的处理方式。

我从四个方面来讲：先说归一化是什么，再说 BN 和 LN 的核心差异，然后讲 Transformer 为什么"嫌弃"BN，最后说 Pre-LN 和 Post-LN 的工程实践。

---
## 1. 先理解什么是归一化——给数据"瘦过身"

归一化本质上是对数据分布做标准化处理。

神经网络训练难，主要是因为每层输入数据的分布会不断变化——这叫**内部协变量偏移（Internal Covariate Shift）**。数据分布乱，梯度就不稳定，收敛就慢。

归一化就是来解决这个问题的：通过把数据变换成均值 0、方差 1 的标准分布，让梯度流动更顺畅。

BN 和 LN 都是归一化，但它们"统计的维度"不一样。

想象把一群身高体重各异的人标准化：
- **BN** 相当于跨所有人统计——把所有样本拉到一起，算整体的平均身高、平均体重
- **LN** 相当于每个人自己统计——只看单个人内部，把他的身高体重都调整到标准范围

这两种方式在处理思路上完全不同，下一节会详细讲。

---
<ReadMoreLock />

## 2. BN vs LN 的核心差异——统计方向决定适用范围

这是本题的核心，必须搞清楚。

**Batch Normalization** 在 batch 方向（N×H×W）统计均值和方差。

什么意思？假设你有一个 batch 的数据，比如 32 张图片，每张图片有多个通道。BN 会把这 32 张图片对应位置的像素放在一起，算出一个均值。换句话说，BN 依赖的是** batch 内所有样本的统计量**。

**Layer Normalization** 在 layer 方向（C×H×W）统计均值和方差。

LN 做的事情是：对单个样本的所有通道、所有位置算均值和方差。它不依赖 batch 里的其他样本，统计的是**单个样本内部**的分布。

用一个更直观的例子：

> 评估一个班级的成绩。BN 要集齐全班的卷子才能打分，LN 只需要看单个人的卷子。

BN 的问题在于：**它对 batch size 很敏感**。

当 batch 较小（比如 4、8）时，统计均值和方差的样本太少，结果就不稳定。今天 batch 里碰巧几个样本分布偏了，BN 的统计量就失真了。

LN 完全不依赖 batch size，对单个样本内部做归一化，所以天然稳定。

---

## 3. Transformer 为何"嫌弃"BN——序列模型的结构特点

Transformer 是为序列任务设计的，这带来了几个 BN 很难处理的问题。

**第一，序列长度可变。**

翻译任务里，"你好"和"今天天气很好"长度完全不同。BN 在 batch 方向统计，意味着它需要 batch 内的样本在维度上对齐。但变长序列无法直接对齐，要么 padding，要么动态 batch。

如果 batch 内句子长短差异很大，BN 的统计量就无法代表整体分布——短句 padding 后全是零，长句信息量完全不同。

**第二，batch size 动态变化。**

训练时可能用 batch size 32，推理时用 batch size 1。BN 依赖的 batch 统计量在不同 batch size 下差异很大，小 batch 时性能显著下降。

有研究指出，**batch size 临界值大约是 16**，小于这个值 BN 的效果会显著下降。

**第三，Transformer 的并行计算特性。**

Self-attention 的核心是让序列中每个位置都能"看到"其他位置。如果归一化依赖 batch 统计量，就会破坏这种并行特性。LN 对单个样本独立处理，完全适配 self-attention 的并行计算。

看一个具体场景：

> 翻译任务 batch：[ "你好", "今天天气很好", "Machine Learning" ]
> - "你好"长度 2，padding 到最大长度
> - "今天天气很好"长度 6，padding 到最大长度
> - "Machine Learning"长度 2，padding 到最大长度
>
> BN 统计均值时，每个位置会混合三个句子对应位置的信息。但如果某个 batch 恰好都是短句，统计量就会偏移。

LN 只看单个句子内部，完全不受这种干扰。

---

## 4. 实战角度——为什么 Pre-LN 逐渐取代 Post-LN

这是工程实践中的重要改进，面试时提到会很加分。

Transformer 原始论文用的是 **Post-LN** 结构：先做 Multi-Head Attention 或 FFN，然后做残差连接，最后做 Layer Normalization。

但研究发现 Post-LN 训练不稳定——残差连接之后直接做 LN，梯度在早期层会比较大，导致训练初期容易梯度爆炸。

**Pre-LN** 的做法是：在每个子层内部做 Layer Normalization。具体来说：

- 子层输入先进 LN
- 再做 Attention 或 FFN
- 最后残差连接

Pre-LN 结构示意：

Post-LN: x -> Attention -> +x -> LN -> ...
Pre-LN:  x -> LN -> Attention -> +x -> ...


类比一下：

> 装修房子。Post-LN 相当于最后统一刷墙，每间房单独装修后再串联；Pre-LN 相当于每间房单独装修、单独刷墙，最后再串联起来。后者施工更稳定，不容易出问题。

现在的 Transformer 实现基本都默认用 Pre-LN，训练更稳定，效果也更好。

---

## 面试怎么答

**基础版（100-150字）：**

> Transformer 用 Layer Normalization 而不是 Batch Normalization，主要有两个原因。第一，BN 在 batch 方向统计均值和方差，对 batch size 敏感，而 Transformer 训练时 batch size 可能较小。第二，Transformer 输入序列长度可变，不同句子 padding 后长度一致但内容不同，BN 的统计量无法正确反映数据分布。LN 在单个样本内部归一化，不依赖 batch 统计量，天然适配变长序列和 self-attention 的并行计算特性。

**加分版（200-250字）：**

> 本质区别在于归一化的方向。BN 在 batch 维度统计，要求 batch 内样本分布一致；LN 在单个样本内部统计，不依赖 batch。
>
> Transformer 不用 BN 的原因：1）序列长度可变，batch 内样本实际长度不一致，BN 统计不准确；2）小 batch 时 BN 统计不稳定，batch size 低于 16 时效果显著下降；3）LN 更适配 self-attention 的并行特性。
>
> 另外还有个工程细节：原始 Transformer 用 Post-LN，训练不稳定。后来发现 Pre-LN（在子层内部做 LN）训练更稳定，现在基本都用 Pre-LN 了。
>
> 如果面试官追问，可以提一下 Group Normalization——它是 BN 和 LN 的折中，把 channel 分组后归一化，结合了两种方法的优势。

---

## 一句话总结

Transformer 用 LN 而非 BN，根本原因是 **LN 不依赖 batch 统计量，天然适配变长序列和小 batch 场景**，而 BN 对 batch size 敏感，无法处理序列长度变化的输入。
