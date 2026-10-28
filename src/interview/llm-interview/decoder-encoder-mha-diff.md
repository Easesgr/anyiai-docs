---
title: Transformer 中，Decoder 阶段的多头自注意力和 Encoder 阶段的多头自注意力是相同的吗？
icon: mdi:compare
tag:
  - Transformer
  - 多头注意力
  - Encoder-Decoder
---

# Transformer 中，Decoder 阶段的多头自注意力和 Encoder 阶段的多头自注意力是相同的吗？

这道题考的是 Encoder 和 Decoder 自注意力的本质区别——**双向全局注意力 vs 单向因果注意力**，以及 Decoder 额外拥有的交叉注意力机制。

我从四个方面来讲：先给结论、Encoder 自注意力的工作方式、Decoder 自注意力的因果掩码、Decoder 的交叉注意力层。

---
## 1. 先给结论——它们完全不同

很多人以为 Decoder 的自注意力和 Encoder 差不多，只是位置不同。这个理解是错的。

Encoder 和 Decoder 的多头自注意力**完全不同**，主要体现在两点：

**第一，注意力方向不同。** Encoder 是双向的，每个位置可以看到序列中的所有位置。Decoder 是单向的，每个位置只能看到当前位置及其之前的内容。

**第二，有没有掩码不同。** Encoder 没有掩码，所有 token 可以自由attend。Decoder 必须通过掩码（mask）把"未来的信息"遮住。

**第三，Decoder 比 Encoder 多一层。** Decoder 还有一个交叉注意力层，用于融合 Encoder 的输出信息。Encoder 没有这一层。

简单记：Encoder 是"看全部"，Decoder 是"只看前面"，还多一个"抄作业"的层。

---
<ReadMoreLock />

## 2. Encoder 多头自注意力——双向全局建模

Encoder 的自注意力，每个 token 可以关注序列中的任意位置。

打个比方：你读一篇文章，可以随时往前翻，也可以往后跳着看。Encoder 处理输入序列时也是这样，所有 token 互相"能看到"，形成双向的全局依赖关系。

具体怎么实现的？

假设输入序列是 "[CLS] 我 爱 机 器 学 习 [SEP]"，经过 embedding 和位置编码后进入多头自注意力层。

计算时，每个 token 都会生成 Query、Key、Value 三个向量。所有 token 的 Query 和所有 token 的 Key 做点积，得到注意力分数。这个分数矩阵是**对称的**（如果忽略具体向量值的差异）。

第 5 个 token "学" 可以关注第 1 个 token "[CLS]"，也可以关注第 6 个 token "习"。没有任何限制。

这种双向注意力让 Encoder 特别适合理解任务，比如文本分类、命名实体识别、句子关系判断。BERT 就是典型的 Encoder-only 模型。

---

## 3. Decoder 多头自注意力——因果掩码机制

Decoder 的自注意力就没这么自由了。

它有严格的条件：**每个位置只能关注当前位置及其之前的位置，不能看到未来的信息。**

为什么？因为 Decoder 是自回归生成——它一个词一个词往外蹦。写第 5 个词的时候，你不能偷看第 6、7、8 个词是什么，那些还没生成呢。

这种机制叫**因果掩码（Causal Mask）**。

具体怎么实现的？

Decoder 内部维护一个 mask 矩阵，长这样：


位置 1: [0, -inf, -inf, -inf, -inf]
位置 2: [0,  0,  -inf, -inf, -inf]
位置 3: [0,  0,   0,  -inf, -inf]
位置 4: [0,  0,   0,   0,  -inf]
位置 5: [0,  0,   0,   0,   0]


0 表示可以attend，-inf 表示不能attend（softmax 后权重变成 0）。

把注意力分数矩阵加上这个 mask，再过 softmax，未来位置的权重就全变成 0 了。

所以 GPT 这种 Decoder-only 的模型，生成文本时就是一个 token 一个 token 往外蹦，每次都只能基于已经生成的内容来预测下一个 token。

---

## 4. Decoder 还有一个交叉注意力层

Decoder 比 Encoder 多一个东西，叫**交叉注意力（Cross-Attention）**，也叫编码器-解码器注意力。

这一层不是 Decoder 自己和自己玩，而是 Decoder 去"看"Encoder 的输出。

怎么实现的？

这一层的 Query 来自 Decoder 当前层，Key 和 Value 来自 Encoder 的最终输出。

你可以理解成：Decoder 在生成每个 token 时，会回头查一下 Encoder 处理过的原序列，找到相关的部分。

最典型的例子是翻译。

输入中文"我爱机器学习"，Encoder 先把它编码好。然后 Decoder 开始生成英文 "I love machine learning"。

生成 "love" 的时候，Decoder 会通过交叉注意力去看 Encoder 的输出，找到"爱"对应的那部分信息。

所以 Decoder 其实有两层注意力：
- **掩码自注意力（Masked Self-Attention）**：看上文的自己
- **交叉注意力（Cross-Attention）**：看 Encoder 的结果

Encoder 只有自注意力，没有交叉注意力。

---

## 面试怎么答

**基础版（直接背）：**

Encoder 和 Decoder 的多头自注意力完全不同。

Encoder 的自注意力是双向的，每个 token 可以看到序列中的所有位置，建模全局依赖关系，没有掩码。

Decoder 的自注意力是单向的，通过因果掩码确保每个位置只能关注当前位置及之前的位置，防止信息泄露。这是自回归生成模型的核心机制。

此外，Decoder 还有一个交叉注意力层，Query 来自 Decoder，Key 和 Value 来自 Encoder 输出，用于融合源序列信息，实现编码器和解码器的交互。

**加分版（加上细节）：**

基础版的内容，加上这些：

从数学实现上，Decoder 在计算注意力分数后，会加一个下三角的 mask 矩阵，把未来位置设为负无穷，softmax 后权重归零，确保第 i 个位置只attend 到 ≤i 的位置。

实际应用中，BERT 是纯 Encoder，GPT 是纯 Decoder，T5 和 BART 是 Encoder-Decoder 架构。不同任务选不同架构：理解任务用 Encoder，生成任务用 Decoder，序列到序列任务用完整的 Encoder-Decoder。

---

## 一句话总结

**Encoder 自注意力是双向全局的，Decoder 自注意力是单向因果的（带掩码），且 Decoder 还多一层交叉注意力来融合 Encoder 的输出。**
