---
title: Transformer 在计算 attention 的时候使用的是点乘还是加法？请说明理由。
icon: mdi:multiplication
tag:
  - Transformer
  - Attention
  - 点积
---

# Transformer 在计算 attention 的时候使用的是点乘还是加法？请说明理由。

这道题考的是 Transformer attention 的底层实现细节，以及你对注意力机制进化历程的理解。面试官想看你不仅会用公式，还能讲清楚"为什么要这样设计"。

我从四个方面来讲：点乘的具体使用、点乘为什么比加法高效、缩放因子为什么不可或缺、注意力机制的历史演进。

---
## Transformer 用的是点乘，不是加法

先说结论：Transformer 在计算 attention 的时候用的是**点乘**，不是加法。

标准 attention 公式是这样的：


Attention(Q, K, V) = softmax(QK^T / √d_k) × V


这里 Q、K、V 是三个矩阵，分别代表 Query（查询）、Key（键）、Value（值）。

点乘操作发生在 QK^T 这一步。Q 的每一行和 K 的每一行做内积，得到一个标量。这个标量就代表了"查询和键的匹配程度"。

你可以把 Q、K、V 想象成图书馆的检索系统：

- Q 是你的问题："我想找一本关于深度学习的书"
- K 是书的索引标签："深度学习"、"机器学习"、"神经网络"
- V 是书的具体内容

点乘的结果越高，说明问题和索引越匹配，attention weight 就越大。系统就能找到最相关的书籍内容返回给你。

---
<ReadMoreLock />

## 点乘为什么比加法更高效？

好，既然点乘和加法都能算相似度，为什么 Transformer 选了点乘？

主要原因是**效率**。

点乘是纯粹的矩阵乘法，可以一次性在 GPU 上并行计算整个矩阵。假设 Q 和 K 的形状都是 (seq_len, d_k)，QK^T 就是一次性算出所有位置的相似度，GPU 的 tensor core 直接加速，毫秒级完成。

加法 attention 呢？它通常长这样：


score = v^T × tanh(W_q Q + W_k K)


这里有个激活函数 tanh。激活函数必须逐项计算，没法打包成矩阵乘法。GPU 的并行优势用不上，计算速度慢很多。

打个比方：

- **点乘**像批量发货。快递公司一辆车装一百个包裹，一次性全拉走。
- **加法**像单个打包。每个包裹单独称重、单独封箱，效率低得多。

除了计算效率，还有一个关键点：**可扩展性**。

点乘 attention 天然支持多头机制。你可以把 Q、K、V 切成多份，每份独立算 attention，最后拼接起来。如果是加法 attention，每一份都需要独立的神经网络参数，参数爆炸，难以训练。

---

## 缩放因子为什么不可或缺？

回到公式里的 / √d_k。

这叫**缩放因子**，别小看它，缺了它模型可能就跑不动了。

为什么要有这个？

当 d_k 很大的时候，Q 和 K 的点积结果会变得很大。假设向量维度是 512，每个元素的均值是 0、方差是 1，那么两个向量点积的均值是 0，但方差是 512。

结果就是点积结果分布非常分散，有些值特别大，有些值特别小。

经过 softmax 的时候会发生什么？

softmax 的公式是 exp(x_i) / Σ exp(x_j)。

如果某个值特别大，exp(大值) 会远远超过其他值，导致 softmax 输出几乎变成 one-hot——一个位置接近 1，其他位置接近 0。

这叫**梯度消失**。softmax 输出太平滑，梯度几乎为零，模型没法学习。

缩放因子把方差从 d_k 降到 1，点积结果的分布更稳定，softmax 输出的分布也更合理，梯度能正常流动。

你可以把它想象成调节音量旋钮：

- 太大声（方差过大）→ 失真，听不清细节
- 太小声（方差过小）→ 噪声淹没信号
- 刚刚好（缩放后）→ 各部分都能被清晰捕捉

所以除以 √d_k 不是随便写的，是为了让 attention score 的方差保持在合理范围。

---

## 从加法到点乘的历史演进

很多人不知道的是，最早的 attention 用的是加法，不是点乘。

2014 年，Bahdanau 等人在论文《Neural Machine Translation by Jointly Learning to Align and Translate》中提出了第一个 attention 机制，用的就是加法：


score(h_t, s_i) = v^T × tanh(W_h h_t + W_s s_i)


这篇论文解决了 seq2seq 模型"信息瓶颈"的问题： encoder 的所有信息压缩成一个固定向量，decoder 根本记不住。加入了 attention 之后，decoder 每一步都能"回头看"encoder 的所有隐藏状态，按需取用。

2017 年，Google 在《Attention is All You Need》中改了设计，用点乘替代加法：


score(Q, K) = QK^T / √d_k


理由也很直接：快、好训练、可并行。

技术迭代就像交通工具的进化。骑马能到终点，但汽车更快更稳。加法 attention 能 work，点乘 attention 更快更稳，所以大家后来都选了点乘。

---

## 面试怎么答

**基础版**（直接背）：

> Transformer 在计算 attention 时使用的是点乘，不是加法。标准公式是 softmax(QK^T / √d_k)V。点乘的优势在于计算效率高，天然支持 GPU 并行化，也更容易扩展到多头注意力机制。加分项是说明除以 √d_k 的作用——防止大维度向量点积过大导致 softmax 梯度消失。

**加分版**（扩展回答）：

> Transformer 使用点乘 attention，最早在 2014 年由 Bahdanau 等人提出加法 attention，2017 年 Google 在《Attention is All You Need》中改进为点乘。点乘的核心优势是计算效率——它是纯矩阵乘法，可充分利用 GPU 的并行计算能力。加法 attention 含激活函数 tanh，必须逐项计算，难以并行。此外，缩放因子 √d_k 必不可少，因为大维度向量点积方差会变大，导致 softmax 输出趋于 one-hot，梯度消失。除以 √d_k 后方差归一化，梯度稳定，模型能正常训练。

---

## 一句话总结

Transformer 用点乘是因为它快、好训练、可并行，而缩放因子 √d_k 是为了防止大维度下梯度消失。
