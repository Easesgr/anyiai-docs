---
title: Transformer 架构是如何工作的？专栏导读与学习路径
icon: mdi:chart-box-outline
tag:
- Transformer架构
- 面试题
- 深度学习
description: 梳理 Transformer 从点积注意力、位置编码到整体架构的学习路径，通过小例子解释 QKV 投影、因果掩码与复杂度瓶颈，提供清晰的实现验证流程。
---

![Transformer 专栏全局导读图示](/images/content-rebuild/38f30a2b86fc67f0/cover.png)

学习 Transformer 架构时，很多开发者容易把**单层注意力的矩阵数学计算**和**完整模型的结构设计**混在一起。如果只背诵公式，一旦遇到自回归生成时的长文本显存占用、掩码如何保证因果顺序、或者为什么模型需要 KV Cache 等工程问题，就很难看清背后的原因。

Transformer 的核心结论很简单：**它抛弃了循环神经网络（RNN）按时间步顺序传递隐藏状态的做法，也舍弃了卷积神经网络（CNN）局部滑动窗口的限制，完全依靠注意力机制来建立输入序列中任意两个词之间的直接联系。**

我们以一个简单的三词序列输入为例：`["机", "器", "学"]`，目标是预测下一个字`"习"`。
- 在传统的 RNN 中，模型必须先处理`"机"`，生成隐藏状态传递给`"器"`，再传递给`"学"`，这个计算过程是严格串行的，无法并行加速；
- 在 Transformer 中，`"机"`、`"器"`、`"学"`对应的三个词向量被打包成一个矩阵同时输入，通过一次矩阵乘法，每个字就能同时计算出它与其它所有字的关联权重。

为了帮助大家系统掌握这一架构，本专栏设计了一条由浅入深的理解与验证路径。

---

## 一、 专栏知识体系的三阶演进

![Transformer 专栏知识体系的三阶演化路径](/images/content-rebuild/38f30a2b86fc67f0/fig-1.png)

本专栏收录的 8 篇核心文章按问题推进顺序编排，分为三个学习阶段：

### 阶段一：核心注意力算子的数学机制
在没有循环单元的情况下，模型如何计算一个序列内部各个词之间的相关性？
- [Transformer 相比 RNN 和 CNN 有什么优势？](/fundamentals/transformer-arch/transformer-vs-rnn-cnn.md)：对比 RNN 的串行瓶颈与 CNN 感受野随层数受限的问题，说明自注意力机制如何以一步矩阵运算跨越任意序列距离，并在 GPU 上实现高度并行的训练。
- [Self-Attention 中 Q、K、V 分别是什么意思？](/fundamentals/transformer-arch/q-k-v-explained.md)：通过参数化投影矩阵把同一个输入词向量分别映射为查询向量（Query）、键向量（Key）和值向量（Value），理解寻址打分与特征加权聚合的具体过程。
- [为什么 Self-Attention 要除以根号 dk？](/fundamentals/transformer-arch/why-divide-sqrt-dk.md)：推导当向量维度 $d_k$ 变大时，点积结果方差随之膨胀的数学原因，解释缩放因子如何防止 Softmax 函数进入梯度趋近于零的饱和区。

### 阶段二：表达能力拓展与时序约束
单纯的点积加权运算对输入词的排列顺序完全不敏感，且单一注意力头容易把权重平均化，需要进行功能补充：
- [Multi-Head Attention 为什么有效？](/fundamentals/transformer-arch/multi-head-attention.md)：解释多头拆分如何在保持总参数量和计算量基本不变的前提下，让模型在不同的低维投影子空间里同时关注不同的语义信息（例如一个头关注语法结构，另一个头关注代词指代）。
- [Transformer 为什么需要位置编码？](/fundamentals/transformer-arch/positional-encoding.md)：论证自注意力对输入顺序的置换等价性（即打乱词序输出向量只跟着打乱对应位置，内部关联没有改变），对比正弦绝对位置编码与现代大模型常用的旋转位置编码（RoPE）。

### 阶段三：宏观网络结构与计算性能边界
把注意力模块、前馈网络、残差连接与层归一化拼接成完整模型后，不同的组合方式决定了模型的适用场景与工程开销：
- [Encoder-only、Decoder-only 和 Encoder-Decoder 架构有什么区别？](/fundamentals/transformer-arch/encoder-decoder-variants.md)：对比双向全可见注意力、因果单向注意力与交叉注意力的适用边界，说明为什么主流大语言模型更偏向 Decoder-only 架构。
- [BERT 和 GPT 的核心区别是什么？](/fundamentals/transformer-arch/bert-vs-gpt.md)：对比掩码语言建模（填空任务）与自回归下一个 Token 预测（因果生成）对模型理解和生成能力带来的不同影响。
- [Transformer 的计算复杂度为什么是瓶颈？](/fundamentals/transformer-arch/transformer-complexity.md)：分析训练期间自注意力矩阵带来的 $O(N^2)$ 显存和算术开销，以及推理阶段为了避免重复计算而引入 KV Cache 产生的显存带宽占用。

---

## 二、 概念依赖关系：区分算子与网络结构

理解 Transformer 的关键，在于区分**单层算子的数学特性**与**完整模型的数据流控制**：

![单层注意力数学推导与全网拓扑数据流的维度对比](/images/content-rebuild/38f30a2b86fc67f0/fig-2.png)

1. **局部算子层**：核心计算公式 $\text{Softmax}(QK^T / \sqrt{d_k})V$ 是一个加权求和算子。输入矩阵的行顺序打乱，输出结果的行顺序也会完全跟着打乱，算子本身不知道哪个词在前面、哪个词在后面。
2. **整体网络层**：
   - **顺序感知**不是由自注意力算子自身学会的，而是依靠在最底层词嵌入上**直接相加位置编码**引入的；
   - **单向生成约束**（不能看到未来的词）也不是算子天然具备的特性，而是在计算 Softmax 之前，通过人工将未来位置的注意力分数加上一个极大的负数（例如 $-\infty$），强制让未来的注意力权重变为 0 实现的。

在阅读后续代码和原理时，必须将“算子的运算逻辑”与“架构施加的外部约束”分开看待。

---

## 三、 从零实现 Transformer 的验证步骤

![从零构建最简Transformer的逐步代码验证闭环](/images/content-rebuild/38f30a2b86fc67f0/fig-3.png)

自己动手编写并验证 Transformer 模块时，建议遵循以下四个具体步骤：

### 步骤 1：验证缩放点积的数值稳定性
随机生成均值为 0、方差为 1 的测试张量 $Q, K \in \mathbb{R}^{B \times N \times d_k}$（例如设 $d_k=64$）。验证：
- 未除以 $\sqrt{d_k}$ 时，$Q K^T$ 矩阵元素的方差接近 $d_k$（约为 64）；
- 除以 $\sqrt{d_k}$ 后，元素方差稳定回落到接近 1.0；
- 观察经过 Softmax 计算后，梯度的分布是否处于正常区间，避免输入方差过大进入饱和区而导致梯度消失。

### 步骤 2：验证多头拆分与张量重排
输入一个形状为 `(batch_size, seq_len, d_model)` 的张量，通过线性投影映射为 Q、K、V。
- 使用 `view` 和 `transpose` 将其转换为 `(batch_size, num_heads, seq_len, d_k)`；
- 验证各注意力头能够利用一次批量矩阵乘法（BMM）并行计算；
- 检查最终多头拼接（Concat）并经输出投影后，张量形状是否严格恢复为 `(batch_size, seq_len, d_model)`。

### 步骤 3：验证位置敏感性与因果掩码
- **位置编码测试**：输入打乱顺序的文本张量。如果未加位置编码，输出张量仅为原输出的行置换；加入位置编码后，对应位置的向量数值应发生真实变化。
- **因果掩码测试**：构建上三角掩码矩阵，将未来位置替换为浮点负无穷 `float('-inf')`（在 FP16 等低精度下需使用 `-float('inf')` 或当前精度的最小值 `torch.finfo(dtype).min`，避免直接写入如 `-1e9` 这种超出负数有限表示范围而发生溢出的常数）。经 Softmax 后，检查每一行中未来位置对应的权重值是否严格为 0。

### 步骤 4：验证残差连接与层归一化
将上述算子与前馈全连接网络（FFN）组合，包裹上残差分支与 LayerNorm：$\text{LayerNorm}(x + \text{Sublayer}(x))$。
- 检验在前向传播时，每层输出张量的均值是否被校正到接近 0，方差接近 1；
- 检验反向传播时，残差分支能否让梯度无损地反向传递，减轻梯度消失风险。

---

## 四、 模块学习自查标准

读完本专栏后，你可以用以下具体问题自查是否真正掌握：

1. 能够说出为什么点积方差会随着向量维度 $d_k$ 膨胀，并写出除以 $\sqrt{d_k}$ 的推导过程；
2. 能够写出单头与多头注意力在 PyTorch 中使用 `view` 和 `transpose` 进行形状转换的代码维度变化；
3. 能够解释自回归解码时为什么需要缓存之前步骤的 K 和 V，并根据显卡参数计算 KV Cache 的显存开销；
4. 能够明确指出 Encoder-only（如 BERT）与 Decoder-only（如 GPT）在注意力掩码矩阵形状上的根本区别。

建议读者按顺序学习，直接前往第一篇：[Transformer 相比 RNN 和 CNN 有什么优势？](/fundamentals/transformer-arch/transformer-vs-rnn-cnn.md)。
