---
title: 为什么 Transformer 采用多头注意力机制？
icon: mdi:head-cog
tag:
- Transformer
- 多头注意力
- 注意力机制
description: 解析 Transformer 采用多头注意力机制（MHA）的核心原因，通过贯穿例句对比单头与多头的表征差异，推导计算量与参数量等价性，并分析头数选择的工程边界与追问考点。
---

在面试中被问及“为什么 Transformer 要使用多头注意力（Multi-Head Attention, MHA）而不是单头注意力”时，不少候选人往往会给出“为了提取更多特征”或“类似卷积神经网络的多通道”等笼统回答。这种回答虽有直觉支撑，但未能说明其背后的数学约束、表征瓶颈以及算力等价性。

本文先给出可在面试中口述的精炼结论，再结合一个贯穿全篇的具体例句，详细分析单头注意力的表征限制、多头注意力的计算过程、参数与计算量等价推导，以及头数选择的工程边界。

![多头注意力机制与子空间投影全景](/images/content-rebuild/1f6d96d3f719f1b4/cover.png)

---

## 30 秒口述结论

如果在面试中被问到该问题，建议按照“**解决加权平均平滑瓶颈**、**多子空间正交表征**、**计算量与参数量等价**”三层逻辑递进作答：

> 1. **避免加权平均抹平多维关系**：单头注意力机制在全序列上只生成一组归一化的注意力权重，对所有 Value 向量做加权求和。当同一个词在上下文中同时承担多种语法和语义角色时，单一分布会将其强行折中。Transformer 原论文明确指出，多头注意力正是为了抵消这一“有效分辨率下降”的问题。
> 2. **多子空间并行建模**：多头注意力通过多组独立的投影矩阵，将 Query、Key、Value 投影到不同的低维子空间。各头可以在各自的子空间内并行关注不同维度的关联（如局部的修饰搭配、跨从句的代词指代、结构对齐等），最后通过拼接与输出投影矩阵 $W^O$ 融合信息。
> 3. **计算与参数量保持等价**：在具体设计上，每个头的维度被缩小为 $d_k = d_v = d_{\text{model}} / h$。将一个大头拆分为 $h$ 个小头后，核心矩阵乘法的参数量和理论浮点运算量（FLOPs）与单一大头完全一致，模型并未因增加头数而付出额外的核心计算代价。

---

## 贯穿例子：单头注意力的表征局限

为了看清单头注意力的瓶颈，我们使用自然语言处理中经典的歧义与指代例句进行分析：

$$\text{"The animal didn't cross the street because it was too tired."}$$

当模型处理句子中的代词 **"it"** 时，为了准确理解语义，它至少需要同时完成两类不同性质的特征抽取：
1. **长距离指代消解**："it" 跨越多个词指向句首的核心名词短语 **"The animal"**（确定主体是谁）；
2. **近距离谓词与状态搭配**："it" 与紧邻的系动词 **"was"** 和形容词 **"tired"** 构成主系表结构（确定主体的状态是什么）。

![单头注意力因均值池化抹平多种语法与指代关系的困境](/images/content-rebuild/1f6d96d3f719f1b4/fig-1.png)

### 单头注意力是如何处理该句的？

在标准缩放点积注意力中，给定输入序列经过线性映射后得到的矩阵 $Q, K, V \in \mathbb{R}^{N \times d_{\text{model}}}$，单头注意力的输出为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_{\text{model}}}}\right)V$$

设注意力权重矩阵为 $A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_{\text{model}}}}\right) \in \mathbb{R}^{N \times N}$。对于词 "it"（假设对应位置为 $i$），其更新后的向量表征 $\mathbf{z}_i$ 是序列中所有位置 Value 向量的凸组合（Convex Combination）：

$$\mathbf{z}_i = \sum_{j=1}^N A_{ij} \mathbf{v}_j, \quad \text{其中 } \sum_{j=1}^N A_{ij} = 1, \ A_{ij} \ge 0$$

由于 $\sum_j A_{ij} = 1$ 的归一化限制，单头注意力只能生成**唯一**一组概率分布：
- 如果模型把较大概率分给 "animal"（例如 $A_{i,\text{animal}} = 0.6$），留给 "tired" 的概率就只剩下 $0.4$ 或更少；
- 最终算出的更新向量 $\mathbf{z}_i$ 是名词向量 $\mathbf{v}_{\text{animal}}$ 和状态向量 $\mathbf{v}_{\text{tired}}$ 的线性加权混合。

这种加权平均操作带来了一个必然的副作用：不同性质的语义特征被强行揉合成了一个中间折中向量。Vaswani 等人在原始论文 *Attention Is All You Need* 中对该现象有明确陈述：

> *“...at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention...”*
> （……代价是由于对注意力加权位置做平均而降低了有效分辨率，我们通过多头注意力机制来抵消这一影响……）

单头注意力在面对多重语义关系时，无法在同一层内同时保留各自独立的清晰特征。

<ReadMoreLock />

---

## 多头注意力机制的计算过程

多头注意力通过将特征维度切分到多个独立的低维子空间，让不同的头在各自空间内独立计算注意力分布。

![多头注意力的分头线性变换、并行注意力与拼接投影数据流](/images/content-rebuild/1f6d96d3f719f1b4/fig-2.png)

### 1. 结构与公式定义

设输入隐藏维度为 $d_{\text{model}}$，头的数量为 $h$。每个注意头的内部计算维度定义为：

$$d_k = d_v = \frac{d_{\text{model}}}{h}$$

对于每一个注意头 $i \in \{1, 2, \dots, h\}$，使用独立的投影参数矩阵将输入 $X \in \mathbb{R}^{N \times d_{\text{model}}}$ 投影到第 $i$ 个子空间：

$$Q_i = X W_i^Q, \quad K_i = X W_i^K, \quad V_i = X W_i^V$$

其中投影矩阵的尺寸分别为 $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$。

每个头在各自的子空间内独立执行缩放点积注意力：

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right)V_i \in \mathbb{R}^{N \times d_v}$$

在前面的例句中：
- **Head 1** 可以在其子空间中将绝大部分注意力权重（如 $0.9$）分配给 "animal"，专门捕获实体指代信息；
- **Head 2** 可以在其子空间中将绝大部分注意力权重（如 $0.85$）分配给 "tired"，专门捕获谓词状态信息；
- 其余的头还可以分别关注句式连接词（如 "because"）或局部标点。

最后，将所有 $h$ 个头的输出张量在最后一个维度上进行拼接（Concatenation），并通过一个输出投影矩阵 $W^O \in \mathbb{R}^{(h \cdot d_v) \times d_{\text{model}}}$（即 $\mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$）进行线性混合：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W^O$$

---

## 计算量与参数量严格等价推导

面试中的常见困惑是：“拆成 $h$ 个头并行计算，计算量和参数量是不是翻了 $h$ 倍？”答案是否定的。

我们假设序列长度为 $N$，模型维度为 $d_{\text{model}}$，头数为 $h$，每个头维度 $d_k = d_v = d_{\text{model}} / h$。按照标准矩阵乘法规则，矩阵乘法 $(M \times K) \times (K \times P)$ 包含 $M \times K \times P$ 次乘法与约 $M \times K \times P$ 次加法（每个输出元素需 $K$ 次乘法与 $K-1$ 次加法），浮点运算量通常计为 $2MKP$ FLOPs（这里忽略 Softmax 等相对较小的低阶项）。

![单头与多头在相同隐藏维度下的计算量与参数量等价性验证](/images/content-rebuild/1f6d96d3f719f1b4/fig-3.png)

### 1. 投影阶段（Q, K, V 映射）

- **单头**：使用参数量为 $3 \times (d_{\text{model}} \times d_{\text{model}})$ 的三个矩阵。
  - 参数量：$3 d_{\text{model}}^2$；
  - FLOPs：$3 \times 2 N d_{\text{model}}^2 = 6 N d_{\text{model}}^2$。
- **多头**：每个头拥有 3 个尺寸为 $d_{\text{model}} \times \frac{d_{\text{model}}}{h}$ 的矩阵，共 $h$ 个头。
  - 参数量：$h \times 3 \times \left(d_{\text{model}} \times \frac{d_{\text{model}}}{h}\right) = 3 d_{\text{model}}^2$；
  - FLOPs：$h \times 3 \times 2 N d_{\text{model}} \left(\frac{d_{\text{model}}}{h}\right) = 6 N d_{\text{model}}^2$。

两者的投影参数量与计算量严格相等。

### 2. 注意力计算阶段（打分与加权）

- **单头**：
  - 打分 $QK^T$：$(N \times d_{\text{model}}) \times (d_{\text{model}} \times N) \to 2 N^2 d_{\text{model}}$ FLOPs；
  - 加权 $A V$：$(N \times N) \times (N \times d_{\text{model}}) \to 2 N^2 d_{\text{model}}$ FLOPs；
  - 阶段总 FLOPs：$4 N^2 d_{\text{model}}$。
- **多头**：每个头的矩阵维度为 $N \times d_k$，共有 $h$ 个头。
  - 单个头打分 $Q_i K_i^T$：$2 N^2 d_k = \frac{2 N^2 d_{\text{model}}}{h}$ FLOPs；
  - 单个头加权 $A_i V_i$：$2 N^2 d_k = \frac{2 N^2 d_{\text{model}}}{h}$ FLOPs；
  - $h$ 个头累加：$h \times \left(\frac{2 N^2 d_{\text{model}}}{h} + \frac{2 N^2 d_{\text{model}}}{h}\right) = 4 N^2 d_{\text{model}}$ FLOPs。

注意力的核心计算量严格相等。

### 3. 输出线性映射阶段（$W^O$）

- **单头**：输入形状为 $N \times d_{\text{model}}$，权重为 $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$，FLOPs 为 $2 N d_{\text{model}}^2$。
- **多头**：拼接后的张量形状仍为 $N \times (h \cdot d_v) = N \times d_{\text{model}}$，权重同为 $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$，FLOPs 同样为 $2 N d_{\text{model}}^2$。

**结论**：多头注意力并不是通过“增加参数量和计算量”来换取更强的表征能力，而是通过**重新分配通道结构**，在理论矩阵乘法开销保持一致的前提下，将单一的加权平均拆解为多个并行子空间的独立加权。

---

## 经验观察：注意头的分工与微型集成

在代码实现中，所有注意头的结构完全一致，并没有人工规则要求“第 1 头负责句法，第 2 头负责指代”。多头的分工是在反向传播优化过程中**自发涌现**出来的。

![不同注意头在自然语言上下文中的实际注意力权重与分工模式](/images/content-rebuild/1f6d96d3f719f1b4/fig-4.png)

实证分析（例如 Voita 等人在 2019 年针对机器翻译模型注意头的探针实验）表明，训练成熟的模型中往往会涌现出几类典型模式：
- **位置邻近模式（Positional Heads）**：注意力主要聚焦在当前 Token 的紧邻前一个或后一个位置，类似于一维卷积提取局部 $n$-gram；
- **句法依赖模式（Syntactic Heads）**：稳定聚焦于具备特定依存关系的词对（如主谓对齐、动宾搭配）；
- **长程与指代模式（Coreference Heads）**：专门越过长距离去锁定对应的先行词；
- **标点与汇聚模式（Delimiter / Special Token Heads）**：将大部分注意力投向句子分隔符（如 `[SEP]`）或标点符号，起到类似“空操作”或全局信息暂存的作用。

在计算完成后，输出投影矩阵 $W^O$ 充当了跨子空间混合器的角色，将不同头提取到的特征进行线性重组，再输入后续的前馈网络（FFN，关于 FFN 的详细作用可参考 [Transformer 中 FFN 层有什么作用？](/interview/llm-interview/ffn-swiglu.md)）。

---

## 极端边界探讨：头数 $h$ 是越多越好吗？

既然多头能够提升表征丰富度，是否可以无限制增加头数？例如在 $d_{\text{model}} = 512$ 时，设 $h = 512$，使每个头的内部维度退化为 $d_k = 1$？

答案是否定的。增加头数存在明确的几何表征限制与硬件执行边界。

![头数极端增大导致标量退化与硬件算力不饱和的失效边界](/images/content-rebuild/1f6d96d3f719f1b4/fig-5.png)

### 1. 几何表征退化：$d_k = 1$ 时的方向性丧失

当 $d_k = 1$ 时，Query 和 Key 在每个头中均退化为单个实数标量 $q, k \in \mathbb{R}$。
标量点积即普通的实数相乘 $q \cdot k$：
- 多维向量点积（$d_k \ge 64$）能够通过方向夹角衡量高维空间中的相对几何关系（模式匹配能力）；
- 标量乘法只能反映数值大小和正负符号的缩放，失去了通过子空间方向夹角表达复杂模式匹配的能力。

### 2. 硬件计算与访存边界

在 GPU / TPU 架构上，矩阵乘法依靠 Tensor Core 等计算单元进行分块（Tiling）并行加速。主流硬件通常针对特定维度（如 16、32、64、128 的倍数）做了深度微架构优化：
- 若头数过多、单头维度 $d_k$ 极小，矩阵切片无法填满计算硬件的分块尺寸，导致计算单元利用率明显下降；
- 在自回归生成推理阶段，注意力计算需要维护 KV Cache。头数越多，内存中需要寻址和管理的张量切片就越细碎，增加了内存访问开销（Memory-bound）。

因此，主流大模型通常将单头维度 $d_k$ 保持在 64 或 128（例如 LLaMA 系列中 $d_{\text{model}}=4096, h=32 \implies d_k=128$），这是在表征几何容量与硬件计算吞吐之间权衡后的工程实践选择。

---

## 面试高频追问与易错点辨析

### 追问一：如果去掉最后的输出矩阵 $W^O$，多头注意力会怎样？

![多头注意力输出矩阵 WO 对各子空间特征进行线性解耦与交叉融合的作用](/images/content-rebuild/1f6d96d3f719f1b4/fig-6.png)

**标准回答**：
输出投影矩阵 $W^O$ 的核心职能是**跨子空间的特征线性混合（Inter-head Information Exchange）**。

各头在计算注意力时彼此相互独立、完全隔离。拼接操作 $\text{Concat}(\text{head}_1, \dots, \text{head}_h)$ 只是在通道维度上将它们拼合，并没有发生任何跨头的线性交叉。
如果去掉 $W^O$：
1. 输出向量的每个维度只能单独继承某一个头的注意力结果，头与头提取到的信息无法在当前层完成跨子空间交互；
2. Transformer 依赖残差连接（$x + \text{Sublayer}(x)$），残差相加要求子层输出必须与输入向量处于同一坐标基底与维度尺度下。缺乏 $W^O$ 的重组映射，子层输出无法有效与主干通道残差对齐（残差机制可参考 [Transformer 中的“残差连接”可以缓解梯度消失问题吗？](/interview/llm-interview/transformer-residual.md)）。

### 追问二：训练好的 Transformer 中，所有头的作用都同等重要吗？

**标准回答**：
并非所有头都同等重要。多项剪枝研究（例如 Michel 等人在论文 *Are Sixteen Heads Really Better than One?* 中的实验）表明：
- 在模型推理阶段，通过剪枝算法剪掉部分注意力层中相当比例的冗余头，模型的整体评估指标（如 BLEU 或准确率）并未出现明显劣化；
- 部分头甚至表现出接近单位矩阵的注意力模式（$A \approx I$），即只关注自身位置。

**为什么训练时仍然需要设置较多头？**
1. **优化阶段的探索能力**：训练初期，模型无法预知哪些子空间包含有效信号。多头提供了更丰富的优化路径与参数容量，有助于稳定优化过程；
2. **推理阶段的权衡**：由于推理自回归生成时 KV Cache 显存带宽压力巨大，为了减少冗余头的显存开销，业界演化出了 Multi-Query Attention (MQA) 与 Grouped-Query Attention (GQA) 等变体（详见 [Multi-Head Attention、MQA 和 GQA 有什么区别？](/interview/llm-interview/mha-mqa-gqa.md)）。

### 常见错误说法辨析

- **错误说法 1：“多头注意力的参数量和计算量是单头的 $h$ 倍。”**
  - **纠偏**：常规设计中每个头的通道维度被除以了 $h$（$d_k = d_{\text{model}} / h$）。推导表明，多头与同等输入输出维度的单头注意力在投影、打分、加权和输出映射上的参数量与理论 FLOPs 完全一致。
- **错误说法 2：“在训练多头注意力时，需要为每个头设计独立的损失函数以强制它们分工。”**
  - **纠偏**：各头共享全局的目标函数（如自回归预测的交叉熵损失），分工是参数随机初始化和反向传播梯度优化过程中自发出现的现象，无需人工设计分工损失。
- **错误说法 3：“为了让大模型在长文本生成时捕捉更多信息，头数 $h$ 设得越大越好。”**
  - **纠偏**：在自回归解码中，每个注意头都需要独立存储 KV Cache。过多头数会导致缓存寻址细碎、增加显存带宽负担，甚至引起硬件算力利用率下降，需要根据 $d_k$ 尺寸和显存吞吐综合权衡。

---

## 读者自测与知识承接

学完本讲后，请尝试回答以下两道问题以检验理解程度：

1. **计算量推导题**：保持隐藏层维度 $d_{\text{model}}$ 和序列长度 $N$ 不变，将头数从 1 增加到 8，理论上的矩阵乘法计算量（FLOPs）是否会增加？请写出推导依据。
2. **结构机制题**：在多头注意力输出拼接后，矩阵 $W^O$ 除了调整维度外，还承担了什么数学与工程职能？

理解了多头注意力在表征多样性与计算等价性上的设计后，建议进一步阅读 [Multi-Head Attention、MQA 和 GQA 有什么区别？](/interview/llm-interview/mha-mqa-gqa.md)，了解在大模型推理场景下，业界如何通过多查询与分组查询注意力在表征能力与显存吞吐之间寻找更优的工程折中。
