---
title: KV Cache 是什么？自回归生成如何用显存换取速度
icon: mdi:memory
tag:
- 推理与生成
- KV Cache
- 推理加速
description: 详解自回归大模型生成中 KV Cache 的原理、单步张量流转与显存公式，分析生成阶段显存带宽瓶颈与优化方向。
---

大语言模型生成文本时，是一字一句往外蹦的。每生成一个新词，模型都需要综合前面的所有历史输入。如果不做任何优化，每一步都把全部历史文本重新算一遍，会产生大量重复的矩阵乘法。

**KV Cache（键值缓存）** 就是为了解决这个问题而存在的：它把前面已经计算好的 Key（键）和 Value（值）张量保存在显存里，每一步解码只对最新输入的一个词做投影计算，然后把新结果追加到缓存中。这种策略用显存容量换取了推理速度，是现代大模型推理引擎的标准配置。

![KV Cache原理封面：讲解员展示自回归解码环与显存中逐渐累积的KV张量块](/images/content-rebuild/b459d0a0fe6b1859/cover.png)

---

## 1. 朴素逐词解码的重复计算

大模型处理文本通常分为两个阶段（在 [/fundamentals/inference-and-generation/inference-pipeline.md](/fundamentals/inference-and-generation/inference-pipeline.md) 中已有介绍）：
1. **预填充阶段（Prefill）**：模型一次性读入用户输入的提示词（Prompt），并行计算所有词的信息，并输出第一个生成词。
2. **解码阶段（Decode）**：模型进入自回归循环，每次将上一轮生成的词拼接到输入末尾，预测下一个词，直到输出结束符号（EOS）或达到最大长度。

我们来看一个具体的文本生成过程。假设用户输入的提示词是「人工智能」，模型接下来要生成「正在改变世界」：

- **第 1 步**：输入「人工智能」，模型计算后输出「正」；
- **第 2 步**：在朴素解码中，将「人工智能正」共 5 个词全部重新输入模型，重新计算它们的隐藏状态，输出「在」；
- **第 3 步**：将「人工智能正在」共 6 个词再次全部输入模型，输出「改」。

在 Transformer 的注意力层中，每个输入词都需要乘以权重矩阵 $W_Q, W_K, W_V$ 来生成对应的 Query、Key 和 Value 向量：
$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

![朴素解码与KV Cache计算流程对比：左侧重复投影历史，右侧仅增量投影新Token](/images/content-rebuild/b459d0a0fe6b1859/fig-1.png)

在因果注意力机制（Causal Attention）与常见的位置编码（如 RoPE）下，前一个词在第 2 步和第 3 步算出来的 Key 和 Value 向量，和第 1 步算出来的数值完全一致。因为历史词的文本内容没有变，模型的投影参数也没有变。

如果在生成长度为 $N$ 的文本时，每一步都把前 $k$ 个词重新做一次投影，前向投影的总计算量就会随长度呈二次方增长：
$$
\sum_{k=1}^N \mathcal{O}(k) = \mathcal{O}(N^2)
$$
大量算力被消耗在重复计算早已知道的结果上。

---

## 2. 为什么只缓存 Key 与 Value，不缓存 Query？

要消除这种重复，需要看单步自注意力运算到底需要什么数据。

当生成第 $t$ 个词时，因果注意力要求当前位置只能看到它自己和前面的所有词。我们把这一步的注意力计算展开：
$$
\text{Attention}(Q_t, K_{\le t}, V_{\le t}) = \text{softmax}\left( \frac{Q_t K_{\le t}^T}{\sqrt{d_k}} \right) V_{\le t}
$$

我们分别看这三个角色的输入：
- **$Q_t$（当前步 Query）**：只包含当前最新词在位置 $t$ 的查询向量。它的作用是问前面的词：「有哪些信息与我当前预测下一个词相关？」因此只需要当前步新计算出的 1 个向量。
- **$K_{\le t}$（历史到当前的 Key 矩阵）**：包含从第 1 个词到第 $t$ 个词的全部 Key 向量。$Q_t$ 必须分别与每一个历史 Key 做点积，才能算出当前词对每个历史词的注意力权重。
- **$V_{\le t}$（历史到当前的 Value 矩阵）**：同样包含第 1 到第 $t$ 个词的 Value 向量。算出来的注意力权重需要对这些 Value 向量做加权求和，汇总出当前步的上下文表示。

![因果注意力单步计算分解：展示当前步Query与历史拼接的Key和Value点积计算](/images/content-rebuild/b459d0a0fe6b1859/fig-2.png)

**为什么历史的 Query（即 $Q_{<t}$）不需要保留？**

因为自回归注意力的依赖是单向的。第 1 步的 Query 只在预测第 2 个词时参与计算；由于后续词不能倒流影响历史，第 1 步的 Query 永远不需要和第 5 步、第 10 步的 Key 做运算。一旦某个时间步预测完成，该步的 Query 就结束了它的任务，可以直接丢弃。

相比之下，每一个新词的生成，都需要用到历史所有词的 Key 和 Value。因此，只要在显存里把历史各步算好的 Key 和 Value 存下来，后续每一步就只需要对最新那 1 个词执行 $W_Q, W_K, W_V$ 投影运算，权重矩阵投影的计算量直接从每步 $\mathcal{O}(t)$ 降到了 $\mathcal{O}(1)$。

注意：即使缓存了 Key，当前步计算注意力分数时，$Q_t$ 与历史所有 Key 的点积运算依然包含 $t$ 次乘加操作。KV Cache 消除的是全序列历史 Token 的重复投影计算与历史各词之间的冗余点积，但当前最新词与历史所有 Key 的点积依然必不可少，计算量随历史长度线性增长。

<ReadMoreLock />

---

## 3. KV Cache 的张量形状与增量更新流转

在底层显存中，KV Cache 是为 Transformer 的每一个自注意力层独立分配的张量。以最标准的多头自注意力（MHA）为例，单层中 Key Cache 和 Value Cache 的形状通常排布为：
$$
[B, n_{heads}, L_{seq}, d_{head}]
$$
其中：
- $B$：批大小（Batch Size），即同时并发处理的请求数；
- $n_{heads}$：该层的注意力头数量；
- $L_{seq}$：当前已经处理的总词数（Prompt 长度 + 已生成长度）；
- $d_{head}$：每个注意力头的向量维度（满足 $n_{heads} \times d_{head} = d_{model}$）。

![KV Cache多维张量存储结构：展示各层张量形状与新Token沿seq_len维度的拼接](/images/content-rebuild/b459d0a0fe6b1859/fig-3.png)

### 单步数值流转走查

我们用一个微型算例把数据流转走通一遍：
- 设批大小 $B=1$，头数 $n_{heads}=1$，头维度 $d_{head}=4$；
- 输入提示词包含 3 个词：$[x_1, x_2, x_3]$。

**第一步：预填充（Prefill）**
1. 输入张量形状为 $(1, 3, 4)$；
2. 投影计算出 3 个词的 Key 和 Value，张量形状均为 $(1, 1, 3, 4)$；
3. 将其写入显存，作为初始缓存：
   $$
   \text{Cache}_K = K_{1:3}, \quad \text{Cache}_V = V_{1:3}
   $$
4. 依据最后一个位置的输出隐向量，经过采样（如 [/fundamentals/inference-and-generation/temperature-topk-topp.md](/fundamentals/inference-and-generation/temperature-topk-topp.md)）预测出第 4 个词 $x_4$。

**第二步：第 1 步解码（生成 $x_5$）**
1. 输入只传 1 个新词 $x_4$，张量形状为 $(1, 1, 4)$；
2. 仅对 $x_4$ 做投影，得到单向量 $Q_4, K_4, V_4$，形状均为 $(1, 1, 1, 4)$；
3. 将 $K_4, V_4$ 沿序列长度维度追加（Append）到原有缓存中：
   $$
   \text{Cache}_K \text{ 形状变为 } (1, 1, 4, 4)
   $$
   $$
   \text{Cache}_V \text{ 形状变为 } (1, 1, 4, 4)
   $$
4. 注意力计算：取单向量 $Q_4$（形状 $1 \times 4$）与完整的 $\text{Cache}_K$ 转置（形状 $4 \times 4$）做点积，得到长度为 4 的注意力分数分布；
5. 用分数对完整的 $\text{Cache}_V$（形状 $4 \times 4$）加权求和，得到输出向量（形状 $1 \times 4$），送入后续网络预测出 $x_5$。

![增量解码单步数值与张量流转算例走查：展示Prefill与Decode第1步的具体拼接与乘法](/images/content-rebuild/b459d0a0fe6b1859/fig-4.png)

后续每一步都重复这个流程：只投影 1 个新词，将新 Key/Value 拼进缓存，带着全量缓存算一次注意力。

---

## 4. KV Cache 显存占用计算公式

KV Cache 节省了计算量，但代价是持续占用显存。在评估 GPU 选型和并发容量时，需要掌握准确的容量预估公式：

$$
\text{显存大小 (Bytes)} = 2 \times B \times L \times n_{layers} \times n_{kv\_heads} \times d_{head} \times P_{bytes}
$$

各参数含义如下：
- **因子 2**：Key 和 Value 分别占用一份显存；
- **$B$**：并发批大小；
- **$L$**：当前上下文总长度（Prompt 词数 + 已生成词数）；
- **$n_{layers}$**：模型的隐藏层总数（Transformer 层数）；
- **$n_{kv\_heads}$**：每一层中 Key/Value 注意力头的数量。在多头注意力（MHA）中等于查询头数；在分组查询注意力（GQA）中小于查询头数；
- **$d_{head}$**：单个注意力头的向量维度；
- **$P_{bytes}$**：每个参数占用的字节数（FP16/BF16 为 2 字节，FP8 为 1 字节）。

![KV Cache显存容量计算公式拆解：展示各变量因果关系及序列变长时的阶梯显存增长](/images/content-rebuild/b459d0a0fe6b1859/fig-5.png)

### 实例测算：以 7B 模型（MHA 配置）为例

我们以一个常见的 7B 模型参数进行推演：
- 层数 $n_{layers} = 32$
- 采用多头注意力（MHA），$n_{kv\_heads} \times d_{head} = d_{model} = 4096$
- 采用 16 位浮点数（BF16/FP16），即 $P_{bytes} = 2$

**单个请求中，每增加 1 个词所需的 KV Cache 显存**为：
$$
2 \times 1 \times 1 \times 32 \times 4096 \times 2 \text{ 字节} = 524,288 \text{ 字节} = 0.5 \text{ MB}
$$

这意味着每个词在单样本上占用 0.5 MB 显存。我们以此推导不同并发与长度下的总显存：

| 并发批大小 ($B$) | 序列长度 ($L$) | 模型自身权重显存 (BF16) | KV Cache 显存占用 | 显存变化影响 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 1,000 | 约 14 GB (静态) | $0.5 \text{ MB} \times 1,000 \approx 0.5 \text{ GB}$ | 缓存占比极小 |
| 1 | 4,000 | 约 14 GB (静态) | $0.5 \text{ MB} \times 4,000 \approx 2.0 \text{ GB}$ | 缓存逐渐增加 |
| 16 | 4,000 | 约 14 GB (静态) | $2.0 \text{ GB} \times 16 = 32.0 \text{ GB}$ | **缓存显存达到模型权重的 2 倍以上** |
| 32 | 8,000 | 约 14 GB (静态) | $0.5 \text{ MB} \times 32 \times 8,000 = 128.0 \text{ GB}$ | 单张 80GB 显卡彻底超显存（OOM） |

由此可见：**模型参数权重在推理过程中是静态不变的，而 KV Cache 随着并发数和文本长度线性膨胀**。在长文本推理或高并发服务中，限制系统承载能力的主要瓶颈往往是 KV Cache。

---

## 5. 算力换显存带来的硬件瓶颈转变

KV Cache 显著降低了浮点运算量，但它在硬件执行层面引发了一个新的瓶颈：**从计算密集型（Compute-bound）转向了访存密集型（Memory-bound）**。

硬件运行速度的瓶颈由**算术强度（Arithmetic Intensity）**决定，即每从显存搬运 1 字节数据，能执行多少次浮点运算（FLOPs/Byte）：

1. **预填充阶段（计算密集）**：
   模型一次性处理整段 Prompt，执行的是矩阵乘矩阵（GEMM）运算。输入数据被加载到芯片缓存后，会与多组参数反复进行点积运算，算术强度高，GPU 的 Tensor Core 可以充分发挥计算算力。
2. **解码阶段（常受显存带宽限制）**：
   在单请求或小批处理场景下，每个时间步输入仅有最新生成的 1 个词，模型投影运算退化为矩阵乘向量（GEMV）类操作；而在注意力计算时，GPU 必须把显存（HBM）中此前缓存的全量 Key 和 Value 数据完整读取到片上缓存。此时算术强度较低，计算核心往往需要等待数据搬运，极易受制于显存带宽墙（Memory-bound）。当并发批大小（Batch Size）显著增大时，解码阶段的算术强度会有所回升，瓶颈特性也会相应改变。

![Prefill与Decode计算瓶颈转变：展示高算力利用率向低算力访存饥饿的转变过程](/images/content-rebuild/b459d0a0fe6b1859/fig-6.png)

这就是为什么在大模型逐词生成时，监控面板上的 GPU 占用率可能是 100%，但计算单元的实际算力利用率（MFU）并不高，大部分时间受制于显存带宽墙。

为了缓解显存占用与显存带宽瓶颈，业界演化出了三类针对性的技术：
- **减少缓存头数（架构优化）**：采用 MQA（多查询注意力）或 GQA（分组查询注意力），让多个 Query 头共享同一份 Key/Value 头，直接将 KV Cache 压缩为原来的 $1/4$ 或 $1/8$；
- **减少显存碎片（内存管理）**：采用 PagedAttention 技术（如 vLLM），像操作系统虚拟内存分页一样管理 KV Cache，消除显存内部碎片；
- **降低数值精度（量化压缩）**：采用 FP8、INT8 或 INT4 对 KV Cache 进行量化存储，直接将搬运量和存储开销减半。
相关机制将在后续的 [/fundamentals/inference-and-generation/context-length.md](/fundamentals/inference-and-generation/context-length.md) 与 [/fundamentals/inference-and-generation/inference-acceleration.md](/fundamentals/inference-and-generation/inference-acceleration.md) 中详细解析。

---

## 6. 小结与延伸阅读

- **核心作用**：利用因果注意力的不变性，缓存历史 Key 和 Value 向量，使生成阶段的权重矩阵投影计算量从 $\mathcal{O}(N^2)$ 降为 $\mathcal{O}(N)$。
- **为什么不存 Query**：历史 Query 在该时间步结束后不再被任何未来计算需要，只有历史 Key 和 Value 需要参与后续所有新词的点积与加权求和。
- **显存与硬件代价**：KV Cache 随序列长度和并发批大小线性增长；在典型中小批量下，解码瓶颈常受制于显存带宽墙（Memory-bound）。

单步 Token 计算完成后，推理引擎如何将这些字句实时传回给客户端应用？下一篇我们将探讨网络传输层面的协议机制：[/fundamentals/inference-and-generation/streaming.md](/fundamentals/inference-and-generation/streaming.md)。
