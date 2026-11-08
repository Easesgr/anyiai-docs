---
title: MHA、MQA 和 GQA 有什么区别？为什么现代大模型多采用 GQA？
icon: mdi:layers-outline
tag:
- 大模型面试
- MHA
- MQA
- GQA
description: 对比 MHA、MQA 与 GQA 的结构差异与推理开销，解释自回归解码中的访存受限瓶颈，以及 GQA 在显存节省与生成质量间的工程平衡。
---

在自回归大语言模型的面试中，注意力机制的三种变体——**MHA**（Multi-Head Attention）、**MQA**（Multi-Query Attention）和 **GQA**（Grouped-Query Attention）是考察工程认知和显存带宽瓶颈的高频题目。

很多开发者能记住头的数量比例，但被追问到“**为什么生成阶段主要是显存带宽受限？**”、“**算术强度怎么算？**”或者“**为什么训练阶段没法像推理那样大幅省显存？**”时，往往卡在细节上。

本文从硬件读取瓶颈出发，解释 MHA、MQA 与 GQA 的结构差异、显存与带宽计算，以及当前主流大模型的选型原因。

![MHA、MQA 与 GQA 机制对比封面图](/images/content-rebuild/0156638235a128ad/cover.png)

---

## 1. 面试 30 秒口述答案

在面试中被问到“请对比 MHA、MQA 与 GQA 的区别”时，可以按照以下三步回答：

1. **结构定义**：
   - **MHA（标准多头注意力）**：每个 Query 头都有一个对应的 Key 头和 Value 头，头数完全相等（$H_q = H_{kv}$）。
   - **MQA（多查询注意力）**：所有 Query 头共享同一套 Key 和 Value 头（$H_{kv} = 1$）。
   - **GQA（分组查询注意力）**：将 Query 头分成 $G$ 个组，每组共享一套 Key 和 Value 头（$1 < H_{kv} < H_q$）。MHA 和 MQA 是 GQA 的两个极端情况。
2. **设计原因（推理瓶颈）**：大模型在生成（Decode）阶段是典型的**访存受限（Memory-bound）**任务。每生成一个 Token，GPU 都需要把之前所有上下文对应的 KV Cache 从显存（HBM）完整读取一遍。MHA 的 KV Cache 占用显存太大、读取带宽消耗高，严重限制了并发数（Batch Size）和上下文长度。
3. **权衡取舍**：MQA 把 KV Cache 压缩到了 $1/H_q$，但在大模型或长文本复杂任务中容易损失表达能力；GQA（如 8:1 分组）在吞吐和显存占用上逼近 MQA，同时几乎完整保留了 MHA 的表达能力，因此成为 LLaMA-2 (70B)、LLaMA-3、Mistral 等现代大模型的主选方案。

---

## 2. 结构对比：投影权重与头数分配

要理解三者的区别，首先看投影层的张量形状变化。多头注意力的基本定义可先参考 [为什么 Transformer 采用多头注意力机制？](/interview/llm-interview/why-multi-head-attention.md)。

设模型的隐藏层维度为 $d_{model}$，单个注意力头的维度为 $d_k$（通常满足 $H_q 	imes d_k = d_{model}$），Query 的头数为 $H_q$。

![MHA、GQA 与 MQA 架构拓扑与投影映射对比](/images/content-rebuild/0156638235a128ad/fig-1.png)

### 2.1 Multi-Head Attention (MHA)
在标准 MHA 中，Key 和 Value 的头数与 Query 严格相等：
$$H_{kv} = H_q$$
- 投影矩阵维度：$W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times (H_q \cdot d_k)}$
- 运行状态：每个 Query 头只与其专属的 Key/Value 头做点积与加权求和。各个头可以在不同子空间中独立提取特征，自由度最高。

### 2.2 Multi-Query Attention (MQA)
为了压减 Key 和 Value 的大小，Noam Shazeer 于 2019 年提出了 MQA，将 Key 和 Value 的头数压缩为 1：
$$H_{kv} = 1$$
- 投影矩阵维度：
  - $W_Q \in \mathbb{R}^{d_{model} \times (H_q \cdot d_k)}$
  - $W_K, W_V \in \mathbb{R}^{d_{model} \times (1 \cdot d_k)}$
- 运行状态：所有 $H_q$ 个 Query 头共用这一份 Key 和 Value 向量。Key 和 Value 的投影参数量以及运行时缓存大小均降至 MHA 的 $1/H_q$。

### 2.3 Grouped-Query Attention (GQA)
GQA 是 MHA 与 MQA 之间的折中方案。它将 $H_q$ 个 Query 头划分为 $G$ 个组（即 $H_{kv} = G$），每组包含 $H_q / H_{kv}$ 个 Query 头：
$$1 < H_{kv} < H_q \quad \text{且通常} \quad H_q \pmod{H_{kv}} = 0$$
- 投影矩阵维度：
  - $W_Q \in \mathbb{R}^{d_{model} \times (H_q \cdot d_k)}$
  - $W_K, W_V \in \mathbb{R}^{d_{model} \times (H_{kv} \cdot d_k)}$
- 运行状态：同一个组内的 Query 头共享对应的 Key/Value 头。当 $H_{kv} = H_q$ 时，GQA 退化为 MHA；当 $H_{kv} = 1$ 时，GQA 退化为 MQA。

<ReadMoreLock />

---

## 3. 为什么优化 KV Cache 能加速？访存瓶颈推导

为什么这几种结构会完全围绕 Key 和 Value 的头数展开？答案在大模型生成阶段的硬件特性中。

关于 KV Cache 的由来与存储机制，可参考 [KV Cache 是什么？](/interview/llm-interview/kv-cache.md)。自回归解码时，模型每一步只输入 1 个新 Token，但需要计算该 Token 与历史 $S$ 个 Token 的注意力关联。

![自回归解码阶段的访存受限瓶颈剖析](/images/content-rebuild/0156638235a128ad/fig-2.png)

### 3.1 单步生成下的算术强度
**算术强度（Arithmetic Intensity）** 定义为：计算量（FLOPs）除以显存数据搬运量（Bytes）：
$$I = \frac{\text{FLOPs}}{\text{Memory Access (Bytes)}}$$
以常见 GPU（如 NVIDIA A100 SXM 80GB）为例：
- FP16 峰值算力约为 312 TFLOPS；
- 显存带宽约为 2 TB/s；
- 达到硬件满载的算术强度平衡点为：$312 \times 10^{12} / (2 \times 10^{12}) = 156 \text{ FLOPs/Byte}$。

再看自回归 Decode 生成 1 个 Token 时，单个注意力头内部的计算量与访存量：
1. **计算量**：
   - 当前 Query 向量与历史 $S$ 个 Key 向量做点积：$2 \times S \times d_k$ FLOPs（每次乘加计 2 次浮点运算）；
   - 注意力权重与历史 $S$ 个 Value 向量加权求和：$2 \times S \times d_k$ FLOPs；
   - 核心计算量合计约 $4 \times S \times d_k$ FLOPs。
2. **显存读取量**：
   - 必须完整从 HBM 显存中读取历史 $S$ 个 Token 对应的 Key 和 Value 向量；
   - 采用 FP16/BF16 存储时（每个数值元素占 2 字节），读取量为 $(S \times d_k \times 2) + (S \times d_k \times 2) = 4 \times S \times d_k$ Bytes。
3. **局部的算术强度**：
$$I_{attn} \approx \frac{4 \times S \times d_k}{4 \times S \times d_k} = 1 \text{ FLOP/Byte}$$

算术强度仅为 1，远低于芯片平衡点 156。这意味着：**GPU 计算核心在绝大多数时钟周期内都在等待显存搬运数据，芯片算力利用率很低**。此时系统的执行瓶颈是显存带宽（Memory-bound）。要让 Decode 变快，唯一有效的办法就是减少每一步必须从显存搬运的数据量。

### 3.2 KV Cache 显存占用公式

![KV Cache 显存容量与搬运开销数学公式拆解](/images/content-rebuild/0156638235a128ad/fig-3.png)

对于层数为 $L$、当前序列长度为 $S$、批大小为 $B$、单头维度为 $d_k$、KV 头数为 $H_{kv}$ 的模型，采用 16-bit 浮点数（FP16/BF16，每个数值元素占 2 字节）存储时，整机 KV Cache 显存占用为：
$$\text{Memory}_{KV} = 2 \times B \times S \times L \times H_{kv} \times d_k \times 2 \text{ (Bytes)}$$
- 最前面的系数 $2$ 表示同时存储 Key 和 Value 两个张量；
- 最后的系数 $2$ 表示 FP16 数据类型占 2 字节。

#### 典型算例：LLaMA-2 70B
- 结构参数：$L=80, d_{model}=8192, H_q=64, d_k=128$
- 推理配置：$B=16, S=4096$

1. **如果使用 MHA ($H_{kv} = H_q = 64$)**：
   $$\text{Memory}_{KV} = 4 \times 16 \times 4096 \times 80 \times 64 \times 128 = 171,798,691,840 \text{ Bytes} \approx 160 \text{ GB}$$
   仅 16 个并发，历史 KV Cache 就要占 160 GB。在单张 80GB 卡上甚至放不下该缓存，每生成一步都需要在显存中搬运整整 160 GB 的数据。
2. **使用 GQA ($H_{kv} = 8$, 即 8:1 分组，LLaMA-2 70B 真实配置)**：
   $$\text{Memory}_{KV} = \frac{160 \text{ GB}}{64 / 8} = 20 \text{ GB}$$
   KV Cache 显存占用直接缩减为原先的 $1/8$，单步显存搬运量减少了 $87.5\%$。
3. **如果使用 MQA ($H_{kv} = 1$)**：
   $$\text{Memory}_{KV} = \frac{160 \text{ GB}}{64} = 2.5 \text{ GB}$$

显存占用从 160 GB 降至 20 GB，意味着原本只能跑 16 并发的显存空间，现在可以支撑上百个并发，或者容纳更长的上下文。

---

## 4. 表达能力与推理吞吐的平衡

压缩 KV 头数大幅削减了带宽开销，但会带来什么代价？

![注意力机制的质量与吞吐帕累托前沿权衡曲线](/images/content-rebuild/0156638235a128ad/fig-4.png)

### 4.1 MQA 的容量损失
多头注意力的作用在于允许模型同时在不同表征子空间中关注上下文（如分别关注语法依存、实体匹配、代词指代等，详见 [Self-Attention 中 Q、K、V 分别是什么意思？](/interview/llm-interview/attention-qkv.md)）。

在 MQA 中，由于所有 Query 头强制共享同一套 Key 向量，模型在寻址时只能基于同一套语义上下文进行投影。实验表明：
- 在小参数模型（如 1B~3B）或单轮简单问答中，网络可以通过加深层数来弥补单层的表征损失，质量下降不明显；
- 但在长文本检索、代码生成、复杂多跳推理以及大参数模型预训练中，MQA 容易出现困惑度（Perplexity）偏高、检索精度受损的现象。

### 4.2 GQA 的工程平衡
Ainslie 等人在 2023 年论文 *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* 中通过对比发现：
1. **精度接近无损**：采用 8 组的 GQA，在多种下游基准测试上的效果与 MHA 非常接近；
2. **速度逼近 MQA**：在生成阶段的吞吐和延迟上，GQA 达到了接近 MQA 的效率；
3. **已有模型平滑升级（Uptraining）**：原本用 MHA 预训练好的检查点，可以通过对同组内 Key/Value 头的投影矩阵做均值池化（Mean Pooling），仅用少量预训练数据（约原训练步数的 5%）微调，就能转为 GQA 模型。

| 比较维度 | MHA (Multi-Head) | GQA (Grouped-Query) | MQA (Multi-Query) |
| :--- | :--- | :--- | :--- |
| **KV 头数比例** | $H_{kv} = H_q$ | $1 < H_{kv} < H_q$（常用 8 或 4） | $H_{kv} = 1$ |
| **Decode KV 显存与搬运量** | 基准（$100\%$） | 大幅降低（常见为基准的 $12.5\% \sim 25\%$） | 极低（约为基准的 $1.5\% \sim 3\%$） |
| **表征能力** | 独立子空间多，上限最高 | 接近 MHA | 复杂长文任务中容易受限 |
| **硬件瓶颈属性** | 严重受制于显存带宽 | 显存带宽压力显著缓解 | 显存带宽压力降至最低 |
| **代表模型** | GPT-3, LLaMA-1 | LLaMA-2 (70B), LLaMA-3, Mistral | Falcon-40B, PaLM, ChatGLM2/3 |

---

## 5. 面试高频追问

### 追问 1：为什么 GQA 对 Prefill 阶段的加速不如 Decode 阶段明显？

关于大模型推理阶段的性能瓶颈分析，可参考 [Transformer 模型的性能瓶颈在哪？](/interview/llm-interview/transformer-bottleneck.md)。

![Prefill 阶段与 Decode 阶段的硬件计算特征对比](/images/content-rebuild/0156638235a128ad/fig-5.png)

**原理解释**：
- **Prefill 阶段（在中长序列或适度并发下通常偏向计算受限 Compute-bound）**：
  系统一次性处理用户输入的提示词序列（长度为 $S$）。在 GQA 中，虽然 KV 头数减少为 $H_{kv}$，但同一组内的各个 Query 头依然持有独立的查询特征，必须分别与共享的 Key 向量计算注意力权重，因此核心注意力点积与加权求和的矩阵乘浮点运算量（FLOPs，与 $S^2$ 成正比）并没有减少；GQA 仅减少了 $W_K, W_V$ 的投影计算量（与 $S$ 成正比）以及写入显存的缓存量。在输入序列较长时，计算密集型矩阵乘（GEMM）耗时占据主导，硬件更易接近算力峰值而非带宽瓶颈，因此对 Prefill 整体耗时的改善相对有限。
- **Decode 阶段（通常偏向访存受限 Memory-bound）**：
  自回归生成时，每步仅输入 1 个新 Token。计算形式退化为向量与矩阵相乘（GEMV），算术强度极低，GPU 计算核心常处于等待数据搬运的状态。每生成一个 Token 都必须从显存完整读取全部历史 KV Cache，此时瓶颈主要在于显存读取带宽。GQA 直接将 KV 搬运体积降低到原先的几分之一，因此加速效果极为显著。

### 追问 2：在训练阶段，MQA/GQA 能减少显存占用吗？

**答案是：能减少一部分，但优化重心不是 KV Cache，而是激活值（Activation Memory）与参数量。**

1. **训练时没有自回归 KV Cache**：
   常规的因果语言模型训练采用 Teacher Forcing 机制，输入是完整序列，并行计算所有位置的注意力与损失，不需要像生成时那样维护动态追加的 KV Cache 结构。
2. **反向传播保留的激活值减少**：
   为了在反向传播中计算梯度，前向传播中计算出的 Key 和 Value 投影张量必须保存在显存中。张量形状从 MHA 的 $[B, S, H_q, d_k]$ 缩减为 $[B, S, H_{kv}, d_k]$，这部分激活值显存随之降低。
3. **投影权重与优化器状态微幅减少**：
   $W_K$ 和 $W_V$ 的参数矩阵大小缩减为原本的 $H_{kv} / H_q$。参数量变少后，对应的梯度以及 Adam 优化器状态（一阶动量与二阶动量）占用的显存也相应减少。
4. **为什么训练总显存收益有限**：
   在大模型全量训练中，显存占用的大头是优化器状态（详见 [Transformer 中，哪个部分最占用显存？](/interview/llm-interview/transformer-memory.md)）以及前馈网络（FFN/MLP）的庞大激活值。注意力投影参数只占全模型权重的一小部分，且训练全程以 GEMM 矩阵乘计算为主。因此，MQA/GQA 的主要收益场景是**推理生成**，而非训练。

### 追问 3：在推理引擎底层实现中，GQA 会在显存中复制多份 KV 吗？

很多开发者误以为 GQA 会在显存里把 1 个 KV 头物理复制多份来和对应的 Query 头做运算，这是错误的。

![GQA 推理引擎的虚拟广播与常见误区排查](/images/content-rebuild/0156638235a128ad/fig-6.png)

**实际工程实现**：
- **显存只存一份**：在 vLLM、PagedAttention 或 TensorRT-LLM 中，显存分配器只为实际存在的 $H_{kv}$ 个头分配物理空间，不会在显存中复制数据，否则显存优化就会失效。
- **片上共享与广播读取（Virtual Broadcast）**：
  在 GPU Kernel（如 FlashDecoding 或自定义 Triton 算子）执行时，同组内的多个 Query 头会被调度到同一个线程块（Thread Block）或相邻线程中。Key 和 Value 仅从显存（HBM）读取一次到芯片片上的共享内存（SRAM）中，然后由这几个 Query 头共同读取并完成计算，从而保证显存搬运量不放大。

---

## 6. 面试常见错误表述排查

在面试中，注意避免以下不严谨或错误的说法：

- ❌ **错误说法 1**：“MQA 和 GQA 的主要目的是降低注意力的 FLOPs 计算量。”
  - **分析**：在整个 Transformer 中，注意力模块的 FLOPs 通常少于 FFN 模块。MQA/GQA 带来的绝对浮点计算削减很少，其核心优化目标是**显存容量（Capacity）与显存搬运带宽（Bandwidth）**，针对的是自回归生成的访存受限问题。
- ❌ **错误说法 2**：“GQA 是把每个注意力头的维度 $d_k$ 变小了。”
  - **分析**：每个注意力头的内部维度 $d_k$（如 128 维）保持不变，改变的是**头的物理数量（Head Count）**，不是单头的向量长度。
- ❌ **错误说法 3**：“大模型训练开启 GQA 后，训练显存可以直接减半。”
  - **分析**：训练显存主要由权重参数、优化器状态和 FFN 层的激活值主导。GQA 缩减的仅是注意力模块中 K、V 的投影张量，对训练总显存的缩减比例通常较为有限。

理解了这三者的权衡机制后，可以继续深入了解注意力计算中的位置表征方案：[Transformer 的位置编码是怎样的？](/interview/llm-interview/transformer-positional-encoding.md) 与 [位置编码是什么？RoPE 为什么很常用？](/interview/llm-interview/positional-encoding.md)。
