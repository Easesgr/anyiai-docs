---
title: Transformer 模型的性能瓶颈在哪？Prefill 与 Decode 有何不同？
icon: mdi:speedometer
tag:
- Transformer
- 性能瓶颈
- 推理优化
description: 解析 Transformer 推理在 Prefill 算力受限与 Decode 访存受限的分流机制，结合 Roofline 模型说明加速方案与批处理收益。
---

在大模型推理与系统工程面试中，“Transformer 的性能瓶颈在哪”是一个高频出现的硬核考题。如果仅仅回答“瓶颈在显存带宽”或“注意力机制计算量太大”，都只看到了系统的一侧。

在现代 GPU 硬件上运行自回归 Transformer 时，性能瓶颈在不同推理阶段存在明显的物理分流：**Prefill（预填充/输入编码）阶段通常受限于计算核心算力**，而**Decode（自回归逐字生成）阶段在中小并发下主要受限于显存带宽**。

![Transformer 性能瓶颈分析封面：计算密集型 Prefill 与访存密集型 Decode 的分流机制](/images/content-rebuild/1364b015b3d0a4db/cover.png)

---

## 30 秒面试口述答案

面试时遇到这个问题，可以用以下三层逻辑清晰作答：

1. **划分推理阶段**：自回归 Transformer 推理分为 Prefill 和 Decode 两个阶段。Prefill 阶段一次性处理用户的全部输入提示词，执行的是大矩阵乘法（GEMM），数据复用率高，计算强度高于 GPU 的硬件拐点，属于**计算密集型（Compute-bound）**，瓶颈在于 Tensor Core 浮点算力；
2. **单步解码瓶颈**：Decode 阶段每一步只处理上一步新生成的 1 个 Token，矩阵乘法退化为矩阵-向量乘法（GEMV）。在常见的小批次场景下，每生成一个 Token 都必须把全量模型权重以及历史的 [KV Cache](/interview/llm-interview/kv-cache.md) 从高带宽显存（HBM）完整读取到片上缓存（SRAM），计算强度远低于硬件拐点，属于**访存密集型（Memory-bound）**，计算核心绝大部分时间在等待显存搬运数据；
3. **常见优化对应**：针对 Prefill 阶段长文本注意力在显存中的频繁读写，采用 **FlashAttention**（分块计算与在线 Softmax）减少 HBM 搬运；针对 Decode 阶段显存碎片的利用率损耗，采用 **PagedAttention** 分页管理显存；针对 Decode 单步访存受限，采用**增大 Batch Size** 或**投机采样（Speculative Decoding）**将访存平摊，提高单次加载权重的计算利用率。

---

## 硬件标尺：用 Roofline 模型判断瓶颈类型

要准确判断瓶颈落在算力还是显存带宽，需要使用计算机系统结构中的 **Roofline 模型**。它描述了算法的计算特征与硬件能力上限之间的关系。

![Roofline 模型与硬件饱和拐点：区分访存瓶颈区与计算瓶颈区](/images/content-rebuild/1364b015b3d0a4db/fig-1.png)

### 1. 两个核心指标

- **计算强度（Operational Intensity, $I$）**：算法每搬运 1 字节显存所执行的浮点运算次数，单位为 $\text{FLOPs/Byte}$。
  $$I = \frac{\text{浮点运算量 (FLOPs)}}{\text{显存访存量 (Bytes)}}$$
- **硬件饱和拐点（Machine Balance, $I_{\text{turn}}$）**：硬件理论峰值算力与显存带宽的比值。
  $$I_{\text{turn}} = \frac{P_{\text{peak}} (\text{FLOPs/s})}{B_{\text{peak}} (\text{Bytes/s})}$$

### 2. 硬件拐点实例

以 NVIDIA A100 SXM 80GB GPU 为例，其半精度 FP16 理论张量峰值约为 $312\text{ TFLOPs/s}$，HBM2e 显存带宽约为 $2039\text{ GB/s}$。它的硬件拐点计算如下：
$$I_{\text{turn}} = \frac{312 \times 10^{12}\text{ FLOPs/s}}{2039 \times 10^9\text{ Bytes/s}} \approx 153\text{ FLOPs/Byte}$$

- **访存瓶颈区（$I < I_{\text{turn}}$）**：计算强度低于 153 时，显存带宽跟不上计算单元的需求。即使 Tensor Core 算力再高，也必须停下来等待数据从 HBM 运送到片上 SRAM，实际吞吐由显存带宽封顶：$P_{\text{actual}} = I \times B_{\text{peak}}$。
- **计算瓶颈区（$I \ge I_{\text{turn}}$）**：计算强度达到或超过 153 时，显存搬运的数据能充分喂饱计算单元，系统瓶颈落在 Tensor Core 本身的计算能力上，实际性能接近硬件峰值 $P_{\text{peak}}$。

<ReadMoreLock />

---

## Prefill 阶段：大矩阵并行的计算密集型特征

在 Prefill（预填充）阶段，模型接收用户输入的整段提示词。假设输入序列长度为 $S$，并发批次为 $B$，隐藏层维度为 $D$。

![Prefill 阶段的大矩阵乘法计算：矩阵并行处理使计算单元充分饱和](/images/content-rebuild/1364b015b3d0a4db/fig-2.png)

### 为什么是 GEMM 且计算强度高？

在模型的 [Self-Attention 投影](/interview/llm-interview/attention-qkv.md) 和 [FFN 层](/interview/llm-interview/ffn-swiglu.md) 中，输入张量的形状为 $(B \times S, D)$，权重形状为 $(D, D_{\text{out}})$：
- **计算量**：矩阵乘法（GEMM）执行乘加计算，浮点运算量约为 $2 \times B \times S \times D \times D_{\text{out}}$ FLOPs。
- **显存访问量**：模型权重只需从 HBM 加载一次到片上 SRAM，大小为 $D \times D_{\text{out}} \times \text{sizeof(dtype)}$ 字节；输入的激活值大小为 $B \times S \times D \times \text{sizeof(dtype)}$ 字节。

当输入提示词长度较大（例如 $S = 2048$）时，加载一次权重参数，可以在芯片内部与这 2048 个 Token 向量连续完成并行点乘。此时的计算强度主要由批次与序列长度决定：
$$I_{\text{Prefill}} \approx \frac{2 \times B \times S \times D \times D_{\text{out}}}{(D \times D_{\text{out}} + B \times S \times D + B \times S \times D_{\text{out}}) \times \text{sizeof(dtype)}}$$

当权重维度占主导时，约分后计算强度与 $B \times S$ 呈正相关。在大提示词场景下，计算强度轻松超过 150 $\text{FLOPs/Byte}$，越过硬件拐点。因此，**Prefill 阶段的大部分算子运行在 Roofline 模型的水平峰值区，瓶颈主要在 Tensor Core 的浮点算力**。

---

## Decode 阶段：单步串行的访存密集型特征

在生成首字之后，模型进入逐字生成的 Decode 阶段。自回归机制决定了模型必须基于上一步预测出的单个 Token，来预测下一个 Token。

![Decode 阶段的 GEMV 访存瓶颈：单 Token 乘法导致计算单元等待 HBM 权重搬运](/images/content-rebuild/1364b015b3d0a4db/fig-3.png)

### 1. 为什么退化为 GEMV 会导致算力闲置？

在常见的低并发场景下（例如单个用户请求，$B=1$），每一步输入的序列长度只有 $S=1$：
- 输入向量的尺寸为 $(1, D)$。
- 每一层的前向传播需要将这个单向量与整个权重矩阵相乘，算子类型退化为**矩阵-向量乘法（GEMV）**。
- **单步计算量**：$2 \times 1 \times D \times D_{\text{out}}$ FLOPs。
- **单步访存量**：必须从 HBM 完整搬运整个权重矩阵 $D \times D_{\text{out}} \times \text{sizeof(dtype)}$ 字节到片上。

以 16 位浮点数（FP16/BF16，每参数 2 字节）为例，忽略极小的激活值开销，该运算的计算强度为：
$$I_{\text{GEMV}} \approx \frac{2 \times 1 \times D \times D_{\text{out}}}{D \times D_{\text{out}} \times 2\text{ Bytes}} = 1\text{ FLOP/Byte}$$

A100 的硬件拐点约为 $153\text{ FLOPs/Byte}$，而单步 GEMV 的计算强度仅有 $1\text{ FLOP/Byte}$，相差了两个数量级。这意味着 **GPU 的绝大部分时钟周期都在等待数据从显存搬运进芯片，Tensor Core 计算核心大多处于饥饿闲置状态**。

### 2. KV Cache 的空间与带宽代价

在 Decode 阶段，为了避免重复计算历史 Token 的 Key 和 Value，推理引擎会将它们缓存起来，即 [KV Cache](/interview/llm-interview/kv-cache.md)。

这里需要明确区分**注意力计算量**与 **KV Cache 显存容量**：
- **计算复杂度**：在 Prefill 阶段或全序列自注意力中，完整注意力矩阵 $Q K^T$ 的尺寸为 $S \times S$，总运算量与序列长度呈平方关系 $O(S^2)$；而在 Decode 阶段，每单步仅用当前 1 个 Token 的 Query 与历史 $S$ 个 Key 计算，单步计算量仅为 $O(S)$；
- **KV Cache 显存占用**：KV Cache 保存每个历史 Token 投影后的向量，存储大小为 $2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times S \times \text{sizeof(dtype)}$，它与已生成的上下文长度 $S$ 是**严格的线性关系 $O(S)$**。

虽然 KV Cache 的大小随长度线性增加，但在长上下文的多轮对话中，每生成一个新 Token，除了加载全量权重外，还必须把此前累积的所有历史 KV Cache 完整搬运进 SRAM 与当前的 Query 向量做点积。这进一步放大了显存带宽的读写压力。

---

## 面试追问：针对不同瓶颈的优化方案

针对 Prefill 与 Decode 不同的物理瓶颈，业界演进出了不同的软硬件协同优化路线。

![FlashAttention 算子融合：通过 SRAM 局部计算消除 HBM 中间矩阵读写](/images/content-rebuild/1364b015b3d0a4db/fig-4.png)

### 1. FlashAttention：解决长序列注意力中的访存冗余

在标准 Attention 计算中，中间矩阵 $S = Q K^T$ 和注意力权重 $P = \text{softmax}(S)$ 的尺寸为 $(S, S)$。在长序列下，反复将这个大张量写回 HBM 再读出，会造成剧烈的显存读写瓶颈。

FlashAttention 的核心不是裁剪运算量（其理论浮点运算量并没有减少，甚至因为重算少量增加），而是**访存感知（I/O-Aware）的算子融合**：
- 利用容量较小但带宽极高的片上 **SRAM**；
- 将输入序列切块（Tiling）放入 SRAM 中；
- 借助**在线 Softmax（Online Softmax）**递推更新分母与局部极大值，无需保存完整的全局注意力矩阵；
- 将 HBM 上的中间矩阵读写量从 $O(S^2)$ 降低到 $O(S)$，大幅缓解了长上下文在预填充与注意力算子中的 HBM 读写压力。

![PagedAttention 分页机制：解决显存碎片并支持并发请求共享](/images/content-rebuild/1364b015b3d0a4db/fig-5.png)

### 2. PagedAttention：消除 KV Cache 显存碎片

Decode 阶段的另一个系统挑战在于显存利用率。早期推理框架通常为每个请求预先分配最大可能长度（如 4096）的连续物理显存，这会带来：
- **内部碎片**：多数短回答并未用完预留空间；
- **外部碎片**：不同请求动态释放后，连续显存空间被切碎，导致显卡明明有剩余容量却无法接纳新请求。

PagedAttention 引入了操作系统的分页思想，把 KV Cache 切分为固定大小的块（Block/Page，例如 16 个 Token 一页）。逻辑上连续的上下文在物理上离散存储，按需申请；同时支持分支请求共享前缀块（如 Prompt 前缀共享），从而成倍提升了单卡可容纳的并发批次。

### 3. 投机采样：以算力换带宽的典型方案

既然 Decode 阶段每生成 1 个 Token 都要搬运一次完整的模型参数，导致显存带宽严重受限，那么是否可以一次生成多个 Token？

投机采样（Speculative Decoding）采用“小模型快速草稿 + 大模型并行验证”的模式：
- **草稿阶段**：使用体积较小的草稿模型（如 1B~3B）快速串行自回归生成 $K$ 个候选 Token；
- **验证阶段**：大模型（如 70B）将这 $K$ 个 Token 作为批量序列一次性输入，执行一次类似 Prefill 的前向计算；
- **核心收益**：大模型加载一次全量 70B 权重的显存访存开销，从原本只验证 1 个 Token，提升为并行验证 $K$ 个候选 Token，把原本落在访存受限区的小 GEMV 转换成了小 GEMM，显著提升了有效计算强度。

---

## 面试深挖：为什么增大并发 Batch Size 能大幅提升 Decode 吞吐？

面试官常会追问工程场景问题：“**为什么在自回归解码阶段，将并发请求的 Batch Size 从 1 提升到 16 或 32 时，显卡每秒输出的 Token 总数能成倍提升，而每个 Token 的单步耗时增加却很少？**”

![Batch Size 对计算强度的提升机制：单次权重读取复用给多请求向量](/images/content-rebuild/1364b015b3d0a4db/fig-6.png)

### 原理解释

用参数复用率与 Roofline 模型可以清晰解释这个现象：

1. **从 GEMV 向 GEMM 演进**：当 $B=1$ 时，计算是 $(1, D) \times (D, D_{\text{out}})$ 的向量-矩阵乘法；当并发增大到 $B=32$ 时，运算变为 $(32, D) \times (D, D_{\text{out}})$ 的小矩阵乘法。
2. **单次权重加载被 32 倍复用**：无论 $B=1$ 还是 $B=32$，模型参数总量 $W_{\text{bytes}}$ 是固定的。GPU 从 HBM 把权重搬进片上 SRAM 一次，这批权重就可以在片上与 32 个请求的输入向量同时完成乘加运算。
3. **计算强度成倍提升**：在模型权重搬运占主导的场景下（如中短上下文），单步显存访存量基本固定，而浮点运算量提升了 $B$ 倍，GEMM 线性层算子的计算强度近似与 $B$ 成正比：
   $$I_{\text{step}} \approx \frac{2 \times B \times D \times D_{\text{out}}}{W_{\text{bytes}}} \propto B$$
4. **掩盖访存延迟**：在 $B=1$ 时，GPU 耗时几乎全被显存搬运占用，计算核心严重饥饿。在权重占主导且未跨过硬件拐点前，增加批次带来的额外乘加运算刚好填补了原本闲置的计算时钟周期，权重搬运时间掩盖了计算时间。因此单步总耗时增加较平缓，而在相同时间内产出的 Token 总数成倍提升，吞吐（Tokens/s）表现出显著增长（若上下文极长导致多请求累加的 KV Cache 访存反超权重，则显存搬运耗时也会随并发明显上升）。

---

## 常见面试错误与避坑提示

| 常见错误表述 | 机制辨析与纠偏 |
| :--- | :--- |
| **“Transformer 的所有阶段都受限于显存带宽”** | 混淆了推理阶段。Prefill 阶段长序列大矩阵乘法通常是**计算密集型**；只有在低并发自回归 Decode 时才是典型的**访存密集型**。当 Decode 并发 Batch Size 达到极大值时，也可能进入计算受限区。 |
| **“KV Cache 的显存占用随文本长度呈平方级膨胀”** | 混淆了全序列注意力矩阵计算与 Cache 存储。标准全序列 Attention 计算包含 $S \times S$ 的完整点积矩阵，总计算量呈平方复杂度；但保留 Key 和 Value 向量的显存空间与序列长度 $S$ 是**严格的线性关系 $O(S)$**。 |
| **“FlashAttention 提速是因为减少了注意力的浮点计算量”** | FlashAttention 算出的数值与标准 Attention 完全一致，没有减少数学计算量（甚至因为在线 Softmax 的局部重算使 FLOPs 略有微增）。它的加速收益来自 **SRAM 分块与算子融合消除了中间大张量在 HBM 上的读写**。 |
| **“显存带宽不够时，只要剪枝降低模型 FLOPs 就能等比加速”** | 在 Roofline 模型的访存瓶颈区，耗时主要取决于数据搬运量（$T \approx \text{Bytes} / B_{\text{peak}}$）。如果减少计算量并未减少需要搬运的参数字节数，GPU 实际执行耗时几乎不会改善。 |
