---
title: KV Cache 是什么？为什么大模型自回归推理需要它与显存如何计算
icon: mdi:memory
tag:
- 大模型面试
- KV Cache
- 推理优化
description: 详解KV Cache工作原理、数据流与显存推导公式，厘清Prefill计算受限向Decode访存受限的硬件瓶颈转变与高频面试追问。
---

在针对大语言模型（LLM）推理性能与系统优化的面试中，**KV Cache（Key-Value Cache）** 是高频出现的核心考点。面试官通常会从基本概念出发，连续考察自注意力矩阵计算流、显存精确占用推导公式，以及自回归生成过程中硬件性能瓶颈的转移机理。

![KV Cache 原理与显存瓶颈全景图](/images/content-rebuild/5a120979d878a94a/cover.png)

---

## 1. 30秒口述结论（面试速答）

如果在面试中被问到“什么是 KV Cache？为什么需要它？”，可以用以下结构在 30 到 45 秒内给出清晰直接的回答：

> **核心口述模版：**
>
> “**KV Cache 是一种通过在显存中缓存历史 Key 和 Value 向量、消除重复投影与注意力计算的自回归推理优化技术。**
>
> 在使用 [Decoder-only 架构](/interview/llm-interview/decoder-only.md) 逐字生成时，由于[注意力因果掩码（Causal Masking）](/interview/llm-interview/attention-masking.md)的存在，历史 Token 只能看到它前面的上下文，不会受未来生成内容的影响。因此历史各位置的 Key 和 Value 向量在后续时间步是恒定不变的。
>
> 如果不加缓存，生成第 $t$ 个词时要把前 $t$ 个词整体重新做一次前向传播，全序列累计的注意力计算量为 $O(S^3)$，投影计算量为 $O(S^2)$；引入 KV Cache 后，历史各层的 K 和 V 已经常驻在显存中，单步解码只需计算当前最新 Token 的 Q、K、V 投影，计算复杂度降到单步 $O(S)$，全序列生成复杂度降为 $O(S^2)$。
>
> 它的主要代价是显存开销，显存占用随并发批次、序列长度、模型层数与隐藏层维度线性增长。这也使得自回归生成从 Prefill 阶段的**计算受限（Compute-bound）**转变为 Decode 阶段的**访存受限（Memory-bound）**。”

---

## 2. 贯穿示例与数据流：为什么 K/V 必须存，而 Q 可以随算随抛？

为了彻底搞清楚计算因果，我们以一个具体的贯穿例子来说明：假设用户的输入 Prompt 是 `今天天气真好，我想去`，模型要逐字生成后续内容 `公园散步`。

在 [Self-Attention 机制](/interview/llm-interview/attention-qkv.md) 中，输入向量矩阵 $X \in \mathbb{R}^{S \times d}$ 经由投影权重 $W_Q, W_K, W_V$ 映射得到 Query、Key 和 Value：

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

其中 $M$ 为因果下三角掩码矩阵（未来位置为 $-\infty$）。

![无 Cache 与有 Cache 生成第 t 个 Token 的数据流对比图](/images/content-rebuild/5a120979d878a94a/fig-1.png)

### 2.1 无 KV Cache 的重复计算浪费

假设模型已经生成了 `公` 字，当前处于第 $t$ 步，正在准备预测 `园` 字：

1. **无缓存做法**：将整个前序序列 `今天天气真好，我想去公`（长度为 $t-1$ 个 Token）以及预测位整体送入模型。
2. **重复计算发生在哪**：因果掩码保证了前 $t-1$ 个 Token 无法看到后面的词。也就是说，`天` 对 `今` 的注意力权重、`好` 对前面所有词的注意力权重，在预测 `公` 的时候算过一遍，现在为了预测 `园`，模型又把包含前 $t-1$ 个词的投影和注意力矩阵从头重算了一遍。
3. **复杂度分析**：如果生成长度为 $S$ 的序列，第 $t$ 步的自注意力计算量是 $O(t^2)$，累积注意力计算总量是 $\sum_{t=1}^S O(t^2) = O(S^3)$；每步对所有已知 Token 重新做线性投影的累积开销是 $O(S^2 \cdot d)$。这种算力浪费随着生成长度增加会迅速拖垮系统。

### 2.2 有 KV Cache 的计算与流向

引入 KV Cache 后，文本生成过程被拆分为两个明确阶段：

- **首字填充阶段（Prefill / Prompt 阶段）**：
  输入为全部 Prompt Token（`今天天气真好，我想去`）。模型并行计算所有位置的 $Q, K, V$ 并输出首个生成字 `公`。在此阶段，系统将输入序列对应的各层 Key 和 Value 矩阵全量写入 GPU 显存预设的缓冲区（KV Cache）中。
- **逐字解码阶段（Decode 阶段）**：
  预测下一个字 `园` 时，输入仅为上一时刻最新生成的**单个 Token**（`公`，对应向量 $x_t \in \mathbb{R}^{1 \times d}$）。
  1. **单 Token 投影**：仅计算当前 Token 的三个向量：
     $$q_t = x_t W_Q \in \mathbb{R}^{1 \times d_k}$$
     $$k_t = x_t W_K \in \mathbb{R}^{1 \times d_k}$$
     $$v_t = x_t W_V \in \mathbb{R}^{1 \times d_v}$$
  2. **追加写入缓存**：将新算出的 $k_t$ 和 $v_t$ 沿着时间维度拼接（Append）到显存的历史缓存中：
     $$K_{\text{all}} = [K_{\text{cache}}; k_t] \in \mathbb{R}^{t \times d_k}$$
     $$V_{\text{all}} = [V_{\text{cache}}; v_t] \in \mathbb{R}^{t \times d_v}$$
  3. **单行注意力计算**：计算当前 Token 对全部历史上下文的注意力权重并加权输出：
     $$A_t = \text{softmax}\left(\frac{q_t K_{\text{all}}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{1 \times t}$$
     $$O_t = A_t V_{\text{all}} \in \mathbb{R}^{1 \times d_v}$$
     *(注：在单序列标准因果自注意力下，$K_{\text{all}}$ 仅累积到当前步 $t$，天然不包含未来位置，因此无须显式添加下三角因果掩码；但在包含 Padding 对齐的批量推理、滑动窗口注意力或投机解码多 Token 验证等场景下，Decode 阶段仍需配合相应的注意力掩码矩阵)*。

### 2.3 为什么只缓存 K、V，而不需要缓存 Q？

- **Key 和 Value 的用途**：Key 表示“当前位置包含什么特征，供后续位置来匹配”，Value 表示“当前位置实际携带的信息”。当后续生成 `园`、`散`、`步` 时，每一个新词都需要回顾历史上下文，因此历史各词的 $K_{\text{all}}$ 与 $V_{\text{all}}$ 必须全部常驻显存供后续查询。
- **Query 的用途**：Query 表示“当前词正在寻找什么样的上下文信息”。在生成 `公` 的这一步，我们只需要 $q_{\text{公}}$ 去历史中检索；当下一步生成 `园` 时，执行检索任务的变成了新词的 $q_{\text{园}}$。旧的 $q_{\text{公}}$ 不会再被未来任何时间步使用。因此，Query 向量随用随算、用完即抛，完全不需要在显存中保留。

<ReadMoreLock />

---

## 3. 显存占用推导公式

在模型服务部署和面试精算中，KV Cache 的显存占用必须能从底层张量维度进行推导。

![KV Cache 张量维度结构与显存占用推导图](/images/content-rebuild/5a120979d878a94a/fig-2.png)

### 3.1 变量符号定义

- $B$ (Batch Size)：并发处理的请求批次大小。
- $S$ (Sequence Length)：当前序列总长度（Prompt 长度 + 已生成 Token 数）。
- $L$ (Num Layers)：Transformer 的 Decoder 块层数。
- $H$ (Hidden Size / $d_{\text{model}}$)：模型隐藏层维度。
- $n_{\text{heads}}$：Query 注意力头数。
- $n_{\text{kv\_heads}}$：Key/Value 注意力头数（标准 MHA 下等于 $n_{\text{heads}}$；GQA 或 MQA 下小于该值）。
- $d_{\text{head}}$：每个注意力头的维度，通常满足 $d_{\text{head}} = H / n_{\text{heads}}$。
- $P$ (Precision Bytes)：每个参数占用的字节数。FP16/BF16 为 2 字节，FP32 为 4 字节，INT8 为 1 字节。

### 3.2 标准 Multi-Head Attention (MHA) 推导

在标准多头注意力下，每个注意力头都维护独立的 Key 和 Value 缓存：

1. **单 Token、单层、单状态（K 或 V）参数量**：$n_{\text{heads}} \times d_{\text{head}} = H$
2. **单 Token、单层包含 K 和 V 两个状态**：$2 \times H$
3. **单 Token 穿透模型全部 $L$ 层所占的总参数量**：$2 \times L \times H$
4. **转换为显存字节大小**（乘以存储精度字节数 $P$）：
   $$\text{Bytes}_{\text{single-token}} = 2 \times L \times H \times P$$
5. **并发批次为 $B$、序列总长为 $S$ 时的系统总显存占用**：
   $$\text{Memory}_{\text{KV}} = 2 \times B \times S \times L \times H \times P \quad (\text{Bytes})$$

如果按照注意力头拆分展开，公式等价写作：
$$\text{Memory}_{\text{KV}} = 2 \times B \times S \times L \times n_{\text{heads}} \times d_{\text{head}} \times P$$

### 3.3 进阶架构变体（GQA 与 MQA）的调整公式

在 [Multi-Head Attention、MQA 和 GQA 有什么区别？](/interview/llm-interview/mha-mqa-gqa.md) 中分析过，现代开源模型（如 LLaMA-2-70B、LLaMA-3）普遍采用 GQA（分组查询注意力）来减少显存带宽和缓存容量压力。在计算公式中，只需将总头数替换为 KV 专用头数：

$$\text{Memory}_{\text{KV-GQA}} = 2 \times B \times S \times L \times n_{\text{kv\_heads}} \times d_{\text{head}} \times P$$

- **MQA (Multi-Query Attention)**：$n_{\text{kv\_heads}} = 1$，所有 Query 头共享同一组 KV，KV Cache 显存直接压缩至标准 MHA 的 $1 / n_{\text{heads}}$。
- **GQA (Grouped-Query Attention)**：每组共享一组 KV。例如 LLaMA-3-8B 中 $n_{\text{heads}} = 32$，$n_{\text{kv\_heads}} = 8$，KV Cache 显存压缩至标准 MHA 的 $1/4$。

---

## 4. Prefill 与 Decode 阶段的硬件瓶颈反差

理解了数据流与显存公式，就能从硬件底层解释为什么大语言模型在生成文字时，GPU 的 Tensor Core（计算核心）利用率通常非常低。这需要用到 GPU 的 Roofline 模型与算术强度概念。

![Prefill 与 Decode 阶段硬件负载对比图](/images/content-rebuild/5a120979d878a94a/fig-3.png)

### 4.1 算术强度（Arithmetic Intensity）

算术强度定义为单位内存数据搬移下能够执行的浮点运算次数：
$$\text{Arithmetic Intensity} = \frac{\text{计算量 (FLOPs)}}{\text{访存量 (Bytes)}}$$

当算法的算术强度高于硬件的“算力 / 带宽”临界比值时，属于**计算受限（Compute-bound）**；反之，若算术强度极低，硬件核心算力绝大部分时间处于空转等待显存加载数据的状态，属于**访存受限（Memory-bound）**。

| 阶段 | 核心矩阵算子 | 算术强度特征 | 硬件物理瓶颈 | 典型针对性技术 |
| :--- | :--- | :--- | :--- | :--- |
| **Prefill 阶段** | 矩阵乘矩阵 (GEMM) | 较高（同一权重矩阵与整句数百上千 Token 连续相乘） | **计算受限 (Compute-bound)** | FlashAttention、算子融合、张量并行 |
| **Decode 阶段** | 矩阵乘向量 (GEMV) | 极低（搬入数十 GB 权重和缓存，仅计算 1 个 Token） | **访存受限 (Memory-bound)** | PagedAttention、KV Cache 量化、GQA、投机采样 |

### 4.2 Decode 阶段的访存墙现实

在逐字生成（Decode）阶段，针对每一步生成的单个 Token，GPU 必须在硬件底层完成两项极其沉重的数据搬移：
1. **读取全量模型权重**：为了完成线性投影与 MLP 前向计算，必须把数十 GB 的静态权重从 GPU 全局显存（HBM）完整读取到片上缓存（SRAM）一次。
2. **读取全量历史 KV Cache**：为了计算自注意力，必须把历史所有的 $K$ 和 $V$ 向量从 HBM 加载到计算单元，与当前单个行向量 $q_t$ 做点积。

这一轮搬移数十甚至上百 GB 的数据，最终却只服务于 1 个 Token 的生成。在单请求或小 Batch 下，算术强度通常跌至个位数 FLOP/Byte，远低于现代计算卡（如 NVIDIA H100 算力与显存带宽比值通常在数十至上百 FLOP/Byte）的平衡点。

因此，**在典型的小批次或低并发场景下，Decode 阶段的首字后单步生成延迟，主要取决于显存物理带宽（HBM 带宽），而不是 GPU 的理论峰值 TFLOPs**；只有当并发批次足够大时，向量乘法聚合为批处理矩阵乘法（GEMM），算术强度提升，系统吞吐瓶颈才可能重新向计算核心算力或显存容量上限转移。关于硬件瓶颈的系统分析，可参阅 [Transformer 模型的性能瓶颈在哪？](/interview/llm-interview/transformer-bottleneck.md)。

---

## 5. 面试高频精算与追问解析

### 5.1 现场手算：13B 模型显存占用估算

> **面试题：**
> 假设一个 13B 规模的模型，参数为：40 层 Transformer（$L=40$），隐藏层维度为 5120（$H=5120$），标准多头注意力（MHA），数据类型为 FP16。
> 请推导计算：当批次大小 $B=4$、序列总长度达到 4096 时，仅 KV Cache 一项需要占用多少 GiB 显存？

![13B 模型在 Batch=4、Seq=4096 时的 KV Cache 显存精算面板](/images/content-rebuild/5a120979d878a94a/fig-4.png)

**解题推导步骤：**

1. **确定精度参数**：FP16 每个参数占用 2 字节，即 $P = 2$ 字节。
2. **计算单 Token 对应的 KV Cache 字节数**：
   $$\text{Bytes}_{\text{single-token}} = 2 \times L \times H \times P = 2 \times 40 \times 5120 \times 2 = 819,200 \text{ 字节} = 800 \text{ KiB}$$
3. **计算总 Token 规模**：
   $$\text{Total Tokens} = B \times S = 4 \times 4096 = 16,384$$
4. **计算总显存占用并换算**：
   $$\text{Total Bytes} = 819,200 \times 16,384 = 13,421,772,800 \text{ 字节}$$
   换算为标准的吉字节（GiB，以 $1024^3$ 进位）：
   $$\text{Memory}_{\text{GiB}} = \frac{13,421,772,800}{1024^3} = \frac{13,421,772,800}{1,073,741,824} = 12.5 \text{ GiB}$$
   *(若按工程十进制 $10^9$ 计算，则约为 $13.42 \text{ GB}$。面试时需显式说明 $1024^3$ 进位方式，体现严谨性)*。

**考点延伸对比**：
13B 模型在 FP16 精度下的模型静态权重本身约为 $13 \times 2 = 26 \text{ GB}$。而在上述配置下，仅仅 4 个中长并发请求，KV Cache 占用的显存（$12.5 \text{ GiB}$）就已经达到了模型静态权重的一半。这直观解释了长上下文推理中显存溢出（OOM）的主要来源。

### 5.2 追问 1：既然 KV Cache 效果显著，为什么训练阶段不用？

1. **训练机制采用 Teacher Forcing**：训练时模型的真实目标标签（Ground Truth）是已知的。输入的是包含整句话的长张量，通过带因果下三角掩码的矩阵乘法并行计算全句所有位置的损失。这是矩阵级别的单次前向传播，**不存在时间步之间的逐步自回归循环**。
2. **训练时激活值管理完全不同**：训练阶段的前向传播为了反向传播求导，本身就会保留全序列完整的 Key 和 Value 中间激活值（Activation）。如果在训练中强行引入推理式的动态增量拼接，不仅无法提升并行效率，反而会破坏高效的整块 GEMM 矩阵计算。推理用的 KV Cache 是为了避免逐步生成的重复计算，在全序列并行的训练中无此需求。

### 5.3 追问 2：FlashAttention 能降低持久化的 KV Cache 显存占用吗？

- **核心边界**：**不能直接降低已持久化存储的 KV Cache 显存容量。**
- **因果机制说明**：FlashAttention 是针对 Attention 算子执行过程中的 GPU 显存读写（Kernel IO）优化。它通过分块计算（Tiling）和在线 Softmax 技术，避免了将中间庞大的 $S \times S$ 注意力权重矩阵写入全局显存（HBM），而是直接在高速片上缓存（SRAM）中完成累加。但是，自回归推理中历史 Token 本身的 Key 和 Value 向量，是跨时间步必需的上下文输入，仍然必须完整保存在 HBM 中。要压缩 KV Cache 本身的显存容量，必须采用 GQA/MQA 架构压缩、KV Cache 量化（如 INT8/FP8/INT4）或稀疏窗口策略。

---

## 6. 工程落地陷阱与常见误区

![显存碎片与动态增长问题解析图](/images/content-rebuild/5a120979d878a94a/fig-5.png)

### 误区 1：在显存中为每个并发请求预先分配最大长度的连续缓冲区
- **缺陷分析**：早期推理引擎按模型最大上下文长度（例如 4096）静态连续分配显存空间。但真实业务中用户的 Prompt 长度不可预知，生成的回答可能只有几十个字。这种静态连续分配造成了大量未被利用的显存槽位（**内部碎片**），同时长块连续分配容易耗尽地址空间导致分配失败（**外部碎片**）。
- **工业界解决方案**：以 vLLM 的 **PagedAttention** 为代表的技术，借鉴了操作系统虚拟内存分页的思想，将 KV Cache 离散切分为固定大小的内存块（Block，如 16 个 Token 一页），按需动态分配，通过逻辑页表映射物理块。它彻底消除了显存分配中的**外部碎片**；而固定块大小带来的**内部碎片**则被严格限制在每个序列的最后一个 Block 内（浪费率通常降至 4% 以下），使实际推理并发容量成倍提升。

### 误区 2：混淆模型权重与 KV Cache 的显存动态特性
- **模型权重显存**：在模型加载完毕后是**静态固定**的，显存占用与并发批次和序列长度无关。
- **KV Cache 显存**：是**随生成时间步严格线性动态增长**的。估算服务最大并发容量时，如果不限定具体的目标序列长度 $S$，得出的显存余量结论没有任何工程指导意义。

![面试应答自查结构树](/images/content-rebuild/5a120979d878a94a/fig-6.png)

---

## 7. 学习效果自测

你可以通过回答以下两个题目，检验自己对本文核心机制与工程量化的掌握程度：

1. 写出标准 FP16 精度下，单 Token 在全模型中对应的 KV Cache 显存字节计算公式（使用层数 $L$ 与隐藏层维度 $H$ 表示）。
2. 在自回归解码（Decode）阶段，限制 GPU 生成吞吐的最主要硬件物理瓶颈是什么？FlashAttention 能够直接缩减显存中持久化存储的 KV Cache 总体积吗？
