---
title: ChatGLM 和 GPT 在网络架构上有什么区别？
icon: mdi:compare
tag:
- ChatGLM
- GPT
- 架构对比
description: 对比初代 ChatGLM 与 GPT 在注意力掩码、2D 位置编码及 KV Cache 上的机制差异，并分析后续版本为何回归单向因果架构。
---

![ChatGLM 与 GPT 架构差异概览](/images/content-rebuild/aff9f07fa83ff4aa/cover.png)

在生成式语言模型的发展过程中，GPT 系列从始至终坚持纯单向因果架构，而 ChatGLM 初代则基于混合掩码自回归（GLM 架构）设计。两者最核心的区别在于注意力掩码（Attention Mask）的组织方式和位置编码的维度。理解这些设计差异，能够帮助开发者看懂大模型在预训练目标、位置表征以及推理服务算子适配上的具体权衡。

---

## 面试 30 秒口述答案

如果面试官询问“ChatGLM 和 GPT 在网络架构上有什么区别”，可以用以下三点回答：

1. **自注意力机制与任务形式**：GPT 是标准的自左向右单向因果模型（Causal Decoder-only），整段输入使用下三角掩码；初代 ChatGLM（以及 ChatGLM-6B）采用自回归空白填充架构，输入分为未被遮蔽的前缀（Part A）和待生成的遮蔽片段（Part B），Part A 内部采用双向全连接注意力，Part B 内部采用单向因果自回归。
2. **位置编码体系**：GPT 采用一维递增位置编码（早期的绝对位置嵌入，现代架构如 GPT-4 路线多采用 RoPE）；初代 ChatGLM 为了让后置生成的片段能准确填补原句缺失位置，设计了 **2D 旋转位置编码（2D RoPE）**，分别记录原句绝对位置与片段内部自回归步长。
3. **架构演化路线**：从 ChatGLM2 开始，后续的 GLM-3 与 GLM-4 均已**彻底放弃双向掩码与 2D 位置编码，回归标准的单向因果 Decoder-only 架构**。这一转变主要是为了直接兼容 FlashAttention、PagedAttention 等行业主流高性能算子，降低高并发多轮对话下的前缀缓存管理成本。

---

## 贯穿例子：4 个 Token 的完形填空任务

为了直观理解这两种架构的不同，我们使用一个包含 4 个 Token 的简单句子作为贯穿示例：

假设原始句子切分为 4 个词：`[x1, x2, x3, x4]`。此时任务要求模型把被遮蔽的 `x3` 预测出来。

- **GPT 的处理流程**：只能按顺序从左往右看。如果要让 GPT 填补中间词，通常需要构造提示词（如“输入 x1 x2 [空] x4，输出是：”），模型按因果顺序逐字计算，每个位置都看不到后面的词。
- **初代 ChatGLM 的处理流程**：将原始句子拆分成两部分：
  - **Part A（前缀上下文）**：`t1=[x1], t2=[x2], t3=[MASK], t4=[x4]`。原句中的 `x3` 被替换为占位标记 `[MASK]`。
  - **Part B（待生成内容）**：`t5=[SOP], t6=[x3]`。以特定起始标记 `[SOP]`（Start of Prefix）开头，接着是要预测的真实 Token `x3`。

![自回归空白填充任务分解流程](/images/content-rebuild/aff9f07fa83ff4aa/fig-1.png)

---

## 预训练任务与注意力掩码矩阵对比

掩码矩阵决定了模型计算自注意力时，哪些位置之间允许传递信息。

### 1. GPT 的严格因果掩码

在标准因果自回归模型中，序列中第 $i$ 个 Token 只能依赖小于或等于 $i$ 的前文 Token。其注意力掩码矩阵 $M$ 为严格下三角结构：

$$M_{ij} = \begin{cases} 0, & i \ge j \\ -\infty, & i < j \end{cases}$$

在 Softmax 计算前，加在注意力打分矩阵上的掩码保证了后续词无法泄露给前文词。关于掩码算子的具体实现，可参考 [/interview/llm-interview/attention-masking.md](/interview/llm-interview/attention-masking.md)。

### 2. 初代 ChatGLM 的混合掩码

GLM 结合了 BERT 的双向语义理解与 GPT 的自回归生成能力，掩码矩阵由四块子区域组成：
- **Part A 对 Part A**：完全互视（全 $0$ 掩码）。`x1`、`x2`、`[MASK]`、`x4` 彼此完全可见，保留完整的双向语境。
- **Part A 对 Part B**：完全阻断（掩码为 $-\infty$）。前缀不能提前看到待恢复的真实内容。
- **Part B 对 Part A**：完全可见（全 $0$ 掩码）。生成的每一个词都可以参考完整的未遮蔽上下文。
- **Part B 对 Part B**：单向因果可见（下三角因果掩码）。自回归生成 Span 内部的 Token，后词不能泄露给前词。

![GLM 混合 Mask 与 GPT 因果 Mask 结构对比](/images/content-rebuild/aff9f07fa83ff4aa/fig-2.png)

### 3. 示例掩码矩阵数值展示

将贯穿示例中的 6 个 Token（4 个 Part A Token + 2 个 Part B Token）展开为 $6 \times 6$ 的注意力掩码矩阵（$0$ 表示允许计算注意力，$\times$ 表示阻断）：

$$
M = \begin{pmatrix}
0 & 0 & 0 & 0 & \times & \times \\
0 & 0 & 0 & 0 & \times & \times \\
0 & 0 & 0 & 0 & \times & \times \\
0 & 0 & 0 & 0 & \times & \times \\
0 & 0 & 0 & 0 & 0 & \times \\
0 & 0 & 0 & 0 & 0 & 0
\end{pmatrix}
$$

左上角 $4 \times 4$ 为全通双向区域，右下角 $2 \times 2$ 为严格因果下三角区域。这就是 GLM 所谓的双向掩码自回归。

---

<ReadMoreLock />

## 位置编码设计：2D 编码与 1D 编码

输入掩码打破了纯线性顺序后，一维递增位置编码无法满足需求，这是理解初代 ChatGLM 的重点。

![GLM 2D 位置编码与 GPT 1D 编码映射解析](/images/content-rebuild/aff9f07fa83ff4aa/fig-3.png)

### 1. 为什么初代 ChatGLM 需要 2D 位置编码？

在上述例子中，待恢复的 Token `x3` 物理上被移到了整个输入的末尾（作为 `t6` 传入）。如果直接使用标准一维递增编码 $[0, 1, 2, 3, 4, 5]$：
- 模型只能知道 `x3` 是输入里的第 6 个词，无法直接感知到它填补的是第 2 位（`[MASK]` 所在位置）的空缺；
- 如果一句话里挖出了多个不同的 Span（例如同时遮蔽 `x2` 和 `x4`），被抽出的片段可能以任意顺序排列，一维编码彻底丢失了它们与原始句子的物理对应关系。

为了解决这个问题，初代 ChatGLM 设计了两个维度的一维位置坐标：
- **Position 1（全局位置）**：记录当前 Token 在原始完整句子中的真实位置。对于 Part B 中生成的 Token，其 Position 1 统一固定为它在原句中所替换的 `[MASK]` 所在的位置。
- **Position 2（局部步长）**：记录当前 Token 在待恢复片段内部的相对生成顺序。对于 Part A 中的所有 Token，该值固定为 $0$；对于 Part B 中的 Token，从 $1$ 开始线性递增。

回到贯穿示例，每个 Token 分配的 2D 位置坐标如下：

| Token 标识 | 角色属性 | 文本内容 | Position 1 (原句位置) | Position 2 (片段内步长) |
| :--- | :--- | :--- | :--- | :--- |
| `t1` | Part A | `x1` | 0 | 0 |
| `t2` | Part A | `x2` | 1 | 0 |
| `t3` | Part A | `[MASK]` | 2 | 0 |
| `t4` | Part A | `x4` | 3 | 0 |
| `t5` | Part B | `[SOP]` | 2 | 1 |
| `t6` | Part B | `x3` | 2 | 2 |

在底层实现上，ChatGLM-6B 将 Rotary Position Embedding（RoPE）的隐藏层通道维度一分为二：前一半通道应用基于 Position 1 计算的旋转矩阵，后一半通道应用基于 Position 2 计算的旋转矩阵。关于 RoPE 旋转矩阵的推导细节，请阅读 [/interview/llm-interview/llama-rope.md](/interview/llm-interview/llama-rope.md)。

### 2. GPT 的一维位置编码

GPT 序列严格线性向前，Token 之间的相对距离完全由一维自然数递增表达：
- GPT-2 / GPT-3 使用可学习的一维绝对位置嵌入（Learned Absolute Positional Embeddings）；
- 现代因果模型（如 GPT-4 风格开源变体、LLaMA 系列）则全面采用一维 RoPE。

---

## 推理阶段与 KV Cache 的行为差异

在生产部署与高并发服务中，混合掩码与单向因果掩码对推理阶段的 [/interview/llm-interview/kv-cache.md](/interview/llm-interview/kv-cache.md) 会产生不同影响。

![双向 Prefill 与单向因果对 KV Cache 读写流向的影响](/images/content-rebuild/aff9f07fa83ff4aa/fig-4.png)

### 1. Prefill 与 Decode 阶段的计算行为

大模型推理通常分为预填充（Prefill，计算 Prompt 内部表示）与解码（Decode，逐步自回归生成后续 Token）两个阶段：

- **GPT 体系（标准因果 Decoder-only）**：
  - Prefill 阶段：因果掩码下，第 $i$ 个 Token 的 Key 和 Value 只依赖前文，产生对应长度的 Key 和 Value 向量顺序存入 KV Cache。
  - Decode 阶段：每生成一个新 Token，单向检索已有 Cache，并将当前位置生成的单个 Key/Value 向量追加到 Cache 末尾。
- **初代 ChatGLM（双向前缀）**：
  - Prefill 阶段：Part A 内部双向可见，所有前缀 Token 之间互相交互一次性完成状态提取，写入 Cache。
  - Decode 阶段：Part B 逐步生成，新 Token 检索 Part A 的全量 Cache 与 Part B 已生成的 Cache，并向后追加写入。

### 2. 算子优化与工程复用挑战

双向前缀看似提升了语境表达，却在实际推理部署中带来了两处明显的工程瓶颈：

1. **通用注意力内核适配成本高**：FlashAttention 主要是通过分块与核函数融合减少高带宽显存（HBM）的读写访存开销，并不改变注意力机制本身的二次复杂度计算量。它的标准硬件融合实现高度依赖单向因果下三角结构进行块级别的跳过。混合掩码由于存在局部的双向全连接与阻断，无法直接套用标准因果内核，往往需要编写专用算子，或者回退到通用掩码逻辑，降低了 GPU 吞吐效率。
2. **多轮对话前缀复用受限（Prefix Caching）**：在现代在线服务中，多个请求可能共享同一段系统提示词（System Prompt）。在纯单向因果模型中，每个 Token 的 Key 和 Value 仅取决于它前面的词，向后追加新内容不会改变前面已生成的表示，因此公共前缀的 KV Cache 可以被多个会话跨请求安全共享或跳过计算。而如果前缀内部采用双向依赖，任何在上下文中的改动或追加，都会导致双向交互区内所有 Token 的 Key 和 Value 发生改变，无法直接进行公共前缀缓存的跨请求复用与增量追加。

---

## 架构演化路线：GLM 为什么全面回归 Causal Decoder-only？

从技术演化看，智谱 AI 团队在推出 ChatGLM-6B 之后，从 ChatGLM2 开始逐步精简架构，后续的 GLM-3 与 GLM-4 完全转向标准的 Causal Decoder-only 方案。

![ChatGLM 体系向 Causal Decoder-only 收敛演化路线](/images/content-rebuild/aff9f07fa83ff4aa/fig-5.png)

### 1. ChatGLM 系列演进轨迹横向对比

| 版本 | 基础架构形式 | 注意力掩码形态 | 位置编码方案 | 注意力头结构 | 归一化策略 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ChatGLM-6B (一代)** | Prefix-LM 混合架构 | Part A 双向 + Part B 因果 | 2D RoPE 空间位置编码 | Multi-Head Attention (MHA) | DeepNorm (Post-LN 变体) |
| **ChatGLM2-6B** | Causal Decoder-only | 纯单向因果下三角 | 1D RoPE | Multi-Query Attention (MQA) | RMSNorm (Pre-LN) |
| **ChatGLM3 / GLM-4** | Causal Decoder-only | 纯单向因果下三角 | 1D RoPE（支持长外推） | Grouped-Query Attention (GQA) | RMSNorm + SwiGLU |

关于 MQA 与 GQA 显存节省机制的横向对比，可参见 [/interview/llm-interview/mha-mqa-gqa.md](/interview/llm-interview/mha-mqa-gqa.md)。关于 Decoder-only 成为行业主流的技术原因，可参见 [/interview/llm-interview/decoder-only.md](/interview/llm-interview/decoder-only.md)。

### 2. 促成这一技术收敛的三个核心考量

1. **预训练数据规模摊平了结构收益**：GLM 最初引入双向掩码填空是为了增强自然语言理解（NLU）表现。但在百亿、千亿参数及数万亿 Token 的训练规模下，纯单向自回归模型在各种理解类基准上的表现已完全不输混合架构，双向掩码带来的先验收益被大规模自回归预训练摊薄。
2. **主流推理生态的协同标准**：vLLM、TensorRT-LLM、SGLang 等高性能推理框架，核心调度和算子（如 PagedAttention）全部针对标准 Causal Decoder-only 深度优化。使用非标准掩码意味着需要为每个新算子定制分支，开发维护成本过高。
3. **训练架构的简化**：标准因果掩码配合 1D RoPE 让模型的数据流和流水线并行切分更加标准化，能无缝接入 Megatron-LM 等通用大规模训练框架。

![非标准 Mask 与定制算子在分布式高并发下的失败链路分析](/images/content-rebuild/aff9f07fa83ff4aa/fig-6.png)

---

## 面试高频追问与常见认知陷阱

### 追问 1：GLM 架构究竟属于 Encoder-Decoder 还是 Decoder-only？

**口述要点**：
GLM 本身是一个**单堆栈（Single-stack）Decoder-only 的变体**，并不包含独立的两套参数（不像 T5 拥有独立的 Encoder 和 Decoder），也没有跨注意力层（Cross-Attention）。所有的 Token 都在同一个 Transformer 深度堆栈中流转，仅仅通过注意力掩码矩阵人为划分出了双向交互区（Part A）与因果生成区（Part B）。

### 追问 2：为什么 ChatGLM-6B 初代采用 DeepNorm 而非标准的 Pre-LN？

**口述要点**：
标准的 Post-LN 在深层网络中容易发生梯度爆炸或残差路径不稳定，而早期的 Pre-LN 虽然训练稳定，但有研究认为深层表示能力受到一定限制。ChatGLM-6B（以及早期的 GLM-130B）为了兼顾深层模型的训练稳定度与表达能力，采用了微软提出的 DeepNorm 方案。该方案通过对残差连接添加缩放系数并结合特定权重初始化，使得深层模型在稳定训练的同时抑制梯度溢出。到了 ChatGLM2 及后续版本，随着技术收敛，模型全面切换为更轻量且训练稳定的 RMSNorm（详细对比参见 [/interview/llm-interview/layernorm-rmsnorm.md](/interview/llm-interview/layernorm-rmsnorm.md)）。

### 常见错误说法辨析

- **错误说法 1：“ChatGLM 全系列模型都采用了 2D 位置编码与双向填空机制。”**
  *事实纠正*：只有初代 GLM-130B 和 ChatGLM-6B 采用了 2D RoPE 与双向完型填空掩码。从 ChatGLM2 开始，智谱已彻底放弃双向掩码与 2D 编码，全面转向标准 1D RoPE 和单向因果 Decoder-only 架构。
- **错误说法 2：“双向前缀模型可以像标准 GPT 一样直接做任意前缀的 KV Cache 共享复用。”**
  *事实纠正*：双向前缀内部所有 Token 互相可见，其键值表示受前缀内全部上下文的联合影响。单向因果模型向后追加文本不改变历史 KV，因此公共系统提示可安全跨会话共享；而双向前缀只要内部内容变动或向后扩展双向区，整段前缀的 Key 和 Value 均会发生变动，无法安全跨请求共享或直接做增量前缀缓存。
- **错误说法 3：“LLaMA 采用了与 ChatGLM2 相同的 Multi-Query Attention（MQA）机制。”**
  *事实纠正*：ChatGLM2-6B 在开源模型中较早落地了 Multi-Query Attention（MQA，所有 Query 头共享 1 组 Key/Value 头）。而初代 LLaMA 全系与 LLaMA-2 7B/13B 均使用传统的 Multi-Head Attention（MHA，每个 Query 头拥有专属的 Key/Value 头），直到 LLaMA-2 70B 及后续版本才开始引入 Grouped-Query Attention（GQA）。
