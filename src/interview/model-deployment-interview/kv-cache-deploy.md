---
title: KV Cache 在大模型部署中有什么作用？显存开销如何计算？
icon: mdi:memory
tag:
- 模型部署面试
- KV Cache
- 显存
description: 解析大模型推理中 KV Cache 的原理、单 Token 显存计算公式、MHA/GQA 架构差异及服务部署中的生命周期与优化方法。
---

![KV Cache 部署原理封面图](/images/content-rebuild/d6f206c584bcdd36/cover.png)

在评估大语言模型部署架构与显存规划时，**KV Cache（键值缓存）**是面试中考察推理性能瓶颈与物理资源分配的核心知识点。本文通过一个具体的文本生成过程，解释为什么自回归解码会产生重复计算，KV Cache 如何消除这些冗余，如何定量计算它的显存占用，以及它对在线推理吞吐带来的物理限制。

---

### 面试口述：一分钟讲清 KV Cache 的作用与瓶颈

如果在面试中被问到“请解释一下 KV Cache 的作用以及它带来的显存问题”，可以直接按以下结构作答：

> **核心口述：**  
> 大语言模型的文本生成是逐个 Token 进行的自回归过程。由于因果注意力掩码（Causal Mask）的存在，前文已经生成的 Token 其语义表示不会受到后续 Token 的影响。这意味着，历史 Token 在每一层计算出的 Key 和 Value 向量是固定的。  
>   
> 如果没有缓存，每生成一个新 Token，都需要将全部历史上下文重新输入模型，重新做全量的线性投影和网络前向传播，使得每生成一步就要承担一次与前文长度成正比的冗余计算。  
>   
> **KV Cache 的本质是“以空间换时间”**：在 Prefill（预填充）阶段计算好 Prompt 中所有 Token 的 Key 和 Value 并保留在显存中；在 Decode（解码）阶段，每一步只需要计算当前新 Token 自身的 Query、Key、Value，并将新的 Key、Value 追加到显存缓存中，然后与历史所有的 Key、Value 做单步注意力计算。  
>   
> **代价与瓶颈：**  
> 虽然去除了历史投影计算，但保存这些中间向量会导致显存占用随请求并发数（Batch Size）和上下文长度（Sequence Length）线性增长。在高并发和长上下文场景下，KV Cache 占用的显存往往会超过模型静态权重本身，成为系统并发容量上限的主要限制因素。

---

### 机制分析：自回归解码中的计算冗余与消除

为了清晰理解这一过程，我们使用一个贯穿全篇的例子：用户输入 Prompt 为“**今天天气真好，我们去**”（共 7 个 Token），模型接下来依次生成“**公**”、“**园**”、“**散**”、“**步**”。

![无缓存与有缓存计算数据流对比](/images/content-rebuild/d6f206c584bcdd36/fig-1.png)

设输入序列的隐藏状态为 $X = [x_1, x_2, \dots, x_t]$，单层 Transformer 的线性投影矩阵为 $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$。

#### 1. 无缓存状态下的计算过程

当输入 Prompt（7 个 Token）进入模型后，模型预测出下一个字是“公”（第 8 个 Token）。  
现在进入生成第 9 个 Token（“园”）的步骤：

1. 模型的输入变成了 8 个 Token：$[x_1, \dots, x_7, x_8]$；
2. 如果不保留中间状态，每一层都需要重新计算整个序列在投影矩阵上的结果：
   $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
3. 这里的 $x_1, \dots, x_7$ 已经在这之前被 $W_K, W_V$ 乘过了，但因为没存下来，系统必须对前 7 个 Token 再次执行矩阵乘法。

随着生成内容增加，生成第 $t$ 个 Token 时，前 $t-1$ 个 Token 的线性投影与前馈网络都需要全量重新计算一次。若总生成步数为 $N$，Prompt 长度为 $S$，累计投影计算量与 $\sum_{t=S}^{S+N} t$ 成正比，投影计算复杂度劣化为 $O((S+N)^2)$。

#### 2. 引入 KV Cache 后的计算流

自回归注意力的核心特征在于：计算第 $t$ 步的输出时，因果掩码遮蔽了未来信息，使得当前步仅需要当前 Token 的 Query 向量 $q_t$，去和前文及自身的所有 Key、Value 向量做注意力聚合：
$$\text{Attention}(q_t, K_{\le t}, V_{\le t}) = \text{Softmax}\left(\frac{q_t K_{\le t}^T}{\sqrt{d_k}}\right) V_{\le t}$$
其中：
$$K_{\le t} = [k_1, k_2, \dots, k_t], \quad V_{\le t} = [v_1, v_2, \dots, v_t]$$

因为参数矩阵 $W_K, W_V$ 是静态的，对于历史位置 $i < t$ 的 Token，其计算结果 $k_i = x_i W_K$ 和 $v_i = x_i W_V$ 永远不会改变。因此，第 $t$ 步只需执行以下操作：

1. **单 Token 投影**：仅对当前输入的新 Token $x_t$ 进行计算，得到 $q_t, k_t, v_t$；
2. **更新缓存**：将 $k_t$ 与 $v_t$ 追加到显存中的历史缓存数组尾部；
3. **单向量与历史矩阵的点积**：从显存中读取完整的 $K_{\le t}$ 与 $V_{\le t}$，与 $q_t$ 执行向量-矩阵乘法（GEMV）。

引入 KV Cache 后，每一步的线性投影计算从针对整个序列降为仅针对当前 1 个 Token（单步投影计算量为 $O(1)$，总投影计算量降为 $O(N)$）。

需要指出的是：**Decode 阶段单步注意力的计算复杂度并没有变成常数**。因为 $q_t$ 必须与前文所有已经缓存的 $K$ 向量做内积，单步注意力点积的计算量仍与当前总长度 $t$ 成正比。KV Cache 省去的是历史 Token 在 QKV 投影层以及前馈网络（FFN）中高昂的矩阵乘法开销。

<ReadMoreLock />

---

### 单 Token 显存开销公式推导与算例

虽然 KV Cache 节省了重复计算时间，但需要持续占用 GPU 显存。在容量规划与并发预估时，可以通过参数维度精确推导显存占用。

![单 Token KV Cache 显存张量构成维度示意](/images/content-rebuild/d6f206c584bcdd36/fig-2.png)

#### 1. 显存计算通用公式

对于标准的 Transformer 解码器架构，输入序列中的每一个 Token，在模型的每一个自注意力层中都需要保存一份 Key 向量和一份 Value 向量。

单个 Token 在单层中包含的元素数量为：
$$\text{Elements}_{\text{layer}} = 2 \times (n_{\text{kv}} \times d_{\text{head}})$$
- 系数 $2$：分别存储 Key 和 Value；
- $n_{\text{kv}}$：模型当前层的 KV 注意力头数（Key/Value Heads）；
- $d_{\text{head}}$：每个注意力头的维度大小（Head Dimension）。

累加模型总层数 $L$ 并乘以存储每个数值所需的字节数 $P_{\text{bytes}}$，即可得出**单个 Token 在整个模型中占用的 KV Cache 显存大小**：
$$\text{Bytes}_{\text{token}} = 2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times P_{\text{bytes}}$$

当系统处理并发请求数为 $B$，平均上下文总长度（Prompt 长度加上已生成 Token 长度）为 $S$ 时，KV Cache 占用的显存总量为：
$$\text{Total Memory}_{\text{KV}} = B \times S \times \text{Bytes}_{\text{token}}$$

#### 2. 主流模型算例：Llama-3-70B

以常见的 **Llama-3-70B** 模型配置为例：
- 模型层数 $L = 80$
- 模型采用 GQA（分组查询注意力），KV 头数 $n_{\text{kv}} = 8$
- 单头维度 $d_{\text{head}} = 128$
- 权重与推理精度为 16 位半精度（FP16/BF16），每个数值占 $P_{\text{bytes}} = 2$ 字节

代入公式计算单个 Token 的显存开销：
$$\text{Bytes}_{\text{token}} = 2 \times 80 \times 8 \times 128 \times 2 \text{ 字节} = 327,680 \text{ 字节} \approx 320 \text{ KB}$$

由此可以估算不同场景下的 KV Cache 显存：
- **单请求 4K 上下文（4,096 Token）**：$4,096 \times 320 \text{ KB} \approx 1.25 \text{ GB}$
- **单请求 32K 上下文（32,768 Token）**：$32,768 \times 320 \text{ KB} \approx 10 \text{ GB}$
- **并发 16 个 32K 上下文请求**：$16 \times 10 \text{ GB} = 160 \text{ GB}$

Llama-3-70B 自身的模型静态权重在 FP16 下约为 140 GB（采用张量并行切分到多卡）。在仅有 16 个长文本请求并发时，KV Cache 占用的显存（160 GB）就已经超过了模型自身的权重体积。这解释了为什么长上下文部署往往受限于显存容量。

---

### 注意力架构对显存的影响：MHA、GQA 与 MQA

为了降低 KV Cache 对显存的占用，注意力结构的头设计经历了由 MHA 到 MQA、再到 GQA 的演进。

![MHA、GQA 与 MQA 的头结构与显存占用对比](/images/content-rebuild/d6f206c584bcdd36/fig-3.png)

#### 1. MHA（多头注意力）
在标准多头注意力中，Query、Key、Value 的头数相等，即 $n_q = n_{\text{kv}}$。  
若前述 70B 模型采用 MHA（假设 Query 头数为 64，则 $n_{\text{kv}} = 64$）：
- 单 Token 显存开销将变为 $2 \times 80 \times 64 \times 128 \times 2 = 2,621,440 \text{ 字节} = 2.5 \text{ MB}$；
- 单个 4K 请求就需要消耗 10 GB 的 KV Cache 显存；
- 在 Decode 阶段，由于每一步都要从显存加载庞大的 Key/Value 张量，推理速度会严重受限于显存带宽（Memory Bandwidth Bound）。

#### 2. MQA（多查询注意力）
在 MQA 中，所有的 Query 头共享同一对 Key 和 Value 头，即强制令 $n_{\text{kv}} = 1$。  
- **效果**：KV Cache 显存缩小至 MHA 的 $1 / n_q$。若有 64 个 Query 头，显存占用降低约 98.4%；
- **代价**：模型将所有注意力头的上下文表示强行压缩进单一子空间，在需要精细依赖或长链推理的任务中，模型生成质量常会出现可测量的下降。

#### 3. GQA（分组查询注意力）
GQA 是在模型表现力与显存开销之间的折中方案。它将 $n_q$ 个 Query 头均分为 $G$ 个组，每组共享一个 Key 头和一个 Value 头，因此 KV 头数 $n_{\text{kv}} = G$，每组包含 $n_q / G$ 个 Query 头。
- **典型参数**：Llama-3-70B 中，Query 头为 64，分为 8 组，每组 8 个 Query 头对应 1 个 KV 头（$n_{\text{kv}} = 8$）；
- **效果**：相比 MHA，KV Cache 显存开销直接缩减至原来的 $1/8$（节省了 87.5% 的 KV Cache 显存），同时基本维持了接近 MHA 的建模表达能力。

---

### KV Cache 的生命周期管理：Prefill、Decode 与释放

在推理服务引擎内部，每个推理请求的 KV Cache 会经历以下三个主要阶段：

![KV Cache 在 Prefill、Decode 到释放的完整生命周期状态流转](/images/content-rebuild/d6f206c584bcdd36/fig-4.png)

#### 阶段一：Prefill（预填充阶段）——批量写入
- **输入**：用户传入的 Prompt（如本例中的 7 个字“今天天气真好，我们去”）。
- **计算行为**：Prompt 中的 Token 是已知的，因此无需串行处理。模型通过矩阵乘法（GEMM）并行完成所有输入 Token 的注意力计算。
- **缓存行为**：一次性为这 7 个 Token 计算出所有层的 Key 与 Value 向量，并全量写入显存缓存中。
- **系统特点**：Prefill 阶段通常属于计算密集型（Compute-bound），GPU 算力利用率较高。这一阶段的计算耗时决定了首 Token 返回延迟（Time to First Token, TTFT）。

#### 阶段二：Decode（逐步解码阶段）——单步读取与动态追加
- **输入**：上一步模型输出的单个 Token（例如第 8 步生成的“公”）。
- **计算行为**：仅投影当前 1 个 Token 的 $q_t, k_t, v_t$。
- **缓存行为**：将新得到的 $k_t, v_t$ 写入缓存尾部；接着 Attention 算子从显存读取前文全部的 Key 和 Value 向量与当前 $q_t$ 计算注意力。
- **系统特点**：Decode 阶段属于访存密集型（Memory-bound）。模型参数与累积的历史 KV Cache 每一轮都需要从显存加载到 GPU 计算核心（SRAM），而每个 Token 计算的浮点运算量相对有限，显存读写带宽成为主要的延迟瓶颈。

#### 阶段三：Termination（请求结束）——显存回收
- **触发**：模型输出终止符（如 `<|eot_id|>`）或达到最大输出长度。
- **系统行为**：推理框架注销该请求的上下文信息，将对应的显存存储块标记为可用，供新请求调度分配。

---

### 面试追问与常见工程陷阱

![长文本与高并发场景下动态显存挤压并发上限示意](/images/content-rebuild/d6f206c584bcdd36/fig-5.png)

#### 追问 1：长上下文高并发下显存严重不足，工程上有哪些主流解决方案？

回答时建议从**结构选型**、**内存管理**和**压缩技术**三个层面进行梳理：

1. **结构设计**：优先选择使用 GQA 或 MQA 架构的模型，从网络结构上降低单个 Token 的 KV 头数；
2. **内存分配优化（分页机制）**：原生连续显存分配容易因请求长度动态变化产生严重的外部显存碎片与过度预留。使用如 vLLM 中的 **PagedAttention** 技术，将显存划分为固定大小的物理块（Block），允许物理上非连续存储并通过块表动态映射，基本消除了外部显存碎片并将内部碎片严格限制在末尾物理块内，实现按需分配；
3. **前缀缓存（Prefix Caching）**：对于带有相同 System Prompt 或固定少样本示例的多轮对话，让并发请求共享相同前缀的只读 KV Cache，避免重复存储；
4. **精度量化**：将 KV Cache 的精度从 FP16 降低到 FP8 或 INT4。以 FP8 为例，存储开销可直接缩减 50%，并在支持对应数据类型的计算单元上减轻访存带宽压力。

#### 追问 2：如果在服务中按每个请求的“最大支持长度”静态分配显存，会发生什么？

![静态预分配导致严重显存碎片与并发崩塌的失败分析](/images/content-rebuild/d6f206c584bcdd36/fig-6.png)

> **回答逻辑：**  
> 这会导致**严重的内部显存碎片（Internal Memory Fragmentation）**，并使服务的**并发吞吐大幅降低**。  
> 如果系统按模型支持的最大长度（例如 32K Token）为每个请求预先分配连续显存：  
> 1. 实际线上场景中，多数用户的实际请求长度可能只有几百 Token（例如 500 Token），分配给该请求的其余 31.5K 空间在整个生命周期中处于闲置状态，却被独占锁定；  
> 2. 80 GB 的显卡可能在并发数仅达到 2 或 3 时就报告显存耗尽（OOM），导致大量计算资源闲置；  
> 3. 这也是推理框架普遍引入动态分页内存管理的主要原因。

#### 追问 3：为什么模型只缓存 Key 和 Value，不缓存 Query？

> **回答逻辑：**  
> 在自回归解码中，历史 Token 的 Query 向量不需要再次参与后续计算。  
> 注意力的计算公式是 $\text{Softmax}(Q K^T) V$。当生成第 $t$ 个 Token 时，我们需要的是**当前最新 Token** 的 Query 向量 $q_t$，去与所有历史位置的 Key 做相关性打分，并据此对历史位置的 Value 做加权求和。  
> 历史步骤产生的 $q_1, \dots, q_{t-1}$ 在各步生成完成后已经完成了使命，后续无论生成多少步，都不会再使用历史的 Query。因此显存中只需保留 Key 和 Value，缓存 Query 只会带来无意义的显存开销。
