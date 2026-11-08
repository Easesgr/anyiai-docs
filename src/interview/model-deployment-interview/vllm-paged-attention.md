---
title: vLLM 是什么？PagedAttention 为什么能提升推理吞吐？
icon: mdi:speedometer
tag:
- 模型部署面试
- vLLM
- PagedAttention
description: 深入剖析 vLLM 的分页显存管理与 PagedAttention 算子机制，解析消除碎片、提升推理并发吞吐与写时复制的工程原理与面试考点。
---

在模型部署面试中，vLLM 与 PagedAttention 是高频考点。面试官考察的核心不仅是“吞吐提升了多少倍”，而是你能否从底层 GPU 显存分配缺陷、操作系统的分页虚拟内存映射，以及 CUDA 算子层面的非连续物理寻址，讲清因果链路与工程边界。

学习本篇前，建议先阅读 [KV Cache 在模型部署中的作用](/interview/model-deployment-interview/kv-cache-deploy.md) 了解基础的自回归缓存机制，后续可进一步学习 [连续批处理 Continuous Batching](/interview/model-deployment-interview/continuous-batching.md) 理解请求级动态调度。

![vLLM 与 PagedAttention 核心架构与分页机制全景](/images/content-rebuild/16a4625896e5f67b/cover.png)

---

## 口述答案：面试 2 分钟极简表达

面试中遇到“**vLLM 是什么？PagedAttention 为什么能显著提升吞吐？**”时，可按以下结构作答：

> **核心定义**：
> vLLM 是一个专为大语言模型设计的高吞吐、低延迟推理与服务框架。其核心技术是 **PagedAttention**——一种受操作系统虚拟内存分页机制启发的高效注意力机制与显存管理算法。
>
> **解决的痛点**：
> 传统推理引擎要求单个序列的 [KV Cache](/interview/model-deployment-interview/kv-cache-deploy.md) 在物理显存中保持连续。由于模型生成长度事先不可知，系统通常不得不按照最大序列长度预分配连续显存，或者频繁扩容重排，导致 60%~80% 的显存被预留浪费或切碎成无法利用的碎片。
>
> **吞吐提升机制**：
> 1. **消除连续性约束**：将 KV Cache 切分为固定容量的逻辑块（如 16 或 32 个 Token），通过块表（Block Table）映射到非连续的物理显存块，将显存浪费降至 4% 以下；
> 2. **提升并发批处理量（Batch Size）**：自回归解码（Decode）阶段严重受限于显存带宽（Memory-bound）。显存浪费消除后，有限显存能同时容纳数倍的并发请求，GPU 算力利用率大幅上升，最终成倍拉升服务系统的每秒 Token 吞吐；
> 3. **零额外显存共享**：借助引用计数实现**写时复制（Copy-on-Write, CoW）**，在束搜索、并行采样和前缀提示词复用场景中，多分支直接共享只读物理块，免去张量深拷贝开销。

---

## 贯穿算例：传统分配下的显存瓶颈

为了直观理解传统连续显存分配的损耗，我们以 **Llama-2-13B**（40 层 Transformer，40 个注意力头，头维度 $D_{head} = 128$）为例，采用 FP16 精度（每个数值 2 字节）进行推演。

单个 Token 在单层中同时保存 Key 和 Value 向量所占用的显存字节数为：

$$\text{Size}_{\text{token}} = 2 \times H_{kv} \times D_{head} \times \text{BytesPerElement}$$

代入模型参数：

$$2 \times 40 \times 128 \times 2 \text{ bytes} = 20,480 \text{ bytes} = 20 \text{ KB}$$

全模型共有 40 层，因此每新增一个 Token，该请求的 KV Cache 净增：

$$20 \text{ KB} \times 40 = 800 \text{ KB}$$

![传统连续显存预分配下的内部碎片与外部碎片问题](/images/content-rebuild/16a4625896e5f67b/fig-1.png)

假设系统支持的最大序列长度（Max Context Length）为 2048。在传统连续显存分配模式下，由于必须保证张量内存地址连续，且无法预知用户会输入多少字、输出多少字，分配器面临两难：

* **极端预分配方案**：请求刚进入系统，立即为其划分 $2048 \times 800 \text{ KB} \approx 1.6 \text{ GB}$ 的连续物理空间。若该用户实际只生成了 100 个 Token（仅需 80 MB），剩余约 1.52 GB 显存全程被无意义锁定，无法分给其他请求。这属于**预留碎片（Reservation Fragmentation）**；
* **动态几何扩容方案**：每次长度超出当前容量时申请更大的连续空间并执行拷贝。由于各个请求并发到达且生成长度完全异步，长短不一的显存块频繁申请与释放，会迅速产生大量无法利用的显存空隙，即**外部碎片（External Fragmentation）**。

在大模型解码阶段，生成一个 Token 的浮点运算量相对较小，主要瓶颈在于将模型权重和庞大的 KV Cache 读入 GPU 缓存的访存延迟（Memory-bound）。要提升整体吞吐，关键在于**做大并发 Batch Size**。由于上述碎片的存在，一张 80GB 的 A100 显存往往仅跑了十几个并发就报 OOM，吞吐无法打满。<ReadMoreLock />

---

## 核心设计：Logical Blocks 与 Block Table 映射

操作系统通过“虚拟页”到“物理页框”的映射解除了内存物理连续的枷锁。vLLM 将这一机制完整搬到了 GPU 显存管理中。

![Logical Blocks、Physical Blocks 与 Block Table 映射关系](/images/content-rebuild/16a4625896e5f67b/fig-2.png)

### 1. 核心数据结构

* **逻辑块（Logical Block）**：对单个请求序列而言，其逻辑上下文依然是连续递增的。系统将连续的 Token 序列按固定步长划分，步长即为 `Block Size`（常见为 16 或 32 个 Token）。
* **物理块（Physical Block）**：GPU 显存初始化时被格式化为一个全局统一的物理块池（Physical Block Pool）。每个物理块刚好能容纳 `Block Size` 个 Token 的一层或多层 Key/Value 数据。各个物理块的物理地址完全可以不连续。
* **块表（Block Table）**：CPU 端的调度器为每个请求维护一个 Block Table。表中每一行记录一个逻辑块映射到哪一个物理块 ID，以及当前物理块已填充的 Token 数量（Offset）。

### 2. 动态自回归写入流程

设定 `Block Size = 4`，观察一个请求从 Prompt 阶段到生成阶段的分配过程：

```text
输入阶段 (Prompt 长度: 7 个 Token):
- 逻辑块 0 [Token 0~3] -> 映射到 物理块 #12 (状态: 已满 4/4)
- 逻辑块 1 [Token 4~6] -> 映射到 物理块 #3  (状态: 填充 3/4)

自回归步骤 1 (模型生成 Token 7):
- 块表显示逻辑块 1 对应物理块 #3 尚有 1 个空位；
- 将 Token 7 的 KV 向量直接写入物理块 #3 的末尾；
- 物理块 #3 状态变为已满 (4/4)，无需申请新块。

自回归步骤 2 (模型生成 Token 8):
- 逻辑块 1 已无空余槽位，调度器为序列追加逻辑块 2；
- 分配器从全局空闲池中弹出一个空闲物理块 (如物理块 #7)；
- 块表追加映射: 逻辑块 2 -> 物理块 #7；
- 将 Token 8 写入物理块 #7 的第 0 个槽位 (状态: 填充 1/4)。
```

在此机制下，除了每条请求最后一个物理块可能存在未填满的尾部空间（平均浪费 $\frac{\text{Block Size}}{2}$ 个 Token，即**内部碎片**），其余所有物理块都是 100% 打满的。由于任意物理块都能满足分配需求，**外部碎片率被彻底降到了 0%**。

---

## 算子机制：非连续块的 GPU 寻址与计算

传统自注意力算子要求 Key/Value 在显存中是步长连续（Stride-continuous）排列的张量。在分页存储下，同一序列的 KV 块散落在物理显存各处。如果在 Attention 计算前先用 `torch.cat` 把离散块拼成连续张量，额外的数据搬运不仅会抵消显存收益，还会拖慢推理速度。

PagedAttention 的核心在于编写了专用的 GPU CUDA Kernel，在计算点积注意力的同时**在片上完成非连续地址换算**。

![PagedAttention GPU Kernel 寻址与 Attention 计算流程](/images/content-rebuild/16a4625896e5f67b/fig-3.png)

### Decode 阶段算子执行步骤

在自回归解码单步中，输入是当前时刻单个 Query 向量 $q_i$：

1. **块表传入与地址解析**：GPU 线程块（Thread Block）获取当前请求在 Block Table 中的物理块 ID 列表 $[b_0, b_1, \dots, b_m]$；
2. **物理基地址计算**：对于逻辑序列中的第 $t$ 个 Token，其在物理显存中的实际存储地址由 Kernel 内部实时换算：
   $$\text{Logical Block Index} = \lfloor t / \text{Block Size} \rfloor$$
   $$\text{Offset In Block} = t \pmod{\text{Block Size}}$$
   $$\text{Physical Addr} = \text{BaseAddr} + b_{\text{index}} \times \text{BlockStride} + \text{Offset} \times \text{TokenStride}$$
3. **按块载入 Shared Memory**：线程块直接从不连续的全局显存物理块中，将对应的 Key Block 批量加载到片上高速共享内存（Shared Memory）中；
4. **局部点积与在线归一化（Online Softmax）**：计算 Query 与当前块中所有 Key 的局部点积得分。由于整条序列被切分为多个离散块，Kernel 借鉴 FlashAttention 的思想，跨块维护当前的局部最大值 $m$ 和指数和累加器 $l$。每遍历完一个物理块，动态修正缩放系数，避免将全序列注意力矩阵完整写回全局显存；
5. **Value 加权累加输出**：将当前物理块对应的 Value 向量载入，按照局部 Softmax 权重加权求和，遍历所有物理块后输出最终的隐藏状态张量。

通过将非连续指针寻址嵌入到算子内层循环，PagedAttention 做到了计算与内存访问的就地融合，完全避免了中间张量的物理拼接。

---

## 高级优化：写时复制（CoW）与前缀缓存

将显存切解耦为逻辑映射后，复杂的序列分支与长前缀复用便能以指针级开销实现。

### 1. 写时复制（Copy-on-Write, CoW）

在并行采样（Parallel Sampling，单 Prompt 生成 $N$ 个答案）或束搜索（Beam Search）中，各分支在初始阶段的前缀完全一致。

![基于引用计数的写时复制机制在多分支生成中的流转](/images/content-rebuild/16a4625896e5f67b/fig-4.png)

* **前缀共享阶段**：系统为所有子分支创建独立的 Block Table，但表项全部指向相同的物理块 ID，物理块的**引用计数（Reference Count）**设为分支数；
* **分歧写入触发**：当分支 A 率先生成新 Token 时，若当前写入的末尾物理块引用计数大于 1，则触发 CoW：显存分配器单独为分支 A 申请一个全新的物理块，将旧块数据拷贝过去并将原物理块的引用计数减 1。后续各分支的私有生成只追加写入各自独占的物理块中。

### 2. 自动前缀缓存（Automatic Prefix Caching, APC）

在多轮对话或固定 System Prompt 的场景下，不同请求的前缀往往完全重合。

![自动前缀缓存树结构与跨请求物理块复用](/images/content-rebuild/16a4625896e5f67b/fig-5.png)

vLLM 内部维护了一棵基于 Token 内容哈希的前缀基数树（Radix Tree）：
* 历史请求计算完毕后，其持有的物理块不会立即销毁，而是缓存在前缀树中；
* 新请求到达后，调度器优先在树中进行前缀哈希匹配；
* 若命中前缀（例如长达 1000 Token 的系统说明），新请求直接将 Block Table 前段映射到已有的物理块；
* 这不仅省去了 1000 个 Token 的显存开销，更**直接跳过了这部分 Token 的 Prefill 计算**，大幅降低了首 Token 返回延迟（Time to First Token, TTFT）。

---

## 面试追问与工程边界

### 追问 1：Block Size 选型时，为什么通常默认 16 或 32，而不是 4 或 128？

![Block Size 选型权衡曲线：显存内部碎片与计算吞吐的折中](/images/content-rebuild/16a4625896e5f67b/fig-6.png)

这道题考察对“显存利用率”与“GPU 硬件访存合并（Memory Coalescing）”之间的权衡：

| 评估维度 | Block Size 过小（如 4） | Block Size 适中（16 或 32） | Block Size 过大（如 128） |
| :--- | :--- | :--- | :--- |
| **显存内部碎片** | **极低**（请求尾部平均浪费 2 个 Token 空间） | **较低**（平均浪费 8~16 个 Token 空间） | **偏高**（短文本请求末尾空闲空间严重） |
| **访存事务对齐** | **较差**。每次读取的数据块过小，难以占满 128-byte 显存事务 | **极优**。单个块大小自然对齐 GPU 访存粒度，带宽利用率高 | **极优**。连续访存效率高 |
| **算子调度开销** | **极高**。遍历整条序列需要循环查找过多 Block Table 表项 | **低**。表项数量适度，寄存器压力平衡 | **极低**。块跳转开销很小 |
| **CPU 调度负担** | 块表项膨胀数倍，调度器分配/回收物理块的 CPU 开销大 | 适中 | 极轻量 |

**结论**：在主流 GPU（如 NVIDIA Ampere、Hopper）上，`Block Size = 16` 或 `32` 恰好处于显存碎片与访存性能的最佳折中点。

### 追问 2：当系统并发打满、物理显存池彻底耗尽时，vLLM 会怎么做？

当大量请求在 Decode 阶段并发扩容且没有空闲物理块时，调度器提供两种防御策略：
1. **换出（Swapping）**：将低优先级请求已持有的物理块经由 PCIe 批量复制到宿主机内存（CPU RAM）中，腾出 GPU 物理块给正在执行的请求。待有显存释放时，再把数据从 CPU RAM 换回（Swap In）；
2. **抢占重计算（Preemption by Recomputation）**：直接释放被抢占请求的全部物理块（将其放回空闲池），并将该请求重置并塞回等待队列。当后续再次调度它时，系统通过一次 Prefill 重新跑一遍历史 Token 的上下文。在高并发但 PCIe 带宽成为瓶颈的场景下，由于现代 GPU 的 Prefill 算力极强，重计算策略的端到端吞吐表现往往优于 Swapping。

---

## 常见认知误区

* **误区 1：PagedAttention 降低了生成单个 Token 所需的数学计算量**
  **纠偏**：自回归解码中计算 $Q K^T$ 和注意力加权输出的浮点计算次数（FLOPs）完全由模型结构决定，PagedAttention 没有减少单步解码的任何一次浮点运算。其高吞吐的本质是通过**消除显存碎片提升了有效 Batch Size**，将受限于显存带宽的低并发状态转变成了高并发饱和服务。
* **误区 2：Block Table 是由 GPU 硬件或 CUDA 运行时自动管理的**
  **纠偏**：Block Table 的分配、引用计数、释放、前缀匹配均由运行在 CPU 上的调度器（Scheduler / BlockManager）以软件形式维护。GPU 算子仅在执行 Attention 时读取 CPU 传过来的块索引表执行地址寻址。
* **误区 3：写时复制（CoW）发生时会拷贝整条序列的 KV Cache**
  **纠偏**：CoW 的粒度是单个 **Block**。当分支生成新 Token 时，所有已经满载的历史前缀物理块始终保持只读共享状态，仅对发生写入冲突的末尾单个物理块分配新物理块并复制数据，拆分出私有副本。

---

## 总结与延伸阅读

掌握 PagedAttention 的核心在于理解从“连续张量分配”到“分块间接寻址”的底层转变，以及由此带来的显存利用率与并发吞吐的提升。

建议继续沿着模型推理优化路径深入：
* 理解请求级调度如何与分页内存配合：[连续批处理 Continuous Batching 是什么？](/interview/model-deployment-interview/continuous-batching.md)
* 掌握多卡分布式部署时的切分策略：[Tensor Parallel、Pipeline Parallel 和 Data Parallel 有什么区别？](/interview/model-deployment-interview/parallel-strategies.md)
* 学习严谨的显存占用测算逻辑：[如何估算一个大模型部署需要多少显存？](/interview/model-deployment-interview/memory-estimation.md)
