---
title: 连续批处理是什么？如何提升大模型推理吞吐？
icon: mdi:layers-outline
tag:
- 模型部署面试
- Continuous Batching
- 并发
description: 解析连续批处理的工作原理、Iteration级调度流程、Chunked Prefill协同机制与显存枯竭时的驱逐策略。
---

![连续批处理调度机制封面图](/images/content-rebuild/a9be260f67ecb1c4/cover.png)

在传统深度学习模型（如图像分类、目标检测）的推理服务中，批处理通常以**请求（Request）**为基本调度单位。一个批次中的所有输入张量完成前向推理后，系统统一返回结果并接收下一批请求。但在基于自回归（Autoregressive）机制的大语言模型（LLM）推理中，这种静态批处理（Static Batching）方式会导致严重的计算资源浪费。

大模型推理包含两个阶段：一次性处理输入 Prompt 的 **Prefill 阶段**，以及逐个生成后续 Token 的 **Decode 阶段**。在自回归生成过程中，不同请求的 Prompt 长度和生成长度具有极高的不确定性。**连续批处理（Continuous Batching）**，在学术论文（如 Orca）中也被称为 In-flight Batching，将调度粒度从**请求级（Request-level）**下沉到**迭代级（Iteration-level / Step-level）**，解决了静态批处理的资源空转问题。

---

## 面试口答：连续批处理与静态批处理有什么区别？

在面试中被问及“什么是连续批处理”时，建议分两层结构作答：

### 30 秒紧凑要点
> 静态批处理以“请求”为调度边界，整个批次必须等待最长序列生成完毕才能统一释放资源，导致先生成完的短序列产生计算空闲和显存闲置。
> 连续批处理将调度粒度细化至自回归生成的单步迭代（Step/Iteration）。每个 Forward Step 结束后，调度器直接回收已结束序列的显存，并立即插入新请求的 Prefill 或等待中请求的 Decode，让 GPU 算力与显存保持持续运转。

### 2 分钟要点展开
> 1. **执行机制**：大模型 Decode 阶段是逐个 Token 循环生成的。静态批处理受木桶效应限制，早停请求必须填充 Padding Token 参与空轮询，长期占用 KV Cache 显存。
> 2. **动态装配**：连续批处理在每一个 Forward Step 的边界进行调度。遇到输出结束符（`<eos>`）或达到最大长度的请求会立即退出，等待队列中的新请求无需等待整批完成，可以直接补位进入下一个 Step。
> 3. **协同演进**：连续批处理解决了序列维度的动态调度问题，与 PagedAttention 提供的离散物理显存管理相结合，使真实并发场景下的整体吞吐量获得数倍提升。为了避免长 Prompt 在 Prefill 阶段长时间占用 GPU 导致存量请求的 Decode 停顿，现代推理框架还会引入 Chunked Prefill，将长输入切块后混入多个 Iteration 协同调度。

---

## 一个贯穿例子：静态批处理的瓶颈在哪里？

为了直观理解连续批处理解决的问题，我们设定一个包含 4 个并发请求的批次：

* 请求 A：生成 16 个 Token 即遇到 `<eos>` 终止；
* 请求 B：生成 32 个 Token 终止；
* 请求 C：生成 64 个 Token 终止；
* 请求 D：长尾请求，需生成 512 个 Token。

![静态批处理算力泡与显存闲置示意图](/images/content-rebuild/a9be260f67ecb1c4/fig-1.png)

在静态批处理下，引擎一次性载入这 4 个请求。在第 16 个 Step 完成后，请求 A 已经产出了全部内容。但由于整批张量形状必须在序列维度对齐，底层计算引擎无法单独结束请求 A，这会带来两方面的开销：

### 1. 计算维度的无效开销（算力泡）
为了维持 Batch 的张量维度，引擎必须在请求 A 后面填充 Pad Token 参与后续 496 次前向计算。虽然可以通过 Attention Mask 屏蔽 Padding 区域以保证输出数值正确，但线性投影层（QKV 投影与 MLP 层）依然在对无效的 Token 执行矩阵乘运算。

更关键的是，Decode 阶段是典型的**显存带宽受限（Memory-bandwidth Bound）**操作，每次迭代都需要从 GPU 显存读取模型完整权重。无效的计算和访存直接占用了 GPU 本可服务其他请求的有效时间。

### 2. 显存维度的锁定闲置
在缺乏动态显存管理的体系中，请求 A、B、C 即使已经生成结束，它们在前文计算中积累的 [KV Cache](/interview/model-deployment-interview/kv-cache-deploy.md) 依然无法释放，必须保留在显存中，直到请求 D 在第 512 步完成。等待队列中的新请求只能被阻塞在显存门外，限制了系统的有效并发能力。

<ReadMoreLock />

---

## Iteration 级调度：动态装配与步进控制

连续批处理打破了“整批同时开始、整批同时结束”的限制，将调度决策点放在每一个模型推理的 Iteration 边界。

![Iteration级动态调度过程示意图](/images/content-rebuild/a9be260f67ecb1c4/fig-2.png)

### 单步调度的执行流程
连续批处理引擎内部通常运行一个高频迭代的调度循环：

$$\text{Loop: } \text{Schedule} \longrightarrow \text{Prepare Batch} \longrightarrow \text{Model Forward (1 Step)} \longrightarrow \text{Process Logits} \longrightarrow \text{Check Completion}$$

在单次 Iteration 内，调度器依次执行以下逻辑：
1. **状态检查与显存回收**：上一轮 Forward 结束后，采样器产出本轮各请求的新 Token。调度器检查所有序列状态：
   * 若序列遇到结束符（如 `<|endoftext|>`）或达到设定的 `max_tokens` 阈值，该请求标记为已完成（Finished）。
   * 调度器通知显存管理器（如 [PagedAttention](/interview/model-deployment-interview/vllm-paged-attention.md)）释放该请求占用的物理块，并向客户端推流返回响应。
2. **候选挑选与批次组装**：调度器从等待队列中拉取新请求，或者恢复之前被挂起的请求。
3. **动态 Batch 构建**：调度器将正在运行中（Running）的未完成请求与新准入的请求合并，打包成当前 Iteration 专用的非规则形状批次输入，送入下一轮 Forward 计算。

通过将调度粒度切分到单个 Step，已完成请求在 1 个 Step 内即可交出显存，新请求在一个 Step 的间隙即可插入执行，GPU 的计算单元与显存带宽得以保持更高利用率。

---

## 解决长 Prompt 阻塞：Chunked Prefill 协同机制

将 Prefill 请求与 Decode 请求放入同一个调度系统后，会暴露出另一个特征差异：**算力密集型与访存密集型的冲突**。

![Prefill与Decode计算特征对比示意图](/images/content-rebuild/a9be260f67ecb1c4/fig-3.png)

* **Prefill 阶段**：输入通常是一整段 Prompt（长序列 $S$），自注意力计算与 MLP 包含大尺寸矩阵乘法（GEMM）。在常见中长 Prompt 和常规并发下，算术强度较高，通常表现为**计算受限型（Compute-Bound）**操作（但当 Prompt 极短或并发极低时仍可能受显存带宽制约）。
* **Decode 阶段**：输入是每个序列最新产出的单 Token（长度为 1），计算主要是较细粒度的矩阵乘法或矩阵-向量乘法（GEMV）。在常见并发规模下，每次迭代都需读取全量模型权重与历史 KV Cache，算术强度较低，通常表现为**显存带宽受限型（Memory-Bound）**操作（只有在并发 Batch 极大时才可能趋向计算受限）。

在未优化的连续批处理中，如果系统正在并发执行数十个请求的 Decode（单步耗时通常为几十毫秒），此时突然到来一个包含数千 Token 的长 Prompt 请求并立即全量执行 Prefill，这个 Step 的执行时间可能会骤增到数百毫秒。

这种耗时突增会导致正在平稳输出字符的存量 Decode 请求出现明显的停顿，使字间延迟（Inter-Token Latency, ITL）发生抖动，影响终端流式输出的流畅度。

![Chunked Prefill拆分与混合批处理示意图](/images/content-rebuild/a9be260f67ecb1c4/fig-4.png)

为了平滑这种抖动，现代推理框架引入了 **Chunked Prefill（分块预填充）** 机制：

1. **设置 Token 预算（Token Budgeting）**：调度器为每个 Iteration 设定一个全局 Token 处理上限（如 `max_num_batched_tokens = 2048`）。
2. **拆解长 Prompt**：当一个长请求到达时，调度器不一次性执行完所有 Prefill，而是将其拆解为多个分块（Chunk）。
3. **混合调度（Co-scheduling）**：调度器在单次 Iteration 内优先保障存量 Decode 请求的 Token 预算，再将剩余预算分配给长 Prompt 的分块进行计算。

```python
# 调度器在单次 Iteration 中混合组装 Chunked Prefill 与 Decode 的逻辑示意
def schedule_iteration(waiting_queue, running_queue, max_budget=2048):
    current_tokens = 0
    batch_to_run = []

    # 1. 优先保证正在生成的 Decode 请求，降低 Inter-Token Latency (ITL) 抖动
    for req in running_queue:
        if current_tokens + 1 <= max_budget:
            batch_to_run.append((req, "DECODE", 1))
            current_tokens += 1
        else:
            break  # 超出单步预算保护

    # 2. 将剩余预算分摊给 Prefill 请求的分块（Chunk）
    # 注意：遍历时复制列表以防止原地 remove 导致迭代跳项（示意实现）
    for req in list(waiting_queue):
        remaining_prompt = req.prompt_len - req.prefilled_len
        if remaining_prompt > 0 and current_tokens < max_budget:
            available_chunk = min(remaining_prompt, max_budget - current_tokens)
            batch_to_run.append((req, "PREFILL_CHUNK", available_chunk))
            current_tokens += available_chunk
            req.prefilled_len += available_chunk
            if req.prefilled_len == req.prompt_len:
                # 该请求的 Prefill 全部完成，在下个 Step 转入 Running
                waiting_queue.remove(req)
                running_queue.append(req)

    return batch_to_run
```

通过分块预填充，计算密集的大矩阵乘法被均摊到多个连续 Step 中，利用 Decode 的显存等待间隙填充算力，同时降低了单步最大执行时间。

---

## 调度器状态机与显存管理

连续批处理的稳定性依赖于状态机对物理显存边界的严格控制。以常见实现为例，系统核心维护三个请求队列：

![调度器状态机与队列流转图](/images/content-rebuild/a9be260f67ecb1c4/fig-5.png)

### 三大核心队列
1. **Waiting 队列**：已到达服务端但尚未开始 Prefill 或尚未完成全部 Prefill 分块的请求。
2. **Running 队列**：已经完成全量 Prefill，正在逐 Step 推进 Decode 的请求集合。这些请求在每个 Step 都会向显存池申请新块以存放生成的 KV 向量。
3. **Swapped 队列**：因显存容量不足被调度器暂停执行，其已生成的 KV Cache 被临时转移至主机内存（Host CPU RAM）的请求集合。

### 状态转换条件
* **Waiting $\rightarrow$ Running**：当显存中存在足够的空闲物理块支撑新请求的 Prefill（或下一个分块）时，请求分配物理显存并推进计算，Prefill 全部完成后转为 Running。
* **Running $\rightarrow$ Finished**：请求生成结束符或达到最大输出长度，调度器注销请求并回收其占用的所有物理显存块。
* **Running $\rightarrow$ Swapped / Preempted**：突发高并发导致显存池耗尽，且存量 Running 请求在下一步必须写入新 KV Cache。为了避免程序因 CUDA Out Of Memory 崩溃，调度器必须主动挑选请求进行**抢占（Preemption）**。
* **Swapped $\rightarrow$ Running**：其他请求完成并释放显存后，处于 Swapped 队列的请求被重新加载回显存并继续解码。

---

## 面试追问：显存完全耗尽时如何驱逐？

**追问**：“如果并发量极高，所有处于 Running 状态的请求在下一步都需要申请新显存块，而此时空闲显存块已经为 0，调度器必须暂停谁？常见的恢复策略 Recompute 与 Swap 各自有什么代价？”

![显存枯竭时的应急驱逐分支对比图](/images/content-rebuild/a9be260f67ecb1c4/fig-6.png)

### 1. 驱逐目标的选择策略
调度器通常遵循**后到先抢占（Last-Come, First-Preempted）**原则，或者优先挑选当前已生成 Token 最少的请求。优先保留已生成较多 Token、接近完结的长请求，可以尽快让其执行完毕并释放大量 KV Cache；而抢占刚开始 Decode 的请求，系统损失的有效算力相对较小。

### 2. 两种抢占恢复策略对比

| 恢复机制 | 执行过程 | 主要资源开销 | 适用场景分析 |
| :--- | :--- | :--- | :--- |
| **重计算（Recompute）** | 直接**物理丢弃**该请求当前的全部 KV Cache，将其退回 `Waiting` 队列。后续显存充裕时，将历史生成的 Token 连同原 Prompt 重新执行一次 Prefill。 | **GPU 算力浪费**<br>需要重新执行一遍矩阵乘运算。 | **适用场景**：Prompt 较短、GPU 算力充足但主机内存或 PCIe 带宽受限的环境。<br>**特点**：实现轻量，不占用主机内存与 PCIe 通信带宽。 |
| **内存换出（Swap to CPU）** | 暂停解码，将该请求在 GPU 显存中的全部 KV Cache 通过 PCIe 链路**拷贝到 Host 内存（CPU RAM）**中，随后释放显存块。 | **PCIe 带宽与 CPU 内存占用**<br>受主机可用内存大小及 PCIe 传输速度制约。 | **适用场景**：超长上下文序列（如 32k 以上）。重新 Prefill 的算力开销极大，此时通过 PCIe 传输换取免算力重跑更合理。 |

在实际生产部署中，两种机制可以按需配置：短序列优先使用 Recompute 降低总线负载；长上下文业务则配置一定规模的 CPU Swap 空间作为显存突发的缓冲缓冲垫。

---

## 常见误区与工程边界

在面试与实际技术选型中，需要厘清以下三个工程认知边界：

* **误区一：连续批处理修改了 Transformer 的模型结构**  
  连续批处理**不需要改动模型权重或神经网络层结构**。它属于推理引擎在服务调度层与显存管理层的逻辑控制。模型底层执行的依然是标准的前向传播和注意力计算。
* **误区二：使用了连续批处理就绝对不会 OOM**  
  连续批处理只解决了单步粒度的调度问题。如果没有配合高效的动态显存管理（如 PagedAttention），或者服务入口没有设置合理的准入控制（如最大并发请求数与最大序列长度限制），极端高并发下物理显存依然会被打满，甚至出现频繁抢占导致的吞吐滑坡。
* **误区三：单步内混合 Prefill 和 Decode 没有额外代价**  
  混合计算要求底层算子支持不规则 Shape 的并行切分（如 Ragged Batch Attention 算子）。如果底层 Kernel 调度不够优化，张量重排（Tensor Re-layout）和不规则访存可能会抵消一部分并发收益。因此，分块大小（如 `max_num_batched_tokens` 设置为 512、1024 还是 2048）需要结合硬件显存带宽与算力实测来确定平衡点。
