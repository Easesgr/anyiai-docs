---
title: 如何估算大模型部署需要多少显存？怎么计算静态与动态开销？
icon: mdi:calculator-variant-outline
tag:
- 模型部署面试
- 显存估算
- 资源规划
description: 拆解大模型推理显存构成的四部分，给出静态权重、GQA下的KV Cache、激活值与运行时常驻的精确计算公式与八卡实战算例。
---

在模型部署面试中，显存估算不仅考察对 Transformer 结构的掌握程度，还考察硬件选型、并发规划与故障排查能力。很多开发者习惯用“参数量乘以 2”做粗略估算，但在实际生产环境中，这种估算往往会导致显卡采购不足，或者在并发流量到来时频繁遭遇 CUDA Out of Memory (OOM)。

![显存容量精算全景概览](/images/content-rebuild/312865b9295a0bbc/cover.png)

显存规划不仅要放得下模型权重，还要为动态请求、算子临时计算以及框架运行时预留空间。理清每一部分的计算方法，才能在技术面试与架构选型中给出经得起验证的方案。

---

## 口述答案：生产环境显存占用的四个部分

如果面试官要求“简明扼要地给出大模型部署显存评估的核算框架”，可以按以下结构口述：

> 部署大模型时的单卡显存占用主要由四部分构成：
>
> 1. **模型静态权重（Weight）**：由参数量和存储精度决定，多卡部署时由张量并行（TP）或流水线并行（PP）切分。
> 2. **动态键值缓存（KV Cache）**：随并发请求数和上下文长度线性增长。单 Token 的开销由模型层数、KV 投影头数（MHA/GQA 架构）以及存储精度决定。
> 3. **瞬态激活值（Activation）**：模型在 Prefill（首字计算）和 Decode（逐字解码）阶段前向传播生成的临时张量，通常通过分块 Prefill（Chunked Prefill）控制峰值大小。
> 4. **框架运行时常驻（Runtime）**：包括 CUDA 上下文、NCCL 通信缓冲区、CUDA Graph 预录制内存池以及内存分配器的预留开销。
>
> 综合估算公式为：
> $$M_{total} = M_{weight} + M_{kv\_cache} + M_{activation} + M_{runtime}$$
> 在规划并发时，先按框架利用率阈值（如预留 10% 防溢出缓冲，可用 72 GiB）扣除静态权重、运行时常驻与激活值峰值，剩余的显存空间才全部分配给 KV Cache 作为并发吞吐池。以 8 卡 H100 部署 70B 模型为例，单卡扣除约 23 GiB 刚性开销后，剩余约 49 GiB 显存全部供 KV Cache 使用，集群整体可承载约 128 万 Token 的并发容量。

![显存四部曲构成解剖图](/images/content-rebuild/312865b9295a0bbc/fig-1.png)

---

## 模型静态显存计算：参数量、精度与多卡切分

静态模型权重占用的显存计算非常直接，关键在于理清数据类型的物理字节数和并行策略。

### 1. 常见数据精度对应的字节数

- **FP32（单精度浮点）**：每个参数占用 $4$ 字节。
- **FP16 / BF16（半精度浮点）**：每个参数占用 $2$ 字节。
- **FP8 / INT8（8位浮点/定点）**：每个参数占用 $1$ 字节。
- **INT4（4位量化，如 GPTQ/AWQ）**：每个参数占用 $0.5$ 字节（加上用于反量化的 scale 和 zero-point 缩放因子，实际显存一般会多出 3% 到 5%）。

量化技术的实现与精细损益可参考 [模型量化是什么？](/interview/model-deployment-interview/quantization.md)。

<ReadMoreLock />

### 2. 张量并行（TP）下的单卡静态显存

在采用张量并行（Tensor Parallelism，详见 [Tensor Parallel、Pipeline Parallel 和 Data Parallel 有什么区别？](/interview/model-deployment-interview/parallel-strategies.md)）时，Transformer 的注意力权重矩阵（$W_q, W_k, W_v, W_o$）和多层感知机矩阵（$W_{gate}, W_{up}, W_{down}$）会被均匀切分到各个 GPU 上。

单卡静态权重显存公式如下：
$$M_{weight} = \frac{N \times P_{bytes}}{1024^3 \times TP} \quad (\text{单位：GiB})$$

其中：
- $N$ 为模型实际参数量（Parameter Count）；
- $P_{bytes}$ 为单参数占用的字节数；
- $TP$ 为张量并行度（Tensor Parallel Size）。

以参数量 $N = 70.6 \times 10^9$ 的 Llama-3-70B 为例，采用 BF16 精度（$P_{bytes} = 2$ 字节）：
- 全局权重大小：$70.6 \times 10^9 \times 2 = 141.2 \times 10^9$ 字节 $\approx 131.50 \text{ GiB}$；
- 使用 $TP = 8$ 部署到 8 张显卡上，单卡承担权重：$131.50 \div 8 \approx 16.44 \text{ GiB}$。

需要注意，词表嵌入层（Embedding）与输出分类头（LM Head）在部分框架中可能采用完整复制或独立切分的方式，因此实际占用通常比理论纯切分多出 100 MiB 到 500 MiB 的对齐与通信缓冲。

![模型静态显存与量化精度换算关系图](/images/content-rebuild/312865b9295a0bbc/fig-2.png)

---

## 动态显存计算：KV Cache 理论与实际占用

自回归解码（逐字生成）的核心显存开销来自 KV Cache（原理详见 [KV Cache 在模型部署中有什么作用？](/interview/model-deployment-interview/kv-cache-deploy.md)）。如果不做细致建模，很容易在并发评估时出现数倍偏差。

### 1. 单个 Token 的 KV Cache 显存开销

Transformer 中，每个 Token 在每个网络层都会计算出一组键向量（Key）和值向量（Value）。假设模型参数如下：
- 模型层数：$L$
- 单个注意力头的隐藏维度：$d_{head}$
- KV 投影头数：$n_{kv\_heads}$
- KV 缓存精度字节数：$P_{kv}$（BF16/FP16 为 2 字节，FP8 为 1 字节）

单个 Token 在所有层生成的 Key 和 Value 向量包含的浮点数总量为：
$$\text{Elements per token} = 2 \times L \times n_{kv\_heads} \times d_{head}$$

对应的物理内存大小（字节）：
$$S_{token} = 2 \times L \times (n_{kv\_heads} \times d_{head}) \times P_{kv}$$

> **核心细节：多头注意力（MHA）与分组查询注意力（GQA）的区别**
> 早期模型（如 LLaMA-1 65B）采用多头注意力（MHA），KV 头数与 Query 头数相同（$n_{kv\_heads} = n_{q\_heads}$），此时 $n_{kv\_heads} \times d_{head} = d_{model}$（模型隐藏维度，例如 8192）。
> 现代主流模型（如 Llama-3-70B、Qwen-72B）普遍采用分组查询注意力（GQA）。在 Llama-3-70B 中，$n_{q\_heads} = 64$，$n_{kv\_heads} = 8$，$d_{head} = 128$。
> 此时每层的 KV 维度只有 $8 \times 128 = 1024$，是模型隐藏层维度 8192 的八分之一。如果直接套用隐藏层维度计算，算出来的 KV Cache 显存会虚增 8 倍。

以 Llama-3-70B（$L = 80, n_{kv\_heads} = 8, d_{head} = 128, P_{kv} = 2$）为例：
$$S_{token} = 2 \times 80 \times (8 \times 128) \times 2 = 327,680 \text{ 字节} = 320 \text{ KiB} = 0.3125 \text{ MiB}$$

### 2. 多卡并发下的 KV Cache 总量

采用 $TP$ 张量并行时，KV 头会均分到各个 GPU 上。单张 GPU 上维护单个 Token 的 KV Cache 显存为：
$$S_{token\_per\_gpu} = \frac{S_{token}}{TP}$$

在 $TP = 8$ 时，单卡单 Token 的 KV Cache 开销为：
$$320 \text{ KiB} \div 8 = 40 \text{ KiB} = 0.0390625 \text{ MiB}$$

如果系统当前并发请求数为 $B$，所有请求当前上下文及生成 Token 数量总和为 $T_{total}$，在 $S_{token\_per\_gpu}$ 以字节（Bytes）为单位（本例为 40,960 字节）时，单卡实际承载的 KV Cache 显存为：
$$M_{kv\_cache} = \frac{T_{total} \times S_{token\_per\_gpu}}{1024^2} \quad (\text{单位：MiB})$$
若已换算为以 MiB 为单位的开销（即 $0.0390625 \text{ MiB/Token}$），则单卡显存直接表示为 $M_{kv\_cache} = T_{total} \times S_{token\_per\_gpu\_MiB}$。

使用 PagedAttention 机制（详见 [vLLM 是什么？PagedAttention 为什么能提升推理吞吐？](/interview/model-deployment-interview/vllm-paged-attention.md)）后，框架按需动态分页申请内存，实际显存占用严格贴合所有活跃请求的真实 Token 数量，消除了由于预分配最大上下文造成的显存浪费。

![GQA 机制与 KV Cache 理论推导图](/images/content-rebuild/312865b9295a0bbc/fig-3.png)

---

## 运行时与临时缓冲显存：CUDA Context、NCCL 与 Activation

除权重和 KV Cache 外，运行时常驻显存和前向计算的临时激活值属于固定开销。在做容量定标时，必须先扣除这部分显存。

![运行时临时缓冲与系统常驻分析图](/images/content-rebuild/312865b9295a0bbc/fig-4.png)

### 1. 运行时常驻显存（Runtime Footprint）

服务启动后、接收第一个请求前，就会常驻一部分显存：
- **CUDA Context**：CUDA 驱动与运行时环境初始化的开销，在 A100/H100 显卡上，每个进程大约占用 $0.5 \text{ GiB} \sim 1.2 \text{ GiB}$。
- **NCCL 通信缓冲区**：在多卡分布式推理环境（$TP > 1$）中，NCCL 用于卡间 All-Reduce 等算子通信的环形缓冲区（Ring Buffers），单卡通常占用 $1.0 \text{ GiB} \sim 2.0 \text{ GiB}$。
- **CUDA Graph 捕获空间**：为消除 CPU 侧发射算子的调度开销，推理引擎会在服务启动预热时，针对多种常用的 Decode Batch 大小（如 1, 2, 4, 8, 16...）预先捕获并编译执行图。每个捕获的图都会保留算子依赖的静态工作区（Workspace），合计占用通常在 $1.5 \text{ GiB} \sim 3.0 \text{ GiB}$。

单卡常驻开销基线一般估算为：
$$M_{runtime} \approx 3.0 \text{ GiB} \sim 5.0 \text{ GiB}$$

### 2. 瞬态激活值（Activation Memory）

激活值是指算子在前向传播中生成的中间张量（例如 LayerNorm 输出、注意力得分中间矩阵等）：
- **Decode 阶段**：每个序列每次只前向计算 1 个 Token，激活值极小（几十兆字节量级），可忽略不计。
- **Prefill 阶段**：当一个包含数千 Token 的 Prompt 进入系统时，模型必须对整段序列同时做前向运算。如果并发进入多个长输入，瞬态激活值可能瞬间达到数个 GiB。

生产级推理框架通常开启 **分块 Prefill（Chunked Prefill）**（详见 [连续批处理 Continuous Batching 是什么？](/interview/model-deployment-interview/continuous-batching.md)），限制每次前向计算的最大上下文 Token 数量（例如单批次最多计算 512 或 2048 个 Token），将瞬态激活值峰值稳定控制在指定阈值（例如 $M_{activation} \le 2.5 \text{ GiB}$）内。

---

## 完整算例：Llama-3-70B 在 8*H100 上的并发容量推导

以常见的生产环境为例：使用 **8 张 NVIDIA H100 80GB SXM5**，部署 **Llama-3-70B（BF16 精度）**。

![Llama-3-70B 八卡实战容量推导时序与吞吐图](/images/content-rebuild/312865b9295a0bbc/fig-5.png)

### 1. 单卡可用显存基线

在工程测算中，通常将 NVIDIA H100 80GB SXM5 的标称显存规格按 $80 \text{ GiB} = 81,920 \text{ MiB}$ 进行基准折算（在物理机中 `nvidia-smi` 驱动实际识别容量约为 81,559 MiB，教学估算取 80 GiB 简化折算，两者误差小于 0.5%）。

在推理框架（如 vLLM）中，通常配置 `gpu_memory_utilization = 0.90`，保留 10% 显存作为全局防溢出缓冲：
$$M_{available} = 80 \times 0.90 = 72.0 \text{ GiB}$$

### 2. 扣除刚性开销

- **单卡模型权重**：
  $$M_{weight} = \frac{70.6 \times 10^9 \times 2}{1024^3 \times 8} \approx 16.44 \text{ GiB}$$
- **运行时与通信缓冲（CUDA + NCCL + CUDA Graph）**：
  $$M_{runtime} \approx 4.0 \text{ GiB}$$
- **Prefill 激活值保留区（分块 Prefill 约束）**：
  $$M_{activation} \approx 2.5 \text{ GiB}$$

单卡必须扣除的刚性开销合计：
$$M_{overhead} = 16.44 + 4.0 + 2.5 = 22.94 \text{ GiB}$$

### 3. 可供动态分配的 KV Cache 显存池

单卡剩余可供 KV Cache 自由分配的空间：
$$M_{kv\_pool} = 72.0 - 22.94 = 49.06 \text{ GiB} = 50,237.44 \text{ MiB}$$

全集群（8 张显卡）聚合的 KV Cache 容量：
$$M_{kv\_cluster} = 49.06 \times 8 = 392.48 \text{ GiB} = 401,899.52 \text{ MiB}$$

### 4. 最大 Token 容纳量与并发换算

前面推导出集群保存 1 个 Token 的 KV Cache 需要 $0.3125 \text{ MiB}$：
$$\text{集群最大容纳 Token 总数} = \frac{401,899.52 \text{ MiB}}{0.3125 \text{ MiB/Token}} = 1,286,078 \text{ Tokens}$$

单卡视角交叉验证：单卡可用池 $50,237.44 \text{ MiB}$ 除以单卡单 Token 开销 $0.0390625 \text{ MiB}$：
$$\text{单卡最大容纳 Token 总数} = \frac{50,237.44 \text{ MiB}}{0.0390625 \text{ MiB/Token}} = 1,286,078 \text{ Tokens}$$

两者完全吻合，集群安全并发的上下文总量约为 **128.6 万 Tokens**。

根据具体业务场景折算并发承载能力：
- **短文本对话**（平均单请求上下文 + 生成共计 2,048 Tokens）：
  $$B_{safe} = \frac{1,286,078}{2,048} \approx 628 \text{ 个并发}$$
- **长文档问答 / RAG**（平均单请求 16,384 Tokens）：
  $$B_{safe} = \frac{1,286,078}{16,384} \approx 78 \text{ 个并发}$$
- **超长文本处理**（单请求达到 65,536 Tokens）：
  $$B_{safe} = \frac{1,286,078}{65,536} \approx 19 \text{ 个并发}$$

显存规划不仅是算卡数，更取决于目标业务的上下文长度分布。针对高吞吐场景的具体调优方案，可参考 [如何优化大模型服务的并发、吞吐和响应速度？](/interview/model-deployment-interview/optimization.md)。

---

## 面试追问：为什么理论计算显存充足，上线后依然频繁遭遇 OOM？

面试官常问：“我的核算公式显示还剩 10 GiB 显存，为什么在请求流量上涨时仍然会报 CUDA Out of Memory？”回答此类问题可以从以下四个工程原因展开：

![生产环境显存踩坑与 OOM 故障根因分析图](/images/content-rebuild/312865b9295a0bbc/fig-6.png)

### 1. 显存外碎片（External Fragmentation）
在没有采用物理分页显存管理（如 PagedAttention）的原生推理实现中，显存分配依靠 PyTorch 底层的 `CachingAllocator`。

由于请求长度各异且随机结束，连续物理显存会被切割成大量不连续的小碎片。当系统尝试为新到达的长序列分配一段连续的张量内存时，即使剩余空闲显存总和远大于所需尺寸，但由于找不到一段足够大的连续物理地址空间，CUDA 依然会直接抛出 OOM。

### 2. 未启用分块 Prefill 导致长 Prompt 冲击
静态估算显存时，很多人只按 Decode 阶段的极低激活值来预留。但如果服务未开启 Chunked Prefill，当突然涌入多个几千 Token 的长 Prompt 请求时，自注意力层在 Softmax 运算以及投影阶段生成的激活值会瞬时暴涨数倍，瞬间吃满预留的显存缓冲。

### 3. CUDA Graph 预录制范围越界触发回退
为了降低 GPU 调度开销，推理框架通常会预录制一批固定 Batch 维度的 CUDA Graph（如 1, 2, 4, 8, 16, 32）。如果突发并发请求数超过了最大录制上限（例如达到 64），框架会被迫回退到 PyTorch 的动态执行模式（Eager Mode）。

此时动态调度会即时分配新的临时工作区，叠加显存重复申请，很容易击穿剩余的安全冗余空间。

### 4. 多进程通信缓冲与上下文泄露
在使用 Ray 或 Python 多进程管理多卡实例时，如果主进程代码中不慎调用了 CUDA 相关函数，且没有显式配置 `CUDA_VISIBLE_DEVICES`，主进程可能会在 Rank 0 显卡上额外初始化一个无用的 CUDA Context，隐式占用近 1 GiB 显存。

此外，如果在网络波动发生后进行连接重试，NCCL 连接在重建前未能完全释放之前的通信环形缓冲，也会在显存中形成通信缓冲泄漏。

防范和处理此类生产运行风险的配套方案，可参考 [模型服务如何做限流、熔断、超时和降级？](/interview/model-deployment-interview/rate-limiting.md) 以及 [部署大模型时如何做日志监控、调用统计和成本控制？](/interview/model-deployment-interview/monitoring-cost.md)。
