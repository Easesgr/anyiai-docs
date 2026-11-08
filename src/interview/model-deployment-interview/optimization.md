---
title: 如何优化大模型服务的并发、吞吐和响应速度？
icon: mdi:tune-vertical
tag:
- 模型部署面试
- 性能优化
- 并发
description: 系统解析大模型推理在首字延迟、单字耗时与系统吞吐之间的物理瓶颈，涵盖算子访存优化、CUDA Graphs、引擎调度参数及推测解码调优实践。
---

在生产环境中部署大模型，性能优化不是简单更换框架或无脑堆砌硬件，而是在相互制约的指标之间寻找最佳平衡点。大模型推理具有清晰的双阶段特征：

1. **预填充阶段（Prefill）**：计算输入 Prompt 的所有 Token 并生成初始 KV Cache。在常见服务批量与中长提示词下，该阶段通常表现为算力密集型（Compute-bound），计算耗时直接决定了用户等待的**首字延迟（Time To First Token, TTFT）**。
2. **解码阶段（Decode）**：每次以前序所有 Token 的 KV Cache 为上下文，自回归预测生成下一个 Token。在中小并发单步自回归时，每次计算量较小且必须完整读取模型权重或对应状态，通常表现为显存带宽受限型（Memory-bound），直接决定了生成过程中的**单字耗时（Time Per Output Token, TPOT）**与并发吞吐容量。

面对高并发业务场景，调优的目标是在满足端到端服务质量指标（如 P99 TTFT < 1.5s，TPOT < 40ms）的前提下，最大化系统的有效吞吐量（Goodput，即在超时限制内成功处理的 Tokens/sec）。

假设我们在一组 8 卡 H100 机器上部署 Llama-3-70B 服务，承接企业知识库问答。用户的输入 Prompt 平均为 2000 个 Token，期望生成 200 个 Token。系统目标是支持 100 以上并发提问，且 TTFT 控制在 1 秒以内。本文将以该场景为贯穿算例，拆解各层调优方案。

![大模型服务综合调优全景看板与权衡矩阵](/images/content-rebuild/cae75a2ca9f2b2d7/cover.png)

---

## 口述答案：大模型推理性能调优方法与权衡逻辑

在面试中被问及“如何优化大模型服务的并发、吞吐和响应速度”时，建议按照**“瓶颈归因—内核加速—运行时编译—调度配置—解码算法”**的逻辑展开：

> “优化大模型推理服务的核心是**区分 Prefill（典型负载下常受算力限制）与 Decode（中小批量下常受显存带宽限制）两个不同阶段的特征，在 TTFT、TPOT 和有效吞吐之间做分层调优**：
>
> 1. **算子与访存层**：采用 FlashAttention-2/3 等算子融合技术，通过分块计算与在线 Softmax，消除注意力中间矩阵对显存高带宽内存（HBM）的重复读写，显著压低长文本 Prefill 的计算与访存耗时；同时配合 FP8 权重量化与 KV Cache 量化，成倍减少 Decode 阶段每一步的显存搬运量。
> 2. **运行时与计算图层**：Decode 阶段单步计算时间极短，极易遇到 CPU 发射内核（Kernel Launch）的调度延迟。使用 CUDA Graphs 对固定 Batch 尺寸进行捕获与预编译，把大量算子发射合并为单次 GPU 触发，消除 CPU 调度等待。
> 3. **引擎调度参数层**：以 vLLM 等主流框架为例，协同配置三大核心参数：
>    - 开启分块预填充（Chunked Prefill），结合合理设置的 `max_num_batched_tokens`，把长 Prompt 拆分成小块与正在 Decode 的请求交替混配，避免 Prefill 独占计算卡死其他并发请求；
>    - 调节 `max_num_seqs` 设定单次迭代活跃序列上限，防止 Decode 状态管理与排队积压；
>    - 设定合理的 `gpu_memory_utilization`（如 0.88~0.90），留出足够的动态激活与图缓冲区，在最大化 KV Cache 块数量的同时避免显存溢出（OOM）或抢占换页（Preemption）。
> 4. **生成算法层**：在代码补全、固定格式提取等输出确定性高、采样温度低的场景下引入推测解码（Speculative Decoding），利用小 Draft 模型预先产出候选词，主模型单次前向并行验证，打破逐步自回归的物理串行瓶颈。”

---

## 底座内核加速：FlashAttention 与 FlashInfer 访存优化

标准 Self-Attention 的计算过程为：

$$S = Q K^T, \quad P = \text{softmax}\left(\frac{S}{\sqrt{d_k}}\right), \quad O = P V$$

在没有算子融合的原生实现中，中间注意力矩阵 $S$ 和 $P$ 的形状为 $[B, H, N, N]$（$B$ 为 batch 大小，$H$ 为注意力头数，$N$ 为序列长度）。在长上下文场景下（如上述 2000 Token Prompt），$N \times N$ 矩阵需要被反复写入 GPU 的全局显存（HBM）并重新读取。这一访存读写过程花费的时间往往远超浮点矩阵乘法本身的计算耗时。

![标准 Attention 与 FlashAttention 在 GPU 存储层级中的数据流向对比](/images/content-rebuild/cae75a2ca9f2b2d7/fig-1.png)

### 算子融合与分块计算（Tiling）
FlashAttention 的突破在于**改变了访存路径，而非改变注意力的数学结果**：
1. **在线 Softmax（Online Softmax）**：利用 Softmax 具备按块更新的数学特性，将输入切分成 GPU 片上高速共享内存（SRAM）能容纳的小块，在遍历计算时动态维护局部最大值与归一化分母。这样就无需在显存 HBM 中保存完整尺寸的 $N \times N$ 注意力矩阵，显存 IO 复杂度从 $O(N^2)$ 降低到 $O(N)$。需要明确的是，FlashAttention 并没有减少全注意力算法理论上的乘加运算次数，其浮点算术计算量仍然是 $O(N^2)$。
2. **硬件针对性演进**：FlashAttention-2 优化了注意力头的并行度与 Warp 级别调度，减少了非矩阵乘法算子的开销；FlashAttention-3 则针对 NVIDIA Hopper 架构，利用硬件张量内存加速器（TMA）与异步指令，实现数据从 HBM 到片上 SRAM 的异步搬运与 Tensor Core 计算完全重叠。
3. **FlashInfer 面向 Decode 的优化**：在 Decode 阶段，$Q$ 的序列长度为 1，而 $K, V$ 为历史上下文总长。由于连续批处理中每个请求的历史 KV Cache 存储在不连续的物理内存块中（如 PagedAttention 格式），FlashInfer 针对非连续页表进行了高效的访存合并（Coalescing）与混合头合并，大幅减少小 Batch 解码时的显存带宽浪费。

---

<ReadMoreLock />

## 计算图与运行时：CUDA Graphs 消除 CPU 发射开销

在大模型自回归生成的 Decode 阶段，单步生成的 Batch 较小（例如并发序列数仅有 4 或 8）时，单个 Transformer 层算子（如 LayerNorm、GEMV 向量乘、RoPE 位置编码）在 GPU 上的实际计算执行时间往往只有几微秒到十余微秒。

此时系统性能瓶颈往往不是 GPU 显存带宽，而是 **CPU 端的内核发射开销（CPU Kernel Launch Overhead）**。CPython 解释执行以及运行时向 CUDA 驱动逐个提交 Kernel 的调用延迟，往往达到数微秒。当 CPU 发射速度追不上 GPU 执行速度时，GPU 计算流水线会出现明显的空闲间隙。

![CUDA Graphs 捕获固定形状消除 CPU 下发开销机制](/images/content-rebuild/cae75a2ca9f2b2d7/fig-2.png)

### CUDA Graphs 的运作原理
CUDA Graphs 允许将由多个 Kernel、CUDA 事件以及设备内存拷贝构成的完整执行依赖图一次性“捕获（Capture）”。在后续推理循环中，CPU 仅需发起一次整体启动（Launch），GPU 的硬件任务分发单元即可按照拓扑依赖连续执行所有内核，彻底消除算子间的发射停顿。

### 动态 Shape 的局限与分桶（Bucketing）对齐
CUDA Graphs 要求计算图内部张量的**内存地址与张量形状严格固定**，不能在图内部动态分配显存或改变输入张量维度。然而在多用户推理服务中，由于请求随时完成退出或新请求随时加入，每一步 Decode 活跃的有效序列数量是动态变动的。

主流推理引擎（如 vLLM、TensorRT-LLM）的通用解决策略是**批尺寸分桶与预捕获（Batch Size Bucketing）**：
- 在服务初始化阶段，引擎针对预先设定的离散 Batch 尺寸（例如 1, 2, 4, 8, 16, 32, 64 等）分别捕获对应的 CUDA Graph。
- 在显存中为每个图预先分配固定的输入、输出和中间激活缓存区。
- 调度器在每一步 Decode 迭代时，根据当前活跃的实际序列数，向上对齐选取最接近的 Graph（若序列数不足则填充占位 Dummy 序列），确保绝大部分 Decode 步骤直接调用静态编译图，规避 CPU 发射延迟。

---

## 引擎调度核心参数配置：联动关系与防爆指南

以遵循连续批处理（Continuous Batching）和分页内存（PagedAttention）架构的推理引擎（如 vLLM）为例，核心参数直接决定了并发承载上限与首字延迟。

![vLLM 核心参数在显存分配与调度迭代中的联动控制图](/images/content-rebuild/cae75a2ca9f2b2d7/fig-3.png)

### 1. `gpu_memory_utilization`（显存利用率上限）
该参数定义了引擎可分配显存占整卡物理显存的比例（默认通常为 0.90）。分配关系遵循：

$$\text{总分配显存} = \text{模型权重} + \text{动态激活与 CUDA Graph 预留缓存} + \text{KV Cache 物理页池}$$

- **配置过高（如 >0.96）**：剩余给运行时临时激活峰值和驱动上下文的余量极小，容易在高并发或突发长上下文请求进来时触发底层 CUDA Out-Of-Memory (OOM) 导致服务崩溃。
- **配置过低（如 <0.80）**：剩余分配给 KV Cache 的物理内存块数量减少，能够承载的并发 Token 容量大幅缩水。当请求暴增时，由于没有可用内存块，引擎调度器被迫触发抢占机制（Preemption），把部分正在计算的序列 KV Cache 交换到 CPU 内存或丢弃重算，造成 P99 延迟严重恶化。

### 2. `max_num_batched_tokens` 与 `max_num_seqs`
- **`max_num_seqs`**：单次调度迭代中允许的最大序列总数，直接对应系统单卡能并行处理的连接数上限。
- **`max_num_batched_tokens`**：单次调度迭代中，所有参与计算的序列 Token 数量总和上限。

这两个参数构成了调度器在每一步打包请求时的硬性约束。以下是生产环境典型配置范例：

```python
# 生产环境典型部署参数示例（基于 vLLM EngineArgs 配置伪代码）
from vllm import EngineArgs

engine_args = EngineArgs(
    model="meta-llama/Llama-3-70B-Instruct",
    tensor_parallel_size=8,          # 8 卡张量并行拆分权重
    gpu_memory_utilization=0.90,     # 留出 10% 显存供激活峰值与静态 CUDA Graph 使用
    max_num_seqs=128,                # 限制单次迭代最大并发序列数
    max_num_batched_tokens=4096,     # 单步最大处理 Token 预算
    enable_chunked_prefill=True,     # 开启分块预填充，避免长 Prompt 霸占整个流水线
    enable_prefix_caching=True,      # 开启 KV Cache 前缀缓存，复用通用系统提示
)
```

### 分块预填充（Chunked Prefill）的平滑作用
在未开启 `enable_chunked_prefill` 时，若 `max_num_batched_tokens` 设置得很大，当一个 2000 Token 的长 Prompt 请求到达时，调度器会将其完整放入一个步骤完成 Prefill。这意味着整个 GPU 的计算核心在该迭代周期内全部用于 GEMM 矩阵运算，正在并发执行 Decode 的其他 50 个请求会被迫等待挂起，导致它们的 TPOT 瞬间产生严重抖动与延迟尖刺。

开启 `enable_chunked_prefill` 后，调度器把长 Prompt 拆分成若干块（例如每块 512 Token）。在单次迭代中，512 个 Prefill Token 与当前所有 Decode 序列合并在同一个批次中协同执行。这种混合批处理既保证了 GPU 计算核心处于饱和状态，又避免了短文本请求在 Decode 过程中被长时间卡顿。

---

## 推测解码：草稿模型机制与接受率边界

自回归模型每预测一个新 Token 都必须执行一次完整的网络前向传播。推测解码（Speculative Decoding）通过**小模型快速起草、大模型单步批量验证**，突破单步串行限制。

![推测解码两阶段执行时序与接受验证机制](/images/content-rebuild/cae75a2ca9f2b2d7/fig-4.png)

### 执行过程推演
1. **草稿阶段（Drafting）**：使用参数量较小的 Draft 模型（例如 1B 参数模型或前向辅助头），连续自回归生成 $\gamma$ 个候选 Token：$[x_1, x_2, \dots, x_\gamma]$。小模型参数量少，Decode 耗时远低于 70B 主模型。
2. **验证阶段（Verification）**：将主模型的原上下文与这 $\gamma$ 个候选 Token 拼接在一起，由主模型执行一次并行前向计算。主模型此时一次性处理 $\gamma+1$ 个位置，属于算力密集计算，能充分发挥 GPU 的并行计算能力。
3. **判决与修正**：对比 Draft 模型的采样分布与主模型的输出分布。若候选 Token 在第 $k$ 个位置被判定拒绝，则丢弃 $k$ 之后的所有候选结果，主模型同时基于修正分布直接输出第 $k$ 个位置的正确 Token。

### 理论期望与适用边界
假设 Draft 模型的单 Token 生成耗时为 $t_{\text{draft}}$，主模型单 Token 解码耗时为 $t_{\text{target}}$，主模型一次性验证 $\gamma$ 个候选 Token 的耗时为 $t_{\text{verify}}$（通常因为矩阵并行度的提升，$t_{\text{verify}}$ 仅略高于或接近 $t_{\text{target}}$）。

假定每个位置的接受概率相互独立且均为 $\alpha$（注：实际场景中各位置接受概率非严格独立，此处用于理论期望推导）：
- Draft 模型生成的 $\gamma$ 个候选 Token 中，连续被接受的期望数量为：

$$\mathbb{E}[N_{\text{draft\_accepted}}] = \sum_{k=1}^\gamma \alpha^k = \frac{\alpha(1 - \alpha^\gamma)}{1 - \alpha}$$

- 无论前序在哪个位置被拒绝，主模型在拒绝位置（或全被接受后的末尾位置）均会附带输出 1 个确认 Token。因此单轮验证产生的总有效 Token 期望数量为：

$$\mathbb{E}[N_{\text{total}}] = 1 + \mathbb{E}[N_{\text{draft\_accepted}}] = \sum_{i=0}^\gamma \alpha^i = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

单轮验证总耗时为：

$$T_{\text{step}} = \gamma \cdot t_{\text{draft}} + t_{\text{verify}}$$

当平均加速比 $\frac{\mathbb{E}[N_{\text{total}}] / T_{\text{step}}}{1 / t_{\text{target}}} > 1$ 时，推测解码产生正向收益。代入展开可得正向加速边界条件：

$$\frac{1 - \alpha^{\gamma+1}}{1 - \alpha} > \frac{\gamma \cdot t_{\text{draft}} + t_{\text{verify}}}{t_{\text{target}}}$$

加速平衡点由小模型与主模型的单步耗时比、验证耗时以及候选步数 $\gamma$ 共同决定，并非全场景统一的经验常数：
- **收益较高场景**：代码自动补全、结构化 JSON 提取、固定模版生成。由于输出语法与词汇高度确定，Draft 模型的预测与主模型重合度高（实测接受率 $\alpha$ 较高），单轮前向能够接受较多候选 Token，抵消 Draft 生成与验证的额外开销后获得端到端加速。
- **负优化陷阱**：高温度发散对话、多跳复杂推理或高创意文本。此时生成分布的信息熵高，候选词易在第 1~2 个 Token 处即被拒绝，单轮有效 Token 数接近 1。主模型不仅未能减少前向次数，还额外承受了 Draft 模型的串行生成延迟与验证计算开销，导致端到端延迟不降反升。

---

## 调优决策树与归因定位：TTFT 与 TPOT 针对性治理

在日常运维中，优化行动应当由 Prometheus/OpenTelemetry 监控指标或引擎的日志信息直接触发。根据下述决策树，可迅速定位当前系统的性能瓶颈：

![大模型部署性能优化诊断与调优决策树](/images/content-rebuild/cae75a2ca9f2b2d7/fig-5.png)

### 场景 1：TTFT（首字延迟）严重超出目标
- **现象观测**：客户端发送消息后有长时间的排队停顿，但一旦首字吐出，后续输出流畅。
- **归因与治理**：
  1. **队列排队等待（Waiting Time 过长）**：当前活跃的并发序列数已打满 `max_num_seqs`，或 KV Cache 可用物理块告罄，新请求在排队队列中等待。治理方式：在显存允许的前提下适当调高 `max_num_seqs`，或通过张量并行/流水线并行扩展显存，并在网关层配合限流排队机制避免流量击穿。
  2. **Prefill 计算耗时长**：输入 Prompt 过长。治理方式：开启 `enable_chunked_prefill` 避免长 Prompt 阻塞流水线；针对固定 Prompt 模板（如长 System Prompt 或知识库文档）开启 `enable_prefix_caching`（前缀缓存）；升级底层注意力内核至 FlashAttention-2/3。

### 场景 2：TPOT（单字耗时）过高，吞吐量未达标
- **现象观测**：首字返回正常，但后续输出吐字卡顿，单卡每秒输出 Token 数偏低。
- **归因与治理**：
  1. **显存带宽打满（Memory-bound 严重）**：显存读取成为瓶颈。治理方式：采用 FP8、AWQ 或 W8A8 权重量化，将模型权重搬运量减半；开启 FP8 KV Cache 量化降低自回归历史上下文的加载量。
  2. **CPU 发射瓶颈（小并发场景）**：并发度低且未启用静态图。治理方式：确认启用 CUDA Graphs 并合理配置 Batch 分桶大小，降低算子下发等待。
  3. **任务确定性强**：在低温度、格式固定的业务中尝试引入推测解码，监控实测接受率指标以确认是否带来正向收益。

### 场景 3：突发高流量时频繁 OOM 崩溃或抢占抖动
- **现象观测**：服务在低负载时运行良好，突发流量到来时进程报 CUDA OOM 退出，或调度器频繁报 Preemption 导致延迟突增。
- **治理方式**：
  - 检查 `gpu_memory_utilization`。若配置超过 0.92，降低至 0.88~0.90，为突发动态激活与图运行预留安全显存余量（Headroom）。
  - 实行长短请求分流：将超长上下文请求（如 >8K Token）与普通短对话请求路由至不同的推理节点集群，避免长文本霸占所有显存页。

![高风险配置与优化手段的边界陷阱对比](/images/content-rebuild/cae75a2ca9f2b2d7/fig-6.png)

---

## 面试追问与常见认知误区

### 追问 1：推测解码既然能加速自回归，为什么通用生产集群不默认全量开启？
**考核点**：对算法成本与硬件开销边界的理解。
1. **显存挤压**：推测解码需要常驻一个 Draft 模型（即使只有 1B 参数，也会额外占用数 GB 显存），同时验证阶段会申请额外的中间激活缓存。这直接挤占了原本属于主模型的 KV Cache 可用物理块空间，导致整个集群的最大并发容量下降。
2. **场景适应性**：在通用开放域对话和高温度采样任务中，预测接受率容易大幅走低。一旦接受率跌破平衡点，推测解码不仅无法带来速度提升，反而会因为两套模型的调度开销与验证计算，导致系统 TPOT 与端到端延迟全面劣化。

### 追问 2：为什么调大 `max_num_batched_tokens` 反而可能让短文本请求的 P99 延迟变差？
**考核点**：对连续批处理引擎内部调度行为的认知。
- 若未开启分块预填充（Chunked Prefill），当调度器在某一步拉取到一个长 Prompt（例如包含 3000 Token）时，由于 `max_num_batched_tokens` 设得很大，调度器会把这 3000 Token 一次性排进当前计算批次完成整段 Prefill。
- 这段密集的 GEMM 运算会长达几百毫秒独占 GPU 的计算核心。正在并发执行 Decode 的短文本请求在此期间只能排队等待，无法完成单步生成。这就导致短文本请求的单字吐出间隔出现骤增，拉高了整个系统的 P99 尾部延迟。

### 常见认知误区
- **“显存利用率（gpu_memory_utilization）设为 1.0 可以榨干硬件最大并发”**：显存配置打满会完全抹去给 PyTorch 临时激活峰值、底层驱动上下文以及 CUDA Graph 静态缓存预留的缓冲空间。一旦遇到突发请求或长序列分配，将立即触发 CUDA Driver OOM 导致服务进程崩溃。
- **“模型量化只会降低显存占用，不会改善单请求的生成延迟”**：在大模型中小 Batch 的 Decode 阶段，单步计算受限于显存带宽（Memory-bound）。通过权重或 KV Cache 量化（如 FP8、INT8、INT4），每一步从显存搬运到计算芯片的字节数成比例减少，显存带宽读取耗时随之降低，因此能带来显著的单字吐字（TPOT）物理加速。
