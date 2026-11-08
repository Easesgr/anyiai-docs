---
title: 大模型推理为什么慢？Prefill 与 Decode 的性能瓶颈在哪里？
icon: mdi:book-open-page-variant
description: 通过 Roofline 性能模型与 7B 模型算例，量化解析 Prefill 算力受限与 Decode 访存受限机制，理解批处理对吞吐与延迟的影响。
---

在大模型部署中，开发者常常会观察到两个现象：
1. 输入一段上万字的长文本，GPU 风扇高速运转，等待几秒后吐出第一个字；
2. 第一个字出来后，模型逐字生成回答的速度却相对固定，哪怕每次只生成一个字，`nvidia-smi` 看到的显存占用和 GPU 利用率依然居高不下。

大模型推理为什么不能像普通软件一样瞬间给出结果？为什么第一个字的等待时间和后续逐字生成的节奏完全不同？

本文建立一个可量化的硬件性能分析视角。我们以一个 7B 参数模型在单张 NVIDIA A100 显卡上的实际运行过程为贯穿示例，拆解自回归推理的两个阶段，利用体系结构中的 Roofline 性能模型推导硬件瓶颈，并说明批处理（Batch Size）如何重塑推理性能。

![大模型推理性能瓶颈与两阶段特征全景图](/images/content-rebuild/dfae9cdca219814c/cover.png)

---

### 1. 贯穿示例与两阶段特征：Prefill 与 Decode

以开源社区常见的 7B 参数 Transformer 模型为例，在 FP16（半精度浮点数，每个参数占用 2 字节）精度下：
- 模型权重体积为：$7 \times 10^9 \times 2 \text{ 字节} = 14\text{ GB}$；
- 运行硬件为一张单卡 NVIDIA A100 SXM 80GB，其 FP16 峰值算力为 $312\text{ TFLOPs}$（每秒 $312 \times 10^{12}$ 次浮点计算），高带宽显存（HBM2e）的读取带宽为 $2039\text{ GB/s}$（约每秒 $2.04\text{ TB}$）。

当用户向该模型提交一段包含 2048 个 Token 的提示词（Prompt）并要求其生成回答时，整个推理过程在物理执行上严格分为两段：预填充阶段（Prefill）与自回归解码阶段（Decode）。

![Prefill 阶段与 Decode 阶段在输入形式与处理机制上的因果对比](/images/content-rebuild/dfae9cdca219814c/fig-1.png)

#### 预填充阶段（Prefill / 处理输入）
在客户端发送提示词后，服务端一次性收到了完整的 2048 个 Token。
- **处理机制**：由于这 2048 个输入 Token 已经全部已知，它们可以在网络各层中拼成矩阵，通过通用的矩阵乘法（GEMM）并行计算出对应的隐藏层状态。
- **主要工作**：抽取输入的语义，并在显存中计算并保存每一层历史 Token 的键向量（Key）和值向量（Value），形成 KV Cache（关于键值缓存机制见 [/llm-100/deployment/kv-cache.md](/llm-100/deployment/kv-cache.md)），最后产出第一个预测 Token。
- **对应体验**：Prefill 的耗时直接构成了用户等待第一个字出现的时间（首字延迟，TTFT）。

#### 解码阶段（Decode / 逐字生成）
输出第一个 Token 后，模型进入自回归循环：每一次将上一轮新生成的单个 Token 作为输入，结合此前所有保存在显存里的 KV Cache，预测下一个 Token。
- **处理机制**：单请求场景下，每一步迭代只输入 1 个 Token。模型的投影计算从多行矩阵乘法退化为向量-矩阵乘法（GEMV）。
- **时序依赖**：第 $t$ 步的输入严格依赖第 $t-1$ 步采样出的新 Token，无法提前并行预测后面的 Token。
- **对应体验**：Decode 每一步的耗时，构成了用户看到字符逐个打在屏幕上的字间延迟（TPOT）。

下表汇总了这两个阶段的关键计算特征：

| 维度 | Prefill 阶段（预填充） | Decode 阶段（逐字生成） |
| :--- | :--- | :--- |
| **单次输入形状** | 序列维度全量输入（如 $S = 2048$） | 单步单个增量输入（$S = 1$） |
| **核心矩阵运算** | GEMM（矩阵乘矩阵） | GEMV（向量乘矩阵，单并发时） |
| **计算并行度** | 序列内 Token 高度并行 | 跨时间步必须串行依赖 |
| **显存权重读取** | 一次载入权重被成千上万个输入 Token 复用 | 仅为预测 1 个 Token，需重新完整读取一次权重 |
| **用户可观测指标**| 首字延迟（TTFT） | 字间延迟（TPOT） |

---

### 2. 分析工具：Roofline 性能模型

要量化为什么两个阶段的速度瓶颈完全不同，需要引入计算机体系结构中用于分析程序性能上限的 **Roofline 性能模型**。

![Roofline 模型坐标系与计算受限区和访存受限区的划分](/images/content-rebuild/dfae9cdca219814c/fig-2.png)

#### 核心概念定义
任何程序在 GPU 上运行，其执行时间都会受限于以下两项硬件物理参数之一：
1. **处理器峰值算力（Peak Performance, $P_{\text{peak}}$）**：芯片在单位时间内能完成的浮点计算次数上限，单位为 FLOPs（每秒浮点运算次数）。
2. **显存访问带宽（Memory Bandwidth, $B_{\text{mem}}$）**：芯片与显存之间每秒能传输的数据字节上限，单位为 Bytes/s。

衡量算法对硬件资源需求的物理量是**算术强度（Operational Intensity, $I$）**：
$$
I = \frac{\text{浮点运算量 (FLOPs)}}{\text{显存访存量 (Bytes)}}
$$
算术强度表示算法每从显存中搬运 1 个字节的数据，能够在计算单元内部完成多少次浮点运算。

Roofline 模型通过下式给出算子能够达到的理论计算性能上限 $P_{\text{attainable}}$：
$$
P_{\text{attainable}} = \min\left(P_{\text{peak}}, I \times B_{\text{mem}}\right)
$$

#### 拐点算术强度与瓶颈分类
两条物理边界的交点被称为硬件的**拐点算术强度**（$I^*$）：
$$
I^* = \frac{P_{\text{peak}}}{B_{\text{mem}}}
$$
- **访存受限区（Memory-bound / 带宽受限）**：当算法的算术强度 $I < I^*$ 时，上限落在斜线部分（$P_{\text{attainable}} = I \times B_{\text{mem}}$）。此时算力单元处于饥饿状态，大量时间在等待显存把数据搬运至芯片内部缓存，算力未被充分使用。要提升处理速度，必须提升显存带宽或减少数据搬运量。
- **计算受限区（Compute-bound / 算力受限）**：当算法的算术强度 $I \ge I^*$ 时，上限落在水平屋顶部分（$P_{\text{attainable}} = P_{\text{peak}}$）。此时显存带宽足以供给数据，计算流水线被填满，硬件吞吐达到峰值。要提升处理速度，必须提升芯片峰值算力或精简总计算量。

#### A100 硬件拐点算例
对上述 A100 SXM 80GB 显卡代入参数：
$$
I^* = \frac{312 \times 10^{12} \text{ FLOP/s}}{2039 \times 10^9 \text{ Bytes/s}} \approx 153.0 \text{ FLOP/Byte}
$$
这个数值意味着：在这张卡上，一个算子每从显存读取 1 个字节，至少要在内部执行 153 次浮点乘加运算，才能让 Tensor Core 计算单元全速运转。如果达不到 153 FLOP/Byte，计算单元就必然存在等待。

---

### 3. Prefill 阶段：为什么长提示词属于计算受限

在 Prefill 阶段，模型处理的是一个长为 $S$ 的输入序列。以网络中的全连接投影层为例分析其算术强度。

![Prefill 阶段长序列 GEMM 矩阵乘法带来的高算术强度与算力占满机制](/images/content-rebuild/dfae9cdca219814c/fig-3.png)

#### 投影层的算术强度推导
设模型的隐藏层维度为 $d$，投影权重矩阵的形状为 $d \times d$。在 FP16 精度下：
- **显存访存量**：从显存读取一次权重矩阵，需要读取 $2 \times d^2$ 字节；输入激活矩阵（大小 $S \times d$）与输出激活矩阵各需要读取/写入 $2 \times S \times d$ 字节，总访存量为 $2d^2 + 4Sd$ 字节。
- **浮点计算量**：输入的 $S$ 个 Token 向量同时与权重矩阵做乘加计算，每次乘加对应 2 次浮点操作，总计算量为 $2 \times S \times d^2$ FLOPs。

因此，该层的算术强度为：
$$
I_{\text{prefill}} = \frac{2 \cdot S \cdot d^2}{2 \cdot d^2 + 4 \cdot S \cdot d} = \frac{S}{1 + \frac{2S}{d}}
$$
在常见的 7B 模型中，隐藏层维度 $d = 4096$。当输入序列长度 $S = 2048$ 时：
$$
I_{\text{prefill}} = \frac{2048}{1 + \frac{2 \times 2048}{4096}} = \frac{2048}{2} = 1024 \text{ FLOP/Byte}
$$

#### 物理运行状态
推导得出的算术强度 $1024\text{ FLOP/Byte}$ 远高于 A100 的硬件拐点 $153.0\text{ FLOP/Byte}$。这意味着权重矩阵只需要从显存加载一次，就会被 2048 个 Token 高频复用。

在 Roofline 模型上，Prefill 阶段落在了水平的**计算受限区**：
1. 显存带宽不再是瓶颈，GPU 的 Tensor Core 流水线处于饱和运行状态；
2. 该阶段的耗时主要取决于序列长度与总计算量；
3. 优化 Prefill 的主要方向包括：使用张量并行分散矩阵乘计算、使用 FlashAttention 降低注意力中间矩阵对显存的冗余搬运。

---

### 4. Decode 阶段：为什么单步逐字生成极度依赖显存带宽

当进入 Decode 阶段，情况发生逆转。假设当前只有 1 个并发请求（Batch Size $B = 1$），每一步只处理 1 个 Token。

![Decode 阶段 GEMV 中显存巨幅搬运与计算核心饥饿空闲的因果机制](/images/content-rebuild/dfae9cdca219814c/fig-4.png)

#### 单步算术强度的计算
继续以 7B 模型单卡推理为例：
- **模型参数量**：70 亿参数，FP16 占用 $14\text{ GB}$ 显存；
- **单步计算量**：对于这 1 个 Token，模型中每个参数进行 1 次乘法和 1 次加法，计算量为 $2 \times 7 \times 10^9 = 14\text{ GFLOPs}$；
- **显存访存量**：由于 $B = 1$，显卡无法在多个 Token 间复用参数。为了生成这 1 个 Token，必须把整整 $14\text{ GB}$ 的模型权重从 HBM 读取到芯片内部一次。由于单步激活值只有几千字节，可以忽略不计，总访存量近似为 $14\text{ GB}$。

此时的算术强度为：
$$
I_{\text{decode}} = \frac{14 \times 10^9 \text{ FLOPs}}{14 \times 10^9 \text{ Bytes}} = 1.0 \text{ FLOP/Byte}
$$

对比 A100 的拐点值 $153.0\text{ FLOP/Byte}$，此时的算术强度甚至不足拐点的 $1\%$。该阶段深陷在 Roofline 模型的最左侧斜线区——**访存受限区**。

#### 理论生成速度与硬件空转真相
在访存受限状态下，计算耗时完全被显存读取耗时掩盖。单步迭代的最短物理耗时为：
$$
t_{\text{step}} \ge \frac{14\text{ GB}}{2039\text{ GB/s}} \approx 6.86 \text{ 毫秒}
$$
因此单流请求下，每秒理论最大生成 Token 数上限约为：
$$
\frac{1}{0.00686 \text{ s}} \approx 145.7 \text{ Tokens/s}
$$
在花费的 $6.86\text{ 毫秒}$ 内，A100 的 Tensor Core 理论上能完成 $312 \times 10^{12} \times 0.00686 \approx 2.14 \times 10^{12}$ 次浮点运算，但实际只完成了 $0.014 \times 10^{12}$ 次。**计算单元的实际算力利用率仅为 $0.65\%$**。

为什么此时在终端运行 `nvidia-smi`，看到的 `GPU-Util` 依然经常显示为 100%？
这是因为 `nvidia-smi` 统计的指标是流式多处理器（SM）的时间片活跃比率。在采样周期内，只要 SM 上有任何线程被分配执行，就会被计入活跃。而在 Decode 阶段，SM 上的线程大部分时钟周期都在发出内存读取请求后停顿等待（Stall），计算单元（Tensor Core / ALU）大部分时间处于空转等待数据的状态。这种高占用率是访存等待带来的表象。

---

### 5. 批处理（Batch Size）对瓶颈的平移机制

要让 Decode 阶段摆脱极低算术强度的困境，直接手段是提升**批处理大小（Batch Size, $B$）**。

![Batch Size 增大时算术强度向上平移与吞吐/延迟的变化曲线](/images/content-rebuild/dfae9cdca219814c/fig-5.png)

#### 算术强度随并发线性增长
当推理引擎将 $B$ 个独立请求合并到同一步迭代中处理时：
- 每一层的模型参数（14 GB）依然只需从显存全量读取**一次**；
- 这份参数被载入内部缓存后，同时与 $B$ 个请求的激活向量进行矩阵计算；
- 此时单步浮点计算量提升为 $B \times 14\text{ GFLOPs}$。

忽略 KV Cache 读取对访存的额外开销，Decode 阶段的算术强度与 $B$ 近似成正比：
$$
I_{\text{decode}}(B) \approx B \times 1.0 \text{ FLOP/Byte}
$$
当 $B = 32$ 时，算术强度提升到 $32\text{ FLOP/Byte}$；当 $B = 128$ 时，算术强度提升到 $128\text{ FLOP/Byte}$，逐步逼近硬件拐点 $153\text{ FLOP/Byte}$。此时计算单元的空转被填补，硬件算力得到更充分的利用。

#### 吞吐量与单请求延迟的权衡
增大 Batch Size 伴随着系统权衡：
1. **系统吞吐量（Tokens/s）显著上升**：显存带宽被有效利用，单位时间内整个集群输出的总 Token 数量大幅提升；
2. **单请求字间延迟（TPOT）变化趋势**：当 $B$ 较小时，算子严重受限于访存，计算增加并未延长整体耗时，单步时间几乎保持平稳；但当 $B$ 超过拐点进入计算受限后，算力耗尽开始排队，单步时间随 $B$ 线性增长，单个用户的字间延迟开始变长；
3. **显存容量瓶颈先于算力到达**：处理 $B$ 个请求意味着显卡必须同时在显存中保留 $B$ 份上下文的 KV Cache。在高并发场景下，显存容量往往比算力更快耗尽。如何管理并发碎片，是 [/llm-100/deployment/paged-attention.md](/llm-100/deployment/paged-attention.md) 和 [/llm-100/deployment/vllm-inference.md](/llm-100/deployment/vllm-inference.md) 所要解决的核心问题。

---

### 6. 推理性能指标拆解与基准测试

评估推理服务性能时，不能使用宽泛的“响应时间”，而需要依据两阶段特征拆解为对应的量化指标。

![端到端推理请求时间线拆解与 TTFT 及 TPOT 测算基准](/images/content-rebuild/dfae9cdca219814c/fig-6.png)

#### 核心指标定义

1. **首字延迟（TTFT, Time To First Token）**
   - 定义：从客户端发出请求到接收到第 1 个生成 Token 的耗时；
   - 构成：包含排队时间、网络往返以及服务端 **Prefill 阶段耗时**。长输入场景下 Prefill 计算量决定了 TTFT 的下限。
   $$
   \text{TTFT} = t_{\text{queue}} + t_{\text{prefill}}(S_{\text{in}})
   $$

2. **字间延迟（TPOT, Time Per Output Token）**
   - 定义：收到首字后，后续每连续生成一个 Token 的平均耗时；
   - 构成：对应服务端当前批处理下的 **Decode 单步耗时**。
   $$
   \text{TPOT} = \frac{t_{\text{decode}}(S_{\text{out}} - 1)}{S_{\text{out}} - 1}
   $$

3. **端到端总延迟（E2E Latency）**
   $$
   \text{Latency}_{\text{E2E}} = \text{TTFT} + (S_{\text{out}} - 1) \times \text{TPOT}
   $$

4. **系统总吞吐量（Throughput）**
   - 定义：推理集群在单位时间内交付的生成 Token 总数（Tokens/s）。

#### 可复现的客户端基准测试脚本
以下脚本通过流式接口采集单请求的 TTFT 与 TPOT：

```python
import time
import statistics
from typing import List, Tuple, Any

def benchmark_inference_step(engine_client: Any, prompt: str) -> Tuple[float, float, float]:
    """
    测量单次推理请求的 TTFT、TPOT 与 端到端总耗时
    """
    # 1. 记录发起时刻
    t_start = time.perf_counter()
    
    # 2. 调用流式生成接口
    stream_response = engine_client.generate_stream(prompt)
    
    t_first_token = None
    token_timestamps: List[float] = []
    
    for _ in stream_response:
        now = time.perf_counter()
        token_timestamps.append(now)
        if t_first_token is None:
            t_first_token = now
            
    t_end = time.perf_counter()
    
    if not token_timestamps:
        raise RuntimeError("未收到任何生成的 Token")
        
    # 3. 统计计算
    ttft = t_first_token - t_start
    output_count = len(token_timestamps)
    
    if output_count > 1:
        intervals = [
            token_timestamps[i] - token_timestamps[i - 1]
            for i in range(1, output_count)
        ]
        tpot = statistics.mean(intervals)
    else:
        tpot = 0.0
        
    e2e_time = t_end - t_start
    return ttft, tpot, e2e_time
```

基准测试注意事项：
- **预热（Warmup）**：正式测量前需运行 3~5 轮请求，确保 GPU 驱动上下文初始化、CUDA 运行时编译缓存均已就绪；
- **固定输入输出长度**：测试时应固定输入 Prompt 的 Token 数与输出 Token 数，避免长短混合干扰均值；
- **单流与高并发隔离**：单并发用于测量硬件最低延迟底线，多并发用于测量系统吞吐瓶颈与排队临界点。

---

### 7. 瓶颈对部署优化的指导

理清 Prefill（算力受限）与 Decode（显存带宽受限）的物理特征后，针对推理性能的优化方向可以清晰分类：

1. **缓解 Decode 阶段显存带宽受限**：
   - **模型权重量化**：将 FP16（2 字节/参数）量化为 INT8（1 字节）或 INT4（0.5 字节）。当模型权重从 14 GB 缩减至 7 GB 或 3.5 GB 时，每步 Decode 从显存搬运的数据量减少一半或四分之三，在带宽不变的情况下，单步耗时直接按比例缩短。原理与方法见 [/llm-100/deployment/quantization-methods.md](/llm-100/deployment/quantization-methods.md)。
   - **连续批处理（Continuous Batching）**：在迭代粒度上动态插入已完成 Prefill 的请求，维持高 Batch Size，提升算术强度。机制解析见 [/llm-100/deployment/high-concurrency-serving.md](/llm-100/deployment/high-concurrency-serving.md)。
   - **投机采样（Speculative Decoding）**：利用小模型先草拟多个 Token，再由大模型单步并行验证，将多次受限于访存的 GEMV 转换为一次受限于算力的 GEMM。

2. **缓解 Prefill 阶段算力与显存压力**：
   - **分块预填充（Chunked Prefill）**：长文本 Prefill 一次性计算会占用大量算力并阻塞其他请求的 Decode，将其切块与 Decode 混合调度可平滑 TTFT 与 TPOT；
   - **KV Cache 高效管理**：长文本 Prefill 会产生大量键值缓存，显存碎片问题需要分页管理，我们在下一讲 [/llm-100/deployment/kv-cache.md](/llm-100/deployment/kv-cache.md) 与 [/llm-100/deployment/paged-attention.md](/llm-100/deployment/paged-attention.md) 中展开。

---

### 8. 动手练习

完成以下两个练习以验证对两阶段瓶颈的理解：

1. **硬件拐点与批处理推导**：
   - 查阅某显卡参数（例如 RTX 4090：FP16 Tensor Core 算力约为 $165\text{ TFLOPs}$，GDDR6X 显存带宽为 $1008\text{ GB/s}$）。
   - 请计算该卡的拐点算术强度 $I^*$。
   - 若在单张 RTX 4090 上运行一个 FP16 精度的 7B 模型，在忽略 KV Cache 显存读取的前提下，Batch Size 至少要达到多少，算术强度才能达到该卡的拐点？

2. **可观测指标验证实验**：
   - 准备一个支持流式输出的本地模型（或使用上述测试脚本连接测试端点）；
   - 保持输出最大长度固定为 64，分别输入长度为 64 Token 和 2048 Token 的 Prompt；
   - 记录两次请求的 TTFT 与 TPOT。观察哪一个指标发生了倍数级增长，哪一个指标基本保持不变，并解释其与 Prefill/Decode 阶段瓶颈的对应关系。
