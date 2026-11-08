---
title: 大模型推理服务如何做高并发？网关调度、PD分离与背压流控
icon: mdi:book-open-page-variant
description: 解析大模型高并发服务的核心瓶颈，通过Token预算路由、前缀缓存亲和、Prefill与Decode分离以及流式背压机制，构建低延迟稳定的推理集群。
---

### 学习目标
1. 区分大模型推理与传统 Web 服务的并发差异，掌握首字延迟（TTFT）与字间间隔（TPOT）等核心指标。
2. 理解长文本对计算与显存的非线性消耗，掌握基于 Token 预算与前缀缓存（Prefix Caching）的网关路由算法。
3. 掌握 Prefill 与 Decode 阶段的物理瓶颈差异，定量核算 PD 分离架构中的跨节点网络传输开销。
4. 掌握流式长连接中的背压（Backpressure）传递与显存主动释放机制，防止僵尸连接耗尽显存。

---

### 贯穿场景：混合负载下的客服助手集群

假设我们正在维护一个多租户企业知识库与问答集群。系统同时承接两类流量：
* **短文本闲聊/问答**：输入约 200 Token，输出约 300 Token，请求占比 80%；
* **超长文档解析与总结**：输入约 4000 ~ 8000 Token，输出约 500 Token，请求占比 20%。

如果在网关层直接使用常规的四层或七层轮询（Round-Robin）将请求转发给后端的单实例推理引擎（如使用 vLLM，参考 [vLLM 为什么推理速度快？](/llm-100/deployment/vllm-inference.md)），系统很快就会暴露出严重问题：
1. 某台 Worker 刚被分发了两个 8000 Token 的长文档任务，其计算单元被长文本矩阵乘法彻底占满；
2. 此时被分配到该 Worker 的短文本用户，迟迟收不到第一个字符（首字延迟激增）；
3. 正在以打字机模式接收回答的用户，输出突然中断卡顿数秒；
4. 移动端弱网用户断开连接后，后台 GPU 仍在全速生成废弃文本，宝贵的 [KV Cache 显存](/llm-100/deployment/kv-cache.md) 无法回收。

大模型高并发工程的核心任务，就是在有限的 GPU 算力与显存带宽约束下，通过合理的流量调度与架构解耦，平抑长尾延迟并保证吞吐。

![大模型高并发集群总体服务架构与流量调度示意图](/images/content-rebuild/20956d81f917352d/cover.png)

---

### 核心指标定义与物理瓶颈

大模型推理服务的性能评估不能只看每秒请求数（RPS）或简单的平均延迟。自回归模型分为两个计算特征截然不同的阶段：

1. **Prefill（预填充阶段）**：模型一次性读入所有输入的 Prompt Token，并行计算自注意力。这个过程是**计算密集型（Compute-bound）**的，主要开销是稠密矩阵乘法（GEMM），算力利用率较高。
2. **Decode（解码阶段）**：模型每一步只能根据前序所有上下文生成一个新 Token。每生成一个 Token，都需要将全部历史的 Key 和 Value 向量从显存加载到计算单元。这个过程是**访存密集型（Memory-bandwidth-bound）**的，受限于 GPU 显存带宽。

基于这两阶段，大模型服务的服务等级目标（SLO）包含以下核心指标：

* **首字延迟（Time to First Token, TTFT）**：从客户端发送请求到收到第一个输出 Token 的时间间隔。包含网络上行延迟、网关排队等待时间以及 Prefill 计算时间。
* **单字生成耗时（Time Per Output Token, TPOT / Inter-Token Latency, ITL）**：在流式输出中，相邻两个 Token 生成的时间差。它决定了打字机输出是否平稳连贯。
* **端到端总延迟（E2E Latency）**：客户端获得完整响应的总耗时：
  $$E2E = TTFT + TPOT \times (N_{out} - 1)$$
  其中 $N_{out}$ 为实际生成的 Token 总数量。

![大模型推理核心延迟指标与请求生命周期时序图](/images/content-rebuild/20956d81f917352d/fig-1.png)

在传统 Web 架构中，系统只要 CPU 和内存未满，并发容量通常与连接数正相关。而在大模型推理中，显存容量是硬约束。一旦活跃请求的上下文长度超出显存池（Block Pool）上限，系统就会被迫触发抢占驱逐（Preemption）或直接拒绝服务。

---

### 网关调度：Token 预算与前缀缓存亲和路由

如果使用常规轮询，各 Worker 承受的真实计算载荷将严重失衡。网关必须建立在大模型专有的状态感知基础上。

#### 1. 基于 Token 预算的动态负载均衡
网关需要维护后端各 Worker 的实时载荷，衡量标准不是“连接数”，而是“活跃计算量”：
* **待计算的 Prompt Token 量**：决定 Prefill 排队深度；
* **正在生成的序列数量与预估剩余长度**：决定 Decode 访存与显存槽位压力。

网关收到请求后，先利用轻量分词器快速估算该请求的输入长度 $S_{in}$，并结合参数中的最大生成限制 `max_tokens`（记为 $S_{out}$）。网关将请求路由至当前综合 Token 负荷最小的节点，避免某台机器被多个长文本同时击穿。

#### 2. 前缀缓存（Prefix Caching）亲和路由
在多轮对话或固定系统提示词（System Prompt）场景下，推理引擎支持前缀缓存功能（如 vLLM 的 Automatic Prefix Caching，原理参见 [PagedAttention 解决了什么问题？](/llm-100/deployment/paged-attention.md)）。如果节点显存中已经保留了该前缀的历史 KV Cache，模型就不必重新对前缀执行 Prefill，首字延迟可大幅下降。

![基于前缀缓存与Token预算的网关路由决策流程图](/images/content-rebuild/20956d81f917352d/fig-2.png)

网关可以通过维护全局 Radix Tree（基数树）或会话哈希映射表来实现前缀亲和路由：
1. 网关提取请求的前缀内容（如系统 Prompt + 历史对话）并计算其特征哈希；
2. 检查此前将该前缀派发给了哪个 Worker 节点；
3. 如果目标节点的排队深度未达到过载阈值，优先将请求派发至同一 Worker 节点；
4. 若目标节点已过载，则退化至负载最低的其他节点。

---

### Prefill-Decode 分离（PD 分离）

在传统的单节点“混跑模式”中，Prefill 和 Decode 运行在同一批 GPU 上。这种共享方式会导致明显的性能冲突：**计算密集的 Prefill 任务会抢占访存密集的 Decode 任务的执行时间**。

当大量短请求正在平稳进行 Decode 输出时，突然涌入一个 4000 Token 的长文档总结请求。即使引擎开启了 Chunked Prefill 将大矩阵运算分片，每一个分片在 GPU 执行时依然会独占计算单元，导致正在输出的 Decode 请求单步耗时（TPOT）大幅增加，用户直观感受就是流式输出发生卡顿。

![Prefill与Decode混跑导致打字机卡顿的冲突机制分析图](/images/content-rebuild/20956d81f917352d/fig-3.png)

#### PD 分离架构原理
为了从物理硬件上隔离这两种计算特性，业界演进出了 **Prefill-Decode 分离（PD 分离）** 架构：

1. **Prefill 集群**：专职负责接收 Prompt，进行高强度的 GEMM 运算，生成首个 Token 和全部上下文的 KV Cache；
2. **Decode 集群**：专职负责自回归逐步输出，配置大容量与高带宽显存，运行高并发的批处理 Decode；
3. **跨节点 KV Cache 传输网络**：Prefill 节点完成计算后，将 KV Cache 张量通过网络发送至 Decode 节点，后续的流式生成全部由 Decode 节点接管。

![PD分离架构下跨节点KV Cache传输与硬件网络瓶颈图](/images/content-rebuild/20956d81f917352d/fig-4.png)

#### 跨节点网络传输开销定量推导
PD 分离的核心工程代价是**网络传输开销**。我们通过一个具体参数量化一次传输的数据体量：

以常见的 **Llama-3-8B（FP16 精度）** 为例：
* 模型层数 $L = 32$
* KV 注意力头数 $H_{kv} = 8$（采用 GQA 结构）
* 每个头的维度 $D = 128$
* 每个元素占用字节数 $P = 2$ 字节（FP16 精度）

单个 Token 产生的 Key 和 Value 向量总字节数为：
$$Bytes\_per\_token = 2 \times L \times H_{kv} \times D \times P = 2 \times 32 \times 8 \times 128 \times 2 = 131,072 \text{ 字节} = 128 \text{ KB}$$

当输入 Prompt 长度为 4096 Token 时，需传输的 KV Cache 状态量为：
$$Total\_Size = 4096 \times 128 \text{ KB} = 524,288 \text{ KB} = 512 \text{ MB}$$

我们对比不同网络环境下的传输时间（仅计算物理带宽传输耗时，不含序列化与协议栈）：
* **普通千兆以太网（1 Gbps $\approx$ 125 MB/s）**：
  $$T = \frac{512 \text{ MB}}{125 \text{ MB/s}} \approx 4.10 \text{ 秒}$$
* **常规万兆以太网（10 Gbps $\approx$ 1250 MB/s）**：
  $$T = \frac{512 \text{ MB}}{1250 \text{ MB/s}} \approx 0.41 \text{ 秒} = 410 \text{ ms}$$
* **数据中心专用高速网络（RDMA / 200 Gbps $\approx$ 25 GB/s）**：
  $$T = \frac{512 \text{ MB}}{25,000 \text{ MB/s}} \approx 0.02 \text{ 秒} = 20 \text{ ms}$$

在万兆以太网下，仅仅网络传输就需要 410 ms，这一延迟甚至超过了 8B 模型在主流加速卡上的 Prefill 本身计算时间。因此，**PD 分离架构高度依赖基于 RDMA（RoCE v2 或 InfiniBand）的高带宽互联网络**。在缺乏高速 RDMA 网络的普通机房环境中，盲目采用 PD 分离反而会导致首字延迟（TTFT）大幅恶化。相关硬件配置评估可参考 [本地部署大模型需要哪些硬件？](/llm-100/deployment/hardware-estimation.md)。

---

### 流式传输、背压机制与显存悬挂保护

大模型服务绝大多数采用流式输出（Server-Sent Events, SSE 或 gRPC Stream）。这种长连接机制带来了一个高并发下的隐蔽故障点：**下游慢消费导致的显存悬挂**。

#### 慢客户端与显存悬挂
如果某个客户端处于弱网环境，下行带宽仅 5 KB/s，而模型推理每秒吐出 40 个 Token：
1. 网关向客户端发送的数据迅速积压在系统的 TCP 发送缓冲区中；
2. 若网关没有背压流控，网关会持续从推理引擎拉取生成结果；
3. 推理引擎持续分配计算与显存槽位，直到该请求全部生成完毕；
4. 更严重的是：若用户中途直接关闭了浏览器窗口，客户端 TCP 连接断开，但推理引擎由于未收到中止信号，仍会继续把后续几百个 Token 计算完毕。

在高并发场景下，少量这种悬挂连接就会把集群的 KV Cache 槽位占满，导致新请求全部排队超时。

![客户端慢消费引发长连接背压传导链条与资源悬挂示意图](/images/content-rebuild/20956d81f917352d/fig-5.png)

#### 异步流式背压与取消处理实现
网关与流式处理层必须建立端到端的背压传递：当网络缓冲区满时暂停拉取，并在客户端断开时主动向引擎发出取消（Abort）请求。

```python
import asyncio
from typing import AsyncGenerator

class MockInferenceEngine:
    """模拟推理引擎接口"""
    async def abort_request(self, request_id: str) -> None:
        # 引擎接收到后立即将请求移出批处理队列，归还其在 PagedAttention 中的显存块
        print(f"[Engine] Request {request_id} aborted, KV Cache released.")

async def stream_tokens_with_backpressure(
    engine_stream: AsyncGenerator[str, None],
    request_id: str,
    engine: MockInferenceEngine,
    max_buffer_size: int = 16
) -> AsyncGenerator[str, None]:
    """
    具备有界缓冲与客户端中断检测的流式生成器
    """
    # 使用有界队列建立生产与消费之间的背压通道
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=max_buffer_size)

    async def producer():
        try:
            async for token in engine_stream:
                # 当客户端消费过慢导致 queue 填满时，put() 会自然协程阻塞
                # 从而停止从上游引擎拉取数据，形成背压
                await queue.put(token)
            await queue.put(None)  # 结束标记
        except asyncio.CancelledError:
            # 生产者任务被取消
            pass

    producer_task = asyncio.create_task(producer())

    try:
        while True:
            token = await queue.get()
            if token is None:
                break
            # yield 给 Web 框架（如 FastAPI/Starlette）推送给客户端网络 Socket
            # 若网络层阻塞，yield 所在协程会等待发送完成
            yield f"data: {token}\n\n"
            queue.task_done()

    except (asyncio.CancelledError, ConnectionResetError, GeneratorExit):
        # 客户端主动切断连接、网络异常中断或生成器被外部关闭，进入异常分支
        print(f"[Gateway] Client disconnected unexpectedly for request {request_id}")
        # 1. 立即取消内部拉取任务
        producer_task.cancel()
        # 2. 显式通知底层推理引擎释放该请求对应的 KV Cache 物理块
        await engine.abort_request(request_id)
        raise
    finally:
        if not producer_task.done():
            producer_task.cancel()
```

---

### 过载防护与可观测性看板

当突发流量超出物理算力极限时，系统需要具备防御机制，防止整组 Worker 发生级联崩溃。

#### 1. 过载丢弃与动态降级
* **等待队列 TTL（Queue Timeout）**：请求在网关等待队列超过设定阈值（例如 5 秒）未被调度时，直接返回 `429 Too Many Requests`。用户在超时后通常已经刷新页面，继续计算毫无意义。
* **动态压制输出长度**：当全集群显存使用率超过 85% 时，临时将非 VIP 租户的 `max_tokens` 参数压制为更低值（例如从 2048 降为 512），缩短请求在 Decode 阶段停留的时间。
* **并发采样降级**：若请求设置了 `n > 1`（如一次生成多条回复供挑选），过载期间强制降级为 `n = 1`，将显存分支开销降至最低。

#### 2. 核心监控看板

| 指标分类 | 核心监控项 | 正常基准参考 | 预警条件与排查方向 |
| :--- | :--- | :--- | :--- |
| **延时指标 (SLO)** | TTFT (P95 / P99) | 500ms ~ 1500ms | P99 > 3s：通常是长文本 Prefill 阻塞或排队过深 |
| | TPOT / ITL (P95 / P99) | 20ms ~ 50ms (依模型尺寸) | 抖动 > 50%：Prefill 干扰 Decode，需检查 Chunk 大小或推进 PD 分离 |
| **显存与调度** | KV Cache 使用率 (%) | 60% ~ 80% | 持续 > 90%：面临 OOM 风险，需扩容或开启过载丢弃 |
| | Prefix Cache 命中率 | 20% ~ 50% (依业务) | 显著下降：检查网关路由亲和性哈希是否失效 |
| **连接与网络** | 活跃流式连接数 | 依业务基线波动 | 突增但有效输出吞吐（TPS）极低，或客户端感知 TPOT 剧烈恶化：排查慢客户端显存悬挂 |
| | 跨节点 RDMA 吞吐 | 依集群规格 | 接近物理网卡上限：PD 分离架构发生网络瓶颈 |

![大模型高并发集群端到端监控体系与核心指标看板](/images/content-rebuild/20956d81f917352d/fig-6.png)

---

### 可检验练习

#### 练习 1：网络吞吐定量计算
某团队计划为其自建的 Qwen2-72B（FP16 精度）推理集群搭建 PD 分离架构：
* 模型结构参数：层数 $L = 80$，注意力头数 $H_{q} = 64$，$H_{kv} = 8$（GQA），头维度 $D = 128$；
* 业务场景：平均输入 Prompt 长度为 2048 Token；系统峰值需要支持每秒完成 20 个请求的 Prefill 并转交 Decode 集群。

**问题**：
1. 每个 Token 的 KV Cache 占用多少字节？
2. 单个 2048 Token 请求产生的 KV Cache 总大小是多少 MB？
3. Prefill 集群与 Decode 集群之间，至少需要多大带宽的高速网络通道（以 Gbps 为单位，按 1 Byte = 8 bits 计算），才能保证跨节点传输不成为瓶颈？

#### 练习 2：故障排查分析
线上监控报警显示：某时段集群的平均 RPS 并没有明显上升，但客户端统计的 TPOT P99 从平时的 30 ms 骤升至 450 ms，导致大量用户反馈打字卡顿。同时观察到 KV Cache 使用率稳定在 65%，并没有触顶。

请结合混跑架构与流式通信机制，列出可能导致该现象的两个排查方向。

---

### 练习参考解析

* **练习 1 解析**：
  1. 单 Token 字节数：$2 \times L \times H_{kv} \times D \times 2 = 2 \times 80 \times 8 \times 128 \times 2 = 327,680 \text{ 字节} = 320 \text{ KB}$。
  2. 单请求 KV Cache：$2048 \times 320 \text{ KB} = 655,360 \text{ KB} = 640 \text{ MB}$。
  3. 每秒完成 20 个请求，则每秒需要传输的数据总量为 $20 \times 640 \text{ MB} = 12,800 \text{ MB/s} = 12.8 \text{ GB/s}$。换算为网络比特率：$12.8 \times 8 = 102.4 \text{ Gbps}$。考虑到网络协议头和重传余量，两集群间必须至少部署 200 Gbps 级别的 RDMA 网络。
* **练习 2 解析**：
  1. **长文本突发导致计算抢占**：虽然总请求数（RPS）没变，但可能混入了输入 Token 数极大的长文本请求。在未开启分块 Prefill 或未做 PD 分离的环境中，长 Prefill 的大 GEMM 计算占用了大部分流处理器（SM），导致原有的 Decode 任务被挂起延后。
  2. **慢客户端引发的网络阻塞**：检查网关下游的 TCP 连接状态，若客户端网络吞吐受限且网关缺少背压流控，生成任务会在缓冲区积压，导致单字输出到网络 socket 的耗时被拉长，统计出的客户端 TPOT 指标出现恶化。
