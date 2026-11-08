---
title: 模型服务如何做限流、熔断、超时和降级？
icon: mdi:shield-check-outline
tag:
- 模型部署面试
- 限流
- 熔断
description: 深入讲解大模型服务网关的流量治理：RPM/TPM 双重限流、KV Cache 引擎背压联动、断连中止与多级降级策略。
---

在部署面向生产环境的大模型推理集群时，流量治理是保障系统高可用的关键。与传统无状态 Web 服务相比，大模型推理具有计算密集与显存密集的双重特征，单个请求消耗的资源存在严重的非对称性。若仍沿用传统基于 QPS（Query Per Second）的粗粒度流量控制，在突发长序列流量冲击下，极易引发 KV Cache 耗尽、请求被反复抢占驱逐，甚至导致整个推理集群雪崩。

![大模型网关流量治理全景](/images/content-rebuild/21fc98d9f5ac2f88/cover.png)

上图展示了大模型网关整体防护框架：网关层位于客户端与推理集群之间，协同执行 Token 限流、引擎背压感知、断连取消与熔断降级。

---

### 口述答案：大模型流量治理与传统 Web 服务的核心区别

面试中被问及“模型服务如何做限流、熔断与降级”时，可以按照以下四个要点组织回答：

1. **限流维度从请求数转为 Token 吞吐**：传统 Web 请求的算力与内存开销差异通常在同一个数量级内，控制 QPS 即可；大模型输入输出长度从几十到上万 Token 不等，算力与显存开销相差数十至上百倍。网关必须实行 **RPM（Request Per Minute）与 TPM（Token Per Minute）** 双重控制，并通过**“入站预扣预估 Token、流式结算退还差额”**防止显存透支。
2. **流量控制由被动截流转为引擎背压（Backpressure）联动**：网关不能仅依据自身计数器放行，需实时采集推理引擎（如 vLLM）的内部状态，核心指标是 **KV Cache 块使用率**与**排队等待请求数**。当显存水位超过安全阈值时，网关主动实施反压，返回 `HTTP 429` 或 `503` 并附带动态退避时间，避免引擎内部发生调度抢占（Preemption）。
3. **超时与熔断强调全链路的主动中止（Cancel on Disconnect）**：大模型生成通常耗时数秒到数十秒。若客户端提前断开连接或网关超时，网关必须向底层引擎发送 `Abort` 信号，立即释放其占用的 KV Cache，杜绝“幽灵解码”；同时将超时细分为排队、首 Token 延迟与 Token 间间隔三个维度，对产生严重显存碎片的 Pod 设立熔断隔离与自动拉起机制。
4. **降级策略围绕上下文缩减与模型路由展开**：除了排队等待，大模型特有的降级手段包括动态压缩非关键上下文（如减少 RAG 召回块数）、自适应收紧 `max_tokens`，以及在过载时将非核心请求分流到小参数模型或量化后备集群。

---

### 为什么传统 QPS 限流在大模型场景下会失效？

传统基于 QPS 或并发连接数的限流方案，预设单次请求的物理开销大致均等。而在大模型推理中，资源占用直接受**输入序列长度（$L_{in}$）**与**输出序列长度（$L_{out}$）**的主导。

![传统QPS与大模型Token级资源消耗对比](/images/content-rebuild/21fc98d9f5ac2f88/fig-1.png)

上图对比了传统服务与大模型服务在相同 QPS 下的计算与显存开销。传统请求资源消耗呈均匀分布，而大模型请求受 Prompt 与生成长度影响，单次计算与显存占用相差极大。

#### 显存开销的量化测算

在自回归解码过程中，每个并发请求都需要持续持有对应的注意力键值缓存（KV Cache）。单请求的 KV Cache 显存占用计算公式如下：

$$\text{KV Cache 显存大小} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times (L_{in} + L_{out}) \times b$$

其中：
* 常数 $2$ 代表 Key 和 Value 两个张量；
* $n_{\text{layers}}$ 为模型层数；
* $n_{\text{kv\_heads}}$ 为键值注意力头数（对于多查询注意力 MQA，该值为 1；对于分组查询注意力 GQA，该值为键值头组数；对于标准多头注意力 MHA，等于查询头数 $n_{\text{heads}}$）；
* $d_{\text{head}}$ 为每个注意力头的维度；
* $L_{in} + L_{out}$ 为请求的总序列长度；
* $b$ 为数据类型的单参数字节数（如 FP16/BF16 为 2 字节，FP8 为 1 字节）。关于显存构成的详细拆解，可参考 [如何估算一个大模型部署需要多少显存？](/interview/model-deployment-interview/memory-estimation.md)。

#### 算例对比

假设网关简单限制固定并发为 10 QPS：

* **场景 A（短问答）**：10 个请求平均 $L_{in} = 64$，$L_{out} = 64$。每个请求占用的 KV Cache 极小，GPU 显存占用平稳，算力主要用于批处理矩阵乘法。
* **场景 B（长文档分析）**：10 个请求平均 $L_{in} = 16,384$，$L_{out} = 2,048$。以常见 70B 级别模型（GQA，$n_{\text{kv\_heads}}=8$，$d_{\text{head}}=128$，$n_{\text{layers}}=80$，FP16 存储）为例，单请求仅 KV Cache 就需要：
  $$2 \times 80 \times 8 \times 128 \times (16,384 + 2,048) \times 2 \approx 6.03\text{ GB}$$
  10 个并发请求瞬时锁定的显存超过 60 GB，加上模型权重占用的显存，足以直接耗尽整张甚至整机 GPU 的 PagedAttention 显存池。

此时即使严格维持在 10 QPS，推理引擎内部也会因显存不足触发重计算或请求驱逐，导致推理延迟成倍上升。因此，**脱离 Token 长度的 QPS 控制在大模型服务中无法有效防范过载**。

---

<ReadMoreLock />

### Token 级限流算法：RPM 与 TPM 双重控制

网关通常在 Redis 层面结合令牌桶算法，对各调用方（租户）实行双轨制配额管理：

1. **RPM 桶**：限制单位时间内的请求总次数，防止恶意爬虫或高频轮询发起大量空转调用。
2. **TPM 桶**：限制单位时间内消耗的总 Token 数（$\text{Token}_{\text{prompt}} + \text{Token}_{\text{completion}}$），约束实际计算与显存占用。

![Token级限流预扣与结算校准流程](/images/content-rebuild/21fc98d9f5ac2f88/fig-2.png)

上图展示了 TPM 令牌桶的双阶段结算生命周期：请求到达时基于 Prompt 和 max_tokens 扣除预估额度，推理流结束时根据实际 Usage 归还未使用的额度。

#### 预扣与事后校准（Reserve & Reconcile）机制

TPM 控制的难点在于：**请求刚到达网关时，无法提前确知其实际生成的 Token 数量**。若仅扣除已知的 Prompt Token，一旦客户端生成长文本，将导致整体 TPM 严重超额穿透；若直接按客户端声明的 `max_tokens`（例如默认 4096）全额扣除，绝大多数短回复请求就会因配额虚假耗尽而被误拦截。

可落地的两阶段解决方案是**预扣与事后校准**：

1. **入站预扣（Reservation）**：
   * 计算请求输入的 Prompt Token 数量 $T_{\text{prompt}}$（通过网关侧的分词器或字符串长度安全比值估算）。
   * 结合用户指定的 `max_tokens` 与历史经验期望值（如 $E_{\text{completion}} = \min(\text{max\_tokens}, \text{P90 历史输出长度})$），计算预扣总量：
     $$T_{\text{reserve}} = T_{\text{prompt}} + E_{\text{completion}}$$
   * 尝试从 Redis TPM 令牌桶中扣除 $T_{\text{reserve}}$。若桶内余额不足，则拒绝请求并返回 `HTTP 429 Too Many Requests`。
2. **离站校准（Reconciliation）**：
   * 推理引擎完成生成，并在响应的最终 Chunk 或 HTTP 尾部返回实际消耗值 `usage`（协议格式参见 [OpenAI-Compatible API 是什么？](/interview/model-deployment-interview/openai-compatible-api.md)）。
   * 网关计算差额：$\Delta T = T_{\text{reserve}} - (T_{\text{prompt}} + T_{\text{actual\_completion}})$。
   * 将 $\Delta T$ 多扣除的部分原子性归还给该租户的 Redis 令牌桶；若实际输出超过预估值，则补充扣减。若请求在生成中途断开或异常超时，已完成的 Prefill 算力与已输出 Token 已对集群造成实际物理消耗，网关需按已确认消耗进行结算，仅退还未执行的差额，不能无条件全额释放已消耗配额。

```python
# 网关侧 Token 预扣与校准处理逻辑示意
def handle_llm_request(request, tenant_id):
    prompt_tokens = count_tokens_fast(request.prompt)
    # 预扣额度：取 max_tokens 与安全经验上限中的较小值
    est_completion = min(request.max_tokens or 2048, 512)
    reserve_tokens = prompt_tokens + est_completion

    # 1. 尝试原子扣除（RPM 与 TPM 联动）
    allowed = redis_token_bucket_acquire(tenant_id, req_count=1, token_count=reserve_tokens)
    if not allowed:
        retry_after = calculate_retry_wait_seconds(tenant_id, reserve_tokens)
        return Response("Rate limit exceeded (TPM/RPM)", status=429, headers={"Retry-After": str(retry_after)})

    try:
        # 2. 转发请求至推理后端（支持流式与非流式）
        response = forward_to_backend(request)
        actual_completion_tokens = response.usage.completion_tokens
    except ConnectionError:
        # 尚未成功连接后端或后端未执行前建连失败，全额退还预扣
        redis_token_bucket_refund(tenant_id, reserve_tokens)
        raise
    except Exception as e:
        # 流式传输中断或中途超时已消耗 Prefill 算力与部分生成，不可全额释放已消耗配额
        streamed_tokens = get_streamed_tokens_before_abort(e)
        unused_tokens = est_completion - streamed_tokens
        if unused_tokens > 0:
            redis_token_bucket_refund(tenant_id, unused_tokens)
        raise e

    # 3. 结算校准：退还未消耗的预估配额
    unused_tokens = est_completion - actual_completion_tokens
    if unused_tokens != 0:
        redis_token_bucket_refund(tenant_id, unused_tokens)

    return response
```

---

### 推理引擎级背压机制：基于 KV Cache 水位与等待队列

租户限流只能约束配额，无法完全反映物理集群的瞬时负载。多租户正常流量叠加长 Prompt 突发时，后端可能在未超出网关限流阈值的情况下显存告急。此时必须建立推理引擎到网关的**自适应背压（Backpressure）机制**。

![基于KV Cache水位的网关自适应背压保护](/images/content-rebuild/21fc98d9f5ac2f88/fig-3.png)

上图说明了网关如何读取推理集群暴露的 Prometheus 实时指标，根据 KV Cache 块占用率与等待队列深度动态切换放行、缓冲或阻断策略。

#### 核心监控指标

以开源推理框架 vLLM 为例，其暴露的监控指标主要包含：

* `vllm:num_requests_waiting`：调度器就绪队列中处于等待（Waiting）状态的请求数量。
* `vllm:num_requests_running`：当前正在执行 Prefill 或 Decode 批处理的请求数量。
* `vllm:gpu_cache_usage_factor`：当前已分配的 KV Cache Block 占总池化显存块的比例（0.0 ~ 1.0）。

#### 分级背压策略

网关负载均衡器通过后台拉取或 Sidecar 方式获取推理节点的指标，实施阶梯防护：

1. **安全水位（Cache Usage < 75% 且 Waiting Requests = 0）**：
   节点负载正常，网关按照轮询或最小连接数正常分发请求。
2. **预警水位（75% ≤ Cache Usage < 90% 或 Waiting Requests > 0）**：
   节点 Prefill 算力或显存进入拥挤状态。网关停止向该节点路由超长上下文请求（如 $L_{in} > 4k$），仅分发短问答，将长请求重定向至相对空闲的副本。
3. **危险水位（Cache Usage ≥ 90%）**：
   显存块即将耗尽。若继续灌入新请求，连续批处理引擎将启动请求抢占（Preemption）——将部分 Decode 阶段的上下文释放或换出到系统内存，造成请求延迟急剧飙升（连续批处理机制可参考 [连续批处理 Continuous Batching 是什么？](/interview/model-deployment-interview/continuous-batching.md)）。
   此时网关对该节点触发**主动退避**，对超出承载能力的流量快速失败，返回 `HTTP 503` 或带有重试建议的 `HTTP 429`，响应头携带 `Retry-After: 3`，促使客户端指数退避。

---

### 超时与熔断机制：断连中止与显存保护

大模型推理服务的流式传输通常持续数秒到数十秒，传统的单一全局超时设置无法有效隔离故障，必须根据其生命周期分阶段治理。

![客户端断连触发推理引擎主动中止流程](/images/content-rebuild/21fc98d9f5ac2f88/fig-4.png)

上图展示了当用户关闭网页或客户端网络断开时，网关通过 Socket 监听向后端发出 Abort 信号并回收 GPU 显存的闭环机制。

#### 客户端断开主动中止（Cancel on Client Disconnect）

在生成式交互中，用户经常在看到前几个 Token 后发现不需要完整回答，从而点击“停止生成”或直接关闭页面。如果网关未处理该事件：

* 客户端连接虽已断开，但后端推理引擎仍在持续执行自回归解码，直至达到 `max_tokens` 或生成终止符。
* 这些无用计算会长时间霸占 GPU 计算单元与 KV Cache 块，造成“幽灵解码”。

**工程实现要求**：网关必须监听上游 HTTP 客户端的断开事件（例如在 FastAPI/Starlette 中监听 `request.is_disconnected()`，在 Go 网关中监听 `r.Context().Done()`）。一旦检测到连接关闭，立刻向对应的推理引擎节点发起取消 RPC 调用（如 vLLM 的 `/abort` 接口）。推理引擎收到信号后，在下一个解码步退出循环，即刻释放对应的显存块。

#### 超时设置的三维拆解

针对长任务特性，将单点超时拆分为三个阶梯：

1. **排队等待超时（Queue Timeout，如 5~15s）**：请求在就绪队列中等待调度的上限。若超时仍未进入 Prefill，说明算力严重积压，直接返回快速失败，避免无谓等待。
2. **首 Token 延迟超时（TTFT Timeout，如 10~30s）**：从进入引擎计算到 Prefill 结束吐出第一个 Token 的耗时。可针对超长 Prompt 设置差异化阈值。
3. **Token 间生成超时（Inter-Token Timeout，如 2~5s）**：解码阶段相邻两个 Token 产生的最大间隔。若超限，通常意味着节点进程死锁、硬件掉卡或网络挂起，应快速切断并介入。

#### 推理服务 Pod 熔断设计

传统的微服务熔断器（如按 50% 错误率熔断）在大模型场景需要调整触发条件。大模型集群中最危险的单点异常是 **CUDA Out Of Memory**。

![推理服务Pod级熔断器状态机](/images/content-rebuild/21fc98d9f5ac2f88/fig-5.png)

上图展示了专为推理集群定制的熔断器状态机。当某 Pod 出现 OOM 或硬件报错时，熔断器立即切断流量，执行隔离并触发 Pod 重建。

* **状态转换说明**：
  * **Closed（正常运行）**：流量正常注入。若返回常规业务参数错误（如 400）或单次被引擎调度器正常捕获并拒绝的超长请求，仅计入常规统计；若出现节点连续 OOM、显存严重碎片化或捕获到底层不可恢复的 CUDA 运行时异常，熔断器判定节点进入不可用状态。
  * **Open（熔断隔离）**：切断流向该 Pod 的全部新流量，将请求路由至健康副本。对于显存受损或抛出不可恢复故障的实例，通知容器编排系统（如 Kubernetes）执行 Pod 重建与模型重新加载，以彻底重置 PyTorch 显存分配器状态。
  * **Half-Open（探活验证）**：Pod 启动并完成权重与显存池初始化后，仅放行小批量、固定长度的测试探针请求（如 128 Token 预热请求），连续校验通过后方才恢复 Closed 状态。

---

### 多级降级策略

在面临极端脉冲流量或集群节点硬件损坏时，网关需要执行有序的多级降级，保障核心功能可用。

![大模型网关多级动态降级漏斗](/images/content-rebuild/21fc98d9f5ac2f88/fig-6.png)

上图描绘了从全功能高算力消耗，到低计算负荷，直至最终阻断的三级防护漏斗。

#### 一级降级：动态缩减非关键上下文与生成长度

在不改动服务模型的前提下，网关主动削减请求负荷：

* **RAG 召回剪裁**：在检索增强生成场景中，将注入 Prompt 的上下文文档块数量从默认的 5 个削减为 2 个，大幅降低 Prefill 阶段的计算量与显存占用（相关切分技巧参考 [文档切分 Chunk 应该怎么做？](/llm-100/rag/chunk-strategy.md)）。
* **收紧 `max_tokens`**：将客户端未明确指定或设置过大的 `max_tokens` 统一压紧至 256 或 512，缩短解码占用显存的停留时长。

#### 二级降级：路由至轻量级模型或量化实例

准备好不同计算开销的模型副本作为梯队：

* **参数降维路由**：主力大参数模型（如 70B）过载时，网关将总结、分类等非强推理任务透明分流至同系列的小参数模型（如 8B）。
* **量化集群接管**：将流量重定向至低精度量化（如 FP8、INT4 AWQ）模型实例。量化模型显存占用更小，单卡可承载更大 Batch（量化机制详见 [模型量化是什么？](/interview/model-deployment-interview/quantization.md)）。

#### 三级降级：静态排队大厅与优雅退避

当后备计算资源全部饱和时，网关启用最后一道防线：

* 返回标准的 `HTTP 429` 响应与结构化错误体：
  ```json
  {
    "error": {
      "message": "Cluster is operating at peak capacity. Please retry.",
      "type": "overloaded_error",
      "retry_after_seconds": 5
    }
  }
  ```
* 客户端结合 `Retry-After` 头实行带抖动的指数退避（Exponential Backoff with Jitter），防止大量客户端同步重试引发二次冲击。

---

### 面试追问与常见认知误区

#### 考官高频追问

* **追问 1：如果一个长 Prompt 请求通过了 TPM 限流检查，但是在生成阶段吐出了远超预期的 Token，把显存打满了怎么办？**
  * *解答思路*：入站限流只能基于预估控制流量，无法绝对杜绝物理显存溢出，因此必须依靠**引擎内部背压配合**。当引擎的 KV Cache 水位触碰危险线（如 90%~95%）时，引擎调度器会暂停接收新批次并可能对部分请求执行抢占；网关侧感知到水位告警后，应立刻对超长请求实行降级拦截，并按比例收缩后续请求的并发配额，返回带 `Retry-After` 的 429 响应。
* **追问 2：在高并发分布式网关下，每次流式返回一个 Token 都去 Redis 校准，会不会把 Redis 打挂？**
  * *解答思路*：流式生成过程中不应逐 Token 同步刷写 Redis。标准做法是**“入站原子预扣一次，流式期间本地累计，连接结束时执行单次校准退还”**。对于超长流式请求，可采用本地内存滑动窗口，只有累积偏差超过一定阈值（如 500 Token）才异步触发一次批量同步。

#### 常见认知误区

* **误区 1：“大模型限流直接配并发连接数即可，比如限 100 并发。”**
  * *澄清*：100 个 100 Token 的短请求与 100 个 16k Token 的长请求，显存开销相差数十倍。只控并发极易在长文本流量下发生显存穿透。
* **误区 2：“客户端断开连接后，推理引擎自然就会停止。”**
  * *澄清*：HTTP 客户端断开连接后，反向代理若未开启连接断开感知并向推理引擎下发 Abort 指令，模型引擎通常会继续自回归计算直到结束，导致算力与显存的严重空耗。
* **误区 3：“为了防止大模型超时，把网关超时设为 10 分钟即可。”**
  * *澄清*：笼统放长超时会导致僵死请求长期占据连接池与引擎队列。必须将超时拆分为排队超时、首 Token 延迟超时与生成步长超时，兼顾长序列生成与异常快速熔断。
