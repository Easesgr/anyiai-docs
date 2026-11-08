---
title: 大模型耗时请求如何做高并发？异步任务与排队系统设计
icon: mdi:clock-outline
description: 大模型推理耗时长，同步调用极易耗尽网关连接池导致雪崩。本文讲解任务提交、队列解耦、双维度令牌桶背压、状态机租约与结果投递的工程落地方案。
---

在开发基于大语言模型（LLM）的业务系统时，工程团队常面临一个物理矛盾：**客户端希望接口在数百毫秒内响应，而大模型执行复杂生成任务通常需要数秒乃至数十秒**。

以一个典型场景为例：用户在后台提交一份 1 万字的财报文件，要求系统调用大模型生成结构化财务指标分析与合规摘要。大模型读取长上下文并输出上千字内容，单次推理往往耗时 15 秒以上。若沿用传统的同步 HTTP 接口，突发流量到来时，应用服务器的线程池和连接会在短时间内被慢请求占满，导致整个服务集群雪崩。

本文承接 [AI 系统设计](/architecture/ai-system-design.md) 与 [模型网关](/architecture/model-gateway.md)，专门解决接口接入层的并发削峰、解耦排队与异步交付问题。分钟至小时级的多步骤长工作流与断点恢复，将在 [长任务执行](/architecture/long-task-execution.md) 中展开讨论。

![高并发异步任务与排队系统全局架构概览](/images/content-rebuild/8f3437ed1410a4e2/cover.png)

---

### 同步阻塞调用的崩溃边界：连接池耗尽与级联雪崩

在同步模式下，客户端发起 HTTP POST 请求后，底层的 TCP 连接和应用服务器的工作线程将保持挂起，直到大模型完成全部推理并将响应写回网络套接字（Socket）。这种模式在低频访问时逻辑简单，但在高并发场景下容错空间极小。

根据排队论中的利特尔法则（Little's Law），在一个稳态系统中，平均并发驻留的请求数 $L$ 等于系统平均请求到达率 $\lambda$ 与单次请求平均处理时长 $W$ 的乘积：

$$L = \lambda \times W$$

假设某个后端服务集群配置了固定的连接池，最大可用工作线程数（或并发连接上限）为 $N = 200$。在前述财报分析场景中，大模型平均处理耗时 $W = 15\text{ s}$。系统在不积压请求的前提下，能够承受的最大吞吐量为：

$$\lambda_{\max} = \frac{N}{W} = \frac{200}{15} \approx 13.33\text{ req/s}$$

如果上游业务突然涌入 $50\text{ req/s}$ 的并发请求，超出系统稳态承载力近 4 倍。此时系统会沿着以下路径逐步瘫痪：

1. **工作线程耗尽**：在不到 5 秒的时间内，200 个处理线程全部被未完成的长推理请求占满。
2. **网关挂起队列溢出**：Nginx、ALB 等接入层网关无法将新连接分发给后端，客户端等待超时，反向代理开始向客户端返回 `504 Gateway Timeout` 或 `502 Bad Gateway`。
3. **级联重试风暴**：上游系统或前端在收到 504 错误后，若配置了立即重试策略，会向本已超载的集群注入更多请求，延缓系统恢复。
4. **容器健康检查失败**：由于所有工作线程都在等待套接字读写，连用于存活探测的 `/healthz` 接口也无法及时响应，导致容器编排平台（如 Kubernetes）判定实例异常并强制重启 Pod，引发大面积服务雪崩。

![同步阻塞模式下的连接池耗尽与级联雪崩机制](/images/content-rebuild/8f3437ed1410a4e2/fig-1.png)

---

### 异步任务架构模型：入队、消费与解耦

避免连接池耗尽的核心手段是将**任务接收**与**任务执行**在网络连接层面彻底解耦。异步任务架构将用户的单次调用拆分为三个阶段：提交任务获得凭证、后台异步计算、通知或拉取结果。

![异步任务三层拓扑与数据流解耦模型](/images/content-rebuild/8f3437ed1410a4e2/fig-2.png)

系统由以下三个核心组件构成：

1. **API 接入层（Producer）**：负责请求参数校验、用户鉴权以及前置配额核验（可结合 [Token 成本控制](/architecture/token-cost.md)）。校验通过后，API 网关生成全局唯一的任务标识 `task_id`，将任务写入消息队列，并**立即**向客户端返回 HTTP `202 Accepted`，响应体中包含 `task_id` 与轮询链接。网关处理时间通常在几十毫秒以内，不等待大模型推理。
2. **解耦缓冲层（Message Broker）**：使用 Redis Stream、RabbitMQ 或 Apache Kafka 等分布式消息中间件暂存任务。队列起到水库蓄水的作用，将瞬时突发的尖峰流量平摊为平缓的任务流。
3. **工作节点池（Worker Pool）**：由无状态的 Worker 计算节点组成，支持根据队列积压深度独立扩缩容。Worker 按照预设的消费速率从队列拉取任务，调用大模型接口，并将结果与统计数据写入持久化存储（如 PostgreSQL 或 Redis）。

<ReadMoreLock />

---

### 并发控制与削峰：基于供应商指标的主动背压

将请求放入队列后，Worker 节点并不能无限制地全速并发调用下游大模型。主流模型服务商均会对 API 账号设置多维度的速率限制（Rate Limits）：

* **RPM (Requests Per Minute)**：每分钟调用请求次数上限。
* **TPM (Tokens Per Minute)**：每分钟处理的输入与输出 Token 总数上限。
* **并发连接数限制**：同时处于处理中的连接上限。

如果 Worker 集群盲目并发，极易触发下游的 HTTP `429 Too Many Requests`，造成重试积压甚至配额惩罚。因此，系统必须在 Worker 消费端实现**主动背压控制（Dynamic Backpressure）**。

![基于双维度令牌桶与动态背压的流量调谐机制](/images/content-rebuild/8f3437ed1410a4e2/fig-3.png)

#### 集中式双维度令牌桶（RPM + TPM）

由于不同任务的输入长度差异极大（有的几百 Token，有的数万 Token），单靠限制请求数（RPM）依然会导致 Token 数（TPM）超限。系统可在 Redis 中通过 Lua 脚本原子核销“请求次数令牌”与“Token 数量令牌”：

```lua
-- Redis Lua: 原子核销 RPM 与预估 TPM 令牌
-- KEYS[1]: rpm_bucket_key, KEYS[2]: tpm_bucket_key
-- ARGV[1]: 当前时间戳, ARGV[2]: 申请的预估 token 数 (Prompt + max_tokens)

local rpm_limit = tonumber(redis.call('HGET', KEYS[1], 'limit'))
local rpm_tokens = tonumber(redis.call('HGET', KEYS[1], 'tokens'))
local tpm_limit = tonumber(redis.call('HGET', KEYS[2], 'limit'))
local tpm_tokens = tonumber(redis.call('HGET', KEYS[2], 'tokens'))
local requested_tokens = tonumber(ARGV[2])

-- 令牌刷新逻辑（根据上次更新时间与产出速率计算新令牌，总数不超过 limit 上限）
-- 此处省略按时间平滑注水代码...

if rpm_tokens >= 1 and tpm_tokens >= requested_tokens then
    redis.call('HSET', KEYS[1], 'tokens', rpm_tokens - 1)
    redis.call('HSET', KEYS[2], 'tokens', tpm_tokens - requested_tokens)
    return 1 -- 允许消费
else
    return 0 -- 令牌不足，触发背压
end
```

#### 背压与额度补偿流程

1. **前置预扣减**：Worker 从队列拿到任务后，先解析 Prompt 长度并加上模型参数 `max_tokens` 作为预估上限。在正式发起网络请求前，先向集中限流器申请这笔配额。
2. **排队等待与退避**：若令牌不足，Worker 暂不消费，或者将任务放回重试队列稍后拉取，避免向下游盲目发送必然失败的请求。
3. **差额补偿校准**：模型调用结束后，Worker 从返回的 `usage` 字段读取实际消耗的 `total_tokens`。若预估扣除 4,000 Token，实际仅使用 1,500 Token，Worker 需将未使用的 2,500 Token 补偿回令牌桶（补偿后总量不超过桶容量上限），保证下游吞吐额度被充分利用。

---

### 任务生命周期状态机与租约机制

为了保证异步执行过程可观测，且在节点异常时不丢失任务，任务状态必须由严格的状态机管理。

![任务生命周期状态机演进与心跳租约流转](/images/content-rebuild/8f3437ed1410a4e2/fig-4.png)

#### 状态流转规则

* **`PENDING`（排队中）**：任务已由网关写入消息队列并保存元数据，等待 Worker 领取。允许转移至：`RUNNING`、`CANCELLED`。
* **`RUNNING`（执行中）**：Worker 已成功拉取任务并持有执行锁，正在调用模型。允许转移至：`SUCCEEDED`、`FAILED`、`CANCELLED`、`PENDING`（租约超时重入队）。
* **`SUCCEEDED`（成功）**：大模型调用完成，输出结果已写入持久化存储。此为终态，不可跃迁。
* **`FAILED`（失败）**：遇到不可重试错误或重试次数达上限。此为终态，不可跃迁。
* **`CANCELLED`（已取消）**：用户主动取消任务。此为终态，不可跃迁。

#### 分布式租约与僵尸任务回收

在异步架构中，Worker 节点可能因内存溢出（OOM）、宿主机硬件故障或网络分区突然宕机。若 Worker 在执行过程中死亡，被置为 `RUNNING` 的任务就会成为无人处理的“僵尸任务”。

系统通常采用**分布式租约机制**进行防护：

1. Worker 开始执行任务时，获取一个带有过期时间（如 30 秒）的任务租约，在数据库或缓存中设置 `lease_expire_at`。
2. 任务执行期间，Worker 内部的后台协程每隔固定时间（如 10 秒）向状态存储刷新心跳，延长 `lease_expire_at`。
3. 系统运行一个独立的巡检进程（Reaper），周期性扫描状态为 `RUNNING` 且 `lease_expire_at` 早于当前时间的任务。一旦发现租约过期，巡检进程判定原 Worker 已经异常断开，根据已执行次数决定将任务重新放回 `PENDING` 队列，还是标记为 `FAILED`。

---

### 结果交付机制：轮询、SSE 与 Webhook 的选型与保障

当后台生成完毕后，系统需要将结果交还给发起方。常见的交付方案如下：

| 交付模式 | 典型适用场景 | 架构开销 | 优缺点对比 |
| :--- | :--- | :--- | :--- |
| **短轮询 / 长轮询** | 普通 Web 页面、无独立公网 IP 的客户端、简单控制台 | 中至高（轮询请求数随等待时间增长，长轮询会占用接入层连接） | **优点**：能够穿透防火墙与 NAT，客户端实现门槛低。<br>**缺点**：大部分请求仅用来确认任务未完成；长轮询在连接超时时需反复建立连接。 |
| **WebSocket / SSE 订阅** | 人机交互对话界面、需要打字机逐字输出的场景 | 中（需维持长连接上下文与心跳） | **优点**：支持服务端主动推送，流式生成体验平滑。<br>**缺点**：移动端网络切换容易断连；大量长连接需要专属的连接网关与广播分发层。 |
| **Webhook 回调** | 系统间调用（B2B）、微服务协作、耗时极长的批处理流水线 | 低（事件触发单次 POST，无常驻长连接） | **优点**：完全异步解耦，调用方无需挂起等待。<br>**缺点**：调用方必须提供可公网访问的接口；需要设计重试与验签机制防伪造。 |

![Webhook 交付安全签名与退避重试时序机制](/images/content-rebuild/8f3437ed1410a4e2/fig-5.png)

#### Webhook 可靠投递设计

如果选用 Webhook 进行系统间通知，必须在协议设计中落实两项工程防护：

##### 1. 基于 HMAC-SHA256 的签名防伪与防重放

不能向外部 URL 发送无验证的明文通知。为了防止中间人伪造请求，每次 Webhook POST 必须在 HTTP 标头中附带时间戳和数字签名：

$$\text{Signature} = \text{HMAC-SHA256}\Big(K_{\text{secret}},\; \text{timestamp} \parallel "." \parallel \text{Payload}\Big)$$

接收方在验证签名时：
* 计算接收时间与标头中 `timestamp` 的差值。若超过容忍窗口（如 300 秒），直接丢弃，以防**重放攻击**。
* 使用双方约定的共享私钥 $K_{\text{secret}}$ 对时间戳和请求体计算签名，并使用**恒定时间比较**（Constant-Time Compare）对比收到与计算的签名，防范**时序攻击**（Timing Attack）。

##### 2. 指数退避与随机抖动（Jitter）

接收方的服务器可能遭遇重启或短暂网络波动。Webhook 发送模块若单次失败就放弃，会导致任务结果丢失。系统应将通知任务放入专门的延时队列，执行带随机抖动的指数退避重试：

$$t_{\text{retry}} = \min\Big(t_{\max},\; t_{\text{base}} \times 2^{\text{attempt}}\Big) \pm \text{Jitter}$$

引入随机抖动（Jitter）的目的是防止同一时刻失败的成百上千个回调请求在同一重试时间点同时爆发，将调用方的接收服务再度冲垮。

---

### 错误分流与死信队列（DLQ）

大模型系统调用的失败原因多样。如果不对错误进行分类，盲目重试不仅白白消耗配额，还会导致整个队列停滞。

![错误类型分类决策树与死信队列隔离流程](/images/content-rebuild/8f3437ed1410a4e2/fig-6.png)

#### 错误属性分类

1. **瞬态错误（Transient Errors）**：
   * 包括服务商 503 临时不可用、网络连接超时（Connect Timeout）、偶发的 429 配额限制、网络重置等。
   * **处置策略**：将任务放入延时重试队列，在退避等待后重新消费，设置最大重试上限（如 3 次）。
2. **不可恢复错误（Permanent Errors）**：
   * 包括请求参数格式非法（400 Bad Request）、API Key 失效（401 Unauthorized）、触发模型供应商安全风控拒绝（Content Filter Violation）、超出模型上下文物理上限等。
   * **处置策略**：不重试，直接将任务标记为 `FAILED`，记录返回的错误明细，如实暴露给调用方。

#### 死信队列（DLQ）与毒丸载荷隔离

所谓“毒丸（Poison Pill）”，是指某些输入载荷存在特殊字符或极端情况，会触发 Worker 内部未捕获的严重异常甚至导致进程崩溃退出。当 Worker 崩溃后，租约巡检器将该任务重新放回队列，接着被下一个 Worker 领取，从而引发连锁崩溃。

死信队列（Dead Letter Queue）用来阻断该死循环：

* 任务载荷中包含 `delivery_count` 字段。
* 当 `delivery_count` 超过系统预设阈值（例如 3 次）时，系统不再将该任务重新投递到消费队列，而是原子写入死信队列并向主队列返回 ACK。
* 系统保存崩溃时的输入数据摘要与异常调用栈（Stack Trace），同时触发可观测性告警（参考 [日志与 Trace](/architecture/logging-trace.md)），由开发人员排查修复。

---

### 消费端核心逻辑参考实现

以下 Python 代码演示了 Worker 消费任务时的主要执行路径，包含双维度令牌申请、租约维护以及错误分流：

```python
import json
import time
import hmac
import hashlib
import requests
from typing import Dict, Any, Optional

class AsyncLLMWorker:
    def __init__(self, broker_client, state_store, rate_limiter, dlq_client):
        self.broker = broker_client
        self.store = state_store
        self.rate_limiter = rate_limiter
        self.dlq = dlq_client
        self.max_retries = 3

    def process_next_task(self):
        task = self.broker.pop_task()
        if not task:
            return

        task_id = task["task_id"]
        estimated_tokens = task["prompt_tokens"] + task.get("max_tokens", 1000)

        # 1. 背压检查：预核销 RPM 与预估 TPM 配额
        while not self.rate_limiter.acquire(rpm=1, tpm=estimated_tokens):
            # 令牌不足时等待，避免盲目请求导致下游 429
            time.sleep(0.5)

        # 2. 状态变更为 RUNNING 并设置 30 秒租约
        self.store.update_status(task_id, "RUNNING", lease_ttl=30)

        try:
            # 3. 调用下游模型接口（伪代码示意真实调用）
            response = self._call_upstream_llm(task["payload"])
            
            # 4. 实际用量结算，将未使用的配额补回令牌桶（受桶容量上限限制）
            actual_tokens = response["usage"]["total_tokens"]
            unused_tokens = max(0, estimated_tokens - actual_tokens)
            self.rate_limiter.compensate_unused(unused_tokens)
            
            # 5. 存储成功结果并触发 Webhook 回调
            self.store.save_result(task_id, status="SUCCEEDED", output=response["output"])
            self._dispatch_webhook(task.get("webhook_url"), task_id, response["output"])
            self.broker.ack(task)

        except UpstreamRateLimitError as e:
            # 瞬态错误：进入延迟重试
            self._handle_retry(task, is_transient=True, error=e)
        except ContentFilterViolationError as e:
            # 不可恢复错误：不重试，直接标记失败
            self.store.save_result(task_id, status="FAILED", error=str(e))
            self.broker.ack(task)
        except Exception as e:
            # 未预期异常，走死信计数检查
            self._handle_retry(task, is_transient=False, error=e)

    def _handle_retry(self, task: Dict[str, Any], is_transient: bool, error: Exception):
        task["retries"] = task.get("retries", 0) + 1
        if is_transient and task["retries"] <= self.max_retries:
            # 指数退避
            backoff_sec = min(60, 2 ** task["retries"])
            self.broker.requeue_delayed(task, delay=backoff_sec)
            self.store.update_status(task["task_id"], "PENDING")
        else:
            # 超过重试次数或无法恢复的毒丸，移入死信队列
            self.dlq.push(task, error=str(error))
            self.store.save_result(task["task_id"], status="FAILED", error="Exceeded retries or poison pill")
        self.broker.ack(task)

    def _dispatch_webhook(self, url: Optional[str], task_id: str, output: Any):
        if not url:
            return
        timestamp = int(time.time())
        payload_dict = {"task_id": task_id, "output": output}
        # 使用有序且标准的 JSON 序列化保证验签一致性
        payload_bytes = json.dumps(payload_dict, sort_keys=True).encode("utf-8")
        
        # 严格对齐正文规范：将 timestamp 与 payload 拼接后进行 HMAC 计算
        sign_data = f"{timestamp}.".encode("utf-8") + payload_bytes
        secret = b"webhook_signing_secret"
        signature = hmac.new(secret, sign_data, hashlib.sha256).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "X-Signature": f"sha256={signature}",
            "X-Timestamp": str(timestamp)
        }
        try:
            # 在实际生产中，应将 Webhook 投递给专用通知队列，避免 Worker 线程因网络超时挂起
            requests.post(url, data=payload_bytes, headers=headers, timeout=5)
        except requests.RequestException:
            pass
```
