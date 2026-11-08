---
title: 什么是模型网关？如何统一管理多供应商接口、凭证与流量？
icon: mdi:lan
description: 探讨模型网关的核心职责与实现架构，涵盖统一API协议适配、API Key凭证池管理、RPM/TPM双维度预占限流、断路器熔断与TTFT/TPOT可观测指标。
---

在企业应用中直接让各个微服务连接第三方大模型 API，往往会遇到以下工程问题：

以一个典型的电商平台为例，系统包含**在线客服机器人**、**售后工单智能分析**与**运营内容生成**三个业务微服务。在线客服需要调用低延迟的内部私有模型（如部署在 vLLM 上的开源模型），工单分析需要长上下文模型（如 OpenAI GPT-4o），运营生成则接入擅长结构化写作的模型（如 Anthropic Claude 3.5 Sonnet）。

如果各微服务自行直连供应商，很快会面临一系列风险：
1. **接口协议各异**：OpenAI 的入参将系统提示词作为列表元素 `messages=[{"role": "system", ...}]`，而 Anthropic 要求将 `system` 作为顶层独立参数；各厂商的流式传输（SSE）事件字段和结束标记也互不兼容。
2. **凭证分散失控**：十几组真实 API Key 配置在不同服务的环境变量中，缺乏统一的配额监控与权限隔离，轮换凭证需通知多个业务方逐一发布。
3. **缺乏集中流控与防雪崩机制**：早高峰突发流量容易将某供应商账号打至 HTTP 429（超额限流）；一旦供应商偶发 500 宕机，业务端若发起无限重试，将耗尽容器连接池，引发服务级联故障。
4. **调用耗时定位模糊**：业务端只感知到请求耗时数秒甚至数十秒，无法区分是排队等待久、上游首包生成慢，还是解码输出耗时长。

**模型网关（Model Gateway）**正是处于内部微服务与外部模型供应商之间的反向代理与管控层。它的核心任务是：**向下屏蔽多家供应商的接口差异，向上提供统一的调用契约，集中管理物理凭证、实施请求限流、提供故障兜底并暴露统一的可观测指标**。

在架构分工上，模型网关专注于协议层转换、传输代理、限流与可用性防护；更上层的业务语义分流与模型选型逻辑由 [多模型路由](/architecture/multi-model-routing.md) 负责，整体系统蓝图请参考 [AI 系统设计](/architecture/ai-system-design.md)。

![模型网关在客户端与多供应商之间的边界管控定位](/images/content-rebuild/cba0a54969a72542/cover.png)

---

### 1. 模型网关的核心职责与边界划分

作为大模型调用的网络收敛点，模型网关承担四个核心维度的职责：

1. **协议规范化（Protocol Normalization）**：向上暴露业界事实标准（通常为 OpenAI 兼容的 REST 与 SSE 规范），向下对接 OpenAI、Anthropic、Google Gemini 以及自建 vLLM 等异构后端。业务开发者只需面向一套标准参数与响应体编程。
2. **凭证隔离与安全边界（Credential Isolation）**：业务方仅持有网关颁发的虚拟租户令牌（Virtual Tenant Token）。真实的公网供应商 API Key 由网关集中加密保管，在请求转发时动态注入，阻断物理凭证直接暴露给业务代码的风险。租户权限设计可参考 [安全与权限](/architecture/security-permission.md)。
3. **多维流量整形（Traffic Shaping）**：上游模型调用通常同时受并发数、每分钟请求数（RPM）以及每分钟 Token 数（TPM）三者制约。网关负责对突发流量进行平滑限流与排队整形，防止触发上游配额上限。
4. **故障熔断与出口收敛（Resilience & Network Boundary）**：收敛内网出站流量至专用代理出口；针对上游网络抖动与偶发 429 实施带抖动的指数退避重试；在上游大面积故障时触发断路器熔断，避免业务线程堆积导致系统雪崩。

![模型网关核心分层与网络边界收敛架构图](/images/content-rebuild/cba0a54969a72542/fig-1.png)

---

### 2. 统一 API 契约：请求体与流式响应的标准化映射

网关需要建立双向转换流水线：接收客户端统一格式请求 $\rightarrow$ 映射为目标供应商专有载荷 $\rightarrow$ 发起网络调用 $\rightarrow$ 映射供应商响应回统一格式 $\rightarrow$ 返回业务客户端。

#### 2.1 请求参数字段映射

主流模型接口在参数命名与组织层级上存在明确差异，网关需通过规则矩阵完成适配：

| 通用抽象字段 | OpenAI API 规范 | Anthropic Claude Messages API | Google Gemini API | 网关转换处理规则 |
| :--- | :--- | :--- | :--- | :--- |
| **系统消息** | `messages` 列表中 `role="system"` 的元素 | 顶层独立参数 `system: string` | `systemInstruction.parts[].text` | 提取 `messages` 中的 system 角色内容，拍平转换为目标模型对应的顶层字段。 |
| **模型角色** | `system` / `user` / `assistant` | `user` / `assistant` | `user` / `model` | 映射角色标识：将通用格式中的 `assistant` 映射为 Gemini 的 `model`。 |
| **最大生成长度** | `max_tokens` / `max_completion_tokens` | `max_tokens`（必选） | `generationConfig.maxOutputTokens` | 若客户端未传，网关必须根据所调用的下游模型规格自动注入默认值，防止 Anthropic 报错。 |
| **停止词序列** | `stop`（字符串或数组） | `stop_sequences`（数组） | `generationConfig.stopSequences` | 标量字符串统一标准化为字符串数组。 |
| **工具调用声明** | `tools[].function` | `tools[].input_schema` | `tools[].functionDeclarations` | 对齐 JSON Schema 结构版本，过滤目标厂商不支持的扩展属性。 |

#### 2.2 流式响应（SSE）的标准化转换

在流式交互中，不同厂商的推流事件格式分歧更大。OpenAI 规范每次通过 `choices[0].delta.content` 增量推送文本；Anthropic 则采用事件驱动的状态机流（包括 `content_block_start`、`content_block_delta`、`content_block_stop` 等事件）。网关需要将这些事件流解析后，重新封装为统一的增量块（Delta Chunks）推给业务客户端。关于流式处理的底层设计可进一步阅读 [流式响应](/architecture/streaming.md)。

```python
# 协议适配流水线：将 Anthropic 的 SSE 事件转换为 OpenAI 兼容格式
def transform_anthropic_chunk_to_openai(event_type: str, chunk_data: dict) -> dict | None:
    if event_type == "content_block_delta":
        # 提取文本增量
        delta_text = chunk_data.get("delta", {}).get("text", "")
        return {
            "id": chunk_data.get("id", "chatcmpl-stream"),
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"content": delta_text},
                "finish_reason": None
            }]
        }
    elif event_type == "message_delta":
        # 提取结束原因
        raw_reason = chunk_data.get("delta", {}).get("stop_reason", "stop")
        # 将 end_turn 映射为标准的 stop
        finish_reason = "stop" if raw_reason == "end_turn" else raw_reason
        return {
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }]
        }
    return None
```

<ReadMoreLock />

![请求体与响应体字段映射转换流水线](/images/content-rebuild/cba0a54969a72542/fig-2.png)

---

### 3. API Key 凭证池管理：租户配额、权重轮换与无感知更新

单一供应商的单组 API Key 通常有默认的调用频率上限。网关通过引入**凭证池（Key Pool）**，让业务端与底层物理凭证彻底解耦。

#### 3.1 凭证池映射与加权调度

凭证池采用三层解耦结构：
1. **虚拟租户层（Virtual Tenant）**：业务方（如“客服机器人”或“工单系统”）持有一个虚拟 Token。网关据此校验其预算、租户级并发限制及模型访问白名单。
2. **物理凭证组（Key Group）**：针对同一供应商（如 OpenAI Direct、Azure OpenAI、Anthropic），池化维护多个物理 API Key，每个 Key 标记有独立权重、预设配额阈值和当前的健康状态。
3. **加权轮换与故障移出**：采用平滑加权轮询（Smooth Weighted Round-Robin）或最小并发负载算法调度物理 Key。当某个 Key 频繁触发 HTTP 429 或 401 鉴权失败时，网关立即将其权重置零并移出活动池，进入冷却探测周期。

#### 3.2 运行时无感知热更新

在生产环境中轮换或追加物理 Key 时，正在传输中的流式连接绝不能被强行中断。网关通常采用基于原子指针（Atomic Pointer）或只读快照切换机制：

```go
// Go 内存中凭证池的原子无锁更新示例
type KeyPoolManager struct {
    // activeSnapshot 保存指向当前凭证映射快照的原子指针
    activeSnapshot atomic.Pointer[map[string][]*UpstreamCredential]
}

func (m *KeyPoolManager) ReloadConfig(newCredentials map[string][]*UpstreamCredential) {
    // 构建新的只读数据快照并完成参数校验
    // 原子替换指针，后续请求立即读取新快照
    m.activeSnapshot.Store(&newCredentials)
    // 已经持有着旧快照引用的在途长连接不会受影响，随连接结束由 GC 回收
}
```

![凭证池轮换、配额匹配与热加载机制图](/images/content-rebuild/cba0a54969a72542/fig-3.png)

---

### 4. 双维度流量控制：并发连接数与 RPM/TPM 令牌桶

模型推理服务的流控与传统 Web 接口有显著区别：传统接口通常只看 QPS，而模型推理既受限于并发连接数（显存与网络连接占用），又受限于供应商严格考核的 **RPM（每分钟请求数）** 和 **TPM（每分钟 Token 吞吐量）**。

#### 4.1 TPM 限流的核心难题：预占与后续校准

RPM 限流非常简单，每来一个请求扣除 1 个令牌即可。但 TPM 限流面临一个固有矛盾：**请求发起时，最终消耗的总 Token 数量是未知的**（输入 Token 可以离线计算，但模型会生成多少 Token 尚未可知）。

- **方案 A（完成后扣减）**：请求完全结束后才扣减令牌。如果有突发的长文本生成并发涌入，瞬间产生的几万 Token 会在短时间内打穿供应商限额，导致后续请求大面积遭遇 HTTP 429。
- **方案 B（按客户端声明全量预扣）**：客户端往往会默认声明 `max_tokens = 4096`。若直接扣除 4096，令牌桶会被虚假占用，即使模型实际只回复了 50 个字，后续请求也会被网关误拦截，造成吞吐利用率大幅下滑。

生产环境中标准的解决方案是：**两阶段预估预占与后续校准机制（Pre-reservation & Post-reconciliation）**。

1. **阶段一：输入估算与有限预占**：
   - 输入 Prompt Token 数计算：网关利用本地快速分词库（如 fast-tokenizer）或字符比例（英文约为 1.3~1.4 Token/词，中文约为 1.2~1.8 Token/字）快速得出 $T_{\text{prompt}}$。
   - 输出 Token 经验预占：根据该模型在历史业务监控中的 P90 生成长度，结合客户端的声明，得出经验输出估算：$E = \min(\text{Request.max\_tokens}, \text{Model.P90\_OutputTokens})$。
   - 预占总令牌数：$T_{\text{reserve}} = T_{\text{prompt}} + E$。
   - 尝试从 TPM 令牌桶中原子预扣 $T_{\text{reserve}}$。若余量不足，请求进入等待队列排队或返回网关层 429。
2. **阶段二：上游执行**：
   - 请求成功放行并透传至模型供应商。
3. **阶段三：实际消耗核销（Reconciliation）**：
   - 当流式结束帧或完整响应体返回时，读取上游官方 `usage` 结构，获取实际总消耗 $T_{\text{actual}} = T_{\text{actual\_prompt}} + T_{\text{actual\_completion}}$。
   - 计算预估偏差：$\Delta T = T_{\text{reserve}} - T_{\text{actual}}$。
   - **多退少补**：若 $\Delta T > 0$（预占过多），将差额 $\Delta T$ 立即归还令牌桶；若 $\Delta T < 0$（实际生成超出了预估），则从令牌桶中追扣 $|\Delta T|$（允许短暂透支，由后续周期平抑）。

```python
# 令牌桶预占与校准状态逻辑示例
import time

class TokenBucketLimiter:
    def __init__(self, capacity: int, fill_rate_per_sec: float):
        self.capacity = capacity
        self.fill_rate = fill_rate_per_sec
        self.tokens = float(capacity)
        self.last_updated = time.time()

    def _replenish(self):
        now = time.time()
        delta = now - self.last_updated
        self.tokens = min(float(self.capacity), self.tokens + delta * self.fill_rate)
        self.last_updated = now

    def reserve(self, estimated_tokens: int) -> bool:
        self._replenish()
        if self.tokens >= estimated_tokens:
            self.tokens -= estimated_tokens
            return True
        return False

    def reconcile(self, reserved: int, actual: int):
        self._replenish()
        diff = reserved - actual
        # 多退少补，上限不能超过桶总容量
        self.tokens = min(float(self.capacity), self.tokens + diff)
```

#### 4.2 算例推演：预占与校准过程

假设在贯穿案例的“工单分析”场景中，网关配置如下：
- 目标模型供应商的 TPM 令牌桶容量为 $10000$ Token；
- 某一瞬间，桶内当前可用余量为 $8000$ Token；
- 历史工单分析任务的 P90 输出 Token 数经验值为 $800$。

**演进时序（假设此瞬间暂不计时间带来的自然注水）**：
1. **请求到达**：客服工单输入文本经本地分词计算为 $1200$ Token，客户端请求中传入 `max_tokens = 4096`。
2. **网关预扣**：计算预占量 $T_{\text{reserve}} = 1200 + \min(4096, 800) = 2000$ Token。令牌桶可用余量更新为 $8000 - 2000 = 6000$ Token。网关放行请求。
3. **推理完成**：模型实际生成较短结论，响应体返回真实用量为 `prompt_tokens = 1200`, `completion_tokens = 350`，实际总消耗 $T_{\text{actual}} = 1550$ Token。
4. **回退校准**：计算差值 $\Delta T = 2000 - 1550 = +450$ Token。网关向令牌桶归还 $450$ 令牌，可用余量校准为 $6450$ Token。

![TPM/RPM 预占与校准状态转移图](/images/content-rebuild/cba0a54969a72542/fig-4.png)

---

### 5. 故障重试与熔断机制：针对 429 与 5xx 的退避与断路器

上游供应商的异常一般分为两类：**瞬时毛刺**（偶发超时、单次 503、瞬时 429）与**持续故障**（大面积宕机、机房网络阻断）。网关应采用分级策略应对。

#### 5.1 响应状态码处理策略

| HTTP 状态码 | 故障分类 | 网关行为决策 | 重试与回退策略 |
| :--- | :--- | :--- | :--- |
| **400 / 422** | 客户端入参非法（Schema 校验错误、字段缺失） | **禁止重试**，透传错误并规范化为通用错误 JSON | 立即返回业务端，不得浪费上游算力。 |
| **401 / 403** | 物理 API Key 失效、权限不足或被封禁 | **禁止向客户端暴露**，告警并剔除该 Key | 将当前 Key 置为不健康，换用凭证池内备用 Key 重试。 |
| **429** | 触发供应商调用限速 | **可控退避重试** | 优先读取响应头中的 `Retry-After` 秒数；若无则执行带抖动的指数退避；重试超限后触发换 Key 或换模型降级。 |
| **500 / 502 / 503** | 供应商服务过载或网关超时 | **有限换路重试** | 切换至同类型备用供应商实例或降级模型，重试次数一般限制为 1~2 次。 |

带随机抖动的指数退避（Exponential Backoff with Jitter）公式如下，引入抖动因子可有效防止重试请求在同一毫秒共振重叠：
$$t_{\text{sleep}} = \min(t_{\text{max}}, \text{base} \times 2^{\text{attempt}}) \times \text{Uniform}(0.5, 1.5)$$

#### 5.2 断路器状态机与半开（Half-Open）安全探测

当上游模型出现大范围故障（例如持续返回 500 持续超过 30 秒）时，如果网关不做拦截，海量业务连接将卡在网关层等待超时，造成文件描述符与线程池耗尽。高可用冗余架构参见 [高可用设计](/architecture/high-availability.md)。

断路器维护三个核心状态：
1. **Closed（闭合状态）**：正常代理请求。网关在滑动窗口（如最近 60 秒）内统计失败率。当失败率超过阈值（如 $50\%$ 且样本数 $\ge 20$），断路器跳闸转换为 **Open** 状态。
2. **Open（断开状态）**：直接拦截所有发往该供应商的流量，执行快速失败（立即返回网关 503）或直接分流至备用降级模型。启动冷却计时器（如 30 秒）。
3. **Half-Open（半开探测状态）**：冷却计时结束后，断路器进入半开探测阶段。

**半开状态下的安全探测机制**：
- **单探针限流（Canary Probe）**：半开状态绝不能一次性放开所有请求，必须仅允许极少量请求（如每秒放行 1 个或严格限制为单并发探针）流向该供应商，其余常规流量继续维持拦截降级。
- **连续成功计数（Consecutive Successes）**：设置判定阈值 $N$（如连续 5 次探测均成功）。
  - 若探针请求返回 200 且正常生成，计数器加 1；当连续成功达到 $N$ 次时，状态机平滑切回 **Closed**，恢复全量放行；
  - **一票否决制**：在半开状态下，只要任意一个探针请求出现 5xx 或 429 错误，状态立即重置为 **Open**，并将冷却时间延长（如指数递增至 60 秒），彻底遏制上游未恢复时的流量冲击。

![断路器三态流转与半开渐进探测机制](/images/content-rebuild/cba0a54969a72542/fig-5.png)

---

### 6. 网关层可观测性：拆解 TTFT、TPOT 与健康指标

与通用 HTTP 请求不同，大模型推理由于包含长时间的自回归解码过程，端到端耗时可达数秒甚至几十秒。网关必须将**连接等待、首字生成与后续解码耗时拆开观测**，整体日志与 Trace 设计可参考 [日志与 Trace](/architecture/logging-trace.md)，底层计算与显存影响可对比 [大模型推理中的首 Token 延迟和平均生成速度分别受什么影响？](/interview/model-deployment-interview/latency-throughput.md)。

#### 6.1 核心可观测延迟指标

1. **TTFT（Time to First Token / 首包延迟）**：
   从网关接收到客户端完整请求，到向上游建立连接、上游完成输入处理（Prefill）并返回第一个有效文本 Token 块的时间差。它反映了网络传输延迟、上游调度排队耗时以及 Prefill 计算速度。
2. **TPOT（Time Per Output Token / 单 Token 解码耗时）**：
   进入文本输出解码阶段后，平均生成每个 Token 所需的耗时。计算公式如下（当生成 Token 总数 $N > 1$ 时）：
   $$\text{TPOT} = \frac{\text{Total Latency} - \text{TTFT}}{N - 1}$$
3. **Total Latency（端到端总延迟）**：从连接建立到收到最后一个 `[DONE]` 标记或 HTTP 响应完全关闭的总耗时。
4. **Rate Limit Hit Ratio（限流命中率）**：网关层预扣限流拦截频次与上游物理 429 发生次数的比率。健康系统的目标是让上游物理 429 趋近于 0。

#### 6.2 Prometheus 监控指标暴露示例

```prometheus
# TTFT 首包耗时直方图（按模型、供应商、租户多维度打标）
llm_gateway_ttft_seconds_bucket{le="0.5", model="gpt-4o", provider="azure", tenant="finance"} 128
llm_gateway_ttft_seconds_bucket{le="1.0", model="gpt-4o", provider="azure", tenant="finance"} 342
llm_gateway_ttft_seconds_bucket{le="2.5", model="gpt-4o", provider="azure", tenant="finance"} 410

# TPM 与 RPM 实际 Token 吞吐消耗计数器
llm_gateway_tokens_consumed_total{type="prompt", model="claude-3-5-sonnet", tenant="search"} 1450200
llm_gateway_tokens_consumed_total{type="completion", model="claude-3-5-sonnet", tenant="search"} 832040

# 断路器状态观测 (0: Closed, 1: Half-Open, 2: Open)
llm_gateway_circuit_breaker_state{provider="openai-direct"} 0
llm_gateway_circuit_breaker_state{provider="anthropic-backup"} 1
```

![Prometheus 指标采集与首包延迟 (TTFT) / 生成延迟拆解观测拓扑](/images/content-rebuild/cba0a54969a72542/fig-6.png)

---

### 7. 生产验收清单

在将模型网关交付生产环境运行前，团队应按照下表完成各项边界验收验证：

| 验证维度 | 测试动作与输入条件 | 预期系统状态演变 | 失败与边界判定标准 |
| :--- | :--- | :--- | :--- |
| **协议适配** | 客户端输入统一 OpenAI 格式消息，后端配置映射至 Anthropic/Gemini。 | 字段正常转换并透传，流式响应统一解析为标准 `choices[0].delta` 结构。 | 遇到目标模型不支持的字段时，网关应按规范宽容丢弃或前置报错，绝不能抛出网关内部 500。 |
| **TPM 预占限流** | 持续高并发发起长输入、大 `max_tokens`（如 4096）请求。 | 令牌桶按预占额正常扣减；请求完成后，差额令牌能够被原子退回。 | 网关排队或返回 429，上游供应商物理 429 发生率必须维持在极低水平（趋近 0）。 |
| **凭证故障隔离** | 模拟物理 API Key 出现 401 鉴权过期或连续触发 429。 | 该 Key 权重立即降为 0 并移入隔离池，请求平滑漂移至同池备用 Key。 | 业务客户端无请求中断；网关触发 Key 异常告警通知管理员。 |
| **断路器熔断** | 将目标服务商 Mock 为 100% 返回 503 错误。 | 失败率达到阈值后断路器跳闸进入 Open 状态，后续请求在网关层毫秒级快速失败或降级。 | 冷却结束后进入半开阶段，仅允许极少数探测请求通过；探测失败立即回滚 Open，绝不上浮堆积连接打崩网关。 |
