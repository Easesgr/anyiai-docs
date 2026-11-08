---
title: 如何设计一个多模型网关？统一协议、动态路由与配额结算实践
icon: mdi:gateway
tag:
- 系统设计面试
- 模型网关
- 路由
description: 梳理多模型网关的核心设计：OpenAI超集契约适配、流式长连接中继、多因子动态路由、Redis两阶段Token预扣减与故障防雪崩方案。
---

![多模型网关系统整体架构图](/images/content-rebuild/290fe1fe0a0c05e9/cover.png)

## 面试速答：90秒口述核心设计

> **面试官提问**：如果让你从零设计一个面向生产环境的多模型网关，你会如何设计核心架构，并保证高可用与成本可控？

**建议口述框架**：

多模型网关主要承担四个职责：**接口契约收敛**、**动态路由与容灾**、**流式长连接中继**以及**精准配额管控**。

1. **契约归一与适配器**：对外暴露以 OpenAI 规范为基础的超集接口，屏蔽 Anthropic、Google 与自建 vLLM 在角色命名（如 Claude 顶层 system 字段）、工具调用（Tool Calls）结构以及流式 Chunk 格式上的差异，业务端无需为换模型修改调用代码。
2. **多因子路由与平滑降级**：通过“静态能力过滤（上下文长度、视觉、结构化输出）+ 动态打分（延迟、成本单价、可用率）”分发流量。各供应商配置独立断路器（健康、熔断、半开试探）；主通道故障时采用阶梯式金丝雀切流并配合并发隔板，防止备用供应商被瞬间流量打垮。
3. **流式中继与级联取消**：采用非缓冲流式管道，显式下发 `X-Accel-Buffering: no` 破除反向代理缓冲；同时监听客户端连接状态，一旦前端页面关闭或断连，立即通过连接池或推理引擎的 `abort` 接口中断上游，避免浪费 GPU 算力。
4. **两阶段 Token 计费**：LLM 生成长度事前不可知，单纯按请求数（RPS）限流无法防范长上下文击穿。网关在请求前先计算输入 Token 并结合最大生成参数进行**原子预冻结**；流式传输中按 Chunk 累加实际输出；连接结束或异常断开时**多退少补**，杜绝恶意欠费与额度虚报。

在整体技术栈中，网关直接承接上层如 [RAG、Agent 与 Workflow 系统](/interview/ai-system-design-interview/rag-agent-workflow.md) 的底层请求，并贯穿于整个 [调用生命周期流程](/llm-100/app-system/model-api-request-lifecycle.md)。

---

## 贯穿场景：一次复杂的模型调用

为了让机制更加具体，我们用一个典型调用贯穿全文：

业务客户端发来一条请求：“请分析附件中的 20 页财报文本，提取 3 个核心风险点，并调用 `create_risk_ticket` 工具录入系统”。
* **输入规模**：约 25,000 Token；
* **特殊需求**：支持函数调用（Tool Calls）、长上下文（>32K）；
* **候选提供商**：OpenAI GPT-4o、Anthropic Claude 3.5 Sonnet、私有化部署的开源模型集群（vLLM 承载的 Qwen-72B）。

网关需要将标准请求翻译给目标供应商，处理并发扣费，在数秒的流式输出中维持连接，并在提供商超时时安全降级。

---

## 统一契约与适配器层设计

若让业务系统直接对接各模型厂商，模型切换将涉及大量分支判断。网关适配器层的任务就是完成双向协议对齐。

![统一API契约与异构厂商适配器层的双向转换流程图](/images/content-rebuild/290fe1fe0a0c05e9/fig-1.png)

### 1. 契约超集定义与差异抹平

网关以 OpenAI Chat Completions 协议作为基准协议扩展，重点处理三处异构点：

* **角色与消息结构**：OpenAI 允许在 `messages` 数组中任意位置放置 `system` 角色；Claude 要求系统提示词必须放在顶层 `system` 字符串字段中，且 `messages` 中的 `user` 与 `assistant` 必须严格交替出现。网关适配器需要自动提取所有系统消息拼接至顶层字段，并将连续出现的同角色消息做文本合并。
* **工具调用（Tool Calls）规范**：OpenAI 使用 `tools: [{type: 'function', function: {...}}]`，流式响应中增量返回 `tool_calls` 的 `index`、`id` 与 `arguments` 片段；Gemini 原生接口使用 `function_declarations`。适配器需要将通用 JSON Schema 翻译为目标厂商格式，并在输出流中将私有增量组装回标准 `chat.completion.chunk` 的 delta 格式。
* **超参数与截断规则**：不同模型对采样参数取值范围不同。例如某些私有推理框架对 `temperature` 超过 1.0 会抛错，而 OpenAI 允许 0 到 2.0；部分模型支持 `max_tokens`，部分模型使用 `max_completion_tokens` 或 `max_new_tokens`。网关需要在此阶段做参数截断与重命名，防止非法请求直接透传至上游。

### 2. 流式响应（SSE）归一化代码实现

在流式传输（Server-Sent Events）模式下，网关不能等整个请求生成完毕再做格式转换，必须在异步生成器中对异构事件流做低延迟逐帧转换。

以下是以 Python 异步生成器为例的转换逻辑，演示如何将 Anthropic 的事件流转换为兼容 OpenAI 的 SSE 数据帧：

```python
import json
from typing import AsyncGenerator

async def normalize_claude_stream_to_openai(
    raw_claude_stream: AsyncGenerator[str, None],
    request_id: str,
    model_alias: str
) -> AsyncGenerator[str, None]:
    """
    将 Anthropic Claude SSE 事件流实时归一化为 OpenAI 标准数据流格式
    """
    async for line in raw_claude_stream:
        line = line.strip()
        if not line or not line.startswith("data: "):
            continue
        
        payload_str = line[len("data: "):]
        if payload_str == "[DONE]":
            yield "data: [DONE]\n\n"
            break
            
        try:
            event = json.loads(payload_str)
        except json.JSONDecodeError:
            continue
            
        event_type = event.get("type")
        delta_text = ""
        finish_reason = None
        
        # 提取 Claude 的内容增量与结束标记
        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                delta_text = delta.get("text", "")
        elif event_type == "message_delta":
            delta = event.get("delta", {})
            stop_reason = delta.get("stop_reason")
            if stop_reason == "end_turn":
                finish_reason = "stop"
            elif stop_reason == "max_tokens":
                finish_reason = "length"
        elif event_type == "message_stop":
            finish_reason = "stop"
            
        # 组装为 OpenAI 标准 Chunk
        openai_chunk = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "model": model_alias,
            "choices": [{
                "index": 0,
                "delta": {"content": delta_text} if delta_text else {},
                "finish_reason": finish_reason
            }]
        }
        
        yield f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"
        
        if finish_reason is not None:
            yield "data: [DONE]\n\n"
            break
```

<ReadMoreLock />

---

## 动态路由与负载均衡设计

在多模型网关中，客户端传入的通常是逻辑别名（如 `general-analysis-tier` 或 `code-fast`），网关需要根据当前状态决定将其打到哪个具体的物理端点。

![多因子动态路由决策矩阵与熔断器状态机联动示意图](/images/content-rebuild/290fe1fe0a0c05e9/fig-2.png)

### 1. 硬性规则过滤（Capability Matching）

第一阶段先排除无法满足技术指标的节点：

* **上下文窗口阈值**：以本文开头的贯穿例子为例，输入含 25,000 Token，预估输出 2,000 Token，总长度达 27K。如果私有集群的 Qwen 端点单实例配置的最大上下文仅为 16K，则该端点在硬过滤阶段直接剔除，避免发到上游触发超长报错。
* **功能特性匹配**：请求声明了工具调用（Tool Calls），如果备选列表中的某轻量模型不支持结构化工具调用，同样被直接过滤。
* **多模态与安全合规**：包含图像输入的请求只能路由给带视觉编码器的端点；对敏感数据要求物理隔离的租户，仅保留私网托管端点。

### 2. 动态多因子加权打分

通过硬过滤后的候选端点集合，进入多因子加权计算：

$$\text{Score}_i = w_c \cdot S_{\text{cost}}(i) + w_l \cdot S_{\text{latency}}(i) + w_a \cdot S_{\text{avail}}(i)$$

各因子的量化方式需要统一标尺归一化（如缩放至 $[0, 1]$ 之间）：
* **成本得分 $S_{\text{cost}}(i)$**：根据每百万 Token 输入输出综合单价折算，价格越低得分越接近 1。在批量异步作业中，可调高 $w_c$ 的权重；
* **延迟得分 $S_{\text{latency}}(i)$**：基于该端点在最近滑动窗口（如最近 3 分钟）内的 P95 首字时间（Time to First Token, TTFT）计算，延迟越低得分越高；
* **可用率得分 $S_{\text{avail}}(i)$**：根据近 100 次调用的非 5xx / 非超时成功率计算，遇到故障迅速扣减。

### 3. 断路器（Circuit Breaker）与故障隔离

为了避免某一供应商故障拖垮整体系统，每个提供商实例挂载独立的熔断状态机：
* **Closed（健康运行）**：请求正常分发。如果统计窗口内错误率超过设定阈值（例如连续 5 次网络超时或 429 比例超过 20%），状态跃迁为 **Open**。
* **Open（熔断拦截）**：网关直接短路该节点的所有流量，候选池中将该节点剔除，由其他节点承接，并启动冷却定时器（如 30 秒）。
* **Half-Open（半开探测）**：冷却期结束后，网关仅放行极小比例（如 2% 或固定每秒 1 个请求）作为健康探针。若探针连续成功且延迟符合预期，则恢复为 Closed；一旦探针失败，立即重置冷却计时并回退至 Open。

关于容灾分级与超时的详细设计，可参阅 [模型调用失败、超时与降级处理机制](/interview/ai-system-design-interview/failure-handling.md)。

---

## 配额管控与两阶段 Token 结算

在传统 Web 架构中，网关通常按请求数（RPS）限流。但在大模型场景中，**单纯依赖 RPS 限流无法防止系统过载与资金坏账**。

一个包含 32K 提示词的长文本请求，与一个发“你好”的短请求，消耗的显存计算时间与计费成本相差上千倍。网关必须引入 **RPM（每分钟请求数）** 与 **TPM（每分钟 Token 数）** 的双重配额。

![基于Redis的Token两阶段预扣减与流式精准结算时序图](/images/content-rebuild/290fe1fe0a0c05e9/fig-3.png)

### 1. 为什么不能等到生成完毕后再扣费？

在 LLM 流式输出中，网关在发起请求时**只能精确统计输入的 Token 数，无法提前预知模型究竟会输出多少 Token**。
* **若事后结算**：恶意用户可以并发发起数百个要求生成长文本的请求，在账单提交前将租户账户余额打穿，网关承担坏账；
* **若按上限全扣**：许多业务为了防止截断，默认设置 `max_tokens: 4096`，但通常模型在回答“是或否”时仅输出几个 Token。若每次都按最大上限全额扣减，会导致高并发下租户额度被瞬间占满，产生大量虚假的“额度不足”拒绝。

因此，网关需要采用**“预冻结 - 实际累加 - 多退少补结算”**的两阶段状态流转。

### 2. Redis Lua 原子两阶段结算流程

依赖 Redis 执行 Lua 脚本，可以保证多实例部署下的并发原子性：

1. **前置估算与预冻结（Reservation）**：
   网关利用本地快速分词工具（如 tiktoken）计算出本次 Prompt 的输入 Token 数 $T_{\text{in}}$。结合请求中指定的预期最大输出 $T_{\text{est\_out}} = \min(\text{req.max\_tokens}, \text{Gateway\_Limit})$，计算最大预估成本：
   $$E_{\text{reserve}} = T_{\text{in}} \cdot P_{\text{in}} + T_{\text{est\_out}} \cdot P_{\text{out}}$$
   网关调用 Redis Lua 脚本校验 `Available_Balance >= E_{reserve}`。如果满足，从可用额度扣除并划入 `Frozen_Balance`，下发本次请求的事务 ID；如果不满足，在毫秒级内返回 HTTP 402 或 429。
2. **流式消费与断点记录（Streaming）**：
   随着上游逐块推流，网关内部统计每帧返回的文本，并累加实际输出的 Token 计数 $T_{\text{actual\_out}}$。
3. **二次结算与差额退还（Settlement）**：
   流式请求正常结束时，网关优先读取上游返回的最终 usage 统计或解析 Chunk 累加实际输出 $T_{\text{actual\_out}}$。若发生客户端中断或上游异常，上游服务端在断开前往往已完成部分解码并产生实际计费，网关需结合上游返回的用量或本地已接收 Chunk 及计费策略核算实际消耗：
   $$C_{\text{actual}} = T_{\text{in}} \cdot P_{\text{in}} + T_{\text{actual\_out}} \cdot P_{\text{out}}$$
   Lua 脚本原子地扣除已冻结的 $E_{\text{reserve}}$，实扣 $C_{\text{actual}}$，并将差额 $(E_{\text{reserve}} - C_{\text{actual}})$ 退还至可用余额，避免未核实服务端消耗就无条件全额退还导致账单穿透。

---

## 流式中继与连接生命周期管理

在 LLM 场景中，流式长连接通常持续数秒至数十秒。如果长连接管理不当，不仅会破坏客户端交互体验，还会导致后台大量空耗。

![流式SSE管道长连接、反向代理缓冲与客户端断连级联取消机制](/images/content-rebuild/290fe1fe0a0c05e9/fig-4.png)

### 1. 禁用反向代理缓冲（Proxy Buffering）

在网关前置部署了 Nginx、OpenResty 或 Kubernetes Ingress 时，**代理层默认开启的缓冲机制会导致打字机效果失效**。代理层会倾向于攒满一个 buffer（如 4KB）或等连接关闭才向下游 flush，造成前端长时间卡顿后一次性弹出大段文本。

解决该问题需要两端配合：
* **显式协议标头**：网关在响应头中必须注入：
  * `Content-Type: text/event-stream; charset=utf-8`
  * `Cache-Control: no-cache, no-transform`
  * `Connection: keep-alive`
  * `X-Accel-Buffering: no`（通知 Nginx/Ingress 立即穿透推流，不要暂存）
* **应用层显式 Flush**：网关在向 TCP 连接写入每一个 SSE 块（chunk）后，必须调用输出流的 `Flush()` 方法，促使操作系统的网络协议栈及时发送数据包。

### 2. 客户端断连检测与级联取消（Cancellation Propagation）

一个普遍的资源浪费场景：用户提问后等待 2 秒便关闭了浏览器标签页，或者前端因路由跳转直接中止了 HTTP 请求。

如果网关没有捕获这一事件，上游的模型服务（无论是商业 API 还是占用显存的 vLLM Worker）仍会继续完整自回归生成，消耗数千 Token 乃至长达数十秒的算力。

**级联取消的工程链路**：
1. 网关将每个请求的连接生命周期与语言运行时的 Context（如 Go 的 `r.Context()`）进行绑定；
2. 监听底层 Socket 的读写报错或客户端断开事件（如收到 TCP RST）；
3. 一旦捕获断连，立即通过 HTTP 客户端取消对上游供应商的未决请求，或向自建推理集群（如 vLLM）下发专用的 `abort` 接口调用，触发推理引擎立即释放解码显存；
4. 结算模块以断开前最后收到的 Chunk 结算实际 Token，完成配额对账。

关于前端与后端在流式交互下的协同，可参考 [前端、后端与大模型服务协同架构](/llm-100/app-system/frontend-backend-model.md)。

---

## 缓存层设计与 Singleflight 防击穿

对于通用问答或重复度高的 Prompt，引入缓存能显著降低延迟并节省 Token 开支。网关可设计两级缓存链。

![L1精确哈希与L2向量语义缓存双层架构及Singleflight防击穿控制](/images/content-rebuild/290fe1fe0a0c05e9/fig-5.png)

### 1. L1 精确哈希与 L2 语义缓存

* **L1 精确哈希缓存（Exact Match）**：
  * **Key 生成规则**：`SHA256(Tenant_ID + Model_Version + System_Prompt + User_Prompt + Params_Hash + Tools_Hash)`
  * **适用条件与边界**：通常在 `temperature == 0` 或业务显式声明幂等时生效。需要注意，哈希匹配仅代表提示词与入参字面一致，若外部工具状态、知识库数据发生更新或涉及权限敏感数据，哈希相等不能保证业务答案依然正确，必须配合合理的 TTL 时效与主动失效机制。
* **L2 向量语义缓存（Semantic Cache）**：
  * **机制**：当 L1 未命中时，将用户 Prompt 通过小型 Embedding 模型生成稠密向量，在向量索引中进行近似最近邻检索。模型选型可参阅 [文本 Embedding 模型选型](/fundamentals/embedding/embedding-model-selection.md)。
  * **边界与风险**：语义缓存依赖预设相似度阈值。绕过语义缓存是基于时效敏感词（如“昨天”、“最新”）、权限隔离及多轮长上下文指代风险的业务策略；但跳过缓存直接调用模型并不保证模型回答必然正确，仍需配合业务层校验。

### 2. Singleflight 机制压制热点击穿

在突发事件或早高峰期间，大量用户可能在同一秒内提交完全相同的提问。如果缓存此时刚好过期或未命中，所有并发请求将瞬间打向上游模型，极易导致供应商抛出 429 或本地推理集群排队暴涨。

网关通过 **Singleflight（并发调用合并）** 抑制此类流量：
* 针对相同请求 Key，网关在进程内使用互斥锁或并发通道进行拦截；
* 第一个到达的协程获准向上游发起模型请求，其余相同请求挂起并加入等待队列；
* 上游响应返回后，结果写入缓存，同时向所有挂起的请求广播这一响应并返回，将成百上千个并发调用压缩为单次实际调用。

---

## 面试追问、高频陷阱与系统边界

![供应商故障全量切换导致的惊群效应与渐进式权重分流对比图](/images/content-rebuild/290fe1fe0a0c05e9/fig-6.png)

### 追问 1：下游主供应商突发 429 限流，全量切流如何避免备用供应商发生“惊群雪崩”？

**原因分析**：
许多系统在故障切换时，简单将主通道切断并将 100% 流量瞬间压给备用通道。备用供应商通常未做足等量容量准备，会在短时间内由于突发流量被冲垮，导致全链路依次熔断瘫痪。

**防御方案**：
1. **隔板模式（Bulkhead）**：每个供应商配置严格的最大并发连接池上限。无论路由调度算法如何倾向备用端点，派发的并发量绝对不能超过该端点的承载容量，超出部分进入排队或快速失败；
2. **阶梯式金丝雀切换**：主供应商熔断后，备用供应商流量按照阶梯比例增加（如 $10\% \to 30\% \to 70\% \to 100\%$），每次放量观察一个时间窗口，确认首字延迟和错误率平稳后再继续加权；
3. **带随机抖动的指数退避（Exponential Backoff with Jitter）**：当收到 429 时，各客户端与网关内部重试间隔增加随机打散因子，打破重试请求在同一毫秒内同步撞车的现象：
   $$T_{\text{wait}} = \min\left(T_{\text{max}}, T_{\text{base}} \cdot 2^{\text{attempt}}\right) + \text{rand}(0, J_{\text{max}})$$

### 追问 2：流式响应推了一半上游报错（例如已经吐出 500 Token 后断连），网关怎么处理？

**机制剖析**：
在流式中继刚开始时，网关已经向下游客户端发送了 `HTTP 200 OK` 状态码和前面的文字块，此时无法再在 HTTP 协议头上把状态码改成 500。

**应对规范**：
1. **SSE 错误帧协议化**：网关捕获上游异常后，在当前的流式长连接中推入一条标准错误事件帧，例如：
   `data: {"error": {"message": "Upstream model interrupted", "code": 502}}`
   并随后推流 `data: [DONE]` 正常关闭长连接。前端 UI 框架捕获该错误事件后，停止光标闪烁，并提供“继续生成”或“重新生成”入口；
2. **按实消费精准结账**：网关记录异常中断事件，若上游返回了最终 usage 元数据则以此核算，未返回时以断连前解析到的 Token 结合上游计费与重试策略评估实扣额度。不能因异常无条件全额免单或退还上游已实际计费的配额，也不应在缺乏依据时直接没收全部预冻结额度。

### 常见设计反模式

* **反模式 1：引入大模型实时决定路由**。在网关入口处调用一次小模型来分析用户意图以决定分发到哪个大模型。这种做法会让端到端首字时间（TTFT）额外增加数百毫秒，且将外部单点故障引入到了关键路径上。路由应基于轻量启发式规则、元数据矩阵或近线分类特征完成。
* **反模式 2：把请求并发数（RPS）等同于 GPU 负载**。在自建 vLLM / SGLang 集群做网关负载均衡时，不能仅统计各 Worker 当前处理的“请求个数”。承载一个 32K 上下文的请求所消耗的 KV Cache 显存，是承载 10 个短请求的数倍。若不考虑请求上下文长度做调度，会导致某些承接了长文档的 Worker 突发 OOM。

---

## 验收指标与观测清单

多模型网关上线前，可通过以下维度进行功能验收与健康度监控：

```
[维度一：调用性能与服务质量 SLO]
├── 首字时间 (TTFT)：P50 / P95 / P99 监控（区分不同供应商与模型）
├── 生成速率 (TPOT)：每秒产出 Token 速度（Tokens Per Second）
├── 路由切换耗时：从主供应商判定异常到流量平滑导流至备选节点的生效延迟
└── 降级比率：因 429/5xx 触发熔断切流的流量在总请求中的占比

[维度二：配额与高可用鲁棒性]
├── 配额预扣减一致性：高并发对同一租户打超额并发，验证是否会发生账单穿透
├── 级联取消生效度：主动中断客户端连接，验证上游 Worker 显存与计算是否在 500ms 内释放
└── 惊群切流压测：模拟主通道 100% 抛 429，验证备用通道流量是否阶梯上升且未引发连锁熔断
```

通过统一协议解除代码层面的供应商锁定，依托双阶段配额解决不可控消耗，并在流式长连接管道中构建健壮的断连传播与平滑降级，多模型网关才能为上层应用提供稳定高效的模型调用底盘。后续可进一步延伸至 [会话与上下文管理架构](/interview/ai-system-design-interview/session-context-management.md) 以及系统的 [日志、监控与成本追踪实践](/interview/ai-system-design-interview/logging-monitoring.md)。
