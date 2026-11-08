---
title: 业务系统如何接入大模型？落地架构与边界设计
icon: mdi:puzzle-outline
tag:
- 高频场景题
- 业务接入
- 落地
description: 系统梳理现有微服务接入大模型时的协议转换、行级鉴权、读写分离与幂等控制，提供标准口述框架与落地实现。
---

![大模型与企业微服务深度集成架构全景示意图](/images/content-rebuild/441126547806322c/cover.png)

在企业生产环境中，“调用大模型 API 跑通对话 demo”与“将模型接入支撑核心业务的微服务集群”完全是两回事。企业现有的业务系统多建立在强类型接口、毫秒级响应、ACID 事务与细粒度数据权限之上；大语言模型则是秒级长耗时、按概率生成且无原生事务支持的外部服务。如果直接在现有 RPC 接口或数据库事务中同步调用大模型，极易导致连接池耗尽、状态脏写以及越权泄密。

本文以“电商售后客服系统接入大模型处理退换货”为贯穿案例，系统梳理企业微服务接入大模型时的协议改造、数据鉴权、事务隔离与可观测性落地设计。

---

## 面试标准口述答案

如果面试官提问：**“如果让你把现有业务系统接入大模型，你会怎么落地？”**，建议直接使用以下结构化框架作答：

> **核心口述**：
> 接入的核心是解决**“秒级概率系统”与“毫秒级确定性微服务”的技术冲突**。落地不能让大模型直接穿透到现有数据库或核心微服务，必须通过中间的 **AI 编排适配层** 进行隔离，具体分三个维度落地：
> 1. **通道与协议改造**：针对交互式对话，网关层配置禁用反向代理缓冲，支持 SSE 流式长连接并引入心跳保活；针对无需实时逐字交互或耗时达分钟级的批量分析任务，采用消息队列解耦为异步任务，客户端通过任务 ID 轮询或 WebSocket 接收进度。
> 2. **数据安全与权限阻断**：网关层解析 JWT 并透传用户与租户上下文。在知识检索时，必须采用**元数据预过滤（Pre-filtering）**，在向量计算前通过租户和部门 ID 物理收窄搜索空间，严禁依赖模型 Prompt 或后置过滤做鉴权。
> 3. **读写分离与防重幂等**：读路径用于信息检索与总结；写路径（如退款、创建工单）实行“意图与执行分离”。模型仅负责提取结构化参数，后端通过 Schema 校验后生成业务确定性幂等键并执行 Redis 状态锁定，涉及资金和敏感操作强制引入人工确认（Human-in-the-loop），绝不让模型直接持有数据库写权限。
> 4. **全链路可观测性**：基于 OpenTelemetry 规范，网关将 Trace ID 透传至编排层、向量库与模型推理 Span，记录首字延迟（TTFT）与 Token 消耗，作为熔断降级与成本核算的依据。

---

## 1. 核心技术冲突：长耗时概率系统与确定性微服务

![大模型长耗时概率特性与传统微服务确定性事务的技术冲突对比图](/images/content-rebuild/441126547806322c/fig-1.png)

在电商售后系统的退款场景中，传统微服务与大模型的运行时特征存在本质差异：

| 维度 | 传统业务微服务（如订单、支付） | 大语言模型服务（LLM） |
| :--- | :--- | :--- |
| **响应延迟** | 通常在 20ms ~ 200ms，超时阈值严苛（如 1s） | 首字延迟（TTFT）数百毫秒，完整生成常需 3s ~ 20s |
| **执行结果** | 确定性，相同输入与状态必有相同输出 | 概率性，受采样参数影响，存在幻觉与格式波动 |
| **事务机制** | 本地 ACID，支持分布式事务（TCC、Saga） | 无本地状态回滚机制，无法参与两阶段提交（2PC） |
| **网络模型** | 同步阻塞短连接或内部 RPC（gRPC/Dubbo） | 强依赖 HTTP Chunked / SSE 流式长连接 |

由于大模型不具备数据库事务回滚能力，若在 Spring 的 `@Transactional` 事务方法内部直接同步调用大模型 API：
1. **数据库连接池耗尽**：一个平时占用 5ms 的数据库事务，会因为等待大模型返回被硬生生拉长到 5 秒以上，HikariCP 等连接池会在高并发下迅速被打满，拖垮整条业务线。
2. **脏数据不可回滚**：若先执行了数据库扣减，后续大模型生成异常或超时断开，外部已调用的第三方变更无法通过本地事务注解自动撤销。

因此，系统架构必须将大模型交互与业务微服务彻底解耦，由编排层充当“防火墙”。

---

<ReadMoreLock />

## 2. 身份鉴权与数据隔离：从 JWT 到元数据预过滤

![用户身份上下文透传与向量检索元数据预过滤流程图](/images/content-rebuild/441126547806322c/fig-2.png)

在售后系统中，普通客服只能查看公开退换货政策与普通工单，资深主管才有权查看大客户特殊补偿细则。若知识库检索阶段未做物理隔离，模型就可能把高管补偿标准作为参考回答给普通用户。

### 2.1 身份上下文透传

API 网关在校验前端 OAuth2/JWT 凭证后，提取用户核心属性，封装在内部请求头（如 `X-User-Context`）中传递给编排层：

```json
{
  "tenant_id": "tenant_ecommerce_01",
  "user_id": "usr_88203",
  "dept_ids": ["dept_customer_service", "dept_region_east"],
  "roles": ["tier1_agent"]
}
```

编排层直接以此为准，杜绝客户端通过请求 Body 伪造身份参数。

### 2.2 元数据预过滤（Pre-filtering） vs 后置过滤（Post-filtering）

在检索增强生成（RAG）中，权限过滤有两种截然不同的实现方式：

- **后置过滤（Post-filtering）**：先按向量余弦相似度召回 Top-50，再根据用户权限遍历剔除未授权文档。若用户提问恰好命中了 50 篇未授权的高管策略文档，过滤后合法文档数为 0，导致系统直接判定“未找到相关政策”，造成严重的召回截断。
- **元数据预过滤（Pre-filtering）**：文档切片灌库时，在向量数据库（如 Qdrant、Milvus、Elasticsearch）的 Payload 中写入可访问的租户、部门及角色标签。检索执行时，底层存储引擎先利用布尔倒排索引圈定当前用户有权查看的候选集，再在该子集内计算向量距离。

以下为 Qdrant 元数据预过滤的查询代码：

```python
# 向量检索前施加租户与行级权限过滤
from qdrant_client.http import models

def search_documents_with_acl(qdrant_client, user_ctx, query_embedding, limit=5):
    tenant_id = user_ctx["tenant_id"]
    user_id = user_ctx["user_id"]
    user_depts = user_ctx["dept_ids"]

    # 构建元数据预过滤规则：租户强隔离 + 部门/公开权限并集
    acl_filter = models.Filter(
        must=[
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
        ],
        should=[
            models.FieldCondition(key="is_public", match=models.MatchValue(value=True)),
            models.FieldCondition(key="authorized_depts", match=models.MatchAny(any=user_depts)),
            models.FieldCondition(key="authorized_users", match=models.MatchAny(any=[user_id]))
        ]
    )

    # 仅在权限命中的文档子集中计算相似度
    return qdrant_client.search(
        collection_name="enterprise_kb",
        query_vector=query_embedding,
        query_filter=acl_filter,
        limit=limit
    )
```

关于向量召回失败或相关性下降的排查细节，可参考 [/interview/high-freq-scenarios/rag-retrieval-failure.md](/interview/high-freq-scenarios/rag-retrieval-failure.md)。

---

## 3. 通道与网关适配：流式 SSE 与连接管理

![微服务流式传输与后端RPC调用的协议转换架构图](/images/content-rebuild/441126547806322c/fig-3.png)

传统企业微服务网关（如 Nginx、Spring Cloud Gateway、APISIX）多针对**短生命周期、一次性 JSON 响应**调优，接入大模型流式生成时常发生三类事故：

1. **代理缓冲截断打字机效果**：Nginx 默认开启 `proxy_buffering`，网关会攒满 4KB/8KB 缓冲区才向前端刷新，用户看到的是长时间卡顿后突发大段文字。
2. **读超时强行断链**：默认网关读超时通常为 15s~30s。若模型推理首字较慢，网关会判定后端无响应而主动向客户端切断并报 `504 Gateway Timeout`。
3. **网关线程挂起**：若网关基于同步阻塞模型开发，秒级长连接会霸占工作线程，造成常规微服务 API 无法分配线程。

### 网关与编排层适配配置

- **禁用网关缓冲**：针对大模型流式路由显式配置禁用缓冲。在 Nginx 中设置：
  ```nginx
  location /api/v1/chat/stream {
      proxy_pass http://ai_orchestrator;
      proxy_buffering off;
      proxy_cache off;
      proxy_read_timeout 120s;
      # 向前置代理（如 CDN 或接入层网关）通知禁用缓冲
      add_header X-Accel-Buffering "no";
  }
  ```
- **心跳保活（Heartbeat Ping）**：当编排层在执行复杂的知识检索或多工具调用时，需每隔 2~3 秒向客户端发送一条 SSE 注释行（如 `: keepalive

`）。该行不会被前端渲染，但能持续重置链路中各级代理与防火墙的空闲超时计时器。
- **前后端协议解耦**：客户端与网关之间建立单向 SSE 长连接；编排层与内部订单、用户等微服务之间依然走高吞吐的短连接 gRPC/REST，避免流式机制向内部业务微服务扩散。

超时控制与熔断降级的具体配置策略，可参阅 [/interview/high-freq-scenarios/timeout-degradation.md](/interview/high-freq-scenarios/timeout-degradation.md)。

---

## 4. 写操作防御：读写分离与幂等控制

![写操作的结构化抽取、幂等校验与二次确认控制机制图](/images/content-rebuild/441126547806322c/fig-4.png)

在客服系统中，用户可能提出“帮我把订单 ORD-20260330 申请退款 80 元”。面对写操作，核心铁律是：**模型只产出意图草稿，执行必须经过严格校验与业务幂等控制器**。

工具集成与调用的标准协议设计可参考 [/llm-100/agent-engineering/agent-tool-integration.md](/llm-100/agent-engineering/agent-tool-integration.md) 与 [/frontier/mcp-protocol/mcp-integration.md](/frontier/mcp-protocol/mcp-integration.md)。

### 4.1 读写路径分离

- **读路径（查询类）**：用户查询 $\rightarrow$ 模型抽取查询参数 $\rightarrow$ 内部只读 RPC / 向量库 $\rightarrow$ 组装上下文 $\rightarrow$ 生成文本回答。具备天然幂等性，失败可安全重试。
- **写路径（变更类）**：模型产出结构化参数 $\rightarrow$ 强类型 Schema 校验 $\rightarrow$ 业务状态机前置检查 $\rightarrow$ 生成确定性幂等键 $\rightarrow$ （高危操作挂起待人工确认） $\rightarrow$ 业务微服务执行。

### 4.2 结构化校验与幂等防护实现

由于网络重试或 Agent 循环逻辑，大模型可能针对同一意图向后端发起多次调用。必须由后端编排层提取确定性业务特征生成幂等键，禁止由模型自行生成随机 UUID：

```python
# 写操作强校验、幂等防重与确认流程
import hashlib
from pydantic import BaseModel, Field, ValidationError

class RefundActionParams(BaseModel):
    order_id: str = Field(..., pattern=r"^ORD-[0-9]{8,12}$")
    refund_amount_cents: int = Field(..., gt=0, le=100000)  # 退款金额需大于0且不超过限额
    reason: str

def execute_llm_write_action(redis_client, payment_client, user_id, action_draft):
    # 1. 结构化 Schema 校验，防御格式漂移与非法参数注入
    try:
        params = RefundActionParams(**action_draft.get("parameters", {}))
    except ValidationError as err:
        return {"code": "PARAM_INVALID", "message": f"参数不合规: {err}"}

    # 2. 构造业务确定性幂等键（绑定用户、订单与当前意图请求标识）
    # 禁止依赖模型随机生成的 UUID
    raw_key = f"{user_id}:{action_draft['client_request_id']}:{params.order_id}"
    idempotency_key = "idemp:refund:" + hashlib.sha256(raw_key.encode()).hexdigest()

    # 3. 基于 Redis SET NX 进行并发互斥防重
    lock_acquired = redis_client.set(idempotency_key, "PROCESSING", nx=True, ex=300)
    if not lock_acquired:
        current_status = redis_client.get(idempotency_key)
        return {"code": "DUPLICATE_INVOCATION", "message": f"该操作处理中或已完成: {current_status}"}

    # 4. 高危操作人机协同阻断（Human-in-the-loop）
    if params.refund_amount_cents > 5000:  # 金额大于 50 元必须弹窗二次确认
        approval_id = create_pending_approval_ticket(user_id, idempotency_key, params.dict())
        return {"code": "PENDING_USER_CONFIRM", "ticket_id": approval_id}

    # 5. 调用下游微服务接口，透传幂等键由数据库唯一索引兜底
    try:
        rpc_resp = payment_client.execute_refund(
            idempotency_key=idempotency_key,
            order_id=params.order_id,
            amount_cents=params.refund_amount_cents
        )
        redis_client.set(idempotency_key, "SUCCESS", ex=86400)
        return {"code": "SUCCESS", "data": rpc_resp}
    except Exception as rpc_err:
        # 下游超时或未知异常不可盲目删除键，避免重试引发并发穿透或重复扣款，需置为待对账状态
        redis_client.set(idempotency_key, "UNKNOWN", ex=1800)
        raise rpc_err
```

若模型在调用外部工具时出现死循环或重试震荡，可进一步参考 [/interview/high-freq-scenarios/agent-failure.md](/interview/high-freq-scenarios/agent-failure.md)。

---

## 5. 长耗时异步任务模式：基于消息队列的解耦

![基于消息队列的长耗时异步任务解耦与状态回调流程图](/images/content-rebuild/441126547806322c/fig-5.png)

对于批量工单情感分析、全店售后退款对账等需要耗时数分钟甚至更久的长任务，SSE 长连接容易受客户端切后台、网络断连等因素影响。此时必须采用基于消息队列（如 Kafka、RabbitMQ）的异步解耦架构。

### 状态流转机制

1. **立即响应（202 Accepted）**：客户端提交批量分析请求，API 网关完成鉴权与轻量校验后，编排服务生成全局唯一的 `task_id`，立即向客户端返回 HTTP `202 Accepted`，携带 `task_id` 与轮询端点。
2. **消息入队**：编排服务将完整任务载荷投递至 Kafka 的 `llm_heavy_jobs` Topic。
3. **Worker 调优消费**：独立的 AI Worker 节点订阅该 Topic。Worker 集群可根据目标大模型当前的并发配额（RPM/TPM）和底层 GPU 资源，执行受控拉取，避免触发上游 API 限流。
4. **状态持久化**：Worker 在处理过程中将阶段进度实时写入 Redis 或 MySQL：
   $$\text{PENDING} \longrightarrow \text{RUNNING (带百分比进度)} \longrightarrow \text{SUCCESS / FAILED}$$
5. **双向进度通知**：
   - **推模式**：客户端保持轻量 WebSocket 连接，Worker 完成阶段任务时通过推送服务发送卡片。
   - **拉模式**：客户端退避轮询（如每 3~5 秒）请求 `/api/v1/tasks/{task_id}` 获取最新状态，即使推通道断开也能保障数据一致。

---

## 6. 全链路追踪与可观测性：贯穿微服务与大模型

![贯穿微服务与大模型全链路的Trace可观测性架构图](/images/content-rebuild/441126547806322c/fig-6.png)

传统微服务通过 SkyWalking 或 OpenTelemetry 已能很好地追踪 RPC 调用，但大模型接入后，调用链往往在此处变成黑盒：监控上只显示一个长达 15 秒的 HTTP 请求，无法区分是向量检索慢、外部模型排队、首字生成（TTFT）耗时长，还是工具执行卡顿。

### 规范落地方案

基于 **OpenTelemetry Semantic Conventions for Generative AI**，微服务网关生成的 `traceparent` 上下文透明注入 AI 编排层，并在当前调用树中细化关键 Span：

```text
[Trace ID: 7c8f2a1b9e0d4c3f8a1234567890abcd]
 ├── (Root Span) API-Gateway: POST /v1/chat/completions (耗时: 4120ms)
 │    ├── (Child Span) Auth-Service: Validate-JWT (耗时: 12ms)
 │    └── (Child Span) AI-Orchestrator: ProcessRequest (耗时: 4095ms)
 │         ├── (Span) VectorDB: Hybrid-Retrieval (耗时: 48ms, 命中块: 4)
 │         ├── (Span) LLM: Model-Inference (耗时: 3820ms)
 │         │    ├── Attribute: gen_ai.system = "deepseek-chat"
 │         │    ├── Attribute: gen_ai.usage.prompt_tokens = 1120
 │         │    ├── Attribute: gen_ai.usage.completion_tokens = 380
 │         │    ├── Metric: time_to_first_token (TTFT) = 520ms
 │         │    └── Metric: tokens_per_second = 31.8
 │         └── (Span) Tool-Call: Order-Service-RPC (耗时: 95ms)
```

### 埋点核心动作

1. **上下文延续**：编排服务接收网关传递的 Header，并在异步任务上下文中通过 `contextvars` 维持 Trace ID 绑定。
2. **关键阶段打点**：将 Embedding、向量检索、Prompt 组装、模型首字返回、流式完成、外部 Tool 调用各自切分为子 Span。
3. **消耗与性能指标沉淀**：在 LLM Span 上打入 Token 消耗、模型名称、温度值及 TTFT。这既是性能瓶颈排查的基础，也是跨部门核算成本的直接凭据（成本优化细节可见 [/interview/high-freq-scenarios/cost-optimization.md](/interview/high-freq-scenarios/cost-optimization.md)）。

---

## 面试高频追问与应对策略

### 追问 1：大模型调用退款或扣款接口时，如何设计幂等性机制防止模型重试导致重复扣款？

> **口述要点**：
> 1. **意图与执行严格解耦**：模型只负责输出结构化参数草稿，不能直接获得数据库写权限或微服务写操作 Token。
> 2. **确定性业务幂等键**：不能依赖模型自身生成随机 UUID（模型重试会产生新 UUID 导致幂等击穿）。幂等键必须由业务编排层在收到请求时，结合用户 ID、前端请求批次号（Client Request ID）和核心业务实体 ID（如订单号）生成哈希键。
> 3. **两阶段防重控制**：
>    - 接入层利用 Redis `SET NX` 进行并发防重，防止高频重复击发；
>    - 核心扣款微服务使用数据库唯一索引（如 `unique_index(idempotency_key)`）进行落库兜底，遇到冲突直接返回先前结果。
> 4. **阈值风控与人机协同**：根据业务风险对写操作设置金额与频次阈值。例如超过 50 元的退款请求，系统自动挂起并返回确认卡片，由用户在前端显式点击确认后再执行写库。

### 追问 2：企业内部存在严格的数据权限（如只能看本部门文档），在向量数据库中如何阻断越权？

> **口述要点**：
> 1. **坚决不用后置过滤**：若“先向量检索 Top-K，再校验权限剔除无权文档”，当排名前列的内容全属于其他部门时，有效文档会被彻底挤出，造成召回断崖。
> 2. **元数据预过滤（Pre-filtering）**：在文档灌库切片时，将租户标识（`tenant_id`）、可读部门 ID 列表（`authorized_depts`）写入向量索引的 Payload 元数据。
> 3. **检索表达式动态注入**：检索时，直接提取网关已鉴权的用户上下文注入为布尔过滤条件。存储引擎在相似度计算前通过倒排索引完成物理裁剪，确保进入相似度计算的候选集全部合法。
> 4. **输入清洗与 Prompt 防御**：通过输入层规则过滤 Prompt 注入（防注入技术详见 [/interview/high-freq-scenarios/prompt-injection-defense.md](/interview/high-freq-scenarios/prompt-injection-defense.md)），并在 Prompt 中明确限定“严格依据提供的资料回答”，防止模型根据通用预训练知识越界推理。

---

## 常见误区与避坑清单

- **误区 1：在 System Prompt 声明“严禁查看财务部文档”就能确保权限隔离**。
  **事实**：Prompt 无法承担访问控制责任。提示词约束极易被精心构造的 Prompt 注入绕过，且模型无法准确记忆和解析企业动态更新的细粒度 ACL。权限控制必须在向量检索阶段利用元数据预过滤进行代码级物理阻断。
- **误区 2：把大模型调用直接写在 `@Transactional` 注解修饰的业务方法中**。
  **事实**：大模型调用耗时长达数秒至数十秒。在数据库事务中发起外部长耗时 HTTP 调用会导致连接池被长期占死，引发雪崩；且模型超时后，分布式环境下已发生的操作无法靠单机本地事务自动回滚。
- **误区 3：单纯把网关超时时间从 30 秒增加到 300 秒来支持长任务**。
  **事实**：这是典型的反模式。面对分钟级、离线批量或无实时流式交互的长任务，长连接会面临网关线程积压、中间防火墙静默断链、移动端切后台中断等问题。对于此类耗时极长或批量分析的任务，建议采用基于消息队列的异步任务（202 Accepted + 状态查询/推送）模式进行解耦，而非盲目拉长长连接超时。

---

## 思考与检验

1. 如果大模型在多轮对话中需要频繁调用 ERP 系统的实时库存查询接口，而 ERP 接口最多只能承受 10 QPS，你在 AI 编排层应如何设计流量保护？
2. 用户在对话中要求“撤回刚才通过 AI 提交的请假审批”，在不可回滚的大模型集成架构中，应该如何设计补偿事务（Compensating Transaction）？
