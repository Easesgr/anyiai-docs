---
title: AI 对话系统如何管理上下文与历史记录？滑动窗口、存储分层与增量压缩
icon: mdi:message-text-outline
tag:
- 系统设计面试
- 会话管理
- 上下文
description: 从面试口述答案出发，剖析多轮对话的存储分层、Token 预算分配、增量摘要与分支树设计，解决长上下文延迟、成本与并发一致性问题。
---

在大模型应用系统设计面试中，“如何设计多轮会话历史与上下文管理系统”是一道高频系统设计题。很多回答容易混淆“数据库里持久化的全量聊天记录”与“单次推理实际传入的 Prompt 上下文”，或者寄希望于长上下文模型而忽视延迟、计算成本与信息稀释问题。

本文结合一个具体的机票预订助手案例，拆解会话生命周期、Token 预算切分、增量摘要提取、双写架构以及分支树设计，提供可口述、可落地的工程方案。

> 前置依赖参考：[/interview/ai-system-design-interview/rag-agent-workflow.md](/interview/ai-system-design-interview/rag-agent-workflow.md) 讨论了 RAG 与工作流的选型边界；本篇进一步聚焦对话状态本身的持久化与装配策略。

![会话与上下文管理核心架构概览](/images/content-rebuild/ee1237fe42c3713a/cover.png)

---

## 面试口述核心答案

面试中被问及“如何设计大模型多轮会话与上下文管理系统”时，建议在 2 分钟内按以下结构口述：

> “我会将**全量会话持久化存储**与**模型运行时工作上下文**明确解耦，整体架构分为三层：
>
> 1. **存储分层**：持久层使用关系型数据库（如 PostgreSQL / MySQL）记录不可变的消息日志，通过 `parent_id` 维护消息拓扑，支持重新生成与多分支；缓存层使用 Redis 缓存最近活跃轮次，避免频繁扫表；长期偏好（如用户习惯）由离线或异步管道提取后沉淀至画像表。
> 2. **动态装配与压缩**：严格设置单次推理的 Token 预算。固定锚定顶部的系统提示词与工具定义；保留最近 $K$ 轮原始对话以维持指代连贯性；超出窗口的中间历史采用‘增量摘要 + 关键实体槽位’进行压缩，并记录原消息 ID 范围作为追溯指针。
> 3. **并发与一致性保障**：使用递增序列号（`seq_id`）保证消息顺序；写入采用‘先写数据库落盘，再更新 Redis 缓存’的策略，缓存未命中或失败时通过数据库重建，并在模型流式响应结束后原子化提交整条 Assistant 消息，防止脏数据或并发乱序。”

---

## 贯穿案例：多轮机票预订助手

为了让机制更加具体，我们使用一个多轮机票咨询场景贯穿全文：
* **第 1 轮**：用户输入：“帮我查 10 月 1 日北京去上海的早班机，预算 1200 元以内，只看国航。”（*关键约束：出发地、目的地、日期、时段、预算、航司*）
* **第 2~7 轮**：系统展示航班列表，用户反复比较起飞时间、改签政策、行李额，并在多趟航班之间来回询问（*产生大量临时性工具调用返回值与解释*）。
* **第 8 轮**：用户提问：“就选第 2 个吧，帮我确认一下前面的退改签费用。”

如果直接将前 7 轮的原生消息与工具返回（数千行 JSON）全量拼接传入模型，很快就会超过 Token 预算，推高首字延迟（TTFT）；如果使用简单的先进先出（FIFO）滑动窗口剔除前几轮，第 1 轮确立的“10 月 1 日、国航、1200 元以内”约束就会被抛弃，导致模型在第 8 轮给出错误的航班建议。

---

## 会话状态模型：全量历史、活跃缓存与工作上下文

解决上述冲突的第一步，是将“用户看到并保存的历史”与“送给模型计算的上下文”分离为不同生命周期的状态对象：

```text
+-------------------------------------------------------------------------+
|                        运行时工作上下文 (Prompt Context)                 |
|  [System Prompt] + [长期画像] + [结构化历史摘要] + [滑动窗口最近 K 轮]    |
+-------------------------------------------------------------------------+
                                     ^ 组装拉取
                                     |
+-------------------------------------------------------------------------+
|                    活跃会话缓存 (Active Session Cache - Redis)           |
|  - 数据结构: Sorted Set (score=seq_id) 或 List                           |
|  - 内容: 最近活跃消息的精简体，设置滑动 TTL（例如 2 小时）                |
+-------------------------------------------------------------------------+
                                     ^ 同步镜像
                                     |
+-------------------------------------------------------------------------+
|                   持久化消息表 (Persistent Storage - PostgreSQL)         |
|  - 全量追加日志 (Append-only Log)                                       |
|  - 支持分支结构 (message_id, parent_id, session_id, seq_id, role, text) |
+-------------------------------------------------------------------------+
```

1. **全量会话历史（Persistent History）**：
   记录用户的每一次输入、模型的每一次完整响应以及工具调用的原始入参与返回。以只追加写入的方式保存在关系型数据库中，作为审计、复盘、重新生成和分支回溯的依据。
2. **活跃会话缓存（Active Cache）**：
   Redis 中仅保留当前活跃会话最近 $N$ 条消息摘要或精简结构，主要目的是避免用户每次交互都对关系数据库执行深度 `SELECT ... ORDER BY seq_id` 查询。
3. **运行时工作上下文（Working Context）**：
   单次调用大模型 API 时实际传递的 `messages` 数组。标准 Transformer 架构在输入处理阶段（Pre-fill）的全注意力算术计算量随输入 Token 数呈平方增长（FlashAttention 等算子主要优化显存读写与峰值占用，无法消除平方级乘加计算量），商用 API 也是按输入 Token 数计费。工作上下文必须限制在严格的 Token 预算内。

![会话状态分层数据流转示意图](/images/content-rebuild/ee1237fe42c3713a/fig-1.png)

<ReadMoreLock />

---

## 上下文窗口装配与预算切分

当一次交互的历史 Token 数接近单次调用上限时，组装引擎需要按优先级分配预算。一个典型的预算模型如下：

| 上下文分区 | 预算占比（示意） | 淘汰与更新机制 | 功能说明 |
| :--- | :--- | :--- | :--- |
| **System 提示词与工具** | 15% | **固定锚定，永不淘汰** | 承载角色定位、输出格式、安全约束与 Function Schema |
| **长期用户偏好** | 10% | 异步加载，低频更新 | 例如“偏好语言：中文”、“乘机人证件类型”等静态标签 |
| **增量历史摘要与槽位** | 20% | 滚动合并，动态更新 | 承载已被滑出近期窗口的核心参数与早期事实 |
| **近期对话滑动窗口** | 40% | FIFO 滚动淘汰 | 保留最近 $K$ 轮原生对话，支撑即时指代（如“第 2 个”） |
| **输出预留缓冲（Buffer）** | 15% | 预留给 Completion | 确保模型回答不会因为触碰上下文最大长度限制而异常截断 |

### 淘汰策略的设计要点
1. **系统提示词固定锚定（Anchoring）**：
   系统定义与工具 Schema 始终位于上下文最前端，无论会话进行了多少轮，都不能被滑动窗口裁掉。提示词结构工程可参考 [/frontier/context-engineering/high-quality-context.md](/frontier/context-engineering/high-quality-context.md)。
2. **中间工具输出折叠（Tool Truncation）**：
   在机票案例的第 3 轮，工具查询可能返回了 20 趟航班的完整 JSON，占用 3000 Tokens。在第 4 轮之后，用户已经锁定了其中 2 趟，此时可以将第 3 轮的工具原始输出替换为紧凑摘要（例如只保留用户选中的两趟航班航班号与起飞时间），释放大量空间。

![上下文窗口装配与预算分配模型图](/images/content-rebuild/ee1237fe42c3713a/fig-2.png)

---

## 长对话压缩机制：增量摘要与结构化槽位

简单的 FIFO 滑动窗口容易丢失第 1 轮的关键信息。如果每次超限都把所有历史记录全量传给小模型做全局总结，摘要调用的输入量会随轮次呈二次方增长（$\sum_{i=1}^N i = O(N^2)$），带来额外的延迟与 API 成本。

工程上推荐采用**增量摘要 + 结构化键值槽位（Slots）**机制：

### 1. 增量摘要更新公式
当最近活跃窗口累积达到阈值时，触发后台异步压缩：
$$\text{Summary}_{\text{new}} = \text{LLM}(\text{Summary}_{\text{old}}, \text{Messages}_{i \dots i+m})$$
模型只需处理“上一版摘要”加上“即将滑出窗口的中间 $m$ 轮消息”，输出新版紧凑摘要，无需重新读取第 1 轮到第 $i-1$ 轮的全部原始记录。

### 2. 槽位提取与指针对齐
纯自然语言的叙述式摘要容易遗漏精确数字或条件。在压缩提示词中，要求模型输出带有“实体槽位”的结构化 JSON：

```json
{
  "summary_text": "用户计划 10 月 1 日从北京飞往上海，优先考虑早班机，已排除晚班方案。",
  "critical_slots": {
    "departure": "北京",
    "destination": "上海",
    "date": "2026-10-01",
    "airline_pref": "国航",
    "budget_cny": 1200
  },
  "covered_message_range": ["msg_001", "msg_014"]
}
```

在组装下一轮 Prompt 时，将该结构转化为紧凑文本注入在系统提示词之后：
```text
[前序背景与约束（覆盖第 1-7 轮）]：
- 核心事实：用户计划 10 月 1 日从北京飞往上海，优先早班机。
- 锁定约束：出发=北京，到达=上海，日期=2026-10-01，航司=国航，预算<=1200元。
```
这样，即便第 1 轮对话的原生文本已从滑动窗口中剔除，关键业务参数依然被明确保留。更多多轮对话状态追踪机制可参考 [/frontier/multi-turn-dialogue/context-management.md](/frontier/multi-turn-dialogue/context-management.md)。

![增量摘要与结构化槽位提取机制图](/images/content-rebuild/ee1237fe42c3713a/fig-3.png)

---

## 高并发读写架构与状态同步

在每秒数百至数千次交互的高并发服务中，必须妥善处理关系数据库持久化与内存缓存的双写时序。

```text
 [客户端提问] 
      │
      ▼
 [API 网关 / 业务服务]
      ├── 1. 校验会话权限并获取自增 seq_id
      ├── 2. 先写主数据库 (PostgreSQL 写入 User 消息)
      ├── 3. 同步写入 Redis 活跃列表 (ZADD 或 RPUSH, 刷新 TTL)
      └── 4. 组装工作上下文 (读 Redis 活跃消息 + 组装 System/摘要)
      │
      ▼
 [大模型流式推理 (SSE)]
      │
      ▼ (接收到完成信号或流式断开)
 [后置落地]
      ├── 5. 写入完整 Assistant 响应至数据库
      └── 6. 追加更新 Redis 缓存
```

### 核心工程权衡
1. **写顺序与缓存失效**：
   采用“先写持久化主库，再更新缓存”的原则。如果 Redis 写入失败，记录告警日志并设置该 Session 缓存失效；后续请求走 Cache-Aside 模式从数据库查询最近记录并重新回填。绝不能以 Redis 为单点权威数据源，以防实例故障导致对话状态丢失。
2. **流式生成的原子性保证**：
   大模型响应通常以 Server-Sent Events（SSE）流式输出。在此期间，**不可**将尚未生成完毕的文字片段实时追加到持久层或共享缓存中。服务端应在内存缓冲完整生成文本，待接收到结束标识（如 `finish_reason: stop`）后，再执行整条 Assistant 消息的数据库写入与缓存更新。异常中断处理可参考 [/interview/ai-system-design-interview/failure-handling.md](/interview/ai-system-design-interview/failure-handling.md)。

![高并发双写读写分离与缓存流转图](/images/content-rebuild/ee1237fe42c3713a/fig-4.png)

---

## 多端同步与分支树设计

用户在实际使用中常有两种破坏线性消息的操作：
* **重新生成（Regenerate）**：对第 3 轮回答不满意，点击重新生成。
* **多端并发提问**：PC 端回答还在流式生成，移动端又发了一条新提问。

扁平的列表结构在处理此类操作时容易发生覆盖或消息错位。为此，消息模型需要引入**单会话递增序列号**与**父指针树状结构**。

### 1. 单会话序列号（Sequence ID）
每个 `session_id` 在 Redis 或发号器中维护一个递增计数器：`INCR session:{session_id}:seq`。
* 每次提交新消息时取得单调递增的 `seq_id`。
* 数据库在 `(session_id, seq_id)` 上建立唯一联合索引。当并发提问发生时，依靠唯一索引或递增序列避免时序倒流。

### 2. 消息分支树（Conversation Tree）
支持重新生成和编辑历史提问时，数据库表需要保存父子指针关系：

```sql
CREATE TABLE session_messages (
    message_id      VARCHAR(64) PRIMARY KEY,
    session_id      VARCHAR(64) NOT NULL,
    parent_id       VARCHAR(64),          -- 指向上一条消息的 message_id，首轮为 NULL
    seq_id          BIGINT NOT NULL,      -- 会话内逻辑递增序号
    role            VARCHAR(16) NOT NULL, -- system | user | assistant | tool
    content         TEXT NOT NULL,
    is_active       BOOLEAN DEFAULT TRUE, -- 是否属于当前选中的主对话分支
    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_session_parent ON session_messages (session_id, parent_id);
CREATE INDEX idx_session_active ON session_messages (session_id, is_active);
```

当用户对第 3 轮回答点击“重新生成”时：
1. 系统生成一条新的 Assistant 消息记录，其 `parent_id` 依然指向第 3 轮的用户提问 `message_id`。
2. 将旧分支上原先的 Assistant 消息的 `is_active` 置为 `FALSE`，新消息置为 `TRUE`。
3. Prompt 装配引擎只需根据最新叶子节点的 `parent_id` 向上回溯，或查询 `is_active = TRUE` 的链路，即可取出当前分支的线性上下文，同时完整保留历史版本以供切换回看。

![会话分支树与父指针追溯机制图](/images/content-rebuild/ee1237fe42c3713a/fig-5.png)

---

## 面试深度追问

### 追问 1：增量摘要生成时如果摘要模型自身发生幻觉，错误在多轮对话中被“滚雪球”放大，工程上如何防范？
**口述要点**：
1. **关键槽位规则校验（Determinism Check）**：
   针对机票、金额、订单号等高敏实体，不依赖模型的自然语言记忆，而是编写代码对比。如果原消息中包含订单号 `CA1204`，但生成的 JSON 变成了 `CA1240`，直接由后置规则校验拦截，强制使用提取阶段的原始字符串覆盖。
2. **自然语言蕴含校验（Entailment Check）**：
   压缩后启动规则检查或轻量分类模型，校验摘要中的陈述能否在对应区间内的原消息中找到依据。若校验不通过，放弃使用该增量摘要，直接降级为“保守截断 + 提示词告警”模式。
3. **指针与区间回溯（Pointer Fallback）**：
   摘要对象中始终记录其替代的原消息范围 `[msg_001, msg_008]`。当后续对话中用户出现质疑（如“我前面说的不是这个时间”）时，系统可通过该区间回溯加载当时的原始消息，修复上下文偏差。

### 追问 2：很多模型已经支持 128K 甚至 1M 的上下文窗口，为什么工程上还要做上下文裁剪与压缩？
**口述要点**：长上下文并不意味着可以无节制传入，主要有三大现实制约：
1. **首字延迟（TTFT）显著增加**：
   长上下文会显著增加 Pre-fill 阶段的计算与显存加载耗时。如果每次交互都传入 10 万 Token 的历史，首字返回时间会从数百毫秒上升至数秒，严重影响人机交互流畅度。
2. **调用成本随轮次累积**：
   商用 API 每次调用均按当次输入的全部 Token 计费。如果不做截断，单次调用的输入大小随轮次线性增长，前 $N$ 轮累计消耗的 Token 总量将按二次方级累积（$\sum_{i=1}^N i = O(N^2)$）。引入窗口与压缩后，单次调用的输入 Token 被限制在常数预算 $B$ 内，累计成本降为线性的 $O(N \cdot B)$。
3. **注意力分散与信息遗忘（Lost in the Middle）**：
   长序列下模型的注意力更容易被大量冗余内容分散。如果上下文中充斥着废弃的工具调用返回和冗长闲聊，模型抓取关键业务规则的准确率会明显下降。详细原理解析可参考 [/fundamentals/inference-and-generation/context-length.md](/fundamentals/inference-and-generation/context-length.md) 与 [/frontier/long-context-app/long-context-pitfalls.md](/frontier/long-context-app/long-context-pitfalls.md)。

![长上下文成本与延迟劣化模型图](/images/content-rebuild/ee1237fe42c3713a/fig-6.png)

---

## 常见误区与反模式

1. **把全量数据库历史无脑拼入 Prompt**：未对持久层与推理上下文进行解耦，会话轮次一多便频繁出现超时或超限报错。
2. **在用户交互主链路上同步做摘要**：用户发送消息后，先等待摘要模型花费 2~3 秒提炼历史，再调用主模型回复。应将摘要任务交由后台异步触发，请求到来时直接读取已生成的上一版摘要快照。
3. **滥用纯向量检索（RAG）替代对话历史**：把每一轮对话切块存入向量库，每次仅靠相似度检索 Top-K 拼入上下文。这会破坏时间线与即时代词（如“它”、“刚才那个”），导致模型失去对话时序感知。
4. **Redis 缓存不做过期时间控制**：全量缓存所有对话且不设 TTL，导致大量非活跃用户会话长久占用高成本内存。活跃缓存应施加滑动过期（例如最后访问后 2 小时释放）。

---

## 推荐阅读与延伸练习

* **前置机制**：[/interview/ai-system-design-interview/rag-agent-workflow.md](/interview/ai-system-design-interview/rag-agent-workflow.md) 了解上下文拼装与知识库检索在架构上的职责划分。
* **网关集成**：[/interview/ai-system-design-interview/multi-model-gateway.md](/interview/ai-system-design-interview/multi-model-gateway.md) 学习多模型调度下的流式传输与配额限流。
* **长文本陷阱**：[/frontier/long-context-app/long-context-pitfalls.md](/frontier/long-context-app/long-context-pitfalls.md) 深入探究大模型超长上下文下的注意力衰减规律与降级方案。
