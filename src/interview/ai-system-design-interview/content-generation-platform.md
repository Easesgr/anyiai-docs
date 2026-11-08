---
title: 如何设计一个 AI 内容生成平台？长文本异步流水线、状态持久化与合规审核
icon: mdi:file-document-plus-outline
tag:
- 系统设计面试
- 内容生成
- 批量生产
description: 以万字研报生成为例，详解长文本平台的异步解耦、DAG阶段编排、检查点断点续跑、双轨机审与DRR公平调度设计。
---

以生成一份 1.5 万字的《新能源汽车出海深度行业研报》为例，用户在界面中输入行业主题、关注维度和参考数据。这类任务无法像即时对话那样在几秒内返回，单次完整生成往往耗时 3 到 8 分钟。

面对这种长任务，如果直接采用传统的同步 HTTP 请求，系统会遭遇一系列工程问题：网关在 60 秒后直接断开并返回 504 Gateway Timeout、移动端切换后台导致 TCP 连接重置、大模型单次生成的输出 Token 达到上限被强行截断，或者在生成接近尾声时偶发网络抖动导致数千 Token 的计算结果与费用全部作废。

设计长文本 AI 内容生成平台，核心是把“向模型发起一次长文本生成请求”，重构为“**异步任务解耦、DAG 分阶段编排、节点检查点持久化、双轨合规机审与公平队列调度**”的工程流水线。

![AI内容生成平台长任务流水线架构概览](/images/content-rebuild/1dd6fc71bba18ce1/cover.png)

---

## 面试口述回答范式

在系统设计面试中被问到“如何设计一个长文本 AI 内容生成平台”时，建议在 90 秒内完成如下结构化回答：

1. **连接与任务解耦**：采用异步轮询与事件推送机制。API 网关接收请求后持久化任务并返回 `task_id`，将任务写入消息队列。Worker 从队列拉取任务执行，执行状态通过 WebSocket 或 SSE 实时推向客户端，避免网关超时与网络闪断中断生成。
2. **多阶段编排（DAG）**：单次模型调用存在最大输出 Token 限制（`max_tokens` 常见为 4096 至 8192），且容易逻辑漂移。系统将长任务拆解为 DAG 工作流：大纲规划 $\rightarrow$ 并行资料检索 $\rightarrow$ 分章节分批撰写 $\rightarrow$ 全文统一润色。各章节通过轻量摘要缓冲区传递承接逻辑。
3. **容错与断点续跑**：DAG 每个执行节点独立打上状态，并将中间产物存入对象存储，写入数据库检查点（Checkpoint）。遇到模型限流（429）或 Worker 故障时，支持单节点重试；任务恢复时跳过已完成节点，避免全量重跑浪费 Token。
4. **安全与资源管控**：合规机审采用“快轨正则/小模型过滤 + 慢轨深度事实核验”双轨机制，单章违规仅触发局部重新生成，不废弃整篇；在并发排队上，采用轻重分流隔离秒级短任务，并在共享池中通过赤字加权轮询（DRR）调度，防止高优长任务持续占用导致普通长任务永久饿死。

---

## 长任务架构解耦：同步长连接到异步事件驱动

大模型依赖自回归方式逐字解码，生成速度受限于硬件算力与网络延迟。一份万字研报需要数千次自回归步长，端到端耗时达数分钟。此时若沿用同步请求-响应模式，中间的反向代理服务器（如 Nginx 默认 `proxy_read_timeout 60s`）会因超时主动切断连接。

![长任务处理架构对比：同步长连接阻塞与异步队列解耦](/images/content-rebuild/1dd6fc71bba18ce1/fig-1.png)

异步解耦方案将任务提交与结果获取拆分为两组独立的交互过程：

1. **任务入队与凭证签发**：前端提交研报生成参数后，API 网关校验用户权限，向 PostgreSQL 写入初始状态为 `PENDING` 的任务记录，随后将任务消息推入消息队列（如 Redis Streams 或 Kafka），立刻向前端返回 HTTP 202 Accepted 以及包含 `task_id` 的查询凭据。
2. **后台消费与执行通知**：调度 Worker 从消息队列中认领任务，初始化执行上下文，并通过 Redis Pub/Sub 发布状态变更事件。
3. **进度同步与断点重连**：客户端通过 WebSocket 或服务器发送事件（SSE，原理可参考 [/fundamentals/inference-and-generation/streaming.md](/fundamentals/inference-and-generation/streaming.md)）监听该任务的执行进度。即使客户端在手机息屏或网络切换时断开连接，后台流水线依然正常运转；客户端重新上线后，调用查询接口传入 `task_id`，即可拉取当前最新快照继续展示。

```python
# 异步任务提交接口典型处理逻辑
@router.post("/api/v1/generation/reports", status_code=status.HTTP_202_ACCEPTED)
async def submit_generation_task(req: GenerationRequest, db: DatabaseSession, queue: TaskQueue):
    task_id = str(uuid.uuid4())
    
    # 1. 持久化任务初始状态，确保凭证有据可查
    await db.execute(
        "INSERT INTO tasks (id, user_id, status, config, created_at) "
        "VALUES (:id, :u, 'PENDING', :c, NOW())",
        {"id": task_id, "u": req.user_id, "c": req.model_dump_json()}
    )
    
    # 2. 将生成任务投入异步消息队列，解耦网关连接
    await queue.publish("generation_tasks", {"task_id": task_id, "user_id": req.user_id})
    
    # 3. 立即响应客户端任务 ID，前端凭此进入轮询或长连接监听
    return {
        "task_id": task_id,
        "status": "PENDING",
        "check_url": f"/api/v1/generation/reports/{task_id}"
    }
```

---

## 多阶段流水线编排：DAG 状态流转

为什么不能用一条提示词要求模型直接写出万字长文？

1. **模型单次输出截断**：即使底层模型支持超长上下文窗口（参考 [/fundamentals/inference-and-generation/context-length.md](/fundamentals/inference-and-generation/context-length.md)），绝大多数商业模型的单次最大生成输出限制（`max_tokens`）仍被硬性限制在 4096 到 8192 Token 之间，无法一次性吐出完整万字结构。
2. **生成漂移与逻辑散乱**：随着单次自回归步数增加，模型对前文指令的遵循能力会逐渐下滑，容易出现反复套话或偏离论点等问题（参考 [/fundamentals/inference-and-generation/generation-pitfalls.md](/fundamentals/inference-and-generation/generation-pitfalls.md)）。
3. **缺乏中间介入能力**：单次黑盒生成无法在中间步骤插入检索信息、执行合规校验或让人工核对大纲。

平台将长文本撰写拆解为有向无环图（DAG）工作流编排（关于工作流与智能体的边界划分，可参考 [/interview/ai-system-design-interview/rag-agent-workflow.md](/interview/ai-system-design-interview/rag-agent-workflow.md)）：

![基于DAG的多阶段长文本生成流水线编排](/images/content-rebuild/1dd6fc71bba18ce1/fig-2.png)

以《新能源汽车出海深度行业研报》为例，DAG 的流转过程如下：

- **阶段一：大纲规划（Planning）**：模型根据用户主题生成结构化 JSON 大纲，划分为“欧洲政策壁垒、东南亚基建现状、南美供应链本土化”等独立章节，明确各章节核心论点与篇幅规划。
- **阶段二：检索与材料组装（Research）**：各个章节并行发起检索增强（RAG），分别拉取对应的关税数据、充电桩覆盖率等外部权威资料。
- **阶段三：分章节撰写（Chapter Generation）**：
  - 针对相互独立的分析章节（如“欧洲市场”与“东南亚市场”），调度 Worker 并行撰写；
  - 针对存在前后承接关系的章节（如“现状分析”与“未来趋势预测”），下游章节启动时只接收上游章节由摘要器压缩后的**滚动上下文摘要缓冲区**（Rolling Summary Buffer），既保持行文衔接，又避免将前文数万字原文全部重复塞入 Prompt。
- **阶段四：全文润色（Polishing）**：收集所有章节后，由审查节点比对全文专有名词、术语缩写与过渡句，消除多章节分别撰写导致的语调分歧。
- **阶段五：排版导出（Rendering）**：将各章节拼接并排版，转为带目录与附注的 Markdown、PDF 或 Word 文件。

<ReadMoreLock />

---

## 检查点（Checkpoint）恢复与断点续跑

在生成长研报的第 4 章节时，Worker 可能发生进程崩溃，或者上游大模型 API 突然返回 429 请求超限（详见 [/interview/ai-system-design-interview/failure-handling.md](/interview/ai-system-design-interview/failure-handling.md)）。若直接推倒重来，前面章节消耗的模型费用与等待时间将全部损失。

![DAG执行检查点持久化与故障恢复机制](/images/content-rebuild/1dd6fc71bba18ce1/fig-3.png)

平台通过**节点级检查点（Checkpoint）**保障故障恢复：

1. **节点执行原子性与持久化**：DAG 的每个节点分配唯一的 `node_key`（如 `task_id + stage_name + chapter_index`）。节点执行完毕后，生成的正文内容写入对象存储（如 MinIO 或 S3），并在 PostgreSQL 中将该节点状态标记为 `COMPLETED`，记录输出文件的存储路径。
2. **重试退避与异常标记**：遇到上游限流或网络闪退时，节点采用带随机抖动的指数退避（Exponential Backoff with Jitter）自动重试。若连续重试耗尽，系统只把当前节点置为 `FAILED`，挂起依赖它的后续节点，已完成的前置章节保持不变。
3. **断点重新拉起**：用户在界面点击“重试”或系统重新调度该任务时，DAG 引擎重新加载该任务的状态快照，识别出所有状态为 `COMPLETED` 的节点并直接跳过，仅从发生失败的节点开始恢复执行。

```json
{
  "task_id": "task_20260408_0982",
  "status": "RUNNING",
  "current_stage": "chapter_generation",
  "nodes": {
    "outline_generation": {"status": "COMPLETED", "output_ref": "s3://drafts/task_0982/outline.json"},
    "chapter_1_policy": {"status": "COMPLETED", "output_ref": "s3://drafts/task_0982/ch_1.md"},
    "chapter_2_infrastructure": {"status": "COMPLETED", "output_ref": "s3://drafts/task_0982/ch_2.md"},
    "chapter_3_supply_chain": {"status": "FAILED", "retry_count": 3, "error": "Model API 429 Rate Limit"},
    "chapter_4_forecast": {"status": "WAITING_DEPENDENCY", "output_ref": null}
  }
}
```

---

## 双轨合规机审流水线：风控与事实准确性

内容生成平台需要满足严格的合规审计与风控要求。如果等整篇 1.5 万字全部写完再统一审核，一旦末尾出现敏感词或违规内容导致驳回，前序生成的开销全被废弃；如果让每个词的生成都经过大模型深度审查，整体耗时又会翻倍。

平台采用**双轨合规校验流水线**：

![双轨合规机审流水线：离线在线分级与局部降级隔离](/images/content-rebuild/1dd6fc71bba18ce1/fig-4.png)

### 1. 双轨机审分工

- **快轨（Fast Track，在线即时拦截）**：基于 DFA 敏感词树过滤与轻量微调分类小模型（处理延迟通常在 50 毫秒以内）。在用户输入阶段校验 Prompt，并在每个章节刚生成完毕时立即进行违规过滤。命中政治、暴恐或个人隐私信息时直接拦截该节点。
- **慢轨（Deep Track，异步深度核验）**：在章节写入存储后，通过独立的后台工作流启动事实核查组件。比对生成文本中出现的销量数据、关税百分比是否与检索到的参考来源相符，识别幻觉并生成批注。

### 2. 章节级局部隔离与定向重写

当第三章节“供应链本土化”触发快轨敏感词或事实核验不通过时：
- **隔离污染源**：系统保留第一、二章节的合格内容，将第三章打标为 `FLAGGED_CONTENT`，阻断其向后续章节传递上下文摘要，防止敏感内容污染后续生成。
- **定向重写与降级**：平台向模型注入包含安全负向约束的专用重写提示词，仅要求模型重写第三章。若连续重试仍未通过，系统向客户端推送章节审核异常事件，界面标红该章节并提示用户手动修改该章大纲，而非直接废弃整份研报。

---

## 版本管理：草稿历史树与分章差异对比

长文本生成通常是人机协同修改的过程。用户看毕初稿后，可能会手动精修第二章中的若干论点，随后要求平台“基于我修改后的第二章，重新生成第三章和第四章”。

为支撑这种回溯与衍生场景，底层文档存储采用类似版本控制的**草稿历史树（Revision Tree）**，而不是直接覆盖数据库记录：

![草稿历史树与分章差异比对数据结构](/images/content-rebuild/1dd6fc71bba18ce1/fig-5.png)

### 数据模型与增量记录

文档对象由指向版本树节点的指针维护，每次重新生成章节或人工保存编辑时，均派生出一个新的版本节点：

- **版本快照（Snapshot）**：记录当前版本包含的章节 ID 列表及其内容散列值（Content Hash）。未被修改的章节（如第一章）直接复用旧版本的对象存储链接，避免存储空间膨胀。
- **字符级差异比对（Diff）**：对人工微调过的段落，采用文本差分算法记录局部变更，前端可直观展示增删改对比高亮。
- **段落溯源元数据（Attribution Metadata）**：精准标记每段文本的生成来源（例如标明由哪个模型版本、哪套提示词生成，或是具体哪位操作者在何时手工键入），为后续文本质量评估（可参考 [/fundamentals/evaluation-metrics/generation-metrics.md](/fundamentals/evaluation-metrics/generation-metrics.md)）提供审计依据。

---

## 面试追问：万级长任务并发下的公平调度

### 追问场景

> **面试官提问**：“如果平台突然涌入大量生成任务，其中既有免费用户的万字研报生成（单任务耗时 8 分钟），又有付费 VIP 用户的单页营销短文案任务（单任务耗时 15 秒）。如果直接放入同一个先进先出（FIFO）队列，VIP 的短任务会被前面的万字长任务长时间阻塞。如果采用绝对优先级队列，只要 VIP 任务持续进入，免费用户的长任务就会完全卡死。你如何设计调度机制保证高优任务响应，同时避免低优长任务饿死？”

![基于赤字加权轮询（DRR）的混合优先级公平队列调度机制](/images/content-rebuild/1dd6fc71bba18ce1/fig-6.png)

### 方案：赤字加权轮询（Deficit Round Robin, DRR）调度

平台在调度层建立多级队列，结合任务成本预估与**赤字加权轮询（DRR）**算法，兼顾优先级差异与公平性。

1. **队列拆分与轻重分流**：
   - **轻重任务物理隔离**：对预估耗时低于 30 秒的短文案任务分流至轻量 Worker 池独立消费，避免与多阶段长任务争抢槽位，解决短任务被长任务队头阻塞的问题；
   - **长任务多级队列**：对共享重型生成资源的万字研报等长流程，划分为 **VIP 高优队列**（权重 $W_{vip} = 3$）与 **标准队列**（权重 $W_{norm} = 1$）。
2. **任务成本度量**：
   - 根据大纲章节数、检索分支数以及目标字数折算长任务预估成本点数（Cost Units, 记为 $C$）。例如普通万字研报基准成本核定为 $C = 4$；若 VIP 用户提交更高精度的深度长研报，成本可核定为 $C = 5$。
3. **DRR 算法执行流程**：
   - 调度器维护轮询周期。每个队列各自分配一个赤字计数器（Deficit Counter, $DC$）。每轮调度开始时，调度器为每个队列补充固定量子点数：$DC_i = DC_i + Quantum \times W_i$。
   - 调度器检查队头任务的预估成本 $C$。只要当前队列的 $DC_i \ge C$，就将队头任务弹出并分发给后台 Worker 执行，同时扣减赤字：$DC_i = DC_i - C$，并继续检查下一个任务；直到队列为空或队头任务成本 $C > DC_i$ 时，保留结余赤字 $DC_i$ 留待下轮累加，调度器转向下一个队列。

#### 调度算例模拟（数值示意）

设定基础调度配额 $Quantum = 2$。
- VIP 队列权重 $W_{vip} = 3$，每轮补充点数 $2 \times 3 = 6$；队头任务成本 $C = 5$。
- 普通队列权重 $W_{norm} = 1$，每轮补充点数 $2 \times 1 = 2$；队头任务成本 $C = 4$。

| 轮次 | 队列分类 | 轮初补充点数 | 当前可用赤字 ($DC$) | 队头任务成本 ($C$) | 调度决策 | 轮末结余赤字 ($DC$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 第 1 轮 | VIP 队列 | +6 | 6 | 5 | **满足条件，弹出执行** | $6 - 5 = 1$ |
| 第 1 轮 | 普通队列 | +2 | 2 | 4 | 赤字不足 ($2 < 4$)，跳过等待 | 2 |
| 第 2 轮 | VIP 队列 | +6 | $1 + 6 = 7$ | 5 | **满足条件，弹出执行** | $7 - 5 = 2$ |
| 第 2 轮 | 普通队列 | +2 | $2 + 2 = 4$ | 4 | **满足条件 ($4 \ge 4$)，弹出执行** | $4 - 4 = 0$ |

在这一机制下，高优先级的 VIP 队列享有更高的吞吐调度比例，但在普通队列累积两轮赤字后，其长任务必然会获得调度槽位，避免了静态高优先级导致的永久饥饿。

---

## 常见设计陷阱与反模式

| 常见错误实现方式 | 潜在系统风险与故障表象 | 系统设计推荐方案 |
| :--- | :--- | :--- |
| **同步 HTTP 长连接等待全部生成** | 反向代理 504 超时，移动端网络切断后无法重连，造成已消耗 Token 无法结算和展示。 | 接口返回 HTTP 202 异步签发 `task_id`，配合消息队列缓冲，由 WebSocket 或 SSE 推送进度。 |
| **无拆解单次超长 Prompt 撰写** | 触碰单次模型最大输出 Token 上限导致末尾截断，上下文过长引起逻辑混乱和格式丢失。 | 采用 DAG 流水线：结构大纲规划 $\rightarrow$ 并行资料检索 $\rightarrow$ 分章节撰写 $\rightarrow$ 全局润色。 |
| **全部产出生成完毕后再做合规审查** | 仅因末尾章节出现违规导致数万字初稿整体报废，返工时间与模型费用极高。 | 双轨分级审核：快轨即时阻断高危词，章节级违规实施局部上下文隔离与定向安全重写。 |
| **执行报错时直接推倒从头重新生成** | Worker 临时网络抖动或限流导致整篇全流程重跑，不仅加剧队列积压，还会重复扣费。 | 节点级别检查点（Checkpoint）持久化，故障恢复时跳过已成功节点，断点续跑。 |
