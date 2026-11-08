---
title: 长任务如何可靠执行？检查点、幂等重试与人机协同
icon: mdi:timer-sand
description: 解析长周期 Agent 任务的状态管理机制，涵盖图状态机建模、检查点快照存储、基于幂等键的写操作隔离、人工挂起恢复与分布式租约回收。
---

在由大语言模型驱动的复杂应用中，任务执行周期常常从普通的毫秒级接口调用，拉长至数分钟乃至数天。例如一个“企业采购报销与自动打款”任务，需要先拉取发票、由模型进行合规审查、核对采购流水，随后调用银行网关打款，若金额超限还需等待财务经理审批，最后发送通知邮件。

在[异步任务](/architecture/async-tasks.md)中，我们通过消息队列解决了请求削峰与服务解耦。但在多步长任务中，如果 Worker 节点在第 5 步因为宿主机维护、容器 OOM（内存溢出）或网络闪断而崩溃，直接依靠队列从第 1 步整体重跑，会带来两类严重后果：第一，前面数次大模型推理已经消耗的 Token 与响应时间全额作废；第二，如果第 4 步已经调用了打款网关，无防护的重跑会导致重复打款。

要解决长任务的执行稳定性，核心是将易失的内存计算流，转换为由状态机驱动、在逻辑步骤间持久化快照的受控流程。

![长执行任务在故障中断时的全量重跑与检查点定点恢复对比](/images/content-rebuild/7e5d6ee48253687b/cover.png)

---

### 长任务执行面临的系统问题

当单次任务运行时间跨越数分钟到数小时，分布式环境中的常规波动更容易在任务执行周期内出现。长任务通常需要应对四类故障：

1. **运行实例丢失**：容器调度、云厂商抢占式实例回收、节点 OOM 会直接销毁内存中的执行调用栈。
2. **外部依赖超时与限流**：模型接口触发 Rate Limit、第三方 SaaS 服务故障或数据库响应慢，导致单步调用异常。
3. **非幂等副作用中断**：任务通常交替执行“只读查询”与“外部写入”（如调用支付接口、发送邮件）。若崩溃发生在写入完成之后、状态提交之前，恢复时极易出现重复写入。
4. **上下文膨胀引发的漂移与超限**：多轮循环如果无条件累积完整历史，会推高单步推理延迟与成本，甚至超出模型上下文窗口，相关控制手段可参考[上下文管理](/architecture/context-management.md)。

![长任务执行面临的四类典型故障场景与中断点分析](/images/content-rebuild/7e5d6ee48253687b/fig-1.png)

---

### 基于图与状态机的任务建模

现代长任务编排系统（如 LangGraph）普遍采用受 Pregel 计算模型启发的状态图（State Graph）来建模多步流程。

<ReadMoreLock />

#### 1. 状态定义与增量更新（Reducer）
任务执行期间维护一个强类型的数据结构（如 Python TypedDict 或 Pydantic 模型）。节点本身不直接修改外部全局变量，而是接收当前状态作为输入，返回状态更新的增量字典（Delta）。增量如何合并由 Reducer 规则决定：

* **覆盖更新（Replace）**：适合标量字段。例如 `current_step: int`，后一步的输出直接替换前一步。
* **追加更新（Append）**：适合列表字段。例如对话消息列表 `messages: list`，每个节点返回新增的消息对象，底层状态机将其自动连接到原有列表末尾。

#### 2. 确定性节点与非确定性推理的边界
工程实现中，长任务节点通常分为两类：
* **确定性逻辑节点**：执行纯规则校验、数据清洗或结构转换等无外部状态依赖的代码。在输入状态固定且无外部环境干扰时，执行逻辑与输出是确定的。若节点包含外部 SQL 读写或网络 I/O，仍需面对网络抖动、并发冲突和外部状态变化。
* **非确定性 Agent 节点**：调用大语言模型进行规划或生成。模型受采样温度和生成机制影响，输出存在波动。

通过将确定性边界插入非确定性节点之间，状态图可以确保即使模型决策出现随机性，后续的数据格式校验与业务屏障依然按既定规则拦截异常输入，防止错误在后续步骤扩散。

![图状态机中确定性规则节点与非确定性LLM节点的交错拓扑结构](/images/content-rebuild/7e5d6ee48253687b/fig-2.png)

---

### Checkpointer 检查点机制

检查点（Checkpointer）用于在状态图的每个超级步（Superstep）执行结束时，将当前状态对象序列化并写入外部持久化存储（如 PostgreSQL、SQLite 或 Redis）。

#### 1. 检查点的数据结构
一条检查点记录通常包含以下字段：
* `thread_id`：标识业务任务的唯一主键（如 `report_task_10029`）。
* `checkpoint_id`：按时序单调递增的版本号或时间戳 UUID。
* `parent_checkpoint_id`：上一版本检查点的 ID，通过指针关联形成一条有向无环的版本链表或版本树。
* `state_data`：序列化后的状态载荷（JSONB 或二进制字节流）。
* `next_nodes`：当前状态下后续计划执行的节点名称列表。

#### 2. 断点恢复与时光倒流（Time Travel）
因为检查点采用追加写入（Append-only）的历史版本链，系统具备两项基础能力：
* **定点续跑（Resume）**：Worker 崩溃重启后，根据 `thread_id` 查询最新的检查点记录，还原状态对象与待执行节点列表，直接从中断处继续运行，省去前置全部节点的重跑开销。
* **分支回滚与人工纠偏（Time Travel）**：如果 Agent 在第 4 步产生幻觉调用了错误工具，操作员可以把指针重新指向第 3 步的 `checkpoint_id`，手动修改状态中的参数字段，派生出一个新的子检查点继续执行。旧的错误分支保留用于复盘与审计，新分支则按照修正后的数据正常推进。

![Checkpointer利用版本树实现状态快照保存与历史节点回溯](/images/content-rebuild/7e5d6ee48253687b/fig-3.png)

---

### 外部调用的副作用隔离与幂等设计

检查点仅能恢复程序内存中的状态变量。如果节点包含外部写操作（如调用打款网关），仅靠检查点重跑依然可能造成外部操作重复执行。因此，系统必须结合幂等机制进行防护。

#### 1. 外部操作类型划分
* **只读操作（无外部副作用）**：如查询向量数据库、拉取用户资料。崩溃后无条件重新执行，不会改变系统外部状态。
* **天然幂等写操作**：如基于固定路径覆盖写入对象存储、按唯一业务键执行数据库 `INSERT ... ON CONFLICT DO UPDATE`（Upsert）。即使重复执行多次，最终系统状态与执行单次一致。
* **非幂等外部写操作**：如发送短信、邮件、调用第三方免密支付接口扣款。这类接口如果收到多次请求，会产生重复扣款或多次推送。

#### 2. 幂等键（Idempotency Key）屏障
对于非幂等外部写操作，Worker 必须由任务上下文派生出全局唯一的幂等键：
$$\text{IdempotencyKey} = \text{Hash}(\text{thread\_id} + \text{checkpoint\_id} + \text{step\_name} + \text{action\_id})$$

在发起外部请求时，该幂等键随请求一并发送给外部服务（或本地幂等记录表）。如果 Worker 在写入执行完毕后、检查点写入数据库前崩溃，新拉起的 Worker 重试该步骤时，外部服务（或拦截网关）通过该幂等键直接返回上一次处理成功的结果，避免实际发生二次扣款。

以下为一个包含检查点持久化与幂等屏障的最小验证实现：

```python
import json
import uuid
from typing import Dict, Any, Optional

class SimpleCheckpointer:
    def __init__(self):
        # 存储格式：{thread_id: {checkpoint_id: 快照内容}}
        self.storage: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # 模拟外部服务或本地的幂等账本：{idempotency_key: 返回凭证}
        self.idempotency_records: Dict[str, str] = {}

    def save_checkpoint(self, thread_id: str, checkpoint_id: str, state: Dict[str, Any], next_node: Optional[str]):
        if thread_id not in self.storage:
            self.storage[thread_id] = {}
        # 深拷贝存储，避免后续就地修改污染历史快照
        self.storage[thread_id][checkpoint_id] = {
            "state": json.loads(json.dumps(state)),
            "next_node": next_node
        }

    def load_latest(self, thread_id: str) -> Optional[Dict[str, Any]]:
        if thread_id not in self.storage or not self.storage[thread_id]:
            return None
        latest_id = sorted(self.storage[thread_id].keys())[-1]
        return {"checkpoint_id": latest_id, **self.storage[thread_id][latest_id]}

def call_payment_gateway(idempotency_key: str, amount: int, checkpointer: SimpleCheckpointer) -> str:
    # 1. 幂等校验屏障：命中则直接返回已有交易流水
    if idempotency_key in checkpointer.idempotency_records:
        return f"[幂等命中-复用流水] tx_id: {checkpointer.idempotency_records[idempotency_key]}"
    
    # 2. 执行真正的扣款副作用
    generated_tx = f"tx_{uuid.uuid4().hex[:8]}"
    checkpointer.idempotency_records[idempotency_key] = generated_tx
    return f"[真实扣款成功] 金额: {amount}, tx_id: {generated_tx}"

def execute_task(thread_id: str, checkpointer: SimpleCheckpointer, trigger_crash: bool = False):
    # 尝试恢复最新检查点
    snapshot = checkpointer.load_latest(thread_id)
    if snapshot:
        state = snapshot["state"]
        current_step = snapshot["next_node"]
        print(f"-> 成功从检查点 {snapshot['checkpoint_id']} 恢复，待执行节点: {current_step}")
    else:
        state = {"payment_done": False, "logs": []}
        current_step = "step_payment"
        print("-> 未发现历史快照，启动全新任务")

    # 步骤 1：调用扣款接口并持久化检查点
    if current_step == "step_payment":
        print("执行节点: step_payment (调用外部支付)")
        # 派生该业务动作唯一幂等键
        idempotency_key = f"{thread_id}_payment_step"
        pay_result = call_payment_gateway(idempotency_key, 200, checkpointer)
        
        state["logs"].append(pay_result)
        state["payment_done"] = True
        print(f"   接口返回: {pay_result}")
        
        # 落盘第 1 步检查点，指向下一步 step_notify
        checkpointer.save_checkpoint(thread_id, "chk_001", state, next_node="step_notify")
        
        if trigger_crash:
            print("   [模拟故障] Worker 触发异常/容器被销毁，进程中断退出！")
            return
        current_step = "step_notify"

    # 步骤 2：发送完成通知
    if current_step == "step_notify":
        print("执行节点: step_notify (发送确认通知)")
        state["logs"].append("通知邮件已送达")
        checkpointer.save_checkpoint(thread_id, "chk_002", state, next_node=None)
        print("任务全部完成，最终状态落盘。")
```

运行测试流程，验证故障恢复：

```python
store = SimpleCheckpointer()
task_id = "task_order_2026_03"

# 第一次运行：执行完扣款后模拟进程崩溃
print("=== 第一次执行（模拟崩溃） ===")
execute_task(task_id, store, trigger_crash=True)

# 第二次运行：新实例启动，恢复相同 task_id
print("\n=== 第二次执行（故障恢复） ===")
execute_task(task_id, store, trigger_crash=False)
```

执行输出表现如下：
```text
=== 第一次执行（模拟崩溃） ===
-> 未发现历史快照，启动全新任务
执行节点: step_payment (调用外部支付)
   接口返回: [真实扣款成功] 金额: 200, tx_id: tx_3f8a19b2
   [模拟故障] Worker 触发异常/容器被销毁，进程中断退出！

=== 第二次执行（故障恢复） ===
-> 成功从检查点 chk_001 恢复，待执行节点: step_notify
执行节点: step_notify (发送确认通知)
任务全部完成，最终状态落盘。
```
第二次执行直接根据 `chk_001` 的记录进入 `step_notify`，避免了重复调用支付接口。即便网络异常导致检查点落盘滞后而重跑了第一步，幂等键机制也会阻止二次扣款。

![带有幂等键屏障的外部写操作重试保护机制](/images/content-rebuild/7e5d6ee48253687b/fig-4.png)

---

### 人工介入（Human-in-the-loop）挂起与恢复

部分关键步骤不能完全交给模型自主决策（例如单笔转账超过一万元、线上生产库 DDL 变更）。系统需要支持挂起任务，等待人工审查。

#### 1. 中断机制（Interrupt）
在同步调用模式中，等待人工审批通常需要工作线程通过 Sleep 轮询，这会导致数据库连接池被长时间占满，计算节点无法释放。在图状态机中，中断被实现为受控挂起：
* **预设中断（Static Interrupt）**：在图构建时声明 `interrupt_before=["execute_transfer"]`。当状态机流转到该节点前，调度器自动保存当前检查点，将任务在调度队列中标记为 `SUSPENDED`，当前 Worker 立即退出执行循环，释放线程与内存。
* **动态中断（Dynamic Interrupt）**：节点在执行内部逻辑时，如果检测到风险特征（如模型给出的置信度过低或退款金额超出阈值），主动抛出框架支持的中断信号（如抛出特定异常），调度器截获后保存当前状态并安全挂起。

#### 2. 人工审查与恢复路径
任务挂起后，系统将状态快照暴露给前端管理系统。人工接入通常提供三类处理途径：
1. **直接放行（Approve）**：审核人核验无误，调用恢复接口（如 `graph.invoke(None, config={"thread_id": id})`）。执行引擎重新加载该 `thread_id` 最新的检查点，继续调度后续节点。
2. **状态修正（State Edit）**：审核人发现模型给出的 SQL 语句中表名存在轻微偏差，在前端直接编辑修改 `sql_query` 字段。更新后的内容写入新的检查点版本，后续节点以人工修正后的状态作为输入继续执行。
3. **驳回与重规划（Reject）**：审核人拒绝执行并附带驳回意见。驳回原因被写入状态消息列表，条件边将流程路由回 Agent 规划节点，促使模型阅读驳回原因后重新规划。

![人工介入机制中的任务挂起、状态审查编辑与恢复执行时序图](/images/content-rebuild/7e5d6ee48253687b/fig-5.png)

---

### 租约机制与僵死任务回收

在分布式部署中，同一个长任务在同一时间只能被一个 Worker 推进。如果某 Worker 在执行长任务过程中遭遇硬件故障断电，系统需要有机制识别它并非在“长时间计算”，而是“已经死亡”，以便由其他健康节点接管。

#### 1. 分布式锁与心跳租约（Lease）
Worker 认领长任务时，必须在集中式存储（如 Redis 或数据库）中获取带有存活时间（TTL）的分布式租约：
```text
SET lock:task:{thread_id} {worker_instance_id} NX EX 30
```
* 认领成功后，Worker 内部启动后台心跳协程，每隔固定周期（如 10 秒）对锁进行续租。
* 只要 Worker 正常工作，锁就不会失效。

#### 2. 看门狗巡检（Watchdog）与毒丸熔断
独立运行的调度看门狗（Watchdog）进程定期扫描数据库中处于 `RUNNING` 状态、但其对应分布式锁已经过期的任务：
* **任务回收重派**：锁过期表明原 Worker 已经失联。Watchdog 将任务状态置回 `PENDING`，交由队列调度给其他空闲 Worker。新 Worker 启动后根据检查点机制定点恢复。
* **毒丸（Poison Pill）熔断**：如果特定任务在同一步骤连续导致 3 次 Worker 崩溃超时，通常说明任务输入数据存在触发底层库崩溃的严重 Bug（例如导致 C++ 扩展崩溃的畸形图片）。Watchdog 必须拦截此类任务，将其状态置为 `FAILED_DEAD_LETTER` 并发出报警，阻止其无限重试消耗计算资源。

![长任务分布式锁租约刷新、超时驱逐与Watchdog自动回收机制](/images/content-rebuild/7e5d6ee48253687b/fig-6.png)

---

### 架构权衡与指标验收

长任务持久化机制引入了额外的存储 I/O 延迟与系统复杂度，在不同场景下的选型考量如下：

| 维度 | 无状态执行流 | 检查点持久化执行流 | 选型与边界建议 |
| :--- | :--- | :--- | :--- |
| **每步存储开销** | 纯内存流转，延迟基本忽略 | 涉及 JSON/二进制序列化与网络写入，每步增加 10~50ms 延迟 | 单轮即时交互通常无需持久化；执行耗时长、单步成本高或包含外部副作用与人工复核的长任务建议开启 |
| **故障重试成本** | 失败必须全量重新执行，重新消耗 Token 与接口时间 | 定点恢复，仅需重跑最后未落盘的一个逻辑步骤 | 执行链路越长、单步模型调用耗时越高，检查点带来的收益越明显 |
| **外部写安全性** | 极易因为重复执行导致二次扣款或重复发信 | 依赖幂等键机制与检查点边界进行隔离 | 涉及真实资产流动或外部通知等业务，必须搭配幂等键设计 |
| **人工介入能力** | 只能靠线程阻塞（Sleep）等待，资源消耗大 | 支持安全挂起释放计算资源，支持恢复与状态修改 | 包含人工复核审批周期的流程，必须使用状态机挂起方案 |

#### 生产可观测性验收指标
1. **检查点落盘延迟（Checkpoint Latency P99）**：单个状态写入数据库的耗时。若该指标异常偏高，需排查状态对象中是否误放入了超大文本或图片 Base64 编码（大体积数据应上传至对象存储，状态中仅保留 URI 引用）。
2. **僵死接管耗时（Zombie Recovery Time）**：从 Worker 实例物理宕机，到 Watchdog 成功将该任务在新节点拉起执行的时间差。该指标用于校准分布式锁 TTL 与 Watchdog 扫描周期的配比。
3. **断点恢复成功率（Resume Success Rate）**：发生意外中断后，能够成功根据最后一个检查点继续推进并完成的任务占比。该指标体现了业务节点纯净度与外部接口幂等键方案的完备程度。
