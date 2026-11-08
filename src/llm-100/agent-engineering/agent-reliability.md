---
title: Agent 怎么在生产环境中保持稳定？循环终止、幂等控制与评测设计
icon: mdi:book-open-page-variant
description: 剖析 Agent 线上失控的常见诱因，详解运行时循环熔断、写工具幂等控制、上下文分层精简以及基于沙箱终态的自动化评测方案。
---

在前面的讨论中，我们了解了 [工具调用失败的防御降级](/llm-100/agent-engineering/agent-tool-failure.md) 以及 [执行轨迹与中间状态的观测追踪](/llm-100/agent-engineering/agent-execution-tracing.md)。很多开发者在本地编写 Agent 演示时运行顺畅，但一旦接入真实业务让其多步自主决策，系统就容易暴露各类稳定性问题：工具返回非预期数据时模型陷入重复尝试死循环、网络抖动重试导致外部接口被重复写入、上下文迅速膨胀导致推理费用飙升且偏离初始目标。

大模型本质上是一个基于概率采样的预测引擎，并不具备天然的确定性控制保证。因此，**生产级 Agent 的稳定性不能押注在“模型每一次都推理正确”上，而必须由外层的宿主调度程序提供确定性的边界与保护机制**。

我们以一个具体的业务场景作为贯穿案例：**电商售后退款 Agent**。该 Agent 接收用户诉求，自主调用“查询订单详情”、“核验退款政策”、“调用支付网关退款”以及“发送通知短信”四个工具。我们将结合这一案例，拆解系统失控的诱因，并落地循环熔断、幂等防护、上下文治理和端到端评测。

![Agent 生产稳定性保障框架概览](/images/content-rebuild/7e10c3b349027e52/cover.png)

---

## 生产环境中 Agent 常见的失控过程

在由模型自主决定工具选择与参数的循环中，单次格式解析错误通常容易通过 Pydantic 校验或类型转换捕获。更难排查的是跨轮次累积的逻辑异常，常见过程可归纳为三类：

### 1. 错误级联（Error Cascading）
在退款案例中，如果“查询订单详情”接口由于网络短暂抖动返回了空字典 `{}`，而工具没有抛出明确的业务异常说明，模型可能会在下一步假设“该用户没有历史订单记录”，并以此错误认知为基础继续推理，回复用户“无权申请退款”。前一步的观察残缺被模型当成了客观事实，导致后续所有动作步步偏离业务目标。

### 2. 动作死循环（Action Looping）
当模型发现某一工具没有返回它想要的结果时，如果没有外部干预，它很容易陷入局部最优陷阱。例如，模型调用“查询退款政策”工具传入参数 `category='electronics'` 返回了未查到，它可能在下一轮仅仅微调大小写为 `category='Electronics'`，再次失败后又尝试 `category='electronic'`。几轮下来除了白白消耗 API 配额和拉长响应耗时，任务毫无实质进展。

### 3. 上下文状态膨胀（State Explosion）
Agent 每次调用工具的输入输出通常都会被追加到对话上下文中。如果“查询订单详情”工具把包含几百条商品 SKU、优惠券明细、历史物流跟踪日志的庞大原始 JSON（甚至上万字符）原样塞进消息列表，上下文长度会呈线性甚至超线性增长。这不仅会带来显存占用上升和每次调用成本增加，还会导致模型对较早前系统提示（System Prompt）中的约束规则关注度下降，产生指令漂移。

![Agent 崩溃链路因果示意图](/images/content-rebuild/7e10c3b349027e52/fig-1.png)

上图反映了三类现象之间的传导关系：工具异常或边界输入诱发模型产生偏差认知，偏差认知引发无效的重复动作，而无休止的调用往复又导致历史记录迅速累积，最终耗尽 Token 配额或触发超时。要阻断这一链条，调度层必须建立硬性的运行时拦截。

<ReadMoreLock />

---

## 循环硬终止机制：步数、重复动作与超时熔断

无论使用纯 Python 自研调度循环，还是基于 [LangGraph 可控工作流](/llm-100/agent-engineering/langgraph-controlled-workflow.md) 编排，都不能依赖模型自身的推理来决定何时完全停下。调度器必须在外部配置三道确定性的物理检查：

### 1. 最大递归步数（Max Steps）
设置一个硬性计数器。例如单次售后会话最多允许执行 10 步工具调用。一旦计数器达到上限，调度循环立刻中断，将任务置为挂起或转人工状态，防止无限制消耗 Token。

### 2. 规范化动作签名与重复检测（Loop Detector）
为了在耗尽最大步数前及早制止死循环，调度器需要记录最近执行的动作。由于大模型输出的 JSON 字典中键值顺序可能每次不同，直接比较字符串并不稳定。工程上应将工具名称与参数进行键排序（Canonical JSON），再计算哈希：

$$\text{ActionSignature} = \text{SHA256}(\text{tool\_name} + \text{json\_dumps\_sorted}(\text{tool\_args}))$$

调度器维护一个固定大小的滑动窗口（如长度为 3）。若检测到同一个签名连续命中阈值，调度程序有两种应对策略：
*   **软干预**：向上下文注入一条系统提示：“系统检测到你正在使用完全相同的参数反复调用该工具且未获进展，请更换工具尝试其他路径，或向用户说明无法完成的客观原因。”
*   **硬熔断**：若下一轮模型依然生成相同动作，立即终止调度并返回错误。

### 3. 物理耗时熔断（Wall-clock Timeout）
外部网络请求的延迟难以完全预测。整个 Agent 任务必须绑定整体超时时间（如 60 秒）。一旦超时，通过异步任务取消机制（如 `asyncio.wait_for`）强制中断模型调用与工具执行，回收连接池资源。

```python
import hashlib
import json
from typing import Any, Dict, List


def generate_action_signature(tool_name: str, tool_args: Dict[str, Any]) -> str:
    # 对字典键进行排序序列化，避免因字典无序导致哈希比对失效
    payload = json.dumps(
        {"tool": tool_name, "args": tool_args}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ExecutionGuard:
    def __init__(self, max_steps: int = 10, loop_threshold: int = 3):
        self.max_steps = max_steps
        self.loop_threshold = loop_threshold
        self.step_count = 0
        self.recent_signatures: List[str] = []

    def check_and_record(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        self.step_count += 1
        if self.step_count > self.max_steps:
            return "STOP_MAX_STEPS_EXCEEDED"

        sig = generate_action_signature(tool_name, tool_args)
        self.recent_signatures.append(sig)

        # 检查滑动窗口内是否连续出现完全相同的动作签名
        if len(self.recent_signatures) >= self.loop_threshold:
            window = self.recent_signatures[-self.loop_threshold :]
            if len(set(window)) == 1:
                return "STOP_LOOP_DETECTED"

        return "PROCEED"
```

![循环硬终止机制与守卫判定](/images/content-rebuild/7e10c3b349027e52/fig-2.png)

如图所示，调度程序在每一轮真正分发工具调用前，先按顺序执行物理时钟、最大步数以及动作签名去重判定。这一系列检查是纯代码逻辑，不依赖模型理解，构成了系统的第一道安全边界。

---

## 工具调用的幂等性设计：防重复执行与状态锁定

在退款案例中，调用“查询订单详情”是无副作用的读操作（Read Operation），即便重复调用也只是多耗费查询时间；但“调用支付网关退款”是典型的有副作用的写操作（Write Operation）。

如果网络出现偶发超时，或者调度重试机制介入，同一个退款请求可能被重复下发。**若底层写工具缺乏幂等控制，重复调用可能直接导致多次扣款或资金重复退回。**

### 1. 幂等键（Idempotency Key）的派生机制
写操作工具不能依靠大模型自发生成 UUID 作为幂等标识，因为模型重试时很容易随机生成一个新的 UUID，使后端的防重逻辑失效。幂等键必须由宿主运行时根据业务意图与参数确定性派生：

$$\text{IdempotencyKey} = \text{Hash}(\text{session\_id} + \text{tool\_name} + \text{json\_dumps\_sorted}(\text{business\_args}))$$

注意，若系统具备静态任务分解，也可使用固定的规划节点 ID（`plan_step_id`），但**绝不能使用随循环自增的执行轮次计数**；否则模型在下一轮尝试重试时，步数递增会导致哈希变化，从而使后端防重拦截彻底失效。在向外部支付系统发起请求时，必须在 HTTP Header 或参数中携带该 `IdempotencyKey`。

### 2. 状态机流转与短路返回
工具网关层需要维护一个记录请求状态的存储层（如 Redis）：
1.  **初次到达**：原子写入键值并设置状态为 `PROCESSING`，设置合理的超时时间（TTL）。
2.  **并发重复到达**：若查询到该键状态为 `PROCESSING`，说明前一个请求仍在处理中，直接拒绝并发并让当前调用等待，防止并发竞态。
3.  **处理完成**：底层服务成功处理后，将该键状态置为 `COMPLETED`，并将响应结果（如退款凭证编号）一同缓存。
4.  **后续重试到达**：若查询到该键状态为 `COMPLETED`，则跳过底层写操作逻辑，直接返回此前缓存的成功结果给 Agent。

### 3. 高危动作的人机确认模式（Human-in-the-loop）
对于超过预设金额（如单笔退款大于 1000 元）的高危写操作，不应完全让 Agent 自动提交。可参考 [Agent 权限隔离与安全控制](/llm-100/agent-engineering/agent-permissions-isolation.md) 的原则，工具在检测到敏感参数时，只生成一个“待确认工单”并将状态挂起（Interrupt），等待人工审核员点击确认后，调度器再将上下文恢复继续执行。

![工具调用的幂等性控制时序图](/images/content-rebuild/7e10c3b349027e52/fig-3.png)

上图对比了重复调用在有无幂等层时的差异：幂等网关在拦截到相同的幂等键后，直接短路并返回历史结果，阻断了重复撞击下游支付网关的风险。

---

## 上下文膨胀的工程治理：窗口、摘要与外置引用

在执行复杂排查或多步操作时，全量保留所有中间输出不仅成本高昂，还会降低长文本下模型对关键系统指令的遵循效果。结合 [Agent 记忆系统设计](/frontier/agent-memory-system/memory-design.md) 的实践，工程上有三项针对性的修剪手段：

### 1. 消息滑动窗口与差异化保留
不要对所有历史消息采用均等丢弃策略：
*   **系统头部（System Prompt & Initial Task）**：永久保留，确保任务初心不漂移。
*   **中间历史工具结果（Past Tool Outputs）**：对于较早轮次的工具原始输出，只保留简要执行状态（如 `status: success`），清除具体长文本报文。
*   **末尾上下文（Recent Turns）**：保留最近 1~2 轮完整的 Thought、Action 和 Observation，维持当下的推理连续性。

### 2. 阶段性摘要压缩（Summarization）
当 Token 消耗接近设定阈值（例如达到模型上下文窗口的 60%）时，由调度程序调用轻量级模型对前面的中间步骤进行结构化提炼：

```text
【当前进展摘要】
1. 已验证用户手机号与订单编号（order_100234）。
2. 订单处于“待收货”状态，购买商品为耳机，签收未超过7天，符合无理由退款政策。
3. 当前待办：调用退款接口执行扣款撤销。
```

生成摘要后，系统将前面 8 轮冗长的对话与工具交互折叠为这一张“进展表”，大幅降低后续调用的输入长度。

### 3. 大体量数据外置存储（Artifact Offloading）
如果工具返回了大量非结构化数据（例如拉取了 2MB 的服务器运行日志，或者数千行的表格数据），千万不要将这篇大文本完整丢进上下文。工具的正确处理方式是：
1. 将大文本写入外部对象存储或临时文件，生成一个引用 ID（如 `artifact://logs/task_8921.txt`）。
2. 仅向模型上下文返回前 3 行预览样本、总行数、大小以及该引用 ID。
3. 告知模型若需分析该数据，需调用具备过滤能力的专用分析工具（如传入 `grep` 关键词和引用 ID），由后端代码在沙箱中按需切片处理，保持主干上下文轻量。

![上下文管理三维治理方案示意](/images/content-rebuild/7e10c3b349027e52/fig-4.png)

上图清晰展示了这三种手段的组合：头部核心提示常驻，中间冗长过程被摘要折叠，体量巨大的数据报文转存为外置引用仅透出句柄，有效控制了上下文长度。

---

## 端到端自动化评测：终态断言与效率衡量

评估 Agent 的表现不能仅靠几条静态输入和人工肉眼看回复。模型输出一句“我已经为您成功办理了退款”，并不代表支付系统里真的产生了一条退款单。

### 为什么单纯的字符串匹配不可靠？
1.  **同义表达多样性**：自然语言回答变化繁多，即便用正则也很难穷尽合规的句式。
2.  **幻觉掩盖行动缺失**：模型完全可能在 Thought 中推理错误，没有输出任何 Action，却在最终回答中宣称“已搞定”。

因此，**Agent 的评测必须以“物理世界的环境终态（State of the World）”为主要评估基准**。

### 核心评测维度与指标设计

一套实用的自动化回归评测通常包含沙箱环境（Mock 数据库与外部 API）、固定的测试用例集以及以下三项量化指标：

| 指标项 | 采集方式 | 业务含义 |
| :--- | :--- | :--- |
| **任务完成率（Success Rate, SR）** | 运行结束后直接查询沙箱数据库：`order.status == 'REFUNDED'` 且产生对应金额流水。 | 断言最终物理状态是否达成目标（0 或 1）。 |
| **工具调用合法率（Tool Validity）** | 审计执行轨迹（Trace）中的工具调用序列，确认关键必经节点是否触发，有无越权或错误调用。 | 验证决策路径是否遵循业务流程规范。 |
| **步骤冗余惩罚（Step Penalty）** | 设定最优路径基准步数 $S_{optimal}$，实际步数为 $S_{actual}$。当 $S_{actual} > S_{optimal}$ 时计算相对偏离度。 | 评估 Agent 决策的经济性与是否有多余试错。 |

综合评分可以将惩罚项转化为正向的步数效率 $(1 - \text{StepPenalty})$ 进行加权求和：

$$\text{Score} = w_1 \cdot \text{SR} + w_2 \cdot \text{ToolValidity} + w_3 \cdot (1 - \text{StepPenalty})$$

其中权重系数满足 $w_1 + w_2 + w_3 = 1$（例如分别设定为 0.6、0.2、0.2），确保各单项与最终得分均严格归一化在 $[0, 1]$ 区间。

```python
def evaluate_refund_agent(trace_log: dict, sandbox_db) -> dict:
    # 1. 终态断言：直接检查沙箱中的业务数据，而非模型说了什么
    order = sandbox_db.get_order("order_100234")
    is_success = (order.status == "REFUNDED") and (order.refund_amount == 199.0)

    # 2. 轨迹审计：是否按预期调用了退款与通知工具，且没有多余非法动作
    executed_tools = [step["tool"] for step in trace_log.get("steps", [])]
    has_refund = "refund_payment" in executed_tools
    has_illegal_call = any(t not in ["query_order", "check_policy", "refund_payment", "send_sms"] for t in executed_tools)
    tool_validity = 1.0 if (has_refund and not has_illegal_call) else 0.0

    # 3. 步骤冗余度计算
    optimal_steps = 4
    actual_steps = len(executed_tools)
    step_penalty = max(0.0, (actual_steps - optimal_steps) / actual_steps) if actual_steps > 0 else 1.0

    return {
        "success": 1.0 if is_success else 0.0,
        "tool_validity": tool_validity,
        "step_penalty": round(step_penalty, 3)
    }
```

![端到端评测评估管线示意图](/images/content-rebuild/7e10c3b349027e52/fig-5.png)

上图展示了自动评测的管线结构：测试套件提供输入基准并在沙箱隔离环境中运行 Agent，最终由断言引擎聚合终态数据、调用轨迹与步数代价，产出量化质量指标。

---

## 生产高可用方案总结与面试解析

在技术交流与面试中，当被问到“如何保证 Agent 系统在复杂长链条任务中的稳定性”时，回答应避免只罗列单点技巧（如“多写几层 try-catch”或“Prompt 里让它别乱调”），而应展现出**分层防御与离线验证相结合**的工程化视角。

### 推荐的系统化答题逻辑
1.  **明确核心矛盾**：大模型的概率性输出与业务生产系统的确定性要求之间存在天然冲突。高可用的本质不是试图彻底消除模型的随机性，而是用宿主程序的硬性边界把随机性框定在安全范围内。
2.  **运行时控制（Runtime Guards）**：
    *   **调度守卫**：配置物理超时时钟、最大步数阈值；对工具名和排序参数计算 SHA-256 哈希，识别连续重复调用并执行熔断。
    *   **写工具幂等**：基于 `SessionID + ToolName + 规范化业务参数`（或静态规划节点 ID）确定性派生幂等键，避免因执行轮次自增而在模型重试时生成新键，在写操作前进行分布式状态锁定以防重复提交；敏感操作配置人工确认中断点。
    *   **上下文精简**：实施固定头部、历史折叠、大数据外置引用（Artifact Offloading），防止上下文膨胀导致性能劣化与指令稀释。
3.  **开发与验证体系（Offline Verification）**：
    *   建立隔离沙箱环境，拒绝依赖纯文本匹配打分。
    *   通过终态断言（状态是否落库）、轨迹审计（动作合规性）以及步数冗余惩罚，构建持续集成的自动化回归评测套件。

![面试复盘：Agent 生产高可用四层防护架构](/images/content-rebuild/7e10c3b349027e52/fig-6.png)

上图将本文涉及的机制进行了结构化梳理：在线运行时依托调度守卫、上下文治理与幂等拦截守护底线，离线阶段依托沙箱评测保障版本演进的质量底线。

---

## 学习检验与课后练习

### 练习题
1. **场景分析题**：如果某个 Agent 任务在第 5 步需要向第三方物流系统提交修改收货地址的请求，由于网络瞬时超时，宿主系统发起了重试。此时模型如果重新生成了一次工具调用，该机制在没有幂等键防护下会面临什么隐患？如何由调度器派生出可靠的幂等键？
2. **评测设计题**：某团队开发了一个根据自然语言生成并执行 SQL 的 Agent，他们原本使用 `assert "SELECT" in response` 和 `assert len(response) > 0` 来验证 Agent 是否成功。请说明这种断言方式的主要缺陷，并设计一个基于沙箱环境的终态断言方案。

---

### 参考答案解析

1. **场景分析题解析**：
   *   **隐患**：第三方系统可能已经在前一次请求中完成了数据修改，但由于返回报文超时丢失，Agent 未能获取成功状态。若无幂等键，重试调用可能被第三方系统当作一次全新的变更请求，造成重复落库、日志混乱，或者在涉及库存/扣费场景时导致资损。
   *   **派生方案**：幂等键不能由模型自发生成随机字符串（重试时容易变化），也不能依赖随动态循环递增的执行轮次计数。应由宿主程序在调度层根据业务上下文确定性派生：例如 `SHA256(SessionID + ToolName + 排序业务参数)`，或基于静态编排中的固定节点 `PlanStepID`。这样无论网络重试还是模型发起下一轮补偿调用，该次特定动作派生出的幂等键始终恒定，第三方系统或本地拦截网关识别到相同键值即可直接返回首次处理的结果快照。

2. **评测设计题解析**：
   *   **缺陷**：字符串检查只验证了模型“说了什么”，不能证明它“做了什么”。模型可能在文本中输出了合法的 `SELECT` 语句，但实际上并没有成功连接数据库执行；或者语句存在语法错误、逻辑错误（如过滤条件错误导致返回空结果集），字符串断言依然会误判为成功。
   *   **终态断言方案**：
       *   **沙箱初始化**：在独立的测试数据库中预先灌入固定样本数据（Ground Truth）。
       *   **终态比对**：运行结束后，不直接比对模型文本，而是由评测脚本执行该 SQL 或检查系统执行结果集，比对结果集的内容、字段、行数与预期标准答案是否完全一致。
       *   **轨迹与安全审计**：检查执行轨迹中是否仅执行了只读查询，断言没有触发非预期的 `DROP`、`UPDATE` 等越权写操作。

> 想要进一步掌握 Agent 底层状态机的工程实现细节，可以参阅进阶章节 [Agent 状态机怎么设计？](/llm-100/agent-engineering/agent-state-machine.md)；如需了解工具与能力模块的标准封装，可继续学习 [Agent Skills 规范与封装设计](/frontier/agent-skills/skill-encapsulation.md)。
