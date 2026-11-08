---
title: Agent 为什么容易陷入死循环？如何做工程兜底？
icon: mdi:cpu-64-bit
date: 2026-07-16
tag:
- 大模型100讲
- Agent
description: 剖析 Agent 陷入死循环的三种表现与真实诱因，掌握动作指纹、滑动窗口检测与分级熔断等确定性防御工程实现。
---

在前一讲 [ReAct 框架是什么？为什么适合 Agent？](/llm-100/agent-basics/ReAct_框架是什么_为什么适合_Agent.md) 中，我们拆解了“思考—行动—观察”的经典运行循环。大模型依赖环境返回的观察结果（Observation）动态决定下一步是继续调用工具还是输出最终结论。

然而，当我们将调度逻辑完全让渡给基于概率生成的语言模型时，系统经常会出现一个典型故障：**在同一个问题上反复打转，直到耗尽 Token 或触发超时报错**。本讲将用一个完整的客服查询场景串起死循环的常见表现与诱因，并给出可以在生产环境中直接落地的软件工程防御方案。

![Agent 循环死锁与兜底控制总览](/images/content-rebuild/f3db71ec12ee1f85/cover.png)

> **配图说明与导读**：上图展示了 Agent 在执行环境中陷入死锁的典型场景。黑色短发讲解员正在操作控制台上的熔断断路器，将原本无序打转的执行流强行引入确定性降级轨道。在阅读后续小节时，请重点观察“模型自主决策”与“外部工程控制面”之间的权责边界。

---

## 一个贯穿例子：订单退款查询

为了看清楚死循环是怎么发生的，我们设定一个具体的业务场景：

用户向客服 Agent 提问：*“帮我查一下上周买的无线耳机退款到账没有，单号好像是 ORD-8899。”*

宿主系统为 Agent 注册了三个工具接口：
1. `get_order_detail(order_id)`：查询订单基础信息。
2. `query_user_account(user_id)`：根据用户 ID 查询账户资产与绑定手机。
3. `check_refund_status(order_id)`：查询指定订单的退款结算状态。

在理想状态下，Agent 依次调用工具获取结果后返回答复。但在实际运行中，一旦某些输入或者返回不符合预期，系统就会表现出以下三种完全不同的死循环特征。

---

## 死循环的三种典型表现

在排查运行日志（Trace）时，不要把死循环笼统地看作“模型卡住了”，需要根据工具名称与参数变化区分为三种模式：

![Agent 三种典型死循环模式对比](/images/content-rebuild/f3db71ec12ee1f85/fig-1.png)

> **配图说明与导读**：上图对比了运行时的三种循环表象：单点原地重试、二元交替震荡以及思维空转。阅读本节时，请注意区分“模型仍在产生工具调用”与“模型仅在输出文本推理”这两类截然不同的失效状态。

### 1. 相同动作与参数重复调用（Idempotent Loop）
模型在连续多轮中产生完全相同的工具名和参数。在我们的订单例子中，模型生成动作 `check_refund_status(order_id="ORD-8899")`，接口返回错误 `{"error": "系统繁忙"}`。下一轮模型生成的推理文本可能是“退款接口刚刚未成功，我需要再次查询”，随后再次发起完全一样的调用，如此循环往复五六次。

### 2. 交替状态震荡（Oscillation Loop / A-B-A-B 乒乓震荡）
模型在两个或多个工具调用之间来回切换。例如：
1. 模型调用 `get_order_detail(order_id="ORD-8899")`，接口提示“需提供 user_id 才能查询详情”；
2. 模型转而调用 `query_user_account` 试图查出 user_id，但由于缺少用户身份标识，该接口提示“需提供 order_id 反查关联账户”；
3. 模型再次回退调用 `get_order_detail`。
单次调用的参数看似在更新，但整体状态在有限的几个局部状态间振荡，无法向前推进。

### 3. 文本空转与重复思考（Deliberation Loop）
模型不再生成任何 Tool Call，也没有触发任务终止，而是在输出中不断重复类似的内容：“我现在已经了解了单号，需要结合上一轮的退款规定重新评估……让我再理清一下逻辑……”。由于上下文里的长篇思考没有产生新的事实输入，下一轮预测仍然停留在“继续分析”的概率倾向里，迟迟无法输出最终答案。

---

## 诱因一：工具错误信息缺乏建设性引导

工具执行出错是常态，导致死循环的往往不是“报错本身”，而是**报错信息没有提供可供下一步决策调整的线索**。

![错误信息设计对 Agent 决策分流的影响](/images/content-rebuild/f3db71ec12ee1f85/fig-2.png)

> **配图说明与导读**：上图展示了哑报错与建设性报错在驱动 Agent 状态转移时的因果分支。注意观察左侧模糊状态如何导致概率收敛到初始高频错误，而右侧结构化错误如何引导模型进行参数修正或工具切换。

如果后端的接口异常直接按常规 Web 服务的做法返回通用状态码：

```json
// 缺乏有效引导的哑报错
{"status": 500, "message": "Internal Server Error"}
```

语言模型在上下文里只看到了“执行失败”，但不知道是参数格式错、权限不够还是记录不存在。在预训练知识中，“遇到临时服务端错误通常可以稍后重试”，于是模型便根据概率惯性继续发起相同调用。

建设性的工具错误返回必须明确包含**失败原因、预期格式与可选替代方案**：

```json
// 具备调整线索的结构化报错
{
  "status": "error",
  "error_code": "INVALID_ORDER_PREFIX",
  "message": "单号 'ORD-8899' 不符合格式规范。合规单号需为 'ORD-' 加 8 位数字（如 ORD-20240315）。",
  "suggestion": "请先调用 query_user_account 获取该用户的最近订单列表，确认正确单号后再调用本接口。"
}
```

当 Observation 提供了清晰的因果链条与下一步建议，大语言模型就能准确转向修正参数或切换工具。

---

## 诱因二：上下文滑动截断丢失失败教训

为了避免超出大模型的上下文窗口上限（Context Window）或节约 Prompt 成本，工程实现通常会引入消息队列的滑动截断（Sliding Window）或自动摘要。

![上下文截断导致失败历史丢失与记忆死锁机制](/images/content-rebuild/f3db71ec12ee1f85/fig-3.png)

> **配图说明与导读**：上图展示了基于 FIFO 的滑动窗口裁剪机制如何抹去关键的失败记录。注意图中早期的“方法 A 失败”记录被移出有效窗口后，模型在当前轮次再次生成方法 A 的因果关系。

如果使用简单的先进先出（FIFO）方式直接裁剪较早的消息，会造成关键否定信息的丢失：
1. 第 2 轮：模型尝试调用 `check_refund_status(order_id="ORD-8899")`，系统明确返回“该单号不存在任何退款记录”；
2. 第 3~8 轮：模型尝试查询账户、调用外部物流等动作，产生大量长文本；
3. 第 9 轮：由于消息总长度超标，第 2 轮中“单号不存在退款记录”的对话记录被移出了 Prompt；
4. 第 10 轮：模型审视当前剩下的对话上下文，发现用户问的是“退款去哪了”，而上下文中没有任何关于“该单号是否查过退款”的记录。模型自然会再次发起 `check_refund_status(order_id="ORD-8899")`。

丢掉否定性证据是长流程 Agent 陷入深层循环的常见原因。关于记忆的生命周期与剪裁细节，可参考 [Agent 的 Memory 应该怎么设计？](/llm-100/agent-basics/Agent_的_Memory_应该怎么设计.md)。

<ReadMoreLock />

---

## 诱因三：终止条件模糊与解析器异常反馈

代码逻辑中的循环有明确的退出判断（如 `while len(results) < 5:`），而在 Agent 系统中，终止与否往往依赖模型对 Prompt 要求的语义判断。

![模糊停止判定与严格停止边界的判定逻辑对比](/images/content-rebuild/f3db71ec12ee1f85/fig-4.png)

> **配图说明与导读**：上图对比了开放式自然语言终止目标与形式化状态机退出条件的执行区别。右侧方案通过定义明确的 Schema 和终止状态节点，消除了模型因自我怀疑而无限发散的风险。

以下两类设计缺陷最容易导致停止逻辑失效：

### 1. 任务终点缺乏客观度量标准
如果 Prompt 写法为：“*请持续查询，直到充分掌握该订单的所有退款明细后，再向用户汇报。*”
“充分掌握”是一个主观形容词。模型查了退款状态后，因为无法确认信息是否“足够全面”，在下一轮决策时仍有很大几率产生“再查一次关联信息”的决策，导致查询发散。

### 2. 终结动作解析失败被误当成工具报错
很多框架依赖特定的 JSON 字段（如 `{"action": "finish"}`）或者指定的输出格式来识别任务完成。如果大模型在给出最终回答时附带了前导解释，导致解析器抛出 `OutputParserException`：
```text
OutputParserException: Could not parse output: '综上所述，您的退款已到账...' 
```
如果宿主程序机械地将这个报错直接包装成 Observation 塞回给模型：“*解析失败，请按规范格式输出*”，模型有时会误以为自己的结论被业务系统驳回，进而放弃输出结论，转头重新去调用其他工具。

---

## 工程防御机制：最大迭代限制与动作指纹检测

大模型的输出本质上是概率采样，在工程设计上不能假设它“永远按预期退出”。必须在宿主程序中建立一道**完全独立于大模型意图的确定性控制面**。

![滑动窗口动作指纹哈希与周期震荡检测算法流程](/images/content-rebuild/f3db71ec12ee1f85/fig-5.png)

> **配图说明与导读**：上图展示了宿主系统在拦截模型输出时的流水线。讲解员展示了指纹计算与双重检测机制（单点频率统计 + 周期序列匹配），拦截器在检测到死循环风险时立即打断流程并转入纠偏流程。

### 1. 最大迭代步数限制（Recursion / Iteration Limit）
在框架层面必须设置单次任务的最大步数上限（例如 LangGraph 提供的 `recursion_limit`）。无论中间推理多么看似合理，一旦步数达到阈值立即强制中断执行。这是防止 Token 消耗与服务超时的底线防护。

### 2. 动作指纹与周期震荡检测器（Loop Detector）
在达到最大步数之前，系统应当及早感知循环迹象。我们可以在宿主端对每次模型生成的“工具名 + 参数”做规范化并计算哈希值（Action Fingerprint），放入滑动窗口中做实时检查。

以下是一个可以直接运行的轻量级检测器实现：

```python
import hashlib
import json
from collections import Counter
from typing import List, Dict, Any, Tuple

def compute_action_fingerprint(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """对工具名称和参数进行排序序列化，生成唯一的动作指纹"""
    normalized_payload = {
        "tool": tool_name.strip().lower(),
        "args": tool_args
    }
    # 保证字典键排序，避免由于序列化顺序不同导致哈希不一致
    serialized = json.dumps(normalized_payload, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()

class LoopDetector:
    def __init__(self, window_size: int = 6, repeat_threshold: int = 3):
        self.window_size = window_size
        self.repeat_threshold = repeat_threshold
        self.history: List[str] = []

    def record_and_check(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, str]:
        fp = compute_action_fingerprint(tool_name, tool_args)
        self.history.append(fp)
        
        # 截取滑动窗口
        window = self.history[-self.window_size:]
        
        # 规则 1：检查单点相同动作调用频次是否超标
        counts = Counter(window)
        if counts[fp] >= self.repeat_threshold:
            return True, f"检测到工具 [{tool_name}] 在近 {len(window)} 步内重复调用达到 {counts[fp]} 次。"
        
        # 规则 2：检查 A-B-A-B 周期震荡模式（周期长度为 2）
        if len(window) >= 4:
            if window[-1] == window[-3] and window[-2] == window[-4] and window[-1] != window[-2]:
                return True, "检测到两个工具交替调用的状态震荡（A-B-A-B 模式）。"
                
        return False, ""

# 验证运行
if __name__ == "__main__":
    detector = LoopDetector(window_size=6, repeat_threshold=3)
    
    # 模拟三次重复调用
    actions = [
        ("get_order_detail", {"order_id": "ORD-8899"}),
        ("get_order_detail", {"order_id": "ORD-8899"}),
        ("get_order_detail", {"order_id": "ORD-8899"})
    ]
    for name, args in actions:
        is_loop, reason = detector.record_and_check(name, args)
        if is_loop:
            print(f"[拦截成功] {reason}")
            break
```

---

## 状态注入与分级熔断

检测到死循环趋势后，并不一定要立即让任务直接报错崩溃。工程上推荐采用分级响应机制：

![分级防御与降级熔断流转图](/images/content-rebuild/f3db71ec12ee1f85/fig-6.png)

> **配图说明与导读**：上图展示了当 Agent 失控时的三级降级防御矩阵。讲解员指引读者从“轻度警示干预”，过渡到“确定性脚本兜底”，最后回退至“人工接管中断”的完整生命周期。

### 第一级：系统提示注入纠偏（Prompt Intervention）
当检测器首次捕获重复调用迹象时，宿主程序可以拦截这次实际的 API 请求，直接构造一条系统提示作为结果返回给模型：

```json
{
  "role": "tool",
  "name": "get_order_detail",
  "content": "[系统拦截通知] 检测到你已连续多次使用相同参数查询该接口，均未获得新的有效数据。当前禁止继续重复调用该接口。请根据现有信息给出结论，或者使用其他可用工具排查。"
}
```

通过将负向事实注入上下文，大多数主流模型能够理解当前动作已受限，从而转向生成备选方案或停止检索。

### 第二级：确定性代码降级（Deterministic Fallback）
正如 Anthropic 在《Building effective agents》中所指出的：对于业务确定性强的路径，应当使用由代码编排的 Workflow，而不是放任 Agent 自由发挥。如果多次注入纠偏仍然无法阻止异常循环，控制面应直接收回调度权：
- 停止调用模型，提取上下文里已经收集到的结构化字段；
- 交由预先编写好的备用逻辑（如按订单状态给出固定模板答复，或直接引导用户点击手动核对入口）。

### 第三级：人工介入与状态挂起（Human-in-the-loop）
在金融、审批等高敏感任务中，死锁可能意味着外部接口数据存在脏数据或冲突。此时最稳妥的方式是利用持久化状态机制将任务挂起：
1. 将当前的会话状态与执行历史序列化保存至持久化存储；
2. 任务状态置为待人工审核（如 LangGraph 的 `interrupt` 机制）；
3. 运维或客服人员在管理端介入，修改错误参数后发起恢复（Resume），任务从断点继续向下流转。

---

## 本讲练习与动手检验

请读者将上述代码复制到本地 Python 环境中，完成以下扩展练习：

1. **检验基础检测器**：编写一组模拟输入，构造 A-B-A-B 模式的四次工具调用（如 `tool_a` 与 `tool_b` 轮流调用），验证 `detector.record_and_check` 是否会在第 4 步准时触发拦截。
2. **支持三步循环检测**：扩展 `LoopDetector`，新增一条规则使其能够识别形如 A-B-C-A-B-C（周期长度为 3）的交替震荡，并编写测试用例确保当第 6 步出现重复时返回正确的拦截原因。

---

## 总结与索引

Agent 死循环不是因为模型参数写得不好，而是软件系统在缺少确定性约束时必然要面对的工程问题：

- **识别表象**：原地单点重试、A-B 乒乓震荡与思维空转的处理思路各不相同；
- **消除诱因**：工具报错要提供修正建议，上下文管理要保留负向教训，任务目标要有可度量的终止界限；
- **筑牢防线**：用最大递归步数兜底，用动作指纹及时拦截，并在必要时回退至代码分支或转人工接管。

在下一讲中，我们将深入长流程任务中的上下文与状态沉淀：
- [Agent 的 Memory 应该怎么设计？](/llm-100/agent-basics/Agent_的_Memory_应该怎么设计.md)

### 知识体系全景链接导航
- **基础篇起点**：[Agent 是什么？它和普通 ChatBot 有什么区别？](/llm-100/agent-basics/Agent_是什么_它和普通_ChatBot_有什么区别.md)
- **核心组件剖析**：[Agent 的核心组成：模型、记忆、工具和规划](/llm-100/agent-basics/Agent_的核心组成_模型_记忆_工具和规划.md)
- **底层交互协议**：[Tool Calling 是什么？大模型如何调用工具？](/llm-100/agent-basics/Tool_Calling_是什么_大模型如何调用工具.md) | [Function Calling 和 MCP 有什么区别？](/llm-100/agent-basics/Function_Calling_和_MCP_有什么区别.md)
- **前置设计框架**：[ReAct 框架是什么？为什么适合 Agent？](/llm-100/agent-basics/ReAct_框架是什么_为什么适合_Agent.md)
- **高阶协作模式**：[多 Agent 协作到底解决什么问题？](/llm-100/agent-basics/多_Agent_协作到底解决什么问题.md)
- **前沿进阶延伸**：[Agent 记忆系统深度演进](/frontier/agent-memory-system/README.md) | [Agent 技能封装与复用](/frontier/agent-skills/README.md)
