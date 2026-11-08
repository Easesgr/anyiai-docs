---
title: 多轮对话中意图变化怎么办？追问、澄清与任务切换设计
icon: mdi:target
tag:
- 多轮对话
- 意图追踪
- 追问澄清
- 任务切换
description: 以机票预订中途插问天气为例，详解多轮对话状态跟踪（DST）、槽位冲突解决、追问与澄清触发边界，以及基于状态栈的任务打断恢复机制。
---

![动态意图追踪与控制流封面图](/images/content-rebuild/e17df3465291bede/cover.png)

在单轮对话系统中，系统只需完成单句话的意图分类与实体抽取。但在真实场景中，用户极少按照系统预设的线性表单逐项回答。用户可能中途改口、补充信息、临时插问无关问题，甚至在两个业务之间来回切换。

例如，用户先说“帮我查下周二去深圳的机票”，在系统询问乘机人时突然插问“深圳周二会下雨吗？”，得到天气答复后又改口“那算了，改去广州，要早上的航班”。如果系统直接用最新的文本覆盖前文，机票预订的上下文就会丢失；如果系统强行忽略天气提问，交互就会显得机械僵硬。

在上一篇[/frontier/multi-turn-dialogue/why-multi-turn-hard.md](/frontier/multi-turn-dialogue/why-multi-turn-hard.md)中，我们分析了指代、省略与语义漂移带来的理解困难；在[/frontier/multi-turn-dialogue/context-management.md](/frontier/multi-turn-dialogue/context-management.md)中，我们探讨了上下文滑动窗口与压缩剪枝。本文聚焦业务流程控制层，解释系统如何通过**对话状态跟踪（DST）**、**参数槽位更新冲突解决**、**追问与澄清判定**，以及**基于状态栈的任务中断与恢复**，应对多轮交互中的意图变化。

---

### 对话状态跟踪（DST）：把离散输入整理为结构化状态

单轮意图分类假设用户的单次输入就包含了完整目标。但在多轮任务型对话中，目标信息通常分多次给出。对话状态跟踪（Dialogue State Tracking, DST）负责在每个交互轮次 $t$ 维护当前的对话状态 $S_t$：

$$S_t = f(S_{t-1}, U_t, A_{t-1})$$

其中：
* $S_{t-1}$ 为上一轮保存的对话状态；
* $U_t$ 为当前轮次用户的自然语言输入；
* $A_{t-1}$ 为上一轮系统向用户执行的动作（例如系统的提问、确认请求或工具执行结果）；
* $f$ 为状态更新函数，负责提取本轮新增信息并合并到历史状态中。

状态 $S_t$ 通常包含两部分核心数据：
1. **当前活跃意图（Active Intent）**：当前正在推进的业务目标（如 `book_flight`、`query_weather`）。
2. **累积槽位集合（Accumulated Slots）**：执行该目标所需的参数键值对（如 `{destination: "广州", departure_date: "2026-04-14", departure_time_range: "morning"}`）。

![多轮对话状态跟踪结构图](/images/content-rebuild/e17df3465291bede/fig-1.png)

通过 DST，系统将自然语言理解与业务状态推进解耦。模型不需要在每一轮重新通读长篇历史闲聊，状态管理器会将提取并校验通过的结构化槽位传递给后续控制逻辑，降低长上下文带来的漏填与幻觉风险。

---

### 槽位填充机制：显式提供、隐式推断与冲突覆盖

槽位填充（Slot Filling）不是简单的字典累加。用户的输入方式不同，信息的可靠性与覆盖规则也不同。

<ReadMoreLock />

参数输入通常来自三种途径：
* **显式提供**：用户直接说出具体值，例如“买下周二的票”。
* **隐式推断**：用户没有明确说出参数，但可以从上下文推导。例如用户先确认了某家酒店，随后问“打车去机场要多久”，系统可推断出发地即为该酒店。
* **修改与冲突覆盖**：用户主动修改之前输入的值，例如“不要周二的了，改成周三”。

若系统采取无条件后入覆盖，当模型出现指代误判时，可能导致已确认的正确参数被错误替换。为了解决冲突，系统应在槽位数据中附带元数据（来源、置信度、是否已确认），并设定覆盖规则：

```python
from dataclasses import dataclass, field
from typing import Any, Optional
import time

@dataclass
class SlotValue:
    name: str
    value: Any
    confidence: float
    source: str          # "user_explicit"（显式输入） | "context_infer"（上下文推断） | "system_default"（系统默认）
    timestamp: float = field(default_factory=time.time)
    confirmed: bool = False

class SlotManager:
    def __init__(self):
        self.slots: dict[str, SlotValue] = {}

    def update_slot(self, new_slot: SlotValue, allow_overwrite_confirmed: bool = False) -> str:
        if new_slot.name not in self.slots:
            self.slots[new_slot.name] = new_slot
            return "INSERT"
        
        current = self.slots[new_slot.name]
        
        # 规则 1：显式输入优先于隐式推断，隐式推断不能直接覆盖已有的显式输入
        if current.source == "user_explicit" and new_slot.source == "context_infer":
            return "REJECT_INFER_ON_EXPLICIT"
        
        # 规则 2：若当前槽位已由用户明确确认，修改前需进入二次确认流程
        if current.confirmed and new_slot.value != current.value:
            if not allow_overwrite_confirmed:
                return "REQUIRE_CONFIRMATION"
        
        self.slots[new_slot.name] = new_slot
        return "UPDATE"
```

![槽位填充与冲突更新判定机制](/images/content-rebuild/e17df3465291bede/fig-2.png)

除了覆盖判定，槽位写入前必须经过类型和范围检查（例如返回日期不能早于出发日期）。未通过校验的参数不能存入有效状态，应向用户反馈错误原因。

---

### 追问与澄清：何时向用户索要更多信息

在参数不齐或理解不确定时，系统需要主动向用户发问。发问分为两类：**追问（Follow-up）**与**澄清（Clarification）**，两者的触发条件和目的不同。

1. **追问（Follow-up）**：目标意图已明确，但缺少业务执行必须的槽位，且无法从上下文安全推导。此时系统需要针对性索要缺失字段。
   * 触发条件：必填槽位集合减去已填槽位集合不为空（$S_{required} \setminus S_{filled} \neq \emptyset$）。
   * 示例：“请问您打算预订哪一天的航班？”

2. **澄清（Clarification）**：用户输入的语义存在歧义，系统无法断定其意图或实体所指。
   * 示例 1（意图歧义）：用户输入“退钱”，系统无法确定是针对未发货订单退款，还是针对已使用服务的售后赔付。若两项操作流转互斥，需发起澄清。
   * 示例 2（实体歧义）：用户输入“去朝阳的票”，系统词库中同时存在“北京市朝阳区”与“辽宁省朝阳市”。
   * 示例 3（高代价操作）：涉及扣款、删除、权限变更等操作。即使模型抽取置信度较高，执行前仍应进行显式确认，防止自然语言歧义导致误操作。

![主动澄清与追问触发决策流程](/images/content-rebuild/e17df3465291bede/fig-3.png)

在工程实现中，澄清和追问应遵循由窄到宽的原则：优先追问单一缺失参数，避免一次性向用户抛出过多字段；遇到多意图冲突时，列出具体选项供用户选择，而不是让模型随意揣测。

---

### 任务打断与恢复：基于状态栈的分支调度

在订票过程中，用户可能会临时插入其他查询，例如“明天深圳天气怎么样？”。如果直接将当前意图重置为天气查询，之前收集的机票参数就会丢失；如果拒绝回答天气，会打断用户的交互意图。

处理此类分支嵌套的常用方法是**任务状态栈（Task State Stack）**。

系统使用后进先出（LIFO）的栈结构管理多任务上下文。每个栈帧包含任务标识、已填槽位、当前执行步骤以及生命周期状态（`Active`、`Suspended`、`Completed`）。

![状态栈任务嵌套与打断恢复流转图](/images/content-rebuild/e17df3465291bede/fig-4.png)

#### 调度步骤：

1. **检测到打断并压栈（Push）**：
   * 系统检测到新输入命中独立的新任务（查天气），且与当前任务（订机票）的槽位填充无关。
   * 将当前主任务状态标记为挂起（`Suspended`），整帧快照压入栈中。
   * 将新任务置为栈顶，标记为活跃（`Active`）。

2. **共享与继承上下文**：
   * 子任务初始化时，可按需读取父任务已有的参数。例如查天气需要 `location`，系统可直接复用主任务中的目的地“深圳”，无需再次询问用户。

3. **子任务结束并出栈（Pop）**：
   * 天气查询完成后，该任务生命周期变为 `Completed`，从栈顶弹出销毁。
   * 原机票预订任务重新成为栈顶，状态恢复为 `Active`。

4. **输出答复并主动唤回**：
   * 系统输出天气结果后，需顺带给出进度提示，告知用户当前已切回原任务。例如：“明天深圳有小雨，气温 22 度。回到刚才的机票预订，请问乘机人姓名是？”这能帮助用户感知当前所处的业务节点。

---

### 混合架构：LLM 语义抽取与确定性状态机分工

如果完全依靠提示词让大模型直接控制任务流转，系统容易因模型输出的不稳定性产生流程越权或死循环。大语言模型擅长处理非结构化口语、提取实体和归纳意图；但在状态跳转边界、权限检查和硬性条件校验上，代码编写的有限状态机（FSM）更为可靠。

常见的工程方案是**LLM 负责抽取，FSM 负责流转**：

* **LLM（感知层）**：将用户的自然语言转为结构化的 JSON 数据（包含预测意图、实体键值对、用户情绪等）。
* **FSM（控制层）**：依据当前节点、必填项完整度及业务规则，判断下一步是追问、确认、调用 API 还是切换任务。

![分层架构：LLM语义解析与FSM业务状态机协同](/images/content-rebuild/e17df3465291bede/fig-5.png)

```python
from enum import Enum
from typing import Dict, List, Optional

class TaskStatus(Enum):
    IDLE = "IDLE"
    FILLING_SLOTS = "FILLING_SLOTS"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    EXECUTING = "EXECUTING"

class DialogueController:
    def __init__(self, required_slots: List[str]):
        self.required_slots = required_slots
        self.status = TaskStatus.IDLE
        self.slot_store: Dict[str, str] = {}

    def step(self, extracted_slots: Dict[str, str], user_intent: str, user_confirmed: Optional[bool] = None) -> dict:
        """
        状态机控制业务流转，输入来自 LLM 提取的结构化数据
        """
        if user_intent == "CANCEL":
            self.status = TaskStatus.IDLE
            self.slot_store.clear()
            return {"action": "RESET", "message": "任务已取消。"}

        # 合并有效槽位
        for k, v in extracted_slots.items():
            if k in self.required_slots and v:
                self.slot_store[k] = v

        missing = [s for s in self.required_slots if s not in self.slot_store]

        # 状态机条件判断
        if missing:
            self.status = TaskStatus.FILLING_SLOTS
            return {"action": "PROMPT_SLOT", "target_slot": missing[0]}
        
        if self.status in (TaskStatus.FILLING_SLOTS, TaskStatus.IDLE):
            self.status = TaskStatus.AWAITING_CONFIRMATION
            return {"action": "CONFIRM_DETAILS", "slots": self.slot_store}
        
        if self.status == TaskStatus.AWAITING_CONFIRMATION:
            if user_confirmed is True:
                self.status = TaskStatus.EXECUTING
                return {"action": "INVOKE_BACKEND_API", "payload": self.slot_store}
            elif user_confirmed is False:
                self.status = TaskStatus.FILLING_SLOTS
                return {"action": "ASK_CORRECTION", "message": "请问哪项信息需要修改？"}

        return {"action": "WAIT_USER_INPUT"}
```

在这一结构中，模型无法绕过状态机的守卫逻辑直接触发后端执行。即使模型误判用户已同意，若上一状态并非 `AWAITING_CONFIRMATION`，控制器也不会发出 API 调用指令。

---

### 异常处理与死循环防范

多轮交互中可能出现系统反复索要同一参数、用户回答始终未命中的死循环，这会导致对话体验中断。

![死循环熔断与降级处理时序图](/images/content-rebuild/e17df3465291bede/fig-6.png)

针对常见异常，系统需设置计数器与降级策略：

1. **追问失败与死循环拦截**：
   * **现象**：系统连续多次（例如达到 3 次）追问同一槽位，依然未能抽取到合法值。
   * **处理**：停止以开放式问题追问，降级为包含选项按钮的界面交互；在无法使用图形界面的通道（如纯语音通话），则转接人工坐席或提示用户换一种表述。

2. **意图高频摇摆**：
   * **现象**：用户在极短轮次内连续切换任务（如“查机票” $\to$ “查火车” $\to$ “查退票”）。
   * **处理**：暂停自动压栈和切换，主动输出当前上下文并请用户确认优先级，例如：“检测到您同时在查看机票和火车票，请问您想先处理哪一项？”

3. **拒绝协作与负向反馈**：
   * **现象**：用户表达强烈不耐烦或明确拒绝回答（如“不用了”、“听不懂”）。
   * **处理**：立即中止当前任务的槽位收集，保留会话记录供回溯排查，并提供人工服务或退回主菜单。

通过对参数来源分级、状态显式存储、利用状态栈管理打断以及引入状态机前置校验，多轮对话系统在面对复杂的用户改口和分支提问时，能够保持流程受控、数据完整与交互自然。
