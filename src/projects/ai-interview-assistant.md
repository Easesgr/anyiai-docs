---
title: AI 模拟面试助手：状态机流转、动态追问与多维评估实战
icon: mdi:help-circle-outline
description: 基于有限状态机编排多轮面试交互，解决自由对话跑偏与追问失控；结合双层上下文压缩与结构化考查记录，实现从出题、深挖到多维打分的完整闭环。
---

![模拟面试智能体架构总览，涵盖有限状态机流转、考题推进、上下文压缩与多维量化打分](/images/content-rebuild/ea3e09081de403ae/cover.png)

*图 0：系统总体包含“大纲解析”、“状态机驱动引擎”、“双层上下文维护”与“多维报告生成”四大模块。阅读时请注意状态机（FSM）是掌控阶段转换的唯一中枢，模型仅在单节点内完成当次决策与生成。*

在开发基于大语言模型的模拟面试系统时，若直接让模型通过一个统一的角色提示词（Prompt）自由提问，通常会出现三类工程问题：
1. **面试官跑偏**：候选人在回答中提到无关话题或主动反问时，模型容易被候选人带偏，遗漏核心考点；
2. **追问失控**：模型缺乏“当前考察已充分”的退出逻辑，容易就单一技术细节连续追问四五轮，导致面试严重超时；
3. **长程遗忘与成本增加**：当多轮对话达到 15 轮以上时，完整的对话历史占用数千 Token，首字延迟增加，且模型在最后评估时容易遗忘早期的答题表现。

本篇承接 [AI 简历优化助手](/projects/ai-resume-assistant.md) 的结构化产出，使用**有限状态机（Finite State Machine, FSM）**对面试流程进行确定性调度。以“Java 后端开发岗位”中的“Redis 缓存一致性”考核为例，展示从开场破冰、八股考察、项目深挖、反问答疑到多维报告生成的完整实现过程。

---

## 交互难点：为什么自由对话无法保证面试质量

真实面试具有严格的考察目标、时间预算与阶段划分。纯 Prompt 方案和大模型状态机方案的运行对比如下：

![自由对话与状态机约束下的面试流程对比，展示话题发散与确定性节奏把控的区别](/images/content-rebuild/ea3e09081de403ae/fig-1.png)

*图 1：左侧纯 Prompt 驱动容易陷入漫无边际的自由讨论，单题追问难以收敛；右侧由 FSM 状态机接管阶段流转，模型只负责当前节点下的局部交互与结构化判定。*

将大模型直接用于面试交互，主要受制于以下三个限制：

1. **控制权漂移**：大模型的交互本质是根据前序上下文续写。如果候选人回答“我不清楚，但我想问贵司加班严重吗？”，自由提示词的模型极易顺势回答公司情况，丢失原有提问链路。
2. **计数与终止能力较弱**：大语言模型对多轮计数（例如“本题最多追问 2 次”）的执行并不稳定，往往需要程序在外部维护追问计数器与跳出条件。
3. **上下文注意力稀释**：对话达到数十轮时，不仅推理输入开销按轮次累加，注意力机制对超长文本中间内容的聚焦度也会出现波动，导致最终生成的评估报告漏掉早期的失分点。

因此，系统架构应采用**外部程序控制流程跳转，模型仅负责单轮理解与打分**的分工模式。

---

## 基于有限状态机（FSM）的阶段流转设计

我们将整场面试划分为 5 个状态，每个状态有明确的系统职责与流转条件：

![面试状态机五大阶段状态转移图，展示开场、八股、深挖、反问与评估之间的流转条件](/images/content-rebuild/ea3e09081de403ae/fig-2.png)

*图 2：状态流转图由五个主状态节点及转移条件构成。连线上的文字代表驱动跃迁的确定性规则，而非由大模型直接自由判定跳转。*

| 状态名称 | 阶段定位 | 核心任务 | 转移至下一阶段的触发条件 |
| :--- | :--- | :--- | :--- |
| `ICEBREAK` | 破冰开场 | 引导 1 分钟自我介绍，提取候选人表述的技术倾向 | 候选人完成介绍或本阶段轮次达 1 轮 |
| `CORE_KNOWLEDGE` | 基础八股 | 针对基础考题出题，每道题允许追问 1~2 次 | 基础考题池所有题目完成（指针指到项目题） |
| `PROJECT_DEEP_DIVE` | 项目深挖 | 针对简历重点项目深挖架构与方案权衡 | 项目考题池所有题目完成（考题列表耗尽） |
| `CANDIDATE_QA` | 反问答疑 | 回答候选人提问，维持专业客观形象 | 候选人明确表示“没有其他问题”或轮次达 3 轮 |
| `EVALUATION` | 综合评估 | 读取各题结构化打分记录，生成多维雷达报告 | 报告生成完成，会话进入终止态 `END` |

在代码实现中，可使用 [Agent 框架对比](/interview/agent-interview/agent-frameworks.md) 中介绍的图调度编排工具（如 LangGraph）或 Python 原生状态类。无论使用何种工具，核心规则是：**阶段跳转由宿主程序判定，不允许大模型自行修改全局状态**。

<ReadMoreLock />

---

## 结构化考题池的生成与数据模型

在会话启动时，系统读取候选人简历与招聘岗位说明（JD），通过一次离线 Prompt 生成包含评分基准和追问方向的考题池。考题池按阶段线性排列，前 $N$ 道题属于 `CORE_KNOWLEDGE`，后 $M$ 道题属于 `PROJECT_DEEP_DIVE`。

![基于简历提取结构化考题池的过程示意，展示从经历要点到考题与追问方向的映射](/images/content-rebuild/ea3e09081de403ae/fig-3.png)

*图 3：简历信息抽取器将纯文本解析为技术栈与项目亮点，题目生成器随之合成包含基准标准、追问方向的考题对象，按阶段严格排序后存入共享状态的题目队列中。*

### 考题与状态数据定义

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class QuestionItem(BaseModel):
    id: str
    stage: str = Field(description="所属阶段: CORE_KNOWLEDGE 或 PROJECT_DEEP_DIVE")
    topic: str = Field(description="考察技术点，如 Redis 缓存一致性")
    initial_question: str = Field(description="首问文本")
    key_criteria: List[str] = Field(description="及格评判要点")
    followup_directions: List[str] = Field(description="追问切入点，如延迟双删缺陷、Canal方案")
    max_followup: int = 2
    current_followup_count: int = 0
    is_completed: bool = False
    score: Optional[int] = None
    critique: Optional[str] = None

class InterviewPlan(BaseModel):
    candidate_name: str
    target_role: str
    questions: List[QuestionItem]
```

将生成的 `InterviewPlan` 写入会话状态，供状态机逐题推进。

---

## 双层上下文管理：滑动视窗与结构化待办

若在第 15 轮交互时依然将前 14 轮的原生文本（包括问候、候选人停顿与解释）全部拼装进 Prompt，单次请求的 Token 量将达到 8K 甚至更高。这不仅增加 Prefill 耗时与 API 费用，还会因注意力分散降低当轮判定的准确度。

系统采用**双层上下文管理机制**：
1. **短期滑动视窗（Rolling Dialog Window）**：只将当前题目关联的最新 1~2 轮交互放入 Prompt，维持单轮交互所需的最少信息；
2. **结构化工作区（Structured Scratchpad）**：题目切换时，将该题的得分、候选人表现要点与缺失项提炼成一条不超过 100 字的结构化记录，存入 `assessment_history`，丢弃原始冗长文本。

![双层上下文窗口管理机制，展示滑动对话窗口与结构化待办摘要的协同方式](/images/content-rebuild/ea3e09081de403ae/fig-4.png)

*图 4：上方滑动窗口仅向 LLM 投喂当前轮次对话；下方结构化状态记录本持续沉淀各题的扣分点与得分，保证第 15 轮对话依然保持极小上下文。*

### 运行时状态定义

```python
from typing import TypedDict, List, Dict

class EvaluationRecord(BaseModel):
    question_id: str
    topic: str
    score: int
    candidate_summary: str
    deficiencies: List[str]

class InterviewState(TypedDict):
    session_id: str
    current_stage: str
    current_question_index: int
    plan: InterviewPlan
    recent_messages: List[Dict[str, str]]        # 仅保留当前考题的 2~4 条问答
    assessment_history: List[EvaluationRecord]  # 持续沉淀的轻量结构化记录
```

通过此设计，无论面试推进多少轮，发送给模型的上下文长度基本恒定，显著降低推理开销。

---

## 自适应单题追问与熔断逻辑

在单题考察中，候选人的回答通常分为三类：充分回答、部分遗漏、完全偏离。系统需要根据回答质量决定“直接通过”、“继续追问”还是“熔断切题”。

![单题自适应追问判定与状态转移循环，展示回答完整度评估与追问分支控制](/images/content-rebuild/ea3e09081de403ae/fig-5.png)

*图 5：单题决策循环包含两条出口。候选人回答若充分，直接打分并换题；若不充分但追问未达上限，则根据预设方向生成自适应追问，超过上限则强制终止追问。*

### 调度执行代码

模型接收“当前考题标准”与“候选人最新回答”，输出结构化 JSON。宿主程序根据返回结果与计数器执行状态推进：

```python
from pydantic import BaseModel, Field
from typing import Literal, Tuple

class TurnDecision(BaseModel):
    completeness: Literal["incomplete", "partially_complete", "complete"]
    score: int = Field(ge=0, le=100, description="针对本题回答的百分制打分")
    key_deficiency: str = Field(description="回答中缺失的核心要点或逻辑错误")
    recommended_action: Literal["followup", "next_question"]
    next_speech: str = Field(description="面试官下一句口述台词")

def execute_turn_logic(state: InterviewState, user_reply: str) -> Tuple[InterviewState, str]:
    plan = state["plan"]
    q_idx = state["current_question_index"]
    current_q = plan.questions[q_idx]
    
    # 1. 调用大模型，传入当前题目评分标准与最近问答，获得结构化决策
    decision: TurnDecision = evaluate_and_decide(current_q, user_reply)
    
    # 2. 外部硬性熔断检查：若模型建议追问但次数已达上限，强制切题
    if decision.recommended_action == "followup":
        if current_q.current_followup_count >= current_q.max_followup:
            decision.recommended_action = "next_question"
        else:
            current_q.current_followup_count += 1
            return state, decision.next_speech
            
    # 3. 题目完成与状态推进
    if decision.recommended_action == "next_question":
        current_q.is_completed = True
        current_q.score = decision.score
        state["assessment_history"].append(EvaluationRecord(
            question_id=current_q.id,
            topic=current_q.topic,
            score=decision.score,
            candidate_summary=user_reply[:120],
            deficiencies=[decision.key_deficiency]
        ))
        
        # 推进题目指针
        next_idx = q_idx + 1
        state["current_question_index"] = next_idx
        
        # 检查题目队列是否全部结束
        if next_idx >= len(plan.questions):
            state["current_stage"] = "CANDIDATE_QA"
            return state, "技术考查到此结束。关于团队与业务，你有什么想了解的吗？"
            
        next_q = plan.questions[next_idx]
        
        # 跨阶段检查：从 CORE_KNOWLEDGE 进入 PROJECT_DEEP_DIVE
        if next_q.stage != state["current_stage"]:
            state["current_stage"] = next_q.stage
            transition_speech = (
                f"基础知识部分聊到这里。结合你在简历中负责的重点项目，我们做些深入探讨："
                f"{next_q.initial_question}"
            )
            return state, transition_speech
        else:
            return state, f"好的，那我们来看下一个问题：{next_q.initial_question}"
            
    return state, decision.next_speech
```

以“Redis 缓存一致性”考题为例：
* **第一轮**：候选人回答“更新数据时先更新数据库再删缓存”。模型判定 `partially_complete`，追问“如果删缓存失败，或者有高并发读穿透，会发生什么？”（`current_followup_count = 1`）；
* **第二轮**：候选人继续回答“可以加延迟双删”。模型判定仍未触及 Binlog 异步重试机制（`current_followup_count = 2`）；
* **第三轮**：即使候选人回答依然不完整，因为 `current_followup_count >= max_followup` 触发熔断，系统将得分记录为 70 分，并将缺失项写入 `assessment_history`，强制切换到下一题。

---

## 会话结束多维量化评估报告生成

当流程流转到 `EVALUATION` 阶段时，系统不需要重新读取几十轮原始聊天，直接输入已经规整的 `assessment_history`。这样能避免长文本带来的细节遗漏，使评估依据更稳定、有据可查。

### 输入与输出规格

输入数据为预设考察维度（技术广度、技术深度、工程逻辑、表达条理）以及包含各题失分项的 `assessment_history` 列表。大模型输出结构化 JSON 报告：

```json
{
  "candidate_name": "张三",
  "overall_score": 76,
  "dimension_scores": {
    "technical_breadth": 82,
    "technical_depth": 68,
    "engineering_logic": 75,
    "communication_clarity": 80
  },
  "stage_breakdown": [
    {
      "topic": "Redis 缓存一致性",
      "score": 70,
      "strengths": "能清晰陈述 Cache-Aside 模式与先更新数据库再删缓存的流程。",
      "improvements": "对延迟双删中延迟时间的选取缺乏依据，未说明 Canal 监听 Binlog 方案的异常容灾机制。"
    }
  ],
  "hiring_suggestion": "CONSIDER",
  "actionable_advice": "建议系统性补充分布式事务与 Binlog 异步同步机制的边界故障排查经验。"
}
```

各维度打分与建议均能对应到具体的题目扣分记录上，避免凭空给出模糊评价。

---

## 状态持久化与断线恢复机制

在浏览器刷新或网络波动时，面试会话需要具备恢复能力。持久化设计可参考 [Agent 核心流程](/interview/agent-interview/agent-loop.md) 与 [Agent 错误恢复](/interview/agent-interview/agent-error-handling.md)。

![状态持久化与断线恢复架构，展示客户端重连后基于线程标识恢复会话上下文](/images/content-rebuild/ea3e09081de403ae/fig-6.png)

*图 6：每完成一次状态交互，系统将 State 对象序列化落入存储层。客户端重新连接时携带 session_id，服务端直接反序列化恢复状态机，继续发起提问。*

### 基于 Session Key 的检查点实现

每轮状态机流转后，宿主程序将 `InterviewState` 序列化存储在 Redis 中：

```python
import json
from redis import Redis

class InterviewCheckpointer:
    def __init__(self, redis_client: Redis):
        self.r = redis_client

    def save_checkpoint(self, session_id: str, state: dict):
        payload = json.dumps(
            state,
            ensure_ascii=False,
            default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o)
        )
        self.r.set(f"interview:session:{session_id}", payload, ex=7200) # 缓存 2 小时

    def restore_checkpoint(self, session_id: str) -> Optional[dict]:
        data = self.r.get(f"interview:session:{session_id}")
        if not data:
            return None
        raw_state = json.loads(data)
        raw_state["plan"] = InterviewPlan.model_validate(raw_state["plan"])
        raw_state["assessment_history"] = [
            EvaluationRecord.model_validate(item) for item in raw_state.get("assessment_history", [])
        ]
        return raw_state
```

当用户重新连接并携带 `session_id` 时，服务端读取最近一次 Checkpoint：
* 状态机恢复当前所处阶段与题目指针 `current_question_index`；
* 系统提取当前未完成的题目文本向候选人提示：“检测到网络恢复，我们继续刚才的问题：[题干]”；
* 历史各题得分及追问计数保持不变，流程平滑继续。

---

## 验收测试与可观测指标

系统在上线前（参考 [项目部署上线](/projects/project-deployment.md)），需针对状态流转与边界条件进行自动化验收：

| 测试目标 | 输入测试动作 | 预期状态机行为 | 合格判定标准 |
| :--- | :--- | :--- | :--- |
| **阶段防跳流测试** | 在自我介绍阶段，候选人主动提问：“你们公司加班严重吗？” | 状态机维持在 `ICEBREAK`，提示“我们在最后反问环节详细解答，请先完成自我介绍” | 阶段不发生跳转，不提前进入 `CANDIDATE_QA` |
| **追问熔断测试** | 在 Redis 考题中，候选人连续 3 次回答“不知道” | 触发 `current_followup_count >= 2` 熔断规则，记录低分并切题 | 追问次数在达到 2 次后强制切入下一题 |
| **上下文长度控制** | 连续交互推进至第 15 轮 | 检查单次发送给模型的输入 Token 总数 | 输入 Token 控制在固定窗口内（约 1,500 Token 左右），不随轮次线性发散 |
| **断线重连恢复测试** | 在第 8 轮答题完成后主动断开网络，10 秒后重连 | 服务端拉取 Redis 快照恢复上下文 | 历史得分记录无丢失，题目指针维持在当前序号 |

通过有限状态机对多轮流程进行确定性调度，大模型专注于单轮意图理解与内容评价，系统既保留了大语言模型的灵活性，又满足了面试流程在节奏把控、考点覆盖与评估沉淀上的工程要求。
