---
title: 多 Agent 系统是什么？核心协作模式与工程落地边界
icon: mdi:account-group-outline
tag:
- Agent面试
- 多Agent
- 协作
description: 解析多 Agent 系统的本质、单 Agent 能力瓶颈、主流协作模式与通信机制，通过算例与代码拆解收敛困难、死锁与 Token 放大等落地代价。
---

![多 Agent 系统的协同模式与架构拓扑总览](/images/content-rebuild/a6b1725c73524add/cover.png)

在很多大模型技术面试中，候选人一遇到复杂长流程业务，往往脱口而出“搞几个 Agent 互相讨论”。然而在工程落地中，缺乏严格约束的多智能体系统（Multi-Agent System, MAS）常常带来调用延迟翻倍、Token 消耗急剧放大、甚至陷入死循环无法收敛。面试官考察多 Agent 系统，核心不是测试候选人对某种框架 API 的熟练度，而是考察候选人是否清楚**单 Agent 的物理边界在哪里**、**多 Agent 真正解决的问题是什么**，以及**引入多 Agent 必须付出的系统复杂性代价**。

本文通过一个贯穿始终的“金融研报生成与合规审查”场景，系统梳理多 Agent 的设计原理、协作模式与生产避坑指南。前置概念可参考 [/interview/agent-interview/what-is-agent.md](/interview/agent-interview/what-is-agent.md) 与 [/interview/agent-interview/agent-loop.md](/interview/agent-interview/agent-loop.md)。

---

## 面试标准口述：多 Agent 系统定义与引入原因

如果面试官直接提问：“什么是多 Agent 系统？为什么单 Agent 做不好复杂业务，非要拆成多 Agent？”，建议采用如下层次分明的口述结构：

> **面试口述速记**：
> 多 Agent 系统是由多个拥有**独立系统提示词（System Prompt）、受控上下文与专属工具集**的 Agent 实例组成的软件系统。这些 Agent 依据确定的控制流或通信协议分工配合，共同完成单模型实例难以稳定处理的复合任务。
> 
> 引入多 Agent 系统不是为了形式上模仿人类团队开会，而是为了解决单 Agent 在超长复杂任务下的三大物理瓶颈：
> 1. **上下文注意力稀释与角色冲突**：当单个 Prompt 中塞入相互冲突的规则（例如既要发散创意写出高风险投资观点，又要严苛做合规红线审查），模型注意力分布被严重稀释，遵从约束的准确率会显著下降。
> 2. **工具 Schema 饱和与误选率攀升**：挂载数十个 API 工具时，工具描述本身就会吃满上下文，并大幅推高函数调用（Tool Calling）的参数错误率与误选率。
> 3. **工程解耦与调试黑盒**：单 Agent 维护超长提示词和巨大状态机，任何局部的 Prompt 微调都可能引发未知的全局副作用。
> 
> 拆分为多 Agent 的核心收益是**职责切分与上下文物理隔离**。每个子 Agent 只暴露最小必要的工具和提示词，局部推演收敛后再将结构化结果传递给下游。

![单 Agent 上下文过载与多 Agent 职责隔离机制对比](/images/content-rebuild/a6b1725c73524add/fig-1.png)

如上图所示，当业务复杂度膨胀时，单 Agent 将所有角色设定、工具声明和历史杂质堆叠在同一上下文窗口内，极易导致注意力竞争；多 Agent 架构通过独立上下文切分，确保每个执行单元在清晰受控的局部环境中完成推演。

<ReadMoreLock />

---

## 贯穿场景定义：金融研报生成与合规审查

为了让抽象概念落地，我们以**自动化金融研报生成与合规审查**作为全文的统一工程算例：
* **业务目标**：用户输入股票代码，系统抓取近半年财报和行情数据，输出一份 800 字的投资要点，并严格执行合规审查（严禁保底承诺、必须包含风险揭示、禁止夸大用词）。
* **工具集分布**：
  * 研报与行情工具：`get_financial_report`、`get_stock_quote`、`search_industry_news`（3 个工具，参数与描述合计约 1,200 Token）；
  * 合规工具：`search_compliance_rules`、`verify_blacklisted_terms`（2 个工具，参数与描述合计约 800 Token）。
* **规则集分布**：
  * 研报写作规则：行业分析框架、财务指标折算标准（约 1,200 Token）；
  * 合规审查红线：严格驳回逻辑、禁止性词汇核查规范（约 1,500 Token）。

下面我们基于这个场景，对比不同的系统组织方式。

---

## 多 Agent 核心协作模式：分层、对等与黑板

在代码层面，多 Agent 并不是全员无序广播消息。依据控制权归属与数据传递路径，业内主要演化出三种经典组织模式：

![分层协作、对等网络与黑板模式三种拓扑结构对比](/images/content-rebuild/a6b1725c73524add/fig-2.png)

### 1. 分层协作模式（Supervisor / Orchestrator-Worker）

在分层模式中，设置一个主管节点（Supervisor）。Supervisor 负责解析用户意图，制定执行计划，并将子任务分发给专门的 Worker Agent，最后由 Supervisor 或聚合节点合并输出。

* **控制流**：自顶向下的指令下发与自底向上的数据回传。Worker 之间互不感知、不直接通信，只与 Supervisor 交互。
* **在金融场景的应用**：Supervisor 接收用户提问后，先调用“调研 Worker”搜集财报；拿到初稿后，再将产出移交给“合规 Worker”审查。若合规不通过，Supervisor 携带驳回意见要求调研 Worker 局部重写。
* **适用与权衡**：步骤存在先后依赖、分工明确的流水线场景首选此模式。其代价是 Supervisor 节点本身存在理解偏差的单点风险。

### 2. 对等网络模式（Peer-to-Peer Mesh）

在对等网络中，系统没有常设的中心调度节点。Agent 之间互为对等实体，通过直接发送消息或基于事件触发执行。

* **控制流**：消息传递驱动。Agent A 产出结果后，根据自身内部决策或前置编排协议，直接将上下文移交给 Agent B。
* **在金融场景的应用**：调研 Agent 撰写研报后，直接向审查 Agent 发起审查请求；审查 Agent 提出修改意见并退回给调研 Agent，双方通过多轮对等交互达成一致。
* **适用与权衡**：适用于辩论机制（Multi-Agent Debate）、红蓝对抗攻防演练等需要自组织互动的场景。由于缺乏中心控制，极难保证系统在有限步数内稳定退出，必须设置外部硬性步数截断。

### 3. 黑板模式（Blackboard / Shared State）

黑板模式将通信逻辑与 Agent 执行彻底解耦。系统维护一个中心化的强类型状态空间（即“黑板”或 Shared State）。

* **控制流**：Agent 不直接互发消息，而是监听黑板状态。当黑板上的数据满足某个 Agent 的触发条件（Precondition）时，该 Agent 被激活读取数据，将处理增量写回黑板，进而触发下游节点。
* **在金融场景的应用**：黑板定义为一个状态结构体：`{ raw_data: None, draft: None, compliance_status: PENDING }`。调研 Agent 提取数据后填入 `draft`；合规 Agent 监听到 `draft != None` 且 `compliance_status == PENDING` 时自动启动，并将校验结果覆写为 `PASSED` 或 `REJECTED`。
* **适用与权衡**：适用于异步复杂系统与多源数据汇聚场景。优势是各模块完全解耦；挑战在于黑板的数据 Schema 必须严谨设计，并需要处理并发写入冲突与状态回滚。

---

## 上下文隔离的算例：为什么多 Agent 能提高准确率？

为什么将一个复杂 Prompt 拆分给多个 Agent 能提升大模型的执行质量？我们用前述金融场景做定量分析。

![提示词隔离与工具集切分的输入开销对比](/images/content-rebuild/a6b1725c73524add/fig-3.png)

### 算例假设与对比（同等业务输出）

设输入用户 Query 占用 100 Token，抓取的财务中间数据占用 1,500 Token，最终生成的研报要点占用 800 Token。

| 评估维度 | 单 Agent 混合模式 | 双 Agent 隔离模式（调研 Agent + 合规审查 Agent） |
| :--- | :--- | :--- |
| **挂载工具集** | 全部 5 个工具（2,000 Token） | 调研 Agent 挂载 3 个（1,200 Token）；审查 Agent 挂载 2 个（800 Token） |
| **System Prompt** | 分析 + 合规合并编写（2,700 Token） | 调研 Agent: 1,200 Token；审查 Agent: 1,500 Token |
| **步骤 1：调研生成** | 输入：$2700 + 2000 + 100 = 4800$ Token<br>调用工具提取数据后生成初稿 | 输入：$1200 + 1200 + 100 = 2500$ Token<br>调用工具提取数据后生成 800 字初稿 |
| **步骤 2：合规审查** | 在同一长上下文中追加审查指令，输入累积前序数据达约 7,200 Token | 输入：$1500 + 800 + 800 = 3100$ Token（**仅接收初稿文本，不接收 1,500 Token 原始财务抓取杂质**） |
| **单次推理峰值上下文** | **约 7,200 Token** | **3,100 Token** |
| **全流程总输入 Token** | 完成两步审查累计约 12,000 Token（若单次端到端直出约 4,800 Token 但缺少独立审查） | 完成两步累计输入：$2500 + 3100 = 5600$ Token |
| **工具误选率与冲突** | 高（模型容易在抓取财务数据时误调用合规词典） | 低（两个环境内的工具 Schema 互不重叠，无干扰） |

**算例结论**：
对比表明：在完成相同“生成+审查”两步流程时，单 Agent 因会话历史累积，步骤 2 的单次峰值上下文膨胀至 7,200 Token，总输入消耗达 12,000 Token；双 Agent 模式通过切分上下文，将单次推理峰值压低至 3,100 Token，两步累计总输入降至 5,600 Token。
若与单 Agent 尝试用单个 Prompt 端到端一次性输出（约 4,800 Token）相比，多 Agent 并不保证输入绝对更省，且增加了网络往返延迟。多 Agent 的工程核心价值在于**将单次推理的峰值上下文大幅降低**，并实现**工具 Schema 与历史杂质的物理隔离**，让合规模型只面对干净的目标文本，显著降低工具误选率并避免角色立场冲突。相关上下文治理可参见 [/interview/agent-interview/context-management.md](/interview/agent-interview/context-management.md)。

---

## 状态机编排：以确定性代码组织通信

在生产工程中，多 Agent 绝不是放任多个大模型自由拉群聊天。Anthropic 在《Building Effective Agents》中明确强调：**最成功的实施往往使用简单、可组合的确定性工作流，把模型调用限制在局部的自主节点中**。目前工业界最成熟的状态机实现是以图驱动的 LangGraph。

![中心化状态机与轮转群聊模式的消息传递机制对比](/images/content-rebuild/a6b1725c73524add/fig-4.png)

### 真实机制代码示例：基于 LangGraph 的状态图编排

下面展示金融研报系统的状态机编排代码。状态通过不可变的数据流转，跳转逻辑完全由确定性的 Python 函数判定，而不是完全依赖大模型随机路由：

```python
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END

# 1. 定义类型安全的全局共享状态（黑板）
class FinancialReportState(TypedDict):
    ticker: str
    research_draft: str
    compliance_passed: bool
    critique_notes: List[str]
    revision_count: int

# 2. 节点实现：模型仅在局部节点内执行单一任务
def researcher_node(state: FinancialReportState) -> dict:
    # 调研 Agent 根据股票代码提取数据并撰写初稿
    ticker = state["ticker"]
    critique = state.get("critique_notes", [])
    # 实际场景下此处调用带工具的专门 LLM
    new_draft = f"{ticker} 2024Q3 季度净利润增长 15%，建议积极配置，预期保底收益 20%" 
    return {
        "research_draft": new_draft,
        "revision_count": state.get("revision_count", 0) + 1
    }

def compliance_node(state: FinancialReportState) -> dict:
    # 合规 Agent 仅对研报初稿执行规则审查，无需感知调研工具
    draft = state["research_draft"]
    # 模拟合规检测：命中“保底收益”红线，驳回
    if "保底收益" in draft:
        return {
            "compliance_passed": False,
            "critique_notes": ["严禁出现'保底收益'等承诺性表述"]
        }
    return {
        "compliance_passed": True,
        "critique_notes": []
    }

# 3. 确定性条件边逻辑：防止死循环的熔断看门狗
def check_compliance_edge(state: FinancialReportState) -> Literal["researcher", END]:
    if state["compliance_passed"]:
        return END
    # 硬性步数熔断：修改超过 2 次直接终止，交由人工接管，防止死循环
    if state["revision_count"] >= 2:
        return END
    return "researcher"

# 4. 构建状态图
workflow = StateGraph(FinancialReportState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("compliance", compliance_node)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "compliance")
workflow.add_conditional_edges("compliance", check_compliance_edge)

app = workflow.compile()
```

相比群聊轮转发言（Round-robin Group Chat），这种基于代码的状态图架构在工程调度上带来了明确边界：
1. **契约化状态流转**：通过类型化状态（TypedDict）明确节点间数据传递契约，调研节点产出初稿，合规节点仅产出审查判定，职责与局部上下文清晰解耦；
2. **调度步数硬上限**：通过 `state["revision_count"] >= 2` 代码条件边在调度层设置确定的最大流转次数，避免依赖大模型自主协商退出带来的失控风险（注意：条件分支保证的是调度拓扑具有退出上限，节点内部的模型输出与外部工具状态仍具有随机性）。

---

## 多 Agent 的三大工程代价与故障分析

面试官常会深挖：“你在生产中遇到过多 Agent 系统的哪些严重问题？”必须能够指出多 Agent 带来的额外故障面。

![多 Agent 死锁与循环审查震荡故障分析](/images/content-rebuild/a6b1725c73524add/fig-5.png)

### 1. 通信 Token 放大效应（Token Amplification）

在未经剪裁的群聊模式中，若有 $K$ 个 Agent 在一个共享历史的上下文内轮流发言：
* 假设总共进行了 $T$ 步交互，每次发言长度为 $\bar{L}$，各 Agent 系统提示词长度平均为 $C_{\text{sys}}$。
* 第 $t$ 步发言时，输入上下文包含了前 $t-1$ 步的所有历史消息。
* 整个流程的累积 Prompt Token 消耗为：
  $$\text{Total Input Tokens} = \sum_{t=1}^{T} \left( C_{\text{sys}} + (t-1)\bar{L} \right) = T \cdot C_{\text{sys}} + \frac{T(T-1)}{2} \bar{L} = O(T^2 \bar{L})$$
* 结果表明，Token 消耗随交互轮次 $T$ 呈**二次方级膨胀**。若缺乏状态剪裁，仅仅 4 个 Agent 讨论 5 轮，就会迅速耗尽上下文配额并产生巨额账单。

### 2. 语义漂移与循环挑刺（Oscillation Trap）

在“生成-审查”对抗模式中，极易发生自然语言震荡：
* 调研 Agent 修改了“保底收益”词汇，但为了补充字数，改写了一句“行业基本面稳如磐石”；
* 合规审查 Agent 再次抓住“稳如磐石”判定为夸大宣传，发起二次驳回；
* 调研 Agent 再次改写，引入了新的措辞变动；
* 双方在自然语言层面礼貌推拉，中间状态看似不断推进，实际业务指标从未收敛。若无全局状态比对（State Diff）与硬性退出机制，系统将持续震荡直至超时报错。

### 3. 接口协议死锁（Protocol Deadlock）

单 Agent 的崩溃通常是局部显式的（如工具返回 500 报错）；多 Agent 的死锁往往是**隐式且合法的**：
* Agent A 输出的 JSON 字符串虽然格式合法，但漏掉了下游 Agent B 依赖的关键路由字段；
* Agent B 无法判断意图，返回一条自然语言形式的追问：“请提供具体股票名称”；
* Agent A 的 Prompt 中未定义对该追问的解析分支，误将该追问当作新的待分析指令，再次生成一份不相干的草稿；
* 双方陷入协议死锁，所有 API 请求的 HTTP 状态码均为 200，但业务状态机彻底停摆。系统级容错与重试设计可阅读 [/interview/agent-interview/agent-error-handling.md](/interview/agent-interview/agent-error-handling.md)。

---

## 架构选型树与生产观测指标

何时该引入多 Agent 系统？可遵循以下排查顺序：

![生产级多 Agent 链路追踪与观测面板](/images/content-rebuild/a6b1725c73524add/fig-6.png)

### 选型决策准则

1. **首选单次 Prompt + RAG**：如果单次模型调用配合准确的召回上下文能解决，不要引入 Agent 循环；
2. **次选单 Agent + 工具调用**：如果任务可以通过 3~5 个确定性工具在单模型内循环完成，使用单 Agent 即可；
3. **第三选硬编码 Workflow**：如果业务步骤完全固定（例如：抓取 $\to$ 抽取 $\to$ 格式化），用 Python 代码直接串联多轮单模型调用，不要让大模型去自主协商下一步由谁做；
4. **引入多 Agent 的典型适用特征（建议评估引入）**：
   * **工具集正交且规模膨胀**：系统依赖大量分属不同业务域的工具，若全量挂载会导致单次 Tool Calling 的 Schema 严重超载与误选，需要按职责切分；
   * **角色立场对抗与审查解耦**：业务上需要独立、客观的对抗验证环境（如代码生成与合规安全审查），两者 Prompt 相互冲突无法在单会话内兼顾；
   * **异构模型成本路由**：前序信息提取使用成本低的小模型节点，核心规划与风险决策调用高推理大模型节点（模型分流策略详见 [/architecture/multi-model-routing.md](/architecture/multi-model-routing.md)）。

### 生产观测必备能力

多 Agent 系统的调试不能依靠打终端日志，必须具备分布式追踪能力（如接入 LangSmith 或 OpenInference）：
* **Span 树与因果调用链**：必须清晰追踪哪个 Agent 激活了哪个下游节点，记录每次调用的完整输入快照；
* **状态变更审计（State Diff）**：记录共享状态在每个节点执行前后的数据变更，排查是哪个节点修改了数据导致语义漂移；
* **全局看门狗（Step Limit Guard）**：在系统调度层硬编码全局最大流转步数（如 10 步），一旦超阈值立即熔断并发出告警。

多 Agent 框架底层异同与横向评测详见 [/interview/agent-interview/agent-frameworks.md](/interview/agent-interview/agent-frameworks.md)；子 Agent 的进阶架构可参考 [/interview/agent-interview/multi-agent-collaboration.md](/interview/agent-interview/multi-agent-collaboration.md)。

---

## 高频面试追问与答题陷阱

### 追问 1：在什么任务特征下，将任务拆分给多个专业 Agent 协作明显优于由单个超大 Prompt 的 Agent 执行？

* **合格回答**：任务很长、单个 Prompt 塞不下、需要调用很多工具的时候拆分更好。
* **高分回答**：拆分显著优于单 Agent 必须满足两个硬性特征：
  1. **工具集的领域正交性与 Schema 规模**。当工具总数较多且分属不相干领域时，拆分后每个子 Agent 仅挂载自身所需的最小工具集，能显著降低 Tool Calling 的幻觉与误选率；
  2. **目标约束冲突与对抗审查需求**。当业务要求“发散生成”与“严格风控审查”在提示词层面物理隔离、防止生成偏好污染审查标准时，多 Agent 隔离是必须的；同时，这允许根据任务难度对节点采用**异构模型配置**，降低总运行成本。

### 追问 2：为什么说多 Agent 系统的死循环排查难度远高于单 Agent？

* **合格回答**：因为节点变多了，互相发的消息很多，翻日志翻不过来。
* **高分回答**：单 Agent 的死循环通常是**局部确定性**的，表现为参数固定的重复工具调用，易通过最近动作历史检测拦截；而多 Agent 的死循环是**语义收敛停滞**。每个 Agent 在单步内都产出了格式完全合法、HTTP 200 的有效回复，但由于改写引入了新微瑕，双方在自然语言层面持续挑刺推拉，全局业务状态并未向前推演。这种死锁缺乏单点报错信号，必须依赖全局状态快照差异对比（State Diff）与硬编码的跳转步数看门狗才能被发现与熔断。

### 常见答题错误辨析

* **错误说法**：“多 Agent 系统一定能输出更高质量的结果，因为‘人多力量大’，大模型互相讨论能激发更多智慧。”
  * **正解事实**：多智能体协作增加了系统的不确定性接缝。在缺乏严密收敛校验规则的情况下，误差会在 Agent 链条间级联放大，导致最终输出比单 Agent 结构化工作流更加不可控。
* **错误说法**：“多 Agent 系统的核心在于用一个大模型做全能的 Supervisor，由它动态感知并决定一切跳转。”
  * **正解事实**：完全由 LLM 驱动的自由动态路由非常脆弱。生产级最佳实践是**以确定性的代码状态图与条件边作为骨架，模型自主 Agent 作为局部计算节点**。能用代码 if-else 解决的流程控制，绝不浪费概率性的大模型调用。
