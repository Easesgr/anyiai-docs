---
title: 如何用 LangGraph 编排确定性业务逻辑与大模型任务？
icon: mdi:cog-outline
description: 以工单退款系统为例，使用 LangGraph 混合手写规则与大模型节点，实现状态持久化、条件路由与人工审批介入。
---

在掌握了 [AI 编程助手](/projects/ai-coding-assistant.md) 的单轮工具调用与代码生成后，当面对涉及资金变动、合规审计及长周期跨系统交互的业务时，如果完全依赖大语言模型的自主循环（ReAct），系统容易出现循环发散、非预期工具调用以及延迟不可控等问题。

解决这一问题的务实方案是混合编排：将手写的确定性业务代码与大模型推理节点组合在同一个执行图中。本文以**企业客户支持与工单退款审批系统**为贯穿实例，使用编排框架 LangGraph 构建一个包含数据校验、语义分类、动态条件路由、状态持久化与人工介入（Human-in-the-loop）的完整任务流。

![企业级混合工作流全景封面](/images/content-rebuild/5be45d371db9c7a1/cover.png)

---

### 架构权衡：自主 Agent、固定工作流与混合编排

在设计自动化系统时，工程团队通常在三种架构模式间权衡：

1. **纯自主 Agent**：模型负责规划路径、选择工具并决定何时结束。这种方式处理非结构化开放任务的能力较好，但执行路径具有随机性，可能产生多余的 API 调用，无法准确预估延迟和成本。
2. **固定工作流（Workflow）**：由工程师通过代码或有向无环图（DAG）硬编码执行步骤。代码控制了分支流转与审计边界，没有模型幻觉，但仍会面临网络和外部服务状态变化，且缺乏处理非结构化语义和模糊输入的能力。
3. **混合编排（Hybrid Orchestration）**：输入格式验证、风控规则、数据库读写与外部接口调用由代码显式控制分支与异常处理；意图理解、语义抽取与个性化回复草拟交由大模型节点处理。

![工作流与纯自主Agent权衡分析图](/images/content-rebuild/5be45d371db9c7a1/fig-1.png)

相关架构分析可参考 Anthropic 的技术报告 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 与站内 [AI Workflow 概念指南](/frontier/ai-workflow/what-is-ai-workflow.md)。混合架构通过程序规则为大模型的不确定性输出建立安全防线。

---

### LangGraph 核心概念：State、Node、Edge 与 Reducer

[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) 是面向状态化、多步骤任务流的底层编排框架。它借鉴了图计算模型，支持包含循环的执行图。构建 LangGraph 需要理解四个核心要素：

- **状态（State）**：跨节点传递的全局上下文数据结构，通常使用 Python 的 `TypedDict` 或 `Pydantic` 定义。状态记录流程中的所有输入、中间计算结果与审计日志。
- **归约器（Reducers）**：定义节点返回值如何与现有状态合并。若未指定 Reducer，同名字段默认被新值直接覆盖；若使用 `operator.add` 等归约器，新值将以追加方式合并到现有列表中。
- **节点（Nodes）**：执行图中的基础函数。每个节点接收当前状态作为输入，返回一个包含更新字段的字典。节点内部可以是手写业务逻辑，也可以是模型 API 调用。
- **边（Edges）**：定义节点间的流转关系。普通边（Normal Edge）定义固定的下一步；条件边（Conditional Edge）根据当前状态动态返回目标节点名称。

![LangGraph核心抽象与State流转机制](/images/content-rebuild/5be45d371db9c7a1/fig-2.png)

有关节点职责与拆分粒度，可进一步参考 [工作流节点设计原理](/frontier/ai-workflow/workflow-nodes.md)。

<ReadMoreLock />

---

### 步骤一：定义工作流状态与数据结构

首先安装核心依赖：

```bash
pip install -U langgraph langchain-openai pydantic
```

定义工单输入结构与图全局状态。其中 `audit_logs` 字段绑定了 `operator.add` 归约器，确保每个节点生成的日志追加到列表尾部，而不是相互覆盖。

```python
import operator
from typing import Annotated, Literal, Optional, List
from pydantic import BaseModel
from typing_extensions import TypedDict

class TicketInput(BaseModel):
    user_id: str
    order_id: str
    query: str
    amount: float = 0.0

class WorkflowState(TypedDict):
    ticket: TicketInput
    is_valid: bool
    validation_error: Optional[str]
    category: Optional[Literal["technical", "refund", "general"]]
    approval_status: Optional[Literal["pending", "approved", "rejected"]]
    final_response: Optional[str]
    # 使用 operator.add，节点返回的新日志会自动追加到列表中
    audit_logs: Annotated[List[str], operator.add]
```

---

### 步骤二：实现校验节点与分类节点

系统包含两种性质不同的节点：
- `validate_input_node` 是确定性 Python 函数，不调用大模型，负责校验基础业务规则；
- `classify_intent_node` 调用大模型，将自然语言问题分类为预定义的结构化标签。

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 确定性规则节点：校验输入数据，由程序代码直接判定
def validate_input_node(state: WorkflowState) -> dict:
    ticket = state["ticket"]
    logs = ["[Rules] 开始执行前置业务数据校验"]
    
    if not ticket.order_id or not ticket.user_id:
        return {
            "is_valid": False,
            "validation_error": "缺少必要的 order_id 或 user_id",
            "audit_logs": logs + ["[Rules] 校验失败：字段不完整"]
        }
    if ticket.amount < 0:
        return {
            "is_valid": False,
            "validation_error": "退款金额不能为负数",
            "audit_logs": logs + ["[Rules] 校验失败：金额非法"]
        }
        
    return {
        "is_valid": True,
        "validation_error": None,
        "audit_logs": logs + ["[Rules] 数据格式与基础业务规则校验通过"]
    }

# 大模型决策节点：处理非结构化用户诉求
def classify_intent_node(state: WorkflowState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    query = state["ticket"].query
    
    system_prompt = (
        "你是一个工单分类助手。根据用户输入分类为以下三类之一：\n"
        "- refund: 涉及退款、取消订单、扣费争议\n"
        "- technical: 涉及功能报错、无法登录、系统异常\n"
        "- general: 常见咨询、感谢、使用方法疑问\n"
        "仅输出类别名称，不要输出任何其他内容。"
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ])
    
    category = response.content.strip().lower()
    if category not in ["refund", "technical", "general"]:
        category = "general"
        
    return {
        "category": category,
        "audit_logs": [f"[LLM] 意图识别完成，分类归属: {category}"]
    }
```

---

### 步骤三：编写业务处理节点与条件路由函数

完成分类后，工作流需要分流：
1. 校验失败直接跳转至错误终止节点；
2. 技术问题派发给技术支持节点；
3. 通用问题流向客服应答节点；
4. 退款问题按金额切分：金额 $\le 100$ 元走小额自动退款分支；金额 $> 100$ 元流向人工审批分支。

![条件边与动态分支跳转流向图](/images/content-rebuild/5be45d371db9c7a1/fig-3.png)

```python
def handle_technical_node(state: WorkflowState) -> dict:
    return {
        "final_response": f"技术支持工单已派发，处理查询: {state['ticket'].query}",
        "audit_logs": ["[Action] 已分发至技术工单池"]
    }

def handle_general_node(state: WorkflowState) -> dict:
    return {
        "final_response": "感谢您的咨询，这是标准服务指引说明。",
        "audit_logs": ["[Action] 已输出通用客服响应"]
    }

def auto_refund_node(state: WorkflowState) -> dict:
    amount = state["ticket"].amount
    return {
        "final_response": f"退款金额 ¥{amount:.2f} 已通过小额通道自动退回。",
        "audit_logs": [f"[Action] 资金系统自动退款完成，金额: {amount}"]
    }

def validation_failed_node(state: WorkflowState) -> dict:
    return {
        "final_response": f"工单已驳回，原因: {state.get('validation_error')}",
        "audit_logs": ["[Terminated] 流程因输入非法终止"]
    }

# 路由函数 1：校验分流
def route_after_validation(state: WorkflowState) -> Literal["classify_intent", "validation_failed"]:
    if state.get("is_valid"):
        return "classify_intent"
    return "validation_failed"

# 路由函数 2：意图与金额分流
def route_after_classification(state: WorkflowState) -> Literal["handle_technical", "handle_general", "auto_refund", "human_approval"]:
    category = state.get("category")
    amount = state["ticket"].amount
    
    if category == "technical":
        return "handle_technical"
    elif category == "general":
        return "handle_general"
    elif category == "refund":
        if amount <= 100.0:
            return "auto_refund"
        else:
            return "human_approval"
    return "handle_general"
```

关于状态与边调度的理论背景，可查阅 [工作流基础概念](/frontier/workflow-orchestration/workflow-concepts.md)。

---

### 步骤四：配置持久化层（Checkpointer）

在真实业务中，涉及外部审批的任务需要等待数小时甚至数天。若仅保存在内存中，进程重启会导致上下文丢失。

LangGraph 的执行由超级步（Super-step）驱动。每执行完一个步骤，Checkpointer 将当前状态与快照序列化到存储中。运行时通过 `thread_id` 区分不同的业务会话。只要传入相同的 `thread_id`，工作流即可从上次暂停的位置继续执行。

![Checkpointer持久化与快照存储机制](/images/content-rebuild/5be45d371db9c7a1/fig-4.png)

```python
from langgraph.checkpoint.memory import MemorySaver

# 开发测试使用内存持久化；生产环境可切换为 PostgresSaver 等外部数据库持久化
checkpointer = MemorySaver()
```

持久化机制与容灾设计的系统方案可参阅 [稳定工作流设计](/frontier/ai-workflow/stable-workflow.md)。

---

### 步骤五：实现人工审批中断（interrupt）与图组装

对于大额退款，系统不能直接划扣资金，必须挂起等待人工决策。LangGraph 提供了原生的 `interrupt` 与 `Command` 机制：

1. 节点执行到 `interrupt(...)` 时抛出中断并挂起，暂停执行并不提交当前节点的不完整状态变更；
2. 外部人员查看挂起数据并通过 `Command(resume=...)` 恢复；
3. 节点从函数入口重新进入执行，运行到 `interrupt()` 时直接获取恢复载荷并继续后续业务，提交最终状态（因此 `interrupt` 前的逻辑应避免非幂等的外部写操作）。

![Human-in-the-loop挂起与恢复流程图](/images/content-rebuild/5be45d371db9c7a1/fig-5.png)

```python
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END

def human_approval_node(state: WorkflowState) -> dict:
    ticket = state["ticket"]
    logs = [f"[Approval] 拦截到高额退款申请(¥{ticket.amount:.2f})，触发人工风控中断"]
    
    # 1. 触发中断，挂起当前工作流并向调用方抛出审批载荷
    approval_decision = interrupt({
        "action": "review_refund_request",
        "order_id": ticket.order_id,
        "user_id": ticket.user_id,
        "amount": ticket.amount,
        "reason": ticket.query
    })
    
    # 2. 外部通过 Command(resume=...) 恢复后，代码从此处继续运行
    status = approval_decision.get("status")
    comment = approval_decision.get("comment", "无备注")
    
    if status == "approved":
        final_res = f"大额退款 ¥{ticket.amount:.2f} 经人工审批通过，已提交财务处理。审核意见: {comment}"
        log_entry = f"[Approval] 审核通过: {comment}"
    else:
        final_res = f"大额退款申请已被人工驳回。驳回原因: {comment}"
        log_entry = f"[Approval] 审核驳回: {comment}"
        
    return {
        "approval_status": status,
        "final_response": final_res,
        "audit_logs": logs + [log_entry]
    }

# 组装图结构
builder = StateGraph(WorkflowState)

builder.add_node("validate_input", validate_input_node)
builder.add_node("classify_intent", classify_intent_node)
builder.add_node("handle_technical", handle_technical_node)
builder.add_node("handle_general", handle_general_node)
builder.add_node("auto_refund", auto_refund_node)
builder.add_node("human_approval", human_approval_node)
builder.add_node("validation_failed", validation_failed_node)

builder.add_edge(START, "validate_input")
builder.add_conditional_edges(
    "validate_input",
    route_after_validation,
    {
        "classify_intent": "classify_intent",
        "validation_failed": "validation_failed"
    }
)
builder.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {
        "handle_technical": "handle_technical",
        "handle_general": "handle_general",
        "auto_refund": "auto_refund",
        "human_approval": "human_approval"
    }
)

for target in ["handle_technical", "handle_general", "auto_refund", "human_approval", "validation_failed"]:
    builder.add_edge(target, END)

workflow_app = builder.compile(checkpointer=checkpointer)
```

---

### 步骤六：端到端执行、测试与验收

下面模拟一个金额为 650 元的退款请求，验证工作流在人工审批节点的挂起、状态查询与恢复执行。

```python
config = {"configurable": {"thread_id": "order_session_2026_001"}}

initial_input = {
    "ticket": TicketInput(
        user_id="usr_9981",
        order_id="ord_7762",
        query="商品屏幕出现破损，申请退款 650 元。",
        amount=650.0
    ),
    "audit_logs": ["[System] 工单生命周期开始"]
}

# 阶段一：运行至挂起点
print("=== 第一阶段：触发运行至挂起点 ===")
for step in workflow_app.stream(initial_input, config=config, stream_mode="values"):
    print(f"当前流转阶段，已记录日志数: {len(step.get('audit_logs', []))}")

# 阶段二：检查中断状态与载荷
state_snapshot = workflow_app.get_state(config)
print("\n=== 当前工作流挂起状态检查 ===")
print(f"下一待执行步骤: {state_snapshot.next}")
print(f"挂起时抛出的中断内容: {state_snapshot.tasks[0].interrupts[0].value}")

# 阶段三：外部管理员审批并恢复执行
print("\n=== 第二阶段：管理员执行审批并恢复工作流 ===")
resume_command = Command(
    resume={
        "status": "approved",
        "comment": "用户已提供有效寄回单号，质检合格，同意退款。"
    }
)

for step in workflow_app.stream(resume_command, config=config, stream_mode="values"):
    pass

# 阶段四：验证最终结果
final_state = workflow_app.get_state(config)
print("\n=== 最终执行结果 ===")
print(f"处理结论: {final_state.values.get('final_response')}")
print("全部审计追踪链路:")
for log in final_state.values.get("audit_logs", []):
    print("  ->", log)
```

#### 运行输出验证

```text
=== 第一阶段：触发运行至挂起点 ===
当前流转阶段，已记录日志数: 1
当前流转阶段，已记录日志数: 3
当前流转阶段，已记录日志数: 4

=== 当前工作流挂起状态检查 ===
下一待执行步骤: ('human_approval',)
挂起时抛出的中断内容: {'action': 'review_refund_request', 'order_id': 'ord_7762', 'user_id': 'usr_9981', 'amount': 650.0, 'reason': '商品屏幕出现破损，申请退款 650 元。'}

=== 第二阶段：管理员执行审批并恢复工作流 ===

=== 最终执行结果 ===
处理结论: 大额退款 ¥650.00 经人工审批通过，已提交财务处理。审核意见: 用户已提供有效寄回单号，质检合格，同意退款。
全部审计追踪链路:
  -> [System] 工单生命周期开始
  -> [Rules] 开始执行前置业务数据校验
  -> [Rules] 数据格式与基础业务规则校验通过
  -> [LLM] 意图识别完成，分类归属: refund
  -> [Approval] 拦截到高额退款申请(¥650.00)，触发人工风控中断
  -> [Approval] 审核通过: 用户已提供有效寄回单号，质检合格，同意退款。
```

![端到端测试与追踪视图](/images/content-rebuild/5be45d371db9c7a1/fig-6.png)

---

### 链路排查与可观测性（LangSmith）

在生产环境中排查分支跳转与模型调用异常，可以通过环境变量启用追踪：

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your_langsmith_api_key"
export LANGSMITH_PROJECT="workflow-refund-audit"
```

启用后，每一次执行的步骤耗时、Prompt 与 Token 消耗、状态快照变更以及条件边跳转结果均会记录到看板中。当退款工单发生争议时，运维人员可直接复查该 `thread_id` 对应的历史快照。

关于服务打包与上线部署，可参考 [项目部署上线](/projects/project-deployment.md)。
