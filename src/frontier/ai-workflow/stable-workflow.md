---
title: 如何构建稳定的 AI 工作流？校验、重试降级与人机协作实践
icon: mdi:shield-check-outline
tag:
- AI Workflow
- 稳定性
- 失败重试
- 人工确认
description: 以发票报销为贯穿案例，解析 AI 工作流中输出校验、错误反馈重试、状态快照幂等恢复与异步人工介入的实现方法与工程边界。
---

大语言模型本质上是通过自回归概率采样逐字生成内容的。这种机制赋予了它处理非结构化文本的灵活性，但在接入企业后端系统时，它也带来了不确定性：格式随时可能畸变、生成的内容可能违反业务规则、外部 API 调用也难免遭遇抖动与超时。

在上一讲[/frontier/ai-workflow/workflow-nodes.md](/frontier/ai-workflow/workflow-nodes.md)中，我们梳理了工作流的基础节点类型（输入、处理、工具、审核、输出）。但在自动化生产环境中，单点节点的输出异常可能引发下游任务崩溃，甚至写坏业务数据。要让 AI 工作流具备生产可用性，系统必须围绕模型构建一套确定性的工程防护网。

本文以一个常见的**企业发票报销审批流水线**（提取发票字段 → 校验供应商库 → 高风险人工审批 → 写入财务系统）为贯穿案例，系统阐述输出校验、重试降级、状态持久化与人机协作的落地实现机制。

![构建高稳定性 AI 工作流的设计全景图](/images/content-rebuild/0634e81d61e55f9a/cover.png)

---

## 一、大模型接入业务系统的三类故障

在发票报销流水线中，大模型主要负责从用户上传的发票图像 OCR 文本中提取结构化信息。在此过程中，系统通常会遇到三类性质截然不同的故障：

1. **数据契约违约（Protocol Violation）**：下游数据库或接口需要标准 JSON 数据，模型却在输出中夹杂了 Markdown 代码块标记（如 ````json ... ````）、末尾漏写闭合括号，或者将约定的驼峰式字段名 `invoiceId` 自由发挥写成 `invoice_number`。解析器在反序列化阶段就会直接报错中断。
2. **业务语义偏离（Semantic Drift）**：模型输出的 JSON 结构完全合法，但字段内容违反实际业务规则。例如将报销金额提取成了负数（`-500.0`），或者提取出的开票方名称在企业供应商白名单库中根本不存在。
3. **系统依赖故障与状态不一致**：大模型托管服务偶发 429 限流或 504 网关超时，或是后续查询外部 ERP 系统的接口连接中断。如果前置步骤已经执行了“扣减部门预算”，而后续节点因网络超时崩溃，系统若直接报错退出，就会导致账实不符的脏数据状态。

![AI 工作流三大典型故障：契约违约、语义偏离与状态破坏分析图](/images/content-rebuild/0634e81d61e55f9a/fig-1.png)

要保障工作流稳定，不能仅仅依靠在提示词中反复强调“请务必输出合规 JSON”，必须在工作流编排层建立由代码兜底的确定性防护机制。

---

## 二、多层校验体系：从低成本类型检查到带反馈自纠错

校验的核心原则是：**从低成本到高成本、确定性规则优先于模型调用**。在发票信息提取节点之后，校验链路按以下三层依次执行：

```text
[ 模型原始文本输出 ]
        │
        ▼
┌────────────────────────┐  校验失败
│ 1. 结构与类型校验 (Schema) │────────────┐
│ (Pydantic / 正则清洗)   │            │
└────────────────────────┘            │
        │ 校验通过                     │
        ▼                             │
┌────────────────────────┐  断言失败   │
│ 2. 确定性业务断言       │────────────┤
│ (正数金额 / 白名单比对) │            │
└────────────────────────┘            ▼
        │ 断言通过              ┌────────────────────────┐
        ▼                       │ 3. 带错误上下文的自纠错 │
[ 流转至下一节点 / 审批流 ]     │ (将具体报错反馈给模型)  │
                                └────────────────────────┘
```

### 1. 结构与强类型校验
模型产物在进入任何业务逻辑前，首先通过本地代码进行文本清洗与类型匹配（例如使用 Python 的 Pydantic 或 JSON Schema）：
- 剥离外围包裹的 Markdown 代码围栏与杂散说明文字。
- 校验必填字段是否存在、数据类型是否匹配（如 `amount` 是否为浮点数、`invoice_id` 是否为非空字符串）。

这一步执行耗时在微秒至毫秒级，且不消耗任何模型 Token，能在第一时间拦截大部分格式畸变。

### 2. 确定性业务逻辑断言
当数据结构合规后，调用轻量业务规则代码进行逻辑校验：
- **数值范围约束**：金额必须大于零（`amount > 0`），税率必须在有效区间内（`0 <= tax_rate <= 1`）。
- **因果与时间逻辑**：开票日期不得晚于当前提交日期。
- **外部存在性检验**：查询本地缓存或数据库，确认发票上的供应商纳税人识别号真实存在。

通过确定性断言，可以拦截“格式合法但业务荒谬”的幻觉内容。

### 3. 带错误上下文的模型自纠错
若前两层校验捕获了异常，简单的“直接重新运行”往往效果有限，因为模型可能在相同输入下重复产生相似错误。此时应启动自纠错闭环：系统提取校验器给出的具体报错（例如 `Field 'amount': Input should be greater than 0`），将其与上一步的错误输出拼接为诊断提示，要求模型仅针对错误点进行修正。

<ReadMoreLock />

![多层校验防御机制数据流向图](/images/content-rebuild/0634e81d61e55f9a/fig-2.png)

---

## 三、节点容错机制：错误分类、修剪重试与兜底降级

当节点发生异常时，盲目重试不仅可能加剧服务拥堵，还会浪费调用额度。工作流引擎必须区分错误类型并采取不同对策。

### 1. 错误分类与退避策略

- **不可重试错误（Non-retryable Errors）**：例如 HTTP 401（鉴权失败）、HTTP 400（参数严重畸变导致接口拒绝）、输入触发内容安全策略拒绝。这类错误重复调用无意义，应立即终止并进入告警流程。
- **瞬态网络与容量错误（Transient Errors）**：例如 HTTP 429（超出并发/配额限流）、HTTP 502/503/504（网关超时）。这类系统层错误应采用**带全抖动的指数退避算法（Exponential Backoff with Full Jitter）**重试：

$$t_{\text{wait}} = \text{Uniform}(0, \min(t_{\text{max}}, t_{\text{base}} \times 2^{\text{attempt}}))$$

其中 $t_{\text{base}}$ 为初始退避基数（如 1 秒），$t_{\text{max}}$ 为最大等待上限（如 30 秒）。加入随机均匀分布的“抖动”，是为了打散同一时刻并发失败的任务，避免所有 Worker 在同一固定间隔后同时发起重试，造成次生流量洪峰。

- **生成格式与语义错误（Content Errors）**：API 正常返回 200，但产物未通过 Pydantic 校验或业务断言。这类错误应走提示词修复分支，通常无需指数等待，直接组装反馈重试即可。

### 2. 修剪上下文的纠错重试
在发票处理中，如果一次识别失败，重试时不应把漫长的系统历史、中间工具日志全部原样传回，而应采用**修剪重试（Context Pruning）**策略：
1. 仅保留最核心的任务目标说明；
2. 附带模型上一轮产出的错误片段；
3. 附带校验器给出的确切错误说明（如“发票号缺少 INV- 前缀”）。

修剪多余上下文既能降低二次推理消耗的 Token，又能减少无关上下文对模型注意力权重的稀释，提高遵循纠偏指令的成功率。

### 3. 兜底降级策略（Fallback）
当节点达到最大重试次数（通常配置为 2~3 次）仍然无法获得合规输出时，流水线不能直接卡死或中断抛错，需要触发降级路径：
- **规则兜底**：使用预设的保守模板填充默认值，将发票打标为“待人工复核”。
- **模型降级**：从复杂且高成本的主力模型切换到结构化指令遵循能力经过专有微调的小参数模型。
- **转入异步工单**：整条流水线在保留当前进度的前提下平滑降级，将任务推入运营处理队列。

![重试、错误上下文组装与降级状态机决策图](/images/content-rebuild/0634e81d61e55f9a/fig-3.png)

---

## 四、状态持久化与断点恢复：Checkpoint 与幂等键

发票报销流程可能包含多个串行步骤，甚至需要等待人工审批数小时。若工作流的全部状态只存放在 Web 服务的内存变量中，一旦服务器重启、发布上线或容器被重新调度，正在执行的工作流就会彻底丢失。

### 1. 基于 Checkpoint 的状态快照
在成熟的状态图编排中（如 [/llm-100/agent-engineering/langgraph-controlled-workflow.md](/llm-100/agent-engineering/langgraph-controlled-workflow.md) 讨论的状态管理机制），每个节点执行完毕后都会将当前系统的运行上下文（State）持久化至可靠存储（如 PostgreSQL 或 Redis）：

- 每一步执行完成后，系统记录 `workflow_run_id`、当前 `node_id` 以及当前状态的序列化快照。
- 若处理在第三步（如查询纳税人状态）时遭遇系统崩溃，服务重启后调度器可直接读取最新的快照版本，从断点直接继续恢复调度，而无需让第一步的发票提取重新消耗一遍 Token。

### 2. 外部调用的幂等性设计
断点续跑带来了一个新的工程要求：**所有对外部产生副作用的节点必须支持幂等性（Idempotency）**。

假设节点 4 是“向财务系统提交报销申请并打款”。如果打款接口请求发出并成功入账，但工作流引擎在写入成功 Checkpoint 前瞬间断电，系统恢复后会重新执行节点 4。如果接口缺乏幂等保护，就会导致向员工重复打款两次。

解决方案是在工作流引擎发起写操作时，基于执行实例与业务操作本身（如 `(workflow_run_id, node_id)` 或包含业务流水号的确定性哈希）计算稳定唯一的**幂等键（Idempotency Key）**，注意不可将重试计数 `attempt_count` 混入写操作幂等键中。财务接口根据此 Key 查重：若该 Key 已经处理过，无论重试发起多少次，均直接返回之前已保存的处理结果，阻止重复扣费或数据二次写入。

![工作流状态快照写入与断点续跑恢复时序图](/images/content-rebuild/0634e81d61e55f9a/fig-4.png)

---

## 五、人机协作机制：异步挂起与恢复唤醒

在发票报销场景中，通常存在业务红线：例如“报销金额超过 10,000 元或发票为新供应商时，必须由财务主管人工审核”。这类高风险动作绝不能完全交由大模型自主决定。

### 1. 长挂起机制（Suspend）
常见的不良设计是在代码中执行 `while` 轮询等待或长连接阻塞。这不仅会长时间霸占应用服务器的 Worker 线程与数据库连接池，一旦连接中断，整个审批流程就会丢失。

高可用的人机协作采用**异步非阻塞挂起（Suspend）**机制：
1. 工作流引擎流转到“主管审批”节点，检测到金额触发审核阈值。
2. 引擎将当前工作流状态标记为 `SUSPENDED`，持久化当前上下文，并生成一个全局唯一的 `resume_token`。
3. 向财务 OA 系统的审批代办队列投递一条审批任务卡片，携带此 Token。
4. 工作流当前线程正常返回退出，**不占用任何运行时 CPU、线程或活动网络连接**。

### 2. 外部唤醒机制（Resume）
财务主管在前端审批界面查看发票信息，操作完成后触发后续流程：
- **批准（Approve）**：主管点击通过，前端调用 `/workflows/{run_id}/resume` 接口，携带 `resume_token`。工作流引擎通过快照恢复上下文，将其调度给下游的“打款与归档”节点继续执行。
- **人工修正（Edit）**：主管发现发票金额存在小数识别错误，直接在表单上将 `12000.50` 修改为 `12000.00`。唤醒接口将修改后的键值覆写进当前 State，直接跳过模型自修正环节向下流转。
- **拒绝（Reject）**：主管驳回，工作流状态流转至 `ABORTED`，触发资源释放与员工撤回通知。

![工作流挂起释放计算资源与人工唤醒流转图](/images/content-rebuild/0634e81d61e55f9a/fig-5.png)

---

## 六、可验证的最小容错执行单元代码

以下 Python 代码使用 Pydantic V2 演示了一个具备**结构清洗、业务断言校验、带反馈修剪重试、以及最终规则降级**的完整发票提取节点。代码可直接在 Python 3.10+ 环境下运行：

```python
import json
from typing import Optional
from pydantic import BaseModel, Field, ValidationError, field_validator

# 1. 契约定义：定义强类型结构与业务逻辑校验规则
class InvoiceExtraction(BaseModel):
    invoice_id: str = Field(..., description="发票编号，以INV-开头")
    amount: float = Field(..., gt=0, description="发票金额，必须大于0")
    vendor: str = Field(..., min_length=2, description="开票方名称")

    @field_validator("invoice_id")
    @classmethod
    def validate_invoice_prefix(cls, v: str) -> str:
        if not v.startswith("INV-"):
            raise ValueError("发票编号格式不合法，必须以 'INV-' 为前缀")
        return v

# 模拟大模型输出（演示第 0 次格式与业务错误，第 1 次前缀错误，第 2 次输出完全合规）
def mock_llm_inference(prompt: str, attempt: int) -> str:
    if attempt == 0:
        # 故障模式 1: 包含 Markdown 标记，且金额违规出现负数
        return "```json\n{\"invoice_id\": \"INV-1001\", \"amount\": -50.0, \"vendor\": \"A\"}\n```"
    elif attempt == 1:
        # 故障模式 2: 修复了负数但发票前缀缺失
        return "{\"invoice_id\": \"1001\", \"amount\": 120.5, \"vendor\": \"Acme Corp\"}"
    else:
        # 合规产出
        return "{\"invoice_id\": \"INV-1002\", \"amount\": 120.5, \"vendor\": \"Acme Corp\"}"

# 2. 具备修剪反馈机制的执行节点
def robust_extract_node(raw_input: str, max_retries: int = 2) -> InvoiceExtraction:
    error_feedback: Optional[str] = None
    last_raw_output: str = ""
    
    for attempt in range(max_retries + 1):
        # 组装修剪后的纠错 Prompt
        if error_feedback:
            prompt = (
                f"任务输入：{raw_input}\n"
                f"上次输出片段：{last_raw_output}\n"
                f"错误原因：{error_feedback}\n"
                f"请针对上述错误进行纠正，仅输出合规的 JSON 对象。"
            )
        else:
            prompt = f"从文本中提取发票信息：{raw_input}"
            
        raw_output = mock_llm_inference(prompt, attempt)
        
        # 基础清洗：去除 Markdown 代码围栏
        cleaned_output = raw_output.strip()
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output.split("\n", 1)[-1]
            cleaned_output = cleaned_output.rsplit("\n", 1)[0].replace("```", "").strip()
            
        # 逐层校验：反序列化与 Pydantic 字段/业务断言验证
        try:
            data_dict = json.loads(cleaned_output)
            validated_obj = InvoiceExtraction.model_validate(data_dict)
            print(f"[校验成功] 第 {attempt} 次尝试提取通过: {validated_obj.model_dump()}")
            return validated_obj
        except (json.JSONDecodeError, ValidationError, ValueError) as err:
            # 捕获校验错误，提取为纯文本上下文供下一轮修正使用
            error_feedback = str(err).replace("\n", "; ")
            last_raw_output = raw_output
            print(f"[校验未通过] 第 {attempt} 次提取失败: {error_feedback}")
            
    # 3. 超限降级策略
    print("[触发降级] 超过最大重试次数，执行默认规则降级")
    return InvoiceExtraction(invoice_id="INV-0000", amount=0.01, vendor="Unknown-Review-Required")

if __name__ == "__main__":
    result = robust_extract_node("收到来自 Acme Corp 的发票，单号 INV-1002，总价 120.5 元。")
    print("最终结果:", result)
```

---

## 七、生产可观测性与验收度量

缺乏可观测性的工作流如同黑盒，链路退化时很难区分是模型能力下降、提示词编写不当还是下游网络故障导致的。

### 1. 结构化 Trace 监控体系
工作流引擎在各节点间传递统一的追踪上下文（符合 OpenTelemetry 规范）：
- `trace_id`：贯穿从发票上传到打款完成的端到端请求。
- `workflow_run_id`：标识一次具体的工作流实例，用于检索快照日志。
- `node_id`：标识当前执行的具体节点（如“发票抽取”或“供应商验证”）。
- `span_id`：标识节点内部的模型推理或工具调用。

每个 Span 必须结构化记录：输入与输出 Token 计数、首字延迟（TTFT）与总延迟、调用模型与版本参数、校验器抛出的原始错误信息、重试次数及退避时长。

### 2. 生产验收指标度量

各指标的基线水平与具体的业务复杂度密切相关。下表给出了在结构化票据提取业务中，工程团队通常关注的核心度量指标及其因果意义：

| 核心指标项 | 统计公式 | 建议观测重点 | 场景因果与工程机理 |
| :--- | :--- | :--- | :--- |
| **一次通过率 (FPPR)** | $\frac{\text{无重试直接成功的执行数}}{\text{总执行数}}$ | 灰度期重点关注趋势 | 反映提示词质量与基础模型指令遵循度。若骤降，需排查模型版本是否发生静默变更或上游 OCR 质量恶化 |
| **纠错重试成功率** | $\frac{\text{触发重试后最终成功的执行数}}{\text{总触发重试的执行数}}$ | 评估纠错 Prompt 有效性 | 衡量“带反馈错误修剪”策略的有效程度。若过低，说明反馈给模型的错误堆栈不清晰或上下文修剪过度 |
| **降级触发率 (Fallback Rate)** | $\frac{\text{进入兜底分支的执行数}}{\text{总执行数}}$ | 预警核心业务风险 | 衡量系统不可用边界。降级率升高通常意味着下游服务中断或长尾样本集中爆发 |
| **人工审批挂起等待时长** | $\text{Avg}(\text{唤醒时间} - \text{挂起时间})$ | 监控业务 SLA | 衡量人机协作审批链条的流转效率，及时发现待办工单积压与处理人瓶颈 |

![工作流全链路 Trace 追踪与可观测性看板示意图](/images/content-rebuild/0634e81d61e55f9a/fig-6.png)

---

## 本讲阅读自测

1. **当模型输出的 JSON 存在格式语法错误时，为什么直接把原始 Prompt 原封不动重新请求一次往往是低效的？正确的上下文组装策略是什么？**
2. **在发票打款等写操作节点崩溃后，断点续跑如何避免向外部系统发起二次重复打款？**
3. **在需要人工审批的高风险节点中，为什么不能使用长期挂起的长连接或线程 sleep，而必须采用 Suspend/Resume 机制？**

---

## 推荐阅读与下步进阶

- 对比工作流与自主 Agent 的边界条件：[/interview/ai-system-design-interview/rag-agent-workflow.md](/interview/ai-system-design-interview/rag-agent-workflow.md)
- 掌握基于状态机的图编排代码实现：[/llm-100/agent-engineering/langgraph-controlled-workflow.md](/llm-100/agent-engineering/langgraph-controlled-workflow.md)
