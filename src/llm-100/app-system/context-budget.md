---
title: 上下文长度不够怎么办？Token预算分配与历史截断压缩实战
icon: mdi:book-open-page-variant
description: 详解多轮对话中的上下文预算切分公式、两阶段 Token 计数、基于轮次的成对截断算法，以及双轨异步摘要状态机与分层淘汰设计。
---

在多轮对话与 Agent 系统中，上下文长度绝非可以随意填塞的无限资源。即便模型宣称支持 32K、128K 甚至更长的窗口，盲目塞入全部历史记录依然会带来显存占用激增、首字延迟（TTFT）延长、长文本中间信息注意力衰减（Lost in the Middle）以及调用成本失控等问题。相关机制分析可参考[上下文长度的基本概念](/fundamentals/inference-and-generation/context-length.md)与[长上下文应用的常见缺陷](/frontier/long-context-app/long-context-pitfalls.md)。

上一讲在[多轮对话历史管理](/llm-100/app-system/conversation-history.md)中讨论了会话数据的底层存储与查询设计。本讲聚焦会话数据输入模型推理接口前的处理流程：如何建立明确的配额划分，在毫秒级时间内完成 Token 预算分配，并通过成对截断与异步语义压缩，保证对话的连贯性与系统的稳定性。

![上下文预算管理与动态压缩全景架构](/images/content-rebuild/221bb91652d39c7f/cover.png)

---

### 1. 贯穿算例与 Token 预算分配公式

大模型推理服务（例如 OpenAI 规范接口）对每一次 API 请求施加的是「输入 Token 数 + 输出预留 Token 数（max_tokens）」的总和上限。如果输入历史与 `max_tokens` 之和超过物理上下文上限，API 会在前置校验阶段直接报错拒绝服务；而若为了避免报错盲目压缩 `max_tokens`，又会导致模型在生成中途触顶截断，输出残缺的 JSON 或中断的代码。

为了让计算清晰可见，本文设定一个贯穿全程的技术支持 Agent 算例：
*   模型标称物理上下文上限：$L_{\text{ctx}} = 8,192$ Token。
*   业务要求模型必须能输出完整排障步骤，硬性预留输出：$R_{\text{gen}} = 2,048$ Token（对应 API 的 `max_tokens`）。
*   系统提示词与角色规范：$T_{\text{sys}} = 600$ Token。
*   注册了 3 个排障排查工具的 JSON Schema 定义：$T_{\text{tools}} = 500$ Token。
*   当前用户提交了一条带有报错日志的新提问：$T_{\text{query}} = 800$ Token。
*   安全缓冲余量：$T_{\text{margin}} = 128$ Token。
*   历史数据库中已累积了 10 轮对话，累计占用 6,500 Token。

在组装请求前，必须通过减法计算出当前轮次允许塞入历史对话的最大预算：

$$B_{\text{history}} = L_{\text{ctx}} - R_{\text{gen}} - T_{\text{sys}} - T_{\text{tools}} - T_{\text{query}} - T_{\text{margin}}$$

将算例数值代入公式：

$$B_{\text{history}} = 8192 - 2048 - 600 - 500 - 800 - 128 = 4116 \text{ Token}$$

历史累积的 6,500 Token 明显超出了 4,116 Token 的剩余预算。如果不做处理直接发送，仅输入部分就达到 $6500 + 600 + 500 + 800 = 8400$ Token，加上预留生成的 2,048 Token，API 校验总需求达到 10,448 Token，远超 8,192 Token 上限，直接触发模型服务报错（HTTP 400 context_length_exceeded）。

正文下图展示了这一配额切分逻辑：系统自底向上优先扣除生成预留与当前必要输入，剩余空间才是历史消息动态分配的有效预算。

![上下文窗口配额划分模型](/images/content-rebuild/221bb91652d39c7f/fig-1.png)

若当前提问本身极其庞大，导致计算出的 $B_{\text{history}} \le 0$，系统必须触发降级路径（如提示用户缩小输入范围或剥离可选工具 Schema），不能盲目向下游提交请求。

<ReadMoreLock />

---

### 2. Token 计数开销权衡：精确分词与两阶段快速估算

在决定历史消息去留前，程序必须知道每条消息的具体 Token 开销。工程中通常存在两种计数路径：

1.  **精确分词计数**：在应用服务器运行专属分词库（如 Python 下的 `tiktoken` 或 HuggingFace Tokenizers）加载词表进行分词。这种方式计算精准，但需要占用 CPU 计算资源。在高并发网关中，如果对几十条历史消息逐一执行完整的 BPE（Byte Pair Encoding）分词，会增加主线程的计算开销与首字延迟（TTFT）。
2.  **字符比率近似估算**：依据字符长度与经验比率估算（例如英文通常折算为 1.3 Token/词，通用中文文本通常折算为 0.6 至 1.2 Token/汉字）。该方案时间复杂度为 $O(1)$，没有计算延迟，但在代码片段、转义字符、公式或多语言混合场景下，估算误差可能超过 30%。

为了兼顾响应速度与边界安全，网关服务常采用**两阶段分层计数机制**：

*   **阶段一（快速过滤）**：利用低成本字符比率乘上安全系数（如 1.2）进行保守估算。若估算出的总 Token 数处于宽松安全水位（例如总 Token 估算值 $< 0.75 \times L_{\text{ctx}}$），直接放行请求，无需执行精细分词。
*   **阶段二（精确校准）**：当估算值接近警戒水位，或者历史消息中含有大段代码与 JSON 数据时，启动精确分词器。为了避免重复全量计算，通常对已存入数据库的历史消息在写入时缓存其精确 Token 数，此时仅需对最新输入与处于截断边界的消息做精确分词。

正文下图展示了该两阶段校验的流转机制：通过估算分流避免无谓的 CPU BPE 运算。

![两阶段 Token 计数与预算核验机制](/images/content-rebuild/221bb91652d39c7f/fig-2.png)

---

### 3. 历史截断策略：物理截断与角色成对截断

在上述技术支持 Agent 算例中，历史消息总共 6,500 Token，而预算配额 $B_{\text{history}}$ 仅为 4,116 Token，必须淘汰约 2,400 Token 的旧记录。

#### 3.1 单条物理截断的隐患
最简单的做法是把历史消息视为平铺队列，从最旧的消息开始单条弹出。这种裁剪方式会引发严重问题：
1.  **问答语义破损**：如果删除了早期的 `user` 提问，却保留了对应的 `assistant` 回复，大模型在阅读上下文时就会面对一段“凭空出现的回答”，缺乏前置问题约束，容易误解上下文语义。
2.  **工具调用协议损坏（孤立工具返回）**：在遵循主流函数调用规范的接口中，一次工具交互由 `assistant(tool_calls=[...])` 与紧随其后的若干条 `tool(tool_call_id=...)` 强绑定。如果截断算法机械裁掉了前置的发起消息，却残留了后置的工具执行结果，模型 API 将直接抛出 HTTP 400（参数校验失败：找不到对应的 `tool_call_id`）。

#### 3.2 角色成对截断与原子单元包装
可落地的工程截断方案必须以**逻辑交互轮次（Turn）**或**原子消息组（Atomic Message Group）**为最小裁剪单元：
*   普通单轮对话：`[user, assistant]` 绑定为一个单元，淘汰时整组剔除。
*   工具交互轮次：`[user, assistant(tool_calls), tool(result_1), tool(result_2), assistant(final_reply)]` 绑定为一个原子单元。只要预算不足以容纳该组，就整组淘汰，绝不拆解内部关联。

正文下图对比了机械逐条截断导致的协议报错与成对安全截断的结构差异：

![基于轮次成对截断与孤立消息规避流程](/images/content-rebuild/221bb91652d39c7f/fig-3.png)

下面给出一个以轮次为原子单元、由新到旧保留历史的 Python 算法实现（基于 Python 3.10+）：

```python
from typing import List, Dict, Any

class AtomicMessageGroup:
    def __init__(self, messages: List[Dict[str, Any]], estimated_tokens: int):
        self.messages = messages
        self.estimated_tokens = estimated_tokens

def build_atomic_groups(raw_history: List[Dict[str, Any]]) -> List[AtomicMessageGroup]:
    """
    将平铺的消息列表聚合成不可分割的原子单元。
    支持普通问答成对与工具调用链成对包装。
    """
    groups: List[AtomicMessageGroup] = []
    current_group: List[Dict[str, Any]] = []
    current_tokens = 0
    
    for msg in raw_history:
        # 示例：此处使用简单估算，生产环境可结合精确 Tokenizer 或读取缓存值
        tokens = max(1, int(len(str(msg.get("content", ""))) * 0.8))
        current_group.append(msg)
        current_tokens += tokens
        
        # 当遇到无后续 tool_calls 的 assistant 消息时，认为当前交互轮次闭合
        is_tool_call = bool(msg.get("tool_calls"))
        if msg.get("role") == "assistant" and not is_tool_call:
            groups.append(AtomicMessageGroup(current_group, current_tokens))
            current_group = []
            current_tokens = 0
            
    # 遗留未闭合消息（例如当前轮次尚未回复的用户提问）作为独立单元保存
    if current_group:
        groups.append(AtomicMessageGroup(current_group, current_tokens))
        
    return groups

def prune_history_by_budget(
    groups: List[AtomicMessageGroup], 
    budget_history_tokens: int
) -> List[Dict[str, Any]]:
    """
    按照时间由新到旧累加原子单元，一旦超出预算即停止，保留最新的完整轮次。
    """
    accepted_messages: List[Dict[str, Any]] = []
    consumed_tokens = 0
    
    # 倒序遍历：从最新对话往历史回溯
    for group in reversed(groups):
        if consumed_tokens + group.estimated_tokens <= budget_history_tokens:
            # 逆序压入暂存列表
            accepted_messages.extend(reversed(group.messages))
            consumed_tokens += group.estimated_tokens
        else:
            # 空间不足以容纳整个原子单元，停止装入，舍弃更早历史
            break
            
    # 翻转恢复正常时间顺序
    accepted_messages.reverse()
    return accepted_messages
```

---

### 4. 语义压缩：后台异步摘要状态机

截断策略虽然能确保 Token 不超限，但在长对话中，早期的关键需求和背景信息会被直接丢弃。为了保留早期核心事实，需要引入**上下文摘要压缩（Context Summarization）**机制。

#### 4.1 为什么不要在用户请求关键路径执行同步摘要
在用户发起对话的 HTTP/SSE 线上请求线程中实时调用模型生成历史摘要存在明显缺陷：
1.  **延迟倍增**：生成一段摘要往往需要 1 至 3 秒甚至更长的时间，直接串联在请求主流程中会导致首字延迟（TTFT）大幅恶化。
2.  **级联故障风险**：摘要调用与主回复调用串联，一旦摘要模型接口发生限流、网络抖动或超时，会导致整个用户请求直接失败。

#### 4.2 双轨异步摘要设计
推荐采用**双轨解耦架构**。会话上下文在逻辑上被划分为三个区间：
1.  **全局核心摘要（Summary Block）**：对更早轮次的高度提炼（如用户身份、明确偏好、已确立的业务前提）。
2.  **滚动原始窗口（Sliding Window Buffer）**：保留最近 $N$ 轮未压缩的高保真原始问答。
3.  **待压缩候选区（Candidate Pool）**：当滚动窗口中的消息数量或 Token 达到高水位线时，划入待压缩候选区，触发后台异步任务进行提炼。

正文下图展示了这一异步处理流程：线上主会话通过降级成对截断保障毫秒级响应，后台异步任务独立消费历史记录，并通过版本号或 CAS（Compare-And-Swap）机制将生成的新摘要写回存储。

![双轨异步摘要生成与回填状态机](/images/content-rebuild/221bb91652d39c7f/fig-4.png)

在后台异步摘要未完成前，线上请求继续使用上一个版本的旧摘要，配合成对截断组装 Prompt。后台任务成功生成新摘要后，将对应消息标记为“已归档”，并原子更新摘要字段。关于更深层次的动态记忆抽取策略，可参阅[多轮对话上下文高级管理](/frontier/multi-turn-dialogue/context-management.md)。

---

### 5. 分层优先级淘汰与上下文装配

在完整的 Agent 场景中，输入上下文往往由异构数据混编而成：系统指令、Few-Shot 示例、用户画像、RAG 检索知识切片、历史会话以及当前提问。如果仅使用简单的先进先出（FIFO）队列淘汰历史，无法满足不同类型信息的保留需求。

系统应当建立**分层优先级装配模型**，在总 Token 预算耗尽时，按业务优先级逆向削减：

| 优先级层级 | 上下文组件 | 淘汰策略 | 处理说明 | 关联参考 |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0（不可淘汰）** | System Prompt、输出格式、当前 Query、生成预留 | **绝对保护** | 维持模型基本指令对齐与输出空间。若空间不足直接报错返回。 | [高质量上下文组织](/frontier/context-engineering/high-quality-context.md) |
| **Tier 1（高保真保留）** | 最近 1 轮原始问答、工具定义（Tool Schemas） | **强保护** | 确保直接问答的连贯性与工具调用的参数定义可用。 | [Prompt 模板系统设计](/llm-100/app-system/prompt-template-system.md) |
| **Tier 2（核心背景）** | 用户偏好画像、历史提炼摘要 | **结构化压缩** | 占用空间小（通常 200~500 Token），提供全局业务背景。 | [大模型应用系统架构](/llm-100/app-system/llm-app-architecture.md) |
| **Tier 3（弹性检索层）** | RAG 检索召回的外部文档切片 | **按分数逐步淘汰** | 根据检索相似度分数，从最低分切片开始逐个丢弃。 | [知识库问答系统设计](/llm-100/app-system/enterprise-knowledge-qa.md) |
| **Tier 4（优先淘汰层）** | 较早的原始历史对话轮次 | **成对截断淘汰** | 优先进入后台异步摘要流程，在线装配时首批予以剥离。 | [多轮对话历史管理](/llm-100/app-system/conversation-history.md) |

正文下图展示了分层装配流水线：装载引擎从 Tier 0 向 Tier 4 逐级核算预算，遇到限额瓶颈时，自底向上削减低优先级模块，确保核心指令与当前输入不受挤压。

![分层优先级淘汰与上下文装配流程](/images/content-rebuild/221bb91652d39c7f/fig-5.png)

---

### 6. 工程监控与异常降级兜底

上下文预算管理器处于业务调用的关键网关位置，系统必须建立对应的兜底机制与监控指标：

1.  **摘要超时与失败降级**：如果后台异步摘要任务发生网络超时、模型返回格式解析失败或达到重试上限，系统必须支持失败降级，保持主会话继续使用“硬淘汰老数据并保留旧摘要”模式，绝不能因为摘要任务失败阻塞用户主会话。
2.  **上下文饱和度告警**：监控指标中应统计**上下文饱和度**（即 $\text{Actual Tokens} / L_{\text{ctx}}$）。当该指标频繁高于 90% 且连续触发截断时，应发出告警，提示业务侧精简 System Prompt 或评估切换至更长上下文基座。
3.  **调试排查元数据**：每次向模型服务发起请求时，网关应在 Trace 日志中记录以下信息：
    *   `budget_history_allocated`：本次动态分配给历史消息的可用配额。
    *   `turns_retained` 与 `turns_evicted`：保留与丢弃的消息轮次数。
    *   `truncation_strategy`：本次触发的淘汰模式（如纯截断、RAG 削减或摘要替换）。
    *   `tokenizer_latency_ms`：分词计数所消耗的 CPU 时间。

正文下图总结了当极端超限或后台子任务失败时，系统从正常装配到多级降级的兜底处理流程。

![上下文预算耗尽与降级兜底拓扑](/images/content-rebuild/221bb91652d39c7f/fig-6.png)

---

### 7. 读者检验与总结

通过对 Token 预算进行硬性切分、按交互轮次成对截断历史，并配合后台异步摘要，系统能够在保证首字响应速度的同时，维持长期对话的连贯性与调用协议的正确性。请通过以下两个问题检验对核心机制的掌握：

1.  **问**：在裁剪历史对话时，如果直接机械丢弃最早的一条消息，为什么可能会破坏对话协议或语义？应该如何解决？
    *   **答**：若机械丢弃单条消息，若丢弃的是早期的 `user` 提问，会导致保留的 `assistant` 回复失去上下文参照；若丢弃了带有 `tool_calls` 的回复却保留了后置的 `tool` 执行结果，会导致接口校验因找不到对应的 `tool_call_id` 而报 HTTP 400 错误。解决方法是将普通 `[user, assistant]` 问答或包含工具调用链的整套消息包装为不可分割的原子消息组，裁剪时以原子组为单位倒序淘汰。
2.  **问**：为什么不能在用户的请求主流程中同步调用大模型生成历史摘要？系统应如何设计兜底路径？
    *   **答**：同步生成摘要会成倍增加用户的首字等待时间（TTFT），且一旦摘要接口发生网络抖动、限流或超时，会直接导致用户当前会话请求崩溃。系统应采用双轨异步设计：用户主请求只依赖当前已有的摘要快照并执行成对物理截断，保证快速响应；后台异步 Worker 独立消费旧消息生成新摘要并原子回填。如果后台摘要失败，主请求继续降级为物理截断，确保会话服务可用性。
