---
title: AI Agent 的记忆系统是什么？短期记忆、长期记忆和用户画像有什么区别？
icon: mdi:brain
tag:
- Agent 记忆系统
- 短期记忆
- 长期记忆
- 用户画像
description: 系统解析大模型 Agent 的短期记忆、长期情景记忆与结构化用户画像的分工、存储介质、生命周期与组装时序。
---

大语言模型（LLM）本身在处理请求时是无状态的：它不保留上一次调用的上下文，也不在内部维护跨会话的数据记录。开发者之所以觉得智能体（Agent）能记住对话，是因为外部工程程序在每次调用模型前，把必要的历史信息重新拼装成了提示词（Prompt）。

如果把所有历史记录不加筛选地追加到提示词中，很快就会遇到模型输入窗口超限、推理延迟上升以及关键指令被无关信息稀释的问题。为了兼顾对话连续性与输入成本，现代 Agent 通常采用三层分工的记忆体系：

1. **短期记忆（Working Memory）**：维护当前任务步骤和当前对话的消息序列，直接驻留在提示词上下文内。
2. **长期记忆（Long-Term Memory）**：将跨越会话的历史经历存放在外部数据库（如向量库）中，在需要时按相关度检索并按预算截断填入。
3. **用户画像（User Profile）**：将高频、稳定的用户属性或偏好以结构化键值（Key-Value）保存，支持确定性读取与覆盖。

以一个代码审查助手为例：用户今天说“帮我写一个拉取用户数据的异步函数”。此时，短期记忆保存刚才两轮关于参数校验的讨论；用户画像提供“该用户习惯使用 Python 且遵循 PEP 8”；长期记忆则按需调出三周前用户提过的“该项目数据库连接池已切换为 asyncpg，查询时务必配置 5 秒超时”。三者拼接在一起，模型才能给出符合要求的代码。

![Agent 记忆系统的三层分工体系全景](/images/content-rebuild/d0b7ce518736e1b6/cover.png)

上图展示了 Agent 记忆系统的三层分工体系。讲解员在中央调度台前协调这三类不同介质的状态流向：短期工作上下文控制当前会话，长期记忆提供历史检索，用户画像提供确定性键值。接下来，我们从模型推理的基础特性开始分析。

---

## 大模型推理是无状态的：为什么模型记不住上一轮对话

大语言模型在标准 API 部署下是一个纯粹的前向计算函数。给定当前输入序列，它计算下一个 Token 的条件概率分布。

两次独立的 API 调用之间，模型内部不会自动保留记忆，原因在于：

* **模型权重完全只读**：在线推理不触发梯度计算，也不会更新参数权重。模型本身不会因为回答了一次提问而改变内在结构。
* **KV 缓存随请求结束而释放**：为了加快自回归生成的解码速度，推理框架会在 GPU 显存中维护键值缓存（KV Cache）。但这部分显存属于单次请求执行期间的临时缓存，生成完毕、连接断开后即被回收，不会跨会话保留。
* **模型没有可写寄存器**：LLM 内部不存在像传统操作系统那样的持久化非易失存储单元。

![大模型单次前向推理的无状态物理机制与显存生命周期](/images/content-rebuild/d0b7ce518736e1b6/fig-1.png)

上图说明了单次前向推理的无状态物理过程。讲解员指出了两次独立 API 调用之间完全断开的运行状态：模型参数完全静态，显存中的 KV Cache 在连接断开后立即释放。因此，Agent 的“连续记忆”是由外部程序在每次发起调用前重新组装输入上下文实现的。

既然必须由外部程序拼装输入，为什么不能把过去的每一次交互无脑拼进输入中？

1. **上下文窗口有物理上限**：即使长文本模型支持数十万 [Token](/fundamentals/token-and-tokenizer/what-is-token.md)，上下文容量仍有边界。
2. **注意力稀释与干扰**：大量与当前任务无关的陈旧历史会增加上下文噪声，降低模型准确遵循关键指令的概率。
3. **推理延迟与成本增长**：Prefill（提示词预填充）阶段的计算与显存开销随输入序列长度增加而上升，盲目拼接会导致单次调用成本和首字返回时间（TTFT）大幅增加。

关于 Agent 的基本运行框架与系统设计，可参考 [AI Agent 是什么？](/interview/agent-interview/what-is-agent.md)。

---

## 短期记忆（Working Memory）：作为当前工作区的上下文窗口

**短期记忆**在工程上对应 Agent 的**工作上下文（Context Window）**。它是当前决策步骤中，模型能直接通过自注意力机制关注的信息工作区。

在 [Anthropic 建议的 Agent 设计模式](https://www.anthropic.com/engineering/building-effective-agents)与 [LangGraph 编排框架](https://docs.langchain.com/oss/python/langgraph/overview)中，工作记忆主要由三部分构成：

1. **系统预设（System Instructions）**：定义 Agent 的角色身份、可用工具协议、安全边界与输出格式。
2. **即时会话列表（Session Messages）**：用户与模型在当前会话中的最近几轮对话消息。
3. **思考过程与工具调用轨迹（Scratchpad / State Tree）**：Agent 在执行复杂任务时的中间思考、发出的工具调用参数（Tool Call）以及外部工具返回的原始观测数据（Observation）。

![短期工作记忆的组成结构与滑动窗口修剪流程](/images/content-rebuild/d0b7ce518736e1b6/fig-2.png)

上图展示了短期记忆在上下文窗口中的物理排布。讲解员展示了工作记忆从系统指令、会话缓冲到工具状态树的组装过程，并在边界处通过修剪器丢弃过时的中间过程。

### 短期记忆的常见维护策略

为了防止短期记忆耗尽上下文预算，工程上通常采用以下控制手段：

* **滑动窗口截断（Sliding Window Buffer）**：仅保留最近 $K$ 轮对话，直接丢弃更早的消息。实现简单，但可能丢失会话最初由用户设定的全局约束。
* **工具调用轨迹修剪（Scratchpad Pruning）**：复杂任务中工具返回的内容往往非常冗长（例如爬取了一整段网页或日志）。一旦后续推理步骤已经提炼并消费了这些内容，编排框架就应将详细日志修剪替换为紧凑摘要，释放空间。
* **滚动摘要生成（Rolling Summarization）**：当滑出窗口的消息达到一定数量时，调用轻量模型将其压缩为一段语义摘要，拼接在后续上下文前部。

短期记忆的生命周期只与**当前任务或当前会话**绑定。会话结束或任务到达终点（`END`）后，短期工作区即被释放或转存为静态日志。

---

<ReadMoreLock />

## 长期记忆（Long-Term Memory）：保存在外部存储中的历史经验

短期记忆无法保留跨会话信息，强行拼接所有历史又不可持续。**长期记忆（Long-Term Memory）**解决的是跨会话信息的持久化与按需调取问题。

长期记忆保存在外部介质中（如向量数据库、倒排索引或文档数据库），**平时不驻留在模型的上下文内**。只有在当前任务与某段历史记录产生关联时，系统才会将其检索出来并注入到提示词中。

![长期记忆的写入编码、存储索引与按需检索激活闭环](/images/content-rebuild/d0b7ce518736e1b6/fig-3.png)

上图描绘了长期记忆的写入与唤醒链路。讲解员演示了对话历史如何通过向量嵌入编码进入外部数据库，并在新任务到来时，经由相似度检索与重排重新激活，作为动态上下文回填给模型。

### 长期记忆的生命周期

长期记忆通常遵循以下处理链路：

1. **提取与清洗**：在任务执行完毕后，系统过滤掉无价值的寒暄、重试错误和临时中间状态，提炼出有参考价值的事实记录或决策结论。
2. **切分与嵌入（Indexing）**：将清洗后的事实文本切分为段落（Chunk），利用嵌入模型计算向量并写入向量数据库建立索引。向量特征原理详见 [Embedding 到底是什么？](/fundamentals/embedding/what-is-embedding.md)。
3. **按需检索（Retrieval）**：当收到新任务时，从用户输入或当前推理目标中提取检索词，结合语义相似度（如余弦相似度）与时间衰减权重进行检索，召回 Top-$K$ 条历史片段。
4. **重排与注入（Hydration）**：对召回内容进行相关性重排后，按预留预算格式化拼入当前轮次的提示词中。

### 边界与限制：按需检索不等于完美回忆

需要明确的是：**长期记忆的检索是概率性匹配，并不能保证必定召回正确信息，更无法彻底消除遗忘**。

* **检索假阳性与漏检**：语义相似度高并不代表逻辑完全相关。可能召回相似但不适用的陈旧方案，也可能因为措辞差异而漏掉关键历史。
* **上下文预算限制依然存在**：长期记忆召回出的内容最终仍要塞入提示词。如果检索出的片段过多，超出预留的 Token 预算，系统仍然只能按分数截断淘汰低分条目。

### 长期记忆与标准文档 RAG 的区别

虽然底层都使用检索和嵌入技术，但两者的对象和目标不同：

* **标准文档 RAG**：检索的是静态、客观、通用的公共资料库（如企业知识库、产品手册、法律条文），是“查参考书”。
* **Agent 长期记忆**：检索的是动态、包含时间戳、与特定用户或任务绑定的主观经历记录（如“用户上一周做过某次重构决策”），是“回忆自身经历”。

更多记忆类型面试解析可参考 [Agent 中的 Memory 有哪些类型？](/interview/agent-interview/agent-memory.md)。

---

## 用户画像（User Profile）：确定性的结构化偏好与状态记录

向量数据库擅长模糊语义匹配，但在处理“明确属性更新”时存在天然短板：**概率匹配无法保证字段级别的确定性覆写与一致性**。

例如，用户第一周说：“我平时主要写 Python”；第三周说：“我现在负责的项目全部转用 Go 语言开发了”。如果仅依赖向量检索，这两条记录都会作为语义相似的高分结果被捞出。模型在提示词中同时读到相互冲突的陈述时，极易产生偏好混乱。

为了保证这类高置信度属性的准确性，系统需要采用**结构化用户画像（User Profile）**。

![结构化用户画像的字段提取、状态合并与精准覆写机制](/images/content-rebuild/d0b7ce518736e1b6/fig-4.png)

上图展示了非结构化对话向结构化实体画像的演化过程。讲解员手持更新表单，展示了提取出的具体字段键值，并直接通过数据库主键更新执行精确覆写。

### 用户画像的存储与更新特性

用户画像通常存储在关系型数据库（如 PostgreSQL）、键值库（如 Redis）或带有 Schema 约束的 JSON 文档中：

* **结构化属性（Key-Value 形式）**：明确记录特定字段，如 `preferred_language: Go`、`output_style: compact`。
* **确定性读写**：通过 `user_id` 进行主键查询，单次读取成本低且结果确定，不受语义漂移影响。
* **原子覆写（Upsert）**：当检测到偏好发生迁移时，直接更新对应字段的值，从根本上避免了新旧偏好在提示词中共存引发的冲突。

### 三类记忆对比

| 维度 | 短期记忆（Working Memory） | 长期情景记忆（Long-Term Memory） | 用户画像（User Profile） |
| :--- | :--- | :--- | :--- |
| **物理存储介质** | 宿主应用会话状态，单次推理时填入上下文 | 外部向量库 / 倒排索引库 / 文档库 | 关系型数据库 / Key-Value 缓存 / JSON 结构 |
| **检索匹配方式** | 模型全序列自注意力直接计算 | 向量相似度模糊匹配 / 混合召回 | 依据用户 ID 或键名直接索引定位 |
| **生命周期** | 当前单次任务或单会话执行期间 | 跨会话长期持久化 | 跨会话长期持久化，通过显式规则更新 |
| **写入确定性** | 确定性追加至当前输入提示词 | 取决于切分、Embedding 质量与抽取逻辑 | 确定性字段级别写入与更新 |
| **更新与冲突解决**| 随窗口滑动截断，无法直接修改已输入内容 | 需标记失效、删除或更新向量索引 | 直接覆写目标字段，状态单一清晰 |
| **常见失效模式** | 超过上下文窗口被截断、早期指令被稀释 | 相似度假阳性、语义漏检、召回超出预算被截断 | 偏好提取规则未命中、字段 Schema 不匹配 |

![三类记忆形态在介质、时延、确定性与容量上的多维对比矩阵](/images/content-rebuild/d0b7ce518736e1b6/fig-5.png)

上图展示了三类记忆在存储物理介质、读写延迟、确定性以及存储容量维度的权衡分布。讲解员对比了从临时会话上下文到外部持久化数据库的技术边界，为架构选型提供了判别依据。

---

## 三类记忆在单次执行循环中的拼接与预算控制

在真实的 Agent 执行循环（如基于 [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) 构建的工作流）中，每一轮推理前都需要把这三类信息汇总成最终的输入提示词。

因为输入窗口存在上限，组装逻辑本质上是一个按优先级分配的 **Token 预算调度过程**。

### 预算分配与截断顺序

假设总输入预算为 4096 Token，预留给最终输出的额度之外，各部分的组装优先级通常如下：

1. **系统预设（刚性开销）**：核心行为约束与工具定义，属于最高优先级底座，不可轻易裁剪。
2. **用户画像（确定性注入）**：结构化键值字符串，体积小、信息密度高，以确定性格式直接拼入系统提示词中。
3. **长期记忆召回（配额约束与弹性填充）**：为历史经历设定预算上限（如扣除系统基线后剩余预算的固定比例），按检索相似度由高到低压入，超出配额即截断丢弃低分条目。
4. **短期即时会话（滑动窗口倒序装配）**：将其余全部可用预算分配给当前会话，由近及远倒序装入，优先保证最新的用户指令与工具返回结果完整可见，超出预算时淘汰较早历史。

![单次 Agent 执行循环中 Token 预算的动态划分与装配流水线](/images/content-rebuild/d0b7ce518736e1b6/fig-6.png)

上图展示了单次推理前提示词组装阶段的 Token 预算刻度尺。讲解员用刻度尺演示了系统指令、用户画像、工作上下文以及按需召回的长期记忆片段是如何在配额限制下组装的。

### 组装逻辑参考实现

以下 Python 示例展示了上述优先级预算分配机制。代码不引入外部黑盒库，直观呈现截断和组装逻辑：

```python
from typing import List, Dict, Any

def estimate_tokens(text: str) -> int:
    # 生产环境中应调用对应模型的分词器（如 tiktoken）计算精确 Token 数
    # 此处按英文字符和中文字符的粗略折算比例进行演示估算
    return max(1, len(text) // 3)

class AgentContextAssembler:
    def __init__(self, max_input_budget: int = 4096):
        self.max_budget = max_input_budget

    def assemble_prompt(
        self,
        system_instruction: str,
        user_profile: Dict[str, Any],
        retrieved_long_term_memories: List[str],
        working_messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        current_tokens = 0
        
        # 1. 组装系统基线与结构化用户画像
        profile_str = "\n".join([f"- {k}: {v}" for k, v in user_profile.items()])
        combined_system = (
            f"{system_instruction}\n\n"
            f"### 用户偏好与静态背景:\n{profile_str}"
        )
        current_tokens += estimate_tokens(combined_system)
        
        if current_tokens > self.max_budget:
            raise ValueError("系统核心指令与画像配置已超出总输入预算，无法组装。")

        # 2. 规划长期记忆可用的最大空间（例如剩余预算的 35%）
        remaining_budget = self.max_budget - current_tokens
        max_memory_budget = int(remaining_budget * 0.35)
        
        admitted_memories = []
        memory_tokens = 0
        for mem in retrieved_long_term_memories:
            cost = estimate_tokens(mem)
            if memory_tokens + cost <= max_memory_budget:
                admitted_memories.append(mem)
                memory_tokens += cost
            else:
                # 超过长期记忆预留上限，截断丢弃得分较低的历史记录
                break
                
        if admitted_memories:
            memory_block = "\n### 检索到的历史相关经验:\n" + "\n".join(
                [f"* {m}" for m in admitted_memories]
            )
            combined_system += memory_block
            current_tokens += estimate_tokens(memory_block)

        # 3. 倒序装配短期即时会话，优先保留最新的上下文
        admitted_messages = []
        available_for_messages = self.max_budget - current_tokens
        
        for msg in reversed(working_messages):
            # 估算内容及角色元数据开销
            cost = estimate_tokens(msg["content"]) + 4
            if available_for_messages - cost >= 0:
                admitted_messages.append(msg)
                available_for_messages -= cost
            else:
                # 剩余空间耗尽，丢弃过早的会话轮次
                break
                
        final_messages = [{"role": "system", "content": combined_system}]
        final_messages.extend(reversed(admitted_messages))
        return final_messages

# 验证用例（数据仅用于说明机制）
assembler = AgentContextAssembler(max_input_budget=1200)
sys_prompt = "你是一个严谨的代码审查助手，请基于规范进行审查。"
profile = {"preferred_language": "Go", "code_style": "Standard Library Style"}
retrieved_episodes = [
    "2025-02-10 记录：本项目接口强制配置 5 秒超时。",
    "2025-01-15 记录：数据库连接池已从 ORM 迁至底层轻量驱动。"
]
working_chat = [
    {"role": "user", "content": "请帮我检查下面这段函数的并发性能。"},
    {"role": "assistant", "content": "请提供具体的函数实现代码。"},
    {"role": "user", "content": "func FetchData() { /* 逻辑省略 */ }"}
]

final_prompt_payload = assembler.assemble_prompt(
    system_instruction=sys_prompt,
    user_profile=profile,
    retrieved_long_term_memories=retrieved_episodes,
    working_messages=working_chat
)

print(f"组装后消息列表长度: {len(final_prompt_payload)}")
print(f"系统段落最终内容:\n{final_prompt_payload[0]['content']}")
```

在这段组装逻辑中：
* 用户偏好通过结构化映射确定性写入系统段落；
* 长期记忆经过配额判断后弹性注入，超限时主动截断；
* 短期即时会话按“由近及远”倒序装入，确保最新的提问与工具反馈最优先获得注意力。

关于记忆长期累积带来的隐私泄露与噪声污染问题，请参阅 [为什么记忆系统不能什么都存？隐私、噪声和上下文污染如何控制？](/frontier/agent-memory-system/memory-risks.md)；有关写入、检索和更新淘汰的具体工程实现，请阅读 [如何设计一个可用的 Agent Memory？写入、检索、更新和遗忘机制怎么做？](/frontier/agent-memory-system/memory-design.md)；系统完整知识图谱可返回 [Agent 记忆系统导航](/frontier/agent-memory-system/README.md)。
