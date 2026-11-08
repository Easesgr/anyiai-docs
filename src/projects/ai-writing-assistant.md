---
title: 如何用大模型写长文？分层规划、章节扩写与状态一致性实战
icon: mdi:pen
description: 拆解长文写作的多阶段流水线：分层大纲拆解、Token预算分配、上下文切片扩写、后置规则审查与断点恢复实战。
---

在大模型应用开发中，生成几百字的邮件或摘要相对容易，但让模型一次性产出数千字甚至上万字、逻辑严密、术语统一的长篇技术报告、行业白皮书或深度教程时，直接调用单次提示词往往会失效。

长文写作不能依赖单次超长文本的自回归采样，而需要拆分为多阶段、有状态的流水线工程。本文以撰写一份约 5000 字（折合 6500 Token）的《分布式高并发存储架构实践》技术文档为贯穿案例，拆解分层规划、篇幅预算控制、上下文切片装配、后置规则审查与人工介入机制，并提供包含状态管理与断点恢复的最小可运行实现。

![长文写作助手的多阶段生成架构看板](/images/content-rebuild/4b39cd9c371e11fc/cover.png)

---

## 单次生成长文的常见失效原因

许多开发者尝试直接向模型下达类似“请写一篇 8000 字关于分布式系统架构的深度文章”的指令，实际输出通常会出现以下问题：

1. **单次解码长度硬性截断**：即使模型的上下文输入窗口（Context Window）达到 128k 甚至更高，其单次响应的最大输出限制（Max Output Tokens）通常固定在 4096 或 8192 个 Token。这意味着无论提示词如何强调“不要简写”，单次 API 调用在接口限制上都无法一次性吐出完整长文。
2. **缺乏全局进度感导致头重脚轻**：大语言模型基于前序 Token 预测下一个 Token，缺乏全局输出进度的内部计数器。在没有外部步进控制的情况下，模型往往在引言和背景定义部分消耗了过半的篇幅，随后察觉到生成即将结束，便在核心推导和架构设计部分草草收尾。
3. **历史原文累积引发注意力分散与计费膨胀**：如果在多轮扩写中把之前写完的全部章节原文原样拼入下一次请求的上下文，不仅每轮累计的输入 Token 总量呈二次方增长（全篇总输入 Token 约为 $\sum_{k=1}^{N} k \cdot L_{chunk} \approx O(N^2)$），而且引入过多与当前章节无关的历史细节容易分散模型对当前指令的关注，导致其对当前章节子任务的遵循度下降，频繁出现前后术语矛盾或论点重复。

解决这些问题的核心是**分层解耦**：将大纲规划、篇幅配额、上下文组装与章节撰写拆分为独立步骤，由外部宿主程序维护状态流转。

---

## 分层大纲拆解与篇幅预算分配

分层规划的目标是将宏观主题层层分解为模型单次调用能够高质量完成的微观任务单元：

```
[用户输入需求与总篇幅]
       │
       ▼
[第一层：全局总纲规划] ──> 确定核心论点、总章节划分 (Chapter 1 ~ N)
       │
       ▼
[第二层：篇幅预算分配] ──> 为每个一级章节分配 Token 配额与浮动区间
       │
       ▼
[第三层：二级子大纲拆解] ──> 生成当前章节的 3~5 个具体要点与约束
```

![分层大纲规划与篇幅预算分配流程](/images/content-rebuild/4b39cd9c371e11fc/fig-1.png)

### 篇幅预算分配算例

大模型无法准确感知自身输出的 Token 精确数值，因此提示词中的字数要求只能作为软约束。工程上必须由宿主程序计算好配额，并在提示词中给出明确的段落数与区间参考。

以目标总篇幅 6500 Token 的《分布式高并发存储架构实践》为例，各章节权重与预算分配如下：

| 章节层级 | 预设权重 | 目标 Token 预算 | 允许浮动容差 (±15%) | 核心约束 |
| :--- | :--- | :--- | :--- | :--- |
| 第 1 章：背景与并发写入瓶颈 | 15% | 975 | 828 ~ 1121 | 聚焦问题场景与单机锁瓶颈，不提前展开分布式解法 |
| 第 2 章：核心数据结构与分片原理 | 30% | 1950 | 1657 ~ 2242 | 详细推导机制，包含至少 2 个数学/算例对比 |
| 第 3 章：工程落地与容灾架构 | 35% | 2275 | 1933 ~ 2616 | 展开组件交互、网络分区恢复与异常处理分支 |
| 第 4 章：演进局限与方案总结 | 20% | 1300 | 1105 ~ 1495 | 归纳当前设计边界，对比其他替代选型 |

在生成具体章节的 Prompt 时，将该 Token 区间与段落数量要求显式传入（例如：`"本章建议生成 1700-2200 Token，分为 4 个自然段展开"`），并在生成后由程序做长度检验。

<ReadMoreLock />

---

## 状态字典设计：大纲、术语表与滚动摘要

为保证整篇文档在分段生成过程中保持概念一致，宿主程序必须在模型外部维护一个**状态字典（State Dictionary）**。它记录大纲结构、已生成的局部摘要和统一术语映射表。

![状态字典数据结构与组件流动示意](/images/content-rebuild/4b39cd9c371e11fc/fig-2.png)

以下是状态字典的核心字段定义：

```python
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class SectionNode(BaseModel):
    section_id: str
    title: str
    token_budget: int
    detailed_points: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending | drafting | reviewed | completed
    content: Optional[str] = None
    local_summary: Optional[str] = None

class WritingState(BaseModel):
    topic: str
    total_budget: int
    style_guide: Dict[str, str] = Field(
        default_factory=lambda: {
            "tone": "严谨客观的技术白皮书风格",
            "forbidden_words": "避免使用'众所周知'、'显而易见'、'颠覆性'等无实际信息量的修饰词",
            "code_standard": "代码统一使用 Python 3.10+ 类型标注"
        }
    )
    glossary: Dict[str, str] = Field(default_factory=dict)  # 统一术语规范映射
    outline: List[SectionNode] = Field(default_factory=list)
    current_index: int = 0
    rolling_summaries: List[str] = Field(default_factory=list)  # 前序章节摘要缓存
```

与偏向表格结构化字段生成的 [AI 视频脚本生成器](/projects/ai-video-script.md) 不同，长文写作系统需要维持自由叙述中的概念连续性。状态字典中的 `glossary` 可以避免第 1 章称呼“协调节点”，而在第 3 章无故切换为“调度中心”；`rolling_summaries` 则为后续章节提供因果线索，而无需输入大量历史原文。

---

## 步进扩写与上下文切片装配

在撰写后续章节（如第 3 章）时，不应将前两章数千字的完整文本输入给模型，而应使用**上下文切片（Context Slicing）**装配提示词：

* **切片 A（全局骨架）**：让模型了解全书结构及当前章节所处位置；
* **切片 B（统一术语与风格指引）**：规定专业名词与文风红线；
* **切片 C（前序摘要历史）**：提供前序章节的滚动摘要（例如紧邻前 1~2 章节的要点），在控制上下文长度的同时提供承接线索；
* **切片 D（本章子任务与篇幅）**：明确本章要展开的要点和 Token 预算。

![分段扩写提示词上下文切片装配机制](/images/content-rebuild/4b39cd9c371e11fc/fig-3.png)

### 提示词组装与摘要提取函数

```python
def build_section_prompt(state: WritingState, current_node: SectionNode) -> str:
    # 1. 组装滚动摘要：保留紧邻前序的最近 2 章节摘要作为背景
    if state.rolling_summaries:
        history_context = "\n".join([f"- {s}" for s in state.rolling_summaries[-2:]])
    else:
        history_context = "无（当前为起始章节）"
    
    # 2. 格式化术语表
    terms_context = ", ".join([f"{k} -> {v}" for k, v in state.glossary.items()])

    # 3. 构造提示词正文
    points_str = "\n".join([f"* {pt}" for pt in current_node.detailed_points])
    
    prompt = f"""# 任务：撰写章节《{current_node.title}》

## 写作风格与术语规范
- 文风基调：{state.style_guide['tone']}
- 禁用表达：{state.style_guide['forbidden_words']}
- 规范术语：{terms_context}

## 前文背景脉络
{history_context}

## 本章编写要求
- 目标篇幅：约 {current_node.token_budget} Token，符合预设章节详略度
- 核心必须论述的要点：
{points_str}

请直接输出本章 Markdown 正文，不要重复输出一级/二级标题，也不要输出开场寒暄。"""
    return prompt
```

关于输入提示词各角色的边界区分，可参考 [System Prompt、User Prompt、Assistant Prompt 分别有什么作用？](/llm-100/prompt-engineering/System_Prompt_User_Prompt_Assistant_Prompt_分别有什么作用.md)。每当一个章节草稿生成完毕，系统立即调用一次精简提示词，将草稿浓缩为 100~150 字的客观事实摘要，存入 `rolling_summaries`，供下一章节使用。

---

## 后置审查节点与回退重试机制

草稿生成后，不直接标记完成，而是流转至审查节点。审查节点负责对文本做确定性规则检验和指令遵从检查。

![后置审查节点校验流与回退分支](/images/content-rebuild/4b39cd9c371e11fc/fig-4.png)

### 审查的核心维度

1. **术语偏离检验**：扫描正文是否出现了 `glossary` 约定之外的同义替代词或口语化命名；
2. **篇幅偏离检验**：统计实际生成文本的 Token 或中文字符数，判断是否脱离了预设预算的容差区间（例如低于 70% 或高于 130%）；
3. **违禁词检测**：检查是否包含提示词明确禁用的空洞修饰词。

### 结构化审查结果格式

```json
{
  "passed": false,
  "word_count_valid": true,
  "terminology_violations": [
    {
      "found": "神经网络系数",
      "expected": "模型权重",
      "line_snippet": "在更新神经网络系数时需要注意梯度爆炸..."
    }
  ],
  "forbidden_words_found": ["众所周知"],
  "revision_instruction": "请将文中出现的'神经网络系数'统一替换为'模型权重'；删去第2段中的'众所周知'。"
}
```

当 `passed` 为 `false` 时，系统将草稿连同 `revision_instruction` 传回模型进行定向修改。为防止无限循环，系统需设置最大重试阈值（通常为 2 次）。若重试仍未通过，则标记警告并转入人工介入队列。

---

## 人工介入（Human-in-the-Loop）与断点恢复

在长文写作这类高复杂度任务中，全自动黑盒运行容易因局部偏差导致后续章节逐步走样。通过断点（Breakpoint）引入人机协同是提升输出可用度的重要手段，这在 [LangGraph 状态图框架原理](https://docs.langchain.com/oss/python/langgraph/overview) 与 [Anthropic 构建高效智能体指南](https://www.anthropic.com/engineering/building-effective-agents) 中均是推荐做法。

![人机协同交互节点与状态编辑切入点](/images/content-rebuild/4b39cd9c371e11fc/fig-5.png)

系统可在两个关键节点设置人工检查门禁：

1. **大纲确认门禁（Outline Gate）**：总纲和二级子大纲生成后，流水线暂停并将状态写入持久化存储。用户可以手动调整章节顺序、删减要点或修改预算。用户确认后，系统才开始执行章节扩写。
2. **章节草稿门禁（Section Gate）**：当某一章生成并初步通过规则审查后，用户可直接对草稿做在线润色。**注意**：一旦人工编辑了章节正文，宿主程序必须基于人工修改后的最终文本重新生成该章节摘要，并更新到 `rolling_summaries` 中，确保后续章节的输入与用户的修改对齐。

---

## 最小可运行装配流水线代码实现

以下是一份完整的可运行代码，包含状态字典维护、分章节调用、局部摘要提取与最终 Markdown 目录及正文的自动装配：

```python
from typing import Callable
from pydantic import BaseModel, Field

class SectionNode(BaseModel):
    section_id: str
    title: str
    token_budget: int
    detailed_points: list[str] = Field(default_factory=list)
    status: str = "pending"
    content: str | None = None
    local_summary: str | None = None

class WritingState(BaseModel):
    topic: str
    total_budget: int
    style_guide: dict[str, str] = Field(default_factory=dict)
    glossary: dict[str, str] = Field(default_factory=dict)
    outline: list[SectionNode] = Field(default_factory=list)
    current_index: int = 0
    rolling_summaries: list[str] = Field(default_factory=list)

def build_section_prompt(state: WritingState, current_node: SectionNode) -> str:
    history_context = "\n".join([f"- {s}" for s in state.rolling_summaries[-2:]]) if state.rolling_summaries else "无（起始章节）"
    terms_context = ", ".join([f"{k}: {v}" for k, v in state.glossary.items()])
    points_str = "\n".join([f"* {p}" for p in current_node.detailed_points])
    
    return f"""任务：撰写章节《{current_node.title}》
规范：{terms_context}；风格基调：{state.style_guide.get('tone', '客观严谨')}
前文摘要：{history_context}
本章要点：
{points_str}
目标预算：约 {current_node.token_budget} Token。请直接输出正文。"""

class WritingEngine:
    def __init__(self, state: WritingState, llm_fn: Callable[[str], str]):
        self.state = state
        self.llm = llm_fn

    def run(self) -> str:
        print(f"[*] 开始撰写任务: {self.state.topic}")
        
        for idx, node in enumerate(self.state.outline):
            self.state.current_index = idx
            print(f"\n[+] 正在撰写第 {idx+1}/{len(self.state.outline)} 章: 《{node.title}》")
            
            # 1. 组装切片并生成草稿
            prompt = build_section_prompt(self.state, node)
            draft = self.llm(prompt)
            node.content = draft
            
            # 2. 生成局部滚动摘要
            summary_prompt = f"用不超过80字提取以下段落的核心结论，不带修饰词：\n{draft}"
            summary = self.llm(summary_prompt)
            node.local_summary = summary
            self.state.rolling_summaries.append(f"《{node.title}》: {summary}")
            
            node.status = "completed"
            print(f"[✓] 完成《{node.title}》，已记录摘要: {summary}")
            
        return self.assemble_markdown()

    def assemble_markdown(self) -> str:
        doc = [f"# {self.state.topic}\n", "## 目录\n"]
        for node in self.state.outline:
            anchor = node.title.lower().replace(" ", "-")
            doc.append(f"- [{node.title}](#{anchor})")
        doc.append("\n---\n")
        
        for node in self.state.outline:
            doc.append(f"## {node.title}\n")
            doc.append(node.content or "*(内容缺失)*")
            doc.append("\n")
            
        return "\n".join(doc)

# --- 模拟执行测试 ---
def mock_llm(prompt: str) -> str:
    if "提取以下段落" in prompt:
        return "阐述了并发写入锁竞争瓶颈与一致性哈希分片选型策略。"
    return "本节探讨了分布式并发写入场景下的数据分片实践。通过一致性哈希将键空间打散，并结合版本向量解决冲突。"

initial_state = WritingState(
    topic="分布式高并发存储架构实践",
    total_budget=6500,
    style_guide={"tone": "客观中立"},
    glossary={"版本向量": "Version Vector", "一致性哈希": "Consistent Hashing"},
    outline=[
        SectionNode(section_id="s1", title="并发写入的瓶颈剖析", token_budget=975, detailed_points=["锁竞争开销", "单机瓶颈"]),
        SectionNode(section_id="s2", title="数据分片与路由机制", token_budget=1950, detailed_points=["一致性哈希原理", "虚拟节点平衡"])
    ]
)

engine = WritingEngine(initial_state, mock_llm)
markdown_output = engine.run()
print("\n===== 组装完成 Markdown 片段 =====\n")
print(markdown_output[:260] + "...")
```

![长文档拼装与跨章节引用校验流水线](/images/content-rebuild/4b39cd9c371e11fc/fig-6.png)

---

## 验收标准与测试设计

长文写作流水线在交付运行前，可通过以下测试用例进行验证：

1. **预算偏离率测试**：运行 5 组长文生成任务，统计各章节实际 Token 数与分配预算的相对误差率 $\delta = \frac{|实际 - 预算|}{预算}$。验证均值是否控制在预设容差阈值内（例如 $\delta \le 20\%$）。
2. **术语一致性检验**：在 `glossary` 中预置 5 对规范术语及其常见错用同义词，对最终输出 Markdown 做字符串匹配扫描，统计规范术语覆盖率和违规混用词检出率。
3. **断点持久化与恢复测试**：在生成到第 2 章节完成后手动终止进程，重新加载持久化的 `WritingState` JSON 文件，检查程序是否能准确读取前序 `rolling_summaries` 并无缝从第 3 章节继续执行，避免从头重新生成已完成章节。

关于多步骤智能体状态机设计的深入讨论，可参考 [AI 自动化工作流](/projects/ai-workflow-automation.md) 与 [AI 编程助手](/projects/ai-coding-assistant.md)；若需在更多业务场景中整合此类工程，请参阅 [AI 项目实战总览](/projects/README.md)。
