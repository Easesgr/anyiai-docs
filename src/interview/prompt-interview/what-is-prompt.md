---
title: Prompt 工程是什么？如何系统优化与评测大模型提示词？
icon: mdi:message-text-outline
tag:
- Prompt面试
- Prompt工程
- 基础概念
description: 从自回归概率预测机制讲清 Prompt 工程的原理与边界，结合工单分类案例，建立基于测试集的量化评测与工程优化流程。
---

![Prompt 工程技术定位与评估闭环封面图](/images/content-rebuild/86ebc10b8a051af0/cover.png)

在很多初学者的理解中，Prompt 工程（提示词工程）常被当成“如何换更客气的措辞让大模型听话”的话术技巧。但在实际业务开发与技术面试中，面试官真正考察的是：你是否清楚大模型基于上下文做概率预测的工作机制、Prompt 能做到什么与做不到什么，以及如何摆脱“人工盲测调词”，建立自动化测试与回归评测流程。

本文从面试口述答案出发，结合工单分类的贯穿例子，拆解 Prompt 的工作机制、评测驱动迭代方法、性能与延迟权衡，以及常见的工程误区。

---

## 1. 面试口述速答：什么是 Prompt 工程？

在技术面试中遇到“什么是 Prompt 工程？为什么调整提示词能改变模型输出？”时，可以这样回答：

> **面试口述参考**：  
> “自回归大语言模型的核心机制，是根据已输入的上下文序列，预测下一个 Token 的条件概率分布 $P(x_t \mid x_1, x_2, \dots, x_{t-1})$。
> 
> Prompt 工程不是单纯的话术修饰，而是**通过在上下文窗口中组织结构化输入（包括角色设定、任务约束、少样本示例和输出格式定义），引导模型的自注意力计算，使其在词表上的生成概率偏向业务预期的输出路径**。
> 
> 调整提示词之所以能改变输出，是因为 Transformer 在处理输入时，不同词元的排列和语义引导会改变各个键值向量之间的注意力权重，从而压制偏离任务的输出，提升符合结构规范和推理要求的 Token 生成概率。在工程落地中，Prompt 优化的核心目标是在不修改模型参数的前提下，以较低的上下文开销和可复现的评测指标，获得稳定可靠的输出。”

![Prompt 对自注意力计算与词表条件概率分布的引导机制](/images/content-rebuild/86ebc10b8a051af0/fig-1.png)

如图 1 所示，自回归生成过程中，未加规范的输入可能让模型在多种可能的回答风格之间游离；而经过明确任务定义和格式约束的 Prompt，能在计算注意力时强化与目标逻辑相关的上下文线索，使得期望输出的 Token 序列在概率分布中更占优势。

---

## 2. Prompt 的能力边界：行为引导不等于知识注入

评估一个技术方案时，明确它做不到什么往往比宣称它能做什么更重要。

### 2.1 引导已有能力 vs 注入未知事实

Prompt 的主要作用是**引导模型已有的语言理解、逻辑推导和模式遵循能力**，而不是向模型持久化注入原本完全不存在的事实知识。

* **适合使用 Prompt 解决的问题**：模型本身具备相关通用知识，但容易误解指令细节、输出格式不固定（例如需要纯 JSON 却返回了闲聊）、推理过程跳步，或者需要遵循特定的角色风格与安全策略。
* **单靠 Prompt 无法解决的问题**：模型训练切断日期之后的最新事实、企业内部未公开的私有业务接口定义、特定客户的订单数据。如果上下文没有提供这些信息，无论提示词写得多么详尽，模型也只能依靠已有参数进行统计联想，极易产生虚假陈述（幻觉）。

当面对动态知识或参数缺失时，必须引入检索增强（RAG，详见 [/interview/prompt-interview/rag-prompt.md](/interview/prompt-interview/rag-prompt.md)）或针对私有语料的模型微调（微调对比详见 [/interview/prompt-interview/prompt-vs-finetuning.md](/interview/prompt-interview/prompt-vs-finetuning.md)）。

<ReadMoreLock />

![模型参数能力上限与 Prompt 引导空间的包含关系对比](/images/content-rebuild/86ebc10b8a051af0/fig-2.png)

如图 2 所示，基座模型能够理解和推断的内容，受限于预训练和对齐阶段积累的参数表示。Prompt 工程是在这个能力范围之内，寻找能让模型输出贴合具体任务的输入形式。对于参数范围之外的专有业务数据，必须依靠外部知识检索来补全上下文。

---

## 3. 评测驱动优化：建立基于测试集的迭代流程

根据 Anthropic 官方提示工程指南的实践建议，**在着手调优提示词之前，必须先明确成功标准并准备经验测试集**。如果没有可复现的测试集，单纯在网页对话框中改一句看一次，很容易出现“修好了用例 A，却搞崩了用例 B”的隐蔽回归。

### 3.1 评测驱动的三步流程

1. **定义客观的成功标准（Success Criteria）**：  
   避免“回答更专业”“表达更通顺”等难以量化的话术，转化为确定性的程序判断指标。例如：JSON 必须可被 `json.loads` 正确反序列化、关键字段覆盖率达到 100%、意图分类准确率达到目标阈值。
2. **构建经验测试用例集（Empirical Test Set）**：  
   准备 30 到 100 条覆盖不同场景的真实输入：  
   * 常见标准输入（正常业务工单）；
   * 带有噪音的长尾输入（拼写错误、口语化简称、多意图混杂）；
   * 恶意诱导用例（测试是否容易被用户指令篡改行为，参见 [/interview/prompt-interview/prompt-injection.md](/interview/prompt-interview/prompt-injection.md)）；
   * 边界反事实用例（信息不足时验证模型是否会主动反问或说明无法处理，参见 [/interview/prompt-interview/reduce-hallucination.md](/interview/prompt-interview/reduce-hallucination.md)）。
3. **固定基准版本与控制变量迭代**：  
   保留一份初始基线（Baseline）提示词。每次修改只调整一个变量（例如仅增加一组输出格式规范，或增加一条多步骤推理要求，参见 [/interview/prompt-interview/chain-of-thought.md](/interview/prompt-interview/chain-of-thought.md)），在全量测试用例集上批量运行，对比准确率和格式达标率的变化。

![基于 Anthropic 最佳实践的评测驱动 Prompt 优化闭环](/images/content-rebuild/86ebc10b8a051af0/fig-3.png)

如图 3 所示，把 Prompt 视作生产代码的一部分，改动提示词后执行测试用例集，统计分类准确率与格式解析成功率，只有在新版本综合指标确认提升且无关键回归后才发布。

### 3.2 示例：客服工单意图识别评测代码与对比

我们以一个客服意图分流任务为例：输入用户文本，要求模型输出严格 JSON，分类为 `BILLING`（账单计费）、`TECHNICAL`（技术故障）或 `COMPLAINT`（服务投诉）三者之一。

```python
# 评估测试集与判定逻辑示例（Python 3.10+）
import json

test_cases = [
    {"input": "我的账单为什么多扣了5块钱？", "expected": "BILLING"},
    {"input": "服务器502报错，网页打不开", "expected": "TECHNICAL"},
    {"input": "你们客服态度太差了，我要投诉工号012", "expected": "COMPLAINT"},
    {"input": "我付了钱但是功能没开通，你们是不是骗子？！", "expected": "COMPLAINT"},  # 复合意图：包含扣费，但核心情绪偏向投诉
    {"input": "API 请求一直报 401 鉴权失败，怎么查？", "expected": "TECHNICAL"},
    {"input": "发票开错了，税号少了一位", "expected": "BILLING"},
    {"input": "昨天续费成功了，但今天看控制台状态还是欠费停止运行", "expected": "TECHNICAL"},  # 复合意图：涉及续费与运行状态阻断
    {"input": "垃圾软件，天天崩，能不能退钱？", "expected": "COMPLAINT"},
    {"input": "可以对公转账吗？需要开具增值税专票", "expected": "BILLING"},
    {"input": "怎么把 Python 3.9 升级到 3.11？", "expected": "TECHNICAL"}
]

def evaluate_prompt_version(predict_fn, dataset):
    """
    测试给定预测函数在数据集上的格式解析率与分类准确率
    """
    format_errors = 0
    correct_predictions = 0
    
    for case in dataset:
        raw_output = predict_fn(case["input"])
        try:
            result = json.loads(raw_output)
            pred_intent = result.get("intent")
            if pred_intent == case["expected"]:
                correct_predictions += 1
        except (json.JSONDecodeError, TypeError, AttributeError):
            format_errors += 1
            
    total = len(dataset)
    accuracy = correct_predictions / total if total > 0 else 0.0
    format_pass_rate = (total - format_errors) / total if total > 0 else 0.0
    return {"accuracy": accuracy, "format_pass_rate": format_pass_rate}
```

下表展示了在该 10 条示例测试集上，三个迭代版本的教学测试表现示意对比（采样温度设为 0 以尽量降低生成随机性）：

| 版本编号 | 提示词改动说明 | 格式合规率 | 分类准确率 | 典型失败分析 |
| :--- | :--- | :--- | :--- | :--- |
| **v1.0 (基线)** | 一句简略指令：“请判断工单意图，输出 JSON” | $70.0\%$ | $50.0\%$ | 模型常夹带“好的，这是为您提取的 JSON”等前导文本，导致解析崩溃；复合意图难以决断 |
| **v2.0** | 增加 JSON Schema 示例并要求只输出 JSON 本身 | $100.0\%$ | $70.0\%$ | 格式全部解析通过，但在“付了钱没开通，骂人骗子”样例上归入了 `BILLING`，抓错主要矛盾 |
| **v3.0** | 补充意图优先级说明（COMPLAINT > TECHNICAL > BILLING）并提供 2 个针对性示例 | $100.0\%$ | $90.0\%$ | 复合意图识别正确，仅在续费耦合故障的边界样例上仍存在归类分歧 |

通过测试集数据可以清楚看到：从 v1.0 到 v2.0，我们解决的是**格式与接口稳定性**问题；从 v2.0 到 v3.0，我们通过**定义冲突优先级和提供样本**解决的是**业务逻辑歧义**问题。具体的提示词结构设计方法可参考 [/interview/prompt-interview/prompt-design.md](/interview/prompt-interview/prompt-design.md)。

---

## 4. 性能与成本权衡：Prefill、Decode 与模型选型

在很多高并发业务场景中，当接口延迟过高时，开发者容易盲目尝试“把 Prompt 中的形容词删掉几个字来降低耗时”。这种做法往往收效甚微。

### 4.1 推理延迟的构成差异

大语言模型的推理过程分为两个计算阶段：
1. **预填充阶段（Prefill）**：模型并行处理输入的 Prompt Tokens，计算自注意力矩阵并生成初始的 KV 缓存。由于输入词元是并行计算的，现代 GPU 吞吐量极高，提示词多出几十个 Token 带来的耗时差异通常只有几毫秒到几十毫秒。
2. **解码阶段（Decode）**：模型根据上一个生成的 Token 逐个自回归生成下一个 Token。由于必须串行前向传播，这一阶段深受显存带宽限制。如果模型输出了 500 个 Token，通常需要数百毫秒甚至数秒的固定时间。

在常规中短输入且输出具有一定长度的场景下，端到端延迟的大头通常在 Decode 阶段；而在超长上下文输入时，Prefill 耗时与首字延迟（TTFT）同样不可忽视。如果一个任务只需返回简短的状态码，却任由模型生成大段解释性文字，Decode 阶段的延迟就会大幅增加。

![模型推理延迟构成分解与模型降级路由策略对比](/images/content-rebuild/86ebc10b8a051af0/fig-4.png)

如图 4 所示，面对响应耗时与 Token 成本过高的问题，更有效的工程策略包括：
* **模型降级与路由（Model Routing）**：对于工单分类、文本抽取等相对结构化的确定性任务，优先选用轻量模型（如 Claude 3.5 Haiku 或 GPT-4o-mini）；复杂多步推理才路由给旗舰模型。
* **控制输出长度（Max Output Tokens）**：在 Prompt 中明确要求直接返回紧凑字段，避免闲聊与自我解释，从根本上缩短串行解码时间。
* **利用服务端缓存（Prompt Caching）**：把长篇规范或不变的系统背景放在 System Prompt 前半部分，利用推理服务商提供的 KV 缓存机制降低 Prefill 计费与首字延迟。

---

## 5. 常见工程陷阱：把系统与数据问题甩给 Prompt

在构建 RAG 或智能体（Agent）系统时，常见的一个反面模式是：**上游数据解析和检索质量不过关，试图靠加长 Prompt“大力出奇迹”**。

### 5.1 上游脏数据导致的伪 Prompt 问题

典型的故障链路如下：
1. 上游解析 PDF 或表格时排版错乱、字段缺失（文档解析相关可见 [/frontier/document-parsing/what-is-file-parsing.md](/frontier/document-parsing/what-is-file-parsing.md)）；
2. 向量检索召回了与用户问题不相关的噪声切片，直接拼接进 Prompt 发给模型；
3. 模型因上下文矛盾或关键信息缺失，输出胡编乱造的答案；
4. 开发者认为是提示词写得不够严谨，于是在 System Prompt 中不断追加矛盾而冗长的规则：“请仔细查找、不要忽略任何线索、即使表格格式乱了也要猜出来”。

![上游数据源污染导致下游模型推理故障的因果链路分析](/images/content-rebuild/86ebc10b8a051af0/fig-5.png)

如图 5 所示，这种修改不仅无法保证推理正确性，还会因为塞入过多冲突规则而稀释模型对核心指令的注意力。工程治理的基本原则是：**数据解析问题在预处理层解决，召回准确度在检索层解决，Prompt 负责规范输入前提下的意图对齐与格式控制**。对上下文全链路的精细治理，可参考 [/frontier/context-engineering/what-is-context-eng.md](/frontier/context-engineering/what-is-context-eng.md)。

---

## 6. 面试深度追问与常见认知误区

### 追问 1：大模型自带的推理能力越来越强，Prompt 工程是不是很快就不需要了？

**建议回答**：  
“不会消失，但**关注的重点发生了转移**。早期那种依赖巧合的‘修辞话术’（比如在结尾加一句‘请假装你是世界顶级专家’）确实逐渐被后训练成熟的基座模型能力所淡化。

但在生产环境中，企业对大模型的核心需求是**稳定交付**。现在的 Prompt 工程更侧重于系统层面的接口规范设计：
1. **结构化契约定义**：利用 XML 标签或 JSON Schema 明确输入与输出边界（参见 [/interview/prompt-interview/structured-output.md](/interview/prompt-interview/structured-output.md)）；
2. **安全防护边界**：区分系统指令与不受信任的用户外部输入，防御越狱与提示注入（参见 [/interview/prompt-interview/prompt-injection.md](/interview/prompt-interview/prompt-injection.md)）；
3. **与系统架构协作**：配合工作流编排、工具调用协议和外部状态管理，确保复杂调用链路可被观测、评测和回放（参见 [/frontier/ai-workflow/what-is-ai-workflow.md](/frontier/ai-workflow/what-is-ai-workflow.md) 与 [/interview/prompt-interview/tool-calling-spring-ai.md](/interview/prompt-interview/tool-calling-spring-ai.md)）。”

### 追问 2：为什么在 Prompt 里写了‘严禁输出 Markdown 标记’，模型有时候还是会输出 ```json？

**建议回答**：  
“这主要是预训练数据分布偏好与负向提示词特性的共同影响：
1. **预训练分布的强大惯性**：在开源代码和网络文本中，JSON 数据绝大多数都包裹在 Markdown 代码块（如 ` ```json `）中。模型在生成 JSON 时，自回归预测下一个词元为代码围栏具有很强的条件先验。
2. **负向指令缺乏直接的符号算子**：自回归模型是依据上下文中的词元做前向预测。当输入中包含‘Markdown’、‘代码块’等词汇时，这些词元会作为上下文被注意力机制计算，反而维持了相关概念的语义活跃度。

工程上的靠谱处理方式是：
* **改用正向指引**：把‘严禁包含 Markdown 格式’改为正向明确要求‘仅输出以 `{` 开头并以 `}` 结尾的纯文本字符’；
* **提供少样本示例**：在提示词中显式给出 1 到 2 个干净的无围栏示例（参见 [/interview/prompt-interview/shot-types.md](/interview/prompt-interview/shot-types.md) 与 [/interview/prompt-interview/few-shot-learning.md](/interview/prompt-interview/few-shot-learning.md)）；
* **代码兜底解析**：在业务系统的后处理环节加入字符串清洗逻辑（如用正则自动剥离外层代码围栏），用确定性的外部程序保证稳定性，不能指望单纯依靠文字提醒实现 100% 绝对合规。”

![常见 Prompt 调试误区与工程化防御手段雷达排查图](/images/content-rebuild/86ebc10b8a051af0/fig-6.png)

如图 6 所示，构建可靠的生产服务需要结合正向指令改写、少样本范例、上游输入治理以及下游代码解析验证，形成多层保障，而不是把所有可靠性寄托在一段文本上。

---

## 7. 知识体系导航与延伸

Prompt 工程是大模型应用开发的基础起点。想要建立完整的系统设计能力，建议按以下模块继续阅读：

* **指令语法与消息角色**：理解 System、User、Assistant 角色的权限差异（见 [/interview/prompt-interview/prompt-types.md](/interview/prompt-interview/prompt-types.md)）与提示词模块化搭建方法（见 [/interview/prompt-interview/prompt-structure.md](/interview/prompt-interview/prompt-structure.md)）。
* **少样本与推理链技术**：掌握 Few-shot 在复杂边界处理中的运用（见 [/interview/prompt-interview/few-shot-learning.md](/interview/prompt-interview/few-shot-learning.md)）以及思维链（CoT）在多步逻辑推导中的实现（见 [/interview/prompt-interview/chain-of-thought-2.md](/interview/prompt-interview/chain-of-thought-2.md)）。
* **工程容错与安全防御**：排查常见失效场景的治理策略（见 [/interview/prompt-interview/prompt-failure.md](/interview/prompt-interview/prompt-failure.md)）与对抗提示注入的系统加固方式（见 [/interview/prompt-interview/prompt-injection.md](/interview/prompt-interview/prompt-injection.md)）。
* **多轮会话与上下文编排**：管理多轮长对话的状态截断与记忆（见 [/interview/prompt-interview/multi-turn-context.md](/interview/prompt-interview/multi-turn-context.md)）以及结合外部知识库的 RAG 提示词编排（见 [/interview/prompt-interview/rag-prompt.md](/interview/prompt-interview/rag-prompt.md)）。
