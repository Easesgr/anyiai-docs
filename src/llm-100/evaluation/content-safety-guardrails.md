---
title: 大模型内容审核与安全护栏：如何设计防御管道？
icon: mdi:book-open-page-variant
description: 从输入过滤、分类小模型、状态机路由到流式输出缓冲，系统讲解大模型安全护栏的三层防御架构与工程实现。
---

在生产环境中运行大语言模型时，仅靠在系统提示词（System Prompt）中写“请不要回答违法问题”是不够的。我们在关于大模型安全风险的文章（见 [/llm-100/evaluation/llm-security-risks.md](/llm-100/evaluation/llm-security-risks.md)）中分析过，提示注入与对抗越狱可以直接改变模型的注意力优先级，使其忽略前置约束。要在工程上保障应用的合规与安全，必须在模型外部建立独立的代码拦截流水线——安全护栏（Guardrails）。

本讲通过一个具体的贯穿案例——“银行智能理财助手”，讲解如何搭建覆盖输入、推理和输出三道防线的安全防御管道。

![安全护栏工程化防御流水线全景](/images/content-rebuild/4a7a6cfe25b02934/cover.png)

---

### 防御管道的三个阶段与贯穿案例

设想我们正在为一家银行开发理财咨询机器人“安小理”。它支持用户查询基金理财知识、测算定投收益，并自动整理生成理财方案。在实际运行中，安小理会遇到四类安全与合规风险：

1. **恶意攻击与违规输入**：攻击者尝试利用特殊字符或提示注入套取内部提示词；
2. **用户隐私泄露（PII）**：用户在对话中直接输入自己的手机号、身份证号或银行卡号；
3. **业务越界（Off-topic）**：用户诱导理财助手写 Python 爬虫，或讨论与金融无关的敏感社会话题；
4. **输出违规与虚假承诺**：模型在生成建议时，可能说出“保证本金无风险、保底收益 15%”等违反金融监管的绝对化承诺，或者吐出畸形 JSON 导致前端崩溃。

为了处理上述风险，防御系统不能只在单一点位拦截，而是划分为三个阶段：

![纵深防御三层职责切分拓扑图](/images/content-rebuild/4a7a6cfe25b02934/fig-1.png)

* **输入预处理阶段（Pre-inference）**：在用户请求发送给主模型前执行。通过高性能规则、敏感词树、向量相似度和正则识别，拦截高危词汇、越狱样本，并对隐私信息完成占位脱敏。该阶段目标是以极低时延（通常在数毫秒至十毫秒内）过滤掉大量常见风险，降低主模型无谓的算力开销。
* **推理配合阶段（In-inference）**：在模型推理期间执行。如果采用私有化部署并拥有解码器控制权，可以通过设置 Logit Bias 压制违规词生成概率，或配置动态停止词（Stop Words）；如果是托管 API 场景，则通过并发分类小模型监控当前轮次风险。
* **输出后处理阶段（Post-inference）**：在内容返回给客户端前执行。重点负责流式文本的实时合规过滤、金融合规校验（关于如何检测模型事实幻觉可参考 [/llm-100/evaluation/reducing-hallucinations.md](/llm-100/evaluation/reducing-hallucinations.md)）、结构化输出 Schema 校验，以及将脱敏占位符还原为真实数据。

---

### 输入前置过滤：敏感词树、向量检索与数据脱敏

输入层承担全量流量的初筛，其核心要求是吞吐高、计算快。

![输入层敏感词匹配与数据脱敏机制](/images/content-rebuild/4a7a6cfe25b02934/fig-2.png)

#### 1. 基于 AC 自动机的高性能敏感词匹配

如果词库包含上万个违规词，用正则表达式或循环字符串匹配会导致 CPU 负载过高。工程上通常采用 Aho-Corasick（AC 自动机）算法。它在 Trie 树（字典树）的基础上加入了类似 KMP 算法的失效跳转指针（Fail Pointer）。

在扫描输入文本时，算法只需要遍历一次文本字符（长度为 $N$），即可同时匹配词典中所有的模式串（总命中数为 $M$），匹配时间复杂度为 $O(N + M)$，与词典词条数量无关。

```python
import ahocorasick

class KeywordGuard:
    def __init__(self, block_words: list[str]):
        self.automaton = ahocorasick.Automaton()
        for idx, word in enumerate(block_words):
            # 将词条加入字典树
            self.automaton.add_word(word, (idx, word))
        # 构建失效指针，完成自动机构建
        self.automaton.make_automaton()

    def inspect(self, text: str) -> tuple[bool, list[str]]:
        matches = []
        # 一次遍历扫描文本中所有匹配的违规词
        for end_index, (idx, original_value) in self.automaton.iter(text):
            matches.append(original_value)
        is_blocked = len(matches) > 0
        return is_blocked, matches
```

#### 2. 向量相似度判定语义边界

精确词匹配无法防范拼音缩写、同音替换或隐晦改写。为此，系统可以维护一个已知的恶意或越狱 Prompt 向量库（如已确认的越狱模板、暴恐语句等）。

系统使用轻量级嵌入模型（Embedding）将用户输入向量化为 $\mathbf{v}_{input}$，并与库中的恶意向量 $\{\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_m\}$ 计算余弦相似度：

$$\text{Score}_{malicious} = \max_{1 \le i \le m} \frac{\mathbf{v}_{input} \cdot \mathbf{u}_i}{\|\mathbf{v}_{input}\| \|\mathbf{u}_i\|}$$

* 若 $\text{Score}_{malicious} \ge \tau_{high}$（例如 0.85），直接判定为高危并阻断请求；
* 若处于 $[\tau_{low}, \tau_{high})$ 区间（例如 0.70 到 0.85），标记为可疑请求，交由下游专用的安全分类模型进一步研判。

#### 3. PII 动态脱敏与还原边界

当用户在理财助手里输入：“我的电话是 13800138000，请帮我查一下昨日定投结果”，手机号不应直接随上下文发送给外部大模型或存入非合规日志。

1. **前置脱敏**：通过正则或命名实体识别（NER）定位敏感数据，替换为结构化占位符，例如 `我的电话是 [PHONE_1]`。
2. **会话映射**：在当前会话的上下文缓存中暂存映射字典：`{"[PHONE_1]": "13800138000"}`，并设置生命周期（TTL）。
3. **后置还原**：大模型完成回答后，如果回答中引用了该占位符（例如“已为手机号 [PHONE_1] 查询完毕”），输出后处理层在发往客户端前将占位符替换回原值。

**工程边界注意**：大模型生成文本具有随机性，有时可能会破坏占位符格式（例如把 `[PHONE_1]` 改写成 `[手机 1]` 或 `PHONE-1`）。因此，还原模块必须具备容错正则解析能力；若还原失败，宁可保留占位符或提示用户重新查询，也不能让未受保护的数据暴露在不合规的存储位置。

---

### 语义审查与专用分类模型：Llama Guard 与推测性并行

当面对复杂的语境诱导或多轮拼接越狱时，单纯依靠词表与向量相似度很容易漏判，需要引入经过安全指令微调的专用分类模型（例如 Meta 开源的 Llama Guard 系列，或轻量级微调分类小模型）。

![分类小模型的串行与推测性并行调度对比](/images/content-rebuild/4a7a6cfe25b02934/fig-3.png)

#### 1. Llama Guard 工作机制

Llama Guard 是一个基于因果语言模型构建的审查模型。它的输入通常包括：
* 预定义的分类规范（Taxonomy，如暴力行为、违法建议、隐私窃取等分类代码）；
* 历史对话上下文；
* 当前待审查的文本（可以是用户的输入，也可以是模型的输出）。

模型经过特殊微调后，在推理时输出确定性的两行文本：
* 若合规：输出 `safe`；
* 若违规：输出 `unsafe\nS{N}`（$S{N}$ 为违规的类别编号，例如 S1 代表暴力，S6 代表财务欺诈）。

#### 2. 调度策略：串行阻断与推测性并行

引入审查模型带来的最大挑战是**时延增加**。如果每个请求都采用**串行阻塞模式**，用户感受到的首字时间（Time-to-First-Token, TTFT）将显著增加：

$$T_{\text{串行首字}} = T_{\text{输入审核}} + T_{\text{主模型Prefill}} + T_{\text{主模型首字生成}}$$

如果输入审核模型推理需要 200ms，用户的等待时间就会直接增加 200ms。为了兼顾响应速度与安全性，工程上常使用**推测性并行（Speculative Parallelism）**策略：

1. 网关收到用户请求后，同时将 Prompt 分发给“主模型”和“输入审核模型”；
2. 主模型开始执行 Prefill 阶段（长文本或大参数模型下，Prefill 加首字生成本身需要 100ms~300ms）；
3. 审核模型通常参数量较小（如 1B~3B 参数或采用量化），与主模型并发执行；
4. **正常情况（绝大多数合规流量）**：审核模型在主模型吐出第一个 Token 之前或刚开始生成时返回 `safe`，主模型正常流式输出，用户感受不到审核引入的时延；
5. **违规情况**：审核模型返回 `unsafe`，网关立即向主模型发送中止信号（Abort/Cancel），截断生成管道，直接向客户端返回合规拒绝提示。

| 调度模式 | 首字延迟（TTFT）影响 | 硬件与算力开销 | 拦截效果 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **串行阻塞** | 增加完整的审核推理耗时 | 仅需审核模型与主模型按序使用的资源 | 杜绝未审先发（审核放行前请求不进入主模型，但精度仍受分类模型自身能力限制） | 金融风控、高密场景、未成年人应用 |
| **推测性并行** | 几乎不增加延迟（取决于两者耗时差） | 增加并发计算开销；违规时主模型前几步计算会被废弃 | 高（只要审核先于内容推送完成即可截断） | 通用智能客服、C 端交互应用 |
| **异步旁路抽检** | 零增加 | 最低（仅对低可信用户或抽样流量触发） | 无法做事前阻断，主要用于事后封禁与审计 | 低敏内部问答、海量只读检索场景 |

---

### 业务边界约束：状态机拦截与意图路由

在很多业务场景下，用户的提问并不违法违规，但**超出了系统的业务职能**。例如在我们的银行理财助手里，用户输入：“帮我写一段 Python 脚本批量下载股票数据”，或者“如何看待昨天的国际政治新闻”。

这类请求如果直接丢给主模型，模型通常会顺应用户回答，导致理财机器人沦为通用的免费代码生成器或陷入非业务舆论争议。NVIDIA 推出的 NeMo Guardrails 提供了一种利用状态机和对话流（Colang）限制模型行为的设计思路。

![NeMo Guardrails 状态机拦截与意图转移流图](/images/content-rebuild/4a7a6cfe25b02934/fig-4.png)

#### 意图路由机制

系统通过规则、嵌入分类或极小的意图识别模型，先将用户输入归类到有限的业务意图集合中。若意图未命中已注册的业务流，则在请求触达主模型前直接短路：

```python
from dataclasses import dataclass

@dataclass
class UserRequest:
    text: str
    session_id: str

def intent_guard_router(req: UserRequest) -> tuple[bool, str]:
    # 1. 意图分类识别
    detected_intent = classify_intent(req.text)
    
    # 2. 允许的业务意图白名单
    ALLOWED_INTENTS = {
        "query_fund_yield",      # 查询基金收益
        "calc_compound_interest", # 计算复利
        "risk_assessment",        # 风险测评
        "human_service"          # 转人工服务
    }
    
    # 3. 意图越界校验
    if detected_intent not in ALLOWED_INTENTS:
        # 直接由护栏返回预置引导语，不调用昂贵的主模型
        return False, "我是您的理财咨询小助手，仅能解答理财与基金相关问题。请问有什么金融业务需要帮您？"
    
    return True, "intent_allowed"
```

通过在前置环节明确限制“允许转移的状态”，可以大幅减少主模型的非目标调用，降低算力消耗与潜在注入风险。

---

### 输出层合规校验：流式缓冲与结构化修复

为了保障用户体验，大模型通常采用流式传输（Streaming Server-Sent Events, SSE）。但流式输出对后置安全审核带来了特有的工程矛盾：**已经发送到客户端的 TCP 数据包无法在网络层撤回**。

如果等到模型全部生成完毕再做审核，流式的低延迟优势便完全丧失；如果模型生成一个 Token 就推一个 Token，一旦模型在中途吐出违规内容（如“保本保息 100% 兑付”），前端就已经把违规内容渲染出来了。

![流式输出的断句滑动缓冲与违规熔断机制](/images/content-rebuild/4a7a6cfe25b02934/fig-5.png)

#### 1. 标点符号滑动缓冲区（Sentence-Level Sliding Buffer）

工程上通行的折中方案是**按标点符号进行整句缓冲审查**：

1. 服务端收到主模型输出的单个 Token 后，暂存至先进先出（FIFO）字符缓冲区；
2. 缓冲区监听自然语言的断句标点（句号、问号、感叹号、换行符等）；
3. 当检测到一个完整意群句子（通常为 15~40 个 Token）时，将该句送入敏感词检测和轻量合规模型：
   * **审核通过**：将这批 Token 通过 SSE 推送给前端；
   * **审核违规**：立即向主模型发送停止推理请求，清空缓冲区，并向前端推送阻断标记事件（Event: `safety_block`）。
4. 前端收到 `safety_block` 事件后，在已展示的内容末尾追加标准合规提示（如：“抱歉，后续回答由于触发合规策略已终止展示”），并关闭连接。

通过将延迟粒度控制在整句级别（单句通常带来数百毫秒的首字等待时延），既保留了用户感知流式出字的节奏，又防止违规整句被直接发送到前端。

#### 2. 结构化 JSON 校验与降级修复

在调用 Agent 工具或返回结构化数据（如理财测算卡片）时，输出通常要求为严格的 JSON 格式。大模型生成过程中可能产生 Markdown 标记包裹、漏写闭合括号或生成多余字段等问题（关于大模型生成评估基准可阅读 [/llm-100/evaluation/llm-evaluation-benchmarks.md](/llm-100/evaluation/llm-evaluation-benchmarks.md) 与 [/llm-100/evaluation/generation-quality-metrics.md](/llm-100/evaluation/generation-quality-metrics.md)）。

```python
import json
from pydantic import BaseModel, ValidationError

class InvestmentPlan(BaseModel):
    plan_name: str
    risk_level: str
    expected_years: int

def validate_and_repair_json(raw_text: str) -> dict:
    # 第一步：清洗常见 Markdown 代码块包裹
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # 第二步：尝试解析并做 Schema 字段校验
    try:
        payload = json.loads(cleaned)
        validated_data = InvestmentPlan(**payload)
        return validated_data.model_dump()
    except (json.JSONDecodeError, ValidationError):
        # 第三步：触发确定性补齐逻辑（如修复未闭合的花括号）
        repaired = attempt_json_repair(cleaned)
        try:
            payload = json.loads(repaired)
            return InvestmentPlan(**payload).model_dump()
        except Exception:
            # 无法恢复时，抛出格式违规异常走兜底话术
            raise ValueError("Output schema validation failed")
```

---

### 降级响应与审计闭环

当系统任一环节触发拦截后，不能直接返回 500 报错或将底层报错堆栈暴露给客户端。需要提供统一的降级处理机制，并完成审计记录。

![安全事件处置与审计追踪闭环拓扑](/images/content-rebuild/4a7a6cfe25b02934/fig-6.png)

#### 1. 分级降级话术矩阵

拦截响应应根据触发的风险级别进行分流：
* **低风险（业务越界）**：友好引导。“作为理财小助手，我仅能协助解答基金理财相关问题，请问有其他金融业务需要咨询吗？”
* **中风险（触发合规敏感词）**：标准声明。“根据相关金融安全合规要求，该咨询内容无法直接生成具体投资指导，请以银行官方产品说明书为准。”
* **高风险（对抗越狱/恶意注入）**：极简静默阻断，不给攻击者提供任何可用于探测边界的提示细节。“请求无法被处理（错误码：SEC-403）。”

#### 2. 安全日志与持续优化闭环

每次拦截动作都应异步写入结构化的安全日志：
* `trace_id`：全链路追踪标识；
* `stage`：触发拦截的具体卡点（`pre_regex`、`pre_vector`、`parallel_guard`、`post_stream_buffer` 等）；
* `masked_input`：经过隐私脱敏后的原始输入；
* `matched_rule`：命中的词条 ID 或分类模型标签与其置信度。

被拦截的高危请求应定期脱敏并打包，交由安全运维人员进行人工复核。确认的新攻击样本将进入测试集用于防御回归验证，或用于安全小模型的微调更新。关于系统运行状态的可观测性与度量指标，在下一讲（见 [/llm-100/evaluation/llm-observability.md](/llm-100/evaluation/llm-observability.md)）中会详细展开。

---

### 学习验收与练习

1. **流式缓冲时延计算**：某模型生成速度为 30 Token/秒，服务端采用以句号断句的滑动缓冲区，平均一句话包含 24 个 Token。假设轻量检测模型的推理耗时为 50ms。客户端用户感知到首字呈现的时间（TTFT），相比无缓冲的逐字流式传输，理论上大约增加了多少毫秒？（写出推导过程）
2. **推测性并行边界分析**：在推测性并行架构中，如果审核模型推理需要 350ms，而主模型由于输入较短，在 120ms 时就输出了第一个 Token 并开始推送给用户。此时系统会面临什么问题？工程上应如何调整策略？

---

#### 练习参考思路

* **练习 1 参考思路**：
  * 在无缓冲模式下，主模型生成第 1 个 Token 即开始推送，等待时间为 $1 / 30 \approx 33.3\text{ms}$。
  * 在标点缓冲模式下，必须等待整句 24 个 Token 全部生成完毕才能送审。生成 24 个 Token 的耗时为 $24 / 30 = 0.8\text{秒} = 800\text{ms}$；
  * 送审耗时为 $50\text{ms}$，因此第一句被放行并推送的时间点为 $800\text{ms} + 50\text{ms} = 850\text{ms}$；
  * 相比无缓冲首次出字（33.3ms），延迟增量约为 $850 - 33.3 \approx 816.7\text{ms}$。这说明标点缓冲虽然保障了句子级安全性，但会增加用户感知首字到达的时间。

* **练习 2 参考思路**：
  * **面临问题**：审核结果比主模型出字慢 230ms（$350 - 120 = 230\text{ms}$）。若主模型直接推送，违规内容可能在审核出结果前就已经被用户看到，推测性并行的安全防线被击穿。
  * **工程调整**：
    1. **结合流式缓冲区**：即使主模型开始吐字，服务端不要立即发往客户端，而是通过本讲提到的句子级滑动缓冲区暂存，等待审核模型出结果后再放行第一句；
    2. **量化与裁剪审核模型**：将审核模型进行 FP8/INT4 量化或换用更小参数模型，确保审核耗时稳定在主模型首字耗时（Prefill + 首个 Decode）的时间窗口之内；
    3. **策略降级**：若某类请求标记为高风险用户，强制切换为串行阻塞模式，宁可牺牲响应时延，也要保证拦截有效性。
