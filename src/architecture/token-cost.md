---
title: Token 成本如何控制？从多级缓存到配额与批处理架构
icon: mdi:piggy-bank
description: 剖析大模型调用费用暴涨的根本成因，通过精确与语义双层缓存、两阶段配额锁死及离线批处理构建低成本高防线的工程架构。
---

大模型系统上线后，调用账单往往比预想膨胀得更快。假设我们上线了一个电商客服助手：用户不仅会高频重复询问退换货规则，多轮对话还会把历史记录反复打包发送给模型；如果突发并发脚本恶意刷量，账户额度可能在几分钟内被全部耗尽。

在[/architecture/model-gateway.md](/architecture/model-gateway.md)中，我们搭建了负责路由与负载均衡的基础网关；在[/architecture/context-management.md](/architecture/context-management.md)中，我们处理了单次请求生命周期内的上下文裁剪。但这还不足以控制整体账单。要让成本保持在可控范围，工程架构上需要建立多道防线：用精确哈希与向量语义双层缓存拦截重复计算，用两阶段配额机制（Two-Phase Reserve-Commit）阻断并发透支，并利用供应商的离线批处理通道（Batch API）将非实时任务引流至批处理折扣队列。

![Token 成本控制与语义缓存架构概览](/images/content-rebuild/6d70950caa9beb3e/cover.png)

---

### 1. Token 成本模型：为什么多轮对话账单会滚雪球？

大语言模型的计费单位是 Token（其基本概念与分词机制可参考[/fundamentals/token-and-tokenizer/what-is-token.md](/fundamentals/token-and-tokenizer/what-is-token.md)与[/fundamentals/token-and-tokenizer/token-budget.md](/fundamentals/token-and-tokenizer/token-budget.md)）。从供应商账单来看，费用主要由以下三项因素决定：

1. **输入与输出的计费差价**：主流商业模型服务商的输出 Token 单价通常显著高于输入 Token（常见为 3 到 5 倍左右，具体由各供应商公布）。定价除商业与供需策略外，也与硬件负载特征相关：在典型服务场景中，输入预填充阶段（Prefill）并行度较高，更易利用计算算力；而输出解码阶段（Decode）需自回归逐 Token 串行推进，每步都需要搬运模型权重与 KV Cache，通常更容易受显存带宽限制（实际瓶颈受批大小与硬件配置影响）。因此，若放任模型生成大量无意义客套话，会放大整体调用成本。
2. **长上下文的历史累积**：多轮会话如果每次都把全部历史消息原样带回，第 $k$ 次交互就必须对前 $k-1$ 轮的全部输入和输出重新付费。在输入每轮恒定增加的假设下，整个会话的总输入 Token 消耗随轮数 $n$ 呈二次方增长（$O(n^2)$）。
3. **失控死循环与无效生成**：在自主 Agent 架构中，如果缺少明确的退出条件（详见[/frontier/loop-engineering/Loop_Engineering_的坑_无限循环_烧_Token_越改越差.md](/frontier/loop-engineering/Loop_Engineering_的坑_无限循环_烧_Token_越改越差.md)），或者前端没有设置合适的 `max_tokens`，模型陷入循环调用或幻觉长文本时，会导致无效扣费。

![大模型调用成本结构与长对话费用二次方增长模型](/images/content-rebuild/6d70950caa9beb3e/fig-1.png)

#### 成本滚雪球算例

假设一个智能客服会话持续 10 轮。用户每轮提问产生 100 个输入 Token，模型每轮返回 300 个输出 Token。模型输入定价设为 $C_{in} = \$2.5 / 10^6\text{ tokens}$，输出定价设为 $C_{out} = \$10.0 / 10^6\text{ tokens}$。

- **方案 A：不做任何截断，累加全量历史**：
  第 $k$ 轮请求发送给模型的输入 Token 为：$T_{in}(k) = (100 + 300)(k - 1) + 100 = 400(k - 1) + 100$。
  10 轮对话的总输入 Token 消耗为：
  $$\sum_{k=1}^{10} [400(k - 1) + 100] = 400 \times \frac{9 \times 10}{2} + 10 \times 100 = 18,000 + 1,000 = 19,000\text{ tokens}$$
  10 轮对话的总输出 Token 消耗为：$300 \times 10 = 3,000\text{ tokens}$。
  总成本为：
  $$\text{Cost}_{raw} = 19,000 \times 2.5 \times 10^{-6} + 3,000 \times 10 \times 10^{-6} = \$0.0475 + \$0.0300 = \$0.0775$$

- **方案 B：网关层启用滑动窗口（仅保留最近 2 轮历史）**：
  第 1 轮输入为 100 tokens；第 2 轮输入为 $400 \times 1 + 100 = 500$ tokens；从第 3 轮开始，输入上限被锁定在最近 2 轮问答加当前提问：$2 \times 400 + 100 = 900$ tokens。
  10 轮对话的总输入 Token 消耗降为：
  $$100 + 500 + 8 \times 900 = 600 + 7,200 = 7,800\text{ tokens}$$
  优化后的总成本为：
  $$\text{Cost}_{opt} = 7,800 \times 2.5 \times 10^{-6} + 3,000 \times 10 \times 10^{-6} = \$0.0195 + \$0.0300 = \$0.0495$$

在输出长度相同的情况下，仅通过限制输入上下文冗余，整体费用就降低了约 36.13%。在此基础上引入缓存进一步拦截重复提问，节省空间还会更大。

<ReadMoreLock />

---

### 2. 精确缓存与语义缓存：适用边界与假阳性风险

为了直接跳过模型推理过程，网关层通常设计两道缓存：基于精确哈希的严格匹配缓存与基于句向量的语义缓存。

![精确哈希匹配与向量语义检索的两级缓存架构](/images/content-rebuild/6d70950caa9beb3e/fig-2.png)

#### 2.1 精确缓存（Exact Match Cache）

精确缓存的实现方式与传统 Web 缓存一致。网关将决定模型输出的各项参数进行序列化并计算哈希值（如 SHA-256）：
$$\text{Key} = \text{SHA256}(\text{Model} \parallel \text{Temperature} \parallel \text{Top\_P} \parallel \text{SystemPrompt} \parallel \text{Messages})$$

- **适用场景**：高频相同的静态查询、模板化客服问答。适用于确定性采样（如 $\text{Temperature} = 0$）。若业务需要依赖较高温度值获得发散多样的创意，强制命中缓存会破坏采样分布，此时应支持客户端通过请求头（如 `Cache-Control: no-cache`）绕过缓存。
- **技术特征**：查询开销小（可直接通过 Redis 等内存键值库按哈希键检索），在输入字面与参数完全一致的前提下能稳定复现既有响应。但需要注意：哈希键相等并不等价于业务答案绝对正确。若未将用户权限、动态工具上下文、模型小版本变更及数据时效（TTL）纳入键空间或失效策略，仍可能返回过时或越权的脏数据。
- **局限性**：对文本格式变化极度敏感。只要提问中多了一个空格、标点符号改变，或者将“退货规则是什么”换成“请问退货规则”，哈希值就会彻底改变导致未命中。

#### 2.2 语义缓存（Semantic Cache）

语义缓存通过将用户提问转换为向量（Embedding），并在向量数据库中检索余弦相似度（Cosine Similarity）高于特定阈值的历史问答。一旦命中，直接返回之前的回复。

- **适用场景**：用户句式多变、但标准答案固定的只读型业务场景，例如通用产品使用指引、办事指南问答。
- **假阳性隐患**：语义缓存的最大风险在于**假阳性误判**。自然语言中，细微的字面差异可能代表相反的意图或完全不同的实体。例如“如何开通免密支付”与“如何关闭免密支付”，“北京飞上海航班”与“上海飞北京航班”，两者在向量空间中往往距离极近。一旦错误命中，系统会给用户返回完全相反或错误的答案。
- **建议跳过或严格受限使用语义缓存的场景**：
  在涉及用户隐私鉴权、动态账户状态、合同合规条款审查，以及对反义与否定极度敏感的医疗或金融核心问答中，通常建议根据业务风险策略禁用或旁路语义缓存。向量空间对“有”与“无”、“开通”与“关闭”等对抗性细微差异的区分能力有限，若发生假阳性误判，可能引发违规或业务损失。需要注意，禁用缓存直接调用模型亦不能保证模型输出绝对正确，但可规避因复用历史语义相似回复而导致的上下文错配与跨用户权限泄漏。

---

### 3. 语义缓存命中策略：阈值、实体校验与代码实现

语义缓存不能仅凭向量相似度判断，必须通过多重校验：合规准入、向量相似度阈值以及命名实体敏感度校验。

![语义缓存三级安全校验与高风险领域拦截流程](/images/content-rebuild/6d70950caa9beb3e/fig-3.png)

#### 3.1 相似度阈值标定

余弦相似度不是绝对的概率值，不同 Embedding 模型在不同业务语料上的分布差异明显：
- **阈值设定原则**：上线前必须采集业务历史真实问答对，构造包含语义相似样本与对抗负样本的测试集，进行阈值敏感度测试。在实际企业问答中，余弦相似度门槛通常需要卡在 0.92 到 0.96 之间。若门槛设得太低（例如低于 0.85），大量语义相反或主体不同的请求会被误判为命中，导致答非所问。

#### 3.2 命名实体与反义词校验（NER Guard）

即使两个句子的语义相似度达到 0.95，其关键实体依然可能存在冲突。在向量检索通过后，网关需要运行轻量级规则或实体识别校验：
1. **数字与金额**：“满 100 减 20”与“满 200 减 50”在向量上极其相似，但数字实体不同，必须判定为未命中。
2. **动作动词与反义词**：“支持开票吗”与“开票失败如何处理”，或者“如何注销”与“如何注册”，一旦触发反义或核心动作冲突，强制穿透缓存。
3. **专有名词**：人名、订单号、设备型号不匹配，拒绝复用缓存。

#### 3.3 双阶段语义缓存过滤示例

下面是一个包含余弦相似度计算与关键实体过滤的双阶段缓存校验实现：

```python
import re
from typing import Dict, Optional, Tuple
import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot_product = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot_product / norm) if norm > 0 else 0.0

class GuardedSemanticCache:
    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        # 模拟内存向量存储: query_text -> (embedding, response)
        self.cache_store: Dict[str, Tuple[np.ndarray, str]] = {}

    def _extract_critical_entities(self, text: str) -> set:
        # 提取关键数字与核心操作动作词（用于实体防撞校验）
        numbers = set(re.findall(r'\d+', text))
        action_keywords = {"退款", "充值", "修改", "查询", "注销", "取消", "开通", "关闭"}
        found_actions = {w for w in action_keywords if w in text}
        return numbers.union(found_actions)

    def set(self, query: str, embedding: np.ndarray, response: str) -> None:
        self.cache_store[query] = (embedding, response)

    def get(self, query: str, query_embedding: np.ndarray) -> Optional[str]:
        query_entities = self._extract_critical_entities(query)

        for cached_query, (cached_emb, cached_response) in self.cache_store.items():
            sim = cosine_similarity(query_embedding, cached_emb)
            
            # 第一道关卡：严格相似度阈值
            if sim >= self.threshold:
                # 第二道关卡：实体与动作校验
                cached_entities = self._extract_critical_entities(cached_query)
                if query_entities == cached_entities:
                    return cached_response  # 实体与相似度均匹配，安全命中
                else:
                    # 相似度虽高，但实体或动作冲突（例如开通 vs 关闭），拒绝复用
                    continue
        return None
```

#### 3.4 缓存淘汰与主动失效

缓存键必须设置生存时间（TTL，例如 24 小时）并结合 LRU（最近最少使用）淘汰。当知识库文章更新、业务规则调整或商品下线时，系统必须支持按业务标签（Tag-based Invalidation）主动批量清除缓存，防止返回过期的旧数据。

---

### 4. 租户配额管理：防止并发透支的两阶段扣减机制

在多租户系统或企业多部门共享网关的场景下，单靠事后统计无法阻断预算透支。

![多租户两阶段预扣与最终结算状态机](/images/content-rebuild/6d70950caa9beb3e/fig-4.png)

#### 4.1 为什么不能采用“调用完再扣费”？

大模型生成长文本通常耗时数秒到几十秒。如果系统采用“事后结算”，当某个租户账户余额仅剩 1 美元时，客户端可以利用高并发在同一秒内并发发起 100 个预期消耗各为 5 美元的长文本请求。由于发起瞬间账户余额大于零，所有请求均通过前置检查；几十秒后模型全部返回，产生 500 美元账单，导致租户账户严重透支（Overdraft）。

#### 4.2 两阶段扣减（Two-Phase Reserve-Commit）状态机

成熟的配额系统应采用类似分布式事务的两阶段扣减机制：

```
[客户端发起请求]
       │
       ▼
[1. Reserve 阶段]
网关计算预估费用: E_reserve = T_in * C_in + max_tokens * C_out
可用余额 Balance_avail = Total - Reserved
       │
       ├─ 若 Balance_avail < E_reserve ──► 抛出 HTTP 429/402，直接拒绝
       │
       └─ 若 Balance_avail >= E_reserve
              │
              ▼
       增加冻结配额: Reserved += E_reserve
              │
              ▼
       调用底层大模型 API
              │
       ┌──────┴──────┐
       ▼             ▼
   [调用成功]     [未实际调用/明确拒绝]
       │             │
       │             ▼
       │      [回滚 Release]
       │      全额解冻: Reserved -= E_reserve
       │      抛出网关异常
       ▼
[2. 结算与异常处理]
若调用成功: 解析真实 Usage，解冻预扣并按实际扣减
若超时/流式中断: 供应商可能已产生计费，按已接收 Token 预扣
或挂起待异步对账，不可盲目全额解冻，防止平台穿透欠费
```

1. **预估阶段（Reserve）**：
   - 网关接收到请求，根据输入消息长度精确计算 $T_{in}$，并根据参数中的 `max_tokens`（若未传则使用网关保护上限）预估最大输出 $T_{est\_out}$。
   - 计算预估冻结费用：$E_{reserve} = T_{in} \times C_{in} + T_{est\_out} \times C_{out}$。
   - 检查租户当前可用余额（$Balance_{avail} = Total - Reserved$）。若 $Balance_{avail} < E_{reserve}$，直接拦截请求并返回 `HTTP 402/429`，不产生底层调用费用。
   - 若余额充足，将 $E_{reserve}$ 计入冻结配额 `Reserved`。
2. **结算阶段（Commit / Release）**：
   - **正常完成**：从服务商返回中读取实际消耗（$T_{actual\_in}, T_{actual\_out}$），计算实际费用 $E_{actual}$。从 `Reserved` 中扣除 $E_{reserve}$，从总余额 `Total` 中扣除 $E_{actual}$。
   - **异常与失败处理**：对于请求未发出的熔断拦截或供应商明确未扣费的早期错误（如鉴权 401、配额拦截 429 等），执行回滚操作，全额解除冻结（$Reserved -= E_{reserve}$）。但需特别注意：若发生**调用超时、网络中断或流式传输被客户端主动关闭**，供应商后端可能已经完成预填充或生成了部分 Token 并计费；此时不能无条件全额解冻，而应先扣减流式已接收到的 Token 消耗，并将未决交易置于待对账状态，待拉取供应商用量账单后再多退少补，防止平台资金被并发透支。

#### 4.3 阶梯限流与预算熔断

根据租户当月预算消耗比例，网关可采取分级策略：
- **预警水位（消耗达 80%）**：异步通知租户管理员充值或确认额度，系统正常服务。
- **降级水位（消耗达 95%）**：限制使用高单价模型，通过多模型路由（见[/architecture/multi-model-routing.md](/architecture/multi-model-routing.md)）将请求降级分流至轻量低成本模型。
- **熔断水位（余额耗尽）**：阻断新发起的在线推理请求。

---

### 5. 离线非实时任务批处理：利用 Batch API 降低成本

并非所有大模型任务都要求在 2 秒内即时响应。例如模型评测（[/architecture/ai-evaluation.md](/architecture/ai-evaluation.md)）、文档离线向量化重构、知识库批量摘要以及测试数据合成等，业务关注的是吞吐量而非即时延迟。

![实时请求链路与离线异步 Batch API 折扣链路对比](/images/content-rebuild/6d70950caa9beb3e/fig-5.png)

#### 5.1 Batch API 的经济特性

主流商业模型供应商（如 OpenAI、Anthropic 等）多提供专门的 Batch 批处理接口：
- **工作机制**：将批量请求打包成标准 JSONL 文件上传，供应商利用闲置算力完成异步调度与执行。完成时效通常以宽裕窗口为准（例如部分供应商提供 24 小时内的交付保障，具体依平台 SLA 为准）。
- **成本优势**：**提供显著的价格折扣**（例如常见政策为输入输出单价降低约 50% 左右，具体以各供应商公布为准），且通常拥有独立的排队限额，不抢占在线实时业务的请求配额。
- **适用边界**：不能用于终端用户即时等待的交互式 Web 问答、流式生成（[/architecture/streaming.md](/architecture/streaming.md)）或高实时性工具编排链路。

#### 5.2 离线任务调度架构

网关可以封装统一的异步任务代理（结合[/architecture/async-tasks.md](/architecture/async-tasks.md)与[/architecture/long-task-execution.md](/architecture/long-task-execution.md)）：
1. **任务入队**：业务系统提交异步批处理请求，网关暂存至任务队列并向业务端返回 Task ID。
2. **聚合打包**：定时任务（例如每 30 分钟）扫盘，将积攒的离线任务聚合打包成 JSONL 文件，调用供应商 Batch API 提交作业。
3. **状态轮询**：后台 Worker 定期拉取作业状态（`validating` $\rightarrow$ `in_progress` $\rightarrow$ `completed`）。
4. **结果回填**：作业完成后拉取结果文件，解析状态并按 Task ID 回调通知业务系统；对个别超时的请求单条重试。

---

### 6. 端到端防御链路与可观测验收

将上述模块串联，可以得到完整的大模型请求处理链路：

![端到端 Token 成本防御系统数据流与控制状态机](/images/content-rebuild/6d70950caa9beb3e/fig-6.png)

1. **准入与精确匹配**：请求到达网关，首先检查场景标签与权限策略；高风险或对语义细微变化极度敏感的场景，按配置跳过语义缓存；常规场景计算请求完整哈希，命中有效缓存则直接返回。
2. **语义缓存检查**：未命中精确缓存时，调用 Embedding 计算向量并检索相似历史；通过相似度阈值和关键实体校验后返回，未通过则继续向下流转。
3. **配额预扣（Reserve）**：根据输入长度与 `max_tokens` 预估最大费用并冻结额度，余额不足则直接抛出 429。
4. **路由与调度**：判断任务优先级，实时交互请求走在线调用，离线大吞吐任务汇入 Batch 队列。
5. **实际结算（Commit）**：获取模型真实 Usage，释放冻结额度并扣除真实费用，记录日志以备对账。

#### 可观测验收指标体系

通过接入网关日志追踪（详见[/architecture/logging-trace.md](/architecture/logging-trace.md)），需要监控以下指标以验证成本控制效果：

| 指标分类 | 指标名称 | 计算公式 | 参考基准（示意） |
| :--- | :--- | :--- | :--- |
| **缓存效率** | 精确缓存命中率 | $\frac{N_{exact\_hits}}{N_{total\_requests}}$ | 高频固定问答场景通常在 $10\% \sim 25\%$ |
| | 语义缓存命中率 | $\frac{N_{semantic\_hits}}{N_{total\_requests}}$ | 知识库客服场景通常在 $15\% \sim 30\%$ |
| | 缓存假阳性率 | $\frac{\text{语义命中但用户反馈答非所问数}}{\text{语义缓存总命中数}}$ | 需持续监控并控制在 $< 0.5\%$ |
| **配额安全** | 租户透支发生率 | $\frac{\text{发生欠费透支的账单数}}{\text{总调用账单数}}$ | 严格为 $0\%$（依赖两阶段预扣拦截） |
| | 批处理折扣覆盖率 | $\frac{\text{通过 Batch API 运行的离线 Token}}{\text{全部离线任务 Token}}$ | 目标 $\ge 95\%$ |
| **上下文控制** | 单会话平均输入 Token 消耗 | $\frac{\text{会话全周期总输入 Token}}{\text{会话总轮数}}$ | 引入滑动窗口后，随轮数增长保持在平稳区间 |

通过在网关层建立缓存拦截、高风险场景隔离、两阶段配额预扣以及非实时批处理，可以在保证业务回答正确性的前提下，有效避免无效 Token 浪费与突发账单风险。
