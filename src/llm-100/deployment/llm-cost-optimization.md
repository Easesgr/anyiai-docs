---
title: 如何降低大模型应用成本？路由分流、Prompt压缩与多级缓存实践
icon: mdi:book-open-page-variant
description: 系统讲解大模型应用层成本控制：从输入输出计算特征、大小模型级联路由、Prompt结构修剪，到精确/语义/KV前缀分层缓存与投机采样落地。
---

在上一讲[/llm-100/deployment/api-vs-self-hosting.md](/llm-100/deployment/api-vs-self-hosting.md)中，我们对比了商业 API 与私有化部署的总拥有成本（TCO）。然而，无论底座选择云端商用模型还是自建推理集群，如果把应用层发出的所有请求全量、原样传递给顶配的基础大模型，系统账单和显存开销都会随请求量与上下文长度呈线性甚至超线性增长。

大模型应用的成本优化不是单一节点的微调，而是贯穿请求接入、提示词组装、路由分发、推理执行到度量监控的全链路工程。本讲以一个日均 10 万次请求的电商智能客服系统为贯穿案例，拆解可落地的端到端成本控制体系。

![大模型应用端到端成本优化全景图](/images/content-rebuild/130227c73b4adbfb/cover.png)

---

## 成本结构分析与多层阻截漏斗

要降低大模型应用的账单，首先需要理解大模型推理在不同阶段的硬件开销特征。在中小批量在线服务场景下，大模型处理文本通常分为两个阶段：
1. **输入阶段（Pre-fill）**：模型将输入的 Prompt 全部转换为内部表示并计算注意力。在常见输入长度下矩阵计算并行度较高，通常更偏向**算力受限（Compute-bound）**。
2. **输出阶段（Decode）**：模型自回归逐字生成回答。每次生成少量 Token 都需要完整读取一次模型权重，在中小批次下计算强度较低，通常受限于**显存带宽（Memory-bandwidth-bound）**。

各大模型服务商公布的 API 计费标准中，**输出 Token 单价通常显著高于输入 Token（常见为 1.5 到 4 倍不等）**，这主要与自回归解码阶段占用 GPU 资源时间较长、难以跑满计算单元的硬件利用特征有关，但也受各供应商商业策略影响。围绕请求吞吐与 Token 消耗，降低成本的核心抓手主要有三个：
- **降频（减少请求总量）**：在最前端通过网关规则和缓存拦截请求，不触发大模型推理。
- **降维（减少 Token 数量）**：精简提示词结构、剔除冗余字段、限制输出最大步数，降低单次请求的 Pre-fill 算力与 Decode 显存占用。
- **降级（降低单位 Token 单价）**：利用任务难度分流，将大量简单任务分配给小参数模型或专用模型，仅把少量复杂任务留给昂贵的基础大模型。

![多层阻截成本治理漏斗与请求处理流向](/images/content-rebuild/130227c73b4adbfb/fig-1.png)

如上图所示，一个健壮的成本治理体系呈现漏斗型拓扑：
- **第一层（网关阻截）**：静态规则校验、无效请求拦截、敏感词/恶意探测过滤，阻断无意义请求。
- **第二层（分层缓存）**：精确匹配缓存与语义向量缓存，直接返回已计算结果。
- **第三层（上下文修剪）**：结构化数据紧凑化、历史对话滑动窗口、冗余标点与停用词修剪。
- **第四层（动态分流）**：依据任务意图与复杂度，将请求路由至轻量级分类器、专用小模型或多模态大模型。
- **第五层（后端加速）**：结合推理引擎的 KV 前缀缓存（Prefix Caching）与投机采样（Speculative Decoding）。

每一层的拦截与缩减，都会直接减少后端推理集群的显存占用与计算压力。

---

## 精准模型分级路由：依据任务复杂度自动分流

以电商智能客服场景为例，每天 10 万次请求中，往往 75% 属于查询运费规则、修改收货地址格式、商品参数提取等简单任务；仅有 25% 涉及多方纠纷仲裁、复杂投诉逻辑推导或长篇合同条款解释。如果所有请求无差别调用顶配大模型，算力将被严重浪费。

![基于确定性规则与小分类器的模型分级路由机制](/images/content-rebuild/130227c73b4adbfb/fig-2.png)

### 级联路由的工作机制

级联路由（Model Cascading）采用“小模型先行尝试，大模型兜底校验”的渐进机制：
1. **前置轻量判别**：用户输入到达后，首先通过关键词规则或轻量分类器（如轻量文本分类小模型）预测任务难度。
2. **小模型低成本生成**：被判定为基础提取、翻译或常见问答的请求，直接派发给小型模型（如 7B/14B 模型或轻量 API）处理。
3. **结果校验与回退（Fallback）**：小模型输出后，系统对其进行格式检验（如 JSON Schema 校验）或置信度评估（如对数概率 Logprobs 均值）。若校验失败或输出置信度低于阈值，系统自动将原始请求升级传递给大模型重试。

### 路由效果定量测算（教学算例）

假设系统每日处理 $N = 100,000$ 次请求，平均每笔请求输入 $1,000$ Tokens，输出 $300$ Tokens。现有两款候选模型：
- **大模型（$M_L$）**：输入 ¥0.02 / 千Tokens，输出 ¥0.06 / 千Tokens。单次全量调用成本为：
  $$C_L = 1 \times 0.02 + 0.3 \times 0.06 = 0.038\text{ 元}$$
- **小模型（$M_S$）**：输入 ¥0.002 / 千Tokens，输出 ¥0.004 / 千Tokens。单次全量调用成本为：
  $$C_S = 1 \times 0.002 + 0.3 \times 0.004 = 0.0032\text{ 元}$$

系统设计如下分流策略：
- 75% 的流量（75,000 次）初筛为简单任务，派发给小模型承接。
- 25% 的流量（25,000 次）初筛为复杂任务，直接分配给大模型。
- 在小模型承接的 75,000 次请求中，假设有 5,000 次（占总流量 5%）因校验失败触发回退，需要重新调用大模型。

此时，大模型总调用量为初始直通流量与回退流量之和：$25,000 + 5,000 = 30,000$ 次。

在核算总成本时，必须计入那 5,000 次失败请求在小模型上已经产生的消耗（双重调用成本）：
$$\text{Cost}_{\text{route}} = 75,000 \times C_S + 30,000 \times C_L$$
$$\text{Cost}_{\text{route}} = 75,000 \times 0.0032 + 30,000 \times 0.038 = 240 + 1,140 = 1,380\text{ 元}$$

而如果不做路由、全量直接调用大模型，每日成本为：
$$\text{Cost}_{\text{direct}} = 100,000 \times 0.038 = 3,800\text{ 元}$$

分级路由使日调用成本降低了约 63.7%。

**关键边界**：级联回退的失败请求会产生“小模型 + 大模型”的双重调用成本，同时还会成倍拉长这部分请求的端到端响应延迟。如果小模型能力过弱导致失败回退率大幅上升（例如失败率超过 30%），小模型的算力损耗与频繁重试将快速抵消降本收益。工程上必须在小模型准确率与阈值设计之间做充分测试与折中。

---

## 提示词精简与无损压缩机制

在工程实践中，开发人员常在系统提示词中堆砌冗余的少样本示例（Few-Shot）、长篇说明和带大量空白缩进的 JSON 数据。冗余的上下文不仅增加计费 Token，还会显著拉长首字时间（Time to First Token, TTFT）。关于推理并发与延迟的关系，可参考[/interview/model-deployment-interview/optimization.md](/interview/model-deployment-interview/optimization.md)。

![提示词精简与上下文压缩对前后端开销的影响对比](/images/content-rebuild/130227c73b4adbfb/fig-3.png)

### 结构化数据紧凑化
在检索增强生成（RAG）或工具调用（Tool Calling）中，上下文通常包含大量数据库查询返回的 JSON 列表。漂亮打印（Pretty Print）带有大量换行符和缩进空格，且键名重复出现。

- **转为紧凑表格或紧凑分隔符**：例如将包含 10 条订单信息的 JSON 数组转为 Markdown 紧凑表格，或者以 Tab 分隔的纯文本行。在保持模型可读性的同时，能够剥离每一行重复出现的字段名，通常可减少 30%~50% 的输入 Token。
- **清理空值与无关元数据**：在将数据库实体序列化放入 Prompt 前，必须过滤掉业务无关的内部调试字段、创建时间戳冗余字段以及取值为 `null` 或空字符串的键。

### 上下文滑动窗口与修剪
多轮对话历史是 Token 消耗膨胀的常见来源。应避免无限制拼接完整历史：
1. **滑动窗口策略**：系统保留首轮 System Prompt（定义角色与规则），仅保留最近 $K$ 轮高频对话。
2. **定期轻量摘要**：当对话轮数超过阈值时，使用小模型对中间的多轮历史生成一份 200 字以内的关键上下文纪要，替换掉原始的数十轮散碎文本。
3. **信息熵修剪（Prompt Pruning）**：对于包含大量引用文本的场景，可以通过轻量评分模型或固定规则，过滤掉大量语气词、客套填充文本和重复标点。注意：对于包含代码实现、数学逻辑或敏感业务约束的场景，严禁做激进剪裁，防止误删关键操作符导致模型理解偏差。

---

## 多级缓存架构：从网关到 KV 前缀

缓存是阻断重复计算最直接的手段。在大模型应用架构中，缓存分为三个层级：

![精确缓存、语义向量缓存与推理引擎KV前缀缓存的分层协作架构](/images/content-rebuild/130227c73b4adbfb/fig-4.png)

### 1. 精确匹配缓存（Exact Match Cache）
* **机制**：对系统提示词（System Prompt）、用户输入（User Query）及模型采样参数（如 Temperature、Top_p）进行标准化拼装，计算 SHA-256 哈希值作为键，存入内存数据库（如 Redis）。
* **特点与适用场景**：命中时响应时间仅取决于网络与缓存读取（通常数毫秒），推理成本为 0 元。适合高频常见 FAQ、静态规则咨询和固定模板查询。
* **边界与一致性风险**：一方面，用户输入发生单个字符的改动（如多了一个问号或错别字）即导致哈希改变而未命中；另一方面，**哈希相等并不保证业务答案正确**。精确缓存必须结合模型版本升级、数据时效（TTL）、依赖的外部工具/数据库状态，以及多租户/用户权限进行校验。若底层业务状态已变更或请求包含私有权限数据，未设权限过滤与时效失效机制的缓存会引发陈旧数据或越权泄露。

### 2. 语义向量缓存（Semantic Cache）
* **机制**：将用户输入转换为稠密向量（Embedding），存入向量数据库。当新请求到达时计算向量相似度，若与历史高频提问的余弦相似度高于预设阈值（例如 0.95），则直接复用历史回答。
* **风险与防护机制**：
  - **租户隔离与权限越权**：不同租户或不同权限等级的用户可能提出相似问题（例如“查看本月审批记录”）。如果直接返回其他用户的缓存结果，将导致严重的数据泄露。**防护机制**：缓存键必须绑定 `tenant_id` 与用户角色 `role_id`，在向量相似度检索前施加元数据硬过滤（Pre-filtering）。
  - **实体与逻辑反转误判**：“北京今天下雨吗”与“北京明天下雨吗”、“张三能转岗吗”与“李四能转岗吗”在向量空间中余弦相似度极高。若直接返回缓存，会造成事实错误。**防护机制**：向量召回后，必须提取关键命名实体、时间词及否定词进行匹配校验，或调用轻量分类器进行等价性复核。
  - **向量计算本身的开销**：语义缓存命中虽然免除了大模型的生成开销，但需要额外调用一次 Embedding 模型并执行向量索引检索，会产生少量毫秒级延迟与向量计算开销，不能等同于精确缓存的纯内存开销。

### 3. 推理引擎 KV 前缀缓存（Prefix Caching）
* **机制**：现代推理框架（如 vLLM）支持在显存中以固定大小的 Block（如 16 个 Token）为基本单位，对公共 Prompt 的 Key 和 Value 张量进行哈希索引与显存复用。关于底层的分页机制，可参考[/llm-100/deployment/kv-cache.md](/llm-100/deployment/kv-cache.md)与[/llm-100/deployment/paged-attention.md](/llm-100/deployment/paged-attention.md)。
* **作用范围边界**：自动前缀缓存**仅能加速 Pre-fill 阶段**（消除共享前缀的重复计算，大幅降低首字延迟 TTFT）；它**不能加速自回归 Decode 阶段**，后续生成每个 Token 依然需要按步完整加载模型权重。
* **应用层排布规范**：应用层必须保证**提示词模板的静态前缀严格对齐**。将固定不变的 System Prompt、工具调用 Schema、Few-shot 示例统一放置在文本最前端，将动态变化的用户输入、当前时间戳放置在序列末端。若在静态内容中间插入动态变量，打断点之前已对齐的完整 Block 仍可命中缓存，但打断点之后的静态文本将因 Token 位置和 Block 边界错位而无法复用，导致前缀缓存命中率大幅下跌。

---

## 输出控制与投机采样

自回归模型每输出一个 Token，都需要从显存加载一次全量模型参数。输出步数越多，耗时与成本越高。

![投机采样中草稿模型生成与目标模型并行验证的步骤循环](/images/content-rebuild/130227c73b4adbfb/fig-5.png)

### 限制无效生成步数
很多模型在完成实际回答后，由于缺乏显式终止条件，会继续输出无意义的总结、甚至陷入复读死循环。应用层必须做好两项配置：
- **严格配置 `stop` 序列**：针对特定输出格式设置阻断符（如 `
Observation:`、`}
` 或自定义的 `<|im_end|>`），一旦解析到结构边界立即截断生成。
- **收紧 `max_tokens` 参数**：不要无差别全局配置 `max_tokens: 4096`。针对分类标签、简答或固定 JSON 返回场景，将上限紧缩至必要范围（如 64 或 128），防止模型因偶发异常跑满上下文窗口导致显存占用暴涨与计费超标。

### 投机采样（Speculative Decoding）机制

投机采样利用小参数“草稿模型”（Draft Model）与大参数“目标模型”（Target Model）协同工作：
1. **草稿生成**：由推理开销极低的小模型快速自回归生成 $K$ 个后续候选 Token（Draft Tokens）。
2. **并行验证**：大目标模型接收原始上下文与这 $K$ 个候选 Token，通过**单次前向传播**（Forward Pass）同时计算出这 $K$ 个位置的条件概率分布。
3. **采样与接受**：大模型根据概率分布依次验证这 $K$ 个 Token。假设前 $M$ 个 Token 满足分布要求被接受，大模型将其直接保留；而第 $M+1$ 个 Token 未通过验证时，大模型从校正分布中重新采样替代第 $M+1$ 个 Token，并丢弃后续所有草稿。

### 接受率的因果机制：为什么受任务类型与领域强烈影响？

投机采样的加速收益取决于草稿 Token 的**接受率**。接受率本质上反映了目标模型与草稿模型在特定上下文下**条件概率分布的重合程度**：
- **任务分布熵（Entropy）差异**：
  - **低熵确定性任务（高接受率）**：在代码生成、SQL 编写或固定格式提取中，Token 的语法结构高度受限，后续合理词汇往往只有 1~2 个选项。草稿模型预测分布与大模型高度一致，接受率较高，大模型单次前向验证能够一次性推进多个 Token，获得显著加速。
  - **高熵发散性任务（低接受率）**：在创意写作、多轮辩论或开放式头脑风暴中，后续语义展开极其发散。大目标模型依赖大量参数捕获深层长程依赖，而小草稿模型多倾向于高频通用词，二者概率分布脱节严重。大模型在验证时会频繁触发拒绝重采，导致接受率暴跌。
- **垂直领域知识覆盖差异**：如果应用涉及冷门垂直领域（如生僻医疗术语、小众工业控制协议），小草稿模型因参数容量有限未拟合该领域分布，极易生成幻觉词。大模型频繁拒绝并回滚，不仅无法节约时间，草稿模型的额外前向计算与大模型的回滚校验反而会导致端到端延迟与算力开销产生负收益。

---

## 成本度量与分摊体系（AI FinOps）

成本治理必须依赖客观可观测指标驱动迭代。关于系统级监控与日志设计，可参考[/interview/model-deployment-interview/monitoring-cost.md](/interview/model-deployment-interview/monitoring-cost.md)与[/architecture/token-cost.md](/architecture/token-cost.md)。

![按租户与功能拆分的AI成本度量、分摊与监控告警闭环](/images/content-rebuild/130227c73b4adbfb/fig-6.png)

### 体系构建的三大基石
1. **全链路统一标签（Trace Tagging）**：在 API 网关处为每个请求注入元数据追踪上下文，包括 `tenant_id`（租户/企业）、`feature_name`（功能模块，如客服、搜索、智能填表）以及 `user_id`。
2. **多维计量与配额熔断**：实时采集各业务维度的输入 Token、输出 Token、缓存命中类型（精确/语义/未命中）以及大模型回退率。为不同业务线配置按日/按月配额，当出现死循环调用的异常客户端时，网关及时触发限流与熔断。
3. **单次业务单位经济（Unit Economics）**：计算单次业务行为的大模型边际成本（例如“平均每成功解决一次退款咨询消耗 ¥0.018 算力”）。通过度量业务价值与算力支出的比例，指导优化策略的优先级排期。

---

## 最小可运行实验：精确缓存与轻量分流网关

下面是一个包含**SHA-256 精确缓存**与**基于规则轻量分流**的 Python 最小模拟实验。代码演示了请求如何在到达昂贵大模型前被多层机制阻截：

```python
import hashlib
import json
from typing import Dict, Any, Tuple

# 模拟两款模型的调用接口与单价（元/千Tokens）
MODEL_PRICING = {
    "deep-large": {"prompt": 0.02, "completion": 0.06},
    "fast-small": {"prompt": 0.002, "completion": 0.004}
}

class CostOptimizerGateway:
    def __init__(self):
        # 模拟内存精确匹配缓存
        self.exact_cache: Dict[str, str] = {}
        self.total_cost: float = 0.0
        self.intercepted_count: int = 0

    def _compute_cache_key(self, system_prompt: str, user_query: str, params: dict) -> str:
        """将提示词与采样参数做标准化哈希"""
        payload = f"{system_prompt.strip()}||{user_query.strip()}||{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _estimate_token_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = MODEL_PRICING[model]
        return (prompt_tokens / 1000.0 * pricing["prompt"]) + \
               (completion_tokens / 1000.0 * pricing["completion"])

    def route_request(self, user_query: str) -> str:
        """规则初筛：长度短且为基础提取/翻译选小模型，长逻辑或复杂推理选大模型"""
        complex_keywords = ["证明", "推理", "深入剖析", "重构代码", "规划架构"]
        if len(user_query) > 100 or any(kw in user_query for kw in complex_keywords):
            return "deep-large"
        return "fast-small"

    def process_query(self, system_prompt: str, user_query: str, params: dict) -> Tuple[str, str, float]:
        cache_key = self._compute_cache_key(system_prompt, user_query, params)
        
        # 1. 命中精确缓存：零模型推理成本
        if cache_key in self.exact_cache:
            self.intercepted_count += 1
            return self.exact_cache[cache_key], "EXACT_CACHE_HIT", 0.0

        # 2. 路由分流决策
        target_model = self.route_request(user_query)
        
        # 3. 模拟模型推理生成（字数近似估算Token，生产环境调用真实服务）
        prompt_tokens = len(system_prompt) + len(user_query)
        if target_model == "fast-small":
            generated_text = f"[小型模型响应] 完成基础处理: {user_query[:10]}"
            completion_tokens = 30
        else:
            generated_text = f"[大型模型响应] 针对复杂命题的深度推理结论"
            completion_tokens = 150

        cost = self._estimate_token_cost(target_model, prompt_tokens, completion_tokens)
        self.total_cost += cost
        
        # 写入缓存
        self.exact_cache[cache_key] = generated_text
        return generated_text, target_model, cost

# 验证实验
if __name__ == "__main__":
    gateway = CostOptimizerGateway()
    system_prompt = "你是一个严谨的电商助手。"
    default_params = {"temperature": 0.0}

    test_queries = [
        "翻译单词：Apple",             # 走小型模型
        "翻译单词：Apple",             # 相同请求，命中精确缓存
        "请证明哥德巴赫猜想的当前进展", # 包含复杂关键词，走大型模型
        "抽取实体：张三申请了退货退款"  # 走小型模型
    ]

    print("--- 开始网关处理模拟 ---")
    for q in test_queries:
        result, route, cost = gateway.process_query(system_prompt, q, default_params)
        print(f"Query: '{q[:12]}' | 路由决策: {route:16} | 产生开销: ¥{cost:.6f}")

    print(f"\n处理统计：缓存拦截 {gateway.intercepted_count} 次，累计支出 ¥{gateway.total_cost:.6f}")
```

运行上述脚本，第二次出现的“翻译单词：Apple”将直接返回 `EXACT_CACHE_HIT`，产生开销为 `¥0.000000`；而其余请求依据长度与关键词分别被分流至 `fast-small` 与 `deep-large`，演示了应用网关的多层阻截逻辑。

---

## 部署与推理优化专题回顾

至此，我们完成了**部署与推理优化专题**的全部内容。回顾整个脉络：
1. [/llm-100/deployment/inference-bottlenecks.md](/llm-100/deployment/inference-bottlenecks.md) 揭示了自回归解码受显存带宽限制（Memory-bound）的物理机理。
2. [/llm-100/deployment/kv-cache.md](/llm-100/deployment/kv-cache.md) 与 [/llm-100/deployment/paged-attention.md](/llm-100/deployment/paged-attention.md) 拆解了 KV 缓存机制以及分页管理解决显存碎片的设计思路。
3. [/llm-100/deployment/vllm-inference.md](/llm-100/deployment/vllm-inference.md) 与 [/llm-100/deployment/high-concurrency-serving.md](/llm-100/deployment/high-concurrency-serving.md) 探讨了高并发场景下的连续批处理（Continuous Batching）工程实现。
4. [/llm-100/deployment/quantization-methods.md](/llm-100/deployment/quantization-methods.md)、[/llm-100/deployment/hardware-estimation.md](/llm-100/deployment/hardware-estimation.md) 与 [/llm-100/deployment/local-serving-tools.md](/llm-100/deployment/local-serving-tools.md) 建立了硬件选型估算与模型量化落地的评估方法。
5. [/llm-100/deployment/api-vs-self-hosting.md](/llm-100/deployment/api-vs-self-hosting.md) 与本讲则站在应用落地视角，完成了自建 vs 云端选型决策与应用层降本闭环。

理解这些软硬件协同的底层规律与边界条件，大模型应用开发就能从被动的调用外部黑盒接口，转变为具备性能预期、开销可测算的严谨工程建设。

---

## 课后检验练习

### 练习目标
掌握级联路由成本折算、语义缓存安全机制，以及投机采样适用边界的因果推理。

### 场景设定
某垂直医疗问诊系统每日产生 200,000 次请求。团队计划实施以下降本改造：
1. 部署语义向量缓存（Semantic Cache），预计可拦截 20% 重复病症咨询。
2. 剩余 80% 请求通过级联路由器分流：60% 基础用药查询分配给 7B 医疗小模型（小模型预计失败率 10%，失败后回退至 70B 模型）；其余 40% 复杂多病症病例直接交由 70B 深度模型处理。
3. 小模型单次调用成本 ¥0.003，大模型单次调用成本 ¥0.040。

### 检验问题
1. 语义缓存（Semantic Cache）在带来成本下降的同时，可能引入哪两类典型业务风险？应通过什么工程手段进行防护？
2. 为什么在投机采样（Speculative Decoding）中，草稿模型接受率的高低不仅取决于小模型参数量，还受提示词领域与任务类型的强烈影响？
3. 结合本节算例，计算该医疗系统在实施语义缓存和级联路由改造后的每日理论 API 成本（假设命中语义缓存产生的大模型推理成本为 0 元）。

### 参考答案与详细解析

#### 问题 1 解析（语义缓存风险与防护）
* **风险一：租户隔离与权限越权**。不同用户可能提出看似相同的查询（例如“查询上月体检异常项”），若仅按输入文本余弦相似度命中缓存，将把前一个用户的敏感体检数据泄露给后一个用户。**防护机制**：在构建向量索引与检索时，必须强制将 `tenant_id` 和 `user_id`/`role_id` 作为元数据过滤字段（Pre-filtering），跨租户或不同权限间严禁共享缓存。
* **风险二：关键实体替换与逻辑反转被忽略**。语义向量计算关注文本全局特征，针对“阿司匹林能与布洛芬同服吗”与“阿司匹林能与头孢同服吗”，或者“儿童能吃吗”与“儿童不能吃吗”，余弦相似度极高（可能超过 0.95）。如果误命中缓存，会导致致命用药事故。**防护机制**：向量相似度召回后，必须通过命名实体提取（NER）或关键词对比对药品名、时间、否定词做一致性硬校验，未完全对齐则判定未命中。

#### 问题 2 解析（投机采样接受率的因果机制）
* **任务熵（Entropy）差异**：代码、SQL、严格格式 JSON 属于低熵任务，语法和上下文严重收敛，后续合法 Token 选项极少。小模型即便容量有限，也能与大目标模型输出重合的概率分布，接受率高。而开放式创意咨询、长链多步诊断属于高熵任务，大模型依赖数百亿参数建立深层逻辑关联，小模型容易退化为输出通用高频词，两者概率分布显著偏离，导致大量草稿 Token 被拒绝。
* **领域知识覆盖差异**：7B 级别草稿模型参数容量有限，若未针对生僻医学病理词汇做深入微调，其对垂直领域的条件概率预测极不准确，频繁出现幻觉 Token。这会导致大模型频繁触发拒绝与回滚，额外增加草稿推理与大模型验证开销，甚至拉低整体吞吐。

#### 问题 3 计算过程与结果
- **第一步：语义缓存拦截**
  - 拦截请求数：$200,000 \times 0.20 = 40,000$ 次（成本 0 元）。
  - 进入模型路由层流量：$200,000 \times (1 - 0.20) = 160,000$ 次。
- **第二步：小模型调用量与成本**
  - 派发给小模型的流量：$160,000 \times 0.60 = 96,000$ 次。
  - 小模型消耗成本：$96,000 \times 0.003 = 288$ 元。
- **第三步：大模型调用量与成本**
  - 复杂请求直接直通大模型：$160,000 \times 0.40 = 64,000$ 次。
  - 小模型校验失败回退重试：$96,000 \times 0.10 = 9,600$ 次。
  - 大模型总调用量：$64,000 + 9,600 = 73,600$ 次。
  - 大模型消耗成本：$73,600 \times 0.040 = 2,944$ 元。
- **第四步：每日理论总成本**
  - $\text{总成本} = 288 + 2,944 = 3,232$ 元。
  - （相比未经改造全量调用大模型的成本 $200,000 \times 0.040 = 8,000$ 元，日均节省约 59.6%）。
