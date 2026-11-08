---
title: 如何设计多模型路由：动态分流与降级策略
icon: mdi:routes
description: 解析多模型路由系统的决策维度、亚 20ms 路由器实现、级联降级树拓扑、防雪崩流量整形与动态配额分流机制。
---

在同时承接多种业务场景的 AI 系统中，没有任何单一模型能同时满足超强推理能力、毫秒级响应、超长大文本处理与极低调用成本。在前序内容 [/architecture/model-gateway.md](/architecture/model-gateway.md) 中，我们讨论了模型网关的协议转换与基础代理；但网关要发挥最大效益，必须配合**路由决策层**，即在请求发出前决定：“这个请求应该发给哪家供应商的哪个模型？”并在上游出现限流或故障时完成可控降级。

设想一个实际业务场景：一个日常处理 50 万次请求的企业级多功能助手，既要处理复杂的 Python 代码排错、长达百页的财报分析，又要响应大量的客服简单问答、打招呼和格式转换。若所有请求均由高规格旗舰模型处理，API 账单会迅速超标，且容易遭遇厂商的每分钟并发频率限制（429 Too Many Requests）；若全部使用轻量开源小模型，代码与长文档分析的完成率则无法满足业务要求。

多模型路由（Multi-Model Routing）系统的任务，就是在请求进入生成链路前的数毫秒至数十毫秒内，根据请求特征、租户配额与节点实时状态，选择最合适的模型或厂商实例，并在主链路异常时平滑降级。

![多模型动态路由架构总览](/images/content-rebuild/524dc56efee9208d/cover.png)

---

## 路由决策的四个关键维度

路由决策不是简单的字符串匹配，而是多个工程约束共同作用下的多目标权衡。一个典型的路由上下文由以下四个维度构成：

```
                    ┌─────────────────────────┐
                    │     用户请求与系统上下文   │
                    └────────────┬────────────┘
                                 │
       ┌────────────────┬────────┴───────┬────────────────┐
       ▼                ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  任务复杂度  │  │  上下文长度 │  │  延迟预算   │  │  财务成本   │
│ (代码/推理/ │  │ (Prompt+历史│  │ (TTFT / 端到 │  │ (单请求上限/│
│  闲聊/分类) │  │  是否超限)  │  │   端超时)   │  │  租户配额)  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       └────────────────┼────────────────┼────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      路由裁决引擎       │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
    ┌──────────────────┐                  ┌──────────────────┐
    │  首选目标模型实例  │                  │   备用降级级联树  │
    └──────────────────┘                  └──────────────────┘
```

### 1. 任务复杂度与类型
不同任务对模型的能力要求差别极大。多步骤逻辑推理、复杂代码编写与严谨的 JSON 结构提取，需要参数量充足、对齐充分的旗舰模型（例如 Claude 3.7 Sonnet、GPT-4o 或 DeepSeek-R1）；而文本轻度改写、意图分类、关键词提取以及日常问候，参数量在 8B 至 14B 级别的轻量模型即可完成。

### 2. 上下文长度与窗口匹配
输入的 Prompt 与历史对话累计 Token 数量直接划定了候选模型的硬性边界。若请求包含 80k Token 的长文档，支持上限仅有 32k 的模型将被立即过滤。此外，长文本的 Prefill（前向计算填充）耗时在不同模型和硬件配置下差异显著，上下文超长还会大幅拉高单次推理的计算与显存开销，相关分析可参考 [/architecture/context-management.md](/architecture/context-management.md) 与 [/interview/model-deployment-interview/latency-throughput.md](/interview/model-deployment-interview/latency-throughput.md)。

### 3. 延迟预算
交互型场景（例如代码实时补全、搜索建议流式输出）对首字延迟（TTFT，Time to First Token）通常要求控制在数百毫秒以内；而后台异步任务（例如批量解析历史日志、生成离线周报）对端到端完成时间更宽容，可排队数分钟。流式传输的具体设计参见 [/architecture/streaming.md](/architecture/streaming.md)。

### 4. 财务成本与配额上限
不同模型每百万 Token 的费用相差十倍至数十倍。在多租户系统内，根据租户的付费等级或单请求成本配额，限制候选模型集合的单价上限，可以防止低优先级的长尾流量耗尽团队预算，具体的核算方法可参考 [/architecture/token-cost.md](/architecture/token-cost.md)。

![路由决策的多维输入指标与权衡空间](/images/content-rebuild/524dc56efee9208d/fig-1.png)

---

## 路由决策器的工程实现与延迟权衡

路由模块位于请求的关键路径（Critical Path）上。如果路由判定本身耗时过长，就会直接吃掉下游换用轻量模型所节省的延迟。生产中常见的路由实现方式主要有三种：

| 路由方式 | 典型延迟范围 | 决策灵敏度 | 维护成本 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **确定性规则匹配** | < 1 ms | 较低（基于前缀、正则、元数据） | 规则条目随业务增长容易冲突 | 租户等级分流、指定端点、强制长文本截断 |
| **轻量特征分类器** | 2 ~ 10 ms | 中等（统计特征与树模型） | 需标注少量样本并离线训练模型 | 高并发、兼顾吞吐的长尾意图分流 |
| **语义向量路由** | 10 ~ 30 ms | 较高（密集向量相似度比较） | 依赖小型 Embedding 模型推理环境 | 固定意图槽位匹配、业务知识库分发 |

<ReadMoreLock />

### 1. 确定性规则匹配
通过解析 HTTP Header（租户 ID、业务场景 Tag）、请求体参数（是否携带工具调用声明、温度系数等）以及估算的 Prompt Token 数量，进行快速短路：
- 请求携带了自定义 Tool 定义且数量较多时，分流至工具遵循能力稳定的旗舰模型；
- 估算 Token 数量大于 64k 时，直接路由至大窗口专用模型；
- 免费试用租户的请求，强制分流至轻量模型池。

### 2. 亚 20ms 的轻量分类器
如果直接调用另一个大语言模型来充当路由器，即使是最小规模的 LLM，首字返回也往往需要 100ms 至 300ms，在实时网关中会带来明显的延迟劣化。构建低延迟路由分类器，工程上通常有两种低成本方案：

- **特征工程 + 浅层树模型**（耗时通常 < 5ms）：提取 Prompt 的统计特征（字符数、代码块标记、标点符号密度、数学符号标记、预置关键词命中情况），输入离线训练好的 GBDT 分类器（如 LightGBM，算法背景可参考 [/fundamentals/ml-basics/tree-models.md](/fundamentals/ml-basics/tree-models.md)），在多核 CPU 上可在数毫秒内给出难度分类。
- **小型 Embedding + 向量相似度**（耗时通常 10 ~ 20ms）：使用轻量级文本编码器（如在 CPU 上用 ONNX Runtime 运行的 30M~100M 参数模型，选型细节见 [/fundamentals/embedding/embedding-model-selection.md](/fundamentals/embedding/embedding-model-selection.md)），将输入转换为低维向量，与预先计算好的意图锚点向量计算余弦相似度。若与高难度意图（如代码架构重构、数理推导）的相似度高于阈值 $\tau$，则分配给旗舰模型。

![三种典型路由器的延迟与吞吐权衡](/images/content-rebuild/524dc56efee9208d/fig-2.png)

下面展示了规则过滤与轻量向量匹配相结合的路由器核心逻辑：

```python
import numpy as np
from typing import Dict, Any

class HybridRouter:
    def __init__(self, high_tier_threshold: float = 0.75):
        self.high_tier_threshold = high_tier_threshold
        # 预先离线提取好的复杂意图中心向量（此处模拟 5 个意图，每个 384 维，已做单位化处理）
        self.complex_intent_centroids = np.random.randn(5, 384)
        self.complex_intent_centroids /= np.linalg.norm(
            self.complex_intent_centroids, axis=1, keepdims=True
        )

    def extract_features(self, prompt: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "length": len(prompt),
            "has_code": "```" in prompt,
            "tool_count": len(metadata.get("tools", [])),
            "user_tier": metadata.get("user_tier", "standard")
        }

    def fast_embed_onnx(self, text: str) -> np.ndarray:
        # 生产环境中通过 ONNX Runtime 调用小型 CPU 编码器（384 维），单次推理通常在 8-15ms
        vec = np.random.randn(384)
        return vec / np.linalg.norm(vec)

    def route(self, prompt: str, metadata: Dict[str, Any]) -> str:
        features = self.extract_features(prompt, metadata)
        
        # 阶段 1：确定性规则短路 (< 1ms)
        if features["tool_count"] > 2 or features["has_code"]:
            return "flagship-model"  # 存在多工具或代码时分配旗舰模型
        if features["length"] > 32000:
            return "long-context-model"
        if features["user_tier"] == "free":
            return "lightweight-model"

        # 阶段 2：轻量向量相似度匹配 (10-15ms)
        query_vec = self.fast_embed_onnx(prompt)
        similarities = np.dot(self.complex_intent_centroids, query_vec)
        max_sim = float(np.max(similarities))

        if max_sim >= self.high_tier_threshold:
            return "flagship-model"
        return "lightweight-model"
```

---

## 级联降级树（Fallback Cascading）

调用外部商业模型 API 或私有集群时，网络超时、偶发 5xx 异常或 429 配额耗尽是常见状态。当首选目标模型调用失败时，系统需要按照预设路径自动切换，保障服务不中断。

![级联降级树拓扑结构](/images/content-rebuild/524dc56efee9208d/fig-3.png)

降级路径通常分为两个级别：

### 1. 同模型跨厂商/跨地域镜像切换（Vendor Failover）
对于开源权重模型（如 Llama 3.3 或 DeepSeek-V3），多家云厂商或自建推理集群均可提供兼容的推理端点。当主供应商发生 502/503/504 错误或连续请求超时时，网关保留请求的原始参数（Prompt、系统提示词、温度系数、JSON Schema），直接将请求转移至备选厂商的同规格镜像。需要注意，不同供应商部署同一开源模型时的量化实现（如 FP8 与 AWQ）、底层推理后端及对齐封装可能产生细微的输出风格差异；此外，若因请求超时而触发切换，原供应商服务端可能仍在继续推理并产生实际计费与配额扣减，重试时网关需防范双重成本损耗。

### 2. 跨模型族降级（Cross-Family Fallback）
当旗舰模型在全球范围内不可用或主模型池全线额度告警时，系统需要将流量转接至备选模型族的次级模型（例如从深度思考模型退化为通用对话模型，或从大参数商业模型降至轻量开源模型）。此时网关需要完成参数适配与请求重写：
- **特殊标记处理**：不同模型族对于推理思考标签（如 `<think>`）或角色标记的处理逻辑不同，需过滤或重构提示词格式；
- **上下文窗口修剪**：若主模型支持 128k 窗口，而降级目标仅支持 32k，降级模块需要执行滑动窗口修剪或丢弃早期历史消息，避免目标端直接返回 400 参数超限错误；
- **结构化输出降级**：部分模型不具备原生的 JSON 严格约束能力，降级时需自动退化为“Prompt 强制提示 + 输出后校验修复”方案。

### 429 限流下的防雪崩流量整形
当主模型返回 HTTP 429（Too Many Requests）时，不能盲目将所有超额流量全量灌向备用模型，否则极易引发级联故障。

其原因在于：
1. **备用池容量通常偏小**：备用通道的配额或实例规模往往只做容灾备用，承载能力弱于主通道。若主通道承载的数百 QPS 瞬时全部倾泻到备用池，备用池可能在极短时间内同样达到限流上限，导致所有降级节点依次被击穿。
2. **调用成本与结算差异**：备用通道的单价可能明显高于主通道，若在洪峰期未经控制地将流量全面转入，会导致账单在短时间内剧烈膨胀。

![429 故障转移中的流量整形与防雪崩机制](/images/content-rebuild/524dc56efee9208d/fig-4.png)

针对 429 的防雪崩应对策略包括：
- **令牌桶限流与平滑注入**：在流向备用池的路径前配置限流漏斗，确保进入备用模型的流量速率不超过其承载上限。超出部分直接返回可读的降级错误及客户端重试建议（`Retry-After` 头）；
- **按优先级分级降级**：优先保障付费租户或核心交互请求转移至备选池，普通低优先级流量返回排队或引导稍后重试；
- **带抖动的指数退避（Exponential Backoff with Jitter）**：结合熔断器状态机，在主模型返回 429 触发熔断阈值后先将其置入打开（Open）状态，阻断后续流量并直接转入降级路径；待冷却窗口期满后再置入半开（Half-Open）状态，以递增时间间隔配合随机抖动发送少量探测请求，避免大量请求在同一时刻集中重试造成二次冲击。

---

## 动态负载均衡与配额调度

在由多供应商、多地域实例组成的模型池中，静态轮询（Round-Robin）容易引发拥堵。因为各供应商背后的算力规模不同，Prefill 与 Decode 吞吐存在波动，且随时间呈现排队差异。高可用路由通常采用结合实时延迟与配额水位的加权调度策略。

![基于 EWMA 延迟与配额水位的动态加权分流算法](/images/content-rebuild/524dc56efee9208d/fig-5.png)

### 指数加权移动平均（EWMA）延迟追踪
模型生成耗时与输入输出 Token 数量强相关。为了反映实例当前的排队拥堵状况，网关需要持续统计每个节点的首字延迟（TTFT）或单位 Token 耗时，采用 EWMA 公式进行平滑追踪：

$$S_t = \alpha \cdot Y_t + (1 - \alpha) \cdot S_{t-1}$$

其中 $Y_t$ 为第 $t$ 次请求实际观测到的首字延迟，$\alpha \in (0, 1)$ 为衰减系数（通常取 0.1 至 0.2）。当某个服务商节点由于内部排队导致首字延迟快速上升时，其综合权重将在数个周期内自适应下调。

### 配额感知分流（Quota-Aware Routing）
由于商业 API 均有严格的 RPM（每分钟请求数）与 TPM（每分钟 Token 吞吐）上限，调度器必须根据滑动窗口内统计的消耗水位调整分配权重：

$$W_i = \max\left(0, 1 - \frac{\text{Current\_TPM}_i}{\text{Limit\_TPM}_i}\right) \times \frac{1}{S_{t, i} + \epsilon}$$

其中 $\epsilon$ 为防止除以零的极小常数。当供应商 A 的 TPM 水位超过 85% 时，其分配权重 $W_A$ 会明显衰减，后续请求会自动向配额充足的供应商 B 倾斜。这种策略能在不触发 429 的前提下，平滑利用多渠道的总并发能力。系统的熔断保护机制可参考 [/architecture/high-availability.md](/architecture/high-availability.md)。

---

## 分流效果观测与 A/B 实验闭环

引入动态路由后，系统的分支路径变多。工程团队需要持续验证：分流是否真正达成了降本目标？低等级模型是否拉低了业务产出质量？降级事件是否掩盖了主干系统的隐形故障？

![多模型路由可观测看板与 A/B 实验闭环](/images/content-rebuild/524dc56efee9208d/fig-6.png)

### 1. 核心可观测性指标
模型网关应在分布式链路追踪元数据（Trace Context）中记录完整的路由决策信息，日志与监控规范可参考 [/architecture/logging-trace.md](/architecture/logging-trace.md)：
- **分流命中比例（Routing Decision Distribution）**：旗舰模型、轻量模型、兜底模型各自承担的流量百分比；
- **路由器自身延迟分布（Router Latency P50/P95/P99）**：确保特征提取、向量编码与相似度判定的耗时稳定在预算以内（如 P99 < 20ms）；
- **降级触发率（Fallback Trigger Rate）**：监控由超时、5xx、429 引发的降级调用比例。若降级率出现异常凸起，通常意味着供应商配额调整有误或主链路故障；
- **单请求平均成本收益（Normalized Cost Per Request）**：对比全量使用旗舰模型的对照基线，核算路由策略带来的实际成本下降。

### 2. A/B 实验与质量评估闭环
不能仅根据账单下降来认定路由策略成功。生产环境通常会保留小比例的对照组（全量使用主模型），与应用动态路由的实验组进行长周期指标比对：
- **业务完成率（Task Success Rate）**：例如代码生成的自动化测试通过率、智能客服会话的单轮解决率；
- **用户隐式反馈（Implicit User Actions）**：界面中的复制、重试与“重新生成”点击率。若分流至轻量模型后，用户的重试率显著上升（例如上升幅度超出 5% 阈值），表明该类意图的难度被低估，需在特征配置中上调权重，将其重新收拢至高规格模型；
- **离线质量抽样评估（Offline LLM-as-a-Judge）**：定期抽取被分流到轻量模型的实际请求记录，由更强的离线评测流程与主模型生成结果进行盲测对比，从而校准意图分类器的判定边界，评测流水线的设计可参考 [/architecture/ai-evaluation.md](/architecture/ai-evaluation.md)。
