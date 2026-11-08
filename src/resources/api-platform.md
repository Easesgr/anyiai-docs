---
title: 商业模型 API 怎么选与怎么接？推理机制、协议网关与高可用实践
icon: mdi:power-plug-outline
tag:
- 大模型 API
- 模型网关
- 开发平台
description: 系统梳理商业大模型 API 的通信协议、Token 计费、首字延迟构成、约束解码与上下文缓存，并提供多厂商容灾网关设计。
---

在开发基于大语言模型的应用时，开发者通常有两种算力路径：一是采购或租用 GPU 运行开源权重，自行搭建推理框架并调优显存，具体部署实践见 [/resources/model-platform.md](/resources/model-platform.md)；二是通过网络调用各厂商托管的商业推理 API。商业 API 将集群调度、并行策略、显存管理与模型权重封装为标准 HTTP 接口，按实际消耗的 Token 数量或调用时长计费。

评估商业 API 不能只看每百万 Token 的标价。真实系统的稳定性受制于两阶段推理延迟、并发配额限制、各厂商协议差异以及故障容灾策略。

本文以一个具体的场景贯穿全文：构建一个**企业客服工单助手**——用户上传一份 5,000 字的产品售后条款并描述故障，系统需要一边流式返回诊断分析，一边输出严格符合 JSON 规范的工单数据（包含分类、优先级和处理建议），并在主模型限流或宕机时平滑切换备选厂商。

![模型API平台与聚合网关调用全景架构](/images/content-rebuild/8197d56d5cf59881/cover.png)

---

### 1. 通信模式、Token 计量与速率限制

商业模型 API 的基本交互形式是客户端发起 HTTP 请求，将结构化提示词（Prompt）与采样参数传递给服务端；服务端调度 GPU 集群执行模型的前向计算，再将生成的文本返回给客户端。

#### 通信模式：非流式与流式传输（SSE）
在工单助手场景中，模型生成 500 字的详细分析通常需要数秒到十数秒。如果采用普通的**非流式响应（Unary Request）**，客户端需等待所有文本全部生成完毕才能拿到完整的 JSON 数据包。这不仅会导致用户在空白屏幕前等待过久，也容易在网络不稳定或反向代理超时时导致整次请求中断。

因此，人机交互场景普遍采用基于 HTTP 的**服务端事件流（Server-Sent Events, SSE）**：
* 客户端请求头声明接收流式响应，服务端在响应头返回 `Content-Type: text/event-stream`。
* 底层推理引擎每解码出一个或几个 Token，就将其封装为事件片段（如 `data: {"choices": [{"delta": {"content": "..."}}]}

`）即时推送到连接中。
* 客户端逐块解析并在前端打字机渲染，使交互感知延迟大幅降低。

#### Token 计量与输入输出差价
大语言模型计算的最小单位不是字符或字节，而是 **Token**。各厂商采用不同的分词算法（如 BPE、WordPiece、Unigram），因此相同的自然语言文本在不同模型下的 Token 计数不同。在常见的多语言词表中，英文通常一个单词对应 1 个 Token 左右，而一个中文汉字通常对应 1 到 2 个 Token，遇到罕见生僻字或特定符号可能消耗更多。

在计费上，各云厂商通常公布不同的**输入 Token（Prompt Tokens）**和**输出 Token（Completion Tokens）**单价，常见定价中输出单价比输入更贵，但定价本身是供应商公布的商业策略，而非由单一物理定律推导而来：
* **输入阶段（Prefill 阶段）**：服务端接收文档时，所有输入 Token 可以并行送入 GPU 进行前向计算，硬件计算单元（如 Tensor Core）通常更容易保持较高利用率。
* **输出阶段（Decode 阶段）**：自回归解码需要逐步串行生成。在多租户批处理环境下，虽然模型权重读取可以被批次内的请求摊销，但每一步都需要从显存搬运各请求累积的 KV Cache，计算访存比较低，常受制于显存带宽（Memory Bandwidth Bound）。

此外，计费以服务端实际完成的前向计算为准。如果请求因网络超时或客户端主动断开而中断，服务端已生成的 Token 往往已产生费用，且已消耗的速率配额（TPM/并发）不会无条件全额释放。

#### 速率限制与限流错误（429）
为了保证多租户环境下的集群稳定性，商业网关会在入口处设置速率限制，通常涵盖三个维度：
1. **RPM（Requests Per Minute）**：每分钟允许提交的请求次数。
2. **TPM（Tokens Per Minute）**：每分钟允许处理的 Token 总量（通常同时计算输入 Token 与预估输出 Token）。
3. **并发连接数（Concurrent Requests）**：系统同一时刻允许处于“正在计算”状态的请求上限。

当工单系统遇到早高峰瞬时脉冲请求时，TPM 或并发指标很容易被击穿。网关会直接返回 HTTP 状态码 `429 Too Many Requests`。规范的 API 平台会在响应头中给出当前限流状态，例如 `x-ratelimit-remaining-tokens` 或指示需要等待时长的 `retry-after`。业务系统必须解析这些头部作为退避依据。

![API供应商的限流与计量管道机制](/images/content-rebuild/8197d56d5cf59881/fig-1.png)

---

### 2. 接口协议差异与聚合网关

虽然主流模型功能类似，但各大模型提供商的原生 API 格式存在明显差异，直接依赖单一厂商的官方 SDK 会给未来的迁移带来很高的代码重构成本。

#### 主流原生协议对比
* **OpenAI 兼容规范（Chat Completions API）**：当前事实上的行业标准。请求体核心是 `messages` 列表，每条消息包含 `role`（如 `system`、`user`、`assistant`）与 `content`；工具调用通过 `tools` 和 `tool_calls` 参数完成；输出统一挂在 `choices[].message`。
* **Anthropic Messages 规范**：参数设计更加严格。`system` 提示词作为与 `messages` 同级的顶层字段，不放在消息历史内；多模态与工具调用均被建模为内容块（Content Blocks），工具调用结果通过显式的 `tool_use` 与 `tool_result` 块回传。
* **Google Gemini 原生规范**：采用 `contents` 数组，角色分为 `user` 和 `model`；消息内容由 `parts` 列表组成，文本、图片二进制流和函数响应均以 Part 形式混排。

#### 模型聚合网关的工程角色
在工单助手的生产架构中，通常会在业务逻辑与外部模型提供商之间架设一层轻量级的模型网关（例如开源的 [LiteLLM](https://docs.litellm.ai/)、[vLLM](https://github.com/vllm-project/vllm) 兼容端点，或第三方聚合路由 [OpenRouter](https://openrouter.ai/)）。网关承担以下职责：
1. **协议转译**：业务服务统一按照 OpenAI 兼容规范发送请求，网关将其透明转换为对应厂商的原生格式（例如将 `system` 消息提升为 Anthropic 顶层参数）。
2. **密钥管理与收敛**：各厂商 API Key 保存在网关配置中，业务系统通过内网鉴权与网关通信，避免密钥泄露并在轮换时不影响业务。
3. **流式格式标准化**：将不同厂商返回的细微差异事件流，标准化为统一格式的 `delta` 事件推给客户端。

![API兼容网关请求转换流程](/images/content-rebuild/8197d56d5cf59881/fig-2.png)

<ReadMoreLock />

---

### 3. API 延迟指标拆解：TTFT 与 TPOT

评估 API 性能时，只看“端到端总时间”无法定位延迟瓶颈。在工单助手处理 5,000 字文档时，耗时由两个不同物理阶段构成。

#### 首字延迟（Time to First Token, TTFT）
TTFT 指从客户端按下发送键到接收到第一个流式字词所经历的时间。对于客户端而言，其构成如下：
$$\text{TTFT} = t_{\text{network\_req}} + t_{\text{queue}} + t_{\text{prefill}} + t_{\text{first\_decode}} + t_{\text{network\_resp}}$$
* $t_{\text{network\_req}} + t_{\text{network\_resp}}$：客户端到服务端的网络往返延迟。如果服务器位于跨国节点，网络往返基础耗时可能达数百毫秒。
* $t_{\text{queue}}$：云厂商排队调度时延。在高峰期算力饱和时，请求需要在服务端等待 GPU 空闲批处理槽位。
* $t_{\text{prefill}}$：预填充计算耗时。GPU 并行处理用户提交的 5,000 字文档，计算全部注意力键值并写入 KV Cache。输入文本越长，$t_{\text{prefill}}$ 越高。
* $t_{\text{first\_decode}}$：模型采样并生成第一个 Token 的耗时。

#### 单字解码耗时（Time Per Output Token, TPOT）
TPOT 指流式推送开始后，后续相邻两个 Token 生成之间的平均间隔时间，它决定了屏幕打字的平滑程度（仅在 $\text{Generated Tokens} > 1$ 时有定义）：
$$\text{TPOT} = \frac{\text{Total Response Time} - \text{TTFT}}{\text{Generated Tokens} - 1} \quad (\text{Generated Tokens} > 1)$$
若请求仅生成单个 Token，则不存在后续解码间隔。在预填充完成后的解码阶段，模型每次前向传播生成新 Token，受显存带宽、在线 Batch 规模和底层调度策略共同影响。如果服务端并发排队过满，显存数据搬运竞争加剧，TPOT 也会随之拉长。

#### 生产监控核心指标
在生产环境中，需要针对以下指标建立监控报警：
* **P95 / P99 TTFT**：监控最差情况下用户的等待时间，排查长提示词或网络跨区域波动。
* **P50 / P95 TPOT**：评估流式阅读体验是否出现顿挫。
* **错误分类率**：严格区分业务错误（4xx 系列）、限流（429）与服务端过载宕机（5xx），云厂商通常在可用性 SLA 中将 429 排除在故障时间之外。

![推理两阶段延迟构成分解图](/images/content-rebuild/8197d56d5cf59881/fig-3.png)

---

### 4. 结构化输出与上下文缓存

对于工单助手，生成的文本最终要沉淀为结构化字段（如 `{ "category": "hardware", "priority": "high" }`）并写入数据库，同时多轮追问不能重复计算 5,000 字的背景说明。这需要依靠两项关键机制。

#### 结构化输出：从 Prompt 诱导到约束解码（Constrained Decoding）
早期的做法是在 Prompt 中要求“请输出 JSON”，但基于纯概率采样的模型在长文本或极端输入下，可能偶尔漏掉闭合大括号、输出转义错误或夹杂解释性文字。

目前主流商业平台支持基于 **JSON Schema 的结构化输出**（如 OpenAI 的 `response_format: {"type": "json_schema", "strict": true}`）。其实现方式是在推理引擎的解码步引入**约束解码**机制：
1. 服务端将传入的 JSON Schema 预编译为上下文无关文法（CFG）或有限状态机（FSM）。
2. 在自回归采样的每一步，状态机根据当前已生成的字符前缀，确定下一个位置所有合法的 Token 集合。
3. 在 Logits 采样前，服务端将所有非法候选 Token（例如在属性名后期望冒号却生成了字母）的概率置为负无穷（$-\infty$）。

通过掩码约束，输出在词法和语法上能严格匹配 Schema。但需注意适用边界：若模型达到了最大输出长度（`max_tokens`）被截断，输出依然可能是一个尚未完成的不闭合结构；同时复杂的 Schema 会在请求初始引入一定的状态机编译耗时。

![语法掩码约束解码机制对比](/images/content-rebuild/8197d56d5cf59881/fig-4.png)

#### 上下文缓存（Context Caching / Prompt Caching）
当用户针对同一份 5,000 字产品手册连续追问三个问题时，若每次都将整个手册发送给模型重新做一次完整的 Prefill，不仅会增加账单成本，还会带来很高的首字延迟。

上下文缓存技术允许服务端在内存或显存中保留该前缀文本计算出的 KV Cache，后续请求直接复用：
* **前缀匹配机制**：上下文缓存基于自左向右的前缀哈希匹配。在调用相同模型及版本、达到供应商规定的最小 Token 门槛，且缓存未因 TTL 或显存逐出策略失效的前提下，只要系统提示词和背景文档的前缀完全一致，即可复用 KV Cache。若在最前端修改了标点或空格，后续缓存将全部失效；此外精确缓存仅复用前缀计算，最终生成结果仍受模型采样参数与上下文影响。
* **厂商实现差异**：
  * **Anthropic**：要求显式设置缓存断点。在消息的 block 中标记 `"cache_control": {"type": "ephemeral"}`，命中缓存时输入 Token 计费享有显著折扣，未命中时写入会有少量额外费用。
  * **OpenAI / DeepSeek**：采用自动前缀匹配。当提示词长度超过阈值（如 1024 Token）且命中近期的热点前缀时，服务端自动复用 KV Cache，无需客户端显式声明。
  * **Google Gemini**：支持显式创建具有存活时间（TTL）的命名缓存对象，适合长期挂载的大型知识库文档。

![上下文缓存的前缀树匹配与复用机制](/images/content-rebuild/8197d56d5cf59881/fig-5.png)

---

### 5. 容灾网关设计：重试、熔断与跨厂商降级

依赖单一直连商业 API 存在单点故障风险。当主力供应商遇到区域网络抖动、算力过载导致 503 报错，或者触发 429 限流时，客户端网关需要自动执行退避重试或路由降级。

#### 错误分流与全抖动指数退避
网关必须针对不同错误码采取不同的处置动作：
* **不可重试错误（400, 401, 403, 422）**：参数格式错误、认证失败或模型不存在。这类错误由客户端代码引起，重试无意义，应立即终止并向业务层抛出异常。
* **限流错误（429）**：配额耗尽。优先读取响应头中的 `retry-after`（秒）或 `retry-after-ms`。若无明确提示，使用指数退避。
* **瞬态服务端错误（500, 502, 503, 504 与连接超时）**：执行指数退避重试；若多次重试失败则触发熔断，切至备用厂商。

重试时如果所有并发客户端使用固定的间隔（如每 2 秒重试一次），会在服务端形成同步冲击（惊群效应）。因此建议采用带**全抖动（Full Jitter）**的指数退避：
$$t_{\text{sleep}} = \text{Uniform}(0, \min(t_{\text{max}}, t_{\text{base}} \times 2^{\text{attempt}}))$$
在最大上限内生成一个随机均匀分布的时间，将重试流量在时间轴上均匀打散。

![模型网关熔断降级与重试时序图](/images/content-rebuild/8197d56d5cf59881/fig-6.png)

#### 最小可用容灾客户端实现
以下 Python 示例展示了一个具备“带全抖动的指数退避”、“错误码分流”以及“主备厂商平滑降级”的网关调度模块：

```python
import time
import random
import requests
from typing import List, Dict, Any

class ProviderEndpoint:
    def __init__(self, name: str, base_url: str, api_key: str, model_id: str):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id

    def call_chat(self, messages: List[Dict[str, str]], timeout: float = 15.0) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.3
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()

class ResilientModelGateway:
    def __init__(self, endpoints: List[ProviderEndpoint], max_retries: int = 2):
        self.endpoints = endpoints
        self.max_retries = max_retries

    def _get_jittered_delay(self, attempt: int, base: float = 0.5, cap: float = 6.0) -> float:
        ceiling = min(cap, base * (2 ** attempt))
        return random.uniform(0, ceiling)

    def complete(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        last_error = None

        for endpoint in self.endpoints:
            for attempt in range(self.max_retries + 1):
                try:
                    return endpoint.call_chat(messages)
                except requests.exceptions.HTTPError as err:
                    status = err.response.status_code if err.response is not None else 0
                    last_error = err

                    # 客户端参数或鉴权错误，无需重试或换厂商
                    if 400 <= status < 500 and status != 429:
                        raise err

                    # 遇到 429 或 5xx 错误
                    if attempt < self.max_retries:
                        # 检查是否有显式 Retry-After 头（单位为秒）
                        retry_after_str = err.response.headers.get("retry-after") if err.response else None
                        if retry_after_str and retry_after_str.isdigit():
                            sleep_duration = float(retry_after_str)
                        else:
                            sleep_duration = self._get_jittered_delay(attempt)
                        time.sleep(sleep_duration)
                    else:
                        # 当前提供商重试次数用尽，退出重试循环，降级到下一个厂商
                        break
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
                    last_error = err
                    if attempt < self.max_retries:
                        time.sleep(self._get_jittered_delay(attempt))
                    else:
                        break

        raise RuntimeError(f"所有可用提供商请求均失败，最后捕获异常: {last_error}")

# 验证示例
if __name__ == "__main__":
    # 配置主厂商与备用厂商端点（均遵循 OpenAI 兼容格式）
    primary_provider = ProviderEndpoint(
        name="Primary-Service",
        base_url="https://api.openai.com/v1",
        api_key="mock-key-1",
        model_id="gpt-4o-mini"
    )
    fallback_provider = ProviderEndpoint(
        name="Fallback-Service",
        base_url="https://api.deepseek.com/v1",
        api_key="mock-key-2",
        model_id="deepseek-chat"
    )

    gateway = ResilientModelGateway(endpoints=[primary_provider, fallback_provider], max_retries=2)
    # 生产环境中调用 complete 即可在主厂商异常时平滑切至备用厂商
```

---

### 6. 主流商业平台特征与选型参考

在选择商业模型服务时，应结合团队的数据合规要求、目标用户的网络物理距离以及特定能力需求进行选型：

| 供应商 / 托管平台 | 官方接入规范入口 | 核心接口特性 | 工程适用场景与限制 |
| :--- | :--- | :--- | :--- |
| **OpenAI API** | [OpenAI 官方文档](https://platform.openai.com/docs) | 原生支持严格 JSON Schema 约束解码；工具调用格式完备；生态 SDK 覆盖广 | 需注意服务器出口网络延迟；多组织层级管理和分级并发配额严格 |
| **Anthropic Claude API** | [Anthropic 官方文档](https://docs.anthropic.com/) | 显式断点 Prompt Caching 粒度精确；长文本推理分析质量领先 | 原生系统提示词在顶层定义；需网关处理其 Content Blocks 协议差异 |
| **Google Gemini / Vertex AI** | [Gemini API 文档](https://ai.google.dev/gemini-api/docs) · [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs) | 原生超长上下文窗口；多模态数据混排支持原生流式处理；Vertex AI 提供企业级 VPC 内网隔离 | 原生接口为 `contents/parts` 架构；使用企业版治理需依赖 Google Cloud 权限体系 |
| **DeepSeek API** | [DeepSeek 官方平台](https://platform.deepseek.com/) | 兼容 OpenAI 标准接口；中文推理与长文本具备显著成本优势；支持自动上下文缓存 | 峰值时段需监控排队延迟波动（TTFT 偶发升高），生产推荐配置降级通道 |
| **国内云平台托管（百炼/方舟）** | [阿里云百炼](https://www.aliyun.com/product/bailian) · [火山方舟](https://www.volcengine.com/product/ark) | 国内合规资质齐备；境内网络延时极低；同时托管原生大模型与开源微调权重 | 厂商自带 SDK 接口命名略有差异；建议经由兼容网关解耦业务依赖 |

在构建完整的企业级 AI 服务时，还需要结合租户权限隔离与多级配额治理，具体方案见 [/interview/ai-system-design-interview/multi-tenant-platform.md](/interview/ai-system-design-interview/multi-tenant-platform.md) 与 [/llm-100/evaluation/llm-platform-design.md](/llm-100/evaluation/llm-platform-design.md)。若需要配合工具调用平台或内容生成流水线，可参考 [/interview/ai-system-design-interview/agent-tool-platform.md](/interview/ai-system-design-interview/agent-tool-platform.md) 与 [/interview/ai-system-design-interview/content-generation-platform.md](/interview/ai-system-design-interview/content-generation-platform.md)；构建外部知识库检索时，可进一步查阅 [/resources/vector-db.md](/resources/vector-db.md) 与 [/resources/rag-framework.md](/resources/rag-framework.md)。
