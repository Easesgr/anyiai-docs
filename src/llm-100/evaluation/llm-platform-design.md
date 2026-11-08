---
title: 如何设计大模型应用平台？统一网关、Token限流与容灾调度
icon: mdi:book-open-page-variant
description: 解析大模型平台的核心架构，讲解协议归一化、Token预扣回补限流、语义缓存与熔断降级的实现机制。
---

以一个典型业务场景为例：一家电商公司的客服机器人、运营文案助手与内部研发代码补全同时接入了大模型。起初各团队直接调用商业厂商 API；但随着业务量上升，系统很快遇到实际故障：运营批量生成文案突发消耗了大量配额，导致在线客服频繁触发厂商 429 报错；夜间上游服务抖动时，前端直接超时报错白屏；由于各团队直接保存 API Key，月末财务也无法核算各部门的真实花费。

大模型应用平台（LLM Gateway / Platform）的作用，就是在业务系统与异构模型供应商之间建立一层可管控的反向代理服务，集中解决协议转换、多租户配额控制、语义缓存和故障容灾问题。

![企业级大模型应用平台全景架构](/images/content-rebuild/691f936ed5df58f5/cover.png)

---

### 1. 架构分层与数据流转

大模型平台在软件拓扑中位于上游业务应用与下游模型供应商之间，整体通常分为四个功能模块与一条审计监控链路：

![平台分层拓扑与数据流转架构](/images/content-rebuild/691f936ed5df58f5/fig-1.png)

整个请求链路按以下顺序流转：

1. **接入网关（API Gateway）**：业务端发起请求，网关统一处理 TLS 终结、租户身份验证（API Key 校验），并提供统一格式的 REST/SSE 接口。
2. **治理与计费引擎（Governance & Metering）**：维护各业务租户的并发请求数（QPS）与每分钟 Token 数（TPM）配额。请求进入时先按预估开销冻结额度，调用结束后按实际消耗结算并落库。
3. **调度与路由引擎（Router）**：先检查语义缓存是否命中；若未命中，再根据模型类型、当前服务健康状态与权重配置，将请求转发给最合适的服务实例。
4. **模型适配连接器（Connectors）**：负责将对外暴露的统一请求格式转为 OpenAI、Anthropic Claude 或私有部署框架（如 vLLM、TGI）专有的请求格式，并在响应时将私有格式封装回统一结构。
5. **安全与审计链路**：横向记录调用延迟、Token 消耗量与完整日志，并对接敏感词过滤与合规检测模块。

---

### 2. 统一协议与流式传输优化

为了降低调用方的学习和迁移成本，平台通常采用目前应用最广泛的 OpenAI Chat Completions 协议作为对外暴露的统一格式。

![跨厂商协议标准化与流式转换机制](/images/content-rebuild/691f936ed5df58f5/fig-2.png)

#### 跨厂商请求与响应规范化

不同厂商在参数名称和请求结构上存在细微差别。例如，Anthropic Claude 的 Messages 接口要求将系统提示词作为顶级参数 `system` 传递，而不是放在 `messages` 数组内；其流式响应事件采用 `content_block_delta`，而 OpenAI 协议则将系统提示词作为 `role: "system"` 放入消息数组，流式增量字段位于 `choices[0].delta.content`。

网关连接器需要针对不同上游维护映射逻辑：
- **入站转换**：将标准 JSON 请求体解析为目标供应商所需的参数结构。
- **出站归一**：将厂商返回的流式块或错误格式转为统一标准响应，例如统一错误码返回：

```json
{
  "error": {
    "message": "上游模型供应商返回限流错误 (429)",
    "type": "rate_limit_error",
    "code": 429,
    "param": null
  }
}
```

#### 流式传输（SSE）与连接参数配置

大模型生成长文本时通常需要持续数秒到数十秒，因此普遍采用 Server-Sent Events（SSE）长连接逐字推送生成内容。反向代理层如果配置不当，容易造成客户端无法实时接收字符：

* **关闭反向代理缓冲区**：在 Nginx 或 Envoy 等前置代理中，需要配置 `proxy_buffering off;`（或者在响应头添加 `X-Accel-Buffering: no`）。如果开启缓冲，代理层会等积攒满 4KB 或 8KB 数据才下发，导致前端出现长时间卡顿后一次性吐出大量文字。
* **调整长连接保活与超时时间**：大模型生成首字（TTFT）可能需要几百毫秒至数秒，整个长文本生成期间两端连接处于持续活跃状态。网关到上游模型的连接池应启用 TCP Keep-Alive，并将 `read_timeout` 适当放宽至 120 秒到 300 秒，避免被中间网络防火墙当作空闲断连。

---

### 3. 多租户配额控制：Token 预扣与异步回补

在传统 Web 网关中，限流通常只针对 QPS（每秒请求数）。但在大模型场景下，单靠 QPS 无法限制算力与成本消耗：1 次消耗 50 个 Token 的简短意图分类，与 1 次消耗 16,000 个 Token 的超长文档分析，对底层算力和账单的占用差异极大。因此平台必须同时限制 **QPS（并发请求量）** 与 **TPM（Tokens Per Minute，每分钟 Token 吞吐量）**。

![带有预扣与回补机制的双重限流流程](/images/content-rebuild/691f936ed5df58f5/fig-3.png)

#### 预扣与回补机制

当请求刚刚到达网关时，平台无法提前得知模型最终会生成多少个字符。如果直接放行，可能导致租户瞬间超出预算上限；如果一律按照模型上限（例如 4096 Token）扣除，并发请求就会因为虚高的预占配额而被大量拒绝。

网关通常采用**预估预扣、实际结算、差额回补**的计算流程：

1. **入站预估**：网关接收请求后，在本地使用轻量分词库（如基于 BPE 的快速计算器）计算输入的 Prompt Token 数量 $T_{\text{in}}$。结合请求中设置的 `max_tokens`（若未指定则赋予合理的工程默认值 $T_{\text{est\_out}}$），计算预冻结配额：
   $$T_{\text{reserve}} = T_{\text{in}} + T_{\text{est\_out}}$$
2. **原子预扣**：在 Redis 等分布式存储中使用滑动窗口算法检查该租户当前的 TPM 额度。如果配额充足，原子扣减 $T_{\text{reserve}}$ 并放行；若不足，立即返回 HTTP 429 响应并携带 `Retry-After` 说明。
3. **执行调用与实际结算**：上游模型生成完成，从流式结束标识或完整响应元数据中读取真实生成的 Token 数量 $T_{\text{actual}} = T_{\text{in\_actual}} + T_{\text{out\_actual}}$。
4. **差额回补**：计算多扣除的差额：
   $$\Delta = T_{\text{reserve}} - T_{\text{actual}}$$
   当回答提前收敛或实际消耗低于预估（$\Delta > 0$）时，系统通过异步任务将 $\Delta$ 返还给该租户的滑动窗口桶，同时把 $T_{\text{actual}}$ 记入真实账单数据库。需要注意，若遇到请求超时、流式异常断开或客户端主动中断，上游模型通常已完成 Prefill 并计算了部分输出 Token，供应商通常已对该部分实际计费；网关此时应依据实际已接收的增量 Token 或对账数据核算已消耗量，不能无条件全额释放已被实际消耗的预占配额，防止账单穿透。

---

### 4. 语义缓存的设计与边界

传统精确缓存通常使用请求字符串的哈希（如 SHA256）作为 Key。但在大模型场景下，即便哈希完全一致，也必须将模型版本、系统提示词、采样参数（如 temperature）及权限时效纳入缓存键校验，哈希相等并不能保证业务答案在当前上下文下依然正确。而在自然语言场景中，用户输入“怎么改密码”与“忘记密码如何重置”在字符上完全不同但意图一致，引入基于向量检索的语义缓存（Semantic Cache）可以在高相似度提问命中时直接复用历史响应，降低上游延迟与算力开销。

![语义缓存的向量检索与命中判定机制](/images/content-rebuild/691f936ed5df58f5/fig-4.png)

#### 向量相似度判定

语义缓存把历史 Query 的向量表示与生成答案存储在向量检索服务中。当新请求到达时，系统先计算新 Query 向量 $\mathbf{e}_q$ 与缓存中候选向量 $\mathbf{e}_c$ 的余弦相似度：

$$S(\mathbf{e}_q, \mathbf{e}_c) = \frac{\mathbf{e}_q \cdot \mathbf{e}_c}{\|\mathbf{e}_q\| \|\mathbf{e}_c\|}$$

当最高相似度 $S_{\max}$ 达到设定阈值 $\tau$ 时，直接提取缓存答案返回。

#### 阈值选择与适用条件

相似度阈值 $\tau$ 的取值需要根据业务实际验证：

* **$\tau \ge 0.98$**：要求输入几乎与原问题完全重合，泛化改写能力较弱，缓存命中率较低。
* **$\tau \le 0.88$**：匹配范围过宽，容易发生“串答”。例如将“怎样开通账户”与“怎样注销账户”判定为同义，导致返回错误答案。
* **常见参考区间（$0.92 \le \tau \le 0.96$）**：在此区间内可以匹配常见的同义短句，但关键实体（人名、单号、金额、时间）不同时，仍然存在错误复用风险。因此实践中通常结合关键实体匹配辅助校验。

**适用条件与决策边界**：是否启用语义缓存是依据业务风险、租户隔离、数据时效与确定性要求制定的工程策略。对于包含个人隐私数据、强上下文依赖的多轮对话或动态工具调用（Function Calling），若无严格的用户命名空间隔离与上下文哈希约束，通常建议跳过语义缓存以规避跨租户数据串答或脏状态复用。同时需注意，跳过缓存直接调用大模型并不保证生成结果绝对正确，仍需依赖下游业务逻辑校验与事实核对。

---

### 5. 容灾与故障降级：熔断状态机

外部大模型提供商的稳定性会受到网络波动、集群并发激增等外部因素影响，常出现偶发 429 限流或 504 超时。网关需要引入熔断状态机来防止请求堆积并提供自动降级能力。

![三态熔断器状态转移模型](/images/content-rebuild/691f936ed5df58f5/fig-5.png)

#### 熔断器三种状态

熔断器针对每个上游模型实例独立维护状态：

1. **Closed（正常通行）**：所有请求正常打向上游模型。系统统计滑动窗口（例如最近 60 秒）内的失败率。当请求总数达到基础样本量（例如 20 次）且失败比例超过阈值（例如 50%）时，状态转换为 **Open**。
2. **Open（熔断阻断）**：立即阻断发往该上游的后续请求。网关不再调用该故障模型，而是直接触发降级规则（如转由备用模型处理），避免连接池在故障节点上积压耗尽。同时启动定时器（例如静默 30 秒）。
3. **Half-Open（半开试探）**：静默时间到期后，熔断器允许放行少量试探请求（如 1~3 次调用）：
   * 若试探请求正常返回，表明上游故障已消除，状态转回 **Closed**，恢复主力路由；
   * 若试探请求再次失败，则重置静默计时器并重新进入 **Open** 状态。

![多级路由降级与灾备时序图](/images/content-rebuild/691f936ed5df58f5/fig-6.png)

如图所示，当主力商业模型不可用时，网关在几毫秒内触发规则将请求转交给自建部署的开源备用模型或次级厂商，保障上层业务不断线。

---

### 6. 代码实现：熔断降级调度器

以下 Python 代码演示了包含熔断状态机与备用降级逻辑的网关路由核心骨架：

```python
import time
from typing import Dict, Any

class ProviderCircuitBreaker:
    def __init__(self, failure_threshold: float = 0.5, recovery_timeout: float = 2.0, min_samples: int = 4):
        self.failure_threshold = failure_threshold  # 触发熔断的失败率阈值 (50%)
        self.recovery_timeout = recovery_timeout    # 熔断休眠时长 (秒)
        self.min_samples = min_samples              # 最小计算样本数
        self.state = "CLOSED"                       # CLOSED, OPEN, HALF_OPEN
        self.last_state_change = time.time()
        self.window_requests = 0
        self.window_failures = 0

    def can_pass(self) -> bool:
        now = time.time()
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            # 静默期结束后转为半开，放行试探流量
            if now - self.last_state_change > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.last_state_change = now
                return True
            return False
        if self.state == "HALF_OPEN":
            return True
        return False

    def record_result(self, success: bool):
        now = time.time()
        if self.state == "CLOSED":
            self.window_requests += 1
            if not success:
                self.window_failures += 1
            if self.window_requests >= self.min_samples:
                failure_rate = self.window_failures / self.window_requests
                if failure_rate >= self.failure_threshold:
                    self.state = "OPEN"
                    self.last_state_change = now
                    self._reset_window()
        elif self.state == "HALF_OPEN":
            if success:
                # 试探成功，切回正常通行
                self.state = "CLOSED"
                self.last_state_change = now
                self._reset_window()
            else:
                # 试探失败，继续熔断
                self.state = "OPEN"
                self.last_state_change = now
                self._reset_window()

    def _reset_window(self):
        self.window_requests = 0
        self.window_failures = 0

class GatewayRouter:
    def __init__(self):
        self.primary_breaker = ProviderCircuitBreaker()

    def mock_call_upstream(self, provider: str, should_fail: bool) -> Dict[str, Any]:
        if should_fail:
            raise ConnectionError(f"{provider} 返回 429 错误")
        return {"status": 200, "provider": provider, "content": "调用成功"}

    def route_request(self, force_primary_fail: bool = False) -> Dict[str, Any]:
        # 1. 尝试调用主力模型
        if self.primary_breaker.can_pass():
            try:
                res = self.mock_call_upstream("Primary-LLM", force_primary_fail)
                self.primary_breaker.record_result(success=True)
                return res
            except ConnectionError:
                self.primary_breaker.record_result(success=False)
                # 发生异常立即降级到备用链路
                return self._fallback_route()
        else:
            # 熔断开启中，直接短路到备用模型
            return self._fallback_route()

    def _fallback_route(self) -> Dict[str, Any]:
        return self.mock_call_upstream("Backup-LLM", should_fail=False)

# 验证熔断转移过程
if __name__ == "__main__":
    router = GatewayRouter()
    print("--- 阶段 1: 主力模型连续报错，观察熔断状态跃迁 ---")
    for i in range(5):
        resp = router.route_request(force_primary_fail=True)
        print(f"请求 {i+1}: 响应提供方 -> {resp['provider']}, 熔断器状态: {router.primary_breaker.state}")

    print("\n--- 阶段 2: 熔断器处于 OPEN，直接短路到备用实例 ---")
    resp = router.route_request(force_primary_fail=True)
    print(f"快速拦截: 响应提供方 -> {resp['provider']}, 熔断器状态: {router.primary_breaker.state}")

    print("\n--- 阶段 3: 等待静默期结束，半开试探并恢复 ---")
    time.sleep(2.1)  # 等待超过 recovery_timeout
    print(f"静默期结束，放行试探: {router.primary_breaker.can_pass()}，当前状态: {router.primary_breaker.state}")
    resp = router.route_request(force_primary_fail=False)  # 模拟上游恢复
    print(f"试探结果: 响应提供方 -> {resp['provider']}, 熔断器最终状态: {router.primary_breaker.state}")
```

运行上述脚本可以清晰验证：当主力服务出现连续故障时，调度器不会任由外部请求超时等待，而是将流量快速切向备用节点；在静默时间结束后，再通过试探流量决定是否切回主力服务。
