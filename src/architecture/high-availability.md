---
title: 外部模型服务中断时，大模型系统如何做容灾转移与多级降级？
icon: mdi:server
description: 解析外部大模型在容量不足、网络挂起和配额耗尽时的故障转移拓扑、主动健康探测状态机、多级降级阶梯及合规边界。
---

在大模型落地到生产业务的过程中，系统可用性的最大变数往往不在内部代码，而在于外部模型供应商的稳定性。传统微服务架构的高可用依赖内部集群横向扩容与多可用区副本，而大模型系统深度依赖第三方推理接口，面临黑盒网络抖动、突发限流、单区域容量告急以及跨洲专线中断等不可控风险。

本文以一个线上智能客服系统为例，梳理请求从网关分发到不同云服务商端点的完整数据流，分析故障发生时的转移逻辑与降级矩阵。如果需要了解单次请求内部的超时判定、网络重试与局部熔断状态机，可参考 [模型网关](/architecture/model-gateway.md)；如果需要了解模型能力画像与意图分流，可参考 [多模型路由](/architecture/multi-model-routing.md)。

![高可用架构与容灾故障转移全景](/images/content-rebuild/c74370136beabbab/cover.png)

## 1. 外部大模型调用的三大典型故障模式

大模型应用的可用性链路比传统 Web 服务更脆弱。模型服务是典型的长耗时、高算力密集型计算，其可用性瓶颈通常表现在三个方面：

1. **服务商容量紧张与服务不可用**：大部分公有云模型服务商承诺的月度可用性（SLA）通常在 99.5% 至 99.9% 之间。这意味着每个月可能存在数十分钟到数小时的不可用窗口。在底层 GPU 算力集群突发重调度或算力紧缺时，供应商常直接返回 HTTP 503（Service Unavailable）或 HTTP 500。
2. **流式传输长尾挂起（Hanging）**：流式生成（SSE / Chunked Transfer）可能在首字延迟（TTFT）正常返回后，在生成过程中网络连接无预警中断，或者传输速率断崖式下跌，导致下游服务的长连接一直占满、工作协程池耗尽。
3. **配额突发击穿（Rate Limit Exhaustion）**：模型调用通常受到每分钟请求数（RPM）和每分钟 Token 数（TPM）限制。当业务流量突增或上游出现重试风暴时，供应商会返回 HTTP 429。若系统没有多通道逃逸机制，业务将被强行阻断。

![外部大模型调用的三大典型故障模式](/images/content-rebuild/c74370136beabbab/fig-1.png)

在没有容灾隔离的系统中，单点故障容易演变为级联雪崩：单个外部端点响应变慢，导致网关工作线程被长时间阻塞；上游客户端发起重试，进一步加剧连接池排队；最终整个系统资源耗尽，非 AI 的常规业务接口也一同受到拖累。

<ReadMoreLock />

## 2. 跨云多区域冗余拓扑与流量调度

为了避免单一供应商或单一地域的数据中心故障导致业务中断，高可用架构通常采用跨供应商、跨地域的冗余部署拓扑。

在接入层，用户请求经由网关集群接入。应用网关自身无状态化部署，下游的模型调度网关则维护横跨多家模型供应商和多个地域的端点池。

![跨云跨区域多活拓扑与调度机制](/images/content-rebuild/c74370136beabbab/fig-2.png)

### 流量分配与两项系统代价

在多区域拓扑中，流量调度策略主要包括：

* **主备模式（Active-Passive）**：主区域承载主要流量，备用区域保持低频心跳待命。主区域故障时，调度层进行切换。
* **动态多活模式（Active-Active with Weights）**：流量按预设健康权重（如 70%:30%）同时分配给主备供应商，以便实时验证各通道可用性，并在某一通道指标劣化时动态调整权重。

这种拓扑需要付出两项系统开销：

1. **上下文状态的跨区同步**：存储于集中缓存中的多轮会话历史需要跨区同步，否则切换区域后模型将丢失多轮对话记忆，详见 [上下文管理](/architecture/context-management.md)。
2. **物理网络延迟增加**：从本地机房切往跨大洲节点，网络往返时间（RTT）必然增加，必须在路由决策中计入这一物理成本。

## 3. 主动心跳探测与平滑状态迁移

故障转移不能仅依赖用户真实请求报错后的被动触发。如果仅靠被动触发，在故障发生的初始阶段，必然有一批真实用户的请求充当“探针”而失败。系统需要结合带外主动探测（Synthetic Probing）与带内请求指标统计。

### 探测策略与状态机设计

主动探测器每隔固定周期（例如 5 秒）向各供应商的探测端点或实际推理接口发送最小化测试请求（例如输入极短 Prompt 并限制 `max_tokens=1`），以较低成本验证整个模型调度与推理通路是否通畅，而不仅是检查 HTTP 反向代理层。

调度层内部维护每个端点的状态机：

* **Healthy（健康态）**：端点成功率处于阈值之上，延迟在正常区间，承接标准配额流量。
* **Degraded（降级态）**：检测到 P99 延迟明显飙升，或开始出现零星 429/503。此时网关降低其分流权重，并触发告警。
* **Unhealthy（离线/熔断态）**：连续多次探测超时或报错，系统将该端点移出活跃路由池，所有新流量切换至备用厂商端点。
* **Probing & Slow Start（试探与慢启动回切）**：当离线端点连续多次探测成功后，进入恢复试探期。网关先分配小比例（如 5%）的流量进行验证，观察端到端指标稳定后，再逐步提高流量权重，避免刚恢复的服务端被瞬间冲垮。

![主动健康探测与状态迁移示意图](/images/content-rebuild/c74370136beabbab/fig-3.png)

## 4. 多级系统降级阶梯与确定性契约

跨云备用模型能缓解单一服务商故障，但当遇到全网算力紧张、骨干网抖动或所有外部公有云端点均受阻的极端场景时，系统需要具备多级降级阶梯（Graceful Degradation），保证核心业务契约不崩溃。

降级阶梯的设计原则是：**逐步减少非核心特性与算力消耗，保留核心功能，保障输出符合下游系统的格式契约。**

| 阶梯等级 | 系统状态 | 模型与依赖组件 | 业务行为变化 | 触发条件 |
| :--- | :--- | :--- | :--- | :--- |
| **L0（标准态）** | 全链路正常 | 旗舰大模型 + 全量多路召回 RAG + 联网检索 | 深度推理、长文本生成、格式高度精细化 | 系统健康指标正常，预算与配额充足 |
| **L1（轻量态）** | 主力模型故障或拥堵 | 快速小模型 + 精简 RAG 上下文 | 回答趋向简明，仅保留核心召回文档（减少 Top-K），维持基础对话 | 主模型 5xx 突增或首字延迟超过预设阈值 |
| **L2（离线态）** | 外部公网模型全断 | 本地部署开源小模型或语义缓存 | 依赖离线预计算知识库或高频缓存问答，响应确定性提升但泛化能力减弱 | 外部网络阻断或所有第三方 API 触发熔断 |
| **L3（兜底态）** | 算力完全不可用 | 确定性规则引擎 / 静态结构化模板 | 停止模型推理，返回预置格式化的业务引导信息 | 所有模型计算引擎宕机，触发终极兜底 |

![四级系统降级阶梯与状态矩阵](/images/content-rebuild/c74370136beabbab/fig-4.png)

### 保障下游数据契约不被破坏

在前端界面或下游系统中，往往依赖固定的 JSON Schema。如果降级后直接向调用方抛出未处理的 HTTP 500，会导致前端页面异常或调用方崩溃。降级模块必须输出符合预期 Schema 的结构化数据。

下面是一个展示请求链路上主备切换与降级兜底的网关调用调度器示例：

```python
import asyncio
from typing import Dict, Any, Optional

class ResilienceGateway:
    def __init__(self, simulate_primary_failure: bool = False):
        self.providers = ["provider_primary", "provider_backup"]
        self.simulate_primary_failure = simulate_primary_failure
        # 维护端点健康状态与连续失败次数
        self.health_status: Dict[str, bool] = {"provider_primary": True, "provider_backup": True}
        self.consecutive_failures: Dict[str, int] = {"provider_primary": 0, "provider_backup": 0}

    async def _mock_remote_call(self, provider: str, prompt: str, timeout: float) -> Optional[Dict[str, Any]]:
        """模拟远程模型 API 调用"""
        if not self.health_status[provider]:
            return None
        try:
            # 模拟网络调用延迟与故障
            if provider == "provider_primary" and self.simulate_primary_failure:
                await asyncio.sleep(timeout + 0.1)  # 模拟主模型超时
            else:
                await asyncio.sleep(0.05)  # 模拟正常网络往返
            return {"status": "success", "provider": provider, "content": f"[{provider}] 回复: {prompt}"}
        except Exception:
            return None

    async def execute_request(self, prompt: str) -> Dict[str, Any]:
        # 1. 尝试 L0: 主模型提供商
        result = await self._call_with_fallback("provider_primary", prompt, timeout_sec=0.5)
        if result:
            return result

        # 2. 尝试 L1: 备用模型提供商
        result = await self._call_with_fallback("provider_backup", prompt, timeout_sec=0.5)
        if result:
            result["degraded_level"] = "L1"
            return result

        # 3. 触发 L3: 结构化兜底模板（保障下游 JSON 契约不破坏）
        return {
            "status": "fallback",
            "provider": "static_rule_engine",
            "degraded_level": "L3",
            "content": "当前系统问答通道繁忙，已转入预置引导流程。",
            "data": {"retry_recommended": True, "backoff_seconds": 5}
        }

    async def _call_with_fallback(self, provider: str, prompt: str, timeout_sec: float) -> Optional[Dict[str, Any]]:
        try:
            task = asyncio.create_task(self._mock_remote_call(provider, prompt, timeout_sec))
            res = await asyncio.wait_for(task, timeout=timeout_sec)
            if res:
                self.consecutive_failures[provider] = 0
                return res
        except (asyncio.TimeoutError, Exception):
            self.consecutive_failures[provider] += 1
            if self.consecutive_failures[provider] >= 3:
                self.health_status[provider] = False  # 连续 3 次失败触发熔断
        return None
```

## 5. 跨区域故障转移的权衡与合规边界

容灾设计不是无代价的，自动切换需要考虑物理延迟与合规红线。

![跨区域故障转移中的延迟与数据主权约束](/images/content-rebuild/c74370136beabbab/fig-5.png)

### 物理网络延迟的算例分析

假设智能客服系统的终端用户位于新加坡，主用节点部署在新加坡本地机房，备用容灾节点部署在美东机房。以下为一个典型的跨洲切换网络延迟变化示意（仅作网络往返耗时对比）：

* **正常路径（新加坡终端 -> 新加坡网关 -> 新加坡本地模型端点）**：
  * 终端至网关 RTT：约 $15\text{ ms}$
  * 网关至模型服务 RTT：约 $5\text{ ms}$
  * 基础网络往返耗时：$15 + 5 = 20\text{ ms}$
* **跨洋容灾路径（新加坡终端 -> 新加坡网关 -> 美东模型端点）**：
  * 终端至网关 RTT：约 $15\text{ ms}$
  * 网关跨洲调度到美东端点的物理光缆 RTT：约 $180\text{ ms}$
  * 基础网络往返耗时：$15 + 180 = 195\text{ ms}$

在流式生成场景下，由于网络往返时间成倍增加，首字延迟（TTFT）与后续每个数据块的到达抖动会更容易被用户感知。因此，备用区域选型应优先遵循地理就近原则，跨大洲节点通常作为多重故障后的次选方案。

### 数据主权与隐私合规约束

比网络延迟更严苛的是法律合规要求。例如在欧盟通用数据保护条例（GDPR）适用场景下，涉及个人身份信息（PII）以及用户输入文本的跨境传输受严格监管。当系统将欧洲用户的请求从法兰克福节点自动故障转移至未通过充分性认定（Adequacy Decision）的境外节点时，若未签署标准合同条款（SCC）或未取得合规授权，可能面临合规处罚。

因此，**高可用切换策略必须受到路由合规策略的硬性约束**。系统在将流量切往备选区域前，需要对租户与请求标签进行校验；对于被数据合规策略锁定的数据，应优先在本地触发 L2 离线模型或 L3 规则降级，而不是直接跨境转移。

## 6. 混沌工程注入与可观测性验收

高可用架构需要通过主动故障注入进行演练与验收。

![混沌工程故障注入与指标验收流程](/images/content-rebuild/c74370136beabbab/fig-6.png)

系统可以在预发或演练环境中，针对模型调用的脆弱环节进行周期性注入测试，具体验收维度如下：

### 1. 供应商 500/503 异常注入
* **注入方式**：在代理层拦截主模型请求，模拟返回 HTTP 500 或 HTTP 503。
* **验收指标**：
  * 故障转移时间（RTO）：系统在预设阈值窗口（例如连续 3 次探测失败或 500 毫秒内）完成备用通道切换。
  * 错误漏出率：切换过程中，对外暴露的未捕获异常比例保持在允许阈值内，下游收到的响应结构满足 Schema 契约。

### 2. 首字长尾悬挂注入（TTFT Hanging）
* **注入方式**：建立 TCP 连接并返回 HTTP 200，但延迟发送第一个 SSE 数据块（如注入 20 秒延迟）。
* **验收指标**：
  * 网关的首字超时判定生效，主动切断悬挂连接并释放连接池资源。
  * 客户端在设定的超时时间内收到备用流式通道的响应或明确的降级提示。

### 3. 限流状态码注入（HTTP 429）
* **注入方式**：模拟返回 HTTP 429 及 `Retry-After` 头。
* **验收指标**：
  * 网关正确提取退避时间参数，主动暂停向该端点分发新流量，将请求平滑引流至备用池。
  * 避免网关发起无退避的连续重试引发流量风暴。

在演练全过程中，所有切换决策和降级行为均应记录在分布式追踪轨迹中，便于事后复盘与健康度度量。链路追踪与指标打标的具体实现，可参考 [日志与 Trace](/architecture/logging-trace.md)。
