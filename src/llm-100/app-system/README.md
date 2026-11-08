---
title: 如何把单次模型调用做成可靠的业务系统？大模型应用系统篇导引
icon: mdi:server
description: 系统化梳理大模型从API调用走向生产系统的工程框架，涵盖长连接流式通信、上下文预算控制与垂直落地实践。
---

![大模型应用系统全局工程全景](/images/content-rebuild/cc425969efab26b2/cover.png)

在很多原型开发中，接入大模型通常只需要几行代码：调用模型提供商的 `chat/completions` 接口，等待文本返回，然后渲染在屏幕上。但在真实的线上生产环境中，这种朴素做法很快会遇到三类阻碍：

1. **通信阻断**：传统接口响应通常在几十到几百毫秒，而大模型生成一段完整回答可能需要 3 到 30 秒。如果采用同步阻塞调用，网关反向代理极易超时断开，用户也会面对长时间白屏。
2. **状态漂移与费用失控**：大模型服务本身是无状态的（Stateless），每次调用不会记住上一句话。为了维持多轮对话，应用必须把历史记录完整打包发送。如果会话持续数十轮，请求体包含的 Token 数量将线性增加，既容易超出模型的上下文窗口上限，又会导致调用费用成倍攀升。
3. **业务逻辑无法闭环**：真实应用需要处理鉴权、租户隔离、重复提问缓存、企业数据权限校验以及多阶段业务流程，这些逻辑无法仅靠一个提示词（Prompt）自动保证。

本模块的核心目标，就是带你把一个单次调用的补全接口，逐步封装成具备高可用、成本可控且状态清晰的完整软件系统。

我们以一个贯穿全文的实际场景为例：**“在线技术支持工单助手”**。当用户提出技术问题时，系统既要即时给出打字机般的解答，又要参考用户的历史设备状态与以往追问，还要避免对相同故障反复调用模型付费。

---

## 通信与网关链路：应对长耗时生成

以“技术支持助手”为例，如果模型需要生成 800 字的故障排查步骤，完整生成时间约为 10 秒。如果系统等待 10 秒后一次性返回，用户体验会极度卡顿，并且中间任何网络抖动都会导致整个请求失败重来。

![长响应与流式输出的双向通信链路分解](/images/content-rebuild/cc425969efab26b2/fig-1.png)

为了保障长耗时请求的可用性，系统需要对通信与网关做分层改造：

### 1. 架构分层与密钥隔离
业务前端严禁直接保存或暴露模型厂商的 API Key。所有请求必须经过业务后端网关：
- 网关负责校验用户身份与接口调用配额。
- 模型厂商返回异常或限流（如 HTTP 429）时，网关负责执行重试退避策略，对前端屏蔽供应商报错细节。
- 详细的分层边界与组件分工，请参考 [/llm-100/app-system/llm-app-architecture.md](/llm-100/app-system/llm-app-architecture.md)。

### 2. 前后端与模型的三方协同
前端负责渲染与采集交互事件，后端维护业务实体（用户身份、当前工单 ID、权限等级），模型仅负责针对给定的提示词生成补全。明确各层职责可以避免将业务校验逻辑错误地寄托在提示词约束上。完整的数据时序交互见 [/llm-100/app-system/frontend-backend-model.md](/llm-100/app-system/frontend-backend-model.md)。

### 3. 生命周期监控与客户端中断（Client Abort）
在长达数十秒的流式传输期间，用户可能发现提问有误并点击“停止生成”，或者直接关闭网页。如果后端没有处理中断信号，模型提供商仍可能持续生成并计费：
- 系统必须建立从客户端通过反向代理、业务后端一直传递到模型调用的取消机制（如基于 HTTP 的连接断开信号）。
- 监控指标需从传统的“总耗时”拆解为首字延迟（TTFT，Time to First Token，从发起请求到收到首个内容块的耗时）与输出阶段的吞吐速度（每秒生成的 Token 数）。生命周期具体设计见 [/llm-100/app-system/model-api-request-lifecycle.md](/llm-100/app-system/model-api-request-lifecycle.md)。

### 4. 传输通道选型：SSE 与 WebSocket
对于单向输出文本的大模型对话场景，服务器发送事件（SSE，Server-Sent Events）基于标准 HTTP 协议，具备天然的文本分块传输优势、较好的反向代理穿透性与原生断线重连支持；而全双工的 WebSocket 更适合需要客户端高频双向传输音频或交互指令的场景。选型权衡与常见 Nginx 缓冲配置踩坑见 [/llm-100/app-system/sse-vs-websocket.md](/llm-100/app-system/sse-vs-websocket.md)。

---

## 状态与资源控制：无状态模型的上下文管理

大模型在推理完成后不保留会话内存。在工单助手中，用户在第 5 轮提问“上面那步做完还是报错，怎么办？”，模型如果看不到前 4 轮的排查记录，就无法理解“上面那步”指代什么。

但这并不意味着我们可以无限制地将所有历史记录一并附带。

![提示词组装与上下文成本控制漏斗](/images/content-rebuild/cc425969efab26b2/fig-2.png)

系统需要建立漏斗式的组装与裁剪机制：

### 1. 结构化模板组装
提示词不应由随意的手写字符串拼接而成，否则极易发生参数混淆甚至提示词注入攻击。系统应使用模板引擎将系统规则（System Prompt）、业务变量（如用户的操作系统版本）和用户输入严格分层渲染，具体实现见 [/llm-100/app-system/prompt-template-system.md](/llm-100/app-system/prompt-template-system.md)。

### 2. 对话历史存储与隔离
多轮对话的每条消息（包括 `user`、`assistant` 以及工具调用记录）都需要持久化到数据库中，并与具体业务会话绑定。如何设计高并发读写下的持久化方案，参见 [/llm-100/app-system/conversation-history.md](/llm-100/app-system/conversation-history.md)。

### 3. Token 预算与动态裁剪
当一次对话持续数十轮时，全量历史会消耗大量 Token，甚至击穿上下文窗口限制：
- **固定滑动窗口**：只保留最近 $K$ 轮对话。
- **动态 Token 预算**：为历史记录分配固定的 Token 上限（例如 2000 Tokens），超出部分丢弃较早轮次，或触发模型对早前对话做压缩摘要（Summary）。
- 算法与预算评估体系详见 [/llm-100/app-system/context-budget.md](/llm-100/app-system/context-budget.md)。对于长期跨会话记忆，还可以参考 [/frontier/agent-memory-system/README.md](/frontier/agent-memory-system/README.md)。

### 4. 降低重复开销的双层缓存
在技术支持场景中，“如何重置密码”这类高频问题占用了大量重复调用。系统可以通过精准哈希缓存与向量语义缓存拦截重复调用，直接返回经审核的标准答案。但需注意适用边界：
- **精准哈希缓存**：不能仅拿提问文本做哈希，键值必须纳入提示词版本、模型版本、参数与租户权限，并设定时效策略；哈希匹配不等于答案对所有上下文都适用。
- **语义缓存**：主要适用于无上下文依赖的单轮高频问答；一旦问题依赖多轮对话历史或动态外部数据，盲目命中语义相似项会导致答非所问。
多级缓存的设计与失效策略参见 [/llm-100/app-system/llm-application-caching.md](/llm-100/app-system/llm-application-caching.md)。

---

## 垂直业务落地：两种典型组织形态

掌握了通信控制与状态预算后，如何将这些模块组合成特定业务形态？本模块提供两类具代表性的系统架构样例：

![垂直业务系统的两种典型架构演化形态](/images/content-rebuild/cc425969efab26b2/fig-3.png)

### 形态 A：状态机主导型系统（以 AI 面试助手为例）
交互流程由严格的业务阶段驱动（如：自我介绍 -> 专业技术考核 -> 场景设计 -> 反问环节）：
- 模型不决定流程走向，而是由应用内的**有限状态机（FSM）**判定当前阶段是否完成。
- 系统在每个节点提取候选人回答要点、调用打分函数，并由状态机决定何时切换到下一话题。
- 具体实践与防刷/超时设计见 [/llm-100/app-system/ai-interview-assistant.md](/llm-100/app-system/ai-interview-assistant.md)。

### 形态 B：数据检索主导型系统（以 企业知识库问答为例）
交互流程由外部数据的动态召回驱动：
- 用户提问后，系统先在企业私有文档中检索出相关片段，再拼入提示词让模型基于文档回答。
- 核心难点在于多租户权限隔离：普通员工不能召回属于管理层文档的分块。系统需在数据库检索层注入权限过滤条件，而不能寄希望于让模型“自觉保密”。
- 详细落地流程见 [/llm-100/app-system/enterprise-knowledge-qa.md](/llm-100/app-system/enterprise-knowledge-qa.md)。

关于更高层级的宏观架构权衡，还可以延展阅读 [/architecture/ai-system-design.md](/architecture/ai-system-design.md)。

---

## 推荐学习路线

建议按以下顺序系统性学习本模块的 10 篇内容：

1. [/llm-100/app-system/llm-app-architecture.md](/llm-100/app-system/llm-app-architecture.md)（建立全景分层与网关分工概念）
2. [/llm-100/app-system/frontend-backend-model.md](/llm-100/app-system/frontend-backend-model.md)（厘清端、云、模型三者的数据流边界）
3. [/llm-100/app-system/model-api-request-lifecycle.md](/llm-100/app-system/model-api-request-lifecycle.md)（掌握异步响应、超时策略与取消机制）
4. [/llm-100/app-system/sse-vs-websocket.md](/llm-100/app-system/sse-vs-websocket.md)（实现标准 SSE 流式推送通道并规避代理缓冲）
5. [/llm-100/app-system/prompt-template-system.md](/llm-100/app-system/prompt-template-system.md)（构建安全、可参数化的结构化提示词）
6. [/llm-100/app-system/conversation-history.md](/llm-100/app-system/conversation-history.md)（实现持久化会话存储）
7. [/llm-100/app-system/context-budget.md](/llm-100/app-system/context-budget.md)（实现动态滑动窗口与 Token 预算裁剪机制）
8. [/llm-100/app-system/llm-application-caching.md](/llm-100/app-system/llm-application-caching.md)（搭建精确与语义两级缓存降本）
9. [/llm-100/app-system/ai-interview-assistant.md](/llm-100/app-system/ai-interview-assistant.md)（综合应用：状态机驱动的交互系统）
10. [/llm-100/app-system/enterprise-knowledge-qa.md](/llm-100/app-system/enterprise-knowledge-qa.md)（综合应用：结合权限过滤的知识问答系统）

---

## 可检验练习：实现滑动窗口上下文预算管理器

在学习后续深入文章前，你可以通过编写一个最小可运行的 Python 模块，验证“动态 Token 预算管理”的基本原理。

### 目标要求
编写一个 `ContextBudgetManager` 类，满足：
1. 固定保留一条 `system` 角色设定消息。
2. 接收包含多轮 `user` 与 `assistant` 的历史对话列表。
3. 在不超过最大 Token 预算（例如设定为 60 字符/Token 示意）的前提下，优先保留最新的消息，从最早的历史消息开始剔除。
4. 若单条最新用户提问本身即超出总预算，应抛出显式截断或错误告警，而不是产生破坏格式的请求。

### 参考验证代码

```python
from typing import List, Dict


def mock_tokenize(text: str) -> int:
    """教学示意：以简单的空白分词或字符近似模拟 Token 计算。"""
    return len(text.strip().split()) if " " in text else len(text)


class ContextBudgetManager:
    def __init__(self, system_prompt: str, max_budget_tokens: int):
        self.system_message = {"role": "system", "content": system_prompt}
        self.system_tokens = mock_tokenize(system_prompt)
        self.max_budget_tokens = max_budget_tokens

        if self.system_tokens >= max_budget_tokens:
            raise ValueError("System prompt 自身已超过最大预算上限")

    def fit_context(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        从后向前保留最新消息，确保总 Token 数 <= max_budget_tokens。
        始终包含 system_message 在第一位。
        """
        available_tokens = self.max_budget_tokens - self.system_tokens
        retained_history: List[Dict[str, str]] = []

        # 若历史非空，优先检查最新单条消息是否已直接超出可用预算
        if history and mock_tokenize(history[-1]["content"]) > available_tokens:
            raise ValueError("单条最新提问已超出剩余上下文预算上限")

        # 从最新消息往历史消息逆向遍历
        current_tokens = 0
        for msg in reversed(history):
            msg_tokens = mock_tokenize(msg["content"])
            if current_tokens + msg_tokens <= available_tokens:
                retained_history.append(msg)
                current_tokens += msg_tokens
            else:
                # 空间不足，更早的历史消息被淘汰
                break

        # 逆序翻转回正序，并拼在 system_message 之后
        retained_history.reverse()
        return [self.system_message] + retained_history


# 验证用例
if __name__ == "__main__":
    manager = ContextBudgetManager("你是一个有用的技术客服。", max_budget_tokens=30)

    history = [
        {"role": "user", "content": "我的机器无法开机。"},             # 9 字符
        {"role": "assistant", "content": "请检查电源线是否插紧。"},    # 11 字符
        {"role": "user", "content": "电源线插紧了，指示灯还是不亮。"},  # 15 字符
    ]

    # system 消息为 12 字符，剩余可用预算为 18 字符
    # 只有最后一条用户消息（15 字符）能装入，早前两条将被裁剪
    fitted = manager.fit_context(history)
    print("保留的消息条数:", len(fitted))
    print("保留的内容:", [m["content"] for m in fitted])
    assert len(fitted) == 2
    assert fitted[1]["content"] == "电源线插紧了，指示灯还是不亮。"
    print("预算裁剪逻辑验证通过！")
```

完成本练习并确认代码逻辑后，你即可进入第一讲，开始具体拆解大模型应用的整体架构分层。
