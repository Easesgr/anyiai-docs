---
title: Agent 框架
icon: mdi:cpu-64-bit
tag:
  - Agent
  - 工作流
  - 多智能体
---

# Agent 框架选型

Agent 框架应帮助你控制状态、工具、失败、人工介入和可观测性。对大多数业务问题，先构建确定性工作流；只有确实需要动态决策时，再把 Agent 放进受约束的步骤。

## 主流选择

### LangGraph

- 官方入口：[LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- 特点：图与状态机编排、持久化、暂停/恢复、人工介入。
- 适合：长任务、需要循环与分支、追求可审计执行轨迹的 Agent。

### OpenAI Agents SDK

- 官方入口：[Agents SDK](https://openai.github.io/openai-agents-python/)
- 特点：围绕 Agent、工具、handoff、guardrail 与 tracing 的轻量 SDK。
- 适合：以 OpenAI 模型和工具生态为主，希望快速实现多 Agent 协作的 Python 项目。

### Microsoft AutoGen

- 官方入口：[AutoGen 文档](https://microsoft.github.io/autogen/)
- 特点：事件驱动的多 Agent 与消息协作抽象。
- 适合：研究型协作模式、需要灵活角色通信或 Microsoft 生态的团队。

### Semantic Kernel

- 官方入口：[Semantic Kernel](https://learn.microsoft.com/semantic-kernel/)
- 特点：插件、函数调用与企业应用集成，支持 .NET、Python、Java。
- 适合：Microsoft 技术栈、希望将 AI 能力嵌入既有企业服务的团队。

### CrewAI

- 官方入口：[CrewAI 文档](https://docs.crewai.com/)
- 特点：角色、任务与团队协作概念直观，上手快。
- 适合：原型、研究和职责边界清晰的协作流程。
- 注意：多角色对话并不天然提升质量；必须有共享状态、验证器和预算上限。

## 什么时候不该用 Agent

- 流程可枚举：用普通代码或工作流更可靠。
- 需要强一致事务：让 Agent 提建议，由确定性服务执行。
- 结果不可自动验证：增加人工审批，不要无限循环“自我反思”。

## 生产最小要求

1. 工具白名单、最小权限、参数校验与沙箱。
2. 每一步存储状态、输入、输出、模型版本和工具结果。
3. 超时、最大步数、token/费用预算和人工接管出口。
4. 独立 verifier：测试、规则、检索引用或人工审核不能由执行 Agent 自证。
5. 用任务成功率、人工介入率、成本和安全事件衡量，不只看演示效果。

## 一个可控 Agent 的状态机

~~~text
接收任务 → 计划 → 调用受限工具 → 观察结果 → 验证
                          ↓失败               ↓通过
                        重试/降级 ← 记录状态 ← 提交结果
                                      ↓
                                   人工接管
~~~

状态应存储在数据库或检查点中，而不是只存在对话历史里。这样任务中断、模型切换或人工审批后才能可靠恢复。

## 工具设计原则

- 一个工具只做一个可验证动作，例如“查询客户信息”，不要设计万能执行器。
- schema 里明确类型、枚举、最大长度和默认值；服务端仍要二次校验。
- 读操作与写操作分开；写操作需要幂等键、审批或 dry-run。
- 工具返回结构化结果和可追溯 ID，不返回无法核验的大段自然语言。

## 多 Agent 什么时候值得

只有在任务可并行、角色有不同专业工具、或“生成者”和“验证者”确实需要隔离时才拆分。否则多 Agent 会增加上下文传递、协调和成本，普通工作流通常更稳。
