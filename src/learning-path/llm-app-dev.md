---
title: 大模型应用开发学习路线
icon: mdi:code-braces
tag:
  - 学习路线
  - 大模型应用
  - LLM 开发
---

# 大模型应用开发学习路线

> 已有 Python 基础，想系统学习 LLM 应用开发，建议 2-3 个月

---

## 阶段 1：API 调用与基础理解（1 周）

### 必学

- [大模型基础](/fundamentals/llm-basics/) - 理解 LLM 的能力边界
- [推理与生成](/fundamentals/inference-and-generation/) - Temperature、Top-P、KV Cache

### 实战

分别调用 OpenAI、Anthropic、通义千问的 API，实现一个最简单的多轮对话机器人，对比不同模型的响应差异。

---

## 阶段 2：Prompt 工程（2 周）

### 必学

- [Prompt 面试题](/interview/prompt-interview/) - 系统学习 Prompt 设计方法

### 实战

做一个"智能客服机器人"：
- 设计 System Prompt 定义角色
- 用 Few-shot 控制输出格式
- 处理拒绝回答、超出范围等边界情况

---

## 阶段 3：Function Calling 与工具调用（2 周）

### 必学

- [Function Calling](/interview/agent-interview/tool-calling/) - 工具调用原理
- [MCP 协议](/frontier/mcp-protocol/) - 标准化工具协议

### 实战

做一个"天气查询 Agent"：
- 设计天气查询的工具 Schema
- 实现 LLM → 工具选择 → 执行 → 整合的完整链路
- 加入错误处理（工具失败时重试）

---

## 阶段 4：上下文工程（2 周）

### 必学

- [上下文工程](/frontier/context-engineering/) - Token 预算、上下文压缩
- [多轮对话](/frontier/multi-turn-dialogue/) - 历史管理策略

### 实战

给上面的客服机器人加上多轮对话能力：
- 设计上下文窗口管理策略
- 实现对话历史的截断和摘要
- 处理 Token 超限时的降级方案

---

## 阶段 5：进阶方向（按兴趣选）

| 方向 | 推荐文章 | 技术栈建议 |
|------|----------|------------|
| RAG | [RAG 面试题](/interview/rag-interview/) | LangChain + Chroma + OpenAI |
| Agent | [Agent 面试题](/interview/agent-interview/) | LangGraph + MCP |
| 工作流 | [AI 工作流](/frontier/ai-workflow/) | Dify / Coze / n8n |
| 本地部署 | [本地大模型](/frontier/local-llm/) | vLLM + Ollama |

---

## 学习节奏建议

| 阶段 | 周期 | 每日投入 |
|------|------|----------|
| API 调用 | 1 周 | 2-3 小时 |
| Prompt 工程 | 2 周 | 2-3 小时 |
| 工具调用 | 2 周 | 3-4 小时 |
| 上下文工程 | 2 周 | 2-3 小时 |
| 进阶方向 | 持续 | 2-3 小时 |

---

## 求职目标

- 大模型应用开发工程师
- AI 产品工程师
- LLM 全栈工程师
