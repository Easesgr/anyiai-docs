---
title: Agent 工程师学习路线
icon: mdi:robot
tag:
  - 学习路线
  - Agent
  - 智能体
---

# Agent 工程师学习路线

> 已有 LLM 应用基础，专攻 Agent 方向，建议 3-4 个月

## 阶段 1：核心概念（1-2 周）

### 必学

- [Agent 面试题](/interview/agent-interview/) - 系统学习 Agent 全貌
- ReAct、Plan-and-Execute 工作模式
- Agent Loop（规划 → 行动 → 观察 → 反思）

## 阶段 2：工具调用基础（2 周）

### 必学

- Function Calling / Tool Calling
- 不同 LLM Provider 的 schema 差异

### 实战

做一个「带工具调用的聊天机器人」：
- 设计 3-5 个工具（搜索、计算、文件读取等）
- 实现完整的 ReAct 循环
- 加入错误处理（工具失败时重试）

## 阶段 3：MCP 协议（1-2 周）

### 必学

- [MCP 协议](/frontier/mcp-protocol/) - Model Context Protocol
- MCP Server 开发

### 实战

为上面的 Agent 接入一个 MCP Server（如文件系统 MCP、GitHub MCP）

## 阶段 4：记忆系统（2 周）

### 必学

- [Agent 记忆系统](/frontier/agent-memory-system/) - 短期/长期记忆
- 向量数据库作为长期记忆
- 记忆压缩与摘要

### 实战

给 Agent 加入：
- 短期对话记忆
- 长期用户偏好记忆
- 跨会话记忆持久化

## 阶段 5：进阶能力（按兴趣选）

### 方向 A：多 Agent 协作

- [多智能体协作](/frontier/agent-skills/) - CrewAI / LangGraph
- Agent 间通信协议（A2A）

### 方向 B：Agent 工作流

- [AI 工作流](/frontier/ai-workflow/) - n8n / Dify / Coze
- [工作流编排](/frontier/workflow-orchestration/) - 复杂任务编排

### 方向 C：Agent 安全

- 工具权限控制
- Prompt Injection 防御
- 操作审计

## 阶段 6：项目实战（2-3 周）

### 推荐项目

- **AI 编程助手** - 用 Agent 实现代码生成/调试
- **研究助手** - 能搜索网络、阅读论文、生成报告的 Agent
- **自动化客服** - 处理退款、查单等多步骤任务

## 推荐开源贡献

- LangChain / LangGraph
- AutoGen
- CrewAI
- MCP Servers

## 求职目标

- Agent 应用工程师
- AI 自动化工程师
- 智能体平台工程师
