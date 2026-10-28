---
title: 资源导航
icon: mdi:compass-outline
---

# AI 资源导航

这里收录的是做 AI 应用时最常遇到的工具与资料。每一页都按“它解决什么、适合谁、何时不该选、官方入口在哪里”的方式整理，尽量避免把短期会变的价格、热度和营销数字写成结论。

> 工具与模型变化很快：将本文当作选型起点，实施前务必阅读对应的官方文档、许可证、价格和数据处理政策。

## 开发与部署

- [API 平台](/resources/api-platform.html)：OpenAI、Anthropic、Google、云厂商与模型网关的接入策略。
- [模型平台](/resources/model-platform.html)：模型社区、本地运行器、推理服务框架与云模型目录。
- [向量数据库](/resources/vector-db.html)：pgvector、Qdrant、Milvus、Pinecone 等检索存储方案。
- [RAG 框架](/resources/rag-framework.html)：文档接入、检索、重排与评估的工程框架。
- [Agent 框架](/resources/agent-framework.html)：状态机、工具调用、多 Agent 与人工介入。

## 生产力与集成

- [AI 编程工具](/resources/ai-coding-tools.html)：IDE 助手、终端 Agent、代码审查与安全使用方法。
- [AI 工具](/resources/ai-tools.html)：对话、图像、音视频、文档和自动化工具。
- [MCP Server](/resources/mcp-server.html)：MCP Registry、Server 接入和最小权限原则。
- [AI 开源项目](/resources/ai-opensource.html)：模型服务、应用网关、RAG 与可观测性项目。

## 系统学习

- [技术书籍](/resources/tech-books.html)：从深度学习、NLP/LLM 到 AI 工程的书与课程。
- [论文资源](/resources/papers.html)：Transformer、对齐、RAG、Agent 的阅读路径。

## 一个实用的选型流程

1. 先用真实任务和数据写出验收标准。
2. 选择可替换的接口，不把业务逻辑绑定在单一厂商或框架上。
3. 核对许可证、数据流向、权限、成本与部署地区。
4. 为质量、延迟、失败、成本和安全建立最小评测与观测。
5. 从可回滚的小规模试点开始，再扩大使用范围。

## 选型时的四类约束

| 约束 | 需要澄清的问题 |
| --- | --- |
| 任务质量 | 什么结果算正确？能否准备真实评测集？ |
| 工程条件 | 已有云、数据库、语言与部署能力是什么？ |
| 治理安全 | 数据能否出域？谁有权限？是否要审计与审批？ |
| 成本规模 | 日请求量、峰值并发、上下文长度和人工复核成本是多少？ |

没有这些约束就直接比较工具，通常只会得到一份无法执行的“热门产品名单”。
