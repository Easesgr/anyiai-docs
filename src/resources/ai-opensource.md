---
title: AI 开源项目
icon: mdi:source-branch
tag:
  - 开源 AI
  - LLM
  - 工具链
---

# 值得关注的 AI 开源项目

开源项目应按“能否解决你的工程问题”来选，而不是只看热度。引入前请检查许可证、发布频率、维护者响应、依赖链、安全公告与是否有可重复的部署方式。

## 模型运行与服务

- [Ollama](https://github.com/ollama/ollama)：本地运行和管理模型，适合开发与演示。
- [vLLM](https://github.com/vllm-project/vllm)：高吞吐 LLM 推理服务，适合 GPU 生产部署。
- [SGLang](https://github.com/sgl-project/sglang)：面向语言模型与 Agent 的高性能推理/编程运行时。
- [Hugging Face Transformers](https://github.com/huggingface/transformers)：模型训练、推理与生态集成的基础库。

## 应用与模型网关

- [LiteLLM](https://github.com/BerriAI/litellm)：多模型统一接口、代理与用量控制。
- [Open WebUI](https://github.com/open-webui/open-webui)：可自托管的 AI 对话界面。
- [Dify](https://github.com/langgenius/dify)：LLM 应用、工作流和知识库平台。
- [n8n](https://github.com/n8n-io/n8n)：自动化工作流，可连接 AI 与业务系统。

## RAG、检索与文档

- [LlamaIndex](https://github.com/run-llama/llama_index)：数据接入、索引和检索编排。
- [Haystack](https://github.com/deepset-ai/haystack)：组件化 RAG 与搜索 Pipeline。
- [Qdrant](https://github.com/qdrant/qdrant)：向量检索和 payload 过滤。
- [Docling](https://github.com/docling-project/docling)：文档转换、版面与表格解析。

## Agent 与观测

- [LangGraph](https://github.com/langchain-ai/langgraph)：有状态 Agent 图编排。
- [AutoGen](https://github.com/microsoft/autogen)：多 Agent 消息协作。
- [Langfuse](https://github.com/langfuse/langfuse)：LLM tracing、评测与提示管理。
- [OpenTelemetry](https://opentelemetry.io/)：跨服务的可观测性标准；AI 应用也应接入统一追踪。

## 使用开源项目的正确姿势

1. 固定版本与镜像，不要在生产环境直接跟随主分支。
2. 从最小可运行示例开始，写下数据、模型、配置和评测集。
3. 检查许可证是否允许你的分发、SaaS 或商业模式。
4. 监控安全公告和依赖漏洞，给升级留出窗口。
5. 为不可维护或不兼容的依赖准备替代方案，避免单点锁定。

## 引入前的评分表

| 维度 | 要问的问题 |
| --- | --- |
| 维护性 | 最近发布是否稳定？Issue 和安全报告是否有人处理？ |
| 可部署性 | 是否有镜像、配置文档、升级与备份方案？ |
| 兼容性 | 与现有模型、数据库、鉴权和云环境能否集成？ |
| 成本 | GPU、存储、日志、运维和人员成本是多少？ |
| 可退出性 | 数据格式、接口和业务逻辑是否可迁移？ |

## 从 PoC 到生产

PoC 阶段只证明“能跑”；生产阶段还要证明“可升级、可监控、可恢复、可审计”。建议为每个核心开源组件保留版本清单、部署脚本、备份演练和替代选项。

## 贡献与反馈

使用开源项目发现问题时，先提供可复现的最小示例、环境版本和脱敏日志。高质量 Issue 或文档修正往往比“不能用”更容易获得维护者响应。
