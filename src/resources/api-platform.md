---
title: API 平台
icon: mdi:power-plug-outline
tag:
  - 大模型 API
  - 模型网关
  - 开发平台
---

# 大模型 API 平台选型

大模型 API 的选择不该只看单次调用价格。更重要的是：模型能力是否匹配任务、区域与合规要求、工具调用与结构化输出是否稳定、限流与可观测性是否满足上线需要。

> 价格、模型名和额度变化很快。本文不固化报价；接入前请以各平台的 Pricing 与模型文档为准。

## 先按场景选

| 场景 | 优先关注 | 常见选择 |
| --- | --- | --- |
| 通用对话、代码、Agent | 工具调用、长上下文、可靠性 | OpenAI、Anthropic、Google |
| 多模态与 Google 生态 | 图像/视频理解、Workspace、Vertex AI | Gemini API / Vertex AI |
| Azure 企业环境 | 身份、网络隔离、区域治理 | Azure OpenAI |
| 低成本批量推理或自托管 | 开放权重、吞吐与部署控制 | DeepSeek、Mistral、Together、Fireworks、vLLM |
| 中国大陆产品与合规接入 | 中文能力、备案、国内网络 | 阿里云百炼、火山方舟、腾讯云、百度智能云、智谱 |

## 国际主流平台

### OpenAI API

- 官方入口：[平台与文档](https://platform.openai.com/docs)
- 强项：通用模型、Responses API、结构化输出、函数/工具调用、实时与多模态能力。
- 适合：希望快速做出产品、使用 OpenAI 生态工具或需要统一 API 体验的团队。
- 注意：先确认目标模型的可用区域、数据保留策略、速率限制和批处理方式。

### Anthropic API

- 官方入口：[Claude API 文档](https://docs.anthropic.com/)
- 强项：长文档理解、复杂推理、代码任务、工具使用与提示缓存。
- 适合：研究分析、企业文档处理、代码 Agent 等对上下文质量要求高的场景。
- 注意：将系统提示、工具定义和大段固定资料设计为可缓存内容，通常比单纯压缩 Prompt 更有效。

### Google Gemini API / Vertex AI

- 官方入口：[Gemini API](https://ai.google.dev/gemini-api/docs) · [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs)
- 强项：原生多模态、超长上下文、Google Cloud 的企业治理能力。
- 适合：图文档案、视频理解、已在 GCP 上运行的企业应用。
- 注意：Gemini Developer API 偏开发者体验；需要私网、审计、IAM 与生产治理时优先评估 Vertex AI。

### Azure OpenAI

- 官方入口：[Azure OpenAI 文档](https://learn.microsoft.com/azure/ai-services/openai/)
- 强项：Azure 身份权限、私有网络、区域部署和企业采购体系。
- 适合：已有 Azure 基础设施、对数据边界和企业治理有明确要求的团队。
- 注意：模型版本、区域容量和 API 版本由 Azure 发布节奏决定，不能假设与 OpenAI 平台完全同步。

### Mistral AI

- 官方入口：[Mistral 文档](https://docs.mistral.ai/)
- 强项：欧洲厂商、开放权重与托管 API 并行，便于在性能、成本与自主部署之间折中。
- 适合：希望保留模型可替换性，或需要欧洲数据主权方案的团队。

### DeepSeek 与国内云模型平台

- 官方入口：[DeepSeek 平台](https://platform.deepseek.com/) · [阿里云百炼](https://www.aliyun.com/product/bailian) · [火山方舟](https://www.volcengine.com/product/ark)
- 强项：中文任务、国内网络连通性、国内云服务与行业能力。
- 适合：面向中国大陆用户、需要国内发票与合规支持的产品。
- 注意：将模型调用封装在自己的 Provider 层，避免业务代码绑定某一家 SDK。

## 网关与聚合层

模型网关不是“多接几个 API”而已。生产环境至少应统一：密钥管理、超时与重试、模型路由、用量归因、日志脱敏、预算限制与故障降级。

- [LiteLLM](https://docs.litellm.ai/)：开源代理与 SDK，适合统一多 Provider 的 OpenAI 兼容接口。
- [OpenRouter](https://openrouter.ai/docs)：适合原型阶段快速试用多个模型；上线前仍应评估数据路径与供应商治理。
- 云厂商网关：AWS Bedrock、Vertex AI、Azure AI Foundry 更适合已有对应云治理体系的组织。

## 上线前检查清单

1. 用真实任务集比较质量、延迟、失败率和总成本，而不是只看公开榜单。
2. 为每个模型设置超时、重试上限、熔断与降级模型。
3. 不在 Prompt、日志或前端暴露密钥与原始隐私数据。
4. 对工具调用和结构化输出做 schema 校验；模型输出永远是不可信输入。
5. 记录请求版本、模型版本、Prompt 模板与评估结果，保证问题可复现。

## 推荐起步组合

- 个人或原型：选一家主 API + LiteLLM，先把评估和日志做起来。
- 企业内部知识库：云厂商模型服务 + 私网/权限治理 + 可观测性平台。
- 高可用产品：至少准备主模型、备用模型和本地/开源模型三层降级策略。

## 接入架构：不要让业务代码直连供应商 SDK

~~~text
业务服务 → 自有 Model Gateway → 路由/缓存/配额/审计 → 多个模型 Provider
                                      ├─ 主模型：高质量
                                      ├─ 备用模型：高可用
                                      └─ 小模型：分类、抽取、兜底
~~~

网关层应把业务请求转换成内部统一协议，并记录 tenant、任务类型、模型版本、输入输出 token、耗时与错误码。这样替换模型时只改路由，而不是改遍业务服务。

## 模型路由的实用规则

| 任务 | 优先模型能力 | 路由建议 |
| --- | --- | --- |
| JSON 抽取 | 结构化输出、低延迟 | 小模型优先，schema 校验失败再重试 |
| 复杂分析 | 推理、长上下文 | 高能力模型，限制最大上下文 |
| 客服问答 | 检索引用、稳定格式 | RAG + 中等模型，低置信度转人工 |
| 图片/文档理解 | 多模态输入 | 先压缩与分页，再调用视觉模型 |
| 批量离线任务 | 吞吐、价格 | Batch API 或自托管推理队列 |

## 成本不能只按“每百万 token”算

真实成本 = 输入 token + 输出 token + 缓存未命中 + 重试 + 检索/重排 + 人工复核 + 失败请求。建议按业务功能建立预算，例如“每份简历解析不超过 X 元”“每个客服会话不超过 Y 次模型调用”，超过预算立即切换轻量路径或人工队列。

## 可复制的上线测试

准备 50～200 条真实脱敏样本，至少覆盖正常、模糊、超长、恶意输入和工具失败。每次切换模型或 Prompt 后，比较：

- 任务成功率与格式通过率；
- P50/P95 首 token 时间、总耗时与超时率；
- 单任务成本与重试次数；
- 安全拒答、引用正确性和人工返工率。
