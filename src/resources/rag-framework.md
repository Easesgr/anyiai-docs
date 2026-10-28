---
title: RAG 框架
icon: mdi:database
tag:
  - RAG
  - 检索增强生成
  - 开发框架
---

# RAG 框架选型指南

RAG 框架不是向量数据库，也不是一键生成问答机器人的 SaaS。它的职责是把数据接入、解析、切分、索引、检索、重排、上下文组装、生成、引用与评测串成一条可维护的链路。

不同框架的真正差异，通常不在“能不能做 RAG”，而在于：

- 是偏向通用 LLM 编排，还是偏向“围绕私有数据”的索引与检索；
- 是否适合 Java 企业栈、Python 原型，或需要复杂文档解析的场景；
- 组件能否替换、流程是否可测试、生产部署是否有足够的治理能力。

> 先用真实问题和文档验证，再决定框架。框架不能修复错误的 OCR、混乱的权限模型和缺失的评测集。

## 先分清四类产品

| 类别 | 解决的问题 | 代表 |
| --- | --- | --- |
| RAG/LLM 开发框架 | 用代码编排数据、检索与模型调用 | LangChain、LlamaIndex、Haystack、Spring AI |
| RAG 引擎 | 强调文档理解、知识库和引用问答 | RAGFlow、GraphRAG |
| 应用平台 | 用可视化方式交付知识库或工作流 | Dify、FastGPT |
| 基础设施 | 存储向量、全文检索、模型推理与观测 | Qdrant、Milvus、Elasticsearch、vLLM、Langfuse |

不要把这四类强行互相替代。一个生产知识库常常同时使用“开发框架 + 检索基础设施 + 观测系统”。

## 核心框架对比

| 框架 | 核心定位 | 语言 | 更适合 | 不适合直接拿来解决 |
| --- | --- | --- | --- | --- |
| LangChain | 通用 LLM 应用与集成组件 | Python、TypeScript | 多模型、多工具、复杂外部集成 | 数据质量与生产治理 |
| LlamaIndex | 数据接入、索引、检索与上下文增强 | Python、TypeScript | 文档/数据为中心的 RAG | 高度确定性的企业流程 |
| Haystack | 显式组件与 Pipeline 编排 | Python | 可测试、可替换的生产检索链路 | 追求最少代码的 demo |
| Spring AI | Java/Spring AI 应用抽象 | Java | Spring Boot 企业服务 | Python 生态实验项目 |
| RAGFlow | 深度文档理解与可引用知识库引擎 | Docker 化服务 | PDF、表格、复杂版面文档 | 极轻量本地试验 |
| GraphRAG | 图谱与社区摘要增强检索 | Python | 跨文档关系、全局主题问题 | 低成本、低延迟的普通 FAQ |

---

## 1. LangChain：通用集成层，不是专用 RAG 引擎

- 官方文档：[LangChain](https://docs.langchain.com/oss/python/langchain/overview)
- 语言：Python、TypeScript
- 定位：模型、消息、工具、检索器、加载器和中间件的通用抽象层。

LangChain 的优势是集成广。它适合需要同时连接多个模型 Provider、向量库、搜索服务、数据库、工具和 Agent 的项目。它提供的 Document Loader、Text Splitter、Retriever、Runnable 等组件可以快速组装 RAG 原型。

### 适合的场景

- 需要频繁替换模型、Embedding、向量库或搜索服务；
- RAG 只是 AI 应用的一部分，系统还需要工具调用、工作流或 Agent；
- 团队愿意自己定义数据管道、评测与服务层。

### 需要注意

LangChain 当前也覆盖 Agent 与 LangGraph 相关能力，因此“直接跟教程堆 Chain”很容易让 RAG 链路变得隐式。生产项目应把解析、入库、检索、重排和回答组装拆成明确模块，接口由业务代码控制。

### 推荐用法

~~~text
Loader/Parser → 自定义清洗与切分 → Vector Store
查询 → 业务检索服务 → Reranker → Prompt Builder → 模型 → 引用校验
~~~

让 LangChain 充当组件与适配层，而不是让业务规则藏在一条很长的链式调用里。

---

## 2. LlamaIndex：以“数据如何进入 LLM”为中心

- 官方文档：[LlamaIndex Framework](https://developers.llamaindex.ai/python/framework/)
- 语言：Python、TypeScript
- 定位：围绕私有数据构建上下文增强、索引、检索与 Agent 工作流的框架。

LlamaIndex 的强项是数据抽象。它使用 Document、Node、Index、Retriever、Query Engine 等概念，把“原始文档如何变成可检索上下文”这件事讲得更完整。对知识库、文档助手、SQL/结构化数据问答等场景，通常比纯通用编排框架更贴近问题。

### 适合的场景

- 多种文档、网页、数据库和 API 数据需要统一接入；
- 希望实验多种索引、检索器、后处理器和 query engine；
- RAG 的数据层是项目核心资产。

### 需要注意

LlamaIndex 提供许多高级索引和 Agent 能力，但不应一开始全部开启。先以简单向量检索建立基线，再增加 metadata filter、hybrid search、reranker 或 query transform，并以评测结果证明收益。

### 重点概念

| 概念 | 在 RAG 中的作用 |
| --- | --- |
| Document | 原始数据与来源信息 |
| Node | 可索引、可引用的细粒度内容单元 |
| Ingestion Pipeline | 清洗、切分、Embedding 与增量处理 |
| Retriever | 召回候选上下文 |
| Postprocessor | 过滤、去重、重排、压缩 |
| Query Engine | 将检索结果与模型回答封装为查询接口 |

---

## 3. Haystack：显式 Pipeline，更适合可测试的生产链路

- 官方文档：[Haystack](https://docs.haystack.deepset.ai/docs/intro)
- 语言：Python
- 定位：通过可复用组件和 Pipeline 构建生产级 RAG、搜索和多模态应用。

Haystack 的思路更接近传统搜索/数据管道工程：每个组件有清晰输入输出，Pipeline 显式连接。因此它适合对可测试性、可替换性和过程透明度有要求的团队。

### 适合的场景

- 要把检索、过滤、重排、生成、验证拆成独立节点；
- 有多套知识库或多条检索策略，需要 A/B 对比；
- 希望对每个节点做单元测试、离线评测和性能测量。

### 一个典型 Pipeline

~~~text
用户问题
  → Query Rewriter
  → Hybrid Retriever（BM25 + Dense）
  → Document Joiner / 去重
  → Reranker
  → Prompt Builder
  → Generator
  → Answer + Citation Validator
~~~

### 需要注意

Pipeline 图本身不是质量保证。仍要为每个节点定义输入边界、失败处理和指标：检索节点看召回，重排节点看排序，生成节点看事实性与引用。

---

## 4. Spring AI：Java/Spring 团队的 AI 集成方式

- 官方文档：[Spring AI](https://docs.spring.io/spring-ai/reference/)
- 语言：Java
- 定位：在 Spring 应用中提供可替换的模型、Embedding、Vector Store、工具调用和 RAG 抽象。

Spring AI 不是 Python 框架的简单移植。它更适合已有 Spring Boot、Spring Security、Spring Cloud、企业服务治理体系的团队：AI 能力可以沿用已有的配置、依赖注入、测试、鉴权和部署方式。

### 适合的场景

- Java 为主的企业后端，希望把 RAG 接入现有服务；
- 已使用 Spring Security、数据库事务、配置中心和监控体系；
- 需要统一接入 OpenAI、Anthropic、Google、Ollama 等模型 Provider。

### 推荐边界

将聊天模型、Embedding、检索和 Prompt 模板封装在 application service 中；Controller 只处理请求和认证；数据同步与向量化放入异步任务。不要在 Controller 里直接拼 Prompt 或直接暴露 Vector Store。

---

## 5. RAGFlow：复杂文档优先的 RAG 引擎

- 官方文档：[RAGFlow](https://ragflow.io/docs/)
- 项目仓库：[infiniflow/ragflow](https://github.com/infiniflow/ragflow)
- 定位：强调深度文档理解、可干预解析和基于引用的问答。

当资料主要是扫描 PDF、带复杂表格的报告、合同、手册、幻灯片或多栏排版文档时，“文本抽出来了”远远不够。RAGFlow 的价值在于把文档解析、分块和检索过程做成更可见、可干预的知识库引擎。

### 适合的场景

- 企业资料以 PDF、表格和复杂格式文档为主；
- 需要让答案可回溯到原始页码或片段；
- 团队希望通过 Docker 部署一套相对完整的 RAG 服务。

### 需要注意

官方部署文档对 CPU、内存、磁盘、Docker 和部分场景的 GPU 有明确要求。它不是一个轻量库；先用一小批最复杂、最有代表性的文件验证解析效果、资源消耗和权限方案。

---

## 6. Microsoft GraphRAG：解决“跨文本关联与全局主题”问题

- 官方文档：[Microsoft GraphRAG](https://microsoft.github.io/graphrag/)
- 论文：[From Local to Global](https://arxiv.org/abs/2404.16130)
- 定位：从原始语料构建实体、关系、社区与摘要，用图结构增强 RAG。

普通向量 RAG 擅长回答“哪一段提到某个问题”。它在需要跨多份资料连接人物、事件、实体关系，或者回答“整个知识库的主要主题/风险/趋势是什么”时容易失效。GraphRAG 的目标正是补足这类局部 chunk 无法覆盖的全局推理。

### 适合的场景

- 调研报告、新闻、会议纪要、组织知识和大型叙事性资料；
- 用户问题天然涉及关系、影响链、主题归纳与跨文档推理；
- 能接受更重的索引成本和离线处理时间。

### 不适合的场景

- 简单 FAQ、少量结构化产品文档；
- 高实时性写入、极低成本要求；
- 还没有完成基础解析、切分、权限和向量检索评测的项目。

先让 baseline RAG 达标，再证明图谱能解决它解决不了的问题；不要为了“先进”而引入昂贵索引。

---

## 7. 应用平台：Dify、FastGPT 与框架的关系

[Dify](https://docs.dify.ai/) 和 [FastGPT](https://doc.fastgpt.io/) 可以快速搭建知识库和工作流，但它们更接近应用开发平台，而不是底层 RAG 框架。

它们适合：

- 非研发角色需要可视化维护知识库；
- 需要快速验证内部客服、文档助手或工作流；
- 团队希望统一账号、发布入口和基础运营界面。

它们不替代：

- 对复杂数据管道的精细控制；
- 特殊检索算法、深度评测与高度定制的权限模型；
- 生产级备份、可观测性、成本治理和安全审计。

选择平台时尤其要确认数据是否可导出、向量库能否替换、权限如何同步、日志是否可审计，以及未来迁移到自研服务的成本。

---

## 如何做最终选择

### 个人或小团队原型

从 LlamaIndex 或 LangChain 开始。用最少组件接通真实文档，建立 20～50 个测试问题，再迭代检索和提示模板。

### Python 生产 RAG 服务

优先评估 Haystack 或将 LangChain/LlamaIndex 作为数据与集成层，同时自建明确的 Pipeline、评测和观测。

### Java 企业系统

优先评估 Spring AI，让 AI 接入既有的 Spring Security、数据库、配置和监控体系。

### 复杂 PDF 与文档知识库

先评估 RAGFlow 或专门的文档解析服务。解析质量、页码引用和人工干预能力通常比选择哪一个 LLM 更重要。

### 全局关系与跨文档洞察

在 baseline RAG 已经稳定的前提下，再评估 GraphRAG；为索引成本、图谱更新和质量验证预留预算。

## 框架无关的验收清单

1. 文档是否可增量同步、删除、重建，并保留版本和来源？
2. 检索阶段是否执行了租户与文档权限过滤？
3. 每个答案能否返回支持它的原文引用？
4. 是否有包含正常、无答案、冲突、过期和越权问题的评测集？
5. 能否测量召回、引用正确性、端到端成功率、延迟和成本？
6. 模型、Embedding、切分策略、索引与 Prompt 变更后，能否做回归测试？

如果这些问题没有答案，再多的框架功能也无法让 RAG 可靠上线。
