---
title: RAG 面试题
icon: mdi:database
tag:
  - RAG面试
  - 面试题
  - RAG
---

# RAG 面试题

RAG（检索增强生成）高频面试问题精选，共 38 篇文章，覆盖基础概念、核心流程、检索优化、文档处理、提示工程、评估方法和进阶方向。

RAG 是目前大模型应用中最主流的技术之一，也是面试中出现频率最高的方向。每篇文章围绕一个具体问题展开，讲清楚"是什么、为什么、怎么做"。

---

## 合集文章导航

### 基础概念篇

1. [RAG 是什么？](/interview/rag-interview/what-is-rag.html) - 它解决了大模型应用中的什么问题
2. [在 LangChain 中如何实现 RAG 应用？](/interview/rag-interview/what-is-rag-langchain.html) - 框架实践
3. [RAG 的完整流程是什么？](/interview/rag-interview/rag-pipeline.html) - 文档解析、切分、检索、生成分别做什么
4. [RAG 的完整工作流程是怎样的？核心步骤有哪些？](/interview/rag-interview/rag-complete-workflow.html) - 全流程详解
5. [RAG 的完整流程是怎么样的？](/interview/rag-interview/rag-full-pipeline.html) - 端到端实现
6. [RAG 的文档处理流程是怎样的？](/interview/rag-interview/rag-document-pipeline.html) - 文档生命周期
7. [文档切分为什么重要？](/interview/rag-interview/chunk-strategy.html) - chunk size 和 overlap 应该怎么选
8. [RAG 系统中为什么要进行文档切割？有哪些切割策略？](/interview/rag-interview/rag-chunk-strategies.html) - 切分方法对比
9. [RAG 中 chunk size 和 overlap 应该怎么设置？](/interview/rag-interview/rag-chunk-size-overlap.html) - 参数调优经验
10. [Embedding 在 RAG 中有什么作用？](/interview/rag-interview/embedding-in-rag.html) - 如何判断向量表示是否有效
11. [RAG 中的 Embedding 向量化是什么？如何工作的？](/interview/rag-interview/rag-embedding.html) - 向量生成机制
12. [RAG 中如何计算文本相似度？常见算法有哪些？](/interview/rag-interview/rag-similarity-algorithms.html) - 相似度计算

### 检索与召回篇

13. [向量检索、关键词检索和混合检索有什么区别？](/interview/rag-interview/retrieval-methods.html) - 三种检索方式对比
14. [为什么 RAG 中常用 BM25 + 向量检索的混合召回？](/interview/rag-interview/hybrid-retrieval.html) - 混合召回的原理
15. [什么是 RAG 混合检索？如何实现向量和关键词检索结合？](/interview/rag-interview/rag-hybrid-retrieval.html) - 混合检索实现
16. [RAG 检索中的 Top-K 是什么意思？K 值如何确定？](/interview/rag-interview/rag-topk.html) - Top-K 调优
17. [RAG 检索时相似度阈值如何设置？设置不当有什么影响？](/interview/rag-interview/rag-similarity-threshold.html) - 阈值经验
18. [什么是向量数据库？主要解决什么问题？](/interview/rag-interview/vector-database.html) - 向量库核心概念
19. [RAG 为什么需要向量数据库？和传统数据库有什么区别？](/interview/rag-interview/vector-db-vs-traditional.html) - 向量库对比
20. [RAG 中如何实现向量数据库的增量更新？](/interview/rag-interview/vector-db-incremental.html) - 数据同步机制
21. [RAG 系统如何利用元数据过滤提升检索精度？](/interview/rag-interview/rag-metadata-filtering.html) - 元数据增强
22. [RAG 系统如何处理 PDF、Word、Markdown 等不同格式文档？](/interview/rag-interview/rag-multiformat-parsing.html) - 多格式解析
23. [RAG 系统如何标注信息来源和提供引用？](/interview/rag-interview/rag-source-citation.html) - 可信度增强

### 重排序与查询优化篇

24. [Rerank 在 RAG 中解决什么问题？](/interview/rag-interview/rerank.html) - 和向量检索有什么区别
25. [什么是 RAG 中的 Rerank？具体需要怎么做？](/interview/rag-interview/rag-rerank.html) - 重排序实现
26. [Query Rewrite 是什么？](/interview/rag-interview/query-rewrite.html) - 它如何提升 RAG 召回效果
27. [RAG 中如何设计 Prompt 来有效利用检索到的文档？](/interview/rag-interview/rag-prompt-design.html) - Prompt 工程
28. [什么是上下文查询增强？它有什么作用？如何处理无关问题？](/interview/rag-interview/contextual-query-enhancement.html) - Query 增强

### 质量与评估篇

29. [RAG 为什么仍然会产生幻觉？](/interview/rag-interview/rag-hallucination.html) - 如何降低幻觉
30. [如何评估一个 RAG 系统？](/interview/rag-interview/rag-evaluation.html) - Hit@K、MRR、召回率和答案支撑度
31. [RAG 中如何处理 Markdown、表格、代码块和图片？](/interview/rag-interview/rag-content-types.html) - 多格式内容处理

### 进阶方向篇

32. [GraphRAG、Self-RAG 和 Agentic RAG 分别想解决什么问题？](/interview/rag-interview/advanced-rag.html) - 三种进阶 RAG 方案
33. [什么是 Advanced RAG？](/interview/rag-interview/advanced-rag.html) - 进阶方法
34. [什么是 Modular RAG？](/interview/rag-interview/modular-rag.html) - 模块化 RAG
35. [Spring AI 提出的模块化 RAG 架构是什么？](/interview/rag-interview/spring-ai-modular-rag.html) - 三阶段架构
36. [什么是 Re-Reading？如何基于 Spring AI 实现？](/interview/rag-interview/re-reading-spring-ai.html) - Re-Reading Advisor
37. [RAG 和微调有什么区别？](/interview/rag-interview/rag-vs-finetuning.html) - 什么场景更适合使用 RAG
38. [如果 RAG 检索结果不相关，你会从哪些环节排查？](/interview/rag-interview/rag-troubleshooting.html) - 问题排查指南
