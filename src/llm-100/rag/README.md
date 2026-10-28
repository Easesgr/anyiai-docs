---
title: RAG 检索增强篇
icon: mdi:database
---

![](/images/rag/rag_cover.png)

# RAG 检索增强篇：这个合集能让你学到什么？

大模型有一个致命短板：它的知识是"冻结"在参数里的。

你问它今天天气怎么样？它不知道。你问它公司内部的文档内容？它也不知道。你问它最新的技术动态？它的回答可能过时了。

RAG（Retrieval-Augmented Generation）就是来解决这个问题的。它的核心思路很简单：先从外部知识库检索相关信息，再把检索到的内容塞给大模型，让它基于这些信息生成回答。就像"开卷考试"——模型不需要记住所有知识，只需要会"查资料"和"用资料"。

本篇共 10 篇文章，从 RAG 的基本概念出发，覆盖文档切分、向量数据库、检索方法、重排序，以及常见问题的排查与优化。

---

## 十篇文章，系统覆盖 RAG 全链路

**基础概念篇（前三篇）**

先搞清楚 RAG 是什么，它和微调有什么区别，什么时候该用 RAG。然后理解文档切分 Chunk 的策略——切太大检索不准，切太小丢失上下文。

1. [RAG 是什么？为什么大模型需要外部知识库？](/llm-100/rag/what-is-rag.html) - RAG 的本质
2. [RAG 和微调有什么区别？什么时候该用 RAG？](/llm-100/rag/rag-vs-finetuning.html) - 两种方案的取舍
3. [文档切分 Chunk 应该怎么做？](/llm-100/rag/chunk-strategy.html) - 切分策略详解

**检索核心篇（中间四篇）**

向量数据库在 RAG 中起什么作用？Embedding 模型如何选择？相似度检索、关键词检索、混合检索有什么区别？BM25 这个"老古董"为什么还能在 RAG 中发挥作用？

4. [向量数据库在 RAG 中起什么作用？](/llm-100/rag/vector-db-in-rag.html) - 向量存储与检索
5. [Embedding 模型如何选择？](/llm-100/rag/embedding-selection.html) - 模型选型指南
6. [相似度检索、关键词检索、混合检索有什么区别？](/llm-100/rag/retrieval-methods.html) - 三种检索方式对比
7. [BM25 为什么还能在 RAG 中发挥作用？](/llm-100/rag/bm25-in-rag.html) - 经典算法的价值

**优化与实战篇（最后三篇）**

Rerank 重排序为什么重要？RAG 为什么会答非所问？如何优化 RAG 的召回率和准确率？这些是 RAG 工程化必须解决的问题。

8. [Rerank 重排序为什么重要？](/llm-100/rag/rerank-importance.html) - 精排的价值
9. [RAG 为什么会答非所问？](/llm-100/rag/rag-wrong-answers.html) - 常见问题排查
10. [面试高频：如何优化 RAG 的召回率和准确率？](/llm-100/rag/optimize-rag.html) - 优化方法论

---

[← 返回大模型100讲](/llm-100/)
