---
title: RAG 实战学习路线
icon: mdi:database-search
tag:
  - 学习路线
  - RAG
  - 检索增强
---

# RAG 实战学习路线

> 已有 LLM 应用基础，专攻 RAG 方向，建议 2-3 个月

## 阶段 1：基础理论（1 周）

### 必学

- [RAG 面试题](/interview/rag-interview/) - 系统学习 RAG 全流程
- [Embedding](/fundamentals/embedding/) - 向量检索基础

## 阶段 2：基础 RAG 搭建（2 周）

### 实战

做一个「PDF 文档问答」项目：
- [文档解析](/frontier/document-parsing/) - PDF/Word/Markdown
- 文本切分（chunk size、overlap）
- Embedding + 向量数据库
- 检索 + Prompt + 生成

技术栈建议：LangChain + Chroma + OpenAI

## 阶段 3：检索优化（2-3 周）

### 必学

- [RAG 面试题 - 检索优化](/interview/rag-interview/) - 各种检索策略
- 混合检索（BM25 + 向量）
- Rerank 重排序
- 元数据过滤

### 实战

优化阶段 2 的项目：
- 加入混合检索
- 加入 Rerank 提升 top-5 质量
- 加入元数据过滤（按章节/时间筛选）

## 阶段 4：评估体系（1 周）

### 必学

- Hit@K、MRR、Recall
- 答案支撑度评估
- LLM-as-a-Judge

### 实战

为阶段 3 的项目搭建评估 pipeline：
- 准备 50+ 测试问题
- 对比不同检索策略的指标

## 阶段 5：进阶方向（按兴趣选）

### 方向 A：Advanced RAG

- Query Rewrite、HyDE
- GraphRAG、Agentic RAG

### 方向 B：生产化

- [AI 工作流](/frontier/ai-workflow/) - 工作流编排
- [工作流编排](/frontier/workflow-orchestration/) - 复杂 RAG Pipeline

### 方向 C：多模态

- [多模态基础](/fundamentals/multimodal-basics/) - 图像/音频 RAG

## 推荐项目作品

完成路线后，简历上至少要有：

1. **基础 RAG 问答系统** - 完整可运行
2. **生产级 RAG 服务** - 有评估、有监控
3. **开源贡献** - 给 LangChain/LlamaIndex 提 PR

## 求职目标

- RAG 算法工程师
- AI 搜索工程师
- 知识库平台工程师
