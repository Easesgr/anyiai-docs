---
title: 作者介绍
icon: mdi:account-outline
description: 安逸Ai文档站作者的学术背景、技术经历、项目实践与联系方式介绍。
---

![作者背景与技术实践概览](/images/content-rebuild/143a39652ec5a737/cover.png)

我是安逸，[安逸Ai](https://github.com/Easesgr/anyiai-docs) 文档站的创作者与维护者。本站的定位是面向中文学习者的 AI 技术学习与实践整理站点，内容力求说清概念背后的因果关系与工程实现细节，不堆砌无解释的代码和公式。

---

## 近况与方向

目前为人工智能专业硕士在读，学术研究方向为图像修复与低层视觉。在完成学术课题的同时，工作重点主要放在大语言模型应用工程的落地探索，核心关注以下技术方向：

- **检索增强生成（RAG）**：文档结构化解析、混合检索（BM25 与向量检索融合）、重排序（Reranking）及检索召回评测。
- **智能体（Agent）系统**：基于 ReAct 范式与 Tool Calling 的多 Agent 协作工作流、状态与记忆管理。
- **模型上下文与工具协议**：MCP（Model Context Protocol）工具封装与本地业务接口对接，以及针对特定场景的提示工程实践。

---

## 教育背景

- **2024.09 - 至今**：云南大学（211）· 人工智能 · 硕士在读  
  主修课程：最优化算法、人工智能导论、高级计算机网络、智能数据工程、软件开发方法与技术、工程数学等。
- **2020.09 - 2024.07**：江西水利电力大学 · 数据科学与大数据技术 · 本科  
  成绩排名专业前 5%（GPA 3.74 / 4.0）。主修课程：人工智能导论、机器学习、深度学习、自然语言处理、数据结构、算法设计与分析等。

---

## 技术栈

- **编程语言与后端**：Python、Java、FastAPI、Flask、Spring Boot
- **前端与站点构建**：Vue 3、TypeScript、VuePress
- **数据与存储**：MySQL、Redis、ChromaDB
- **大模型与 Agent 生态**：LangChain、CrewAI、MCP 协议、Claude Code、Trae

---

## 项目经历

### 1. 基于 CrewAI 的多 Agent 内容创作与生成流程（2026.04 - 2026.05）
- **角色**：全栈开发
- **实现机制**：基于 CrewAI 搭建自动化内容生产流水线，拆解出选题调研、提纲设计、正文起草、图文素材渲染与视频脚本合成 5 个流程阶段，共划分为 32 个 Agent/Crew 节点与 17 个自定义工具函数。
- **工程配置**：采用 YAML 解耦角色设定与工作流逻辑，配合 Playwright 与 FFmpeg 实现媒体内容生成。
- **技术栈**：Python、CrewAI、Flask、Playwright、FFmpeg、Vue 3、TypeScript。

### 2. AI 面试 RAG 知识检索系统（2026.05 - 2026.06）
- **角色**：后端开发
- **实现机制**：配合 VuePress 知识库搭建问答检索后端，设计并实现了“向量检索 + BM25 稀疏检索 + RRF 倒数排名融合 + BGE-Reranker 重排序”的多路混合检索链路，提升专有名词和长尾面试考点的召回率，支持检索来源溯源。
- **技术栈**：VuePress、Python、FastAPI、LangChain、ChromaDB、BM25、BGE-Reranker。

---

## 为什么做安逸Ai？

在自学和实践 AI 技术的过程中，常常遇到三类问题：内容过于简略缺乏背景因果、数学推导与代码实现脱节、以及资料过于碎片化缺乏系统主线。搭建 [安逸Ai](https://github.com/Easesgr/anyiai-docs) 的目的，是把踩坑经验与学习笔记按照体系整理出来，提供一条普通开发者能看懂、具备工程可操作性的学习路径。

目前站点已开放的主要内容包括：

- [AI 基础](/fundamentals/)：包含机器学习基础、深度学习基础、Transformer 架构、微调与推理等 10 大核心模块。
- [AI 经典面试题](/interview/) 与 [大模型100讲](/llm-100/)：针对面试考点与大模型系统工程的系统梳理。
- [前沿专题](/frontier/) 与 [学习路径](/learning-path/)：跟踪 MCP 协议、Agentic RAG 等工程前沿。
- [资料包下载](/about/pdf-download.md)：提供对应板块的阶段复习 PDF 文档。

如果你在阅读中发现了错别字、原理解释偏差或代码逻辑问题，欢迎参考 [贡献指南](/about/contributing-guide.md) 反馈问题或提交改进建议。

---

## 联系与交流

由于精力有限，本站未设立任何微信群、QQ群或实时答疑群。如有勘误建议、技术探讨或交流需求，可以通过以下渠道找到我：

- **GitHub**：[Easesgr](https://github.com/Easesgr)（文档主仓库：[anyiai-docs](https://github.com/Easesgr/anyiai-docs)）
- **微信公众号**：安逸Ai（可在后台直接发送留言交流）

欢迎扫描下方二维码关注公众号：

<div class="qrcode-card">
  <div class="qrcode-wrap">
    <img src="/images/wechat-qrcode.png" alt="安逸Ai 微信公众号二维码" class="qrcode-img" />
  </div>
  <p class="qrcode-text">微信搜索公众号：<strong>安逸Ai</strong></p>
</div>
