---
title: 从原理到 AI 工程学习路线
icon: mdi:transit-connection-variant
description: 从机器学习与深度学习原理出发，逐步掌握 Transformer、大模型、RAG、Agent 和 AI 应用系统工程。
tag:
  - 学习路线
  - AI 原理
  - 大模型工程
---

# 从原理到 AI 工程学习路线

这条路线适合不满足于“会调用 API”，希望真正理解模型为何有效、系统为何失效，并能独立完成 AI 应用设计与交付的学习者。建议投入 **16～24 周**，每周学习 8～12 小时。

路线不是把所有文章从头读一遍，而是围绕“理解—实现—验证—复盘”建立能力。每个阶段都应留下可运行的代码、设计文档或实验报告。

## 完成路线后应具备的能力

- 能解释机器学习、神经网络、注意力机制和 Transformer 的核心原理。
- 能判断 Prompt、RAG、微调与 Agent 分别适合解决什么问题。
- 能搭建包含文档解析、检索、生成、评估和监控的 RAG 系统。
- 能设计工具调用、记忆、规划、错误恢复与人工介入机制。
- 能从准确率、延迟、成本、安全和可维护性角度评审 AI 系统。

## 阶段一：机器学习与深度学习基础（第 1～4 周）

先建立训练、泛化和评估的共同语言。这里不追求推导所有公式，而是理解每个概念会怎样影响模型行为。

### 核心内容

- [机器学习基础](/fundamentals/ml-basics/)：监督学习、无监督学习、损失函数、过拟合与正则化。
- [深度学习基础](/fundamentals/dl-basics/)：神经网络、反向传播、优化器、CNN 与 RNN。
- [评估指标](/fundamentals/evaluation-metrics/)：分类、生成和检索任务的指标选择。
- [训练与微调](/fundamentals/training-and-finetuning/)：训练流程、参数更新与微调方式。

### 阶段实践

用 PyTorch 完成一个小型分类任务，记录训练集与验证集曲线，主动制造一次过拟合，再通过正则化、数据增强或早停修正。最终写一页实验结论：问题是什么、证据是什么、修改为何有效。

## 阶段二：Transformer 与大模型原理（第 5～8 周）

这一阶段要从“模型能生成文本”深入到“输入如何变成 Token、信息如何在层间传播、输出为何具有随机性”。

### 核心内容

- [Transformer 架构](/fundamentals/transformer-arch/)：自注意力、多头注意力、位置编码、残差连接与归一化。
- [Token 与分词](/fundamentals/token-and-tokenizer/)：BPE、词表和上下文长度。
- [Embedding](/fundamentals/embedding/)：语义表示、相似度与向量空间。
- [大模型基础](/fundamentals/llm-basics/)：预训练、指令微调、对齐与涌现能力。
- [推理与生成](/fundamentals/inference-and-generation/)：Temperature、Top-P、KV Cache 与生成策略。
- [大模型 100 讲·基础篇](/llm-100/basics/) 与 [训练原理篇](/llm-100/training/)。

### 阶段实践

加载一个开源小模型，对同一输入调整 Temperature、Top-P、最大长度和系统提示词，记录输出差异。再实现一个简化版 Attention，验证张量形状、Mask 和 Softmax 的作用。

## 阶段三：Prompt 与可靠输出（第 9～10 周）

Prompt 不是修辞技巧，而是对任务、上下文、约束和输出协议的工程化描述。

### 核心内容

- [提示词工程](/llm-100/prompt-engineering/)：角色、任务、示例、约束与结构化输出。
- [Prompt 面试题](/interview/prompt-interview/)：Few-shot、思维链、注入攻击与失败处理。
- 多轮对话中的上下文裁剪、摘要和状态管理。

### 阶段实践

实现一个结构化信息抽取服务：输入非结构化文本，输出经过 Schema 校验的 JSON；为格式错误、字段缺失、拒答和恶意输入设计重试与降级策略。

## 阶段四：RAG 系统（第 11～14 周）

RAG 的核心不只是“向量库 + 大模型”，而是让外部知识经过解析、切分、索引、召回、重排和引用后可靠进入生成过程。

### 核心内容

- [RAG 检索增强篇](/llm-100/rag/)：完整链路与模块边界。
- [RAG 面试题](/interview/rag-interview/)：Chunk、Embedding、混合检索、Rerank、Query Rewrite 与评估。
- [文档解析](/frontier/document-parsing/)：PDF、表格、图片和复杂版面。
- [Agentic RAG](/frontier/agentic-rag/) 与 [GraphRAG](/frontier/graphrag/)：复杂检索策略。

### 阶段实践

完成一个带来源引用的知识库问答系统，至少包含：多格式解析、可配置切分、向量检索、关键词检索、Rerank、无答案判断和离线评估集。记录命中率、答案正确率、P95 延迟与单次成本。

## 阶段五：Agent 与工具系统（第 15～18 周）

Agent 的价值来自对任务循环和工具系统的控制，而不是无限增加自主性。

### 核心内容

- [Agent 基础篇](/llm-100/agent-basics/)：ReAct、规划、行动与观察。
- [Agent 工程篇](/llm-100/agent-engineering/)：工具协议、状态、错误恢复和可观测性。
- [Agent 面试题](/interview/agent-interview/)：Tool Calling、记忆、多 Agent 与安全。
- [MCP 协议](/frontier/mcp-protocol/)、[Agent Skills](/frontier/agent-skills/) 和 [Agent 记忆系统](/frontier/agent-memory-system/)。

### 阶段实践

构建一个能读取资料、调用搜索或数据库、生成结果并自检的任务 Agent。限制最大循环次数，为每个工具定义超时、重试、权限和审计日志，并加入人工确认节点。

## 阶段六：AI 应用系统工程（第 19～24 周）

最后把模型能力放进真实软件系统。重点从“效果演示”转向可靠性、成本、性能和迭代效率。

### 核心内容

- [大模型应用系统篇](/llm-100/app-system/)：模型网关、会话、任务队列与数据闭环。
- [部署与推理优化篇](/llm-100/deployment/)：吞吐、延迟、量化、批处理与缓存。
- [评估、安全与面试综合篇](/llm-100/evaluation/)：离线评估、在线反馈与安全治理。
- [AI 系统设计面试题](/interview/ai-system-design-interview/) 与 [模型部署面试题](/interview/model-deployment-interview/)。

### 毕业项目

选择知识助手、智能客服、研究 Agent 或代码助手中的一个，交付以下材料：

1. 可运行的前后端或 API 服务。
2. 系统架构图、数据流和关键技术决策。
3. 至少 30 条评估样本及结果报告。
4. 延迟、成本、错误率和降级策略。
5. README、演示视频和一次完整复盘。

## 学习节奏建议

- 每周 40% 时间阅读原理，40% 编码实验，20% 整理输出。
- 每完成一个阶段，用自己的话回答“它解决什么问题、为什么有效、什么时候会失败”。
- 不同时学习多个框架；先用一个技术栈跑通，再比较替代方案。
- 每两周复盘一次，把失败案例加入测试集，而不是只保留成功演示。

真正完成这条路线的标志，不是阅读进度达到 100%，而是你能解释关键原理、独立定位失败原因，并把一个 AI 原型变成可评估、可维护的工程系统。
