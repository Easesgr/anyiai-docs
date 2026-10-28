---
title: 技术书籍
icon: mdi:book-open-page-variant-outline
tag:
  - AI 学习
  - 技术书籍
  - LLM
---

# AI 技术书籍与课程

书籍适合建立长期知识框架，官方文档适合追踪工具的最新变化。最有效的学习方式通常是：读一个章节、实现一个最小实验、再用真实问题复盘。

## 基础与深度学习

### 《动手学深度学习》

- 在线阅读：[Dive into Deep Learning](https://d2l.ai/)
- 推荐理由：从线性模型到 Transformer，理论与可运行代码结合，中文学习资料完整。
- 适合：从零开始建立深度学习实现能力。

### 《Deep Learning》

- 在线阅读：[Deep Learning Book](https://www.deeplearningbook.org/)
- 推荐理由：系统覆盖数学基础、表示学习与深度网络，是长期可查阅的理论参考。
- 适合：希望补足原理与数学基础的读者。

### fast.ai

- 官方课程：[fast.ai](https://course.fast.ai/)
- 推荐理由：先做出可用模型再回到原理，强调实践、数据与迭代。
- 适合：已有基本 Python 能力、希望快速做项目的学习者。

## NLP 与大模型

### 《Speech and Language Processing》

- 在线阅读：[Jurafsky & Martin](https://web.stanford.edu/~jurafsky/slp3/)
- 推荐理由：NLP 的系统教材，持续覆盖现代语言模型和语音/对话主题。

### 《Natural Language Processing with Transformers》

- 官方资源：[Hugging Face Course](https://huggingface.co/learn/nlp-course/)
- 推荐理由：与 Transformers 生态配套，适合把 Transformer 知识落到具体任务。

### 《Build a Large Language Model (From Scratch)》

- 官方资源：[作者主页](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- 推荐理由：通过从零实现理解 tokenizer、注意力、预训练与微调流程。

## AI 工程与系统设计

### 《Designing Machine Learning Systems》

- 作者资源：[Chip Huyen](https://huyenchip.com/)
- 推荐理由：从数据、训练、部署、监控到迭代讨论 ML 系统的真实约束。

### 《AI Engineering》

- 官方入口：[AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/)
- 推荐理由：聚焦 LLM 应用、评测、检索、Agent 与工程权衡。

## 推荐阅读顺序

1. 入门：D2L + fast.ai，完成一个可复现项目。
2. 原理：补线性代数、概率与 Transformer/NLP 基础。
3. LLM：结合 Hugging Face Course 和从零实现类材料。
4. 工程：学习评测、数据治理、部署、可观测性与安全。

阅读时请留意版本：框架用法变化很快，书中的 API 示例应与当前官方文档交叉核对。

## 12 周实践学习安排

| 阶段 | 建议投入 | 产出 |
| --- | --- | --- |
| 第 1～3 周 | D2L/fast.ai 基础章节 | 一个可训练、可复现的分类或回归项目 |
| 第 4～6 周 | Transformer 与 NLP | 手写 attention 或完成一个文本任务 |
| 第 7～9 周 | LLM 与 RAG | 文档问答原型与一套评测问题 |
| 第 10～12 周 | AI 工程 | 部署、日志、成本和安全复盘报告 |

每一阶段都应留下代码仓库、实验记录和复盘，而不是只收藏链接。

## 阅读论文和书籍的笔记模板

1. 它要解决的真实问题是什么？
2. 核心假设和输入输出是什么？
3. 哪个公式、图或实验最关键？
4. 在自己的数据或场景里会在哪一步失效？
5. 下一步能做的最小实验是什么？

这样读完的资料才会转化为可迁移的判断力。
