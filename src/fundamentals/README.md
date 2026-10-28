---
title: AI 基础
icon: mdi:brain
---

# AI 基础

![](/images/fundamentals/fundamentals_cover.png)

从零开始，系统构建你的 AI 知识体系。

这里不是零散的知识点堆砌，而是一条完整的学习路径——从机器学习的基本概念，到深度学习的核心原理，再到大模型时代的工程实践。每篇文章都围绕一个具体问题展开，讲清楚"是什么、为什么、怎么做"。

---

## 十大模块

### 1. 机器学习基础（10 篇）

![](/images/ml-basics/ml-basics_cover.png)

经典算法与核心概念，AI 入门的第一站。覆盖数据与特征、训练集划分、损失函数与梯度下降、过拟合与欠拟合、线性回归与逻辑回归、分类与回归评估指标、决策树与随机森林、聚类与降维。

[→ 进入机器学习基础](/fundamentals/ml-basics/.html)

---

### 2. 深度学习基础（12 篇）

![](/images/dl-basics/dl-basics_cover.png)

从神经网络到 Transformer，理解深度学习的核心机制。覆盖神经网络基础、激活函数、前向与反向传播、梯度消失与爆炸、优化器、归一化、Dropout 与正则化、CNN、RNN/LSTM/GRU、注意力机制。

[→ 进入深度学习基础](/fundamentals/dl-basics/.html)

---

### 3. Transformer 架构（8 篇）

![](/images/transformer-arch/transformer-arch_cover.png)

现代大模型的底座，面试的"钉子户"。覆盖 Self-Attention 的 QKV、缩放因子 √dk、Multi-Head Attention、位置编码、Encoder/Decoder 架构变体、BERT vs GPT、计算复杂度与优化。

[→ 进入 Transformer 架构](/fundamentals/transformer-arch/.html)

---

### 4. 大模型基础（10 篇）

![](/images/llm-basics/llm-basics_cover.png)

从参数规模到应用系统，建立对大模型的全景认知。覆盖大模型是什么、Token 与上下文表示、预训练原理、微调技术（SFT/LoRA/QLoRA）、Prompt 基础、上下文窗口、幻觉问题、推理参数调优、Function Calling、应用系统设计。

[→ 进入大模型基础](/fundamentals/llm-basics/.html)

---

### 5. Token 与分词（5 篇）

![](/images/token-and-tokenizer/token-and-tokenizer_cover.png)

理解大模型如何"看"文本，以及为什么中文更贵。覆盖 Token 的本质、Tokenizer 工作原理（BPE/WordPiece/SentencePiece）、中文分词的特殊性、Token 预算与费用计算、Tokenization 对 Prompt/RAG 的影响。

[→ 进入 Token 与分词](/fundamentals/token-and-tokenizer/.html)

---

### 6. Embedding（6 篇）

![](/images/embedding/embedding_cover.png)

语义检索的基石，RAG 的核心组件。覆盖 Embedding 的本质、余弦相似度与向量空间、模型选型（维度/中文效果/成本）、Embedding 在 RAG 中的作用、效果优化（切分/归一化/混合检索）、稀疏向量 vs 稠密向量 vs 多向量检索。

[→ 进入 Embedding](/fundamentals/embedding/.html)

---

### 7. 训练与微调（8 篇）

![](/images/training-and-finetuning/training-and-finetuning_cover.png)

大模型训练的完整流程，从预训练到偏好对齐。覆盖训练流程全景、预训练深度解析、SFT 监督微调、LoRA 与 QLoRA、RLHF 与 DPO、微调数据构造、微调常见问题（过拟合/灾难性遗忘）、Prompt/RAG/微调的技术选型。

[→ 进入训练与微调](/fundamentals/training-and-finetuning/.html)

---

### 8. 推理与生成（8 篇）

![](/images/inference-and-generation/inference-and-generation_cover.png)

从 Prompt 到输出的完整链路，以及如何让它更快更稳。覆盖推理流程、Temperature/Top-k/Top-p 采样策略、KV Cache 加速原理、上下文长度与成本、流式输出（SSE）、结构化输出与 Function Calling、推理加速（量化/批处理/vLLM）、生成常见问题（幻觉/重复/截断）。

[→ 进入推理与生成](/fundamentals/inference-and-generation/.html)

---

### 9. 多模态基础（7 篇）

![](/images/multimodal-basics/multimodal-basics_cover.png)

文本、图像、音频的统一理解，AI 的下一个战场。覆盖多模态模型概述、CLIP 图文对齐、视觉 Token、多模态大模型工作原理、OCR 与文档理解、多模态 RAG、多模态模型的局限（幻觉/定位错误/细节丢失）。

[→ 进入多模态基础](/fundamentals/multimodal-basics/.html)

---

### 10. 评测指标（6 篇）

![](/images/evaluation-metrics/evaluation-metrics_cover.png)

不被分数迷惑，才能真正做好模型选型。覆盖模型评测三维度（准确性/鲁棒性/可用性）、分类任务指标（Accuracy/Precision/Recall/F1）、生成任务指标（BLEU/ROUGE/BERTScore）、大模型评测（MMLU/HumanEval/Arena）、RAG 评测、AI 应用评测方法。

[→ 进入评测指标](/fundamentals/evaluation-metrics/.html)
