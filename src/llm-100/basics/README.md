---
title: 基础篇
icon: mdi:book-open-page-variant-outline
---

![](/images/basics/basics_cover.png)

# 基础篇：这个合集能让你学到什么？

你有没有想过，大模型到底"大"在哪里？为什么 Transformer 能成为所有大模型的底座？Attention 机制到底在算什么？

这些问题，是理解大模型的第一道门槛。跨过去，后面的训练、微调、Prompt 工程、RAG、Agent 才能真正学明白。跨不过去，你永远只能在表面打转。

这个系列共 10 篇文章，从最底层的概念出发，逐层拆解大模型的核心原理。每篇围绕一个具体问题展开，讲清楚"是什么、为什么、怎么做"。

---

## 十篇文章，带你从零到入门

**概念基础篇（前三篇）**

先搞清楚大模型的"大"到底体现在哪里。参数量、数据量、算力需求——这三个维度决定了大模型和传统 AI 的本质区别。然后深入 Transformer 架构，理解它为什么能取代 RNN 和 CNN 成为大模型的标配。

1. [大模型到底"大"在哪里？一文讲清参数、数据与算力](/llm-100/basics/大模型到底_大_在哪里_一文讲清参数_数据与算力.html) - 理解大模型的三要素
2. [Transformer 为什么能成为大模型的核心架构？](/llm-100/basics/Transformer_为什么能成为大模型的核心架构.html) - 架构选择的原因
3. [Attention 机制到底在算什么？](/llm-100/basics/Attention_机制到底在算什么.html) - 注意力的本质

**注意力机制篇（接着两篇）**

Attention 是 Transformer 的灵魂。搞懂 Q、K、V 的含义，理解 Self-Attention 和 Multi-Head Attention 的区别，才算真正理解了 Transformer 的核心。

4. [Self-Attention 和 Multi-Head Attention 有什么区别？](/llm-100/basics/Self-Attention_和_Multi-Head_Attention_有什么区别.html) - 单头 vs 多头
5. [Token 是什么？为什么大模型不是按字理解世界？](/llm-100/basics/Token_是什么_为什么大模型不是按字理解世界.html) - 分词的本质

**分词与向量篇（中间三篇）**

大模型不直接处理文字，而是处理 Token 和向量。BPE、WordPiece、SentencePiece 三种分词算法有什么区别？Embedding 如何把文字变成高维向量？位置编码为什么重要？

6. [BPE、WordPiece、SentencePiece 分词有什么区别？](/llm-100/basics/BPE_WordPiece_SentencePiece_分词有什么区别.html) - 三大分词算法对比
7. [Embedding 是什么？大模型如何把文字变成向量？](/llm-100/basics/Embedding_是什么_大模型如何把文字变成向量.html) - 向量化的核心
8. [位置编码为什么重要？RoPE、ALiBi 分别解决什么问题？](/llm-100/basics/位置编码为什么重要_RoPE_ALiBi_分别解决什么问题.html) - 位置信息的编码

**架构对比篇（最后两篇）**

Decoder-only、Encoder-only、Encoder-Decoder 三种架构有什么区别？GPT、BERT、T5 分别代表什么？理解这些，才算真正入门大模型架构。

9. [Decoder-only、Encoder-only、Encoder-Decoder 架构怎么区分？](/llm-100/basics/Decoder-only_Encoder-only_Encoder-Decoder_架构怎么区分.html) - 三大架构对比
10. [GPT、BERT、T5 的核心差异是什么？](/llm-100/basics/GPT_BERT_T5_的核心差异是什么.html) - 代表模型对比

---

[← 返回大模型100讲](/llm-100/)
