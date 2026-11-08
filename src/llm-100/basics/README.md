---
title: 基础篇导读：文本如何转化为计算？大模型前向流程与十讲脉络
icon: mdi:book-open-page-variant-outline
description: 梳理从输入文本到输出概率的大模型前向计算全链路，拆解基础篇十讲的递进逻辑，并通过可运行张量代码验证注意力形状演变。
---

大语言模型在应用层表现为聊天、补全或工具调用，但在计算层，它是一个由确定性矩阵乘法、非线性激活与归一化构成的参数化网络。任何提示词进入模型，最终产出的都是词表中每个词元（Token）的预测概率。

要建立扎实的工程认知，不能只停留在提示词调优，而需要理解自然语言字符串如何转化为高维向量、如何在自注意力网络中聚合上下文，并以自回归方式逐字生成。本篇梳理基础篇十讲的知识递进脉络，并通过一个具体的张量实验提供自测验证方法。

![大模型基础模块知识图谱](/images/content-rebuild/9770d2485b44acd3/cover.png)

---

## 1. 贯穿示例：从字符串到概率分布的前向全流程

以输入短语 `"小猫 追 蝴蝶"` 为例，大模型生成下一个词的过程可以拆解为四个连续阶段：

1. **切分与查表（Tokenization）**：
   计算机无法直接对中文字符做矩阵运算。分词器根据预先构建的词表，将字符串拆分为词元，并转换为整数索引。假设切分结果为 `["小猫", "追", "蝴蝶"]`，对应的整数 ID 为 `[2048, 512, 4096]`。此时输入张量的形状为 `[Batch_Size=1, Seq_Len=3]`。

2. **向量映射与位置注入（Embedding + Positional Encoding）**：
   模型通过嵌入矩阵（Embedding Matrix）将每个整数 ID 映射为一个连续的稠密向量（例如隐层维度 $d_{\text{model}} = 4096$）。同时，由于自注意力运算本身不具备时序先验，模型需要引入位置信息：传统方式是在嵌入向量上叠加绝对位置编码，得到形状为 `[1, 3, 4096]` 的特征张量；而在现代主流架构中，RoPE 等旋转位置编码则是在后续自注意力层对 Query 与 Key 动态注入。

3. **上下文特征聚合（Transformer Blocks）**：
   张量进入多层堆叠的 Transformer 结构块。在每个块内：
   - **自注意力层（Self-Attention）**：计算序列中词元之间的关联程度。在生成式因果模型中，因果掩码确保“小猫”只能看到自身，而“追”能看到“小猫”和“追”，防止未来信息泄露。
   - **前馈网络层（FFN）**：通过两层线性映射和非线性激活函数对聚合后的信息进行特征变换。
   残差连接要求每个结构块的输入与输出张量形状严格保持一致（均为 `[1, 3, 4096]`）。

4. **输出投影与采样（Unembedding & Sampling）**：
   经过最后一层 Transformer 块后，序列末尾词元（“蝴蝶”）对应的输出向量通过反嵌入矩阵（形状为 `[4096, 词表大小 V]`）映射为全词表的对数几率（Logits），形状为 `[1, 1, V]`。经由 Softmax 归一化为概率分布后，解码策略（如贪心搜索、Top-p 采样）从中选出下一个 Token ID（例如代表标点 `"。"` 的 ID），完成单步生成。

![自然语言在 Transformer 中的数据流动与张量形状演变流程图](/images/content-rebuild/9770d2485b44acd3/fig-1.png)

---

## 2. 十讲知识递进图谱

基础篇的十篇内容按照计算链路自下而上展开，前一讲的结论是后一讲的前提：

```
[系统规模] 大模型三要素 (第1讲)
    │
[离散映射] Token机制 (第2讲) ──> 分词算法 (第3讲) ──> 嵌入向量 (第4讲)
                                                        │
[计算核心] Transformer架构 (第5讲) <────────────────────┘
    │
    ├──> 注意力算子 (第6讲) ──> 多头注意力 (第7讲)
    │
    └──> 位置编码演进 (第8讲)
            │
[整机选型] 架构拓扑对比 (第9讲) ──> GPT/BERT/T5范式对比 (第10讲)
```

### 第一阶：离散表示与物理约束
- [/llm-100/basics/大模型到底_大_在哪里_一文讲清参数_数据与算力.md](/llm-100/basics/大模型到底_大_在哪里_一文讲清参数_数据与算力.md)：拆解参数量、训练 Token 量与计算浮点数（FLOPs）的缩放关系，建立算力与显存开销的基准常识。
- [/llm-100/basics/Token_是什么_为什么大模型不是按字理解世界.md](/llm-100/basics/Token_是什么_为什么大模型不是按字理解世界.md)：分析纯字符级切分与全词切分的局限，说明子词化（Subword）如何平衡词表体积与表达泛化。
- [/llm-100/basics/BPE_WordPiece_SentencePiece_分词有什么区别.md](/llm-100/basics/BPE_WordPiece_SentencePiece_分词有什么区别.md)：对比常见分词算法的构建规则、空白符处理方式与多语言跨语种场景下的工程差异。
- [/llm-100/basics/Embedding_是什么_大模型如何把文字变成向量.md](/llm-100/basics/Embedding_是什么_大模型如何把文字变成向量.md)：剖析高维稀疏表示与稠密向量空间的区别，说明向量内积如何衡量语义相似度。

### 第二阶：核心算子与位置建模
- [/llm-100/basics/Transformer_为什么能成为大模型的核心架构.md](/llm-100/basics/Transformer_为什么能成为大模型的核心架构.md)：对比 RNN 的串行循环与 CNN 的局部感受野，解释全序列并行训练与任意两点间恒定计算路径的工程优势。
- [/llm-100/basics/Attention_机制到底在算什么.md](/llm-100/basics/Attention_机制到底在算什么.md)：推导缩放点积注意力公式 $\text{Softmax}(\frac{QK^T}{\sqrt{d_k}})V$ 的计算过程，解释方差缩放与因果下三角掩码的具体作用。
- [/llm-100/basics/Self-Attention_和_Multi-Head_Attention_有什么区别.md](/llm-100/basics/Self-Attention_和_Multi-Head_Attention_有什么区别.md)：分析将隐藏层拆分为多个注意力头的机制，说明多子空间投影如何并行关注不同维度的语法与上下文线索。
- [/llm-100/basics/位置编码为什么重要_RoPE_ALiBi_分别解决什么问题.md](/llm-100/basics/位置编码为什么重要_RoPE_ALiBi_分别解决什么问题.md)：解释置换不变性的破除方式，对比绝对位置编码、相对位置偏置与旋转位置编码（RoPE）的外推特性。

### 第三阶：架构分类与技术收敛
- [/llm-100/basics/Decoder-only_Encoder-only_Encoder-Decoder_架构怎么区分.md](/llm-100/basics/Decoder-only_Encoder-only_Encoder-Decoder_架构怎么区分.md)：通过注意力掩码的可见性，厘清自回归解码器、双向编码器与编码-解码器架构在结构上的异同。
- [/llm-100/basics/GPT_BERT_T5_的核心差异是什么.md](/llm-100/basics/GPT_BERT_T5_的核心差异是什么.md)：从预训练目标（自回归、掩码语言模型、片段替换）出发，解释主流大模型在生成任务上收敛至 Decoder-only 架构的原因。

![基础篇十讲的递进因果链条与前置依赖结构拓扑](/images/content-rebuild/9770d2485b44acd3/fig-2.png)

---

## 3. 学习目标与工程边界划分

学习本模块时，建议区分模型本身的计算逻辑与下游部署工程：

### 本篇核心目标
- **掌握张量维度流转**：能够独立推导输入序列在分词、嵌入、多头注意力、FFN 与输出投影全过程中的形状变化（Batch, Seq_Len, Hidden_Dim）。
- **理解基础算子机制**：清楚缩放除以 $\sqrt{d_k}$ 对梯度数值稳定性的作用、Softmax 归一化流程，以及因果下三角掩码对前向计算的限制。
- **理解主流架构选型依据**：能从注意力机制的并行预训练、自回归解码的 KV 缓存复用机制等角度解释 Decoder-only 的工程取舍。

### 基础篇暂不展开的工程议题
- **大规模集群分布式并行**：张量并行（TP）、流水线并行（PP）以及 ZeRO 显存切分等分布式训练通信调度。
- **底层硬件加速算子**：FlashAttention 的 SRAM 显存分块读写优化与 Triton/CUDA 原生算子实现。
- **应用层编排系统**：提示词工程、向量检索与多 Agent 协作状态机等，相关内容见 [/frontier/agentic-rag/README.md](/frontier/agentic-rag/README.md) 与 [/architecture/README.md](/architecture/README.md)。

![基础计算机制与生产部署工程的职责边界对比](/images/content-rebuild/9770d2485b44acd3/fig-3.png)

---

## 4. 动手验证：用纯张量操作实现单头因果注意力

在阅读后续具体算子前，可以通过一段纯 PyTorch 代码验证因果自注意力的张量变换过程。以下示例设定为单头注意力（$d_k = d_{\text{model}}$）：

```python
import torch
import torch.nn.functional as F

# 设定示例维度超参数
batch_size = 2
seq_len = 4
d_k = 16  # 单头注意力特征维度

# 模拟经过线性投影后的 Query, Key, Value 张量
# 形状均为: [batch_size, seq_len, d_k]
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)

# 1. 缩放点积注意力分数: [B, T, d_k] x [B, d_k, T] -> [B, T, T]
# 除以 sqrt(d_k) 用于将点积后的方差缩放回接近 1，避免进入 Softmax 梯度饱和区
scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

# 2. 因果下三角掩码：仅允许当前词元关注自身及之前的词元，右上三角置为 -inf
mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
masked_scores = scores + mask

# 3. 归一化注意力权重与加权求和: [B, T, T] x [B, T, d_k] -> [B, T, d_k]
attn_weights = F.softmax(masked_scores, dim=-1)
output = torch.matmul(attn_weights, V)

# 验证输出形状与输入序列保持一致，满足残差连接要求
assert output.shape == (batch_size, seq_len, d_k)
print("验证通过！输出张量形状为:", output.shape)
```

输出张量与输入张量保持相同的维度，这一特性使得 Transformer 的隐藏层能够配合残差连接进行逐层堆叠。

---

## 关联导航

- 模块第 1 讲：[/llm-100/basics/大模型到底_大_在哪里_一文讲清参数_数据与算力.md](/llm-100/basics/大模型到底_大_在哪里_一文讲清参数_数据与算力.md)（三要素解析）
- 模块第 5 讲：[/llm-100/basics/Transformer_为什么能成为大模型的核心架构.md](/llm-100/basics/Transformer_为什么能成为大模型的核心架构.md)（Transformer 核心结构全景）
- 架构进阶：[/architecture/README.md](/architecture/README.md)（服务化网关与推理部署架构）
