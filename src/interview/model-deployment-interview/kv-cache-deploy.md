---
title: KV Cache 在模型部署中有什么作用？
icon: mdi:memory
tag:
  - 模型部署面试
  - KV Cache
  - 显存
---

![](/images/model-deployment-interview/kv-cache-deploy_cover.png)

# KV Cache 在模型部署中有什么作用？为什么它会占用大量显存？

这是一道模型部署的高频面试题，**核心考的是你对 LLM 推理流程的理解，以及为什么 KV Cache 是用显存换速度的关键技术**。搞懂这个，你对大模型推理优化的理解就上了一个台阶。

---

## KV Cache 是什么？——用显存换速度的推理神器

先说个生活场景。你写一篇长文章，写到一半想引用前面的内容，你会从头再读一遍自己写的吗？不会，你会直接翻到前面看。LLM 生成文本也是这个道理。

**KV Cache 的本质就是"缓存"**。在自注意力机制里，每个 token 都需要和前面所有 token 做注意力计算。生成第 N 个 token 时，需要计算它和前面 N-1 个 token 的关系。如果没有缓存，每次都要重新算一遍历史 token 的 K（Key）和 V（Value）矩阵，浪费得要命。

有了 KV Cache，我们把前面所有 token 的 K/V 矩阵提前算好存起来。新 token 进来只需要：
- 算自己的 Q（Query）、K、V
- 从缓存里拿出历史的 K/V
- 做一次注意力计算

**这样就把"重算历史"变成了"查表"，速度直接起飞。**

代价呢？这些 K/V 矩阵得存在显存里，这就是为什么 KV Cache 会吃显存。

![无KV Cache vs 有KV Cache的attention计算流程对比图](/images/model-deployment-interview/20260609_044400_258688_01.png)

---
<ReadMoreLock />

## Prefill 和 Decode——KV Cache 如何在两阶段配合

LLM 生成文本分成两个阶段：**Prefill（预填充）和 Decode（解码）**。

**Prefill 阶段**：你输入一个 prompt，比如"请写一首关于春天的诗"。模型一次性处理整个 prompt，把每个 token 的 K/V 都算出来存好。这个阶段是**并行计算**，速度很快。

**Decode 阶段**：模型开始一个字一个字地生成。每生成一个新 token，只需要：
1. 用新 token 的 Q，去查缓存里的 K/V
2. 做注意力计算
3. 生成下一个 token

这就是为什么你感觉 ChatGPT 输出是一个字一个字蹦出来的——因为 Decode 阶段是**逐个 token 生成**的，每步都要查缓存、算一次前向传播。

类比一下：**Prefill 就像老师先通读一遍全班试卷建缓存，Decode 就是老师逐题批改，只看学生新写的答案，不用每次都从头看一遍试卷。**

![LLM生成两阶段时序图，标注Prefill和Decode分界](/images/model-deployment-interview/20260609_044411_355584_01.png)

---

## 为什么 KV Cache 吃显存？——三重因素叠加

说到显存占用，很多人第一反应是"模型权重"。但 KV Cache 的显存需求有时候比权重还离谱，尤其是长对话场景。

**原因有三个，看完你就理解为什么显存不够用了：**

**第一，每个 token 都要存 K/V 矩阵。** 你对话 1000 轮，历史 1000 个 token 的 K/V 都得存着。

**第二，多头注意力把存储放大 N 倍。** 假设模型有 32 个注意力头，每个头的 K/V 矩阵都要存。这意味着存储量直接乘以 32。

**第三，序列长度线性增长。** 你对话越长，需要存的 token 就越多，显存占用呈线性增长，没有上界。

给你算个账：70B 参数的模型，用 FP16精度，假设 40 层、32 个注意力头、每个头 128 维。如果你有 2048 个 token 的上下文，KV Cache 大约需要 160GB 显存。而模型权重本身才 140GB。**缓存比权重还大**，这就是长上下文场景下的噩梦。

类比一下：**KV Cache 就像书架，每个 token 是一本书，多头注意力是每个 token 要复印 N 份，长对话就是书架无限续杯地往上加层。**

![KV Cache显存占用构成因素分解图](/images/model-deployment-interview/20260609_044401_790528_01.png)

---

## 显存优化三板斧——MQA/GQA、PagedAttention、量化

别慌，业界已经有很多优化手段了。

**第一板斧：MQA 和 GQA——减少"副本"数量**

MQA（Multi-Query Attention）让所有注意力头共享一套 K/V 矩阵，不再每个头都存一份。GQA（Grouped-Query Attention）是折中方案，把注意力头分成几组，组内共享 K/V。

效果：**显存减少 30%-50%**，推理速度还能保持不错的水平。现在 Llama 3、Mistral 这些主流模型都在用 GQA。

**第二板斧：PagedAttention——消除"碎片"**

传统方式预分配一块连续显存，对话一长就容易产生内存碎片——前面释放了 100 tokens，后面又来了 50 tokens，中间那块空着浪费。

vLLM 提出的 PagedAttention 借鉴了操作系统内存管理的思路：**把显存切成固定大小的 Block（类似内存页），动态分配，按需释放。** 这样显存利用率直接从 20%-30% 飙升到 80%+，能支持的并发数翻好几倍。

![vLLM PagedAttention Block管理机制示意图](/images/model-deployment-interview/20260609_044427_370880_01.png)

**第三板斧：量化——缩小"体积"**

模型权重可以量化（FP16 → INT8 → INT4），KV Cache 也可以。但 KV Cache 量化更敏感，因为它直接影响生成质量。

常用方案：**K/V 量化到 INT8**，或者更激进的 **FP8**。有些工作还会对 KV Cache 做非均匀量化——高频部分用高精度，低频部分用低精度。

---

## 面试怎么答

**基础版（直接背）：**

> KV Cache 的作用是缓存历史 token 的 Key/Value 矩阵，避免重复计算。LLM 推理分两个阶段：Prefill 阶段并行处理 prompt 并建立缓存，Decode 阶段逐个生成 token 并复用缓存。
>
> KV Cache 占用显存主要有三个原因：每个 token 都要存 K/V、多头注意力把存储放大 N 倍、序列长度线性增长。优化方向有三个：MQA/GQA 减少共享副本、PagedAttention 消除内存碎片、量化缩小存储体积。

**加分版（能画图、能推导）：**

> 如果面试官追问，你可以补充：没有 KV Cache 的情况下，生成 N 个 token 的计算复杂度是 O(N³)（因为每步都要重新算所有历史 token 的 attention）；有了 KV Cache 可以降到 O(N²)。
>
> vLLM 的 PagedAttention 借鉴了操作系统的分页思想，把显存切成 4KB 的 Block，通过 Block Table 做虚拟地址映射，支持 KV Cache 的动态分配和共享。显存利用率能从 20% 提升到 80% 以上。

---

## 一句话总结

**KV Cache 是用显存换速度的核心技术，通过缓存历史 token 的 K/V 矩阵避免重复计算，但每个 token 都要存 K/V、多头注意力放大 N 倍、长序列线性增长，这三重因素导致显存占用爆炸，优化方向是 MQA/GQA 共享、PagedAttention 分块、量化压缩。**