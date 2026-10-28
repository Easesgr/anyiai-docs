---
title: vLLM 是什么？PagedAttention 为什么能提升推理吞吐？
icon: mdi:speedometer
tag:
  - 模型部署面试
  - vLLM
  - PagedAttention
---

![](/images/model-deployment-interview/vllm-paged-attention_cover.png)

# vLLM 是什么？PagedAttention 为什么能提升推理吞吐？

这道题考的是 **vLLM 推理框架的设计原理**，核心是理解 PagedAttention 如何解决 LLM 推理中的内存瓶颈。面试官想看你不仅知道"是什么"，更要理解"为什么能快"。

---

## 1. vLLM 的定位：一句话理解

vLLM 是 UC Berkeley 开发的大模型推理框架，核心创新是 **PagedAttention** 技术。它的灵感来自操作系统虚拟内存的分页管理，可以让显存利用率大幅提升。

![vLLM 与传统推理框架的吞吐量对比柱状图](/images/model-deployment-interview/20260609_044957_776576_01.png)

数字说话：**相比 HuggingFace Transformers，vLLM 吞吐量最高提升 24 倍**。这不是微优化，是架构级的改变。

---
<ReadMoreLock />

## 2. LLM 推理的两阶段：为什么内存是瓶颈？

理解 vLLM 之前，得先搞清楚 LLM 推理卡在哪。

LLM 推理分两个阶段：

**Prefill 阶段**：处理输入 prompt，生成第一个 token。这个阶段可以并行计算，算力是瓶颈。类比一下，就像超市进货，一次性搬进来很多货。

**Decode 阶段**：逐个生成后续 token，每次生成都要读取之前所有 token 的 KV Cache。这个阶段是串行的，内存是瓶颈。就像顾客结账，一个一个排队，而且每个人结账时都要翻阅之前所有人的购物小票。

关键数据来了：**OPT-13B 模型，每个 token 的 KV Cache 占用约 208KB**。假设生成 1000 个 token，光缓存就要 200MB。一个 16GB 显存的 GPU 最多放多少个请求？算算就知道，内存根本不够用。

![Prefill 和 Decode 两阶段对比示意图](/images/model-deployment-interview/20260609_045003_648064_01.png)

---

## 3. 传统 KV Cache 的两大致命问题

传统方法管理 KV Cache，就像给每个学生发一本固定大小的笔记本——不管你写多少字，都用整本。这造成了两个致命问题：

**问题一：内存碎片化**

预分配连续内存会产生三种碎片：
- 内部碎片：分配的太大，用不完
- 外部碎片：分配的位置不连续，找不到足够大的连续空间
- 预留碎片：为了应对未来增长，先多预留一些

举个例子，你想存 100 个 token 的 KV Cache，系统可能直接给你分配能存 128 个 token 的连续空间。然后下次要存 105 个，128 不够了，只能找新的位置。一来二去，显存被切割得七零八落，利用率惨不忍睹。

**问题二：复杂推理场景无法共享**

现代 LLM 应用经常需要并行采样（一个 prompt 生成多个回答）、束搜索（ beam search 保留多个候选序列）、共享前缀（系统 prompt 大家都用）——这些场景下，不同序列的 KV Cache 其实有大量重叠。

传统方法怎么处理？**直接复制**。每个序列都有一份完整的 KV Cache，显存直接爆炸。

![传统连续内存分配的碎片化问题示意图](/images/model-deployment-interview/20260609_044958_173632_01.png)

---

## 4. PagedAttention：操作系统的智慧

PagedAttention 的核心思想很简单：**借鉴操作系统的虚拟内存分页管理**。

操作系统怎么解决内存碎片和共享问题？用页表。把逻辑地址和物理地址分开，物理内存可以离散存放，按需分配。

PagedAttention 做了三件事：

**第一，固定大小 block 分块存储**

不再一次性分配一大块连续内存，而是把 KV Cache 切成固定大小的 block。默认 16 个 token 一个 block。

**第二，Block table 维护映射关系**

每个序列有一个逻辑上的 block 列表（逻辑块），实际存储在物理显存上是另一套位置（物理块）。Block table 就是映射表，记录"逻辑块 0 → 物理块 3"、"逻辑块 1 → 物理块 7"。

这样逻辑上连续，物理上可以分散。内存不够了？随便找个空闲 block 塞进去，不用担心找不到连续空间。

**第三，Copy-on-write 支持共享**

多个序列可以共享同一个物理 block。比如系统 prompt 的 KV Cache，大家都一样，那就指向同一个物理块。

只有当某个序列要修改（比如要在 prefix 后面继续生成）时，才触发 copy-on-write：复制一份新的物理块给这个序列用。

类比一下，就像图书馆借书系统。书可以分散放在不同书架，通过索引目录查找。一本书可以同时被多人借阅，只有你要做笔记时才复印一本。

![Block table 逻辑块与物理块映射关系示意图](/images/model-deployment-interview/20260609_045022_781376_01.png)

---

## 面试怎么答

**基础版（能过的回答）**

> vLLM 是 UC Berkeley 开发的大模型推理框架，核心创新是 PagedAttention 技术。它借鉴了操作系统虚拟内存的分页管理思想，将 KV Cache 分块存储，通过 Block table 维护逻辑块与物理块的映射关系，实现非连续存储。这样做有两个好处：一是消除内存碎片，显存利用率大幅提升；二是支持 Copy-on-write，多个序列可以共享相同的 KV Cache，只有写入时才复制。实测相比 HuggingFace，吞吐量提升最高 24 倍。

**加分版（让面试官眼前一亮）**

> vLLM 的核心价值在于解决了 Decode 阶段的"内存墙"问题。Decode 阶段串行执行但 KV Cache 持续增长，传统连续内存分配会产生大量碎片。PagedAttention 用分块管理替代连续分配，物理块可以离散存放，Block table 实现逻辑到物理的映射。更妙的是 Copy-on-write 机制，让并行采样、束搜索、共享前缀等复杂场景的 KV Cache 共享成为可能。vLLM 还采用 FCFS 调度策略抢占最近到达的请求，通过 Swapping 或 Recomputation 处理 OOM。

---

## 一句话总结

**vLLM 通过 PagedAttention 的分块存储和 Block table 映射，消除了 KV Cache 的内存碎片，支持 KV Cache 共享，从而实现 24 倍吞吐量提升。**

![面试回答要点总结](/images/model-deployment-interview/20260609_045016_125056_01.png)