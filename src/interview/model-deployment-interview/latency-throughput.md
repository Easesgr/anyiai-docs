---
title: 大模型推理中的首 Token 延迟和平均生成速度分别受什么影响？
icon: mdi:timer-outline
tag:
  - 模型部署面试
  - 首Token延迟
  - 生成速度
---

![](/images/model-deployment-interview/latency-throughput_cover.png)

# 大模型推理中的首 Token 延迟和平均生成速度分别受什么影响？

这道题考的是你对大模型推理两个核心指标的理解。**首 Token 延迟（TTFT）** 反映的是用户等待"第一个字出现"要多久，**平均生成速度** 反映的是模型"持续输出"有多快。两者背后的原理完全不同——一个受计算能力限制，一个受内存带宽限制。

## 1. 先搞懂两个指标——什么是 TTFT 和吞吐量

### 什么是首 Token 延迟（TTFT）

TTFT 全称 Time To First Token，就是从你发出请求到模型吐出第一个字的时间。你在 ChatGPT 输入问题后、看到"正在思考"到第一个字出现之间的等待，就是 TTFT。

### 什么是平均生成速度

也叫吞吐量（Throughput），是模型生成后续 token 的速度，通常用 tokens/second 来衡量。这个指标决定了你等待完整回答的总时长。

### 打个比方

就像去餐厅吃饭：
- **TTFT** = 等第一道菜上桌的时间
- **平均生成速度** = 后续所有菜上桌的节奏

第一道菜等了 10 分钟，你会觉得这家店真慢。但等菜节奏稳定，每 30 秒上一道，1 小时后你还是吃到了很多菜。推理也一样——TTFT 和吞吐量有时候需要 trade-off。

![TTFT 从请求到首 Token 的时间 vs 吞吐量的斜率对比](/images/model-deployment-interview/20260609_051116_993600_01.png)
<ReadMoreLock />

## 2. Prefill vs Generation——两个阶段的本质差异

### Prefill 阶段：计算受限

Prefill 阶段处理的是用户的全部输入 prompt。它需要：
1. 把输入的所有 token 都过一遍 attention
2. 算出 KV 缓存
3. 生成第一个 token

**关键点**：输入的所有 token 可以并行计算！就像食堂一次性炒 100 份宫保鸡丁，大锅一起炒，效率极高。

这个阶段是**计算受限（Compute-bound）** 的——GPU 的算力不够用，而不是数据传输跟不上。

### Generation 阶段：内存带宽受限

Generation 阶段是自回归生成下一个 token。它需要：
1. 读取全部模型权重（从 HBM 到芯片）
2. 读取当前所有 KV 缓存
3. 计算下一个 token

**关键点**：一次只能生成 1 个 token，每次都要读全部权重。想象食堂只能一份一份地出菜，每份都要从冷库搬食材。

这个阶段是**内存带宽受限（Memory-bandwidth-bound）** 的——GPU 大部分时间在等数据，而不是在计算。

### 临界 batch size

两个阶段的分界线大约在 **batch size = 240 tokens**。低于这个值，内存带宽是瓶颈；高于这个值，计算成为瓶颈。

![Prefill 并行计算 vs Generation 逐个自回归生成对比](/images/model-deployment-interview/20260609_051120_994112_01.png)

## 3. 谁拖慢了首 Token 延迟？

TTFT 主要受 Prefill 阶段影响。以下几个因素决定了你等第一字要多久：

### 模型规模

模型越大，TTFT 越高。7B 模型光权重就 14GB，要全部加载到芯片上做计算。模型翻倍，TTFT 基本上也会翻倍。

### 输入序列长度

用户输入的 token 数越多，Prefill 要算的矩阵越大。输入长度从 512 变到 2048，TTFT 可能变成 4 倍。

### Batch size

这里有个反直觉的点——**batch size 越小，TTFT 可能越高**。因为小 batch 没充分利用 GPU 的并行能力，GPU 算力被浪费了。

### 算力

H100 的 TTFT 肯定比 A100 快，因为计算能力更强。这个阶段是算力为王。

### 量化能帮上忙吗？

int8 量化可以降低 TTFT，因为它减少了要计算的数据量。但这只是 trade-off，不是一味的好事。

![影响首 Token 延迟的因素：模型规模、序列长度、算力、batch size](/images/model-deployment-interview/20260609_051130_919744_01.png)

## 4. 谁拖慢了平均生成速度？

吞吐量主要受 Generation 阶段影响。这个阶段的瓶颈是内存带宽，不是算力。

### 内存带宽是王道

每次生成一个 token，模型需要从 HBM 读取全部权重（7B 模型 ≈ 14GB）。HBM 带宽就那么多，大部分时间 GPU 在等数据，而不是在计算。

### KV 缓存的访问开销

KV 缓存随生成的 token 数量线性增长。虽然单次访问量不大，但累积起来也是开销。

### Attention 计算的坑

生成每个 token 都要做一次 attention，涉及当前 token 和所有历史 token 的矩阵运算。序列越长，attention 越慢。

**Flash Attention** 牛的地方在于：避免了把大矩阵完整物化到 HBM，减少了读写次数，直接提升吞吐量。

### 为什么是内存带宽受限？

因为：
- 每次要读 14GB 权重（固定开销）
- 计算量很小（就一个 token）
- GPU 等数据的时间 >> 计算时间

打个比方：就像在图书馆借书，你需要从很远的大书架取书（读权重），然后翻开看一眼（计算），再决定下一本取哪本。大部分时间在跑腿，而不是在看书。

![Generation 阶段内存访问路径：权重 + KV 缓存从 HBM 读取](/images/model-deployment-interview/20260609_051134_568960_01.png)

## 5. 实战优化——怎么同时优化两者？

### Prefill/Generation 分离部署

工业界的主流方案是把 Prefill 和 Generation 拆开，部署在不同机器上：
- Prefill 节点：堆算力，专注计算
- Generation 节点：堆带宽，专注内存

就像餐厅专业化——大厨专门炒菜，传菜员专门上菜，而不是一个人又炒又端。

### 连续批处理（Continuous Batching）

当一个请求生成完毕，立刻塞一个新请求进来。这样 GPU 不空闲，利用率拉满。Orca 论文提出的技术，现在是标配。

### Paged Attention

VLLM 的核心技术。类比操作系统的分页管理，KV 缓存不再连续存储，减少碎片，显存利用率大幅提升。

### GQA / MQA

Query 多、KV 少？让多个 query 头共享一份 KV 缓存。内存占用骤降，吞吐量飞升。

### 量化

int8 量化减少内存带宽压力，改善延迟-吞吐 trade-off。**但注意**：它不会提升最大吞吐量，只是让曲线更平滑。

### 实际部署推荐组合

**Prefill/Generation 分离 + Continuous Batching + Flash Attention + Paged Attention + GQA + Interleaved（多个 Prefill token 插一个 Generation token）**

Interleaved 配置比简单的 Prefill-Generation 交替更好，因为能更均匀地分配负载。

![Prefill/Generation 分布式解耦部署架构图](/images/model-deployment-interview/20260609_051136_571200_01.png)

## 面试怎么答？

### 基础版（直接背）

首 Token 延迟由 Prefill 阶段决定，这个阶段处理完整输入提示词，是计算受限的，主要受模型规模、输入序列长度和 GPU 算力影响。平均生成速度由 Generation 阶段决定，这个阶段自回归生成 token，是内存带宽受限的，主要受 HBM 带宽、KV 缓存大小和 Attention 实现方式影响。两者的本质区别在于：Prefill 可以批量并行计算，Generation 只能逐个生成。

### 加分版（多背几句）

可以补充：临界 batch size 大约 240 tokens，此时 Prefill 的计算瓶颈开始凸显。工业界最优方案是 Prefill/Generation 分布式解耦部署。Flash Attention 通过避免大矩阵物化减少 HBM 读写，GQA/MQA 通过共享 KV 缓存降低内存占用。int8 量化能改善延迟-吞吐折中，但不提升最大吞吐量。

## 一句话总结

**首 Token 延迟看计算能力（Prefill），平均生成速度看内存带宽（Generation），两者瓶颈不同的根本原因是并行度差异。**