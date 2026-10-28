---
title: Tensor Parallel、Pipeline Parallel 和 Data Parallel 有什么区别？
icon: mdi:source-branch
tag:
  - 模型部署面试
  - 并行策略
  - 分布式
---

![](/images/model-deployment-interview/monitoring-cost_cover.png)

# Tensor Parallel、Pipeline Parallel 和 Data Parallel 有什么区别？

这题是大模型训练中的必考题，核心考的就是你对并行策略的理解。三个概念名字很像，但它们"切"的东西完全不同——**Data Parallel 切数据，Pipeline Parallel 切层，Tensor Parallel 切矩阵**。记住这句话，你就抓住了这道题的精髓。

---
## 板块1：它们切的东西不一样

先把这个核心区别刻进脑子里：

- **Data Parallel（DP）**：切数据 batch。每个 GPU 都存一份完整模型，拿到不同的数据分片，各自算完再同步梯度。
- **Pipeline Parallel（PP）**：切模型层。把模型按层分组，比如 1-20 层放 GPU0，21-40 层放 GPU1，流水线执行。
- **Tensor Parallel（TP）**：切层内部。一个大矩阵运算拆成多份，比如矩阵乘法 A×B，把 B 按列切分，多卡并行算再汇总。

用三个厨师做菜来类比一下：

- **DP** 就像多个厨师每人做一整道菜，你做红烧肉我做清蒸鱼，最后上桌就行。
- **PP** 就像流水线上，你洗菜我切菜他炒菜，一道大菜分环节完成。
- **TP** 就像做佛跳墙，要把炖汤、备料、装盅这些步骤拆开，多人同时处理同一道菜的不同部位。

![三厨师分工对比图：DP每人做完整菜、PP流水线分工、TP单菜拆步合作](/images/model-deployment-interview/20260609_051904_929600_01.png)

---
<ReadMoreLock />

## 板块2：通信频率决定你选哪个

这三种并行策略的通信频率差别很大，直接影响你能不能"省钱"。

**Tensor Parallel 通信最频繁**。每次层计算都要 AllReduce 操作，多卡算完立刻要汇总结果。就像打桥牌，每出一轮都要商量。

**Data Parallel 和 Pipeline Parallel 通信相对低频**。基本是一个训练 step 结束才同步一次梯度。像是开会讨论，阶段成果出来才碰个头。

为什么这个重要？因为通信开销直接决定了你需要什么硬件。TP 要求高速互联（NVLink 那种），不然带宽会卡死你。DP 和 PP 对通信带宽的要求就没那么苛刻。

![通信时序对比图：展示DP/PP在step末通信、TP在每次层计算都通信的时序差异](/images/model-deployment-interview/20260609_051854_232064_01.png)

---

## 板块3：内存不够？选对并行能省一大半

面试官爱问这个，因为能看出你到底理不理解这些并行策略解决了什么问题。

**Data Parallel 的内存问题**：每个 GPU 都要存完整模型。70B 的模型，光模型参数就要 140GB 显存（fp16），至少两张 80GB 的卡。这还是不算 optimizer 状态的版本。

**Pipeline Parallel 的内存优势**：模型按层分片，n 张卡就约等于 1/n 的模型内存。70B 模型分到 8 卡，每卡不到 10GB 模型参数。但 PP 有流水线气泡（bubble）问题，GPU 有时候在闲着等数据。

**Tensor Parallel 的内存优势**：针对单层大矩阵。大模型里那些注意力层、前馈层的矩阵非常大，单卡算不动也存不下。TP 把大矩阵拆开，每卡只算一部分。内存需求约等于单层/n。

用搬家来类比：

- **DP** 是复印多份完整家具给多个人搬——东西还是那些东西，只是人多了。
- **PP** 是把家具拆成客厅组、卧室组，不同人搬不同区域。
- **TP** 是把大沙发拆成零件分给多人带，每个人只拿一部分。

![内存分布对比图：展示三种并行下各GPU的内存占用和模型分片情况](/images/model-deployment-interview/20260609_051908_268224_01.png)

---

## 板块4：什么场景选什么并行

**模型能放得下，想加速训练？** → 用 Data Parallel。最简单直接，多搞几张卡就能线性提升吞吐量。

**模型太大单卡放不下？** → 用 Pipeline Parallel。把层分到多卡，降低单卡显存压力。实际训练大模型基本都用到这个。

**单层矩阵太大单卡算不了？** → 用 Tensor Parallel。注意力层的 QKV 矩阵、MLP 层的大矩阵，拆开算才能跑起来。

但现实是，**大模型训练通常是三种并行组合使用**。比如：

- **Megatron-LM**：TP + DP。每个模型副本内部用 TP 切矩阵，多个副本用 DP 跑数据并行。
- **DeepSpeed ZeRO**：在 DP 基础上又加了分片策略，配合 PP 使用。
- **PyTorch FSDP**：更灵活的 DP 变体，通信和计算可以 overlap。

这也是为什么你在看 LLaMA、Falcon 这些大模型训练代码时会发现，它们往往同时用好几种并行策略。单靠一种，很难训得动千亿参数的模型。

![场景选择决策图：展示根据模型大小、层大小、显存条件的并行策略选择路径](/images/model-deployment-interview/20260609_051918_609728_01.png)

---

## 面试怎么答

### 基础版（能过的回答）

> Data Parallel、Pipeline Parallel 和 Tensor Parallel 的核心区别是切分维度不同。Data Parallel 切数据 batch，每个 GPU 存完整模型副本；Pipeline Parallel 切模型层，不同 GPU 处理不同阶段；Tensor Parallel 切层内部矩阵，把大矩阵运算拆分到多卡。通信频率上，TP 最高（每次层计算都要 AllReduce），DP 和 PP 相对低频（一个 step 同步一次）。内存占用上，DP 需要完整模型显存，PP 约为 1/n，TP 约为单层/n。实际选择要看场景：模型能放下用 DP 加速训练，模型太大用 PP 分层，单层矩阵太大用 TP 拆分。大模型训练通常是三种组合使用。

### 加分版（让面试官眼前一亮）

> 如果从内存视角看，DP 显存需求 = 1×模型，PP ≈ 模型/n，TP ≈ 单层/n。这里要注意 PP 的流水线气泡（bubble）问题，microbatches 划分可以有效缓解。实际训练千亿参数模型，比如 DeepSpeed 和 Megatron-LM，都是 TP + PP + DP 三者结合：TP 解决单层算力问题，PP 解决模型太大放不下的问题，DP 解决数据吞吐量问题。还有一点很关键：TP 对通信带宽要求很高，必须用 NVLink 这类高速互联，否则带宽会成为瓶颈。

---

## 一句话总结

三种并行的本质区别是**切分维度不同**：DP 切数据、PP 切层、TP 切矩阵，组合使用才能训得动大模型。