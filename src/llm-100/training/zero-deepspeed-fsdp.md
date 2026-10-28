---
title: "ZeRO、DeepSpeed、FSDP 解决了什么问题？"
icon: mdi:tune-vertical
tag:
  - 训练原理
  - 大模型100讲
  - 预训练
---

# ZeRO、DeepSpeed、FSDP 解决了什么问题？ - 大模型分布式训练内存优化三剑客

> 一句话核心：ZeRO 通过分片消除数据并行中的冗余状态，DeepSpeed 是微软的 ZeRO 实现，FSDP 是 PyTorch 原生的 ZeRO 实现，三者共同解决「万亿参数模型装不进单卡」的内存墙问题。

## 核心概念（术语表）

- **ZeRO（Zero Redundancy Optimizer）**：微软提出的内存优化算法，通过分片模型状态（参数/梯度/优化器状态）到各 GPU，消除数据并行中的冗余存储
- **DeepSpeed**：微软开源的深度学习优化库，实现了 ZeRO 算法，是当前最主流的 ZeRO 生产级实现
- **FSDP（Fully Sharded Data Parallel）**：PyTorch 1.11+ 原生实现的 ZeRO，代码与 PyTorch 深度集成，HuggingFace Accelerate 支持双后端切换
- **ZeRO-DP**：ZeRO 针对「模型状态」的优化方案，分 ZeRO-1/2/3 三个阶段，显存节省分别为 4倍/8倍/N倍
- **ZeRO-R**：ZeRO 针对「剩余状态」（激活值、临时缓冲区、内存碎片）的优化方案
- **混合精度训练**：前向/反向用 fp16/bf16 计算加速，更新时转回 fp32 精度保护权重
- **内存墙（Memory Wall）**：GPU 显存增长速度远落后于模型参数增长，单卡装不下大模型的现象
- **ALL-Gather**：分布式通信原语，所有 GPU 收集彼此的分片数据
- **Reduce-Scatter**：分布式通信原语，将梯度聚合后分发到各 GPU


<ReadMoreLock />## 历史背景 / 来源

ZeRO 由微软 DeepSpeed 团队提出，核心论文分三版演进：

- **ZeRO v1（2020）**：Memory Optimizations Toward Training Trillion Parameter Models，解决万亿参数训练的显存瓶颈
- **ZeRO-Offload（2021）**：Democratizing Billion-Scale Model Training，将部分数据 offload 到 CPU，使单卡可训练十亿级模型
- **ZeRO-Infinity（2021）**：Breaking the GPU Memory Wall for Extreme Scale Deep Learning，进一步将参数/梯度/优化器卸载到 NVMe，实现 GPU+CPU+NVMe 异构训练

DeepSpeed 作为 ZeRO 的工程实现，被 GPT-3（1750亿参数）、Bloom（1760亿参数）等超大模型采用。FSDP 则由 PyTorch 在 2021 年引入，作为 PyTorch 原生的 ZeRO 实现。

## 工作原理 / 核心机制

### 整体思路（一句话）
数据并行中每个 GPU 都存有完整的模型状态（参数、梯度、优化器状态），ZeRO 将这三类状态分片到不同 GPU，实现「显存随 GPU 数量线性下降」。

### 内存消耗数学模型
假设模型参数量为 Φ（读作 phi），训练中显存占用主要来自：

| 状态类型 | 单精度(fp32) 显存 | 半精度(fp16) 显存 |
|---------|-----------------|------------------|
| 优化器状态（Adam 一阶/二阶矩） | 8Φ | 8Φ |
| 梯度 | 2Φ | 2Φ |
| 模型参数 | 4Φ | 2Φ |
| **总计** | **16Φ** | **12Φ** |

即单卡训练 Φ 参数模型，至少需要 12-16Φ 字节显存。70亿参数模型（7B）需要约 84-112GB，远超单卡。

### 核心步骤详解

**第一步：ZeRO-1 优化器状态分片**

- 输入：全量模型参数 W（16Φ 字节）、全量优化器状态 O（8Φ 字节）
- 处理：将优化器状态按 GPU 数量 N 分成 N 份，每 GPU 只存 1/N
- 输出：单卡显存从 16Φ 降至 (4Φ + 4Φ/N) ≈ 4Φ
- 通信：ALL-Reduce 聚合梯度，单卡通信量 = 2Φ

**第二步：ZeRO-2 梯度也分片**

- 输入：全量梯度 G（2Φ 字节）
- 处理：反向传播后立即执行 Reduce-Scatter，每 GPU 只保留自己负责的 1/N 梯度
- 输出：单卡显存降至 (2Φ + 2Φ/N) ≈ 2Φ
- 通信：使用 Reduce-Scatter（1Φ）+ ALL-Gather（1Φ），总通信量仍为 2Φ

**第三步：ZeRO-3 参数也分片**

- 输入：全量参数 W（4Φ 字节）
- 处理：前向/反向计算前 ALL-Gather 获取完整参数，计算后丢弃不属于本 GPU 的部分
- 输出：单卡显存 = (4Φ/N)，即显存与 GPU 数量呈线性关系
- 通信：前向 1Φ + 反向 1Φ + 梯度 Reduce-Scatter 1Φ，总通信量 = 3Φ

## 关键知识点

1. **ZeRO-1/2/3 显存节省比例**：4倍 / 8倍 / N倍（N=GPU数量），这是论文原文数据
2. **ZeRO-1/2 通信量不增加**：始终为 2Φ，ZeRO-3 通信量增加 50%至 3Φ
3. **DeepSpeed 始终保持 fp32 主权重**：即使模型用 bf16 加载，内部会创建 fp32 参数用于优化器更新
4. **FSDP 默认用 torch_dtype 创建参数**：优化器在 bf16 精度执行，而非强制上转 fp32
5. **混合精度经验法则**：可训参数（优化器状态）应保持 fp32，前向/反向可用低精度
6. **ZeRO-R 三策略**：分区激活检查点 + 恒定缓冲区 + 内存碎片整理
7. **ZeRO-Offload 划分**：GPU 负责前向/反向（计算密集），CPU 负责优化器（内存密集）
8. **ZeRO-Infinity 支持 NVMe**：参数可 offload 到 SSD，单卡也能训练百亿参数模型
9. **HuggingFace Accelerate 0.30.0**：对齐了 FSDP 和 DeepSpeed 的精度行为，可无缝切换
10. **DeepSpeed ZeRO3 vs FSDP 性能**：IBM Granite 7B 模型 + 4×A100 + batch size 8，两者 MFU 和每秒词元数相近
11. **FSDP 两种模式**：低精度模式（全部 bf16）和混合精度模式（准备/优化器 fp32）
12. **DeepSpeed 内存开销**：少量 GPU 时，上转 fp32 可导致显存增加高达 2 倍
13. **FSDP 更灵活**：支持用户以低精度操作 PyTorch 优化器，不强制上转
14. **3D Parallel**：DeepSpeed 支持数据并行 + 模型并行 + 流水线并行的组合
15. **ALL-Gather vs Reduce-Scatter**：ZeRO-1 用 ALL-Reduce（全量），ZeRO-2/3 用 Reduce-Scatter（分片）

## 应用场景

- **场景1**：Meta 训练 LLaMA-65B，使用 DeepSpeed ZeRO-3 + 8×80GB A100，单卡显存从约 1TB 降至约 128GB
- **场景2**：BigScience 训练 BLOOM-176B，使用 DeepSpeed + ZeRO-3 + 384×80GB A100，耗时约 3.5 个月
- **场景3**：IBM Granite 7B 模型吞吐量测试，4×A100 + batch size 8，对比 FSDP 和 DeepSpeed ZeRO3，MFU 和每秒词元数相近
- **场景4**：单机单卡场景，用 ZeRO-Offload 将优化器 offload 到 CPU，单张 32GB V100 可微调 70亿参数模型
- **场景5**：HuggingFace Accelerate 集成 DeepSpeed 和 FSDP 双后端，用户可通过配置一键切换，无需改代码

## 常见误区 / 踩坑

- ❌ **误区1**：很多人以为「DeepSpeed 和 FSDP 完全等价，可以随意切换」
  ✅ **正解**：两者在精度处理上存在差异——DeepSpeed 强制将优化器参数上转为 fp32，FSDP 默认用 torch_dtype。直接切换可能导致训练不收敛。解决方案：在 HuggingFace Accelerate 0.30.0+ 中启用 FSDP 混合精度模式

- ❌ **误区2**：认为「ZeRO-3 一定比 ZeRO-1/2 好」
  ✅ **正解**：ZeRO-3 显存节省最多，但通信量增加 50%。在 GPU 间带宽有限（如多机互联）时，ZeRO-1/2 可能反而更快。选型原则：带宽高选 ZeRO-3，带宽低选 ZeRO-1/2

- ❌ **误区3**：FSDP 损失不收敛时，只调大学习率
  ✅ **正解**：问题可能出在精度设置。DeepSpeed 内部执行 fp32 上转，同样的学习率下可收敛；FSDP 若用 bf16 优化器则可能不收敛。正确做法：启用 FSDP 混合精度模式，或将学习率按 GPU 数量缩放

- ❌ **误区4**：以为「ZeRO-Offload 只卸载到 CPU」
  ✅ **正解**：ZeRO-Infinity 支持卸载到 NVMe SSD，可突破 CPU 内存瓶颈。但数据迁移开销会显著增加训练时间

- ❌ **误区5**：在少量 GPU（1-2张）上使用 ZeRO-3
  ✅ **正解**：ZeRO-3 的优势在于 N 很大时。1-2 张 GPU 时 ALL-Gather 通信占比高，可能反而不如简单数据并行。少量 GPU 推荐 ZeRO-1/2

- ❌ **误区6**：混合精度训练时，所有参数都用 fp16/bf16
  ✅ **正解**：混合精度经验法则要求「可训参数保持 fp32」，即优化器状态必须用 fp32。前向/反向计算可用低精度，但权重更新必须用 fp32

## 性能 / 复杂度

- **时间复杂度**：O(Φ) — 前向/反向计算本身与模型大小线性，ZeRO-3 增加 50% 通信开销
- **空间复杂度**：O(Φ/N) — ZeRO-3 下显存与 GPU 数量 N 成反比
- **通信复杂度**：O(Φ) — ZeRO-1/2 通信量 = 2Φ，ZeRO-3 = 3Φ

**与替代方案对比**：

| 方案 | 显存 | 通信量 | 适用场景 |
|------|------|--------|----------|
| 普通数据并行 (DDP) | 12-16Φ | 2Φ | Φ < 单卡容量 |
| ZeRO-1 | 4Φ | 2Φ | 多 GPU + 中等模型 |
| ZeRO-2 | 2Φ | 2Φ | 多 GPU + 中等模型 |
| ZeRO-3 | Φ/N | 3Φ | 多 GPU + 超大模型 |
| 模型并行 (Megatron) | Φ/N | 高 | 超大模型 + NVLink |

**临界点**：
- Φ < 单卡容量：普通 DDP 即可
- Φ > 单卡容量：选 ZeRO-1/2（通信不增）
- Φ >> 单卡容量：选 ZeRO-3（显存线性下降）
- 多机场景：先评估 NVLink/InfiniBand 带宽，低带宽优先 ZeRO-1/2

**性能数字**：
- DeepSpeed ZeRO-3 训练 OPT-175B：使用 400×80GB A100，MFU 约 52%
- FSDP vs DeepSpeed 吞吐对比：IBM Granite 7B + 4×A100，两者每秒词元数相近
- ZeRO-Infinity 单卡训练百亿参数：相比纯 GPU 训练，训练速度降低 10-100 倍，但突破显存限制

## 与相关概念的区别

### vs 模型并行（Tensor Parallel / Pipeline Parallel）

- **内存维度**：模型并行将单层参数拆分到多卡，ZeRO 将参数/梯度/优化器状态分片到各卡
- **通信维度**：模型并行通信发生在层间（前向/反向），ZeRO 通信发生在 All-Gather 时
- **粒度维度**：模型并行粒度细（单层），ZeRO 粒度粗（整个模型）
- **怎么选**：单层参数超单卡选模型并行，整个模型超单卡选 ZeRO；超大模型两者结合（3D Parallel）

### vs 梯度累积（Gradient Accumulation）

- **原理维度**：梯度累积在逻辑 batch 层面工作，不减少物理显存占用；ZeRO 从根本上减少每个 GPU 的存储
- **兼容性维度**：梯度累积可与 ZeRO 叠加使用，进一步放大有效 batch size
- **通信维度**：梯度累积不增加通信；ZeRO-3 增加通信
- **怎么选**：batch size 不足时用梯度累积；显存不足时用 ZeRO

### vs CPU Offload（PyTorch DeepSpeed Offload）

- **显存维度**：CPU Offload 将数据卸载到 CPU 内存，彻底释放 GPU 显存；ZeRO 仅分片不卸载
- **速度维度**：Offload 引入 PCIe 带宽瓶颈，训练速度可能降低数倍；ZeRO 通信可被计算覆盖
- **精度维度**：两者都支持混合精度，DeepSpeed Offload 同样保持 fp32 主权重
- **怎么选**：显存严重不足且训练时间充裕选 Offload；显存勉强够选 ZeRO

## 进阶 / 面试加分项

- **最新进展**：DeepSpeed 已支持 ZeRO-3 + NVMe Offload 的组合（ZeRO-Infinity），Meta 最新训练框架完全基于此；PyTorch FSDP2（完全分片优化）正在开发中，将提供更细粒度的控制
- **业界争议**：FSDP 和 DeepSpeed 在小规模（<8 GPU）场景下的最优配置尚无定论；HuggingFace Accelerate 虽已对齐行为，但用户仍需理解底层差异才能正确调参
- **一句话送给候选人**：ZeRO 的本质是「用通信换内存」，记住 ZeRO-1/2 不增加通信、ZeRO-3 增加 50% 这两个数字，面试就能答对 80% 的问题


## 面试如何回答


### 🟢 ZeRO 是什么？它解决了什么问题？

**回答要点**：

ZeRO（Zero Redundancy Optimizer）是微软提出的显存优化算法，解决「大模型装不进单卡」的问题。数据并行中每个 GPU 都存有完整的模型状态（参数、梯度、优化器状态），ZeRO 将这三类状态分片到不同 GPU，消除冗余存储。核心原理：用通信换内存。具体来说，ZeRO-1 分片优化器状态，显存节省 4 倍；ZeRO-2 额外分片梯度，节省 8 倍；ZeRO-3 再分片参数，显存随 GPU 数量线性下降。举例：训练 70 亿参数模型，普通 DDP 需要约 84GB 显存，ZeRO-3 用 8 张 GPU 时每卡仅需约 10.5GB。


### 🟡 ZeRO-1、ZeRO-2、ZeRO-3 的区别是什么？请从显存节省、通信量、适用场景三个维度回答。

**回答要点**：

三个阶段的分片粒度不同：

1. **显存节省**：ZeRO-1 分片优化器状态，节省约 4 倍；ZeRO-2 额外分片梯度，节省约 8 倍；ZeRO-3 再分片模型参数，显存与 GPU 数量 N 呈线性关系，可节省 N 倍。

2. **通信量**：ZeRO-1/2 的通信量 = 2Φ（与普通 DDP 相同），因为优化器和梯度的通信可被计算覆盖；ZeRO-3 通信量 = 3Φ，增加 50%，因为前向和反向各需要一次 ALL-Gather 收集参数。

3. **适用场景**：ZeRO-1/2 适合多 GPU + 中等模型（参数量级在几十亿），因为通信不增加；ZeRO-3 适合超大模型（参数量级在百亿以上），当显存不足且 GPU 间带宽充足时使用。简单记忆：显存不够选 ZeRO-3，通信带宽不够选 ZeRO-1/2。


### 🟡 DeepSpeed 和 FSDP 有什么区别？为什么同样的配置一个能收敛、一个不能？

**回答要点**：

DeepSpeed 是微软的 ZeRO 实现，FSDP 是 PyTorch 原生的 ZeRO 实现，两者在精度处理上存在关键差异。

DeepSpeed 的设计哲学是「始终将主权重保持为 fp32」，即使模型用 bf16 加载，内部也会创建 fp32 优化器状态。FSDP 默认使用 torch_dtype 创建参数，优化器可以在 bf16 精度执行。

这导致：当学习率相同时，DeepSpeed 的 fp32 优化器可以收敛，而 FSDP 的 bf16 优化器可能不收敛。解决方案有两个：一是将 FSDP 的学习率按 GPU 数量缩放（4 GPU 则学习率×4）；二是启用 FSDP 混合精度模式（HuggingFace Accelerate 0.30.0+ 已对齐行为）。

另一个差异：少量 GPU 时，DeepSpeed 的全量 fp32 上转可能使显存增加高达 2 倍，而 FSDP 更灵活，支持用户以低精度操作优化器。


### 🟡 什么是混合精度训练？为什么可训参数必须保持 fp32？

**回答要点**：

混合精度训练是一种利用低精度加速计算、同时保护模型精度的技术。具体做法：准备两套模型状态——fp32 类型的优化器状态和模型参数，以及 fp16/bf16 类型的计算参数。在前向传播和反向传播时使用 fp16/bf16 进行计算（前向/反向是计算密集，低精度可加速）；在参数更新时，将梯度与学习率相乘后更新到 fp32 类型的模型状态上。

为什么要保持可训参数为 fp32？因为 fp32 的动态范围和精度远高于 fp16/bf16。以 fp16 为例：指数位只有 5 位，尾数位只有 10 位，累加多个小梯度时会出现「舍入误差」和「下溢」问题，导致训练不收敛。而 fp32 有 8 位指数和 23 位尾数，精度高出约 1000 倍。所以经验法则：可训参数（优化器状态）必须保持 fp32，前向/反向计算可用低精度。


### 🔴 ZeRO-Offload 和 ZeRO-Infinity 解决了什么问题？它们与 ZeRO-3 的关系是什么？

**回答要点**：

ZeRO-Offload 和 ZeRO-Infinity 解决的是「即使分片也装不下」的问题，即单 GPU 显存或单台机器内存仍然不足的场景。

ZeRO-Offload 的核心思路是「计算密集型放 GPU，内存密集型放 CPU」。将优化器状态和参数更新卸载到 CPU，GPU 负责前向和反向传播。这样单张 32GB V100 也能微调 70 亿参数模型。代价是 CPU-GPU 通信增加训练时间。

ZeRO-Infinity 更进一步，支持将参数、梯度、优化器卸载到 NVMe SSD，实现 GPU+CPU+NVMe 三级异构存储。当 GPU 显存和 CPU 内存都不够时，可以借助 SSD 的容量。代价是 SSD 带宽远低于内存和显存，训练速度可能降低 10-100 倍。

三者关系：ZeRO-3 是基础（GPU 内分片），ZeRO-Offload 是 CPU offload（内存扩展），ZeRO-Infinity 是 NVMe offload（容量扩展）。实际使用中常组合：ZeRO-3 + Offload 以在有限硬件上训练超大模型。


### 🔴 在 8 张 GPU 上训练一个 130 亿参数模型，请给出 ZeRO 阶段选择和理由。

**回答要点**：

先算显存需求：130 亿参数（13B）按 fp16 计算约 26GB（2字节/参数），加上梯度 26GB、优化器状态 104GB（Adam 需要 8 字节/参数），总计约 156GB。8 张 80GB A100 总显存 640GB，理论上每卡 80GB 足够，但实际还有激活值、临时缓冲区开销。

**推荐配置：ZeRO-2 + 激活值 checkpointing**。

理由：
1. ZeRO-2 将显存降至约 52GB/卡（优化器 13GB + 梯度 13GB + 参数 26GB），相比 ZeRO-3 的 ALL-Gather 开销更小
2. 启用激活值 checkpointing（ZeRO-R 的一部分），将激活值从显存转移到计算（约 50% 显存节省）
3. 如果显存仍不足，再切换到 ZeRO-3

为什么不直接用 ZeRO-3？ZeRO-3 的通信量比 ZeRO-2 多 50%（3Φ vs 2Φ），在前向和反向传播中各需要一次 ALL-Gather。130 亿参数模型的分片通信量很大，如果 GPU 间带宽不足（如多机场景），可能反而更慢。ZeRO-2 在通信和显存之间取得平衡。


### 🟢 FSDP 和 DDP（DistributedDataParallel）有什么区别？

**回答要点**：

DDP（DistributedDataParallel）是最基础的数据并行，每个 GPU 都保存完整的模型副本，通过 All-Reduce 同步梯度。FSDP（Fully Sharded Data Parallel）本质上是 DDP + ZeRO-3 的结合，在 DDP 的基础上增加了参数分片。

核心区别：
1. **显存占用**：DDP 每 GPU 存完整模型，FSDP 每 GPU 只存 1/N（N=GPU数量）
2. **通信模式**：DDP 用 All-Reduce 同步梯度（每 GPU 收到完整梯度），FSDP 用 Reduce-Scatter 聚合梯度后分发（每 GPU 只收到自己负责的部分）
3. **通信量**：FSDP ZeRO-3 的参数 All-Gather 通信量比 DDP 多约 50%

简单记忆：DDP 是「存全量、多通信」，FSDP 是「存分片、换通信」。当模型能装进单卡时用 DDP（简单高效）；当模型装不下时用 FSDP（以通信换内存）。FSDP 在 PyTorch 1.11+ 作为 torch.distributed.fsdp 模块提供。


### 🟡 训练 1000 亿参数模型，需要多少 GPU？ZeRO 能帮我们节省多少？

**回答要点**：

先算显存需求：1000 亿参数（100B），fp16 存储需要 200GB 梯度，加上 fp32 优化器状态 800GB，仅模型状态就约 1TB。加上激活值、中间计算，保守估计需要 1.5-2TB 显存。

**不用 ZeRO**：单卡肯定不行。假设用 8×80GB A100 = 640GB，总显存不足。

**用 ZeRO-3 + 8 GPU**：每卡显存 = (200GB 梯度 + 800GB 优化器 + 200GB 参数) / 8 = 150GB，仍超 80GB。

**用 ZeRO-3 + 16 GPU**：每卡 = 1TB / 16 ≈ 62.5GB，刚好够。

**用 ZeRO-3 + 32 GPU**：每卡 = 1TB / 32 ≈ 31.25GB，非常充裕。

总结：1000 亿参数模型，至少需要 16 张 80GB GPU 才能用 ZeRO-3 装下。如果只有 8 张 GPU，则需要额外启用 ZeRO-Offload 将优化器卸载到 CPU，或使用 ZeRO-Infinity 卸载到 NVMe。ZeRO 的价值：让显存需求从 O(Φ) 降到 O(Φ/N)，使得万亿参数模型的训练从「不可能」变成「可能但需要很多卡」。

