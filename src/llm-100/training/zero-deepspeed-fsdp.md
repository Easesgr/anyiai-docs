---
title: ZeRO、DeepSpeed 与 FSDP 解决了什么问题？显存状态分片与通信代价拆解
icon: mdi:tune-vertical
tag:
- 训练原理
- 大模型100讲
- 预训练
description: 讲解分布式训练中 ZeRO-1/2/3 显存分片原理、集合通信算子拆解、通信量数学推导以及 DeepSpeed 与 PyTorch FSDP
  的工程落地差异。
---

在大模型分布式训练中，传统的分布式数据并行（Distributed Data Parallel, DDP）曾是最常见、吞吐最高的并行方式。在 DDP 中，多张 GPU 分别处理不同批次的数据，但每张卡都必须持有一份完整的模型状态。

随着模型参数量增长到百亿甚至千亿，单张显卡的物理显存再也装不下完整的模型状态。我们在[《大模型训练为什么这么吃显存？》](/llm-100/training/why-high-vram.md)中讨论过混合精度训练中静态状态的构成，在[《数据并行、模型并行、流水线并行分别是什么？》](/llm-100/training/parallel-strategies.md)中介绍了基础并行分类。

零冗余优化器（Zero Redundancy Optimizer, ZeRO）以及实现它的 DeepSpeed、PyTorch FSDP（Fully Sharded Data Parallel），核心目标就是：**保留数据并行的计算逻辑，同时打破“每张卡都必须存全量状态”的限制，把模型状态切散到各个 GPU 上存储，按需通信。**

![显存状态切分全景导览图](/images/content-rebuild/dd39b3a5cde2c005/cover.png)
*图释：分布式训练从传统 DDP 的全量冗余存储演进至 ZeRO 的逐级状态切分，以受控的通信开销打破单卡显存墙。*

---

## 传统数据并行的内存冗余瓶颈

以标准的混合精度训练（FP16/BF16 计算 + FP32 Adam 优化器）为例，设模型参数量为 $\Phi$。在反向传播和更新过程中，显卡需要常驻以下三类静态状态：

1. **模型参数（FP16/BF16）**：每个参数占 2 字节，合计 $2\Phi$ 字节；
2. **梯度（FP16/BF16）**：每个参数反向传播得到的局部梯度占 2 字节，合计 $2\Phi$ 字节；
3. **优化器状态（FP32 Adam）**：
   - FP32 主权重：$4\Phi$ 字节（用于高精度累加更新）；
   - 动量一阶矩：$4\Phi$ 字节；
   - 方差二阶矩：$4\Phi$ 字节；
   - 合计 $12\Phi$ 字节。

因此，单副本静态状态的显存开销固定为：
$$M_{\text{state}} = 2\Phi + 2\Phi + 12\Phi = 16\Phi \text{ 字节}$$

![传统 DDP 显存冗余机制示意图](/images/content-rebuild/dd39b3a5cde2c005/fig-1.png)
*图释：DDP 中每张 GPU 拥有完全相同的静态状态副本，通信只发生在梯度同步阶段，多卡之间存在极高的内存冗余。*

假设我们用 8 张 80GB 的 GPU 训练一个 13B（130 亿参数，$\Phi = 13 \times 10^9$）的模型：
- 单份静态模型状态大小为：$13 \times 10^9 \times 16 \text{ 字节} \approx 208\text{ GB}$；
- 集群的总显存虽然有 $8 \times 80\text{ GB} = 640\text{ GB}$，但经典 DDP 要求每一张卡都要存下这 208 GB，训练在第一步初始化时就会触发显存溢出（OOM）。

在经典 DDP 中，8 张卡上的 Adam 优化器每一步都在做完全相同的数值运算，并保留着 8 份一模一样的 FP32 主权重和动量状态。消除这种重复存储，就是显存状态切分的切入点。

<ReadMoreLock />

---

## ZeRO-1、ZeRO-2 与 ZeRO-3 的分级状态切分

ZeRO 将显存状态的切分划分为三个递进阶段：

![ZeRO 分级状态切分机制对比](/images/content-rebuild/dd39b3a5cde2c005/fig-2.png)
*图释：ZeRO-1 切分优化器状态，ZeRO-2 进一步切分梯度，ZeRO-3 将参数、梯度与优化器状态彻底分片存储。*

### 1. ZeRO-1：切分优化器状态（$P_{os}$）
在 ZeRO-1 中，所有 GPU 仍然保存完整的模型参数（$2\Phi$）和反向传播梯度（$2\Phi$）。但是，将 Adam 优化器维护的 $12\Phi$ 字节状态平均切分成 $N$ 份（$N$ 为参与并行的 GPU 数量）。第 $i$ 张显卡只维护第 $i$ 份优化器状态。

- **单卡显存**：$M_{\text{ZeRO-1}} = 2\Phi + 2\Phi + \frac{12\Phi}{N} = 4\Phi + \frac{12\Phi}{N}$
- **收益极限**：当卡数 $N$ 很大时，$\frac{12\Phi}{N} \to 0$，单卡静态显存降至约 $4\Phi$ 字节。相比原始的 $16\Phi$，显存节省上限约为 75%。

### 2. ZeRO-2：切分优化器状态 + 梯度（$P_{os+g}$）
反向传播时，每张卡虽然都需要计算前向产生的损失对参数的梯度，但更新时只有负责对应参数分片的卡才需要这份梯度。

因此在 ZeRO-2 中，梯度在反向传播计算出的同时，就通过算子规约并分散到目标卡上，每张卡最终只保留自己那份优化器状态所对应的梯度分片（大小为 $\frac{2\Phi}{N}$）。

- **单卡显存**：$M_{\text{ZeRO-2}} = 2\Phi + \frac{2\Phi}{N} + \frac{12\Phi}{N} = 2\Phi + \frac{14\Phi}{N}$
- **收益极限**：当卡数 $N$ 很大时，单卡静态显存降至约 $2\Phi$ 字节（仅保留 FP16 参数本身），相比原始 $16\Phi$ 节省上限约为 87.5%。

### 3. ZeRO-3：切分优化器状态 + 梯度 + 模型参数（$P_{os+g+p}$）
ZeRO-3 将参数本身也均分给 $N$ 张显卡，每张卡常驻保存的模型参数仅为 $\frac{2\Phi}{N}$ 字节。在空闲和常驻阶段，没有任何一张卡持有完整的模型参数。

- **单卡显存**：$M_{\text{ZeRO-3}} = \frac{2\Phi}{N} + \frac{2\Phi}{N} + \frac{12\Phi}{N} = \frac{16\Phi}{N}$
- **收益极限**：单卡静态显存与卡数 $N$ 严格成反比。理论上只要持续增加 GPU 数量，任意规模的模型都可以装入集群显存。

### 算例对照：7B 模型在 8 卡（8×80GB）下的静态显存
以参数量 $\Phi = 7 \times 10^9$（7B）为例，参数和梯度以 2 字节（FP16/BF16）计，Adam 优化器状态以 12 字节（FP32）计：

| 并行方案 | 参数显存 | 梯度显存 | 优化器状态显存 | 静态显存总计 | 相对 DDP 节省比例 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **传统 DDP** | $14\text{ GB}$ | $14\text{ GB}$ | $84\text{ GB}$ | $112\text{ GB}$ | 0%（单卡 80GB 无法装下） |
| **ZeRO-1** ($N=8$) | $14\text{ GB}$ | $14\text{ GB}$ | $10.5\text{ GB}$ | $38.5\text{ GB}$ | 65.6% |
| **ZeRO-2** ($N=8$) | $14\text{ GB}$ | $1.75\text{ GB}$ | $10.5\text{ GB}$ | $26.25\text{ GB}$ | 76.6% |
| **ZeRO-3** ($N=8$) | $1.75\text{ GB}$ | $1.75\text{ GB}$ | $10.5\text{ GB}$ | $14.0\text{ GB}$ | 87.5% |

> **注意**：上述数字仅包含**静态模型状态**。实际训练中还包含前向传播保存的中间层激活值（Activation）、通信缓冲区（Bucket）以及临时张量。如果不开启激活检查点（Activation Checkpointing），较长的训练上下文（例如 4096 或 8192）产生的激活值依然可能耗尽显存。

---

## 通信算子与开销分析：为什么 ZeRO-1/2 不增加通信量？

许多人直觉上认为：将状态切散之后，频繁通信一定会成倍推高网络传输量。但通过集合通信算子的数学分解可以发现：**ZeRO-1 和 ZeRO-2 的总通信量与传统 DDP 完全相等，只有 ZeRO-3 增加了 50% 的通信量。**

![集合通信算子转化与通信量拆解](/images/content-rebuild/dd39b3a5cde2c005/fig-3.png)
*图释：Ring All-Reduce 在数学上等价于 Reduce-Scatter 与 All-Gather 的串联。ZeRO-1/2 调整了算子执行阶段，通信量与 DDP 保持一致。*

### 1. 传统 DDP 的通信基准
在环形集合通信（Ring-based Collective Communication）中，设需要同步的张量大小为 $S$ 字节（对应 FP16 梯度的 $2\Phi$ 字节）。
- 一次环形 `Reduce-Scatter` 算子：各节点将分块梯度规约求和，每个节点的单向发送（与接收）数据量各为 $\frac{N-1}{N} S$；
- 一次环形 `All-Gather` 算子：各节点将各自持有的规约结果分发给所有节点，每个节点的单向发送（与接收）数据量各为 $\frac{N-1}{N} S$；
- 传统 DDP 使用的 `All-Reduce` 算子，在实现上等价于一次 `Reduce-Scatter` 接一次 `All-Gather`。以分布式系统通用的单向发送量统计，单卡总通信量为：
$$C_{\text{DDP}} = 2 \times \frac{N-1}{N} S \approx 2S \quad (N \gg 1)$$
以参数字节数 $S = 2\Phi$ 为基准，单卡在一个训练步内的通信量约为 $4\Phi$ 字节。

### 2. ZeRO-1 与 ZeRO-2 的通信过程
ZeRO-1 和 ZeRO-2 改变了算子执行的节点，但并未改变传输的总字节量：
1. **反向传播结束时**：不执行全量 `All-Reduce`，而是调用 `Reduce-Scatter` 算子对梯度求和。每个 rank 仅接收并保留自己负责更新的那部分梯度。单卡通信量为 $\frac{N-1}{N} S \approx 2\Phi$；
2. **优化器更新阶段**：各个 rank 分别利用本地梯度更新自己负责的那部分优化器状态与参数分片；
3. **更新完成后**：各 rank 调用 `All-Gather` 算子，将更新后的参数分片广播给集群内的其他节点，使每张卡在下一次前向前重新获得完整的 FP16 参数。单卡通信量为 $\frac{N-1}{N} S \approx 2\Phi$。

两个阶段累加：
$$C_{\text{ZeRO-1/2}} = \text{Reduce-Scatter} (2\Phi) + \text{All-Gather} (2\Phi) = 4\Phi$$
**结论**：ZeRO-1 与 ZeRO-2 只是将原本集中在反向末期的一次 `All-Reduce`，拆分成了更新前的 `Reduce-Scatter` 与更新后的 `All-Gather`，总通信量没有增加。

### 3. ZeRO-3 的通信代价推导
ZeRO-3 在常驻阶段不保留完整的参数副本，参数必须在计算过程中按需拉取并即时丢弃：

![ZeRO-3 动态通信与参数生命周期](/images/content-rebuild/dd39b3a5cde2c005/fig-4.png)
*图释：ZeRO-3 在前向与反向逐层按需拉取全量参数并立即释放，反向计算后通过 Reduce-Scatter 聚合梯度分片。*

1. **前向传播阶段**：当计算推进到第 $L$ 层时，各卡调用一次 `All-Gather`，从其他卡拉取该层的参数分片并拼装成完整参数。执行完前向计算后，**立刻释放非本地参数**。前向全模型累计拉取量为 $\frac{N-1}{N} S \approx 2\Phi$；
2. **反向传播阶段**：计算推进到第 $L$ 层反向时，由于前向拉取的参数已被释放，各卡必须再次调用 `All-Gather` 重新拉取该层参数来计算梯度。反向累计拉取量为 $\frac{N-1}{N} S \approx 2\Phi$；
3. **梯度同步阶段**：第 $L$ 层梯度算完后，调用 `Reduce-Scatter` 将梯度分片累加并分散到负责该分片的卡上，非本地梯度立刻释放。梯度同步累计量为 $\frac{N-1}{N} S \approx 2\Phi$；
4. **参数更新阶段**：各卡只更新本地维护的参数分片，无需跨卡通信。

总通信量核算：
$$C_{\text{ZeRO-3}} = 2\Phi (\text{前向拉取}) + 2\Phi (\text{反向拉取}) + 2\Phi (\text{梯度规约}) = 6\Phi$$

对比基准：
$$\frac{C_{\text{ZeRO-3}}}{C_{\text{DDP}}} = \frac{6\Phi}{4\Phi} = 1.5$$
**结论**：ZeRO-3 的总通信量相比 DDP 增加了 50%。在跨机带宽受限的环境中（例如缺少高速 RoCE/InfiniBand 网络，仅有普通千兆/万兆以太网），ZeRO-3 频繁的参数通信很容易成为整机吞吐的严重瓶颈。

---

## ZeRO-Offload：异构计算与显存权衡

在单张或少量显卡场景下，即使开启 ZeRO-2/3，显存可能依然不足以支撑超大模型。ZeRO-Offload 引入了 CPU 主存来分担压力。

![ZeRO-Offload 数据流与异构分工](/images/content-rebuild/dd39b3a5cde2c005/fig-5.png)
*图释：GPU 负责高算力消耗的前向与反向计算，CPU 负责承载大容量优化器状态并完成参数更新。*

### 1. 计算密集与内存密集的解耦
训练过程中不同环节的计算强度差异巨大：
- **前向与反向传播**：主要由矩阵乘法（GEMM）构成，计算量与批次大小、序列长度和模型参数量成正比，计算强度极高，必须在 GPU 上执行；
- **优化器更新（Adam）**：主要是逐元素向量运算，计算复杂度为 $O(\Phi)$，计算强度很低，但需要维护庞大的 FP32 状态（$12\Phi$ 字节）。

ZeRO-Offload 将计算强度低的优化器更新和状态卸载到主机 CPU 内存：
1. GPU 负责前向与反向计算；
2. 反向传播产出 FP16 梯度后，将其通过 PCIe 总线传给 CPU 内存；
3. CPU 内存驻留 FP32 优化器状态，利用 CPU 多核指令集执行 Adam 更新，生成新的 FP16 权重；
4. 新权重再通过 PCIe 回传给 GPU 进行下一步训练。

### 2. 通信隐藏与硬件瓶颈
PCIe 带宽（PCIe 4.0 x16 双向约 64 GB/s）远低于显存带宽（H100/A100 超过 1.5~3 TB/s）。为了避免 PCIe 传输导致 GPU 闲置，实现中通常利用 CUDA Stream 与异步拷贝将梯度上传、反向计算和权重下发进行流水线重叠。

**生效边界**：只有当 GPU 反向计算耗时大于或接近 PCIe 传输耗时时，数据搬运开销才能被有效掩盖。如果批次过小导致 GPU 计算速度极快，系统吞吐就会受制于 PCIe 传输和 CPU 优化器计算速度。

---

## 工程实现对比：DeepSpeed ZeRO 与 PyTorch FSDP

在当前大模型训练工程中，微软开源的 **DeepSpeed** 与 PyTorch 原生的 **FSDP** 是两套最常用的状态分片实现。

![DeepSpeed 与 PyTorch FSDP 架构实现对比](/images/content-rebuild/dd39b3a5cde2c005/fig-6.png)
*图释：DeepSpeed 采用配置化与全局引擎封装，FSDP 深度契合 PyTorch 的 Module 树状结构并支持细粒度嵌套封装。*

### 1. 核心架构与设计取向
- **DeepSpeed**：通过统一的 JSON 配置文件驱动，将模型、优化器和数据加载器打包封装至全局 Engine 中。DeepSpeed 倾向于将模型参数拉平为扁平张量（Flattened Tensor）来管理通信 Bucket。其生态包含 ZeRO-Offload、ZeRO-Infinity 以及流水线并行，开箱即用度高。
- **PyTorch FSDP**：深度集成在 PyTorch 的 `nn.Module` 层级体系中。它采用嵌套包装（Nested Wrapping）机制，以子模块（如单个 TransformerBlock）为单元建立分片。与 PyTorch 2.x 的 `torch.compile`、原生混合精度以及检查点机制集成更平滑。

### 2. 混合精度配置的常见陷阱

| 对比维度 | DeepSpeed (ZeRO-2/3) | PyTorch FSDP | 实践注意事项 |
| :--- | :--- | :--- | :--- |
| **主权重维护机制** | 优化器内部自动维护一套 FP32 精度的主权重，即使输入参数是 FP16/BF16。 | 由用户的 `MixedPrecision` 配置决定。若未显式指定，可能直接以参数类型跑优化器。 | 在 FSDP 中跑混合精度时，需显式配置 `reduce_dtype` 或保留 FP32 优化器主权重，防止精度下溢。 |
| **参数切分粒度** | 展平的大缓冲区（Flattened Buffer），按字节大小划界切分。 | 树状模块嵌套切分（Nested Module Wrapping），以配置的 Layer 为粒度。 | FSDP 必须配置合理的 `auto_wrap_policy`，若只包最外层，每次前向都会退化为拉取整个模型的参数。 |
| **通信与计算重叠** | 基于 Bucket 累积梯度，达到设定字节阈值触发一次规约通信。 | 通过 CUDA Event 与反向 Hook 挂钩计算图，支持 `backward_prefetch`。 | 在显存紧张时，FSDP 需调优 `limit_all_gathers=True`，防止通信预取过多块导致显存尖刺。 |

### 3. 可运行的 FSDP 配置样例
以下代码展示了如何在标准 PyTorch 环境中配置基于 Transformer 层的嵌套包装 FSDP（对应 ZeRO-3 策略）：

```python
import functools
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def setup_fsdp_model(model: nn.Module) -> FSDP:
    # 1. 配置混合精度：参数与计算使用 BF16，梯度通信累加保持 FP32 精度
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,       # 前向与反向中的张量精度
        reduce_dtype=torch.float32,       # 梯度 Reduce-Scatter 累积精度
        buffer_dtype=torch.bfloat16,      # 模块内部持久 buffer 精度
    )

    # 2. 配置嵌套包装策略：以单个 LlamaDecoderLayer 为单元进行分片
    # 这样前向与反向可以逐层拉取与释放参数，避免一次性拉取整个模型
    llama_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    # 3. 构造 FSDP 模型（FULL_SHARD 对应 ZeRO-3 全切分）
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=llama_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE, # 预取上一层参数以重叠通信与计算
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,                         # 限制前向预取数量，防止显存峰值溢出
    )
    return fsdp_model
```

---

## 检验自测

完成学习后，可通过以下问题检验对显存切分机制的掌握情况：

1. **状态跟踪**：在 ZeRO-3 下执行一次完整的前向与反向传播，某一个特定的 Transformer 层权重一共被全量拉取（All-Gather）了几次？分别在什么时候拉取？计算完后为什么需要立即释放？
2. **通信核算**：为什么说 ZeRO-1 和 ZeRO-2 相比经典 DDP 没有增加总通信量？增加的 50% 通信量具体发生在 ZeRO-3 的哪个阶段？
3. **架构分析**：在多机多卡训练中，若机内卡间为 NVLink（数百 GB/s），机间仅为普通千兆或万兆以太网，全集群启用 ZeRO-3 会遇到什么瓶颈？通常如何解决？

<details>
<summary>自测参考思路</summary>

1. **状态跟踪**：共被拉取 2 次。第一次在前向传播计算该层输出时拉取，算完后丢弃非本地分片以节省激活值并行的显存空间；第二次在反向传播计算该层输入梯度与权重梯度时重新拉取，梯度算完后通过 Reduce-Scatter 分发回目标卡并再次丢弃全量参数。
2. **通信核算**：经典 DDP 的 All-Reduce 在数学上等同于 Reduce-Scatter（梯度规约）加 All-Gather（梯度分发），单卡总传输量为 $2S$。ZeRO-1/2 只是将前向反向完成后的 Reduce-Scatter 与参数更新后的 All-Gather 分开执行，总量依旧为 $2S$。ZeRO-3 在前向拉取参数（$S$）、反向拉取参数（$S$）、反向规约梯度（$S$），总计 $3S$，相对 DDP 的 $2S$ 增加了 50% 的跨卡传输开销。
3. **架构分析**：ZeRO-3 逐层 All-Gather 和 Reduce-Scatter 会频繁跨节点传输大张量，低速机间网络将导致 GPU 核心长期处于通信等待状态，算力利用率（MFU）大幅下降。常见做法包括：采用混合切分（节点内采用 ZeRO-3，节点间保持 ZeRO-1 或传统数据并行，如 ZeRO++ / Hybrid Sharding），或结合流水线并行/张量并行来减少跨机通信频次。
</details>
