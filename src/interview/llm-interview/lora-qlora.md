---
title: LoRA 和 QLoRA 是什么？如何用 4-bit 权重微调大模型？
icon: mdi:memory
tag:
- 大模型面试
- LoRA
- QLoRA
description: 梳理 LoRA 与 QLoRA 的核心机制，详解 NF4 分位数分箱、双重量化显存计算、分页优化器与反量化数据流。
---

在了解 QLoRA 之前，建议先掌握 [LoRA 的数学原理与低秩分解机制](/interview/llm-interview/lora-math-principle.md)。标准 LoRA 虽将可训练参数比例缩减到 1% 以下，但基座模型的 16-bit 主干权重依然完整驻留显存。QLoRA（Quantized Low-Rank Adaptation）通过把冻结的基座权重压缩至 4-bit，使单张消费级显卡微调百亿参数模型成为可能。

## 30 秒口述结论

面试中被问到“LoRA 与 QLoRA 的区别以及 QLoRA 的实现原理”时，可以这样组织口述回答：

> **LoRA** 冻结原模型全量参数，在旁边挂载低秩分解矩阵 $A$ 和 $B$ 来学习增量，省去的是主干权重的梯度与优化器状态显存，但基座权重仍以 16-bit 浮点常驻显存。
>
> **QLoRA** 在 LoRA 基础上进一步将冻结的基座权重压缩为 4-bit，核心由三项技术构成：
> 1. **NF4（NormalFloat4）数据类型**：利用预训练权重接近正态分布的先验，按等概率分位数划分量化区间，比普通 INT4 更好地保留权重信息。
> 2. **双重量化（Double Quantization）**：对第一层分块量化的缩放因子再做一次 8-bit 量化，将缩放常数开销从每参数 0.5 bit 压缩到 0.127 bit，每参数净省约 0.37 bit。
> 3. **分页优化器（Paged Optimizers）**：依托 CUDA 统一内存机制，在长文本出现显存瞬时尖峰时，把暂时不参与计算的优化器状态换出到 CPU 内存，防止 OOM 崩溃。
>
> 在实际计算中，基座权重并非直接做 4-bit 矩阵乘，而是在前向传播中按需瞬时反量化为 16-bit 浮点完成 GEMM 计算，反向传播只求导并更新 16-bit 的 LoRA 适配器。

![QLoRA 核心机制概览：NF4 量化权重、双重量化与分页优化器协同工作](/images/content-rebuild/5fd447bb5e3bdbbb/cover.png)

---

## 贯穿例子：微调显存账本与降维策略

以微调一个典型的 70 亿参数（7B）LLaMA 模型为例，输入序列长度设为 2048。大模型训练过程中的显存开销主要由四部分构成：基座权重、梯度、优化器状态和前向激活值（具体核算可参考 [Transformer 显存占用分析](/interview/llm-interview/transformer-memory.md)）。

我们对比三种微调方案在同一任务下的静态显存底噪（不含动态激活值）：

1. **全量参数微调（Full SFT，FP16/BF16）**：
   - 模型权重（16-bit）：$7\times 10^9 \times 2 \text{ 字节} \approx 14\text{ GB}$
   - 梯度（16-bit）：$14\text{ GB}$
   - AdamW 优化器状态（一阶矩与二阶矩均为 FP32，每参数 8 字节）：$7\times 10^9 \times 8 \text{ 字节} \approx 56\text{ GB}$
   - 仅权重相关静态显存就达到 $14 + 14 + 56 = 84\text{ GB}$，必须依赖多卡张量并行或 ZeRO 分片。

2. **标准 LoRA（16-bit 基座 + 16-bit LoRA 适配器）**：
   - 基座权重（16-bit）：依然需要完整载入，占 **14 GB**。
   - 适配器权重（假设秩 $r=16$，可训练参数约占 0.2%，约 1400 万参数）：约 0.028 GB。
   - 适配器梯度与优化器状态：$14\times 10^6 \times (2 + 8) \text{ 字节} \approx 0.14\text{ GB}$。
   - 静态显存降到了约 **14.2 GB**。但加上批大小（Batch Size）带来的激活值后，单张 16GB 显卡（如 V100/T4）仍会溢出，24GB 显卡（如 RTX 3090/4090）也只能运行较小的批次。

3. **QLoRA（4-bit NF4 基座 + 16-bit LoRA 适配器）**：
   - 基座模型完全冻结，只参与前向特征提取和反向链式法则的矩阵乘，不更新也不存梯度。
   - 基座权重以 4-bit NF4 存放，14 GB 显存直接缩减至原来的约四分之一：$7\times 10^9 \times 0.5 \text{ 字节} \approx 3.5\text{ GB}$。
   - 加上量化缩放常数（约 0.11 GB）和 LoRA 适配器，静态显存仅需约 **3.7 GB**，在 16GB 甚至消费级 8GB/12GB 显卡上也能拉起训练。

![全量微调、标准 LoRA 与 QLoRA 在显存各组成部分的占用对比](/images/content-rebuild/5fd447bb5e3bdbbb/fig-1.png)

<ReadMoreLock />

---

## NF4 数据类型：基于正态分布分位数设计

在传统量化（如 INT4 线性均匀量化）中，数值范围被等间距切分为 16 个固定大小的区间。但深度学习预训练模型的权重张量并不服从均匀分布，而是高度符合均值为 0、方差为 $\sigma^2$ 的高斯正态分布 $W \sim \mathcal{N}(0, \sigma^2)$。

正态分布的特点是“两头稀疏、中间密集”。如果使用等间距划分：
- 绝大多数落在均值 0 附近的有效权重，会被挤压在中间极少数几个量化桶中，丢失大量细节精度；
- 出现概率极低的两端异常值，却白白占用了两翼的离散挡位。

NF4（4-bit NormalFloat）通过信息论中的**等概率分位数划分**来构造码本：每个量化区间内包含的概率质量（Area under curve）完全相等，使得每个量化离散值携带的信息熵达到最大。

设标准正态分布累积分布函数为 $\Phi(x)$，逆函数为分位数函数 $Q_X(p) = \Phi^{-1}(p)$。对于 4-bit（共 $2^4 = 16$ 个桶），理想情况下每个桶应占据 $\frac{1}{16}$ 的概率。

但神经网络计算中有一个特殊要求：**码本中必须存在一个精确的数值 0**。如果 0 存在量化偏差，对 Padding 填充项、零偏置以及稀疏矩阵的计算会引入系统性漂移。

为此，NF4 采用非对称切分算法构造 16 个点：
1. 在负数侧（概率区间 $[0, 0.5]$）估计 $2^{3} = 8$ 个正态分布分位数点，包含均值 0；
2. 在正数侧（概率区间 $[0.5, 1]$）估计 $2^{3} + 1 = 9$ 个正态分布分位数点，包含均值 0；
3. 负数侧与正数侧在 0 处重合，去重后恰好得到 $8 + 9 - 1 = 16$ 个离散点。

所有离散点在离线生成后，除以其绝对值最大值，归一化到 $[-1, 1]$ 范围内并固定为常数查找表：

```text
NF4 离散码本查找表（16 个常数点）：
[-1.0, -0.6961928, -0.5250730, -0.3949175, -0.2844414, -0.1845006, -0.0910500, 0.0,
  0.0795803,  0.1609302,  0.2447120,  0.3379152,  0.4407078,  0.5626170,  0.7229568, 1.0]
```

![NF4 数据类型分位数划分与线性均匀量化 INT4 的概率密度对比](/images/content-rebuild/5fd447bb5e3bdbbb/fig-2.png)

在将浮点权重映射为 NF4 时，只需计算权重属于哪一个分位数区间，并将其存储为对应的 4-bit 索引（0 到 15）。

---

## 双重量化：将缩放常数开销压缩到每参数 0.127 bit

为了避免权重矩阵中的局部异常大值破坏整张矩阵的量化精度，量化通常采用分块机制（Block-wise Quantization），即每隔 $B_1 = 64$ 个参数划分为一个小块，每个块维护一个独立的绝对值最大值作为缩放常数 $c_1$：
$$c_1 = \max(|W_{\text{block}}|)$$

如果直接保存 FP32 单精度浮点数作为 $c_1$，每个参数额外分摊的显存为：
$$\frac{32 \text{ bits}}{64} = 0.5 \text{ bit/parameter}$$
对于目标只有 4-bit 的压缩算法而言，0.5 bit 相当于额外增加了 12.5% 的体积。

**双重量化（Double Quantization, DQ）**的思路是对这些缩放常数本身再做一次量化：
1. **第一层量化**：以块大小 $B_1 = 64$ 将 16-bit 浮点权重压入 4-bit NF4，产出缩放常数数组 $c_1$（FP32 格式）。
2. **第二层量化**：把 $c_1$ 数组看作新的连续数据，以更大的块大小 $B_2 = 256$ 对其进行 8-bit FP8 量化。这一步产出 8-bit 的压缩缩放因子数组，以及第二层数量极少的 FP32 缩放常数 $c_2$。

![双重量化（Double Quantization）两级缩放常数压缩流程](/images/content-rebuild/5fd447bb5e3bdbbb/fig-3.png)

我们来核算双重量化后的每个参数实际缩放开销：
$$\text{Overhead}_{\text{DQ}} = \frac{8 \text{ bits}}{B_1} + \frac{32 \text{ bits}}{B_1 \times B_2} = \frac{8}{64} + \frac{32}{64 \times 256} = 0.125 + 0.00195 \approx 0.127 \text{ bit/parameter}$$

对比单层分块量化，每个参数节省：
$$0.5 \text{ bit} - 0.127 \text{ bit} = 0.373 \text{ bit/parameter}$$

以 65B 模型为例，这一步节省的显存量为：
$$65 \times 10^9 \times 0.373 \text{ bits} \approx 24.25 \times 10^9 \text{ bits} \approx 3.03 \text{ GB}$$
在单卡显存逼近物理极限时，省下的 3GB 显存往往直接决定训练能跑通还是爆显存。

---

## 分页优化器：应对显存瞬时突刺

大模型训练中的显存占用由静态和动态两部分组成。即便基座被压到了 4-bit，长序列输入在反向传播计算自注意力梯度时，瞬时激活值（Activation Memory）依然会产生峰值（Spike）。

在常规训练中，一旦瞬时总显存超过硬件物理上限，程序会立即报错 CUDA OOM 退出。然而，在反向传播的前中阶段，优化器状态（动量与方差）完全静止，直到最后执行 `optimizer.step()` 时才被读取。

QLoRA 借助 CUDA 统一内存（Unified Memory）实现**分页优化器（Paged Optimizers）**：
1. 将优化器状态内存分配在 CPU 内存与 GPU 显存共享的虚拟分页空间中。
2. **换出（Evict）**：当长序列激活值导致显存触顶时，驱动把暂时不用的优化器状态页面自动换出到 Host CPU 内存，为激活值让出空间。
3. **预取换入（Prefetch）**：当反向传播计算完毕、进入参数更新阶段时，再将对应的优化器页面通过 PCIe 预取回 GPU 显存执行更新。

![分页优化器在显存峰值突刺时通过 PCIe 调度优化器状态页面至 CPU 内存](/images/content-rebuild/5fd447bb5e3bdbbb/fig-4.png)

**使用前提与限制**：分页优化器仅用于“削峰”，用来承受偶尔出现的激活值突刺。如果模型本身的常驻参数就已经超出显存上限，会造成优化器页面在 CPU 和 GPU 之间高频频繁换入换出（Page Thrashing），PCIe 总线带宽会成为严重瓶颈，导致训练步长延迟暴增几十倍。

---

## 计算执行流程：前向瞬时反量化与反向 LoRA 更新

常见的认知误区是以为 QLoRA 在硬件底层开发了“4-bit 乘 16-bit”的专用矩阵乘法算子。实际上目前主流 GPU 的 Tensor Core 并不能直接计算这种非标精度乘法。

QLoRA 的执行逻辑是：**“显存中 4-bit 存储，计算时片上瞬时反量化（On-the-fly Dequantization）”**。

数学表达为：
$$Y = X \cdot \text{dequantize}(W^{\text{NF4}}) + \frac{\alpha}{r} (X A) B$$

以下为说明执行机制的伪代码：

```python
# 教学伪代码：展示 QLoRA 前向与反向张量数据流
import torch
import torch.nn as nn

def dequantize_nf4(weight_packed, scales_c1, scales_c2, b1=64, b2=256):
    """
    在 GPU 片上缓存中将 4-bit NF4 恢复为 16-bit 浮点
    1. 用 scales_c2 (FP32) 反量化 scales_c1 (FP8)，恢复第一层缩放系数 c1
    2. 读取 uint8 打包权重，拆分为两个 4-bit 索引
    3. 查表 NF4 码本并乘以对应的 c1，输出 BF16 精度矩阵
    """
    pass

class QLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=16, alpha=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        total_params = in_features * out_features
        
        # 1. 基座权重以 4-bit 打包存储（每字节存两个 4-bit 索引，显存占用 1/4）
        self.register_buffer("weight_nf4", torch.zeros(total_params // 2, dtype=torch.uint8))
        
        # 2. 双重量化缩放参数存储
        num_blocks_1 = total_params // 64
        num_blocks_2 = (num_blocks_1 + 256 - 1) // 256
        self.register_buffer("scales_c1_8bit", torch.zeros(num_blocks_1, dtype=torch.uint8))
        self.register_buffer("scales_c2_fp32", torch.zeros(num_blocks_2, dtype=torch.float32))
        
        # 3. LoRA 适配器使用可训练的 16-bit 浮点（BF16/FP16）
        self.lora_A = nn.Parameter(torch.randn(r, in_features, dtype=torch.bfloat16) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.bfloat16))
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 形状: [batch_size, seq_len, in_features]，类型: BF16
        
        # 步骤 1: 瞬时反量化（在显存读取阶段还原基座矩阵为 BF16）
        w_bf16 = dequantize_nf4(self.weight_nf4, self.scales_c1_8bit, self.scales_c2_fp32)
        base_out = torch.matmul(x, w_bf16.view(self.out_features, self.in_features).t())
        
        # 步骤 2: 计算 LoRA 旁路增量分支
        lora_out = torch.matmul(torch.matmul(x, self.lora_A.t()), self.lora_B.t()) * self.scaling
        
        # 步骤 3: 累加输出
        return base_out + lora_out
```

![QLoRA 前向反量化计算与反向传播仅计算 LoRA 梯度的张量执行流](/images/content-rebuild/5fd447bb5e3bdbbb/fig-5.png)

在反向传播阶段：
1. 误差梯度回传至当前层输出 $\nabla Y$；
2. 仅对 LoRA 分支的 $A$ 和 $B$ 求梯度：$\nabla B = (X A)^T \nabla Y$，$\nabla A = X^T (\nabla Y B^T)$；
3. 对基座权重 $W$ **完全不计算梯度，也不分配优化器状态**；
4. 当需要把梯度继续向前一层回传时，基座分支需将 $W$ 临时反量化以计算 $\nabla X_{\text{base}} = \nabla Y W^T$，与 LoRA 旁路回传的梯度累加得到传递给前一层的完整梯度 $\nabla X$。

---

## 面试追问：QLoRA 显存变小了，训练速度会变快还是变慢？

**结论：在相同硬件与批大小下，QLoRA 的单步耗时（Step Time）普遍比标准 LoRA 慢约 20% 到 40%。**

原因并非算力不足，而是访存与编解码开销导致的：
1. **前向与反向的反量化算子开销**：每次遇到全连接层，都需要先启动 CUDA 解码内核，把打包的 4-bit 索引转换为 16-bit 浮点数，再送进 Tensor Core。这增加了算子调度延迟（Kernel Launch Latency）与片上计算负担。
2. **小批次下访存未能完全掩盖解码**：虽然读取 4-bit 节省了显存带宽传输量，但将 4-bit 索引查表反量化的指令运算开销，在 Batch Size 较小时往往大于显存带宽节省带来的收益。
3. **分页换出换入的潜在阻塞**：如果显存波动触发了 Paged Optimizer，CPU-GPU 间的 PCIe 传输时间会进一步拉长单步耗时。

![LoRA 与 QLoRA 单步耗时时间线对比：反量化开销与内联解码延迟](/images/content-rebuild/5fd447bb5e3bdbbb/fig-6.png)

**工程上的权衡视角**：虽然单步时间变长，但因为 QLoRA 节约了大量显存，工程上可以在单卡上开出更大（甚至成倍）的 Batch Size，或者通过更长上下文进行训练。在衡量整体吞吐量（Tokens per Second）与硬件成本时，需结合具体设备综合评估。更多关于微调参数的选择与配比，可参考 [LoRA 超参数应该怎么设置？有什么经验法则？](/interview/llm-interview/lora-hyperparams.md)。

---

## 常见误区与方案选型

| 对比维度 | 全量参数微调 (Full SFT) | 标准 LoRA (16-bit) | QLoRA (4-bit NF4) |
| :--- | :--- | :--- | :--- |
| **基座权重物理存储** | FP16 / BF16 (16-bit) | FP16 / BF16 (16-bit) | **NF4 (4-bit)** |
| **适配器参数格式** | 无（全量更新） | FP16 / BF16 | FP16 / BF16 |
| **优化器状态覆盖** | 全量参数（每参 8 字节） | 仅覆盖 LoRA 适配器 | 仅覆盖 LoRA 适配器（支持分页） |
| **7B 模型基座静态显存** | ~14 GB | ~14 GB | **~3.5 GB - 3.8 GB** |
| **单步运算耗时** | 较快（标准 GEMM） | 极快（参数少，无反量化） | **较慢**（需执行瞬时解码） |
| **单卡可行性（24GB 显卡）** | 跑不动 7B（需多卡） | 跑 7B 较充裕，13B 吃紧 | **7B/13B 轻松运行，70B 借助多卡运行** |
| **最终权重合并交付** | 原生模型权重 | $\Delta W = \frac{\alpha}{r} B A$ 直接加回基座 | 可反量化基座后相加输出 16-bit 权重 |

### 常见误区排查

- **误区 1：QLoRA 训练出来的模型只能在 4-bit 下推理？**
  不对。训练完成后，LoRA 适配器保存的是 16-bit 的权重参数。你可以选择直接反量化基座模型，把 $\frac{\alpha}{r} B A$ 累加回原 FP16 矩阵，导出成标准 16-bit 浮点权重，供 vLLM、TGI 等任意推理引擎部署。
- **误区 2：NF4 会改变基座模型权重的数值分布？**
  不会。NF4 只是量化码本的设计方案，依据的是权重原本就接近高斯分布的客观事实，通过等概率分位数把权重近似映射到 16 个点上，并不改变原模型的分布规律。
- **误区 3：开启分页优化器后显存就永远不会 OOM？**
  不对。分页优化器只能换出优化器状态，模型基座权重、LoRA 参数和前向反向的中间激活值依然必须常驻在显存中。如果单个样本输入序列过长导致激活值本身撑爆显存，分页优化器无法阻止 OOM，此时需要结合梯度检查点（Gradient Checkpointing）或序列切分。

学习了低显存微调的基本机制后，如需继续了解适配层具体的超参数调节与微调配置策略，可阅读 [LoRA 超参数应该怎么设置？有什么经验法则？](/interview/llm-interview/lora-hyperparams.md)。
