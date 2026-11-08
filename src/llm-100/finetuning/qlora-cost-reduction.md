---
title: QLoRA 如何进一步降低微调显存？NF4、双重量化与分页优化器详解
icon: mdi:tune-vertical
tag:
- 微调与对齐
- 大模型100讲
- RLHF
description: 详解 QLoRA 的核心机制：NF4 正态分位数数据类型、两级双重量化与 CUDA 统一内存分页优化器，解析微调显存构成与动态反量化计算流。
---

低秩适配（LoRA）通过冻结预训练基座、仅训练低秩分解矩阵，将大模型微调时的可训练参数量和优化器状态压缩了两个数量级。然而在单卡或消费级显卡部署中，开发者很快会遇到下一道硬件门槛：**即使可训练参数不到千分之一，以 16-bit 浮点数常驻显存的预训练基座权重依然占据了绝大部分物理空间**。

以 70B 参数的基座模型为例，在 FP16 或 BF16 精度下，仅存储基座权重就需要占用：

$$70 \times 10^9 \times 2 \text{ 字节} \approx 140 \text{ GB}$$

这意味着微调还没开始，仅加载模型就需要至少两张 80GB 的 GPU。QLoRA（Quantized Low-Rank Adaptation）正是为了打破这一物理限制而提出的一套工程方案。它通过 **NF4（NormalFloat4）数据类型**、**双重量化（Double Quantization）** 和 **分页优化器（Paged Optimizers）** 三项核心技术，将基座模型压缩到 4-bit 显存驻留，使得单张 48GB 显卡（或双卡 24GB）即可稳定启动 70B 级别大模型的微调任务。

![QLoRA 显存优化核心架构总览](/images/content-rebuild/67a366302ff30f1e/cover.png)

---

### 学习目标

1. 精确核算 LoRA 与 QLoRA 微调时的物理显存构成，理解基座权重为何是最大开销瓶颈。
2. 理解预训练权重的正态分布特性，推导 NF4 数据类型的等分位数划分与零点对齐机制。
3. 计算双重量化两级压缩的显存开销，掌握标度因子从 0.5 bit/参数降至 0.127 bit/参数的数学推导。
4. 掌握分页优化器利用 CUDA 统一内存置换显存瞬时峰值（Spikes）的运作过程与适用边界。
5. 理清前向与反向传播中的动态反量化（Dequantization）计算流，理解 QLoRA“以算力换显存”的时间开销权衡。
6. 运用 Hugging Face `transformers`、`peft` 与 `bitsandbytes` 完成 QLoRA 最小微调实验并验证底层状态。

---

### 显存构成的精确拆解：LoRA 的显存瓶颈在哪里？

在单张 GPU 上对因果语言模型进行自回归微调训练时，总显存开销由以下六部分构成：

1. **基座权重（Base Model Weights, $W_0$）**：模型原始参数，在训练中完全冻结。对于参数量为 $N$ 的模型，以 16-bit 浮点（FP16/BF16）载入时，固定占用 $2N$ 字节。
2. **适配器权重（LoRA Weights, $A$ 与 $B$）**：低秩增量矩阵。设隐藏维度为 $d$，秩为 $r$，单个线性层的参数量从 $d \times d$ 缩减为 $2 \times d \times r$。在常见配置下，适配器参数量通常仅占基座总参数量的 $0.1\% \sim 0.5\%$。
3. **适配器梯度（Adapter Gradients）**：反向传播时仅为可训练的适配器参数计算并存储梯度，基座不产生梯度，开销与适配器参数量成正比（FP16 下为 $2 \times$ 适配器参数量）。
4. **优化器状态（Optimizer States）**：以标准 AdamW 为例，优化器需要为每个可训练参数维护一阶动量（FP32，4 字节）与二阶动量（FP32，4 字节），通常还需要维护 FP32 的参数备份（4 字节），每个可训练参数额外消耗 8 至 12 字节。由于基座冻结，优化器状态仅针对 LoRA 增量参数生效。
5. **前向激活值（Activations）**：前向传播过程中各层输出的特征张量，用于反向传播计算链式法则。大小与批次大小（Batch Size）、序列长度（Sequence Length）以及注意力头数成正比。即使启用激活检查点（Activation Checkpointing），长文本下的瞬时开销依然显著。
6. **运行时临时缓冲区（Workspace & Buffers）**：CUDA 算子调度、通信张量以及临时反量化所需的瞬时显存。

![微调显存构成对比](/images/content-rebuild/67a366302ff30f1e/fig-1.png)

从上述拆解可以看出，标准 LoRA 已经将梯度和优化器状态从全参数微调时的数百吉字节压低到了百兆至数吉字节区间。此时，**未被量化的基座权重在总显存中的占比超过了 75% 甚至 85%**。基座权重成为单卡微调无法逾越的物理高墙。

| 微调方案 | 70B 基座参数显存 | 适配器参数与梯度 | 适配器优化器状态 (AdamW) | 激活与临时缓冲 | 70B 模型最低显存门槛 (估算) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **全参数微调 (16-bit)** | $\approx 140 \text{ GB}$ | 包含在基座中 | $\approx 560 \sim 840 \text{ GB}$ (全量维护) | 较高 (依赖激活检查点) | $\approx 800\sim 1000 \text{ GB}$ (需多卡并行集群) |
| **标准 LoRA (16-bit 基座)** | $\approx 140 \text{ GB}$ | $\approx 0.4 \sim 0.8 \text{ GB}$ | $\approx 1.6 \sim 3.2 \text{ GB}$ | 取决于批次和序列长度 | $\approx 150 \sim 160 \text{ GB}$ (需两张 80GB 卡) |
| **QLoRA (4-bit 基座)** | $\approx 35 \sim 38 \text{ GB}$ | $\approx 0.4 \sim 0.8 \text{ GB}$ | $\approx 1.6 \sim 3.2 \text{ GB}$ (支持分页换出) | 增加瞬时反量化解包缓冲 | $\approx 42 \sim 48 \text{ GB}$ (单张 48GB 或双张 24GB) |

<ReadMoreLock />

---

### NF4 数据类型：基于正态分布的高保真分位数编解码

把权重从 16-bit 压缩到 4-bit，最直接的方式是均匀量化（Uniform INT4）。然而，直接采用 INT4 会导致语言模型微调后的文本生成质量显著退化。QLoRA 针对预训练权重的分布特性，提出了专有的 **NF4（NormalFloat4）** 数据类型。

#### 为什么均匀量化会导致权重严重失真？

经过充分预训练并采用权重衰减（Weight Decay）等正则化手段的大语言模型，其各层权重矩阵在统计上高度符合**以 0 为均值的正态分布**：$W \sim \mathcal{N}(0, \sigma^2)$。

这意味着：
- 绝大多数权重数值高度集中在 0 附近的极小区间内。
- 两侧绝对值较大的数值属于稀疏的长尾（Outliers）。

若采用均匀 INT4 量化，即把 $[-1, 1]$ 等距划分为 16 个网格区间：
1. **中心密集区信息大量丢失**：大量取值微小但语义关键的权重被合并映射到同一个离散整数上，产生巨大的量化误差方差。
2. **边缘稀疏区编码浪费**：很少出现权重的外围尾部区域被分配了同等数量的离散编码槽位，浪费了仅有的 4-bit（共 16 个）离散表示能力。

![NF4 正态分位数与均匀 INT4 量化对比](/images/content-rebuild/67a366302ff30f1e/fig-2.png)

#### 信息论视角下的等分位数构建

根据香农信息论，在给定比特容量的离散量化信道中，当且仅当**每个离散量化点被命中的概率相等**时，离散符号的熵达到最大，量化所带来的信息损失理论上最小。对于已知分布的数据，最优标量量化器要求相邻网格点之间的累积分布函数（CDF）积分面积严格相等。

设标准正态分布的累积分布函数为 $\Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-\frac{t^2}{2}} dt$，其逆函数（分位数函数）为 $Q_X(p) = \Phi^{-1}(p)$。对于 4-bit 量化，我们需要选出 16 个离散点 $q_i$。理想等分位数切分公式为：

$$q_i = \frac{1}{2} \left( Q_X\left( \frac{i}{2^k} \right) + Q_X\left( \frac{i+1}{2^k} \right) \right), \quad i \in \{0, 1, \dots, 2^k - 1\}$$

在每个区间内，权重落入的理论概率均为 $\frac{1}{16} = 6.25\%$。

#### 零点对齐修正（Zero-point Alignment）

在标准正态分布中，若严格按照 16 个等概率分位区间划分，离散网格点会以 0 为中心对称分布在正负两侧，**集合中不存在严格等于 0 的量化点**。

然而，神经网络中包含大量的精确零值（如 Padding 填充、未激活通道、结构化剪枝等）。若没有精确的零点表示，所有的 0 都会被迫偏移量化为一个微小的正数或负数，在多层前向传播中形成持续累加的基准偏差（Bias Offset）。

为此，NF4 对对称性进行了微调修正：
- 负数区域分配 7 个分位点；
- 显式保留一个严格的 $0$；
- 正数区域分配 8 个分位点。

这 16 个点在确定后，统一除以其绝对值最大值，完成归一化映射到 $[-1, 1]$。在运行时，权重块首先除以该块的绝对值峰值缩放因子映射至 $[-1, 1]$，然后通过二分查找找到最近的 NF4 离散值对应的 4-bit 索引（0 至 15）。

---

### 双重量化：把量化常数显存压低 75%

将权重压缩至 4-bit 可以将基座存储降低到原来的四分之一，但分块量化引入的**标度常数（Quantization Scales）**在长模型和大参数下会带来不可忽略的附加显存。

#### 量化标度因子的开销分析

为了防止矩阵局部的离群值（Outliers）拉大整层的缩放范围、导致普通权重精度骤降，QLoRA 采用了小粒度的分块量化（Block-wise Quantization），典型块大小设为 $B_1 = 64$。这意味着每 64 个参数就需要维护一个独立的浮点缩放系数 $c_1 = \max(|W_{\text{block}}|)$。

若每个 $c_1$ 采用标准 FP32（32 bits）存储，分摊到每个参数上的附加显存为：

$$\frac{32 \text{ bits}}{64 \text{ 参数}} = 0.5 \text{ bit/参数}$$

在 4-bit 量化体系中，每个权重本身只占 4 bits，而这 0.5 bit 的缩放常数相当于额外增加了 $12.5\%$ 的基座显存开销。对于 65B 规模的模型，$0.5 \text{ bit/参数}$ 对应的纯常数显存为：

$$\frac{65 \times 10^9 \times 0.5 \text{ bits}}{8 \text{ bits/Byte}} \approx 4.06 \text{ GB}$$

![双重量化存储结构拆解](/images/content-rebuild/67a366302ff30f1e/fig-3.png)

#### 两级量化的具体计算过程

双重量化（Double Quantization, DQ）通过对第一级量化产生的一系列缩放系数 $c_1$ 实施第二次量化，进一步降低显存开销：

1. **第一级量化（权重 $\to$ 4-bit NF4）**：以块大小 $B_1 = 64$ 将权重压缩为 4-bit，产出一组以 FP32 格式表达的第一级缩放因子张量 $C_1$。
2. **第二级量化（标度 $c_1 \to$ 8-bit FP8）**：将所有第一级缩放因子 $C_1$ 收集起来，以更大的块大小 $B_2 = 256$ 进行第二次量化，将 $c_1$ 压缩为 8-bit 浮点数（FP8），同时为每 256 个 $c_1$ 维护一个全局的 FP32 二级缩放因子 $c_2$。

核算启用双重量化后的综合显存开销：

$$\text{Cost}_{\text{DQ}} = 4 \text{ bits} + \frac{8 \text{ bits}}{B_1} + \frac{32 \text{ bits}}{B_1 \times B_2} = 4 + \frac{8}{64} + \frac{32}{64 \times 256} = 4 + 0.125 + 0.00195 \approx 4.127 \text{ bits/参数}$$

相比未经双重量化的 $4.5 \text{ bits/参数}$，双重量化每参数节省了约：

$$0.5 - 0.127 = 0.373 \text{ bits/参数}$$

对于 65B 参数的模型，这一优化直接省出了约 $3.03 \text{ GB}$ 物理显存；对于 70B 模型则节省约 $3.26 \text{ GB}$。在显存处于 48GB 临界边缘时，这 3GB 往往决定了程序是顺利运行还是直接抛出 CUDA OOM。

---

### 分页优化器：借助 CUDA 统一内存削平显存脉冲

在大模型指令微调过程中，训练中断最常见的原因并非平均显存超标，而是**瞬时显存脉冲（Memory Spikes）**导致的 CUDA 内存溢出。

#### 显存突增的产生原因

自回归模型训练时，样本的序列长度通常参差不齐。当遇到训练批次中包含极长上下文序列，或者反向传播结合激活检查点重新计算激活值的交汇瞬间，临时显存需求会在数个算子步内产生剧烈的峰值脉冲。即使平均常驻显存仅占物理容量的 80%，这种脉冲也能击穿剩余的显存空间导致训练崩溃。

![分页优化器在统一内存中的换页行为](/images/content-rebuild/67a366302ff30f1e/fig-4.png)

#### 基于 Unified Memory 的换页机制

QLoRA 引入了分页优化器（Paged Optimizers），底层依托 NVIDIA CUDA 统一内存（Unified Memory，由 `cudaMallocManaged` API 支持）机制：

1. **地址空间统一映射**：将 LoRA 适配器的优化器状态（一阶动量与二阶动量张量）分配到受驱动管理的托管内存页中，GPU 与主机 CPU 共享同一连续虚拟地址。
2. **物理显存换出（Page Eviction）**：在前向与反向传播过程中，优化器状态处于休眠不使用状态。当长序列激活计算造成 GPU 物理显存告急时，CUDA 驱动程序通过 PCIe 总线自动将处于非活跃状态的优化器内存页写出（Page-out）到系统宿主机 CPU 内存中，腾出物理空间保障前向与反向计算顺畅完成。
3. **缺页回拉（Page Fault Handling）**：反向传播结束、梯度计算完毕，训练进入参数更新步（Optimizer Step）。此时 GPU 核心重新发起对优化器状态的读写，MMU 触发缺页中断，CUDA 驱动将对应内存页预取拉回（Page-in）至 GPU 显存执行更新。

#### 适用前提与局限

- **为什么开销极小？**：LoRA 微调中优化器状态仅针对适配器参数，体积一般在数百兆到两三吉字节左右，并且在整个训练步中仅在末尾被集中访问一次。因此，PCIe 数据交换的时间损耗在整个 step 耗时中占比较小。
- **局限性**：分页优化器只能驱逐优化器状态。**如果长文本的前向激活值本身就已经超过了 GPU 物理显存的总容量，分页优化器无法阻止 OOM**。此时仍须结合梯度累积（Gradient Accumulation）与激活检查点来约束激活体积。

---

### 运行时计算流：动态反量化与吞吐权衡

关于 QLoRA 常见的一个认知误区是：“权重变成了 4-bit，所以矩阵乘法也是在 4-bit 精度下加速运行的”。

**事实并非如此：QLoRA 的 4-bit 纯粹用于静态显存存储与访存节约，所有的矩阵运算依然在 16-bit 浮点精度（BF16/FP16）下完成。**

#### 前向与反向传播的数据流转

设输入激活为 $X \in \mathbb{R}^{b \times d_{\text{in}}}$（$b$ 为批次序列长度或 token 数），线性层的冻结基座权重存储为 4-bit 离散张量 $W^{\text{NF4}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，LoRA 适配器包含降维矩阵 $A \in \mathbb{R}^{r \times d_{\text{in}}}$ 与升维矩阵 $B \in \mathbb{R}^{d_{\text{out}} \times r}$（均为 BF16）。前向计算表达式为：

$$Y = X \cdot \text{dequantize}(c_1, c_2, W^{\text{NF4}})^T + \frac{\alpha}{r} (X A^T) B^T$$

![QLoRA 计算流与动态反量化](/images/content-rebuild/67a366302ff30f1e/fig-5.png)

计算过程的时序如下：
1. **前向动态反量化**：读取 4-bit 离散索引与两级标度系数，CUDA 内核在片上寄存器与临时显存缓冲中即时解包为临时 16-bit 浮点权重 $\tilde{W}^{\text{BF16}}$。
2. **执行浮点 GEMM**：将输入激活 $X^{\text{BF16}}$ 与解包后的 $\tilde{W}^{\text{BF16}}$ 执行常规矩阵乘法，同时并行计算 LoRA 适配器分支并求和。
3. **即时释放解包内存**：算子计算完成的一瞬间，临时还原的 $\tilde{W}^{\text{BF16}}$ 立即被销毁，显存中只保留原始的 4-bit 紧凑权重。
4. **反向传播计算**：损失函数回传输出梯度 $\nabla_Y$。
   - 因为基座权重完全冻结，**不需要为基座权重计算梯度 $\nabla_W$**，从而省去了基座权重的梯度存储与反向更新。
   - 适配器梯度按链式法则计算：$\nabla_B = \frac{\alpha}{r} \nabla_Y^T (X A^T)$（维度为 $d_{\text{out}} \times r$），$\nabla_A = \frac{\alpha}{r} (\nabla_Y B)^T X$（维度为 $r \times d_{\text{in}}$）。
   - 为了将激活梯度传递给前一层网络，需计算基座与适配器两路回传梯度的和 $\nabla_X = \nabla_Y \cdot \tilde{W}^{\text{BF16}} + \frac{\alpha}{r} (\nabla_Y B) A$。因此在反向传播经过该层时，**4-bit 基座权重必须再次被动态反量化为 16-bit 浮点数 $\tilde{W}^{\text{BF16}}$** 以完成矩阵乘运算，随后立即释放。

#### 吞吐权衡（Throughput Trade-off）

前向与反向过程中反复执行的动态反量化解包操作引入了额外的算子调用与计算时延：
- 在相同硬件与批次下，**QLoRA 的训练单步耗时（Step Time）通常比标准全精度 16-bit LoRA 慢约 $20\% \sim 35\%$**。
- QLoRA 的核心定位是**以合理的计算耗时与吞吐下降，换取关键的显存容量节约**。当显存足够装下 16-bit 基座时，标准 LoRA 在训练速度上始终更优；而当显存不足以运行 16-bit 时，QLoRA 提供了能够拉起微调任务的可行路径。

---

### 全线性层注入准则（All-Linear Modules）

在标准 LoRA 的早期使用中，开发者常习惯仅在注意力机制的 Query 与 Value 投影层（`q_proj`, `v_proj`）添加低秩适配器。但在 QLoRA 中，基座权重已经被压缩至 4-bit，单一矩阵不可避免存在轻微的量化精度损失。

为了弥补这一精度降级，QLoRA 原论文经过大量消融实验给出了明确结论：**必须在 Transformer 的所有线性层上全量注入 LoRA 适配器（All-Linear）**。

![全线性模块注入示意](/images/content-rebuild/67a366302ff30f1e/fig-6.png)

注入目标覆盖以下全部模块（以主流 LLaMA 架构为例）：
- 注意力子层：`q_proj`, `k_proj`, `v_proj`, `o_proj`；
- 前馈网络（MLP）子层：`gate_proj`, `up_proj`, `down_proj`。

实验表明，在所有线性层配置较低秩（例如 $r=16$ 或 $r=8$），其语言表征拟合效果明显优于仅在 `q_proj` 和 `v_proj` 上配置超大秩（如 $r=64$），微调后的下游评测表现能够追平 16-bit 全参数微调基线。

---

### 动手实践：QLoRA 训练配置与底层检查

下面的可运行代码演示如何使用 Hugging Face `transformers`、`bitsandbytes` 与 `peft` 搭建标准的 QLoRA 4-bit 微调流水线。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 示例采用开源小模型以便本地快速验证，70B 模型配置逻辑完全相同
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 1. 声明 QLoRA 核心量化配置（BitsAndBytesConfig）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # 启用 4-bit 权重加载
    bnb_4bit_quant_type="nf4",                   # 采用 NF4 数据类型
    bnb_4bit_use_double_quant=True,             # 开启双重量化（压缩一级标度因子）
    bnb_4bit_compute_dtype=torch.bfloat16       # 运算时的反量化目标精度（推荐 bfloat16）
)

# 2. 以 4-bit 格式载入基座模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 3. 准备模型以支持低比特训练（冻结量化权重，将 LayerNorm 转为 FP32 保证数值稳定）
model = prepare_model_for_kbit_training(model)

# 4. 配置 LoRA 适配器：遵循 All-Linear 注入准则
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",                 # PEFT 官方支持一键指定全部线性层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 5. 注入适配器并构建 PEFT 模型
model = get_peft_model(model, peft_config)

# 验证可训练参数占比
model.print_trainable_parameters()

# 检查底层权重实际存储形态与量化状态
for name, module in model.named_modules():
    if hasattr(module, "weight") and hasattr(module.weight, "quant_state"):
        print(f"\n[验证成功] 找到量化层: {name}")
        print(f"  - 物理张量类型: {module.weight.dtype}")
        print(f"  - 量化状态对象: {type(module.weight.quant_state)}")
        print(f"  - 一级量化类型: {module.weight.quant_state.quant_type}")
        break
```

#### 运行结果预期

1. `model.print_trainable_parameters()` 输出中，可训练参数占比通常在 `0.1% ~ 0.5%` 范围。
2. 打印量化层信息时，底层张量物理类型显示为 `torch.uint8`（以单字节打包两个 4-bit 离散值），并且其附带的 `quant_state` 对象中包含 `quant_type='nf4'` 以及嵌套的双重量化二级标度信息。

---

### 工程陷阱与边界条件

在落地 QLoRA 工程前，需注意以下边界与约束：

1. **权重不能直接在 4-bit 空间数学相加**：在标准 LoRA 中，微调完成后通常执行 $W = W_0 + \frac{\alpha}{r}BA$ 并保存，以消除部署推理时的分支时延。但在 QLoRA 中，$W_0$ 已经高度非线性离散化，浮点更新矩阵无法直接累加到 4-bit 离散点上。若要合并导出，必须先将 $W_0$ 动态反量化至 16-bit 浮点，完成浮点矩阵相加后导出为 16-bit 权重；若目标环境仍需以 4-bit 运行推理，需重新使用 AWQ 或 GPTQ 等量化工具对导出后的 16-bit 模型进行推理量化。
2. **硬件算力架构对计算精度的影响**：配置中的 `bnb_4bit_compute_dtype` 强烈建议设置为 `torch.bfloat16`。在 NVIDIA Ampere、Ada Lovelace 及 Hopper 架构（如 RTX 30/40 系列、A100、H100）上，BF16 拥有与 FP32 完全相同的 8-bit 指数位宽，能有效承受反量化产生的数值范围突变；但在较旧的 Turing 或更早架构（如 T4、V100）上缺少原生硬件 BF16 算力，若强行指定会导致软件模拟降速，此时需显式回退为 `torch.float16`。
3. **多卡 DDP 与 FSDP 的适配兼容性**：QLoRA 的 4-bit 封装对 PyTorch 原生 DDP（Distributed Data Parallel）与 FSDP（Fully Sharded Data Parallel）的参数切分提出了额外挑战。直接对 4-bit 封装层进行通信可能导致反量化算子无法识别量化状态。多卡训练建议使用 Hugging Face `accelerate` 的 DeepSpeed ZeRO 配置，或借助 PEFT 与 FSDP 针对低比特微调的专用兼容补丁。

---

### 检验练习

1. **推导计算**：假设某大模型具有 $130 \times 10^9$（130B）个参数。采用块大小 $B_1 = 64$ 的 NF4 量化。请分别计算：
   - 仅采用 4-bit 权重（单级 FP32 标度，每 64 权重对应一个 FP32 标度）时，存储该基座权重需要的显存量（单位：GB，按 $10^9$ 字节计）。
   - 开启二级块大小 $B_2 = 256$、标度以 FP8 存储的双重量化后，基座权重所需的显存量，并计算双重量化节约的具体显存值。
2. **可执行代码练习**：修改文中的最小可运行实验代码，使用 `AdamW` 的分页优化器变体（`bitsandbytes.optim.PagedAdamW8bit` 或 `bitsandbytes.optim.PagedAdamW32bit`），并构造一个单样本由 10 个 token 组成的虚拟张量执行一次 `loss.backward()` 与 `optimizer.step()`，打印验证梯度是否正常产生在适配器矩阵上。
