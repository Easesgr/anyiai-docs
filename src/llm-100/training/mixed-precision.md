---
title: 混合精度训练中 FP16 与 BF16 有什么区别？该如何选型？
icon: mdi:tune-vertical
tag:
- 训练原理
- 大模型100讲
- 预训练
description: 对比 FP16 与 BF16 的位结构、动态范围与数值稳定性，拆解损失缩放、主权重机制与训练显存账本，给出选型与验证代码。
---

在大模型训练中，如果全流程使用 32 位单精度浮点数（FP32），显存开销和计算耗时都会非常高。混合精度训练（Mixed Precision Training）的核心思路，是在前向与反向的大量矩阵乘法中，使用 16 位浮点数替代 32 位浮点数，从而加快计算速度并减少前向激活值的显存占用。

目前主流的 16 位格式有两种：**FP16（半精度浮点）** 与 **BF16（Brain Floating Point 16）**。它们总共都占用 16 个二进制位（2 个字节），但内部位分配完全不同。本讲通过具体数值计算说明两者的差异、常见数值故障及解决方案。

### 本讲学习目标
1. 理解 FP16 与 BF16 的比特位分配、动态范围和精度差异。
2. 搞清楚 FP16 梯度下溢的原因以及动态损失缩放（Loss Scaling）的具体工作流程。
3. 理解混合精度训练中为什么必须维护 FP32 主权重，并掌握准确的显存账本核算。
4. 掌握不同硬件平台上的选型规则，以及在 PyTorch 和 Hugging Face Trainer 中的标准配置。

![FP16 与 BF16 混合精度训练技术图景](/images/content-rebuild/33c870b86835cbd9/cover.png)

---

### 一个贯穿全文的数值计算例子

我们通过具体的数值算例，直观观察低精度计算会遇到的两个核心问题：

1. **梯度下溢归零**：
   假设反向传播计算出的某项微弱梯度为 $g = 1.0 \times 10^{-8}$。FP16 能表示的最小正规格化数约为 $6.10 \times 10^{-5}$，即便计入非规格化数，其绝对下限也仅为 $2^{-24} \approx 5.96 \times 10^{-8}$。在 FP16 格式下保存该梯度，数值会直接**下溢为 0**（若硬件开启 Flush-to-Zero 模式，低于 $6.10 \times 10^{-5}$ 就会直接归零），导致模型参数完全停止更新。而 BF16 最小正数可达 $\approx 1.18 \times 10^{-38}$，可直接保留该微弱梯度。
2. **数值吞没（Swamping）**：
   假设网络中某个权重当前值 $W = 1.0$，经过优化器计算出的单步理论更新量为 $\Delta W = 1.0 \times 10^{-6}$。
   在参数更新环节执行 $W + \Delta W = 1.0 + 1.0 \times 10^{-6}$：
   - BF16（尾数 7 位）的机器精度仅为 $2^{-7} \approx 7.81 \times 10^{-3}$，FP16（尾数 10 位）机器精度为 $2^{-10} \approx 9.77 \times 10^{-4}$。面对 $1.0 \times 10^{-6}$ 的微小更新，16 位浮点加法会将更新量直接舍入抹除，累加后依然是 $1.0$。这种大数吃掉小数的现象叫作**数值吞没**。
   - FP32（尾数 23 位）的机器精度达 $2^{-23} \approx 1.19 \times 10^{-7}$，能够准确分辨并累加 $1.0 \times 10^{-6}$。
3. **解决路径**：
   - 对于梯度下溢：FP16 需要引入**损失缩放（Loss Scaling）**人为放大梯度；BF16 凭借 8 位宽指数位天然避开下溢。
   - 对于数值吞没：无论是 FP16 还是 BF16，优化器都必须常驻维护一份 **FP32 主权重**，参数累加更新必须在 32 位高精度下完成。

下面我们展开拆解具体的表示格式与工程实现。

---

### IEEE 754 浮点数表示基础与位结构对比

计算机中的标准浮点数通常采用类似科学计数法的方式存储。一个规格化浮点数 $x$ 由三个二进制部分组成：

$$x = (-1)^S \times 2^{E - \text{Bias}} \times \left(1 + \frac{M}{2^{\text{fraction\_bits}}}\right)$$

- **符号位（Sign, $S$）**：占 1 位。0 表示正数，1 表示负数。
- **指数位（Exponent, $E$）**：决定数值能够表达的大小范围（动态范围）。指数存储时加上了偏移量 $\text{Bias} = 2^{\text{exp\_bits} - 1} - 1$。
- **尾数位（Mantissa / Fraction, $M$）**：决定数值的有效数字精度。规格化浮点数默认前面带有一个不存储的隐藏位 1。

![IEEE 754 浮点格式位分配对比](/images/content-rebuild/33c870b86835cbd9/fig-1.png)

#### 常见格式的结构对比

| 格式 | 总位宽 | 符号位 | 指数位 | 尾数位 | 指数偏移量（Bias） | 最小正规格化数 | 最大可表示数值 | 十进制有效位 | 典型硬件支持 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | 32 | 1 | 8 | 23 | 127 | $\approx 1.18 \times 10^{-38}$ | $\approx 3.40 \times 10^{38}$ | 6 ~ 9 位 | 通用 CPU / GPU |
| **FP16** | 16 | 1 | 5 | 10 | 15 | $\approx 6.10 \times 10^{-5}$ | $65504$ | 3 ~ 4 位 | NVIDIA V100 及后续架构 |
| **BF16** | 16 | 1 | 8 | 7 | 127 | $\approx 1.18 \times 10^{-38}$ | $\approx 3.39 \times 10^{38}$ | 2 ~ 3 位 | Google TPU / NVIDIA A100 及后续架构 |
| **TF32** | 19 | 1 | 8 | 10 | 127 | $\approx 1.18 \times 10^{-38}$ | $\approx 3.40 \times 10^{38}$ | 3 ~ 4 位 | NVIDIA Ampere 及后续 Tensor Core 内部运算 |

*注：TF32（TensorFloat-32）不是显存中的数据存储格式，而是 NVIDIA Ampere 及之后架构在 Tensor Core 内部执行 FP32 矩阵乘法时的计算模式。数据在显存中依然按照 FP32 读写，进入计算单元时尾数被截断以换取吞吐速度。*

从位分配可以看出两种 16 位格式的设计取舍：
- **FP16 偏向精度**：分配了 10 位尾数，有效数字相对更准；但指数位仅有 5 位，最大值只能到 65504，最小规格化正数仅到 $6.10 \times 10^{-5}$，数值范围非常窄。
- **BF16 偏向范围**：直接继承了 FP32 的 8 位指数，指数范围与 FP32 完全相同，上可至 $3.39 \times 10^{38}$，下可至 $1.18 \times 10^{-38}$；代价是将尾数压缩到了 7 位，有效数字只有 2 到 3 位。

<ReadMoreLock />

---

### FP16 的下溢风险与动态损失缩放（Loss Scaling）

在大模型训练的反向传播中，越往浅层传递，经过多次矩阵乘与缩放后的梯度往往越小。统计显示，Transformer 训练中大量梯度的绝对值集中在 $10^{-7}$ 到 $10^{-4}$ 区间。

在 FP16 格式下：
- 当梯度小于 $2^{-14} \approx 6.10 \times 10^{-5}$ 时，进入非规格化数区间，精度受损；
- 当梯度小于非规格化数的绝对下限 $2^{-24} \approx 5.96 \times 10^{-8}$ 时，数值直接**下溢变为 0**；
- 另一方面，如果前向传播或反向求导的某个中间值超过 65504，就会发生**上溢变为 Inf**，并迅速污染整个网络导致 NaN。

![FP16 梯度下溢与动态损失缩放机制](/images/content-rebuild/33c870b86835cbd9/fig-2.png)

#### 动态损失缩放如何工作？

为了防止微小梯度直接下溢归零，框架会在反向传播前人为放大梯度。

依据链式求导法则，标量损失（Loss）乘以常数 $S$，反向求出的所有梯度也会同步放大 $S$ 倍：
$$\nabla_W (S \cdot \mathcal{L}) = S \cdot \nabla_W \mathcal{L}$$

具体执行状态机如下：

1. **前向与缩放**：计算得到标量损失 $\mathcal{L}$ 后，将其乘以缩放因子 $S$（例如初始值 $2^{16} = 65536$），得到放大后的损失 $\mathcal{L}_{\text{scaled}} = S \cdot \mathcal{L}$。
2. **反向传播**：在 16 位精度下执行反向传播。因为损失被放大了 $S$ 倍，求出的梯度也同步放大了 $S$ 倍，原本小于 $6.10 \times 10^{-5}$ 的微小梯度被推入安全区间，不会下溢归零。
3. **梯度反缩放（Unscaling）**：在权重更新前，把梯度显式除以 $S$ 还原真实幅度：$g = g_{\text{scaled}} / S$。
4. **溢出检测与自适应更新**：
   - **检测到 Inf 或 NaN**：说明当前的缩放因子 $S$ 设得过大，导致部分梯度上溢。此时**跳过本次参数更新**（优化器不执行 `step()`），同时调小缩放因子（通常 $S \leftarrow S \times 0.5$），并重置连续未溢出计数器。
   - **未检测到溢出**：执行优化器参数更新。若连续若干步（例如连续 2000 步）都没有溢出，说明当前数值环境安全，可尝试调大缩放因子（通常 $S \leftarrow S \times 2$），以便覆盖更小的梯度。

损失缩放解决了 FP16 的下溢问题，但溢出检测和跳步机制增加了训练的不确定性。如果缩放因子选择不当，可能导致频繁跳步甚至训练发散。

---

### BF16 的结构取舍与数值稳定性

BF16 舍弃了更高的尾数精度，换取了与 FP32 完全一致的动态范围。

![BF16 与 FP32 截断及精度舍入特征](/images/content-rebuild/33c870b86835cbd9/fig-3.png)

#### 为什么大模型训练更青睐宽动态范围？

大模型训练使用的小批量随机梯度下降（SGD/AdamW）本身具有采样方差。经验与实验证明，深度网络对随机微小的舍入误差具有较强的容忍度，但对于“梯度直接归零（下溢）”或“数值超出上限（溢出为 Inf）”极其敏感：

1. **不需要损失缩放器（Loss Scaler）**：BF16 的下溢下限约为 $1.18 \times 10^{-38}$，大模型训练中几乎不会出现低于该量级的有效梯度。因此采用 BF16 训练时，框架可以直接停用 Loss Scaler，避免了溢出检查、跳步重试以及缩放因子的状态维护。
2. **与 FP32 转换开销极低**：FP32 转换为 BF16 时，指数位不需要重新偏置，通常只需保留高 16 位并配合最后一位进行就近舍入（Round-to-nearest-even）。

#### BF16 的精度防护措施

BF16 的机器精度极限仅为 $\epsilon = 2^{-7} \approx 7.81 \times 10^{-3}$。当累加大量微小数值时，容易产生严重的累积舍入误差。因此在实际框架实现中，必须遵循以下防御规则：

- **敏感算子保持高精度**：LayerNorm / RMSNorm（均值与方差计算）、Softmax（指数求和归一化）等对累加敏感的算子，必须在 FP32 下执行计算。
- **Tensor Core 内部累加**：NVIDIA Ampere/Hopper 等架构的 Tensor Core 在执行 BF16 矩阵乘时，内部累加器统一使用 FP32 寄存器，最终输出时再按需写回 BF16 或保持 FP32。

---

### 混合精度训练的数据流（AMP）

自动混合精度（AMP, Automatic Mixed Precision）不是全流程无脑使用 16 位。如果所有地方都用 16 位，前面提到的“数值吞没”现象会导致权重停止更新。

因此，标准混合精度训练采用**双副本数据流**。

![混合精度训练的前向与后向完整数据流](/images/content-rebuild/33c870b86835cbd9/fig-4.png)

#### 数据流执行过程

1. **维护 FP32 主权重**：优化器常驻维护一套完整的 FP32 模型权重（Master Weights）。
2. **生成 16 位副本**：每轮前向计算前，将 FP32 主权重转换为 BF16 或 FP16 副本。
3. **低精度前向传播**：线性层矩阵乘（GEMM）在 Tensor Core 上以 16 位输入执行，激活值以 16 位保存在显存中（用于反向求导）；数值敏感算子（如 Softmax、LayerNorm）自动转为 FP32 计算。
4. **反向传播**：依据 16 位激活值和计算图反向计算出梯度。
5. **FP32 状态更新**：在进入优化器更新参数前，梯度转换为 FP32（FP16 模式下先在此处执行反缩放）。AdamW 优化器内的一阶动量、二阶动量以及步长计算全部在 FP32 精度下进行。
6. **更新主权重**：计算出的参数更新量直接累加到 FP32 主权重上。这样微小的更新量就不会被大数值吞没。

#### 显存账本：混合精度训练总显存为什么没有减半？

常有初学者认为：“32 位换成 16 位，训练显存直接省一半”。我们以一个参数量为 $P$ 的模型和 AdamW 优化器为例，拆解静态显存的实际占用（详细分析可回顾[《大模型训练为什么这么吃显存？》](/llm-100/training/why-high-vram.md)）：

![FP32 与混合精度静态及动态显存开销对比](/images/content-rebuild/33c870b86835cbd9/fig-5.png)

| 显存项 | 全 FP32 训练 (Bytes/参数) | 混合精度训练 (AMP) (Bytes/参数) | 说明 |
| :--- | :--- | :--- | :--- |
| **模型权重** | $4P$ (FP32) | $4P$ (FP32 主权重) + $2P$ (16 位运算副本) = $6P$ | AMP 需要额外保留 16 位前向副本 |
| **梯度** | $4P$ (FP32) | $2P$ (16 位) 或 $4P$ (FP32 累加) | 依框架实现而定，常见为 $2P \sim 4P$ |
| **优化器状态 (AdamW)** | $8P$ ($4P$ 动量 + $4P$ 方差) | $8P$ ($4P$ 动量 + $4P$ 方差，必须保持 FP32) | 动量与方差在 16 位下极不稳定，必须用 FP32 |
| **静态显存小计** | **$16P$ 字节** | **$16P \sim 18P$ 字节** | **静态显存不仅没有减少，反而略有增加** |

**混合精度到底省在哪里？**

混合精度节省的不是静态参数与优化器状态（静态切分需要配合[《ZeRO、DeepSpeed、FSDP 解决了什么问题？》](/llm-100/training/zero-deepspeed-fsdp.md)中讨论的技术），而是**前向激活值（Activation Memory）**：

长上下文和大 Batch Size 下，中间激活值占用的显存往往是静态权重的数倍。激活值从每个元素 4 字节降到 2 字节，直接使激活显存削减近 50%。同时，Tensor Core 对 16 位矩阵乘法的计算吞吐远高于 FP32，训练速度得以显著提升。

---

### 硬件兼容性与配置建议

不同算力架构对 BF16 的硬件支持存在明确代际：

![NVIDIA GPU 架构代际与数值格式支持演进](/images/content-rebuild/33c870b86835cbd9/fig-6.png)

#### 硬件选型条件
- **Volta 架构（如 V100） / Turing 架构（如 T4, RTX 2080）**：仅原生支持 **FP16** Tensor Core 加速。在这些卡上如果强行指定 BF16，PyTorch 会被迫转为纯软件模拟，速度会下降数倍。在 V100/T4 上训练**必须选择 FP16** 并开启 GradScaler。
- **Ampere 架构（如 A100, RTX 3090） / Ada Lovelace 架构（如 RTX 4090）**：第三/四代 Tensor Core 开始原生支持 **BF16**。硬件矩阵乘吞吐与 FP16 相同，**推荐首选 BF16**，无需损失缩放。
- **Hopper 架构（如 H100） / Blackwell 架构**：除成熟支持 BF16/FP16 外，还增加了对 FP8（E4M3/E5M2）的原生支持。

#### 框架实操配置

若使用 Hugging Face 的 `Trainer`（参考官方文档标准实现），只需在 `TrainingArguments` 中设定互斥开关：

```python
from transformers import TrainingArguments

# 场景 A：现代 GPU (A100, H100, RTX 3090/4090 等)，首选 BF16
training_args_modern = TrainingArguments(
    output_dir="./outputs",
    bf16=True,             # 启用 BF16 混合精度
    fp16=False,
    learning_rate=2e-5,
)

# 场景 B：老旧 GPU (V100, T4 等)，必须使用 FP16 (自动开启动态损失缩放)
training_args_legacy = TrainingArguments(
    output_dir="./outputs",
    fp16=True,             # 启用 FP16 混合精度
    bf16=False,
    learning_rate=2e-5,
)
```

若使用原生 PyTorch 训练循环，推荐使用官方标准的 `torch.amp` 接口：

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.Linear(1024, 1024).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 检测硬件是否原生支持 BF16
use_bf16 = torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

# FP16 需要启用 GradScaler，BF16 模式下停用 (enabled=False)
scaler = torch.amp.GradScaler("cuda", enabled=(not use_bf16))

for x, target in dataloader:
    optimizer.zero_grad()
    
    # 使用自动混合精度上下文管理前向传播
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
    
    # 反向传播与梯度更新
    if not use_bf16:
        # FP16: 损失缩放 -> 反向求导 -> 反缩放与溢出检测 -> 步进更新
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        # BF16: 直接反向求导与步进更新
        loss.backward()
        optimizer.step()
```

关于梯度在网络中具体是如何传播与计算的，可参考[《梯度下降、反向传播在大模型里如何工作？》](/llm-100/training/gradient-descent-backprop.md)。

---

### 可检验练习

请在本地 Python 环境中运行以下脚本，亲自检验浮点下溢与数值吞没机制：

```python
import torch

def test_floating_point_behaviors():
    print("--- 1. 检验梯度下溢 ---")
    grad_val = 1e-8
    # 转换精度
    g_fp16 = torch.tensor(grad_val, dtype=torch.float16)
    g_bf16 = torch.tensor(grad_val, dtype=torch.bfloat16)
    
    print(f"真实梯度值: {grad_val}")
    print(f"FP16 表示值: {g_fp16.item()} (是否下溢归零: {g_fp16.item() == 0.0})")
    print(f"BF16 表示值: {g_bf16.item()} (是否保留非零: {g_bf16.item() > 0.0})")
    
    # 断言：FP16 下溢为 0，而 BF16 能保留该数值
    assert g_fp16.item() == 0.0, "预期 FP16 在 1e-8 下溢为 0"
    assert g_bf16.item() > 0.0, "预期 BF16 能表达 1e-8"

    print("\n--- 2. 检验数值吞没 (Swamping) ---")
    weight = 1.0
    small_update = 1e-6
    
    # 在 BF16 精度下直接相加
    w_bf16 = torch.tensor(weight, dtype=torch.bfloat16)
    up_bf16 = torch.tensor(small_update, dtype=torch.bfloat16)
    res_bf16 = w_bf16 + up_bf16
    
    # 在 FP32 精度下相加
    w_fp32 = torch.tensor(weight, dtype=torch.float32)
    up_fp32 = torch.tensor(small_update, dtype=torch.float32)
    res_fp32 = w_fp32 + up_fp32
    
    print(f"BF16 直接累加: 1.0 + 1e-6 = {res_bf16.item()} (更新量是否丢失: {res_bf16.item() == 1.0})")
    print(f"FP32 主权重累加: 1.0 + 1e-6 = {res_fp32.item()} (更新是否成功: {res_fp32.item() > 1.0})")
    
    # 断言：BF16 精度不足导致吞没，FP32 正常累加
    assert res_bf16.item() == 1.0, "预期 BF16 发生数值吞没"
    assert res_fp32.item() > 1.0, "预期 FP32 成功保留更新"
    print("\n所有检验测试通过！")

if __name__ == "__main__":
    test_floating_point_behaviors()
```

---

### 课程体系与延伸导航

本篇是训练原理篇中关于数值格式与算力底层特性的核心章节。

**训练原理篇进阶学习路径**：
1. [大模型预训练到底在训练什么？](/llm-100/training/what-is-pretraining.md)
2. [自回归语言模型为什么能预测下一个词？](/llm-100/training/autoregressive-lm.md)
3. [Masked Language Model 和 Causal Language Model 有什么区别？](/llm-100/training/mlm-vs-clm.md)
4. [大模型训练中的 Loss 是怎么计算的？](/llm-100/training/loss-calculation.md)
5. [梯度下降、反向传播在大模型里如何工作？](/llm-100/training/gradient-descent-backprop.md)
6. [训练大模型为什么需要海量数据？](/llm-100/training/why-massive-data.md)
7. [大模型训练为什么这么吃显存？](/llm-100/training/why-high-vram.md)
8. **混合精度训练 FP16、BF16 有什么区别？**（当前讲）
9. [数据并行、模型并行、流水线并行分别是什么？](/llm-100/training/parallel-strategies.md)
10. [ZeRO、DeepSpeed、FSDP 解决了什么问题？](/llm-100/training/zero-deepspeed-fsdp.md)
