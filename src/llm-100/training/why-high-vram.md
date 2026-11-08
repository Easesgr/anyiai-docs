---
title: 大模型训练为什么这么吃显存？显存构成量化拆解与优化
icon: mdi:tune-vertical
tag:
- 训练原理
- 大模型100讲
- 预训练
description: 以 7B 模型为例，量化拆解模型参数、梯度、AdamW 状态与动态激活值的显存占用，澄清训练与推理的显存机制差异，并给出可实测的显存优化路径。
---

在尝试全参数训练或微调大语言模型时，开发者最常遇到的报错往往是显存溢出（`CUDA out of memory`）。

一个以 16 位浮点数（FP16 或 BF16）存储的 70 亿参数（7B）模型，权重文件大约占用 14 GB（即 $14 \times 10^9$ 字节，约 13.04 GiB）。直觉上，一张 24 GB 显存的显卡看起来足够放下。但在单卡全参数训练时，即使将批次大小（Batch Size）设为 1，程序依然会在反向传播时迅速崩溃。

显存之所以消耗巨大，是因为训练显存由两大部分组成：
1. **静态模型状态（Model States）**：包含模型参数、梯度，以及优化器所维护的历史状态；
2. **动态激活值（Activations）与运行时缓存**：前向传播中每层算子生成的中间输出张量，必须保留在显存中供反向传播求导使用。

本文以 7B 模型为具体算例，拆解每一部分的显存计算公式，解释训练与推理在显存机制上的差异，并介绍激活重计算（Activation Checkpointing）的优化原理与实测代码。

![大模型训练显存全景拆解图](/images/content-rebuild/11e04fd3be4cfb58/cover.png)

---

## 静态显存拆解：参数、梯度与优化器状态

静态模型状态的大小主要由模型总参数量 $\Phi$ 以及选用的优化算法决定。以常见的大语言模型为例，设总参数量为 $\Phi$（7B 模型对应 $\Phi = 7 \times 10^9$）。

### 1. 模型参数（Parameters）
每个参数占用的显存取决于存储精度：
- 单精度浮点数（FP32）：每个参数占用 4 个字节（Bytes）；
- 半精度浮点数（FP16 或 BF16）：每个参数占用 2 个字节。

采用 FP16 存储时，7B 模型自身的参数显存为：
$$M_{\text{param}} = \Phi \times 2 \text{ Bytes} = 7 \times 10^9 \times 2 = 14 \times 10^9 \text{ Bytes} \approx 13.04 \text{ GiB}$$

### 2. 梯度（Gradients）
在反向传播过程中，程序根据链式法则计算损失函数对每个参数的偏导数。每个可学习参数都对应一个大小相同的梯度标量。在混合精度训练中，梯度通常以 FP16/BF16 格式存储，每个参数对应 2 个字节：
$$M_{\text{grad}} = \Phi \times 2 \text{ Bytes} = 14 \times 10^9 \text{ Bytes} \approx 13.04 \text{ GiB}$$

![模型参数与梯度的张量对应关系](/images/content-rebuild/11e04fd3be4cfb58/fig-1.png)

仅参数和梯度两项，每参数就需要 $2 + 2 = 4$ 字节，7B 模型合计约 26.08 GiB，已经超过了 24 GB 显卡的物理上限。

<ReadMoreLock />

---

## 优化器状态：AdamW 为什么占显存大头

训练自回归大语言模型时，常用优化器是 AdamW。AdamW 不仅使用当前时间步的梯度，还要记录历史梯度的动量（一阶矩）和方差（二阶矩）。

在混合精度训练中，半精度浮点数（FP16）的动态范围较窄（正规数最小约 $6.10 \times 10^{-5}$）。当学习率设置在 $10^{-4}$ 到 $10^{-5}$ 区间时，梯度乘以学习率后的微小更新步长很容易发生下溢，直接被截断为零。为了维持数值更新的准确性，标准混合精度训练框架要求 AdamW 在后台以单精度（FP32）维护三组状态：

1. **FP32 主权重副本（Master Weights）**：用于高精度累加微小的梯度更新量，每参数 4 字节；
2. **FP32 一阶动量（Momentum）**：跟踪梯度的指数滑动平均，每参数 4 字节；
3. **FP32 二阶方差（Variance）**：跟踪梯度平方的指数滑动平均，每参数 4 字节。

![AdamW 优化器状态构成拆解](/images/content-rebuild/11e04fd3be4cfb58/fig-2.png)

在使用标准 AdamW 的混合精度训练下，优化器状态本身的显存消耗为：
$$M_{\text{opt}} = 4\Phi + 4\Phi + 4\Phi = 12\Phi \quad (\text{Bytes})$$
对于 7B 模型，仅优化器状态就需要消耗 $12 \times 7 \times 10^9 \text{ Bytes} = 84 \times 10^9 \text{ Bytes} \approx 78.23 \text{ GiB}$。

### 纯 FP32 训练与混合精度训练的静态显存对比
很多初学者认为“换成 FP16 混合精度，模型显存就会减半”。在标准 AdamW 优化器假设下，这一结论对静态显存并不成立：

- **纯 FP32 训练**：
  - 参数：$4\Phi$
  - 梯度：$4\Phi$
  - AdamW 状态（无需额外主权重副本，只需动量与方差）：$4\Phi + 4\Phi = 8\Phi$
  - 静态状态总计：$4\Phi + 4\Phi + 8\Phi = 16\Phi$ 字节。

- **FP16/BF16 混合精度训练**：
  - 参数：$2\Phi$
  - 梯度：$2\Phi$
  - AdamW 状态（主权重 + 动量 + 方差）：$4\Phi + 4\Phi + 4\Phi = 12\Phi$
  - 静态状态总计：$2\Phi + 2\Phi + 12\Phi = 16\Phi$ 字节。

**结论**：在搭配标准 AdamW 优化器时，无论是纯 FP32 训练还是标准混合精度训练，静态模型状态显存均约为 **$16\Phi$ 字节（每 10 亿参数约需 16 GB / 14.9 GiB 显存）**。

对于 7B 模型，仅静态状态总计就需要：
$$16 \times 7 \times 10^9 \text{ Bytes} = 112 \times 10^9 \text{ Bytes} \approx 104.31 \text{ GiB}$$
即使是拥有 80 GB 显存的单张 A100/H100 显卡，也无法单独放下 7B 模型的全套静态训练状态。这是预训练与全参数微调通常必须采用分布式状态分片（如 ZeRO 或 FSDP）的根本原因。

---

## 动态激活值显存：反向传播的算子缓存

静态状态在模型初始化后大小固定，而动态激活值（Activations）则随着前向传播逐层计算并驻留显存，直到反向传播求导完毕后才被释放。

### 为什么必须保留激活值？
反向传播的链式求导依赖前向传播的输入张量。以最基础的全连接层 $Y = XW$ 为例：
$$\frac{\partial \mathcal{L}}{\partial W} = X^T \frac{\partial \mathcal{L}}{\partial Y}$$
为了计算权重矩阵 $W$ 的梯度，程序必须使用输入矩阵 $X$。因此，在前向计算出 $Y$ 之后，$X$ 不能立即被垃圾回收，必须在显存中保留到反向传播回传至该层为止。

随着网络层数加深，从第一层到最后一层所产生的中间张量会持续在显存中累积，直到反向传播开始才逐步释放，从而在网络前向传播末端形成显存峰值。

![激活值在前向堆积与反向释放的时序图](/images/content-rebuild/11e04fd3be4cfb58/fig-3.png)

### 激活值开销估算
设单卡微批次大小为 $b$，输入序列长度为 $s$，隐藏层维度为 $h$，注意力头数为 $a$，网络层数为 $L$。在采用半精度（2 字节/元素）的标准 Transformer 架构中，单层需要保留的典型激活值包含：
- 归一化层输入与输出；
- 投影线性层输入；
- 注意力得分矩阵 $QK^T$（形状为 $b \times a \times s \times s$）与 Softmax 概率矩阵；
- MLP 块的前向扩展特征。

在标准全注意力实现中，单层激活值占用的显存近似为：
$$M_{\text{act\_per\_layer}} \approx 34 b s h + 5 b a s^2 \quad (\text{Bytes})$$
全网 $L$ 层的总激活显存为：
$$M_{\text{act}} = L \times M_{\text{act\_per\_layer}} \approx L \cdot (34 b s h + 5 b a s^2) \quad (\text{Bytes})$$

从该公式可以看出激活显存的关键特性：
1. 与序列长度 $s$ 存在二次方依赖关系（$O(L \cdot b \cdot a \cdot s^2)$），主要来自注意力得分矩阵；
2. 当处理长文本（如 $s \ge 4096$ 或 $32768$）时，注意力矩阵的激活值显存会呈二次方增长，往往会迅速超越静态模型状态，成为引发 OOM 的主因。

---

## 训练显存 vs 推理显存：澄清 KV Cache 误区

在显存机制的讨论中，经常出现一个疑问：**大模型在训练时是否也需要维护 KV Cache？**

结论很明确：**自回归语言模型在标准全序列并行训练中，不维护推理阶段的 KV Cache 结构。**

### 1. 机制差异：并行计算 vs 逐词生成
- **训练阶段（全序列并行 / Teacher Forcing）**：
  整个训练序列的所有上下文 Token 和目标 Token 在输入前就已经完全给出。模型使用因果注意力掩码（Causal Mask）一次性完成全序列的注意力得分计算。这个过程是一次性的大矩阵乘法，没有逐时间步的循环生成，不需要在步间保留与拼接历史 Key 和 Value 缓存。

- **推理阶段（自回归逐步生成）**：
  模型逐个生成新 Token。每生成一个词，就要将其加入上下文以预测下一个词。为了避免每一步都重新计算历史 Token 的 Key 和 Value 向量，系统才开辟专用的 KV Cache 显存空间，复用历史计算结果。

![训练全序列并行与推理自回归 KV Cache 的显存机制对比](/images/content-rebuild/11e04fd3be4cfb58/fig-4.png)

### 2. 训练与推理显存对比

| 维度 | 标准训练阶段（Training） | 自回归推理阶段（Inference） |
| :--- | :--- | :--- |
| **计算方式** | 全序列并行计算（Teacher Forcing） | 逐 Token 串行自回归生成 |
| **KV Cache** | **不维护**（单次矩阵乘法完成，无跨步复用） | **显式维护**（以显存换取逐步生成速度） |
| **反向求导** | 存在（需在显存中保留全网络中间激活值） | 不存在（前向完成后立即丢弃中间算子输出） |
| **优化器状态** | 占用大头（标准 AdamW 需 $12\Phi$ 字节） | 无（权重只读，不更新） |
| **显存主要压力** | 优化器状态 + 随层数堆积的中间激活值 | 静态权重参数 + 随并发与序列长度增长的 KV Cache |

需要说明的是：虽然训练不使用推理专用的 KV Cache 动态缓存结构，但前向传播计算出的全序列 Key 和 Value 张量，作为反向传播求导所需的中间激活值，在这一轮计算结束前依然会驻留在显存中。

---

## 激活重计算：以时间换空间的工程权衡

当激活值随模型层数堆积导致显存不足时，**激活重计算（Activation Checkpointing / Gradient Checkpointing）**是最常用的单卡显存优化手段。

### 1. 运行机制
在标准训练中，前向传播会保留每一个算子的中间输出。开启激活重计算后：
1. **前向阶段**：以 Transformer Block 为单位，**只保留每个 Block 的输入张量（检查点）**，Block 内部生成的中间激活值（如注意力得分矩阵、中间激活函数输出）在前向完成后立刻从显存中丢弃释放。
2. **反向阶段**：当反向传播计算逐层回传到某个 Block 时，程序利用先前保留的 Block 输入张量，**重新执行一次该 Block 的前向计算**，现场恢复丢弃的中间激活值；随后立即完成该 Block 权重的梯度求导，并在求导完成后再次释放这些临时激活值。

![激活重计算的工作流与显存释放机制](/images/content-rebuild/11e04fd3be4cfb58/fig-5.png)

### 2. 代价与收益分析
- **显存收益**：开启全量重计算后，显存中不再同时堆积全部 $L$ 层的内部激活值，只需保留每层的边界输入张量和当前正在计算的单个 Block 内部激活值。深层 Transformer 的激活显存峰值可降低约 $60\% \sim 80\%$。
- **计算开销**：反向传播的计算量理论上约为前向传播的 2 倍（分别对输入求导和对权重求导）。
  - 原始训练计算量：$1\text{（前向）} + 2\text{（反向）} = 3\text{ 单位}$；
  - 激活重计算计算量：$1\text{（前向）} + 1\text{（重算前向）} + 2\text{（反向）} = 4\text{ 单位}$；
  - 理论计算量增加：$\frac{4 - 3}{3} \approx 33.3\%$。

以约 $33\%$ 的理论计算时间开销换取激活显存的大幅降低，使得在单卡上训练更大序列长度成为可能。

### 3. 可检验代码：PyTorch 激活检查点实测
以下脚本展示如何在 PyTorch 中对比标准前向与激活重计算的显存峰值：

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class HeavyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        residual = x
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(h)))
        return residual + h

class DemoModel(nn.Module):
    def __init__(self, layers=16, dim=2048, use_checkpoint=False):
        super().__init__()
        self.layers = nn.ModuleList([HeavyBlock(dim) for _ in range(layers)])
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint:
                # use_reentrant=False 使用 PyTorch 推荐的新版机制
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    data = torch.randn(8, 512, 2048, dtype=torch.float16, device=device, requires_grad=True)

    # 1. 基准测试：不开启检查点
    torch.cuda.reset_peak_memory_stats()
    model_std = DemoModel(layers=16, dim=2048, use_checkpoint=False).to(device).half()
    loss_std = model_std(data).sum()
    loss_std.backward()
    peak_std = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"标准训练显存峰值: {peak_std:.2f} MiB")

    # 2. 优化测试：开启激活重计算
    del model_std, loss_std
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_chk = DemoModel(layers=16, dim=2048, use_checkpoint=True).to(device).half()
    loss_chk = model_chk(data).sum()
    loss_chk.backward()
    peak_chk = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"激活重计算显存峰值: {peak_chk:.2f} MiB")
```

在单张支持 CUDA 的 GPU 上运行此代码，可以直观观察到开启检查点后显存峰值的下降。

---

## 训练显存排查与优化路径

当训练程序触发 `CUDA out of memory` 报错时，排查可遵循如下顺序：

![训练显存溢出 OOM 排查与优化决策树](/images/content-rebuild/11e04fd3be4cfb58/fig-6.png)

### 步骤一：核算静态模型状态
计算静态状态的刚性开销：
$$M_{\text{model\_states}} \approx 16 \times \Phi \quad (\text{Bytes})$$
- **判断**：若 $16\Phi$ 已经大于显卡物理显存（例如 24 GB 显卡尝试单卡微调 7B 模型，静态状态即需约 104 GiB），此时调小 Batch Size 或 Sequence Length 无法解决问题。
- **解决方向**：必须采用参数高效微调（如 LoRA，大幅削减可训练参数量 $\Phi$），或采用分布式状态分片（如 ZeRO-2 / ZeRO-3、PyTorch FSDP）将优化器和梯度切分到多张卡上。

### 步骤二：核算与压缩动态激活值
若显存能够容纳静态状态，但前向或反向时仍然报错，说明显存超载来自动态激活值或框架临时缓冲区：
1. **开启激活重计算（Gradient Checkpointing）**：以大约 33% 的理论计算量增长换取激活显存成倍下降；
2. **采用梯度累积（Gradient Accumulation）**：减小微批次大小（Micro-batch Size），降低单次前向的激活峰值，多次累积梯度后再执行参数更新；
3. **采用混合精度（FP16/BF16）**：相比 FP32，半精度可将动态激活值张量所占显存减半。

---

## 练习与检验

1. **算例检验**：在标准 FP16 混合精度且搭配 AdamW 优化器的设置下，若要全参数训练一个 13B（$13 \times 10^9$ 参数）模型，仅静态模型状态（参数 + 梯度 + 优化器）理论上至少需要多少 GiB 显存？单张 80 GB 显存的显卡能否单独容纳其静态状态？
2. **机制判断**：在长文本预训练中，当序列长度从 2048 扩展到 8192 时，模型参数和优化器状态的显存占用是否会发生变化？注意力激活值显存大约会扩大到原来的多少倍？

---

## 课程目录索引

- 前置课程：
  - [/llm-100/training/what-is-pretraining.md](/llm-100/training/what-is-pretraining.md) - 大模型预训练到底在训练什么？
  - [/llm-100/training/autoregressive-lm.md](/llm-100/training/autoregressive-lm.md) - 自回归语言模型为什么能预测下一个词？
  - [/llm-100/training/mlm-vs-clm.md](/llm-100/training/mlm-vs-clm.md) - Masked Language Model 和 Causal Language Model 有什么区别？
  - [/llm-100/training/loss-calculation.md](/llm-100/training/loss-calculation.md) - 大模型训练中的 Loss 是怎么计算的？
  - [/llm-100/training/gradient-descent-backprop.md](/llm-100/training/gradient-descent-backprop.md) - 梯度下降、反向传播在大模型里如何工作？
  - [/llm-100/training/why-massive-data.md](/llm-100/training/why-massive-data.md) - 训练大模型为什么需要海量数据？
- **当前课程**：
  - [/llm-100/training/why-high-vram.md](/llm-100/training/why-high-vram.md) - 大模型训练显存构成的量化拆解与瓶颈分析
- 后续进阶课程：
  - [/llm-100/training/mixed-precision.md](/llm-100/training/mixed-precision.md) - 混合精度训练 FP16、BF16 有什么区别？
  - [/llm-100/training/混合精度训练_FP16_BF16_有什么区别.md](/llm-100/training/混合精度训练_FP16_BF16_有什么区别.md) - 延伸练习：混合精度数值稳定性
  - [/llm-100/training/parallel-strategies.md](/llm-100/training/parallel-strategies.md) - 数据并行、模型并行、流水线并行分别是什么？
  - [/llm-100/training/zero-deepspeed-fsdp.md](/llm-100/training/zero-deepspeed-fsdp.md) - ZeRO、DeepSpeed、FSDP 解决了什么问题？
  - [/llm-100/training/README.md](/llm-100/training/README.md) - 训练原理篇导读
