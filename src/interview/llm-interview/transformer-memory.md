---
title: 大模型训练与推理显存开销拆解：参数、梯度、优化器与激活值
icon: mdi:memory
tag:
- Transformer
- 显存
- 推理优化
description: 系统推导大模型训练与推理阶段的显存开销公式，拆解AdamW混合精度16字节构成与KVCache动态占用，并给出A100微调7B模型的实战分析。
---

![大模型训练与推理显存开销全景图](/images/content-rebuild/97605008dd33f0e9/cover.png)

显存溢出（CUDA Out of Memory，简称 OOM）是大模型训练与部署阶段最常遭遇的工程阻碍。在面试和架构选型中，仅凭经验估算“7B 需要几张卡”无法应对长上下文、动态并发以及不同微调配置的真实挑战。掌握训练与推理显存的物理构成和数学推导，是评估吞吐量、设计批处理策略与排查 OOM 故障的前提。

---

## 面试简述核心结论

在面试中被问及“大模型显存花在哪里”时，可以从训练与推理两个截然不同的物理过程切入：

1. **训练阶段（由模型状态与激活值共同决定）**：
   - **模型状态（Model States）**：在标准的 FP16/BF16 混合精度训练下，优化器采用 AdamW 时，每个模型参数固定需要 **16 字节**显存。包含：FP16 参数（2 字节）、FP16 梯度（2 字节）、FP32 权重副本（4 字节）、FP32 一阶动量（4 字节）、FP32 二阶动量（4 字节）。
   - **激活值显存（Activation Memory）**：前向传播为反向传播计算梯度而暂存的中间张量。它随批大小（$b$）与序列长度（$s$）线性甚至二次方增长，是长文本训练下 OOM 的主因，可通过**激活重计算（Activation Checkpointing）**将跨层常驻显存大幅降低。
2. **推理阶段（由静态权重与动态 KV Cache 主导）**：
   - **静态模型权重（Static Weights）**：无梯度与优化器状态。FP16 占用 2 字节/参数，INT8 占用 1 字节，INT4 占用 0.5 字节。
   - **KV Cache（动态显存）**：自回归解码为了避免前文重复计算，缓存每层的 Key 与 Value 张量。显存随并发请求量、序列长度和注意力层配置动态线性膨胀。
   - **临时工作空间（Scratchpad）**：单层算子执行矩阵乘法或 Softmax 所需的临时缓冲区，通常在数百兆至数吉字节之间，由最大单个算子决定，全局复用。

<ReadMoreLock />

---

## 训练显存定量推导：为什么 AdamW 混合精度需要 16 字节？

![AdamW 混合精度训练下单个参数的 16 字节显存拆解图](/images/content-rebuild/97605008dd33f0e9/fig-1.png)

大模型训练广泛采用自动混合精度（Automatic Mixed Precision，AMP），计算时使用 FP16 或 BF16。初学者常误以为“既然是 FP16 训练，每个参数就只占 2 字节”，这混淆了模型权重与训练状态的差异。

在反向传播与参数更新时，学习率与梯度的乘积（更新量）通常远小于权重当前量级（例如更新量与权重的比值小于 FP16 的尾数精度 $2^{-11}$）。若直接在 FP16 下执行加法累加，更新量会在阶码对齐时被舍入消除（即大数吃小数，导致 $W - \eta g = W$）；且历史动量统计也需高精度平滑。因此，标准混合精度训练框架会维护一份 **FP32 主权重副本（Master Weights）**，在 FP32 下累加更新后再转回 FP16 参与前向计算；而梯度本身的绝对值下溢则通常依靠损失缩放（Loss Scaling）来解决。

设大模型的参数量为 $\Phi$（例如 7B 模型即 $\Phi = 7 \times 10^9$）。采用标准 AdamW 优化器进行混合精度训练时，静态模型状态包含三部分：

### 1. 模型参数（Parameters）
- **精度**：FP16 / BF16（半精度浮点数，占 2 字节）。
- **用途**：前向传播计算与反向传播求梯度时的输入。
- **显存占用**：$2\Phi$ 字节。

### 2. 梯度（Gradients）
- **精度**：FP16 / BF16。
- **用途**：反向传播链式求导计算得出的梯度值，在更新前暂存并传递给优化器。
- **显存占用**：$2\Phi$ 字节。

### 3. AdamW 优化器状态（Optimizer States）
优化器需要在高精度下维护历史统计量并执行数值更新：
- **FP32 主权重副本（Master Weights）**：$4\Phi$ 字节。梯度反向传播后，在 FP32 精度下累加更新，更新后再转换回 FP16 参与下一轮前向传播。
- **FP32 一阶动量（$m_t$）**：$4\Phi$ 字节。记录历史梯度的指数移动平均。
- **FP32 二阶动量（$v_t$）**：$4\Phi$ 字节。记录历史梯度平方的指数移动平均。
- **优化器状态合计**：$4\Phi + 4\Phi + 4\Phi = 12\Phi$ 字节。

### 定量汇总公式
$$\text{模型状态总显存} = 2\Phi + 2\Phi + 12\Phi = 16\Phi \text{ 字节}$$

以十进制（GB，即 $10^9$ 字节）与二进制（GiB，即 $1024^3$ 字节）折算：
$$\text{显存开销 (GB)} = 16 \times \frac{\Phi}{10^9} \text{ GB}$$
$$\text{显存开销 (GiB)} = \frac{16 \times \Phi}{1024^3} \approx 14.9 \times \frac{\Phi}{10^9} \text{ GiB}$$

这意味着：**一个 7B 参数的模型，在尚未载入任何批次数据、尚未生成任何前向激活值之前，仅模型状态就需要占用约 $7 \times 14.901 \approx 104.3\text{ GiB}$（即 $112\text{ GB}$）显存。** 这从数学上解释了为什么单张标称 80GB 的 GPU 无法裸跑 7B 模型的全量微调。

---

## 训练激活值与激活重计算

如果模型状态显存是固定的，为什么加大批大小（Batch Size）或序列长度（Sequence Length）就会触发 OOM？原因在于**激活值显存（Activation Memory）**。

![反向传播依赖前向激活值的因果链条](/images/content-rebuild/97605008dd33f0e9/fig-2.png)

### 为什么反向传播必须保留激活值？
以全连接层（Linear Layer）前向过程为例：$Y = X \cdot W$。
在反向传播时，根据微积分链式法则，损失函数 $\mathcal{L}$ 对权重 $W$ 的梯度计算公式为：
$$\frac{\partial \mathcal{L}}{\partial W} = X^T \cdot \frac{\partial \mathcal{L}}{\partial Y}$$
**如果不保留前向传播时的输入张量 $X$，反向传播就无法计算权重梯度 $\frac{\partial \mathcal{L}}{\partial W}$**。因此，网络中每一层的中间输出或输入张量，都必须暂留在显存中，直到反向传播计算到该层后才能释放。

### Transformer 单层激活值模型
对于一个由 $L$ 层组成的标准 Transformer，设隐层维度为 $h$、头数为 $a$、序列长度为 $s$、批大小为 $b$：
1. **自注意力机制（Self-Attention）**：
   - $Q, K, V$ 投影输入与输出投影：占用量为 $O(b \cdot s \cdot h)$；
   - 注意力得分矩阵 $Q K^T$ 与 Softmax 概率矩阵：维度为 $b \times a \times s \times s$。当序列变长时，以 $s^2$ 二次方增长；
   - Dropout Mask：对应布尔张量。
2. **前馈网络（MLP / FFN）**：
   - 升维矩阵输入、非线性激活（GELU / SwiGLU）及降维矩阵输入：占用量为 $O(b \cdot s \cdot h)$。
3. **归一化层（LayerNorm / RMSNorm）**：
   - 输入张量及均值、方差等统计量。

当层数 $L$ 达到数十层、上下文长度 $s$ 达到数千时，前向保存的激活显存往往能达到数十甚至上百 GiB，体量完全不亚于模型状态。

![激活重计算以 20%-30% 计算时间换取巨大显存空间的原理机制](/images/content-rebuild/97605008dd33f0e9/fig-3.png)

### 激活重计算（Activation Checkpointing）机制
为了打破“显存随层数线性堆叠”的瓶颈，工程上普遍采用**激活重计算**（Gradient Checkpointing）：
- **机制设计**：前向传播时不保存每个子层（如注意力得分矩阵、GELU 激活前张量）的细粒度激活值，仅保留每个 Transformer Block 入口处的输入张量（Checkpoints）。
- **时空置换**：当反向传播计算到该 Block 时，以保留的入口输入为起点，**重新执行一次该 Block 的前向计算**，动态产生反向传播所需的中间激活；求完该 Block 的梯度后，中间激活立即释放销毁。
- **工程收益与代价**：
  - **空间节省**：跨层常驻的激活显存从所有层的总和降为“单层峰值激活 + $L$ 个 Block 入口张量”，激活显存通常可减少 60% 至 80% 以上。
  - **时间代价**：由于每个 Block 多执行了一次前向计算，训练总计算量（FLOPs）增加约 20%~30%，总训练时间相应延长。

进一步的优化是 **FlashAttention**。需要区分的是，FlashAttention 利用 GPU 高速片上 SRAM 进行分块计算与在线 Softmax 统计，主要节约了注意力矩阵在显存（HBM）与 SRAM 之间的读写带宽与落盘显存，但并没有改变标准全自注意力的算术复杂度。

---

## 推理显存构成：权重、KV Cache 与 Scratchpad

![推理阶段显存由静态权重、动态增长的 KV Cache 与临时 Scratchpad 构成](/images/content-rebuild/97605008dd33f0e9/fig-4.png)

在自回归（Autoregressive）推理阶段，不需要优化器、无需计算梯度，也不需要跨层保存反向激活图。推理显存主要由三部分组成：

### 1. 静态模型权重（Static Model Weights）
推理时不保存梯度和主权重，显存仅取决于参数量与量化精度：
- **FP16 / BF16**：$2 \times \Phi$ 字节（7B 模型约 14 GB / 13.04 GiB）；
- **INT8 量化**：$1 \times \Phi$ 字节（约 7 GB / 6.52 GiB）；
- **INT4 量化**：$0.5 \times \Phi$ 字节（约 3.5 GB / 3.26 GiB）。

### 2. 动态 KV Cache（显存瓶颈）
关于 KV Cache 的物理意义详见 [/interview/llm-interview/kv-cache.md](/interview/llm-interview/kv-cache.md)。在自回归生成时，前文生成的每个 Token 的 Key 和 Value 向量必须被缓存，否则每生成一个新 Token 都要将整个历史前文从头计算一遍。

**定量计算公式**：
设模型层数为 $n_{\text{layers}}$，Key/Value 向量的总维度为 $h_{\text{kv}}$，序列总长度为 $s$（Prompt 长度 + 已生成 Token 数），批次大小为 $b$。若以 FP16（2 字节）存储：
$$\text{KV Cache 显存 (字节)} = 2 \times 2 \times n_{\text{layers}} \times h_{\text{kv}} \times s \times b$$
- 第一个 $2$：Key 和 Value 两个独立张量；
- 第二个 $2$：FP16 格式每个元素占 2 字节；
- $h_{\text{kv}}$：在标准 MHA 中等于隐层维度 $h$；在 MQA 或 GQA 中为 $n_{\text{kv\_heads}} \times d_{\text{head}}$。

**算例对比**：
假设一个 13B 模型（$n_{\text{layers}}=40, h=5120$），标准 MHA，FP16 存储，并发批次 $b=16$，序列总长度 $s=4096$：
$$\text{KV Cache 显存} = 4 \times 40 \times 5120 \times 4096 \times 16 = 53,687,091,200 \text{ 字节} \approx 50 \text{ GiB} \approx 53.69 \text{ GB}$$
此时，动态 KV Cache（50 GiB）消耗的显存已经是模型自身权重（约 24.2 GiB / 26 GB）的 2 倍以上。长序列或高并发下，推理显存会被 KV Cache 迅速占满，这也是推动模型架构转向 [MQA / GQA 架构](/interview/llm-interview/mha-mqa-gqa.md) 的根本原因。

### 3. 工作临时内存（Scratchpad）
模型执行单层矩阵乘法、Softmax 等算子时临时申请的 Buffer。这部分显存不会随层数累积，由单层计算中最大的算子决定，通常在几百 MB 到 1~2 GiB 左右，算子执行完毕即复用。

---

## 面试实战算题：80G A100 能否全量微调 7B 模型？

![单卡 80G A100 微调 7B 模型 OOM 现场与显存缺口计算](/images/content-rebuild/97605008dd33f0e9/fig-5.png)

### 题目
> **面试官**：我们有一台配有 80GB 显存的 A100 服务器，能否使用标准 FP16 混合精度与 AdamW 优化器，对一个 7B 的模型进行全量微调（Full Fine-Tuning）？如果不加任何特殊优化直接启动，运行会发生什么？

### 参考作答步骤

1. **定量拆解模型状态（Model States）**：
   - 参数量 $\Phi = 7 \times 10^9$。
   - 混合精度 AdamW 训练下的模型状态固定开销为：
     - FP16 参数：$7 \times 2 = 14\text{ GB}$
     - FP16 梯度：$7 \times 2 = 14\text{ GB}$
     - AdamW 状态（FP32 主权重 + 一阶动量 + 二阶动量）：$7 \times 12 = 84\text{ GB}$
   - 仅模型状态显存总计：$14 + 14 + 84 = 112\text{ GB}$（折合二进制约为 $104.3\text{ GiB}$）。
2. **对比物理硬件上限**：
   - A100 标称显存为 80 GB（实际物理显存为 80 GiB，即 81,920 MiB）。
   - 仅模型状态（104.3 GiB / 112 GB）就已大幅超出 80 GB 的物理硬件上限，更不用说前向计算所需的激活显存以及 PyTorch/CUDA 运行时常驻开销（Context Cache 约 1~2 GiB）。
3. **直接运行的后果**：
   - 在优化器初始化（分配一阶与二阶动量显存）或反向传播构建计算图阶段，PyTorch 会直接抛出：
     `torch.cuda.OutOfMemoryError: CUDA out of memory.`
4. **工程追问与破局路径**：
   如果面试官接着追问：“若硬件受限只有这台 80G A100，有哪些可落地的解决方案？”

![大模型显存降维与优化策略决策树](/images/content-rebuild/97605008dd33f0e9/fig-6.png)

可给出以下三条明确的落地演进路径：
- **方案 A：保持全量微调，使用 ZeRO-Offload / FSDP-Offload**
  - 将 84 GB 的 AdamW 优化器状态卸载（Offload）到主机内存（CPU RAM），GPU 仅保留 FP16 参数与梯度（28 GB），配合激活重计算即可在 80G 内跑通，代价是 CPU-GPU 之间 PCIe 传输带宽会成为吞吐瓶颈。
- **方案 B：优化器降级，采用 8-bit Adam**
  - 使用 bitsandbytes 等库将一阶与二阶动量分块量化为 8-bit 存储（从 8 字节降至 2 字节），FP32 主权重保留 4 字节，使优化器状态显存从 $12\Phi$ 降到 $6\Phi$（42 GB）。整体模型状态显存（参数 + 梯度 + 优化器）从 112 GB 压降至 70 GB（约 65.2 GiB），为激活值腾出显存空间，配合激活重计算即可跑通微调。
- **方案 C：转向参数高效微调（PEFT / LoRA）**
  - 放弃全量微调，冻结 7B 基座参数。基座仅以 FP16 占用 14 GB（甚至 QLoRA 4-bit 仅需约 3.5 GB），仅对低秩矩阵（约占原参数 0.1%）计算梯度与 AdamW 状态，总显存开销轻松控制在 20 GB 以内，详见 [LoRA 数学原理](/interview/llm-interview/lora-math-principle.md) 与 [QLoRA 实战](/interview/llm-interview/lora-qlora.md)。

---

## 常见面试认知误区

| 常见错误说法 | 事实真相与机制解释 |
| :--- | :--- |
| **“混合精度 FP16 训练比全精度 FP32 训练省一半显存”** | **错误**。FP32 纯精度训练每参数占 16 字节（参数 4B + 梯度 4B + AdamW 8B）；FP16 混合精度因为需要维护 FP32 主权重，每参数**同样占用 16 字节**。混合精度训练的核心收益是**大幅提升 Tensor Core 计算吞吐**，而非节省模型状态显存。 |
| **“激活重计算能把激活显存彻底清零”** | **错误**。激活重计算丢弃的是 Block 内部各个细粒度算子的中间张量，每个 Block 的入口张量（Checkpoints）必须常驻显存，否则反向传播到达时无法以此为输入重新进行前向重算。 |
| **“推理时最吃显存的一定是模型权重”** | **不全面**。在长上下文（如 32K/128K）与高并发 Batch 场景下，KV Cache 的动态显存膨胀速度极快，很容易超过静态模型权重本身，成为系统吞吐量的主导瓶颈。 |
| **“训练时 KV Cache 占用了大量显存”** | **混淆概念**。自回归模型的训练通常采用 Teacher Forcing 模式进行全序列并行前向，此时整个序列由因果掩码（Causal Mask）控制，**不使用也不维护推理时的 KV Cache**，训练保存的是通用中间激活。 |
