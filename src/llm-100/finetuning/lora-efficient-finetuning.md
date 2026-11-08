---
title: LoRA 为什么能高效微调大模型？低秩矩阵分解与参数合并
icon: mdi:tune-vertical
tag:
- 微调与对齐
- 大模型100讲
- RLHF
description: 详解 LoRA 低秩微调的工作原理、显存节省机制、初始化推导与权重合并方案，配有可运行验证代码。
---

在微调大语言模型时，全参数微调通常需要极高的硬件成本。以一个包含 70 亿参数（7B）的模型为例，如果使用 16 位浮点数（FP16/BF16）进行全参数微调，仅模型权重本身就需要占用约 14 GB 显存；在反向传播与更新过程中，除了反向传播产生的梯度外，优化器（如 AdamW）还需要为每个参数记录一阶动量与二阶动量（混合精度下通常还包含 FP32 主权重副本），额外带来约 8 到 12 字节/参数的优化器状态开销（约 56~84 GB 显存）。此外，每次针对不同任务微调后，都需要保存一份独立的 7B 权重文件，给存储与多任务部署带来了巨大负担。

LoRA（Low-Rank Adaptation，低秩自适应微调）通过一种直接的代数方法解决了这一难题：**冻结预训练模型的全部原始权重，在关键线性层旁边并联低秩矩阵分支，仅训练这些新增的小矩阵。** 训练完成后，还可以把新增矩阵的乘积直接加回到原权重中，实现推理期零额外计算延迟。

本讲的目标是掌握 LoRA 的低秩分解原理、维度变换、非对称初始化原因、缩放参数设计以及离线权重合并方法，并通过一个可运行的 PyTorch 练习验证其正确性。

![LoRA 低秩分解与权重合并总览](/images/content-rebuild/404a32ce1d3cf157/cover.png)

---

## 低内在秩假设：参数变化量不需要占满全部维度

全参数微调假设适应新任务需要调整模型内部的每一个权重。然而，过度参数化的预训练模型在学习具体下游任务时，是否真的需要在每个维度上自由变化？

![低内在秩与子空间投影机制](/images/content-rebuild/404a32ce1d3cf157/fig-1.png)

先验研究（Li 等人，2018；Aghajanyan 等人，2020）提出了**内在维度（Intrinsic Dimension）**的概念：虽然大模型的参数位于极高维的空间中，但在适配特定下游任务（例如将通用模型微调为客服问答助手）时，参数更新的有效轨迹往往局限在一个维度远低于原始参数维度的子空间内。

LoRA 据此提出了**低内在秩（Intrinsic Rank）**假设：
设预训练权重矩阵为 $W_0 \in \mathbb{R}^{d \times k}$，其在新任务上的参数更新量记为 $\Delta W \in \mathbb{R}^{d \times k}$。该假设认为 $\Delta W$ 本质上具有很低的秩（Rank）：

$$\operatorname{rank}(\Delta W) \le r \ll \min(d, k)$$

既然 $\Delta W$ 实际起作用的维度很小，就没有必要直接去更新包含 $d \times k$ 个参数的庞大矩阵，而是可以将 $\Delta W$ 分解为两个极小矩阵的乘积。

**适用边界：** 低内在秩是一个经验假设。如果下游任务与预训练语料分布相近（例如通用指令遵循、客服对话、领域文本分类），使用极小的秩（例如 $r=8$ 甚至 $r=4$）就能取得接近全参数微调的效果；但如果目标任务与预训练知识差异极大（如全新未见的编码系统、特殊加密符号转换），较小的秩可能会出现容量受限、欠拟合的现象。

---

## 旁路分解机制：维度变换与显存节省计算

在标准的 Transformer 线性层中，输入向量 $x \in \mathbb{R}^k$ 与冻结权重 $W_0 \in \mathbb{R}^{d \times k}$ 相乘，前向计算为：

$$h = W_0 x \in \mathbb{R}^d$$

LoRA 在保持 $W_0$ 冻结不变的前提下，在旁边并联了一条低秩分支。该分支由两个连续的矩阵相乘组成：
- 降维矩阵 $A \in \mathbb{R}^{r \times k}$
- 升维矩阵 $B \in \mathbb{R}^{d \times r}$

其中秩 $r$ 满足 $r \ll \min(d, k)$。引入旁路后，前向计算公式变为：

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$

![LoRA 旁路维度变换与梯度流动](/images/content-rebuild/404a32ce1d3cf157/fig-2.png)

### 维度变换过程

利用矩阵乘法的结合律，输入向量 $x$ 先与 $A$ 计算，再与 $B$ 计算：
1. **降维**：$A x$ 将输入从原始的 $k$ 维压缩至极窄的 $r$ 维空间；
2. **升维**：$B (A x)$ 将 $r$ 维的中间特征重新投影映射回原模型的 $d$ 维输出空间；
3. **缩放相加**：乘以缩放因子 $\frac{\alpha}{r}$ 后，与主干输出 $W_0 x$ 按位相加。

### 参数量对比算例

以常见的隐藏层维度 $d = k = 4096$ 为例，设定秩 $r = 8$：
- **全参数更新矩阵 $\Delta W$**：包含 $4096 \times 4096 = 16,777,216$ 个参数（约 16.78M）。
- **LoRA 分解矩阵 $A$ 与 $B$**：包含 $r \times k + d \times r = 8 \times 4096 + 4096 \times 8 = 65,536$ 个参数（约 0.065M）。
- **参数量减少幅度**：可训练参数量缩小至原来的 $\frac{65,536}{16,777,216} \approx 0.39\%$，缩减了 99.6% 以上。

### 显存到底省在哪里？

需要明确区分训练显存的几个组成部分：
1. **静态模型权重**：基础权重 $W_0$ 依然需要保存在显存中参与前向推理（FP16 下每 1B 参数约占 2 GB 显存）。LoRA 本身不压缩这部分显存；
2. **梯度与优化器状态**：因为 $W_0$ 处于冻结状态（`requires_grad = False`），优化器完全不需要为 $W_0$ 分配梯度、一阶动量与二阶动量显存；显存中只为占总量不到 1% 的矩阵 $A$ 和 $B$ 维护优化器状态，使得微调时的显存峰值大幅降低；
3. **激活值（Activation）**：反向传播计算 $A$ 和 $B$ 的梯度时，依然需要保存前向传播的部分激活值，因此批大小（Batch Size）和上下文长度仍然会影响显存消耗。

<ReadMoreLock />

---

## 权重初始化规则：为什么矩阵 A 随机、矩阵 B 全零？

在开始训练的第一步（Step 0），LoRA 旁路的分支矩阵必须满足严格的数值约束：
- **矩阵 $A$**：采用高斯分布随机初始化，即 $A \sim \mathcal{N}\left(0, \sigma^2\right)$（通常使用 Kaiming 初始化）；
- **矩阵 $B$**：初始化为全零矩阵，即 $B = 0$。

![权重初始化因果与反向传播初态](/images/content-rebuild/404a32ce1d3cf157/fig-3.png)

### 为什么矩阵 B 初始化为全零？

在训练刚开始的瞬间（Step 0），低秩分支的权重增量为：

$$\Delta W = B A = 0 \times A = 0$$

此时模型的前向输出严格等于预训练模型的原始输出：

$$h = W_0 x + \frac{\alpha}{r} (0 \cdot A) x = W_0 x$$

这样做的目的是：**确保微调的起始状态完全继承预训练模型已有的能力。** 如果矩阵 $B$ 也采用随机分布初始化，初始时刻 $\Delta W$ 就会带有随机噪声，模型在一开始就会输出混乱的分布，破坏已有的语言能力，导致初始损失（Loss）剧增。

### 为什么矩阵 A 不能也初始化为全零？

若矩阵 $A$ 和 $B$ 同时初始化为全零，初始时刻虽然也有 $\Delta W = 0$，但在第一次反向传播计算梯度时：

$$\frac{\partial \mathcal{L}}{\partial A} = B^T \left( \frac{\partial \mathcal{L}}{\partial h} \right) x^T = 0$$
$$\frac{\partial \mathcal{L}}{\partial B} = \left( \frac{\partial \mathcal{L}}{\partial h} \right) (A x)^T = 0$$

由于两个矩阵初始全为 0，反向传播计算出的梯度也会全为 0，导致两个矩阵无法更新，发生梯度对称死锁。

而当 $A$ 为随机值、$B = 0$ 时：
- 第一步反向传播计算中，$B$ 的梯度为 $\frac{\partial \mathcal{L}}{\partial B} = \left( \frac{\partial \mathcal{L}}{\partial h} \right) (A x)^T \neq 0$；
- 矩阵 $B$ 在第一步即可根据有效梯度脱离全零状态；
- 随后步数中，由于 $B$ 不再为零，反向传播的梯度就能顺利传递给矩阵 $A$，两边协同进入正常训练。

---

## 缩放参数 alpha / r 的作用机制

在前向计算公式中，增量项乘上了一个常数比值 $\frac{\alpha}{r}$：

$$h = W_0 x + \frac{\alpha}{r} B A x$$

其中 $r$ 是低秩矩阵的秩，$\alpha$（对应配置中的 `lora_alpha`）是一个固定标量超参数（例如设为 16 或 32）。

![缩放系数 alpha 与 r 的平衡机制](/images/content-rebuild/404a32ce1d3cf157/fig-4.png)

### 为什么需要除以 r？

矩阵乘积 $B A x$ 的本质是 $r$ 个秩为 1 的向量外积求和。当我们在调优实验中增大秩 $r$（例如从 $r=4$ 增大到 $r=32$）时，累加项的增多会导致旁路输出的数值方差自然变大，进而影响到梯度的尺度。

除以 $r$ 起到了**归一化低秩输出幅度**的作用。当开发者需要尝试不同的 $r$ 来评估模型容量时，引入 $\frac{\alpha}{r}$ 能使旁路更新的数值尺度保持在相似范围内，降低了每次调整 $r$ 都必须重新微调学习率的成本。

### 实践建议

在大多数微调任务中，业界习惯将比例 $\frac{\alpha}{r}$ 保持为固定常数（通常取 1 或 2）：
- 如果设置 $r = 8, \alpha = 16$，则缩放比值为 2；
- 若后续将秩扩充至 $r = 16$，建议同步将 $\alpha$ 调整为 32，使 $\frac{\alpha}{r} = 2$ 保持不变；
- 这样可以解耦模型容量与参数更新步长，让实验对比更加可控。

---

## 微调目标模块的选择权衡

在标准 Transformer 架构中，LoRA 适配器可以并联在不同位置的线性层上。通常有两种配置方式：

![微调模块选型对比与容量权衡](/images/content-rebuild/404a32ce1d3cf157/fig-5.png)

### 方式一：仅微调注意力层（Attention-Only）

在原始 LoRA 论文中，实验主要对自注意力层的投影矩阵进行适配：
- 常见做法是仅对 Query 投影（`q_proj`）和 Value 投影（`v_proj`）注入 LoRA 分支；
- 进一步可以覆盖 `k_proj` 和 `o_proj`。

**特点：** 参数量极小，显存占用最低。适合任务形式相对固定、只需要调整输出风格或执行简单分类的场景。

### 方式二：覆盖全部线性层（All-Linear）

随着 LLaMA、Qwen 等开源模型的发展，社区实践表明，若仅微调注意力模块，在面对需要注入新知识或涉及复杂代码与数学推理的任务时，模型容量容易出现瓶颈。

目前主流微调框架（如 Hugging Face PEFT）推荐将 LoRA 应用于 Transformer 块内的全部线性层：
- **注意力模块**：`q_proj`, `k_proj`, `v_proj`, `o_proj`
- **前馈网络（FFN/MLP）**：`gate_proj`, `up_proj`, `down_proj`

**两种方案对比：**

| 对比维度 | 仅微调 `q_proj`, `v_proj` | 覆盖全部线性层 (All-Linear) |
| :--- | :--- | :--- |
| **参数量占比** | 约 $0.05\% \sim 0.1\%$ | 约 $0.2\% \sim 0.6\%$ |
| **适用场景** | 风格对话、简单分类、显存极度受限 | 领域知识微调、复杂指令遵循、代码与逻辑任务 |
| **训练开销** | 激活值显存占用更低，训练稍快 | 激活值显存略增，前向与反向耗时轻微上升 |
| **推荐参数配置** | 建议配置较高的秩（如 $r=16 \sim 32$） | 较低的秩（如 $r=8 \sim 16$）即可取得优良表现 |

工程经验显示：在显存允许的前提下，**低秩覆盖全线性层（如 $r=8$ 作用于 All-Linear）**的综合表现通常优于**高秩仅覆盖局部注意力层（如 $r=64$ 仅作用于 $W_q, W_v$）**。

---

## 生产部署：权重合并与零额外推理延迟

LoRA 在生产服务中最实用的工程特性之一，是它不会增加在线推理的延迟。

![权重融合与部署流程](/images/content-rebuild/404a32ce1d3cf157/fig-6.png)

### 离线权重融合原理

利用矩阵乘法的分配律，在线性变换中：

$$h = W_0 x + \frac{\alpha}{r} (B A) x = \left( W_0 + \frac{\alpha}{r} B A \right) x$$

因此，在模型上线前，可以直接计算出融合后的单一权重矩阵 $W_{\text{merged}}$：

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$$

权重合并完成后：
1. 旁路矩阵 $A$ 和 $B$ 可以直接从内存中卸载，仅保留计算后的 $W_{\text{merged}}$；
2. 模型的网络拓扑结构恢复为标准 Transformer 结构，完全没有增加层数、旁路或拼接操作；
3. 推理阶段的内存带宽和算力消耗与原模型完全相同，可直接无缝接入 vLLM、TensorRT-LLM 或 TGI 等高性能推理引擎。

### 多租户场景下的权衡

权重合并适用于为单个垂直任务部署独立服务。然而在多租户系统（一台服务器需要同时服务几十个不同下游任务的请求）中：
- 如果把每个任务的 LoRA 权重都与基础模型合并，显存将被几十份庞大的主干权重撑爆；
- 此时通常选择**不合并权重**，显存中只保留一份冻结的底模权重，将轻量的 LoRA 权重（每个仅几十兆字节）按需加载，利用专用的批处理引擎（如 S-LoRA、Punica）实现跨任务并发推理。

---

## 编程练习：最小 LoRA 线性层实现与验证

以下 Python 脚本使用纯 PyTorch 实现了一个最小 LoRA 线性层，并通过三个断言测试完成验证：
1. **Step 0 功能等价验证**：初始状态下，旁路输出严格为 0，模型行为与原模型一致；
2. **梯度截断验证**：基础权重不计算梯度，梯度仅流向矩阵 $A$ 和 $B$；
3. **离线合并精度验证**：合并后的单层模型输出与合并前的旁路输出数值一致。

```python
import torch
import torch.nn as nn


class MinimalLoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = alpha / r

        # 1. 模拟冻结的基础模型权重
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        # 2. 并联低秩分支：A 随机初始化，B 全零初始化
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主干输出：h_base = x @ W_0^T
        base_out = nn.functional.linear(x, self.weight)

        # 旁路输出：根据结合律，先与 A 转置相乘，再与 B 转置相乘
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """将低秩矩阵离线加回主干权重，输出标准线性层"""
        merged_weight = self.weight.data + (self.lora_B.data @ self.lora_A.data) * self.scaling
        merged_layer = nn.Linear(self.in_features, self.out_features, bias=False)
        merged_layer.weight.data.copy_(merged_weight)
        return merged_layer


# 验证逻辑与测试断言
torch.manual_seed(42)
in_dim, out_dim = 16, 32
batch_size = 2
x = torch.randn(batch_size, in_dim)

layer = MinimalLoRALinear(in_dim, out_dim, r=4, alpha=8.0)

# 验收点 1：Step 0 输出必须与原模型完全一致（因为 B=0）
with torch.no_grad():
    init_out = layer(x)
    pure_base_out = nn.functional.linear(x, layer.weight)
    diff = torch.max(torch.abs(init_out - pure_base_out)).item()
    assert diff == 0.0, f"验收失败：初始输出存在偏差，diff={diff}"
    print("✓ 验收点 1 通过：初始化时刻旁路增量严格为 0，模型行为与底模一致。")

# 验收点 2：反向传播时，基础权重不接收梯度，仅低秩参数更新
loss = layer(x).sum()
loss.backward()
assert layer.weight.grad is None, "基础权重未被成功冻结，仍有梯度！"
assert layer.lora_A.grad is not None and layer.lora_B.grad is not None, "LoRA 分支未接收到有效梯度！"
print("✓ 验收点 2 通过：基础权重梯度被完全截断，仅 A 和 B 矩阵参与优化。")

# 模拟一步参数更新
with torch.no_grad():
    layer.lora_A.data -= 0.01 * layer.lora_A.grad
    layer.lora_B.data -= 0.01 * layer.lora_B.grad

# 验收点 3：合并后的权重与合并前的推理结果严格一致
layer.eval()
with torch.no_grad():
    out_before_merge = layer(x)
    merged_module = layer.merge_weights()
    out_after_merge = merged_module(x)

    merge_diff = torch.max(torch.abs(out_before_merge - out_after_merge)).item()
    assert merge_diff < 1e-6, f"权重合并后输出发生精度偏差，diff={merge_diff}"
    print(f"✓ 验收点 3 通过：离线权重合并成功，推理误差小于 1e-6 (diff={merge_diff:.2e})，无额外计算开销。")
```

---

## 思考与检验

1. **修改秩的大小**：在上述代码中，将秩 $r$ 从 4 修改为 16，同时将 $\alpha$ 修改为 32。重新运行脚本，观察验收点 1 和验收点 3 的数值精度是否受到影响。
2. **观察全零初始化的破坏性**：尝试将 `self.lora_A` 和 `self.lora_B` 都初始化为全零，观察验收点 2 的反向传播梯度变成了什么数值，验证梯度对称死锁的现象。

---

## 本讲小结与下讲预告

本讲梳理了 LoRA 微调的核心逻辑：
1. **参数分解**：将 $d \times k$ 的全量更新矩阵分解为两个较小矩阵 $A$（$r \times k$）与 $B$（$d \times r$），大幅削减可训练参数量；
2. **显存机制**：冻结基础模型权重，免除了为几十亿参数保存优化器状态的显存开销；但基础权重本身的静态显存和前向激活值显存依然存在；
3. **初始化设计**：$A$ 采用高斯分布随机初始化，$B$ 全零初始化，保证训练起始状态与预训练底模完全一致，并打破了梯度反向传播的死锁；
4. **零额外延迟**：推理部署前可将 $\frac{\alpha}{r} B A$ 静态加回原权重，保持原网络拓扑结构不变。

尽管 LoRA 解决了优化器显存过大的问题，但对于 70B 等超大模型，即便是只加载冻结的模型权重，也需要约 140 GB 显存。在下一讲中，我们将探讨 [QLoRA 如何进一步降低微调成本？](/llm-100/finetuning/qlora-cost-reduction.md)，解析 4-bit NormalFloat 量化与分页优化器如何让大模型微调进一步普及到消费级显卡上。
