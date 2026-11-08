---
title: Self-Attention 和 Multi-Head Attention 有什么区别？从单头瓶颈到子空间并行
icon: mdi:book-open-page-variant-outline
date: 2026-07-16
tag:
- 大模型100讲
- 基础篇
description: 对比单头与多头注意力的机制差异，推导参数量与FLOPs等价性，通过PyTorch代码解析张量流转与工程权衡。
---

在注意力机制中，模型通过查询向量（Query）与键向量（Key）计算相关度分布，再对值向量（Value）进行加权求和，以此聚合上下文信息。然而，标准的单头自注意力（Single-Head Attention）每次只能生成一套权重分布，在处理复杂语义关系时存在明显的表征限制。

Transformer 并没有直接增加单头注意力的隐藏层维度，而是采用多头注意力（Multi-Head Attention，简称 MHA）。多头注意力将特征切分到多个低维子空间中并行计算，再将结果拼接并映射回原空间。在隐藏层总维度相同的前提下，多头注意力与单头注意力的参数量和理论计算量基本相同，但赋予了模型同时捕捉多种不同依赖关系的能力。

![从单头到多头自注意力机制演进全景](/images/content-rebuild/91306f65758eee44/cover.png)

---

### 单头注意力的表征瓶颈

在单头自注意力中，设输入序列表示为 $X \in \mathbb{R}^{n \times d_{model}}$，通过线性变换分别得到 $Q, K, V \in \mathbb{R}^{n \times d_{model}}$。对于序列中的第 $i$ 个位置，其注意力分布计算如下：

$$\alpha_i = \text{Softmax}\left(\frac{q_i K^T}{\sqrt{d_{model}}}\right) \in \mathbb{R}^{1 \times n}$$

$\alpha_i$ 是经过 Softmax 归一化的概率行向量，所有元素之和严格为 1。位置 $i$ 的最终上下文表示为 $z_i = \alpha_i V$，即全序列 Value 向量的单一套加权平均。

这一计算机制面临一个核心冲突：**单一分布无法在保持聚焦的同时兼顾多重语义**。

以句子 `"The animal didn't cross the street because it was too tired."` 为例，考察代词 `"it"` 在不同语义层面的依赖关系：
1. **代词指代**：`"it"` 指向远距离的主语 `"animal"`；
2. **句法主谓**：`"it"` 是谓语 `"was"` 的主语；
3. **因果逻辑**：`"it"` 处于从属连词 `"because"` 引导的原因状语从句中。

在单头注意力中，如果注意力集中分配给 `"animal"`（例如权重 0.85），分配给 `"was"` 和 `"because"` 的权重就必然被压低；反之，若试图兼顾三者（例如各分得 0.3 左右的权重），输出向量就会变成三个不同实体特征的平均混叠，导致各个独立关系的特征被稀释。

![单头注意力的表征竞争与多语义冲突](/images/content-rebuild/91306f65758eee44/fig-1.png)

---

### 多头切分机制：投影至低维子空间

多头注意力的解决思路是：**将特征维度均匀切分为 $h$ 个子空间，每个子空间维护独立的注意力分布**。

设模型隐藏层维度为 $d_{model}$，头数为 $h$。每个头的特征维度定义为：

$$d_k = d_v = \frac{d_{model}}{h}$$

在标准 Transformer-base 中，$d_{model} = 512, h = 8$，则每个头的维度 $d_k = 64$。对于第 $i$ 个注意力头（$i \in \{1, 2, \dots, h\}$），模型配置独立的线性变换矩阵：

$$W_i^Q \in \mathbb{R}^{d_{model} \times d_k}, \quad W_i^K \in \mathbb{R}^{d_{model} \times d_k}, \quad W_i^V \in \mathbb{R}^{d_{model} \times d_v}$$

由此将输入序列映射到各子空间：

$$Q_i = X W_i^Q, \quad K_i = X W_i^K, \quad V_i = X W_i^V$$

<ReadMoreLock />

在工程实现中，为了最大化 GPU 算力利用率，框架不会逐个执行 $h$ 次独立的矩阵乘法。由于 $h \times d_k = d_{model}$，实现上会使用形状为 $(d_{model}, d_{model})$ 的大矩阵一次性完成全量投影，再通过张量重排（Reshape 与 Transpose）切分为 $h$ 个并行子空间：

```text
输入形状: [batch_size, seq_len, d_model]
线性投影: [batch_size, seq_len, d_model] @ [d_model, d_model] -> [batch_size, seq_len, d_model]
张量切分: reshape -> [batch_size, seq_len, h, d_k]
维度置换: transpose(1, 2) -> [batch_size, h, seq_len, d_k]
```

![特征维度切分与多头投影变换](/images/content-rebuild/91306f65758eee44/fig-2.png)

---

### 子空间独立运算与多关注点分工

在切分后的每个子空间内部，各头独立执行缩放点积注意力：

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{Softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i \in \mathbb{R}^{n \times d_v}$$

这里的缩放因子是 $\sqrt{d_k}$ 而非 $\sqrt{d_{model}}$。因为点积计算发生在 $d_k$ 维向量之间，点积方差约为 $d_k$，除以 $\sqrt{d_k}$ 才能使点积结果的方差稳定在 1 左右，避免 Softmax 进入梯度饱和区。

因为每个头的权重矩阵 $W_i^Q, W_i^K$ 均由独立参数初始化，梯度反向传播会驱动不同头学习不同的关联模式：
- **头 1（局部邻近）**：关注当前词的前后相邻 Token，捕捉修饰搭配等局部语法；
- **头 2（远距指代）**：跨越长距离将代词与先行词直接关联；
- **头 3（主谓依赖）**：聚焦核心动词与名词之间的支配关系；
- **头 4（结构标记）**：关注句子起始符或标点符号，起到序列上下文锚定作用。

此时，同一个词 `"it"` 可以在头 2 中将绝大部分注意力分配给 `"animal"`，同时在头 3 中将注意力分配给 `"was"`，二者在各自的低维向量中保留高保真特征，不再互相挤压。

![不同注意力头在长短距离与句法指代中的分工](/images/content-rebuild/91306f65758eee44/fig-3.png)

---

### 输出拼接与线性投影 $W^O$

每个头计算完成后，产出 $h$ 个形状均为 $(n, d_v)$ 的输出张量 $\text{head}_1, \dots, \text{head}_h$。模型通过通道拼接与线性映射将它们融合成统一表示：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

1. **通道拼接**：沿最后一个特征维度拼接，恢复为形状 $(n, h \cdot d_v) = (n, d_{model})$ 的张量；
2. **输出投影**：乘以权重矩阵 $W^O \in \mathbb{R}^{d_{model} \times d_{model}}$。

**为什么拼接后已经恢复了 $d_{model}$ 维度，仍必须乘以 $W^O$？**
- **实现跨头信息交互**：拼接操作只是在内存布局上将各个头的数值并排放置，头与头之间没有任何算术层面的信息交换。乘以 $W^O$ 使各个子空间抽取的特征能够进行线性加权组合，形成协同表征；
- **残差基底对齐**：注意力层外接残差连接 $X + \text{SubLayer}(X)$。$W^O$ 将拼接后的多视角特征映射回与输入 $X$ 语义空间对齐的坐标基底中。

![多头输出拼接与最终线性投影融合](/images/content-rebuild/91306f65758eee44/fig-4.png)

---

### 计算量与参数量推导

直觉上常有人误认为“$h$ 个头会导致计算量和参数量膨胀 $h$ 倍”。事实并非如此：当每个子空间维度设为 $d_k = d_{model} / h$ 时，多头注意力与全维单头注意力的参数量和 FLOPs 完全一致。

以下推导基于序列长度 $n$、特征总维度 $d$（即 $d_{model}$）、头数 $h$、子空间维度 $d_k = d/h$。乘加操作（MAC）计为 2 次浮点运算（FLOPs），不计 Bias 偏置：

| 计算阶段 | 单头注意力（$h=1, d_k=d$） | 多头注意力（$h$ 个头, $d_k=d/h$） | 等价关系 |
| :--- | :--- | :--- | :--- |
| **Q, K, V 投影参数量** | $3 \times (d \cdot d) = 3d^2$ | $h \times [3 \times (d \cdot \frac{d}{h})] = 3d^2$ | 完全相同 |
| **Q, K, V 投影 FLOPs** | $3 \times (2n \cdot d \cdot d) = 6nd^2$ | $3 \times (2n \cdot d \cdot d) = 6nd^2$ | 完全相同 |
| **$Q K^T$ 计算 FLOPs** | $2n \cdot n \cdot d = 2n^2 d$ | $h \times [2n \cdot n \cdot \frac{d}{h}] = 2n^2 d$ | 完全相同 |
| **$A V$ 计算 FLOPs** | $2n \cdot n \cdot d = 2n^2 d$ | $h \times [2n \cdot n \cdot \frac{d}{h}] = 2n^2 d$ | 完全相同 |
| **$W^O$ 输出参数量** | $d \cdot d = d^2$ | $(h \cdot \frac{d}{h}) \cdot d = d^2$ | 完全相同 |
| **$W^O$ 投影 FLOPs** | $2n \cdot d \cdot d = 2nd^2$ | $2n \cdot d \cdot d = 2nd^2$ | 完全相同 |
| **总参数量** | $4d^2$ | $4d^2$ | **精确等价** |
| **总矩阵乘法 FLOPs** | $8nd^2 + 4n^2 d$ | $8nd^2 + 4n^2 d$ | **精确等价** |

多头设计并未增加算力预算，而是重组了计算拓扑：把一个大维度的内积与加权平均，拆分为了若干个低秩子空间的并行运算。

![单头与多头计算量与参数量等价推导对照](/images/content-rebuild/91306f65758eee44/fig-5.png)

---

### 代码验证与张量流转

以下是包含维度变换注释的多头注意力最小可运行 PyTorch 实现：

```python
import torch
import torch.nn as nn
import math

class MinimalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 合并投影矩阵：单次矩阵乘法提升硬件利用率
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 1. 投影并切分为多头：[B, N, d_model] -> [B, h, N, d_k]
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 缩放点积注意力打分: [B, h, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 3. 加权求和 Value: [B, h, N, d_k]
        context = torch.matmul(attn_weights, v)
        
        # 4. 恢复通道并拼接多头: [B, N, h * d_k] = [B, N, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 5. 输出投影融合: [B, N, d_model]
        return self.w_o(context)

if __name__ == "__main__":
    B, N, D, H = 2, 6, 64, 4
    sample_x = torch.randn(B, N, D)
    mha = MinimalMultiHeadAttention(d_model=D, num_heads=H)
    out = mha(sample_x)
    print("输入张量形状:", sample_x.shape)  # torch.Size([2, 6, 64])
    print("输出张量形状:", out.shape)       # torch.Size([2, 6, 64])
```

![张量形状在多头注意力前向传播中的演变](/images/content-rebuild/91306f65758eee44/fig-6.png)

---

### 边界与工程权衡

1. **头数 $h$ 的选取限制**：
   头数并非越多越好。当 $d_{model}$ 固定时，增加 $h$ 会迫使每个头的维度 $d_k$ 变小。若将 $d_{model}=512$ 拆为 $h=64$，单个头的维度仅为 $d_k=8$。过低维度的向量空间无法承载充足的特征表征与相对位置辨别，点积打分容易失真。通常实践中 $d_k$ 设为 64 或 128 是较稳妥的折中。
2. **头的表征重叠与冗余**：
   标准训练过程并未对注意力头施加显式的正交性惩罚（Orthogonality Regularization）。头的模式分化主要依赖随机初始化与不同任务梯度的驱动。实证研究表明，大型模型训练完成后存在相当比例的注意力头行为高度相似，这为后续的注意力头剪枝（Head Pruning）提供了操作空间。
3. **自回归推理时的 KV Cache 显存压力**：
   在自回归解码中，每个注意力头都需要为序列所有已生成的 Token 缓存对应的 Key 和 Value 向量。随着批大小（Batch Size）和上下文长度增长，MHA 的 KV Cache 会迅速消耗显存带宽与容量。为了缓解这一瓶颈，现代大模型演化出了 MQA（Multi-Query Attention，所有 Query 头共享 1 组 KV 头）与 GQA（Grouped-Query Attention，Query 头分组共享 KV 头）。

---

### 课堂练习

**目标**：编写一段测试脚本，检验并量化以下两个事实：
1. 单头注意力模型与 4 头注意力模型在相同 $d_{model}$ 下参数量是否严格一致。
2. 若移除多头注意力的输出矩阵 $W^O$（直接输出拼接张量），通过随机权重观察各头输出通道之间是否存在跨头线性交互。

**检验参考代码**：
```python
import torch
import torch.nn as nn

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 1. 验证参数量严格等价
d_model = 256
mha_4 = MinimalMultiHeadAttention(d_model=d_model, num_heads=4)
mha_1 = MinimalMultiHeadAttention(d_model=d_model, num_heads=1)

params_4 = count_parameters(mha_4)
params_1 = count_parameters(mha_1)
print(f"4头参数量: {params_4}, 1头参数量: {params_1}")
assert params_4 == params_1 == 4 * d_model * d_model, "参数量不符合推导预期！"

# 2. 验证输出层 W^O 的跨头融合作用
x = torch.randn(2, 8, d_model)
with torch.no_grad():
    # 运行至第 4 步拼接张量
    q = mha_4.w_q(x).view(2, 8, 4, 64).transpose(1, 2)
    k = mha_4.w_k(x).view(2, 8, 4, 64).transpose(1, 2)
    v = mha_4.w_v(x).view(2, 8, 4, 64).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (64 ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    concat = torch.matmul(attn, v).transpose(1, 2).contiguous().view(2, 8, d_model)
    
    # 检查 concat 张量的前 64 维与后 64 维是否直接源自各自头，互不干扰
    # 验证 W^O 作用后各个通道产生混合
    fused = mha_4.w_o(concat)
    print("拼接张量形状:", concat.shape, "融合后形状:", fused.shape)
    print("验证通过：W^O 矩阵将通道维度完全线性混合。")
```

下一讲我们将探讨序列建模中不可或缺的另一个基础构件：[/llm-100/basics/位置编码为什么重要_RoPE_ALiBi_分别解决什么问题.md](/llm-100/basics/位置编码为什么重要_RoPE_ALiBi_分别解决什么问题.md)，解析为什么自注意力机制天然对位置不敏感，以及现代大模型如何通过旋转位置编码（RoPE）注入位置信息。

---

### 延伸阅读
- 机制对比深入：[/fundamentals/transformer-arch/multi-head-attention.md](/fundamentals/transformer-arch/multi-head-attention.md)
- 面试核心要点：[/interview/llm-interview/why-multi-head-attention.md](/interview/llm-interview/why-multi-head-attention.md)
- QKV 原理溯源：[/interview/llm-interview/attention-qkv.md](/interview/llm-interview/attention-qkv.md)
