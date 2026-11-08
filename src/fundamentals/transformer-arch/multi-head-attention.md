---
title: 多头注意力机制为什么有效？它是怎么工作的？
icon: mdi:head-lightbulb-outline
tag:
- Transformer架构
- Multi-Head Attention
- 注意力机制
description: 解析多头注意力如何通过低维投影子空间解决单头加权平均的表征冲突，包含参数量与计算量推导、张量流向及数值算例。
---

在了解了 [Self-Attention 中 Q、K、V 的物理意义](/fundamentals/transformer-arch/q-k-v-explained.md) 与 [缩放因子除以根号 dk 的数值稳定性原理](/fundamentals/transformer-arch/why-divide-sqrt-dk.md) 后，一个很自然的疑问是：点积注意力已经在全序列范围建立了词与词的关联，为什么 Transformer 不直接在隐层总维度 $d_{\text{model}}$ 上做一次单头注意力，而是把它拆成多个“头”（Heads）并行计算？

结论先说在前面：**单头注意力受限于单一组归一化权重，只能对所有特征维度共用一套加权平均系数，容易在处理多重关联时顾此失彼。多头注意力（Multi-Head Attention, MHA）把高维特征投影到多个低维子空间，让不同头独立关注不同性质的依赖（如句法、指代或局部修饰），最后拼合并通过线性层融合；在 $d_k = d_v = d_{\text{model}} / h$ 的设计下，整体参数量与矩阵乘法浮点计算量与全维度单头注意力几乎完全一致。**

![多头注意力机制全景流向](/images/content-rebuild/2f54802c0c33512c/cover.png)

上图展示了多头注意力的核心骨架：输入序列映射为 Q、K、V 后，被拆分进 $h$ 个低维子空间独立计算注意力，最终拼接并通过输出矩阵 $W^O$ 恢复全局维度。讲解员正引导我们顺着数据流观察特征的分流与汇总。

---

### 单头注意力的局限：同一个标量权重的妥协

以经典例句为例：

> *"The animal didn't cross the street because **it** was too tired."*

句子中的代词 **"it"** 在理解语境时需要同时处理两层关联：
1. **指代关联**："it" 究竟指代的是 "The animal" 还是 "the street"？
2. **谓语与状态关联**："it" 与后文的谓语状态 "was too tired" 的语法搭配。

单头缩放点积注意力的计算方式如下：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

对于位置 $i$ 上的查询向量 $q_i$，它与序列中所有位置 $j$ 的键 $k_j$ 计算内积并做 Softmax 归一化，得到一组标量权重 $\alpha_{ij}$（满足 $\sum_j \alpha_{ij} = 1$），该位置的输出为：

$$\text{Output}_i = \sum_{j=1}^{S} \alpha_{ij} v_j$$

![单头注意力面对多元依赖时的概率妥协](/images/content-rebuild/2f54802c0c33512c/fig-1.png)

如图所示，当代词 "it" 需要同时关联句法上的动词谓语和语义上的指代实体时，单头注意力被迫用同一个标量分布进行妥协。讲解员通过黑板上的权重指针标出了这种“拆东墙补西墙”的注意力稀释。

这意味着：**所有特征通道共享同一组空间加权系数 $\alpha_{ij}$**。如果模型把 0.8 的权重分配给 "The animal"，分配给 "tired" 的权重就最多只剩 0.2。单头注意力无法同时满足“提取指代主语”和“提取谓语状态”两种不同的查询意图，最终计算出的加权和只能是一个相互折中的模糊向量。

<ReadMoreLock />

---

### 多头机制：把特征拆分到不同的线性子空间

为了打破“全局共用单一标量权重”的瓶颈，Transformer 引入了多头子空间投影机制。设模型总隐层维度为 $d_{\text{model}}$，设定头数为 $h$，每个头的特征维度为：

$$d_k = d_v = \frac{d_{\text{model}}}{h}$$

对每一个头 $i \in \{1, 2, \dots, h\}$，分配独立的投影矩阵：

$$W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$$

每个头通过线性映射，将原始的 $d_{\text{model}}$ 维输入投射到各自的 $d_k$ 维子空间中：

$$Q_i = X W_i^Q, \quad K_i = X W_i^K, \quad V_i = X W_i^V$$

![独立线性投影将原始表征投射入多个语义子空间](/images/content-rebuild/2f54802c0c33512c/fig-2.png)

上图对比了高维原始向量投射到两个不同子空间时的状态：在子空间 1 中距离相近的句法关联词，在子空间 2 中根据指代属性被重新组织。讲解员正在坐标架前调整两个投影面。

在不同的投影子空间内，点积度量的是不同侧面的关系：
* **Head 1**：在句法子空间中拉近 "it" 与 "was tired" 的距离，专注捕获主谓关系；
* **Head 2**：在实体子空间中拉近 "it" 与 "The animal" 的距离，专注解决指代消歧；
* **Head 3**：也可以专注于局部的相邻词搭配，形成类似于短局部卷积的注意力模式。

每个头独立计算各自的注意力矩阵，互不占用归一化配额，分别产出 $\text{head}_i \in \mathbb{R}^{S \times d_v}$。

---

### 参数量与计算量等价性推导

将注意力拆成 $h$ 个头后，计算量和参数量是否膨胀了 $h$ 倍？

只要遵循标准设计 $d_k = d_v = d_{\text{model}} / h$，**多头注意力与全维度单头注意力的核心矩阵乘法参数量和浮点运算量（FLOPs）严格相等**。设序列长度为 $S$，隐层维度为 $d$：

#### 1. 线性投影阶段（Q、K、V 映射）
* **单头大注意力**（$d_k = d$）：
  * 3 个投影矩阵 $W^Q, W^K, W^V \in \mathbb{R}^{d \times d}$，参数总量为 $3 \times d^2$。
  * 投影计算量：$3 \times (2S d^2) = 6S d^2$ FLOPs。
* **多头注意力**（$h$ 个头，每个头 $d/h$ 维）：
  * 每个头有 3 个投影矩阵，大小为 $d \times \frac{d}{h}$。
  * $h$ 个头参数总量：$h \times 3 \times \left(d \times \frac{d}{h}\right) = 3 d^2$。
  * 投影计算量：$h \times 3 \times \left(2S \cdot d \cdot \frac{d}{h}\right) = 6S d^2$ FLOPs。

#### 2. 注意力计算阶段（$QK^T$ 与权重乘 $V$）
* **单头大注意力**：
  * $QK^T$ 乘法：$(S \times d) \times (d \times S) \rightarrow 2S^2 d$ FLOPs。
  * 权重乘 $V$：$(S \times S) \times (S \times d) \rightarrow 2S^2 d$ FLOPs。
  * 注意力部分总计算量：$4S^2 d$ FLOPs。
* **多头注意力**：
  * 每个头的 $Q_i K_i^T$：$(S \times \frac{d}{h}) \times (\frac{d}{h} \times S) \rightarrow 2S^2 \frac{d}{h}$ FLOPs。
  * 每个头的权重乘 $V_i$：$(S \times S) \times (S \times \frac{d}{h}) \rightarrow 2S^2 \frac{d}{h}$ FLOPs。
  * $h$ 个头合计：$h \times \left(2S^2 \frac{d}{h} + 2S^2 \frac{d}{h}\right) = 4S^2 d$ FLOPs。

加上最后的输出线性变换 $W^O \in \mathbb{R}^{d \times d}$（单头与多头同样各需 $d^2$ 参数与 $2S d^2$ 计算量），二者在总参数量和计算复杂度上完全等价。

![单头与多头计算量与参数量拆解等价性](/images/content-rebuild/2f54802c0c33512c/fig-3.png)

上图将参数矩阵与矩阵乘法拆解为单块大矩形与 $h$ 个切片小矩形。讲解员用天平示意两端计算量在数学上的严格相等，消除了“头越多计算量翻倍”的误解。

---

### 并行张量计算与数值算例

在代码实现中，多头注意力并不通过循环逐头计算，而是把所有头的投影矩阵拼成大矩阵一次性完成乘法，随后通过张量形状变换（Reshape 与 Transpose）借助显卡的批处理矩阵乘法（Batched Matmul）全并行执行。

![多头注意力的批处理张量变换与并行流向](/images/content-rebuild/2f54802c0c33512c/fig-4.png)

上图详细标示了张量形状的变化历程：从 `(B, S, D)` 经投影变换为 `(B, S, h, d_k)`，再转置为 `(B, h, S, d_k)` 送入批矩阵乘法。讲解员手持分步卡片说明转置的作用。

#### 张量流动过程
1. **一次性投影**：输入 $X \in \mathbb{R}^{B \times S \times d_{\text{model}}}$ 乘以合并的投影矩阵，直接产出包含所有头的 $Q, K, V$。
2. **维度重排**：利用 `.view(B, S, h, d_k)` 将隐藏维度展开为“头数”和“分头维度”，再通过 `.transpose(1, 2)` 转置为 `(B, h, S, d_k)`。这样就把“头索引”并入了批次维度，硬件可直接调度批矩阵乘法。
3. **并行注意力与求和**：在批次维度下执行 $Q K^T$，得到 `(B, h, S, S)` 的注意力分数图谱，缩放并沿最后一维做 Softmax 后与 $V$ 相乘，输出 `(B, h, S, d_v)`。
4. **重排与拼接**：通过 `.transpose(1, 2)` 回到 `(B, S, h, d_v)`，再通过 `.contiguous().view(B, S, h * d_v)` 完成多头在通道轴上的物理拼接。

#### 逐步最小数值算例
设序列长度 $S = 2$，输入特征维度 $d_{\text{model}} = 4$，头数 $h = 2$，分头维度 $d_k = d_v = 2$。
假设经切分后，两个头得到的 $Q, K, V$ 张量如下：

**Head 1 子空间：**
$$Q_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad K_1 = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}, \quad V_1 = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$$

1. 缩放系数分母：$\sqrt{d_k} = \sqrt{2} \approx 1.414$。
2. 点积结果：$Q_1 K_1^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$。
3. 除以 $\sqrt{2}$：$\begin{bmatrix} 0.707 & 0.707 \\ 0 & 0.707 \end{bmatrix}$。
4. 沿行做 Softmax：
   * 第 1 行：$\text{softmax}([0.707, 0.707]) = [0.5, 0.5]$
   * 第 2 行：$\text{softmax}([0, 0.707]) = [\frac{e^0}{e^0 + e^{0.707}}, \frac{e^{0.707}}{e^0 + e^{0.707}}] \approx [0.33, 0.67]$
5. 计算加权和输出 $\text{head}_1 = A_1 V_1$：
   $$\text{head}_1 = \begin{bmatrix} 0.5 & 0.5 \\ 0.33 & 0.67 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 1.0 & 1.0 \\ 0.66 & 1.34 \end{bmatrix}$$

**Head 2 子空间：**
假设在 Head 2 中，$Q_2$ 与 $K_2$ 产生的注意力矩阵分配完全不同：
$$A_2 = \begin{bmatrix} 0.9 & 0.1 \\ 0.1 & 0.9 \end{bmatrix}, \quad V_2 = \begin{bmatrix} 1 & 3 \\ 3 & 1 \end{bmatrix}$$

计算输出 $\text{head}_2$：
$$\text{head}_2 = \begin{bmatrix} 0.9 & 0.1 \\ 0.1 & 0.9 \end{bmatrix} \begin{bmatrix} 1 & 3 \\ 3 & 1 \end{bmatrix} = \begin{bmatrix} 1.2 & 2.8 \\ 2.8 & 1.2 \end{bmatrix}$$

**通道拼接（Concat）：**
将 $\text{head}_1$ 和 $\text{head}_2$ 沿通道方向横向拼合：
$$\text{Concat}(\text{head}_1, \text{head}_2) = \begin{bmatrix} 1.0 & 1.0 & 1.2 & 2.8 \\ 0.66 & 1.34 & 2.8 & 1.2 \end{bmatrix} \quad (\text{形状 } 2 \times 4)$$

![微型算例中双头的独立计算与拼接过程](/images/content-rebuild/2f54802c0c33512c/fig-5.png)

上图以数值方块矩阵还原了上述算例：左侧为 Head 1 平衡的注意力权重与输出，中间为 Head 2 对角占优的注意力与输出，右侧为横向拼接后的张量结果。讲解员指向拼接缝隙，强调此时子空间尚未发生信息混合。

---

### 为什么最后必须乘输出矩阵 $W^O$？

多头计算完成后，拼接张量 $\text{Concat}(\text{head}_1, \dots, \text{head}_h)$ 恢复了 $d_{\text{model}}$ 的维度，但这仅仅是内存地址上的相邻放置。**Head 1 算出的句法特征与 Head 2 算出的指代特征之间，此时还没有发生任何计算上的交互。**

必须乘以输出权重矩阵 $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

![输出投影矩阵 Wo 促成跨头特征的线性重组](/images/content-rebuild/2f54802c0c33512c/fig-6.png)

如图所示，拼接后的特征通道通过全连接矩阵 $W^O$ 进行密集加权求和，原本孤立存在于不同子空间的信息在这里发生交叉融合。讲解员展示了输出向量如何被送往后续的残差连接与前馈网络。

$W^O$ 在架构中起到两个实际作用：
1. **跨头特征的线性重组**：$W^O$ 的每个输出维度都是所有头通道的加权和，使不同头提取到的多元模式（如主语实体与谓语状态）能够在线性组合中形成综合表征。
2. **残差连接的基底对齐**：Transformer 子层输出要与输入 $X$ 执行残差相加（$X + \text{Sublayer}(X)$）。$W^O$ 负责把多头拼接特征投射回与输入 $X$ 语义对齐的特征基底中。

#### PyTorch 最小代码实现
```python
import torch
import torch.nn as nn
import math

class MinimalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 一次性投影 Q, K, V 提高矩阵计算效率
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        
        # 1. 线性投影并重排为 (3, B, num_heads, S, d_k)
        qkv = self.qkv_proj(x).reshape(B, S, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 2. 批量缩放点积注意力，得到注意力图谱 (B, num_heads, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 3. 权重对 V 加权求和，重排并恢复拼接通道 (B, S, D)
        context = torch.matmul(attn_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous().reshape(B, S, D)
        
        # 4. 输出投影融合
        return self.out_proj(context)

if __name__ == "__main__":
    layer = MinimalMultiHeadAttention(d_model=512, num_heads=8)
    dummy_input = torch.randn(2, 16, 512) # batch=2, seq_len=16
    out = layer(dummy_input)
    print("输入形状:", dummy_input.shape)
    print("输出形状:", out.shape)
    assert out.shape == dummy_input.shape
```

---

### 机制边界与实践权衡

1. **头数并非越多越好**：当固定总维度 $d_{\text{model}}$ 时，头数 $h$ 越大，单头维度 $d_k$ 就会按比例缩水。若 $d_k$ 过小（例如低于 16），点积度量相似度的几何容量不足，表达能力反而会退化。主流大模型通常把 $d_k$ 保持在 64 或 128 左右（例如 LLaMA-7B 采用 $d_{\text{model}}=4096, h=32$，对应 $d_k=128$）。
2. **多头的表征冗余**：多项模型分析研究表明，预训练好的 Transformer 中存在不同头功能重叠的现象，剪去部分注意力头在许多下游任务中不会引起明显性能下降。但在训练阶段，多头结构提供了多条相互独立的优化搜索路径，显著降低了参数收敛到局部劣质解的风险。
3. **多头机制本身不具备位置感知**：多头拆分解决了特征维度上的加权冲突，但如果打乱输入 token 的物理顺序，各个头的注意力权重只会发生相应的行列置换，并不包含先后次序信息。要让序列建模感知时序先后，仍必须配合 [Transformer 为什么需要位置编码？](/fundamentals/transformer-arch/positional-encoding.md)。
