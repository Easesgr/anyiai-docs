---
title: Transformer 中 FFN 层有什么作用？从两层升降维到 SwiGLU 门控
icon: mdi:filter-variant
tag:
- 大模型面试
- FFN
- SwiGLU
description: 解析 Transformer 前馈层的作用、逐位置计算与升降维机制、键值记忆假说，以及 SwiGLU 门控激活与 8/3 中间维度对齐推导。
---

在标准 Transformer 架构中，多头自注意力层（Multi-Head Attention）与前馈神经网络层（Feed-Forward Network，FFN）交替堆叠。自注意力机制主要解决 Token 之间的上下文信息聚合与路由，而 FFN 则在每个位置上独立完成特征的非线性重映射。

从早期经典的 ReLU 双层 FFN，到 GPT 系列广泛采用的 GELU，再到主流开源大模型（如 LLaMA、Mistral 等）普遍采用的 SwiGLU，前馈层的结构与激活函数经历了明确的演进。本文从 FFN 的基本结构、知识存储假说、门控激活机制以及隐藏层维度等价对齐计算，梳理其面试与工程核心考点。

![Transformer中Attention与FFN的分工协同与FFN结构概览](/images/content-rebuild/42c56a1ec8da1ba0/cover.png)

---

### 口述简洁答案

面试中被问到“**Transformer 中的 FFN 有什么作用？为什么要演进到 SwiGLU？**”时，可分为三个层次口述：

1. **功能定位与分工**：Self-Attention 负责不同 Token 之间的跨位置信息交互，而 FFN 是**逐位置独立（Position-wise）**计算的。它通过先升维再降维的两层投影，配合非线性激活函数，对每个 Token 的表征向量进行特征提炼。研究表明，模型的大部分静态知识和事实关联主要存储在 FFN 的权重中。
2. **从标准 FFN 到 SwiGLU**：标准 FFN 仅包含单条升维-激活-降维通道；SwiGLU 引入了**门控机制（GLU）**与平滑非单调的 **Swish（SiLU）** 激活函数。它将输入分为“门控分支”和“值分支”，两者做逐元素相乘，根据输入内容动态调节特征通道的流通程度，提升了模型的表达能力与训练稳定性。
3. **参数量与计算量对齐**：标准 FFN 有 2 个投影矩阵（升维和降维），中间维度通常取 $4d$（参数量约 $8d^2$）；SwiGLU 变为 3 个投影矩阵（门控、升维、降维）。为了保持总参数量和矩阵乘法计算量（FLOPs）基本一致，工程上将中间维度缩减为约 $\frac{8}{3}d$（即约 $2.67d$），并通常对齐到 64 或 256 的倍数。

---

### 一个贯穿示例：FFN 的逐位置独立计算

假设输入句子为：

> `爱因斯坦` `提出` `了` `相对论`

1. 在 **Self-Attention 层**中，每个 Token 都会与其它 Token 交互。例如，“爱因斯坦”会通过注意力权重聚合后面“提出了相对论”的上下文信息，让当前的向量表征融入句法和语义关系。
2. 接下来进入 **FFN 层**。此时模型不再做跨 Token 的信息混合，而是让每个 Token 的向量**独立且并行**地穿过同一套前馈权重：
   * “爱因斯坦”对应的向量独立进入 FFN，完成特征提取和模式匹配；
   * “提出”对应的向量也独立进入同一套 FFN；
   * 处理每个位置的计算公式完全相同，各位置之间互不干扰。

![经典FFN两层线性变换与逐位置独立升降维过程](/images/content-rebuild/42c56a1ec8da1ba0/fig-1.png)

#### 经典结构的数学形式

在 Vaswani 等人的论文《Attention Is All You Need》中，FFN 定义如下：

$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

设输入 Token 向量 $x \in \mathbb{R}^{1 \times d_{\text{model}}}$，参数矩阵为：
* $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$：将向量从低维空间投影到高维中间层（通常 $d_{ff} = 4 d_{\text{model}}$）；
* $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$：将非线性激活后的高维表征重新压缩回原始维度。

整个过程的作用是：**升维展开使特征解耦，经非线性激活筛选后再降维写回残差流**。如果缺少中间的非线性激活函数，连续两次线性变换 $x W_1 W_2$ 将直接退化为单层矩阵乘法 $x W'$，无法提供更高阶的特征拟合能力。

---

### 知识存储假说：键值记忆网络（Key-Value Memory）

在大语言模型中，FFN 参数量通常占非嵌入层总参数量的约三分之二。为什么 FFN 需要占用这么多参数？学界常将其解释为一个**键值记忆库**。

Geva 等人（2021）在论文《Transformer Feed-Forward Layers Are Key-Value Memories》中，将忽略偏置的 FFN 前向过程进行了展开：

$$\text{FFN}(x) = g(x W_1) W_2 = \sum_{i=1}^{d_{ff}} g(x \cdot u_i) v_i$$

我们把其中的矩阵分拆来看：
* $u_i \in \mathbb{R}^{d_{\text{model}}}$ 是第一层矩阵 $W_1$ 的第 $i$ 列向量，扮演**键向量（Key）**的角色；
* $g(\cdot)$ 是非线性激活函数，$g(x \cdot u_i)$ 是输入向量 $x$ 与键向量 $u_i$ 的内积激活值，代表**匹配得分**；
* $v_i \in \mathbb{R}^{d_{\text{model}}}$ 是第二层矩阵 $W_2$ 的第 $i$ 行向量，扮演**值向量（Value）**的角色。

<ReadMoreLock />

此时，FFN 的输出本质上是：**输入向量 $x$ 与 $d_{ff}$ 个“键”进行模式匹配，将得到的匹配系数作为权重，对对应的 $d_{ff}$ 个“值向量”进行加权求和**：

$$x_{\text{out}} = x + \sum_{i=1}^{d_{ff}} a_i v_i \quad (a_i = g(x \cdot u_i))$$

以贯穿示例中的“爱因斯坦”为例：
* 输入向量 $x$ 匹配到了某些深层键向量（例如表征“理论物理学家”、“诺贝尔奖得主”的概念模式），对应的激活系数 $a_i$ 显著变大；
* 矩阵 $W_2$ 中对应的值向量 $v_i$ 包含该概念的相关属性，加权求和后叠加到残差主干中，丰富了该 Token 的上下文语义。

需要说明的是，这种“键值对应”是一个概念解释假说。在实际大模型中，大部分知识并不是由单个神经元单独记录的，而是由多个神经元以分布式叠加（Superposition）的形式协同表达。

![FFN展开为键值记忆网络检索与残差流更新示意](/images/content-rebuild/42c56a1ec8da1ba0/fig-2.png)

---

### 激活函数演进：ReLU、GELU 到 SwiGLU

FFN 的非线性能力由其激活函数决定。大模型在前馈层激活函数的选型上经历了三次代表性演进：

#### 1. ReLU 的局限
早期 Transformer 使用 $\text{ReLU}(x) = \max(0, x)$：
* **负半轴导数为 0**：如果某个神经元的输入长期处于负区间，其梯度将恒为 0，导致权重无法更新（“神经元坏死”）；
* **零点不可导**：在 $x = 0$ 处导数不连续，在训练优化初期易引起震荡。

#### 2. GELU 的平滑过渡
GPT 和 BERT 普遍采用 GELU（Gaussian Error Linear Unit）：

$$\text{GELU}(x) = x \cdot \Phi(x)$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数。GELU 在全实数域连续可导，并在负半轴允许小幅度的非单调负值，避免了 ReLU 的硬截断，提升了深层网络的收敛稳定性。

#### 3. SwiGLU 门控机制
门控线性单元（GLU）由 Dauphin 等人在 2017 年提出，其核心思路是设置双路变换：一路作为输入内容，另一路作为门控开关控制信息通过量：

$$\text{GLU}(x) = (x W_1) \odot \sigma(x W_2)$$

2020 年，Noam Shazeer 在《GLU Variants Improve Transformer》中系统评估了多种组合，提出了使用 Swish 激活函数的变体 **SwiGLU**：

$$\text{Swish}_{\beta}(x) = x \cdot \sigma(\beta x)$$

通常取 $\beta = 1$，即 $\text{Swish}_1(x) = x \cdot \text{sigmoid}(x)$（在 PyTorch 中为 `F.silu`）。

带有输出投影的完整 SwiGLU 前馈层定义如下（现代大模型通常省略偏置）：

$$\text{FFN}_{\text{SwiGLU}}(x) = \Big( \text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \Big) W_{\text{down}}$$

* **门控分支**（$W_{\text{gate}}$）：将输入投影到中间维度，经过 Swish 激活计算门控权重；
* **升维分支**（$W_{\text{up}}$）：同样投影到中间维度，保留线性值内容；
* **逐元素相乘**（$\odot$）：通过门控动态决定保留哪些特征；
* **降维投影**（$W_{\text{down}}$）：将相乘后的特征压缩回 $d_{\text{model}}$ 并写回残差流。

![ReLU、GELU与Swish激活函数曲线对比及负半轴平滑性分析](/images/content-rebuild/42c56a1ec8da1ba0/fig-3.png)

---

### 为什么 SwiGLU 的中间维度取约 8/3 倍？

引入 SwiGLU 后，权重矩阵由 2 个增加到 3 个（$W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$）。如果继续沿用传统 FFN 的 $d_{ff} = 4d$ 设置，参数量和计算量将直接增加 50%。

为了在相同的计算预算下进行对齐比较与训练，需要重新推导隐藏层维度。

![标准FFN与SwiGLU网络拓扑与三矩阵数据流对比](/images/content-rebuild/42c56a1ec8da1ba0/fig-4.png)

#### 参数量与计算量等价推导

设输入输出维度均为 $d$（忽略偏置项）：

1. **标准两层 FFN（隐藏维度 $d_{ff} = 4d$）**：
   * 矩阵 $W_1 \in \mathbb{R}^{d \times 4d}$，参数量 $4d^2$；
   * 矩阵 $W_2 \in \mathbb{R}^{4d \times d}$，参数量 $4d^2$；
   * **总参数量**：$P_{\text{standard}} = 4d^2 + 4d^2 = 8d^2$；
   * **单 Token 矩阵乘法 FLOPs**（每个乘加算 2 次）：$2 \times (2 \times d \times 4d) = 16d^2$。

2. **SwiGLU 前馈层（设隐藏维度为 $d_{ff}'$）**：
   * 矩阵 $W_{\text{gate}} \in \mathbb{R}^{d \times d_{ff}'}$，参数量 $d \cdot d_{ff}'$；
   * 矩阵 $W_{\text{up}} \in \mathbb{R}^{d \times d_{ff}'}$，参数量 $d \cdot d_{ff}'$；
   * 矩阵 $W_{\text{down}} \in \mathbb{R}^{d_{ff}' \times d}$，参数量 $d \cdot d_{ff}'$；
   * **总参数量**：$P_{\text{SwiGLU}} = 3 d \cdot d_{ff}'$；
   * **单 Token 矩阵乘法 FLOPs**：$3 \times (2 \times d \cdot d_{ff}') = 6 d \cdot d_{ff}'$。

3. **对齐约束**：
   若要求总参数量与矩阵乘法 FLOPs 守恒：

   $$3 d \cdot d_{ff}' \approx 8d^2 \implies d_{ff}' \approx \frac{8}{3}d \approx 2.67d$$

#### 工程中的硬件对齐取整
在实际工程实现中，中间维度通常会进一步对齐到 64 或 256 的整数倍，以匹配 GPU Tensor Core 矩阵分块（Tiling）的高效执行。

以 **LLaMA-7B** 为例：
* $d_{\text{model}} = 4096$；
* 理论维度：$\frac{8}{3} \times 4096 \approx 10922.67$；
* 向上对齐到 256 的整倍数后，选取 $d_{ff}' = 11008$（因为 $11008 = 256 \times 43$）。

![标准FFN与SwiGLU在8比3维度调整下的参数量与计算量等价推导](/images/content-rebuild/42c56a1ec8da1ba0/fig-5.png)

---

### 代码实现：标准 FFN 与 SwiGLU 对比

以下为标准 GELU FFN 与 SwiGLU FFN 的 PyTorch 实现，并验证其参数量守恒关系：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardFFN(nn.Module):
    """标准两层前馈网络（以 GELU 为例）"""
    def __init__(self, d_model: int):
        super().__init__()
        d_ff = 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        return self.w2(F.gelu(self.w1(x)))

class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈网络（维度对齐到 8/3 倍并按硬件对齐）"""
    def __init__(self, d_model: int, multiple_of: int = 256):
        super().__init__()
        # 8/3 * d_model 向上取整到 multiple_of 的倍数
        hidden_dim = int(2 * (4 * d_model) / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim
        
        # 三个投影矩阵
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up   = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU (Swish-1) 门控与值分支逐元素相乘
        gate = F.silu(self.w_gate(x))
        val = self.w_up(x)
        return self.w_down(gate * val)

# 参数量守恒验证（以 d_model = 4096 为例）
d_model = 4096
std_ffn = StandardFFN(d_model)
swiglu_ffn = SwiGLUFFN(d_model, multiple_of=256)

p_std = sum(p.numel() for p in std_ffn.parameters())
p_swi = sum(p.numel() for p in swiglu_ffn.parameters())

print(f"标准 FFN 中间维度: {4 * d_model}, 参数量: {p_std}")
print(f"SwiGLU 中间维度: {swiglu_ffn.hidden_dim}, 参数量: {p_swi}")
print(f"参数量相对偏差: {abs(p_swi - p_std) / p_std:.4%}")
# 运行输出：偏差约为 0.78%，在保持计算预算基本一致的前提下完成了结构转换
```

---

### 面试高频追问与常见误区

#### 追问 1：SwiGLU 多了一个矩阵投影和逐元素相乘，访存开销变大，工程上如何优化？

**回答思路**：
1. **经验收益更高**：在相同训练计算量预算下，多篇实验表明 SwiGLU 相比标准 ReLU/GELU 显著降低了模型预训练的验证集困惑度（Perplexity），模型容量收益高于额外的访存开销。
2. **算子融合（Operator Fusion）**：SwiGLU 中间产生的 `gate` 和 `val` 张量如果都写回全局显存（HBM），会导致显存带宽瓶颈。在现代推理/训练框架（如 FlashFeedForward、vLLM 或 Triton 自定义内核）中，通常将 `Swish` 激活与逐元素相乘融合为一个 Kernel，在 GPU 的片上高速缓存（SRAM）中一次性完成计算，避免了中间大张量在 HBM 与片上缓存之间的反复搬运。

![SwiGLU算子融合优化显存带宽与访存瓶颈解析](/images/content-rebuild/42c56a1ec8da1ba0/fig-6.png)

#### 追问 2：为什么标准 Self-Attention 没有引入类似 SwiGLU 的门控？

**回答思路**：
Self-Attention 模块本身就是一个动态寻址系统：其 Softmax 注意力矩阵直接依赖于当前输入的 Query 和 Key，计算得到的权重天然是动态且依赖数据的（Data-dependent）。而标准 FFN 的权重在训练完成后是固定的静态映射，缺少基于当前输入的动态调节机制。引入 GLU 门控分支，以相对低廉的逐元素相乘为 FFN 补充了输入自适应筛选的能力。

#### 常见误区辨析

* **误区一：SwiGLU 增加了 50% 的模型参数**。
  * **事实**：如果隐藏维度仍保持 $4d$，参数确实会增加 50%。但现代大模型在采用 SwiGLU 时，普遍将中间维度按比例调整为约 $\frac{8}{3}d$，使得总参数量和计算量与两层标准 FFN 保持持平。
* **误区二：GLU 是 Noam Shazeer 提出的**。
  * **事实**：门控线性单元（GLU）由 Dauphin 等人在 2017 年提出（《Language Modeling with Gated Convolutional Networks》）。Noam Shazeer 在 2020 年的贡献是系统验证了 GLU 与 Swish、GELU 等平滑激活函数在 Transformer 架构下的组合收益。
* **误区三：FFN 中每个神经元都单独对应现实世界的一个独立知识点**。
  * **事实**：键值记忆是对 FFN 计算形态的数学类比。大模型普遍存在“多义性（Polysemanticity）”和“分布式叠加（Superposition）”现象，一个复杂实体或事实往往由成百上千个神经元的激活模式共同编码，不能等同于传统关系型数据库的一对一单行记录。

---

### 推荐相关阅读

* 架构基础：[Transformer 和 LLM 有哪些区别](/interview/llm-interview/transformer-vs-llm.md)（掌握 Transformer 基础子模块构成）
* 注意力对照：[Multi-Head Attention、MQA 和 GQA 有什么区别？](/interview/llm-interview/mha-mqa-gqa.md)（了解注意力层与前馈层的参数分布差异）
* 归一化配合：[LayerNorm 和 RMSNorm 有什么区别？](/interview/llm-interview/layernorm-rmsnorm.md)（理解现代大模型如何配合 Pre-RMSNorm 保障深层梯度流动）
* 显存分析：[Transformer 的哪个部分最占用显存？](/interview/llm-interview/transformer-memory.md)（深入了解预训练与推理期间参数与激活显存分布）
