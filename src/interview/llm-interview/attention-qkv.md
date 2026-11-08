---
title: Self-Attention 中 Q、K、V 分别是什么意思？计算过程是怎样的？
icon: mdi:brain
tag:
- 大模型面试
- Self-Attention
- QKV
description: 梳理自注意力机制中Query、Key、Value的定义、矩阵计算五步流程、FLOPs与显存复杂度推导，以及经典面试追问。
---

![Self-Attention 中 Q、K、V 的物理意义与矩阵计算流程封面图](/images/content-rebuild/c9d298f9e5ff62bb/cover.png)

在技术面试中，“自注意力机制（Self-Attention）里的 Q、K、V 到底是什么、怎么算出来的”是一道高频基础题。很多候选人能熟练默写公式 $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$，但一旦被追问“为什么不能直接拿输入特征算相关性”、“矩阵乘法具体如何降维与聚合”，回答往往变得模糊。本文先给出一段口述答案，再拆解背后的数学计算与边界条件。

---

## 面试速答：30秒口述核心机制

> **口述参考**：
> “自注意力机制是一种通过动态权重来聚合上下文信息的特征提取方法。
> 
> 输入序列经过三个独立的线性变换矩阵，分别映射为三种不同用途的表征向量：
> 1. **Query（查询向量，$Q$）**：代表当前位置发起的寻址需求，用来衡量‘我需要关注什么样的词’；
> 2. **Key（键向量，$K$）**：代表候选位置对外暴露的索引特征，用来衡量‘我能被什么样的查询匹配’；
> 3. **Value（值向量，$V$）**：代表候选位置实际承载的语义特征，用来衡量‘一旦被选中，我向外传递什么内容’。
> 
> 它的计算包含五个步骤：线性投影得到 $Q、K、V$；计算点积 $QK^T$ 获得相似度得分；除以 $\sqrt{d_k}$ 缩放数值方差；按行执行 Softmax 归一化得到注意力权重矩阵；最后与 $V$ 相乘完成加权求和。若序列长度为 $N$、隐藏层特征维度为 $d$，整个模块的时间复杂度与中间注意力矩阵的空间复杂度分别为 $O(N^2 d)$ 和 $O(N^2)$。”

![软寻址机制与传统硬寻址对比图](/images/content-rebuild/c9d298f9e5ff62bb/fig-1.png)

如图所示，传统计算机内存寻址是“硬匹配”：一个确定的地址只能找到一个唯一的物理单元，权重非 0 即 1，不可导；自注意力则是“软匹配”：Query 向量与序列中每一个 Key 向量做内积，经过 Softmax 映射为一组大于 0 且和为 1 的平滑概率分布，并对所有 Value 向量进行加权求和。这种全过程连续可微的设计，使注意力权重可以在反向传播中与整个网络一起进行梯度优化。

---

## 概念拆解：为什么需要将输入拆为 Q、K、V？

我们以一个具体的短语“**小猫 抓 蝴蝶**”作为贯穿例子，观察自注意力如何建立词与词之间的联系。

### 1. 三个角色的分工

在输入模型时，每个词首先被转换为一个初始向量 $X$。经过线性变换矩阵 $W_Q, W_K, W_V$ 处理后，每个词分别派生出 $Q, K, V$ 三个向量：

- **Query（$Q = X W_Q$）**：关注发起方的意图。例如动词“抓”的 Query 向量，主要表达寻找动作施事者（主语）和动作受事者（宾语）的特征模式。
- **Key（$K = X W_K$）**：候选方的索引标签。例如名词“小猫”的 Key 向量，突出其作为生物主体的名词属性；“蝴蝶”的 Key 向量，突出其作为被捕食目标的宾语属性。“抓”的 Query 与二者的 Key 计算点积时，都会产生较高的匹配分数。
- **Value（$V = X W_V$）**：候选方实际输出的内容。当“抓”确认需要高度关注“小猫”后，模型最终需要提取的是“小猫”的具体语义信息（如形态、状态或上下文指代）。Key 只负责判断“该不该看”，Value 负责决定“看到了什么”。

![Q、K、V 物理角色解耦与非对称匹配机制](/images/content-rebuild/c9d298f9e5ff62bb/fig-2.png)

### 2. 为什么不能直接让 $K = V$？

如果取消这种区分，强行让 $K = V$（即匹配标签和实际提取内容使用同一组表征），会给特征建模带来两个限制：

1. **检索特征与内容特征发生冲突**：用于匹配关系的特征往往比较抽象（如词性、句法角色），而用于下游理解和生成的特征往往更具体。解耦 $K$ 和 $V$ 后，Key 可以专注于对齐语法与逻辑依赖，Value 可以保留更丰富的实体与细节特征。
2. **语言依赖的方向性要求非对称建模**：在自然语言中，词与词之间的关系通常是有向的。例如在“抓 蝴蝶”中，“抓”对“蝴蝶”的依赖程度，与“蝴蝶”对“抓”的依赖程度并不完全等价。如果匹配机制和内容表征没有解耦，模型很难针对不同方向的关系单独调整权重。

<ReadMoreLock />

---

## 矩阵运算全流程拆解

现代深度学习框架中，自注意力完全通过矩阵并行计算。我们以单头注意力为例，逐步梳理输入矩阵的维度演变与运算细节。

### 1. 维度约定与参数定义

- 输入矩阵 $X \in \mathbb{R}^{N \times d}$，其中 $N$ 为序列长度（Token 数量），$d$ 为词嵌入或隐层维度。
- 变换权重矩阵为：
  $$
  W_Q \in \mathbb{R}^{d \times d_k}, \quad W_K \in \mathbb{R}^{d \times d_k}, \quad W_V \in \mathbb{R}^{d \times d_v}
  $$
  在标准的单头注意力中，通常设定 $d_k = d_v = d$（在多头注意力中，若头数为 $h$，则每个头的维度为 $d_k = d_v = d / h$）。

![自注意力矩阵计算全流程张量维度流向图](/images/content-rebuild/c9d298f9e5ff62bb/fig-3.png)

### 2. 五大标准计算步骤

整个前向传播严格分为五个算子步骤：

#### 第一步：线性投影（Linear Projection）
输入矩阵 $X$ 分别与三个权重矩阵相乘，得到全序列的查询、键、值矩阵：
$$
Q = X W_Q \in \mathbb{R}^{N \times d_k}
$$
$$
K = X W_K \in \mathbb{R}^{N \times d_k}
$$
$$
V = X W_V \in \mathbb{R}^{N \times d_v}
$$

#### 第二步：点积打分（Dot-Product Scores）
计算 Query 矩阵与 Key 矩阵转置的点积，得到两两 Token 之间的原始相似度方阵：
$$
S = Q K^T \in \mathbb{R}^{N \times N}
$$
矩阵中位于第 $i$ 行第 $j$ 列的元素 $S_{ij} = q_i k_j^T$，表示第 $i$ 个 Token 发出的 Query 与第 $j$ 个 Token 的 Key 之间的未缩放匹配分值。

#### 第三步：尺度缩放（Scaling）
将得分矩阵的每个元素除以缩放因子 $\sqrt{d_k}$：
$$
S_{\text{scaled}} = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{N \times N}
$$
当特征维度 $d_k$ 较大时，两个独立随机向量内积的方差会扩大到 $d_k$。除以 $\sqrt{d_k}$ 可以将内积的方差拉回稳定范围，避免 Softmax 进入梯度极小的平坦区。数学推导与数值实验可参考专项解析 [/interview/llm-interview/self-attention-qk.md](/interview/llm-interview/self-attention-qk.md)。

#### 第四步：行归一化（Softmax Normalization）
对缩放后的得分矩阵沿最后一维（行方向）执行 Softmax 操作，使每一行所有元素非负且和为 1：
$$
A = \text{softmax}\left(S_{\text{scaled}}\right) \in \mathbb{R}^{N \times N}
$$
其中注意力权重 $A_{ij}$ 的计算公式为：
$$
A_{ij} = \frac{\exp\left(S_{\text{scaled}, ij}\right)}{\sum_{m=1}^N \exp\left(S_{\text{scaled}, im}\right)}, \quad \sum_{j=1}^N A_{ij} = 1
$$
矩阵 $A$ 即为注意力矩阵（Attention Map），表示序列中每个位置对其他所有位置的分配权重。

#### 第五步：加权求和（Value Aggregation）
用注意力矩阵 $A$ 乘以值矩阵 $V$：
$$
\text{Output} = A V \in \mathbb{R}^{N \times d_v}
$$
输出矩阵第 $i$ 行 $\text{Output}_i = \sum_{j=1}^N A_{ij} v_j$，正是所有 Token 的 Value 向量按照第 $i$ 行权重计算出的加权和。

### 3. 微型算例：2 个 Token、2 维特征的手算验证

为直观理解数值流动，我们设置一个极简数值算例（设 $N=2, d_k=d_v=2$）。

假设线性映射后的 $Q, K, V$ 如下，缩放因子取 $\sqrt{d_k} = \sqrt{2} \approx 1.414$：
$$
Q = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}, \quad 
K = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad 
V = \begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix}
$$

![微型 2-token 算例矩阵计算推导流程](/images/content-rebuild/c9d298f9e5ff62bb/fig-4.png)

1. **计算 $S = Q K^T$**：
   $$
   K^T = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}
   $$
   $$
   S = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 1 \times 1 + 0 \times 1 & 1 \times 0 + 0 \times 1 \\ 0 \times 1 + 2 \times 1 & 0 \times 0 + 2 \times 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 2 & 2 \end{bmatrix}
   $$
2. **执行缩放 $S_{\text{scaled}} = S / \sqrt{2}$**：
   $$
   S_{\text{scaled}} = \begin{bmatrix} 1/1.414 & 0/1.414 \\ 2/1.414 & 2/1.414 \end{bmatrix} \approx \begin{bmatrix} 0.707 & 0 \\ 1.414 & 1.414 \end{bmatrix}
   $$
3. **按行计算 Softmax**：
   - 第 1 行：$\exp(0.707) \approx 2.028$，$\exp(0) = 1.000$。和为 $3.028$。  
     $A_{11} = 2.028 / 3.028 \approx 0.67$，$A_{12} = 1.000 / 3.028 \approx 0.33$。
   - 第 2 行：两项得分相同（$1.414$），归一化后各为 $0.50$。
   $$
   A = \begin{bmatrix} 0.67 & 0.33 \\ 0.50 & 0.50 \end{bmatrix}
   $$
4. **与 $V$ 相乘得到输出 $\text{Output} = A V$**：
   $$
   \text{Output} = \begin{bmatrix} 0.67 & 0.33 \\ 0.50 & 0.50 \end{bmatrix} \begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix} = \begin{bmatrix} 0.67 \times 2 + 0.33 \times 6 & 0.67 \times 4 + 0.33 \times 8 \\ 0.50 \times 2 + 0.50 \times 6 & 0.50 \times 4 + 0.50 \times 8 \end{bmatrix} = \begin{bmatrix} 3.32 & 5.32 \\ 4.00 & 6.00 \end{bmatrix}
   $$
结果显示：Token 1 的输出向量由自身 Value（权重 0.67）与 Token 2 的 Value（权重 0.33）组合而成。关于注意力机制的历史演进，可参考基础篇 [/fundamentals/dl-basics/attention-mechanism.md](/fundamentals/dl-basics/attention-mechanism.md)。

---

## 复杂度分析：为什么是 $O(N^2 d)$？

在评估自注意力性能时，通常关注**浮点运算次数（FLOPs）**和**中间激活显存占用**两项指标。

### 1. 浮点运算量（FLOPs）分解

设序列长度为 $N$，特征维度为 $d$，令 $d_k = d_v = d$。通常约定一次乘加操作记为 2 FLOPs（1 次乘法 + 1 次加法）：

| 计算环节 | 矩阵运算规格 | 输出张量形状 | 浮点运算量 (FLOPs) | 复杂度项 |
| :--- | :--- | :--- | :--- | :--- |
| **1. Q/K/V 投影** | 3 次 $(N, d) \times (d, d)$ | 3 个 $(N, d)$ | $3 \times (2 N d^2) = 6 N d^2$ | $O(N d^2)$ |
| **2. 点积打分** | $(N, d) \times (d, N)$ | $(N, N)$ | $2 N^2 d$ | $O(N^2 d)$ |
| **3. 缩放与 Softmax** | 标量除法、指数运算与逐行求和 | $(N, N)$ | 约 $c \cdot N^2$（非主导常数项） | $O(N^2)$ |
| **4. 加权聚合** | $(N, N) \times (N, d)$ | $(N, d)$ | $2 N^2 d$ | $O(N^2 d)$ |
| **5. 输出线性投影** | $(N, d) \times (d, d)$ | $(N, d)$ | $2 N d^2$ | $O(N d^2)$ |

总核心矩阵乘法 FLOPs 为：
$$
\text{FLOPs} = 8 N d^2 + 4 N^2 d
$$

![自注意力时间与空间复杂度热点拆解图](/images/content-rebuild/c9d298f9e5ff62bb/fig-5.png)

从公式中可以看到计算瓶颈的转移规律：
- **短序列场景（$N < d$）**：例如早期句子级任务（$N = 128, d = 768$），$8 N d^2$ 占比更大，主要耗时集中在参数矩阵的线性映射上。
- **长序列场景（$N \gg d$）**：当上下文长度扩展至 8k、32k 甚至更高时，$4 N^2 d$ 随着序列长度二次方快速增长，两两 Token 之间的交互计算与求和成为主要计算开销。因此时间复杂度通常表述为 **$O(N^2 d)$**。

### 2. 空间复杂度与显存占用

在模型训练的反向传播过程中，前向计算生成的中间激活张量必须保留在显存中用于梯度计算：
- 投影向量 $Q, K, V$ 占用显存：$3 \times O(N d)$；
- 注意力矩阵 $A$ 占用显存：**$O(N^2)$**。

当序列长度达到数万时，仅存储 $A$ 矩阵就需要消耗可观的高带宽显存（HBM）。这也是 FlashAttention 等显存优化算子被广泛采用的原因：通过在前向与反向传播中采用分块计算（Tiling），在高速 SRAM 内动态重计算局部注意力，从而避免将完整的 $N \times N$ 注意力矩阵写入全局显存。关于自回归推理中的显存复用机制，可参阅 [/interview/llm-interview/kv-cache.md](/interview/llm-interview/kv-cache.md)。

---

## 核心追问：直接用输入 X 计算注意力不行吗？

面试官常会追问：“既然 $Q, K, V$ 都是由同一个 $X$ 线性映射得来的，为什么不直接令 $Q = K = V = X$，用 $X X^T$ 计算注意力并乘以 $X$？省去三个权重矩阵有什么问题？”

![独立投影与直接使用 X 的计算对比及退化机制](/images/content-rebuild/c9d298f9e5ff62bb/fig-6.png)

如果省去可学习的参数矩阵，直接令 $W_Q = W_K = W_V = I$，自注意力机制会出现以下三项退化：

### 1. 得分矩阵对称，无法表示有向依赖
若 $Q = K = X$，未缩放的得分矩阵为：
$$
S = X X^T
$$
这是一个对称矩阵，即 $S_{ij} = x_i \cdot x_j = S_{ji}$。这意味着在经过 Softmax 之前，第 $i$ 个词关注第 $j$ 个词的初始分值，与第 $j$ 个词关注第 $i$ 个词完全一致。虽然按行做 Softmax 会因为每行分母不同导致最终权重产生微小数值差，但分子完全受限于无向的点积相似度，无法准确建模“动词寻找宾语”这类具有明确方向性的语法结构。

引入独立的可学习参数矩阵后，得分矩阵变为：
$$
S = (X W_Q)(X W_K)^T = X (W_Q W_K^T) X^T
$$
只要 $W_Q W_K^T$ 不是对称矩阵，就能保证 $S_{ij} \neq S_{ji}$，赋予模型刻画有向依赖的能力。

### 2. 自点积偏大，容易导致过度自关注
在直接使用 $X X^T$ 的情况下，对角线上的元素为：
$$
S_{ii} = x_i \cdot x_i = \|x_i\|^2
$$
根据柯西-施瓦茨不等式，$x_i \cdot x_j \le \|x_i\| \|x_j\|$。若序列中词向量的模长相近，$S_{ii}$ 往往是整行中最大的数值。这会导致 Softmax 后的对角线概率偏大，每个词的注意力绝大部分分配给自己，削弱了跨位置聚合上下文信息的能力。

### 3. 特征空间受限，缺少子空间投影能力
若 $V = X$，加权求和的输出 $\text{Output} = A X$ 仅仅是原始输入向量的凸组合（Convex Combination），输出特征被严格限制在输入向量张成的空间内部，无法进行跨特征维度的非线性重组。通过独立的参数矩阵映射，模型可以把输入投射到不同关注角度的子空间中，这也为多头注意力机制（Multi-Head Attention）打下了基础。关于多头分工的更多解释，可参考 [/interview/llm-interview/why-multi-head-attention.md](/interview/llm-interview/why-multi-head-attention.md) 与 [/llm-100/basics/Attention_机制到底在算什么.md](/llm-100/basics/Attention_机制到底在算什么.md)。

---

## 常见面试连环问

**问：$Q$ 和 $K$ 的维度必须一致，但 $V$ 的维度可以和它们不一样吗？**
- **答**：**数学上可以不同，实际工程中通常设为一致**。从矩阵相乘规则来看，$Q \in \mathbb{R}^{N \times d_k}$ 与 $K^T \in \mathbb{R}^{d_k \times N}$ 相乘，充要条件是两者的最后一维同为 $d_k$；而生成的注意力矩阵 $A$ 形状是 $(N, N)$，它与任意形状为 $(N, d_v)$ 的 $V$ 相乘在代数上均可进行。但在 Transformer 标准架构中，为了便于进行残差相加（$X + \text{Sublayer}(X)$），通常将输出投影回隐层维度 $d$，因此将 $d_k$ 和 $d_v$ 设为相同数值可以保持各模块张量规格整齐。

**问：Self-Attention 与 Cross-Attention 中，Q、K、V 的来源有什么区别？**
- **答**：在 Self-Attention 中，$Q、K、V$ 全部来自于同一个输入序列（例如 Encoder 内部或 Decoder 内部的当前隐层表征）；而在 Cross-Attention（交叉注意力）中，**$Q$ 来自解码器（Decoder）当前序列，而 $K$ 和 $V$ 来自编码器（Encoder）的输出序列**。这意味着解码端拿着自身的查询需求（Query），去编码端提供的信息池（Key 和 Value）中检索对应的上下文特征。关于二者的结构对比，可参考 [/interview/llm-interview/transformer-vs-llm.md](/interview/llm-interview/transformer-vs-llm.md)。
