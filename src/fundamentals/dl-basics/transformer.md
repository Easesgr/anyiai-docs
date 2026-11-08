---
title: Transformer 是什么？它是如何工作的？
icon: mdi:robot-outline
tag:
- 深度学习
- Transformer
- 大模型基础
description: 拆解 Transformer 的自注意力、因果掩码、多头机制、位置编码与 Pre-LN 架构，解析并行训练与自回归推理的工作机制。
---

在自然语言处理中，理解词与词之间的关联一直是个核心难题。例如在句子：

> “小动物 没有 穿过 马路，因为 **它** 太 累 了。”

人类能一眼看出代词“它”指的是“小动物”，而不是“马路”。但对计算机来说，要建立这种远距离关联并不容易。

2017 年，Google 团队在论文《Attention Is All You Need》（[Vaswani et al., 2017](https://arxiv.org/html/1706.03762)）中提出了 **Transformer** 架构。它彻底抛弃了以往循环神经网络（RNN）沿时间步逐步传递状态的做法，完全依靠**自注意力机制（Self-Attention）**让所有词在单一步骤内两两计算关联，并借助多头投影、位置编码和残差层归一化搭建出深度网络。如今，从 GPT 到 LLaMA 等绝大多数主流大模型，其底层算子均脱胎于这一架构。

本篇将通过贯穿实例，拆解 Transformer 的内部结构、数学推导与执行流程。

![Transformer 编码器-解码器完整拓扑结构图](/images/content-rebuild/b8f84af469756dab/cover.png)

---

## 1. 为什么抛弃循环？从 RNN 的瓶颈看 Transformer 的设计初衷

在 Transformer 出现前，序列建模的主流方案是循环神经网络（RNN、LSTM 或 GRU，详见[/fundamentals/dl-basics/rnn-lstm-gru.md](/fundamentals/dl-basics/rnn-lstm-gru.md)）。

RNN 的核心计算公式为：

$$h_t = f(h_{t-1}, x_t)$$

隐藏状态 $h_t$ 必须等待上一时刻的 $h_{t-1}$ 计算完毕后才能生成。这种时序串行依赖带来了两个严重限制：

1. **无法利用 GPU 充分并行**：处理长度为 $T$ 的文本必须走完 $T$ 步循环，长文本的训练耗时极长，显存中保存的历史状态也会拖慢批处理吞吐。
2. **远距离信息衰减**：句子首尾两个词若要发生关联，信息必须经过中间所有时间步的反复压缩与传递，极易发生梯度衰减或信息遗忘。

Transformer 的解法是将时序递推转变为**全序列集合的两两直接交互**。原始论文采用了标准的**编码器-解码器（Encoder-Decoder）**结构（以 6 层编码器与 6 层解码器堆叠为例）：

- **编码器（Encoder）**：每一层包含一个“多头自注意力层”和一个“逐位置前馈网络（FFN）”。它接收源序列，让所有词双向互相关联，输出富含全局上下文的向量表示。
- **解码器（Decoder）**：每一层在上述两个子层之间，额外插入了一个“交叉注意力层（Cross-Attention）”，用解码器当前生成状态去检索编码器的输出；同时，解码器的自注意力被施加了“因果遮蔽”，防止生成当前词时偷看未来的答案。

每个子层外部均配置了残差连接与层归一化（Layer Normalization，详见[/fundamentals/dl-basics/normalization.md](/fundamentals/dl-basics/normalization.md)），以保障反向传播梯度的平稳回传。

---

## 2. 自注意力（Self-Attention）与因果遮蔽

### 2.1 序列内部自交互：Q、K、V 的计算

关于缩放点积注意力的基础推导，我们在[/fundamentals/dl-basics/attention-mechanism.md](/fundamentals/dl-basics/attention-mechanism.md)中已作详细分析。在自注意力中，序列不是与外部向量比对，而是**自己与自己比对**。

设输入序列经过词嵌入后的矩阵为 $X \in \mathbb{R}^{T \times d_{model}}$（其中 $T$ 为词数，$d_{model}$ 为特征维度）。模型使用三个参数矩阵将每个词映射为三种角色：

- **查询矩阵（Query, $Q$）**：代表“当前词需要寻找什么样的上下文”，$Q = X W^Q$；
- **键矩阵（Key, $K$）**：代表“当前词具有什么样的特征可供匹配”，$K = X W^K$；
- **值矩阵（Value, $V$）**：代表“当前词实际包含的具体内容表征”，$V = X W^V$。

计算公式如下：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

以“小动物 没有 穿过 马路，因为 它 太 累 了”为例，当词元“它”作为 Query 与整句话所有词的 Key 进行点积匹配时，“小动物”的 Key 与“它”的 Query 计算出的内积得分最高，经过 Softmax 归一化后获得最大的权重。最终加权求和输出的向量中就自然融入了“小动物”的特征信息。

引入缩放因子 $\sqrt{d_k}$ 的目的在于：当向量维度 $d_k$ 很大时，内积结果的方差会随着维度线性增大，导致点积值极大或极小，使 Softmax 输出进入饱和区，梯度几乎为 0。除以 $\sqrt{d_k}$ 可以将方差拉回稳定范围。

### 2.2 因果遮蔽（Causal Mask）原理

在文本生成任务中，模型必须按照自左向右的时序单向生成。当预测第 $i$ 个词时，模型绝对不能读取第 $j > i$ 位的后文信息。

但在矩阵乘法 $Q K^T$ 中，所有词是全量并行计算的，第 $i$ 行天然包含了对后文所有词的点积。为了阻断未来信息泄漏，解码器自注意力层中引入了因果掩码矩阵 $M \in \mathbb{R}^{T \times T}$：

$$M_{ij} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}$$

将掩码与未归一化的得分相加后再送入 Softmax：

$$\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

根据 Softmax 定义 $\text{softmax}(z)_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$，当输入项加上 $-\infty$ 时，$e^{-\infty} = 0$。未来所有位置分配到的注意力权重被严格置为 0，无论前向计算还是反向求导，未来信息通道均被切断。

![自注意力矩阵运算与因果遮蔽原理拆解](/images/content-rebuild/b8f84af469756dab/fig-1.png)

<ReadMoreLock />

---

## 3. 多头注意力（Multi-Head Attention）

### 3.1 为什么需要分头？

在单一注意力头中，所有注意力分数经过一次 Softmax 归一化，得到的是唯一的概率分布。

然而，语言中的关联往往是多维度的。在我们的例子中，“它”既需要和主语“小动物”建立**代词指代关联**，也需要和谓语“累”建立**语法修饰关联**。如果只用单个注意力头，加权平均操作很容易顾此失彼，平滑掉微弱但关键的句法特征。

多头注意力（MHA）通过将高维特征投影到多个独立的低维子空间，让不同头并行学习不同视角的关联模式。

### 3.2 形式化计算与维度重整

设头数为 $h$，每个头的特征维度为 $d_k = d_v = d_{model} / h$。对于第 $i$ 个头（$i \in \{1, \dots, h\}$），其投影参数矩阵为：

$$W_i^Q \in \mathbb{R}^{d_{model} \times d_k}, \quad W_i^K \in \mathbb{R}^{d_{model} \times d_k}, \quad W_i^V \in \mathbb{R}^{d_{model} \times d_v}$$

每个头在各自的子空间内独立运行点积注意力：

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V) \in \mathbb{R}^{T \times d_v}$$

计算完毕后，将所有头的输出向量沿特征维度拼接（Concat），最后通过一个整合投影矩阵 $W^O \in \mathbb{R}^{h d_v \times d_{model}}$ 映射回原始维度：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

![多头注意力子空间切分、并行计算与拼接重整流程](/images/content-rebuild/b8f84af469756dab/fig-2.png)

### 3.3 算力核算：多头是否比单头更消耗运算量？

直觉上，计算 $h$ 个头似乎会让计算量膨胀 $h$ 倍。我们按照标准矩阵乘法 $A_{m \times k} \times B_{k \times n}$ 耗费 $2 m k n$ 次浮点运算（乘加各算一次）来进行定量核算。

设序列长度为 $T$，特征维度为 $d$：

1. **单头注意力（维度为 $d$）**：
   - 生成 $Q, K, V$ 投影：$3 \times (2 T \cdot d \cdot d) = 6 T d^2$
   - 计算点积矩阵 $Q K^T$：$2 T \cdot d \cdot T = 2 T^2 d$
   - 权重矩阵乘以 $V$：$2 T \cdot T \cdot d = 2 T^2 d$
   - 输出投影 $W^O$：$2 T \cdot d \cdot d = 2 T d^2$
   - **总运算量**：$8 T d^2 + 4 T^2 d$

2. **多头注意力（$h$ 个头，每个头维度 $d_k = d / h$）**：
   - 生成所有头的 $Q, K, V$ 投影：$h \times [3 \times (2 T \cdot d \cdot (d/h))] = 6 T d^2$
   - 各头计算点积矩阵：$h \times [2 T \cdot (d/h) \cdot T] = 2 T^2 d$
   - 各头加权聚合 $V$：$h \times [2 T \cdot T \cdot (d/h)] = 2 T^2 d$
   - 输出拼接后映射 $W^O$：$2 T \cdot (h \cdot d/h) \cdot d = 2 T d^2$
   - **总运算量**：$8 T d^2 + 4 T^2 d$

**结论**：在 $d_k = d_{model} / h$ 的设计下，多头注意力的理论浮点运算量与单头完全相等。多头机制并没有增加总计算成本，而是把单一的大矩阵运算拆分重构为多个并行的子空间比对，大幅提升了模型的表征容量。

---

## 4. 恢复语序：位置编码的设计原理

### 4.1 为什么自注意力缺乏位置感知？

纯粹的自注意力公式具备**置换等变性（Permutation Equivariance）**。若没有额外的标识，模型本质上是一个集合处理器。

设打乱序列顺序的置换矩阵为 $P \in \mathbb{R}^{T \times T}$（满足 $P P^T = I$）。如果将乱序后的输入 $X' = P X$ 传入注意力模块：

$$Q' = P X W^Q = P Q, \quad K' = P K, \quad V' = P V$$

代入计算：

$$\text{Attention}(Q', K', V') = \text{softmax}\left(\frac{P Q (P K)^T}{\sqrt{d_k}}\right) (P V) = P \cdot \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) P^T P V = P \cdot \text{Attention}(Q, K, V)$$

计算结果只是对原输出进行了行顺序置换，词本身的注意力表征并不包含该词在句子中的先后次序。对模型而言，“小动物 咬了 鳄鱼”和“鳄鱼 咬了 小动物”的处理方式毫无差异。因此，必须在输入端显式注入**位置编码（Positional Encoding, PE）**。

### 4.2 正弦与余弦绝对位置编码

Vaswani 等人设计了无需学习参数的正弦和余弦函数来生成位置向量。对于绝对位置 $pos \in [0, T-1]$，在其特征通道的偶数索引 $2i$ 与奇数索引 $2i+1$ 处分别计算：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

该设计具备两个重要性质：

1. **波长从短到长平滑分布**：波长 $\lambda_i = 2\pi \cdot 10000^{2i / d_{model}}$。低维通道变化极快，对应高频波形；高维通道变化平缓，对应低频波形。不同维度的组合为每个位置赋予了唯一的数值指纹。
2. **相对位置的三角变换特性**：根据三角函数和差化积公式，对于任意固定的相对偏移 $k$，位置 $pos+k$ 处的正弦与余弦值可以表示为位置 $pos$ 处分量的线性组合：

$$\begin{pmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{pmatrix} = \begin{pmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{pmatrix} \begin{pmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{pmatrix}$$

作者假设这种旋转矩阵形式能帮助网络更容易通过线性投影学习到词与词之间的相对距离。原版 Transformer 直接将位置向量与词嵌入向量逐元素相加（$X + PE$）作为最终输入。

![正弦余弦位置编码频域波长与旋转变换机制](/images/content-rebuild/b8f84af469756dab/fig-3.png)

---

## 5. 残差连接与归一化：Post-LN 与 Pre-LN

为了训练数十甚至上百层的深度网络，残差连接（$x + \text{Sublayer}(x)$）与层归一化是不可或缺的组合。但归一化层的放置顺序对反向传播稳定性有极大影响。

### 5.1 Post-LN（原版设计）：为何需要谨慎的 Warm-up？

原始论文采用的是 **Post-LN** 结构，即先做残差相加，再做 LayerNorm：

$$x_{l+1} = \text{LayerNorm}(x_l + \text{Sublayer}(x_l))$$

对该式求导可知，反向传播的梯度每次穿透主干时，都必须连乘当前层 LayerNorm 的雅可比矩阵：

$$\frac{\partial x_{l+1}}{\partial x_l} = \frac{\partial \text{LayerNorm}(z_l)}{\partial z_l} \cdot \left( I + \frac{\partial \text{Sublayer}(x_l)}{\partial x_l} \right)$$

当网络层数较深时，由于每一层方差分布不同，梯度在穿透多层 LayerNorm 导数后极易出现幅度波动或衰减。这也是原版 Transformer 训练初期非常不稳定、必须依赖多步学习率预热（Warm-up）将学习率从 0 极其平缓拉起的原因。

### 5.2 Pre-LN（现代标配）：纯净的残差高速路

后来的研究与绝大多数开源模型（如 LLaMA）转向了 **Pre-LN** 结构，即将归一化放在每个子层运算之前，残差主干只做纯相加：

$$x_{l+1} = x_l + \text{Sublayer}(\text{LayerNorm}(x_l))$$

展开至第 $L$ 层，其形式为：

$$x_L = x_0 + \sum_{l=0}^{L-1} \text{Sublayer}(\text{LayerNorm}(x_l))$$

反向传播对初始输入 $x_0$ 的梯度（设训练损失为标量 $\mathcal{L}$）为：

$$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \frac{\partial x_L}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \left( I + \sum_{l=0}^{L-1} \frac{\partial \text{Sublayer}(\text{LayerNorm}(x_l))}{\partial x_0} \right)$$

这里的恒等单位矩阵 $I$ 构成了一条无障碍传递梯度的通道。深层损失可以直接穿透多层直达底层输入，极大地提高了百层超深模型的收敛稳定性。

![Post-LN 与 Pre-LN 拓扑结构与反向梯度流向对比](/images/content-rebuild/b8f84af469756dab/fig-4.png)

---

## 6. 逐位置前馈网络（Position-wise FFN）

在自注意力子层之后，每个位置的向量都会独立输入一个两层的全连接前馈网络：

$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

在标准配置中，隐藏层维度通常拓展为特征维度的 4 倍（即 $d_{model}=512, d_{ff}=2048$）：

```python
import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # 现代模型常用 GELU 或 SwiGLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 形状: (batch_size, seq_len, d_model)
        # 升维非线性映射 -> dropout -> 降维还原回 d_model
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

# 验证维度输入输出一致性
x = torch.randn(2, 16, 512)
ffn = PositionwiseFeedForward(512, 2048)
out = ffn(x)
assert out.shape == (2, 16, 512)
```

### 为什么注意力之后一定要跟一个 FFN？

1. **引入非线性与维度重构**：自注意力本质上是对全句 Token 向量做加权凸组合，这是一种跨位置的线性混合。如果仅堆叠注意力层，模型相当于在反复进行线性组合。FFN 负责在每个 Token 内部的通道维度施加非线性变换，重构特征表征。
2. **存储静态模式关联**：近年的模型解释性研究指出，FFN 的第一层参数相当于模式检测器（Key），第二层参数相当于知识写入器（Value），大模型参数量中占比超过一半的 FFN 承担了大量事实知识与语法规则的存储任务。

![逐位置前馈网络升维映射与非线性特征重整机制](/images/content-rebuild/b8f84af469756dab/fig-5.png)

---

## 7. 训练与推理的工作机制对比

虽然 Transformer 摆脱了循环计算，但在自回归文本生成任务中，训练和推理呈现出截然不同的行为特征：

| 对比维度 | 训练阶段（Training） | 推理阶段（Inference / Generation） |
| :--- | :--- | :--- | 
| **核心机制** | 并行化前向传播（Teacher Forcing） | 单步串行自回归生成（Autoregressive） |
| **输入方式** | 真实标签序列整体切片传入 | 由 Prompt 启动，前一步生成的词作为下一步输入 |
| **前向计算次数** | 单次矩阵运算完成全序列预测 | 生成 $N$ 个词需循环执行 $N$ 次前向传播 |
| **掩码作用** | 强制屏蔽上三角防止偷看后文；结合 Padding 掩码屏蔽填充位 | Prefill 阶段与投机解码仍需因果掩码；单 Token 解码无未来 KV，但批量生成仍需 Padding 掩码 |
| **显存与计算侧重点** | 需保存各层中间激活值以备反向传播（PyTorch autograd） | 引入 KV Cache 缓存历史键值，避免重复矩阵乘法 |

### 7.1 训练期：Teacher Forcing 实现单步并行

在训练阶段，由于标注答案已知，模型无需等待上一个词生成完成。我们直接将完整目标序列右移一位后一次性送入解码器。

由于有因果掩码的遮挡，位置 1 只能看到第一项并预测第二项；位置 5 只能看到前五项并预测第六项。全序列每个位置的交叉熵损失可以在一次 GPU 矩阵乘法中并行求出，随后依靠 PyTorch 自动求导（`loss.backward()`，见 [PyTorch autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)）同步更新全部参数。

### 7.2 推理期：Prefill 与 Decode 循环

推理阶段没有完整参考答案，模型必须自回归步进，分为两个阶段：

1. **Prefill（预填充）阶段**：模型一次性读入用户输入的 Prompt，计算所有提示词的自注意力，并输出生成的第一个 Token。
2. **Decode（解码）阶段**：将新生成的 Token 拼入序列末尾，再次执行前向传递预测下一个 Token。为了避免每次生成新词时都把所有历史词重新投影一遍 $K$ 和 $V$，工程实现上会把历史计算出的 $K$ 和 $V$ 缓存在显存中，这就是 **KV Cache**。

![Transformer 并行训练与自回归单步推理时序对比图](/images/content-rebuild/b8f84af469756dab/fig-6.png)

---

## 总结与延伸

Transformer 通过**自注意力**消除了序列建模对循环传递的依赖，通过**多头机制**在同等计算量下捕捉了多视角的语义关联，通过**位置编码**注入了时序坐标，并通过 **Pre-LN** 结构保障了深层梯度的顺畅传播。

理解 Transformer 的前向张量流动与矩阵维度变化，是进一步掌握大语言模型显存开销（详见[/interview/llm-interview/transformer-memory.md](/interview/llm-interview/transformer-memory.md)）以及序列生成算法优化（详见[/interview/llm-interview/transformer-seq2seq.md](/interview/llm-interview/transformer-seq2seq.md)）的重要基础。
