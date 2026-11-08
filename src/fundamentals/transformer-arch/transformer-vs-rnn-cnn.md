---
title: Transformer 相比 RNN 和 CNN 有什么优势？信息传递路径与并行计算解析
icon: mdi:swap-horizontal-bold
tag:
- Transformer架构
- RNN
- CNN
- 对比分析
description: 对比 Transformer、RNN 与 CNN 在序列建模中的信号传递路径、训练并行度与计算复杂度，解析自注意力机制的优势与工程取舍。
---

![序列建模范式演进封面图](/images/content-rebuild/7efc8cd652d68c6b/cover.png)

处理自然语言等序列数据时，模型的核心任务是把一串词元（Token）转化为包含上下文信息的特征向量。

例如面对这样一句话：
> **“虽然 小明 早上 遇到了 严重的 交通 堵塞 ， 但 他 依然 准时 赶到了 办公室 。”**

要让模型理解代词“他”指的是“小明”，或者判断“准时赶到”与前面的“虽然……遇到了堵塞”之间的转折呼应，网络必须让相隔较远的词元之间能够顺畅交换信息。

在 Transformer 出现前，主流架构分为两派：一派是以 LSTM、GRU 为代表的**循环神经网络（RNN）**，另一派是以 ByteNet、ConvS2S 为代表的**卷积神经网络（CNN）**。2017 年，Google 团队在论文《Attention Is All You Need》中提出了完全摒弃循环和卷积的 **Transformer**。它之所以迅速取代传统架构，关键就在于解决了两个核心问题：**两个词之间传递信息要走多少步**，以及**GPU 在训练时能不能全句并发计算**。

## 衡量序列架构的两个核心标尺

评估不同网络结构在长序列和现代并行算力上的表现，学术界通常关注两个指标：

1. **最大信号路径长度（Maximum Path Length）**：网络中任意两个词元在前向传播传递信息或反向传播传递梯度时，中间需要经过的操作步骤数。步骤越多，信号在逐层非线性变换中越容易衰减或混叠，长距离依赖就越难建立。
2. **单样本顺序操作步数（Sequential Operations）**：在并行硬件（如 GPU）上处理单个样本时，由于数据前后依赖而必须按先后顺序执行的串行步数。串行步数越低，硬件的多核心并行效率就越高。

下面我们结合开头的例句，看这三类架构是如何处理这两个指标的。

## RNN 的链式传递：串行等待与长程衰减

循环神经网络处理序列的方式就像“传话”。它沿着时间步依次更新隐藏状态：

$$h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

其中 $x_t$ 是当前位置的输入向量，$h_{t-1}$ 是前一个位置传递过来的隐状态，$W$ 是权重矩阵，$f$ 通常是 $\tanh$ 等激活函数。

![RNN 沿时间维度的链式递推与反向传播依赖图](/images/content-rebuild/7efc8cd652d68c6b/fig-1.png)

在我们的例句中，“小明”位于第 2 位，“他”位于第 10 位。要让位置 10 的“他”获取“小明”的信息：
* **前向传递必须串行走过中间 8 步**：必须先算完“早上”，再算“遇到了”，一直串行递推到“他”。这意味着**最大信号路径长度是 $O(n)$**。
* **单样本在时间维度无法并行**：即便在训练时整句话已经全部输入，但计算 $h_t$ 必须等待上一时刻的 $h_{t-1}$ 输出。对于长度为 $n$ 的句子，必须串行执行 $n$ 步，即**顺序操作步数是 $O(n)$**。GPU 的成千上万个计算核心只能等待时间步逐步推进，单样本计算无法充分并发。
* **梯度连乘衰减**：在反向传播计算梯度时，误差从第 10 步传回第 2 步，需要沿着链条连续相乘多个雅可比矩阵（$\prod \frac{\partial h_k}{\partial h_{k-1}}$）。尽管 LSTM 和 GRU 引入门控机制缓解了短程梯度消失，但在跨越几十乃至几百步之后，早期的细粒度信息依然容易被反复的线性变换和压缩操作冲淡。

<ReadMoreLock />

## CNN 的局部滑动：感受野扩张与层数成本

为了解决 RNN 必须一步步串行计算的问题，后续研究尝试使用卷积神经网络（CNN）建模文本，典型代表包括 Facebook 的 ConvS2S 与 DeepMind 的 ByteNet。

卷积核在整条序列的所有局部窗口上可以同时并发计算，单层卷积内部不需要等待前面的词算完，顺序操作步数降到了 $O(1)$。

但卷积的局限在于**局部视野**。一个大小为 $k$ 的卷积核，单层只能看到相邻的 $k$ 个词元。要让相距较远的两个词产生关联，必须像搭积木一样向上堆叠多层网络，逐步扩大感受野：

![CNN 在序列上的感受野扩展机制](/images/content-rebuild/7efc8cd652d68c6b/fig-2.png)

不同卷积设计扩大感受野的效率不同：
* **标准卷积堆叠（如 ConvS2S）**：每增加一层网络，感受野只增加常数跨度（通常为 $k-1$）。如果要让相距为 $n$ 的两个词（例如相距 1000 个词元）建立联系，网络所需的堆叠层数随距离线性增加，**最大路径长度为 $O(n/k)$**。
* **膨胀卷积堆叠（如 ByteNet）**：通过在卷积核采样点中插入指数级间隔（Dilation Rate 呈 $1, 2, 4, 8, \dots$ 翻倍），高层的感受野呈指数级放大。此时连接两个相距 $n$ 的词元所需的网络层数缩短为对数级，**最大路径长度为 $O(\log_k n)$**。

尽管膨胀卷积大幅降低了路径步数，但远距离信息依然需要穿过多层中间隐藏层和非线性激活。随着层数增加，局部细节可能逐层损耗，且网络深度直接受到模型参数量的制约。

## Transformer 的自注意力机制：一步直达与全序列并发

Transformer 完全放弃了循环与卷积，采用**自注意力机制（Self-Attention）**让序列中的词元两两直接计算关联。

![三种架构的信息传递拓扑与路径长度对比](/images/content-rebuild/7efc8cd652d68c6b/fig-3.png)

在计算例句中的“他”时，自注意力层会直接把“他”的查询向量（Query）与全句所有词元（包括“小明”、“堵塞”）的键向量（Key）进行点积运算，计算相似度并归一化为注意力权重，然后直接对各个词的值向量（Value）做加权求和。

这带来了两项直接变化：
1. **最大路径长度恒为 $O(1)$**：无论两个词在句子中相隔 3 个词还是 3000 个词，在同一个自注意力层中，词元 $i$ 与词元 $j$ 的交互都只需要通过单次点积加权完成，反向传播的梯度也是一步传达，不再随距离拉长而经过多次层间连乘。
2. **单样本训练顺序操作步数为 $O(1)$**：当整句输入已知时，所有词元之间的点积与加权可以通过标准的矩阵乘法（$Q K^T$ 以及乘 $V$）一次性计算完成。GPU 可以直接满载处理整个长矩阵，彻底消除了时间维度上的串行等待。

论文 Table 1 总结了三种机制的理论指标对比（设序列长度为 $n$，特征维度为 $d$，卷积核大小为 $k$）：

| 架构类型 | 代表模型 | 最大信号路径长度 | 训练顺序操作步数 | 每层计算复杂度 |
| :--- | :--- | :--- | :--- | :--- |
| **循环神经网络 (RNN)** | LSTM, GRU | $O(n)$ | $O(n)$ | $O(n \cdot d^2)$ |
| **线性卷积堆叠 (CNN)** | ConvS2S | $O(n/k)$ | $O(1)$ | $O(k \cdot n \cdot d^2)$ |
| **膨胀卷积堆叠 (CNN)** | ByteNet | $O(\log_k n)$ | $O(1)$ | $O(k \cdot n \cdot d^2)$ |
| **纯自注意力 (Transformer)**| 原版 Transformer | $O(1)$ | $O(1)$ | $O(n^2 \cdot d)$ |

### 理论数值推演示例
设序列长度 $n = 1024$，卷积核大小 $k = 2$：
* **RNN** 从首个词元递推到第 1024 个词元需要依次串行走完 **1023 步**隐状态转移（处理全序列共需 1024 步），无法在单样本内部并发；
* **标准卷积（ConvS2S）** 若每层感受野增量为 $k-1 = 1$，要覆盖两端相距 1023 的跨度，理论上需要堆叠约 **1023 层**；
* **膨胀卷积（ByteNet）** 若按 2 的幂次扩张，覆盖 1024 跨度需要经过 $\log_2(1024) = \mathbf{10}$ 个网络操作层；
* **Transformer** 在单层内任意两个位置的连接步数恒为 **1 步**，顺序操作步数为 $O(1)$。

## 结构代价：加权平均的分辨率损失与多头注意力

$O(1)$ 的全连接直达虽然缩短了路径，但也引入了新的问题。Vaswani 等人在原论文第 2 节中明确指出了自注意力的代价：

> *"...at the cost of reduced effective resolution due to averaging attention-weighted positions..."*

这种有效分辨率的损失是如何产生的？
在单个注意力计算中，每个位置输出的新表征是所有位置 Value 向量的加权平均：

$$\text{Output}_i = \sum_{j=1}^n \alpha_{ij} V_j$$

其中权重 $\alpha_{ij} \ge 0$ 且 $\sum_j \alpha_{ij} = 1$。
回到我们的例句：在分析“他”这个词时，模型既需要关注主语“小明”（处理代词指代），又需要兼顾动作“赶到了”（维持谓语关联）。如果只用一组权重进行简单的加权平均，多个词的特征向量混在一起，就会像调色板把多种颜料混合成灰褐色一样，互相冲淡彼此鲜明的语义特征。

![单头注意力分辨率损失与多头注意力子空间解耦机制](/images/content-rebuild/7efc8cd652d68c6b/fig-4.png)

为了解决这个问题，Transformer 引入了 **Multi-Head Attention（多头注意力）**：
1. 模型不直接在原特征维度上算一组注意力，而是通过矩阵投影，将特征切分到 $h$ 个不同的低维子空间（例如原始特征 512 维，拆分成 8 个头，每头 64 维）；
2. 每个注意力头在独立的子空间里各自计算注意力权重分布；
3. 不同的头分工关注不同维度的关系：一个头专门关注代词与先行词（“他”与“小明”），另一个头专门关注主谓搭配（“他”与“赶到了”）；
4. 最后把所有头的输出拼接并通过线性变换整合还原，既保留了 $O(1)$ 的信息通路，又避免了单次加权平均造成的信息混叠。

关于多头注意力的矩阵推导与参数拆分细节，可以在模块后续的 [/fundamentals/transformer-arch/multi-head-attention.md](/fundamentals/transformer-arch/multi-head-attention.md) 中继续深入。

## 代码验证：首尾词元梯度传递敏感度对比

我们可以写一段精简的 PyTorch 脚本，直观观察在固定步长下，序列末尾的输出对序列开头输入的梯度变化。

为了保证比较基准一致，我们考察相同的问题：**输入序列首端词元 $x_0$ 发生微小变化时，序列末尾输出所产生的梯度敏感度（$\frac{\partial \text{Output}_{\text{last}}}{\partial x_0}$）**。

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
seq_len = 32
d_model = 16

# 准备测试序列输入，开启对输入的梯度追踪
x_rnn = torch.randn(seq_len, 1, d_model, requires_grad=True)
x_attn = x_rnn.clone().detach().requires_grad_(True)

# 1. 模拟单层 RNN 链式前向传递
rnn = nn.RNN(input_size=d_model, hidden_size=d_model, batch_first=False)
output_rnn, _ = rnn(x_rnn)
# 考察末尾时刻输出对开头输入 x_rnn[0] 的敏感度
loss_rnn = output_rnn[-1].sum()
loss_rnn.backward()
grad_rnn_x0 = x_rnn.grad[0].norm().item()
print(f"RNN 序列首端输入 x[0] 的梯度范数: {grad_rnn_x0:.6e}")

# 2. 模拟 Transformer 缩放点积自注意力前向传递
# 为保持结构精简，使用单头自注意力核心计算
q = x_attn
k = x_attn
v = x_attn

# 计算全序列两两注意力分数矩阵 (seq_len, seq_len)
scores = torch.matmul(q.squeeze(1), k.squeeze(1).transpose(0, 1)) / (d_model ** 0.5)
attn_weights = torch.softmax(scores, dim=-1)
output_attn = torch.matmul(attn_weights, v.squeeze(1))

# 考察末尾位置输出对开头输入 x_attn[0] 的敏感度
loss_attn = output_attn[-1].sum()
loss_attn.backward()
grad_attn_x0 = x_attn.grad[0].norm().item()
print(f"Transformer 序列首端输入 x[0] 的梯度范数: {grad_attn_x0:.6e}")
```

在未作特殊正交初始化或门控调优的情况下运行上述脚本：
* 在 RNN 中，信号穿过 32 步连续的 $\tanh$ 激活与权重相乘，首端输入 $x_0$ 传到末尾的梯度发生显著衰减；
* 在 Transformer 中，末尾输出直接包含带有注意力权重的 $v_0$ 项，反向传播通过一层缩放点积注意力即可回传，不存在沿时序的矩阵链式连乘。需要注意，这里的 $O(1)$ 路径是指操作步数与依赖层数为常数，梯度数值仍会受 Softmax 归一化权重稀释的影响，并非跨度拉长后梯度恒定不变；此脚本为教学机制示意，不代表多层深层模型的实际训练收敛表现。

![长程依赖衰减对比实验与响应曲线](/images/content-rebuild/7efc8cd652d68c6b/fig-5.png)

## 优势背后的工程代价与局限

Transformer 取代循环与卷积，是用更多的显存和计算量换取全局直连的路径与全序列并发的硬件利用率。

这一设计在带来高并行度与短路径的同时，也伴随着具体的硬件开销与适用条件：

![序列建模架构多维权衡对比全景图](/images/content-rebuild/7efc8cd652d68c6b/fig-6.png)

1. **$O(n^2)$ 计算与显存开销**：
   自注意力需要计算所有词元两两之间的关联矩阵，计算量和激活值显存占用与序列长度 $n$ 呈平方级关系。当处理数万乃至几十万词元时，这一开销会迅速消耗 GPU 显存。关于自注意力显存与算力占用的具体推导，可参考 [/fundamentals/transformer-arch/transformer-complexity.md](/fundamentals/transformer-arch/transformer-complexity.md)。
2. **缺失位置先验，必须显式注入位置编码**：
   RNN 天然具备前后时序推进的偏置，CNN 具备局域平移不变性。而纯自注意力对序列顺序是完全钝感的（置换不变性），把输入打乱成词袋后矩阵计算结果完全对称。为了让模型知道“小明”在“堵塞”之前，必须显式引入位置编码，详细原理见 [/fundamentals/transformer-arch/positional-encoding.md](/fundamentals/transformer-arch/positional-encoding.md)。
3. **架构适配的分化**：
   针对双向理解任务（如掩码语言建模）或自回归单向生成任务（如 GPT 式补全），自注意力分别演化出了只编码器、只解码器及编码器-解码器结构，具体对比可参考 [/fundamentals/transformer-arch/encoder-decoder-variants.md](/fundamentals/transformer-arch/encoder-decoder-variants.md) 与 [/fundamentals/transformer-arch/bert-vs-gpt.md](/fundamentals/transformer-arch/bert-vs-gpt.md)。
