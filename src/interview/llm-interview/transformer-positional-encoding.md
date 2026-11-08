---
title: Transformer 的位置编码是怎样的？原理与推导
icon: mdi:map-marker
tag:
- Transformer
- 位置编码
- 正弦编码
description: 详解原始 Transformer 正弦绝对位置编码的设计动机、公式含义、相对位置线性变换推导与相加机制，分析其长文本外推局限及演进路线。
---

![正余弦位置编码与相对变换矩阵结构图](/images/content-rebuild/5857b4f2a26b9dab/cover.png)

在大模型与深度学习面试中，Transformer 的位置编码属于必考的高频基础题。面试官通常不仅考察“输入向量加上位置向量”这一基本事实，还会深入追问：为什么自注意力本身无法感知位置？为什么选正余弦函数？相对位置关系是如何内隐推导出来的？直接相加会不会破坏词向量本身的语义？

本文承接 [Transformer 和 LLM 有哪些区别](/interview/llm-interview/transformer-vs-llm.md) 中关于基础架构的讨论，结合 Vaswani 等人在 2017 年论文 *Attention Is All You Need* 中的设计，对正弦绝对位置编码（Sinusoidal Positional Encoding）进行可口述、可推导、可验证的完整拆解。

---

## 30秒口述回答

在面试中被问到“Transformer 的位置编码原理是怎样的”，可以分四个层次作答：

> 1. **设计动机**：自注意力层的点积计算仅依赖两个词的特征向量，数学上对输入顺序具有**置换等变性**（Permutation Equivariance）。若不注入位置编码，模型无法分辨“小明打小红”与“小红打小明”的语序区别。
> 2. **编码公式**：原版 Transformer 采用固定、无参数的正余弦函数。偶数维度使用正弦函数，奇数维度使用余弦函数，角频率沿特征维度按几何级数衰减，波长覆盖范围从 $2\pi$ 延伸到 $10000 \cdot 2\pi$。
> 3. **核心数学性质**：依据正余弦的和角公式，对于任意固定的相对距离 $k$，位置 $pos + k$ 的编码向量可以通过位置 $pos$ 的编码向量乘以一个只与 $k$ 相关的正交块对角旋转矩阵得到（$PE_{pos+k} = M_k PE_{pos}$）。这使绝对位置输入在理论上具备表达相对位置关系的能力。
> 4. **工程实现与演进**：词嵌入向量在乘上缩放因子 $\sqrt{d_{\text{model}}}$ 后与位置编码直接逐元素相加。虽然该公式能对任意长度计算数值，但在超出训练长度时，未见过的低频相位和绝对位置交叉项会导致外推性能急剧下降，因此现代主流大模型已基本转向旋转位置编码（RoPE）或相对位置偏置。

---

## 一、为什么自注意力无法识别语序？

![自注意力置换不变性与位置注入机制](/images/content-rebuild/5857b4f2a26b9dab/fig-1.png)

循环神经网络（RNN）依赖逐步迭代的隐藏状态 $h_t = f(h_{t-1}, x_t)$ 推进，天然具备时间先后顺序。而 Transformer 的自注意力机制采用全序列并行计算，其核心计算步骤如下：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

关于注意力基础计算的详细过程，可参考 [Self-Attention 中 Q、K、V 分别是什么意思？](/interview/llm-interview/attention-qkv.md)。

假设施加一个置换矩阵 $P \in \mathbb{R}^{L \times L}$ 打乱输入行序列（$X' = PX$），由于线性投影矩阵 $W_Q, W_K, W_V$ 作用在各个 Token 向量上，得到的投影矩阵满足 $Q' = PQ, K' = PK, V' = PV$。代入自注意力公式：

$$\text{Attention}(PQ, PK, PV) = \text{softmax}\left(\frac{PQ(PK)^T}{\sqrt{d_k}}\right)PV$$

因为置换矩阵是正交矩阵，满足 $P^T P = I$，展开中间项可得：

$$PQ(PK)^T = PQ K^T P^T$$

对矩阵行进行 Softmax 操作时，由于 Softmax 逐行独立计算且归一化分母求和满足加法交换律，左乘 $P$ 保持行重排、右乘 $P^T$ 保持列重排，置换矩阵可移至外侧：

$$\text{softmax}\left(\frac{PQ K^T P^T}{\sqrt{d_k}}\right) = P \cdot \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) P^T$$

再乘以 $PV$，利用正交性 $P^T P = I$ 相抵消：

$$P \cdot \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) P^T P V = P \cdot \text{Attention}(Q, K, V)$$

**结论**：自注意力计算对序列行变换是**严格置换等变的**（Permutation Equivariant）。对于任意两个词 $x_i$ 和 $x_j$，它们之间的注意力权重只取决于两个向量的内积，与它们各自在句中位于第几个位置、彼此相隔多远完全无关。如果不显式引入位置信息，模型感知到的输入只是一组无序的词袋（Bag of Words）。

---

## 二、正弦绝对位置编码的公式与几何直觉

在 *Attention Is All You Need* 中，Vaswani 等人提出了一种不需要学习参数的固定位置编码方案。设 Token 在序列中的绝对索引为 $pos \in [0, L-1]$，向量维度为 $d_{\text{model}}$，通道索引为 $i \in [0, \frac{d_{\text{model}}}{2}-1]$。

位置向量 $PE_{pos} \in \mathbb{R}^{d_{\text{model}}}$ 的各个分量定义如下：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$

设通道角频率 $\omega_i = \frac{1}{10000^{2i / d_{\text{model}}}}$，公式可紧凑写为：

$$PE_{(pos, 2i)} = \sin(\omega_i \cdot pos)$$

$$PE_{(pos, 2i+1)} = \cos(\omega_i \cdot pos)$$

![正弦位置编码多频率波长分布示意图](/images/content-rebuild/5857b4f2a26b9dab/fig-2.png)

### 频率与波长的分布直觉

1. **高频分量（低通道编号，如 $i=0$）**：
   角频率 $\omega_0 = \frac{1}{10000^0} = 1$，对应的正弦周期为 $T_0 = \frac{2\pi}{\omega_0} = 2\pi \approx 6.28$。波长仅跨越约 6 个 Token，在相邻词之间数值剧烈变化，负责提供**局部细粒度的区分能力**。
2. **低频分量（高通道编号，如 $2i \to d_{\text{model}}-2$）**：
   角频率 $\omega_{d/2-1} = \frac{1}{10000}$，对应的周期为 $T = 20000\pi \approx 62831$。波长极其宽广，在常规上下文长度内呈现平缓单调的变化，负责提供**全局长距离定位能力**。
3. **唯一性保证**：
   该设计类似于机械式时钟的“秒针、分针、时针”结构，或二进制计数器的各位翻转。各维度以几何级数连续衰减的角频率构成了一组唯一标识序列位置的连续指纹向量，在数万个步长内不会出现完全相同的编码。

---

<ReadMoreLock />

## 三、核心数学推导：相对位置的线性表示

原论文选择正弦和余弦函数的核心动机是：**对于任意固定的相对位移 $k$，位置 $pos+k$ 的编码向量可以表示为位置 $pos$ 的线性函数。**

这是面试现场白板推导的高频考点，完整推导过程如下：

![正弦位置编码二维旋转矩阵变换推导](/images/content-rebuild/5857b4f2a26b9dab/fig-3.png)

针对某一个通道角频率 $\omega_i$，取出其对应的两个相邻奇偶分量构成二维子向量：

$$v_{pos}^{(i)} = \begin{bmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{bmatrix} = \begin{bmatrix} \sin(\omega_i \cdot pos) \\ \cos(\omega_i \cdot pos) \end{bmatrix}$$

当位置偏移 $k$ 个步长后，新位置的二维子向量为：

$$v_{pos+k}^{(i)} = \begin{bmatrix} \sin(\omega_i(pos + k)) \\ \cos(\omega_i(pos + k)) \end{bmatrix}$$

利用三角函数的和角公式展开：

$$\sin(\omega_i pos + \omega_i k) = \sin(\omega_i pos)\cos(\omega_i k) + \cos(\omega_i pos)\sin(\omega_i k)$$

$$\cos(\omega_i pos + \omega_i k) = \cos(\omega_i pos)\cos(\omega_i k) - \sin(\omega_i pos)\sin(\omega_i k)$$

将其整理为矩阵与向量相乘的形式：

$$\begin{bmatrix} \sin(\omega_i(pos + k)) \\ \cos(\omega_i(pos + k)) \end{bmatrix} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix} \begin{bmatrix} \sin(\omega_i pos) \\ \cos(\omega_i pos) \end{bmatrix}$$

定义二维旋转变换矩阵 $M_k^{(i)}$：

$$M_k^{(i)} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix}$$

将全部分割的 $d_{\text{model}} / 2$ 个二维子空间拼接起来，即可构造一个大小为 $d_{\text{model}} \times d_{\text{model}}$ 的块对角矩阵（Block Diagonal Matrix）$M_k$：

$$M_k = \begin{bmatrix} M_k^{(0)} & 0 & \dots & 0 \\ 0 & M_k^{(1)} & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & M_k^{(d/2 - 1)} \end{bmatrix}$$

因此，在整个空间中恒成立：

$$PE_{pos+k} = M_k \cdot PE_{pos}$$

**推导结论**：变换矩阵 $M_k$ 只取决于相对步长 $k$ 和各维度角频率 $\omega_i$，**与当前 Token 所在的具体绝对位置 $pos$ 完全无关**。这意味着模型只需通过线性层，理论上就能将任意绝对坐标投影成特定相对偏移后的向量。

### Python 数值验证

通过以下简明脚本可验证上述数学关系的严格成立：

```python
import numpy as np

def get_sinusoidal_pe(pos: int, d_model: int) -> np.ndarray:
    pe = np.zeros(d_model)
    for i in range(d_model // 2):
        omega = 1.0 / (10000.0 ** (2 * i / d_model))
        pe[2 * i] = np.sin(pos * omega)
        pe[2 * i + 1] = np.cos(pos * omega)
    return pe

def get_transition_matrix(k: int, d_model: int) -> np.ndarray:
    m = np.zeros((d_model, d_model))
    for i in range(d_model // 2):
        omega = 1.0 / (10000.0 ** (2 * i / d_model))
        cos_k = np.cos(omega * k)
        sin_k = np.sin(omega * k)
        # 对应子空间矩阵: [[cos, sin], [-sin, cos]]
        m[2 * i, 2 * i] = cos_k
        m[2 * i, 2 * i + 1] = sin_k
        m[2 * i + 1, 2 * i] = -sin_k
        m[2 * i + 1, 2 * i + 1] = cos_k
    return m

# 验证参数
d_model = 64
pos = 17
k = 5

pe_pos = get_sinusoidal_pe(pos, d_model)
pe_pos_plus_k = get_sinusoidal_pe(pos + k, d_model)
m_k = get_transition_matrix(k, d_model)

# 线性变换预测
predicted = m_k @ pe_pos

# 计算最大绝对误差
max_error = np.max(np.abs(pe_pos_plus_k - predicted))
print(f"最大绝对误差: {max_error:.2e}")  # 输出在 1e-16 数量级，属于浮点精度截断
```

---

## 四、面试高频追问：为什么直接相加？会冲淡语义吗？

面试官常问：**“为什么位置向量和词向量是直接相加（Add），而不是拼接（Concat）？直接相加会不会把词语原本的语义信息搞乱？”**

![嵌入相加与注意力内积四项展开图解](/images/content-rebuild/5857b4f2a26b9dab/fig-4.png)

### 追问 1：为什么不采用拼接（Concat）？
拼接操作形如 $[x_{pos} \,;\, PE_{pos}] \in \mathbb{R}^{d_x + d_{pe}}$。虽然几何上二者完全独立，但存在以下代价：
1. **参数与显存开销**：若拼接，输入维度变为 $d_x + d_{pe}$，导致第一层 Self-Attention 的投影矩阵参数量和单步矩阵乘法算力开销成倍上升；
2. **模块一致性**：Transformer 全网络统一设定隐藏层维度 $d_{\text{model}} = 512$，相加方式使残差连接与多层级联的接口完全对齐，无需引入多余的投影降维层。

### 追问 2：直接相加为什么没有破坏词义？
1. **模长尺度放大（Scale Factor）**：
   在原始论文代码实现中，词嵌入矩阵在加上位置编码之前，会先乘上缩放系数 $\sqrt{d_{\text{model}}}$（在 $d=512$ 时约为 $22.6$）。因为词嵌入层初始化参数方差较小，该缩放使得词语义向量的整体模长明显大于位置编码向量（$PE$ 各维度分量取值在 $[-1, 1]$ 之间），保证了初层语义特征占据主导地位。
2. **注意力点积四项解耦**：
   设输入向量经相加后为 $z_i = x_i + p_i$。在 Self-Attention 计算点积得分时展开：

   $$q_i k_j^T = (x_i W_Q + p_i W_Q)(x_j W_K + p_j W_K)^T$$

   $$= \underbrace{x_i W_Q W_K^T x_j^T}_{\text{词语义 - 词语义}} + \underbrace{x_i W_Q W_K^T p_j^T}_{\text{词语义 - 目标位置}} + \underbrace{p_i W_Q W_K^T x_j^T}_{\text{源位置 - 目标词语义}} + \underbrace{p_i W_Q W_K^T p_j^T}_{\text{纯位置偏置}}$$

   这四项在网络中承担了不同的分工：第一项衡量内容相关性，第二和第三项衡量特定词对上下文特定位置的依赖，第四项仅衡量两者的相对/绝对位置偏置。网络在反向传播中可以通过优化 $W_Q, W_K$，在同一个空间内调和语义特征与距离特征。

---

## 五、工程局限：为什么正弦编码难以长文本外推？

开发者初读论文常产生误解：“既然正弦公式对任意 $pos$ 都能算出数值，那是不是意味着它可以无缝外推到无限长的上下文？”

答案是否定的。**数学公式的连续可计算，不等于神经网络具备泛化能力。**

![正弦位置编码外推失效机制对比图](/images/content-rebuild/5857b4f2a26b9dab/fig-5.png)

当测试序列长度超过训练序列长度（$L_{\text{test}} > L_{\text{train}}$）时，正弦绝对位置编码表现迅速退化：

1. **低频维度的未见相位（Unseen Phase）**：
   高频通道周期短，在较短的训练长度内已经历了完整的正负循环；而对于高通道索引的极低频分量，其波长长达数万步。如果在训练期上下文仅为 2048，低频维度的值仅局限在很窄的相位区间内（如 $[0, 0.2]$）。一旦推理步长达到 8192，该维度将出现训练期从未见过的数值，导致高层前馈网络激活异常。
2. **绝对位置交叉污染**：
   注意力展开式中的交叉项（如 $x_i W_Q W_K^T p_j^T$）将绝对坐标 $p_j$ 与内容向量强耦合。神经网络学会的是“在输入长为 500 时，第 490 步词向量的注意力加权规则”，无法将这种模式无损平移到“在输入长为 5000 时，第 4990 步词向量”上。

![位置编码技术选型演进路线图](/images/content-rebuild/5857b4f2a26b9dab/fig-6.png)

正是由于绝对位置编码的上述局限，促成了后续位置编码方案的演进：

* **可学习绝对位置编码（Learned Absolute PE）**：如 BERT、GPT-2，采用查找表（Look-up Table）。每个位置一个可训练参数，最大长度被字典大小死死定死，完全无法处理超长序列。
* **相对位置偏置（Relative Position Bias）**：如 T5、ALiBi，放弃在输入端叠加位置向量，直接在自注意力分数矩阵上加上与距离 $|i - j|$ 相关的偏置项，具备更好的外推性。
* **旋转位置编码（RoPE）**：如 LLaMA、ChatGLM 等。RoPE 继承了正交旋转矩阵的思想，但将旋转操作直接施加在 $Q_i$ 与 $K_j$ 上，使注意力内积严格仅取决于相对偏移 $j - i$，彻底消除了输入端的绝对位置交叉污染。深入推导可参考 [位置编码是什么？RoPE 为什么很常用？](/interview/llm-interview/positional-encoding.md) 与 [了解 LLaMA 中的旋转位置编码吗？](/interview/llm-interview/llama-rope.md)；对底层原理感兴趣的读者可进一步阅读 [Transformer 为什么需要位置编码？](/fundamentals/transformer-arch/positional-encoding.md)。

---

## 面试避坑与高频误区总结

| 常见错误说法 | 事实与正确辨析 |
| :--- | :--- |
| “没有位置编码时，Self-Attention 具有置换不变性，输出完全相同。” | 严格概念是**置换等变性**（$Attention(PX) = P \cdot Attention(X)$）。打乱词序后，输出向量只是按相同规则调换了位置，词与词之间的点积权重只取决于内容本身，缺失前后次序感知。 |
| “因为 $PE_{pos+k} = M_k PE_{pos}$，所以整个多层 Transformer 天然掌握相对位置。” | 该线性关系仅在未经非线性的单个向量层面成立。多层 Transformer 包含 Softmax 归一化和 MLP 的非线性激活，深层特征并不能保持这种刚性欧氏旋转关系。 |
| “正余弦编码支持任意数值输入，因此模型天生长文本外推好。” | 公式可计算不代表网络能泛化。高维低频分量在长序列下出现“未见相位”，绝对位置交叉项导致深层注意力发散。 |
| “RoPE 是一种全新的机制，与正弦绝对位置编码没有联系。” | RoPE 正是由正余弦编码演变而来，其核心同样是利用复数旋转/块对角正交阵满足和角公式，只是将其从“输入端相加”移至“$Q, K$ 内部做旋转变换”。 |
