---
title: RNN 为什么会梯度消失？LSTM 是如何缓解的？
icon: mdi:trending-down
tag:
- LSTM
- RNN
- 梯度消失
description: 对比 RNN 与 LSTM 的梯度反向传播推导，解释细胞状态加法直通路径与遗忘门偏置初始化的作用及边界。
---

在序列建模的面试中，“循环神经网络（RNN）为什么会梯度消失？LSTM 又是如何解决的？”是一道考察反向传播推导与模型架构理解的经典高频题。很多回答容易停留在“遗忘门控制遗忘、加法比乘法好”的感性表述上，未能展示出清晰的数学推导与工程边界。

本篇先给出一个在面试中能够口述的简洁答案，再结合具体的长时依赖例子推导 RNN 梯度的指数级衰减过程，剖析长短期记忆网络（LSTM）的加法更新通路，最后分析 LSTM 为什么仍无法应对现代大语言模型数千 Token 的长上下文场景。

![RNN与LSTM梯度传播机制对比封面图](/images/content-rebuild/f40bcc9538cbebc1/cover.png)

---

## 贯穿例子与面试速答

### 贯穿例子：长跨度信息回传
设想一个语言模型任务，输入序列如下：
> “小明出生在**法国**，小时候学会了讲法语，后来游历了许多国家……三十年后他最熟练的母语依然是 [待预测词]。”

在句尾时间步 $T$，模型预测出“法语”需要依赖开头的“法国”。如果预测错误，损失函数 $L_T$ 产生的梯度信号需要从时间步 $T$ 一路反向传回时间步 $1$，用来更新隐藏状态转移矩阵。若梯度在回传途中衰减为 0，模型参数就无法感知开头的词汇信息。

### 面试口述答案（建议 45 秒内表达完成）
> “标准 RNN 发生梯度消失的根源，在于误差沿时间步反向传播（BPTT）时需要连续相乘状态转移的雅可比矩阵。该矩阵由权重矩阵 $W_{hh}$ 与激活函数导数 $\tanh'$ 组合而成。由于 $\tanh$ 的导数最大只有 1，且只要权重矩阵的模长较小，梯度范数就会随时间差呈指数级衰减，使早期输入无法得到有效更新。
>
> LSTM 缓解该问题的核心在于引入了独立的**细胞状态（Cell State）加法更新机制**：$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$。当损失对历史状态 $c_{t-1}$ 求导时，直接项为遗忘门 $\operatorname{diag}(f_t)$，梯度传递不再强制连续穿透权重矩阵与压缩型激活函数。在工程上将遗忘门偏置初始化为正数（如 1.0）后，训练初期遗忘门接近 1，误差信号能够沿着加法通道近乎线性地传回早期时间步，从而有效缓解了梯度消失。”

---

## RNN 梯度消失的数学推导

要讲透 LSTM 的改进，需要先写出标准 RNN 梯度反向传播的数学过程。

![RNN展开网络与BPTT反向传播连乘路径](/images/content-rebuild/f40bcc9538cbebc1/fig-1.png)

### 1. RNN 前向传播
设在 $t$ 时刻，输入向量为 $x_t$，隐藏状态为 $h_t$，输出为 $y_t$。计算公式为：

$$
a_t = W_{hh} h_{t-1} + W_{xh} x_t + b_h
$$
$$h_t = \tanh(a_t)$$
$$
\hat{y}_t = \operatorname{softmax}(W_{hy} h_t + b_y)
$$

总损失为各时间步损失之和：$L = \sum_{t=1}^T L_t$。

### 2. 误差反向传播中的链式相乘
假设只考察时间步 $T$ 的损失 $L_T$ 对早期隐藏状态 $h_k$（$k < T$）的影响。根据链式法则（约定梯度为行向量）：

$$
\frac{\partial L_T}{\partial h_k} = \frac{\partial L_T}{\partial h_T} \prod_{j=k+1}^T \frac{\partial h_j}{\partial h_{j-1}}
$$

关注连乘项中的单步转移雅可比矩阵 $\frac{\partial h_j}{\partial h_{j-1}}$。由 $h_j = \tanh(W_{hh} h_{j-1} + W_{xh} x_j + b_h)$ 可得：

$$
\frac{\partial h_j}{\partial h_{j-1}} = \operatorname{diag}\left(1 - h_j^2\right) W_{hh}
$$

其中 $\operatorname{diag}\left(1 - h_j^2\right)$ 是 $\tanh$ 导数组成的对角矩阵，对角线上的数值范围落在 $(0, 1]$ 之间。

![激活函数导数收缩与矩阵奇异值对梯度范数的双重影响](/images/content-rebuild/f40bcc9538cbebc1/fig-2.png)

### 3. 梯度消失与爆炸的边界
两时刻相距 $\Delta t = T - k$ 步，跨步梯度为：

$$
\frac{\partial h_T}{\partial h_k} = \prod_{j=k+1}^T \left( \operatorname{diag}\left(1 - h_j^2\right) W_{hh} \right)
$$

利用矩阵范数不等式可以估计该项的范数上界：

$$
\left\| \frac{\partial h_T}{\partial h_k} \right\| \le \prod_{j=k+1}^T \left\| \operatorname{diag}\left(1 - h_j^2\right) \right\| \|W_{hh}\| \le \left( \sigma_{\max}(W_{hh}) \right)^{\Delta t} \prod_{j=k+1}^T \gamma_j
$$

其中 $\sigma_{\max}(W_{hh})$ 为循环权重矩阵的最大奇异值，$\gamma_j \le 1$ 为对角阵中 $\tanh'$ 的最大对角元素：

1. **梯度消失**：若 $\sigma_{\max}(W_{hh}) < 1$，由于 $\gamma_j \le 1$，连乘项范数随时间差 $\Delta t$ 呈指数级衰减。即使 $\sigma_{\max}(W_{hh}) \approx 1$，只要神经元激活值进入饱和区（$\tanh'(x) \to 0$），连乘也会快速趋近于 0。这导致远距离的时间步无法接收到反向回传的梯度，模型无法学习跨时步的长距离依赖。
2. **梯度爆炸**：若 $\sigma_{\max}(W_{hh}) > 1$ 且隐藏状态处于非饱和区，连乘项可能随 $\Delta t$ 剧增，导致数值溢出。在工程实践中，梯度爆炸可以通过[梯度裁剪（Gradient Clipping）](/interview/llm-interview/gradient-clipping.md)强行截断，但梯度消失无法简单通过数值放大来解决。关于数值稳定性背景，可延伸参考[梯度消失与爆炸基础教程](/fundamentals/dl-basics/gradient-vanishing-exploding.md)。

<ReadMoreLock />

---

## LSTM 如何改变梯度的流动路径

长短期记忆网络（LSTM）并没有抛弃循环结构，而是将状态拆分成负责长期传递的**细胞状态（Cell State）**和负责即时暴露的**隐藏状态（Hidden State）**，并通过门控结构控制信息存取。

![LSTM三门一态内部数据拓扑与运算流](/images/content-rebuild/f40bcc9538cbebc1/fig-3.png)

### 1. 门控与状态更新公式
在每个时间步 $t$，LSTM 接收输入向量 $x_t$ 与上一时间步的隐藏状态 $h_{t-1}$：

1. **遗忘门（$f_t$）**：计算保留上一时刻细胞状态的比例。
   $$f_t = \sigma\left(W_f [h_{t-1}, x_t] + b_f\right)$$
2. **输入门（$i_t$）与候选细胞状态（$\tilde{c}_t$）**：决定当前输入写入细胞状态的量。
   $$i_t = \sigma\left(W_i [h_{t-1}, x_t] + b_i\right)$$
   $$\tilde{c}_t = \tanh\left(W_c [h_{t-1}, x_t] + b_c\right)$$
3. **细胞状态更新（$c_t$）**：加法更新核心。
   $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
4. **输出门（$o_t$）与隐藏状态（$h_t$）**：控制向下一层和下一时间步暴露哪些信息。
   $$o_t = \sigma\left(W_o [h_{t-1}, x_t] + b_o\right)$$
   $$h_t = o_t \odot \tanh(c_t)$$

其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 输出在 $(0, 1)$ 区间，$\odot$ 表示逐元素相乘。

---

## 细胞状态求导：加法路径与梯度直通

解释 LSTM 为何能缓解梯度消失，关键在于计算细胞状态转移导数 $\frac{\partial c_t}{\partial c_{t-1}}$，并与标准 RNN 对比。

![LSTM细胞状态加法路径与直通导数机制](/images/content-rebuild/f40bcc9538cbebc1/fig-4.png)

### 1. 细胞状态转移求导展开
因为 $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$，且 $f_t, i_t, \tilde{c}_t$ 依赖于 $h_{t-1}$（$h_{t-1}$ 又依赖于 $c_{t-1}$），对 $c_{t-1}$ 求全导数：

$$
\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t) + \underbrace{\operatorname{diag}(c_{t-1}) \frac{\partial f_t}{\partial c_{t-1}} + \operatorname{diag}(\tilde{c}_t) \frac{\partial i_t}{\partial c_{t-1}} + \operatorname{diag}(i_t) \frac{\partial \tilde{c}_t}{\partial c_{t-1}}}_{\text{经由 } h_{t-1} \text{ 的间接梯度项}}
$$

其中第二部分间接项经过了与 $h_{t-1}$ 相关的权重矩阵与激活函数；而第一项 $\operatorname{diag}(f_t)$ 则是加法操作带来的**直接直通通道**。

### 2. 跨时步连乘对比
当梯度从 $T$ 回传到 $k$ 时：

$$
\frac{\partial c_T}{\partial c_k} = \prod_{j=k+1}^T \left( \operatorname{diag}(f_j) + \text{间接项}_j \right)
$$

在主干通道上，梯度近似为各时间步遗忘门输出的连乘：

$$
\frac{\partial c_T}{\partial c_k} \approx \prod_{j=k+1}^T \operatorname{diag}(f_j)
$$

将两者结构进行直接对比：

| 维度 | 标准 RNN（状态 $h$） | LSTM（细胞状态 $c$） |
| :--- | :--- | :--- |
| **单步转移导数主项** | $\operatorname{diag}(1 - h_j^2) W_{hh}$ | $\operatorname{diag}(f_j)$ |
| **是否必经权重矩阵相乘** | 是，每一步都必须乘 $W_{hh}$ | 否，主干更新通道无权重矩阵连乘 |
| **是否必经压缩激活函数** | 是，每一步必乘 $\tanh' \le 1$ | 否，主干更新通道为线性加法 |
| **数值可调节性** | 难以随输入动态维持模长为 1 | 门控值 $f_j$ 可由模型根据输入动态调节 |

当模型判定某段历史信息需要长期保留时，可通过参数更新让遗忘门 $f_j \approx 1$。此时 $\prod_{j=k+1}^T \operatorname{diag}(f_j) \approx I$，误差信号能够沿加法通道平稳传回早期时间步。这与后续现代 Transformer 架构中的[残差连接（Residual Connection）](/interview/llm-interview/transformer-residual.md)在设计思路上有相通之处。

![遗忘门偏置初始化为正数的物理效果对比](/images/content-rebuild/f40bcc9538cbebc1/fig-5.png)

### 3. 工程细节：遗忘门偏置为什么要初始化为正数？
在模型权重随机初始化阶段（通常在 0 附近）：
- 若遗忘门偏置 $b_f$ 设为 0，则经过 Sigmoid 激活后 $f_t = \sigma(0) = 0.5$；
- 连续反向传播 30 步后，主项衰减为 $0.5^{30} \approx 9.3 \times 10^{-10}$，加法通道的优势在训练初期就会丧失。

因此，在经典文献（如 Jozefowicz et al., 2015）与工程实践中，**通常推荐将遗忘门偏置 $b_f$ 初始化为正数（例如 1.0 或 2.0）**（例如 TensorFlow 的 LSTM 默认开启 `unit_forget_bias=True`；而在 PyTorch 中 `nn.LSTM` 默认偏置来自均匀分布，通常需开发者手动显式赋值）：

$$
f_t^{(init)} = \sigma(1.0) \approx 0.731, \quad \sigma(2.0) \approx 0.881
$$

这使模型在训练伊始默认采取“先不遗忘”的策略，保持反向传播通道畅通，待参数更新稳定后再根据任务自适应学习何时主动遗忘。

---

## 数值连乘算例示意

以下脚本使用固定的经验系数展示两类连乘在步数增加时的衰减差异（仅用于教学对比，非真实模型运行日志）：

```python
# 梯度衰减示意代码（用于直观理解衰减幅度，非真实模型训练）
def simulate_gradient_flow(seq_len=50):
    # 假设 RNN 单步组合衰减因子为 0.54 (如 tanh 导数 0.6 * 权重影响 0.9)
    rnn_factor = 0.54
    rnn_grad = [1.0]
    for _ in range(seq_len):
        rnn_grad.append(rnn_grad[-1] * rnn_factor)

    # 假设 LSTM 遗忘门稳定保持在 0.98
    lstm_factor = 0.98
    lstm_grad = [1.0]
    for _ in range(seq_len):
        lstm_grad.append(lstm_grad[-1] * lstm_factor)

    return rnn_grad, lstm_grad

rnn_g, lstm_g = simulate_gradient_flow(50)
print(f"第 10 步梯度比例 - RNN: {rnn_g[10]:.2e}, LSTM: {lstm_g[10]:.2e}")
print(f"第 50 步梯度比例 - RNN: {rnn_g[50]:.2e}, LSTM: {lstm_g[50]:.2e}")
```

**输出说明**：
- 在第 10 步时，RNN 梯度衰减为 $0.54^{10} \approx 2.11 \times 10^{-3}$，而 LSTM 梯度仍保持在 $0.98^{10} \approx 0.817$；
- 在第 50 步时，RNN 梯度已下降到 $4.02 \times 10^{-14}$，在实际计算中已无法驱动权重有效更新；而 LSTM 仍保留在 $0.98^{50} \approx 0.364$。

该算例直观展示了，在遗忘门保持高激活的理想通路下，加法直通机制能够显著减缓连乘链条导致的数值萎缩。

---

## 面试深度追问：LSTM 为什么依然无法胜任现代大模型？

面试官常会顺着这个问题追问：**“既然 LSTM 缓解了梯度消失，能处理更长的序列，为什么在大语言模型时代依然被 Transformer 取代了？”**

回答需要从**梯度边界**、**计算吞吐**与**信息容量**三个层面展开：

![LSTM时序串行瓶颈与Transformer全局并行机制对比](/images/content-rebuild/f40bcc9538cbebc1/fig-6.png)

### 追问 1：LSTM 彻底消除了梯度消失吗？
**回答要点**：没有。
- 首先，如果任务要求网络丢弃部分无用历史信息，遗忘门输出就会小于 1（$f_t < 1$），长期连乘依然会指数级衰减；
- 其次，由隐藏状态 $h_t$ 传递到当前输入 $x_t$ 的梯度路径依然包含激活函数和权重变换。经验上，LSTM 的有效记忆跨度一般在数百步以内，面对数千到数万 Token 的超长文本，早期信号仍难以保持。

### 追问 2：为什么 LSTM 无法充分利用 GPU 硬件？
- **LSTM 具备严格的时序依赖**：第 $t$ 步的输入必须依赖第 $t-1$ 步计算完成的隐藏状态 $h_{t-1}$ 和细胞状态 $c_{t-1}$。这导致时序前向和反向传播的计算复杂度在序列长度维度上是严格串行的 $O(T)$，无法做跨时步的并行计算。
- **Transformer 依靠矩阵乘法并行**：Transformer 抛弃了循环连接，使用[自注意力机制（Self-Attention）](/interview/llm-interview/attention-qkv.md)一次性对整个序列的所有位置执行矩阵运算，充分利用了 GPU 的并行张量计算核心，极大提升了训练吞吐。

### 追问 3：固定维度的向量信息瓶颈
- **LSTM 的信息压缩限制**：无论上下文长度是 100 步还是 10,000 步，历史信息最终都要被压缩进固定维度的向量（$c_t, h_t$）中。随着[上下文窗口（Context Window）](/interview/llm-interview/context-window.md)扩大，单个向量的表示能力必然达到上限，产生信息挤压和细节丢失。
- **Transformer 的动态寻址**：自注意力机制下任意两个 Token 之间的交互距离都是 $O(1)$，模型无需把全局信息压成单向量，而是根据 Query 与 Key 的点乘权重按需直接寻址任意历史位置。

---

## 常见答题误区

1. **误区一：把加法通道说成“LSTM 的梯度恒等于 1”**
   - *分析*：遗忘门 $f_t$ 是在 $(0, 1)$ 间动态变化的向量，并非恒等于 1。此外求导中还包含经由 $h_{t-1}$ 的间接梯度项。只有当模型学得 $f_t \approx 1$ 时，主干通道才近似无衰减传播。
2. **误区二：认为 RNN 梯度消失完全是因为 $\tanh$ 激活函数**
   - *分析*：即使使用线性激活函数（导数恒为 1），只要权重矩阵 $W_{hh}$ 的最大奇异值小于 1，连乘后梯度范数依然会指数级衰减。梯度消失是矩阵连乘与激活函数收缩共同作用的结果。
3. **误区三：断言 LSTM“绝对无法记忆超过 100 步”**
   - *分析*：不要给出绝对阈值。记忆跨度取决于具体任务形式、数据分布、门控取值及网络规模。在简单周期或规则序列上，LSTM 可以保持较长记忆；但在复杂的自然语言建模中，受限于信息覆盖和梯度衰减，其有效跨度通常在数百步量级。

---

## 关联阅读与下一阶段
- 理解残差连接如何跨越网络深度传递梯度：[Transformer 中的“残差连接”可以缓解梯度消失问题吗？](/interview/llm-interview/transformer-residual.md)
- 学习在工程层面应对数值溢出的具体实现：[大模型训练中的梯度裁剪（Gradient Clipping）](/interview/llm-interview/gradient-clipping.md)
- 了解自注意力机制的 $O(1)$ 交互设计：[Self-Attention 中 Q、K、V 分别是什么意思？](/interview/llm-interview/attention-qkv.md)
