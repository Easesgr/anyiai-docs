---
title: 大模型推理是什么？输入提示后如何逐词生成？
icon: mdi:lightning-bolt-outline
tag:
- 推理与生成
- 推理流程
- Prompt
description: 大模型推理不是单次生成整篇文本，而是经历提示词并行计算与逐词串行解码两个阶段。本文通过极简算例拆解自回归推理的完整过程与硬件瓶颈。
---

大语言模型（LLM）在聊天界面中吐出回答时，看起来像是在一段一段地实时打字。但在硬件内部，模型既不是一开始就构思好了完整段落一次性返回，也不是随机抓取词句。它经历了一个严格的数学过程：**先一次性读完你给的提示词，然后进入一个“每算一步只吐一个词、再把这个词拼回输入端”的循环**。

这种计算方式在计算机科学中被称为**自回归（Autoregressive）推理**。

我们先建立一个贯穿全程的具体例子：假设词表中只有 5 个词，用户输入提示词 `"大 模"`，模型最终生成回复 `"型 好"` 并停止。模型到底经历了几次计算？矩阵的形状怎么变？为什么会越算越慢？本文为你完整还原整个执行链路。

![大模型自回归推理全景图](/images/content-rebuild/55d5c64d70627310/cover.png)

---

## 文本如何变成模型能算的向量？

计算机无法直接对中文字符做矩阵乘法。当用户输入提示词 `"大 模"` 时，第一步必须将字符转化成计算机能处理的连续数值向量。这一过程分为分词、查表与注入位置三步：

### 1. 分词（Tokenization）
分词器（Tokenizer）按照预先建好的词表，将连续的字符串切分成由 Token ID（离散整数）组成的序列。每个 Token 可以是一个字、一个词或子词片段：

$$\text{Prompt: } "\text{大 模}" \xrightarrow{\text{Tokenizer}} [\text{Token}_1, \text{Token}_2] = [0, 1]$$

这里的整数 $0$ 和 $1$ 是词表中的行索引。现代主流开源模型的词表大小 $V$ 通常在 32,000 到 150,000 之间。

### 2. 词嵌入查表（Embedding Lookup）
模型内部有一张静态权重矩阵，称为词嵌入矩阵 $W_e \in \mathbb{R}^{V \times d_{\text{model}}}$，其中 $d_{\text{model}}$ 是模型的隐藏层维度（例如 4096）。每个 Token ID 对应矩阵中的一行。对于输入 $[0, 1]$，模型直接从矩阵中取出第 0 行和第 1 行，得到一个形状为 $(2, d_{\text{model}})$ 的实数矩阵。

### 3. 注入位置编码（Positional Encoding）
单纯的向量查表无法反映先后顺序——无论“大模”还是“模大”，查出来的两行向量都完全一样。为了让模型感知语序，系统会在词嵌入向量上叠加或融合位置编码（如旋转位置编码 RoPE），将每个词所处的绝对或相对物理位置刻进数值里。

![提示词文本转向量流水线](/images/content-rebuild/55d5c64d70627310/fig-1.png)

经过这三步，离散的文本就转化成了连续的张量 $X_0 \in \mathbb{R}^{L \times d_{\text{model}}}$（其中 $L$ 为输入长度，此处 $L=2$），准备送入 Transformer 的多层网络堆栈。

<ReadMoreLock />

---

## Prefill 阶段：为什么提示词可以一次性并行算完？

当输入向量准备好后，模型进入第一个计算阶段：**预填充阶段（Prefill 阶段，也称上下文编码阶段）**。

在这个阶段，输入的提示词（示例中的 `"大 模"`，共 2 个 Token）是全部已知的。既然所有词都已经摆在显存里，GPU 就没有必要一个词一个词地慢速等待，而是利用强大的并行算力，把这 2 个 Token 打包成矩阵乘法一次性处理完毕。

在 Transformer 的自注意力计算中：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

虽然因果掩码矩阵 $M$（Causal Mask）强制将上三角位置设为 $-\infty$（经过 Softmax 映射后未来位置的注意力权重归零，防止前方词偷看后方词），但这并不破坏前向计算的并行性，硬件依然在同一时刻并行完成整个提示词序列的矩阵乘法与注意力投影。

![Prefill 阶段计算结构](/images/content-rebuild/55d5c64d70627310/fig-2.png)

### 关键动作：只取末位向量
在 Transformer 最后一层计算完成后，模型输出了提示词中每个位置的隐藏向量 $H = [h_1, h_2] \in \mathbb{R}^{2 \times d_{\text{model}}}$。

此时，为了生成回复的第一个词，系统**只截取最后一个位置的隐藏状态 $h_2$**。原因在于因果掩码的设计：$h_1$ 只看到了第 1 个字“大”，而 $h_2$ 已经同时看过了第 1 个字“大”和第 2 个字“模”。$h_2$ 代表了“在读完整个输入提示后，上下文汇总形成的语境向量”。

这个末位向量 $h_2$ 会被立即送往输出层去预测下一个词，同时各层计算出来的 Key 和 Value 向量会被存入显存，避免后续重复计算。显存复用细节可参阅 [KV Cache 是什么？](/fundamentals/inference-and-generation/kv-cache.md)。

在 Prefill 阶段，GPU 的计算核心（Tensor Core）被大规模矩阵乘法填满，属于典型的**计算密集型（Compute-bound）**任务。关于计算与内存带宽的硬件瓶颈，可参阅 [大模型推理为什么慢？瓶颈在哪里？](/llm-100/deployment/inference-bottlenecks.md)。

---

## Decode 阶段：为什么生成回答只能逐词串行？

当 Prefill 阶段利用末位向量 $h_2$ 算出第一个新词 `"型"`（Token ID 为 2）之后，推理进入第二阶段：**解码阶段（Decode 阶段）**。

自回归模型的核心特性在这里体现：模型在概率上遵循链式分解：

$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

要生成第 $t$ 个词，必须以已有的前 $t-1$ 个词作为条件输入。因为未来的词在物理上还未被选出，模型**无法预先跨步计算未来位置**，必须按部就班地串行循环：

1. **第 1 轮循环**：将刚生成的 Token `2`（"型"）送入模型（结合之前缓存的历史信息），模型输出当前步的隐藏向量，据此预测出下一个 Token `3`（"好"）；
2. **第 2 轮循环**：将新选出的 Token `3` 拼入已有序列末尾，驱动下一轮前向计算，预测出结束标记 Token `4`（`<eos>`）；
3. **循环退出**：检测到结束标记，整趟生成任务完成。

![Decode 阶段自回归迭代流程](/images/content-rebuild/55d5c64d70627310/fig-3.png)

在单序列或低并发的 Decode 阶段，GPU 每次前向计算通常只为每个序列处理长度为 1 的单个新 Token。因为单步参与矩阵乘法的有效 Token 极少，GPU 庞大的计算单元很难被填满，反而是每生成一个 Token，芯片都需要将整整几十吉字节（GB）的模型权重从显存完整搬进计算核心一次。因此在典型单请求场景下，Decode 阶段是典型的**访存密集型（Memory-bandwidth-bound）**任务（只有当系统并发批次极大、GEMM 算力占比显著上升时，瓶颈才可能向计算受限转移）。

聊天界面上跳动的一字一顿的打字机体验，其底层原因正是每生成一个词，硬件就必须执行一次完整的单步网络前向计算与权重搬运。这种流式输出的网络传输实现细节见 [流式输出怎么实现？](/fundamentals/inference-and-generation/streaming.md)。

---

## 语言模型头（LM Head）：从隐藏向量到词表概率

无论是 Prefill 阶段末尾选出首字，还是 Decode 阶段每一步迭代，模型最后一层吐出的隐藏向量 $h_t \in \mathbb{R}^{d_{\text{model}}}$ 都只是连续的高维数字。它是怎么变成词表里具体的某一个字的？

这个转换依靠**语言模型头（LM Head）**完成。LM Head 本质上是一个没有非线性激活函数的线性全连接层，其权重矩阵大小为 $W_{\text{head}} \in \mathbb{R}^{V \times d_{\text{model}}}$。

计算非常直接：

$$z_t = W_{\text{head}} \cdot h_t \in \mathbb{R}^V$$

产出的向量 $z_t$ 被称为 **Logits（未归一化对数概率）**。它的长度正好等于词表大小 $V$。向量中的每一个位置的值，对应了词表中对应词的偏好打分（可以是任意正负实数）。

![LM Head 线性投影机制](/images/content-rebuild/55d5c64d70627310/fig-4.png)

这里需要明确一个关键事实：**Transformer 神经网络本身是一个完全确定性的数学函数**。只要权重固定、输入的 Token 序列固定，模型前向推导出的 Logits 向量数值就是唯一确定的，模型内部没有任何“掷骰子”的逻辑。

日常体验到的模型随机多样性，来源于模型**外部推理系统**的采样决策：

$$\text{Probabilities} = \text{softmax}(z_t)$$

* **贪心搜索（Greedy Search）**：直接取得分最高的一项（$\arg\max(z_t)$），每次运行结果完全一致；
* **随机采样（Sampling）**：将 Logits 经过 Softmax 转化为概率分布后，依据概率分布进行加权抽样。具体的截断与平滑策略见 [Temperature、Top-k、Top-p 是什么？](/fundamentals/inference-and-generation/temperature-topk-topp.md)。

![模型确定性与外部采样策略边界](/images/content-rebuild/55d5c64d70627310/fig-5.png)

---

## 用一段纯 Python 代码跑通全流程

为了直观展示张量维度的变化，我们使用 NumPy 搭建一个只包含 5 个词的极简玩具模型，把贯穿算例完整跑一遍：

```python
import numpy as np

# 1. 词表与基础维度配置
vocab = {0: "大", 1: "模", 2: "型", 3: "好", 4: "<eos>"}
V = len(vocab)       # 词表大小 V = 5
d_model = 4          # 隐藏维度 d_model = 4

# 为复现“大 模 -> 型 -> 好 -> <eos>”推导链，设定教学用确定性权重
# 词嵌入使用基向量表示各个词的核心特征
W_embed = np.array([
    [1.0, 0.0, 0.0, 0.0],  # 0: "大"
    [0.0, 1.0, 0.0, 0.0],  # 1: "模"
    [0.0, 0.0, 1.0, 0.0],  # 2: "型"
    [0.0, 0.0, 0.0, 1.0],  # 3: "好"
    [0.0, 0.0, 0.0, 0.0],  # 4: "<eos>"
])

# LM Head 投影矩阵 (5, 4)：前一个词的特征映射到下一个目标词的高分 Logits
W_head = np.array([
    [0.0, 0.0, 0.0, 0.0],  # 0: "大"
    [0.0, 0.0, 0.0, 0.0],  # 1: "模"
    [0.0, 2.0, 0.0, 0.0],  # 2: "型"（当输入为“模”时得分最高）
    [0.0, 0.0, 2.0, 0.0],  # 3: "好"（当输入为“型”时得分最高）
    [0.0, 0.0, 0.0, 2.0],  # 4: "<eos>"（当输入为“好”时得分最高）
])

def mock_transformer_layer(hidden_states):
    """教学示意：模拟多层 Transformer 变换，输入与输出形状一致"""
    return hidden_states + 0.1 * np.tanh(hidden_states)

# --- 阶段一：Prefill 阶段（处理全量提示词） ---
prompt_tokens = [0, 1]  # 代表提示词 "大 模"
print(f"[Prefill 阶段] 输入提示词 Token IDs: {prompt_tokens}")

# 查表得到词嵌入向量，形状为 (2, 4)
input_embeddings = W_embed[prompt_tokens]
hidden_states = mock_transformer_layer(input_embeddings)

# 只截取最后一个位置的隐藏向量（对应 '模'），形状为 (4,)
last_h = hidden_states[-1]

# LM Head 投影，输出长度为 5 的 Logits 向量
logits = np.dot(W_head, last_h)
next_token = int(np.argmax(logits))  # 贪心选择得分最高者
print(f"[Prefill 完成] 产出首个新 Token ID: {next_token} ('{vocab[next_token]}')")

# --- 阶段二：Decode 阶段（串行自回归循环） ---
current_tokens = prompt_tokens + [next_token]
max_steps = 4

for step in range(max_steps):
    # 将已有全量序列送入（注：实际生产中使用 KV Cache 避免全量重算）
    seq_embed = W_embed[current_tokens]
    h_all = mock_transformer_layer(seq_embed)
    step_last_h = h_all[-1]
    
    step_logits = np.dot(W_head, step_last_h)
    predicted_token = int(np.argmax(step_logits))
    
    current_tokens.append(predicted_token)
    print(f"[Decode 步数 {step + 1}] 生成 Token ID: {predicted_token} ('{vocab[predicted_token]}')")
    
    # 检查是否命中结束符
    if predicted_token == 4:  # 遇到 <eos>
        print("[Decode 终止] 命中 <eos> 标记，推理结束。")
        break
```

运行这段代码，可以直观观察到两个核心特征：
1. 隐藏向量的最后一维始终是 $d_{\text{model}} = 4$，而 LM Head 输出的 Logits 长度始终是词表大小 $V = 5$；
2. 每一个循环生成出来的 Token 必须被追加回序列，作为驱动下一步计算的依据。

---

## 循环什么时候停下来？

在代码中，Decode 阶段本质上是一个 `while True` 循环。如果没有明确的拦截机制，模型就会无限生成下去。实际系统中，推理引擎依赖以下四种条件触发退出：

![生成循环终止状态判定机](/images/content-rebuild/55d5c64d70627310/fig-6.png)

1. **命中结束标记（EOS Token）**：模型在训练阶段学会了何时该收尾。当模型认为表述已完整时，LM Head 投影到 `<eos>`、`<|endoftext|>` 或 `<|im_end|>` 的得分最高并被选中。引擎检测到该 ID 后正常退出。
2. **达到最大生成长度（Max Tokens）**：客户端请求时通常带有 `max_tokens` 参数。一旦生成步数达到设定上限，系统强制截断退出，防止模型出现复读死循环耗尽资源。
3. **命中预设停止词（Stop Sequences）**：在工具调用或特定格式输出中，开发者可以配置停止字符串（例如 `Observation:` 或 `}`）。解码器每产出一个词，就会检查尾部文本，一旦匹配即刻停止。详见 [结构化输出和 Function Calling 是什么？](/fundamentals/inference-and-generation/structured-output.md)。
4. **客户端主动断开或超时**：如果用户在界面点击了“停止”按钮，或者网关 HTTP 连接超时，底层计算协程收到中断信号后会立刻释放 GPU 资源并终止循环。

---

## 常见误区澄清

* **误区 1：大模型是整句话想好后一次性吐出来的。**
  * 真实机制：大模型内部完全是按时间步递推的因果链。每一步前向计算都只能以过去的静态历史为输入，不存在“从后往前修改”或“整体全局排版”的计算路径。
* **误区 2：LM Head 算出来的直接就是最终概率。**
  * 真实机制：LM Head 是一次普通的矩阵乘法，输出的是无范围限制的实数得分（Logits）。必须经过 Softmax 函数映射后，各分项才转化为在 $[0, 1]$ 之间且总和为 1 的概率分布。
* **误区 3：Prefill 阶段处理 Token 的吞吐率远高于 Decode 阶段，是因为 Prefill 采用了压缩算法。**
  * 真实机制：Prefill 的高吞吐率并不是因为压缩了数据，而是因为提示词已知，GPU 可以用大矩阵乘法把硬件计算单元充分填满；而在典型单请求 Decode 阶段，GPU 被迫一个词一个词地串行计算，瓶颈主要受限于显存权重搬运带宽。关于如何破解这一瓶颈，详见 [vLLM 为什么推理速度快？](/llm-100/deployment/vllm-inference.md) 与 [推理加速有哪些方法？](/fundamentals/inference-and-generation/inference-acceleration.md)。

---

## 学习验收与下一步

读完本文，请对照自测以下核心概念：

- [ ] 分词器（Tokenizer）输出的是整数 ID，而词嵌入矩阵（Embedding Table）输出的是连续实数向量；
- [ ] Prefill 阶段虽然利用因果掩码阻断了后视信息，但因为输入已知，所有 Token 是一起并行做矩阵乘法的；
- [ ] Decode 阶段单步只能前向计算一个 Token，生成耗时主要受限于权重搬运带宽而非算力；
- [ ] 模型的随机性不来自模型网络本身，而是来自外部采样策略。

在 Decode 阶段，如果每生成一个新词都要像示例代码一样，把包含历史提示词在内的全部文本重新算一遍，序列越长计算量就会剧增。下一讲我们将深入解析工业界如何用缓存技术避免全量重算：[KV Cache 是什么？](/fundamentals/inference-and-generation/kv-cache.md)。
