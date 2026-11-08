---
title: 从原始 Transformer 到现代大模型，架构有哪些关键区别？
icon: mdi:compare
tag:
- Transformer
- LLM
- 架构
description: 系统梳理原始 Transformer 与现代主流 LLM 在宏观拓扑、归一化、激活函数、位置编码与词表设计上的五大演进差异与面试回答策略。
---

在深度学习与大语言模型面试中，“Transformer 与现代 LLM 有什么区别”是一道高频出现的宏观架构题。许多候选人容易简单回答“参数变大了”或“层数变多了”，但面试官真正考察的是：从 2017 年机器翻译任务出发的原始架构，演进到如今主流的自回归开源大模型（如 LLaMA、Mistral、Qwen），工程界为了克服大规模训练稳定性、长序列建模以及推理显存吞吐等瓶颈，在底层模块上做出了哪些具体的改良。

![原始 Transformer 与现代 LLM 架构演化全景](/images/content-rebuild/4ae01dfdfc54f6bf/cover.png)

---

## 30 秒口述结论

如果面试官要求在半分钟内总结，可以直接给出以下结构化回答：

> 2017 年 Vaswani 等人提出的原始 Transformer 是针对跨语言翻译设计的**双塔 Encoder-Decoder 结构**，采用 Post-LN 归一化、绝对正弦余弦位置编码、标准 ReLU 激活函数以及 32k 大小的任务特定词表。
>
> 现代主流大模型（LLM）则普遍收敛为**单塔 Causal Decoder-only 架构**，并在核心子模块上完成了四项关键替换：
> 1. **归一化位置与算子**：从 Post-LN 演进为 Pre-LN / RMSNorm，移除了均值计算与偏置项（Bias），保证百层深层网络中反向传播梯度的稳定；
> 2. **激活函数**：从 ReLU / GELU 演进为 SwiGLU 门控结构，在保持参数量相近的前提下提升表征容量；
> 3. **位置编码**：从绝对正弦编码演进为旋转位置编码（RoPE），赋予注意力矩阵相对位置感知，便于长上下文扩展；
> 4. **分词策略**：从面向特定语料的 32k 分词器扩展到 100k+ 的 Byte-level BPE，彻底消除未知词（OOV），大幅提升跨语言压缩效率。

---

## 贯穿示例：处理同一段文本的数据流转对比

为了直观理解这两种架构的不同运作方式，我们来看一个具体例子：
输入提示为 `“今天天气真好，我想去”`，目标是接着生成 `“公园散步”`。

### 1. 原始 Transformer 的处理路径
原始 Transformer 假定任务是“输入一个完整句子，翻译或转换为另一个句子”：
1. **编码阶段**：源句子 `“今天天气真好，我想去”` 完整输入给 Encoder。每个 Token 加上基于绝对位置索引的正弦向量，在无掩码的双向自注意力层中互相交互，输出最终的全局表征向量矩阵。
2. **解码阶段**：Decoder 必须启动两类注意力。在生成 `“公园”` 时，Decoder 内部先做 Masked Self-Attention（因果自注意力）；紧接着做 **Cross-Attention（交叉注意力）**，使用 Decoder 当前生成的 Query 去查询 Encoder 输出的 Key 和 Value。
3. **推理开销**：服务系统既要长期常驻 Encoder 的表征缓存，又要维护 Decoder 自身的增量 KV 缓存，跨网络调度和显存管理的复杂度较高。

### 2. 现代 Decoder-only LLM 的处理路径
现代大模型将输入与输出视为同一段自回归文本的先后片段：
1. **统一序列输入**：输入 Prompt `“今天天气真好，我想去”` 直接进入单塔 Decoder。
2. **因果下三角掩码**：所有 Token 在同一个注意力掩码下流动，前文只能看到前文，无法看到未来的位置。
3. **自回归递推**：网络计算完 Prompt 后预测出下一个词 `“公”`；随后将 `“公”` 拼入序列，仅更新新 Token 对应的键值并写入 KV Cache，继续预测 `“园”`。
整个系统只存在一套计算管线与一份自注意力缓存。

---

## 架构拓扑演进：双塔到单塔自回归

宏观结构的变化并非审美选择，而是工程训练和推理调度权衡的结果。

![双塔与单塔拓扑结构对比](/images/content-rebuild/4ae01dfdfc54f6bf/fig-1.png)

### 为什么工业界放弃了 Encoder-Decoder？

1. **无监督预训练范式的统一**：
   双塔架构需要成对的“源序列-目标序列”数据或人工设计的填充掩码目标（如 T5 的 Span Corruption）。而单塔因果自回归模型（Next-token prediction）可以直接消费海量的无标注互联网纯文本，输入与输出形式天然统一。
2. **推理引擎与显存调度的精简**：
   在自回归推理解码时，Decoder-only 只需维护持续单调增长的自身 KV Cache。如果保留 Cross-Attention，显存调度系统不仅要为不同请求保存大小不一的 Encoder 输出矩阵，还要在 Prefill（首字生成）和 Decode（逐字生成）两套阶段维持异构计算图，难以像单塔结构那样轻松接入 PagedAttention 等显存分页算法。
3. **上下文学习（In-Context Learning）的适配**：
   单塔结构将示范样例（Few-shot Examples）和实际问题天然排布在同一因果时序流中，使模型在同一个注意力空间中学习规律。更深入的拓扑取舍分析见 [为什么主流大模型采用 Decoder-only 架构？](/interview/llm-interview/decoder-only.md)。

<ReadMoreLock />

---

## 核心子模块的四项工程替换

下表汇总了原始 Transformer 与现代大模型在各个关键组件上的对比：

| 维度 | 原始 Transformer (2017) | 现代主流开源 LLM (LLaMA / Qwen 等) | 工程改进原因 |
| :--- | :--- | :--- | :--- |
| **整体结构** | Encoder-Decoder 双塔 | Causal Decoder-only 单塔 | 统一预训练目标，精简推理显存管理 |
| **归一化位置** | Post-LN | Pre-LN / RMSNorm | 构筑无阻碍恒等映射通道，保证深层梯度平稳 |
| **归一化计算** | 标准 LayerNorm（计算均值与方差） | RMSNorm（仅计算均方根） | 移除均值规约操作，降低 GPU 显存带宽读取开销 |
| **激活函数** | 标准 ReLU（或部分变体的 GELU） | SwiGLU 门控结构 | 引入动态门控，提供更平滑的非线性表征能力 |
| **位置编码** | 绝对正弦余弦编码（直接加在词向量上） | 旋转位置编码 RoPE（作用于 Q、K 向量） | 具备相对位置保持性，原生支持长文本扩展与外推 |
| **注意力变体** | 标准 Multi-Head Attention (MHA) | Grouped-Query Attention (GQA) | 降低推理自回归阶段的 KV Cache 显存容量与带宽瓶颈 |

### 1. 归一化演进：Post-LN 到 Pre-LN 与 RMSNorm

在原始 Transformer 中，子层输出的公式为：
$$x_{l+1} = \text{LayerNorm}(x_l + \text{Sublayer}(x_l))$$
这种将归一化置于残差加法之后的结构称为 **Post-LN**。

![Post-LN 与 Pre-LN 梯度传播机制对比](/images/content-rebuild/4ae01dfdfc54f6bf/fig-2.png)

在 Post-LN 中，反向传播的梯度经过每一层时，都必须乘以该层 LayerNorm 的导数矩阵。当模型层数加深（例如达到 32 层或 80 层）时，复合导数的连乘极易导致浅层梯度的方差剧烈波动，造成深层训练前期的不稳定。因此，原始 Transformer 对 Warmup 阶段的学习率调度非常敏感。

现代大模型普遍采用 **Pre-LN**：
$$x_{l+1} = x_l + \text{Sublayer}(\text{Norm}(x_l))$$
在 Pre-LN 拓扑下，主干残差路径在深度方向上构成了一条直通的恒等映射通道（$\frac{\partial x_{l+1}}{\partial x_l} = I + \dots$）。浅层参数可以直接接收到顶层回传的未经缩放扭曲的梯度，使得百层以上的超深网络能够在无严苛超参数调优的前提下稳定收敛。

![LayerNorm 与 RMSNorm 运算简化对比](/images/content-rebuild/4ae01dfdfc54f6bf/fig-3.png)

在此基础上，现代模型进一步使用 **RMSNorm** 替换了标准 **LayerNorm**。
标准 LayerNorm 包含减均值和下方差缩放两个规约步骤：
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta, \quad \mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$$
RMSNorm 认为归一化的关键收益来自对神经元激活尺度的控制，中心化偏移（减去均值 $\mu$）对最终表征质量的贡献极小，因此直接去除均值计算：
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$
在 GPU 物理执行上，LayerNorm 需要多次在片上共享内存（SRAM）与高带宽显存（HBM）之间同步和加载中间张量。RMSNorm 省去了一次全局规约与元素减法，有效减少了显存带宽的读写次数，提升了算子执行速度。更多推导见 [LayerNorm 和 RMSNorm 有什么区别？](/interview/llm-interview/layernorm-rmsnorm.md)。

### 2. 激活函数演进：ReLU 到 SwiGLU

原始 Transformer 的 FFN 层由两层全连接与一个 ReLU 激活函数拼接而成：
$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

![激活函数演进：ReLU 与 SwiGLU 门控机制](/images/content-rebuild/4ae01dfdfc54f6bf/fig-4.png)

现代主流模型改用门控线性单元（GLU）的平滑变体——**SwiGLU**。其表达式为：
$$\text{SwiGLU}(x) = \left( \text{Swish}_{1}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}$$
其中 $\text{Swish}_1(z) = z \cdot \text{sigmoid}(z)$，$\odot$ 表示逐元素相乘。

**参数量等价算例**：
- 原始 Transformer 的标准 FFN 包含两组变换：$W_1 \in \mathbb{R}^{d \times 4d}$，$W_2 \in \mathbb{R}^{4d \times d}$，其权重参数总量为：
  $$2 \times (d \times 4d) = 8d^2$$
- SwiGLU 包含了三个投影矩阵：$W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{ff}}$ 以及 $W_{\text{down}} \in \mathbb{R}^{d_{ff} \times d}$。若直接令 $d_{ff} = 4d$，参数量将增加为 $3 \times 4d^2 = 12d^2$（膨胀 50%）。
- 为了在引入门控机制的同时保持整体计算量和参数量与传统结构相当，工程设计（如 LLaMA）将中间升维维度缩小为原先的 $\frac{2}{3}$，即：
  $$d_{ff} \approx \frac{8}{3}d$$
  此时 SwiGLU 的总参数量为 $3 \times \left(d \times \frac{8}{3}d\right) = 8d^2$，与传统 FFN 完全持平。实验表明，这种双路径门控过滤在相同参数预算下具有更优的收敛速度和下游准确率。详细细节见 [Transformer 中 FFN 层有什么作用？](/interview/llm-interview/ffn-swiglu.md)。

### 3. 位置编码演进：绝对正弦编码到 RoPE

在原始 Transformer 中，位置信息是在网络的最底层一次性加到词嵌入向量上的：
$$x_{\text{input}} = x_{\text{embed}} + P_{\text{sinusoidal}}$$
这种加法机制存在两个明显短板：
1. 位置信号只在输入层注入，随着网络深度增加，显式的位置信息容易被逐层的非线性投影逐渐稀释；
2. 它是绝对坐标的物理叠加，在自注意力点积计算 $q_m^\top k_n$ 时，难以显式呈现相对位移 $(m - n)$ 的数学约束。

![位置编码演进：绝对正弦加法与 RoPE 旋转矩阵乘法](/images/content-rebuild/4ae01dfdfc54f6bf/fig-5.png)

现代开源模型几乎全盘转向**旋转位置编码（RoPE）**。RoPE 不在 Embedding 上执行加法，而是在计算注意力分数前，对处于序列绝对索引 $m$ 的 Query 向量与绝对索引 $n$ 的 Key 向量分别乘以一个正交旋转矩阵：
$$\tilde{q}_m = R_{\Theta, m} q_m, \quad \tilde{k}_n = R_{\Theta, n} k_n$$
此时两者的点积为：
$$\langle \tilde{q}_m, \tilde{k}_n \rangle = (R_{\Theta, m} q_m)^\top (R_{\Theta, n} k_n) = q_m^\top \left( R_{\Theta, m}^\top R_{\Theta, n} \right) k_n = q_m^\top R_{\Theta, n - m} k_n$$
结果表明，注意力权重只取决于原始特征向量 $q, k$ 以及相对位移 $(n - m)$。这一特性让模型既保留了绝对位置的计算便利性，又具备了相对位置的平移不变性，为后续通过内插或外推算法支持超长上下文（如 32k 到 128k）奠定了数学基础。数学推导与代码实现见 [了解 LLaMA 中的旋转位置编码吗？](/interview/llm-interview/llama-rope.md)。

---

## 分词与词表：从专用 32k 到通用 100k+ 字节级编码

模型底层的表征粒度也发生了显著跃升。

![分词器词表与 Token 压缩比对比](/images/content-rebuild/4ae01dfdfc54f6bf/fig-6.png)

原始 Transformer 针对双语机器翻译设计，词表规模普遍较小，大约在 **32,000** 左右。较小的词表在处理中文、代码以及复杂字符时，容易将一个词拆碎为数个甚至十几个细小片段。一旦遇到训练语料中未覆盖的特殊符号，系统往往只能输出未登录词（Out-Of-Vocabulary, `<unk>`）标记。

现代主流大模型全面采用 **Byte-level BPE**（结合字节回退机制的 BPE，如 tiktoken 或 SentencePiece 的 byte-fallback 实现），词表规模普遍扩展至 **100,000 到 150,000**（例如 LLaMA 3 为 128k，Qwen 系列达 150k+）。

这带来了三个核心维度的收益与权衡：
1. **彻底杜绝 `<unk>` 标记**：遇到非高频词汇时，分词器会自动向下回退到基础 UTF-8 字节进行分块编码。任何 Unicode 字符均可被准确拆分和无损重构，不会出现字符丢失。
2. **提高文本压缩比与推理速度**：更大的词表使模型能够将常用的成语、多语言短语和代码关键词压缩为单个 Token。在固定上下文窗口长度（例如 4096）下，大词表可以装载更多自然语言信息；同时，生成相同语义所需的自回归串行步数更少，显著减少了端到端的生成时间，详见 [Token 是什么？分词算法有什么区别？](/interview/llm-interview/tokenization.md)。
3. **显存与参数的平衡**：词表过大也会带来代价。输入 Embedding 层和输出反向投影头（LM Head）的矩阵尺寸等于 `词表大小 × 隐藏维度 d`。在超大词表下，这两层参数在模型总参数中的占比会上升，需要在文本压缩收益与静态参数显存之间做出工程折中。

---

## 面试追问与边界考察

在回答完主体演进后，面试官通常会进一步深挖工程取舍背后的原因。

### 追问 1：既然 Pre-LN 训练更稳定，为什么 2017 年的原始 Transformer 选择了 Post-LN？

**答案解析**：
在 2017 年论文实验中，网络层数较浅（基础版 Encoder 和 Decoder 各仅 6 层，总参数量约 6500 万）。在小规模、浅层网络配合仔细调优的学习率预热（Warmup）时，**Post-LN 充分收敛后的性能上限往往略高于 Pre-LN**。
这是因为 Post-LN 每次残差相加后都强制重新归一化，各层表征的尺度受到强力约束，网络各层的特征提取潜力被利用得更加充分。而 Pre-LN 的残差主干随着深度增加，信号方差会逐渐累加，导致深层子层对主干的相对修改幅度逐渐变小。但是，当大模型的参数量提升至百亿、千亿，层数达到 32、80 甚至更高时，Post-LN 的梯度数值剧烈震荡会导致训练中途崩溃。工业界追求“能够稳定按期完成千卡训练”，因此 Pre-LN 成为了事实标准。

### 追问 2：为什么现代大模型普遍移除了线性层中的偏置项（Bias）？

**答案解析**：
包括 LLaMA、PaLM 在内的现代架构大多去除了投影全连接层中的偏置项 $b$（即仅计算 $y = x W$），这主要基于三点工程考量：
1. **保障长上下文外推时的数值平稳**：偏置项是一个固定的向量漂移。在长序列的逐层自回归累加中，固定偏置项容易累积为激活张量的系统性均值偏移，引发注意力权重向极值饱和。
2. **规避低精度数值溢出**：在 FP16 或 BF16 混合精度预训练中，极端偏置与动态激活的相加容易触碰数值动态范围的边界，增加溢出风险。
3. **精简 GPU 访存操作**：去除偏置项后，前向和反向传播可以纯粹执行为标准的高性能通用矩阵乘法（GEMM），无需在显存中额外读写偏置张量并执行元素级加法（Element-wise Add），微幅提高了算子运行效率。

### 3. 追问 3：现代大模型是否彻底淘汰了 Encoder 结构？

**答案解析**：
对于通用文本生成和代码编写任务，Causal Decoder-only 凭借自回归预训练极高的计算利用率和扩展效率占据了绝对主流。但这并不意味着 Encoder 结构在工业界完全消失：
- **稠密语义检索与嵌入模型**：在 RAG 的文本向量化阶段，BGE、Contriever 等检索模型仍然采用双向 Encoder（如 BERT 结构）。由于不需要生成后续文本，双向全可见的自注意力能够提取信息密度更高的全篇语义嵌入。
- **连续感知信号编码**：在语音处理（如 OpenAI Whisper）和多模态图像感知（如 CLIP、ViT）中，前端普遍使用 Encoder 抽取密集的连续频谱或图像块表征，后端再通过交叉注意力或线性映射输入给自回归解码器完成文本生成。结构选型始终取决于输入数据是需要双向全局感知的静态信号，还是需要因果递推生成的离散时序。
