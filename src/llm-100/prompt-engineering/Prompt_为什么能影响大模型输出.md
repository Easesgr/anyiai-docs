---
title: Prompt 为什么能影响大模型输出？从条件概率到自回归生成
icon: mdi:message-text-outline
date: 2026-07-16
tag:
- 大模型100讲
- Prompt
description: 剖析 Prompt 影响模型输出的因果链条：权重冻结下，Prompt 作为前置条件构建键值缓存，改变词表未归一化分值，从而收窄候选词采样范围。
---

在日常调用大语言模型时，修改一段提示词（Prompt）往往会带来完全不同的生成结果。有些解释喜欢将提示词比作人际交流中的“情绪唤醒”或“心理暗示”。这种拟人化的理解虽然直观，却无法解释模型实际的工程计算过程。

大语言模型在推理阶段的权重参数是完全冻结的（梯度 $\nabla_\theta = 0$）。Prompt 并不能改写模型内部任何一个浮点数参数，它之所以能够决定输出内容，是因为它作为**前置条件序列**输入了因果自注意力网络：

1. Prompt 首先构建了前向传播中的键值缓存（KV Cache）；
2. 这一组缓存向量参与后续每一个生成步的注意力打分；
3. 打分结果改变了末位隐向量在词表上的未归一化打分（Logits），从而大幅收窄下一个词元的采样范围。

以“苹果”这个词为例：如果直接输入“苹果”，模型既可能补全为水果特征，也可能补全为科技公司动态；一旦在前文加上“请以科技行业分析师的视角，解读苹果公司的最新季度重点：”，候选词在“财报”、“营收”、“净利润”上的概率就会被显著推高，而“水果”、“维生素”等选项的概率则降至接近于零。

![条件概率与自回归生成的整体机制全景](/images/content-rebuild/fc2ace959941306a/cover.png)

上图展示了 Prompt 从输入文本到干预模型采样的全链路：离散 Token 转化为初始向量后，构建起自回归注意力的前置条件，最终在固定的参数矩阵上导向不同的 Token 输出空间。阅读本篇时，请将注意力集中在“前置条件如何约束分布”这一主线上。

---

## 1. 自回归生成定义：从序列联合概率到条件采样

自回归语言模型的基本任务，是估计文本词元序列 $X = (x_1, x_2, \dots, x_T)$ 的联合概率分布：

$$P(X) = P(x_1, x_2, \dots, x_T)$$

利用概率乘法公式，任意文本序列的联合概率都可以拆解为单步条件概率的连乘积：

$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t \mid x_{<t}) = \prod_{t=1}^T P(x_t \mid x_1, \dots, x_{t-1})$$

当我们给模型提供一段由 $m$ 个 Token 组成的提示词 $X_{\text{prompt}} = (x_1, x_2, \dots, x_m)$ 时，模型不需要从头预测这前 $m$ 个词，而是直接计算后续回答序列 $Y = (y_1, y_2, \dots, y_n)$ 在该提示词条件下的后验概率：

$$P(Y \mid X_{\text{prompt}}) = \prod_{j=1}^n P(y_j \mid x_1, \dots, x_m, y_1, \dots, y_{j-1})$$

![自回归生成中的条件概率推演流程](/images/content-rebuild/fc2ace959941306a/fig-1.png)

上图展示了联合概率链式分解在因果解码器中的展开形式。图中的时间箭头表明：每一个生成 Token 都必须严格基于历史上下文与初始 Prompt。请注意前置 Token 长度如何直接拉长了条件依赖链条。

在工程实现中，因果解码器不能在输入维度为 0 时计算前向传播，必须至少提供一个序列起始符（如 `<BOS>`）。如果只输入起始符而不提供具体的业务上下文，模型在第一步只能依赖预训练语料的起始词平均分布，后续生成必然漫无边际。

编写 Prompt 的过程，本质上就是在为模型划定计算切片。Prompt 中的词汇越清晰、约束越完备，条件分布 $P(y_t \mid X_{\text{prompt}}, y_{<t})$ 的不确定性（信息熵）就越低，生成内容也就越容易收敛到预期目标。

---

## 2. Prefill 阶段与 KV 缓存：Prompt 在注意力计算中的物理实体

从文本到张量计算，输入的 Prompt 文本首先经由分词器（Tokenizer）转换为离散词元序列，再通过嵌入层（Embedding）与位置编码相加，形成连续的输入矩阵：

$$E_0 = \text{Embedding}(X_{\text{prompt}}) \in \mathbb{R}^{m \times d_{\text{model}}}$$

在预填充（Prefill）阶段，该矩阵被输入 Transformer 各层。在每一层中，输入向量分别与可学习的投影矩阵 $W_Q, W_K, W_V$ 相乘，得到对应的查询（Query）、键（Key）和值（Value）张量：

$$Q = E W_Q, \quad K = E W_K, \quad V = E W_V$$

因果注意力使用下三角掩码矩阵 $M$ 来阻止当前位置查看未来的词元：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

其中掩码矩阵 $M$ 在位置 $j > i$ 处为 $-\infty$，在 $j \le i$ 处为 $0$。

![Prefill 阶段 KV 向量构建与因果掩码计算结构](/images/content-rebuild/fc2ace959941306a/fig-2.png)

上图清晰地给出了 Prefill 阶段的矩阵运算拓扑。下三角注意力掩码切断了未来的信息回流，使得 Prompt 中的每一个 Token 都生成了一组稳定的 Key 与 Value 向量并存入缓存中。理解这一点是掌握后续推理开销与上下文机制的前提。

Prefill 计算完成后，模型生成了常驻在显存中的初始 **KV Cache（键值缓存）**：

$$K_{\text{context}} = [k_1, k_2, \dots, k_m], \quad V_{\text{context}} = [v_1, v_2, \dots, v_m]$$

在预填充阶段结束时，末位词元 $x_m$ 汇聚全序列注意力后输出隐藏状态 $h_m$，直接映射为 Logits 采样出第一个词元 $y_1$。当进入后续自回归解码步预测下一个词元（例如 $y_2$）时，新输入的词元 $y_1$ 产生新的查询向量 $q_{m+1}$，与显存中常驻的 Prompt 键向量 $k_1 \dots k_m$ 及新键向量计算内积。Prompt 在物理层面并不是转瞬即逝的字符串，而是转化为一组常驻显存的参考坐标（Key）与信息载荷（Value），在后续每一步自回归采样中持续参与加权。

<ReadMoreLock />

---

## 3. Logits 偏移：提示词如何改变全词表的采样概率

经过多层注意力网络与前馈网络（FFN）抽取后，模型顶层在当前序列末尾输出一个上下文隐藏状态向量 $h_T \in \mathbb{R}^{d_{\text{model}}}$。

该向量经过最终层归一化后，与未嵌入矩阵（Unembedding Matrix，即 LM Head）$W_{\text{vocab}} \in \mathbb{R}^{|\mathcal{V}| \times d_{\text{model}}}$ 发生点乘，输出整个词表上每个候选词的未归一化打分向量（Logits）：

$$z = W_{\text{vocab}} h_T \in \mathbb{R}^{|\mathcal{V}|}$$

通过带温度系数 $T$ 的 Softmax 函数，Logits 被转换为概率分布：

$$P(y_t = v \mid X) = \frac{\exp(z_v / T)}{\sum_{w \in \mathcal{V}} \exp(z_w / T)}$$

不同的 Prompt 会改变末位隐向量 $h_T$ 在向量空间中的位置，进而直接改变整个词表的 Logits 分布。这就是 Prompt 调控采样的核心路径。

![Logits 偏移与采样空间约束对比机制](/images/content-rebuild/fc2ace959941306a/fig-3.png)

上图对比了在宽泛提示词与高约束提示词下，词表 Logits 分布的变化情况。当提示词施加明确约束时，目标词元的 Logits 显著拉升，将概率质量集中于极窄区间，使无效选项在采样前即被概率截断。有关解码参数的进一步控制，可参考 [/llm-100/prompt-engineering/Temperature_Top-p_Top-k_分别控制什么.md](/llm-100/prompt-engineering/Temperature_Top-p_Top-k_分别控制什么.md)。

为了直观展示这一偏移，我们用一个包含 4 个候选词的简化模型进行测算。

### 简化算例：Prompt 介入前后的概率再分配

假设简化词表仅有 4 个候选词元：`["苹果", "香蕉", "手机", "财报"]`，解码温度取 $T=1.0$。

**场景 A：模糊提示词**
* 输入：“谈谈最近关注的内容。”
* 模型顶层输出的未缩放 Logits 为：
  $$\mathbf{z}_A = [2.0, 1.8, 2.1, 1.9]$$
* 计算 $\exp(z)$ 得：$[7.39, 6.05, 8.17, 6.69]$，求和 $\sum \exp(z) \approx 28.30$。
* 经过 Softmax 归一化后的预测概率为：
  $$\mathbf{p}_A \approx [0.261, 0.214, 0.289, 0.236]$$
* 现象：4 个选项的生成概率都在 $20\% \sim 29\%$ 之间，系统不确定性高，任何一个词都有可能被采到。

**场景 B：加入领域约束的提示词**
* 输入：“请以科技行业分析师的视角，解读苹果公司的最新季度财报重点。”
* “分析师”、“科技行业”、“季度财报”等词元经过注意力聚合后，将末位隐向量 $h_T$ 强烈拉向财务分析语义区。模型给出的 Logits 发生偏移：
  $$\mathbf{z}_B = [3.5, -1.0, 2.8, 6.2]$$
* 计算 $\exp(z)$ 得：$[33.12, 0.37, 16.44, 492.75]$，求和 $\sum \exp(z) \approx 542.68$。
* 经过 Softmax 归一化后的预测概率为：
  $$\mathbf{p}_B \approx [0.061, 0.0007, 0.030, 0.908]$$
* 现象：候选词“财报”的概率提升到 $90.8\%$，“香蕉”的概率则降至 $0.07\%$。模型在这一步几乎必然选中与财报分析相关的词元。

模型并没有学到“新知识”，而是通过改变注意力上下文，把原本低概率的特定词汇推上了绝对优势区。

---

## 4. 模式匹配：提示词如何调动预训练知识

既然模型权重未发生改变，Prompt 是如何让模型表现出“写代码”、“做翻译”或“做推理解题”等不同角色的？

在 Transformer 结构中，自注意力模块负责在词与词之间搬运与整合上下文信息，而前馈网络（FFN）通常被视作存储键值联想记忆的模块：
* FFN 的第一层线性变换负责检测当前残差流中是否存在某种特定的模式组合；
* 第二层线性变换则把该模式对应的信息特征加回到残差流中。

![提示词在残差流与神经元联想通路中的路由机制](/images/content-rebuild/fc2ace959941306a/fig-4.png)

上图展示了 Prompt 表征在残差流中调动特定 FFN 神经元模式的机制。密集 Transformer 并不存在物理上的离散分流开关，所有输入均经过全量矩阵投影；Prompt 提供的专业词元在残差流中构建起特定的语义表征，使其在连续向量投影下更强烈地激活 FFN 中对应的神经元子集。关于如何用角色前缀进行精准引导，可在 [/llm-100/prompt-engineering/System_Prompt_User_Prompt_Assistant_Prompt_分别有什么作用.md](/llm-100/prompt-engineering/System_Prompt_User_Prompt_Assistant_Prompt_分别有什么作用.md) 中找到不同角色的定位差异。

预训练语料库包含了各种体裁的文本（开源代码仓库、百科全书、技术论文、小说散文）。如果提示词写得十分口语化，模型残差流便会倾向于激活日常闲聊的神经元组合；如果提示词明确声明：

> “你是一名资深分布式系统架构师，请评估 Raft 协议在跨地域高延迟场景下的日志复制瓶颈。”

这组专业词元组合会在浅层注意力层中快速组合，将残差流定向推入与分布式系统相关的特征区域，从而抑制闲聊、童话等无关文本模式，激活专业词汇（如 RTT、心跳超时、多数派仲裁）的预测概率。

Prompt 就像一把检索钥匙：它调动的是预训练阶段已经固化在参数里的关联结构，无法凭空创造模型训练阶段完全未曾接触过的真实世界事实。

---

## 5. Prompt 生效的前提与边界

在工程实践中，盲目加长或堆叠 Prompt 并不总能提升生成效果。Prompt 的控制力受到模型运行机制的多重制约：

### 边界一：背景信息竞争与长距离衰减
在长上下文处理中，常出现提示词指令被忽略的现象（即“迷失在中间”现象）。

从计算机制上看，注意力分数是各键向量相对打分的结果：
1. **背景键向量竞争**：当输入长篇背景材料时，材料中成千上万个词元的键向量（Key）会共同竞争注意力打分份额，导致核心指令分配到的注意力权重被稀释；
2. **位置编码外推特性**：现代相对位置编码（如 RoPE）随着 token 间距拉长，高频相对位置信号逐渐衰减，模型对超长距离外的指令敏感度会有所下降。关于如何精简组织有效上下文，可参考 [/frontier/context-engineering/high-quality-context.md](/frontier/context-engineering/high-quality-context.md)。

![长序列下无关键向量竞争与长程表征退化的注意力衰减分析](/images/content-rebuild/fc2ace959941306a/fig-5.png)

上图展示了长序列下注意力失效的实际物理机理。画面直观展现了噪音 Key 的打分竞争以及长程位置编码退化，解释了为何深埋在中间的指令会被边缘化，从而在解码步骤中失去对采样空间的绝对控制力。

### 边界二：分词差异引起的路径漂移
自然语言在输入模型前必须由分词器切分成离散的 Token ID。字面语义接近的两句话，切分出的 Token 序列及其对应的向量可能大不相同。

例如，“解释TCP协议”与“请解释 TCP 协议”在词元切分位置、Token 数量及相对位置编号上都有差异。自回归生成具有较强的步进依赖性，前一个 Token 采样的微小变化会在接下来的每一步迭代中持续积累，最终使两段回答的表述风格出现明显分野。详细分析可见 [/llm-100/prompt-engineering/为什么同一个_Prompt_每次输出不一样.md](/llm-100/prompt-engineering/为什么同一个_Prompt_每次输出不一样.md)。

### 边界三：超出预训练知识分布时的退化
Prompt 只能引导模型在已学得的参数规律内进行组合。如果指令强行要求模型回答其预训练未覆盖的事实（例如捏造不存在的历史事件），注意力机制无法在参数记忆中找到强支持。为了满足自回归语言生成的流畅度，模型会将采样概率转移到局部高频的共现词汇上，从而产生一本正经的幻觉生成。

![分布外指令导致的先验偏离与幻觉生成轨迹](/images/content-rebuild/fc2ace959941306a/fig-6.png)

上图说明了分布外提示词引发的失效路径。当提示词要求超出预训练流形支撑集的输出时，指令强加的约束与底层语言平滑先验发生冲突，模型无法形成有效注意力焦点，最终沿着局部高频的幻觉路径完成自回归。

---

## 6. 可检验练习：验证 Prompt 约束对首词信息熵的影响

通过对模型首个生成 Token 的概率分布计算信息熵，可以量化观察 Prompt 对采样空间的收窄效应。信息熵越低，说明模型的输出越聚焦、不确定性越小。

香农信息熵公式：
$$H(P) = - \sum_{i} P(v_i) \log_2 P(v_i)$$

运行以下 Python 代码验证这一现象：

```python
import numpy as np

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))  # 防止数值溢出
    return exp_scaled / np.sum(exp_scaled)

def calculate_entropy(probs: np.ndarray) -> float:
    valid_p = probs[probs > 1e-12]
    return -np.sum(valid_p * np.log2(valid_p))

# 场景 1：输入开放提示词 "今天天气"
# Top-5 候选词 Logits: ["真好", "不错", "有点", "阴沉", "如何"]
open_logits = np.array([5.2, 4.9, 4.8, 4.5, 4.1])

# 场景 2：输入强约束提示词 "根据气象局标准，天气回答只能为'晴'或'雨'。今天天气："
# Top-5 候选词 Logits: ["晴", "雨", "未知", "阴", "雾"]
constrained_logits = np.array([12.5, 12.1, 1.2, 0.5, 0.1])

p_open = softmax(open_logits)
p_constrained = softmax(constrained_logits)

print(f"开放提示词首词信息熵: {calculate_entropy(p_open):.3f} bits")
print(f"受限提示词首词信息熵: {calculate_entropy(p_constrained):.3f} bits")
```

**检验标准**：
1. 开放提示词由于候选词分布平缓，首词信息熵相对较高（通常在 2.0 bits 以上）；
2. 增加格式与候选词约束后，概率主要集中在前两个选项上，信息熵显著下降（通常降至 1.0 bit 左右）。这直观验证了 Prompt 是通过概率重塑来控制生成方向的。

关于如何编写结构化规则来降低歧义，可参考 [/llm-100/prompt-engineering/结构化_Prompt_应该怎么写.md](/llm-100/prompt-engineering/结构化_Prompt_应该怎么写.md)。

---

## 验收测试

1. **计算机制验证**：在模型推理权重完全锁定的情况下，Prompt 是通过哪个关键物理组件（张量）在后续每一步自回归中发挥影响的？
2. **边界分析**：为什么在超长背景资料中，即便 Prompt 写得很清晰，模型有时也会忽略中间的规则指令？请给出至少两种可能的计算原因。

完成本讲后，建议继续阅读 [/llm-100/prompt-engineering/System_Prompt_User_Prompt_Assistant_Prompt_分别有什么作用.md](/llm-100/prompt-engineering/System_Prompt_User_Prompt_Assistant_Prompt_分别有什么作用.md)，了解系统级角色前缀如何对条件概率施加优先级更高的引导；也可以跳转至 [/frontier/context-engineering/what-is-context-eng.md](/frontier/context-engineering/what-is-context-eng.md) 了解系统级上下文组装的设计方法。
