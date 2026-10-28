---
title: "Masked Language Model 和 Causal Language Model 有什么区别？"
icon: mdi:tune-vertical
tag:
  - 训练原理
  - 大模型100讲
  - 预训练
---

# Masked Language Model 和 Causal Language Model 有什么区别？

> 一句话核心：MLM（遮蔽语言模型）双向预测缺失词，适合理解任务；CLM（因果语言模型）单向预测下一个词，适合生成任务。两者是预训练语言模型的两种核心范式，分别对应 BERT（编码器）和 GPT（解码器）架构。

## 核心概念（术语表）
- **Masked Language Model (MLM)**：用 [MASK] 替换输入中 15% 的词，双向预测缺失词。用途：预训练理解型模型，如 BERT、RoBERTa。
- **Causal Language Model (CLM)**：只基于前文预测下一个词，单向自回归生成。用途：文本生成任务，如 GPT 系列、LLaMA。
- **Self-Attention（自注意力）**：计算序列内每对词之间的依赖关系。MLM 用双向注意力，CLM 用因果（单向）注意力。
- **Positional Encoding（位置编码）**：为每个 token 添加位置信息，使模型理解词序。BERT 和 GPT 都使用，但 MLM 用双向上下文，CLM 只用前文。
- **Bidirectional Encoder（双向编码器）**：MLM 特有的架构，能同时看到 [MASK] 前后的词。代表：BERT 的 Transformer Encoder。
- **Causal Decoder（因果解码器）**：CLM 特有的解码器结构，预测下一个词时只能看到之前的词。代表：GPT 的 Transformer Decoder。
- **Next-Word Prediction（下一个词预测）**：CLM 的训练目标，基于前文预测下一个 token。
- **Tokenization（分词）**：将文本拆分为 token（词、子词或字符）。两种模型都需此步骤。


<ReadMoreLock />## 历史背景 / 来源
- **MLM 起源**：2018 年 Google 的 Devlin 等人提出 BERT，同时引入 MLM 作为预训练目标。论文《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》。
- **CLM 起源**：GPT-1 于 2018 年由 OpenAI 提出，采用因果语言模型训练。GPT-2（2019）、GPT-3（2020）延续此范式。
- **核心突破**：BERT 在 GLUE 基准上刷新 11 项记录，证明 MLM 预训练+微调范式优于传统方法。GPT 系列则展示了 scaling law 下的涌现能力。

## 工作原理 / 核心机制

### 整体思路（一句话）
MLM 通过双向上下文预测被遮蔽的词，理解语义；CLM 通过自回归方式基于前文生成下一个词，完成文本生成。

### MLM 工作流程

**第一步：数据预处理与 Mask 操作**
- 输入：原始文本 "The weather is sunny today"
- 处理：随机替换 15% 的词为 [MASK] token（WordPiece 分词后）
- 输出："The weather is [MASK] today"（假设 sunny 被 mask）

**第二步：双向嵌入生成**
- 输入：每个 token 生成词嵌入 + 位置编码
- 处理：Transformer Encoder 计算双向自注意力，每个 [MASK] 位置的表示融合了前后文信息
- 输出：每个 [MASK] 位置得到上下文相关的向量表示

**第三步：概率分布预测**
- 输入：[MASK] 位置的向量表示
- 处理：全连接层 + Softmax，输出词表上每个词的概率
- 输出：P(sunny|[MASK]) = 0.92，P(cloudy) = 0.05，P(其他) = 0.03

### CLM 工作流程

**第一步：Tokenization（分词）**
- 输入：原始文本 "The weather is"
- 处理：分词为 ["The", "weather", "is"]
- 输出：token IDs [464, 3290, 318]

**第二步：因果注意力计算**
- 输入：所有 token 的嵌入
- 处理：因果掩码（causal mask）确保每个位置只能看到之前的词
- 输出：每个位置的表示只包含前文信息

**第三步：Next-Word Prediction（下一个词预测）**
- 输入："The weather is"
- 处理：预测下一个词的概率分布
- 输出：P(sunny|[The, weather, is]) = 0.85
- 迭代：将预测结果加入输入，重复生成下一个词
- 最终输出："The weather is sunny today"

## 关键知识点

1. **MLM 通常 mask 15% 的词**：BERT 论文设定，mask 比例过高会丢失太多上下文，过低则训练信号不足。
2. **CLM 是自回归生成**：每个词的生成依赖之前所有词，训练和推理时计算方式一致。
3. **BERT 是 Encoder-only**：双向架构，能同时看到 [MASK] 左右两侧的词。
4. **GPT 是 Decoder-only**：单向架构，训练时用 teacher forcing，推理时自回归生成。
5. **MLM 支持双向推理**：给定完整句子填缺词，适合阅读理解、文本分类等理解任务。
6. **CLM 支持文本生成**：从起始 prompt 一直生成到 EOS token，适合对话、写作、代码生成。
7. **位置编码的差异**：MLM 和 CLM 都用位置编码，但 CLM 的因果掩码限制注意力范围。
8. **BERT 预训练两个任务**：MLM + Next Sentence Prediction (NSP)。GPT 只做 CLM。
9. **RoBERTa 改进了 MLM**：去掉 NSP，动态 mask，更大数据量和训练时间，性能超越 BERT。
10. **GPT 系列 scaling**：GPT-3 1750 亿参数，in-context learning 能力涌现。
11. **LLaMA 的改进**：Meta 开源的 LLaMA，用更少参数达到 GPT-3 水平，推动开源社区发展。
12. **推理效率对比**：CLM 自回归生成 O(n) 步，MLM 只需 O(1) 步（前向传播）。
13. **训练数据利用效率**：MLM 每个 token 都参与损失计算，CLM 只计算下一个词。
14. **注意力复杂度**：标准自注意力 O(n²)，FlashAttention 等优化将复杂度降至 O(n)。
15. **两者的互补性**：BERT（MLM）擅长理解，GPT（CLM）擅长生成，实际系统常组合使用。

## 应用场景

- **场景 1**：Google 搜索用 BERT 理解查询意图，MLM 预训练的 BERT 替换了旧排序模型，提升了 10% 的搜索相关性准确率。
- **场景 2**：OpenAI ChatGPT 用 GPT-3.5/4（CLM）实现多轮对话，1750 亿参数模型生成连贯长文本，支持代码编写、数学推理。
- **场景 3**：GitHub Copilot 用 CLM 代码生成，在 VS Code 中实时补全代码，开发者效率提升约 55%（微软内部数据）。
- **场景 4**：Facebook 翻译系统用 BERT（MLM）做 encoder，提升翻译质量，尤其在低资源语言对上表现突出。
- **场景 5**：智能客服用 CLM 生成回复，MLM 做意图分类，两者协同：MLM 理解用户问题类型，CLM 生成具体回复。

## 常见误区 / 踩坑

- ❌ **误区 1**：认为 MLM 生成的文本比 CLM 更准确
  ✅ **正解**：MLM 不擅长生成，它的设计目标是"填词"而非"造句"。生成任务必须用 CLM。

- ❌ **误区 2**：认为 CLM 训练比 MLM 简单
  ✅ **正解**：CLM 需要自回归生成，训练时梯度不稳定（teacher forcing vs 自回归），推理时错误累积问题。

- ❌ **误区 3**：认为 BERT 不能做生成
  ✅ **正解**：BERT 可以通过 mask-predict 策略做生成（如 BERT4Seq2Seq），但效率低于 CLM，不适合大规模生成。

- ❌ **误区 4**：混淆 mask 比例
  ✅ **正解**：BERT 默认 15% mask（其中 80% 替换为 [MASK]，10% 随机替换，10% 不变）。这个比例经过消融实验确定。

- ❌ **误区 5**：认为双向注意力一定优于单向
  ✅ **正解**：对于生成任务，双向注意力会导致"信息泄露"（看到未来词），必须用因果掩码。

- ❌ **误区 6**：忽视推理效率差异
  ✅ **正解**：MLM 对整个句子只需一次前向传播，CLM 生成 n 个词需要 n 次前向传播，长文本生成成本高。

## 性能 / 复杂度

- **时间复杂度**：
  - MLM：O(n × d × h)，n=序列长度，d=层数，h=隐藏维度。一次前向传播。
  - CLM 推理：O(n² × d × h)（生成 n 个词，每次 O(n) 注意力）。
  - 适用边界：生成超过 50 个词时，CLM 推理时间显著长于 MLM。

- **空间复杂度**：
  - MLM：O(n × h) 保存激活值。
  - CLM：额外需要保存 causal mask 和 past key-value 缓存。
  - 适用边界：长序列（n>2048）时，CLM 显存需求约为 MLM 的 2-3 倍。

- **与替代方案对比**：
  - 方案 A：传统 RNN/LSTM，O(n × h) 时间，无法捕捉长距离依赖（梯度消失）。
  - 方案 B：Transformer Encoder (MLM)，O(n²) 注意力，全局感受野，适合理解。
  - 方案 C：Transformer Decoder (CLM)，O(n²) 注意力 + O(n) 步生成，适合生成。
  - 临界点：任务类型决定选择，理解任务选 MLM，生成任务选 CLM。

- **性能数字**：
  - BERT-base：1.1 亿参数，GLUE 基准 82.1 分。
  - GPT-3：1750 亿参数，in-context learning 在 100+ 任务上达到 SOTA。
  - LLaMA-7B：用 7B 参数达到 GPT-3 水平，训练数据 1T tokens。

## 与相关概念的区别

- **vs LSTM/RNN**：
  - 维度 1（并行化）：LSTM 训练慢（时序展开），Transformer 可并行。
  - 维度 2（长距离依赖）：LSTM 梯度消失，Transformer 全局注意力。
  - 维度 3（适用）：LSTM 适合小数据，Transformer 适合大数据。
  - 怎么选：小数据集（<10K）选 LSTM，大规模预训练选 Transformer。

- **vs Seq2Seq（Encoder-Decoder）**：
  - 维度 1（架构）：Seq2Seq 有 encoder 和 decoder 分离，MLM/CLM 只有 encoder 或 decoder。
  - 维度 2（任务）：Seq2Seq 适合翻译（输入→输出），MLM/CLM 是预训练范式。
  - 维度 3（代表模型）：T5 是 Encoder-Decoder 架构，可做 MLM 也可做 CLM-style 生成。
  - 怎么选：翻译任务选 Seq2Seq，理解任务选 MLM，生成任务选 CLM。

- **vs Prefix LM（前缀语言模型）**：
  - 维度 1（注意力）：Prefix LM 对 prefix 部分用双向注意力，生成部分用因果注意力。
  - 维度 2（灵活性）：Prefix LM 比 CLM 更灵活，可处理 prefix 理解+后续生成。
  - 维度 3（代表）：UniLM、GLM 采用此架构。
  - 怎么选：需要 prefix 理解的生成任务选 Prefix LM，纯生成选 CLM。

## 进阶 / 面试加分项

- **最新进展**：ChatGPT 背后的 RLHF（人类反馈强化学习）将 CLM 与强化学习结合，通过 reward model 优化生成质量。LLaMA 2 进一步引入 RLHF，在有用性和安全性上显著提升。
- **业界争议/未解问题**：
  1. CLM 的 scaling law 是否会见顶？GPT-4 的架构细节未公开，社区猜测可能用了 mixture of experts (MoE)。
  2. MLM 和 CLM 的统一：UL2、GLM 等模型试图用一个模型同时做 MLM 和 CLM，但效果尚未全面超越专用模型。
  3. 效率优化：Linear Attention、State Space Models（如 Mamba）尝试将 O(n²) 复杂度降至 O(n)，可能颠覆现有架构。
- **一句话送给候选人**：MLM 学"读"，CLM 学"写"，理解两者差异就像理解语文的阅读理解和作文——面试中考的是你能否根据任务选对模型，而不是死记硬背架构细节。

## 附录：代表性模型一览

| 模型 | 架构 | 预训练目标 | 参数规模 | 推出时间 | 推出机构 |
|------|------|------------|----------|----------|----------|
| BERT | Encoder-only | MLM + NSP | 340M (base) / 1.1B (large) | 2018 | Google |
| GPT-1 | Decoder-only | CLM | 117M | 2018 | OpenAI |
| GPT-2 | Decoder-only | CLM | 1.5B | 2019 | OpenAI |
| GPT-3 | Decoder-only | CLM | 175B | 2020 | OpenAI |
| RoBERTa | Encoder-only | MLM (动态) | 355M | 2019 | Facebook |
| LLaMA | Decoder-only | CLM | 7B-65B | 2023 | Meta |
| LLaMA 2 | Decoder-only | CLM + RLHF | 7B-70B | 2023 | Meta |

## 核心对比速查表

| 维度 | MLM (BERT) | CLM (GPT) |
|------|------------|-----------|
| 注意力方向 | 双向 | 单向（因果） |
| 架构 | Encoder | Decoder |
| 训练目标 | 预测 [MASK] | 预测下一个词 |
| 适合任务 | 理解（分类、NER、QA） | 生成（对话、写作、代码） |
| 推理次数 | 1 次前向传播 | n 次前向传播 |
| 代表模型 | BERT, RoBERTa, ELECTRA | GPT, LLaMA, PaLM |
| 参数效率 | 较高（理解任务） | 较低（生成需要更多参数） |
| 上下文利用 | 全部可见 | 仅前文可见 |


## 面试如何回答


### 🟢 请用一句话解释 Masked Language Model 和 Causal Language Model 的核心区别

**回答要点**：

MLM（遮蔽语言模型）用双向注意力预测句子中被 [MASK] 掉的词，适合理解任务；CLM（因果语言模型）用单向自回归方式基于前文预测下一个词，适合生成任务。两者核心区别在于：MLM "读" 完整句子填缺词，CLM "写" 从头造句。具体来说，BERT 用 MLM 预训练，预测被 mask 的 "sunny" 时能看到 "The weather is [MASK] today" 的完整上下文；而 GPT 用 CLM 预训练，预测 "sunny" 时只能看到 "The weather is"，看不到后面的 "today"。这种差异决定了 BERT 适合文本分类、问答等理解任务，GPT 适合对话、写作等生成任务。面试官想听到的是你能根据任务类型选择合适的模型，而不是死记硬背架构。


### 🟡 为什么 Causal Language Model 不能看到未来的词？具体是怎么实现的？

**回答要点**：

CLM 不能看未来词是为了避免"信息泄露"，确保训练和推理行为一致。实现方式是在自注意力层加入 causal mask（因果掩码）。具体来说，对于位置 i 的词，注意力分数只计算与位置 ≤ i 的词的内积，位置 > i 的词被设为负无穷，经过 Softmax 后概率为 0。例如预测 "sunny" 时（第 4 个位置），注意力只能聚焦在 "The"、"weather"、"is" 三个词上，看不到 "today"。这种设计还带来另一个好处：训练时可以用 teacher forcing 加速收敛（直接拿真实前文预测），推理时则自回归逐步生成。面试官追问实现细节时，可以画出 causal mask 的矩阵图：下三角为 0，上三角为 -∞。这是 GPT、BART decoder、LLaMA 等生成模型的标配。


### 🟡 MLM 的 15% mask 比例是怎么确定的？BERT 还用了哪些训练技巧？

**回答要点**：

BERT 默认 mask 15% 的词，这个比例是经过消融实验确定的：比例过低（如 5%）训练信号太弱，模型学不到足够的上下文表示；比例过高（如 50%）则丢失太多原始句子信息，预测难度过大。BERT 还用了两个训练技巧：1）Next Sentence Prediction (NSP)：额外增加一个二分类任务，让模型学习句子间关系，帮助问答、自然语言推理等任务。2）动态 Mask：RoBERTa 发现每次 epoch 用不同的 mask 位置比固定 mask 效果更好，所以去掉了 NSP，增加了训练数据量和步数，最终在多项基准上超越 BERT。实际面试中，能说出 15% 这个数字和背后的 trade-off 已经足够，如果能补充 RoBERTa 的改进则体现你对前沿工作的关注。


### 🟡 一个客服机器人需要同时做用户意图分类和生成回复，应该如何设计模型架构？

**回答要点**：

最佳方案是 MLM+CLM 组合使用。意图分类是理解任务，适合用 MLM（如 BERT）；回复生成是生成任务，适合用 CLM（如 GPT）。具体架构设计：第一步，用 BERT encoder 编码用户输入，通过 [CLS] token 的表示做意图分类（多分类softmax），这一步只需一次前向传播。第二步，将意图标签和用户输入拼接作为 prompt，用 GPT decoder 自回归生成回复。这样做的好处是：两个任务解耦，模型独立优化，意图分类准确率高（BERT 在此任务上通常 >95%），生成回复质量也好（GPT 4B 参数通常优于 1B）。如果资源有限，可以用 prefix LM（如 UniLM）在单一模型中兼顾两种能力，但效果通常不如组合。面试官想看到的是你理解任务类型决定模型选择，而不是盲目追求大模型。


### 🔴 CLM 生成 100 个词需要多少次前向传播？为什么说它的推理效率低？

**回答要点**：

CLM 生成 100 个词需要 100 次前向传播，因为每个词都是自回归生成的：先生成第 1 个词，再用前 2 个词预测第 2 个词，以此类推。相比之下，MLM 对整个句子只需 1 次前向传播就能预测所有 [MASK] 位置。这就是为什么 BERT 推理快（O(1) 步），GPT 推理慢（O(n) 步）。长文本生成时这个差距非常显著，例如生成 1000 词的文章，CLM 需要 1000 次前向传播，MLM 只需 1 次。业界常用 KV cache 优化：保存已计算的 key-value 矩阵，避免重复计算，但仍需 n 次解码步骤。另一个优化方向是 speculative decoding：用小模型快速生成 draft，大模型验证，加速 2-3 倍。面试时能说出具体的计算次数和优化方向，体现你对工程实践的理解。


### 🟡 MLM 和 CLM 在预训练数据利用效率上有什么区别？哪个更好？

**回答要点**：

MLM 训练时所有 token 都参与损失计算，每个被 mask 的词都有监督信号；CLM 训练时只有下一个词参与损失，前面的词没有直接的监督。粗略估计，MLM 的数据利用效率是 CLM 的 2-3 倍（取决于序列长度）。但这不是绝对的：CLM 虽然只预测下一个词，但每个词都会作为下一个词的输入进行训练，实际上所有词的表示都在优化。BERT 的 MLM 还有一个 80/10/10 的技巧：80% 替换为 [MASK]，10% 随机替换，10% 保持不变，这增加了训练难度但避免了预训练和微调的不一致。实际选择要看任务：理解任务用 MLM 数据效率高，生成任务只能用 CLM。面试官追问时，可以补充 ELECTRA 的替换 token 检测 (Replaced Token Detection) 比 MLM 更数据高效。


### 🟡 Prefix LM 和 Causal LM 有什么区别？什么时候应该用 Prefix LM？

**回答要点**：

Prefix LM（前缀语言模型）的注意力机制是混合的：对用户输入的 prefix 部分用双向注意力（像 MLM），对模型生成的 part 用因果注意力（像 CLM）。典型代表是 UniLM、GLM。举例来说，用户输入 "请帮我写一封请假邮件，内容是"，这部分用双向注意力充分理解；模型生成 "尊敬的领导：..." 时用因果注意力避免信息泄露。这种设计的好处是：既能在 prefix 上充分利用上下文，又能在生成时保持一致性。适用场景包括：需要理解复杂指令的生成任务（如 ChatGPT 的 system prompt）、多轮对话（前面的对话历史用双向注意力，当前回复用因果注意力）。如果纯生成任务且 prefix 简单，直接用 CLM 更高效。面试时能说出前缀注意力和因果注意力的区别，已经超出基础水平。


### 🔴 如果让你设计一个能同时做理解和生成的模型，你会怎么做？有哪些最新进展？

**回答要点**：

有三种主流方案。第一种是 Encoder-Decoder 架构（T5、BART），encoder 用 MLM 理解输入，decoder 用 CLM 生成输出，适合 seq2seq 任务（翻译、摘要）。第二种是 Prefix LM（UniLM、GLM），单一模型中混合双向和单向注意力，prefix 部分双向，生成部分单向，更灵活。第三种是混合专家模型（MoE），如 GPT-4 猜测采用 MoE 架构，多个专家网络按需激活，兼顾容量和效率。最新的趋势是将 CLM 与 RLHF 结合（LLaMA 2），通过 reward model 和强化学习优化生成质量，在有用性和安全性上显著提升。面试中能讨论这些前沿方向，体现你对领域发展的持续关注。加分项可以提到 State Space Models（Mamba）等新架构，试图用 O(n) 复杂度颠覆 Transformer。

