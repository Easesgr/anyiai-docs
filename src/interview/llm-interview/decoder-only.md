---
title: 为什么主流大模型普遍采用 Decoder-only 架构？
icon: mdi:arrow-decision-outline
tag:
- 大模型面试
- Transformer
- 架构
description: 从预训练监督密度、自回归生成接口、KV Cache 推理调度与分布式并行四个维度，讲透主流大模型收敛至 Decoder-only 架构的工程权衡与适用边界。
---

![Decoder-only 架构选型深度解析封面图](/images/content-rebuild/76550d0fe81cd6b8/cover.png)

在以 GPT-4、LLaMA、Mistral 为代表的主流大语言模型中，底层架构几乎全部收敛到了**单向因果解码器（Causal Decoder-only）**。而在自然语言处理早期，**双向编码器（Encoder-only，如 BERT）**和**编码-解码器（Encoder-Decoder，如 T5）**曾占据半壁江山。

大模型团队最终选择 Decoder-only，并不是因为单向注意力在文本理解上优于双向注意力，而是该架构在**预训练数据利用率、自回归生成接口、推理缓存管理以及大规模并行计算**上达到了最佳的综合工程平衡。

--- 

## 30秒面试口述回答

> **面试官提问**：为什么现在的通用大语言模型几乎都采用 Decoder-only 架构，而不是 BERT 那样的 Encoder-only 或者 T5 那样的 Encoder-Decoder？

**口述核心回答模板**：

主流大模型采用 Decoder-only 架构，是**训练效率、推理机制与多任务接口**三者权衡的结果，核心原因有四点：

1. **预训练监督信号密度最高**：Decoder-only 使用单向因果掩码（Causal Mask），序列中每个位置都在预测下一个 Token。在相同训练步长与上下文长度下，除首字外的所有 Token 都直接参与计算交叉熵损失，算力转化为语言建模目标的密度接近 100%；相比之下，BERT 的 MLM 任务通常只有 15% 的 Token 产生预测损失，T5 则只有输出段产生自回归损失。
2. **生成任务与上下文学习（In-Context Learning）天然同构**：提示词、少样本示例和生成内容统一拼为单一序列，模型只需持续向后预测下一个 Token 即可完成问答、翻译或代码编写，不需要对输入与输出做割裂的接口区分。
3. **推理期 KV Cache 调度极简**：因果掩码保证已生成的历史 Token 不受未来生成内容的影响，历史 Key/Value 向量可以单向线性追加并静态复用；若引入双向注意力（如 Prefix LM），多轮对话中历史 Token 的上下文表示会因新输入而发生动态改变，导致前缀缓存失效。
4. **算子同构便于大规模分布式扩展**：模型从底到顶全是结构完全一致的因果层，张量并行与流水线并行的切分十分均匀，避免了 Encoder-Decoder 架构中因交叉注意力带来的额外跨卡通信与负载倾斜。

---

## 贯穿例子：一个生成任务在三大架构下的执行差异

为了直观理解这三种架构的差异，我们用一个统一的任务贯穿全文：输入提示词“今天天气真好，我们去”，模型需要预测生成“公园玩吧”。

```
Encoder-only (双向自注意力):        Decoder-only (单向因果自注意力):     Encoder-Decoder (分离解耦):
[今] <-> [天] <-> [气] <-> [好]    [今] -> [天] -> [气] -> [好]        Encoder: [今] <-> [天] <-> [气] <-> [好]
(每个 Token 均可看见全文)         (每个 Token 仅能看见它左侧的词)                       |
                                                                             v (交叉注意力 Cross-Attention)
                                                                        Decoder: [公] -> [园] -> [玩] -> [吧]
```

![三种架构模式的注意力掩码与信息流动拓扑对比](/images/content-rebuild/76550d0fe81cd6b8/fig-1.png)

*图 1 展示了三种架构在注意力矩阵掩码上的区别。Causal Decoder-only 采用严格的下三角掩码，右上角被完全遮蔽，从而锁定了从左到右单向传递的信息流。*

| 架构类型 | 代表模型 | 注意力连接方式 | 典型预训练任务 | 文本生成方式 | 核心短板 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Encoder-only** | BERT, RoBERTa | 全双向互见 | 掩码语言建模（MLM，随机挖空预测） | 无法自然增量解码；若强行自回归，每生成一个词必须全序列从头重算 | 缺乏自回归生成链路，难以直接胜任开放式长文本生成 |
| **Encoder-Decoder** | T5, BART | Encoder 双向 + Decoder 因果 + 交叉注意力 | 跨序列映射（Seq2Seq，文本到文本转换） | Encoder 一次性编码输入，Decoder 借助交叉注意力逐词自回归生成 | 架构切分为两套结构，Prompt 与输出边界硬编码，分布式通信与缓存维护较重 |
| **Causal Decoder-only** | GPT 系列, LLaMA | 严格单向因果（仅见历史与当前词） | 下一个 Token 预测（Next Token Prediction） | 统一拼为单序列，以自回归方式逐字追加生成 | 缺乏预先纵览全文的双向交互（需依赖模型深度与足够参数量建立长程关联） |

在上一篇 [Transformer 和 LLM 有哪些区别](/interview/llm-interview/transformer-vs-llm.md) 中我们讨论过注意力算子的演进，这种底层掩码设计的取舍，直接决定了模型在大规模扩展时的训练成本与硬件吞吐。

---

## 训练算力预算与监督信号密度推导

大模型预训练耗费数百万 GPU 时，训练预算受硬件浮点运算量（FLOPs）严格限制。根据预训练扩展定律的经验公式：

$$C \approx 6ND$$

其中 $C$ 为训练浮点运算量，$N$ 为非词表模型参数量，$D$ 为训练所消耗的 Token 总数。在固定浮点运算预算 $C$ 下，哪种架构能够从相同量级的数据中提取最多的语言建模监督信号？

<ReadMoreLock />

### 逐步算例：相同 Token 吞吐下的损失计算量

设输入序列长度 $L = 4096$，模型隐层维度为 $d$。每批次前向传播吞吐 4096 个 Token 时：

1. **Encoder-only（以 BERT 为例）**：
   - **训练逻辑**：随机将 15% 的 Token 替换为 `[MASK]` 或其他词，模型最终仅在这 15% 的掩码位置上计算交叉熵损失。
   - **损失计算 Token 数**：$4096 \times 15\% \approx 614$ 个 Token。
   - **计算特征**：其余 85% 的 Token 虽然完整参与了前向和反向传播的张量乘法，并为被遮蔽位置提供了上下文表示，但它们自身并不直接作为自回归生成的监督目标。
2. **Encoder-Decoder（以 T5 为例）**：
   - **训练逻辑**：前段为输入上下文（长度设为 $L_{in} = 2048$），后段为目标输出（长度设为 $L_{out} = 2048$）。
   - **损失计算 Token 数**：只有处于 Decoder 端的 2048 个 Token 会计算下一个词的预测损失。
   - **监督覆盖度**：$\frac{2048}{4096} = 50\%$。输入端的 2048 个 Token 仅充当只读条件，不产生自回归损失。
3. **Causal Decoder-only（以 LLaMA 为例）**：
   - **训练逻辑**：整个 4096 长度的序列无须人为切分输入输出，全序列使用下三角因果掩码。
   - **损失计算 Token 数**：位置 1 预测位置 2，位置 2 预测位置 3……直到位置 4095 预测位置 4096，共计算 $4096 - 1 = 4095$ 个损失。
   - **监督覆盖度**：$\frac{4095}{4096} \approx 100\%$。

![不同预训练架构在相同 Token 预算下的监督密度与有效梯度计算对比](/images/content-rebuild/76550d0fe81cd6b8/fig-2.png)

*图 2 展现了相同计算预算下不同模型产生直接预测损失的 Token 比例。Decoder-only 让序列中除首词外的每一个位置都直接服务于下一个词的预测。*

当预训练数据达到数万亿 Token 时，Decoder-only 让每一次矩阵乘法的前向计算结果几乎都能转化为具体的交叉熵损失项，单位算力的学习效率更直接。关于自回归损失函数的数学推导，可参考 [大模型预训练目标是什么？](/interview/llm-interview/pretraining-objective.md)。

---

## 推理与缓存：Prefix LM 与 Decoder-only 的工程取舍

在探索架构的过程中，研究界曾尝试过一种折中形态——**Prefix LM（前缀语言模型）**。其核心思路是：在 Prompt（输入前缀）部分开启双向全可见注意力，在生成（回答）部分则切换为单向因果注意力。

Prefix LM 试图融合 BERT 的全局理解与 GPT 的自回归生成能力，但在万亿级部署中，主流选型依然回到了纯粹的 Causal Decoder-only。核心原因在于**推理期 [KV Cache](/interview/llm-interview/kv-cache.md) 的冻结与追加机制**。

以下 Python 伪代码演示了两者的注意力掩码构建差异：

```python
import torch

def build_masks(seq_len: int = 4, prefix_len: int = 2):
    # 1. Causal Mask: 严格下三角矩阵
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    
    # 2. Prefix Mask: 前缀内部双向互见，后缀部分因果单向
    prefix_mask = torch.zeros(seq_len, seq_len)
    # 前缀相互可见（左上角全 1）
    prefix_mask[:prefix_len, :prefix_len] = 1.0
    # 后续生成部分遵循因果依赖，并可见全部前缀
    for i in range(prefix_len, seq_len):
        prefix_mask[i, :i+1] = 1.0
        
    return causal_mask, prefix_mask

causal_m, prefix_m = build_masks()
print("Causal Mask:\n", causal_m)
print("Prefix Mask:\n", prefix_m)
```

输出的掩码矩阵结构如下：

```text
Causal Mask:                  Prefix Mask:
[[1., 0., 0., 0.],            [[1., 1., 0., 0.],
 [1., 1., 0., 0.],             [1., 1., 0., 0.],
 [1., 1., 1., 0.],             [1., 1., 1., 0.],
 [1., 1., 1., 1.]]             [1., 1., 1., 1.]]
```

![Prefix LM 与 Full Causal LM 在多轮对话和增量解码时的 KV Cache 状态演变对比](/images/content-rebuild/76550d0fe81cd6b8/fig-3.png)

*图 3 对比了新 Token 接入或提示词变动时两者的缓存状态。因果掩码具有单向只读性，历史 Key/Value 缓存一旦计算便可保持冻结；双向掩码则会导致历史缓存与后续交互重新耦合。*

### KV Cache 调度复杂度分析

1. **前缀缓存（Prompt Cache）的多轮静态复用**：
   - 在 **Causal Decoder-only** 中，因果掩码阻断了未来信息对过去的渗透。在多轮对话中，第 1 轮生成的回答会被作为第 2 轮输入的一部分，由于所有历史位置的 Key 和 Value 向量是严格静态不变的，推理引擎可以直接复用上轮计算完毕的 KV Cache，只需计算新输入的 Token，极大降低了首字延迟（TTFT）。
   - 在 **Prefix LM** 中，若新一轮对话将上一轮的生成内容吸收为新前缀的一部分，根据其掩码规则，新前缀内部需要开启双向注意力。这意味着旧 Token 的注意力表征必须与新 Token 发生双向交互，原有的静态缓存必须被废弃并重新计算，破坏了前缀缓存的可复用性。
2. **Prefill 与 Decode 阶段的计算图一致性**：
   - Causal Decoder-only 在 Prefill（批量处理提示词）和 Decode（逐字自回归解码）两个阶段使用完全同构的下三角因果计算图。Decode 阶段仅仅是把 Query 矩阵的长度缩减为 1，然后单向查询并追加 KV Cache，调度逻辑高度统一，易于在连续批处理（Continuous Batching）中做显存块管理（如 PagedAttention）。
   - Prefix LM 的掩码形状取决于前缀与后缀的动态边界，计算图在不同阶段出现掩码逻辑切换，增加了调度和算子优化的工程复杂度。

关于 Prefix LM 演进轨迹的细节，可阅读 [chatGLM 和 GPT 在结构上有什么区别？](/interview/llm-interview/chatglm-vs-gpt.md)。

---

## 面试高频追问

### 追问 1：BERT 的双向表征能力理论上更充分，为什么不能直接用于自回归长文本生成？

**口述要点**：
- **计算复杂度恶化**：双向注意力的计算假设当前词能够“看到未来所有位置”。如果强行让 BERT 进行自回归逐词生成，由于没有单向因果掩码的隔离，第 $t$ 步生成新词后，所有前序 $t$ 个词的上下文表征全部会发生联动改变，导致**无法利用 KV Cache 缓存历史状态**。
  - **Decoder-only（借助 KV Cache）**：第 $t$ 步解码只需针对新来的 1 个 Token 计算 QKV，与历史已缓存的 $t$ 个 KV 向量做注意力点积，单步注意力开销为 $O(t)$，前馈网络（FFN）开销为常数 $O(1)$。生成 $N$ 个 Token 的累计 FFN 开销为 $O(N)$，累计注意力矩阵点积开销为 $\sum_{t=1}^N t = O(N^2)$。
  - **双向 Encoder 重算模式**：每生成 1 个新词，必须把整个序列重新输入模型做全量前向传播。第 $t$ 步的全注意力计算为 $O(t^2)$，FFN 为 $O(t)$。生成 $N$ 个 Token 的累计 FFN 开销激增至 $O(N^2)$，全注意力累加计算恶化至 $\sum_{t=1}^N t^2 = O(N^3)$，在长文本场景下延迟呈爆炸式增长。
- **条件依赖断裂**：如果不逐词自回归，而是非自回归地一次性预测多个 `[MASK]`，由于去掉了自回归联合概率链式分解 $P(y_1, y_2, ...) = \prod P(y_t | y_{<t})$，各位置之间假设条件独立，极易造成生成内容的语法不连贯和语义重复。

![BERT 迭代全序列重算生成与 Causal LM 单向增量解码开销对比](/images/content-rebuild/76550d0fe81cd6b8/fig-4.png)

*图 4 直观呈现了双向模型进行自回归生成时缺乏因果保护带来的计算代价。由于无法直接追加 KV Cache，全序列重算会导致计算量随生成长度出现阶数级暴增。*

### 追问 2：为什么 Encoder-Decoder 架构在百亿、千亿参数大模型上退出了通用基座的统治地位？

**口述要点**：
- **零样本与多任务接口的统一性**：Decoder-only 把多轮问答、少样本示例（Few-shot Examples）、代码指令、工具调用全部抽象为一段连贯的文本流。模型只需要学会“补全序列”，即可执行各种任务。而 Encoder-Decoder 必须在数据和接口层面明确切分“输入端”和“输出端”，当遇到多轮穿插补充说明或复杂的思维链（Chain of Thought）推理时，人工界定输入输出边界会变得别扭且缺乏通用性。
- **分布式集群中的跨卡通信开销**：
  - Encoder-Decoder 包含 Encoder 自注意力、Decoder 自注意力和两者之间的**交叉注意力（Cross-Attention）**三类核心计算。
  - 在大规模集群进行张量并行（Tensor Parallelism）与流水线并行（Pipeline Parallelism）切分时，Decoder 每一层都需要跨节点拉取 Encoder 最后一层输出的 Key 和 Value 矩阵。这增加了卡间通信量，并且往往导致 Encoder 节点与 Decoder 节点之间的计算负载不均衡（Bubble 时间变长）。Decoder-only 架构结构完全相同，流水线切分极为整齐。

![分布式部署中 Encoder-Decoder 异构组件通信与 Decoder-only 同构组件通信对比](/images/content-rebuild/76550d0fe81cd6b8/fig-5.png)

*图 5 展现了在多卡分布式集群中两者的拓扑差异。Decoder-only 算子一致、数据单向流转，在流水线编排和张量切分上具备更平滑的通信与负载表现。*

---

## 架构适用边界与常见认知误区

尽管 Decoder-only 统治了通用生成基座，但并不意味着它在所有自然语言处理任务上都是绝对最优解。工程选型始终需要结合输入输出的物理特征来决定。

![架构选型决策矩阵与场景失效边界](/images/content-rebuild/76550d0fe81cd6b8/fig-6.png)

*图 6 总结了主流架构的分流决策路径。对于检索向量与固定模态映射任务，专用架构往往比单向生成模型更紧凑、更高效。*

### 架构选择决策清单

1. **文本向量检索与语义匹配（Embeddings & Reranker）**：
   - **首选**：Encoder-only（如 BERT、RoBERTa、ModernBERT）。
   - **原因**：这类任务的核心诉求是一次性产出代表全局语义的向量，不需要逐词生成文本。双向注意力能让句首和句尾的所有词互相交互，产出的句向量表征密度显著优于单向因果模型；若用 Decoder-only 提取句向量，排在前面的词无法融入后文信息，往往需要借助复杂的微调或均值池化手段进行补偿。
2. **固定模态与固定输入输出转换（如语音识别、强约束机器翻译）**：
   - **典型选用**：Encoder-Decoder（如语音转文字模型 Whisper、机器翻译模型 NLLB）。
   - **原因**：在这类端到端任务中，输入（如长音频频谱特征）与输出（目标语言文本）的数据边界非常固定，且不需要动态注入复杂的思维链提示词。Encoder 负责充分抽取长源信号的双向上下文，Decoder 专注精准转录，在受限参数量下具备极高的转换精度。
3. **通用指令遵循、代码生成、对话助手与 Agent 推理**：
   - **首选**：Causal Decoder-only。
   - **原因**：输入输出长度极不固定，高度依赖上下文学习与自回归推导，需要最大限度复用前缀缓存以支持高并发低延迟调度。

### 必须规避的常见认知误区

- **误区一：Decoder-only 能胜出，是因为它的语言理解能力在数学上超越了双向 Encoder。**
  - **勘误**：在**相同参数量、相同训练步数**下，针对分类、判别和双向语义蕴含任务，双向全交互（BERT）提取的特征质量往往优于缺乏未来视野的单向 Decoder。Decoder-only 的胜出，是由于其训练监督信号密度更高、生成任务算力利用率更强，并在庞大的参数量和数据量（Scaling Law）加持下，涌现出了超越小尺寸双向模型的长程泛化能力。
- **误区二：推理时只要启用了 KV Cache，所有架构的延迟和显存开销都差不多。**
  - **勘误**：Encoder-Decoder 在解码时需要常驻两组不同的 KV 缓存：一组是 Decoder 自注意力随着生成长度递增的动态缓存，另一组是 Encoder 输出的全量静态缓存。在显存带宽受限（Memory-Bound）的增量解码阶段，同时访存这两组张量并计算交叉注意力，其内存占用与访存带宽开销明显高于 Decoder-only 的单一直线追加缓存。

下一篇我们将进入模型输入的底层离散单元：[Token 是什么？如何计算和控制 Token 数量？](/interview/llm-interview/token-basics.md)。
