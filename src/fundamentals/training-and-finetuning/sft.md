---
title: SFT 监督微调是什么？如何让基座模型学会听懂指令
icon: mdi:tune-vertical
tag:
- 训练与微调
- SFT
- 指令微调
description: 详解监督微调（SFT）的核心原理、对话模板与特殊Token、损失掩码实现、全量微调显存估算以及SFT的能力边界与工程适用场景。
---

刚完成海量语料预训练的基座模型（Base Model），本质上是一个统计概率文本补全器。如果你向它输入：

```text
请列出三种常见的数据结构：
```

它接下来的输出很可能是：

```text
4. 树
5. 图
这是一道经典的计算机考题，由张老师命题。
```

模型并没有故意捣乱，它只是依据训练数据中互联网文本的统计概率，认为这段话最可能出自某份考卷或练习题，因而继续向下接龙。基座模型缺乏对“提问-解答”这种交互契约的理解。

监督微调（Supervised Fine-Tuning，简称 SFT）就是让模型从“无差别续写器”转变为“能够按用户意图回答问题的助手”的关键步骤。本文将从训练目标、对话模板、损失掩码（Loss Masking）、全参数训练显存构成以及 SFT 的实际能力边界展开讲解。

![监督微调SFT在模型训练全流程中的核心定位与模式转变](/images/content-rebuild/899c1a90c9ffd2a9/cover.png)

---

## 01 SFT 的定位：从文本续写到遵循指令

在大模型完整训练流程中（可参考 [大模型训练流程是什么？](/fundamentals/training-and-finetuning/training-pipeline.md)），预训练让模型在大规模无标注文本上通过自回归方式预测下一个 Token（具体原理见 [预训练到底在学什么？](/fundamentals/training-and-finetuning/pretraining-deep.md)）：

$$\mathcal{L}_{\text{Pretrain}}(\theta) = - \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

在预训练阶段，模型吸收了丰富的语言结构与世界知识，但它并不知道人类希望它扮演什么角色。SFT 依然使用因果自回归交叉熵损失，但对训练数据的组织形式和监督范围做了明确限制：

1. **数据组织为成对范式**：不再使用散装网页抓取文本，而是统一加工成包含“系统指令、用户输入、模型回答”的结构化样本。
2. **引入明确的角色边界**：通过专用的特殊标记，明确划分哪里是系统设定、哪里是用户提问、哪里是助手的回答。
3. **仅对回答部分计算损失**：前向传播时输入完整上下文，但在反向传播计算梯度时，只对“助手生成的回答”部分计入损失，不让模型去拟合用户提示词的语法特征。

通过这一阶段，模型学会了在接收到用户指令后，以“助手”的角色和语气给出对应解答，并主动输出终止标记来结束对话。

---

## 02 对话数据模板与特殊 Token

计算机底层的 Transformer 接收的只是一维整数数组（Token ID）。为了让自回归模型在处理多轮交互时分清谁在说话，需要统一的文本包裹规则，即对话模板（Chat Template）。行业内常见的有 ChatML 等格式。

![ChatML对话模板结构与特殊Token角色边界切分](/images/content-rebuild/899c1a90c9ffd2a9/fig-1.png)

在分词器（Tokenizer）中，会预先注册一组保留的特殊 Token（Special Tokens），例如 `<|im_start|>` 和 `<|im_end|>`。一条标准的多轮或单轮指令在经过模板格式化后，文本如下：

```text
<|im_start|>system
你是一个严谨的Python编程助手。<|im_end|>
<|im_start|>user
如何反转一个列表？<|im_end|>
<|im_start|>assistant
在Python中可以使用切片[::-1]或reverse()方法。<|im_end|>
```

这些特殊 Token 具有专门的 Token ID，并在微调过程中更新其表征向量。它们的作用包括：

- **角色锚定**：明确区分 System、User 和 Assistant，避免模型发生角色混淆，例如自问自答或模仿用户发言。
- **生成截断**：助手回答完毕后会输出 `<|im_end|>`。推理引擎检测到这个结束标记（EOS），就会停止采样输出，避免模型无休止续写无意义内容。
- **防止结构混淆**：正常用户输入的特殊文本会被 Tokenizer 转义或拆解，不会轻易触发系统控制 Token，为应用层交互建立边界。

<ReadMoreLock />

---

## 03 损失掩码（Loss Masking）：为什么只对 Assistant 计损

在因果自回归训练中，模型会对整个序列的前向状态进行计算。但在 SFT 数据集中，用户输入和系统提示词属于“给定的条件”，而不是“模型需要学会预测的目标”。

![Loss Masking机制对用户输入掩码与助手回复计损对比](/images/content-rebuild/899c1a90c9ffd2a9/fig-2.png)

### 为什么不对 Prompt 计算损失？

假设我们对用户的提问“如何反转一个列表？”也计算交叉熵损失并回传梯度，就会带来两个负面效果：

1. **目标偏离**：用户的提问千变万化，可以写成“Python list 反转”、“怎么把数组倒过来排”、“如何反转一个列表”。SFT 的目标是让模型在收到任意提问时输出高质量答案，而不是去死记硬背用户习惯用哪种句式提问。对 Prompt 计算损失会浪费参数更新容量去过拟合提问句式。
2. **长上下文拟合偏差**：在多轮对话中，若模型对前面轮次的用户文本也施加梯度更新，会干扰模型根据历史上下文生成当前轮次回答的条件概率建模。

### PyTorch 中的实现方式：-100

在 PyTorch 的交叉熵损失函数 `torch.nn.CrossEntropyLoss` 中，默认包含参数 `ignore_index=-100`。任何在标签张量（Labels）中数值为 `-100` 的位置，算子在计算损失分母与分子时都会直接跳过，既不产生损失值，也不产生梯度。

因此，在 SFT 数据构建中：
- **`input_ids`（模型输入）**：保留整条序列的所有 Token（包括 System、User、Assistant 以及全部特殊标记）。
- **`labels`（目标标签）**：将 System Prompt、User Query 以及对应的边界标记位置全部填为 `-100`；仅将 Assistant 回答的 Token 以及结尾的 `<|im_end|>` 保留为真实 Token ID。

---

## 04 原生 PyTorch 掩码损失计算实战

自回归语言建模中有一个关键机制：**因果位移（Causal Shift）**。在时间步 $t$ 生成的隐藏层状态和 Logits，预测的目标是时间步 $t+1$ 的 Token。因此在计算交叉熵之前，必须将 Logits 和 Labels 错开一位进行对齐。

![自回归移位输入与目标标签对齐的损失计算流程](/images/content-rebuild/899c1a90c9ffd2a9/fig-3.png)

以下是一段最小可复现代码，演示如何对标签设置掩码并计算因果位移损失：

```python
import torch
import torch.nn as nn

# 设定模拟参数：词表大小 1000，序列长度 8，批大小 1
vocab_size = 1000
batch_size = 1
seq_len = 8

# 假设序列 token 分布：
# [0, 1] 为用户输入; [2] 为分隔符; [3, 4, 5] 为助手回答; [6] 为结束符 EOS; [7] 为填充符 PAD
input_ids = torch.tensor([[101, 205, 32, 401, 502, 603, 999, 0]])

# 构造 Labels：非助手回答部分全部设为 -100
# 索引 3, 4, 5（回答正文）和 6（结束符）保留实际 Token ID 用于计损
labels = torch.tensor([[-100, -100, -100, 401, 502, 603, 999, -100]])

# 模拟模型最后一层线性输出 Logits: (batch_size, seq_len, vocab_size)
torch.manual_seed(42)
logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

# 因果语言建模的 Shift 操作：
# 时间步 t 的预测结果 shift_logits[..., t, :] 对应时间步 t+1 的目标 shift_labels[..., t]
shift_logits = logits[..., :-1, :].contiguous()  # 去掉最后一个时间步的预测
shift_labels = labels[..., 1:].contiguous()       # 去掉第一个输入，作为实际预测目标

# 使用 ignore_index=-100 忽略非目标区域
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
loss = loss_fn(
    shift_logits.view(-1, vocab_size),
    shift_labels.view(-1)
)

print(f"有效计损 Token 数量: {(shift_labels != -100).sum().item()}")
print(f"计算所得损失值: {loss.item():.4f}")

# 反向传播验证掩码效果
loss.backward()
print(f"用户输入位置对应的 Logits 梯度范数（被掩码，应为0）: "
      f"{logits.grad[0, 0].abs().sum().item():.4f}")
print(f"助手回答位置对应的 Logits 梯度范数（有效计损，大于0）: "
      f"{logits.grad[0, 3].abs().sum().item():.4f}")
```

在计算过程中，虽然输入序列完整进入模型进行前向注意力计算，但梯度只在未被掩码的回答位置产生并向后传播。

---

## 05 全量 SFT 的显存开销拆解

在做微调工程规划时，模型训练的显存开销远大于推理显存。以使用 **标准 AdamW 优化器 + 混合精度（BF16/FP16）** 进行全参数微调为例，参数量记为 $\Phi$（以 7B 即 $7 \times 10^9$ 参数为例）：

![全量微调7B模型训练显存占用四大构成剖析](/images/content-rebuild/899c1a90c9ffd2a9/fig-4.png)

### 1. 静态显存（与批大小、序列长度无关）

- **模型参数（Weights）**：采用 16 位浮点数加载，每个参数占 2 字节（Bytes）。
  $$\text{Mem}_{\text{weight}} = 2 \times 7\text{B} \approx 14\text{ GB}$$
- **反向传播梯度（Gradients）**：与参数精度相同，每个参数的梯度占 2 字节。
  $$\text{Mem}_{\text{grad}} = 2 \times 7\text{B} \approx 14\text{ GB}$$
- **AdamW 优化器状态（Optimizer States）**：
  为了保证小梯度累积的数值稳定性，AdamW 需要使用 32 位浮点数（FP32）维护以下数据：
  - FP32 主权重副本：4 字节/参数
  - FP32 一阶动量（First Momentum）：4 字节/参数
  - FP32 二阶动量（Second Momentum / Variance）：4 字节/参数
  
  优化器状态每个参数合计占用 $4 + 4 + 4 = 12$ 字节：
  $$\text{Mem}_{\text{optimizer}} = 12 \times 7\text{B} \approx 84\text{ GB}$$

**静态显存小计**：$2 + 2 + 12 = 16$ 字节/参数。针对 7B 模型，静态开销约为：
$$7 \times 10^9 \times 16 \text{ 字节} \approx 112\text{ GB}$$

这意味着即使不考虑前向传播的激活值，单张 80GB 的显卡也无法容纳 7B 模型的全参数 AdamW 状态。工程上通常需要采用 ZeRO（零冗余优化器）分片、参数高效微调（如 LoRA，详见 [LoRA 和 QLoRA 是什么？](/fundamentals/training-and-finetuning/lora-qlora.md)）或多卡并行来解决。

### 2. 动态显存（与批大小、序列长度强相关）

动态显存主要由前向传播中缓存的**激活值（Activations）**构成，用于反向传播时根据链式法则计算梯度。如果上下文长度从 2048 扩展到 8192，激活值显存会随序列长度显著上升。开启激活检查点（Activation Checkpointing / Gradient Checkpointing）后，模型在前向阶段只保存少量检查点，反向传播时重新计算各层激活，能够将激活值显存大幅降低，代价是增加约 30% 到 35% 的计算时间。

---

## 06 SFT 的能力边界：学行为模式，而非充当知识库

在业务实践中，经常有团队尝试通过 SFT 将最新的公司手册、技术文档写进模型参数，希望模型能准确回答所有内部细节。这种做法往往收效甚微，甚至引发新的问题。

![将SFT用于事实知识写入导致的幻觉与知识冲突分析](/images/content-rebuild/899c1a90c9ffd2a9/fig-5.png)

### SFT 适合做的事

1. **格式规范与输出约束**：例如训练模型严格输出符合 JSON Schema 规范的文本，或者在遇到特定场景时调用固定格式的工具接口。
2. **交互风格与语气调优**：将模型的表达风格调整为严谨、活泼、客服口吻等。
3. **特定任务模式学习**：例如专业领域的实体抽取、文本分类、特定报告摘要等结构化任务。
4. **安全与边界对齐**：学会对不合理或超出能力范围的提问进行合规拒绝。

### 为什么 SFT 不适合作为知识存储手段？

1. **参数化记忆容易产生幻觉**：神经网络的权重是分布式表征，微调修改的是网络各层连接的统计关联。通过少量微调样本很难强行写入高精度的离散事实。当遇到相似提问时，模型容易将新旧知识的参数混淆，输出看似通顺但数字、主体完全错误的幻觉内容。
2. **不可溯源与不可验证**：业务场景需要可查证的依据。SFT 后的模型无法直接给出其结论源自哪一份文档的哪一章节，一旦出错很难排查因果。
3. **更新成本与灾难性遗忘**：当企业业务规则发生变更时，重新微调需要重新准备数据并承担训练算力成本；频繁针对新数据微调还容易导致模型在原先擅长任务上的能力退化（灾难性遗忘）。
4. **无法精确定点删除**：如果涉及隐私或法规要求下架某项数据，微调模型无法像数据库一样执行精确的“定点删除”。

对于需要精准事实调取和频繁更新的业务知识，更合理的基础架构是检索增强生成（RAG），具体选型权衡可参考 [Prompt、RAG 和微调怎么选？](/fundamentals/training-and-finetuning/prompt-rag-finetuning.md)。

---

## 07 总结与训练流水线演进

监督微调在模型生命周期中起到承上启下的作用：

- **承接预训练**：利用预训练获得的语言理解与模式推导能力，将其引导并约束到指令响应的形式上；
- **开启对齐与偏好优化**：经过 SFT 后的模型具备了良好的指令遵循能力，为后续的直接偏好优化（DPO）或基于人类反馈的强化学习（RLHF，详见 [RLHF 和 DPO 有什么区别？](/fundamentals/training-and-finetuning/rlhf-dpo.md)）提供了符合人类表达规范的基准策略分布。

![大模型监督微调在训练流水线中的前后承接与落地选型](/images/content-rebuild/899c1a90c9ffd2a9/fig-6.png)

关于微调样本的具体构造规范与质量把控，可继续阅读 [微调数据怎么构造？](/fundamentals/training-and-finetuning/finetuning-data.md)；在训练过程中遇到显存不足或损失异常时，可参考 [微调常见问题有哪些？](/fundamentals/training-and-finetuning/finetuning-pitfalls.md)。
