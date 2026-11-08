---
title: 大模型如何训练？从单步预测到分布式集群的工程链路
icon: mdi:tune-vertical
description: 系统梳理大语言模型的训练技术栈，解析因果建模、损失计算、显存分布与分布式扩展，提供可运行的最小训练与困惑度评估示例。
---

大语言模型的训练是一个高度工程化的过程。从数学目标上看，它通过自监督学习在大规模文本语料上建模词与词之间的条件概率分布；从系统架构上看，它是在有限显存、计算吞吐和互联带宽的物理约束下，将海量参数高效扩展到多机多卡的分布式工程。

本模块作为 [Anyi AI](/README.md) 进阶课程的衔接内容，上接 Transformer 结构与注意力机制，下接指令对齐与下游 [AI 工程架构](/architecture/README.md) 落地。我们将用一个具体的句子预测过程串联起从前向损失计算到集群切分的完整链路。

![大模型训练知识体系与全栈承接路径图](/images/content-rebuild/c58e5e07a16483c1/cover.png)

---

## 一个贯穿例子：模型如何预测下一个词

假设输入句子为：`"大 语言 模型 能够 预测 下一个 词"`。

分词后，模型接收前缀并依次给出下一个词的预测概率分布。例如，当看到前缀序列 `["大", "语言", "模型"]` 时，模型在前向计算中会输出整个词表大小的打分（Logits）。若目标词是 `"能够"`，训练的目标就是通过调整模型参数，让 `"能够"` 对应的 Softmax 概率尽可能接近 1，其余词的概率尽可能接近 0。

与推理阶段逐词生成的过程不同，在训练阶段，模型并非一个词一个词地串行计算。训练通过**下三角因果掩码（Causal Mask）**实现整段序列的并行前向计算：输入序列整体送入网络，每个位置只能看到自己及之前的 Token，然后所有位置同时计算预测下一个 Token 的交叉熵损失（Cross-Entropy Loss），一次性完成反向传播并更新梯度。

---

## 预训练知识体系与章节导航

理解大模型训练体系需要掌握三个递进层次：建模目标、优化与资源开销、分布式系统扩展。本模块通过 10 篇系统主讲与 1 篇实验练习逐一展开：

1. **自监督目标与因果建模**：
   - [/llm-100/training/what-is-pretraining.md](/llm-100/training/what-is-pretraining.md)：解析预训练如何利用海量文本的统计规律建立基础表示；
   - [/llm-100/training/autoregressive-lm.md](/llm-100/training/autoregressive-lm.md)：讲解自回归语言模型（Autoregressive LM）的 Next-Token Prediction 机制；
   - [/llm-100/training/mlm-vs-clm.md](/llm-100/training/mlm-vs-clm.md)：对比双向掩码语言模型（如 BERT 的 MLM）与单向因果语言模型（如 GPT 的 CLM）在注意力结构与适用场景上的差异；
   - [/llm-100/training/why-massive-data.md](/llm-100/training/why-massive-data.md)：说明参数规模、训练数据量与计算量之间的缩放法则（Scaling Laws）。

2. **计算开销与梯度优化**：
   - [/llm-100/training/loss-calculation.md](/llm-100/training/loss-calculation.md)：推导因果交叉熵损失的计算过程与输入标签的位移对齐（Shift Labels）；
   - [/llm-100/training/gradient-descent-backprop.md](/llm-100/training/gradient-descent-backprop.md)：追踪误差梯度经多层注意力与前馈网络反向传播的具体路径；
   - [/llm-100/training/why-high-vram.md](/llm-100/training/why-high-vram.md)：拆解模型权重、梯度、优化器状态与中间激活值在单卡上的显存占用构成。

3. **数值精度与分布式并行扩展**：
   - [/llm-100/training/mixed-precision.md](/llm-100/training/mixed-precision.md)：对比 FP32、FP16 与 BF16 的数值范围、精度损失与溢出特性，配套动手练习 [/llm-100/training/混合精度训练_FP16_BF16_有什么区别.md](/llm-100/training/混合精度训练_FP16_BF16_有什么区别.md) 验证混合精度对显存和计算吞吐的影响；
   - [/llm-100/training/parallel-strategies.md](/llm-100/training/parallel-strategies.md)：区分数据并行（DP）、张量并行（TP）与流水线并行（PP）的切分逻辑与通信开销；
   - [/llm-100/training/zero-deepspeed-fsdp.md](/llm-100/training/zero-deepspeed-fsdp.md)：分析 ZeRO 状态切分（ZeRO-1/2/3）与 PyTorch FSDP 的参数通信机制。

![自回归模型前向计算、损失求解与梯度反向传播闭环图](/images/content-rebuild/c58e5e07a16483c1/fig-1.png)

---

## 预训练与指令微调（SFT）的差异

预训练与后续的监督微调（SFT）虽然都使用自回归结构，但在数据组织和损失计算范围上有明确差异：

| 维度 | 基础预训练（Pre-training） | 指令微调（Supervised Fine-Tuning） |
| :--- | :--- | :--- |
| **数据形式** | 大规模非结构化文本，按固定长度（如 2048/4096）连续拼接截断 | 结构化问答与对话数据，包含清晰的 Prompt 与 Response 角色区分 |
| **学习目标** | 学习通用的语法、常识与推理链路，建立广泛的语言概率分布 | 学习遵循用户指令、安全对齐以及特定的输出格式 |
| **损失掩码** | 全序列计算：除填充位（Padding）外，每个位置的 Token 均参与交叉熵损失计算 | 条件计算：Prompt（用户输入）部分的标签被设为 `-100`，损失函数仅对 Response（助手回答）部分计算损失 |

在代码实现中，PyTorch 的交叉熵损失函数 `torch.nn.CrossEntropyLoss` 默认参数 `ignore_index=-100`。在 SFT 阶段，将 Prompt 部分的标签替换为 `-100` 后，反向传播就不会针对提问文本更新梯度，模型只专注提升回答文本的生成质量。

![预训练自监督预测与SFT指令掩码计算机制对比图](/images/content-rebuild/c58e5e07a16483c1/fig-2.png)

---

## 训练阶段的物理瓶颈与显存构成

大模型训练受制于三项底层物理限制：算力吞吐（FLOPs）、显存容量与带宽（VRAM / HBM），以及跨卡互联通信带宽（NVLink / InfiniBand）。

许多开发者容易将训练显存与推理显存混淆。推理阶段的显存主要由模型静态权重与动态 KV 缓存（KV Cache）组成；而训练阶段的显存占用主要由以下四部分构成（以使用 AdamW 优化器的标准 FP16/BF16 混合精度训练为例，设模型参数量为 $\Phi$）：

1. **模型权重（Weights）**：以 16 位浮点数存储，占用 $2\Phi$ 字节。
2. **梯度（Gradients）**：以 16 位浮点数存储，反向传播时占用 $2\Phi$ 字节。
3. **优化器状态（Optimizer States）**：AdamW 为每个参数维护一份 FP32 精度的一阶动量（4 字节）、二阶动量（4 字节）以及 FP32 主权重副本（4 字节），共计占用 $12\Phi$ 字节（若在不使用混合精度的纯 FP32 下训练，无需主权重副本，优化器状态本身占 $8\Phi$ 字节，此时参数、梯度与优化器状态三者合计为 $16\Phi$ 字节）。
4. **中间激活值（Activations）**：前向计算中暂存用于反向传播链式求导的中间张量。它的大小直接与序列长度、批大小（Batch Size）以及注意力层数成正比，可以通过激活重算（Activation Checkpointing）以额外计算为代价大幅换取显存节省。

当单张显卡的显存不足以容纳这些开销时，就需要借助分布式切分。数据并行切分训练批次，张量并行切分矩阵乘法，流水线并行切分模型层数，ZeRO 则对优化器状态、梯度和模型参数进行分片存储并在前向/后向中动态聚合通信。

![单卡显存开销解构与跨卡通信瓶颈分析图](/images/content-rebuild/c58e5e07a16483c1/fig-3.png)

---

## 动手实验：最小因果语言模型训练与评估

使用 Hugging Face 的 Transformers 与 Datasets 库，可以在单机环境中运行一个极简的自回归语言模型训练与验证流程。以下代码加载 `distilgpt2`，在切分数据集上执行训练并计算困惑度（Perplexity，即损失的指数 $e^{\text{loss}}$）：

```python
import math
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# 1. 载入分词器与最小因果模型
model_id = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 确保分词器具有填充 Token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)

# 2. 载入测试语料并划分为训练集与验证集
raw_datasets = load_dataset("dany0407/eli5_category", split="train[:600]")
split_datasets = raw_datasets.train_test_split(test_size=0.1667, seed=42)

def tokenize_function(examples):
    # 截断输入至固定长度窗口
    return tokenizer(examples["title"], truncation=True, max_length=64)

tokenized_datasets = split_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

# 3. 构造因果语言模型 Collator（mlm=False 声明自回归 Next-Token 预测）
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 4. 配置训练超参数
training_args = TrainingArguments(
    output_dir="./clm_baseline_output",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="no",
    # 若环境无独立 GPU 支持，可设 fp16=False
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)

# 5. 执行训练并评估验证集困惑度 (PPL)
trainer.train()
eval_results = trainer.evaluate()
eval_loss = eval_results["eval_loss"]
perplexity = math.exp(eval_loss)

print(f"Validation Loss: {eval_loss:.4f}")
print(f"Validation Perplexity: {perplexity:.2f}")
```

---

## 检验练习

1. **验证损失与困惑度换算**：运行上述代码，记录得到的 `eval_loss`。手动计算 $e^{\text{loss}}$，验证输出的 `perplexity` 是否一致。困惑度的物理意义是什么？如果词表大小为 50257，完全随机猜测时的理论困惑度上限是多少？
2. **观察显存与序列长度关系**：将代码中的 `max_length` 从 64 提升到 128 和 256，固定 `batch_size=4`，记录显存峰值变化。哪一部分显存开销增长最显著？
