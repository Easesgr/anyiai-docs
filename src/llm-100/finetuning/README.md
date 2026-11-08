---
title: 如何微调与对齐大模型？从数据准备到偏好优化的工程路径
icon: mdi:tune-vertical
description: 系统梳理大模型微调与人类偏好对齐的技术链条、显存构成、训练排障与选型决策，包含数据掩码可检验练习。
---

![微调与对齐模块学习全景图](/images/content-rebuild/b47a238db2f27550/cover.png)

## 1. 为什么需要微调与对齐？

通过海量无标注文本训练出来的原始基座模型（Base Model），本质上是一个依据上文预测下一个词概率的接续文本生成器。例如，向一个未微调的基座模型输入“请帮我将患者描述分诊到对应科室：头痛发热两天”，它可能不会给出科室建议，而是顺着语感接续写出类似“伴随窗外下起的暴雨，门诊大厅人头攒动”的小说情节。基座模型具备语言理解与通识特征，但并不理解问答对话的交互协议，也不清楚何时停止生成。

大模型工程通常将训练分为三个先后衔接的阶段：
1. **无监督预训练（Pre-training）**：在大规模通用语料上学习语言规律与事实表征。
2. **监督微调（Supervised Fine-Tuning, SFT）**：在由“提示词-标准回答”构成的指令对上训练，让模型学会遵循指令、输出期望格式（例如严格输出 JSON）并识别终止符。
3. **偏好对齐（Preference Alignment）**：通过 RLHF 或 DPO 等方法，使模型在多种可能的正确回答中，选择更符合人类偏好、更安全、更诚实的回答，抑制幻觉和危险言论。

微调改变的是模型的参数权重，主要用于改变语言风格、固化特定输入输出协议或内化特定任务的推理逻辑。如果要让模型获取实时外部事实或可随时检索核验的私有企业文档，应优先使用检索增强生成（RAG），微调并不适合替代外部事实检索。

![微调与对齐技术递进全景学习流向图](/images/content-rebuild/b47a238db2f27550/fig-1.png)

---

## 2. 模块递进学习路径

本模块包含 10 篇技术讲座，按照从数据工程、参数适配、偏好优化到评测验收的工程链条展开：

1. **数据准备与基础概念**
   - [/llm-100/finetuning/pretraining-vs-finetuning.md](/llm-100/finetuning/pretraining-vs-finetuning.md)：分析预训练、全量微调与指令微调在目标函数与训练状态上的差异。
   - [/llm-100/finetuning/finetuning-data-construction.md](/llm-100/finetuning/finetuning-data-construction.md)：学习指令数据集构建、对话模板渲染与损失掩码（Label Masking）处理。
2. **监督微调与高效适配**
   - [/llm-100/finetuning/sft-instruction-tuning.md](/llm-100/finetuning/sft-instruction-tuning.md)：使用因果语言模型交叉熵损失实现标准 SFT 训练流程。
   - [/llm-100/finetuning/adapter-prefix-ptuning.md](/llm-100/finetuning/adapter-prefix-ptuning.md)：梳理早期的 Adapter、Prefix-tuning 与 P-Tuning 方案及其权衡。
   - [/llm-100/finetuning/lora-efficient-finetuning.md](/llm-100/finetuning/lora-efficient-finetuning.md)：推导低秩分解更新矩阵原理，冻结基座权重并训练轻量旁路。
   - [/llm-100/finetuning/qlora-cost-reduction.md](/llm-100/finetuning/qlora-cost-reduction.md)：结合 4-bit NormalFloat 量化与双重量化，在单张消费级显卡上微调大模型。
3. **人类偏好对齐**
   - [/llm-100/finetuning/why-rlhf.md](/llm-100/finetuning/why-rlhf.md)：解释监督微调在多样性偏好下的局限，解析奖励模型训练与强化学习策略更新。
   - [/llm-100/finetuning/dpo-vs-rlhf.md](/llm-100/finetuning/dpo-vs-rlhf.md)：推导直接偏好优化（DPO）如何通过闭式解将策略与参考模型对比，跳过独立奖励模型与在线采样。
4. **训练稳定性与交付评测**
   - [/llm-100/finetuning/finetuning-overfitting.md](/llm-100/finetuning/finetuning-overfitting.md)：解决特定领域微调中的过拟合、通用能力遗忘与数据分布偏移问题。
   - [/llm-100/finetuning/evaluate-finetuned-model.md](/llm-100/finetuning/evaluate-finetuned-model.md)：搭建基于基准测试集、模型裁判（LLM-as-a-Judge）与防数据泄露的评测流程。

---

## 3. 三大工程故障与排障流程

### 故障一：显存溢出（Out Of Memory, OOM）
- **原因剖析**：训练时的显存消耗远大于推理。显存主要由两部分构成：
  1. **静态显存**：模型参数、梯度以及优化器状态。在常见的 16 位混合精度（fp16/bf16）训练中，模型参数占 2 字节，梯度占 2 字节；而标准的 fp32 AdamW 优化器需为每个可训练参数额外维护 fp32 权重备份（4 字节）、一阶动量（4 字节）和二阶动量（4 字节），仅优化器状态每个参数就占 12 字节，远超参数本身的显存开销。
  2. **动态显存**：前向传播保存的中间激活值（Activations），其大小随批次大小（Batch Size）和序列长度线性或二次方增长。
- **解决措施**：开启激活检查点（Activation Checkpointing）以重计算换取显存；改用 LoRA 或 QLoRA，将需要计算梯度的参数量缩减到基座的 1% 以下，从而大幅削减优化器状态开销；合理截断长文本序列。

### 故障二：Loss 不收敛或梯度震荡
- **原因剖析**：微调时若未对 Prompt 部分设置掩码，模型会被迫计算预测 Prompt 内容的交叉熵损失，破坏原有的指令跟随表征；若学习率过大且未配置 Warmup，训练刚开始的巨大梯度更新会冲垮模型已有权重。
- **解决措施**：在分词编码阶段，将对话模板中 Instruction 与 Context 对应的标签（Labels）填充为 `-100`（PyTorch 交叉熵损失函数默认忽略索引 `-100`），只对模型实际生成的 Response 计算 Loss；引入线性预热（Linear Warmup）和余弦衰减，同时配置梯度裁剪（Gradient Clipping）。

### 故障三：通用能力下降（灾难性遗忘）
- **原因剖析**：在狭窄单一任务数据集上进行多轮全参数训练，模型权重会逐渐偏向特定语料的模式，导致通用问答、代码和多步推理能力明显退化。
- **解决措施**：在微调语料中按比例混入 5% 至 10% 的通用高质量多任务数据；优先选用轻量 LoRA 微调而非全参数微调；在偏好对齐时保留对参考模型的 KL 散度惩罚，防止策略偏离原始基座分布过远。

---

## 4. 技术选型路径

在业务需求落地时，通常遵循“提示词工程 $\to$ 检索增强 $\to$ 轻量参数微调 $\to$ 全参数重塑”的阶梯进行评估，避免一开始就进行高成本的模型微调：

![垂直场景大模型能力改造选型决策树](/images/content-rebuild/b47a238db2f27550/fig-2.png)

- **上下文学习（ICL / Prompt 工程）**：如果基座模型理解能力足够，仅需规范输出格式、增加少样本示例（Few-Shot）即可满足要求，应优先采用提示词工程，研发和运维成本最低。
- **RAG（检索增强生成）**：当系统依赖外部知识库、业务系统数据库，或者知识更新频繁且需要严格溯源引用时，应采用 RAG 架构外挂知识，而不是依靠微调去“背诵”新知识。
- **LoRA / QLoRA**：当任务要求模型严格遵循特定通信协议（如特定结构的 JSON Schema）、吸收特定专业术语规范，或在消费级硬件上做适配时，LoRA/QLoRA 是成本与效果平衡的方案。生成的权重文件通常只有几十到几百兆字节，便于多任务切换。
- **全参数微调（Full Fine-Tuning）**：当需要注入全新语言体系、从底层改变模型的整体行为风格，且有充足算力支撑显存与优化器开销时，才考虑全参数微调。

需要说明的是，上述方案并非孤立互斥：在复杂生产实践中，微调与 RAG 经常组合使用（例如通过 LoRA 微调固化专业问答结构与指令遵循规范，同时接入 RAG 检索并注入最新的企业私有知识）。

---

## 5. 开源微调工具链协同关系

现代大模型微调实践通常基于 Hugging Face 及其周边生态展开，各组件分工明确：

![Hugging Face 微调开源技术栈协作架构图](/images/content-rebuild/b47a238db2f27550/fig-3.png)

- **Transformers**：负责底座模型权重加载、自注意力掩码生成以及分词器（Tokenizer）文本编码与解码。
- **PEFT**：在模型线性层旁路动态注入可训练的低秩矩阵（如 $A$ 和 $B$ 矩阵），并在推理交付时提供将 LoRA 权重与基座主干权重合并（Merge）的接口。
- **TRL**：基于 Transformers 和 PEFT 封装了 `SFTTrainer` 与 `DPOTrainer`，内置了训练时的 Prompt 自动掩码计算以及偏好对比损失计算逻辑。
- **Bitsandbytes / Accelerate**：Bitsandbytes 提供 4-bit/8-bit 量化算子支持；Accelerate 提供多卡张量调度与混合精度训练（fp16/bf16）支持。

---

## 6. 可检验练习：验证微调中的 Prompt 损失掩码

在监督微调中，必须确保模型只针对回答（Response）部分计算交叉熵损失，而提示词（Prompt）部分的标签应设为 `-100`。请在本地 Python 环境中运行以下验证脚本，观察分词器对标签的处理逻辑：

```python
from transformers import AutoTokenizer

# 以开源常用的 Qwen/Qwen2.5-0.5B 分词器为例（可替换为本地分词器）
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

prompt = "患者自述头痛发热两天，请分诊到对应科室。"
response = "建议挂号：神经内科或发热门诊。"

# 1. 分别对 prompt 与 response 编码
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
response_ids = tokenizer.encode(response, add_special_tokens=False)

# 拼接完整的输入 input_ids，并在末尾加上结束符
input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]

# 2. 构建微调 labels：prompt 部分填充 -100，response 及 eos 部分保留原 token_id
labels = [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]

# 3. 验证掩码切分是否正确
print("完整序列长度:", len(input_ids))
print("Prompt 对应 labels 是否全为 -100:", all(x == -100 for x in labels[:len(prompt_ids)]))
print("Response 第一个计算损失的 Token:", tokenizer.decode([labels[len(prompt_ids)]]))

# 检验断言
assert len(input_ids) == len(labels), "input_ids 与 labels 长度必须一致"
assert labels[len(prompt_ids) - 1] == -100, "Prompt 最后一个 token 标签必须是 -100"
assert labels[len(prompt_ids)] != -100, "Response 第一个 token 必须参与损失计算"
print("损失掩码构建检验通过！")
```

运行该代码，观察输出是否通过断言。在实际工程微调中，`TRL` 的 `DataCollatorForCompletionOnlyLM` 或自带的 `SFTTrainer` 均依照这一逻辑将 Prompt 对应的标签设为 `-100`，避免模型反向传播计算提示词的生成误差。
