---
title: 预训练、微调与指令微调有什么区别？大模型各训练阶段机制与选型指南
icon: mdi:tune-vertical
tag:
- 微调与对齐
- 大模型100讲
- RLHF
description: 拆解预训练、领域继续预训练与监督指令微调的数据形式、损失掩码与硬件开销，理清事实记忆与行为对齐的边界。
---

在开发基于大语言模型的业务应用时，许多团队经常遇到这类问题：面对具体的专业任务，到底应该直接调用通用基座、补充行业语料继续预训练，还是收集几千条问答做监督微调？如果选错了阶段，不仅可能耗费高昂的算力，还可能让模型丢失原有的推理和代码能力。

我们以一个具体的贯穿任务展开：**构建一个“临床用药与病历分析助手”**。
* 如果直接给纯预训练的基座模型输入“请指出阿司匹林的禁忌症：”，它可能不会回答，而是接着写出“，本品为白色结晶粉末，熔点135℃……”——它只学会了文本接龙，尚未建立“我是助手、正在回答用户提问”的行为规范。
* 如果给通用对话模型输入一段包含冷僻基因位点与医院私有缩写的病历，其参数中缺少该领域的实体共现先验，难以准确理解上下文语义，这时才涉及领域继续预训练。
* 如果希望它严格按照 JSON 格式输出用药风险等级，则需要指令微调。

本讲将从训练目标、数据形态、损失计算机制和硬件显存四个维度，理清这三个训练阶段的技术差异与选型依据。

![训练阶段全景纵览](/images/content-rebuild/023e29f94c1e36a9/cover.png)

---

### 预训练：因果语言模型的无监督预测

#### 优化目标与自回归计算
预训练（Pre-training）的目标是让模型学习自然语言中的词法、句法以及跨领域的实体共现关系。主流自回归语言模型（Causal Language Model, CLM）采用自左向右的单向生成机制，通过因果注意力掩码（Causal Mask）确保每个位置只能看到当前及之前的 Token。

设输入文本序列切分后的 Token 序列为 $(x_1, x_2, \dots, x_T)$，模型参数为 $\theta$。预训练的优化目标是最大化序列整体的条件对数似然，对应损失函数为全序列的平均交叉熵损失：

$$\mathcal{L}_{\text{CLM}}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \dots, x_{t-1}; \theta)$$

在预训练中，文本中的每一个位置都在计算损失并反向传播梯度。模型在每个位置根据前面所有 Token 预测下一个 Token 的概率分布，并与实际出现的词计算交叉熵。

![预训练的自回归预测机制](/images/content-rebuild/023e29f94c1e36a9/fig-1.png)

#### 数据特征与所得能力
预训练使用的数据通常达到数万亿（Trillion）Token 级别，来源包括网页抓取、百科、图书、期刊文献以及开源代码。数据处理时，长文本被连续拼合并切分为固定长度的窗口（例如 4096 或 8192 Token），不带有用户与助手的对话角色标记。

完成预训练后，产出的是**基座模型（Base Model）**。基座模型具备强大的语言补全和上下文关联能力，但它并不默认遵从问答指令。若在提示词后不加约束，它只会沿着高概率路径继续接龙。

深入了解预训练的底层分布式训练与分词细节，可参阅 [/fundamentals/training-and-finetuning/pretraining-deep.md](/fundamentals/training-and-finetuning/pretraining-deep.md)。

<ReadMoreLock />

---

### 领域继续预训练：适应专有文本分布与防止遗忘

当基座模型面对临床病历、未公开芯片手册或特定地方司法文书时，常常会遇到两类障碍：分词器（Tokenizer）将专业术语切分成大量碎片、模型参数中缺少该领域的专业词汇先验共现。此时可采用**领域继续预训练（Domain-adaptive Pretraining，简称 DAPT 或 Continued Pre-training）**。

#### 数据形态与训练目标
继续预训练沿用了预训练的全部机制：依然是因果语言建模目标（CLM），依然是对无标注长文本切片，每个 Token 都参与计算损失。不同之处在于语料分布高度集中于垂直领域，规模通常在数十亿到数百亿 Token。

![领域继续预训练与领域漂移](/images/content-rebuild/023e29f94c1e36a9/fig-2.png)

#### 分布偏移与灾难性遗忘
在垂直语料上持续更新参数，会导致模型权重朝新分布快速移动，原先在通用分布上优化的局部极小值可能被破坏，这种现象即为**灾难性遗忘（Catastrophic Forgetting）**。例如，纯粹使用 200 亿 Token 医疗文献微调基座，模型的通用代码生成和多步数学推导能力常会出现明显回落。

业界缓解该问题的典型策略是**数据重放（Data Replay）**：在领域数据集中按比例混合通用预训练语料（通常占 10% 到 20%），包含通识百科、开源代码和通用多语言数据，使梯度更新方向兼顾通用特征流形。

---

### 监督指令微调（SFT）：人机契约与损失掩码

监督指令微调（Supervised Fine-Tuning, SFT）又称指令微调（Instruction Fine-Tuning），其核心目标是让模型学会**识别用户意图、遵从提示词约束、以助手的身份进行结构化回复**。

#### 对话模板（Chat Template）的结构化包装
SFT 阶段不再输入无界连续文本，而是通过特殊分隔符划分出角色边界。以常见的 ChatML 模板为例：

```text
<|im_start|>system
你是一位专业的临床药剂师助手。请根据用户描述指出药物禁忌。<|im_end|>
<|im_start|>user
患者有活动性消化道溃疡，能服用阿司匹林肠溶片吗？<|im_end|>
<|im_start|>assistant
不能。活动性消化道溃疡是阿司匹林的禁忌症，可能诱发或加重出血。<|im_end|>
```

#### 损失掩码机制（Prompt Loss Masking）
SFT 与预训练在代码实现上最重要的差别在于：**模型只对 Assistant 的回复部分计算损失，不为 Prompt（包括 System、User 标记与内容）计算梯度**。

如果对用户提问部分也计算交叉熵，模型就会耗费参数容量去“预测用户下一句话会问什么”，甚至容易对特定指令句式过拟合。在 PyTorch 与主流框架中，通常把 Prompt 对应的 Token 目标标签设为 `-100`（即 `torch.nn.CrossEntropyLoss` 的默认 `ignore_index`），前向计算损失和反向传播时将自动跳过这些位置：

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{|\text{Response}|}\sum_{t \in \text{Response}} \log P(x_t \mid x_{<t}; \theta)$$

![SFT 的 Prompt-Response 结构与 Loss Mask 机制](/images/content-rebuild/023e29f94c1e36a9/fig-3.png)

通过对回复段的监督，模型习得了三项关键行为：
1. **角色认知**：识别指令边界，在 `<|im_start|>assistant` 后开始作答。
2. **终止能力**：在回答结束时稳定吐出 `<|im_end|>` 等结束符（EOS Token），主动停止推理。
3. **格式约束**：稳定产出符合要求的 JSON 字段、Markdown 表格或特定编程语言代码。

构建高质量微调数据集的方法与清洗流程，可参考 [/llm-100/finetuning/finetuning-data-construction.md](/llm-100/finetuning/finetuning-data-construction.md) 与 [/llm-100/finetuning/sft-instruction-tuning.md](/llm-100/finetuning/sft-instruction-tuning.md)。

---

### 知识机制：参数硬记与上下文检索的边界

在实际工程中，一个高频出现的误区是试图通过 SFT 注入新事实：例如准备 2000 条包含医院各科室专家出诊排班表和药品最新单价的问答对，希望微调后模型能准确回答“王医生周几出诊”。上线测试后，模型往往在具体日期和价格上频繁产生幻觉。

#### 为什么 SFT 注入孤立事实极其不可靠？
1. **统计证据的稀疏性**：预训练中的常识往往以成千上万种不同句式在海量语料中重复出现，参数在不同上下文之间形成了泛化关联。而人工标注的几千条微调样本中，某个事实可能仅出现一两次，模型很难建立稳固的实体关联，反而容易过拟合到提问的表面词句。
2. **输出分布的虚假自信**：在微调的高学习率或多次 Epoch 迭代下，模型对未在训练集中充分交叉验证的实体也会给出高概率预测，导致回答虽然语气肯定，但关键数值和实体完全失真。

![知识沉淀与行为激活的概率分布差异](/images/content-rebuild/023e29f94c1e36a9/fig-4.png)

#### 工程选型准则：参数行为与检索增强
学术与工程界形成的主流经验是：**预训练建立广泛事实与语言先验，微调规范输出结构与对话行为**。对于高频变动、要求可追溯、容错率低的事实性数据（如排班表、商品库存、接口文档），首选方案是**检索增强生成（RAG）**：

![事实注入失败对比：SFT 强记与 RAG 上下文读取](/images/content-rebuild/023e29f94c1e36a9/fig-5.png)

关于 Prompt 工程、知识库检索与微调的取舍决策，可进一步查阅 [/fundamentals/training-and-finetuning/prompt-rag-finetuning.md](/fundamentals/training-and-finetuning/prompt-rag-finetuning.md)。

---

### 算力与显存：三阶段硬件资源测算

在规划训练任务时，必须量化算力（FLOPs）与单步训练显存需求，避免硬件选型失误。

#### 计算量估算模型
标准 Transformer 模型在因果自回归训练中，处理单个 Token 的浮点运算量近似公式为：

$$\text{FLOPs/Token} \approx 6N$$

其中 $N$ 为模型非 Embedding 参数量。前向传播约消耗 $2N$ FLOPs，反向传播约消耗 $4N$ FLOPs（激活值梯度与参数梯度各占约 $2N$）。

以 7B（70 亿参数）模型为例，对比各阶段处理总量与计算开销：
* **预训练**：训练 2T（20000 亿）Token，总计算量约为 $6 \times (7 \times 10^9) \times (2 \times 10^{12}) = 8.4 \times 10^{22}$ FLOPs。通常需要数百至数千张 GPU 集群并行数周。
* **领域继续预训练**：训练 10B（100 亿）领域 Token，总计算量约为 $6 \times (7 \times 10^9) \times (1 \times 10^{10}) = 4.2 \times 10^{20}$ FLOPs，约为预训练总量的 0.5%。
* **监督微调（SFT）**：若包含 10 万条样本，平均长度 1024 Token，训练 3 个 Epoch，总 Token 吞吐量约为 $3 \times 10^8$。总计算量约为 $6 \times (7 \times 10^9) \times (3 \times 10^8) = 1.26 \times 10^{19}$ FLOPs，仅为预训练的几千分之一到万分之一。

#### 单步训练显存拆解（以 7B 全参数微调为例）
注意：**虽然 SFT 处理的总 Token 数少，但如果采用全参数微调，单步迭代时的显存占用与预训练在同等序列长度下几乎一致**。在 16 位混合精度（FP16/BF16）与标准 AdamW 优化器配置下，静态显存开销包括：
* **模型参数（Weights）**：每个参数 2 字节（FP16/BF16），7B 参数占用约 $14\,\text{GB}$。
* **参数梯度（Gradients）**：每个参数 2 字节，占用约 $14\,\text{GB}$。
* **AdamW 优化器状态（Optimizer States）**：需维护 FP32 参数主副本（4 字节）、一阶动量（4 字节）与二阶动量（4 字节），每个参数共 12 字节，占用约 $84\,\text{GB}$。

三者合计静态显存达 $14 + 14 + 84 = 112\,\text{GB}$。由于未计入前向激活值（Activations）与 PyTorch 运行时开销，单张 80GB 显存的 GPU 无法容纳 7B 模型的全参数训练，必须依赖 FSDP、ZeRO-3 分片或采用参数高效微调（PEFT，如 LoRA）。

![阶段工程资源消耗对比](/images/content-rebuild/023e29f94c1e36a9/fig-6.png)

#### 三个训练阶段核心差异总结

| 比较维度 | 预训练（Pre-training） | 领域继续预训练（DAPT） | 监督指令微调（SFT） |
| :--- | :--- | :--- | :--- |
| **主要优化目标** | 全序列自回归预测损失 | 全序列自回归预测损失 | 仅对回复段计算交叉熵损失 |
| **数据组织形式** | 连续长文本切片，无角色边界 | 行业长文本切片，需混合通用语料 | 对话模板包装的指令-响应样本对 |
| **样本规模典型值** | 1T ~ 15T Tokens | 1B ~ 50B Tokens | 1万 ~ 100万 条对话对 |
| **核心解决问题** | 建立基础语义表示与广度常识 | 调整特定领域词汇分布与术语先验 | 确立人机交互规范、对齐业务格式 |
| **遗忘风险** | 无（作为初始训练） | 较高（易损害通用推理与代码能力） | 较低（需防范特定任务过拟合） |
| **全参数单步显存** | 与参数量强相关（7B约需超112GB） | 与预训练相同规模 | 与预训练相同规模（全参数下） |
| **低成本适配方案** | 不适用 | 极少使用参数微调，通常需全量更新 | 可使用 LoRA/QLoRA 实现单卡训练 |

关于如何使用低秩矩阵分解在消费级显卡上微调 7B/14B 模型，可继续阅读 [/llm-100/finetuning/lora-efficient-finetuning.md](/llm-100/finetuning/lora-efficient-finetuning.md)。

---

### 动手检验：验证 SFT 损失掩码的数据构建

为了检验你对 SFT 核心机制的理解，以下提供一段标准的 Hugging Face 风格对话预处理逻辑。请阅读代码并观察 `labels` 是如何对 Prompt 进行掩码处理的：

```python
import torch
from transformers import AutoTokenizer

# 假设使用 Qwen 风格分词器（此处演示逻辑）
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 定义一条单轮训练样本
prompt = "请将‘心房颤动’翻译成英文，并提取缩写。"
response = "心房颤动的英文是 Atrial Fibrillation，常用临床缩写为 AF。"

# 使用 Chat 模板分别获取 prompt 部分与整体文本
system_prompt = "你是一名医学术语助手。"

# 构造带特殊 token 的指令部分
prompt_text = (
    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    f"<|im_start|>user\n{prompt}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
full_text = prompt_text + f"{response}<|im_end|>\n"

# 编码
prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
full_ids = tokenizer.encode(full_text, add_special_tokens=False)

# 构造 Labels：复制 full_ids，但将 prompt 对应长度的位置全部置为 -100
labels = list(full_ids)
for i in range(len(prompt_ids)):
    labels[i] = -100

# 验证断言：确保 Prompt 部分完全不计算梯度
assert labels[:len(prompt_ids)] == [-100] * len(prompt_ids), "掩码构建错误：Prompt 未完全被忽略"
assert labels[len(prompt_ids):] == full_ids[len(prompt_ids):], "回复段标签错误：Response 未保留实际 Token ID"

print(f"总 Token 长度: {len(full_ids)}")
print(f"被屏蔽的 Prompt 长度: {len(prompt_ids)}")
print(f"实际计算 Loss 的 Response 长度: {len(full_ids) - len(prompt_ids)}")
```

#### 课后练习题
1. **场景决策**：某金融科技公司希望大模型能够从每日更新的 50 份上市公司财报 PDF 中，精准提取当季营收与净利润。团队中有成员建议“每周收集 1000 条财报问答，全量微调模型参数以注入最新财报数据”。这种方案存在什么风险？更合理的工程架构应当如何设计？
2. **代码逻辑**：如果由于逻辑缺陷，在构建 SFT 数据集时忘记将 Prompt 部分设为 `-100`，直接传入 `labels = full_ids` 进行训练，模型在推理时最可能出现哪种退化行为？
