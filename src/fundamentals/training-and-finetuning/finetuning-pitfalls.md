---
title: 大模型微调常见问题有哪些？排查排坑与验收指南
icon: mdi:alert-circle-outline
tag:
- 训练与微调
- 过拟合
- 灾难性遗忘
description: 系统梳理大模型微调中遗忘、过拟合、Loss变NaN、显存OOM与评估虚高五大典型故障，提供清晰的因果解释与排查验收方案。
---

在大模型开发中，许多开发者把数据整理好、写好脚本启动训练后，往往会遇到一系列意料之外的阻碍：跑了几轮后模型连加减法都不会算了；刚训练几个 Step，Loss 突然跳成 `NaN`；显存毫无征兆地报出 CUDA OOM；或者在测试集上准确率接近 100%，实际提问时却满嘴胡话。

微调并不是简单地把预训练模型拿来用新数据跑几次梯度下降。由于模型参数规模大、混合精度表示范围有限，以及数据分布容易失衡，整个训练全链路很容易在数值、显存、泛化和评估上踩坑。

为了让排查过程更具体，我们以一个贯穿任务为例：**基于 7B 基座模型微调一个“企业工单问答与派单助手”**。本文结合 [SFT 监督微调是什么？](/fundamentals/training-and-finetuning/sft.md) 与 [LoRA 和 QLoRA 是什么？](/fundamentals/training-and-finetuning/lora-qlora.md)，带你逐一拆解微调全流程中的五大高频问题及排查验收方法。

![模型微调故障排查全景图](/images/content-rebuild/85d33df27000deac/cover.png)

---

## 灾难性遗忘：专业能力学会了，通用能力崩塌了

### 什么是灾难性遗忘？
在微调工单助手时，你可能会发现：模型确实学会了按照特定格式派发工单，但当用户顺便问一句“今天周三，后天是周几”或者让它写一段简单的 Python 脚本时，它给出的回答语无伦次，甚至连基本的常识判断都丢失了。

这种**模型在新数据分布上拟合后，原有基座模型的通用常识、多步推理或代码能力发生显著退化的现象，被称为灾难性遗忘（Catastrophic Forgetting）**。

### 为什么会发生遗忘？
根本原因在于梯度更新方向完全被新数据占据：
1. **梯度更新的分布偏差**：在标准微调中，损失函数只针对新数据集 $\mathcal{D}_{\text{new}}$ 计算：
   $$\min_{\theta} \mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{new}}} [\mathcal{L}(f_\theta(x), y)]$$
   模型每一轮更新的梯度向量 $g = \nabla_\theta \mathcal{L}_{\text{new}}$ 完全由新任务主导。如果微调样本的语义空间很窄（例如全都是工单描述与分类），模型参数就会朝着仅满足新任务的方向持续偏移，原先在海量预训练文本中建立起来的多领域特征响应便会被强行冲刷和覆盖。
2. **微调方式的差异**：
   - **全参数微调**：全量参数在每一步都会更新，新梯度的持续冲击极易让未受约束的深层参数偏离原先的通用表达。
   - **LoRA / QLoRA 微调**：基座原始权重 $W_0$ 保持冻结，仅更新低秩矩阵 $\Delta W = B \times A$。因为可更新参数量少且被限制在低秩子空间内，LoRA 天然具备一定的抗遗忘能力；但如果微调轮数过多或学习率过大，增量权重依然会主导模型的注意力打分，造成行为退化。

![灾难性遗忘机制与通用数据回放混合方案](/images/content-rebuild/85d33df27000deac/fig-1.png)

### 缓解对策：通用数据回放混合（Replay Mixing）
解决灾难性遗忘最直接有效的工程方法不是事后打补丁，而是在训练集构建阶段引入**回放数据混合**：
- **基线封存**：在启动微调前，先对基座模型在通用评测集（如常识理解、基础数学推理、通用多轮对话）上跑出基准分数并留存记录。
- **数据回放（Replay）**：在垂直业务数据中混入约 $5\% \sim 15\%$ 的高质量通用指令数据（包含日常对话、逻辑推理与代码样本）。

```python
# 示意代码：在数据集构建时实现通用回放混合（Replay Mixing）
def build_mixed_finetuning_dataset(domain_dataset, general_replay_dataset, replay_ratio=0.1):
    total_domain_samples = len(domain_dataset)
    # 按照比例计算需要补充的通用样本数
    num_replay = int(total_domain_samples * replay_ratio / (1 - replay_ratio))
    sampled_replay = general_replay_dataset.shuffle(seed=42).select(range(num_replay))
    
    # 混合并打乱，避免模型在训练前半段只学通用、后半段只学垂直领域
    mixed_dataset = concatenate_datasets([domain_dataset, sampled_replay])
    return mixed_dataset.shuffle(seed=42)
```

混合通用数据后，损失函数中始终保留了一部分通用任务的梯度信号，让优化过程兼顾新业务与通用语义，有效阻断能力崩塌。

---

## 过拟合与模式坍缩：复读机与套话泛滥

### 典型表现：复读机与句式模板僵化
在大模型生成任务中，过拟合通常不表现为分类准确率变化，而表现为**生成模式的坍缩（Mode Collapse）**：
- **Token 级循环（复读机）**：模型生成某个关键词后陷入自回归死循环，不断重复同一个短语，直到强行达到最大生成长度 `max_new_tokens` 截断。
- **句式模板固化**：无论输入的问题是什么，模型开头永远输出一模一样的模板套话（例如：“您好，工单助手为您查询到以下规则……”）。即使问题只是一句简单的寒暄，它也会强行贴靠这套模板。

### 发生机制
1. **Prompt 模板过于单一**：如果在按照 [微调数据怎么构造？](/fundamentals/training-and-finetuning/finetuning-data.md) 准备数据时，几千条指令的问法和系统提示词几乎完全一致，自回归模型为了快速降低交叉熵损失，就会本能地把权重押注在固定开头的局部高频词上，丧失了根据具体语境灵活应变的能力。
2. **训练轮次过多或学习率过高**：大语言模型在训练时采用 Teacher Forcing 模式（即每一步都基于真实的标注前缀来预测下一个 Token）。如果训练轮次（Epoch）过大，模型会把微调样本的特定字词死记硬背下来，导致一旦推理时的真实前缀稍有波动，模型就会因为找不到匹配路径而陷入重复循环。

<ReadMoreLock />

### 监控与应对：验证集困惑度与早停法（Early Stopping）

防范过拟合的核心在于监控独立的验证集损失（Evaluation Loss）以及验证困惑度（Perplexity, $\text{PPL} = e^{\mathcal{L}_{\text{eval}}}$）。

![过拟合与模式坍缩形成机制及早停监测](/images/content-rebuild/85d33df27000deac/fig-2.png)

工程实践中不能等到整轮训练结束才看一眼结果，而应设置按步评估（如每 100 步计算一次验证集损失）。一旦验证集损失停止下降并连续反弹，应立即触发早停截断：

```python
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./output_checkpoints",
    evaluation_strategy="steps",
    eval_steps=100,              # 每 100 步在验证集上跑一次评估
    save_steps=100,
    save_total_limit=3,          # 仅保留指标最好的前 3 个检查点
    load_best_model_at_end=True, # 训练结束后自动载入验证集表现最佳的权重
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=20,
    learning_rate=2e-5,
    num_train_epochs=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # 验证损失连续 3 次反弹则提前终止
)
```

---

## 数值不稳定：Loss 突变 NaN/Inf 的原因与排查

微调过程中最棘手的情况，莫过于原本训练平稳的曲线在某一个 Step 突然崩掉，Loss 变成 `NaN`（Not a Number）或 `Inf`（无穷大），随后的梯度全部变为 0 或无效值。

当 Loss 突变为 `NaN` 时，**排查的第一步是观察混合精度训练的 Loss Scaler（动态损失缩放因子）以及崩溃前几步的梯度全局范数（Grad Norm）**。如果因反向梯度持续溢出（Overflow）导致 Loss Scaler 不断减半趋近于零，或者 Grad Norm 在某一步瞬间跨数量级暴增，说明数值已击穿浮点数的有效表示区间。

### 根因一：FP16 混合精度的动态范围溢出
在 IEEE 754 标准的 FP16 格式中，分配了 5 位阶码（Exponent）和 10 位尾数（Mantissa）。FP16 能够表示的最大正实数为：
$$X_{\text{max}} = 2^{15} \times (1 + \sum_{i=1}^{10} 2^{-i}) = 32768 \times (2 - 2^{-10}) = 65504$$

在 Transformer 内部，注意力的点积计算 $Q K^T / \sqrt{d_k}$ 随着序列增长容易出现较大数值。如果直接使用 FP16 且未做减去最大值的数值平移操作，当输入大于 $\ln(65504) \approx 11.09$ 时，自然指数 $\exp(x)$ 就会立即上溢变为 `Inf`。
虽然标准的 Safe Softmax 会先减去最大值（使指数输入恒 $\le 0$）避免指数直接上溢，但若前向计算出的未归一化 logits 本身已超出 FP16 上限（$> 65504$）溢出为 `Inf`，或者整行被注意力掩码完全屏蔽，传入 Softmax 时便会出现 $\text{Inf} - \text{Inf}$ 或 $0 / 0$ 产生 `NaN`，随后的反向传播梯度也会变成 `NaN`，破坏主权重。

**为什么优先推荐 BF16？**  
Bfloat16（BF16）同样占用 16 位显存，但重新划分了结构：保留 8 位阶码，仅留 7 位尾数。它的阶码位数与 FP32 完全相同，最大可表示数值高达 $\sim 3.39 \times 10^{38}$。虽然尾数精度略低于 FP16，但在深度学习微调中，BF16 彻底消除了因动态范围不足导致的数值上溢问题。只要 GPU 硬件支持（如 NVIDIA Ampere 架构 A100 及后续显卡），混合精度训练应优先选用 BF16。

![FP16与BF16数值范围对比及NaN排查流程](/images/content-rebuild/85d33df27000deac/fig-3.png)

### 根因二：缺少学习率预热（Warmup）
在微调刚开始的数十个步数内，参数尚处于随机增量状态，如果第一步就采用最大设计学习率，极大的梯度步长很容易把模型推入未知的数值畸变区。必须在训练参数中显式设置线性 Warmup（通常占总步数的 $3\% \sim 10\%$），让学习率从零平滑爬升到预设值。

### 根因三：缺少梯度裁剪（Gradient Clipping）
当微调语料中偶尔出现极端长句、异常字符或脏样本时，反向传播可能在单个 Step 计算出局部巨大的梯度向量。如果缺少全局范数裁剪，该梯度会直接把优化轨迹踹偏：
$$\text{若 } \|g\|_2 > \tau, \quad g \leftarrow g \cdot \frac{\tau}{\|g\|_2}$$
在训练配置中设置 `max_grad_norm=1.0`，在参数更新前强行约束梯度向量长度，能有效避免梯度突刺毁掉模型。

---

## 显存 OOM 排查：区分训练显存与推理显存

微调过程中突然中断并弹出 CUDA Out of Memory（OOM），往往是因为没有分清训练显存与推理显存的结构差异。推理阶段显存主要消耗在静态模型权重和动态增长的 KV 缓存（KV Cache）上；而**训练阶段并不维护 KV 缓存，显存主要由静态权重、优化器状态、反向梯度和前向激活值（Activations）构成**。

### 训练显存开销拆解（以 AdamW 全参数微调为例）
假设模型参数量为 $\Phi$（如 7B 模型）：
1. **静态开销（参数、梯度与优化器状态）**：
   - 模型参数（FP16/BF16）：每个参数 2 字节，共 $2\Phi$ 字节。
   - 参数反向梯度（FP16/BF16）：每个参数 2 字节，共 $2\Phi$ 字节。
   - AdamW 优化器状态：需要维护 FP32 参数主副本（4 字节）+ 一阶动量（4 字节）+ 二阶动量（4 字节），共 $12\Phi$ 字节。
   - **静态小计**：$(2 + 2 + 12)\Phi = 16\Phi$ 字节。对于 7B 参数模型，静态占用便达到 $16 \times 7\text{ GB} \approx 112\text{ GB}$。这就是全参数微调通常必须借助 DeepSpeed ZeRO 状态分片或使用 LoRA 的原因。
2. **动态开销（前向激活值 Activations）**：
   - Transformer 每一层的注意力矩阵与中间 MLP 输出必须留在显存中，直到反向传播计算梯度完成才能释放。激活值开销随着 Batch Size $B$ 和序列长度 $L$ 显著增加。

![显存开销拆解与动态长度突刺防护](/images/content-rebuild/85d33df27000deac/fig-4.png)

### 常见 OOM 触发原因与应对清单

| 故障原因 | 触发机制 | 工程修复方案 |
| :--- | :--- | :--- |
| **动态长度突刺** | 数据集样本长短不齐。若某个 Batch 恰好装入了数条接近最大长度 `max_seq_len` 的样本，中间激活值显存瞬间冲破显存上限。 | 设置 `group_by_length=True`，让长度相近的样本组装在同一个 Batch；或者提前对异常超长样本进行截断过滤。 |
| **激活值撑满显存** | 序列长度较长时，各层完整保留前向激活张量导致显存溢出。 | 必须开启 `gradient_checkpointing=True`（梯度检查点）。反向传播时丢弃部分中间激活值并重新计算，用轻微的计算开销大幅释放显存。 |
| **Padding 填充浪费** | Batch 内的短文本被强行填补了大量的 `[PAD]` Token，白白占据显存参与矩阵计算。 | 使用动态 Padding（`DataCollatorWithPadding`）或 Packing 技术，将短序列拼接紧凑，减少无意义的 Padding 占用。 |
| **单卡 Batch 过大** | 单张 GPU 的物理显存无法承载设定的 `per_device_train_batch_size`。 | 调小单卡 Batch Size（例如设为 1 或 2），同时按比例增大 `gradient_accumulation_steps`（梯度累加步数），保持等效总 Batch 不变。 |

---

## 评估陷阱：跑分虚高与格式幻觉

微调完成后，许多开发者仅凭自动化评测跑分就认定模型已经“训练成功”，但上线实际对接工单时却问题频出。这里存在三个典型的评估误区：

![评估偏差来源与盲测验证体系](/images/content-rebuild/85d33df27000deac/fig-5.png)

### 陷阱一：测试集数据泄漏（Data Contamination）
在准备训练语料或使用大模型合成数据时，如果没有对验证集或通用基准题库进行严格的 N-gram 重合过滤，测试题目可能会以细微改写的形式混入训练集。
此时模型在验证集上的高分只是在重放记忆内容，面对真实场景中未曾出现过的表达，泛化性能会大打折扣。

### 陷阱二：裁判模型的同源偏见（Self-Enhancement Bias）
如果使用大模型作为裁判（LLM-as-a-Judge）对生成结果打分，需要注意两种固有偏差：
- **同源偏见**：如果你的微调数据是由模型 A 蒸馏生成的，在评估阶段若继续使用模型 A 作为裁判，它会天然倾向于给自己生成风格的文本打出高分。
- **长度偏见（Verbosity Bias）**：裁判模型往往更偏好篇幅冗长、分点繁复的回答，即便里面的有效信息密度很低。

### 陷阱三：格式合规陷阱（Format Mimicry）
微调非常擅长学习表面格式。例如要求输出严格的工单 JSON 结构，微调后的模型在 JSON 解析成功率上可能达到 100%，导致评测脚本给出满分。但在解析后的字段内容里，工单分类、派单对象甚至问题描述可能完全是凭空编造的幻觉。**绝不能把“格式输出合规”等同于“内容事实正确”**。

---

## 端到端微调工程健康验收清单

为了保证交付的微调模型兼顾垂直任务质量与通用基础能力，建议在模型导出前执行四步健康验收：

![端到端微调健康验收全流程流水线](/images/content-rebuild/85d33df27000deac/fig-6.png)

### 步骤一：训练动力学指标核查
- [ ] **Loss 走势**：Training Loss 与 Eval Loss 平稳收敛，未在训练中后期出现持续且剧烈的 U 型反弹。
- [ ] **梯度范数（Grad Norm）**：全程未出现突发性暴增，没有持续触发梯度裁剪上限。
- [ ] **数值稳定性**：训练日志中无任何 `NaN` 或 `Inf` 记录，BF16/FP16 缩放正常。

### 步骤二：通用能力保底回归
- [ ] **通用基准对比**：对比基座模型微调前后的常识推理与指令遵循基准，降幅控制在业务允许的安全阈值内（通常建议通用指标降幅不超过 $3\% \sim 5\%$）。
- [ ] **代码与逻辑保留**：若原有业务依赖结构化输出或逻辑运算，需运行基准回归集核验该能力是否被破坏。

### 步骤三：输入扰动与边界测试
- [ ] **抗扰动测试**：在测试用例的 Prompt 中插入无关前缀、同义修饰词或打乱语序，验证输出结果是否依然保持稳定。若输入微调即崩，说明模型形成了严重的字词位置死记。
- [ ] **长上下文截断测试**：注入接近上下文上限的长文本，核查模型是否会触发显存越界或陷入死循环复读。

### 步骤四：真实业务盲测抽检
- [ ] **人工双盲抽检**：随机抽取至少 100 条未参与微调的真实业务长尾请求，将微调前基座、微调后模型与原模型的结果打乱盲审，重点核查工单派发的实际逻辑与事实准确率。

如果在完成上述微调和健康验收后，模型在具体任务上的偏好表现、安全边界或拒绝回答逻辑仍需精细对齐，可以继续阅读后续章节 [RLHF 和 DPO 有什么区别？](/fundamentals/training-and-finetuning/rlhf-dpo.md)，了解如何利用偏好对数据进一步校准模型行为。
