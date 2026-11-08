---
title: SFT 指令微调到底怎么做？模型训练时的损失计算与批次处理
icon: mdi:tune-vertical
tag:
- 微调与对齐
- 大模型100讲
- RLHF
description: 详解监督指令微调中的 Teacher Forcing 机制、Prompt 损失掩码实现、PyTorch 错位切片细节，以及动态填充与样本拼接的工程权衡。
---

在[微调数据应该怎么构造？](/llm-100/finetuning/finetuning-data-construction.md)中，我们完成了指令样本的清洗与模板拼接。拿到这些由系统提示、用户提问与目标回答组成的文本后，送进模型训练时，模型究竟是如何计算损失并更新权重的？

监督指令微调（Supervised Fine-Tuning，简称 SFT）的核心目标是训练模型在给定上下文的条件下，顺着提示语生成期望的回答，而不是去背诵用户提问的句式。为了搞清楚这一过程，本文以一个具体的“客服工单摘要”任务为贯穿示例，深入拆解自回归训练的前向传播、损失掩码实现细节、批次排布与训练稳定性控制。

![监督指令微调训练执行全景概览](/images/content-rebuild/b01bbf140237253a/cover.png)

---

### 贯穿示例：客服工单摘要任务

假设我们要训练模型执行如下任务：
- **Prompt（提示上下文）**：`“提取问题：包裹滞留在转运中心三天了，单号 SF101。”`
- **Response（目标回答）**：`“单号SF101滞留超72小时。<eos>”`

如果将整句话合并送入模型，总序列就是“提取问题：包裹滞留在转运中心三天了，单号 SF101。单号SF101滞留超72小时。`<eos>`”。训练程序需要做的是：让模型根据前面的提示生成后面的回答，同时避免让前面的提问反向干扰模型本身的语言习惯。

---

### 1. Teacher Forcing 自回归前向传播机制

大语言模型在实际部署推理时，只能一步一步串行生成：先根据输入的 Prompt 预测第 1 个字，把第 1 个字拼到输入末尾，再输入模型预测第 2 个字，直到生成终止符 `<eos>`。这种方式称为自回归采样。

如果训练阶段也像推理一样逐字串行循环生成，会有两个严重问题：
1. **无法并行计算**：GPU 的矩阵运算单元在大序列长度下无法一次性打满，训练速度极慢；
2. **误差单向累积**：如果模型在生成第 2 个字时猜错了，后续步骤都会在这个错误的基础上继续运算，导致偏离预定轨迹，梯度难以收敛。

因此，因果语言模型的 SFT 统一使用 **Teacher Forcing（教师强制）** 机制进行并行训练。

![Teacher Forcing 训练与自回归推理前向流程对比](/images/content-rebuild/b01bbf140237253a/fig-1.png)

Teacher Forcing 的运行机制如下：
1. **前向输入一次性给齐**：整个序列的真实 Token（包括 Prompt 和目标 Response）作为一个整体直接送入模型。在计算预测第 $t$ 个位置的目标词时，模型所依赖的历史上下文全部直接取自数据集的真实文本 $(x_1, x_2, \dots, x_{t-1})$，而不是模型自身上一步采样出来的预测结果。
2. **下三角因果注意力掩码（Causal Mask）**：在 Transformer 的自注意力计算中，通过将注意力权重矩阵的右上角置为负无穷（$-\infty$），强制第 $t$ 个位置只能关注自己和它前面的 Token，禁止模型“偷看”后面的真实答案。
3. **单次前向输出全序列预测**：模型只需要做一次前向传播，就能在 GPU 上并行计算出整条序列所有位置对应的隐藏状态和词表预测得分（Logits）。

在 Teacher Forcing 模式下，即使模型在第 $t-1$ 步对目标词的预测概率很低，在第 $t$ 步计算时，送入模型的输入依然是数据集里正确的真实 Token。这使得整条序列的损失和梯度可以在一次前向和反向传播中并行处理完成。

---

### 2. 为什么主流 SFT 通常只对 Response 计算损失？

在预训练阶段，模型学习的是文本的无条件语言概率分布，需要对整个文本序列计算交叉熵损失：

$$\mathcal{L}_{\text{Pretrain}} = - \frac{1}{L} \sum_{t=1}^{L} \log P(x_t \mid x_{<t}; \theta)$$

但在指令微调中，输入序列明确由两部分组成：前半部分的提示上下文 Prompt（包含指令、背景材料、历史对话，长度记为 $L_{\text{prompt}}$），以及后半部分的目标回答 Response（长度记为 $L_{\text{response}}$），总长度 $L = L_{\text{prompt}} + L_{\text{response}}$。

在大部分指令对齐任务中，通常选择只对目标回答（Response）部分计算损失。如果在此类任务中仍对全序列（包含 Prompt）计算损失，往往会面临两个工程层面的权衡考量：

![输入序列中的因果注意力掩码与损失计算掩码分工机制](/images/content-rebuild/b01bbf140237253a/fig-2.png)

#### 2.1 任务目标与输出分布的因果解耦
微调的业务目标是让模型掌握**条件生成概率** $P(\text{Response} \mid \text{Prompt})$，即在用户给出特定上下文时，输出符合预期的答案。模型不需要去拟合用户提问的先验概率 $P(\text{Prompt})$。真实业务中用户的提问千奇百怪，如果对 Prompt 计算损失，反向传播的梯度会强迫模型去迎合、记忆输入语句的行文习惯，占用模型的参数更新幅度，甚至削弱预训练模型原有的通用理解能力。

#### 2.2 长上下文下的有效梯度稀释
在 RAG（检索增强生成）或长文档提取场景中，Prompt 的长度可能达到 3000 Token，而目标 Response 往往只有 50 Token。如果全序列都参与损失计算，Prompt 部分的梯度将在总损失中占据超过 98% 的权重。期望的答案信号会被大量无关的输入文字稀释，导致微调收敛缓慢甚至完全学不会目标格式。

因此，主流指令微调通常将损失目标聚焦于条件生成概率：

$$\mathcal{L}_{\text{SFT}} = - \frac{1}{L_{\text{response}}} \sum_{t=L_{\text{prompt}}+1}^{L} \log P(x_t \mid x_{<t}; \theta)$$

这里需要精确区分两个核心掩码的作用范围与梯度流向：
- **因果注意力掩码（Causal Attention Mask）**：在 Transformer 注意力层生效，限制信息流向。Response 的 Token 可以通过注意力关注 Prompt 的上下文信息。
- **损失掩码（Loss Mask / Label Mask）**：在计算最终交叉熵损失时生效，将 Prompt 预测下一个 Prompt 词的位置标签设为 `-100`。需要注意：Prompt 部分不计入损失，仅意味着 Prompt 的自回归预测不产生直接的交叉熵损失分量；在反向传播时，Response 部分产生的损失依然会通过自注意力网络回传至 Prompt 对应位置的隐藏状态与输入嵌入层。

<ReadMoreLock />

---

### 3. PyTorch 中 labels 掩码（-100）与错位切片实现

在 PyTorch 与 Hugging Face Transformers 的自回归模型（如 `AutoModelForCausalLM`）中，损失计算统一使用 `-100` 作为忽略标记。

#### 3.1 底层切片与对齐机制
PyTorch 交叉熵损失函数声明如下：
```python
torch.nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean')
```
参数 `ignore_index` 的默认值就是 `-100`。计算损失时，真实标签为 `-100` 的位置会被自动跳过，其损失值为 0，且在做均值平均（`reduction='mean'`）时，分母中的有效 Token 总数也不会包含这些位置。

在因果语言模型的前向传播实现中，模型需要将预测的 Logits 与真实标签进行**错位切片（Shift）**对齐：
```python
# 假设前向输入模型后：
# logits 形状: [batch_size, seq_len, vocab_size]
# labels 形状: [batch_size, seq_len]

# 位置 t 输出的隐藏状态用于预测位置 t+1 的真实 Token，因此需要做错位切片
shift_logits = logits[..., :-1, :].contiguous()  # 取索引 0 到 seq_len-2
shift_labels = labels[..., 1:].contiguous()   # 取索引 1 到 seq_len-1

loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
```

![PyTorch 下 labels 错位切片与 -100 忽略索引的张量计算过程](/images/content-rebuild/b01bbf140237253a/fig-3.png)

#### 3.2 算例拆解：错位切片如何对齐预测与标签
以贯穿示例的简化 Token 序列进行对照。假设分词后：
- Prompt 对应 2 个 Token：`[101, 102]`（代表“提取单号”）
- Response 对应 2 个 Token：`[201, 202]`（代表“SF101”和 `<eos>`）
- 拼接后的完整输入：`input_ids = [101, 102, 201, 202]`
- 构造的标签序列：`labels = [-100, -100, 201, 202]`

在模型内部，模型输出 4 个位置的 Logits（记为 $L_0, L_1, L_2, L_3$）。经过切片操作：
- `shift_logits = [L_0, L_1, L_2]`（长度为 3，分别是由 Token `101`, `102`, `201` 产生的前向输出）
- `shift_labels = labels[1:] = [-100, 201, 202]`（长度为 3）

具体的对齐计算如下表：

| 切片位置索引 | 当前输入的 Token | 对应 Logits | 对齐的目标标签 (`shift_labels`) | 损失计算处理 |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `101` (Prompt 1) | $L_0$ | `-100` (原 Prompt 2) | 命中 `ignore_index`，**不计入损失，无梯度回传** |
| 1 | `102` (Prompt 2) | $L_1$ | `201` (Response 1) | **正常计算交叉熵，产生有效梯度** |
| 2 | `201` (Response 1) | $L_2$ | `202` (Response 2) | **正常计算交叉熵，产生有效梯度** |

这个表格清晰地展现了关键逻辑：
1. 用 Prompt 的第一个词预测 Prompt 的第二个词时，目标标签是 `-100`，梯度被阻断；
2. **用 Prompt 的最后一个词预测 Response 的第一个词时**（位置 1），输入是 `102`，目标标签是 `201`，此处的损失必须计算。因为这正是模型从“接收指令”转向“生成回答”的起点；
3. 后续 Response 内部的词逐个自回归预测，正常产生梯度。

#### 3.3 数据处理实现代码
在制作训练集时，我们需要将 Prompt 对应的标签区域填充为 `-100`。以下为教学原理演示代码（生产中多使用 `apply_chat_template` 统一处理对话模板与分词边界）：

```python
import torch

def build_sft_example(prompt_text: str, response_text: str, tokenizer, max_length: int = 512):
    # 1. 编码 Prompt 与 Response（注意不同分词器对前导空格及模板标记的约定）
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    
    # 2. 为目标回答追加终止符 EOS
    if tokenizer.eos_token_id is not None:
        response_ids.append(tokenizer.eos_token_id)
        
    # 3. 拼接完整的模型输入序列与 labels 张量
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    
    # 4. 长度截断与全 -100 防御
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        # 边界防御：若 Prompt 过长导致截断后无有效 Response 目标，避免 CrossEntropyLoss 返回 NaN
        if all(label == -100 for label in labels):
            raise ValueError("截断后序列无有效预测标签，请调大 max_length 或过滤超长 Prompt。")
        
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }
```

---

### 4. 批次训练排布：动态填充与序列拼接（Sample Packing）

不同样本的文本长度往往参差不齐。在组织成 Mini-Batch 供 GPU 训练时，有两种排布策略：

![动态填充与序列拼接在 Batch 维度的排布结构对比](/images/content-rebuild/b01bbf140237253a/fig-4.png)

#### 4.1 动态填充（Dynamic Padding）
- **静态填充（Static Padding）**：在数据预处理阶段，把所有样本全部用 `[PAD]` 补齐到全局最大长度（例如 4096）。如果样本的平均长度只有 400，那么每批数据里超过 90% 的计算量都在对无意义的 `[PAD]` 进行矩阵乘法，造成显存和算力的巨大浪费。
- **动态填充（Dynamic Padding）**：通过自定义 `DataLoader` 的 `collate_fn`，每次只根据当前批次内最长样本的长度进行补齐。如果某批数据中最长只有 600，则该批次只对齐到 600。这大幅削减了批次内的无效填充量。

#### 4.2 序列拼接（Sample Packing）
在大量短文本训练场景下，动态填充依然存在浪费。例如一条 2000 Token 的长样本和多条 200 Token 的短样本分在同一个批次，短样本依然要补齐大量的 `[PAD]`。

**Sample Packing（样本拼接）**策略将多条独立的样本在序列维度首尾相接，拼入固定的最大长度（例如 4096），大幅减少批次内用于补齐的 `[PAD]` 占位符，从而显著提升 GPU 计算资源的有效利用率（具体提升幅度取决于装箱填充率）。

但是，直接拼接多条样本会导致严重的**上下文跨样本污染（Cross-Sample Attention Contamination）**：排在后面的样本 B 会通过自注意力机制关注到前面的样本 A，导致模型错误地将无关样本视作上下文。

![序列拼接场景下的多文档独立注意力掩码与位置编码重置](/images/content-rebuild/b01bbf140237253a/fig-5.png)

要解决跨样本污染，必须配合两项机制：
1. **块对角因果注意力掩码（Block-Diagonal Attention Mask）**：
   将注意力矩阵约束在各自样本的区间内。样本 A 的 Token 只能在样本 A 内部做下三角因果注意力；到了样本 B 时，注意力权重被物理阻断，无法查看样本 A 的任何内容。在工程实现中，FlashAttention 提供了变长序列接口（如 `flash_attn_varlen_func`），只需传入记录每个样本累积边界的一维张量 `cu_seqlens`（如 `[0, 500, 1200, 4096]`），无需显式构建巨大的 2D 稀疏矩阵，直接在底层的 CUDA 算子中限制注意力范围。
2. **位置编码（Position IDs）排布**：
   在拼接长序列中，各框架常将后续样本的 `position_ids` 重新从 0 开始计数（即 `[0..499, 0..699, ...]`）。需要注意：对于采用旋转位置编码（RoPE）的模型，若跨样本注意力已被分块掩码完全阻断，样本内部任意两词的相对距离 $(m - n)$ 在整体平移下保持不变，因此重置并非维持 RoPE 相对距离的数学前提；但在实践中重置 `position_ids` 可以防止拼接总长度越过预训练时设定的单样本上下文窗口，并兼容绝对位置编码模型。

---

### 5. 训练稳定性控制：梯度累积、混合精度与学习率预热

大语言模型训练对显存占用要求极高。单张 GPU 的物理显存通常只能容纳很小的批次大小（如 `per_device_train_batch_size = 1 ~ 4`）。批次过小会导致梯度估计方差剧烈，训练难以平稳收敛。

![SFT 训练循环中的梯度累积与混合精度数据流](/images/content-rebuild/b01bbf140237253a/fig-6.png)

#### 5.1 梯度累积（Gradient Accumulation）
梯度累积通过在多个微批次（Micro-Batch）上连续累加梯度，延缓参数更新的频率，从而在有限物理显存下模拟大批次训练的效果：

$$\text{等效总 Batch Size} = \text{单卡 Micro-Batch Size} \times \text{梯度累积步数 } K \times \text{GPU 卡数}$$

在代码实现中，常规做法是在反向传播前将当前 Micro-Batch 损失除以累积步数 $K$ 进行缩放。需要注意：若各 Micro-Batch 中参与损失计算的有效 Token 数量不均等，直接除以 $K$ 累加属于各批次等权近似，严格等价的做法需按各批次有效 Token 数进行全局加权；此外在训练循环结束时需处理不足 $K$ 步的收尾更新：
```python
optimizer.zero_grad()
for step, batch in enumerate(dataloader):
    outputs = model(**batch)
    # 各 Micro-Batch 均值损失除以 K 进行尺度缩放
    loss = outputs.loss / accumulation_steps
    loss.backward()
    
    if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防发散
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```
如果漏掉了 `loss / accumulation_steps`，反向传播累加的梯度尺度将被人为放大 $K$ 倍。虽然在 AdamW 等自适应优化器中二阶矩缩放使更新步长不会简单线性放大 $K$ 倍，但这会破坏动量估计与权重衰减比例，极大增加数值溢出风险。

#### 5.2 混合精度选择：FP16 与 BF16
现代微调普遍采用 16 位混合精度训练，以降低显存占用并调用 GPU 的 Tensor Core 单元：
- **FP16（半精度浮点）**：由 1 位符号位、5 位指数位、10 位尾数位组成。其动态数值范围较窄（最大约 65504，最小正规数约 $6.10 \times 10^{-5}$，最小非正规数约 $5.96 \times 10^{-8}$）。在微调深层网络时，浅层回传的微小梯度容易发生数值下溢归零。因此，FP16 训练通常**配合动态损失缩放（Dynamic Loss Scaler）**：在前向传播后将 Loss 放大 $2^N$ 倍，带动反向传播的梯度落在非零有效区间，在优化器更新前再除回对应倍数。
- **BF16（Bfloat16）**：由 1 位符号位、8 位指数位、7 位尾数位组成。其指数位与 FP32 一致，动态范围上限约 $3.39 \times 10^{38}$，常规梯度下极难发生下溢，因此通常**无需配置 Loss Scaler**，训练配置比 FP16 更为简便。
- **显存与优化器状态认知**：使用混合精度并不代表显存里全部都是 16 位浮点数。以标准的混合精度 AdamW 优化器为例，为了保证参数累积微调时的数值精度，主流实现通常会额外维护一份 **FP32 主权重（Master Weights）** 以及 **FP32 动量与方差状态**。模型前向与反向传播在 16 位下计算，参数在 32 位精度下更新后，再同步回 16 位模型权重。

#### 5.3 学习率预热（Warmup）与衰减
在微调刚开始的几十到几百步内，模型的输出分布尚未适应当前任务的指令格式，初始的损失往往偏大，产生的梯度波动剧烈。如果一开始就使用设定的峰值学习率，容易破坏预训练模型原有的通用特征表示。
- **Warmup（预热）**：通常设置总训练步数的 $3\% \sim 10\%$ 作为预热阶段。学习率从 0 线性增加到设定的峰值（全参数微调通常设为 $1\times 10^{-5} \sim 5\times 10^{-5}$，LoRA 通常设为 $1\times 10^{-4} \sim 3\times 10^{-4}$）。
- **Cosine Decay（余弦衰减）**：预热结束后，学习率按照余弦曲线平滑降低至最小学习率（通常为峰值的 $10\%$），帮助模型在训练后期精细收敛。

---

### 6. 可检验练习：验证错位切片与掩码梯度行为

为了检验自己对 SFT 错位切片和掩码机制的理解，可以在本地运行以下纯 PyTorch 验证脚本。该脚本模拟了一个简易序列的前向与反向传播，验证梯度是否精准停在指定位置：

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
vocab_size = 10
seq_len = 5

# 模拟模型输出的 Logits（需开启梯度追踪）
# 假设输入包含 2 个 Prompt Token 与 3 个 Response Token
logits = torch.randn(1, seq_len, vocab_size, requires_grad=True)

# 构造包含 -100 掩码的真实标签
# 前 2 个位置是 Prompt，后 3 个位置是 Response 目标词（词表中 ID 分别为 3, 5, 2）
labels = torch.tensor([[ -100, -100, 3, 5, 2 ]], dtype=torch.long)

# 1. 模拟 Causal LM 的错位切片逻辑
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()

# 2. 计算损失
loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
loss.backward()

print(f"计算得到的 SFT 损失值: {loss.item():.4f}")

# 3. 检查 logits 各位置的梯度绝对值和
grad_pos0 = logits.grad[0, 0].abs().sum().item()
grad_pos1 = logits.grad[0, 1].abs().sum().item()
grad_pos2 = logits.grad[0, 2].abs().sum().item()

print(f"位置 0 梯度 (Prompt 内部预测): {grad_pos0:.4f}")
print(f"位置 1 梯度 (Prompt 末尾预测 Response 首词): {grad_pos1:.4f}")
print(f"位置 2 梯度 (Response 内部预测): {grad_pos2:.4f}")

# 验证断言
assert grad_pos0 == 0.0, "错误：Prompt 内部不应产生任何反向传播梯度！"
assert grad_pos1 > 0.0, "错误：从 Prompt 转向 Response 的起点必须产生梯度！"
assert grad_pos2 > 0.0, "错误：Response 内部目标必须产生梯度！"
print("\n验证通过：损失掩码与切片机制严格生效，仅目标条件区间参与梯度更新。")
```

完成全参数指令微调机制的学习后，我们会发现更新全部参数需要耗费巨大的显存来保存优化器状态。在计算资源有限的条件下，如何在几乎不损失性能的前提下高效微调？下一讲我们将深入参数高效微调的代表方案：[Adapter、Prefix Tuning、P-Tuning 有什么区别？](/llm-100/finetuning/adapter-prefix-ptuning.md) 与 [LoRA 为什么能高效微调大模型？](/llm-100/finetuning/lora-efficient-finetuning.md)。
