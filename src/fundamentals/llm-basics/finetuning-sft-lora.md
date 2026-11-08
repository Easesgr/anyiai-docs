---
title: 微调是什么？SFT、LoRA 与 QLoRA 分别解决什么问题？
icon: mdi:tune-vertical
tag:
- 大模型基础
- 微调
- LoRA
description: 拆解大模型微调机制，对比 SFT、LoRA 与 QLoRA 的显存构成、低秩分解与量化计算原理，明确行为对齐与知识更新的工程边界。
---

![模型微调机制核心全景](/images/content-rebuild/d8f1890a88d236c2/cover.png)

开源大语言模型在刚完成预训练时，本质上是一台“互联网文本续写机”。如果向一个 7B（70 亿参数）的预训练基座输入：

```text
请将以下用户反馈整理为工单 JSON：账号无法登录，提示密码错误三次。
```

基座模型很可能不会输出结构化 JSON，而是顺着上文继续续写：“客服人员随后联系了该用户”、“或者提示用户点击忘记密码”，甚至自顾自地编造出一段新的论坛讨论。这是因为预训练的目标只是最大化下一个 Token 的预测概率，模型还没有学会“遵循用户指令并按照特定格式输出”。

微调正是为了解决这个问题。本文通过这套工单结构化任务，解释从监督微调（SFT）到参数高效微调（LoRA、QLoRA）的运作机制、显存开销拆解，以及微调在实际系统中的适用边界。

---

### 监督微调（SFT）：通过损失掩码建立指令遵循习惯

预训练阶段，模型学习的是全网语料的自回归概率分布：给定前面的词，预测下一个词。而在监督微调（Supervised Fine-Tuning, SFT）中，训练样本被组织为结构化的会话格式：

```text
<|im_start|>system
你是一位工单分类助手，请将用户问题解析为固定格式的 JSON。<|im_end|>
<|im_start|>user
账号无法登录，提示密码错误三次。<|im_end|>
<|im_start|>assistant
{"category": "auth", "sub_issue": "password_lock"}<|im_end|>
```

如果直接对整段文本计算交叉熵损失，模型不仅在学习如何输出目标 JSON，也在学习“预测用户下一句会问什么”。这对于模型成为特定任务的助手是没有必要的，甚至会干扰训练效果。

![SFT 指令与回复的损失掩码计算流](/images/content-rebuild/d8f1890a88d236c2/fig-1.png)

因此，SFT 引入了损失掩码（Loss Masking）机制。在一次前向传播中，模型虽然读入了包括 System Prompt、User Query 与 Assistant Response 在内的全部 Token，但在计算交叉熵损失时，系统指令和用户输入对应的损失被直接置零（Masked）。反向传播的梯度仅仅由 Assistant 回复部分的 Token 产生。

通过这种方式，损失函数聚焦在条件概率 $P(Y_{\text{Response}} \mid X_{\text{Prompt}})$ 上。模型被收敛到特定的角色设定、回答风格与输出模式中。

---

### 全量微调的显存瓶颈：为什么 24GB 显卡跑不动 7B 模型？

一个采用 16 位浮点数（FP16 或 BF16）存储的 7B 模型，参数文件体积约为 14 GB。不少开发者会疑惑：为什么一张 24 GB 显存的显卡（如 RTX 3090 或 4090），连 Batch Size 为 1 的全量微调都无法运行？

这是因为在训练状态下，显存并非只存放模型参数，而是包含四大组成部分：

![全量微调下显存构成的分层堆叠](/images/content-rebuild/d8f1890a88d236c2/fig-2.png)

#### 1. 静态参数与梯度
在 16 位混合精度训练中：
*   **模型参数（Weights）**：每个参数占 2 字节（Bytes），7B 模型约为 $7 \times 10^9 \times 2 \approx 14 \text{ GB}$。
*   **反向传播梯度（Gradients）**：每个可训练参数都需要保存对应的梯度值，同样为 16 位表示，占用 $14 \text{ GB}$。

#### 2. 优化器状态（Optimizer States）
训练大模型普遍采用 AdamW 优化器。为了防止更新步长出现浮点下溢或舍入误差，AdamW 必须维护 32 位单精度（FP32，每个数值 4 字节）的状态副本：
*   **FP32 主权重副本**：$4 \text{ Bytes} \times N$
*   **一阶动量（梯度的均值）**：$4 \text{ Bytes} \times N$
*   **二阶动量（梯度的方差）**：$4 \text{ Bytes} \times N$

每个参数在优化器中需要占用 $4 + 4 + 4 = 12 \text{ 字节}$。对于 7B 模型，仅优化器状态就需要 $7 \times 10^9 \times 12 \approx 84 \text{ GB}$。

三项静态显存相加：$14 + 14 + 84 = 112 \text{ GB}$，相当于模型参数本体体积的 8 倍。这还没有计入前向传播储存的中间激活值（Activation Memory）以及算子运算临时显存。面对上百吉字节的开销，单张消费级显卡无法承受全量微调，参数高效微调（PEFT）技术因此成为主流。

<ReadMoreLock />

---

### LoRA 原理：低秩矩阵分解与零推理延迟

LoRA（Low-Rank Adaptation）的核心观察是：模型在适应下游任务时，参数更新量 $\Delta W$ 并不需要在整个高维参数空间中自由变化，其有效变化往往集中在一个低维子空间中，具有较低的内在秩（Intrinsic Rank）。

![LoRA 旁路低秩矩阵分解与推理权重合并机制](/images/content-rebuild/d8f1890a88d236c2/fig-3.png)

#### 矩阵分解过程
设基座模型某一层线性变换的原权重矩阵为 $W_0 \in \mathbb{R}^{d \times k}$。在全量微调中，输入 $x$ 经过更新后的计算为：

$$h = (W_0 + \Delta W) x$$

LoRA 选择直接冻结预训练权重 $W_0$（不计算梯度，也不分配优化器状态），并将更新量 $\Delta W$ 分解为两个小矩阵的乘积：

$$\Delta W = \frac{\alpha}{r} (B \cdot A)$$

其中 $A \in \mathbb{R}^{r \times k}$，$B \in \mathbb{R}^{d \times r}$，设置的秩 $r \ll \min(d, k)$（通常取 8、16 或 64）。前向传播变为：

$$h = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$

超参数 $\alpha$ 是一个常数缩放因子。当我们在实验中调整低秩维度 $r$ 时，通过维持 $\frac{\alpha}{r}$ 的比例，可以相对稳定梯度更新的尺度，降低调参联动成本。

#### 初始化保证平滑起步
如果矩阵 $A$ 和 $B$ 都采用随机高斯初始化，那么在训练开始的第一步，$\Delta W = B \cdot A$ 会产生非零的随机扰动，破坏预训练模型原有的表征能力。LoRA 采用非对称初始化规则：
*   矩阵 $A$ 采用高斯分布随机初始化：$A \sim \mathcal{N}(0, \sigma^2)$
*   矩阵 $B$ 全部初始化为 0：$B = 0$

在训练的初始状态下，$\Delta W = 0 \cdot A = 0$。模型初始输出严格等于 $W_0 x$，保证了微调从原基座的能力点平稳启动。

#### 推理阶段的无损合并
LoRA 的另一个工程优势在于推理部署。训练完成后，根据矩阵乘法的分配律，我们可以直接把两个小矩阵相乘并加到基座权重中：

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} (B \cdot A)$$

合并后的权重结构与原基座完全相同，在线上推理服务中不会增加额外的矩阵分支计算，没有任何推迟或额外延迟。官方 PEFT 库通过 `model.merge_and_unload()` 即可完成这一操作。

---

### QLoRA 机制：NF4 量化与计算精度流动

LoRA 虽然把可训练参数压缩到了原模型的 0.1% 到 1%，省去了绝大多数优化器显存，但冻结的基座权重 $W_0$ 依然以 16 位浮点数格式驻留在显存中（7B 占 14 GB，70B 占 140 GB）。单张 24 GB 显卡依然无法微调更大规模的模型。QLoRA 针对这一瓶颈提出了三项工程改进。

![QLoRA 计算过程中的解量化与数据流动](/images/content-rebuild/d8f1890a88d236c2/fig-4.png)

#### 1. 4-bit NormalFloat (NF4) 数据类型
普通的 INT4 量化假设数据在数值区间内是均匀分布的。然而经过预训练和层归一化后，大模型的权重分布通常呈现均值为 0 的正态分布形态。NF4 利用这一统计先验，通过正态分布的反累积分布函数划分出 16 个非均匀分布的离散分位数。这使得每个量化桶内落入的权重数量大致均等，在 4 位存储下最大程度减少了量化失真。

#### 2. 双重反量化（Double Quantization）
为了进一步控制量化误差，算法会以块（Block）为单位（例如每 64 个参数为一组）记录量化缩放常数（Scale）。这些常数本身（通常为 32 位浮点数）会带来额外的显存开销。双重反量化将这些缩放常数再次进行一次 8 位量化（块大小设为 256），把量化常数带来的开销从每个参数约 0.5 bit 压缩到 0.127 bit 左右。

#### 3. 分页优化器（Paged Optimizers）
在处理长上下文训练时，前向传播的激活显存可能会瞬间冲顶诱发 OOM（显存溢出）。QLoRA 基于 CUDA 统一内存机制，在显存紧张时将 LoRA 优化器内部的状态数据换出到 CPU 内存中，待梯度反向计算执行完毕后再按页调回显存，提高了长周期训练的容错率。

#### 计算时的精度流动逻辑
一个常见的理解误区是“QLoRA 使用 4-bit 完成矩阵乘法”。实际计算流向如下：
*   **显存存储态**：基座权重以 4-bit NF4 格式冻结在显存中（7B 模型仅占约 3.8 GB）。
*   **运算展开态**：当输入向量 $x$ 到达某一层时，计算单元按块将该层的 4-bit NF4 权重动态反量化（Dequantize）恢复为 16-bit 浮点数（BF16/FP16），与输入向量 $x$ 执行正常的浮点矩阵乘法。
*   **旁路分支**：LoRA 适配器 $A$ 与 $B$ 始终以 16-bit 浮点数保存，负责累加梯度并更新。

这种“低精度显存存储、高精度计算还原”的模式，使单张 16 GB 甚至 24 GB 显卡微调 7B 乃至更大模型成为可能。

---

### 显存算例对比与长序列分析

以 7B 模型（输入序列长度 2048，Batch Size = 2）为例，开启标准梯度检查点（Gradient Checkpointing）时的训练显存占用估算如下：

![不同微调技术下的显存对比与随长度变化规律](/images/content-rebuild/d8f1890a88d236c2/fig-5.png)

| 显存构成项目 | 全量微调 (Full SFT BF16) | LoRA (BF16 基座 + All-linear) | QLoRA (NF4 基座 + All-linear) |
| :--- | :--- | :--- | :--- |
| **基座权重** | 14.0 GB (参与更新) | 14.0 GB (冻结) | **约 3.8 GB** (4-bit 冻结) |
| **梯度与适配参数** | 14.0 GB (仅梯度显存) | 约 0.3 GB (适配器权重与梯度) | 约 0.3 GB (适配器权重与梯度) |
| **AdamW 优化器状态** | 84.0 GB (FP32 全量) | 约 0.9 GB (仅对应 LoRA) | 约 0.9 GB (仅对应 LoRA) |
| **激活显存 (动态)** | 约 8.0 GB (序列长 2048 估算) | 约 8.0 GB (序列长 2048 估算) | 约 8.0 GB (序列长 2048 估算) |
| **总计显存预估** | **> 120.0 GB** | **约 23.2 GB** | **约 13.0 GB** |
| **硬件门槛** | 多卡 A100/H800 集群 | 单卡 RTX 3090/4090 (24GB) | 单卡 RTX 3060/4070 (16GB) |

*注：激活显存取决于具体模型架构、注意力实现（如是否开启 FlashAttention）及重算策略，上表数值为标准设置下的工程示意估算。*

#### 为什么 LoRA 无法缩减激活显存？
LoRA 极大地减少了模型参数的梯度与优化器状态，但并没有改变 Transformer 骨干网络的前向计算结构。在反向传播计算梯度时，为了把误差信号传导至各个并联的 LoRA 模块，自注意力层生成的注意力得分矩阵以及前馈网络（FFN）中间激活值仍必须完整保留。因此，当训练序列长度增长时，激活显存依然会显著增加。控制上下文长度或引入 FlashAttention，依然是参数高效微调中控制显存的关键手段。

以下是使用 Hugging Face `peft` 与 `bitsandbytes` 库构建 QLoRA 模型的典型配置：

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. 设置 4-bit NF4 量化与双重量化配置
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 2. 加载基座模型并准备进行 k-bit 训练
base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-base",
    quantization_config=quant_config,
    device_map="auto"
)
base_model = prepare_model_for_kbit_training(base_model)

# 3. 配置 LoRA 模块（作用于所有线性层以保持充足的表征空间）
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# 终端显示可训练参数占比通常在 0.1% ~ 0.4% 之间
```

---

### 微调的适用边界：行为对齐 vs 事实更新

回到最初的工单分类任务：如果企业增加了十类新的退款政策，或者某个接口的参数定义每周都在调整，我们应该通过微调把这些业务事实“灌入”大模型吗？

答案通常是否定的。在工程架构设计中，必须分清微调与外挂检索的分工边界。

![微调与检索在任务特性上的分工决策边界](/images/content-rebuild/d8f1890a88d236c2/fig-6.png)

#### 微调擅长解决的问题（行为模式与格式规范）
1.  **特定输出格式约束**：例如强制模型输出符合特定 JSON Schema 或 XML 标签的响应，避免解析报错。
2.  **角色风格与交互规范**：例如模仿特定客服的语调口吻，或遵循特定的思维链（CoT）推理步骤。
3.  **内化专用领域协议**：例如学习特定领域的代码框架调用逻辑，提升结构化工具调用（Tool Calling）的参数填充稳定性。

#### 为什么不适合通过微调灌注高频事实？
1.  **灾难性遗忘（Catastrophic Forgetting）**：强行通过梯度微调写入特定事实，容易破坏预训练阶段学到的通用推理与常识联想能力。
2.  **高置信度幻觉**：微调样本数量（通常几千到几万条）远小于预训练语料。模型只学到了实体的字词表面共现，难以建立多视角的底层约束，容易以极其流畅的口吻生成看似真实的虚假事实（详见[大模型幻觉分析](/fundamentals/llm-basics/hallucination.md)）。
3.  **无法精确撤回与更新**：如果某项产品定价发生变化，微调无法像数据库 `UPDATE` 或 `DELETE` 语句那样精准修改某个权重。每次更新都需要重新打包数据与训练，迭代周期长且不可控。
4.  **缺乏证据溯源凭据**：微调模型给出的答案是权重概率采样的结果，无法直接标注该结论出自某篇文档的某一行，无法满足严肃业务场景对可审计性的要求。

因此，合理的工程原则是：**用微调塑造行为（格式约束、交互习惯、角色语气），用上下文供给事实（借助 [RAG 检索增强](/interview/rag-interview/rag-vs-finetuning.md) 或工具链调用注入最新信息）。**

---

### 延伸阅读与知识导航

微调技术体系与落地权衡可进一步参考以下内容：
*   关于 LoRA 矩阵秩 $r$ 与缩放因子 $\alpha$ 的工程选取技巧，请阅读 [LoRA 超参数设置经验法则](/interview/llm-interview/lora-hyperparams.md)。
*   关于内在维度假说与低秩适应的数学推导，请参考 [LoRA 数学原理推导](/interview/llm-interview/lora-math-principle.md)。
*   关于业务场景中何时微调、何时采用外挂知识库的完整选型对比，请参考 [RAG 与微调的区别与选型指南](/interview/rag-interview/rag-vs-finetuning.md)。
