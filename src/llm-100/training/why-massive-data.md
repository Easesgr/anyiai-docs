---
title: 为什么大模型训练需要海量数据？从扩展定律到数据治理
icon: mdi:tune-vertical
tag:
- 训练原理
- 大模型100讲
- 预训练
description: 从 Chinchilla 扩展定律、齐普夫长尾分布、语料配比与清洗管线四个维度，拆解大模型预训练依赖数万亿 Token 的技术原因与工程权衡。
---

如果只是为了让模型学会基本的语法规则，数千万字的现代汉语词典和语法书已经足够涵盖几乎所有句式结构。但在实际工程中，训练一个 7B（70 亿参数）到 70B 的大语言模型，消耗的数据量动辄从 1 万亿（1T）跨越到 15 万亿（15T）Token。

为什么参数量增加必须伴随数据量同比例甚至超比例增加？为什么不能用少量精选文本循环训练几十遍？

一个贯穿本文的核心结论是：**大模型预训练本质上是在极高维度的参数空间中，通过自监督预测下一个 Token（Next-Token Prediction）来建立对现实世界知识和推理逻辑的条件概率建模。模型不仅要学习“通顺的语法”，还要记住海量的长尾事实，并建立稳固的上下文推理关联。这既受制于计算资源的分配效率（Chinchilla 扩展定律），也取决于自然语言中知识出现的统计规律（齐普夫定律）。**

![大模型预训练海量数据工程全景](/images/content-rebuild/5d59c67f192bc620/cover.png)
*图释：大模型预训练依赖的海量数据并非杂乱堆砌，而是经过严格的尺度定律约束、流形覆盖优化、多模态配比设计与多层级清洗管线提炼而成的高密度知识载体。*

---

### 1. 计算最优扩展规律：Chinchilla 比例与“过度训练”

大模型训练消耗的算力通常用浮点运算次数（FLOPs）衡量。对于基于标准 Transformer 架构的因果自回归模型，在序列长度处于常见范围（即自注意力平方项开销远小于前馈层和注意力线性投影的矩阵乘法）时：
* 每个 Token 的前向传播大约需要 $2N$ 次浮点运算（每个参数对应一次乘法和一次加法，不计词表嵌入等小比例参数）；
* 反向传播计算梯度时约为前向传播的两倍（约 $4N$ 次浮点运算）。

因此，训练一个非词表参数量为 $N$、处理 $D$ 个 Token 的模型，总算力预算 $C$ 可近似估算为：

$$C \approx 6ND$$

在 2020 年，Kaplan 等人（OpenAI）提出的早期扩展规律（Scaling Law）认为，模型性能对参数量 $N$ 的依赖远大于对数据量 $D$ 的依赖，主张增加算力时应优先扩大参数量，数据量只需缓慢增长。

但在 2022 年，Hoffmann 等人（DeepMind）指出了该实验中学习率调度策略导致的偏差。在训练并评估了 400 多个模型后，他们提出了 **Chinchilla 扩展定律**：在固定总计算预算 $C$ 的条件下，要使最终的预训练损失最小，**参数量 $N$ 与训练 Token 量 $D$ 应当以大致相等的比例同步缩放**：

$$N \propto C^{0.5}, \quad D \propto C^{0.5}$$

根据拟合经验常数，在计算最优条件下，参数量与 Token 量的比例大致为 $1 : 20$：

$$D_{\text{optimal}} \approx 20 \times N$$

![Chinchilla 计算最优曲线与等计算量权衡](/images/content-rebuild/5d59c67f192bc620/fig-1.png)
*图释：在固定的算力预算等高线（IsoFLOP Profile）下，存在一个使预训练验证损失最小的最优平衡点。图中虚线展示了计算最优路径，偏移该路径（如过大模型配过少数据）均会导致算力利用效率的下降。*

#### 算例：计算最优与推理部署成本的权衡

假设我们准备训练一个 $N = 7\text{B}$（$7 \times 10^9$）参数的模型：

1. **计算最优数据量**：
   按照 Chinchilla 比例：
   $$D_{\text{optimal}} \approx 20 \times (7 \times 10^9) = 140 \times 10^9 = 140\text{B Tokens}$$
   对应消耗的总算力为：
   $$C \approx 6 \times (7 \times 10^9) \times (140 \times 10^9) = 5.88 \times 10^{21} \text{ FLOPs}$$

2. **工业界的“过度训练”（Overtraining / Inference-optimal）策略**：
   查阅 Meta 发布的 LLaMA-1/2/3 技术报告可以发现，其 7B/8B 模型实际训练了 1T 到 15T 的 Token，远超 140B 的理论计算最优值。
   这并非违背了扩展规律，而是工程目标的变化：
   * **Chinchilla 目标**：在**单次预训练计算预算固定**的前提下，让训练结束时的损失最低。
   * **产品部署目标**：追求**模型全生命周期内的总成本最低**（训练成本 + 上千万次线上推理成本）。
   * 较小尺寸的模型在推理时显存占用少、吞吐量高、延迟低。因此，用远超计算最优比例的数据（如 10 到 100 倍）对 7B 模型进行“过度训练”，能让小参数模型吸收更多信息，从而在部署端享受小模型的低延迟与低成本。

<ReadMoreLock />

---

### 2. 长尾事实分布与记忆容量平衡

为什么不能拿 1 亿 Token 的精选教科书，在 7B 模型上循环迭代 1000 次来达到等效的训练步数？这由自然语言的分布特征与神经网络的拟合行为共同决定。

#### 齐普夫定律（Zipf's Law）与长尾现象
自然语言中的词汇与事实实体分布呈现强烈的齐普夫幂律特征：一个词或实体的出现频率 $f$，与其在频率排序中的名次 $r$ 近似成反比：

$$f(r) \propto \frac{1}{r^s} \quad (s \approx 1)$$

* **头部内容**：常见虚词、基础句式、通用连接词出现极其频繁。模型在训练几十亿 Token 时就已经能熟练掌握语法和常用表达。
* **长尾内容**：具体的函数接口说明、专业学术定理、历史事件细节、特定领域术语等，在整个互联网语料库中的出现频率极低。

![Zipf 定律与高维流形的长尾覆盖机制](/images/content-rebuild/5d59c67f192bc620/fig-2.png)
*图释：自然语言知识呈现典型的齐普夫（Zipf）幂律长尾分布。头部高频语法与词汇仅占知识流形极小区域，海量的专业知识、实体关系与复杂推导均隐藏在极低频率的尾部。只有极大规模的语料才能提供足够的梯度更新信号来锚定长尾流形。*

如果预训练数据仅有百亿 Token，大量长尾专业知识在整个数据集中出现的次数可能只有 0 到 1 次。因果自回归模型依赖上下文条件概率转移权重 $P(w_t \mid w_{<t})$，样本出现频次太低时，优化器难以让对应的关联权重稳定收敛。

海量语料的一个核心作用，就是**把长尾事实和低频表达的绝对出现频次，提升到优化器能有效学习的阈值之上**。

#### 参数容量与过拟合
7B 模型拥有 70 亿个浮点参数，具有极高的拟合能力。如果训练集规模过小（例如仅数千万 Token），模型很快会通过死记硬背表面词序将训练损失降到极低，而无法学到更普遍的因果推理逻辑与语义泛化规律。海量的新样本输入提供了持续的、不重复的监督信号，迫使模型去提取内在的统计规律而非单纯记忆文本。

---

### 3. 语料配比对模型能力的塑造

预训练数据并不是单一的普通网页文本，而是由具备不同结构特征的语料混合而成的。

![多源语料混合配比与能力图谱塑造](/images/content-rebuild/5d59c67f192bc620/fig-3.png)
*图释：不同类型的语料对应不同能力的表征学习机制。网络百科构建世界知识，代码语料培养控制流与逻辑状态追踪，学术论文塑造长程因果推导，多语言语料则在共享隐空间内促进跨语言语义对齐。*

| 语料类型 | 典型占比范围 | 主要贡献 | 潜在问题 |
| :--- | :--- | :--- | :--- |
| **通用网页 (Common Crawl)** | 50% - 70% | 覆盖海量世界常识、跨领域实体与日常行文风格 | 噪音大、广告/SEO 灌水文本多、逻辑较浅 |
| **高质量百科与书籍** | 5% - 15% | 提供严谨的事实叙述、长篇连贯论述与深层背景知识 | 总存量有限、更新存在时间滞后 |
| **代码仓库 (GitHub 等)** | 10% - 30% | 培养严密的逻辑因果、状态追踪与指令依从能力 | 包含废弃项目、语法错误或无意义配置文件 |
| **学术论文与技术报告** | 2% - 10% | 强化多步逻辑推导、数学证明与公式符号解析 | 专业门槛高、解析存在大量排版乱码 |
| **多语言语料** | 5% - 20% | 促进不同语言概念间的对齐，支持跨语言理解与翻译 | 语种分布极不平衡，低资源语言噪音较多 |

#### 为什么自然语言模型需要包含大量代码？
工业界模型普遍在预训练中加入 15% 到 30% 的高质量代码。实验表明，加入代码不仅能提升编程能力，还能显著提升自然语言环境下的**多步推理（Chain-of-Thought）能力**。

原因在于：代码具有严格的语法层级结构、明确的变量作用域定义、条件分支以及函数调用栈。自回归模型在预测代码 Token 时，必须在更长的上下文距离内准确追踪变量状态与逻辑跳转，这种训练直接强化了模型长距离依赖追踪的注意力权重分布。

#### 学习率退火（Annealing）与动态配比
很多模型的预训练并非全局采用固定配比，而是采用阶段性调度：
1. **主体训练阶段（前 90%-95% Token）**：使用过滤后的通用语料（包含大量清洗网页、代码与常识），快速建立通用的语言理解与广泛世界知识。
2. **退火阶段（最后 5%-10% Token）**：在学习率线性衰减至接近 0 的同时，调高**高质量合成数据、精选数学与代码、高质量教科书**的采样比例。这一步能有效拉高模型在各项基准跑分上的上限。

---

### 4. 数据清洗管线：从原始网络文本到高信噪比语料

未经清洗的网络数据直接送入集群训练，不仅会严重降低训练效率，还可能导致损失震荡或产生大量重复退化。典型的数据治理管线包含以下阶段：

![工业级数据清洗与治理全流程](/images/content-rebuild/5d59c67f192bc620/fig-4.png)
*图释：从原始 PB 级通用爬虫网页到最终能够输入集群训练的高净值 Token，必须经过文本提取、启发式过滤、文档模糊去重、质量分类器筛选以及安全去毒五个核心阶段，通常仅有 15% 至 30% 的原始数据得以保留。*

#### 阶段一：规则过滤（Heuristic Filtering）
通过规则低成本剔除明显的无效数据：
* **长度与词数**：剔除总词数过少、过长或行平均字符异常的文档；
* **标点与杂质占比**：过滤标点符号比例过高（如纯符号列表）、无意义特殊字符过多的文档；
* **语言与停用词**：使用 fastText 等轻量分类器做语种判别，并核查高频停用词占比。若一篇文章完全缺少基础代词和连词，通常属于机器生成的关键词堆叠。

#### 阶段二：文档去重（Deduplication）
去重是清洗环节中对模型泛化能力影响最大的步骤之一。如果数据集中充斥着成千上万篇完全相同的版权声明、免责协议或抓取模板，模型会把大量权重分配到这些模式记忆上。

大规模去重通常使用 **MinHash + LSH（局部敏感哈希）** 算法，将文档相似度比对的时间复杂度从两两对比的 $O(N^2)$ 降低到近似线性复杂度。

![MinHash 与 LSH 模糊去重原理](/images/content-rebuild/5d59c67f192bc620/fig-5.png)
*图释：MinHash 将长文档的 N-gram 集合压缩为固定长度的特征签名矩阵，使得签名碰撞概率等于 Jaccard 相似度；LSH 通过分带（Banding）哈希将具有相似签名的文档映射到相同哈希桶内，大幅减少海量文档之间的两两比对开销。*

##### MinHash 去重机制与可验证代码
MinHash 的核心数学性质是：对于两个集合 $A$ 和 $B$，在其元素经随机置换哈希后，最小哈希值相等的概率等于两个集合的 Jaccard 相似度：

$$P(\min(h(A)) = \min(h(B))) = J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

通过计算 $K$ 个不同的独立哈希值，即可得到长度为 $K$ 的签名向量，签名的一致率直接近似集合相似度。以下 Python 代码演示了该算法的核心实现：

```python
import hashlib
from typing import List, Set

def get_ngrams(text: str, n: int = 3) -> Set[str]:
    """切分字符级 N-gram 集合"""
    text = "".join(text.split())
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}

def minhash_signature(tokens: Set[str], num_hashes: int = 16) -> List[int]:
    """利用加盐哈希模拟 K 个独立哈希函数，提取最小哈希值签名"""
    signature = []
    for i in range(num_hashes):
        min_val = float('inf')
        for token in tokens:
            # 加盐后计算 md5 并转为整数
            digest = hashlib.md5(f"{i}_{token}".encode("utf-8")).hexdigest()
            val = int(digest, 16)
            if val < min_val:
                min_val = val
        signature.append(min_val)
    return signature

def estimate_jaccard(sig_a: List[int], sig_b: List[int]) -> float:
    """根据 MinHash 签名的一致率估算 Jaccard 相似度"""
    assert len(sig_a) == len(sig_b)
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)

# 验证示例
doc_1 = "大模型预训练需要利用大量多源语料进行自监督学习"
doc_2 = "大模型预训练依赖海量多源语料进行自监督表征学习"  # 近义改写
doc_3 = "显存不足通常需要结合模型并行与梯度累积优化"    # 完全不相关的工程文档

sig1 = minhash_signature(get_ngrams(doc_1, n=3))
sig2 = minhash_signature(get_ngrams(doc_2, n=3))
sig3 = minhash_signature(get_ngrams(doc_3, n=3))

print(f"Doc1 与 Doc2 估计相似度: {estimate_jaccard(sig1, sig2):.2f}")  # 预期显著大于 0
print(f"Doc1 与 Doc3 估计相似度: {estimate_jaccard(sig1, sig3):.2f}")  # 预期趋近于 0
```

#### 阶段三：质量分类打分与合成数据补充
* **质量打分器**：通常使用高质量文本（如维基百科、精选技术文档、同行评审论文）作为正样本，使用未经清洗的爬虫垃圾作为负样本，微调一个轻量级分类器（如 FastText 或小尺寸 Transformer）。对清洗后的网页计算分值，剔除低于特定分位数的页面。
* **合成数据（Synthetic Data）**：当高质量自然文本存量受限时，业界通常会利用成熟的大模型重写低质文本、生成数学解题推导步骤，或将非结构化文档转为标准问答对，以此增加高价值数据密度。

---

### 5. 重复训练旧数据的边际递减

面对数据获取成本上升，能否直接将手头现有的高质量数据集多训练几个 Epoch？

![多 Epoch 重复训练的边际递减与过拟合效应](/images/content-rebuild/5d59c67f192bc620/fig-6.png)
*图释：在海量独立数据上训练时，验证集损失随 Token 增加呈幂律稳步下降；而在有限数据上重复训练多个 Epoch 时，尽管训练损失继续减小，验证损失却在第 4 个 Epoch 之后迅速饱和甚至拐头向上，表现出明显的边际收益递减与泛化退化。*

Muennighoff 等人（2023）在关于数据约束下语言模型扩展规律的研究中给出了关键结论：
1. **多周期收益衰减极快**：在预训练场景下，相同数据集如果重复训练超过 **4 个 Epoch**，每多训练一个 Token 带来的验证集损失下降幅度会大幅削弱；
2. **记忆化与泛化分水岭**：当超过一定 Epoch 后，虽然训练集上的交叉熵损失仍在下降，但未见评估集上的损失开始饱和甚至出现反弹。模型开始过度记忆具体的字句组合，丧失了在未见句式上的泛化能力；
3. **因此，重复旧数据无法等价替代全新数据**。要让模型持续提升推理与泛化能力，必须引入具有新上下文组合与新实体的语料。

---

### 6. 进阶动手练习：验证数据量对语言建模潜力的影响

基于 Hugging Face 的因果语言模型工具链，我们可以通过一个精简脚本，亲自验证数据量与评估困惑度（Perplexity, PPL）之间的关系。

> 官方参考：[Hugging Face Causal Language Modeling Documentation](https://huggingface.co/docs/transformers/tasks/language_modeling)

#### 练习目标
1. 加载一个小尺寸因果语言模型（如 `distilbert/distilgpt2`）；
2. 在固定独立评估集与相同训练步数（`max_steps`）的控制变量条件下，对比海量新鲜样本与小规模循环样本在评估集损失（Loss）与困惑度（$\text{PPL} = e^{\text{Loss}}$）上的差异。

```python
import math
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# 1. 加载数据集，先拆分出固定独立的评估集（500 条）以保持评估基准一致
dataset = load_dataset("dany0407/eli5_category", split="train[:5000]")
split_data = dataset.train_test_split(test_size=500, seed=42)
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["title"], truncation=True, max_length=128)

# 对照实验：可切换使用完整 4500 条新鲜样本，或仅取 450 条重复样本
train_pool = split_data["train"]  # 对照组改为 split_data["train"].select(range(450))
eval_set = split_data["test"]

tokenized_train = train_pool.map(tokenize_function, batched=True, remove_columns=train_pool.column_names)
tokenized_eval = eval_set.map(tokenize_function, batched=True, remove_columns=eval_set.column_names)

# 2. 初始化模型
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

# 3. 配置轻量训练参数：通过 max_steps 锁定相同训练步数，剥离步数差异对收敛的干扰
training_args = TrainingArguments(
    output_dir="./clm_test_output",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    max_steps=100,
    weight_decay=0.01,
    save_strategy="no",
    logging_steps=20,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# 4. 训练并在相同的独立评估集上计算困惑度
trainer.train()
eval_results = trainer.evaluate()
perplexity = math.exp(eval_results['eval_loss'])
print(f"评估集困惑度 (Perplexity): {perplexity:.2f}")
```

**检验标准**：记录当前 4500 条样本在 100 步下的困惑度后，将训练集替换为 `split_data["train"].select(range(450))` 重新运行相同 100 步。在固定评估基准与相同更新步数下，观察仅有 450 条样本（重复循环过拟合）相比 4500 条新鲜样本时困惑度的明显上升，直观验证数据规模与多样性对自回归模型泛化性能的决定性作用。

---

### 7. 本讲总结与后续导引

训练大模型之所以需要海量数据，是因为：
1. **算力效率驱动**：Chinchilla 扩展定律决定了参数与数据需协同按比例缩放，而推理部署成本进一步催生了对轻量小模型的“过度训练”；
2. **分布规律约束**：齐普夫定律决定了长尾知识极其稀疏，必须有数万亿级 Token 的基数才能让关键实体获得足够的优化梯度；
3. **数据质量兜底**：单纯堆砌量级没有意义，通过 MinHash/LSH 模糊去重与多级规则清洗，剔除无效模式，模型才能稳定建立健康的概率转移空间。

然而，当我们将数以万亿计的 Token 连续送入集群进行前向传播和反向更新时，庞大的参数矩阵、梯度以及优化器中间状态，会以数十倍于模型本体权重的体量挤占硬件显存。在下一讲 [/llm-100/training/why-high-vram.md](/llm-100/training/why-high-vram.md) 中，我们将打开显卡内部，详细计算预训练显存的真实构成，并拆解分布式训练解决显存瓶颈的底层方法。
