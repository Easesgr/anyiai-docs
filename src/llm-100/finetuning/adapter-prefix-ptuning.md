---
title: "Adapter、Prefix Tuning、P-Tuning 有什么区别？"
icon: mdi:tune-vertical
tag:
  - 微调与对齐
  - 大模型100讲
  - RLHF
---

# Adapter、Prefix Tuning、P-Tuning 有什么区别？ - 一句话定位

> 三者都是「参数高效微调（PEFT）」家族的代表方法，分别在 Transformer 的不同位置插入少量可训练参数：Adapter 在层内插入小 MLP 模块、Prefix Tuning 在每层 Attention 的 K/V 前拼接可训练前缀向量、P-Tuning 则在输入 Embedding 层用 LSTM 编码的连续 prompt 替代离散 token。面试常问是因为这是大模型时代必须会的"低成本适配下游任务"核心技术，2023 年后所有主流开源模型（ChatGLM、LLaMA、Qwen）的微调方案都跑不出这几种。

## 核心概念（术语表）
- **PEFT（Parameter-Efficient Fine-Tuning）**：参数高效微调，只训练少量参数即可让大模型适配下游任务，原模型权重冻结。
- **Adapter（适配器）**：2019 年 Houlsby 等人提出，在 Transformer 每层插入一个「下投影→非线性→上投影」的小 bottleneck 模块。
- **Prefix Tuning（前缀微调）**：2021 年 1 月 Google 提出，在每一层 Attention 的 Key 和 Value 前拼接一段连续可训练向量（前缀），前缀长度通常 10-200。
- **Prompt Tuning（提示微调）**：2021 年 4 月提出，只在输入 Embedding 层加一组可训练的 soft prompt，不动模型内部。
- **P-Tuning**：清华 2022 年 2 月提出，用 LSTM 将 prompt token 编码后再输入，并加入 anchor token（如"?"）提升 NLU 任务效果。
- **P-Tuning v2**：清华 2021 年 10 月（注意发表更早但命名在后），本质是把 Prefix Tuning 搬到了 NLU 任务，在每一层都加前缀。
- **AdapterFusion**：在 Adapter 基础上引入「知识提取 + 知识组合」两阶段，用 Query-Key-Value 注意力机制组合多个 Adapter。
- **LoRA（Low-Rank Adaptation）**：2021 年微软提出，用两个低秩矩阵 A（高斯初始化）、B（零初始化）的乘积 BA 近似权重更新量 ΔW。


<ReadMoreLock />## 历史背景 / 来源

Adapter 最早由 Houlsby N 等人于 2019 年 2 月在论文《Parameter-Efficient Transfer Learning for NLP》（arxiv:1902.00751）中提出，作为全参数微调的替代方案；Prefix Tuning 由 Google 的 Li & Liang 于 2021 年 1 月发表（arxiv:2101.00190），专为生成式任务设计；P-Tuning 由清华 Liu 等人于 2021 年 10 月提出、2022 年 2 月正式发表（arxiv:2103.10385 / 2110.08730），专门解决 NLU 任务上 Prompt 效果差的问题。三者都属于 PEFT 大类中的「Additive methods（添加式方法）」，即在原模型结构上额外添加少量参数，原权重全部冻结，主流来源：CSDN 与知乎两篇综述。

## 工作原理 / 核心机制（详细讲解）

**整体思路**：三种方法都遵循「冻结预训练参数 + 训练少量新增参数」的范式，但新增参数的位置、形状、初始化方式各不相同。

**Adapter 工作机制**：在 Transformer 每个 block 的 Attention 输出后和 FFN 输出后各插入一个 Adapter 模块。Adapter 内部是「下投影（d→r）→ ReLU → 上投影（r→d）→ 残差连接」结构，r 是 bottleneck 维度（典型值 8-64）。训练时冻结主模型，只更新 Adapter 的权重（约原模型 3-5% 参数）；推理时 Adapter 可与主模型合并，无额外延迟。

**Prefix Tuning 工作机制**：第一步：在每一层 Attention 计算前，给 K 和 V 矩阵各拼接一段长度 P 的可训练向量前缀（Prompt¹⁻ᴸ，P 通常 10-200）。第二步：训练时冻结整个 Transformer，只优化这些前缀向量。第三步：输入 x 经 Embedding 后，前缀向量也用同一个 Embedding 矩阵做重参数化（防止训练不稳定），推理时可去掉重参数化层。生成的 K̃、Ṽ 维度从 (d, n) 变成 (d, n+P)，Attention 输出公式变为 softmax(QK̃ᵀ/√d)Ṽ。

**P-Tuning 工作机制**：第一步：用 LSTM（替代 MLP）对 prompt token 的 embedding 做编码，得到连续 prompt 表示（这一步解决「随机初始化+小梯度」导致 embedding 难以优化的问题）。第二步：在模板中加入 anchor token，例如 RTE 任务模板从 [PRE][prompt][HYP][MASK] 改为 [PRE][prompt][HYP]?[prompt][MASK]，加上"?"后效果显著提升。第三步：训练时只更新 prompt 参数（约 0.1%-3%）和 LSTM，主模型冻结。

## 关键知识点（10-15 条 bullet，每条都要具体）

- Adapter 论文 2019 年发表，是 PEFT 领域最早的工作，原始设计在每层插 2 个 bottleneck 模块
- Adapter bottleneck 维度 r 通常取 8-64，参数量约为原模型的 3-5%
- Adapter 推理时可与主模型合并，理论上无延迟；但合并前有额外计算开销
- Prefix Tuning 论文 2021.1 Google 提出，专为 GPT-2 的 table-to-text 和摘要任务设计
- Prefix Tuning 前缀长度 P 通常 10-200，比 Prompt Tuning 长，占用有效序列长度
- Prompt Tuning 2021.4 提出，模型越大效果越好，10B+ 模型上能追平全参数微调
- P-Tuning 2022.2 清华发表，专门为 NLU 任务设计，模板含 anchor token（如"?"）
- P-Tuning 用 LSTM 编码 prompt，解决 embedding 优化困难的问题
- P-Tuning v2 = Prefix Tuning 思路搬到 NLU，每层都加前缀，监督信号更丰富
- LoRA 在 attention 层的 Q/V 矩阵上加低秩旁路，A 高斯初始化、B 零初始化，保证初始 BA=0
- LoRA 显存优势显著：单卡 14GB 即可微调，P-Tuning V2 需 18GB，全量 SFT 需 4×48GB
- AdapterFusion 两阶段训练：第一阶段学 Adapter，第二阶段固定它们学组合权重
- Prefix Tuning 缺点：占用有效序列长度、训练不稳定、效果通常不如全量微调
- Prompt Tuning 缺点：小模型上效果差，需百亿参数以上才逼近全量微调
- 行业现状（2023+）：ChatGLM、LLaMA、Qwen 等模型官方微调方案以 LoRA + P-Tuning V2 为主

## 应用场景（3-5 个真实例子）

- **场景 1：Stable Diffusion + LoRA**：CivitAI 社区用 LoRA 微调 SD 模型生成特定画风/角色，单张 RTX 3090（24GB）即可训练，画风 LoRA 仅需几十张参考图，权重文件通常 10-200MB。
- **场景 2：ChatGLM-6B + P-Tuning V2**：清华官方推荐用 P-Tuning V2 微调 ChatGLM，单卡 18GB 显存即可启动（vs 全量 SFT 需 4×48GB），是国内最早普及 PEFT 的开源大模型。
- **场景 3：BERT + Adapter**：Houlsby 2019 论文中，在 GLUE 基准上 Adapter 比 Full Fine-tuning 高 0.4%，参数仅 3.6%，已在 Google 内部多个搜索/NLU 任务部署。
- **场景 4：GPT-2 + Prefix Tuning**：Google 在 table-to-text（如 WebNLG）和摘要任务上用 Prefix Tuning，BLEU 与全量微调持平，参数效率提升 1000 倍。
- **场景 5：LLaMA-7B + QLoRA**：2023 年主流方案，4-bit 量化 + LoRA 让单卡 24GB 微调 65B 模型成为可能，是 Alpaca、Vicuna 等开源对话模型的标配。

## 常见误区 / 踩坑（4-6 条，每条给"错在哪 + 为什么错 + 正确的是什么"）

- ❌ **误区 1**：以为 Adapter 和 Prefix Tuning 是一回事
  ✅ **正解**：Adapter 是在层内插入 bottleneck MLP 模块，作用于 hidden state；Prefix Tuning 是在 Attention 的 K/V 前拼接前缀向量，不改变模型结构。
- ❌ **误区 2**：以为 P-Tuning 就是 Prompt Tuning
  ✅ **正解**：Prompt Tuning 只在 Embedding 层加可训练 soft prompt；P-Tuning 用 LSTM 编码 prompt 并加 anchor token，专门解决 NLU 任务 prompt 难训的问题。
- ❌ **误区 3**：以为 LoRA 是某种 Adapter 变体
  ✅ **正解**：LoRA 属于「重参数化方法」，在权重矩阵旁加低秩旁路 A×B，不改变原网络的前向结构；Adapter 是「添加式方法」，在层与层之间插入新模块，会改变网络深度。
- ❌ **误区 4**：以为 Prefix Tuning 训练很稳定
  ✅ **正解**：实际训练中 prefix 容易不稳定，需要做 reparameterization（用一个 MLP 先编码再投影）才能稳定收敛。
- ❌ **误区 5**：以为 P-Tuning v2 是 P-Tuning 的小升级
  ✅ **正解**：P-Tuning v2 本质是 Prefix Tuning 的 NLU 版本，每层都加前缀，是清华团队同期两个工作的命名冲突。
- ❌ **误区 6**：以为 PEFT 只能用于小模型
  ✅ **正解**：Prompt Tuning 论文明确指出，模型规模超过 10B 后，Prompt Tuning 效果追平甚至超过全量微调。

## 性能 / 复杂度（数据驱动）

- **时间复杂度**：Adapter O(L·d·r)、Prefix Tuning O(L·(n+P)·d)、P-Tuning O(L·d·|prompt|)、LoRA O(L·d·r)，其中 L 是层数、d 是 hidden 维度、r 是低秩维度、P 是前缀长度。
- **空间复杂度**：可训练参数占比 Adapter 约 3-5%、Prefix Tuning 约 0.1-1%、P-Tuning 约 0.1-3%、LoRA 约 0.5-5%（rank 4-32 时）。
- **显存实测对比**（来源：CSDN 综述）：
  - SFT 全量微调：4×48GB（单卡 48346MiB）
  - P-Tuning V2：1×18GB（18426MiB）
  - LoRA：1×14GB（14082MiB）
- **临界点选择**：当可训练参数 < 0.1% 选 Prompt Tuning；0.1-1% 选 P-Tuning / Prefix Tuning；1-5% 选 Adapter / LoRA；>5% 选 IA³ 或 Full Fine-tuning。

## 与相关概念的区别（至少 3 对）

- **vs Adapter**：
  - 维度 1（插入位置）：Adapter 插入到层内部 MLP 之后，Prefix Tuning 插入到 Attention 的 K/V 之前
  - 维度 2（参数量）：Adapter 约 3-5%，Prefix Tuning 约 0.1-1%
  - 维度 3（推理延迟）：Adapter 合并前有延迟，Prefix Tuning 永远有额外 P 长度的 K/V 计算
  - 怎么选：NLU/分类选 Adapter，生成任务选 Prefix Tuning
- **vs P-Tuning**：
  - 维度 1（插入层数）：Prefix Tuning 每层都加，P-Tuning 只在 Embedding 层加
  - 维度 2（任务类型）：Prefix Tuning 偏生成，P-Tuning 偏 NLU
  - 维度 3（训练难度）：Prefix Tuning 需重参数化稳定，P-Tuning 用 LSTM 稳定
  - 怎么选：生成任务选 Prefix，NLU 任务且数据少选 P-Tuning/P-Tuning v2
- **vs LoRA**：
  - 维度 1（实现方式）：Adapter/Prefix/P-Tuning 都是「添加新结构」，LoRA 是「重参数化原权重」
  - 维度 2（推理延迟）：Adapter 可合并无延迟，Prefix/P-Tuning 永久有额外计算，LoRA 可合并无延迟
  - 维度 3（显存占用）：LoRA 单卡 14GB 最省，P-Tuning V2 需 18GB
  - 怎么选：追求极致显存效率选 LoRA，部署多任务共享前缀选 Prefix Tuning

## 进阶 / 面试加分项（2-3 条）

- **最新进展**：2023 年起 IA³、QLoRA、AdaLoRA、UniPELT 等新方法涌现，IA³ 用学习向量缩放激活比 LoRA 更省参（0.01%）；QLoRA 4-bit 量化 + LoRA 让单卡微调 65B 模型；UniPELT 动态组合多种 PEFT 方法。
- **业界争议**：PEFT 是否能完全替代全量微调仍存争议——研究表明在小模型（<1B）上 PEFT 普遍弱于 SFT，但在 10B+ 大模型上差距消失甚至反超；此外 PEFT 在复杂推理任务（数学、代码）上的表现仍待验证。
- **金句**："PEFT 的本质是用 0.1% 的参数撬动 100% 的能力，关键是找到任务相关的『内在低秩维度』。"



## 面试如何回答


### 🟢 请用一句话分别定义 Adapter、Prefix Tuning 和 P-Tuning，并指出它们的核心区别。

**回答要点**：

Adapter 是在 Transformer 每层插入 bottleneck 小模块（参数量约 3-5%）；Prefix Tuning 是在每层 Attention 的 K/V 前拼接可训练前缀向量（参数量约 0.1-1%）；P-Tuning 是在输入 Embedding 层用 LSTM 编码的连续 prompt 替代离散 token。核心区别在于「插入位置」和「任务类型」：Adapter 改层结构、Prefix 改注意力输入、P-Tuning 改 Embedding；Adapter 通用、Prefix 偏生成、P-Tuning 偏 NLU。面试时可补充：三者都属于 PEFT 的 Additive methods 家族，原模型参数全部冻结。


### 🟢 Adapter 模块的内部结构是什么？为什么用 bottleneck 设计？

**回答要点**：

Adapter 内部结构是「下投影（d→r）→ ReLU 非线性 → 上投影（r→d）→ 残差连接」，其中 d 是 hidden 维度、r 是 bottleneck 维度（典型 8-64）。Bottleneck 设计有两个目的：一是降低参数量（参数量约为 d×r 而非 d×d，r<<d 时可压缩 90% 以上）；二是约束 Adapter 的表达能力，防止它学到一个「近似 identity」从而绕过主干网络。Adapter 通常插在 Attention 输出后和 FFN 输出后各一个，所以每层约 2×d×r 参数。


### 🟡 Prefix Tuning 训练不稳定的问题怎么解决？

**回答要点**：

Prefix Tuning 直接训练 prefix 参数时容易不稳定，因为 prefix 长度 P 一般 10-200，输入空间巨大且没有强监督信号。Google 论文给出解法：用重参数化（reparameterization）——先用一个 MLP（或 LSTM）把 prefix 编码出来，再投影到 K/V 维度，训练时更新 MLP 参数，推理时去掉 MLP 留下投影结果。此外学习率要设小（论文用 5e-5）、训练步数要够多。这套技巧后来成为 P-Tuning v2 的标配，本质是用一个结构化网络降低 prefix 优化的难度。


### 🟡 P-Tuning 相比 Prompt Tuning 改了什么？为什么 NLU 任务效果更好？

**回答要点**：

Prompt Tuning 是在 Embedding 层加一组 soft prompt，直接端到端训练；P-Tuning 做了两个关键改动：第一，用 LSTM（替代 MLP）编码 prompt token，因为 Embedding 空间是离散的、随机初始化后梯度很小，LSTM 能引入时序归纳偏置让 prompt 更易优化；第二，加入 anchor token，例如 RTE 任务模板从 [PRE][prompt][HYP][MASK] 改为 [PRE][prompt][HYP]?[prompt][MASK]，加个"?"能提供更明确的任务监督。这两个改动让 P-Tuning 在 NLU 任务（如 RTE、CoNLL）上首次超越精调效果，是 Prompt 派系在 NLU 上的里程碑。


### 🟡 LoRA 显存为什么比 P-Tuning V2 还低？原理上有什么本质区别？

**回答要点**：

根据 CSDN 综述实测，LoRA 微调单卡占用 14082MiB（14GB），P-Tuning V2 占用 18426MiB（18GB），SFT 全量微调需 4×48GB（每卡 48346MiB）。LoRA 更省的关键有两点：第一，LoRA 是重参数化方法，在 Q/V 矩阵旁加低秩旁路 A×B（A 是 d×r，B 是 r×d），新增参数量只与秩 r 线性相关，r 通常取 4-32；第二，LoRA 优化器状态只需保存这 0.5-5% 的小参数，而 P-Tuning V2 在每层都加 prefix（前缀累加），参数量虽小但激活值和梯度通路更长。根本区别：LoRA 改的是权重矩阵本身，P-Tuning 改的是输入空间。


### 🔴 如果让你为一个 7B 的开源大模型设计多任务微调方案，你会怎么选 PEFT 方法？为什么？

**回答要点**：

我会按场景分层选择。通用场景选 LoRA：单卡 24GB 即可训练，参数量仅 0.5-5%，推理时可合并无延迟，社区生态最成熟（Alpaca、Vicuna 都是这套）。多任务共享场景选 AdapterFusion：每个任务一个 Adapter（约 3% 参数），用 Attention 机制组合，避免灾难性遗忘和任务干扰。NLU 分类场景选 P-Tuning V2：在每层加 prefix 提升监督信号，对小样本友好。生成任务选 Prefix Tuning：天然适配 table-to-text、摘要等场景。关键决策因子是「显存预算」「任务数」「是否需要推理合并」，单一 PEFT 很少能 cover 所有需求，UniPELT 这类动态组合方法会是未来方向。


### 🔴 PEFT 一定能替代全参数微调吗？什么情况下 PEFT 会失效？

**回答要点**：

不是。研究表明 PEFT 的效果高度依赖模型规模和数据规模：在 10B+ 大模型 + 足量数据上，PEFT（尤其是 Prompt Tuning、LoRA）能追平甚至超过 SFT；但在 1B 以下小模型上，PEFT 普遍弱于 SFT 1-3 个点。容易失效的场景包括：复杂推理任务（数学、代码）、需要深度语义改写的生成任务（如风格迁移）、数据分布与预训练差异极大的领域（如法律、医疗专业文本）。此外 PEFT 训练步数通常要更多（3-5 倍），学习率要更小，否则容易欠拟合。面试金句：「PEFT 的本质是用 0.1% 的参数撬动 100% 的能力，但前提是模型本身已经足够强」。


### 🟡 AdapterFusion 解决了 Adapter 的什么缺陷？两阶段训练具体怎么做？

**回答要点**：

Adapter 的核心缺陷是「灾难性遗忘」和「任务间干扰」——多任务联合训练时不同 Adapter 会互相冲突，且习得的知识不能跨任务迁移。AdapterFusion 用两阶段解法：第一阶段「知识提取」，在 N 个下游任务上分别训练 N 个独立 Adapter 模块，把任务知识封装到 Adapter 参数中；第二阶段「知识组合」，冻结预训练参数和所有 Adapter 参数，引入新的可学习参数 Ψ（含 Key、Value、Query），让模型学习如何用 Attention 机制（Query·Key → Softmax 加权 Value）组合多个 Adapter。这种设计允许一个模型同时利用多任务知识，且新增参数极少。在 MRPC、RTE 等任务上 AdapterFusion 显著优于单独 Adapter 和 Full Fine-tuning。

