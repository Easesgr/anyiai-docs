---
title: "QLoRA 如何进一步降低微调成本？- 一句话定位"
icon: mdi:tune-vertical
tag:
  - 微调与对齐
  - 大模型100讲
  - RLHF
---

# QLoRA 如何进一步降低微调成本？- 一句话定位

> 一句话核心：QLoRA = 4bit 量化基座 + 全精度 LoRA 适配器，在 LoRA 已经很省的基础上，把"基座模型本身"从 fp16/bf16 再压缩到 4bit，让 8B/14B 这种规模的大模型也能在单卡甚至消费级显卡上微调，且几乎不掉点。面试官问这个题，是看你能否说清"为什么普通 LoRA 还不够、QLoRA 是怎么在三个工程层面抠显存的"。

## 核心概念（术语表）
- **QLoRA（Quantized LoRA）**：把基座模型量化到 4bit、再挂上 LoRA 适配器的微调方法，论文 arxiv.org/abs/2305.14314。
- **LoRA（Low-Rank Adaptation）**：在原始权重 W₀ 上加一个低秩增量 ΔW = BA，B∈R^(d×r)、A∈R^(r×k)，训练时只更新 A、B。
- **NF4（NormalFloat-4）**：专门为"近似正态分布的权重"设计的 4bit 量化数据类型，4bit→16 个量化刻度，但不是等分，而是"中间密、两端疏"。
- **Double Quantization（双重量化）**：对权重做完一次块内 4bit 量化后，再对所有 block 的 scale 本身做一次 8bit 量化，再省一截存储。
- **Paged Optimizers（分页优化器）**：当 GPU 显存紧张时，把优化器状态（如 Adam 的 m、v）自动卸载到 CPU 内存，需要时再 page in，避开 OOM。
- **bitsandbytes（bnb）**：HuggingFace 生态里做 4bit/8bit 量化 + 高效算子的库，提供 BitsAndBytesConfig。
- **PEFT（Parameter-Efficient Fine-Tuning）**：HuggingFace 出的高效微调库，get_peft_model 把 LoRA 注入到基座里。
- **NF4 vs FP4**：NF4 是按正态分布设计刻度，FP4 是普通浮点 4bit；QLoRA 论文用的是 NF4。


<ReadMoreLock />## 历史背景 / 来源
QLoRA 由 Tim Dettmers 等人 2023 年 5 月提出（论文 arxiv:2305.14314），同年发表在 NeurIPS 2023。它的直接动机是：LoRA 虽然把可训练参数压到 1% 以下，但 base 模型在 fp16 下依然吃显存——65B 的 LLaMA 用 LoRA 微调需要约 780GB 显存，根本跑不动。QLoRA 把 base 量化到 4bit 后，33B 模型单卡 24GB 就能跑、65B 模型单卡 48GB 就能跑，并在 Guanaco 等聊天场景中达到和 fp16 LoRA 相当的对话质量。配套的 DoRA（arxiv:2402.09353，2024 年 2 月）则把权重拆成"方向 × 标量幅度"做更精细控制，定位是 LoRA 的另一个升级方向。

## 工作原理 / 核心机制（详细讲解）

**整体思路（一句话）**：基座权重 W₀ 量化成 4bit 冻住不动，前向时临时反量化到 bf16 参与计算，反向时只对 LoRA 的 A、B 两个小矩阵求梯度，把"省显存"这件事做到极致。

**输入/输出**：输入是一个 fp16/bf16 的预训练基座（论文里主要是 LLaMA 系列，33B/65B）；输出是一个只新增了 ~0.1%~1% 参数量的 LoRA 适配器，推理时可以把 LoRA 权重合并回基座，无额外延迟。

**核心步骤详解**：

**第一步 - 4bit 量化加载基座**：用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype="bfloat16")` 加载模型。W₀ 从原始 fp16（每参数 2 字节）压到 4bit（每参数 0.5 字节），理论上权重本身显存立刻降到 1/4。一个 65B 模型 fp16 约 130GB，4bit 后只剩约 32.5GB。

**第二步 - NF4 按"权重长什么样"分配刻度**：4bit 一共只有 16 个刻度，普通做法是把 [-1, 1] 均匀切成 16 份，但神经网络权重近似正态分布，绝大多数值挤在 0 附近。NF4 把 16 个刻度按"正态分布分位数"摆位——中间密、两端疏，等于把精度全用在权重真正密集的地方。类比：温度计的 15–35°C 段比 -20°C 和 80°C 段刻得更细。

**第三步 - Double Quantization 再砍一刀**：量化每个 block 时需要一个 scale（甚至 zero_point），普通做法下 scale 用 fp16 存，单个很"贵"。Double Quantization 第一层对每个 block 做 4bit 量化得 code_i 和 scale_i；第二层把所有 scale_i 再做一次 8bit 量化得 scale_codes 和一个 global_scale。最终存储结构 = 4bit code + 量化过的 scale code + 少量全局常数。结果是：每 64 个权重原来要存 64×0.5B 的 code + 一个 fp16 scale，现在 scale 又被压缩，存储再省约 0.4 bit/参数。

**第四步 - Paged Optimizers 兜底 OOM**：训练时优化器状态（Adam 的 m 和 v，fp32 各一份）显存占用甚至超过权重本身。Paged Optimizers 在 CUDA OOM 触发时自动把优化器状态卸载到 CPU 内存（page out），需要时再 page in，对训练过程几乎透明。

**第五步 - 注入 LoRA 训练**：用 `LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"])` + `get_peft_model` 把 LoRA 挂上去。前向：W₀ 反量化到 bf16 后和 BA 相加参与计算；反向：梯度只回传给 A、B，W₀ 不更新、不存梯度。

**第六步 - 推理可合并**：训练完后可以把 A、B 合并回 W₀，得到一个和原模型完全等价的 fp16（或 4bit）模型，部署时无额外推理开销。

## 关键知识点（10-15 条 bullet，每条都要具体）
- QLoRA 论文 arxiv:2305.14314（2023-05），DoRA 论文 arxiv:2402.09353（2024-02）。
- QLoRA = 4bit 量化基座 + 全精度 LoRA 适配器，"前向反量化、反向只回传 LoRA"。
- 一句话公式：QLoRA 显存 ≈ 4bit 权重 + LoRA(fp16) + 优化器状态(可分页)。
- NF4 关键设计：4bit 共 16 个刻度，按正态分布分位数摆放，中间密、两端疏。
- Uniform 4bit 等分 16 段，NF4 不等分——这是 QLoRA 不掉点的关键。
- Double Quantization = 块内 4bit 量化 + 对 block scale 再做 8bit 量化。
- Paged Optimizers 触发时机：CUDA OOM 时自动把 Adam 的 m/v 卸载到 CPU 内存。
- 显存压减效果（论文数据）：65B 模型 fp16 LoRA 约 780GB，QLoRA 单卡 48GB 即可。
- 33B 模型：QLoRA 单卡 24GB（一张 RTX 3090/4090）即可微调。
- 训练参数比例：LoRA 适配器通常占 0.1%–1%，99.9% 以上参数冻结。
- 工程栈：bitsandbytes（量化）+ transformers + peft（LoRA 注入）+ accelerate（设备分配）。
- BitsAndBytesConfig 四个关键参数：load_in_4bit、bnb_4bit_quant_type="nf4"、bnb_4bit_use_double_quant=True、bnb_4bit_compute_dtype="bfloat16"。
- 量化基础补充：对称量化零点固定为 0、数值范围 [-127,127]；非对称量化动态算 zero_point、范围 [0,255]。
- 推理阶段 LoRA 可合并回 W₀，无额外延迟；多任务场景也可保留多份 LoRA 热切换。
- 常用 target_modules：q_proj、k_proj、v_proj、o_proj（attention）；MLP 的 gate/up/down_proj 也可加。

## 应用场景（3-5 个真实例子，每例都给"公司/项目 + 数据 + 解决了什么问题"）
- **场景 1：消费级显卡微调 33B 模型**：单人开发者在 RTX 4090（24GB）上微调 LLaMA-33B，原本 fp16 LoRA 需要 4 张 A100，现在单卡 QLoRA 即可启动，把微调硬件门槛降低 ~75%。
- **场景 2：企业定制中文垂直模型**：电商公司用 QLoRA 在自有客服对话数据上微调 7B/13B 模型，1–2 张 A100 训 3–5 个 epoch，就能拿到领域模型，部署成本降到原来的 1/3 以下。
- **场景 3：多任务适配器并存**：一个 13B 基座上同时挂"客服 LoRA"、"代码 LoRA"、"摘要 LoRA"，每份只有几十 MB，存储成本忽略不计，推理时按需热插拔。
- **场景 4：边缘设备 + 量化推理**：训练用 QLoRA，训练完后把 4bit 基座 + LoRA 合并，再做 GPTQ/AWQ 后量化，部署到边缘 GPU 甚至消费级笔记本。
- **场景 5：学术复现**：Guanaco（QLoRA 论文里的对话模型）用 QLoRA 在 24 小时内、用单卡微调出接近 ChatGPT 表现的 33B/65B 对话模型，成为低成本 SFT 的标杆。

## 常见误区 / 踩坑（4-6 条，每条给"错在哪 + 为什么错 + 正确的是什么"）
- ❌ **误区 1**：以为 4bit 量化一定会大幅掉点。
  ✅ **正解**：NF4 针对正态分布权重专门设计刻度，QLoRA 论文在 Guanaco 上证明对话质量与 fp16 LoRA 相当，肉眼几乎无感。
- ❌ **误区 2**：把 QLoRA 的 4bit 当成"全程 4bit 计算"。
  ✅ **正解**：存储是 4bit，前向计算时要反量化到 bf16 参与矩阵乘，真正算的还是 bf16，只是常驻显存省了。
- ❌ **误区 3**：觉得 QLoRA 训练速度一定比 LoRA 慢很多。
  ✅ **正解**：主要瓶颈是矩阵乘，反量化开销很小，端到端速度一般只慢 10%–30%，远小于"显存省 4 倍"带来的收益。
- ❌ **误区 4**：以为 Double Quantization 等价于"一次性把整个 W 用更粗的精度量化"。
  ✅ **正解**：一次性粗量化要么范围太广导致中部精度崩、要么范围太窄导致 block 截断；Double Quantization 通过"块内精、块间再压"两步走解决。
- ❌ **误区 5**：LoRA 的 rank 越高越好。
  ✅ **正解**：r=8/16/32 在大多数任务已足够，过高反而逼近全参数微调，QLoRA 论文一般推荐 r=16、alpha=32。
- ❌ **误区 6**：Paged Optimizers 是"必须配置"。
  ✅ **正解**：Paged Optimizers 是兜底机制，只有在显存紧张、Adam 状态塞不下时才触发；显存充足时几乎用不到，但加上没坏处。

## 性能 / 复杂度（数据驱动）
- **时间复杂度**：前向 O(d×k×n_tokens)；反向只回传 LoRA 的 A、B，实际梯度计算量 = 正常训练的 (r×(d+k))/(d×k) ≈ 0.01%~1%。
- **空间复杂度**：权重 4bit + LoRA fp16 + 优化器 fp32 + 激活值 bf16。65B 模型总显存峰值约 48GB（单卡），相比 fp16 LoRA 的 780GB 节省 ~94%。
- **与替代方案对比**：
  - **全参数微调（Full Fine-Tuning）**：65B 需要 ~780GB 显存 + 全部梯度 + Adam 状态，几乎必须多卡 + ZeRO；适用场景是预算充足、有海量数据。
  - **普通 LoRA（fp16）**：65B 仍需约 130GB 权重 + 大量激活，4×A100 80GB 才能跑。
  - **QLoRA（4bit + LoRA）**：65B 单卡 48GB 即可，效果与 fp16 LoRA 相当；适用场景是显存受限、多任务、快速迭代。
  - **临界点**：模型 ≤7B 时，普通 LoRA 在消费卡上即可，QLoRA 收益有限；模型 ≥13B 时，QLoRA 显存优势明显放大；≥33B 时基本是唯一可行选项。
- **性能数字**：论文报告 33B Guanaco 在 24 小时内单卡微调完成；与 fp16 16-bit LoRA 相比，QLoRA 在 MMLU 等基准上平均差距 < 1 个百分点；Guanaco 65B 在 Vicuna benchmark 上达到 ChatGPT 97% 的水平。

## 与相关概念的区别（至少 3 对，每对给"维度 + 各自优势 + 一句话总结怎么选"）
- **vs 普通 LoRA**：
  - 维度 1（显存）：LoRA fp16 65B 需 ~130GB 权重，QLoRA 65B 仅 ~32.5GB；QLoRA 优。
  - 维度 2（精度）：LoRA 全精度无损，QLoRA NF4 极轻微损失（论文 <1%）；LoRA 略优。
  - 维度 3（速度）：LoRA 略快 10–30%，但 QLoRA 显存省 4 倍；QLoRA 性价比更高。
  - 怎么选：模型 ≥13B 或单卡微调 → QLoRA；模型 ≤7B 且显存充足 → 普通 LoRA 也行。
- **vs 全参数微调（Full FT）**：
  - 维度 1（可训练参数）：Full FT 100%、QLoRA 0.1–1%；QLoRA 完胜。
  - 维度 2（显存）：Full FT 需存权重 + 梯度 + Adam 三套；QLoRA 只多 LoRA 一套；QLoRA 完胜。
  - 维度 3（极端任务表现）：Full FT 在数据量极大时略胜，QLoRA 在中小数据上几乎追平；多数场景 QLoRA 更划算。
  - 怎么选：数据量 <100k、追求快速迭代 → QLoRA；数据 > 1M、有算力 → Full FT。
- **vs DoRA**：
  - 维度 1（思路）：DoRA 把 W 拆成方向 m / 标量幅度 ‖V‖_c，分别微调；QLoRA 把 W 量化到 4bit。
  - 维度 2（显存）：DoRA 不省显存、QLoRA 省 4 倍；QLoRA 显存优。
  - 维度 3（精度）：DoRA 在部分基准比 LoRA 略好，QLoRA 与 LoRA 几乎持平。
  - 怎么选：要省显存 → QLoRA；显存够、要进一步提点 → DoRA；极致省钱提点 → QLoRA + DoRA 组合（论文已验证可行）。
- **vs GPTQ/AWQ（推理量化）**：
  - 维度 1（阶段）：GPTQ/AWQ 是训后量化（PTQ），QLoRA 是训练时量化；
  - 维度 2（目标）：GPTQ/AWQ 只为推理省显存、不能直接训练，QLoRA 训练 + 推理都省；
  - 怎么选：纯推理部署用 GPTQ/AWQ；要微调就用 QLoRA，训完可以再叠 PTQ。

## 进阶 / 面试加分项（2-3 条，超出基础的高阶理解）
- **最新进展**：QLoRA 之后出现了"QDoRA = QLoRA + DoRA"组合（论文已报告 65B 在 Vicuna benchmark 略超 ChatGPT 99%）、LongLoRA 用 QLoRA 微调 100k 上下文窗口、QLoRA-stile 微调也扩散到多模态（QLoRA-CLIP）等。趋势是把 QLoRA 当成"低显存微调底座"再叠加结构创新。
- **业界争议/未解问题**：NF4 在权重分布偏离正态（比如 outlier 严重）的层（如 LLM 的 attention 头）会丢精度；一种解法是 mixed precision——大部分层 NF4、敏感层留 8bit/fp16。另一个争议是 Paged Optimizers 在 CPU–GPU 之间 page in/out 的延迟对极小 batch 大模型是否构成瓶颈，多数论文报告影响可忽略。
- **一句话送给候选人**：QLoRA 不是"新算法"，而是"LoRA + 工程极致抠显存"的组合拳——NF4 抠精度分配、Double Quantization 抠 scale 存储、Paged Optimizers 抠优化器状态；三件套一起上，才能让 65B 单卡 48GB 跑起来。回答时按"背景痛点 → 三件套 → 效果数据 → 与 LoRA/Full FT/DoRA 对比"四步讲，面试官会立刻认可你的深度。


## 面试如何回答


### 🟢 QLoRA 和普通 LoRA 到底有什么区别？为什么已经有了 LoRA 还需要 QLoRA？

**回答要点**：

QLoRA = 4bit 量化基座 + 全精度 LoRA 适配器，本质就是把 LoRA 的 base 模型从 fp16 进一步压到 4bit。普通 LoRA 虽然只训练 0.1%–1% 参数，但 base 模型本身在 fp16 下仍然吃显存——65B 模型 fp16 LoRA 仍然要约 130GB 权重 + 大量激活，单机多卡都吃紧。QLoRA 的改进点有三件套：①NF4 按正态分布定制的 4bit 刻度，让权重密集区精度更高；②Double Quantization 对 block 的 scale 再做一次 8bit 量化，省下 scale 的存储；③Paged Optimizers 在 CUDA OOM 时把 Adam 状态自动卸载到 CPU。结果是 65B 模型在单卡 48GB 就能跑。论文报告 QLoRA 与 fp16 LoRA 在 MMLU 上差距 < 1 个百分点。回答模板：先一句话定义，再说"为什么 LoRA 不够"的显存瓶颈，最后用三件套串起来。


### 🟢 NF4 为什么要按正态分布设计刻度？Uniform 4bit 不行吗？

**回答要点**：

NF4 的设计动机是神经网络的权重近似正态分布——绝大多数值挤在 0 附近，远离 0 的极大/极小值很少。Uniform 4bit 把 [-1,1] 均匀切成 16 段，等于"给常用温度范围和极端温度一样的刻度"，结果是中间密集区域被切得很粗、两端稀疏区域却浪费格子。NF4 用 16 个刻度按正态分布的分位数摆位：中间密、两端疏，等于把精度全用在权重真正密集的地方。类比：温度计在 15–35°C 段刻度超密，-20°C 和 80°C 段粗放。QLoRA 论文报告在 LLaMA 系列上 NF4 比 FP4 效果显著更好，正是因为权重分布"天生就是中间密"。如果面试官追问：为什么不用 8bit？因为 8bit 单参数 1 字节，65B 模型还要 65GB，达不到"单卡 48GB"的目标；NF4 用 4bit（0.5 字节）才能压到 32.5GB。


### 🟡 Double Quantization 既然要把 scale 再压一遍，为什么不能一次直接用更粗的精度量化整个 W？

**回答要点**：

问题在于"一次性粗量化"会破坏"块内精度"。具体来说：如果把整个 W 直接用粗刻度量化，scale 范围必须覆盖所有 block 的极值，导致刻度被极端值拉得很宽，中部权重密集区被切得非常粗、误差爆炸；如果 scale 范围收窄，又会截断部分 block 的极大极小值，loss 直接发散。Double Quantization 通过两层结构绕开这个矛盾：第一层在每个 block 内部用 4bit 量化，让每块自己挑合适的 scale 范围，块内精度高；第二层再对所有 block 的 scale 集合做一次低比特（8bit）压缩，利用"scale 之间也有结构"这一事实省存储。最终存储 = 4bit code + 量化过的 scale code + 一个 global_scale，既保留块内精度，又显著压缩存储。论文报告平均每参数再省约 0.4 bit，对 65B 模型就是 3–4GB 的额外节省。


### 🟡 Paged Optimizers 是什么？为什么 Adam 状态会撑爆显存？

**回答要点**：

Adam 优化器需要存两份状态：第一动量 m 和第二动量 v，通常都是 fp32，再加上权重本身是 fp32，等于"权重 + 梯度 + m + v"四份拷贝。对 65B 模型来说，Adam 状态本身就约 520GB（4×65×2），比权重还多。所以即使权重做了 4bit 量化，Adam 状态没优化依然会 OOM。Paged Optimizers 的做法是：当 GPU 显存紧张时，自动把 m、v 卸载到 CPU 内存（page out），需要更新参数时再 page in 回 GPU，整个过程对训练代码几乎透明。这是 bitsandbytes 库里 PagedAdamW 等优化器的核心能力，配合 accelerate 一起用。优点是不用改训练逻辑、显存紧张时自动兜底；代价是极端小 batch 下 CPU–GPU 搬运会有少量延迟，但论文报告端到端影响可忽略。一句话总结：权重 4bit 解决了"模型本身多大"的问题，Paged Optimizers 解决了"训练时还要再算四份临时数据"的问题。


### 🟡 QLoRA 在实际工程里怎么用？请给出完整的 BitsAndBytesConfig 配置。

**回答要点**：

完整配置通常是这样的：`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16", bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")`。四个参数含义：load_in_4bit 启用 4bit 加载；bnb_4bit_quant_type="nf4" 用 NormalFloat 而不是 FP4；bnb_4bit_use_double_quant=True 开启二次量化；bnb_4bit_compute_dtype="bfloat16" 指定前向计算时的反量化目标精度（bf16 比 fp16 数值范围大，训练更稳）。接着用 `AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")` 加载模型。再用 `LoraConfig(task_type=TaskType.CAUSAL_LM, r=8 或 16, lora_alpha=16 或 32, lora_dropout=0.05, bias="none", target_modules=["q_proj","k_proj","v_proj","o_proj"])` + `get_peft_model(model, lora_config)` 注入 LoRA，最后 `model.print_trainable_parameters()` 验证可训练参数占比 0.1%–1%。整体技术栈：bitsandbytes + transformers + peft + accelerate。


### 🔴 QLoRA 微调一个 65B 模型具体需要多少显存？为什么？

**回答要点**：

QLoRA 把 65B 模型做到单卡 48GB 即可微调，论文原话。拆开算账：①权重 4bit → 65B × 0.5B = 32.5GB（NF4 存储）；②LoRA 参数 r=16 → 假设挂了 q/k/v/o + gate/up/down 八个投影层，参数量约 65B × 16 × 8 / (d+k) ≈ 80M，fp16 占 ~160MB，可忽略；③前向反量化到 bf16 的临时 buffer 通常 4–8GB；④激活值 checkpoint + 序列长度相关，单卡 8k 上下文约 2–4GB；⑤优化器状态只针对 LoRA 那 80M 参数，Adam fp32 ≈ 640MB，可忽略；Paged Optimizers 在紧张时把 m/v 卸载到 CPU。算下来静态显存峰值约 40–48GB，正好压在一张 A100 80GB 或两张 RTX 4090 48GB 的预算内。如果不做 QLoRA，fp16 LoRA 的 65B 单是权重就要 130GB，根本塞不进任何单卡。回答时按"权重 / LoRA / 激活 / 优化器"四项拆账，面试官会觉得你真的算过。


### 🔴 QLoRA 和 DoRA 是什么关系？可以叠加用吗？

**回答要点**：

QLoRA 和 DoRA 是 LoRA 的两个独立升级方向，思路完全不同。QLoRA 解决"显存不够"——把 base 量化到 4bit 抠存储；DoRA 解决"控制粒度太粗"——把权重 W 拆成"方向 m / 标量幅度 ‖V‖_c"，分别做微调，理论上更接近全参数微调的能力。两者完全可以叠加：用 4bit 量化基座加载 + DoRA 分解 + LoRA 低秩增量，这就是论文报告的 QDoRA，在 Vicuna benchmark 上 65B 达到约 ChatGPT 99% 的水平，超过纯 QLoRA 一点点。回答时讲清楚：QLoRA 是"省显存"主线，DoRA 是"提精度"主线，叠加后是"既省显存又提精度"。面试官通常还会追问：那为什么不直接用 Full FT？因为 Full FT 65B 要存四份 fp32 拷贝共 520GB，根本跑不起；QDoRA 用 0.1% 参数达到 99% 的 Full FT 效果，性价比远超 Full FT。


### 🟡 训练完后，QLoRA 的 LoRA 适配器怎么处理？会拖累推理速度吗？

**回答要点**：

不会拖累。QLoRA 训练出的 LoRA 适配器有两种使用方式：①合并模式——把 BA 加回到 W₀，得到一个和原模型完全等价的 fp16 或 4bit 模型，推理时和普通模型一样，无任何额外延迟；②外挂模式——保留 4bit 基座 + fp16 LoRA，推理时临时做 W₀ + BA = W_eff 再算，延迟略高但支持多任务热切换。生产部署一般用合并模式，把多个 LoRA 适配器分别合并到 W₀ 的不同副本做 A/B，或者用 peft 的 `model.merge_and_unload()` 一键合并。一句话总结：训练阶段用 4bit + LoRA 省显存，部署阶段可以无损合并回 fp16，对在线推理零影响。这也是 QLoRA 比纯 4bit 推理量化（GPTQ/AWQ）灵活的地方——后者只能训后量化、不能继续微调，QLoRA 则把训练和推理两条路都打通了。

