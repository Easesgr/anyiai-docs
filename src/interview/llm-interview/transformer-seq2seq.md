---
title: Transformer 如何实现序列到序列映射？端到端数据流与面试核心拆解
icon: mdi:swap-horizontal
tag:
- Transformer
- Encoder-Decoder
- Seq2Seq
description: 解析标准 Transformer 的 Seq2Seq 映射机制，涵盖编码器与解码器分工、训练期并行 Teacher Forcing、推理期自回归生成与暴露偏差。
---

## 30秒口述结论

标准 Transformer（Encoder-Decoder 架构）实现序列到序列映射，核心依赖两套模块分工与两种运行模式：

1. **结构分工**：编码器使用无掩码的双向自注意力，一次性提取源序列全局上下文特征；解码器包含两层注意力，先用因果自注意力只看历史已生成词，再通过交叉注意力（Cross-Attention）检索编码器的输出特征，完成源序列到目标序列的对齐。
2. **训练模式（并行 Teacher Forcing）**：目标文本向右移动一位，并在开头补上 `<BOS>` 作为输入，真实标签作为预测目标。配合下三角因果掩码遮蔽未来时刻，GPU 可以在单次前向传播中同时算出目标序列所有位置的损失，无需串行循环。
3. **推理模式（串行自回归展开）**：编码器对源序列只执行一次前向计算，输出全局上下文表征向量；解码器各层分别据此计算并缓存交叉注意力的 Key 和 Value 矩阵。解码器从输入 `<BOS>` 开始，每一步根据已生成的历史词与交叉注意力缓存预测下一个词，追加到输入后继续循环，直到输出终止符 `<EOS>` 或达到设定的最大长度。

![Transformer Seq2Seq 架构端到端数据流全景](/images/content-rebuild/a30580bf012b8d3c/cover.png)

---

## 贯穿示例：英译中任务全流程

为了清晰说明张量如何在网络中流转，本文以一个简单的机器翻译样本贯穿全程：

- **源输入文本**：`Knowledge is power`（长度 $S=3$）
- **目标真实文本**：`知识 就是 力量 <EOS>`（长度 $T=4$）

两端序列长度通常不同（$S \ne T$），模型必须在不固定长度绑定的前提下完成映射。

![Transformer Seq2Seq 内部三层注意力机制拓扑流向](/images/content-rebuild/a30580bf012b8d3c/fig-1.png)

---

## 结构组成：三类注意力子层的分工

编码器与解码器内部共使用了三类注意力，输入来源和掩码规则各不相同：

| 注意力子层类型 | Query 来源 | Key / Value 来源 | 掩码策略 | 职责说明 |
| :--- | :--- | :--- | :--- | :--- |
| **编码器自注意力**<br>*(Encoder Self-Attention)* | 编码器上一层输出 | 编码器上一层输出 | 仅 Padding 掩码 | 计算源文本所有词两两之间的关联，生成上下文语义向量。 |
| **解码器因果自注意力**<br>*(Decoder Causal Self-Attention)* | 解码器上一层输出 | 解码器上一层输出 | Padding 掩码 +<br>因果下三角掩码 | 提取目标文本已生成词的依赖，遮蔽未来词。 |
| **交叉注意力**<br>*(Cross-Attention)* | 解码器下层掩码自注意力输出 | 编码器最后一层输出 ($Z_{enc}$) | 源序列 Padding 掩码 | 解码器主动检索源文本中与当前生成位置相关的信息。 |

### 交叉注意力如何连接两端

交叉注意力是连接源序列与目标序列的枢纽。它的数学形式如下：

$$\text{Attention}(Q_{dec}, K_{enc}, V_{enc}) = \operatorname{softmax}\left(\frac{Q_{dec} K_{enc}^T}{\sqrt{d_k}}\right) V_{enc}$$

- $Q_{dec}$ 来自解码器，形状为 $(B, H, T, d_k)$，表示目标序列 $T$ 个位置的查询向量；
- $K_{enc}$ 与 $V_{enc}$ 由解码器交叉注意力层对编码器输出表征 $Z_{enc}$ 线性投影得到，形状为 $(B, H, S, d_k)$ 和 $(B, H, S, d_v)$，表示源序列 $S$ 个词映射后的 Key 与 Value 特征；
- 矩阵乘法 $Q_{dec} K_{enc}^T$ 生成形状为 $(B, H, T, S)$ 的相关性矩阵。该矩阵第 $i$ 行代表目标词 $i$ 对源语言各位置的关注度权重；
- 经过 Softmax 归一化后乘上 $V_{enc}$，得到 $(B, H, T, d_v)$ 的表征，把源序列中与当前目标词相关的信息聚合进解码器。

---

## 训练阶段数据流：并行 Teacher Forcing 与目标序列右移

如果训练阶段也按词逐个循环步进，GPU 算力利用率会非常低。Transformer 训练时采用全序列并行计算。

<ReadMoreLock />

### 为什么目标序列必须“向右移动一位”？

训练时，模型已知完整的真实目标序列 `[知识, 就是, 力量, <EOS>]`。
如果把该序列直接输入解码器，在第一层注意力中，第 1 个位置向量本身就是“知识”的词嵌入。即使加了因果掩码遮蔽后面位置，模型只要直接抄写自身位置的静态输入，就能得到 100% 准确率，导致网络退化为恒等映射。

为了强制模型根据历史预测未来，必须将目标序列整体右移一位：

![训练阶段 Teacher Forcing 右移对齐与因果掩码并行前向机制](/images/content-rebuild/a30580bf012b8d3c/fig-2.png)

```text
时间步 t:            1         2         3         4
解码器输入序列:     <BOS>     知识      就是      力量
模型目标标签:        知识      就是      力量     <EOS>
```

通过右移并在首位填入 `<BOS>`，移出末尾的 `<EOS>`：
- 在位置 1，解码器只看到 `[<BOS>]` 和源序列，必须根据英文 `Knowledge is power` 预测出首词“知识”；
- 在位置 2，因果掩码限制解码器只能看 `[<BOS>, 知识]`，去预测“就是”；
- 在位置 4，解码器看到 `[<BOS>, 知识, 就是, 力量]`，去预测终止符 `<EOS>`。

所有位置的预测互相独立，GPU 单次前向即可算出整条序列的交叉熵损失。

```python
import torch
import torch.nn as nn

def training_step(src_tokens, tgt_tokens, model, criterion):
    # 1. 构造右移解码器输入与训练目标
    # 假设 BOS_ID=1
    batch_size = tgt_tokens.size(0)
    bos = torch.full((batch_size, 1), fill_value=1, dtype=torch.long, device=tgt_tokens.device)
    decoder_input = torch.cat([bos, tgt_tokens[:, :-1]], dim=1) # 右移一位
    targets = tgt_tokens                                        # 对应目标标签

    # 2. 构造因果掩码（严格上三角置为 -inf，保留对角线及下三角历史可见性）
    seq_len = decoder_input.size(1)
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=tgt_tokens.device), diagonal=1)

    # 3. 矩阵并行前向计算
    # 编码器提取源序列全局上下文 (B, S, D)
    encoder_memory = model.encoder(src_tokens)
    # 解码器一次性输入完整目标序列，依靠因果掩码阻隔未来信息 (B, T, D)
    logits = model.decoder(decoder_input, encoder_memory, causal_mask)

    # 4. 全位置并行计算交叉熵损失
    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    return loss
```

---

## 推理阶段数据流：单次编码与串行自回归展开

在推理阶段，模型没有目标序列的参考答案，无法并行，必须回退到自回归循环展开。

![推理阶段自回归循环与 KV 缓存机制流转图](/images/content-rebuild/a30580bf012b8d3c/fig-3.png)

整个推理流程分为两个计算阶段：

### 阶段一：编码器单次执行（Encode 阶段）
1. 源序列 `[Knowledge, is, power]` 完整送入编码器，经 $N$ 层自注意力与前馈网络，输出源端顶层表征张量 $Z_{enc}$。
2. 解码器各层交叉注意力分别利用各自独立的权重矩阵，预先计算出对应的 Key 矩阵 $K_{enc}^{(l)} = Z_{enc} W_K^{(l)}$ 和 Value 矩阵 $V_{enc}^{(l)} = Z_{enc} W_V^{(l)}$ 并保存在显存中。
3. 因为源文本在生成过程中不再变动，**编码器前向计算与解码器各层交叉注意力的静态 KV 缓存全流程只计算这一次**。

### 阶段二：解码器逐步循环生成（Decode）
1. **Step 1**：将起始符 `[<BOS>]` 输入解码器。自注意力计算当前向量，交叉注意力读取已缓存的 $K_{enc}, V_{enc}$，经线性层投影输出词表概率分布，通过贪心或采样选出首个词：“知识”。
2. **Step 2**：将新词追加到序列后，输入变为 `[<BOS>, 知识]`，计算后输出词：“就是”。
3. **Step $t$**：重复该流程。如果不使用 KV Cache，每步都要对历史所有 Token 重新算一次注意力；如果启用解码器自注意力的 KV Cache（详见 [/interview/llm-interview/kv-cache.md](/interview/llm-interview/kv-cache.md)），当前步只需传入新增的那一个 Token，计算其 $K_t, V_t$ 写入缓存，避免重复计算。

---

## 循环终止条件：EOS 判定与最大长度截断

解码器不能无休止循环下去，其退出受两个边界控制：

![自回归终止判定边界：EOS 识别与最大长度保护分支](/images/content-rebuild/a30580bf012b8d3c/fig-4.png)

1. **语义终止（命中 `<EOS>`）**：模型某一步预测概率最高的 Token（或采样选中）为 `<EOS>`，表示句子语义完整，正常结束。
2. **防御性截断（达到最大长度 $T_{max}$）**：如果模型由于退化陷入局部循环（如连续输出重复短语），可能一直无法生成 `<EOS>`。必须设置最大步数阈值 $T_{max}$，一旦达到立即强制退出，防止显存与计算资源耗尽。

---

## 面试高频追问：暴露偏差（Exposure Bias）与应对方案

面试中经常会考察训练与推理的分布差异，这被称为暴露偏差。

### 什么是暴露偏差？

- **训练期**：采用 Teacher Forcing，解码器每一步输入的历史都是**标准真值（Ground Truth）**；
- **推理期**：解码器输入的历史是**上一时刻模型自己生成的 Token（可能存在错误）**。

![暴露偏差成因机制：真值环境与误差自累积分布漂移对比](/images/content-rebuild/a30580bf012b8d3c/fig-5.png)

这种条件不一致会导致**误差级联累积**：一旦模型在第 2 步生成了一个错误的词，第 3 步面临的上下文便脱离了训练时见过的真值分布。因为模型训练时从未学过如何从错误上下文中恢复，后续预测极易快速恶化。

### 常见的应对与缓解手段

1. **计划采样（Scheduled Sampling）**：
   - 训练时按一定概率 $\epsilon$ 决定当前步输入使用真值还是模型上一时刻的预测值。前期 $\epsilon$ 大（多用真值保证收敛），后期逐渐减小 $\epsilon$ 让模型适应带噪声的输入。
   - *工程限制*：如果依赖前一步的预测，就无法在矩阵维度上全序列并行计算，会大幅降低 Transformer 在 GPU 上的训练效率。
2. **强化学习与全句损失微调（RL / DPO / Minimum Risk Training）**：
   - 在自回归生成完整句子后，计算整句层面的评分（如 BLEU 或奖励模型分值），通过强化学习或偏好对齐优化全句质量，让模型对局部偏差具备一定的容错度。
3. **推理端使用束搜索（Beam Search）**：
   - 推理时不使用单步贪心，而是维护 $K$ 条候选路径，选取整句累积概率最高的组合，降低单步失误对后续生成的毁灭性影响。

---

## 常见认知误区辨析

在面试答题时，需要澄清以下三个常见概念混淆：

![原生 Seq2Seq 架构与 Decoder-only 任务拼接机制对比](/images/content-rebuild/a30580bf012b8d3c/fig-6.png)

### 误区一：混淆 Seq2Seq 任务与 Encoder-Decoder 结构
- **辨析**：Seq2Seq 是“序列到序列”的任务形式（如机器翻译、摘要提取）。Encoder-Decoder 是实现该任务的双塔模型结构。现在的 Decoder-only 模型（如 GPT 系列）同样可以完成 Seq2Seq 任务，其做法是将 Prompt 与 Target 拼在同一序列中，依靠统一的因果注意力自回归生成（详见 [/interview/llm-interview/decoder-only.md](/interview/llm-interview/decoder-only.md)）。

### 误区二：认为解码器在训练时也是按步循环生成的
- **辨析**：训练阶段没有按步循环。借助 Teacher Forcing 和下三角因果掩码，整个目标序列作为单张矩阵一次性输入解码器，所有位置并行计算梯度，属于单次前向传播。

### 误区三：认为推理时每生成一个词，编码器都要重新算一次
- **辨析**：源序列在生成过程中内容固定。编码器仅在首步前向计算一次，输出源端表征张量 $Z_{enc}$。解码器各层交叉注意力据此投影计算的 Key 和 Value 缓存可被永久复用，后续自回归生成只在解码器内部迭代。

---

## 知识体系链接

- **前置依赖**：
  - [/interview/llm-interview/decoder-encoder-mha-diff.md](/interview/llm-interview/decoder-encoder-mha-diff.md) 拆解了自注意力与交叉注意力的矩阵维度差异。
  - [/interview/llm-interview/attention-masking.md](/interview/llm-interview/attention-masking.md) 解析了填充掩码（Padding Mask）与因果掩码（Causal Mask）的具体实现。
- **横向扩展与演进**：
  - [/interview/llm-interview/decoder-only.md](/interview/llm-interview/decoder-only.md) 对比了 Encoder-Decoder 与 Decoder-only 结构的架构取舍。
  - [/interview/llm-interview/kv-cache.md](/interview/llm-interview/kv-cache.md) 深入了解自回归推理中缓存管理与显存瓶颈的优化细节。
