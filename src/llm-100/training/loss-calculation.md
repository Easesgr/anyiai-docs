---
title: 大模型训练中的 Loss 是怎么计算的？从移位对齐到困惑度 PPL
icon: mdi:tune-vertical
tag:
- 训练原理
- 大模型100讲
- 预训练
description: 解析因果语言模型交叉熵损失计算机制：详细推导时序移位对齐、Softmax与NLL Loss算例、Padding掩码过滤及困惑度PPL换算，附PyTorch验证代码与练习。
---

因果语言模型（Causal Language Model）的训练目标是“给定前面的词，预测下一个词”。

在代码实现中，损失函数并不是直接把模型输出和原始输入标签送进交叉熵。如果你直接把形状为 `(批大小, 序列长度, 词表大小)` 的模型输出与输入的文本标签做对比，模型会瞬间学会“自己抄自己”，损失迅速降为 0，但推理时完全无法生成连贯文本。

本讲承接上一篇 [自回归语言模型为什么能预测下一个词？](./autoregressive-lm.md)，通过一个贯穿全文的 4 词文本例子，讲透自回归损失计算中的**时序错开对齐（Shift）**、**负对数似然损失（NLL Loss）**、**填充符掩码（loss_mask）**以及**困惑度（Perplexity, PPL）**的换算逻辑，并在最后提供一份可直接运行的 PyTorch 校验代码与练习题。

![因果自回归语言模型损失计算全景图](/images/content-rebuild/42a9a72092ac604e/cover.png)

---

## 贯穿例子：模型到底在预测什么？

假设我们有一句训练样本：“我 爱 学 习”，经过分词（Tokenization）后得到包含 4 个整数 ID 的序列：

$$X = [x_1, x_2, x_3, x_4] = [\text{“我”}, \text{“爱”}, \text{“学”}, \text{“习”}]$$

因果语言模型的工作方式是全序列并行前向计算。在因果注意力掩码（Causal Mask）的限制下：
- 第 1 个位置 $t=1$ 输入“我”，模型只能看到“我”，输出向量 $z_1$ 用于预测下一个词——它应该逼近“爱”（$x_2$）。
- 第 2 个位置 $t=2$ 输入“爱”，结合前面的“我”，模型看到“我 爱”，输出向量 $z_2$ 用于预测下一个词——它应该逼近“学”（$x_3$）。
- 第 3 个位置 $t=3$ 输入“学”，模型看到“我 爱 学”，输出向量 $z_3$ 用于预测下一个词——它应该逼近“习”（$x_4$）。
- 第 4 个位置 $t=4$ 输入“习”，模型看到“我 爱 学 习”，输出向量 $z_4$ 预测的是这首诗或这句话后面的第 5 个词。但在当前训练样本里并没有第 5 个词，因此 $z_4$ 在本条样本中没有监督信号。

同时观察输入序列的第 1 个词“我”（$x_1$）：在它之前没有任何上文，模型不可能凭空预测出第 1 个词应该是什么。因此，“我”不能作为任何输出位置的预测目标。

这就引出了因果模型训练的核心规则：**预测输出（Logits）与真实目标（Labels）必须在时间步上错开一位对齐**。

---

## 时序对齐机制：为什么必须做 Shift？

将上述逻辑转化为张量操作。假设批大小（Batch Size）为 $B$，序列长度（Sequence Length）为 $T$，词表大小（Vocab Size）为 $V$。

模型前向传播输出的未归一化对数几率张量（Logits）记为 $Z$，形状为 $(B, T, V)$；原始文本标签张量记为 $Y$，形状为 $(B, T)$。

![Logits 与 Labels 的错位对齐机制](/images/content-rebuild/42a9a72092ac604e/fig-1.png)

为了让第 $t$ 步的输出预测第 $t+1$ 步的目标，必须执行切片移位（Shift）：

1. **预测张量（Shift Logits）**：舍弃序列最后一个时间步（因为没有后续词作为标签），保留前 $T-1$ 个步长。
   ```python
   shift_logits = logits[..., :-1, :].contiguous()  # 形状变为 (B, T-1, V)
   ```
2. **目标张量（Shift Labels）**：舍弃序列第一个时间步（因为没有前文能预测它），保留第 2 个到最后一个词。
   ```python
   shift_labels = labels[..., 1:].contiguous()  # 形状变为 (B, T-1)
   ```

### 为什么要调用 `.contiguous()`？
切片操作 `[:-1, :]` 并不会在内存中开辟新空间复制张量，它只是改变了张量的步长元数据（Stride）。如果随后直接调用 `.view(-1, V)` 将张量压平送入损失函数，PyTorch 会因为张量在内存中存储不连续而抛出运行时错误（RuntimeError）。调用 `.contiguous()` 会在必要时重新分配连续内存块，确保后续变形安全执行。

<ReadMoreLock />

---

## 交叉熵与负对数似然损失（NLL Loss）的计算算例

完成时序对齐后，张量形状变成了 $(B \times (T - 1), V)$ 与 $(B \times (T - 1))$。每个有效位置的计算，本质上都是一个词表维度 $V$ 上的单样本多分类任务。

![Softmax 归一化与负对数似然损失计算映射](/images/content-rebuild/42a9a72092ac604e/fig-2.png)

### 1. 计算公式

设某时间步模型输出的实数向量为 $z = [z_1, z_2, \dots, z_V]$，真实目标词在词表中的索引为 $y^*$。

第一步，通过 Softmax 函数将实数向量归一化为概率分布：

$$p_i = \frac{\exp(z_i)}{\sum_{j=1}^V \exp(z_j)}, \quad i \in \{1, \dots, V\}$$

第二步，真实标签的分布是独热向量（One-Hot Vector），仅目标类别 $y^*$ 位置为 1，其余全为 0。多分类交叉熵损失简化为**负对数似然损失（Negative Log-Likelihood Loss, NLL Loss）**：

$$\mathcal{L} = -\ln(p_{y^*})$$

损失的大小只由模型在真实目标类别上的预测概率决定：
- 当预测概率 $p_{y^*} \to 1$ 时，$\mathcal{L} \to 0$；
- 当预测概率 $p_{y^*} \to 0$ 时，$\mathcal{L} \to +\infty$。

在底层实现中，为了防止指数运算产生浮点数溢出，PyTorch 的 `torch.nn.CrossEntropyLoss` 并不会先算 `softmax` 再取 `log`，而是结合 Log-Sum-Exp 技巧一步完成：

$$\ln p_{y^*} = z_{y^*} - m - \ln \sum_{j=1}^V \exp(z_j - m), \quad \text{其中 } m = \max_j(z_j)$$

这保证了即使在半精度（FP16 / BF16）训练下，数值也不会产生上溢或下溢。

### 2. 单步数值算例

假设我们定义一个玩具词表，总共只有 4 个词：`["我", "爱", "学", "习"]`，词表大小 $V = 4$，索引依次为 $0, 1, 2, 3$。

在位置 $t=2$（模型看完了“我 爱”）：
- 模型给出的预测 Logits 为：$z = [1.0, 2.0, 3.0, 1.0]$
- 真实目标是下一个词“学”，对应索引 $y^* = 2$

按步骤计算损失：
1. 计算各个分量的指数值 $\exp(z_i)$：
   - $\exp(1.0) \approx 2.7183$
   - $\exp(2.0) \approx 7.3891$
   - $\exp(3.0) \approx 20.0855$
   - $\exp(1.0) \approx 2.7183$
2. 计算分母配分和：
   - $\sum \exp(z_j) = 2.7183 + 7.3891 + 20.0855 + 2.7183 = 32.9112$
3. 计算目标词“学”（索引 2）的预测概率：
   - $p_2 = \frac{20.0855}{32.9112} \approx 0.6103$
4. 计算单步交叉熵损失：
   - $\mathcal{L} = -\ln(0.6103) \approx 0.4938$

如果模型继续优化，将目标词“学”的分数提升到 $5.0$（其他保持不变），则 $p_2$ 会升至 $0.9205$，单步损失降至 $0.0829$。

---

## 批处理填充与损失掩码（loss_mask）

在真实训练中，数据是以小批量（Batch）形式输入的。不同句子的长短并不一致，为了拼成矩形张量，必须对较短的序列进行尾部填充（Padding）。

![批处理填充与损失掩码过滤机制](/images/content-rebuild/42a9a72092ac604e/fig-3.png)

填充标记（Padding Token，记为 `[PAD]`）不是真实的语言内容，它带来两个必须处理的问题：
1. **不能让模型去拟合 `[PAD]`**：若强迫模型预测填充符，模型在生成时就会频繁吐出无意义的占位符。
2. **不能让填充位置稀释平均损失**：如果一个批次总共有 1000 个位置，其中 800 个是填充符，若直接除以 1000，算出的损失会被严重拉低，梯度幅度被错误削弱。

### ignore_index 与 loss_mask 机制

在工程上，通用做法是设定一个特殊的占位值 `-100`（PyTorch `CrossEntropyLoss` 的默认 `ignore_index` 参数值）：

1. **构建 Labels**：把所有不需要计算损失的位置（如填充词位置）的标签设为 `-100`。在指令微调（SFT）中，用户输入的 Prompt 部分也可以置为 `-100`，使模型仅对模型回答（Response）部分计算损失。
2. **移位切片**：移位后，原本位于尾部的 `-100` 仍然保留在对应位置。
3. **有效位置归一化**：

设一个批次在移位后总共有 $N = B \times (T - 1)$ 个位置，每个位置的损失为 $\ell_k$，目标值为 $y_k$。整体平均损失为：

$$\mathcal{L}_{\text{batch}} = \frac{\sum_{k=1}^N \mathbb{I}(y_k \neq -100) \cdot \ell_k}{\sum_{k=1}^N \mathbb{I}(y_k \neq -100)}$$

其中指示函数 $\mathbb{I}(y_k \neq -100)$ 仅在标签不是 `-100` 时为 1。分母严格等于**当前批次中实际参与计算的有效 Token 数量**，而不是张量尺寸乘积 $N$。

---

## 训练损失（Loss）与评估困惑度（Perplexity, PPL）

在大模型训练日志和学术报告中，除了记录 Loss，还会频繁使用**困惑度（Perplexity，简称 PPL）**作为模型语言建模能力的评估指标。

### 1. 数学推导与换算公式

在测试序列 $X = [x_1, x_2, \dots, x_M]$ 上，困惑度定义为测试集联合概率几何平均数的倒数：

$$PPL(X) = \left( \prod_{i=1}^M P(x_i \mid x_{<i}) \right)^{-\frac{1}{M}}$$

对两边取自然对数：

$$\ln(PPL) = -\frac{1}{M} \sum_{i=1}^M \ln P(x_i \mid x_{<i})$$

等式右侧恰好是模型在该序列上每个有效 Token 的**平均交叉熵损失 $\overline{\mathcal{L}}$**：

$$\ln(PPL) = \overline{\mathcal{L}}$$

两边取以自然底数 $e$ 为底的指数，即可得到二者的换算关系：

$$PPL = \exp(\overline{\mathcal{L}})$$

这也是 Hugging Face 官方文档中计算评估困惑度的标准方式：直接对评估损失取指数 `math.exp(eval_loss)`。

![交叉熵损失与困惑度 PPL 的换算关系](/images/content-rebuild/42a9a72092ac604e/fig-4.png)

### 2. 困惑度的物理含义：等效候选分支数

困惑度直观代表了模型在预测下一个词时，平均在多少个“等概率候选词”中犹豫不决：
- **理论下限（理想状态）**：若平均损失 $\overline{\mathcal{L}} = 0$（每次预测概率都是 1.0），则 $PPL = e^0 = 1$。说明模型毫无困惑，精准命中。
- **未训练状态**：假设词表大小 $V = 32000$，模型输出完全随机均匀分布，每个词预测概率均为 $1/32000$。此时平均损失为 $-\ln(1/32000) = \ln(32000) \approx 10.37$，对应困惑度 $PPL = e^{10.37} = 32000$。说明模型的不确定性等价于在 32000 个选项里随机抓阄。
- **训练收敛中**：若在验证集上测得交叉熵损失为 $2.3026$，则 $PPL = \exp(2.3026) \approx 10$。这意味着模型预测下一个词时的困惑程度，等价于在一个包含 10 个等可能选项的列表里做单选。

PPL 越低，模型的预测置信度越高，生成文本的连贯性和流畅度通常越好。

---

## 常见错误排查与分析

在编写自定义训练流程或处理自定义数据集时，损失计算部分最容易出现以下三类隐蔽错误：

![时序对齐错误排查对比](/images/content-rebuild/42a9a72092ac604e/fig-5.png)

### 错误 1：遗漏 Shift 直接计算损失
直接把未移位的 `logits`（形状 `(B, T, V)`）和 `labels`（形状 `(B, T)`）压平传入交叉熵函数：
```python
# 错误写法：未做错位对齐
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
```
- **表现**：模型训练损失迅速收敛到非常接近 0 的数值。
- **原因**：因为因果注意力允许第 $t$ 个位置看到第 $t$ 个位置自身的输入词嵌入，如果目标标签也是第 $t$ 个词，模型只需把当前输入的向量直接复制到输出层即可完成任务（恒等映射）。
- **后果**：模型完全丧失自回归推断能力，推理解码时陷入乱码或无限死循环。

### 错误 2：切片后未调用 `.contiguous()` 导致张量内存不连续
切片操作改变了张量内存的步长。如果不调用 `.contiguous()` 就直接使用 `.view()` 展平，PyTorch 会抛出运行时异常：`RuntimeError: view size is not compatible with input tensor's size and stride`。

### 错误 3：填充位置未设为 `-100`，直接参与反向传播
如果分词器将 Padding Token 编码为 0，而构建 labels 时没有将填充位置设为 `-100`，交叉熵会将 0 当作一个正常的类别进行学习。
- **后果**：模型为了拉低损失，会在每一个生成的步长都给 `0` 分配极高概率，导致模型输出大量无意义的填充符。

---

## 可验证代码：手工移位与掩码计算

下面给出一份使用纯 PyTorch 编写的完整验证脚本。脚本模拟了一个包含填充序列的小批次，分别通过**手工掩码拆解**与**标准接口调用**两种方式计算损失，并打印困惑度。

![Hugging Face 内部前向传播与损失计算流向](/images/content-rebuild/42a9a72092ac604e/fig-6.png)

```python
import torch
import torch.nn.functional as F


def verify_causal_lm_loss():
    batch_size = 2
    seq_len = 4
    vocab_size = 6
    pad_token_id = 0
    ignore_index = -100

    # 1. 构造模拟输入与标签
    # 样本 0：有效长度 4，序列为 [3, 4, 2, 5]
    # 样本 1：有效长度 2，尾部填充 0，序列为 [1, 2, 0, 0]
    input_ids = torch.tensor([[3, 4, 2, 5], [1, 2, 0, 0]], dtype=torch.long)

    # 构建训练标签：将 Padding 部分标记为 ignore_index (-100)
    labels = input_ids.clone()
    labels[labels == pad_token_id] = ignore_index

    # 2. 模拟模型输出的 Logits (B, T, V)
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # -----------------------------------------------------
    # 方式 A：标准 Shift 切片与 PyTorch 接口计算
    # -----------------------------------------------------
    # Logits 舍弃最后一个时间步，保留前 T-1 步
    shift_logits = logits[..., :-1, :].contiguous()
    # Labels 舍弃第一个时间步，保留后 T-1 步
    shift_labels = labels[..., 1:].contiguous()

    # 压平为二维与一维张量，满足 CrossEntropy 接口输入要求
    flat_shift_logits = shift_logits.view(-1, vocab_size)
    flat_shift_labels = shift_labels.view(-1)

    # 计算标准交叉熵损失（内部自动忽略 -100）
    standard_loss = F.cross_entropy(
        flat_shift_logits, flat_shift_labels, ignore_index=ignore_index
    )

    # -----------------------------------------------------
    # 方式 B：底层掩码过滤与有效分母求均值拆解
    # -----------------------------------------------------
    # 计算全部位置未平均的单点损失
    unreduced_loss = F.cross_entropy(
        flat_shift_logits, flat_shift_labels, reduction="none"
    )
    # 构造有效掩码
    valid_mask = flat_shift_labels != ignore_index

    # 仅统计有效 Token 的损失之和，并除以有效 Token 数量
    active_loss_sum = unreduced_loss[valid_mask].sum()
    num_active_tokens = valid_mask.sum()
    step_by_step_loss = active_loss_sum / num_active_tokens

    # -----------------------------------------------------
    # 计算困惑度 PPL
    # -----------------------------------------------------
    perplexity = torch.exp(standard_loss)

    print(f"移位后 Logits 形状: {shift_logits.shape}")
    print(f"移位后 Labels 张量:\n{shift_labels}")
    print(f"有效参与计算的 Token 数量: {num_active_tokens.item()}")
    print(f"标准接口损失: {standard_loss.item():.4f}")
    print(f"拆解计算损失: {step_by_step_loss.item():.4f}")
    print(f"对应困惑度 (PPL): {perplexity.item():.4f}")

    assert torch.isclose(
        standard_loss, step_by_step_loss
    ), "标准接口计算与底层拆解计算结果不一致！"


if __name__ == "__main__":
    verify_causal_lm_loss()
```

运行上述代码，终端输出如下：

```text
移位后 Logits 形状: torch.Size([2, 3, 6])
移位后 Labels 张量:
tensor([[   4,    2,    5],
        [   2, -100, -100]])
有效参与计算的 Token 数量: 4
标准接口损失: 2.1068
拆解计算损失: 2.1068
对应困惑度 (PPL): 8.2217
```

从结果中可以看到：
- 移位后的序列长度从 4 缩减为 3。样本 0 的有效监督目标为 `[4, 2, 5]`（3 个 Token），样本 1 的有效监督目标为 `[2, -100, -100]`（仅 1 个 Token）。
- 扁平化后总位置数为 $2 \times 3 = 6$，但有效 Token 数量为 4。损失分母严格除以 4，两个无意义的填充位置被完全过滤。

---

## 进阶练习与自测

### 练习目标
掌握自回归因果语言模型在微调（SFT）场景下的 Prompt 损失掩码处理。

### 练习任务
假设你正在执行单条问答对微调，输入序列由 Prompt（提示词）和 Response（回答）拼接而成：
- Token 序列为：`[10, 20, 30, 40, 50]`
- 其中 Prompt 部分为前 3 个 Token：`[10, 20, 30]`
- Response 部分为后 2 个 Token：`[40, 50]`

**要求**：
1. 构建对应的 `labels` 张量，要求模型只对 Response 部分计算损失，Prompt 部分不贡献损失。
2. 对 `logits` 和 `labels` 进行 Shift 移位。
3. 写出移位后的 `shift_labels` 张量内容，并统计最终参与损失计算的有效 Token 数量。

<details>
<summary>点击查看参考答案与解析</summary>

**解析**：
1. 构造初始 labels 时，将 Prompt 部分（索引 0, 1, 2）设为 `-100`：
   `labels = [-100, -100, -100, 40, 50]`
2. 进行移位切片：`shift_labels = labels[1:]`，保留后 4 个步长。
3. 切片结果为：
   `shift_labels = [-100, -100, 40, 50]`
4. 参与损失计算的有效 Token：
   - 第 1 个目标是预测 `-100`（原输入 20 对应的目标，被忽略）；
   - 第 2 个目标是预测 `-100`（原输入 30 对应的目标，被忽略）；
   - 第 3 个目标是预测 `40`（结合前文 10, 20, 30 预测第一个回答词 40，参与计算）；
   - 第 4 个目标是预测 `50`（结合前文 10, 20, 30, 40 预测第二个回答词 50，参与计算）。
   
   因此，参与损失计算的有效 Token 数量严格等于 **2**。
</details>

---

理解了前向传播中损失函数的计算机制后，接下来需要解决的核心问题是：算出来的这个标量损失，如何沿着网络反向传播转化为权重矩阵的梯度，并更新数以百亿计的参数？请阅读下一讲：[梯度下降、反向传播在大模型里如何工作？](./gradient-descent-backprop.md)。
