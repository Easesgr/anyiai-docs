---
title: Adapter、Prefix Tuning 与 P-Tuning 有什么区别？早期参数高效微调的机制与工程权衡
icon: mdi:tune-vertical
tag:
- 微调与对齐
- 大模型100讲
- RLHF
description: 对比 Adapter、Prefix Tuning 与 P-Tuning 的结构实现、数学形式与工程局限，解释它们在现代推理中被 LoRA
  取代的原因。
---

在优化大语言模型时，全参数微调需要更新并保存整个模型的所有权重。当模型体量达到数十亿或数百亿参数时，这一方式不仅需要庞大的训练显存，还会为每个下游任务产生一份完整的模型副本，带来沉重的存储与部署负担。

为了降低微调门槛，参数高效微调（Parameter-Efficient Fine-Tuning，简称 PEFT）应运而生。其核心策略是：**冻结预训练基座模型的大部分或全部参数，仅通过向模型中引入极少量的新参数（通常低于总参数量的 1%）来拟合下游任务**。在以 LoRA 为代表的旁路低秩分解成为主流之前，学术界与工业界主要沿着两条思路探索：

1. **在模型层间串联插入微型瓶颈层**：代表方案为 **Adapter Tuning**。
2. **在输入或注意力机制中注入连续的虚拟向量**：代表方案为 **Prefix Tuning** 与 **P-Tuning（v1 与 v2）**。

本讲通过一个贯穿的 7B 模型场景，解析这三类方案的数学构造、运行机制，以及它们在工程部署中面临的时延与显存权衡。

![全参数微调与参数高效微调在显存与存储维度的演化路径](/images/content-rebuild/3aa63e6b50d8993e/cover.png)

*图 0：全参数微调与参数高效微调（PEFT）在显存占用与多任务权重管理维度的技术演进。讲解员梳理了从全参数更新到层内插入、键值前缀与软提示的技术脉络。*

---

### 贯穿场景与全参数微调的开销测算

以一个参数量 $\Phi = 7 \times 10^9$（7B）的语言模型为例，假设每个参数使用 16 位浮点数（FP16 或 BF16，每个参数 2 字节）存储，隐藏层维度 $d = 4096$。我们需要针对“金融问答”和“代码生成”两个不同的下游任务分别进行微调。

如果采用全参数微调，并使用标准的 AdamW 优化器，在混合精度训练下，仅**静态训练状态**就需要占用巨额显存（此处尚未计入前向传播的激活值显存和上下文缓存）：

1. **模型静态权重**：$7\text{B} \times 2\text{ 字节} = 14\text{ GB}$；
2. **梯度张量**：与参数量一致，以 16 位浮点数存储，占用 $7\text{B} \times 2\text{ 字节} = 14\text{ GB}$；
3. **AdamW 优化器状态**：需要维护 32 位浮点数（FP32）的主权重副本（4 字节）、一阶动量（4 字节）以及二阶动量（4 字节），合计每个参数 12 字节，占用 $7\text{B} \times 12\text{ 字节} = 84\text{ GB}$。

这意味着仅静态状态就需要 $14 + 14 + 84 = 112\text{ GB}$ 的显存，单张 80GB 的 GPU 无法直接装载启动训练。同时，每个微调任务都会生成一份 14 GB 的完整模型文件，当面对数十个业务场景时，权重存储与多租户切卡加载成本很高。

PEFT 的基本思想是将基座参数 $\Theta_0$ 保持冻结，仅更新少量新增参数 $\Delta\Phi$。因为优化的参数量大幅缩减，优化器状态所占用的显存随之骤降，每个任务也仅需保存几兆到几十兆的外挂权重。

<ReadMoreLock />

---

### Adapter Tuning：层间插入瓶颈适配器

Houlsby 等人于 2019 年提出的 Adapter 结构，是最早系统性应用于预训练 Transformer 的参数高效微调方法之一。

#### 结构与数学公式

Adapter 的基本思想是在 Transformer 的特定子层之后插入一个结构微型的前馈瓶颈网络（Bottleneck），并配有残差连接。

假设进入 Adapter 的隐藏特征为 $h \in \mathbb{R}^{B \times T \times d}$，其中 $B$ 为批大小，$T$ 为序列长度，$d$ 为模型隐藏维度：

$$\text{Adapter}(h) = \sigma(h W_{\text{down}} + b_{\text{down}}) W_{\text{up}} + b_{\text{up}} + h$$

各变量含义如下：
- **下投影矩阵** $W_{\text{down}} \in \mathbb{R}^{d \times r}$：将维度从 $d$ 压缩至较低的瓶颈维度 $r$（通常 $r \ll d$，例如取 $r=64$）；
- **非线性激活函数** $\sigma(\cdot)$：通常使用 ReLU 或 GELU；
- **上投影矩阵** $W_{\text{up}} \in \mathbb{R}^{r \times d}$：将维度从 $r$ 恢复至模型的原始维度 $d$；
- **残差连接** $+ h$：确保在初始化阶段（例如 $W_{\text{up}}$ 初始化接近于 0 时），模块的输出接近恒等映射，不破坏基座模型原有能力。

![Adapter 瓶颈层在 Transformer Block 内部的串联数据流](/images/content-rebuild/3aa63e6b50d8993e/fig-1.png)

*图 1：Houlsby 经典 Adapter 模块在 Transformer Block 内部的串联插装拓扑。讲解员展示了特征从下投影降维、经过非线性激活、再经上投影恢复并与残差合并的全过程。*

在经典 Houlsby 架构中，每个 Transformer 层内部通常会串联插入两个 Adapter 模块：一个紧随多头注意力（MHA）投影层之后，另一个紧随前馈网络（FFN）之后。后来的 Pfeiffer 等人简化了该设计，仅在 FFN 输出后保留一个 Adapter 模块。

#### 为什么 Adapter 无法在推理时静态合并？

在推理部署中，工程师常希望把微调学到的增量权重与原权重相加，从而消除额外的新层。然而 Adapter **无法直接通过矩阵相加合并进主干全连接层**。

原因在于中间夹带的**非线性激活函数** $\sigma$。在线性代数中，对于非线性函数 $\sigma$：

$$\sigma(h W_{\text{down}}) W_{\text{up}} \neq h (W_{\text{down}} W_{\text{up}})$$

既然无法将其简化为一个单一的等效线性矩阵，在自回归推理的每一步（解码每一个 Token 时），GPU 都必须依次执行下投影矩阵乘法、非线性激活核函数、上投影矩阵乘法以及残差加法。

这会导致**GPU 算子调度开销（Kernel Launch Overhead）**。大语言模型自回归推理在小批次场景下往往受限于显存带宽与算子启动时延。层层堆叠微小的串联计算会切碎计算流，引起端到端推理时延上升。

---

### Prefix Tuning：注意力键值空间的连续虚向量追加

为了避免在网络层间引入新的串联算子，Li & Liang 于 2021 年提出了 Prefix Tuning（前缀微调）。

#### 注意力层拼接机制

Prefix Tuning 不改变模型各层的计算拓扑，而是在多头自注意力计算时，在 Key 和 Value 矩阵的序列维度前直接拼接一组可学习的连续虚拟向量。

设输入序列的投影向量分别为查询 $Q$、键 $K$、值 $V \in \mathbb{R}^{T \times d}$。Prefix Tuning 保持模型参数冻结，为每层定义两个可训练的前缀张量 $P_K, P_V \in \mathbb{R}^{l \times d}$，其中 $l$ 为设定的前缀长度（通常在 10 到 100 之间）：

$$K_{\text{prefix}} = [P_K; K], \quad V_{\text{prefix}} = [P_V; V]$$

其中 $[\cdot ; \cdot]$ 表示在序列长度（Token 步数）维度的拼接。注意力计算随之变为：

$$\text{Attention}(Q, K_{\text{prefix}}, V_{\text{prefix}}) = \text{softmax}\left( \frac{Q [P_K; K]^\top}{\sqrt{d_k}} \right) [P_V; V]$$

![Prefix Tuning 在注意力层追加 Key-Value 虚拟前缀向量](/images/content-rebuild/3aa63e6b50d8993e/fig-2.png)

*图 2：Prefix Tuning 在 Multi-Head Attention 计算中的虚拟键值注入机制。讲解员指出了 Query 保持原状、Key 和 Value 拼接前缀向量后的注意力矩阵形态变化。*

在这一机制下，$Q$ 保持长度 $T$，而键与值的序列长度扩展至 $l + T$。在计算注意力分布时，真实的 Token 可以自由分配权重给这 $l$ 个虚拟前缀向量，从而引导后续的输出分布。

#### 为什么需要重参数化（Reparameterization）？

如果直接将 $P_K$ 与 $P_V$ 作为可学习的独立参数并随机初始化，通过梯度下降直接优化时容易出现训练不收敛、对学习率极度敏感的现象。

为了让优化过程平滑，Prefix Tuning 提出了**重参数化机制**：
1. 定义一个较小维度的嵌入矩阵 $E \in \mathbb{R}^{l \times d_{\text{mid}}}$；
2. 使用一个双层前馈网络（包含 Tanh 激活函数）将 $E$ 映射出各层所需的键值前缀：
   $$P_K^{(i)}, P_V^{(i)} = \text{MLP}^{(i)}(E)$$
3. 训练期间，仅更新 $E$ 与 MLP 的参数；
4. **推理期解耦**：训练完成后，将固定的 $E$ 输入 MLP 计算一次，直接保存生成后的各层张量 $P_K^{(i)}, P_V^{(i)}$。在实际推理时丢弃 MLP，仅将静态前缀张量预先存入每层的 KV Cache 头部。

![Prefix Tuning 训练期重参数化与推理期前缀解耦流程](/images/content-rebuild/3aa63e6b50d8993e/fig-3.png)

*图 3：Prefix Tuning 在训练阶段依赖 MLP 重参数化保障梯度平滑，并在推理部署期将前缀解耦并存为静态张量的全生命周期流程。*

---

### P-Tuning v1 与 v2：从单层软提示到多层深度前缀

与面向文本生成的 Prefix Tuning 相似，清华团队提出了针对自然语言理解（NLU）任务的 P-Tuning 体系。

#### P-Tuning v1：输入层的连续软提示与时序编码

传统的离散提示工程（Prompt Engineering）通过人工编写模板（例如“下面这段话的情感是：[MASK]”），但模型输出对模板措辞极其敏感。Prompt Tuning 尝试在输入 Embedding 层添加连续可微的虚拟 Token 向量。

然而，当直接优化输入层的虚拟 Token 时，由于预训练模型的 Embedding 向量分布高度非线性且离散，从头训练出的孤立向量彼此缺乏上下文关联，导致收敛缓慢。

P-Tuning v1 做出了两点改进：
1. **提示编码器（Prompt Encoder）**：使用一个双向 LSTM（BiLSTM）配合双层 MLP 来生成虚拟 Token 的输入向量：
   $$h_i = \text{MLP}([\overrightarrow{\text{LSTM}}(p_i); \overleftarrow{\text{LSTM}}(p_i)])$$
   借助 LSTM 的时序平滑归纳偏置，让前后相邻的虚拟 Token 之间具备连续关联性。
2. **锚定标记（Anchor Tokens）**：在虚拟 Token 序列中穿插少量离散的真实词（如标点或关键字），在连续向量空间中建立参照点。

#### P-Tuning v2：向多层深度前缀（Deep Prompt Tuning）演进

P-Tuning v1 的局限在于**仅作用于最底层的输入 Embedding 层**：
- 浅层引入的扰动在向前传播通过几十层复杂的 Transformer 模块时，对深层隐藏状态的控制力容易减弱；
- 反向传播时，监督信号的梯度需要穿透所有 Transformer 层才能回传至输入层，在中小模型上优化难度较大。

P-Tuning v2 对此进行了架构调整：
- **逐层注入（Deep Prompts）**：放弃只在输入层追加软提示，转为在模型的**每一个 Transformer 层**都引入独立的可学习前缀张量。这与 Prefix Tuning 在每一层扩展 Key 与 Value 的思想完全契合。
- **移除 LSTM 编码器**：当每一层都拥有独立的前缀向量时，各层可以直接接收来自深层的梯度反向传播，此时直接优化前缀参数即可稳定收敛，不再需要外挂 BiLSTM 编码器。
- **支持更广泛的任务**：使百亿参数量以下的中小模型在抽取式问答、命名实体识别等任务上取得了接近全参数微调的表现。

![P-Tuning v1 输入层软提示与 P-Tuning v2 多层深度前缀的结构对比](/images/content-rebuild/3aa63e6b50d8993e/fig-4.png)

*图 4：P-Tuning v1（仅在输入 Embedding 层通过 LSTM 编码注入）与 P-Tuning v2（在各 Transformer 层逐层引入深度连续前缀）的拓扑对比。*

---

### 机制验证：最小前向计算与张量维度实验

通过以下精简的 PyTorch 示例，可以直观观察 Adapter 串联结构与 Prefix 拼接机制在前向传播中的张量维度行为：

```python
import torch
import torch.nn as nn
import math

class ToyTransformerLayerWithPEFT(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4, adapter_dim=8, prefix_len=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.prefix_len = prefix_len
        
        # 1. 模拟冻结的主干注意力映射
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 2. 模拟串联 Adapter (瓶颈层)
        self.adapter_down = nn.Linear(hidden_dim, adapter_dim, bias=False)
        self.adapter_act = nn.ReLU()
        self.adapter_up = nn.Linear(adapter_dim, hidden_dim, bias=False)
        
        # 3. 模拟 Prefix Tuning 键值前缀
        self.prefix_k = nn.Parameter(torch.randn(num_heads, prefix_len, self.head_dim))
        self.prefix_v = nn.Parameter(torch.randn(num_heads, prefix_len, self.head_dim))
        
    def forward(self, x, mode="baseline"):
        B, T, C = x.shape
        
        # 投影计算 Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        if mode == "prefix":
            # 在序列长度维度拼接前缀，扩展为 [B, num_heads, prefix_len + T, head_dim]
            pk = self.prefix_k.unsqueeze(0).expand(B, -1, -1, -1)
            pv = self.prefix_v.unsqueeze(0).expand(B, -1, -1, -1)
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
            
        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(attn_out) + x  # 主残差连接
        
        if mode == "adapter":
            # 串联执行瓶颈变换并累加残差
            adapter_h = self.adapter_act(self.adapter_down(out))
            adapter_out = self.adapter_up(adapter_h)
            out = out + adapter_out
            
        return out, scores.shape

# 验证维度变化
layer = ToyTransformerLayerWithPEFT()
x_sample = torch.randn(2, 8, 64)  # Batch=2, SeqLen=8, Dim=64

out_base, shape_base = layer(x_sample, mode="baseline")
out_adapter, shape_adapter = layer(x_sample, mode="adapter")
out_prefix, shape_prefix = layer(x_sample, mode="prefix")

print(f"Baseline 注意力矩阵形状: {shape_base}")
print(f"Adapter 注意力矩阵形状:  {shape_adapter}")
print(f"Prefix 注意力矩阵形状:   {shape_prefix}")
```

运行上述代码，控制台输出为：
```text
Baseline 注意力矩阵形状: torch.Size([2, 4, 8, 8])
Adapter 注意力矩阵形状:  torch.Size([2, 4, 8, 8])
Prefix 注意力矩阵形状:   torch.Size([2, 4, 8, 12])
```

从形状变化可以清晰观察到机制差异：
- Adapter 保持注意力矩阵的维度（`8 x 8`）不变，但在最终输出路径上增加了额外的非线性全连接运算；
- Prefix Tuning 将注意力得分矩阵由原本的 `8 x 8` 扩展到了 `8 x 12`（增加了设定的前缀长度 4），增加了注意力点积与归一化的计算负载。

---

### 三类方案的横向技术特征对比

| 微调方案 | 插入位置 | 典型参数量占比 | 训练时是否需要重参数化 | 能否静态合并入原权重 | 适用场景特点 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Adapter Tuning** | MHA 或 FFN 层之后的串联路径 | 1% - 3% | 否 | **否**（受非线性激活阻断） | 通用特征适配、模型层数较浅的分类任务 |
| **Prefix Tuning** | 每一层 MHA 的 Key 与 Value 空间 | 0.1% - 1% | **是**（需低维 MLP 辅助平滑收敛） | **否**（作为前缀向量常驻 KV Cache） | 条件生成任务（如摘要、翻译、表格生成） |
| **P-Tuning v1** | 输入 Embedding 词表层 | < 0.1% | **是**（依赖 BiLSTM 建模时序连续性） | 可直接固化为词向量 | 短文本分类、抽取式问答（NLU） |
| **P-Tuning v2** | 每一层的多头注意力键值空间 | 0.1% - 1% | 否（深层直接接收梯度，收敛较稳） | **否**（作为深度前缀常驻多层） | 覆盖更广的全功能 NLU 任务 |

---

### 早期 PEFT 方案在现代大模型部署中的工程局限

在以自回归生成为核心的大语言模型工程落地过程中，Adapter、Prefix Tuning 与 P-Tuning 逐渐不再是线上服务的首选方案。理解其工程瓶颈，有助于理解为何后续行业普遍转向了 LoRA 类方案。

![早期 PEFT 方案在生产部署时面临的推理时延与上下文挤占瓶颈](/images/content-rebuild/3aa63e6b50d8993e/fig-5.png)

*图 5：早期 PEFT 方案在生产环境中遭遇的工程瓶颈剖析：包含 Adapter 串联小算子带来的执行时延上升，以及 Prefix 挤占上下文窗口与 KV Cache 显存容量。*

#### 局限一：无法消除的额外推理计算时延
- **Adapter 的算子碎片化**：如前所述，由于非线性激活函数存在，Adapter 无法与原有全连接层提前折叠。在具有 32 甚至 80 层的现代大模型中，自回归生成的每一步都需要频繁调用这些小尺寸矩阵乘法与激活函数，增加了 GPU Kernel 启动与调度等待时间。
- **Prefix 与 P-Tuning 的持续注意力计算**：由于前缀作为虚拟 Token 始终存在于 Key 和 Value 中，自回归解码每个新 Token 时，注意力点积计算的序列长度都必须包含这部分前缀，无法完全等同于原生基座的运算开销。

#### 局限二：有效上下文长度缩减与 KV Cache 膨胀
- **上下文窗口挤占**：如果 Prefix Tuning 或 P-Tuning v2 设置的前缀长度 $l = 128$，对于最大上下文长度为 4096 的模型，留给真实用户输入与输出的可用窗口长度就缩减为 $4096 - 128 = 3968$。
- **KV Cache 显存占用增加**：自回归生成高度依赖显存中的键值缓存（KV Cache）。以前缀长度 $l = 128$、隐藏层维度 $d = 4096$、共 32 层、批大小 $B = 16$ 为例：
  $$\text{新增 KV 缓存} = 2 \times 32 \times 16 \times 128 \times 4096 \times 2\text{ 字节} \approx 1\text{ GB}$$
  在高并发服务下，这部分由于虚拟前缀产生的显存占用将直接压缩并发服务能承载的最大请求量（Concurrency Batch Size）。

#### 局限三：动态批处理（Dynamic Batching）与 PagedAttention 调度难度
在现代推理服务框架（如 vLLM、TGI）中，系统通过显存分页管理（PagedAttention）与连续批处理调度（Continuous Batching）来提升吞吐。如果服务网关同时挂载多个针对不同任务微调的前缀模型，且各自的前缀长度 $l$ 不一致，管理与对齐不规则的 KV 显存块会显著增加调度开销。

![不同微调方案在任务类型、上下文敏感度与延迟要求下的工程选型图谱](/images/content-rebuild/3aa63e6b50d8993e/fig-6.png)

*图 6：参数微调方案的工程决策逻辑图。讲解员对照检查清单，指出当要求“零推理时延”且“不挤占上下文”时，技术路线必然走向低秩重参数化方案。*

正因为早期方案在推理时延、上下文长度与显存缓存方面的取舍，后续微软提出的 LoRA（Low-Rank Adaptation）改用**旁路线性低秩分解**：
$$\Delta W = B \times A$$
在推理前，只需执行 $W = W_0 + \Delta W$ 即可实现**静态权重合并**。合并后的模型在推理阶段不增加任何算子调用，不改变序列长度，也不占用多余的 KV Cache。在下一讲 [/llm-100/finetuning/lora-efficient-finetuning.md](/llm-100/finetuning/lora-efficient-finetuning.md) 中，我们将详细解析 LoRA 的数学原理与具体应用。

---

### 进阶验证练习

请使用本讲提供的 `ToyTransformerLayerWithPEFT` 模块完成以下可验证练习：

**任务：量化并验证 Adapter 与 Prefix 的可训练参数量**

编写一段测试脚本，统计在隐藏维度 $d=4096$、注意头数 $h=32$、单层 Transformer 下：
1. 采用瓶颈维度 $r=64$ 的单个 Houlsby 风格 Adapter 模块所包含的可训练参数量（不含偏置）；
2. 采用前缀长度 $l=32$ 时，单层 Prefix Tuning 所包含的独立前缀张量参数量（即 $P_K$ 与 $P_V$ 的参数总和）。

**检验参考代码与预期输出**：
```python
import torch
import torch.nn as nn

hidden_dim = 4096
adapter_dim = 64
prefix_len = 32

# 1. 计算 Adapter 参数量: W_down (4096 x 64) + W_up (64 x 4096)
adapter_params = (hidden_dim * adapter_dim) + (adapter_dim * hidden_dim)

# 2. 计算 Prefix 参数量: P_K (32 x 4096) + P_V (32 x 4096)
prefix_params = 2 * (prefix_len * hidden_dim)

print(f"单层 Adapter 参数量: {adapter_params} ({adapter_params / 1e6:.3f}M)")
print(f"单层 Prefix 参数量:  {prefix_params} ({prefix_params / 1e6:.3f}M)")
```
预期输出：
```text
单层 Adapter 参数量: 524288 (0.524M)
单层 Prefix 参数量:  262144 (0.262M)
```
通过该练习可以验证：在典型超参数配置下，单层的 PEFT 参数量均在数十万级（0.2M - 0.5M 左右），仅为原层全连接权重（数十兆）的很小一部分。
