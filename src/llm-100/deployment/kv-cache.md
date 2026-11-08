---
title: KV Cache 是什么？为什么能加速大模型解码推理？
icon: mdi:book-open-page-variant
description: 解释自回归推理中 KV Cache 的原理、单步与累积复杂度对比、显存占用精确计算公式与显存碎片挑战，包含 PyTorch 实测验证代码。
---

在[大模型推理为什么慢？瓶颈在哪里？](/llm-100/deployment/inference-bottlenecks.md)中，我们分析了自回归解码阶段具有明显的访存受限（Memory-Bound）特征。在模型逐字生成文本的过程中，决定单步时延与并发上限的核心组件，就是 **键值缓存（Key-Value Cache，简称 KV Cache）**。

大语言模型的推理通常分为两个阶段：
1. **预填充阶段（Prefill）**：将用户输入的整个 Prompt 并行送入网络，计算并输出第一个生成词；
2. **自回归解码阶段（Decode）**：每次前向传播只输入上一步生成的单个新词，并预测出下一个词，循环往复直到生成结束。

如果在解码阶段不保存中间状态，每次预测新词都必须把从开头到当前的所有 Token 重新输入网络计算一遍。KV Cache 的作用就是在 GPU 显存中持久化保存所有历史 Token 的 Key（键）和 Value（值）张量，让每一步前向计算只针对当前这一个新词展开，用显存空间换取大幅度的时间节省。

![KV Cache 封面：以显存换取自回归解码时间的架构权衡](/images/content-rebuild/dc6d6f5a2b96e769/cover.png)

---

### 贯穿示例：模型是如何一步步生成文本的？

为了直观理解 KV Cache 到底存了什么、省了什么，我们看一个具体的例子。

假设用户输入的 Prompt 是 `"今天天气"`（2 个 Token），模型接下来依次生成 `"真"`、`"好"` 两个词。

#### 1. 没有 Cache 的运行过程
* **第 1 步（Prefill）**：输入 `["今天", "天气"]`。网络计算这两个词在所有网络层的 Q、K、V 投影，通过因果自注意力计算，输出下一个词 `"真"`。
* **第 2 步（Decode 1）**：输入序列变成 `["今天", "天气", "真"]`（3 个 Token）。模型把这 3 个词从第 1 层到最后一层全部重新计算一遍投影矩阵，得到新的 Q、K、V 并输出 `"好"`。前两个词在此前已经计算过的表示被完全重复计算了一次。
* **第 3 步（Decode 2）**：输入 `["今天", "天气", "真", "好"]`（4 个 Token），整整 4 个词再次全量过一遍网络。

每多生成一个词，全序列就要重算一遍，序列越长，单步计算耗时就越高。

#### 2. 启用 KV Cache 的运行过程
* **第 1 步（Prefill）**：输入 `["今天", "天气"]`。模型计算这两个词的 Q、K、V，在预测出 `"真"` 的同时，把 `"今天天气"` 在每一层对应的 $K_1, K_2$ 和 $V_1, V_2$ 写入 GPU 显存中保存下来。
* **第 2 步（Decode 1）**：输入**仅为单个词** `"真"`。网络仅对 `"真"` 做线性变换，得到当前步的 $q_3, k_3, v_3$。接着把 $k_3, v_3$ 写入显存，与历史的 $[k_1, k_2]$ 和 $[v_1, v_2]$ 拼成完整的 3 个 K 和 V。随后，仅用当前步的单个 $q_3$ 与历史拼接后的 3 个 K 做点积打分，对 3 个 V 加权求和，输出 `"好"`。
* **第 3 步（Decode 2）**：输入**仅为单个词** `"好"`。网络仅计算当前步的 $q_4, k_4, v_4$，追加写入缓存，直接预测下一个词。

在前向投影阶段，历史 Token 的投影计算被彻底免除，单步输入始终为 1 个 Token。

---

### 为什么历史 K、V 在后续生成中可以保持不变？

为什么我们能放心地复用历史计算出来的 K 和 V，而不用担心它们随未来词的产生而失效？这是由 Transformer 解码器的**因果自注意力（Causal Self-Attention）**机制决定的。

对于包含 $t$ 个 Token 的序列，第 $l$ 层注意力的计算形式为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

其中 $M$ 是因果掩码矩阵（Causal Mask）。行索引代表当前查询位置 $i$，列索引代表被注意的键位置 $j$。由于生成只能参考过去、不能看到未来，所有 $j > i$ 的位置都被掩码赋予负无穷大：$M_{i,j} = -\infty$。

![时序因果掩码与历史 KV 冻结机制说明](/images/content-rebuild/dc6d6f5a2b96e769/fig-1.png)

这意味着：
1. 在任意层中，位置 $\tau$ 的隐藏层输出向量 $h_\tau$，只能聚合位置 $1$ 到 $\tau$ 的信息，完全无法感知任何后续位置 $\tau+1, \dots, t$ 的输入；
2. 当模型网络权重固定时，只要历史上下文 $x_{1:\tau}$ 不变，位置 $\tau$ 处在该层产生的所有隐藏特征就完全恒定；
3. 每一层的 Key 和 Value 是由该层输入的隐藏特征通过权重矩阵线性映射得到的（$k_\tau = h_\tau W_K, v_\tau = h_\tau W_V$）。因此，$k_\tau$ 和 $v_\tau$ 一旦在第 $\tau$ 步算好，在后续的所有生成步骤中数值完全不变。

每一步真正变化并需要和全部历史做比较的，只有当前新 Token 计算出的当前步 Query 向量 $q_t$：

$$\alpha_t = \text{softmax}\left(\frac{q_t \cdot [k_1, k_2, \dots, k_t]^T}{\sqrt{d_k}}\right)$$

既然历史的 $k_{1:t-1}$ 与 $v_{1:t-1}$ 数值恒定不变，提前将其缓存起来直接读取，就省去了每一层对历史 Token 的重复前向映射计算。需要说明的是，单 Token 解码因为历史缓存中不存在未生成的未来 Token，所以在单序列标准全注意力下无需因果掩码；但若涉及批量 Padding 填充、滑动窗口局部注意力或推测解码（Speculative Decoding）多 Token 验证，解码阶段仍须使用对应的注意力掩码。

---

### 计算复杂度对比：单步开销与累积开销

评估 KV Cache 的收益，需要将“单步生成时延”与“生成整段长文本的累积总开销”区分开来看。

设输入 Prompt 长度为 $S_0$，自回归阶段连续生成 $N$ 个新 Token，序列最终长度增长到 $S_0 + N$。

#### 1. 无 Cache 模式
* **单步计算特征**：生成第 $t$ 个词时，系统必须把完整的 $t$ 个 Token 重新输入网络。包含 $W_Q, W_K, W_V$ 以及全连接前馈网络（FFN）在内的线性层投影，单步计算量为 $O(t \cdot d_{\text{model}}^2)$；自注意力点积为 $t \times t$ 维度的矩阵乘法（GEMM），单步计算量为 $O(t^2 \cdot d_{\text{model}})$。
* **累积总开销**：在连续生成 $N$ 步的过程中对单步求和：
  * 线性投影累积计算量：$\sum_{t=S_0+1}^{S_0+N} O(t \cdot d_{\text{model}}^2) = O(N^2 \cdot d_{\text{model}}^2)$（当 $N \gg S_0$ 时）；
  * 注意力累积计算量：$\sum_{t=S_0+1}^{S_0+N} O(t^2 \cdot d_{\text{model}}) = O(N^3 \cdot d_{\text{model}})$。
* **工程表现**：单步生成耗时随着文本变长而越来越慢，长文本生成的后期出现极高延迟。

#### 2. 启用 KV Cache 模式
* **单步计算特征**：Prefill 阶段一次性计算 Prompt 对应的初始 $K_{1:S_0}, V_{1:S_0}$。进入 Decode 阶段后，每一步模型输入的序列长度恒定为 $1$。线性层投影的单步计算量恒定为 $O(1 \cdot d_{\text{model}}^2)$；注意力计算退化为单个 Query 向量与缓存中 $t$ 个 Key 向量的乘法（GEMV），单步开销仅为 $O(t \cdot d_{\text{model}})$。
* **累积总开销**：在连续生成 $N$ 步的过程中（对应序列总长度 $t = S_0+1, \dots, S_0+N$）：
  * 线性投影累积计算量降为：$\sum_{i=1}^{N} O(1 \cdot d_{\text{model}}^2) = O(N \cdot d_{\text{model}}^2)$，复杂度从 $O(N^2 \cdot d_{\text{model}}^2)$ 降为 $O(N \cdot d_{\text{model}}^2)$（当 $N \gg S_0$ 时）；
  * 注意力累积计算量降为：$\sum_{t=S_0+1}^{S_0+N} O(t \cdot d_{\text{model}}) = O((N \cdot S_0 + N^2) \cdot d_{\text{model}})$，当 $N \gg S_0$ 时为 $O(N^2 \cdot d_{\text{model}})$，复杂度从 $O(N^3 \cdot d_{\text{model}})$ 降为 $O(N^2 \cdot d_{\text{model}})$。

![无 Cache 与有 Cache 计算量与单步耗时趋势对比](/images/content-rebuild/dc6d6f5a2b96e769/fig-2.png)

#### 算力密集型向访存受限型的转移

虽然计算次数大幅缩减，但系统瓶颈发生了根本性转变：
* 在 Prefill 阶段，全序列并行参与计算，GPU 可以充分发挥张量核心的大规模矩阵乘法（GEMM）吞吐，表现为**算力密集型（Compute-Bound）**；
* 在 Decode 阶段，单步输入的张量极小（仅 1 个 Token），但每一步都必须将所有历史 Token 的 KV 缓存从 GPU 的高带宽显存（HBM）完整读取到片上 SRAM 中进行点积计算。算力利用率极低，GPU 绝大部分时间耗费在等待显存数据传输上，系统彻底转变为**显存带宽密集型（Memory-Bound）**。

---

### KV Cache 的张量维度与存储布局

在代码实现中，KV Cache 是模型每个注意力层中独立维护的多维张量。

设模型总层数为 $L$，每层的 Key/Value 注意力头数为 $H_{\text{kv}}$，每个注意力头的隐藏维度为 $D_{\text{head}}$。当并发处理的批次大小为 $B$，当前已处理的序列总长度为 $S$ 时，单层中 Key 与 Value 各自对应一个独立的四维张量。

在主流框架的标准实现中，单层的 KV Cache 张量形状通常为：

$$\text{Shape}_{K} = \text{Shape}_{V} = [B, H_{\text{kv}}, S, D_{\text{head}}]$$

![KV Cache 多维张量内存结构与动态扩展](/images/content-rebuild/dc6d6f5a2b96e769/fig-3.png)

四个维度的含义如下：
1. **批次大小（$B$）**：并行处理的请求数量，各请求之间互相独立维护自身缓存；
2. **KV 头数（$H_{\text{kv}}$）**：Key 和 Value 的独立头数量（由具体的注意力变体架构决定）；
3. **序列长度（$S$）**：当前请求累积的 Token 总数。随着生成推进，每输出一个新词，该维度的长度就自增 1；
4. **头维度（$D_{\text{head}}$）**：单个注意力头的向量长度（通常为 64 或 128），满足 $H_q \times D_{\text{head}} = d_{\text{model}}$。

全模型总共有 $L$ 对这样的张量。在解码单步中，当前层前向传播仅产生形状为 $[B, H_{\text{kv}}, 1, D_{\text{head}}]$ 的新切片，并沿序列维度与已有缓存拼接。

---

### 显存占用计算公式与 MHA、GQA、MQA 差异

部署大语言模型时，显存不仅要装下模型本身的权重参数，还要为运行时的 KV Cache 留足空间。在长文本与多并发场景下，KV Cache 占用的显存往往会超过模型权重本身。

#### 1. 通用显存计算公式

对于任意 Transformer 解码模型，单个 Token 在所有网络层累积占用的 KV Cache 净显存大小为：

$$\text{Bytes}_{\text{per\_token}} = 2 \times L \times H_{\text{kv}} \times D_{\text{head}} \times P$$

各参数含义如下：
* **系数 $2$**：同时存储 Key 和 Value 两个张量；
* **$L$**：模型网络层数（`num_hidden_layers`）；
* **$H_{\text{kv}}$**：Key/Value 的注意力头数（`num_key_value_heads`）；
* **$D_{\text{head}}$**：单头隐层向量维度（`head_dim`）；
* **$P$**：每个参数占用的字节数（FP16/BF16 为 2 字节，FP8 为 1 字节，INT4 为 0.5 字节）。

当服务并发请求数为 $B$，平均上下文长度达到 $S$ 时，系统所需的 KV Cache 显存总量为：

$$\text{Total\_Memory}_{\text{kv}} = B \times S \times \text{Bytes}_{\text{per\_token}}$$

#### 2. MHA、GQA 与 MQA 的显存差异

为了减轻 KV Cache 对显存的压力，模型架构在注意力头的组织方式上经历了多次演进：

* **多头注意力（MHA，Multi-Head Attention）**：$H_{\text{kv}} = H_q$。每个 Query 头配有一组独立的 Key/Value 头，KV Cache 显存占用最高；
* **多查询注意力（MQA，Multi-Query Attention）**：$H_{\text{kv}} = 1$。所有 Query 头共享唯一一组 Key/Value 头，显存占用降至极低，但在复杂长文本建模上容量容易受限；
* **分组查询注意力（GQA，Grouped-Query Attention）**：将 $H_q$ 个 Query 头划分为多个组，每组共享一个 Key/Value 头，即 $H_{\text{kv}} = H_q / G$。GQA 在维持与 MHA 相当建模性能的同时大幅削减了显存占用，是目前开源大模型（如 Llama、Qwen 系列）的主流选择。

![MHA、GQA 与 MQA 的头结构与显存占用对比](/images/content-rebuild/dc6d6f5a2b96e769/fig-4.png)

#### 3. 典型算例：Llama-3-70B

以 **Llama-3-70B** 为例进行量化换算（参数：$L=80, H_q=64, H_{\text{kv}}=8, D_{\text{head}}=128$，采用 16 位浮点精度 $P=2$ 字节）：

单个 Token 在该模型全网络层占用的显存为：

$$\text{Bytes}_{\text{per\_token}} = 2 \times 80 \times 8 \times 128 \times 2 \text{ 字节} = 327,680 \text{ 字节} = 320 \text{ KiB}$$

如果单条请求的上下文长度达到 $S = 4096$：

$$\text{Memory}_{4096} = 4096 \times 327,680 \text{ 字节} = 1,342,177,280 \text{ 字节} = 1.25 \text{ GiB}$$

**对比分析**：如果该模型使用早期的 MHA 架构（即 $H_{\text{kv}} = 64$），单条 4096 序列的 KV 显存将高达 **10 GiB**。正是因为 GQA 将 KV 头数缩减为 8，才使得显存开销压缩到原来的八分之一，为推理服务留出了成倍的并发容纳能力。更完整的推理显存构成计算，可参阅[模型部署显存估算手册](/interview/model-deployment-interview/memory-estimation.md)。

---

### 显存管理困境：预分配与显存碎片

KV Cache 虽然解决了重复计算问题，却给显存管理带来了新难题：**自回归生成的输出长度事先是未知的**。在模型吐出终止符之前，系统无法预知请求最终会写下 10 个词还是 2000 个词。

而在常规深度学习框架中，张量必须占用一段**物理连续的显存空间**。面对这种动态增长，朴素的显存分配策略会陷入两难：

![连续显存预分配造成的显存碎片与并发压制](/images/content-rebuild/dc6d6f5a2b96e769/fig-5.png)

#### 策略 1：静态预分配（Pre-allocation）
为了避免在解码途中因为显存用尽导致请求崩溃，系统在请求进入时，就直接按照最大序列长度（例如 `max_seq_len = 4096`）开辟一块完整的连续显存空间。
* **内部碎片严重（显存空洞）**：在实际业务中，许多问答在生成数十个词后就提前结束。预留给 4096 长度的显存空间中，超过 80% 从未被真正使用，却在请求结束前被硬性占用锁定；
* **并发上限被锁死**：假设一张显卡扣除模型权重后仅剩 20 GiB 可用显存，若每条请求按 4096 预留 1.25 GiB，系统最多只能同时跑 $20 / 1.25 = 16$ 个并发请求。即使当前 16 条请求每条才刚生成 3 个词，显存也无法接纳第 17 个请求。

#### 策略 2：动态扩容拼接（Dynamic Re-allocation）
如果一开始只按当前长度分配显存，每生成一个新词就调用一次拼接操作（如 `torch.cat`）：
* **显存复制与延迟抖动**：当张量后面的物理地址已被其他张量占用时，底层分配器必须在显存中开辟一块更大的新空间，把历史数据完整复制过去并释放旧空间。频繁的显存重分配和数据复制会引发明显的生成延迟抖动；
* **外部碎片化与伪 OOM**：频繁的分配和释放会导致物理显存布满细小的碎片空洞。即便系统剩余的总显存充足，也会因为找不到一段足够大的连续地址而抛出显存溢出（Out-Of-Memory，OOM）异常。

这种物理连续分配与不可预测长度之间的矛盾，正是促使业界借鉴操作系统的虚拟内存分页机制、设计出分页键值缓存的原因。这部分技术方案将在下一讲的 [PagedAttention 机制](/llm-100/deployment/paged-attention.md) 中详细展开。

---

### 动手实验：用最小 PyTorch 循环验证加速效果

下面给出一个自包含的 PyTorch 最小验证代码。代码实现了一个简易的自注意力模块，分别模拟“无 Cache 模式”与“启用 KV Cache 模式”，并在连续生成 32 步的过程中测量每一步前向传播的耗时变化。

```python
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward_without_cache(self, x_seq):
        # x_seq 形状: [batch, seq_len, d_model]
        B, S, _ = x_seq.shape
        Q = self.q_proj(x_seq).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_seq).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_seq).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # 施加因果掩码遮蔽未来位置
        mask = torch.triu(torch.full((S, S), float('-inf'), device=x_seq.device), diagonal=1)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, self.d_model)
        # 解码时仅取最后一个 Token 的隐层输出用于预测下一个词
        return self.out_proj(out[:, -1:, :])

    def forward_with_cache(self, x_in, kv_cache=None):
        # x_in 形状: [batch, seq_len, d_model]，兼容 Prefill (seq_len > 1) 与 Decode (seq_len = 1)
        B, S, _ = x_in.shape
        q = self.q_proj(x_in).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x_in).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x_in).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        if kv_cache is None:
            K, V = k_new, v_new
        else:
            past_k, past_v = kv_cache
            K = torch.cat([past_k, k_new], dim=-2)
            V = torch.cat([past_v, v_new], dim=-2)
            
        scores = torch.matmul(q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Prefill 阶段需掩码遮蔽未来；单 Token 解码在无 padding 的全注意力场景下无需因果掩码
        if S > 1:
            mask = torch.triu(torch.full((S, S), float('-inf'), device=x_in.device), diagonal=1)
            scores = scores + mask
            
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(out), (K, V)

# 运行测试
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MinimalAttention().to(device).eval()
steps = 32
prompt_len = 128

print(f"设备: {device} | Prompt 长度: {prompt_len} | 解码步数: {steps}")

# 实验 1: 无 Cache 模式
inputs_history = torch.randn(1, prompt_len, 512, device=device)
times_no_cache = []
with torch.no_grad():
    for step in range(steps):
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.forward_without_cache(inputs_history)
        if device == "cuda": torch.cuda.synchronize()
        times_no_cache.append((time.perf_counter() - t0) * 1000)
        # 模拟生成新词后追加到输入序列
        inputs_history = torch.cat([inputs_history, torch.randn(1, 1, 512, device=device)], dim=1)

# 实验 2: 启用 KV Cache 模式
cache = None
times_with_cache = []
prompt_input = torch.randn(1, prompt_len, 512, device=device)
with torch.no_grad():
    # 预填充阶段建立初始缓存
    _, cache = model.forward_with_cache(prompt_input, kv_cache=None)
    for step in range(steps):
        next_token = torch.randn(1, 1, 512, device=device)
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, cache = model.forward_with_cache(next_token, kv_cache=cache)
        if device == "cuda": torch.cuda.synchronize()
        times_with_cache.append((time.perf_counter() - t0) * 1000)

print(f"第 1 步时延  -> 无 Cache: {times_no_cache[0]:.3f} ms | 有 Cache: {times_with_cache[0]:.3f} ms")
print(f"第 32 步时延 -> 无 Cache: {times_no_cache[-1]:.3f} ms | 有 Cache: {times_with_cache[-1]:.3f} ms")
```

![实验验证有无 KV Cache 的解码单步时延曲线](/images/content-rebuild/dc6d6f5a2b96e769/fig-6.png)

从实验结果中可以观察到两个核心现象：
1. **无 Cache 模式耗时持续爬升**：因为每一步都需要将累加后的整个序列重新输入模型，矩阵运算量随长度增长，耗时呈明显上升趋势；
2. **有 Cache 模式耗时低且维持平稳**：每一步只对 1 个 Token 执行线性映射，线性层计算量为常数。虽然注意力点积阶段的乘法次数随历史长度微增，但在该规模下其增量极小，单步解码耗时显著更低且维持相对平稳。

---

### 学习检查与进阶练习

完成本讲学习后，建议通过以下练习验证对 KV Cache 机制与显存规划的掌握情况：

1. **手动算例推导**：
   查阅某开源模型（如 Qwen2.5-7B，参数：$L=28, H_q=28, H_{\text{kv}}=4, D_{\text{head}}=128$，精度为 BF16）。假设需要部署该模型支持并发数 $B=16$、最大上下文长度 $S=8192$，请计算该场景下系统仅为 KV Cache 需要预留多少 GiB 显存。
2. **代码修改验证**：
   在上面的验证代码中加入 `torch.cuda.memory_allocated()` 的打印调用，记录在启用 KV Cache 模式下，随着连续解码推进，GPU 显存占用的具体数值变化，并观察步进增加的显存量是否符合理论预期张量尺寸大小。

在掌握了 KV Cache 的空间与时间权衡之后，我们将进入下一讲：现代推理框架（如 [vLLM](/llm-100/deployment/vllm-inference.md)）是如何利用非连续分页技术彻底解决预分配显存空洞与显存碎片的——请阅读 [PagedAttention 解决了什么问题？](/llm-100/deployment/paged-attention.md)。
