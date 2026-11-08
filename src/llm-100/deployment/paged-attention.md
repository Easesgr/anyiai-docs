---
title: PagedAttention 解决了什么问题？大模型推理中的显存分页与复用
icon: mdi:book-open-page-variant
description: 解析 PagedAttention 如何借鉴虚拟内存分页消除 KV Cache 的显存碎片，详解映射寻址、专用算子与写时复制机制。
---

在自回归大语言模型生成过程中，每个请求都需要缓存历史 Token 的 Key 和 Value 向量（即 KV Cache），以避免每一轮生成都从头做投影计算。KV Cache 的基础原理可参考 [/llm-100/deployment/kv-cache.md](/llm-100/deployment/kv-cache.md)。然而在实际服务部署中，显存往往先于算力耗尽，直接制约了并发请求的处理能力。

这一瓶颈的核心症结在于**传统连续显存分配方式在面对动态未知的序列长度时，产生了大量无法利用的显存碎片**。PagedAttention 借鉴操作系统虚拟内存分页的设计，将逻辑连续的键值缓存切分并存放在离散的物理显存块中，从底层消除了内存碎片，并支持高效的多分支前缀复用。

![PagedAttention 核心机制概览：从连续显存分配到虚拟分页映射](/images/content-rebuild/47bd8425e707876b/cover.png)

---

### 贯穿场景：传统分配方式下的显存浪费

设想一个具体场景：我们在单张 GPU 上部署一个 7B 参数模型，扣除模型权重后，剩余 10 GB 显存用于存放 KV Cache。假设每个 Token 在所有注意力层产生的 KV 向量总计占用 0.5 MB 显存。

此时，服务接入了一个新请求，并要求系统为它准备显存空间：
- **无法预知的生成长度**：请求可能在第 50 个 Token 遇到结束符（EOS）退出，也可能持续生成到模型允许的最大长度（如 2048 个 Token）。
- **连续存储假设**：传统的深度学习框架（如初期基于 PyTorch 动态张量的推理实现）通常假设一个请求的 KV Cache 在显存中是一整段物理连续的数组。

为了满足连续性，传统系统通常有两种处理策略，但这两种策略都会导致显存浪费：

1. **策略 A：按最大长度预分配（引发内部碎片）**  
   系统按最大长度 2048 为该请求一次性申请 $2048 \times 0.5\text{ MB} = 1024\text{ MB}$ 的连续空间。如果该请求实际生成 50 个 Token 就提前结束，实际仅消耗 25 MB，剩余 999 MB 的显存槽位在整个生成周期内一直被该请求独占，其他并发请求无法使用。这种“分了却用不上”的空间称为**内部碎片（Internal Fragmentation）**。
2. **策略 B：按需动态倍增扩容（引发外部碎片）**  
   系统先分配一小段连续显存，填满后再分配翻倍的连续空间并将旧数据拷贝过去。然而，并发服务中请求到达与退出的时间完全随机、生成长度各不相同。随着大量请求交替申请与释放，显存空间会被切割得支离破碎。此时即使显存总剩余量仍有数个 GB，却无法凑出一整段足够长的连续地址来容纳新请求的连续缓冲区，导致系统被迫拒绝请求或触发 OOM。这种“零散空间总和够大，但单段连续长度不够”的现象称为**外部碎片（External Fragmentation）**。

![传统连续显存分配中的内部碎片与外部碎片成因](/images/content-rebuild/47bd8425e707876b/fig-1.png)

根据 vLLM 论文（SOSP 2023）对实际负载的测量，在传统的连续显存分配机制下，真正用来存储有效 Token KV 数据的显存比例通常只有 20% 到 40%，剩余 60% 到 80% 的显存被内部与外部碎片闲置。这严重压低了 GPU 推理服务的并发吞吐（Batch Size）。

---

### 虚拟内存分页思想在 GPU 显存中的映射

操作系统在管理物理内存时，为了解决多进程连续分配的碎片问题，提出了**虚拟内存分页**：把进程的虚拟地址切分为固定大小的“页面（Page）”，物理内存切分为等大的“页帧（Page Frame）”，两者通过页表（Page Table）记录映射。应用程序看到的是连续的虚拟地址，物理内存则完全离散。

PagedAttention 将这种设计迁移至 GPU 显存中的 KV Cache 管理：

- **逻辑块（Logical Block）**：将每个请求生成的 Token 序列，按照固定长度 $B$（Block Size，如 16 个 Token）切分为逻辑块。从请求视角看，逻辑块按 0, 1, 2... 递增。
- **物理块（Physical Block）**：在系统启动时，将用于 KV Cache 的显存池预先切分为固定大小的物理块。一个物理块刚好容纳 $B$ 个 Token 在模型所有层与注意力头上的 Key 和 Value 数据。
- **块表（Block Table）**：为每个活跃请求维护一张映射表，记录该请求的各个逻辑块分别对应显存池中的哪一个物理块 ID，以及最后一个物理块当前填充的 Token 数量。

![逻辑块与物理块在 Block Table 协调下的解耦映射机制](/images/content-rebuild/47bd8425e707876b/fig-2.png)

显存池由于全部切分为等长的小型物理块，分配操作变成了简单的“从空闲链表中取出一个物理块 ID”，释放操作则是“将块 ID 放回空闲链表”。所有空闲块都可以无差别地供给任意请求使用，因此**彻底消除了外部碎片**。

---

### 寻址计算与自回归物理块追加

在离散物理块架构下，模型在每一步计算注意力时，需要准确知道第 $t$ 个 Token 的 KV 向量存储在显存的什么位置（$t$ 从 0 开始编号）。寻址计算由整除与取模完成：

1. **计算逻辑块索引**：
   $$\text{Logical Block Index} = \lfloor t / B \rfloor$$
2. **计算块内偏移**：
   $$\text{Block Offset} = t \pmod B$$
3. **查表获得物理块 ID**：
   $$\text{Physical Block ID} = \text{BlockTable}[\text{Logical Block Index}]$$
4. **换算物理显存地址**：
   根据物理块基准地址与单个 Token 占用的跨度（Stride），计算目标显存偏移：
   $$\text{Address} = \text{BaseAddr} + (\text{Physical Block ID} \times \text{Stride}_{\text{block}}) + (\text{Block Offset} \times \text{Stride}_{\text{token}})$$

![Token 寻址与自回归增量写入的物理块分配流程](/images/content-rebuild/47bd8425e707876b/fig-3.png)

在自回归 Decode 阶段，系统每一步只为当前序列产出一个新 Token（设当前已有 $L$ 个 Token，即将写入第 $t = L$ 个 Token）：
- 若 $t \pmod B \ne 0$：说明当前最后一个物理块还有空闲槽位，系统直接在当前物理块的偏移 $t \pmod B$ 处写入新计算的 Key 和 Value。
- 若 $t \pmod B == 0$：说明前一个物理块已填满（或序列刚开始），显存管理器从全局空闲池中取出一个新的物理块追加到该请求的 Block Table 末尾，并将新 Token 写入该块的偏移 0 处。

下面是用 Python 模拟 Block Table 映射与自回归物理块增量分配的逻辑实现：

```python
class PagedKVMemoryManager:
    def __init__(self, block_size: int, total_blocks: int):
        self.block_size = block_size
        # 全局空闲物理块池（存储物理块 ID）
        self.free_blocks = list(range(total_blocks))
        # 记录每个物理块的引用计数
        self.ref_counts = [0] * total_blocks

    def allocate_block(self) -> int:
        if not self.free_blocks:
            raise MemoryError("GPU KV Cache 显存池已耗尽 (OOM)")
        block_id = self.free_blocks.pop(0)
        self.ref_counts[block_id] = 1
        return block_id

    def free_block(self, block_id: int):
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            self.free_blocks.append(block_id)

    def compute_address(self, block_table: list[int], token_idx: int) -> tuple[int, int]:
        """返回 (physical_block_id, block_offset)"""
        logical_idx = token_idx // self.block_size
        offset = token_idx % self.block_size
        physical_block_id = block_table[logical_idx]
        return physical_block_id, offset

    def append_token(self, block_table: list[int], current_seq_len: int) -> None:
        """自回归生成一步：若当前块写满，则向显存池申领新物理块"""
        offset = current_seq_len % self.block_size
        if offset == 0:  # 块已满或首个 Token，开辟新物理块
            new_block = self.allocate_block()
            block_table.append(new_block)
```

---

### 专为非连续显存设计的 CUDA 注意力核函数

将数据切分成离散小块后，产生了一个新的工程问题：**标准注意力算子（如标准 cuDNN Attention 或矩阵乘法 GEMM）要求 $K$ 和 $V$ 在显存中具有连续步长**。如果每次计算注意力前，都先把离散物理块中的数据拷贝成一段连续张量，拷贝带来的访存延迟会抹平显存节省带来的吞吐收益。

因此，PagedAttention 编写了专门的 CUDA 核函数（Kernel），让 GPU 在执行注意力计算时能够**边查表边读取非连续显存**：

在自回归 Decode 阶段，输入仅为当前步新生成的 Query 向量 $q_t \in \mathbb{R}^{1 \times d}$，其计算逻辑为：
$$O_t = \text{Softmax}\left(\frac{q_t K_{0:t}^T}{\sqrt{d}}\right) V_{0:t}$$

![PagedAttention 专用 CUDA 核函数在非连续显存块上的并行计算机制](/images/content-rebuild/47bd8425e707876b/fig-4.png)

PagedAttention 核函数的执行过程如下：

1. **指针传递**：在启动核函数时，系统将各序列当前的 Query 向量与对应的 `block_table` 整数数组指针一起传给 GPU。
2. **按块迭代加载**：一个 GPU 线程块（Thread Block）或 Warp 负责处理一个注意力头。线程根据 `block_table` 中的索引，按逻辑顺序遍历各个物理块。
3. **片上显存点积**：每次将一个物理块的 $K$ 向量直接从全局显存读入共享显存（Shared Memory）或寄存器，与 $q_t$ 计算点积 $q_t \cdot K_{\text{block}}^T / \sqrt{d}$。
4. **在线流式 Softmax（Online Softmax）**：利用在线 Softmax 算法，在遍历物理块的过程中动态维护局部最大值 $m$ 与分母累加和 $l$，不需要在显存中展开完整的全序列注意力权重矩阵。
5. **Value 累加**：紧随其后加载该块对应的 $V$ 向量，将局部加权结果累加到输出寄存器中。整个过程不需要对离散显存做任何物理搬运或拼接重排。

在序列长度极长或并发批次较小时，为了充分利用 GPU 的计算单元，PagedAttention 还设计了 **Split-KV** 机制：沿序列长度维度将不同的物理块分配给多个线程块并行计算局部注意力，最后通过一个轻量级的 Reduction 核函数规约合并输出。

---

### 写时复制（Copy-on-Write）与前缀共享

实际推理场景中，存在大量多分支或包含相同上下文的请求，例如：
- **并行采样（Parallel Sampling）**：单条 Prompt 同时生成多个候选回答（`n=4`）；
- **束搜索（Beam Search）**：同时维护多个候选分支，分支间的历史前缀大部分相同；
- **长系统提示（System Prompt）/ 预置前缀**：多个独立用户的请求携带完全一致的背景说明或示例规则。

如果为每个分支或请求都完整复制一份前缀的 KV 向量，显存会成倍增加。PagedAttention 利用类似操作系统 `fork()` 的**写时复制（Copy-on-Write, CoW）**机制实现了显存复用：

![基于写时复制（CoW）的多分支前缀物理显存复用流程](/images/content-rebuild/47bd8425e707876b/fig-5.png)

其运行流程如下：

1. **前缀物理块只读共享**：处理公共 Prompt（Prefill 阶段）产生的物理块（如物理块 0 和 1）直接供所有分支共用。初始化各分支的 Block Table，使其包含相同的物理块 ID，并将这些物理块的引用计数（`ref_count`）设为分支数量（例如 4）。
2. **只读计算安全**：在后续生成中，所有分支的注意力核函数只需按表读取这些共享块。此时 4 个分支的前缀只占 1 份物理显存。
3. **写时复制触发**：
   - 当某个分支生成新 Token 时，若目标物理块尚未填满且 `ref_count > 1`，说明该块正被其他分支共用，**不能直接原地写入**（否则会污染其他分支的数据）。
   - 该分支触发 CoW：从显存池申请一个新的空闲物理块，将原共享块中已有数据拷贝到新块，原块的 `ref_count` 减 1，更新该分支 Block Table 指向新块，随后在新块的对应偏移处写入私有 Token。
   - 若目标物理块本身已满（偏移为 0 开辟新块），各分支直接申请全新的私有块（`ref_count = 1`），无需拷贝历史块中的内容。

关于 vLLM 在调度层如何结合连续批处理进一步释放前缀缓存效能，可参考 [/llm-100/deployment/vllm-inference.md](/llm-100/deployment/vllm-inference.md)。

---

### 块大小（Block Size）选型的工程权衡

物理块大小 $B$（Block Size）是系统设计中的关键超参数。它不能任意设定，必须在**显存内部碎片率**与**GPU 访存带宽效率**之间做工程权衡。

![块大小（Block Size）在显存碎片率与访存带宽利用率之间的工程权衡曲线](/images/content-rebuild/47bd8425e707876b/fig-6.png)

#### 1. 显存碎片率约束（倾向选择较小的 $B$）
在 PagedAttention 中，所有已经填满的历史物理块都没有内部碎片。显存浪费**仅发生在一个序列正在写入的最后一个未满物理块中**。设序列总长度为 $L$，块大小为 $B$：
- 最后一个物理块未填满的槽位数为 $(B - (L \pmod B)) \pmod B$。
- 最坏情况下（刚申请新块，只写入 1 个 Token），浪费 $B - 1$ 个 Token 槽位。
- 理论内部碎片率上限为：
  $$\text{Internal Fragmentation Rate} \le \frac{B - 1}{L}$$

如果选取 $B = 16$，最坏情况浪费 15 个 Token 槽位。当序列长度 $L \ge 375$ 时，内部碎片率上限不超过 $\frac{15}{375} = 4\%$；当 $L \ge 2048$ 时，最坏碎片率不足 $0.73\%$。但如果将 $B$ 设为 128 或更大，短序列场景下的内部碎片比例就会显著增加。

#### 2. GPU 访存合并约束（倾向选择较大的 $B$）
既然较小的 $B$ 能够降低碎片率，为什么不直接设为 $B = 1$？
- **显存事务与访存合并（Memory Coalescing）**：现代 GPU 的全局显存以 32 字节（Sector）甚至 128 字节缓存行（Cache Line）为粒度进行事务读取。一个包含 32 个线程的 Warp 在访问连续显存地址时，能够触发访存合并，单次指令即可充分打满数据总线。
- 若 $B$ 过小（例如 $B=1$ 或 $B=2$），不同 Token 散落于显存各处，GPU 计算核心频繁跨块寻址，引发大量非连续、未对齐的稀疏读取指令，导致有效内存带宽利用率严重下跌。KV Cache 的自回归生成属于典型的**访存受限型（Memory-bound）任务**，访存效率下降会直接拉低整个推理服务的吞吐，更多底层瓶颈分析详见 [/llm-100/deployment/inference-bottlenecks.md](/llm-100/deployment/inference-bottlenecks.md)。
- **元数据开销**：$B$ 越小，管理相同 Token 数所需的 Block Table 映射项越庞大，CPU 与 GPU 之间同步和调度元数据的管理开销呈线性增加。

综合考虑 128 字节硬件对齐与主流模型序列长度分布，主流实现通常将 **$B = 16$ 或 $B = 32$** 作为标准选型。该配置既能将长文本场景下的内部碎片控制在极低水平，又能保证注意力核函数充分打满 GPU 显存总线带宽。

---

### 进阶练习：带引用计数与写时复制的内存池

请在本地运行并阅读以下测试代码，验证 PagedAttention 显存管理的核心机制：包括块分配、写时复制（CoW）触发，以及请求释放时的引用计数回收。

```python
class PagedKVCacheSimulator:
    def __init__(self, block_size: int, total_blocks: int):
        self.block_size = block_size
        self.free_blocks = list(range(total_blocks))
        self.ref_counts = {i: 0 for i in range(total_blocks)}
        # 模拟物理块中实际存放的数据：{block_id: list}
        self.physical_memory = {i: [] for i in range(total_blocks)}

    def alloc_block(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("Out of Memory")
        b_id = self.free_blocks.pop(0)
        self.ref_counts[b_id] = 1
        self.physical_memory[b_id] = []
        return b_id

    def free_block(self, b_id: int):
        self.ref_counts[b_id] -= 1
        if self.ref_counts[b_id] == 0:
            self.physical_memory[b_id] = []
            self.free_blocks.append(b_id)

    def fork_sequence(self, block_table: list[int]) -> list[int]:
        """分叉请求：新分支共享现有物理块，引用计数递增"""
        for b_id in block_table:
            self.ref_counts[b_id] += 1
        return list(block_table)

    def append_token(self, block_table: list[int], token_val: str):
        """追加一个 Token，必要时触发 CoW 或申请新块"""
        if not block_table:
            new_b = self.alloc_block()
            block_table.append(new_b)

        curr_block = block_table[-1]
        curr_data = self.physical_memory[curr_block]

        if len(curr_data) < self.block_size:
            # 当前块未满
            if self.ref_counts[curr_block] > 1:
                # 触发写时复制 (CoW)
                new_b = self.alloc_block()
                self.physical_memory[new_b] = list(curr_data)  # 拷贝历史数据
                self.free_block(curr_block)                    # 原块引用减 1
                block_table[-1] = new_b
                curr_block = new_b
            self.physical_memory[curr_block].append(token_val)
        else:
            # 当前块已满，申请全新的独立物理块
            new_b = self.alloc_block()
            block_table.append(new_b)
            self.physical_memory[new_b].append(token_val)

# --- 验证测试 ---
if __name__ == "__main__":
    sim = PagedKVCacheSimulator(block_size=2, total_blocks=5)
    
    # 1. 主序列写入 2 个 Token（刚好填满 1 个物理块）
    seq_a = []
    sim.append_token(seq_a, "Hello")
    sim.append_token(seq_a, "World")
    block_0 = seq_a[0]
    assert sim.ref_counts[block_0] == 1
    assert sim.physical_memory[block_0] == ["Hello", "World"]

    # 2. 分支请求 seq_b 共享 seq_a
    seq_b = sim.fork_sequence(seq_a)
    assert sim.ref_counts[block_0] == 2
    assert seq_b == [block_0]

    # 3. 此时当前块已满，seq_a 追加新 Token 时直接分配新块，不触发原块拷贝
    sim.append_token(seq_a, "!")
    assert len(seq_a) == 2
    assert sim.ref_counts[block_0] == 2  # block_0 仍被共享
    block_1 = seq_a[1]
    assert sim.ref_counts[block_1] == 1

    # 4. 释放 seq_a
    for b in seq_a:
        sim.free_block(b)
    assert sim.ref_counts[block_0] == 1  # seq_b 仍持有 block_0
    assert block_1 in sim.free_blocks    # block_1 彻底回收

    print("PagedKVCache 核心分配与 CoW 验证通过！")
```
