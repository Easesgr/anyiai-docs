---
title: 大模型如何逐字生成文本？推理与生成导读
icon: mdi:lightning-bolt-outline
tag:
- 推理与生成
- 推理优化
- 大模型基础
description: 解析大模型自回归生成的完整生命周期，说明 Prefill 与 Decode 两阶段差异，并结合算例剖析算力瓶颈与显存带宽瓶颈对吞吐的影响。
---

![推理与生成导读封面图](/images/content-rebuild/5bd4d49e94e6fb00/cover.png)

在网页或应用中使用大语言模型时，用户输入一段文字，往往几秒内就能看到模型逐字吐出回答。在硬件和底层计算层面，模型并不是一次性把完整段落“写”出来，而是在执行一套严格的**离散自回归迭代流程**。

本导读通过一个具体例子讲清自回归生成的运转机制，梳理 Prefill（预填充）与 Decode（解码）两个关键阶段的物理差异，并解释制约推理速度的算力与显存带宽瓶颈，为本模块的后续专题提供清晰的阅读导航。

---

## 什么是自回归生成：以一个续写例子开始

大语言模型生成文本的基本范式是**自回归模型（Autoregressive Model）**。所谓自回归，是指模型每次只预测“下一个符号”，并将预测出的新符号追加到输入尾部，作为下一次预测的输入，不断循环。

假设用户输入提示词：

> `今天天气真好，我们去`

模型生成回答的完整过程如下：

1. **文本切分为 Token**：分词器（Tokenizer）先将文本切分成离散的单位——Token。Token 不完全等同于汉字或英文单词，例如上面的句子可能被切为 `["今天", "天气", "真好", "，", "我们", "去"]` 6 个 Token，每个 Token 对应词表中的一个数字编号（Token ID）。
2. **预测下一个 Token**：模型接收这 6 个 Token，经过多层神经网络运算，在最后一层输出一个与词表等长的分数向量（Logits）。如果词表大小为 100,000，Logits 就包含 100,000 个未归一化的数值。
3. **转换为概率分布**：通过 Softmax 函数将 Logits 转化为和为 1 的概率分布：
   $$P(x_t \mid x_{<t}) = \text{softmax}(\mathbf{z}_t)$$
   此时词表中的每个词元都有了一个对应的预测概率。
4. **采样挑词**：采样策略根据概率分布挑出一个具体 Token。假设系统选中了“公”。
5. **循环迭代**：系统将“公”拼接到提示词尾部，输入变成 `["今天", "天气", "真好", "，", "我们", "去", "公"]`，再次执行计算，挑出下一个 Token“园”。

这一过程反复进行，直到模型预测出特定的终止符号（如 `<eos>`）或达到开发者设定的最大生成长度上限。生成 $N$ 个 Token，硬件就必须执行 $N$ 轮串行依赖的前向推导。

---

## 推理生命周期：Prefill 与 Decode 两阶段

单次大模型推理请求在底层并不是匀速执行的，而是划分为性质迥异的两个物理阶段：

![自回归推理生命周期的两阶段流转过程](/images/content-rebuild/5bd4d49e94e6fb00/fig-1.png)

### 1. 预填充阶段（Prefill 阶段）

用户输入的提示词包含多个已知的 Token（例如上述例子中的 6 个 Token）。因为这些输入已经完整就绪，模型不需要逐字串行计算，而是利用因果注意力掩码（Causal Mask），让所有输入 Token 在单次计算中**并行前向推导**。

在 Prefill 阶段，模型的主要任务是：
- 计算输入序列中所有 Token 的注意力表征；
- 将各网络层计算得到的键值向量（Key 与 Value）写入显存，构建初始的 **KV 缓存（KV Cache）**；
- 产出首个新 Token 的概率分布，完成首字采样。

Prefill 阶段耗费的时间直接决定了用户感知的**首字延迟（Time to First Token, TTFT）**。

### 2. 解码阶段（Decode 阶段）

从生成第二个 Token 开始，系统进入逐步迭代的 Decode 阶段。此阶段每次只向模型输入前一步刚生成的**单个 Token**，其工作流程是：
- 从显存读取历史上下文已保存的 KV 缓存；
- 计算当前新 Token 与所有历史 Token 的注意力关联；
- 将当前新 Token 的 Key 与 Value 追加进 KV 缓存；
- 输出当前步的预测结果并采样。

Decode 阶段是严格的串行依赖，每一次迭代产生一个 Token。在流式传输（Streaming）中，这单个 Token 会立即通过 Server-Sent Events（SSE）或 Chunked Transfer 协议推送到前端渲染，形成打字机效果。Decode 阶段单步耗时决定了**每输出一个词的间隔时间（Time Per Output Token, TPOT）**。

关于执行管线的代码实现与流式传输协议细节，可参阅 [/fundamentals/inference-and-generation/inference-pipeline.md](/fundamentals/inference-and-generation/inference-pipeline.md) 与 [/fundamentals/inference-and-generation/streaming.md](/fundamentals/inference-and-generation/streaming.md)。

---

## 硬件瓶颈分析：算力受限与显存带宽受限

了解 Prefill 与 Decode 的差异，才能理解大模型推理在硬件部署上的核心矛盾：**算力瓶颈与显存带宽瓶颈在两阶段之间剧烈切换**。

![大模型推理阶段的算力与显存带宽瓶颈对比](/images/content-rebuild/5bd4d49e94e6fb00/fig-2.png)

### Prefill 阶段：算力受限（Compute-bound）
在 Prefill 阶段，输入包含数百或数千个 Token。模型的核心操作是大规模矩阵乘法（GEMM，Matrix-Matrix Multiplication）。每个读入的权重参数可以与批量的输入向量同时进行多次浮点运算，计算强度（每搬运 1 字节数据所执行的浮点运算次数）较高。此时 GPU 的张量核心（Tensor Cores）能满载运转，处理速度主要受限于 GPU 的**浮点运算峰值能力（TFLOPS）**。

### Decode 阶段：显存带宽受限（Memory-bound）
在小批次（尤其是单请求场景，Batch Size = 1）的 Decode 阶段，输入仅有 1 个新 Token，核心运算退化为矩阵-向量乘法（GEMV，Matrix-Vector Multiplication）。

这里存在一个直观但严峻的物理制约：
- **权重搬运量**：以一个 70 亿参数（7B）、采用 16 位半精度浮点数（FP16）存储的模型为例，其权重体积约为 14 GB。
- **计算量**：每生成 1 个 Token，对 7B 参数做一次乘加运算，大致只需要 $2 \times 7\text{B} = 1.4 \times 10^{10}$ 次浮点运算（14 GFLOPs）。现代 GPU 的浮点算力通常可达数百 TFLOPS，完成这 14 GFLOPs 的纯计算只需不到 0.1 毫秒。
- **带宽瓶颈**：为了计算这 1 个 Token，GPU 必须把整整 14 GB 的模型全量权重从高带宽显存（HBM）完整读取到计算核心（SRAM）一次。假设 GPU 显存带宽为 1000 GB/s，搬运 14 GB 权重至少需要：
  $$\frac{14\text{ GB}}{1000\text{ GB/s}} = 14\text{ ms}$$

这意味着，哪怕计算时间忽略不计，光搬运权重就耗时 14 毫秒，单请求的理论生成速度上限被卡在每秒 70 词左右。在此期间，GPU 的计算单元大部分时间都在等待显存送入权重数据。因此，在单请求或小批次场景下，Decode 阶段主要受制于**显存带宽（Memory Bandwidth）**而非算力峰值；当服务端通过批处理聚合较多并发序列、使单次读入的权重可被多次复用时，Decode 阶段的计算强度才会显著提升并向算力受限转变。

### 历史状态的累积：KV 缓存增长
除了模型权重，显存中还要维护每轮生成的 KV 缓存。随着生成序列长度 $L$ 增加，KV 缓存的体积呈线性增长。在长文本场景下，不仅显存容易被吃满，每次 Decode 还要从显存中读取更长的 KV 缓存数据，进一步加重显存带宽负担。有关 KV 缓存的容量推导与长上下文边界，可参阅 [/fundamentals/inference-and-generation/kv-cache.md](/fundamentals/inference-and-generation/kv-cache.md) 与 [/fundamentals/inference-and-generation/context-length.md](/fundamentals/inference-and-generation/context-length.md)。

---

## 本模块学习路径

本模块共包含 8 篇专题文章，从原理运转、资源制约到工程优化，构建完整的认知链条：

![推理与生成模块知识拓扑与阅读路径](/images/content-rebuild/5bd4d49e94e6fb00/fig-3.png)

### 第一层：执行流水线与采样控制（基础运转机制）
- [/fundamentals/inference-and-generation/inference-pipeline.md](/fundamentals/inference-and-generation/inference-pipeline.md)：追踪输入分词、前向计算、Logits 输出到反分词的端到端数据流。
- [/fundamentals/inference-and-generation/temperature-topk-topp.md](/fundamentals/inference-and-generation/temperature-topk-topp.md)：解析 Temperature 如何调节分布平滑度，以及 Top-k 与 Top-p 如何截断低概率尾部候选。
- [/fundamentals/inference-and-generation/streaming.md](/fundamentals/inference-and-generation/streaming.md)：讲解服务端单 Token 推送至客户端的流式协议（SSE/Chunked）与乱码缓冲策略。

### 第二层：状态缓存与序列边界（物理资源约束）
- [/fundamentals/inference-and-generation/kv-cache.md](/fundamentals/inference-and-generation/kv-cache.md)：推导 KV 缓存节省重复计算的原理，以及其显存占用与显存带宽消耗的定量计算。
- [/fundamentals/inference-and-generation/context-length.md](/fundamentals/inference-and-generation/context-length.md)：分析注意力机制在长文本下的计算复杂度、显存压力与位置编码外推限制。

### 第三层：输出约束、系统加速与故障排查（工程生产落地）
- [/fundamentals/inference-and-generation/structured-output.md](/fundamentals/inference-and-generation/structured-output.md)：比较提示词规范、Logit 偏置与语法约束解码（JSON Schema/CFG）保证输出格式的实现原理。
- [/fundamentals/inference-and-generation/inference-acceleration.md](/fundamentals/inference-and-generation/inference-acceleration.md)：梳理权重与激活量化（INT8/INT4）、连续批处理（Continuous Batching）以及 PagedAttention 显存优化方案。
- [/fundamentals/inference-and-generation/generation-pitfalls.md](/fundamentals/inference-and-generation/generation-pitfalls.md)：针对重复死循环、异常截断与幻觉等高频工程问题，提供排查定位清单。
