---
title: 训练原理篇
icon: mdi:tune-vertical
---

![](/images/training/training_cover.png)

# 训练原理篇：这个合集能让你学到什么？

大模型是怎么被"训练"出来的？

不是你想象的那种"给它一堆数据，它就学会了"。背后是一套精密的工程：预训练目标怎么定？Loss 怎么算？梯度怎么传？为什么训练一个大模型要几千张 GPU 跑几周？为什么显存总是不够用？

这些问题，是理解大模型的第二道门槛。搞懂训练原理，你才能理解为什么大模型会有幻觉、为什么需要微调、为什么推理比训练快得多。

本篇覆盖大模型训练的完整链路：从预训练目标、自回归语言模型原理，到梯度下降、混合精度训练、分布式并行策略。每篇围绕一个具体问题展开。

---

## 十篇文章，系统覆盖训练全链路

**预训练基础篇（前四篇）**

先搞清楚预训练到底在训练什么。自回归语言模型为什么能预测下一个词？Masked LM 和 Causal LM 有什么区别？为什么训练大模型需要海量数据？

1. [大模型预训练到底在训练什么？](/llm-100/training/what-is-pretraining.html) - 预训练的本质
2. [自回归语言模型为什么能预测下一个词？](/llm-100/training/autoregressive-lm.html) - 自回归原理
3. [Masked LM 和 Causal LM 有什么区别？](/llm-100/training/mlm-vs-clm.html) - 两种语言模型对比
4. [训练大模型为什么需要海量数据？](/llm-100/training/why-massive-data.html) - 数据规模的重要性

**训练核心篇（中间三篇）**

Loss 是怎么计算的？梯度下降和反向传播在大模型里如何工作？为什么大模型训练这么吃显存？这些是训练工程的核心问题。

5. [大模型训练中的 Loss 是怎么计算的？](/llm-100/training/loss-calculation.html) - 损失函数详解
6. [梯度下降、反向传播在大模型里如何工作？](/llm-100/training/gradient-descent-backprop.html) - 优化算法原理
7. [大模型训练为什么这么吃显存？](/llm-100/training/why-high-vram.html) - 显存瓶颈分析

**工程优化篇（最后三篇）**

混合精度训练能省多少显存？数据并行、模型并行、流水线并行分别解决什么问题？ZeRO、DeepSpeed、FSDP 如何让训练更快更省？

8. [混合精度训练 FP16、BF16 有什么区别？](/llm-100/training/mixed-precision.html) - 精度与效率的平衡
9. [数据并行、模型并行、流水线并行分别是什么？](/llm-100/training/parallel-strategies.html) - 三大并行策略
10. [ZeRO、DeepSpeed、FSDP 解决了什么问题？](/llm-100/training/zero-deepspeed-fsdp.html) - 分布式训练优化

---

[← 返回大模型100讲](/llm-100/)
