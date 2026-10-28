---
title: 微调与对齐篇
icon: mdi:tune-vertical
---

![](/images/finetuning/finetuning_cover.png)

# 微调与对齐篇：这个合集能让你学到什么？

预训练出来的大模型，就像一个读过很多书但不知道怎么跟人交流的学者。它知道很多知识，但不会按指令回答问题，不会拒绝不当请求，也不会给出有帮助的格式。

微调和对齐，就是把这位"学者"变成"助手"的过程。

SFT 教它按指令回答，RLHF 教它人类偏好，DPO 简化了对齐流程，LoRA 让微调成本降到可接受的范围。这些技术组合在一起，才有了今天 ChatGPT、Claude 这些好用的大模型。

本篇共 10 篇文章，从预训练 vs 微调的区别出发，覆盖 SFT、RLHF、DPO、LoRA、QLoRA 等核心技术。

---

## 十篇文章，系统覆盖微调全链路

**基础概念篇（前三篇）**

先搞清楚预训练、微调、指令微调分别是什么，它们之间有什么关系。然后深入 SFT 指令微调的具体流程，理解 RLHF 为什么是大模型对齐的关键。

1. [预训练、微调、指令微调分别是什么？](/llm-100/finetuning/pretraining-vs-finetuning.html) - 三个阶段的区别
2. [SFT 指令微调到底怎么做？](/llm-100/finetuning/sft-instruction-tuning.html) - SFT 流程详解
3. [为什么大模型需要 RLHF？](/llm-100/finetuning/why-rlhf.html) - 人类反馈的价值

**对齐方法篇（中间四篇）**

DPO 和 RLHF 有什么区别？LoRA 为什么能高效微调大模型？QLoRA 如何进一步降低成本？Adapter、Prefix Tuning、P-Tuning 三种方法有什么区别？

4. [DPO 是什么？它和 RLHF 有什么区别？](/llm-100/finetuning/dpo-vs-rlhf.html) - 两种对齐方法对比
5. [LoRA 为什么能高效微调大模型？](/llm-100/finetuning/lora-efficient-finetuning.html) - 低秩适配原理
6. [QLoRA 如何进一步降低微调成本？](/llm-100/finetuning/qlora-cost-reduction.html) - 量化+LoRA
7. [Adapter、Prefix Tuning、P-Tuning 有什么区别？](/llm-100/finetuning/adapter-prefix-ptuning.html) - 三种高效微调方法

**工程实践篇（最后三篇）**

微调数据应该怎么构造？为什么微调会过拟合？如何评估一个微调模型是否真的变好了？这些是微调工程化必须面对的问题。

8. [微调数据应该怎么构造？](/llm-100/finetuning/finetuning-data-construction.html) - 数据构造方法
9. [大模型微调为什么会过拟合？](/llm-100/finetuning/finetuning-overfitting.html) - 过拟合原因与对策
10. [如何评估一个微调模型是否真的变好了？](/llm-100/finetuning/evaluate-finetuned-model.html) - 评估方法论

---

[← 返回大模型100讲](/llm-100/)
