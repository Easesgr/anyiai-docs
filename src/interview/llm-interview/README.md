---
title: 大模型面试题
icon: mdi:robot-outline
tag:
  - 大模型面试
  - 面试题
  - LLM
---

# 大模型面试题

大模型面试高频问题精选，共 47 篇文章，覆盖基础概念、Transformer 架构、训练流程和推理优化四大方向。

每篇文章围绕一个具体问题展开，讲清楚"是什么、为什么、怎么做"，适合准备 AI 岗位面试的候选人。

---

## 合集文章导航

### 基础概念篇

1. [大模型到底是什么？](/interview/llm-interview/what-is-llm.html) - 和传统 NLP 模型的区别
2. [为什么主流大模型采用 Decoder-only 架构？](/interview/llm-interview/decoder-only.html) - 架构选择的原因
3. [Token 是什么？分词算法有什么区别？](/interview/llm-interview/tokenization.html) - BPE、WordPiece、SentencePiece
4. [Token 数量如何计算和控制？](/interview/llm-interview/token-basics.html) - 上下文长度与成本的关系
5. [Embedding 在大模型中起什么作用？](/interview/llm-interview/embedding.html) - Token Embedding 和 Position Embedding
6. [Word2Vec 到 BERT 有怎样的改进？](/interview/llm-interview/word2vec-to-bert.html) - 静态到动态词向量的演进
7. [Word2Vec 如何获取和评估词向量？](/interview/llm-interview/word2vec-evaluation.html) - 相似度任务与类比推理
8. [chatGLM 和 GPT 在结构上有什么区别？](/interview/llm-interview/chatglm-vs-gpt.html) - Prefix Decoder vs Decoder-only
9. [什么是大模型的涌现能力？](/interview/llm-interview/emergent-abilities.html) - 涌现现象的成因
10. [什么是 ViT（Vision Transformer）？](/interview/llm-interview/vit-vision-transformer.html) - Transformer 在视觉的应用
11. [ViLT 模型是如何将 Transformer 应用于图像识别的？](/interview/llm-interview/vilt-vision-transformer.html) - 视觉-语言统一架构
12. [LSTM 是如何解决梯度消失问题的？](/interview/llm-interview/lstm-gradient-vanishing.html) - 与 RNN 的对比

### Transformer 架构篇

13. [Self-Attention 中 Q、K、V 分别是什么意思？](/interview/llm-interview/attention-qkv.html) - 为什么要除以根号 dk
14. [Self-Attention 中 K 和 Q 是用来做什么的？](/interview/llm-interview/self-attention-qk.html) - 查询-键值机制的本质
15. [注意力遮蔽（Attention Masking）的工作原理是什么？](/interview/llm-interview/attention-masking.html) - Causal/Padding Mask
16. [Transformer 在计算 attention 时用点乘还是加法？](/interview/llm-interview/attention-dot-vs-add.html) - 缩放点积注意力的优势
17. [为什么 Transformer 采用多头注意力机制？](/interview/llm-interview/why-multi-head-attention.html) - 多子空间表达
18. [Multi-Head Attention、MQA 和 GQA 有什么区别？](/interview/llm-interview/mha-mqa-gqa.html) - KV Cache 开销优化
19. [位置编码是什么？RoPE 为什么很常用？](/interview/llm-interview/positional-encoding.html) - 旋转位置编码原理
20. [LLaMA 中的旋转位置编码了解吗？](/interview/llm-interview/llama-rope.html) - RoPE 的数学推导
21. [Transformer 的位置编码是怎样的？](/interview/llm-interview/transformer-positional-encoding.html) - 绝对/相对位置编码
22. [Transformer 中 FFN 层有什么作用？](/interview/llm-interview/ffn-swiglu.html) - SwiGLU 激活函数
23. [LayerNorm 和 RMSNorm 有什么区别？](/interview/llm-interview/layernorm-rmsnorm.html) - 为什么大模型用 RMSNorm
24. [Transformer 为什么采用 Layer Normalization 而不是 Batch Norm？](/interview/llm-interview/layer-normalization.html) - 归一化方法对比
25. [Decoder 和 Encoder 阶段的多头自注意力有什么区别？](/interview/llm-interview/decoder-encoder-mha-diff.html) - Cross-Attention 与 Masked Self-Attention
26. [Transformer 如何实现序列到序列的映射？](/interview/llm-interview/transformer-seq2seq.html) - Encoder-Decoder 架构
27. [Transformer 中的残差连接可以缓解梯度消失吗？](/interview/llm-interview/transformer-residual.html) - 残差连接的原理
28. [Transformer 如何处理大型数据集？](/interview/llm-interview/transformer-large-dataset.html) - 分布式训练与梯度累积
29. [Transformer 的性能瓶颈在哪？](/interview/llm-interview/transformer-bottleneck.html) - 计算/显存/访存分析
30. [Transformer 哪个部分最占用显存？](/interview/llm-interview/transformer-memory.html) - KV Cache 显存开销
31. [Transformer 训练完成后如何评估性能？](/interview/llm-interview/transformer-evaluation.html) - 困惑度与下游任务
32. [Transformer 和 LLM 有哪些区别？](/interview/llm-interview/transformer-vs-llm.html) - 架构与模型的关系

### 训练流程篇

33. [大模型预训练目标是什么？](/interview/llm-interview/pretraining-objective.html) - Next Token Prediction
34. [预训练、SFT、偏好对齐分别解决什么问题？](/interview/llm-interview/training-pipeline.html) - 训练流程全景
35. [Transformer 模型训练中的梯度裁剪了解吗？](/interview/llm-interview/gradient-clipping.html) - 防止梯度爆炸
36. [RLHF 和 DPO 有什么区别？](/interview/llm-interview/rlhf-dpo.html) - 偏好对齐方法对比
37. [RLHF 的完整训练流程是怎样的？](/interview/llm-interview/rlhf-pipeline.html) - SFT 到 Reward Model 到 PPO
38. [ORPO 是什么？它如何将指令微调和偏好对齐合二为一？](/interview/llm-interview/orpo-alignment.html) - 一步式对齐
39. [LoRA 和 QLoRA 是什么？](/interview/llm-interview/lora-qlora.html) - 参数高效微调
40. [LoRA 的数学原理是什么？](/interview/llm-interview/lora-math-principle.html) - 低秩分解近似全量微调
41. [LoRA 的超参数应该怎么设置？](/interview/llm-interview/lora-hyperparams.html) - rank、alpha 的经验法则

### 推理优化篇

42. [上下文窗口是什么？](/interview/llm-interview/context-window.html) - 长上下文的局限
43. [KV Cache 是什么？](/interview/llm-interview/kv-cache.html) - 自回归生成加速
44. [Temperature、Top-k、Top-p 分别控制什么？](/interview/llm-interview/sampling-params.html) - 采样参数调优
45. [Temperature 和 Top_p 参数有什么作用？](/interview/llm-interview/sampling-temperature-topp.html) - 如何选择合适的值
46. [大模型幻觉是什么？](/interview/llm-interview/hallucination.html) - 幻觉的成因与缓解
47. [MoE 架构是什么？](/interview/llm-interview/moe.html) - 专家混合结构
