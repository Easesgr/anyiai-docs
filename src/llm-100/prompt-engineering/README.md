---
title: Prompt 工程篇
icon: mdi:message-text-outline
---

![](/images/prompt-engineering/prompt-engineering_cover.png)

# Prompt 工程篇：这个合集能让你学到什么？

你有没有遇到过这种情况？

明明问的是同一个问题，换个说法，大模型的回答质量天差地别。有时候它给你一个完美的答案，有时候它胡说八道，有时候它直接拒绝回答。

这不是模型的"心情"问题，而是你的 Prompt 写法不同。

Prompt 是你和大模型之间的唯一沟通方式。写得好，模型能发挥出 120% 的能力；写得不好，再强的模型也救不了你。这就是 Prompt 工程的价值——它不是玄学，而是一套可复用的设计方法论。

这个系列共 10 篇文章，从 Prompt 的基本原理出发，覆盖提示策略、结构化输出、参数控制和安全防御。

---

## 十篇文章，带你从入门到实战

**基础原理篇（前三篇）**

先搞清楚 Prompt 为什么能影响大模型输出。然后理解 Zero-shot、Few-shot、Chain-of-Thought 三种提示策略的区别，以及 CoT 思维链为什么有效。

1. [Prompt 为什么能影响大模型输出？](/llm-100/prompt-engineering/Prompt_为什么能影响大模型输出.html) - Prompt 的本质
2. [Zero-shot、Few-shot、Chain-of-Thought 有什么区别？](/llm-100/prompt-engineering/Zero-shot_Few-shot_Chain-of-Thought_有什么区别.html) - 三大提示策略
3. [CoT 思维链提示为什么有效？](/llm-100/prompt-engineering/CoT_思维链提示为什么有效.html) - 思维链的原理

**设计方法篇（中间四篇）**

Prompt 不是随便写几句话就行。结构化 Prompt 怎么写？System、User、Assistant 三种角色分别有什么作用？如何让大模型稳定输出 JSON？Temperature、Top-p、Top-k 分别控制什么？

4. [结构化 Prompt 应该怎么写？](/llm-100/prompt-engineering/结构化_Prompt_应该怎么写.html) - Prompt 设计方法论
5. [System Prompt、User Prompt、Assistant Prompt 分别有什么作用？](/llm-100/prompt-engineering/System_Prompt_User_Prompt_Assistant_Prompt_分别有什么作用.html) - 三种角色的分工
6. [如何让大模型稳定输出 JSON？](/llm-100/prompt-engineering/如何让大模型稳定输出_JSON.html) - 结构化输出技巧
7. [Temperature、Top-p、Top-k 分别控制什么？](/llm-100/prompt-engineering/Temperature_Top-p_Top-k_分别控制什么.html) - 采样参数详解

**问题排查与安全篇（最后三篇）**

为什么同一个 Prompt 每次输出不一样？Prompt 注入攻击是什么？如何防范？如何设计一个高质量 Prompt？这些是工程实践中必须面对的问题。

8. [Prompt 注入攻击是什么？如何防范？](/llm-100/prompt-engineering/Prompt_注入攻击是什么_如何防范.html) - 安全防御
9. [为什么同一个 Prompt 每次输出不一样？](/llm-100/prompt-engineering/为什么同一个_Prompt_每次输出不一样.html) - 输出一致性
10. [面试高频：如何设计一个高质量 Prompt？](/llm-100/prompt-engineering/面试高频_如何设计一个高质量_Prompt.html) - 综合实战

---

[← 返回大模型100讲](/llm-100/)
