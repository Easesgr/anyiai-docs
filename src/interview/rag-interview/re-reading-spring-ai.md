---
title: 什么是 Re-Reading？如何基于 Spring AI 实现 Re-Reading Advisor？
icon: mdi:book-refresh
tag:
  - Re-Reading
  - Spring AI
  - RAG
---

# 什么是 Re-Reading？如何基于 Spring AI 实现 Re-Reading Advisor？

这道题考的是推理增强技术（Re-Reading）的原理理解，以及 Spring AI 中 Advisor 拦截机制的掌握。面试官想看你能不能把"如何给 AI 装上一个复读插件"这件事讲清楚。

我分四个板块来讲：Re-Reading 是什么、Spring AI Advisor 是什么、如何用 Advisor 实现 Re-Reading、以及代价与适用场景。

---
## Re-Reading 是什么？让 AI "回头看" 的技巧

Re-Reading 翻译过来就是"重读"。但这里说的不是让 AI 读两遍同样的内容，而是让它**重新审视自己收到的问题**。

具体怎么操作？

把原始问题 Q，改成 Q + "Please answer the above question."，再提交给模型一次。

就这么简单。

举个例子。假设用户问：

> "牛顿第二定律的公式是什么？"

用 Re-Reading 的话，实际发给模型的请求会变成：

> "牛顿第二定律的公式是什么？Please answer the above question."

模型会怎么反应？

它会先看到问题，产生第一轮理解；然后看到"请回答上面的问题"这句话，触发第二轮审视。这个二次审视过程能帮助模型纠正一些第一次理解的偏差。

**原理其实很朴素**：就像你考试做完卷子，回头检查时会突然发现"咦这题我好像理解错了"。

模型也是类似的。第一次看到问题，它的注意力被问题的前半部分牵引；被要求"回答上面那个问题"时，它会重新扫一遍整个问题，从整体角度再审视一遍。

这个技巧有论文背书，叫 **Re2**（Reasoning with Reasoning）。实验证明，在数学推理、逻辑推理等任务上，Re-Reading 能显著提升准确率。

---
<ReadMoreLock />

## Spring AI Advisor 是什么？拦截器的设计哲学

说完了 Re-Reading，再来说 Spring AI 里的 Advisor。

**Advisor 就是 Spring AI 的拦截器**，或者说，它是 AOP（面向切面编程）思想在 AI 场景的落地。

Spring AI 提供两个核心接口：

- **CallAroundAdvisor**：拦截普通请求
- **StreamAroundAdvisor**：拦截流式请求

实现这两个接口，你可以在请求发出前、收到后做各种增强处理。

关键点：**getOrder() 方法控制执行顺序**。

Spring AI 会把所有 Advisor 排成一条链，按 order 值从小到大依次执行。**链末位的 Advisor 负责真正发送请求给 LLM**。

你可以把 Advisor 链想象成一条快递分拣流水线：

- 每个 Advisor 是一个质检站点
- 消息从第一个站点出发，依次经过每个站点
- 每个站点可以对消息做检查、修改、记录
- 最后一个站点把包裹发出去

这样设计的好处是**职责分离、灵活插拔**。你想给 ChatClient 加什么功能，不用改核心代码，装一个 Advisor 就行。

Spring AI 内置了很多 Advisor，比如：

- **MessageChatMemoryAdvisor**：管理对话记忆
- **QuestionAnswerAdvisor**：做问答增强
- **SafetyAdvisor**：做内容安全检查

你需要什么功能，就加什么 Advisor。

---

## 如何用 Advisor 实现 Re-Reading？三步搞定

现在把两个知识点串起来：用 Spring AI 的 Advisor 机制实现 Re-Reading 功能。

整个过程分三步：

**第一步：实现 CallAroundAdvisor 接口**

你需要写一个类，实现 CallAroundAdvisor 接口。在 aroundDo 方法里，把用户输入改写一下，加上 "Please answer the above question."，然后继续执行链条。

java
public class ReReadingAdvisor implements CallAroundAdvisor {
    
    @Override
    public ClientResponse aroundDo(
            AdvisorChain chain, 
            ClientRequest request) {
        
        // 取出原始消息
        List userMessage = request.getMessages();
        
        // 改写：追加 Re-Reading 后缀
        String originalText = extractText(userMessage);
        String reReadText = originalText 
            + "\nPlease answer the above question.";
        
        // 重新包装消息
        userMessage = rewriteMessage(userMessage, reReadText);
        
        // 继续执行 Advisor 链
        return chain.nextClientRequest(
            ClientRequest.from(request).withMessages(userMessage).build()
        );
    }
    
    @Override
    public int getOrder() {
        return 0; // 可以调整顺序
    }
}


**第二步：通过 .withAdvisor() 集成到 ChatClient**

这一步简单到不行。原来的代码怎么调 ChatClient，现在就在后面加一行 .withAdvisor(new ReReadingAdvisor())。

java
ChatClient chatClient = ChatClient.create(chatModel);

String result = chatClient.prompt()
    .user("牛顿第二定律的公式是什么？")
    .withAdvisor(new ReReadingAdvisor())  // 装上复读插件
    .call()
    .content();


**第三步：请求自动被 Re2 处理，业务代码零改动**

所有经过这个 ChatClient 的请求，都会自动过一遍 Re-Reading 逻辑。你的业务代码完全不用改，只需要在初始化的地方配上 Advisor。

这就是 Spring AI Advisor 的精髓：**业务逻辑和增强逻辑解耦，想加就加，想删就删**。

---

## Re-Reading 的代价与适用场景

Re-Reading 效果好，但不是免费的。

**输入长度翻倍**。

因为你要把原问题完整地重复一遍再发一次，所以每次请求的 token 消耗大约翻倍。对应到 API 成本，也就是翻倍。

这意味着什么？

在 C 端面向用户的场景，如果调用量很大，成本会很明显。这时候要权衡：**我愿意多花一倍的钱买这个准确率提升吗？**

Re-Reading 适合的场景：

- **推理质量优先**，对成本不那么敏感
- 数学、代码、逻辑分析等需要高准确率的任务
- 企业内部辅助决策类应用

不太适合的场景：

- 高频调用、成本敏感的 C 端产品
- 简单问答类任务（不需要二次推理）

另外，不同模型对 Re-Reading 的效果可能不同。推理能力强的模型，二次审视带来的提升更明显。

---

## 面试怎么答

**基础版（能过）：**

Re-Reading 是一种推理增强技巧，原理是把用户的问题后面加上"Please answer the above question."重新提交给模型，让模型二次审视问题，纠正第一次理解的偏差。Spring AI 的 Advisor 机制是基于 AOP 思想设计的拦截器，通过实现 CallAroundAdvisor 或 StreamAroundAdvisor 接口，可以拦截请求做增强处理。使用时只需要在 ChatClient 的链式调用里加一个 .withAdvisor(new ReReadingAdvisor())，就能让所有请求自动走一遍 Re-Reading 逻辑。不过需要注意，Re-Reading 会让输入 token 翻倍，成本也会翻倍，要根据实际场景权衡使用。

**加分版（眼前一亮）：**

Re-Reading 有论文叫 Re2，核心原理是利用模型对问题的二次理解来提升推理准确率。在 Spring AI 里，Advisor 不仅仅是拦截器，它是一套完整的 AOP 体系——通过 getOrder() 控制执行顺序，多个 Advisor 组成链，末位负责真正发送 LLM 请求。Spring AI 内置了很多实用的 Advisor，比如 MessageChatMemoryAdvisor 管对话记忆、QuestionAnswerAdvisor 做问答增强。我之前实现 Re-Reading Advisor 时，同时实现了 CallAroundAdvisor 和 StreamAroundAdvisor，这样不管是普通调用还是流式调用都能支持。实际经验是，Advisor 要保持职责单一，不要在里面做耗时操作，否则会拖慢整个请求链路。

---

## 一句话总结

Re-Reading 就是让 AI "回头看" 问题提升准确率，Spring AI 通过 Advisor 机制可以零侵入地给 ChatClient 加上这个复读插件，但要注意 token 翻倍带来的成本问题。
