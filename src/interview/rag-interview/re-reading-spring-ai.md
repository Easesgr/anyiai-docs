---
title: 什么是 Re-Reading？如何基于 Spring AI 实现 Re-Reading Advisor？
icon: mdi:book-refresh
tag:
- Re-Reading
- Spring AI
- RAG
description: 解释长上下文 RAG 中的 Re-Reading 策略，剖析位置偏置与注意力漂移，并给出基于 Spring AI Advisor 的无侵入工程实现与成本测算。
---

## 1. 面试口述答案

在面试中被问到“什么是 Re-Reading，以及如何在 Spring AI 中落地”时，可以直接按以下三层逻辑口述：

> **核心定义**：
> Re-Reading（重读增强）是一种通过调整 Prompt 结构来改善大模型长上下文理解的提示工程技巧。在检索增强生成（RAG）场景中，它采用 **$Q + C + Q'$（用户提问 + 检索文档 + 核心问题与约束复述）** 的三段式结构，将核心问题在长参考资料的尾部再次强调一遍。
>
> **产生原因与作用**：
> Transformer 架构在处理长文本时存在可验证的位置偏置（即“迷失在中间”，Lost in the Middle 现象）。当检索召回的参考文档长达数千 Token，且夹杂大量业务细节或干扰信息时，模型在末尾生成回答时容易受到紧邻上下文的惯性影响，忽略序列最前端的原始提问约束。在文档尾部追加简要的提问复述 $Q'$，能够为解码生成提供强局部近因激活，显著降低多文档推理时的意图漂移和幻觉。
>
> **Spring AI 工程落地**：
> 在 Spring AI 中，推荐通过 `CallAroundAdvisor` 拦截链解耦实现。实现的难点在于**原始提问的状态隔离**：框架自带的 `QuestionAnswerAdvisor` 会在检索完成后，用模板将文档和问题直接重写进 `AdvisedRequest` 的 `userText`。因此，必须通过优先级更高的前置切面将干净的原始提问存入 `adviseContext` 容器中；后置的 `ReReadingAdvisor` 再从上下文取出原始提问，把精简的 $Q'$ 拼接到末尾。这种做法既保证了业务无侵入，又避免了误把数千字上下文重复复述导致 Token 翻倍。

![Re-Reading 机制概览与 Spring AI Advisor 架构](/images/content-rebuild/cd6f50fb6916fa9f/cover.png)

---

## 2. 问题来源：长上下文下的注意力分散与位置偏置

在标准的 [RAG 的完整工作流程](/interview/rag-interview/rag-complete-workflow.md) 中，输入大模型的典型结构为：先给出用户提问 $Q$，接着拼接向量检索召回的参考文档 $C$（通常 3 到 8 个片段，长度约 1500 至 4000 Token），最后等待大模型输出回答 $A$。

我们以一个具体场景贯穿说明：
- **用户提问 $Q$**：“研发二部 2024 年第三季度差旅报销中，高铁二等座超标的审批流是什么？”
- **检索文档 $C$**：包含《研发二部组织架构说明》、长达 2000 字的《全集团差旅通用细则》（包含行政部、市场部、各子公司的各类特批条款），最后夹着一小段《研发二部专用细则修订版》。

在自回归语言模型中，这种排布在长上下文场景下容易出现两类问题：

1. **首尾位置偏置（Lost in the Middle）**：
   实证研究表明，大语言模型对序列输入的首部（Primacy）和尾部（Recency）信息通常保持较高的注意力敏感度，而对处于序列中段的长篇文本，模型提炼关键事实的召回率会有所下降。
2. **解码起点的近因干扰**：
   自回归解码是从输入 Prompt 的最后一个 Token 开始逐字向后预测的。在 $Q + C$ 结构中，生成回答的起始位置紧邻在长文档 $C$ 的末尾，距离序列开头的 $Q$ 隔了数千 Token。此时，生成起点被参考文档末尾的内容紧密包裹。如果末尾段落恰好提到了“市场部超标由事业群总监审批”，模型极易顺着最后几句话的上下文惯性作答，遗漏了开篇所限定的“研发二部”与“2024 年 Q3”双重约束。

![长上下文下中间文档的 Key 竞争与尾部重锚定机制](/images/content-rebuild/cd6f50fb6916fa9f/fig-1.png)

Re-Reading 的目的，就是在模型准备落笔生成答案的位置，把最关键的问题和约束再次提醒一遍，拉回模型的注意力。

---

## 3. Prompt 结构对比与设计边界

在 RAG 实践中，提示词排布通常有以下三种结构：

| 结构类型 | 排布顺序 | 预填充阶段表现 | 解码生成阶段表现 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **单向先行 (Q + C)** | `问题 -> 文档 -> 生成` | 带着问题通读文档，目标明确 | 生成起点离原始问题太远，复杂约束易被中后段噪声稀释 | 上下文较短（<500 Token）的简单单文档检索 |
| **尾置单充 (C + Q)** | `文档 -> 问题 -> 生成` | 先读大段无目标文档，提取关键信息缺乏先验引导 | 生成起点紧邻问题，指令跟随度高 | 短文本抽取、明确单任务字段结构化输出 |
| **双向重读 (Q + C + Q')** | `问题 -> 文档 -> 尾部重读 -> 生成` | 开头明确目标引导文档理解 | 尾部二次提示拉回关键约束，抵抗长程噪声漂移 | 长上下文跨文档推理、带强条件约束的复杂 RAG |

<ReadMoreLock />

![单向先行、尾置单充与双向重读 Prompt 拓扑对比](/images/content-rebuild/cd6f50fb6916fa9f/fig-2.png)

### 尾部 $Q'$ 的编写边界

尾部的 $Q'$ **绝对不能把整段提问和背景又完整复制一遍**，更不能混入文档内容。$Q'$ 应当只包含：
1. 核心实体和目标动作；
2. 负向约束（例如“若资料中未提及请明确说明，不要推测”）；
3. 期望的输出形式。

例如贯穿例子中的重读设计：

```text
【系统指令】你是一个严谨的企业知识库助理...
【用户提问 Q】研发二部 2024 年第三季度差旅报销中，高铁二等座超标的审批流是什么？
【参考资料 C】
[文档 1] 研发二部组织架构说明...
[文档 2] 2024 年全集团通用差旅制度...（长篇通用条款）
[文档 3] 研发二部专用细则修订版：高铁二等座超标需直属总监审批并抄送 HRBP...
【尾部重读 Q'】
请结合上述资料作答：研发二部 2024 年 Q3 高铁二等座超标的具体审批流及抄送对象？若资料无提及请直接说明。
```

---

## 4. 基于 Spring AI Advisor 实现 Re-Reading 拦截链

在 [Spring AI 模块化架构](/interview/rag-interview/spring-ai-modular-rag.md) 中，数据处理流水线被抽象为拦截器链（Advisor Chain）。如果要在框架中落地 Re-Reading，必须保证原有上下文不受污染。

### 4.1 执行顺序与状态隔离陷阱

Spring AI 内置的检索增强切面是 `QuestionAnswerAdvisor`。它的默认逻辑是：调用向量检索拿到文档后，通过内置模板把“参考文档”和“用户问题”拼接在一起，然后整体写回 `AdvisedRequest.userText()` 中。

这意味着，如果在 `QuestionAnswerAdvisor` 之后执行自定义 Advisor，此时请求里的 `userText` **已经变成了包含几千 Token 文档的长字符串**。如果此时直接读取 `advisedRequest.userText()` 并套入重读模板，请求就会变成：
$$(Q + C) + \text{重读模板} + (Q + C)$$
这不仅会导致数千字的参考资料在 Prompt 里重复出现两次，使得 Token 消耗翻倍，甚至会直接挤爆模型上下文窗口。

因此，必须采用**状态隔离传递**方案：
1. **前置捕获（QueryCaptureAdvisor，Order = -10）**：在文档检索前执行，截获纯净的原始提问 $Q$，存入不可变的请求扩展容器 `adviseContext` 中。
2. **检索注入（QuestionAnswerAdvisor，Order = 0）**：正常执行向量检索，将文档和提问组装后写入 `userText`。
3. **尾部重读（ReReadingAdvisor，Order = 10）**：在检索完成后执行，从 `adviseContext` 取出之前暂存的纯净提问，组装成 $Q'$ 追加到当前的 `userText` 末尾。

![Spring AI Advisor 拦截链执行顺序与请求流转拓扑](/images/content-rebuild/cd6f50fb6916fa9f/fig-3.png)

### 4.2 可运行代码实现

首先编写前置提问捕获切面：

```java
package com.anyi.ai.advisor;

import org.springframework.ai.chat.client.advisor.api.AdvisedRequest;
import org.springframework.ai.chat.client.advisor.api.AdvisedResponse;
import org.springframework.ai.chat.client.advisor.api.CallAroundAdvisor;
import org.springframework.ai.chat.client.advisor.api.CallAroundAdvisorChain;

import java.util.HashMap;
import java.util.Map;

public class QueryCaptureAdvisor implements CallAroundAdvisor {

    public static final String ORIGINAL_QUERY_KEY = "RE_READING_ORIGINAL_QUERY";

    @Override
    public AdvisedResponse aroundCall(AdvisedRequest advisedRequest, CallAroundAdvisorChain chain) {
        // 复制一份现有的上下文字典，防止直接修改原引用
        Map<String, Object> context = new HashMap<>(advisedRequest.adviseContext());
        // 在检索发生前，保存原始纯净提问
        context.put(ORIGINAL_QUERY_KEY, advisedRequest.userText());

        // Spring AI 的 AdvisedRequest 为不可变对象，需使用 from 构建新实例
        AdvisedRequest mutated = AdvisedRequest.from(advisedRequest)
                .withAdviseContext(context)
                .build();
        return chain.nextAroundCall(mutated);
    }

    @Override
    public int getOrder() {
        // 设置为 -10，确保早于 QuestionAnswerAdvisor (通常为 0) 执行
        return -10;
    }

    @Override
    public String getName() {
        return this.getClass().getSimpleName();
    }
}
```

接着编写尾部重读追加切面：

```java
package com.anyi.ai.advisor;

import org.springframework.ai.chat.client.advisor.api.AdvisedRequest;
import org.springframework.ai.chat.client.advisor.api.AdvisedResponse;
import org.springframework.ai.chat.client.advisor.api.CallAroundAdvisor;
import org.springframework.ai.chat.client.advisor.api.CallAroundAdvisorChain;

public class ReReadingAdvisor implements CallAroundAdvisor {

    private static final String RE_READ_TEMPLATE = 
        "\n\n【请再次核对核心问题并作答】\n请严格基于上方提供的参考资料回答以下问题，忽略无关干扰：\n%s";

    private final int order;

    public ReReadingAdvisor(int order) {
        this.order = order;
    }

    public ReReadingAdvisor() {
        // 默认 Order 设为 10，确保在检索注入完成后执行
        this(10);
    }

    @Override
    public AdvisedResponse aroundCall(AdvisedRequest advisedRequest, CallAroundAdvisorChain chain) {
        // 1. 从 adviseContext 中安全提取检索前暂存的原始提问
        String originalQuery = (String) advisedRequest.adviseContext().get(QueryCaptureAdvisor.ORIGINAL_QUERY_KEY);
        
        // 降级防御：若未提取到前置保存的原始提问（如漏配前置切面），
        // 绝不能直接回退至已注入长文档的 userText，应直接放行原请求以防 Token 翻倍
        if (originalQuery == null || originalQuery.isBlank()) {
            return chain.nextAroundCall(advisedRequest);
        }

        String currentText = advisedRequest.userText();
        if (currentText != null && !currentText.isBlank()) {
            // 2. 仅将原始问题套入精炼模板，绝对不能把 currentText 再次套入
            String reReadSuffix = String.format(RE_READ_TEMPLATE, originalQuery.trim());
            String finalUserText = currentText + reReadSuffix;

            AdvisedRequest mutatedRequest = AdvisedRequest.from(advisedRequest)
                    .withUserText(finalUserText)
                    .build();

            return chain.nextAroundCall(mutatedRequest);
        }

        return chain.nextAroundCall(advisedRequest);
    }

    @Override
    public int getOrder() {
        return this.order;
    }

    @Override
    public String getName() {
        return this.getClass().getSimpleName();
    }
}
```

![基于 adviseContext 传递原始 Query 与请求不可变重塑机制](/images/content-rebuild/cd6f50fb6916fa9f/fig-4.png)

### 4.3 客户端组装与流式支持

在 Spring 配置类中按顺序装配切面：

```java
@Bean
public ChatClient ragChatClient(ChatClient.Builder builder, VectorStore vectorStore) {
    return builder
        .defaultAdvisors(
            // 1. 拦截并暂存原始 Query (Order: -10)
            new QueryCaptureAdvisor(),
            // 2. 检索并拼接参考文档 (默认 Order: 0)
            new QuestionAnswerAdvisor(vectorStore),
            // 3. 提取原始提问并在尾部追加重读 (Order: 10)
            new ReReadingAdvisor(10)
        )
        .build();
}
```

若业务涉及流式输出（Server-Sent Events），上述类应同时实现 `StreamAroundAdvisor` 接口，并在重写的 `aroundStream` 方法中复用完全相同的请求改写逻辑即可。

---

## 5. 成本核算与实验验收

### 5.1 Token 增量核算

Re-Reading 的增量成本取决于它只重读了“问题”还是误重读了“上下文”。

以常见的长上下文问答为例进行核算：
- 系统指令（System Instruction）：约 150 Token
- 检索上下文 $C$（Top-5 文档块，每块约 400 Token）：约 2000 Token
- 原始提问 $Q$：约 30 Token
- **未加重读的常规输入** = $150 + 2000 + 30 = 2180$ Token

开启 Re-Reading 后的开销增量：
- 尾部引导模板 + $Q'$：约 $25 + 30 = 55$ Token
- **Re-Reading 实际输入** = $2180 + 55 = 2235$ Token
- **相对增量**：$\frac{55}{2180} \approx 2.52\%$

在合理实现的前提下，输入 Token 增长通常在 2%~5% 左右，成本和延迟增加极为有限。只有在代码实现错误、误把几千字的文档上下文再次复读时，才会产生翻倍的异常开销。

![Re-Reading 在 RAG 抗干扰测试中的 Token 占比与效果验收](/images/content-rebuild/cd6f50fb6916fa9f/fig-5.png)

### 5.2 效果验收与离线评测

在 [RAG 系统评估](/interview/rag-interview/rag-evaluation.md) 体系中，可通过以下两组对照实验验证 Re-Reading 的收益：

1. **干扰文档鲁棒性测试**：在召回的上下文中间段，人为混入与提问主题相似、但属于其他业务部门或过往年份的干扰切片。观察模型在单向先行（$Q + C$）与双向重读（$Q + C + Q'$）两种模式下的答案忠实度（Faithfulness）。实测表明，尾部带有明确主体限定的重读能显著降低被中段相似干扰带偏的概率。
2. **负向样本拒答率**：向系统输入知识库未收录的问题。在重读模板中包含“资料未提及请直接说明”的强约束，验证模型是如实说明未提及，还是顺着末尾文档内容强行编造。

---

## 6. 面试追问与常见误区

### 追问 1：既然成本只增加 2% 左右，是不是所有问答场景都应该默认加上 Re-Reading？

**回答要点**：
不是。应结合上下文长度和任务类型权衡：
- **短上下文无显著收益**：如果单轮对话输入不到 200 Token（例如单句翻译、短文本润色），模型的全序列注意力能够完全覆盖，此时加重读不仅没有收益，还会徒增几十个 Token 的首字生成延迟。
- **未做指代消解的多轮对话需谨慎**：如果用户上一句问“研发二部”，本句问“那他们报销高铁二等座怎么批”，如果直接在尾部复述未经改写（[Query Rewrite](/interview/rag-interview/query-rewrite.md)）的短句“那他们报销高铁二等座怎么批”，代词“他们”依然模糊。多轮对话中应优先重读经过改写的独立语义问题。

### 追问 2：如果 Advisor 的 Order 顺序配错了，会导致什么故障？

**回答要点**：
在 Spring 的拦截链设计中，`Order` 数值越小越先执行：
- 如果错误地把 `ReReadingAdvisor` 的 Order 设为 `-5`（排在 `QuestionAnswerAdvisor` 的 `0` 之前），重读逻辑会在文档检索之前触发。
- 此时由于文档尚未注入，请求先被改造成了 $Q + Q'$；接着请求进入 `QuestionAnswerAdvisor`，该切面执行检索并把几千字参考文档追加在末尾，最终组装出的 Prompt 结构畸变为：$$(Q + Q') + C$$
- 此时模型的解码起始点紧贴在长文档 $C$ 之后，尾部重读的近因强化作用彻底落空。

![Advisor 执行顺序错误引发的拓扑畸变故障树](/images/content-rebuild/cd6f50fb6916fa9f/fig-6.png)

### 追问 3：Re-Reading 可以替代 Rerank（重排）或混合检索吗？

**回答要点**：
完全不能。必须严格区分**注意力分配问题**与**召回缺陷问题**：
- Re-Reading 解决的是**读到了但没用好**：即知识库已经成功召回了关键文档，但由于文档太长或噪声太多，模型在生成时注意力漂移，导致产生[RAG 幻觉](/interview/rag-interview/rag-hallucination.md)。
- 如果检索阶段本身发生了漏召回（Recall Failure），返回的参考切片里根本没有答案所需的事实依据，无论在末尾重读多少遍提问，模型也无法无中生有。必须继续依赖 [BM25 + 向量混合检索](/interview/rag-interview/hybrid-retrieval.md) 与 [Rerank 精排](/interview/rag-interview/rerank.md) 来保障底层的检索召回率。
