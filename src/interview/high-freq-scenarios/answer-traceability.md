---
title: 如何让 AI 回答有来源、有依据、可追溯？
icon: mdi:link-variant
tag:
  - 高频场景题
  - 来源追溯
  - 可信AI
---

# 如何让 AI 回答有来源、有依据、可追溯？

![](/images/high-freq-scenarios/answer-traceability_cover.png)

这道题考的是 **RAG 系统的来源追溯能力**。核心是：当 AI 从知识库检索内容作答时，怎么让它的回答"标明出处"，像学术论文一样有参考文献。

实际应用场景很明确——合同审核、医疗咨询、法律文档这些高风险场景，你不能只说"根据相关规定"，得能拿出具体依据。

---

## RAG来源追溯是什么——让AI回答"有据可查"

先搞清楚问题本质。

AI 回答不可信，主要是因为它"一本正经地胡说八道"。RAG 来源追溯就是在回答时明确标注：这条结论来自哪篇文档第几段。

逻辑很简单：

用户提问 → 检索知识库 → 返回相关文档 → **带上文档来源信息** → AI 生成答案时引用 → 输出带标注的最终回答

这样做有两个好处：

- **用户可验证**：答案说的对不对，直接点进去看原文
- **出了问题可追溯**：这个回答基于哪份文档，什么时候入库的，清清楚楚

就像看新闻时文末的参考链接，你点进去能看原始报道，而不是只听记者转述。

![RAG来源追溯流程图，展示用户提问→检索知识库→带来源的答案返回全过程](/images/high-freq-scenarios/20260609_070126_387008_01.png)

---
<ReadMoreLock />

## 三步实现法的技术原理——从入库到输出的完整链路

具体怎么实现？分三步：

### 第一步：入库时 Metadata 写入

文档进知识库时，除了内容本身，还要把**来源信息**写进 Metadata。

用 TikaDocumentReader 读取文档，它会自动把 source（文件名）写入 Metadata。你也可以手动指定：

```java
Map<String, Object> metadata = new HashMap<>();
metadata.put("source", "合同_2024_001.pdf");
metadata.put("入库时间", "2024-06-09");
document.setMetadata(metadata);
```

这一步是基础——来源信息没存进来，后面根本没法追溯。

### 第二步：自定义 MetadataAwareAdvisor 拼接 Prompt

检索到相关文档后，需要把来源信息和原文一起传给大模型。

自定义 Advisor 干的就是这件事：把用户问题、相关文档（**含 Metadata**）、原始原文，拼成一个完整的 Prompt。

```java
public class MetadataAwareAdvisor implements Advisor {
    // 拼接用户问题 + 检索到的文档（含来源信息）+ 原文片段
    // 关键：要把文档的 source、页码等信息传进去
}
```

### 第三步：System Prompt 引导 AI 标注来源

Prompt 里要**明确要求** AI 在回答时标注引用：

```
你是一个合同审核助手。根据提供的文档片段回答问题。
每当你使用某个文档的信息时，必须注明来源。
格式：[来源: 文件名, 页码]
```

AI 收到这个指令，就会养成标注来源的习惯。

---

整个链路是这样的：

**文档入库（Metadata 写入） → 检索时带 Metadata → Advisor 拼接 Prompt → System Prompt 要求标注 → 大模型输出带来源的答案**

![三步实现法流程图，展示TikaDocumentReader→MetadataAwareAdvisor→大模型输出的技术链路](/images/high-freq-scenarios/20260609_070131_490688_01.png)

---

## Advisor优先级机制——Integer.MAX_VALUE-1的秘密

这里有个坑，很多人不注意。

Advisor 在 LangChain4j 里是按优先级执行的。**数字越大越先执行**。

为什么要用 `Integer.MAX_VALUE - 1`？

看执行顺序：

- `ChatMemoryAdvisor`：负责实际调用 LLM，它的优先级是 **Integer.MAX_VALUE**（最高）
- 自定义 MetadataAwareAdvisor：必须在这之前执行，把来源信息塞进 Prompt

如果你也用 `Integer.MAX_VALUE`，会排在 ChatMemoryAdvisor **之后**。这时候 AI 已经拿到对话历史了，你的 Prompt 修改对它没用了——轮子都转完了你才去加水，流不过去。

所以正确的写法：

```java
public class MetadataAwareAdvisor implements Advisor {
    @Override
    public int priority() {
        return Integer.MAX_VALUE - 1;  // 确保在 LLM 调用前执行
    }
}
```

这个细节面试官喜欢问，因为能看出你是不是真的理解执行顺序原理。

![优先级机制对比图，展示MAX_VALUE（错误）和MAX_VALUE-1（正确）的执行顺序差异](/images/high-freq-scenarios/20260609_070129_318336_01.png)

---

## 企业级方案对比——什么时候需要自定义Advisor

你可能会问：有没有现成的方案？

有，但不够用。

| 方案 | 适用场景 | 溯源能力 |
|------|----------|----------|
| 默认 QuestionAnswerAdvisor | 简单问答机器人 | ❌ 不支持文档溯源 |
| 自定义 MetadataAwareAdvisor | 企业级应用、高风险场景 | ✅ 支持来源标注 |

默认 QuestionAnswerAdvisor 能做检索+问答，但它不保留文档来源信息。问答结果从哪来的，你不知道。

对于**合同审核、医疗咨询、法律文档**这类场景，必须能回答"这条结论来自哪份文件第几条"。

自定义 Advisor 是企业级应用的标准选择。代价是需要自己实现，但换来的是完整的溯源能力。

![企业级方案对比表，展示默认方案与自定义方案的适用场景差异](/images/high-freq-scenarios/20260609_070146_607232_01.png)

---

## 面试加分项——面试官想听的高级知识点

上面讲的是基础，能让你过关。下面几个点说出来会让面试官高看你一眼。

### 1. SSE 流式输出的引用处理

很多应用用 SSE 流式返回答案，引用怎么同步？

方案是后端**同时推送两路数据**：

- 主通道：流式输出答案 token
- 附带：引用列表（文档 ID、文件名、片段位置）

前端收到后，在流式渲染答案的同时，把引用列表展示在侧边栏。点击引用高亮对应的原文片段。

```javascript
// 前端示例
const eventSource = new EventSource('/api/chat', {
    onMessage: (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'token') {
            appendToken(data.content);  // 流式渲染答案
        } else if (data.type === 'reference') {
            updateReferenceList(data.list);  // 更新引用列表
        }
    }
});
```

### 2. 文档分块策略影响溯源精度

知识入库时文档怎么切，直接影响溯源效果。

- **按段落切**：溯源精确到段落，但可能切断上下文
- **按语义切**：用 NLP 分析句子关系，切出来更连贯
- **重叠切**：相邻块之间留重复内容，避免关键信息被切断

选哪种取决于业务场景。合同审核可能需要精确到条款，客服问答按语义切更自然。

### 3. TikaDocumentReader 的实现细节

Tika 读取文档时会自动提取 Metadata：

- PDF：作者、创建时间、页数
- Word：标题、段落结构
- 同时自动写入 `source` 字段

了解这个实现细节，说明你对整个流程吃得比较透。

![SSE流式输出架构图，展示后端流式推送+引用列表+前端展示的完整流程](/images/high-freq-scenarios/20260609_070146_133440_01.png)

---

## 面试怎么答

### 基础版（能过的回答）

> 要让 AI 回答有来源可追溯，核心是 RAG 来源追溯的三步实现法：
>
> **第一步**，文档入库时把来源信息写入 Metadata，比如文件名、页码。
>
> **第二步**，自定义 MetadataAwareAdvisor，检索到文档后把原文和 Metadata 一起拼进 Prompt。
>
> **第三步**，System Prompt 里明确要求 AI 回答时标注来源，格式是" [来源: 文件名] "。
>
> 需要注意 Advisor 优先级要设成 Integer.MAX_VALUE-1，确保在 LLM 实际调用前执行，把来源信息注入 Prompt。默认的 QuestionAnswerAdvisor 不支持溯源，企业级应用必须用自定义方案。

### 加分版（让面试官眼前一亮）

> 除了基本流程，还有几个进阶点：
>
> 一是 Advisor 优先级设计原理。ChatMemoryAdvisor 优先级最高负责实际调用 LLM，自定义 Advisor 必须排在它之前才能修改 Prompt，所以用 MAX_VALUE-1。
>
> 二是流式输出时引用列表的处理。后端同时推送 token 流和引用列表，前端流式渲染答案的同时展示引用栏，支持点击跳转原文。
>
> 三是文档分块策略。入库时怎么切文档影响溯源精度——按段落切精确但可能切断语义，按语义切更连贯但实现复杂，实际选型要看业务场景。
>
> 细节上，TikaDocumentReader 会自动把 source 写入 Metadata，这也是实现时的一个便利点。

---

## 一句话总结

RAG 来源追溯的本质是：**入库存 Metadata → Advisor 拼接带来源的 Prompt → System Prompt 要求 AI 标注引用 → Integer.MAX_VALUE-1 确保执行顺序正确**。