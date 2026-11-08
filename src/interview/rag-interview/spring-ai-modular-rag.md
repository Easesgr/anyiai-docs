---
title: Spring AI 的模块化 RAG 架构是如何实现的？预检索、检索和后检索各自负责什么？
icon: mdi:puzzle-outline
tag:
- Spring AI
- Modular RAG
- RAG
description: 梳理 Spring AI 基于 Advisor 责任链实现模块化 RAG 的机制，详解预检索改写、向量召回与后检索重排三阶段分工与企业级避坑点。
---

在企业知识库系统开发中，最原始的做法是在 Controller 或 Service 中硬编码：先调向量数据库查出几条文档，再拼成一段提示词丢给大模型。这种写法一旦遇到查询改写、多租户权限过滤、重排序（Rerank）、脱敏拦截或流式打字机输出，代码就会迅速退化为不可维护的意大利面条逻辑。

[什么是 Modular RAG？](/interview/rag-interview/modular-rag.md) 从架构理念上指出了将检索增强生成解耦为独立模块的演进方向。而在 Java 与 Spring 生态中，Spring AI 并没有另起炉灶发明晦涩的图计算模型，而是借鉴了成熟的 **AOP 环绕切面与责任链模式**，通过 **Advisor 拦截器链路** 实现了模块化 RAG 的标准分层编排。

![Spring AI 模块化 RAG 架构封面图](/images/content-rebuild/f09c7890b7175dfe/cover.png)

---

## 1. 核心口述答案：Spring AI 模块化 RAG 的架构设计与阶段分工

面试中被问到“Spring AI 是如何实现模块化 RAG 的？各个阶段分别负责什么？”，建议在 90 秒内按如下结构清晰作答：

> **核心口述回答**：
> 
> Spring AI 并没有把检索拼装硬编码进大模型接口，而是基于 **Advisor（顾问拦截器）责任链** 落地了模块化 RAG 架构。
> 
> 1. **架构模型**：将 RAG 的完整生命周期抽象为切面链，所有增强逻辑通过实现 `CallAroundAdvisor`（同步单次调用）或 `StreamAroundAdvisor`（流式输出）接口挂载到 `ChatClient` 上。拦截器之间通过共享的 `AdvisedRequest` 上下文容器 `adviseContext` 传递中间计算结果，并通过 Spring 的 `Ordered` 接口严格控制执行优先级。
> 2. **三阶段核心分工**：
>    - **预检索阶段（Pre-Retrieval）**：在访问向量库前执行。负责合规与敏感词过滤（命中则短路拦截，不浪费后续算力）、多轮对话代词消解与 [Query Rewrite（查询改写）](/interview/rag-interview/query-rewrite.md)，把改写后的搜索词存入上下文；
>    - **检索中阶段（Retrieval）**：核心内置组件是 `QuestionAnswerAdvisor`。它读取改写后的搜索词，调用 [向量数据库](/interview/rag-interview/vector-database.md)（`VectorStore`）执行带有租户权限和相似度阈值的召回，把原始文档集合存入上下文；
>    - **后检索阶段（Post-Retrieval）**：在文档召回后、大模型推理前执行。负责调用精排模型完成 [Rerank 重排序](/interview/rag-interview/rag-rerank.md)、关键信息位置优化、上下文截断与 PII 敏感信息脱敏，把提炼后的内容拼接进最终提示词；
> 3. **生态集成优势**：天然复用 Spring 框架原生的依赖注入、Micrometer 全链路指标追踪（记录每段 Advisor 耗时）与基于线程池隔离的容灾降级机制。

![Advisor 拦截器链路整体架构](/images/content-rebuild/f09c7890b7175dfe/fig-1.png)

---

## 2. 贯穿示例：企业人事助手的一次问答生命周期

为了直观理解三阶段的协同关系，我们设定一个具体的贯穿业务场景：

- **用户身份**：研发一部员工（租户 ID：`dept_rd_01`）；
- **多轮历史**：上一轮用户询问了关于季度绩效评级标准；
- **当前提问**：“那如果我对它有异议，应该联系谁复议？发邮件还是打电话？”

如果直接拿这句话去向量库做嵌入检索，由于存在代词“它”且缺失业务主体，向量检索大概率会检索到公文投诉或通用运维流程，造成严重漏召回。在 Spring AI 的模块化链路中，整个流转过程如下：

1. **预检索拦截器**：识别上下文，将原始提问改写为 `“研发一部 季度绩效评级复议流程 申诉联系人及沟通方式”`，并写入上下文键值对 `adviseContext["REWRITTEN_QUERY"]`；
2. **检索拦截器**：`QuestionAnswerAdvisor` 从上下文读取改写词，带上元数据过滤条件 `department == 'dept_rd_01'`，向向量库请求相似度最高的 15 个分块，将这 15 个原始分块存入 `adviseContext["qa_retrieved_documents"]`；
3. **后检索拦截器**：读取召回的 15 个分块，调用本地 Cross-Encoder 重排模型选出关联度最高的 3 个分块，并用脱敏规则将分块中包含的人事专员身份证号打码，最后组装成标准模板注入请求参数，递交底层的 `ChatModel` 开展自回归生成。

---

## 3. Advisor 拦截链路的工作机制

在 Spring AI 中，`Advisor` 本质上是一个双向环绕切面（Around Advice）。无论是普通非流式调用还是流式生成，都基于特定的接口规范流转。

### 3.1 同步与流式接口定义

- **`CallAroundAdvisor`**：拦截单次阻塞式调用。方法签名为：
  ```java
  AdvisedResponse aroundCall(AdvisedRequest advisedRequest, CallAroundAdvisorChain chain);
  ```
  入参是包含提示词、模型参数和上下文的 `AdvisedRequest`，返回值是包含模型生成内容与响应上下文的 `AdvisedResponse`。

- **`StreamAroundAdvisor`**：拦截流式调用。方法签名为：
  ```java
  Flux<AdvisedResponse> aroundStream(AdvisedRequest advisedRequest, StreamAroundAdvisorChain chain);
  ```
  返回 Project Reactor 的响应式流 `Flux<AdvisedResponse>`。

两个接口都继承自基础标记接口 `Advisor`，并通过实现 `Ordered` 接口中的 `getOrder()` 方法确定自身在责任链中的嵌套深度。`order` 数值越小，越靠近外层，越优先执行前置处理。

![CallAroundAdvisor 与 StreamAroundAdvisor 的调用时序与流式处理对比](/images/content-rebuild/f09c7890b7175dfe/fig-2.png)

### 3.2 AdvisedRequest 与跨阶段数据传递

在拦截器链路中流转的核心数据载体是 `AdvisedRequest`。其关键设计在于提供了一个只在单次请求调用链生命周期内有效的共享容器：

```java
public record AdvisedRequest(
    ChatModel chatModel,
    String userText,
    Map<String, Object> userParams,
    String systemText,
    Map<String, Object> systemParams,
    List<Media> media,
    List<FunctionCallback> functionCallbacks,
    List<Message> messages,
    ChatOptions chatOptions,
    Map<String, Object> adviseContext // 跨 Advisor 共享状态数据容器
) {
    // Spring AI 提供了基于 Builder 的不可变复制机制
    public static Builder from(AdvisedRequest from) { ... }
}
```

`adviseContext` 是一个标准的 `Map<String, Object>`。各阶段 Advisor 严禁直接破坏原始用户请求对象，而是通过读取与写入 `adviseContext`，实现改写查询词、传递检索分块、缓存耗时指标等状态传递。

### 3.3 流式场景下的特别机制：检索不能流式进行

部分初学者会误以为“开启大模型流式输出后，知识库检索也是随着 token 一个个流式搜出来的”。这是对大语言模型计算机制的误解。

大语言模型的流式输出（Streaming）是指在模型完成 **Prefill 阶段（输入提示词的全量计算）** 后，在 **Decode 阶段（自回归解码）** 将生成的 Token 逐个通过 SSE（Server-Sent Events）推向客户端。大模型必须在生成第一个 Token 之前，获得包含检索文档的完整 Prompt。

因此在 `StreamAroundAdvisor` 中：
1. 预检索改写、向量检索与后检索重排依然在请求发起阶段**完整计算并拼装完毕**；
2. 拼装好的请求送入下游链，底层调用大模型接口返回 `Flux<ChatResponse>`；
3. Advisor 的流式处理主要发生在后置响应阶段：利用 Reactor 的 `doOnNext`、`doFinally` 统计打字流经过的 Token 总量，或者在流结束时记录审计日志。不能在响应流内部调用阻塞操作（如 `.collectList().block()`），否则会导致前端打字机效果彻底失效，退化为同步等待。

---

## 4. 预检索阶段：Query 改写与安全短路拦截

预检索（Pre-Retrieval）发生在向存储层发起查询之前。这一阶段的核心目标是提升查询词质量，并拦截非法请求。

![预检索阶段的 Query 改写与安全短路流程](/images/content-rebuild/f09c7890b7175dfe/fig-3.png)

### 4.1 预检索拦截器的工程实现

我们可以实现一个集成安全过滤与查询改写的自定义 Advisor：

```java
import org.springframework.ai.chat.client.advisor.api.AdvisedRequest;
import org.springframework.ai.chat.client.advisor.api.AdvisedResponse;
import org.springframework.ai.chat.client.advisor.api.CallAroundAdvisor;
import org.springframework.ai.chat.client.advisor.api.CallAroundAdvisorChain;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.core.Ordered;
import java.util.List;

public class PreRetrievalGuardAndRewriteAdvisor implements CallAroundAdvisor {

    private final SensitiveWordFilter wordFilter;
    private final QueryRewriter queryRewriter;

    public PreRetrievalGuardAndRewriteAdvisor(SensitiveWordFilter filter, QueryRewriter rewriter) {
        this.wordFilter = filter;
        this.queryRewriter = rewriter;
    }

    @Override
    public int getOrder() {
        // 优先级高于检索拦截器（默认检索组件通常设为 0，数值越小越先执行）
        return -100;
    }

    @Override
    public String getName() {
        return "PreRetrievalGuardAndRewriteAdvisor";
    }

    @Override
    public AdvisedResponse aroundCall(AdvisedRequest advisedRequest, CallAroundAdvisorChain chain) {
        String rawQuery = advisedRequest.userText();

        // 1. 安全合规前置检查：命中敏感词直接构建假响应短路返回，避免浪费后续检索与推理资源
        if (wordFilter.containsBlockedWords(rawQuery)) {
            ChatResponse shortCircuitResponse = ChatResponse.builder()
                .withGenerations(List.of(new Generation("输入内容包含合规拦截词，无法继续处理。")))
                .build();
            return new AdvisedResponse(shortCircuitResponse, advisedRequest.adviseContext());
        }

        // 2. 结合多轮上下文进行代词补全与关键词提炼
        String rewrittenQuery = queryRewriter.rewrite(rawQuery, advisedRequest.adviseContext());

        // 3. 将改写后的搜索词压入共享上下文，供下一环节的 VectorStore 使用
        advisedRequest.adviseContext().put("REWRITTEN_QUERY", rewrittenQuery);

        // 传递给责任链的下一个组件
        return chain.nextAroundCall(advisedRequest);
    }
}
```

关于多轮对话上下文改写的深入机制，可参考 [什么是上下文查询增强？](/interview/rag-interview/contextual-query-enhancement.md)。

<ReadMoreLock />

---

## 5. 检索阶段：QuestionAnswerAdvisor 与 VectorStore 集成

检索阶段（Retrieval）负责调用存储基础设施，根据预检索阶段生成的查询词获取候选文档。

Spring AI 内置的 `QuestionAnswerAdvisor` 是这一阶段的典型实现。它将向量检索、元数据过滤与上下文装填逻辑进行了标准化封装。

![QuestionAnswerAdvisor 内部数据流与检索集成](/images/content-rebuild/f09c7890b7175dfe/fig-4.png)

### 5.1 检索拦截器的内部工作机制

以下是 `QuestionAnswerAdvisor` 的简化执行机制：

```java
public class QuestionAnswerAdvisor implements CallAroundAdvisor {
    private final VectorStore vectorStore;
    private final SearchRequest searchRequest;
    private final String userTextAdvise;

    public static final String RETRIEVED_DOCUMENTS = "qa_retrieved_documents";
    public static final String DEFAULT_USER_TEXT_ADVISE = """
        Context information is below.
        ---------------------
        {question_answer_context}
        ---------------------
        Given the context and provided history information, answer the query.
        """;

    @Override
    public AdvisedResponse aroundCall(AdvisedRequest advisedRequest, CallAroundAdvisorChain chain) {
        // 1. 读取预检索生成的改写词；若无改写词则兜底使用原始用户输入
        String query = (String) advisedRequest.adviseContext()
            .getOrDefault("REWRITTEN_QUERY", advisedRequest.userText());

        // 2. 调用向量数据库执行相似度检索
        SearchRequest requestToUse = this.searchRequest.withQuery(query);
        List<Document> documents = this.vectorStore.similaritySearch(requestToUse);

        // 3. 将检索出来的原始文档集存入上下文，供后检索组件使用
        advisedRequest.adviseContext().put(RETRIEVED_DOCUMENTS, documents);

        // 4. 将召回文本拼接到提示词参数模板中
        String documentContext = documents.stream()
            .map(Document::getText)
            .collect(Collectors.joining(System.lineSeparator()));

        Map<String, Object> userParams = new HashMap<>(advisedRequest.userParams());
        userParams.put("question_answer_context", documentContext);

        AdvisedRequest updatedRequest = AdvisedRequest.from(advisedRequest)
            .withUserText(advisedRequest.userText() + System.lineSeparator() + this.userTextAdvise)
            .withUserParams(userParams)
            .build();

        // 5. 驱动责任链继续向下流转
        return chain.nextAroundCall(updatedRequest);
    }
}
```

### 5.2 向量库检索与元数据过滤

在企业生产场景下，检索很少使用无差别的全库相似度比对。通过设置 [元数据过滤（Metadata Filtering）](/interview/rag-interview/rag-metadata-filtering.md)，可以在向量库索引过滤阶段完成多租户或业务线权限隔离：

```java
SearchRequest searchRequest = SearchRequest.builder()
    .topK(15) // 粗排召回较多候选块
    .similarityThreshold(0.70) // 过滤低相似度噪声
    .filterExpression(new Filter.Expression(
        Filter.ExpressionType.EQ,
        new Filter.Key("department_id"),
        new Filter.Value("dept_rd_01")
    ))
    .build();
```

---

## 6. 后检索阶段：Rerank 精排与脱敏注入

后检索阶段（Post-Retrieval）位于“文档召回之后，模型推理之前”。这一阶段的目标是对粗排召回的一批分块进行精细处理，解决两类问题：
1. **首尾效应（Lost in the Middle）**：模型在长上下文中往往更关注开头和结尾，粗排召回的低相关性文档如果堆在关键位置，会干扰最终生成；
2. **敏感数据泄漏**：知识库原文中包含的真实手机号、身份证号或内部秘钥，需要在送给大模型前完成脱敏。

![后检索阶段 Rerank 重排与脱敏裁剪流程](/images/content-rebuild/f09c7890b7175dfe/fig-5.png)

### 6.1 后检索拦截器的嵌套执行逻辑

后检索 Advisor 的优先级数值应设置得比检索组件更靠后（例如 `order = 100`）。在责任链执行时，外层的 `QuestionAnswerAdvisor` 调用 `chain.nextAroundCall(updatedRequest)`，此时内层的后检索 Advisor 被触发：

```java
public class PostRetrievalRerankAdvisor implements CallAroundAdvisor {

    private final RerankService rerankService;
    private final MaskingService maskingService;
    private final int finalTopK;

    public PostRetrievalRerankAdvisor(RerankService rerankService, MaskingService maskingService, int topK) {
        this.rerankService = rerankService;
        this.maskingService = maskingService;
        this.finalTopK = topK;
    }

    @Override
    public int getOrder() {
        return 100; // 晚于 QuestionAnswerAdvisor 执行
    }

    @Override
    public String getName() {
        return "PostRetrievalRerankAdvisor";
    }

    @Override
    public AdvisedResponse aroundCall(AdvisedRequest advisedRequest, CallAroundAdvisorChain chain) {
        // 1. 从共享上下文中提取检索阶段放入的原始候选文档
        @SuppressWarnings("unchecked")
        List<Document> rawDocs = (List<Document>) advisedRequest.adviseContext()
            .get(QuestionAnswerAdvisor.RETRIEVED_DOCUMENTS);

        if (rawDocs != null && !rawDocs.isEmpty()) {
            String query = (String) advisedRequest.adviseContext()
                .getOrDefault("REWRITTEN_QUERY", advisedRequest.userText());

            // 2. 调用 Cross-Encoder 精排打分，将 15 篇候选文档截断提炼为最相关的 Top-3
            List<Document> rerankedDocs = rerankService.rerank(query, rawDocs, this.finalTopK);

            // 3. 对精排胜出的分块执行敏感信息正则脱敏
            List<Document> sanitizedDocs = rerankedDocs.stream()
                .map(doc -> new Document(
                    doc.getId(),
                    maskingService.maskSensitiveInfo(doc.getText()),
                    doc.getMetadata()
                ))
                .toList();

            // 4. 重写上下文变量，覆盖掉检索阶段粗排拼装的大段内容
            String refinedContext = sanitizedDocs.stream()
                .map(Document::getText)
                .collect(Collectors.joining(System.lineSeparator() + "---" + System.lineSeparator()));

            Map<String, Object> updatedParams = new HashMap<>(advisedRequest.userParams());
            updatedParams.put("question_answer_context", refinedContext);

            advisedRequest = AdvisedRequest.from(advisedRequest)
                .withUserParams(updatedParams)
                .build();
        }

        // 5. 递交给底层 ChatModel 开展最终推理
        return chain.nextAroundCall(advisedRequest);
    }
}
```

关于重排序算法与双塔召回的精度对比，详见 [Rerank 重排序技术与交叉编码器应用](/interview/rag-interview/rag-rerank.md)。

---

## 7. 企业级工程集成与可靠性设计

将模块化 RAG 落地到真实企业环境时，不能只关注逻辑是否走通，还必须结合 Spring 生态的基础设施解决高并发、网络抖动与多租户权限问题。

| 集成维度 | 潜在故障与架构风险 | Spring 生态应对方案 | 核心验证与验收手段 |
| :--- | :--- | :--- | :--- |
| **线程池与 IO 隔离** | 向量检索与精排网络耗时较长，容易耗尽 Web 容器的工作线程 | 为 Advisor 内部的向量计算配置独立的 `TaskExecutor` 线程池，或采用虚拟线程 | 压力测试下观察 Tomcat 工作线程池活跃数与请求排队情况 |
| **全链路追踪** | 链路跨越“改写 -> 向量库 -> 精排 -> 模型”，定位慢调用耗时困难 | 引入 `micrometer-tracing`，自动在 Advisor 之间传递 TraceId | 在分布式追踪系统（如 Zipkin/Tempo）中观察各 Span 的耗时分布 |
| **依赖故障降级** | 向量数据库发生超时或网络抖动，直接导致接口返回 HTTP 500 | 结合 `Resilience4j` 或在 Advisor 内捕获异常，清空上下文降级为模型直接回答 | 单元测试模拟 VectorStore 断连，验证接口是否返回保底响应 |
| **多租户安全** | 异步调度或线程切换导致 Spring Security 上下文丢失，引发越权检索 | 使用 `DelegatingSecurityContextAsyncTaskExecutor` 透传安全上下文 | 单元测试验证子线程获取到的租户 Token 与父线程严格一致 |

![企业级架构保障：线程隔离、观测指标与故障降级](/images/content-rebuild/f09c7890b7175dfe/fig-6.png)

### 7.1 企业级 ChatClient 组装示例

以下配置演示了如何在 Spring 容器中将三个阶段的组件以有序声明的方式注册进 `ChatClient`：

```java
@Configuration
public class RagPipelineConfiguration {

    @Bean
    public ChatClient enterpriseRagChatClient(
        ChatClient.Builder builder,
        VectorStore vectorStore,
        SensitiveWordFilter sensitiveWordFilter,
        QueryRewriter queryRewriter,
        RerankService rerankService,
        MaskingService maskingService
    ) {
        return builder
            .defaultSystem("你是一个严谨的企业知识助手。必须基于提供的背景知识回答。")
            // 1. 预检索阶段：安全拦截与 Query 改写（Order: -100，最外层优先执行）
            .defaultAdvisors(new PreRetrievalGuardAndRewriteAdvisor(sensitiveWordFilter, queryRewriter))
            // 2. 检索阶段：召回候选文档并放入上下文容器（Order: 0）
            .defaultAdvisors(new QuestionAnswerAdvisor(vectorStore, SearchRequest.builder().topK(15).build()))
            // 3. 后检索阶段：精排截断与 PII 脱敏（Order: 100，内层在模型调用前处理）
            .defaultAdvisors(new PostRetrievalRerankAdvisor(rerankService, maskingService, 3))
            .build();
    }
}
```

---

## 8. 面试高频追问与架构辨析

### 追问 1：Spring AI 的 Advisor 链与 LangChain 的 LCEL（表达式语言）在设计哲学上有什么区别？
- **回答要点**：
  - LangChain 的 LCEL 基于函数式有向无环图（DAG），强调算子之间的数据管道映射（`Prompt | Model | OutputParser`），图的拓扑编排非常灵活，但 Java 开发者上手函数式链式语法的认知成本较高。
  - Spring AI 的 Advisor 采用经典的 Java Web Filter / Spring AOP 环绕切面模式。每个 Advisor 既能处理入站请求，又能拦截出站响应；借助 `adviseContext` 状态容器，各组件可以在不改变下游接口签名的前提下透明传递数据。这种设计与 Spring MVC 拦截器和 Spring Security 过滤器链一脉相承，符合 Java 工程习惯，但在表达复杂网状分支时不如图编排机制直观。

### 追问 2：如果在流式调用（Streaming）中，向量检索因网络超时报错，会发生什么？如何保证可用性？
- **回答要点**：
  - 在流式模式下，向量检索发生在请求下游链之前（即进入 Prefill 推理前）。此时 HTTP 响应式流尚未发射数据 Chunk。
  - 如果在 `StreamAroundAdvisor` 内部抛出 `QueryTimeoutException`，且没有做保护处理，整个请求会直接中断，Controller 抛出异常并向客户端返回 HTTP 500。
  - **健壮做法**：在检索 Advisor 内部使用 `try-catch` 捕获数据库异常，或者在 Reactor 流中配置降级逻辑。当向量库超时不可用时，记录告警日志，将上下文置为空，并继续向下游大模型传递请求，让大模型以常规知识或统一的兜底话术回答，避免系统完全不可用。

### 追问 3：为什么文档切分（Chunking）和向量化（Embedding Ingestion）不在 Spring AI 的 Advisor 体系里？
- **回答要点**：
  - **生命周期完全不同**：Advisor 专用于管理用户的**在线实时推理请求生命周期**（Online Inference）；而文档切分、清洗与向量化属于**离线或准实时的批处理数据管道**（ETL Ingestion Pipeline）。
  - Spring AI 对离线数据处理提供了专门的抽象，即 `DocumentReader`、`DocumentTransformer`（如 `TokenTextSplitter`）与 `VectorStore.write()`。不能把数据写入流水线与在线请求拦截切面混为一谈。

---

## 9. 常见认知误区与避坑指南

- ❌ **误区一：误以为后检索 Advisor 只能在模型推理结束后修改文本**
  - *机制纠正*：Advisor 是**环绕通知**（Around Advice）。在 `PostRetrievalRerankAdvisor` 调用 `chain.nextAroundCall()` 之前，当前处于向内穿透的请求阶段。此时它完全可以修改 `AdvisedRequest` 中的提示词参数（例如重新精排并替换上下文变量），然后再交给真正的 `ChatModel` 推理。
- ❌ **误区二：在 Advisor 内部使用未受管的自定义线程池导致租户凭证丢失**
  - *机制纠正*：在 Spring 体系中，用户身份与租户 ID 默认存放在 `ThreadLocal`（如 `SecurityContextHolder`）中。如果在 Advisor 内部直接使用裸线程池（如 `Executors.newFixedThreadPool()`）执行异步任务，子线程无法继承主线程上下文，会导致后续向量库查询缺少租户条件，甚至引发越权安全事故。必须使用 `DelegatingSecurityContextAsyncTaskExecutor` 或通过 `adviseContext` 显式传参。
- ❌ **误区三：在 StreamAroundAdvisor 响应链路中调用阻塞操作**
  - *机制纠正*：在流式生成场景下，禁止在响应流的下游拦截器中编写 `flux.collectList().block()` 来统计 Token。这会导致模型输出被强行缓冲至内存，前端的打字机体验彻底失效。

在掌握了 Advisor 拦截链路的设计之后，建议进一步阅读 [什么是 Re-Reading？如何基于 Spring AI 实现 Re-Reading Advisor？](/interview/rag-interview/re-reading-spring-ai.md)，了解如何利用 Advisor 模式为提示词注入二次复读增强能力。
