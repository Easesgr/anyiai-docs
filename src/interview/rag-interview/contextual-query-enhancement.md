---
title: 什么是上下文查询增强？它有什么作用？如何基于 Spring AI 实现上下文查询增强来处理无关问题？
icon: mdi:magnify-plus
tag:
  - 查询增强
  - Spring AI
  - RAG
---

# 什么是上下文查询增强？它有什么作用？如何基于 Spring AI 实现上下文查询增强来处理无关问题？

这道题考的是 RAG 系统中 Advanced RAG 阶段的核心优化技术。面试官想看你能不能把"用户问的东西和知识库写的东西怎么对上"这件事讲清楚。

我从四个方面来讲：概念定义、解决什么问题、核心原理（Query Rewrite 和 HyDE）、Spring AI 怎么实现。

---
## 1. 先搞懂概念——上下文查询增强是什么？

上下文查询增强（Contextual Query Enrichment）是 Advanced RAG 阶段的关键技术。

简单说就是：**在检索之前，先把用户的问题改写一遍，让它更容易从知识库里找到东西**。

RAG 发展有三个阶段：

**Naive RAG** → 用户问什么就直接去检索，容易踩坑

**Advanced RAG** → 加了预处理环节，Query Rewrite 就是典型代表

**Modular RAG** → 各个模块自由组合，更灵活

上下文查询增强属于 Advanced RAG 阶段做的事。它不是直接拿用户的问题去检索，而是先"翻译"一下，把口语化的、模糊的、简略的提问，变成更适合检索的形式。

打个比方：用户像个"说不清需求的甲方"，上下文查询增强就是"需求分析师"，帮你把需求整理清楚，再去找知识库要答案。

---
<ReadMoreLock />

## 2. 它解决什么问题？

RAG 系统有六大痛点：

1. **Missing Content** —— 知识库里压根没有
2. **Missed Top Ranked** —— 正确答案没排到前面
3. **Not in Context** —— 检索到了内容，但和问题对不上
4. **Wrong Format** —— 结果格式不对
5. **Incomplete** —— 结果不完整
6. **Not Extracted** —— 检索到了但没提取出来

上下文查询增强主要解决 **Not in Context** 这个问题。

什么意思？来看个例子：

用户问："怎么重启服务？"

知识库里写的是："服务重启步骤包括：1. 停止进程 2. 检查端口 3. 启动服务"

Embedding 直接匹配的话，"重启"和"重启步骤"可能匹配度不高。但如果先把问题改写成"服务重启步骤有哪些"，再去检索，就能命中了。

这就是上下文查询增强的价值：**弥合用户问法和知识库表述之间的差异**。

---
## 3. 核心原理—— Query Rewrite 和 HyDE

上下文查询增强有两种主流实现方式：**Query Rewrite** 和 **HyDE**。

### Query Rewrite：翻译官

Query Rewrite 的思路很直接：**用 LLM 把用户的问题改写一遍**。

改写的方式有几种：

**同义词替换**："电脑蓝屏" → "计算机显示蓝色错误画面"

**问题澄清**：用户问得很模糊，比如"那个问题怎么办"，LLM 会结合对话上下文推断具体指什么

**查询扩展**：把一个简单问题扩展成多个相关问题，"Java 怎么优化" → "Java性能优化方法"、"Java垃圾回收优化"、"Java内存管理最佳实践"

改写前："内存溢出咋整"

改写后："内存溢出（OutOfMemoryError）的排查步骤和解决方案"

检索结果：✅ 精准命中知识库内容

这就是 Query Rewrite 的魔力。

### HyDE：先猜答案再找证据

HyDE 全称 Hypothetical Document Embeddings，翻译过来是"假设性文档嵌入"。

它的思路很特别：**不直接改写问题，而是让 LLM 先根据问题生成一个假设性的答案片段，然后用这个假设答案去检索**。

流程是这样的：

1. 用户问："Spring Boot 如何集成 Redis"
2. LLM 生成假设答案："在 Spring Boot 项目中，可以通过添加 spring-boot-starter-data-redis 依赖，配置 RedisConnectionFactory..."
3. 用这个假设答案的 embedding 去向量数据库检索
4. 找到真正匹配的文档

HyDE 的原理是：**假设答案和真实答案的 embedding 更接近**，因为它们语义相似。这是一种"以结果为导向"的检索思路。

### 混合检索 + Rerank

单独的 Query Rewrite 或 HyDE 有时候还不够用，业界常用的做法是：

**混合检索**：BM25（关键词检索）+ 向量检索一起跑

**Rerank**：用 Cross-Encoder 对初步检索结果重新排序，把最相关的排到前面

**上下文压缩**：把检索到的长文档压缩，只保留和问题相关的部分

这四步组合起来，就是完整的 Advanced RAG 优化链路。

---

## 4. Spring AI 实战——怎么实现上下文查询增强？

Spring AI 提供了统一的抽象，类似 JDBC 屏蔽数据库差异那样，屏蔽了底层向量库的差异。

### 实现 Query Rewrite

核心是用 Spring AI 的 ChatClient 调用 LLM 做改写：

java
@Autowired
private ChatClient chatClient;

public String rewriteQuery(String userQuery) {
    String prompt = """
        你是一个查询改写助手。将用户的原始问题改写成更适合检索的形式。
        改写要求：
        1. 使用更正式、技术性的表达
        2. 可以添加同义词和相关概念
        3. 保持原意不变
        
        原始问题：%s
        改写后的问题：
        """.formatted(userQuery);
    
    return chatClient.prompt()
        .user(prompt)
        .call()
        .content();
}


这样就能把用户输入转化成更适合检索的形式。

### 结合向量数据库

假设你用的是 Milvus：

java
@Bean
public VectorStore vectorStore() {
    return new MilvusVectorStore(new MilvusVectorStoreOptions.Builder()
        .host("localhost")
        .port(19530)
        .databaseName("knowledge_base")
        .build());
}


### 实现 HyDE

HyDE 的实现稍微复杂一点，需要先生成假设答案：

java
public String generateHypotheticalAnswer(String query) {
    String prompt = """
        基于以下问题，生成一段假设性的回答。这个回答可能不完全准确，但语义上与正确答案接近。
        问题：%s
        """.formatted(query);
    
    return chatClient.prompt()
        .user(prompt)
        .call()
        .content();
}


然后用这个假设答案去检索：

java
public ListhydeSearch(String query) {
    // 1. 生成假设答案
    String hypotheticalAnswer = generateHypotheticalAnswer(query);
    
    // 2. 用假设答案检索
    return vectorStore.similaritySearch(hypotheticalAnswer);
}


### 完整链路示例


用户问题 → Query Rewrite → 混合检索(BM25 + 向量) → Rerank → 上下文压缩 → LLM 生成


Spring AI 支持多种向量数据库，Milvus、Pinecone、Chroma、Weaviate 都可以，选型建议是：

- 小规模项目用 Chroma（轻量、本地部署）
- 生产环境用 Milvus 或 Pinecone（支持分布式、可扩展）

---

## 面试怎么答？

### 基础版

上下文查询增强是 Advanced RAG 的核心技术。它解决的是"用户问的东西和知识库写的东西对不上"的问题。

实现方式主要是 Query Rewrite，用 LLM 把用户的原始问题改写成更适合检索的形式，比如同义词替换、问题澄清、查询扩展。另一种方式是 HyDE，先生成假设性答案片段，再用它去检索。

### 加分版

可以画出 RAG 演进三阶段：Naive RAG → Advanced RAG → Modular RAG，上下文查询增强位于 Advanced RAG 阶段。

Query Rewrite 和 HyDE 各有适用场景：Query Rewrite 适合处理口语化、模糊的提问；HyDE 适合问题表述和答案风格差异大的情况。

完整优化链路是：Query Rewrite + 混合检索（BM25 + 向量）+ Cross-Encoder Rerank + 上下文压缩。Spring AI 可以通过 ChatClient + VectorStore 实现这套链路，底层向量库可以根据规模选择 Milvus 或 Pinecone。

---

## 一句话总结

上下文查询增强就是**在检索之前先"翻译"一下用户的问题**，让口语化的提问变成技术化的检索语句，弥合用户问法和知识库表述之间的鸿沟。
