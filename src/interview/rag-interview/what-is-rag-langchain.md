---
title: 什么是 RAG？在 LangChain 中如何实现 RAG 应用？
icon: mdi:database-search
tag:
  - RAG
  - LangChain
  - 检索增强
---

# 什么是 RAG？在 LangChain 中如何实现 RAG 应用？

这道题考的是 RAG（检索增强生成）技术的完整理解，从检索到生成的全链路设计。面试官想看你能不能把"大模型 + 知识库"这件事讲清楚。

我从四个方面来讲：RAG 核心原理、双管道架构、LangChain 实现细节、面试加分项。

---
## RAG 是什么？用"开卷考试"理解它

RAG = Retrieval（检索）+ Augmentation（增强）+ Generation（生成）。

大白话就是：让大模型"查资料"回答问题，而不是靠"背答案"。

你想啊，LLM 的知识是固定的，训练完就定型了。你让它回答你们公司的内部规章制度，它肯定一脸懵。但 RAG 解决了这个问题——把公司制度存到向量数据库，用户问问题时先检索相关段落，再让 LLM 结合这些上下文回答。

**核心思想就一句话：把知识从模型权重（隐式记忆）转移到外部向量数据库（显式检索）。**

### "开卷考试"比喻

RAG 对 LLM 就像开卷考试对学生。

闭卷考试考的是记忆，学生得把公式全背下来。开卷考试不一样，学生可以带参考资料进场，考的是推理和理解能力。

RAG 就是这么回事。它不是让 LLM 死记硬背所有知识，而是让它"带参考资料进场"，实时检索上下文来回答。

这个比喻面试时说出来，比干巴巴念定义强多了。

---
<ReadMoreLock />

## RAG 的两大管道

RAG 系统有两套核心流程：**索引管道**和**检索生成管道**。

### 索引管道（离线准备）

这套流程在用户提问之前就跑完了，相当于"提前备好菜"。

**加载 → 分割 → 存储**

加载文档，分割成小块，向量化后存进向量数据库。

这个过程只跑一次，之后就可以反复检索。

### 检索生成管道（在线运行）

用户提问时才触发这套流程。

**查询 → 检索 → 增强 → 生成**

用户问"年假怎么休"，系统先把这句话向量化，去向量数据库里找最相似的文档块，然后把找到的内容塞进提示词模板，丢给 LLM 生成回答。

### 做饭的比喻

索引管道就像做饭前洗菜切菜放进冰箱，都是提前准备工作。

检索生成就像客人点餐后，从冰箱拿出食材开始炒菜上桌。

---
## LangChain 实现 RAG 的索引三步骤

在 LangChain 里实现 RAG，索引管道分三步：

### 第一步：加载（Loading）

用 DocumentLoader 把各种格式的文档读进来。

LangChain 支持 160+ 种文档加载器，PDF、Word、网页、数据库都能接。

python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("公司制度.txt")
documents = loader.load()  # 返回 Document 对象列表


### 第二步：分割（Splitting）

文档太大塞不进 LLM 的上下文窗口，得切成小块。

用 TextSplitter，按 chunk_size 分割。

python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # 每块1000个字符
    chunk_overlap=200   # 相邻块重叠200字符
)
chunks = splitter.split_documents(documents)


**chunk_size 很关键。** 设太大会超过 LLM 的上下文窗口，设太小会丢失语义。1000 左右是常见值，具体看你的 LLM 支持多长的上下文。

chunk_overlap 的作用是保证相邻块之间的上下文连贯性，防止一句话被拦腰截断。

### 第三步：存储（Storing）

用嵌入模型把文本块转成向量，存进向量数据库。

python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)


LangChain 支持 40+ 种向量数据库，Chroma、Milvus、Pinecone、FAISS 都能选。

---

## LangChain 检索生成与 LCEL 链式调用

索引建好了，现在看怎么用。

### 检索

把向量存储转成检索器：

python
retriever = vectorstore.as_retriever()


这一步把向量数据库封装成 Retriever 接口，之后查询直接调 .invoke() 就行。

### LCEL 链式调用

LangChain 的 LCEL（LangChain Expression Language）用 | 操作符串联各个组件，像工厂流水线一样。

python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template(
    "根据以下上下文回答问题。\n\n上下文：{context}\n\n问题：{question}"
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | llm 
    | StrOutputParser()
)

result = chain.invoke("年假怎么休？")


这个链的逻辑：

1. {"context": retriever, "question": RunnablePassthrough()} — 用户问题触发检索，同时传给 question
2. | prompt — 把检索结果和原问题填进提示模板
3. | llm — LLM 根据增强后的提示生成回答
4. | StrOutputParser() — 解析成字符串输出

### 更简洁的写法

LangChain 提供了内置的 create_retrieval_chain，一行搞定：

python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

retrieval_chain = create_retrieval_chain(retriever, document_chain)


这个链返回的结果里还有个 source_documents 字段，能追溯答案是从哪些文档来的。

---

## RAG 优化：面试加分项

上面讲的是基础版，能跑起来。加分项在于优化。

### MultiQueryRetriever

用户问"怎么请假"，但数据库里可能写的是"休假申请流程"。

同一个意思有多种表达方式，MultiQueryRetriever 会自动生成多个角度的查询，提高召回率。

python
from langchain.retrievers import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)


### 上下文压缩

检索回来的文档可能包含大量无关内容，压缩一下再喂给 LLM，省 Token。

python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)


### 元数据过滤器

精准筛选，比如只查"2024年"或"技术部"的文档。

python
retriever = vectorstore.as_retriever(
    search_kwargs={"filter": {"department": "技术部"}}
)


---

## 面试怎么答

### 基础版（能过）

> RAG 是检索增强生成，通过向量数据库存储知识，查询时先检索相关文档块，再让 LLM 结合上下文生成回答。
> 
> LangChain 实现分两步。索引管道：DocumentLoader 加载文档，TextSplitter 按 chunk_size=1000 分割，VectorStore 向量化存储。检索生成管道：用户问题先向量化做相似度检索，取 Top-K 结果填充提示模板，LLM 生成回答。LangChain 用 LCEL 的 | 操作符串联 retriever、prompt、llm 组成链。

### 加分版（眼前一亮）

> RAG 对 LLM 就像开卷考试对学生——考的是推理能力，不是记忆能力。
> 
> 我在项目里用 LangChain 的 DocumentLoader 加载 PDF/Word 文档，TextSplitter 分割时 chunk_size 根据 LLM 上下文窗口调整，overlap 设置 200 保证语义连贯。检索用 vectorstore.as_retriever()，生成用 LCEL 链串联。
> 
> 优化方面用过 MultiQueryRetriever 从多角度生成查询提高召回，上下文压缩减少 Token 消耗。生产环境还用到 create_retrieval_chain 的 source_documents 做答案溯源。

---

## 一句话总结

RAG 就是让大模型"查资料"回答问题，LangChain 通过 DocumentLoader 加载、TextSplitter 分割、VectorStore 存储构建索引，用 LCEL 链串联检索和生成。
