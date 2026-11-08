---
title: 如何在 LangChain 中实现 RAG 应用？
icon: mdi:database-search
tag:
- RAG
- LangChain
- 检索增强
description: 系统讲解 RAG 的定义与原理，结合企业知识库案例剖析 LangChain 离线索引与在线生成的组件拆解、LCEL 数据流转、数据类型对齐与生产陷阱。
---

大语言模型的预训练知识具有固定的时间截止点，且无法直接获知企业私有数据。如果直接向模型提问内部制度，模型往往只能根据通用经验推测，甚至给出看似合理却完全错误的答案（即模型幻觉）。

**RAG（Retrieval-Augmented Generation，检索增强生成）** 的解决思路是：在向模型提问之前，先根据用户的问题从外部知识库中检索出相关的资料片段，再将这些片段拼装进提示词（Prompt）中一并提交给大语言模型，让模型基于参考材料作答。

为了让整套流程容易理解，本文使用一个具体的贯穿案例：**企业 HR 助手**。知识库中存有公司的考勤与休假管理制度，用户的提问是：“*入司未满一年，有带薪年假吗？*”。

![LCEL 中的 RAG 执行链路全景映射](/images/content-rebuild/499f87a390cef2b2/cover.png)

---

## 1. 面试速答：1 分钟口述答案

在面试中被问及该问题时，建议先给出结构化的口述总结：

> **口述模板**：
> “在 LangChain 中实现基础 RAG 应用，主要划分为**离线索引**和**在线检索生成**两个阶段：
> 
> 1. **离线索引阶段**：使用 `DocumentLoader` 读取原始文档（如 PDF 或 Markdown），生成标准化 `Document` 列表；再通过 `RecursiveCharacterTextSplitter` 将长文档按指定大小切分为文本块；接着调用 `Embeddings` 模型将文本块转换为向量；最后存入 `VectorStore`（如 FAISS、Qdrant）构建向量索引。
> 2. **在线检索与生成阶段**：通过 `vectorstore.as_retriever()` 将向量库封装为符合 LangChain 统一协议的 `Retriever`。在 LCEL（LangChain 表达式语言）中，利用 `RunnableParallel`（或字典解构）构建两条并行的输入支流：一条支流把问题交给检索器获取相关文档，并通过格式化函数拼接为纯文本字符串注入上下文参数；另一条支流通过 `RunnablePassthrough` 将原始问题透传给问题参数。
> 3. **结果汇总与生成**：双支流合并后的字典传入 `ChatPromptTemplate` 渲染 Prompt，交由 `ChatModel` 计算，最终通过 `StrOutputParser` 提取出纯文本回复。”

<ReadMoreLock />

---

## 2. 核心抽象拆解：从原始文件到可检索组件

在 LangChain 中实现上述过程，需要理解离线索引和在线检索所涉及的核心类与接口契约。

![LangChain 离线索引四大组件与对象转换关系](/images/content-rebuild/499f87a390cef2b2/fig-1.png)

### 2.1 离线索引组件：DocumentLoader 与 TextSplitter
- **DocumentLoader**：负责文件解析与 I/O 读取。无论是本地的 PDF、Word，还是网页 HTML，经过 Loader 处理后都会统一转化为 LangChain 规范的最小数据载体 `Document` 对象。每个 `Document` 包含两个属性：
  - `page_content: str`：切分前或解析出的文本内容。
  - `metadata: dict`：记录元数据，如文件来源路径（`source`）、页码（`page`）或修改时间。
- **TextSplitter**：负责解决文本长度与语义完整性的平衡问题。大模型和嵌入模型的输入上下文长度有限，且整篇长文档直接检索容易稀释关键信息。切分器将大文档切割成带有适度重叠（overlap）的块。关于切分窗口与重叠长度的详细设置，可参考 [文档切分为什么重要？](chunk-strategy.md)。

### 2.2 VectorStore 与 Retriever 的区别
很多初学者会疑惑：**既然 VectorStore 已经提供了向量相似度搜索方法，为什么 LangChain 还要单独定义 Retriever 接口？**

![VectorStore 存储接口与 Retriever 检索抽象的架构解耦](/images/content-rebuild/499f87a390cef2b2/fig-2.png)

两者在系统职责与接口设计上存在明确分工：

| 维度 | VectorStore（向量存储库） | Retriever（检索器） |
| :--- | :--- | :--- |
| **设计定位** | **数据存储与底层索引**（存储层） | **通用查询协议**（执行调度层） |
| **核心职责** | 负责向量与元数据的增删改查（CRUD）、底层向量近邻索引构建（如 HNSW、IVF）及持久化。 | 负责封装检索策略，接收自然语言查询，返回候选文档列表。 |
| **接口形式** | 提供 `add_documents`、`similarity_search` 等数据库专有接口。 | 继承 `Runnable[str, List[Document]]`，统一提供 `invoke(query)`、`ainvoke(query)` 和 `astream(query)`。 |
| **底层实现依赖** | 强绑定具体存储系统（如 Chroma、FAISS、PGVector、Qdrant）。 | **不依赖具体存储形式**。检索器既可以来自 VectorStore，也可以封装 BM25 词频检索、外部搜索引擎接口，或多路召回融合。 |

通过将检索行为标准化为 `Retriever`，在线生成链路只需依赖统一的 `Runnable` 契约。即使后续将底层的内存向量表替换为分布式向量数据库，生成链的代码也无需任何改动。关于向量数据库的具体原理，可参考 [什么是向量数据库？](vector-database.md)。

---

## 3. 基于 LCEL 的链式组装与代码实现

在现代 LangChain（v0.1+ 及后续版本）中，推荐使用 LCEL（LangChain Expression Language）通过管道符 `|` 进行声明式组装。

以下是以 HR 助手为例的完整运行代码：

```python
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. 模拟切分后的规章制度文档块
raw_docs = [
    Document(
        page_content="年假政策：员工入职转正且连续工龄满一年后，每年享受5个工作日的带薪年假。未满一年者不享有法定带薪年假。",
        metadata={"source": "hr_handbook.pdf", "chapter": "假期管理"}
    ),
    Document(
        page_content="加班与调休：非工作日加班需在OA系统提前提交申请，经主管审批通过后可按1:1折算调休时间。",
        metadata={"source": "hr_handbook.pdf", "chapter": "工时制度"}
    )
]

# 2. 向量化并构建索引，转换为统一的 Retriever 接口
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(raw_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 3. 定义文档格式化函数：将文档列表转为单段提示词文本
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# 4. 定义提示词模板，包含 context 和 question 两个变量插槽
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个严谨的HR助手。请根据下述参考文档回答问题。如果参考文档没有提及，请明确说明无法从已知信息中确认：\n\n{context}"),
    ("human", "{question}")
])

# 5. 实例化聊天模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 6. 使用 LCEL 串联整条 RAG 链路
rag_chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)

# 7. 执行调用
query = "入司未满一年，有带薪年假吗？"
response = rag_chain.invoke(query)
print(response)
```

### 为什么在此处必须使用 RunnableParallel？
外部调用 `rag_chain.invoke(query)` 时，输入是一个单独的字符串（`"入司未满一年，有带薪年假吗？"`）。然而下游的 `prompt` 模板需要接收一个包含 `context` 和 `question` 两个键的字典。

![RunnableParallel 内部并行分发与字典归并数据流](/images/content-rebuild/499f87a390cef2b2/fig-3.png)

`RunnableParallel` 在此处起到了分流与重组的作用：
1. 它将上游传入的用户问题复制分发给字典内部的两个独立分支。
2. **支路一（context 分支）**：调用 `retriever` 获取与问题相关的 `List[Document]`，紧接着通过 `format_docs` 转换为单一字符串，写入字典的 `context` 键中。
3. **支路二（question 分支）**：通过 `RunnablePassthrough()` 将用户问题原样保留，写入字典的 `question` 键中。
4. 两条支路执行完成后，合并为一个符合预期的字典 `{"context": "...", "question": "..."}`，无缝传递给后续的 `ChatPromptTemplate`。

---

## 4. 链路中的数据类型流动

为了在代码出错时能快速定位，需要清楚数据在 LCEL 管道中流经每个节点时的具体类型演进：

![端到端 LCEL 执行栈中的数据类型单向演进图](/images/content-rebuild/499f87a390cef2b2/fig-4.png)

| 处理节点 | 接收类型 | 执行动作 | 输出类型 |
| :--- | :--- | :--- | :--- |
| **链入口 (Entry)** | `str` | 接收外部调用输入的提问字符串 | `str` |
| **Retriever** | `str` | 将提问向量化，在向量库中执行 Top-K 检索 | `List[Document]` |
| **format_docs** | `List[Document]` | 提取各个文档的 `page_content` 并拼接文本 | `str` |
| **RunnablePassthrough** | `str` | 原样透传用户原始提问 | `str` |
| **RunnableParallel 汇聚** | `str` | 并行分发并合并子分支结果，组装字典 | `Dict[str, str]` |
| **ChatPromptTemplate** | `Dict[str, str]` | 将字典对应键填充进 Prompt 模板插槽 | `ChatPromptValue` |
| **ChatOpenAI** | `ChatPromptValue` | 序列化为大模型 API 请求体并调用推理 | `AIMessage` |
| **StrOutputParser** | `AIMessage` | 提取模型回复对象中的 `.content` 纯文本 | `str` |

### 常见错误：忽略格式化直接注入
如果将代码写成 `"context": retriever` 而省略了 `| format_docs`，字典中的 `context` 键值将是 `List[Document]`。
此时 Python 渲染模板时会对该列表调用默认的 `str()`，导致形如 `[Document(page_content='年假政策...', metadata={'source': ...})]` 的调试代码字符串被拼入 Prompt。这不仅白白浪费 Token，还会增加模型在理解文本时的解析负担。因此，显式的文本提取与拼装不可缺少。

---

## 5. 生产环境常见陷阱与应对策略

上述代码足以用于理解原理与本地验证，但在并发服务与生产调用中，需要注意以下常见问题。

![生产环境下异步事件调度与同步阻塞陷阱对比](/images/content-rebuild/499f87a390cef2b2/fig-5.png)

### 陷阱 1：在异步框架中调用同步 invoke
现代 Web 服务（如 FastAPI 或 Tornado）依赖主线程中的异步事件循环（Event Loop）。若在异步接口中直接调用 `rag_chain.invoke()`：
- 检索阶段的数据库网络请求会阻塞主线程。
- 大模型生成推理的首字等待和完整响应会再次阻塞主线程。
- 在此期间，当前进程的事件循环完全停滞，无法响应其他用户的并发请求。

**应对方案**：在异步路由中全面采用异步方法 `ainvoke` 或流式方法 `astream`：
```python
# 推荐：非阻塞协程调用
response = await rag_chain.ainvoke(query)

# 推荐：流式逐字输出
async for chunk in rag_chain.astream(query):
    yield chunk
```
同时需确认所选用的向量数据库组件与底层客户端支持原生异步操作（如 VectorStore 实现了原生的 `asimilarity_search` 且底层网络客户端为非阻塞 I/O），避免底层因缺失异步支持而回退到线程池进行伪异步模拟。

### 陷阱 2：空召回与未捕获的外部异常
原生链式调用假设各个环节均正常返回。然而在真实网络环境下：
- **空召回**：当用户提问与知识库毫无重合时，检索器可能返回空列表 `[]`，拼接出的上下文为空字符串。模型在没有约束的情况下可能根据预训练知识编造内容。更多幻觉机制可参考 [RAG 为什么仍然会产生幻觉？](rag-hallucination.md)。
- **外部组件超时**：若向量数据库断连或响应超时，检索步骤抛出异常，会导致整条请求直接中断报错。

![空检索与异常注入下的 Fallback 降级防御流](/images/content-rebuild/499f87a390cef2b2/fig-6.png)

**应对方案**：在格式化函数中补充空召回兜底提示，并利用 LCEL 的 `.with_fallbacks()` 配置降级路径：
```python
def safe_format_docs(docs: List[Document]) -> str:
    if not docs:
        return "【未在知识库中找到相关参考依据，请告知用户该信息暂未收录】"
    return "\n\n".join(doc.page_content for doc in docs)

# 备用降级链：当主链发生网络异常等故障时介入
# 降级链入参需保持与主链一致（接收 query 字符串），故需先映射为包含 question 的字典
fallback_prompt = ChatPromptTemplate.from_template(
    "知识库连接异常，请礼貌告知用户当前检索服务不可用，并建议稍后重试。用户问题：{question}"
)
fallback_chain = (
    RunnableParallel({"question": RunnablePassthrough()})
    | fallback_prompt
    | llm
    | StrOutputParser()
)

# 挂载降级保障
robust_rag_chain = rag_chain.with_fallbacks([fallback_chain])
```

### 陷阱 3：缺乏调试可见性
使用 `|` 串联的执行链将中间状态封装在内部。线上如果出现“回答与文档不符”，仅凭最终的输出字符串很难断定问题出在文档切分被截断、检索阶段召回错位，还是模型没有遵循系统提示词。
- 在实际开发中，应利用回调机制（Callbacks）或接入链路追踪系统（如 OpenTelemetry、LangSmith），完整记录每次调用中实际召回的 `Document` 元数据与最终渲染出的完整 Prompt 文本。

---

## 6. 常见面试追问

### 追问 1：手写 LCEL 与官方 `create_retrieval_chain` 有什么区别？
LangChain 官方在 `langchain.chains` 中封装了 `create_retrieval_chain`。两者的核心区别在于**输出结构与溯源信息的保留**：
- 本文手写的 LCEL 链路通过 `format_docs` 提前将 `List[Document]` 转为了纯字符串，链的最终输出仅为纯文本字符串 `str`，原始的文档对象和来源元数据在此过程中丢失。
- `create_retrieval_chain` 内部封装了文档整合逻辑，其最终输出是一个字典：`{"input": str, "context": List[Document], "answer": str}`。这使得前端可以直接拿到检索到的原文档列表，提取 `metadata["source"]` 或页码并展示参考来源与引用标号，便于满足可追溯性要求。

### 追问 2：如果需要实现向量与 BM25 的混合检索，LangChain 的抽象层如何支持？
由于 LCEL 下游仅依赖统一的 `Retriever` 接口，引入混合检索无需修改下游的数据格式化和 Prompt 组装逻辑。可以直接使用 `EnsembleRetriever` 替代原有的单一向量检索器：

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 构建 BM25 关键词检索器与向量检索器
bm25_retriever = BM25Retriever.from_documents(raw_docs)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 使用互惠排名融合（RRF）合并多路结果
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# 下游 LCEL 链路完全无需变更，仅需替换检索器实例
rag_chain = (
    RunnableParallel({"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)
```
关于混合检索中融合算法的深入说明，可参考 [什么是 RAG 混合检索？](rag-hybrid-retrieval.md)。
