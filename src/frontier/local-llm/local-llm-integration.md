---
title: 本地大模型如何接入自己的应用？OpenAI-Compatible API、RAG 和工具调用怎么做？
icon: mdi:connection
tag:
  - 本地大模型
  - API 接入
  - RAG
  - 工具调用
---

![COVER](/images/local-llm/local-llm-integration_cover.png)

# 本地大模型如何接入自己的应用？OpenAI-Compatible API、RAG 和工具调用怎么做？

这道题考的是**实战落地能力**——能不能把大模型真正用起来。核心就三件事：用 LangChain 统一接口、用 OpenAI 兼容协议降低成本、用 RAG 和工具调用让模型真正"动手干活"。

下面一个个拆开讲。

---

## 板块1：一条代码切换模型 —— LangChain 的核心价值

先问你个问题：如果你的应用一开始接了 DeepSeek，后来客户要求换成 Claude，你打算怎么改？

答案应该是：**尽量不改代码**。

LangChain 就是干这个的。它把所有大模型封装成统一的 **Runnable 接口**，不管你接 DeepSeek、OpenAI 还是 Claude，调用方式一模一样：

```python

<ReadMoreLock />

## 接 DeepSeek
llm = ChatDeepSeek(model="deepseek-chat")

## 换成 Claude，一行就够
llm = ChatAnthropic(model="claude-3-5-sonnet")

## 调用方式完全一样
response = llm.invoke("你好")
```

GitHub 17 万星，说明这套思路确实被社区认可了。

它的设计哲学很清晰：**你的业务代码不绑定任何一家模型厂商**。换模型？改一行初始化代码，其他不动。

![LangChain 统一抽象层架构图，展示应用层、LangChain Runnable 接口层、DeepSeek/OpenAI/Claude 等模型 API 的层级关系](/images/local-llm/20260717_005659_976320_01.png)

---

## 板块2：两条接入路径 —— 官方包 vs 兼容接口

DeepSeek 怎么接？两种走法。

**路径一：DeepSeek 官方包**

```python
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key="your-key"
)
```

用 `langchain-deepseek` 包，原生支持思考链特性。适合明确要用 DeepSeek 的场景。

**路径二：OpenAI 兼容接口**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_base="https://api.deepseek.com/v1",
    api_key="your-key"
)
```

用 `langchain-openai`，但把 base URL 指向 DeepSeek 的 OpenAI 兼容端点。

这两种怎么选？

- 官方包像是**直飞航班**，功能全、班次少（只支持 DeepSeek）
- 兼容接口像是**中转航班**，灵活但多一步配置（可以接任何平台：硅基流动、七牛云、月之暗面……）

生产环境建议用**环境变量**配置 API key，不要硬编码。七牛云这种企业级供应商还带 SLA 保障，稳定性要求高的场景可以考虑。

![两种接入方式对比图，左侧 langchain-deepseek 直连 DeepSeek 官方，右侧 langchain-openai 连接多平台](/images/local-llm/20260717_005658_679232_01.png)

---

## 板块3：V3 能调工具，R1 不行 —— 模型能力陷阱

用 DeepSeek 的一定要小心这个坑：

- **DeepSeek-V3**：支持工具调用（Tool Calling）
- **DeepSeek-R1**：不支持工具调用

R1 是推理模型，擅长思考，但不会"动手"。你让它写代码可以，让它真正执行命令？做不到。

怎么解决？**LangGraph 双模型串联**。

思路很简单：让 R1 负责推理判断，V3 负责执行工具调用。

```python
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph

## R1 节点：推理
def reasoning_node(state):
    response = chat_r1.invoke([HumanMessage(content=state["question"])])
    return {"conclusion": response.content}

## V3 节点：执行
def execution_node(state):
    response = chat_v3.bind_tools(tools).invoke(
        [HumanMessage(content=f"根据结论执行：{state['conclusion']}")]
    )
    return {"result": response}

## 串联成图
graph = StateGraph(AgentState).add_sequence([reasoning_node, execution_node])
```

就像老司机（R1）负责判断方向，副驾驶（V3）负责踩油门打方向盘。

![LangGraph 双模型串联流程图，展示 R1 推理节点 → 结论 → V3 节点 → 工具调用的完整链路](/images/local-llm/20260717_005709_974848_01.png)

---

## 板块4：RAG + 工具调用 —— 让大模型用上你的专属知识

最后讲怎么让模型"用你的数据、替你干活"。

### RAG：给模型一本专业词典

你的私有文档、数据库知识，大模型本身不知道。你得主动告诉它。

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

## 文档向量化
loader = WebBaseLoader(["你的文档链接"])
docs = loader.load()
splits = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())

## 检索相关上下文
retriever = vectorstore.as_retriever()
docs = retriever.get_relevant_documents("用户问题")
```

检索出来的内容塞进 prompt，大模型就能基于你的知识回答了。

### 工具调用：给模型一双能操作的手

定义一个工具，大模型就能调用：

```python
from langchain_core.tools import tool

@tool
def查天气(城市: str) -> str:
    """查询城市天气"""
    return f"{城市}今天晴，25度"
```

LangChain 自动把这个函数转成模型能理解的 schema，你调用模型时把工具传进去：

```python
response = llm.bind_tools([查天气]).invoke("北京天气怎么样")
## 模型会自动决定调用查天气工具
```

![RAG + 工具调用完整流程图，展示文档 → 向量化 → 检索 → 生成 → 工具执行的闭环](/images/local-llm/20260717_005722_968448_01.png)

把 RAG 和工具调用结合起来，就是完整的 Agent 架构：

1. 用户提问 → RAG 检索相关知识
2. 大模型思考 → 决定是否需要调用工具
3. 工具执行 → 返回结果
4. 模型整合 → 最终回答

![工具调用接口标准化示意图，展示 @tool 装饰器如何被 LangChain 转换为模型可识别的工具 schema](/images/local-llm/20260717_005721_747264_01.png)

---

## 面试怎么答

**基础版**（能说清楚基本路径）：

> 本地大模型接入应用，我用 LangChain 做统一封装。切换模型只需改一行初始化代码，不影响业务逻辑。接入方式有两种：DeepSeek 官方包 langchain-deepseek 原生支持思考链，或者用 langchain-openai 的 OpenAI 兼容接口接任何平台。另外要注意 DeepSeek-V3 支持工具调用，但 R1 不支持，需要用 LangGraph 把 R1 的推理结论传给 V3 执行。

**加分版**（能说清楚架构和权衡）：

> 核心是三层分离：业务层不绑定模型，通过 LangChain 的 Runnable 接口统一调用。接入层按需选择——DeepSeek 官方包功能全，OpenAI 兼容接口更灵活，可接硅基流动、七牛云等平台。生产环境建议环境变量配置，带 SLA 的供应商更稳定。工具调用这块，DeepSeek-R1 不支持 function calling，解决思路是用 LangGraph 把 R1 当推理引擎、V3 当执行引擎串联。RAG 方面用向量数据库检索私有知识，配合工具调用实现完整 Agent 能力。

---

## 一句话总结

**LangChain 统一接口 + OpenAI 兼容协议降低成本 + RAG 补知识 + 工具调用让模型动手干活**——这就是本地大模型接入应用的标准姿势。