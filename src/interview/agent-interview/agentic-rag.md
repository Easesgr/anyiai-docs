---
title: Agentic RAG 是什么？
icon: mdi:database-search
tag:
  - Agent面试
  - Agentic RAG
  - 检索增强
---

![](/images/agent-interview/agentic-rag_cover.png)

# Agentic RAG 是什么？和普通 RAG 有什么区别？

这道题考的是你对 RAG 系统进化方向的理解。Agentic RAG 本质上就是给普通 RAG 加了个"大脑"，让它从被动响应变成主动决策。

---

## 一句话定义 Agentic RAG

先说清楚本质：**Agentic RAG = RAG + Agent 思维**

普通 RAG 像个图书馆管理员——你问什么，它找什么，找到就走。Agentic RAG 像个全能项目经理——接到任务会先思考"该找谁帮忙"、"资源够不够"、"结果对不对"。

一个被动，一个主动，这就是最核心的区别。

![对比图，左侧普通RAG标注"被动检索"，右侧Agentic RAG标注"主动决策"](/images/agent-interview/20260608_224214_924096_01.png)

---
<ReadMoreLock />

## 核心区别：线性流程 vs 循环流程

普通 RAG 是"检索→增强→生成"三步走，不可逆。Agentic RAG 是"思考→行动→观察→再思考"循环，能自我纠错和重试。

想象你要找一本书：
- 普通 RAG 只会去书架，拿了书就走
- Agentic RAG 会先想"这本书在哪个书架"、去拿了发现没有、换另一个书架试试、最后拿到才离开

**关键差异就是：普通 RAG 单程票，Agentic RAG 支持回头票。**

![流程图，普通RAG单向箭头"检索→增强→生成"，Agentic RAG环形循环"思考-行动-观察-再思考"](/images/agent-interview/20260608_224224_920128_01.png)

---

## 架构差异：从单兵作战到团队协作

普通 RAG 就一个检索组件，配个向量数据库，加个大模型，三件套打完收工。

Agentic RAG 架构复杂多了：
- **规划者（Orchestrator）**：分解任务、协调各方
- **多个专业 Agent**：分工执行、各管一摊
- **Memory 模块**：记住中间步骤
- **Planning 模块**：规划执行路径
- **支持并行执行**：多个 Agent 同时干活
- **跨数据源检索**：MCP 服务器连 N 个数据库

类比一下：普通 RAG 是"一个人干完所有活"，Agentic RAG 是"项目经理拆解任务 + 多个专家分工执行 + 各自汇报结果"。

![架构图展示Orchestrator、多Agent、Memory、Planning模块及MCP服务器连接多个数据源](/images/agent-interview/20260608_224217_314496_01.png)

---

## 选哪个——看任务复杂度

简单任务用普通 RAG：
- 查个定义
- 总结文档
- 回答固定问题

复杂任务用 Agentic RAG：
- 需要多步推理
- 跨多个数据源
- 对结果准确度要求高

选普通 RAG 就像打车去隔壁小区，快。选 Agentic RAG 就像出门办事前先规划路线，灵活但慢。

![表格对比图，左列普通RAG适用场景，右列Agentic RAG适用场景，标注各自优缺点](/images/agent-interview/20260608_224232_418048_01.png)

---

## 面试怎么答

**基础版（能拿到分）：**

"Agentic RAG 是具备 Agent 思维能力的 RAG 系统。普通 RAG 是线性流程——检索、增强、生成，三步走完就结束。Agentic RAG 是循环流程——思考、行动、观察、再思考，遇到问题能回头重试。架构上，普通 RAG 单兵作战，Agentic RAG 是规划者加多个 Agent 协作。简单问答用普通 RAG，复杂推理用 Agentic RAG。"

**加分版（能拿高分）：**

"从架构看，Agentic RAG 有 Orchestrator 做规划者，配合多个专业 Agent、Memory 和 Planning 模块，支持并行执行和跨数据源检索。从推理框架看，底层常用 ReAct——让 Agent 先推理再行动、观察结果再调整。现在生态上有 DSPy 做编程范式、LangChain 和 LlamaIndex 做 RAG 编排、CrewAI 和 Swarm 做 Multi-Agent 协作。趋势是 Multi-Agent RAG，一个 Agent 管检索、一个管推理、一个管验证，分工越来越细。"

---

## 一句话总结

Agentic RAG 就是让 RAG"活"起来——从被动查资料，变成主动想方案。

![生态图，展示DSPy、LangChain、LlamaIndex、CrewAI、Swarm等框架在Agentic RAG生态中的位置](/images/agent-interview/20260608_224232_542464_01.png)