---
title: 如何搭建多文档知识库？混合检索、分块与权限隔离实战
icon: mdi:database
description: 以内部运维故障库为贯穿案例，详解结构感知分块、密集与BM25混合检索、RRF融合排序、精排以及前置权限过滤的落地实现与评测。
---

在上一讲 [AI 文档问答系统](/projects/ai-doc-qa.md) 中，我们实现过针对单文档的轻量问答。但在面对包含成千上万篇技术规范、运维手册和工单记录的知识库时，简单的单文档向量化方法很快会遇到瓶颈：

1. **专业代号与错误码容易漏查**：嵌入向量模型（Embedding）擅长语义联想，但在面对特定产品型号（如 `CX-9200-TX`）或错误代码（如 `ERR_403_STORAGE`）时，容易因语义距离计算产生偏差，导致精确词无法召回。
2. **多租户与部门权限隔离**：不同部门（如基础设施运维组 `dept_infra` 与信息安全组 `dept_sec`）对文档切片的阅读权限不同。如果检索不带权限约束，就有泄漏未授权信息的风险。
3. **召回与排名的权衡**：只取前 3 个向量结果容易遗漏关键证据；但若盲目取前 50 个切片直接塞入大模型，不仅增加上下文开销，还容易引入无关干扰信息。

本篇以一个**企业内部 IT 运维知识库**为贯穿实例，从输入输出定义开始，手把手搭建一套包含结构分块、密集与 BM25 混合检索、互惠排序融合（RRF）、Cross-Encoder 精排和 Payload 前置过滤的检索系统，并通过命中率与 MRR 指标进行量化验收。

![企业知识库多文档检索系统整体架构](/images/content-rebuild/b0b752e62ec4b565/cover.png)

---

## 目标与系统输入输出定义

为了让系统行为可测试、可验收，我们先明确整个检索流水线的输入与输出规格：

* **系统输入**：
  * **查询输入**：用户自然语言提问，例如：`"CX-9200-TX 路由器固件升级报 ERR_403_STORAGE 怎么处理？"`
  * **鉴权上下文**：当前请求用户的身份信息，例如：`{"user_id": "u_1024", "tenant_id": "tenant_dev", "roles": ["dept_infra"]}`
  * **候选知识库**：包含 Markdown 格式的架构文档、DOCX 故障手册以及表格配置。
* **系统输出**：
  * **排序证据切片**：经过权限过滤、粗排融合与二次精排后排名前 3 的文本块，附带文档来源路径（如 `网络设备/CX-9000系列手册.md > 存储故障章节`）。

按照 [LlamaIndex RAG concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts/) 的检索阶段划分，我们将通过“粗排扩召回、精排提精度、索引层限权限”三步实现这一目标。

![企业级文档入库流水线与元数据提取流向](/images/content-rebuild/b0b752e62ec4b565/fig-1.png)

---

## 步骤一：文档结构解析与元数据抽取

多文档入库不能直接把文本抽成一段扁平无序的字符串，否则分块后大模型将丢失章节上下文。更多细节可参考 [文件解析后如何进入知识库？切分、清洗、元数据和索引怎么设计？](/frontier/document-parsing/parsing-to-knowledge-base.md)。

我们首先定义入库文档的标准化数据容器 `RawDocument`：

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class RawDocument:
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    headings: List[Dict[str, Any]] = field(default_factory=list)
```

在解析 Markdown 或 DOCX 时，程序需解析标题层级（H1、H2、H3），并生成标题树。解析出的元数据必须携带租户标识和访问控制列表（ACL）：

```python
def parse_markdown_with_metadata(doc_id: str, content: str, tenant_id: str, allowed_depts: List[str]) -> RawDocument:
    # 提取大纲并挂载访问控制属性
    metadata = {
        "doc_id": doc_id,
        "tenant_id": tenant_id,
        "acl_departments": allowed_depts,
        "is_public": len(allowed_depts) == 0
    }
    return RawDocument(doc_id=doc_id, text=content, metadata=metadata)
```

<ReadMoreLock />

---

## 步骤二：递归字符切分与面包屑上下文注入

如果直接按固定 500 字符强行截断，往往会把表格行、参数列表或排障步骤从中间切开。文本切分策略并不能保证 100% 的语义完整，但可以通过层级分隔符优先拆解，并在每个切片头部补充“面包屑导航（Breadcrumb）”来缓解语义碎片化问题。

```python
from typing import List

def recursive_structural_split(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    separators: List[str] = None
) -> List[str]:
    if separators is None:
        # 优先级：段落 -> 单行换行 -> 中文句号 -> 空格 -> 字符截断
        separators = ["\n\n", "\n", "。", " ", ""]
    
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    # 寻找当前文本中存在的最高优先级分隔符
    chosen_separator = ""
    for sep in separators:
        if sep == "" or sep in text:
            chosen_separator = sep
            break
            
    if chosen_separator == "":
        # 无可用分隔符时强行按滑动窗口截取
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
        
    splits = text.split(chosen_separator)
    chunks = []
    current_chunk = ""
    
    for split in splits:
        candidate = current_chunk + (chosen_separator if current_chunk else "") + split
        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # 如果单段内容依然超过 chunk_size，递归下沉到下一级分隔符
            if len(split) > chunk_size:
                next_separators = separators[separators.index(chosen_separator) + 1:]
                sub_chunks = recursive_structural_split(split, chunk_size, chunk_overlap, next_separators)
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = split
                
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks
```

切分后，系统将标题路径注入切片正文前侧：

```text
[来源: 网络设备手册 > CX-9000系列 > 存储错误代码]
当出现 ERR_403_STORAGE 报错时，通常表示内部 Flash 存储分区已写满。请进入诊断模式执行 storage clear-cache 命令清除日志。
```

这保证了即使切片被单独召回，模型也能清楚它所归属的硬件型号与功能模块。

![文档递归字符切分与面包屑层级绑定过程](/images/content-rebuild/b0b752e62ec4b565/fig-2.png)

---

## 步骤三：密集向量与稀疏 BM25 混合检索

选型细节可参考 [什么是向量数据库？在基于大模型的应用开发中，向量数据库主要解决什么问题？](/interview/rag-interview/vector-database.md)。在生产检索中，单靠任何一种检索方式都有明显盲区：

* **密集向量检索（Dense Vector）**：利用语义模型将文本映射为连续向量（如 768 或 1024 维）。用户输入“设备存储坏了怎么办”，能通过语义相关性找到“Flash 存储分区故障”，但不擅长区分极其相似的料号（如 `CX-9200-TX` 与 `CX-9200-RX`）。
* **稀疏关键词检索（Sparse BM25）**：基于词频（TF）与逆文档频率（IDF）打分。对 `ERR_403_STORAGE` 这类极具区分度且词频罕见的符号具备很高的字面命中能力，但无法匹配缺少字面重合的同义表达。

参考 [Qdrant hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/) 的典型工程实现，我们可以通过并行查询密集向量索引与倒排词索引，各取 Top-K 候选进入下一步融合。

![密集向量空间与稀疏BM25空间联合查询机制](/images/content-rebuild/b0b752e62ec4b565/fig-3.png)

---

## 步骤四：互惠排序融合（RRF）与 Cross-Encoder 二次精排

多路召回返回的结果打分机制不同：余弦相似度区间通常为 $[-1, 1]$ 或 $[0, 1]$，而 BM25 是一组无上限的正实数。两者数值范围不同，不能直接相加。

### 互惠排序融合（RRF）计算原理

互惠排序融合（Reciprocal Rank Fusion, RRF）直接依据候选切片在各路列表中的**名次（Rank）**计算得分，公式如下：

$$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

其中：
* $M$ 为检索路数集合（例如密集向量路、BM25 关键词路）。
* $r_m(d)$ 为文档切片 $d$ 在第 $m$ 路结果中的排位名次（从 1 开始计数）。若在某路未召回，则该项记为 0。
* $k$ 为平滑常数，常规默认值为 60。

#### 贯穿算例演示

针对查询 `"CX-9200-TX ERR_403_STORAGE 恢复"`，假设两路分别召回 Top-3 切片：
* **切片 A**：标题含 `CX-9200-TX` 的常规介绍（密集路排第 1，BM25 路排第 3）
* **切片 B**：包含 `ERR_403_STORAGE 修复指南`（密集路排第 2，BM25 路排第 1）
* **切片 C**：其他路由器的存储说明（密集路排第 3，BM25 路未出现）

取 $k = 60$ 进行融合评分推导：

$$\begin{aligned}
RRF(A) &= \frac{1}{60 + 1} + \frac{1}{60 + 3} = \frac{1}{61} + \frac{1}{63} \approx 0.01639 + 0.01587 = 0.03226 \\
RRF(B) &= \frac{1}{60 + 2} + \frac{1}{60 + 1} = \frac{1}{62} + \frac{1}{61} \approx 0.01613 + 0.01639 = 0.03252 \\
RRF(C) &= \frac{1}{60 + 3} + 0 = \frac{1}{63} \approx 0.01587
\end{aligned}$$

经过名次打分融合后，切片 $B$（综合得分 $0.03252$）超越了单路排第一的切片 $A$（得分 $0.03226$），成为粗排第一名。RRF 算法有效提携了在两路中均有较好表现的切片。

```python
from typing import List, Tuple
from collections import defaultdict

def reciprocal_rank_fusion(
    rank_lists: List[List[str]], 
    k: int = 60
) -> List[Tuple[str, float]]:
    scores = defaultdict(float)
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] += 1.0 / (k + rank)
            
    # 按综合得分降序排列
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results
```

### Cross-Encoder 联合注意力精排

双塔向量模型在检索时，Query 与 Document 是各自独立编码的，两者无法产生词对词的跨注意力（Cross-Attention）交互。

Cross-Encoder 将 `Query` 与 `Candidate Chunk` 拼接到一起（如 `[CLS] 查询文本 [SEP] 切片文本`）送入模型做全注意力打分。由于 Cross-Encoder 计算开销大、推理延迟高，工程上通常采用**两阶段方案**：
1. 双路检索 + RRF 粗排筛选出 Top-20 候选。
2. 将这 20 个切片送入轻量级 Cross-Encoder 模型精排，输出最相关的 Top-3 注入大模型上下文。

![多路召回、RRF融合与Cross-Encoder精排双阶段流程](/images/content-rebuild/b0b752e62ec4b565/fig-4.png)

---

## 步骤五：多租户与部门权限的前置过滤（Payload Pre-filtering）

企业知识库常需要按租户和部门隔离数据。常见错误做法是“后置过滤（Post-filtering）”：先通过向量检索召回 Top-20，再在代码里循环判断用户是否有权限读取。如果恰好排名前 20 的文档都属于保密文档或其它部门，过滤后用户拿到的结果就会变成空列表（出现 Top-K 塌缩）。

正确做法是使用数据库引擎支持的**前置过滤（Pre-filtering）**：在索引查找候选点时，直接把权限元数据加入检索过滤条件中。

以 Qdrant 的 Payload 过滤配置为例：

```python
# 检索时的过滤参数：租户必须匹配，且切片要么公开，要么包含用户的所属部门
search_filter = {
    "must": [
        {"key": "tenant_id", "match": {"value": "tenant_dev"}}
    ],
    "should": [
        {"key": "is_public", "match": {"value": True}},
        {"key": "acl_departments", "match": {"any": ["dept_infra"]}}
    ]
}
```

通过把权限过滤下推到检索阶段，系统在图索引查找或倒排遍历时只评估符合权限条件的切片，确保召回结果既符合权限范围，数量又稳定满足 Top-K 需求。

![向量数据库前置过滤与后置过滤机制对比](/images/content-rebuild/b0b752e62ec4b565/fig-5.png)

---

## 步骤六：检索质量验收与自动化评测（Hit Rate 与 MRR）

搭建完知识库流水线后，不能仅凭几条输入做主观测试，必须通过评测集进行量化评估。

### 评测指标定义

1. **命中率（Hit Rate @ K）**：在召回的前 $K$ 个结果中，只要包含了标注的标准答案切片即记为命中。
   $$Hit@K = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \mathbb{I}(\text{rank}_q \le K)$$
   其中 $\mathbb{I}(\cdot)$ 为指示函数，条件满足时为 1，否则为 0。

2. **平均倒数排名（Mean Reciprocal Rank, MRR @ K）**：衡量标准答案在排位列表中的位置。排在第 1 位贡献 $1/1$，排在第 3 位贡献 $1/3$。若在前 $K$ 位中未出现，则倒数排名计为 0。
   $$MRR@K = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{\mathbb{I}(\text{rank}_q \le K)}{\text{rank}_q}$$

### 验收算例测试

准备 3 条基准测试样本（$|Q| = 3$），设定窗口 $K = 3$ 进行测试验收：
* **Query 1**：真实相关切片排在第 1 位。$\text{rank}_1 = 1 \implies Hit@3 = 1$，$RR_1 = 1 / 1 = 1.0$
* **Query 2**：真实相关切片排在第 3 位。$\text{rank}_2 = 3 \implies Hit@3 = 1$，$RR_2 = 1 / 3 \approx 0.333$
* **Query 3**：真实相关切片排在第 5 位（超出 $K=3$）。$\text{rank}_3 > 3 \implies Hit@3 = 0$，$RR_3 = 0$

指标计算结果：

$$\begin{aligned}
Hit\ Rate@3 &= \frac{1 + 1 + 0}{3} = \frac{2}{3} \approx 0.667 \ (66.7\%) \\
MRR@3 &= \frac{1.0 + 0.333 + 0}{3} = \frac{1.333}{3} \approx 0.444
\end{aligned}$$

自动化测试脚本实现如下：

```python
from typing import List, Dict, Any, Callable

def evaluate_retrieval(
    test_cases: List[Dict[str, Any]], 
    search_pipeline_func: Callable[[str], List[str]], 
    top_k: int = 3
) -> Dict[str, float]:
    hits = 0
    reciprocal_ranks = []
    
    for case in test_cases:
        query = case["query"]
        expected_doc_id = case["expected_doc_id"]
        
        # 调用完整检索函数获取候选切片列表
        retrieved_ids = search_pipeline_func(query)[:top_k]
        
        if expected_doc_id in retrieved_ids:
            hits += 1
            rank = retrieved_ids.index(expected_doc_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
            
    total = len(test_cases)
    return {
        f"Hit_Rate@{top_k}": hits / total if total else 0.0,
        f"MRR@{top_k}": sum(reciprocal_ranks) / total if total else 0.0
    }
```

![检索评测基准与Hit Rate及MRR评分矩阵](/images/content-rebuild/b0b752e62ec4b565/fig-6.png)

---

## 总结与下一步

搭建多文档知识库的关键环节可以归结为四个工程要点：
1. **切分与上下文挂载**：递归拆分降低句子断裂概率，面包屑标题注入保证切片信息完整。
2. **多路混合检索**：密集向量负责语义发散，BM25 负责料号与错误码精确命中。
3. **两阶段粗精排**：通过 RRF 融合异构分数扩大候选集，再由 Cross-Encoder 联合注意力输出精选 Top-N。
4. **前置权限过滤**：在存储层联合 Payload 条件进行候选剪枝，防止后置过滤引起的 Top-K 塌缩。

在后续篇章中，我们将在 [PrepRAG — 详细实现文档](/projects/rag-implementation.md) 与 [AI 自动化工作流](/projects/ai-workflow-automation.md) 中，把该知识库检索流水线接入具备分支和循环的业务 Agent；有关生产部署的高可用与分片规划，可参考 [如何设计一个企业知识库问答系统？](/interview/ai-system-design-interview/knowledge-base-qa.md)。
