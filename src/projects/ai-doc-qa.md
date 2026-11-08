---
title: 如何用 Python 构建单文件文档问答系统？轻量 RAG 最小可运行闭环
icon: mdi:file-document-outline
description: 从文档解析、滑动分块、内存向量检索到带约束 Prompt 生成，用 Python 构建单文件文档问答系统的最小运行闭环与验收测试。
---

大语言模型（LLM）的预训练权重无法包含你私有的本地文件。如果你有一份刚签署的《技术服务协议》，想询问“提前解除合同需要赔付多少违约金？”，模型要么因缺乏背景而拒绝回答，要么调用公开语料里的通用经验给出错误推测。

RAG（Retrieval-Augmented Generation，检索增强生成）通过以下方式解决该问题：在推理阶段，先从外部文档中检索出与提问最相关的几个片段，再将这些片段拼入提示词（Prompt）提供给模型，由模型基于真实片段生成回答。

![单文件文档问答系统的最小运行闭环数据流](/images/content-rebuild/832b3585039d754b/cover.png)

很多开发者在初学 RAG 时直接引入重量级框架与外置向量数据库，一旦出现检索跑偏或模型幻觉，难以快速定位是文本提取、分块策略、向量检索还是提示词约束出了问题。本文脱离重型组件，仅使用 Python 标准库、NumPy 矩阵运算和基础模型接口，带你手写一个可以在内存中完整运行的单文件文档问答原型。

---

## 一、系统输入输出定义与验收标准

在写代码之前，先界定系统的输入、输出、运行时状态与边界条件。

### 1. 输入与输出定义
- **输入**：
  1. 本地单文件路径（支持 `.txt` 或排版清晰的单栏 `.pdf`，建议字数在 10 万字以内）。
  2. 自然语言提问字符串（Query），例如 `"提前解除合同需要赔付多少违约金？"`。
- **输出**：
  1. 答案正文（Answer），基于参考片段生成的精准事实回复。
  2. 溯源引用（Sources），包含支撑该回答的原文分块、分块编号及检索相似度得分。
  3. 状态标识（Status），明确标记成功回答（`success`）或因未检索到相关内容而拦截（`rejected_by_threshold`）。

### 2. 运行时状态设计
为了让系统尽量精简，当前原型采用以下边界：
- **全内存索引**：分块文本与向量矩阵均保存在进程内存中，脚本停止即释放，不依赖磁盘持久化存储。
- **单次无状态问答**：不引入多轮对话记忆历史，每次提问都是独立的检索与生成。
- **单路稠密检索**：使用 Embedding 向量的余弦相似度匹配，暂不叠加关键词检索（如 BM25）或二次重排序（Reranking）。

| 阶段 | 输入 | 内部处理流 | 输出状态 |
| :--- | :--- | :--- | :--- |
| **索引期** | 本地文件路径 | 文本读取 $\to$ 乱码与控制符清洗 $\to$ 滑动窗口分块 $\to$ 批量向量化 | 内存分块列表 + NumPy 浮点矩阵 |
| **查询期** | 提问字符串 | Query 向量化 $\to$ 归一化点积 $\to$ Top-K 召回 $\to$ 相似度阈值判定 $\to$ 提示词组装 $\to$ LLM 生成 | 最终回答 + 引用片段 |

---

## 二、文档读取与清洗：规避格式乱码与排版断行

文档提取的文本质量直接决定了后续切分和向量匹配的效果。如果提取出的内容包含大量格式控制符或非正常的折行断词，文本的语义连贯性会被严重破坏。

![PDF 与文本提取中的字符清洗与跨行折叠机制](/images/content-rebuild/832b3585039d754b/fig-1.png)

常见文本污染主要来自两方面：
1. **编码冲突**：Windows 下生成的文本文件常为 GBK 编码，若强制以 UTF-8 读取会抛出解码异常。
2. **PDF 排版伪换行**：PDF 本质上是排版指令集合。提取出的纯文本经常包含页面结束符（如 `\x0c`）、因排版对齐产生的硬换行（单句话中间被强行换行）以及英文行末连字符（如 `imple-\nmentation`）。

我们实现一个带容错的读取和清洗函数：

```python
import re
from pathlib import Path

def load_and_clean_text(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {file_path}")
    
    raw_text = ""
    suffix = path.suffix.lower()
    
    # 1. 读取文本，兼顾编码容错
    if suffix == ".txt":
        try:
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_text = path.read_text(encoding="gbk", errors="ignore")
    elif suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        extracted_pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            extracted_pages.append(page_text)
        raw_text = "\n".join(extracted_pages)
    else:
        raise ValueError(f"暂不支持的文件格式: {suffix}")
    
    # 2. 清洗管道
    # 移除换页符与零宽字符
    cleaned = re.sub(r"[\x0c\u200b]", "", raw_text)
    # 合并行末英文连字符断词 (例: agree-\nment -> agreement)
    cleaned = re.sub(r"(\w+)-\n(\w+)", r"\1\2", cleaned)
    # 将非段落标记的单换行折叠为空格，保留双换行代表的段落结构
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
    # 压缩多余连续空格和制表符
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    
    return cleaned.strip()
```

---

## 三、滑动窗口分块：控制切分步长与重叠区

为什么不能把整篇几万字的文章直接传给向量模型生成一个向量？
1. **模型输入长度限制**：常见嵌入模型的上下文长度在 512 到 8192 个 Token 不等，超长文本会被直接截断。
2. **语义稀释**：向量嵌入是用固定长度的浮点数数组（如 1536 维）表示文本整体信息。文本越长，包含的子主题越多，单句细节在整体向量中的权重就被稀释得越严重，导致相似度匹配失真。

<ReadMoreLock />

为了兼顾细节与检索命中率，需要将长文本切分成较小的分块（Chunks）。但若进行绝对机械的按字数切断，原本完整的事实句可能被切分在两个不同的块中。例如：
> “……按照本合同总金额的百分之二十（20%）向守约方支付违约金。”

如果切分点正好落在“百分之二十”和“违约金”中间，前一块缺少赔付名目，后一块缺少比例数值，单独检索任何一块都无法完整回答问题。

![滑动窗口切分与重叠区对语义连续性的保留](/images/content-rebuild/832b3585039d754b/fig-2.png)

解决办法是在相邻分块之间保留一定的重叠区（Chunk Overlap）。滑动切分的步长公式为：
$$\text{Stride} = \text{Chunk Size} - \text{Chunk Overlap}$$

其中必须满足条件：$0 \le \text{Chunk Overlap} < \text{Chunk Size}$。

实现代码如下：

```python
from typing import List, Dict

def sliding_window_chunking(
    text: str, 
    chunk_size: int = 300, 
    chunk_overlap: int = 50
) -> List[Dict[str, any]]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    
    chunks = []
    stride = chunk_size - chunk_overlap
    text_len = len(text)
    start_idx = 0
    chunk_id = 0
    
    while start_idx < text_len:
        end_idx = min(start_idx + chunk_size, text_len)
        chunk_content = text[start_idx:end_idx].strip()
        
        if chunk_content:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_content,
                "start_char": start_idx,
                "end_char": end_idx
            })
            chunk_id += 1
            
        if end_idx == text_len:
            break
        start_idx += stride
        
    return chunks
```

在中文合同或技术规范场景下，初期可采用基准值 `chunk_size = 300` 字符，`chunk_overlap = 50` 字符（重叠比例约为 15%~20%）。重叠过大会导致召回结果内容重复，重叠过小则无法跨边界承接语意。

---

## 四、内存向量索引：归一化点积计算余弦相似度

将文本转为向量后，需要计算用户提问与各个分块之间的几何相似度。

两个向量 $\vec{u}$ 与 $\vec{v}$ 之间的夹角余弦值定义为：
$$\cos(\theta) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$

若在向量入库时，先对每个向量做 $L_2$ 范数归一化，即让 $\|\vec{v}\| = 1$（单位向量），同时查询向量也做归一化，那么分母恒等于 1，**余弦相似度即可直接通过点积（Dot Product）求得**：
$$\cos(\theta) = \vec{u} \cdot \vec{v} = \sum_{i=1}^d u_i v_i$$

![内存向量相似度计算与 Top-K 检索排序原理](/images/content-rebuild/832b3585039d754b/fig-3.png)

对于包含 $N$ 个分块的矩阵 $\mathbf{M} \in \mathbb{R}^{N \times d}$ 与提问向量 $\mathbf{q} \in \mathbb{R}^{1 \times d}$，批量相似度计算就是一次矩阵乘法：
$$\mathbf{S} = \mathbf{M} \mathbf{q}^T$$

在 Python 中借助 NumPy 的底层 BLAS 优化，单次乘法在微秒到毫秒级即可完成，对于万级以内的轻量检索，无需部署复杂的外部数据库。

```python
import numpy as np
from typing import List, Tuple

class MemoryVectorStore:
    def __init__(self):
        self.chunks: List[Dict[str, any]] = []
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
    
    def add_documents(self, chunks: List[Dict[str, any]], embeddings: List[List[float]]):
        if len(chunks) != len(embeddings):
            raise ValueError("文本块数量必须与向量数量保持一致")
            
        self.chunks.extend(chunks)
        raw_mat = np.array(embeddings, dtype=np.float32)
        
        # 对入库向量执行 L2 归一化
        norms = np.linalg.norm(raw_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # 避免除以 0
        normalized_mat = raw_mat / norms
        
        if self.embeddings.size == 0:
            self.embeddings = normalized_mat
        else:
            self.embeddings = np.vstack([self.embeddings, normalized_mat])
            
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[Dict[str, any], float]]:
        if self.embeddings.size == 0:
            return []
            
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
            
        # 归一化向量的点积即为余弦相似度
        scores = np.dot(self.embeddings, q)
        
        # 获取最高得分的前 k 个索引
        actual_k = min(top_k, len(self.chunks))
        # 使用 argpartition 取出前 k 大的无序索引，再对这 k 个数进行倒序排列，比全排序更快
        partitioned_indices = np.argpartition(scores, -actual_k)[-actual_k:]
        sorted_indices = partitioned_indices[np.argsort(-scores[partitioned_indices])]
        
        return [(self.chunks[idx], float(scores[idx])) for idx in sorted_indices]
```

---

## 五、防幻觉提示词：组装上下文与事实约束

检索到相关的原文片段后，需要把它们和用户的提问一起包装进一段结构化的提示词中。

![上下文组装与防幻觉事实约束的 Prompt 结构](/images/content-rebuild/832b3585039d754b/fig-4.png)

如果直接对大模型提问：“根据上面几段话回答问题”，模型往往会结合预训练中见过的百科知识来自由发挥，甚至把常识强加在合同约定之上。防范幻觉的提示词设计应包含三个要素：
1. **闭包约束**：要求模型仅能依赖给出的参考片段，严禁发散补充外部事实。
2. **证据引用**：要求模型输出答案时标注来源片段编号（如 `[片段1]`），强制模型在文本之间建立对应关系。
3. **明确兜底出口**：约定未提及时的统一拒答话术，避免模型在证据不足时强行猜测。

```python
PROMPT_TEMPLATE = """你是一个只依据参考文档回答问题的助手。
请阅读给出的参考文档片段，并回答用户的问题。

### 规则约束：
1. 只能根据下方的参考文档片段作答。如果片段中没有提及或无法直接推断，必须明确回答："根据提供的文档内容，无法回答该问题。"，严禁编造。
2. 语言保持简明客观，陈述事实时在句末标注对应的片段编号，例如：[片段1]。

### 参考文档片段：
{context}

### 用户提问：
{query}

### 回答："""

def build_prompt(query: str, retrieved_chunks: List[Tuple[Dict[str, any], float]]) -> str:
    context_items = []
    for idx, (chunk, score) in enumerate(retrieved_chunks):
        context_items.append(f"[片段{idx + 1}] (相似度: {score:.3f}):\n{chunk['text']}")
    
    context_str = "\n\n".join(context_items)
    return PROMPT_TEMPLATE.format(context=context_str, query=query)
```

---

## 六、双重兜底拒答与端到端运行验收

一个健壮的问答系统不能假设“检索永远能查到想要的东西”。当用户提问与文档毫不相干时，系统需要具备防御机制。

本原型设立两层防御：
1. **检索层硬阈值截断**：若最高相似度得分低于安全阈值（例如 `0.35`，具体需根据所用向量模型的实际分布校准），判定为完全未命中，直接拦截，**不触发 LLM 生成调用**，节省调用成本。
2. **生成层语义拒答**：若得分勉强跨过阈值，但片段内确实缺乏关键结论，依靠 Prompt 中的约束规则，让模型主动输出免责拒答。

![相似度阈值判定与兜底拒答的双重防御流程](/images/content-rebuild/832b3585039d754b/fig-5.png)

以下是端到端的完整调度器代码，集成了可离线运行的 Mock 模式与在线真实 API 模式：

```python
import os
from typing import Optional

# 环境变量控制：如无 API Key 可设置为 true 体验离线矩阵计算流程
USE_MOCK = os.getenv("USE_MOCK_LLM", "false").lower() == "true"

class RAGPipeline:
    def __init__(self, score_threshold: float = 0.35):
        self.vector_store = MemoryVectorStore()
        self.score_threshold = score_threshold
        
    def get_embedding(self, text: str) -> List[float]:
        if USE_MOCK:
            # 离线伪向量：利用字符频数哈希生成稠密向量，使包含相同词汇的文本具备正向相似度，保证离线测试闭环可运行
            vec = np.zeros(128, dtype=np.float32)
            for char in text:
                vec[ord(char) % 128] += 1.0
            return vec.tolist()
            
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return resp.data[0].embedding

    def index_document(self, file_path: str, chunk_size: int = 300, chunk_overlap: int = 50):
        text = load_and_clean_text(file_path)
        chunks = sliding_window_chunking(text, chunk_size, chunk_overlap)
        if not chunks:
            raise ValueError("文档解析结果为空，无法建库")
            
        # 顺序计算嵌入（工程中可批量打包调用 batch 接口）
        embeddings = [self.get_embedding(c["text"]) for c in chunks]
        self.vector_store.add_documents(chunks, embeddings)
        print(f"[索引就绪] 成功载入 {len(chunks)} 个分块。")

    def query(self, user_query: str, top_k: int = 3) -> Dict[str, any]:
        q_emb = self.get_embedding(user_query)
        results = self.vector_store.search(q_emb, top_k=top_k)
        
        # 第一道防线：最高分低于安全阈值直接拦截
        if not results or results[0][1] < self.score_threshold:
            return {
                "status": "rejected_by_threshold",
                "answer": "文档中未找到与该问题相关的内容。",
                "sources": []
            }
            
        valid_results = [r for r in results if r[1] >= self.score_threshold]
        prompt = build_prompt(user_query, valid_results)
        
        if USE_MOCK:
            answer = f"[Mock 模式] 命中 {len(valid_results)} 个片段，首片段得分: {valid_results[0][1]:.3f}"
        else:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content
            
        return {
            "status": "success",
            "answer": answer,
            "sources": valid_results
        }
```

### 运行测试与用例验收

我们以贯穿例子《技术服务协议》测试系统在“精准命中”与“未命中兜底”两种情境下的表现：

```python
if __name__ == "__main__":
    # 1. 模拟写入贯穿示例文件
    sample_contract = """
    技术服务协议第二条：本服务协议服务期为自 2024 年 1 月 1 日起至 2024 年 12 月 31 日止。
    违约责任部分：若任何一方在服务期内单方面提前解除合同，应提前三十日书面通知对方，
    并按照本合同总金额的百分之二十（20%）向守约方支付违约金。
    因不可抗力导致合同不能履行的，双方互不承担违约责任。
    """
    sample_file = Path("sample_contract.txt")
    sample_file.write_text(sample_contract, encoding="utf-8")
    
    try:
        pipeline = RAGPipeline(score_threshold=0.35)
        pipeline.index_document("sample_contract.txt", chunk_size=120, chunk_overlap=30)
        
        # 测试用例 A：命中相关事实
        print("\n--- 测试用例 A：提问合同违约金比例 ---")
        res_a = pipeline.query("提前解除合同需要赔付多少违约金？")
        print("状态:", res_a["status"])
        print("回答:", res_a["answer"])
        for doc, score in res_a["sources"]:
            print(f"  [引用得分: {score:.3f}] {doc['text'][:40]}...")
            
        # 测试用例 B：跨领域提问（触发拒答拦截）
        print("\n--- 测试用例 B：提问文档无关内容 ---")
        res_b = pipeline.query("火星基地居住舱的设计标准是什么？")
        print("状态:", res_b["status"])
        print("回答:", res_b["answer"])
        
    finally:
        if sample_file.exists():
            sample_file.unlink()
```

验收期望：
- 用例 A：状态应为 `success`。若配置了真实 API Key 调用在线大模型，回答将包含 `20%` 或 `百分之二十` 并标注 `[片段1]` 等溯源标号；若处于离线 Mock 模式，将输出命中片段数量与检索得分。
- 用例 B：跨领域提问得分未达阈值，直接被检索阈值过滤，状态为 `rejected_by_threshold`，不触发模型生成。

---

## 七、局限性与向更复杂系统的演进

本原型验证了 RAG 的核心数据流，代码清晰且无外部复杂依赖。但在真实生产环境中，还需要克服以下结构和性能瓶颈：

![单文件最小 RAG 原型的边界瓶颈与工程演进路径](/images/content-rebuild/832b3585039d754b/fig-6.png)

1. **双栏排版错位**：学术论文与研报常见双栏排版，纯字符流读取会横向跨栏提取文字，导致左右两栏内容交错拼合。此类复杂板式需要结合版面分析（Layout Analysis）算法进行块重组，进阶方案可参考 [/projects/ai-paper-reader.md](/projects/ai-paper-reader.md)。
2. **表格二维关系塌陷**：将 PDF 表格强行抽成纯文本后，表头与数据行会被拍扁为单行字符串，行列对应关系断裂，模型极易看错数值。对此需要引入 Markdown 表格转换或多模态结构解析，参见 [/projects/ai-knowledge-base.md](/projects/ai-knowledge-base.md)。
3. **内存与计算扩展性**：当前采用全内存存储和暴力矩阵点积，时间复杂度为 $O(N \cdot d)$（其中 $N$ 为分块数，$d$ 为向量维度）。在 $N < 2,000$（约对应几十万字文档）时耗时通常在数毫秒内；但当分块数量达到数十万甚至数百万时，内存消耗和单次搜索延迟将无法满足在线要求，必须切换到基于 HNSW 等索引的持久化向量数据库与搜索引擎，相关架构设计可阅读 [/projects/ai-search-engine.md](/projects/ai-search-engine.md) 与 [/projects/rag-project-experience.md](/projects/rag-project-experience.md)。
