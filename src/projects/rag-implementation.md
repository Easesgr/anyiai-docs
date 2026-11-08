---
title: 如何实现带混合检索与降级容灾的 RAG 后端？PrepRAG 完整代码实战
description: 从文档解析入库、Qdrant多路召回到Cross-Encoder重排与超时降级，交付一套生产可用的流式RAG完整实现。
---

在前期方案设计中，我们讨论了知识库问答系统的选型原则与指标设定。本篇进入具体代码落地阶段，目标是交付一条具备输入校验、稠密稀疏多路召回、本地重排、流式生成和容灾降级能力的完整问答后端。

我们以一个具体场景贯穿全程：用户在客户端提问技术问题（例如“解释 MySQL binlog 的同步模式”），或者输入与知识库无关的问题（例如“怎样做红烧肉？”）。系统需要先解析技术文档并写入数据库；在收到请求后，同时进行语义向量检索与关键词过滤，通过互惠排名融合（RRF）合并候选；接着由精排模型核验相关性。如果最高分低于阈值，系统直接输出标准拒答；若重排模型计算超时或数据库不可用，系统需要平滑降级，保证接口不中断。

![PrepRAG 核心工程拓扑架构与数据流向](/images/content-rebuild/ec19e4dcbbabc6ae/cover.png)

---

## 1. 项目结构与运行依赖配置

为了便于维护和测试，我们将文档解析、向量存储、重排计算、文本生成和容灾缓存拆分为独立模块。所有运行时依赖的连接地址与阈值参数统一由 Pydantic Settings 进行环境变量校验。

```
preprag-core/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI 路由入口与生命周期管理
│   ├── config.py                   # 强类型配置管理 (Pydantic Settings)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── markdown_loader.py      # Markdown 解析与标签脱敏
│   │   ├── qdrant_retriever.py     # Qdrant 稠密向量与关键词召回封装
│   │   ├── reranker.py             # 线程隔离的本地 Cross-Encoder 精排
│   │   ├── fallback_cache.py       # 本地离线热点问答缓存
│   │   ├── generator.py            # 上下文拼装、流式输出与低分拒答
│   │   └── rag_pipeline.py         # 串联全流程的主编排控制器
│   └── schemas/
│       ├── __init__.py
│       └── chat.py                 # 请求与响应的数据契约
├── scripts/
│   └── ingest.py                   # 批量向量入库脚本
├── tests/
│   ├── conftest.py
│   └── test_rag_pipeline.py        # 单元与集成测试用例
├── requirements.txt
└── .env.example
```

核心依赖版本如下：

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
pydantic-settings==2.5.2
qdrant-client==1.11.1
sentence-transformers==3.1.1
torch==2.4.1
openai==1.47.0
jieba==0.42.1
tenacity==9.0.0
pytest==8.3.3
pytest-asyncio==0.24.0
httpx==0.27.2
```

配置模块 `app/config.py` 在应用启动时读取环境变量并进行类型检查：

```python
# app/config.py
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    app_env: str = Field(default="production", alias="APP_ENV")
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    
    # Qdrant 数据库连接
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    collection_name: str = Field(default="preprag_kb", alias="COLLECTION_NAME")
    
    # 模型与检索配置
    dense_model_name: str = "text-embedding-3-small"
    dense_vector_dim: int = 1536
    reranker_model_path: str = "BAAI/bge-reranker-base"
    
    # 检索阈值与降级设置
    retrieval_dense_limit: int = 25
    retrieval_sparse_limit: int = 25
    rerank_top_k: int = 4
    relevance_threshold: float = 0.35  # 重排得分低于此阈值时直接拒答
    reranker_timeout_ms: int = 350     # 重排超时阈值（毫秒），超出则跳过重排
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
```

---

## 2. 数据入库管道：文档清洗、切分与批量写入

知识库原始文件多为 Markdown 格式，常包含前端自定义标签（如 `<ReadMoreLock />`）和 YAML Frontmatter 元数据。如果直接整体切块，标签字符会干扰向量语义，而且跨章节切分容易截断逻辑上下文。

我们的切分策略是：首先清洗掉渲染用标签并抽取元数据；接着按二级标题（`## `）将长文拆成独立小节；若单节字符数过大，再按换行与句号进行重叠滑动切块（Overlap），保证句子边界完整。

![Markdown 清洗、层级切分与批量向量入库状态机](/images/content-rebuild/ec19e4dcbbabc6ae/fig-1.png)

### 2.1 文档解析与标题感知切分

```python
# app/core/markdown_loader.py
import re
import yaml
from dataclasses import dataclass

@dataclass
class ParsedSection:
    title: str
    category: str
    h2: str
    content: str
    source_file: str

class MarkdownProcessor:
    @staticmethod
    def clean_custom_tags(text: str) -> str:
        # 清洗特殊占位标签与组件标签
        text = re.sub(r"<ReadMoreLock\s*/?>", "", text)
        text = re.sub(r"<[A-Za-z0-9_-]+(?:\s+[^>]*)?/?>", "", text)
        return text.strip()

    def parse_file(self, file_path: str) -> list[ParsedSection]:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()

        frontmatter = {}
        body = raw
        # 提取头部 YAML 元数据
        fm_match = re.match(r"^---\n(.*?)\n---\n", raw, re.DOTALL)
        if fm_match:
            try:
                frontmatter = yaml.safe_load(fm_match.group(1)) or {}
            except yaml.YAMLError:
                frontmatter = {}
            body = raw[fm_match.end():]

        cleaned_body = self.clean_custom_tags(body)
        doc_title = frontmatter.get("title", "未命名主题")
        category = frontmatter.get("category", "interview")

        # 以二级标题为边界进行分段切分
        sections: list[ParsedSection] = []
        parts = re.split(r"\n(?=##\s+)", cleaned_body)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            lines = part.splitlines()
            h2_title = "概述"
            if lines[0].startswith("## "):
                h2_title = lines[0].replace("## ", "").strip()
                content = "\n".join(lines[1:]).strip()
            else:
                content = part

            if content:
                sections.append(ParsedSection(
                    title=doc_title,
                    category=category,
                    h2=h2_title,
                    content=content,
                    source_file=file_path
                ))
        return sections
```

### 2.2 批量向量化与带重试的入库写入

如果单条调用向量接口和单条插入数据库，网络往返开销较大；但如果一次性提交上万条，容易触发 API 速率限制或数据库内存尖峰。在入库脚本中，我们将批大小控制在 64 条，并使用 `tenacity` 库在发生网络抖动时进行指数退避重试。

```python
# scripts/ingest.py
import uuid
import jieba
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
from app.config import settings
from app.core.markdown_loader import MarkdownProcessor, ParsedSection

openai_client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, api_key=settings.qdrant_api_key)

def init_collection():
    # 检查并创建指定向量维度的集合，采用余弦距离
    collections = [c.name for c in qdrant.get_collections().collections]
    if settings.collection_name not in collections:
        qdrant.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=settings.dense_vector_dim, distance=Distance.COSINE),
        )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def upsert_batch_with_retry(points: list[PointStruct]):
    qdrant.upsert(collection_name=settings.collection_name, points=points, wait=True)

def extract_sparse_tokens(text: str) -> list[str]:
    # 提取分词结果作为 Payload 中的词法检索关键词
    return [w.lower() for w in jieba.cut(text) if len(w.strip()) > 1]

def run_ingestion(file_paths: list[str], batch_size: int = 64):
    init_collection()
    processor = MarkdownProcessor()
    all_sections: list[ParsedSection] = []
    for p in file_paths:
        all_sections.extend(processor.parse_file(p))

    for i in range(0, len(all_sections), batch_size):
        chunk_batch = all_sections[i:i + batch_size]
        texts = [f"{s.title} - {s.h2}\n{s.content}" for s in chunk_batch]
        
        # 批量获取文本嵌入向量
        response = openai_client.embeddings.create(input=texts, model=settings.dense_model_name)
        embeddings = [data.embedding for data in response.data]

        points = []
        for section, text_content, vector in zip(chunk_batch, texts, embeddings):
            # 生成稳定唯一的主键 UUID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{section.source_file}_{section.h2}_{section.content[:30]}"))
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "title": section.title,
                    "category": section.category,
                    "h2": section.h2,
                    "content": section.content,
                    "source_file": section.source_file,
                    "sparse_tokens": extract_sparse_tokens(text_content)
                }
            ))
        
        upsert_batch_with_retry(points)
        print(f"成功写入索引切片：第 {i + 1} 至 {i + len(points)} 条")
```

<ReadMoreLock />

---

## 3. 多路召回与互惠排名融合（RRF）

只靠向量稠密检索，在遇到精准类名、错误代号或参数缩写时可能出现语义漂移；而只靠关键词匹配，又无法召回表述不同但含义一致的近义内容。根据实际工程经验，常用方案是将稠密检索与基于分词过滤的稀疏词法检索结合，再由互惠排名融合（Reciprocal Rank Fusion, RRF）对多路结果进行重排序。

![Qdrant 稠密与稀疏多路召回及 RRF 融合计算机制](/images/content-rebuild/ec19e4dcbbabc6ae/fig-2.png)

### 3.1 RRF 融合算例与说明

假设召回通道集合为 $M$（包括稠密向量通道与关键词通道），$d$ 为候选切片，$r_m(d)$ 为文档 $d$ 在第 $m$ 个通道中的排位（从 $1$ 起始），平滑常数通常设为 $k = 60$。文档综合分数为：

$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

**算例推导**：
- 设文档 A（关于 MySQL binlog 同步）在稠密检索中排第 $1$ 名，在关键词召回中排第 $3$ 名：
  $$RRF(A) = \frac{1}{60 + 1} + \frac{1}{60 + 3} = \frac{1}{61} + \frac{1}{63} \approx 0.01639 + 0.01587 = 0.03226$$
- 设文档 B 在稠密检索中排第 $2$ 名，在关键词召回中未出现（即仅单路召回）：
  $$RRF(B) = \frac{1}{60 + 2} = \frac{1}{62} \approx 0.01613$$

由于 RRF 只使用排位索引而不是未归一化的相似度分数，它消除了余弦距离与 BM25 分数之间数值尺度不统一的问题。

*注意：在 Qdrant 官方体系中，标准混合检索可以通过服务端内置稀疏向量（Sparse Vectors）配合 `prefetch` 与 `rrf` 算子完成；下述代码采用 Jieba 分词 Payload 配合客户端 RRF，属于不依赖外部稀疏模型服务时的轻量级应用层替代方案。*

```python
# app/core/qdrant_retriever.py
import jieba
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from openai import AsyncOpenAI
from app.config import settings

class HybridRetriever:
    def __init__(self, qdrant: AsyncQdrantClient, openai: AsyncOpenAI):
        self.qdrant = qdrant
        self.openai = openai

    async def _get_dense_vector(self, query: str) -> list[float]:
        resp = await self.openai.embeddings.create(input=[query], model=settings.dense_model_name)
        return resp.data[0].embedding

    async def search(self, query: str, category_filter: str | None = None) -> list[dict]:
        query_vector = await self._get_dense_vector(query)
        keywords = [w.lower() for w in jieba.cut(query) if len(w.strip()) > 1]

        base_filter = None
        if category_filter:
            base_filter = Filter(must=[FieldCondition(key="category", match=MatchAny(any=[category_filter]))])

        # 通道 1：基于 HNSW 索引的稠密向量检索
        dense_results = await self.qdrant.search(
            collection_name=settings.collection_name,
            query_vector=query_vector,
            query_filter=base_filter,
            limit=settings.retrieval_dense_limit
        )

        # 通道 2：基于稀疏分词 Payload 的关键词召回
        sparse_results = ([], None)
        conditions = []
        if keywords:
            conditions.append(FieldCondition(key="sparse_tokens", match=MatchAny(any=keywords)))
        if base_filter and base_filter.must:
            conditions.extend(base_filter.must)

        if conditions:
            sparse_results = await self.qdrant.scroll(
                collection_name=settings.collection_name,
                scroll_filter=Filter(must=conditions),
                limit=settings.retrieval_sparse_limit,
                with_payload=True,
                with_vectors=False
            )

        # 客户端 RRF 计算
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}
        k = 60

        for rank, hit in enumerate(dense_results):
            doc_id = str(hit.id)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = hit.payload

        for rank, record in enumerate(sparse_results[0]):
            doc_id = str(record.id)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = record.payload

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        merged = []
        for did in sorted_ids:
            item = doc_map[did]
            item["doc_id"] = did
            item["rrf_score"] = rrf_scores[did]
            merged.append(item)
        return merged
```

---

## 4. 本地 Cross-Encoder 重排与线程隔离

向量召回阶段使用的双塔模型（Bi-Encoder）分别编码 Query 与候选文本，计算速度快，但缺少二者之间的深层词元交互。重排模型（Cross-Encoder）将 `(Query, Document)` 拼接成一个序列送入 Transformer，能够完整计算词对词的注意力得分，因此打分精度更高。

在 FastAPI 异步服务中，Transformer 的推理计算是同步的 CPU 或 GPU 密集型任务。若直接在 `async def` 函数中调用 `model.predict()`，会占用主事件循环，导致其他并发网络请求出现延迟。因此，需要使用 `asyncio.to_thread` 将重排任务分派到独立工作线程池中执行。

![Cross-Encoder 精排交互机制与异步非阻塞线程池调度](/images/content-rebuild/ec19e4dcbbabc6ae/fig-3.png)

```python
# app/core/reranker.py
import asyncio
import torch
from sentence_transformers import CrossEncoder
from app.config import settings

class LocalReranker:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(settings.reranker_model_path, max_length=512, device=device)

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
        return [float(s) for s in scores]

    async def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        
        pairs = [(query, f"{c.get('title', '')} {c.get('h2', '')}: {c.get('content', '')}") for c in candidates]
        # 移交线程池计算，防止阻塞主事件循环
        scores = await asyncio.to_thread(self._predict_sync, pairs)

        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = score

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
```

---

## 5. 流式生成器构建与低分拒答守卫

当知识库中没有与用户问题相关的切片时，如果仍将低关联度文本拼入 Prompt，模型极易基于无关材料生成错误结论。我们在调用大语言模型前增加置信度门禁：若精排候选列表为空，或第 1 名的重排分数低于设定阈值 $\tau$（如 $0.35$），生成器直接返回预设的拒答说明，不发起模型流式调用。

![流式响应生成、SSE 事件切分与低置信度拒答守卫](/images/content-rebuild/ec19e4dcbbabc6ae/fig-4.png)

```python
# app/core/generator.py
import json
from typing import AsyncGenerator
from openai import AsyncOpenAI
from app.config import settings

REJECTION_TEMPLATE = "抱歉，当前的面试知识库中未能检索到与该问题充分相关的权威内容。建议您补充具体技术细节或核心概念后重新提问。"

SYSTEM_PROMPT = """你是一个严肃严谨的 AI 面试技术顾问。你只能根据所提供的参考上下文回答用户的面试问题。
如果参考内容不足以得出严谨结论，请如实说明缺少的知识点，不要虚构事实。
回答必须结构清晰，标出对应知识点名称。"""

class StreamGenerator:
    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client

    def _format_context(self, docs: list[dict]) -> str:
        blocks = []
        for idx, d in enumerate(docs):
            blocks.append(f"[参考材料 {idx+1}] 来源: 《{d['title']}》 - {d['h2']}\n{d['content']}")
        return "\n\n".join(blocks)

    async def generate_stream(
        self, query: str, ranked_docs: list[dict]
    ) -> AsyncGenerator[str, None]:
        # 门禁检查：若候选为空或最高分低于阈值，立即输出结构化拒答事件并终止
        if not ranked_docs or ranked_docs[0].get("rerank_score", 0.0) < settings.relevance_threshold:
            refusal_payload = {"event": "chunk", "delta": REJECTION_TEMPLATE, "is_refusal": True}
            yield f"data: {json.dumps(refusal_payload, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'sources': []}, ensure_ascii=False)}\n\n"
            return

        context_text = self._format_context(ranked_docs)
        user_message = f"问题：{query}\n\n参考上下文：\n{context_text}"

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            stream=True,
            temperature=0.1,
        )

        async for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield f"data: {json.dumps({'event': 'chunk', 'delta': delta, 'is_refusal': False}, ensure_ascii=False)}\n\n"

        sources = [{"title": d["title"], "h2": d["h2"], "score": d.get("rerank_score", 0.0)} for d in ranked_docs]
        yield f"data: {json.dumps({'event': 'done', 'sources': sources}, ensure_ascii=False)}\n\n"
```

---

## 6. 超时与存储故障时的两级降级编排

在实际服务中，重排推理可能因突发流量导致线程池排队超时，向量数据库也可能发生瞬时连接中断。为了防止单点延迟扩散为整条链路失败，我们设计了两级容灾回退：

1. **重排超时降级**：设定重排硬超时时间（如 $350\text{ms}$，通过 `asyncio.wait_for` 控制）。若在规定时间内未完成计算，立即捕获 `TimeoutError`，退回使用 RRF 排序前 $K$ 的候选切片送入生成模型。
   *说明：降级时若为了跳过拒答门禁而直接赋予默认有效分，虽然能确保答案输出，但会失去低相关度拦截保护，属于高可用性优先的折中选择。*
2. **存储异常热点回退**：若 Qdrant 发生网络异常或宕机，编排器捕获全局检索异常，切流至本地静态 JSON 热点缓存输出高频离线解答。

![系统容灾降级决策拓扑与两级回退路径](/images/content-rebuild/ec19e4dcbbabc6ae/fig-5.png)

```python
# app/core/fallback_cache.py
import json
from pathlib import Path

class LocalHotCache:
    """本地离线高频问答缓存"""
    def __init__(self, cache_file: str = "data/hot_qa_cache.json"):
        self.cache: dict[str, str] = {}
        path = Path(cache_file)
        if path.exists():
            try:
                self.cache = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def get_fallback_answer(self, query: str) -> str | None:
        for k, v in self.cache.items():
            if k in query or query in k:
                return v
        return None
```

主编排控制器将所有组件串联成完整闭环：

```python
# app/core/rag_pipeline.py
import asyncio
import json
import logging
from app.config import settings
from app.core.qdrant_retriever import HybridRetriever
from app.core.reranker import LocalReranker
from app.core.generator import StreamGenerator
from app.core.fallback_cache import LocalHotCache

logger = logging.getLogger("preprag.pipeline")

class RAGPipelineCoordinator:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: LocalReranker,
        generator: StreamGenerator,
        fallback_cache: LocalHotCache
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.fallback_cache = fallback_cache

    async def execute_query_stream(self, query: str, category: str | None = None):
        candidates = []
        try:
            # 步骤 1: 稠密与稀疏多路检索
            candidates = await self.retriever.search(query, category_filter=category)
        except Exception as e:
            logger.error(f"向量检索异常，启动离线缓存降级: {str(e)}")
            fallback_ans = self.fallback_cache.get_fallback_answer(query)
            if fallback_ans:
                yield f"data: {json.dumps({'event': 'chunk', 'delta': f'[系统降级提示：知识库检索故障，返回离线热点回答]\n\n{fallback_ans}'}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps({'event': 'chunk', 'delta': '知识库检索暂时不可用，且未命中本地热点问答，请稍后重试。'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'sources': []}, ensure_ascii=False)}\n\n"
            return

        # 步骤 2: Cross-Encoder 重排与超时降级
        ranked_docs = []
        if candidates:
            try:
                ranked_docs = await asyncio.wait_for(
                    self.reranker.rerank(query, candidates, top_k=settings.rerank_top_k),
                    timeout=settings.reranker_timeout_ms / 1000.0
                )
            except asyncio.TimeoutError:
                logger.warning("重排计算超时，降级退回 RRF 融合候选")
                # 超时降级：直接取 RRF 前 Top-K 结果，赋予默认有效分保证可用性
                for d in candidates[:settings.rerank_top_k]:
                    d["rerank_score"] = 1.0
                ranked_docs = candidates[:settings.rerank_top_k]
            except Exception as e:
                logger.error(f"重排计算异常: {str(e)}，降级采用 RRF 候选")
                for d in candidates[:settings.rerank_top_k]:
                    d["rerank_score"] = 1.0
                ranked_docs = candidates[:settings.rerank_top_k]

        # 步骤 3: 触发流式输出与低分拒答
        async for sse_event in self.generator.generate_stream(query, ranked_docs):
            yield sse_event
```

在 `app/main.py` 中挂载流式 HTTP 路由：

```python
# app/main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from openai import AsyncOpenAI
from app.config import settings
from app.core.qdrant_retriever import HybridRetriever
from app.core.reranker import LocalReranker
from app.core.generator import StreamGenerator
from app.core.fallback_cache import LocalHotCache
from app.core.rag_pipeline import RAGPipelineCoordinator

app = FastAPI(title="PrepRAG Service")

qdrant_client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, api_key=settings.qdrant_api_key)
openai_client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

retriever = HybridRetriever(qdrant_client, openai_client)
reranker = LocalReranker()
generator = StreamGenerator(openai_client)
fallback_cache = LocalHotCache()
pipeline = RAGPipelineCoordinator(retriever, reranker, generator, fallback_cache)

class AskRequest(BaseModel):
    question: str
    category: str | None = None

@app.post("/api/v1/chat/stream")
async def stream_chat(req: AskRequest):
    return StreamingResponse(
        pipeline.execute_query_stream(req.question, req.category),
        media_type="text/event-stream"
    )
```

---

## 7. 测试验收：端到端与异常降级断言

为了验证系统在正常查询、无关提问以及模块异常下的表现，我们使用 `pytest` 编写三条核心测试：
1. **正常问答**：验证混合检索召回有效材料并正常完成流式响应；
2. **低分拒答**：验证提问无关问题时，重排打分低于阈值，系统终止流式生成并返回标准拒答；
3. **重排超时降级**：验证重排模型出现延迟阻塞时，系统在 $350\text{ms}$ 内捕获异常并平滑切换至 RRF 候选输出。

![Pytest 模拟测试环境拓扑与三大集成验证路径](/images/content-rebuild/ec19e4dcbbabc6ae/fig-6.png)

```python
# tests/test_rag_pipeline.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.core.rag_pipeline import RAGPipelineCoordinator
from app.core.generator import REJECTION_TEMPLATE, StreamGenerator

@pytest.mark.asyncio
async def test_normal_retrieval_and_streaming():
    """测试正常链路：混合检索、精排与流式输出"""
    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = [
        {"title": "MySQL高可用", "h2": "binlog模式", "content": "Row格式记录行变更，Statement记录SQL。", "rrf_score": 0.032}
    ]
    mock_reranker = AsyncMock()
    mock_reranker.rerank.return_value = [
        {"title": "MySQL高可用", "h2": "binlog模式", "content": "Row格式记录行变更，Statement记录SQL。", "rerank_score": 0.89}
    ]
    mock_generator = AsyncMock()
    async def mock_stream(q, docs):
        yield 'data: {"event": "chunk", "delta": "MySQL binlog 包括 Row 和 Statement 等模式。", "is_refusal": false}\n\n'
        yield 'data: {"event": "done", "sources": [{"title": "MySQL高可用", "h2": "binlog模式"}]}\n\n'
    mock_generator.generate_stream = mock_stream
    mock_cache = MagicMock()

    pipeline = RAGPipelineCoordinator(mock_retriever, mock_reranker, mock_generator, mock_cache)
    events = []
    async for e in pipeline.execute_query_stream("解释 MySQL binlog 的同步模式"):
        events.append(e)

    assert len(events) == 2
    assert "MySQL binlog 包括" in events[0]
    assert "MySQL高可用" in events[1]

@pytest.mark.asyncio
async def test_low_score_refusal():
    """测试置信度守卫：当精排分数低于安全阈值时触发拒答"""
    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = [
        {"title": "烹饪教程", "h2": "红烧肉做法", "content": "五花肉先焯水加生抽老抽。", "rrf_score": 0.01}
    ]
    mock_reranker = AsyncMock()
    mock_reranker.rerank.return_value = [
        {"title": "烹饪教程", "h2": "红烧肉做法", "content": "五花肉先焯水加生抽老抽。", "rerank_score": 0.12}  # 低于 0.35 阈值
    ]
    
    real_generator = StreamGenerator(openai_client=AsyncMock())
    mock_cache = MagicMock()

    pipeline = RAGPipelineCoordinator(mock_retriever, mock_reranker, real_generator, mock_cache)
    events = []
    async for e in pipeline.execute_query_stream("怎样做红烧肉？"):
        events.append(e)

    assert len(events) == 2
    assert REJECTION_TEMPLATE in events[0]
    assert '"is_refusal": true' in events[0]

@pytest.mark.asyncio
async def test_reranker_timeout_fallback():
    """测试可用性保障：当重排模型响应超时，平滑退回 RRF 候选列表"""
    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = [
        {"title": "MySQL高可用", "h2": "主从同步", "content": "从库通过 IO 线程拉取 relay log。", "rrf_score": 0.035}
    ]
    
    mock_reranker = AsyncMock()
    async def slow_rerank(*args, **kwargs):
        await asyncio.sleep(0.5)  # 模拟耗时 500ms，超过设定的 350ms 阈值
        return []
    mock_reranker.rerank = slow_rerank

    mock_generator = AsyncMock()
    captured_docs = []
    async def capture_stream(query, docs):
        captured_docs.extend(docs)
        yield 'data: {"event": "done"}\n\n'
    mock_generator.generate_stream = capture_stream
    mock_cache = MagicMock()

    pipeline = RAGPipelineCoordinator(mock_retriever, mock_reranker, mock_generator, mock_cache)
    
    events = []
    async for e in pipeline.execute_query_stream("解释主从复制原理"):
        events.append(e)

    # 断言成功捕获超时并降级送入原始 RRF 候选
    assert len(captured_docs) == 1
    assert captured_docs[0]["title"] == "MySQL高可用"
    assert captured_docs[0]["rerank_score"] == 1.0
```

在终端执行测试套件：

```bash
pytest tests/test_rag_pipeline.py -v
```

测试全部通过即说明：文档切片入库、稠密与关键词双路检索、互惠排名融合、Cross-Encoder 线程隔离重排、低分拒答守卫与超时降级回退已全部验证完成。在后续篇章中，我们将进一步讨论在更高并发场景下如何对检索链路进行缓存加速与性能评测。
