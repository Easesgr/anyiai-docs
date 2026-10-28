# PrepRAG — 详细实现文档

## 项目结构

```
preprag/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI 入口
│   ├── config.py                   # 配置管理
│   ├── api/
│   │   ├── __init__.py
│   │   └── chat.py                 # 对话接口
│   ├── core/
│   │   ├── __init__.py
│   │   ├── document_loader.py      # 文档加载（Markdown + 图片处理）
│   │   ├── chunker.py              # 文档切分（标题感知）
│   │   ├── embedder.py             # 向量化模块
│   │   ├── vector_store.py         # 向量存储（ChromaDB）
│   │   ├── bm25_store.py           # BM25 关键词检索
│   │   ├── hybrid_retriever.py     # 混合检索 + RRF
│   │   ├── category_router.py      # 查询分类路由
│   │   ├── query_rewriter.py       # 查询改写
│   │   ├── reranker.py             # 重排序
│   │   ├── generator.py            # LLM 生成
│   │   ├── self_rag.py             # Self-RAG 评估
│   │   └── rag_pipeline.py         # 流水线编排
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py              # Pydantic 数据模型
│   └── utils/
│       ├── __init__.py
│       ├── markdown_parser.py      # VuePress Markdown 解析
│       └── image_processor.py      # 图片内容提取
├── data/
│   ├── raw/                        # 原始知识库 Markdown + 图片
│   └── processed/                  # 处理后的 chunks (JSON)
├── vector_store/                   # ChromaDB 持久化
├── scripts/
│   ├── ingest.py                   # 数据导入脚本
│   └── evaluate.py                 # 评估脚本
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## 1. VuePress Markdown 解析

知识库的 Markdown 有 VuePress 特有格式，需要专门处理。

### 1.1 Frontmatter 提取

```python
# app/utils/markdown_parser.py
import re
import yaml

class VuePressMarkdownParser:
    """解析 VuePress 格式的 Markdown 文件"""

    def parse(self, file_path: str) -> dict:
        content = open(file_path, "r", encoding="utf-8").read()

        # 1. 提取 frontmatter
        frontmatter = {}
        fm_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
        if fm_match:
            frontmatter = yaml.safe_load(fm_match.group(1))
            body = content[fm_match.end():]
        else:
            body = content

        # 2. 移除 VuePress 组件
        body = self._remove_vuepress_components(body)

        # 3. 提取图片信息
        images = self._extract_images(body)

        # 4. 提取标题结构
        headings = self._extract_headings(body)

        return {
            "frontmatter": frontmatter,  # {title, icon, tag}
            "body": body,
            "images": images,
            "headings": headings,
            "file_path": file_path,
        }

    def _remove_vuepress_components(self, text: str) -> str:
        """移除 VuePress 自定义组件，如 <ReadMoreLock />"""
        text = re.sub(r"<ReadMoreLock\s*/?>", "", text)
        text = re.sub(r"<[^>]+/>", "", text)  # 其他自闭合组件
        return text.strip()

    def _extract_images(self, text: str) -> list[dict]:
        """提取 Markdown 图片，返回 alt + path"""
        pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
        return [
            {"alt": alt, "path": path}
            for alt, path in re.findall(pattern, text)
        ]

    def _extract_headings(self, text: str) -> list[dict]:
        """提取标题层级结构"""
        headings = []
        for match in re.finditer(r"^(#{1,3})\s+(.+)$", text, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append({"level": level, "title": title, "pos": match.start()})
        return headings
```

### 1.2 带元数据的切分

```python
# app/core/chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MarkdownChunker:
    """基于标题层级的语义切分，保留元数据"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", ".", " "],
        )

    def chunk(self, parsed_doc: dict) -> list[dict]:
        body = parsed_doc["body"]
        frontmatter = parsed_doc["frontmatter"]
        images = parsed_doc["images"]
        headings = parsed_doc["headings"]

        # 按 H2 分段
        sections = self._split_by_h2(body)

        chunks = []
        for section in sections:
            # 对每个 H2 段落，如果太长就二次切分
            texts = self.splitter.split_text(section["text"])

            for i, text in enumerate(texts):
                # 找到当前文本所属的图片
                chunk_images = [
                    img for img in images
                    if img["alt"] in text
                ]

                chunks.append({
                    "content": text,
                    "metadata": {
                        "title": frontmatter.get("title", ""),
                        "tags": frontmatter.get("tag", []),
                        "h2": section["h2"],
                        "file_path": parsed_doc["file_path"],
                        "category": self._infer_category(parsed_doc["file_path"]),
                        "images": chunk_images,
                        "image_descriptions": [
                            img.get("description", img["alt"])
                            for img in chunk_images
                        ],
                    },
                })

        return chunks

    def _split_by_h2(self, text: str) -> list[dict]:
        """按 H2 标题分段"""
        import re
        parts = re.split(r"\n(?=## )", text)
        sections = []
        for part in parts:
            h2_match = re.match(r"## (.+)\n", part)
            h2_title = h2_match.group(1).strip() if h2_match else "概述"
            sections.append({"h2": h2_title, "text": part})
        return sections

    def _infer_category(self, file_path: str) -> str:
        """从文件路径推断所属类目"""
        # src/fundamentals/ml-basics/what-is-ml.md → fundamentals/ml-basics
        # src/interview/rag-interview.md → interview
        parts = file_path.split("/")
        if "fundamentals" in parts:
            idx = parts.index("fundamentals")
            if idx + 2 < len(parts):
                return f"fundamentals/{parts[idx+1]}"
            return "fundamentals"
        elif "interview" in parts:
            return "interview"
        return "other"
```

---

## 2. 图片内容增强

### 2.1 图片描述提取

```python
# app/utils/image_processor.py
from openai import OpenAI
import base64
from pathlib import Path

class ImageProcessor:
    """使用 GPT-4o Vision 提取图片内容描述"""

    def __init__(self):
        self.client = OpenAI()

    def describe_image(self, image_path: str, alt_text: str = "") -> str:
        """对单张图片生成结构化描述"""
        image_data = base64.b64encode(
            Path(image_path).read_bytes()
        ).decode()

        prompt = f"""请为这张技术知识库中的图片生成结构化描述，用于后续的文本检索。

图片的原始 alt 文本：{alt_text if alt_text else "无"}

请按以下格式输出：
- 图片类型：（流程图 / 对比图 / 架构图 / 示意图 / 表格 / 公式 / 其他）
- 核心内容：（用 2-3 句话概括图片展示的主要信息）
- 关键元素：（列出图片中的关键组件、步骤或数据点，用逗号分隔）

要求：描述要包含足够的语义信息，使得用户通过自然语言提问时能检索到这张图片。"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        return response.choices[0].message.content

    def process_batch(self, image_dir: str, output_file: str):
        """批量处理知识库中的所有图片"""
        import json

        results = {}
        image_files = list(Path(image_dir).rglob("*.png"))

        for img_path in image_files:
            # 跳过封面图
            if "_cover" in img_path.stem:
                continue

            description = self.describe_image(str(img_path))
            results[str(img_path)] = description
            print(f"Processed: {img_path.name}")

        # 保存结果
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return results
```

### 2.2 图片描述注入 chunk

在 `chunker.py` 中，将图片描述注入到 chunk 的 content 中：

```python
def _enrich_chunk_with_images(self, chunk_text: str, images: list[dict]) -> str:
    """将图片描述附加到 chunk 文本中"""
    if not images:
        return chunk_text

    image_contexts = []
    for img in images:
        desc = img.get("description", img.get("alt", ""))
        if desc:
            image_contexts.append(f"[图片内容：{desc}]")

    if image_contexts:
        return chunk_text + "\n\n" + "\n".join(image_contexts)
    return chunk_text
```

---

## 3. 分类感知检索

### 3.1 查询分类器

```python
# app/core/category_router.py
from openai import OpenAI

# 知识库类目体系
CATEGORIES = {
    "fundamentals": {
        "ml-basics": "机器学习基础：数据、训练、损失函数、优化器、过拟合等",
        "dl-basics": "深度学习基础：神经网络、反向传播、CNN、RNN、注意力机制等",
        "transformer": "Transformer 架构详解",
        "llm-basics": "大模型基础：GPT、LLaMA 等",
        "token-and-tokenizer": "Token 与分词",
        "embedding": "Embedding 向量化",
        "training-and-finetuning": "训练与微调：SFT、LoRA、RLHF 等",
        "inference-and-generation": "推理与生成：采样策略、解码方法",
        "multimodal-basics": "多模态基础",
        "evaluation-metrics": "评测指标：BLEU、ROUGE、Perplexity 等",
    },
    "interview": {
        "llm-interview": "大模型面试题",
        "prompt-interview": "Prompt 面试题",
        "rag-interview": "RAG 面试题",
        "agent-interview": "Agent 面试题",
        "ai-coding-interview": "AI 编程面试题",
        "ai-system-design-interview": "AI 系统设计面试题",
        "model-deployment-interview": "模型部署面试题",
        "high-freq-scenarios": "高频场景题",
    },
}

class CategoryRouter:
    """判断用户问题属于哪个知识域和子类目"""

    def __init__(self):
        self.client = OpenAI()

    def classify(self, query: str) -> dict:
        """返回分类结果"""
        category_desc = "\n".join(
            f"- {domain}/{sub}: {desc}"
            for domain, subs in CATEGORIES.items()
            for sub, desc in subs.items()
        )

        prompt = f"""你是一个知识库分类专家。请判断用户的问题属于以下哪个类目。

类目体系：
{category_desc}

用户问题：{query}

请只输出 JSON 格式：
{{"domain": "fundamentals 或 interview", "subcategory": "子类目名", "confidence": 0.0-1.0}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        import json
        result = json.loads(response.choices[0].message.content)
        return result
```

### 3.2 带过滤的向量检索

```python
# app/core/vector_store.py
import chromadb
from app.core.embedder import Embedder

class VectorStore:
    def __init__(self, db_path: str, collection_name: str = "preprag"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)
        self.embedder = Embedder()

    def search(
        self,
        query: str,
        top_k: int = 20,
        category: str = None,
        subcategory: str = None,
    ) -> list[dict]:
        """带元数据过滤的向量检索"""
        query_embedding = self.embedder.embed(query)

        # 构建 ChromaDB where 过滤条件
        where = None
        if subcategory:
            where = {"category": {"$eq": subcategory}}
        elif category:
            where = {"category": {"$contains": category}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return [
            {
                "content": doc,
                "metadata": meta,
                "score": 1 - dist,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
```

---

## 4. 混合检索 + RRF

```python
# app/core/hybrid_retriever.py
from rank_bm25 import BM25Okapi
import jieba

class HybridRetriever:
    """向量检索 + BM25 + RRF 融合"""

    def __init__(self, vector_store, all_chunks: list[dict]):
        self.vector_store = vector_store
        self.all_chunks = all_chunks

        # 构建 BM25 索引
        tokenized = [list(jieba.cut(c["content"])) for c in all_chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20, category: str = None) -> list[dict]:
        # 向量检索
        vector_results = self.vector_store.search(query, top_k=top_k, category=category)

        # BM25 检索
        tokens = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_top_idx = bm25_scores.argsort()[-top_k:][::-1]
        bm25_results = [
            {"content": self.all_chunks[i]["content"],
             "metadata": self.all_chunks[i]["metadata"],
             "bm25_score": float(bm25_scores[i])}
            for i in bm25_top_idx
        ]

        # RRF 融合
        rrf_k = 60
        score_map = {}

        for rank, r in enumerate(vector_results):
            key = r["content"][:80]
            score_map[key] = score_map.get(key, {"rrf": 0, "data": r})
            score_map[key]["rrf"] += 1 / (rrf_k + rank + 1)

        for rank, r in enumerate(bm25_results):
            key = r["content"][:80]
            score_map[key] = score_map.get(key, {"rrf": 0, "data": r})
            score_map[key]["rrf"] += 1 / (rrf_k + rank + 1)

        sorted_items = sorted(score_map.values(), key=lambda x: x["rrf"], reverse=True)
        return [item["data"] for item in sorted_items[:top_k]]
```

---

## 5. 查询改写

```python
# app/core/query_rewriter.py
from openai import OpenAI

class QueryRewriter:
    def __init__(self):
        self.client = OpenAI()

    def rewrite(self, query: str, chat_history: list[dict] = None) -> list[str]:
        history_text = ""
        if chat_history:
            for msg in chat_history[-6:]:
                role = "用户" if msg["role"] == "user" else "助手"
                history_text += f"{role}: {msg['content'][:200]}\n"

        prompt = f"""你是一个查询改写专家。根据用户的问题，生成 2-3 个不同角度的搜索查询。

{f"对话历史：\n{history_text}" if history_text else ""}

用户问题：{query}

要求：
1. 如果有指代词（"它"、"这个"），结合对话历史消解
2. 生成的查询覆盖问题的不同表达方式
3. 每个查询独立一行，不要编号

生成的搜索查询："""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        queries = [
            q.strip()
            for q in response.choices[0].message.content.strip().split("\n")
            if q.strip()
        ]

        if query not in queries:
            queries.insert(0, query)

        return queries[:3]
```

---

## 6. 重排序

```python
# app/core/reranker.py
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        if not documents:
            return []

        pairs = [(query, doc["content"]) for doc in documents]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        return sorted(documents, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
```

---

## 7. Self-RAG 评估

```python
# app/core/self_rag.py
from openai import OpenAI
import json

class SelfRAGEvaluator:
    def __init__(self):
        self.client = OpenAI()

    def evaluate(self, query: str, context: str, answer: str) -> dict:
        prompt = f"""评估 RAG 系统输出质量。

用户问题：{query}
检索到的参考内容：
{context}
系统回答：
{answer}

从三个维度评估（YES/NO + 原因）：
1. 相关性：参考内容是否与问题相关？
2. 支撑度：回答是否基于参考内容？
3. 有用性：回答是否解答了问题？

输出 JSON：{{"relevance": true/false, "support": true/false, "usefulness": true/false, "reason": "..."}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        checks = [result["relevance"], result["support"], result["usefulness"]]
        result["confidence"] = sum(checks) / len(checks)
        return result
```

---

## 8. 流水线编排

```python
# app/core/rag_pipeline.py
from app.core.category_router import CategoryRouter
from app.core.query_rewriter import QueryRewriter
from app.core.hybrid_retriever import HybridRetriever
from app.core.reranker import Reranker
from app.core.generator import Generator
from app.core.self_rag import SelfRAGEvaluator

class RAGPipeline:
    def __init__(self, hybrid_retriever: HybridRetriever):
        self.router = CategoryRouter()
        self.rewriter = QueryRewriter()
        self.retriever = hybrid_retriever
        self.reranker = Reranker()
        self.generator = Generator()
        self.self_rag = SelfRAGEvaluator()

    async def run(self, query: str, chat_history: list[dict] = None) -> dict:
        # 1. 分类
        category = self.router.classify(query)

        # 2. 查询改写
        queries = self.rewriter.rewrite(query, chat_history)

        # 3. 混合检索（带类目过滤）
        all_results = []
        seen = set()
        for q in queries:
            results = self.retriever.search(
                q, top_k=10, category=category.get("subcategory")
            )
            for r in results:
                key = r["content"][:80]
                if key not in seen:
                    seen.add(key)
                    all_results.append(r)

        # 4. 重排序
        reranked = self.reranker.rerank(query, all_results, top_k=5)

        # 5. 组装上下文（含图片描述）
        context_parts = []
        for r in reranked:
            text = f"[来源: {r['metadata'].get('title', '')} > {r['metadata'].get('h2', '')}]\n{r['content']}"
            img_descs = r["metadata"].get("image_descriptions", [])
            if img_descs:
                text += "\n" + "\n".join(f"[图片：{d}]" for d in img_descs)
            context_parts.append(text)
        context = "\n\n---\n\n".join(context_parts)

        # 6. 生成
        answer = await self.generator.generate(query, context, chat_history)

        # 7. Self-RAG 评估
        eval_result = self.self_rag.evaluate(query, context, answer)

        # 8. 评估不通过则二次检索
        if eval_result["confidence"] < 0.67:
            broader = self.retriever.search(query, top_k=30)
            reranked = self.reranker.rerank(query, broader, top_k=8)
            context = "\n\n---\n\n".join(r["content"] for r in reranked)
            answer = await self.generator.generate(query, context, chat_history)

        return {
            "answer": answer,
            "sources": [
                {
                    "title": r["metadata"].get("title", ""),
                    "section": r["metadata"].get("h2", ""),
                    "category": r["metadata"].get("category", ""),
                }
                for r in reranked
            ],
            "confidence": eval_result["confidence"],
            "detected_category": category,
        }
```

---

## 9. 数据导入脚本

```python
# scripts/ingest.py
from pathlib import Path
from app.utils.markdown_parser import VuePressMarkdownParser
from app.utils.image_processor import ImageProcessor
from app.core.chunker import MarkdownChunker
from app.core.embedder import Embedder
import chromadb
import json

def ingest(knowledge_base_dir: str, image_descriptions_file: str, db_path: str):
    parser = VuePressMarkdownParser()
    chunker = MarkdownChunker()
    embedder = Embedder()

    # 加载图片描述（如果已生成）
    image_descs = {}
    if Path(image_descriptions_file).exists():
        image_descs = json.loads(Path(image_descriptions_file).read_text())

    # 遍历所有 Markdown 文件
    md_files = list(Path(knowledge_base_dir).rglob("*.md"))
    md_files = [f for f in md_files if "README.md" not in f.name]

    all_chunks = []
    for md_file in md_files:
        parsed = parser.parse(str(md_file))

        # 跳过占位页
        if "占位页" in parsed["body"]:
            continue

        # 注入图片描述
        for img in parsed["images"]:
            if str(img["path"]) in image_descs:
                img["description"] = image_descs[str(img["path"])]

        chunks = chunker.chunk(parsed)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    # 存入 ChromaDB
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="preprag",
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c["content"] for c in batch]
        embeddings = embedder.embed_batch(texts)
        metadatas = []
        for c in batch:
            m = c["metadata"].copy()
            # ChromaDB metadata 只支持基本类型
            m["tags"] = ",".join(m.get("tags", []))
            m["images"] = json.dumps(m.get("images", []), ensure_ascii=False)
            m["image_descriptions"] = ",".join(m.get("image_descriptions", []))
            metadatas.append(m)
        ids = [f"chunk_{i+j}" for j in range(len(batch))]

        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    print(f"Ingested {len(all_chunks)} chunks into ChromaDB")

    # 同时保存原始 chunks 用于 BM25 索引
    Path("data/processed/chunks.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2)
    )

if __name__ == "__main__":
    import sys
    ingest(
        knowledge_base_dir=sys.argv[1],
        image_descriptions_file="data/processed/image_descriptions.json",
        db_path="vector_store/",
    )
```

---

## 10. API 服务

```python
# app/api/chat.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float
    detected_category: dict

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    from app.main import pipeline
    result = await pipeline.run(request.message, request.history)
    return ChatResponse(**result)

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    from app.main import pipeline

    async def generate():
        async for chunk in pipeline.run_stream(request.message, request.history):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

```python
# app/main.py
from fastapi import FastAPI
from app.api.chat import router as chat_router
from app.core.rag_pipeline import RAGPipeline
from app.core.hybrid_retriever import HybridRetriever
from app.core.vector_store import VectorStore
import json

app = FastAPI(title="PrepRAG", version="1.0.0")
app.include_router(chat_router, prefix="/api/v1")

# 初始化
vector_store = VectorStore(db_path="vector_store/")
all_chunks = json.loads(open("data/processed/chunks.json").read())
pipeline = RAGPipeline(HybridRetriever(vector_store, all_chunks))

@app.get("/health")
async def health():
    return {"status": "ok"}
```

---

## 11. 评估

### 评估数据集构建

```python
# 手动构建测试集，每个类目 5-10 个问题
test_set = [
    {
        "query": "什么是梯度消失？怎么解决？",
        "expected_category": "fundamentals/dl-basics",
        "expected_docs": ["gradient-vanishing-exploding"],
        "expected_answer_keywords": ["梯度", "激活函数", "残差连接", "BatchNorm"],
    },
    {
        "query": "LoRA 和全量微调有什么区别？",
        "expected_category": "fundamentals/training-and-finetuning",
        "expected_docs": ["training-and-finetuning"],
        "expected_answer_keywords": ["低秩", "参数量", "效率", "效果"],
    },
    # ...
]
```

### 评估指标

| 指标 | 定义 | 目标 |
|------|------|------|
| Hit@5 | Top-5 中是否包含正确文档 | > 85% |
| 分类准确率 | 查询分类是否正确 | > 90% |
| 回答相关性 | 人工评分 1-5 | > 4.0 |
| 可追溯性 | 回答是否被检索内容支撑 | > 90% |
| 图片检索召回 | 含图内容是否被正确检索 | > 75% |

---

## 12. 配置

```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    vision_model: str = "gpt-4o-mini"
    chroma_db_path: str = "./vector_store"
    collection_name: str = "preprag"
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60

    class Config:
        env_file = ".env"
```

```txt
# requirements.txt
fastapi==0.115.0
uvicorn==0.30.0
langchain==0.3.0
langchain-community==0.3.0
openai==1.50.0
chromadb==0.5.0
rank-bm25==0.2.2
jieba==0.42.1
sentence-transformers==3.0.0
pydantic-settings==2.5.0
pyyaml==6.0.2
```
