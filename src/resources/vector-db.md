---
title: 向量数据库
icon: mdi:vector-square
tag:
  - 向量数据库
  - RAG
  - Embedding
---

# 向量数据库选型

向量库负责在大量 Embedding 中找“语义上相近”的内容，但它并不替代文档解析、切分、重排或答案评估。RAG 效果差时，先确认数据和检索设计，再换数据库。

## 先选架构，不要先选产品

| 你的约束 | 首选方向 |
| --- | --- |
| 已有 PostgreSQL，数据量中等 | pgvector |
| 快速原型、本地试验 | Chroma、LanceDB、Qdrant Local |
| 托管优先、希望少运维 | Pinecone、Qdrant Cloud、Zilliz Cloud、Weaviate Cloud |
| 大规模、多向量字段或集群 | Milvus / Zilliz |
| JSON 过滤复杂、Rust 服务友好 | Qdrant |
| 已使用 Elasticsearch / OpenSearch | 直接评估其向量与混合检索能力 |

## 核心候选

### pgvector

- 官方入口：[pgvector](https://github.com/pgvector/pgvector)
- 定位：PostgreSQL 扩展，在熟悉的 SQL、事务、权限和备份体系内增加向量搜索。
- 适合：业务数据已在 Postgres、需要关联查询/一致性、规模尚不需要专用集群的团队。
- 注意：用真实过滤条件和并发压测 HNSW / IVFFlat；不要只拿纯向量 benchmark 代替业务测试。

### Qdrant

- 官方入口：[Qdrant 文档](https://qdrant.tech/documentation/)
- 定位：开源向量搜索引擎，重视 payload 过滤、稠密/稀疏向量和托管服务。
- 适合：检索过滤复杂、希望自托管与云托管可切换的 RAG 系统。
- 注意：设计好 collection、payload 索引与租户隔离；过滤字段未索引会直接拖慢线上查询。

### Milvus / Zilliz Cloud

- 官方入口：[Milvus](https://milvus.io/docs) · [Zilliz Cloud](https://docs.zilliz.com/)
- 定位：面向大规模向量工作负载的开源系统及其托管服务。
- 适合：多模态、大数据量、多向量字段、需要水平扩展的团队。
- 注意：分布式能力意味着更高运维复杂度；先从 Lite/Standalone 或云服务验证容量模型。

### Pinecone

- 官方入口：[Pinecone 文档](https://docs.pinecone.io/)
- 定位：托管向量数据库，强调低运维和快速接入。
- 适合：优先交付应用、团队不想维护集群、接受云服务依赖的场景。
- 注意：提前设计 namespace、元数据、删除与重建策略，并核对区域和数据驻留要求。

### Weaviate

- 官方入口：[Weaviate 文档](https://docs.weaviate.io/)
- 定位：开源向量数据库，提供混合检索与模块化集成。
- 适合：需要较完整的对象模型、多模态或希望使用其生态模块的团队。
- 注意：不要把“内置生成/向量化”当作不可替换黑盒；Embedding、检索与生成应可独立评估。

### Chroma 与 LanceDB

- 官方入口：[Chroma](https://docs.trychroma.com/) · [LanceDB](https://lancedb.github.io/lancedb/)
- 定位：开发者友好的本地/嵌入式方案。
- 适合：Notebook、PoC、单机工具和小型应用。
- 注意：扩展到多实例、备份、权限、多租户时应重新评估，而不是机械地“把本地库搬上服务器”。

## RAG 检索必备设计

1. 每个 chunk 保存来源、标题、更新时间、权限和版本等 metadata。
2. 同时评估 dense、关键词/稀疏和 hybrid 检索；只做向量检索常会漏掉精确实体。
3. 使用 reranker 处理候选集，再把少量高质量上下文交给模型。
4. 用命中率、MRR/nDCG、引用正确性和最终回答质量评估，而不是只看检索耗时。
5. 设计删除、重嵌入、版本回滚和权限变更传播流程。

## 最小决策建议

已有 Postgres 先试 pgvector；需要复杂过滤可优先试 Qdrant；要托管和快速上线评估 Pinecone 或 Zilliz；数据和并发真正上来后，再以压测结果决定是否引入分布式集群。

## 索引与参数：先理解取舍

| 方案 | 优点 | 代价 | 常见用途 |
| --- | --- | --- | --- |
| Flat | 精确召回 | 数据大时慢 | 小数据、评测基线 |
| HNSW | 高召回、低延迟 | 内存和建索引成本高 | 在线知识库 |
| IVF/PQ | 节省内存、可扩展 | 参数调优与召回损失 | 大规模向量 |

无论选哪种索引，都要分别测“无过滤”和“带权限/时间过滤”的结果。生产查询往往不是纯 ANN，过滤策略会改变索引效果。

## 文档数据模型示例

~~~json
{
  "id": "handbook-v3-p12-c04",
  "vector": ["embedding"],
  "text": "原始段落或清洗后的文本",
  "metadata": {
    "source": "employee-handbook.pdf",
    "document_version": "v3",
    "page": 12,
    "tenant_id": "acme",
    "acl": ["hr", "manager"],
    "updated_at": "2026-07-16"
  }
}
~~~

这类 metadata 不是附属信息：引用、权限过滤、增量更新和排障都依赖它。

## 失败排查顺序

先看原文是否被正确解析与切分；再检查查询重写和 embedding 是否一致；然后检查过滤是否把正确答案排除了；最后才考虑换数据库或调索引。多数“向量库不准”问题发生在入库和评测之前。
