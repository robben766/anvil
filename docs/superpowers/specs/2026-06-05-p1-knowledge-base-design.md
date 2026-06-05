# P1 产品①:通用知识库(anvil-kb)设计文档

> 状态:已批准(2026-06-05)。P1 的单一事实来源。
> 前置:P0 公共底座已完成(gateway / obs / eval 三件套,见 `2026-06-04-p0-common-platform-design.md`)。

## 1. 目标与非目标

**目标**:一个可用的 RAG 知识库产品,同时是 RAG 全链路的深度学习载体——检索内核(BM25、RRF、rerank 编排)全部自研,只在"无学习收益的轮子"(embedding 模型推理、PDF 底层解码)上用现成库。

- 上传文档 → 解析分块 → 双路索引(向量 + BM25)→ 混合检索 → 重排 → 带引用回链的生成
- 每个里程碑用 eval 分数验收:检索指标(recall@k / precision@k,自研)+ 生成指标(P0 的 RAGAS 四指标,直接复用)
- Day-1 有 Web UI:上传、问答(流式)、**引用面板**(点击引用高亮原文)——引用回链是本产品唯一必须 UI 才能展示的核心特性

**非目标**(刻意减法):多知识库/多租户、用户体系、对话历史、文档协同编辑、爬虫接入。GraphRAG 列为 P1 可选尾巴,不进必做范围。

## 2. 架构:库优先 + 薄壳(延续 P0 哲学)

```
anvil/
├─ packages/core/{gateway,obs,eval}     ← P0 底座,直接复用
├─ packages/kb/                          ← 知识库核心,纯库,可独立测试
│  └─ src/anvil_kb/
│     ├─ ingest/
│     │  ├─ chunker.py     markdown 标题感知分块 + 字符窗口 + overlap;保留原文 offset
│     │  └─ pipeline.py    ingest_markdown(): 解析→分块→嵌入→入库,一个事务
│     ├─ embed.py          Embedder 协议 + FastEmbedEmbedder(bge-small-zh-v1.5, 512 维,本地)
│     ├─ store/
│     │  ├─ base.py        VectorStore / SparseIndex 协议(抽象接口,Day-1 定义)
│     │  ├─ pg.py          PgVectorStore(pgvector,MVP 唯一实现;Qdrant 实现刻意后置)
│     │  └─ bm25.py        [KB-M2] 手写 BM25:jieba 分词 + 倒排索引(PG 持久化)
│     ├─ retrieve/
│     │  ├─ dense.py       向量 top-k
│     │  ├─ fusion.py      [KB-M2] RRF 融合(手写)
│     │  ├─ rerank.py      [KB-M3] cross-encoder 重排
│     │  └─ retriever.py   检索编排入口(策略可配,逐里程碑增强)
│     ├─ generate.py       带引用回链的生成:编号上下文 → gateway chat → [n] 标记解析回 chunk
│     ├─ db.py             SQLAlchemy 模型(kb_documents / kb_chunks)
│     └─ cli.py            anvil-kb ingest / query / eval
├─ apps/kb-api/                          ← FastAPI 薄壳(uv workspace 成员)
└─ apps/kb-web/                          ← Next.js 15 + pnpm(不进 uv workspace)
```

跨包依赖:`anvil_kb` → `anvil_gateway`(生成走网关,自动获得记账+trace)、`fastembed`。`apps/kb-api` → `anvil_kb`。`anvil_eval` 新增检索指标但**不依赖** `anvil_kb`(指标只吃纯数据)。

## 3. 数据与存储

复用 anvil-postgres(同库 `anvil`,测试用 `anvil_test`),镜像从 `postgres:16-alpine` 换为 `pgvector/pgvector:pg16`(CI service 同步换)。kb 独立 Alembic 迁移链(`packages/kb/alembic`,`version_table="kb_alembic_version"`,与 gateway 的迁移链互不干扰)。

```
kb_documents: id(uuid pk) / title / source_name / content(text,全文,供引用高亮) / created_at
kb_chunks:    id(uuid pk) / document_id(fk, cascade) / seq / content / header_path
              / start_offset / end_offset          ← 指回 documents.content 的绝对偏移
              / embedding vector(512)              ← bge-small-zh-v1.5
              / created_at;HNSW 索引(vector_cosine_ops)
```

`start_offset/end_offset` 是引用回链的根基:UI 拿 chunk 的偏移在原文中高亮,不靠字符串再匹配。

## 4. 接口契约(store/base.py)

```python
class VectorStore(Protocol):
    async def upsert_chunks(self, doc: Document, chunks: list[Chunk]) -> None: ...
    async def search(self, query_vector: list[float], k: int) -> list[ScoredChunk]: ...
    async def delete_document(self, document_id: UUID) -> None: ...

class SparseIndex(Protocol):       # KB-M2 落地实现,Day-1 只定义
    async def index_chunks(self, chunks: list[Chunk]) -> None: ...
    async def search(self, query: str, k: int) -> list[ScoredChunk]: ...
```

`ScoredChunk = (chunk, score)`,score 统一为"越大越相关"(向量侧 = 1 − cosine 距离)。

## 5. 分块策略(MVP)

markdown 标题感知两级分块:先按 `#/##/###` 切 section(`header_path` 记录标题链,如 `条款 > 第二章 > 等待期`),section 内按字符窗口切(默认 size=600、overlap=100,CJK 按字符计)。每个 chunk 记录其在**原始全文**中的绝对偏移。表格(markdown table)整体作为独立 chunk 不切断。层级/父子分块、PDF 版面解析后置到 KB-M4。

## 6. 生成与引用回链协议

1. 检索 top-k chunk,按 `[1]..[k]` 编号拼入 prompt,指令要求:答案中引用资料处标 `[n]`;资料不含答案时明确回答"资料中未找到",不得编造。
2. 走 `anvil_gateway.chat("chat-default", ...)`(自动获得成本记账 + Langfuse trace + fallback)。
3. 解析回答中的 `[n]` 标记 → 映射回 chunk → `citations: [{n, chunk_id, document_id, quote, start_offset, end_offset}]`。
4. 流式:SSE 三类事件——`sources`(检索结果先行推送)→ `delta`(token)→ `done`(实际被引用的 n 列表)。

## 7. API 契约(apps/kb-api)

```
POST   /v1/kb/documents          multipart 上传 .md/.txt → {id, title, chunk_count}
GET    /v1/kb/documents          列表
GET    /v1/kb/documents/{id}     含全文(供引用高亮)
DELETE /v1/kb/documents/{id}
POST   /v1/kb/query              {"question": ..., "k": 5, "stream": true} → SSE(见 §6)
```

错误映射与可选 Bearer 鉴权沿用 P0 proxy 的约定。

## 8. Web UI(apps/kb-web,Day-1,刻意克制)

Next.js 15 单页三块:文档上传列表 / 问答框(SSE 流式渲染)/ 引用面板(点击答案中的 `[n]` 高亮原文对应片段)。KB-M2 起加**检索调试视图**(每路召回分数与 RRF 融合过程)。不做的见 §1 非目标。

## 9. eval 方案(Day-1 纪律:先建 golden set 再写代码)

**golden 语料**:沿用 P0 demo.jsonl 的虚构「星辉人寿」保险宇宙,自写 3 篇 markdown 文档(条款/理赔指南/产品说明,入仓 `packages/kb/golden/corpus/`)——虚构=零版权风险,自控=分块策略变化时 eval 仍稳定。公开技术书只作演示语料(脚本拉取,不入仓)。

**golden set**(`packages/kb/golden/kb.jsonl`,≥12 例,含 ≥2 个拒答例):

```json
{"id": "kb-01", "question": "...", "reference": "...",
 "evidences": ["原文子串1", "原文子串2"], "answerable": true}
```

`anvil_eval.dataset.GoldenCase` 增加可选字段 `evidences: list[str]`、`answerable: bool = True`(向后兼容)。

**检索指标**(新增 `anvil_eval/metrics/retrieval.py`,纯函数、确定性、不调 LLM):
- 判定:chunk 命中 = 包含某条 evidence 子串(空白归一化后)
- `recall@k` = 被 top-k 覆盖的 evidence 数 / evidence 总数;`precision@k` = top-k 中命中 chunk 数 / k
- 与 RAGAS 四指标同样配手算锁定测试

**验收口径**:每个里程碑的验收 = eval 分数变化。KB-M1 门禁:answerable 用例 recall@5 ≥ 0.8;KB-M2 混合检索 recall@5 必须 ≥ 纯向量基线;KB-M3 rerank 后 precision 必须提升——涨不动就写清楚为什么(本身即文章素材)。生成侧 RAGAS 指标作 live 评测(需真 key),不进默认 CI。

## 10. 里程碑(每个手写一个"行业黑盒")

| 里程碑 | 交付 | 自研重点 / 吃透原理 |
|---|---|---|
| KB-M1a core | 朴素 RAG 全链路库 + 检索 eval + CLI(本期) | embedding 几何意义、余弦 top-k、引用回链协议 |
| KB-M1b product | kb-api + kb-web(上传/问答/引用面板) | SSE 产品化、引用高亮 |
| KB-M2 | 手写 BM25(jieba)+ RRF 混合检索 + 调试视图 | BM25 公式、RRF、dense/sparse 互补性 |
| KB-M3 | cross-encoder rerank + 对比实验 | 双塔 vs 交叉编码的精度/成本权衡 |
| KB-M4 | PDF 解析(公开保险条款:版面+表格独立 chunk) | 深度文档理解派在赌什么 |
| KB-M5 | Contextual Retrieval(LLM 补上下文) | prompt cache 压成本(复用 M2 网关缓存实测) |
| KB-M6(可选) | GraphRAG local search | 实体→社区→摘要 |

## 11. 风险与对策

- **alpine→debian 镜像切换**:已有 usage_records 数据卷,collation 版本可能告警 → 切换后 `REINDEX DATABASE` 兜底;数据仅为开发记账,最坏可重建。
- **asyncpg + pgvector 类型注册**:需在连接建立时 `register_vector`,用 SQLAlchemy connect 事件挂载;真 PG 测试覆盖。
- **中文 evidence 子串匹配脆弱**:分块若切断 evidence 会误判未命中 → evidence 取短句(<40 字),且 golden 语料与分块参数同仓固定。
- **fastembed 模型下载**(~100MB):CI 缓存 `~/.cache/fastembed`;eval 包 M4 已趟过此路。
