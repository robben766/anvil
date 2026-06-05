# KB-M5:Contextual Retrieval 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 Anthropic Contextual Retrieval:ingest 时对每个 chunk 用 LLM 生成 1-2 句定位上下文,前缀进检索表示(embedding + BM25),原文与 offset 不动(引用高亮零影响);**全文档作为 prompt 共享前缀吃 DeepSeek prompt cache**,富集成本与缓存命中率直接从 P0 网关记账读出——P0 底座与 P1 产品的闭环实证。

**核心机制:** messages = [system: "以下是完整文档:\n{全文}"(跨 chunk 不变 → 缓存前缀), user: "给出下面这段在文档中的定位上下文(1-2 句,直接输出):\n{chunk}"]。M2 实测:>1K token 前缀命中率 ~96%、缓存价 0.02 vs 1 元/M——富集一份文档的边际成本极低,这正是 Anthropic 原方案的经济学。

**数据设计:** kb_chunks 新列 `context_prefix TEXT NOT NULL DEFAULT ''`;检索表示 = `context_prefix + "\n" + content`(embedding 输入与 BM25 索引输入);`content`/offset 不变。

---

### Task C1: enricher + 迁移 0003 + pipeline 接线(TDD,mock chat)

**Files:** Create `packages/kb/src/anvil_kb/ingest/enrich.py`;Create `packages/kb/alembic/versions/0003_context_prefix.py`(down_revision=0002;加列+对称 downgrade);Modify `db.py`(ChunkRow+context_prefix)、`store/base.py`(Chunk dataclass +`context_prefix: str = ""`)、`ingest/pipeline.py`、`store/pg.py`(读写新列)、`store/bm25.py`(index_chunks 的分词输入改为 `(c.context_prefix + "\n" + c.content) if c.context_prefix else c.content`)
- enrich.py:
```python
async def enrich_chunks(*, doc_text: str, drafts: list[ChunkDraft], chat=None,
                        session_id: str = "anvil-kb-enrich") -> list[str]:
    """逐 chunk 串行调 chat("chat-default"),system=全文(共享缓存前缀),user=chunk 定位指令。
    返回与 drafts 等长的 context 列表;单 chunk 失败(异常)→ 该位置空串并继续(fail-open,
    富集是增强不是依赖);chat 可注入。温度 0.3,max_tokens 120。"""
```
- pipeline.ingest_markdown +`enricher: bool/callable?`——签名定为 `enrich_chat=None`:非 None 时调 enrich_chunks(chat=enrich_chat) 并把结果写入 Chunk.context_prefix;embedding 输入同步改为富集表示(有 prefix 时)。**注意批量 embed 仍一次调用。**
- 测试(mock chat 记录 messages):system 含全文且跨调用相同(缓存前缀断言);user 含对应 chunk;返回写入 context_prefix;embedding 输入 == prefix+"\n"+content(假 embedder 记录输入);某次 chat 抛错 → 对应 prefix=""、其余正常、不抛;enrich_chat=None 零行为变化(全回归);BM25 分词输入含 prefix 词(假 sparse 或真 PG 验证 search 能命中仅存在于 prefix 的词);PgVectorStore 回读 context_prefix

### Task C2: CLI --enrich + 富集成本报告(读 P0 记账)

**Files:** Modify `packages/kb/src/anvil_kb/cli.py`;Test test_cli.py 补
- ingest 与 eval +`--enrich`:传 enrich_chat=anvil_gateway.chat(函数内 import)
- 富集后打印成本报告:查 gateway 的 usage_records(`session_id='anvil-kb-enrich'`,本次运行起始时间之后的行),汇总:调用数/总 token/缓存命中 token/命中率/总成本(读法参照 packages/core/gateway 的 db.py UsageRecordRow,直接 SQLAlchemy 查,**不改 gateway 任何代码**)。输出形如:`enrich: 29 calls, 45k prompt tokens, cache hit 92.3%, cost ¥0.012`
- 测试:flag 透传(mock);成本汇总函数纯逻辑测试(构造假行)

### Task C3: live 对比实验 + 文档 + PR

- 真跑(真 PG+真 embedding+真 LLM 富集,~30 chunk 次调用):
  1. 基线已知(hybrid MRR 0.881);`eval --enrich --mode hybrid` 重灌富集语料跑 16 例;再 `--enrich --mode hybrid --rerank` 一次
  2. 记录:MRR/recall 对比(重点 kb-15/16 换述例——上下文前缀对换述的理论增益);**富集成本报告原样贴**(调用数/缓存命中率/总成本)——验证"全文前缀吃缓存"的经济学,跟 M2 实测呼应
  3. examples/04-kb/README「Contextual Retrieval 实验(KB-M5)」:对比表+成本表+如实解读(没提升也原样写——富集对"语料本就短小自洽"的场景增益可能有限,这本身是诚实结论)
  4. 实验后恢复非富集语料(重跑基线 eval)
- README/CLAUDE.md 状态 KB-M5(P1 必做里程碑全部完成);全量回归;PR `P1 KB-M5: contextual retrieval riding the prompt cache`

---

## 自审
- spec §10 KB-M5 行覆盖(Contextual Retrieval + prompt cache 压成本);GraphRAG(KB-M6)维持可选不做,P1 必做闭环
- 引用回链零影响(content/offset 不动)是硬约束,C1 测试锁定
- fail-open 设计:富集失败退化为普通 ingest,不阻断产品路径
- 不改 packages/core(成本报告只读 usage_records 表)
