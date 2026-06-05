"""手写 BM25(Okapi BM25):jieba 分词 + PG 倒排索引。

参数:  k1=1.5, b=0.75 (Okapi 默认值)

IDF 公式(Okapi +1 平滑,恒正):
    idf(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

打分公式:
    score(q, c) = Σ_t  idf(t) * tf(t,c) * (k1+1)
                       ─────────────────────────────────────────
                       tf(t,c) + k1 * (1 - b + b * dl(c) / avgdl)
"""

from __future__ import annotations

import math
import re
import uuid
from collections import Counter

from sqlalchemy import Float, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_kb.db import ChunkRow, PostingRow
from anvil_kb.store.base import Chunk, ScoredChunk

K1: float = 1.5
B: float = 0.75


def tokenize(text_input: str) -> list[str]:
    """jieba cut_for_search → 过滤空 token 及纯标点 → lower。

    jieba 懒加载:首次调用约 ~1s,之后缓存。
    """
    import jieba  # noqa: PLC0415 — 懒加载，避免模块导入期开销

    tokens: list[str] = []
    for tok in jieba.cut_for_search(text_input):
        tok = tok.strip().lower()
        # 丢弃空 token 以及不含任何字母/数字/汉字的 token（纯标点/空白）
        if tok and re.search(r"[a-z0-9一-鿿]", tok):
            tokens.append(tok)
    return tokens


class PgBM25Index:
    """BM25 稀疏索引:倒排索引存 kb_postings 表,单事务读写。

    Args:
        session_factory: async_sessionmaker[AsyncSession]
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def index_chunks(self, chunks: list[Chunk]) -> None:
        """为 chunks 建倒排索引。

        - 幂等:同 chunk 重复 index 前先删其旧 postings,再写新 postings。
        - 单事务:所有 chunk 在同一事务内完成。
        """
        # NOTE: 单写者假设——并发对同一 chunk 调 index_chunks 可能撞 (term, chunk_id) 复合 PK;
        # 当前 ingest 串行,无需加锁
        if not chunks:
            return

        async with self._session_factory() as session:
            async with session.begin():
                chunk_ids = [c.id for c in chunks]

                # 先删这批 chunk 的旧 postings（幂等保证）
                await session.execute(
                    delete(PostingRow).where(PostingRow.chunk_id.in_(chunk_ids))
                )

                for chunk in chunks:
                    tokens = tokenize(chunk.content)
                    tf_counter = Counter(tokens)
                    token_count = len(tokens)

                    # 写入 postings
                    for term, tf_val in tf_counter.items():
                        session.add(
                            PostingRow(
                                term=term,
                                chunk_id=chunk.id,
                                tf=tf_val,
                            )
                        )

                    # 更新 token_count（使用 synchronize_session=False 避免 ORM 状态冲突）
                    await session.execute(
                        update(ChunkRow)
                        .where(ChunkRow.id == chunk.id)
                        .values(token_count=token_count)
                        .execution_options(synchronize_session=False)
                    )

    async def search(self, query: str, k: int) -> list[ScoredChunk]:
        """BM25 检索:tokenize query → 查倒排 → Python 累加打分 → top-k。

        Returns:
            top-k ScoredChunk,score = BM25 分；分相同时按 chunk_id 稳定排序。
            若 N=0 或 query 词全不在库中 → 返回 []。
        """
        query_terms = list(dict.fromkeys(tokenize(query)))  # 去重保序
        if not query_terms:
            return []

        async with self._session_factory() as session:
            # ── 1. 全局统计:N 和 avgdl ────────────────────────────────────
            # BM25 宇宙 = 已建倒排的 chunk 子集(token_count > 0)。
            # dense-only chunk(token_count=0)不在 kb_postings 中,
            # 若纳入统计会虚增 N 并拉低 avgdl,污染 IDF 和长度归一。
            stats_row = await session.execute(
                select(
                    func.count(ChunkRow.id).label("n"),
                    func.avg(ChunkRow.token_count).cast(Float).label("avgdl"),
                ).where(ChunkRow.token_count > 0)
            )
            stats = stats_row.one()
            n_total: int = stats.n
            avgdl: float = float(stats.avgdl) if stats.avgdl is not None else 0.0

            if n_total == 0:
                return []
            if avgdl == 0.0:
                # 所有文档长度均为 0，无法归一化，退出
                avgdl = 1.0

            # ── 2. 取 query 词命中的所有 postings(含 token_count) ──────────
            postings_result = await session.execute(
                select(
                    PostingRow.chunk_id,
                    PostingRow.term,
                    PostingRow.tf,
                    ChunkRow.token_count,
                )
                .join(ChunkRow, ChunkRow.id == PostingRow.chunk_id)
                .where(PostingRow.term.in_(query_terms))
            )
            postings = postings_result.all()

            if not postings:
                return []

            # ── 3. 统计 df(文档频率):每个 term 在多少不同 chunk 中出现 ───────
            df_counter: Counter[str] = Counter()
            for row in postings:
                df_counter[row.term] += 1  # 每行代表一个 (term, chunk) 对

            # ── 4. Python 端逐 chunk 累加 BM25 分数 ─────────────────────────
            scores: dict[uuid.UUID, float] = {}
            for row in postings:
                term = row.term
                tf_val = row.tf
                dl = row.token_count
                df_val = df_counter[term]

                # idf(Okapi +1 平滑)
                idf = math.log((n_total - df_val + 0.5) / (df_val + 0.5) + 1)

                # BM25 term score
                term_score = (
                    idf
                    * tf_val
                    * (K1 + 1)
                    / (tf_val + K1 * (1 - B + B * dl / avgdl))
                )
                scores[row.chunk_id] = scores.get(row.chunk_id, 0.0) + term_score

            if not scores:
                return []

            # ── 5. top-k 排序(分数降序,chunk_id 升序 tie-break) ─────────────
            sorted_ids = sorted(
                scores.keys(), key=lambda cid: (-scores[cid], str(cid))
            )
            top_ids = sorted_ids[:k]

            # ── 6. 批量取回 Chunk 字段 ────────────────────────────────────────
            chunks_result = await session.execute(
                select(ChunkRow).where(ChunkRow.id.in_(top_ids))
            )
            chunk_rows = {row.id: row for row in chunks_result.scalars()}

        # 按排序顺序组装结果
        result: list[ScoredChunk] = []
        for cid in top_ids:
            row = chunk_rows.get(cid)
            if row is None:
                continue
            result.append(
                ScoredChunk(
                    chunk=Chunk(
                        id=row.id,
                        document_id=row.document_id,
                        seq=row.seq,
                        content=row.content,
                        header_path=row.header_path,
                        start_offset=row.start_offset,
                        end_offset=row.end_offset,
                        embedding=None,  # 不回传向量，省流量
                    ),
                    score=scores[cid],
                )
            )
        return result
