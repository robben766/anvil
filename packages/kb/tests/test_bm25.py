"""手写 BM25 测试:单元(纯 Python 手算) + 集成(真 PG)。

═══════════════════════════════════════════════════════════════════════
手算演算说明
═══════════════════════════════════════════════════════════════════════

语料设计(3 个 chunk):
  chunk A: 'apple banana cherry'           tokens: [apple, banana, cherry]  dl=3
  chunk B: 'apple apple banana'            tokens: [apple, apple, banana]   dl=3
  chunk C: 'apple banana cherry date elderberry fig'
                                           tokens: [apple,banana,cherry,date,elderberry,fig]  dl=6

注:jieba.cut_for_search 对纯 ASCII 单词的分词行为与空格分词一致——逐词切分，
  不拆单词内部，下方先用 assert 锁定实际输出，再以该输出手算。

全局统计:
  N = 3 (chunk 总数)
  avgdl = (3 + 3 + 6) / 3 = 4.0

文档频率(df):
  apple:      3 (A, B, C)
  banana:     3 (A, B, C)
  cherry:     2 (A, C)
  date:       1 (C)
  elderberry: 1 (C)
  fig:        1 (C)

BM25 参数:  k1=1.5, b=0.75

─────────────────────────────────────────────────────
IDF 计算(Okapi +1 平滑):  idf(t) = ln((N-df+0.5)/(df+0.5) + 1)
─────────────────────────────────────────────────────
  idf(apple)  = ln((3-3+0.5)/(3+0.5) + 1)
              = ln(0.5/3.5 + 1)
              = ln(1.142857...)
              ≈ 0.13353139

  idf(cherry) = ln((3-2+0.5)/(2+0.5) + 1)
              = ln(1.5/2.5 + 1)
              = ln(1.6)
              ≈ 0.47000363

  idf(date)   = ln((3-1+0.5)/(1+0.5) + 1)
              = ln(2.5/1.5 + 1)
              = ln(1 + 5/3)
              = ln(8/3)
              ≈ 0.98082925

─────────────────────────────────────────────────────
打分公式:
  score_term = idf * tf * (k1+1) / (tf + k1*(1 - b + b*dl/avgdl))
─────────────────────────────────────────────────────

【tf 影响验证】Query: "apple"  (apple: df=3, idf≈0.13353139)

  chunk A: tf=1, dl=3
    denom = 1 + 1.5*(1 - 0.75 + 0.75*3/4.0)
          = 1 + 1.5*(1 - 0.75 + 0.5625)
          = 1 + 1.5*0.8125
          = 1 + 1.21875 = 2.21875
    score = 0.13353139 * 1 * 2.5 / 2.21875 ≈ 0.15045791

  chunk B: tf=2, dl=3
    denom = 2 + 1.5*(1 - 0.75 + 0.75*3/4.0)
          = 2 + 1.21875 = 3.21875
    score = 0.13353139 * 2 * 2.5 / 3.21875 ≈ 0.20742741

  chunk C: tf=1, dl=6
    denom = 1 + 1.5*(1 - 0.75 + 0.75*6/4.0)
          = 1 + 1.5*(1 - 0.75 + 1.125)
          = 1 + 1.5*1.375
          = 1 + 2.0625 = 3.0625
    score = 0.13353139 * 1 * 2.5 / 3.0625 ≈ 0.10900522

  排序: B(0.20742741) > A(0.15045791) > C(0.10900522)
  验证:B 的 tf=2 > A 的 tf=1,同 dl;C 的 dl=6 > A 的 dl=3,同 tf → B>A>C ✓

【df 影响验证】Query: "apple cherry"

  chunk A: apple_score + cherry_score
    apple:  0.15045791 (同上)
    cherry: tf=1, dl=3, idf≈0.47000363
      denom = 2.21875
      score = 0.47000363 * 1 * 2.5 / 2.21875 ≈ 0.52958155
    total ≈ 0.68003946

  chunk B: apple_score + cherry_score
    apple:  0.20742741 (同上)
    cherry: tf=0 → score=0
    total ≈ 0.20742741

  chunk C: apple_score + cherry_score
    apple:  0.10900522 (同上)
    cherry: tf=1, dl=6, idf≈0.47000363
      denom = 3.0625
      score = 0.47000363 * 1 * 2.5 / 3.0625 ≈ 0.38367643
    total ≈ 0.49268165

  排序: A(0.68003946) > C(0.49268165) > B(0.20742741)
  验证:cherry 是 df=2 的稀有词,idf 高于 apple(df=3);A 既有 cherry 又有 apple → 总分最高 ✓

【长度归一验证】cherry 在 A(dl=3) vs C(dl=6),同 tf=1,同 idf:
  A: 0.52958155  C: 0.38367643
  A 短文档得分高 → 长度归一有效 ✓

【稀有词验证】Query: "date"  (date: df=1, idf≈0.98082925)

  chunk A: tf=0 → score=0
  chunk B: tf=0 → score=0
  chunk C: tf=1, dl=6
    denom = 3.0625
    score = 0.98082925 * 1 * 2.5 / 3.0625 ≈ 0.80067694

  只有 C 得分 > 0 ✓
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import os
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from anvil_kb.db import make_engine
from anvil_kb.store.base import Chunk
from anvil_kb.store.bm25 import K1, B, PgBM25Index, tokenize

# ─────────────────────────────────────────────────────────────────────
# 测试辅助常量(与手算对齐)
# ─────────────────────────────────────────────────────────────────────
# 与 conftest 同口径:CI 里 ANVIL_TEST_DATABASE_URL 指向 anvil 库,本地默认 anvil_test
TEST_DB_URL = os.environ.get(
    "ANVIL_TEST_DATABASE_URL",
    "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test",
)

# 手算精度:误差 1e-6
APPROX_REL = 1e-6

_DOC_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")

_CHUNK_A_ID = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
_CHUNK_B_ID = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000001")
_CHUNK_C_ID = uuid.UUID("cccccccc-0000-0000-0000-000000000001")

CONTENT_A = "apple banana cherry"
CONTENT_B = "apple apple banana"
CONTENT_C = "apple banana cherry date elderberry fig"


def _make_chunk(cid: uuid.UUID, content: str, seq: int = 0) -> Chunk:
    return Chunk(
        id=cid,
        document_id=_DOC_ID,
        seq=seq,
        content=content,
        header_path="",
        start_offset=0,
        end_offset=len(content),
        embedding=None,
    )


# ══════════════════════════════════════════════════════════════════════
# 单元测试:纯 Python,无 DB
# ══════════════════════════════════════════════════════════════════════


class TestTokenize:
    """锁定 jieba 对测试语料的实际分词输出,确保手算基础可复核。"""

    def test_ascii_words_split_by_space(self) -> None:
        """纯 ASCII 单词以空格分隔时,jieba 逐词切分,不拆单词内部。"""
        result = tokenize(CONTENT_A)
        assert result == ["apple", "banana", "cherry"]

    def test_repeated_token_preserved(self) -> None:
        """重复词不去重,Counter 才能正确计 tf。"""
        result = tokenize(CONTENT_B)
        assert result == ["apple", "apple", "banana"]

    def test_long_chunk_tokens(self) -> None:
        """6 个词的 chunk,dl=6。"""
        result = tokenize(CONTENT_C)
        assert result == ["apple", "banana", "cherry", "date", "elderberry", "fig"]

    def test_filter_punctuation(self) -> None:
        """纯标点/空白 token 被过滤。"""
        result = tokenize("hello , . ! world")
        # 逗号、句号、感叹号不含字母/汉字/数字,应被丢弃
        assert "," not in result
        assert "." not in result
        assert "!" not in result
        assert "hello" in result
        assert "world" in result

    def test_lowercase(self) -> None:
        """输出全小写。"""
        result = tokenize("Apple BANANA")
        assert result == ["apple", "banana"]

    def test_empty_string(self) -> None:
        result = tokenize("")
        assert result == []


class TestBM25HandCalc:
    """手算锁定测试:在纸面演算基础上用 pytest.approx 断言数值。"""

    # ── IDF ──────────────────────────────────────────────────────────
    def _idf(self, n: int, df: int) -> float:
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    # ── 单 term BM25 打分 ─────────────────────────────────────────────
    def _score_term(self, idf: float, tf: int, dl: int, avgdl: float) -> float:
        return idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))

    def test_idf_common_vs_rare(self) -> None:
        """idf(rare) > idf(common):df=1 的稀有词 idf 高于 df=3 的常见词。"""
        n = 3
        idf_apple = self._idf(n, 3)   # df=3(常见)
        idf_cherry = self._idf(n, 2)  # df=2(较常见)
        idf_date = self._idf(n, 1)    # df=1(稀有)

        assert idf_apple == pytest.approx(0.13353139, rel=APPROX_REL)
        assert idf_cherry == pytest.approx(0.47000363, rel=APPROX_REL)
        assert idf_date == pytest.approx(0.98082925, rel=APPROX_REL)
        assert idf_date > idf_cherry > idf_apple

    def test_tf_influence_same_dl(self) -> None:
        """同 dl 下,tf=2 得分高于 tf=1(BM25 饱和不线性,但仍单调增)。"""
        n, avgdl = 3, 4.0
        idf_apple = self._idf(n, 3)
        dl = 3

        score_tf1 = self._score_term(idf_apple, 1, dl, avgdl)
        score_tf2 = self._score_term(idf_apple, 2, dl, avgdl)

        assert score_tf1 == pytest.approx(0.15045791, rel=APPROX_REL)
        assert score_tf2 == pytest.approx(0.20742741, rel=APPROX_REL)
        assert score_tf2 > score_tf1

    def test_length_normalization(self) -> None:
        """同 tf=1,dl=3 文档得分高于 dl=6 文档(长度归一)。"""
        n, avgdl = 3, 4.0
        idf_cherry = self._idf(n, 2)

        score_short = self._score_term(idf_cherry, 1, 3, avgdl)  # chunk A
        score_long = self._score_term(idf_cherry, 1, 6, avgdl)   # chunk C

        assert score_short == pytest.approx(0.52958155, rel=APPROX_REL)
        assert score_long == pytest.approx(0.38367643, rel=APPROX_REL)
        assert score_short > score_long

    def test_query_apple_cherry_ranking(self) -> None:
        """Query 'apple cherry' → A > C > B。

        A 同时含 apple+cherry,B 缺 cherry,C 长文档降权。
        """
        n, avgdl = 3, 4.0
        idf_apple = self._idf(n, 3)
        idf_cherry = self._idf(n, 2)

        # chunk A: apple(tf=1,dl=3) + cherry(tf=1,dl=3)
        score_A = (
            self._score_term(idf_apple, 1, 3, avgdl)
            + self._score_term(idf_cherry, 1, 3, avgdl)
        )
        # chunk B: apple(tf=2,dl=3) + cherry(tf=0,dl=3)
        score_B = self._score_term(idf_apple, 2, 3, avgdl)
        # chunk C: apple(tf=1,dl=6) + cherry(tf=1,dl=6)
        score_C = (
            self._score_term(idf_apple, 1, 6, avgdl)
            + self._score_term(idf_cherry, 1, 6, avgdl)
        )

        assert score_A == pytest.approx(0.68003946, rel=APPROX_REL)
        assert score_B == pytest.approx(0.20742741, rel=APPROX_REL)
        assert score_C == pytest.approx(0.49268165, rel=APPROX_REL)
        assert score_A > score_C > score_B

    def test_rare_word_only_hits_one_doc(self) -> None:
        """Query 'date':只有 chunk C 含 date(df=1),A 和 B score=0。"""
        n, avgdl = 3, 4.0
        idf_date = self._idf(n, 1)

        score_C = self._score_term(idf_date, 1, 6, avgdl)
        assert score_C == pytest.approx(0.80067694, rel=APPROX_REL)


# ══════════════════════════════════════════════════════════════════════
# 集成测试:真 PG
# ══════════════════════════════════════════════════════════════════════


async def _get_bm25_store() -> PgBM25Index:
    engine = make_engine(TEST_DB_URL)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return PgBM25Index(factory)


@pytest.fixture
async def bm25_store(_run_kb_migrations):
    """BM25 索引 + 干净的 DB(TRUNCATE before/after)。"""
    from sqlalchemy import text

    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE kb_postings, kb_chunks, kb_documents CASCADE")
        )
    await engine.dispose()

    factory = async_sessionmaker(
        create_async_engine(TEST_DB_URL, poolclass=NullPool),
        class_=AsyncSession,
        expire_on_commit=False,
    )
    store = PgBM25Index(factory)
    yield store

    engine2 = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    async with engine2.begin() as conn:
        await conn.execute(
            text("TRUNCATE kb_postings, kb_chunks, kb_documents CASCADE")
        )
    await engine2.dispose()


async def _insert_doc_and_chunks(session_factory, chunks: list[Chunk]) -> None:
    """先插 DocumentRow + ChunkRow(embedding 全零占位),再 index_chunks。"""
    from anvil_kb.db import EMBEDDING_DIM

    zero_emb = [0.0] * EMBEDDING_DIM

    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    async with engine.begin() as conn:
        # Upsert document
        from sqlalchemy import text as sqla_text

        await conn.execute(
            sqla_text(
                "INSERT INTO kb_documents(id, title, source_name, content) "
                "VALUES (:id, :title, :sn, :content) ON CONFLICT DO NOTHING"
            ),
            {"id": str(_DOC_ID), "title": "test doc", "sn": "test_source", "content": "x"},
        )
        # Insert chunks
        for c in chunks:
            await conn.execute(
                sqla_text(
                    "INSERT INTO kb_chunks(id, document_id, seq, content, header_path, "
                    "start_offset, end_offset, embedding, token_count) "
                    "VALUES (:id, :doc_id, :seq, :content, :hp, :so, :eo, :emb, 0) "
                    "ON CONFLICT DO NOTHING"
                ),
                {
                    "id": str(c.id),
                    "doc_id": str(c.document_id),
                    "seq": c.seq,
                    "content": c.content,
                    "hp": c.header_path,
                    "so": c.start_offset,
                    "eo": c.end_offset,
                    "emb": str(zero_emb),
                },
            )
    await engine.dispose()


@pytest.mark.asyncio
async def test_index_and_search_ranking(bm25_store: PgBM25Index) -> None:
    """集成:index 3 chunks → query 'apple cherry' → A > C > B。"""
    chunks = [
        _make_chunk(_CHUNK_A_ID, CONTENT_A, seq=0),
        _make_chunk(_CHUNK_B_ID, CONTENT_B, seq=1),
        _make_chunk(_CHUNK_C_ID, CONTENT_C, seq=2),
    ]
    await _insert_doc_and_chunks(None, chunks)
    await bm25_store.index_chunks(chunks)

    results = await bm25_store.search("apple cherry", k=10)
    assert len(results) == 3

    ids = [r.chunk.id for r in results]
    assert ids[0] == _CHUNK_A_ID, "A 应排第一(同时含 apple+cherry,短文档)"
    assert ids[1] == _CHUNK_C_ID, "C 应排第二"
    assert ids[2] == _CHUNK_B_ID, "B 应排第三(缺 cherry)"

    # 分数精度
    assert results[0].score == pytest.approx(0.68003946, rel=APPROX_REL)
    assert results[1].score == pytest.approx(0.49268165, rel=APPROX_REL)
    assert results[2].score == pytest.approx(0.20742741, rel=APPROX_REL)


@pytest.mark.asyncio
async def test_search_unknown_query_returns_empty(bm25_store: PgBM25Index) -> None:
    """查询词完全不在库 → 返回 []。"""
    chunks = [_make_chunk(_CHUNK_A_ID, CONTENT_A)]
    await _insert_doc_and_chunks(None, chunks)
    await bm25_store.index_chunks(chunks)

    results = await bm25_store.search("xyzzy_nonexistent", k=5)
    assert results == []


@pytest.mark.asyncio
async def test_cascade_delete_cleans_postings(bm25_store: PgBM25Index) -> None:
    """删除 Document 后,kb_postings 通过 CASCADE 自动清空。

    双重断言:
    1. 直接查 kb_postings 行数 == 0(不依赖 search() early-exit 假绿)
    2. search() 也返回 []
    """
    from sqlalchemy import func, select, text

    from anvil_kb.db import PostingRow

    chunks = [
        _make_chunk(_CHUNK_A_ID, CONTENT_A),
        _make_chunk(_CHUNK_B_ID, CONTENT_B),
    ]
    await _insert_doc_and_chunks(None, chunks)
    await bm25_store.index_chunks(chunks)

    # 确认能搜到
    results = await bm25_store.search("apple", k=5)
    assert len(results) > 0

    # 删除 document(CASCADE: kb_chunks → kb_postings)
    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(
            text(f"DELETE FROM kb_documents WHERE id = '{_DOC_ID}'")
        )
    await engine.dispose()

    # 断言 1: 直接查 kb_postings 行数为 0(不依赖 search 的 n_total==0 早退)
    from sqlalchemy.ext.asyncio import (
        AsyncSession,  # noqa: PLC0415
        async_sessionmaker,
    )
    from sqlalchemy.ext.asyncio import create_async_engine as _cae

    verify_engine = _cae(TEST_DB_URL, poolclass=NullPool)
    verify_factory = async_sessionmaker(verify_engine, class_=AsyncSession, expire_on_commit=False)
    async with verify_factory() as session:
        count_row = await session.execute(
            select(func.count()).select_from(PostingRow)
        )
        posting_count = count_row.scalar_one()
    await verify_engine.dispose()
    assert posting_count == 0, f"kb_postings 应为空,实际剩 {posting_count} 行"

    # 断言 2: search 也返回 []
    results2 = await bm25_store.search("apple", k=5)
    assert results2 == []


@pytest.mark.asyncio
async def test_digit_token_search(bm25_store: PgBM25Index) -> None:
    """含数字的中文文本能被正确分词并命中。

    语料: '等待期90天,缴费2024年'
    先打印/断言 tokenize 实际切分结果,再验证 search 能命中。
    """
    content = "等待期90天,缴费2024年"
    tokens = tokenize(content)

    # 打印实际切分结果以便调试/复查
    print(f"\ntokenize({content!r}) = {tokens}")

    # 切分结果必须包含数字相关 token(jieba 会切出 "90", "天", "2024", "年" 等)
    # 以实际切分为准:只要含有 "90" 或 "90天" 即可
    has_90 = any("90" in t for t in tokens)
    has_2024 = any("2024" in t for t in tokens)
    assert has_90, f"tokenize 结果 {tokens} 中未找到包含 '90' 的 token"
    assert has_2024, f"tokenize 结果 {tokens} 中未找到包含 '2024' 的 token"

    # 插入并索引该 chunk
    digit_doc_id = uuid.UUID("dddddddd-0000-0000-0000-000000000001")
    digit_chunk_id = uuid.UUID("dddddddd-0000-0000-0000-000000000002")
    digit_chunk = Chunk(
        id=digit_chunk_id,
        document_id=digit_doc_id,
        seq=0,
        content=content,
        header_path="",
        start_offset=0,
        end_offset=len(content),
        embedding=None,
    )

    from sqlalchemy import text as sqla_text  # noqa: PLC0415
    from sqlalchemy.ext.asyncio import create_async_engine as _cae

    from anvil_kb.db import EMBEDDING_DIM

    ins_engine = _cae(TEST_DB_URL, poolclass=NullPool)
    async with ins_engine.begin() as conn:
        await conn.execute(
            sqla_text(
                "INSERT INTO kb_documents(id, title, source_name, content) "
                "VALUES (:id, :title, :sn, :content) ON CONFLICT DO NOTHING"
            ),
            {
                "id": str(digit_doc_id),
                "title": "digit doc",
                "sn": "digit_source",
                "content": content,
            },
        )
        zero_emb = [0.0] * EMBEDDING_DIM
        await conn.execute(
            sqla_text(
                "INSERT INTO kb_chunks(id, document_id, seq, content, header_path, "
                "start_offset, end_offset, embedding, token_count) "
                "VALUES (:id, :doc_id, :seq, :content, :hp, :so, :eo, :emb, 0) "
                "ON CONFLICT DO NOTHING"
            ),
            {
                "id": str(digit_chunk_id),
                "doc_id": str(digit_doc_id),
                "seq": 0,
                "content": content,
                "hp": "",
                "so": 0,
                "eo": len(content),
                "emb": str(zero_emb),
            },
        )
    await ins_engine.dispose()

    await bm25_store.index_chunks([digit_chunk])

    # 找出实际含 "90" 的 token 作为查询词
    token_90 = next(t for t in tokens if "90" in t)
    token_2024 = next(t for t in tokens if "2024" in t)

    # search("90天") 或 search("90") 应能命中
    results_90 = await bm25_store.search(token_90, k=5)
    assert len(results_90) > 0, f"search({token_90!r}) 应命中,tokens={tokens}"
    assert results_90[0].chunk.id == digit_chunk_id

    # search("2024") 应能命中
    results_2024 = await bm25_store.search(token_2024, k=5)
    assert len(results_2024) > 0, f"search({token_2024!r}) 应命中,tokens={tokens}"
    assert results_2024[0].chunk.id == digit_chunk_id


@pytest.mark.asyncio
async def test_idempotent_re_index(bm25_store: PgBM25Index) -> None:
    """重复 index 同 chunk 不翻倍 tf,分数保持一致。"""
    chunks = [
        _make_chunk(_CHUNK_A_ID, CONTENT_A),
        _make_chunk(_CHUNK_B_ID, CONTENT_B),
        _make_chunk(_CHUNK_C_ID, CONTENT_C),
    ]
    await _insert_doc_and_chunks(None, chunks)

    # 第一次 index
    await bm25_store.index_chunks(chunks)
    results1 = await bm25_store.search("apple", k=3)

    # 第二次 index(幂等)
    await bm25_store.index_chunks(chunks)
    results2 = await bm25_store.search("apple", k=3)

    # 分数相同
    for r1, r2 in zip(results1, results2, strict=True):
        assert r1.chunk.id == r2.chunk.id
        assert r1.score == pytest.approx(r2.score, rel=APPROX_REL)


@pytest.mark.asyncio
async def test_empty_db_returns_empty(bm25_store: PgBM25Index) -> None:
    """空库(N=0)→ search 返回 []。"""
    results = await bm25_store.search("apple", k=5)
    assert results == []


@pytest.mark.asyncio
async def test_context_prefix_words_searchable(bm25_store: PgBM25Index) -> None:
    """context_prefix 含独特词 'XQZW定位词' → search 该词命中该 chunk(真 PG)。

    index_chunks 在 prefix 非空时拼接 prefix + '\\n' + content 再分词,
    前缀词应进入倒排索引。
    """
    from sqlalchemy import text as sqla_text
    from sqlalchemy.ext.asyncio import create_async_engine as _cae

    from anvil_kb.db import EMBEDDING_DIM

    prefix_chunk_id = uuid.UUID("eeeeeeee-0000-0000-0000-000000000001")
    other_chunk_id = uuid.UUID("ffffffff-0000-0000-0000-000000000001")
    doc_id = uuid.UUID("11111111-0000-0000-0000-000000000001")
    unique_term = "XQZW定位词"

    prefix_chunk = Chunk(
        id=prefix_chunk_id,
        document_id=doc_id,
        seq=0,
        content="这是一段关于保险条款的普通内容",
        header_path="",
        start_offset=0,
        end_offset=20,
        embedding=None,
        context_prefix=f"本段在文档中描述了{unique_term}相关的核心概念",
    )
    other_chunk = Chunk(
        id=other_chunk_id,
        document_id=doc_id,
        seq=1,
        content="另一段完全不同的内容，不含特殊词汇",
        header_path="",
        start_offset=20,
        end_offset=40,
        embedding=None,
        context_prefix="",
    )

    # Insert rows into DB (required before index_chunks)
    ins_engine = _cae(TEST_DB_URL, poolclass=NullPool)
    async with ins_engine.begin() as conn:
        await conn.execute(
            sqla_text(
                "INSERT INTO kb_documents(id, title, source_name, content) "
                "VALUES (:id, :title, :sn, :content) ON CONFLICT DO NOTHING"
            ),
            {
                "id": str(doc_id),
                "title": "prefix test doc",
                "sn": "prefix_test_src",
                "content": "x",
            },
        )
        zero_emb = [0.0] * EMBEDDING_DIM
        for c in [prefix_chunk, other_chunk]:
            await conn.execute(
                sqla_text(
                    "INSERT INTO kb_chunks(id, document_id, seq, content, header_path, "
                    "start_offset, end_offset, embedding, token_count, context_prefix) "
                    "VALUES (:id, :doc_id, :seq, :content, :hp, :so, :eo, :emb, 0, :ctx) "
                    "ON CONFLICT DO NOTHING"
                ),
                {
                    "id": str(c.id),
                    "doc_id": str(c.document_id),
                    "seq": c.seq,
                    "content": c.content,
                    "hp": c.header_path,
                    "so": c.start_offset,
                    "eo": c.end_offset,
                    "emb": str(zero_emb),
                    "ctx": c.context_prefix,
                },
            )
    await ins_engine.dispose()

    await bm25_store.index_chunks([prefix_chunk, other_chunk])

    # Search the unique term from the prefix — should hit prefix_chunk only
    results = await bm25_store.search(unique_term, k=5)
    assert len(results) > 0, f"search({unique_term!r}) 应命中前缀词所在的 chunk"
    assert results[0].chunk.id == prefix_chunk_id, (
        f"应命中 prefix_chunk({prefix_chunk_id}),实际得 {results[0].chunk.id}"
    )
