"""Tests for anvil_kb.ingest.pipeline.ingest_markdown.

TDD Step 1: write tests that FAIL before implementation.
"""
from __future__ import annotations

import pytest

from anvil_kb.ingest.chunker import chunk_markdown

# ── Fake embedder ─────────────────────────────────────────────────────────────


def _vec(i: int) -> list[float]:
    """One-hot vector of dim 512: position i is 1.0, rest 0.0."""
    v = [0.0] * 512
    v[i % 512] = 1.0
    return v


class FakeEmbedder:
    """Fake embedder: returns one-hot vectors; records call count."""

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def dim(self) -> int:
        return 512

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [_vec(i) for i in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        return _vec(0)


# ── Test markdown fixture ─────────────────────────────────────────────────────

TWO_SECTION_MD = """\
# 条款总览
本保险产品适用以下条款。

## 等待期
等待期为90天。
意外伤害不受等待期限制。

## 免赔额
每次就诊免赔额为100元。
年度累计免赔额为500元。
"""


# ── Tests ─────────────────────────────────────────────────────────────────────


async def test_returns_doc_and_chunk_count(kb_store):
    """ingest_markdown returns (Document, chunk_count) matching chunk_markdown output."""
    from anvil_kb.ingest.pipeline import ingest_markdown

    embedder = FakeEmbedder()
    doc, n = await ingest_markdown(
        title="测试文档",
        source_name="test/two-sections",
        text=TWO_SECTION_MD,
        embedder=embedder,
        store=kb_store,
        size=600,
        overlap=100,
    )
    expected_drafts = chunk_markdown(TWO_SECTION_MD, size=600, overlap=100)
    assert n == len(expected_drafts)
    assert doc.title == "测试文档"
    assert doc.source_name == "test/two-sections"


async def test_embed_called_exactly_once(kb_store):
    """embedder.embed_texts must be called exactly once (batch, not per-chunk)."""
    from anvil_kb.ingest.pipeline import ingest_markdown

    embedder = FakeEmbedder()
    await ingest_markdown(
        title="批量测试",
        source_name="test/batch-check",
        text=TWO_SECTION_MD,
        embedder=embedder,
        store=kb_store,
    )
    assert embedder.call_count == 1


async def test_chunks_stored_match_drafts(kb_store):
    """Stored chunks have content/start_offset/end_offset/header_path/seq matching drafts."""
    from anvil_kb.ingest.pipeline import ingest_markdown

    embedder = FakeEmbedder()
    doc, _ = await ingest_markdown(
        title="字段对比",
        source_name="test/field-match",
        text=TWO_SECTION_MD,
        embedder=embedder,
        store=kb_store,
    )
    expected_drafts = chunk_markdown(TWO_SECTION_MD, size=600, overlap=100)

    # Use search with each one-hot vector to retrieve stored chunks
    # Since embedder returns _vec(i) for position i, _vec(0) most similar to chunk 0
    hits_by_seq: dict[int, object] = {}
    for i in range(len(expected_drafts)):
        results = await kb_store.search(_vec(i), k=1)
        assert results, f"No result for chunk {i}"
        hits_by_seq[results[0].chunk.seq] = results[0].chunk

    assert len(hits_by_seq) == len(expected_drafts)

    for draft in expected_drafts:
        chunk = hits_by_seq[draft.seq]
        assert chunk.content == draft.content, f"seq={draft.seq}: content mismatch"
        assert chunk.start_offset == draft.start_offset, f"seq={draft.seq}: start_offset mismatch"
        assert chunk.end_offset == draft.end_offset, f"seq={draft.seq}: end_offset mismatch"
        assert chunk.header_path == draft.header_path, f"seq={draft.seq}: header_path mismatch"
        assert chunk.document_id == doc.id, f"seq={draft.seq}: document_id mismatch"


async def test_empty_text_raises_value_error(kb_store):
    """Empty or whitespace-only text raises ValueError."""
    from anvil_kb.ingest.pipeline import ingest_markdown

    embedder = FakeEmbedder()
    with pytest.raises(ValueError, match="no chunks"):
        await ingest_markdown(
            title="空文档",
            source_name="test/empty",
            text="",
            embedder=embedder,
            store=kb_store,
        )

    with pytest.raises(ValueError, match="no chunks"):
        await ingest_markdown(
            title="空白文档",
            source_name="test/whitespace",
            text="   \n\n   ",
            embedder=embedder,
            store=kb_store,
        )


async def test_idempotent_second_ingest_replaces(kb_store):
    """Ingesting same source_name twice does not accumulate chunks; second wins."""
    from anvil_kb.ingest.pipeline import ingest_markdown

    embedder = FakeEmbedder()
    source = "test/idempotent"

    first_md = """\
## 第一版
这是第一次入库的内容，有一些文字。
"""
    second_md = TWO_SECTION_MD  # more chunks

    _, n1 = await ingest_markdown(
        title="v1", source_name=source, text=first_md,
        embedder=embedder, store=kb_store,
    )

    embedder2 = FakeEmbedder()
    doc2, n2 = await ingest_markdown(
        title="v2", source_name=source, text=second_md,
        embedder=embedder2, store=kb_store,
    )

    # Search should only return chunks from doc2
    all_hits = await kb_store.search(_vec(0), k=100)
    assert len(all_hits) == n2, (
        f"Expected {n2} chunks after re-ingest, got {len(all_hits)}"
    )
    assert all(h.chunk.document_id == doc2.id for h in all_hits), \
        "All chunks should belong to the second document"
