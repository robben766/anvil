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


# ── Fake sparse index ─────────────────────────────────────────────────────────


class FakeSparseIndex:
    """Fake SparseIndex that records which chunks it received."""

    def __init__(self) -> None:
        self.received_chunks: list | None = None
        self.call_count: int = 0

    async def index_chunks(self, chunks: list) -> None:
        self.call_count += 1
        self.received_chunks = list(chunks)

    async def search(self, query: str, k: int) -> list:
        return []


# ── Dual-write tests ──────────────────────────────────────────────────────────


async def test_sparse_index_receives_same_chunks_as_store(kb_store):
    """When sparse_index is provided, index_chunks receives the exact same
    Chunk objects (same ids) that were upserted to the vector store."""
    from anvil_kb.ingest.pipeline import ingest_markdown

    embedder = FakeEmbedder()
    sparse = FakeSparseIndex()

    doc, n = await ingest_markdown(
        title="双写测试",
        source_name="test/dual-write",
        text=TWO_SECTION_MD,
        embedder=embedder,
        store=kb_store,
        sparse_index=sparse,
    )

    assert sparse.call_count == 1, "index_chunks must be called exactly once"
    assert sparse.received_chunks is not None
    assert len(sparse.received_chunks) == n, "sparse must receive all chunks"

    stored_ids = {c.id for c in sparse.received_chunks}
    # Verify these chunks actually exist in the vector store
    all_hits = await kb_store.search(_vec(0), k=100)
    stored_vector_ids = {h.chunk.id for h in all_hits}
    # All chunks sent to sparse must be in the vector store (same set)
    assert stored_ids == stored_vector_ids, (
        "sparse_index must receive the same chunk ids as the vector store"
    )


async def test_no_sparse_index_skips_dual_write(kb_store):
    """Without sparse_index, ingest works as before (no error, no sparse call)."""
    from anvil_kb.ingest.pipeline import ingest_markdown

    embedder = FakeEmbedder()
    # sparse_index defaults to None — no FakeSparseIndex provided
    doc, n = await ingest_markdown(
        title="非双写",
        source_name="test/no-sparse",
        text=TWO_SECTION_MD,
        embedder=embedder,
        store=kb_store,
    )
    assert n > 0


async def test_sparse_index_called_after_vector_store(kb_store):
    """sparse_index.index_chunks is called AFTER store.upsert_chunks
    (dual-write order: upsert→index)."""
    from anvil_kb.ingest.pipeline import ingest_markdown

    call_log: list[str] = []

    class OrderedFakeStore:
        async def upsert_chunks(self, doc, chunks):
            call_log.append("upsert")

        async def search(self, query_vector, k):
            return []

        async def delete_document(self, document_id):
            pass

    class OrderedFakeSparse:
        async def index_chunks(self, chunks):
            call_log.append("index")

        async def search(self, query, k):
            return []

    await ingest_markdown(
        title="顺序测试",
        source_name="test/order",
        text=TWO_SECTION_MD,
        embedder=FakeEmbedder(),
        store=OrderedFakeStore(),
        sparse_index=OrderedFakeSparse(),
    )

    assert call_log == ["upsert", "index"], (
        f"Expected ['upsert', 'index'], got {call_log}"
    )


# ── Contextual Retrieval enrichment tests ─────────────────────────────────────


def _make_enrich_chat(responses: list[str]):
    """Build a fake chat returning given responses in order."""
    from types import SimpleNamespace

    idx = 0
    calls: list[dict] = []

    async def fake_chat(model: str, messages: list[dict], **kw):
        nonlocal idx
        calls.append({"model": model, "messages": messages, **kw})
        resp = responses[idx] if idx < len(responses) else ""
        idx += 1
        return SimpleNamespace(content=resp)

    fake_chat.calls = calls  # type: ignore[attr-defined]
    return fake_chat


async def test_enrich_chat_sets_context_prefix(kb_store):
    """enrich_chat 非 None → Chunk.context_prefix 按 enrich_chunks 返回值设置。"""
    from anvil_kb.ingest.pipeline import ingest_markdown

    drafts_expected = chunk_markdown(TWO_SECTION_MD, size=600, overlap=100)
    fake_responses = [f"定位上下文_{i}" for i in range(len(drafts_expected))]
    enrich_chat = _make_enrich_chat(fake_responses)

    embedder = FakeEmbedder()
    # Capture chunks via FakeSparseIndex
    sparse = FakeSparseIndex()

    await ingest_markdown(
        title="富集测试",
        source_name="test/enrich-prefix",
        text=TWO_SECTION_MD,
        embedder=embedder,
        store=kb_store,
        sparse_index=sparse,
        enrich_chat=enrich_chat,
    )

    assert sparse.received_chunks is not None
    for i, chunk in enumerate(sparse.received_chunks):
        assert chunk.context_prefix == fake_responses[i], (
            f"chunk[{i}].context_prefix mismatch"
        )


async def test_enrich_chat_embedding_input_uses_prefix(kb_store):
    """embedding 输入 = prefix + '\\n' + content(逐条验证)。"""
    from anvil_kb.ingest.pipeline import ingest_markdown

    drafts_expected = chunk_markdown(TWO_SECTION_MD, size=600, overlap=100)
    fake_responses = [f"前缀_{i}" for i in range(len(drafts_expected))]
    enrich_chat = _make_enrich_chat(fake_responses)

    captured_inputs: list[list[str]] = []

    class CapturingEmbedder:
        call_count = 0

        @property
        def dim(self) -> int:
            return 512

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.call_count += 1
            captured_inputs.append(list(texts))
            return [_vec(i) for i in range(len(texts))]

        def embed_query(self, text: str) -> list[float]:
            return _vec(0)

    embedder = CapturingEmbedder()
    sparse = FakeSparseIndex()

    await ingest_markdown(
        title="嵌入输入测试",
        source_name="test/embed-input",
        text=TWO_SECTION_MD,
        embedder=embedder,
        store=kb_store,
        sparse_index=sparse,
        enrich_chat=enrich_chat,
    )

    assert len(captured_inputs) == 1  # single batch call
    inputs = captured_inputs[0]
    for i, draft in enumerate(drafts_expected):
        expected_input = f"前缀_{i}\n{draft.content}"
        assert inputs[i] == expected_input, (
            f"embed input[{i}] expected {expected_input!r}, got {inputs[i]!r}"
        )


async def test_enrich_chat_none_prefix_empty_no_behavior_change(kb_store):
    """enrich_chat=None → context_prefix 全 ''，embedding 输入 == content(零行为变化)。"""
    from anvil_kb.ingest.pipeline import ingest_markdown

    drafts_expected = chunk_markdown(TWO_SECTION_MD, size=600, overlap=100)
    captured_inputs: list[list[str]] = []

    class CapturingEmbedder:
        @property
        def dim(self) -> int:
            return 512

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            captured_inputs.append(list(texts))
            return [_vec(i) for i in range(len(texts))]

        def embed_query(self, text: str) -> list[float]:
            return _vec(0)

    sparse = FakeSparseIndex()

    await ingest_markdown(
        title="无富集测试",
        source_name="test/no-enrich",
        text=TWO_SECTION_MD,
        embedder=CapturingEmbedder(),
        store=kb_store,
        sparse_index=sparse,
        enrich_chat=None,
    )

    # Embedding inputs must equal content exactly
    inputs = captured_inputs[0]
    for i, draft in enumerate(drafts_expected):
        assert inputs[i] == draft.content, (
            f"Without enrich, embed input[{i}] should equal content"
        )

    # All context_prefix must be ""
    assert sparse.received_chunks is not None
    for chunk in sparse.received_chunks:
        assert chunk.context_prefix == "", "enrich_chat=None → context_prefix must be ''"


async def test_enrich_chat_fail_open_does_not_fail_ingest(kb_store):
    """富集失败(enrich_chat 抛异常)不许让 ingest 失败;prefix 降级为 ''。"""
    from types import SimpleNamespace

    from anvil_kb.ingest.pipeline import ingest_markdown

    call_count = 0

    async def sometimes_failing_chat(model, messages, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("模拟富集失败")
        return SimpleNamespace(content=f"ctx_{call_count}")

    sparse = FakeSparseIndex()

    # Must not raise
    doc, n = await ingest_markdown(
        title="fail-open 测试",
        source_name="test/fail-open",
        text=TWO_SECTION_MD,
        embedder=FakeEmbedder(),
        store=kb_store,
        sparse_index=sparse,
        enrich_chat=sometimes_failing_chat,
    )
    assert n > 0

    # First chunk's context_prefix should be "" (failed), rest should have values
    assert sparse.received_chunks is not None
    assert sparse.received_chunks[0].context_prefix == ""


async def test_citation_fields_unchanged_after_enrich(kb_store):
    """富集后 start_offset/end_offset/content 与未富集一致(引用高亮硬约束)。"""
    from anvil_kb.ingest.pipeline import ingest_markdown

    drafts_expected = chunk_markdown(TWO_SECTION_MD, size=600, overlap=100)
    n = len(drafts_expected)
    fake_responses = [f"ctx_{i}" for i in range(n)]
    enrich_chat = _make_enrich_chat(fake_responses)

    sparse_with = FakeSparseIndex()
    sparse_without = FakeSparseIndex()

    await ingest_markdown(
        title="富集对比-有",
        source_name="test/cite-with-enrich",
        text=TWO_SECTION_MD,
        embedder=FakeEmbedder(),
        store=kb_store,
        sparse_index=sparse_with,
        enrich_chat=enrich_chat,
    )

    await ingest_markdown(
        title="富集对比-无",
        source_name="test/cite-without-enrich",
        text=TWO_SECTION_MD,
        embedder=FakeEmbedder(),
        store=kb_store,
        sparse_index=sparse_without,
        enrich_chat=None,
    )

    assert sparse_with.received_chunks is not None
    assert sparse_without.received_chunks is not None

    chunks_with = sorted(sparse_with.received_chunks, key=lambda c: c.seq)
    chunks_without = sorted(sparse_without.received_chunks, key=lambda c: c.seq)

    for cw, cno in zip(chunks_with, chunks_without, strict=True):
        assert cw.content == cno.content, f"seq={cw.seq}: content changed after enrich"
        assert cw.start_offset == cno.start_offset, f"seq={cw.seq}: start_offset changed"
        assert cw.end_offset == cno.end_offset, f"seq={cw.seq}: end_offset changed"
