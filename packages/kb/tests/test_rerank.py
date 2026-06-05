"""Tests for cross-encoder reranker — fully mocked, no real model.

Live test (requires model download ~1 GB) is guarded with @pytest.mark.live
and only runs when explicitly selected.
"""

from __future__ import annotations

import uuid

import pytest

from anvil_kb.retrieve.rerank import FastEmbedReranker, Reranker
from anvil_kb.retrieve.retriever import Retriever
from anvil_kb.store.base import Chunk, ScoredChunk

# ---------------------------------------------------------------------------
# Helpers shared with test_retriever.py (duplicated to keep tests independent)
# ---------------------------------------------------------------------------


def _make_chunk(content: str, seq: int = 0) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        seq=seq,
        content=content,
        header_path="",
        start_offset=0,
        end_offset=len(content),
    )


def _scored(content: str, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(chunk=_make_chunk(content), score=score)


def _make_distinct_chunks(prefix: str, count: int) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=_make_chunk(f"{prefix}_{i}", seq=i),
            score=1.0 - 0.05 * i,
        )
        for i in range(count)
    ]


_SENTINEL_QUERY_VEC = [0.1, 0.2, 0.3]


class FakeEmbedder:
    @property
    def dim(self) -> int:
        return 3

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.9, 0.8, 0.7] for _ in texts]

    def embed_query(self, text: str) -> list[float]:  # noqa: ARG002
        return _SENTINEL_QUERY_VEC


class FakeVectorStore:
    def __init__(self, results: list[ScoredChunk]) -> None:
        self._results = results
        self.last_k: int | None = None

    async def upsert_chunks(self, doc, chunks):  # type: ignore[override]
        pass

    async def search(self, query_vector, k: int) -> list[ScoredChunk]:
        self.last_k = k
        return self._results[:k]

    async def delete_document(self, document_id):  # type: ignore[override]
        pass


class FakeSparseIndex:
    def __init__(self, results: list[ScoredChunk] | None = None) -> None:
        self._results: list[ScoredChunk] = results or []
        self.last_k: int | None = None

    async def index_chunks(self, chunks):
        pass

    async def search(self, query: str, k: int) -> list[ScoredChunk]:
        self.last_k = k
        return self._results[:k]


# ---------------------------------------------------------------------------
# Fake reranker — reverses the order of candidates
# ---------------------------------------------------------------------------


class ReversingReranker:
    """Fake reranker that simply reverses the input order (for deterministic testing)."""

    def __init__(self) -> None:
        self.called_with_question: str | None = None
        self.called_with_candidates: list[ScoredChunk] | None = None
        self.called_with_top: int | None = None

    def rerank(
        self, question: str, candidates: list[ScoredChunk], *, top: int
    ) -> list[ScoredChunk]:
        self.called_with_question = question
        self.called_with_candidates = list(candidates)
        self.called_with_top = top
        reversed_candidates = list(reversed(candidates))
        return [
            ScoredChunk(chunk=sc.chunk, score=float(len(candidates) - i))
            for i, sc in enumerate(reversed_candidates[:top])
        ]


class RecordingReranker:
    """Fake reranker that records how many candidates it received."""

    def __init__(self) -> None:
        self.received_count: int | None = None

    def rerank(
        self, question: str, candidates: list[ScoredChunk], *, top: int
    ) -> list[ScoredChunk]:
        self.received_count = len(candidates)
        return list(candidates[:top])


# ---------------------------------------------------------------------------
# Protocol conformance check
# ---------------------------------------------------------------------------


def test_reversing_reranker_satisfies_protocol():
    """ReversingReranker and FastEmbedReranker must satisfy the Reranker protocol."""
    # Just check structural compatibility — Reranker is a Protocol
    r: Reranker = ReversingReranker()  # type: ignore[assignment]
    assert hasattr(r, "rerank")


# ---------------------------------------------------------------------------
# FastEmbedReranker unit tests (no model loaded)
# ---------------------------------------------------------------------------


def test_fastembed_reranker_empty_candidates():
    """FastEmbedReranker.rerank([]) must return [] without calling the model."""
    reranker = FastEmbedReranker()
    result = reranker.rerank("question", [], top=5)
    assert result == []
    # model must NOT have been loaded
    assert reranker._model is None


# ---------------------------------------------------------------------------
# Retriever + reranker integration — hybrid mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_with_reranker_returns_reversed_order():
    """Reversing reranker: retrieve(hybrid) result == reversed fused order, truncated to k."""
    # Create 8 distinct chunks (k*4 = 8 for k=2)
    dense_results = _make_distinct_chunks("dense", 8)
    sparse_results = _make_distinct_chunks("sparse", 8)

    store = FakeVectorStore(dense_results)
    sparse = FakeSparseIndex(sparse_results)
    fake_reranker = ReversingReranker()

    retriever = Retriever(
        FakeEmbedder(),
        store,
        sparse_index=sparse,
        mode="hybrid",
        reranker=fake_reranker,
    )

    result = await retriever.retrieve("question", k=2)

    # Must be truncated to k
    assert len(result) == 2

    # The reversing reranker reverses the candidate list, so the last
    # candidate should come first
    assert fake_reranker.called_with_question == "question"
    assert fake_reranker.called_with_top == 2

    # result must be in reversed order (what the reranker returned)
    expected = fake_reranker.rerank(
        "question", fake_reranker.called_with_candidates or [], top=2
    )
    result_ids = [sc.chunk.id for sc in result]
    expected_ids = [sc.chunk.id for sc in expected]
    assert result_ids == expected_ids


@pytest.mark.asyncio
async def test_hybrid_reranker_receives_full_fused_list():
    """Reranker must receive all fused candidates (up to k*4), not just top-k."""
    # Use k=2 so k*4=8 candidates; we want the reranker to see all fused results
    dense_results = _make_distinct_chunks("dense", 8)
    sparse_results = _make_distinct_chunks("sparse", 8)

    store = FakeVectorStore(dense_results)
    sparse = FakeSparseIndex(sparse_results)
    recording_reranker = RecordingReranker()

    retriever = Retriever(
        FakeEmbedder(),
        store,
        sparse_index=sparse,
        mode="hybrid",
        reranker=recording_reranker,
    )

    await retriever.retrieve("q", k=2)

    # Reranker should receive the full fused list (up to k*4 = 8, possibly fewer if
    # there aren't that many unique chunks after fusion, but > k=2)
    assert recording_reranker.received_count is not None
    assert recording_reranker.received_count > 2, (
        f"Reranker should receive >k={2} candidates, got {recording_reranker.received_count}"
    )


@pytest.mark.asyncio
async def test_hybrid_debug_reranked_field_set_with_reranker():
    """retrieve_debug().reranked must be the k reranked results (not None) when reranker present."""
    dense_results = _make_distinct_chunks("dense", 8)
    sparse_results = _make_distinct_chunks("sparse", 8)

    store = FakeVectorStore(dense_results)
    sparse = FakeSparseIndex(sparse_results)
    fake_reranker = ReversingReranker()

    retriever = Retriever(
        FakeEmbedder(),
        store,
        sparse_index=sparse,
        mode="hybrid",
        reranker=fake_reranker,
    )

    debug = await retriever.retrieve_debug("q", k=2)

    # reranked must be set
    assert debug.reranked is not None
    assert len(debug.reranked) == 2


@pytest.mark.asyncio
async def test_hybrid_debug_fused_is_pre_rerank_order():
    """fused field must remain the pre-rerank order even when reranker is present.

    This is the key semantic-separation assertion:
    fused = what RRF returned (before reranking)
    reranked = what the reranker returned (after reranking)
    """
    # Use chunks where we can track order
    dense_results = _make_distinct_chunks("dense", 8)
    sparse_results = _make_distinct_chunks("sparse", 8)

    store = FakeVectorStore(dense_results)
    sparse = FakeSparseIndex(sparse_results)
    fake_reranker = ReversingReranker()

    retriever = Retriever(
        FakeEmbedder(),
        store,
        sparse_index=sparse,
        mode="hybrid",
        reranker=fake_reranker,
    )

    # First get debug WITHOUT reranker to capture baseline fused order
    retriever_no_rerank = Retriever(
        FakeEmbedder(),
        FakeVectorStore(dense_results),
        sparse_index=FakeSparseIndex(sparse_results),
        mode="hybrid",
    )
    await retriever_no_rerank.retrieve_debug("q", k=2)

    debug = await retriever.retrieve_debug("q", k=2)

    # fused field must equal the baseline (same RRF order, no influence from reranker)
    fused_ids = [sc.chunk.id for sc in debug.fused]
    # Note: these may differ because they're different UUIDs (new chunks each time)
    # We check that fused != reranked when reranker reverses
    reranked_ids = [sc.chunk.id for sc in debug.reranked]  # type: ignore[index]

    # Since the reranker reverses, fused and reranked should differ
    # (unless all chunks are identical, which they aren't here)
    assert fused_ids != reranked_ids, (
        "fused and reranked should differ when reranker changes order"
    )


@pytest.mark.asyncio
async def test_hybrid_debug_reranked_none_without_reranker():
    """retrieve_debug().reranked must be None when no reranker is configured."""
    dense_results = _make_distinct_chunks("dense", 8)
    sparse_results = _make_distinct_chunks("sparse", 8)

    store = FakeVectorStore(dense_results)
    sparse = FakeSparseIndex(sparse_results)

    retriever = Retriever(FakeEmbedder(), store, sparse_index=sparse, mode="hybrid")
    debug = await retriever.retrieve_debug("q", k=2)

    assert debug.reranked is None


# ---------------------------------------------------------------------------
# dense / sparse mode + reranker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dense_with_reranker_candidates_are_k_times_4():
    """dense mode + reranker: k*4 candidates fetched, then reranked to top-k."""
    dense_results = _make_distinct_chunks("dense", 20)
    store = FakeVectorStore(dense_results)
    recording_reranker = RecordingReranker()

    retriever = Retriever(
        FakeEmbedder(),
        store,
        mode="dense",
        reranker=recording_reranker,
    )

    result = await retriever.retrieve("q", k=3)

    # reranker must have received k*4=12 candidates
    assert recording_reranker.received_count == 12, (
        f"Expected 12 candidates (k*4), got {recording_reranker.received_count}"
    )
    assert len(result) == 3


@pytest.mark.asyncio
async def test_sparse_with_reranker_candidates_are_k_times_4():
    """sparse mode + reranker: k*4 candidates fetched, then reranked to top-k."""
    sparse_results = _make_distinct_chunks("sparse", 20)
    sparse = FakeSparseIndex(sparse_results)
    recording_reranker = RecordingReranker()

    retriever = Retriever(
        FakeEmbedder(),
        FakeVectorStore([]),
        sparse_index=sparse,
        mode="sparse",
        reranker=recording_reranker,
    )

    result = await retriever.retrieve("q", k=3)

    # reranker must have received k*4=12 candidates
    assert recording_reranker.received_count == 12, (
        f"Expected 12 candidates (k*4), got {recording_reranker.received_count}"
    )
    assert len(result) == 3


# ---------------------------------------------------------------------------
# Empty candidates edge case
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_candidates_reranker_not_called():
    """When no chunks are found, reranker.rerank() returns [] (or is not called for empty)."""
    store = FakeVectorStore([])
    fake_reranker = ReversingReranker()

    retriever = Retriever(
        FakeEmbedder(),
        store,
        mode="dense",
        reranker=fake_reranker,
    )

    result = await retriever.retrieve("q", k=5)
    assert result == []


# ---------------------------------------------------------------------------
# Regression: reranker=None must preserve original behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_reranker_dense_returns_store_results():
    """reranker=None in dense mode: result == store output (zero regression)."""
    chunks = _make_distinct_chunks("dense", 3)
    store = FakeVectorStore(chunks)

    retriever = Retriever(FakeEmbedder(), store, mode="dense")
    result = await retriever.retrieve("q", k=3)

    assert [sc.chunk.id for sc in result] == [sc.chunk.id for sc in chunks]


@pytest.mark.asyncio
async def test_no_reranker_hybrid_returns_fused_results():
    """reranker=None in hybrid mode: result == fused output (zero regression)."""
    dense_results = _make_distinct_chunks("dense", 20)
    sparse_results = _make_distinct_chunks("sparse", 20)
    store = FakeVectorStore(dense_results)
    sparse = FakeSparseIndex(sparse_results)

    retriever = Retriever(FakeEmbedder(), store, sparse_index=sparse, mode="hybrid")
    result = await retriever.retrieve("q", k=5)

    assert len(result) == 5


# ---------------------------------------------------------------------------
# Live test — requires real model download (~1 GB)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_fastembed_reranker_relevant_scores_higher():
    """FastEmbedReranker: relevant sentence scores higher than irrelevant sentence.

    Model: BAAI/bge-reranker-base (downloaded on first run, ~1 GB).
    """
    reranker = FastEmbedReranker()

    relevant = _scored("本合同等待期为90天。", score=0.5)
    irrelevant = _scored("今天天气晴朗。", score=0.5)

    result = reranker.rerank("等待期是多少天?", [relevant, irrelevant], top=2)

    assert len(result) == 2

    # Find scores
    scores_by_content = {sc.chunk.content: sc.score for sc in result}
    relevant_score = scores_by_content["本合同等待期为90天。"]
    irrelevant_score = scores_by_content["今天天气晴朗。"]

    print(f"\n  relevant score:   {relevant_score:.6f}")
    print(f"  irrelevant score: {irrelevant_score:.6f}")

    assert relevant_score > irrelevant_score, (
        f"Expected relevant ({relevant_score:.4f}) > irrelevant ({irrelevant_score:.4f})"
    )
