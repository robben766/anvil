"""Tri-mode retriever: dense / sparse / hybrid (default).

KB-M2 — Retriever now supports three modes:

dense:
    Pure vector-similarity search via the VectorStore.
    Behaviour identical to the original KB-M1 retriever.

sparse:
    Pure BM25 search via the SparseIndex (PgBM25Index or compatible).

hybrid (default):
    Both dense and sparse are called with k*4 candidates each.
    Results are fused with hand-rolled RRF (k=60) and the top-k
    fused results are returned.  No search is ever duplicated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from anvil_kb.retrieve.fusion import RetrievalDebug, rrf_fuse
from anvil_kb.store.base import ScoredChunk, SparseIndex, VectorStore

if TYPE_CHECKING:
    from anvil_kb.embed import Embedder


class Retriever:
    """Tri-mode retriever: dense | sparse | hybrid.

    Args:
        embedder:      Embedder used to convert questions to query vectors.
        vector_store:  Dense (pgvector) store.
        sparse_index:  BM25/sparse index.  Required for 'sparse' and 'hybrid'
                       modes; raises ValueError at construction time if absent.
        mode:          One of 'dense', 'sparse', 'hybrid' (default 'hybrid').

    Raises:
        ValueError: At construction time when mode in {'sparse','hybrid'} and
                    sparse_index is None.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        *,
        sparse_index: SparseIndex | None = None,
        mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
    ) -> None:
        if mode in ("sparse", "hybrid") and sparse_index is None:
            raise ValueError(
                f"mode={mode!r} requires a sparse_index but none was provided"
            )
        self._embedder = embedder
        self._vector_store = vector_store
        self._sparse_index = sparse_index
        self._mode = mode

    async def retrieve(self, question: str, k: int = 5) -> list[ScoredChunk]:
        """Return top-k chunks for *question* using the configured mode.

        hybrid path reuses a single internal search pass (no double retrieval).
        """
        if self._mode == "dense":
            return await self._dense_search(question, k)

        if self._mode == "sparse":
            assert self._sparse_index is not None  # guaranteed by __init__
            return await self._sparse_index.search(question, k)

        # hybrid — run once, fuse, return top-k
        debug = await self._hybrid_debug(question, k)
        return debug.fused

    async def retrieve_debug(self, question: str, k: int = 5) -> RetrievalDebug:
        """Always runs the full hybrid pipeline and returns a debug view.

        Even when mode != 'hybrid', the debug view shows all three paths.
        This is useful for diagnosing retrieval quality regardless of the
        configured production mode.
        """
        return await self._hybrid_debug(question, k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _dense_search(self, question: str, k: int) -> list[ScoredChunk]:
        query_vector = self._embedder.embed_query(question)
        return await self._vector_store.search(query_vector, k)

    async def _hybrid_debug(self, question: str, k: int) -> RetrievalDebug:
        """Run hybrid search once and return the full debug payload.

        Candidate expansion: both dense and sparse use k*4 candidates so the
        RRF pool is larger than the final top-k, giving fusion room to work.
        """
        assert self._sparse_index is not None  # guaranteed by mode check in __init__

        candidates = k * 4

        # ── 1. Fetch candidates from both indexes ────────────────────────────
        query_vector = self._embedder.embed_query(question)
        dense_candidates = await self._vector_store.search(query_vector, candidates)
        sparse_candidates = await self._sparse_index.search(question, candidates)

        # ── 2. Fuse with RRF ─────────────────────────────────────────────────
        fused = rrf_fuse([dense_candidates, sparse_candidates], k=60, top=k)

        # ── 3. Build contributions map ───────────────────────────────────────
        # contributions[chunk_id_str] = {'dense': rank | None, 'sparse': rank | None}
        dense_ranks: dict[str, int] = {
            str(sc.chunk.id): pos + 1
            for pos, sc in enumerate(dense_candidates)
        }
        sparse_ranks: dict[str, int] = {
            str(sc.chunk.id): pos + 1
            for pos, sc in enumerate(sparse_candidates)
        }

        all_ids = set(dense_ranks) | set(sparse_ranks)
        contributions: dict[str, dict[str, int | None]] = {
            cid: {
                "dense": dense_ranks.get(cid),
                "sparse": sparse_ranks.get(cid),
            }
            for cid in all_ids
        }

        return RetrievalDebug(
            dense=dense_candidates[:k],
            sparse=sparse_candidates[:k],
            fused=fused,
            contributions=contributions,
        )
