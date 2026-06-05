"""Unit tests for RRF fusion — hand-calculated, no real DB.

RRF formula: score(c) = Σ_lists 1/(k + rank_i(c)), rank 1-based, k=60 (original paper).
"""
from __future__ import annotations

import uuid

import pytest

from anvil_kb.store.base import (  # noqa: F401 – Chunk used in TestRrfFuseChunkIdentity
    Chunk,
    ScoredChunk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(content: str) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        seq=0,
        content=content,
        header_path="",
        start_offset=0,
        end_offset=len(content),
    )


def _scored(chunk: Chunk, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score)


# ---------------------------------------------------------------------------
# Main hand-calculated fusion test
# ---------------------------------------------------------------------------


class TestRrfFuseHandCalc:
    """Two lists of 3 items each, 1 intersection item — paper-verified math."""

    def setup_method(self):
        # Shared chunk for intersection
        self.shared = _chunk("intersection chunk")

        # List A: [a0, a1, shared] — shared is at rank 3 in list A
        self.a0 = _chunk("A0 only")
        self.a1 = _chunk("A1 only")

        # List B: [b0, shared, b1] — shared is at rank 2 in list B
        self.b0 = _chunk("B0 only")
        self.b1 = _chunk("B1 only")

        self.list_a = [
            _scored(self.a0, 0.9),
            _scored(self.a1, 0.8),
            _scored(self.shared, 0.7),
        ]
        self.list_b = [
            _scored(self.b0, 0.95),
            _scored(self.shared, 0.85),
            _scored(self.b1, 0.6),
        ]

    def test_intersection_score_hand_calc(self):
        """
        Hand calc for shared chunk:
          - rank in list_a = 3  →  1/(60+3) = 1/63  ≈ 0.01587302
          - rank in list_b = 2  →  1/(60+2) = 1/62  ≈ 0.01612903
          - sum                 = 1/63 + 1/62        ≈ 0.03200205

        Exact fractions: 1/63 = 0.015873015873..., 1/62 = 0.016129032258...
        Sum = 0.032002048131...
        """
        from anvil_kb.retrieve.fusion import rrf_fuse

        result = rrf_fuse([self.list_a, self.list_b], k=60, top=10)

        shared_result = next(r for r in result if r.chunk.id == self.shared.id)
        expected = 1.0 / 63.0 + 1.0 / 62.0  # ≈ 0.032002048131...
        assert shared_result.score == pytest.approx(expected, rel=1e-9)

    def test_single_list_item_score_hand_calc(self):
        """
        a0 only appears in list_a at rank 1:
          score = 1/(60+1) = 1/61 ≈ 0.016393443...

        b0 only appears in list_b at rank 1:
          score = 1/(60+1) = 1/61 ≈ 0.016393443...
        """
        from anvil_kb.retrieve.fusion import rrf_fuse

        result = rrf_fuse([self.list_a, self.list_b], k=60, top=10)

        a0_result = next(r for r in result if r.chunk.id == self.a0.id)
        b0_result = next(r for r in result if r.chunk.id == self.b0.id)

        expected_single = 1.0 / 61.0  # 1/(60+1)
        assert a0_result.score == pytest.approx(expected_single, rel=1e-9)
        assert b0_result.score == pytest.approx(expected_single, rel=1e-9)

    def test_ranking_order(self):
        """
        Shared should rank #1 (highest score), a0 and b0 tied at #2/#3.
        a1 and b1 should be ranked below (lower rank in their respective lists).
        """
        from anvil_kb.retrieve.fusion import rrf_fuse

        result = rrf_fuse([self.list_a, self.list_b], k=60, top=10)

        assert result[0].chunk.id == self.shared.id, "shared item (2 lists) must be rank 1"

        # a0 and b0 both at 1/(60+1): tied for rank 2/3
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True), "results must be descending by score"


# ---------------------------------------------------------------------------
# Degenerate cases
# ---------------------------------------------------------------------------


class TestRrfFuseDegenerate:
    def test_empty_lists_returns_empty(self):
        from anvil_kb.retrieve.fusion import rrf_fuse

        assert rrf_fuse([]) == []

    def test_list_of_empty_lists_returns_empty(self):
        from anvil_kb.retrieve.fusion import rrf_fuse

        assert rrf_fuse([[], []]) == []

    def test_single_list_preserves_order(self):
        """Single list: order must be identical to input ranking."""
        from anvil_kb.retrieve.fusion import rrf_fuse

        chunks = [_chunk(f"c{i}") for i in range(3)]
        ranked = [_scored(c, 1.0 - 0.1 * i) for i, c in enumerate(chunks)]

        result = rrf_fuse([ranked], k=60, top=10)

        # Scores must be decreasing, order must match input
        for i, r in enumerate(result):
            assert r.chunk.id == chunks[i].id, f"position {i} mismatch"

    def test_single_list_scores_1_over_k_plus_rank(self):
        """With a single list, score(rank=i) == 1/(60+i), 1-based."""
        from anvil_kb.retrieve.fusion import rrf_fuse

        chunks = [_chunk(f"c{i}") for i in range(3)]
        ranked = [_scored(c, 0.9) for c in chunks]  # original scores don't matter

        result = rrf_fuse([ranked], k=60, top=10)

        for i, r in enumerate(result):
            expected = 1.0 / (60 + i + 1)  # rank is 1-based
            assert r.score == pytest.approx(expected, rel=1e-9), f"rank {i+1} wrong"


# ---------------------------------------------------------------------------
# Top truncation
# ---------------------------------------------------------------------------


class TestRrfFuseTopTruncation:
    def test_top_truncates_output(self):
        from anvil_kb.retrieve.fusion import rrf_fuse

        chunks = [_chunk(f"c{i}") for i in range(5)]
        ranked = [_scored(c, 1.0 - 0.1 * i) for i, c in enumerate(chunks)]

        result = rrf_fuse([ranked], k=60, top=3)
        assert len(result) == 3

    def test_top_larger_than_pool_returns_all(self):
        from anvil_kb.retrieve.fusion import rrf_fuse

        chunks = [_chunk(f"c{i}") for i in range(3)]
        ranked = [_scored(c, 0.9) for c in chunks]

        result = rrf_fuse([ranked], k=60, top=100)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Tie-break stability
# ---------------------------------------------------------------------------


class TestRrfFuseTieBreakStability:
    """When two chunks have equal fused scores, the one appearing first
    in list-order (first list first, then rank within list) must come first."""

    def test_tie_break_by_first_occurrence_list_order(self):
        """
        Both c0 and c1 appear at rank 1 in different lists.
        c0 is in list_0 (index 0), c1 is in list_1 (index 1).
        → c0 must come before c1 (tie-break by first-occurrence list index).
        """
        from anvil_kb.retrieve.fusion import rrf_fuse

        c0 = _chunk("c0")
        c1 = _chunk("c1")

        list_0 = [_scored(c0, 0.9)]
        list_1 = [_scored(c1, 0.9)]

        result = rrf_fuse([list_0, list_1], k=60, top=10)

        assert result[0].chunk.id == c0.id, "c0 from first list must come first on tie"
        assert result[1].chunk.id == c1.id

    def test_tie_break_within_same_list_by_rank(self):
        """
        c0 at rank 2 in list_0 and c1 at rank 2 in list_1 → tied.
        c0 first occurrence is list index 0 → c0 before c1.
        """
        from anvil_kb.retrieve.fusion import rrf_fuse

        anchor = _chunk("anchor")  # rank 1 in both lists
        c0 = _chunk("c0")  # rank 2 in list_0
        c1 = _chunk("c1")  # rank 2 in list_1

        list_0 = [_scored(anchor, 0.95), _scored(c0, 0.8)]
        list_1 = [_scored(anchor, 0.9), _scored(c1, 0.75)]

        result = rrf_fuse([list_0, list_1], k=60, top=10)

        c0_pos = next(i for i, r in enumerate(result) if r.chunk.id == c0.id)
        c1_pos = next(i for i, r in enumerate(result) if r.chunk.id == c1.id)
        assert c0_pos < c1_pos, "c0 (first list, same rank) must come before c1"


# ---------------------------------------------------------------------------
# Chunk object identity — first occurrence wins
# ---------------------------------------------------------------------------


class TestRrfFuseChunkIdentity:
    def test_chunk_object_from_first_occurrence(self):
        """When a chunk appears in multiple lists, the returned object is from
        the first list (first occurrence by list index, then by rank)."""
        from anvil_kb.retrieve.fusion import rrf_fuse

        shared_id = uuid.uuid4()

        chunk_a = Chunk(
            id=shared_id,
            document_id=uuid.uuid4(),
            seq=0,
            content="from list A",
            header_path="",
            start_offset=0,
            end_offset=10,
        )
        chunk_b = Chunk(
            id=shared_id,
            document_id=uuid.uuid4(),
            seq=1,
            content="from list B",
            header_path="",
            start_offset=0,
            end_offset=10,
        )

        list_a = [_scored(chunk_a, 0.9)]
        list_b = [_scored(chunk_b, 0.85)]

        result = rrf_fuse([list_a, list_b], k=60, top=10)

        matched = next(r for r in result if r.chunk.id == shared_id)
        assert matched.chunk.content == "from list A", "must use chunk from first list"
