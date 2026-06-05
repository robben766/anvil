"""Unit tests for anvil-kb CLI — all real I/O mocked, no DB/embedding/LLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from anvil_kb.cli import _build_parser, _run_eval_command

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scored_chunk(content: str) -> object:
    """Build a minimal ScoredChunk-like object for injection."""
    chunk = MagicMock()
    chunk.content = content
    scored = MagicMock()
    scored.chunk = chunk
    scored.score = 0.9
    return scored


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParser:
    def test_eval_default_k(self):
        parser = _build_parser()
        args = parser.parse_args(["eval", "--dataset", "foo.jsonl", "--corpus", "corpus/"])
        assert args.k == 5

    def test_eval_custom_k(self):
        parser = _build_parser()
        args = parser.parse_args(["eval", "--dataset", "d.jsonl", "--corpus", "c/", "--k", "3"])
        assert args.k == 3

    def test_eval_k_is_int(self):
        parser = _build_parser()
        args = parser.parse_args(["eval", "--dataset", "d.jsonl", "--corpus", "c/"])
        assert isinstance(args.k, int)

    def test_eval_default_recall_threshold(self):
        parser = _build_parser()
        args = parser.parse_args(["eval", "--dataset", "d.jsonl", "--corpus", "c/"])
        assert args.recall_threshold == pytest.approx(0.8)

    def test_eval_custom_recall_threshold(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["eval", "--dataset", "d.jsonl", "--corpus", "c/", "--recall-threshold", "0.6"]
        )
        assert args.recall_threshold == pytest.approx(0.6)

    def test_eval_recall_threshold_is_float(self):
        parser = _build_parser()
        args = parser.parse_args(["eval", "--dataset", "d.jsonl", "--corpus", "c/"])
        assert isinstance(args.recall_threshold, float)

    def test_query_default_k(self):
        parser = _build_parser()
        args = parser.parse_args(["query", "some question"])
        assert args.k == 5

    def test_query_custom_k(self):
        parser = _build_parser()
        args = parser.parse_args(["query", "some question", "--k", "10"])
        assert args.k == 10

    def test_ingest_parses_files(self):
        parser = _build_parser()
        args = parser.parse_args(["ingest", "a.md", "b.md"])
        assert args.files == ["a.md", "b.md"]


# ---------------------------------------------------------------------------
# _run_eval_command: threshold & exit code
# ---------------------------------------------------------------------------


class TestRunEvalCommand:
    """Test the eval logic in isolation — mock ingest and retriever."""

    def _make_case(
        self,
        *,
        id: str,
        question: str,
        evidences: list[str],
        answerable: bool = True,
    ):
        from anvil_eval.dataset import GoldenCase

        return GoldenCase(
            id=id,
            question=question,
            reference="ref",
            evidences=evidences,
            answerable=answerable,
        )

    @pytest.mark.asyncio
    async def test_exit_0_when_mean_recall_gte_threshold(self, tmp_path, capsys):
        """Two answerable cases with recall 1.0 each → mean 1.0 ≥ 0.8 → exit 0."""
        case1 = self._make_case(
            id="t1", question="q1", evidences=["evidence one"]
        )
        case2 = self._make_case(
            id="t2", question="q2", evidences=["evidence two"]
        )

        # Retriever returns chunks whose content contains the evidence
        sc1 = _make_scored_chunk("evidence one found here")
        sc2 = _make_scored_chunk("evidence two found here")

        fake_retriever = AsyncMock()
        fake_retriever.retrieve = AsyncMock(side_effect=[[sc1], [sc2]])

        with pytest.raises(SystemExit) as exc:
            await _run_eval_command(
                cases=[case1, case2],
                retriever=fake_retriever,
                k=5,
                recall_threshold=0.8,
            )
        assert exc.value.code == 0

    @pytest.mark.asyncio
    async def test_exit_1_when_mean_recall_lt_threshold(self, tmp_path, capsys):
        """One recall=1.0 + one recall=0.0 → mean 0.5 < 0.8 → exit 1."""
        case1 = self._make_case(
            id="t1", question="q1", evidences=["evidence one"]
        )
        case2 = self._make_case(
            id="t2", question="q2", evidences=["evidence two"]
        )

        sc_hit = _make_scored_chunk("evidence one found here")
        sc_miss = _make_scored_chunk("completely unrelated content")

        fake_retriever = AsyncMock()
        fake_retriever.retrieve = AsyncMock(side_effect=[[sc_hit], [sc_miss]])

        with pytest.raises(SystemExit) as exc:
            await _run_eval_command(
                cases=[case1, case2],
                retriever=fake_retriever,
                k=5,
                recall_threshold=0.8,
            )
        assert exc.value.code == 1

    @pytest.mark.asyncio
    async def test_unanswerable_cases_skipped(self, tmp_path, capsys):
        """answerable=False cases are skipped and do NOT contribute to the mean."""
        answerable_case = self._make_case(
            id="t1", question="q1", evidences=["evidence one"]
        )
        skip_case = self._make_case(
            id="t2", question="q2", evidences=[], answerable=False
        )

        sc_hit = _make_scored_chunk("evidence one found here")

        fake_retriever = AsyncMock()
        # retrieve should only be called for the answerable case
        fake_retriever.retrieve = AsyncMock(return_value=[sc_hit])

        with pytest.raises(SystemExit) as exc:
            await _run_eval_command(
                cases=[answerable_case, skip_case],
                retriever=fake_retriever,
                k=5,
                recall_threshold=0.8,
            )
        # Only 1 answerable case with recall=1.0 → mean 1.0 ≥ 0.8 → exit 0
        assert exc.value.code == 0
        # retrieve called only once (for the answerable case)
        assert fake_retriever.retrieve.call_count == 1

    @pytest.mark.asyncio
    async def test_unanswerable_output_contains_skip_message(self, capsys):
        """answerable=False cases should print a skip notice."""
        skip_case = self._make_case(
            id="skip1", question="q?", evidences=[], answerable=False
        )
        answerable_case = self._make_case(
            id="ok1", question="q1", evidences=["some evidence"]
        )
        sc = _make_scored_chunk("some evidence here")

        fake_retriever = AsyncMock()
        fake_retriever.retrieve = AsyncMock(return_value=[sc])

        with pytest.raises(SystemExit):
            await _run_eval_command(
                cases=[skip_case, answerable_case],
                retriever=fake_retriever,
                k=5,
                recall_threshold=0.8,
            )
        captured = capsys.readouterr()
        assert "skip" in captured.out.lower() or "跳过" in captured.out

    @pytest.mark.asyncio
    async def test_exit_0_when_mean_recall_equals_threshold(self, capsys):
        """mean recall == threshold (0.5) → exit 0 (≥ semantics).

        Two answerable cases: recall 1.0 (hit) + recall 0.0 (miss) → mean 0.5.
        With --recall-threshold 0.5 the run is exactly on the boundary and
        must still pass (exit 0).
        """
        case1 = self._make_case(
            id="t1", question="q1", evidences=["evidence one"]
        )
        case2 = self._make_case(
            id="t2", question="q2", evidences=["evidence two"]
        )

        sc_hit = _make_scored_chunk("evidence one found here")
        sc_miss = _make_scored_chunk("completely unrelated content")

        fake_retriever = AsyncMock()
        fake_retriever.retrieve = AsyncMock(side_effect=[[sc_hit], [sc_miss]])

        with pytest.raises(SystemExit) as exc:
            await _run_eval_command(
                cases=[case1, case2],
                retriever=fake_retriever,
                k=5,
                recall_threshold=0.5,
            )
        # mean recall 0.5 >= threshold 0.5 → exit 0
        assert exc.value.code == 0

    @pytest.mark.asyncio
    async def test_all_unanswerable_dataset_exits_1(self, capsys):
        """All cases answerable=False → mean recall 0.0 → exit 1.

        retrieve must never be called because no answerable case exists.
        """
        case1 = self._make_case(
            id="u1", question="q1", evidences=[], answerable=False
        )
        case2 = self._make_case(
            id="u2", question="q2", evidences=[], answerable=False
        )

        fake_retriever = AsyncMock()
        fake_retriever.retrieve = AsyncMock()

        with pytest.raises(SystemExit) as exc:
            await _run_eval_command(
                cases=[case1, case2],
                retriever=fake_retriever,
                k=5,
                recall_threshold=0.8,
            )
        # mean recall 0.0 < threshold 0.8 → exit 1
        assert exc.value.code == 1
        # retrieve must not have been called — no answerable case to process
        fake_retriever.retrieve.assert_not_called()
