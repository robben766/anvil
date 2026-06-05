"""Unit tests for anvil-kb CLI — all real I/O mocked, no DB/embedding/LLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anvil_kb.cli import _build_parser, _run_eval_command, _run_ingest_command, _run_query_command

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

    @pytest.mark.asyncio
    async def test_mrr_column_in_output(self, capsys):
        """eval 输出应含 MRR 列头,每行含 mrr 分,均值行含 mean MRR。"""
        case1 = self._make_case(id="t1", question="q1", evidences=["evidence one"])
        sc1 = _make_scored_chunk("evidence one found here")

        fake_retriever = AsyncMock()
        fake_retriever.retrieve = AsyncMock(return_value=[sc1])

        with pytest.raises(SystemExit):
            await _run_eval_command(
                cases=[case1],
                retriever=fake_retriever,
                k=5,
                recall_threshold=0.8,
            )
        captured = capsys.readouterr()
        assert "mrr" in captured.out.lower()

    @pytest.mark.asyncio
    async def test_mrr_value_for_rank1_hit(self, capsys):
        """rank-1 命中时 MRR = 1.000,输出中可见。"""
        case1 = self._make_case(id="t1", question="q1", evidences=["evidence one"])
        sc1 = _make_scored_chunk("evidence one found here")

        fake_retriever = AsyncMock()
        fake_retriever.retrieve = AsyncMock(return_value=[sc1])

        with pytest.raises(SystemExit):
            await _run_eval_command(
                cases=[case1],
                retriever=fake_retriever,
                k=5,
                recall_threshold=0.8,
            )
        captured = capsys.readouterr()
        # MRR = 1.0 for rank-1 hit; output should contain "1.000"
        assert "1.000" in captured.out


# ---------------------------------------------------------------------------
# Parser --mode tests
# ---------------------------------------------------------------------------


class TestParserMode:
    def test_eval_default_mode_is_hybrid(self):
        parser = _build_parser()
        args = parser.parse_args(["eval", "--dataset", "d.jsonl", "--corpus", "c/"])
        assert args.mode == "hybrid"

    def test_eval_mode_dense(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["eval", "--dataset", "d.jsonl", "--corpus", "c/", "--mode", "dense"]
        )
        assert args.mode == "dense"

    def test_eval_mode_sparse(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["eval", "--dataset", "d.jsonl", "--corpus", "c/", "--mode", "sparse"]
        )
        assert args.mode == "sparse"

    def test_eval_mode_hybrid_explicit(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["eval", "--dataset", "d.jsonl", "--corpus", "c/", "--mode", "hybrid"]
        )
        assert args.mode == "hybrid"


# ---------------------------------------------------------------------------
# _run_query_command: hybrid wiring tests
# ---------------------------------------------------------------------------


class TestRunQueryCommand:
    """Smoke-test _run_query_command: verify hybrid Retriever is wired correctly.

    cli.py uses local imports inside the async functions, so we patch at the
    source module level (e.g. 'anvil_kb.store.pg.PgVectorStore') rather than
    at 'anvil_kb.cli.PgVectorStore' which is never bound as a module attribute.
    """

    @pytest.mark.asyncio
    async def test_query_command_uses_hybrid_retriever(self, capsys):
        """_run_query_command constructs Retriever with sparse_index + mode='hybrid'."""
        fake_embedder = MagicMock()
        fake_session_factory = MagicMock()

        fake_store = MagicMock()
        fake_bm25 = MagicMock()

        scored_chunk = _make_scored_chunk("some relevant content")
        fake_retriever = AsyncMock()
        fake_retriever.retrieve = AsyncMock(return_value=[scored_chunk])

        fake_answer = MagicMock()
        fake_answer.text = "42 天"
        fake_answer.citations = []

        with (
            patch("anvil_kb.store.pg.PgVectorStore", return_value=fake_store) as mock_vs,
            patch("anvil_kb.store.bm25.PgBM25Index", return_value=fake_bm25) as mock_bm25,
            patch(
                "anvil_kb.retrieve.retriever.Retriever", return_value=fake_retriever
            ) as mock_retriever_cls,
            patch("anvil_kb.generate.answer", new=AsyncMock(return_value=fake_answer)),
        ):
            await _run_query_command("等待期是多少天?", 3, fake_embedder, fake_session_factory)

        # PgVectorStore and PgBM25Index each constructed with session_factory
        mock_vs.assert_called_once_with(fake_session_factory)
        mock_bm25.assert_called_once_with(fake_session_factory)

        # Retriever must be constructed with sparse_index=bm25 and mode="hybrid"
        # reranker=None when --rerank not passed (default)
        mock_retriever_cls.assert_called_once_with(
            fake_embedder, fake_store, sparse_index=fake_bm25, mode="hybrid", reranker=None
        )

        # retrieve was called with the question and k
        fake_retriever.retrieve.assert_called_once_with("等待期是多少天?", k=3)

        # Answer printed to stdout
        captured = capsys.readouterr()
        assert "42 天" in captured.out

    @pytest.mark.asyncio
    async def test_query_command_does_not_raise_without_sparse_index_arg(self):
        """_run_query_command does NOT raise ValueError (hybrid wiring is present).

        Previously Retriever(embedder, store) with default mode='hybrid' would
        raise ValueError because sparse_index was None.  After the fix, the CLI
        constructs PgBM25Index and passes it explicitly.
        """
        fake_embedder = MagicMock()
        fake_session_factory = MagicMock()

        fake_store = MagicMock()
        fake_bm25 = MagicMock()

        scored_chunk = _make_scored_chunk("some content")
        fake_retriever = AsyncMock()
        fake_retriever.retrieve = AsyncMock(return_value=[scored_chunk])

        fake_answer = MagicMock()
        fake_answer.text = "ok"
        fake_answer.citations = []

        with (
            patch("anvil_kb.store.pg.PgVectorStore", return_value=fake_store),
            patch("anvil_kb.store.bm25.PgBM25Index", return_value=fake_bm25),
            patch("anvil_kb.retrieve.retriever.Retriever", return_value=fake_retriever),
            patch("anvil_kb.generate.answer", new=AsyncMock(return_value=fake_answer)),
        ):
            # Must not raise — previously raised ValueError("mode='hybrid' requires sparse_index")
            await _run_query_command("test question", 5, fake_embedder, fake_session_factory)


# ---------------------------------------------------------------------------
# _run_ingest_command: hybrid dual-write wiring tests
# ---------------------------------------------------------------------------


class TestRunIngestCommand:
    """Smoke-test _run_ingest_command: verify sparse_index is passed to ingest_markdown."""

    @pytest.mark.asyncio
    async def test_ingest_command_passes_sparse_index(self, tmp_path):
        """_run_ingest_command calls ingest_markdown with sparse_index=bm25."""
        md_file = tmp_path / "test_doc.md"
        md_file.write_text("# Test\n\nSome content here.", encoding="utf-8")

        fake_embedder = MagicMock()
        fake_session_factory = MagicMock()
        fake_store = MagicMock()
        fake_bm25 = MagicMock()

        mock_ingest = AsyncMock(return_value=(MagicMock(), 2))

        with (
            patch("anvil_kb.store.pg.PgVectorStore", return_value=fake_store),
            patch("anvil_kb.store.bm25.PgBM25Index", return_value=fake_bm25) as mock_bm25_cls,
            patch("anvil_kb.ingest.pipeline.ingest_markdown", mock_ingest),
        ):
            await _run_ingest_command(
                [str(md_file)], fake_embedder, fake_session_factory
            )

        # PgBM25Index constructed with session_factory
        mock_bm25_cls.assert_called_once_with(fake_session_factory)

        # ingest_markdown must have been called with sparse_index=bm25
        assert mock_ingest.call_count == 1
        _, kwargs = mock_ingest.call_args
        assert kwargs.get("sparse_index") is fake_bm25, (
            "ingest_markdown must be called with sparse_index=bm25 "
            f"(got sparse_index={kwargs.get('sparse_index')!r})"
        )

    @pytest.mark.asyncio
    async def test_ingest_command_multiple_files_all_get_sparse_index(self, tmp_path):
        """Each file in a multi-file ingest is dual-written with the same bm25 instance."""
        files = []
        for i in range(3):
            f = tmp_path / f"doc{i}.md"
            f.write_text(f"# Doc {i}\n\nContent {i}.", encoding="utf-8")
            files.append(str(f))

        fake_embedder = MagicMock()
        fake_session_factory = MagicMock()
        fake_store = MagicMock()
        fake_bm25 = MagicMock()

        mock_ingest = AsyncMock(return_value=(MagicMock(), 1))

        with (
            patch("anvil_kb.store.pg.PgVectorStore", return_value=fake_store),
            patch("anvil_kb.store.bm25.PgBM25Index", return_value=fake_bm25),
            patch("anvil_kb.ingest.pipeline.ingest_markdown", mock_ingest),
        ):
            await _run_ingest_command(files, fake_embedder, fake_session_factory)

        # ingest_markdown called once per file, each time with sparse_index=bm25
        assert mock_ingest.call_count == 3
        for i, individual_call in enumerate(mock_ingest.call_args_list):
            _, kwargs = individual_call
            assert kwargs.get("sparse_index") is fake_bm25, (
                f"call {i}: ingest_markdown must receive sparse_index=bm25"
            )


# ---------------------------------------------------------------------------
# --rerank flag tests  (R2)
# ---------------------------------------------------------------------------


class TestRerankFlag:
    """--rerank flag wires FastEmbedReranker into Retriever; default → no reranker."""

    def test_parser_rerank_flag_default_false(self):
        """--rerank not passed → args.rerank is False (store_true default)."""
        parser = _build_parser()
        args = parser.parse_args(["eval", "--dataset", "d.jsonl", "--corpus", "c/"])
        assert args.rerank is False

    def test_parser_rerank_flag_set_true(self):
        """--rerank passed → args.rerank is True."""
        parser = _build_parser()
        args = parser.parse_args(["eval", "--dataset", "d.jsonl", "--corpus", "c/", "--rerank"])
        assert args.rerank is True

    def test_parser_query_rerank_flag_default_false(self):
        parser = _build_parser()
        args = parser.parse_args(["query", "some question"])
        assert args.rerank is False

    def test_parser_query_rerank_flag_set_true(self):
        parser = _build_parser()
        args = parser.parse_args(["query", "some question", "--rerank"])
        assert args.rerank is True

    @pytest.mark.asyncio
    async def test_eval_with_rerank_passes_reranker_to_retriever(self):
        """--rerank=True → Retriever is constructed with reranker=FastEmbedReranker instance."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from anvil_kb.cli import _run_eval_with_ingest

        fake_embedder = MagicMock()
        fake_session_factory = MagicMock()
        fake_store = MagicMock()
        fake_bm25 = MagicMock()

        fake_retriever = AsyncMock()
        fake_case = MagicMock()
        fake_case.answerable = True
        fake_case.id = "c1"
        fake_case.question = "q?"
        fake_case.evidences = ["ev"]
        sc = _make_scored_chunk("ev content here")
        fake_retriever.retrieve = AsyncMock(return_value=[sc])

        fake_reranker_instance = MagicMock()

        with (
            patch("anvil_kb.store.pg.PgVectorStore", return_value=fake_store),
            patch("anvil_kb.store.bm25.PgBM25Index", return_value=fake_bm25),
            patch(
                "anvil_kb.retrieve.retriever.Retriever", return_value=fake_retriever
            ) as mock_retriever_cls,
            patch(
                "anvil_kb.retrieve.rerank.FastEmbedReranker",
                return_value=fake_reranker_instance,
            ) as mock_reranker_cls,
            patch("anvil_eval.dataset.load_dataset", return_value=[fake_case]),
            patch(
                "anvil_kb.ingest.pipeline.ingest_markdown",
                new=AsyncMock(return_value=(MagicMock(), 1)),
            ),
        ):
            with pytest.raises(SystemExit):
                await _run_eval_with_ingest(
                    dataset_path="d.jsonl",
                    corpus_dir=".",
                    k=5,
                    recall_threshold=0.8,
                    embedder=fake_embedder,
                    session_factory=fake_session_factory,
                    mode="dense",
                    rerank=True,
                )

        # FastEmbedReranker must have been constructed
        mock_reranker_cls.assert_called_once()
        # Retriever must have been constructed with reranker=fake_reranker_instance
        call_kwargs = mock_retriever_cls.call_args
        assert call_kwargs.kwargs.get("reranker") is fake_reranker_instance, (
            f"Retriever must receive reranker=FastEmbedReranker, "
            f"got reranker={call_kwargs.kwargs.get('reranker')!r}"
        )

    @pytest.mark.asyncio
    async def test_eval_without_rerank_passes_no_reranker(self):
        """--rerank not set (default) → Retriever constructed with reranker=None."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from anvil_kb.cli import _run_eval_with_ingest

        fake_embedder = MagicMock()
        fake_session_factory = MagicMock()
        fake_store = MagicMock()
        fake_bm25 = MagicMock()

        fake_retriever = AsyncMock()
        fake_retriever.retrieve = AsyncMock(return_value=[])

        with (
            patch("anvil_kb.store.pg.PgVectorStore", return_value=fake_store),
            patch("anvil_kb.store.bm25.PgBM25Index", return_value=fake_bm25),
            patch(
                "anvil_kb.retrieve.retriever.Retriever", return_value=fake_retriever
            ) as mock_retriever_cls,
            patch("anvil_kb.retrieve.rerank.FastEmbedReranker") as mock_reranker_cls,
            patch("anvil_eval.dataset.load_dataset", return_value=[]),
            patch(
                "anvil_kb.ingest.pipeline.ingest_markdown",
                new=AsyncMock(return_value=(MagicMock(), 1)),
            ),
        ):
            with pytest.raises(SystemExit):
                await _run_eval_with_ingest(
                    dataset_path="d.jsonl",
                    corpus_dir=".",
                    k=5,
                    recall_threshold=0.8,
                    embedder=fake_embedder,
                    session_factory=fake_session_factory,
                    mode="dense",
                    rerank=False,
                )

        # FastEmbedReranker must NOT have been constructed
        mock_reranker_cls.assert_not_called()
        # Retriever must have been constructed with reranker=None (or absent)
        call_kwargs = mock_retriever_cls.call_args
        assert call_kwargs.kwargs.get("reranker") is None


# ---------------------------------------------------------------------------
# eval retrieval latency stats  (R2)
# ---------------------------------------------------------------------------


class TestEvalRetrievalLatency:
    """eval subcommand prints 'mean retrieval latency: X.X ms' line after the table."""

    @pytest.mark.asyncio
    async def test_eval_prints_mean_retrieval_latency_line(self, capsys):
        """Output must contain a line matching 'mean retrieval latency: ... ms'."""
        from anvil_eval.dataset import GoldenCase

        from anvil_kb.cli import _run_eval_command

        case1 = GoldenCase(
            id="t1", question="q1", reference="r", evidences=["ev one"], answerable=True
        )
        sc = _make_scored_chunk("ev one here")
        fake_retriever = AsyncMock()
        fake_retriever.retrieve = AsyncMock(return_value=[sc])

        with pytest.raises(SystemExit):
            await _run_eval_command(
                cases=[case1],
                retriever=fake_retriever,
                k=5,
                recall_threshold=0.8,
            )

        captured = capsys.readouterr()
        assert "mean retrieval latency" in captured.out.lower(), (
            f"Expected 'mean retrieval latency' line in output, got:\n{captured.out}"
        )
        assert "ms" in captured.out, "Latency line must include 'ms' unit"

    @pytest.mark.asyncio
    async def test_eval_latency_not_printed_for_zero_answerable_cases(self, capsys):
        """All unanswerable → retrieve never called → latency line should still appear
        (shows 0.0 ms or similar) so callers can parse it consistently."""
        from anvil_eval.dataset import GoldenCase

        from anvil_kb.cli import _run_eval_command

        skip_case = GoldenCase(
            id="s1", question="q?", reference="r", evidences=[], answerable=False
        )
        fake_retriever = AsyncMock()
        fake_retriever.retrieve = AsyncMock()

        with pytest.raises(SystemExit):
            await _run_eval_command(
                cases=[skip_case],
                retriever=fake_retriever,
                k=5,
                recall_threshold=0.8,
            )

        captured = capsys.readouterr()
        assert "mean retrieval latency" in captured.out.lower(), (
            "latency line must appear even when no answerable cases"
        )


# ---------------------------------------------------------------------------
# PDF ingestion via CLI  (D3)
# ---------------------------------------------------------------------------


class TestRunIngestCommandPdf:
    """CLI ingest must route .pdf through parse_pdf → ingest_markdown.

    .md / .txt paths must NOT call parse_pdf (zero-change guarantee).
    """

    @pytest.mark.asyncio
    async def test_pdf_file_calls_parse_pdf_then_ingest(self, tmp_path):
        """.pdf → parse_pdf called with raw bytes; ingest_markdown called with result."""
        pdf_file = tmp_path / "doc.pdf"
        fake_bytes = b"%PDF-1.4 fake"
        pdf_file.write_bytes(fake_bytes)

        fake_embedder = MagicMock()
        fake_session_factory = MagicMock()
        fake_store = MagicMock()
        fake_bm25 = MagicMock()

        fake_parsed_text = "## 解析后\n\n内容。\n"
        mock_parse_pdf = MagicMock(return_value=fake_parsed_text)
        mock_ingest = AsyncMock(return_value=(MagicMock(), 3))

        with (
            patch("anvil_kb.store.pg.PgVectorStore", return_value=fake_store),
            patch("anvil_kb.store.bm25.PgBM25Index", return_value=fake_bm25),
            patch("anvil_kb.ingest.pipeline.ingest_markdown", mock_ingest),
            patch("anvil_kb.ingest.pdf.parse_pdf", mock_parse_pdf),
        ):
            await _run_ingest_command(
                [str(pdf_file)], fake_embedder, fake_session_factory
            )

        # parse_pdf must have been called with the raw bytes
        mock_parse_pdf.assert_called_once_with(fake_bytes)

        # ingest_markdown must have been called with the parsed text
        assert mock_ingest.call_count == 1
        _, kwargs = mock_ingest.call_args
        assert kwargs.get("text") == fake_parsed_text, (
            f"ingest_markdown must receive text=parsed_markdown, got {kwargs.get('text')!r}"
        )
        assert kwargs.get("title") == "doc"
        # source_name is the path string (relative if under cwd, else absolute)
        source_name = kwargs.get("source_name", "")
        assert "doc.pdf" in source_name

    @pytest.mark.asyncio
    async def test_md_file_does_not_call_parse_pdf(self, tmp_path):
        """.md → parse_pdf must NOT be called; UTF-8 read path unchanged."""
        md_file = tmp_path / "notes.md"
        md_file.write_text("# Notes\n\nContent.", encoding="utf-8")

        fake_embedder = MagicMock()
        fake_session_factory = MagicMock()
        fake_store = MagicMock()
        fake_bm25 = MagicMock()

        mock_parse_pdf = MagicMock()
        mock_ingest = AsyncMock(return_value=(MagicMock(), 1))

        with (
            patch("anvil_kb.store.pg.PgVectorStore", return_value=fake_store),
            patch("anvil_kb.store.bm25.PgBM25Index", return_value=fake_bm25),
            patch("anvil_kb.ingest.pipeline.ingest_markdown", mock_ingest),
            patch("anvil_kb.ingest.pdf.parse_pdf", mock_parse_pdf),
        ):
            await _run_ingest_command(
                [str(md_file)], fake_embedder, fake_session_factory
            )

        # parse_pdf must NOT be called for .md files
        mock_parse_pdf.assert_not_called()
        # ingest_markdown must still be called
        assert mock_ingest.call_count == 1


# ---------------------------------------------------------------------------
# PDF corpus support in eval (D4)
# ---------------------------------------------------------------------------


class TestEvalWithPdfCorpus:
    """eval --corpus now accepts .pdf files alongside .md files.

    _run_eval_with_ingest must route .pdf corpus files through parse_pdf
    and .md corpus files through the plain UTF-8 read path.
    """

    @pytest.mark.asyncio
    async def test_eval_corpus_dir_with_pdf_calls_parse_pdf(self, tmp_path):
        """.pdf in corpus dir → parse_pdf called; ingest_markdown receives parsed text."""
        from anvil_kb.cli import _run_eval_with_ingest

        pdf_file = tmp_path / "doc.pdf"
        fake_bytes = b"%PDF-1.4 fake"
        pdf_file.write_bytes(fake_bytes)

        fake_embedder = MagicMock()
        fake_session_factory = MagicMock()
        fake_store = MagicMock()
        fake_bm25 = MagicMock()

        parsed_text = "## 解析后\n\n正文。\n"
        mock_parse_pdf = MagicMock(return_value=parsed_text)
        mock_ingest = AsyncMock(return_value=(MagicMock(), 2))

        fake_retriever = AsyncMock()
        fake_retriever.retrieve = AsyncMock(return_value=[])

        with (
            patch("anvil_kb.store.pg.PgVectorStore", return_value=fake_store),
            patch("anvil_kb.store.bm25.PgBM25Index", return_value=fake_bm25),
            patch("anvil_kb.ingest.pipeline.ingest_markdown", mock_ingest),
            patch("anvil_kb.ingest.pdf.parse_pdf", mock_parse_pdf),
            patch("anvil_kb.retrieve.retriever.Retriever", return_value=fake_retriever),
            patch("anvil_eval.dataset.load_dataset", return_value=[]),
        ):
            with pytest.raises(SystemExit):
                await _run_eval_with_ingest(
                    dataset_path="d.jsonl",
                    corpus_dir=str(tmp_path),
                    k=5,
                    recall_threshold=0.8,
                    embedder=fake_embedder,
                    session_factory=fake_session_factory,
                    mode="dense",
                    rerank=False,
                )

        # parse_pdf must have been called with the raw bytes
        mock_parse_pdf.assert_called_once_with(fake_bytes)
        # ingest_markdown must have received the parsed text
        assert mock_ingest.call_count == 1
        _, kwargs = mock_ingest.call_args
        assert kwargs.get("text") == parsed_text

    @pytest.mark.asyncio
    async def test_eval_corpus_dir_with_md_does_not_call_parse_pdf(self, tmp_path):
        """.md in corpus dir → parse_pdf must NOT be called; plain read path used."""
        from anvil_kb.cli import _run_eval_with_ingest

        md_file = tmp_path / "notes.md"
        md_file.write_text("# Notes\n\nContent.", encoding="utf-8")

        fake_embedder = MagicMock()
        fake_session_factory = MagicMock()
        fake_store = MagicMock()
        fake_bm25 = MagicMock()

        mock_parse_pdf = MagicMock()
        mock_ingest = AsyncMock(return_value=(MagicMock(), 1))

        fake_retriever = AsyncMock()
        fake_retriever.retrieve = AsyncMock(return_value=[])

        with (
            patch("anvil_kb.store.pg.PgVectorStore", return_value=fake_store),
            patch("anvil_kb.store.bm25.PgBM25Index", return_value=fake_bm25),
            patch("anvil_kb.ingest.pipeline.ingest_markdown", mock_ingest),
            patch("anvil_kb.ingest.pdf.parse_pdf", mock_parse_pdf),
            patch("anvil_kb.retrieve.retriever.Retriever", return_value=fake_retriever),
            patch("anvil_eval.dataset.load_dataset", return_value=[]),
        ):
            with pytest.raises(SystemExit):
                await _run_eval_with_ingest(
                    dataset_path="d.jsonl",
                    corpus_dir=str(tmp_path),
                    k=5,
                    recall_threshold=0.8,
                    embedder=fake_embedder,
                    session_factory=fake_session_factory,
                    mode="dense",
                    rerank=False,
                )

        mock_parse_pdf.assert_not_called()
        assert mock_ingest.call_count == 1
