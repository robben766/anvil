"""CLI: anvil-kb ingest / query / eval

anvil-kb ingest <file.md ...> [--enrich]
    title=stem; source_name=relative path from cwd.
    --enrich: enable Contextual Retrieval enrichment via anvil_gateway.chat.

anvil-kb query "<question>" [--k 5]
    Retrieve + generate; prints answer and citations.
    Requires DEEPSEEK_API_KEY (loaded from .env automatically).

anvil-kb eval --dataset <kb.jsonl> --corpus <dir> [--k 5] [--recall-threshold 0.8] [--enrich]
    Ingest all .md in corpus (idempotent), then evaluate retrieval only (no LLM).
    Exit 0 if mean recall@k >= threshold, else exit 1.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

# ---------------------------------------------------------------------------
# Enrichment cost accounting (public surface for tests)
# ---------------------------------------------------------------------------

_ENRICH_SESSION_ID = "anvil-kb-enrich"


def summarize_enrich_usage(rows) -> dict:
    """Aggregate UsageRecordRow sequence into enrichment cost summary.

    Accepts either ORM row objects (attribute access) or plain dicts.
    Fields read: prompt_tokens, cached_tokens, cost_cny.

    Returns:
        dict with keys: calls, prompt_tokens, cached_tokens,
                        cache_hit_rate (float), cost (Decimal).
    """

    def _get(row, field):
        if isinstance(row, dict):
            return row[field]
        return getattr(row, field)

    calls = 0
    total_prompt = 0
    total_cached = 0
    total_cost = Decimal("0")

    for row in rows:
        calls += 1
        total_prompt += _get(row, "prompt_tokens")
        total_cached += _get(row, "cached_tokens")
        total_cost += Decimal(str(_get(row, "cost_cny"))) if not isinstance(
            _get(row, "cost_cny"), Decimal
        ) else _get(row, "cost_cny")

    cache_hit_rate = (total_cached / total_prompt) if total_prompt > 0 else 0.0

    return {
        "calls": calls,
        "prompt_tokens": total_prompt,
        "cached_tokens": total_cached,
        "cache_hit_rate": cache_hit_rate,
        "cost": total_cost,
    }


def _query_enrich_usage(gateway_session_factory, since: datetime):
    """Query usage_records for anvil-kb-enrich session rows since *since*.

    Reads the gateway DB read-only; never modifies any gateway table.
    Returns a list of UsageRecordRow instances.
    """
    from anvil_gateway.db import UsageRecordRow  # noqa: PLC0415

    with gateway_session_factory() as session:
        rows = (
            session.query(UsageRecordRow)
            .filter(
                UsageRecordRow.session_id == _ENRICH_SESSION_ID,
                UsageRecordRow.created_at >= since,
            )
            .all()
        )
    return rows


def _make_gateway_session_factory():
    """Create a SQLAlchemy session factory pointing at the gateway DB.

    Reads ANVIL_DATABASE_URL from env (same DB as kb components).
    """
    import os  # noqa: PLC0415

    from sqlalchemy import create_engine  # noqa: PLC0415
    from sqlalchemy.orm import sessionmaker  # noqa: PLC0415

    url = os.environ.get("ANVIL_DATABASE_URL", "")
    engine = create_engine(url)
    return sessionmaker(bind=engine)


# ---------------------------------------------------------------------------
# Internal helpers (public surface for tests)
# ---------------------------------------------------------------------------


def _load_env() -> None:
    """Load .env from the anvil repo root (parent of packages/) if present."""
    try:
        from dotenv import load_dotenv  # type: ignore[import]
    except ImportError:
        return
    # Walk up from cwd looking for .env; stop at filesystem root.
    here = Path.cwd()
    for candidate in [here, *here.parents]:
        env_file = candidate / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            break


def _make_components():
    """Construct embedder + session_factory once (called at command entry)."""
    from anvil_kb.db import make_session_factory
    from anvil_kb.embed import FastEmbedEmbedder

    embedder = FastEmbedEmbedder()
    session_factory = make_session_factory()  # reads ANVIL_DATABASE_URL from env
    return embedder, session_factory


async def _run_ingest_command(
    files: list[str], embedder, session_factory, *, enrich: bool = False
) -> None:
    from anvil_kb.ingest.pdf import parse_pdf
    from anvil_kb.ingest.pipeline import ingest_markdown
    from anvil_kb.store.bm25 import PgBM25Index
    from anvil_kb.store.pg import PgVectorStore

    # Capture start time before ingestion begins (for usage ledger query)
    run_start = datetime.now(tz=UTC)

    enrich_chat = None
    if enrich:
        import anvil_gateway  # noqa: PLC0415
        enrich_chat = anvil_gateway.chat

    store = PgVectorStore(session_factory)
    bm25 = PgBM25Index(session_factory)
    cwd = Path.cwd()
    for file_str in files:
        path = Path(file_str)
        title = path.stem
        try:
            source_name = str(path.relative_to(cwd))
        except ValueError:
            source_name = str(path)
        if path.suffix.lower() == ".pdf":
            raw = path.read_bytes()
            text = parse_pdf(raw)
        else:
            text = path.read_text(encoding="utf-8")
        doc, n_chunks = await ingest_markdown(
            title=title,
            source_name=source_name,
            text=text,
            embedder=embedder,
            store=store,
            sparse_index=bm25,
            enrich_chat=enrich_chat,
        )
        print(f"ingested: {source_name!r}  title={title!r}  chunks={n_chunks}")

    if enrich:
        gw_session_factory = _make_gateway_session_factory()
        rows = _query_enrich_usage(gw_session_factory, since=run_start)
        summary = summarize_enrich_usage(rows)
        print(
            f"enrich: {summary['calls']} calls, "
            f"{summary['prompt_tokens']} prompt tokens, "
            f"cache hit {summary['cache_hit_rate']:.1%}, "
            f"cost ¥{summary['cost']}"
        )


async def _run_query_command(
    question: str, k: int, embedder, session_factory, *, rerank: bool = False
) -> None:
    from anvil_kb.generate import answer
    from anvil_kb.retrieve.retriever import Retriever
    from anvil_kb.store.bm25 import PgBM25Index
    from anvil_kb.store.pg import PgVectorStore

    store = PgVectorStore(session_factory)
    bm25 = PgBM25Index(session_factory)

    reranker = None
    if rerank:
        from anvil_kb.retrieve.rerank import FastEmbedReranker
        reranker = FastEmbedReranker()

    retriever = Retriever(embedder, store, sparse_index=bm25, mode="hybrid", reranker=reranker)
    retrieved = await retriever.retrieve(question, k=k)
    kb_answer = await answer(question, retrieved)
    print(kb_answer.text)
    if kb_answer.citations:
        print()
        print("--- 引用 ---")
        for c in kb_answer.citations:
            print(f"[{c.n}] {c.quote[:120]}")


async def _run_eval_command(
    cases,
    retriever,
    k: int,
    recall_threshold: float,
) -> None:
    """Core eval loop — separated for testability (retriever injected).

    Exits the process with code 0 (pass) or 1 (fail).
    """
    import time

    from anvil_eval.metrics.retrieval import mrr, precision_at_k, recall_at_k

    recall_scores: list[float] = []
    precision_scores: list[float] = []
    mrr_scores: list[float] = []
    retrieval_latencies_ms: list[float] = []

    print(f"{'id':<12} {'recall@k':>10} {'precision@k':>12} {'mrr':>8}")
    print("-" * 48)

    for case in cases:
        if not case.answerable:
            print(f"{case.id:<12}  跳过(拒答用例)")
            continue

        t0 = time.perf_counter()
        retrieved = await retriever.retrieve(case.question, k=k)
        t1 = time.perf_counter()
        retrieval_latencies_ms.append((t1 - t0) * 1000)

        texts = [sc.chunk.content for sc in retrieved]

        r = recall_at_k(texts, case.evidences, k)
        p = precision_at_k(texts, case.evidences, k)
        m = mrr(texts, case.evidences)

        r_val = r if r is not None else 0.0
        p_val = p if p is not None else 0.0
        m_val = m if m is not None else 0.0
        recall_scores.append(r_val)
        precision_scores.append(p_val)
        mrr_scores.append(m_val)
        print(f"{case.id:<12} {r_val:>10.3f} {p_val:>12.3f} {m_val:>8.3f}")

    print("-" * 48)
    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    mean_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    mean_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    mean_latency_ms = (
        sum(retrieval_latencies_ms) / len(retrieval_latencies_ms)
        if retrieval_latencies_ms
        else 0.0
    )
    print(f"{'mean':<12} {mean_recall:>10.3f} {mean_precision:>12.3f} {mean_mrr:>8.3f}")
    print()
    print(
        f"mean recall@{k} = {mean_recall:.4f}"
        f"  threshold = {recall_threshold}"
        f"  mean MRR = {mean_mrr:.4f}"
    )
    print(f"mean retrieval latency: {mean_latency_ms:.1f} ms")

    if mean_recall >= recall_threshold:
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)


async def _run_eval_with_ingest(
    dataset_path: str,
    corpus_dir: str,
    k: int,
    recall_threshold: float,
    embedder,
    session_factory,
    mode: str = "hybrid",
    rerank: bool = False,
    enrich: bool = False,
) -> None:
    """Ingest corpus then run eval loop.

    Supports .md and .pdf files in *corpus_dir*.  PDF bytes are parsed via
    ``parse_pdf`` before being handed to ``ingest_markdown`` — the same path as
    the CLI ingest command.
    """
    from anvil_eval.dataset import load_dataset

    from anvil_kb.ingest.pdf import parse_pdf
    from anvil_kb.ingest.pipeline import ingest_markdown
    from anvil_kb.retrieve.retriever import Retriever
    from anvil_kb.store.bm25 import PgBM25Index
    from anvil_kb.store.pg import PgVectorStore

    enrich_chat = None
    run_start: datetime | None = None
    if enrich:
        run_start = datetime.now(tz=UTC)
        import anvil_gateway  # noqa: PLC0415
        enrich_chat = anvil_gateway.chat

    # Ingest corpus
    store = PgVectorStore(session_factory)
    sparse_index: PgBM25Index | None = None
    if mode in ("sparse", "hybrid"):
        sparse_index = PgBM25Index(session_factory)

    corpus = Path(corpus_dir)
    corpus_files = sorted(
        f for f in corpus.iterdir() if f.suffix.lower() in (".md", ".pdf")
    )
    if not corpus_files:
        print(f"WARNING: no .md/.pdf files found in {corpus_dir}", file=sys.stderr)

    print(f"Ingesting {len(corpus_files)} file(s) from {corpus_dir!r} (mode={mode!r}) ...")
    for file_path in corpus_files:
        source_name = file_path.name
        title = file_path.stem
        if file_path.suffix.lower() == ".pdf":
            text = parse_pdf(file_path.read_bytes())
        else:
            text = file_path.read_text(encoding="utf-8")
        _, n_chunks = await ingest_markdown(
            title=title,
            source_name=source_name,
            text=text,
            embedder=embedder,
            store=store,
            sparse_index=sparse_index,
            enrich_chat=enrich_chat,
        )
        print(f"  {source_name}  ({n_chunks} chunks)")

    # Print enrichment cost report after corpus ingest (before eval)
    if enrich and run_start is not None:
        gw_session_factory = _make_gateway_session_factory()
        rows = _query_enrich_usage(gw_session_factory, since=run_start)
        summary = summarize_enrich_usage(rows)
        print(
            f"enrich: {summary['calls']} calls, "
            f"{summary['prompt_tokens']} prompt tokens, "
            f"cache hit {summary['cache_hit_rate']:.1%}, "
            f"cost ¥{summary['cost']}"
        )

    # Build reranker if requested
    reranker = None
    if rerank:
        from anvil_kb.retrieve.rerank import FastEmbedReranker
        reranker = FastEmbedReranker()

    # Build retriever
    retriever = Retriever(embedder, store, sparse_index=sparse_index, mode=mode, reranker=reranker)

    # Load golden cases
    cases = load_dataset(dataset_path)
    print(f"\nEvaluating {len(cases)} case(s) (k={k}, mode={mode!r}) ...")
    print()

    await _run_eval_command(cases, retriever, k, recall_threshold)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anvil-kb",
        description="anvil knowledge base CLI: ingest / query / eval",
    )
    sub = parser.add_subparsers(dest="command")

    # --- ingest ---
    ingest_p = sub.add_parser("ingest", help="Ingest markdown file(s) into the knowledge base")
    ingest_p.add_argument("files", nargs="+", help="Markdown file paths to ingest")
    ingest_p.add_argument(
        "--enrich",
        action="store_true",
        default=False,
        help="Enable Contextual Retrieval enrichment via anvil_gateway.chat",
    )

    # --- query ---
    query_p = sub.add_parser("query", help="Retrieve + generate an answer")
    query_p.add_argument("question", help="Question to answer")
    query_p.add_argument(
        "--k", type=int, default=5, help="Number of chunks to retrieve (default: 5)"
    )
    query_p.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help="Enable cross-encoder reranking (FastEmbedReranker; loads ~1GB model on first use)",
    )

    # --- eval ---
    eval_p = sub.add_parser("eval", help="Evaluate retrieval recall/precision on a golden dataset")
    eval_p.add_argument("--dataset", required=True, help="Path to golden JSONL file")
    eval_p.add_argument("--corpus", required=True, help="Directory with .md corpus files to ingest")
    eval_p.add_argument("--k", type=int, default=5, help="Retrieval top-k (default: 5)")
    eval_p.add_argument(
        "--recall-threshold",
        type=float,
        default=0.8,
        dest="recall_threshold",
        help="Mean recall@k gate threshold (default: 0.8)",
    )
    eval_p.add_argument(
        "--mode",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Retrieval mode: dense | sparse | hybrid (default: hybrid)",
    )
    eval_p.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help="Enable cross-encoder reranking (FastEmbedReranker; loads ~1GB model on first use)",
    )
    eval_p.add_argument(
        "--enrich",
        action="store_true",
        default=False,
        help="Enable Contextual Retrieval enrichment via anvil_gateway.chat",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _load_env()

    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(2)

    if args.command == "ingest":
        embedder, session_factory = _make_components()
        asyncio.run(_run_ingest_command(args.files, embedder, session_factory, enrich=args.enrich))

    elif args.command == "query":
        embedder, session_factory = _make_components()
        asyncio.run(
            _run_query_command(
                args.question, args.k, embedder, session_factory, rerank=args.rerank
            )
        )

    elif args.command == "eval":
        embedder, session_factory = _make_components()
        asyncio.run(
            _run_eval_with_ingest(
                dataset_path=args.dataset,
                corpus_dir=args.corpus,
                k=args.k,
                recall_threshold=args.recall_threshold,
                embedder=embedder,
                session_factory=session_factory,
                mode=args.mode,
                rerank=args.rerank,
                enrich=args.enrich,
            )
        )

    else:
        parser.print_help()
        sys.exit(2)
