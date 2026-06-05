"""CLI: anvil-kb ingest / query / eval

anvil-kb ingest <file.md ...>
    title=stem; source_name=relative path from cwd.

anvil-kb query "<question>" [--k 5]
    Retrieve + generate; prints answer and citations.
    Requires DEEPSEEK_API_KEY (loaded from .env automatically).

anvil-kb eval --dataset <kb.jsonl> --corpus <dir> [--k 5] [--recall-threshold 0.8]
    Ingest all .md in corpus (idempotent), then evaluate retrieval only (no LLM).
    Exit 0 if mean recall@k >= threshold, else exit 1.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

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


async def _run_ingest_command(files: list[str], embedder, session_factory) -> None:
    from anvil_kb.ingest.pipeline import ingest_markdown
    from anvil_kb.store.pg import PgVectorStore

    store = PgVectorStore(session_factory)
    cwd = Path.cwd()
    for file_str in files:
        path = Path(file_str)
        title = path.stem
        try:
            source_name = str(path.relative_to(cwd))
        except ValueError:
            source_name = str(path)
        text = path.read_text(encoding="utf-8")
        doc, n_chunks = await ingest_markdown(
            title=title,
            source_name=source_name,
            text=text,
            embedder=embedder,
            store=store,
        )
        print(f"ingested: {source_name!r}  title={title!r}  chunks={n_chunks}")


async def _run_query_command(question: str, k: int, embedder, session_factory) -> None:
    from anvil_kb.generate import answer
    from anvil_kb.retrieve.retriever import Retriever
    from anvil_kb.store.pg import PgVectorStore

    store = PgVectorStore(session_factory)
    retriever = Retriever(embedder, store)
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
    from anvil_eval.metrics.retrieval import precision_at_k, recall_at_k

    recall_scores: list[float] = []
    precision_scores: list[float] = []

    print(f"{'id':<12} {'recall@k':>10} {'precision@k':>12}")
    print("-" * 38)

    for case in cases:
        if not case.answerable:
            print(f"{case.id:<12}  跳过(拒答用例)")
            continue

        retrieved = await retriever.retrieve(case.question, k=k)
        texts = [sc.chunk.content for sc in retrieved]

        r = recall_at_k(texts, case.evidences, k)
        p = precision_at_k(texts, case.evidences, k)

        r_val = r if r is not None else 0.0
        p_val = p if p is not None else 0.0
        recall_scores.append(r_val)
        precision_scores.append(p_val)
        print(f"{case.id:<12} {r_val:>10.3f} {p_val:>12.3f}")

    print("-" * 38)
    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    mean_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    print(f"{'mean':<12} {mean_recall:>10.3f} {mean_precision:>12.3f}")
    print()
    print(f"mean recall@{k} = {mean_recall:.4f}  threshold = {recall_threshold}")

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
) -> None:
    """Ingest corpus then run eval loop."""
    from anvil_eval.dataset import load_dataset

    from anvil_kb.ingest.pipeline import ingest_markdown
    from anvil_kb.retrieve.retriever import Retriever
    from anvil_kb.store.pg import PgVectorStore

    # Ingest corpus
    store = PgVectorStore(session_factory)
    corpus = Path(corpus_dir)
    md_files = sorted(corpus.glob("*.md"))
    if not md_files:
        print(f"WARNING: no .md files found in {corpus_dir}", file=sys.stderr)

    print(f"Ingesting {len(md_files)} file(s) from {corpus_dir!r} ...")
    for md_path in md_files:
        source_name = md_path.name
        title = md_path.stem
        text = md_path.read_text(encoding="utf-8")
        _, n_chunks = await ingest_markdown(
            title=title,
            source_name=source_name,
            text=text,
            embedder=embedder,
            store=store,
        )
        print(f"  {source_name}  ({n_chunks} chunks)")

    # Build retriever
    retriever = Retriever(embedder, store)

    # Load golden cases
    cases = load_dataset(dataset_path)
    print(f"\nEvaluating {len(cases)} case(s) (k={k}) ...")
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

    # --- query ---
    query_p = sub.add_parser("query", help="Retrieve + generate an answer")
    query_p.add_argument("question", help="Question to answer")
    query_p.add_argument(
        "--k", type=int, default=5, help="Number of chunks to retrieve (default: 5)"
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
        asyncio.run(_run_ingest_command(args.files, embedder, session_factory))

    elif args.command == "query":
        embedder, session_factory = _make_components()
        asyncio.run(_run_query_command(args.question, args.k, embedder, session_factory))

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
            )
        )

    else:
        parser.print_help()
        sys.exit(2)
