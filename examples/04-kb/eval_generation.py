"""KB-M1b: Generation-side RAGAS live eval.

Runs faithfulness + answer_relevancy on the 10 answerable cases in
packages/kb/golden/kb.jsonl. Uses the running anvil-postgres (port 5434)
and anvil-gateway (DEEPSEEK_API_KEY) via .env.

Usage:
    cd /home/itachi/workspace/ai/anvil
    set -a; source .env; set +a
    uv run python examples/04-kb/eval_generation.py

Outputs per-case scores and means to stdout.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from anvil_eval.metrics.answer_relevancy import answer_relevancy
from anvil_eval.metrics.faithfulness import faithfulness
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_kb.db import make_engine
from anvil_kb.embed import FastEmbedEmbedder
from anvil_kb.generate import answer
from anvil_kb.retrieve.retriever import Retriever
from anvil_kb.store.pg import PgVectorStore

REPO_ROOT = Path(__file__).resolve().parents[2]
KB_JSONL = REPO_ROOT / "packages/kb/golden/kb.jsonl"


def load_answerable_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("answerable", True):
            cases.append(row)
    return cases


async def main() -> None:
    # Ensure DATABASE_URL is set (fallback for direct runs without .env)
    if not os.environ.get("ANVIL_DATABASE_URL"):
        os.environ["ANVIL_DATABASE_URL"] = (
            "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil"
        )

    engine = make_engine()
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    embedder = FastEmbedEmbedder()
    store = PgVectorStore(session_factory)
    retriever = Retriever(embedder, store)

    cases = load_answerable_cases(KB_JSONL)
    print(f"Loaded {len(cases)} answerable cases\n")

    results = []
    for case in cases:
        qid = case["id"]
        question = case["question"]
        print(f"[{qid}] {question}")

        # Retrieve k=5 contexts
        retrieved = await retriever.retrieve(question, k=5)
        contexts = [sc.chunk.content for sc in retrieved]

        # Generate answer
        kb_answer = await answer(question, retrieved, session_id=f"eval-{qid}")
        ans_text = kb_answer.text
        print(f"  answer: {ans_text[:80]}...")

        # Evaluate
        faith = await faithfulness(answer=ans_text, contexts=contexts)
        relevancy = await answer_relevancy(question=question, answer=ans_text)

        print(f"  faithfulness={faith:.4f}  answer_relevancy={relevancy:.4f}")
        results.append(
            {
                "id": qid,
                "faithfulness": faith,
                "answer_relevancy": relevancy,
            }
        )

    # --- print score table ---
    print("\n\n## Generation-side RAGAS Scores\n")
    print("| case | faithfulness | answer_relevancy |")
    print("| --- | --- | --- |")
    for r in results:
        print(f"| {r['id']} | {r['faithfulness']:.4f} | {r['answer_relevancy']:.4f} |")

    mean_faith = sum(r["faithfulness"] for r in results) / len(results)
    mean_rel = sum(r["answer_relevancy"] for r in results) / len(results)
    print(f"| **means** | {mean_faith:.4f} | {mean_rel:.4f} |")
    print(f"\n**overall**: {(mean_faith + mean_rel) / 2:.4f}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
