"""CLI: anvil-council judge / calibrate.

judge: run the jury over a JSONL of {question, reference, answer} cases and print
each verdict (overall, per-criterion, confidence, disagreements)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from anvil_council.aggregate import aggregate
from anvil_council.rubric import DEFAULT_RUBRIC
from anvil_council.seats import score_case

DEFAULT_MODELS = ["deepseek-chat", "qwen-plus"]


def _load_cases(path: str) -> list[dict]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


async def _run_judge(dataset: str, models: list[str]) -> None:
    cases = _load_cases(dataset)
    for i, case in enumerate(cases, 1):
        scores = await score_case(case, models, DEFAULT_RUBRIC)
        v = aggregate(scores, DEFAULT_RUBRIC)
        print(f"[{i}] {case.get('question', '')[:40]}")
        print(f"    总分={v.overall:.2f}  置信度={v.confidence:.2f}")
        print(f"    分项={ {k: round(x, 2) for k, x in v.per_criterion.items()} }")
        if v.disagreements:
            print(f"    ⚠ 分歧维度: {', '.join(v.disagreements)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anvil-council", description="Multi-model evaluation jury"
    )
    sub = parser.add_subparsers(dest="command")
    j = sub.add_parser("judge", help="Run the jury over a JSONL of {question,reference,answer}")
    j.add_argument("--dataset", required=True, help="Path to cases JSONL")
    j.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated juror models (default: deepseek-chat,qwen-plus)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "judge":
        asyncio.run(
            _run_judge(args.dataset, [m.strip() for m in args.models.split(",") if m.strip()])
        )
    else:
        parser.print_help()
        sys.exit(2)
