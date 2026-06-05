"""CLI: anvil-eval run --dataset golden.jsonl --threshold 0.8

P0 阶段答案生成器:把 case.question 直接发给 chat("deepseek-chat") 并以 contexts 为 system 资料。
退出码:达标 0,不达标 1(CI 门禁语义)。
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from anvil_eval.dataset import GoldenCase, load_dataset
from anvil_eval.runner import run_eval
from anvil_gateway import chat


async def _default_answer(case: GoldenCase) -> str:
    system_content = "仅根据给定资料回答,资料:" + "\n".join(case.contexts)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": case.question},
    ]
    resp = await chat("deepseek-chat", messages, session_id="anvil-eval-run")
    return resp.content or ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anvil-eval",
        description="Run RAGAS-style evaluation over a golden JSONL dataset.",
    )
    sub = parser.add_subparsers(dest="command")
    run_p = sub.add_parser("run", help="Evaluate a dataset")
    run_p.add_argument("--dataset", required=True, help="Path to golden JSONL file")
    run_p.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Overall score threshold for CI gate (default: 0.8)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command != "run":
        parser.print_help()
        sys.exit(2)

    cases = load_dataset(args.dataset)
    report = asyncio.run(run_eval(cases, _default_answer))
    print(report.to_markdown())

    if report.passed(args.threshold):
        sys.exit(0)
    else:
        sys.exit(1)
