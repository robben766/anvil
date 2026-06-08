"""CLI: anvil-eval run --dataset golden.jsonl --threshold 0.8

P0 阶段答案生成器:把 case.question 直接发给 chat("deepseek-chat") 并以 contexts 为 system 资料。
退出码:达标 0,不达标 1(CI 门禁语义)。
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from anvil_eval.calibration import build_report, load_calibration
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


_CALIB_RUBRIC = (
    "判断【候选答案】相对【参考答案】的正确程度,给一个 0 到 1 的分数:"
    "完全正确=1.0,部分正确=0.5,错误或矛盾=0.0。"
    "只输出 JSON:{\"reason\": \"简短理由\", \"score\": 数字}。"
)


async def _judge_calibration(cases) -> tuple[list[float], list[float]]:
    """Return (judge_scores, human_scores) aligned by case order."""
    from anvil_eval.judge import judge_json

    judge_scores: list[float] = []
    human_scores: list[float] = []
    for c in cases:
        out = await judge_json(
            _CALIB_RUBRIC,
            {"问题": c.question, "参考答案": c.reference, "候选答案": c.answer},
        )
        judge_scores.append(float(out.get("score", 0.0)))
        human_scores.append(c.human_score)
    return judge_scores, human_scores


def _calibrate(args) -> int:
    cases = load_calibration(args.dataset)
    judge_scores, human_scores = asyncio.run(_judge_calibration(cases))
    report = build_report(judge_scores, human_scores)
    print(report.to_markdown())
    if report.kappa < args.threshold:
        print(
            f"⚠️  judge 校准 κ={report.kappa:.3f} 低于阈值 {args.threshold} "
            f"({report.interpretation()}) — judge 评分不可信,需复核 rubric 或换模型。"
        )
    return 0  # calibration is a warning gate, never blocks CI


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
    cal_p = sub.add_parser("calibrate", help="Measure judge↔human agreement (Cohen's kappa)")
    cal_p.add_argument("--dataset", required=True, help="Path to calibration JSONL")
    cal_p.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Warn if kappa falls below this (default: 0.6 = substantial)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        cases = load_dataset(args.dataset)
        report = asyncio.run(run_eval(cases, _default_answer))
        print(report.to_markdown())
        sys.exit(0 if report.passed(args.threshold) else 1)
    elif args.command == "calibrate":
        sys.exit(_calibrate(args))
    else:
        parser.print_help()
        sys.exit(2)
