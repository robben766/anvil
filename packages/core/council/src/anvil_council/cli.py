"""CLI: anvil-council judge / calibrate.

judge: run the jury over a JSONL of {question, reference, answer} cases and print
each verdict (overall, per-criterion, confidence, disagreements)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from anvil_eval.calibration import load_calibration

from anvil_council.aggregate import aggregate
from anvil_council.agreement import compare_jury
from anvil_council.rubric import DEFAULT_RUBRIC
from anvil_council.seats import score_case

DEFAULT_MODELS = ["deepseek-chat", "qwen-plus"]


def _load_env() -> None:
    """Load .env from the nearest ancestor of cwd so live CLI runs pick up API keys
    and ANVIL_DATABASE_URL. Graceful no-op if python-dotenv is absent or no .env found."""
    try:
        from dotenv import load_dotenv  # type: ignore[import]
    except ImportError:
        return
    here = Path.cwd()
    for candidate in [here, *here.parents]:
        env_file = candidate / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            break


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


async def _run_calibrate(dataset: str, models: list[str]) -> int:
    cases = load_calibration(dataset)
    juror_overalls: dict[str, list[float]] = {m: [] for m in models}
    jury_overalls: list[float] = []
    human_scores: list[float] = []
    skipped = 0
    for case in cases:
        payload = {"question": case.question, "reference": case.reference, "answer": case.answer}
        try:
            scores = await score_case(payload, models, DEFAULT_RUBRIC)
        except Exception as e:  # noqa: BLE001 — a flaky juror should not kill the batch
            skipped += 1
            print(f"  (skipped {case.id}: {type(e).__name__})")
            continue
        v = aggregate(scores, DEFAULT_RUBRIC)
        jury_overalls.append(v.overall)
        human_scores.append(case.human_score)
        by_model = {s.model: s.overall for s in scores}
        for m in models:
            juror_overalls[m].append(by_model.get(m, 0.0))
    if not human_scores:
        print("没有可用的评分结果(全部跳过),无法计算 κ。")
        return 0
    report = compare_jury(juror_overalls, jury_overalls, human_scores)
    print(report.to_markdown())
    if skipped:
        print(f"(共跳过 {skipped} 条因评委输出无效)")
    if not report.jury_beats_best_single:
        print(
            "⚠️  陪审团未跑赢最佳单评委 —— 多模型集成在本集上没有增益,如实记录"
            "(可能是强/弱模型平均拉低,或样本太小)。"
        )
    return 0  # diagnostic, never blocks CI


def _calibrate(args) -> int:
    return asyncio.run(
        _run_calibrate(args.dataset, [m.strip() for m in args.models.split(",") if m.strip()])
    )


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
    c = sub.add_parser("calibrate", help="Compare jury vs human vs best-single (Cohen's κ)")
    c.add_argument("--dataset", required=True, help="Path to calibration JSONL (with human_score)")
    c.add_argument(
        "--models", default=",".join(DEFAULT_MODELS), help="Comma-separated juror models"
    )
    return parser


def main() -> None:
    _load_env()
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "judge":
        asyncio.run(
            _run_judge(args.dataset, [m.strip() for m in args.models.split(",") if m.strip()])
        )
    elif args.command == "calibrate":
        sys.exit(_calibrate(args))
    else:
        parser.print_help()
        sys.exit(2)
