"""EvalReport + run_eval: aggregate four RAGAS-style metrics over a golden set.

Cases without contexts skip faithfulness/context_recall/context_precision
(those three are recorded as None and excluded from means).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from anvil_eval.dataset import GoldenCase
from anvil_eval.metrics.answer_relevancy import answer_relevancy
from anvil_eval.metrics.context_precision import context_precision
from anvil_eval.metrics.context_recall import context_recall
from anvil_eval.metrics.faithfulness import faithfulness

_METRICS = ("faithfulness", "context_recall", "answer_relevancy", "context_precision")


@dataclass
class EvalReport:
    per_case: dict[str, dict[str, float | None]] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)

    @property
    def overall(self) -> float:
        if not self.means:
            return 0.0
        return sum(self.means.values()) / len(self.means)

    def passed(self, threshold: float) -> bool:
        return self.overall >= threshold

    def to_markdown(self) -> str:
        header = "| case | faithfulness | context_recall | answer_relevancy | context_precision |"
        sep = "| --- | --- | --- | --- | --- |"
        rows = [header, sep]
        for case_id, scores in self.per_case.items():
            def _fmt(v: float | None) -> str:
                return f"{v:.4f}" if v is not None else "-"
            row = (
                f"| {case_id}"
                f" | {_fmt(scores.get('faithfulness'))}"
                f" | {_fmt(scores.get('context_recall'))}"
                f" | {_fmt(scores.get('answer_relevancy'))}"
                f" | {_fmt(scores.get('context_precision'))}"
                " |"
            )
            rows.append(row)
        # means row
        def _mean_fmt(k: str) -> str:
            v = self.means.get(k)
            return f"{v:.4f}" if v is not None else "-"
        rows.append(
            f"| **means**"
            f" | {_mean_fmt('faithfulness')}"
            f" | {_mean_fmt('context_recall')}"
            f" | {_mean_fmt('answer_relevancy')}"
            f" | {_mean_fmt('context_precision')}"
            " |"
        )
        rows.append(f"\n**overall**: {self.overall:.4f}")
        return "\n".join(rows)


async def run_eval(
    cases: list[GoldenCase],
    answer_fn: Callable[[GoldenCase], Awaitable[str]],
) -> EvalReport:
    per_case: dict[str, dict[str, float | None]] = {}

    for case in cases:
        answer = await answer_fn(case)
        scores: dict[str, float | None] = {}

        if case.contexts:
            scores["faithfulness"] = await faithfulness(
                answer=answer, contexts=case.contexts
            )
            scores["context_recall"] = await context_recall(
                reference=case.reference, contexts=case.contexts
            )
            scores["answer_relevancy"] = await answer_relevancy(
                question=case.question, answer=answer
            )
            scores["context_precision"] = await context_precision(
                question=case.question, contexts=case.contexts
            )
        else:
            scores["faithfulness"] = None
            scores["context_recall"] = None
            scores["answer_relevancy"] = await answer_relevancy(
                question=case.question, answer=answer
            )
            scores["context_precision"] = None

        per_case[case.id] = scores

    # compute means: for each metric, average over non-None values; skip metric if all None
    means: dict[str, float] = {}
    for metric in _METRICS:
        values = [s[metric] for s in per_case.values() if s.get(metric) is not None]
        if values:
            means[metric] = sum(values) / len(values)

    return EvalReport(per_case=per_case, means=means)
