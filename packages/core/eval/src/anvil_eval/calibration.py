"""Judge calibration: measure agreement between LLM-judge scores and human labels.

Hand-rolled Cohen's κ (no sklearn) — quantize both continuous scores into 3 ordinal
buckets, then compute κ. κ interpretation: <0.2 poor, 0.2–0.4 fair, 0.4–0.6 moderate,
0.6–0.8 substantial, >0.8 near-perfect. A low κ means the judge cannot be trusted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def quantize(score: float) -> int:
    """Map a [0,1] score to an ordinal bucket: 0 (low) / 1 (mid) / 2 (high)."""
    if score < 1 / 3:
        return 0
    if score < 2 / 3:
        return 1
    return 2


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's κ for two equal-length lists of categorical labels."""
    if len(a) != len(b):
        raise ValueError("rater label lists must be equal length")
    if not a:
        raise ValueError("cannot compute kappa over empty input")
    n = len(a)
    categories = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pe = 0.0
    for c in categories:
        pa = sum(1 for x in a if x == c) / n
        pb = sum(1 for y in b if y == c) / n
        pe += pa * pb
    if pe >= 1.0:  # both raters used a single identical category → perfect by definition
        return 1.0
    return (po - pe) / (1 - pe)


@dataclass
class CalibrationCase:
    id: str
    question: str
    reference: str
    answer: str
    human_score: float


def load_calibration(path: str) -> list[CalibrationCase]:
    cases: list[CalibrationCase] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "question", "reference", "answer", "human_score"):
            if key not in row:
                raise ValueError(f"line {lineno}: missing required field {key!r}")
        if row["id"] in seen:
            raise ValueError(f"line {lineno}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        cases.append(
            CalibrationCase(
                id=row["id"],
                question=row["question"],
                reference=row["reference"],
                answer=row["answer"],
                human_score=float(row["human_score"]),
            )
        )
    return cases


@dataclass
class CalibrationReport:
    kappa: float
    n: int
    judge_labels: list[int]
    human_labels: list[int]

    def interpretation(self) -> str:
        k = self.kappa
        if k < 0.2:
            return "poor"
        if k < 0.4:
            return "fair"
        if k < 0.6:
            return "moderate"
        if k < 0.8:
            return "substantial"
        return "near-perfect"

    def to_markdown(self) -> str:
        return (
            f"## Judge Calibration\n\n"
            f"- n = {self.n}\n"
            f"- Cohen's κ = {self.kappa:.3f} ({self.interpretation()})\n"
        )


def build_report(judge_scores: list[float], human_scores: list[float]) -> CalibrationReport:
    """Quantize both score lists and compute the κ report."""
    judge_labels = [quantize(s) for s in judge_scores]
    human_labels = [quantize(s) for s in human_scores]
    kappa = cohen_kappa(judge_labels, human_labels)
    return CalibrationReport(
        kappa=kappa, n=len(judge_scores), judge_labels=judge_labels, human_labels=human_labels
    )
