"""Verdict data model: a juror's scores and the aggregated jury verdict."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class JurorScore:
    model: str
    per_criterion: dict[str, float]  # criterion key -> score in [0,1]
    overall: float
    reason: str


@dataclass
class Verdict:
    overall: float
    per_criterion: dict[str, float]
    confidence: float  # 1 - mean per-criterion spread, clamped to [0,1]
    disagreements: list[str]  # criterion keys where jurors diverged beyond threshold
    jurors: list[JurorScore] = field(default_factory=list)
