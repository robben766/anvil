"""Aggregate independent juror scores into a verdict.

Per-criterion median (robust to one outlier juror) + disagreement flag when the
spread (max-min) on a criterion exceeds a threshold. Confidence = 1 - mean spread.
Pure function, no network."""

from __future__ import annotations

import statistics

from anvil_council.rubric import DEFAULT_RUBRIC, Rubric
from anvil_council.verdict import JurorScore, Verdict


def aggregate(
    scores: list[JurorScore],
    rubric: Rubric = DEFAULT_RUBRIC,
    disagreement_threshold: float = 0.5,
) -> Verdict:
    if not scores:
        raise ValueError("cannot aggregate an empty juror list")
    per_criterion: dict[str, float] = {}
    disagreements: list[str] = []
    spreads: list[float] = []
    for c in rubric.criteria:
        vals = [s.per_criterion[c.key] for s in scores]
        per_criterion[c.key] = statistics.median(vals)
        spread = max(vals) - min(vals)
        spreads.append(spread)
        if spread > disagreement_threshold:
            disagreements.append(c.key)
    overall = statistics.median([s.overall for s in scores])
    confidence = max(0.0, 1.0 - (sum(spreads) / len(spreads)))
    return Verdict(
        overall=overall,
        per_criterion=per_criterion,
        confidence=confidence,
        disagreements=disagreements,
        jurors=list(scores),
    )
