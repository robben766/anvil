"""anvil-council: multi-model evaluation jury (圈1 universal)."""

from anvil_council.aggregate import aggregate
from anvil_council.rubric import DEFAULT_RUBRIC, Criterion, Rubric
from anvil_council.seats import score_case, score_one
from anvil_council.verdict import JurorScore, Verdict

__version__ = "0.1.0"
__all__ = [
    "DEFAULT_RUBRIC",
    "Criterion",
    "JurorScore",
    "Rubric",
    "Verdict",
    "aggregate",
    "score_case",
    "score_one",
]
