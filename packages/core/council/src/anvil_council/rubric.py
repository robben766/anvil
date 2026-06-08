"""Scoring rubric = a set of FUNCTIONAL criteria (correctness/evidence/...), NOT
hand-written personas. Jurors all score the same rubric independently."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Criterion:
    key: str
    description: str


@dataclass(frozen=True)
class Rubric:
    name: str
    criteria: tuple[Criterion, ...]

    def keys(self) -> list[str]:
        return [c.key for c in self.criteria]


DEFAULT_RUBRIC = Rubric(
    name="general-answer-quality",
    criteria=(
        Criterion("correctness", "答案是否与参考一致、事实正确,无错误或矛盾"),
        Criterion("evidence", "答案是否有充分依据/证据支撑,而非凭空断言"),
        Criterion("completeness", "答案是否完整覆盖问题的关键要点"),
        Criterion("relevance", "答案是否切题,无离题或冗余内容"),
    ),
)
