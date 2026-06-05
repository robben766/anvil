"""golden set 数据集:JSONL,一行一个用例。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_REQUIRED = ("id", "question", "reference")


@dataclass
class GoldenCase:
    id: str
    question: str
    reference: str
    contexts: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def load_dataset(path: str) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in _REQUIRED:
            if not row.get(key):
                raise ValueError(f"line {lineno}: missing required field {key!r}")
        if row["id"] in seen:
            raise ValueError(f"line {lineno}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        cases.append(
            GoldenCase(
                id=row["id"],
                question=row["question"],
                reference=row["reference"],
                contexts=row.get("contexts") or [],
                tags=row.get("tags") or [],
            )
        )
    return cases
