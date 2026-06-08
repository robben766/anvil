"""Bug-fix eval task: a repo with a failing test the agent must make pass."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    repo: str          # path to the buggy repo (relative to dataset dir or absolute)
    prompt: str        # instruction given to the agent
    verify_cmd: str    # command whose success == task solved


def load_tasks(path: str) -> list[Task]:
    tasks: list[Task] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tasks.append(
                Task(id=d["id"], repo=d["repo"], prompt=d["prompt"], verify_cmd=d["verify_cmd"])
            )
    return tasks
