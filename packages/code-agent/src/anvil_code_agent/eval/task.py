"""Bug-fix eval task: a repo with a failing test the agent must make pass."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    repo: str          # absolute path after load; relative entries in jsonl are resolved
                       # against the dataset dir. Constructing Task directly with an
                       # absolute path (e.g. in tests) is also fine.
    prompt: str        # instruction given to the agent
    verify_cmd: str    # command whose success == task solved


def load_tasks(path: str) -> list[Task]:
    dataset_dir = os.path.dirname(os.path.abspath(path))
    tasks: list[Task] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            repo = d["repo"]
            if not os.path.isabs(repo):
                repo = os.path.join(dataset_dir, repo)
            tasks.append(
                Task(id=d["id"], repo=repo, prompt=d["prompt"], verify_cmd=d["verify_cmd"])
            )
    return tasks
