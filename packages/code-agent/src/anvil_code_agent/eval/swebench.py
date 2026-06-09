"""Adapt a SWE-bench(-Lite) instance to anvil's bug-fix Task: clone the repo at
base_commit, git-apply the test_patch (which adds the failing tests) and commit it so a
worktree's HEAD carries it, then verify the FAIL_TO_PASS tests. We do NOT reproduce the
official per-instance Docker harness — environment build is the benchmark's own concern."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field

from anvil_code_agent.eval.task import Task


def _as_list(v: object) -> list[str]:
    if isinstance(v, str):
        return list(json.loads(v))
    return list(v) if v else []


@dataclass
class SweInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str] = field(default_factory=list)


def load_instances(path: str) -> list[SweInstance]:
    out: list[SweInstance] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(
                SweInstance(
                    instance_id=d["instance_id"],
                    repo=d["repo"],
                    base_commit=d["base_commit"],
                    problem_statement=d["problem_statement"],
                    test_patch=d.get("test_patch", ""),
                    fail_to_pass=_as_list(d.get("FAIL_TO_PASS", [])),
                    pass_to_pass=_as_list(d.get("PASS_TO_PASS", [])),
                )
            )
    return out
