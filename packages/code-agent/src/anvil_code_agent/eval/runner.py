"""Run an agent against a bug-fix Task in an isolated worktree, then verify with the
task's command. pass = the verify command succeeds after the agent runs."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass

from anvil_code_agent.eval.task import Task
from anvil_code_agent.harness.loop import run
from anvil_code_agent.sandbox import DockerSandbox, Worktree, has_docker
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry
from anvil_code_agent.tools.fs import edit_file, read_file
from anvil_code_agent.tools.search import grep, repo_map
from anvil_code_agent.tools.shell import bash
from anvil_code_agent.tools.verify import run_tests

SYSTEM_PROMPT = (
    "You are a coding agent. Use the tools to read code, edit files, run the tests, "
    "and keep fixing until the tests pass. When the tests pass, stop and say DONE."
)


@dataclass
class RunResult:
    task_id: str
    passed: bool
    steps: int
    diff: str


def default_registry() -> ToolRegistry:
    return ToolRegistry([read_file, edit_file, bash, run_tests, repo_map, grep])


@contextmanager
def staged_repo(path: str):
    """Yield a path that is a standalone git repo root suitable for Worktree.

    - If *path* is already its own git repo root, yield it unchanged (no copy).
    - Otherwise (non-git dir, or a subdir nested inside a larger repo such as the
      golden fixture nested inside the anvil repo), copy the tree into a fresh
      tempdir, git-init it, commit everything, yield the temp root, then clean up.
    """
    abs_path = os.path.abspath(path)
    # Check if path is its own git root
    result = subprocess.run(
        ["git", "-C", abs_path, "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and os.path.abspath(result.stdout.strip()) == abs_path:
        yield abs_path
        return

    # Not its own git root: copy into a fresh tempdir and init a git repo there
    tmp_root = tempfile.mkdtemp(prefix="anvil-staged-")
    tmp_repo = os.path.join(tmp_root, "repo")
    try:
        shutil.copytree(abs_path, tmp_repo)
        subprocess.run(["git", "init", "-q"], cwd=tmp_repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "staged@anvil"],
            cwd=tmp_repo, check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "anvil-staged"],
            cwd=tmp_repo, check=True,
        )
        subprocess.run(["git", "add", "."], cwd=tmp_repo, check=True)
        subprocess.run(["git", "commit", "-qm", "staged"], cwd=tmp_repo, check=True)
        yield tmp_repo
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


async def solve_task(
    task: Task, *, model: str, max_steps: int = 20, use_docker: bool = False
) -> RunResult:
    with staged_repo(task.repo) as repo_root:
        with Worktree(repo_root) as wt:
            if use_docker and has_docker():
                with DockerSandbox(wt.path) as box:
                    ctx = ToolContext(workdir=wt.path, executor=box.exec)
                    state = AgentState.new(
                        system=SYSTEM_PROMPT, task=task.prompt, workdir=wt.path, max_steps=max_steps
                    )
                    final = await run(state, model, default_registry(), ctx)
                    # Ensure test dependencies (pytest) are available inside the container.
                    # If install fails (e.g. no network), raise immediately so the caller
                    # sees an infra error rather than a silent test-failure.
                    pip_rc, pip_out = box.exec("pip install -q pytest")
                    if pip_rc != 0:
                        raise RuntimeError(
                            f"docker setup failed: pip install pytest rc={pip_rc}: {pip_out[:200]}"
                        )
                    # Note: the container runs as root, so files written into the
                    # bind-mounted worktree are root-owned (acceptable for throwaway
                    # worktrees; a future hardening is to pass --user to `docker run`).
                    # verify runs in-container; wt.diff() runs on the host.
                    rc, out = box.exec(task.verify_cmd)
                    return RunResult(
                        task_id=task.id, passed=rc == 0, steps=final.step, diff=wt.diff()
                    )
            ctx = ToolContext(workdir=wt.path)
            state = AgentState.new(
                system=SYSTEM_PROMPT, task=task.prompt, workdir=wt.path, max_steps=max_steps
            )
            final = await run(state, model, default_registry(), ctx)
            verdict = run_tests({"cmd": task.verify_cmd}, ctx)
            return RunResult(task_id=task.id, passed=verdict.ok, steps=final.step, diff=wt.diff())


def pass_rate(results: list[RunResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.passed) / len(results)
