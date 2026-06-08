"""Run an agent against a bug-fix Task in an isolated worktree, then verify with the
task's command. pass = the verify command succeeds after the agent runs."""

from __future__ import annotations

from dataclasses import dataclass

from anvil_code_agent.eval.task import Task
from anvil_code_agent.harness.loop import run
from anvil_code_agent.sandbox import Worktree
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry
from anvil_code_agent.tools.fs import edit_file, read_file
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
    return ToolRegistry([read_file, edit_file, bash, run_tests])


async def solve_task(task: Task, *, model: str, max_steps: int = 20) -> RunResult:
    with Worktree(task.repo) as wt:
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
