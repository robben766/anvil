"""Tests for per-task failure isolation in the eval batch loop."""

from __future__ import annotations

from anvil_code_agent.cli import _run_eval
from anvil_code_agent.eval.runner import RunResult
from anvil_code_agent.eval.task import Task


async def _fake_solve_pass(task, *, model, max_steps):
    return RunResult(task_id=task.id, passed=True, steps=2, diff="")


async def _fake_solve_raise(task, *, model, max_steps):
    raise RuntimeError("agent exploded")


def _make_task(tid):
    return Task(id=tid, repo="/tmp/x", prompt="fix", verify_cmd="pytest")


async def test_eval_batch_continues_after_task_exception(monkeypatch):
    """An exception in one task must not abort the batch; the failing task is
    recorded as passed=False and the rest continue."""
    tasks = [_make_task("ok-task"), _make_task("bad-task")]

    async def _solve(task, *, model, max_steps):
        if task.id == "bad-task":
            raise RuntimeError("agent exploded")
        return RunResult(task_id=task.id, passed=True, steps=2, diff="")

    monkeypatch.setattr("anvil_code_agent.cli.solve_task", _solve)
    results = await _run_eval(tasks, model="deepseek-chat", max_steps=5)

    assert len(results) == 2
    by_id = {r.task_id: r for r in results}
    assert by_id["ok-task"].passed is True
    assert by_id["bad-task"].passed is False


async def test_eval_batch_all_pass(monkeypatch):
    """All tasks passing should yield all passed=True."""
    tasks = [_make_task("t1"), _make_task("t2")]
    monkeypatch.setattr("anvil_code_agent.cli.solve_task", _fake_solve_pass)
    results = await _run_eval(tasks, model="deepseek-chat", max_steps=5)
    assert all(r.passed for r in results)
    assert len(results) == 2
