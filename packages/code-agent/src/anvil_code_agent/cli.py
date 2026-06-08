"""CLI: `anvil-code-agent solve` (one repo+prompt) / `eval` (a tasks.jsonl dataset)."""

from __future__ import annotations

import argparse
import asyncio
import os

from anvil_code_agent.eval.runner import pass_rate, solve_task
from anvil_code_agent.eval.task import Task, load_tasks
from anvil_gateway import configure


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="anvil-code-agent")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("solve", help="solve one bug-fix task")
    s.add_argument("--repo", required=True)
    s.add_argument("--prompt", required=True)
    s.add_argument("--verify-cmd", default="python -m pytest -q")
    s.add_argument("--model", default="deepseek-chat")
    s.add_argument("--max-steps", type=int, default=20)

    e = sub.add_parser("eval", help="run a tasks.jsonl dataset")
    e.add_argument("--dataset", required=True)
    e.add_argument("--model", default="deepseek-chat")
    e.add_argument("--max-steps", type=int, default=20)
    return p


def _configure_gateway() -> None:
    url = os.environ.get("ANVIL_DATABASE_URL")
    if url:
        configure(database_url=url)


async def _run(ns: argparse.Namespace) -> int:
    _configure_gateway()
    if ns.command == "solve":
        task = Task(id="adhoc", repo=ns.repo, prompt=ns.prompt, verify_cmd=ns.verify_cmd)
        res = await solve_task(task, model=ns.model, max_steps=ns.max_steps)
        print(f"task={res.task_id} passed={res.passed} steps={res.steps}")
        print(res.diff)
        return 0 if res.passed else 1
    # eval
    tasks = load_tasks(ns.dataset)
    results = []
    for t in tasks:
        res = await solve_task(t, model=ns.model, max_steps=ns.max_steps)
        print(f"  {res.task_id}: {'PASS' if res.passed else 'FAIL'} ({res.steps} steps)")
        results.append(res)
    rate = pass_rate(results)
    print(f"pass rate: {rate:.2%} ({sum(r.passed for r in results)}/{len(results)})")
    return 0


def main() -> None:
    ns = build_parser().parse_args()
    raise SystemExit(asyncio.run(_run(ns)))


if __name__ == "__main__":
    main()
