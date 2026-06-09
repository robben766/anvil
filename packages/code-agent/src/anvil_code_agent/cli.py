"""CLI: `anvil-code-agent solve` (one repo+prompt) / `eval` (a tasks.jsonl dataset)."""

from __future__ import annotations

import argparse
import asyncio
import os

from anvil_code_agent.eval.runner import RunResult, pass_rate, solve_task
from anvil_code_agent.eval.swebench import load_instances, prepare_instance
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
    s.add_argument("--docker", action="store_true", help="run commands in a Docker sandbox")

    e = sub.add_parser("eval", help="run a tasks.jsonl dataset")
    e.add_argument("--dataset", required=True)
    e.add_argument("--model", default="deepseek-chat")
    e.add_argument("--max-steps", type=int, default=20)
    e.add_argument("--docker", action="store_true", help="run commands in a Docker sandbox")

    w = sub.add_parser("swebench", help="run SWE-bench-format instances (clones repos; live)")
    w.add_argument("--dataset", required=True, help="jsonl of SWE-bench instances")
    w.add_argument("--limit", type=int, default=0, help="max instances (0 = all)")
    w.add_argument("--workdir", default="/tmp/anvil-swebench", help="dir to clone repos into")
    w.add_argument("--model", default="deepseek-chat")
    w.add_argument("--max-steps", type=int, default=40)
    w.add_argument("--docker", action="store_true", help="run each instance in a Docker sandbox (installs deps in-container)")
    return p


def _configure_gateway() -> None:
    url = os.environ.get("ANVIL_DATABASE_URL")
    if url:
        configure(database_url=url)


async def _run_eval(
    tasks: list[Task], *, model: str, max_steps: int, use_docker: bool = False
) -> list[RunResult]:
    """Run eval over all tasks, isolating per-task failures so one error never aborts
    the whole batch. A task that raises is recorded as passed=False."""
    results: list[RunResult] = []
    for t in tasks:
        try:
            res = await solve_task(t, model=model, max_steps=max_steps, use_docker=use_docker)
        except Exception as e:  # noqa: BLE001 — a broken task must not abort the batch
            print(f"  {t.id}: ERROR ({e})")
            res = RunResult(task_id=t.id, passed=False, steps=0, diff="")
        else:
            print(f"  {res.task_id}: {'PASS' if res.passed else 'FAIL'} ({res.steps} steps)")
        results.append(res)
    return results


async def _run(ns: argparse.Namespace) -> int:
    _configure_gateway()
    if ns.command == "solve":
        task = Task(id="adhoc", repo=ns.repo, prompt=ns.prompt, verify_cmd=ns.verify_cmd)
        res = await solve_task(task, model=ns.model, max_steps=ns.max_steps, use_docker=ns.docker)
        print(f"task={res.task_id} passed={res.passed} steps={res.steps}")
        print(res.diff)
        return 0 if res.passed else 1
    if ns.command == "swebench":
        instances = load_instances(ns.dataset)
        if ns.limit:
            instances = instances[: ns.limit]
        results = []
        for inst in instances:
            dest = os.path.join(ns.workdir, inst.instance_id)
            try:
                task = prepare_instance(inst, dest)
                res = await solve_task(
                    task,
                    model=ns.model,
                    max_steps=ns.max_steps,
                    use_docker=ns.docker,
                    image=inst.image,
                    setup_cmd=inst.install_cmd or None,
                )
                print(f"  {res.task_id}: {'PASS' if res.passed else 'FAIL'} ({res.steps} steps)")
                results.append(res)
            except Exception as e:  # noqa: BLE001 — one broken instance must not kill the run
                print(f"  {inst.instance_id}: ERROR ({e})")
                results.append(RunResult(task_id=inst.instance_id, passed=False, steps=0, diff=""))
        rate = pass_rate(results)
        print(f"SWE-bench pass@1: {rate:.2%} ({sum(r.passed for r in results)}/{len(results)})")
        return 0
    # eval
    tasks = load_tasks(ns.dataset)
    results = await _run_eval(tasks, model=ns.model, max_steps=ns.max_steps, use_docker=ns.docker)
    rate = pass_rate(results)
    print(f"pass rate: {rate:.2%} ({sum(r.passed for r in results)}/{len(results)})")
    return 0


def main() -> None:
    ns = build_parser().parse_args()
    raise SystemExit(asyncio.run(_run(ns)))


if __name__ == "__main__":
    main()
