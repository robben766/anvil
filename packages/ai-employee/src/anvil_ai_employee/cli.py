"""anvil-ai-employee CLI: add-schedule / tick / work / run-now / report / chat.

The cron *ticker* (`tick`) and the *worker* (`work`) are separate processes — system
cron can call `tick`, or run `tick --loop` / `work --loop` as long-running daemons.
The `chat` subcommand starts an interactive REPL backed by the given memory strategy."""

from __future__ import annotations

import argparse
import asyncio
import uuid
from datetime import UTC, datetime

from croniter import croniter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import JobRow, ScheduleRow, make_session_factory
from anvil_ai_employee.memory.mem0 import Mem0Strategy
from anvil_ai_employee.memory.strategy import NoMemoryStrategy
from anvil_ai_employee.scheduler.queue import enqueue
from anvil_ai_employee.scheduler.trigger import CronTrigger
from anvil_ai_employee.sessions import SessionStore
from anvil_ai_employee.worker import run_once


async def add_schedule(
    sf: async_sessionmaker[AsyncSession], *, name: str, cron: str, skill: str,
    now: datetime | None = None,
) -> uuid.UUID:
    now = now or datetime.now(UTC)
    next_run = croniter(cron, now).get_next(datetime)
    sid = uuid.uuid4()
    async with sf() as s:
        async with s.begin():
            s.add(ScheduleRow(id=sid, name=name, cron_expr=cron, skill=skill,
                              payload={}, next_run_at=next_run, enabled=True))
    return sid


async def run_now(sf: async_sessionmaker[AsyncSession], *, skill: str) -> uuid.UUID:
    return await enqueue(sf, skill=skill, payload={})


async def show_report(sf: async_sessionmaker[AsyncSession], *, job_id: uuid.UUID) -> str:
    async with sf() as s:
        row = (await s.execute(select(JobRow).where(JobRow.id == job_id))).scalar_one_or_none()
    if row is None:
        return f"未找到 job {job_id}"
    return f"[{row.status}] skill={row.skill}\n{row.result or row.error or '(无输出)'}"


async def _tick_loop(
    sf: async_sessionmaker[AsyncSession], *, once: bool, interval: float = 60.0
) -> None:
    trig = CronTrigger(sf)
    while True:
        n = await trig.due(datetime.now(UTC))
        print(f"tick: enqueued {n} job(s)")
        if once:
            return
        await asyncio.sleep(interval)


async def _work_loop(
    sf: async_sessionmaker[AsyncSession],
    *,
    model: str,
    worker_id: str,
    once: bool,
    interval: float = 5.0,
) -> None:
    while True:
        did = await run_once(sf, model=model, worker_id=worker_id)
        if not did:
            if once:
                return
            await asyncio.sleep(interval)


_DEFAULT_PERSONA = (
    "你是一个有长期记忆的私人助理,会记住用户告诉你的事并在以后用上。"
)


def make_strategy(name: str, sf: async_sessionmaker[AsyncSession], model: str):
    """Pure factory: name ∈ {"none", "mem0"} → concrete MemoryStrategy instance."""
    if name == "none":
        return NoMemoryStrategy()
    if name == "mem0":
        from anvil_kb.embed import FastEmbedEmbedder

        return Mem0Strategy(sf, embedder=FastEmbedEmbedder(), model=model)
    raise ValueError(f"Unknown memory strategy: {name!r}")


def main() -> None:
    p = argparse.ArgumentParser(prog="anvil-ai-employee")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add-schedule")
    a.add_argument("--name", required=True)
    a.add_argument("--cron", required=True)
    a.add_argument("--skill", default="kb_digest")

    t = sub.add_parser("tick")
    t.add_argument("--loop", action="store_true")

    w = sub.add_parser("work")
    w.add_argument("--model", default="deepseek-chat")
    w.add_argument("--worker-id", default="w1")
    w.add_argument("--loop", action="store_true")

    r = sub.add_parser("run-now")
    r.add_argument("--skill", default="kb_digest")

    rp = sub.add_parser("report")
    rp.add_argument("--job", required=True)

    ch = sub.add_parser("chat")
    ch.add_argument("--employee", default="assistant")
    ch.add_argument("--memory", default="mem0", choices=["none", "mem0"])
    ch.add_argument("--model", default="deepseek-chat")
    ch.add_argument("--persona", default=_DEFAULT_PERSONA)

    args = p.parse_args()
    sf = make_session_factory()  # reads ANVIL_DATABASE_URL

    if args.cmd == "add-schedule":
        sid = asyncio.run(add_schedule(sf, name=args.name, cron=args.cron, skill=args.skill))
        print(f"schedule {sid} created")
    elif args.cmd == "tick":
        asyncio.run(_tick_loop(sf, once=not args.loop))
    elif args.cmd == "work":
        asyncio.run(_work_loop(sf, model=args.model, worker_id=args.worker_id, once=not args.loop))
    elif args.cmd == "run-now":
        jid = asyncio.run(run_now(sf, skill=args.skill))
        print(f"job {jid} enqueued")
    elif args.cmd == "report":
        print(asyncio.run(show_report(sf, job_id=uuid.UUID(args.job))))
    elif args.cmd == "chat":
        from anvil_ai_employee.chat import chat_repl

        strategy = make_strategy(args.memory, sf, args.model)
        asyncio.run(
            chat_repl(
                persona=args.persona,
                strategy=strategy,
                employee=args.employee,
                model=args.model,
                session_store=SessionStore(sf),
            )
        )
