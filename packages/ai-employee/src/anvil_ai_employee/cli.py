"""anvil-ai-employee CLI: add-schedule / tick / work / run-now / report / chat / inbox / run-hitl.

The cron *ticker* (`tick`) and the *worker* (`work`) are separate processes — system
cron can call `tick`, or run `tick --loop` / `work --loop` as long-running daemons.
The `chat` subcommand starts an interactive REPL backed by the given memory strategy.
The `inbox` subcommand manages the HITL Agent Inbox (list/approve/edit/reject/respond).
The `run-hitl` subcommand runs a demo that suspends on a high-risk tool call."""

from __future__ import annotations

import argparse
import asyncio
import json
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
    """Pure factory: name ∈ {"none", "mem0", "letta"} → concrete MemoryStrategy instance."""
    if name == "none":
        return NoMemoryStrategy()
    if name == "mem0":
        from anvil_kb.embed import FastEmbedEmbedder

        return Mem0Strategy(sf, embedder=FastEmbedEmbedder(), model=model)
    if name == "letta":
        from anvil_ai_employee.memory.letta import LettaStrategy
        from anvil_kb.embed import FastEmbedEmbedder

        return LettaStrategy(sf, embedder=FastEmbedEmbedder(), model=model)
    raise ValueError(f"Unknown memory strategy: {name!r}")


async def inbox_list_text(sf: async_sessionmaker[AsyncSession]) -> str:
    """Return a human-readable summary of pending inbox items (empty = '(空)')."""
    from anvil_ai_employee.inbox import InboxStore

    rows = await InboxStore(sf).list_pending()
    if not rows:
        return "Agent Inbox (空) — 没有待审批的操作。"
    lines = ["Agent Inbox — 待审批操作:"]
    for row in rows:
        args_summary = str(row.tool_args)[:60]
        lines.append(
            f"  [{str(row.id)[:8]}] employee={row.employee}"
            f"  tool={row.tool_name}  risk={row.risk}"
            f"  args={args_summary}"
        )
    return "\n".join(lines)


async def inbox_resolve(
    sf: async_sessionmaker[AsyncSession],
    *,
    inbox_id: uuid.UUID,
    decision: str,
    payload: dict,
) -> None:
    """Resolve an inbox item and write the intervention to long-term memory."""
    from anvil_ai_employee.hitl_memory import record_intervention
    from anvil_ai_employee.inbox import InboxStore
    from anvil_kb.embed import FastEmbedEmbedder

    store = InboxStore(sf)
    # Fetch before resolving so we have tool_name/tool_args/employee
    row = await store.get(inbox_id)
    if row is None:
        raise ValueError(f"inbox item {inbox_id} not found")
    await store.resolve(inbox_id, decision=decision, payload=payload)
    await record_intervention(
        sf,
        embedder=FastEmbedEmbedder(),
        employee=row.employee,
        tool_name=row.tool_name,
        decision=decision,
        payload=payload,
        tool_args=row.tool_args,
    )


async def _run_hitl_demo(
    sf: async_sessionmaker[AsyncSession],
    *,
    persona: str,
    task: str,
    employee: str,
    model: str,
) -> None:
    """Demo: run hitl_run with a high-risk 'shell' tool until it suspends, then persist to inbox."""
    from anvil_code_agent.state import AgentState
    from anvil_code_agent.tools.base import ToolContext, ToolRegistry, ToolResult, tool

    from anvil_ai_employee.hitl import hitl_run, suspend_high
    from anvil_ai_employee.inbox import InboxStore

    @tool(name="shell", description="Execute a shell command.", params={"cmd": {"type": "string"}},
          required=["cmd"])
    def shell(args: dict, ctx: ToolContext) -> ToolResult:  # pragma: no cover
        return ToolResult(content=f"ran: {args['cmd']}", ok=True)

    registry = ToolRegistry([shell])
    ctx = ToolContext(workdir="/tmp")
    state = AgentState.new(system=persona, task=task, workdir="/tmp", max_steps=10)

    print(f"[run-hitl] 开始跑 model={model}, employee={employee}")
    print(f"[run-hitl] task: {task}")

    try:
        out = await hitl_run(state, model, registry, ctx, policy=suspend_high)
    except Exception as exc:  # noqa: BLE001
        print(f"[run-hitl] 模型调用失败(可能缺 API key): {exc}")
        print("[run-hitl] 提示: 设置 DEEPSEEK_API_KEY 后重试")
        return

    if out.status == "suspended":
        iid = await InboxStore(sf).suspend(employee=employee, state=out)
        print("[run-hitl] Agent 挂起!高风险动作已进入 Inbox。")
        print(f"[run-hitl] inbox_id = {iid}")
        print("[run-hitl] 查看: anvil-ai-employee inbox list")
        print(f"[run-hitl] 审批: anvil-ai-employee inbox approve {iid}")
    elif out.status == "done":
        print(f"[run-hitl] Agent 完成(未触发挂起)。步数={out.step}")
    else:
        print(f"[run-hitl] 终态: {out.status}")


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
    ch.add_argument("--memory", default="mem0", choices=["none", "mem0", "letta"])
    ch.add_argument("--model", default="deepseek-chat")
    ch.add_argument("--persona", default=_DEFAULT_PERSONA)

    # inbox subcommand — HITL Agent Inbox management
    inbox_p = sub.add_parser("inbox")
    inbox_sub = inbox_p.add_subparsers(dest="inbox_cmd", required=True)
    inbox_sub.add_parser("list")

    ib_approve = inbox_sub.add_parser("approve")
    ib_approve.add_argument("id")

    ib_edit = inbox_sub.add_parser("edit")
    ib_edit.add_argument("id")
    ib_edit.add_argument("--args", required=True, dest="args_json", metavar="JSON")

    ib_reject = inbox_sub.add_parser("reject")
    ib_reject.add_argument("id")
    ib_reject.add_argument("--reason", required=True)

    ib_respond = inbox_sub.add_parser("respond")
    ib_respond.add_argument("id")
    ib_respond.add_argument("--message", required=True)

    # run-hitl demo subcommand
    rh = sub.add_parser("run-hitl")
    rh.add_argument("--persona", default="你是一个 AI 助理。")
    rh.add_argument("--task", default="请帮我清理临时文件。")
    rh.add_argument("--employee", default="assistant")
    rh.add_argument("--model", default="deepseek-chat")

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
        paging = None
        if args.memory == "letta":
            from anvil_code_agent.harness.context import llm_summarizer

            paging = {
                "budget": 6000,
                "warn_ratio": 0.7,
                "summarizer": llm_summarizer(args.model),
            }
        asyncio.run(
            chat_repl(
                persona=args.persona,
                strategy=strategy,
                employee=args.employee,
                model=args.model,
                session_store=SessionStore(sf),
                paging=paging,
            )
        )
    elif args.cmd == "inbox":
        if args.inbox_cmd == "list":
            print(asyncio.run(inbox_list_text(sf)))
        else:
            iid = uuid.UUID(args.id)
            if args.inbox_cmd == "approve":
                payload: dict = {}
                decision = "approve"
            elif args.inbox_cmd == "edit":
                payload = {"args": json.loads(args.args_json)}
                decision = "edit"
            elif args.inbox_cmd == "reject":
                payload = {"reason": args.reason}
                decision = "reject"
            elif args.inbox_cmd == "respond":
                payload = {"message": args.message}
                decision = "respond"
            else:
                p.error(f"unknown inbox subcommand: {args.inbox_cmd}")
            asyncio.run(inbox_resolve(sf, inbox_id=iid, decision=decision, payload=payload))
            print(f"inbox {str(iid)[:8]} → {decision} 已记录")
    elif args.cmd == "run-hitl":
        asyncio.run(_run_hitl_demo(sf, persona=args.persona, task=args.task,
                                   employee=args.employee, model=args.model))
