"""Aggregator: once every child job of a goal is terminal (done or failed), synthesize
their outputs into the goal's final result. Failed subtasks are included and labelled so
the final deliverable honestly reflects what did not complete (ACI, at the fleet level)."""

from __future__ import annotations

import uuid

from anvil_obs import span
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import GoalRow, JobRow
from anvil_gateway import chat

_TERMINAL = {"done", "failed"}


async def children_terminal(
    session_factory: async_sessionmaker[AsyncSession], goal_id: uuid.UUID
) -> bool:
    async with session_factory() as s:
        rows = (
            (await s.execute(select(JobRow).where(JobRow.goal_id == goal_id))).scalars().all()
        )
    if not rows:
        return True  # no children → trivially terminal
    return all(r.status in _TERMINAL for r in rows)


async def aggregate(
    session_factory: async_sessionmaker[AsyncSession],
    goal_id: uuid.UUID,
    *,
    model: str,
) -> str | None:
    """Return the synthesized final result (and persist it) if all children are terminal;
    otherwise return None without writing. Idempotent: if the goal is already done, return
    its stored result without re-synthesizing (no extra LLM call, no overwrite)."""
    async with session_factory() as s:
        goal = (await s.execute(select(GoalRow).where(GoalRow.id == goal_id))).scalar_one()
        children = (
            (await s.execute(select(JobRow).where(JobRow.goal_id == goal_id))).scalars().all()
        )
    if goal.status == "done":
        return goal.result  # already aggregated — idempotent, no re-synthesis
    if not children:
        # No children yet (fan_out hasn't run): NOT terminal. Avoids a vacuous "done" and
        # the orphan-children race if status is polled between goal creation and fan_out.
        return None
    if not all(c.status in _TERMINAL for c in children):
        return None
    parts = []
    for c in children:
        if c.status == "done":
            parts.append(f"### 员工 {c.employee} 的产出\n{c.result or ''}")
        else:
            parts.append(f"### 员工 {c.employee}(未完成: {c.error or '失败'})")
    synthesis_prompt = (
        f"总目标:{goal.objective}\n\n以下是各员工的子任务产出,请综合成一份连贯的最终中文交付物;"
        f"若某员工未完成,在最终产出中如实标注该部分缺失。\n\n" + "\n\n".join(parts)
    )
    with span("ai_employee.fleet.aggregate", goal=str(goal_id)):
        resp = await chat(model, [{"role": "user", "content": synthesis_prompt}])
        final = resp.raw["choices"][0]["message"]["content"] or ""
    async with session_factory() as s:
        async with s.begin():
            await s.execute(
                update(GoalRow)
                .where(GoalRow.id == goal_id, GoalRow.status != "done")
                .values(status="done", result=final, finished_at=func.now())
            )
    return final
