# packages/ai-employee/src/anvil_ai_employee/fleet/supervisor.py
"""Supervisor: decompose a goal into independent subtasks (one per employee) and fan them
out onto the M1 PG queue. Decomposition uses guard.structured_chat (json mode); illegal or
empty plans fall back so the fleet never spins on nothing."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from anvil_guard import StructuredOutputError, structured_chat
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.scheduler.queue import enqueue

# Each employee maps to the M1 skill its worker registry is built from. Fleet jobs are
# tagged with both employee (selects persona/registry) and this skill (M1 compatibility).
_EMPLOYEE_SKILL = "kb_digest"


@dataclass
class SubTask:
    employee: str
    task: str


_DECOMPOSE_PROMPT = (
    "你是一个团队主管。把下面的总目标拆解成若干条**相互独立**的子任务,每条指派给一名员工。\n"
    """
可用员工(name: 能力):
{roster}

只能指派给上面列出的员工 name。输出 JSON,格式:
{{"subtasks": [{{"employee": "<员工name>", "task": "<该员工要做的具体子任务>"}}, ...]}}

总目标:{goal}"""
)


async def decompose(goal: str, *, model: str, employees: list[str]) -> list[SubTask]:
    roster = "\n".join(f"- {e}" for e in employees)
    prompt = _DECOMPOSE_PROMPT.format(roster=roster, goal=goal)
    try:
        data = await structured_chat(
            model,
            [{"role": "user", "content": prompt}],
            schema={"required": ["subtasks"]},
        )
        raw = data.get("subtasks") or []
    except StructuredOutputError:
        raw = []
    allowed = set(employees)
    subs = [
        SubTask(employee=item["employee"], task=item["task"])
        for item in raw
        if isinstance(item, dict)
        and item.get("employee") in allowed
        and item.get("task")
    ]
    if not subs:
        # Never fan out nothing: assign the whole goal to the first employee.
        subs = [SubTask(employee=employees[0], task=goal)]
    return subs


async def fan_out(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    goal_id: uuid.UUID,
    subtasks: list[SubTask],
) -> list[uuid.UUID]:
    ids: list[uuid.UUID] = []
    for sub in subtasks:
        jid = await enqueue(
            session_factory,
            skill=_EMPLOYEE_SKILL,
            payload={"task": sub.task},
            goal_id=goal_id,
            employee=sub.employee,
        )
        ids.append(jid)
    return ids
