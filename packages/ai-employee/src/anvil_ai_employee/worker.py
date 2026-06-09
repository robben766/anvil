"""Worker: claim a job, run the matching skill's agent loop (reusing P3 harness),
persist outcome. submit_report already marks the job done; worker handles the rest."""

from __future__ import annotations

import tempfile

from anvil_code_agent.harness.loop import run as agent_run
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext
from anvil_obs import span
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import JobRow
from anvil_ai_employee.fleet.team import EMPLOYEES
from anvil_ai_employee.scheduler.queue import claim_one, fail
from anvil_ai_employee.tools import EmployeeContext


async def _job_status(session_factory, job_id) -> str:
    async with session_factory() as s:
        row = (await s.execute(select(JobRow).where(JobRow.id == job_id))).scalar_one()
        return row.status


async def run_once(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    model: str,
    worker_id: str,
    max_steps: int = 12,
) -> bool:
    """Claim and run one job. Returns False if the queue was empty."""
    job = await claim_one(session_factory, worker_id=worker_id)
    if job is None:
        return False
    with span("ai_employee.job", skill=job.skill, worker=worker_id):
        try:
            employee_name = job.employee or "kb_reporter"
            emp = EMPLOYEES.get(employee_name)
            if emp is None:
                await fail(
                    session_factory, job.id, error=f"unknown employee: {employee_name}"
                )
                return True
            ctx = EmployeeContext(
                session_factory=session_factory, employee=employee_name, job_id=job.id
            )
            registry = emp.build_registry(ctx)
            task = job.payload.get("task") or emp.task_prompt
            with tempfile.TemporaryDirectory() as workdir:
                state = AgentState.new(
                    system=emp.persona, task=task, workdir=workdir, max_steps=max_steps
                )
                tc = ToolContext(workdir=workdir)
                await agent_run(state, model, registry, tc)
            # submit_report already completed the job; fail if the agent never submitted
            if await _job_status(session_factory, job.id) != "done":
                await fail(
                    session_factory,
                    job.id,
                    error="agent finished without calling submit_report",
                )
        except Exception as e:  # noqa: BLE001 — isolate one job's failure
            await fail(session_factory, job.id, error=f"{type(e).__name__}: {e}")
    return True
