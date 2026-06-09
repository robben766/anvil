"""Agent Inbox store: persist a suspended agent for human review and resolution."""

from __future__ import annotations

import uuid

from anvil_code_agent.harness.permission import risk_level
from anvil_code_agent.harness.recovery import dump_state
from anvil_code_agent.state import AgentState
from sqlalchemy import select
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql import func

from anvil_ai_employee.db import InboxRow
from anvil_ai_employee.hitl import _args_of, _unanswered_tool_calls


class InboxStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._sf = session_factory

    async def suspend(
        self, *, employee: str, state: AgentState, job_id: uuid.UUID | None = None
    ) -> uuid.UUID:
        pending = _unanswered_tool_calls(state)
        if not pending:
            raise ValueError("suspend: state has no pending tool call")
        tc = pending[0]
        name = tc["function"]["name"]
        iid = uuid.uuid4()
        async with self._sf() as s:
            async with s.begin():
                s.add(
                    InboxRow(
                        id=iid,
                        job_id=job_id,
                        employee=employee,
                        tool_name=name,
                        tool_args=_args_of(tc),
                        risk=risk_level(name),
                        state_json=dump_state(state),
                        status="pending",
                    )
                )
        return iid

    async def list_pending(self, *, employee: str | None = None) -> list[InboxRow]:
        async with self._sf() as s:
            q = select(InboxRow).where(InboxRow.status == "pending").order_by(InboxRow.created_at)
            if employee is not None:
                q = q.where(InboxRow.employee == employee)
            return list((await s.execute(q)).scalars().all())

    async def get(self, inbox_id: uuid.UUID) -> InboxRow | None:
        async with self._sf() as s:
            return (
                await s.execute(select(InboxRow).where(InboxRow.id == inbox_id))
            ).scalar_one_or_none()

    async def resolve(self, inbox_id: uuid.UUID, *, decision: str, payload: dict) -> None:
        async with self._sf() as s:
            async with s.begin():
                await s.execute(
                    sa_update(InboxRow)
                    .where(InboxRow.id == inbox_id)
                    .where(InboxRow.status == "pending")  # idempotent: only resolve once
                    .values(
                        status="resolved",
                        decision=decision,
                        decision_payload=payload,
                        resolved_at=func.now(),
                    )
                )
