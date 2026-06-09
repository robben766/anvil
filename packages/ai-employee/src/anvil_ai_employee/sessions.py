"""Session tier: persist a chat's conversation messages (NOT the per-turn re-injected
system block) so a session can resume across runs. messages are plain dicts (P3 fact)."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import SessionRow


class SessionStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._sf = session_factory

    async def create(self, *, employee: str) -> uuid.UUID:
        sid = uuid.uuid4()
        async with self._sf() as s:
            async with s.begin():
                s.add(SessionRow(id=sid, employee=employee, messages=[], status="active"))
        return sid

    async def save(self, sid: uuid.UUID, messages: list[dict], *, status: str = "active") -> None:
        async with self._sf() as s:
            async with s.begin():
                await s.execute(sa_update(SessionRow).where(SessionRow.id == sid)
                                .values(messages=messages, status=status))

    async def load(self, sid: uuid.UUID) -> tuple[dict, ...]:
        async with self._sf() as s:
            result = await s.execute(select(SessionRow).where(SessionRow.id == sid))
            row = result.scalar_one_or_none()
        return tuple(row.messages) if row else ()
