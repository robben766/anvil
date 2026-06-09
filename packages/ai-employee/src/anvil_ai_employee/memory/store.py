"""Minimal long-term memory. M1: only 'report_marker' — remembers where the reporter
left off so it never repeats. M2 grows this into the full three-tier memory."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import MemoryRow


class MemoryStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._sf = session_factory

    async def write(self, *, employee: str, kind: str, content: str) -> None:
        async with self._sf() as s:
            async with s.begin():
                s.add(MemoryRow(employee=employee, kind=kind, content=content))

    async def last(self, *, employee: str, kind: str) -> str | None:
        """Most recent memory of *kind* for *employee* (pure recency; no vectors in M1).

        Orders by the monotonic ``seq`` Identity column so that two writes in the
        same millisecond are still deterministically ordered by insertion sequence.
        """
        async with self._sf() as s:
            row = (
                await s.execute(
                    select(MemoryRow)
                    .where(MemoryRow.employee == employee)
                    .where(MemoryRow.kind == kind)
                    .order_by(MemoryRow.seq.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            return row.content if row else None
