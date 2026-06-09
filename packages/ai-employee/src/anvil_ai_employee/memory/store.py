"""Minimal long-term memory. M1: only 'report_marker' — remembers where the reporter
left off so it never repeats. M2 grows this into the full three-tier memory."""

from __future__ import annotations

import uuid

from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import MemoryRow


class MemoryStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._sf = session_factory

    async def write(self, *, employee: str, kind: str, content: str) -> None:
        async with self._sf() as s:
            async with s.begin():
                s.add(MemoryRow(employee=employee, kind=kind, content=content))

    async def insert(
        self,
        *,
        employee: str,
        kind: str,
        content: str,
        embedding: list[float] | None = None,
    ) -> uuid.UUID:
        mid = uuid.uuid4()
        async with self._sf() as s:
            async with s.begin():
                s.add(
                    MemoryRow(
                        id=mid,
                        employee=employee,
                        kind=kind,
                        content=content,
                        embedding=embedding,
                    )
                )
        return mid

    async def update(
        self, mem_id: uuid.UUID, *, content: str, embedding: list[float] | None = None
    ) -> None:
        async with self._sf() as s:
            async with s.begin():
                await s.execute(
                    sa_update(MemoryRow)
                    .where(MemoryRow.id == mem_id)
                    .values(content=content, embedding=embedding)
                )

    async def delete(self, mem_id: uuid.UUID) -> None:
        async with self._sf() as s:
            async with s.begin():
                await s.execute(sa_delete(MemoryRow).where(MemoryRow.id == mem_id))

    async def list_facts(self, *, employee: str, kind: str = "fact") -> list[MemoryRow]:
        async with self._sf() as s:
            return list(
                (
                    await s.execute(
                        select(MemoryRow)
                        .where(MemoryRow.employee == employee)
                        .where(MemoryRow.kind == kind)
                        .order_by(MemoryRow.seq)
                    )
                )
                .scalars()
                .all()
            )

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
