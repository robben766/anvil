"""Vector KNN over ae_memories. NOT reusing kb's PgVectorStore — that one binds ChunkRow.
Filters by employee (multi-employee isolation) and kind."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import MemoryRow


class MemoryVectorStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._sf = session_factory

    async def knn(
        self, *, employee: str, kinds: list[str], query_vec: list[float], k: int
    ) -> list[tuple[MemoryRow, float]]:
        async with self._sf() as s:
            rows = (
                await s.execute(
                    select(MemoryRow, MemoryRow.embedding.cosine_distance(query_vec).label("d"))
                    .where(MemoryRow.employee == employee)
                    .where(MemoryRow.kind.in_(kinds))
                    .where(MemoryRow.embedding.is_not(None))
                    .order_by("d")
                    .limit(k)
                )
            ).all()
        return [(row[0], 1.0 - float(row[1])) for row in rows]
