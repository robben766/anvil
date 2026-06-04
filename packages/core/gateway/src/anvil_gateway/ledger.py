"""PostgreSQL 记账(SQLAlchemy async)。schema 由 Alembic 管理;init_schema 供测试/本地快速建表。"""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from anvil_gateway.db import Base, UsageRecordRow
from anvil_gateway.usage import UsageRecord


class PostgresLedger:
    # NOTE: NullPool 简化生命周期(configure 重置不泄漏连接池);高 QPS 时换连接池并管理 dispose。
    def __init__(self, database_url: str) -> None:
        self._engine: AsyncEngine = create_async_engine(database_url, poolclass=NullPool)

    async def init_schema(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def insert(self, r: UsageRecord) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                UsageRecordRow.__table__.insert().values(
                    provider=r.provider,
                    model=r.model,
                    prompt_tokens=r.prompt_tokens,
                    completion_tokens=r.completion_tokens,
                    cached_tokens=r.cached_tokens,
                    cost_cny=r.cost_cny,
                    latency_ms=r.latency_ms,
                    ttft_ms=r.ttft_ms,
                    request_id=r.request_id,
                    session_id=r.session_id,
                    created_at=r.created_at,
                )
            )

    async def count(self) -> int:
        async with self._engine.connect() as conn:
            result = await conn.execute(select(func.count()).select_from(UsageRecordRow))
            return int(result.scalar_one())

    async def total_cost(self) -> Decimal:
        async with self._engine.connect() as conn:
            result = await conn.execute(select(func.coalesce(func.sum(UsageRecordRow.cost_cny), 0)))
            return Decimal(result.scalar_one())

    async def clear(self) -> None:
        """测试辅助:清空表。"""
        async with self._engine.begin() as conn:
            await conn.execute(delete(UsageRecordRow))

    async def close(self) -> None:
        await self._engine.dispose()
