"""PG-native job queue. Workers claim with FOR UPDATE SKIP LOCKED — no Redis, one PG."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql import func

from anvil_ai_employee.db import JobRow


@dataclass(frozen=True)
class ClaimedJob:
    id: uuid.UUID
    skill: str
    payload: dict[str, Any]
    status: str
    locked_by: str
    started_at: Any
    goal_id: uuid.UUID | None = None
    employee: str | None = None


async def enqueue(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    skill: str,
    payload: dict[str, Any],
    schedule_id: uuid.UUID | None = None,
    goal_id: uuid.UUID | None = None,
    employee: str | None = None,
) -> uuid.UUID:
    job_id = uuid.uuid4()
    async with session_factory() as s:
        async with s.begin():
            await s.execute(
                insert(JobRow).values(
                    id=job_id,
                    schedule_id=schedule_id,
                    skill=skill,
                    payload=payload,
                    status="pending",
                    goal_id=goal_id,
                    employee=employee,
                )
            )
    return job_id


async def claim_one(
    session_factory: async_sessionmaker[AsyncSession], *, worker_id: str
) -> ClaimedJob | None:
    """Atomically grab one pending job. Concurrent workers never take the same row."""
    async with session_factory() as s:
        async with s.begin():
            row = (
                await s.execute(
                    select(JobRow)
                    .where(JobRow.status == "pending")
                    .order_by(JobRow.created_at)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            await s.execute(
                update(JobRow)
                .where(JobRow.id == row.id)
                .values(status="running", locked_by=worker_id, started_at=func.now())
            )
            # re-read for the populated started_at
            fresh = (
                await s.execute(select(JobRow).where(JobRow.id == row.id))
            ).scalar_one()
            return ClaimedJob(
                id=fresh.id,
                skill=fresh.skill,
                payload=fresh.payload,
                status=fresh.status,
                locked_by=fresh.locked_by,
                started_at=fresh.started_at,
                goal_id=fresh.goal_id,
                employee=fresh.employee,
            )


async def complete(
    session_factory: async_sessionmaker[AsyncSession], job_id: uuid.UUID, *, result: str
) -> None:
    async with session_factory() as s:
        async with s.begin():
            await s.execute(
                update(JobRow)
                .where(JobRow.id == job_id)
                .values(status="done", result=result, finished_at=func.now())
            )


async def fail(
    session_factory: async_sessionmaker[AsyncSession], job_id: uuid.UUID, *, error: str
) -> None:
    async with session_factory() as s:
        async with s.begin():
            await s.execute(
                update(JobRow)
                .where(JobRow.id == job_id)
                .values(status="failed", error=error, finished_at=func.now())
            )
