"""Triggers turn time/events into queued jobs. M1: CronTrigger. The Trigger protocol
keeps webhook/on-demand triggers pluggable without touching the worker."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Protocol

from croniter import croniter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import JobRow, ScheduleRow


class Trigger(Protocol):
    async def due(self, now: datetime) -> int:
        """Enqueue jobs that are due as of *now*; return how many were enqueued."""
        ...


class CronTrigger:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._sf = session_factory

    async def due(self, now: datetime) -> int:
        enqueued = 0
        async with self._sf() as s:
            async with s.begin():
                rows = (
                    await s.execute(
                        select(ScheduleRow)
                        .where(ScheduleRow.enabled.is_(True))
                        .where(ScheduleRow.next_run_at <= now)
                        .with_for_update(skip_locked=True)
                    )
                ).scalars().all()
                for sched in rows:
                    # Inline JobRow insertion in the same transaction as next_run_at advance —
                    # atomicity: if this txn rolls back, no job is inserted either.
                    s.add(
                        JobRow(
                            id=uuid.uuid4(),
                            skill=sched.skill,
                            payload=sched.payload,
                            schedule_id=sched.id,
                            status="pending",
                        )
                    )
                    sched.next_run_at = croniter(sched.cron_expr, now).get_next(datetime)
                    enqueued += 1
        return enqueued
