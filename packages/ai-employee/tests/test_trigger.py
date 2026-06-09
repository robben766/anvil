from datetime import UTC, datetime, timedelta

import pytest
from anvil_ai_employee.db import JobRow, ScheduleRow
from anvil_ai_employee.scheduler.trigger import CronTrigger
from sqlalchemy import select

pytestmark = pytest.mark.asyncio


async def _add_schedule(session_factory, *, name, cron, next_run_at, enabled=True):
    async with session_factory() as s:
        async with s.begin():
            s.add(ScheduleRow(
                name=name, cron_expr=cron, skill="kb_digest",
                payload={}, next_run_at=next_run_at, enabled=enabled,
            ))


async def test_due_enqueues_and_advances(session_factory):
    now = datetime(2026, 6, 9, 9, 0, tzinfo=UTC)
    # daily at 09:00; due exactly now
    await _add_schedule(session_factory, name="daily", cron="0 9 * * *",
                        next_run_at=now)
    trig = CronTrigger(session_factory)
    n = await trig.due(now)
    assert n == 1
    async with session_factory() as s:
        jobs = (await s.execute(select(JobRow))).scalars().all()
        assert len(jobs) == 1 and jobs[0].skill == "kb_digest"
        sched = (await s.execute(select(ScheduleRow))).scalar_one()
        # advanced to next 09:00 (tomorrow)
        assert sched.next_run_at == now + timedelta(days=1)


async def test_not_due_does_nothing(session_factory):
    now = datetime(2026, 6, 9, 8, 0, tzinfo=UTC)
    await _add_schedule(session_factory, name="daily", cron="0 9 * * *",
                        next_run_at=datetime(2026, 6, 9, 9, 0, tzinfo=UTC))
    trig = CronTrigger(session_factory)
    assert await trig.due(now) == 0
    async with session_factory() as s:
        assert (await s.execute(select(JobRow))).scalars().all() == []


async def test_disabled_skipped(session_factory):
    now = datetime(2026, 6, 9, 9, 0, tzinfo=UTC)
    await _add_schedule(session_factory, name="off", cron="0 9 * * *",
                        next_run_at=now, enabled=False)
    trig = CronTrigger(session_factory)
    assert await trig.due(now) == 0
