from datetime import UTC, datetime

import pytest
from anvil_ai_employee.cli import add_schedule, run_now, show_report

pytestmark = pytest.mark.asyncio


async def test_add_schedule_inserts_with_next_run(session_factory):
    now = datetime(2026, 6, 9, 8, 0, tzinfo=UTC)
    sid = await add_schedule(session_factory, name="周报", cron="0 9 * * 1",
                             skill="kb_digest", now=now)
    from anvil_ai_employee.db import ScheduleRow
    from sqlalchemy import select
    async with session_factory() as s:
        row = (await s.execute(select(ScheduleRow).where(ScheduleRow.id == sid))).scalar_one()
        assert row.name == "周报" and row.next_run_at > now


async def test_run_now_enqueues(session_factory):
    jid = await run_now(session_factory, skill="kb_digest")
    from anvil_ai_employee.db import JobRow
    from sqlalchemy import select
    async with session_factory() as s:
        row = (await s.execute(select(JobRow).where(JobRow.id == jid))).scalar_one()
        assert row.status == "pending" and row.skill == "kb_digest"


async def test_show_report_for_missing_job(session_factory):
    import uuid
    out = await show_report(session_factory, job_id=uuid.uuid4())
    assert "未找到" in out or "not found" in out.lower()


async def test_make_strategy_mem0_and_none(session_factory):
    from anvil_ai_employee.cli import make_strategy
    from anvil_ai_employee.memory.mem0 import Mem0Strategy
    from anvil_ai_employee.memory.strategy import NoMemoryStrategy
    assert isinstance(make_strategy("none", session_factory, "deepseek-chat"), NoMemoryStrategy)
    assert isinstance(make_strategy("mem0", session_factory, "deepseek-chat"), Mem0Strategy)


async def test_make_strategy_letta(session_factory):
    from anvil_ai_employee.cli import make_strategy
    from anvil_ai_employee.memory.letta import LettaStrategy
    assert isinstance(make_strategy("letta", session_factory, "deepseek-chat"), LettaStrategy)
