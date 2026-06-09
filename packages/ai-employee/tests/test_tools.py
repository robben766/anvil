from datetime import UTC, datetime, timedelta

import pytest
from anvil_ai_employee.db import JobRow
from anvil_ai_employee.memory.store import MemoryStore
from anvil_ai_employee.scheduler.queue import claim_one, enqueue
from anvil_ai_employee.tools import EmployeeContext, build_employee_tools
from anvil_code_agent.tools.base import ToolContext
from sqlalchemy import select

from anvil_kb.db import DocumentRow

pytestmark = pytest.mark.asyncio


async def _add_doc(session_factory, *, title, source, content, created_at):
    async with session_factory() as s:
        async with s.begin():
            s.add(
                DocumentRow(
                    title=title, source_name=source, content=content, created_at=created_at
                )
            )


async def test_kb_recent_filters_by_since(session_factory):
    base = datetime(2026, 6, 1, tzinfo=UTC)
    await _add_doc(session_factory, title="old", source="o.md", content="old body", created_at=base)
    await _add_doc(
        session_factory,
        title="new",
        source="n.md",
        content="new body",
        created_at=base + timedelta(days=5),
    )
    ctx = EmployeeContext(session_factory=session_factory, employee="kb_reporter", job_id=None)
    tc = ToolContext(workdir="/tmp")
    tools = {t.name: t for t in build_employee_tools(ctx)}
    res = tools["kb_recent"]({"since_iso": (base + timedelta(days=1)).isoformat()}, tc)
    assert res.ok
    assert "new" in res.content and "old" not in res.content


async def test_recall_marker_empty(session_factory):
    ctx = EmployeeContext(session_factory=session_factory, employee="kb_reporter", job_id=None)
    tools = {t.name: t for t in build_employee_tools(ctx)}
    res = tools["recall_marker"]({}, ToolContext(workdir="/tmp"))
    assert res.ok and ("从未" in res.content or "never" in res.content.lower())


async def test_submit_report_writes_result_and_marker(session_factory):
    job_id = await enqueue(session_factory, skill="kb_digest", payload={})
    claimed = await claim_one(session_factory, worker_id="w1")
    ctx = EmployeeContext(
        session_factory=session_factory, employee="kb_reporter", job_id=claimed.id
    )
    tools = {t.name: t for t in build_employee_tools(ctx)}
    res = tools["submit_report"](
        {"markdown": "# 周报\n本期 1 篇", "covered_until_iso": "2026-06-08T00:00:00+00:00"},
        ToolContext(workdir="/tmp"),
    )
    assert res.ok
    # job result set
    async with session_factory() as s:
        row = (await s.execute(select(JobRow).where(JobRow.id == job_id))).scalar_one()
        assert row.status == "done" and "周报" in row.result
    # marker written
    last = await MemoryStore(session_factory).last(employee="kb_reporter", kind="report_marker")
    assert last is not None and "2026-06-08" in last
