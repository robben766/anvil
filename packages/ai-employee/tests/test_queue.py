import asyncio

import pytest
from anvil_ai_employee.db import JobRow
from anvil_ai_employee.scheduler.queue import claim_one, complete, enqueue, fail
from sqlalchemy import select

pytestmark = pytest.mark.asyncio


async def test_enqueue_then_claim_marks_running(session_factory):
    job_id = await enqueue(session_factory, skill="kb_digest", payload={"a": 1})
    claimed = await claim_one(session_factory, worker_id="w1")
    assert claimed is not None
    assert claimed.id == job_id
    assert claimed.status == "running"
    assert claimed.locked_by == "w1"
    assert claimed.started_at is not None


async def test_claim_empty_returns_none(session_factory):
    assert await claim_one(session_factory, worker_id="w1") is None


async def test_concurrent_claims_no_double_take(session_factory):
    ids = {await enqueue(session_factory, skill="kb_digest", payload={}) for _ in range(5)}
    # 8 workers race for 5 jobs
    results = await asyncio.gather(
        *[claim_one(session_factory, worker_id=f"w{i}") for i in range(8)]
    )
    claimed = [r for r in results if r is not None]
    assert len(claimed) == 5
    assert {c.id for c in claimed} == ids  # each job taken exactly once


async def test_complete_and_fail(session_factory):
    j1 = await enqueue(session_factory, skill="s", payload={})
    c1 = await claim_one(session_factory, worker_id="w1")
    await complete(session_factory, c1.id, result="done!")
    async with session_factory() as s:
        row = (await s.execute(select(JobRow).where(JobRow.id == j1))).scalar_one()
        assert row.status == "done" and row.result == "done!" and row.finished_at is not None

    j2 = await enqueue(session_factory, skill="s", payload={})
    c2 = await claim_one(session_factory, worker_id="w1")
    await fail(session_factory, c2.id, error="boom")
    async with session_factory() as s:
        row = (await s.execute(select(JobRow).where(JobRow.id == j2))).scalar_one()
        assert row.status == "failed" and row.error == "boom" and row.finished_at is not None
