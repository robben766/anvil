import pytest
from anvil_ai_employee.memory.store import MemoryStore

pytestmark = pytest.mark.asyncio


async def test_write_then_last(session_factory):
    store = MemoryStore(session_factory)
    await store.write(
        employee="kb_reporter",
        kind="report_marker",
        content='{"covered_until": "2026-06-01T00:00:00+00:00"}',
    )
    await store.write(
        employee="kb_reporter",
        kind="report_marker",
        content='{"covered_until": "2026-06-08T00:00:00+00:00"}',
    )
    last = await store.last(employee="kb_reporter", kind="report_marker")
    assert last is not None and "2026-06-08" in last


async def test_last_none_when_empty(session_factory):
    store = MemoryStore(session_factory)
    assert await store.last(employee="kb_reporter", kind="report_marker") is None


async def test_employee_isolation(session_factory):
    store = MemoryStore(session_factory)
    await store.write(employee="a", kind="report_marker", content="A")
    await store.write(employee="b", kind="report_marker", content="B")
    assert await store.last(employee="a", kind="report_marker") == "A"
    assert await store.last(employee="b", kind="report_marker") == "B"
