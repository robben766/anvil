import pytest
from anvil_ai_employee.memory.store import MemoryStore
from anvil_ai_employee.memory.vectorstore import MemoryVectorStore

pytestmark = pytest.mark.asyncio


def _vec(seed: float) -> list[float]:
    return [seed] * 512


async def test_insert_update_delete_list(session_factory):
    store = MemoryStore(session_factory)
    mid = await store.insert(employee="u1", kind="fact", content="住在北京", embedding=_vec(0.1))
    facts = await store.list_facts(employee="u1")
    assert len(facts) == 1 and facts[0].content == "住在北京"
    await store.update(mid, content="住在上海", embedding=_vec(0.2))
    facts = await store.list_facts(employee="u1")
    assert facts[0].content == "住在上海"
    await store.delete(mid)
    assert await store.list_facts(employee="u1") == []


async def test_last_still_works_for_report_marker(session_factory):
    store = MemoryStore(session_factory)
    await store.insert(employee="u1", kind="report_marker", content="m1")
    await store.insert(employee="u1", kind="report_marker", content="m2")
    assert await store.last(employee="u1", kind="report_marker") == "m2"


async def test_vector_knn_filters_employee_and_kind(session_factory):
    store = MemoryStore(session_factory)
    await store.insert(employee="u1", kind="fact", content="北京", embedding=_vec(0.9))
    await store.insert(employee="u1", kind="fact", content="猫", embedding=_vec(0.1))
    await store.insert(employee="u2", kind="fact", content="别人", embedding=_vec(0.9))
    vs = MemoryVectorStore(session_factory)
    hits = await vs.knn(employee="u1", kinds=["fact"], query_vec=_vec(0.9), k=5)
    contents = [row.content for row, score in hits]
    assert "北京" in contents and "别人" not in contents  # employee isolation
    assert hits[0][0].content == "北京"  # nearest first
