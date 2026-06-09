import pytest
from anvil_ai_employee.memory.coreblocks import CoreBlockStore

pytestmark = pytest.mark.asyncio


async def test_default_blocks_and_append(session_factory):
    store = CoreBlockStore(session_factory, char_limit=50)
    blocks = await store.get_all(employee="u1")  # auto-creates persona+human defaults
    assert set(blocks.keys()) == {"persona", "human"}
    ok = await store.append(employee="u1", label="human", text="住在北京")
    assert ok is True
    blocks = await store.get_all(employee="u1")
    assert "住在北京" in blocks["human"]


async def test_replace_substring(session_factory):
    store = CoreBlockStore(session_factory, char_limit=200)
    await store.append(employee="u1", label="human", text="住在北京")
    ok = await store.replace(employee="u1", label="human", old="北京", new="上海")
    assert ok is True
    blocks = await store.get_all(employee="u1")
    assert "上海" in blocks["human"] and "北京" not in blocks["human"]


async def test_replace_missing_old_returns_false(session_factory):
    store = CoreBlockStore(session_factory, char_limit=200)
    await store.get_all(employee="u1")
    assert await store.replace(employee="u1", label="human", old="不存在", new="x") is False


async def test_append_over_limit_returns_false(session_factory):
    store = CoreBlockStore(session_factory, char_limit=10)
    await store.get_all(employee="u1")
    assert await store.append(employee="u1", label="human", text="x" * 50) is False
