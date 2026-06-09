import pytest
from anvil_ai_employee.memory.coreblocks import CoreBlockStore
from anvil_ai_employee.memory.letta_tools import LettaToolContext, build_letta_tools
from anvil_ai_employee.memory.store import MemoryStore
from anvil_ai_employee.sessions import SessionStore
from anvil_code_agent.tools.base import ToolContext

pytestmark = pytest.mark.asyncio
TC = ToolContext(workdir="/tmp")


class StubEmbedder:
    def embed_texts(self, texts): return [[0.5] * 512 for _ in texts]
    def embed_query(self, text): return [0.5] * 512


def _tools(ctx):
    return {t.name: t for t in build_letta_tools(ctx)}


async def test_core_memory_replace_writes_db(session_factory):
    cb = CoreBlockStore(session_factory)
    await cb.append(employee="u1", label="human", text="住在北京")
    ctx = LettaToolContext(session_factory=session_factory, embedder=StubEmbedder(),
                           employee="u1", session_id=None)
    res = _tools(ctx)["core_memory_replace"](
        {"label": "human", "old": "北京", "new": "上海"}, TC)
    assert res.ok
    blocks = await cb.get_all(employee="u1")
    assert "上海" in blocks["human"]


async def test_archival_insert_then_search(session_factory):
    ctx = LettaToolContext(session_factory=session_factory, embedder=StubEmbedder(),
                           employee="u1", session_id=None)
    tools = _tools(ctx)
    assert tools["archival_insert"]({"text": "用户的生日是 5 月"}, TC).ok
    res = tools["archival_search"]({"query": "生日"}, TC)
    assert res.ok and "5 月" in res.content
    # archival lands in ae_memories kind=archival
    facts = await MemoryStore(session_factory).list_facts(employee="u1", kind="archival")
    assert len(facts) == 1


async def test_conversation_search_over_session(session_factory):
    ss = SessionStore(session_factory)
    sid = await ss.create(employee="u1")
    await ss.save(sid, [{"role": "user", "content": "我最喜欢的颜色是蓝色"},
                        {"role": "assistant", "content": "好的"}], status="active")
    ctx = LettaToolContext(session_factory=session_factory, embedder=StubEmbedder(),
                           employee="u1", session_id=sid)
    res = _tools(ctx)["conversation_search"]({"query": "颜色"}, TC)
    assert res.ok and "蓝色" in res.content


async def test_core_append_over_limit_is_feedback_not_crash(session_factory):
    cb = CoreBlockStore(session_factory, char_limit=5)
    await cb.get_all(employee="u1")
    ctx = LettaToolContext(session_factory=session_factory, embedder=StubEmbedder(),
                           employee="u1", session_id=None, char_limit=5)
    res = _tools(ctx)["core_memory_append"]({"label": "human", "text": "x" * 50}, TC)
    assert res.ok is False and "limit" in res.content.lower() or not res.ok
