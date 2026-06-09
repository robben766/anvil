import pytest
from anvil_ai_employee.memory.coreblocks import CoreBlockStore
from anvil_ai_employee.memory.letta import LettaChatCtx, LettaStrategy

pytestmark = pytest.mark.asyncio


class StubEmbedder:
    def embed_texts(self, texts): return [[0.5] * 512 for _ in texts]
    def embed_query(self, text): return [0.5] * 512


def test_build_registry_has_five_tools(session_factory):
    strat = LettaStrategy(session_factory, embedder=StubEmbedder(), model="deepseek-chat")
    reg = strat.build_registry(LettaChatCtx(employee="u1", session_id=None))
    names = {s["function"]["name"] for s in reg.schemas()}
    assert names == {"core_memory_append", "core_memory_replace", "archival_insert",
                     "archival_search", "conversation_search"}


async def test_system_prefix_injects_core_blocks(session_factory):
    cb = CoreBlockStore(session_factory)
    await cb.append(employee="u1", label="human", text="叫小明")
    strat = LettaStrategy(session_factory, embedder=StubEmbedder(), model="deepseek-chat")
    prefix = await strat.system_prefix("u1", "hi")
    assert "core_memory" in prefix and "小明" in prefix


async def test_after_turn_is_noop(session_factory):
    strat = LettaStrategy(session_factory, embedder=StubEmbedder(), model="deepseek-chat")
    await strat.after_turn("u1", None, [])  # must not raise
