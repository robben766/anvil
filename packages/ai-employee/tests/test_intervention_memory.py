import pytest
from anvil_ai_employee.hitl_memory import record_intervention
from anvil_ai_employee.memory.store import MemoryStore

pytestmark = pytest.mark.asyncio


class StubEmbedder:
    def embed_texts(self, texts): return [[0.5] * 512 for _ in texts]
    def embed_query(self, text): return [0.5] * 512


async def test_records_reject_as_memory(session_factory):
    await record_intervention(session_factory, embedder=StubEmbedder(), employee="assistant",
                              tool_name="bash", decision="reject",
                              payload={"reason": "危险"}, tool_args={"cmd": "rm"})
    facts = await MemoryStore(session_factory).list_facts(employee="assistant", kind="hitl")
    assert len(facts) == 1 and "拒绝" in facts[0].content and "bash" in facts[0].content
    assert facts[0].embedding is not None  # recallable
