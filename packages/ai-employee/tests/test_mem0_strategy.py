import json

import httpx
import pytest
import respx
from anvil_ai_employee.memory.mem0 import Mem0Strategy
from anvil_ai_employee.memory.store import MemoryStore

pytestmark = pytest.mark.asyncio
DS_URL = "https://api.deepseek.com/v1/chat/completions"


class StubEmbedder:
    """Deterministic: vector keyed by whether text mentions 居住/北京/上海 so that
    '住在上海' and '住在北京' are nearest neighbors (the mem0 near-neighbor assumption)."""

    def _vec(self, text):
        base = 0.9 if ("住" in text or "京" in text or "海" in text) else 0.1
        return [base] * 512

    def embed_texts(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _json_resp(obj):
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(obj, ensure_ascii=False),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


@respx.mock
async def test_first_fact_is_added(session_factory):
    # extract → 1 fact ; reconcile → ADD
    respx.post(DS_URL).mock(
        side_effect=[
            _json_resp({"facts": ["用户住在北京"]}),
            _json_resp({"op": "ADD"}),
        ]
    )
    strat = Mem0Strategy(session_factory, embedder=StubEmbedder(), model="deepseek-chat")
    await strat.after_turn(
        "u1",
        None,
        [
            {"role": "user", "content": "我住在北京"},
            {"role": "assistant", "content": "好的"},
        ],
    )
    facts = await MemoryStore(session_factory).list_facts(employee="u1")
    assert len(facts) == 1 and "北京" in facts[0].content
    assert facts[0].embedding is not None  # ADD path used embed_texts


@respx.mock
async def test_contradiction_updates_not_double_adds(session_factory):
    store = MemoryStore(session_factory)
    # seed an existing 北京 fact
    emb = StubEmbedder().embed_texts(["用户住在北京"])[0]
    existing = await store.insert(
        employee="u1", kind="fact", content="用户住在北京", embedding=emb
    )
    # extract 上海 → reconcile UPDATE target=existing
    respx.post(DS_URL).mock(
        side_effect=[
            _json_resp({"facts": ["用户住在上海"]}),
            _json_resp({"op": "UPDATE", "target_id": str(existing)}),
        ]
    )
    strat = Mem0Strategy(session_factory, embedder=StubEmbedder(), model="deepseek-chat")
    await strat.after_turn(
        "u1",
        None,
        [
            {"role": "user", "content": "我搬到上海了"},
            {"role": "assistant", "content": "记住了"},
        ],
    )
    facts = await store.list_facts(employee="u1")
    assert len(facts) == 1 and "上海" in facts[0].content  # updated, not double-added


@respx.mock
async def test_illegal_op_falls_back_to_noop(session_factory):
    respx.post(DS_URL).mock(
        side_effect=[
            _json_resp({"facts": ["用户喜欢猫"]}),
            _json_resp({"op": "FROBNICATE"}),  # illegal
        ]
    )
    strat = Mem0Strategy(session_factory, embedder=StubEmbedder(), model="deepseek-chat")
    await strat.after_turn(
        "u1",
        None,
        [
            {"role": "user", "content": "我喜欢猫"},
            {"role": "assistant", "content": "ok"},
        ],
    )
    # NOOP fallback → nothing written
    assert await MemoryStore(session_factory).list_facts(employee="u1") == []


@respx.mock
async def test_system_prefix_recalls(session_factory):
    store = MemoryStore(session_factory)
    emb = StubEmbedder().embed_texts(["用户住在北京"])[0]
    await store.insert(employee="u1", kind="fact", content="用户住在北京", embedding=emb)
    strat = Mem0Strategy(session_factory, embedder=StubEmbedder(), model="deepseek-chat")
    prefix = await strat.system_prefix("u1", "我住哪来着")
    assert "北京" in prefix
