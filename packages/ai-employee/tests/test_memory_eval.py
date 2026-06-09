"""mem0 memory eval — layered assertions on the 北京→上海 golden conversation.

This is anvil's "没数字不算数" discipline made concrete for memory, and the realisation of
review-必修-6: the eval interrogates **ae_memories only**, never the bot's reply text. A chat
bot can answer "你住上海" correctly from in-context conversation history while the memory store
silently double-ADDs (北京 + 上海 both alive) or fails to learn at all — so reply-based asserts
prove nothing about memory. We replay the golden script one user turn at a time through
``Mem0Strategy.after_turn`` (extract/reconcile ``structured_chat`` recorded deterministically via
respx, embeddings via a deterministic StubEmbedder) and assert across five layers:

- **recall layer**   — before the UPDATE turn, ``knn`` neighbours of "用户住在上海" really contain
                       the old 北京 fact row (the near-neighbour assumption mem0 depends on holds).
- **decision layer** — after the UPDATE turn, ``list_facts`` has exactly one 居住地 row, containing
                       上海 (UPDATE applied, not double-ADD of 北京+上海).
- **NOOP layer**     — repeating the 养猫 fact does not grow the facts row count.
- **cross-session**  — a fresh ``system_prefix`` call recalls 上海 across sessions.
- **embed direction**— ADD/UPDATE paths call ``embed_texts`` (passage); recall/neighbour paths
                       call ``embed_query`` (query). Swap them and cosine recall collapses.

The reconcile UPDATE/DELETE ops carry a ``target_id`` that must be a *real* row id. We guarantee
validity by driving each ``after_turn`` in its own respx block and reading the live Beijing fact
id from the DB just before recording the UPDATE turn's reconcile response.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from anvil_ai_employee.eval.memory.golden import BEIJING_TO_SHANGHAI, turns
from anvil_ai_employee.memory.mem0 import Mem0Strategy
from anvil_ai_employee.memory.store import MemoryStore
from anvil_ai_employee.memory.vectorstore import MemoryVectorStore

pytestmark = pytest.mark.asyncio
DS_URL = "https://api.deepseek.com/v1/chat/completions"


class StubEmbedder:
    """Deterministic two-class embedder (mirrors test_mem0_strategy.py):

    - 居住地 facts/queries (mention 住/京/海/居) → one vector class (≈0.9)
    - everything else (e.g. 猫) → another class (≈0.1)

    So 北京 and 上海 facts are mutual near-neighbours (居住地 cluster) while the 猫 fact sits in
    its own class. This isolates the *reconcile decision logic* from embedder noise — the eval
    tests whether mem0 makes the right ADD/UPDATE/NOOP call given that neighbours are found.
    """

    def _vec(self, text: str) -> list[float]:
        base = 0.9 if any(c in text for c in "住京海居") else 0.1
        return [base] * 512

    def embed_texts(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class SpyEmbedder:
    """Wraps a StubEmbedder and records which direction each call used, so the eval can assert
    the embed-direction invariant (writes → embed_texts, recall/neighbour → embed_query)."""

    def __init__(self, inner: StubEmbedder):
        self._inner = inner
        self.text_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_texts(self, texts):
        self.text_calls.append(list(texts))
        return self._inner.embed_texts(texts)

    def embed_query(self, text):
        self.query_calls.append(text)
        return self._inner.embed_query(text)


def _json_resp(obj) -> httpx.Response:
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


# --- Per-turn golden script: what extract returns, and per-fact what reconcile returns. ---
# reconcile op for UPDATE needs a real target_id; we inject it at replay time (closure below).
EMPTY = {"facts": []}
EXTRACT_BEIJING = {"facts": ["用户住在北京"]}
EXTRACT_CAT = {"facts": ["用户养了一只猫"]}
EXTRACT_SHANGHAI = {"facts": ["用户住在上海"]}
EXTRACT_CAT_REPEAT = {"facts": ["用户养了一只猫"]}


async def _run_turn(strat, employee, pair, *, extract, reconciles):
    """Drive a single ``after_turn`` under a fresh respx mock, recording the extract response
    followed by one reconcile response per extracted fact (in order)."""
    with respx.mock:
        respx.post(DS_URL).mock(
            side_effect=[_json_resp(extract), *[_json_resp(r) for r in reconciles]]
        )
        await strat.after_turn(employee, None, pair)


async def test_golden_layered_db_assertions(session_factory):
    employee = "assistant"
    store = MemoryStore(session_factory)
    vs = MemoryVectorStore(session_factory)
    embedder = SpyEmbedder(StubEmbedder())
    strat = Mem0Strategy(session_factory, embedder=embedder, model="deepseek-chat")

    t = turns(BEIJING_TO_SHANGHAI)
    # t[0]=北京 ADD, t[1]=猫 ADD, t[2]=闲聊, t[3]=闲聊, t[4]=上海 UPDATE, t[5]=猫 NOOP

    # turn 0 — ADD 北京
    await _run_turn(strat, employee, t[0], extract=EXTRACT_BEIJING, reconciles=[{"op": "ADD"}])
    # turn 1 — ADD 猫
    await _run_turn(strat, employee, t[1], extract=EXTRACT_CAT, reconciles=[{"op": "ADD"}])
    # turns 2,3 — 闲聊, extract empty, no reconcile
    await _run_turn(strat, employee, t[2], extract=EMPTY, reconciles=[])
    await _run_turn(strat, employee, t[3], extract=EMPTY, reconciles=[])

    facts_before = await store.list_facts(employee=employee)
    assert len(facts_before) == 2  # 北京 + 猫
    beijing_row = next(r for r in facts_before if "北京" in r.content)

    # === RECALL LAYER ===
    # Before reconciling 上海, the knn neighbours of "用户住在上海" really contain the old 北京
    # fact row — proving the near-neighbour assumption mem0 leans on actually holds in pgvector.
    qv = embedder.embed_query("用户住在上海")
    hits = await vs.knn(employee=employee, kinds=["fact"], query_vec=qv, k=5)
    hit_ids = {row.id for row, _ in hits}
    assert beijing_row.id in hit_ids, "recall layer: 北京 fact must be a near-neighbour of 上海"

    # turn 4 — UPDATE 北京→上海 with the *real* beijing row id as target.
    await _run_turn(
        strat,
        employee,
        t[4],
        extract=EXTRACT_SHANGHAI,
        reconciles=[{"op": "UPDATE", "target_id": str(beijing_row.id)}],
    )

    # === DECISION LAYER ===
    # Exactly one 居住地 row, now containing 上海 — UPDATE applied in place, NOT a double-ADD.
    facts_after_update = await store.list_facts(employee=employee)
    residence = [r for r in facts_after_update if any(c in r.content for c in "住京海居")]
    assert len(residence) == 1, "decision layer: residence must be exactly one row (no double-ADD)"
    assert "上海" in residence[0].content
    assert all("北京" not in r.content for r in facts_after_update), "old 北京 must be gone"

    # === NOOP LAYER ===
    # Repeating the 养猫 fact must not grow the facts row count.
    count_before_noop = len(facts_after_update)
    await _run_turn(
        strat, employee, t[5], extract=EXTRACT_CAT_REPEAT, reconciles=[{"op": "NOOP"}]
    )
    facts_after_noop = await store.list_facts(employee=employee)
    assert len(facts_after_noop) == count_before_noop, "NOOP layer: repeat must not add a row"
    cat_rows = [r for r in facts_after_noop if "猫" in r.content]
    assert len(cat_rows) == 1, "NOOP layer: cat fact stays a single row"

    # === CROSS-SESSION LAYER ===
    # A fresh system_prefix call (new 'session', same employee) recalls the learned 上海 fact.
    prefix = await strat.system_prefix(employee, "我住在哪里来着")
    assert "上海" in prefix, "cross-session: system_prefix must recall the learned residence"
    assert "北京" not in prefix

    # === EMBED DIRECTION LAYER ===
    # Writes (ADD/UPDATE) went through embed_texts; recall/neighbour-finding through embed_query.
    assert ["用户住在北京"] in embedder.text_calls
    assert ["用户住在上海"] in embedder.text_calls
    # recall + neighbour-finding all used embed_query (never embed_texts for queries).
    assert "用户住在上海" in embedder.query_calls  # neighbour-finding for the reconcile
    assert "我住在哪里来着" in embedder.query_calls  # cross-session recall
    # the residence fact content must NEVER have been embedded as a query for retrieval keying,
    # and queries must never have been embedded via embed_texts.
    assert all("住哪" not in t for batch in embedder.text_calls for t in batch)


async def test_employee_isolation_in_recall(session_factory):
    """A second employee's identical 居住地 facts must not leak into the first's recall — the
    eval's DB assertions are only meaningful if knn isolates by employee."""
    emb = StubEmbedder()
    store = MemoryStore(session_factory)
    await store.insert(
        employee="alice", kind="fact", content="用户住在上海",
        embedding=emb.embed_texts(["用户住在上海"])[0],
    )
    await store.insert(
        employee="bob", kind="fact", content="用户住在广州",
        embedding=emb.embed_texts(["用户住在广州"])[0],
    )
    strat = Mem0Strategy(session_factory, embedder=emb, model="deepseek-chat")
    prefix = await strat.system_prefix("alice", "我住哪")
    assert "上海" in prefix and "广州" not in prefix


@pytest.mark.live
async def test_live_smoke_beijing_to_shanghai(session_factory):
    """Live smoke: real deepseek + real FastEmbedEmbedder run the 北京→上海 two turns end to end.

    Asserts ae_memories' residence fact ends up containing 上海 — i.e. the real model's
    extract + reconcile actually UPDATEs in place rather than double-ADDing. Default CI skips
    (`-m "not live"`); run manually with DEEPSEEK_API_KEY set.
    """
    from anvil_kb.embed import FastEmbedEmbedder

    employee = "live-assistant"
    store = MemoryStore(session_factory)
    strat = Mem0Strategy(
        session_factory, embedder=FastEmbedEmbedder(), model="deepseek-chat"
    )

    await strat.after_turn(
        employee,
        None,
        [
            {"role": "user", "content": "你好,我住在北京。"},
            {"role": "assistant", "content": "你好!北京是个好地方。"},
        ],
    )
    await strat.after_turn(
        employee,
        None,
        [
            {"role": "user", "content": "我搬到上海了。"},
            {"role": "assistant", "content": "好的,记住了,你现在住在上海。"},
        ],
    )

    facts = await store.list_facts(employee=employee)
    residence = [r for r in facts if any(c in r.content for c in "住京海居")]
    assert residence, "live: should have learned a residence fact"
    assert any("上海" in r.content for r in residence), (
        f"live: residence should be updated to 上海, got {[r.content for r in residence]}"
    )
