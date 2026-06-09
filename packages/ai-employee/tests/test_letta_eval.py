"""Letta/MemGPT eval — the agent self-calls memory tools, read-your-write is eventual.

Contrast with the mem0 eval (``test_memory_eval.py``), where the *orchestrator* writes memory
in ``after_turn``. Here a respx ``side_effect`` list stands in for the model and drives
``run_one_turn`` (with ``LettaStrategy``) to call memory tools *itself* inside one turn:
``core_memory_replace`` → ``archival_insert`` → plain-text close.

Assertions read the DB **only after** ``run_one_turn`` returns — never the reply text — because a
core-block edit this turn is not visible to the model until next turn's ``system_prefix``
(read-your-write is eventual). We also exercise archival recall directly and the self-paging
flush/no-orphan invariant.
"""

import json

import httpx
import pytest
import respx
from anvil_ai_employee.chat import apply_self_paging, run_one_turn
from anvil_ai_employee.eval.memory.letta_golden import (
    ARCHIVAL_FACT,
    EXPECTED_TOOL_SEQUENCE,
    FINAL_REPLY,
    HUMAN_BLOCK_SEED,
    USER_INPUT,
)
from anvil_ai_employee.memory.coreblocks import CoreBlockStore
from anvil_ai_employee.memory.letta import LettaStrategy
from anvil_ai_employee.memory.letta_tools import LettaToolContext, build_letta_tools
from anvil_ai_employee.memory.store import MemoryStore
from anvil_code_agent.harness.context import estimate_tokens
from anvil_code_agent.tools.base import ToolContext

pytestmark = pytest.mark.asyncio
DS_URL = "https://api.deepseek.com/v1/chat/completions"
TC = ToolContext(workdir="/tmp")


class StubEmbedder:
    """Deterministic embedder: every text/query maps to the same vector, so archival_search
    always returns the inserted rows (we only assert presence, not ranking)."""

    def embed_texts(self, texts):
        return [[0.5] * 512 for _ in texts]

    def embed_query(self, text):
        return [0.5] * 512


def _tool_call_response(tool_id, name, arguments):
    """assistant turn that emits one tool_call (finish_reason='tool_calls')."""
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
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _text_response(text):
    """assistant turn that closes the turn with plain text (finish_reason='stop')."""
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _golden_side_effect():
    """Build the respx side_effect from the golden script: one tool_calls response per
    expected tool, then a final plain-text close."""
    effects = []
    for i, (name, args) in enumerate(EXPECTED_TOOL_SEQUENCE):
        effects.append(_tool_call_response(f"c{i}", name, args))
    effects.append(_text_response(FINAL_REPLY))
    return effects


@respx.mock
async def test_agent_self_calls_tools_and_writes_db(session_factory):
    """The agent itself drives core_memory_replace + archival_insert inside one turn; after the
    turn the DB reflects both edits. We never assert the reply text reflects the change
    (read-your-write is eventual)."""
    employee = "letta-eval"
    cb = CoreBlockStore(session_factory)
    # Seed the human core block so core_memory_replace has an `old` substring to hit.
    await cb.append(employee=employee, label="human", text=HUMAN_BLOCK_SEED)

    strat = LettaStrategy(session_factory, embedder=StubEmbedder(), model="deepseek-chat")
    respx.post(DS_URL).mock(side_effect=_golden_side_effect())

    reply, history = await run_one_turn(
        persona="你是助理",
        user_input=USER_INPUT,
        history=(),
        strategy=strat,
        employee=employee,
        session=None,
        model="deepseek-chat",
        max_steps=8,
    )

    # The turn ended on the plain-text close.
    assert reply == FINAL_REPLY

    # --- DB assertions ONLY (not the reply text): the agent's tool calls hit the DB. ---
    # 1) human core block was edited in place: 北京 → 上海.
    blocks = await cb.get_all(employee=employee)
    assert "上海" in blocks["human"]
    assert "北京" not in blocks["human"]

    # 2) archival memory gained the inserted fact.
    facts = await MemoryStore(session_factory).list_facts(employee=employee, kind="archival")
    assert len(facts) == 1
    assert ARCHIVAL_FACT in facts[0].content


@respx.mock
async def test_archival_recall_after_insert(session_factory):
    """After the agent's archival_insert, a direct archival_search (via LettaToolContext) recalls
    the stored fact — closing the write→read loop on the archival layer."""
    employee = "letta-eval"
    cb = CoreBlockStore(session_factory)
    await cb.append(employee=employee, label="human", text=HUMAN_BLOCK_SEED)

    strat = LettaStrategy(session_factory, embedder=StubEmbedder(), model="deepseek-chat")
    respx.post(DS_URL).mock(side_effect=_golden_side_effect())

    await run_one_turn(
        persona="你是助理",
        user_input=USER_INPUT,
        history=(),
        strategy=strat,
        employee=employee,
        session=None,
        model="deepseek-chat",
        max_steps=8,
    )

    # Direct tool call: archival_search must hit the just-stored fact.
    ctx = LettaToolContext(
        session_factory=session_factory,
        embedder=StubEmbedder(),
        employee=employee,
        session_id=None,
    )
    tools = {t.name: t for t in build_letta_tools(ctx)}
    res = tools["archival_search"]({"query": "过敏"}, TC)
    assert res.ok and ARCHIVAL_FACT in res.content


def test_self_paging_flush_shrinks_tokens_and_no_orphan_tool_messages():
    """An over-budget history flushed via apply_self_paging shrinks token count and leaves no
    orphan tool messages: every role=='tool' message is preceded (somewhere earlier) by an
    assistant carrying tool_calls, and messages[0] stays the system block."""
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "开聊"}]
    # Build an over-budget history of well-formed assistant(tool_calls) → tool pairs.
    for i in range(30):
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {"name": "archival_search", "arguments": "{}"},
                    }
                ],
            }
        )
        msgs.append(
            {"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 400}
        )

    before = estimate_tokens(msgs)

    def _stub_summarizer(_messages):
        return "[摘要] 之前讨论了若干记忆操作。"

    out, _warned = apply_self_paging(
        msgs, budget=before // 2, warn_ratio=0.7, summarizer=_stub_summarizer
    )

    # token count went down (flush happened).
    assert estimate_tokens(out) < before
    # system block is still first — structure not broken.
    assert out[0]["role"] == "system"

    # no orphan tool messages: each tool message has an earlier assistant with tool_calls.
    for idx, m in enumerate(out):
        if m.get("role") == "tool":
            assert any(
                out[j].get("role") == "assistant" and out[j].get("tool_calls")
                for j in range(idx)
            ), f"orphan tool message at index {idx}"


@pytest.mark.live
async def test_live_smoke_agent_self_manages_memory(session_factory):
    """Live smoke: real deepseek drives LettaStrategy. We tell it a fact about the user and run
    one turn; assert the model *itself* called some memory tool (core or archival) and the DB
    changed (human block non-empty OR archival has a row). Default CI skips (`-m "not live"`);
    run manually with DEEPSEEK_API_KEY set. Real model behaviour is non-deterministic, so this
    is a smoke check, not a strict transcript replay."""
    from anvil_kb.embed import FastEmbedEmbedder

    employee = "letta-live"
    cb = CoreBlockStore(session_factory)
    store = MemoryStore(session_factory)
    strat = LettaStrategy(
        session_factory, embedder=FastEmbedEmbedder(), model="deepseek-chat"
    )

    await run_one_turn(
        persona="你是用户的私人助理,遇到关于用户的重要信息要主动用记忆工具记下来。",
        user_input="我叫小明,住在上海。",
        history=(),
        strategy=strat,
        employee=employee,
        session=None,
        model="deepseek-chat",
        max_steps=8,
    )

    blocks = await cb.get_all(employee=employee)
    archival = await store.list_facts(employee=employee, kind="archival")
    human_written = bool(blocks.get("human", "").strip())
    persona_written = bool(blocks.get("persona", "").strip())
    assert human_written or persona_written or archival, (
        "live: agent should have self-called a memory tool (core or archival), "
        f"got blocks={blocks!r} archival={[r.content for r in archival]!r}"
    )
