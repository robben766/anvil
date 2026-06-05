"""Unit tests for generate.py — fully mocked (no real API calls)."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from anvil_kb.generate import KbAnswer, answer, answer_stream, build_messages, parse_citations
from anvil_kb.store.base import Chunk, ScoredChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str, seq: int = 0, header_path: str = "") -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        seq=seq,
        content=content,
        header_path=header_path,
        start_offset=seq * 100,
        end_offset=seq * 100 + len(content),
    )


def _scored(content: str, seq: int = 0, header_path: str = "") -> ScoredChunk:
    return ScoredChunk(chunk=_make_chunk(content, seq, header_path), score=0.9 - seq * 0.1)


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


def test_build_messages_system_has_numbered_context():
    retrieved = [
        _scored("等待期为90天。", seq=0, header_path="条款 > 等待期"),
        _scored("宽限期60天。", seq=1, header_path="条款 > 宽限期"),
    ]
    msgs = build_messages("等待期?", retrieved)

    assert len(msgs) == 2
    system_content = msgs[0]["content"]
    # Each chunk should appear with its 1-based number
    assert "[1]" in system_content
    assert "[2]" in system_content
    # header_path should appear in parentheses
    assert "(条款 > 等待期)" in system_content
    assert "(条款 > 宽限期)" in system_content
    # chunk content should appear
    assert "等待期为90天。" in system_content
    assert "宽限期60天。" in system_content
    # user message
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "等待期?"


def test_build_messages_empty_header_path_omits_parens():
    retrieved = [_scored("无标题内容。", seq=0, header_path="")]
    msgs = build_messages("q", retrieved)
    system_content = msgs[0]["content"]
    # No parenthetical section when header_path is empty
    assert "[1] 无标题内容。" in system_content
    assert "(" not in system_content.split("[1]")[1].split("\n")[0]


def test_build_messages_roles():
    msgs = build_messages("hello", [_scored("content")])
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


# ---------------------------------------------------------------------------
# parse_citations
# ---------------------------------------------------------------------------


def test_parse_citations_basic():
    retrieved = [_scored("等待期为90天。", seq=0), _scored("宽限期60天。", seq=1)]
    citations = parse_citations("等待期是90天 [1]。", retrieved)
    assert len(citations) == 1
    assert citations[0].n == 1
    assert citations[0].quote == "等待期为90天。"


def test_parse_citations_dedup_preserves_first_order():
    """[1] appearing twice should produce only one Citation."""
    retrieved = [_scored("等待期为90天。", seq=0), _scored("宽限期60天。", seq=1)]
    citations = parse_citations("[1] 文字 [2] 更多文字 [1] 重复", retrieved)
    ns = [c.n for c in citations]
    assert ns == [1, 2], "should be in first-appearance order, deduplicated"


def test_parse_citations_out_of_range_ignored():
    retrieved = [_scored("A", seq=0), _scored("B", seq=1)]
    # [9] is out of range (only 2 chunks), [0] is below 1
    citations = parse_citations("见 [9] 和 [0]", retrieved)
    assert citations == []


def test_parse_citations_zero_ignored():
    retrieved = [_scored("A", seq=0)]
    citations = parse_citations("[0] is not valid", retrieved)
    assert citations == []


def test_parse_citations_len_plus_1_ignored():
    retrieved = [_scored("A", seq=0), _scored("B", seq=1)]
    citations = parse_citations(f"[{len(retrieved) + 1}] out of range", retrieved)
    assert citations == []


def test_parse_citations_chunk_fields():
    """Citation fields must map from the ScoredChunk correctly."""
    retrieved = [_scored("等待期为90天。", seq=0, header_path="条款")]
    citations = parse_citations("[1] 等待期", retrieved)
    c = citations[0]
    assert c.n == 1
    assert c.chunk_id == retrieved[0].chunk.id
    assert c.document_id == retrieved[0].chunk.document_id
    assert c.quote == retrieved[0].chunk.content
    assert c.start_offset == retrieved[0].chunk.start_offset
    assert c.end_offset == retrieved[0].chunk.end_offset


# ---------------------------------------------------------------------------
# answer (non-streaming)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_maps_citations_back_to_chunks():
    retrieved = [_scored("等待期为90天。", seq=0), _scored("宽限期60天。", seq=1)]

    async def fake_chat(model, messages, **kw):
        return SimpleNamespace(content="等待期是90天 [1],与宽限期无关 [1]。")

    result = await answer("等待期?", retrieved, chat=fake_chat)
    assert [c.n for c in result.citations] == [1]  # deduplication
    assert result.citations[0].quote == "等待期为90天。"


@pytest.mark.asyncio
async def test_citation_out_of_range_ignored():
    retrieved = [_scored("A", seq=0), _scored("B", seq=1)]

    async def fake_chat(model, messages, **kw):
        return SimpleNamespace(content="答案 [9] 和 [0]。")

    result = await answer("q?", retrieved, chat=fake_chat)
    assert result.citations == []


@pytest.mark.asyncio
async def test_answer_returns_kbanswer():
    retrieved = [_scored("内容。", seq=0)]

    async def fake_chat(model, messages, **kw):
        return SimpleNamespace(content="回答 [1]。")

    result = await answer("q?", retrieved, chat=fake_chat)
    assert isinstance(result, KbAnswer)
    assert "回答" in result.text


@pytest.mark.asyncio
async def test_answer_passes_model_and_messages():
    """answer() should call chat with model='chat-default' and correct messages."""
    retrieved = [_scored("内容。", seq=0)]
    calls = []

    async def fake_chat(model, messages, **kw):
        calls.append({"model": model, "messages": messages})
        return SimpleNamespace(content="答案")

    await answer("问题", retrieved, chat=fake_chat)

    assert len(calls) == 1
    assert calls[0]["model"] == "chat-default"
    msgs = calls[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "问题"


# ---------------------------------------------------------------------------
# answer_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_stream_event_order():
    """Events: ("sources", list) → ("delta", str)×N → ("done", KbAnswer)."""
    retrieved = [_scored("等待期为90天。", seq=0), _scored("宽限期60天。", seq=1)]

    async def fake_chat_stream(model, messages, **kw):
        # Mirror real ChatChunk: dataclass with .delta field
        from anvil_gateway.types import ChatChunk

        async def _gen():
            yield ChatChunk(delta="答案是90天 ", finish_reason=None)
            yield ChatChunk(delta="[1]。", finish_reason="stop")

        return _gen()

    events = []
    async for event_type, payload in answer_stream("等待期?", retrieved, chat=fake_chat_stream):
        events.append((event_type, payload))

    types = [e[0] for e in events]
    # First event must be "sources"
    assert types[0] == "sources"
    # Last event must be "done"
    assert types[-1] == "done"
    # All middle events must be "delta"
    for t in types[1:-1]:
        assert t == "delta"
    # sources payload is the retrieved list
    assert events[0][1] == retrieved
    # done payload is a KbAnswer
    done_answer = events[-1][1]
    assert isinstance(done_answer, KbAnswer)
    # Citations derived from full assembled text
    assert any(c.n == 1 for c in done_answer.citations)


@pytest.mark.asyncio
async def test_answer_stream_done_citations_match_full_text():
    """done KbAnswer.citations should match parse_citations on the full assembled text."""
    retrieved = [_scored("等待期为90天。", seq=0), _scored("宽限期60天。", seq=1)]

    async def fake_chat_stream(model, messages, **kw):
        from anvil_gateway.types import ChatChunk

        async def _gen():
            yield ChatChunk(delta="[1] 和 ", finish_reason=None)
            yield ChatChunk(delta="[2]。", finish_reason="stop")

        return _gen()

    events = []
    async for event_type, payload in answer_stream("q?", retrieved, chat=fake_chat_stream):
        events.append((event_type, payload))

    done_answer = events[-1][1]
    expected_citations = parse_citations(done_answer.text, retrieved)
    assert [c.n for c in done_answer.citations] == [c.n for c in expected_citations]


@pytest.mark.asyncio
async def test_answer_stream_delta_payloads_are_strings():
    retrieved = [_scored("内容。", seq=0)]

    async def fake_chat_stream(model, messages, **kw):
        from anvil_gateway.types import ChatChunk

        async def _gen():
            yield ChatChunk(delta="hello ", finish_reason=None)
            yield ChatChunk(delta="world", finish_reason="stop")

        return _gen()

    deltas = []
    async for event_type, payload in answer_stream("q?", retrieved, chat=fake_chat_stream):
        if event_type == "delta":
            deltas.append(payload)

    assert all(isinstance(d, str) for d in deltas)
    assert "".join(deltas) == "hello world"


# ---------------------------------------------------------------------------
# C-1 regression: default import path is reachable via monkeypatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_uses_real_default_import_path(monkeypatch):
    """C-1 regression: anvil_gateway.chat is patchable — confirms the default import path works."""
    from anvil_gateway.types import ChatChunk  # noqa: F401 – ensure types importable

    called_with: list[str] = []

    async def fake_chat(model, messages, **kw):
        called_with.append(model)
        return SimpleNamespace(content="fake [1]")

    import anvil_gateway

    monkeypatch.setattr(anvil_gateway, "chat", fake_chat)

    retrieved = [_scored("内容。", seq=0)]
    result = await answer("q", retrieved, chat=None)

    # fake was reached, confirming the default import path is live
    assert called_with == ["chat-default"]
    assert isinstance(result, KbAnswer)


# ---------------------------------------------------------------------------
# I-1: streaming usage tail chunk must not emit extra delta events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_stream_usage_tail_chunk_no_extra_delta():
    """I-1: a ChatChunk(delta='', finish_reason='stop', usage=...) must not emit a delta event."""
    from anvil_gateway.types import ChatChunk

    retrieved = [_scored("内容。", seq=0)]

    async def fake_chat_stream(model, messages, **kw):
        async def _gen():
            yield ChatChunk(delta="token1 ", finish_reason=None)
            yield ChatChunk(delta="token2", finish_reason=None)
            # real usage tail block as emitted by _stream_one
            yield ChatChunk(delta="", finish_reason="stop", usage=None)

        return _gen()

    events = []
    async for event_type, payload in answer_stream("q?", retrieved, chat=fake_chat_stream):
        events.append((event_type, payload))

    delta_payloads = [payload for event_type, payload in events if event_type == "delta"]
    # usage tail block (delta="") must NOT produce a delta event
    assert "" not in delta_payloads
    # only the real content tokens
    assert delta_payloads == ["token1 ", "token2"]


# ---------------------------------------------------------------------------
# I-2: response.content == None → KbAnswer(text="", citations=[])
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_content_none_returns_empty_kbanswer():
    """I-2: SimpleNamespace(content=None) → KbAnswer(text='', citations=[])."""
    retrieved = [_scored("内容。", seq=0)]

    async def fake_chat(model, messages, **kw):
        return SimpleNamespace(content=None)

    result = await answer("q", retrieved, chat=fake_chat)
    assert isinstance(result, KbAnswer)
    assert result.text == ""
    assert result.citations == []


# ---------------------------------------------------------------------------
# M-2: streaming path uses model="chat-default"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_stream_uses_chat_default_model():
    """M-2: answer_stream must call chat with model='chat-default'."""
    from anvil_gateway.types import ChatChunk

    retrieved = [_scored("内容。", seq=0)]
    captured_model: list[str] = []

    async def fake_chat_stream(model, messages, **kw):
        captured_model.append(model)

        async def _gen():
            yield ChatChunk(delta="ok", finish_reason="stop")

        return _gen()

    async for _ in answer_stream("q?", retrieved, chat=fake_chat_stream):
        pass

    assert captured_model == ["chat-default"]


# ---------------------------------------------------------------------------
# M-3: empty retrieved list smoke test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_empty_retrieved_smoke():
    """M-3: retrieved=[] → answer returns KbAnswer normally with empty citations."""

    async def fake_chat(model, messages, **kw):
        return SimpleNamespace(content="无相关资料。")

    result = await answer("q", [], chat=fake_chat)
    assert isinstance(result, KbAnswer)
    assert result.citations == []


@pytest.mark.asyncio
async def test_answer_stream_empty_retrieved_smoke():
    """M-3: retrieved=[] → answer_stream completes normally, done.citations == []."""
    from anvil_gateway.types import ChatChunk

    async def fake_chat_stream(model, messages, **kw):
        async def _gen():
            yield ChatChunk(delta="无相关资料。", finish_reason="stop")

        return _gen()

    events = []
    async for event_type, payload in answer_stream("q?", [], chat=fake_chat_stream):
        events.append((event_type, payload))

    done_answer = events[-1][1]
    assert isinstance(done_answer, KbAnswer)
    assert done_answer.citations == []
