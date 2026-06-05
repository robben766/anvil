"""Tests for anvil_kb.ingest.enrich — Contextual Retrieval enricher.

TDD: these tests are written before the implementation (RED phase).

chat 真实签名 (anvil_gateway.client.chat):
    async def chat(model, messages, *, stream=False, session_id=None,
                   temperature=None, max_tokens=None, ...)
    -> ChatResponse | AsyncIterator[ChatChunk]

所以 enrich_chunks 调用:
    await chat("chat-default", messages, session_id=session_id, temperature=0.3, max_tokens=120)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest  # noqa: I001

from anvil_kb.ingest.chunker import ChunkDraft

# ── Helpers ───────────────────────────────────────────────────────────────────


def _draft(content: str, seq: int = 0) -> ChunkDraft:
    return ChunkDraft(
        content=content,
        header_path="",
        start_offset=seq * 100,
        end_offset=seq * 100 + len(content),
        seq=seq,
    )


# ── Fake chat builders ────────────────────────────────────────────────────────


def _make_fake_chat(responses: list[str]):
    """Returns a fake async chat function that returns responses in order."""
    calls: list[dict] = []
    idx = 0

    async def fake_chat(model: str, messages: list[dict], **kw):
        nonlocal idx
        calls.append({"model": model, "messages": messages, **kw})
        resp_text = responses[idx] if idx < len(responses) else ""
        idx += 1
        return SimpleNamespace(content=resp_text)

    fake_chat.calls = calls  # type: ignore[attr-defined]
    return fake_chat


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_enrich_returns_one_context_per_draft():
    """enrich_chunks returns a list of str with len == len(drafts)."""
    from anvil_kb.ingest.enrich import enrich_chunks

    doc_text = "完整文档内容在这里。"
    drafts = [_draft("第一段", 0), _draft("第二段", 1), _draft("第三段", 2)]
    fake_chat = _make_fake_chat(["上下文1", "上下文2", "上下文3"])

    contexts = await enrich_chunks(doc_text=doc_text, drafts=drafts, chat=fake_chat)

    assert len(contexts) == len(drafts)
    assert contexts == ["上下文1", "上下文2", "上下文3"]


@pytest.mark.asyncio
async def test_enrich_system_message_contains_full_doc():
    """每次调用的 system 消息中必须包含完整文档(缓存前缀断言)。"""
    from anvil_kb.ingest.enrich import enrich_chunks

    doc_text = "完整文档内容在这里。"
    drafts = [_draft("段落A", 0), _draft("段落B", 1)]
    fake_chat = _make_fake_chat(["ctx_A", "ctx_B"])

    await enrich_chunks(doc_text=doc_text, drafts=drafts, chat=fake_chat)

    assert len(fake_chat.calls) == 2
    for call in fake_chat.calls:
        # system message should contain the full document
        messages = call["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert doc_text in system_msg["content"], (
            "system 消息必须包含完整文档文本"
        )


@pytest.mark.asyncio
async def test_enrich_user_message_contains_chunk_content():
    """每次调用的 user 消息中必须包含对应 chunk 的内容。"""
    from anvil_kb.ingest.enrich import enrich_chunks

    doc_text = "文档全文。"
    drafts = [_draft("等待期90天是保险条款", 0), _draft("宽限期60天", 1)]
    fake_chat = _make_fake_chat(["ctx1", "ctx2"])

    await enrich_chunks(doc_text=doc_text, drafts=drafts, chat=fake_chat)

    assert fake_chat.calls[0]["messages"][-1]["content"].find(drafts[0].content) >= 0
    assert fake_chat.calls[1]["messages"][-1]["content"].find(drafts[1].content) >= 0


@pytest.mark.asyncio
async def test_enrich_session_id_passed_through():
    """session_id 参数透传到每次 chat 调用。"""
    from anvil_kb.ingest.enrich import enrich_chunks

    doc_text = "文档。"
    drafts = [_draft("段落", 0)]
    fake_chat = _make_fake_chat(["ctx"])

    await enrich_chunks(
        doc_text=doc_text, drafts=drafts, chat=fake_chat, session_id="my-session-42"
    )

    assert fake_chat.calls[0]["session_id"] == "my-session-42"


@pytest.mark.asyncio
async def test_enrich_temperature_and_max_tokens_passed():
    """chat 调用时 temperature=0.3, max_tokens=120 必须传递。"""
    from anvil_kb.ingest.enrich import enrich_chunks

    doc_text = "文档。"
    drafts = [_draft("段落", 0)]
    fake_chat = _make_fake_chat(["ctx"])

    await enrich_chunks(doc_text=doc_text, drafts=drafts, chat=fake_chat)

    assert fake_chat.calls[0]["temperature"] == pytest.approx(0.3)
    assert fake_chat.calls[0]["max_tokens"] == 120


@pytest.mark.asyncio
async def test_enrich_model_is_chat_default():
    """chat 调用必须使用 'chat-default' 模型。"""
    from anvil_kb.ingest.enrich import enrich_chunks

    doc_text = "文档。"
    drafts = [_draft("段落", 0)]
    fake_chat = _make_fake_chat(["ctx"])

    await enrich_chunks(doc_text=doc_text, drafts=drafts, chat=fake_chat)

    assert fake_chat.calls[0]["model"] == "chat-default"


@pytest.mark.asyncio
async def test_enrich_strips_whitespace():
    """返回的上下文应去掉前后空白。"""
    from anvil_kb.ingest.enrich import enrich_chunks

    doc_text = "文档。"
    drafts = [_draft("段落", 0)]
    fake_chat = _make_fake_chat(["  有空白的上下文  \n"])

    contexts = await enrich_chunks(doc_text=doc_text, drafts=drafts, chat=fake_chat)

    assert contexts[0] == "有空白的上下文"


@pytest.mark.asyncio
async def test_enrich_fail_open_single_error():
    """某 chunk 的 chat 抛错 → 该位置返回 '' 且其余正常(fail-open)。"""
    from anvil_kb.ingest.enrich import enrich_chunks

    doc_text = "文档。"
    drafts = [_draft("正常段落", 0), _draft("问题段落", 1), _draft("另一段", 2)]

    call_idx = 0

    async def failing_chat(model, messages, **kw):
        nonlocal call_idx
        i = call_idx
        call_idx += 1
        if i == 1:
            raise RuntimeError("模拟 API 失败")
        return SimpleNamespace(content=f"ctx_{i}")

    contexts = await enrich_chunks(doc_text=doc_text, drafts=drafts, chat=failing_chat)

    assert len(contexts) == 3
    assert contexts[0] == "ctx_0"
    assert contexts[1] == ""  # 失败位置为空
    assert contexts[2] == "ctx_2"


@pytest.mark.asyncio
async def test_enrich_fail_open_does_not_propagate():
    """任何单点异常不许让 enrich_chunks 本身抛出。"""
    from anvil_kb.ingest.enrich import enrich_chunks

    doc_text = "文档。"
    drafts = [_draft("段落", 0)]

    async def always_failing_chat(model, messages, **kw):
        raise Exception("complete failure")

    # Must not raise
    contexts = await enrich_chunks(doc_text=doc_text, drafts=drafts, chat=always_failing_chat)
    assert contexts == [""]


@pytest.mark.asyncio
async def test_enrich_empty_drafts():
    """空 drafts → 返回空列表,chat 不被调用。"""
    from anvil_kb.ingest.enrich import enrich_chunks

    fake_chat = _make_fake_chat([])
    contexts = await enrich_chunks(doc_text="文档。", drafts=[], chat=fake_chat)

    assert contexts == []
    assert len(fake_chat.calls) == 0


@pytest.mark.asyncio
async def test_enrich_call_count_equals_draft_count():
    """chat 被调用次数 == len(drafts)。"""
    from anvil_kb.ingest.enrich import enrich_chunks

    doc_text = "文档。"
    n = 5
    drafts = [_draft(f"段落{i}", i) for i in range(n)]
    fake_chat = _make_fake_chat([f"ctx_{i}" for i in range(n)])

    await enrich_chunks(doc_text=doc_text, drafts=drafts, chat=fake_chat)

    assert len(fake_chat.calls) == n


@pytest.mark.asyncio
async def test_enrich_system_messages_all_same():
    """同文档所有 chunk 的 system 消息内容相同(缓存共享条件)。"""
    from anvil_kb.ingest.enrich import enrich_chunks

    doc_text = "完整文档内容：第一章节是关于等待期的规定。"
    drafts = [_draft(f"段落{i}", i) for i in range(3)]
    fake_chat = _make_fake_chat(["a", "b", "c"])

    await enrich_chunks(doc_text=doc_text, drafts=drafts, chat=fake_chat)

    system_messages = [
        next(m["content"] for m in call["messages"] if m["role"] == "system")
        for call in fake_chat.calls
    ]
    # All system messages must be identical (shared cache prefix)
    assert len(set(system_messages)) == 1, "所有 chunk 的 system 消息必须相同"


@pytest.mark.asyncio
async def test_enrich_default_session_id():
    """默认 session_id='anvil-kb-enrich'。"""
    from anvil_kb.ingest.enrich import enrich_chunks

    fake_chat = _make_fake_chat(["ctx"])

    await enrich_chunks(doc_text="doc", drafts=[_draft("p", 0)], chat=fake_chat)

    assert fake_chat.calls[0]["session_id"] == "anvil-kb-enrich"


@pytest.mark.asyncio
async def test_enrich_uses_default_gateway_import_path(monkeypatch):
    """chat=None → 从 anvil_gateway 导入 chat(默认路径可 monkeypatch)。"""
    from anvil_kb.ingest import enrich as enrich_mod

    called_with: list[str] = []

    async def fake_chat(model, messages, **kw):
        called_with.append(model)
        return SimpleNamespace(content="ctx")

    import anvil_gateway

    monkeypatch.setattr(anvil_gateway, "chat", fake_chat)
    # Also patch the module-level reference that enrich.py will import
    monkeypatch.setattr(enrich_mod, "_get_default_chat", lambda: fake_chat)

    contexts = await enrich_mod.enrich_chunks(
        doc_text="doc", drafts=[_draft("p", 0)], chat=None
    )

    assert contexts == ["ctx"]
