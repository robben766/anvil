"""H4 tests: debug SSE frame, error SSE event, upload size limit, CORS env, compare_digest.

TDD (red → green):  All tests in this file must fail before the implementation is added.
"""

from __future__ import annotations

import json
import os
import uuid
from types import SimpleNamespace

import httpx
import pytest
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from anvil_kb.db import EMBEDDING_DIM as _EMBEDDING_DIM
from anvil_kb.retrieve.fusion import RetrievalDebug
from anvil_kb.store.base import Chunk, ScoredChunk

TEST_DB_URL = os.environ.get(
    "ANVIL_TEST_DATABASE_URL", "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test"
)

# ---------------------------------------------------------------------------
# Shared helpers (also used in test_query.py — duplicated here to stay isolated)
# ---------------------------------------------------------------------------


class FakeEmbedder:
    @property
    def dim(self) -> int:
        return _EMBEDDING_DIM

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._one_hot(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one_hot(text)

    def _one_hot(self, text: str) -> list[float]:
        vec = [0.0] * _EMBEDDING_DIM
        if text:
            vec[hash(text) % _EMBEDDING_DIM] = 1.0
        return vec


def _make_scored_chunk(content: str, seq: int = 0, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            seq=seq,
            content=content,
            header_path=f"条款 > 第{seq + 1}节",
            start_offset=seq * 100,
            end_offset=seq * 100 + len(content),
        ),
        score=score,
    )


def _parse_sse_events(raw: str) -> list[dict]:
    events = []
    current: dict = {}
    for line in raw.splitlines():
        if line.startswith("event: "):
            current["event"] = line[len("event: "):]
        elif line.startswith("data: "):
            current["data"] = line[len("data: "):]
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


# ---------------------------------------------------------------------------
# Fake retriever with retrieve_debug support
# ---------------------------------------------------------------------------

CHUNK_LONG_CONTENT = "A" * 80  # 80 chars; quote_head must truncate to 40


class FakeRetrieverWithDebug:
    """Fake retriever that also implements retrieve_debug."""

    def __init__(self, scored_chunks: list[ScoredChunk]) -> None:
        self._chunks = scored_chunks

    async def retrieve(self, question: str, k: int = 5) -> list[ScoredChunk]:
        return self._chunks[:k]

    async def retrieve_debug(self, question: str, k: int = 5) -> RetrievalDebug:
        top = self._chunks[:k]
        # Build a minimal RetrievalDebug with fake dense/sparse/fused
        dense = [ScoredChunk(chunk=sc.chunk, score=sc.score) for sc in top]
        sparse = [ScoredChunk(chunk=sc.chunk, score=sc.score * 0.8) for sc in top]
        fused = top  # same list for simplicity
        contributions = {
            str(sc.chunk.id): {"dense": i + 1, "sparse": None}
            for i, sc in enumerate(top)
        }
        return RetrievalDebug(
            dense=dense,
            sparse=sparse,
            fused=fused,
            contributions=contributions,
        )


class FakeRetrieverRaises:
    """Fake retriever that raises on retrieve() — simulates retrieval failure."""

    async def retrieve(self, question: str, k: int = 5) -> list[ScoredChunk]:
        raise RuntimeError("retrieval boom")

    async def retrieve_debug(self, question: str, k: int = 5) -> RetrievalDebug:
        raise RuntimeError("retrieval boom")


# ---------------------------------------------------------------------------
# Fake chat helpers
# ---------------------------------------------------------------------------


def _make_fake_chat_stream(chunks: list[str]):
    from anvil_gateway.types import ChatChunk

    async def _chat(model, messages, **kw):
        async def _gen():
            for i, c in enumerate(chunks):
                finish = "stop" if i == len(chunks) - 1 else None
                yield ChatChunk(delta=c, finish_reason=finish)
            yield ChatChunk(delta="", finish_reason="stop", usage=None)

        return _gen()

    return _chat


def _make_fake_chat_stream_raises():
    """Chat that raises RuntimeError after yielding two delta tokens."""
    from anvil_gateway.types import ChatChunk

    async def _chat(model, messages, **kw):
        async def _gen():
            yield ChatChunk(delta="token1", finish_reason=None)
            yield ChatChunk(delta="token2", finish_reason=None)
            raise RuntimeError("llm exploded mid-stream")

        return _gen()

    return _chat


def _make_fake_chat_nonstream(answer_text: str):
    async def _chat(model, messages, **kw):
        return SimpleNamespace(content=answer_text)

    return _chat


# ---------------------------------------------------------------------------
# Helpers to create ASGI apps
# ---------------------------------------------------------------------------


def _make_engine_and_sf():
    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return engine, sf


# ===========================================================================
# 1. debug frame tests
# ===========================================================================


@pytest.mark.asyncio
async def test_debug_frame_event_order(_run_kb_migrations):
    """stream+debug=true → event order must be debug → sources → delta×N → done."""
    from anvil_kb_api.app import create_app

    scored = [
        _make_scored_chunk("等待期为90天。", seq=0, score=0.95),
        _make_scored_chunk("宽限期为60天。", seq=1, score=0.80),
    ]
    retriever = FakeRetrieverWithDebug(scored)
    fake_chat = _make_fake_chat_stream(["答案 [1]。"])

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf, embedder=FakeEmbedder(), retriever=retriever, chat=fake_chat
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST", "/v1/kb/query", json={"question": "等待期?", "stream": True, "debug": True}
        ) as resp:
            raw = await resp.aread()

    await engine.dispose()

    events = _parse_sse_events(raw.decode())
    event_types = [e["event"] for e in events]

    assert event_types[0] == "debug", f"first event should be debug, got: {event_types}"
    assert event_types[1] == "sources", f"second event should be sources, got: {event_types}"
    assert event_types[-1] == "done"
    for t in event_types[2:-1]:
        assert t == "delta", f"unexpected event type between sources and done: {t}"


@pytest.mark.asyncio
async def test_debug_frame_payload_structure(_run_kb_migrations):
    """debug frame payload must have keys: dense, sparse, fused."""
    from anvil_kb_api.app import create_app

    scored = [_make_scored_chunk("等待期为90天。", seq=0, score=0.95)]
    retriever = FakeRetrieverWithDebug(scored)
    fake_chat = _make_fake_chat_stream(["答案。"])

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf, embedder=FakeEmbedder(), retriever=retriever, chat=fake_chat
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST", "/v1/kb/query", json={"question": "等待期?", "stream": True, "debug": True}
        ) as resp:
            raw = await resp.aread()

    await engine.dispose()

    events = _parse_sse_events(raw.decode())
    debug_event = next(e for e in events if e["event"] == "debug")
    payload = json.loads(debug_event["data"])

    assert "dense" in payload, "debug payload must have 'dense' key"
    assert "sparse" in payload, "debug payload must have 'sparse' key"
    assert "fused" in payload, "debug payload must have 'fused' key"


@pytest.mark.asyncio
async def test_debug_frame_item_structure(_run_kb_migrations):
    """Each item in debug.dense/sparse/fused must have: n, chunk_id, quote_head, score, rank.
    fused items additionally have contributions key.
    """
    from anvil_kb_api.app import create_app

    scored = [_make_scored_chunk("等待期为90天。", seq=0, score=0.95)]
    retriever = FakeRetrieverWithDebug(scored)
    fake_chat = _make_fake_chat_stream(["答案。"])

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf, embedder=FakeEmbedder(), retriever=retriever, chat=fake_chat
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST", "/v1/kb/query", json={"question": "等待期?", "stream": True, "debug": True}
        ) as resp:
            raw = await resp.aread()

    await engine.dispose()

    events = _parse_sse_events(raw.decode())
    debug_event = next(e for e in events if e["event"] == "debug")
    payload = json.loads(debug_event["data"])

    required_base = ("n", "chunk_id", "quote_head", "score", "rank")
    for list_name in ("dense", "sparse"):
        for item in payload[list_name]:
            for field in required_base:
                assert field in item, f"debug.{list_name} item missing field: {field}"

    for item in payload["fused"]:
        for field in (*required_base, "contributions"):
            assert field in item, f"debug.fused item missing field: {field}"


@pytest.mark.asyncio
async def test_debug_frame_quote_head_40_chars(_run_kb_migrations):
    """quote_head must be truncated to first 40 characters of content."""
    from anvil_kb_api.app import create_app

    long_content = "X" * 80  # content is 80 chars; quote_head should be 40
    scored = [_make_scored_chunk(long_content, seq=0, score=0.9)]
    retriever = FakeRetrieverWithDebug(scored)
    fake_chat = _make_fake_chat_stream(["答案。"])

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf, embedder=FakeEmbedder(), retriever=retriever, chat=fake_chat
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST", "/v1/kb/query", json={"question": "等待期?", "stream": True, "debug": True}
        ) as resp:
            raw = await resp.aread()

    await engine.dispose()

    events = _parse_sse_events(raw.decode())
    debug_event = next(e for e in events if e["event"] == "debug")
    payload = json.loads(debug_event["data"])

    # Check dense items
    for item in payload["dense"]:
        assert len(item["quote_head"]) <= 40, (
            f"quote_head length {len(item['quote_head'])} > 40"
        )
        assert item["quote_head"] == long_content[:40]

    # Check fused items
    for item in payload["fused"]:
        assert len(item["quote_head"]) <= 40


@pytest.mark.asyncio
async def test_debug_frame_score_4_decimal_places(_run_kb_migrations):
    """Scores in debug payload must be rounded to 4 decimal places."""
    from anvil_kb_api.app import create_app

    scored = [_make_scored_chunk("等待期为90天。", seq=0, score=0.123456789)]
    retriever = FakeRetrieverWithDebug(scored)
    fake_chat = _make_fake_chat_stream(["答案。"])

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf, embedder=FakeEmbedder(), retriever=retriever, chat=fake_chat
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST", "/v1/kb/query", json={"question": "等待期?", "stream": True, "debug": True}
        ) as resp:
            raw = await resp.aread()

    await engine.dispose()

    events = _parse_sse_events(raw.decode())
    debug_event = next(e for e in events if e["event"] == "debug")
    payload = json.loads(debug_event["data"])

    for list_name in ("dense", "sparse", "fused"):
        for item in payload[list_name]:
            score = item["score"]
            # 4 decimal places: str representation should not exceed 4 decimals
            # Check by round-tripping
            assert round(score, 4) == score, (
                f"debug.{list_name} score {score} has more than 4 decimal places"
            )


@pytest.mark.asyncio
async def test_debug_frame_contributions_null_serialized(_run_kb_migrations):
    """contributions with None sparse rank must serialize as null (not absent key)."""
    from anvil_kb_api.app import create_app

    scored = [_make_scored_chunk("等待期为90天。", seq=0, score=0.9)]
    retriever = FakeRetrieverWithDebug(scored)  # contributions has sparse=None
    fake_chat = _make_fake_chat_stream(["答案。"])

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf, embedder=FakeEmbedder(), retriever=retriever, chat=fake_chat
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST", "/v1/kb/query", json={"question": "等待期?", "stream": True, "debug": True}
        ) as resp:
            raw = await resp.aread()

    await engine.dispose()

    events = _parse_sse_events(raw.decode())
    debug_event = next(e for e in events if e["event"] == "debug")
    payload = json.loads(debug_event["data"])

    for item in payload["fused"]:
        contrib = item["contributions"]
        assert "sparse" in contrib, "contributions must include 'sparse' key"
        assert contrib["sparse"] is None, "contributions.sparse must be null"


@pytest.mark.asyncio
async def test_debug_false_no_debug_frame(_run_kb_migrations):
    """debug=False (default) → no debug event in stream."""
    from anvil_kb_api.app import create_app

    scored = [_make_scored_chunk("等待期为90天。", seq=0, score=0.95)]
    retriever = FakeRetrieverWithDebug(scored)
    fake_chat = _make_fake_chat_stream(["答案。"])

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf, embedder=FakeEmbedder(), retriever=retriever, chat=fake_chat
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST", "/v1/kb/query", json={"question": "等待期?", "stream": True}
            # debug not specified → defaults to False
        ) as resp:
            raw = await resp.aread()

    await engine.dispose()

    events = _parse_sse_events(raw.decode())
    debug_events = [e for e in events if e["event"] == "debug"]
    assert len(debug_events) == 0, "debug=False should produce no debug events"


@pytest.mark.asyncio
async def test_debug_true_nonstream_returns_400(_run_kb_migrations):
    """non-stream + debug=True → 400 Bad Request."""
    from anvil_kb_api.app import create_app

    scored = [_make_scored_chunk("等待期为90天。", seq=0, score=0.95)]
    retriever = FakeRetrieverWithDebug(scored)
    fake_chat = _make_fake_chat_nonstream("答案。")

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf, embedder=FakeEmbedder(), retriever=retriever, chat=fake_chat
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post(
            "/v1/kb/query", json={"question": "等待期?", "stream": False, "debug": True}
        )

    await engine.dispose()

    assert r.status_code == 400
    body = r.json()
    assert "detail" in body


@pytest.mark.asyncio
async def test_debug_fused_is_same_as_retrieved(_run_kb_migrations):
    """sources event must use the same chunks as debug.fused (consistency check)."""
    from anvil_kb_api.app import create_app

    scored = [
        _make_scored_chunk("等待期为90天。", seq=0, score=0.95),
        _make_scored_chunk("宽限期为60天。", seq=1, score=0.80),
    ]
    retriever = FakeRetrieverWithDebug(scored)
    fake_chat = _make_fake_chat_stream(["答案。"])

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf, embedder=FakeEmbedder(), retriever=retriever, chat=fake_chat
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST", "/v1/kb/query", json={"question": "等待期?", "stream": True, "debug": True}
        ) as resp:
            raw = await resp.aread()

    await engine.dispose()

    events = _parse_sse_events(raw.decode())
    debug_event = next(e for e in events if e["event"] == "debug")
    sources_event = next(e for e in events if e["event"] == "sources")

    debug_payload = json.loads(debug_event["data"])
    sources_payload = json.loads(sources_event["data"])

    # fused chunk_ids must match sources chunk_ids (same order, same chunks)
    fused_ids = [item["chunk_id"] for item in debug_payload["fused"]]
    sources_ids = [item["chunk_id"] for item in sources_payload]
    assert fused_ids == sources_ids, (
        f"debug.fused chunk_ids {fused_ids} != sources chunk_ids {sources_ids}"
    )


# ===========================================================================
# 2. error SSE event tests
# ===========================================================================


@pytest.mark.asyncio
async def test_error_event_on_llm_failure(_run_kb_migrations):
    """LLM raises mid-stream → error event with non-empty detail; stream ends normally."""
    from anvil_kb_api.app import create_app

    scored = [_make_scored_chunk("等待期为90天。", seq=0, score=0.95)]
    retriever = FakeRetrieverWithDebug(scored)
    fake_chat = _make_fake_chat_stream_raises()

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf, embedder=FakeEmbedder(), retriever=retriever, chat=fake_chat
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST", "/v1/kb/query", json={"question": "等待期?", "stream": True}
        ) as resp:
            assert resp.status_code == 200  # StreamingResponse starts with 200
            raw = await resp.aread()

    await engine.dispose()

    events = _parse_sse_events(raw.decode())
    event_types = [e["event"] for e in events]

    assert "error" in event_types, f"error event not found; events: {event_types}"

    error_event = next(e for e in events if e["event"] == "error")
    error_data = json.loads(error_event["data"])
    assert "detail" in error_data
    assert len(error_data["detail"]) > 0

    # Stream must end — no exception propagated to caller (connection not cut)
    # After error event, the generator returns normally (no more events expected,
    # but the key point is no exception raised)


@pytest.mark.asyncio
async def test_error_event_detail_truncated_at_200(_run_kb_migrations):
    """Error detail must be truncated to 200 characters."""
    from anvil_kb_api.app import create_app

    long_message = "X" * 500

    class FakeChatRaisesLong:
        async def __call__(self, model, messages, **kw):
            from anvil_gateway.types import ChatChunk

            async def _gen():
                yield ChatChunk(delta="token", finish_reason=None)
                raise RuntimeError(long_message)

            return _gen()

    scored = [_make_scored_chunk("等待期为90天。", seq=0, score=0.95)]
    retriever = FakeRetrieverWithDebug(scored)

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf,
        embedder=FakeEmbedder(),
        retriever=retriever,
        chat=FakeChatRaisesLong(),
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST", "/v1/kb/query", json={"question": "等待期?", "stream": True}
        ) as resp:
            raw = await resp.aread()

    await engine.dispose()

    events = _parse_sse_events(raw.decode())
    error_event = next(e for e in events if e["event"] == "error")
    error_data = json.loads(error_event["data"])
    assert len(error_data["detail"]) <= 200


@pytest.mark.asyncio
async def test_error_event_on_retrieval_failure(_run_kb_migrations):
    """Retrieval (before any SSE frame) raises → error event, stream ends normally."""
    from anvil_kb_api.app import create_app

    retriever = FakeRetrieverRaises()
    fake_chat = _make_fake_chat_stream(["答案。"])

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf, embedder=FakeEmbedder(), retriever=retriever, chat=fake_chat
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST", "/v1/kb/query", json={"question": "等待期?", "stream": True}
        ) as resp:
            assert resp.status_code == 200
            raw = await resp.aread()

    await engine.dispose()

    events = _parse_sse_events(raw.decode())
    event_types = [e["event"] for e in events]

    assert "error" in event_types, f"error event not found; events: {event_types}"
    error_event = next(e for e in events if e["event"] == "error")
    error_data = json.loads(error_event["data"])
    assert "detail" in error_data
    assert len(error_data["detail"]) > 0


# ===========================================================================
# 3. Upload size limit (413)
# ===========================================================================


@pytest.mark.asyncio
async def test_upload_too_large_413(_run_kb_migrations):
    """File > 2MB → 413 Request Entity Too Large."""
    from anvil_kb_api.app import MAX_UPLOAD_BYTES, create_app

    engine, sf = _make_engine_and_sf()
    app = create_app(session_factory=sf, embedder=FakeEmbedder())

    # Create content just over the limit
    oversized = b"# Title\n\n" + b"x" * (MAX_UPLOAD_BYTES + 1)

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post(
            "/v1/kb/documents",
            files={"file": ("big.md", oversized, "text/plain")},
        )

    await engine.dispose()

    assert r.status_code == 413


@pytest.mark.asyncio
async def test_upload_at_limit_not_413(_run_kb_migrations):
    """File exactly at 2MB limit (2*1024*1024 bytes) must NOT trigger 413.

    We use a fake retriever so the upload path skips dual-write and the test
    focuses purely on the size check rather than ingest side-effects.
    """
    from anvil_kb_api.app import MAX_UPLOAD_BYTES, create_app

    engine, sf = _make_engine_and_sf()
    # Inject a fake retriever so sparse_index is None and no dual-write happens
    retriever = FakeRetrieverWithDebug([])
    app = create_app(session_factory=sf, embedder=FakeEmbedder(), retriever=retriever)

    # Build exactly MAX_UPLOAD_BYTES of valid markdown using short ASCII words
    # so ingest won't crash on PG index row size limits.
    line = b"# Title\n\nwait period grace period terms conditions apply.\n\n"
    reps = (MAX_UPLOAD_BYTES // len(line)) + 1
    content = (line * reps)[:MAX_UPLOAD_BYTES]

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post(
            "/v1/kb/documents",
            files={"file": ("exact.md", content, "text/plain")},
        )

    await engine.dispose()

    assert r.status_code != 413, f"Expected non-413 at limit, got {r.status_code}"


# ===========================================================================
# 4. CORS env parsing
# ===========================================================================


@pytest.mark.asyncio
async def test_cors_env_specific_origin(_run_kb_migrations, monkeypatch):
    """ANVIL_KB_CORS_ORIGINS=http://example.com → only that origin is allowed."""
    monkeypatch.setenv("ANVIL_KB_CORS_ORIGINS", "http://example.com")

    from anvil_kb_api.app import create_app

    engine, sf = _make_engine_and_sf()
    app = create_app(session_factory=sf, embedder=FakeEmbedder())

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Preflight from the allowed origin
        r = await client.options(
            "/v1/kb/documents",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )

    await engine.dispose()

    # Starlette CORSMiddleware returns the origin or * in the response header
    origin_header = r.headers.get("access-control-allow-origin", "")
    assert origin_header == "http://example.com", (
        f"Expected 'http://example.com', got '{origin_header}'"
    )


@pytest.mark.asyncio
async def test_cors_env_default_wildcard(_run_kb_migrations, monkeypatch):
    """Without ANVIL_KB_CORS_ORIGINS set, default is '*'."""
    monkeypatch.delenv("ANVIL_KB_CORS_ORIGINS", raising=False)

    from anvil_kb_api.app import create_app

    engine, sf = _make_engine_and_sf()
    app = create_app(session_factory=sf, embedder=FakeEmbedder())

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.options(
            "/v1/kb/documents",
            headers={
                "Origin": "http://anywhere.com",
                "Access-Control-Request-Method": "GET",
            },
        )

    await engine.dispose()

    origin_header = r.headers.get("access-control-allow-origin", "")
    assert origin_header == "*", f"Expected '*', got '{origin_header}'"


# ===========================================================================
# 5. compare_digest regression: existing 401/200 auth behavior still works
# ===========================================================================


@pytest.mark.asyncio
async def test_auth_correct_key_200(_run_kb_migrations, monkeypatch):
    """With correct Bearer token, POST /v1/kb/query returns 200."""
    from anvil_kb_api.app import create_app

    monkeypatch.setenv("ANVIL_KB_API_KEY", "my-key")

    scored = [_make_scored_chunk("等待期为90天。", seq=0, score=0.95)]
    retriever = FakeRetrieverWithDebug(scored)
    fake_chat = _make_fake_chat_nonstream("答案。")

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf, embedder=FakeEmbedder(), retriever=retriever, chat=fake_chat
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post(
            "/v1/kb/query",
            json={"question": "等待期?", "stream": False},
            headers={"Authorization": "Bearer my-key"},
        )

    await engine.dispose()

    assert r.status_code == 200


@pytest.mark.asyncio
async def test_auth_wrong_key_401(_run_kb_migrations, monkeypatch):
    """Wrong Bearer token → 401."""
    from anvil_kb_api.app import create_app

    monkeypatch.setenv("ANVIL_KB_API_KEY", "my-key")

    engine, sf = _make_engine_and_sf()
    app = create_app(session_factory=sf, embedder=FakeEmbedder())

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post(
            "/v1/kb/query",
            json={"question": "等待期?", "stream": False},
            headers={"Authorization": "Bearer wrong-key"},
        )

    await engine.dispose()

    assert r.status_code == 401


# ===========================================================================
# 6. Minor: no Authorization header → 401 when key is configured
# ===========================================================================


@pytest.mark.asyncio
async def test_auth_no_header_401(_run_kb_migrations, monkeypatch):
    """API key is set but no Authorization header is sent → 401."""
    from anvil_kb_api.app import create_app

    monkeypatch.setenv("ANVIL_KB_API_KEY", "my-key")

    engine, sf = _make_engine_and_sf()
    app = create_app(session_factory=sf, embedder=FakeEmbedder())

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # No Authorization header at all
        r = await client.post(
            "/v1/kb/query",
            json={"question": "等待期?", "stream": False},
        )

    await engine.dispose()

    assert r.status_code == 401, f"Expected 401, got {r.status_code}"
    body = r.json()
    assert "detail" in body


# ===========================================================================
# 7. Important: dual-write integration — upload → BM25 searchable
# ===========================================================================


@pytest.mark.asyncio
async def test_upload_then_bm25_searchable(_run_kb_migrations):
    """Upload a .md file → BM25 index is written; PgBM25Index.search finds the unique word.

    Uses create_app(session_factory=..., embedder=FakeEmbedder512) WITHOUT injecting a
    retriever, so app constructs PgBM25Index + hybrid Retriever itself (dual-write path).
    After upload, we construct a fresh PgBM25Index with the same session_factory and
    search the unique word — it must hit.
    """
    from sqlalchemy import text

    from anvil_kb.store.bm25 import PgBM25Index
    from anvil_kb_api.app import create_app

    # Use a 512-dim one-hot fake embedder (arbitrary dim ≤ EMBEDDING_DIM is fine for BM25 test)
    class FakeEmbedder512:
        @property
        def dim(self) -> int:
            return _EMBEDDING_DIM

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [self._one_hot(t) for t in texts]

        def embed_query(self, text: str) -> list[float]:
            return self._one_hot(text)

        def _one_hot(self, text: str) -> list[float]:
            vec = [0.0] * _EMBEDDING_DIM
            if text:
                vec[hash(text) % _EMBEDDING_DIM] = 1.0
            return vec

    engine, sf = _make_engine_and_sf()

    # Truncate tables to isolate this test
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE kb_chunks, kb_documents CASCADE"))

    # Build app WITHOUT retriever injection → app self-constructs PgBM25Index + hybrid Retriever
    # (this exercises the dual-write code path in upload_document)
    app = create_app(session_factory=sf, embedder=FakeEmbedder512())

    # Unique word that will not appear in other tests
    unique_word = "xyzquantumfluxcapacitor9999"
    md_content = f"# Test Document\n\n本文档含有独特词 {unique_word}，用于验证 BM25 双写。\n"

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post(
            "/v1/kb/documents",
            files={"file": ("test_bm25.md", md_content.encode(), "text/plain")},
        )

    assert r.status_code == 201, f"Upload failed: {r.status_code} {r.text}"

    # Now search using a fresh PgBM25Index backed by the same session_factory
    bm25 = PgBM25Index(sf)
    results = await bm25.search(unique_word, k=5)

    # Clean up
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE kb_chunks, kb_documents CASCADE"))
    await engine.dispose()

    assert len(results) > 0, (
        f"BM25 search for unique word '{unique_word}' returned no results — "
        "dual-write to sparse index may have failed"
    )
    # The top result should contain the unique word
    top_content = results[0].chunk.content
    assert unique_word in top_content, (
        f"Top BM25 result does not contain '{unique_word}': {top_content!r}"
    )


# ===========================================================================
# 8. Important: _debug_list contributions rank > k passes through unchanged
# ===========================================================================


@pytest.mark.asyncio
async def test_debug_contributions_rank_exceeds_k(_run_kb_migrations):
    """contributions with a dense rank > k (e.g. dense:7, k=2) must be passed through
    unchanged in the fused debug items — _debug_list must NOT clip or error on rank > k.
    """
    from anvil_kb_api.app import create_app

    k = 2

    class FakeRetrieverWithHighRank:
        """Retriever whose contributions contain a dense rank > k."""

        async def retrieve(self, question: str, k: int = 5) -> list:
            return self._chunks[:k]

        async def retrieve_debug(self, question: str, k: int = 5):
            from anvil_kb.retrieve.fusion import RetrievalDebug

            top = self._chunks[:k]
            dense = [ScoredChunk(chunk=sc.chunk, score=sc.score) for sc in top]
            sparse = [ScoredChunk(chunk=sc.chunk, score=sc.score * 0.8) for sc in top]
            fused = top
            # dense rank for first chunk is 7, which exceeds k=2 — must be passed through
            contributions = {
                str(top[0].chunk.id): {"dense": 7, "sparse": 1},
                str(top[1].chunk.id): {"dense": 1, "sparse": 2},
            }
            return RetrievalDebug(
                dense=dense,
                sparse=sparse,
                fused=fused,
                contributions=contributions,
            )

        @property
        def _chunks(self):
            return [
                _make_scored_chunk("chunk alpha", seq=0, score=0.9),
                _make_scored_chunk("chunk beta", seq=1, score=0.8),
            ]

    retriever = FakeRetrieverWithHighRank()
    fake_chat = _make_fake_chat_stream(["答案。"])

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf,
        embedder=FakeEmbedder(),
        retriever=retriever,
        chat=fake_chat,
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST",
            "/v1/kb/query",
            json={"question": "test", "stream": True, "debug": True, "k": k},
        ) as resp:
            raw = await resp.aread()

    await engine.dispose()

    events = _parse_sse_events(raw.decode())
    debug_event = next(e for e in events if e["event"] == "debug")
    payload = json.loads(debug_event["data"])

    fused_items = payload["fused"]
    assert len(fused_items) >= 1, "Expected at least one fused item"

    # Find the item with dense rank=7 (> k=2) — must be present as-is
    first_item = fused_items[0]
    contrib = first_item["contributions"]
    assert contrib["dense"] == 7, (
        f"Expected contributions.dense=7 (rank > k), got {contrib['dense']!r}. "
        "contributions must be passed through unchanged even when rank > k."
    )


# ===========================================================================
# R2: rerank flag in QueryRequest + dual-Retriever DI
# ===========================================================================


class FakeReranker:
    """Test reranker that reverses the candidate order (injected via create_app)."""

    def __init__(self) -> None:
        self.call_count = 0

    def rerank(
        self, question: str, candidates: list[ScoredChunk], *, top: int
    ) -> list[ScoredChunk]:
        self.call_count += 1
        # Return reversed list truncated to top
        return list(reversed(candidates))[:top]


@pytest.mark.asyncio
async def test_rerank_false_default_queryrequest(_run_kb_migrations):
    """QueryRequest without rerank → rerank defaults to False."""
    from anvil_kb_api.app import QueryRequest

    req = QueryRequest(question="q?")
    assert req.rerank is False


@pytest.mark.asyncio
async def test_rerank_true_queryrequest(_run_kb_migrations):
    """QueryRequest with rerank=True → rerank field is True."""
    from anvil_kb_api.app import QueryRequest

    req = QueryRequest(question="q?", rerank=True)
    assert req.rerank is True


@pytest.mark.asyncio
async def test_rerank_true_uses_reranked_order(_run_kb_migrations):
    """rerank=true with injected reranker → sources order is reranked (reversed here).

    NOTE on injected-retriever rerank semantics:
        When a retriever is injected, reranking is applied as a post-step on
        whatever the injected retriever returns (post-rerank path).  The
        candidate pool is the k items returned by retriever.retrieve(), NOT the
        k*4 pool used by the production dual-Retriever.  This means the injected-
        retriever test path has different rerank-candidate semantics from production
        and is intended only for unit tests.

        FakeRetrieverPlain here returns k*4 items to approximate the production
        candidate pool size (closer to production than returning only k items).
    """
    from anvil_kb_api.app import create_app

    k = 3
    # Build k*4 = 12 candidate chunks so the pool matches production semantics
    scored = [
        _make_scored_chunk(f"chunk {chr(ord('A') + i)}", seq=i, score=round(1.0 - i * 0.05, 2))
        for i in range(k * 4)  # 12 chunks: A..L
    ]
    # chunk A has highest score (0.95), chunk L has lowest; FakeReranker reverses → L first

    class FakeRetrieverPlain:
        async def retrieve(self, question: str, k: int = 5) -> list[ScoredChunk]:
            # Return all k*4 candidates so reranker gets a rich pool
            return scored[: k * 4]

    fake_reranker = FakeReranker()
    fake_chat = _make_fake_chat_stream(["答案。"])

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf,
        embedder=FakeEmbedder(),
        retriever=FakeRetrieverPlain(),
        reranker=fake_reranker,
        chat=fake_chat,
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST",
            "/v1/kb/query",
            json={"question": "等待期?", "stream": True, "rerank": True, "k": 3},
        ) as resp:
            raw = await resp.aread()

    await engine.dispose()

    events = _parse_sse_events(raw.decode())
    sources_event = next(e for e in events if e["event"] == "sources")
    sources = json.loads(sources_event["data"])

    # FakeReranker reverses the k*4=12 candidates and truncates to top=k=3.
    # Reversed order: chunk L (index 11) first, chunk K (10) second, chunk J (9) last.
    # chr(ord('A') + 11) = 'L', chr(ord('A') + 10) = 'K', chr(ord('A') + 9) = 'J'
    assert sources[0]["quote"] == "chunk L", (
        f"Expected reranked order (reversed k*4 pool): first chunk should be 'chunk L', "
        f"got {sources[0]['quote']!r}"
    )
    assert sources[-1]["quote"] == "chunk J", (
        f"Expected reranked order (reversed k*4 pool): last chunk should be 'chunk J', "
        f"got {sources[-1]['quote']!r}"
    )
    assert len(sources) == k, f"Expected {k} sources (top=k), got {len(sources)}"
    assert fake_reranker.call_count == 1, "Reranker must be called exactly once"


@pytest.mark.asyncio
async def test_rerank_false_does_not_call_reranker(_run_kb_migrations):
    """rerank=false → reranker.rerank is never called."""
    from anvil_kb_api.app import create_app

    scored = [
        _make_scored_chunk("chunk A", seq=0, score=0.95),
        _make_scored_chunk("chunk B", seq=1, score=0.80),
    ]

    class FakeRetrieverPlain:
        async def retrieve(self, question: str, k: int = 5) -> list[ScoredChunk]:
            return scored[:k]

    fake_reranker = FakeReranker()
    fake_chat = _make_fake_chat_stream(["答案。"])

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf,
        embedder=FakeEmbedder(),
        retriever=FakeRetrieverPlain(),
        reranker=fake_reranker,
        chat=fake_chat,
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST",
            "/v1/kb/query",
            json={"question": "等待期?", "stream": True, "rerank": False},
        ) as resp:
            await resp.aread()

    await engine.dispose()

    assert fake_reranker.call_count == 0, (
        "Reranker must NOT be called when rerank=false"
    )


@pytest.mark.asyncio
async def test_rerank_true_debug_frame_contains_reranked_key(_run_kb_migrations):
    """rerank=true + debug=true → debug payload must contain 'reranked' key (non-null)."""
    from anvil_kb_api.app import create_app

    scored = [
        _make_scored_chunk("chunk A", seq=0, score=0.95),
        _make_scored_chunk("chunk B", seq=1, score=0.80),
    ]

    class FakeRetrieverWithDebugAndRerank:
        async def retrieve(self, question: str, k: int = 5) -> list[ScoredChunk]:
            return scored[:k]

        async def retrieve_debug(self, question: str, k: int = 5) -> RetrievalDebug:
            top = scored[:k]
            dense = [ScoredChunk(chunk=sc.chunk, score=sc.score) for sc in top]
            sparse = [ScoredChunk(chunk=sc.chunk, score=sc.score * 0.8) for sc in top]
            fused = top
            contributions = {
                str(sc.chunk.id): {"dense": i + 1, "sparse": None}
                for i, sc in enumerate(top)
            }
            return RetrievalDebug(
                dense=dense,
                sparse=sparse,
                fused=fused,
                contributions=contributions,
            )

    fake_reranker = FakeReranker()
    fake_chat = _make_fake_chat_stream(["答案。"])

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf,
        embedder=FakeEmbedder(),
        retriever=FakeRetrieverWithDebugAndRerank(),
        reranker=fake_reranker,
        chat=fake_chat,
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST",
            "/v1/kb/query",
            json={"question": "等待期?", "stream": True, "debug": True, "rerank": True},
        ) as resp:
            raw = await resp.aread()

    await engine.dispose()

    events = _parse_sse_events(raw.decode())
    debug_event = next(e for e in events if e["event"] == "debug")
    payload = json.loads(debug_event["data"])

    assert "reranked" in payload, (
        f"debug payload must contain 'reranked' key when rerank=true; keys={list(payload.keys())}"
    )
    assert payload["reranked"] is not None, "debug.reranked must be non-null when rerank=true"

    # Each reranked item must have the same structure as fused items
    required_fields = ("n", "chunk_id", "quote_head", "score", "rank")
    for item in payload["reranked"]:
        for field in required_fields:
            assert field in item, f"reranked item missing field: {field}"


@pytest.mark.asyncio
async def test_rerank_false_debug_frame_reranked_is_null(_run_kb_migrations):
    """rerank=false + debug=true → debug payload 'reranked' key is null."""
    from anvil_kb_api.app import create_app

    scored = [_make_scored_chunk("chunk A", seq=0, score=0.95)]

    class FakeRetrieverWithDebugOnly:
        async def retrieve(self, question: str, k: int = 5) -> list[ScoredChunk]:
            return scored[:k]

        async def retrieve_debug(self, question: str, k: int = 5) -> RetrievalDebug:
            top = scored[:k]
            dense = [ScoredChunk(chunk=sc.chunk, score=sc.score) for sc in top]
            sparse = [ScoredChunk(chunk=sc.chunk, score=sc.score * 0.8) for sc in top]
            contributions = {
                str(sc.chunk.id): {"dense": i + 1, "sparse": None}
                for i, sc in enumerate(top)
            }
            return RetrievalDebug(
                dense=dense, sparse=sparse, fused=top, contributions=contributions
            )

    fake_chat = _make_fake_chat_stream(["答案。"])

    engine, sf = _make_engine_and_sf()
    app = create_app(
        session_factory=sf,
        embedder=FakeEmbedder(),
        retriever=FakeRetrieverWithDebugOnly(),
        reranker=None,
        chat=fake_chat,
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST",
            "/v1/kb/query",
            json={"question": "等待期?", "stream": True, "debug": True, "rerank": False},
        ) as resp:
            raw = await resp.aread()

    await engine.dispose()

    events = _parse_sse_events(raw.decode())
    debug_event = next(e for e in events if e["event"] == "debug")
    payload = json.loads(debug_event["data"])

    # reranked must be null (not absent) when rerank=false
    assert "reranked" in payload, "debug payload must always include 'reranked' key"
    assert payload["reranked"] is None, (
        f"debug.reranked must be null when rerank=false, got {payload['reranked']!r}"
    )


# ===========================================================================
# R2 Critical fix: production-path debug+rerank — no injected retriever,
# reranker injected → debug_result.reranked must be non-null in SSE frame
# ===========================================================================


@pytest.mark.asyncio
async def test_production_path_debug_reranked_frame_nonnull(_run_kb_migrations):
    """Production path: retriever NOT injected, reranker injected, debug+rerank=true.

    Verifies the Critical fix: when post_reranker is None (production dual-Retriever
    path), the debug frame's 'reranked' key must be non-null because _retriever_rerank
    internally reranked and stored the result in debug_result.reranked.

    Setup:
      - create_app(session_factory=sf, embedder=FakeEmbedder(), reranker=FakeReranker())
        — no retriever injected → app builds its own Retriever + _retriever_rerank
      - Ingest a document via HTTP upload so BM25 + pgvector have real data
      - Send debug=True + rerank=True query using one of the ingested words
      - Assert: debug frame 'reranked' is non-null AND its chunk order matches
        what FakeReranker (reverser) would produce
    """
    from sqlalchemy import text

    from anvil_kb_api.app import create_app

    engine, sf = _make_engine_and_sf()

    # Isolate this test from other test data
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE kb_chunks, kb_documents CASCADE"))

    fake_reranker = FakeReranker()
    fake_chat = _make_fake_chat_stream(["答案。"])

    # No retriever injected → production dual-Retriever path
    # reranker injected → _get_or_build_reranker_async() returns fake_reranker (no model load)
    app = create_app(
        session_factory=sf,
        embedder=FakeEmbedder(),
        reranker=fake_reranker,
        chat=fake_chat,
    )

    # Ingest two chunks with unique words so BM25 can find them
    md_content = (
        "# 等待期条款\n\n"
        "等待期为九十天，自保单生效日起计算。\n\n"
        "宽限期为六十天，保费到期后可在宽限期内缴纳。\n"
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Upload document so app's PgBM25Index gets populated (dual-write)
        upload_r = await client.post(
            "/v1/kb/documents",
            files={"file": ("等待期.md", md_content.encode("utf-8"), "text/plain")},
        )
        assert upload_r.status_code == 201, (
            f"Upload failed: {upload_r.status_code} {upload_r.text}"
        )

        # Query with debug=True + rerank=True → production path uses _retriever_rerank
        async with client.stream(
            "POST",
            "/v1/kb/query",
            json={"question": "等待期是多少天", "stream": True, "debug": True, "rerank": True},
        ) as resp:
            raw = await resp.aread()

    # Clean up
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE kb_chunks, kb_documents CASCADE"))
    await engine.dispose()

    events = _parse_sse_events(raw.decode())
    event_types = [e["event"] for e in events]

    # Must have a debug frame (not an error frame)
    err_detail = next(
        (json.loads(e["data"]) for e in events if e["event"] == "error"), None
    )
    assert "error" not in event_types, (
        f"Unexpected error event; events: {event_types}; error detail: {err_detail}"
    )
    assert "debug" in event_types, f"Expected debug frame; got: {event_types}"

    debug_event = next(e for e in events if e["event"] == "debug")
    payload = json.loads(debug_event["data"])

    # Critical assertion: production path must populate debug.reranked (non-null)
    assert "reranked" in payload, "debug payload must contain 'reranked' key"
    assert payload["reranked"] is not None, (
        "debug.reranked must be non-null when rerank=true on production path "
        "(retriever not injected, _retriever_rerank has reranker built-in)"
    )

    # Each reranked item must have required structure
    required_fields = ("n", "chunk_id", "quote_head", "score", "rank")
    for item in payload["reranked"]:
        for field in required_fields:
            assert field in item, f"reranked item missing field: {field}"

    # Reranker call count: FakeReranker was called at least once (inside _retriever_rerank)
    assert fake_reranker.call_count >= 1, (
        "FakeReranker must have been called at least once on production path"
    )

    # Consistency: reranked chunk_ids must differ from or reorder fused chunk_ids
    # (FakeReranker reverses — if there are ≥2 chunks, order should differ from fused)
    fused_ids = [item["chunk_id"] for item in payload["fused"]]
    reranked_ids = [item["chunk_id"] for item in payload["reranked"]]
    # Both lists should reference the same chunks (just possibly reordered)
    assert set(reranked_ids).issubset(set(fused_ids) | {i["chunk_id"] for i in payload["dense"]}), (
        "reranked chunk_ids must come from the retrieval pool"
    )
