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
