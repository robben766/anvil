"""Tests for POST /v1/kb/query — non-streaming JSON and SSE streaming (TDD)."""

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
from anvil_kb.store.base import Chunk, ScoredChunk
from anvil_kb_api.app import _sse_event

TEST_DB_URL = os.environ.get(
    "ANVIL_TEST_DATABASE_URL", "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test"
)


class FakeEmbedder:
    """One-hot embedder: no model needed, dimension matches EMBEDDING_DIM."""

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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

CHUNK_A_CONTENT = "等待期为90天。"
CHUNK_B_CONTENT = "宽限期为60天。"


def _make_scored_chunk(content: str, seq: int = 0, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            seq=seq,
            content=content,
            header_path=f"条款 > 第{seq + 1}节" if seq == 0 else "",
            start_offset=seq * 100,
            end_offset=seq * 100 + len(content),
        ),
        score=score,
    )


def _parse_sse_events(raw: str) -> list[dict]:
    """Parse raw SSE text into list of {event, data} dicts.

    JSON payload 由 json.dumps 保证单行,本解析器按 single-data-per-event 处理。
    """
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
# _sse_event unit tests
# ---------------------------------------------------------------------------


def test_sse_event_format():
    frame = _sse_event("delta", {"text": "hello"})
    assert frame == 'event: delta\ndata: {"text": "hello"}\n\n'


def test_sse_event_sources():
    frame = _sse_event("sources", [{"n": 1, "chunk_id": "abc"}])
    assert frame.startswith("event: sources\n")
    assert "data: " in frame
    assert frame.endswith("\n\n")
    payload = json.loads(frame.split("data: ", 1)[1].strip())
    assert payload[0]["n"] == 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeRetriever:
    """Returns a fixed list of ScoredChunks ignoring the question."""

    def __init__(self, scored_chunks: list[ScoredChunk]) -> None:
        self._chunks = scored_chunks

    async def retrieve(self, question: str, k: int = 5) -> list[ScoredChunk]:
        return self._chunks[:k]


def _make_fake_chat_nonstream(answer_text: str):
    async def _chat(model, messages, **kw):
        return SimpleNamespace(content=answer_text)

    return _chat


def _make_fake_chat_stream(chunks: list[str]):
    """Streaming fake chat: yields ChatChunk(delta=...) for each token, then usage tail."""
    from anvil_gateway.types import ChatChunk

    async def _chat(model, messages, **kw):
        async def _gen():
            for i, c in enumerate(chunks):
                finish = "stop" if i == len(chunks) - 1 else None
                yield ChatChunk(delta=c, finish_reason=finish)
            # usage tail block — must NOT produce a delta event
            yield ChatChunk(delta="", finish_reason="stop", usage=None)

        return _gen()

    return _chat


@pytest.fixture
async def query_client_nonstream(_run_kb_migrations, monkeypatch):  # noqa: ARG001
    """ASGI client wired with fake retriever + non-streaming fake chat."""
    from anvil_kb_api.app import create_app

    scored = [
        _make_scored_chunk(CHUNK_A_CONTENT, seq=0, score=0.95),
        _make_scored_chunk(CHUNK_B_CONTENT, seq=1, score=0.80),
    ]
    retriever = FakeRetriever(scored)

    answer_text = "等待期是90天 [1]。"
    fake_chat = _make_fake_chat_nonstream(answer_text)

    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    app = create_app(
        session_factory=sf,
        embedder=FakeEmbedder(),
        retriever=retriever,
        chat=fake_chat,
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, scored

    await engine.dispose()


@pytest.fixture
async def query_client_stream(_run_kb_migrations, monkeypatch):  # noqa: ARG001
    """ASGI client wired with fake retriever + streaming fake chat."""
    from anvil_kb_api.app import create_app

    scored = [
        _make_scored_chunk(CHUNK_A_CONTENT, seq=0, score=0.95),
        _make_scored_chunk(CHUNK_B_CONTENT, seq=1, score=0.80),
    ]
    retriever = FakeRetriever(scored)

    # The answer references [1] so citations should have one entry
    stream_tokens = ["等待期是90天 ", "[1]。"]
    fake_chat = _make_fake_chat_stream(stream_tokens)

    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    app = create_app(
        session_factory=sf,
        embedder=FakeEmbedder(),
        retriever=retriever,
        chat=fake_chat,
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, scored

    await engine.dispose()


# ---------------------------------------------------------------------------
# Non-streaming tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_nonstream_200(query_client_nonstream):
    client, scored = query_client_nonstream
    r = await client.post(
        "/v1/kb/query", json={"question": "等待期是多少?", "stream": False}
    )
    assert r.status_code == 200
    body = r.json()
    assert "text" in body
    assert "citations" in body


@pytest.mark.asyncio
async def test_query_nonstream_citation_fields(query_client_nonstream):
    """Citations must have all required fields; quote == chunk content; offsets match."""
    client, scored = query_client_nonstream
    r = await client.post(
        "/v1/kb/query", json={"question": "等待期是多少?", "stream": False}
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["citations"]) == 1
    c = body["citations"][0]
    # Required fields
    required = (
        "n", "chunk_id", "document_id", "quote", "header_path", "start_offset", "end_offset"
    )
    for field in required:
        assert field in c, f"missing field: {field}"
    # n is 1-based
    assert c["n"] == 1
    # quote must equal the chunk's content
    assert c["quote"] == CHUNK_A_CONTENT
    # IDs must be strings (uuid stringified)
    assert isinstance(c["chunk_id"], str)
    assert isinstance(c["document_id"], str)
    # offsets match the scored chunk
    sc = scored[0]
    assert c["start_offset"] == sc.chunk.start_offset
    assert c["end_offset"] == sc.chunk.end_offset
    # header_path must come from the chunk
    assert c["header_path"] == sc.chunk.header_path


@pytest.mark.asyncio
async def test_query_nonstream_text_contains_answer(query_client_nonstream):
    client, _ = query_client_nonstream
    r = await client.post(
        "/v1/kb/query", json={"question": "等待期是多少?", "stream": False}
    )
    assert "[1]" in r.json()["text"]


@pytest.mark.asyncio
async def test_query_nonstream_default_stream_is_true(query_client_stream):
    """Omitting 'stream' field defaults to true → must return text/event-stream."""
    client, _ = query_client_stream

    async with client.stream("POST", "/v1/kb/query", json={"question": "等待期?"}) as resp:
        assert resp.status_code == 200
        ct = resp.headers["content-type"]
        assert "text/event-stream" in ct
        assert "charset=utf-8" in ct


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_stream_content_type(query_client_stream):
    client, _ = query_client_stream
    req_body = {"question": "等待期?", "stream": True}
    async with client.stream("POST", "/v1/kb/query", json=req_body) as resp:
        assert resp.status_code == 200
        ct = resp.headers["content-type"]
        assert "text/event-stream" in ct
        assert "charset=utf-8" in ct


@pytest.mark.asyncio
async def test_query_stream_event_order(query_client_stream):
    """Event order must be: sources → delta×N → done."""
    client, _ = query_client_stream
    req_body = {"question": "等待期?", "stream": True}
    async with client.stream("POST", "/v1/kb/query", json=req_body) as resp:
        raw = await resp.aread()

    events = _parse_sse_events(raw.decode())
    event_types = [e["event"] for e in events]

    assert event_types[0] == "sources"
    assert event_types[-1] == "done"
    for t in event_types[1:-1]:
        assert t == "delta", f"unexpected event type: {t}"


@pytest.mark.asyncio
async def test_query_stream_sources_fields(query_client_stream):
    """sources array must contain score and all required citation-like fields."""
    client, scored = query_client_stream
    req_body = {"question": "等待期?", "stream": True}
    async with client.stream("POST", "/v1/kb/query", json=req_body) as resp:
        raw = await resp.aread()

    events = _parse_sse_events(raw.decode())
    sources_event = next(e for e in events if e["event"] == "sources")
    sources = json.loads(sources_event["data"])

    assert len(sources) == len(scored)
    required_src = (
        "n", "chunk_id", "document_id", "quote",
        "header_path", "start_offset", "end_offset", "score",
    )
    for src in sources:
        for field in required_src:
            assert field in src, f"missing field in sources item: {field}"
    # n values are 1-based sequential
    assert [s["n"] for s in sources] == list(range(1, len(scored) + 1))
    # score is present and a float
    assert isinstance(sources[0]["score"], float)


@pytest.mark.asyncio
async def test_query_stream_delta_events(query_client_stream):
    """Delta events must be present and their text concatenated matches done.text."""
    client, _ = query_client_stream
    req_body = {"question": "等待期?", "stream": True}
    async with client.stream("POST", "/v1/kb/query", json=req_body) as resp:
        raw = await resp.aread()

    events = _parse_sse_events(raw.decode())
    delta_payloads = [json.loads(e["data"])["text"] for e in events if e["event"] == "delta"]
    done_event = next(e for e in events if e["event"] == "done")
    done_data = json.loads(done_event["data"])

    assert len(delta_payloads) > 0
    assert "".join(delta_payloads) == done_data["text"]


@pytest.mark.asyncio
async def test_query_stream_done_citations(query_client_stream):
    """done event citations must match expected structure."""
    client, scored = query_client_stream
    req_body = {"question": "等待期?", "stream": True}
    async with client.stream("POST", "/v1/kb/query", json=req_body) as resp:
        raw = await resp.aread()

    events = _parse_sse_events(raw.decode())
    done_event = next(e for e in events if e["event"] == "done")
    done_data = json.loads(done_event["data"])

    assert "text" in done_data
    assert "citations" in done_data
    citations = done_data["citations"]
    assert len(citations) == 1
    c = citations[0]
    required = (
        "n", "chunk_id", "document_id", "quote", "header_path", "start_offset", "end_offset"
    )
    for field in required:
        assert field in c, f"missing field in done.citations: {field}"
    assert c["quote"] == CHUNK_A_CONTENT
    # header_path from the chunk
    assert c["header_path"] == scored[0].chunk.header_path


@pytest.mark.asyncio
async def test_query_stream_done_citations_consistent_with_full_text(query_client_stream):
    """done.citations [n] markers must be consistent with the full assembled text."""
    client, _ = query_client_stream
    req_body = {"question": "等待期?", "stream": True}
    async with client.stream("POST", "/v1/kb/query", json=req_body) as resp:
        raw = await resp.aread()

    events = _parse_sse_events(raw.decode())
    done_event = next(e for e in events if e["event"] == "done")
    done_data = json.loads(done_event["data"])

    full_text = done_data["text"]
    for c in done_data["citations"]:
        assert f"[{c['n']}]" in full_text


# ---------------------------------------------------------------------------
# k=1 test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_stream_k1_sources_length_1(_run_kb_migrations):
    """k=1 → sources event must contain exactly 1 item."""
    from anvil_kb_api.app import create_app

    scored = [
        _make_scored_chunk(CHUNK_A_CONTENT, seq=0, score=0.95),
        _make_scored_chunk(CHUNK_B_CONTENT, seq=1, score=0.80),
    ]
    retriever = FakeRetriever(scored)
    fake_chat = _make_fake_chat_stream(["答案 [1]。"])

    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
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
            "POST", "/v1/kb/query", json={"question": "等待期?", "k": 1, "stream": True}
        ) as resp:
            raw = await resp.aread()

    events = _parse_sse_events(raw.decode())
    sources_event = next(e for e in events if e["event"] == "sources")
    sources = json.loads(sources_event["data"])
    assert len(sources) == 1

    await engine.dispose()


# ---------------------------------------------------------------------------
# Validation: empty/blank question → 400
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_empty_question_400(_run_kb_migrations):
    from anvil_kb_api.app import create_app

    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    app = create_app(session_factory=sf, embedder=FakeEmbedder())

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post("/v1/kb/query", json={"question": "", "stream": False})
        assert r.status_code == 400

        r2 = await client.post("/v1/kb/query", json={"question": "   ", "stream": False})
        assert r2.status_code == 400

    await engine.dispose()


@pytest.mark.asyncio
async def test_query_empty_question_stream_400(_run_kb_migrations):
    """Empty question → 400 also for stream=true."""
    from anvil_kb_api.app import create_app

    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    app = create_app(session_factory=sf, embedder=FakeEmbedder())

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post("/v1/kb/query", json={"question": "", "stream": True})
        assert r.status_code == 400

    await engine.dispose()


@pytest.mark.asyncio
async def test_query_k_zero_422(_run_kb_migrations):
    """k=0 violates ge=1 constraint → Pydantic validation error 422."""
    from anvil_kb_api.app import create_app

    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    app = create_app(session_factory=sf, embedder=FakeEmbedder())

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post(
            "/v1/kb/query", json={"question": "等待期?", "k": 0, "stream": False}
        )
        assert r.status_code == 422

    await engine.dispose()


# ---------------------------------------------------------------------------
# Auth: POST /v1/kb/query → 401 when ANVIL_KB_API_KEY set and no header
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_auth_no_header_returns_401(
    _run_kb_migrations, monkeypatch  # noqa: ARG001
):
    """With ANVIL_KB_API_KEY set, POST /v1/kb/query without Authorization → 401."""
    from anvil_kb_api.app import create_app

    monkeypatch.setenv("ANVIL_KB_API_KEY", "secret-key")

    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    app = create_app(session_factory=sf, embedder=FakeEmbedder())

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post(
            "/v1/kb/query", json={"question": "等待期?", "stream": False}
        )
        assert r.status_code == 401

    await engine.dispose()
