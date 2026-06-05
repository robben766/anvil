"""anvil-kb-api: FastAPI CRUD shell for the anvil-kb RAG pipeline.

Endpoints (prefix /v1/kb):
  POST   /v1/kb/documents      — upload .md/.txt, ingest, return 201
  GET    /v1/kb/documents      — list all documents with chunk counts
  GET    /v1/kb/documents/{id} — get document detail including content
  DELETE /v1/kb/documents/{id} — delete document (204, idempotent)
  POST   /v1/kb/query          — RAG query; stream=true (default) → SSE, stream=false → JSON

Auth: set ANVIL_KB_API_KEY to require Bearer token on all endpoints.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_kb.db import ChunkRow, DocumentRow, make_engine
from anvil_kb.embed import Embedder, FastEmbedEmbedder
from anvil_kb.generate import answer, answer_stream
from anvil_kb.ingest.pipeline import ingest_markdown
from anvil_kb.retrieve.retriever import Retriever
from anvil_kb.store.base import ScoredChunk
from anvil_kb.store.pg import PgVectorStore

_ALLOWED_EXTENSIONS = {".md", ".txt"}


# ---------------------------------------------------------------------------
# SSE helper (pure function — unit-testable)
# ---------------------------------------------------------------------------


def _sse_event(name: str, payload: object) -> str:
    """Serialise *payload* as a named SSE frame.

    Returns a string of the form::

        event: <name>\\n
        data: <json>\\n
        \\n
    """
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str
    k: int = 5
    stream: bool = True


# ---------------------------------------------------------------------------
# Citation serialisation helpers
# ---------------------------------------------------------------------------


def _citation_to_dict(c, retrieved: list[ScoredChunk]) -> dict:
    """Serialise a Citation dataclass to a JSON-safe dict.

    Citation has no header_path field; we backfill it from the corresponding
    ScoredChunk (retrieved[n-1]).
    """
    sc = retrieved[c.n - 1]
    return {
        "n": c.n,
        "chunk_id": str(c.chunk_id),
        "document_id": str(c.document_id),
        "quote": c.quote,
        "header_path": sc.chunk.header_path,
        "start_offset": c.start_offset,
        "end_offset": c.end_offset,
    }


def _sources_list(retrieved: list[ScoredChunk]) -> list[dict]:
    """Build the SSE ``sources`` payload: one item per ScoredChunk (1-based n)."""
    items = []
    for i, sc in enumerate(retrieved):
        items.append(
            {
                "n": i + 1,
                "chunk_id": str(sc.chunk.id),
                "document_id": str(sc.chunk.document_id),
                "quote": sc.chunk.content,
                "header_path": sc.chunk.header_path,
                "start_offset": sc.chunk.start_offset,
                "end_offset": sc.chunk.end_offset,
                "score": sc.score,
            }
        )
    return items


def _check_auth(authorization: str | None) -> None:
    """Mirror proxy/app.py auth check: ANVIL_KB_API_KEY controls requirement."""
    expected = os.environ.get("ANVIL_KB_API_KEY")
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid api key")


def create_app(
    *,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    embedder: Embedder | None = None,
    retriever: Any | None = None,
    chat: Any | None = None,
) -> FastAPI:
    """Factory for the FastAPI application (DI entry point for tests).

    session_factory=None → lazy-build from ANVIL_DATABASE_URL env var.
    embedder=None        → FastEmbedEmbedder() (lazy local model).
    retriever=None       → Retriever(embedder, store) (default dense retriever).
    chat=None            → anvil_gateway.chat (default LLM backend).
    """
    # Resolve dependencies
    if session_factory is None:
        engine = make_engine()  # reads ANVIL_DATABASE_URL
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    if embedder is None:
        embedder = FastEmbedEmbedder()

    store = PgVectorStore(session_factory)

    # retriever can be injected for tests; defaults to Retriever(embedder, store)
    _retriever = retriever if retriever is not None else Retriever(embedder, store)

    app = FastAPI(title="anvil-kb-api", version="0.1.0")

    # ------------------------------------------------------------------ #
    # POST /v1/kb/documents
    # ------------------------------------------------------------------ #
    @app.post("/v1/kb/documents", status_code=201)
    async def upload_document(
        file: UploadFile,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(authorization)

        # Validate extension
        filename = file.filename or ""
        suffix = ""
        if "." in filename:
            suffix = "." + filename.rsplit(".", 1)[-1].lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported file type '{suffix}'; allowed: .md, .txt",
            )

        raw = await file.read()
        text = raw.decode("utf-8")

        stem = filename.rsplit(".", 1)[0] if "." in filename else filename

        try:
            doc, chunk_count = await ingest_markdown(
                title=stem,
                source_name=filename,
                text=text,
                embedder=embedder,
                store=store,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "id": str(doc.id),
            "title": doc.title,
            "source_name": doc.source_name,
            "chunk_count": chunk_count,
        }

    # ------------------------------------------------------------------ #
    # GET /v1/kb/documents
    # ------------------------------------------------------------------ #
    @app.get("/v1/kb/documents")
    async def list_documents(
        authorization: str | None = Header(default=None),
    ) -> list[dict[str, Any]]:
        _check_auth(authorization)

        async with session_factory() as session:
            # Left-join with chunk count, ordered by created_at asc
            stmt = (
                select(
                    DocumentRow.id,
                    DocumentRow.title,
                    DocumentRow.source_name,
                    DocumentRow.created_at,
                    func.count(ChunkRow.id).label("chunk_count"),
                )
                .outerjoin(ChunkRow, ChunkRow.document_id == DocumentRow.id)
                .group_by(
                    DocumentRow.id,
                    DocumentRow.title,
                    DocumentRow.source_name,
                    DocumentRow.created_at,
                )
                .order_by(DocumentRow.created_at.asc())
            )
            result = await session.execute(stmt)
            rows = result.all()

        return [
            {
                "id": str(row.id),
                "title": row.title,
                "source_name": row.source_name,
                "chunk_count": row.chunk_count,
                "created_at": _serialize_dt(row.created_at),
            }
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    # GET /v1/kb/documents/{id}
    # ------------------------------------------------------------------ #
    @app.get("/v1/kb/documents/{doc_id}")
    async def get_document(
        doc_id: uuid.UUID,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_auth(authorization)

        async with session_factory() as session:
            stmt = (
                select(
                    DocumentRow.id,
                    DocumentRow.title,
                    DocumentRow.source_name,
                    DocumentRow.content,
                    DocumentRow.created_at,
                    func.count(ChunkRow.id).label("chunk_count"),
                )
                .outerjoin(ChunkRow, ChunkRow.document_id == DocumentRow.id)
                .where(DocumentRow.id == doc_id)
                .group_by(
                    DocumentRow.id,
                    DocumentRow.title,
                    DocumentRow.source_name,
                    DocumentRow.content,
                    DocumentRow.created_at,
                )
            )
            result = await session.execute(stmt)
            row = result.one_or_none()

        if row is None:
            raise HTTPException(status_code=404, detail="document not found")

        return {
            "id": str(row.id),
            "title": row.title,
            "source_name": row.source_name,
            "chunk_count": row.chunk_count,
            "created_at": _serialize_dt(row.created_at),
            "content": row.content,
        }

    # ------------------------------------------------------------------ #
    # DELETE /v1/kb/documents/{id}
    # ------------------------------------------------------------------ #
    @app.delete("/v1/kb/documents/{doc_id}", status_code=204)
    async def delete_document(
        doc_id: uuid.UUID,
        authorization: str | None = Header(default=None),
    ) -> Response:
        _check_auth(authorization)
        await store.delete_document(doc_id)
        return Response(status_code=204)

    # ------------------------------------------------------------------ #
    # POST /v1/kb/query
    # ------------------------------------------------------------------ #
    @app.post("/v1/kb/query")
    async def query_kb(
        req: QueryRequest,
        authorization: str | None = Header(default=None),
    ):
        _check_auth(authorization)

        if not req.question or not req.question.strip():
            raise HTTPException(status_code=400, detail="question must not be empty")

        if req.stream:
            # --- SSE streaming path ---
            async def _event_generator():
                async for event_type, payload in answer_stream(
                    req.question,
                    await _retriever.retrieve(req.question, req.k),
                    chat=chat,
                ):
                    if event_type == "sources":
                        # payload is list[ScoredChunk]
                        yield _sse_event("sources", _sources_list(payload))
                    elif event_type == "delta":
                        # payload is str
                        yield _sse_event("delta", {"text": payload})
                    elif event_type == "done":
                        # payload is KbAnswer — need the retrieved list to build citations
                        # Re-retrieve is not ideal; instead we capture retrieved inside the gen
                        pass

            # We need retrieved for citation serialisation in done event,
            # so we collect it upfront and drive the stream manually.
            async def _sse_generator():
                retrieved = await _retriever.retrieve(req.question, req.k)
                async for event_type, payload in answer_stream(
                    req.question,
                    retrieved,
                    chat=chat,
                ):
                    if event_type == "sources":
                        yield _sse_event("sources", _sources_list(retrieved))
                    elif event_type == "delta":
                        yield _sse_event("delta", {"text": payload})
                    elif event_type == "done":
                        kb_answer = payload
                        done_payload = {
                            "text": kb_answer.text,
                            "citations": [
                                _citation_to_dict(c, retrieved)
                                for c in kb_answer.citations
                            ],
                        }
                        yield _sse_event("done", done_payload)

            return StreamingResponse(
                _sse_generator(),
                media_type="text/event-stream",
            )

        else:
            # --- Non-streaming JSON path ---
            retrieved = await _retriever.retrieve(req.question, req.k)
            kb_answer = await answer(req.question, retrieved, chat=chat)
            return {
                "text": kb_answer.text,
                "citations": [
                    _citation_to_dict(c, retrieved) for c in kb_answer.citations
                ],
            }

    return app


def _serialize_dt(dt: datetime) -> str:
    """Serialize a timezone-aware datetime to ISO-8601 string."""
    return dt.isoformat()


def run() -> None:
    """Entry point: uvicorn on 0.0.0.0:8400."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8400)
