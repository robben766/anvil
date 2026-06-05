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

import asyncio
import json
import os
import secrets
import uuid
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_kb.db import ChunkRow, DocumentRow, make_engine
from anvil_kb.embed import Embedder, FastEmbedEmbedder
from anvil_kb.generate import answer, answer_stream
from anvil_kb.ingest.pdf import parse_pdf
from anvil_kb.ingest.pipeline import ingest_markdown
from anvil_kb.retrieve.retriever import Retriever
from anvil_kb.store.base import ScoredChunk
from anvil_kb.store.bm25 import PgBM25Index
from anvil_kb.store.pg import PgVectorStore

_ALLOWED_EXTENSIONS = {".md", ".txt", ".pdf"}

# H4: maximum upload size (2 MB)
MAX_UPLOAD_BYTES = 2 * 1024 * 1024


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
    k: int = Field(5, ge=1, le=20)
    stream: bool = True
    debug: bool = False
    rerank: bool = False


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


def _debug_list(scored_chunks: list[ScoredChunk], *, include_contributions: bool = False,
                contributions: dict | None = None) -> list[dict]:
    """Build a debug sub-list (dense, sparse, or fused).

    Args:
        scored_chunks:         List of ScoredChunk items.
        include_contributions: If True, add 'contributions' key per item (fused only).
        contributions:         The full contributions map from RetrievalDebug.

    Note on ``rank``:
        rank = 本列表内位置(1..k)；fused 项的 contributions 是 k*4 候选池内排位,
        可能 > k（例如 dense:7 表示该 chunk 在稠密候选池中排第 7，超出最终返回的 k）。
        contributions 的值原样透传，不裁剪。
    """
    items = []
    for i, sc in enumerate(scored_chunks):
        item: dict[str, Any] = {
            "n": i + 1,
            "chunk_id": str(sc.chunk.id),
            "quote_head": sc.chunk.content[:40],
            "score": round(sc.score, 4),
            "rank": i + 1,
        }
        if include_contributions and contributions is not None:
            chunk_contrib = contributions.get(str(sc.chunk.id), {"dense": None, "sparse": None})
            item["contributions"] = chunk_contrib
        items.append(item)
    return items


def _build_debug_payload(debug_result, *, reranked: list | None = None) -> dict:
    """Convert a RetrievalDebug into the SSE debug frame payload.

    Args:
        debug_result:  RetrievalDebug from retrieve_debug().
        reranked:      Post-rerank ScoredChunk list (None → null in JSON).
    """
    payload: dict = {
        "dense": _debug_list(debug_result.dense),
        "sparse": _debug_list(debug_result.sparse),
        "fused": _debug_list(
            debug_result.fused,
            include_contributions=True,
            contributions=debug_result.contributions,
        ),
        "reranked": _debug_list(reranked) if reranked is not None else None,
    }
    return payload


def _check_auth(authorization: str | None) -> None:
    """Mirror proxy/app.py auth check: ANVIL_KB_API_KEY controls requirement.

    Uses secrets.compare_digest to prevent timing attacks.
    Both sides are encoded to bytes to avoid TypeError on non-ASCII input.
    """
    expected = os.environ.get("ANVIL_KB_API_KEY")
    if not expected:
        return
    expected_header = f"Bearer {expected}"
    provided = authorization or ""
    # encode both to bytes — secrets.compare_digest requires both to be the same type
    if not secrets.compare_digest(provided.encode(), expected_header.encode()):
        raise HTTPException(status_code=401, detail="invalid api key")


def create_app(
    *,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    embedder: Embedder | None = None,
    retriever: Any | None = None,
    reranker: Any | None = None,
    chat: Any | None = None,
) -> FastAPI:
    """Factory for the FastAPI application (DI entry point for tests).

    session_factory=None → lazy-build from ANVIL_DATABASE_URL env var.
    embedder=None        → FastEmbedEmbedder() (lazy local model).
    retriever=None       → Retriever(embedder, store, sparse_index, mode='hybrid').
    reranker=None        → lazy-construct FastEmbedReranker on first rerank=true request.
    chat=None            → anvil_gateway.chat (default LLM backend).

    Dual-Retriever design (R2):
        Two Retriever instances sharing the same store/embedder/sparse_index:
          _retriever       — plain retriever (no reranker), used when rerank=false
          _retriever_rerank — retriever with reranker, used when rerank=true
        The reranker for _retriever_rerank is:
          - injected `reranker` param (tests / custom deployment), OR
          - a module-level shared FastEmbedReranker (lazy-loaded on first rerank=true request)

    NOTE on injected retriever semantics:
        When a ``retriever`` is injected (test path), reranking is applied as a
        post-step on whatever the injected retriever returns (``post_reranker``
        path).  The candidate pool is therefore the k items returned by
        ``retriever.retrieve()``, NOT the k*4 pool used by the production
        dual-Retriever.  This means the injected-retriever test path has
        different rerank-candidate semantics from production and is intended
        only for unit tests.
    """
    # Resolve dependencies
    if session_factory is None:
        engine = make_engine()  # reads ANVIL_DATABASE_URL
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    if embedder is None:
        embedder = FastEmbedEmbedder()

    store = PgVectorStore(session_factory)

    # H4: build hybrid retriever with PgBM25Index when retriever is not injected
    if retriever is None:
        sparse_index = PgBM25Index(session_factory)
        _retriever = Retriever(embedder, store, sparse_index=sparse_index, mode="hybrid")
    else:
        # Injected retriever (tests); also keep sparse_index reference for ingest dual-write
        _retriever = retriever
        sparse_index = None  # no dual-write when test retriever is injected

    # R2: dual-Retriever — rerank variant shares store/embedder/sparse_index
    # The reranker is injected (tests) or will be lazy-loaded on first rerank=true request.
    # We capture the injected reranker; lazy-load state lives in the closure below.
    _injected_reranker = reranker
    # Mutable container so closure can assign
    _shared_reranker: list[Any] = []  # _shared_reranker[0] once initialised
    # asyncio.Lock eliminates TOCTOU race when multiple concurrent requests trigger
    # the first rerank=true call simultaneously (lazy-load is not idempotent because
    # FastEmbedReranker downloads / loads the model on construction).
    _reranker_lock: asyncio.Lock = asyncio.Lock()

    async def _get_or_build_reranker_async() -> Any:
        """Return the effective reranker: injected takes priority, else lazy-create shared.

        Uses asyncio.Lock to prevent double-initialisation under concurrent requests
        (TOCTOU guard: check-then-build is now atomic within the event loop).
        """
        if _injected_reranker is not None:
            return _injected_reranker
        async with _reranker_lock:
            if not _shared_reranker:
                from anvil_kb.retrieve.rerank import FastEmbedReranker
                _shared_reranker.append(FastEmbedReranker())
        return _shared_reranker[0]

    if retriever is None:
        # Build the rerank variant now (reranker resolved lazily at request time via closure)
        # We build it with a placeholder; actual reranker fetched per-request.
        # Simpler: build it lazily when first rerank=true request arrives.
        _retriever_rerank: Any = None  # sentinel — built lazily below
    else:
        _retriever_rerank = None  # injected retriever path; reranker applied manually

    # H4: parse CORS origins from environment
    # 空串/纯空格的 env 值视为未配置，fallback 到 ["*"]，避免 CORSMiddleware 收到空列表
    cors_origins_env = os.environ.get("ANVIL_KB_CORS_ORIGINS", "*")
    if cors_origins_env == "*":
        cors_origins: list[str] = ["*"]
    else:
        cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
        if not cors_origins:
            # empty string or whitespace-only env → treat as unconfigured → wildcard
            cors_origins = ["*"]

    app = FastAPI(title="anvil-kb-api", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
                detail=f"unsupported file type '{suffix}'; allowed: .md, .txt, .pdf",
            )

        raw = await file.read()

        # H4: enforce upload size limit
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"file too large: {len(raw)} bytes exceeds"
                    f" limit of {MAX_UPLOAD_BYTES} bytes"
                ),
            )

        if suffix == ".pdf":
            # PDF branch: parse binary bytes → markdown text (no UTF-8 decode)
            try:
                text = parse_pdf(raw)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc
        else:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"file content is not valid UTF-8: {exc}",
                ) from exc

        stem = filename.rsplit(".", 1)[0] if "." in filename else filename

        try:
            doc, chunk_count = await ingest_markdown(
                title=stem,
                source_name=filename,
                text=text,
                embedder=embedder,
                store=store,
                sparse_index=sparse_index,  # H4: dual-write to BM25 index
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

        # H4: non-stream + debug is not supported
        if not req.stream and req.debug:
            raise HTTPException(
                status_code=400,
                detail=(
                    "debug=true is only supported with stream=true;"
                    " use stream=true to get debug frames"
                ),
            )

        # R2: resolve the effective retriever for this request.
        # When rerank=true:
        #   - If a retriever was injected (tests), apply the reranker manually post-retrieve.
        #   - If no retriever was injected, use (or lazily build) a Retriever with reranker.
        # When rerank=false:
        #   - Always use _retriever (zero behaviour change).
        async def _effective_retriever_and_reranker():
            """Return (active_retriever, post_reranker_or_None).

            post_reranker_or_None is the reranker to apply manually after retrieve()
            when the retriever itself does not encapsulate reranking.
            """
            if not req.rerank:
                return _retriever, None

            # rerank=true path
            if retriever is None:
                # No injected retriever: build or reuse a Retriever with the reranker.
                # We use a nonlocal-like pattern via the outer _retriever_rerank variable.
                # Since we can't reassign the outer variable directly in Python, we
                # build it lazily using the reranker closure and wrap into a new Retriever.
                nonlocal _retriever_rerank
                if _retriever_rerank is None:
                    assert sparse_index is not None
                    _retriever_rerank = Retriever(
                        embedder,
                        store,
                        sparse_index=sparse_index,
                        mode="hybrid",
                        reranker=await _get_or_build_reranker_async(),
                    )
                return _retriever_rerank, None
            else:
                # Injected retriever — apply the effective reranker manually after retrieve
                return _retriever, await _get_or_build_reranker_async()

        active_retriever, post_reranker = await _effective_retriever_and_reranker()

        if req.stream:
            # --- SSE streaming path ---
            async def _sse_generator():
                try:
                    if req.debug:
                        # H4: debug path — retrieve_debug then use fused as retrieved
                        debug_result = await active_retriever.retrieve_debug(req.question, req.k)
                        fused_candidates = debug_result.fused

                        # R2: apply reranker to fused candidates
                        # Three cases:
                        #   1. rerank=true + post_reranker set → injected-retriever test path:
                        #      apply reranker manually on fused_candidates.
                        #   2. rerank=true + post_reranker=None → production path:
                        #      _retriever_rerank already ran the reranker internally;
                        #      debug_result.reranked holds the result.
                        #   3. rerank=false → no reranking; reranked_chunks stays None.
                        if req.rerank and post_reranker is not None:
                            reranked_chunks = post_reranker.rerank(
                                req.question, fused_candidates, top=req.k
                            )
                        elif req.rerank:
                            # Production path: Retriever internally reranked; take from debug
                            reranked_chunks = debug_result.reranked
                        else:
                            reranked_chunks = None

                        retrieved = (
                            reranked_chunks if reranked_chunks is not None else fused_candidates
                        )

                        # Emit debug frame BEFORE sources
                        yield _sse_event(
                            "debug",
                            _build_debug_payload(debug_result, reranked=reranked_chunks),
                        )
                    else:
                        # Normal path — zero extra overhead
                        raw_retrieved = await active_retriever.retrieve(req.question, req.k)
                        # R2: manual reranker application (injected retriever path)
                        if req.rerank and post_reranker is not None:
                            retrieved = post_reranker.rerank(
                                req.question, raw_retrieved, top=req.k
                            )
                        else:
                            retrieved = raw_retrieved

                    async for event_type, payload in answer_stream(
                        req.question,
                        retrieved,
                        chat=chat,
                    ):
                        if event_type == "sources":
                            # payload is the retrieved list yielded by answer_stream
                            yield _sse_event("sources", _sources_list(payload))
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
                except Exception as exc:  # noqa: BLE001
                    # H4: any exception (retrieval or generation) → error SSE frame
                    detail = str(exc)[:200]
                    yield _sse_event("error", {"detail": detail})
                    # Generator returns normally — StreamingResponse ends cleanly

            return StreamingResponse(
                _sse_generator(),
                media_type="text/event-stream; charset=utf-8",
            )

        else:
            # --- Non-streaming JSON path ---
            raw_retrieved = await active_retriever.retrieve(req.question, req.k)
            if req.rerank and post_reranker is not None:
                retrieved = post_reranker.rerank(req.question, raw_retrieved, top=req.k)
            else:
                retrieved = raw_retrieved
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
