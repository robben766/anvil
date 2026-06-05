"""anvil-kb-api: FastAPI CRUD shell for the anvil-kb RAG pipeline.

Endpoints (prefix /v1/kb):
  POST   /v1/kb/documents      — upload .md/.txt, ingest, return 201
  GET    /v1/kb/documents      — list all documents with chunk counts
  GET    /v1/kb/documents/{id} — get document detail including content
  DELETE /v1/kb/documents/{id} — delete document (204, idempotent)

Auth: set ANVIL_KB_API_KEY to require Bearer token on all endpoints.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_kb.db import ChunkRow, DocumentRow, make_engine
from anvil_kb.embed import Embedder, FastEmbedEmbedder
from anvil_kb.ingest.pipeline import ingest_markdown
from anvil_kb.store.pg import PgVectorStore

_ALLOWED_EXTENSIONS = {".md", ".txt"}


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
) -> FastAPI:
    """Factory for the FastAPI application (DI entry point for tests).

    session_factory=None → lazy-build from ANVIL_DATABASE_URL env var.
    embedder=None        → FastEmbedEmbedder() (lazy local model).
    """
    # Resolve dependencies
    if session_factory is None:
        engine = make_engine()  # reads ANVIL_DATABASE_URL
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    if embedder is None:
        embedder = FastEmbedEmbedder()

    store = PgVectorStore(session_factory)

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

    return app


def _serialize_dt(dt: datetime) -> str:
    """Serialize a timezone-aware datetime to ISO-8601 string."""
    return dt.isoformat()


def run() -> None:
    """Entry point: uvicorn on 0.0.0.0:8400."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8400)
