"""Test fixtures for anvil-kb-api: DB setup, fake embedder, ASGI client."""

from __future__ import annotations

import os
import pathlib

import httpx
import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from anvil_kb.db import EMBEDDING_DIM

TEST_DB_URL = os.environ.get(
    "ANVIL_TEST_DATABASE_URL", "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test"
)

# Alembic config lives in packages/kb
_KB_ROOT = pathlib.Path(__file__).parent.parent.parent.parent / "packages" / "kb"
_ALEMBIC_INI = str(_KB_ROOT / "alembic.ini")


@pytest.fixture(scope="session", autouse=True)
def _run_kb_migrations():
    """Run kb alembic upgrade head against the test database (session-scoped)."""
    cfg = Config(_ALEMBIC_INI)
    cfg.set_main_option("sqlalchemy.url", TEST_DB_URL)
    cfg.set_main_option("script_location", str(_KB_ROOT / "alembic"))
    command.upgrade(cfg, "head")


async def _truncate_kb_tables(url: str) -> None:
    eng = create_async_engine(url, poolclass=NullPool)
    async with eng.begin() as conn:
        await conn.execute(text("TRUNCATE kb_chunks, kb_documents CASCADE"))
    await eng.dispose()


class FakeEmbedder:
    """One-hot embedder: no model needed, dimension matches EMBEDDING_DIM."""

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._one_hot(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one_hot(text)

    def _one_hot(self, text: str) -> list[float]:
        vec = [0.0] * EMBEDDING_DIM
        if text:
            vec[hash(text) % EMBEDDING_DIM] = 1.0
        return vec


@pytest.fixture(scope="session")
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


async def _make_client(session_factory, embedder) -> httpx.AsyncClient:
    """Create an ASGI test client (caller must close it)."""
    from anvil_kb_api.app import create_app

    app = create_app(session_factory=session_factory, embedder=embedder)
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def api_client(_run_kb_migrations, fake_embedder):
    """Provide a clean ASGI client backed by the test PG database."""
    await _truncate_kb_tables(TEST_DB_URL)

    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    from anvil_kb_api.app import create_app

    app = create_app(session_factory=session_factory, embedder=fake_embedder)

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client

    await engine.dispose()
    await _truncate_kb_tables(TEST_DB_URL)


@pytest.fixture
async def auth_client_no_header(_run_kb_migrations, fake_embedder, monkeypatch):
    """Client where ANVIL_KB_API_KEY='secret-key' is set; no auth header pre-configured."""
    monkeypatch.setenv("ANVIL_KB_API_KEY", "secret-key")
    monkeypatch.setenv("ANVIL_DATABASE_URL", TEST_DB_URL)

    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    from anvil_kb_api.app import create_app

    app = create_app(session_factory=session_factory, embedder=fake_embedder)

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client

    await engine.dispose()


@pytest.fixture
async def auth_client_correct_key(_run_kb_migrations, fake_embedder, monkeypatch):
    """Client where ANVIL_KB_API_KEY='secret-key' and correct header is pre-set."""
    monkeypatch.setenv("ANVIL_KB_API_KEY", "secret-key")
    monkeypatch.setenv("ANVIL_DATABASE_URL", TEST_DB_URL)

    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    from anvil_kb_api.app import create_app

    app = create_app(session_factory=session_factory, embedder=fake_embedder)

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer secret-key"},
    ) as client:
        yield client

    await engine.dispose()
