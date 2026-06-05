import os
import pathlib

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from anvil_kb.db import make_session_factory
from anvil_kb.store.pg import PgVectorStore

TEST_DB_URL = os.environ.get(
    "ANVIL_TEST_DATABASE_URL", "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test"
)

_KB_ROOT = pathlib.Path(__file__).parent.parent
_ALEMBIC_INI = str(_KB_ROOT / "alembic.ini")


@pytest.fixture(scope="session", autouse=True)
def _run_kb_migrations():
    """Run kb alembic upgrade head against the test database (session-scoped)."""
    cfg = Config(_ALEMBIC_INI)
    cfg.set_main_option("sqlalchemy.url", TEST_DB_URL)
    # Resolve script_location to absolute path so pytest can run from any cwd.
    cfg.set_main_option("script_location", str(_KB_ROOT / "alembic"))
    command.upgrade(cfg, "head")


async def _truncate_kb_tables(url: str) -> None:
    """TRUNCATE kb_chunks and kb_documents with CASCADE."""
    eng = create_async_engine(url, poolclass=NullPool)
    async with eng.begin() as conn:
        await conn.execute(text("TRUNCATE kb_chunks, kb_documents CASCADE"))
    await eng.dispose()


@pytest.fixture
async def kb_session(_run_kb_migrations):
    """Provide a clean AsyncSession per test; TRUNCATE tables before and after.

    pgvector.sqlalchemy.Vector uses the bind_processor (text protocol) which
    asyncpg handles natively — no register_vector listener required when using
    the ORM layer.
    """
    # Pre-test cleanup ensures isolation even when prior runs left data.
    await _truncate_kb_tables(TEST_DB_URL)

    engine = create_async_engine(TEST_DB_URL, echo=False, poolclass=NullPool)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session

    await engine.dispose()

    # Post-test cleanup.
    await _truncate_kb_tables(TEST_DB_URL)


@pytest.fixture
async def kb_store(_run_kb_migrations):
    """Provide a PgVectorStore backed by the test database; tables are truncated before/after."""
    await _truncate_kb_tables(TEST_DB_URL)

    session_factory = make_session_factory(TEST_DB_URL)
    store = PgVectorStore(session_factory)
    yield store

    await _truncate_kb_tables(TEST_DB_URL)
