import os

import pytest
import pytest_asyncio


@pytest.fixture(autouse=True)
def _gateway_env(monkeypatch):
    """Worker tests drive the gateway under respx mock — it still needs a key + a
    configured ledger DB. Mirrors packages/code-agent/tests/conftest.py."""
    # Only inject a dummy key when no real one is present — respx-mocked tests never
    # hit the network, but @pytest.mark.live tests need the real key to survive.
    if not os.environ.get("DEEPSEEK_API_KEY"):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "k1")
    from anvil_gateway import configure

    configure(
        database_url=os.environ.get(
            "ANVIL_TEST_DATABASE_URL",
            "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test",
        ),
        retry_base_delay=0,
    )


@pytest_asyncio.fixture
async def engine():
    from anvil_ai_employee.db import Base, make_engine

    url = os.environ.get("ANVIL_DATABASE_URL")
    if not url:
        pytest.skip("ANVIL_DATABASE_URL not set (needs real PG@5434)")
    eng = make_engine(url)
    # ai-employee tables + kb tables (tools read kb_documents)
    from anvil_kb.db import Base as KbBase

    async with eng.begin() as conn:
        await conn.run_sync(KbBase.metadata.drop_all)
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(KbBase.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
