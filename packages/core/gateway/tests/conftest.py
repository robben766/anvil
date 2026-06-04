import os

import pytest

from anvil_gateway.ledger import PostgresLedger

TEST_DB_URL = os.environ.get(
    "ANVIL_TEST_DATABASE_URL", "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test"
)


@pytest.fixture
async def pg_ledger():
    """提供已建表、已清空的 ledger;用例间互不污染。"""
    ledger = PostgresLedger(TEST_DB_URL)
    await ledger.init_schema()
    await ledger.clear()
    yield ledger
    await ledger.close()
