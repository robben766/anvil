import os
from decimal import Decimal

from anvil_gateway.ledger import PostgresLedger
from anvil_gateway.usage import UsageRecord

TEST_DB_URL = os.environ.get(
    "ANVIL_TEST_DATABASE_URL", "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test"
)


def _record(cost="0.5"):
    return UsageRecord(
        provider="deepseek",
        model="deepseek-chat",
        prompt_tokens=100,
        completion_tokens=50,
        cached_tokens=30,
        cost_cny=Decimal(cost),
        latency_ms=800,
        request_id="req-1",
        session_id="s-1",
    )


async def test_insert_and_count(pg_ledger):
    await pg_ledger.insert(_record())
    await pg_ledger.insert(_record())
    assert await pg_ledger.count() == 2


async def test_total_cost_decimal_exact(pg_ledger):
    await pg_ledger.insert(_record("0.1"))
    await pg_ledger.insert(_record("0.2"))
    assert await pg_ledger.total_cost() == Decimal("0.3")  # NUMERIC 精确,无浮点误差


async def test_persists_across_connections(pg_ledger):
    await pg_ledger.insert(_record())
    other = PostgresLedger(TEST_DB_URL)
    assert await other.count() == 1
    await other.close()
