from decimal import Decimal

from anvil_gateway.ledger import SqliteLedger
from anvil_gateway.usage import UsageRecord


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


def test_insert_and_count(tmp_path):
    ledger = SqliteLedger(str(tmp_path / "l.sqlite3"))
    ledger.insert(_record())
    ledger.insert(_record())
    assert ledger.count() == 2
    ledger.close()


def test_total_cost_decimal_exact(tmp_path):
    ledger = SqliteLedger(str(tmp_path / "l.sqlite3"))
    ledger.insert(_record("0.1"))
    ledger.insert(_record("0.2"))
    assert ledger.total_cost() == Decimal("0.3")  # 文本存储,无浮点误差
    ledger.close()


def test_reopen_persists(tmp_path):
    path = str(tmp_path / "l.sqlite3")
    ledger = SqliteLedger(path)
    ledger.insert(_record())
    ledger.close()
    assert SqliteLedger(path).count() == 1
