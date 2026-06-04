"""SQLite 记账。cost 以 TEXT 存 Decimal 字符串,避免浮点误差。"""

from __future__ import annotations

import sqlite3
import threading
from decimal import Decimal

from anvil_gateway.usage import UsageRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,
    cost_cny TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    ttft_ms INTEGER,
    request_id TEXT,
    session_id TEXT,
    created_at TEXT NOT NULL
)
"""


class SqliteLedger:
    def __init__(self, path: str = "anvil_ledger.sqlite3") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def insert(self, r: UsageRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage_records (provider, model, prompt_tokens, completion_tokens,"
                " cached_tokens, cost_cny, latency_ms, ttft_ms, request_id, session_id, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    r.provider,
                    r.model,
                    r.prompt_tokens,
                    r.completion_tokens,
                    r.cached_tokens,
                    str(r.cost_cny),
                    r.latency_ms,
                    r.ttft_ms,
                    r.request_id,
                    r.session_id,
                    r.created_at.isoformat(),
                ),
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM usage_records").fetchone()[0]

    def total_cost(self) -> Decimal:
        with self._lock:
            rows = self._conn.execute("SELECT cost_cny FROM usage_records").fetchall()
        return sum((Decimal(v) for (v,) in rows), Decimal(0))

    def close(self) -> None:
        self._conn.close()
