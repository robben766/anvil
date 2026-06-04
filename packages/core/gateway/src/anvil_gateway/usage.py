"""跨 provider 归一化的用量记录(spec §4.3)。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal


@dataclass
class UsageRecord:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost_cny: Decimal
    latency_ms: int
    request_id: str
    ttft_ms: int | None = None
    session_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0
