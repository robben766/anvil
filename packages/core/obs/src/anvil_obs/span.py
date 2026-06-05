"""自研 span:trace 树、上下文传播、属性与状态。不依赖 opentelemetry-sdk。"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_current: ContextVar[Span | None] = ContextVar("anvil_obs_current_span", default=None)


def _hex(n_bytes: int) -> str:
    return os.urandom(n_bytes).hex()


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_ns: int
    end_ns: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    status_ok: bool = True
    status_message: str = ""

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


def current_span() -> Span | None:
    return _current.get()


@contextmanager
def span(name: str, **attrs: Any):
    parent = _current.get()
    s = Span(
        name=name,
        trace_id=parent.trace_id if parent else _hex(16),
        span_id=_hex(8),
        parent_span_id=parent.span_id if parent else None,
        start_ns=time.time_ns(),
        attributes=dict(attrs),
    )
    token = _current.set(s)
    try:
        yield s
    except BaseException as e:
        s.status_ok = False
        s.status_message = str(e)
        raise
    finally:
        s.end_ns = time.time_ns()
        _current.reset(token)
        from anvil_obs.exporter import enqueue  # 延迟导入避免环

        enqueue(s)
