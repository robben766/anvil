"""OTLP protobuf exporter: batching, background flush, fail-open."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import TYPE_CHECKING

import httpx
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.proto.common.v1.common_pb2 import (
    AnyValue,
    InstrumentationScope,
    KeyValue,
)
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import (
    ResourceSpans,
    ScopeSpans,
    Status,
)
from opentelemetry.proto.trace.v1.trace_pb2 import Span as ProtoSpan

if TYPE_CHECKING:
    from anvil_obs.span import Span

_log = logging.getLogger("anvil_obs")

# Module-level state
_active: OtlpExporter | None = None


def enqueue(s: Span) -> None:  # noqa: ANN001
    """Append span to the active exporter batch; silently discard if no exporter."""
    try:
        if _active is not None:
            _active._batch.append(s)
    except Exception:  # noqa: BLE001
        pass


def drain_for_test() -> list[Span]:
    """Test helper: take and return the current batch without flushing."""
    if _active is None:
        return []
    batch = list(_active._batch)
    _active._batch.clear()
    return batch


def _span_to_proto(s: Span) -> ProtoSpan:
    """Serialize an anvil Span to a protobuf ProtoSpan."""
    # Convert hex ids to bytes
    trace_id_bytes = bytes.fromhex(s.trace_id)
    span_id_bytes = bytes.fromhex(s.span_id)
    parent_span_id_bytes = bytes.fromhex(s.parent_span_id) if s.parent_span_id else b""

    # Build attributes — IMPORTANT: check bool before int (bool is a subclass of int)
    attrs = []
    for key, value in s.attributes.items():
        if isinstance(value, bool):
            av = AnyValue(bool_value=value)
        elif isinstance(value, int):
            av = AnyValue(int_value=value)
        elif isinstance(value, float):
            av = AnyValue(double_value=value)
        elif isinstance(value, str):
            av = AnyValue(string_value=value)
        else:
            av = AnyValue(string_value=str(value))
        attrs.append(KeyValue(key=key, value=av))

    # Build status
    if s.status_ok:
        status = Status(code=Status.STATUS_CODE_OK)
    else:
        status = Status(code=Status.STATUS_CODE_ERROR, message=s.status_message)

    return ProtoSpan(
        trace_id=trace_id_bytes,
        span_id=span_id_bytes,
        parent_span_id=parent_span_id_bytes,
        name=s.name,
        start_time_unix_nano=s.start_ns,
        end_time_unix_nano=s.end_ns,
        attributes=attrs,
        status=status,
    )


def _build_request(batch: list[Span]) -> bytes:
    """Serialize a list of Spans into a serialized ExportTraceServiceRequest."""
    proto_spans = [_span_to_proto(s) for s in batch]

    resource = Resource(
        attributes=[
            KeyValue(key="service.name", value=AnyValue(string_value="anvil")),
        ]
    )
    scope = InstrumentationScope(name="anvil-obs")
    scope_spans = ScopeSpans(scope=scope, spans=proto_spans)
    resource_spans = ResourceSpans(resource=resource, scope_spans=[scope_spans])
    req = ExportTraceServiceRequest(resource_spans=[resource_spans])
    return req.SerializeToString()


class OtlpExporter:
    """Batching OTLP/protobuf exporter with background flush and fail-open semantics.

    Module-level singleton: constructing a new instance makes it the active exporter —
    any subsequently completed spans are routed to it.  The previously active instance
    (if any) is displaced but **not** closed; callers that hold a reference to the old
    instance must still call ``await old_instance.aclose()`` to stop its background
    flush task.
    """

    def __init__(
        self,
        endpoint: str,
        public_key: str,
        secret_key: str,
        flush_interval: float = 5.0,
    ) -> None:
        self._endpoint = endpoint
        self._auth = "Basic " + base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self._flush_interval = flush_interval
        self._batch: list[Span] = []
        self._bg_task: asyncio.Task | None = None

        # Activate as the global exporter
        global _active
        _active = self

        # Start background flush task if an event loop is running
        try:
            loop = asyncio.get_running_loop()
            if loop is not None:
                self._bg_task = loop.create_task(self._background_loop())
        except RuntimeError:
            # No running event loop; rely on explicit flush() calls
            pass

    async def _background_loop(self) -> None:
        """Periodically flush the batch."""
        try:
            while True:
                await asyncio.sleep(self._flush_interval)
                await self.flush()
        except asyncio.CancelledError:
            pass

    async def flush(self) -> None:
        """Serialize the current batch and POST to the OTLP endpoint."""
        # Take the current batch atomically
        batch = list(self._batch)
        self._batch.clear()

        if not batch:
            return

        try:
            body = _build_request(batch)
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    self._endpoint,
                    content=body,
                    headers={
                        "Content-Type": "application/x-protobuf",
                        "Authorization": self._auth,
                    },
                )
            if resp.status_code < 200 or resp.status_code >= 300:
                _log.warning(
                    "anvil_obs: OTLP export got non-2xx status %d; %d spans dropped",
                    resp.status_code,
                    len(batch),
                )
        except Exception:  # noqa: BLE001
            _log.warning(
                "anvil_obs: OTLP export failed; %d spans dropped",
                len(batch),
                exc_info=True,
            )

    async def aclose(self) -> None:
        """Final flush, cancel background task, and deactivate."""
        global _active

        try:
            # Cancel background task first
            if self._bg_task is not None and not self._bg_task.done():
                self._bg_task.cancel()
                try:
                    await self._bg_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
                self._bg_task = None

            # Final flush — CancelledError is absorbed so aclose() never propagates it
            try:
                await self.flush()
            except asyncio.CancelledError:
                pass
        finally:
            # Deactivate regardless of cancellation
            if _active is self:
                _active = None
