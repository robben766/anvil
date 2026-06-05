import base64

import httpx
import respx
from anvil_obs.exporter import OtlpExporter
from anvil_obs.span import span
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)

OTLP = "http://localhost:3100/api/public/otel/v1/traces"


@respx.mock
async def test_export_batch_protobuf_roundtrip():
    captured = {}

    def handler(request):
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = request.content
        return httpx.Response(200)

    respx.post(OTLP).mock(side_effect=handler)
    exp = OtlpExporter(endpoint=OTLP, public_key="pk", secret_key="sk", flush_interval=0.05)
    with span("root"):
        with span("child"):
            pass
    await exp.flush()
    req = ExportTraceServiceRequest()
    req.ParseFromString(captured["body"])
    spans = req.resource_spans[0].scope_spans[0].spans
    assert {s.name for s in spans} == {"root", "child"}
    root = next(s for s in spans if s.name == "root")
    child = next(s for s in spans if s.name == "child")
    assert child.trace_id == root.trace_id and child.parent_span_id == root.span_id
    expected = "Basic " + base64.b64encode(b"pk:sk").decode()
    assert captured["auth"] == expected


@respx.mock
async def test_export_failure_never_raises(caplog):
    respx.post(OTLP).mock(return_value=httpx.Response(500))
    exp = OtlpExporter(endpoint=OTLP, public_key="pk", secret_key="sk")
    with span("x"):
        pass
    await exp.flush()  # 不抛异常即通过


def test_enqueue_without_exporter_is_noop():
    """enqueue when no active exporter must never raise."""
    with span("noop"):
        pass  # enqueue is called internally; must not raise


@respx.mock
async def test_aclose_swallows_cancelled_error():
    """aclose() must not propagate CancelledError even if flush is cancelled mid-flight."""
    import asyncio

    import anvil_obs.exporter as exp_mod

    # Use a real async side_effect that raises CancelledError to simulate
    # the task being cancelled while flush's HTTP call is in progress.
    async def slow_then_cancel(request):
        raise asyncio.CancelledError()

    respx.post(OTLP).mock(side_effect=slow_then_cancel)
    exp = OtlpExporter(endpoint=OTLP, public_key="pk", secret_key="sk", flush_interval=9999)
    with span("x"):
        pass

    # aclose() must not raise even though flush() will encounter a CancelledError
    await exp.aclose()  # 不抛即通过

    # _active must be cleared regardless
    assert exp_mod._active is None
