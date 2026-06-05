import time

from anvil_obs.span import current_span, span


def test_span_ids_and_timing():
    with span("root") as s:
        time.sleep(0.01)
    assert len(s.trace_id) == 32 and len(s.span_id) == 16  # OTLP hex 规格
    assert s.end_ns > s.start_ns
    assert s.parent_span_id is None


def test_nesting_propagates_trace_and_parent():
    with span("parent") as p:
        with span("child") as c:
            assert current_span() is c
        assert current_span() is p
    assert c.trace_id == p.trace_id
    assert c.parent_span_id == p.span_id


def test_attributes_and_status():
    with span("op") as s:
        s.set_attribute("k", "v")
        s.set_attribute("n", 42)
    assert s.attributes["k"] == "v" and s.attributes["n"] == 42
    assert s.status_ok
    try:
        with span("bad") as s2:
            raise ValueError("boom")
    except ValueError:
        pass
    assert not s2.status_ok and "boom" in s2.status_message


def test_siblings_share_trace_under_same_root():
    with span("root") as r:
        with span("a") as a:
            pass
        with span("b") as b:
            pass
    assert a.trace_id == b.trace_id == r.trace_id
    assert a.parent_span_id == b.parent_span_id == r.span_id


def test_context_restored_after_inner_exception():
    with span("outer") as outer:
        try:
            with span("inner"):
                raise ValueError("boom")
        except ValueError:
            pass
        assert current_span() is outer
    assert current_span() is None


async def test_concurrent_tasks_isolated():
    import asyncio

    async def make_span(name):
        with span(name) as s:
            await asyncio.sleep(0.01)
            assert current_span() is s
        return s

    a, b = await asyncio.gather(make_span("a"), make_span("b"))
    assert a.trace_id != b.trace_id  # 各自独立的根 trace
    assert current_span() is None
