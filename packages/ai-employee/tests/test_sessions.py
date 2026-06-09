import pytest
from anvil_ai_employee.sessions import SessionStore

pytestmark = pytest.mark.asyncio


async def test_create_save_load_roundtrip(session_factory):
    ss = SessionStore(session_factory)
    sid = await ss.create(employee="assistant")
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    await ss.save(sid, msgs, status="active")
    loaded = await ss.load(sid)
    assert loaded == tuple(msgs)


async def test_load_missing_returns_empty_tuple(session_factory):
    import uuid
    ss = SessionStore(session_factory)
    assert await ss.load(uuid.uuid4()) == ()
