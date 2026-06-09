import pytest

pytestmark = pytest.mark.asyncio


async def test_put_then_env_for_roundtrip(session_factory):
    from anvil_ai_employee.mcp.tokens import McpTokenStore

    store = McpTokenStore(session_factory)
    await store.put(employee="alice", connector="gmail", env_key="GMAIL_TOKEN", secret="s3cr3t")
    await store.put(employee="alice", connector="gmail", env_key="GMAIL_USER", secret="alice@x.com")
    env = await store.env_for(employee="alice", connector="gmail")
    assert env == {"GMAIL_TOKEN": "s3cr3t", "GMAIL_USER": "alice@x.com"}


async def test_put_is_upsert(session_factory):
    from anvil_ai_employee.mcp.tokens import McpTokenStore

    store = McpTokenStore(session_factory)
    await store.put(employee="alice", connector="gmail", env_key="GMAIL_TOKEN", secret="old")
    await store.put(employee="alice", connector="gmail", env_key="GMAIL_TOKEN", secret="new")
    env = await store.env_for(employee="alice", connector="gmail")
    assert env == {"GMAIL_TOKEN": "new"}


async def test_env_for_scopes_to_employee_and_connector(session_factory):
    from anvil_ai_employee.mcp.tokens import McpTokenStore

    store = McpTokenStore(session_factory)
    await store.put(employee="alice", connector="gmail", env_key="K", secret="a")
    await store.put(employee="bob", connector="gmail", env_key="K", secret="b")
    await store.put(employee="alice", connector="slack", env_key="K", secret="c")
    assert await store.env_for(employee="alice", connector="gmail") == {"K": "a"}
