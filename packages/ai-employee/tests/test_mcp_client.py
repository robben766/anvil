import sys

import pytest

_SERVER = ["-m", "anvil_ai_employee.mcp.mock_servers.email_server"]


def _client(env=None):
    from anvil_ai_employee.mcp.client import McpClient

    return McpClient(connector="gmail", command=sys.executable, args=_SERVER, env=env or {})


def test_start_handshakes_and_lists_tools():
    c = _client()
    try:
        specs = c.start()
        names = {s.name for s in specs}
        assert names == {"list_events", "send_email"}
        send = next(s for s in specs if s.name == "send_email")
        assert send.annotations["destructiveHint"] is True
        assert "to" in send.input_schema["properties"]
    finally:
        c.close()


def test_start_is_idempotent():
    c = _client()
    try:
        first = c.start()
        second = c.start()
        assert [s.name for s in first] == [s.name for s in second]
    finally:
        c.close()


def test_call_read_tool():
    c = _client()
    try:
        c.start()
        out = c.call_tool("list_events", {"date": "2026-06-09"})
        assert "standup" in out
    finally:
        c.close()


def test_call_tool_without_token_returns_error_text():
    c = _client(env={})
    try:
        c.start()
        out = c.call_tool("send_email", {"to": "a", "subject": "s", "body": "b"})
        assert "[MCP error]" in out and "GMAIL_TOKEN not configured" in out
    finally:
        c.close()


def test_call_tool_redacts_injected_secret():
    c = _client(env={"GMAIL_TOKEN": "SUPERSECRET"})
    try:
        c.start()
        out = c.call_tool("send_email", {"to": "a", "subject": "s", "body": "b"})
        assert "SUPERSECRET" not in out  # redacted
        assert "***" in out
        assert "已发送" in out
    finally:
        c.close()


def test_close_is_idempotent():
    c = _client()
    c.start()
    c.close()
    c.close()  # must not raise


def test_start_failure_tears_down_loop_thread():
    """A server that exits immediately fails the handshake; start() must raise AND
    self-clean (no leaked loop thread) rather than leaving run_forever() hanging."""
    from anvil_ai_employee.mcp.client import McpClient
    from anvil_ai_employee.mcp.transport import McpTransportError

    c = McpClient(
        connector="dead",
        command=sys.executable,
        args=["-c", "import sys; sys.exit(0)"],
        env={},
    )
    with pytest.raises(McpTransportError):
        c.start()
    assert c._thread is None or not c._thread.is_alive()
    c.close()  # still idempotent after a failed start
