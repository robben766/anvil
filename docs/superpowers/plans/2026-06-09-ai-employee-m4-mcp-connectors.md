# P4-M4 MCP Connectors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the AI employee call external MCP-server tools through a self-built JSON-RPC 2.0 stdio client, with credentials held server-side (agent never sees them) and high-risk MCP actions auto-routed through the M3 Agent Inbox.

**Architecture:** A `mcp/` subpackage in `anvil_ai_employee`: `transport.py` (asyncio subprocess + newline-delimited JSON-RPC framing), `client.py` (`McpClient` owning a background event-loop thread for a persistent session: initialize handshake → tools/list → tools/call), `tokens.py` (`McpTokenStore` over a new `ae_mcp_tokens` table; secrets injected into the server's env at spawn), `adapter.py` (wrap each MCP tool as a P3 `Tool` with namespaced names + risk mapping), `connector.py` (`ConnectorConfig` + `build_mcp_registry` + `mcp_risk_policy`), and `mock_servers/email_server.py` (a zero-dependency stdio MCP server for real integration tests). HITL/inbox/memory from M3 are reused unchanged — MCP high-risk tools route through them automatically.

**Tech Stack:** Python 3.12, asyncio subprocess, JSON-RPC 2.0, SQLAlchemy 2 + asyncpg + PG@5434, pytest/pytest-asyncio. Reuses P3 `tools/base.py` + `permission.py`, M3 `hitl.py`/`inbox.py`/`inbox_resume.py`/`hitl_memory.py`, M1 worker/cli, M2a `MemoryStore`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/anvil_ai_employee/db.py` (modify) | Add `McpTokenRow` (`ae_mcp_tokens`) table |
| `src/anvil_ai_employee/mcp/__init__.py` (create) | Package marker |
| `src/anvil_ai_employee/mcp/tokens.py` (create) | `McpTokenStore.put/env_for` |
| `src/anvil_ai_employee/mcp/transport.py` (create) | `StdioTransport`, `McpTransportError` |
| `src/anvil_ai_employee/mcp/client.py` (create) | `McpClient`, `McpToolSpec`, `McpProtocolError`, `PROTOCOL_VERSION` |
| `src/anvil_ai_employee/mcp/adapter.py` (create) | `mcp_risk`, `mcp_tools` |
| `src/anvil_ai_employee/mcp/connector.py` (create) | `ConnectorConfig`, `mcp_risk_policy`, `build_mcp_registry` |
| `src/anvil_ai_employee/mcp/mock_servers/__init__.py` (create) | Package marker |
| `src/anvil_ai_employee/mcp/mock_servers/email_server.py` (create) | Mock stdio MCP server (named `email_server` to avoid shadowing stdlib `email`) |
| `src/anvil_ai_employee/cli.py` (modify) | `mcp list-tools/put-token` + `run-mcp` subcommands |
| `examples/11-ai-employee-mcp/README.md` (create) | Milestone walkthrough |
| `CLAUDE.md` (modify) | M4 subsection under anvil-ai-employee |

**Test files:** `tests/test_mcp_tokens.py`, `tests/test_mcp_transport.py`, `tests/test_mcp_mock_server.py`, `tests/test_mcp_client.py`, `tests/test_mcp_adapter.py`, `tests/test_mcp_connector.py`, `tests/test_ai_employee_cli.py` (extend).

**Verification rule (M1–M3 lesson):** after every task run the WHOLE repo `uv run pytest -m "not live"` from `packages/ai-employee` AND once across the repo root to catch test-file-name / fixture collisions. Live tests are opt-in (`-m live`) and must never run in CI.

---

### Task 1: `ae_mcp_tokens` table + `McpTokenStore`

**Files:**
- Modify: `packages/ai-employee/src/anvil_ai_employee/db.py` (add `McpTokenRow` after `InboxRow`)
- Create: `packages/ai-employee/src/anvil_ai_employee/mcp/__init__.py`
- Create: `packages/ai-employee/src/anvil_ai_employee/mcp/tokens.py`
- Test: `packages/ai-employee/tests/test_mcp_tokens.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/ai-employee/tests/test_mcp_tokens.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/ai-employee && uv run pytest tests/test_mcp_tokens.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'anvil_ai_employee.mcp'`

- [ ] **Step 3: Add the table to `db.py`**

Add this class at the end of `packages/ai-employee/src/anvil_ai_employee/db.py` (after `InboxRow`):

```python
class McpTokenRow(Base):
    __tablename__ = "ae_mcp_tokens"

    # Server-side credential custody: the agent never sees these. McpClient injects them
    # into the MCP server subprocess env at spawn; tool-call args never carry them.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee: Mapped[str] = mapped_column(Text, nullable=False)
    connector: Mapped[str] = mapped_column(Text, nullable=False)
    env_key: Mapped[str] = mapped_column(Text, nullable=False)
    secret: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    __table_args__ = (
        UniqueConstraint("employee", "connector", "env_key", name="uq_mcp_token"),
    )
```

- [ ] **Step 4: Create the `mcp` package marker**

```python
# packages/ai-employee/src/anvil_ai_employee/mcp/__init__.py
"""MCP (Model Context Protocol) connectors: self-built JSON-RPC 2.0 stdio client,
server-side token custody, MCP tools as P3 @tool routed through M3 HITL."""
```

- [ ] **Step 5: Implement `McpTokenStore`**

```python
# packages/ai-employee/src/anvil_ai_employee/mcp/tokens.py
"""Server-side credential custody for MCP connectors. Secrets are stored per
(employee, connector, env_key) and handed to the MCP server subprocess as env vars at
spawn — the agent's tool-call arguments never carry them."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import McpTokenRow


class McpTokenStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._sf = session_factory

    async def put(self, *, employee: str, connector: str, env_key: str, secret: str) -> None:
        async with self._sf() as s:
            existing = (
                await s.execute(
                    select(McpTokenRow).where(
                        McpTokenRow.employee == employee,
                        McpTokenRow.connector == connector,
                        McpTokenRow.env_key == env_key,
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                existing.secret = secret
            else:
                s.add(
                    McpTokenRow(
                        employee=employee, connector=connector, env_key=env_key, secret=secret
                    )
                )
            await s.commit()

    async def env_for(self, *, employee: str, connector: str) -> dict[str, str]:
        async with self._sf() as s:
            rows = (
                (
                    await s.execute(
                        select(McpTokenRow).where(
                            McpTokenRow.employee == employee,
                            McpTokenRow.connector == connector,
                        )
                    )
                )
                .scalars()
                .all()
            )
            return {r.env_key: r.secret for r in rows}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd packages/ai-employee && uv run pytest tests/test_mcp_tokens.py -v`
Expected: 3 passed

- [ ] **Step 7: Run ruff + whole-package suite**

Run: `cd packages/ai-employee && uv run ruff check --fix src tests && uv run pytest -m "not live" -q`
Expected: ruff clean; all tests pass (no collisions with existing suite)

- [ ] **Step 8: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/db.py \
        packages/ai-employee/src/anvil_ai_employee/mcp/__init__.py \
        packages/ai-employee/src/anvil_ai_employee/mcp/tokens.py \
        packages/ai-employee/tests/test_mcp_tokens.py
git commit -m "feat(ai-employee): ae_mcp_tokens table + McpTokenStore (server-side credential custody)"
```

---

### Task 2: `StdioTransport` (subprocess + newline-delimited JSON-RPC frames)

**Files:**
- Create: `packages/ai-employee/src/anvil_ai_employee/mcp/transport.py`
- Test: `packages/ai-employee/tests/test_mcp_transport.py`

- [ ] **Step 1: Write the failing test**

The test spawns a tiny inline Python echo server that reads one JSON line and writes back `{"echo": <msg>}`, plus a server that exits immediately to exercise EOF.

```python
# packages/ai-employee/tests/test_mcp_transport.py
import sys

import pytest

pytestmark = pytest.mark.asyncio

_ECHO = (
    "import sys, json\n"
    "for line in sys.stdin:\n"
    "    line = line.strip()\n"
    "    if not line:\n"
    "        continue\n"
    "    msg = json.loads(line)\n"
    "    sys.stdout.write(json.dumps({'echo': msg}) + '\\n')\n"
    "    sys.stdout.flush()\n"
)


async def test_send_receive_roundtrip():
    from anvil_ai_employee.mcp.transport import StdioTransport

    t = StdioTransport(command=sys.executable, args=["-c", _ECHO], env={})
    await t.start()
    try:
        await t.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        reply = await t.receive()
        assert reply == {"echo": {"jsonrpc": "2.0", "id": 1, "method": "ping"}}
    finally:
        await t.close()


async def test_env_is_passed_to_subprocess():
    from anvil_ai_employee.mcp.transport import StdioTransport

    prog = (
        "import sys, os, json\n"
        "sys.stdout.write(json.dumps({'tok': os.environ.get('MY_TOKEN')}) + '\\n')\n"
        "sys.stdout.flush()\n"
    )
    t = StdioTransport(command=sys.executable, args=["-c", prog], env={"MY_TOKEN": "abc"})
    await t.start()
    try:
        assert await t.receive() == {"tok": "abc"}
    finally:
        await t.close()


async def test_receive_on_eof_raises():
    from anvil_ai_employee.mcp.transport import McpTransportError, StdioTransport

    t = StdioTransport(command=sys.executable, args=["-c", "import sys; sys.exit(0)"], env={})
    await t.start()
    try:
        with pytest.raises(McpTransportError):
            await t.receive()
    finally:
        await t.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/ai-employee && uv run pytest tests/test_mcp_transport.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'anvil_ai_employee.mcp.transport'`

- [ ] **Step 3: Implement `StdioTransport`**

```python
# packages/ai-employee/src/anvil_ai_employee/mcp/transport.py
"""MCP stdio transport: spawn the server as a subprocess and exchange
newline-delimited JSON-RPC 2.0 messages over its stdin/stdout. (MCP stdio framing is
one JSON object per line — NOT LSP-style Content-Length headers.)"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any


class McpTransportError(RuntimeError):
    pass


class StdioTransport:
    def __init__(self, *, command: str, args: list[str], env: dict[str, str]):
        self._command = command
        self._args = list(args)
        self._env = dict(env)
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **self._env},
        )

    async def send(self, msg: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(msg) + "\n").encode())
        await self._proc.stdin.drain()

    async def receive(self) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdout is not None
        line = await self._proc.stdout.readline()
        if not line:
            stderr = b""
            if self._proc.stderr is not None:
                try:
                    stderr = await self._proc.stderr.read()
                except Exception:  # noqa: BLE001
                    stderr = b""
            raise McpTransportError(
                f"MCP server closed stdout (EOF). stderr: {stderr.decode(errors='replace')}"
            )
        return json.loads(line)

    async def close(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        try:
            if proc.stdin is not None and not proc.stdin.is_closing():
                proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/ai-employee && uv run pytest tests/test_mcp_transport.py -v`
Expected: 3 passed

- [ ] **Step 5: Run ruff + whole-package suite**

Run: `cd packages/ai-employee && uv run ruff check --fix src tests && uv run pytest -m "not live" -q`
Expected: ruff clean; all tests pass

- [ ] **Step 6: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/mcp/transport.py \
        packages/ai-employee/tests/test_mcp_transport.py
git commit -m "feat(ai-employee): MCP stdio transport (newline-delimited JSON-RPC 2.0)"
```

---

### Task 3: Mock stdio MCP server (`email_server.py`)

**Files:**
- Create: `packages/ai-employee/src/anvil_ai_employee/mcp/mock_servers/__init__.py`
- Create: `packages/ai-employee/src/anvil_ai_employee/mcp/mock_servers/email_server.py`
- Test: `packages/ai-employee/tests/test_mcp_mock_server.py`

This server is both a demo connector AND the integration fixture for Tasks 4/6. It is spec-faithful so it also reverse-validates the client handshake.

- [ ] **Step 1: Write the failing test**

The test drives the server directly over a subprocess pipe (no client yet), feeding the full `initialize → initialized → tools/list → tools/call` sequence.

```python
# packages/ai-employee/tests/test_mcp_mock_server.py
import json
import subprocess
import sys


def _run_sequence(messages, env=None):
    """Feed newline-delimited JSON messages to the mock server, collect its responses."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "anvil_ai_employee.mcp.mock_servers.email_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    payload = "".join(json.dumps(m) + "\n" for m in messages)
    out, _ = proc.communicate(payload, timeout=10)
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def test_initialize_and_tools_list():
    replies = _run_sequence(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ]
    )
    # initialized is a notification → no reply; so 2 replies for ids 1 and 2.
    assert replies[0]["id"] == 1
    assert replies[0]["result"]["protocolVersion"]
    tools = {t["name"]: t for t in replies[1]["result"]["tools"]}
    assert set(tools) == {"list_events", "send_email"}
    assert tools["list_events"]["annotations"]["readOnlyHint"] is True
    assert tools["send_email"]["annotations"]["destructiveHint"] is True


def test_send_email_requires_token():
    import os

    base = dict(os.environ)
    base.pop("GMAIL_TOKEN", None)
    replies = _run_sequence(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "send_email", "arguments": {"to": "x", "subject": "s", "body": "b"}},
            }
        ],
        env=base,
    )
    assert replies[0]["result"]["isError"] is True


def test_send_email_with_token_echoes_redactable_token():
    import os

    base = dict(os.environ)
    base["GMAIL_TOKEN"] = "TOKVAL"
    replies = _run_sequence(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "send_email", "arguments": {"to": "x", "subject": "s", "body": "b"}},
            }
        ],
        env=base,
    )
    text = replies[0]["result"]["content"][0]["text"]
    assert "TOKVAL" in text  # client-side redaction (Task 4) will scrub this


def test_unknown_method_is_jsonrpc_error():
    replies = _run_sequence([{"jsonrpc": "2.0", "id": 9, "method": "no/such"}])
    assert replies[0]["error"]["code"] == -32601
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/ai-employee && uv run pytest tests/test_mcp_mock_server.py -v`
Expected: FAIL — module `anvil_ai_employee.mcp.mock_servers.email_server` not found

- [ ] **Step 3: Create the package marker**

```python
# packages/ai-employee/src/anvil_ai_employee/mcp/mock_servers/__init__.py
"""Zero-dependency stdio MCP servers used for demos and integration tests."""
```

- [ ] **Step 4: Implement the mock server**

```python
# packages/ai-employee/src/anvil_ai_employee/mcp/mock_servers/email_server.py
"""A minimal, zero-dependency stdio MCP server (email/calendar). Spec-faithful enough to
reverse-validate the self-built client: implements initialize / notifications/initialized
/ tools/list / tools/call over newline-delimited JSON-RPC 2.0.

send_email reads GMAIL_TOKEN from the env (proving server-side credential injection) and
echoes it so the client's redaction can be verified end to end. Run as:
    python -m anvil_ai_employee.mcp.mock_servers.email_server
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

PROTOCOL_VERSION = "2025-06-18"

TOOLS = [
    {
        "name": "list_events",
        "description": "List calendar events for a date.",
        "inputSchema": {
            "type": "object",
            "properties": {"date": {"type": "string"}},
            "required": ["date"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
        "annotations": {"destructiveHint": True},
    },
]


def _ok(rid: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def _text(s: str, *, is_error: bool = False) -> dict:
    out: dict = {"content": [{"type": "text", "text": s}]}
    if is_error:
        out["isError"] = True
    return out


def handle(msg: dict) -> dict | None:
    method = msg.get("method")
    rid = msg.get("id")
    if method == "initialize":
        return _ok(
            rid,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock-email", "version": "0.1.0"},
            },
        )
    if method == "notifications/initialized":
        return None  # notification: no response
    if method == "tools/list":
        return _ok(rid, {"tools": TOOLS})
    if method == "tools/call":
        params = msg.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "list_events":
            return _ok(rid, _text(f"{args.get('date')}: 10:00 standup; 14:00 design review"))
        if name == "send_email":
            token = os.environ.get("GMAIL_TOKEN")
            if not token:
                return _ok(rid, _text("GMAIL_TOKEN not configured", is_error=True))
            return _ok(
                rid,
                _text(f"已发送 to={args.get('to')} subject={args.get('subject')} (via token={token})"),
            )
        return _ok(rid, _text(f"unknown tool: {name}", is_error=True))
    return _err(rid, -32601, f"method not found: {method}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd packages/ai-employee && uv run pytest tests/test_mcp_mock_server.py -v`
Expected: 4 passed

- [ ] **Step 6: Run ruff + whole-package suite**

Run: `cd packages/ai-employee && uv run ruff check --fix src tests && uv run pytest -m "not live" -q`
Expected: ruff clean; all tests pass

- [ ] **Step 7: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/mcp/mock_servers/ \
        packages/ai-employee/tests/test_mcp_mock_server.py
git commit -m "feat(ai-employee): zero-dependency mock stdio MCP server (email/calendar)"
```

---

### Task 4: `McpClient` — persistent session on a background loop thread

**Files:**
- Create: `packages/ai-employee/src/anvil_ai_employee/mcp/client.py`
- Test: `packages/ai-employee/tests/test_mcp_client.py`

The client owns a background thread running a dedicated asyncio loop; the subprocess transport lives entirely on that loop. Synchronous callers (`@tool` fns) submit coroutines via `run_coroutine_threadsafe(...).result()`. The client also redacts its own injected secrets from tool output.

- [ ] **Step 1: Write the failing test (integration against the real mock server)**

```python
# packages/ai-employee/tests/test_mcp_client.py
import sys

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/ai-employee && uv run pytest tests/test_mcp_client.py -v`
Expected: FAIL — `anvil_ai_employee.mcp.client` not found

- [ ] **Step 3: Implement `McpClient`**

```python
# packages/ai-employee/src/anvil_ai_employee/mcp/client.py
"""Self-built MCP client over stdio JSON-RPC 2.0. The MCP session is long-lived
(initialize once, call many times), so the subprocess transport lives on a dedicated
background event-loop thread. Synchronous @tool callers submit work via
run_coroutine_threadsafe(...).result(). Injected credentials are redacted from output."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future as ThreadFuture
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any

from anvil_ai_employee.mcp.transport import McpTransportError, StdioTransport

PROTOCOL_VERSION = "2025-06-18"


class McpProtocolError(RuntimeError):
    pass


@dataclass
class McpToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] = field(default_factory=dict)


def _flatten_content(content: list[dict]) -> str:
    parts = [c.get("text", "") for c in content if c.get("type") == "text"]
    if parts:
        return "\n".join(parts)
    return "" if not content else str(content)


class McpClient:
    def __init__(self, *, connector: str, command: str, args: list[str], env: dict[str, str]):
        self.connector = connector
        self._command = command
        self._args = list(args)
        self._env = dict(env)
        self._redact = [v for v in env.values() if v]
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._transport: StdioTransport | None = None
        self._reader: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._specs: list[McpToolSpec] | None = None

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> list[McpToolSpec]:
        if self._specs is not None:
            return self._specs
        ready: ThreadFuture = ThreadFuture()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            ready.set_result(True)
            loop.run_forever()

        self._thread = threading.Thread(target=_run, name=f"mcp-{self.connector}", daemon=True)
        self._thread.start()
        ready.result(timeout=10)
        self._specs = asyncio.run_coroutine_threadsafe(self._async_start(), self._loop).result(
            timeout=30
        )
        return self._specs

    async def _async_start(self) -> list[McpToolSpec]:
        self._transport = StdioTransport(command=self._command, args=self._args, env=self._env)
        await self._transport.start()
        self._reader = asyncio.ensure_future(self._read_loop())
        await self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "anvil-ai-employee", "version": "0.1.0"},
            },
        )
        await self._transport.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        result = await self._request("tools/list")
        return [
            McpToolSpec(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get(
                    "inputSchema", {"type": "object", "properties": {}, "required": []}
                ),
                annotations=t.get("annotations", {}) or {},
            )
            for t in result.get("tools", [])
        ]

    async def _read_loop(self) -> None:
        try:
            while True:
                msg = await self._transport.receive()
                mid = msg.get("id")
                if mid is not None and mid in self._pending:
                    fut = self._pending.pop(mid)
                    if not fut.done():
                        fut.set_result(msg)
                # notifications (no id) are ignored
        except (McpTransportError, asyncio.CancelledError) as e:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(
                        e if isinstance(e, Exception) else McpTransportError("reader stopped")
                    )
            self._pending.clear()

    async def _request(self, method: str, params: dict | None = None) -> dict:
        assert self._loop is not None and self._transport is not None
        self._next_id += 1
        mid = self._next_id
        fut: asyncio.Future = self._loop.create_future()
        self._pending[mid] = fut
        await self._transport.send(
            {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}
        )
        resp = await fut
        if "error" in resp:
            raise McpProtocolError(str(resp["error"]))
        return resp.get("result", {})

    # -- tool calls ----------------------------------------------------------------

    def call_tool(self, name: str, arguments: dict, *, timeout: float = 60.0) -> str:
        if self._loop is None:
            return "[MCP error] client not started"

        async def _call() -> str:
            result = await self._request("tools/call", {"name": name, "arguments": arguments})
            text = _flatten_content(result.get("content", []))
            if result.get("isError"):
                return f"[MCP error] {text}"
            return text

        try:
            text = asyncio.run_coroutine_threadsafe(_call(), self._loop).result(timeout=timeout)
        except FutureTimeoutError:
            return f"[MCP timeout] tool {name} exceeded {timeout}s"
        except McpProtocolError as e:
            return f"[MCP error] {e}"
        except McpTransportError as e:
            return f"[MCP error] transport: {e}"
        return self._redact_secrets(text)

    def _redact_secrets(self, text: str) -> str:
        for secret in self._redact:
            if secret and secret in text:
                text = text.replace(secret, "***")
        return text

    def close(self) -> None:
        if self._loop is None:
            return
        loop = self._loop

        async def _shutdown() -> None:
            if self._reader is not None:
                self._reader.cancel()
            if self._transport is not None:
                await self._transport.close()

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), loop).result(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None
        self._transport = None
        self._reader = None
        self._specs = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/ai-employee && uv run pytest tests/test_mcp_client.py -v`
Expected: 6 passed

- [ ] **Step 5: Run ruff + whole-package suite**

Run: `cd packages/ai-employee && uv run ruff check --fix src tests && uv run pytest -m "not live" -q`
Expected: ruff clean; all tests pass

- [ ] **Step 6: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/mcp/client.py \
        packages/ai-employee/tests/test_mcp_client.py
git commit -m "feat(ai-employee): McpClient — persistent stdio session on background loop thread, secret redaction"
```

---

### Task 5: Adapter — `mcp_risk` + `mcp_tools`

**Files:**
- Create: `packages/ai-employee/src/anvil_ai_employee/mcp/adapter.py`
- Test: `packages/ai-employee/tests/test_mcp_adapter.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/ai-employee/tests/test_mcp_adapter.py
from anvil_ai_employee.mcp.client import McpToolSpec


def _spec(name, annotations, props=None):
    return McpToolSpec(
        name=name,
        description=f"{name} desc",
        input_schema={"type": "object", "properties": props or {}, "required": []},
        annotations=annotations,
    )


def test_mcp_risk_mapping():
    from anvil_ai_employee.mcp.adapter import mcp_risk

    assert mcp_risk(_spec("read", {"readOnlyHint": True})) == "low"
    assert mcp_risk(_spec("send", {"destructiveHint": True})) == "high"
    assert mcp_risk(_spec("none", {})) == "high"  # no annotations → conservative
    assert mcp_risk(_spec("idem", {"idempotentHint": True})) == "medium"


class _FakeClient:
    connector = "gmail"

    def __init__(self):
        self.calls = []

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return f"called {name} with {arguments}"


def test_mcp_tools_namespacing_and_schema():
    from anvil_ai_employee.mcp.adapter import mcp_tools

    client = _FakeClient()
    specs = [_spec("send_email", {"destructiveHint": True}, props={"to": {"type": "string"}})]
    tools = mcp_tools(client, specs)
    assert tools[0].name == "gmail__send_email"
    assert tools[0].schema["function"]["parameters"]["properties"] == {"to": {"type": "string"}}


def test_mcp_tools_fn_dispatches_to_client_with_unqualified_name():
    from anvil_ai_employee.mcp.adapter import mcp_tools
    from anvil_code_agent.tools.base import ToolContext

    client = _FakeClient()
    specs = [
        _spec("send_email", {"destructiveHint": True}),
        _spec("list_events", {"readOnlyHint": True}),
    ]
    tools = {t.name: t for t in mcp_tools(client, specs)}
    res = tools["gmail__send_email"]({"to": "x"}, ToolContext(workdir="/tmp"))
    assert res.ok is True
    assert "called send_email" in res.content
    assert client.calls == [("send_email", {"to": "x"})]  # unqualified name to server
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/ai-employee && uv run pytest tests/test_mcp_adapter.py -v`
Expected: FAIL — `anvil_ai_employee.mcp.adapter` not found

- [ ] **Step 3: Implement the adapter**

```python
# packages/ai-employee/src/anvil_ai_employee/mcp/adapter.py
"""Wrap MCP tools as P3 Tools. Names are namespaced ({connector}__{tool}) to avoid
collisions; risk is derived from MCP tool annotations and fed to the HITL policy so
read-only tools run directly while destructive ones suspend into the Agent Inbox."""

from __future__ import annotations

from anvil_code_agent.tools.base import Tool, ToolContext, ToolResult

from anvil_ai_employee.mcp.client import McpToolSpec


def mcp_risk(spec: McpToolSpec) -> str:
    a = spec.annotations or {}
    if a.get("readOnlyHint") is True:
        return "low"
    if a.get("destructiveHint") is True:
        return "high"
    if not a:
        return "high"  # unknown side effects → conservative
    return "medium"


def mcp_tools(client, specs: list[McpToolSpec]) -> list[Tool]:
    tools: list[Tool] = []
    for spec in specs:
        qualified = f"{client.connector}__{spec.name}"
        schema = spec.input_schema or {}
        params = schema.get("properties", {})
        required = schema.get("required", [])

        def _make(unqualified: str):
            def _fn(args: dict, ctx: ToolContext) -> ToolResult:
                return ToolResult(content=client.call_tool(unqualified, args), ok=True)

            return _fn

        tools.append(Tool(qualified, spec.description, params, required, _make(spec.name)))
    return tools
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/ai-employee && uv run pytest tests/test_mcp_adapter.py -v`
Expected: 4 passed

- [ ] **Step 5: Run ruff + whole-package suite**

Run: `cd packages/ai-employee && uv run ruff check --fix src tests && uv run pytest -m "not live" -q`
Expected: ruff clean; all tests pass

- [ ] **Step 6: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/mcp/adapter.py \
        packages/ai-employee/tests/test_mcp_adapter.py
git commit -m "feat(ai-employee): MCP tool adapter — namespaced P3 @tool + annotation-based risk"
```

---

### Task 6: Connector config + `mcp_risk_policy` + `build_mcp_registry`

**Files:**
- Create: `packages/ai-employee/src/anvil_ai_employee/mcp/connector.py`
- Test: `packages/ai-employee/tests/test_mcp_connector.py`

This wires everything: take token store + connector configs → live clients → a `ToolRegistry` of MCP tools → a `HitlPolicy` that suspends only the genuinely high-risk tools. The integration test runs the real mock server through `hitl_step` to prove `list_events` executes and `send_email` suspends.

- [ ] **Step 1: Write the failing test**

```python
# packages/ai-employee/tests/test_mcp_connector.py
import sys

import pytest

from anvil_ai_employee.hitl import HitlDecision

_SERVER = ["-m", "anvil_ai_employee.mcp.mock_servers.email_server"]


def test_mcp_risk_policy_routes_by_tool_risk():
    from anvil_ai_employee.mcp.connector import mcp_risk_policy

    policy = mcp_risk_policy({"gmail__send_email": "high", "gmail__list_events": "low"})
    assert policy("gmail__send_email", {}, "high") == HitlDecision.SUSPEND
    assert policy("gmail__list_events", {}, "low") == HitlDecision.EXECUTE
    # unknown tool falls back to risk_level() (unknown → high → suspend)
    assert policy("mystery_tool", {}, "high") == HitlDecision.SUSPEND


@pytest.mark.asyncio
async def test_build_mcp_registry_lists_tools_and_policy(session_factory):
    from anvil_ai_employee.mcp.connector import ConnectorConfig, build_mcp_registry
    from anvil_ai_employee.mcp.tokens import McpTokenStore

    store = McpTokenStore(session_factory)
    await store.put(employee="alice", connector="gmail", env_key="GMAIL_TOKEN", secret="TOK")
    cfg = ConnectorConfig(name="gmail", command=sys.executable, args=_SERVER)
    registry, clients, policy = await build_mcp_registry(
        configs=[cfg], employee="alice", token_store=store
    )
    try:
        names = {s["function"]["name"] for s in registry.schemas()}
        assert names == {"gmail__list_events", "gmail__send_email"}
        assert policy("gmail__list_events", {}, "low") == HitlDecision.EXECUTE
        assert policy("gmail__send_email", {}, "high") == HitlDecision.SUSPEND
    finally:
        for c in clients:
            c.close()


@pytest.mark.asyncio
async def test_hitl_step_executes_read_suspends_write(session_factory):
    """End-to-end through the real mock server: read tool runs, write tool suspends."""
    from anvil_code_agent.state import AgentState
    from anvil_code_agent.tools.base import ToolContext

    from anvil_ai_employee.hitl import hitl_step
    from anvil_ai_employee.mcp.connector import ConnectorConfig, build_mcp_registry
    from anvil_ai_employee.mcp.tokens import McpTokenStore

    store = McpTokenStore(session_factory)
    await store.put(employee="alice", connector="gmail", env_key="GMAIL_TOKEN", secret="TOK")
    cfg = ConnectorConfig(name="gmail", command=sys.executable, args=_SERVER)
    registry, clients, policy = await build_mcp_registry(
        configs=[cfg], employee="alice", token_store=store
    )
    ctx = ToolContext(workdir="/tmp")
    try:
        # craft a state whose last assistant message proposes a read-only tool call
        def state_calling(tool_name, args):
            st = AgentState.new(system="p", task="t", workdir="/tmp", max_steps=5)
            assistant = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": __import__("json").dumps(args)},
                    }
                ],
            }
            return st.append(assistant)

        read_state = state_calling("gmail__list_events", {"date": "2026-06-09"})
        after_read = await hitl_step(read_state, "deepseek-chat", registry, ctx, policy=policy)
        assert after_read.status == "running"
        assert any(
            m.get("role") == "tool" and "standup" in (m.get("content") or "")
            for m in after_read.messages
        )

        write_state = state_calling("gmail__send_email", {"to": "x", "subject": "s", "body": "b"})
        after_write = await hitl_step(write_state, "deepseek-chat", registry, ctx, policy=policy)
        assert after_write.status == "suspended"
    finally:
        for c in clients:
            c.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/ai-employee && uv run pytest tests/test_mcp_connector.py -v`
Expected: FAIL — `anvil_ai_employee.mcp.connector` not found

- [ ] **Step 3: Implement connector wiring**

```python
# packages/ai-employee/src/anvil_ai_employee/mcp/connector.py
"""Assemble MCP connectors into a live ToolRegistry + HITL policy. A connector names a
stdio MCP server command; credentials are pulled from McpTokenStore (never from config)
and injected into the server's env at spawn. The returned policy suspends only the tools
whose MCP annotations mark them high-risk."""

from __future__ import annotations

from dataclasses import dataclass, field

from anvil_code_agent.harness.permission import risk_level
from anvil_code_agent.tools.base import ToolRegistry

from anvil_ai_employee.hitl import HitlDecision, HitlPolicy
from anvil_ai_employee.mcp.adapter import mcp_risk, mcp_tools
from anvil_ai_employee.mcp.client import McpClient
from anvil_ai_employee.mcp.tokens import McpTokenStore


@dataclass
class ConnectorConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)


def mcp_risk_policy(risk_by_tool: dict[str, str]) -> HitlPolicy:
    def policy(name: str, args: dict, risk: str) -> HitlDecision:
        r = risk_by_tool.get(name) or risk_level(name)
        return HitlDecision.SUSPEND if r == "high" else HitlDecision.EXECUTE

    return policy


async def build_mcp_registry(
    *, configs: list[ConnectorConfig], employee: str, token_store: McpTokenStore
) -> tuple[ToolRegistry, list[McpClient], HitlPolicy]:
    all_tools = []
    clients: list[McpClient] = []
    risk_by_tool: dict[str, str] = {}
    for cfg in configs:
        env = await token_store.env_for(employee=employee, connector=cfg.name)
        client = McpClient(connector=cfg.name, command=cfg.command, args=cfg.args, env=env)
        specs = client.start()  # blocking handshake on the client's own loop thread
        clients.append(client)
        all_tools.extend(mcp_tools(client, specs))
        for spec in specs:
            risk_by_tool[f"{cfg.name}__{spec.name}"] = mcp_risk(spec)
    return ToolRegistry(all_tools), clients, mcp_risk_policy(risk_by_tool)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/ai-employee && uv run pytest tests/test_mcp_connector.py -v`
Expected: 3 passed

- [ ] **Step 5: Run ruff + whole-package suite**

Run: `cd packages/ai-employee && uv run ruff check --fix src tests && uv run pytest -m "not live" -q`
Expected: ruff clean; all tests pass

- [ ] **Step 6: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/mcp/connector.py \
        packages/ai-employee/tests/test_mcp_connector.py
git commit -m "feat(ai-employee): build_mcp_registry + mcp_risk_policy (MCP tools into M3 HITL)"
```

---

### Task 7: CLI `mcp` subcommands + `run-mcp`

**Files:**
- Modify: `packages/ai-employee/src/anvil_ai_employee/cli.py`
- Test: `packages/ai-employee/tests/test_ai_employee_cli.py` (extend)

`run-mcp` mirrors `_run_hitl_demo` but uses the MCP registry+policy. Cross-process inbox resume for MCP tools needs the connector rebuilt with a live client, so the demo offers in-process `--auto-approve` to show the full closed loop (suspend → approve → real dispatch to the mock server → redacted result). Without the flag it just suspends and prints the inbox id like `run-hitl`.

- [ ] **Step 1: Write the failing test**

```python
# append to packages/ai-employee/tests/test_ai_employee_cli.py
import sys

import pytest

_MCP_SERVER = ["-m", "anvil_ai_employee.mcp.mock_servers.email_server"]


@pytest.mark.asyncio
async def test_mcp_list_tools_text(session_factory):
    from anvil_ai_employee.cli import mcp_list_tools_text
    from anvil_ai_employee.mcp.connector import ConnectorConfig

    cfg = ConnectorConfig(name="gmail", command=sys.executable, args=_MCP_SERVER)
    text = await mcp_list_tools_text(session_factory, employee="alice", config=cfg)
    assert "gmail__list_events" in text
    assert "low" in text and "high" in text


@pytest.mark.asyncio
async def test_mcp_put_token(session_factory):
    from anvil_ai_employee.cli import mcp_put_token
    from anvil_ai_employee.mcp.tokens import McpTokenStore

    await mcp_put_token(
        session_factory, employee="alice", connector="gmail", env_key="GMAIL_TOKEN", secret="T"
    )
    env = await McpTokenStore(session_factory).env_for(employee="alice", connector="gmail")
    assert env == {"GMAIL_TOKEN": "T"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/ai-employee && uv run pytest tests/test_ai_employee_cli.py -k mcp -v`
Expected: FAIL — `cannot import name 'mcp_list_tools_text'`

- [ ] **Step 3: Add helper functions to `cli.py`**

Add these functions near the other async helpers (e.g. after `_run_hitl_demo`, before `def main()`):

```python
async def mcp_put_token(
    sf: async_sessionmaker[AsyncSession],
    *,
    employee: str,
    connector: str,
    env_key: str,
    secret: str,
) -> None:
    from anvil_ai_employee.mcp.tokens import McpTokenStore

    await McpTokenStore(sf).put(
        employee=employee, connector=connector, env_key=env_key, secret=secret
    )


async def mcp_list_tools_text(
    sf: async_sessionmaker[AsyncSession],
    *,
    employee: str,
    config,
) -> str:
    from anvil_ai_employee.mcp.adapter import mcp_risk
    from anvil_ai_employee.mcp.client import McpClient
    from anvil_ai_employee.mcp.tokens import McpTokenStore

    env = await McpTokenStore(sf).env_for(employee=employee, connector=config.name)
    client = McpClient(connector=config.name, command=config.command, args=config.args, env=env)
    try:
        specs = client.start()
        lines = []
        for spec in specs:
            ann = ",".join(k for k, v in (spec.annotations or {}).items() if v is True) or "-"
            lines.append(f"  {config.name}__{spec.name}  risk={mcp_risk(spec)}  hints={ann}")
    finally:
        client.close()
    return f"connector {config.name} tools:\n" + "\n".join(lines)


async def _run_mcp_demo(
    sf: async_sessionmaker[AsyncSession],
    *,
    persona: str,
    task: str,
    employee: str,
    model: str,
    config,
    auto_approve: bool,
) -> None:
    from anvil_code_agent.state import AgentState
    from anvil_code_agent.tools.base import ToolContext
    from anvil_kb.embed import FastEmbedEmbedder

    from anvil_ai_employee.hitl import hitl_run
    from anvil_ai_employee.inbox import InboxStore
    from anvil_ai_employee.inbox_resume import resume_from_inbox
    from anvil_ai_employee.mcp.connector import build_mcp_registry
    from anvil_ai_employee.mcp.tokens import McpTokenStore

    registry, clients, policy = await build_mcp_registry(
        configs=[config], employee=employee, token_store=McpTokenStore(sf)
    )
    ctx = ToolContext(workdir="/tmp")
    state = AgentState.new(system=persona, task=task, workdir="/tmp", max_steps=10)
    print(f"[run-mcp] connector={config.name} model={model} employee={employee}")
    print(f"[run-mcp] task: {task}")
    try:
        out = await hitl_run(state, model, registry, ctx, policy=policy)
        if out.status == "suspended":
            iid = await InboxStore(sf).suspend(employee=employee, state=out)
            print(f"[run-mcp] 高风险 MCP 动作挂起进 Inbox。inbox_id={iid}")
            if auto_approve:
                store = InboxStore(sf)
                await inbox_resolve(sf, inbox_id=iid, decision="approve", payload={})
                row = await store.get(iid)
                resumed = await resume_from_inbox(
                    row, registry=registry, ctx=ctx, model=model,
                    session_factory=sf, embedder=FastEmbedEmbedder(),
                )
                reply = next(
                    (m.get("content") for m in reversed(resumed.messages)
                     if m.get("role") == "tool"), "",
                )
                print(f"[run-mcp] --auto-approve:已执行 MCP 工具,结果(已脱敏):{reply}")
        elif out.status == "done":
            print(f"[run-mcp] 完成(未触发挂起)。步数={out.step}")
        else:
            print(f"[run-mcp] 终态: {out.status}")
    except Exception as exc:  # noqa: BLE001
        print(f"[run-mcp] 失败(可能缺 API key 或连接器): {exc}")
    finally:
        for c in clients:
            c.close()
```

> NOTE for implementer: `cli.py` does NOT import `ToolContext` or `sys` at module level. `_run_mcp_demo` imports `ToolContext` locally (shown above). Add `import sys` to the top-level imports of `cli.py` (needed by the argparse defaults in Step 4).

- [ ] **Step 4: Register the subcommands in `main()`**

In `main()`, after the `run-hitl` parser block and before `sf = make_session_factory()`, add:

```python
    mcp_p = sub.add_parser("mcp")
    mcp_sub = mcp_p.add_subparsers(dest="mcp_cmd", required=True)

    mcp_lt = mcp_sub.add_parser("list-tools")
    mcp_lt.add_argument("--connector", default="gmail")
    mcp_lt.add_argument("--command", default=sys.executable)
    mcp_lt.add_argument(
        "--args", default="-m anvil_ai_employee.mcp.mock_servers.email_server",
        help="space-separated server argv",
    )
    mcp_lt.add_argument("--employee", default="assistant")

    mcp_pt = mcp_sub.add_parser("put-token")
    mcp_pt.add_argument("--connector", required=True)
    mcp_pt.add_argument("--env-key", required=True, dest="env_key")
    mcp_pt.add_argument("--secret", required=True)
    mcp_pt.add_argument("--employee", default="assistant")

    rm = sub.add_parser("run-mcp")
    rm.add_argument("--persona", default="你是一个 AI 助理,可使用外部工具。")
    rm.add_argument("--task", default="查一下 2026-06-09 的日程。")
    rm.add_argument("--employee", default="assistant")
    rm.add_argument("--model", default="deepseek-chat")
    rm.add_argument("--connector", default="gmail")
    rm.add_argument("--command", default=sys.executable)
    rm.add_argument(
        "--args", default="-m anvil_ai_employee.mcp.mock_servers.email_server",
        help="space-separated server argv",
    )
    rm.add_argument("--auto-approve", action="store_true", dest="auto_approve")
```

Add `import sys` at the top of `cli.py` if not already present.

In the dispatch section (where other `elif args.cmd == ...` branches live), add:

```python
    elif args.cmd == "mcp":
        from anvil_ai_employee.mcp.connector import ConnectorConfig

        if args.mcp_cmd == "list-tools":
            cfg = ConnectorConfig(
                name=args.connector, command=args.command, args=args.args.split()
            )
            print(asyncio.run(mcp_list_tools_text(sf, employee=args.employee, config=cfg)))
        elif args.mcp_cmd == "put-token":
            asyncio.run(
                mcp_put_token(
                    sf, employee=args.employee, connector=args.connector,
                    env_key=args.env_key, secret=args.secret,
                )
            )
            print(f"token 已写入: {args.employee}/{args.connector}/{args.env_key}")
    elif args.cmd == "run-mcp":
        from anvil_ai_employee.mcp.connector import ConnectorConfig

        cfg = ConnectorConfig(name=args.connector, command=args.command, args=args.args.split())
        asyncio.run(
            _run_mcp_demo(
                sf, persona=args.persona, task=args.task, employee=args.employee,
                model=args.model, config=cfg, auto_approve=args.auto_approve,
            )
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd packages/ai-employee && uv run pytest tests/test_ai_employee_cli.py -k mcp -v`
Expected: 2 passed

- [ ] **Step 6: Run ruff + whole-package suite**

Run: `cd packages/ai-employee && uv run ruff check --fix src tests && uv run pytest -m "not live" -q`
Expected: ruff clean; all tests pass

- [ ] **Step 7: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/cli.py \
        packages/ai-employee/tests/test_ai_employee_cli.py
git commit -m "feat(ai-employee): CLI mcp list-tools/put-token + run-mcp demo (in-process --auto-approve)"
```

---

### Task 8: Example 11 README + CLAUDE.md M4 section

**Files:**
- Create: `examples/11-ai-employee-mcp/README.md`
- Modify: `CLAUDE.md` (anvil-ai-employee section)

- [ ] **Step 1: Write the example README**

```markdown
# 11 — AI 员工·MCP 连接器(P4-M4:接外部世界)

M1-M3 让员工能定时干活、有三层记忆、高风险动作挂起等人审批。M4 让它**通过 MCP(Model Context Protocol)调用外部工具**(邮件/IM/日历/工单)——而且 **agent 永远拿不到长期凭证**,高风险 MCP 动作**自动**走 M3 的 Agent Inbox。

## 自研 client,吃透协议

不套 `mcp` PyPI SDK,手写一个 stdio JSON-RPC 2.0 client:把 MCP server 当子进程拉起,走 `initialize` 握手 → `notifications/initialized` 通知 → `tools/list` → `tools/call`。MCP stdio 帧 = **每行一个 JSON 对象**(不是 LSP 的 Content-Length)。

**会话生命周期是本期最大的坑**:MCP session 是长连接(initialize 一次,call 多次),而 `@tool` 是同步的。M1 的 `block_on`(每次新 event loop)绑不住子进程。解法:`McpClient` 持有一个**后台线程 + 独占 event loop**,子进程 transport 永远活在那个 loop 上,同步工具用 `run_coroutine_threadsafe(...).result()` 投递调用。

## 凭证服务端托管

`ae_mcp_tokens` 表按 (employee, connector, env_key) 存密钥;`McpClient` 在 spawn server 时把它们注入**子进程的 env**。agent 调 `gmail__send_email{to,subject,body}` 时参数里**没有 token**;结果文本里若回显了 token,client 自动脱敏成 `***`。

## 风险分级直接喂给 M3 HITL

读 MCP 工具的 annotation:`readOnlyHint`→low、`destructiveHint`(或无 hint)→high、其余 medium。`mcp_risk_policy` 把每个工具的真实 risk 喂给 M3 的 `hitl_run`:`calendar__list_events` 直接执行,`gmail__send_email` 挂起进 Inbox。**HITL/inbox/干预记忆全部不改**。

## 跑一遍(需 `ANVIL_DATABASE_URL`;run-mcp 需 `DEEPSEEK_API_KEY`)

```bash
export ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil

# 1) 看连接器暴露了哪些工具 + 各自风险(起 mock server,握手,tools/list)
uv run anvil-ai-employee mcp list-tools --connector gmail
#   gmail__list_events  risk=low   hints=readOnlyHint
#   gmail__send_email   risk=high  hints=destructiveHint

# 2) 把凭证托管到服务端(agent 拿不到)
uv run anvil-ai-employee mcp put-token --connector gmail --env-key GMAIL_TOKEN --secret my-oauth-token --employee assistant

# 3) 读类任务:直接执行,不挂起
uv run anvil-ai-employee run-mcp --task "查 2026-06-09 的日程"

# 4) 写类任务:挂起进 Inbox;--auto-approve 在同进程内演示完整闭环
uv run anvil-ai-employee run-mcp --task "给 boss@x.com 发封周报邮件" --auto-approve
# → [run-mcp] 高风险 MCP 动作挂起进 Inbox。inbox_id=...
# → [run-mcp] --auto-approve:已执行 MCP 工具,结果(已脱敏):已发送 to=boss@x.com ... (via token=***)
```

**真实验证**:真子进程 mock server 完成 initialize 握手、tools/list 返回带 annotation 的两个工具;读类工具经 `hitl_step` 直接执行返回假日程;写类工具被挂起进 ae_inbox;approve 后 client 真发 JSON-RPC `tools/call`,server 读到 env 注入的 token 并回显,结果在 client 侧脱敏成 `***`。整条 spawn→handshake→list→call→HITL→脱敏在真子进程上走通。

## 跨进程 Inbox resume(本期边界)

M3 的 `inbox approve <id>` 用的是 demo registry(无 MCP 工具)。MCP 工具的 resume 需要在配置了连接器、client 活着的进程里做——所以 M4 的 demo 用同进程 `--auto-approve`。跨进程 MCP resume(worker 重建连接器)留作螺旋。

## 复用与新建

- ✓ 真复用:P3 `tools/base`(Tool/ToolRegistry)、`permission.risk_level`(policy fallback);M3 `hitl_run`/`hitl_step`/`apply_decision`/InboxStore/`resume_from_inbox`/干预写记忆(**全不改**);M1 worker/cli 骨架;M2a MemoryStore。
- ✗ 新建:`mcp/`(transport/client/tokens/adapter/connector + mock server);`ae_mcp_tokens` 表;CLI `mcp` 子命令 + `run-mcp`。

## 留待
真 OAuth 浏览器授权码流 / token refresh、SSE/HTTP 传输、MCP resources/prompts、凭证加密、跨进程 MCP resume、M5 多员工编队。
```

- [ ] **Step 2: Add the M4 subsection to `CLAUDE.md`**

Find the `anvil-ai-employee` section (it has M1/M2a/M2b/M3 subsections) and append an M4 subsection after the M3 one. Match the existing terse, evidence-style tone. Content to add (adapt heading depth to match siblings):

```markdown
#### M4「MCP 连接器」(examples/11)

- **自研 stdio JSON-RPC 2.0 client**(`mcp/transport.py` + `mcp/client.py`):不套 `mcp` SDK;`initialize`→`notifications/initialized`→`tools/list`→`tools/call`;stdio 帧 = 每行一个 JSON。
- **会话生命周期坑**:MCP session 是长连接而 `@tool` 同步;`McpClient` 用**后台线程+独占 event loop** 持有子进程 transport,同步工具走 `run_coroutine_threadsafe(...).result()`(不是 M1 的 block_on——那个每次换 loop 绑不住子进程)。
- **凭证服务端托管**(`mcp/tokens.py` + `ae_mcp_tokens` 表):密钥按 (employee, connector, env_key) 存,spawn server 时注入子进程 env;agent 的 tool args 永不含凭证;结果文本里的 token 在 client 侧脱敏成 `***`。
- **风险→HITL 零改造**(`mcp/adapter.py` + `mcp/connector.py`):`mcp_risk` 读 annotation(readOnlyHint→low / destructiveHint 或无 hint→high / 其余 medium);`mcp_risk_policy` 喂给 M3 `hitl_run`——读类直接执行、写类挂起进 ae_inbox;`apply_decision`/inbox/干预记忆全不改。
- 工具名命名空间 `{connector}__{tool}` 防撞车。mock server `mcp/mock_servers/email_server.py`(零依赖,reverse-validate 握手)。CLI:`mcp list-tools`/`mcp put-token`/`run-mcp [--auto-approve]`。
- 边界:跨进程 MCP inbox resume 需重建连接器(demo 用同进程 `--auto-approve`);真 OAuth/refresh、SSE/HTTP 传输、resources/prompts 留作螺旋。
```

- [ ] **Step 3: Commit**

```bash
git add examples/11-ai-employee-mcp/README.md CLAUDE.md
git commit -m "docs(ai-employee): M4 MCP connectors example + CLAUDE.md section"
```

---

## Final Verification (after all tasks)

- [ ] **Whole-repo suite green (CI parity):** from repo root run `uv run pytest -m "not live" -q` across all packages. Expected: all pass, no collection errors (watch for test-file-name collisions per the M1 lesson).
- [ ] **Optional live end-to-end (needs `DEEPSEEK_API_KEY` + PG):** `cd packages/ai-employee && uv run pytest -m live -q` if any live MCP integration test was added; otherwise manually run the example README's `run-mcp --auto-approve` flow and confirm the redacted `(via token=***)` result.
- [ ] **Dispatch final code reviewer** (opus) for the whole M4 branch.
- [ ] **Use superpowers:finishing-a-development-branch** to open the PR, wait for CI green, self-merge.

## Spec Coverage Check

| Spec requirement | Task |
|---|---|
| stdio JSON-RPC 2.0 transport (newline frames, EOF handling) | Task 2 |
| McpClient persistent session on background loop thread; initialize/tools.list/tools.call | Task 4 |
| Secret redaction from tool output | Task 4 |
| `ae_mcp_tokens` + server-side custody / env injection | Tasks 1, 4, 6 |
| MCP tools → namespaced P3 @tool | Task 5 |
| Annotation-based risk mapping | Task 5 |
| `mcp_risk_policy` into M3 HITL (read executes / write suspends) | Task 6 |
| `build_mcp_registry` | Task 6 |
| Mock stdio MCP server (annotations distinguish read/write; token via env) | Task 3 |
| CLI `mcp list-tools` / `put-token` / `run-mcp` | Task 7 |
| Example 11 + CLAUDE.md | Task 8 |
| Reuse M3 hitl/inbox/inbox_resume/hitl_memory unchanged | Tasks 6, 7 (no edits to those files) |
