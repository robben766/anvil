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
