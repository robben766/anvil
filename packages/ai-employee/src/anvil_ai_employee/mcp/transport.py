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
        except (TimeoutError, ProcessLookupError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
