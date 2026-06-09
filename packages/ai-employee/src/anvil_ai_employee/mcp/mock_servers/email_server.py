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
                _text(
                    f"已发送 to={args.get('to')} subject={args.get('subject')} "
                    f"(via token={token})"
                ),
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
