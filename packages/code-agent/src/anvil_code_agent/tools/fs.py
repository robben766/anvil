"""Filesystem tools: read_file (numbered lines + truncation) and edit_file
(SEARCH-REPLACE with strict guardrails — editing reliability决定可用性)."""

from __future__ import annotations

import os

from anvil_code_agent.tools.base import ToolContext, ToolResult, tool


def _resolve(workdir: str, path: str) -> str:
    return os.path.join(workdir, path)


@tool(
    name="read_file",
    description="Read a file relative to the working dir. Returns lines prefixed with line numbers.",
    params={
        "path": {"type": "string", "description": "file path relative to working dir"},
    },
    required=["path"],
)
def read_file(args: dict, ctx: ToolContext) -> ToolResult:
    full = _resolve(ctx.workdir, args["path"])
    if not os.path.isfile(full):
        return ToolResult(content=f"file not found: {args['path']}", ok=False)
    with open(full, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    numbered = "\n".join(f"{i + 1}\t{ln}" for i, ln in enumerate(text.splitlines()))
    if len(numbered) > ctx.max_output:
        head = numbered[: ctx.max_output]
        return ToolResult(
            content=head + f"\n... [truncated, {len(numbered)} chars total]",
            ok=True,
            truncated=True,
        )
    return ToolResult(content=numbered, ok=True)
