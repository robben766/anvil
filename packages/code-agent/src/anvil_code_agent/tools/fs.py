"""Filesystem tools: read_file (numbered lines + truncation) and edit_file
(SEARCH-REPLACE with strict guardrails — editing reliability决定可用性)."""

from __future__ import annotations

import os

from anvil_code_agent.tools.base import ToolContext, ToolResult, tool


def _resolve(workdir: str, path: str) -> str:
    return os.path.join(workdir, path)


@tool(
    name="read_file",
    description=(
        "Read a file relative to the working dir. Returns lines prefixed with line numbers. "
        "NOTE: paths are not sandboxed in M1 — absolute or '../' paths can escape the worktree; "
        "true filesystem isolation is deferred to M3 (Docker)."
    ),
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


@tool(
    name="edit_file",
    description=(
        "Replace an EXACT unique snippet in a file. 'search' must match exactly once "
        "(whitespace-sensitive). If it matches zero or multiple times, the edit is rejected "
        "and you must read the file and provide a more specific search block. "
        "NOTE: paths are not sandboxed in M1 — absolute or '../' paths can escape the worktree; "
        "true filesystem isolation is deferred to M3 (Docker)."
    ),
    params={
        "path": {"type": "string", "description": "file path relative to working dir"},
        "search": {"type": "string", "description": "exact snippet to find (must be unique)"},
        "replace": {"type": "string", "description": "replacement snippet"},
    },
    required=["path", "search", "replace"],
)
def edit_file(args: dict, ctx: ToolContext) -> ToolResult:
    path, search, replace = args["path"], args["search"], args["replace"]
    if search == "":
        return ToolResult(content="empty search block is not allowed", ok=False)
    full = _resolve(ctx.workdir, path)
    if not os.path.isfile(full):
        return ToolResult(content=f"file not found: {path}", ok=False)
    with open(full, encoding="utf-8") as fh:
        text = fh.read()
    count = text.count(search)
    if count == 0:
        return ToolResult(
            content=f"search block not found in {path}; re-read the file and copy an exact snippet",
            ok=False,
        )
    if count > 1:
        return ToolResult(
            content=(
                f"search block matched {count} times (multiple matches) in {path};"
                " make it more specific (unique)"
            ),
            ok=False,
        )
    new_text = text.replace(search, replace, 1)
    with open(full, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    return ToolResult(content=f"edited {path}", ok=True)
