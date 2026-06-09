"""Search tools: grep (pure-Python regex walk, CI-safe — no ripgrep dependency) and
repo_map (Aider-style symbol map to help the agent locate code)."""

from __future__ import annotations

import os
import re

from anvil_code_agent.repomap.build import build_repo_map
from anvil_code_agent.tools.base import ToolContext, ToolResult, tool

_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"}


@tool(
    name="grep",
    description="Search file contents by regex. Returns 'relpath:lineno: line' matches. "
    "Paths are not sandboxed in M1 (workdir-relative walk).",
    params={
        "pattern": {"type": "string", "description": "Python regex"},
        "glob": {"type": "string", "description": "optional filename suffix filter, e.g. '.py'"},
    },
    required=["pattern"],
)
def grep(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        rx = re.compile(args["pattern"])
    except re.error as e:
        return ToolResult(content=f"invalid regex: {e}", ok=False)
    suffix = args.get("glob") or ""
    hits: list[str] = []
    for dirpath, dirnames, filenames in os.walk(ctx.workdir):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if suffix and not fn.endswith(suffix):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ctx.workdir)
            try:
                with open(full, encoding="utf-8", errors="replace") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            hits.append(f"{rel}:{i}: {line.rstrip()}")
            except OSError:
                continue
    if not hits:
        return ToolResult(content="no matches", ok=True)
    text = "\n".join(hits)
    if len(text) > ctx.max_output:
        return ToolResult(
            content=text[: ctx.max_output] + f"\n... [truncated, {len(hits)} matches]",
            ok=True,
            truncated=True,
        )
    return ToolResult(content=text, ok=True)


def _list_py_files(workdir: str) -> list[str]:
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(workdir):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                files.append(os.path.relpath(os.path.join(dirpath, fn), workdir))
    return files


@tool(
    name="repo_map",
    description="Return a ranked map of the repo's Python files and their top-level "
    "definitions (functions/classes), most-referenced files first. Use it to locate code.",
    params={},
    required=[],
)
def repo_map(args: dict, ctx: ToolContext) -> ToolResult:
    files = _list_py_files(ctx.workdir)
    if not files:
        return ToolResult(content="no python files found", ok=True)
    text = build_repo_map(ctx.workdir, files, max_chars=ctx.max_output)
    return ToolResult(content=text or "no python files found", ok=True)
