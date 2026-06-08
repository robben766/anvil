"""bash tool: subprocess with timeout + head/tail output truncation."""

from __future__ import annotations

import subprocess

from anvil_code_agent.tools.base import ToolContext, ToolResult, tool


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    half = limit // 2
    return text[:half] + f"\n... [truncated {len(text)} chars] ...\n" + text[-half:], True


@tool(
    name="bash",
    description="Run a shell command in the working dir. Returns combined stdout+stderr.",
    params={"cmd": {"type": "string", "description": "shell command"}},
    required=["cmd"],
)
def bash(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        proc = subprocess.run(
            args["cmd"],
            shell=True,
            cwd=ctx.workdir,
            capture_output=True,
            text=True,
            timeout=ctx.timeout,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(content=f"command timed out after {ctx.timeout}s", ok=False)
    out = (proc.stdout or "") + (proc.stderr or "")
    out, truncated = _truncate(out, ctx.max_output)
    if proc.returncode != 0:
        return ToolResult(content=f"exit code {proc.returncode}\n{out}", ok=False, truncated=truncated)
    return ToolResult(content=out, ok=True, truncated=truncated)
