"""verify tool: run the project's test command, return structured pass/fail + summary.
This is the closed-loop feedback channel — the agent reads failures and keeps fixing."""

from __future__ import annotations

from anvil_code_agent.tools.base import ToolContext, ToolResult, tool
from anvil_code_agent.tools.shell import bash

DEFAULT_TEST_CMD = "python -m pytest -q"


@tool(
    name="run_tests",
    description="Run the test suite in the working dir. Returns pass/fail and failure output.",
    params={
        "cmd": {"type": "string", "description": f"test command (default: '{DEFAULT_TEST_CMD}')"},
    },
    required=[],
)
def run_tests(args: dict, ctx: ToolContext) -> ToolResult:
    cmd = args.get("cmd") or DEFAULT_TEST_CMD
    res = bash({"cmd": cmd}, ctx)
    # bash already sets ok=False on nonzero exit (pytest exits nonzero on failure);
    # surface the same content but framed as a test verdict.
    verdict = "TESTS PASSED" if res.ok else "TESTS FAILED"
    return ToolResult(content=f"{verdict}\n{res.content}", ok=res.ok, truncated=res.truncated)
