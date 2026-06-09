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
