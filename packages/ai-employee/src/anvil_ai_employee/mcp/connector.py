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
