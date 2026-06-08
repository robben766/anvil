"""The reducer loop. step() = one gateway tool_use round-trip + tool execution.
run() drives step() until the model finishes or a guard fires."""

from __future__ import annotations

import json

from anvil_gateway import chat
from anvil_obs import span

from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry


async def step(
    state: AgentState, model: str, registry: ToolRegistry, ctx: ToolContext
) -> AgentState:
    """One reduce: call the model, execute any tool calls, return the new state."""
    resp = await chat(model, list(state.messages), tools=registry.schemas())
    assistant_msg = resp.raw["choices"][0]["message"]
    if resp.tool_calls:
        new = state.append(assistant_msg)
        for tc in resp.tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            with span("code_agent.tool", tool=name):
                result = registry.dispatch(name, args, ctx)
            new = new.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": result.content}
            )
        return new.advance()
    # no tool calls → the model is done
    return state.append(assistant_msg).advance().finish("done")
