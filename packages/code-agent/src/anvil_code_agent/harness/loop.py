"""The reducer loop. step() = one gateway tool_use round-trip + tool execution.
run() drives step() until the model finishes or a guard fires."""

from __future__ import annotations

import json

from anvil_obs import span

from anvil_code_agent.harness.context import compact
from anvil_code_agent.harness.permission import ApprovalPolicy, auto_approve, risk_level
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry
from anvil_gateway import chat


async def step(
    state: AgentState,
    model: str,
    registry: ToolRegistry,
    ctx: ToolContext,
    *,
    policy: ApprovalPolicy = auto_approve,
    token_budget: int | None = None,
    summarizer=None,
) -> AgentState:
    """One reduce: call the model, execute any tool calls, return the new state."""
    msgs = list(state.messages)
    if token_budget is not None:
        msgs = compact(msgs, max_tokens=token_budget, summarizer=summarizer)
    resp = await chat(model, msgs, tools=registry.schemas())
    assistant_msg = resp.raw["choices"][0]["message"]
    if resp.tool_calls:
        new = state.append(assistant_msg)
        for tc in resp.tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            risk = risk_level(name)
            if not policy(name, args, risk):
                new = new.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": f"tool '{name}' denied by approval policy (risk={risk})",
                    }
                )
                continue
            with span("code_agent.tool", tool=name, risk=risk):
                result = registry.dispatch(name, args, ctx)
            new = new.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": result.content}
            )
        return new.advance()
    # no tool calls → the model is done
    return state.append(assistant_msg).advance().finish("done")


async def run(
    state: AgentState,
    model: str,
    registry: ToolRegistry,
    ctx: ToolContext,
    *,
    policy: ApprovalPolicy = auto_approve,
    token_budget: int | None = None,
    summarizer=None,
) -> AgentState:
    """Drive step() until the model finishes or max_steps is hit."""
    with span("code_agent.run", model=model):
        while state.status == "running":
            if state.step >= state.max_steps:
                return state.finish("exhausted")
            state = await step(
                state, model, registry, ctx,
                policy=policy, token_budget=token_budget, summarizer=summarizer,
            )
        return state
