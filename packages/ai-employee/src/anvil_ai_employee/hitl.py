"""HITL agent loop: suspend on high-risk tool calls for human review, resume after.
The pending action is simply the unanswered tool_call in the last assistant message —
AgentState (via recovery.dump_state) fully captures the suspension point. We do NOT fork
P3's step(); hitl_step does one thing per call (process one pending tool, or one model
call) so suspension is a clean return."""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from anvil_code_agent.harness.permission import risk_level
from anvil_code_agent.state import AgentState
from anvil_obs import span

from anvil_gateway import chat


class HitlDecision(StrEnum):
    EXECUTE = "execute"
    SUSPEND = "suspend"
    DENY = "deny"


HitlPolicy = Callable[[str, dict[str, Any], str], HitlDecision]


def suspend_high(name: str, args: dict[str, Any], risk: str) -> HitlDecision:
    return HitlDecision.SUSPEND if risk == "high" else HitlDecision.EXECUTE


def _last_assistant_with_calls(messages) -> dict | None:
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            return m
        if m.get("role") == "tool":
            continue
        if m.get("role") == "assistant":
            return None  # a plain assistant message after = no pending
    return None


def _unanswered_tool_calls(state: AgentState) -> list[dict]:
    msg = _last_assistant_with_calls(state.messages)
    if msg is None:
        return []
    answered = {m.get("tool_call_id") for m in state.messages if m.get("role") == "tool"}
    return [tc for tc in msg["tool_calls"] if tc["id"] not in answered]


def _args_of(tc: dict) -> dict[str, Any]:
    try:
        return json.loads(tc["function"].get("arguments") or "{}")
    except json.JSONDecodeError:
        return {}


def _tool_msg(tcid: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tcid, "content": content}


async def hitl_step(state, model, registry, ctx, *, policy: HitlPolicy) -> AgentState:
    pending = _unanswered_tool_calls(state)
    if pending:
        tc = pending[0]
        name = tc["function"]["name"]
        args = _args_of(tc)
        risk = risk_level(name)
        d = policy(name, args, risk)
        if d == HitlDecision.SUSPEND:
            return state.finish("suspended")
        if d == HitlDecision.DENY:
            return state.append(_tool_msg(tc["id"], f"denied by policy (risk={risk})"))
        with span("ai_employee.hitl.tool", tool=name, risk=risk):
            result = registry.dispatch(name, args, ctx)
        return state.append(_tool_msg(tc["id"], result.content))
    resp = await chat(model, list(state.messages), tools=registry.schemas())
    assistant = resp.raw["choices"][0]["message"]
    if resp.tool_calls:
        return state.append(assistant).advance()
    return state.append(assistant).advance().finish("done")


async def hitl_run(state, model, registry, ctx, *, policy: HitlPolicy = suspend_high) -> AgentState:
    with span("ai_employee.hitl.run", model=model):
        while state.status == "running":
            if state.step >= state.max_steps:
                return state.finish("exhausted")
            state = await hitl_step(state, model, registry, ctx, policy=policy)
            if state.status == "suspended":
                return state
        return state
