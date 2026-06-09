"""Checkpoint recovery: dump an AgentState to a JSON-safe dict and reload it.
messages are already plain dicts, so persistence is trivial — enabling resume after
a crash or a deliberate pause (12-Factor #6: launch/pause/resume)."""

from __future__ import annotations

from typing import Any

from anvil_code_agent.state import AgentState


def dump_state(state: AgentState) -> dict[str, Any]:
    return {
        "messages": [dict(m) for m in state.messages],
        "step": state.step,
        "max_steps": state.max_steps,
        "workdir": state.workdir,
        "status": state.status,
    }


def load_state(d: dict[str, Any]) -> AgentState:
    return AgentState(
        messages=tuple(d["messages"]),
        step=d["step"],
        max_steps=d["max_steps"],
        workdir=d["workdir"],
        status=d["status"],
    )
