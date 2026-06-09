"""Resume a resolved inbox item: apply the human decision, log it to memory, continue
the HITL loop until the next suspension or completion."""

from __future__ import annotations

from anvil_code_agent.harness.recovery import load_state

from anvil_ai_employee.hitl import apply_decision, hitl_run, suspend_high
from anvil_ai_employee.hitl_memory import record_intervention


async def resume_from_inbox(
    inbox_row, *, registry, ctx, model, session_factory, embedder, policy=suspend_high
):
    state = load_state(inbox_row.state_json)
    # Apply the decision first; record the intervention only after it succeeds so a
    # malformed decision never leaves an orphan memory.
    state = apply_decision(
        state,
        decision=inbox_row.decision,
        payload=inbox_row.decision_payload or {},
        registry=registry,
        ctx=ctx,
    )
    await record_intervention(
        session_factory,
        embedder=embedder,
        employee=inbox_row.employee,
        tool_name=inbox_row.tool_name,
        decision=inbox_row.decision,
        payload=inbox_row.decision_payload or {},
        tool_args=inbox_row.tool_args,
    )
    return await hitl_run(state, model, registry, ctx, policy=policy)
