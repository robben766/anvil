"""Write each human intervention back to long-term memory (越用越懂你): future runs can
recall how the human decided last time. Stored kind='hitl' with embedding so mem0 recall
picks it up."""

from __future__ import annotations

from anvil_ai_employee.memory.store import MemoryStore


def _phrase(employee, tool_name, decision, payload, tool_args) -> str:
    if decision == "approve":
        return f"审批人批准了 {employee} 的 {tool_name} 操作(参数 {tool_args})。"
    if decision == "edit":
        return f"审批人把 {employee} 的 {tool_name} 参数改成 {payload.get('args')}。"
    if decision == "reject":
        return f"审批人拒绝了 {employee} 的 {tool_name} 操作,原因:{payload.get('reason', '')}。"
    if decision == "respond":
        return f"对 {employee} 的 {tool_name},审批人直接答复:{payload.get('message', '')}。"
    return f"对 {tool_name} 的未知决策 {decision}。"


async def record_intervention(session_factory, *, embedder, employee, tool_name,
                              decision, payload, tool_args) -> None:
    text = _phrase(employee, tool_name, decision, payload, tool_args)
    emb = embedder.embed_texts([text])[0]
    await MemoryStore(session_factory).insert(
        employee=employee, kind="hitl", content=text, embedding=emb)
