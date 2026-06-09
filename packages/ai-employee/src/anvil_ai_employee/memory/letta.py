"""LettaStrategy — MemGPT self-managed memory: the agent edits memory via tools.
Contrast Mem0Strategy (orchestrator-managed). after_turn is a no-op: the agent already
made its edits inside the turn. Core blocks are injected each turn (read-your-write is
eventual: a core_memory_replace this turn is visible next turn's system block)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anvil_code_agent.tools.base import ToolRegistry

from anvil_ai_employee.memory.coreblocks import CoreBlockStore
from anvil_ai_employee.memory.letta_tools import LettaToolContext, build_letta_tools


@dataclass
class LettaChatCtx:
    employee: str
    session_id: Any


class LettaStrategy:
    def __init__(self, session_factory, *, embedder, model: str, char_limit: int = 500):
        self._sf = session_factory
        self._embedder = embedder
        self._model = model
        self._char_limit = char_limit
        self._cb = CoreBlockStore(session_factory, char_limit=char_limit)

    def build_registry(self, ctx: Any) -> ToolRegistry:
        employee = getattr(ctx, "employee", "assistant")
        session_id = getattr(ctx, "session_id", None)
        tctx = LettaToolContext(
            session_factory=self._sf,
            embedder=self._embedder,
            employee=employee,
            session_id=session_id,
            char_limit=self._char_limit,
        )
        return ToolRegistry(build_letta_tools(tctx))

    async def system_prefix(self, employee: str, user_msg: str) -> str:
        blocks = await self._cb.get_all(employee=employee)
        lines = [f"[{label}] {content}" for label, content in blocks.items()]
        return (
            "<core_memory>\n" + "\n".join(lines) + "\n</core_memory>\n"
            "你有记忆工具:core_memory_append/replace 编辑常驻记忆,"
            "archival_insert/archival_search 存取长期知识,conversation_search 翻查历史。"
            "遇到关于用户的重要信息,主动用这些工具记下来。"
        )

    async def after_turn(self, employee: str, session: Any, msgs: list[dict]) -> None:
        return None  # agent self-manages memory inside the turn
