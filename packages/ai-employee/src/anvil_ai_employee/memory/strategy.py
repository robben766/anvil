"""MemoryStrategy: the seam between a chat employee and a memory philosophy.
mem0 (managed) drives memory in the orchestrator (system_prefix recall + after_turn update).
Letta (self-managed, M2b) drives it via agent tools. NoMemoryStrategy is the baseline."""

from __future__ import annotations

from typing import Any, Protocol

from anvil_code_agent.tools.base import ToolRegistry


class MemoryStrategy(Protocol):
    def build_registry(self, ctx: Any) -> ToolRegistry: ...
    async def system_prefix(self, employee: str, user_msg: str) -> str: ...
    async def after_turn(self, employee: str, session: Any, msgs: list[dict]) -> None: ...


class NoMemoryStrategy:
    """Baseline: no recall, no tools, no learning."""

    def build_registry(self, ctx: Any) -> ToolRegistry:
        return ToolRegistry([])

    async def system_prefix(self, employee: str, user_msg: str) -> str:
        return ""

    async def after_turn(self, employee: str, session: Any, msgs: list[dict]) -> None:
        return None
