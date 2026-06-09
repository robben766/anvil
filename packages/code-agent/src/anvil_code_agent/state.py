"""AgentState: immutable snapshot of one agent run. Reducer steps return new states."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

Message = dict[str, Any]
Status = str  # "running" | "done" | "exhausted" | "error"


@dataclass(frozen=True)
class AgentState:
    messages: tuple[Message, ...]
    step: int
    max_steps: int
    workdir: str
    status: Status = "running"

    @classmethod
    def new(cls, *, system: str, task: str, workdir: str, max_steps: int) -> AgentState:
        return cls(
            messages=(
                {"role": "system", "content": system},
                {"role": "user", "content": task},
            ),
            step=0,
            max_steps=max_steps,
            workdir=workdir,
            status="running",
        )

    def append(self, *msgs: Message) -> AgentState:
        return replace(self, messages=self.messages + tuple(msgs))

    def advance(self) -> AgentState:
        return replace(self, step=self.step + 1)

    def finish(self, status: Status) -> AgentState:
        return replace(self, status=status)

    def resume(self, user_msg: Message) -> AgentState:
        """Re-arm a finished chat state for another user turn: append the user message,
        reset to running, reset the per-turn step counter (max_steps is per-turn)."""
        return replace(self, messages=self.messages + (user_msg,), status="running", step=0)

    @classmethod
    def from_messages(
        cls,
        messages: tuple[Message, ...],
        *,
        workdir: str,
        max_steps: int,
        status: Status = "running",
    ) -> AgentState:
        """Rehydrate a state from a persisted message tuple (chat session resume)."""
        return cls(messages=messages, step=0, max_steps=max_steps, workdir=workdir, status=status)
