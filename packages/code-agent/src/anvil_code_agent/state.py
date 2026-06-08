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
