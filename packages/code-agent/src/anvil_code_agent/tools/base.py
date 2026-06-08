"""Tool protocol + registry. ACI principle: tools ALWAYS return a readable result,
even on failure — the failure text is feedback the model uses to retry (12-Factor #9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolResult:
    content: str
    ok: bool
    truncated: bool = False


@dataclass
class ToolContext:
    workdir: str
    timeout: float = 120.0
    max_output: int = 4096


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        params: dict[str, Any],
        required: list[str],
        fn: Callable[[dict[str, Any], ToolContext], ToolResult],
    ):
        self.name = name
        self.description = description
        self._fn = fn
        self.schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": required,
                },
            },
        }

    def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self._fn(args, ctx)


def tool(*, name: str, description: str, params: dict[str, Any], required: list[str]):
    def deco(fn: Callable[[dict[str, Any], ToolContext], ToolResult]) -> Tool:
        return Tool(name, description, params, required, fn)

    return deco


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema for t in self._tools.values()]

    def dispatch(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        t = self._tools.get(name)
        if t is None:
            return ToolResult(content=f"unknown tool: {name}", ok=False)
        try:
            return t(args, ctx)
        except Exception as e:  # noqa: BLE001 — failures are feedback, never crash the loop
            return ToolResult(content=f"tool '{name}' raised {type(e).__name__}: {e}", ok=False)
