"""Context engineering: estimate token budget + structurally-safe compaction.

Compaction only TRUNCATES the content of old tool-result messages — it never drops or
reorders messages, so tool_call/tool pairing stays valid. This is best-effort: if the
recent window itself exceeds budget it may not fully fit, which is the honest behavior.
(LLM summarization of dropped turns is a future tier.)"""

from __future__ import annotations

from typing import Any

Message = dict[str, Any]


def estimate_tokens(messages: list[Message]) -> int:
    """Cheap heuristic: ~4 chars per token over all stringified content/tool_calls."""
    chars = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        if m.get("tool_calls"):
            chars += len(str(m["tool_calls"]))
    return chars // 4


def compact(
    messages: list[Message],
    *,
    max_tokens: int,
    keep_recent: int = 6,
    tool_cap: int = 200,
) -> list[Message]:
    """Return messages shrunk toward max_tokens by truncating old tool outputs.
    Protects messages[0] (system), messages[1] (original task), and the last keep_recent."""
    if estimate_tokens(messages) <= max_tokens:
        return messages
    n = len(messages)
    out: list[Message] = []
    for i, m in enumerate(messages):
        protected = i < 2 or i >= n - keep_recent
        content = m.get("content")
        if (
            not protected
            and m.get("role") == "tool"
            and isinstance(content, str)
            and len(content) > tool_cap
        ):
            mm = dict(m)
            mm["content"] = content[:tool_cap] + " ...[older tool output truncated]"
            out.append(mm)
        else:
            out.append(m)
    return out
