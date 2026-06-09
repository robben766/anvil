"""Context engineering: estimate token budget + structurally-safe compaction.

Compaction only TRUNCATES the content of old tool-result messages — it never drops or
reorders messages, so tool_call/tool pairing stays valid. This is best-effort: if the
recent window itself exceeds budget it may not fully fit, which is the honest behavior.
Tier 2 (M6): if a summarizer callback is provided and budget is still exceeded after
truncation, the middle region is replaced with a single summary message. The cut is
aligned to a non-tool boundary so tool_call/tool pairing stays valid."""

from __future__ import annotations

from collections.abc import Callable
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
    summarizer: Callable[[list[Message]], str] | None = None,
) -> list[Message]:
    """Shrink messages toward max_tokens. Tier 1: truncate old tool-message content.
    Tier 2 (if summarizer given and still over budget): replace the middle region
    (everything after system+task, before the recent window) with a single summary
    message. The cut is aligned to a non-tool boundary so tool_call/tool pairing stays
    valid. Protects messages[0] (system), messages[1] (task), and the last keep_recent."""
    if estimate_tokens(messages) <= max_tokens:
        return messages
    n = len(messages)
    # Tier 1: truncate old tool outputs (structurally safe).
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
    # Tier 2: summarize the middle if still over budget and a summarizer is available.
    if summarizer is None or estimate_tokens(out) <= max_tokens:
        return out
    cut = n - keep_recent
    # back the cut up to a non-tool message so the kept tail starts cleanly (no orphan tool)
    while cut > 2 and out[cut].get("role") == "tool":
        cut -= 1
    if cut <= 2:
        return out  # nothing summarizable without breaking pairing
    middle = out[2:cut]
    summary = summarizer(middle)
    summary_msg = {"role": "user", "content": f"[Summary of earlier work]\n{summary}"}
    return out[:2] + [summary_msg] + out[cut:]


def llm_summarizer(model: str) -> Callable[[list[Message]], str]:
    """A summarizer backed by the gateway. compact() is sync but is called from inside the
    async agent loop, so we run the async chat() in a fresh event loop on a worker thread
    (calling run_until_complete on the already-running loop would raise)."""
    import asyncio
    import concurrent.futures

    from anvil_gateway import chat

    def summarize(middle: list[Message]) -> str:
        transcript = "\n".join(
            f"{m.get('role')}: {(m.get('content') or '')[:500]}" for m in middle
        )
        msgs = [
            {"role": "system", "content": "Summarize the assistant's earlier work concisely "
             "(files read, edits made, test results) in 3-5 bullet points."},
            {"role": "user", "content": transcript},
        ]

        def _call() -> str:
            resp = asyncio.run(chat(model, msgs))
            return resp.content or "(no summary)"

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_call).result()

    return summarize
