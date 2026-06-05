"""Citation-grounded generation — non-streaming and streaming answer helpers."""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from anvil_kb.store.base import ScoredChunk

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    n: int                  # 1-based reference number as it appears in the answer
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    quote: str              # full chunk content
    start_offset: int
    end_offset: int


@dataclass
class KbAnswer:
    text: str
    citations: list[Citation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = (
    "你是知识库问答助手。仅根据下列编号资料回答;引用资料处必须标注 [编号];"
    "若资料不足以回答,明确回答:资料中未找到相关内容。不得编造。\n\n{contexts}"
)


def _format_context(n: int, sc: ScoredChunk) -> str:
    """Format a single numbered context entry.

    With header_path : ``[1](条款 > 等待期) 等待期为90天。``
    Without          : ``[1] 等待期为90天。``
    """
    chunk = sc.chunk
    if chunk.header_path:
        return f"[{n}]({chunk.header_path}) {chunk.content}"
    return f"[{n}] {chunk.content}"


def build_messages(question: str, retrieved: list[ScoredChunk]) -> list[dict]:
    """Build the [system, user] message list for the LLM call."""
    contexts = "\n".join(_format_context(i + 1, sc) for i, sc in enumerate(retrieved))
    system_content = SYSTEM_TEMPLATE.format(contexts=contexts)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]


# ---------------------------------------------------------------------------
# Citation parsing
# ---------------------------------------------------------------------------


def parse_citations(text: str, retrieved: list[ScoredChunk]) -> list[Citation]:
    """Extract [n] references from *text*, filter to valid indices, deduplicate."""
    raw_numbers = re.findall(r"\[(\d+)\]", text)
    seen: set[int] = set()
    citations: list[Citation] = []
    for num_str in raw_numbers:
        n = int(num_str)
        if n < 1 or n > len(retrieved):
            continue
        if n in seen:
            continue
        seen.add(n)
        sc = retrieved[n - 1]
        chunk = sc.chunk
        citations.append(
            Citation(
                n=n,
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                quote=chunk.content,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
            )
        )
    return citations


# ---------------------------------------------------------------------------
# answer (non-streaming)
# ---------------------------------------------------------------------------


async def answer(
    question: str,
    retrieved: list[ScoredChunk],
    *,
    chat=None,
    session_id: str = "anvil-kb",
) -> KbAnswer:
    """Return a KbAnswer with citation-grounded text."""
    if chat is None:
        from anvil_gateway.chat import chat as _default_chat  # noqa: PLC0415

        chat = _default_chat

    messages = build_messages(question, retrieved)
    response = await chat("chat-default", messages, session_id=session_id)
    text: str = response.content or ""
    return KbAnswer(text=text, citations=parse_citations(text, retrieved))


# ---------------------------------------------------------------------------
# answer_stream
# ---------------------------------------------------------------------------


async def answer_stream(
    question: str,
    retrieved: list[ScoredChunk],
    *,
    chat=None,
    session_id: str = "anvil-kb",
) -> AsyncIterator[tuple[str, object]]:
    """Async generator producing a typed event stream.

    Yields:
        ("sources", list[ScoredChunk])   — once, immediately
        ("delta",   str)                 — one per streamed token chunk
        ("done",    KbAnswer)            — once, at the end
    """
    if chat is None:
        from anvil_gateway.chat import chat as _default_chat  # noqa: PLC0415

        chat = _default_chat

    yield ("sources", retrieved)

    messages = build_messages(question, retrieved)
    stream = await chat("chat-default", messages, stream=True, session_id=session_id)

    accumulated = ""
    async for chunk in stream:
        delta: str = chunk.delta
        if delta:
            accumulated += delta
            yield ("delta", delta)

    final_answer = KbAnswer(
        text=accumulated,
        citations=parse_citations(accumulated, retrieved),
    )
    yield ("done", final_answer)
