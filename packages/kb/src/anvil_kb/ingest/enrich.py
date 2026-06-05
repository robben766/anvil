"""Contextual Retrieval(Anthropic):chunk 前缀定位上下文。

经济学:system 放完整文档 → 同文档所有 chunk 共享缓存前缀(DeepSeek 实测 >1K token
前缀命中 ~96%,缓存价 1/50)→ 富集边际成本极低。
"""

from __future__ import annotations

from anvil_kb.ingest.chunker import ChunkDraft

_INSTRUCTION = (
    "上面是完整文档。请为下面这段文字写 1-2 句定位上下文,说明它在文档中的位置与主题,"
    "用于改进检索。直接输出上下文,不要任何前缀或解释。\n\n段落:\n{chunk}"
)


def _get_default_chat():
    """懒加载默认 chat(与 generate.py 同模式)。"""
    from anvil_gateway import chat as _default_chat  # noqa: PLC0415

    return _default_chat


async def enrich_chunks(
    *,
    doc_text: str,
    drafts: list[ChunkDraft],
    chat=None,
    session_id: str = "anvil-kb-enrich",
) -> list[str]:
    """为每个 ChunkDraft 生成定位上下文前缀。

    Args:
        doc_text:   完整文档文本(放入 system 消息,供跨 chunk 共享缓存)。
        drafts:     待富集的 ChunkDraft 列表。
        chat:       异步 chat 函数;None 时使用 anvil_gateway.chat。
        session_id: 透传至 chat 调用的 session_id。

    Returns:
        与 drafts 等长的字符串列表。单点异常 fail-open → 失败位置返回 ""。
    """
    if chat is None:
        chat = _get_default_chat()

    system_content = "以下是完整文档:\n" + doc_text
    contexts: list[str] = []

    for draft in drafts:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": _INSTRUCTION.format(chunk=draft.content)},
        ]
        try:
            resp = await chat(
                "chat-default",
                messages,
                session_id=session_id,
                temperature=0.3,
                max_tokens=120,
            )
            contexts.append(resp.content.strip() if resp.content else "")
        except Exception:  # noqa: BLE001 — fail-open by design
            contexts.append("")

    return contexts
