"""Letta/MemGPT self-managed memory tools. The agent calls these itself (contrast mem0,
where the orchestrator manages memory). Sync @tool protocol; async DB/vec via block_on."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anvil_code_agent.tools.base import Tool, ToolContext, ToolResult, tool

from anvil_ai_employee.asyncbridge import block_on
from anvil_ai_employee.memory.coreblocks import CoreBlockStore
from anvil_ai_employee.memory.store import MemoryStore
from anvil_ai_employee.memory.vectorstore import MemoryVectorStore
from anvil_ai_employee.sessions import SessionStore


@dataclass
class LettaToolContext:
    session_factory: Any
    embedder: Any
    employee: str
    session_id: Any  # uuid | None
    char_limit: int = 500
    archival_k: int = 5


def build_letta_tools(ctx: LettaToolContext) -> list[Tool]:
    cb = CoreBlockStore(ctx.session_factory, char_limit=ctx.char_limit)
    store = MemoryStore(ctx.session_factory)
    vs = MemoryVectorStore(ctx.session_factory)
    ss = SessionStore(ctx.session_factory)

    @tool(
        name="core_memory_append",
        description="往 core memory 块(label=persona/human)追加一行文本。超长度上限会失败。",
        params={"label": {"type": "string"}, "text": {"type": "string"}},
        required=["label", "text"],
    )
    def core_memory_append(args, tc: ToolContext) -> ToolResult:
        ok = block_on(cb.append(employee=ctx.employee, label=args["label"], text=args["text"]))
        if not ok:
            return ToolResult(
                content="append 失败:超出 core 块长度上限,请改存 archival_insert。", ok=False
            )
        return ToolResult(content="core 块已追加。", ok=True)

    @tool(
        name="core_memory_replace",
        description="在 core memory 块内把子串 old 替换成 new。old 不存在会失败。",
        params={
            "label": {"type": "string"},
            "old": {"type": "string"},
            "new": {"type": "string"},
        },
        required=["label", "old", "new"],
    )
    def core_memory_replace(args, tc: ToolContext) -> ToolResult:
        ok = block_on(
            cb.replace(employee=ctx.employee, label=args["label"], old=args["old"], new=args["new"])
        )
        if not ok:
            return ToolResult(content="replace 失败:old 不在块内或超长。", ok=False)
        return ToolResult(content="core 块已更新。", ok=True)

    @tool(
        name="archival_insert",
        description="把一段文本存入 archival 长期记忆(向量化,可日后检索)。",
        params={"text": {"type": "string"}},
        required=["text"],
    )
    def archival_insert(args, tc: ToolContext) -> ToolResult:
        emb = ctx.embedder.embed_texts([args["text"]])[0]
        block_on(
            store.insert(
                employee=ctx.employee, kind="archival", content=args["text"], embedding=emb
            )
        )
        return ToolResult(content="已存入 archival。", ok=True)

    @tool(
        name="archival_search",
        description="按语义检索 archival 长期记忆,返回最相关的若干条。",
        params={"query": {"type": "string"}},
        required=["query"],
    )
    def archival_search(args, tc: ToolContext) -> ToolResult:
        qv = ctx.embedder.embed_query(args["query"])
        hits = block_on(
            vs.knn(employee=ctx.employee, kinds=["archival"], query_vec=qv, k=ctx.archival_k)
        )
        if not hits:
            return ToolResult(content="archival 无匹配。", ok=True)
        return ToolResult(content="\n".join(f"- {row.content}" for row, _ in hits), ok=True)

    @tool(
        name="conversation_search",
        description="在本次会话的历史消息里按关键词检索过去说过的话(recall 记忆)。",
        params={"query": {"type": "string"}},
        required=["query"],
    )
    def conversation_search(args, tc: ToolContext) -> ToolResult:
        if ctx.session_id is None:
            return ToolResult(content="无会话历史可检索。", ok=True)
        msgs = block_on(ss.load(ctx.session_id))
        q = args["query"]
        hits = [m for m in msgs if isinstance(m.get("content"), str) and q in m["content"]]
        if not hits:
            return ToolResult(content="对话史无匹配。", ok=True)
        return ToolResult(
            content="\n".join(f"- {m['role']}: {m['content']}" for m in hits), ok=True
        )

    return [
        core_memory_append,
        core_memory_replace,
        archival_insert,
        archival_search,
        conversation_search,
    ]
