"""The KB reporter's ACI. P3 @tool protocol (sync fn); async DB/retrieval bridged via
block_on. Tools capture an EmployeeContext at registry-build time."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from anvil_code_agent.tools.base import Tool, ToolContext, ToolResult, tool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.asyncbridge import block_on
from anvil_ai_employee.memory.store import MemoryStore
from anvil_ai_employee.scheduler.queue import complete
from anvil_kb.db import DocumentRow


@dataclass
class EmployeeContext:
    session_factory: async_sessionmaker[AsyncSession]
    employee: str
    job_id: Any  # uuid.UUID | None
    _retriever: Any = None  # lazily built dense Retriever


def _get_retriever(ctx: EmployeeContext):
    if ctx._retriever is None:
        from anvil_kb.embed import FastEmbedEmbedder
        from anvil_kb.retrieve.retriever import Retriever
        from anvil_kb.store.pg import PgVectorStore

        store = PgVectorStore(ctx.session_factory)
        ctx._retriever = Retriever(FastEmbedEmbedder(), store, mode="dense")
    return ctx._retriever


def build_employee_tools(ctx: EmployeeContext) -> list[Tool]:
    async def _recent(since: datetime) -> list[DocumentRow]:
        async with ctx.session_factory() as s:
            return list(
                (
                    await s.execute(
                        select(DocumentRow)
                        .where(DocumentRow.created_at > since)
                        .order_by(DocumentRow.created_at)
                    )
                )
                .scalars()
                .all()
            )

    @tool(
        name="recall_marker",
        description="回忆上次周报覆盖到的时间点(ISO)。没有则提示从未报告。先调它确定起点。",
        params={},
        required=[],
    )
    def recall_marker(args: dict, tc: ToolContext) -> ToolResult:
        last = block_on(
            MemoryStore(ctx.session_factory).last(
                employee=ctx.employee, kind="report_marker"
            )
        )
        if not last:
            return ToolResult(content="从未报告过;请覆盖最近 7 天的新增文档。", ok=True)
        return ToolResult(content=f"上次报告标记: {last}", ok=True)

    @tool(
        name="kb_recent",
        description="列出 since_iso 之后新入库的知识库文档(title/source/created_at + 内容预览)。",
        params={"since_iso": {"type": "string", "description": "ISO8601 起始时间"}},
        required=["since_iso"],
    )
    def kb_recent(args: dict, tc: ToolContext) -> ToolResult:
        since = datetime.fromisoformat(args["since_iso"])
        docs = block_on(_recent(since))
        if not docs:
            return ToolResult(content="自该时间点起无新增文档。", ok=True)
        lines = []
        for d in docs:
            preview = (d.content or "")[:200].replace("\n", " ")
            lines.append(
                f"- [{d.created_at.isoformat()}] {d.title} (source={d.source_name})\n  {preview}"
            )
        return ToolResult(content="新增文档:\n" + "\n".join(lines), ok=True)

    @tool(
        name="kb_search",
        description="对某主题在知识库做语义检索(dense),返回 top-k 片段。用于深入某条目。",
        params={
            "query": {"type": "string"},
            "k": {"type": "integer", "description": "默认 5"},
        },
        required=["query"],
    )
    def kb_search(args: dict, tc: ToolContext) -> ToolResult:
        k = int(args.get("k", 5))
        scored = block_on(_get_retriever(ctx).retrieve(args["query"], k=k))
        if not scored:
            return ToolResult(content="无检索结果。", ok=True)
        out = "\n---\n".join(f"score={s.score:.3f}\n{s.chunk.content}" for s in scored)
        return ToolResult(content=out, ok=True)

    @tool(
        name="submit_report",
        description=(
            "提交最终周报 markdown。covered_until_iso 取你这次见过的最大文档时间。提交即完成。"
        ),
        params={
            "markdown": {"type": "string"},
            "covered_until_iso": {"type": "string"},
        },
        required=["markdown", "covered_until_iso"],
    )
    def submit_report(args: dict, tc: ToolContext) -> ToolResult:
        markdown = args["markdown"]
        covered = args["covered_until_iso"]

        async def _persist() -> None:
            if ctx.job_id is not None:
                await complete(ctx.session_factory, ctx.job_id, result=markdown)
            await MemoryStore(ctx.session_factory).write(
                employee=ctx.employee,
                kind="report_marker",
                content=json.dumps(
                    {"covered_until": covered, "summary_head": markdown[:120]},
                    ensure_ascii=False,
                ),
            )

        block_on(_persist())
        return ToolResult(content="报告已提交。", ok=True)

    return [recall_marker, kb_recent, kb_search, submit_report]
