"""Mem0Strategy — the mem0 memory philosophy: the *orchestrator* manages memory,
the agent never knows memory exists.

Two hooks carry the philosophy:
- ``system_prefix``: before a turn, vector-recall the user's known facts and inject
  them into the system block (recall side → ``embed_query``, retrieval prefix).
- ``after_turn``: after a turn, LLM-extract discrete facts → for each fact find the
  semantic near-neighbours → LLM-reconcile into ADD/UPDATE/DELETE/NOOP → apply to the
  vector store (write side → ``embed_texts``, passage, no prefix).

embed direction is a hard rule: writes (ADD/UPDATE) use ``embed_texts`` (passage), recall
and neighbour-finding use ``embed_query`` (query). Swap them and cosine recall collapses.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from anvil_code_agent.tools.base import ToolRegistry
from anvil_guard import StructuredOutputError, structured_chat
from anvil_obs import span

from anvil_ai_employee.memory.store import MemoryStore
from anvil_ai_employee.memory.vectorstore import MemoryVectorStore

_VALID_OPS = {"ADD", "UPDATE", "DELETE", "NOOP"}

EXTRACT_PROMPT = (
    "你是一个记忆抽取器。给你一段对话(json 数组,每条含 role/content),"
    "请抽取其中关于【用户本人】的离散、稳定的事实(如居住地、职业、偏好、人际关系等),"
    "忽略助手的话和一次性的闲聊。每条事实写成一句独立、自包含的中文陈述。"
    '只返回形如 {"facts": ["事实1", "事实2"]} 的 json 对象;没有可抽取的事实时返回 {"facts": []}。'
)

RECONCILE_PROMPT = (
    "你是一个记忆调和器。给你一条【新事实】和若干条【已有记忆】(各带 id 与 content),"
    "请判断如何把新事实并入记忆库,从以下操作中选一个:\n"
    "- ADD:新事实是全新信息,已有记忆里没有它。\n"
    "- UPDATE:新事实与某条已有记忆讲的是同一件事但内容有更新/矛盾(如搬家),"
    "需带上该记忆的 target_id。\n"
    "- DELETE:新事实表明某条已有记忆已不再成立,需带上该记忆的 target_id。\n"
    "- NOOP:新事实与某条已有记忆等价或无新增信息,无需改动。\n"
    '只返回形如 {"op": "ADD"} 或 {"op": "UPDATE", "target_id": "<id>"} 的 json 对象。'
)


def _reconcile_user(fact: str, neighbors: list[tuple[Any, float]]) -> str:
    lines = [f"新事实:{fact}", "已有记忆:"]
    if neighbors:
        for row, _score in neighbors:
            lines.append(f"- id={row.id} content={row.content}")
    else:
        lines.append("(无)")
    lines.append('请按要求返回 json。')
    return "\n".join(lines)


class Mem0Strategy:
    """Managed (mem0) memory: orchestrator-driven recall + extract/reconcile/apply."""

    def __init__(
        self,
        session_factory,
        *,
        embedder,
        model: str,
        recall_k: int = 5,
        neighbor_k: int = 5,
        window: int = 10,
    ):
        self._store = MemoryStore(session_factory)
        self._vs = MemoryVectorStore(session_factory)
        self._embedder = embedder
        self._model = model
        self._recall_k = recall_k
        self._neighbor_k = neighbor_k
        self._window = window
        # warm the embedder once so the first turn doesn't pay model-load latency.
        try:
            self._embedder.embed_query("warmup")
        except Exception:  # noqa: BLE001 — warm-up is best-effort, never fatal
            pass

    def build_registry(self, ctx: Any) -> ToolRegistry:
        # mem0 backend agent holds no memory tools (pure chat) — memory lives in the orchestrator.
        return ToolRegistry([])

    async def system_prefix(self, employee: str, user_msg: str) -> str:
        qv = self._embedder.embed_query(user_msg)
        hits = await self._vs.knn(
            employee=employee, kinds=["fact"], query_vec=qv, k=self._recall_k
        )
        if not hits:
            return ""
        return "# 关于用户你已知道:\n" + "\n".join(f"- {row.content}" for row, _ in hits)

    async def after_turn(self, employee: str, session: Any, msgs: list[dict]) -> None:
        with span("ai_employee.mem0.after_turn"):
            window = self._build_window(session, msgs)
            try:
                extract = await structured_chat(
                    self._model,
                    [
                        {"role": "system", "content": EXTRACT_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(window, ensure_ascii=False),
                        },
                    ],
                    schema={"required": ["facts"]},
                )
            except StructuredOutputError as e:
                with span("ai_employee.mem0.extract_failed", error=str(e)):
                    pass
                return

            facts = extract.get("facts") or []
            for fact in facts:
                await self._reconcile_one(employee, fact)

    def _build_window(self, session: Any, msgs: list[dict]) -> list[dict]:
        """Extraction context = recent session messages (if any) + the current pair.

        M2a keeps it simple: the current turn's msgs are the primary signal; if a session
        carries a ``messages`` list we tail-prepend up to ``window`` of them for context.
        Rolling summary S is deferred (spiral)."""
        prior: list[dict] = []
        sess_msgs = getattr(session, "messages", None)
        if isinstance(sess_msgs, list) and sess_msgs:
            prior = list(sess_msgs)[-self._window :]
        return prior + list(msgs)

    async def _reconcile_one(self, employee: str, fact: str) -> None:
        nb = await self._vs.knn(
            employee=employee,
            kinds=["fact"],
            query_vec=self._embedder.embed_query(fact),
            k=self._neighbor_k,
        )
        try:
            dec = await structured_chat(
                self._model,
                [
                    {"role": "system", "content": RECONCILE_PROMPT},
                    {"role": "user", "content": _reconcile_user(fact, nb)},
                ],
                schema={"required": ["op"]},
            )
        except StructuredOutputError as e:
            with span("ai_employee.mem0.reconcile_failed", error=str(e)):
                pass
            return

        # structured_chat only checks required-key presence, not enum values — validate here.
        op = str(dec.get("op", "")).upper()
        target_id = dec.get("target_id")
        valid_ids = {str(row.id) for row, _ in nb}

        if op not in _VALID_OPS:
            with span("ai_employee.mem0.illegal_op", op=op):
                pass
            return  # NOOP fallback
        if op in {"UPDATE", "DELETE"} and (target_id is None or str(target_id) not in valid_ids):
            with span("ai_employee.mem0.bad_target", op=op, target_id=str(target_id)):
                pass
            return  # NOOP fallback

        if op == "ADD":
            await self._store.insert(
                employee=employee,
                kind="fact",
                content=fact,
                embedding=self._embedder.embed_texts([fact])[0],
            )
        elif op == "UPDATE":
            await self._store.update(
                uuid.UUID(str(target_id)),
                content=fact,
                embedding=self._embedder.embed_texts([fact])[0],
            )
        elif op == "DELETE":
            await self._store.delete(uuid.UUID(str(target_id)))
        # op == "NOOP": nothing to do.
