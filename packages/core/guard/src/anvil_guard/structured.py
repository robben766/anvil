"""Force a model to return a valid JSON object: native json_object mode + parse + retry-once.

Consolidates the parse/retry/fence-stripping logic that previously lived inline in
anvil_eval.judge. Callers must include the word "json" somewhere in their messages —
DeepSeek's json_object response_format requires it.
"""

from __future__ import annotations

import json
from typing import Any

from anvil_gateway import chat


class StructuredOutputError(ValueError):
    """Raised when the model fails to produce a valid JSON object after all retries."""


def _parse_json_object(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    obj = json.loads(text.strip())
    if not isinstance(obj, dict):
        raise ValueError("structured output is not a JSON object")
    return obj


async def structured_chat(
    model: str,
    messages: list[dict[str, Any]],
    *,
    schema: dict[str, Any] | None = None,
    max_retries: int = 1,
    **chat_kwargs: Any,
) -> dict[str, Any]:
    """Return a parsed JSON object from the model.

    schema: optional minimal JSON-schema subset; only ``schema["required"]`` (a list of
    keys that must be present) is enforced. Pass None to accept any JSON object.
    max_retries: number of *extra* attempts after the first (default 1 = up to 2 calls).
    Extra kwargs (temperature, session_id, ...) pass through to gateway chat().
    """
    required = list((schema or {}).get("required", []))
    response_format = chat_kwargs.pop("response_format", {"type": "json_object"})
    last_err: Exception | None = None
    for _ in range(max_retries + 1):
        resp = await chat(model, messages, response_format=response_format, **chat_kwargs)
        try:
            obj = _parse_json_object(resp.content or "")
            missing = [k for k in required if k not in obj]
            if missing:
                raise ValueError(f"missing required keys: {missing}")
            return obj
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
    raise StructuredOutputError(
        f"structured output invalid after {max_retries} retr(y/ies): {last_err}"
    )
