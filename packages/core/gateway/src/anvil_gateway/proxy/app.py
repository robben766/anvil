"""OpenAI 兼容 proxy 薄壳:同一套 chat() 核心,SDK/HTTP 双形态。

启动:uvicorn anvil_gateway.proxy.app:app --port 8400
鉴权:设置 ANVIL_PROXY_API_KEY 后要求 Authorization: Bearer <key>;未设置放行(本地)。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from anvil_gateway import chat
from anvil_gateway.errors import (
    AllProvidersFailedError,
    FatalAuthError,
    FatalRequestError,
)
from anvil_gateway.types import ChatResponse

app = FastAPI(title="anvil gateway proxy")


def _check_auth(authorization: str | None) -> None:
    expected = os.environ.get("ANVIL_PROXY_API_KEY")
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid api key")


def _to_openai(resp: ChatResponse) -> dict[str, Any]:
    u = resp.usage
    return {
        "id": u.request_id or "anvil-proxy",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": resp.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": resp.content,
                    **({"tool_calls": resp.tool_calls} if resp.tool_calls else {}),
                },
                "finish_reason": resp.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.prompt_tokens + u.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": u.cached_tokens},
        },
    }


async def _stream_response(body: dict[str, Any], kwargs: dict[str, Any]) -> StreamingResponse:
    async def gen():
        chunks = await chat(body["model"], body["messages"], stream=True, **kwargs)
        async for c in chunks:
            if c.delta or c.finish_reason is not None:
                payload: dict[str, Any] = {
                    "id": "anvil-proxy-stream",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": ({"content": c.delta} if c.delta else {}),
                            "finish_reason": c.finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if c.usage is not None:
                u = c.usage
                usage_payload: dict[str, Any] = {
                    "id": "anvil-proxy-stream",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": body["model"],
                    "choices": [],
                    "usage": {
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                        "total_tokens": u.prompt_tokens + u.completion_tokens,
                        "prompt_tokens_details": {"cached_tokens": u.cached_tokens},
                    },
                }
                yield f"data: {json.dumps(usage_payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request, authorization: str | None = Header(default=None)
):
    _check_auth(authorization)
    body = await request.json()
    kwargs = {
        "session_id": body.get("user"),
        "temperature": body.get("temperature"),
        "max_tokens": body.get("max_tokens"),
        "tools": body.get("tools"),
        "response_format": body.get("response_format"),
    }
    try:
        if body.get("stream"):
            return await _stream_response(body, kwargs)
        resp = await chat(body["model"], body["messages"], **kwargs)
        return _to_openai(resp)
    except FatalRequestError as e:
        return JSONResponse(status_code=400, content={"error": {"message": str(e)}})
    except FatalAuthError as e:
        return JSONResponse(status_code=401, content={"error": {"message": str(e)}})
    except AllProvidersFailedError as e:
        return JSONResponse(status_code=502, content={"error": {"message": str(e)}})
