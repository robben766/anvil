"""Run an async coroutine to completion from inside a synchronous tool, even when an
outer event loop is already running. Same trick code-agent M6 uses for the summarizer:
hand the coroutine to a worker thread that owns its own asyncio.run."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def block_on(coro: Coroutine[Any, Any, Any]) -> Any:
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()
