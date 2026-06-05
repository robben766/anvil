"""演示:自研 span + OTLP 导出到 Langfuse。一个业务函数内调两次 chat(),形成调用树。"""

import asyncio
import os

from anvil_obs.exporter import OtlpExporter
from anvil_obs.span import span

from anvil_gateway import chat


async def explain_then_summarize(topic: str) -> None:
    with span(f"task {topic}"):
        r1 = await chat(
            "deepseek-chat",
            [{"role": "user", "content": f"用两句话解释{topic}"}],
            session_id="demo",
        )
        r2 = await chat(
            "deepseek-chat",
            [{"role": "user", "content": f"把这段话压缩成一句:{r1.content}"}],
            session_id="demo",
        )
        print(r2.content)


async def main() -> None:
    exporter = OtlpExporter(
        endpoint=os.environ["LANGFUSE_OTLP_ENDPOINT"] + "/v1/traces",
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    )
    await explain_then_summarize("模型网关")
    await exporter.aclose()  # 收尾 flush


asyncio.run(main())
