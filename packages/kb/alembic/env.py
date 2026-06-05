import asyncio
import os

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from anvil_kb.db import Base

target_metadata = Base.metadata


def _get_url() -> str:
    # Programmatic override via config (e.g. from conftest) takes precedence.
    cfg_url = context.config.get_main_option("sqlalchemy.url")
    if cfg_url:
        return cfg_url
    return os.environ.get(
        "ANVIL_DATABASE_URL", "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil"
    )


def run_migrations_offline() -> None:
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        version_table="kb_alembic_version",
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table="kb_alembic_version",
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = create_async_engine(_get_url())
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
