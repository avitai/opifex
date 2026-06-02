"""Async persistence backend for the Neural Functional Registry.

This module owns the SQLAlchemy *infrastructure* concerns for the registry:
constructing the async engine, creating the schema, and handing out sessions.
The domain service (:mod:`opifex.platform.registry.core`) depends only on the
:class:`RegistryDatabase` abstraction and the ORM models — never on a concrete
driver — keeping the layering contract intact.

The default backend is SQLite via the ``aiosqlite`` async driver
(``sqlite+aiosqlite``), which makes the registry usable out of the box without
a running database server while persisting state to a single file. The same
ORM transparently targets PostgreSQL (``postgresql+asyncpg``) for production
deployments because the column types in
:mod:`opifex.platform.registry.models` are dialect-agnostic.

No connection is opened at import time: the engine is created lazily on first
use and torn down via :meth:`RegistryDatabase.close`.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    AsyncEngine,
    AsyncSession,
    create_async_engine,
)

from opifex.platform.registry.models import Base


logger = logging.getLogger(__name__)

#: Default async SQLite driver URL scheme.
_SQLITE_ASYNC_SCHEME = "sqlite+aiosqlite"


def default_sqlite_url(storage_path: str | Path) -> str:
    """Build the default async SQLite URL for a storage directory.

    Args:
        storage_path: Directory under which ``registry.db`` is placed.

    Returns:
        A ``sqlite+aiosqlite:///<abs-path>/registry.db`` URL.
    """
    db_file = (Path(storage_path) / "registry.db").resolve()
    return f"{_SQLITE_ASYNC_SCHEME}:///{db_file}"


class RegistryDatabase:
    """Lazily-initialized async database backend for the registry.

    Wraps an :class:`~sqlalchemy.ext.asyncio.AsyncEngine` and an
    :class:`~sqlalchemy.ext.asyncio.async_sessionmaker`. The engine is created
    on first use; :meth:`create_schema` materializes the ORM tables; and
    :meth:`session` yields a transactional :class:`AsyncSession` scoped to an
    ``async with`` block.
    """

    def __init__(self, database_url: str) -> None:
        """Initialize the backend.

        Args:
            database_url: A SQLAlchemy *async* URL (e.g.
                ``sqlite+aiosqlite:///…`` or ``postgresql+asyncpg://…``).

        Raises:
            ValueError: If ``database_url`` is empty or whitespace-only.
        """
        if not database_url or not database_url.strip():
            raise ValueError("database_url must be a non-empty SQLAlchemy async URL")
        self._database_url = database_url
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._schema_ready = False

    @property
    def database_url(self) -> str:
        """Return the configured async database URL."""
        return self._database_url

    def _ensure_engine(self) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
        """Create the engine and session factory on first use (lazy)."""
        if self._engine is None or self._session_factory is None:
            self._engine = create_async_engine(self._database_url, future=True)
            self._session_factory = async_sessionmaker(bind=self._engine, expire_on_commit=False)
            logger.info("Initialized RegistryDatabase engine for %s", self._database_url)
        return self._engine, self._session_factory

    async def create_schema(self) -> None:
        """Create all registry tables if they do not already exist."""
        engine, _ = self._ensure_engine()
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        self._schema_ready = True

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a transactional :class:`AsyncSession`.

        The session commits on clean exit and rolls back on exception, then
        is always closed. The schema is created on first access so callers
        never operate against missing tables.

        Yields:
            An open :class:`~sqlalchemy.ext.asyncio.AsyncSession`.
        """
        if not self._schema_ready:
            await self.create_schema()
        _, session_factory = self._ensure_engine()
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def close(self) -> None:
        """Dispose of the engine and release all pooled connections."""
        if self._engine is not None:
            await self._engine.dispose()
            logger.info("Disposed RegistryDatabase engine for %s", self._database_url)
            self._engine = None
            self._session_factory = None
            self._schema_ready = False
