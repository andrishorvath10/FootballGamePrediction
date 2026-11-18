"""Engine and session management.

Usage::

    from football_predictor.db.session import session_scope

    with session_scope() as session:
        session.add(Team(name="Example FC"))
    # committed here (or rolled back on exception)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from football_predictor.config import get_settings
from football_predictor.db.models import Base

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
    """SQLite ignores FOREIGN KEY constraints unless this PRAGMA is set per connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_db_engine(database_url: str, echo: bool = False) -> Engine:
    """Create an engine for ``database_url``.

    For file-based SQLite it also creates the parent directory and turns on
    foreign-key enforcement. Other back-ends (PostgreSQL) need nothing special.
    """
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite":
        if url.database and url.database != ":memory:":
            Path(url.database).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, echo=echo)
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    else:
        engine = create_engine(url, echo=echo, pool_pre_ping=True)
    return engine


def get_engine() -> Engine:
    """Return the process-wide engine built from the configured database URL."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_db_engine(settings.database_url, echo=settings.sql_echo)
        logger.debug("Created engine for %s", _engine.url.render_as_string(hide_password=True))
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide ``sessionmaker`` bound to :func:`get_engine`."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


@contextmanager
def session_scope(factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on any exception."""
    session = (factory or get_session_factory())()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine | None = None, drop_existing: bool = False) -> None:
    """Bring the schema up to date via Alembic migrations (idempotent).

    An empty database gets every migration, an older one only the pending ones,
    and a legacy ``create_all`` database is stamped (see
    :func:`football_predictor.db.migrate.ensure_schema`).

    Args:
        engine: Engine to use; defaults to the configured one.
        drop_existing: If True, drop all tables first (DESTROYS DATA).
    """
    from football_predictor.db import migrate  # imports this module via env.py

    engine = engine or get_engine()
    if drop_existing:
        logger.warning("Dropping all tables on %s", engine.url.render_as_string(hide_password=True))
        Base.metadata.drop_all(engine)
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {migrate.VERSION_TABLE}"))
    migrate.ensure_schema(engine)
    logger.info(
        "Database ready (revision %s) at %s",
        migrate.current_revision(engine),
        engine.url.render_as_string(hide_password=True),
    )
