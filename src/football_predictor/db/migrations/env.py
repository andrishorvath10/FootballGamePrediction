"""Alembic environment.

The database URL comes from (in order): an open connection passed in by
:mod:`football_predictor.db.migrate`, ``sqlalchemy.url`` in the Alembic config,
or the application settings (``FP_DATABASE_URL`` / ``.env``).
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import Connection

from football_predictor.config import get_settings
from football_predictor.db.models import Base
from football_predictor.db.session import create_db_engine

config = context.config
target_metadata = Base.metadata


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # SQLite cannot ALTER most things; batch mode rebuilds the table instead.
        render_as_batch=connection.dialect.name == "sqlite",
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_offline() -> None:
    """Emit SQL to stdout instead of executing it (``alembic upgrade head --sql``)."""
    url = config.get_main_option("sqlalchemy.url") or get_settings().database_url
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True,
                      render_as_batch=url.startswith("sqlite"))
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _configure(connection)
        return
    url = config.get_main_option("sqlalchemy.url") or get_settings().database_url
    engine = create_db_engine(url)
    try:
        with engine.connect() as conn:
            _configure(conn)
            conn.commit()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
