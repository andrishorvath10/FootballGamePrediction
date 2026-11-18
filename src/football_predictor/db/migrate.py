"""Schema migrations with Alembic.

The ORM models in :mod:`football_predictor.db.models` are the source of truth;
Alembic revisions in ``db/migrations/versions`` record how the schema got there,
so an existing database (with real data) can be upgraded in place instead of
being dropped and re-scraped.

Workflow when a model changes::

    football-predictor db revision -m "add referee to match"   # autogenerate
    # review the generated file, then
    football-predictor db upgrade

Databases created before migrations existed (``create_all``) have the tables but
no ``alembic_version`` row; :func:`ensure_schema` checks that their schema
matches the models and *stamps* them at the head revision.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, inspect

from football_predictor.db.models import Base

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).with_name("migrations")
VERSION_TABLE = "alembic_version"


class SchemaMismatchError(RuntimeError):
    """An unversioned database whose tables do not match the current models."""


def alembic_config(engine: Engine | None = None) -> Config:
    """Alembic config pointing at the packaged migrations (no alembic.ini needed)."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("file_template", "%%(year)d%%(month).2d%%(day).2d_%%(rev)s_%%(slug)s")
    if engine is not None:
        # '%' must be escaped for configparser interpolation.
        url = engine.url.render_as_string(hide_password=False).replace("%", "%%")
        cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def _run(engine: Engine, fn: Any, *args: Any, **kwargs: Any) -> None:
    """Run an Alembic command on ``engine`` through a single shared connection.

    Sharing the connection also makes in-memory SQLite work (each new
    connection would otherwise see an empty database).
    """
    cfg = alembic_config(engine)
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        fn(cfg, *args, **kwargs)


def head_revision() -> str | None:
    """The newest revision in the migrations directory."""
    return ScriptDirectory.from_config(alembic_config()).get_current_head()


def current_revision(engine: Engine) -> str | None:
    """The revision the database is at (None if unversioned or empty)."""
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def schema_diff(engine: Engine) -> list[Any]:
    """Differences between the live database and the ORM models (empty = in sync)."""
    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"compare_type": True})
        return compare_metadata(ctx, Base.metadata)


def upgrade(engine: Engine, revision: str = "head") -> None:
    """Apply migrations up to ``revision``."""
    _run(engine, command.upgrade, revision)


def downgrade(engine: Engine, revision: str) -> None:
    """Revert migrations down to ``revision`` (``"base"`` = empty schema)."""
    _run(engine, command.downgrade, revision)


def stamp(engine: Engine, revision: str = "head") -> None:
    """Record ``revision`` as applied without running any migration."""
    _run(engine, command.stamp, revision)


def ensure_schema(engine: Engine) -> None:
    """Bring ``engine``'s database to the head revision.

    * empty database -> run every migration;
    * versioned database -> run the pending ones (no-op when up to date);
    * legacy ``create_all`` database -> stamp head if its schema matches the models,
      otherwise raise :class:`SchemaMismatchError` rather than guess.
    """
    tables = set(inspect(engine).get_table_names())
    app_tables = set(Base.metadata.tables)
    if tables & app_tables and VERSION_TABLE not in tables:
        diff = schema_diff(engine)
        if diff:
            raise SchemaMismatchError(
                "Database has tables but no migration history, and its schema differs "
                f"from the models: {diff}. Back it up and recreate it with "
                "`football-predictor init-db --drop`, or stamp it manually."
            )
        logger.info("Existing schema matches the models; stamping it at %s", head_revision())
        stamp(engine)
        return
    before = current_revision(engine)
    upgrade(engine)
    after = current_revision(engine)
    if before != after:
        logger.info("Migrated database from %s to %s", before or "empty", after)


def autogenerate_revision(engine: Engine, message: str) -> None:
    """Write a new revision file with the operations needed to reach the models.

    ``engine`` must be at head, otherwise Alembic refuses (the diff would mix
    pending migrations with the new changes).
    """
    _run(engine, command.revision, message=message, autogenerate=True)
