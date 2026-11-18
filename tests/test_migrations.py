"""Alembic migrations: they must build exactly the schema the ORM models describe."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, inspect, text

from football_predictor.cli import main
from football_predictor.db import migrate
from football_predictor.db.models import Base, Team
from football_predictor.db.session import create_db_engine, init_db


@pytest.fixture()
def engine(tmp_path: Path) -> Iterator[Engine]:
    eng = create_db_engine(f"sqlite:///{tmp_path / 'm.db'}")
    yield eng
    eng.dispose()


def test_single_head_and_linear_history() -> None:
    script = migrate.ScriptDirectory.from_config(migrate.alembic_config())
    assert len(script.get_heads()) == 1


def test_upgrade_matches_models(engine: Engine) -> None:
    """Guard: a model change without a migration fails here (run `db revision`)."""
    migrate.upgrade(engine)
    assert migrate.current_revision(engine) == migrate.head_revision()
    assert migrate.schema_diff(engine) == []


def test_downgrade_to_base_and_back(engine: Engine) -> None:
    migrate.upgrade(engine)
    migrate.downgrade(engine, "base")
    assert set(inspect(engine).get_table_names()) == {migrate.VERSION_TABLE}
    migrate.upgrade(engine)
    assert migrate.schema_diff(engine) == []


def test_legacy_create_all_database_is_stamped_and_keeps_data(engine: Engine) -> None:
    Base.metadata.create_all(engine)  # how databases were created before migrations
    with engine.begin() as conn:
        conn.execute(Team.__table__.insert().values(name="Example FC"))
    assert migrate.current_revision(engine) is None

    init_db(engine)
    assert migrate.current_revision(engine) == migrate.head_revision()
    with engine.connect() as conn:
        assert conn.execute(text("SELECT name FROM team")).scalar_one() == "Example FC"


def test_legacy_database_with_different_schema_is_refused(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE team ADD COLUMN stadium VARCHAR(50)"))
    with pytest.raises(migrate.SchemaMismatchError):
        init_db(engine)


def test_init_db_is_idempotent_and_drop_resets(engine: Engine) -> None:
    init_db(engine)
    with engine.begin() as conn:
        conn.execute(Team.__table__.insert().values(name="Example FC"))
    init_db(engine)  # no-op
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM team")).scalar_one() == 1
    init_db(engine, drop_existing=True)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM team")).scalar_one() == 0
    assert migrate.current_revision(engine) == migrate.head_revision()


def test_cli_db_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import football_predictor.db.session as db_session
    from football_predictor.config import get_settings

    monkeypatch.setenv("FP_DATABASE_URL", f"sqlite:///{tmp_path / 'cli.db'}")
    get_settings.cache_clear()
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_session_factory", None)
    try:
        assert main(["db", "upgrade"]) == 0
        assert main(["db", "check"]) == 0
        assert main(["db", "current"]) == 0
        assert main(["db", "downgrade"]) == 2  # target required
        assert main(["db", "downgrade", "base"]) == 0
        assert main(["db", "check"]) == 1  # tables gone -> differs from models
    finally:
        db_session.get_engine().dispose()
        get_settings.cache_clear()
