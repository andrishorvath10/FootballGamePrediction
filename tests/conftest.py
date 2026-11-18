"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy.orm import Session, sessionmaker

from football_predictor.db.session import create_db_engine, init_db


@pytest.fixture()
def session() -> Iterator[Session]:
    """A fresh in-memory SQLite database (with FK enforcement) per test."""
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    factory = sessionmaker(bind=engine)
    with factory() as s:
        yield s
    engine.dispose()
