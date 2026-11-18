"""Application configuration.

Settings are read from environment variables (prefix ``FP_``). A ``.env`` file in
the working directory is loaded first, but real environment variables always win.
This lets you switch e.g. from SQLite to PostgreSQL by changing ``FP_DATABASE_URL``
without touching any code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_DATABASE_URL = "sqlite:///data/football.db"


def _as_bool(value: str) -> bool:
    """Interpret common truthy strings ("1", "true", "yes", "on") as True."""
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable container for all runtime settings."""

    database_url: str = DEFAULT_DATABASE_URL
    raw_data_dir: Path = Path("data/raw")
    log_level: str = "INFO"
    sql_echo: bool = False
    anonymize: bool = False  # dashboard: start with player/team names hidden

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables (after loading ``.env``)."""
        load_dotenv(override=False)  # never override variables already set
        return cls(
            database_url=os.getenv("FP_DATABASE_URL", DEFAULT_DATABASE_URL),
            raw_data_dir=Path(os.getenv("FP_RAW_DATA_DIR", "data/raw")),
            log_level=os.getenv("FP_LOG_LEVEL", "INFO").upper(),
            sql_echo=_as_bool(os.getenv("FP_SQL_ECHO", "false")),
            anonymize=_as_bool(os.getenv("FP_ANONYMIZE", "false")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings object (read once, then cached)."""
    return Settings.from_env()
