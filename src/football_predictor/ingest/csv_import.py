"""CSV import (for leagues without a usable website).

Files (UTF-8, comma or semicolon separated - detected automatically):

``matches.csv`` (required)
    competition, season, round, date, home_team, away_team, home_goals, away_goals,
    status, source_url
    Only ``round``, ``home_team``, ``away_team`` are mandatory (``season`` too, unless
    given on the command line; ``competition`` defaults to the --competition option).
    Empty goals = not played yet.

``goals.csv`` (optional)
    [competition,] season, round, home_team, away_team, team, player, minute, penalty, own_goal

``appearances.csv`` (optional)
    [competition,] season, round, home_team, away_team, team, player, minutes

The first four columns of the detail files identify the match. This module only
*parses*; writing to the database is done by :class:`LeagueLoader`.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from football_predictor.db.models import MatchStatus
from football_predictor.ingest.records import (
    DEFAULT_COMPETITION,
    AppearanceRecord,
    GoalRecord,
    MatchRecord,
    clean_name,
    name_key,
)

FixtureKey = tuple[str, str, int, str, str]
DATE_FORMATS = ("%Y-%m-%d", "%Y.%m.%d.", "%Y.%m.%d", "%d/%m/%Y")
TRUE_VALUES = {"1", "true", "yes", "y", "igen", "i", "x"}


class CsvImportError(ValueError):
    """Raised with *all* problems found in a file, so they can be fixed in one go."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"{len(errors)} CSV error(s):\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------- helpers
def _read_rows(path: Path) -> list[tuple[int, dict[str, str]]]:
    """Return (line number, row) pairs with normalised header names and stripped values."""
    text = Path(path).read_text(encoding="utf-8-sig")  # -sig: tolerate Excel's BOM
    first_line = text.splitlines()[0] if text else ""
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    reader.fieldnames = [h.strip().lower() for h in reader.fieldnames or []]
    rows = []
    for row in reader:
        cleaned = {k: (v or "").strip() for k, v in row.items() if k is not None}
        if any(cleaned.values()):  # skip blank lines
            rows.append((reader.line_num, cleaned))
    return rows


def _parse_int(value: str, field: str) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field}: not an integer: {value!r}") from exc


def _parse_date(value: str) -> date | None:
    if value == "":
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"date: unrecognised format {value!r} (use YYYY-MM-DD)")


def _parse_bool(value: str) -> bool:
    return value.casefold() in TRUE_VALUES


def _require(row: dict[str, str], field: str) -> str:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"missing required field {field!r}")
    return value


def _season(row: dict[str, str], default_season: str | None) -> str:
    season = row.get("season") or default_season
    if not season:
        raise ValueError("missing 'season' (column or --season option)")
    return season


def _competition(row: dict[str, str], default_competition: str) -> str:
    return row.get("competition") or default_competition


def _row_fixture_key(
    row: dict[str, str], default_season: str | None, default_competition: str
) -> FixtureKey:
    rnd = _parse_int(_require(row, "round"), "round")
    assert rnd is not None
    return (
        clean_name(_competition(row, default_competition)),
        clean_name(_season(row, default_season)),
        rnd,
        name_key(_require(row, "home_team")),
        name_key(_require(row, "away_team")),
    )


# --------------------------------------------------------------------- readers
def read_matches_csv(
    path: Path,
    default_season: str | None = None,
    default_competition: str = DEFAULT_COMPETITION,
) -> list[MatchRecord]:
    """Parse ``matches.csv`` into records (no goal details yet)."""
    records: list[MatchRecord] = []
    errors: list[str] = []
    for line, row in _read_rows(path):
        try:
            rnd = _parse_int(_require(row, "round"), "round")
            status_text = row.get("status", "").lower()
            rec = MatchRecord(
                season=_season(row, default_season),
                round=rnd,  # type: ignore[arg-type]
                home_team=_require(row, "home_team"),
                away_team=_require(row, "away_team"),
                match_date=_parse_date(row.get("date", "")),
                home_goals=_parse_int(row.get("home_goals", ""), "home_goals"),
                away_goals=_parse_int(row.get("away_goals", ""), "away_goals"),
                status=MatchStatus(status_text) if status_text else None,
                source_url=row.get("source_url") or None,
                competition=_competition(row, default_competition),
            )
            if (rec.home_goals is None) != (rec.away_goals is None):
                raise ValueError("only one of home_goals/away_goals is filled")
            records.append(rec)
        except ValueError as exc:
            errors.append(f"{Path(path).name}:{line}: {exc}")
    if errors:
        raise CsvImportError(errors)
    return records


def read_goals_csv(
    path: Path, default_season: str | None = None, default_competition: str = DEFAULT_COMPETITION
) -> dict[FixtureKey, list[GoalRecord]]:
    """Parse ``goals.csv``; returns goals grouped by match."""
    result: dict[FixtureKey, list[GoalRecord]] = defaultdict(list)
    errors: list[str] = []
    for line, row in _read_rows(path):
        try:
            key = _row_fixture_key(row, default_season, default_competition)
            result[key].append(
                GoalRecord(
                    team=_require(row, "team"),
                    player=row.get("player") or None,
                    minute=_parse_int(row.get("minute", ""), "minute"),
                    is_penalty=_parse_bool(row.get("penalty", "")),
                    is_own_goal=_parse_bool(row.get("own_goal", "")),
                )
            )
        except ValueError as exc:
            errors.append(f"{Path(path).name}:{line}: {exc}")
    if errors:
        raise CsvImportError(errors)
    return dict(result)


def read_appearances_csv(
    path: Path, default_season: str | None = None, default_competition: str = DEFAULT_COMPETITION
) -> dict[FixtureKey, list[AppearanceRecord]]:
    """Parse ``appearances.csv``; returns appearances grouped by match."""
    result: dict[FixtureKey, list[AppearanceRecord]] = defaultdict(list)
    errors: list[str] = []
    for line, row in _read_rows(path):
        try:
            key = _row_fixture_key(row, default_season, default_competition)
            result[key].append(
                AppearanceRecord(
                    team=_require(row, "team"),
                    player=_require(row, "player"),
                    minutes_played=_parse_int(row.get("minutes", ""), "minutes"),
                )
            )
        except ValueError as exc:
            errors.append(f"{Path(path).name}:{line}: {exc}")
    if errors:
        raise CsvImportError(errors)
    return dict(result)


def read_league_csv(
    matches_path: Path,
    goals_path: Path | None = None,
    appearances_path: Path | None = None,
    default_season: str | None = None,
    default_competition: str = DEFAULT_COMPETITION,
) -> list[MatchRecord]:
    """Read all CSV files and attach goal/appearance details to their matches."""
    records = read_matches_csv(matches_path, default_season, default_competition)
    by_key = {rec.fixture_key(): rec for rec in records}
    errors: list[str] = []

    if goals_path is not None:
        goals = read_goals_csv(goals_path, default_season, default_competition)
        for key in goals.keys() - by_key.keys():
            errors.append(f"{Path(goals_path).name}: goals for unknown match {key}")
        for key, rec in by_key.items():
            if key in goals:
                rec.goals = goals[key]
            elif rec.home_goals == 0 and rec.away_goals == 0:
                rec.goals = []  # 0-0: "no goal rows" is complete information
            # else: leave None -> score known, scorers unknown

    if appearances_path is not None:
        apps = read_appearances_csv(appearances_path, default_season, default_competition)
        for key in apps.keys() - by_key.keys():
            errors.append(f"{Path(appearances_path).name}: appearances for unknown match {key}")
        for key, rec in by_key.items():
            if key in apps:
                rec.appearances = apps[key]

    if errors:
        raise CsvImportError(errors)
    return records
