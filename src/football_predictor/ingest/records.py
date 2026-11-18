"""Source-independent record types.

Every data source (synthetic generator, CSV, web scraper) produces these plain
dataclasses. A single loader (:mod:`football_predictor.ingest.loader`) then writes
them to the database. This keeps parsing and persistence separate: parsers can be
unit-tested without a database, and upsert logic exists in exactly one place.

Convention for optional fields: ``None`` means "unknown / not provided by this
source", and the loader will never overwrite an existing value with ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from football_predictor.db.models import MatchStatus

DEFAULT_COMPETITION = "Default league"


def clean_name(name: str) -> str:
    """Trim and collapse internal whitespace: ``"  Alpha   FC "`` -> ``"Alpha FC"``."""
    return " ".join(name.split())


def name_key(name: str) -> str:
    """Case-insensitive lookup key for a name (used for team aliases and joins)."""
    return clean_name(name).casefold()


@dataclass
class GoalRecord:
    """One goal.

    ``team`` is the team credited with the goal. For an own goal, ``player`` is the
    opponent who scored it (``is_own_goal=True``). ``player=None`` = unknown scorer.
    """

    team: str
    player: str | None = None
    minute: int | None = None
    is_penalty: bool = False
    is_own_goal: bool = False
    player_external_id: str | None = None  # stable source ID, e.g. "mlsz:396006"


@dataclass
class AppearanceRecord:
    """A player's appearance in a match (line-up data)."""

    team: str
    player: str
    minutes_played: int | None = None
    player_external_id: str | None = None


@dataclass
class MatchRecord:
    """A fixture as delivered by any source.

    ``goals``/``appearances``: ``None`` = no detail available (existing rows are
    kept); a list (possibly empty) = the complete, authoritative set for this match.
    """

    season: str
    round: int
    home_team: str
    away_team: str
    match_date: date | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    status: MatchStatus | None = None
    source_url: str | None = None
    goals: list[GoalRecord] | None = None
    appearances: list[AppearanceRecord] | None = None
    competition: str = DEFAULT_COMPETITION
    external_id: str | None = None  # stable source ID of the match, e.g. "mlsz:2204364"

    def effective_status(self) -> MatchStatus | None:
        """Explicit status if given; PLAYED if a score is present; otherwise None."""
        if self.status is not None:
            return self.status
        if self.home_goals is not None and self.away_goals is not None:
            return MatchStatus.PLAYED
        return None

    def fixture_key(self) -> tuple[str, str, int, str, str]:
        """Natural key used to join e.g. a goals CSV to its match."""
        return (
            clean_name(self.competition),
            clean_name(self.season),
            self.round,
            name_key(self.home_team),
            name_key(self.away_team),
        )


@dataclass
class PointAdjustmentRecord:
    """A federation point adjustment (negative = deduction) for one team."""

    season: str
    competition: str
    team: str
    points: int
    reason: str | None = None
