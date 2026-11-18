"""Load matches from the database into pandas DataFrames for modelling.

The model code works on plain DataFrames (no ORM objects), which keeps it fast,
easy to test and independent of the storage layer.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session, aliased

from football_predictor.db.models import (
    Appearance,
    Competition,
    Goal,
    Match,
    MatchStatus,
    Player,
    PointAdjustment,
    Season,
    Team,
)

MATCH_COLUMNS = [
    "match_id", "round", "date", "home", "away", "home_goals", "away_goals", "status",
]


def load_matches(session: Session, competition: str, season: str) -> pd.DataFrame:
    """All fixtures (played and unplayed) of one competition-season.

    Columns: match_id, round, date (Timestamp), home, away, home_goals, away_goals
    (float, NaN if not played), status (str).
    """
    home, away = aliased(Team), aliased(Team)
    stmt = (
        select(
            Match.id, Match.round, Match.match_date, home.name, away.name,
            Match.home_goals, Match.away_goals, Match.status,
        )
        .join(Competition, Competition.id == Match.competition_id)
        .join(Season, Season.id == Match.season_id)
        .join(home, home.id == Match.home_team_id)
        .join(away, away.id == Match.away_team_id)
        .where(Competition.name == competition, Season.name == season)
        .order_by(Match.round, Match.id)
    )
    df = pd.DataFrame(session.execute(stmt).all(), columns=MATCH_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    df["home_goals"] = df["home_goals"].astype(float)
    df["away_goals"] = df["away_goals"].astype(float)
    df["status"] = df["status"].astype(str)
    return df


def load_point_adjustments(session: Session, competition: str, season: str) -> dict[str, int]:
    """Team -> point adjustment (e.g. -3 for a deduction) of one competition-season."""
    stmt = (
        select(Team.name, PointAdjustment.points)
        .join(Team, Team.id == PointAdjustment.team_id)
        .join(Competition, Competition.id == PointAdjustment.competition_id)
        .join(Season, Season.id == PointAdjustment.season_id)
        .where(Competition.name == competition, Season.name == season)
    )
    return {name: int(points) for name, points in session.execute(stmt).all()}


def _competition_season_filter(stmt, competition: str, season: str):  # type: ignore[no-untyped-def]
    return (
        stmt.join(Competition, Competition.id == Match.competition_id)
        .join(Season, Season.id == Match.season_id)
        .where(Competition.name == competition, Season.name == season)
    )


APPEARANCE_COLUMNS = ["match_id", "team", "player_id", "player", "minutes"]
GOAL_COLUMNS = ["match_id", "team", "player_id", "player", "minute", "is_penalty", "is_own_goal"]


def load_appearances(session: Session, competition: str, season: str) -> pd.DataFrame:
    """Who played how many minutes for which team in each match of a competition-season.

    A missing minutes value (e.g. a CSV without minutes) is treated as a full 90.
    """
    stmt = _competition_season_filter(
        select(Appearance.match_id, Team.name, Appearance.player_id, Player.name,
               Appearance.minutes_played)
        .join(Match, Match.id == Appearance.match_id)
        .join(Team, Team.id == Appearance.team_id)
        .join(Player, Player.id == Appearance.player_id),
        competition, season,
    )
    df = pd.DataFrame(session.execute(stmt).all(), columns=APPEARANCE_COLUMNS)
    df["minutes"] = df["minutes"].fillna(90).astype(float)
    return df


def load_goals(session: Session, competition: str, season: str) -> pd.DataFrame:
    """Goal rows; ``team`` is the team the goal counted FOR (also for own goals)."""
    stmt = _competition_season_filter(
        select(Goal.match_id, Team.name, Goal.player_id, Player.name, Goal.minute,
               Goal.is_penalty, Goal.is_own_goal)
        .join(Match, Match.id == Goal.match_id)
        .join(Team, Team.id == Goal.team_id)
        .outerjoin(Player, Player.id == Goal.player_id),
        competition, season,
    )
    df = pd.DataFrame(session.execute(stmt).all(), columns=GOAL_COLUMNS)
    df["player_id"] = df["player_id"].astype("Int64")
    return df


def played_only(matches: pd.DataFrame) -> pd.DataFrame:
    """Only matches with a final score."""
    return matches[matches["status"] == MatchStatus.PLAYED.value].copy()


def list_competition_seasons(session: Session) -> pd.DataFrame:
    """Overview: competition, season, number of matches and played matches."""
    stmt = (
        select(
            Competition.name, Season.name, func.count(Match.id),
            func.sum(case((Match.status == MatchStatus.PLAYED, 1), else_=0)),
            func.max(Match.round),
        )
        .join(Competition, Competition.id == Match.competition_id)
        .join(Season, Season.id == Match.season_id)
        .group_by(Competition.name, Season.name)
        .order_by(Competition.name, Season.name)
    )
    return pd.DataFrame(
        session.execute(stmt).all(),
        columns=["competition", "season", "matches", "played", "rounds"],
    )
