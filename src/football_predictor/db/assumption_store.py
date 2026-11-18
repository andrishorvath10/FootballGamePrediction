"""Saving and loading expert assumptions (:class:`~football_predictor.db.models.Assumption`)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from football_predictor.db.models import Assumption, Competition, Season, Team

logger = logging.getLogger(__name__)


@dataclass
class StoredAssumption:
    """An assumption as read back from the database."""

    team: str
    change: float
    start: pd.Timestamp
    drift_factor: float = 1.0
    note: str | None = None
    created_at: datetime | None = None


def _ids(session: Session, competition: str, season: str) -> tuple[int, int]:
    comp_id = session.scalar(select(Competition.id).where(Competition.name == competition))
    season_id = session.scalar(select(Season.id).where(Season.name == season))
    if comp_id is None or season_id is None:
        raise ValueError(f"Unknown competition/season: {competition!r} {season!r}")
    return comp_id, season_id


def _filter(stmt, comp_id: int, season_id: int):  # type: ignore[no-untyped-def]
    return stmt.where(Assumption.competition_id == comp_id, Assumption.season_id == season_id)


def list_assumptions(session: Session, competition: str, season: str) -> list[StoredAssumption]:
    """All assumptions of one competition-season, by team name."""
    try:
        comp_id, season_id = _ids(session, competition, season)
    except ValueError:
        return []
    rows = session.execute(_filter(
        select(Team.name, Assumption).join(Team, Team.id == Assumption.team_id),
        comp_id, season_id).order_by(Team.name)).all()
    return [StoredAssumption(name, a.change, pd.Timestamp(a.start_date), a.drift_factor,
                             a.note, a.created_at) for name, a in rows]


def save_assumption(
    session: Session, competition: str, season: str, team: str, change: float,
    start: date | None = None, drift_factor: float = 1.0, note: str | None = None,
) -> None:
    """Insert or replace the assumption for ``team`` (start defaults to today)."""
    comp_id, season_id = _ids(session, competition, season)
    team_id = session.scalar(select(Team.id).where(Team.name == team))
    if team_id is None:
        raise ValueError(f"Unknown team {team!r}")
    row = session.scalar(_filter(select(Assumption), comp_id, season_id)
                         .where(Assumption.team_id == team_id))
    if row is None:
        row = Assumption(season_id=season_id, competition_id=comp_id, team_id=team_id)
        session.add(row)
    row.change = float(change)
    row.start_date = start or date.today()
    row.drift_factor = float(drift_factor)
    row.note = note
    row.created_at = datetime.now()
    session.flush()
    logger.info("Saved assumption: %s %+.0f%% from %s", team, 100 * change, row.start_date)


def delete_assumption(session: Session, competition: str, season: str, team: str) -> bool:
    """Remove the assumption for ``team``; True if there was one."""
    comp_id, season_id = _ids(session, competition, season)
    row = session.scalar(_filter(select(Assumption).join(Team, Team.id == Assumption.team_id),
                                 comp_id, season_id).where(Team.name == team))
    if row is None:
        return False
    session.delete(row)
    session.flush()
    return True


def replace_assumptions(
    session: Session, competition: str, season: str, rows: list[StoredAssumption]
) -> None:
    """Make the stored set equal ``rows`` (used by the dashboard's save button).

    Unchanged rows keep their original ``created_at``.
    """
    existing = {a.team: a for a in list_assumptions(session, competition, season)}
    wanted = {r.team: r for r in rows}
    for team in set(existing) - set(wanted):
        delete_assumption(session, competition, season, team)
    for team, r in wanted.items():
        old = existing.get(team)
        if old is not None and (old.change, old.start, old.drift_factor, old.note) == (
                r.change, r.start, r.drift_factor, r.note):
            continue
        save_assumption(session, competition, season, team, r.change, r.start.date(),
                        r.drift_factor, r.note)
