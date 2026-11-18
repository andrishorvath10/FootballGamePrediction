"""Consistency checks that cannot be expressed as simple SQL constraints."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from football_predictor.db.models import Goal, Match, MatchStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GoalMismatch:
    """A played match whose Goal rows disagree with the final score."""

    match_id: int
    team_id: int
    expected: int  # from Match.home_goals / away_goals
    recorded: int  # number of Goal rows credited to this team


def find_goal_mismatches(session: Session, season_id: int | None = None) -> list[GoalMismatch]:
    """Compare per-team Goal row counts with the final score of every played match.

    Matches with *no* Goal rows at all are skipped: many amateur sources only
    publish the score, and that is not an inconsistency, just missing detail.
    Every mismatch is logged as a warning.
    """
    goal_counts = (
        select(Goal.match_id, Goal.team_id, func.count().label("n"))
        .group_by(Goal.match_id, Goal.team_id)
        .subquery()
    )

    stmt = select(Match).where(Match.status == MatchStatus.PLAYED)
    if season_id is not None:
        stmt = stmt.where(Match.season_id == season_id)
    matches = session.scalars(stmt).all()

    counts: dict[tuple[int, int], int] = {
        (row.match_id, row.team_id): row.n for row in session.execute(select(goal_counts))
    }

    mismatches: list[GoalMismatch] = []
    for m in matches:
        home_rec = counts.get((m.id, m.home_team_id), 0)
        away_rec = counts.get((m.id, m.away_team_id), 0)
        if home_rec == 0 and away_rec == 0:
            continue  # score-only match, no goal details
        for team_id, expected, recorded in (
            (m.home_team_id, m.home_goals, home_rec),
            (m.away_team_id, m.away_goals, away_rec),
        ):
            if expected != recorded:
                mm = GoalMismatch(m.id, team_id, expected or 0, recorded)
                logger.warning(
                    "Goal mismatch in match %d, team %d: score says %d, %d goal rows",
                    mm.match_id, mm.team_id, mm.expected, mm.recorded,
                )
                mismatches.append(mm)
    return mismatches
