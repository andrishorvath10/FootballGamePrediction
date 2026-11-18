"""Tests for the idempotent upsert loader."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from football_predictor.db.models import Goal, Match, MatchStatus, Player, PlayerTeam, Team
from football_predictor.ingest.loader import LeagueLoader, add_team_alias
from football_predictor.ingest.records import GoalRecord, MatchRecord


def _count(session: Session, model: type) -> int:
    return session.scalar(select(func.count()).select_from(model))


def _records() -> list[MatchRecord]:
    return [
        MatchRecord(
            season="S1", round=1, home_team="Alpha FC", away_team="Beta SE",
            match_date=date(2025, 8, 10), home_goals=2, away_goals=1,
            goals=[
                GoalRecord(team="Alpha FC", player="A One", minute=5),
                GoalRecord(team="Alpha FC", player="B One", minute=50, is_own_goal=True),
                GoalRecord(team="Beta SE", player="B Two", minute=70, is_penalty=True),
            ],
        ),
        MatchRecord(season="S1", round=2, home_team="Beta SE", away_team="Alpha FC",
                    match_date=date(2025, 8, 17)),
    ]


def test_load_twice_is_idempotent(session: Session) -> None:
    first = LeagueLoader(session).load(_records())
    session.commit()
    counts = {m: _count(session, m) for m in (Team, Match, Goal, Player, PlayerTeam)}

    second = LeagueLoader(session).load(_records())
    session.commit()

    assert first.matches_created == 2
    assert second.matches_created == 0 and second.matches_updated == 0
    assert second.matches_unchanged == 2
    assert {m: _count(session, m) for m in counts} == counts


def test_status_inferred_and_season_dates_set(session: Session) -> None:
    LeagueLoader(session).load(_records())
    played, scheduled = session.scalars(select(Match).order_by(Match.round)).all()
    assert played.status == MatchStatus.PLAYED
    assert scheduled.status == MatchStatus.SCHEDULED
    assert played.season.start_date == date(2025, 8, 10)
    assert played.season.end_date == date(2025, 8, 17)


def test_own_goal_scorer_belongs_to_opponent(session: Session) -> None:
    LeagueLoader(session).load(_records())
    og = session.scalars(select(Goal).where(Goal.is_own_goal)).one()
    assert og.team.name == "Alpha FC"  # credited to Alpha...
    membership = session.scalars(
        select(PlayerTeam).where(PlayerTeam.player_id == og.player_id)
    ).one()
    assert membership.team.name == "Beta SE"  # ...but scored by a Beta player


def test_update_result_and_goals(session: Session) -> None:
    LeagueLoader(session).load(_records())
    session.commit()

    played_later = MatchRecord(
        season="S1", round=2, home_team="Beta SE", away_team="Alpha FC",
        home_goals=1, away_goals=0, goals=[GoalRecord(team="Beta SE", player="B Two")],
    )
    stats = LeagueLoader(session).load([played_later])
    session.commit()

    assert stats.matches_updated == 1
    m = session.scalars(select(Match).where(Match.round == 2)).one()
    assert (m.home_goals, m.away_goals, m.status) == (1, 0, MatchStatus.PLAYED)
    assert m.match_date == date(2025, 8, 17)  # None in the update did not erase it
    assert len(m.goals) == 1
    assert _count(session, Player) == 3  # "B Two" reused, not duplicated


def test_none_goals_keep_existing_details(session: Session) -> None:
    LeagueLoader(session).load(_records())
    session.commit()
    score_only = MatchRecord(season="S1", round=1, home_team="Alpha FC",
                             away_team="Beta SE", home_goals=2, away_goals=1)
    LeagueLoader(session).load([score_only])
    session.commit()
    assert _count(session, Goal) == 3


def test_alias_and_spelling_variants_map_to_one_team(session: Session) -> None:
    LeagueLoader(session).load(_records())
    add_team_alias(session, "Alpha", "Alpha FC")
    session.commit()

    variants = [
        MatchRecord(season="S1", round=3, home_team="ALPHA  fc", away_team="Beta SE"),
        MatchRecord(season="S1", round=4, home_team="Alpha", away_team="Beta SE"),
    ]
    stats = LeagueLoader(session).load(variants)
    assert stats.teams_created == 0
    assert _count(session, Team) == 2


def test_alias_conflict_rejected(session: Session) -> None:
    LeagueLoader(session).load(_records())
    with pytest.raises(ValueError):
        add_team_alias(session, "Beta SE", "Alpha FC")


def test_goal_for_team_not_in_match_rejected(session: Session) -> None:
    bad = MatchRecord(season="S1", round=1, home_team="Alpha FC", away_team="Beta SE",
                      home_goals=1, away_goals=0, goals=[GoalRecord(team="Gamma")])
    with pytest.raises(ValueError):
        LeagueLoader(session).load([bad])
