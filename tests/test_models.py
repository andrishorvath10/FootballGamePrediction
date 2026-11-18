"""Tests for the ORM schema: relationships, constraints and validation."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from football_predictor.db.models import (
    Competition,
    Goal,
    Match,
    MatchStatus,
    Player,
    PlayerTeam,
    Season,
    Team,
    TeamAlias,
)
from football_predictor.db.validation import find_goal_mismatches


def _comp(session: Session) -> Competition:
    """Get-or-create the single test competition."""
    comp = session.scalar(select(Competition).where(Competition.name == "Test League"))
    if comp is None:
        comp = Competition(name="Test League", region="Test", level=1)
        session.add(comp)
    return comp


def _match(session: Session, season: Season, **kwargs: object) -> Match:
    """A Match in the test competition."""
    return Match(season=season, competition=_comp(session), **kwargs)


def _setup_basic(session: Session) -> tuple[Season, Team, Team, Player]:
    season = Season(name="2025/26", start_date=date(2025, 8, 1), end_date=date(2026, 6, 1))
    home = Team(name="Alpha FC", short_name="ALP")
    away = Team(name="Beta SE", short_name="BET")
    player = Player(name="Test Player")
    session.add_all([season, home, away, player])
    session.flush()  # assigns primary keys
    session.add(PlayerTeam(player_id=player.id, team_id=home.id, season_id=season.id))
    return season, home, away, player


def test_create_team_match_goal(session: Session) -> None:
    season, home, away, player = _setup_basic(session)
    match = Match(
        season=season, competition=_comp(session), round=1, match_date=date(2025, 8, 10),
        home_team=home, away_team=away,
        home_goals=1, away_goals=0, status=MatchStatus.PLAYED,
    )
    match.goals.append(Goal(player=player, team=home, minute=37))
    session.add(match)
    session.commit()

    loaded = session.scalars(select(Match)).one()
    assert loaded.home_team.name == "Alpha FC"
    assert loaded.is_played
    assert len(loaded.goals) == 1
    assert loaded.goals[0].player.name == "Test Player"
    assert loaded.goals[0].is_penalty is False
    assert find_goal_mismatches(session) == []


def test_scheduled_match_has_no_score(session: Session) -> None:
    season, home, away, _ = _setup_basic(session)
    session.add(_match(session, season, round=2, home_team=home, away_team=away))
    session.commit()
    m = session.scalars(select(Match)).one()
    assert m.status == MatchStatus.SCHEDULED
    assert m.home_goals is None and not m.is_played


def test_duplicate_fixture_rejected(session: Session) -> None:
    season, home, away, _ = _setup_basic(session)
    session.add(_match(session, season, round=1, home_team=home, away_team=away))
    session.commit()
    session.add(_match(session, season, round=1, home_team=home, away_team=away))
    with pytest.raises(IntegrityError):
        session.commit()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"home_goals": -1, "away_goals": 0, "status": MatchStatus.PLAYED},  # negative goals
        {"home_goals": 1, "away_goals": None},  # half a score
        {"status": MatchStatus.PLAYED},  # played but no score
    ],
)
def test_match_check_constraints(session: Session, kwargs: dict) -> None:
    season, home, away, _ = _setup_basic(session)
    session.add(_match(session, season, round=1, home_team=home, away_team=away, **kwargs))
    with pytest.raises(IntegrityError):
        session.commit()


def test_team_cannot_play_itself(session: Session) -> None:
    season, home, _, _ = _setup_basic(session)
    session.add(_match(session, season, round=1, home_team=home, away_team=home))
    with pytest.raises(IntegrityError):
        session.commit()


@pytest.mark.parametrize("minute", [0, 131])
def test_goal_minute_range(session: Session, minute: int) -> None:
    season, home, away, _ = _setup_basic(session)
    match = _match(session, season, round=1, home_team=home, away_team=away,
                  home_goals=1, away_goals=0, status=MatchStatus.PLAYED)
    match.goals.append(Goal(team=home, minute=minute))
    session.add(match)
    with pytest.raises(IntegrityError):
        session.commit()


def test_foreign_keys_enforced(session: Session) -> None:
    """SQLite only enforces FKs with PRAGMA foreign_keys=ON - make sure it is set."""
    session.add(Goal(match_id=999, team_id=999))
    with pytest.raises(IntegrityError):
        session.commit()


def test_team_alias_resolves_to_team(session: Session) -> None:
    team = Team(name="Alpha FC")
    team.aliases.append(TeamAlias(alias="Alpha"))
    session.add(team)
    session.commit()
    alias = session.scalars(select(TeamAlias).where(TeamAlias.alias == "Alpha")).one()
    assert alias.team.name == "Alpha FC"


def test_goal_mismatch_detected(session: Session) -> None:
    season, home, away, player = _setup_basic(session)
    match = _match(session, season, round=1, home_team=home, away_team=away,
                  home_goals=2, away_goals=0, status=MatchStatus.PLAYED)
    match.goals.append(Goal(player=player, team=home, minute=10))  # only 1 of 2 recorded
    session.add(match)
    session.commit()

    mismatches = find_goal_mismatches(session)
    assert len(mismatches) == 1
    assert (mismatches[0].expected, mismatches[0].recorded) == (2, 1)
