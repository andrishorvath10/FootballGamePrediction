"""Tests for the synthetic league generator."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sqlalchemy.orm import Session

from football_predictor.db.models import MatchStatus
from football_predictor.db.validation import find_goal_mismatches
from football_predictor.ingest.loader import LeagueLoader
from football_predictor.ingest.sample_data import double_round_robin, generate_sample_league


def test_double_round_robin_is_balanced() -> None:
    teams = [f"T{i}" for i in range(10)]
    rounds = double_round_robin(teams)
    assert len(rounds) == 18
    for rnd in rounds:  # everybody plays exactly once per round
        assert sorted(t for pair in rnd for t in pair) == sorted(teams)
    pairs = Counter(p for rnd in rounds for p in rnd)
    assert len(pairs) == 90 and set(pairs.values()) == {1}  # each ordered pair once
    home_games = Counter(h for rnd in rounds for h, _ in rnd)
    assert set(home_games.values()) == {9}


def test_generator_is_deterministic() -> None:
    a = generate_sample_league(seed=7)
    b = generate_sample_league(seed=7)
    assert a.truth == b.truth
    assert [(r.home_goals, r.away_goals) for r in a.records] == [
        (r.home_goals, r.away_goals) for r in b.records
    ]


def test_generator_shape_and_truth() -> None:
    league = generate_sample_league(n_teams=10, played_rounds=14, seed=1)
    assert len(league.records) == 90
    unplayed = [r for r in league.records if r.round > 14]
    assert all(r.status == MatchStatus.SCHEDULED and r.home_goals is None for r in unplayed)
    # geometric mean of the strengths is 1 by construction
    assert np.isclose(np.exp(np.mean(np.log(list(league.truth.attack.values())))), 1.0)
    # each team's scoring shares sum to 1
    for team in league.truth.attack:
        total = sum(p.scoring_share for p in league.truth.players if p.team == team)
        assert np.isclose(total, 1.0)


def test_goals_consistent_with_lineups_and_score() -> None:
    league = generate_sample_league(seed=3)
    for rec in league.records:
        if rec.status != MatchStatus.PLAYED:
            continue
        assert sum(g.team == rec.home_team for g in rec.goals) == rec.home_goals
        assert sum(g.team == rec.away_team for g in rec.goals) == rec.away_goals
        on_pitch = {a.player for a in rec.appearances}
        assert all(g.player in on_pitch for g in rec.goals)  # scorers actually played
        assert len(rec.appearances) == 2 * 14  # 11 starters + 3 subs per side


def test_sample_loads_cleanly(session: Session) -> None:
    league = generate_sample_league(seed=5)
    LeagueLoader(session).load(league.records)
    session.commit()
    assert find_goal_mismatches(session) == []
    again = LeagueLoader(session).load(league.records)
    assert again.matches_unchanged == 90
