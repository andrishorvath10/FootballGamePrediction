"""Tests for the goalscorer model."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from football_predictor.ingest.loader import LeagueLoader
from football_predictor.ingest.sample_data import (
    SYNTHETIC_COMPETITION_NAME as COMP,
)
from football_predictor.ingest.sample_data import (
    SYNTHETIC_SEASON_NAME as SEASON,
)
from football_predictor.ingest.sample_data import (
    generate_sample_league,
)
from football_predictor.model.dataset import load_appearances, load_goals, load_matches
from football_predictor.model.scorers import (
    ScorerForecaster,
    ScorerParams,
    anonymize,
    fit_scorers,
    scorer_probabilities,
    scorer_score_table,
    scorer_walk_forward,
)
from football_predictor.model.strength import StrengthParams


def tiny() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A 2-0 B, then B 0-1 A. p1 plays everything, p2 45' then bench, p3 plays for B."""
    matches = pd.DataFrame({
        "match_id": [1, 2], "round": [1, 2],
        "date": pd.to_datetime(["2025-08-10", "2025-08-17"]),
        "home": ["A", "B"], "away": ["B", "A"],
        "home_goals": [2.0, 0.0], "away_goals": [0.0, 1.0], "status": ["played"] * 2,
    })
    apps = pd.DataFrame({
        "match_id": [1, 1, 1, 2, 2, 2], "team": ["A", "A", "B", "A", "A", "B"],
        "player_id": [1, 2, 3, 1, 2, 3], "player": ["p1", "p2", "p3", "p1", "p2", "p3"],
        "minutes": [90.0, 45.0, 90.0, 90.0, 0.0, 90.0],
    })
    goals = pd.DataFrame({
        "match_id": [1, 1, 2], "team": ["A", "A", "A"], "player_id": [1, 2, 1],
        "player": ["p1", "p2", "p1"], "minute": [10, 30, 60],
        "is_penalty": [False, False, True], "is_own_goal": [False, False, False],
    })
    goals["player_id"] = goals["player_id"].astype("Int64")
    return matches, apps, goals


def by_player(model) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    return model.players.set_index("player")


def test_share_formula_matches_hand_calculation() -> None:
    matches, apps, goals = tiny()
    model = fit_scorers(matches, apps, goals, ScorerParams(shrinkage=2.0))
    p = by_player(model)
    # exposure: p1 = 2·90/90 + 1·90/90 = 3, p2 = 2·45/90 = 1, p3 = 0 (B never scored)
    assert p.loc["p1", "exposure"] == pytest.approx(3.0)
    assert p.loc["p2", "exposure"] == pytest.approx(1.0)
    assert model.league_share == pytest.approx(3 / 4)  # 3 goals / 4 exposure
    # share = (goals + k·s0) / (exposure + k)
    assert p.loc["p1", "share"] == pytest.approx((2 + 1.5) / (3 + 2))
    assert p.loc["p2", "share"] == pytest.approx((1 + 1.5) / (1 + 2))
    assert p.loc["p3", "share"] == pytest.approx(0.75)  # no data -> league share
    assert (p.loc["p1", "goals"], p.loc["p1", "penalties"]) == (2, 1)
    # expected minutes over the last 5 team matches (only 2 exist)
    assert p.loc["p1", "exp_minutes"] == pytest.approx(90)
    assert p.loc["p2", "exp_minutes"] == pytest.approx(22.5)
    assert p.loc["p2", "apps"] == 1  # 0 minutes on the bench is not an appearance


def test_no_shrinkage_gives_raw_ratio_and_no_division_by_zero() -> None:
    matches, apps, goals = tiny()
    p = by_player(fit_scorers(matches, apps, goals, ScorerParams(shrinkage=0)))
    assert p.loc["p1", "share"] == pytest.approx(2 / 3)
    assert p.loc["p2", "share"] == pytest.approx(1.0)
    assert p.loc["p3", "share"] == pytest.approx(0.75)  # 0/0 falls back to s0


def test_own_goals_and_unknown_scorers_are_not_credited() -> None:
    matches, apps, goals = tiny()
    goals.loc[1, "is_own_goal"] = True  # p2's goal becomes an own goal for A
    goals.loc[2, "player_id"] = pd.NA  # scorer unknown
    p = by_player(fit_scorers(matches, apps, goals, ScorerParams(shrinkage=0)))
    assert p.loc["p1", "goals"] == 1
    assert p.loc["p2", "goals"] == 0
    assert p.loc["p1", "exposure"] == pytest.approx(3.0)  # team goals still count


def test_minutes_only_baseline_gives_everyone_league_share() -> None:
    matches, apps, goals = tiny()
    model = fit_scorers(matches, apps, goals, ScorerParams(use_skill=False))
    assert np.allclose(model.players["share"], 0.75)


def test_form_window_uses_only_recent_matches() -> None:
    matches, apps, goals = tiny()
    p = by_player(fit_scorers(matches, apps, goals, ScorerParams(form_matches=1)))
    assert p.loc["p2", "exp_minutes"] == 0  # benched in the last match
    model = fit_scorers(matches, apps, goals, ScorerParams(form_matches=1))
    assert "p2" not in set(model.candidates("A")["player"])


def test_scorer_probabilities_poisson_thinning() -> None:
    matches, apps, goals = tiny()
    model = fit_scorers(matches, apps, goals, ScorerParams(shrinkage=2.0))
    probs = scorer_probabilities(model, "A", lam=2.0).set_index("player")
    xg = 2.0 * 0.7 * 1.0  # λ · share · minutes fraction
    assert probs.loc["p1", "xg"] == pytest.approx(xg)
    assert probs.loc["p1", "p_score"] == pytest.approx(1 - math.exp(-xg))
    assert probs.loc["p1", "p_2plus"] == pytest.approx(1 - math.exp(-xg) * (1 + xg))
    assert list(probs.index) == ["p1", "p2"]  # sorted, team B players excluded


def test_thinning_by_simulation() -> None:
    """P(player scores) from simulating team goals + per-goal scorer draws."""
    rng = np.random.default_rng(0)
    lam, q = 2.5, 0.3
    team_goals = rng.poisson(lam, 200_000)
    player_goals = rng.binomial(team_goals, q)
    assert (player_goals > 0).mean() == pytest.approx(1 - math.exp(-lam * q), abs=0.005)


@pytest.fixture()
def synthetic(session: Session) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    league = generate_sample_league(seed=3, played_rounds=18, postponed_rate=0.0)
    LeagueLoader(session).load(league.records)
    session.commit()
    return (load_matches(session, COMP, SEASON), load_appearances(session, COMP, SEASON),
            load_goals(session, COMP, SEASON), league.truth)


def test_loaders_return_lineups_and_goals(synthetic) -> None:  # type: ignore[no-untyped-def]
    matches, apps, goals, _ = synthetic
    played = matches[matches["status"] == "played"]
    assert set(apps["match_id"]) == set(played["match_id"])
    assert len(goals) == int(played["home_goals"].sum() + played["away_goals"].sum())
    assert goals["is_own_goal"].any()


def test_recovers_true_scoring_shares(synthetic) -> None:  # type: ignore[no-untyped-def]
    matches, apps, goals, truth = synthetic
    model = fit_scorers(matches, apps, goals, ScorerParams(shrinkage=2.0))
    true = pd.DataFrame([(p.team, p.name, p.scoring_share) for p in truth.players],
                        columns=["team", "player", "true_share"])
    joined = model.players.merge(true, on=["team", "player"])
    assert np.corrcoef(joined["share"], joined["true_share"])[0, 1] > 0.7


def test_walk_forward_has_no_look_ahead(synthetic) -> None:  # type: ignore[no-untyped-def]
    matches, apps, goals, _ = synthetic
    fcs = [ScorerForecaster("s", ScorerParams(), StrengthParams()),
           ScorerForecaster("base", ScorerParams(use_skill=False), None)]
    preds = scorer_walk_forward(matches, apps, goals, fcs, start_round=5)
    assert set(preds["round"]) == set(range(5, 19))

    # Changing results AFTER round 10 must not change the round-10 predictions.
    later = matches["round"] > 10
    changed = matches.assign(home_goals=np.where(later, 0.0, matches["home_goals"]))
    late_ids = set(matches.loc[later, "match_id"])
    preds2 = scorer_walk_forward(changed, apps, goals[~goals["match_id"].isin(late_ids)],
                                 fcs, start_round=5)
    r10 = preds[preds["round"] == 10].reset_index(drop=True)
    r10b = preds2[preds2["round"] == 10].reset_index(drop=True)
    pd.testing.assert_frame_equal(r10, r10b)

    table = scorer_score_table(preds)
    assert set(table["forecaster"]) == {"s", "base"}
    assert (table["mean_p"] > 0).all()
    # every forecaster is scored on exactly the same (match, team, player) rows
    keys = {name: set(zip(g["match_id"], g["team"], g["player_id"], strict=True))
            for name, g in preds.groupby("forecaster")}
    assert keys["s"] == keys["base"]


def test_fixed_rows_across_form_windows(synthetic) -> None:  # type: ignore[no-untyped-def]
    matches, apps, goals, _ = synthetic
    fcs = [ScorerForecaster(f"form{n}", ScorerParams(form_matches=n), StrengthParams())
           for n in (1, 10)]
    preds = scorer_walk_forward(matches, apps, goals, fcs, start_round=8)
    sizes = preds.groupby("forecaster").size()
    assert sizes["form1"] == sizes["form10"]
    # a 1-match window gives some rostered players probability 0
    assert (preds.loc[preds["forecaster"] == "form1", "p_score"] == 0).any()


def test_anonymize_is_stable() -> None:
    out = anonymize(pd.Series(["Kiss", "Nagy", "Kiss"]))
    assert list(out) == ["Player 001", "Player 002", "Player 001"]
