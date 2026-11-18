"""Tests for the strength-drift random walk and its tuning scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_predictor.ingest.sample_data import generate_sample_league
from football_predictor.model.drift import (
    days_ahead,
    drift_log_scores,
    drift_paths,
    drift_summary,
    lag_sd,
    past_strength_change,
    squad_turnover,
    team_drift_scales,
)
from football_predictor.model.simulate import simulate_season
from football_predictor.model.strength import StrengthParams


def test_random_walk_variance_grows_linearly_and_persists() -> None:
    rng = np.random.default_rng(0)
    n, sigma = 10, 0.2
    walk = drift_paths(rng, 40_000, n, np.array([30.0, 60.0, 150.0]), sigma)
    assert walk.shape == (40_000, 3, 2 * n)
    centring = (n - 1) / n  # variance lost by centring over teams
    for k, d in enumerate([30, 60, 150]):
        assert walk[:, k, 0].var() == pytest.approx(sigma**2 * d / 30 * centring, rel=0.03)
    # persistent: corr(W(30), W(60)) = sqrt(30/60) for a random walk
    assert np.corrcoef(walk[:, 0, 0], walk[:, 1, 0])[0, 1] == pytest.approx(np.sqrt(0.5), abs=0.02)
    # strengths stay relative: attack and defence blocks sum to zero
    assert np.allclose(walk[..., :n].sum(axis=-1), 0)
    assert np.allclose(walk[..., n:].sum(axis=-1), 0)
    assert not drift_paths(rng, 5, n, np.array([30.0]), 0.0).any()


def test_days_ahead_clips_past_and_guesses_missing_dates() -> None:
    fx = pd.DataFrame({"round": [10, 11, 12],
                       "date": pd.to_datetime(["2026-03-01", None, "2026-01-01"])})
    days = days_ahead(fx, pd.Timestamp("2026-02-01"))
    assert days[0] == 28 and days[2] == 0  # past date -> 0
    assert days[1] == 14  # 2nd round after the first open one, 7 days each


def _league() -> pd.DataFrame:
    league = generate_sample_league(n_teams=8, played_rounds=14, seed=11, postponed_rate=0.0)
    return pd.DataFrame([
        {"match_id": k, "round": r.round, "date": pd.Timestamp(r.match_date),
         "home": r.home_team, "away": r.away_team,
         "home_goals": float(r.home_goals), "away_goals": float(r.away_goals),
         "status": "played"}
        for k, r in enumerate(league.records) if r.home_goals is not None
    ])


def test_drift_widens_simulated_points() -> None:
    m = _league()
    m.loc[m["round"] > 7, ["home_goals", "away_goals"]] = np.nan
    m.loc[m["round"] > 7, "status"] = "scheduled"
    params = StrengthParams(model="mle", prior_sd=0.5)
    still = simulate_season(m, params, n_sims=4000, seed=0)
    moving = simulate_season(m, params, n_sims=4000, seed=0, drift=0.3)
    assert moving.drift == 0.3
    band = lambda r: (r.summary["points_p90"] - r.summary["points_p10"]).mean()  # noqa: E731
    assert band(moving) > band(still)
    assert np.allclose(moving.positions.sum(axis=0), 1)


def test_drift_scores_prefer_no_drift_on_a_static_league() -> None:
    """The synthetic league has constant strengths: large σ must not win."""
    scores = drift_log_scores(_league(), StrengthParams(model="mle", prior_sd=0.5),
                              sigmas=(0.0, 0.5), start_round=4, step=3, n_draws=300)
    assert set(scores["sigma"]) == {0.0, 0.5}
    summary = drift_summary(scores).set_index("sigma")
    assert summary.loc[0.0, "log_loss"] < summary.loc[0.5, "log_loss"]
    assert (scores["days_ahead"] >= 0).all()


# --- team-specific drift -----------------------------------------------------------

def test_squad_turnover_hand_calculation() -> None:
    matches = pd.DataFrame({"match_id": [1, 2, 3],
                            "date": pd.to_datetime(["2025-08-01", "2025-08-08", "2025-08-15"])})
    apps = pd.DataFrame({
        "match_id": [1, 1, 2, 2, 2, 3, 3],
        "team": ["A"] * 7,
        "player_id": [1, 2, 1, 2, 3, 1, 9],
        "minutes": [90, 90, 90, 90, 90, 90, 20],  # player 9: only 20 minutes
    })
    # overlaps: {1,2} vs {1,2,3} = 2/3, {1,2,3} vs {1} = 1/3 -> turnover 1 - 1/2
    assert squad_turnover(apps, matches)["A"] == pytest.approx(0.5)
    # restricted to the first two matches
    assert squad_turnover(apps, matches.iloc[:2])["A"] == pytest.approx(1 / 3)


def test_team_drift_scales_redistribute_but_keep_mean_variance() -> None:
    teams = ["A", "B", "C", "D"]
    measure = pd.Series({"A": 0.2, "B": 0.4, "C": 0.6})  # D has no line-ups
    assert np.allclose(team_drift_scales(measure, teams, 0.0), 1.0)
    scales = team_drift_scales(measure, teams, 0.5)
    assert np.mean(scales**2) == pytest.approx(1.0)  # overall drift level unchanged
    assert scales[0] < scales[1] < scales[2]  # more turnover -> more drift
    assert scales[3] == pytest.approx(scales[1])  # missing -> league average (z = 0)
    assert np.all(np.diff(team_drift_scales(measure, teams, -0.5)[:3]) < 0)


def test_drift_paths_respect_team_scales() -> None:
    walk = drift_paths(np.random.default_rng(1), 40_000, 6, np.array([60.0]), 0.2,
                       scales=np.array([2.0, 1, 1, 1, 1, 1]))
    var = walk[:, 0, :6].var(axis=0)
    assert var[0] > 3 * var[1]
    assert np.allclose(walk[..., :6].sum(axis=-1), 0)


def test_manual_team_drift_in_simulation() -> None:
    m = _league()
    m.loc[m["round"] > 7, ["home_goals", "away_goals"]] = np.nan
    m.loc[m["round"] > 7, "status"] = "scheduled"
    params = StrengthParams(model="mle", prior_sd=0.5)
    team = sorted(m["home"])[0]
    base = simulate_season(m, params, n_sims=6000, seed=0, drift=0.2).summary.set_index("team")
    wild = simulate_season(m, params, n_sims=6000, seed=0, drift=0.2,
                           drift_scales={team: 4.0}).summary.set_index("team")
    spread = lambda s: s["points_p90"] - s["points_p10"]  # noqa: E731
    assert spread(wild)[team] > spread(base)[team]


def test_past_strength_change_and_covariate_scoring() -> None:
    m = _league()
    params = StrengthParams(model="mle", prior_sd=0.5)
    teams = sorted(set(m["home"]) | set(m["away"]))
    change = past_strength_change(m, params, teams)
    assert set(change.index) == set(teams) and (change >= 0).all()
    scores = drift_log_scores(m, params, sigmas=(0.1,), betas=(0.0, 0.5), start_round=8,
                              step=4, n_draws=100, covariate="past_change")
    assert set(scores["beta"]) == {0.0, 0.5}


def test_lag_sd_hand_calculation() -> None:
    # A played once 60 days ago: its estimate is 60 days old -> Var = 60/30 = 2
    # B played 60 days ago and today (equal weights): w'min(age)w / (Σw)² = 60/30/4
    played = pd.DataFrame({
        "home": ["A", "B", "B"], "away": ["X", "Y", "Z"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-03-02"]),
    })
    sd = lag_sd(played, StrengthParams(model="mle", xi=0.0), ["A", "B", "C"],
                as_of=pd.Timestamp("2026-03-02"))
    assert sd == pytest.approx([np.sqrt(2.0), np.sqrt(0.5), 0.0])
