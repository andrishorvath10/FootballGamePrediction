"""Tests for the simple strength model, incl. recovery of known synthetic strengths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_predictor.ingest.sample_data import generate_sample_league
from football_predictor.model.predict import next_round, predict_round
from football_predictor.model.strength import (
    StrengthParams,
    fit_strengths,
    time_weights,
)


def _frame(records: list) -> pd.DataFrame:
    """MatchRecords -> the DataFrame layout produced by model.dataset.load_matches."""
    return pd.DataFrame([
        {
            "match_id": i, "round": r.round, "date": pd.Timestamp(r.match_date),
            "home": r.home_team, "away": r.away_team,
            "home_goals": np.nan if r.home_goals is None else float(r.home_goals),
            "away_goals": np.nan if r.away_goals is None else float(r.away_goals),
            "status": str(r.status),
        }
        for i, r in enumerate(records)
    ])


def _tiny() -> pd.DataFrame:
    d = pd.Timestamp("2025-08-10")
    return pd.DataFrame({
        "date": [d, d, d + pd.Timedelta(days=7), d + pd.Timedelta(days=7)],
        "home": ["A", "C", "B", "D"], "away": ["B", "D", "A", "C"],
        "home_goals": [3.0, 1.0, 0.0, 2.0], "away_goals": [0.0, 1.0, 1.0, 2.0],
    })


def test_time_weights_halve_after_half_life() -> None:
    xi = np.log(2) / 30  # half-life 30 days
    dates = pd.Series(pd.to_datetime(["2025-01-31", "2025-01-01"]))
    w = time_weights(dates, pd.Timestamp("2025-01-31"), xi)
    assert np.allclose(w, [1.0, 0.5])


def test_home_advantage_and_base() -> None:
    s = fit_strengths(_tiny(), StrengthParams(shrinkage=0))
    assert np.isclose(s.base, 1.0)  # away goals 0+1+1+2 over 4 matches
    assert np.isclose(s.home_advantage, 1.5)  # home 6/4 vs away 4/4


def test_shrinkage_formula_and_pull_towards_average() -> None:
    raw = fit_strengths(_tiny(), StrengthParams(shrinkage=0))
    weak = fit_strengths(_tiny(), StrengthParams(shrinkage=0.5))
    strong = fit_strengths(_tiny(), StrengthParams(shrinkage=50))
    # A scored 3 at home (average team: 1.5) and 1 away (average: 1.0): 4 / 2.5
    assert np.isclose(raw.attack["A"], 1.6)
    # prior = k * m = 0.5 * (1.5 + 1.0) / 2 = 0.625 pseudo-goals on both sides
    assert np.isclose(weak.attack["A"], (4 + 0.625) / (2.5 + 0.625))
    # Close to 1 when 50 average pseudo-matches dominate 2 real ones.
    assert abs(strong.attack["A"] - 1) < 0.1
    assert all(abs(v - 1) < 0.1 for v in strong.defense.values())


def test_last_n_uses_only_recent_matches() -> None:
    s = fit_strengths(_tiny(), StrengthParams(shrinkage=0, last_n=1))
    # A's last match: away at B, scored 1 where an average away team scores 1.0
    assert np.isclose(s.attack["A"], 1.0)
    assert s.matches["A"] == 1


def test_unknown_team_is_average() -> None:
    s = fit_strengths(_tiny(), teams=["NEW"])
    assert s.attack["NEW"] == 1.0 and s.defense["NEW"] == 1.0


@pytest.mark.parametrize("seed", [11, 12])
def test_recovers_known_synthetic_strengths(seed: int) -> None:
    """The model should rank teams like the true (hidden) parameters do.

    14 teams x 26 rounds = 182 matches. Correlation is computed on log-strengths,
    centred, because the model's ratings are normalised by arithmetic league
    averages while the truth has geometric mean 1 (a constant offset in log space).
    """
    league = generate_sample_league(n_teams=14, played_rounds=26, seed=seed,
                                    strength_sigma=0.4)
    played = _frame(league.records)
    played = played[played["status"] == "played"]
    s = fit_strengths(played, StrengthParams(shrinkage=2))

    teams = sorted(league.truth.attack)
    for est, true in ((s.attack, league.truth.attack), (s.defense, league.truth.defense)):
        est_log = np.log([est[t] for t in teams])
        true_log = np.log([true[t] for t in teams])
        assert np.corrcoef(est_log, true_log)[0, 1] > 0.8
    assert abs(s.home_advantage - league.truth.home_advantage) < 0.3


def test_predict_next_round_of_synthetic_league() -> None:
    league = generate_sample_league(seed=42, played_rounds=14)
    matches = _frame(league.records)
    rnd = next_round(matches)
    preds, _ = predict_round(matches, rnd)
    assert rnd == 15 and len(preds) == 5
    for p in preds:
        assert np.isclose(p.p_home + p.p_draw + p.p_away, 1.0)
        assert p.lam > 0 and p.mu > 0


def test_no_lookahead_in_predict_round() -> None:
    """Changing a *future* result must not change predictions for an earlier round."""
    league = generate_sample_league(seed=42, played_rounds=14)
    matches = _frame(league.records)
    future = matches.index[(matches["round"] == 16)][0]
    altered = matches.copy()
    altered.loc[future, ["home_goals", "away_goals", "status"]] = [9.0, 0.0, "played"]
    a, _ = predict_round(matches, 15)
    b, _ = predict_round(altered[altered["round"] <= 15], 15)
    assert [p.lam for p in a] == pytest.approx([p.lam for p in b])


def test_next_round_skips_catch_up_matches() -> None:
    from football_predictor.model.predict import catch_up_rounds

    league = generate_sample_league(seed=42, played_rounds=14, postponed_rate=0.0)
    matches = _frame(league.records)
    first = matches.index[matches["round"] == 1][0]
    matches.loc[first, ["home_goals", "away_goals", "status"]] = [np.nan, np.nan, "postponed"]
    assert next_round(matches) == 15
    assert catch_up_rounds(matches, 15) == [1]
    preds, _ = predict_round(matches, [1, 15])
    assert sorted({p.round for p in preds}) == [1, 15] and len(preds) == 6
