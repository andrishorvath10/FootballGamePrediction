"""Tests for scoring rules, walk-forward backtest and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_predictor.ingest.sample_data import generate_sample_league
from football_predictor.model.evaluate import (
    Forecaster,
    baselines,
    brier,
    calibration_table,
    log_loss,
    outcome_index,
    rps,
    score_table,
    walk_forward,
)
from football_predictor.model.strength import StrengthParams


def test_outcome_index() -> None:
    assert outcome_index(np.array([2, 1, 0]), np.array([1, 1, 3])).tolist() == [0, 1, 2]


def test_uniform_scores_have_known_values() -> None:
    probs = np.full((3, 3), 1 / 3)
    outcomes = np.array([0, 1, 2])
    assert np.allclose(log_loss(probs, outcomes), np.log(3))
    assert np.allclose(brier(probs, outcomes), 2 / 3)  # (2/3)² + 2·(1/3)²
    # RPS: home win -> cum P = (1/3, 2/3), cum Y = (1, 1): ½((2/3)² + (1/3)²) = 5/18
    #      draw     -> cum Y = (0, 1): ½((1/3)² + (1/3)²) = 1/9
    assert np.allclose(rps(probs, outcomes), [5 / 18, 1 / 9, 5 / 18])


def test_perfect_forecast_scores_zero() -> None:
    probs = np.eye(3)
    outcomes = np.array([0, 1, 2])
    assert np.allclose(brier(probs, outcomes), 0)
    assert np.allclose(rps(probs, outcomes), 0)
    assert np.allclose(log_loss(probs, outcomes), 0)


def test_rps_rewards_being_close_in_order() -> None:
    """Home win happened: predicting 'draw' is less bad than predicting 'away'."""
    home_win = np.array([0])
    assert rps(np.array([[0, 1.0, 0]]), home_win) < rps(np.array([[0, 0, 1.0]]), home_win)
    # Brier cannot tell these two apart - that is why RPS is preferred for football.
    assert brier(np.array([[0, 1.0, 0]]), home_win) == brier(np.array([[0, 0, 1.0]]), home_win)


def _matches() -> pd.DataFrame:
    league = generate_sample_league(seed=8, played_rounds=18)
    return pd.DataFrame([
        {"match_id": i, "round": r.round, "date": pd.Timestamp(r.match_date),
         "home": r.home_team, "away": r.away_team,
         "home_goals": np.nan if r.home_goals is None else float(r.home_goals),
         "away_goals": np.nan if r.away_goals is None else float(r.away_goals),
         "status": str(r.status)}
        for i, r in enumerate(league.records)
    ])


class SpyForecaster(Forecaster):
    """Records the latest training date seen for every predicted round."""

    seen: list = []

    def predict(self, train: pd.DataFrame, fixtures: pd.DataFrame) -> np.ndarray:
        SpyForecaster.seen.append((train["date"].max(), fixtures["date"].min()))
        return np.full((len(fixtures), 3), 1 / 3)


def test_walk_forward_never_trains_on_the_future() -> None:
    SpyForecaster.seen = []
    walk_forward(_matches(), [SpyForecaster("spy", kind="uniform")], start_round=2)
    assert SpyForecaster.seen and all(last_train < first_test
                                      for last_train, first_test in SpyForecaster.seen)


def test_walk_forward_and_scores() -> None:
    p = StrengthParams(model="mle", prior_sd=0.4)
    preds = walk_forward(_matches(), [*baselines(), Forecaster(p.label(), p)], start_round=4)
    played_from_4 = (_matches().query("round >= 4")["status"] == "played").sum()
    assert len(preds) == 3 * played_from_4
    assert np.allclose(preds[["p_home", "p_draw", "p_away"]].sum(axis=1), 1)
    table = score_table(preds)
    assert set(table["forecaster"]) == {"uniform 1/3", "league H/D/A rates", p.label()}
    assert np.isclose(table.set_index("forecaster").loc["uniform 1/3", "log_loss"], np.log(3))


def test_calibration_table_counts() -> None:
    preds = pd.DataFrame({"p_home": [0.75, 0.2], "p_draw": [0.15, 0.3], "p_away": [0.1, 0.5],
                          "outcome": [0, 2]})
    table = calibration_table(preds, bins=10)
    assert table["count"].sum() == 6
    assert table.loc[table["bin_range"] == "0.7-0.8", "observed"].item() == 1.0
