"""Walk-forward backtesting, scoring rules, baselines and calibration.

Walk-forward
------------
For every round r (from ``start_round`` on) the model is fitted ONLY on matches
played before the first match of round r, then it predicts round r. This mimics
real use and prevents look-ahead bias. Catch-up matches belong to their original
round but are still predicted with information from before that round.

Scoring rules for 1X2 probabilities p = (p_home, p_draw, p_away), outcome o
-----------------------------------------------------------------------------
* log loss = -log p_o (lower is better; punishes confident mistakes very hard).
  Uniform 1/3 guessing gives ln 3 ≈ 1.0986.
* Brier = Σ_k (p_k - y_k)²  where y is the one-hot outcome (0 = perfect, 2 = worst).
* RPS (ranked probability score) = ½ Σ_{k=1}^{2} (P_k - Y_k)², with cumulative
  P = (p_home, p_home + p_draw). It respects the ORDER home < draw < away:
  predicting a draw when the home side won is "less wrong" than predicting an away
  win. The standard metric for football forecasts.
* accuracy = how often the most likely outcome happened (intuitive, but ignores
  how confident the forecast was).

Baselines (a model is only useful if it beats these)
----------------------------------------------------
* uniform: 1/3 each.
* league frequencies: home/draw/away rates from the training matches.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_predictor.model.dataset import played_only
from football_predictor.model.poisson import predict_match
from football_predictor.model.strength import StrengthParams, fit_model

logger = logging.getLogger(__name__)

OUTCOMES = ("H", "D", "A")
PROB_COLUMNS = ["p_home", "p_draw", "p_away"]


def outcome_index(home_goals: np.ndarray, away_goals: np.ndarray) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    return np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))


# ------------------------------------------------------------------ scoring rules
def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Per-match log loss; probabilities clipped to avoid log(0)."""
    p = probs[np.arange(len(outcomes)), outcomes]
    return -np.log(np.clip(p, eps, 1.0))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Per-match multi-class Brier score."""
    onehot = np.eye(3)[outcomes]
    return np.sum((probs - onehot) ** 2, axis=1)


def rps(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Per-match ranked probability score for the ordered outcomes H < D < A."""
    onehot = np.eye(3)[outcomes]
    cum_p = np.cumsum(probs, axis=1)[:, :2]
    cum_y = np.cumsum(onehot, axis=1)[:, :2]
    return 0.5 * np.sum((cum_p - cum_y) ** 2, axis=1)


def summarize(df: pd.DataFrame) -> dict[str, float]:
    """Mean scores of a predictions frame (columns p_home, p_draw, p_away, outcome)."""
    probs = df[PROB_COLUMNS].to_numpy()
    outcomes = df["outcome"].to_numpy()
    return {
        "matches": len(df),
        "log_loss": float(log_loss(probs, outcomes).mean()),
        "brier": float(brier(probs, outcomes).mean()),
        "rps": float(rps(probs, outcomes).mean()),
        "accuracy": float((probs.argmax(axis=1) == outcomes).mean()),
    }


# -------------------------------------------------------------------- forecasters
@dataclass(frozen=True)
class Forecaster:
    """Something that maps (training matches, fixtures) -> 1X2 probabilities."""

    name: str
    params: StrengthParams | None = None  # None for baselines
    kind: str = "model"  # "model", "uniform" or "frequency"

    def predict(self, train: pd.DataFrame, fixtures: pd.DataFrame) -> np.ndarray:
        """(len(fixtures), 3) array of probabilities."""
        if self.kind == "uniform":
            return np.full((len(fixtures), 3), 1 / 3)
        if self.kind == "frequency":
            counts = np.bincount(
                outcome_index(train["home_goals"].to_numpy(), train["away_goals"].to_numpy()),
                minlength=3,
            )
            freq = (counts + 1) / (counts.sum() + 3)  # Laplace smoothing
            return np.tile(freq, (len(fixtures), 1))

        teams = set(fixtures["home"]) | set(fixtures["away"])
        strengths = fit_model(train, self.params, teams=teams)
        out = np.empty((len(fixtures), 3))
        for k, row in enumerate(fixtures.itertuples()):
            lam, mu = strengths.expected_goals(row.home, row.away)
            p = predict_match(row.home, row.away, lam, mu, rho=strengths.rho)
            out[k] = (p.p_home, p.p_draw, p.p_away)
        return out


def baselines() -> list[Forecaster]:
    """The two naive reference forecasters."""
    return [Forecaster("uniform 1/3", kind="uniform"),
            Forecaster("league H/D/A rates", kind="frequency")]


# ------------------------------------------------------------------- walk-forward
def walk_forward(
    matches: pd.DataFrame,
    forecasters: Iterable[Forecaster],
    start_round: int = 4,
) -> pd.DataFrame:
    """Predict every played match of round >= start_round using only earlier results.

    Returns one row per (forecaster, match) with probabilities and the outcome.
    """
    played = played_only(matches)
    forecasters = list(forecasters)
    rows: list[pd.DataFrame] = []
    for rnd in sorted(played["round"].unique()):
        if rnd < start_round:
            continue
        test = played[played["round"] == rnd]
        cutoff = test["date"].min()
        train = played[played["date"] < cutoff]
        if train.empty or pd.isna(cutoff):
            continue
        outcome = outcome_index(test["home_goals"].to_numpy(), test["away_goals"].to_numpy())
        for fc in forecasters:
            probs = fc.predict(train, test)
            rows.append(pd.DataFrame({
                "forecaster": fc.name, "round": rnd, "date": test["date"].to_numpy(),
                "home": test["home"].to_numpy(), "away": test["away"].to_numpy(),
                "home_goals": test["home_goals"].to_numpy(),
                "away_goals": test["away_goals"].to_numpy(),
                "p_home": probs[:, 0], "p_draw": probs[:, 1], "p_away": probs[:, 2],
                "outcome": outcome,
            }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def score_table(predictions: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Mean metrics per forecaster (and optionally per extra columns), best RPS first."""
    keys = ["forecaster", *(by or [])]
    table = pd.DataFrame([
        {**dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True)),
         **summarize(group)}
        for key, group in predictions.groupby(keys)
    ])
    return table.sort_values([*(by or []), "rps"]).reset_index(drop=True)


# -------------------------------------------------------------------- calibration
def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Predicted probability vs. observed frequency, pooling all three outcomes.

    A perfectly calibrated model has observed ≈ predicted in every bin: of all the
    events it gave ~30 %, about 30 % actually happened.
    """
    probs = predictions[PROB_COLUMNS].to_numpy().ravel()
    happened = np.eye(3)[predictions["outcome"].to_numpy()].ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "predicted": probs, "observed": happened})
    out = df.groupby("bin").agg(predicted=("predicted", "mean"),
                                observed=("observed", "mean"),
                                count=("observed", "size")).reset_index()
    out["bin_range"] = [f"{edges[b]:.1f}-{edges[b + 1]:.1f}" for b in out["bin"]]
    return out[["bin_range", "predicted", "observed", "count"]]


# ------------------------------------------------------------------------- tuning
def parameter_grid(
    models: Iterable[str] = ("simple", "mle", "dc"),
    xis: Iterable[float] = (0.0, 0.002, 0.005, 0.01),
    shrinkages: Iterable[float] = (0.5, 1, 2, 5, 10),
    prior_sds: Iterable[float] = (0.15, 0.25, 0.4, 0.7, 1.0, 1.5),
) -> list[Forecaster]:
    """All model/parameter combinations to try."""
    grid: list[Forecaster] = []
    for model in models:
        for xi in xis:
            if model == "simple":
                for k in shrinkages:
                    p = StrengthParams(model="simple", xi=xi, shrinkage=k)
                    grid.append(Forecaster(p.label(), p))
            else:
                for sd in prior_sds:
                    p = StrengthParams(model=model, xi=xi, prior_sd=sd)
                    grid.append(Forecaster(p.label(), p))
    return grid
