"""Team attack/defence strengths - the simple ("ratio of averages") model.

Model
-----
Expected goals in a match between home team h and away team a::

    λ (home goals) = base × attack[h] × defence[a] × home_advantage
    μ (away goals) = base × attack[a] × defence[h]

* ``base``           = (weighted) average goals of an away team in this league
* ``home_advantage`` = average home goals / average away goals
* ``attack > 1``     = scores more than an average team would in the same games
* ``defence > 1``    = concedes more than average (i.e. a WEAK defence)

Estimation (per team)
---------------------
For every match of team i we know how many goals an *average* team would be
expected to score there: the league home average if i played at home, the away
average otherwise. Then::

    attack[i]  = (Σ w·scored   + k·m) / (Σ w·expected_scored   + k·m)
    defence[i] = (Σ w·conceded + k·m) / (Σ w·expected_conceded + k·m)

* ``w = exp(-ξ · days_ago)`` - exponential time decay ("form"); ξ = 0 means all
  matches count equally. Half-life = ln(2)/ξ days.
* ``k`` (shrinkage) - number of *pseudo-matches* of exactly average performance
  added to every team, ``m`` = average goals per team per match. With little data the
  ratio is pulled towards 1 (league average); with lots of data the prior fades.
  This is exactly the posterior mean of a Poisson rate with a Gamma(k·m, k·m)
  prior (Gamma–Poisson conjugacy), so it has a clean Bayesian interpretation.
* Optionally only each team's last ``n`` matches are used (the classic "form table"
  approach), to compare against exponential decay.

Limitation: this simple version ignores *who* the opponents were. If a team has
only played the weakest sides so far, its attack is overestimated. The maximum
likelihood model (next step) estimates all teams jointly and fixes this.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MODELS = ("simple", "mle", "dc")


@dataclass(frozen=True)
class StrengthParams:
    """Tunable hyper-parameters of all strength models.

    ``model``: "simple" (ratio of averages, this module), "mle" (joint maximum
    likelihood) or "dc" (maximum likelihood + Dixon–Coles), see ``model/mle.py``.
    """

    model: str = "simple"
    xi: float = 0.0  # time decay per day (0 = no decay), all models
    shrinkage: float = 2.0  # simple model: pseudo-matches of average performance (k)
    last_n: int | None = None  # simple model: use only each team's last n matches
    prior_sd: float = 0.5  # mle/dc: sd of the Normal prior on log-strengths

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")

    def label(self) -> str:
        """Short human-readable description, e.g. ``dc(xi=0.005, sd=0.5)``."""
        if self.model == "simple":
            extra = f"k={self.shrinkage:g}" + (f", last{self.last_n}" if self.last_n else "")
        else:
            extra = f"sd={self.prior_sd:g}"
        return f"{self.model}(xi={self.xi:g}, {extra})"


@dataclass
class TeamStrengths:
    """Fitted model: league constants plus per-team multipliers."""

    base: float
    home_advantage: float
    attack: dict[str, float]
    defense: dict[str, float]
    matches: dict[str, int] = field(default_factory=dict)
    params: StrengthParams = field(default_factory=StrengthParams)
    rho: float = 0.0  # Dixon–Coles low-score correction (0 = plain Poisson)

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """(λ, μ) for a fixture; unknown teams are treated as league average."""
        lam = (self.base * self.attack.get(home, 1.0) * self.defense.get(away, 1.0)
               * self.home_advantage)
        mu = self.base * self.attack.get(away, 1.0) * self.defense.get(home, 1.0)
        return lam, mu

    def to_frame(self) -> pd.DataFrame:
        """Table of team strengths, best attack first."""
        df = pd.DataFrame({
            "attack": self.attack, "defense": self.defense, "matches": self.matches,
        })
        df["net"] = np.log(df["attack"]) - np.log(df["defense"])  # >0 = better than average
        return df.sort_values("net", ascending=False)


def time_weights(dates: pd.Series, as_of: pd.Timestamp, xi: float) -> np.ndarray:
    """Exponential decay weights ``exp(-xi * days_since)``; future dates count as 0 days."""
    if xi == 0:
        return np.ones(len(dates))
    days = (as_of - dates).dt.days.clip(lower=0).fillna(0).to_numpy(dtype=float)
    return np.exp(-xi * days)


def to_team_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Long format: every match becomes two rows, one from each team's perspective."""
    home = pd.DataFrame({
        "team": matches["home"], "opponent": matches["away"], "is_home": True,
        "scored": matches["home_goals"], "conceded": matches["away_goals"],
        "date": matches["date"],
    })
    away = pd.DataFrame({
        "team": matches["away"], "opponent": matches["home"], "is_home": False,
        "scored": matches["away_goals"], "conceded": matches["home_goals"],
        "date": matches["date"],
    })
    return pd.concat([home, away], ignore_index=True)


def _last_n_mask(rows: pd.DataFrame, n: int) -> np.ndarray:
    """True for each team's n most recent matches."""
    order = rows.sort_values("date", kind="stable")
    recency = order.groupby("team").cumcount(ascending=False)  # 0 = most recent
    return (recency < n).reindex(rows.index).to_numpy()


def fit_model(
    played: pd.DataFrame,
    params: StrengthParams | None = None,
    as_of: pd.Timestamp | None = None,
    teams: Iterable[str] | None = None,
) -> TeamStrengths:
    """Fit whichever model ``params.model`` selects."""
    params = params or StrengthParams()
    if params.model == "simple":
        return fit_strengths(played, params, as_of, teams)
    from football_predictor.model.mle import fit_mle  # local import avoids a cycle

    return fit_mle(played, params, as_of, teams)


def fit_strengths(
    played: pd.DataFrame,
    params: StrengthParams | None = None,
    as_of: pd.Timestamp | None = None,
    teams: Iterable[str] | None = None,
) -> TeamStrengths:
    """Estimate team strengths from played matches.

    Args:
        played: matches with columns date, home, away, home_goals, away_goals.
        params: hyper-parameters (decay, shrinkage, last-n).
        as_of: reference date for the time decay (default: last match date).
        teams: teams to include even if they have no matches yet (rated 1.0).
    """
    params = params or StrengthParams()
    if played.empty:
        raise ValueError("No played matches to fit on")
    as_of = as_of if as_of is not None else played["date"].max()

    # 1) League constants (time-weighted averages over all matches).
    w_match = time_weights(played["date"], as_of, params.xi)
    home_avg = float(np.sum(w_match * played["home_goals"]) / np.sum(w_match))
    away_avg = float(np.sum(w_match * played["away_goals"]) / np.sum(w_match))
    if away_avg <= 0 or home_avg <= 0:
        raise ValueError("League averages must be positive")

    # 2) Per-team weighted sums of actual vs. "average team" expected goals.
    rows = to_team_rows(played)
    w = time_weights(rows["date"], as_of, params.xi)
    if params.last_n is not None:
        w = w * _last_n_mask(rows, params.last_n)
    rows = rows.assign(
        w=w,
        w_scored=w * rows["scored"],
        w_exp_scored=w * np.where(rows["is_home"], home_avg, away_avg),
        w_conceded=w * rows["conceded"],
        w_exp_conceded=w * np.where(rows["is_home"], away_avg, home_avg),
        n=(w > 0).astype(int),
    )
    sums = rows.groupby("team")[
        ["w_scored", "w_exp_scored", "w_conceded", "w_exp_conceded", "n"]
    ].sum()

    # 3) Shrink towards 1 with k pseudo-matches of average goals m.
    prior = params.shrinkage * (home_avg + away_avg) / 2
    attack = (sums["w_scored"] + prior) / (sums["w_exp_scored"] + prior)
    defense = (sums["w_conceded"] + prior) / (sums["w_exp_conceded"] + prior)

    all_teams = set(sums.index) | set(teams or [])
    return TeamStrengths(
        base=away_avg,
        home_advantage=home_avg / away_avg,
        attack={t: float(attack.get(t, 1.0)) for t in all_teams},
        defense={t: float(defense.get(t, 1.0)) for t in all_teams},
        matches={t: int(sums["n"].get(t, 0)) for t in all_teams},
        params=params,
    )
