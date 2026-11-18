"""Checking an expert assumption against the matches played after it took effect.

An assumption "team T is +20 % from date D, drift factor k" says two things:

* **direction** - from D on, T scores 1.2× and concedes 1/1.2× what the model would
  have expected;
* **uncertainty** - T's form keeps drifting k times as much as a typical team's
  (:mod:`football_predictor.model.drift`), i.e. "I am not sure where this goes".

Checking the direction:

1. **Baseline** - fit the team model only on matches *before* D, so the fit has not
   seen the period being judged, and compute for every match of T on/after D the
   expected goals for (a_i) and against (c_i) without any assumption.
2. **Realised shift** - the one-parameter model "T's goals ~ Pois(a_i·e^δ), goals
   against ~ Pois(c_i·e^−δ)" has the log-likelihood

       ℓ(δ) = GF·δ − A·e^δ − GA·δ − C·e^−δ        (A = Σ a_i, C = Σ c_i,
                                                    GF, GA = goals for / against)

   Setting ℓ'(δ) = 0 and writing u = e^δ gives A·u² − (GF − GA)·u − C = 0, so the
   maximum-likelihood estimate is closed-form::

       e^δ̂ = [(GF − GA) + sqrt((GF − GA)² + 4·A·C)] / (2·A)

   and the Fisher information I = −ℓ''(δ̂) = A·e^δ̂ + C·e^−δ̂ gives the sampling error.
3. **Baseline error** - the a_i, c_i are themselves estimates. By the implicit
   function theorem δ̂ moves with the log expected goals as

       ∂δ̂/∂log a_i = −a_i·e^δ̂ / I,      ∂δ̂/∂log c_i = +c_i·e^−δ̂ / I,

   so with the Laplace covariance Σ of the baseline fit the delta method adds
   gᵀΣg to the variance (g = gradient of δ̂ w.r.t. the baseline parameters), and
   the fit's lag behind drifting teams adds V_lag (see below).
4. **Verdict** - the assumed shift inside the 95 % interval of δ̂ is "consistent",
   above it "too optimistic", below it "too pessimistic". The log-score gain says
   whether the assumption improved the probability of the actual results.

Checking the drift factor: the assumption, read as a forecast, says δ̂ should land
around ln(1 + change), with a predictive variance

    V(k) = 1/I + gᵀΣg + k²·V_team + V_opp + V_lag

where V_team is the variance the random walk of T's own attack and defence adds to
δ̂ (σ = drift per 30 days, walk starting at the last match before D)::

    V_team = σ²/30 · (w_aᵀ M w_a + w_cᵀ M w_c),   M_ij = min(d_i, d_j),
    w_a,i = a_i·e^δ̂ / I,   w_c,i = c_i·e^−δ̂ / I,   d_i = days to match i

(a Brownian motion has Cov(W(s), W(t)) ∝ min(s, t)), and V_opp the same for the
opponents' walks (factor 1, each opponent independent), and V_lag what the
baseline's lag behind the drifting teams adds (:func:`~football_predictor.model.drift.lag_sd`;
a constant offset, so its weights are summed). The standardised residual
z = (δ̂ − ln(1 + change)) / √V(k) beyond ±1.96 means the deviation was larger than
the stated uncertainty allows: "too confident". One team can only ever show that -
a small deviation is consistent with any k - so *too cautious* is judged over
many assumptions at once (:func:`calibration`): a well calibrated forecaster has
mean z² ≈ 1, and the pooled maximum-likelihood k says what factor would have fitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import chi2

from football_predictor.model.dataset import played_only
from football_predictor.model.drift import DAYS_PER_UNIT, DRIFT_DEFAULT, lag_sd
from football_predictor.model.strength import StrengthParams, fit_model

Z95 = 1.959964


@dataclass
class ShiftCheck:
    """Result of :func:`check_shift` (changes as fractions: 0.2 = +20 %)."""

    team: str
    assumed: float
    start: pd.Timestamp
    matches: int
    drift_factor: float = 1.0
    goals_for: int = 0
    goals_against: int = 0
    expected_for: float = 0.0  # baseline, without the assumption
    expected_against: float = 0.0
    realized: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    log_gain: float | None = None  # per match; > 0 = the assumption helped
    verdict: str = "pending"  # pending | consistent | too optimistic | too pessimistic
    # --- drift factor ---
    residual: float | None = None  # δ̂ − ln(1 + assumed), log scale
    fixed_var: float | None = None  # sampling + baseline + opponents' drift + lag
    team_drift_var: float | None = None  # V_team at factor 1
    z: float | None = None  # residual / √V(drift_factor)
    implied_factor: float | None = None  # k with z² = 1 (one team: very noisy)
    factor_gain: float | None = None  # log density at the stated k minus at k = 1
    drift_verdict: str = "pending"  # pending | consistent | too confident

    def predictive_var(self, factor: float) -> float:
        """V(k) of the module docstring."""
        assert self.fixed_var is not None and self.team_drift_var is not None
        return self.fixed_var + factor**2 * self.team_drift_var


def realized_shift(a: float, c: float, gf: float, ga: float) -> tuple[float, float]:
    """(δ̂, sampling standard error) of the log shift; see the module docstring."""
    if a <= 0 or c <= 0:
        raise ValueError("expected goals must be positive")
    diff = gf - ga
    u = (diff + np.sqrt(diff**2 + 4 * a * c)) / (2 * a)
    info = a * u + c / u
    return float(np.log(u)), float(1 / np.sqrt(info))


def walk_variance(weights: np.ndarray, days: np.ndarray) -> float:
    """Var(Σ w_i·W(d_i)) for a unit random walk per 30 days: wᵀ min(d_i, d_j) w / 30."""
    d = np.maximum(np.asarray(days, dtype=float), 0.0)
    return float(weights @ np.minimum.outer(d, d) @ weights / DAYS_PER_UNIT)


def _log_pois(k: np.ndarray, rate: np.ndarray) -> np.ndarray:
    return k * np.log(rate) - rate - gammaln(k + 1)


def _log_normal(x: float, var: float) -> float:
    return float(-0.5 * (np.log(2 * np.pi * var) + x * x / var))


def check_shift(
    matches: pd.DataFrame, team: str, change: float, start: pd.Timestamp,
    params: StrengthParams | None = None, drift_factor: float = 1.0,
    sigma: float = DRIFT_DEFAULT,
) -> ShiftCheck:
    """Compare an assumed change (and drift factor) of ``team`` from ``start`` with
    what happened. ``sigma``: the league-wide drift per 30 days."""
    params = params or StrengthParams(model="mle", xi=0.005, prior_sd=0.7)
    start = pd.Timestamp(start)
    played = played_only(matches)
    before = played[played["date"] < start]
    after = played[(played["date"] >= start)
                   & ((played["home"] == team) | (played["away"] == team))]
    result = ShiftCheck(team=team, assumed=change, start=start, matches=len(after),
                        drift_factor=drift_factor)
    if after.empty or before.empty:
        return result

    teams = sorted(set(matches["home"]) | set(matches["away"]))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    cov = None
    if params.model == "simple":
        strengths = fit_model(before, params, teams=teams)
    else:
        from football_predictor.model.mle import fit_posterior

        posterior = fit_posterior(before, params, teams=teams)
        strengths, cov = posterior.strengths, posterior.covariance
    xg = np.array([strengths.expected_goals(h, a) for h, a in
                   zip(after["home"], after["away"], strict=True)])
    at_home = (after["home"] == team).to_numpy()
    opponent = np.where(at_home, after["away"], after["home"])
    exp_for = np.where(at_home, xg[:, 0], xg[:, 1])
    exp_against = np.where(at_home, xg[:, 1], xg[:, 0])
    gf = np.where(at_home, after["home_goals"], after["away_goals"])
    ga = np.where(at_home, after["away_goals"], after["home_goals"])

    delta, se = realized_shift(exp_for.sum(), exp_against.sum(), gf.sum(), ga.sum())
    u = np.exp(delta)
    info = 1 / se**2
    w_a = exp_for * u / info  # ∂δ̂/∂log a_i (up to sign)
    w_c = exp_against / u / info  # ∂δ̂/∂log c_i

    # baseline error: delta method on the parameter vector (att.., def.., base, home)
    base_var = 0.0
    if cov is not None:
        g = np.zeros(2 * n + 2)
        t, opp = index[team], np.array([index[o] for o in opponent])
        # log a_i = base + home·[T at home] + att_T + def_opp, weight −w_a
        g[t] -= w_a.sum()
        np.add.at(g, n + opp, -w_a)
        g[2 * n] += w_c.sum() - w_a.sum()
        # log c_i = base + home·[opp at home] + att_opp + def_T, weight +w_c
        g[n + t] += w_c.sum()
        np.add.at(g, opp, w_c)
        g[2 * n + 1] = w_c[~at_home].sum() - w_a[at_home].sum()
        base_var = float(g @ cov @ g)
    # ... and the baseline lags behind drifting teams (lag_sd): a constant offset per
    # team and component over the judged matches, so its weights are summed
    sig2 = sigma**2
    lag = dict(zip(teams, lag_sd(before, params, teams), strict=True))
    lag_var = sig2 * (lag[team] ** 2 * (w_a.sum() ** 2 + w_c.sum() ** 2)
                      + sum(lag[o] ** 2 * (w_a[opponent == o].sum() ** 2
                                           + w_c[opponent == o].sum() ** 2)
                            for o in set(opponent)))
    dir_se = np.sqrt(se**2 + base_var + lag_var)

    assumed = np.log1p(change)
    base = _log_pois(gf, exp_for) + _log_pois(ga, exp_against)
    factor = np.exp(assumed)
    shifted = _log_pois(gf, exp_for * factor) + _log_pois(ga, exp_against / factor)
    low, high = delta - Z95 * dir_se, delta + Z95 * dir_se
    if assumed > high:
        verdict = "too optimistic"
    elif assumed < low:
        verdict = "too pessimistic"
    else:
        verdict = "consistent"
    result.goals_for, result.goals_against = int(gf.sum()), int(ga.sum())
    result.expected_for, result.expected_against = float(exp_for.sum()), float(exp_against.sum())
    result.realized = float(np.expm1(delta))
    result.ci_low, result.ci_high = float(np.expm1(low)), float(np.expm1(high))
    result.log_gain = float((shifted - base).mean())
    result.verdict = verdict

    # drift factor: random walks start at the last match the baseline has seen
    days = (after["date"] - before["date"].max()).dt.days.to_numpy(dtype=float)
    team_var = sig2 * (walk_variance(w_a, days) + walk_variance(w_c, days))
    opp_var = sig2 * sum(walk_variance(w_a[opponent == o], days[opponent == o])
                         + walk_variance(w_c[opponent == o], days[opponent == o])
                         for o in set(opponent))
    residual = delta - assumed
    result.residual = float(residual)
    result.fixed_var = float(se**2 + base_var + opp_var + lag_var)
    result.team_drift_var = float(team_var)
    var_k = result.predictive_var(drift_factor)
    result.z = float(residual / np.sqrt(var_k))
    result.drift_verdict = "too confident" if abs(result.z) > Z95 else "consistent"
    if team_var > 0:
        excess = residual**2 - result.fixed_var
        result.implied_factor = float(np.sqrt(max(excess, 0.0) / team_var))
        result.factor_gain = (_log_normal(residual, var_k)
                              - _log_normal(residual, result.predictive_var(1.0)))
    return result


@dataclass
class Calibration:
    """How well a set of assumptions' stated uncertainty matched reality."""

    n: int
    mean_z2: float  # ≈ 1 when calibrated; > 1 overconfident, < 1 too cautious
    outside: float  # share with |z| > 1.96 (≈ 5 % when calibrated)
    p_value: float  # two-sided χ²_n test of Σ z² = n
    pooled_factor: float | None  # maximum-likelihood drift factor over all of them
    verdict: str  # calibrated | overconfident | too cautious | pending


def pooled_factor(checks: list[ShiftCheck], upper: float = 10.0) -> float | None:
    """The drift factor k maximising Σ log N(residual_j; 0, V_j(k))."""
    done = [c for c in checks if c.residual is not None and c.team_drift_var]
    if not done:
        return None
    r = np.array([c.residual for c in done])
    fixed = np.array([c.fixed_var for c in done])
    team = np.array([c.team_drift_var for c in done])

    def neg_ll(k: float) -> float:
        v = fixed + k * k * team
        return float(0.5 * np.sum(np.log(v) + r * r / v))

    return float(minimize_scalar(neg_ll, bounds=(0.0, upper), method="bounded").x)


def calibration(checks: list[ShiftCheck]) -> Calibration:
    """Pool the drift checks: mean z², share outside ±1.96, χ² test, pooled k."""
    z = np.array([c.z for c in checks if c.z is not None])
    if len(z) == 0:
        return Calibration(0, np.nan, np.nan, np.nan, None, "pending")
    s = float(np.sum(z * z))
    p = float(min(1.0, 2 * min(chi2.cdf(s, len(z)), chi2.sf(s, len(z)))))
    verdict = ("calibrated" if p >= 0.05
               else "overconfident" if s > len(z) else "too cautious")
    return Calibration(n=len(z), mean_z2=s / len(z), outside=float(np.mean(np.abs(z) > Z95)),
                       p_value=p, pooled_factor=pooled_factor(checks), verdict=verdict)


def no_change_checks(
    matches: pd.DataFrame, start: pd.Timestamp, params: StrengthParams | None = None,
    sigma: float = DRIFT_DEFAULT,
) -> list[ShiftCheck]:
    """"No change from ``start``" checked for every team that played after it.

    A test of the model itself rather than of an expert: if its uncertainty is right,
    the pooled :func:`calibration` of these checks is "calibrated" and about 5 % of
    the directions are flagged.
    """
    teams = sorted(set(matches["home"]) | set(matches["away"]))
    checks = [check_shift(matches, t, 0.0, start, params, sigma=sigma) for t in teams]
    return [c for c in checks if c.residual is not None]


def checks_frame(checks: list[ShiftCheck]) -> pd.DataFrame:
    """Table of checks (one row per assumption)."""
    cols = ["team", "assumed", "start", "matches", "goals_for", "goals_against",
            "expected_for", "expected_against", "realized", "ci_low", "ci_high",
            "log_gain", "verdict", "drift_factor", "z", "implied_factor", "factor_gain",
            "drift_verdict"]
    return pd.DataFrame([c.__dict__ for c in checks], columns=cols)
