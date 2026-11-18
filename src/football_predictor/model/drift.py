"""Strength drift: team strengths keep changing during the rest of the season.

The fitted strengths describe a team *today*. Over the following months players
leave or get injured, a new coach arrives, the winter break reshuffles squads -
so a fixture in May is less predictable from November's estimate than next
week's. We model this as a Gaussian random walk on every team's log attack and
log defence::

    θ_team(t) = θ_team(now) + W(t),   W(t) - W(s) ~ N(0, σ² · (t - s) / 30 days)

* the variance grows linearly with the time ahead (calendar days, so the
  three-month winter break counts as three months, not one round);
* within one simulated season the walk is *persistent*: a team that drops off
  in March stays weaker in April too (unlike independent noise per match), which
  is what makes final tables more spread out;
* increments are centred across teams (strengths are relative: sum to zero);
* the fit itself lags behind the walk (:func:`lag_sd`): a time-weighted estimate is an
  average of past strengths, which matters most after the winter break.

σ (per 30 days) is estimated from data, not from the final tables (only 9 per
season): :func:`drift_log_scores` freezes each season every few rounds and scores
the predictive probability of every later result at all horizons, which gives
thousands of (match, horizon) pairs. The predictive distribution of a match
``d`` days ahead is a mixture over posterior draws and drift draws::

    P(x, y) = E_θ,W [ Pois(x; λ(θ + W_d)) · Pois(y; μ(θ + W_d)) ]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from football_predictor.model.dataset import played_only
from football_predictor.model.strength import StrengthParams

DAYS_PER_UNIT = 30.0
# σ per 30 days, estimated with `tune-drift --season 2025/26` (see README).
DRIFT_DEFAULT = 0.10


def centre(z: np.ndarray, n_teams: int) -> np.ndarray:
    """Centre the attack and the defence block (last axis, 2·n_teams) over teams."""
    z = z.copy()
    z[..., :n_teams] -= z[..., :n_teams].mean(axis=-1, keepdims=True)
    z[..., n_teams:] -= z[..., n_teams:].mean(axis=-1, keepdims=True)
    return z


def centred_normal(
    rng: np.random.Generator, shape: tuple[int, ...], n_teams: int,
    scales: np.ndarray | None = None,
) -> np.ndarray:
    """N(0, scale_team²) draws for (att, def) of every team, each block centred.

    Last axis has length 2·n_teams (attack block, then defence block);
    ``scales`` (n_teams,) makes some teams drift more than others.
    """
    z = rng.standard_normal((*shape, 2 * n_teams))
    if scales is not None:
        z = z * np.tile(scales, 2)
    return centre(z, n_teams)


def drift_paths(
    rng: np.random.Generator, n_sims: int, n_teams: int, days: np.ndarray, sigma: float,
    scales: np.ndarray | None = None,
) -> np.ndarray:
    """Random-walk displacement at each of the sorted ``days`` ahead.

    Returns (n_sims, len(days), 2·n_teams); the walk starts at 0 on day 0.
    ``scales``: per-team multipliers of σ (see :func:`team_drift_scales`).
    """
    days = np.asarray(days, dtype=float)
    if sigma == 0 or len(days) == 0:
        return np.zeros((n_sims, len(days), 2 * n_teams))
    steps = np.diff(np.concatenate([[0.0], np.maximum(days, 0.0)]))
    scale = sigma * np.sqrt(steps / DAYS_PER_UNIT)
    increments = centred_normal(rng, (n_sims, len(days)), n_teams, scales)
    return np.cumsum(increments * scale[None, :, None], axis=1)


# ---------------------------------------------------------------- team-specific drift
def squad_turnover(appearances: pd.DataFrame, matches: pd.DataFrame,
                   min_minutes: int = 45) -> pd.Series:
    """How much each team's line-up changes from one match to the next.

    1 − mean Jaccard overlap of the players with ≥ ``min_minutes`` minutes in
    consecutive matches (0 = always the same eleven, 1 = a new team every week).
    Only the matches in ``matches`` count, so callers can restrict it to the past.
    """
    regular = appearances[appearances["minutes"] >= min_minutes]
    regular = regular.merge(matches[["match_id", "date"]], on="match_id")
    out = {}
    for team, g in regular.groupby("team"):
        lineups = [set(x["player_id"]) for _, x in
                   g.sort_values(["date", "match_id"]).groupby("match_id", sort=False)]
        overlap = [len(a & b) / len(a | b) for a, b in zip(lineups, lineups[1:], strict=False)
                   if a | b]
        if overlap:
            out[team] = 1.0 - float(np.mean(overlap))
    return pd.Series(out, dtype=float)


def past_strength_change(played: pd.DataFrame, params: StrengthParams,
                         teams: list[str]) -> pd.Series:
    """How much each team's strength moved so far: fit the first and the second
    half of ``played`` (by date) separately, return sqrt(Δatt² + Δdef²)."""
    from football_predictor.model.mle import fit_mle

    played = played.sort_values("date")
    half = len(played) // 2
    if half < len(teams):
        return pd.Series(dtype=float)
    first = fit_mle(played.iloc[:half], params, teams=teams)
    second = fit_mle(played.iloc[half:], params, teams=teams)
    return pd.Series({
        t: float(np.hypot(np.log(second.attack[t] / first.attack[t]),
                          np.log(second.defense[t] / first.defense[t])))
        for t in teams
    })


def team_drift_scales(turnover: pd.Series, teams: list[str], beta: float) -> np.ndarray:
    """Per-team multipliers of σ from a team measure (squad turnover, past change).

    σ²_team ∝ exp(2β·z_team), z = the measure standardised within the league
    (clipped at ±2, 0 for teams without line-ups), normalised so the *mean*
    variance stays σ²: β only redistributes drift between teams, the overall
    level is still set by σ. β = 0 gives every team the same σ.
    """
    z = turnover.reindex(teams)
    if beta == 0 or z.notna().sum() < 2 or z.std() == 0:
        return np.ones(len(teams))
    z = ((z - z.mean()) / z.std()).clip(-2, 2).fillna(0.0).to_numpy()
    w = np.exp(2 * beta * z)
    return np.sqrt(w / w.mean())


def lag_sd(played: pd.DataFrame, params: StrengthParams, teams: list[str],
           as_of: pd.Timestamp | None = None) -> np.ndarray:
    """How far each team's fitted strength lags behind its current one, per unit σ.

    A time-weighted fit estimates a weighted *average* of the team's past strengths,
    θ̂ ≈ Σ w_j θ(t_j) / Σ w_j. If θ follows a random walk, the lag θ(now) − θ̂ =
    Σ w_j (θ(now) − θ(t_j)) / Σ w_j has variance (looking back in time, the walk's
    covariance is again min(age_j, age_k))::

        Var = σ²/30 · Σ_jk w_j w_k min(age_j, age_k) / (Σ_j w_j)²

    Returned is that sd for σ = 1 (multiply by σ), one value per team, using the
    team's own matches. After a winter break the autumn matches are 4+ months old,
    so the lag - and with it the uncertainty about the team *now* - is large.
    """
    from football_predictor.model.strength import time_weights

    as_of = as_of if as_of is not None else played["date"].max()
    out = np.zeros(len(teams))
    for i, team in enumerate(teams):
        dates = played.loc[(played["home"] == team) | (played["away"] == team), "date"]
        if dates.empty:
            continue
        w = time_weights(dates, as_of, params.xi)
        age = (as_of - dates).dt.days.clip(lower=0).to_numpy(dtype=float)
        out[i] = np.sqrt(w @ np.minimum.outer(age, age) @ w / DAYS_PER_UNIT) / w.sum()
    return out


def days_ahead(fixtures: pd.DataFrame, as_of: pd.Timestamp) -> np.ndarray:
    """Calendar days from ``as_of`` to each fixture (0 for missing / past dates).

    A missing date is guessed as 7 days per round after the last dated round.
    """
    days = (fixtures["date"] - as_of).dt.days.astype(float)
    if days.isna().any():
        guess = (fixtures["round"] - fixtures["round"].min() + 1) * 7.0
        days = days.fillna(guess)
    return np.maximum(days.to_numpy(), 0.0)


def _poisson_logpmf(k: np.ndarray, rate: np.ndarray) -> np.ndarray:
    return k * np.log(rate) - rate - gammaln(k + 1)


def drift_log_scores(
    matches: pd.DataFrame,
    params: StrengthParams,
    sigmas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3),
    omegas: tuple[float, ...] = (0.0,),
    start_round: int = 5,
    step: int = 2,
    n_draws: int = 400,
    seed: int = 0,
    appearances: pd.DataFrame | None = None,
    betas: tuple[float, ...] = (0.0,),
    covariate: str = "turnover",
    taus: tuple[float, ...] = (0.0,),
    lags: tuple[float, ...] = (0.0,),
) -> pd.DataFrame:
    """Log predictive probability of every later result, per freeze point and σ.

    For freeze rounds ``start_round, start_round + step, ...``: fit the posterior
    on rounds before the freeze, then for every match actually played later
    compute ``log P(exact score)`` under each σ. Same random numbers for every σ
    (common random numbers) so the comparison is not blurred by Monte Carlo noise.

    ``betas`` ≠ 0 make σ team-specific via a team measure taken *before* the freeze
    (:func:`team_drift_scales`): ``covariate="turnover"`` (squad turnover, needs
    ``appearances``) or ``"past_change"`` (:func:`past_strength_change`).

    ``taus`` ≠ 0 add a persistent jump J ~ N(0, τ²) per team at the freeze (variance
    τ² + σ²·d/30 instead of σ²·d/30): extra uncertainty that does not grow with the
    horizon, e.g. because the time-weighted fit lags behind the team's current form.

    ``lags`` ≠ 0 add the fit's lag behind a drifting team (:func:`lag_sd`, times σ,
    times the given multiplier; 1 = what the random walk implies) as a persistent jump.

    Columns: freeze_round, match_id, days_ahead, sigma, omega, beta, tau, lag, log_score.
    """
    from football_predictor.model.mle import fit_posterior
    from football_predictor.model.simulate import hide_results_from

    rng = np.random.default_rng(seed)
    teams = sorted(set(matches["home"]) | set(matches["away"]))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    actual = played_only(matches)
    rows = []
    for freeze in range(start_round, int(matches["round"].max()) + 1, step):
        frozen = played_only(hide_results_from(matches, freeze))
        targets = actual[~actual["match_id"].isin(frozen["match_id"])]
        if frozen.empty or targets.empty:
            continue
        posterior = fit_posterior(frozen, params, teams=teams)
        theta = posterior.sample(rng, n_draws)  # (S, 2n+2)
        raw = rng.standard_normal((n_draws, 2 * n))  # one drift direction per draw
        jump_raw = rng.standard_normal((n_draws, 2 * n))  # one jump per draw
        jump = centre(jump_raw, n)
        lag_unit = np.tile(lag_sd(frozen, params, teams), 2)
        if covariate == "past_change":
            turnover = past_strength_change(frozen, params, teams)
        elif appearances is not None:
            turnover = squad_turnover(appearances, frozen)
        else:
            turnover = pd.Series(dtype=float)
        e = rng.standard_normal((2, n_draws, len(targets)))  # match-level noise
        hi = targets["home"].map(index).to_numpy()
        ai = targets["away"].map(index).to_numpy()
        days = days_ahead(targets, frozen["date"].max())
        x = targets["home_goals"].to_numpy()[None, :]
        y = targets["away_goals"].to_numpy()[None, :]
        for sigma, omega, beta, tau, lag in ((s_, o_, b_, t_, g_) for s_ in sigmas
                                             for o_ in omegas for b_ in betas for t_ in taus
                                             for g_ in lags):
            z = centre(raw * np.tile(team_drift_scales(turnover, teams, beta), 2), n)
            # marginal drift of a match d days ahead: σ·sqrt(d/30)·z
            shift = z[:, None, :] * (sigma * np.sqrt(days / DAYS_PER_UNIT))[None, :, None]
            shift = shift + tau * jump[:, None, :]
            if lag:
                shift = shift + centre(jump_raw * (lag * sigma * lag_unit), n)[:, None, :]
            r = np.arange(len(targets))
            att_h = theta[:, hi] + shift[:, r, hi]
            att_a = theta[:, ai] + shift[:, r, ai]
            def_h = theta[:, n + hi] + shift[:, r, n + hi]
            def_a = theta[:, n + ai] + shift[:, r, n + ai]
            base, home = theta[:, [2 * n]], theta[:, [2 * n + 1]]
            lam = np.exp(base + home + att_h + def_a + omega * e[0])
            mu = np.exp(base + att_a + def_h + omega * e[1])
            log_p = logsumexp(_poisson_logpmf(x, lam) + _poisson_logpmf(y, mu), axis=0)
            log_p -= np.log(n_draws)
            rows.append(pd.DataFrame({
                "freeze_round": freeze, "match_id": targets["match_id"].to_numpy(),
                "days_ahead": days, "sigma": sigma, "omega": omega, "beta": beta,
                "tau": tau, "lag": lag, "log_score": log_p,
            }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["freeze_round", "match_id", "days_ahead", "sigma", "omega", "beta", "tau",
                 "lag", "log_score"])


def drift_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean log loss (−log score) per (σ, ω, β, τ), overall and by forecast horizon."""
    bins = [-1, 30, 90, 180, 10_000]
    labels = ["≤30 d", "31–90 d", "91–180 d", ">180 d"]
    df = scores.assign(log_loss=-scores["log_score"],
                       horizon=pd.cut(scores["days_ahead"], bins, labels=labels))
    keys = [k for k in ("sigma", "omega", "beta", "tau", "lag") if k in df]
    by_h = df.pivot_table(index=keys, columns="horizon", values="log_loss",
                          aggfunc="mean", observed=False)
    overall = df.groupby(keys)["log_loss"].agg(["size", "mean"])
    overall.columns = ["predictions", "log_loss"]
    return overall.join(by_h).reset_index()
