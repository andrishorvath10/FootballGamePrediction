"""League table and Monte Carlo season simulation.

League table
------------
3 points for a win, 1 for a draw, plus any point adjustment (MLSZ "büntetőpont"
deductions). Equal points are separated as in the MLSZ county competition rules
(versenykiírás, "A bajnokság helyezéseinek eldöntése"):

1. more wins, 2. goal difference, 3. more goals scored,
4. points in the matches between the tied teams, 5. goal difference in those,
6. fair play ranking, 7. drawing lots.

Fair play is not in the data, so 6-7 become alphabetical order in the table and
a random draw in the simulation. Checked against 387 official MLSZ table
snapshots (see README).

Monte Carlo simulation
----------------------
1. Fit the team model on the matches played so far.
2. For every remaining fixture compute (λ, μ). With ``uncertainty=True`` every
   simulated season first draws its *own* team strengths from the Laplace
   posterior (:func:`football_predictor.model.mle.fit_posterior`), so λ, μ differ
   per simulation: "how good is this team really?" is uncertain too, not just the
   match results. Early in the season this widens the distribution noticeably.
3. Draw the score of every remaining fixture ``n_sims`` times: independent
   Poisson(λ), Poisson(μ) - with Dixon–Coles via rejection sampling (propose
   independent Poissons, accept with probability τ(x, y) / max τ).
4. Add the simulated points to the current table, rank every simulated season.
5. Counting over the simulations gives P(finishing 1st), P(top 3), expected points,
   the full distribution of final positions, etc.

Everything is vectorised with numpy: one array of shape (n_sims, n_fixtures) per
side, so 10 000 seasons take well under a second.

Without ``uncertainty`` the strengths are treated as known, which makes the
probabilities over-confident, mostly in autumn (see the README backtest).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_predictor.db.models import MatchStatus
from football_predictor.model.dataset import played_only
from football_predictor.model.predict import OPEN_STATUSES
from football_predictor.model.strength import StrengthParams, TeamStrengths, fit_model

logger = logging.getLogger(__name__)

TABLE_COLUMNS = ["team", "played", "won", "drawn", "lost", "gf", "ga", "gd",
                 "adjustment", "points"]


def _pair_results(played: pd.DataFrame, teams: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """n×n matrices: points / goal difference team i collected against team j."""
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    pts, gd = np.zeros((n, n)), np.zeros((n, n))
    hi = played["home"].map(index).to_numpy(dtype=int)
    ai = played["away"].map(index).to_numpy(dtype=int)
    hg, ag = played["home_goals"].to_numpy(), played["away_goals"].to_numpy()
    np.add.at(pts, (hi, ai), np.where(hg > ag, 3, np.where(hg == ag, 1, 0)))
    np.add.at(pts, (ai, hi), np.where(ag > hg, 3, np.where(hg == ag, 1, 0)))
    np.add.at(gd, (hi, ai), hg - ag)
    np.add.at(gd, (ai, hi), ag - hg)
    return pts, gd


def league_table(
    matches: pd.DataFrame, point_adjustments: dict[str, int] | None = None
) -> pd.DataFrame:
    """Current standings from the played matches (all teams of the fixture list).

    ``points`` already includes ``adjustment`` (e.g. -3 for a deduction).
    """
    played = played_only(matches)
    teams = sorted(set(matches["home"]) | set(matches["away"]))
    hg, ag = played["home_goals"].to_numpy(), played["away_goals"].to_numpy()
    rows = pd.concat([
        pd.DataFrame({"team": played["home"].to_numpy(), "gf": hg, "ga": ag}),
        pd.DataFrame({"team": played["away"].to_numpy(), "gf": ag, "ga": hg}),
    ])
    rows["won"] = (rows["gf"] > rows["ga"]).astype(int)
    rows["drawn"] = (rows["gf"] == rows["ga"]).astype(int)
    rows["lost"] = (rows["gf"] < rows["ga"]).astype(int)
    table = rows.groupby("team").agg(
        played=("gf", "size"), won=("won", "sum"), drawn=("drawn", "sum"),
        lost=("lost", "sum"), gf=("gf", "sum"), ga=("ga", "sum"),
    ).reindex(teams, fill_value=0)
    table["gd"] = table["gf"] - table["ga"]
    adjustments = point_adjustments or {}
    table["adjustment"] = [adjustments.get(t, 0) for t in teams]
    table["points"] = 3 * table["won"] + table["drawn"] + table["adjustment"]
    table = table.astype(int)

    # Criteria 4-5: head-to-head among teams still level on points, wins, gd, gf.
    pair_pts, pair_gd = _pair_results(played, teams)
    table["h2h_points"], table["h2h_gd"] = 0, 0
    level = table.groupby(["points", "won", "gd", "gf"])["played"].transform("size") > 1
    for _, group in table[level].groupby(["points", "won", "gd", "gf"]):
        idx = [teams.index(t) for t in group.index]
        sub = np.ix_(idx, idx)
        table.loc[group.index, "h2h_points"] = pair_pts[sub].sum(axis=1).astype(int)
        table.loc[group.index, "h2h_gd"] = pair_gd[sub].sum(axis=1).astype(int)

    table = table.reset_index().rename(columns={"index": "team"})
    table = table.sort_values(
        ["points", "won", "gd", "gf", "h2h_points", "h2h_gd", "team"],
        ascending=[False, False, False, False, False, False, True])
    table.insert(0, "pos", range(1, len(table) + 1))
    return table.reset_index(drop=True)[["pos", *TABLE_COLUMNS]]


def hide_results_from(matches: pd.DataFrame, round_: int) -> pd.DataFrame:
    """Pretend the season stands just before round ``round_`` (to backtest).

    Hides the results of rounds >= ``round_`` *and* of earlier-round matches played on
    or after that round's first date (postponed games): those were not known then.
    """
    out = matches.copy()
    cutoff = out.loc[out["round"] == round_, "date"].min()
    later = out["round"] >= round_
    if pd.notna(cutoff):
        later |= out["date"] >= cutoff
    later &= out["status"] == MatchStatus.PLAYED.value
    out.loc[later, ["home_goals", "away_goals"]] = np.nan
    out.loc[later, "status"] = MatchStatus.SCHEDULED.value
    return out


@dataclass(frozen=True)
class TeamShift:
    """An expected change of a team's strength (expert knowledge, not estimated).

    ``change`` = +0.2: from ``start`` on the team scores 1.2× and concedes 1/1.2×
    the goals the model expects, i.e. log attack += ln 1.2, log defence −= ln 1.2.
    ``start`` None = from the next fixture on.
    """

    change: float
    start: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if self.change <= -1:
            raise ValueError("change must be > -100 %")

    @property
    def log_shift(self) -> float:
        return float(np.log1p(self.change))

    @classmethod
    def parse(cls, text: str) -> TeamShift:
        """``"+20%"``, ``"-0.1"`` or ``"+20%@2027-03-01"``."""
        value, _, start = text.partition("@")
        value = value.strip()
        change = float(value.rstrip("%")) / 100 if value.endswith("%") else float(value)
        return cls(change, pd.Timestamp(start) if start.strip() else None)


def shift_log_rates(fixtures: pd.DataFrame,
                    shifts: dict[str, TeamShift]) -> tuple[np.ndarray, np.ndarray]:
    """Additive changes of log λ and log μ for every fixture from ``shifts``.

    A shifted home team raises its own λ (attack) and lowers μ (its defence);
    a shifted away team the other way round. Fixtures before ``start`` (or
    without a date when a start is given) are not affected.
    """
    d_lam = np.zeros(len(fixtures))
    d_mu = np.zeros(len(fixtures))
    dates = fixtures["date"]
    for team, shift in shifts.items():
        active = np.ones(len(fixtures), dtype=bool) if shift.start is None else (
            (dates >= shift.start).fillna(False).to_numpy())
        home = (fixtures["home"] == team).to_numpy() & active
        away = (fixtures["away"] == team).to_numpy() & active
        d_lam += shift.log_shift * (home.astype(float) - away)
        d_mu += shift.log_shift * (away.astype(float) - home)
    return d_lam, d_mu


@dataclass
class SimulationResult:
    """Output of :func:`simulate_season`."""

    summary: pd.DataFrame  # one row per team, sorted by expected points
    positions: pd.DataFrame  # team x final position -> probability
    remaining: int  # number of simulated fixtures
    n_sims: int
    strengths: TeamStrengths
    uncertainty: bool = False
    drift: float = 0.0


def _dc_tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray,
            rho: float) -> np.ndarray:
    """Dixon–Coles factor τ(x, y) elementwise (1 outside the four low-score cells)."""
    tau = np.ones_like(lam)
    tau = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1 + lam * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1 + mu * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
    return np.maximum(tau, 0.0)


def _sample_scores(
    rng: np.random.Generator, lam: np.ndarray, mu: np.ndarray, rho: float
) -> tuple[np.ndarray, np.ndarray]:
    """Simulated home and away goals, same shape as ``lam`` / ``mu``.

    Dixon–Coles target p(x, y) = τ(x, y)·q(x, y) with q = independent Poissons, so
    rejection sampling is exact: propose from q, accept with τ / M where M ≥ τ.
    τ is close to 1, so almost every proposal is accepted.
    """
    home, away = rng.poisson(lam), rng.poisson(mu)
    if rho == 0.0:
        return home, away
    bound = np.maximum.reduce([np.ones_like(lam), 1 - lam * mu * rho, 1 + lam * rho,
                               1 + mu * rho, np.full_like(lam, 1 - rho)])
    todo = np.ones(lam.shape, dtype=bool)
    while True:
        idx = np.nonzero(todo)
        tau = _dc_tau(home[idx], away[idx], lam[idx], mu[idx], rho)
        rejected = rng.random(tau.shape) * bound[idx] >= tau
        if not rejected.any():
            return home, away
        todo[:] = False
        todo[tuple(i[rejected] for i in idx)] = True
        home[todo], away[todo] = rng.poisson(lam[todo]), rng.poisson(mu[todo])


def _resolve_head_to_head(
    order: np.ndarray, keys: np.ndarray, lots: np.ndarray,
    pair_played: tuple[np.ndarray, np.ndarray],
    hi: np.ndarray, ai: np.ndarray, hg: np.ndarray, ag: np.ndarray,
) -> None:
    """Re-order (in place) runs of teams level on points, wins, gd and gf by their
    head-to-head points, then head-to-head goal difference, then lots.

    Only the few simulations that contain such a run are visited; for each one
    the pairwise results are the played ones plus that simulation's draws.
    """
    ranked = np.take_along_axis(keys, order[..., None], axis=1)  # (sims, n, 4)
    level = (ranked[:, 1:] == ranked[:, :-1]).all(axis=-1)  # rank r level with r+1
    for s in np.nonzero(level.any(axis=1))[0]:
        pts, gd = pair_played[0].copy(), pair_played[1].copy()
        h, a = hg[s], ag[s]
        np.add.at(pts, (hi, ai), np.where(h > a, 3, np.where(h == a, 1, 0)))
        np.add.at(pts, (ai, hi), np.where(a > h, 3, np.where(h == a, 1, 0)))
        np.add.at(gd, (hi, ai), h - a)
        np.add.at(gd, (ai, hi), a - h)
        r, n = 0, order.shape[1]
        while r < n - 1:
            end = r
            while end < n - 1 and level[s, end]:
                end += 1
            if end > r:  # ranks r..end are level
                group = order[s, r:end + 1]
                sub = np.ix_(group, group)
                h2h_pts, h2h_gd = pts[sub].sum(axis=1), gd[sub].sum(axis=1)
                order[s, r:end + 1] = group[np.lexsort((lots[s, group], -h2h_gd, -h2h_pts))]
            r = end + 1


def _final_positions(
    table: pd.DataFrame, played: pd.DataFrame, teams: list[str],
    hi: np.ndarray, ai: np.ndarray, hg: np.ndarray, ag: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Final position and points of every team in every simulated season.

    ``table``: current table indexed by team (in ``teams`` order); ``hg``/``ag``:
    (n_sims, n_fixtures) simulated goals of the fixtures ``hi`` vs ``ai``.
    """
    n_sims, n = hg.shape[0], len(teams)

    def start(col: str) -> np.ndarray:
        return np.tile(table[col].to_numpy(dtype=float), (n_sims, 1))

    pts, won, gd, gf = start("points"), start("won"), start("gd"), start("gf")
    rows = np.arange(n_sims)[:, None]
    np.add.at(pts, (rows, hi), np.where(hg > ag, 3, np.where(hg == ag, 1, 0)))
    np.add.at(pts, (rows, ai), np.where(ag > hg, 3, np.where(hg == ag, 1, 0)))
    np.add.at(won, (rows, hi), hg > ag)  # add.at: a team may appear several times
    np.add.at(won, (rows, ai), ag > hg)
    np.add.at(gd, (rows, hi), hg - ag)
    np.add.at(gd, (rows, ai), ag - hg)
    np.add.at(gf, (rows, hi), hg)
    np.add.at(gf, (rows, ai), ag)

    # Rank: points, wins, goal difference, goals for (MLSZ criteria 1-3), then a
    # random key. np.lexsort sorts by the LAST key first, along each row.
    lots = rng.random((n_sims, n))
    order = np.lexsort((lots, -gf, -gd, -won, -pts), axis=1)
    _resolve_head_to_head(order, np.stack([pts, won, gd, gf], axis=-1), lots,
                          _pair_results(played, teams), hi, ai, hg, ag)
    position = np.empty_like(order)
    np.put_along_axis(position, order, np.arange(1, n + 1)[None, :].repeat(n_sims, 0), axis=1)
    return position, pts


def simulate_season(
    matches: pd.DataFrame,
    params: StrengthParams | None = None,
    n_sims: int = 10_000,
    seed: int = 0,
    point_adjustments: dict[str, int] | None = None,
    uncertainty: bool = False,
    drift: float = 0.0,
    drift_scales: dict[str, float] | None = None,
    shifts: dict[str, TeamShift] | None = None,
) -> SimulationResult:
    """Simulate the rest of the season ``n_sims`` times.

    ``uncertainty=True`` draws fresh team strengths for every simulated season
    from the Laplace posterior; ``drift`` > 0 additionally lets them follow a
    random walk with that standard deviation per 30 days until each fixture, plus
    the lag of the time-weighted fit behind it (:func:`~football_predictor.model.drift.lag_sd`;
    :mod:`football_predictor.model.drift`). Both need ``model`` "mle" or "dc".
    ``drift_scales`` multiplies σ for individual teams (expert knowledge, e.g.
    ``{"X SE": 2.0}`` for a team that is rebuilding its squad; default 1).
    ``shifts`` move individual teams' strengths by an expected amount from a
    given date (:class:`TeamShift`); works with every model.
    """
    params = params or StrengthParams()
    played = played_only(matches)
    teams = sorted(set(matches["home"]) | set(matches["away"]))
    index = {t: i for i, t in enumerate(teams)}
    table = league_table(matches, point_adjustments).set_index("team").loc[teams]
    remaining = matches[matches["status"].isin(OPEN_STATUSES)]
    n = len(teams)
    hi = remaining["home"].map(index).to_numpy(dtype=int)
    ai = remaining["away"].map(index).to_numpy(dtype=int)
    rng = np.random.default_rng(seed)

    if uncertainty or drift > 0:
        from football_predictor.model.drift import (
            centred_normal,
            days_ahead,
            drift_paths,
            lag_sd,
        )
        from football_predictor.model.mle import fit_posterior

        posterior = fit_posterior(played, params, teams=teams)
        strengths = posterior.strengths
        if uncertainty:
            theta = posterior.sample(rng, n_sims)  # one parameter vector per season
        else:
            theta = np.tile(posterior.mean, (n_sims, 1))
        base, home_adv = theta[:, [2 * n]], theta[:, [2 * n + 1]]
        # strengths at each fixture's date: today's + the season's random walk
        days = days_ahead(remaining, played["date"].max())
        dates, at = np.unique(days, return_inverse=True)
        scales = np.array([(drift_scales or {}).get(t, 1.0) for t in teams])
        walk = drift_paths(rng, n_sims, n, dates, drift, scales)  # (n_sims, n_dates, 2n)
        if drift > 0:
            # the fit lags behind drifting teams (large after a winter break): one
            # persistent offset per team and season, sd = σ · lag_sd
            lag = centred_normal(rng, (n_sims,), n, drift * lag_sd(played, params, teams))
            walk = walk + lag[:, None, :]
        att_h = theta[:, hi] + walk[:, at, hi]
        att_a = theta[:, ai] + walk[:, at, ai]
        def_h = theta[:, n + hi] + walk[:, at, n + hi]
        def_a = theta[:, n + ai] + walk[:, at, n + ai]
        lam = np.exp(base + home_adv + att_h + def_a)  # (n_sims, n_fixtures)
        mu = np.exp(base + att_a + def_h)
    else:
        strengths = fit_model(played, params, teams=teams)
        xg = np.array([strengths.expected_goals(h, a) for h, a in
                       zip(remaining["home"], remaining["away"], strict=True)]).reshape(-1, 2)
        lam = np.broadcast_to(xg[:, 0], (n_sims, len(remaining))).copy()
        mu = np.broadcast_to(xg[:, 1], (n_sims, len(remaining))).copy()
    if shifts:
        unknown = set(shifts) - set(teams)
        if unknown:
            raise ValueError(f"Unknown team(s) in shifts: {sorted(unknown)}")
        d_lam, d_mu = shift_log_rates(remaining, shifts)
        lam, mu = lam * np.exp(d_lam), mu * np.exp(d_mu)
    hg, ag = _sample_scores(rng, lam, mu, strengths.rho)

    position, pts = _final_positions(table, played, teams, hi, ai, hg, ag, rng)

    pos_probs = np.stack([(position == p).mean(axis=0) for p in range(1, n + 1)], axis=1)
    positions = pd.DataFrame(pos_probs, index=teams, columns=range(1, n + 1))
    summary = pd.DataFrame({
        "team": teams,
        "points_now": table["points"].to_numpy(),
        "played": table["played"].to_numpy(),
        "exp_points": pts.mean(axis=0),
        "points_p10": np.percentile(pts, 10, axis=0),
        "points_p90": np.percentile(pts, 90, axis=0),
        "exp_position": position.mean(axis=0),
        "p_title": (position == 1).mean(axis=0),
        "p_top3": (position <= 3).mean(axis=0),
        "p_last": (position == n).mean(axis=0),
    }).sort_values(["exp_points", "exp_position"], ascending=[False, True])
    positions = positions.loc[summary["team"]]
    return SimulationResult(summary=summary.reset_index(drop=True), positions=positions,
                            remaining=len(remaining), n_sims=n_sims, strengths=strengths,
                            uncertainty=uncertainty, drift=drift)


def position_scores(result: SimulationResult, final: pd.DataFrame,
                    eps: float = 1e-3) -> pd.DataFrame:
    """Score a simulation against the final table, one row per team.

    * ``log_loss``: -log P(the position the team actually finished in), clipped at eps;
    * ``rps``: ranked probability score over positions - compares the cumulative
      distributions, so predicting 2nd for the champion is penalised less than 9th.
    """
    pos = result.positions
    n = pos.shape[1]
    actual = final.set_index("team")["pos"].reindex(pos.index)
    p_actual = pos.to_numpy()[np.arange(n), actual.to_numpy() - 1]
    cum = pos.cumsum(axis=1).to_numpy()
    hit = (np.arange(1, n + 1)[None, :] >= actual.to_numpy()[:, None]).astype(float)
    summary = result.summary.set_index("team").loc[pos.index]
    return pd.DataFrame({
        "team": pos.index,
        "final_pos": actual.to_numpy(),
        "exp_position": summary["exp_position"].to_numpy(),
        "p_actual": p_actual,
        "log_loss": -np.log(np.clip(p_actual, eps, 1.0)),
        "rps": ((cum[:, :-1] - hit[:, :-1]) ** 2).sum(axis=1) / (n - 1),
        "p_title": summary["p_title"].to_numpy(),
    })


def backtest_simulation(
    matches: pd.DataFrame,
    params: StrengthParams,
    fractions: tuple[float, ...] = (1 / 3, 1 / 2, 2 / 3),
    n_sims: int = 5000,
    seed: int = 0,
    point_adjustments: dict[str, int] | None = None,
    drift: float = 0.0,
) -> pd.DataFrame:
    """Freeze a finished season part-way, simulate the rest with fixed strengths,
    with parameter uncertainty, and (if ``drift`` > 0) with uncertainty plus
    drift; score each against the real final table.

    ``point_adjustments`` (the final deductions) are applied to the final table and,
    as an approximation since their timing is unknown, to every frozen simulation.
    """
    final = league_table(matches, point_adjustments)
    last_round = int(matches["round"].max())
    rows = []
    for frac in fractions:
        frozen_at = int(round(frac * last_round)) + 1
        frozen = hide_results_from(matches, frozen_at)
        variants = [("fixed", False, 0.0), ("uncertain", True, 0.0)]
        if drift > 0:
            variants.append(("uncertain + drift", True, drift))
        for label, uncertainty, drift_ in variants:
            res = simulate_season(frozen, params, n_sims=n_sims, seed=seed,
                                  uncertainty=uncertainty, drift=drift_,
                                  point_adjustments=point_adjustments)
            scores = position_scores(res, final)
            champion = scores.loc[scores["final_pos"] == 1].iloc[0]
            rows.append({
                "fraction": round(frac, 2), "frozen_at_round": frozen_at,
                "strengths": label, "teams": len(scores),
                "log_loss": scores["log_loss"].mean(), "rps": scores["rps"].mean(),
                "pos_mae": (scores["exp_position"] - scores["final_pos"]).abs().mean(),
                "p_champion": champion["p_title"],
                "favourite_won": bool(scores["p_title"].idxmax() == champion.name),
            })
    return pd.DataFrame(rows)
