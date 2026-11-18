"""Goalscorer model: how likely is a given player to score in a given match?

Idea: Poisson thinning
----------------------
The team model gives the team's expected goals λ_T. If the team's goal count is
Poisson(λ_T) and every goal is independently scored by player i with probability
q_i, then player i's goal count is itself Poisson(λ_T · q_i) ("thinning" of a
Poisson process). Hence::

    xG_i     = λ_T · q_i
    P(score) = 1 - exp(-xG_i)               ("anytime scorer")
    P(2+)    = 1 - exp(-xG_i) · (1 + xG_i)

q_i is split into two parts that are estimated separately:

    q_i = share_i × f_i

* ``f_i`` = expected fraction of the match the player is on the pitch: his average
  minutes / 90 over the team's last ``form_matches`` matches (0 if he did not
  appear, which also handles transfers and injuries).
* ``share_i`` = the player's share of the team's goals *while he is on the pitch*.

Estimating the share (Gamma–Poisson shrinkage, as in the team model)
-------------------------------------------------------------------
Exposure = how many team goals the player "had the chance" to score::

    E_i = Σ_matches w · team_goals · minutes_i / 90

(assumes the team's goals are spread evenly over the 90 minutes). The model is
goals_i ~ Poisson(share_i · E_i); with a Gamma prior centred on the league average
share s0 the posterior mean is::

    share_i = (Σ w·goals_i + k · s0) / (E_i + k)

* s0 = Σ goals / Σ exposure over the whole league - it is below 1/11 because own
  goals and goals with an unknown scorer are not credited to anyone.
* k (shrinkage) = pseudo team-goals of exposure at the league-average share. A
  player with 1 goal in 2 matches is not yet a 50 % scorer.
* ``w = exp(-ξ·days_ago)`` optional time decay, like the team model.

Over-dispersion: negative binomial instead of Poisson
-----------------------------------------------------
Backtesting showed that xG_i is right *on average* but P(score) was too high for
the favourites: a striker's goals cluster (a hat-trick in a 9-0 win rather than one
goal in each of three games), so Var(goals) > mean, unlike the Poisson. With
``dispersion = r`` the player's goals follow a negative binomial with mean xG and
variance xG + xG²/r (= a Poisson whose rate is Gamma-distributed match to match)::

    P(0) = (r / (r + xG))^r        P(1) = r · xG/(r + xG) · P(0)

r → ∞ gives back the Poisson.

Only team-matches with lineup data contribute (both goals and exposure), otherwise a
goal without minutes would inflate the share.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from football_predictor.model.dataset import played_only
from football_predictor.model.strength import StrengthParams, fit_model, time_weights

MATCH_MINUTES = 90.0


@dataclass(frozen=True)
class ScorerParams:
    """Hyper-parameters of the scorer model."""

    shrinkage: float = 2.0  # k: pseudo team-goals of exposure at the league share
    xi: float = 0.0  # time decay per day
    form_matches: int = 5  # expected minutes = average over the team's last N matches
    use_skill: bool = True  # False = baseline: every player gets the league share s0
    dispersion: float | None = None  # negative binomial size r; None = Poisson

    def label(self) -> str:
        """Short description, e.g. ``scorer(k=2, xi=0, form5, nb1.5)``."""
        nb = f", nb{self.dispersion:g}" if self.dispersion else ""
        if not self.use_skill:
            return f"minutes only (form{self.form_matches}{nb})"
        return f"scorer(k={self.shrinkage:g}, xi={self.xi:g}, form{self.form_matches}{nb})"


@dataclass
class ScorerModel:
    """Fitted scorer model: one row per (team, player_id)."""

    players: pd.DataFrame
    league_share: float
    params: ScorerParams

    def candidates(self, team: str) -> pd.DataFrame:
        """Players of ``team`` who are expected to play (exp_minutes > 0)."""
        df = self.players[self.players["team"] == team]
        return df[df["exp_minutes"] > 0]


def team_match_rows(played: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team) with the team's official goals."""
    home = played[["match_id", "date", "home", "home_goals"]].set_axis(
        ["match_id", "date", "team", "team_goals"], axis=1)
    away = played[["match_id", "date", "away", "away_goals"]].set_axis(
        ["match_id", "date", "team", "team_goals"], axis=1)
    return pd.concat([home, away], ignore_index=True)


def _expected_minutes(apps: pd.DataFrame, n: int) -> pd.Series:
    """Average minutes (missing = 0) over each team's last ``n`` matches with lineups."""
    team_matches = apps[["team", "match_id", "date"]].drop_duplicates()
    recent = (team_matches.sort_values("date", kind="stable")
              .groupby("team").tail(n))
    n_recent = recent.groupby("team").size()
    sub = apps.merge(recent[["team", "match_id"]], on=["team", "match_id"])
    total = sub.groupby(["team", "player_id"])["minutes"].sum()
    teams = total.index.get_level_values("team")
    return total / n_recent.reindex(teams).to_numpy()


def fit_scorers(
    played: pd.DataFrame,
    appearances: pd.DataFrame,
    goals: pd.DataFrame,
    params: ScorerParams | None = None,
    as_of: pd.Timestamp | None = None,
) -> ScorerModel:
    """Estimate every player's goal share and expected minutes.

    Args:
        played: played matches (columns match_id, date, home, away, home_goals, away_goals).
        appearances: columns match_id, team, player_id, player, minutes.
        goals: columns match_id, team, player_id, is_own_goal (and is_penalty).
        params: hyper-parameters.
        as_of: reference date for the time decay (default: last match date).
    """
    params = params or ScorerParams()
    played = played_only(played) if "status" in played else played
    if played.empty:
        raise ValueError("No played matches to fit on")
    as_of = as_of if as_of is not None else played["date"].max()

    tm = team_match_rows(played)
    apps = appearances.merge(tm, on=["match_id", "team"])  # played matches only
    with_lineup = apps[["match_id", "team"]].drop_duplicates()
    apps = apps.assign(w=time_weights(apps["date"], as_of, params.xi))
    apps["exposure"] = apps["w"] * apps["team_goals"] * apps["minutes"] / MATCH_MINUTES

    scored = goals[~goals["is_own_goal"].astype(bool) & goals["player_id"].notna()]
    scored = scored.merge(tm, on=["match_id", "team"]).merge(with_lineup,
                                                              on=["match_id", "team"])
    scored = scored.assign(w=time_weights(scored["date"], as_of, params.xi),
                           player_id=scored["player_id"].astype(int))

    key = ["team", "player_id"]
    per_player = apps.groupby(key).agg(
        player=("player", "last"), apps=("minutes", lambda m: int((m > 0).sum())),
        minutes=("minutes", "sum"), exposure=("exposure", "sum"),
    )
    g = scored.groupby(key).agg(goals=("w", "size"), goals_w=("w", "sum"),
                                penalties=("is_penalty", "sum"))
    per_player = per_player.join(g, how="outer")
    if per_player["player"].isna().any():  # scorer without a lineup entry (data gap)
        names = scored.groupby(key)["player"].last()
        per_player["player"] = per_player["player"].fillna(names)
    per_player = per_player.fillna({"apps": 0, "minutes": 0.0, "exposure": 0.0, "goals": 0,
                                    "goals_w": 0.0, "penalties": 0})

    total_exposure = per_player["exposure"].sum()
    s0 = float(per_player["goals_w"].sum() / total_exposure) if total_exposure > 0 else 0.0
    if params.use_skill:
        k = params.shrinkage
        denom = per_player["exposure"] + k
        share = (per_player["goals_w"] + k * s0) / denom.where(denom > 0)
        per_player["share"] = share.fillna(s0)
    else:
        per_player["share"] = s0

    per_player["exp_minutes"] = _expected_minutes(apps, params.form_matches).reindex(
        per_player.index).fillna(0.0)
    per_player = per_player.astype({"apps": int, "goals": int, "penalties": int})
    return ScorerModel(players=per_player.reset_index(), league_share=s0, params=params)


def scorer_probabilities(
    model: ScorerModel, team: str, lam: float, whole_roster: bool = False
) -> pd.DataFrame:
    """Anytime-scorer probabilities for one team given its expected goals ``lam``.

    By default only expected players (exp_minutes > 0); ``whole_roster`` returns every
    player who has played for the team (the others get probability 0).
    """
    if whole_roster:
        df = model.players[(model.players["team"] == team) & (model.players["apps"] > 0)].copy()
    else:
        df = model.candidates(team).copy()
    xg = lam * df["share"] * df["exp_minutes"] / MATCH_MINUTES
    r = model.params.dispersion
    if r:
        p0 = (r / (r + xg)) ** r
        p1 = r * xg / (r + xg) * p0
    else:
        p0 = np.exp(-xg)
        p1 = xg * p0
    df["xg"], df["p_score"], df["p_2plus"] = xg, 1 - p0, 1 - p0 - p1
    return df.sort_values("p_score", ascending=False).reset_index(drop=True)


def predict_fixture_scorers(
    matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    appearances: pd.DataFrame,
    goals: pd.DataFrame,
    team_params: StrengthParams | None = None,
    scorer_params: ScorerParams | None = None,
) -> pd.DataFrame:
    """Scorer probabilities for every team in ``fixtures``, using all played matches.

    Returns one row per (fixture, team, candidate player).
    """
    played = played_only(matches)
    teams = set(matches["home"]) | set(matches["away"])
    strengths = fit_model(played, team_params, teams=teams)
    model = fit_scorers(played, appearances, goals, scorer_params)
    parts = []
    for row in fixtures.itertuples():
        lam, mu = strengths.expected_goals(row.home, row.away)
        for team, opponent, team_xg in ((row.home, row.away, lam), (row.away, row.home, mu)):
            probs = scorer_probabilities(model, team, team_xg)
            parts.append(probs.assign(match_id=row.match_id, round=row.round, date=row.date,
                                      opponent=opponent, home=team == row.home,
                                      team_xg=team_xg))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# --------------------------------------------------------------------- evaluation
@dataclass(frozen=True)
class ScorerForecaster:
    """A scorer model plus the team model that provides λ (None = league average λ)."""

    name: str
    params: ScorerParams
    team_params: StrengthParams | None = None


def _league_average_xg(train: pd.DataFrame, fixtures: pd.DataFrame) -> dict[int, tuple]:
    home, away = float(train["home_goals"].mean()), float(train["away_goals"].mean())
    return {int(m): (home, away) for m in fixtures["match_id"]}


def scorer_walk_forward(
    matches: pd.DataFrame,
    appearances: pd.DataFrame,
    goals: pd.DataFrame,
    forecasters: list[ScorerForecaster],
    start_round: int = 4,
) -> pd.DataFrame:
    """Walk-forward scorer backtest (same protocol as the team backtest).

    For every round, fit on matches before the round, then for every team-match of
    the round that has lineup data predict P(score) for every player who has played
    for the team so far (the same rows for every forecaster, so scores are
    comparable; players outside the form window get probability 0). Outcome: scored
    at least one non-own goal. Players who end up not playing still count (outcome
    0), exactly as in real use.
    """
    played = played_only(matches)
    lineup_keys = set(zip(appearances["match_id"], appearances["team"], strict=True))
    scorer_goals = goals[~goals["is_own_goal"].astype(bool) & goals["player_id"].notna()]
    goal_counts = (scorer_goals.assign(player_id=scorer_goals["player_id"].astype(int))
                   .groupby(["match_id", "team", "player_id"]).size().to_dict())
    rows = []
    for rnd in sorted(played["round"].unique()):
        if rnd < start_round:
            continue
        test = played[played["round"] == rnd]
        cutoff = test["date"].min()
        train = played[played["date"] < cutoff]
        if train.empty or pd.isna(cutoff):
            continue
        train_ids = set(train["match_id"])
        tr_apps = appearances[appearances["match_id"].isin(train_ids)]
        tr_goals = goals[goals["match_id"].isin(train_ids)]
        if tr_apps.empty:
            continue
        teams = set(test["home"]) | set(test["away"])
        xg_cache: dict[StrengthParams | None, dict[int, tuple]] = {}
        model_cache: dict[ScorerParams, ScorerModel] = {}
        for fc in forecasters:
            if fc.team_params not in xg_cache:
                if fc.team_params is None:
                    xg_cache[None] = _league_average_xg(train, test)
                else:
                    st = fit_model(train, fc.team_params, teams=teams)
                    xg_cache[fc.team_params] = {
                        int(r.match_id): st.expected_goals(r.home, r.away)
                        for r in test.itertuples()}
            fit_key = replace(fc.params, dispersion=None)  # dispersion does not affect the fit
            if fit_key not in model_cache:
                model_cache[fit_key] = fit_scorers(train, tr_apps, tr_goals, fit_key)
            fitted = model_cache[fit_key]
            model = ScorerModel(fitted.players, fitted.league_share, fc.params)
            xgs = xg_cache[fc.team_params]
            for r in test.itertuples():
                for team, team_xg in ((r.home, xgs[int(r.match_id)][0]),
                                      (r.away, xgs[int(r.match_id)][1])):
                    if (r.match_id, team) not in lineup_keys:
                        continue
                    probs = scorer_probabilities(model, team, team_xg, whole_roster=True)
                    n_goals = [goal_counts.get((r.match_id, team, int(p)), 0)
                               for p in probs["player_id"]]
                    rows.append(pd.DataFrame({
                        "forecaster": fc.name, "round": rnd, "match_id": r.match_id,
                        "team": team, "player_id": probs["player_id"].to_numpy(),
                        "xg": probs["xg"].to_numpy(), "p_score": probs["p_score"].to_numpy(),
                        "goals": n_goals, "scored": [int(g > 0) for g in n_goals],
                    }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_binary(df: pd.DataFrame, eps: float = 1e-3) -> dict[str, float]:
    """Log loss, Brier, mean predicted vs. observed rate and top-1 hit rate.

    Probabilities are clipped to [eps, 1-eps]: a "0 %" player who then scores costs
    log(1/eps) ≈ 6.9 instead of infinity.
    """
    df = df.reset_index(drop=True)  # idxmax below needs unique labels
    p = np.clip(df["p_score"].to_numpy(), eps, 1 - eps)
    y = df["scored"].to_numpy()
    top = df.loc[df.groupby(["match_id", "team"])["p_score"].idxmax()]
    return {
        "player_matches": len(df),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "mean_p": float(p.mean()),
        "observed": float(y.mean()),
        "top1_hit": float(top["scored"].mean()),
    }


def scorer_score_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per forecaster, best log loss first."""
    table = pd.DataFrame([{"forecaster": name, **summarize_binary(group)}
                          for name, group in predictions.groupby("forecaster")])
    return table.sort_values("log_loss").reset_index(drop=True)


def binary_calibration(predictions: pd.DataFrame,
                       edges: tuple[float, ...] = (0, .05, .1, .2, .3, .4, .5, .7, 1.0),
                       ) -> pd.DataFrame:
    """Predicted vs. observed scoring rate in probability bins."""
    bins = pd.cut(predictions["p_score"], list(edges), include_lowest=True)
    return (predictions.groupby(bins, observed=True)
            .agg(predicted=("p_score", "mean"), observed=("scored", "mean"),
                 count=("scored", "size"))
            .reset_index().rename(columns={"p_score": "bin"}))


def scorer_grid(
    shrinkages: tuple[float, ...] = (4, 8, 16),
    xis: tuple[float, ...] = (0.0,),
    form_matches: tuple[int, ...] = (3, 5, 10, 20, 40),
    dispersions: tuple[float | None, ...] = (None, 0.5, 1.0, 2.0),
) -> list[ScorerParams]:
    """Parameter combinations to tune.

    (Earlier, wider runs showed time decay of the shares makes no difference and
    k < 4 or k > 16 is worse, so the default grid is narrowed to the useful region.)
    """
    return [ScorerParams(shrinkage=k, xi=xi, form_matches=n, dispersion=r)
            for k in shrinkages for xi in xis for n in form_matches for r in dispersions]


def anonymize(names: pd.Series, prefix: str = "Player") -> pd.Series:
    """Replace real names with stable pseudonyms (order of first appearance)."""
    codes = {name: f"{prefix} {i + 1:03d}" for i, name in enumerate(dict.fromkeys(names))}
    return names.map(codes)
