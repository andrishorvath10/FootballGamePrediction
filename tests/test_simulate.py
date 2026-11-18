"""Tests for the league table and the Monte Carlo season simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_predictor.model.simulate import hide_results_from, league_table, simulate_season
from football_predictor.model.strength import StrengthParams


def fixture_list(results: list[tuple[str, str, float | None, float | None]]) -> pd.DataFrame:
    rows = []
    for k, (h, a, hg, ag) in enumerate(results):
        rows.append({"match_id": k + 1, "round": k // 2 + 1,
                     "date": pd.Timestamp("2025-08-10") + pd.Timedelta(days=7 * (k // 2)),
                     "home": h, "away": a, "home_goals": hg, "away_goals": ag,
                     "status": "played" if hg is not None else "scheduled"})
    df = pd.DataFrame(rows)
    df[["home_goals", "away_goals"]] = df[["home_goals", "away_goals"]].astype(float)
    return df


def test_league_table_points_and_order() -> None:
    m = fixture_list([("A", "B", 2, 0), ("C", "D", 1, 1), ("B", "C", 3, 3), ("D", "A", 0, 1)])
    t = league_table(m).set_index("team")
    assert t.loc["A", ["played", "won", "points", "gf", "ga", "gd"]].tolist() == [2, 2, 6, 3, 0, 3]
    assert t.loc["C", ["drawn", "points", "gf"]].tolist() == [2, 2, 4]
    assert t.loc["B", "points"] == 1 and t.loc["D", "points"] == 1
    # B and C... C has 2 points; B (gd -2, gf 3) is ahead of D (gd -1? no: D gd -1, gf 1)
    assert list(t.sort_values("pos").index) == ["A", "C", "D", "B"]


def test_point_adjustment_and_unplayed_teams() -> None:
    m = fixture_list([("A", "B", 1, 0), ("C", "D", None, None)])
    t = league_table(m, point_adjustments={"A": -4}).set_index("team")
    assert t.loc["A", "points"] == -1
    assert t.loc["C", "played"] == 0  # team without a played match is still listed
    assert len(t) == 4


def test_hide_results_from() -> None:
    m = fixture_list([("A", "B", 1, 0), ("C", "D", 2, 2), ("B", "A", 0, 3), ("D", "C", 1, 0)])
    hidden = hide_results_from(m, 2)
    assert (hidden.loc[hidden["round"] == 2, "status"] == "scheduled").all()
    assert hidden.loc[hidden["round"] == 2, "home_goals"].isna().all()
    assert (hidden.loc[hidden["round"] == 1, "status"] == "played").all()


def test_hide_results_from_hides_postponed_games_played_later() -> None:
    m = fixture_list([("A", "B", 1, 0), ("C", "D", 2, 2), ("B", "A", 0, 3), ("D", "C", 1, 0)])
    # C-D of round 1 was postponed and played after round 2 had started
    m.loc[1, "date"] = m.loc[m["round"] == 2, "date"].min() + pd.Timedelta(days=3)
    hidden = hide_results_from(m, 2)
    assert hidden.loc[0, "status"] == "played"
    assert hidden.loc[1, "status"] == "scheduled"  # not known before round 2


def double_round_robin_league(seed: int = 0) -> pd.DataFrame:
    """4 teams, everyone played everyone once; second half unplayed."""
    teams = ["A", "B", "C", "D"]
    pairs = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C")]
    rng = np.random.default_rng(seed)
    first = [(h, a, float(rng.poisson(2.5 if h == "A" else 1.2)),
              float(rng.poisson(1.0))) for h, a in pairs]
    second = [(a, h, None, None) for h, a, *_ in first]
    assert set(teams) == {t for p in pairs for t in p}
    return fixture_list(first + second)


def test_simulation_probabilities_are_consistent() -> None:
    m = double_round_robin_league()
    res = simulate_season(m, StrengthParams(shrinkage=2.0), n_sims=4000, seed=1)
    assert res.remaining == 6
    pos = res.positions
    assert np.allclose(pos.sum(axis=1), 1)  # every team ends somewhere
    assert np.allclose(pos.sum(axis=0), 1)  # every position is taken by someone
    s = res.summary.set_index("team")
    assert s["p_title"].sum() == pytest.approx(1)
    assert s["p_last"].sum() == pytest.approx(1)
    assert (s["exp_points"] >= s["points_now"]).all()
    assert (s["exp_points"] <= s["points_now"] + 9).all()  # 3 remaining games each
    assert s["exp_position"].sum() == pytest.approx(1 + 2 + 3 + 4)


def test_finished_season_is_deterministic() -> None:
    m = fixture_list([("A", "B", 2, 0), ("C", "D", 0, 1), ("B", "C", 1, 0), ("D", "A", 0, 0)])
    res = simulate_season(m, n_sims=200)
    assert res.remaining == 0
    table = league_table(m)
    s = res.summary.set_index("team")
    for row in table.itertuples():
        assert s.loc[row.team, "exp_position"] == row.pos
    assert s.loc[table.iloc[0]["team"], "p_title"] == 1.0


def test_expected_points_match_analytic_value() -> None:
    """One remaining match: E[points] = 3·P(win) + P(draw) from the score matrix."""
    from football_predictor.model.poisson import predict_match

    m = double_round_robin_league()
    m = m[(m["status"] == "played") | (m["match_id"] == 7)]  # keep one open fixture
    params = StrengthParams(shrinkage=2.0)
    res = simulate_season(m, params, n_sims=200_000, seed=3)
    fx = m[m["match_id"] == 7].iloc[0]
    lam, mu = res.strengths.expected_goals(fx["home"], fx["away"])
    p = predict_match(fx["home"], fx["away"], lam, mu)
    s = res.summary.set_index("team")
    gained = s.loc[fx["home"], "exp_points"] - s.loc[fx["home"], "points_now"]
    assert gained == pytest.approx(3 * p.p_home + p.p_draw, abs=0.01)


def test_dixon_coles_sampling_runs() -> None:
    m = double_round_robin_league(seed=2)
    res = simulate_season(m, StrengthParams(model="dc", prior_sd=0.5), n_sims=500, seed=0)
    assert np.allclose(res.positions.sum(axis=1), 1)


def test_dixon_coles_rejection_sampler_matches_score_matrix() -> None:
    from football_predictor.model.poisson import dixon_coles_matrix
    from football_predictor.model.simulate import _sample_scores

    lam, mu, rho = 1.3, 0.9, -0.2
    shape = (400_000, 1)
    hg, ag = _sample_scores(np.random.default_rng(0), np.full(shape, lam), np.full(shape, mu), rho)
    mat = dixon_coles_matrix(lam, mu, rho, 15)
    for x, y in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]:
        assert ((hg == x) & (ag == y)).mean() == pytest.approx(mat[x, y], abs=0.003)


def test_uncertainty_widens_the_distribution() -> None:
    m = double_round_robin_league(seed=1)
    params = StrengthParams(model="mle", prior_sd=0.5, xi=0.0)
    fixed = simulate_season(m, params, n_sims=20_000, seed=0)
    unsure = simulate_season(m, params, n_sims=20_000, seed=0, uncertainty=True)
    assert unsure.uncertainty and not fixed.uncertainty
    assert np.allclose(unsure.positions.sum(axis=1), 1)
    assert np.allclose(unsure.positions.sum(axis=0), 1)
    band = lambda r: (r.summary["points_p90"] - r.summary["points_p10"]).mean()  # noqa: E731
    assert band(unsure) > band(fixed)
    # the favourite is less certain to win the title
    assert unsure.summary["p_title"].max() < fixed.summary["p_title"].max()


def test_position_scores_on_known_outcome() -> None:
    from football_predictor.model.simulate import position_scores

    m = fixture_list([("A", "B", 2, 0), ("C", "D", 0, 1), ("B", "C", 1, 0), ("D", "A", 0, 0)])
    res = simulate_season(m, n_sims=100)  # nothing left to play -> certain
    scores = position_scores(res, league_table(m))
    assert np.allclose(scores["log_loss"], 0) and np.allclose(scores["rps"], 0)


def test_backtest_simulation_compares_both_modes() -> None:
    from football_predictor.model.simulate import backtest_simulation

    m = double_round_robin_league(seed=2)
    rng = np.random.default_rng(5)
    open_ = m["status"] == "scheduled"
    m.loc[open_, ["home_goals", "away_goals"]] = rng.poisson(1.5, (open_.sum(), 2)).astype(float)
    m.loc[open_, "status"] = "played"
    out = backtest_simulation(m, StrengthParams(model="mle", prior_sd=0.5), fractions=(0.5,),
                              n_sims=500)
    assert list(out["strengths"]) == ["fixed", "uncertain"]
    assert (out["rps"] >= 0).all() and (out["log_loss"] >= 0).all()


# --- MLSZ tie-breakers and point adjustments -------------------------------------

def test_more_wins_beat_better_goal_difference() -> None:
    # A: W1 L1 (+5), B: D3 (0) -> both 3 points; MLSZ criterion 1 is wins, so A first.
    m = fixture_list([("A", "C", 6, 0), ("A", "D", 0, 1), ("B", "C", 0, 0), ("B", "D", 1, 1),
                      ("B", "E", 2, 2)])
    t = league_table(m).set_index("team")
    assert t.loc["A", "points"] == t.loc["B", "points"] == 3
    assert t.loc["A", "pos"] < t.loc["B", "pos"]


def test_head_to_head_decides_after_goals() -> None:
    # A and B level on points, wins, gd, gf; B won the match between them.
    m = fixture_list([("A", "B", 0, 1), ("A", "C", 2, 1), ("B", "D", 1, 2), ("C", "D", 0, 0)])
    t = league_table(m).set_index("team")
    assert tuple(t.loc["A", ["points", "won", "gd", "gf"]]) == tuple(
        t.loc["B", ["points", "won", "gd", "gf"]])
    assert t.loc["B", "pos"] < t.loc["A", "pos"]  # alphabetical order would say A


def test_point_adjustment_column_and_order() -> None:
    m = fixture_list([("A", "B", 1, 0), ("C", "D", 2, 1)])
    t = league_table(m, {"A": -3}).set_index("team")
    assert (t.loc["A", "adjustment"], t.loc["A", "points"]) == (-3, 0)
    assert t.loc["C", "pos"] == 1
    sim = simulate_season(m, n_sims=50, point_adjustments={"A": -3}).summary.set_index("team")
    assert sim.loc["A", "points_now"] == 0


def test_simulation_ranks_exactly_like_league_table() -> None:
    """_final_positions (vectorised + head-to-head pass) == league_table per season."""
    from football_predictor.model.dataset import played_only
    from football_predictor.model.simulate import _final_positions, _pair_results

    rng = np.random.default_rng(7)
    teams = ["A", "B", "C", "D", "E"]
    pairs = [(h, a) for h in teams for a in teams if h != a]
    m = fixture_list([(h, a, None, None) for h, a in pairs])
    first = m.index < 8  # 8 matches already played, low scores -> many ties
    m.loc[first, ["home_goals", "away_goals"]] = rng.poisson(0.6, (8, 2)).astype(float)
    m.loc[first, "status"] = "played"
    played, rest = played_only(m), m[~first]
    table = league_table(m).set_index("team").loc[teams]
    index = {t: i for i, t in enumerate(teams)}
    hi = rest["home"].map(index).to_numpy()
    ai = rest["away"].map(index).to_numpy()
    hg, ag = rng.poisson(0.6, (400, len(rest))), rng.poisson(0.6, (400, len(rest)))
    position, _ = _final_positions(table, played, teams, hi, ai, hg, ag, rng)

    checked = 0
    for s in range(400):
        final = m.copy()
        final.loc[~first, "home_goals"], final.loc[~first, "away_goals"] = hg[s], ag[s]
        final.loc[~first, "status"] = "played"
        expected = league_table(final).set_index("team")
        # skip seasons where teams stay level even on head-to-head (lots vs alphabet)
        pp, pg = _pair_results(played_only(final), teams)
        keys = expected[["points", "won", "gd", "gf"]].apply(tuple, axis=1)
        if any(len(g) > 1 and len({(pp[np.ix_(ix, ix)].sum(1)[k], pg[np.ix_(ix, ix)].sum(1)[k])
                                   for k in range(len(ix))}) < len(ix)
               for g in keys.groupby(keys).groups.values()
               for ix in [[index[t] for t in g]]):
            continue
        checked += 1
        assert list(position[s]) == [expected.loc[t, "pos"] for t in teams], s
    assert checked > 200


# --- expert assumptions: expected strength changes -------------------------------

def test_team_shift_parsing() -> None:
    from football_predictor.model.simulate import TeamShift

    assert TeamShift.parse("+20%") == TeamShift(0.2)
    assert TeamShift.parse("-0.1") == TeamShift(-0.1)
    shift = TeamShift.parse("+25%@2027-03-01")
    assert shift.start == pd.Timestamp("2027-03-01")
    assert shift.log_shift == pytest.approx(np.log(1.25))
    with pytest.raises(ValueError):
        TeamShift.parse("-150%")
    with pytest.raises(ValueError):
        TeamShift.parse("sok")


def test_shift_log_rates_signs_and_start_date() -> None:
    from football_predictor.model.simulate import TeamShift, shift_log_rates

    fx = pd.DataFrame({"home": ["A", "B", "A"], "away": ["B", "A", "C"],
                       "date": pd.to_datetime(["2026-11-20", "2027-03-10", "2027-04-01"])})
    d_lam, d_mu = shift_log_rates(fx, {"A": TeamShift(0.2, pd.Timestamp("2027-03-01"))})
    k = np.log(1.2)
    assert np.allclose(d_lam, [0, -k, k])  # before the start: untouched
    assert np.allclose(d_mu, [0, k, -k])  # away A attacks better, home A concedes less


def test_shift_moves_expected_points_by_the_analytic_amount() -> None:
    from football_predictor.model.poisson import predict_match
    from football_predictor.model.simulate import TeamShift

    m = double_round_robin_league()
    m = m[(m["status"] == "played") | (m["match_id"] == 7)]  # one open fixture
    fx = m[m["match_id"] == 7].iloc[0]
    params = StrengthParams(shrinkage=2.0)
    res = simulate_season(m, params, n_sims=200_000, seed=3,
                          shifts={fx["home"]: TeamShift(0.3)})
    lam, mu = res.strengths.expected_goals(fx["home"], fx["away"])
    p = predict_match(fx["home"], fx["away"], lam * 1.3, mu / 1.3)
    s = res.summary.set_index("team")
    gained = s.loc[fx["home"], "exp_points"] - s.loc[fx["home"], "points_now"]
    assert gained == pytest.approx(3 * p.p_home + p.p_draw, abs=0.01)
    with pytest.raises(ValueError):
        simulate_season(m, params, n_sims=10, shifts={"Nobody": TeamShift(0.1)})
