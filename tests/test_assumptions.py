"""Tests for storing and checking expert assumptions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar
from sqlalchemy import select
from sqlalchemy.orm import Session

from football_predictor.db import assumption_store as store
from football_predictor.db.models import Team
from football_predictor.ingest.loader import LeagueLoader
from football_predictor.ingest.sample_data import (
    SYNTHETIC_COMPETITION_NAME as COMP,
)
from football_predictor.ingest.sample_data import (
    SYNTHETIC_SEASON_NAME as SEASON,
)
from football_predictor.ingest.sample_data import (
    generate_sample_league,
)
from football_predictor.model.assumptions import (
    calibration,
    check_shift,
    no_change_checks,
    realized_shift,
    walk_variance,
)


def test_closed_form_matches_numerical_mle() -> None:
    a, c, gf, ga = 14.0, 11.0, 21.0, 7.0
    delta, se = realized_shift(a, c, gf, ga)

    def neg_ll(d: float) -> float:
        return -(gf * d - a * np.exp(d) - ga * d - c * np.exp(-d))

    assert delta == pytest.approx(minimize_scalar(neg_ll).x, abs=1e-6)
    h = 1e-4
    curvature = (neg_ll(delta + h) - 2 * neg_ll(delta) + neg_ll(delta - h)) / h**2
    assert se == pytest.approx(1 / np.sqrt(curvature), rel=1e-4)
    # exactly as expected -> no shift
    assert realized_shift(10, 10, 10, 10)[0] == pytest.approx(0)


def _league_with_jump(seed: int, team_jump: float) -> tuple[pd.DataFrame, str]:
    """10 teams, four round-robins; team T00 is (1 + team_jump)× better from day 200."""
    rng = np.random.default_rng(seed)
    n = 10
    att, dfn = rng.normal(0, 0.25, n), rng.normal(0, 0.25, n)
    rows, day = [], 0
    for _ in range(4):
        for h in range(n):
            for a in range(n):
                if h == a:
                    continue
                k = np.log1p(team_jump) if day >= 200 else 0.0
                lam = np.exp(0.35 + 0.2 + att[h] + dfn[a] + k * ((h == 0) - (a == 0)))
                mu = np.exp(0.35 + att[a] + dfn[h] + k * ((a == 0) - (h == 0)))
                rows.append({"match_id": len(rows), "round": len(rows) // 5 + 1,
                             "date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=day),
                             "home": f"T{h:02d}", "away": f"T{a:02d}",
                             "home_goals": float(rng.poisson(lam)),
                             "away_goals": float(rng.poisson(mu)), "status": "played"})
                day += 1
    return pd.DataFrame(rows), "T00"


def test_check_recovers_a_known_change() -> None:
    matches, team = _league_with_jump(seed=1, team_jump=0.5)
    start = matches["date"].min() + pd.Timedelta(days=200)
    right = check_shift(matches, team, 0.5, start)
    assert right.matches > 20
    assert right.ci_low < 0.5 < right.ci_high
    assert right.verdict == "consistent"
    assert right.log_gain > 0
    wrong = check_shift(matches, team, 0.0, start)
    assert wrong.verdict == "too pessimistic"  # the team improved more than assumed
    assert check_shift(matches, team, 1.5, start).verdict == "too optimistic"


def test_walk_variance_matches_monte_carlo() -> None:
    rng = np.random.default_rng(0)
    days = np.array([3.0, 10.0, 10.0, 45.0, 90.0])
    w = np.array([0.5, -0.2, 0.3, 0.1, 0.4])
    steps = np.diff(np.concatenate([[0.0], days]))
    walks = np.cumsum(rng.standard_normal((200_000, len(days))) * np.sqrt(steps / 30), axis=1)
    assert walk_variance(w, days) == pytest.approx(np.var(walks @ w), rel=0.02)


def _drifting_league(seed: int, sigma: float, n: int = 12) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Four round-robins (a round every 2 days); every team's log attack / defence
    follows a random walk with ``sigma`` per 30 d all season (the check starts halfway,
    so its baseline lags behind the drifting teams, as in real leagues)."""
    rng = np.random.default_rng(seed)
    att, dfn = rng.normal(0, 0.25, n), rng.normal(0, 0.25, n)
    total = 8 * (n - 1)
    rows = []
    for r in range(total):
        if r > 0:  # centred: strengths are relative, the league average stays put
            for x in (att, dfn):
                step = rng.normal(0, sigma * np.sqrt(2 / 30), n)
                x += step - step.mean()
        pairs = rng.permutation(n).reshape(-1, 2)
        for h, a in pairs:
            rows.append({"match_id": len(rows), "round": r + 1,
                         "date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=2 * r),
                         "home": f"T{h:02d}", "away": f"T{a:02d}",
                         "home_goals": float(rng.poisson(np.exp(0.55 + att[h] + dfn[a]))),
                         "away_goals": float(rng.poisson(np.exp(0.35 + att[a] + dfn[h]))),
                         "status": "played"})
    return pd.DataFrame(rows), pd.Timestamp("2025-08-01") + pd.Timedelta(days=total)


def _no_change_checks(true_sigma: float, stated_sigma: float, leagues: int = 6) -> list:
    checks = []
    for seed in range(leagues):
        matches, start = _drifting_league(seed, true_sigma)
        checks += no_change_checks(matches, start, sigma=stated_sigma)
    return checks


def test_without_drift_the_uncertainty_is_calibrated() -> None:
    # sampling + baseline error alone must explain the residuals (tests the delta method)
    checks = _no_change_checks(true_sigma=0.0, stated_sigma=0.0)
    cal = calibration(checks)
    assert 0.6 < cal.mean_z2 < 1.4
    assert cal.verdict == "calibrated"
    assert np.mean([c.verdict != "consistent" for c in checks]) < 0.12


def test_drift_factor_is_recovered() -> None:
    drifting = calibration(_no_change_checks(true_sigma=0.25, stated_sigma=0.25))
    assert drifting.verdict == "calibrated"
    assert 0.5 < drifting.pooled_factor < 1.5
    still = calibration(_no_change_checks(true_sigma=0.0, stated_sigma=0.25))
    assert still.verdict == "too cautious"
    assert still.pooled_factor < 0.3
    assert calibration(_no_change_checks(0.25, 0.05)).verdict == "overconfident"


def test_single_drift_verdict() -> None:
    matches, team = _league_with_jump(seed=1, team_jump=0.5)
    start = matches["date"].min() + pd.Timedelta(days=200)
    sure = check_shift(matches, team, 0.0, start, drift_factor=1.0, sigma=0.02)
    assert sure.drift_verdict == "too confident"  # a +50 % jump, no change expected
    unsure = check_shift(matches, team, 0.0, start, drift_factor=5.0, sigma=0.02)
    assert abs(unsure.z) < abs(sure.z)
    assert unsure.factor_gain > 0  # admitting uncertainty helped
    assert unsure.implied_factor > 1
    assert check_shift(matches, team, 0.5, start, sigma=0.02).drift_verdict == "consistent"
    assert calibration([]).verdict == "pending"


def test_check_without_later_matches_is_pending() -> None:
    matches, team = _league_with_jump(seed=2, team_jump=0.0)
    res = check_shift(matches, team, 0.2, matches["date"].max() + pd.Timedelta(days=1))
    assert (res.matches, res.verdict, res.realized) == (0, "pending", None)


def test_store_roundtrip(session: Session) -> None:
    LeagueLoader(session).load(generate_sample_league(seed=1).records)
    teams = sorted(session.scalars(select(Team.name)))
    store.save_assumption(session, COMP, SEASON, teams[0], 0.2,
                          pd.Timestamp("2025-10-01").date(), 2.0, "new striker")
    store.save_assumption(session, COMP, SEASON, teams[0], 0.3)  # replaces
    (a,) = store.list_assumptions(session, COMP, SEASON)
    assert (a.team, a.change, a.drift_factor, a.note) == (teams[0], 0.3, 1.0, None)
    store.replace_assumptions(session, COMP, SEASON, [
        store.StoredAssumption(teams[1], -0.1, pd.Timestamp("2025-11-01"))])
    assert [x.team for x in store.list_assumptions(session, COMP, SEASON)] == [teams[1]]
    assert store.delete_assumption(session, COMP, SEASON, teams[1])
    assert not store.delete_assumption(session, COMP, SEASON, teams[1])
    with pytest.raises(ValueError):
        store.save_assumption(session, COMP, SEASON, "Nobody FC", 0.1)
    assert store.list_assumptions(session, "Nope", SEASON) == []


def test_cli_assume_and_simulate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 capsys: pytest.CaptureFixture[str]) -> None:
    import football_predictor.db.session as db_session
    from football_predictor.cli import main
    from football_predictor.config import get_settings

    monkeypatch.setenv("FP_DATABASE_URL", f"sqlite:///{tmp_path / 'cli.db'}")
    get_settings.cache_clear()
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_session_factory", None)
    try:
        assert main(["generate-sample", "--truth-out", str(tmp_path / "t.json")]) == 0
        with db_session.session_scope() as s:
            team = sorted(s.scalars(select(Team.name)))[0]
        args = ["--competition", COMP, "--season", SEASON]
        assert main(["assume", "add", team, "+25%@2025-09-15", *args,
                     "--note", "test"]) == 0
        assert main(["assume", "add", "Nobody FC", "+5%", *args]) == 2
        assert main(["assume", "list", *args]) == 0
        assert main(["assume", "check", *args]) == 0
        out = capsys.readouterr().out
        assert "Assumption check" in out and team in out
        assert main(["simulate", *args, "--sims", "200"]) == 0
        assert main(["assume", "remove", team, *args]) == 0
        assert main(["assume", "list", *args]) == 0
        assert "No assumptions stored" in capsys.readouterr().out
    finally:
        db_session.get_engine().dispose()
        get_settings.cache_clear()


def test_no_change_checks_cover_teams_that_played_after_start() -> None:
    matches, start = _drifting_league(0, 0.0)
    checks = no_change_checks(matches, start, sigma=0.0)
    after = matches[matches["date"] >= start]
    assert {c.team for c in checks} == set(after["home"]) | set(after["away"])
    assert all(c.assumed == 0.0 and c.z is not None for c in checks)
    assert no_change_checks(matches, matches["date"].max() + pd.Timedelta(days=1)) == []
