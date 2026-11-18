"""Tests for the maximum likelihood / Dixon–Coles model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from football_predictor.ingest.sample_data import generate_sample_league
from football_predictor.model.mle import PoissonProblem, fit_mle, fit_posterior
from football_predictor.model.poisson import dixon_coles_matrix, predict_match
from football_predictor.model.strength import StrengthParams, fit_strengths


def _played(seed: int, n_teams: int = 10, rounds: int = 14, sigma: float = 0.3) -> pd.DataFrame:
    league = generate_sample_league(n_teams=n_teams, played_rounds=rounds, seed=seed,
                                    strength_sigma=sigma)
    return pd.DataFrame([
        {"date": pd.Timestamp(r.match_date), "home": r.home_team, "away": r.away_team,
         "home_goals": float(r.home_goals), "away_goals": float(r.away_goals)}
        for r in league.records if r.home_goals is not None
    ]), league.truth


@pytest.mark.parametrize("model", ["mle", "dc"])
def test_analytic_gradient_matches_finite_differences(model: str) -> None:
    played, _ = _played(seed=3)
    problem = PoissonProblem(played, StrengthParams(model=model, xi=0.01, prior_sd=0.4))
    rng = np.random.default_rng(0)
    theta = problem.initial() + rng.normal(0, 0.1, problem.size)
    if model == "dc":
        theta[-1] = -0.05
    _, analytic = problem.objective(theta)
    numeric = approx_fprime(theta, lambda t: problem.objective(t)[0], 1e-6)
    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-3)


def test_identifiability_constraints() -> None:
    played, _ = _played(seed=4)
    s = fit_mle(played, StrengthParams(model="mle"))
    assert np.isclose(np.sum(np.log(list(s.attack.values()))), 0, atol=1e-8)
    assert np.isclose(np.sum(np.log(list(s.defense.values()))), 0, atol=1e-8)


def test_mle_recovers_truth_better_than_simple_on_average() -> None:
    """Averaged over seeds, joint estimation should be at least as accurate."""
    err_simple, err_mle = [], []
    for seed in range(8):
        played, truth = _played(seed=seed)
        teams = sorted(truth.attack)
        true = np.log([truth.attack[t] for t in teams])
        for fit, errs in ((fit_strengths(played, StrengthParams(shrinkage=2)), err_simple),
                          (fit_mle(played, StrengthParams(model="mle", prior_sd=0.3)), err_mle)):
            est = np.log([fit.attack[t] for t in teams])
            errs.append(np.sqrt(np.mean(((est - est.mean()) - (true - true.mean())) ** 2)))
    assert np.mean(err_mle) <= np.mean(err_simple) * 1.02


def test_home_advantage_recovered() -> None:
    played, truth = _played(seed=21, n_teams=14, rounds=26)
    s = fit_mle(played, StrengthParams(model="mle"))
    assert abs(s.home_advantage - truth.home_advantage) < 0.25


def test_small_prior_sd_shrinks_to_average() -> None:
    played, _ = _played(seed=5)
    s = fit_mle(played, StrengthParams(model="mle", prior_sd=0.01))
    assert all(abs(v - 1) < 0.02 for v in s.attack.values())


def test_dixon_coles_matrix_effects() -> None:
    lam, mu = 1.2, 0.9
    plain = predict_match("A", "B", lam, mu)
    dc = predict_match("A", "B", lam, mu, rho=-0.1)
    assert dc.matrix[0, 0] > plain.matrix[0, 0]  # rho < 0: more 0-0...
    assert dc.matrix[1, 0] < plain.matrix[1, 0]  # ...fewer 1-0
    assert np.isclose(dc.p_home + dc.p_draw + dc.p_away, 1.0)
    assert np.allclose(dixon_coles_matrix(lam, mu, 0.0, 8)[2:, 2:],
                       dixon_coles_matrix(lam, mu, -0.1, 8)[2:, 2:])  # other cells untouched


def test_dc_fit_runs_and_rho_in_bounds() -> None:
    played, _ = _played(seed=6)
    s = fit_mle(played, StrengthParams(model="dc"))
    assert -0.25 <= s.rho <= 0.25


def test_last_n_rejected_for_mle() -> None:
    played, _ = _played(seed=7)
    with pytest.raises(ValueError):
        fit_mle(played, StrengthParams(model="mle", last_n=5))


# --- Laplace posterior ------------------------------------------------------------

def test_hessian_matches_finite_differences_of_gradient() -> None:
    played, _ = _played(seed=3)
    problem = PoissonProblem(played, StrengthParams(model="mle", xi=0.01, prior_sd=0.4))
    theta = problem.solve()
    n = problem.n
    # objective(θ) depends on centred att/def: d²/dθ² = Cᵀ H C with C the centring map
    c = np.eye(problem.size)
    c[:n, :n] -= 1 / n
    c[n:2 * n, n:2 * n] -= 1 / n
    h = 1e-5
    numeric = np.array([(problem.objective(theta + h * e)[1] - problem.objective(theta - h * e)[1])
                        / (2 * h) for e in np.eye(problem.size)])
    assert np.allclose(numeric, c.T @ problem.hessian(theta) @ c, atol=1e-6)


def _league_from_prior(rng: np.random.Generator, n: int, sd: float) -> tuple:
    att, dfn = rng.normal(0, sd, n), rng.normal(0, sd, n)
    att, dfn = att - att.mean(), dfn - dfn.mean()
    rows = []
    for k, (h, a) in enumerate((h, a) for h in range(n) for a in range(n) if h != a):
        lam = np.exp(np.log(1.4) + 0.25 + att[h] + dfn[a])
        mu = np.exp(np.log(1.4) + att[a] + dfn[h])
        rows.append({"date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=k),
                     "home": f"T{h:02d}", "away": f"T{a:02d}",
                     "home_goals": float(rng.poisson(lam)), "away_goals": float(rng.poisson(mu))})
    return pd.DataFrame(rows), att, dfn


def test_laplace_intervals_have_nominal_coverage() -> None:
    """Truth drawn from the prior -> 90 % posterior intervals should cover ~90 %."""
    rng = np.random.default_rng(0)
    params = StrengthParams(model="mle", xi=0.0, prior_sd=0.3)
    covered = []
    for _ in range(80):
        played, att, dfn = _league_from_prior(rng, n=8, sd=0.3)
        post = fit_posterior(played, params)
        sd = post.sd_frame()
        covered += list(np.abs(post.mean[:8] - att) < 1.645 * sd["attack_sd"].to_numpy())
        covered += list(np.abs(post.mean[8:16] - dfn) < 1.645 * sd["defense_sd"].to_numpy())
    assert 0.86 < np.mean(covered) < 0.94


def test_posterior_samples_and_shape() -> None:
    played, _ = _played(seed=4)
    post = fit_posterior(played, StrengthParams(model="mle", prior_sd=0.5))
    n = len(post.teams)
    assert np.allclose(np.log(post.strengths.attack[post.teams[0]]), post.mean[0])
    draws = post.sample(np.random.default_rng(0), 50_000)
    assert draws.shape == (50_000, 2 * n + 2)
    assert np.allclose(draws[:, :n].sum(axis=1), 0, atol=1e-9)  # constraint holds per draw
    assert np.allclose(np.cov(draws.T), post.covariance, atol=3e-3)
    # more data -> narrower posterior
    fewer = fit_posterior(played[played["date"] < played["date"].median()],
                          StrengthParams(model="mle", prior_sd=0.5))
    assert (fewer.sd_frame()["attack_sd"].mean() > post.sd_frame()["attack_sd"].mean())


def test_posterior_needs_likelihood_model() -> None:
    played, _ = _played(seed=4)
    with pytest.raises(ValueError):
        fit_posterior(played, StrengthParams(model="simple"))
