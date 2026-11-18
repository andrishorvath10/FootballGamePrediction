"""Maximum likelihood (MAP) Poisson model with optional Dixon–Coles correction.

Model (all strengths on the log scale, which keeps them positive):

    log λ = base + home + att[h] + def[a]
    log μ = base        + att[a] + def[h]

with Σ att = Σ def = 0 (identifiability: otherwise adding c to every attack and
subtracting c from ``base`` would give identical predictions).

Unlike the simple ratio model, all teams are estimated *jointly*: a team that
scored a lot against weak defences gets less credit, because those defences'
weakness is estimated from all their other matches too.

Objective (minimised with L-BFGS-B and analytic gradients)::

    -Σ_m w_m · [ log Pois(x_m; λ_m) + log Pois(y_m; μ_m) + log τ(x_m, y_m) ]
      + (Σ att² + Σ def²) / (2 · prior_sd²)

* ``w_m = exp(-ξ · days_ago)``: time-weighted likelihood ("form").
* The last term is a Normal(0, prior_sd²) prior on log-strengths, i.e. this is a
  MAP estimate. Small ``prior_sd`` = strong pull towards the league average
  (the MLE counterpart of the simple model's shrinkage).
* ``τ`` is the Dixon–Coles (1997) correction. Independent Poissons slightly
  mis-price low scores (0-0, 1-1 happen more often in practice); τ adjusts just
  those four cells with one extra parameter ρ:

      τ(0,0) = 1 - λμρ    τ(0,1) = 1 + λρ    τ(1,0) = 1 + μρ    τ(1,1) = 1 - ρ

  ρ < 0 makes 0-0 and 1-1 more likely, 1-0 and 0-1 less likely.

Parameter uncertainty (Laplace approximation)
---------------------------------------------
Near its maximum the log posterior is approximately quadratic, so the posterior
is approximately Normal with mean = the MAP estimate and covariance = the inverse
of the Hessian of the negative log posterior::

    H = Σ_m w_m λ_m x_m x_mᵀ + Σ_m w_m μ_m z_m z_mᵀ + diag(1/prior_sd² on att, def)

where x_m / z_m are the 0/1 design rows of log λ_m / log μ_m (for a Poisson GLM
the second derivative of -log L w.r.t. the linear predictor is just the mean).
Few matches -> little curvature -> wide posterior. The sum-to-zero constraint is
handled by working in an orthonormal basis of the constrained subspace. With
time decay the weighted likelihood counts old matches less, so they also add less
certainty. ρ is held at its estimate (its curvature term is negligible here).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.optimize import minimize
from scipy.special import gammaln

from football_predictor.model.strength import StrengthParams, TeamStrengths, time_weights

logger = logging.getLogger(__name__)

RHO_BOUNDS = (-0.25, 0.25)
TAU_FLOOR = 1e-10


def _dixon_coles_terms(
    x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """log τ per match and its derivatives w.r.t. log λ, log μ and ρ."""
    tau = np.ones_like(lam)
    d_eta_h = np.zeros_like(lam)  # d log τ / d log λ
    d_eta_a = np.zeros_like(lam)  # d log τ / d log μ
    d_rho = np.zeros_like(lam)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + mu[m10] * rho
    tau[m11] = 1 - rho
    tau = np.maximum(tau, TAU_FLOOR)

    # chain rule: d/d(log λ) = λ · d/dλ
    d_eta_h[m00] = -lam[m00] * mu[m00] * rho / tau[m00]
    d_eta_a[m00] = -lam[m00] * mu[m00] * rho / tau[m00]
    d_rho[m00] = -lam[m00] * mu[m00] / tau[m00]
    d_eta_h[m01] = lam[m01] * rho / tau[m01]
    d_rho[m01] = lam[m01] / tau[m01]
    d_eta_a[m10] = mu[m10] * rho / tau[m10]
    d_rho[m10] = mu[m10] / tau[m10]
    d_rho[m11] = -1 / tau[m11]
    return np.log(tau), d_eta_h, d_eta_a, d_rho


class PoissonProblem:
    """Data + objective of the penalised, weighted Poisson likelihood.

    Kept as a class (not a closure) so tests can check the analytic gradient
    against finite differences.
    """

    def __init__(
        self,
        played: pd.DataFrame,
        params: StrengthParams,
        as_of: pd.Timestamp | None = None,
        teams: Iterable[str] | None = None,
    ) -> None:
        self.use_dc = params.model == "dc"
        as_of = as_of if as_of is not None else played["date"].max()
        self.teams = sorted(set(played["home"]) | set(played["away"]) | set(teams or []))
        self.index = {t: i for i, t in enumerate(self.teams)}
        self.n = len(self.teams)
        self.hi = played["home"].map(self.index).to_numpy()
        self.ai = played["away"].map(self.index).to_numpy()
        self.x = played["home_goals"].to_numpy(dtype=float)
        self.y = played["away_goals"].to_numpy(dtype=float)
        self.w = time_weights(played["date"], as_of, params.xi)
        # log x! terms do not depend on the parameters, but keep the loss a true -log L
        self.const = np.sum(self.w * (gammaln(self.x + 1) + gammaln(self.y + 1)))
        self.penalty = 1.0 / (2.0 * params.prior_sd**2)

    @property
    def size(self) -> int:
        """Number of free parameters."""
        return 2 * self.n + 2 + (1 if self.use_dc else 0)

    def initial(self) -> np.ndarray:
        """Start: all teams average, base = log(mean goals per team)."""
        theta = np.zeros(self.size)
        mean_goals = np.sum(self.w * (self.x + self.y)) / (2 * np.sum(self.w))
        theta[2 * self.n] = np.log(max(float(mean_goals), 1e-3))
        return theta

    def bounds(self) -> list[tuple[float | None, float | None]]:
        """Only ρ is bounded."""
        return [(None, None)] * (2 * self.n + 2) + ([RHO_BOUNDS] if self.use_dc else [])

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """(att, def, base, home, rho); att/def centred so that they sum to 0."""
        n = self.n
        att = theta[:n] - theta[:n].mean()
        dfn = theta[n:2 * n] - theta[n:2 * n].mean()
        rho = float(theta[2 * n + 2]) if self.use_dc else 0.0
        return att, dfn, float(theta[2 * n]), float(theta[2 * n + 1]), rho

    def objective(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Negative log posterior and its gradient."""
        n, hi, ai, x, y, w = self.n, self.hi, self.ai, self.x, self.y, self.w
        att, dfn, base, home, rho = self.unpack(theta)
        eta_h = base + home + att[hi] + dfn[ai]
        eta_a = base + att[ai] + dfn[hi]
        lam, mu = np.exp(eta_h), np.exp(eta_a)

        ll = np.sum(w * (x * eta_h - lam + y * eta_a - mu)) - self.const
        g_h = w * (x - lam)  # d LL / d eta_h: the classic Poisson score "observed - expected"
        g_a = w * (y - mu)
        g_rho = 0.0
        if self.use_dc:
            log_tau, dh, da, dr = _dixon_coles_terms(x, y, lam, mu, rho)
            ll += np.sum(w * log_tau)
            g_h = g_h + w * dh
            g_a = g_a + w * da
            g_rho = float(np.sum(w * dr))

        # Accumulate per team: att[h] appears in eta_h, att[a] in eta_a, etc.
        g_att = np.bincount(hi, g_h, n) + np.bincount(ai, g_a, n)
        g_def = np.bincount(ai, g_h, n) + np.bincount(hi, g_a, n)
        loss = -ll + self.penalty * (att @ att + dfn @ dfn)
        g_att = -g_att + 2 * self.penalty * att
        g_def = -g_def + 2 * self.penalty * dfn
        # Centring is a linear projection, so the raw gradient is the centred gradient.
        grad = np.concatenate([
            g_att - g_att.mean(),
            g_def - g_def.mean(),
            [-(g_h.sum() + g_a.sum()), -g_h.sum()],
            [-g_rho] if self.use_dc else [],
        ])
        return float(loss), grad

    def hessian(self, theta: np.ndarray) -> np.ndarray:
        """Hessian of the negative log posterior w.r.t. (att, def, log base, log home).

        Plain-Poisson part plus the prior; ρ is held fixed (see module docstring).
        """
        n, hi, ai, w = self.n, self.hi, self.ai, self.w
        att, dfn, base, home, _ = self.unpack(theta)
        lam = np.exp(base + home + att[hi] + dfn[ai])
        mu = np.exp(base + att[ai] + dfn[hi])
        rows = np.arange(len(hi))
        xh = np.zeros((len(hi), 2 * n + 2))
        xh[rows, hi] = 1
        xh[rows, n + ai] = 1
        xh[:, 2 * n:] = 1  # base and home
        xa = np.zeros_like(xh)
        xa[rows, ai] = 1
        xa[rows, n + hi] = 1
        xa[:, 2 * n] = 1
        h = (xh.T * (w * lam)) @ xh + (xa.T * (w * mu)) @ xa
        h[np.arange(2 * n), np.arange(2 * n)] += 2 * self.penalty
        return h

    def solve(self) -> np.ndarray:
        """MAP estimate (raw θ)."""
        result = minimize(self.objective, self.initial(), jac=True, method="L-BFGS-B",
                          bounds=self.bounds())
        if not result.success:
            # L-BFGS-B sometimes reports "ABNORMAL_TERMINATION" after reaching the optimum
            # within float precision; the parameters are still usable, so only log it.
            logger.debug("Optimiser: %s (loss %.4f)", result.message, result.fun)
        return result.x

    def strengths(self, theta: np.ndarray, played: pd.DataFrame,
                  params: StrengthParams) -> TeamStrengths:
        """Package θ as :class:`TeamStrengths` (multiplicative scale)."""
        att, dfn, base, home, rho = self.unpack(theta)
        counts = pd.concat([played["home"], played["away"]]).value_counts().to_dict()
        return TeamStrengths(
            base=float(np.exp(base)),
            home_advantage=float(np.exp(home)),
            attack={t: float(np.exp(att[i])) for t, i in self.index.items()},
            defense={t: float(np.exp(dfn[i])) for t, i in self.index.items()},
            matches={t: int(counts.get(t, 0)) for t in self.teams},
            params=params,
            rho=rho,
        )


def _check_params(played: pd.DataFrame, params: StrengthParams) -> None:
    if played.empty:
        raise ValueError("No played matches to fit on")
    if params.last_n is not None:
        raise ValueError("last_n is only supported by the simple model; use xi instead")


def fit_mle(
    played: pd.DataFrame,
    params: StrengthParams | None = None,
    as_of: pd.Timestamp | None = None,
    teams: Iterable[str] | None = None,
) -> TeamStrengths:
    """Fit the (weighted, penalised) Poisson model; Dixon–Coles if params.model == "dc"."""
    params = params or StrengthParams(model="mle")
    _check_params(played, params)
    problem = PoissonProblem(played, params, as_of, teams)
    return problem.strengths(problem.solve(), played, params)


@dataclass
class StrengthPosterior:
    """Laplace approximation of the posterior of the team strengths.

    ``mean`` is the MAP estimate as one log-scale vector
    ``(att_1..att_n, def_1..def_n, log base, log home)``; draws are
    ``mean + basis @ chol @ ε`` with ε ~ N(0, I), i.e. Normal in the constrained
    (sum-to-zero) subspace with covariance ``chol @ cholᵀ``.
    """

    strengths: TeamStrengths  # the MAP fit itself
    teams: list[str]
    mean: np.ndarray
    basis: np.ndarray  # (2n+2, 2n) orthonormal columns spanning the constrained space
    chol: np.ndarray  # Cholesky factor of the covariance in basis coordinates

    @property
    def covariance(self) -> np.ndarray:
        """Covariance of ``mean`` (singular: sum-to-zero directions have variance 0)."""
        cov = self.chol @ self.chol.T
        return self.basis @ cov @ self.basis.T

    def sd_frame(self) -> pd.DataFrame:
        """Posterior standard deviation of every team's log attack / log defence."""
        n, sd = len(self.teams), np.sqrt(np.diag(self.covariance))
        return pd.DataFrame({"attack_sd": sd[:n], "defense_sd": sd[n:2 * n]}, index=self.teams)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """(size, 2n+2) draws of the log-scale parameter vector."""
        eps = rng.standard_normal((size, self.chol.shape[0]))
        return self.mean + (eps @ self.chol.T) @ self.basis.T


def constrained_basis(n: int) -> np.ndarray:
    """Orthonormal basis for {Σ att = 0, Σ def = 0} × R² (base, home)."""
    q = null_space(np.ones((1, n)))  # n × (n-1)
    basis = np.zeros((2 * n + 2, 2 * n))
    basis[:n, :n - 1] = q
    basis[n:2 * n, n - 1:2 * n - 2] = q
    basis[2 * n, 2 * n - 2] = basis[2 * n + 1, 2 * n - 1] = 1.0
    return basis


def fit_posterior(
    played: pd.DataFrame,
    params: StrengthParams | None = None,
    as_of: pd.Timestamp | None = None,
    teams: Iterable[str] | None = None,
) -> StrengthPosterior:
    """MAP fit plus its Laplace (Normal) approximation of the posterior."""
    params = params or StrengthParams(model="mle")
    if params.model == "simple":
        raise ValueError("Parameter uncertainty needs a likelihood model (mle or dc)")
    _check_params(played, params)
    problem = PoissonProblem(played, params, as_of, teams)
    theta = problem.solve()
    att, dfn, base, home, _ = problem.unpack(theta)
    mean = np.concatenate([att, dfn, [base, home]])
    basis = constrained_basis(problem.n)
    h_red = basis.T @ problem.hessian(theta) @ basis
    # Σ = H⁻¹; Cholesky of Σ via that of H: H = LLᵀ -> Σ = L⁻ᵀL⁻¹, factor L⁻ᵀ
    # (not lower-triangular, but any C with CCᵀ = Σ samples correctly).
    chol_h = np.linalg.cholesky(h_red)
    chol = np.linalg.solve(chol_h.T, np.eye(len(h_red)))
    return StrengthPosterior(
        strengths=problem.strengths(theta, played, params),
        teams=problem.teams, mean=mean, basis=basis, chol=chol,
    )
