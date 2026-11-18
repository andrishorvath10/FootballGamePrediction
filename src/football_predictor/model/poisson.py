"""Match outcome probabilities from expected goals (independent Poisson model).

If home goals X ~ Poisson(λ) and away goals Y ~ Poisson(μ) are independent, then

    P(X = i, Y = j) = P(X = i) · P(Y = j)

so the whole score distribution is the outer product of two Poisson pmf vectors -
the *score matrix*. Every market is a sum over cells of that matrix:

* home win  = Σ_{i>j},  draw = Σ_{i=j} (the diagonal),  away win = Σ_{i<j}
* over 2.5  = Σ_{i+j≥3}
* both teams to score = Σ_{i≥1, j≥1}

Truncation: amateur leagues can have λ ≈ 4-5, where P(X > 8) is several percent.
All probabilities are therefore computed on a grid large enough to hold
essentially all the mass; only the *displayed* matrix is cut at 8 goals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

DISPLAY_MAX_GOALS = 8


def score_matrix(lam: float, mu: float, max_goals: int = DISPLAY_MAX_GOALS) -> np.ndarray:
    """``M[i, j] = P(home scores i, away scores j)`` for 0..max_goals goals."""
    goals = np.arange(max_goals + 1)
    return np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))


def dixon_coles_matrix(lam: float, mu: float, rho: float, max_goals: int) -> np.ndarray:
    """Independent-Poisson matrix with the Dixon–Coles factors on 0-0, 0-1, 1-0, 1-1."""
    m = score_matrix(lam, mu, max_goals)
    m[0, 0] *= max(1 - lam * mu * rho, 0.0)
    m[0, 1] *= max(1 + lam * rho, 0.0)
    m[1, 0] *= max(1 + mu * rho, 0.0)
    m[1, 1] *= max(1 - rho, 0.0)
    return m


def _grid_size(lam: float, mu: float) -> int:
    """Goals needed so the truncated tail is negligible (< ~1e-10)."""
    top = max(lam, mu)
    return int(max(10, np.ceil(top + 10 * np.sqrt(top) + 5)))


@dataclass
class MatchPrediction:
    """All derived probabilities for one fixture."""

    home: str
    away: str
    lam: float  # expected home goals
    mu: float  # expected away goals
    p_home: float
    p_draw: float
    p_away: float
    top_scores: list[tuple[int, int, float]]  # (home goals, away goals, probability)
    p_over_2_5: float
    p_btts: float
    matrix: np.ndarray  # 0..8 x 0..8 for display
    round: int | None = None  # filled in by callers that know the fixture
    date: object | None = None

    @property
    def most_likely_outcome(self) -> str:
        """"1", "X" or "2"."""
        probs = {"1": self.p_home, "X": self.p_draw, "2": self.p_away}
        return max(probs, key=probs.get)


def predict_from_matrix(
    home: str, away: str, lam: float, mu: float, full: np.ndarray, top_n: int = 5
) -> MatchPrediction:
    """Derive all markets from a (large, normalised) score matrix.

    Kept separate from :func:`predict_match` so that a corrected matrix
    (e.g. Dixon–Coles) can reuse exactly the same market calculations.
    """
    full = full / full.sum()  # remove the (tiny) truncation error exactly
    i, j = np.indices(full.shape)

    flat = np.argsort(full, axis=None)[::-1][:top_n]
    top = [(int(a), int(b), float(full[a, b])) for a, b in zip(*np.unravel_index(flat, full.shape),
                                                                 strict=True)]
    return MatchPrediction(
        home=home, away=away, lam=lam, mu=mu,
        p_home=float(full[i > j].sum()),
        p_draw=float(full[i == j].sum()),
        p_away=float(full[i < j].sum()),
        top_scores=top,
        p_over_2_5=float(full[i + j > 2.5].sum()),
        p_btts=float(full[1:, 1:].sum()),
        matrix=full[: DISPLAY_MAX_GOALS + 1, : DISPLAY_MAX_GOALS + 1].copy(),
    )


def predict_match(
    home: str, away: str, lam: float, mu: float, top_n: int = 5, rho: float = 0.0
) -> MatchPrediction:
    """Full prediction for one fixture from its expected goals (and optional DC ρ)."""
    size = _grid_size(lam, mu)
    full = dixon_coles_matrix(lam, mu, rho, size) if rho else score_matrix(lam, mu, size)
    return predict_from_matrix(home, away, lam, mu, full, top_n)
