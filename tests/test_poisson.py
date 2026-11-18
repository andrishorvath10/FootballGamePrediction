"""Tests for the Poisson outcome model, checked against closed-form results."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import poisson

from football_predictor.model.poisson import predict_match, score_matrix


@pytest.mark.parametrize(("lam", "mu"), [(1.5, 1.1), (0.3, 0.2), (2.5, 3.0)])
def test_score_matrix_sums_to_one(lam: float, mu: float) -> None:
    assert np.isclose(score_matrix(lam, mu, 30).sum(), 1.0, atol=1e-12)


def test_display_matrix_truncation_is_visible_for_high_scoring_games() -> None:
    # With λ = 4.5, more than 1 % of the mass lies beyond 8 goals: that is why
    # probabilities are computed on a larger grid than the displayed 0..8 matrix.
    assert score_matrix(4.5, 1.0, 8).sum() < 0.99
    pred = predict_match("A", "B", 4.5, 1.0)
    assert np.isclose(pred.p_home + pred.p_draw + pred.p_away, 1.0, atol=1e-12)


@pytest.mark.parametrize(("lam", "mu"), [(1.5, 1.1), (0.8, 2.2), (4.5, 1.0)])
def test_1x2_sums_to_one(lam: float, mu: float) -> None:
    p = predict_match("A", "B", lam, mu)
    assert np.isclose(p.p_home + p.p_draw + p.p_away, 1.0)


def test_symmetry_equal_teams() -> None:
    p = predict_match("A", "B", 1.3, 1.3)
    assert np.isclose(p.p_home, p.p_away)


def test_over_under_matches_sum_of_poissons() -> None:
    # X + Y ~ Poisson(λ + μ) for independent Poissons, so
    # P(total > 2.5) = 1 - P(total <= 2) has a closed form.
    lam, mu = 1.7, 1.2
    p = predict_match("A", "B", lam, mu)
    assert np.isclose(p.p_over_2_5, 1 - poisson.cdf(2, lam + mu))


def test_btts_closed_form() -> None:
    lam, mu = 1.7, 1.2
    p = predict_match("A", "B", lam, mu)
    assert np.isclose(p.p_btts, (1 - np.exp(-lam)) * (1 - np.exp(-mu)))


def test_draw_closed_form_for_0_0() -> None:
    lam, mu = 1.4, 0.9
    p = predict_match("A", "B", lam, mu)
    assert np.isclose(p.matrix[0, 0], np.exp(-lam) * np.exp(-mu))


def test_top_scores_sorted_and_most_likely() -> None:
    p = predict_match("A", "B", 1.6, 0.9)
    probs = [s[2] for s in p.top_scores]
    assert len(probs) == 5 and probs == sorted(probs, reverse=True)
    assert p.top_scores[0][:2] == (1, 0)  # mode of Poisson(1.6) is 1, of Poisson(0.9) is 0
    assert p.most_likely_outcome == "1"
