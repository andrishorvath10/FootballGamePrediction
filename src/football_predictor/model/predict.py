"""High-level helpers: predict a round of a competition-season.

Used by the CLI and (later) the dashboard, so both show identical numbers.
"""

from __future__ import annotations

import pandas as pd

from football_predictor.db.models import MatchStatus
from football_predictor.model.dataset import played_only
from football_predictor.model.poisson import MatchPrediction, predict_match
from football_predictor.model.strength import StrengthParams, TeamStrengths, fit_model

OPEN_STATUSES = {MatchStatus.SCHEDULED.value, MatchStatus.POSTPONED.value}


def next_round(matches: pd.DataFrame) -> int | None:
    """The first round after the last round with a played match that has open fixtures.

    Earlier rounds may still contain postponed/catch-up matches; those are found
    by :func:`catch_up_rounds`.
    """
    is_open = matches["status"].isin(OPEN_STATUSES)
    played_rounds = matches.loc[matches["status"] == MatchStatus.PLAYED.value, "round"]
    last_played = int(played_rounds.max()) if not played_rounds.empty else 0
    later = matches.loc[is_open & (matches["round"] > last_played), "round"]
    if not later.empty:
        return int(later.min())
    open_rounds = matches.loc[is_open, "round"]
    return int(open_rounds.min()) if not open_rounds.empty else None


def catch_up_rounds(matches: pd.DataFrame, before: int) -> list[int]:
    """Rounds before ``before`` that still have open (postponed/catch-up) fixtures."""
    is_open = matches["status"].isin(OPEN_STATUSES)
    earlier = matches.loc[is_open & (matches["round"] < before), "round"]
    return sorted(int(r) for r in earlier.unique())


def predict_round(
    matches: pd.DataFrame, rnd: int | list[int], params: StrengthParams | None = None
) -> tuple[list[MatchPrediction], TeamStrengths]:
    """Fit on all matches played so far, then predict the open fixtures of ``rnd``.

    ``rnd`` may be a list of rounds (e.g. the next round plus catch-up matches).

    Only results that are already known are used, so there is no look-ahead.
    """
    played = played_only(matches)
    all_teams = set(matches["home"]) | set(matches["away"])
    strengths = fit_model(played, params, teams=all_teams)
    rounds = [rnd] if isinstance(rnd, int) else list(rnd)
    fixtures = matches[matches["round"].isin(rounds) & matches["status"].isin(OPEN_STATUSES)]
    predictions = []
    for row in fixtures.itertuples():
        lam, mu = strengths.expected_goals(row.home, row.away)
        pred = predict_match(row.home, row.away, lam, mu, rho=strengths.rho)
        pred.round = int(row.round)
        pred.date = row.date
        predictions.append(pred)
    return predictions, strengths


def predictions_to_frame(predictions: list[MatchPrediction]) -> pd.DataFrame:
    """Tabular summary of predictions (probabilities in 0..1)."""
    return pd.DataFrame([
        {
            "round": p.round,
            "date": p.date.date() if isinstance(p.date, pd.Timestamp) else None,
            "home": p.home, "away": p.away,
            "xg_home": p.lam, "xg_away": p.mu,
            "p_1": p.p_home, "p_x": p.p_draw, "p_2": p.p_away,
            "tip": p.most_likely_outcome,
            "top_score": f"{p.top_scores[0][0]}-{p.top_scores[0][1]}",
            "p_top_score": p.top_scores[0][2],
            "p_over_2_5": p.p_over_2_5, "p_btts": p.p_btts,
        }
        for p in predictions
    ])
