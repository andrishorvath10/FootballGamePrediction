"""Tests for loading matches from the database into DataFrames."""

from __future__ import annotations

from sqlalchemy.orm import Session

from football_predictor.ingest.loader import LeagueLoader
from football_predictor.ingest.sample_data import (
    SYNTHETIC_COMPETITION_NAME,
    SYNTHETIC_SEASON_NAME,
    generate_sample_league,
)
from football_predictor.model.dataset import list_competition_seasons, load_matches, played_only


def test_load_matches_and_overview(session: Session) -> None:
    league = generate_sample_league(seed=1, played_rounds=14, postponed_rate=0.0)
    LeagueLoader(session).load(league.records)
    session.commit()

    df = load_matches(session, SYNTHETIC_COMPETITION_NAME, SYNTHETIC_SEASON_NAME)
    assert len(df) == 90
    assert len(played_only(df)) == 70
    assert df["home_goals"].isna().sum() == 20  # 4 unplayed rounds x 5 matches
    assert df["date"].dtype.kind == "M"

    overview = list_competition_seasons(session)
    row = overview.iloc[0]
    assert (row["matches"], row["played"], row["rounds"]) == (90, 70, 18)
