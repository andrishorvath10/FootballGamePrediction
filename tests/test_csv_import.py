"""Tests for CSV parsing and import."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from football_predictor.db.models import Goal, Match, MatchStatus
from football_predictor.db.validation import find_goal_mismatches
from football_predictor.ingest.csv_import import CsvImportError, read_league_csv
from football_predictor.ingest.loader import LeagueLoader

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_example_files_parse() -> None:
    records = read_league_csv(EXAMPLES / "matches.csv", EXAMPLES / "goals.csv")
    assert len(records) == 6
    by_round = {(r.round, r.home_team): r for r in records}

    first = by_round[(1, "Ködvár SE")]
    assert first.effective_status() == MatchStatus.PLAYED
    assert first.goals is not None and len(first.goals) == 3
    assert first.goals[1].is_penalty and first.goals[2].is_own_goal

    assert by_round[(1, "Holdfény TC")].goals == []  # 0-0 -> complete, empty
    assert by_round[(2, "Szélmalom FC")].goals is None  # 3-3 but no scorers -> unknown
    assert by_round[(2, "Csillagrét SK")].status == MatchStatus.POSTPONED
    assert by_round[(3, "Ködvár SE")].effective_status() is None  # scheduled


def test_example_import_into_db(session: Session) -> None:
    records = read_league_csv(EXAMPLES / "matches.csv", EXAMPLES / "goals.csv")
    LeagueLoader(session).load(records)
    session.commit()
    assert session.scalar(select(func.count()).select_from(Match)) == 6
    assert session.scalar(select(func.count()).select_from(Goal)) == 3
    assert find_goal_mismatches(session) == []


def test_semicolon_and_hungarian_dates(tmp_path: Path) -> None:
    f = tmp_path / "m.csv"
    f.write_text(
        "round;date;home_team;away_team;home_goals;away_goals\n"
        "1;2025.08.10.;A;B;1;0\n",
        encoding="utf-8",
    )
    (rec,) = read_league_csv(f, default_season="S")
    assert rec.match_date.isoformat() == "2025-08-10"
    assert rec.season == "S"


def test_errors_collected_with_line_numbers(tmp_path: Path) -> None:
    f = tmp_path / "m.csv"
    f.write_text(
        "season,round,home_team,away_team,home_goals,away_goals\n"
        "S,x,A,B,1,0\n"  # bad round
        "S,2,A,,1,0\n"  # missing away team
        "S,3,A,B,1,\n",  # half score
        encoding="utf-8",
    )
    with pytest.raises(CsvImportError) as exc:
        read_league_csv(f)
    assert len(exc.value.errors) == 3
    assert "m.csv:2" in exc.value.errors[0]


def test_goals_for_unknown_match_rejected(tmp_path: Path) -> None:
    m = tmp_path / "m.csv"
    m.write_text("season,round,home_team,away_team\nS,1,A,B\n", encoding="utf-8")
    g = tmp_path / "g.csv"
    g.write_text("season,round,home_team,away_team,team\nS,9,A,B,A\n", encoding="utf-8")
    with pytest.raises(CsvImportError):
        read_league_csv(m, g)
