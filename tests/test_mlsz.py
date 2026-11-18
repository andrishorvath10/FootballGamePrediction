"""Tests for the MLSZ adatbank adapter (on fictional HTML with the real structure)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from football_predictor.db.models import Appearance, Competition, Goal, Match, MatchStatus, Player
from football_predictor.db.validation import find_goal_mismatches
from football_predictor.ingest.loader import LeagueLoader
from football_predictor.ingest.mlsz import (
    MlszCompetition,
    MlszConfig,
    MlszScraper,
    parse_match_page,
    parse_minute,
    parse_round_page,
    parse_rounds,
    parse_standings,
    point_adjustments_from_standings,
    round_url,
)
from football_predictor.model.dataset import load_point_adjustments

FIXTURES = Path(__file__).parent / "fixtures"
ROUND_HTML = (FIXTURES / "mlsz_round.html").read_text(encoding="utf-8")
MATCH_HTML = (FIXTURES / "mlsz_match.html").read_text(encoding="utf-8")
HOME, AWAY = "KÖDVÁR SE", "SZÉLMALOM FC"


def test_parse_rounds() -> None:
    assert parse_rounds(ROUND_HTML) == [1, 2, 3, 4]


def test_parse_round_page() -> None:
    fixtures = parse_round_page(ROUND_HTML, 3)
    assert len(fixtures) == 3  # the #sorsolas_panel list is ignored

    played = fixtures[0]
    assert (played.home_team, played.away_team) == (HOME, AWAY)  # trailing space trimmed
    assert played.score == (3, 2) and played.status == MatchStatus.PLAYED
    assert played.match_date == date(2026, 11, 8)
    assert played.match_id == "5000001"

    assert fixtures[1].score == (0, 0)

    unplayed = fixtures[2]
    assert unplayed.score is None  # "03-14" is a date, NOT a 3-14 result
    assert unplayed.status == MatchStatus.SCHEDULED
    assert unplayed.match_date == date(2027, 3, 14)
    assert unplayed.match_url is None


def test_parse_standings_and_penalty_points() -> None:
    rows = parse_standings(ROUND_HTML)
    assert [r.team for r in rows] == ["KÖDVÁR SE", "HOLDFÉNY TC", "SZÉLMALOM FC", "RÓKAHEGY SC"]
    top = rows[0]
    assert (top.played, top.won, top.drawn, top.lost, top.goals_for, top.goals_against,
            top.points) == (3, 2, 1, 0, 7, 2, 7)
    assert top.point_adjustment == 0
    assert rows[2].point_adjustment == -3  # 6 points earned, 3 shown
    assert point_adjustments_from_standings(rows) == {"SZÉLMALOM FC": (-3, "Büntetőpont: -3")}
    assert parse_standings("<html></html>") == []


def test_parse_minute() -> None:
    assert parse_minute("4'") == 4
    assert parse_minute("90+2'") == 92
    assert parse_minute("") is None


def test_parse_match_goals() -> None:
    goals = parse_match_page(MATCH_HTML, HOME, AWAY).goals
    assert goals is not None and len(goals) == 5
    assert [g.team for g in goals] == [HOME, AWAY, HOME, AWAY, HOME]
    assert [g.minute for g in goals] == [4, 20, 50, 70, 92]
    assert goals[0].player == "ÁGAS BENCE" and goals[0].player_external_id == "mlsz:100001"
    assert goals[1].is_penalty and not goals[1].is_own_goal
    # own goal: credited to the home side, scored by an away player
    assert goals[2].is_own_goal and goals[2].team == HOME
    assert goals[2].player_external_id == "mlsz:200002"
    assert goals[4].player is None  # unknown scorer


def test_parse_match_minutes_played() -> None:
    apps = parse_match_page(MATCH_HTML, HOME, AWAY).appearances
    assert apps is not None
    minutes = {a.player: a.minutes_played for a in apps}
    assert minutes == {
        "ÁGAS BENCE": 90,
        "ERDŐS MÁRK": 61,     # substituted off at 61'
        "FÜRGE NOEL": 29,     # came on at 61'
        "BERKES HUBA": 90,
        "CSERMELY ÁDÁM": 90,
        "DOMBI IMRE": 80,     # red card at 80'
        "GYÖKERES OTTÓ": 46,  # name has a double space in the timeline title
        "HAMVAS PÉTER": 44,
    }  # unused substitute and the coach are excluded
    assert {a.team for a in apps if a.player == "HAMVAS PÉTER"} == {AWAY}


class FakeFetcher:
    """Serves fixture HTML by URL and records requests."""

    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.calls: list[str] = []

    def fetch(self, url: str, refresh: bool = False, offline: bool = False) -> str:
        self.calls.append(url)
        return self.pages.get(url, "<html></html>")

    def cache_age(self, url: str) -> float | None:
        return 0.0


def _config() -> MlszConfig:
    return MlszConfig(
        federation_id=8,
        season_ids={"2026/27": 67},
        competitions=(MlszCompetition("Test Megye II", {"2026/27": 1}, "Test", 2),),
    )


def test_scraper_end_to_end_into_db(session: Session) -> None:
    pages = {
        round_url(67, 8, 1, 3): ROUND_HTML,
        "https://adatbank.mlsz.hu/match/67/8/1/3/5000001.html": MATCH_HTML,
    }
    # Rounds 1, 2, 4: same page but without fixtures (round 1 still lists all rounds).
    empty_round = ROUND_HTML.replace('class="schedule "', 'class="x"')
    for rnd in (1, 2, 4):
        pages[round_url(67, 8, 1, rnd)] = empty_round
    fetcher = FakeFetcher(pages)
    scraper = MlszScraper(_config(), fetcher)  # type: ignore[arg-type]
    (comp, season), = scraper.iter_targets()
    records = scraper.scrape(comp, season)

    assert len(records) == 3
    assert records[0].external_id == "mlsz:5000001"
    assert records[1].goals == []  # 0-0 match page has no goals -> empty list is fine

    LeagueLoader(session).load(records)
    session.commit()
    assert session.scalar(select(func.count()).select_from(Match)) == 3
    assert session.scalar(select(func.count()).select_from(Goal)) == 5
    assert session.scalar(select(func.count()).select_from(Appearance)) == 8
    assert session.scalar(select(Competition.name)) == "Test Megye II"
    # players are identified by MLSZ id
    dombi = select(Player.name).where(Player.external_id == "mlsz:200003")
    assert session.scalar(dombi) == "DOMBI IMRE"
    assert find_goal_mismatches(session) == []

    again = LeagueLoader(session).load(records)
    assert again.matches_unchanged == 3

    # the deduction from the standings note is collected and synced idempotently
    (adjustments,) = scraper.adjustments.values()
    assert [(a.team, a.points) for a in adjustments] == [("SZÉLMALOM FC", -3)]
    loader = LeagueLoader(session)
    assert loader.sync_point_adjustments(season, comp.name, adjustments) == 1
    assert loader.sync_point_adjustments(season, comp.name, adjustments) == 0
    assert load_point_adjustments(session, comp.name, season) == {"SZÉLMALOM FC": -3}
    assert loader.sync_point_adjustments(season, comp.name, []) == 1  # revoked
    assert load_point_adjustments(session, comp.name, season) == {}


def test_no_details_mode_skips_match_pages() -> None:
    fetcher = FakeFetcher({round_url(67, 8, 1, 1): ROUND_HTML})
    scraper = MlszScraper(_config(), fetcher, fetch_details=False)  # type: ignore[arg-type]
    (comp, season), = scraper.iter_targets()
    scraper.scrape(comp, season)
    assert not any("/match/" in url for url in fetcher.calls)


def test_round_list_is_refreshed_when_stale() -> None:
    class StaleFetcher(FakeFetcher):
        def cache_age(self, url: str) -> float | None:
            return 10 * 24 * 3600.0  # ten days old

        def fetch(self, url: str, refresh: bool = False, offline: bool = False) -> str:
            self.calls.append(f"{url} refresh={refresh}")
            return self.pages.get(url, "<html></html>")

    fetcher = StaleFetcher({round_url(67, 8, 1, 1): ROUND_HTML})
    scraper = MlszScraper(_config(), fetcher, fetch_details=False)  # type: ignore[arg-type]
    (comp, season), = scraper.iter_targets()
    scraper.scrape(comp, season)
    assert f"{round_url(67, 8, 1, 1)} refresh=True" in fetcher.calls
