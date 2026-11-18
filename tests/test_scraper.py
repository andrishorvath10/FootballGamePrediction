"""Tests for the scraper: parsing a saved HTML sample and the polite fetcher."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import requests

from football_predictor.db.models import MatchStatus
from football_predictor.ingest.scraper import (
    PoliteFetcher,
    RobotsDisallowedError,
    ScraperConfig,
    SelectorConfig,
    parse_fixtures,
    parse_score,
)

FIXTURE_HTML = Path(__file__).parent / "fixtures" / "fixtures_page.html"

CONFIG = ScraperConfig(
    season="2025/26",
    selectors=SelectorConfig(
        match_row="table.fixtures tr.match", round="td.round", date="td.date",
        home_team="td.home", away_team="td.away", score="td.score",
    ),
    pages=(),
    date_format="%Y.%m.%d.",
)


# ------------------------------------------------------------------- parsing
def test_parse_saved_page() -> None:
    records = parse_fixtures(FIXTURE_HTML.read_text(encoding="utf-8"), CONFIG, source_url="x")
    assert len(records) == 4  # header row ignored, row without away team skipped

    r1 = records[0]
    assert (r1.round, r1.home_team, r1.away_team) == (1, "Ködvár SE", "Szélmalom FC")
    assert (r1.home_goals, r1.away_goals, r1.status) == (2, 1, MatchStatus.PLAYED)
    assert r1.match_date == date(2025, 8, 10)
    assert r1.source_url == "x"

    assert (records[1].home_goals, records[1].away_goals) == (0, 0)
    assert records[2].status == MatchStatus.POSTPONED and records[2].home_goals is None
    assert records[3].status == MatchStatus.SCHEDULED


@pytest.mark.parametrize(
    ("text", "expected"),
    [("2-1", (2, 1)), ("2 : 1", (2, 1)), ("10–0", (10, 0)), ("-", None), ("", None)],
)
def test_parse_score(text: str, expected: tuple[int, int] | None) -> None:
    assert parse_score(text) == expected


def test_config_from_toml() -> None:
    cfg = ScraperConfig.from_toml(Path(__file__).parent.parent / "examples/scraper.example.toml")
    assert cfg.selectors.match_row == "table.fixtures tr.match"
    assert len(cfg.pages) == 2


# ------------------------------------------------------------------- fetching
class FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeHttp:
    """Stands in for requests.Session; records the URLs requested."""

    def __init__(self, responses: dict[str, list[FakeResponse]]) -> None:
        self.responses = responses
        self.calls: list[str] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, timeout: float) -> FakeResponse:
        self.calls.append(url)
        return self.responses[url].pop(0)


def _fetcher(tmp_path: Path, http: FakeHttp, sleeps: list[float]) -> PoliteFetcher:
    return PoliteFetcher(tmp_path, http=http, delay_seconds=2.0,  # type: ignore[arg-type]
                         sleep=sleeps.append, clock=lambda: 0.0)


def test_fetch_saves_to_cache_and_reuses_it(tmp_path: Path) -> None:
    url = "https://league.test/fixtures?round=1"
    http = FakeHttp({
        "https://league.test/robots.txt": [FakeResponse(404)],
        url: [FakeResponse(200, "<html>ok</html>")],
    })
    fetcher = _fetcher(tmp_path, http, [])

    assert fetcher.fetch(url) == "<html>ok</html>"
    assert fetcher.cache_path(url).exists()
    assert fetcher.fetch(url) == "<html>ok</html>"  # second call: from disk
    assert http.calls.count(url) == 1


def test_robots_disallow_is_respected(tmp_path: Path) -> None:
    robots = "User-agent: *\nDisallow: /private/\n"
    http = FakeHttp({"https://league.test/robots.txt": [FakeResponse(200, robots)]})
    fetcher = _fetcher(tmp_path, http, [])
    with pytest.raises(RobotsDisallowedError):
        fetcher.fetch("https://league.test/private/page")
    assert fetcher.is_allowed("https://league.test/public/page")


def test_delay_and_retry_on_server_error(tmp_path: Path) -> None:
    url = "https://league.test/p"
    http = FakeHttp({
        "https://league.test/robots.txt": [FakeResponse(404)],
        url: [FakeResponse(503), FakeResponse(200, "fine")],
    })
    sleeps: list[float] = []
    fetcher = _fetcher(tmp_path, http, sleeps)
    assert fetcher.fetch(url) == "fine"
    # Clock frozen at 0. Request 1 (robots.txt): no wait. Request 2 (page, -> 503):
    # 2 s delay. Then back-off 2**1. Request 3 (page, -> 200): 2 s delay again.
    assert sleeps == [2.0, 2, 2.0]


def test_offline_mode_never_hits_network(tmp_path: Path) -> None:
    from football_predictor.ingest.scraper import NotCachedError

    http = FakeHttp({})
    with pytest.raises(NotCachedError):
        _fetcher(tmp_path, http, []).fetch("https://league.test/x", offline=True)
    assert http.calls == []
