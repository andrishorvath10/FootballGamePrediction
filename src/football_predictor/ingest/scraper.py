"""Configurable, polite web scraper template for fixture/result pages.

Two clearly separated stages:

1. **Fetch** (:class:`PoliteFetcher`): downloads HTML while respecting robots.txt,
   waiting between requests, retrying transient errors, and caching every page
   in ``data/raw/``. A cached page is never downloaded again unless ``refresh``.
2. **Parse** (:func:`parse_fixtures`): a pure function ``HTML -> list[MatchRecord]``
   driven by CSS selectors from the config. No network, so it is easy to test on
   a saved HTML file and to re-run after fixing a selector.

The selectors in ``examples/scraper.example.toml`` are placeholders: every league
website has its own HTML structure, so adapt them to the target site.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag

from football_predictor.db.models import MatchStatus
from football_predictor.ingest.records import MatchRecord, clean_name

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "football-predictor/0.1 (personal non-commercial research project)"
RETRY_STATUS = {429, 500, 502, 503, 504}

SCORE_RE = re.compile(r"(\d+)\s*[-:–]\s*(\d+)")
STATUS_KEYWORDS: dict[MatchStatus, tuple[str, ...]] = {
    MatchStatus.POSTPONED: ("elhalasztva", "halasztva", "elh.", "postponed"),
    MatchStatus.CANCELLED: ("törölve", "elmarad", "cancelled", "canceled"),
}


# ----------------------------------------------------------------------- config
@dataclass(frozen=True)
class SelectorConfig:
    """CSS selectors. ``match_row`` is global; the others are relative to a row."""

    match_row: str
    home_team: str
    away_team: str
    score: str
    round: str | None = None
    date: str | None = None


@dataclass(frozen=True)
class PageConfig:
    """A page to scrape; ``round`` is used when the rows carry no round number."""

    url: str
    round: int | None = None


@dataclass(frozen=True)
class ScraperConfig:
    """Complete scraper configuration (normally loaded from a TOML file)."""

    season: str
    selectors: SelectorConfig
    pages: tuple[PageConfig, ...]
    date_format: str = "%Y-%m-%d"
    delay_seconds: float = 3.0
    timeout_seconds: float = 20.0
    max_retries: int = 3
    user_agent: str = DEFAULT_USER_AGENT

    @classmethod
    def from_toml(cls, path: Path) -> ScraperConfig:
        """Load the config from a TOML file (see ``examples/scraper.example.toml``)."""
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        general = data.get("scraper", {})
        return cls(
            season=general["season"],
            selectors=SelectorConfig(**data["selectors"]),
            pages=tuple(PageConfig(**p) for p in data.get("pages", [])),
            date_format=general.get("date_format", "%Y-%m-%d"),
            delay_seconds=float(general.get("delay_seconds", 3.0)),
            timeout_seconds=float(general.get("timeout_seconds", 20.0)),
            max_retries=int(general.get("max_retries", 3)),
            user_agent=general.get("user_agent", DEFAULT_USER_AGENT),
        )


# ------------------------------------------------------------------------ fetch
class RobotsDisallowedError(RuntimeError):
    """The site's robots.txt forbids fetching this URL."""


class NotCachedError(RuntimeError):
    """Offline mode was requested but the page is not in the cache."""


class PoliteFetcher:
    """HTTP fetcher with robots.txt support, rate limiting, retries and a disk cache."""

    def __init__(
        self,
        raw_dir: Path,
        user_agent: str = DEFAULT_USER_AGENT,
        delay_seconds: float = 3.0,
        timeout_seconds: float = 20.0,
        max_retries: int = 3,
        http: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.user_agent = user_agent
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.http = http or requests.Session()
        self.http.headers["User-Agent"] = user_agent
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self._robots: dict[str, RobotFileParser] = {}

    def cache_path(self, url: str) -> Path:
        """Readable, collision-free file name for a URL."""
        parsed = urlparse(url)
        raw = parsed.netloc + parsed.path + (f"-{parsed.query}" if parsed.query else "")
        slug = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-")[:80]
        digest = hashlib.sha256(url.encode()).hexdigest()[:10]
        return self.raw_dir / f"{slug}__{digest}.html"

    def cache_age(self, url: str) -> float | None:
        """Seconds since ``url`` was cached, or None if it is not in the cache."""
        path = self.cache_path(url)
        return time.time() - path.stat().st_mtime if path.exists() else None

    def _throttle(self, delay: float) -> None:
        """Sleep so that at least ``delay`` seconds pass between two requests."""
        if self._last_request is not None:
            wait = delay - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _robots_for(self, url: str) -> RobotFileParser:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        if host in self._robots:
            return self._robots[host]

        robots = RobotFileParser()
        robots_url = f"{host}/robots.txt"
        try:
            self._throttle(self.delay_seconds)
            resp = self.http.get(robots_url, timeout=self.timeout_seconds)
            if resp.status_code in (401, 403):
                robots.disallow_all = True
            elif 400 <= resp.status_code < 500:
                robots.allow_all = True  # no robots.txt -> no restrictions
            elif resp.ok:
                robots.parse(resp.text.splitlines())
            else:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
        except requests.RequestException as exc:
            # Be conservative: if we cannot read the rules, we do not crawl.
            logger.warning("Could not read %s (%s) - treating site as disallowed", robots_url, exc)
            robots.disallow_all = True
        self._robots[host] = robots
        return robots

    def is_allowed(self, url: str) -> bool:
        """True if robots.txt permits our user agent to fetch ``url``."""
        return self._robots_for(url).can_fetch(self.user_agent, url)

    def fetch(self, url: str, refresh: bool = False, offline: bool = False) -> str:
        """Return the HTML of ``url``, from cache if possible.

        Args:
            refresh: ignore the cache and download again.
            offline: never touch the network; raise :class:`NotCachedError` if missing.
        """
        path = self.cache_path(url)
        if path.exists() and not refresh:
            logger.debug("Cache hit %s -> %s", url, path.name)
            return path.read_text(encoding="utf-8")
        if offline:
            raise NotCachedError(url)

        robots = self._robots_for(url)
        if not robots.can_fetch(self.user_agent, url):
            raise RobotsDisallowedError(url)
        delay = max(self.delay_seconds, float(robots.crawl_delay(self.user_agent) or 0))

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle(delay)
            try:
                resp = self.http.get(url, timeout=self.timeout_seconds)
                if resp.status_code in RETRY_STATUS:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                status = getattr(exc.response, "status_code", None)
                if status is not None and status not in RETRY_STATUS:
                    raise  # 404 etc.: retrying will not help
                last_error = exc
                logger.warning("Attempt %d/%d for %s failed: %s",
                               attempt, self.max_retries, url, exc)
                if attempt < self.max_retries:
                    self._sleep(2**attempt)  # exponential back-off
                continue

            # requests guesses ISO-8859-1 when the server omits the charset,
            # which garbles Hungarian accents; let it detect the real encoding.
            if resp.encoding is None or resp.encoding.lower() == "iso-8859-1":
                resp.encoding = resp.apparent_encoding
            html = resp.text
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
            logger.info("Fetched %s -> %s", url, path.name)
            return html

        raise RuntimeError(f"Giving up on {url}") from last_error


# ------------------------------------------------------------------------ parse
def parse_score(text: str) -> tuple[int, int] | None:
    """Extract a score like ``"2-1"``, ``"2 : 1"`` or ``"2–1"``; None if absent."""
    m = SCORE_RE.search(text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_status_text(text: str) -> MatchStatus | None:
    """Detect postponed/cancelled keywords (Hungarian or English)."""
    lowered = text.casefold()
    for status, keywords in STATUS_KEYWORDS.items():
        if any(k in lowered for k in keywords):
            return status
    return None


def _cell_text(row: Tag, selector: str | None) -> str:
    if not selector:
        return ""
    cell = row.select_one(selector)
    return clean_name(cell.get_text(" ")) if cell else ""


def _parse_date(text: str, fmt: str) -> date | None:
    if not text:
        return None
    try:
        return datetime.strptime(text, fmt).date()
    except ValueError:
        logger.warning("Unparseable date %r (expected format %s)", text, fmt)
        return None


def parse_fixtures(
    html: str,
    config: ScraperConfig,
    source_url: str | None = None,
    default_round: int | None = None,
) -> list[MatchRecord]:
    """Turn a fixtures/results HTML page into match records (pure function)."""
    soup = BeautifulSoup(html, "html.parser")
    sel = config.selectors
    records: list[MatchRecord] = []

    for i, row in enumerate(soup.select(sel.match_row), start=1):
        home = _cell_text(row, sel.home_team)
        away = _cell_text(row, sel.away_team)
        if not home or not away:
            logger.warning("Row %d: missing team name, skipped", i)
            continue

        round_text = _cell_text(row, sel.round)
        round_match = re.search(r"\d+", round_text)
        rnd = int(round_match.group()) if round_match else default_round
        if rnd is None:
            logger.warning("Row %d (%s - %s): no round number, skipped", i, home, away)
            continue

        score_text = _cell_text(row, sel.score)
        score = parse_score(score_text)
        if score is not None:
            status = MatchStatus.PLAYED
        else:
            status = parse_status_text(score_text) or MatchStatus.SCHEDULED

        records.append(
            MatchRecord(
                season=config.season,
                round=rnd,
                home_team=home,
                away_team=away,
                match_date=_parse_date(_cell_text(row, sel.date), config.date_format),
                home_goals=score[0] if score else None,
                away_goals=score[1] if score else None,
                status=status,
                source_url=source_url,
            )
        )
    logger.info("Parsed %d matches from %s", len(records), source_url or "HTML")
    return records


def scrape(
    config: ScraperConfig, fetcher: PoliteFetcher, refresh: bool = False, offline: bool = False
) -> tuple[list[MatchRecord], list[str]]:
    """Fetch and parse every configured page.

    Returns (records, failed URLs). One failing page does not stop the others.
    """
    records: list[MatchRecord] = []
    failed: list[str] = []
    for page in config.pages:
        try:
            html = fetcher.fetch(page.url, refresh=refresh, offline=offline)
        except Exception as exc:  # noqa: BLE001 - log and continue with other pages
            logger.error("Failed to fetch %s: %s", page.url, exc)
            failed.append(page.url)
            continue
        records.extend(parse_fixtures(html, config, source_url=page.url, default_round=page.round))
    return records, failed
