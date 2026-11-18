"""Adapter for the Hungarian FA's public data bank (https://adatbank.mlsz.hu).

URL scheme (discovered by inspecting the site):

* round page:  /league/{season_id}/{federation_id}/{league_id}/{round}.html
  - every ``#match_panel div.schedule`` is one fixture: home/away team, result, date
    (``#sorsolas_panel`` holds a different list with the same markup - ignore it)
  - played: ``span.schedule-points`` ("5 - 2") inside a link to the match page
  - not played: ``span.schedule-date`` ("03-14") - looks like a score, careful!
  - ``select#turns`` lists all round numbers of the league
* match page:  /match/{season_id}/{federation_id}/{league_id}/{round}/{match_id}.html
  - ``div.goals``: one row per goal, running score in ``td.goals_intimes``,
    event icon (event_goal / event_penalty_goal / ...) and minute
  - ``#left_team`` / ``#right_team``: starters, then a "CSERÉK" (substitutes) block
  - timeline ``img`` titles: substitutions ("61', Csere / Le: X / Be: Y"), red cards

IDs: MLSZ league IDs and team IDs change every season, player and match IDs are
stable. Players are therefore stored with ``external_id = "mlsz:<player id>"``.

As with the generic scraper, fetching (:class:`PoliteFetcher`) and parsing (pure
functions below) are separate, and every page is cached in ``data/raw/``.

PRIVACY: match pages contain real people's names (personal data). Scraped HTML and
the database stay local (git-ignored); never commit them.
"""

from __future__ import annotations

import logging
import re
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from football_predictor.db.models import MatchStatus
from football_predictor.ingest.records import (
    AppearanceRecord,
    GoalRecord,
    MatchRecord,
    PointAdjustmentRecord,
    clean_name,
    name_key,
)
from football_predictor.ingest.scraper import (
    DEFAULT_USER_AGENT,
    PoliteFetcher,
    parse_score,
    parse_status_text,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://adatbank.mlsz.hu"
MATCH_MINUTES = 90
DATE_RE = re.compile(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})")
MINUTE_RE = re.compile(r"(\d+)\s*(?:\+\s*(\d+))?")
PLAYER_ID_RE = re.compile(r"/player/(\d+)\.html")
MATCH_ID_RE = re.compile(r"/match/(?:\d+/){4}(\d+)\.html")
ICON_RE = re.compile(r"timeline/([A-Za-z_]+)\.png")
PENALTY_POINTS_RE = re.compile(r"Büntetőpont:\s*([+-]?\d+)", re.IGNORECASE)
KNOWN_GOAL_ICONS = {"event_goal", "event_penalty_goal"}


# ----------------------------------------------------------------------- config
@dataclass(frozen=True)
class MlszCompetition:
    """One division, with its MLSZ league ID per season."""

    name: str
    league_ids: dict[str, int]  # season name ("2026/27") -> MLSZ league id
    region: str | None = None
    level: int | None = None


@dataclass(frozen=True)
class MlszConfig:
    """What to scrape from the MLSZ data bank (see ``config/mlsz_gyms.toml``)."""

    federation_id: int
    season_ids: dict[str, int]  # season name -> MLSZ season id; first = default
    competitions: tuple[MlszCompetition, ...]
    delay_seconds: float = 3.0
    timeout_seconds: float = 20.0
    max_retries: int = 3
    user_agent: str = DEFAULT_USER_AGENT
    round_page_max_age_hours: float = 12.0

    @classmethod
    def from_toml(cls, path: Path) -> MlszConfig:
        """Load the configuration from a TOML file."""
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        general = data["mlsz"]
        return cls(
            federation_id=int(general["federation_id"]),
            season_ids={k: int(v) for k, v in general["season_ids"].items()},
            competitions=tuple(
                MlszCompetition(
                    name=c["name"],
                    league_ids={k: int(v) for k, v in c["league_ids"].items()},
                    region=c.get("region"),
                    level=c.get("level"),
                )
                for c in data.get("competitions", [])
            ),
            delay_seconds=float(general.get("delay_seconds", 3.0)),
            timeout_seconds=float(general.get("timeout_seconds", 20.0)),
            max_retries=int(general.get("max_retries", 3)),
            user_agent=general.get("user_agent", DEFAULT_USER_AGENT),
            round_page_max_age_hours=float(general.get("round_page_max_age_hours", 12.0)),
        )

    @property
    def default_season(self) -> str:
        """The first season listed in the config (normally the current one)."""
        return next(iter(self.season_ids))


def round_url(season_id: int, federation_id: int, league_id: int, rnd: int) -> str:
    """URL of one round's fixture page."""
    return f"{BASE_URL}/league/{season_id}/{federation_id}/{league_id}/{rnd}.html"


# ---------------------------------------------------------------- round pages
@dataclass
class MlszFixture:
    """One row of a round page."""

    round: int
    home_team: str
    away_team: str
    match_date: date | None
    score: tuple[int, int] | None
    status: MatchStatus
    match_url: str | None = None
    match_id: str | None = None


def _team_name(cell: Tag | None) -> str:
    if cell is None:
        return ""
    link = cell.select_one("a")
    if link is not None and link.get("title"):
        return clean_name(str(link["title"]))
    return clean_name(cell.get_text(" "))


def _parse_hu_date(text: str) -> date | None:
    """Parse "2026. 11. 08." (time part, if any, is ignored)."""
    m = DATE_RE.search(text)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def parse_rounds(html: str) -> list[int]:
    """All round numbers of the league, from the round selector."""
    soup = BeautifulSoup(html, "html.parser")
    rounds = []
    for opt in soup.select("select#turns option"):
        value = str(opt.get("value", ""))
        if value.isdigit() and int(value) > 0:
            rounds.append(int(value))
    return sorted(set(rounds))


@dataclass
class StandingRow:
    """One row of the official standings table shown on every round page."""

    position: int
    team: str
    played: int
    won: int
    drawn: int
    lost: int
    goals_for: int
    goals_against: int
    points: int
    note: str = ""

    @property
    def point_adjustment(self) -> int:
        """Official points minus 3·W + D: non-zero means a deduction (or award)."""
        return self.points - (3 * self.won + self.drawn)


def parse_standings(html: str) -> list[StandingRow]:
    """The official table ("Tabella" tab) of a round page; empty list if absent."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[StandingRow] = []
    for tr in soup.select("#tabella_panel table tr[data-origrank]"):
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 11:
            continue
        nums = [c.get_text(strip=True) for c in cells[3:11]]
        try:
            played, won, drawn, lost, gf, ga, _gd, points = (int(x) for x in nums)
            position = int(cells[0].get_text(strip=True))
        except ValueError:
            continue
        note = cells[12].get_text(" ", strip=True) if len(cells) > 12 else ""
        rows.append(StandingRow(position, clean_name(cells[2].get_text(" ")), played, won,
                                drawn, lost, gf, ga, points, note))
    return rows


def point_adjustments_from_standings(rows: list[StandingRow]) -> dict[str, tuple[int, str]]:
    """Team -> (points, note) from the "Megjegyzés" column ("Büntetőpont: -4").

    The note holds the team's *current* total deduction, also on old round pages
    whose table predates it, so it is preferred over points - (3·W + D).
    """
    out = {}
    for row in rows:
        m = PENALTY_POINTS_RE.search(row.note)
        if m and int(m.group(1)) != 0:
            out[row.team] = (int(m.group(1)), row.note)
    return out


def parse_round_page(html: str, rnd: int) -> list[MlszFixture]:
    """Parse all fixtures of one round page (pure function, no network)."""
    soup = BeautifulSoup(html, "html.parser")
    fixtures: list[MlszFixture] = []
    panel = soup.select_one("#match_panel")
    if panel is None:
        logger.warning("Round %d: no #match_panel on page - layout changed?", rnd)
        return fixtures
    for box in panel.select("div.schedule"):
        home = _team_name(box.select_one(".home_team"))
        away = _team_name(box.select_one(".away_team"))
        result = box.select_one(".result-cont")
        if not home or not away or result is None:
            continue

        # Only a span.schedule-points holds a real score; span.schedule-date ("03-14")
        # would otherwise be mis-read as a 3-14 result.
        points = result.select_one(".schedule-points")
        score = parse_score(points.get_text()) if points else None
        if score is not None:
            status = MatchStatus.PLAYED
        else:
            status = parse_status_text(result.get_text(" ")) or MatchStatus.SCHEDULED

        link = result.select_one("a[href*='/match/']")
        match_url = str(link["href"]) if link else None
        id_match = MATCH_ID_RE.search(match_url) if match_url else None

        date_cell = box.select_one(".team_sorsolas_date")
        fixtures.append(
            MlszFixture(
                round=rnd,
                home_team=home,
                away_team=away,
                match_date=_parse_hu_date(date_cell.get_text(" ")) if date_cell else None,
                score=score,
                status=status,
                match_url=match_url,
                match_id=id_match.group(1) if id_match else None,
            )
        )
    return fixtures


# ---------------------------------------------------------------- match pages
@dataclass
class _LineupPlayer:
    name: str
    external_id: str | None
    team: str
    start: int | None  # minute he came on (0 for starters); None = unused substitute
    end: int = MATCH_MINUTES


@dataclass
class MatchDetails:
    """Goals and appearances parsed from a match page.

    ``None`` means "not available on the page" (as opposed to an empty list).
    """

    goals: list[GoalRecord] | None = None
    appearances: list[AppearanceRecord] | None = None
    warnings: list[str] = field(default_factory=list)


def parse_minute(text: str) -> int | None:
    """``"4'"`` -> 4, ``"90+3'"`` -> 93 (capped at 130, the schema limit)."""
    m = MINUTE_RE.search(text)
    if not m:
        return None
    minute = int(m.group(1)) + int(m.group(2) or 0)
    return min(max(minute, 1), 130)


def _player_link(cell: Tag) -> tuple[str | None, str | None]:
    """(name, external id) of the first player link in ``cell``."""
    link = cell.select_one("a[href*='/player/']")
    if link is None:
        return None, None
    name = clean_name(str(link.get("title") or link.get_text(" ")))
    m = PLAYER_ID_RE.search(str(link["href"]))
    return name or None, f"mlsz:{m.group(1)}" if m else None


def _parse_goals(
    soup: BeautifulSoup, home: str, away: str, warnings: list[str]
) -> list[GoalRecord]:
    goals: list[GoalRecord] = []
    prev = (0, 0)
    for row in soup.select("div.goals tr"):
        score_cell = row.select_one("td.goals_intimes")
        running = parse_score(score_cell.get_text()) if score_cell else None
        if running is None:
            continue
        # The running score tells us which team was credited - this also works for
        # own goals, whose scorer is listed but plays for the other side.
        if running == (prev[0] + 1, prev[1]):
            team = home
        elif running == (prev[0], prev[1] + 1):
            team = away
        else:
            warnings.append(f"unexpected running score {prev} -> {running}")
            prev = running
            continue
        prev = running

        info_cell = next(
            (c for c in row.select("td.left_team_player, td.right_team_player")
             if c.select_one(".goals_info")),
            None,
        )
        if info_cell is None:
            goals.append(GoalRecord(team=team))
            continue
        player, external_id = _player_link(info_cell)
        info = info_cell.select_one(".goals_info")
        icon_span = info.select_one("span") if info else None
        icon_match = ICON_RE.search(str(icon_span.get("style", ""))) if icon_span else None
        icon = icon_match.group(1) if icon_match else ""
        is_own = "own" in icon or "ongol" in icon or "ongoal" in icon
        if icon and icon not in KNOWN_GOAL_ICONS and not is_own:
            warnings.append(f"unknown goal icon {icon!r}")
        # Home side: <span/><p>4'</p>; away side is mirrored with an extra empty <p>.
        minute_texts = [el.get_text(strip=True) for el in info.select("p")] if info else []
        minute_text = next((t for t in minute_texts if any(ch.isdigit() for ch in t)), "")
        goals.append(
            GoalRecord(
                team=team,
                player=player,
                player_external_id=external_id,
                minute=parse_minute(minute_text) if minute_text else None,
                is_penalty="penalty" in icon,
                is_own_goal=is_own,
            )
        )
    return goals


def _parse_lineup(team_div: Tag, team: str) -> list[_LineupPlayer]:
    players: list[_LineupPlayer] = []
    section: str | None = "starters"
    for row in team_div.select("tr"):
        subhead = row.select_one(".match_table_subhead")
        if subhead is not None:
            section = "subs" if "CSER" in subhead.get_text().upper() else None
            continue
        if section is None:
            continue
        cell = row.select_one("td.match_players_name")
        if cell is None:
            continue
        name, external_id = _player_link(cell)
        if name:
            players.append(
                _LineupPlayer(name=name, external_id=external_id, team=team,
                              start=0 if section == "starters" else None)
            )
    return players


def _apply_timeline(soup: BeautifulSoup, lineup: list[_LineupPlayer], warnings: list[str]) -> None:
    """Use substitution and red-card events to compute on-pitch intervals."""
    by_name: dict[str, _LineupPlayer] = {}
    for p in lineup:
        by_name.setdefault(name_key(p.name), p)

    for img in soup.select("img[src*='/timeline/']"):
        src = str(img.get("src", ""))
        title = str(img.get("title", ""))
        minute = parse_minute(title.split(",")[0]) or 0
        minute = min(minute, MATCH_MINUTES)
        if "event_swap" in src:
            off = re.search(r"Le:\s*(.+)", title)
            on = re.search(r"Be:\s*(.+)", title)
            if off and (p := by_name.get(name_key(off.group(1)))):
                p.end = minute
            if on and (p := by_name.get(name_key(on.group(1)))):
                p.start = minute
            if not (off and on):
                warnings.append(f"unparsed substitution {title!r}")
        elif "red" in src.rsplit("/", 1)[-1]:
            name = title.splitlines()[-1] if "\n" in title else ""
            if p := by_name.get(name_key(name.replace("\xa0", " "))):
                p.end = min(p.end, minute)


def parse_match_page(html: str, home_team: str, away_team: str) -> MatchDetails:
    """Parse goals and line-ups (with minutes played) from a match page."""
    soup = BeautifulSoup(html, "html.parser")
    details = MatchDetails()

    if soup.select_one("div.goals") is not None:
        details.goals = _parse_goals(soup, home_team, away_team, details.warnings)

    lineup: list[_LineupPlayer] = []
    for div_id, team in (("left_team", home_team), ("right_team", away_team)):
        team_div = soup.find(id=div_id)
        if isinstance(team_div, Tag):
            lineup += _parse_lineup(team_div, team)
    if lineup:
        _apply_timeline(soup, lineup, details.warnings)
        details.appearances = [
            AppearanceRecord(
                team=p.team,
                player=p.name,
                player_external_id=p.external_id,
                minutes_played=max(0, p.end - p.start),
            )
            for p in lineup
            if p.start is not None  # unused substitutes did not appear
        ]
    return details


# --------------------------------------------------------------------- crawler
class MlszScraper:
    """Crawls round pages (and optionally match pages) of configured competitions."""

    def __init__(
        self,
        config: MlszConfig,
        fetcher: PoliteFetcher,
        fetch_details: bool = True,
        refresh: bool = False,
        offline: bool = False,
        today: date | None = None,
    ) -> None:
        self.config = config
        self.fetcher = fetcher
        self.fetch_details = fetch_details
        self.refresh = refresh
        self.offline = offline
        self.today = today or date.today()
        self.failed: list[str] = []
        # (competition, season) -> deductions read from the freshest standings table
        self.adjustments: dict[tuple[str, str], list[PointAdjustmentRecord]] = {}

    def _first_round_html(self, url: str) -> str:
        """Round 1 page, refreshed after the max age even if fully played.

        It carries the list of all rounds; leagues often publish the spring
        fixtures later, so the list must be re-read to discover new rounds.
        """
        html = self.fetcher.fetch(url, refresh=self.refresh, offline=self.offline)
        if self.refresh or self.offline:
            return html
        age = self.fetcher.cache_age(url)
        if age is not None and age > self.config.round_page_max_age_hours * 3600:
            html = self.fetcher.fetch(url, refresh=True)
        return html

    def _round_html(self, url: str, rnd: int) -> str:
        """Fetch a round page; re-download a cached copy only if it may be outdated.

        A cached round page is stale when it still has unplayed matches whose date
        has passed (results may have arrived), and it is older than the max age.
        Fully played rounds are never re-downloaded.
        """
        html = self.fetcher.fetch(url, refresh=self.refresh, offline=self.offline)
        if self.refresh or self.offline:
            return html
        age = self.fetcher.cache_age(url)
        max_age = self.config.round_page_max_age_hours * 3600
        if age is not None and age > max_age:
            pending = [
                f for f in parse_round_page(html, rnd)
                if f.score is None and (f.match_date is None or f.match_date <= self.today)
            ]
            if pending:
                logger.info("Round %d has %d pending match(es); refreshing", rnd, len(pending))
                html = self.fetcher.fetch(url, refresh=True)
        return html

    def scrape(self, competition: MlszCompetition, season: str) -> list[MatchRecord]:
        """All fixtures (with details for played matches) of one competition-season."""
        season_id = self.config.season_ids[season]
        league_id = competition.league_ids[season]
        fed = self.config.federation_id

        first_url = round_url(season_id, fed, league_id, 1)
        first_html = self._first_round_html(first_url)
        rounds = parse_rounds(first_html) or [1]
        logger.info("%s %s: %d rounds", competition.name, season, len(rounds))

        records: list[MatchRecord] = []
        freshest: tuple[float, str] | None = None  # (cache age, html)
        for rnd in rounds:
            url = round_url(season_id, fed, league_id, rnd)
            try:
                html = first_html if rnd == 1 else self._round_html(url, rnd)
            except Exception as exc:  # noqa: BLE001 - keep going with other rounds
                logger.error("Round %d failed: %s", rnd, exc)
                self.failed.append(url)
                continue
            age = self.fetcher.cache_age(url)
            age = 0.0 if age is None else age
            if freshest is None or age < freshest[0]:
                freshest = (age, html)
            for fx in parse_round_page(html, rnd):
                records.append(self._to_record(fx, competition.name, season, url))
        standings = parse_standings(freshest[1]) if freshest is not None else []
        if standings:  # no table at all -> leave stored adjustments untouched
            found = point_adjustments_from_standings(standings)
            self.adjustments[(competition.name, season)] = [
                PointAdjustmentRecord(season=season, competition=competition.name, team=team,
                                      points=points, reason=note)
                for team, (points, note) in sorted(found.items())
            ]
        return records

    def _to_record(
        self, fx: MlszFixture, competition: str, season: str, round_page: str
    ) -> MatchRecord:
        rec = MatchRecord(
            season=season,
            competition=competition,
            round=fx.round,
            home_team=fx.home_team,
            away_team=fx.away_team,
            match_date=fx.match_date,
            home_goals=fx.score[0] if fx.score else None,
            away_goals=fx.score[1] if fx.score else None,
            status=fx.status,
            source_url=fx.match_url or round_page,
            external_id=f"mlsz:{fx.match_id}" if fx.match_id else None,
        )
        if fx.score == (0, 0):
            rec.goals = []  # a goalless draw needs no page to know its goal list
        if fx.score is None or not self.fetch_details or not fx.match_url:
            return rec
        try:
            html = self.fetcher.fetch(fx.match_url, refresh=self.refresh, offline=self.offline)
        except Exception as exc:  # noqa: BLE001
            logger.error("Match page %s failed: %s", fx.match_url, exc)
            self.failed.append(fx.match_url)
            return rec

        details = parse_match_page(html, fx.home_team, fx.away_team)
        for w in details.warnings:
            logger.warning("%s: %s", fx.match_url, w)
        if details.goals is not None and (details.goals or sum(fx.score) == 0):
            rec.goals = details.goals
        rec.appearances = details.appearances
        return rec

    def iter_targets(
        self, seasons: list[str] | None = None, competitions: list[str] | None = None
    ) -> Iterator[tuple[MlszCompetition, str]]:
        """(competition, season) pairs to scrape, filtered by the given names."""
        seasons = seasons or [self.config.default_season]
        wanted = {name_key(c) for c in competitions} if competitions else None
        for comp in self.config.competitions:
            if wanted is not None and name_key(comp.name) not in wanted:
                continue
            for season in seasons:
                if season not in comp.league_ids:
                    logger.warning("%s has no league id for %s - skipped", comp.name, season)
                    continue
                yield comp, season
