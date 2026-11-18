"""Idempotent loading of :class:`MatchRecord` objects into the database (upsert).

Loading the same records twice leaves the database unchanged:

* seasons and competitions are found by name, teams via the ``team_alias`` table,
* matches by their natural key (competition, season, round, home team, away team),
* players by their source ID (e.g. MLSZ player ID) if the source provides one,
  otherwise by (name, team) - a player who already played for this team is reused,
* goals / appearances of a match are compared with the incoming set and only
  replaced if they differ (they have no reliable natural key of their own).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from football_predictor.db.models import (
    Appearance,
    Competition,
    Goal,
    Match,
    MatchStatus,
    Player,
    PlayerTeam,
    PointAdjustment,
    Season,
    Team,
    TeamAlias,
)
from football_predictor.ingest.records import (
    MatchRecord,
    PointAdjustmentRecord,
    clean_name,
    name_key,
)

logger = logging.getLogger(__name__)

GoalTuple = tuple[int, int | None, int | None, bool, bool]


@dataclass
class LoadStats:
    """Counters describing what a load did."""

    matches_created: int = 0
    matches_updated: int = 0
    matches_unchanged: int = 0
    teams_created: int = 0
    players_created: int = 0

    def __str__(self) -> str:
        return (
            f"matches: {self.matches_created} created, {self.matches_updated} updated, "
            f"{self.matches_unchanged} unchanged; teams created: {self.teams_created}; "
            f"players created: {self.players_created}"
        )


def _goal_sort_key(g: GoalTuple) -> tuple[int, int, int, bool, bool]:
    """Sort key tolerant of None values (None sorts first)."""
    return (g[0], -1 if g[1] is None else g[1], -1 if g[2] is None else g[2], g[3], g[4])


class LeagueLoader:
    """Writes records into the database inside the caller's session/transaction."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.stats = LoadStats()
        self._seasons: dict[str, Season] = {}
        self._competitions: dict[str, int] = {}
        self._teams: dict[str, int] = {}
        self._players: dict[tuple[str, int], int] = {}
        self._memberships: set[tuple[int, int, int]] = set()

    # ------------------------------------------------------------------ public
    def load(self, records: Iterable[MatchRecord]) -> LoadStats:
        """Upsert all records; returns statistics. Does not commit."""
        for rec in records:
            self._load_match(rec)
        self.session.flush()
        self._update_season_dates()
        logger.info("Load finished - %s", self.stats)
        return self.stats

    def sync_point_adjustments(
        self, season: str, competition: str, records: Iterable[PointAdjustmentRecord]
    ) -> int:
        """Make the stored adjustments of one competition-season equal ``records``.

        The source lists the complete current set, so rows it no longer lists
        (a revoked deduction) are deleted. Returns the number of changed rows.
        """
        season_id = self._season(season).id
        comp_id = self._competition_id(competition)
        existing = {
            row.team_id: row for row in self.session.scalars(
                select(PointAdjustment).where(PointAdjustment.season_id == season_id,
                                              PointAdjustment.competition_id == comp_id))
        }
        changed = 0
        incoming: set[int] = set()
        for rec in records:
            team_id = self._team_id(rec.team)
            incoming.add(team_id)
            row = existing.get(team_id)
            if row is None:
                self.session.add(PointAdjustment(season_id=season_id, competition_id=comp_id,
                                                 team_id=team_id, points=rec.points,
                                                 reason=rec.reason))
                changed += 1
            elif (row.points, row.reason) != (rec.points, rec.reason):
                row.points, row.reason = rec.points, rec.reason
                changed += 1
        for team_id, row in existing.items():
            if team_id not in incoming:
                self.session.delete(row)
                changed += 1
        self.session.flush()
        if changed:
            logger.info("%s %s: %d point adjustment(s) changed", competition, season, changed)
        return changed

    # ---------------------------------------------------------------- lookups
    def _season(self, name: str) -> Season:
        name = clean_name(name)
        season = self._seasons.get(name)
        if season is None:
            season = self.session.scalar(select(Season).where(Season.name == name))
            if season is None:
                season = Season(name=name)
                self.session.add(season)
                self.session.flush()
                logger.info("Created season %r", name)
            self._seasons[name] = season
        return season

    def _competition_id(self, name: str) -> int:
        name = clean_name(name)
        comp_id = self._competitions.get(name)
        if comp_id is None:
            comp_id = self.session.scalar(select(Competition.id).where(Competition.name == name))
            if comp_id is None:
                comp = Competition(name=name)
                self.session.add(comp)
                self.session.flush()
                comp_id = comp.id
                logger.info("Created competition %r", name)
            self._competitions[name] = comp_id
        return comp_id

    def _team_id(self, name: str) -> int:
        """Resolve a team name through its aliases, creating the team if new."""
        key = name_key(name)
        team_id = self._teams.get(key)
        if team_id is not None:
            return team_id

        team_id = self.session.scalar(select(TeamAlias.team_id).where(TeamAlias.alias == key))
        if team_id is None:
            team = self.session.scalar(select(Team).where(Team.name == clean_name(name)))
            if team is None:
                team = Team(name=clean_name(name))
                self.session.add(team)
                self.session.flush()
                self.stats.teams_created += 1
                logger.info("Created team %r", team.name)
            self.session.add(TeamAlias(alias=key, team_id=team.id))
            team_id = team.id
        self._teams[key] = team_id
        return team_id

    def _player_id(
        self, name: str, team_id: int, season_id: int, external_id: str | None = None
    ) -> int:
        """Find (or create) a player; ensure the (player, team, season) membership.

        With an ``external_id`` the player is identified by it (robust to transfers
        and namesakes); without one, by name within the team.
        """
        name = clean_name(name)
        cache_key = (external_id, 0) if external_id else (name, team_id)
        player_id = self._players.get(cache_key)
        if player_id is None and external_id:
            player = self.session.scalar(select(Player).where(Player.external_id == external_id))
            if player is None:
                player = Player(name=name, external_id=external_id)
                self.session.add(player)
                self.session.flush()
                self.stats.players_created += 1
            player_id = player.id
            self._players[cache_key] = player_id
        if player_id is None:
            player_id = self.session.scalar(
                select(Player.id)
                .join(PlayerTeam, PlayerTeam.player_id == Player.id)
                .where(Player.name == name, PlayerTeam.team_id == team_id)
                .order_by(Player.id)
                .limit(1)
            )
            if player_id is None:
                player = Player(name=name)
                self.session.add(player)
                self.session.flush()
                player_id = player.id
                self.stats.players_created += 1
            self._players[cache_key] = player_id

        membership = (player_id, team_id, season_id)
        if membership not in self._memberships:
            pk = {"player_id": player_id, "team_id": team_id, "season_id": season_id}
            if self.session.get(PlayerTeam, pk) is None:
                self.session.add(PlayerTeam(**pk))
            self._memberships.add(membership)
        return player_id

    # ---------------------------------------------------------------- matches
    def _load_match(self, rec: MatchRecord) -> None:
        season = self._season(rec.season)
        competition_id = self._competition_id(rec.competition)
        home_id = self._team_id(rec.home_team)
        away_id = self._team_id(rec.away_team)
        if home_id == away_id:
            raise ValueError(f"Team plays itself: {rec.home_team!r} vs {rec.away_team!r}")

        match = self.session.scalar(
            select(Match).where(
                Match.season_id == season.id,
                Match.competition_id == competition_id,
                Match.round == rec.round,
                Match.home_team_id == home_id,
                Match.away_team_id == away_id,
            )
        )
        created = match is None
        if match is None:
            match = Match(
                season_id=season.id,
                competition_id=competition_id,
                round=rec.round,
                home_team_id=home_id,
                away_team_id=away_id,
                status=MatchStatus.SCHEDULED,
            )
            self.session.add(match)

        changed = self._apply_fields(match, rec)
        if rec.goals is not None:
            changed |= self._sync_goals(match, rec, season.id, home_id, away_id)
        if rec.appearances is not None:
            changed |= self._sync_appearances(match, rec, season.id, home_id, away_id)

        if created:
            self.stats.matches_created += 1
        elif changed:
            self.stats.matches_updated += 1
        else:
            self.stats.matches_unchanged += 1

    @staticmethod
    def _apply_fields(match: Match, rec: MatchRecord) -> bool:
        """Copy non-None scalar fields onto the match; return True if anything changed."""
        updates = {
            "match_date": rec.match_date,
            "home_goals": rec.home_goals,
            "away_goals": rec.away_goals,
            "status": rec.effective_status(),
            "source_url": rec.source_url,
            "external_id": rec.external_id,
        }
        changed = False
        for field, value in updates.items():
            if value is not None and getattr(match, field) != value:
                setattr(match, field, value)
                changed = True
        return changed

    def _sync_goals(
        self, match: Match, rec: MatchRecord, season_id: int, home_id: int, away_id: int
    ) -> bool:
        desired: list[GoalTuple] = []
        for g in rec.goals or []:
            team_id = self._team_id(g.team)
            if team_id not in (home_id, away_id):
                raise ValueError(f"Goal team {g.team!r} did not play in this match")
            player_id = None
            if g.player:
                # An own-goal scorer belongs to the *other* team.
                opponent_id = away_id if team_id == home_id else home_id
                scorer_team = opponent_id if g.is_own_goal else team_id
                player_id = self._player_id(
                    g.player, scorer_team, season_id, g.player_external_id
                )
            desired.append((team_id, player_id, g.minute, g.is_penalty, g.is_own_goal))

        current: list[GoalTuple] = [
            (x.team_id, x.player_id, x.minute, x.is_penalty, x.is_own_goal) for x in match.goals
        ]
        if sorted(desired, key=_goal_sort_key) == sorted(current, key=_goal_sort_key):
            return False

        match.goals.clear()  # delete-orphan cascade removes the old rows
        for team_id, player_id, minute, pen, og in desired:
            match.goals.append(
                Goal(team_id=team_id, player_id=player_id, minute=minute,
                     is_penalty=pen, is_own_goal=og)
            )
        return True

    def _sync_appearances(
        self, match: Match, rec: MatchRecord, season_id: int, home_id: int, away_id: int
    ) -> bool:
        desired: dict[int, tuple[int, int | None]] = {}
        for a in rec.appearances or []:
            team_id = self._team_id(a.team)
            if team_id not in (home_id, away_id):
                raise ValueError(f"Appearance team {a.team!r} did not play in this match")
            player_id = self._player_id(a.player, team_id, season_id, a.player_external_id)
            if player_id in desired:
                raise ValueError(f"Duplicate appearance for {a.player!r}")
            desired[player_id] = (team_id, a.minutes_played)

        current = {x.player_id: (x.team_id, x.minutes_played) for x in match.appearances}
        if desired == current:
            return False

        match.appearances.clear()
        for player_id, (team_id, minutes) in desired.items():
            match.appearances.append(
                Appearance(player_id=player_id, team_id=team_id, minutes_played=minutes)
            )
        return True

    def _update_season_dates(self) -> None:
        """Widen each touched season's start/end dates to cover its matches."""
        for season in self._seasons.values():
            first, last = self.session.execute(
                select(func.min(Match.match_date), func.max(Match.match_date)).where(
                    Match.season_id == season.id
                )
            ).one()
            if first is not None and (season.start_date is None or first < season.start_date):
                season.start_date = first
            if last is not None and (season.end_date is None or last > season.end_date):
                season.end_date = last


def ensure_competition(
    session: Session, name: str, region: str | None = None, level: int | None = None
) -> Competition:
    """Create the competition if missing and fill in its region/level metadata."""
    name = clean_name(name)
    comp = session.scalar(select(Competition).where(Competition.name == name))
    if comp is None:
        comp = Competition(name=name)
        session.add(comp)
    if region is not None:
        comp.region = region
    if level is not None:
        comp.level = level
    return comp


def add_team_alias(session: Session, alias: str, team_name: str) -> None:
    """Register ``alias`` as another spelling of the existing team ``team_name``.

    Add aliases *before* importing data that uses them; otherwise the alias will
    already have been created as a separate team.
    """
    team_id = session.scalar(
        select(TeamAlias.team_id).where(TeamAlias.alias == name_key(team_name))
    ) or session.scalar(select(Team.id).where(Team.name == clean_name(team_name)))
    if team_id is None:
        raise ValueError(f"Unknown team: {team_name!r}")

    existing = session.scalar(select(TeamAlias).where(TeamAlias.alias == name_key(alias)))
    if existing is not None:
        if existing.team_id != team_id:
            raise ValueError(f"Alias {alias!r} already belongs to another team")
        return
    session.add(TeamAlias(alias=name_key(alias), team_id=team_id))
    logger.info("Alias %r -> team %d", alias, team_id)
