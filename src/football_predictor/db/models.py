"""ORM models (SQLAlchemy 2.0, typed declarative style).

Design notes
------------
* ``Match`` stores both played and *scheduled* fixtures. Scheduled matches have
  ``home_goals``/``away_goals`` = NULL; these are the rows we will predict.
* ``Goal.team_id`` is always the team that *benefits* from the goal (whose score
  increases). For an own goal, ``player_id`` points to the opponent who scored it,
  while ``team_id`` is still the benefiting team. This makes
  "count goals per team" a trivial GROUP BY.
* Players can move between teams, so team membership lives in ``PlayerTeam``
  (player, team, season) instead of a ``team_id`` column on ``Player``.
* ``TeamAlias`` maps alternative spellings ("Kiskunfélegyháza FC", "KFC", ...)
  to one canonical ``Team`` so imports from different sources don't create
  duplicate teams.
"""

from __future__ import annotations

import enum
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Deterministic constraint names: needed for clean migrations later (e.g. Alembic)
# and they make integrity error messages readable.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Common base class for all ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class MatchStatus(enum.StrEnum):
    """Lifecycle state of a fixture."""

    SCHEDULED = "scheduled"
    PLAYED = "played"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


# --------------------------------------------------------------------------- #
# Season
# --------------------------------------------------------------------------- #
class Season(Base):
    """A league season, e.g. "2025/26"."""

    __tablename__ = "season"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)

    matches: Mapped[list[Match]] = relationship(back_populates="season")

    __table_args__ = (
        CheckConstraint(
            "start_date IS NULL OR end_date IS NULL OR start_date <= end_date",
            name="date_order",
        ),
    )

    def __repr__(self) -> str:
        return f"Season(id={self.id}, name={self.name!r})"


# --------------------------------------------------------------------------- #
# Competition
# --------------------------------------------------------------------------- #
class Competition(Base):
    """A league / division, independent of season (e.g. "GYMS Megye II Észak").

    Models are fitted per (competition, season): teams in different divisions
    never meet, so their strengths are not directly comparable.
    """

    __tablename__ = "competition"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    region: Mapped[str | None] = mapped_column(String(100))
    level: Mapped[int | None] = mapped_column(Integer)  # 1 = highest tier in the region

    matches: Mapped[list[Match]] = relationship(back_populates="competition")

    def __repr__(self) -> str:
        return f"Competition(id={self.id}, name={self.name!r})"


# --------------------------------------------------------------------------- #
# Team + TeamAlias
# --------------------------------------------------------------------------- #
class Team(Base):
    """A club/team. ``name`` is the canonical spelling."""

    __tablename__ = "team"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    short_name: Mapped[str | None] = mapped_column(String(20))

    aliases: Mapped[list[TeamAlias]] = relationship(
        back_populates="team", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Team(id={self.id}, name={self.name!r})"


class TeamAlias(Base):
    """Alternative spelling of a team name, mapped to the canonical team."""

    __tablename__ = "team_alias"

    id: Mapped[int] = mapped_column(primary_key=True)
    alias: Mapped[str] = mapped_column(String(100), unique=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("team.id", ondelete="CASCADE"), index=True)

    team: Mapped[Team] = relationship(back_populates="aliases")

    def __repr__(self) -> str:
        return f"TeamAlias(alias={self.alias!r} -> team_id={self.team_id})"


# --------------------------------------------------------------------------- #
# Player + PlayerTeam
# --------------------------------------------------------------------------- #
class Player(Base):
    """A player. Names are NOT unique (two people can share a name)."""

    __tablename__ = "player"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    # Stable ID from the data source (e.g. "mlsz:396006"); NULL if the source has none.
    external_id: Mapped[str | None] = mapped_column(String(50), unique=True)

    memberships: Mapped[list[PlayerTeam]] = relationship(
        back_populates="player", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Player(id={self.id}, name={self.name!r})"


class PlayerTeam(Base):
    """Membership: player X played for team Y in season Z.

    A player who changes clubs mid-season simply gets two rows.
    """

    __tablename__ = "player_team"

    player_id: Mapped[int] = mapped_column(
        ForeignKey("player.id", ondelete="CASCADE"), primary_key=True
    )
    team_id: Mapped[int] = mapped_column(
        ForeignKey("team.id", ondelete="CASCADE"), primary_key=True
    )
    season_id: Mapped[int] = mapped_column(
        ForeignKey("season.id", ondelete="CASCADE"), primary_key=True
    )

    player: Mapped[Player] = relationship(back_populates="memberships")
    team: Mapped[Team] = relationship()
    season: Mapped[Season] = relationship()

    # "Who plays for team T in season S?" is a frequent query.
    __table_args__ = (Index("ix_player_team_team_season", "team_id", "season_id"),)


# --------------------------------------------------------------------------- #
# Match
# --------------------------------------------------------------------------- #
class Match(Base):
    """A fixture. Played matches have a score; scheduled ones have NULL goals."""

    __tablename__ = "match"

    id: Mapped[int] = mapped_column(primary_key=True)
    season_id: Mapped[int] = mapped_column(ForeignKey("season.id", ondelete="CASCADE"))
    competition_id: Mapped[int] = mapped_column(
        ForeignKey("competition.id", ondelete="CASCADE")
    )
    round: Mapped[int] = mapped_column(Integer)
    match_date: Mapped[date | None] = mapped_column(Date)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("team.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("team.id"))
    home_goals: Mapped[int | None] = mapped_column(Integer)
    away_goals: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[MatchStatus] = mapped_column(
        Enum(
            MatchStatus,
            name="match_status",
            native_enum=False,  # plain VARCHAR -> identical on SQLite and PostgreSQL
            create_constraint=True,  # ... plus a CHECK limiting it to the 4 values
            values_callable=lambda e: [m.value for m in e],
            length=20,
        ),
        default=MatchStatus.SCHEDULED,
    )
    source_url: Mapped[str | None] = mapped_column(String(500))
    external_id: Mapped[str | None] = mapped_column(String(50), unique=True)

    season: Mapped[Season] = relationship(back_populates="matches")
    competition: Mapped[Competition] = relationship(back_populates="matches")
    home_team: Mapped[Team] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped[Team] = relationship(foreign_keys=[away_team_id])
    goals: Mapped[list[Goal]] = relationship(
        back_populates="match", cascade="all, delete-orphan"
    )
    appearances: Mapped[list[Appearance]] = relationship(
        back_populates="match", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # Natural key -> enables idempotent upserts and prevents duplicates.
        UniqueConstraint(
            "season_id", "competition_id", "round", "home_team_id", "away_team_id",
            name="uq_match_fixture",
        ),
        CheckConstraint("round >= 1", name="round_positive"),
        CheckConstraint("home_team_id <> away_team_id", name="different_teams"),
        CheckConstraint(
            "(home_goals IS NULL OR home_goals >= 0) AND (away_goals IS NULL OR away_goals >= 0)",
            name="goals_non_negative",
        ),
        # Either both scores are known or neither is.
        CheckConstraint("(home_goals IS NULL) = (away_goals IS NULL)", name="score_complete"),
        # A played match must have a score.
        CheckConstraint(
            "status <> 'played' OR home_goals IS NOT NULL", name="played_has_score"
        ),
        Index("ix_match_season_status", "season_id", "status"),
        Index("ix_match_competition_season", "competition_id", "season_id"),
        Index("ix_match_home_team", "home_team_id"),
        Index("ix_match_away_team", "away_team_id"),
        Index("ix_match_date", "match_date"),
    )

    @property
    def is_played(self) -> bool:
        """True if the match has been played and has a final score."""
        return self.status == MatchStatus.PLAYED and self.home_goals is not None

    def __repr__(self) -> str:
        score = f"{self.home_goals}-{self.away_goals}" if self.home_goals is not None else "vs"
        return (
            f"Match(id={self.id}, round={self.round}, "
            f"{self.home_team_id} {score} {self.away_team_id}, status={self.status})"
        )


# --------------------------------------------------------------------------- #
# Goal
# --------------------------------------------------------------------------- #
class Goal(Base):
    """A single goal.

    ``team_id`` = the team credited with the goal (its score goes up).
    ``player_id`` = the scorer; NULL if unknown. For an own goal it is the
    opponent player who put the ball into his own net.
    """

    __tablename__ = "goal"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("match.id", ondelete="CASCADE"))
    player_id: Mapped[int | None] = mapped_column(ForeignKey("player.id", ondelete="SET NULL"))
    team_id: Mapped[int] = mapped_column(ForeignKey("team.id"))
    minute: Mapped[int | None] = mapped_column(Integer)
    is_penalty: Mapped[bool] = mapped_column(Boolean, default=False)
    is_own_goal: Mapped[bool] = mapped_column(Boolean, default=False)

    match: Mapped[Match] = relationship(back_populates="goals")
    player: Mapped[Player | None] = relationship()
    team: Mapped[Team] = relationship()

    __table_args__ = (
        # 130 = 120 min extra time + stoppage. NULL allowed: amateur reports often omit it.
        CheckConstraint("minute IS NULL OR (minute BETWEEN 1 AND 130)", name="minute_range"),
        CheckConstraint("NOT (is_penalty AND is_own_goal)", name="penalty_not_own_goal"),
        Index("ix_goal_match", "match_id"),
        Index("ix_goal_player", "player_id"),
        Index("ix_goal_team", "team_id"),
    )

    def __repr__(self) -> str:
        flags = "".join(["P" if self.is_penalty else "", "OG" if self.is_own_goal else ""])
        return (
            f"Goal(match={self.match_id}, team={self.team_id}, player={self.player_id}, "
            f"min={self.minute}{' ' + flags if flags else ''})"
        )


# --------------------------------------------------------------------------- #
# Appearance (optional line-up data)
# --------------------------------------------------------------------------- #
class Appearance(Base):
    """A player's participation in a match (optional; only if line-ups exist)."""

    __tablename__ = "appearance"

    match_id: Mapped[int] = mapped_column(
        ForeignKey("match.id", ondelete="CASCADE"), primary_key=True
    )
    player_id: Mapped[int] = mapped_column(
        ForeignKey("player.id", ondelete="CASCADE"), primary_key=True
    )
    team_id: Mapped[int] = mapped_column(ForeignKey("team.id"))
    minutes_played: Mapped[int | None] = mapped_column(Integer)

    match: Mapped[Match] = relationship(back_populates="appearances")
    player: Mapped[Player] = relationship()
    team: Mapped[Team] = relationship()

    __table_args__ = (
        CheckConstraint(
            "minutes_played IS NULL OR (minutes_played BETWEEN 0 AND 130)",
            name="minutes_range",
        ),
        Index("ix_appearance_player", "player_id"),
    )


# --------------------------------------------------------------------------- #
# PointAdjustment
# --------------------------------------------------------------------------- #
class PointAdjustment(Base):
    """Points added to / removed from a team's total by the federation.

    Typically a deduction (MLSZ "büntetőpont", e.g. -3 for fielding an ineligible
    player). One row per team per competition-season, holding the current total.
    """

    __tablename__ = "point_adjustment"

    id: Mapped[int] = mapped_column(primary_key=True)
    season_id: Mapped[int] = mapped_column(ForeignKey("season.id", ondelete="CASCADE"))
    competition_id: Mapped[int] = mapped_column(
        ForeignKey("competition.id", ondelete="CASCADE")
    )
    team_id: Mapped[int] = mapped_column(ForeignKey("team.id", ondelete="CASCADE"))
    points: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str | None] = mapped_column(String(200))

    team: Mapped[Team] = relationship()

    __table_args__ = (
        UniqueConstraint("season_id", "competition_id", "team_id",
                         name="uq_point_adjustment_team"),
        CheckConstraint("points <> 0", name="non_zero"),
    )

    def __repr__(self) -> str:
        return f"PointAdjustment(team_id={self.team_id}, points={self.points})"


# --------------------------------------------------------------------------- #
# Assumption
# --------------------------------------------------------------------------- #
class Assumption(Base):
    """A user's expert assumption about a team, applied by the season simulation.

    ``change`` = +0.2: from ``start_date`` the team scores 1.2× and concedes
    1/1.2× what the model expects; ``drift_factor`` widens its form random walk.
    Stored so it can later be checked against the matches played after
    ``start_date`` (``assume check``). One per team per competition-season.
    """

    __tablename__ = "assumption"

    id: Mapped[int] = mapped_column(primary_key=True)
    season_id: Mapped[int] = mapped_column(ForeignKey("season.id", ondelete="CASCADE"))
    competition_id: Mapped[int] = mapped_column(
        ForeignKey("competition.id", ondelete="CASCADE")
    )
    team_id: Mapped[int] = mapped_column(ForeignKey("team.id", ondelete="CASCADE"))
    change: Mapped[float] = mapped_column(Float, default=0.0)
    start_date: Mapped[date] = mapped_column(Date)
    drift_factor: Mapped[float] = mapped_column(Float, default=1.0)
    note: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime)

    team: Mapped[Team] = relationship()

    __table_args__ = (
        UniqueConstraint("season_id", "competition_id", "team_id", name="uq_assumption_team"),
        CheckConstraint("change > -1", name="change_range"),
        CheckConstraint("drift_factor > 0", name="drift_factor_positive"),
    )

    def __repr__(self) -> str:
        return f"Assumption(team_id={self.team_id}, change={self.change:+.2f})"
