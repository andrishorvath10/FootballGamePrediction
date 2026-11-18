"""Initial schema: everything that existed before migrations were introduced.

Revision ID: d78e82a12ecc
Revises:
Create Date: 2026-11-17 20:10:57.087811
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d78e82a12ecc"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "competition",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("region", sa.String(length=100), nullable=True),
        sa.Column("level", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_competition")),
        sa.UniqueConstraint("name", name=op.f("uq_competition_name")),
    )
    op.create_table(
        "player",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("external_id", sa.String(length=50), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_player")),
        sa.UniqueConstraint("external_id", name=op.f("uq_player_external_id")),
    )
    op.create_index(op.f("ix_player_name"), "player", ["name"], unique=False)

    op.create_table(
        "season",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.CheckConstraint(
            "start_date IS NULL OR end_date IS NULL OR start_date <= end_date",
            name=op.f("ck_season_date_order"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_season")),
        sa.UniqueConstraint("name", name=op.f("uq_season_name")),
    )
    op.create_table(
        "team",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("short_name", sa.String(length=20), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_team")),
        sa.UniqueConstraint("name", name=op.f("uq_team_name")),
    )
    op.create_table(
        "match",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("season_id", sa.Integer(), nullable=False),
        sa.Column("competition_id", sa.Integer(), nullable=False),
        sa.Column("round", sa.Integer(), nullable=False),
        sa.Column("match_date", sa.Date(), nullable=True),
        sa.Column("home_team_id", sa.Integer(), nullable=False),
        sa.Column("away_team_id", sa.Integer(), nullable=False),
        sa.Column("home_goals", sa.Integer(), nullable=True),
        sa.Column("away_goals", sa.Integer(), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "scheduled",
                "played",
                "postponed",
                "cancelled",
                name="match_status",
                native_enum=False,
                create_constraint=True,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("source_url", sa.String(length=500), nullable=True),
        sa.Column("external_id", sa.String(length=50), nullable=True),
        sa.CheckConstraint(
            "status <> 'played' OR home_goals IS NOT NULL", name=op.f("ck_match_played_has_score")
        ),
        sa.CheckConstraint(
            "(home_goals IS NULL OR home_goals >= 0) AND (away_goals IS NULL OR away_goals >= 0)",
            name=op.f("ck_match_goals_non_negative"),
        ),
        sa.CheckConstraint(
            "(home_goals IS NULL) = (away_goals IS NULL)", name=op.f("ck_match_score_complete")
        ),
        sa.CheckConstraint("home_team_id <> away_team_id", name=op.f("ck_match_different_teams")),
        sa.CheckConstraint("round >= 1", name=op.f("ck_match_round_positive")),
        sa.ForeignKeyConstraint(
            ["away_team_id"], ["team.id"], name=op.f("fk_match_away_team_id_team")
        ),
        sa.ForeignKeyConstraint(
            ["competition_id"],
            ["competition.id"],
            name=op.f("fk_match_competition_id_competition"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["home_team_id"], ["team.id"], name=op.f("fk_match_home_team_id_team")
        ),
        sa.ForeignKeyConstraint(
            ["season_id"], ["season.id"], name=op.f("fk_match_season_id_season"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_match")),
        sa.UniqueConstraint("external_id", name=op.f("uq_match_external_id")),
        sa.UniqueConstraint(
            "season_id",
            "competition_id",
            "round",
            "home_team_id",
            "away_team_id",
            name="uq_match_fixture",
        ),
    )
    op.create_index("ix_match_away_team", "match", ["away_team_id"], unique=False)
    op.create_index(
        "ix_match_competition_season", "match", ["competition_id", "season_id"], unique=False
    )
    op.create_index("ix_match_date", "match", ["match_date"], unique=False)
    op.create_index("ix_match_home_team", "match", ["home_team_id"], unique=False)
    op.create_index("ix_match_season_status", "match", ["season_id", "status"], unique=False)

    op.create_table(
        "player_team",
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("season_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["player_id"],
            ["player.id"],
            name=op.f("fk_player_team_player_id_player"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["season_id"],
            ["season.id"],
            name=op.f("fk_player_team_season_id_season"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["team.id"], name=op.f("fk_player_team_team_id_team"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("player_id", "team_id", "season_id", name=op.f("pk_player_team")),
    )
    op.create_index(
        "ix_player_team_team_season", "player_team", ["team_id", "season_id"], unique=False
    )

    op.create_table(
        "team_alias",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("alias", sa.String(length=100), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["team_id"], ["team.id"], name=op.f("fk_team_alias_team_id_team"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_team_alias")),
        sa.UniqueConstraint("alias", name=op.f("uq_team_alias_alias")),
    )
    op.create_index(op.f("ix_team_alias_team_id"), "team_alias", ["team_id"], unique=False)

    op.create_table(
        "appearance",
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("minutes_played", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "minutes_played IS NULL OR (minutes_played BETWEEN 0 AND 130)",
            name=op.f("ck_appearance_minutes_range"),
        ),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["match.id"],
            name=op.f("fk_appearance_match_id_match"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["player_id"],
            ["player.id"],
            name=op.f("fk_appearance_player_id_player"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["team_id"], ["team.id"], name=op.f("fk_appearance_team_id_team")),
        sa.PrimaryKeyConstraint("match_id", "player_id", name=op.f("pk_appearance")),
    )
    op.create_index("ix_appearance_player", "appearance", ["player_id"], unique=False)

    op.create_table(
        "goal",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=True),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("minute", sa.Integer(), nullable=True),
        sa.Column("is_penalty", sa.Boolean(), nullable=False),
        sa.Column("is_own_goal", sa.Boolean(), nullable=False),
        sa.CheckConstraint(
            "NOT (is_penalty AND is_own_goal)", name=op.f("ck_goal_penalty_not_own_goal")
        ),
        sa.CheckConstraint(
            "minute IS NULL OR (minute BETWEEN 1 AND 130)", name=op.f("ck_goal_minute_range")
        ),
        sa.ForeignKeyConstraint(
            ["match_id"], ["match.id"], name=op.f("fk_goal_match_id_match"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["player_id"], ["player.id"], name=op.f("fk_goal_player_id_player"), ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["team_id"], ["team.id"], name=op.f("fk_goal_team_id_team")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_goal")),
    )
    op.create_index("ix_goal_match", "goal", ["match_id"], unique=False)
    op.create_index("ix_goal_player", "goal", ["player_id"], unique=False)
    op.create_index("ix_goal_team", "goal", ["team_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_goal_team", table_name="goal")
    op.drop_index("ix_goal_player", table_name="goal")
    op.drop_index("ix_goal_match", table_name="goal")

    op.drop_table("goal")
    op.drop_index("ix_appearance_player", table_name="appearance")

    op.drop_table("appearance")
    op.drop_index(op.f("ix_team_alias_team_id"), table_name="team_alias")

    op.drop_table("team_alias")
    op.drop_index("ix_player_team_team_season", table_name="player_team")

    op.drop_table("player_team")
    op.drop_index("ix_match_season_status", table_name="match")
    op.drop_index("ix_match_home_team", table_name="match")
    op.drop_index("ix_match_date", table_name="match")
    op.drop_index("ix_match_competition_season", table_name="match")
    op.drop_index("ix_match_away_team", table_name="match")

    op.drop_table("match")
    op.drop_table("team")
    op.drop_table("season")
    op.drop_index(op.f("ix_player_name"), table_name="player")

    op.drop_table("player")
    op.drop_table("competition")
