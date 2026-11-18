"""Add point_adjustment: federation point deductions (MLSZ "büntetőpont").

Revision ID: 8b2ec7c857d1
Revises: d78e82a12ecc
Create Date: 2026-11-18 00:38:29.552753
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8b2ec7c857d1"
down_revision: str | Sequence[str] | None = "d78e82a12ecc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "point_adjustment",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("season_id", sa.Integer(), nullable=False),
        sa.Column("competition_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("points", sa.Integer(), nullable=False),
        sa.Column("reason", sa.String(length=200), nullable=True),
        sa.CheckConstraint("points <> 0", name=op.f("ck_point_adjustment_non_zero")),
        sa.ForeignKeyConstraint(
            ["competition_id"],
            ["competition.id"],
            name=op.f("fk_point_adjustment_competition_id_competition"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["season_id"],
            ["season.id"],
            name=op.f("fk_point_adjustment_season_id_season"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"],
            ["team.id"],
            name=op.f("fk_point_adjustment_team_id_team"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_point_adjustment")),
        sa.UniqueConstraint(
            "season_id", "competition_id", "team_id", name="uq_point_adjustment_team"
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("point_adjustment")
