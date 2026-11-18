"""Add assumption: stored expert assumptions about teams, checked against later results.

Revision ID: 2e0300914857
Revises: 8b2ec7c857d1
Create Date: 2026-11-18 01:31:53.645935
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2e0300914857"
down_revision: str | Sequence[str] | None = "8b2ec7c857d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "assumption",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("season_id", sa.Integer(), nullable=False),
        sa.Column("competition_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("change", sa.Float(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("drift_factor", sa.Float(), nullable=False),
        sa.Column("note", sa.String(length=200), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("change > -1", name=op.f("ck_assumption_change_range")),
        sa.CheckConstraint("drift_factor > 0", name=op.f("ck_assumption_drift_factor_positive")),
        sa.ForeignKeyConstraint(
            ["competition_id"],
            ["competition.id"],
            name=op.f("fk_assumption_competition_id_competition"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["season_id"],
            ["season.id"],
            name=op.f("fk_assumption_season_id_season"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["team.id"], name=op.f("fk_assumption_team_id_team"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_assumption")),
        sa.UniqueConstraint("season_id", "competition_id", "team_id", name="uq_assumption_team"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("assumption")
