"""Add provenance for diagnostic economic qualifications.

Revision ID: 20260810_28
Revises: 20260807_27
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260810_28"
down_revision: str | None = "20260807_27"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "ml_economic_qualifications",
        sa.Column(
            "provenance",
            sa.String(48),
            nullable=False,
            server_default="automatic_development_candidate",
        ),
    )
    op.create_check_constraint(
        "ck_ml_economic_qualification_provenance",
        "ml_economic_qualifications",
        "provenance IN ('automatic_development_candidate',"
        "'diagnostic_stage3_near_miss')",
    )
    op.alter_column("ml_economic_qualifications", "provenance", server_default=None)


def downgrade() -> None:
    op.drop_constraint(
        "ck_ml_economic_qualification_provenance",
        "ml_economic_qualifications",
        type_="check",
    )
    op.drop_column("ml_economic_qualifications", "provenance")
