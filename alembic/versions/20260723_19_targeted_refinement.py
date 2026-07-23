"""Add bounded targeted-refinement evidence.

Revision ID: 20260723_19
Revises: 20260723_18
"""

from collections.abc import Sequence

from alembic import op
from app.optimization.targeted_refinement_memory import (
    TargetedRefinementCandidateRecord,
    TargetedRefinementRunRecord,
)

revision: str = "20260723_19"
down_revision: str | None = "20260723_18"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    TargetedRefinementRunRecord.__table__.create(bind, checkfirst=True)
    TargetedRefinementCandidateRecord.__table__.create(bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    TargetedRefinementCandidateRecord.__table__.drop(bind, checkfirst=True)
    TargetedRefinementRunRecord.__table__.drop(bind, checkfirst=True)
