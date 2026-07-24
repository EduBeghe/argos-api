"""Add progressive composite-feature search evidence.

Revision ID: 20260724_20
Revises: 20260723_19
"""

from collections.abc import Sequence

from alembic import op
from app.optimization.composite_search_memory import (
    CompositeSearchCandidateRecord,
    CompositeSearchRunRecord,
)

revision: str = "20260724_20"
down_revision: str | None = "20260723_19"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    CompositeSearchRunRecord.__table__.create(bind, checkfirst=True)
    CompositeSearchCandidateRecord.__table__.create(bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    CompositeSearchCandidateRecord.__table__.drop(bind, checkfirst=True)
    CompositeSearchRunRecord.__table__.drop(bind, checkfirst=True)
