"""Create relational edge-guided evaluation memory.

Revision ID: 20260721_13
Revises: 20260721_12
"""

from collections.abc import Sequence

from alembic import op
from app.optimization.edge_guided_memory import (
    EdgeGuidedCandidateRecord,
    EdgeGuidedRunRecord,
    EdgeGuidedWindowRecord,
)

revision: str = "20260721_13"
down_revision: str | None = "20260721_12"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = (
    EdgeGuidedRunRecord.__table__,
    EdgeGuidedCandidateRecord.__table__,
    EdgeGuidedWindowRecord.__table__,
)


def upgrade() -> None:
    bind = op.get_bind()
    for table in TABLES:
        table.create(bind)


def downgrade() -> None:
    bind = op.get_bind()
    for table in reversed(TABLES):
        table.drop(bind)
