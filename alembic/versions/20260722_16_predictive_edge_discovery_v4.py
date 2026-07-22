"""Add bounded Predictive Edge Discovery v4 evidence.

Revision ID: 20260722_16
Revises: 20260722_15
"""

from collections.abc import Sequence

from alembic import op
from app.optimization.predictive_edge_memory import (
    PredictiveEdgeNearMissRecord,
    PredictiveEdgeRunDiagnosticRecord,
    PredictiveEdgeSensitivityRecord,
    PredictiveEdgeV4CacheRecord,
)

revision: str = "20260722_16"
down_revision: str | None = "20260722_15"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = (
    PredictiveEdgeRunDiagnosticRecord.__table__,
    PredictiveEdgeNearMissRecord.__table__,
    PredictiveEdgeSensitivityRecord.__table__,
    PredictiveEdgeV4CacheRecord.__table__,
)


def upgrade() -> None:
    bind = op.get_bind()
    for table in TABLES:
        table.create(bind)


def downgrade() -> None:
    bind = op.get_bind()
    for table in reversed(TABLES):
        table.drop(bind)
