"""Create relational Predictive Edge Map memory.

Revision ID: 20260721_12
Revises: 20260721_11
"""

from collections.abc import Sequence

from alembic import op
from app.optimization.predictive_edge_memory import (
    PredictiveEdgeAggregateRecord,
    PredictiveEdgeBenchmarkRecord,
    PredictiveEdgeBinAggregateRecord,
    PredictiveEdgeRunRecord,
    PredictiveEdgeWindowMetricRecord,
    PredictiveEdgeWindowRecord,
    StrategyHypothesisRecord,
)

revision: str = "20260721_12"
down_revision: str | None = "20260721_11"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = (
    PredictiveEdgeRunRecord.__table__,
    PredictiveEdgeWindowRecord.__table__,
    PredictiveEdgeAggregateRecord.__table__,
    PredictiveEdgeWindowMetricRecord.__table__,
    PredictiveEdgeBinAggregateRecord.__table__,
    StrategyHypothesisRecord.__table__,
    PredictiveEdgeBenchmarkRecord.__table__,
)


def upgrade() -> None:
    bind = op.get_bind()
    for table in TABLES:
        table.create(bind)


def downgrade() -> None:
    bind = op.get_bind()
    for table in reversed(TABLES):
        table.drop(bind)
