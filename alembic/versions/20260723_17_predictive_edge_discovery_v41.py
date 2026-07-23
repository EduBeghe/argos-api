"""Add regime-conditioned near-miss identity.

Revision ID: 20260723_17
Revises: 20260722_16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260723_17"
down_revision: str | None = "20260722_16"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    existing = {
        item["name"]
        for item in sa.inspect(op.get_bind()).get_columns("predictive_edge_near_misses")
    }
    columns = (
        sa.Column("kind", sa.String(16), nullable=False, server_default="actionable"),
        sa.Column("directional_predictor", sa.String(64)),
        sa.Column("context_predictor", sa.String(64)),
        sa.Column("context_regime", sa.String(32)),
    )
    for column in columns:
        if column.name not in existing:
            op.add_column("predictive_edge_near_misses", column)
    diagnostic_columns = {
        item["name"]
        for item in sa.inspect(op.get_bind()).get_columns(
            "predictive_edge_run_diagnostics"
        )
    }
    if "regime_conditioned_conclusion" not in diagnostic_columns:
        op.add_column(
            "predictive_edge_run_diagnostics",
            sa.Column("regime_conditioned_conclusion", sa.String(64)),
        )
    if "regime_conditioned_stage_counters" not in diagnostic_columns:
        op.add_column(
            "predictive_edge_run_diagnostics",
            sa.Column("regime_conditioned_stage_counters", postgresql.JSONB()),
        )
    hypothesis_columns = {
        item["name"]
        for item in sa.inspect(op.get_bind()).get_columns("strategy_hypotheses")
    }
    for column in (
        sa.Column("context_predictor_name", sa.String(64)),
        sa.Column("context_predictor_version", sa.Integer()),
        sa.Column("context_regime", sa.String(32)),
        sa.Column("context_lower_bound", sa.Numeric(38, 18)),
        sa.Column("context_upper_bound", sa.Numeric(38, 18)),
    ):
        if column.name not in hypothesis_columns:
            op.add_column("strategy_hypotheses", column)
    constraints = {
        item["name"]
        for item in sa.inspect(op.get_bind()).get_check_constraints(
            "predictive_edge_near_misses"
        )
    }
    if "ck_predictive_edge_near_misses_kind" not in constraints:
        op.create_check_constraint(
            "ck_predictive_edge_near_misses_kind",
            "predictive_edge_near_misses",
            "kind IN ('actionable', 'diagnostic')",
        )


def downgrade() -> None:
    op.drop_column("strategy_hypotheses", "context_upper_bound")
    op.drop_column("strategy_hypotheses", "context_lower_bound")
    op.drop_column("strategy_hypotheses", "context_regime")
    op.drop_column("strategy_hypotheses", "context_predictor_version")
    op.drop_column("strategy_hypotheses", "context_predictor_name")
    op.drop_column(
        "predictive_edge_run_diagnostics", "regime_conditioned_stage_counters"
    )
    op.drop_column("predictive_edge_run_diagnostics", "regime_conditioned_conclusion")
    op.drop_constraint(
        "ck_predictive_edge_near_misses_kind",
        "predictive_edge_near_misses",
        type_="check",
    )
    op.drop_column("predictive_edge_near_misses", "context_regime")
    op.drop_column("predictive_edge_near_misses", "context_predictor")
    op.drop_column("predictive_edge_near_misses", "directional_predictor")
    op.drop_column("predictive_edge_near_misses", "kind")
