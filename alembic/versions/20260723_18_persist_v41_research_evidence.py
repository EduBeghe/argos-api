"""Persist complete regime-conditioned v4.1 research evidence.

Revision ID: 20260723_18
Revises: 20260723_17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260723_18"
down_revision: str | None = "20260723_17"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    diagnostic_columns = {
        item["name"]
        for item in sa.inspect(op.get_bind()).get_columns(
            "predictive_edge_run_diagnostics"
        )
    }
    if "regime_conditioned_summary" not in diagnostic_columns:
        op.add_column(
            "predictive_edge_run_diagnostics",
            sa.Column("regime_conditioned_summary", postgresql.JSONB()),
        )
    existing = {
        item["name"]
        for item in sa.inspect(op.get_bind()).get_columns("predictive_edge_near_misses")
    }
    columns = (
        sa.Column("category_rank", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("engine", sa.String(16), nullable=False, server_default="v4"),
        sa.Column(
            "engine_contract_version", sa.Integer(), nullable=False, server_default="4"
        ),
        sa.Column("directional_predictor_version", sa.Integer()),
        sa.Column("context_predictor_version", sa.Integer()),
        sa.Column("selected_bin", sa.Integer()),
        sa.Column("directional_lower_bound", sa.Numeric(38, 18)),
        sa.Column("directional_upper_bound", sa.Numeric(38, 18)),
        sa.Column("context_lower_bound", sa.Numeric(38, 18)),
        sa.Column("context_upper_bound", sa.Numeric(38, 18)),
        sa.Column("direction_mapping", sa.String(32)),
        sa.Column("suggested_action", sa.String(32)),
        sa.Column("training_evidence", postgresql.JSONB()),
        sa.Column("directional_baseline_evidence", postgresql.JSONB()),
        sa.Column("validation_evidence", postgresql.JSONB()),
        sa.Column("test_evidence", postgresql.JSONB()),
        sa.Column("context_improvement", sa.Numeric(38, 18)),
        sa.Column("expected_gross_movement", sa.Numeric(38, 18)),
        sa.Column("expected_aligned_movement", sa.Numeric(38, 18)),
        sa.Column("directional_hit_rate", sa.Numeric(38, 18)),
        sa.Column("benchmark", sa.Numeric(38, 18)),
        sa.Column("benchmark_improvement", sa.Numeric(38, 18)),
        sa.Column("observations", sa.Integer()),
        sa.Column("usable_window_count", sa.Integer()),
        sa.Column("confirmed_window_count", sa.Integer()),
    )
    for column in columns:
        if column.name not in existing:
            op.add_column("predictive_edge_near_misses", column)


def downgrade() -> None:
    for name in (
        "confirmed_window_count",
        "observations",
        "benchmark_improvement",
        "benchmark",
        "directional_hit_rate",
        "expected_aligned_movement",
        "expected_gross_movement",
        "usable_window_count",
        "context_improvement",
        "test_evidence",
        "validation_evidence",
        "directional_baseline_evidence",
        "training_evidence",
        "suggested_action",
        "direction_mapping",
        "context_upper_bound",
        "context_lower_bound",
        "directional_upper_bound",
        "directional_lower_bound",
        "selected_bin",
        "context_predictor_version",
        "directional_predictor_version",
        "engine_contract_version",
        "engine",
        "category_rank",
    ):
        op.drop_column("predictive_edge_near_misses", name)
    op.drop_column("predictive_edge_run_diagnostics", "regime_conditioned_summary")
