"""Create walk-forward evaluation memory.

Revision ID: 20260721_11
Revises: 20260720_10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260721_11"
down_revision: str | None = "20260720_10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MONEY = sa.Numeric(38, 18)


def upgrade() -> None:
    op.create_table(
        "walk_forward_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("experiment_key", sa.String(64), nullable=False),
        sa.Column(
            "refinement_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("strategy_refinement_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("algorithm_version", sa.Integer(), nullable=False),
        sa.Column("dataset_fingerprint", sa.String(64), nullable=False),
        sa.Column("training_days", sa.Integer(), nullable=False),
        sa.Column("validation_days", sa.Integer(), nullable=False),
        sa.Column("test_days", sa.Integer(), nullable=False),
        sa.Column("step_days", sa.Integer(), nullable=False),
        sa.Column("maximum_windows", sa.Integer(), nullable=False),
        sa.Column("maximum_candidates", sa.Integer(), nullable=False),
        sa.Column("minimum_valid_windows", sa.Integer(), nullable=False),
        sa.Column("minimum_profitable_window_ratio", MONEY, nullable=False),
        sa.Column("maximum_drawdown", MONEY, nullable=False),
        sa.Column("maximum_return_standard_deviation", MONEY, nullable=False),
        sa.Column("initial_cash", MONEY, nullable=False),
        sa.Column("fee_rate", MONEY, nullable=False),
        sa.Column("slippage_rate", MONEY, nullable=False),
        sa.Column("decision_interval_seconds", sa.Integer(), nullable=False),
        sa.Column("windows_generated", sa.Integer(), nullable=False),
        sa.Column("windows_completed", sa.Integer(), nullable=False),
        sa.Column("candidates_evaluated", sa.Integer(), nullable=False),
        sa.Column("evaluations_computed", sa.Integer(), nullable=False),
        sa.Column("evaluations_reused", sa.Integer(), nullable=False),
        sa.Column("compute_avoided_estimate", sa.Integer(), nullable=False),
        sa.Column(
            "recommended_candidate_id",
            sa.String(64),
            sa.ForeignKey("strategy_candidates.stable_candidate_id"),
        ),
        sa.Column("recommendation_explanation", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("elapsed_seconds", MONEY, nullable=False),
        sa.CheckConstraint(
            "status IN ('completed', 'insufficient_history')",
            name="ck_walk_forward_runs_status",
        ),
        sa.UniqueConstraint(
            "experiment_key", name="uq_walk_forward_runs_experiment_key"
        ),
    )
    op.create_index(
        "ix_walk_forward_runs_refinement_started",
        "walk_forward_runs",
        ["refinement_run_id", "started_at"],
    )
    op.create_table(
        "walk_forward_windows",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "walk_forward_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("walk_forward_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("training_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("training_to", sa.DateTime(timezone=True), nullable=False),
        sa.Column("validation_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("validation_to", sa.DateTime(timezone=True), nullable=False),
        sa.Column("test_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("test_to", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decision_count", sa.Integer(), nullable=False),
        sa.Column("dataset_fingerprint", sa.String(64), nullable=False),
        sa.CheckConstraint("ordinal > 0", name="ck_walk_forward_windows_ordinal"),
        sa.CheckConstraint(
            "training_from < training_to AND training_to = validation_from "
            "AND validation_from < validation_to AND validation_to = test_from "
            "AND test_from < test_to",
            name="ck_walk_forward_windows_chronology",
        ),
        sa.UniqueConstraint(
            "walk_forward_run_id",
            "ordinal",
            name="uq_walk_forward_windows_ordinal",
        ),
    )
    op.create_table(
        "walk_forward_candidate_evaluations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "walk_forward_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("walk_forward_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "window_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("walk_forward_windows.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "stable_candidate_id",
            sa.String(64),
            sa.ForeignKey("strategy_candidates.stable_candidate_id"),
            nullable=False,
        ),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("total_return", MONEY),
        sa.Column("absolute_pnl", MONEY),
        sa.Column("max_drawdown", MONEY),
        sa.Column("final_equity", MONEY),
        sa.Column("fees_paid", MONEY),
        sa.Column("trade_count", sa.Integer()),
        sa.Column("return_over_drawdown", MONEY),
        sa.Column("failure_reason", sa.String(128)),
        sa.CheckConstraint(
            "status IN ('completed', 'failed')",
            name="ck_walk_forward_candidate_evaluations_status",
        ),
        sa.CheckConstraint(
            "(status = 'completed' AND total_return IS NOT NULL "
            "AND absolute_pnl IS NOT NULL AND max_drawdown IS NOT NULL "
            "AND final_equity IS NOT NULL AND fees_paid IS NOT NULL "
            "AND trade_count IS NOT NULL AND return_over_drawdown IS NOT NULL "
            "AND failure_reason IS NULL) OR "
            "(status = 'failed' AND total_return IS NULL "
            "AND absolute_pnl IS NULL AND max_drawdown IS NULL "
            "AND final_equity IS NULL AND fees_paid IS NULL "
            "AND trade_count IS NULL AND return_over_drawdown IS NULL "
            "AND failure_reason IS NOT NULL)",
            name="ck_walk_forward_candidate_evaluations_lifecycle",
        ),
        sa.UniqueConstraint(
            "walk_forward_run_id",
            "window_id",
            "stable_candidate_id",
            name="uq_walk_forward_candidate_evaluations_identity",
        ),
    )
    op.create_index(
        "ix_walk_forward_candidate_evaluations_page",
        "walk_forward_candidate_evaluations",
        ["walk_forward_run_id", "stable_candidate_id", "window_id"],
    )
    op.create_table(
        "walk_forward_benchmark_evaluations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "walk_forward_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("walk_forward_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "window_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("walk_forward_windows.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("benchmark", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("total_return", MONEY),
        sa.Column("absolute_pnl", MONEY),
        sa.Column("max_drawdown", MONEY),
        sa.Column("final_equity", MONEY),
        sa.Column("fees_paid", MONEY),
        sa.Column("trade_count", sa.Integer()),
        sa.Column("return_over_drawdown", MONEY),
        sa.Column("failure_reason", sa.String(128)),
        sa.CheckConstraint(
            "benchmark IN ('production_baseline', 'buy_and_hold')",
            name="ck_walk_forward_benchmark_evaluations_name",
        ),
        sa.CheckConstraint(
            "status IN ('completed', 'failed')",
            name="ck_walk_forward_benchmark_evaluations_status",
        ),
        sa.CheckConstraint(
            "(status = 'completed' AND total_return IS NOT NULL "
            "AND absolute_pnl IS NOT NULL AND max_drawdown IS NOT NULL "
            "AND final_equity IS NOT NULL AND fees_paid IS NOT NULL "
            "AND trade_count IS NOT NULL AND return_over_drawdown IS NOT NULL "
            "AND failure_reason IS NULL) OR "
            "(status = 'failed' AND total_return IS NULL "
            "AND absolute_pnl IS NULL AND max_drawdown IS NULL "
            "AND final_equity IS NULL AND fees_paid IS NULL "
            "AND trade_count IS NULL AND return_over_drawdown IS NULL "
            "AND failure_reason IS NOT NULL)",
            name="ck_walk_forward_benchmark_evaluations_lifecycle",
        ),
        sa.UniqueConstraint(
            "walk_forward_run_id",
            "window_id",
            "benchmark",
            name="uq_walk_forward_benchmark_evaluations_identity",
        ),
    )
    op.create_table(
        "walk_forward_candidate_stability",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "walk_forward_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("walk_forward_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "stable_candidate_id",
            sa.String(64),
            sa.ForeignKey("strategy_candidates.stable_candidate_id"),
            nullable=False,
        ),
        sa.Column("ranking", sa.Integer(), nullable=False),
        sa.Column("classification", sa.String(32), nullable=False),
        sa.Column("windows_evaluated", sa.Integer(), nullable=False),
        sa.Column("profitable_windows", sa.Integer(), nullable=False),
        sa.Column("losing_windows", sa.Integer(), nullable=False),
        sa.Column("profitable_window_ratio", MONEY, nullable=False),
        sa.Column("mean_return", MONEY, nullable=False),
        sa.Column("median_return", MONEY, nullable=False),
        sa.Column("worst_window_return", MONEY, nullable=False),
        sa.Column("best_window_return", MONEY, nullable=False),
        sa.Column("return_standard_deviation", MONEY, nullable=False),
        sa.Column("mean_max_drawdown", MONEY, nullable=False),
        sa.Column("worst_max_drawdown", MONEY, nullable=False),
        sa.Column("aggregate_fees_paid", MONEY, nullable=False),
        sa.Column("aggregate_trade_count", sa.Integer(), nullable=False),
        sa.Column("positive_excess_return_vs_production_ratio", MONEY, nullable=False),
        sa.Column(
            "positive_excess_return_vs_buy_and_hold_ratio", MONEY, nullable=False
        ),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "classification IN ('stable_candidate', 'unstable_candidate', "
            "'consistently_unprofitable', 'insufficient_evidence', "
            "'evaluation_failed')",
            name="ck_walk_forward_candidate_stability_classification",
        ),
        sa.UniqueConstraint(
            "walk_forward_run_id",
            "stable_candidate_id",
            name="uq_walk_forward_candidate_stability_identity",
        ),
    )
    op.create_index(
        "ix_walk_forward_candidate_stability_classification",
        "walk_forward_candidate_stability",
        ["walk_forward_run_id", "classification", "stable_candidate_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_walk_forward_candidate_stability_classification",
        table_name="walk_forward_candidate_stability",
    )
    op.drop_table("walk_forward_candidate_stability")
    op.drop_table("walk_forward_benchmark_evaluations")
    op.drop_index(
        "ix_walk_forward_candidate_evaluations_page",
        table_name="walk_forward_candidate_evaluations",
    )
    op.drop_table("walk_forward_candidate_evaluations")
    op.drop_table("walk_forward_windows")
    op.drop_index(
        "ix_walk_forward_runs_refinement_started",
        table_name="walk_forward_runs",
    )
    op.drop_table("walk_forward_runs")
