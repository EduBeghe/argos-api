"""Create strategy-refinement experiment memory.

Revision ID: 20260720_10
Revises: 20260719_09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260720_10"
down_revision: str | None = "20260719_09"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "strategy_refinement_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("experiment_key", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("asset", sa.String(32), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("market_type", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(16), nullable=False),
        sa.Column("dataset_fingerprint", sa.String(64), nullable=False),
        sa.Column("requested_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("requested_to", sa.DateTime(timezone=True), nullable=False),
        sa.Column("score_model_name", sa.String(64), nullable=False),
        sa.Column("score_model_version", sa.Integer(), nullable=False),
        sa.Column("strategy_name", sa.String(64), nullable=False),
        sa.Column("strategy_version", sa.Integer(), nullable=False),
        sa.Column("simulation_engine_version", sa.Integer(), nullable=False),
        sa.Column("initial_cash", sa.Numeric(38, 18), nullable=False),
        sa.Column("account_currency", sa.String(16), nullable=False),
        sa.Column("fee_rate", sa.Numeric(38, 18), nullable=False),
        sa.Column("slippage_rate", sa.Numeric(38, 18), nullable=False),
        sa.Column("decision_interval_seconds", sa.Integer(), nullable=False),
        sa.Column("training_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("training_to", sa.DateTime(timezone=True), nullable=False),
        sa.Column("validation_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("validation_to", sa.DateTime(timezone=True), nullable=False),
        sa.Column("test_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("test_to", sa.DateTime(timezone=True), nullable=False),
        sa.Column("refinement_budget", postgresql.JSONB(), nullable=False),
        sa.Column("planner_allocation", postgresql.JSONB(), nullable=False),
        sa.Column("search_planner_version", sa.Integer(), nullable=False),
        sa.Column("history_cutoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recommendation_status", sa.String(64)),
        sa.Column("recommended_candidate_id", sa.String(64)),
        sa.Column("recommendation_explanation", sa.Text()),
        sa.Column("counts", postgresql.JSONB()),
        sa.Column("search_plan", postgresql.JSONB()),
        sa.Column("result_payload", postgresql.JSONB()),
        sa.Column("elapsed_seconds", sa.Numeric(38, 18)),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("failed_at", sa.DateTime(timezone=True)),
        sa.Column("failure_reason", sa.String(512)),
        sa.Column("last_verified_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_strategy_refinement_runs_status",
        ),
        sa.CheckConstraint(
            "(status = 'running' AND completed_at IS NULL AND failed_at IS NULL) OR "
            "(status = 'completed' AND completed_at IS NOT NULL "
            "AND failed_at IS NULL) OR "
            "(status = 'failed' AND failed_at IS NOT NULL AND completed_at IS NULL)",
            name="ck_strategy_refinement_runs_lifecycle",
        ),
        sa.UniqueConstraint("experiment_key", name="uq_strategy_refinement_runs_key"),
    )
    op.create_index(
        "ix_strategy_refinement_runs_compatibility",
        "strategy_refinement_runs",
        [
            "status",
            "score_model_name",
            "score_model_version",
            "strategy_name",
            "strategy_version",
            "simulation_engine_version",
            "search_planner_version",
            "started_at",
        ],
    )
    op.create_table(
        "strategy_candidates",
        sa.Column("stable_candidate_id", sa.String(64), primary_key=True),
        sa.Column("score_model_name", sa.String(64), nullable=False),
        sa.Column("score_model_version", sa.Integer(), nullable=False),
        sa.Column("strategy_name", sa.String(64), nullable=False),
        sa.Column("strategy_version", sa.Integer(), nullable=False),
        sa.Column("score_parameters", postgresql.JSONB(), nullable=False),
        sa.Column("strategy_parameters", postgresql.JSONB(), nullable=False),
        sa.Column("canonical_parameters", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "strategy_candidate_evaluations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "refinement_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("strategy_refinement_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "stable_candidate_id",
            sa.String(64),
            sa.ForeignKey("strategy_candidates.stable_candidate_id"),
            nullable=False,
        ),
        sa.Column("phase", sa.String(32), nullable=False),
        sa.Column("segment", sa.String(16), nullable=False),
        sa.Column("disabled_signal", sa.String(32)),
        sa.Column("total_return", sa.Numeric(38, 18), nullable=False),
        sa.Column("absolute_pnl", sa.Numeric(38, 18), nullable=False),
        sa.Column("max_drawdown", sa.Numeric(38, 18), nullable=False),
        sa.Column("trade_count", sa.Integer(), nullable=False),
        sa.Column("fees_paid", sa.Numeric(38, 18), nullable=False),
        sa.Column("final_equity", sa.Numeric(38, 18), nullable=False),
        sa.Column("return_over_drawdown", sa.Numeric(38, 18), nullable=False),
        sa.Column("rank", sa.Integer()),
        sa.Column("eligible", sa.Boolean()),
        sa.Column("rejection_reason", sa.String(128)),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "phase IN ('score_screening', 'joint_training', 'validation', 'test', "
            "'controlled_ablation')",
            name="ck_strategy_candidate_evaluations_phase",
        ),
        sa.CheckConstraint(
            "segment IN ('screening', 'training', 'validation', 'test')",
            name="ck_strategy_candidate_evaluations_segment",
        ),
        sa.UniqueConstraint(
            "refinement_run_id",
            "stable_candidate_id",
            "phase",
            "segment",
            name="uq_strategy_candidate_evaluations_identity",
        ),
    )
    op.create_index(
        "ix_strategy_candidate_evaluations_profile",
        "strategy_candidate_evaluations",
        ["stable_candidate_id", "segment", "evaluated_at"],
    )
    op.create_table(
        "strategy_benchmark_evaluations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "refinement_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("strategy_refinement_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("benchmark", sa.String(32), nullable=False),
        sa.Column("segment", sa.String(16), nullable=False),
        sa.Column("total_return", sa.Numeric(38, 18), nullable=False),
        sa.Column("absolute_pnl", sa.Numeric(38, 18), nullable=False),
        sa.Column("max_drawdown", sa.Numeric(38, 18), nullable=False),
        sa.Column("trade_count", sa.Integer(), nullable=False),
        sa.Column("fees_paid", sa.Numeric(38, 18), nullable=False),
        sa.Column("final_equity", sa.Numeric(38, 18), nullable=False),
        sa.Column("return_over_drawdown", sa.Numeric(38, 18), nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "benchmark IN ('production_baseline', 'buy_and_hold')",
            name="ck_strategy_benchmark_evaluations_name",
        ),
        sa.UniqueConstraint(
            "refinement_run_id",
            "benchmark",
            "segment",
            name="uq_strategy_benchmark_evaluations_identity",
        ),
    )


def downgrade() -> None:
    op.drop_table("strategy_benchmark_evaluations")
    op.drop_index(
        "ix_strategy_candidate_evaluations_profile",
        table_name="strategy_candidate_evaluations",
    )
    op.drop_table("strategy_candidate_evaluations")
    op.drop_table("strategy_candidates")
    op.drop_index(
        "ix_strategy_refinement_runs_compatibility",
        table_name="strategy_refinement_runs",
    )
    op.drop_table("strategy_refinement_runs")
