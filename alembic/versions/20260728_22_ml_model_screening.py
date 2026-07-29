"""Add reproducible ML model screening persistence.

Revision ID: 20260728_22
Revises: 20260724_21
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260728_22"
down_revision = "20260724_21"
branch_labels = None
depends_on = None


def upgrade() -> None:
    metric = sa.Numeric(38, 18)
    op.create_table(
        "ml_model_screening_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("experiment_key", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("screening_version", sa.Integer(), nullable=False),
        sa.Column("experiment_contract_version", sa.Integer(), nullable=False),
        sa.Column("variable_registry_version", sa.Integer(), nullable=False),
        sa.Column("missing_data_policy_version", sa.Integer(), nullable=False),
        sa.Column("dataset_fingerprint", sa.String(64), nullable=False),
        sa.Column("asset", sa.String(32), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("market_type", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(16), nullable=False),
        sa.Column("requested_engines", postgresql.JSONB(), nullable=False),
        sa.Column("requested_future_durations", postgresql.JSONB(), nullable=False),
        sa.Column("library_versions", postgresql.JSONB(), nullable=False),
        sa.Column("deterministic_seed", sa.Integer(), nullable=False),
        sa.Column("configuration", postgresql.JSONB(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("failure_type", sa.String(128)),
        sa.Column("failure_message", sa.String(512)),
        sa.Column("evaluations_computed", sa.Integer(), nullable=False),
        sa.Column("evaluations_reused", sa.Integer(), nullable=False),
        sa.Column("results_generated", sa.Integer(), nullable=False),
        sa.Column("pareto_retained_count", sa.Integer(), nullable=False),
        sa.Column("elapsed_seconds", metric),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_ml_screening_run_status",
        ),
        sa.CheckConstraint(
            "screening_version > 0", name="ck_ml_screening_version_positive"
        ),
    )
    op.create_index(
        "ix_ml_screening_completed_key",
        "ml_model_screening_runs",
        ["experiment_key", "status"],
    )
    evaluation_columns = [
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_model_screening_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("engine", sa.String(16), nullable=False),
        sa.Column("future_duration_seconds", sa.Integer(), nullable=False),
        sa.Column("fold_name", sa.String(64), nullable=False),
        sa.Column("training_row_count", sa.Integer(), nullable=False),
        sa.Column("validation_row_count", sa.Integer(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
    ]
    for name in (
        "mean_absolute_error",
        "root_mean_squared_error",
        "mean_prediction",
        "mean_actual_return",
        "pearson_correlation",
        "spearman_correlation",
        "directional_accuracy",
        "zero_prediction_mae",
        "mae_improvement_over_zero",
        "elapsed_seconds",
    ):
        evaluation_columns.append(
            sa.Column(
                name,
                metric,
                nullable=name in {"pearson_correlation", "spearman_correlation"},
            )
        )
    evaluation_columns.extend(
        [
            sa.Column("positive_prediction_count", sa.Integer(), nullable=False),
            sa.Column("negative_prediction_count", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint(
                "run_id",
                "engine",
                "future_duration_seconds",
                "fold_name",
                name="uq_ml_screening_evaluation",
            ),
            sa.CheckConstraint(
                "sample_count > 0", name="ck_ml_screening_sample_positive"
            ),
        ]
    )
    op.create_table("ml_model_screening_evaluations", *evaluation_columns)
    result_columns = [
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_model_screening_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("engine", sa.String(16), nullable=False),
        sa.Column("future_duration_seconds", sa.Integer(), nullable=False),
        sa.Column("folds_evaluated", sa.Integer(), nullable=False),
        sa.Column("total_validation_samples", sa.Integer(), nullable=False),
    ]
    for name in (
        "median_mae_improvement_over_zero",
        "worst_fold_mae_improvement_over_zero",
        "median_directional_accuracy",
        "worst_fold_directional_accuracy",
        "median_pearson_correlation",
        "median_spearman_correlation",
        "total_training_time_seconds",
    ):
        result_columns.append(
            sa.Column(
                name,
                metric,
                nullable=name
                in {"median_pearson_correlation", "median_spearman_correlation"},
            )
        )
    result_columns.extend(
        [
            sa.Column("folds_beating_zero", sa.Integer(), nullable=False),
            sa.Column("folds_with_positive_spearman", sa.Integer(), nullable=False),
            sa.Column("pareto_retained", sa.Boolean(), nullable=False),
            sa.Column("display_rank", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint(
                "run_id",
                "engine",
                "future_duration_seconds",
                name="uq_ml_screening_result",
            ),
        ]
    )
    op.create_table("ml_model_screening_results", *result_columns)


def downgrade() -> None:
    op.drop_table("ml_model_screening_results")
    op.drop_table("ml_model_screening_evaluations")
    op.drop_index("ix_ml_screening_completed_key", table_name="ml_model_screening_runs")
    op.drop_table("ml_model_screening_runs")
