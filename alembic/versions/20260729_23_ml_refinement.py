"""Add parameterized ML refinement persistence.

Revision ID: 20260729_23
Revises: 20260728_22
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260729_23"
down_revision = "20260728_22"
branch_labels = None
depends_on = None


def upgrade() -> None:
    metric = sa.Numeric(38, 18)
    op.create_table(
        "ml_refinement_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("experiment_key", sa.String(64), nullable=False),
        sa.Column(
            "source_screening_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_model_screening_runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "source_screening_result_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_model_screening_results.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("contract_version", sa.Integer(), nullable=False),
        sa.Column("classification_version", sa.Integer(), nullable=False),
        sa.Column("dataset_fingerprint", sa.String(64), nullable=False),
        sa.Column("variable_registry_version", sa.Integer(), nullable=False),
        sa.Column("missing_data_policy_version", sa.Integer(), nullable=False),
        sa.Column("library_versions", postgresql.JSONB(), nullable=False),
        sa.Column("normalized_configuration", postgresql.JSONB(), nullable=False),
        sa.Column("notes", sa.String(1000)),
        sa.Column("maximum_trials", sa.Integer(), nullable=False),
        sa.Column("maximum_runtime_seconds", sa.Integer(), nullable=False),
        sa.Column("combinations_planned", sa.Integer(), nullable=False),
        sa.Column("evaluations_computed", sa.Integer(), nullable=False),
        sa.Column("evaluations_reused", sa.Integer(), nullable=False),
        sa.Column("evaluations_omitted", sa.Integer(), nullable=False),
        sa.Column("budget_exhausted", sa.Boolean(), nullable=False),
        sa.Column("termination_reason", sa.String(64)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("failed_at", sa.DateTime(timezone=True)),
        sa.Column("failure_type", sa.String(128)),
        sa.Column("failure_message", sa.String(512)),
        sa.Column("elapsed_seconds", metric),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('queued','running','completed','failed','cancelled')",
            name="ck_ml_refinement_run_status",
        ),
        sa.CheckConstraint(
            "(status = 'queued' AND started_at IS NULL "
            "AND completed_at IS NULL AND failed_at IS NULL) OR "
            "(status = 'running' AND started_at IS NOT NULL "
            "AND completed_at IS NULL AND failed_at IS NULL) OR "
            "(status = 'completed' AND started_at IS NOT NULL "
            "AND completed_at IS NOT NULL AND failed_at IS NULL) OR "
            "(status = 'failed' AND started_at IS NOT NULL "
            "AND completed_at IS NULL AND failed_at IS NOT NULL) OR "
            "(status = 'cancelled' AND started_at IS NOT NULL "
            "AND completed_at IS NOT NULL AND failed_at IS NULL)",
            name="ck_ml_refinement_run_lifecycle",
        ),
        sa.CheckConstraint(
            "maximum_trials > 0 AND maximum_runtime_seconds > 0 "
            "AND combinations_planned >= 0 AND evaluations_computed >= 0 "
            "AND evaluations_reused >= 0 AND evaluations_omitted >= 0",
            name="ck_ml_refinement_run_budgets",
        ),
        sa.UniqueConstraint("experiment_key", name="uq_ml_refinement_experiment_key"),
    )
    op.create_table(
        "ml_refinement_evaluations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_refinement_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "reused_from_evaluation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_refinement_evaluations.id", ondelete="SET NULL"),
        ),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("engine", sa.String(16), nullable=False),
        sa.Column("future_duration_seconds", sa.Integer(), nullable=False),
        sa.Column("objective", sa.String(64), nullable=False),
        sa.Column("hyperparameters", postgresql.JSONB(), nullable=False),
        sa.Column("feature_policy", postgresql.JSONB(), nullable=False),
        sa.Column("confidence_buckets", postgresql.JSONB(), nullable=False),
        sa.Column("fold_evidence", postgresql.JSONB(), nullable=False),
        sa.Column("aggregate_metrics", postgresql.JSONB(), nullable=False),
        sa.Column("classification", sa.String(64), nullable=False),
        sa.Column("classification_version", sa.Integer(), nullable=False),
        sa.Column("failed_rules", postgresql.JSONB(), nullable=False),
        sa.Column("elapsed_seconds", metric, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "run_id", "fingerprint", name="uq_ml_refinement_evaluation"
        ),
    )
    op.create_index(
        "ix_ml_refinement_evaluation_fingerprint",
        "ml_refinement_evaluations",
        ["fingerprint"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ml_refinement_evaluation_fingerprint",
        table_name="ml_refinement_evaluations",
    )
    op.drop_table("ml_refinement_evaluations")
    op.drop_table("ml_refinement_runs")
