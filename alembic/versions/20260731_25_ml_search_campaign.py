"""Add durable exhaustive ML search campaigns.

Revision ID: 20260731_25
Revises: 20260729_24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260731_25"
down_revision: str | None = "20260729_24"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ml_search_campaigns",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("experiment_key", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("outcome", sa.String(64)),
        sa.Column("contract_version", sa.Integer(), nullable=False),
        sa.Column("policy_version", sa.Integer(), nullable=False),
        sa.Column("classification_version", sa.Integer(), nullable=False),
        sa.Column("normalized_request", postgresql.JSONB(), nullable=False),
        sa.Column("maximum_evaluations", sa.Integer(), nullable=False),
        sa.Column("maximum_runtime_seconds", sa.Integer(), nullable=False),
        sa.Column("families_total", sa.Integer(), nullable=False),
        sa.Column("families_covered", sa.Integer(), nullable=False),
        sa.Column("families_uncovered", sa.Integer(), nullable=False),
        sa.Column("evaluations_computed", sa.Integer(), nullable=False),
        sa.Column("evaluations_reused", sa.Integer(), nullable=False),
        sa.Column("current_stage", sa.Integer(), nullable=False),
        sa.Column("cancellation_requested_at", sa.DateTime(timezone=True)),
        sa.Column("first_claimed_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("failed_at", sa.DateTime(timezone=True)),
        sa.Column("failure_type", sa.String(128)),
        sa.Column("failure_message", sa.String(512)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('queued','running','completed','failed','cancelled')",
            name="ck_ml_search_campaign_status",
        ),
        sa.CheckConstraint(
            "maximum_evaluations > 0 AND maximum_runtime_seconds > 0",
            name="ck_ml_search_campaign_limits",
        ),
        sa.CheckConstraint(
            "families_total >= 0 AND families_covered >= 0 "
            "AND families_uncovered >= 0 AND evaluations_computed >= 0 "
            "AND evaluations_reused >= 0",
            name="ck_ml_search_campaign_counters",
        ),
        sa.CheckConstraint(
            "families_covered + families_uncovered = families_total",
            name="ck_ml_search_campaign_coverage",
        ),
        sa.CheckConstraint(
            "outcome IS NULL OR outcome IN ('coverage_incomplete',"
            "'no_strong_candidate','search_budget_exhausted',"
            "'strong_candidates_found')",
            name="ck_ml_search_campaign_outcome",
        ),
        sa.UniqueConstraint(
            "experiment_key", name="uq_ml_search_campaign_experiment_key"
        ),
    )
    op.create_table(
        "ml_search_campaign_sources",
        sa.Column(
            "campaign_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_search_campaigns.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "source_screening_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_model_screening_runs.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column("dataset_fingerprint", sa.String(64), nullable=False),
    )
    op.create_table(
        "ml_search_campaign_families",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "campaign_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_search_campaigns.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_screening_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_model_screening_runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("dataset_fingerprint", sa.String(64), nullable=False),
        sa.Column("asset", sa.String(32), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("market_type", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(16), nullable=False),
        sa.Column("engine", sa.String(16), nullable=False),
        sa.Column("future_duration_seconds", sa.Integer(), nullable=False),
        sa.Column("objective", sa.String(64), nullable=False),
        sa.Column("feature_variant", sa.String(16), nullable=False),
        sa.Column("state", sa.String(64), nullable=False),
        sa.Column("current_stage", sa.Integer(), nullable=False),
        sa.Column("evaluations_planned", sa.Integer(), nullable=False),
        sa.Column("evaluations_completed", sa.Integer(), nullable=False),
        sa.Column("evaluations_reused", sa.Integer(), nullable=False),
        sa.Column(
            "best_evaluation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_refinement_evaluations.id", ondelete="SET NULL"),
        ),
        sa.Column("best_metrics", postgresql.JSONB()),
        sa.Column("decision_reason", sa.String(128)),
        sa.Column("deterministic_ordinal", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "current_stage BETWEEN 0 AND 4", name="ck_ml_search_family_stage"
        ),
        sa.CheckConstraint(
            "state IN ('pending','scheduled','survived','discarded',"
            "'strong_development_candidate')",
            name="ck_ml_search_family_state",
        ),
        sa.UniqueConstraint(
            "campaign_id", "fingerprint", name="uq_ml_search_campaign_family"
        ),
    )
    op.create_index(
        "ix_ml_search_family_order",
        "ml_search_campaign_families",
        ["campaign_id", "current_stage", "state", "deterministic_ordinal"],
    )
    op.create_table(
        "ml_search_campaign_children",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "campaign_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_search_campaigns.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "family_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_search_campaign_families.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "refinement_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_refinement_runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("stage", sa.Integer(), nullable=False),
        sa.Column("reserved_evaluations", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "campaign_id",
            "family_id",
            "stage",
            name="uq_ml_search_campaign_child_stage",
        ),
        sa.UniqueConstraint(
            "refinement_run_id", name="uq_ml_search_campaign_child_run"
        ),
        sa.CheckConstraint(
            "stage BETWEEN 1 AND 4 AND reserved_evaluations > 0",
            name="ck_ml_search_campaign_child_stage",
        ),
    )


def downgrade() -> None:
    op.drop_table("ml_search_campaign_children")
    op.drop_index("ix_ml_search_family_order", table_name="ml_search_campaign_families")
    op.drop_table("ml_search_campaign_families")
    op.drop_table("ml_search_campaign_sources")
    op.drop_table("ml_search_campaigns")
