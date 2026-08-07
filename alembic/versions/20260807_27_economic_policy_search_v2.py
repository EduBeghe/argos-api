"""Add economic policy search v2 identities and qualifications.

Revision ID: 20260807_27
Revises: 20260804_26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260807_27"
down_revision: str | None = "20260804_26"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "ml_search_campaigns",
        sa.Column(
            "preset",
            sa.String(64),
            nullable=False,
            server_default="exhaustive_ohlcv_v1",
        ),
    )
    op.add_column(
        "ml_search_campaign_families",
        sa.Column(
            "preset",
            sa.String(64),
            nullable=False,
            server_default="exhaustive_ohlcv_v1",
        ),
    )
    op.add_column(
        "ml_search_campaign_families",
        sa.Column(
            "search_track",
            sa.String(32),
            nullable=False,
            server_default="exhaustive",
        ),
    )
    op.add_column(
        "ml_search_campaign_families",
        sa.Column("seed_family_id", postgresql.UUID(as_uuid=True)),
    )
    op.add_column(
        "ml_search_campaign_families",
        sa.Column(
            "seed_provenance",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "ml_search_campaign_families",
        sa.Column("selected_policy", postgresql.JSONB()),
    )
    op.add_column(
        "ml_search_campaign_families",
        sa.Column("selected_policy_fingerprint", sa.String(64)),
    )
    op.create_foreign_key(
        "fk_ml_search_family_seed",
        "ml_search_campaign_families",
        "ml_search_campaign_families",
        ["seed_family_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "ix_ml_search_campaign_preset",
        "ml_search_campaigns",
        ["preset", "created_at"],
    )
    op.create_index(
        "ix_ml_search_family_v2_identity",
        "ml_search_campaign_families",
        [
            "campaign_id",
            "preset",
            "search_track",
            "seed_family_id",
            "selected_policy_fingerprint",
        ],
    )
    op.drop_constraint(
        "ck_ml_search_campaign_outcome", "ml_search_campaigns", type_="check"
    )
    op.create_check_constraint(
        "ck_ml_search_campaign_outcome",
        "ml_search_campaigns",
        "outcome IS NULL OR outcome IN ('coverage_incomplete',"
        "'no_strong_candidate','search_budget_exhausted',"
        "'strong_candidates_found','qualified_candidates_found',"
        "'no_qualified_candidate')",
    )
    op.create_table(
        "ml_economic_qualifications",
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
        sa.Column("candidate_fingerprint", sa.String(64), nullable=False),
        sa.Column("policy_fingerprint", sa.String(64), nullable=False),
        sa.Column("qualification_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("classification", sa.String(16)),
        sa.Column("engine", sa.String(16), nullable=False),
        sa.Column("objective", sa.String(64), nullable=False),
        sa.Column("search_track", sa.String(32), nullable=False),
        sa.Column("seed_family_id", postgresql.UUID(as_uuid=True)),
        sa.Column("frozen_inputs", postgresql.JSONB(), nullable=False),
        sa.Column("temporal_manifest", postgresql.JSONB(), nullable=False),
        sa.Column("temporal_manifest_fingerprint", sa.String(64), nullable=False),
        sa.Column("policy_identity", postgresql.JSONB(), nullable=False),
        sa.Column("scenario_metrics", postgresql.JSONB()),
        sa.Column("quarter_returns", postgresql.JSONB()),
        sa.Column("trades", postgresql.JSONB()),
        sa.Column("failed_rules", postgresql.JSONB()),
        sa.Column("evidence_end_exclusive", sa.DateTime(timezone=True), nullable=False),
        sa.Column("worker_id", sa.String(128)),
        sa.Column("claimed_at", sa.DateTime(timezone=True)),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True)),
        sa.Column("cancellation_requested_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("failed_at", sa.DateTime(timezone=True)),
        sa.Column("failure_type", sa.String(128)),
        sa.Column("failure_message", sa.String(512)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "candidate_fingerprint",
            "qualification_version",
            name="uq_ml_economic_qualification_candidate",
        ),
        sa.CheckConstraint(
            "status IN ('queued','running','completed','failed','cancelled')",
            name="ck_ml_economic_qualification_status",
        ),
        sa.CheckConstraint(
            "classification IS NULL OR classification IN ('qualified','rejected')",
            name="ck_ml_economic_qualification_classification",
        ),
        sa.CheckConstraint(
            "(status = 'queued' AND classification IS NULL AND worker_id IS NULL "
            "AND completed_at IS NULL AND failed_at IS NULL) OR "
            "(status = 'running' AND classification IS NULL AND worker_id IS NOT NULL "
            "AND claimed_at IS NOT NULL AND heartbeat_at IS NOT NULL "
            "AND completed_at IS NULL AND failed_at IS NULL) OR "
            "(status = 'completed' AND classification IS NOT NULL "
            "AND scenario_metrics IS NOT NULL AND quarter_returns IS NOT NULL "
            "AND failed_rules IS NOT NULL AND completed_at IS NOT NULL "
            "AND worker_id IS NULL AND failed_at IS NULL) OR "
            "(status = 'failed' AND classification IS NULL AND failed_at IS NOT NULL "
            "AND failure_type IS NOT NULL AND failure_message IS NOT NULL "
            "AND worker_id IS NULL) OR "
            "(status = 'cancelled' AND classification IS NULL "
            "AND completed_at IS NOT NULL AND worker_id IS NULL)",
            name="ck_ml_economic_qualification_lifecycle",
        ),
        sa.CheckConstraint(
            "evidence_end_exclusive = TIMESTAMPTZ '2025-01-01 00:00:00+00'",
            name="ck_ml_economic_qualification_boundary",
        ),
    )
    op.create_index(
        "ix_ml_economic_qualification_list",
        "ml_economic_qualifications",
        ["campaign_id", "classification", "engine", "objective", "search_track"],
    )
    op.create_index(
        "ix_ml_economic_qualification_claim",
        "ml_economic_qualifications",
        ["status", "claimed_at", "created_at"],
    )
    op.create_table(
        "ml_economic_qualification_campaigns",
        sa.Column(
            "campaign_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_search_campaigns.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "qualification_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_economic_qualifications.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "family_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_search_campaign_families.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "campaign_id",
            "family_id",
            name="uq_ml_economic_qualification_campaign_family",
        ),
    )


def downgrade() -> None:
    op.drop_table("ml_economic_qualification_campaigns")
    op.drop_index(
        "ix_ml_economic_qualification_claim",
        table_name="ml_economic_qualifications",
    )
    op.drop_index(
        "ix_ml_economic_qualification_list",
        table_name="ml_economic_qualifications",
    )
    op.drop_table("ml_economic_qualifications")
    op.drop_constraint(
        "ck_ml_search_campaign_outcome", "ml_search_campaigns", type_="check"
    )
    op.create_check_constraint(
        "ck_ml_search_campaign_outcome",
        "ml_search_campaigns",
        "outcome IS NULL OR outcome IN ('coverage_incomplete',"
        "'no_strong_candidate','search_budget_exhausted','strong_candidates_found')",
    )
    op.drop_index(
        "ix_ml_search_family_v2_identity", table_name="ml_search_campaign_families"
    )
    op.drop_index("ix_ml_search_campaign_preset", table_name="ml_search_campaigns")
    op.drop_constraint(
        "fk_ml_search_family_seed", "ml_search_campaign_families", type_="foreignkey"
    )
    op.drop_column("ml_search_campaign_families", "selected_policy")
    op.drop_column("ml_search_campaign_families", "selected_policy_fingerprint")
    op.drop_column("ml_search_campaign_families", "seed_provenance")
    op.drop_column("ml_search_campaign_families", "seed_family_id")
    op.drop_column("ml_search_campaign_families", "search_track")
    op.drop_column("ml_search_campaign_families", "preset")
    op.drop_column("ml_search_campaigns", "preset")
