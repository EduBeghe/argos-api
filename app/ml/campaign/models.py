from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.base import Base

METRIC = Numeric(38, 18)


class MLSearchCampaignRecord(Base):
    __tablename__ = "ml_search_campaigns"
    __table_args__ = (
        UniqueConstraint("experiment_key", name="uq_ml_search_campaign_experiment_key"),
        CheckConstraint(
            "status IN ('queued','running','completed','failed','cancelled')",
            name="ck_ml_search_campaign_status",
        ),
        CheckConstraint(
            "maximum_evaluations > 0 AND maximum_runtime_seconds > 0",
            name="ck_ml_search_campaign_limits",
        ),
        CheckConstraint(
            "families_total >= 0 AND families_covered >= 0 "
            "AND families_uncovered >= 0 AND evaluations_computed >= 0 "
            "AND evaluations_reused >= 0",
            name="ck_ml_search_campaign_counters",
        ),
        CheckConstraint(
            "families_covered + families_uncovered = families_total",
            name="ck_ml_search_campaign_coverage",
        ),
        CheckConstraint(
            "outcome IS NULL OR outcome IN ('coverage_incomplete',"
            "'no_strong_candidate','search_budget_exhausted',"
            "'strong_candidates_found','qualified_candidates_found',"
            "'no_qualified_candidate')",
            name="ck_ml_search_campaign_outcome",
        ),
        Index("ix_ml_search_campaign_preset", "preset", "created_at"),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    experiment_key: Mapped[str] = mapped_column(String(64), nullable=False)
    preset: Mapped[str] = mapped_column(
        String(64), nullable=False, default="exhaustive_ohlcv_v1"
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    outcome: Mapped[str | None] = mapped_column(String(64))
    contract_version: Mapped[int] = mapped_column(Integer, nullable=False)
    policy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    classification_version: Mapped[int] = mapped_column(Integer, nullable=False)
    normalized_request: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    maximum_evaluations: Mapped[int] = mapped_column(Integer, nullable=False)
    maximum_runtime_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    families_total: Mapped[int] = mapped_column(Integer, nullable=False)
    families_covered: Mapped[int] = mapped_column(Integer, nullable=False)
    families_uncovered: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluations_computed: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluations_reused: Mapped[int] = mapped_column(Integer, nullable=False)
    current_stage: Mapped[int] = mapped_column(Integer, nullable=False)
    cancellation_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    first_claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_type: Mapped[str | None] = mapped_column(String(128))
    failure_message: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class MLSearchCampaignSourceRecord(Base):
    __tablename__ = "ml_search_campaign_sources"
    campaign_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_search_campaigns.id", ondelete="CASCADE"),
        primary_key=True,
    )
    source_screening_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_model_screening_runs.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)


class MLSearchCampaignFamilyRecord(Base):
    __tablename__ = "ml_search_campaign_families"
    __table_args__ = (
        UniqueConstraint(
            "campaign_id", "fingerprint", name="uq_ml_search_campaign_family"
        ),
        CheckConstraint(
            "current_stage BETWEEN 0 AND 4", name="ck_ml_search_family_stage"
        ),
        CheckConstraint(
            "state IN ('pending','scheduled','survived','discarded',"
            "'strong_development_candidate')",
            name="ck_ml_search_family_state",
        ),
        Index(
            "ix_ml_search_family_v2_identity",
            "campaign_id",
            "preset",
            "search_track",
            "seed_family_id",
            "selected_policy_fingerprint",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    campaign_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_search_campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_screening_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_model_screening_runs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    preset: Mapped[str] = mapped_column(
        String(64), nullable=False, default="exhaustive_ohlcv_v1"
    )
    search_track: Mapped[str] = mapped_column(
        String(32), nullable=False, default="exhaustive"
    )
    seed_family_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_search_campaign_families.id", ondelete="RESTRICT"),
    )
    seed_provenance: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    asset: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(16), nullable=False)
    engine: Mapped[str] = mapped_column(String(16), nullable=False)
    future_duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    objective: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_variant: Mapped[str] = mapped_column(String(16), nullable=False)
    state: Mapped[str] = mapped_column(String(64), nullable=False)
    current_stage: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluations_planned: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluations_completed: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluations_reused: Mapped[int] = mapped_column(Integer, nullable=False)
    best_evaluation_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_refinement_evaluations.id", ondelete="SET NULL"),
    )
    best_metrics: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    selected_policy: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    selected_policy_fingerprint: Mapped[str | None] = mapped_column(String(64))
    decision_reason: Mapped[str | None] = mapped_column(String(128))
    deterministic_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class MLSearchCampaignChildRecord(Base):
    __tablename__ = "ml_search_campaign_children"
    __table_args__ = (
        UniqueConstraint(
            "campaign_id",
            "family_id",
            "stage",
            name="uq_ml_search_campaign_child_stage",
        ),
        Index("ix_ml_search_campaign_child_run", "refinement_run_id"),
        CheckConstraint(
            "stage BETWEEN 1 AND 4 AND reserved_evaluations > 0",
            name="ck_ml_search_campaign_child_stage",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    campaign_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_search_campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )
    family_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_search_campaign_families.id", ondelete="CASCADE"),
        nullable=False,
    )
    refinement_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_refinement_runs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    stage: Mapped[int] = mapped_column(Integer, nullable=False)
    reserved_evaluations: Mapped[int] = mapped_column(Integer, nullable=False)
    reuses_existing_run: Mapped[bool] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class MLEconomicQualificationRecord(Base):
    __tablename__ = "ml_economic_qualifications"
    __table_args__ = (
        UniqueConstraint(
            "candidate_fingerprint",
            "qualification_version",
            name="uq_ml_economic_qualification_candidate",
        ),
        CheckConstraint(
            "status IN ('queued','running','completed','failed','cancelled')",
            name="ck_ml_economic_qualification_status",
        ),
        CheckConstraint(
            "classification IS NULL OR classification IN ('qualified','rejected')",
            name="ck_ml_economic_qualification_classification",
        ),
        CheckConstraint(
            "provenance IN ('automatic_development_candidate',"
            "'diagnostic_stage3_near_miss')",
            name="ck_ml_economic_qualification_provenance",
        ),
        CheckConstraint(
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
        CheckConstraint(
            "evidence_end_exclusive = TIMESTAMPTZ '2025-01-01 00:00:00+00'",
            name="ck_ml_economic_qualification_boundary",
        ),
        Index(
            "ix_ml_economic_qualification_list",
            "campaign_id",
            "classification",
            "engine",
            "objective",
            "search_track",
        ),
        Index(
            "ix_ml_economic_qualification_claim",
            "status",
            "claimed_at",
            "created_at",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    campaign_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_search_campaigns.id", ondelete="CASCADE"),
        nullable=False,
    )
    family_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_search_campaign_families.id", ondelete="CASCADE"),
        nullable=False,
    )
    candidate_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    qualification_version: Mapped[int] = mapped_column(Integer, nullable=False)
    provenance: Mapped[str] = mapped_column(
        String(48), nullable=False, default="automatic_development_candidate"
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    classification: Mapped[str | None] = mapped_column(String(16))
    engine: Mapped[str] = mapped_column(String(16), nullable=False)
    objective: Mapped[str] = mapped_column(String(64), nullable=False)
    search_track: Mapped[str] = mapped_column(String(32), nullable=False)
    seed_family_id: Mapped[UUID | None] = mapped_column(PostgreSQLUUID(as_uuid=True))
    frozen_inputs: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    temporal_manifest: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    temporal_manifest_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    policy_identity: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    scenario_metrics: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    quarter_returns: Mapped[list[str] | None] = mapped_column(JSONB)
    trades: Mapped[list[dict[str, object]] | None] = mapped_column(JSONB)
    failed_rules: Mapped[list[str] | None] = mapped_column(JSONB)
    evidence_end_exclusive: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    worker_id: Mapped[str | None] = mapped_column(String(128))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancellation_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_type: Mapped[str | None] = mapped_column(String(128))
    failure_message: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class MLEconomicQualificationCampaignRecord(Base):
    __tablename__ = "ml_economic_qualification_campaigns"
    __table_args__ = (
        UniqueConstraint(
            "campaign_id",
            "family_id",
            name="uq_ml_economic_qualification_campaign_family",
        ),
    )
    campaign_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_search_campaigns.id", ondelete="CASCADE"),
        primary_key=True,
    )
    qualification_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_economic_qualifications.id", ondelete="CASCADE"),
        primary_key=True,
    )
    family_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_search_campaign_families.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
