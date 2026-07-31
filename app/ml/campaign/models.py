from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
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
            "'strong_candidates_found')",
            name="ck_ml_search_campaign_outcome",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    experiment_key: Mapped[str] = mapped_column(String(64), nullable=False)
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
        UniqueConstraint("refinement_run_id", name="uq_ml_search_campaign_child_run"),
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

