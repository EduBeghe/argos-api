from datetime import datetime
from decimal import Decimal
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


class MLRefinementRunRecord(Base):
    __tablename__ = "ml_refinement_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued','running','completed','failed','cancelled')",
            name="ck_ml_refinement_run_status",
        ),
        CheckConstraint(
            "(status = 'queued' AND started_at IS NULL AND completed_at IS NULL "
            "AND failed_at IS NULL AND worker_id IS NULL AND claimed_at IS NULL "
            "AND heartbeat_at IS NULL) OR "
            "(status = 'running' AND started_at IS NOT NULL "
            "AND completed_at IS NULL AND failed_at IS NULL AND worker_id IS NOT NULL "
            "AND claimed_at IS NOT NULL AND heartbeat_at IS NOT NULL) OR "
            "(status = 'completed' AND started_at IS NOT NULL "
            "AND completed_at IS NOT NULL AND failed_at IS NULL AND worker_id IS NULL "
            "AND claimed_at IS NULL AND heartbeat_at IS NULL) OR "
            "(status = 'failed' AND started_at IS NOT NULL "
            "AND completed_at IS NULL AND failed_at IS NOT NULL AND worker_id IS NULL "
            "AND claimed_at IS NULL AND heartbeat_at IS NULL) OR "
            "(status = 'cancelled' AND completed_at IS NOT NULL "
            "AND failed_at IS NULL AND worker_id IS NULL AND claimed_at IS NULL "
            "AND heartbeat_at IS NULL)",
            name="ck_ml_refinement_run_lifecycle",
        ),
        CheckConstraint(
            "maximum_trials > 0 AND maximum_runtime_seconds > 0 "
            "AND combinations_planned >= 0 AND evaluations_computed >= 0 "
            "AND evaluations_reused >= 0 AND evaluations_omitted >= 0 "
            "AND evaluations_selected >= 0 "
            "AND evaluations_computed + evaluations_reused <= evaluations_selected",
            name="ck_ml_refinement_run_budgets",
        ),
        UniqueConstraint("experiment_key", name="uq_ml_refinement_experiment_key"),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    experiment_key: Mapped[str] = mapped_column(String(64), nullable=False)
    source_screening_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_model_screening_runs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    source_screening_result_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_model_screening_results.id", ondelete="RESTRICT"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    contract_version: Mapped[int] = mapped_column(Integer, nullable=False)
    classification_version: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    variable_registry_version: Mapped[int] = mapped_column(Integer, nullable=False)
    missing_data_policy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    library_versions: Mapped[dict[str, str]] = mapped_column(JSONB, nullable=False)
    normalized_configuration: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False
    )
    notes: Mapped[str | None] = mapped_column(String(1000))
    maximum_trials: Mapped[int] = mapped_column(Integer, nullable=False)
    maximum_runtime_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    combinations_planned: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluations_computed: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluations_reused: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluations_omitted: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluations_selected: Mapped[int] = mapped_column(Integer, nullable=False)
    budget_exhausted: Mapped[bool] = mapped_column(nullable=False)
    termination_reason: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_type: Mapped[str | None] = mapped_column(String(128))
    failure_message: Mapped[str | None] = mapped_column(String(512))
    worker_id: Mapped[str | None] = mapped_column(String(128))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancellation_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    elapsed_seconds: Mapped[Decimal | None] = mapped_column(METRIC)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class MLRefinementEvaluationRecord(Base):
    __tablename__ = "ml_refinement_evaluations"
    __table_args__ = (
        UniqueConstraint("run_id", "fingerprint", name="uq_ml_refinement_evaluation"),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_refinement_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    reused_from_evaluation_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_refinement_evaluations.id", ondelete="SET NULL"),
    )
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    engine: Mapped[str] = mapped_column(String(16), nullable=False)
    future_duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    objective: Mapped[str] = mapped_column(String(64), nullable=False)
    hyperparameters: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    feature_policy: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    confidence_buckets: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    fold_evidence: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB, nullable=False
    )
    aggregate_metrics: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    classification: Mapped[str] = mapped_column(String(64), nullable=False)
    classification_version: Mapped[int] = mapped_column(Integer, nullable=False)
    failed_rules: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    elapsed_seconds: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
