from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    Boolean,
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


class MLModelScreeningRunRecord(Base):
    __tablename__ = "ml_model_screening_runs"
    __table_args__ = (
        Index("ix_ml_screening_completed_key", "experiment_key", "status"),
        CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_ml_screening_run_status",
        ),
        CheckConstraint(
            "screening_version > 0", name="ck_ml_screening_version_positive"
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    experiment_key: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    screening_version: Mapped[int] = mapped_column(Integer, nullable=False)
    experiment_contract_version: Mapped[int] = mapped_column(Integer, nullable=False)
    variable_registry_version: Mapped[int] = mapped_column(Integer, nullable=False)
    missing_data_policy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    asset: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(16), nullable=False)
    requested_engines: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    requested_future_durations: Mapped[list[int]] = mapped_column(JSONB, nullable=False)
    library_versions: Mapped[dict[str, str]] = mapped_column(JSONB, nullable=False)
    deterministic_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    configuration: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_type: Mapped[str | None] = mapped_column(String(128))
    failure_message: Mapped[str | None] = mapped_column(String(512))
    evaluations_computed: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluations_reused: Mapped[int] = mapped_column(Integer, nullable=False)
    results_generated: Mapped[int] = mapped_column(Integer, nullable=False)
    pareto_retained_count: Mapped[int] = mapped_column(Integer, nullable=False)
    elapsed_seconds: Mapped[Decimal | None] = mapped_column(METRIC)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class MLModelScreeningEvaluationRecord(Base):
    __tablename__ = "ml_model_screening_evaluations"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "engine",
            "future_duration_seconds",
            "fold_name",
            name="uq_ml_screening_evaluation",
        ),
        CheckConstraint("sample_count > 0", name="ck_ml_screening_sample_positive"),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_model_screening_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    engine: Mapped[str] = mapped_column(String(16), nullable=False)
    future_duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    fold_name: Mapped[str] = mapped_column(String(64), nullable=False)
    training_row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_absolute_error: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    root_mean_squared_error: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    mean_prediction: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    mean_actual_return: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    pearson_correlation: Mapped[Decimal | None] = mapped_column(METRIC)
    spearman_correlation: Mapped[Decimal | None] = mapped_column(METRIC)
    directional_accuracy: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    zero_prediction_mae: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    mae_improvement_over_zero: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    positive_prediction_count: Mapped[int] = mapped_column(Integer, nullable=False)
    negative_prediction_count: Mapped[int] = mapped_column(Integer, nullable=False)
    elapsed_seconds: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class MLModelScreeningResultRecord(Base):
    __tablename__ = "ml_model_screening_results"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "engine", "future_duration_seconds", name="uq_ml_screening_result"
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_model_screening_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    engine: Mapped[str] = mapped_column(String(16), nullable=False)
    future_duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    folds_evaluated: Mapped[int] = mapped_column(Integer, nullable=False)
    total_validation_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    median_mae_improvement_over_zero: Mapped[Decimal] = mapped_column(
        METRIC, nullable=False
    )
    worst_fold_mae_improvement_over_zero: Mapped[Decimal] = mapped_column(
        METRIC, nullable=False
    )
    median_directional_accuracy: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    worst_fold_directional_accuracy: Mapped[Decimal] = mapped_column(
        METRIC, nullable=False
    )
    median_pearson_correlation: Mapped[Decimal | None] = mapped_column(METRIC)
    median_spearman_correlation: Mapped[Decimal | None] = mapped_column(METRIC)
    folds_beating_zero: Mapped[int] = mapped_column(Integer, nullable=False)
    folds_with_positive_spearman: Mapped[int] = mapped_column(Integer, nullable=False)
    total_training_time_seconds: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    pareto_retained: Mapped[bool] = mapped_column(Boolean, nullable=False)
    display_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
