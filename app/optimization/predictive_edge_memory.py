from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.base import Base


class PredictiveEdgeRunRecord(Base):
    __tablename__ = "predictive_edge_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_predictive_edge_runs_status",
        ),
        CheckConstraint(
            "range_start < range_end AND training_days > 0 AND validation_days > 0 "
            "AND test_days > 0 AND step_days > 0 AND maximum_windows > 0 "
            "AND bin_count BETWEEN 2 AND 10",
            name="ck_predictive_edge_runs_configuration",
        ),
        UniqueConstraint("experiment_key", name="uq_predictive_edge_runs_key"),
        Index(
            "ix_predictive_edge_runs_instrument_started",
            "asset",
            "provider",
            "market_type",
            "interval",
            "started_at",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    experiment_key: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    asset: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(16), nullable=False)
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    range_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    range_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    algorithm_version: Mapped[int] = mapped_column(Integer, nullable=False)
    score_model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    score_model_version: Mapped[int] = mapped_column(Integer, nullable=False)
    fee_rate: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    slippage_rate: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    neutral_threshold: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    decision_interval_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    training_days: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_days: Mapped[int] = mapped_column(Integer, nullable=False)
    test_days: Mapped[int] = mapped_column(Integer, nullable=False)
    step_days: Mapped[int] = mapped_column(Integer, nullable=False)
    maximum_windows: Mapped[int] = mapped_column(Integer, nullable=False)
    bin_count: Mapped[int] = mapped_column(Integer, nullable=False)
    minimum_observations: Mapped[int] = mapped_column(Integer, nullable=False)
    minimum_windows: Mapped[int] = mapped_column(Integer, nullable=False)
    minimum_stable_direction_ratio: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    minimum_absolute_correlation: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    minimum_directional_hit_rate: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    windows_generated: Mapped[int] = mapped_column(Integer, nullable=False)
    aggregate_metrics_computed: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics_reused: Mapped[int] = mapped_column(Integer, nullable=False)
    compute_avoided_estimate: Mapped[int] = mapped_column(Integer, nullable=False)
    reuse_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_reason: Mapped[str | None] = mapped_column(String(512))


class PredictiveEdgeWindowRecord(Base):
    __tablename__ = "predictive_edge_windows"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "ordinal", name="uq_predictive_edge_windows_ordinal"
        ),
        CheckConstraint(
            "training_from < training_to AND training_to = validation_from "
            "AND validation_from < validation_to AND validation_to = test_from "
            "AND test_from < test_to",
            name="ck_predictive_edge_windows_chronology",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("predictive_edge_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    training_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    training_to: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    validation_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    validation_to: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    test_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    test_to: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    usable_observations: Mapped[int] = mapped_column(Integer, nullable=False)
    selection_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)


class PredictiveEdgeAggregateRecord(Base):
    __tablename__ = "predictive_edge_aggregates"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "predictor_name",
            "horizon_seconds",
            "condition",
            name="uq_predictive_edge_aggregates_identity",
        ),
        Index(
            "ix_predictive_edge_aggregates_filter",
            "run_id",
            "classification",
            "predictor_name",
            "horizon_seconds",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("predictive_edge_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    predictor_name: Mapped[str] = mapped_column(String(64), nullable=False)
    predictor_version: Mapped[int] = mapped_column(Integer, nullable=False)
    predictor_source: Mapped[str] = mapped_column(String(128), nullable=False)
    direction_convention: Mapped[str] = mapped_column(String(128), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    horizon_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    condition: Mapped[str] = mapped_column(String(64), nullable=False)
    windows_evaluated: Mapped[int] = mapped_column(Integer, nullable=False)
    sufficient_windows: Mapped[int] = mapped_column(Integer, nullable=False)
    total_usable_observations: Mapped[int] = mapped_column(Integer, nullable=False)
    stable_direction_ratio: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    weighted_mean_correlation: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    median_correlation: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    worst_correlation: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    mean_directional_hit_rate: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    mean_positive_net_ratio: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    mean_net_return: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    correlation_dispersion: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    classification: Mapped[str] = mapped_column(String(48), nullable=False)


class PredictiveEdgeWindowMetricRecord(Base):
    __tablename__ = "predictive_edge_window_metrics"
    __table_args__ = (
        UniqueConstraint(
            "window_id",
            "predictor_name",
            "horizon_seconds",
            "condition",
            name="uq_predictive_edge_window_metrics_identity",
        ),
        Index(
            "ix_predictive_edge_window_metrics_edge",
            "run_id",
            "predictor_name",
            "horizon_seconds",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("predictive_edge_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    window_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("predictive_edge_windows.id", ondelete="CASCADE"),
        nullable=False,
    )
    predictor_name: Mapped[str] = mapped_column(String(64), nullable=False)
    horizon_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    condition: Mapped[str] = mapped_column(String(64), nullable=False)
    usable_count: Mapped[int] = mapped_column(Integer, nullable=False)
    missing_count: Mapped[int] = mapped_column(Integer, nullable=False)
    correlation: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    directional_hit_rate: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    mean_gross_return: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    mean_net_return: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    positive_net_ratio: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    standard_error: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    evidence_status: Mapped[str] = mapped_column(String(48), nullable=False)
    insufficient_reason: Mapped[str | None] = mapped_column(String(256))
    training_bin_bounds: Mapped[str] = mapped_column(Text, nullable=False)
    volatility_threshold: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    evidence_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)


class PredictiveEdgeBinAggregateRecord(Base):
    __tablename__ = "predictive_edge_bin_aggregates"
    __table_args__ = (
        UniqueConstraint(
            "window_metric_id",
            "ordinal",
            name="uq_predictive_edge_bin_aggregates_ordinal",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    window_metric_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("predictive_edge_window_metrics.id", ondelete="CASCADE"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    lower_bound: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    upper_bound: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_forward_return: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    median_forward_return: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    up_count: Mapped[int] = mapped_column(Integer, nullable=False)
    down_count: Mapped[int] = mapped_column(Integer, nullable=False)
    neutral_count: Mapped[int] = mapped_column(Integer, nullable=False)
    positive_net_ratio: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    mean_net_return: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    evidence_status: Mapped[str] = mapped_column(String(48), nullable=False)


class StrategyHypothesisRecord(Base):
    __tablename__ = "strategy_hypotheses"
    __table_args__ = (
        UniqueConstraint("hypothesis_id", name="uq_strategy_hypotheses_stable_id"),
        Index("ix_strategy_hypotheses_run_status", "run_id", "evidence_status"),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    hypothesis_id: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("predictive_edge_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    predictor_name: Mapped[str] = mapped_column(String(64), nullable=False)
    predictor_version: Mapped[int] = mapped_column(Integer, nullable=False)
    direction: Mapped[str] = mapped_column(String(32), nullable=False)
    horizon_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    condition: Mapped[str] = mapped_column(String(64), nullable=False)
    expected_gross_return: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    expected_net_return: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    directional_hit_rate: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    stability_ratio: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    evaluation_interval_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    holding_period_from_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    holding_period_to_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    evidence_status: Mapped[str] = mapped_column(String(48), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    limitations: Mapped[str] = mapped_column(Text, nullable=False)


class PredictiveEdgeBenchmarkRecord(Base):
    __tablename__ = "predictive_edge_benchmarks"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "horizon_seconds",
            "benchmark",
            name="uq_predictive_edge_benchmarks_identity",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("predictive_edge_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    horizon_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    benchmark: Mapped[str] = mapped_column(String(64), nullable=False)
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    directional_hit_rate: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    mean_return: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
