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
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.base import Base


class PredictiveEdgeRunRecord(Base):
    __tablename__ = "predictive_edge_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'insufficient_history', 'failed')",
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
    status: Mapped[str] = mapped_column(String(32), nullable=False)
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
    minimum_hit_rate_improvement_over_benchmark: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    windows_available: Mapped[int] = mapped_column(Integer, nullable=False)
    windows_required: Mapped[int] = mapped_column(Integer, nullable=False)
    conclusion_reason: Mapped[str | None] = mapped_column(String(256))
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
            "direction_mapping",
            "bin_ordinal",
            "suggested_action",
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
    semantic_kind: Mapped[str] = mapped_column(String(48), nullable=False)
    direction_convention: Mapped[str] = mapped_column(String(128), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    horizon_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    condition: Mapped[str] = mapped_column(String(64), nullable=False)
    windows_evaluated: Mapped[int] = mapped_column(Integer, nullable=False)
    direction_mapping: Mapped[str] = mapped_column(String(32), nullable=False)
    bin_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    lower_bound: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    upper_bound: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    suggested_action: Mapped[str] = mapped_column(String(32), nullable=False)
    supporting_bin_ordinals: Mapped[list[int]] = mapped_column(
        ARRAY(Integer), nullable=False
    )
    windows_selected_for_test: Mapped[int] = mapped_column(Integer, nullable=False)
    windows_confirmed: Mapped[int] = mapped_column(Integer, nullable=False)
    total_usable_observations: Mapped[int] = mapped_column(Integer, nullable=False)
    stable_direction_ratio: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    weighted_mean_correlation: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    median_correlation: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    worst_correlation: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    mean_directional_hit_rate: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    mean_movement_after_friction_ratio: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    mean_net_movement_magnitude_after_friction: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    mean_strategy_aligned_gross_return: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 18)
    )
    mean_strategy_aligned_net_return: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 18)
    )
    mean_expected_avoided_downside: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 18)
    )
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
    semantic_kind: Mapped[str] = mapped_column(String(48), nullable=False)
    horizon_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    condition: Mapped[str] = mapped_column(String(64), nullable=False)
    training_center: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    direction_mapping: Mapped[str] = mapped_column(String(32), nullable=False)
    validation_status: Mapped[str] = mapped_column(String(48), nullable=False)
    validation_direction: Mapped[str] = mapped_column(String(32), nullable=False)
    validation_usable_count: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_correlation: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    validation_directional_hit_rate: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 18)
    )
    validation_mean_strategy_aligned_net_return: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 18)
    )
    selected_for_test: Mapped[bool] = mapped_column(nullable=False)
    selection_reason: Mapped[str] = mapped_column(String(256), nullable=False)
    selected_bin_ordinals: Mapped[list[int]] = mapped_column(
        ARRAY(Integer), nullable=False
    )
    test_confirmation_status: Mapped[str] = mapped_column(String(48), nullable=False)
    test_usable_count: Mapped[int] = mapped_column(Integer, nullable=False)
    test_missing_count: Mapped[int] = mapped_column(Integer, nullable=False)
    test_correlation: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    test_directional_hit_rate: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    mean_forward_gross_return: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    mean_absolute_movement: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    mean_net_movement_magnitude_after_friction: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    positive_movement_after_friction_ratio: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    mean_strategy_aligned_gross_return: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 18)
    )
    mean_strategy_aligned_net_return: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 18)
    )
    mean_expected_avoided_downside: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 18)
    )
    test_standard_error: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    test_evidence_status: Mapped[str] = mapped_column(String(48), nullable=False)
    test_insufficient_reason: Mapped[str | None] = mapped_column(String(256))
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
    mean_forward_gross_return: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    median_forward_gross_return: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    up_count: Mapped[int] = mapped_column(Integer, nullable=False)
    down_count: Mapped[int] = mapped_column(Integer, nullable=False)
    neutral_count: Mapped[int] = mapped_column(Integer, nullable=False)
    positive_movement_after_friction_ratio: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    mean_net_movement_magnitude_after_friction: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    evidence_status: Mapped[str] = mapped_column(String(48), nullable=False)
    suggested_action: Mapped[str] = mapped_column(String(32), nullable=False)
    validation_observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_gross_return: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    validation_long_aligned_net_return: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 18)
    )
    validation_avoided_downside: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    validation_directional_hit_rate: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 18)
    )
    selected_for_test: Mapped[bool] = mapped_column(nullable=False)
    test_observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    test_confirmation_status: Mapped[str] = mapped_column(String(48), nullable=False)
    test_gross_return: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    test_long_aligned_net_return: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 18)
    )
    test_avoided_downside: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    test_directional_hit_rate: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    rejection_reason: Mapped[str | None] = mapped_column(String(256))


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
    hypothesis_contract_version: Mapped[int] = mapped_column(Integer, nullable=False)
    direction_mapping: Mapped[str] = mapped_column(String(32), nullable=False)
    bin_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    lower_bound: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    upper_bound: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    suggested_action: Mapped[str] = mapped_column(String(32), nullable=False)
    horizon_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    condition: Mapped[str] = mapped_column(String(64), nullable=False)
    supporting_bin_ordinals: Mapped[list[int]] = mapped_column(
        ARRAY(Integer), nullable=False
    )
    expected_gross_return: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    expected_net_return: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    expected_avoided_loss: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    directional_hit_rate: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    stability_ratio: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    evaluation_interval_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    holding_period_from_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    holding_period_to_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    evidence_status: Mapped[str] = mapped_column(String(48), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    limitations: Mapped[str] = mapped_column(Text, nullable=False)
    context_predictor_name: Mapped[str | None] = mapped_column(String(64))
    context_predictor_version: Mapped[int | None] = mapped_column(Integer)
    context_regime: Mapped[str | None] = mapped_column(String(32))
    context_lower_bound: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    context_upper_bound: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))


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
    mean_forward_gross_return: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )


class PredictiveEdgeRunDiagnosticRecord(Base):
    """Bounded v4 run summary; v3 runs remain readable without this child."""

    __tablename__ = "predictive_edge_run_diagnostics"
    __table_args__ = (
        UniqueConstraint("run_id", name="uq_predictive_edge_run_diagnostics_run"),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("predictive_edge_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    contract_version: Mapped[int] = mapped_column(Integer, nullable=False)
    effective_parameters: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB, nullable=False
    )
    rejection_counts_by_reason: Mapped[dict[str, int]] = mapped_column(
        JSONB, nullable=False
    )
    rejection_counts_by_predictor: Mapped[dict[str, int]] = mapped_column(
        JSONB, nullable=False
    )
    rejection_counts_by_horizon: Mapped[dict[str, int]] = mapped_column(
        JSONB, nullable=False
    )
    rejection_counts_by_condition: Mapped[dict[str, int]] = mapped_column(
        JSONB, nullable=False
    )
    rejection_counts_by_neutral_policy: Mapped[dict[str, int]] = mapped_column(
        JSONB, nullable=False
    )
    rejection_counts_by_cadence: Mapped[dict[str, int]] = mapped_column(
        JSONB, nullable=False
    )
    strongest_observed_signal: Mapped[str | None] = mapped_column(Text)
    bottleneck_stage: Mapped[str] = mapped_column(String(32), nullable=False)
    diagnostic_conclusion: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome_explanation: Mapped[str] = mapped_column(Text, nullable=False)
    stage_timings: Mapped[dict[str, str]] = mapped_column(JSONB, nullable=False)
    computed_counts: Mapped[dict[str, int]] = mapped_column(JSONB, nullable=False)
    reused_counts: Mapped[dict[str, int]] = mapped_column(JSONB, nullable=False)
    regime_conditioned_conclusion: Mapped[str | None] = mapped_column(String(64))
    regime_conditioned_stage_counters: Mapped[dict[str, object] | None] = mapped_column(
        JSONB
    )


class PredictiveEdgeNearMissRecord(Base):
    __tablename__ = "predictive_edge_near_misses"
    __table_args__ = (
        UniqueConstraint("run_id", "rank", name="uq_predictive_edge_near_misses_rank"),
        CheckConstraint(
            "kind IN ('actionable', 'diagnostic')",
            name="ck_predictive_edge_near_misses_kind",
        ),
        Index("ix_predictive_edge_near_misses_run_rank", "run_id", "rank"),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("predictive_edge_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="actionable"
    )
    edge_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    predictor_name: Mapped[str] = mapped_column(String(64), nullable=False)
    directional_predictor: Mapped[str | None] = mapped_column(String(64))
    context_predictor: Mapped[str | None] = mapped_column(String(64))
    context_regime: Mapped[str | None] = mapped_column(String(32))
    cadence_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    horizon_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    bin_count: Mapped[int] = mapped_column(Integer, nullable=False)
    condition: Mapped[str] = mapped_column(String(64), nullable=False)
    neutral_policy: Mapped[str] = mapped_column(String(32), nullable=False)
    profile: Mapped[str] = mapped_column(String(32), nullable=False)
    classification: Mapped[str] = mapped_column(String(48), nullable=False)
    primary_rejection_reason: Mapped[str] = mapped_column(String(64), nullable=False)
    passed_requirements: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False
    )
    failed_requirements: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False
    )
    requirement_margins: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB, nullable=False
    )
    usable_observations: Mapped[int] = mapped_column(Integer, nullable=False)
    stable_direction_ratio: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    expected_net_movement: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    normalized_missing_distance: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )


class PredictiveEdgeSensitivityRecord(Base):
    __tablename__ = "predictive_edge_threshold_sensitivity"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "edge_fingerprint",
            "threshold_name",
            name="uq_predictive_edge_sensitivity_identity",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("predictive_edge_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    edge_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    threshold_name: Mapped[str] = mapped_column(String(64), nullable=False)
    current_value: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    nearest_passing_value: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    candidate_would_advance: Mapped[bool] = mapped_column(nullable=False)
    other_requirements_still_failed: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False
    )


class PredictiveEdgeV4CacheRecord(Base):
    __tablename__ = "predictive_edge_v4_cache"
    __table_args__ = (
        UniqueConstraint("cache_key", name="uq_predictive_edge_v4_cache_key"),
        CheckConstraint(
            "status IN ('completed', 'running', 'failed')",
            name="ck_predictive_edge_v4_cache_status",
        ),
        Index("ix_predictive_edge_v4_cache_kind", "evidence_kind"),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    cache_key: Mapped[str] = mapped_column(String(64), nullable=False)
    cache_namespace: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    evidence_kind: Mapped[str] = mapped_column(String(40), nullable=False)
    contract_version: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    cadence_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    horizon_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    bin_count: Mapped[int | None] = mapped_column(Integer)
    condition: Mapped[str | None] = mapped_column(String(64))
    neutral_policy: Mapped[str | None] = mapped_column(String(32))
    profile: Mapped[str | None] = mapped_column(String(32))
    fee_rate: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    slippage_rate: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
