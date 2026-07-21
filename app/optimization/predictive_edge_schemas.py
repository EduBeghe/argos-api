from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from app.optimization.predictive_edges import (
    DirectionMapping,
    EvidenceStatus,
    PredictorSemanticKind,
    TestConfirmationStatus,
    ValidationStatus,
)


class PredictiveEdgeRunResponse(BaseModel):
    id: UUID
    status: str
    asset: str
    provider: str
    market_type: str
    interval: str
    dataset_fingerprint: str
    range_start: datetime
    range_end: datetime
    algorithm_version: int
    windows_available: int
    windows_required: int
    conclusion_reason: str | None
    windows_generated: int
    aggregate_metrics_computed: int
    metrics_reused: int
    compute_avoided_estimate: int
    started_at: datetime
    completed_at: datetime | None
    failed_at: datetime | None
    failure_reason: str | None


class PredictiveEdgeRunPageResponse(BaseModel):
    items: tuple[PredictiveEdgeRunResponse, ...]
    next_cursor: str | None
    has_more: bool


class PredictiveEdgeResponse(BaseModel):
    id: UUID
    predictor_name: str
    predictor_version: int
    predictor_source: str
    semantic_kind: PredictorSemanticKind
    direction_convention: str
    explanation: str
    horizon_seconds: int
    condition: str
    direction_mapping: DirectionMapping
    supporting_bin_ordinals: tuple[int, ...]
    windows_evaluated: int
    windows_selected_for_test: int
    windows_confirmed: int
    total_usable_observations: int
    stable_direction_ratio: Decimal
    weighted_mean_correlation: Decimal | None
    median_correlation: Decimal | None
    worst_correlation: Decimal | None
    mean_directional_hit_rate: Decimal | None
    mean_movement_after_friction_ratio: Decimal
    mean_net_movement_magnitude_after_friction: Decimal
    mean_strategy_aligned_gross_return: Decimal | None
    mean_strategy_aligned_net_return: Decimal | None
    mean_expected_avoided_downside: Decimal | None
    correlation_dispersion: Decimal | None
    classification: EvidenceStatus


class PredictiveEdgePageResponse(BaseModel):
    items: tuple[PredictiveEdgeResponse, ...]
    next_cursor: str | None
    has_more: bool


class PredictiveEdgeWindowMetricResponse(BaseModel):
    id: UUID
    window_ordinal: int
    semantic_kind: PredictorSemanticKind
    training_center: Decimal
    direction_mapping: DirectionMapping
    validation_status: ValidationStatus
    validation_direction: DirectionMapping
    validation_usable_count: int
    validation_correlation: Decimal | None
    validation_directional_hit_rate: Decimal | None
    validation_mean_strategy_aligned_net_return: Decimal | None
    selected_for_test: bool
    selection_reason: str
    selected_bin_ordinals: tuple[int, ...]
    test_confirmation_status: TestConfirmationStatus
    test_usable_count: int
    test_missing_count: int
    test_correlation: Decimal | None
    test_directional_hit_rate: Decimal | None
    mean_forward_gross_return: Decimal
    mean_absolute_movement: Decimal
    mean_net_movement_magnitude_after_friction: Decimal
    positive_movement_after_friction_ratio: Decimal
    mean_strategy_aligned_gross_return: Decimal | None
    mean_strategy_aligned_net_return: Decimal | None
    mean_expected_avoided_downside: Decimal | None
    test_standard_error: Decimal | None
    test_evidence_status: EvidenceStatus
    test_insufficient_reason: str | None
    training_bin_bounds: tuple[Decimal, ...]
    volatility_threshold: Decimal


class PredictiveEdgeWindowPageResponse(BaseModel):
    items: tuple[PredictiveEdgeWindowMetricResponse, ...]
    next_cursor: str | None
    has_more: bool


class StrategyHypothesisResponse(BaseModel):
    hypothesis_id: str
    predictor_name: str
    predictor_version: int
    direction: str
    horizon_seconds: int
    condition: str
    supporting_bin_ordinals: tuple[int, ...]
    expected_gross_return: Decimal | None
    expected_net_return: Decimal | None
    expected_avoided_loss: Decimal | None
    directional_hit_rate: Decimal | None
    stability_ratio: Decimal
    evaluation_interval_seconds: int
    holding_period_from_seconds: int
    holding_period_to_seconds: int
    evidence_status: EvidenceStatus
    rationale: str
    limitations: str


class StrategyHypothesisPageResponse(BaseModel):
    items: tuple[StrategyHypothesisResponse, ...]
    next_cursor: str | None
    has_more: bool


class PredictiveEdgeRunDetailResponse(BaseModel):
    run: PredictiveEdgeRunResponse
    configuration: dict[str, int | Decimal | str]
