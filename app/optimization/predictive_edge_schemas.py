from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from app.optimization.predictive_edges import EvidenceStatus


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
    direction_convention: str
    explanation: str
    horizon_seconds: int
    condition: str
    windows_evaluated: int
    sufficient_windows: int
    total_usable_observations: int
    stable_direction_ratio: Decimal
    weighted_mean_correlation: Decimal | None
    median_correlation: Decimal | None
    worst_correlation: Decimal | None
    mean_directional_hit_rate: Decimal | None
    mean_positive_net_ratio: Decimal
    mean_net_return: Decimal
    correlation_dispersion: Decimal | None
    classification: EvidenceStatus


class PredictiveEdgePageResponse(BaseModel):
    items: tuple[PredictiveEdgeResponse, ...]
    next_cursor: str | None
    has_more: bool


class PredictiveEdgeWindowMetricResponse(BaseModel):
    id: UUID
    window_ordinal: int
    usable_count: int
    missing_count: int
    correlation: Decimal | None
    directional_hit_rate: Decimal | None
    mean_gross_return: Decimal
    mean_net_return: Decimal
    positive_net_ratio: Decimal
    standard_error: Decimal | None
    evidence_status: EvidenceStatus
    insufficient_reason: str | None
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
    expected_gross_return: Decimal
    expected_net_return: Decimal
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

