from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel

from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.edge_guided import EdgeGuidedCandidateStatus
from app.optimization.predictive_edges import (
    Condition,
    DirectionMapping,
    SuggestedAction,
)
from app.optimization.refinement import RefinementSegmentName
from app.optimization.schemas import InstrumentIdentityResponse


class EdgeGuidedRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EdgeGuidedActionFilter(StrEnum):
    LONG_ENTRY = "long_entry"
    EXIT_OR_AVOID = "exit_or_avoid"


class EdgeGuidedStatusCountsResponse(BaseModel):
    confirmed_candidate: int
    rejected_validation: int
    rejected_test: int


class EdgeGuidedRunResponse(BaseModel):
    id: UUID
    experiment_key: str
    predictive_edge_run_id: UUID
    status: EdgeGuidedRunStatus
    instrument: InstrumentIdentityResponse
    dataset_fingerprint: str
    range_start: datetime
    range_end: datetime
    initial_cash: Decimal
    account_currency: str
    fee_rate: Decimal
    slippage_rate: Decimal
    decision_interval_seconds: int
    maximum_candidates: int
    algorithm_version: int
    generated_count: int
    evaluated_count: int
    windows_available: int
    started_at: datetime
    completed_at: datetime | None
    candidate_status_counts: EdgeGuidedStatusCountsResponse


class EdgeGuidedRunPageResponse(BaseModel):
    items: tuple[EdgeGuidedRunResponse, ...]
    total: int
    limit: int
    offset: int


class EdgeGuidedSegmentMetricsResponse(BaseModel):
    segment: RefinementSegmentName
    total_return: Decimal
    max_drawdown: Decimal
    trade_count: int
    fees_paid: Decimal
    return_difference_from_production: Decimal
    return_difference_from_buy_and_hold: Decimal


class EdgeGuidedCandidateResponse(BaseModel):
    candidate_evaluation_id: UUID
    run_id: UUID
    candidate_fingerprint: str
    source_hypothesis_id: str
    source_edge_fingerprint: str
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    dataset_fingerprint: str
    range_start: datetime
    range_end: datetime
    predictor_name: str
    predictor_version: int
    hypothesis_contract_version: int
    condition: Condition
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    suggested_action: SuggestedAction
    horizon_seconds: int
    decision_interval_seconds: int
    direction_mapping: DirectionMapping
    score_model_name: str
    score_model_version: int
    strategy_name: str
    strategy_version: int
    strategy_configuration_version: int
    account_currency: str
    fee_rate: Decimal
    slippage_rate: Decimal
    status: EdgeGuidedCandidateStatus
    explanation: str
    unconditional_benchmark: str
    valid_window_count: int
    confirmed_window_count: int
    rejected_validation_window_count: int
    rejected_test_window_count: int
    training_metrics: EdgeGuidedSegmentMetricsResponse
    validation_metrics: EdgeGuidedSegmentMetricsResponse
    test_metrics: EdgeGuidedSegmentMetricsResponse


class EdgeGuidedCandidatePageResponse(BaseModel):
    items: tuple[EdgeGuidedCandidateResponse, ...]
    total: int
    limit: int
    offset: int


class EdgeGuidedWindowResponse(BaseModel):
    window_ordinal: int
    volatility_threshold: Decimal
    training_metrics: EdgeGuidedSegmentMetricsResponse
    validation_metrics: EdgeGuidedSegmentMetricsResponse
    test_metrics: EdgeGuidedSegmentMetricsResponse | None
    validation_passed: bool
    test_confirmed: bool


class EdgeGuidedCandidateDetailResponse(EdgeGuidedCandidateResponse):
    windows: tuple[EdgeGuidedWindowResponse, ...]
