from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel

from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.edge_guided import (
    CandidateFamily,
    EdgeGuidedCandidateStatus,
    EdgeGuidedRejectionReason,
    EntryRule,
    ExitRule,
    ResearchOutcome,
)
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
    qualification_contract_version: int
    legacy: bool
    generated_count: int
    evaluated_count: int
    windows_available: int
    started_at: datetime
    completed_at: datetime | None
    candidate_status_counts: EdgeGuidedStatusCountsResponse
    research_outcome: ResearchOutcome | None = None
    outcome_explanation: str | None = None
    funnel: dict[str, int] | None = None
    timings: dict[str, Decimal] | None = None


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
    sufficient_trades: bool
    positive_net_return: bool
    beats_production: bool
    beats_buy_and_hold: bool
    risk_acceptable: bool


class EdgeGuidedRankingResponse(BaseModel):
    median_validation_return: Decimal
    worst_validation_return: Decimal
    maximum_drawdown: Decimal
    return_over_drawdown: Decimal | None
    trade_count: int
    total_fees: Decimal
    cost_efficiency: Decimal | None
    positive_window_ratio: Decimal
    positive_window_count: int
    candidate_fingerprint: str


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
    qualification_contract_version: int
    legacy: bool
    eligible_under_current_contract: bool
    rejection_reasons: tuple[EdgeGuidedRejectionReason, ...]
    explanation: str
    unconditional_benchmark: str
    valid_window_count: int
    confirmed_window_count: int
    rejected_validation_window_count: int
    rejected_test_window_count: int
    training_metrics: EdgeGuidedSegmentMetricsResponse
    validation_metrics: EdgeGuidedSegmentMetricsResponse
    test_metrics: EdgeGuidedSegmentMetricsResponse | None
    candidate_family: CandidateFamily | None = None
    entry_rule: EntryRule | None = None
    exit_rule: ExitRule | None = None
    cadence_seconds: int | None = None
    minimum_holding_seconds: int | None = None
    maximum_holding_seconds: int | None = None
    cooldown_seconds: int | None = None
    target_allocation_percentage: Decimal | None = None
    generation_contract_version: int | None = None
    ranking: EdgeGuidedRankingResponse | None = None
    cost_efficiency: Decimal | None = None
    positive_window_ratio: Decimal | None = None
    worst_window_return: Decimal | None = None


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
    rejection_reasons: tuple[EdgeGuidedRejectionReason, ...]


class EdgeGuidedCandidateDetailResponse(EdgeGuidedCandidateResponse):
    windows: tuple[EdgeGuidedWindowResponse, ...]
