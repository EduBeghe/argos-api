from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.experience import EvidenceClassification

MAX_OPTIMIZATION_PAGE_SIZE = 200


class RefinementRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ScoreParametersResponse(BaseModel):
    trend_weight: Decimal
    rsi_weight: Decimal
    return_1_weight: Decimal
    return_12_weight: Decimal
    volume_confirmation_weight: Decimal


class StrategyParametersResponse(BaseModel):
    entry_threshold: Decimal
    exit_threshold: Decimal
    open_position_target_percentage: Decimal


class RefinementBudgetResponse(BaseModel):
    score_screening_top_k: int
    validation_top_k: int
    max_score_configurations: int
    max_joint_candidates: int


class PlannerAllocationResponse(BaseModel):
    known_rechecks: Decimal
    promising_neighborhoods: Decimal
    uncertain_rechecks: Decimal
    novel_exploration: Decimal
    reconsider_rejected_after_runs: int


class RefinementCountsResponse(BaseModel):
    score_configurations_generated: int = 0
    joint_candidates_evaluated: int = 0
    simulations_total: int = 0
    evaluations_computed: int = 0
    evaluations_reused: int = 0
    compute_avoided_estimate: int = 0


class InstrumentIdentityResponse(BaseModel):
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval

    @classmethod
    def from_instrument(cls, value: Instrument) -> "InstrumentIdentityResponse":
        return cls(
            asset=value.asset,
            provider=value.provider,
            market_type=value.market_type,
            interval=value.interval,
        )


class TimeRangeResponse(BaseModel):
    start: datetime
    end: datetime


class SegmentBoundaryResponse(BaseModel):
    start: datetime
    end: datetime


class RefinementRunQuery(BaseModel):
    asset: Asset | None = None
    provider: Provider | None = None
    market_type: MarketType | None = None
    interval: Interval | None = None
    status: RefinementRunStatus | None = None
    recommendation_status: str | None = None
    started_from: datetime | None = None
    started_to: datetime | None = None
    limit: int = Field(default=50, ge=1, le=MAX_OPTIMIZATION_PAGE_SIZE)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def normalize_time_bounds(self) -> Self:
        for name in ("started_from", "started_to"):
            value = getattr(self, name)
            if value is not None:
                if value.tzinfo is None or value.utcoffset() is None:
                    raise ValueError(f"{name} must be timezone-aware")
                setattr(self, name, value.astimezone(UTC))
        if (
            self.started_from is not None
            and self.started_to is not None
            and self.started_to < self.started_from
        ):
            raise ValueError("started_to must not be before started_from")
        return self


class RefinementRunSummaryResponse(BaseModel):
    id: UUID
    experiment_key: str
    status: RefinementRunStatus
    instrument: InstrumentIdentityResponse
    requested_range: TimeRangeResponse
    training: SegmentBoundaryResponse
    validation: SegmentBoundaryResponse
    test: SegmentBoundaryResponse
    score_model_name: str
    score_model_version: int
    strategy_name: str
    strategy_version: int
    simulation_engine_version: int
    planner_version: int
    initial_cash: Decimal
    account_currency: str
    fee_rate: Decimal
    slippage_rate: Decimal
    decision_interval_seconds: int
    refinement_budget: RefinementBudgetResponse
    planner_allocation: PlannerAllocationResponse
    history_cutoff: datetime
    recommendation_status: str | None
    recommended_candidate_id: str | None
    counts: RefinementCountsResponse | None
    elapsed_seconds: Decimal | None
    started_at: datetime
    completed_at: datetime | None
    failed_at: datetime | None
    failure_reason: str | None
    last_verified_at: datetime | None


class SearchPlanResponse(BaseModel):
    planner_version: int
    known_rechecks: tuple[str, ...]
    promising_neighborhoods: tuple[str, ...]
    uncertain_rechecks: tuple[str, ...]
    novel_exploration: tuple[str, ...]
    known_recheck_count: int
    promising_neighborhood_count: int
    uncertain_recheck_count: int
    novel_exploration_count: int


class SegmentMetricsResponse(BaseModel):
    segment: str
    total_return: Decimal
    absolute_pnl: Decimal
    max_drawdown: Decimal
    trade_count: int
    fees_paid: Decimal
    final_equity: Decimal
    return_over_drawdown: Decimal


class EvaluationResponse(SegmentMetricsResponse):
    id: UUID
    stable_candidate_id: str
    phase: str
    disabled_signal: str | None
    rank: int | None
    eligible: bool | None
    rejection_reason: str | None
    evaluated_at: datetime


class EvaluationGroupResponse(BaseModel):
    phase: str
    segment: str
    evaluations: tuple[EvaluationResponse, ...]


class CandidateDefinitionResponse(BaseModel):
    stable_candidate_id: str
    score_model_name: str
    score_model_version: int
    strategy_name: str
    strategy_version: int
    score_parameters: ScoreParametersResponse
    strategy_parameters: StrategyParametersResponse
    created_at: datetime


class RefinementRunDetailResponse(BaseModel):
    run: RefinementRunSummaryResponse
    search_plan: SearchPlanResponse | None
    production_baseline: tuple[SegmentMetricsResponse, ...]
    buy_and_hold: tuple[SegmentMetricsResponse, ...]
    recommended_candidate: CandidateDefinitionResponse | None
    explanation: str | None
    result_digest: str | None
    evaluations: tuple[EvaluationGroupResponse, ...]


class CandidateQuery(BaseModel):
    asset: Asset | None = None
    provider: Provider | None = None
    market_type: MarketType | None = None
    interval: Interval | None = None
    evidence_classification: EvidenceClassification | None = None
    score_model_name: str | None = None
    score_model_version: int | None = Field(default=None, ge=1)
    strategy_name: str | None = None
    strategy_version: int | None = Field(default=None, ge=1)
    minimum_validation_runs: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=MAX_OPTIMIZATION_PAGE_SIZE)
    offset: int = Field(default=0, ge=0)


class CandidateProfileResponse(BaseModel):
    evidence_classification: EvidenceClassification
    profile_version: int
    compatible_run_count: int
    validation_evaluation_count: int
    test_evaluation_count: int
    validation_win_rate: Decimal
    test_confirmation_rate: Decimal
    median_validation_return: Decimal | None
    median_test_return: Decimal | None
    median_validation_excess_return: Decimal | None
    median_test_excess_return: Decimal | None
    worst_validation_drawdown: Decimal | None
    worst_test_drawdown: Decimal | None
    median_trade_count: Decimal | None
    median_fees_paid: Decimal | None
    first_evaluated_at: datetime
    last_evaluated_at: datetime
    consecutive_recent_failures: int
    instruments: tuple[InstrumentIdentityResponse, ...]


class CandidateSummaryResponse(BaseModel):
    definition: CandidateDefinitionResponse
    profile: CandidateProfileResponse


class CandidateEvaluationHistoryResponse(EvaluationResponse):
    run_id: UUID
    instrument: InstrumentIdentityResponse
    excess_return_against_baseline: Decimal
    was_recommended: bool


class CandidateDetailResponse(BaseModel):
    definition: CandidateDefinitionResponse
    profile: CandidateProfileResponse
    was_ever_recommended: bool
    related_run_ids: tuple[UUID, ...]
    evaluations: tuple[CandidateEvaluationHistoryResponse, ...]
