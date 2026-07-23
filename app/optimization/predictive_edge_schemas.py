from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from app.optimization.predictive_edge_v4 import (
    DiagnosticConclusion,
    ParameterSource,
    RejectionReason,
)
from app.optimization.predictive_edges import (
    DirectionMapping,
    EvidenceStatus,
    PredictorSemanticKind,
    SuggestedAction,
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
    bin_ordinal: int
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    suggested_action: SuggestedAction
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


class PredictiveEdgeBinSelectionResponse(BaseModel):
    bin_ordinal: int
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    suggested_action: SuggestedAction
    validation_observation_count: int
    validation_gross_return: Decimal
    validation_long_aligned_net_return: Decimal | None
    validation_avoided_downside: Decimal | None
    validation_directional_hit_rate: Decimal | None
    selected_for_test: bool
    test_observation_count: int
    test_confirmation_status: TestConfirmationStatus
    test_gross_return: Decimal
    test_long_aligned_net_return: Decimal | None
    test_avoided_downside: Decimal | None
    test_directional_hit_rate: Decimal | None
    rejection_reason: str | None


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
    bin_selections: tuple[PredictiveEdgeBinSelectionResponse, ...]


class PredictiveEdgeWindowPageResponse(BaseModel):
    items: tuple[PredictiveEdgeWindowMetricResponse, ...]
    next_cursor: str | None
    has_more: bool


class StrategyHypothesisResponse(BaseModel):
    hypothesis_id: str
    predictor_name: str
    predictor_version: int
    hypothesis_contract_version: int
    direction_mapping: DirectionMapping
    bin_ordinal: int
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    suggested_action: SuggestedAction
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
    context_predictor_name: str | None = None
    context_predictor_version: int | None = None
    context_regime: str | None = None
    context_lower_bound: Decimal | None = None
    context_upper_bound: Decimal | None = None


class StrategyHypothesisPageResponse(BaseModel):
    items: tuple[StrategyHypothesisResponse, ...]
    next_cursor: str | None
    has_more: bool


class PredictiveEdgeRunDetailResponse(BaseModel):
    run: PredictiveEdgeRunResponse
    configuration: dict[str, int | Decimal | str]


class AuditedParameterResponse(BaseModel):
    name: str
    value: str
    source: ParameterSource
    rationale: str
    contract_version: int


class RequirementMarginResponse(BaseModel):
    requirement: RejectionReason
    observed_value: Decimal
    required_value: Decimal
    absolute_margin: Decimal
    relative_margin: Decimal | None
    passed: bool


class PredictiveEdgeDiagnosticResponse(BaseModel):
    contract_version: int
    effective_parameters: tuple[AuditedParameterResponse, ...]
    rejection_counts_by_reason: dict[str, int]
    rejection_counts_by_predictor: dict[str, int]
    rejection_counts_by_horizon: dict[str, int]
    rejection_counts_by_condition: dict[str, int]
    rejection_counts_by_neutral_policy: dict[str, int]
    rejection_counts_by_cadence: dict[str, int]
    strongest_observed_signal: str | None
    bottleneck_stage: str
    diagnostic_conclusion: DiagnosticConclusion
    outcome_explanation: str
    stage_timings: dict[str, str]
    computed_counts: dict[str, int]
    reused_counts: dict[str, int]
    regime_conditioned_conclusion: str | None = None
    regime_conditioned_stage_counters: dict[str, object] | None = None
    regime_conditioned_summary: "RegimeConditionedSummaryResponse | None" = None


class RegimeConditionedPrimitiveCountersResponse(BaseModel):
    price_index_builds: int
    sampled_pair_requests: int
    sampled_pair_builds: int
    sampled_pair_reuses: int
    slice_fingerprint_requests: int
    slice_fingerprint_builds: int
    slice_fingerprint_reuses: int


class RegimeConditionedSummaryResponse(BaseModel):
    contract_version: int
    confirmed_interaction_count: int
    actionable_near_miss_count: int
    diagnostic_near_miss_count: int
    hypotheses_generated: int
    primitive_counters: RegimeConditionedPrimitiveCountersResponse | None = None


class RegimeConditionedEvidenceResponse(BaseModel):
    observations: int
    gross_movement: Decimal | None
    aligned_gross_movement: Decimal | None
    net_movement: Decimal | None
    directional_hit_rate: Decimal | None
    benchmark: Decimal | None
    benchmark_improvement: Decimal | None
    standard_error: Decimal | None
    stable: bool
    economic: bool
    passes: bool
    usable_window_count: int
    stable_window_count: int
    positive_aligned_window_count: int
    confirmed_window_count: int


class PredictiveEdgeNearMissResponse(BaseModel):
    id: UUID
    rank: int
    category_rank: int = 1
    engine: str = "v4"
    engine_contract_version: int = 4
    kind: str = "actionable"
    edge_fingerprint: str
    predictor_name: str
    directional_predictor: str | None = None
    context_predictor: str | None = None
    context_regime: str | None = None
    directional_predictor_version: int | None = None
    context_predictor_version: int | None = None
    selected_bin: int | None = None
    directional_lower_bound: Decimal | None = None
    directional_upper_bound: Decimal | None = None
    context_lower_bound: Decimal | None = None
    context_upper_bound: Decimal | None = None
    direction_mapping: str | None = None
    suggested_action: str | None = None
    cadence_seconds: int
    horizon_seconds: int
    bin_count: int
    condition: str
    neutral_policy: str
    profile: str
    classification: str
    primary_rejection_reason: str
    passed_requirements: tuple[str, ...]
    failed_requirements: tuple[str, ...]
    requirement_margins: tuple[RequirementMarginResponse, ...]
    usable_observations: int
    stable_direction_ratio: Decimal
    expected_net_movement: Decimal | None
    expected_gross_movement: Decimal | None = None
    expected_aligned_movement: Decimal | None = None
    directional_hit_rate: Decimal | None = None
    benchmark: Decimal | None = None
    benchmark_improvement: Decimal | None = None
    observations: int | None = None
    normalized_missing_distance: Decimal
    training_evidence: RegimeConditionedEvidenceResponse | None = None
    directional_baseline_evidence: RegimeConditionedEvidenceResponse | None = None
    validation_evidence: RegimeConditionedEvidenceResponse | None = None
    test_evidence: RegimeConditionedEvidenceResponse | None = None
    context_improvement: Decimal | None = None
    usable_window_count: int | None = None
    confirmed_window_count: int | None = None


class PredictiveEdgeNearMissPageResponse(BaseModel):
    items: tuple[PredictiveEdgeNearMissResponse, ...]
    next_cursor: int | None
    has_more: bool


class ThresholdSensitivityResponse(BaseModel):
    threshold_name: RejectionReason
    current_value: Decimal
    nearest_passing_value: Decimal
    candidate_would_advance: bool
    other_requirements_still_failed: tuple[RejectionReason, ...]


class PredictiveEdgeNearMissDetailResponse(BaseModel):
    near_miss: PredictiveEdgeNearMissResponse
    sensitivity: tuple[ThresholdSensitivityResponse, ...]
