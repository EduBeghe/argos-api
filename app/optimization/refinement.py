from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from app.market.catalog import Instrument
from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    build_historical_feature_matrix,
)
from app.optimization.historical_data import HistoricalDataset
from app.optimization.simulation import (
    DEFAULT_DECISION_INTERVAL,
    HistoricalSimulationResult,
    simulate_historical_feature_matrix,
)
from app.paper.config import (
    DEFAULT_ACCOUNT_CURRENCY,
    SimulationConfiguration,
    validate_initial_cash,
)
from app.scores.calculation import MODEL_NAME, MODEL_VERSION
from app.strategies.baseline_long_only import (
    BASELINE_LONG_ONLY_V1_PARAMETERS,
    STRATEGY_NAME,
    STRATEGY_VERSION,
    BaselineLongOnlyParameters,
)

DEFAULT_VALIDATION_TOP_K = 10
MAX_VALIDATION_DRAWDOWN = Decimal("0.10")
_ENTRY_THRESHOLDS = tuple(Decimal(value) for value in ("0.55", "0.60", "0.65", "0.70"))
_EXIT_THRESHOLDS = tuple(
    Decimal(value) for value in ("0.10", "0.15", "0.20", "0.25", "0.30")
)
_TARGET_PERCENTAGES = tuple(Decimal(value) for value in ("10", "15", "20"))


class HistoricalRefinementError(ValueError):
    """Raised when historical evidence cannot support deterministic refinement."""


class RefinementSegmentName(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"


class RecommendationStatus(StrEnum):
    RECOMMENDED_AND_TEST_CONFIRMED = "recommended_and_test_confirmed"
    RECOMMENDED_BUT_TEST_FAILED = "recommended_but_test_failed"
    NO_VALIDATION_CANDIDATE = "no_validation_candidate"


@dataclass(frozen=True, slots=True)
class RefinementCandidate:
    candidate_index: int
    parameters: BaselineLongOnlyParameters
    is_baseline: bool


@dataclass(frozen=True, slots=True)
class RefinementSegment:
    name: RefinementSegmentName
    first_boundary: datetime
    last_boundary: datetime
    decision_count: int


@dataclass(frozen=True, slots=True)
class CandidateSegmentResult:
    candidate: RefinementCandidate
    segment: RefinementSegmentName
    total_return: Decimal
    max_drawdown: Decimal
    trade_count: int
    fees_paid: Decimal
    final_equity: Decimal


@dataclass(frozen=True, slots=True)
class BaselineSegmentResults:
    training: CandidateSegmentResult
    validation: CandidateSegmentResult
    test: CandidateSegmentResult


@dataclass(frozen=True, slots=True)
class RefinementRecommendation:
    candidate: RefinementCandidate
    validation: CandidateSegmentResult
    test: CandidateSegmentResult


@dataclass(frozen=True, slots=True)
class HistoricalRefinementResult:
    dataset_fingerprint: str
    instrument: Instrument
    score_model_name: str
    score_model_version: int
    strategy_name: str
    strategy_version: int
    initial_cash: Decimal
    account_currency: str
    fee_rate: Decimal
    slippage_rate: Decimal
    decision_interval: timedelta
    segments: tuple[RefinementSegment, RefinementSegment, RefinementSegment]
    total_candidate_count: int
    validation_candidate_count: int
    baseline_parameters: BaselineLongOnlyParameters
    baseline: BaselineSegmentResults
    training_ranking: tuple[CandidateSegmentResult, ...]
    validation_results: tuple[CandidateSegmentResult, ...]
    recommended_parameters: BaselineLongOnlyParameters | None
    recommended_test: CandidateSegmentResult | None
    test_baseline: CandidateSegmentResult
    recommendation: RefinementRecommendation | None
    recommendation_status: RecommendationStatus
    recommendation_explanation: str


def generate_baseline_long_only_candidates() -> tuple[RefinementCandidate, ...]:
    """Return the stable, finite version-1 refinement grid."""
    parameters = tuple(
        BaselineLongOnlyParameters(entry, exit_, target)
        for entry in _ENTRY_THRESHOLDS
        for exit_ in _EXIT_THRESHOLDS
        for target in _TARGET_PERCENTAGES
        if exit_ < entry
    )
    return tuple(
        RefinementCandidate(
            candidate_index=index,
            parameters=value,
            is_baseline=value == BASELINE_LONG_ONLY_V1_PARAMETERS,
        )
        for index, value in enumerate(parameters)
    )


def refine_historical_strategy(
    *,
    dataset: HistoricalDataset,
    feature_matrix: HistoricalFeatureMatrix,
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    validation_top_k: int = DEFAULT_VALIDATION_TOP_K,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
) -> HistoricalRefinementResult:
    """Rank a finite parameter grid without persistence or test-set selection."""
    if validation_top_k <= 0:
        raise HistoricalRefinementError("validation_top_k must be positive")
    effective_initial_cash = validate_initial_cash(initial_cash)
    configuration = SimulationConfiguration(
        account_currency=DEFAULT_ACCOUNT_CURRENCY,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    segment_matrices, segments = split_feature_matrix(
        feature_matrix, decision_interval=decision_interval
    )
    candidates = generate_baseline_long_only_candidates()
    baseline_candidate = next(item for item in candidates if item.is_baseline)
    cache: dict[
        tuple[BaselineLongOnlyParameters, RefinementSegmentName],
        CandidateSegmentResult,
    ] = {}

    def evaluate(
        candidate: RefinementCandidate, segment: RefinementSegmentName
    ) -> CandidateSegmentResult:
        key = (candidate.parameters, segment)
        if key not in cache:
            simulation = simulate_historical_feature_matrix(
                dataset=dataset,
                feature_matrix=segment_matrices[segment],
                initial_cash=effective_initial_cash,
                fee_rate=configuration.fee_rate,
                slippage_rate=configuration.slippage_rate,
                decision_interval=decision_interval,
                strategy_parameters=candidate.parameters,
            )
            cache[key] = _compact(candidate, segment, simulation)
        return cache[key]

    training = tuple(
        sorted(
            (
                evaluate(candidate, RefinementSegmentName.TRAINING)
                for candidate in candidates
            ),
            key=_ranking_key,
        )
    )
    validation_candidates = training[: min(validation_top_k, len(training))]
    validation = tuple(
        sorted(
            (
                evaluate(item.candidate, RefinementSegmentName.VALIDATION)
                for item in validation_candidates
            ),
            key=_ranking_key,
        )
    )
    eligible = tuple(
        item
        for item in validation
        if item.total_return > 0
        and item.max_drawdown <= MAX_VALIDATION_DRAWDOWN
        and item.trade_count > 0
        and (configuration.fee_rate > 0 or configuration.slippage_rate > 0)
    )
    baseline = BaselineSegmentResults(
        evaluate(baseline_candidate, RefinementSegmentName.TRAINING),
        evaluate(baseline_candidate, RefinementSegmentName.VALIDATION),
        evaluate(baseline_candidate, RefinementSegmentName.TEST),
    )
    if not eligible:
        status = RecommendationStatus.NO_VALIDATION_CANDIDATE
        recommendation = None
        explanation = (
            "No candidate satisfied the validation requirements for positive net "
            "return, drawdown at or below 10%, at least one trade, and enabled "
            "cost evidence. Historical evidence does not guarantee future results."
        )
    else:
        selected_validation = eligible[0]
        selected_test = evaluate(
            selected_validation.candidate, RefinementSegmentName.TEST
        )
        recommendation = RefinementRecommendation(
            selected_validation.candidate,
            selected_validation,
            selected_test,
        )
        confirmed = (
            selected_test.total_return > 0
            and selected_test.total_return > baseline.test.total_return
        )
        status = (
            RecommendationStatus.RECOMMENDED_AND_TEST_CONFIRMED
            if confirmed
            else RecommendationStatus.RECOMMENDED_BUT_TEST_FAILED
        )
        explanation = _explain(
            selected_validation,
            baseline.validation,
            selected_test,
            baseline.test,
            confirmed=confirmed,
        )
    return HistoricalRefinementResult(
        dataset.fingerprint,
        dataset.identity.instrument,
        MODEL_NAME,
        MODEL_VERSION,
        STRATEGY_NAME,
        STRATEGY_VERSION,
        effective_initial_cash,
        configuration.account_currency,
        configuration.fee_rate,
        configuration.slippage_rate,
        decision_interval,
        segments,
        len(candidates),
        len(validation),
        BASELINE_LONG_ONLY_V1_PARAMETERS,
        baseline,
        training,
        validation,
        recommendation.candidate.parameters if recommendation else None,
        recommendation.test if recommendation else None,
        baseline.test,
        recommendation,
        status,
        explanation,
    )


def refine_historical_dataset(
    *,
    dataset: HistoricalDataset,
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    validation_top_k: int = DEFAULT_VALIDATION_TOP_K,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
) -> HistoricalRefinementResult:
    """Build one Feature matrix and refine the historical dataset in memory."""
    feature_matrix = build_historical_feature_matrix(dataset)
    return refine_historical_strategy(
        dataset=dataset,
        feature_matrix=feature_matrix,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        validation_top_k=validation_top_k,
        decision_interval=decision_interval,
    )


def split_feature_matrix(
    feature_matrix: HistoricalFeatureMatrix,
    *,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
) -> tuple[
    dict[RefinementSegmentName, HistoricalFeatureMatrix],
    tuple[RefinementSegment, RefinementSegment, RefinementSegment],
]:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    rows = tuple(
        row
        for row in feature_matrix.rows
        if (row.candle_open_time + timedelta(minutes=5) - epoch) % decision_interval
        == timedelta(0)
    )
    training_end = len(rows) * 60 // 100
    validation_end = training_end + len(rows) * 20 // 100
    groups = (
        rows[:training_end],
        rows[training_end:validation_end],
        rows[validation_end:],
    )
    if any(not group for group in groups):
        raise HistoricalRefinementError(
            "at least five decision boundaries are required for 60/20/20 refinement"
        )
    names = tuple(RefinementSegmentName)
    matrices = {
        name: HistoricalFeatureMatrix(feature_matrix.dataset_fingerprint, group)
        for name, group in zip(names, groups, strict=True)
    }
    segments = tuple(
        RefinementSegment(
            name,
            group[0].candle_open_time + timedelta(minutes=5),
            group[-1].candle_open_time + timedelta(minutes=5),
            len(group),
        )
        for name, group in zip(names, groups, strict=True)
    )
    return matrices, (segments[0], segments[1], segments[2])


def _compact(
    candidate: RefinementCandidate,
    segment: RefinementSegmentName,
    result: HistoricalSimulationResult,
) -> CandidateSegmentResult:
    return CandidateSegmentResult(
        candidate,
        segment,
        result.total_return,
        result.max_drawdown,
        result.trade_count,
        result.fees_paid,
        result.final_equity,
    )


def _ranking_key(
    result: CandidateSegmentResult,
) -> tuple[Decimal, Decimal, int, Decimal, Decimal, Decimal]:
    parameters = result.candidate.parameters
    return (
        -result.total_return,
        result.max_drawdown,
        result.trade_count,
        parameters.entry_threshold,
        parameters.exit_threshold,
        parameters.open_position_target_percentage,
    )


def _explain(
    candidate: CandidateSegmentResult,
    baseline: CandidateSegmentResult,
    candidate_test: CandidateSegmentResult,
    baseline_test: CandidateSegmentResult,
    *,
    confirmed: bool,
) -> str:
    parameters = candidate.candidate.parameters
    test_result = "outperformed" if confirmed else "underperformed"
    return (
        f"Candidate {parameters.entry_threshold}/{parameters.exit_threshold}/"
        f"{parameters.open_position_target_percentage}% is recommended because "
        f"validation net return changed from {baseline.total_return} to "
        f"{candidate.total_return}, trades changed from {baseline.trade_count} to "
        f"{candidate.trade_count}, and max drawdown changed from "
        f"{baseline.max_drawdown} to {candidate.max_drawdown}. On the untouched test "
        f"segment it {test_result} baseline ({candidate_test.total_return} versus "
        f"{baseline_test.total_return}). This is historical evidence, not a guarantee "
        "of future profitability."
    )
