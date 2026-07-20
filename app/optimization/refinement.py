from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from enum import StrEnum
from hashlib import sha256
from uuid import UUID

from app.market.catalog import Instrument
from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
    build_historical_feature_matrix,
)
from app.optimization.historical_data import HistoricalDataset
from app.optimization.simulation import (
    DEFAULT_DECISION_INTERVAL,
    HistoricalSimulationResult,
    simulate_historical_feature_matrix,
)
from app.paper.calculation import calculate_open_position, calculate_sell_all
from app.paper.config import (
    DEFAULT_ACCOUNT_CURRENCY,
    SimulationConfiguration,
    quantize_financial,
    validate_initial_cash,
)
from app.paper.models import ExecutionCalculation, PaperPosition, PaperPositionStatus
from app.scores.calculation import MODEL_NAME, MODEL_VERSION
from app.scores.calculator import (
    BASELINE_TECHNICAL_V1_PARAMETERS,
    BaselineTechnicalParameters,
)
from app.strategies.baseline_long_only import (
    BASELINE_LONG_ONLY_V1_PARAMETERS,
    STRATEGY_NAME,
    STRATEGY_VERSION,
    BaselineLongOnlyParameters,
)

DEFAULT_SCORE_SCREENING_TOP_K = 12
DEFAULT_VALIDATION_TOP_K = 10
DEFAULT_MAX_SCORE_CONFIGURATIONS = 200
DEFAULT_MAX_JOINT_CANDIDATES = 1000
MAX_VALIDATION_DRAWDOWN = Decimal("0.10")
BUY_AND_HOLD_CASH_BUFFER_RATE = Decimal("0.000001")
_SIMPLEX_STEP = Decimal("0.25")
_VOLUME_WEIGHTS = tuple(
    Decimal(value) for value in ("0", "0.125", "0.25", "0.375", "0.50")
)
_ENTRY_THRESHOLDS = tuple(Decimal(value) for value in ("0.55", "0.60", "0.65", "0.70"))
_EXIT_THRESHOLDS = tuple(
    Decimal(value) for value in ("0.10", "0.15", "0.20", "0.25", "0.30")
)
_TARGET_PERCENTAGES = tuple(Decimal(value) for value in ("10", "15", "20"))
_WEIGHT_QUANTUM = Decimal("0.000000000000000001")
_ZERO_UUID = UUID(int=0)


class HistoricalRefinementError(ValueError):
    """Raised when historical evidence cannot support deterministic refinement."""


class RefinementSegmentName(StrEnum):
    SCREENING = "screening"
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"


class RecommendationStatus(StrEnum):
    RECOMMENDED_AND_TEST_CONFIRMED = "recommended_and_test_confirmed"
    RECOMMENDED_BUT_TEST_FAILED = "recommended_but_test_failed"
    NO_VALIDATION_CANDIDATE = "no_validation_candidate"


@dataclass(frozen=True, slots=True)
class RefinementBudget:
    score_screening_top_k: int = DEFAULT_SCORE_SCREENING_TOP_K
    validation_top_k: int = DEFAULT_VALIDATION_TOP_K
    max_score_configurations: int = DEFAULT_MAX_SCORE_CONFIGURATIONS
    max_joint_candidates: int = DEFAULT_MAX_JOINT_CANDIDATES

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (
                self.score_screening_top_k,
                self.validation_top_k,
                self.max_score_configurations,
                self.max_joint_candidates,
            )
        ):
            raise ValueError("refinement budget values must be positive integers")
        if self.score_screening_top_k > self.max_score_configurations:
            raise ValueError("score screening top K exceeds score configuration budget")
        if self.validation_top_k > self.max_joint_candidates:
            raise ValueError("validation top K exceeds joint candidate budget")


DEFAULT_REFINEMENT_BUDGET = RefinementBudget()


@dataclass(frozen=True, slots=True)
class RefinementCandidate:
    score_parameters: BaselineTechnicalParameters
    strategy_parameters: BaselineLongOnlyParameters
    is_production_baseline: bool
    stable_candidate_id: str


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
    absolute_pnl: Decimal
    max_drawdown: Decimal
    trade_count: int
    fees_paid: Decimal
    final_equity: Decimal
    return_over_drawdown: Decimal


@dataclass(frozen=True, slots=True)
class BenchmarkSegmentResult:
    segment: RefinementSegmentName
    total_return: Decimal
    absolute_pnl: Decimal
    max_drawdown: Decimal
    trade_count: int
    fees_paid: Decimal
    final_equity: Decimal
    return_over_drawdown: Decimal


@dataclass(frozen=True, slots=True)
class SegmentResults:
    training: CandidateSegmentResult
    validation: CandidateSegmentResult
    test: CandidateSegmentResult


@dataclass(frozen=True, slots=True)
class BenchmarkResults:
    training: BenchmarkSegmentResult
    validation: BenchmarkSegmentResult
    test: BenchmarkSegmentResult


@dataclass(frozen=True, slots=True)
class AblationResult:
    disabled_signal: str
    baseline_score_parameters: BaselineTechnicalParameters
    ablated_score_parameters: BaselineTechnicalParameters
    baseline: CandidateSegmentResult
    ablated: CandidateSegmentResult
    return_difference_from_baseline: Decimal
    drawdown_difference_from_baseline: Decimal
    trade_difference_from_baseline: int


@dataclass(frozen=True, slots=True)
class RefinementCounts:
    score_configurations_generated: int
    score_configurations_screened: int
    score_configurations_retained: int
    joint_candidates_evaluated: int
    validation_candidates_evaluated: int
    test_candidates_evaluated: int
    simulation_count_total: int


@dataclass(frozen=True, slots=True)
class RefinementRecommendation:
    candidate: RefinementCandidate
    validation: CandidateSegmentResult
    test: CandidateSegmentResult
    outperformed_buy_and_hold_test: bool


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
    budget: RefinementBudget
    screening_segment: RefinementSegment
    segments: tuple[RefinementSegment, RefinementSegment, RefinementSegment]
    production_baseline: SegmentResults
    buy_and_hold: BenchmarkResults
    score_screening_ranking: tuple[CandidateSegmentResult, ...]
    retained_score_parameters: tuple[BaselineTechnicalParameters, ...]
    training_ranking: tuple[CandidateSegmentResult, ...]
    validation_results: tuple[CandidateSegmentResult, ...]
    ablation_summary: tuple[AblationResult, ...]
    recommendation: RefinementRecommendation | None
    recommended_score_parameters: BaselineTechnicalParameters | None
    recommended_strategy_parameters: BaselineLongOnlyParameters | None
    recommendation_status: RecommendationStatus
    recommendation_explanation: str
    counts: RefinementCounts


def generate_score_configurations() -> tuple[BaselineTechnicalParameters, ...]:
    """Generate the stable 0.25 simplex grid plus the exact production baseline."""
    directional = tuple(
        tuple(Decimal(value) * _SIMPLEX_STEP for value in (a, b, c, 4 - a - b - c))
        for a in range(5)
        for b in range(5 - a)
        for c in range(5 - a - b)
    )
    generated = tuple(
        BaselineTechnicalParameters(
            weights[0], weights[1], weights[2], weights[3], volume
        )
        for weights in directional
        for volume in _VOLUME_WEIGHTS
    )
    return (
        generated
        if BASELINE_TECHNICAL_V1_PARAMETERS in generated
        else (*generated, BASELINE_TECHNICAL_V1_PARAMETERS)
    )


def generate_strategy_configurations() -> tuple[BaselineLongOnlyParameters, ...]:
    return tuple(
        BaselineLongOnlyParameters(entry, exit_, target)
        for entry in _ENTRY_THRESHOLDS
        for exit_ in _EXIT_THRESHOLDS
        for target in _TARGET_PERCENTAGES
        if exit_ < entry
    )


def candidate_for(
    score: BaselineTechnicalParameters,
    strategy: BaselineLongOnlyParameters,
) -> RefinementCandidate:
    canonical = "|".join(
        _decimal(value)
        for value in (
            score.trend_weight,
            score.rsi_weight,
            score.return_1_weight,
            score.return_12_weight,
            score.volume_confirmation_weight,
            strategy.entry_threshold,
            strategy.exit_threshold,
            strategy.open_position_target_percentage,
        )
    )
    return RefinementCandidate(
        score,
        strategy,
        score == BASELINE_TECHNICAL_V1_PARAMETERS
        and strategy == BASELINE_LONG_ONLY_V1_PARAMETERS,
        sha256(canonical.encode("ascii")).hexdigest(),
    )


def refine_historical_strategy(
    *,
    dataset: HistoricalDataset,
    feature_matrix: HistoricalFeatureMatrix,
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    budget: RefinementBudget = DEFAULT_REFINEMENT_BUDGET,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
) -> HistoricalRefinementResult:
    """Screen score weights, then refine score and Strategy jointly in memory."""
    if feature_matrix.dataset_fingerprint != dataset.fingerprint:
        raise HistoricalRefinementError(
            "Feature matrix fingerprint does not match dataset"
        )
    cash = validate_initial_cash(initial_cash)
    configuration = SimulationConfiguration(
        account_currency=DEFAULT_ACCOUNT_CURRENCY,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    matrices, segments = split_feature_matrix(
        feature_matrix, decision_interval=decision_interval
    )
    screening_matrix, screening_segment = _screening_matrix(
        matrices[RefinementSegmentName.TRAINING], decision_interval
    )
    score_configurations = generate_score_configurations()
    if len(score_configurations) > budget.max_score_configurations:
        raise HistoricalRefinementError(
            "generated score configurations exceed max_score_configurations"
        )
    strategies = generate_strategy_configurations()
    baseline_candidate = candidate_for(
        BASELINE_TECHNICAL_V1_PARAMETERS, BASELINE_LONG_ONLY_V1_PARAMETERS
    )
    simulations = 0

    def evaluate(
        candidate: RefinementCandidate,
        segment: RefinementSegmentName,
        matrix: HistoricalFeatureMatrix | None = None,
    ) -> CandidateSegmentResult:
        nonlocal simulations
        simulations += 1
        simulation = simulate_historical_feature_matrix(
            dataset=dataset,
            feature_matrix=matrix or matrices[segment],
            initial_cash=cash,
            fee_rate=configuration.fee_rate,
            slippage_rate=configuration.slippage_rate,
            decision_interval=decision_interval,
            strategy_parameters=candidate.strategy_parameters,
            score_parameters=candidate.score_parameters,
        )
        return _compact(candidate, segment, simulation)

    screening = tuple(
        sorted(
            (
                evaluate(
                    candidate_for(score, BASELINE_LONG_ONLY_V1_PARAMETERS),
                    RefinementSegmentName.SCREENING,
                    screening_matrix,
                )
                for score in score_configurations
            ),
            key=_ranking_key,
        )
    )
    retained = screening[: min(budget.score_screening_top_k, len(screening))]
    joint = tuple(
        candidate_for(screen.score_parameters, strategy)
        for screen in (item.candidate for item in retained)
        for strategy in strategies
    )
    if len(joint) > budget.max_joint_candidates:
        raise HistoricalRefinementError(
            "generated joint candidates exceed max_joint_candidates"
        )
    training = tuple(
        sorted(
            (
                evaluate(candidate, RefinementSegmentName.TRAINING)
                for candidate in joint
            ),
            key=_ranking_key,
        )
    )
    baseline = SegmentResults(
        evaluate(baseline_candidate, RefinementSegmentName.TRAINING),
        evaluate(baseline_candidate, RefinementSegmentName.VALIDATION),
        evaluate(baseline_candidate, RefinementSegmentName.TEST),
    )
    validation_candidates = training[: min(budget.validation_top_k, len(training))]
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
        and item.total_return > baseline.validation.total_return
        and item.max_drawdown <= MAX_VALIDATION_DRAWDOWN
        and item.trade_count > 0
        and (configuration.fee_rate > 0 or configuration.slippage_rate > 0)
    )
    buy_and_hold = BenchmarkResults(
        _buy_and_hold(
            matrices[RefinementSegmentName.TRAINING],
            RefinementSegmentName.TRAINING,
            dataset.identity.instrument,
            cash,
            configuration,
        ),
        _buy_and_hold(
            matrices[RefinementSegmentName.VALIDATION],
            RefinementSegmentName.VALIDATION,
            dataset.identity.instrument,
            cash,
            configuration,
        ),
        _buy_and_hold(
            matrices[RefinementSegmentName.TEST],
            RefinementSegmentName.TEST,
            dataset.identity.instrument,
            cash,
            configuration,
        ),
    )
    if eligible:
        chosen = eligible[0]
        chosen_test = evaluate(chosen.candidate, RefinementSegmentName.TEST)
        confirmed = (
            chosen_test.total_return > 0
            and chosen_test.total_return > baseline.test.total_return
            and chosen_test.max_drawdown <= MAX_VALIDATION_DRAWDOWN
        )
        recommendation = RefinementRecommendation(
            chosen.candidate,
            chosen,
            chosen_test,
            chosen_test.total_return > buy_and_hold.test.total_return,
        )
        status = (
            RecommendationStatus.RECOMMENDED_AND_TEST_CONFIRMED
            if confirmed
            else RecommendationStatus.RECOMMENDED_BUT_TEST_FAILED
        )
        explanation = _explain(recommendation, baseline, buy_and_hold, confirmed)
    else:
        recommendation = None
        status = RecommendationStatus.NO_VALIDATION_CANDIDATE
        explanation = (
            "No complete score-and-strategy candidate beat the production baseline "
            "with positive net validation return, acceptable drawdown, at least one "
            "trade, and enabled costs. Buy and hold remains a separate historical "
            "benchmark. Historical evidence does not guarantee future results."
        )
    screening_baseline = next(
        item for item in screening if item.candidate.is_production_baseline
    )
    ablation = _controlled_ablation(
        screening_baseline,
        tuple(
            evaluate(
                candidate_for(parameters, BASELINE_LONG_ONLY_V1_PARAMETERS),
                RefinementSegmentName.SCREENING,
                screening_matrix,
            )
            for parameters in generate_ablation_configurations()
        ),
    )
    counts = RefinementCounts(
        len(score_configurations),
        len(screening),
        len(retained),
        len(training),
        len(validation),
        2 if recommendation else 1,
        simulations,
    )
    return HistoricalRefinementResult(
        dataset.fingerprint,
        dataset.identity.instrument,
        MODEL_NAME,
        MODEL_VERSION,
        STRATEGY_NAME,
        STRATEGY_VERSION,
        cash,
        configuration.account_currency,
        configuration.fee_rate,
        configuration.slippage_rate,
        decision_interval,
        budget,
        screening_segment,
        segments,
        baseline,
        buy_and_hold,
        screening,
        tuple(item.candidate.score_parameters for item in retained),
        training,
        validation,
        ablation,
        recommendation,
        recommendation.candidate.score_parameters if recommendation else None,
        recommendation.candidate.strategy_parameters if recommendation else None,
        status,
        explanation,
        counts,
    )


def refine_historical_dataset(
    *,
    dataset: HistoricalDataset,
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    budget: RefinementBudget = DEFAULT_REFINEMENT_BUDGET,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
) -> HistoricalRefinementResult:
    matrix = build_historical_feature_matrix(dataset)
    return refine_historical_strategy(
        dataset=dataset,
        feature_matrix=matrix,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        budget=budget,
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
    names = (
        RefinementSegmentName.TRAINING,
        RefinementSegmentName.VALIDATION,
        RefinementSegmentName.TEST,
    )
    matrices = {
        name: HistoricalFeatureMatrix(feature_matrix.dataset_fingerprint, group)
        for name, group in zip(names, groups, strict=True)
    }
    segments = tuple(
        _segment(name, group) for name, group in zip(names, groups, strict=True)
    )
    return matrices, (segments[0], segments[1], segments[2])


def _screening_matrix(
    training: HistoricalFeatureMatrix, decision_interval: timedelta
) -> tuple[HistoricalFeatureMatrix, RefinementSegment]:
    count = len(training.rows) // 2
    if count == 0:
        raise HistoricalRefinementError(
            "training has no first-half screening boundaries"
        )
    rows = training.rows[:count]
    matrix = HistoricalFeatureMatrix(training.dataset_fingerprint, rows)
    return matrix, _segment(RefinementSegmentName.SCREENING, rows)


def _segment(
    name: RefinementSegmentName, rows: tuple[HistoricalFeatureRow, ...]
) -> RefinementSegment:
    first = rows[0].candle_open_time + timedelta(minutes=5)
    last = rows[-1].candle_open_time + timedelta(minutes=5)
    return RefinementSegment(name, first, last, len(rows))


def _compact(
    candidate: RefinementCandidate,
    segment: RefinementSegmentName,
    result: HistoricalSimulationResult,
) -> CandidateSegmentResult:
    return CandidateSegmentResult(
        candidate,
        segment,
        result.total_return,
        result.absolute_pnl,
        result.max_drawdown,
        result.trade_count,
        result.fees_paid,
        result.final_equity,
        _return_over_drawdown(result.total_return, result.max_drawdown),
    )


def _buy_and_hold(
    matrix: HistoricalFeatureMatrix,
    segment: RefinementSegmentName,
    instrument: Instrument,
    initial_cash: Decimal,
    configuration: SimulationConfiguration,
) -> BenchmarkSegmentResult:
    first, last = matrix.rows[0], matrix.rows[-1]
    opened = _calculate_buy_and_hold_entry(
        cash=initial_cash,
        observed_price=first.close_price,
        configuration=configuration,
    )
    position = PaperPosition(
        _ZERO_UUID,
        _ZERO_UUID,
        asset=instrument.asset,
        provider=instrument.provider,
        market_type=instrument.market_type,
        interval=instrument.interval,
        status=PaperPositionStatus.OPEN,
        quantity=opened.quantity,
        average_entry_price=opened.execution_price,
        total_entry_notional=opened.executed_notional,
        accumulated_entry_fees=opened.fee_amount,
        opened_at=first.candle_open_time + timedelta(minutes=5),
        closed_at=None,
        exit_price=None,
        gross_realized_pnl=None,
        net_realized_pnl=None,
    )
    sold = calculate_sell_all(
        cash=opened.resulting_cash,
        position=position,
        observed_price=last.close_price,
        configuration=configuration,
    )
    equities = tuple(
        quantize_financial(opened.resulting_cash + opened.quantity * row.close_price)
        for row in matrix.rows
    )
    final_equity = sold.resulting_cash
    absolute = quantize_financial(final_equity - initial_cash)
    drawdown = _max_drawdown((initial_cash, *equities, final_equity))
    total_return = quantize_financial(absolute / initial_cash)
    return BenchmarkSegmentResult(
        segment,
        total_return,
        absolute,
        drawdown,
        2,
        quantize_financial(opened.fee_amount + sold.fee_amount),
        final_equity,
        _return_over_drawdown(total_return, drawdown),
    )


def _calculate_buy_and_hold_entry(
    *,
    cash: Decimal,
    observed_price: Decimal,
    configuration: SimulationConfiguration,
) -> ExecutionCalculation:
    """Fund a near-total benchmark purchase with a quantization cash buffer."""
    target = (
        Decimal(100)
        * (Decimal(1) - BUY_AND_HOLD_CASH_BUFFER_RATE)
        / (Decimal(1) + configuration.fee_rate)
    )
    opened = calculate_open_position(
        cash=cash,
        total_equity=cash,
        observed_price=observed_price,
        target_percentage=target,
        configuration=configuration,
    )
    if opened.resulting_cash < 0:
        raise HistoricalRefinementError(
            "buy-and-hold entry produced negative cash after technical buffer"
        )
    return opened


def _max_drawdown(equities: tuple[Decimal, ...]) -> Decimal:
    peak = equities[0]
    maximum = Decimal(0)
    for equity in equities[1:]:
        peak = max(peak, equity)
        drawdown = (
            quantize_financial((peak - equity) / peak) if peak > 0 else Decimal(0)
        )
        maximum = max(maximum, drawdown)
    return maximum


def _return_over_drawdown(total_return: Decimal, drawdown: Decimal) -> Decimal:
    return quantize_financial(total_return / drawdown) if drawdown > 0 else Decimal(0)


def _ranking_key(result: CandidateSegmentResult) -> tuple[Decimal, Decimal, int, str]:
    return (
        -result.total_return,
        result.max_drawdown,
        result.trade_count,
        result.candidate.stable_candidate_id,
    )


def generate_ablation_configurations() -> tuple[BaselineTechnicalParameters, ...]:
    """Disable one signal at a time from the production score configuration.

    For a disabled directional weight ``d``, each remaining weight ``w`` becomes
    ``w / (1 - d)``. Values are quantized with ROUND_HALF_EVEN to 18 places; the
    last enabled weight receives the deterministic residual so the directional
    sum remains exactly one. Volume ablation changes only its own weight.
    """
    baseline = BASELINE_TECHNICAL_V1_PARAMETERS
    directional = (
        baseline.trend_weight,
        baseline.rsi_weight,
        baseline.return_1_weight,
        baseline.return_12_weight,
    )
    configurations = []
    for disabled_index, disabled_weight in enumerate(directional):
        enabled_indexes = tuple(
            index for index in range(len(directional)) if index != disabled_index
        )
        redistributed = [Decimal(0)] * len(directional)
        assigned = Decimal(0)
        for index in enabled_indexes[:-1]:
            weight = (directional[index] / (Decimal(1) - disabled_weight)).quantize(
                _WEIGHT_QUANTUM, rounding=ROUND_HALF_EVEN
            )
            redistributed[index] = weight
            assigned += weight
        redistributed[enabled_indexes[-1]] = Decimal(1) - assigned
        configurations.append(
            BaselineTechnicalParameters(
                redistributed[0],
                redistributed[1],
                redistributed[2],
                redistributed[3],
                baseline.volume_confirmation_weight,
            )
        )
    configurations.append(
        BaselineTechnicalParameters(
            baseline.trend_weight,
            baseline.rsi_weight,
            baseline.return_1_weight,
            baseline.return_12_weight,
            Decimal(0),
        )
    )
    return tuple(configurations)


def _controlled_ablation(
    baseline: CandidateSegmentResult,
    ablated: tuple[CandidateSegmentResult, ...],
) -> tuple[AblationResult, ...]:
    names = ("trend", "rsi", "return_1", "return_12", "volume_confirmation")
    return tuple(
        AblationResult(
            name,
            baseline.candidate.score_parameters,
            result.candidate.score_parameters,
            baseline,
            result,
            result.total_return - baseline.total_return,
            result.max_drawdown - baseline.max_drawdown,
            result.trade_count - baseline.trade_count,
        )
        for name, result in zip(names, ablated, strict=True)
    )


def _explain(
    recommendation: RefinementRecommendation,
    baseline: SegmentResults,
    benchmark: BenchmarkResults,
    confirmed: bool,
) -> str:
    score = recommendation.candidate.score_parameters
    strategy = recommendation.candidate.strategy_parameters
    disabled = (
        ", ".join(
            name
            for name, weight in (
                ("trend", score.trend_weight),
                ("RSI", score.rsi_weight),
                ("return_1", score.return_1_weight),
                ("return_12", score.return_12_weight),
                ("volume confirmation", score.volume_confirmation_weight),
            )
            if weight == 0
        )
        or "none"
    )
    test_word = "confirmed" if confirmed else "did not confirm"
    validation = recommendation.validation
    baseline_score = BASELINE_TECHNICAL_V1_PARAMETERS
    baseline_strategy = BASELINE_LONG_ONLY_V1_PARAMETERS
    return (
        f"Candidate {recommendation.candidate.stable_candidate_id[:12]} changes score "
        f"weights from trend={baseline_score.trend_weight}, "
        f"RSI={baseline_score.rsi_weight}, return_1={baseline_score.return_1_weight}, "
        f"return_12={baseline_score.return_12_weight}, "
        f"volume={baseline_score.volume_confirmation_weight} to "
        f"trend={score.trend_weight}, RSI={score.rsi_weight}, "
        f"return_1={score.return_1_weight}, return_12={score.return_12_weight}, "
        f"volume={score.volume_confirmation_weight}; disabled signals: {disabled}. "
        f"Strategy entry/exit/size changes from "
        f"{baseline_strategy.entry_threshold}/{baseline_strategy.exit_threshold}/"
        f"{baseline_strategy.open_position_target_percentage}% to "
        f"{strategy.entry_threshold}/{strategy.exit_threshold}/"
        f"{strategy.open_position_target_percentage}%. "
        f"In validation net return changed from production baseline "
        f"{baseline.validation.total_return} to {validation.total_return}, "
        f"drawdown from {baseline.validation.max_drawdown} to "
        f"{validation.max_drawdown}, trades from "
        f"{baseline.validation.trade_count} to {validation.trade_count}, "
        f"and fees from {baseline.validation.fees_paid} to "
        f"{validation.fees_paid}. Untouched test {test_word} the "
        f"candidate against production baseline and its return was "
        f"{recommendation.test.total_return} versus buy and hold "
        f"{benchmark.test.total_return}. This is historical evidence, not causality "
        "or a guarantee of future profitability."
    )


def _decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")
