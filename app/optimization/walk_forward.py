import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from enum import StrEnum
from hashlib import sha256
from statistics import median

from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.historical_data import HistoricalDataset
from app.optimization.refinement import (
    BenchmarkSegmentResult,
    RefinementCandidate,
    RefinementSegmentName,
    candidate_for,
    simulate_buy_and_hold,
)
from app.optimization.simulation import (
    DEFAULT_DECISION_INTERVAL,
    HistoricalSimulationError,
    HistoricalSimulationResult,
    simulate_historical_feature_matrix,
)
from app.paper.calculation import PaperExecutionError
from app.paper.config import (
    DEFAULT_ACCOUNT_CURRENCY,
    SimulationConfiguration,
    quantize_financial,
    validate_initial_cash,
)
from app.scores.calculator import BASELINE_TECHNICAL_V1_PARAMETERS
from app.strategies.baseline_long_only import BASELINE_LONG_ONLY_V1_PARAMETERS

WALK_FORWARD_ALGORITHM_VERSION = 1
DEFAULT_TRAINING_DAYS = 180
DEFAULT_VALIDATION_DAYS = 60
DEFAULT_TEST_DAYS = 60
DEFAULT_STEP_DAYS = 60
DEFAULT_MAXIMUM_WINDOWS = 6
DEFAULT_MAXIMUM_CANDIDATES = 10
DEFAULT_MINIMUM_VALID_WINDOWS = 3
DEFAULT_MINIMUM_PROFITABLE_RATIO = Decimal("0.60")
DEFAULT_MAXIMUM_DRAWDOWN = Decimal("0.20")
DEFAULT_MAXIMUM_RETURN_STANDARD_DEVIATION = Decimal("0.20")
MAXIMUM_WINDOW_LIMIT = 24
MAXIMUM_CANDIDATE_LIMIT = 50
MAXIMUM_DURATION_DAYS = 3650
RATIO_QUANTUM = Decimal("0.000000000000000001")


class WalkForwardError(ValueError):
    """Raised when walk-forward evidence or configuration is invalid."""


class WalkForwardConclusion(StrEnum):
    COMPLETED = "completed"
    INSUFFICIENT_HISTORY = "insufficient_history"


class WalkForwardEvaluationStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class WalkForwardClassification(StrEnum):
    STABLE_CANDIDATE = "stable_candidate"
    UNSTABLE_CANDIDATE = "unstable_candidate"
    CONSISTENTLY_UNPROFITABLE = "consistently_unprofitable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    EVALUATION_FAILED = "evaluation_failed"


@dataclass(frozen=True, slots=True)
class WalkForwardConfiguration:
    training_days: int = DEFAULT_TRAINING_DAYS
    validation_days: int = DEFAULT_VALIDATION_DAYS
    test_days: int = DEFAULT_TEST_DAYS
    step_days: int = DEFAULT_STEP_DAYS
    maximum_windows: int = DEFAULT_MAXIMUM_WINDOWS
    maximum_candidates: int = DEFAULT_MAXIMUM_CANDIDATES
    minimum_valid_windows: int = DEFAULT_MINIMUM_VALID_WINDOWS
    minimum_profitable_window_ratio: Decimal = DEFAULT_MINIMUM_PROFITABLE_RATIO
    maximum_drawdown: Decimal = DEFAULT_MAXIMUM_DRAWDOWN
    maximum_return_standard_deviation: Decimal = (
        DEFAULT_MAXIMUM_RETURN_STANDARD_DEVIATION
    )

    def __post_init__(self) -> None:
        integers = (
            self.training_days,
            self.validation_days,
            self.test_days,
            self.step_days,
            self.maximum_windows,
            self.maximum_candidates,
            self.minimum_valid_windows,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integers
        ):
            raise WalkForwardError(
                "walk-forward durations and limits must be positive integers"
            )
        if self.minimum_valid_windows > self.maximum_windows:
            raise WalkForwardError(
                "minimum valid windows cannot exceed maximum windows"
            )
        if (
            max(
                self.training_days,
                self.validation_days,
                self.test_days,
                self.step_days,
            )
            > MAXIMUM_DURATION_DAYS
        ):
            raise WalkForwardError(
                f"walk-forward durations cannot exceed {MAXIMUM_DURATION_DAYS} days"
            )
        if self.maximum_windows > MAXIMUM_WINDOW_LIMIT:
            raise WalkForwardError(
                f"maximum windows cannot exceed {MAXIMUM_WINDOW_LIMIT}"
            )
        if self.maximum_candidates > MAXIMUM_CANDIDATE_LIMIT:
            raise WalkForwardError(
                f"maximum candidates cannot exceed {MAXIMUM_CANDIDATE_LIMIT}"
            )
        if not Decimal(0) < self.minimum_profitable_window_ratio <= Decimal(1):
            raise WalkForwardError("minimum profitable window ratio must be in (0, 1]")
        if self.maximum_drawdown < 0:
            raise WalkForwardError("maximum drawdown must not be negative")
        if self.maximum_return_standard_deviation < 0:
            raise WalkForwardError(
                "maximum return standard deviation must not be negative"
            )


DEFAULT_WALK_FORWARD_CONFIGURATION = WalkForwardConfiguration()


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    ordinal: int
    training_from: datetime
    training_to: datetime
    validation_from: datetime
    validation_to: datetime
    test_from: datetime
    test_to: datetime
    decision_count: int
    dataset_fingerprint: str


@dataclass(frozen=True, slots=True)
class WalkForwardMetrics:
    total_return: Decimal
    absolute_pnl: Decimal
    max_drawdown: Decimal
    final_equity: Decimal
    fees_paid: Decimal
    trade_count: int
    return_over_drawdown: Decimal


@dataclass(frozen=True, slots=True)
class WalkForwardCandidateEvaluation:
    window: WalkForwardWindow
    candidate: RefinementCandidate
    status: WalkForwardEvaluationStatus
    metrics: WalkForwardMetrics | None
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class WalkForwardBenchmarkEvaluation:
    window: WalkForwardWindow
    benchmark: str
    status: WalkForwardEvaluationStatus
    metrics: WalkForwardMetrics | None
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class WalkForwardStability:
    candidate: RefinementCandidate
    classification: WalkForwardClassification
    windows_evaluated: int
    profitable_windows: int
    losing_windows: int
    profitable_window_ratio: Decimal
    mean_return: Decimal
    median_return: Decimal
    worst_window_return: Decimal
    best_window_return: Decimal
    return_standard_deviation: Decimal
    mean_max_drawdown: Decimal
    worst_max_drawdown: Decimal
    aggregate_fees_paid: Decimal
    aggregate_trade_count: int
    positive_excess_return_vs_production_ratio: Decimal
    positive_excess_return_vs_buy_and_hold_ratio: Decimal
    explanation: str


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    conclusion: WalkForwardConclusion
    configuration: WalkForwardConfiguration
    windows: tuple[WalkForwardWindow, ...]
    candidate_evaluations: tuple[WalkForwardCandidateEvaluation, ...]
    benchmark_evaluations: tuple[WalkForwardBenchmarkEvaluation, ...]
    stability: tuple[WalkForwardStability, ...]
    recommended_candidate_id: str | None
    recommendation_explanation: str
    evaluations_computed: int
    evaluations_reused: int
    compute_avoided_estimate: int


def completed_window_count(result: WalkForwardResult) -> int:
    """Count windows whose candidate and benchmark evaluations all completed."""
    completed = 0
    for window in result.windows:
        candidate_items = tuple(
            item
            for item in result.candidate_evaluations
            if item.window.ordinal == window.ordinal
        )
        benchmark_items = tuple(
            item
            for item in result.benchmark_evaluations
            if item.window.ordinal == window.ordinal
        )
        if (
            candidate_items
            and benchmark_items
            and all(
                item.status is WalkForwardEvaluationStatus.COMPLETED
                for item in candidate_items
            )
            and all(
                item.status is WalkForwardEvaluationStatus.COMPLETED
                for item in benchmark_items
            )
        ):
            completed += 1
    return completed


def build_walk_forward_windows(
    *,
    dataset: HistoricalDataset,
    feature_matrix: HistoricalFeatureMatrix,
    configuration: WalkForwardConfiguration,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
) -> tuple[WalkForwardWindow, ...]:
    """Build deterministic half-open chronological windows without future overlap."""
    if feature_matrix.dataset_fingerprint != dataset.fingerprint:
        raise WalkForwardError("Feature matrix fingerprint does not match dataset")
    start = dataset.identity.range.start
    end = dataset.identity.range.end
    if start.tzinfo is not UTC or end.tzinfo is not UTC:
        raise WalkForwardError("walk-forward dataset bounds must use UTC")
    training = timedelta(days=configuration.training_days)
    validation = timedelta(days=configuration.validation_days)
    test = timedelta(days=configuration.test_days)
    step = timedelta(days=configuration.step_days)
    windows: list[WalkForwardWindow] = []
    training_from = start
    while len(windows) < configuration.maximum_windows:
        training_to = training_from + training
        validation_to = training_to + validation
        test_to = validation_to + test
        if test_to > end:
            break
        test_rows = _test_rows(
            feature_matrix, validation_to, test_to, decision_interval
        )
        decision_count = len(test_rows)
        windows.append(
            WalkForwardWindow(
                ordinal=len(windows) + 1,
                training_from=training_from,
                training_to=training_to,
                validation_from=training_to,
                validation_to=validation_to,
                test_from=validation_to,
                test_to=test_to,
                decision_count=decision_count,
                dataset_fingerprint=_window_fingerprint(
                    dataset.fingerprint,
                    training_from,
                    training_to,
                    validation_to,
                    test_to,
                ),
            )
        )
        training_from += step
    return tuple(windows)


def evaluate_walk_forward(
    *,
    dataset: HistoricalDataset,
    feature_matrix: HistoricalFeatureMatrix,
    candidates: tuple[RefinementCandidate, ...],
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    configuration: WalkForwardConfiguration = DEFAULT_WALK_FORWARD_CONFIGURATION,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
) -> WalkForwardResult:
    """Evaluate fixed candidates on each window's isolated out-of-sample test range."""
    if len(candidates) > configuration.maximum_candidates:
        raise WalkForwardError("candidate count exceeds walk-forward maximum")
    unique = {candidate.stable_candidate_id: candidate for candidate in candidates}
    if len(unique) != len(candidates):
        raise WalkForwardError("walk-forward candidates must be unique")
    baseline = candidate_for(
        BASELINE_TECHNICAL_V1_PARAMETERS, BASELINE_LONG_ONLY_V1_PARAMETERS
    )
    ordered = (
        baseline,
        *(candidate for candidate in candidates if candidate != baseline),
    )[: configuration.maximum_candidates]
    windows = build_walk_forward_windows(
        dataset=dataset,
        feature_matrix=feature_matrix,
        configuration=configuration,
        decision_interval=decision_interval,
    )
    if len(windows) < configuration.minimum_valid_windows:
        return WalkForwardResult(
            WalkForwardConclusion.INSUFFICIENT_HISTORY,
            configuration,
            windows,
            (),
            (),
            (),
            None,
            (
                f"Only {len(windows)} valid windows could be built; "
                f"{configuration.minimum_valid_windows} are required. No "
                "walk-forward recommendation was produced."
            ),
            0,
            0,
            0,
        )
    cash = validate_initial_cash(initial_cash)
    paper_configuration = SimulationConfiguration(
        account_currency=DEFAULT_ACCOUNT_CURRENCY,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    candidate_evaluations: list[WalkForwardCandidateEvaluation] = []
    benchmarks: list[WalkForwardBenchmarkEvaluation] = []
    for window in windows:
        matrix = HistoricalFeatureMatrix(
            dataset.fingerprint,
            _test_rows(
                feature_matrix,
                window.test_from,
                window.test_to,
                decision_interval,
            ),
        )
        for candidate in ordered:
            try:
                simulation = simulate_historical_feature_matrix(
                    dataset=dataset,
                    feature_matrix=matrix,
                    initial_cash=cash,
                    fee_rate=paper_configuration.fee_rate,
                    slippage_rate=paper_configuration.slippage_rate,
                    decision_interval=decision_interval,
                    strategy_parameters=candidate.strategy_parameters,
                    score_parameters=candidate.score_parameters,
                )
                candidate_evaluations.append(
                    WalkForwardCandidateEvaluation(
                        window,
                        candidate,
                        WalkForwardEvaluationStatus.COMPLETED,
                        _simulation_metrics(simulation),
                        None,
                    )
                )
            except (HistoricalSimulationError, PaperExecutionError) as error:
                candidate_evaluations.append(
                    WalkForwardCandidateEvaluation(
                        window,
                        candidate,
                        WalkForwardEvaluationStatus.FAILED,
                        None,
                        _sanitize_failure(error),
                    )
                )
        try:
            buy_and_hold = simulate_buy_and_hold(
                matrix,
                RefinementSegmentName.TEST,
                dataset.identity.instrument,
                cash,
                paper_configuration,
            )
            benchmarks.append(
                WalkForwardBenchmarkEvaluation(
                    window,
                    "buy_and_hold",
                    WalkForwardEvaluationStatus.COMPLETED,
                    _benchmark_metrics(buy_and_hold),
                    None,
                )
            )
        except (HistoricalSimulationError, ValueError) as error:
            benchmarks.append(
                WalkForwardBenchmarkEvaluation(
                    window,
                    "buy_and_hold",
                    WalkForwardEvaluationStatus.FAILED,
                    None,
                    _sanitize_failure(error),
                )
            )
    stability = tuple(
        _aggregate(
            candidate,
            windows,
            tuple(candidate_evaluations),
            tuple(benchmarks),
            baseline.stable_candidate_id,
            configuration,
        )
        for candidate in ordered
    )
    stable = tuple(
        item
        for item in stability
        if item.classification is WalkForwardClassification.STABLE_CANDIDATE
        and not item.candidate.is_production_baseline
    )
    recommended = (
        sorted(
            stable,
            key=lambda item: (
                -item.median_return,
                item.worst_max_drawdown,
                item.return_standard_deviation,
                item.candidate.stable_candidate_id,
            ),
        )[0]
        if stable
        else None
    )
    explanation = (
        _recommendation_explanation(recommended)
        if recommended is not None
        else (
            "No candidate satisfied every version-1 stability criterion across "
            "multiple test windows. No production change is recommended. Historical "
            "evidence does not guarantee future results."
        )
    )
    computed = len(candidate_evaluations) + len(benchmarks)
    return WalkForwardResult(
        WalkForwardConclusion.COMPLETED,
        configuration,
        windows,
        tuple(candidate_evaluations),
        tuple(benchmarks),
        stability,
        recommended.candidate.stable_candidate_id if recommended else None,
        explanation,
        computed,
        0,
        0,
    )


def _aggregate(
    candidate: RefinementCandidate,
    windows: tuple[WalkForwardWindow, ...],
    evaluations: tuple[WalkForwardCandidateEvaluation, ...],
    benchmarks: tuple[WalkForwardBenchmarkEvaluation, ...],
    baseline_id: str,
    configuration: WalkForwardConfiguration,
) -> WalkForwardStability:
    selected = tuple(
        item
        for item in evaluations
        if item.candidate.stable_candidate_id == candidate.stable_candidate_id
        and item.status is WalkForwardEvaluationStatus.COMPLETED
        and item.metrics is not None
    )
    if len(selected) < configuration.minimum_valid_windows:
        classification = (
            WalkForwardClassification.EVALUATION_FAILED
            if len(selected) < len(windows)
            else WalkForwardClassification.INSUFFICIENT_EVIDENCE
        )
        return _empty_stability(candidate, classification, len(selected))
    metrics = tuple(item.metrics for item in selected if item.metrics is not None)
    returns = tuple(item.total_return for item in metrics)
    drawdowns = tuple(item.max_drawdown for item in metrics)
    profitable = sum(value > 0 for value in returns)
    ratio = _ratio(profitable, len(returns))
    mean_return = _mean(returns)
    median_return = quantize_financial(Decimal(median(returns)))
    deviation = _standard_deviation(returns, mean_return)
    baseline_by_window = _candidate_metrics_by_window(evaluations, baseline_id)
    buy_hold_by_window = {
        item.window.ordinal: item.metrics
        for item in benchmarks
        if item.metrics is not None
    }
    positive_baseline = sum(
        item.metrics is not None
        and (baseline_metric := baseline_by_window.get(item.window.ordinal)) is not None
        and item.metrics.total_return > baseline_metric.total_return
        for item in selected
    )
    positive_buy_hold = sum(
        item.metrics is not None
        and (benchmark := buy_hold_by_window.get(item.window.ordinal)) is not None
        and item.metrics.total_return > benchmark.total_return
        for item in selected
    )
    if profitable == 0:
        classification = WalkForwardClassification.CONSISTENTLY_UNPROFITABLE
    elif (
        profitable >= 2
        and ratio >= configuration.minimum_profitable_window_ratio
        and max(drawdowns) <= configuration.maximum_drawdown
        and sum(item.trade_count for item in metrics) > 0
        and deviation <= configuration.maximum_return_standard_deviation
    ):
        classification = WalkForwardClassification.STABLE_CANDIDATE
    else:
        classification = WalkForwardClassification.UNSTABLE_CANDIDATE
    explanation = (
        f"{classification.value}: profitable in {profitable} of {len(returns)} "
        f"windows; worst return {min(returns)}; worst drawdown {max(drawdowns)}; "
        f"beat production in {positive_baseline} windows and buy-and-hold in "
        f"{positive_buy_hold}. Historical evidence does not guarantee future results."
    )
    return WalkForwardStability(
        candidate,
        classification,
        len(returns),
        profitable,
        len(returns) - profitable,
        ratio,
        mean_return,
        median_return,
        min(returns),
        max(returns),
        deviation,
        _mean(drawdowns),
        max(drawdowns),
        quantize_financial(sum((item.fees_paid for item in metrics), Decimal(0))),
        sum(item.trade_count for item in metrics),
        _ratio(positive_baseline, len(returns)),
        _ratio(positive_buy_hold, len(returns)),
        explanation,
    )


def _empty_stability(
    candidate: RefinementCandidate,
    classification: WalkForwardClassification,
    windows_evaluated: int,
) -> WalkForwardStability:
    zero = Decimal(0)
    return WalkForwardStability(
        candidate,
        classification,
        windows_evaluated,
        0,
        0,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        0,
        zero,
        zero,
        (
            f"{classification.value}: only {windows_evaluated} valid evaluations "
            "were available. Historical evidence does not guarantee future results."
        ),
    )


def _test_rows(
    matrix: HistoricalFeatureMatrix,
    start: datetime,
    end: datetime,
    decision_interval: timedelta,
) -> tuple[HistoricalFeatureRow, ...]:
    candle_duration = timedelta(minutes=5)
    return tuple(
        row
        for row in matrix.rows
        if start <= row.candle_open_time + candle_duration < end
        and (row.candle_open_time + candle_duration - datetime(1970, 1, 1, tzinfo=UTC))
        % decision_interval
        == timedelta(0)
    )


def _window_fingerprint(
    parent: str,
    training_from: datetime,
    training_to: datetime,
    validation_to: datetime,
    test_to: datetime,
) -> str:
    payload = (
        parent,
        training_from.isoformat(),
        training_to.isoformat(),
        validation_to.isoformat(),
        test_to.isoformat(),
        WALK_FORWARD_ALGORITHM_VERSION,
    )
    return sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _simulation_metrics(result: HistoricalSimulationResult) -> WalkForwardMetrics:
    return WalkForwardMetrics(
        total_return=result.total_return,
        absolute_pnl=result.absolute_pnl,
        max_drawdown=result.max_drawdown,
        final_equity=result.final_equity,
        fees_paid=result.fees_paid,
        trade_count=result.trade_count,
        return_over_drawdown=(
            quantize_financial(result.total_return / result.max_drawdown)
            if result.max_drawdown > 0
            else Decimal(0)
        ),
    )


def _benchmark_metrics(result: BenchmarkSegmentResult) -> WalkForwardMetrics:
    return WalkForwardMetrics(
        result.total_return,
        result.absolute_pnl,
        result.max_drawdown,
        result.final_equity,
        result.fees_paid,
        result.trade_count,
        result.return_over_drawdown,
    )


def _candidate_metrics_by_window(
    evaluations: tuple[WalkForwardCandidateEvaluation, ...], candidate_id: str
) -> dict[int, WalkForwardMetrics]:
    return {
        item.window.ordinal: item.metrics
        for item in evaluations
        if item.candidate.stable_candidate_id == candidate_id
        and item.metrics is not None
    }


def _mean(values: tuple[Decimal, ...]) -> Decimal:
    return quantize_financial(sum(values, Decimal(0)) / Decimal(len(values)))


def _standard_deviation(values: tuple[Decimal, ...], mean: Decimal) -> Decimal:
    variance = sum(((value - mean) ** 2 for value in values), Decimal(0)) / Decimal(
        len(values)
    )
    return variance.sqrt().quantize(RATIO_QUANTUM, rounding=ROUND_HALF_EVEN)


def _ratio(numerator: int, denominator: int) -> Decimal:
    return (
        (Decimal(numerator) / Decimal(denominator)).quantize(
            RATIO_QUANTUM, rounding=ROUND_HALF_EVEN
        )
        if denominator
        else Decimal(0)
    )


def _sanitize_failure(error: Exception) -> str:
    return type(error).__name__[:128]


def _recommendation_explanation(value: WalkForwardStability) -> str:
    return (
        f"Candidate {value.candidate.stable_candidate_id} is decision-support only: "
        f"profitable in {value.profitable_windows} of {value.windows_evaluated} "
        f"windows, worst return {value.worst_window_return}, maximum drawdown "
        f"{value.worst_max_drawdown}, production outperformance ratio "
        f"{value.positive_excess_return_vs_production_ratio}, and buy-and-hold "
        f"outperformance ratio {value.positive_excess_return_vs_buy_and_hold_ratio}. "
        "It is not promoted automatically. Historical evidence does not guarantee "
        "future results."
    )

