
import json
from dataclasses import asdict, dataclass
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
    RefinementBudget,
    RefinementCandidate,
    RefinementSegmentName,
    candidate_for,
    generate_score_configurations,
    generate_strategy_configurations,
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
from app.strategies.baseline_long_only import (
    BASELINE_LONG_ONLY_V1_PARAMETERS,
)

WALK_FORWARD_ALGORITHM_VERSION = 2
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
        if max(integers[:4]) > MAXIMUM_DURATION_DAYS:
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
    selection_fingerprint: str


@dataclass(frozen=True, slots=True)
class WalkForwardMatrices:
    training: HistoricalFeatureMatrix
    validation: HistoricalFeatureMatrix
    test: HistoricalFeatureMatrix


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
class WalkForwardCandidateSelection:
    window: WalkForwardWindow
    candidate: RefinementCandidate
    validation_rank: int
    selected_for_test: bool
    selection_reason: str
    validation_metrics: WalkForwardMetrics


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
    selections: tuple[WalkForwardCandidateSelection, ...]
    candidate_evaluations: tuple[WalkForwardCandidateEvaluation, ...]
    benchmark_evaluations: tuple[WalkForwardBenchmarkEvaluation, ...]
    stability: tuple[WalkForwardStability, ...]
    recommended_candidate_id: str | None
    recommendation_explanation: str
    evaluations_computed: int
    evaluations_reused: int
    compute_avoided_estimate: int


def completed_window_count(result: WalkForwardResult) -> int:
    completed = 0
    for window in result.windows:
        selected = tuple(
            item
            for item in result.selections
            if item.window.ordinal == window.ordinal and item.selected_for_test
        )
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
            selected
            and len(candidate_items) == len(selected)
            and len(benchmark_items) == 2
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
    refinement_budget: RefinementBudget,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
) -> tuple[WalkForwardWindow, ...]:
    """Build only the most recent valid half-open chronological windows."""
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
    complete_span = training + validation + test
    available = end - start - complete_span
    if available < timedelta(0):
        return ()
    valid_count = int(available // step) + 1
    first_index = max(0, valid_count - configuration.maximum_windows)
    selected_count = valid_count - first_index

    windows: list[WalkForwardWindow] = []
    for ordinal, index in enumerate(range(first_index, valid_count), start=1):
        training_from = start + index * step
        training_to = training_from + training
        validation_to = training_to + validation
        test_to = validation_to + test
        ranges = (
            (training_from, training_to),
            (training_to, validation_to),
            (validation_to, test_to),
        )
        matrices = _matrices_for_ranges(feature_matrix, ranges, decision_interval)
        selection_fingerprint = _selection_fingerprint(
            matrices.training,
            matrices.validation,
            configuration,
            refinement_budget,
            decision_interval,
        )
        dataset_fingerprint = _window_fingerprint(
            selection_fingerprint,
            matrices.test,
            ranges,
        )
        windows.append(
            WalkForwardWindow(
                ordinal=ordinal,
                training_from=training_from,
                training_to=training_to,
                validation_from=training_to,
                validation_to=validation_to,
                test_from=validation_to,
                test_to=test_to,
                decision_count=len(matrices.test.rows),
                dataset_fingerprint=dataset_fingerprint,
                selection_fingerprint=selection_fingerprint,
            )
        )
    if len(windows) != selected_count:
        raise WalkForwardError("walk-forward recent-window selection is inconsistent")
    return tuple(windows)


def build_window_matrices(
    feature_matrix: HistoricalFeatureMatrix,
    window: WalkForwardWindow,
    *,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
) -> WalkForwardMatrices:
    return _matrices_for_ranges(
        feature_matrix,
        (
            (window.training_from, window.training_to),
            (window.validation_from, window.validation_to),
            (window.test_from, window.test_to),
        ),
        decision_interval,
    )


def evaluate_walk_forward(
    *,
    dataset: HistoricalDataset,
    feature_matrix: HistoricalFeatureMatrix,
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    refinement_budget: RefinementBudget,
    configuration: WalkForwardConfiguration = DEFAULT_WALK_FORWARD_CONFIGURATION,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
) -> WalkForwardResult:
    """Select within each window, then evaluate only its isolated test range."""
    windows = build_walk_forward_windows(
        dataset=dataset,
        feature_matrix=feature_matrix,
        configuration=configuration,
        refinement_budget=refinement_budget,
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
    selections: list[WalkForwardCandidateSelection] = []
    evaluations: list[WalkForwardCandidateEvaluation] = []
    benchmarks: list[WalkForwardBenchmarkEvaluation] = []
    computations = 0
    baseline = candidate_for(
        BASELINE_TECHNICAL_V1_PARAMETERS, BASELINE_LONG_ONLY_V1_PARAMETERS
    )
    for window in windows:
        matrices = build_window_matrices(
            feature_matrix, window, decision_interval=decision_interval
        )
        window_selections, selection_computations = _select_candidates(
            dataset=dataset,
            window=window,
            matrices=matrices,
            initial_cash=cash,
            fee_rate=paper_configuration.fee_rate,
            slippage_rate=paper_configuration.slippage_rate,
            decision_interval=decision_interval,
            refinement_budget=refinement_budget,
            maximum_candidates=configuration.maximum_candidates,
        )
        selections.extend(window_selections)
        computations += selection_computations
        for selection in window_selections:
            if not selection.selected_for_test:
                continue
            evaluation = _evaluate_candidate(
                dataset,
                matrices.test,
                window,
                selection.candidate,
                cash,
                paper_configuration,
                decision_interval,
            )
            evaluations.append(evaluation)
            computations += 1
        baseline_evaluation = _evaluate_candidate(
            dataset,
            matrices.test,
            window,
            baseline,
            cash,
            paper_configuration,
            decision_interval,

        )
        benchmarks.append(
            WalkForwardBenchmarkEvaluation(
                window,
                "production_baseline",
                baseline_evaluation.status,
                baseline_evaluation.metrics,
                baseline_evaluation.failure_reason,
            )
        )
        computations += 1
        benchmarks.append(
            _evaluate_buy_and_hold(
                dataset, matrices.test, window, cash, paper_configuration
            )
        )
        computations += 1
    candidates = tuple(
        {
            item.candidate.stable_candidate_id: item.candidate
            for item in selections
            if item.selected_for_test
        }.values()
    )
    stability = tuple(
        _aggregate(
            candidate,
            windows,
            tuple(selections),
            tuple(evaluations),
            tuple(benchmarks),
            configuration,
        )
        for candidate in candidates
    )
    stable = tuple(
        item
        for item in stability
        if item.classification is WalkForwardClassification.STABLE_CANDIDATE
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
            "No candidate satisfied every version-2 stability criterion across "
            "multiple independently selected test windows. No production change is "
            "recommended. Historical evidence does not guarantee future results."
        )
    )
    return WalkForwardResult(
        WalkForwardConclusion.COMPLETED,
        configuration,
        windows,
        tuple(selections),
        tuple(evaluations),
        tuple(benchmarks),
        stability,
        recommended.candidate.stable_candidate_id if recommended else None,
        explanation,
        computations,
        0,
        0,
    )


def _select_candidates(
    *,
    dataset: HistoricalDataset,
    window: WalkForwardWindow,
    matrices: WalkForwardMatrices,
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    decision_interval: timedelta,
    refinement_budget: RefinementBudget,
    maximum_candidates: int,
) -> tuple[tuple[WalkForwardCandidateSelection, ...], int]:
    if not matrices.training.rows or not matrices.validation.rows:
        return (), 0
    scores = generate_score_configurations()
    if len(scores) > refinement_budget.max_score_configurations:
        raise WalkForwardError(
            "score configurations exceed the source refinement budget"
        )
    score_results = tuple(
        (
            candidate_for(score, BASELINE_LONG_ONLY_V1_PARAMETERS),
            _simulate_metrics(
                dataset,
                matrices.training,
                initial_cash,
                fee_rate,
                slippage_rate,
                decision_interval,
                candidate_for(score, BASELINE_LONG_ONLY_V1_PARAMETERS),
            ),
        )
        for score in scores
    )
    retained_scores = tuple(
        item[0].score_parameters
        for item in sorted(score_results, key=_candidate_metric_ranking)
    )[: refinement_budget.score_screening_top_k]
    joint = tuple(
        {
            candidate.stable_candidate_id: candidate
            for score in retained_scores
            for strategy in generate_strategy_configurations()
            for candidate in (candidate_for(score, strategy),)
        }.values()
    )
    if len(joint) > refinement_budget.max_joint_candidates:
        raise WalkForwardError("joint candidates exceed the source refinement budget")
    training_results = tuple(
        (
            candidate,
            _simulate_metrics(
                dataset,
                matrices.training,
                initial_cash,
                fee_rate,
                slippage_rate,
                decision_interval,
                candidate,
            ),
        )
        for candidate in joint
    )
    validation_candidates = tuple(
        item[0] for item in sorted(training_results, key=_candidate_metric_ranking)
    )[: refinement_budget.validation_top_k]
    validation_results = tuple(
        (
            candidate,
            _simulate_metrics(
                dataset,
                matrices.validation,

                initial_cash,
                fee_rate,
                slippage_rate,
                decision_interval,
                candidate,
            ),
        )
        for candidate in validation_candidates
    )
    ranked = tuple(sorted(validation_results, key=_candidate_metric_ranking))
    selections = tuple(
        WalkForwardCandidateSelection(
            window=window,
            candidate=candidate,
            validation_rank=rank,
            selected_for_test=rank <= maximum_candidates,
            selection_reason=(
                "selected_by_window_validation_rank"
                if rank <= maximum_candidates
                else "outside_window_test_candidate_limit"
            ),
            validation_metrics=metrics,
        )
        for rank, (candidate, metrics) in enumerate(ranked, start=1)
    )
    computations = len(score_results) + len(training_results) + len(ranked)
    return selections, computations


def _simulate_metrics(
    dataset: HistoricalDataset,
    matrix: HistoricalFeatureMatrix,
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    decision_interval: timedelta,
    candidate: RefinementCandidate,
) -> WalkForwardMetrics:
    result = simulate_historical_feature_matrix(
        dataset=dataset,
        feature_matrix=matrix,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        decision_interval=decision_interval,
        strategy_parameters=candidate.strategy_parameters,
        score_parameters=candidate.score_parameters,
    )
    return _simulation_metrics(result)


def _evaluate_candidate(
    dataset: HistoricalDataset,
    matrix: HistoricalFeatureMatrix,
    window: WalkForwardWindow,
    candidate: RefinementCandidate,
    initial_cash: Decimal,
    configuration: SimulationConfiguration,
    decision_interval: timedelta,
) -> WalkForwardCandidateEvaluation:
    try:
        metrics = _simulate_metrics(
            dataset,
            matrix,
            initial_cash,
            configuration.fee_rate,
            configuration.slippage_rate,
            decision_interval,
            candidate,
        )
        return WalkForwardCandidateEvaluation(
            window,
            candidate,
            WalkForwardEvaluationStatus.COMPLETED,
            metrics,
            None,
        )
    except (HistoricalSimulationError, PaperExecutionError) as error:
        return WalkForwardCandidateEvaluation(
            window,
            candidate,
            WalkForwardEvaluationStatus.FAILED,
            None,
            _sanitize_failure(error),
        )


def _evaluate_buy_and_hold(
    dataset: HistoricalDataset,
    matrix: HistoricalFeatureMatrix,
    window: WalkForwardWindow,
    initial_cash: Decimal,
    configuration: SimulationConfiguration,
) -> WalkForwardBenchmarkEvaluation:
    try:
        result = simulate_buy_and_hold(
            matrix,
            RefinementSegmentName.TEST,
            dataset.identity.instrument,
            initial_cash,
            configuration,
        )
        return WalkForwardBenchmarkEvaluation(
            window,
            "buy_and_hold",
            WalkForwardEvaluationStatus.COMPLETED,
            _benchmark_metrics(result),
            None,
        )
    except (HistoricalSimulationError, PaperExecutionError) as error:
        return WalkForwardBenchmarkEvaluation(
            window,
            "buy_and_hold",
            WalkForwardEvaluationStatus.FAILED,
            None,
            _sanitize_failure(error),
        )


def _aggregate(
    candidate: RefinementCandidate,
    windows: tuple[WalkForwardWindow, ...],
    selections: tuple[WalkForwardCandidateSelection, ...],
    evaluations: tuple[WalkForwardCandidateEvaluation, ...],
    benchmarks: tuple[WalkForwardBenchmarkEvaluation, ...],
    configuration: WalkForwardConfiguration,
) -> WalkForwardStability:
    selected_ordinals = {
        item.window.ordinal
        for item in selections
        if item.candidate.stable_candidate_id == candidate.stable_candidate_id
        and item.selected_for_test
    }
    completed = tuple(
        item
        for item in evaluations
        if item.candidate.stable_candidate_id == candidate.stable_candidate_id
        and item.window.ordinal in selected_ordinals
        and item.status is WalkForwardEvaluationStatus.COMPLETED
        and item.metrics is not None
    )
    if len(completed) < configuration.minimum_valid_windows:
        return _empty_stability(
            candidate,
            WalkForwardClassification.INSUFFICIENT_EVIDENCE,
            len(completed),
        )
    metrics = tuple(item.metrics for item in completed if item.metrics is not None)
    returns = tuple(item.total_return for item in metrics)
    drawdowns = tuple(item.max_drawdown for item in metrics)

    profitable = sum(value > 0 for value in returns)
    ratio = _ratio(profitable, len(returns))
    mean_return = _mean(returns)
    median_return = quantize_financial(Decimal(median(returns)))
    deviation = _standard_deviation(returns, mean_return)
    production = _benchmark_metrics_by_window(benchmarks, "production_baseline")
    buy_hold = _benchmark_metrics_by_window(benchmarks, "buy_and_hold")
    positive_production = sum(
        item.metrics is not None
        and (reference := production.get(item.window.ordinal)) is not None
        and item.metrics.total_return > reference.total_return
        for item in completed
    )
    positive_buy_hold = sum(
        item.metrics is not None
        and (reference := buy_hold.get(item.window.ordinal)) is not None
        and item.metrics.total_return > reference.total_return
        for item in completed
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
        f"{classification.value}: legitimately selected and completed in "
        f"{len(returns)} windows; profitable in {profitable}; worst return "
        f"{min(returns)}; worst drawdown {max(drawdowns)}; beat production in "
        f"{positive_production} windows and buy-and-hold in {positive_buy_hold}. "
        "Historical evidence does not guarantee future results."
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
        _ratio(positive_production, len(returns)),
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
            f"{classification.value}: only {windows_evaluated} legitimately selected "
            "test evaluations completed. Historical evidence does not guarantee "
            "future results."
        ),
    )


def _matrices_for_ranges(
    matrix: HistoricalFeatureMatrix,
    ranges: tuple[
        tuple[datetime, datetime],
        tuple[datetime, datetime],
        tuple[datetime, datetime],
    ],
    decision_interval: timedelta,
) -> WalkForwardMatrices:
    values = tuple(
        HistoricalFeatureMatrix(
            matrix.dataset_fingerprint,
            _rows_in_range(matrix, start, end, decision_interval),
        )
        for start, end in ranges
    )
    return WalkForwardMatrices(values[0], values[1], values[2])


def _rows_in_range(
    matrix: HistoricalFeatureMatrix,
    start: datetime,
    end: datetime,
    decision_interval: timedelta,
) -> tuple[HistoricalFeatureRow, ...]:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    candle_duration = timedelta(minutes=5)
    return tuple(
        row
        for row in matrix.rows
        if start <= (boundary := row.candle_open_time + candle_duration) < end
        and (boundary - epoch) % decision_interval == timedelta(0)
    )


def _selection_fingerprint(
    training: HistoricalFeatureMatrix,
    validation: HistoricalFeatureMatrix,
    configuration: WalkForwardConfiguration,
    refinement_budget: RefinementBudget,
    decision_interval: timedelta,
) -> str:
    payload = {
        "algorithm_version": WALK_FORWARD_ALGORITHM_VERSION,
        "training": _matrix_fingerprint(training),
        "validation": _matrix_fingerprint(validation),
        "configuration": _canonical_dataclass(configuration),
        "refinement_budget": _canonical_dataclass(refinement_budget),
        "decision_interval_seconds": int(decision_interval.total_seconds()),
    }
    return _hash_payload(payload)


def _window_fingerprint(
    selection_fingerprint: str,
    test: HistoricalFeatureMatrix,
    ranges: tuple[
        tuple[datetime, datetime],

        tuple[datetime, datetime],
        tuple[datetime, datetime],
    ],
) -> str:
    return _hash_payload(
        {
            "selection_fingerprint": selection_fingerprint,
            "test": _matrix_fingerprint(test),
            "ranges": tuple(
                (start.isoformat(), end.isoformat()) for start, end in ranges
            ),
        }
    )


def _matrix_fingerprint(matrix: HistoricalFeatureMatrix) -> str:
    digest = sha256()
    for row in matrix.rows:
        digest.update(
            json.dumps(
                (
                    row.candle_open_time.isoformat(),
                    str(row.close_price),
                    str(row.ema_20),
                    str(row.ema_50),
                    str(row.rsi_14),
                    str(row.atr_14),
                    str(row.return_1),
                    str(row.return_12),
                    str(row.relative_volume_20),
                ),
                separators=(",", ":"),
            ).encode()
        )
    return digest.hexdigest()


def _hash_payload(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _canonical_dataclass(
    value: WalkForwardConfiguration | RefinementBudget,
) -> dict[str, object]:
    return {
        name: str(item) if isinstance(item, Decimal) else item
        for name, item in asdict(value).items()
    }


def _candidate_metric_ranking(
    value: tuple[RefinementCandidate, WalkForwardMetrics],
) -> tuple[Decimal, Decimal, int, str]:
    candidate, metrics = value
    return (
        -metrics.total_return,
        metrics.max_drawdown,
        metrics.trade_count,
        candidate.stable_candidate_id,
    )


def _simulation_metrics(result: HistoricalSimulationResult) -> WalkForwardMetrics:
    return WalkForwardMetrics(
        result.total_return,
        result.absolute_pnl,
        result.max_drawdown,
        result.final_equity,
        result.fees_paid,
        result.trade_count,
        (
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


def _benchmark_metrics_by_window(
    evaluations: tuple[WalkForwardBenchmarkEvaluation, ...], benchmark: str
) -> dict[int, WalkForwardMetrics]:
    return {
        item.window.ordinal: item.metrics
        for item in evaluations
        if item.benchmark == benchmark and item.metrics is not None
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
        f"legitimately selected and profitable in {value.profitable_windows} of "
        f"{value.windows_evaluated} completed windows, worst return "
        f"{value.worst_window_return}, maximum drawdown {value.worst_max_drawdown}, "
        f"production outperformance ratio "
        f"{value.positive_excess_return_vs_production_ratio}, and buy-and-hold "
        f"outperformance ratio {value.positive_excess_return_vs_buy_and_hold_ratio}. "
        "It is not promoted automatically. Historical evidence does not guarantee "
        "future results."
    )
