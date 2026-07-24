import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from enum import StrEnum
from itertools import combinations
from statistics import median

from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.historical_data import HistoricalDataset
from app.optimization.predictive_edges import EvidenceThresholds
from app.optimization.refinement import RefinementSegmentName, simulate_buy_and_hold
from app.optimization.simulation import (
    SIMULATION_ENGINE_VERSION,
    HistoricalSimulationResult,
    SimulationDecisionOverride,
    simulate_historical_feature_matrix,
)
from app.optimization.walk_forward import WalkForwardWindow
from app.paper.config import SimulationConfiguration
from app.strategies.baseline_long_only import StrategyAction

COMPOSITE_FEATURE_POOL_VERSION = 1
COMPOSITE_SEARCH_ALGORITHM_VERSION = 2
COMPOSITE_RANKING_VERSION = 2
COMPOSITE_NORMALIZATION_VERSION = 2
COMPOSITE_QUANTUM = Decimal("0.000000000000000001")
COMPOSITE_FEATURE_POOL = (
    "ema_20_relative",
    "ema_50_relative",
    "rsi_14",
    "atr_14_relative",
    "return_1",
    "return_12",
    "relative_volume_20",
)


class CompositeSearchError(ValueError):
    """Raised when composite research evidence violates its frozen contract."""


class CompositeCandidateStatus(StrEnum):
    RETAINED_TRAINING = "retained_training"
    REJECTED_VALIDATION = "rejected_validation"
    PASSED_VALIDATION = "passed_validation"
    REJECTED_TEST = "rejected_test"
    CONFIRMED_CANDIDATE = "confirmed_candidate"


class CompositeSearchConclusion(StrEnum):
    NO_EVALUABLE_FEATURES = "no_evaluable_features"
    SEARCH_BUDGET_EXHAUSTED = "search_budget_exhausted"
    NO_TRAINING_SIGNAL = "no_training_signal"
    TRAINING_SIGNAL_NOT_STABLE = "training_signal_not_stable"
    CANDIDATES_REJECTED_VALIDATION = "candidates_rejected_validation"
    CANDIDATES_PASSED_VALIDATION = "candidates_passed_validation"
    CANDIDATES_REJECTED_TEST = "candidates_rejected_test"
    CONFIRMED_COMPOSITE_CANDIDATES = "confirmed_composite_candidates"


@dataclass(frozen=True, slots=True)
class CompositeSearchBudget:
    beam_width_per_depth: int = 12
    maximum_variants_per_subset: int = 3
    maximum_cheap_evaluations: int = 420
    maximum_training_simulations: int = 72
    maximum_validation_per_depth: int = 8
    maximum_test_candidates: int = 12
    maximum_persisted_candidates: int = 80

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if any(isinstance(value, bool) or value <= 0 for value in values):
            raise CompositeSearchError("composite search limits must be positive")


DEFAULT_COMPOSITE_SEARCH_BUDGET = CompositeSearchBudget()


@dataclass(frozen=True, slots=True)
class NormalizationParameter:
    feature_name: str
    median: Decimal
    interquartile_range: Decimal


@dataclass(frozen=True, slots=True)
class CompositeWeight:
    feature_name: str
    weight: Decimal


@dataclass(frozen=True, slots=True)
class CompositeCandidate:
    features: tuple[str, ...]
    weights: tuple[CompositeWeight, ...]
    normalization: tuple[NormalizationParameter, ...]
    entry_threshold: Decimal
    exit_threshold: Decimal
    cadence: timedelta
    prediction_horizon: timedelta
    minimum_holding_period: timedelta
    maximum_holding_period: timedelta
    cooldown: timedelta
    fee_rate: Decimal
    slippage_rate: Decimal
    initial_cash: Decimal
    simulation_engine_version: int
    algorithm_version: int
    feature_pool_version: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class CompositeMetrics:
    observations: int
    correlation: Decimal
    directional_hit_rate: Decimal
    expected_gross_movement: Decimal
    expected_net_movement: Decimal
    dispersion: Decimal
    profitable_window_ratio: Decimal
    usable_window_count: int = 0
    total_return: Decimal = Decimal(0)
    absolute_pnl: Decimal = Decimal(0)
    max_drawdown: Decimal = Decimal(0)
    trade_count: int = 0
    fees_paid: Decimal = Decimal(0)
    final_equity: Decimal = Decimal(0)
    production_baseline_return: Decimal = Decimal(0)
    excess_return_over_production: Decimal = Decimal(0)
    buy_and_hold_return: Decimal = Decimal(0)
    excess_return_over_buy_and_hold: Decimal = Decimal(0)
    return_stability: Decimal = Decimal(0)


@dataclass(frozen=True, slots=True)
class CompositeRankingBreakdown:
    net_movement: Decimal
    hit_rate: Decimal
    stability: Decimal
    complexity_penalty: Decimal
    concentration_penalty: Decimal
    total: Decimal


@dataclass(frozen=True, slots=True)
class CompositeCandidateResult:
    candidate: CompositeCandidate
    parent_fingerprint: str | None
    depth: int
    status: CompositeCandidateStatus
    ranking: CompositeRankingBreakdown
    training_metrics: CompositeMetrics
    validation_metrics: CompositeMetrics | None
    test_metrics: CompositeMetrics | None
    passed_requirements: tuple[str, ...]
    failed_requirements: tuple[str, ...]
    rejection_reason: str | None
    normalized_distance: Decimal


@dataclass(frozen=True, slots=True)
class CompositeDepthFunnel:
    depth: int
    subsets_generated: int
    weight_configurations_generated: int
    duplicates_removed: int
    cheap_evaluations_computed: int
    cheap_evaluations_reused: int
    training_simulations_computed: int
    training_simulations_reused: int
    branches_retained: int
    validation_evaluated: int
    validation_retained: int
    test_evaluated: int


@dataclass(frozen=True, slots=True)
class CompositeSearchResult:
    conclusion: CompositeSearchConclusion
    maximum_depth_reached: int
    feature_pool: tuple[str, ...]
    excluded_features: dict[str, str]
    funnel_by_depth: tuple[CompositeDepthFunnel, ...]
    candidates: tuple[CompositeCandidateResult, ...]
    total_evaluations: int
    computed: int
    reused: int
    budget_exhausted: bool
    cheap_evaluations_computed: int
    cheap_evaluations_reused: int
    training_simulations_computed: int
    training_simulations_reused: int
    validation_simulations_computed: int
    validation_simulations_reused: int
    test_simulations_computed: int
    test_simulations_reused: int
    best_depth: int | None
    best_candidate_fingerprint: str | None
    dominant_rejection_reason: str | None


def _empty_metrics() -> CompositeMetrics:
    return CompositeMetrics(
        observations=0,
        correlation=Decimal(0),
        directional_hit_rate=Decimal(0),
        expected_gross_movement=Decimal(0),
        expected_net_movement=Decimal(0),
        dispersion=Decimal(0),
        profitable_window_ratio=Decimal(0),
    )


def feature_value(row: HistoricalFeatureRow, name: str) -> Decimal:
    values = {
        "ema_20_relative": row.ema_20 / row.close_price - 1,
        "ema_50_relative": row.ema_50 / row.close_price - 1,
        "rsi_14": row.rsi_14,
        "atr_14_relative": row.atr_14 / row.close_price,
        "return_1": row.return_1,
        "return_12": row.return_12,
        "relative_volume_20": row.relative_volume_20,
    }
    try:
        return values[name]
    except KeyError as error:
        raise CompositeSearchError(f"unknown composite feature: {name}") from error


def validate_feature_pool(matrix: HistoricalFeatureMatrix) -> dict[str, str]:
    if len(set(COMPOSITE_FEATURE_POOL)) != len(COMPOSITE_FEATURE_POOL):
        raise CompositeSearchError("composite feature pool contains duplicates")
    excluded: dict[str, str] = {}
    for name in COMPOSITE_FEATURE_POOL:
        values = tuple(feature_value(row, name) for row in matrix.rows)
        if len(values) < 30:
            excluded[name] = "insufficient_observations"
        elif len(set(values)) == 1:
            excluded[name] = "constant_feature"
    return excluded


def fit_normalization(
    rows: tuple[HistoricalFeatureRow, ...], features: tuple[str, ...]
) -> tuple[NormalizationParameter, ...]:
    parameters: list[NormalizationParameter] = []
    for name in features:
        values = sorted(feature_value(row, name) for row in rows)
        if len(values) < 4:
            raise CompositeSearchError(
                "training has insufficient normalization evidence"
            )
        center = Decimal(median(values))
        lower = values[(len(values) - 1) // 4]
        upper = values[((len(values) - 1) * 3) // 4]
        spread = upper - lower
        if spread == 0:
            raise CompositeSearchError(f"training feature is constant: {name}")
        parameters.append(NormalizationParameter(name, center, spread))
    return tuple(parameters)


def normalize_weights(
    values: tuple[tuple[str, Decimal], ...],
) -> tuple[CompositeWeight, ...]:
    ordered = tuple(sorted(values))
    if (
        len(ordered) < 2
        or len(ordered) > 7
        or len({name for name, _ in ordered}) != len(ordered)
    ):
        raise CompositeSearchError(
            "composite candidates require 2 to 7 unique features"
        )
    if any(value == 0 for _, value in ordered):
        raise CompositeSearchError("composite weights cannot be zero")
    magnitude = sum((abs(value) for _, value in ordered), Decimal(0))
    normalized = [
        CompositeWeight(
            name,
            (value / magnitude).quantize(COMPOSITE_QUANTUM, rounding=ROUND_HALF_EVEN),
        )
        for name, value in ordered
    ]
    correction = Decimal(1) - sum((abs(item.weight) for item in normalized), Decimal(0))
    last = normalized[-1]
    normalized[-1] = CompositeWeight(
        last.feature_name,
        last.weight + (correction if last.weight > 0 else -correction),
    )
    return tuple(normalized)


def build_candidate(
    *,
    weights: tuple[tuple[str, Decimal], ...],
    training_rows: tuple[HistoricalFeatureRow, ...],
    simulation: SimulationConfiguration,
    cadence: timedelta,
    horizon: timedelta,
    entry_threshold: Decimal = Decimal("0.50"),
    exit_threshold: Decimal = Decimal("-0.10"),
    minimum_holding_period: timedelta = timedelta(minutes=30),
    maximum_holding_period: timedelta = timedelta(hours=12),
    cooldown: timedelta = timedelta(minutes=30),
    initial_cash: Decimal = Decimal("10000"),
) -> CompositeCandidate:
    normalized_weights = normalize_weights(weights)
    features = tuple(item.feature_name for item in normalized_weights)
    normalization = fit_normalization(training_rows, features)
    payload = {
        "features": features,
        "weights": [
            (item.feature_name, str(item.weight)) for item in normalized_weights
        ],
        "normalization": [
            (item.feature_name, str(item.median), str(item.interquartile_range))
            for item in normalization
        ],
        "entry_threshold": str(entry_threshold),
        "exit_threshold": str(exit_threshold),
        "cadence": int(cadence.total_seconds()),
        "horizon": int(horizon.total_seconds()),
        "minimum_holding_period": int(minimum_holding_period.total_seconds()),
        "maximum_holding_period": int(maximum_holding_period.total_seconds()),
        "cooldown": int(cooldown.total_seconds()),
        "fee_rate": str(simulation.fee_rate),
        "slippage_rate": str(simulation.slippage_rate),
        "initial_cash": str(initial_cash),
        "simulation_engine_version": SIMULATION_ENGINE_VERSION,
        "algorithm_version": COMPOSITE_SEARCH_ALGORITHM_VERSION,
        "feature_pool_version": COMPOSITE_FEATURE_POOL_VERSION,
        "normalization_version": COMPOSITE_NORMALIZATION_VERSION,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CompositeCandidate(
        features,
        normalized_weights,
        normalization,
        entry_threshold,
        exit_threshold,
        cadence,
        horizon,
        minimum_holding_period,
        maximum_holding_period,
        cooldown,
        simulation.fee_rate,
        simulation.slippage_rate,
        initial_cash,
        SIMULATION_ENGINE_VERSION,
        COMPOSITE_SEARCH_ALGORITHM_VERSION,
        COMPOSITE_FEATURE_POOL_VERSION,
        fingerprint,
    )


def composite_score(
    candidate: CompositeCandidate, row: HistoricalFeatureRow
) -> Decimal:
    parameters = {item.feature_name: item for item in candidate.normalization}
    return sum(
        (
            (
                (
                    feature_value(row, item.feature_name)
                    - parameters[item.feature_name].median
                )
                / parameters[item.feature_name].interquartile_range
            )
            * item.weight
            for item in candidate.weights
        ),
        Decimal(0),
    )


def _rows_between(
    matrix: HistoricalFeatureMatrix, start: object, end: object
) -> tuple[HistoricalFeatureRow, ...]:
    return tuple(row for row in matrix.rows if start <= row.candle_open_time < end)  # type: ignore[operator]


def _phase_rows(
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    phase: str,
) -> tuple[HistoricalFeatureRow, ...]:
    selected: dict[datetime, HistoricalFeatureRow] = {}
    for window in windows:
        start = getattr(window, f"{phase}_from")
        end = getattr(window, f"{phase}_to")
        for row in _rows_between(matrix, start, end):
            selected[row.candle_open_time] = row
    return tuple(selected[key] for key in sorted(selected))


def _cheap_metrics(
    candidate: CompositeCandidate,
    rows: tuple[HistoricalFeatureRow, ...],
) -> CompositeMetrics:
    horizon_steps = max(1, int(candidate.prediction_horizon / timedelta(minutes=5)))
    observations: list[tuple[Decimal, Decimal]] = []
    for index in range(0, max(0, len(rows) - horizon_steps)):
        score = composite_score(candidate, rows[index])
        future = rows[index + horizon_steps].close_price / rows[index].close_price - 1
        observations.append((score, future))
    if not observations:
        return _empty_metrics()
    mean_score = sum((x for x, _ in observations), Decimal(0)) / len(observations)
    mean_return = sum((y for _, y in observations), Decimal(0)) / len(observations)
    covariance = sum(
        ((x - mean_score) * (y - mean_return) for x, y in observations), Decimal(0)
    )
    score_variance = sum(((x - mean_score) ** 2 for x, _ in observations), Decimal(0))
    return_variance = sum(((y - mean_return) ** 2 for _, y in observations), Decimal(0))
    correlation = (
        covariance / (score_variance.sqrt() * return_variance.sqrt())
        if score_variance and return_variance
        else Decimal(0)
    )
    hits = sum(1 for score, value in observations if (score >= 0) == (value >= 0))
    gross = sum(
        (value if score >= 0 else -value for score, value in observations),
        Decimal(0),
    ) / len(observations)
    friction = (candidate.fee_rate + candidate.slippage_rate) * 2
    dispersion = (return_variance / len(observations)).sqrt()
    return CompositeMetrics(
        len(observations),
        correlation.quantize(COMPOSITE_QUANTUM, rounding=ROUND_HALF_EVEN),
        (Decimal(hits) / len(observations)).quantize(
            COMPOSITE_QUANTUM, rounding=ROUND_HALF_EVEN
        ),
        gross.quantize(COMPOSITE_QUANTUM, rounding=ROUND_HALF_EVEN),
        (gross - friction).quantize(COMPOSITE_QUANTUM, rounding=ROUND_HALF_EVEN),
        dispersion.quantize(COMPOSITE_QUANTUM, rounding=ROUND_HALF_EVEN),
        (Decimal(1) if gross > friction else Decimal(0)),
    )


def _cheap_window_metrics(
    candidate: CompositeCandidate,
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    phase: str,
) -> CompositeMetrics:
    usable = tuple(
        value
        for window in windows
        if (
            value := _cheap_metrics(
                candidate,
                _rows_between(
                    matrix,
                    getattr(window, f"{phase}_from"),
                    getattr(window, f"{phase}_to"),
                ),
            )
        ).observations
    )
    if not usable:
        return _cheap_metrics(candidate, ())
    count = Decimal(len(usable))
    return CompositeMetrics(
        sum(item.observations for item in usable),
        (sum((item.correlation for item in usable), Decimal(0)) / count),
        (sum((item.directional_hit_rate for item in usable), Decimal(0)) / count),
        (sum((item.expected_gross_movement for item in usable), Decimal(0)) / count),
        (sum((item.expected_net_movement for item in usable), Decimal(0)) / count),
        (sum((item.dispersion for item in usable), Decimal(0)) / count),
        (sum((item.profitable_window_ratio for item in usable), Decimal(0)) / count),
    )


def _association(
    rows: tuple[HistoricalFeatureRow, ...], name: str, horizon: timedelta
) -> Decimal:
    steps = max(1, int(horizon / timedelta(minutes=5)))
    pairs = tuple(
        (
            feature_value(rows[index], name),
            rows[index + steps].close_price / rows[index].close_price - 1,
        )
        for index in range(max(0, len(rows) - steps))
    )
    if len(pairs) < 2:
        return Decimal(0)
    mean_x = sum((x for x, _ in pairs), Decimal(0)) / len(pairs)
    mean_y = sum((y for _, y in pairs), Decimal(0)) / len(pairs)
    covariance = sum(((x - mean_x) * (y - mean_y) for x, y in pairs), Decimal(0))
    variance_x = sum(((x - mean_x) ** 2 for x, _ in pairs), Decimal(0))
    variance_y = sum(((y - mean_y) ** 2 for _, y in pairs), Decimal(0))
    return (
        covariance / (variance_x.sqrt() * variance_y.sqrt())
        if variance_x and variance_y
        else Decimal(0)
    )


def _training_evidence(
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    horizon: timedelta,
) -> dict[str, Decimal]:
    by_feature: dict[str, list[Decimal]] = {name: [] for name in COMPOSITE_FEATURE_POOL}
    for window in windows:
        rows = _rows_between(matrix, window.training_from, window.training_to)
        for name in by_feature:
            value = _association(rows, name, horizon)
            if value:
                by_feature[name].append(value)
    return {
        name: sum(values, Decimal(0)) / len(values) if values else Decimal(0)
        for name, values in by_feature.items()
    }


def _weight_variants(
    features: tuple[str, ...],
    evidence: dict[str, Decimal],
    parent: CompositeCandidate | None,
    limit: int,
) -> tuple[tuple[tuple[str, Decimal], ...], ...]:
    signed = tuple(
        (name, Decimal(1) if evidence[name] >= 0 else Decimal(-1)) for name in features
    )
    weighted = tuple(
        (name, evidence[name] if evidence[name] else value) for name, value in signed
    )
    variants: list[tuple[tuple[str, Decimal], ...]] = [signed, weighted]
    if parent is not None:
        parent_weights = {item.feature_name: item.weight for item in parent.weights}
        variants.append(
            tuple(
                (
                    name,
                    parent_weights[name] * Decimal("0.90")
                    if name in parent_weights
                    else (evidence[name] or Decimal("0.10")),
                )
                for name in features
            )
        )
    unique: dict[tuple[tuple[str, Decimal], ...], None] = {}
    for variant in variants:
        normalized = normalize_weights(variant)
        unique[tuple((item.feature_name, item.weight) for item in normalized)] = None
    return tuple(unique)[:limit]


_OPERATIONAL_CONFIGURATIONS = (
    (
        timedelta(minutes=30),
        timedelta(minutes=30),
        Decimal("0.03"),
        Decimal("-0.03"),
        timedelta(minutes=30),
        timedelta(hours=12),
        timedelta(minutes=30),
    ),
    (
        timedelta(hours=4),
        timedelta(hours=4),
        Decimal("0.35"),
        Decimal("-0.05"),
        timedelta(hours=4),
        timedelta(hours=24),
        timedelta(hours=4),
    ),
    (
        timedelta(hours=8),
        timedelta(hours=12),
        Decimal("0.65"),
        Decimal("0.00"),
        timedelta(hours=8),
        timedelta(days=2),
        timedelta(hours=8),
    ),
    (
        timedelta(hours=12),
        timedelta(hours=12),
        Decimal("0.50"),
        Decimal("-0.20"),
        timedelta(hours=12),
        timedelta(days=3),
        timedelta(hours=12),
    ),
)


def _candidate_with_window_normalization(
    candidate: CompositeCandidate, rows: tuple[HistoricalFeatureRow, ...]
) -> CompositeCandidate:
    return replace(candidate, normalization=fit_normalization(rows, candidate.features))


def _composite_action_filter(candidate: CompositeCandidate):  # type: ignore[no-untyped-def]
    previous_score: Decimal | None = None
    opened_at: datetime | None = None
    last_exit_at: datetime | None = None

    def decide(
        row: HistoricalFeatureRow,
        position_is_open: bool,
        _baseline_action: StrategyAction,
    ) -> StrategyAction | SimulationDecisionOverride:
        nonlocal previous_score, opened_at, last_exit_at
        boundary = row.candle_open_time + timedelta(minutes=5)
        score = composite_score(candidate, row)
        entry_crossed = score >= candidate.entry_threshold and (
            previous_score is None or previous_score < candidate.entry_threshold
        )
        exit_crossed = (
            score <= candidate.exit_threshold
            and previous_score is not None
            and previous_score > candidate.exit_threshold
        )
        previous_score = score
        if position_is_open:
            if opened_at is None:
                opened_at = boundary
            held = boundary - opened_at
            if held >= candidate.maximum_holding_period or (
                held >= candidate.minimum_holding_period and exit_crossed
            ):
                opened_at = None
                last_exit_at = boundary
                return SimulationDecisionOverride(
                    StrategyAction.SELL_ALL,
                    applied_rule="composite_exit",
                    explanation="Frozen composite exit or maximum holding rule.",
                )
            return StrategyAction.NO_ACTION
        cooldown_complete = (
            last_exit_at is None or boundary - last_exit_at >= candidate.cooldown
        )
        if entry_crossed and cooldown_complete:
            opened_at = boundary
            return SimulationDecisionOverride(
                StrategyAction.OPEN_POSITION,
                target_allocation_percentage=Decimal("15.00"),
                applied_rule="composite_entry",
                explanation="Frozen composite entry threshold crossed.",
            )
        return StrategyAction.NO_ACTION

    return decide


def _economic_metrics(
    *,
    candidate: CompositeCandidate,
    dataset: HistoricalDataset,
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    phase: str,
    simulation: SimulationConfiguration,
    initial_cash: Decimal,
) -> CompositeMetrics:
    results: list[HistoricalSimulationResult] = []
    baseline_returns: list[Decimal] = []
    buy_hold_returns: list[Decimal] = []
    for window in windows:
        training_rows = _rows_between(matrix, window.training_from, window.training_to)
        phase_rows = _rows_between(
            matrix, getattr(window, f"{phase}_from"), getattr(window, f"{phase}_to")
        )
        if len(training_rows) < 4 or not phase_rows:
            continue
        frozen = _candidate_with_window_normalization(candidate, training_rows)
        try:
            result = simulate_historical_feature_matrix(
                dataset=dataset,
                feature_matrix=HistoricalFeatureMatrix(
                    matrix.dataset_fingerprint, phase_rows
                ),
                initial_cash=initial_cash,
                fee_rate=simulation.fee_rate,
                slippage_rate=simulation.slippage_rate,
                decision_interval=frozen.cadence,
                decision_action_filter=_composite_action_filter(frozen),
            )
            baseline = simulate_historical_feature_matrix(
                dataset=dataset,
                feature_matrix=HistoricalFeatureMatrix(
                    matrix.dataset_fingerprint, phase_rows
                ),
                initial_cash=initial_cash,
                fee_rate=simulation.fee_rate,
                slippage_rate=simulation.slippage_rate,
                decision_interval=frozen.cadence,
            )
            buy_hold = simulate_buy_and_hold(
                HistoricalFeatureMatrix(matrix.dataset_fingerprint, phase_rows),
                RefinementSegmentName.TEST,
                dataset.identity.instrument,
                initial_cash,
                simulation,
            )
        except (ValueError, IndexError):
            continue
        results.append(result)
        baseline_returns.append(baseline.total_return)
        buy_hold_returns.append(buy_hold.total_return)
    if not results:
        return _empty_metrics()
    count = Decimal(len(results))
    returns = tuple(item.total_return for item in results)
    mean_return = sum(returns, Decimal(0)) / count
    mean_baseline = sum(baseline_returns, Decimal(0)) / count
    mean_buy_hold = sum(buy_hold_returns, Decimal(0)) / count
    stability = sum((Decimal(1) for value in returns if value > 0), Decimal(0)) / count
    return CompositeMetrics(
        observations=sum(item.decision_count for item in results),
        correlation=Decimal(0),
        directional_hit_rate=Decimal(0),
        expected_gross_movement=Decimal(0),
        expected_net_movement=mean_return,
        dispersion=(
            sum(((value - mean_return) ** 2 for value in returns), Decimal(0)) / count
        ).sqrt(),
        profitable_window_ratio=stability,
        usable_window_count=len(results),
        total_return=mean_return,
        absolute_pnl=sum((item.absolute_pnl for item in results), Decimal(0)) / count,
        max_drawdown=max(item.max_drawdown for item in results),
        trade_count=sum(item.trade_count for item in results),
        fees_paid=sum((item.fees_paid for item in results), Decimal(0)),
        final_equity=sum((item.final_equity for item in results), Decimal(0)) / count,
        production_baseline_return=mean_baseline,
        excess_return_over_production=mean_return - mean_baseline,
        buy_and_hold_return=mean_buy_hold,
        excess_return_over_buy_and_hold=mean_return - mean_buy_hold,
        return_stability=stability,
    )


def _ranking(
    candidate: CompositeCandidate, metrics: CompositeMetrics
) -> CompositeRankingBreakdown:
    complexity = Decimal(len(candidate.features) - 2) * Decimal("0.01")
    concentration = max(abs(item.weight) for item in candidate.weights)
    concentration_penalty = max(Decimal(0), concentration - Decimal("0.60"))
    stability = metrics.return_stability or (
        Decimal(1) - min(Decimal(1), metrics.dispersion)
    )
    total = (
        metrics.total_return
        + metrics.excess_return_over_production
        + metrics.expected_net_movement
        + metrics.directional_hit_rate / Decimal(10)
        + stability
        - metrics.max_drawdown
        - metrics.fees_paid / max(metrics.final_equity, Decimal(1))
        - complexity
        - concentration_penalty
    )
    return CompositeRankingBreakdown(
        metrics.expected_net_movement,
        metrics.directional_hit_rate,
        stability,
        complexity,
        concentration_penalty,
        total.quantize(COMPOSITE_QUANTUM, rounding=ROUND_HALF_EVEN),
    )


def search_composite_features(
    *,
    dataset: HistoricalDataset,
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    simulation: SimulationConfiguration,
    initial_cash: Decimal,
    budget: CompositeSearchBudget = DEFAULT_COMPOSITE_SEARCH_BUDGET,
) -> CompositeSearchResult:
    excluded = validate_feature_pool(matrix)
    pool = tuple(name for name in COMPOSITE_FEATURE_POOL if name not in excluded)
    if len(pool) < 2 or not windows:
        return CompositeSearchResult(
            conclusion=CompositeSearchConclusion.NO_EVALUABLE_FEATURES,
            maximum_depth_reached=0,
            feature_pool=pool,
            excluded_features=excluded,
            funnel_by_depth=(),
            candidates=(),
            total_evaluations=0,
            computed=0,
            reused=0,
            budget_exhausted=False,
            cheap_evaluations_computed=0,
            cheap_evaluations_reused=0,
            training_simulations_computed=0,
            training_simulations_reused=0,
            validation_simulations_computed=0,
            validation_simulations_reused=0,
            test_simulations_computed=0,
            test_simulations_reused=0,
            best_depth=None,
            best_candidate_fingerprint=None,
            dominant_rejection_reason="insufficient_evidence",
        )
    training_rows = _phase_rows(matrix, windows, "training")
    evidence = _training_evidence(matrix, windows, timedelta(hours=4))
    thresholds = EvidenceThresholds()
    results: list[CompositeCandidateResult] = []
    funnels: list[CompositeDepthFunnel] = []
    seen: set[str] = set()
    cheap_computed = 0
    training_computed = 0
    validation_computed = 0
    test_computed = 0
    parents: tuple[CompositeCandidateResult, ...] = ()
    budget_exhausted = False
    maximum_depth = min(7, len(pool))
    for depth in range(2, maximum_depth + 1):
        proposals: list[tuple[tuple[str, ...], CompositeCandidateResult | None]] = []
        if depth == 2:
            proposals = [(subset, None) for subset in combinations(pool, 2)]
        else:
            for retained_parent in parents:
                for name in pool:
                    if name not in retained_parent.candidate.features:
                        proposals.append(
                            (
                                tuple(
                                    sorted((*retained_parent.candidate.features, name))
                                ),
                                retained_parent,
                            )
                        )
        if not proposals:
            break
        proposal_keys: set[tuple[tuple[str, ...], str | None]] = set()
        unique_proposals: list[
            tuple[tuple[str, ...], CompositeCandidateResult | None]
        ] = []
        for subset, proposal_parent in proposals:
            key = (
                subset,
                proposal_parent.candidate.fingerprint if proposal_parent else None,
            )
            if key not in proposal_keys:
                proposal_keys.add(key)
                unique_proposals.append((subset, proposal_parent))
        unique_proposals.sort(
            key=lambda value: (
                -sum((abs(evidence[name]) for name in value[0]), Decimal(0)),
                -(value[1].ranking.total if value[1] is not None else Decimal(0)),
                value[0],
                value[1].candidate.fingerprint if value[1] is not None else "",
            )
        )
        remaining_depths = maximum_depth - depth + 1
        remaining_cheap = budget.maximum_cheap_evaluations - cheap_computed
        if remaining_cheap <= 0:
            budget_exhausted = True
            break
        depth_cheap_limit = max(1, remaining_cheap // remaining_depths)
        depth_cheap_stop = cheap_computed + depth_cheap_limit
        cheap_ranked: list[tuple[CompositeCandidate, str | None, CompositeMetrics]] = []
        configurations = 0
        duplicates = 0
        for subset, proposal_parent in unique_proposals:
            if cheap_computed >= depth_cheap_stop:
                budget_exhausted = True
                break
            for weights in _weight_variants(
                subset,
                evidence,
                proposal_parent.candidate if proposal_parent else None,
                budget.maximum_variants_per_subset,
            ):
                for (
                    cadence,
                    horizon,
                    entry,
                    exit_,
                    minimum,
                    maximum,
                    cooldown,
                ) in _OPERATIONAL_CONFIGURATIONS:
                    if cheap_computed >= depth_cheap_stop:
                        budget_exhausted = True
                        break
                    configurations += 1
                    candidate = build_candidate(
                        weights=weights,
                        training_rows=training_rows,
                        simulation=simulation,
                        cadence=cadence,
                        horizon=horizon,
                        entry_threshold=entry,
                        exit_threshold=exit_,
                        minimum_holding_period=minimum,
                        maximum_holding_period=maximum,
                        cooldown=cooldown,
                        initial_cash=initial_cash,
                    )
                    if candidate.fingerprint in seen:
                        duplicates += 1
                        continue
                    seen.add(candidate.fingerprint)
                    cheap_computed += 1
                    cheap = _cheap_window_metrics(
                        candidate, matrix, windows, "training"
                    )
                    cheap_ranked.append(
                        (
                            candidate,
                            proposal_parent.candidate.fingerprint
                            if proposal_parent
                            else None,
                            cheap,
                        )
                    )
        cheap_ranked.sort(
            key=lambda value: (
                -_ranking(value[0], value[2]).total,
                value[0].fingerprint,
            )
        )
        if not cheap_ranked:
            break
        remaining_training = budget.maximum_training_simulations - training_computed
        if remaining_training <= 0:
            budget_exhausted = True
            break
        depth_training_limit = max(1, remaining_training // remaining_depths)
        simulation_slots = min(
            len(cheap_ranked),
            budget.beam_width_per_depth * 2,
            depth_training_limit,
        )
        if simulation_slots < len(cheap_ranked):
            budget_exhausted = True
        depth_results: list[CompositeCandidateResult] = []
        for candidate, parent_fingerprint, _cheap in cheap_ranked[:simulation_slots]:
            training = _economic_metrics(
                candidate=candidate,
                dataset=dataset,
                matrix=matrix,
                windows=windows,
                phase="training",
                simulation=simulation,
                initial_cash=initial_cash,
            )
            training_computed += 1
            training_passes = (
                training.usable_window_count >= thresholds.minimum_windows
                and training.total_return > 0
                and training.trade_count >= training.usable_window_count
                and training.max_drawdown <= Decimal("0.35")
                and training.return_stability
                >= thresholds.minimum_stable_direction_ratio
            )
            failed = () if training_passes else ("training_economic_qualification",)
            depth_results.append(
                CompositeCandidateResult(
                    candidate,
                    parent_fingerprint,
                    depth,
                    CompositeCandidateStatus.RETAINED_TRAINING,
                    _ranking(candidate, training),
                    training,
                    None,
                    None,
                    ("training_economic_qualification",) if training_passes else (),
                    failed,
                    failed[0] if failed else None,
                    Decimal(0) if training_passes else Decimal(1),
                )
            )
        depth_results.sort(
            key=lambda item: (-item.ranking.total, item.candidate.fingerprint)
        )
        parents = tuple(item for item in depth_results if not item.failed_requirements)[
            : budget.beam_width_per_depth
        ]
        validation_limit = min(len(parents), budget.maximum_validation_per_depth)
        finalized: list[CompositeCandidateResult] = []
        for item in parents[:validation_limit]:
            validation = _economic_metrics(
                candidate=item.candidate,
                dataset=dataset,
                matrix=matrix,
                windows=windows,
                phase="validation",
                simulation=simulation,
                initial_cash=initial_cash,
            )
            validation_computed += 1
            validation_passes = (
                validation.usable_window_count >= thresholds.minimum_windows
                and validation.total_return > 0
                and validation.trade_count >= validation.usable_window_count
                and validation.max_drawdown <= Decimal("0.35")
                and validation.return_stability
                >= thresholds.minimum_stable_direction_ratio
                and validation.excess_return_over_production > 0
            )
            test = None
            if validation_passes and test_computed < budget.maximum_test_candidates:
                test = _economic_metrics(
                    candidate=item.candidate,
                    dataset=dataset,
                    matrix=matrix,
                    windows=windows,
                    phase="test",
                    simulation=simulation,
                    initial_cash=initial_cash,
                )
                test_computed += 1
            elif validation_passes:
                budget_exhausted = True
            test_confirmed = bool(
                test
                and test.usable_window_count >= thresholds.minimum_windows
                and test.total_return > 0
                and test.trade_count >= test.usable_window_count
                and test.max_drawdown <= Decimal("0.35")
                and test.return_stability >= thresholds.minimum_stable_direction_ratio
                and test.excess_return_over_production > 0
            )
            status = (
                CompositeCandidateStatus.CONFIRMED_CANDIDATE
                if test_confirmed
                else CompositeCandidateStatus.REJECTED_TEST
                if test is not None
                else CompositeCandidateStatus.PASSED_VALIDATION
                if validation_passes
                else CompositeCandidateStatus.REJECTED_VALIDATION
            )
            failed = (
                ()
                if test_confirmed or (validation_passes and test is None)
                else (
                    ("test_economic_qualification",)
                    if test
                    else ("validation_economic_qualification",)
                )
            )
            final_metrics = test or validation
            finalized.append(
                CompositeCandidateResult(
                    item.candidate,
                    item.parent_fingerprint,
                    depth,
                    status,
                    _ranking(item.candidate, final_metrics),
                    item.training_metrics,
                    validation,
                    test,
                    item.passed_requirements
                    + (
                        ("validation_economic_qualification",)
                        if validation_passes
                        else ()
                    )
                    + (("test_economic_qualification",) if test_confirmed else ()),
                    failed,
                    failed[0] if failed else None,
                    Decimal(0)
                    if test_confirmed or (validation_passes and test is None)
                    else Decimal(1),
                )
            )
        persisted = tuple(
            finalized
            + list(parents[validation_limit:])
            + [item for item in depth_results if item not in parents]
        )
        remaining_persistence = budget.maximum_persisted_candidates - len(results)
        if remaining_persistence > 0:
            results.extend(persisted[:remaining_persistence])
        if len(persisted) > max(0, remaining_persistence):
            budget_exhausted = True
        funnels.append(
            CompositeDepthFunnel(
                depth,
                len(unique_proposals),
                configurations,
                duplicates,
                len(cheap_ranked),
                0,
                simulation_slots,
                0,
                len(parents),
                validation_limit,
                sum(
                    1
                    for item in finalized
                    if item.status
                    in {
                        CompositeCandidateStatus.PASSED_VALIDATION,
                        CompositeCandidateStatus.REJECTED_TEST,
                        CompositeCandidateStatus.CONFIRMED_CANDIDATE,
                    }
                ),
                sum(1 for item in finalized if item.test_metrics is not None),
            )
        )
        if not parents or training_computed >= budget.maximum_training_simulations:
            if training_computed >= budget.maximum_training_simulations:
                budget_exhausted = True
            break
    confirmed_candidates = [
        item
        for item in results
        if item.status is CompositeCandidateStatus.CONFIRMED_CANDIDATE
    ]
    passed_validation = [
        item
        for item in results
        if item.status is CompositeCandidateStatus.PASSED_VALIDATION
    ]
    rejected_test = [
        item
        for item in results
        if item.status is CompositeCandidateStatus.REJECTED_TEST
    ]
    rejected_validation = [
        item
        for item in results
        if item.status is CompositeCandidateStatus.REJECTED_VALIDATION
    ]
    conclusion = (
        CompositeSearchConclusion.CONFIRMED_COMPOSITE_CANDIDATES
        if confirmed_candidates
        else CompositeSearchConclusion.CANDIDATES_PASSED_VALIDATION
        if passed_validation
        else CompositeSearchConclusion.CANDIDATES_REJECTED_TEST
        if rejected_test
        else CompositeSearchConclusion.CANDIDATES_REJECTED_VALIDATION
        if rejected_validation
        else CompositeSearchConclusion.NO_TRAINING_SIGNAL
        if results
        else CompositeSearchConclusion.SEARCH_BUDGET_EXHAUSTED
        if budget_exhausted
        else CompositeSearchConclusion.NO_TRAINING_SIGNAL
    )
    priority = {
        CompositeCandidateStatus.CONFIRMED_CANDIDATE: 5,
        CompositeCandidateStatus.PASSED_VALIDATION: 4,
        CompositeCandidateStatus.REJECTED_TEST: 3,
        CompositeCandidateStatus.RETAINED_TRAINING: 2,
        CompositeCandidateStatus.REJECTED_VALIDATION: 1,
    }
    best = max(
        results,
        key=lambda item: (
            priority[item.status],
            item.ranking.total,
            item.candidate.fingerprint,
        ),
        default=None,
    )
    total_evaluations = (
        cheap_computed + training_computed + validation_computed + test_computed
    )
    return CompositeSearchResult(
        conclusion=conclusion,
        maximum_depth_reached=funnels[-1].depth if funnels else 0,
        feature_pool=pool,
        excluded_features=excluded,
        funnel_by_depth=tuple(funnels),
        candidates=tuple(results),
        total_evaluations=total_evaluations,
        computed=total_evaluations,
        reused=0,
        budget_exhausted=budget_exhausted,
        cheap_evaluations_computed=cheap_computed,
        cheap_evaluations_reused=0,
        training_simulations_computed=training_computed,
        training_simulations_reused=0,
        validation_simulations_computed=validation_computed,
        validation_simulations_reused=0,
        test_simulations_computed=test_computed,
        test_simulations_reused=0,
        best_depth=best.depth if best else None,
        best_candidate_fingerprint=best.candidate.fingerprint if best else None,
        dominant_rejection_reason=(
            None if confirmed_candidates else "economic_evidence_not_confirmed"
        ),
    )
