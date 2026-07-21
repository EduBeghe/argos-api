import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from enum import StrEnum
from math import sqrt
from statistics import median

from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.historical_data import HistoricalDataset
from app.optimization.walk_forward import WalkForwardConfiguration, WalkForwardWindow
from app.paper.config import SimulationConfiguration
from app.scores.calculator import BaselineScoreInput, calculate_baseline_score

PREDICTIVE_EDGE_ALGORITHM_VERSION = 1
PREDICTOR_VERSION = 1
DEFAULT_HORIZONS = tuple(
    timedelta(minutes=value) for value in (30, 60, 120, 360, 720, 1440, 2880)
)
MAX_HORIZONS = 12
MAX_HORIZON = timedelta(days=14)
MAX_PREDICTORS = 16
MAX_CONDITIONS = 5
MAX_BINS = 10
RATIO_QUANTUM = Decimal("0.000000000000000001")
DEFAULT_EDGE_WINDOWS = WalkForwardConfiguration()


class PredictiveEdgeError(ValueError):
    """Raised when predictive-edge evidence or configuration is invalid."""


class Direction(StrEnum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class EvidenceStatus(StrEnum):
    PROMISING_POSITIVE = "promising_positive_edge"
    PROMISING_NEGATIVE = "promising_negative_edge"
    WEAK = "weak_edge"
    UNSTABLE = "unstable_edge"
    NONE = "no_detectable_edge"
    INSUFFICIENT = "insufficient_evidence"


class Condition(StrEnum):
    ALL = "all"
    EMA_ABOVE = "ema20_above_ema50"
    EMA_BELOW = "ema20_below_ema50"
    HIGH_VOLATILITY = "high_volatility"
    NORMAL_VOLATILITY = "normal_volatility"


@dataclass(frozen=True, slots=True)
class PredictorDefinition:
    name: str
    explanation: str
    source: str
    direction_convention: str
    version: int = PREDICTOR_VERSION


PREDICTORS = (
    PredictorDefinition("ema_20", "EMA 20 level.", "feature:ema_20", "higher value"),
    PredictorDefinition("ema_50", "EMA 50 level.", "feature:ema_50", "higher value"),
    PredictorDefinition(
        "rsi_14", "RSI 14 oscillator.", "feature:rsi_14", "higher is more positive"
    ),
    PredictorDefinition(
        "atr_14", "ATR 14 volatility.", "feature:atr_14", "higher is more volatile"
    ),
    PredictorDefinition(
        "return_1", "One-candle return.", "feature:return_1", "positive is upward"
    ),
    PredictorDefinition(
        "return_12", "Twelve-candle return.", "feature:return_12", "positive is upward"
    ),
    PredictorDefinition(
        "relative_volume_20",
        "Volume relative to its prior baseline.",
        "feature:relative_volume_20",
        "higher confirms activity",
    ),
    PredictorDefinition(
        "trend_component",
        "ATR-normalized EMA separation.",
        "score:baseline_technical:1",
        "positive is upward",
    ),
    PredictorDefinition(
        "rsi_contribution",
        "Weighted RSI contribution.",
        "score:baseline_technical:1",
        "positive is upward",
    ),
    PredictorDefinition(
        "return_1_contribution",
        "Weighted short-return contribution.",
        "score:baseline_technical:1",
        "positive is upward",
    ),
    PredictorDefinition(
        "return_12_contribution",
        "Weighted long-return contribution.",
        "score:baseline_technical:1",
        "positive is upward",
    ),
    PredictorDefinition(
        "volume_confirmation",
        "Score volume confirmation factor.",
        "score:baseline_technical:1",
        "higher confirms direction",
    ),
    PredictorDefinition(
        "directional_score",
        "Sum of directional contributions.",
        "score:baseline_technical:1",
        "positive is upward",
    ),
    PredictorDefinition(
        "overall_score",
        "Final baseline technical score.",
        "score:baseline_technical:1",
        "positive is upward",
    ),
)


@dataclass(frozen=True, slots=True)
class EvidenceThresholds:
    minimum_observations: int = 30
    minimum_windows: int = 3
    minimum_stable_direction_ratio: Decimal = Decimal("0.60")
    minimum_absolute_correlation: Decimal = Decimal("0.10")
    minimum_directional_hit_rate: Decimal = Decimal("0.55")


@dataclass(frozen=True, slots=True)
class PredictiveEdgeConfiguration:
    horizons: tuple[timedelta, ...] = DEFAULT_HORIZONS
    decision_interval: timedelta = timedelta(minutes=30)
    bins: int = 5
    neutral_threshold: Decimal | None = None
    conditions: tuple[Condition, ...] = tuple(Condition)
    windows: WalkForwardConfiguration = DEFAULT_EDGE_WINDOWS
    thresholds: EvidenceThresholds = EvidenceThresholds()

    def __post_init__(self) -> None:
        if not self.horizons or len(self.horizons) > MAX_HORIZONS:
            raise PredictiveEdgeError("horizon count is outside the allowed bounds")
        if any(h <= timedelta(0) or h > MAX_HORIZON for h in self.horizons):
            raise PredictiveEdgeError("horizons must be positive and at most 14 days")
        if len(set(self.horizons)) != len(self.horizons):
            raise PredictiveEdgeError("horizons must be unique")
        if any(h % self.decision_interval for h in self.horizons):
            raise PredictiveEdgeError("horizons must align to the decision cadence")
        if not 2 <= self.bins <= MAX_BINS:
            raise PredictiveEdgeError(f"bins must be between 2 and {MAX_BINS}")
        if not self.conditions or len(self.conditions) > MAX_CONDITIONS:
            raise PredictiveEdgeError("condition count is outside the allowed bounds")


@dataclass(frozen=True, slots=True)
class Observation:
    boundary: datetime
    predictor_value: Decimal
    forward_return: Decimal
    net_return: Decimal
    direction: Direction
    condition: Condition


@dataclass(frozen=True, slots=True)
class EdgeMetrics:
    usable_count: int
    missing_count: int
    correlation: Decimal | None
    directional_hit_rate: Decimal | None
    up_precision: Decimal | None
    down_precision: Decimal | None
    neutral_rate: Decimal
    mean_gross_return: Decimal
    median_gross_return: Decimal
    mean_net_return: Decimal
    positive_net_ratio: Decimal
    average_absolute_movement: Decimal
    sample_standard_deviation: Decimal | None
    standard_error: Decimal | None
    evidence_status: EvidenceStatus
    insufficient_reason: str | None


@dataclass(frozen=True, slots=True)
class BinAggregate:
    ordinal: int
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    observation_count: int
    mean_forward_return: Decimal
    median_forward_return: Decimal
    up_count: int
    down_count: int
    neutral_count: int
    positive_net_ratio: Decimal
    mean_net_return: Decimal
    evidence_status: EvidenceStatus


@dataclass(frozen=True, slots=True)
class WindowEdge:
    window_ordinal: int
    predictor: PredictorDefinition
    horizon: timedelta
    condition: Condition
    training_bin_bounds: tuple[Decimal, ...]
    volatility_threshold: Decimal
    metrics: EdgeMetrics
    bins: tuple[BinAggregate, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class StabilityEdge:
    predictor: PredictorDefinition
    horizon: timedelta
    condition: Condition
    windows_evaluated: int
    sufficient_windows: int
    positive_correlation_windows: int
    negative_correlation_windows: int
    stable_direction_ratio: Decimal
    weighted_mean_correlation: Decimal | None
    median_correlation: Decimal | None
    worst_correlation: Decimal | None
    mean_directional_hit_rate: Decimal | None
    mean_positive_net_ratio: Decimal
    mean_net_return: Decimal
    correlation_dispersion: Decimal | None
    total_usable_observations: int
    classification: EvidenceStatus


@dataclass(frozen=True, slots=True)
class StrategyHypothesis:
    hypothesis_id: str
    predictor_name: str
    predictor_version: int
    direction: str
    horizon: timedelta
    condition: Condition
    expected_gross_return: Decimal
    expected_net_return: Decimal
    directional_hit_rate: Decimal | None
    stability_ratio: Decimal
    minimum_evaluation_interval: timedelta
    minimum_holding_period_from: timedelta
    minimum_holding_period_to: timedelta
    evidence_window_ordinals: tuple[int, ...]
    evidence_status: EvidenceStatus
    rationale: str
    limitations: str


@dataclass(frozen=True, slots=True)
class BenchmarkAggregate:
    horizon: timedelta
    benchmark: str
    observation_count: int
    directional_hit_rate: Decimal | None
    mean_return: Decimal


@dataclass(frozen=True, slots=True)
class PredictiveEdgeResult:
    configuration: PredictiveEdgeConfiguration
    window_edges: tuple[WindowEdge, ...]
    stability_edges: tuple[StabilityEdge, ...]
    hypotheses: tuple[StrategyHypothesis, ...]
    benchmarks: tuple[BenchmarkAggregate, ...]


def analyze_predictive_edges(
    *,
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    simulation: SimulationConfiguration,
    configuration: PredictiveEdgeConfiguration,
) -> PredictiveEdgeResult:
    rows = {row.candle_open_time: row for row in matrix.rows}
    friction = Decimal(2) * (simulation.fee_rate + simulation.slippage_rate)
    neutral = (
        configuration.neutral_threshold
        if configuration.neutral_threshold is not None
        else friction
    )
    window_edges: list[WindowEdge] = []
    benchmark_observations: dict[timedelta, list[Observation]] = {}
    for window in windows:
        training = _phase_rows(matrix.rows, window.training_from, window.training_to)
        volatility = _quantile(
            tuple(_predictors(row)["atr_14"] / row.close_price for row in training),
            Decimal("0.80"),
        )
        for predictor in PREDICTORS:
            training_values = tuple(
                _predictors(row)[predictor.name] for row in training
            )
            bounds = _quantile_bounds(training_values, configuration.bins)
            for horizon in configuration.horizons:
                for condition in configuration.conditions:
                    observations, missing = _observations(
                        rows,
                        window,
                        predictor,
                        horizon,
                        condition,
                        volatility,
                        friction,
                        neutral,
                        configuration.decision_interval,
                    )
                    if predictor.name == "overall_score" and condition is Condition.ALL:
                        benchmark_observations.setdefault(horizon, []).extend(
                            observations
                        )
                    metrics = _metrics(observations, missing, configuration.thresholds)
                    bin_metrics = _bin_aggregates(
                        observations, bounds, configuration.thresholds
                    )
                    payload = (
                        window.ordinal,
                        predictor.name,
                        int(horizon.total_seconds()),
                        condition.value,
                        bounds,
                        metrics,
                    )
                    window_edges.append(
                        WindowEdge(
                            window.ordinal,
                            predictor,
                            horizon,
                            condition,
                            bounds,
                            volatility,
                            metrics,
                            bin_metrics,
                            _fingerprint(payload),
                        )
                    )
    stability = _stability(tuple(window_edges), configuration.thresholds)
    hypotheses = tuple(
        _hypothesis(
            edge,
            tuple(
                item.window_ordinal
                for item in window_edges
                if item.predictor == edge.predictor
                and item.horizon == edge.horizon
                and item.condition == edge.condition
                and item.metrics.evidence_status is not EvidenceStatus.INSUFFICIENT
            ),
            configuration.decision_interval,
        )
        for edge in stability
        if edge.classification
        in (EvidenceStatus.PROMISING_POSITIVE, EvidenceStatus.PROMISING_NEGATIVE)
    )
    benchmarks = _benchmarks(benchmark_observations)
    return PredictiveEdgeResult(
        configuration, tuple(window_edges), stability, hypotheses, benchmarks
    )


def build_predictive_windows(
    *,
    dataset: HistoricalDataset,
    matrix: HistoricalFeatureMatrix,
    configuration: PredictiveEdgeConfiguration,
) -> tuple[WalkForwardWindow, ...]:
    """Build recent half-open windows without coupling evidence to a Strategy."""

    if dataset.fingerprint != matrix.dataset_fingerprint:
        raise PredictiveEdgeError("dataset and feature matrix fingerprints differ")
    window = configuration.windows
    training = timedelta(days=window.training_days)
    validation = timedelta(days=window.validation_days)
    test = timedelta(days=window.test_days)
    step = timedelta(days=window.step_days)
    start = dataset.identity.range.start
    end = dataset.identity.range.end
    available = end - start - training - validation - test
    if available < timedelta(0):
        return ()
    count = int(available // step) + 1
    first = max(0, count - window.maximum_windows)
    output: list[WalkForwardWindow] = []
    for ordinal, index in enumerate(range(first, count), start=1):
        training_from = start + index * step
        training_to = training_from + training
        validation_to = training_to + validation
        test_to = validation_to + test
        identity = (
            dataset.fingerprint,
            training_from,
            training_to,
            validation_to,
            test_to,
        )
        output.append(
            WalkForwardWindow(
                ordinal,
                training_from,
                training_to,
                training_to,
                validation_to,
                validation_to,
                test_to,
                sum(
                    validation_to <= row.candle_open_time < test_to
                    for row in matrix.rows
                ),
                _fingerprint(identity),
                _fingerprint((*identity, "training-only-selection")),
            )
        )
    return tuple(output)


def _predictors(row: HistoricalFeatureRow) -> dict[str, Decimal]:
    score = calculate_baseline_score(
        BaselineScoreInput(
            row.close_price,
            row.ema_20,
            row.ema_50,
            row.rsi_14,
            row.atr_14,
            row.return_1,
            row.return_12,
            row.relative_volume_20,
        )
    )
    return {
        "ema_20": row.ema_20,
        "ema_50": row.ema_50,
        "rsi_14": row.rsi_14,
        "atr_14": row.atr_14,
        "return_1": row.return_1,
        "return_12": row.return_12,
        "relative_volume_20": row.relative_volume_20,
        "trend_component": score.trend_score,
        "rsi_contribution": score.rsi_contribution,
        "return_1_contribution": score.return_1_contribution,
        "return_12_contribution": score.return_12_contribution,
        "volume_confirmation": score.volume_confirmation_factor,
        "directional_score": score.directional_score,
        "overall_score": score.overall_score,
    }


def _phase_rows(
    rows: tuple[HistoricalFeatureRow, ...], start: datetime, end: datetime
) -> tuple[HistoricalFeatureRow, ...]:
    return tuple(row for row in rows if start <= row.candle_open_time < end)


def _observations(
    rows: dict[datetime, HistoricalFeatureRow],
    window: WalkForwardWindow,
    predictor: PredictorDefinition,
    horizon: timedelta,
    condition: Condition,
    volatility_threshold: Decimal,
    friction: Decimal,
    neutral: Decimal,
    cadence: timedelta,
) -> tuple[tuple[Observation, ...], int]:
    result: list[Observation] = []
    missing = 0
    for boundary, row in sorted(rows.items()):
        if (
            not window.test_from <= boundary < window.test_to
            or (boundary - window.test_from) % cadence
        ):
            continue
        future_time = boundary + horizon
        if future_time >= window.test_to or future_time not in rows:
            missing += 1
            continue
        ratio = row.atr_14 / row.close_price
        if (
            (condition is Condition.EMA_ABOVE and row.ema_20 <= row.ema_50)
            or (condition is Condition.EMA_BELOW and row.ema_20 >= row.ema_50)
            or (condition is Condition.HIGH_VOLATILITY and ratio < volatility_threshold)
            or (
                condition is Condition.NORMAL_VOLATILITY
                and ratio >= volatility_threshold
            )
        ):
            continue
        value = _predictors(row)[predictor.name]
        gross = rows[future_time].close_price / row.close_price - Decimal(1)
        direction = (
            Direction.UP
            if gross > neutral
            else Direction.DOWN
            if gross < -neutral
            else Direction.NEUTRAL
        )
        result.append(
            Observation(
                boundary, value, gross, abs(gross) - friction, direction, condition
            )
        )
    return tuple(result), missing


def _metrics(
    values: tuple[Observation, ...], missing: int, thresholds: EvidenceThresholds
) -> EdgeMetrics:
    count = len(values)
    if not count:
        return EdgeMetrics(
            0,
            missing,
            None,
            None,
            None,
            None,
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            None,
            None,
            EvidenceStatus.INSUFFICIENT,
            "no usable observations",
        )
    x = tuple(item.predictor_value for item in values)
    y = tuple(item.forward_return for item in values)
    correlation = _correlation(x, y)
    predicted = tuple(
        Direction.UP
        if item.predictor_value > 0
        else Direction.DOWN
        if item.predictor_value < 0
        else Direction.NEUTRAL
        for item in values
    )
    nonneutral = tuple(
        (p, item.direction)
        for p, item in zip(predicted, values, strict=True)
        if item.direction is not Direction.NEUTRAL
    )
    hit = (
        _ratio(sum(p is actual for p, actual in nonneutral), len(nonneutral))
        if nonneutral
        else None
    )
    up = tuple(actual for p, actual in nonneutral if p is Direction.UP)
    down = tuple(actual for p, actual in nonneutral if p is Direction.DOWN)
    gross_mean = sum(y, Decimal(0)) / count
    net_mean = sum((item.net_return for item in values), Decimal(0)) / count
    std = _sample_std(y)
    status = (
        EvidenceStatus.INSUFFICIENT
        if count < thresholds.minimum_observations or correlation is None
        else EvidenceStatus.WEAK
    )
    reason = (
        f"requires at least {thresholds.minimum_observations} observations"
        if count < thresholds.minimum_observations
        else "constant predictor or target"
        if correlation is None
        else None
    )
    return EdgeMetrics(
        count,
        missing,
        correlation,
        hit,
        _ratio(sum(v is Direction.UP for v in up), len(up)) if up else None,
        _ratio(sum(v is Direction.DOWN for v in down), len(down)) if down else None,
        _ratio(sum(item.direction is Direction.NEUTRAL for item in values), count),
        gross_mean,
        median(y),
        net_mean,
        _ratio(sum(item.net_return > 0 for item in values), count),
        sum((abs(v) for v in y), Decimal(0)) / count,
        std,
        std / Decimal(str(sqrt(count))) if std is not None else None,
        status,
        reason,
    )


def _bin_aggregates(
    values: tuple[Observation, ...],
    bounds: tuple[Decimal, ...],
    thresholds: EvidenceThresholds,
) -> tuple[BinAggregate, ...]:
    groups: list[list[Observation]] = [[] for _ in range(len(bounds) + 1)]
    for value in values:
        index = next(
            (i for i, bound in enumerate(bounds) if value.predictor_value <= bound),
            len(bounds),
        )
        groups[index].append(value)
    output: list[BinAggregate] = []
    for index, group in enumerate(groups):
        count = len(group)
        gross = tuple(item.forward_return for item in group)
        output.append(
            BinAggregate(
                ordinal=index + 1,
                lower_bound=bounds[index - 1] if index else None,
                upper_bound=bounds[index] if index < len(bounds) else None,
                observation_count=count,
                mean_forward_return=(
                    sum(gross, Decimal(0)) / count if count else Decimal(0)
                ),
                median_forward_return=median(gross) if gross else Decimal(0),
                up_count=sum(item.direction is Direction.UP for item in group),
                down_count=sum(item.direction is Direction.DOWN for item in group),
                neutral_count=sum(
                    item.direction is Direction.NEUTRAL for item in group
                ),
                positive_net_ratio=(
                    _ratio(sum(item.net_return > 0 for item in group), count)
                    if count
                    else Decimal(0)
                ),
                mean_net_return=(
                    sum((item.net_return for item in group), Decimal(0)) / count
                    if count
                    else Decimal(0)
                ),
                evidence_status=(
                    EvidenceStatus.WEAK
                    if count >= thresholds.minimum_observations
                    else EvidenceStatus.INSUFFICIENT
                ),
            )
        )
    return tuple(output)


def _benchmarks(
    groups: dict[timedelta, list[Observation]],
) -> tuple[BenchmarkAggregate, ...]:
    output: list[BenchmarkAggregate] = []
    for horizon, values in sorted(groups.items()):
        count = len(values)
        mean = (
            sum((item.forward_return for item in values), Decimal(0)) / count
            if count
            else Decimal(0)
        )
        up_hit = (
            _ratio(sum(item.direction is Direction.UP for item in values), count)
            if count
            else None
        )
        down_hit = (
            _ratio(sum(item.direction is Direction.DOWN for item in values), count)
            if count
            else None
        )
        random_hits = sum(
            (
                Direction.UP
                if int(
                    hashlib.sha256(item.boundary.isoformat().encode()).hexdigest(), 16
                )
                % 2
                else Direction.DOWN
            )
            is item.direction
            for item in values
        )
        output.extend(
            (
                BenchmarkAggregate(
                    horizon,
                    "deterministic_random_v1",
                    count,
                    _ratio(random_hits, count) if count else None,
                    mean,
                ),
                BenchmarkAggregate(horizon, "always_up", count, up_hit, mean),
                BenchmarkAggregate(horizon, "always_down", count, down_hit, mean),
                BenchmarkAggregate(horizon, "no_signal", count, None, mean),
                BenchmarkAggregate(
                    horizon, "unconditional_asset_return", count, None, mean
                ),
            )
        )
    return tuple(output)


def _stability(
    edges: tuple[WindowEdge, ...], thresholds: EvidenceThresholds
) -> tuple[StabilityEdge, ...]:
    groups: dict[tuple[str, timedelta, Condition], list[WindowEdge]] = {}
    for edge in edges:
        groups.setdefault(
            (edge.predictor.name, edge.horizon, edge.condition), []
        ).append(edge)
    output: list[StabilityEdge] = []
    for (_, horizon, condition), group in groups.items():
        sufficient = [
            e
            for e in group
            if e.metrics.evidence_status is not EvidenceStatus.INSUFFICIENT
            and e.metrics.correlation is not None
        ]
        correlations = tuple(
            e.metrics.correlation
            for e in sufficient
            if e.metrics.correlation is not None
        )
        positive = sum(v > 0 for v in correlations)
        negative = sum(v < 0 for v in correlations)
        dominant = max(positive, negative)
        stable = _ratio(dominant, len(correlations)) if correlations else Decimal(0)
        total = sum(e.metrics.usable_count for e in group)
        weighted = (
            sum(
                (
                    e.metrics.correlation * e.metrics.usable_count
                    for e in sufficient
                    if e.metrics.correlation is not None
                ),
                Decimal(0),
            )
            / total
            if total and correlations
            else None
        )
        mean_net = (
            sum((e.metrics.mean_net_return for e in sufficient), Decimal(0))
            / len(sufficient)
            if sufficient
            else Decimal(0)
        )
        mean_hit_values = tuple(
            e.metrics.directional_hit_rate
            for e in sufficient
            if e.metrics.directional_hit_rate is not None
        )
        classification = _classify(
            len(sufficient),
            total,
            stable,
            weighted,
            mean_hit_values,
            mean_net,
            thresholds,
        )
        output.append(
            StabilityEdge(
                group[0].predictor,
                horizon,
                condition,
                len(group),
                len(sufficient),
                positive,
                negative,
                stable,
                weighted,
                median(correlations) if correlations else None,
                min(correlations, key=abs) if correlations else None,
                sum(mean_hit_values, Decimal(0)) / len(mean_hit_values)
                if mean_hit_values
                else None,
                sum((e.metrics.positive_net_ratio for e in sufficient), Decimal(0))
                / len(sufficient)
                if sufficient
                else Decimal(0),
                mean_net,
                _sample_std(correlations),
                total,
                classification,
            )
        )
    return tuple(
        sorted(output, key=lambda e: (e.predictor.name, e.horizon, e.condition.value))
    )


def _classify(
    windows: int,
    observations: int,
    stable: Decimal,
    correlation: Decimal | None,
    hits: tuple[Decimal, ...],
    net: Decimal,
    thresholds: EvidenceThresholds,
) -> EvidenceStatus:
    if (
        windows < thresholds.minimum_windows
        or observations < thresholds.minimum_observations
    ):
        return EvidenceStatus.INSUFFICIENT
    if stable < thresholds.minimum_stable_direction_ratio:
        return EvidenceStatus.UNSTABLE
    hit = sum(hits, Decimal(0)) / len(hits) if hits else Decimal(0)
    if (
        correlation is not None
        and abs(correlation) >= thresholds.minimum_absolute_correlation
        and hit >= thresholds.minimum_directional_hit_rate
        and net > 0
    ):
        return (
            EvidenceStatus.PROMISING_POSITIVE
            if correlation > 0
            else EvidenceStatus.PROMISING_NEGATIVE
        )
    if correlation is None or abs(correlation) < Decimal("0.02"):
        return EvidenceStatus.NONE
    return EvidenceStatus.WEAK


def _hypothesis(
    edge: StabilityEdge, ordinals: tuple[int, ...], cadence: timedelta
) -> StrategyHypothesis:
    key = _fingerprint(
        (
            edge.predictor.name,
            edge.predictor.version,
            edge.horizon.total_seconds(),
            edge.condition.value,
            edge.classification.value,
        )
    )
    return StrategyHypothesis(
        key,
        edge.predictor.name,
        edge.predictor.version,
        "long-positive"
        if edge.classification is EvidenceStatus.PROMISING_POSITIVE
        else "exit/avoid-negative",
        edge.horizon,
        edge.condition,
        edge.mean_net_return,
        edge.mean_net_return,
        edge.mean_directional_hit_rate,
        edge.stable_direction_ratio,
        min(cadence, edge.horizon),
        edge.horizon / 2,
        edge.horizon,
        ordinals,
        edge.classification,
        f"{edge.predictor.name} shows repeatable {edge.classification.value} "
        f"evidence at {edge.horizon} under {edge.condition.value}.",
        "Historical correlation is not causality or a guarantee of profitability; "
        "no entry threshold or position size is implied.",
    )


def _quantile_bounds(values: tuple[Decimal, ...], bins: int) -> tuple[Decimal, ...]:
    if not values:
        return ()
    ordered = sorted(values)
    return tuple(
        ordered[min(len(ordered) - 1, (len(ordered) * index) // bins)]
        for index in range(1, bins)
    )


def _quantile(values: tuple[Decimal, ...], q: Decimal) -> Decimal:
    if not values:
        raise PredictiveEdgeError("training phase contains no evidence")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(Decimal(len(ordered)) * q))]


def _correlation(x: tuple[Decimal, ...], y: tuple[Decimal, ...]) -> Decimal | None:
    if len(x) < 2:
        return None
    mx = sum(x, Decimal(0)) / len(x)
    my = sum(y, Decimal(0)) / len(y)
    numerator = sum(
        ((a - mx) * (b - my) for a, b in zip(x, y, strict=True)), Decimal(0)
    )
    dx = sum(((a - mx) ** 2 for a in x), Decimal(0))
    dy = sum(((b - my) ** 2 for b in y), Decimal(0))
    if dx == 0 or dy == 0:
        return None
    return numerator / (dx * dy).sqrt()


def _sample_std(values: tuple[Decimal, ...]) -> Decimal | None:
    if len(values) < 2:
        return None
    mean = sum(values, Decimal(0)) / len(values)
    return (
        sum(((value - mean) ** 2 for value in values), Decimal(0)) / (len(values) - 1)
    ).sqrt()


def _ratio(numerator: int, denominator: int) -> Decimal:
    return (Decimal(numerator) / Decimal(denominator)).quantize(
        RATIO_QUANTUM, rounding=ROUND_HALF_EVEN
    )


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            default=lambda item: (
                asdict(item)
                if hasattr(item, "__dataclass_fields__")
                else item.total_seconds()
                if isinstance(item, timedelta)
                else item.isoformat()
                if isinstance(item, datetime)
                else str(item)
            ),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

