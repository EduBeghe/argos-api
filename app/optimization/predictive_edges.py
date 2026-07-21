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

PREDICTIVE_EDGE_ALGORITHM_VERSION = 2
PREDICTOR_VERSION = 2
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


class PredictorSemanticKind(StrEnum):
    DIRECTIONAL_CENTERED = "directional_centered"
    DIRECTIONAL_TRAINING_CENTERED = "directional_training_centered"
    MAGNITUDE_OR_ACTIVITY = "magnitude_or_activity"
    CONFIRMATION_ONLY = "confirmation_only"


class DirectionMapping(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NOT_APPLICABLE = "not_applicable"


class ValidationStatus(StrEnum):
    SELECTED = "selected"
    REJECTED = "rejected"
    INSUFFICIENT = "insufficient_evidence"
    NOT_DIRECTIONAL = "not_directional"


class TestConfirmationStatus(StrEnum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    NOT_SELECTED = "not_selected"
    INSUFFICIENT = "insufficient_evidence"


class PredictiveEdgeConclusion(StrEnum):
    COMPLETED = "completed"
    INSUFFICIENT_HISTORY = "insufficient_history"


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
    semantic_kind: PredictorSemanticKind
    direction_convention: str
    version: int = PREDICTOR_VERSION


PREDICTORS = (
    PredictorDefinition(
        "ema_20_relative",
        "EMA 20 relative to the current close: EMA 20 / close - 1.",
        "feature:ema_20",
        PredictorSemanticKind.DIRECTIONAL_TRAINING_CENTERED,
        "training learns center and positive/negative return mapping",
    ),
    PredictorDefinition(
        "ema_50_relative",
        "EMA 50 relative to the current close: EMA 50 / close - 1.",
        "feature:ema_50",
        PredictorSemanticKind.DIRECTIONAL_TRAINING_CENTERED,
        "training learns center and positive/negative return mapping",
    ),
    PredictorDefinition(
        "rsi_14",
        "RSI 14 oscillator centered from training evidence.",
        "feature:rsi_14",
        PredictorSemanticKind.DIRECTIONAL_TRAINING_CENTERED,
        "training learns center and positive/negative return mapping",
    ),
    PredictorDefinition(
        "atr_14_relative",
        "ATR 14 divided by close; predicts movement magnitude, not direction.",
        "feature:atr_14",
        PredictorSemanticKind.MAGNITUDE_OR_ACTIVITY,
        "higher/lower values relate only to absolute future movement",
    ),
    PredictorDefinition(
        "return_1",
        "One-candle return.",
        "feature:return_1",
        PredictorSemanticKind.DIRECTIONAL_CENTERED,
        "zero-centered; training freezes positive or negative mapping",
    ),
    PredictorDefinition(
        "return_12",
        "Twelve-candle return.",
        "feature:return_12",
        PredictorSemanticKind.DIRECTIONAL_CENTERED,
        "zero-centered; training freezes positive or negative mapping",
    ),
    PredictorDefinition(
        "relative_volume_20",
        "Volume relative to its prior baseline; an activity predictor.",
        "feature:relative_volume_20",
        PredictorSemanticKind.MAGNITUDE_OR_ACTIVITY,
        "relates to absolute movement; never implies UP by itself",
    ),
    PredictorDefinition(
        "trend_component",
        "ATR-normalized EMA separation.",
        "score:baseline_technical:1",
        PredictorSemanticKind.DIRECTIONAL_CENTERED,
        "zero-centered; training freezes positive or negative mapping",
    ),
    PredictorDefinition(
        "rsi_contribution",
        "Weighted RSI contribution.",
        "score:baseline_technical:1",
        PredictorSemanticKind.DIRECTIONAL_CENTERED,
        "zero-centered; training freezes positive or negative mapping",
    ),
    PredictorDefinition(
        "return_1_contribution",
        "Weighted short-return contribution.",
        "score:baseline_technical:1",
        PredictorSemanticKind.DIRECTIONAL_CENTERED,
        "zero-centered; training freezes positive or negative mapping",
    ),
    PredictorDefinition(
        "return_12_contribution",
        "Weighted long-return contribution.",
        "score:baseline_technical:1",
        PredictorSemanticKind.DIRECTIONAL_CENTERED,
        "zero-centered; training freezes positive or negative mapping",
    ),
    PredictorDefinition(
        "volume_confirmation",
        "Score volume-confirmation factor; confirmation is not direction.",
        "score:baseline_technical:1",
        PredictorSemanticKind.CONFIRMATION_ONLY,
        "confirmation-only; directional metrics are not applicable",
    ),
    PredictorDefinition(
        "directional_score",
        "Sum of persisted directional contributions.",
        "score:baseline_technical:1",
        PredictorSemanticKind.DIRECTIONAL_CENTERED,
        "zero-centered; training freezes positive or negative mapping",
    ),
    PredictorDefinition(
        "overall_score",
        "Final baseline technical score; training center is explicit.",
        "score:baseline_technical:1",
        PredictorSemanticKind.DIRECTIONAL_TRAINING_CENTERED,
        "training learns center and positive/negative return mapping",
    ),
)


@dataclass(frozen=True, slots=True)
class EvidenceThresholds:
    minimum_observations: int = 30
    minimum_windows: int = 3
    minimum_stable_direction_ratio: Decimal = Decimal("0.60")
    minimum_absolute_correlation: Decimal = Decimal("0.10")
    minimum_directional_hit_rate: Decimal = Decimal("0.55")
    minimum_hit_rate_improvement_over_benchmark: Decimal = Decimal("0.02")


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
    forward_gross_return: Decimal
    absolute_movement: Decimal
    net_movement_magnitude_after_friction: Decimal
    direction: Direction


@dataclass(frozen=True, slots=True)
class DirectionBenchmarks:
    always_up: Decimal
    always_down: Decimal
    deterministic_random: Decimal
    observed_majority: Decimal

    @property
    def strongest(self) -> Decimal:
        return max(
            self.always_up,
            self.always_down,
            self.deterministic_random,
            self.observed_majority,
        )


@dataclass(frozen=True, slots=True)
class EdgeMetrics:
    usable_count: int
    missing_count: int
    correlation: Decimal | None
    directional_hit_rate: Decimal | None
    up_precision: Decimal | None
    down_precision: Decimal | None
    neutral_rate: Decimal
    mean_forward_gross_return: Decimal
    median_forward_gross_return: Decimal
    mean_absolute_movement: Decimal
    mean_net_movement_magnitude_after_friction: Decimal
    positive_movement_after_friction_ratio: Decimal
    mean_strategy_aligned_gross_return: Decimal | None
    mean_strategy_aligned_net_return: Decimal | None
    mean_expected_avoided_downside: Decimal | None
    sample_standard_deviation: Decimal | None
    standard_error: Decimal | None
    direction_benchmarks: DirectionBenchmarks | None
    evidence_status: EvidenceStatus
    insufficient_reason: str | None


@dataclass(frozen=True, slots=True)
class BinAggregate:
    ordinal: int
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    observation_count: int
    mean_forward_gross_return: Decimal
    median_forward_gross_return: Decimal
    up_count: int
    down_count: int
    neutral_count: int
    positive_movement_after_friction_ratio: Decimal
    mean_net_movement_magnitude_after_friction: Decimal
    evidence_status: EvidenceStatus


@dataclass(frozen=True, slots=True)
class WindowEdge:
    window_ordinal: int
    predictor: PredictorDefinition
    horizon: timedelta
    condition: Condition
    training_center: Decimal
    direction_mapping: DirectionMapping
    training_bin_bounds: tuple[Decimal, ...]
    volatility_threshold: Decimal
    validation_status: ValidationStatus
    validation_direction: DirectionMapping
    validation_metrics: EdgeMetrics
    selected_for_test: bool
    selection_reason: str
    selected_bin_ordinals: tuple[int, ...]
    test_confirmation_status: TestConfirmationStatus
    test_metrics: EdgeMetrics
    bins: tuple[BinAggregate, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class StabilityEdge:
    predictor: PredictorDefinition
    horizon: timedelta
    condition: Condition
    direction_mapping: DirectionMapping
    supporting_bin_ordinals: tuple[int, ...]
    windows_evaluated: int
    windows_selected_for_test: int
    windows_confirmed: int
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
    supporting_bin_ordinals: tuple[int, ...]
    expected_gross_return: Decimal | None
    expected_net_return: Decimal | None
    expected_avoided_loss: Decimal | None
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
    mean_forward_gross_return: Decimal


@dataclass(frozen=True, slots=True)
class PredictiveEdgeResult:
    conclusion: PredictiveEdgeConclusion
    windows_available: int
    windows_required: int
    reason: str | None
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
    required = configuration.thresholds.minimum_windows
    if len(windows) < required:
        return PredictiveEdgeResult(
            PredictiveEdgeConclusion.INSUFFICIENT_HISTORY,
            len(windows),
            required,
            f"requires at least {required} valid walk-forward windows",
            configuration,
            (),
            (),
            (),
            (),
        )
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
        training_rows = _phase_rows(
            matrix.rows, window.training_from, window.training_to
        )
        volatility = _quantile(
            tuple(row.atr_14 / row.close_price for row in training_rows),
            Decimal("0.80"),
        )
        for predictor in PREDICTORS:
            training_values = tuple(
                _predictors(row)[predictor.name] for row in training_rows
            )
            center = (
                median(training_values)
                if predictor.semantic_kind
                is PredictorSemanticKind.DIRECTIONAL_TRAINING_CENTERED
                else Decimal(0)
            )
            bounds = _quantile_bounds(training_values, configuration.bins)
            for horizon in configuration.horizons:
                for condition in configuration.conditions:
                    training, _ = _observations(
                        rows,
                        phase_start=window.training_from,
                        phase_end=window.training_to,
                        predictor=predictor,
                        horizon=horizon,
                        condition=condition,
                        volatility_threshold=volatility,
                        friction=friction,
                        neutral=neutral,
                        cadence=configuration.decision_interval,
                    )
                    mapping = _learn_direction_mapping(predictor, training)
                    validation, validation_missing = _observations(
                        rows,
                        phase_start=window.validation_from,
                        phase_end=window.validation_to,
                        predictor=predictor,
                        horizon=horizon,
                        condition=condition,
                        volatility_threshold=volatility,
                        friction=friction,
                        neutral=neutral,
                        cadence=configuration.decision_interval,
                    )
                    validation_metrics = _metrics(
                        validation,
                        validation_missing,
                        predictor=predictor,
                        center=center,
                        mapping=mapping,
                        friction=friction,
                        thresholds=configuration.thresholds,
                    )
                    validation_status, reason = _select_for_test(
                        predictor, validation_metrics, configuration.thresholds
                    )
                    selected = validation_status is ValidationStatus.SELECTED
                    selected_bins = (
                        _selected_bins(
                            validation,
                            bounds,
                            mapping,
                            configuration.thresholds,
                        )
                        if selected
                        else ()
                    )
                    if selected and not selected_bins:
                        validation_status = ValidationStatus.REJECTED
                        selected = False
                        reason = "validation found no promising training-defined bin"
                    test, test_missing = _observations(
                        rows,
                        phase_start=window.test_from,
                        phase_end=window.test_to,
                        predictor=predictor,
                        horizon=horizon,
                        condition=condition,
                        volatility_threshold=volatility,
                        friction=friction,
                        neutral=neutral,
                        cadence=configuration.decision_interval,
                    )
                    test_metrics = _metrics(
                        test,
                        test_missing,
                        predictor=predictor,
                        center=center,
                        mapping=mapping,
                        friction=friction,
                        thresholds=configuration.thresholds,
                    )
                    confirmation = _confirm_test(
                        selected, predictor, test_metrics, configuration.thresholds
                    )
                    bins = _bin_aggregates(test, bounds, configuration.thresholds)
                    payload = (
                        window.ordinal,
                        predictor.name,
                        int(horizon.total_seconds()),
                        condition.value,
                        center,
                        mapping.value,
                        bounds,
                        validation_status.value,
                        confirmation.value,
                        test_metrics,
                    )
                    window_edges.append(
                        WindowEdge(
                            window.ordinal,
                            predictor,
                            horizon,
                            condition,
                            center,
                            mapping,
                            bounds,
                            volatility,
                            validation_status,
                            mapping,
                            validation_metrics,
                            selected,
                            reason,
                            selected_bins,
                            confirmation,
                            test_metrics,
                            bins,
                            _fingerprint(payload),
                        )
                    )
                    if predictor.name == "overall_score" and condition is Condition.ALL:
                        benchmark_observations.setdefault(horizon, []).extend(test)
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
                and item.test_confirmation_status is TestConfirmationStatus.CONFIRMED
            ),
            configuration.decision_interval,
        )
        for edge in stability
        if edge.classification
        in (EvidenceStatus.PROMISING_POSITIVE, EvidenceStatus.PROMISING_NEGATIVE)
    )
    return PredictiveEdgeResult(
        PredictiveEdgeConclusion.COMPLETED,
        len(windows),
        required,
        None,
        configuration,
        tuple(window_edges),
        stability,
        hypotheses,
        _benchmarks(benchmark_observations),
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
        "ema_20_relative": row.ema_20 / row.close_price - Decimal(1),
        "ema_50_relative": row.ema_50 / row.close_price - Decimal(1),
        "rsi_14": row.rsi_14,
        "atr_14_relative": row.atr_14 / row.close_price,
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
    *,
    phase_start: datetime,
    phase_end: datetime,
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
            not phase_start <= boundary < phase_end
            or (boundary - phase_start) % cadence
        ):
            continue
        future_time = boundary + horizon
        if future_time >= phase_end or future_time not in rows:
            missing += 1
            continue
        ratio = row.atr_14 / row.close_price
        excluded = (
            (condition is Condition.EMA_ABOVE and row.ema_20 <= row.ema_50)
            or (condition is Condition.EMA_BELOW and row.ema_20 >= row.ema_50)
            or (condition is Condition.HIGH_VOLATILITY and ratio < volatility_threshold)
            or (
                condition is Condition.NORMAL_VOLATILITY
                and ratio >= volatility_threshold
            )
        )
        if excluded:
            continue
        gross = rows[future_time].close_price / row.close_price - Decimal(1)
        absolute = abs(gross)
        direction = (
            Direction.UP
            if gross > neutral
            else Direction.DOWN
            if gross < -neutral
            else Direction.NEUTRAL
        )
        result.append(
            Observation(
                boundary,
                _predictors(row)[predictor.name],
                gross,
                absolute,
                max(absolute - friction, Decimal(0)),
                direction,
            )
        )
    return tuple(result), missing


def _learn_direction_mapping(
    predictor: PredictorDefinition, values: tuple[Observation, ...]
) -> DirectionMapping:
    if predictor.semantic_kind in (
        PredictorSemanticKind.MAGNITUDE_OR_ACTIVITY,
        PredictorSemanticKind.CONFIRMATION_ONLY,
    ):
        return DirectionMapping.NOT_APPLICABLE
    correlation = _correlation(
        tuple(item.predictor_value for item in values),
        tuple(item.forward_gross_return for item in values),
    )
    if correlation is None or correlation >= 0:
        return DirectionMapping.POSITIVE
    return DirectionMapping.NEGATIVE


def _metrics(
    values: tuple[Observation, ...],
    missing: int,
    *,
    predictor: PredictorDefinition,
    center: Decimal,
    mapping: DirectionMapping,
    friction: Decimal,
    thresholds: EvidenceThresholds,
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
            None,
            None,
            None,
            None,
            EvidenceStatus.INSUFFICIENT,
            "no usable observations",
        )
    x = tuple(item.predictor_value for item in values)
    signed = tuple(item.forward_gross_return for item in values)
    magnitude = tuple(item.absolute_movement for item in values)
    correlation_target = (
        magnitude
        if predictor.semantic_kind
        in (
            PredictorSemanticKind.MAGNITUDE_OR_ACTIVITY,
            PredictorSemanticKind.CONFIRMATION_ONLY,
        )
        else signed
    )
    correlation = _correlation(x, correlation_target)
    directional = mapping is not DirectionMapping.NOT_APPLICABLE
    predicted: tuple[Direction, ...] = ()
    hit: Decimal | None = None
    up_precision: Decimal | None = None
    down_precision: Decimal | None = None
    benchmarks: DirectionBenchmarks | None = None
    aligned_gross: Decimal | None = None
    aligned_net: Decimal | None = None
    avoided_downside: Decimal | None = None
    if directional:
        multiplier = Decimal(1) if mapping is DirectionMapping.POSITIVE else Decimal(-1)
        predicted = tuple(
            Direction.UP
            if (item.predictor_value - center) * multiplier > 0
            else Direction.DOWN
            if (item.predictor_value - center) * multiplier < 0
            else Direction.NEUTRAL
            for item in values
        )
        comparable = tuple(
            (prediction, item)
            for prediction, item in zip(predicted, values, strict=True)
            if prediction is not Direction.NEUTRAL
            and item.direction is not Direction.NEUTRAL
        )
        hit = (
            _ratio(
                sum(prediction is item.direction for prediction, item in comparable),
                len(comparable),
            )
            if comparable
            else None
        )
        up = tuple(
            item for prediction, item in comparable if prediction is Direction.UP
        )
        down = tuple(
            item for prediction, item in comparable if prediction is Direction.DOWN
        )
        up_precision = (
            _ratio(sum(item.direction is Direction.UP for item in up), len(up))
            if up
            else None
        )
        down_precision = (
            _ratio(sum(item.direction is Direction.DOWN for item in down), len(down))
            if down
            else None
        )
        benchmarks = _direction_benchmarks(values)
        long_signals = tuple(
            item.forward_gross_return
            for prediction, item in zip(predicted, values, strict=True)
            if prediction is Direction.UP
        )
        if long_signals:
            aligned_gross = sum(long_signals, Decimal(0)) / len(long_signals)
            aligned_net = aligned_gross - friction
        avoided = tuple(
            abs(item.forward_gross_return)
            for prediction, item in zip(predicted, values, strict=True)
            if prediction is Direction.DOWN and item.forward_gross_return < 0
        )
        if avoided:
            avoided_downside = sum(avoided, Decimal(0)) / len(avoided)
    mean_gross = sum(signed, Decimal(0)) / count
    std = _sample_std(correlation_target)
    insufficient = count < thresholds.minimum_observations or correlation is None
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
        up_precision,
        down_precision,
        _ratio(sum(item.direction is Direction.NEUTRAL for item in values), count),
        mean_gross,
        median(signed),
        sum(magnitude, Decimal(0)) / count,
        sum((item.net_movement_magnitude_after_friction for item in values), Decimal(0))
        / count,
        _ratio(
            sum(item.net_movement_magnitude_after_friction > 0 for item in values),
            count,
        ),
        aligned_gross,
        aligned_net,
        avoided_downside,
        std,
        std / Decimal(str(sqrt(count))) if std is not None else None,
        benchmarks,
        EvidenceStatus.INSUFFICIENT if insufficient else EvidenceStatus.WEAK,
        reason,
    )


def _select_for_test(
    predictor: PredictorDefinition,
    metrics: EdgeMetrics,
    thresholds: EvidenceThresholds,
) -> tuple[ValidationStatus, str]:
    if metrics.evidence_status is EvidenceStatus.INSUFFICIENT:
        return (
            ValidationStatus.INSUFFICIENT,
            metrics.insufficient_reason or "insufficient evidence",
        )
    if predictor.semantic_kind in (
        PredictorSemanticKind.MAGNITUDE_OR_ACTIVITY,
        PredictorSemanticKind.CONFIRMATION_ONLY,
    ):
        return (
            ValidationStatus.NOT_DIRECTIONAL,
            "diagnostic predictor does not select directional hypotheses",
        )
    assert metrics.direction_benchmarks is not None
    if (
        metrics.correlation is None
        or abs(metrics.correlation) < thresholds.minimum_absolute_correlation
    ):
        return ValidationStatus.REJECTED, "validation correlation threshold not met"
    if (
        metrics.directional_hit_rate is None
        or metrics.directional_hit_rate < thresholds.minimum_directional_hit_rate
    ):
        return ValidationStatus.REJECTED, "validation directional hit threshold not met"
    if (
        metrics.directional_hit_rate
        < metrics.direction_benchmarks.strongest
        + thresholds.minimum_hit_rate_improvement_over_benchmark
    ):
        return (
            ValidationStatus.REJECTED,
            "validation does not improve materially over directional benchmark",
        )
    if (metrics.mean_strategy_aligned_net_return or Decimal(0)) <= 0 and (
        metrics.mean_expected_avoided_downside or Decimal(0)
    ) <= 0:
        return (
            ValidationStatus.REJECTED,
            "validation has no favorable long return or avoided downside",
        )
    return ValidationStatus.SELECTED, "selected by frozen validation evidence"


def _confirm_test(
    selected: bool,
    predictor: PredictorDefinition,
    metrics: EdgeMetrics,
    thresholds: EvidenceThresholds,
) -> TestConfirmationStatus:
    if not selected:
        return TestConfirmationStatus.NOT_SELECTED
    if metrics.evidence_status is EvidenceStatus.INSUFFICIENT:
        return TestConfirmationStatus.INSUFFICIENT
    status, _ = _select_for_test(predictor, metrics, thresholds)
    return (
        TestConfirmationStatus.CONFIRMED
        if status is ValidationStatus.SELECTED
        else TestConfirmationStatus.REJECTED
    )


def _direction_benchmarks(values: tuple[Observation, ...]) -> DirectionBenchmarks:
    comparable = tuple(
        item for item in values if item.direction is not Direction.NEUTRAL
    )
    if not comparable:
        return DirectionBenchmarks(*(Decimal(0) for _ in range(4)))
    count = len(comparable)
    up = _ratio(sum(item.direction is Direction.UP for item in comparable), count)
    down = _ratio(sum(item.direction is Direction.DOWN for item in comparable), count)
    random_hits = sum(
        (
            Direction.UP
            if int(hashlib.sha256(item.boundary.isoformat().encode()).hexdigest(), 16)
            % 2
            else Direction.DOWN
        )
        is item.direction
        for item in comparable
    )
    random = _ratio(random_hits, count)
    return DirectionBenchmarks(up, down, random, max(up, down))


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
        gross = tuple(item.forward_gross_return for item in group)
        output.append(
            BinAggregate(
                index + 1,
                bounds[index - 1] if index else None,
                bounds[index] if index < len(bounds) else None,
                count,
                sum(gross, Decimal(0)) / count if count else Decimal(0),
                median(gross) if gross else Decimal(0),
                sum(item.direction is Direction.UP for item in group),
                sum(item.direction is Direction.DOWN for item in group),
                sum(item.direction is Direction.NEUTRAL for item in group),
                _ratio(
                    sum(
                        item.net_movement_magnitude_after_friction > 0 for item in group
                    ),
                    count,
                )
                if count
                else Decimal(0),
                sum(
                    (item.net_movement_magnitude_after_friction for item in group),
                    Decimal(0),
                )
                / count
                if count
                else Decimal(0),
                EvidenceStatus.WEAK
                if count >= thresholds.minimum_observations
                else EvidenceStatus.INSUFFICIENT,
            )
        )
    return tuple(output)


def _selected_bins(
    values: tuple[Observation, ...],
    bounds: tuple[Decimal, ...],
    mapping: DirectionMapping,
    thresholds: EvidenceThresholds,
) -> tuple[int, ...]:
    bins = _bin_aggregates(values, bounds, thresholds)
    minimum_bin_observations = max(
        2, thresholds.minimum_observations // (len(bounds) + 1)
    )
    return tuple(
        item.ordinal
        for item in bins
        if item.observation_count >= minimum_bin_observations
        and item.mean_net_movement_magnitude_after_friction > 0
        and (
            (
                mapping is DirectionMapping.POSITIVE
                and item.mean_forward_gross_return > 0
            )
            or (
                mapping is DirectionMapping.NEGATIVE
                and item.mean_forward_gross_return < 0
            )
        )
    )


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
        selected = [edge for edge in group if edge.selected_for_test]
        confirmed = [
            edge
            for edge in selected
            if edge.test_confirmation_status is TestConfirmationStatus.CONFIRMED
        ]
        mapping = group[0].direction_mapping
        selected_bin_sets = [set(edge.selected_bin_ordinals) for edge in confirmed]
        supporting_bins = tuple(
            sorted(set.intersection(*selected_bin_sets)) if selected_bin_sets else ()
        )
        same_mapping = sum(edge.direction_mapping is mapping for edge in confirmed)
        stable = _ratio(same_mapping, len(selected)) if selected else Decimal(0)
        correlations = tuple(
            edge.test_metrics.correlation
            for edge in confirmed
            if edge.test_metrics.correlation is not None
        )
        total = sum(edge.test_metrics.usable_count for edge in confirmed)
        weighted = (
            sum(
                (
                    edge.test_metrics.correlation * edge.test_metrics.usable_count
                    for edge in confirmed
                    if edge.test_metrics.correlation is not None
                ),
                Decimal(0),
            )
            / total
            if total and correlations
            else None
        )
        hits = tuple(
            edge.test_metrics.directional_hit_rate
            for edge in confirmed
            if edge.test_metrics.directional_hit_rate is not None
        )
        aligned_gross = tuple(
            edge.test_metrics.mean_strategy_aligned_gross_return
            for edge in confirmed
            if edge.test_metrics.mean_strategy_aligned_gross_return is not None
        )
        aligned_net = tuple(
            edge.test_metrics.mean_strategy_aligned_net_return
            for edge in confirmed
            if edge.test_metrics.mean_strategy_aligned_net_return is not None
        )
        avoided = tuple(
            edge.test_metrics.mean_expected_avoided_downside
            for edge in confirmed
            if edge.test_metrics.mean_expected_avoided_downside is not None
        )
        classification = _classify_stability(
            group[0].predictor, selected, confirmed, stable, weighted, thresholds
        )
        output.append(
            StabilityEdge(
                group[0].predictor,
                horizon,
                condition,
                mapping,
                supporting_bins,
                len(group),
                len(selected),
                len(confirmed),
                stable,
                weighted,
                median(correlations) if correlations else None,
                min(correlations) if correlations else None,
                sum(hits, Decimal(0)) / len(hits) if hits else None,
                sum(
                    (
                        edge.test_metrics.positive_movement_after_friction_ratio
                        for edge in confirmed
                    ),
                    Decimal(0),
                )
                / len(confirmed)
                if confirmed
                else Decimal(0),
                sum(
                    (
                        edge.test_metrics.mean_net_movement_magnitude_after_friction
                        for edge in confirmed
                    ),
                    Decimal(0),
                )
                / len(confirmed)
                if confirmed
                else Decimal(0),
                sum(aligned_gross, Decimal(0)) / len(aligned_gross)
                if aligned_gross
                else None,
                sum(aligned_net, Decimal(0)) / len(aligned_net)
                if aligned_net
                else None,
                sum(avoided, Decimal(0)) / len(avoided) if avoided else None,
                _sample_std(correlations),
                total,
                classification,
            )
        )
    return tuple(
        sorted(
            output,
            key=lambda edge: (edge.predictor.name, edge.horizon, edge.condition.value),
        )
    )


def _classify_stability(
    predictor: PredictorDefinition,
    selected: list[WindowEdge],
    confirmed: list[WindowEdge],
    stable: Decimal,
    correlation: Decimal | None,
    thresholds: EvidenceThresholds,
) -> EvidenceStatus:
    if predictor.semantic_kind in (
        PredictorSemanticKind.MAGNITUDE_OR_ACTIVITY,
        PredictorSemanticKind.CONFIRMATION_ONLY,
    ):
        return EvidenceStatus.WEAK if confirmed else EvidenceStatus.NONE
    if len(selected) < thresholds.minimum_windows:
        return EvidenceStatus.INSUFFICIENT
    if (
        len(confirmed) < thresholds.minimum_windows
        or stable < thresholds.minimum_stable_direction_ratio
    ):
        return EvidenceStatus.UNSTABLE
    if (
        correlation is None
        or abs(correlation) < thresholds.minimum_absolute_correlation
    ):
        return EvidenceStatus.NONE
    return (
        EvidenceStatus.PROMISING_POSITIVE
        if correlation > 0
        else EvidenceStatus.PROMISING_NEGATIVE
    )


def _hypothesis(
    edge: StabilityEdge, ordinals: tuple[int, ...], cadence: timedelta
) -> StrategyHypothesis:
    positive = edge.classification is EvidenceStatus.PROMISING_POSITIVE
    key = _fingerprint(
        (
            edge.predictor.name,
            edge.predictor.version,
            edge.horizon.total_seconds(),
            edge.condition.value,
            edge.direction_mapping.value,
            edge.supporting_bin_ordinals,
            edge.classification.value,
        )
    )
    return StrategyHypothesis(
        key,
        edge.predictor.name,
        edge.predictor.version,
        "long-positive" if positive else "exit/avoid-negative",
        edge.horizon,
        edge.condition,
        edge.supporting_bin_ordinals,
        edge.mean_strategy_aligned_gross_return if positive else None,
        edge.mean_strategy_aligned_net_return if positive else None,
        None if positive else edge.mean_expected_avoided_downside,
        edge.mean_directional_hit_rate,
        edge.stable_direction_ratio,
        min(cadence, edge.horizon),
        edge.horizon / 2,
        edge.horizon,
        ordinals,
        edge.classification,
        f"Validation selected {edge.predictor.name}; isolated test confirmed the "
        f"frozen {edge.direction_mapping.value} mapping at {edge.horizon} under "
        f"{edge.condition.value}.",
        "Historical correlation is not causality or a guarantee of profitability; "
        "no entry threshold, short position, or position size is implied.",
    )


def _benchmarks(
    groups: dict[timedelta, list[Observation]],
) -> tuple[BenchmarkAggregate, ...]:
    output: list[BenchmarkAggregate] = []
    for horizon, values in sorted(groups.items()):
        count = len(values)
        mean = (
            sum((item.forward_gross_return for item in values), Decimal(0)) / count
            if count
            else Decimal(0)
        )
        benchmarks = _direction_benchmarks(tuple(values))
        for name, hit in (
            ("always_up", benchmarks.always_up),
            ("always_down", benchmarks.always_down),
            ("deterministic_random_v1", benchmarks.deterministic_random),
            ("observed_majority", benchmarks.observed_majority),
            ("no_signal", None),
            ("unconditional_asset_return", None),
        ):
            output.append(BenchmarkAggregate(horizon, name, count, hit, mean))
    return tuple(output)


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
    mean_x = sum(x, Decimal(0)) / len(x)
    mean_y = sum(y, Decimal(0)) / len(y)
    numerator = sum(
        ((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True)), Decimal(0)
    )
    x_variance = sum(((a - mean_x) ** 2 for a in x), Decimal(0))
    y_variance = sum(((b - mean_y) ** 2 for b in y), Decimal(0))
    if x_variance == 0 or y_variance == 0:
        return None
    return numerator / (x_variance * y_variance).sqrt()


def _sample_std(values: tuple[Decimal, ...]) -> Decimal | None:
    if len(values) < 2:
        return None
    mean_value = sum(values, Decimal(0)) / len(values)
    return (
        sum(((value - mean_value) ** 2 for value in values), Decimal(0))
        / (len(values) - 1)
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
