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

PREDICTIVE_EDGE_ALGORITHM_VERSION = 3
PREDICTOR_VERSION = 3
HYPOTHESIS_CONTRACT_VERSION = 2
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


class SuggestedAction(StrEnum):
    LONG_ENTRY = "long_entry"
    EXIT_OR_AVOID = "exit_or_avoid"
    DIAGNOSTIC_ONLY = "diagnostic_only"


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
class BinSelection:
    ordinal: int
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
    bin_selections: tuple[BinSelection, ...]
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
    bin_ordinal: int
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    suggested_action: SuggestedAction
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
    hypothesis_contract_version: int
    direction_mapping: DirectionMapping
    bin_ordinal: int
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    suggested_action: SuggestedAction
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
    context_predictor_name: str | None = None
    context_predictor_version: int | None = None
    context_regime: str | None = None
    context_lower_bound: Decimal | None = None
    context_upper_bound: Decimal | None = None


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
            PredictiveEdgeConclusion.INS÷;¶‰žËkºwµçh€€€Ñ¡É•Í¡½±‘ÌèÙ¥‘•¹•Q¡É•Í¡½±‘Ì°(€€€ÑÉ…¥¹¥¹}ÍÕ™™¥¥•¹Ðè‰½½°°(¤€´øÑÕÁ±•m	¥¹M•±•Ñ¥½¸°€¸¸¹tè(€€€Ù…±¥‘…Ñ¥½¹}É½ÕÁÌ€ô}‰¥¹}É½ÕÁÌ¡Ù…±¥‘…Ñ¥½¸°‰½Õ¹‘Ì¤(€€€Ñ•ÍÑ}É½ÕÁÌ€ô}‰¥¹}É½ÕÁÌ¡Ñ•ÍÐ°‰½Õ¹‘Ì¤(€€€Ù…±¥‘…Ñ¥½¹}‰•¹¡µ…É­Ì€ô}‘¥É•Ñ¥½¹}‰•¹¡µ…É­Ì¡Ù…±¥‘…Ñ¥½¸¤(€€€Ñ•ÍÑ}‰•¹¡µ…É­Ì€ô}‘¥É•Ñ¥½¹}‰•¹¡µ…É­Ì¡Ñ•ÍÐ¤(€€€µ¥¹¥µÕµ}‰¥¹}½‰Í•ÉÙ…Ñ¥½¹Ì€ôµ…à (€€€€€€€€È°Ñ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}½‰Í•ÉÙ…Ñ¥½¹Ì€¼¼€¡±•¸¡‰½Õ¹‘Ì¤€¬€Ä¤(€€€€¤(€€€½ÕÑÁÕÐè±¥ÍÑm	¥¹M•±•Ñ¥½¹t€ômt(€€€™½È¥¹‘•à°Ù…±¥‘…Ñ¥½¹}Ù…±Õ•Ì¥¸•¹Õµ•É…Ñ”¡Ù…±¥‘…Ñ¥½¹}É½ÕÁÌ¤è(€€€€€€€±½Ý•È€ô‰½Õ¹‘Ím¥¹‘•à€´€Åt¥˜¥¹‘•à•±Í”9½¹”(€€€€€€€ÕÁÁ•È€ô‰½Õ¹‘Ím¥¹‘•át¥˜¥¹‘•à€ð±•¸¡‰½Õ¹‘Ì¤•±Í”9½¹”(€€€€€€€…Ñ¥½¸€ô}ÍÕ•ÍÑ•‘}…Ñ¥½¸ (€€€€€€€€€€€±½Ý•É}‰½Õ¹õ±½Ý•È°ÕÁÁ•É}‰½Õ¹õÕÁÁ•È°•¹Ñ•Èõ•¹Ñ•È°µ…ÁÁ¥¹œõµ…ÁÁ¥¹œ(€€€€€€€€¤(€€€€€€€Ù…±¥‘…Ñ¥½¹}•Ù¥‘•¹”€ô}…Ñ¥½¹}•Ù¥‘•¹”¡Ù…±¥‘…Ñ¥½¹}Ù…±Õ•Ì°…Ñ¥½¸°™É¥Ñ¥½¸¤(€€€€€€€Ñ•ÍÑ}•Ù¥‘•¹”€ô}…Ñ¥½¹}•Ù¥‘•¹”¡Ñ•ÍÑ}É½ÕÁÍm¥¹‘•át°…Ñ¥½¸°™É¥Ñ¥½¸¤(€€€€€€€Ù}½Õ¹Ð°Ù}É½ÍÌ°Ù}±½¹}¹•Ð°Ù}…Ù½¥‘•°Ù}¡¥Ð€ôÙ…±¥‘…Ñ¥½¹}•Ù¥‘•¹”(€€€€€€€…Ñ¥½¹…‰±”€ô…Ñ¥½¸¥Ì¹½ÐMÕ•ÍÑ•‘Ñ¥½¸¹%9=MQ%}=91d(€€€€€€€Ù…±¥‘…Ñ¥½¹}‰•¹¡µ…É¬€ô€ (€€€€€€€€€€€Ù…±¥‘…Ñ¥½¹}‰•¹¡µ…É­Ì¹…±Ý…åÍ}ÕÀ(€€€€€€€€€€€¥˜…Ñ¥½¸¥ÌMÕ•ÍÑ•‘Ñ¥½¸¹1=9}9QId(€€€€€€€€€€€•±Í”Ù…±¥‘…Ñ¥½¹}‰•¹¡µ…É­Ì¹…±Ý…åÍ}‘½Ý¸(€€€€€€€€¤(€€€€€€€Ù…±¥‘…Ñ¥½¹}¥µÁÉ½Ù•Ì€ô‰½½° (€€€€€€€€€€€Ù}¡¥Ð¥Ì¹½Ð9½¹”(€€€€€€€€€€€…¹Ù}¡¥Ð(€€€€€€€€€€€€øôÙ…±¥‘…Ñ¥½¹}‰•¹¡µ…É¬(€€€€€€€€€€€€¬Ñ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}¡¥Ñ}É…Ñ•}¥µÁÉ½Ù•µ•¹Ñ}½Ù•É}‰•¹¡µ…É¬(€€€€€€€€¤(€€€€€€€™…Ù½É…‰±•}Ù…±¥‘…Ñ¥½¸€ô€ (€€€€€€€€€€€€¡Ù}±½¹}¹•Ð¥Ì¹½Ð9½¹”…¹Ù}±½¹}¹•Ð€ø€À¤(€€€€€€€€€€€¥˜…Ñ¥½¸¥ÌMÕ•ÍÑ•‘Ñ¥½¸¹1=9}9QId(€€€€€€€€€€€•±Í”€¡Ù}…Ù½¥‘•¥Ì¹½Ð9½¹”…¹Ù}…Ù½¥‘•€ø€À¤(€€€€€€€€¤(€€€€€€€Í•±•Ñ•€ô‰½½° (€€€€€€€€€€€ÑÉ…¥¹¥¹}ÍÕ™™¥¥•¹Ð(€€€€€€€€€€€…¹…Ñ¥½¹…‰±”(€€€€€€€€€€€…¹Ù}½Õ¹Ð€øôµ¥¹¥µÕµ}‰¥¹}½‰Í•ÉÙ…Ñ¥½¹Ì(€€€€€€€€€€€…¹Ù}¡¥Ð¥Ì¹½Ð9½¹”(€€€€€€€€€€€…¹Ù}¡¥Ð€øôÑ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”(€€€€€€€€€€€…¹Ù…±¥‘…Ñ¥½¹}¥µÁÉ½Ù•Ì(€€€€€€€€€€€…¹™…Ù½É…‰±•}Ù…±¥‘…Ñ¥½¸(€€€€€€€€¤(€€€€€€€Ñ}½Õ¹Ð°Ñ}É½ÍÌ°Ñ}±½¹}¹•Ð°Ñ}…Ù½¥‘•°Ñ}¡¥Ð€ôÑ•ÍÑ}•Ù¥‘•¹”(€€€€€€€™…Ù½É…‰±•}Ñ•ÍÐ€ô€ (€€€€€€€€€€€€¡Ñ}±½¹}¹•Ð¥Ì¹½Ð9½¹”…¹Ñ}±½¹}¹•Ð€ø€À¤(€€€€€€€€€€€¥˜…Ñ¥½¸¥ÌMÕ•ÍÑ•‘Ñ¥½¸¹1=9}9QId(€€€€€€€€€€€•±Í”€¡Ñ}…Ù½¥‘•¥Ì¹½Ð9½¹”…¹Ñ}…Ù½¥‘•€ø€À¤(€€€€€€€€¤(€€€€€€€Ñ•ÍÑ}‰•¹¡µ…É¬€ô€ (€€€€€€€€€€€Ñ•ÍÑ}‰•¹¡µ…É­Ì¹…±Ý…åÍ}ÕÀ(€€€€€€€€€€€¥˜…Ñ¥½¸¥ÌMÕ•ÍÑ•‘Ñ¥½¸¹1=9}9QId(€€€€€€€€€€€•±Í”Ñ•ÍÑ}‰•¹¡µ…É­Ì¹…±Ý…åÍ}‘½Ý¸(€€€€€€€€¤(€€€€€€€Ñ•ÍÑ}¥µÁÉ½Ù•Ì€ô‰½½° (€€€€€€€€€€€Ñ}¡¥Ð¥Ì¹½Ð9½¹”(€€€€€€€€€€€…¹Ñ}¡¥Ð(€€€€€€€€€€€€øôÑ•ÍÑ}‰•¹¡µ…É¬€¬Ñ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}¡¥Ñ}É…Ñ•}¥µÁÉ½Ù•µ•¹Ñ}½Ù•É}‰•¹¡µ…É¬(€€€€€€€€¤(€€€€€€€½¹™¥Éµ•€ô‰½½° (€€€€€€€€€€€Í•±•Ñ•(€€€€€€€€€€€…¹Ñ}½Õ¹Ð€øôµ¥¹¥µÕµ}‰¥¹}½‰Í•ÉÙ…Ñ¥½¹Ì(€€€€€€€€€€€…¹Ñ}¡¥Ð¥Ì¹½Ð9½¹”(€€€€€€€€€€€…¹Ñ}¡¥Ð€øôÑ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”(€€€€€€€€€€€…¹Ñ•ÍÑ}¥µÁÉ½Ù•Ì(€€€€€€€€€€€…¹™…Ù½É…‰±•}Ñ•ÍÐ(€€€€€€€€¤(€€€€€€€ÍÑ…ÑÕÌ€ô€ (€€€€€€€€€€€Q•ÍÑ½¹™¥Éµ…Ñ¥½¹MÑ…ÑÕÌ¹9=Q}M1Q(€€€€€€€€€€€¥˜¹½ÐÍ•±•Ñ•(€€€€€€€€€€€•±Í”Q•ÍÑ½¹™¥Éµ…Ñ¥½¹MÑ…ÑÕÌ¹%9MU%%9P(€€€€€€€€€€€¥˜Ñ}½Õ¹Ð€ðµ¥¹¥µÕµ}‰¥¹}½‰Í•ÉÙ…Ñ¥½¹Ì(€€€€€€€€€€€•±Í”Q•ÍÑ½¹™¥Éµ…Ñ¥½¹MÑ…ÑÕÌ¹=9%I5(€€€€€€€€€€€¥˜½¹™¥Éµ•(€€€€€€€€€€€•±Í”Q•ÍÑ½¹™¥Éµ…Ñ¥½¹MÑ…ÑÕÌ¹I)Q(€€€€€€€€¤(€€€€€€€É•…Í½¸€ô€ (€€€€€€€€€€€9½¹”(€€€€€€€€€€€¥˜½¹™¥Éµ•(€€€€€€€€€€€•±Í”€‰Ù…±¥‘…Ñ¥½¸‰¥¸‘¥¹½Ð¥¹‘•Á•¹‘•¹Ñ±ä±•…È•Ù¥‘•¹”Ñ¡É•Í¡½±‘Ìˆ(€€€€€€€€€€€¥˜¹½ÐÍ•±•Ñ•(€€€€€€€€€€€•±Í”˜‰É•ÅÕ¥É•Ì…Ð±•…ÍÐíµ¥¹¥µÕµ}‰¥¹}½‰Í•ÉÙ…Ñ¥½¹ÍôÑ•ÍÐ½‰Í•ÉÙ…Ñ¥½¹Ìˆ(€€€€€€€€€€€¥˜Ñ}½Õ¹Ð€ðµ¥¹¥µÕµ}‰¥¹}½‰Í•ÉÙ…Ñ¥½¹Ì(€€€€€€€€€€€•±Í”€‰™É½é•¸‰¥¸…Ñ¥½¸‘¥¹½Ð½¹™¥É´…™Ñ•È½ÍÑÌ¥¸Ñ•ÍÐˆ(€€€€€€€€¤(€€€€€€€½ÕÑÁÕÐ¹…ÁÁ•¹ (€€€€€€€€€€€	¥¹M•±•Ñ¥½¸ (€€€€€€€€€€€€€€€¥¹‘•à€¬€Ä°(€€€€€€€€€€€€€€€±½Ý•È°(€€€€€€€€€€€€€€€ÕÁÁ•È°(€€€€€€€€€€€€€€€…Ñ¥½¸°(€€€€€€€€€€€€€€€Ù}½Õ¹Ð°(€€€€€€€€€€€€€€€Ù}É½ÍÌ°(€€€€€€€€€€€€€€€Ù}±½¹}¹•Ð°(€€€€€€€€€€€€€€€Ù}…Ù½¥‘•°(€€€€€€€€€€€€€€€Ù}¡¥Ð°(€€€€€€€€€€€€€€€Í•±•Ñ•°(€€€€€€€€€€€€€€€Ñ}½Õ¹Ð°(€€€€€€€€€€€€€€€ÍÑ…ÑÕÌ°(€€€€€€€€€€€€€€€Ñ}É½ÍÌ°(€€€€€€€€€€€€€€€Ñ}±½¹}¹•Ð°(€€€€€€€€€€€€€€€Ñ}…Ù½¥‘•°(€€€€€€€€€€€€€€€Ñ}¡¥Ð°(€€€€€€€€€€€€€€€É•…Í½¸°(€€€€€€€€€€€€¤(€€€€€€€€¤(€€€É•ÑÕÉ¸ÑÕÁ±”¡½ÕÑÁÕÐ¤(()‘•˜}ÍÑ…‰¥±¥Ñä (€€€•‘•ÌèÑÕÁ±•m]¥¹‘½Ý‘”°€¸¸¹t°Ñ¡É•Í¡½±‘ÌèÙ¥‘•¹•Q¡É•Í¡½±‘Ì(¤€´øÑÕÁ±•mMÑ…‰¥±¥Ñå‘”°€¸¸¹tè(€€€É½ÕÁÌè‘¥Ñl(€€€€€€€ÑÕÁ±•mÍÑÈ°Ñ¥µ•‘•±Ñ„°½¹‘¥Ñ¥½¸°¥É•Ñ¥½¹5…ÁÁ¥¹œ°¥¹Ð°MÕ•ÍÑ•‘Ñ¥½¹t°(€€€€€€€±¥ÍÑmÑÕÁ±•m]¥¹‘½Ý‘”°	¥¹M•±•Ñ¥½¹ut°(€€€t€ôíô(€€€™½È•‘”¥¸•‘•Ìè(€€€€€€€™½ÈÍ•±•Ñ¥½¸¥¸•‘”¹‰¥¹}Í•±•Ñ¥½¹Ìè(€€€€€€€€€€€¥˜Í•±•Ñ¥½¸¹ÍÕ•ÍÑ•‘}…Ñ¥½¸¥ÌMÕ•ÍÑ•‘Ñ¥½¸¹%9=MQ%}=91dè(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€€€€€É½ÕÁÌ¹Í•Ñ‘•™…Õ±Ð (€€€€€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€€€€•‘”¹ÁÉ•‘¥Ñ½È¹¹…µ”°(€€€€€€€€€€€€€€€€€€€•‘”¹¡½É¥é½¸°(€€€€€€€€€€€€€€€€€€€•‘”¹½¹‘¥Ñ¥½¸°(€€€€€€€€€€€€€€€€€€€•‘”¹‘¥É•Ñ¥½¹}µ…ÁÁ¥¹œ°(€€€€€€€€€€€€€€€€€€€Í•±•Ñ¥½¸¹½É‘¥¹…°°(€€€€€€€€€€€€€€€€€€€Í•±•Ñ¥½¸¹ÍÕ•ÍÑ•‘}…Ñ¥½¸°(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€mt°(€€€€€€€€€€€€¤¹…ÁÁ•¹ ¡•‘”°Í•±•Ñ¥½¸¤¤(€€€½ÕÑÁÕÐè±¥ÍÑmMÑ…‰¥±¥Ñå‘•t€ômt(€€€™½È€ (€€€€€€€|°(€€€€€€€¡½É¥é½¸°(€€€€€€€½¹‘¥Ñ¥½¸°(€€€€€€€µ…ÁÁ¥¹œ°(€€€€€€€‰¥¹}½É‘¥¹…°°(€€€€€€€…Ñ¥½¸°(€€€€¤°É½ÕÀ¥¸É½ÕÁÌ¹¥Ñ•µÌ ¤è(€€€€€€€Í•±•Ñ•€ôm¥Ñ•´™½È¥Ñ•´¥¸É½ÕÀ¥˜¥Ñ•µlÅt¹Í•±•Ñ•‘}™½É}Ñ•ÍÑt(€€€€€€€½¹™¥Éµ•€ôl(€€€€€€€€€€€¥Ñ•´(€€€€€€€€€€€™½È¥Ñ•´¥¸Í•±•Ñ•(€€€€€€€€€€€¥˜¥Ñ•µlÅt¹Ñ•ÍÑ}½¹™¥Éµ…Ñ¥½¹}ÍÑ…ÑÕÌ¥ÌQ•ÍÑ½¹™¥Éµ…Ñ¥½¹MÑ…ÑÕÌ¹=9%I5(€€€€€€€t(€€€€€€€ÍÑ…‰±”€ô}É…Ñ¥¼¡±•¸¡½¹™¥Éµ•¤°±•¸¡Í•±•Ñ•¤¤¥˜Í•±•Ñ••±Í”•¥µ…° À¤(€€€€€€€½ÉÉ•±…Ñ¥½¹Ì€ôÑÕÁ±” (€€€€€€€€€€€•‘”¹Ñ•ÍÑ}µ•ÑÉ¥Ì¹½ÉÉ•±…Ñ¥½¸(€€€€€€€€€€€™½È•‘”°|¥¸½¹™¥Éµ•(€€€€€€€€€€€¥˜•‘”¹Ñ•ÍÑ}µ•ÑÉ¥Ì¹½ÉÉ•±…Ñ¥½¸¥Ì¹½Ð9½¹”(€€€€€€€€¤(€€€€€€€Ñ½Ñ…°€ôÍÕ´¡Í•±•Ñ¥½¸¹Ñ•ÍÑ}½‰Í•ÉÙ…Ñ¥½¹}½Õ¹Ð™½È|°Í•±•Ñ¥½¸¥¸½¹™¥Éµ•¤(€€€€€€€Ý•¥¡Ñ•€ô€ (€€€€€€€€€€€ÍÕ´ (€€€€€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€€€€•‘”¹Ñ•ÍÑ}µ•ÑÉ¥Ì¹½ÉÉ•±…Ñ¥½¸€¨Í•±•Ñ¥½¸¹Ñ•ÍÑ}½‰Í•ÉÙ…Ñ¥½¹}½Õ¹Ð(€€€€€€€€€€€€€€€€€€€™½È•‘”°Í•±•Ñ¥½¸¥¸½¹™¥Éµ•(€€€€€€€€€€€€€€€€€€€¥˜•‘”¹Ñ•ÍÑ}µ•ÑÉ¥Ì¹½ÉÉ•±…Ñ¥½¸¥Ì¹½Ð9½¹”(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€•¥µ…° À¤°(€€€€€€€€€€€€¤(€€€€€€€€€€€€¼Ñ½Ñ…°(€€€€€€€€€€€¥˜Ñ½Ñ…°…¹½ÉÉ•±…Ñ¥½¹Ì(€€€€€€€€€€€•±Í”9½¹”(€€€€€€€€¤(€€€€€€€¡¥ÑÌ€ôÑÕÁ±” (€€€€€€€€€€€Í•±•Ñ¥½¸¹Ñ•ÍÑ}‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”(€€€€€€€€€€€™½È|°Í•±•Ñ¥½¸¥¸½¹™¥Éµ•(€€€€€€€€€€€¥˜Í•±•Ñ¥½¸¹Ñ•ÍÑ}‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”¥Ì¹½Ð9½¹”(€€€€€€€€¤(€€€€€€€…±¥¹•‘}É½ÍÌ€ôÑÕÁ±” (€€€€€€€€€€€Í•±•Ñ¥½¸¹Ñ•ÍÑ}É½ÍÍ}É•ÑÕÉ¸(€€€€€€€€€€€™½È|°Í•±•Ñ¥½¸¥¸½¹™¥Éµ•(€€€€€€€€€€€¥˜…Ñ¥½¸¥ÌMÕ•ÍÑ•‘Ñ¥½¸¹1=9}9QId(€€€€€€€€¤(€€€€€€€…±¥¹•‘}¹•Ð€ôÑÕÁ±” (€€€€€€€€€€€Í•±•Ñ¥½¸¹Ñ•ÍÑ}±½¹}…±¥¹•‘}¹•Ñ}É•ÑÕÉ¸(€€€€€€€€€€€™½È|°Í•±•Ñ¥½¸¥¸½¹™¥Éµ•(€€€€€€€€€€€¥˜Í•±•Ñ¥½¸¹Ñ•ÍÑ}±½¹}…±¥¹•‘}¹•Ñ}É•ÑÕÉ¸¥Ì¹½Ð9½¹”(€€€€€€€€¤(€€€€€€€…Ù½¥‘•€ôÑÕÁ±” (€€€€€€€€€€€Í•±•Ñ¥½¸¹Ñ•ÍÑ}…Ù½¥‘•‘}‘½Ý¹Í¥‘”(€€€€€€€€€€€™½È|°Í•±•Ñ¥½¸¥¸½¹™¥Éµ•(€€€€€€€€€€€¥˜Í•±•Ñ¥½¸¹Ñ•ÍÑ}…Ù½¥‘•‘}‘½Ý¹Í¥‘”¥Ì¹½Ð9½¹”(€€€€€€€€¤(€€€€€€€±…ÍÍ¥™¥…Ñ¥½¸€ô}±…ÍÍ¥™å}ÍÑ…‰¥±¥Ñä (€€€€€€€€€€€…Ñ¥½¸°±•¸¡Í•±•Ñ•¤°±•¸¡½¹™¥Éµ•¤°ÍÑ…‰±”°Ñ¡É•Í¡½±‘Ì(€€€€€€€€¤(€€€€€€€±½Ý•É}Ù…±Õ•Ì€ôÑÕÁ±” (€€€€€€€€€€€Í•±•Ñ¥½¸¹±½Ý•É}‰½Õ¹(€€€€€€€€€€€™½È|°Í•±•Ñ¥½¸¥¸½¹™¥Éµ•½ÈÍ•±•Ñ•½ÈÉ½ÕÀ(€€€€€€€€€€€¥˜Í•±•Ñ¥½¸¹±½Ý•É}‰½Õ¹¥Ì¹½Ð9½¹”(€€€€€€€€¤(€€€€€€€ÕÁÁ•É}Ù…±Õ•Ì€ôÑÕÁ±” (€€€€€€€€€€€Í•±•Ñ¥½¸¹ÕÁÁ•É}‰½Õ¹(€€€€€€€€€€€™½È|°Í•±•Ñ¥½¸¥¸½¹™¥Éµ•½ÈÍ•±•Ñ•½ÈÉ½ÕÀ(€€€€€€€€€€€¥˜Í•±•Ñ¥½¸¹ÕÁÁ•É}‰½Õ¹¥Ì¹½Ð9½¹”(€€€€€€€€¤(€€€€€€€½ÕÑÁÕÐ¹…ÁÁ•¹ (€€€€€€€€€€€MÑ…‰¥±¥Ñå‘” (€€€€€€€€€€€€€€€É½ÕÁlÁulÁt¹ÁÉ•‘¥Ñ½È°(€€€€€€€€€€€€€€€¡½É¥é½¸°(€€€€€€€€€€€€€€€½¹‘¥Ñ¥½¸°(€€€€€€€€€€€€€€€µ…ÁÁ¥¹œ°(€€€€€€€€€€€€€€€‰¥¹}½É‘¥¹…°°(€€€€€€€€€€€€€€€µ•‘¥…¸¡±½Ý•É}Ù…±Õ•Ì¤¥˜±½Ý•É}Ù…±Õ•Ì•±Í”9½¹”°(€€€€€€€€€€€€€€€µ•‘¥…¸¡ÕÁÁ•É}Ù…±Õ•Ì¤¥˜ÕÁÁ•É}Ù…±Õ•Ì•±Í”9½¹”°(€€€€€€€€€€€€€€€…Ñ¥½¸°(€€€€€€€€€€€€€€€€¡‰¥¹}½É‘¥¹…°°¤°(€€€€€€€€€€€€€€€±•¸¡É½ÕÀ¤°(€€€€€€€€€€€€€€€±•¸¡Í•±•Ñ•¤°(€€€€€€€€€€€€€€€±•¸¡½¹™¥Éµ•¤°(€€€€€€€€€€€€€€€ÍÑ…‰±”°(€€€€€€€€€€€€€€€Ý•¥¡Ñ•°(€€€€€€€€€€€€€€€µ•‘¥…¸¡½ÉÉ•±…Ñ¥½¹Ì¤¥˜½ÉÉ•±…Ñ¥½¹Ì•±Í”9½¹”°(€€€€€€€€€€€€€€€µ¥¸¡½ÉÉ•±…Ñ¥½¹Ì¤¥˜½ÉÉ•±…Ñ¥½¹Ì•±Í”9½¹”°(€€€€€€€€€€€€€€€ÍÕ´¡¡¥ÑÌ°•¥µ…° À¤¤€¼±•¸¡¡¥ÑÌ¤¥˜¡¥ÑÌ•±Í”9½¹”°(€€€€€€€€€€€€€€€ÍÕ´ (€€€€€€€€€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€€€€€€€€•¥µ…° Ä¤(€€€€€€€€€€€€€€€€€€€€€€€¥˜€¡Í•±•Ñ¥½¸¹Ñ•ÍÑ}±½¹}…±¥¹•‘}¹•Ñ}É•ÑÕÉ¸½È•¥µ…° À¤¤€ø€À(€€€€€€€€€€€€€€€€€€€€€€€½È€¡Í•±•Ñ¥½¸¹Ñ•ÍÑ}…Ù½¥‘•‘}‘½Ý¹Í¥‘”½È•¥µ…° À¤¤€ø€À(€€€€€€€€€€€€€€€€€€€€€€€•±Í”•¥µ…° À¤(€€€€€€€€€€€€€€€€€€€€€€€™½È|°Í•±•Ñ¥½¸¥¸½¹™¥Éµ•(€€€€€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€€€€€•¥µ…° À¤°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€¼±•¸¡½¹™¥Éµ•¤(€€€€€€€€€€€€€€€¥˜½¹™¥Éµ•(€€€€€€€€€€€€€€€•±Í”•¥µ…° À¤°(€€€€€€€€€€€€€€€ÍÕ´ (€€€€€€€€€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€€€€€€€€Í•±•Ñ¥½¸¹Ñ•ÍÑ}±½¹}…±¥¹•‘}¹•Ñ}É•ÑÕÉ¸(€€€€€€€€€€€€€€€€€€€€€€€½ÈÍ•±•Ñ¥½¸¹Ñ•ÍÑ}…Ù½¥‘•‘}‘½Ý¹Í¥‘”(€€€€€€€€€€€€€€€€€€€€€€€½È•¥µ…° À¤(€€€€€€€€€€€€€€€€€€€€€€€™½È|°Í•±•Ñ¥½¸¥¸½¹™¥Éµ•(€€€€€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€€€€€•¥µ…° À¤°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€¼±•¸¡½¹™¥Éµ•¤(€€€€€€€€€€€€€€€¥˜½¹™¥Éµ•(€€€€€€€€€€€€€€€•±Í”•¥µ…° À¤°(€€€€€€€€€€€€€€€ÍÕ´¡…±¥¹•‘}É½ÍÌ°•¥µ…° À¤¤€¼±•¸¡…±¥¹•‘}É½ÍÌ¤(€€€€€€€€€€€€€€€¥˜…±¥¹•‘}É½ÍÌ(€€€€€€€€€€€€€€€•±Í”9½¹”°(€€€€€€€€€€€€€€€ÍÕ´¡…±¥¹•‘}¹•Ð°•¥µ…° À¤¤€¼±•¸¡…±¥¹•‘}¹•Ð¤(€€€€€€€€€€€€€€€¥˜…±¥¹•‘}¹•Ð(€€€€€€€€€€€€€€€•±Í”9½¹”°(€€€€€€€€€€€€€€€ÍÕ´¡…Ù½¥‘•°•¥µ…° À¤¤€¼±•¸¡…Ù½¥‘•¤¥˜…Ù½¥‘••±Í”9½¹”°(€€€€€€€€€€€€€€€}Í…µÁ±•}ÍÑ¡½ÉÉ•±…Ñ¥½¹Ì¤°(€€€€€€€€€€€€€€€Ñ½Ñ…°°(€€€€€€€€€€€€€€€±…ÍÍ¥™¥…Ñ¥½¸°(€€€€€€€€€€€€¤(€€€€€€€€¤(€€€É•ÑÕÉ¸ÑÕÁ±” (€€€€€€€Í½ÉÑ• (€€€€€€€€€€€½ÕÑÁÕÐ°(€€€€€€€€€€€­•äõ±…µ‰‘„•‘”è€¡•‘”¹ÁÉ•‘¥Ñ½È¹¹…µ”°•‘”¹¡½É¥é½¸°•‘”¹½¹‘¥Ñ¥½¸¹Ù…±Õ”¤°(€€€€€€€€¤(€€€€¤(()‘•˜}±…ÍÍ¥™å}ÍÑ…‰¥±¥Ñä (€€€…Ñ¥½¸èMÕ•ÍÑ•‘Ñ¥½¸°(€€€Í•±•Ñ•è¥¹Ð°(€€€½¹™¥Éµ•è¥¹Ð°(€€€ÍÑ…‰±”è•¥µ…°°(€€€Ñ¡É•Í¡½±‘ÌèÙ¥‘•¹•Q¡É•Í¡½±‘Ì°(¤€´øÙ¥‘•¹•MÑ…ÑÕÌè(€€€¥˜…Ñ¥½¸¥ÌMÕ•ÍÑ•‘Ñ¥½¸¹%9=MQ%}=91dè(€€€€€€€É•ÑÕÉ¸Ù¥‘•¹•MÑ…ÑÕÌ¹9=9(€€€¥˜Í•±•Ñ•€ðÑ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}Ý¥¹‘½ÝÌè(€€€€€€€É•ÑÕÉ¸Ù¥‘•¹•MÑ…ÑÕÌ¹%9MU%%9P(€€€¥˜€ (€€€€€€€½¹™¥Éµ•€ðÑ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}Ý¥¹‘½ÝÌ(€€€€€€€½ÈÍÑ…‰±”€ðÑ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}ÍÑ…‰±•}‘¥É•Ñ¥½¹}É…Ñ¥¼(€€€€¤è(€€€€€€€É•ÑÕÉ¸Ù¥‘•¹•MÑ…ÑÕÌ¹U9MQ	1(€€€É•ÑÕÉ¸€ (€€€€€€€Ù¥‘•¹•MÑ…ÑÕÌ¹AI=5%M%9}A=M%Q%Y(€€€€€€€¥˜…Ñ¥½¸¥ÌMÕ•ÍÑ•‘Ñ¥½¸¹1=9}9QId(€€€€€€€•±Í”Ù¥‘•¹•MÑ…ÑÕÌ¹AI=5%M%9}9Q%Y(€€€€¤(()‘•˜}¡åÁ½Ñ¡•Í¥Ì (€€€•‘”èMÑ…‰¥±¥Ñå‘”°½É‘¥¹…±ÌèÑÕÁ±•m¥¹Ð°€¸¸¹t°…‘•¹”èÑ¥µ•‘•±Ñ„(¤€´øMÑÉ…Ñ•å!åÁ½Ñ¡•Í¥Ìè(€€€­•ä€ô}™¥¹•ÉÁÉ¥¹Ð (€€€€€€€€ (€€€€€€€€€€€•‘”¹ÁÉ•‘¥Ñ½È¹¹…µ”°(€€€€€€€€€€€•‘”¹ÁÉ•‘¥Ñ½È¹Ù•ÉÍ¥½¸°(€€€€€€€€€€€!eA=Q!M%M}=9QIQ}YIM%=8°(€€€€€€€€€€€•‘”¹¡½É¥é½¸¹Ñ½Ñ…±}Í•½¹‘Ì ¤°(€€€€€€€€€€€•‘”¹½¹‘¥Ñ¥½¸¹Ù…±Õ”°(€€€€€€€€€€€•‘”¹‘¥É•Ñ¥½¹}µ…ÁÁ¥¹œ¹Ù…±Õ”°(€€€€€€€€€€€•‘”¹‰¥¹}½É‘¥¹…°°(€€€€€€€€€€€•‘”¹±½Ý•É}‰½Õ¹°(€€€€€€€€€€€•‘”¹ÕÁÁ•É}‰½Õ¹°(€€€€€€€€€€€•‘”¹ÍÕ•ÍÑ•‘}…Ñ¥½¸¹Ù…±Õ”°(€€€€€€€€¤(€€€€¤(€€€É•ÑÕÉ¸MÑÉ…Ñ•å!åÁ½Ñ¡•Í¥Ì (€€€€€€€­•ä°(€€€€€€€•‘”¹ÁÉ•‘¥Ñ½È¹¹…µ”°(€€€€€€€•‘”¹ÁÉ•‘¥Ñ½È¹Ù•ÉÍ¥½¸°(€€€€€€€!eA=Q!M%M}=9QIQ}YIM%=8°(€€€€€€€•‘”¹‘¥É•Ñ¥½¹}µ…ÁÁ¥¹œ°(€€€€€€€•‘”¹‰¥¹}½É‘¥¹…°°(€€€€€€€•‘”¹±½Ý•É}‰½Õ¹°(€€€€€€€•‘”¹ÕÁÁ•É}‰½Õ¹°(€€€€€€€•‘”¹ÍÕ•ÍÑ•‘}…Ñ¥½¸°(€€€€€€€•‘”¹¡½É¥é½¸°(€€€€€€€•‘”¹½¹‘¥Ñ¥½¸°(€€€€€€€•‘”¹ÍÕÁÁ½ÉÑ¥¹}‰¥¹}½É‘¥¹…±Ì°(€€€€€€€•‘”¹µ•…¹}ÍÑÉ…Ñ•å}…±¥¹•‘}É½ÍÍ}É•ÑÕÉ¸°(€€€€€€€•‘”¹µ•…¹}ÍÑÉ…Ñ•å}…±¥¹•‘}¹•Ñ}É•ÑÕÉ¸°(€€€€€€€•‘”¹µ•…¹}•áÁ•Ñ•‘}…Ù½¥‘•‘}‘½Ý¹Í¥‘”°(€€€€€€€•‘”¹µ•…¹}‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”°(€€€€€€€•‘”¹ÍÑ…‰±•}‘¥É•Ñ¥½¹}É…Ñ¥¼°(€€€€€€€µ¥¸¡…‘•¹”°•‘”¹¡½É¥é½¸¤°(€€€€€€€•‘”¹¡½É¥é½¸€¼€È°(€€€€€€€•‘”¹¡½É¥é½¸°(€€€€€€€½É‘¥¹…±Ì°(€€€€€€€•‘”¹±…ÍÍ¥™¥…Ñ¥½¸°(€€€€€€€˜‰Y…±¥‘…Ñ¥½¸Í•±•Ñ•‰¥¸í•‘”¹‰¥¹}½É‘¥¹…±ô™½È€ˆ(€€€€€€€˜‰í•‘”¹ÍÕ•ÍÑ•‘}…Ñ¥½¸¹Ù…±Õ•ôì€ˆ(€€€€€€€˜‰¥Í½±…Ñ•Ñ•ÍÐ½¹™¥Éµ•Ñ¡”Í…µ”™É½é•¸‰½Õ¹‘Ì…¹€ˆ(€€€€€€€˜‰í•‘”¹‘¥É•Ñ¥½¹}µ…ÁÁ¥¹œ¹Ù…±Õ•ôµ…ÁÁ¥¹œ…Ðí•‘”¹¡½É¥é½¹ôÕ¹‘•È€ˆ(€€€€€€€˜‰í•‘”¹½¹‘¥Ñ¥½¸¹Ù…±Õ•ô¸ˆ°(€€€€€€€€‰!¥ÍÑ½É¥…°½ÉÉ•±…Ñ¥½¸¥Ì¹½Ð…ÕÍ…±¥Ñä½È„Õ…É…¹Ñ•”½˜ÁÉ½™¥Ñ…‰¥±¥Ñäì€ˆ(€€€€€€€€‰¹¼•¹ÑÉäÑ¡É•Í¡½±°Í¡½ÉÐÁ½Í¥Ñ¥½¸°½ÈÁ½Í¥Ñ¥½¸Í¥é”¥Ì¥µÁ±¥•¸ˆ°(€€€€¤(()‘•˜}‰•¹¡µ…É­Ì (€€€É½ÕÁÌè‘¥ÑmÑ¥µ•‘•±Ñ„°±¥ÍÑm=‰Í•ÉÙ…Ñ¥½¹ut°(¤€´øÑÕÁ±•m	•¹¡µ…É­É•…Ñ”°€¸¸¹tè(€€€½ÕÑÁÕÐè±¥ÍÑm	•¹¡µ…É­É•…Ñ•t€ômt(€€€™½È¡½É¥é½¸°Ù…±Õ•Ì¥¸Í½ÉÑ•¡É½ÕÁÌ¹¥Ñ•µÌ ¤¤è(€€€€€€€½Õ¹Ð€ô±•¸¡Ù…±Õ•Ì¤(€€€€€€€µ•…¸€ô€ (€€€€€€€€€€€ÍÕ´ ¡¥Ñ•´¹™½ÉÝ…É‘}É½ÍÍ}É•ÑÕÉ¸™½È¥Ñ•´¥¸Ù…±Õ•Ì¤°•¥µ…° À¤¤€¼½Õ¹Ð(€€€€€€€€€€€¥˜½Õ¹Ð(€€€€€€€€€€€•±Í”•¥µ…° À¤(€€€€€€€€¤(€€€€€€€‰•¹¡µ…É­Ì€ô}‘¥É•Ñ¥½¹}‰•¹¡µ…É­Ì¡ÑÕÁ±”¡Ù…±Õ•Ì¤¤(€€€€€€€™½È¹…µ”°¡¥Ð¥¸€ (€€€€€€€€€€€€ ‰…±Ý…åÍ}ÕÀˆ°‰•¹¡µ…É­Ì¹…±Ý…åÍ}ÕÀ¤°(€€€€€€€€€€€€ ‰…±Ý…åÍ}‘½Ý¸ˆ°‰•¹¡µ…É­Ì¹…±Ý…åÍ}‘½Ý¸¤°(€€€€€€€€€€€€ ‰‘•Ñ•Éµ¥¹¥ÍÑ¥}É…¹‘½µ}ØÄˆ°‰•¹¡µ…É­Ì¹‘•Ñ•Éµ¥¹¥ÍÑ¥}É…¹‘½´¤°(€€€€€€€€€€€€ ‰½‰Í•ÉÙ•‘}µ…©½É¥Ñäˆ°‰•¹¡µ…É­Ì¹½‰Í•ÉÙ•‘}µ…©½É¥Ñä¤°(€€€€€€€€€€€€ ‰¹½}Í¥¹…°ˆ°9½¹”¤°(€€€€€€€€€€€€ ‰Õ¹½¹‘¥Ñ¥½¹…±}…ÍÍ•Ñ}É•ÑÕÉ¸ˆ°9½¹”¤°(€€€€€€€€¤è(€€€€€€€€€€€½ÕÑÁÕÐ¹…ÁÁ•¹¡	•¹¡µ…É­É•…Ñ”¡¡½É¥é½¸°¹…µ”°½Õ¹Ð°¡¥Ð°µ•…¸¤¤(€€€É•ÑÕÉ¸ÑÕÁ±”¡½ÕÑÁÕÐ¤(()‘•˜}ÅÕ…¹Ñ¥±•}‰½Õ¹‘Ì¡Ù…±Õ•ÌèÑÕÁ±•m•¥µ…°°€¸¸¹t°‰¥¹Ìè¥¹Ð¤€´øÑÕÁ±•m•¥µ…°°€¸¸¹tè(€€€¥˜¹½ÐÙ…±Õ•Ìè(€€€€€€€É•ÑÕÉ¸€ ¤(€€€½É‘•É•€ôÍ½ÉÑ•¡Ù…±Õ•Ì¤(€€€É•ÑÕÉ¸ÑÕÁ±” (€€€€€€€½É‘•É•‘mµ¥¸¡±•¸¡½É‘•É•¤€´€Ä°€¡±•¸¡½É‘•É•¤€¨¥¹‘•à¤€¼¼‰¥¹Ì¥t(€€€€€€€™½È¥¹‘•à¥¸É…¹” Ä°‰¥¹Ì¤(€€€€¤(()‘•˜}ÅÕ…¹Ñ¥±”¡Ù…±Õ•ÌèÑÕÁ±•m•¥µ…°°€¸¸¹t°Äè•¥µ…°¤€´ø•¥µ…°è(€€€¥˜¹½ÐÙ…±Õ•Ìè(€€€€€€€É…¥Í”AÉ•‘¥Ñ¥Ù•‘•ÉÉ½È ‰ÑÉ…¥¹¥¹œÁ¡…Í”½¹Ñ…¥¹Ì¹¼•Ù¥‘•¹”ˆ¤(€€€½É‘•É•€ôÍ½ÉÑ•¡Ù…±Õ•Ì¤(€€€É•ÑÕÉ¸½É‘•É•‘mµ¥¸¡±•¸¡½É‘•É•¤€´€Ä°¥¹Ð¡•¥µ…°¡±•¸¡½É‘•É•¤¤€¨Ä¤¥t(()‘•˜}½ÉÉ•±…Ñ¥½¸¡àèÑÕÁ±•m•¥µ…°°€¸¸¹t°äèÑÕÁ±•m•¥µ…°°€¸¸¹t¤€´ø•¥µ…°ð9½¹”è(€€€¥˜±•¸¡à¤€ð€Èè(€€€€€€€É•ÑÕÉ¸9½¹”(€€€µ•…¹}à€ôÍÕ´¡à°•¥µ…° À¤¤€¼±•¸¡à¤(€€€µ•…¹}ä€ôÍÕ´¡ä°•¥µ…° À¤¤€¼±•¸¡ä¤(€€€¹Õµ•É…Ñ½È€ôÍÕ´ (€€€€€€€€ ¡„€´µ•…¹}à¤€¨€¡ˆ€´µ•…¹}ä¤™½È„°ˆ¥¸é¥À¡à°ä°ÍÑÉ¥ÐõQÉÕ”¤¤°•¥µ…° À¤(€€€€¤(€€€á}Ù…É¥…¹”€ôÍÕ´  ¡„€´µ•…¹}à¤€¨¨€È™½È„¥¸à¤°•¥µ…° À¤¤(€€€å}Ù…É¥…¹”€ôÍÕ´  ¡ˆ€´µ•…¹}ä¤€¨¨€È™½Èˆ¥¸ä¤°•¥µ…° À¤¤(€€€¥˜á}Ù…É¥…¹”€ôô€À½Èå}Ù…É¥…¹”€ôô€Àè(€€€€€€€É•ÑÕÉ¸9½¹”(€€€É•ÑÕÉ¸¹Õµ•É…Ñ½È€¼€¡á}Ù…É¥…¹”€¨å}Ù…É¥…¹”¤¹ÍÅÉÐ ¤(()‘•˜}Í…µÁ±•}ÍÑ¡Ù…±Õ•ÌèÑÕÁ±•m•¥µ…°°€¸¸¹t¤€´ø•¥µ…°ð9½¹”è(€€€¥˜±•¸¡Ù…±Õ•Ì¤€ð€Èè(€€€€€€€É•ÑÕÉ¸9½¹”(€€€µ•…¹}Ù…±Õ”€ôÍÕ´¡Ù…±Õ•Ì°•¥µ…° À¤¤€¼±•¸¡Ù…±Õ•Ì¤(€€€É•ÑÕÉ¸€ (€€€€€€€ÍÕ´  ¡Ù…±Õ”€´µ•…¹}Ù…±Õ”¤€¨¨€È™½ÈÙ…±Õ”¥¸Ù…±Õ•Ì¤°•¥µ…° À¤¤(€€€€€€€€¼€¡±•¸¡Ù…±Õ•Ì¤€´€Ä¤(€€€€¤¹ÍÅÉÐ ¤(()‘•˜}É…Ñ¥¼¡¹Õµ•É…Ñ½Èè¥¹Ð°‘•¹½µ¥¹…Ñ½Èè¥¹Ð¤€´ø•¥µ…°è(€€€É•ÑÕÉ¸€¡•¥µ…°¡¹Õµ•É…Ñ½È¤€¼•¥µ…°¡‘•¹½µ¥¹…Ñ½È¤¤¹ÅÕ…¹Ñ¥é” (€€€€€€€IQ%=}EU9QU4°É½Õ¹‘¥¹œõI=U9}!1}Y8(€€€€¤(()‘•˜}™¥¹•ÉÁÉ¥¹Ð¡Ù…±Õ”è½‰©•Ð¤€´øÍÑÈè(€€€É•ÑÕÉ¸¡…Í¡±¥ˆ¹Í¡„ÈÔØ (€€€€€€€©Í½¸¹‘ÕµÁÌ (€€€€€€€€€€€Ù…±Õ”°(€€€€€€€€€€€‘•™…Õ±Ðõ±…µ‰‘„¥Ñ•´è€ (€€€€€€€€€€€€€€€…Í‘¥Ð¡¥Ñ•´¤(€€€€€€€€€€€€€€€¥˜¡…Í…ÑÑÈ¡¥Ñ•´°€‰}}‘…Ñ…±…ÍÍ}™¥•±‘Í}|ˆ¤(€€€€€€€€€€€€€€€•±Í”¥Ñ•´¹Ñ½Ñ…±}Í•½¹‘Ì ¤(€€€€€€€€€€€€€€€¥˜¥Í¥¹ÍÑ…¹”¡¥Ñ•´°Ñ¥µ•‘•±Ñ„¤(€€€€€€€€€€€€€€€•±Í”¥Ñ•´¹¥Í½™½Éµ…Ð ¤(€€€€€€€€€€€€€€€¥˜¥Í¥¹ÍÑ…¹”¡¥Ñ•´°‘…Ñ•Ñ¥µ”¤(€€€€€€€€€€€€€€€•±Í”ÍÑÈ¡¥Ñ•´¤(€€€€€€€€€€€€¤°(€€€€€€€€€€€Í½ÉÑ}­•åÌõQÉÕ”°(€€€€€€€€€€€Í•Á…É…Ñ½ÉÌô ˆ°ˆ°€ˆèˆ¤°(€€€€€€€€¤¹•¹½‘” ¤(€€€€¤¹¡•á‘¥•ÍÐ ¤(