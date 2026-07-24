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
    frictionãÏ4¶‰žËkºwµç\°±•¸¡Á½½°¤¤(€€€™½È‘•ÁÑ ¥¸É…¹” È°µ…á¥µÕµ}‘•ÁÑ €¬€Ä¤è(€€€€€€€ÁÉ½Á½Í…±Ìè±¥ÍÑmÑÕÁ±•mÑÕÁ±•mÍÑÈ°€¸¸¹t°½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•I•ÍÕ±Ðð9½¹•ut€ômt(€€€€€€€¥˜‘•ÁÑ €ôô€Èè(€€€€€€€€€€€ÁÉ½Á½Í…±Ì€ôl¡ÍÕ‰Í•Ð°9½¹”¤™½ÈÍÕ‰Í•Ð¥¸½µ‰¥¹…Ñ¥½¹Ì¡Á½½°°€È¥t(€€€€€€€•±Í”è(€€€€€€€€€€€™½ÈÉ•Ñ…¥¹•‘}Á…É•¹Ð¥¸Á…É•¹ÑÌè(€€€€€€€€€€€€€€€™½È¹…µ”¥¸Á½½°è(€€€€€€€€€€€€€€€€€€€¥˜¹…µ”¹½Ð¥¸É•Ñ…¥¹•‘}Á…É•¹Ð¹…¹‘¥‘…Ñ”¹™•…ÑÕÉ•Ìè(€€€€€€€€€€€€€€€€€€€€€€€ÁÉ½Á½Í…±Ì¹…ÁÁ•¹ (€€€€€€€€€€€€€€€€€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ÑÕÁ±” (€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Í½ÉÑ•  ©É•Ñ…¥¹•‘}Á…É•¹Ð¹…¹‘¥‘…Ñ”¹™•…ÑÕÉ•Ì°¹…µ”¤¤(€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€É•Ñ…¥¹•‘}Á…É•¹Ð°(€€€€€€€€€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€¥˜¹½ÐÁÉ½Á½Í…±Ìè(€€€€€€€€€€€‰É•…¬(€€€€€€€ÁÉ½Á½Í…±}­•åÌèÍ•ÑmÑÕÁ±•mÑÕÁ±•mÍÑÈ°€¸¸¹t°ÍÑÈð9½¹•ut€ôÍ•Ð ¤(€€€€€€€Õ¹¥ÅÕ•}ÁÉ½Á½Í…±Ìè±¥ÍÑl(€€€€€€€€€€€ÑÕÁ±•mÑÕÁ±•mÍÑÈ°€¸¸¹t°½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•I•ÍÕ±Ðð9½¹•t(€€€€€€€t€ômt(€€€€€€€™½ÈÍÕ‰Í•Ð°ÁÉ½Á½Í…±}Á…É•¹Ð¥¸ÁÉ½Á½Í…±Ìè(€€€€€€€€€€€­•ä€ô€ (€€€€€€€€€€€€€€€ÍÕ‰Í•Ð°(€€€€€€€€€€€€€€€ÁÉ½Á½Í…±}Á…É•¹Ð¹…¹‘¥‘…Ñ”¹™¥¹•ÉÁÉ¥¹Ð¥˜ÁÉ½Á½Í…±}Á…É•¹Ð•±Í”9½¹”°(€€€€€€€€€€€€¤(€€€€€€€€€€€¥˜­•ä¹½Ð¥¸ÁÉ½Á½Í…±}­•åÌè(€€€€€€€€€€€€€€€ÁÉ½Á½Í…±}­•åÌ¹…‘¡­•ä¤(€€€€€€€€€€€€€€€Õ¹¥ÅÕ•}ÁÉ½Á½Í…±Ì¹…ÁÁ•¹ ¡ÍÕ‰Í•Ð°ÁÉ½Á½Í…±}Á…É•¹Ð¤¤(€€€€€€€Õ¹¥ÅÕ•}ÁÉ½Á½Í…±Ì¹Í½ÉÐ (€€€€€€€€€€€­•äõ±…µ‰‘„Ù…±Õ”è€ (€€€€€€€€€€€€€€€€µÍÕ´ ¡…‰Ì¡•Ù¥‘•¹•m¹…µ•t¤™½È¹…µ”¥¸Ù…±Õ•lÁt¤°•¥µ…° À¤¤°(€€€€€€€€€€€€€€€€´¡Ù…±Õ•lÅt¹É…¹­¥¹œ¹Ñ½Ñ…°¥˜Ù…±Õ•lÅt¥Ì¹½Ð9½¹”•±Í”•¥µ…° À¤¤°(€€€€€€€€€€€€€€€Ù…±Õ•lÁt°(€€€€€€€€€€€€€€€Ù…±Õ•lÅt¹…¹‘¥‘…Ñ”¹™¥¹•ÉÁÉ¥¹Ð¥˜Ù…±Õ•lÅt¥Ì¹½Ð9½¹”•±Í”€ˆˆ°(€€€€€€€€€€€€¤(€€€€€€€€¤(€€€€€€€É•µ…¥¹¥¹}‘•ÁÑ¡Ì€ôµ…á¥µÕµ}‘•ÁÑ €´‘•ÁÑ €¬€Ä(€€€€€€€É•µ…¥¹¥¹}¡•…À€ô‰Õ‘•Ð¹µ…á¥µÕµ}¡•…Á}•Ù…±Õ…Ñ¥½¹Ì€´¡•…Á}½µÁÕÑ•(€€€€€€€¥˜É•µ…¥¹¥¹}¡•…À€ðô€Àè(€€€€€€€€€€€‰Õ‘•Ñ}•á¡…ÕÍÑ•€ôQÉÕ”(€€€€€€€€€€€‰É•…¬(€€€€€€€‘•ÁÑ¡}¡•…Á}±¥µ¥Ð€ôµ…à Ä°É•µ…¥¹¥¹}¡•…À€¼¼É•µ…¥¹¥¹}‘•ÁÑ¡Ì¤(€€€€€€€‘•ÁÑ¡}¡•…Á}ÍÑ½À€ô¡•…Á}½µÁÕÑ•€¬‘•ÁÑ¡}¡•…Á}±¥µ¥Ð(€€€€€€€¡•…Á}É…¹­•è±¥ÍÑmÑÕÁ±•m½µÁ½Í¥Ñ•…¹‘¥‘…Ñ”°ÍÑÈð9½¹”°½µÁ½Í¥Ñ•5•ÑÉ¥Íut€ômt(€€€€€€€½¹™¥ÕÉ…Ñ¥½¹Ì€ô€À(€€€€€€€‘ÕÁ±¥…Ñ•Ì€ô€À(€€€€€€€™½ÈÍÕ‰Í•Ð°ÁÉ½Á½Í…±}Á…É•¹Ð¥¸Õ¹¥ÅÕ•}ÁÉ½Á½Í…±Ìè(€€€€€€€€€€€¥˜¡•…Á}½µÁÕÑ•€øô‘•ÁÑ¡}¡•…Á}ÍÑ½Àè(€€€€€€€€€€€€€€€‰Õ‘•Ñ}•á¡…ÕÍÑ•€ôQÉÕ”(€€€€€€€€€€€€€€€‰É•…¬(€€€€€€€€€€€™½ÈÝ•¥¡ÑÌ¥¸}Ý•¥¡Ñ}Ù…É¥…¹ÑÌ (€€€€€€€€€€€€€€€ÍÕ‰Í•Ð°(€€€€€€€€€€€€€€€•Ù¥‘•¹”°(€€€€€€€€€€€€€€€ÁÉ½Á½Í…±}Á…É•¹Ð¹…¹‘¥‘…Ñ”¥˜ÁÉ½Á½Í…±}Á…É•¹Ð•±Í”9½¹”°(€€€€€€€€€€€€€€€‰Õ‘•Ð¹µ…á¥µÕµ}Ù…É¥…¹ÑÍ}Á•É}ÍÕ‰Í•Ð°(€€€€€€€€€€€€¤è(€€€€€€€€€€€€€€€™½È€ (€€€€€€€€€€€€€€€€€€€…‘•¹”°(€€€€€€€€€€€€€€€€€€€¡½É¥é½¸°(€€€€€€€€€€€€€€€€€€€•¹ÑÉä°(€€€€€€€€€€€€€€€€€€€•á¥Ñ|°(€€€€€€€€€€€€€€€€€€€µ¥¹¥µÕ´°(€€€€€€€€€€€€€€€€€€€µ…á¥µÕ´°(€€€€€€€€€€€€€€€€€€€½½±‘½Ý¸°(€€€€€€€€€€€€€€€€¤¥¸}=AIQ%=91}=9%UIQ%=9Lè(€€€€€€€€€€€€€€€€€€€¥˜¡•…Á}½µÁÕÑ•€øô‘•ÁÑ¡}¡•…Á}ÍÑ½Àè(€€€€€€€€€€€€€€€€€€€€€€€‰Õ‘•Ñ}•á¡…ÕÍÑ•€ôQÉÕ”(€€€€€€€€€€€€€€€€€€€€€€€‰É•…¬(€€€€€€€€€€€€€€€€€€€½¹™¥ÕÉ…Ñ¥½¹Ì€¬ô€Ä(€€€€€€€€€€€€€€€€€€€…¹‘¥‘…Ñ”€ô‰Õ¥±‘}…¹‘¥‘…Ñ” (€€€€€€€€€€€€€€€€€€€€€€€Ý•¥¡ÑÌõÝ•¥¡ÑÌ°(€€€€€€€€€€€€€€€€€€€€€€€ÑÉ…¥¹¥¹}É½ÝÌõÑÉ…¥¹¥¹}É½ÝÌ°(€€€€€€€€€€€€€€€€€€€€€€€Í¥µÕ±…Ñ¥½¸õÍ¥µÕ±…Ñ¥½¸°(€€€€€€€€€€€€€€€€€€€€€€€…‘•¹”õ…‘•¹”°(€€€€€€€€€€€€€€€€€€€€€€€¡½É¥é½¸õ¡½É¥é½¸°(€€€€€€€€€€€€€€€€€€€€€€€•¹ÑÉå}Ñ¡É•Í¡½±õ•¹ÑÉä°(€€€€€€€€€€€€€€€€€€€€€€€•á¥Ñ}Ñ¡É•Í¡½±õ•á¥Ñ|°(€€€€€€€€€€€€€€€€€€€€€€€µ¥¹¥µÕµ}¡½±‘¥¹}Á•É¥½õµ¥¹¥µÕ´°(€€€€€€€€€€€€€€€€€€€€€€€µ…á¥µÕµ}¡½±‘¥¹}Á•É¥½õµ…á¥µÕ´°(€€€€€€€€€€€€€€€€€€€€€€€½½±‘½Ý¸õ½½±‘½Ý¸°(€€€€€€€€€€€€€€€€€€€€€€€¥¹¥Ñ¥…±}…Í õ¥¹¥Ñ¥…±}…Í °(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€¥˜…¹‘¥‘…Ñ”¹™¥¹•ÉÁÉ¥¹Ð¥¸Í••¸è(€€€€€€€€€€€€€€€€€€€€€€€‘ÕÁ±¥…Ñ•Ì€¬ô€Ä(€€€€€€€€€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€€€€€€€€€€€€€Í••¸¹…‘¡…¹‘¥‘…Ñ”¹™¥¹•ÉÁÉ¥¹Ð¤(€€€€€€€€€€€€€€€€€€€¡•…Á}½µÁÕÑ•€¬ô€Ä(€€€€€€€€€€€€€€€€€€€¡•…À€ô}¡•…Á}Ý¥¹‘½Ý}µ•ÑÉ¥Ì (€€€€€€€€€€€€€€€€€€€€€€€…¹‘¥‘…Ñ”°µ…ÑÉ¥à°Ý¥¹‘½ÝÌ°€‰ÑÉ…¥¹¥¹œˆ(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€¡•…Á}É…¹­•¹…ÁÁ•¹ (€€€€€€€€€€€€€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€€€€€€€€€€€€…¹‘¥‘…Ñ”°(€€€€€€€€€€€€€€€€€€€€€€€€€€€ÁÉ½Á½Í…±}Á…É•¹Ð¹…¹‘¥‘…Ñ”¹™¥¹•ÉÁÉ¥¹Ð(€€€€€€€€€€€€€€€€€€€€€€€€€€€¥˜ÁÉ½Á½Í…±}Á…É•¹Ð(€€€€€€€€€€€€€€€€€€€€€€€€€€€•±Í”9½¹”°(€€€€€€€€€€€€€€€€€€€€€€€€€€€¡•…À°(€€€€€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€¡•…Á}É…¹­•¹Í½ÉÐ (€€€€€€€€€€€­•äõ±…µ‰‘„Ù…±Õ”è€ (€€€€€€€€€€€€€€€€µ}É…¹­¥¹œ¡Ù…±Õ•lÁt°Ù…±Õ•lÉt¤¹Ñ½Ñ…°°(€€€€€€€€€€€€€€€Ù…±Õ•lÁt¹™¥¹•ÉÁÉ¥¹Ð°(€€€€€€€€€€€€¤(€€€€€€€€¤(€€€€€€€¥˜¹½Ð¡•…Á}É…¹­•è(€€€€€€€€€€€‰É•…¬(€€€€€€€É•µ…¥¹¥¹}ÑÉ…¥¹¥¹œ€ô‰Õ‘•Ð¹µ…á¥µÕµ}ÑÉ…¥¹¥¹}Í¥µÕ±…Ñ¥½¹Ì€´ÑÉ…¥¹¥¹}½µÁÕÑ•(€€€€€€€¥˜É•µ…¥¹¥¹}ÑÉ…¥¹¥¹œ€ðô€Àè(€€€€€€€€€€€‰Õ‘•Ñ}•á¡…ÕÍÑ•€ôQÉÕ”(€€€€€€€€€€€‰É•…¬(€€€€€€€‘•ÁÑ¡}ÑÉ…¥¹¥¹}±¥µ¥Ð€ôµ…à Ä°É•µ…¥¹¥¹}ÑÉ…¥¹¥¹œ€¼¼É•µ…¥¹¥¹}‘•ÁÑ¡Ì¤(€€€€€€€Í¥µÕ±…Ñ¥½¹}Í±½ÑÌ€ôµ¥¸ (€€€€€€€€€€€±•¸¡¡•…Á}É…¹­•¤°(€€€€€€€€€€€‰Õ‘•Ð¹‰•…µ}Ý¥‘Ñ¡}Á•É}‘•ÁÑ €¨€È°(€€€€€€€€€€€‘•ÁÑ¡}ÑÉ…¥¹¥¹}±¥µ¥Ð°(€€€€€€€€¤(€€€€€€€¥˜Í¥µÕ±…Ñ¥½¹}Í±½ÑÌ€ð±•¸¡¡•…Á}É…¹­•¤è(€€€€€€€€€€€‰Õ‘•Ñ}•á¡…ÕÍÑ•€ôQÉÕ”(€€€€€€€‘•ÁÑ¡}É•ÍÕ±ÑÌè±¥ÍÑm½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•I•ÍÕ±Ñt€ômt(€€€€€€€™½È…¹‘¥‘…Ñ”°Á…É•¹Ñ}™¥¹•ÉÁÉ¥¹Ð°}¡•…À¥¸¡•…Á}É…¹­•‘léÍ¥µÕ±…Ñ¥½¹}Í±½ÑÍtè(€€€€€€€€€€€ÑÉ…¥¹¥¹œ€ô}•½¹½µ¥}µ•ÑÉ¥Ì (€€€€€€€€€€€€€€€…¹‘¥‘…Ñ”õ…¹‘¥‘…Ñ”°(€€€€€€€€€€€€€€€‘…Ñ…Í•Ðõ‘…Ñ…Í•Ð°(€€€€€€€€€€€€€€€µ…ÑÉ¥àõµ…ÑÉ¥à°(€€€€€€€€€€€€€€€Ý¥¹‘½ÝÌõÝ¥¹‘½ÝÌ°(€€€€€€€€€€€€€€€Á¡…Í”ô‰ÑÉ…¥¹¥¹œˆ°(€€€€€€€€€€€€€€€Í¥µÕ±…Ñ¥½¸õÍ¥µÕ±…Ñ¥½¸°(€€€€€€€€€€€€€€€¥¹¥Ñ¥…±}…Í õ¥¹¥Ñ¥…±}…Í °(€€€€€€€€€€€€¤(€€€€€€€€€€€ÑÉ…¥¹¥¹}½µÁÕÑ•€¬ô€Ä(€€€€€€€€€€€ÑÉ…¥¹¥¹}Á…ÍÍ•Ì€ô€ (€€€€€€€€€€€€€€€ÑÉ…¥¹¥¹œ¹ÕÍ…‰±•}Ý¥¹‘½Ý}½Õ¹Ð€øôÑ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}Ý¥¹‘½ÝÌ(€€€€€€€€€€€€€€€…¹ÑÉ…¥¹¥¹œ¹Ñ½Ñ…±}É•ÑÕÉ¸€ø€À(€€€€€€€€€€€€€€€…¹ÑÉ…¥¹¥¹œ¹ÑÉ…‘•}½Õ¹Ð€øôÑÉ…¥¹¥¹œ¹ÕÍ…‰±•}Ý¥¹‘½Ý}½Õ¹Ð(€€€€€€€€€€€€€€€…¹ÑÉ…¥¹¥¹œ¹µ…á}‘É…Ý‘½Ý¸€ðô•¥µ…° ˆÀ¸ÌÔˆ¤(€€€€€€€€€€€€€€€…¹ÑÉ…¥¹¥¹œ¹É•ÑÕÉ¹}ÍÑ…‰¥±¥Ñä(€€€€€€€€€€€€€€€€øôÑ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}ÍÑ…‰±•}‘¥É•Ñ¥½¹}É…Ñ¥¼(€€€€€€€€€€€€¤(€€€€€€€€€€€™…¥±•€ô€ ¤¥˜ÑÉ…¥¹¥¹}Á…ÍÍ•Ì•±Í”€ ‰ÑÉ…¥¹¥¹}•½¹½µ¥}ÅÕ…±¥™¥…Ñ¥½¸ˆ°¤(€€€€€€€€€€€‘•ÁÑ¡}É•ÍÕ±ÑÌ¹…ÁÁ•¹ (€€€€€€€€€€€€€€€½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•I•ÍÕ±Ð (€€€€€€€€€€€€€€€€€€€…¹‘¥‘…Ñ”°(€€€€€€€€€€€€€€€€€€€Á…É•¹Ñ}™¥¹•ÉÁÉ¥¹Ð°(€€€€€€€€€€€€€€€€€€€‘•ÁÑ °(€€€€€€€€€€€€€€€€€€€½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹IQ%9}QI%9%9°(€€€€€€€€€€€€€€€€€€€}É…¹­¥¹œ¡…¹‘¥‘…Ñ”°ÑÉ…¥¹¥¹œ¤°(€€€€€€€€€€€€€€€€€€€ÑÉ…¥¹¥¹œ°(€€€€€€€€€€€€€€€€€€€9½¹”°(€€€€€€€€€€€€€€€€€€€9½¹”°(€€€€€€€€€€€€€€€€€€€€ ‰ÑÉ…¥¹¥¹}•½¹½µ¥}ÅÕ…±¥™¥…Ñ¥½¸ˆ°¤¥˜ÑÉ…¥¹¥¹}Á…ÍÍ•Ì•±Í”€ ¤°(€€€€€€€€€€€€€€€€€€€™…¥±•°(€€€€€€€€€€€€€€€€€€€™…¥±•‘lÁt¥˜™…¥±••±Í”9½¹”°(€€€€€€€€€€€€€€€€€€€•¥µ…° À¤¥˜ÑÉ…¥¹¥¹}Á…ÍÍ•Ì•±Í”•¥µ…° Ä¤°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€¤(€€€€€€€‘•ÁÑ¡}É•ÍÕ±ÑÌ¹Í½ÉÐ (€€€€€€€€€€€­•äõ±…µ‰‘„¥Ñ•´è€ µ¥Ñ•´¹É…¹­¥¹œ¹Ñ½Ñ…°°¥Ñ•´¹…¹‘¥‘…Ñ”¹™¥¹•ÉÁÉ¥¹Ð¤(€€€€€€€€¤(€€€€€€€Á…É•¹ÑÌ€ôÑÕÁ±”¡¥Ñ•´™½È¥Ñ•´¥¸‘•ÁÑ¡}É•ÍÕ±ÑÌ¥˜¹½Ð¥Ñ•´¹™…¥±•‘}É•ÅÕ¥É•µ•¹ÑÌ¥l(€€€€€€€€€€€€è‰Õ‘•Ð¹‰•…µ}Ý¥‘Ñ¡}Á•É}‘•ÁÑ (€€€€€€€t(€€€€€€€Ù…±¥‘…Ñ¥½¹}±¥µ¥Ð€ôµ¥¸¡±•¸¡Á…É•¹ÑÌ¤°‰Õ‘•Ð¹µ…á¥µÕµ}Ù…±¥‘…Ñ¥½¹}Á•É}‘•ÁÑ ¤(€€€€€€€™¥¹…±¥é•è±¥ÍÑm½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•I•ÍÕ±Ñt€ômt(€€€€€€€™½È¥Ñ•´¥¸Á…É•¹ÑÍléÙ…±¥‘…Ñ¥½¹}±¥µ¥Ñtè(€€€€€€€€€€€Ù…±¥‘…Ñ¥½¸€ô}•½¹½µ¥}µ•ÑÉ¥Ì (€€€€€€€€€€€€€€€…¹‘¥‘…Ñ”õ¥Ñ•´¹…¹‘¥‘…Ñ”°(€€€€€€€€€€€€€€€‘…Ñ…Í•Ðõ‘…Ñ…Í•Ð°(€€€€€€€€€€€€€€€µ…ÑÉ¥àõµ…ÑÉ¥à°(€€€€€€€€€€€€€€€Ý¥¹‘½ÝÌõÝ¥¹‘½ÝÌ°(€€€€€€€€€€€€€€€Á¡…Í”ô‰Ù…±¥‘…Ñ¥½¸ˆ°(€€€€€€€€€€€€€€€Í¥µÕ±…Ñ¥½¸õÍ¥µÕ±…Ñ¥½¸°(€€€€€€€€€€€€€€€¥¹¥Ñ¥…±}…Í õ¥¹¥Ñ¥…±}…Í °(€€€€€€€€€€€€¤(€€€€€€€€€€€Ù…±¥‘…Ñ¥½¹}½µÁÕÑ•€¬ô€Ä(€€€€€€€€€€€Ù…±¥‘…Ñ¥½¹}Á…ÍÍ•Ì€ô€ (€€€€€€€€€€€€€€€Ù…±¥‘…Ñ¥½¸¹ÕÍ…‰±•}Ý¥¹‘½Ý}½Õ¹Ð€øôÑ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}Ý¥¹‘½ÝÌ(€€€€€€€€€€€€€€€…¹Ù…±¥‘…Ñ¥½¸¹Ñ½Ñ…±}É•ÑÕÉ¸€ø€À(€€€€€€€€€€€€€€€…¹Ù…±¥‘…Ñ¥½¸¹ÑÉ…‘•}½Õ¹Ð€øôÙ…±¥‘…Ñ¥½¸¹ÕÍ…‰±•}Ý¥¹‘½Ý}½Õ¹Ð(€€€€€€€€€€€€€€€…¹Ù…±¥‘…Ñ¥½¸¹µ…á}‘É…Ý‘½Ý¸€ðô•¥µ…° ˆÀ¸ÌÔˆ¤(€€€€€€€€€€€€€€€…¹Ù…±¥‘…Ñ¥½¸¹É•ÑÕÉ¹}ÍÑ…‰¥±¥Ñä(€€€€€€€€€€€€€€€€øôÑ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}ÍÑ…‰±•}‘¥É•Ñ¥½¹}É…Ñ¥¼(€€€€€€€€€€€€€€€…¹Ù…±¥‘…Ñ¥½¸¹•á•ÍÍ}É•ÑÕÉ¹}½Ù•É}ÁÉ½‘ÕÑ¥½¸€ø€À(€€€€€€€€€€€€¤(€€€€€€€€€€€Ñ•ÍÐ€ô9½¹”(€€€€€€€€€€€¥˜Ù…±¥‘…Ñ¥½¹}Á…ÍÍ•Ì…¹Ñ•ÍÑ}½µÁÕÑ•€ð‰Õ‘•Ð¹µ…á¥µÕµ}Ñ•ÍÑ}…¹‘¥‘…Ñ•Ìè(€€€€€€€€€€€€€€€Ñ•ÍÐ€ô}•½¹½µ¥}µ•ÑÉ¥Ì (€€€€€€€€€€€€€€€€€€€…¹‘¥‘…Ñ”õ¥Ñ•´¹…¹‘¥‘…Ñ”°(€€€€€€€€€€€€€€€€€€€‘…Ñ…Í•Ðõ‘…Ñ…Í•Ð°(€€€€€€€€€€€€€€€€€€€µ…ÑÉ¥àõµ…ÑÉ¥à°(€€€€€€€€€€€€€€€€€€€Ý¥¹‘½ÝÌõÝ¥¹‘½ÝÌ°(€€€€€€€€€€€€€€€€€€€Á¡…Í”ô‰Ñ•ÍÐˆ°(€€€€€€€€€€€€€€€€€€€Í¥µÕ±…Ñ¥½¸õÍ¥µÕ±…Ñ¥½¸°(€€€€€€€€€€€€€€€€€€€¥¹¥Ñ¥…±}…Í õ¥¹¥Ñ¥…±}…Í °(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€Ñ•ÍÑ}½µÁÕÑ•€¬ô€Ä(€€€€€€€€€€€•±¥˜Ù…±¥‘…Ñ¥½¹}Á…ÍÍ•Ìè(€€€€€€€€€€€€€€€‰Õ‘•Ñ}•á¡…ÕÍÑ•€ôQÉÕ”(€€€€€€€€€€€Ñ•ÍÑ}½¹™¥Éµ•€ô‰½½° (€€€€€€€€€€€€€€€Ñ•ÍÐ(€€€€€€€€€€€€€€€…¹Ñ•ÍÐ¹ÕÍ…‰±•}Ý¥¹‘½Ý}½Õ¹Ð€øôÑ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}Ý¥¹‘½ÝÌ(€€€€€€€€€€€€€€€…¹Ñ•ÍÐ¹Ñ½Ñ…±}É•ÑÕÉ¸€ø€À(€€€€€€€€€€€€€€€…¹Ñ•ÍÐ¹ÑÉ…‘•}½Õ¹Ð€øôÑ•ÍÐ¹ÕÍ…‰±•}Ý¥¹‘½Ý}½Õ¹Ð(€€€€€€€€€€€€€€€…¹Ñ•ÍÐ¹µ…á}‘É…Ý‘½Ý¸€ðô•¥µ…° ˆÀ¸ÌÔˆ¤(€€€€€€€€€€€€€€€…¹Ñ•ÍÐ¹É•ÑÕÉ¹}ÍÑ…‰¥±¥Ñä€øôÑ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}ÍÑ…‰±•}‘¥É•Ñ¥½¹}É…Ñ¥¼(€€€€€€€€€€€€€€€…¹Ñ•ÍÐ¹•á•ÍÍ}É•ÑÕÉ¹}½Ù•É}ÁÉ½‘ÕÑ¥½¸€ø€À(€€€€€€€€€€€€¤(€€€€€€€€€€€ÍÑ…ÑÕÌ€ô€ (€€€€€€€€€€€€€€€½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹=9%I5}9%Q(€€€€€€€€€€€€€€€¥˜Ñ•ÍÑ}½¹™¥Éµ•(€€€€€€€€€€€€€€€•±Í”½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹I)Q}QMP(€€€€€€€€€€€€€€€¥˜Ñ•ÍÐ¥Ì¹½Ð9½¹”(€€€€€€€€€€€€€€€•±Í”½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹AMM}Y1%Q%=8(€€€€€€€€€€€€€€€¥˜Ù…±¥‘…Ñ¥½¹}Á…ÍÍ•Ì(€€€€€€€€€€€€€€€•±Í”½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹I)Q}Y1%Q%=8(€€€€€€€€€€€€¤(€€€€€€€€€€€™…¥±•€ô€ (€€€€€€€€€€€€€€€€ ¤(€€€€€€€€€€€€€€€¥˜Ñ•ÍÑ}½¹™¥Éµ•½È€¡Ù…±¥‘…Ñ¥½¹}Á…ÍÍ•Ì…¹Ñ•ÍÐ¥Ì9½¹”¤(€€€€€€€€€€€€€€€•±Í”€ (€€€€€€€€€€€€€€€€€€€€ ‰Ñ•ÍÑ}•½¹½µ¥}ÅÕ…±¥™¥…Ñ¥½¸ˆ°¤(€€€€€€€€€€€€€€€€€€€¥˜Ñ•ÍÐ(€€€€€€€€€€€€€€€€€€€•±Í”€ ‰Ù…±¥‘…Ñ¥½¹}•½¹½µ¥}ÅÕ…±¥™¥…Ñ¥½¸ˆ°¤(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€¤(€€€€€€€€€€€™¥¹…±}µ•ÑÉ¥Ì€ôÑ•ÍÐ½ÈÙ…±¥‘…Ñ¥½¸(€€€€€€€€€€€™¥¹…±¥é•¹…ÁÁ•¹ (€€€€€€€€€€€€€€€½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•I•ÍÕ±Ð (€€€€€€€€€€€€€€€€€€€¥Ñ•´¹…¹‘¥‘…Ñ”°(€€€€€€€€€€€€€€€€€€€¥Ñ•´¹Á…É•¹Ñ}™¥¹•ÉÁÉ¥¹Ð°(€€€€€€€€€€€€€€€€€€€‘•ÁÑ °(€€€€€€€€€€€€€€€€€€€ÍÑ…ÑÕÌ°(€€€€€€€€€€€€€€€€€€€}É…¹­¥¹œ¡¥Ñ•´¹…¹‘¥‘…Ñ”°™¥¹…±}µ•ÑÉ¥Ì¤°(€€€€€€€€€€€€€€€€€€€¥Ñ•´¹ÑÉ…¥¹¥¹}µ•ÑÉ¥Ì°(€€€€€€€€€€€€€€€€€€€Ù…±¥‘…Ñ¥½¸°(€€€€€€€€€€€€€€€€€€€Ñ•ÍÐ°(€€€€€€€€€€€€€€€€€€€¥Ñ•´¹Á…ÍÍ•‘}É•ÅÕ¥É•µ•¹ÑÌ(€€€€€€€€€€€€€€€€€€€€¬€ (€€€€€€€€€€€€€€€€€€€€€€€€ ‰Ù…±¥‘…Ñ¥½¹}•½¹½µ¥}ÅÕ…±¥™¥…Ñ¥½¸ˆ°¤(€€€€€€€€€€€€€€€€€€€€€€€¥˜Ù…±¥‘…Ñ¥½¹}Á…ÍÍ•Ì(€€€€€€€€€€€€€€€€€€€€€€€•±Í”€ ¤(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€€¬€  ‰Ñ•ÍÑ}•½¹½µ¥}ÅÕ…±¥™¥…Ñ¥½¸ˆ°¤¥˜Ñ•ÍÑ}½¹™¥Éµ••±Í”€ ¤¤°(€€€€€€€€€€€€€€€€€€€™…¥±•°(€€€€€€€€€€€€€€€€€€€™…¥±•‘lÁt¥˜™…¥±••±Í”9½¹”°(€€€€€€€€€€€€€€€€€€€•¥µ…° À¤(€€€€€€€€€€€€€€€€€€€¥˜Ñ•ÍÑ}½¹™¥Éµ•½È€¡Ù…±¥‘…Ñ¥½¹}Á…ÍÍ•Ì…¹Ñ•ÍÐ¥Ì9½¹”¤(€€€€€€€€€€€€€€€€€€€•±Í”•¥µ…° Ä¤°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€¤(€€€€€€€Á•ÉÍ¥ÍÑ•€ôÑÕÁ±” (€€€€€€€€€€€™¥¹…±¥é•(€€€€€€€€€€€€¬±¥ÍÐ¡Á…É•¹ÑÍmÙ…±¥‘…Ñ¥½¹}±¥µ¥Ðét¤(€€€€€€€€€€€€¬m¥Ñ•´™½È¥Ñ•´¥¸‘•ÁÑ¡}É•ÍÕ±ÑÌ¥˜¥Ñ•´¹½Ð¥¸Á…É•¹ÑÍt(€€€€€€€€¤(€€€€€€€É•µ…¥¹¥¹}Á•ÉÍ¥ÍÑ•¹”€ô‰Õ‘•Ð¹µ…á¥µÕµ}Á•ÉÍ¥ÍÑ•‘}…¹‘¥‘…Ñ•Ì€´±•¸¡É•ÍÕ±ÑÌ¤(€€€€€€€¥˜É•µ…¥¹¥¹}Á•ÉÍ¥ÍÑ•¹”€ø€Àè(€€€€€€€€€€€É•ÍÕ±ÑÌ¹•áÑ•¹¡Á•ÉÍ¥ÍÑ•‘léÉ•µ…¥¹¥¹}Á•ÉÍ¥ÍÑ•¹•t¤(€€€€€€€¥˜±•¸¡Á•ÉÍ¥ÍÑ•¤€øµ…à À°É•µ…¥¹¥¹}Á•ÉÍ¥ÍÑ•¹”¤è(€€€€€€€€€€€‰Õ‘•Ñ}•á¡…ÕÍÑ•€ôQÉÕ”(€€€€€€€™Õ¹¹•±Ì¹…ÁÁ•¹ (€€€€€€€€€€€½µÁ½Í¥Ñ••ÁÑ¡Õ¹¹•° (€€€€€€€€€€€€€€€‘•ÁÑ °(€€€€€€€€€€€€€€€±•¸¡Õ¹¥ÅÕ•}ÁÉ½Á½Í…±Ì¤°(€€€€€€€€€€€€€€€½¹™¥ÕÉ…Ñ¥½¹Ì°(€€€€€€€€€€€€€€€‘ÕÁ±¥…Ñ•Ì°(€€€€€€€€€€€€€€€±•¸¡¡•…Á}É…¹­•¤°(€€€€€€€€€€€€€€€€À°(€€€€€€€€€€€€€€€Í¥µÕ±…Ñ¥½¹}Í±½ÑÌ°(€€€€€€€€€€€€€€€€À°(€€€€€€€€€€€€€€€±•¸¡Á…É•¹ÑÌ¤°(€€€€€€€€€€€€€€€Ù…±¥‘…Ñ¥½¹}±¥µ¥Ð°(€€€€€€€€€€€€€€€ÍÕ´ (€€€€€€€€€€€€€€€€€€€€Ä(€€€€€€€€€€€€€€€€€€€™½È¥Ñ•´¥¸™¥¹…±¥é•(€€€€€€€€€€€€€€€€€€€¥˜¥Ñ•´¹ÍÑ…ÑÕÌ(€€€€€€€€€€€€€€€€€€€¥¸ì(€€€€€€€€€€€€€€€€€€€€€€€½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹AMM}Y1%Q%=8°(€€€€€€€€€€€€€€€€€€€€€€€½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹I)Q}QMP°(€€€€€€€€€€€€€€€€€€€€€€€½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹=9%I5}9%Q°(€€€€€€€€€€€€€€€€€€€ô(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€ÍÕ´ Ä™½È¥Ñ•´¥¸™¥¹…±¥é•¥˜¥Ñ•´¹Ñ•ÍÑ}µ•ÑÉ¥Ì¥Ì¹½Ð9½¹”¤°(€€€€€€€€€€€€¤(€€€€€€€€¤(€€€€€€€¥˜¹½ÐÁ…É•¹ÑÌ½ÈÑÉ…¥¹¥¹}½µÁÕÑ•€øô‰Õ‘•Ð¹µ…á¥µÕµ}ÑÉ…¥¹¥¹}Í¥µÕ±…Ñ¥½¹Ìè(€€€€€€€€€€€¥˜ÑÉ…¥¹¥¹}½µÁÕÑ•€øô‰Õ‘•Ð¹µ…á¥µÕµ}ÑÉ…¥¹¥¹}Í¥µÕ±…Ñ¥½¹Ìè(€€€€€€€€€€€€€€€‰Õ‘•Ñ}•á¡…ÕÍÑ•€ôQÉÕ”(€€€€€€€€€€€‰É•…¬(€€€½¹™¥Éµ•‘}…¹‘¥‘…Ñ•Ì€ôl(€€€€€€€¥Ñ•´(€€€€€€€™½È¥Ñ•´¥¸É•ÍÕ±ÑÌ(€€€€€€€¥˜¥Ñ•´¹ÍÑ…ÑÕÌ¥Ì½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹=9%I5}9%Q(€€€t(€€€Á…ÍÍ•‘}Ù…±¥‘…Ñ¥½¸€ôl(€€€€€€€¥Ñ•´(€€€€€€€™½È¥Ñ•´¥¸É•ÍÕ±ÑÌ(€€€€€€€¥˜¥Ñ•´¹ÍÑ…ÑÕÌ¥Ì½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹AMM}Y1%Q%=8(€€€t(€€€É•©•Ñ•‘}Ñ•ÍÐ€ôl(€€€€€€€¥Ñ•´(€€€€€€€™½È¥Ñ•´¥¸É•ÍÕ±ÑÌ(€€€€€€€¥˜¥Ñ•´¹ÍÑ…ÑÕÌ¥Ì½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹I)Q}QMP(€€€t(€€€É•©•Ñ•‘}Ù…±¥‘…Ñ¥½¸€ôl(€€€€€€€¥Ñ•´(€€€€€€€™½È¥Ñ•´¥¸É•ÍÕ±ÑÌ(€€€€€€€¥˜¥Ñ•´¹ÍÑ…ÑÕÌ¥Ì½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹I)Q}Y1%Q%=8(€€€t(€€€½¹±ÕÍ¥½¸€ô€ (€€€€€€€½µÁ½Í¥Ñ•M•…É¡½¹±ÕÍ¥½¸¹=9%I5}=5A=M%Q}9%QL(€€€€€€€¥˜½¹™¥Éµ•‘}…¹‘¥‘…Ñ•Ì(€€€€€€€•±Í”½µÁ½Í¥Ñ•M•…É¡½¹±ÕÍ¥½¸¹9%QM}AMM}Y1%Q%=8(€€€€€€€¥˜Á…ÍÍ•‘}Ù…±¥‘…Ñ¥½¸(€€€€€€€•±Í”½µÁ½Í¥Ñ•M•…É¡½¹±ÕÍ¥½¸¹9%QM}I)Q}QMP(€€€€€€€¥˜É•©•Ñ•‘}Ñ•ÍÐ(€€€€€€€•±Í”½µÁ½Í¥Ñ•M•…É¡½¹±ÕÍ¥½¸¹9%QM}I)Q}Y1%Q%=8(€€€€€€€¥˜É•©•Ñ•‘}Ù…±¥‘…Ñ¥½¸(€€€€€€€•±Í”½µÁ½Í¥Ñ•M•…É¡½¹±ÕÍ¥½¸¹9=}QI%9%9}M%90(€€€€€€€¥˜É•ÍÕ±ÑÌ(€€€€€€€•±Í”½µÁ½Í¥Ñ•M•…É¡½¹±ÕÍ¥½¸¹MI!}	UQ}a!UMQ(€€€€€€€¥˜‰Õ‘•Ñ}•á¡…ÕÍÑ•(€€€€€€€•±Í”½µÁ½Í¥Ñ•M•…É¡½¹±ÕÍ¥½¸¹9=}QI%9%9}M%90(€€€€¤(€€€ÁÉ¥½É¥Ñä€ôì(€€€€€€€½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹=9%I5}9%Qè€Ô°(€€€€€€€½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹AMM}Y1%Q%=8è€Ð°(€€€€€€€½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹I)Q}QMPè€Ì°(€€€€€€€½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹IQ%9}QI%9%9è€È°(€€€€€€€½µÁ½Í¥Ñ•…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹I)Q}Y1%Q%=8è€Ä°(€€€ô(€€€‰•ÍÐ€ôµ…à (€€€€€€€É•ÍÕ±ÑÌ°(€€€€€€€­•äõ±…µ‰‘„¥Ñ•´è€ (€€€€€€€€€€€ÁÉ¥½É¥Ñåm¥Ñ•´¹ÍÑ…ÑÕÍt°(€€€€€€€€€€€¥Ñ•´¹É…¹­¥¹œ¹Ñ½Ñ…°°(€€€€€€€€€€€¥Ñ•´¹…¹‘¥‘…Ñ”¹™¥¹•ÉÁÉ¥¹Ð°(€€€€€€€€¤°(€€€€€€€‘•™…Õ±Ðõ9½¹”°(€€€€¤(€€€Ñ½Ñ…±}•Ù…±Õ…Ñ¥½¹Ì€ô€ (€€€€€€€¡•…Á}½µÁÕÑ•€¬ÑÉ…¥¹¥¹}½µÁÕÑ•€¬Ù…±¥‘…Ñ¥½¹}½µÁÕÑ•€¬Ñ•ÍÑ}½µÁÕÑ•(€€€€¤(€€€É•ÑÕÉ¸½µÁ½Í¥Ñ•M•…É¡I•ÍÕ±Ð (€€€€€€€½¹±ÕÍ¥½¸õ½¹±ÕÍ¥½¸°(€€€€€€€µ…á¥µÕµ}‘•ÁÑ¡}É•…¡•õ™Õ¹¹•±Íl´Åt¹‘•ÁÑ ¥˜™Õ¹¹•±Ì•±Í”€À°(€€€€€€€™•…ÑÕÉ•}Á½½°õÁ½½°°(€€€€€€€•á±Õ‘•‘}™•…ÑÕÉ•Ìõ•á±Õ‘•°(€€€€€€€™Õ¹¹•±}‰å}‘•ÁÑ õÑÕÁ±”¡™Õ¹¹•±Ì¤°(€€€€€€€…¹‘¥‘…Ñ•ÌõÑÕÁ±”¡É•ÍÕ±ÑÌ¤°(€€€€€€€Ñ½Ñ…±}•Ù…±Õ…Ñ¥½¹ÌõÑ½Ñ…±}•Ù…±Õ…Ñ¥½¹Ì°(€€€€€€€½µÁÕÑ•õÑ½Ñ…±}•Ù…±Õ…Ñ¥½¹Ì°(€€€€€€€É•ÕÍ•ôÀ°(€€€€€€€‰Õ‘•Ñ}•á¡…ÕÍÑ•õ‰Õ‘•Ñ}•á¡…ÕÍÑ•°(€€€€€€€¡•…Á}•Ù…±Õ…Ñ¥½¹Í}½µÁÕÑ•õ¡•…Á}½µÁÕÑ•°(€€€€€€€¡•…Á}•Ù…±Õ…Ñ¥½¹Í}É•ÕÍ•ôÀ°(€€€€€€€ÑÉ…¥¹¥¹}Í¥µÕ±…Ñ¥½¹Í}½µÁÕÑ•õÑÉ…¥¹¥¹}½µÁÕÑ•°(€€€€€€€ÑÉ…¥¹¥¹}Í¥µÕ±…Ñ¥½¹Í}É•ÕÍ•ôÀ°(€€€€€€€Ù…±¥‘…Ñ¥½¹}Í¥µÕ±…Ñ¥½¹Í}½µÁÕÑ•õÙ…±¥‘…Ñ¥½¹}½µÁÕÑ•°(€€€€€€€Ù…±¥‘…Ñ¥½¹}Í¥µÕ±…Ñ¥½¹Í}É•ÕÍ•ôÀ°(€€€€€€€Ñ•ÍÑ}Í¥µÕ±…Ñ¥½¹Í}½µÁÕÑ•õÑ•ÍÑ}½µÁÕÑ•°(€€€€€€€Ñ•ÍÑ}Í¥µÕ±…Ñ¥½¹Í}É•ÕÍ•ôÀ°(€€€€€€€‰•ÍÑ}‘•ÁÑ õ‰•ÍÐ¹‘•ÁÑ ¥˜‰•ÍÐ•±Í”9½¹”°(€€€€€€€‰•ÍÑ}…¹‘¥‘…Ñ•}™¥¹•ÉÁÉ¥¹Ðõ‰•ÍÐ¹…¹‘¥‘…Ñ”¹™¥¹•ÉÁÉ¥¹Ð¥˜‰•ÍÐ•±Í”9½¹”°(€€€€€€€‘½µ¥¹…¹Ñ}É•©•Ñ¥½¹}É•…Í½¸ô (€€€€€€€€€€€9½¹”¥˜½¹™¥Éµ•‘}…¹‘¥‘…Ñ•Ì•±Í”€‰•½¹½µ¥}•Ù¥‘•¹•}¹½Ñ}½¹™¥Éµ•ˆ(€€€€€€€€¤°(€€€€¤(