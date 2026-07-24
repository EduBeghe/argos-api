import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from enum import StrEnum
from itertools import combinations
from statistics import median

from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.walk_forward import WalkForwardWindow
from app.paper.config import SimulationConfiguration

COMPOSITE_FEATURE_POOL_VERSION = 1
COMPOSITE_SEARCH_ALGORITHM_VERSION = 1
COMPOSITE_RANKING_VERSION = 1
COMPOSITE_NORMALIZATION_VERSION = 1
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
    REJECTED_TEST = "rejected_test"
    CONFIRMED_CANDIDATE = "confirmed_candidate"


class CompositeSearchConclusion(StrEnum):
    NO_EVALUABLE_FEATURES = "no_evaluable_features"
    SEARCH_BUDGET_EXHAUSTED = "search_budget_exhausted"
    NO_TRAINING_SIGNAL = "no_training_signal"
    TRAINING_SIGNAL_NOT_STABLE = "training_signal_not_stable"
    CANDIDATES_REJECTED_VALIDATION = "candidates_rejected_validation"
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
    best_depth: int | None
    best_candidate_fingerprint: str | None
    dominant_rejection_reason: str | None


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


def _metrics(
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
        return CompositeMetrics(
            0, Decimal(0), Decimal(0), Decimal(0), Decimal(0), Decimal(0), Decimal(0)
        )
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


def _window_metrics(
    candidate: CompositeCandidate,
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    phase: str,
) -> CompositeMetrics:
    usable = tuple(
        value
        for window in windows
        if (
            value := _metrics(
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
        return _metrics(candidate, ())
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


def _ranking(
    candidate: CompositeCandidate, metrics: CompositeMetrics
) -> CompositeRankingBreakdown:
    complexity = Decimal(len(candidate.features) - 2) * Decimal("0.01")
    concentration = max(abs(item.weight) for item in candidate.weights)
    concentration_penalty = max(Decimal(0), concentration - Decimal("0.60"))
    stability = Decimal(1) - min(Decimal(1), metrics.dispersion)
    total = (
        metrics.expected_net_movement
        + metrics.directional_hit_rate
        + stability
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
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    simulation: SimulationConfiguration,
    budget: CompositeSearchBudget = DEFAULT_COMPOSITE_SEARCH_BUDGET,
) -> CompositeSearchResult:
    excluded = validate_feature_pool(matrix)
    pool = tuple(name for name in COMPOSITE_FEATURE_POOL if name not in excluded)
    if len(pool) < 2 or not windows:
        return CompositeSearchResult(
            CompositeSearchConclusion.NO_EVALUABLE_FEATURES,
            0,
            pool,
            excluded,
            (),
            (),
            0,
            0,
            0,
            None,
            None,
            "insufficient_evidence",
        )
    training_rows = _phase_rows(matrix, windows, "training")
    results: list[CompositeCandidateResult] = []
    funnels: list[CompositeDepthFunnel] = []
    seen: set[str] = set()
    computed = 0
    best_parent: dict[tuple[str, ...], str] = {}
    cadences = (
        timedelta(minutes=30),
        timedelta(hours=4),
        timedelta(hours=8),
        timedelta(hours=12),
    )
    horizons = (timedelta(minutes=30), timedelta(hours=4), timedelta(hours=12))
    for depth in range(2, min(7, len(pool)) + 1):
        generated = list(combinations(pool, depth))
        depth_results: list[CompositeCandidateResult] = []
        configurations = 0
        duplicates = 0
        for subset in generated:
            if computed >= budget.maximum_cheap_evaluations:
                break
            evidence = []
            for name in subset:
                single_values = tuple(feature_value(row, name) for row in training_rows)
                direction = (
                    Decimal(1)
                    if single_values[-1] >= Decimal(median(single_values))
                    else Decimal(-1)
                )
                evidence.append((name, direction))
            variants = (
                tuple(evidence),
                tuple(
                    (name, sign * Decimal(index + 1))
                    for index, (name, sign) in enumerate(evidence)
                ),
                tuple(
                    (name, sign * Decimal(len(evidence) - index))
                    for index, (name, sign) in enumerate(evidence)
                ),
            )[: budget.maximum_variants_per_subset]
            for variant_index, weights in enumerate(variants):
                configurations += 1
                candidate = build_candidate(
                    weights=weights,
                    training_rows=training_rows,
                    simulation=simulation,
                    cadence=cadences[(depth + variant_index) % len(cadences)],
                    horizon=horizons[(depth + variant_index) % len(horizons)],
                )
                if candidate.fingerprint in seen:
                    duplicates += 1
                    continue
                seen.add(candidate.fingerprint)
                computed += 1
                training = _window_metrics(candidate, matrix, windows, "training")
                ranking = _ranking(candidate, training)
                parent = best_parent.get(subset[:-1])
                depth_results.append(
                    CompositeCandidateResult(
                        candidate,
                        parent,
                        depth,
                        CompositeCandidateStatus.RETAINED_TRAINING,
                        ranking,
                        training,
                        None,
                        None,
                        ("minimum_observations",)
                        if training.observations >= 30
                        else (),
                        ()
                        if training.observations >= 30
                        else ("minimum_observations",),
                        None if training.observations >= 30 else "minimum_observations",
                        Decimal(0) if training.observations >= 30 else Decimal(1),
                    )
                )
        depth_results.sort(
            key=lambda item: (-item.ranking.total, item.candidate.fingerprint)
        )
        retained = depth_results[: budget.beam_width_per_depth]
        validation_limit = min(len(retained), budget.maximum_validation_per_depth)
        validated: list[CompositeCandidateResult] = []
        for item in retained[:validation_limit]:
            validation = _window_metrics(item.candidate, matrix, windows, "validation")
            validation_passes = (
                validation.observations >= 30 and validation.expected_net_movement > 0
            )
            test = (
                _window_metrics(item.candidate, matrix, windows, "test")
                if validation_passes
                and sum(1 for value in results if value.test_metrics is not None)
                < budget.maximum_test_candidates
                else None
            )
            test_confirmed = (
                test is not None
                and test.observations >= 30
                and test.expected_net_movement > 0
            )
            status = (
                CompositeCandidateStatus.CONFIRMED_CANDIDATE
                if test_confirmed
                else CompositeCandidateStatus.REJECTED_TEST
                if test is not None
                else CompositeCandidateStatus.REJECTED_VALIDATION
            )
            failed = (
                ()
                if test_confirmed
                else (
                    ("positive_test_net_movement",)
                    if test is not None
                    else ("positive_validation_net_movement",)
                )
            )
            validated.append(
                CompositeCandidateResult(
                    item.candidate,
                    item.parent_fingerprint,
                    depth,
                    status,
                    item.ranking,
                    item.training_metrics,
                    validation,
                    test,
                    item.passed_requirements
                    + (("validation",) if validation_passes else ()),
                    failed,
                    failed[0] if failed else None,
                    Decimal(0) if test_confirmed else Decimal(1),
                )
            )
        persisted = tuple(validated + retained[validation_limit:])
        results.extend(persisted[: budget.maximum_persisted_candidates - len(results)])
        for item in retained:
            best_parent[item.candidate.features] = item.candidate.fingerprint
        funnels.append(
            CompositeDepthFunnel(
                depth,
                len(generated),
                configurations,
                duplicates,
                len(depth_results),
                0,
                len(retained),
                0,
                len(retained),
                validation_limit,
                sum(
                    1
                    for item in validated
                    if item.validation_metrics and not item.failed_requirements
                ),
                sum(1 for item in validated if item.test_metrics is not None),
            )
        )
    confirmed_candidates = [
        item
        for item in results
        if item.status is CompositeCandidateStatus.CONFIRMED_CANDIDATE
    ]
    validation_passed = [item for item in results if item.test_metrics is not None]
    conclusion = (
        CompositeSearchConclusion.CONFIRMED_COMPOSITE_CANDIDATES
        if confirmed_candidates
        else CompositeSearchConclusion.CANDIDATES_REJECTED_TEST
        if validation_passed
        else CompositeSearchConclusion.CANDIDATES_REJECTED_VALIDATION
        if results
        else CompositeSearchConclusion.NO_TRAINING_SIGNAL
    )
    best = max(
        results,
        key=lambda item: (item.ranking.total, item.candidate.fingerprint),
        default=None,
    )
    return CompositeSearchResult(
        conclusion,
        funnels[-1].depth if funnels else 0,
        pool,
        excluded,
        tuple(funnels),
        tuple(results),
        computed,
        computed,
        0,
        best.depth if best else None,
        best.candidate.fingerprint if best else None,
        None if confirmed_candidates else "economic_evidence_not_confirmed",
    )
