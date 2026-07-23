"""Bounded regime-conditioned Predictive Edge Discovery v4.1.

The module is deliberately research-only. Derived predictors and regimes exist only in
memory, are learned from training, and are frozen before validation and isolated test.
"""

import hashlib
import json
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from math import sqrt

from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.predictive_edge_v4 import DISCOVERY_CADENCES, DISCOVERY_HORIZONS
from app.optimization.predictive_edges import (
    Condition,
    DirectionMapping,
    EvidenceStatus,
    StrategyHypothesis,
    SuggestedAction,
)
from app.optimization.walk_forward import WalkForwardWindow
from app.paper.config import SimulationConfiguration

PREDICTIVE_EDGE_V41_CONTRACT_VERSION = 42
DERIVED_PREDICTOR_VERSION = 2
DIRECTIONAL_RELATIONSHIP_LIMIT = 24
CONTEXT_RELATIONSHIP_LIMIT = 8
INTERACTION_GENERATION_LIMIT = 192
INTERACTION_RETENTION_LIMIT = 60
VALIDATION_LIMIT = 40
ISOLATED_TEST_LIMIT = 12
LOOKBACKS = (12, 48, 288)
MINIMUM_OBSERVATIONS = 8
MINIMUM_HIT_IMPROVEMENT = Decimal("0.01")
ECONOMIC_MARGIN = Decimal("0.0001")
MINIMUM_CONFIRMATION_WINDOWS = 3
DIRECTIONAL_BIN_COUNT = 3


class PredictorRole(StrEnum):
    DIRECTIONAL = "directional"
    CONTEXT = "context"


class ContextRegime(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    LOWER = "lower"
    MIDDLE = "middle"
    UPPER = "upper"


class V41Conclusion(StrEnum):
    DIRECTIONAL_SIGNAL_NOT_IMPROVED_BY_CONTEXT = (
        "directional_signal_not_improved_by_context"
    )
    CONTEXT_IMPROVES_SIGNAL_BUT_NOT_ECONOMIC = (
        "context_improves_signal_but_not_economic"
    )
    ECONOMIC_INTERACTION_BUT_UNSTABLE = "economic_interaction_but_unstable"
    STABLE_INTERACTION_BELOW_BENCHMARK = "stable_interaction_below_benchmark"
    REGIME_CONDITIONED_EDGE_CONFIRMED = "regime_conditioned_edge_confirmed"
    NO_REGIME_CONDITIONED_EDGE = "no_regime_conditioned_edge"


@dataclass(frozen=True, slots=True)
class DerivedPredictorDefinition:
    name: str
    version: int
    role: PredictorRole
    lookback: int
    formula: str
    source: str = "historical_feature_matrix_in_memory"


@dataclass(frozen=True, slots=True)
class DerivedFeatureRow:
    candle_open_time: datetime
    values: dict[str, Decimal]


@dataclass(frozen=True, slots=True)
class FrozenRegime:
    context_predictor: str
    context_version: int
    regime: ContextRegime
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    training_observations: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class DirectionalScreen:
    predictor: str
    version: int
    cadence: timedelta
    horizon: timedelta
    score: Decimal
    fingerprint: str


@dataclass(frozen=True, slots=True)
class PhaseEvidence:
    observations: int
    gross_movement: Decimal | None
    aligned_gross_movement: Decimal | None
    net_movement: Decimal | None
    directional_hit_rate: Decimal | None
    benchmark: Decimal | None
    benchmark_improvement: Decimal | None
    standard_error: Decimal | None
    stable: bool
    economic: bool
    passes: bool


@dataclass(frozen=True, slots=True)
class RegimeConditionedRelationship:
    directional_predictor: str
    directional_version: int
    context_predictor: str
    context_version: int
    regime: ContextRegime
    cadence: timedelta
    horizon: timedelta
    frozen_regime: FrozenRegime
    selected_bin: int
    directional_lower_bound: Decimal | None
    directional_upper_bound: Decimal | None
    direction_mapping: DirectionMapping
    action: SuggestedAction
    profile: str
    training: PhaseEvidence
    directional_baseline: PhaseEvidence
    context_improvement: Decimal
    fingerprint: str


@dataclass(frozen=True, slots=True)
class V41Candidate:
    relationship: RegimeConditionedRelationship
    validation: PhaseEvidence
    test: PhaseEvidence | None
    selected_for_test: bool
    confirmed: bool
    rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class V41StageCounters:
    generated: int
    computed: int
    reused: int
    retained: int


@dataclass(frozen=True, slots=True)
class V41PrimitiveCounters:
    price_index_builds: int
    sampled_pair_requests: int
    sampled_pair_builds: int
    sampled_pair_reuses: int
    slice_fingerprint_requests: int
    slice_fingerprint_builds: int
    slice_fingerprint_reuses: int


@dataclass(frozen=True, slots=True)
class PredictiveEdgeV41Result:
    directional_relationships: tuple[DirectionalScreen, ...]
    context_relationships: tuple[str, ...]
    interactions: tuple[RegimeConditionedRelationship, ...]
    candidates: tuple[V41Candidate, ...]
    actionable_near_misses: tuple[V41Candidate, ...]
    diagnostic_near_misses: tuple[str, ...]
    confirmed_candidates: tuple[V41Candidate, ...]
    hypotheses: tuple[StrategyHypothesis, ...]
    conclusion: V41Conclusion
    stage_a1: V41StageCounters
    stage_a2: V41StageCounters
    stage_a3: V41StageCounters
    stage_b: V41StageCounters
    stage_c: V41StageCounters
    cache_writes: dict[str, PhaseEvidence]
    primitive_counters: V41PrimitiveCounters = V41PrimitiveCounters(0, 0, 0, 0, 0, 0, 0)


class _ExecutionIndex:
    """Bounded per-execution indexes for repeated deterministic primitives."""

    def __init__(
        self,
        matrix: HistoricalFeatureMatrix,
        values: dict[datetime, dict[str, Decimal]],
    ) -> None:
        self.matrix = matrix
        self.values = values
        self.prices = {row.candle_open_time: row.close_price for row in matrix.rows}
        self.sampled: dict[
            tuple[object, ...], tuple[tuple[datetime, Decimal, Decimal], ...]
        ] = {}
        self.slices: dict[tuple[datetime, datetime], str] = {}
        self.sampled_requests = self.sampled_builds = 0
        self.slice_requests = self.slice_builds = 0

    def counters(self) -> V41PrimitiveCounters:
        return V41PrimitiveCounters(
            1,
            self.sampled_requests,
            self.sampled_builds,
            self.sampled_requests - self.sampled_builds,
            self.slice_requests,
            self.slice_builds,
            self.slice_requests - self.slice_builds,
        )


_EXECUTION_INDEX: ContextVar[_ExecutionIndex | None] = ContextVar(
    "predictive_edge_v41_execution_index", default=None
)


DERIVED_PREDICTORS = (
    DerivedPredictorDefinition(
        "ema_20_slope_12",
        1,
        PredictorRole.DIRECTIONAL,
        12,
        "ema_20[t] / ema_20[t-12] - 1",
    ),
    DerivedPredictorDefinition(
        "ema_50_slope_48",
        1,
        PredictorRole.DIRECTIONAL,
        48,
        "ema_50[t] / ema_50[t-48] - 1",
    ),
    DerivedPredictorDefinition(
        "ema_spread_change_12",
        1,
        PredictorRole.DIRECTIONAL,
        12,
        "(ema_20 / ema_50 - 1)[t] - (ema_20 / ema_50 - 1)[t-12]",
    ),
    DerivedPredictorDefinition(
        "return_acceleration_12",
        1,
        PredictorRole.DIRECTIONAL,
        12,
        "return_1[t] - mean(return_1[t-12:t])",
    ),
    DerivedPredictorDefinition(
        "rolling_breakout_position_48",
        1,
        PredictorRole.DIRECTIONAL,
        48,
        "(close - rolling_min_48) / (rolling_max_48 - rolling_min_48)",
    ),
    DerivedPredictorDefinition(
        "rolling_breakout_position_288",
        1,
        PredictorRole.CONTEXT,
        288,
        "(close - rolling_min_288) / (rolling_max_288 - rolling_min_288)",
    ),
)

DIRECTIONAL_BASE = (
    "ema_20_relative",
    "ema_50_relative",
    "rsi_14",
    "return_1",
    "return_12",
    "trend_component",
    "rsi_contribution",
    "return_1_contribution",
    "return_12_contribution",
    "directional_score",
    "overall_score",
)
CONTEXT_BASE = (
    "atr_14_relative",
    "relative_volume_20",
    "volume_confirmation",
    "trend_regime",
    "rsi_regime",
    "breakout_regime",
)


def build_derived_feature_rows(
    matrix: HistoricalFeatureMatrix,
) -> tuple[DerivedFeatureRow, ...]:
    """Calculate versioned predictors without future data or persistence."""

    result: list[DerivedFeatureRow] = []
    for index, row in enumerate(matrix.rows):
        values = _base_values(row)
        for definition in DERIVED_PREDICTORS:
            if index < definition.lookback:
                continue
            previous = matrix.rows[index - definition.lookback]
            history = matrix.rows[index - definition.lookback : index]
            if definition.name == "ema_20_slope_12":
                values[definition.name] = _ratio(row.ema_20, previous.ema_20)
            elif definition.name == "ema_50_slope_48":
                values[definition.name] = _ratio(row.ema_50, previous.ema_50)
            elif definition.name == "ema_spread_change_12":
                values[definition.name] = _ratio(row.ema_20, row.ema_50) - _ratio(
                    previous.ema_20, previous.ema_50
                )
            elif definition.name == "return_acceleration_12":
                values[definition.name] = row.return_1 - sum(
                    (item.return_1 for item in history), Decimal(0)
                ) / Decimal(len(history))
            elif definition.name.startswith("rolling_breakout_position"):
                closes = tuple(item.close_price for item in history)
                low, high = min(closes), max(closes)
                values[definition.name] = (
                    Decimal("0.5")
                    if high == low
                    else (row.close_price - low) / (high - low)
                )
        result.append(DerivedFeatureRow(row.candle_open_time, values))
    return tuple(result)


def enrich_feature_matrix(
    matrix: HistoricalFeatureMatrix,
) -> HistoricalFeatureMatrix:
    """Attach research-only derived values for the edge-guided simulation path."""

    derived = {
        item.candle_open_time: item.values
        for item in build_derived_feature_rows(matrix)
    }
    return HistoricalFeatureMatrix(
        matrix.dataset_fingerprint,
        tuple(
            HistoricalFeatureRow(
                row.candle_open_time,
                row.close_price,
                row.ema_20,
                row.ema_50,
                row.rsi_14,
                row.atr_14,
                row.return_1,
                row.return_12,
                row.relative_volume_20,
                derived.get(row.candle_open_time, {}),
            )
            for row in matrix.rows
        ),
    )


def analyze_predictive_edges_v41(
    *,
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    simulation: SimulationConfiguration,
    stage_cache: dict[str, PhaseEvidence] | None = None,
    force_recompute: bool = False,
) -> PredictiveEdgeV41Result:
    """Run A1/A2/A3/B/C with shared bounded deterministic indexes."""

    rows = build_derived_feature_rows(matrix)
    values_by_time = {row.candle_open_time: row.values for row in rows}
    index = _ExecutionIndex(matrix, values_by_time)
    token = _EXECUTION_INDEX.set(index)
    try:
        result = _analyze_predictive_edges_v41(
            matrix=matrix,
            windows=windows,
            simulation=simulation,
            stage_cache=stage_cache,
            force_recompute=force_recompute,
            prepared_values=values_by_time,
        )
        return replace(result, primitive_counters=index.counters())
    finally:
        _EXECUTION_INDEX.reset(token)


def _analyze_predictive_edges_v41(
    *,
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    simulation: SimulationConfiguration,
    stage_cache: dict[str, PhaseEvidence] | None = None,
    force_recompute: bool = False,
    prepared_values: dict[datetime, dict[str, Decimal]] | None = None,
) -> PredictiveEdgeV41Result:
    """Reference implementation used beneath the indexed public boundary."""

    values_by_time = prepared_values or {
        row.candle_open_time: row.values for row in build_derived_feature_rows(matrix)
    }
    cache = stage_cache or {}
    writes: dict[str, PhaseEvidence] = {}
    directional = tuple(DIRECTIONAL_BASE) + tuple(
        item.name
        for item in DERIVED_PREDICTORS
        if item.role is PredictorRole.DIRECTIONAL
    )
    contexts = (
        "atr_14_relative",
        "relative_volume_20",
        "trend_regime",
        "rsi_regime",
        "rolling_breakout_position_288",
    )
    directional_rank, a1_computed, a1_reused = _screen_directional_relationships(
        values_by_time, matrix, windows, directional, cache, writes, force_recompute
    )
    context_rank, a2_computed, a2_reused = _screen_predictors_cached(
        values_by_time,
        matrix,
        windows,
        contexts,
        False,
        cache,
        writes,
        force_recompute,
    )
    context_rank = context_rank[:CONTEXT_RELATIONSHIP_LIMIT]
    generated_pairs = tuple(
        (direction, context)
        for direction in directional_rank
        for context in context_rank
    )[:INTERACTION_GENERATION_LIMIT]
    computed = reused = 0
    interactions: list[RegimeConditionedRelationship] = []
    for direction, context in generated_pairs:
        frozen = _freeze_regimes(values_by_time, windows, context)
        selected_bin, lower, upper, mapping, action = _freeze_directional_rule(
            values_by_time, matrix, windows, direction
        )
        baseline = _evaluate(
            values_by_time,
            matrix,
            windows,
            direction.predictor,
            None,
            "training",
            simulation,
            action,
            direction.cadence,
            direction.horizon,
            lower,
            upper,
        )
        for regime in frozen:
            if len(interactions) >= INTERACTION_GENERATION_LIMIT:
                break
            evidence, stage_computed, stage_reused = _evaluate_cached(
                values_by_time,
                matrix,
                windows,
                direction.predictor,
                context,
                regime,
                "training",
                simulation,
                action,
                direction.cadence,
                direction.horizon,
                lower,
                upper,
                cache,
                writes,
                force_recompute,
            )
            computed += stage_computed
            reused += stage_reused
            improvement = (evidence.net_movement or Decimal(0)) - (
                baseline.net_movement or Decimal(0)
            )
            interactions.append(
                RegimeConditionedRelationship(
                    direction.predictor,
                    direction.version,
                    context,
                    _predictor_version(context),
                    regime.regime,
                    direction.cadence,
                    direction.horizon,
                    regime,
                    selected_bin,
                    lower,
                    upper,
                    mapping,
                    action,
                    "balanced_v41",
                    evidence,
                    baseline,
                    improvement,
                    _fingerprint(
                        direction,
                        context,
                        regime.fingerprint,
                        selected_bin,
                        lower,
                        upper,
                        mapping,
                        action,
                        "balanced_v41",
                    ),
                )
            )
        if len(interactions) >= INTERACTION_GENERATION_LIMIT:
            break
    retained = _diversified_interactions(tuple(interactions))
    validation_values: list[V41Candidate] = []
    validation_computed = validation_reused = 0
    for relationship in retained[:VALIDATION_LIMIT]:
        validation, stage_computed, stage_reused = _evaluate_cached(
            values_by_time,
            matrix,
            windows,
            relationship.directional_predictor,
            relationship.context_predictor,
            relationship.frozen_regime,
            "validation",
            simulation,
            relationship.action,
            relationship.cadence,
            relationship.horizon,
            relationship.directional_lower_bound,
            relationship.directional_upper_bound,
            cache,
            writes,
            force_recompute,
        )
        validation_computed += stage_computed
        validation_reused += stage_reused
        selected_for_test = validation.passes and relationship.context_improvement > 0
        validation_values.append(
            V41Candidate(
                relationship,
                validation,
                None,
                selected_for_test,
                False,
                None if selected_for_test else _rejection(validation, relationship),
            )
        )
    selected_candidates = tuple(
        sorted(
            (item for item in validation_values if item.selected_for_test),
            key=_candidate_rank,
        )[:ISOLATED_TEST_LIMIT]
    )
    selected_ids = {item.relationship.fingerprint for item in selected_candidates}
    final: list[V41Candidate] = []
    test_computed = test_reused = 0
    for candidate in validation_values:
        if candidate.relationship.fingerprint not in selected_ids:
            final.append(candidate)
            continue
        relationship = candidate.relationship
        test, stage_computed, stage_reused = _evaluate_cached(
            values_by_time,
            matrix,
            windows,
            relationship.directional_predictor,
            relationship.context_predictor,
            relationship.frozen_regime,
            "test",
            simulation,
            relationship.action,
            relationship.cadence,
            relationship.horizon,
            relationship.directional_lower_bound,
            relationship.directional_upper_bound,
            cache,
            writes,
            force_recompute,
        )
        test_computed += stage_computed
        test_reused += stage_reused
        final.append(
            V41Candidate(
                relationship,
                candidate.validation,
                test,
                True,
                test.passes,
                None if test.passes else _rejection(test, relationship),
            )
        )
    confirmed = tuple(item for item in final if item.confirmed)
    hypotheses = _build_hypotheses(confirmed, windows)
    actionable = tuple(
        sorted(
            (item for item in final if not item.selected_for_test),
            key=_candidate_rank,
        )[:50]
    )
    diagnostic = tuple(context_rank)
    conclusion = _conclusion(tuple(final), confirmed)
    return PredictiveEdgeV41Result(
        directional_rank,
        context_rank,
        retained,
        tuple(final),
        actionable,
        diagnostic,
        confirmed,
        hypotheses,
        conclusion,
        V41StageCounters(
            len(directional), a1_computed, a1_reused, len(directional_rank)
        ),
        V41StageCounters(len(contexts), a2_computed, a2_reused, len(context_rank)),
        V41StageCounters(len(interactions), computed, reused, len(retained)),
        V41StageCounters(
            len(validation_values),
            validation_computed,
            validation_reused,
            len(selected_candidates),
        ),
        V41StageCounters(
            len(selected_candidates), test_computed, test_reused, len(confirmed)
        ),
        writes,
    )


def _base_values(row: HistoricalFeatureRow) -> dict[str, Decimal]:
    ema20 = _ratio(row.close_price, row.ema_20)
    ema50 = _ratio(row.close_price, row.ema_50)
    spread = _ratio(row.ema_20, row.ema_50)
    rsi_centered = (row.rsi_14 - Decimal(50)) / Decimal(50)
    return {
        "ema_20_relative": ema20,
        "ema_50_relative": ema50,
        "rsi_14": rsi_centered,
        "return_1": row.return_1,
        "return_12": row.return_12,
        "trend_component": spread,
        "rsi_contribution": rsi_centered,
        "return_1_contribution": row.return_1,
        "return_12_contribution": row.return_12,
        "directional_score": spread + rsi_centered + row.return_1 + row.return_12,
        "overall_score": spread + rsi_centered + row.return_1 + row.return_12,
        "atr_14_relative": row.atr_14 / row.close_price
        if row.close_price
        else Decimal(0),
        "relative_volume_20": row.relative_volume_20,
        "volume_confirmation": row.relative_volume_20,
        "trend_regime": spread,
        "rsi_regime": rsi_centered,
    }


def _ratio(current: Decimal, previous: Decimal) -> Decimal:
    return Decimal(0) if previous == 0 else current / previous - Decimal(1)


def _phase_bounds(window: WalkForwardWindow, phase: str) -> tuple[datetime, datetime]:
    return getattr(window, f"{phase}_from"), getattr(window, f"{phase}_to")


def _sampled_pairs(
    values: dict[datetime, dict[str, Decimal]],
    matrix: HistoricalFeatureMatrix,
    window: WalkForwardWindow,
    phase: str,
    predictor: str,
    cadence: timedelta,
    horizon: timedelta,
) -> tuple[tuple[datetime, Decimal, Decimal], ...]:
    """Return predictor and T+horizon returns on cadence-aligned phase boundaries."""

    index = _EXECUTION_INDEX.get()
    key = (window.ordinal, phase, predictor, cadence, horizon)
    if index is not None and index.matrix is matrix and index.values is values:
        index.sampled_requests += 1
        existing = index.sampled.get(key)
        if existing is not None:
            return existing
        index.sampled_builds += 1
        prices = index.prices
    else:
        prices = {row.candle_open_time: row.close_price for row in matrix.rows}
    start, end = _phase_bounds(window, phase)
    result: list[tuple[datetime, Decimal, Decimal]] = []
    for at in sorted(values):
        future = at + horizon
        if (
            start <= at < end
            and future < end
            and (at - start) % cadence == timedelta(0)
            and predictor in values[at]
            and at in prices
            and future in prices
            and prices[at] != 0
        ):
            result.append(
                (at, values[at][predictor], prices[future] / prices[at] - Decimal(1))
            )
    value = tuple(result)
    if index is not None and index.matrix is matrix and index.values is values:
        index.sampled[key] = value
    return value


def _correlation(pairs: tuple[tuple[Decimal, Decimal], ...]) -> Decimal:
    """Scale-invariant Pearson correlation calculated with Decimal inputs."""

    if len(pairs) < 2:
        return Decimal(0)
    mean_x = sum((x for x, _ in pairs), Decimal(0)) / Decimal(len(pairs))
    mean_y = sum((y for _, y in pairs), Decimal(0)) / Decimal(len(pairs))
    covariance = sum(((x - mean_x) * (y - mean_y) for x, y in pairs), Decimal(0))
    variance_x = sum(((x - mean_x) ** 2 for x, _ in pairs), Decimal(0))
    variance_y = sum(((y - mean_y) ** 2 for _, y in pairs), Decimal(0))
    if variance_x == 0 or variance_y == 0:
        return Decimal(0)
    return covariance / (variance_x * variance_y).sqrt()


def _screen_directional_relationships(
    values: dict[datetime, dict[str, Decimal]],
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    names: tuple[str, ...],
    cache: dict[str, PhaseEvidence],
    writes: dict[str, PhaseEvidence],
    force_recompute: bool,
) -> tuple[tuple[DirectionalScreen, ...], int, int]:
    scored: list[DirectionalScreen] = []
    computed = reused = 0
    for name in names:
        for cadence in DISCOVERY_CADENCES:
            for horizon in DISCOVERY_HORIZONS:
                if horizon < cadence or horizon % cadence:
                    continue
                window_scores: list[Decimal] = []
                observation_counts: list[int] = []
                fingerprints: list[str] = []
                eligible_windows = 0
                for window in windows:
                    start, end = _phase_bounds(window, "training")
                    slice_fingerprint = _slice_fingerprint(matrix, start, end)
                    key = _fingerprint(
                        "a1", name, cadence, horizon, start, end, slice_fingerprint
                    )
                    fingerprints.append(key)
                    if not force_recompute and key in cache:
                        cached = cache[key]
                        score = cached.gross_movement or Decimal(0)
                        observation_counts.append(cached.observations)
                        eligible_windows += int(
                            cached.observations >= MINIMUM_OBSERVATIONS
                        )
                        reused += 1
                    else:
                        raw = _sampled_pairs(
                            values, matrix, window, "training", name, cadence, horizon
                        )
                        score = abs(_correlation(tuple((x, y) for _, x, y in raw)))
                        observation_counts.append(len(raw))
                        eligible_windows += int(len(raw) >= MINIMUM_OBSERVATIONS)
                        writes[key] = PhaseEvidence(
                            len(raw),
                            score,
                            score,
                            score,
                            None,
                            None,
                            None,
                            None,
                            len(raw) >= MINIMUM_OBSERVATIONS,
                            True,
                            True,
                        )
                        computed += 1
                    window_scores.append(score)
                if (
                    window_scores
                    and eligible_windows >= MINIMUM_CONFIRMATION_WINDOWS
                    and any(score > 0 for score in window_scores)
                ):
                    mean_score = sum(window_scores, Decimal(0)) / Decimal(
                        len(window_scores)
                    )
                    mean_observations = Decimal(sum(observation_counts)) / Decimal(
                        len(observation_counts)
                    )
                    coverage = min(Decimal(1), mean_observations / Decimal(24))
                    score = mean_score * coverage
                    scored.append(
                        DirectionalScreen(
                            name,
                            _predictor_version(name),
                            cadence,
                            horizon,
                            score,
                            _fingerprint(name, cadence, horizon, tuple(fingerprints)),
                        )
                    )
    # Round-robin by predictor prevents one relationship from consuming A1.
    buckets: dict[str, list[DirectionalScreen]] = {}
    for item in sorted(scored, key=lambda value: (-value.score, value.fingerprint)):
        buckets.setdefault(item.predictor, []).append(item)
    retained: list[DirectionalScreen] = []
    while len(retained) < DIRECTIONAL_RELATIONSHIP_LIMIT and any(buckets.values()):
        for name in sorted(buckets):
            if buckets[name] and len(retained) < DIRECTIONAL_RELATIONSHIP_LIMIT:
                retained.append(buckets[name].pop(0))
    return tuple(retained), computed, reused


def _freeze_directional_rule(
    values: dict[datetime, dict[str, Decimal]],
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    relationship: DirectionalScreen,
) -> tuple[int, Decimal | None, Decimal | None, DirectionMapping, SuggestedAction]:
    """Learn one action-aligned quantile rule exclusively from training evidence."""

    pairs = tuple(
        pair
        for window in windows
        for pair in _sampled_pairs(
            values,
            matrix,
            window,
            "training",
            relationship.predictor,
            relationship.cadence,
            relationship.horizon,
        )
    )
    ordered = sorted(value for _, value, _ in pairs)
    if not ordered:
        return 1, None, None, DirectionMapping.POSITIVE, SuggestedAction.LONG_ENTRY
    cuts = tuple(
        ordered[len(ordered) * index // DIRECTIONAL_BIN_COUNT] for index in (1, 2)
    )
    bins: list[tuple[Decimal, int, Decimal | None, Decimal | None, Decimal]] = []
    for ordinal in range(1, DIRECTIONAL_BIN_COUNT + 1):
        lower = None if ordinal == 1 else cuts[ordinal - 2]
        upper = None if ordinal == DIRECTIONAL_BIN_COUNT else cuts[ordinal - 1]
        returns = tuple(
            gross
            for _, value, gross in pairs
            if (lower is None or value > lower) and (upper is None or value <= upper)
        )
        if returns:
            mean = sum(returns, Decimal(0)) / Decimal(len(returns))
            bins.append((abs(mean), ordinal, lower, upper, mean))
    _, ordinal, lower, upper, mean = sorted(bins, key=lambda item: (-item[0], item[1]))[
        0
    ]
    action = SuggestedAction.LONG_ENTRY if mean >= 0 else SuggestedAction.EXIT_OR_AVOID
    mapping = DirectionMapping.POSITIVE if mean >= 0 else DirectionMapping.NEGATIVE
    return ordinal, lower, upper, mapping, action


def _screen_predictors_cached(
    values: dict[datetime, dict[str, Decimal]],
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    names: tuple[str, ...],
    directional: bool,
    cache: dict[str, PhaseEvidence],
    writes: dict[str, PhaseEvidence],
    force_recompute: bool,
) -> tuple[tuple[str, ...], int, int]:
    scored: list[tuple[Decimal, str]] = []
    computed = reused = 0
    for name in names:
        window_scores: list[Decimal] = []
        for window in windows:
            start, end = _phase_bounds(window, "training")
            key = _fingerprint(
                "a2", name, start, end, _slice_fingerprint(matrix, start, end)
            )
            if not force_recompute and key in cache:
                score = cache[key].gross_movement or Decimal(0)
                reused += 1
            else:
                raw = _sampled_pairs(
                    values,
                    matrix,
                    window,
                    "training",
                    name,
                    DISCOVERY_CADENCES[0],
                    DISCOVERY_HORIZONS[0],
                )
                pairs = tuple((value, gross) for _, value, gross in raw)
                score = _screen_score(pairs, directional)
                writes[key] = PhaseEvidence(
                    len(pairs),
                    score,
                    score,
                    score,
                    None,
                    None,
                    None,
                    None,
                    len(pairs) >= MINIMUM_OBSERVATIONS,
                    True,
                    True,
                )
                computed += 1
            window_scores.append(score)
        if not window_scores:
            continue
        score = sum(window_scores, Decimal(0)) / Decimal(len(window_scores))
        scored.append((score, name))
    return (
        tuple(name for _, name in sorted(scored, key=lambda item: (-item[0], item[1]))),
        computed,
        reused,
    )


def _screen_score(
    pairs: tuple[tuple[Decimal, Decimal], ...], directional: bool
) -> Decimal:
    if not pairs:
        return Decimal(0)
    if directional:
        return abs(_correlation(pairs))
    overall = sum((abs(y) for _, y in pairs), Decimal(0)) / Decimal(len(pairs))
    ordered_pairs = sorted(pairs)
    tail = ordered_pairs[len(ordered_pairs) * 2 // 3 :]
    if not tail or overall == 0:
        return Decimal(0)
    segmented = sum((abs(y) for _, y in tail), Decimal(0)) / Decimal(len(tail))
    return abs(segmented - overall) / overall


def _freeze_regimes(
    values: dict[datetime, dict[str, Decimal]],
    windows: tuple[WalkForwardWindow, ...],
    context: str,
) -> tuple[FrozenRegime, ...]:
    training = sorted(
        row[context]
        for at, row in values.items()
        if context in row
        and any(window.training_from <= at < window.training_to for window in windows)
    )
    if len(training) < 3:
        return ()
    lower, upper = training[len(training) // 3], training[(len(training) * 2) // 3]
    names = (ContextRegime.LOW, ContextRegime.NORMAL, ContextRegime.HIGH)
    if context == "trend_regime":
        names = (ContextRegime.BEARISH, ContextRegime.NEUTRAL, ContextRegime.BULLISH)
    elif context == "rolling_breakout_position_288":
        names = (ContextRegime.LOWER, ContextRegime.MIDDLE, ContextRegime.UPPER)
    return tuple(
        FrozenRegime(
            context,
            _predictor_version(context),
            name,
            None if index == 0 else lower if index == 1 else upper,
            lower if index == 0 else upper if index == 1 else None,
            len(training),
            _fingerprint(context, name, lower, upper),
        )
        for index, name in enumerate(names)
    )


def _evaluate(
    values: dict[datetime, dict[str, Decimal]],
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    predictor: str,
    regime: FrozenRegime | None,
    phase: str,
    simulation: SimulationConfiguration,
    action: SuggestedAction | None = None,
    cadence: timedelta = timedelta(minutes=30),
    horizon: timedelta = timedelta(minutes=30),
    directional_lower_bound: Decimal | None = None,
    directional_upper_bound: Decimal | None = None,
) -> PhaseEvidence:
    returns: list[Decimal] = []
    hits: list[Decimal] = []
    universe_returns: list[Decimal] = []
    for window in windows:
        for at, predictor_value, gross in _sampled_pairs(
            values, matrix, window, phase, predictor, cadence, horizon
        ):
            universe_returns.append(gross)
            row = values[at]
            if not (
                (
                    directional_lower_bound is None
                    or predictor_value > directional_lower_bound
                )
                and (
                    directional_upper_bound is None
                    or predictor_value <= directional_upper_bound
                )
            ):
                continue
            if regime is not None and not _inside(
                row.get(regime.context_predictor), regime
            ):
                continue
            returns.append(gross)
            selected_action = action or SuggestedAction.LONG_ENTRY
            expected_positive = selected_action is SuggestedAction.LONG_ENTRY
            hits.append(Decimal(expected_positive == (gross >= 0)))
    if not returns:
        return PhaseEvidence(
            0, None, None, None, None, None, None, None, False, False, False
        )
    gross = sum(returns, Decimal(0)) / Decimal(len(returns))
    selected_action = action or (
        SuggestedAction.LONG_ENTRY if gross >= 0 else SuggestedAction.EXIT_OR_AVOID
    )
    aligned = gross if selected_action is SuggestedAction.LONG_ENTRY else -gross
    cost = Decimal(2) * (simulation.fee_rate + simulation.slippage_rate)
    net = aligned - cost
    hit = sum(hits, Decimal(0)) / Decimal(len(hits))
    up_rate = sum(
        (Decimal(value >= 0) for value in universe_returns), Decimal(0)
    ) / Decimal(len(universe_returns))
    benchmark = max(up_rate, Decimal(1) - up_rate)
    improvement = hit - benchmark
    mean = gross
    variance = sum(((item - mean) ** 2 for item in returns), Decimal(0)) / Decimal(
        max(1, len(returns) - 1)
    )
    standard_error = Decimal(str(sqrt(float(variance / Decimal(len(returns))))))
    stable = len(returns) >= MINIMUM_OBSERVATIONS and hit > Decimal("0.50")
    economic = net >= ECONOMIC_MARGIN
    return PhaseEvidence(
        len(returns),
        gross,
        aligned,
        net,
        hit,
        benchmark,
        improvement,
        standard_error,
        stable,
        economic,
        stable and economic and improvement >= MINIMUM_HIT_IMPROVEMENT,
    )


def _evaluate_cached(
    values: dict[datetime, dict[str, Decimal]],
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    directional: str,
    context: str,
    regime: FrozenRegime,
    phase: str,
    simulation: SimulationConfiguration,
    action: SuggestedAction,
    cadence: timedelta,
    horizon: timedelta,
    directional_lower_bound: Decimal | None,
    directional_upper_bound: Decimal | None,
    cache: dict[str, PhaseEvidence],
    writes: dict[str, PhaseEvidence],
    force_recompute: bool,
) -> tuple[PhaseEvidence, int, int]:
    """Reuse exact completed phase slices and aggregate them deterministically."""

    evidence: list[PhaseEvidence] = []
    computed = reused = 0
    for window in windows:
        key = _cache_key(
            matrix,
            (window,),
            directional,
            context,
            regime,
            phase,
            simulation,
            cadence,
            horizon,
            directional_lower_bound,
            directional_upper_bound,
            action,
        )
        if not force_recompute and key in cache:
            value = cache[key]
            reused += 1
        else:
            value = _evaluate(
                values,
                matrix,
                (window,),
                directional,
                regime,
                phase,
                simulation,
                action,
                cadence,
                horizon,
                directional_lower_bound,
                directional_upper_bound,
            )
            writes[key] = value
            computed += 1
        evidence.append(value)
    return _combine_phase_evidence(tuple(evidence)), computed, reused


def _combine_phase_evidence(values: tuple[PhaseEvidence, ...]) -> PhaseEvidence:
    total = sum(item.observations for item in values)
    if total == 0:
        return PhaseEvidence(
            0, None, None, None, None, None, None, None, False, False, False
        )

    def weighted(name: str) -> Decimal | None:
        pairs = tuple(
            (getattr(item, name), item.observations)
            for item in values
            if getattr(item, name) is not None and item.observations
        )
        if not pairs:
            return None
        return sum((value * count for value, count in pairs), Decimal(0)) / Decimal(
            sum(count for _, count in pairs)
        )

    gross = weighted("gross_movement")
    aligned = weighted("aligned_gross_movement")
    net = weighted("net_movement")
    hit = weighted("directional_hit_rate")
    benchmark = weighted("benchmark")
    improvement = weighted("benchmark_improvement")
    standard_error = weighted("standard_error")
    usable_windows = tuple(item for item in values if item.observations)
    stable = (
        len(usable_windows) >= MINIMUM_CONFIRMATION_WINDOWS
        and all(item.stable for item in usable_windows)
        and all(
            (item.aligned_gross_movement or Decimal(0)) > 0 for item in usable_windows
        )
    )
    economic = net is not None and net >= ECONOMIC_MARGIN
    passes = (
        stable
        and economic
        and improvement is not None
        and improvement >= MINIMUM_HIT_IMPROVEMENT
    )
    return PhaseEvidence(
        total,
        gross,
        aligned,
        net,
        hit,
        benchmark,
        improvement,
        standard_error,
        stable,
        economic,
        passes,
    )


def _inside(value: Decimal | None, regime: FrozenRegime) -> bool:
    if value is None:
        return False
    return (regime.lower_bound is None or value >= regime.lower_bound) and (
        regime.upper_bound is None or value <= regime.upper_bound
    )


def _diversified_interactions(
    values: tuple[RegimeConditionedRelationship, ...],
) -> tuple[RegimeConditionedRelationship, ...]:
    buckets: dict[tuple[str, str], list[RegimeConditionedRelationship]] = {}
    for value in sorted(
        values,
        key=lambda item: (
            -item.context_improvement,
            -(item.training.net_movement or Decimal(0)),
            item.fingerprint,
        ),
    ):
        buckets.setdefault(
            (value.directional_predictor, value.context_predictor), []
        ).append(value)
    result: list[RegimeConditionedRelationship] = []
    while len(result) < INTERACTION_RETENTION_LIMIT and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(result) < INTERACTION_RETENTION_LIMIT:
                result.append(buckets[key].pop(0))
    return tuple(result)


def _candidate_rank(value: V41Candidate) -> tuple[object, ...]:
    return (
        -(value.validation.net_movement or Decimal(0)),
        -(value.validation.benchmark_improvement or Decimal(0)),
        value.relationship.fingerprint,
    )


def _rejection(
    evidence: PhaseEvidence, relationship: RegimeConditionedRelationship
) -> str:
    if relationship.context_improvement <= 0:
        return V41Conclusion.DIRECTIONAL_SIGNAL_NOT_IMPROVED_BY_CONTEXT.value
    if not evidence.economic:
        return V41Conclusion.CONTEXT_IMPROVES_SIGNAL_BUT_NOT_ECONOMIC.value
    if not evidence.stable:
        return V41Conclusion.ECONOMIC_INTERACTION_BUT_UNSTABLE.value
    return V41Conclusion.STABLE_INTERACTION_BELOW_BENCHMARK.value


def _conclusion(
    candidates: tuple[V41Candidate, ...], confirmed: tuple[V41Candidate, ...]
) -> V41Conclusion:
    if confirmed:
        return V41Conclusion.REGIME_CONDITIONED_EDGE_CONFIRMED
    reasons = {item.rejection_reason for item in candidates}
    for value in (
        V41Conclusion.CONTEXT_IMPROVES_SIGNAL_BUT_NOT_ECONOMIC,
        V41Conclusion.ECONOMIC_INTERACTION_BUT_UNSTABLE,
        V41Conclusion.STABLE_INTERACTION_BELOW_BENCHMARK,
        V41Conclusion.DIRECTIONAL_SIGNAL_NOT_IMPROVED_BY_CONTEXT,
    ):
        if value.value in reasons:
            return value
    return V41Conclusion.NO_REGIME_CONDITIONED_EDGE


def _build_hypotheses(
    confirmed: tuple[V41Candidate, ...],
    windows: tuple[WalkForwardWindow, ...],
) -> tuple[StrategyHypothesis, ...]:
    """Translate confirmed interactions into executable, versioned hypotheses."""

    hypotheses: list[StrategyHypothesis] = []
    for candidate in confirmed:
        relationship = candidate.relationship
        test = candidate.test
        if test is None:
            continue
        identity = _fingerprint(
            "strategy-hypothesis",
            relationship.directional_predictor,
            relationship.directional_version,
            relationship.selected_bin,
            relationship.directional_lower_bound,
            relationship.directional_upper_bound,
            relationship.direction_mapping,
            relationship.context_predictor,
            relationship.context_version,
            relationship.regime,
            relationship.frozen_regime.lower_bound,
            relationship.frozen_regime.upper_bound,
            relationship.cadence,
            relationship.horizon,
            relationship.action,
        )
        hypotheses.append(
            StrategyHypothesis(
                identity,
                relationship.directional_predictor,
                relationship.directional_version,
                PREDICTIVE_EDGE_V41_CONTRACT_VERSION,
                relationship.direction_mapping,
                relationship.selected_bin,
                relationship.directional_lower_bound,
                relationship.directional_upper_bound,
                relationship.action,
                relationship.horizon,
                Condition.ALL,
                (relationship.selected_bin,),
                test.gross_movement,
                test.net_movement
                if relationship.action is SuggestedAction.LONG_ENTRY
                else None,
                test.net_movement
                if relationship.action is SuggestedAction.EXIT_OR_AVOID
                else None,
                test.directional_hit_rate,
                Decimal(len(windows)) / Decimal(max(1, len(windows))),
                relationship.cadence,
                relationship.horizon,
                relationship.horizon,
                tuple(window.ordinal for window in windows),
                EvidenceStatus.PROMISING_POSITIVE
                if relationship.direction_mapping is DirectionMapping.POSITIVE
                else EvidenceStatus.PROMISING_NEGATIVE,
                (
                    "Confirmed in validation and isolated test across at least three "
                    "walk-forward windows using a training-frozen directional bin and "
                    "context regime."
                ),
                (
                    "Research evidence only; it does not activate a Strategy "
                    "automatically."
                ),
                relationship.context_predictor,
                relationship.context_version,
                relationship.regime.value,
                relationship.frozen_regime.lower_bound,
                relationship.frozen_regime.upper_bound,
            )
        )
    return tuple(sorted(hypotheses, key=lambda item: item.hypothesis_id))


def _predictor_version(name: str) -> int:
    return (
        DERIVED_PREDICTOR_VERSION
        if any(item.name == name for item in DERIVED_PREDICTORS)
        else 3
    )


def _cache_key(
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    directional: str,
    context: str,
    regime: FrozenRegime,
    phase: str,
    simulation: SimulationConfiguration,
    cadence: timedelta,
    horizon: timedelta,
    directional_lower_bound: Decimal | None,
    directional_upper_bound: Decimal | None,
    action: SuggestedAction,
) -> str:
    slices = tuple(
        (
            getattr(window, f"{phase}_from"),
            getattr(window, f"{phase}_to"),
            _slice_fingerprint(
                matrix, getattr(window, f"{phase}_from"), getattr(window, f"{phase}_to")
            ),
        )
        for window in windows
    )
    return _fingerprint(
        "cache",
        directional,
        _predictor_version(directional),
        context,
        _predictor_version(context),
        phase,
        regime.fingerprint,
        cadence,
        horizon,
        directional_lower_bound,
        directional_upper_bound,
        action,
        slices,
        simulation.fee_rate,
        simulation.slippage_rate,
    )


def _slice_fingerprint(
    matrix: HistoricalFeatureMatrix, start: datetime, end: datetime
) -> str:
    index = _EXECUTION_INDEX.get()
    key = (start, end)
    if index is not None and index.matrix is matrix:
        index.slice_requests += 1
        existing = index.slices.get(key)
        if existing is not None:
            return existing
        index.slice_builds += 1
    value = _fingerprint(
        tuple(row for row in matrix.rows if start <= row.candle_open_time < end)
    )
    if index is not None and index.matrix is matrix:
        index.slices[key] = value
    return value


def _fingerprint(*values: object) -> str:
    encoded = json.dumps(
        (PREDICTIVE_EDGE_V41_CONTRACT_VERSION, values),
        default=str,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()
