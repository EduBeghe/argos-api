"""Bounded regime-conditioned Predictive Edge Discovery v4.1.

The module is deliberately research-only. Derived predictors and regimes exist only in
memory, are learned from training, and are frozen before validation and isolated test.
"""

import hashlib
import json
from dataclasses import dataclass
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
    """Run A1/A2/A3/B/C sequentially with frozen training regimes."""

    rows = build_derived_feature_rows(matrix)
    values_by_time = {row.candle_open_time: row.values for row in rows}
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
    fã¿}¶‰žËkºwµç@€€™½È…Ð°É½Ü¥¸Ù…±Õ•Ì¹¥Ñ•µÌ ¤(€€€€€€€¥˜½¹Ñ•áÐ¥¸É½Ü(€€€€€€€…¹…¹ä¡Ý¥¹‘½Ü¹ÑÉ…¥¹¥¹}™É½´€ðô…Ð€ðÝ¥¹‘½Ü¹ÑÉ…¥¹¥¹}Ñ¼™½ÈÝ¥¹‘½Ü¥¸Ý¥¹‘½ÝÌ¤(€€€€¤(€€€¥˜±•¸¡ÑÉ…¥¹¥¹œ¤€ð€Ìè(€€€€€€€É•ÑÕÉ¸€ ¤(€€€±½Ý•È°ÕÁÁ•È€ôÑÉ…¥¹¥¹m±•¸¡ÑÉ…¥¹¥¹œ¤€¼¼€Ít°ÑÉ…¥¹¥¹l¡±•¸¡ÑÉ…¥¹¥¹œ¤€¨€È¤€¼¼€Ít(€€€¹…µ•Ì€ô€¡½¹Ñ•áÑI•¥µ”¹1=\°½¹Ñ•áÑI•¥µ”¹9=I50°½¹Ñ•áÑI•¥µ”¹!% ¤(€€€¥˜½¹Ñ•áÐ€ôô€‰ÑÉ•¹‘}É•¥µ”ˆè(€€€€€€€¹…µ•Ì€ô€¡½¹Ñ•áÑI•¥µ”¹	I%M °½¹Ñ•áÑI•¥µ”¹9UQI0°½¹Ñ•áÑI•¥µ”¹	U11%M ¤(€€€•±¥˜½¹Ñ•áÐ€ôô€‰É½±±¥¹}‰É•…­½ÕÑ}Á½Í¥Ñ¥½¹|Èààˆè(€€€€€€€¹…µ•Ì€ô€¡½¹Ñ•áÑI•¥µ”¹1=]H°½¹Ñ•áÑI•¥µ”¹5%1°½¹Ñ•áÑI•¥µ”¹UAAH¤(€€€É•ÑÕÉ¸ÑÕÁ±” (€€€€€€€É½é•¹I•¥µ” (€€€€€€€€€€€½¹Ñ•áÐ°(€€€€€€€€€€€}ÁÉ•‘¥Ñ½É}Ù•ÉÍ¥½¸¡½¹Ñ•áÐ¤°(€€€€€€€€€€€¹…µ”°(€€€€€€€€€€€9½¹”¥˜¥¹‘•à€ôô€À•±Í”±½Ý•È¥˜¥¹‘•à€ôô€Ä•±Í”ÕÁÁ•È°(€€€€€€€€€€€±½Ý•È¥˜¥¹‘•à€ôô€À•±Í”ÕÁÁ•È¥˜¥¹‘•à€ôô€Ä•±Í”9½¹”°(€€€€€€€€€€€±•¸¡ÑÉ…¥¹¥¹œ¤°(€€€€€€€€€€€}™¥¹•ÉÁÉ¥¹Ð¡½¹Ñ•áÐ°¹…µ”°±½Ý•È°ÕÁÁ•È¤°(€€€€€€€€¤(€€€€€€€™½È¥¹‘•à°¹…µ”¥¸•¹Õµ•É…Ñ”¡¹…µ•Ì¤(€€€€¤(()‘•˜}•Ù…±Õ…Ñ” (€€€Ù…±Õ•Ìè‘¥Ñm‘…Ñ•Ñ¥µ”°‘¥ÑmÍÑÈ°•¥µ…±ut°(€€€µ…ÑÉ¥àè!¥ÍÑ½É¥…±•…ÑÕÉ•5…ÑÉ¥à°(€€€Ý¥¹‘½ÝÌèÑÕÁ±•m]…±­½ÉÝ…É‘]¥¹‘½Ü°€¸¸¹t°(€€€ÁÉ•‘¥Ñ½ÈèÍÑÈ°(€€€É•¥µ”èÉ½é•¹I•¥µ”ð9½¹”°(€€€Á¡…Í”èÍÑÈ°(€€€Í¥µÕ±…Ñ¥½¸èM¥µÕ±…Ñ¥½¹½¹™¥ÕÉ…Ñ¥½¸°(€€€…Ñ¥½¸èMÕ•ÍÑ•‘Ñ¥½¸ð9½¹”€ô9½¹”°(€€€…‘•¹”èÑ¥µ•‘•±Ñ„€ôÑ¥µ•‘•±Ñ„¡µ¥¹ÕÑ•ÌôÌÀ¤°(€€€¡½É¥é½¸èÑ¥µ•‘•±Ñ„€ôÑ¥µ•‘•±Ñ„¡µ¥¹ÕÑ•ÌôÌÀ¤°(€€€‘¥É•Ñ¥½¹…±}±½Ý•É}‰½Õ¹è•¥µ…°ð9½¹”€ô9½¹”°(€€€‘¥É•Ñ¥½¹…±}ÕÁÁ•É}‰½Õ¹è•¥µ…°ð9½¹”€ô9½¹”°(¤€´øA¡…Í•Ù¥‘•¹”è(€€€É•ÑÕÉ¹Ìè±¥ÍÑm•¥µ…±t€ômt(€€€¡¥ÑÌè±¥ÍÑm•¥µ…±t€ômt(€€€Õ¹¥Ù•ÉÍ•}É•ÑÕÉ¹Ìè±¥ÍÑm•¥µ…±t€ômt(€€€™½ÈÝ¥¹‘½Ü¥¸Ý¥¹‘½ÝÌè(€€€€€€€™½È…Ð°ÁÉ•‘¥Ñ½É}Ù…±Õ”°É½ÍÌ¥¸}Í…µÁ±•‘}Á…¥ÉÌ (€€€€€€€€€€€Ù…±Õ•Ì°µ…ÑÉ¥à°Ý¥¹‘½Ü°Á¡…Í”°ÁÉ•‘¥Ñ½È°…‘•¹”°¡½É¥é½¸(€€€€€€€€¤è(€€€€€€€€€€€Õ¹¥Ù•ÉÍ•}É•ÑÕÉ¹Ì¹…ÁÁ•¹¡É½ÍÌ¤(€€€€€€€€€€€É½Ü€ôÙ…±Õ•Ím…Ñt(€€€€€€€€€€€¥˜¹½Ð€ (€€€€€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€€€€‘¥É•Ñ¥½¹…±}±½Ý•É}‰½Õ¹¥Ì9½¹”(€€€€€€€€€€€€€€€€€€€½ÈÁÉ•‘¥Ñ½É}Ù…±Õ”€ø‘¥É•Ñ¥½¹…±}±½Ý•É}‰½Õ¹(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€…¹€ (€€€€€€€€€€€€€€€€€€€‘¥É•Ñ¥½¹…±}ÕÁÁ•É}‰½Õ¹¥Ì9½¹”(€€€€€€€€€€€€€€€€€€€½ÈÁÉ•‘¥Ñ½É}Ù…±Õ”€ðô‘¥É•Ñ¥½¹…±}ÕÁÁ•É}‰½Õ¹(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€¤è(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€€€€€¥˜É•¥µ”¥Ì¹½Ð9½¹”…¹¹½Ð}¥¹Í¥‘” (€€€€€€€€€€€€€€€É½Ü¹•Ð¡É•¥µ”¹½¹Ñ•áÑ}ÁÉ•‘¥Ñ½È¤°É•¥µ”(€€€€€€€€€€€€¤è(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€€€€€É•ÑÕÉ¹Ì¹…ÁÁ•¹¡É½ÍÌ¤(€€€€€€€€€€€Í•±•Ñ•‘}…Ñ¥½¸€ô…Ñ¥½¸½ÈMÕ•ÍÑ•‘Ñ¥½¸¹1=9}9QId(€€€€€€€€€€€•áÁ•Ñ•‘}Á½Í¥Ñ¥Ù”€ôÍ•±•Ñ•‘}…Ñ¥½¸¥ÌMÕ•ÍÑ•‘Ñ¥½¸¹1=9}9QId(€€€€€€€€€€€¡¥ÑÌ¹…ÁÁ•¹¡•¥µ…°¡•áÁ•Ñ•‘}Á½Í¥Ñ¥Ù”€ôô€¡É½ÍÌ€øô€À¤¤¤(€€€¥˜¹½ÐÉ•ÑÕÉ¹Ìè(€€€€€€€É•ÑÕÉ¸A¡…Í•Ù¥‘•¹” (€€€€€€€€€€€€À°9½¹”°9½¹”°9½¹”°9½¹”°9½¹”°9½¹”°9½¹”°…±Í”°…±Í”°…±Í”(€€€€€€€€¤(€€€É½ÍÌ€ôÍÕ´¡É•ÑÕÉ¹Ì°•¥µ…° À¤¤€¼•¥µ…°¡±•¸¡É•ÑÕÉ¹Ì¤¤(€€€Í•±•Ñ•‘}…Ñ¥½¸€ô…Ñ¥½¸½È€ (€€€€€€€MÕ•ÍÑ•‘Ñ¥½¸¹1=9}9QId¥˜É½ÍÌ€øô€À•±Í”MÕ•ÍÑ•‘Ñ¥½¸¹a%Q}=I}Y=%(€€€€¤(€€€…±¥¹•€ôÉ½ÍÌ¥˜Í•±•Ñ•‘}…Ñ¥½¸¥ÌMÕ•ÍÑ•‘Ñ¥½¸¹1=9}9QId•±Í”€µÉ½ÍÌ(€€€½ÍÐ€ô•¥µ…° È¤€¨€¡Í¥µÕ±…Ñ¥½¸¹™••}É…Ñ”€¬Í¥µÕ±…Ñ¥½¸¹Í±¥ÁÁ…•}É…Ñ”¤(€€€¹•Ð€ô…±¥¹•€´½ÍÐ(€€€¡¥Ð€ôÍÕ´¡¡¥ÑÌ°•¥µ…° À¤¤€¼•¥µ…°¡±•¸¡¡¥ÑÌ¤¤(€€€ÕÁ}É…Ñ”€ôÍÕ´ (€€€€€€€€¡•¥µ…°¡Ù…±Õ”€øô€À¤™½ÈÙ…±Õ”¥¸Õ¹¥Ù•ÉÍ•}É•ÑÕÉ¹Ì¤°•¥µ…° À¤(€€€€¤€¼•¥µ…°¡±•¸¡Õ¹¥Ù•ÉÍ•}É•ÑÕÉ¹Ì¤¤(€€€‰•¹¡µ…É¬€ôµ…à¡ÕÁ}É…Ñ”°•¥µ…° Ä¤€´ÕÁ}É…Ñ”¤(€€€¥µÁÉ½Ù•µ•¹Ð€ô¡¥Ð€´‰•¹¡µ…É¬(€€€µ•…¸€ôÉ½ÍÌ(€€€Ù…É¥…¹”€ôÍÕ´  ¡¥Ñ•´€´µ•…¸¤€¨¨€È™½È¥Ñ•´¥¸É•ÑÕÉ¹Ì¤°•¥µ…° À¤¤€¼•¥µ…° (€€€€€€€µ…à Ä°±•¸¡É•ÑÕÉ¹Ì¤€´€Ä¤(€€€€¤(€€€ÍÑ…¹‘…É‘}•ÉÉ½È€ô•¥µ…°¡ÍÑÈ¡ÍÅÉÐ¡™±½…Ð¡Ù…É¥…¹”€¼•¥µ…°¡±•¸¡É•ÑÕÉ¹Ì¤¤¤¤¤¤(€€€ÍÑ…‰±”€ô±•¸¡É•ÑÕÉ¹Ì¤€øô5%9%5U5}=	MIYQ%=9L…¹¡¥Ð€ø•¥µ…° ˆÀ¸ÔÀˆ¤(€€€•½¹½µ¥Œ€ô¹•Ð€øô=9=5%}5I%8(€€€É•ÑÕÉ¸A¡…Í•Ù¥‘•¹” (€€€€€€€±•¸¡É•ÑÕÉ¹Ì¤°(€€€€€€€É½ÍÌ°(€€€€€€€…±¥¹•°(€€€€€€€¹•Ð°(€€€€€€€¡¥Ð°(€€€€€€€‰•¹¡µ…É¬°(€€€€€€€¥µÁÉ½Ù•µ•¹Ð°(€€€€€€€ÍÑ…¹‘…É‘}•ÉÉ½È°(€€€€€€€ÍÑ…‰±”°(€€€€€€€•½¹½µ¥Œ°(€€€€€€€ÍÑ…‰±”…¹•½¹½µ¥Œ…¹¥µÁÉ½Ù•µ•¹Ð€øô5%9%5U5}!%Q}%5AI=Y59P°(€€€€¤(()‘•˜}•Ù…±Õ…Ñ•}…¡• (€€€Ù…±Õ•Ìè‘¥Ñm‘…Ñ•Ñ¥µ”°‘¥ÑmÍÑÈ°•¥µ…±ut°(€€€µ…ÑÉ¥àè!¥ÍÑ½É¥…±•…ÑÕÉ•5…ÑÉ¥à°(€€€Ý¥¹‘½ÝÌèÑÕÁ±•m]…±­½ÉÝ…É‘]¥¹‘½Ü°€¸¸¹t°(€€€‘¥É•Ñ¥½¹…°èÍÑÈ°(€€€½¹Ñ•áÐèÍÑÈ°(€€€É•¥µ”èÉ½é•¹I•¥µ”°(€€€Á¡…Í”èÍÑÈ°(€€€Í¥µÕ±…Ñ¥½¸èM¥µÕ±…Ñ¥½¹½¹™¥ÕÉ…Ñ¥½¸°(€€€…Ñ¥½¸èMÕ•ÍÑ•‘Ñ¥½¸°(€€€…‘•¹”èÑ¥µ•‘•±Ñ„°(€€€¡½É¥é½¸èÑ¥µ•‘•±Ñ„°(€€€‘¥É•Ñ¥½¹…±}±½Ý•É}‰½Õ¹è•¥µ…°ð9½¹”°(€€€‘¥É•Ñ¥½¹…±}ÕÁÁ•É}‰½Õ¹è•¥µ…°ð9½¹”°(€€€…¡”è‘¥ÑmÍÑÈ°A¡…Í•Ù¥‘•¹•t°(€€€ÝÉ¥Ñ•Ìè‘¥ÑmÍÑÈ°A¡…Í•Ù¥‘•¹•t°(€€€™½É•}É•½µÁÕÑ”è‰½½°°(¤€´øÑÕÁ±•mA¡…Í•Ù¥‘•¹”°¥¹Ð°¥¹Ñtè(€€€€ˆˆ‰I•ÕÍ”•á…Ð½µÁ±•Ñ•Á¡…Í”Í±¥•Ì…¹…É•…Ñ”Ñ¡•´‘•Ñ•Éµ¥¹¥ÍÑ¥…±±ä¸ˆˆˆ((€€€•Ù¥‘•¹”è±¥ÍÑmA¡…Í•Ù¥‘•¹•t€ômt(€€€½µÁÕÑ•€ôÉ•ÕÍ•€ô€À(€€€™½ÈÝ¥¹‘½Ü¥¸Ý¥¹‘½ÝÌè(€€€€€€€­•ä€ô}…¡•}­•ä (€€€€€€€€€€€µ…ÑÉ¥à°(€€€€€€€€€€€€¡Ý¥¹‘½Ü°¤°(€€€€€€€€€€€‘¥É•Ñ¥½¹…°°(€€€€€€€€€€€½¹Ñ•áÐ°(€€€€€€€€€€€É•¥µ”°(€€€€€€€€€€€Á¡…Í”°(€€€€€€€€€€€Í¥µÕ±…Ñ¥½¸°(€€€€€€€€€€€…‘•¹”°(€€€€€€€€€€€¡½É¥é½¸°(€€€€€€€€€€€‘¥É•Ñ¥½¹…±}±½Ý•É}‰½Õ¹°(€€€€€€€€€€€‘¥É•Ñ¥½¹…±}ÕÁÁ•É}‰½Õ¹°(€€€€€€€€€€€…Ñ¥½¸°(€€€€€€€€¤(€€€€€€€¥˜¹½Ð™½É•}É•½µÁÕÑ”…¹­•ä¥¸…¡”è(€€€€€€€€€€€Ù…±Õ”€ô…¡•m­•åt(€€€€€€€€€€€É•ÕÍ•€¬ô€Ä(€€€€€€€•±Í”è(€€€€€€€€€€€Ù…±Õ”€ô}•Ù…±Õ…Ñ” (€€€€€€€€€€€€€€€Ù…±Õ•Ì°(€€€€€€€€€€€€€€€µ…ÑÉ¥à°(€€€€€€€€€€€€€€€€¡Ý¥¹‘½Ü°¤°(€€€€€€€€€€€€€€€‘¥É•Ñ¥½¹…°°(€€€€€€€€€€€€€€€É•¥µ”°(€€€€€€€€€€€€€€€Á¡…Í”°(€€€€€€€€€€€€€€€Í¥µÕ±…Ñ¥½¸°(€€€€€€€€€€€€€€€…Ñ¥½¸°(€€€€€€€€€€€€€€€…‘•¹”°(€€€€€€€€€€€€€€€¡½É¥é½¸°(€€€€€€€€€€€€€€€‘¥É•Ñ¥½¹…±}±½Ý•É}‰½Õ¹°(€€€€€€€€€€€€€€€‘¥É•Ñ¥½¹…±}ÕÁÁ•É}‰½Õ¹°(€€€€€€€€€€€€¤(€€€€€€€€€€€ÝÉ¥Ñ•Ím­•åt€ôÙ…±Õ”(€€€€€€€€€€€½µÁÕÑ•€¬ô€Ä(€€€€€€€•Ù¥‘•¹”¹…ÁÁ•¹¡Ù…±Õ”¤(€€€É•ÑÕÉ¸}½µ‰¥¹•}Á¡…Í•}•Ù¥‘•¹”¡ÑÕÁ±”¡•Ù¥‘•¹”¤¤°½µÁÕÑ•°É•ÕÍ•(()‘•˜}½µ‰¥¹•}Á¡…Í•}•Ù¥‘•¹”¡Ù…±Õ•ÌèÑÕÁ±•mA¡…Í•Ù¥‘•¹”°€¸¸¹t¤€´øA¡…Í•Ù¥‘•¹”è(€€€Ñ½Ñ…°€ôÍÕ´¡¥Ñ•´¹½‰Í•ÉÙ…Ñ¥½¹Ì™½È¥Ñ•´¥¸Ù…±Õ•Ì¤(€€€¥˜Ñ½Ñ…°€ôô€Àè(€€€€€€€É•ÑÕÉ¸A¡…Í•Ù¥‘•¹” (€€€€€€€€€€€€À°9½¹”°9½¹”°9½¹”°9½¹”°9½¹”°9½¹”°9½¹”°…±Í”°…±Í”°…±Í”(€€€€€€€€¤((€€€‘•˜Ý•¥¡Ñ•¡¹…µ”èÍÑÈ¤€´ø•¥µ…°ð9½¹”è(€€€€€€€Á…¥ÉÌ€ôÑÕÁ±” (€€€€€€€€€€€€¡•Ñ…ÑÑÈ¡¥Ñ•´°¹…µ”¤°¥Ñ•´¹½‰Í•ÉÙ…Ñ¥½¹Ì¤(€€€€€€€€€€€™½È¥Ñ•´¥¸Ù…±Õ•Ì(€€€€€€€€€€€¥˜•Ñ…ÑÑÈ¡¥Ñ•´°¹…µ”¤¥Ì¹½Ð9½¹”…¹¥Ñ•´¹½‰Í•ÉÙ…Ñ¥½¹Ì(€€€€€€€€¤(€€€€€€€¥˜¹½ÐÁ…¥ÉÌè(€€€€€€€€€€€É•ÑÕÉ¸9½¹”(€€€€€€€É•ÑÕÉ¸ÍÕ´ ¡Ù…±Õ”€¨½Õ¹Ð™½ÈÙ…±Õ”°½Õ¹Ð¥¸Á…¥ÉÌ¤°•¥µ…° À¤¤€¼•¥µ…° (€€€€€€€€€€€ÍÕ´¡½Õ¹Ð™½È|°½Õ¹Ð¥¸Á…¥ÉÌ¤(€€€€€€€€¤((€€€É½ÍÌ€ôÝ•¥¡Ñ• ‰É½ÍÍ}µ½Ù•µ•¹Ðˆ¤(€€€…±¥¹•€ôÝ•¥¡Ñ• ‰…±¥¹•‘}É½ÍÍ}µ½Ù•µ•¹Ðˆ¤(€€€¹•Ð€ôÝ•¥¡Ñ• ‰¹•Ñ}µ½Ù•µ•¹Ðˆ¤(€€€¡¥Ð€ôÝ•¥¡Ñ• ‰‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”ˆ¤(€€€‰•¹¡µ…É¬€ôÝ•¥¡Ñ• ‰‰•¹¡µ…É¬ˆ¤(€€€¥µÁÉ½Ù•µ•¹Ð€ôÝ•¥¡Ñ• ‰‰•¹¡µ…É­}¥µÁÉ½Ù•µ•¹Ðˆ¤(€€€ÍÑ…¹‘…É‘}•ÉÉ½È€ôÝ•¥¡Ñ• ‰ÍÑ…¹‘…É‘}•ÉÉ½Èˆ¤(€€€ÕÍ…‰±•}Ý¥¹‘½ÝÌ€ôÑÕÁ±”¡¥Ñ•´™½È¥Ñ•´¥¸Ù…±Õ•Ì¥˜¥Ñ•´¹½‰Í•ÉÙ…Ñ¥½¹Ì¤(€€€ÍÑ…‰±”€ô€ (€€€€€€€±•¸¡ÕÍ…‰±•}Ý¥¹‘½ÝÌ¤€øô5%9%5U5}=9%I5Q%=9}]%9=]L(€€€€€€€…¹…±°¡¥Ñ•´¹ÍÑ…‰±”™½È¥Ñ•´¥¸ÕÍ…‰±•}Ý¥¹‘½ÝÌ¤(€€€€€€€…¹…±° (€€€€€€€€€€€€¡¥Ñ•´¹…±¥¹•‘}É½ÍÍ}µ½Ù•µ•¹Ð½È•¥µ…° À¤¤€ø€À™½È¥Ñ•´¥¸ÕÍ…‰±•}Ý¥¹‘½ÝÌ(€€€€€€€€¤(€€€€¤(€€€•½¹½µ¥Œ€ô¹•Ð¥Ì¹½Ð9½¹”…¹¹•Ð€øô=9=5%}5I%8(€€€Á…ÍÍ•Ì€ô€ (€€€€€€€ÍÑ…‰±”(€€€€€€€…¹•½¹½µ¥Œ(€€€€€€€…¹¥µÁÉ½Ù•µ•¹Ð¥Ì¹½Ð9½¹”(€€€€€€€…¹¥µÁÉ½Ù•µ•¹Ð€øô5%9%5U5}!%Q}%5AI=Y59P(€€€€¤(€€€É•ÑÕÉ¸A¡…Í•Ù¥‘•¹” (€€€€€€€Ñ½Ñ…°°(€€€€€€€É½ÍÌ°(€€€€€€€…±¥¹•°(€€€€€€€¹•Ð°(€€€€€€€¡¥Ð°(€€€€€€€‰•¹¡µ…É¬°(€€€€€€€¥µÁÉ½Ù•µ•¹Ð°(€€€€€€€ÍÑ…¹‘…É‘}•ÉÉ½È°(€€€€€€€ÍÑ…‰±”°(€€€€€€€•½¹½µ¥Œ°(€€€€€€€Á…ÍÍ•Ì°(€€€€¤(()‘•˜}¥¹Í¥‘”¡Ù…±Õ”è•¥µ…°ð9½¹”°É•¥µ”èÉ½é•¹I•¥µ”¤€´ø‰½½°è(€€€¥˜Ù…±Õ”¥Ì9½¹”è(€€€€€€€É•ÑÕÉ¸…±Í”(€€€É•ÑÕÉ¸€¡É•¥µ”¹±½Ý•É}‰½Õ¹¥Ì9½¹”½ÈÙ…±Õ”€øôÉ•¥µ”¹±½Ý•É}‰½Õ¹¤…¹€ (€€€€€€€É•¥µ”¹ÕÁÁ•É}‰½Õ¹¥Ì9½¹”½ÈÙ…±Õ”€ðôÉ•¥µ”¹ÕÁÁ•É}‰½Õ¹(€€€€¤(()‘•˜}‘¥Ù•ÉÍ¥™¥•‘}¥¹Ñ•É…Ñ¥½¹Ì (€€€Ù…±Õ•ÌèÑÕÁ±•mI•¥µ•½¹‘¥Ñ¥½¹•‘I•±…Ñ¥½¹Í¡¥À°€¸¸¹t°(¤€´øÑÕÁ±•mI•¥µ•½¹‘¥Ñ¥½¹•‘I•±…Ñ¥½¹Í¡¥À°€¸¸¹tè(€€€‰Õ­•ÑÌè‘¥ÑmÑÕÁ±•mÍÑÈ°ÍÑÉt°±¥ÍÑmI•¥µ•½¹‘¥Ñ¥½¹•‘I•±…Ñ¥½¹Í¡¥Áut€ôíô(€€€™½ÈÙ…±Õ”¥¸Í½ÉÑ• (€€€€€€€Ù…±Õ•Ì°(€€€€€€€­•äõ±…µ‰‘„¥Ñ•´è€ (€€€€€€€€€€€€µ¥Ñ•´¹½¹Ñ•áÑ}¥µÁÉ½Ù•µ•¹Ð°(€€€€€€€€€€€€´¡¥Ñ•´¹ÑÉ…¥¹¥¹œ¹¹•Ñ}µ½Ù•µ•¹Ð½È•¥µ…° À¤¤°(€€€€€€€€€€€¥Ñ•´¹™¥¹•ÉÁÉ¥¹Ð°(€€€€€€€€¤°(€€€€¤è(€€€€€€€‰Õ­•ÑÌ¹Í•Ñ‘•™…Õ±Ð (€€€€€€€€€€€€¡Ù…±Õ”¹‘¥É•Ñ¥½¹…±}ÁÉ•‘¥Ñ½È°Ù…±Õ”¹½¹Ñ•áÑ}ÁÉ•‘¥Ñ½È¤°mt(€€€€€€€€¤¹…ÁÁ•¹¡Ù…±Õ”¤(€€€É•ÍÕ±Ðè±¥ÍÑmI•¥µ•½¹‘¥Ñ¥½¹•‘I•±…Ñ¥½¹Í¡¥Át€ômt(€€€Ý¡¥±”±•¸¡É•ÍÕ±Ð¤€ð%9QIQ%=9}IQ9Q%=9}1%5%P…¹…¹ä¡‰Õ­•ÑÌ¹Ù…±Õ•Ì ¤¤è(€€€€€€€™½È­•ä¥¸Í½ÉÑ•¡‰Õ­•ÑÌ¤è(€€€€€€€€€€€¥˜‰Õ­•ÑÍm­•åt…¹±•¸¡É•ÍÕ±Ð¤€ð%9QIQ%=9}IQ9Q%=9}1%5%Pè(€€€€€€€€€€€€€€€É•ÍÕ±Ð¹…ÁÁ•¹¡‰Õ­•ÑÍm­•åt¹Á½À À¤¤(€€€É•ÑÕÉ¸ÑÕÁ±”¡É•ÍÕ±Ð¤(()‘•˜}…¹‘¥‘…Ñ•}É…¹¬¡Ù…±Õ”èXÐÅ…¹‘¥‘…Ñ”¤€´øÑÕÁ±•m½‰©•Ð°€¸¸¹tè(€€€É•ÑÕÉ¸€ (€€€€€€€€´¡Ù…±Õ”¹Ù…±¥‘…Ñ¥½¸¹¹•Ñ}µ½Ù•µ•¹Ð½È•¥µ…° À¤¤°(€€€€€€€€´¡Ù…±Õ”¹Ù…±¥‘…Ñ¥½¸¹‰•¹¡µ…É­}¥µÁÉ½Ù•µ•¹Ð½È•¥µ…° À¤¤°(€€€€€€€Ù…±Õ”¹É•±…Ñ¥½¹Í¡¥À¹™¥¹•ÉÁÉ¥¹Ð°(€€€€¤(()‘•˜}É•©•Ñ¥½¸ (€€€•Ù¥‘•¹”èA¡…Í•Ù¥‘•¹”°É•±…Ñ¥½¹Í¡¥ÀèI•¥µ•½¹‘¥Ñ¥½¹•‘I•±…Ñ¥½¹Í¡¥À(¤€´øÍÑÈè(€€€¥˜É•±…Ñ¥½¹Í¡¥À¹½¹Ñ•áÑ}¥µÁÉ½Ù•µ•¹Ð€ðô€Àè(€€€€€€€É•ÑÕÉ¸XÐÅ½¹±ÕÍ¥½¸¹%IQ%=91}M%91}9=Q}%5AI=Y}	e}=9QaP¹Ù…±Õ”(€€€¥˜¹½Ð•Ù¥‘•¹”¹•½¹½µ¥Œè(€€€€€€€É•ÑÕÉ¸XÐÅ½¹±ÕÍ¥½¸¹=9QaQ}%5AI=YM}M%91}	UQ}9=Q}=9=5%¹Ù…±Õ”(€€€¥˜¹½Ð•Ù¥‘•¹”¹ÍÑ…‰±”è(€€€€€€€É•ÑÕÉ¸XÐÅ½¹±ÕÍ¥½¸¹=9=5%}%9QIQ%=9}	UQ}U9MQ	1¹Ù…±Õ”(€€€É•ÑÕÉ¸XÐÅ½¹±ÕÍ¥½¸¹MQ	1}%9QIQ%=9}	1=]}	9!5I,¹Ù…±Õ”(()‘•˜}½¹±ÕÍ¥½¸ (€€€…¹‘¥‘…Ñ•ÌèÑÕÁ±•mXÐÅ…¹‘¥‘…Ñ”°€¸¸¹t°½¹™¥Éµ•èÑÕÁ±•mXÐÅ…¹‘¥‘…Ñ”°€¸¸¹t(¤€´øXÐÅ½¹±ÕÍ¥½¸è(€€€¥˜½¹™¥Éµ•è(€€€€€€€É•ÑÕÉ¸XÐÅ½¹±ÕÍ¥½¸¹I%5}=9%Q%=9}}=9%I5(€€€É•…Í½¹Ì€ôí¥Ñ•´¹É•©•Ñ¥½¹}É•…Í½¸™½È¥Ñ•´¥¸…¹‘¥‘…Ñ•Íô(€€€™½ÈÙ…±Õ”¥¸€ (€€€€€€€XÐÅ½¹±ÕÍ¥½¸¹=9QaQ}%5AI=YM}M%91}	UQ}9=Q}=9=5%°(€€€€€€€XÐÅ½¹±ÕÍ¥½¸¹=9=5%}%9QIQ%=9}	UQ}U9MQ	1°(€€€€€€€XÐÅ½¹±ÕÍ¥½¸¹MQ	1}%9QIQ%=9}	1=]}	9!5I,°(€€€€€€€XÐÅ½¹±ÕÍ¥½¸¹%IQ%=91}M%91}9=Q}%5AI=Y}	e}=9QaP°(€€€€¤è(€€€€€€€¥˜Ù…±Õ”¹Ù…±Õ”¥¸É•…Í½¹Ìè(€€€€€€€€€€€É•ÑÕÉ¸Ù…±Õ”(€€€É•ÑÕÉ¸XÐÅ½¹±ÕÍ¥½¸¹9=}I%5}=9%Q%=9}(()‘•˜}‰Õ¥±‘}¡åÁ½Ñ¡•Í•Ì (€€€½¹™¥Éµ•èÑÕÁ±•mXÐÅ…¹‘¥‘…Ñ”°€¸¸¹t°(€€€Ý¥¹‘½ÝÌèÑÕÁ±•m]…±­½ÉÝ…É‘]¥¹‘½Ü°€¸¸¹t°(¤€´øÑÕÁ±•mMÑÉ…Ñ•å!åÁ½Ñ¡•Í¥Ì°€¸¸¹tè(€€€€ˆˆ‰QÉ…¹Í±…Ñ”½¹™¥Éµ•¥¹Ñ•É…Ñ¥½¹Ì¥¹Ñ¼•á•ÕÑ…‰±”°Ù•ÉÍ¥½¹•¡åÁ½Ñ¡•Í•Ì¸ˆˆˆ((€€€¡åÁ½Ñ¡•Í•Ìè±¥ÍÑmMÑÉ…Ñ•å!åÁ½Ñ¡•Í¥Ít€ômt(€€€™½È…¹‘¥‘…Ñ”¥¸½¹™¥Éµ•è(€€€€€€€É•±…Ñ¥½¹Í¡¥À€ô…¹‘¥‘…Ñ”¹É•±…Ñ¥½¹Í¡¥À(€€€€€€€Ñ•ÍÐ€ô…¹‘¥‘…Ñ”¹Ñ•ÍÐ(€€€€€€€¥˜Ñ•ÍÐ¥Ì9½¹”è(€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€¥‘•¹Ñ¥Ñä€ô}™¥¹•ÉÁÉ¥¹Ð (€€€€€€€€€€€€‰ÍÑÉ…Ñ•äµ¡åÁ½Ñ¡•Í¥Ìˆ°(€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹‘¥É•Ñ¥½¹…±}ÁÉ•‘¥Ñ½È°(€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹‘¥É•Ñ¥½¹…±}Ù•ÉÍ¥½¸°(€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹Í•±•Ñ•‘}‰¥¸°(€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹‘¥É•Ñ¥½¹…±}±½Ý•É}‰½Õ¹°(€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹‘¥É•Ñ¥½¹…±}ÕÁÁ•É}‰½Õ¹°(€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹‘¥É•Ñ¥½¹}µ…ÁÁ¥¹œ°(€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹½¹Ñ•áÑ}ÁÉ•‘¥Ñ½È°(€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹½¹Ñ•áÑ}Ù•ÉÍ¥½¸°(€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹É•¥µ”°(€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹™É½é•¹}É•¥µ”¹±½Ý•É}‰½Õ¹°(€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹™É½é•¹}É•¥µ”¹ÕÁÁ•É}‰½Õ¹°(€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹…‘•¹”°(€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹¡½É¥é½¸°(€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹…Ñ¥½¸°(€€€€€€€€¤(€€€€€€€¡åÁ½Ñ¡•Í•Ì¹…ÁÁ•¹ (€€€€€€€€€€€MÑÉ…Ñ•å!åÁ½Ñ¡•Í¥Ì (€€€€€€€€€€€€€€€¥‘•¹Ñ¥Ñä°(€€€€€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹‘¥É•Ñ¥½¹…±}ÁÉ•‘¥Ñ½È°(€€€€€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹‘¥É•Ñ¥½¹…±}Ù•ÉÍ¥½¸°(€€€€€€€€€€€€€€€AI%Q%Y}}XÐÅ}=9QIQ}YIM%=8°(€€€€€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹‘¥É•Ñ¥½¹}µ…ÁÁ¥¹œ°(€€€€€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹Í•±•Ñ•‘}‰¥¸°(€€€€€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹‘¥É•Ñ¥½¹…±}±½Ý•É}‰½Õ¹°(€€€€€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹‘¥É•Ñ¥½¹…±}ÕÁÁ•É}‰½Õ¹°(€€€€€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹…Ñ¥½¸°(€€€€€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹¡½É¥é½¸°(€€€€€€€€€€€€€€€½¹‘¥Ñ¥½¸¹10°(€€€€€€€€€€€€€€€€¡É•±…Ñ¥½¹Í¡¥À¹Í•±•Ñ•‘}‰¥¸°¤°(€€€€€€€€€€€€€€€Ñ•ÍÐ¹É½ÍÍ}µ½Ù•µ•¹Ð°(€€€€€€€€€€€€€€€Ñ•ÍÐ¹¹•Ñ}µ½Ù•µ•¹Ð(€€€€€€€€€€€€€€€¥˜É•±…Ñ¥½¹Í¡¥À¹…Ñ¥½¸¥ÌMÕ•ÍÑ•‘Ñ¥½¸¹1=9}9QId(€€€€€€€€€€€€€€€•±Í”9½¹”°(€€€€€€€€€€€€€€€Ñ•ÍÐ¹¹•Ñ}µ½Ù•µ•¹Ð(€€€€€€€€€€€€€€€¥˜É•±…Ñ¥½¹Í¡¥À¹…Ñ¥½¸¥ÌMÕ•ÍÑ•‘Ñ¥½¸¹a%Q}=I}Y=%(€€€€€€€€€€€€€€€•±Í”9½¹”°(€€€€€€€€€€€€€€€Ñ•ÍÐ¹‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”°(€€€€€€€€€€€€€€€•¥µ…°¡±•¸¡Ý¥¹‘½ÝÌ¤¤€¼•¥µ…°¡µ…à Ä°±•¸¡Ý¥¹‘½ÝÌ¤¤¤°(€€€€€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹…‘•¹”°(€€€€€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹¡½É¥é½¸°(€€€€€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹¡½É¥é½¸°(€€€€€€€€€€€€€€€ÑÕÁ±”¡Ý¥¹‘½Ü¹½É‘¥¹…°™½ÈÝ¥¹‘½Ü¥¸Ý¥¹‘½ÝÌ¤°(€€€€€€€€€€€€€€€Ù¥‘•¹•MÑ…ÑÕÌ¹AI=5%M%9}A=M%Q%Y(€€€€€€€€€€€€€€€¥˜É•±…Ñ¥½¹Í¡¥À¹‘¥É•Ñ¥½¹}µ…ÁÁ¥¹œ¥Ì¥É•Ñ¥½¹5…ÁÁ¥¹œ¹A=M%Q%Y(€€€€€€€€€€€€€€€•±Í”Ù¥‘•¹•MÑ…ÑÕÌ¹AI=5%M%9}9Q%Y°(€€€€€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€€€€€‰½¹™¥Éµ•¥¸Ù…±¥‘…Ñ¥½¸…¹¥Í½±…Ñ•Ñ•ÍÐ…É½ÍÌ…Ð±•…ÍÐÑ¡É•”€ˆ(€€€€€€€€€€€€€€€€€€€€‰Ý…±¬µ™½ÉÝ…ÉÝ¥¹‘½ÝÌÕÍ¥¹œ„ÑÉ…¥¹¥¹œµ™É½é•¸‘¥É•Ñ¥½¹…°‰¥¸…¹€ˆ(€€€€€€€€€€€€€€€€€€€€‰½¹Ñ•áÐÉ•¥µ”¸ˆ(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€€€€€‰I•Í•…É •Ù¥‘•¹”½¹±äì¥Ð‘½•Ì¹½Ð…Ñ¥Ù…Ñ”„MÑÉ…Ñ•ä€ˆ(€€€€€€€€€€€€€€€€€€€€‰…ÕÑ½µ…Ñ¥…±±ä¸ˆ(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹½¹Ñ•áÑ}ÁÉ•‘¥Ñ½È°(€€€€€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹½¹Ñ•áÑ}Ù•ÉÍ¥½¸°(€€€€€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹É•¥µ”¹Ù…±Õ”°(€€€€€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹™É½é•¹}É•¥µ”¹±½Ý•É}‰½Õ¹°(€€€€€€€€€€€€€€€É•±…Ñ¥½¹Í¡¥À¹™É½é•¹}É•¥µ”¹ÕÁÁ•É}‰½Õ¹°(€€€€€€€€€€€€¤(€€€€€€€€¤(€€€É•ÑÕÉ¸ÑÕÁ±”¡Í½ÉÑ•¡¡åÁ½Ñ¡•Í•Ì°­•äõ±…µ‰‘„¥Ñ•´è¥Ñ•´¹¡åÁ½Ñ¡•Í¥Í}¥¤¤(()‘•˜}ÁÉ•‘¥Ñ½É}Ù•ÉÍ¥½¸¡¹…µ”èÍÑÈ¤€´ø¥¹Ðè(€€€É•ÑÕÉ¸€ (€€€€€€€I%Y}AI%Q=I}YIM%=8(€€€€€€€¥˜…¹ä¡¥Ñ•´¹¹…µ”€ôô¹…µ”™½È¥Ñ•´¥¸I%Y}AI%Q=IL¤(€€€€€€€•±Í”€Ì(€€€€¤(()‘•˜}…¡•}­•ä (€€€µ…ÑÉ¥àè!¥ÍÑ½É¥…±•…ÑÕÉ•5…ÑÉ¥à°(€€€Ý¥¹‘½ÝÌèÑÕÁ±•m]…±­½ÉÝ…É‘]¥¹‘½Ü°€¸¸¹t°(€€€‘¥É•Ñ¥½¹…°èÍÑÈ°(€€€½¹Ñ•áÐèÍÑÈ°(€€€É•¥µ”èÉ½é•¹I•¥µ”°(€€€Á¡…Í”èÍÑÈ°(€€€Í¥µÕ±…Ñ¥½¸èM¥µÕ±…Ñ¥½¹½¹™¥ÕÉ…Ñ¥½¸°(€€€…‘•¹”èÑ¥µ•‘•±Ñ„°(€€€¡½É¥é½¸èÑ¥µ•‘•±Ñ„°(€€€‘¥É•Ñ¥½¹…±}±½Ý•É}‰½Õ¹è•¥µ…°ð9½¹”°(€€€‘¥É•Ñ¥½¹…±}ÕÁÁ•É}‰½Õ¹è•¥µ…°ð9½¹”°(€€€…Ñ¥½¸èMÕ•ÍÑ•‘Ñ¥½¸°(¤€´øÍÑÈè(€€€Í±¥•Ì€ôÑÕÁ±” (€€€€€€€€ (€€€€€€€€€€€•Ñ…ÑÑÈ¡Ý¥¹‘½Ü°˜‰íÁ¡…Í•õ}™É½´ˆ¤°(€€€€€€€€€€€•Ñ…ÑÑÈ¡Ý¥¹‘½Ü°˜‰íÁ¡…Í•õ}Ñ¼ˆ¤°(€€€€€€€€€€€}Í±¥•}™¥¹•ÉÁÉ¥¹Ð (€€€€€€€€€€€€€€€µ…ÑÉ¥à°•Ñ…ÑÑÈ¡Ý¥¹‘½Ü°˜‰íÁ¡…Í•õ}™É½´ˆ¤°•Ñ…ÑÑÈ¡Ý¥¹‘½Ü°˜‰íÁ¡…Í•õ}Ñ¼ˆ¤(€€€€€€€€€€€€¤°(€€€€€€€€¤(€€€€€€€™½ÈÝ¥¹‘½Ü¥¸Ý¥¹‘½ÝÌ(€€€€¤(€€€É•ÑÕÉ¸}™¥¹•ÉÁÉ¥¹Ð (€€€€€€€€‰…¡”ˆ°(€€€€€€€‘¥É•Ñ¥½¹…°°(€€€€€€€}ÁÉ•‘¥Ñ½É}Ù•ÉÍ¥½¸¡‘¥É•Ñ¥½¹…°¤°(€€€€€€€½¹Ñ•áÐ°(€€€€€€€}ÁÉ•‘¥Ñ½É}Ù•ÉÍ¥½¸¡½¹Ñ•áÐ¤°(€€€€€€€Á¡…Í”°(€€€€€€€É•¥µ”¹™¥¹•ÉÁÉ¥¹Ð°(€€€€€€€…‘•¹”°(€€€€€€€¡½É¥é½¸°(€€€€€€€‘¥É•Ñ¥½¹…±}±½Ý•É}‰½Õ¹°(€€€€€€€‘¥É•Ñ¥½¹…±}ÕÁÁ•É}‰½Õ¹°(€€€€€€€…Ñ¥½¸°(€€€€€€€Í±¥•Ì°(€€€€€€€Í¥µÕ±…Ñ¥½¸¹™••}É…Ñ”°(€€€€€€€Í¥µÕ±…Ñ¥½¸¹Í±¥ÁÁ…•}É…Ñ”°(€€€€¤(()‘•˜}Í±¥•}™¥¹•ÉÁÉ¥¹Ð (€€€µ…ÑÉ¥àè!¥ÍÑ½É¥…±•…ÑÕÉ•5…ÑÉ¥à°ÍÑ…ÉÐè‘…Ñ•Ñ¥µ”°•¹è‘…Ñ•Ñ¥µ”(¤€´øÍÑÈè(€€€É•ÑÕÉ¸}™¥¹•ÉÁÉ¥¹Ð (€€€€€€€ÑÕÁ±”¡É½Ü™½ÈÉ½Ü¥¸µ…ÑÉ¥à¹É½ÝÌ¥˜ÍÑ…ÉÐ€ðôÉ½Ü¹…¹‘±•}½Á•¹}Ñ¥µ”€ð•¹¤(€€€€¤(()‘•˜}™¥¹•ÉÁÉ¥¹Ð ©Ù…±Õ•Ìè½‰©•Ð¤€´øÍÑÈè(€€€•¹½‘•€ô©Í½¸¹‘ÕµÁÌ (€€€€€€€€¡AI%Q%Y}}XÐÅ}=9QIQ}YIM%=8°Ù…±Õ•Ì¤°(€€€€€€€‘•™…Õ±ÐõÍÑÈ°(€€€€€€€Í½ÉÑ}­•åÌõQÉÕ”°(€€€€€€€Í•Á…É…Ñ½ÉÌô ˆ°ˆ°€ˆèˆ¤°(€€€€¤(€€€É•ÑÕÉ¸¡…Í¡±¥ˆ¹Í¡„ÈÔØ¡•¹½‘•¹•¹½‘” ¤¤¹¡•á‘¥•ÍÐ ¤(