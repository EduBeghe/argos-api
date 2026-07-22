"""Real bounded Predictive Edge Discovery v4 runtime.

The three stages are intentionally separate functions.  Training selects relationships,
validation freezes and ranks configurations, and isolated test only confirms the
already-frozen identities.
"""

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import timedelta
from decimal import Decimal
from enum import Enum
from math import sqrt
from statistics import median
from time import perf_counter

from app.optimization.feature_matrix import HistoricalFeatureMatrix
from app.optimization.predictive_edge_v4 import (
    CHEAP_RELATIONSHIP_LIMIT,
    DISCOVERY_BIN_COUNTS,
    DISCOVERY_CADENCES,
    DISCOVERY_HORIZONS,
    ECONOMIC_EDGE_MARGIN,
    ISOLATED_TEST_CONFIGURATION_LIMIT,
    NEAR_MISS_LIMIT,
    PREDICTIVE_EDGE_V4_CONTRACT_VERSION,
    SENSITIVITY_LIMIT,
    VALIDATION_CONFIGURATION_LIMIT,
    AuditedParameter,
    DiagnosticConclusion,
    EdgeDiagnostic,
    EvidenceProfile,
    NeutralBandPolicy,
    ParameterSource,
    RankedNearMiss,
    RedundancyEvidence,
    RejectionReason,
    RequirementMargin,
    ThresholdSensitivity,
    diagnostic_conclusion,
    neutral_threshold_for_training,
    rank_near_misses,
    requirement_margin,
    threshold_sensitivity,
)
from app.optimization.predictive_edges import (
    HYPOTHESIS_CONTRACT_VERSION,
    PREDICTORS,
    Condition,
    DirectionMapping,
    EdgeMetrics,
    EvidenceStatus,
    EvidenceThresholds,
    Observation,
    PredictorDefinition,
    PredictorSemanticKind,
    StrategyHypothesis,
    SuggestedAction,
    _correlation,
    _learn_direction_mapping,
    _metrics,
    _observations,
    _phase_rows,
    _predictors,
    _quantile_bounds,
    predictive_edge_volatility_threshold,
)
from app.optimization.walk_forward import WalkForwardWindow
from app.paper.config import SimulationConfiguration


@dataclass(frozen=True, slots=True)
class CheapRelationship:
    predictor: PredictorDefinition
    cadence: timedelta
    horizon: timedelta
    observation_count: int
    mean_absolute_correlation: Decimal
    score: Decimal
    redundancy_group: str
    fingerprint: str
    reused: bool = False


@dataclass(frozen=True, slots=True)
class ThresholdCalibration:
    name: str
    observed_value: Decimal
    final_value: Decimal
    floor: Decimal
    ceiling: Decimal
    method: str
    source: ParameterSource


@dataclass(frozen=True, slots=True)
class FrozenConfiguration:
    fingerprint: str
    predictor: PredictorDefinition
    cadence: timedelta
    horizon: timedelta
    bins: int
    condition: Condition
    neutral_policy: NeutralBandPolicy
    profile: EvidenceProfile
    direction_mapping: DirectionMapping
    bin_bounds: tuple[Decimal, ...]
    selected_bin_ordinal: int
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    suggested_action: SuggestedAction
    neutral_threshold: Decimal
    thresholds: EvidenceThresholds
    threshold_calibrations: tuple[ThresholdCalibration, ...]
    volatility_thresholds: tuple[tuple[int, Decimal], ...]


@dataclass(frozen=True, slots=True)
class ValidationConfigurationResult:
    configuration: FrozenConfiguration
    diagnostic: EdgeDiagnostic
    status: str
    selected_for_test: bool
    benchmark: Decimal | None
    standard_error: Decimal | None
    gross_expected_movement: Decimal | None
    net_expected_movement: Decimal | None
    reused: bool = False


@dataclass(frozen=True, slots=True)
class IsolatedTestResult:
    configuration_fingerprint: str
    confirmed: bool
    observation_count: int
    directional_hit_rate: Decimal | None
    benchmark: Decimal | None
    gross_expected_movement: Decimal | None
    net_expected_movement: Decimal | None
    rejection_reason: RejectionReason | None
    reused: bool = False


@dataclass(frozen=True, slots=True)
class StageCounters:
    generated: int
    computed: int
    reused: int
    discarded: int
    retained: int


@dataclass(frozen=True, slots=True)
class PredictiveEdgeV4Result:
    cheap_relationships: tuple[CheapRelationship, ...]
    validation_configurations: tuple[ValidationConfigurationResult, ...]
    isolated_test_results: tuple[IsolatedTestResult, ...]
    confirmed_edges: tuple[ValidationConfigurationResult, ...]
    diagnostic_only_edges: tuple[ValidationConfigurationResult, ...]
    near_misses: tuple[RankedNearMiss, ...]
    sensitivity: tuple[ThresholdSensitivity, ...]
    hypotheses: tuple[StrategyHypothesis, ...]
    rejection_counts: dict[str, int]
    rejection_counts_by_predictor: dict[str, int]
    rejection_counts_by_horizon: dict[str, int]
    rejection_counts_by_condition: dict[str, int]
    rejection_counts_by_neutral_policy: dict[str, int]
    rejection_counts_by_cadence: dict[str, int]
    effective_parameters: tuple[AuditedParameter, ...]
    redundancy: tuple[RedundancyEvidence, ...]
    conclusion: DiagnosticConclusion
    bottleneck_stage: str
    strongest_observed_signal: str | None
    outcome_explanation: str
    training_counters: StageCounters
    validation_counters: StageCounters
    test_counters: StageCounters
    timings: dict[str, Decimal]
    cache_writes: dict[str, object]


def analyze_predictive_edges_v4(
    *,
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    simulation: SimulationConfiguration,
    cache_namespace: str = "uncached",
    stage_cache: Mapping[str, object] | None = None,
    force_recompute: bool = False,
) -> PredictiveEdgeV4Result:
    """Execute the bounded 120 -> 40 -> 12 discovery funnel sequentially."""

    started = perf_counter()
    cache = stage_cache or {}
    cache_writes: dict[str, object] = {}
    training_started = perf_counter()
    redundancy = _training_redundancy(matrix, windows)
    relationships, generated, training_computed, training_reused = (
        _cheap_training_relationships(
            matrix,
            windows,
            simulation,
            redundancy,
            cache_namespace,
            cache,
            cache_writes,
            force_recompute,
        )
    )
    training_elapsed = perf_counter() - training_started

    validation_started = perf_counter()
    frozen = _bounded_frozen_configurations(matrix, windows, simulation, relationships)
    validation_values: list[ValidationConfigurationResult] = []
    validation_computed = 0
    validation_reused = 0
    for frozen_configuration in frozen:
        cache_key = _fingerprint(
            cache_namespace, "validation", frozen_configuration.fingerprint
        )
        cached_validation = None if force_recompute else cache.get(cache_key)
        if isinstance(cached_validation, ValidationConfigurationResult):
            validation_values.append(
                ValidationConfigurationResult(
                    cached_validation.configuration,
                    cached_validation.diagnostic,
                    cached_validation.status,
                    cached_validation.selected_for_test,
                    cached_validation.benchmark,
                    cached_validation.standard_error,
                    cached_validation.gross_expected_movement,
                    cached_validation.net_expected_movement,
                    True,
                )
            )
            validation_reused += 1
        else:
            validation_result = _evaluate_validation(
                matrix, windows, simulation, frozen_configuration
            )
            validation_values.append(validation_result)
            cache_writes[cache_key] = validation_result
            validation_computed += 1
    validation = tuple(validation_values)
    selected = tuple(
        sorted(
            (item for item in validation if item.selected_for_test),
            key=_validation_rank,
        )[:ISOLATED_TEST_CONFIGURATION_LIMIT]
    )
    selected_ids = {item.configuration.fingerprint for item in selected}
    validation = tuple(
        ValidationConfigurationResult(
            item.configuration,
            item.diagnostic,
            item.status,
            item.configuration.fingerprint in selected_ids,
            item.benchmark,
            item.standard_error,
            item.gross_expected_movement,
            item.net_expected_movement,
            item.reused,
        )
        for item in validation
    )
    diagnostics = tuple(item.diagnostic for item in validation)
    near_misses = rank_near_misses(diagnostics, limit=NEAR_MISS_LIMIT)
    sensitivity = threshold_sensitivity(near_misses)[:SENSITIVITY_LIMIT]
    validation_elapsed = perf_counter() - validation_started

    test_started = perf_counter()
    test_values: list[IsolatedTestResult] = []
    test_computed = 0
    test_reused = 0
    for selected_validation in selected:
        cache_key = _fingerprint(
            cache_namespace,
            "isolated_test",
            selected_validation.configuration.fingerprint,
        )
        cached_test = None if force_recompute else cache.get(cache_key)
        if isinstance(cached_test, IsolatedTestResult):
            test_values.append(
                IsolatedTestResult(
                    cached_test.configuration_fingerprint,
                    cached_test.confirmed,
                    cached_test.observation_count,
                    cached_test.directional_hit_rate,
                    cached_test.benchmark,
                    cached_test.gross_expected_movement,
                    cached_test.net_expected_movement,
                    cached_test.rejection_reason,
                    True,
                )
            )
            test_reused += 1
        else:
            test_result = _evaluate_isolated_test(
                matrix, windows, simulation, selected_validation.configuration
            )
            test_values.append(test_result)
            cache_writes[cache_key] = test_result
            test_computed += 1
    test_results = tuple(test_values)
    confirmed_ids = {
        item.configuration_fingerprint for item in test_results if item.confirmed
    }
    confirmed = tuple(
        item
        for item in validation
        if item.configuration.fingerprint in confirmed_ids
        and item.configuration.suggested_action is not SuggestedAction.DIAGNOSTIC_ONLY
    )
    diagnostic_only = tuple(
        item
        for item in validation
        if item.configuration.profile is EvidenceProfile.EXPLORATORY_V4
        or item.configuration.suggested_action is SuggestedAction.DIAGNOSTIC_ONLY
    )
    hypotheses = tuple(_hypothesis(item) for item in confirmed)
    test_elapsed = perf_counter() - test_started
    conclusion = diagnostic_conclusion(diagnostics, confirmed_count=len(confirmed))
    rejection_counts = _count_rejections(validation, test_results)
    strongest = min(validation, key=_validation_rank, default=None)
    bottleneck = _bottleneck(len(relationships), validation, test_results)
    total = perf_counter() - started
    return PredictiveEdgeV4Result(
        relationships,
        validation,
        test_results,
        confirmed,
        diagnostic_only,
        near_misses,
        sensitivity,
        hypotheses,
        rejection_counts,
        _group_rejections(validation, lambda item: item.configuration.predictor.name),
        _group_rejections(
            validation,
            lambda item: str(int(item.configuration.horizon.total_seconds())),
        ),
        _group_rejections(validation, lambda item: item.configuration.condition.value),
        _group_rejections(
            validation, lambda item: item.configuration.neutral_policy.value
        ),
        _group_rejections(
            validation,
            lambda item: str(int(item.configuration.cadence.total_seconds())),
        ),
        _effective_parameters(validation, simulation),
        redundancy,
        conclusion,
        bottleneck,
        strongest.configuration.fingerprint if strongest else None,
        _outcome_explanation(conclusion, rejection_counts, bottleneck),
        StageCounters(
            generated,
            training_computed,
            training_reused,
            generated - len(relationships),
            len(relationships),
        ),
        StageCounters(
            len(frozen),
            validation_computed,
            validation_reused,
            len(frozen) - len(selected),
            len(selected),
        ),
        StageCounters(len(selected), test_computed, test_reused, 0, len(confirmed)),
        {
            "training": Decimal(str(training_elapsed)),
            "validation": Decimal(str(validation_elapsed)),
            "isolated_test": Decimal(str(test_elapsed)),
            "total": Decimal(str(total)),
        },
        cache_writes,
    )


def _cheap_training_relationships(
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    simulation: SimulationConfiguration,
    redundancy: tuple[RedundancyEvidence, ...],
    cache_namespace: str,
    cache: Mapping[str, object],
    cache_writes: dict[str, object],
    force_recompute: bool,
) -> tuple[tuple[CheapRelationship, ...], int, int, int]:
    rows = {row.candle_open_time: row for row in matrix.rows}
    group_by_predictor = _redundancy_groups(redundancy)
    candidates: list[CheapRelationship] = []
    generated = 0
    computed = 0
    reused = 0
    for predictor in PREDICTORS:
        if predictor.semantic_kind is PredictorSemanticKind.CONFIRMATION_ONLY:
            continue
        for cadence in DISCOVERY_CADENCES:
            for horizon in DISCOVERY_HORIZONS:
                if horizon % cadence:
                    continue
                generated += 1
                cache_key = _fingerprint(
                    cache_namespace,
                    "training",
                    predictor.name,
                    predictor.version,
                    cadence,
                    horizon,
                    matrix.dataset_fingerprint,
                    simulation.fee_rate,
                    simulation.slippage_rate,
                )
                cached = None if force_recompute else cache.get(cache_key)
                if isinstance(cached, CheapRelationship):
                    candidates.append(
                        CheapRelationship(
                            cached.predictor,
                            cached.cadence,
                            cached.horizon,
                            cached.observation_count,
                            cached.mean_absolute_correlation,
                            cached.score,
                            cached.redundancy_group,
                            cached.fingerprint,
                            True,
                        )
                    )
                    reused += 1
                    continue
                correlations: list[Decimal] = []
                observations = 0
                for window in wiën=¶‰žËkºwµç@¡¥ÑÌ€ôÑÕÁ±” (€€€€€€€¥Ñ•´¹‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”(€€€€€€€™½È¥Ñ•´¥¸Ý¥¹‘½Ý}µ•ÑÉ¥Ì(€€€€€€€¥˜¥Ñ•´¹‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”¥Ì¹½Ð9½¹”(€€€€¤(€€€‰•¹¡µ…É­Ì€ôÑÕÁ±” (€€€€€€€¥Ñ•´¹‘¥É•Ñ¥½¹}‰•¹¡µ…É­Ì¹ÍÑÉ½¹•ÍÐ(€€€€€€€™½È¥Ñ•´¥¸Ý¥¹‘½Ý}µ•ÑÉ¥Ì(€€€€€€€¥˜¥Ñ•´¹‘¥É•Ñ¥½¹}‰•¹¡µ…É­Ì¥Ì¹½Ð9½¹”(€€€€¤(€€€ÕÍ…‰±”€ôÍÕ´¡¥Ñ•´¹ÕÍ…‰±•}½Õ¹Ð™½È¥Ñ•´¥¸Ý¥¹‘½Ý}µ•ÑÉ¥Ì¤(€€€½ÉÉ•±…Ñ¥½¸€ô€ (€€€€€€€ÍÕ´ ¡…‰Ì¡¥Ñ•´¤™½È¥Ñ•´¥¸½ÉÉ•±…Ñ¥½¹Ì¤°•¥µ…° À¤¤(€€€€€€€€¼•¥µ…°¡±•¸¡½ÉÉ•±…Ñ¥½¹Ì¤¤(€€€€€€€¥˜½ÉÉ•±…Ñ¥½¹Ì(€€€€€€€•±Í”•¥µ…° À¤(€€€€¤(€€€¡¥Ð€ôÍÕ´¡¡¥ÑÌ°•¥µ…° À¤¤€¼•¥µ…°¡±•¸¡¡¥ÑÌ¤¤¥˜¡¥ÑÌ•±Í”•¥µ…° À¤(€€€‰•¹¡µ…É¬€ôµ…à¡‰•¹¡µ…É­Ì¤¥˜‰•¹¡µ…É­Ì•±Í”•¥µ…° À¤(€€€¥µÁÉ½Ù•µ•¹Ð€ô¡¥Ð€´‰•¹¡µ…É¬(€€€É½ÍÌ€ô€ (€€€€€€€ÍÕ´ ¡¥Ñ•´¹™½ÉÝ…É‘}É½ÍÍ}É•ÑÕÉ¸™½È¥Ñ•´¥¸Ù…±Õ•Ì¤°•¥µ…° À¤¤(€€€€€€€€¼•¥µ…°¡±•¸¡Ù…±Õ•Ì¤¤(€€€€€€€¥˜Ù…±Õ•Ì(€€€€€€€•±Í”•¥µ…° À¤(€€€€¤(€€€¹•Ð€ô…‰Ì¡É½ÍÌ¤€´•¥µ…° È¤€¨€¡Í¥µÕ±…Ñ¥½¸¹™••}É…Ñ”€¬Í¥µÕ±…Ñ¥½¸¹Í±¥ÁÁ…•}É…Ñ”¤(€€€ÍÑ…¹‘…É‘}•ÉÉ½ÉÌ€ôÑÕÁ±” (€€€€€€€¥Ñ•´¹ÍÑ…¹‘…É‘}•ÉÉ½È(€€€€€€€™½È¥Ñ•´¥¸Ý¥¹‘½Ý}µ•ÑÉ¥Ì(€€€€€€€¥˜¥Ñ•´¹ÍÑ…¹‘…É‘}•ÉÉ½È¥Ì¹½Ð9½¹”(€€€€¤(€€€ÍÑ…¹‘…É‘}•ÉÉ½È€ô€ (€€€€€€€ÍÕ´¡ÍÑ…¹‘…É‘}•ÉÉ½ÉÌ°•¥µ…° À¤¤€¼•¥µ…°¡±•¸¡ÍÑ…¹‘…É‘}•ÉÉ½ÉÌ¤¤(€€€€€€€¥˜ÍÑ…¹‘…É‘}•ÉÉ½ÉÌ(€€€€€€€•±Í”9½¹”(€€€€¤(€€€Í¥¹Ì€ôÑÕÁ±” Ä¥˜¥Ñ•´€ø€À•±Í”€´Ä¥˜¥Ñ•´€ð€À•±Í”€À™½È¥Ñ•´¥¸½ÉÉ•±…Ñ¥½¹Ì¤(€€€‘½µ¥¹…¹Ð€ôµ…à¡Í¥¹Ì¹½Õ¹Ð Ä¤°Í¥¹Ì¹½Õ¹Ð ´Ä¤¤¥˜Í¥¹Ì•±Í”€À(€€€ÍÑ…‰¥±¥Ñä€ô•¥µ…°¡‘½µ¥¹…¹Ð¤€¼•¥µ…°¡±•¸¡Í¥¹Ì¤¤¥˜Í¥¹Ì•±Í”•¥µ…° À¤(€€€‘¥ÍÁ•ÉÍ¥½¸€ô€ (€€€€€€€µ…à¡½ÉÉ•±…Ñ¥½¹Ì¤€´µ¥¸¡½ÉÉ•±…Ñ¥½¹Ì¤¥˜±•¸¡½ÉÉ•±…Ñ¥½¹Ì¤€ø€Ä•±Í”•¥µ…° À¤(€€€€¤(€€€Ñ¡É•Í¡½±‘Ì€ô½¹™¥ÕÉ…Ñ¥½¸¹Ñ¡É•Í¡½±‘Ì(€€€µ…É¥¹Ì€ô€ (€€€€€€€É•ÅÕ¥É•µ•¹Ñ}µ…É¥¸ (€€€€€€€€€€€I•©•Ñ¥½¹I•…Í½¸¹%9MU%%9Q}=	MIYQ%=9L°(€€€€€€€€€€€•¥µ…°¡ÕÍ…‰±”¤°(€€€€€€€€€€€•¥µ…°¡Ñ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}½‰Í•ÉÙ…Ñ¥½¹Ì¤°(€€€€€€€€¤°(€€€€€€€É•ÅÕ¥É•µ•¹Ñ}µ…É¥¸ (€€€€€€€€€€€I•©•Ñ¥½¹I•…Í½¸¹%9MU%%9Q}]%9=]L°(€€€€€€€€€€€•¥µ…°¡±•¸¡Ý¥¹‘½Ý}µ•ÑÉ¥Ì¤¤°(€€€€€€€€€€€•¥µ…°¡Ñ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}Ý¥¹‘½ÝÌ¤°(€€€€€€€€¤°(€€€€€€€É•ÅÕ¥É•µ•¹Ñ}µ…É¥¸ (€€€€€€€€€€€I•©•Ñ¥½¹I•…Í½¸¹U9MQ	1}%IQ%=8°(€€€€€€€€€€€ÍÑ…‰¥±¥Ñä°(€€€€€€€€€€€Ñ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}ÍÑ…‰±•}‘¥É•Ñ¥½¹}É…Ñ¥¼°(€€€€€€€€¤°(€€€€€€€É•ÅÕ¥É•µ•¹Ñ}µ…É¥¸ (€€€€€€€€€€€I•©•Ñ¥½¹I•…Í½¸¹=II1Q%=9}	1=]}Q!IM!=1°(€€€€€€€€€€€½ÉÉ•±…Ñ¥½¸°(€€€€€€€€€€€Ñ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}…‰Í½±ÕÑ•}½ÉÉ•±…Ñ¥½¸°(€€€€€€€€¤°(€€€€€€€É•ÅÕ¥É•µ•¹Ñ}µ…É¥¸ (€€€€€€€€€€€I•©•Ñ¥½¹I•…Í½¸¹!%Q}IQ}	1=]}Q!IM!=1°(€€€€€€€€€€€¡¥Ð°(€€€€€€€€€€€Ñ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”°(€€€€€€€€¤°(€€€€€€€É•ÅÕ¥É•µ•¹Ñ}µ…É¥¸ (€€€€€€€€€€€I•©•Ñ¥½¹I•…Í½¸¹	9!5I-}%5AI=Y59Q}	1=]}Q!IM!=1°(€€€€€€€€€€€¥µÁÉ½Ù•µ•¹Ð°(€€€€€€€€€€€Ñ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}¡¥Ñ}É…Ñ•}¥µÁÉ½Ù•µ•¹Ñ}½Ù•É}‰•¹¡µ…É¬°(€€€€€€€€¤°(€€€€€€€É•ÅÕ¥É•µ•¹Ñ}µ…É¥¸ (€€€€€€€€€€€I•©•Ñ¥½¹I•…Í½¸¹5=Y59Q}=M}9=Q}=YI}=MQL°¹•Ð°=9=5%}}5I%8(€€€€€€€€¤°(€€€€€€€É•ÅÕ¥É•µ•¹Ñ}µ…É¥¸ (€€€€€€€€€€€I•©•Ñ¥½¹I•…Í½¸¹aMM%Y}%MAIM%=8°(€€€€€€€€€€€•¥µ…° ˆÀ¸ÔÀˆ¤€´‘¥ÍÁ•ÉÍ¥½¸°(€€€€€€€€€€€•¥µ…° À¤°(€€€€€€€€¤°(€€€€€€€É•ÅÕ¥É•µ•¹Ñ}µ…É¥¸ (€€€€€€€€€€€I•©•Ñ¥½¹I•…Í½¸¹%9=9M%MQ9Q}M%8°(€€€€€€€€€€€ÍÑ…‰¥±¥Ñä°(€€€€€€€€€€€Ñ¡É•Í¡½±‘Ì¹µ¥¹¥µÕµ}ÍÑ…‰±•}‘¥É•Ñ¥½¹}É…Ñ¥¼°(€€€€€€€€¤°(€€€€¤(€€€™…¥±•€ôÑÕÁ±”¡¥Ñ•´¹É•ÅÕ¥É•µ•¹Ð™½È¥Ñ•´¥¸µ…É¥¹Ì¥˜¹½Ð¥Ñ•´¹Á…ÍÍ•¤(€€€Á…ÍÍ•€ôÑÕÁ±”¡¥Ñ•´¹É•ÅÕ¥É•µ•¹Ð™½È¥Ñ•´¥¸µ…É¥¹Ì¥˜¥Ñ•´¹Á…ÍÍ•¤(€€€É•ÑÕÉ¸‘•¥…¹½ÍÑ¥Œ (€€€€€€€½¹™¥ÕÉ…Ñ¥½¸¹™¥¹•ÉÁÉ¥¹Ð°(€€€€€€€€‰ÁÉ½µ¥Í¥¹œˆ¥˜¹½Ð™…¥±••±Í”€‰¹•…É}µ¥ÍÌˆ°(€€€€€€€™…¥±•°(€€€€€€€Á…ÍÍ•°(€€€€€€€™…¥±•‘lÁt¥˜™…¥±••±Í”9½¹”°(€€€€€€€µ…É¥¹Ì°(€€€€€€€ÕÍ…‰±”°(€€€€€€€±•¸¡Ý¥¹‘½Ý}µ•ÑÉ¥Ì¤°(€€€€€€€‘½µ¥¹…¹Ð°(€€€€€€€ÍÑ…‰¥±¥Ñä°(€€€€€€€½ÉÉ•±…Ñ¥½¸¥˜½ÉÉ•±…Ñ¥½¹Ì•±Í”9½¹”°(€€€€€€€µ•‘¥…¸¡½ÉÉ•±…Ñ¥½¹Ì¤¥˜½ÉÉ•±…Ñ¥½¹Ì•±Í”9½¹”°(€€€€€€€µ¥¸¡½ÉÉ•±…Ñ¥½¹Ì¤¥˜½ÉÉ•±…Ñ¥½¹Ì•±Í”9½¹”°(€€€€€€€¡¥Ð¥˜¡¥ÑÌ•±Í”9½¹”°(€€€€€€€‰•¹¡µ…É¬¥˜‰•¹¡µ…É­Ì•±Í”9½¹”°(€€€€€€€¥µÁÉ½Ù•µ•¹Ð¥˜‰•¹¡µ…É­Ì…¹¡¥ÑÌ•±Í”9½¹”°(€€€€€€€É½ÍÌ¥˜Ù…±Õ•Ì•±Í”9½¹”°(€€€€€€€¹•Ð¥˜Ù…±Õ•Ì•±Í”9½¹”°(€€€€€€€ÍÑ…¹‘…É‘}•ÉÉ½È°(€€€€€€€‘¥ÍÁ•ÉÍ¥½¸¥˜½ÉÉ•±…Ñ¥½¹Ì•±Í”9½¹”°(€€€€¤(()‘•˜}•™™•Ñ¥Ù•}Á…É…µ•Ñ•ÉÌ (€€€Ù…±¥‘…Ñ¥½¸èÑÕÁ±•mY…±¥‘…Ñ¥½¹½¹™¥ÕÉ…Ñ¥½¹I•ÍÕ±Ð°€¸¸¹t°(€€€Í¥µÕ±…Ñ¥½¸èM¥µÕ±…Ñ¥½¹½¹™¥ÕÉ…Ñ¥½¸°(¤€´øÑÕÁ±•mÕ‘¥Ñ•‘A…É…µ•Ñ•È°€¸¸¹tè(€€€Ù…±Õ•Ìè±¥ÍÑmÕ‘¥Ñ•‘A…É…µ•Ñ•Ét€ôl(€€€€€€€Õ‘¥Ñ•‘A…É…µ•Ñ•È (€€€€€€€€€€€€‰™••}É…Ñ”ˆ°(€€€€€€€€€€€ÍÑÈ¡Í¥µÕ±…Ñ¥½¸¹™••}É…Ñ”¤°(€€€€€€€€€€€A…É…µ•Ñ•ÉM½ÕÉ”¹%a}U1P°(€€€€€€€€€€€€‰Í¥µÕ±…Ñ¥½¸½ÍÐ…ÍÍÕµÁÑ¥½¸ˆ°(€€€€€€€€¤°(€€€€€€€Õ‘¥Ñ•‘A…É…µ•Ñ•È (€€€€€€€€€€€€‰Í±¥ÁÁ…•}É…Ñ”ˆ°(€€€€€€€€€€€ÍÑÈ¡Í¥µÕ±…Ñ¥½¸¹Í±¥ÁÁ…•}É…Ñ”¤°(€€€€€€€€€€€A…É…µ•Ñ•ÉM½ÕÉ”¹%a}U1P°(€€€€€€€€€€€€‰Í¥µÕ±…Ñ¥½¸½ÍÐ…ÍÍÕµÁÑ¥½¸ˆ°(€€€€€€€€¤°(€€€€€€€Õ‘¥Ñ•‘A…É…µ•Ñ•È (€€€€€€€€€€€€‰¡•…Á}É•±…Ñ¥½¹Í¡¥Á}±¥µ¥Ðˆ°(€€€€€€€€€€€ÍÑÈ¡!A}I1Q%=9M!%A}1%5%P¤°(€€€€€€€€€€€A…É…µ•Ñ•ÉM½ÕÉ”¹%a}U1P°(€€€€€€€€€€€€‰‰½Õ¹‘•ÑÉ…¥¹¥¹œ™Õ¹¹•°ˆ°(€€€€€€€€¤°(€€€€€€€Õ‘¥Ñ•‘A…É…µ•Ñ•È (€€€€€€€€€€€€‰Ù…±¥‘…Ñ¥½¹}½¹™¥ÕÉ…Ñ¥½¹}±¥µ¥Ðˆ°(€€€€€€€€€€€ÍÑÈ¡Y1%Q%=9}=9%UIQ%=9}1%5%P¤°(€€€€€€€€€€€A…É…µ•Ñ•ÉM½ÕÉ”¹%a}U1P°(€€€€€€€€€€€€‰‰½Õ¹‘•Ù…±¥‘…Ñ¥½¸™Õ¹¹•°ˆ°(€€€€€€€€¤°(€€€€€€€Õ‘¥Ñ•‘A…É…µ•Ñ•È (€€€€€€€€€€€€‰¥Í½±…Ñ•‘}Ñ•ÍÑ}½¹™¥ÕÉ…Ñ¥½¹}±¥µ¥Ðˆ°(€€€€€€€€€€€ÍÑÈ¡%M=1Q}QMQ}=9%UIQ%=9}1%5%P¤°(€€€€€€€€€€€A…É…µ•Ñ•ÉM½ÕÉ”¹%a}U1P°(€€€€€€€€€€€€‰¥Í½±…Ñ•Ñ•ÍÐ‰Õ‘•Ðˆ°(€€€€€€€€¤°(€€€t(€€€™½ÈÉ•ÍÕ±Ð¥¸Ù…±¥‘…Ñ¥½¸è(€€€€€€€½¹™¥ÕÉ…Ñ¥½¸€ôÉ•ÍÕ±Ð¹½¹™¥ÕÉ…Ñ¥½¸(€€€€€€€ÁÉ•™¥à€ô½¹™¥ÕÉ…Ñ¥½¸¹™¥¹•ÉÁÉ¥¹Ð(€€€€€€€™½È¹…µ”°Ù…±Õ”°Í½ÕÉ”°É…Ñ¥½¹…±”¥¸€ (€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€‰…‘•¹”ˆ°(€€€€€€€€€€€€€€€½¹™¥ÕÉ…Ñ¥½¸¹…‘•¹”°(€€€€€€€€€€€€€€€A…É…µ•Ñ•ÉM½ÕÉ”¹QI%9%9}1%	IQ°(€€€€€€€€€€€€€€€€‰Í•±•Ñ•‰äÑÉ…¥¹¥¹œµ½¹±äÍÉ••¹¥¹œ…¹™É½é•¸‰•™½É”Ù…±¥‘…Ñ¥½¸ˆ°(€€€€€€€€€€€€¤°(€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€‰¡½É¥é½¸ˆ°(€€€€€€€€€€€€€€€½¹™¥ÕÉ…Ñ¥½¸¹¡½É¥é½¸°(€€€€€€€€€€€€€€€A…É…µ•Ñ•ÉM½ÕÉ”¹QI%9%9}1%	IQ°(€€€€€€€€€€€€€€€€‰Í•±•Ñ•‰äÑÉ…¥¹¥¹œµ½¹±äÍÉ••¹¥¹œ…¹™É½é•¸‰•™½É”Ù…±¥‘…Ñ¥½¸ˆ°(€€€€€€€€€€€€¤°(€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€‰‰¥¹Ìˆ°(€€€€€€€€€€€€€€€½¹™¥ÕÉ…Ñ¥½¸¹‰¥¹Ì°(€€€€€€€€€€€€€€€A…É…µ•Ñ•ÉM½ÕÉ”¹%a}U1P°(€€€€€€€€€€€€€€€€‰Ù•ÉÍ¥½¹•‰½Õ¹‘••áÁ…¹Í¥½¸ˆ°(€€€€€€€€€€€€¤°(€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€‰½¹‘¥Ñ¥½¸ˆ°(€€€€€€€€€€€€€€€½¹™¥ÕÉ…Ñ¥½¸¹½¹‘¥Ñ¥½¸¹Ù…±Õ”°(€€€€€€€€€€€€€€€A…É…µ•Ñ•ÉM½ÕÉ”¹%a}U1P°(€€€€€€€€€€€€€€€€‰•á¥ÍÑ¥¹œ½¹‘¥Ñ¥½¸½¹ÑÉ…Ðˆ°(€€€€€€€€€€€€¤°(€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€‰¹•ÕÑÉ…±}Á½±¥äˆ°(€€€€€€€€€€€€€€€½¹™¥ÕÉ…Ñ¥½¸¹¹•ÕÑÉ…±}Á½±¥ä¹Ù…±Õ”°(€€€€€€€€€€€€€€€A…É…µ•Ñ•ÉM½ÕÉ”¹%a}U1P°(€€€€€€€€€€€€€€€€‰Ù•ÉÍ¥½¹•¹•ÕÑÉ…°Á½±¥äˆ°(€€€€€€€€€€€€¤°(€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€‰¹•ÕÑÉ…±}Ñ¡É•Í¡½±ˆ°(€€€€€€€€€€€€€€€½¹™¥ÕÉ…Ñ¥½¸¹¹•ÕÑÉ…±}Ñ¡É•Í¡½±°(€€€€€€€€€€€€€€€A…É…µ•Ñ•ÉM½ÕÉ”¹QI%9%9}1%	IQ(€€€€€€€€€€€€€€€¥˜½¹™¥ÕÉ…Ñ¥½¸¹¹•ÕÑÉ…±}Á½±¥ä¥Ì9•ÕÑÉ…±	…¹‘A½±¥ä¹QI%9%9}9=%M(€€€€€€€€€€€€€€€•±Í”A…É…µ•Ñ•ÉM½ÕÉ”¹%a}U1P°(€€€€€€€€€€€€€€€€‰™É½é•¸‰•™½É”Ù…±¥‘…Ñ¥½¸ˆ°(€€€€€€€€€€€€¤°(€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€‰ÁÉ½™¥±”ˆ°(€€€€€€€€€€€€€€€½¹™¥ÕÉ…Ñ¥½¸¹ÁÉ½™¥±”¹Ù…±Õ”°(€€€€€€€€€€€€€€€A…É…µ•Ñ•ÉM½ÕÉ”¹%a}U1P°(€€€€€€€€€€€€€€€€‰Ù•ÉÍ¥½¹•É•Í•…É ÁÉ½™¥±”ˆ°(€€€€€€€€€€€€¤°(€€€€€€€€¤è(€€€€€€€€€€€Ù…±Õ•Ì¹…ÁÁ•¹ (€€€€€€€€€€€€€€€Õ‘¥Ñ•‘A…É…µ•Ñ•È¡˜‰íÁÉ•™¥áôéí¹…µ•ôˆ°ÍÑÈ¡Ù…±Õ”¤°Í½ÕÉ”°É…Ñ¥½¹…±”¤(€€€€€€€€€€€€¤(€€€€€€€™½È…±¥‰É…Ñ¥½¸¥¸½¹™¥ÕÉ…Ñ¥½¸¹Ñ¡É•Í¡½±‘}…±¥‰É…Ñ¥½¹Ìè(€€€€€€€€€€€Ù…±Õ•Ì¹…ÁÁ•¹ (€€€€€€€€€€€€€€€Õ‘¥Ñ•‘A…É…µ•Ñ•È (€€€€€€€€€€€€€€€€€€€˜‰íÁÉ•™¥áôéí…±¥‰É…Ñ¥½¸¹¹…µ•ôˆ°(€€€€€€€€€€€€€€€€€€€ÍÑÈ¡…±¥‰É…Ñ¥½¸¹™¥¹…±}Ù…±Õ”¤°(€€€€€€€€€€€€€€€€€€€…±¥‰É…Ñ¥½¸¹Í½ÕÉ”°(€€€€€€€€€€€€€€€€€€€…±¥‰É…Ñ¥½¸¹µ•Ñ¡½°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€¤(€€€É•ÑÕÉ¸ÑÕÁ±”¡Ù…±Õ•Ì¤(()‘•˜}¡åÁ½Ñ¡•Í¥Ì¡Ù…±Õ”èY…±¥‘…Ñ¥½¹½¹™¥ÕÉ…Ñ¥½¹I•ÍÕ±Ð¤€´øMÑÉ…Ñ•å!åÁ½Ñ¡•Í¥Ìè(€€€½¹™¥œ€ôÙ…±Õ”¹½¹™¥ÕÉ…Ñ¥½¸(€€€¥‘•¹Ñ¥™¥•È€ô}™¥¹•ÉÁÉ¥¹Ð (€€€€€€€½¹™¥œ¹™¥¹•ÉÁÉ¥¹Ð°€‰¡åÁ½Ñ¡•Í¥Ìˆ°!eA=Q!M%M}=9QIQ}YIM%=8(€€€€¤(€€€É•ÑÕÉ¸MÑÉ…Ñ•å!åÁ½Ñ¡•Í¥Ì (€€€€€€€¥‘•¹Ñ¥™¥•È°(€€€€€€€½¹™¥œ¹ÁÉ•‘¥Ñ½È¹¹…µ”°(€€€€€€€½¹™¥œ¹ÁÉ•‘¥Ñ½È¹Ù•ÉÍ¥½¸°(€€€€€€€!eA=Q!M%M}=9QIQ}YIM%=8°(€€€€€€€½¹™¥œ¹‘¥É•Ñ¥½¹}µ…ÁÁ¥¹œ°(€€€€€€€½¹™¥œ¹Í•±•Ñ•‘}‰¥¹}½É‘¥¹…°°(€€€€€€€½¹™¥œ¹±½Ý•É}‰½Õ¹°(€€€€€€€½¹™¥œ¹ÕÁÁ•É}‰½Õ¹°(€€€€€€€½¹™¥œ¹ÍÕ•ÍÑ•‘}…Ñ¥½¸°(€€€€€€€½¹™¥œ¹¡½É¥é½¸°(€€€€€€€½¹™¥œ¹½¹‘¥Ñ¥½¸°(€€€€€€€€¡½¹™¥œ¹Í•±•Ñ•‘}‰¥¹}½É‘¥¹…°°¤°(€€€€€€€Ù…±Õ”¹É½ÍÍ}•áÁ•Ñ•‘}µ½Ù•µ•¹Ð°(€€€€€€€Ù…±Õ”¹¹•Ñ}•áÁ•Ñ•‘}µ½Ù•µ•¹Ð(€€€€€€€¥˜½¹™¥œ¹ÍÕ•ÍÑ•‘}…Ñ¥½¸¥ÌMÕ•ÍÑ•‘Ñ¥½¸¹1=9}9QId(€€€€€€€•±Í”9½¹”°(€€€€€€€…‰Ì¡Ù…±Õ”¹¹•Ñ}•áÁ•Ñ•‘}µ½Ù•µ•¹Ð½È•¥µ…° À¤¤(€€€€€€€¥˜½¹™¥œ¹ÍÕ•ÍÑ•‘}…Ñ¥½¸¥ÌMÕ•ÍÑ•‘Ñ¥½¸¹a%Q}=I}Y=%(€€€€€€€•±Í”9½¹”°(€€€€€€€Ù…±Õ”¹‘¥…¹½ÍÑ¥Œ¹‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”°(€€€€€€€Ù…±Õ”¹‘¥…¹½ÍÑ¥Œ¹ÍÑ…‰±•}‘¥É•Ñ¥½¹}É…Ñ¥¼°(€€€€€€€½¹™¥œ¹…‘•¹”°(€€€€€€€½¹™¥œ¹¡½É¥é½¸°(€€€€€€€½¹™¥œ¹¡½É¥é½¸°(€€€€€€€ÑÕÁ±”¡É…¹”¡Ù…±Õ”¹‘¥…¹½ÍÑ¥Œ¹Ý¥¹‘½ÝÍ}•Ù…±Õ…Ñ•¤¤°(€€€€€€€Ù¥‘•¹•MÑ…ÑÕÌ¹AI=5%M%9}A=M%Q%Y°(€€€€€€€€‰ØÐÙ…±¥‘…Ñ¥½¸•Ù¥‘•¹”Ý…Ì™É½é•¸‰•™½É”¥Í½±…Ñ•Ñ•ÍÐ…¹½¹™¥Éµ•€ˆ(€€€€€€€€‰Ý¥Ñ¡½ÕÐÁ…É…µ•Ñ•È¡…¹•Ìˆ°(€€€€€€€€‰¡¥ÍÑ½É¥…°•Ù¥‘•¹”‘½•Ì¹½ÐÕ…É…¹Ñ•”™ÕÑÕÉ”É•ÍÕ±ÑÌì¹¼Í¡½ÉÐÍ•±±¥¹œ€ˆ(€€€€€€€€‰½È…ÕÑ½µ…Ñ¥Œ…Ñ¥Ù…Ñ¥½¸ˆ°(€€€€¤(()‘•˜}Ù…±¥‘…Ñ¥½¹}É…¹¬¡Ù…±Õ”èY…±¥‘…Ñ¥½¹½¹™¥ÕÉ…Ñ¥½¹I•ÍÕ±Ð¤€´øÑÕÁ±•m½‰©•Ð°€¸¸¹tè(€€€É•ÑÕÉ¸€ (€€€€€€€±•¸¡Ù…±Õ”¹‘¥…¹½ÍÑ¥Œ¹™…¥±•‘}É•ÅÕ¥É•µ•¹ÑÌ¤°(€€€€€€€€´¡Ù…±Õ”¹¹•Ñ}•áÁ•Ñ•‘}µ½Ù•µ•¹Ð½È•¥µ…° À¤¤°(€€€€€€€€µÙ…±Õ”¹‘¥…¹½ÍÑ¥Œ¹ÍÑ…‰±•}‘¥É•Ñ¥½¹}É…Ñ¥¼°(€€€€€€€€µÙ…±Õ”¹‘¥…¹½ÍÑ¥Œ¹ÕÍ…‰±•}½‰Í•ÉÙ…Ñ¥½¹Ì°(€€€€€€€Ù…±Õ”¹½¹™¥ÕÉ…Ñ¥½¸¹™¥¹•ÉÁÉ¥¹Ð°(€€€€¤(()‘•˜}½Õ¹Ñ}É•©•Ñ¥½¹Ì (€€€Ù…±¥‘…Ñ¥½¸èÑÕÁ±•mY…±¥‘…Ñ¥½¹½¹™¥ÕÉ…Ñ¥½¹I•ÍÕ±Ð°€¸¸¹t°(€€€Ñ•ÍÑÌèÑÕÁ±•m%Í½±…Ñ•‘Q•ÍÑI•ÍÕ±Ð°€¸¸¹t°(¤€´ø‘¥ÑmÍÑÈ°¥¹Ñtè(€€€É•ÍÕ±Ðè‘¥ÑmÍÑÈ°¥¹Ñt€ôíô(€€€™½È¥Ñ•´¥¸Ù…±¥‘…Ñ¥½¸è(€€€€€€€™½ÈÉ•…Í½¸¥¸¥Ñ•´¹‘¥…¹½ÍÑ¥Œ¹™…¥±•‘}É•ÅÕ¥É•µ•¹ÑÌè(€€€€€€€€€€€É•ÍÕ±ÑmÉ•…Í½¸¹Ù…±Õ•t€ôÉ•ÍÕ±Ð¹•Ð¡É•…Í½¸¹Ù…±Õ”°€À¤€¬€Ä(€€€™½ÈÑ•ÍÑ}É•ÍÕ±Ð¥¸Ñ•ÍÑÌè(€€€€€€€¥˜Ñ•ÍÑ}É•ÍÕ±Ð¹É•©•Ñ¥½¹}É•…Í½¸¥Ì¹½Ð9½¹”è(€€€€€€€€€€€É•ÍÕ±ÑmÑ•ÍÑ}É•ÍÕ±Ð¹É•©•Ñ¥½¹}É•…Í½¸¹Ù…±Õ•t€ô€ (€€€€€€€€€€€€€€€É•ÍÕ±Ð¹•Ð¡Ñ•ÍÑ}É•ÍÕ±Ð¹É•©•Ñ¥½¹}É•…Í½¸¹Ù…±Õ”°€À¤€¬€Ä(€€€€€€€€€€€€¤(€€€É•ÑÕÉ¸É•ÍÕ±Ð(()‘•˜}É½ÕÁ}É•©•Ñ¥½¹Ì (€€€Ù…±¥‘…Ñ¥½¸èÑÕÁ±•mY…±¥‘…Ñ¥½¹½¹™¥ÕÉ…Ñ¥½¹I•ÍÕ±Ð°€¸¸¹t°(€€€­•äè…±±…‰±•mmY…±¥‘…Ñ¥½¹½¹™¥ÕÉ…Ñ¥½¹I•ÍÕ±Ñt°ÍÑÉt°(¤€´ø‘¥ÑmÍÑÈ°¥¹Ñtè(€€€É•ÍÕ±Ðè‘¥ÑmÍÑÈ°¥¹Ñt€ôíô(€€€™½È¥Ñ•´¥¸Ù…±¥‘…Ñ¥½¸è(€€€€€€€¥˜¥Ñ•´¹‘¥…¹½ÍÑ¥Œ¹™…¥±•‘}É•ÅÕ¥É•µ•¹ÑÌè(€€€€€€€€€€€¹…µ”€ô­•ä¡¥Ñ•´¤(€€€€€€€€€€€É•ÍÕ±Ñm¹…µ•t€ôÉ•ÍÕ±Ð¹•Ð¡¹…µ”°€À¤€¬€Ä(€€€É•ÑÕÉ¸É•ÍÕ±Ð(()‘•˜}‰½ÑÑ±•¹•¬ (€€€ÑÉ…¥¹¥¹}É•Ñ…¥¹•è¥¹Ð°(€€€Ù…±¥‘…Ñ¥½¸èÑÕÁ±•mY…±¥‘…Ñ¥½¹½¹™¥ÕÉ…Ñ¥½¹I•ÍÕ±Ð°€¸¸¹t°(€€€Ñ•ÍÑÌèÑÕÁ±•m%Í½±…Ñ•‘Q•ÍÑI•ÍÕ±Ð°€¸¸¹t°(¤€´øÍÑÈè(€€€¥˜¹½ÐÑÉ…¥¹¥¹}É•Ñ…¥¹•è(€€€€€€€É•ÑÕÉ¸€‰ÑÉ…¥¹¥¹œˆ(€€€¥˜¹½Ð…¹ä¡¥Ñ•´¹Í•±•Ñ•‘}™½É}Ñ•ÍÐ™½È¥Ñ•´¥¸Ù…±¥‘…Ñ¥½¸¤è(€€€€€€€É•ÑÕÉ¸€‰Ù…±¥‘…Ñ¥½¸ˆ(€€€¥˜¹½Ð…¹ä¡¥Ñ•´¹½¹™¥Éµ•™½È¥Ñ•´¥¸Ñ•ÍÑÌ¤è(€€€€€€€É•ÑÕÉ¸€‰¥Í½±…Ñ•‘}Ñ•ÍÐˆ(€€€É•ÑÕÉ¸€‰½¹™¥Éµ•ˆ(()‘•˜}½ÕÑ½µ•}•áÁ±…¹…Ñ¥½¸ (€€€½¹±ÕÍ¥½¸è¥…¹½ÍÑ¥½¹±ÕÍ¥½¸°½Õ¹ÑÌè‘¥ÑmÍÑÈ°¥¹Ñt°‰½ÑÑ±•¹•¬èÍÑÈ(¤€´øÍÑÈè(€€€‘½µ¥¹…¹Ð€ôµ…à (€€€€€€€½Õ¹ÑÌ¹¥Ñ•µÌ ¤°­•äõ±…µ‰‘„¥Ñ•´è€¡¥Ñ•µlÅt°¥Ñ•µlÁt¤°‘•™…Õ±Ðô ‰¹½¹”ˆ°€À¤(€€€€¤(€€€É•ÑÕÉ¸€ (€€€€€€€˜‰í½¹±ÕÍ¥½¸¹Ù…±Õ•ôì‰½ÑÑ±•¹•¬õí‰½ÑÑ±•¹•­ôì‘½µ¥¹…¹Ñ}É•©•Ñ¥½¸ôˆ(€€€€€€€˜‰í‘½µ¥¹…¹ÑlÁuô€¡í‘½µ¥¹…¹ÑlÅuô¤¸…¥±ÕÉ”Ñ¼‘•Ñ•Ð…¸•‘”¥Ì¹½ÐÁÉ½½˜€ˆ(€€€€€€€€‰Ñ¡…Ð¹½¹”•á¥ÍÑÌ°…¹¡¥ÍÑ½É¥…°•Ù¥‘•¹”‘½•Ì¹½ÐÕ…É…¹Ñ•”™ÕÑÕÉ”É•ÍÕ±ÑÌ¸ˆ(€€€€¤(()‘•˜}‰¥¹}½É‘¥¹…°¡Ù…±Õ”è•¥µ…°°‰½Õ¹‘ÌèÑÕÁ±•m•¥µ…°°€¸¸¹t¤€´ø¥¹Ðè(€€€É•ÑÕÉ¸ÍÕ´¡Ù…±Õ”€ø‰½Õ¹™½È‰½Õ¹¥¸‰½Õ¹‘Ì¤(()‘•˜}™¥¹•ÉÁÉ¥¹Ð ©Ù…±Õ•Ìè½‰©•Ð¤€´øÍÑÈè(€€€•¹½‘•€ô©Í½¸¹‘ÕµÁÌ (€€€€€€€€¡AI%Q%Y}}XÑ}=9QIQ}YIM%=8°Ù…±Õ•Ì¤°(€€€€€€€‘•™…Õ±ÐõÍÑÈ°(€€€€€€€Í½ÉÑ}­•åÌõQÉÕ”°(€€€€€€€Í•Á…É…Ñ½ÉÌô ˆ°ˆ°€ˆèˆ¤°(€€€€¤(€€€É•ÑÕÉ¸¡…Í¡±¥ˆ¹Í¡„ÈÔØ¡•¹½‘•¹•¹½‘” ¤¤¹¡•á‘¥•ÍÐ ¤(()}!}Q1MMLè‘¥ÑmÍÑÈ°…±±…‰±•l¸¸¸°½‰©•Ñut€ôì(€€€Ù…±Õ”¹}}¹…µ•}|èÙ…±Õ”(€€€™½ÈÙ…±Õ”¥¸€ (€€€€€€€AÉ•‘¥Ñ½É•™¥¹¥Ñ¥½¸°(€€€€€€€Ù¥‘•¹•Q¡É•Í¡½±‘Ì°(€€€€€€€¡•…ÁI•±…Ñ¥½¹Í¡¥À°(€€€€€€€Q¡É•Í¡½±‘…±¥‰É…Ñ¥½¸°(€€€€€€€É½é•¹½¹™¥ÕÉ…Ñ¥½¸°(€€€€€€€I•ÅÕ¥É•µ•¹Ñ5…É¥¸°(€€€€€€€‘•¥…¹½ÍÑ¥Œ°(€€€€€€€Y…±¥‘…Ñ¥½¹½¹™¥ÕÉ…Ñ¥½¹I•ÍÕ±Ð°(€€€€€€€%Í½±…Ñ•‘Q•ÍÑI•ÍÕ±Ð°(€€€€¤)ô)}!}9U5Lè‘¥ÑmÍÑÈ°…±±…‰±•mmÍÑÉt°½‰©•Ñut€ôì(€€€Ù…±Õ”¹}}¹…µ•}|èÙ…±Õ”(€€€™½ÈÙ…±Õ”¥¸€ (€€€€€€€AÉ•‘¥Ñ½ÉM•µ…¹Ñ¥-¥¹°(€€€€€€€¥É•Ñ¥½¹5…ÁÁ¥¹œ°(€€€€€€€½¹‘¥Ñ¥½¸°(€€€€€€€MÕ•ÍÑ•‘Ñ¥½¸°(€€€€€€€Ù¥‘•¹•AÉ½™¥±”°(€€€€€€€9•ÕÑÉ…±	…¹‘A½±¥ä°(€€€€€€€A…É…µ•Ñ•ÉM½ÕÉ”°(€€€€€€€I•©•Ñ¥½¹I•…Í½¸°(€€€€¤)ô(()‘•˜•¹½‘•}ØÑ}…¡•}Ù…±Õ”¡Ù…±Õ”è½‰©•Ð¤€´ø½‰©•Ðè(€€€€ˆˆ‰¹½‘”½¹”½µÁ±•Ñ•ÍÑ…”É•ÍÕ±Ð¥¹Ñ¼„)M=8µ½µÁ…Ñ¥‰±”ÑåÁ•Á…å±½…¸ˆˆˆ((€€€¥˜Ù…±Õ”¥Ì9½¹”½È¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°€¡ÍÑÈ°¥¹Ð°‰½½°¤¤è(€€€€€€€É•ÑÕÉ¸Ù…±Õ”(€€€¥˜¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°•¥µ…°¤è(€€€€€€€É•ÑÕÉ¸ìˆ‘‘•¥µ…°ˆèÍÑÈ¡Ù…±Õ”¥ô(€€€¥˜¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°Ñ¥µ•‘•±Ñ„¤è(€€€€€€€É•ÑÕÉ¸ìˆ‘Ñ¥µ•‘•±Ñ…}Í•½¹‘ÌˆèÍÑÈ¡Ù…±Õ”¹Ñ½Ñ…±}Í•½¹‘Ì ¤¥ô(€€€¥˜¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°¹Õ´¤è(€€€€€€€É•ÑÕÉ¸ìˆ‘•¹Õ´ˆèÑåÁ”¡Ù…±Õ”¤¹}}¹…µ•}|°€‰Ù…±Õ”ˆèÍÑÈ¡Ù…±Õ”¹Ù…±Õ”¥ô(€€€¥˜¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°ÑÕÁ±”¤è(€€€€€€€É•ÑÕÉ¸ìˆ‘ÑÕÁ±”ˆèm•¹½‘•}ØÑ}…¡•}Ù…±Õ”¡¥Ñ•´¤™½È¥Ñ•´¥¸Ù…±Õ•uô(€€€¥˜¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°±¥ÍÐ¤è(€€€€€€€É•ÑÕÉ¸m•¹½‘•}ØÑ}…¡•}Ù…±Õ”¡¥Ñ•´¤™½È¥Ñ•´¥¸Ù…±Õ•t(€€€¥˜¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°‘¥Ð¤è(€€€€€€€É•ÑÕÉ¸íÍÑÈ¡­•ä¤è•¹½‘•}ØÑ}…¡•}Ù…±Õ”¡¥Ñ•´¤™½È­•ä°¥Ñ•´¥¸Ù…±Õ”¹¥Ñ•µÌ ¥ô(€€€¥˜¥Í}‘…Ñ…±…ÍÌ¡Ù…±Õ”¤…¹¹½Ð¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°ÑåÁ”¤è(€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€ˆ‘ÑåÁ”ˆèÑåÁ”¡Ù…±Õ”¤¹}}¹…µ•}|°(€€€€€€€€€€€€‰™¥•±‘Ìˆèì(€€€€€€€€€€€€€€€¥Ñ•´¹¹…µ”è•¹½‘•}ØÑ}…¡•}Ù…±Õ”¡•Ñ…ÑÑÈ¡Ù…±Õ”°¥Ñ•´¹¹…µ”¤¤(€€€€€€€€€€€€€€€™½È¥Ñ•´¥¸™¥•±‘Ì¡Ù…±Õ”¤(€€€€€€€€€€€ô°(€€€€€€€ô(€€€É…¥Í”QåÁ•ÉÉ½È¡˜‰Õ¹ÍÕÁÁ½ÉÑ•ØÐ…¡”Ù…±Õ”èíÑåÁ”¡Ù…±Õ”¤¹}}¹…µ•}}ôˆ¤(()‘•˜‘•½‘•}ØÑ}…¡•}Ù…±Õ”¡Ù…±Õ”è½‰©•Ð¤€´ø½‰©•Ðè(€€€€ˆˆ‰•½‘”½¹±äÑ¡”±½Í•Í•Ð½˜Ù•ÉÍ¥½¸´ÐÍÑ…”•Ù¥‘•¹”ÑåÁ•Ì¸ˆˆˆ((€€€¥˜Ù…±Õ”¥Ì9½¹”½È¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°€¡ÍÑÈ°¥¹Ð°‰½½°¤¤è(€€€€€€€É•ÑÕÉ¸Ù…±Õ”(€€€¥˜¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°±¥ÍÐ¤è(€€€€€€€É•ÑÕÉ¸m‘•½‘•}ØÑ}…¡•}Ù…±Õ”¡¥Ñ•´¤™½È¥Ñ•´¥¸Ù…±Õ•t(€€€¥˜¹½Ð¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°‘¥Ð¤è(€€€€€€€É…¥Í”QåÁ•ÉÉ½È ‰¥¹Ù…±¥ØÐ…¡”Á…å±½…ˆ¤(€€€¥˜€ˆ‘‘•¥µ…°ˆ¥¸Ù…±Õ”è(€€€€€€€É•ÑÕÉ¸•¥µ…°¡ÍÑÈ¡Ù…±Õ•lˆ‘‘•¥µ…°‰t¤¤(€€€¥˜€ˆ‘Ñ¥µ•‘•±Ñ…}Í•½¹‘Ìˆ¥¸Ù…±Õ”è(€€€€€€€Í•½¹‘Ì€ô•¥µ…°¡ÍÑÈ¡Ù…±Õ•lˆ‘Ñ¥µ•‘•±Ñ…}Í•½¹‘Ì‰t¤¤(€€€€€€€É•ÑÕÉ¸Ñ¥µ•‘•±Ñ„¡µ¥É½Í•½¹‘Ìõ¥¹Ð¡Í•½¹‘Ì€¨•¥µ…° Å|ÀÀÁ|ÀÀÀ¤¤¤(€€€¥˜€ˆ‘ÑÕÁ±”ˆ¥¸Ù…±Õ”è(€€€€€€€¥Ñ•µÌ€ôÙ…±Õ•lˆ‘ÑÕÁ±”‰t(€€€€€€€¥˜¹½Ð¥Í¥¹ÍÑ…¹”¡¥Ñ•µÌ°±¥ÍÐ¤è(€€€€€€€€€€€É…¥Í”QåÁ•ÉÉ½È ‰¥¹Ù…±¥…¡•ÑÕÁ±”ˆ¤(€€€€€€€É•ÑÕÉ¸ÑÕÁ±”¡‘•½‘•}ØÑ}…¡•}Ù…±Õ”¡¥Ñ•´¤™½È¥Ñ•´¥¸¥Ñ•µÌ¤(€€€¥˜€ˆ‘•¹Õ´ˆ¥¸Ù…±Õ”è(€€€€€€€¹…µ”€ôÍÑÈ¡Ù…±Õ•lˆ‘•¹Õ´‰t¤(€€€€€€€½¹ÍÑÉÕÑ½È€ô}!}9U5L¹•Ð¡¹…µ”¤(€€€€€€€¥˜½¹ÍÑÉÕÑ½È¥Ì9½¹”è(€€€€€€€€€€€É…¥Í”QåÁ•ÉÉ½È¡˜‰Õ¹­¹½Ý¸…¡••¹Õ´èí¹…µ•ôˆ¤(€€€€€€€É•ÑÕÉ¸½¹ÍÑÉÕÑ½È¡ÍÑÈ¡Ù…±Õ•l‰Ù…±Õ”‰t¤¤(€€€¥˜€ˆ‘ÑåÁ”ˆ¥¸Ù…±Õ”è(€€€€€€€¹…µ”€ôÍÑÈ¡Ù…±Õ•lˆ‘ÑåÁ”‰t¤(€€€€€€€½¹ÍÑÉÕÑ½È€ô}!}Q1MML¹•Ð¡¹…µ”¤(€€€€€€€É…Ý}™¥•±‘Ì€ôÙ…±Õ”¹•Ð ‰™¥•±‘Ìˆ¤(€€€€€€€¥˜½¹ÍÑÉÕÑ½È¥Ì9½¹”½È¹½Ð¥Í¥¹ÍÑ…¹”¡É…Ý}™¥•±‘Ì°‘¥Ð¤è(€€€€€€€€€€€É…¥Í”QåÁ•ÉÉ½È¡˜‰Õ¹­¹½Ý¸…¡••Ù¥‘•¹”ÑåÁ”èí¹…µ•ôˆ¤(€€€€€€€‘•½‘•€ôì(€€€€€€€€€€€ÍÑÈ¡­•ä¤è‘•½‘•}ØÑ}…¡•}Ù…±Õ”¡¥Ñ•´¤™½È­•ä°¥Ñ•´¥¸É…Ý}™¥•±‘Ì¹¥Ñ•µÌ ¤(€€€€€€€ô(€€€€€€€É•ÑÕÉ¸½¹ÍÑÉÕÑ½È ¨©‘•½‘•¤(€€€É•ÑÕÉ¸íÍÑÈ¡­•ä¤è‘•½‘•}ØÑ}…¡•}Ù…±Õ”¡¥Ñ•´¤™½È­•ä°¥Ñ•´¥¸Ù…±Õ”¹¥Ñ•µÌ ¥ô