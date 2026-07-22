"""Real bounded Predictive Edge Discovery v4 runtime.

The three stages are intentionally separate functions.  Training selects relationships,
validation freezes and ranks configurations, and isolated test only confirms the
already-frozen identities.
"""

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from math import sqrt
from statistics import median
from time import perf_counter

from app.optimization.feature_matrix import HistoricalFeatureMatrix
from app.optimization.predictive_edge_v4 import (
    CHEAP_RELATIONSHIP_LIMIT,
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
    Direction,
    DirectionBenchmarks,
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
class TrainingWindowEvidence:
    observation_count: int
    absolute_correlation: Decimal | None


@dataclass(frozen=True, slots=True)
class PhaseWindowEvidence:
    phase: str
    gross_returns: tuple[Decimal, ...]
    metrics: EdgeMetrics


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
        validation_result, computed, reused = _evaluate_validation(
            matrix,
            windows,
            simulation,
            frozen_configuration,
            cache_namespace,
            cache,
            cache_writes,
            force_recompute,
        )
        validation_values.append(validation_result)
        validation_computed += computed
        validation_reused += reused
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
        test_result, computed, reused = _evaluate_isolated_test(
            matrix,
            windows,
            simulation,
            selected_validation.configuration,
            cache_namespace,
            cache,
            cache_writes,
            force_recompute,
        )
        test_values.append(test_result)
        test_computed += computed
        test_reused += reused
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
            validation_computed + validation_reused,
            validation_computed,
            validation_reused,
            len(frozen) - len(selected),
            len(selected),
        ),
        StageCounters(
            test_computed + test_reused,
            test_computed,
            test_reused,
            0,
            len(confirmed),
        ),
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
                correlations: list[Decimal] = []
                observations = 0
                for window in windows:
                    training_rows = _phase_rows(
                        matrix.rows, window.training_from, window.training_to
                    )
                    if not training_rows:
                        continue
                    generated += 1
                    cache_key = _fingerprint(
                        cache_namespace,
                        "training",
                        predictor.name,
                        predictor.version,
                        cadence,
                        horizon,
                        window.training_from,
                        window.training_to,
                        _slice_fingerprint(
                            matrix, window.training_from, window.training_to
                        ),
                        simulation.fee_rate,
                        simulation.slippage_rate,
                    )
                    cached = None if force_recompute else cache.get(cache_key)
                    if isinstance(cached, TrainingWindowEvidence):
                        evidence = cached
                        reused += 1
                    else:
                        values, _ = _observations(
                            rows,
                            phase_start=window.training_from,
                            phase_end=window.training_to,
                            predictor=predictor,
                            horizon=horizon,
                            condition=Condition.ALL,
                            volatility_threshold=predictive_edge_volatility_threshold(
                                training_rows
                            ),
                            friction=Decimal(2)
                            * (simulation.fee_rate + simulation.slippage_rate),
                            neutral=Decimal(0),
                            cadence=cadence,
                        )
                        correlation = _correlation(
                            tuple(item.predictor_value for item in values),
                            tuple(item.forward_gross_return for item in values),
                        )
                        evidence = TrainingWindowEvidence(
                            len(values),
                            abs(correlation) if correlation is not None else None,
                        )
                        cache_writes[cache_key] = evidence
                        computed += 1
                    observations += evidence.observation_count
                    if evidence.absolute_correlation is not None:
                        correlations.append(evidence.absolute_correlation)
                mean_correlation = (
                    sum(correlations, Decimal(0)) / Decimal(len(correlations))
                    if correlations
                    else Decimal(0)
                )
                score = mean_correlation * Decimal(str(sqrt(max(observations, 1))))
                fingerprint = _fingerprint(
                    predictor.name,
                    predictor.version,
                    cadence,
                    horizon,
                    "training",
                )
                relationship = CheapRelationship(
                    predictor,
                    cadence,
                    horizon,
                    observations,
                    mean_correlation,
                    score,
                    group_by_predictor.get(predictor.name, predictor.name),
                    fingerprint,
                )
                candidates.append(relationship)
    return _diversified_cheap(tuple(candidates)), generated, computed, reused


def _training_redundancy(
    matrix: HistoricalFeatureMatrix, windows: tuple[WalkForwardWindow, ...]
) -> tuple[RedundancyEvidence, ...]:
    if not windows:
        return ()
    timestamps = {
        row.candle_open_time
        for window in windows
        for row in matrix.rows
        if window.training_from <= row.candle_open_time < window.training_to
    }
    rows = tuple(row for row in matrix.rows if row.candle_open_time in timestamps)
    values = {
        item.name: tuple(_predictors(row)[item.name] for row in rows)
        for item in PREDICTORS
    }
    evidence: list[RedundancyEvidence] = []
    for left_index, left in enumerate(PREDICTORS):
        for right in PREDICTORS[left_index + 1 :]:
            correlation = _correlation(values[left.name], values[right.name])
            if correlation is None or abs(correlation) < Decimal("0.90"):
                continue
            group = min(left.name, right.name)
            evidence.append(
                RedundancyEvidence(left.name, right.name, correlation, group)
            )
    return tuple(evidence)


def _redundancy_groups(evidence: tuple[RedundancyEvidence, ...]) -> dict[str, str]:
    parent: dict[str, str] = {item.name: item.name for item in PREDICTORS}

    def root(value: str) -> str:
        while parent[value] != value:
            value = parent[value]
        return value

    for item in evidence:
        left, right = root(item.left_predictor), root(item.right_predictor)
        parent[max(left, right)] = min(left, right)
    return {name: root(name) for name in parent}


def _diversified_cheap(
    values: tuple[CheapRelationship, ...],
) -> tuple[CheapRelationship, ...]:
    buckets: dict[str, list[CheapRelationship]] = {}
    for item in sorted(values, key=lambda value: (-value.score, value.fingerprint)):
        buckets.setdefault(item.redundancy_group, []).append(item)
    result: list[CheapRelationship] = []
    while len(result) < CHEAP_RELATIONSHIP_LIMIT and any(buckets.values()):
        for group in sorted(buckets):
            if buckets[group] and len(result) < CHEAP_RELATIONSHIP_LIMIT:
                result.append(buckets[group].pop(0))
    return tuple(result)


def _bounded_frozen_configurations(
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    simulation: SimulationConfiguration,
    relationships: tuple[CheapRelationship, ...],
) -> tuple[FrozenConfiguration, ...]:
    # This bounded design set is deliberate rather than an index-to-configuration
    # mapping.  It exercises every dimension without constructing the full Cartesian
    # product.  All variants are scored only from training evidence below.
    variants = (
        (
            3,
            Condition.ALL,
            NeutralBandPolicy.ECONOMIC_FRICTION,
            EvidenceProfile.STRICT_V3_COMPATIBLE,
        ),
        (
            5,
            Condition.ALL,
            NeutralBandPolicy.ZERO_DIRECTION,
            EvidenceProfile.BALANCED_V4,
        ),
        (
            7,
            Condition.ALL,
            NeutralBandPolicy.TRAINING_NOISE,
            EvidenceProfile.EXPLORATORY_V4,
        ),
        (
            5,
            Condition.EMA_ABOVE,
            NeutralBandPolicy.ECONOMIC_FRICTION,
            EvidenceProfile.BALANCED_V4,
        ),
        (
            5,
            Condition.EMA_BELOW,
            NeutralBandPolicy.TRAINING_NOISE,
            EvidenceProfile.EXPLORATORY_V4,
        ),
        (
            7,
            Condition.HIGH_VOLATILITY,
            NeutralBandPolicy.ZERO_DIRECTION,
            EvidenceProfile.BALANCED_V4,
        ),
        (
            3,
            Condition.NORMAL_VOLATILITY,
            NeutralBandPolicy.TRAINING_NOISE,
            EvidenceProfile.EXPLORATORY_V4,
        ),
    )
    # Reserve half the budget for relationship breadth.  The remaining capacity goes
    # to additional training-ranked variants of the strongest relationships.
    ranked_relationships = sorted(
        relationships,
        key=lambda item: (-item.score, -item.observation_count, item.fingerprint),
    )[: VALIDATION_CONFIGURATION_LIMIT // 2]
    ranked_by_relationship: list[
        tuple[CheapRelationship, tuple[FrozenConfiguration, ...]]
    ] = []
    for relationship in ranked_relationships:
        candidates = tuple(
            _freeze_configuration(
                matrix,
                windows,
                simulation,
                relationship,
                bins,
                condition,
                policy,
                profile,
            )
            for bins, condition, policy, profile in variants
        )
        ranked_by_relationship.append(
            (
                relationship,
                tuple(
                    sorted(
                        candidates,
                        key=lambda item: (
                            -_training_configuration_proxy(matrix, windows, item),
                            item.fingerprint,
                        ),
                    )
                ),
            )
        )
    selected = [items[0] for _, items in ranked_by_relationship if items]
    extras = sorted(
        (
            (relationship.score, rank, item)
            for relationship, items in ranked_by_relationship
            for rank, item in enumerate(items[1:], start=1)
        ),
        key=lambda value: (-value[0], value[1], value[2].fingerprint),
    )
    selected.extend(
        item for _, _, item in extras[: VALIDATION_CONFIGURATION_LIMIT - len(selected)]
    )
    return tuple(selected)


def _training_configuration_proxy(
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    configuration: FrozenConfiguration,
) -> Decimal:
    """Rank bounded alternatives using only selected-bin training evidence."""

    rows = {row.candle_open_time: row for row in matrix.rows}
    selected_returns: list[Decimal] = []
    for window in windows:
        observations, _ = _observations(
            rows,
            phase_start=window.training_from,
            phase_end=window.training_to,
            predictor=configuration.predictor,
            horizon=configuration.horizon,
            condition=configuration.condition,
            volatility_threshold=dict(configuration.volatility_thresholds)[
                window.ordinal
            ],
            friction=Decimal(0),
            neutral=configuration.neutral_threshold,
            cadence=configuration.cadence,
        )
        selected_returns.extend(
            item.forward_gross_return
            for item in observations
            if _bin_ordinal(item.predictor_value, configuration.bin_bounds)
            == configuration.selected_bin_ordinal
        )
    if not selected_returns:
        return Decimal(0)
    gross = sum(selected_returns, Decimal(0)) / Decimal(len(selected_returns))
    aligned = _action_aligned_gross(gross, configuration.suggested_action)
    return aligned * Decimal(str(sqrt(len(selected_returns))))


def _freeze_configuration(
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    simulation: SimulationConfiguration,
    relationship: CheapRelationship,
    bins: int,
    condition: Condition,
    policy: NeutralBandPolicy,
    profile: EvidenceProfile,
) -> FrozenConfiguration:
    rows = {row.candle_open_time: row for row in matrix.rows}
    gross_returns: list[Decimal] = []
    training_observations: list[Observation] = []
    volatility: list[tuple[int, Decimal]] = []
    for window in windows:
        phase = _phase_rows(matrix.rows, window.training_from, window.training_to)
        threshold = predictive_edge_volatility_threshold(phase)
        volatility.append((window.ordinal, threshold))
        values, _ = _observations(
            rows,
            phase_start=window.training_from,
            phase_end=window.training_to,
            predictor=relationship.predictor,
            horizon=relationship.horizon,
            condition=condition,
            volatility_threshold=threshold,
            friction=Decimal(2) * (simulation.fee_rate + simulation.slippage_rate),
            neutral=Decimal(0),
            cadence=relationship.cadence,
        )
        training_observations.extend(values)
        gross_returns.extend(item.forward_gross_return for item in values)
    neutral = neutral_threshold_for_training(
        policy=policy,
        training_absolute_returns=tuple(abs(item) for item in gross_returns),
        fee_rate=simulation.fee_rate,
        slippage_rate=simulation.slippage_rate,
    )
    mapping = _learn_direction_mapping(
        relationship.predictor, tuple(training_observations)
    )
    predictor_values = tuple(item.predictor_value for item in training_observations)
    bounds = _quantile_bounds(predictor_values, bins)
    thresholds, calibrations = _profile_thresholds(
        profile, tuple(training_observations), relationship
    )
    ordinal, lower, upper, action = _training_bin_identity(
        tuple(training_observations), bounds, mapping, relationship.predictor
    )
    fingerprint = _fingerprint(
        relationship.fingerprint,
        bins,
        condition.value,
        policy.value,
        profile.value,
        neutral,
        thresholds,
        bounds,
        mapping.value,
        simulation.fee_rate,
        simulation.slippage_rate,
    )
    return FrozenConfiguration(
        fingerprint,
        relationship.predictor,
        relationship.cadence,
        relationship.horizon,
        bins,
        condition,
        policy,
        profile,
        mapping,
        bounds,
        ordinal,
        lower,
        upper,
        action,
        neutral,
        thresholds,
        calibrations,
        tuple(volatility),
    )


def _profile_thresholds(
    profile: EvidenceProfile,
    observations: tuple[Observation, ...],
    relationship: CheapRelationship,
) -> tuple[EvidenceThresholds, tuple[ThresholdCalibration, ...]]:
    strict = EvidenceThresholds()
    if profile is EvidenceProfile.STRICT_V3_COMPATIBLE:
        values = (
            ("minimum_observations", Decimal(strict.minimum_observations)),
            ("minimum_absolute_correlation", strict.minimum_absolute_correlation),
            ("minimum_directional_hit_rate", strict.minimum_directional_hit_rate),
        )
        return strict, tuple(
            ThresholdCalibration(
                name,
                value,
                value,
                value,
                value,
                "v3 fixed default",
                ParameterSource.FIXED_DEFAULT,
            )
            for name, value in values
        )
    count = Decimal(len(observations))
    observed_correlation = relationship.mean_absolute_correlation
    if profile is EvidenceProfile.BALANCED_V4:
        minimum_observations = int(
            max(Decimal(20), min(Decimal(60), count / Decimal(10)))
        )
        correlation = max(
            Decimal("0.05"),
            min(Decimal("0.12"), observed_correlation * Decimal("0.75")),
        )
        hit = max(
            Decimal("0.52"),
            min(Decimal("0.56"), Decimal("0.50") + observed_correlation / Decimal(2)),
        )
        stability = Decimal("0.55")
    else:
        minimum_observations = int(
            max(Decimal(15), min(Decimal(30), count / Decimal(15)))
        )
        correlation = max(
            Decimal("0.02"),
            min(Decimal("0.08"), observed_correlation * Decimal("0.50")),
        )
        hit = max(
            Decimal("0.50"),
            min(Decimal("0.53"), Decimal("0.49") + observed_correlation / Decimal(2)),
        )
        stability = Decimal("0.50")
    thresholds = EvidenceThresholds(
        minimum_observations=minimum_observations,
        minimum_windows=3,
        minimum_stable_direction_ratio=stability,
        minimum_absolute_correlation=correlation,
        minimum_directional_hit_rate=hit,
        minimum_hit_rate_improvement_over_benchmark=Decimal("0.01"),
    )
    calibrations = (
        ThresholdCalibration(
            "minimum_observations",
            count / Decimal(10),
            Decimal(minimum_observations),
            Decimal(15 if profile is EvidenceProfile.EXPLORATORY_V4 else 20),
            Decimal(30 if profile is EvidenceProfile.EXPLORATORY_V4 else 60),
            "training observation count / divisor with floors and ceilings",
            ParameterSource.TRAINING_CALIBRATED,
        ),
        ThresholdCalibration(
            "minimum_absolute_correlation",
            observed_correlation,
            correlation,
            Decimal("0.02" if profile is EvidenceProfile.EXPLORATORY_V4 else "0.05"),
            Decimal("0.08" if profile is EvidenceProfile.EXPLORATORY_V4 else "0.12"),
            "training correlation fraction with floors and ceilings",
            ParameterSource.TRAINING_CALIBRATED,
        ),
        ThresholdCalibration(
            "minimum_directional_hit_rate",
            Decimal("0.50") + observed_correlation / Decimal(2),
            hit,
            Decimal("0.50" if profile is EvidenceProfile.EXPLORATORY_V4 else "0.52"),
            Decimal("0.53" if profile is EvidenceProfile.EXPLORATORY_V4 else "0.56"),
            "training correlation-derived hit floor",
            ParameterSource.TRAINING_CALIBRATED,
        ),
    )
    return thresholds, calibrations


def _training_bin_identity(
    observations: tuple[Observation, ...],
    bounds: tuple[Decimal, ...],
    mapping: DirectionMapping,
    predictor: PredictorDefinition,
) -> tuple[int, Decimal | None, Decimal | None, SuggestedAction]:
    if not observations or mapping is DirectionMapping.NOT_APPLICABLE:
        return 0, None, bounds[0] if bounds else None, SuggestedAction.DIAGNOSTIC_ONLY
    bins = len(bounds) + 1
    candidates: list[tuple[Decimal, int]] = []
    for ordinal in range(bins):
        values = [
            item
            for item in observations
            if _bin_ordinal(item.predictor_value, bounds) == ordinal
        ]
        mean = (
            sum((item.forward_gross_return for item in values), Decimal(0))
            / Decimal(len(values))
            if values
            else Decimal(0)
        )
        candidates.append((abs(mean), ordinal))
    ordinal = max(candidates)[1]
    selected = [
        item.forward_gross_return
        for item in observations
        if _bin_ordinal(item.predictor_value, bounds) == ordinal
    ]
    mean = (
        sum(selected, Decimal(0)) / Decimal(len(selected)) if selected else Decimal(0)
    )
    action = SuggestedAction.LONG_ENTRY if mean > 0 else SuggestedAction.EXIT_OR_AVOID
    return (
        ordinal,
        bounds[ordinal - 1] if ordinal else None,
        bounds[ordinal] if ordinal < len(bounds) else None,
        action,
    )


def _evaluate_validation(
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    simulation: SimulationConfiguration,
    configuration: FrozenConfiguration,
    cache_namespace: str,
    cache: Mapping[str, object],
    cache_writes: dict[str, object],
    force_recompute: bool,
) -> tuple[ValidationConfigurationResult, int, int]:
    values, window_metrics, computed, reused = _phase_evidence(
        matrix,
        windows,
        simulation,
        configuration,
        "validation",
        cache_namespace,
        cache,
        cache_writes,
        force_recompute,
    )
    diagnostic = _diagnostic(configuration, values, window_metrics, simulation)
    failed = set(diagnostic.failed_requirements)
    economics_pass = RejectionReason.MOVEMENT_DOES_NOT_COVER_COSTS not in failed
    selected = (
        not failed
        and economics_pass
        and configuration.suggested_action is not SuggestedAction.DIAGNOSTIC_ONLY
    )
    if configuration.profile is EvidenceProfile.EXPLORATORY_V4 and not economics_pass:
        selected = False
    status = "selected_for_test" if selected else "near_miss" if values else "rejected"
    return (
        ValidationConfigurationResult(
            configuration,
            diagnostic,
            status,
            selected,
            diagnostic.strongest_directional_benchmark,
            diagnostic.standard_error,
            diagnostic.gross_expected_movement,
            diagnostic.net_expected_movement,
        ),
        computed,
        reused,
    )


def _evaluate_isolated_test(
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    simulation: SimulationConfiguration,
    configuration: FrozenConfiguration,
    cache_namespace: str,
    cache: Mapping[str, object],
    cache_writes: dict[str, object],
    force_recompute: bool,
) -> tuple[IsolatedTestResult, int, int]:
    values, window_metrics, computed, reused = _phase_evidence(
        matrix,
        windows,
        simulation,
        configuration,
        "test",
        cache_namespace,
        cache,
        cache_writes,
        force_recompute,
    )
    diagnostic = _diagnostic(configuration, values, window_metrics, simulation)
    confirmed = (
        not diagnostic.failed_requirements
        and configuration.suggested_action is not SuggestedAction.DIAGNOSTIC_ONLY
    )
    return (
        IsolatedTestResult(
            configuration.fingerprint,
            confirmed,
            len(values),
            diagnostic.directional_hit_rate,
            diagnostic.strongest_directional_benchmark,
            diagnostic.gross_expected_movement,
            diagnostic.net_expected_movement,
            diagnostic.primary_rejection_reason if not confirmed else None,
        ),
        computed,
        reused,
    )


def _phase_evidence(
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    simulation: SimulationConfiguration,
    configuration: FrozenConfiguration,
    phase: str,
    cache_namespace: str,
    cache: Mapping[str, object],
    cache_writes: dict[str, object],
    force_recompute: bool,
) -> tuple[tuple[Observation, ...], tuple[EdgeMetrics, ...], int, int]:
    rows = {row.candle_open_time: row for row in matrix.rows}
    all_values: list[Observation] = []
    metrics: list[EdgeMetrics] = []
    computed = 0
    reused = 0
    volatility = dict(configuration.volatility_thresholds)
    friction = Decimal(2) * (simulation.fee_rate + simulation.slippage_rate)
    for window in windows:
        start = getattr(window, f"{phase}_from")
        end = getattr(window, f"{phase}_to")
        cache_key = _fingerprint(
            cache_namespace,
            phase,
            configuration.fingerprint,
            start,
            end,
            _slice_fingerprint(matrix, start, end),
            simulation.fee_rate,
            simulation.slippage_rate,
        )
        cached = None if force_recompute else cache.get(cache_key)
        if isinstance(cached, PhaseWindowEvidence):
            evidence = cached
            reused += 1
        else:
            observations, missing = _observations(
                rows,
                phase_start=start,
                phase_end=end,
                predictor=configuration.predictor,
                horizon=configuration.horizon,
                condition=configuration.condition,
                volatility_threshold=volatility[window.ordinal],
                friction=friction,
                neutral=configuration.neutral_threshold,
                cadence=configuration.cadence,
            )
            selected = tuple(
                item
                for item in observations
                if _bin_ordinal(item.predictor_value, configuration.bin_bounds)
                == configuration.selected_bin_ordinal
            )
            evidence = PhaseWindowEvidence(
                phase,
                tuple(item.forward_gross_return for item in selected),
                _metrics(
                    selected,
                    missing,
                    predictor=configuration.predictor,
                    center=Decimal(0),
                    mapping=configuration.direction_mapping,
                    friction=friction,
                    thresholds=configuration.thresholds,
                ),
            )
            cache_writes[cache_key] = evidence
            computed += 1
        # Diagnostics need only the selected forward returns and the fully calculated
        # window metric. Reconstruct minimal immutable observations without provider
        # I/O.
        all_values.extend(
            Observation(
                start,
                Decimal(0),
                gross,
                abs(gross),
                abs(gross) - friction,
                Direction.UP
                if gross > 0
                else Direction.DOWN
                if gross < 0
                else Direction.NEUTRAL,
            )
            for gross in evidence.gross_returns
        )
        metrics.append(evidence.metrics)
    return tuple(all_values), tuple(metrics), computed, reused


def _slice_fingerprint(
    matrix: HistoricalFeatureMatrix, start: datetime, end: datetime
) -> str:
    """Fingerprint exactly the immutable feature rows consumed by one phase slice."""

    return _fingerprint(
        tuple(row for row in matrix.rows if start <= row.candle_open_time < end)
    )


def _diagnostic(
    configuration: FrozenConfiguration,
    values: tuple[Observation, ...],
    window_metrics: tuple[EdgeMetrics, ...],
    simulation: SimulationConfiguration,
) -> EdgeDiagnostic:
    correlations = tuple(
        item.correlation for item in window_metrics if item.correlation is not None
    )
    hits = tuple(
        item.directional_hit_rate
        for item in window_metrics
        if item.directional_hit_rate is not None
    )
    benchmarks = tuple(
        item.direction_benchmarks.strongest
        for item in window_metrics
        if item.direction_benchmarks is not None
    )
    usable = sum(item.usable_count for item in window_metrics)
    correlation = (
        sum((abs(item) for item in correlations), Decimal(0))
        / Decimal(len(correlations))
        if correlations
        else Decimal(0)
    )
    hit = sum(hits, Decimal(0)) / Decimal(len(hits)) if hits else Decimal(0)
    benchmark = max(benchmarks) if benchmarks else Decimal(0)
    improvement = hit - benchmark
    gross = (
        sum((item.forward_gross_return for item in values), Decimal(0))
        / Decimal(len(values))
        if values
        else Decimal(0)
    )
    aligned_gross = _action_aligned_gross(gross, configuration.suggested_action)
    net = aligned_gross - Decimal(2) * (simulation.fee_rate + simulation.slippage_rate)
    standard_errors = tuple(
        item.standard_error
        for item in window_metrics
        if item.standard_error is not None
    )
    standard_error = (
        sum(standard_errors, Decimal(0)) / Decimal(len(standard_errors))
        if standard_errors
        else None
    )
    signs = tuple(1 if item > 0 else -1 if item < 0 else 0 for item in correlations)
    dominant = max(signs.count(1), signs.count(-1)) if signs else 0
    stability = Decimal(dominant) / Decimal(len(signs)) if signs else Decimal(0)
    dispersion = (
        max(correlations) - min(correlations) if len(correlations) > 1 else Decimal(0)
    )
    thresholds = configuration.thresholds
    margins = (
        requirement_margin(
            RejectionReason.INSUFFICIENT_OBSERVATIONS,
            Decimal(usable),
            Decimal(thresholds.minimum_observations),
        ),
        requirement_margin(
            RejectionReason.INSUFFICIENT_WINDOWS,
            Decimal(len(window_metrics)),
            Decimal(thresholds.minimum_windows),
        ),
        requirement_margin(
            RejectionReason.UNSTABLE_DIRECTION,
            stability,
            thresholds.minimum_stable_direction_ratio,
        ),
        requirement_margin(
            RejectionReason.CORRELATION_BELOW_THRESHOLD,
            correlation,
            thresholds.minimum_absolute_correlation,
        ),
        requirement_margin(
            RejectionReason.HIT_RATE_BELOW_THRESHOLD,
            hit,
            thresholds.minimum_directional_hit_rate,
        ),
        requirement_margin(
            RejectionReason.BENCHMARK_IMPROVEMENT_BELOW_THRESHOLD,
            improvement,
            thresholds.minimum_hit_rate_improvement_over_benchmark,
        ),
        requirement_margin(
            RejectionReason.MOVEMENT_DOES_NOT_COVER_COSTS, net, ECONOMIC_EDGE_MARGIN
        ),
        requirement_margin(
            RejectionReason.EXCESSIVE_DISPERSION,
            Decimal("0.50") - dispersion,
            Decimal(0),
        ),
        requirement_margin(
            RejectionReason.INCONSISTENT_SIGN,
            stability,
            thresholds.minimum_stable_direction_ratio,
        ),
    )
    failed = tuple(item.requirement for item in margins if not item.passed)
    passed = tuple(item.requirement for item in margins if item.passed)
    return EdgeDiagnostic(
        configuration.fingerprint,
        "promising" if not failed else "near_miss",
        failed,
        passed,
        failed[0] if failed else None,
        margins,
        usable,
        len(window_metrics),
        dominant,
        stability,
        correlation if correlations else None,
        median(correlations) if correlations else None,
        min(correlations) if correlations else None,
        hit if hits else None,
        benchmark if benchmarks else None,
        improvement if benchmarks and hits else None,
        aligned_gross if values else None,
        net if values else None,
        standard_error,
        dispersion if correlations else None,
    )


def _action_aligned_gross(
    gross_expected_movement: Decimal, action: SuggestedAction
) -> Decimal:
    """Align economic value with the frozen long-only action, never magnitude."""

    if action is SuggestedAction.LONG_ENTRY:
        return gross_expected_movement
    if action is SuggestedAction.EXIT_OR_AVOID:
        return -gross_expected_movement
    return Decimal(0)


def _effective_parameters(
    validation: tuple[ValidationConfigurationResult, ...],
    simulation: SimulationConfiguration,
) -> tuple[AuditedParameter, ...]:
    values: list[AuditedParameter] = [
        AuditedParameter(
            "fee_rate",
            str(simulation.fee_rate),
            ParameterSource.FIXED_DEFAULT,
            "simulation cost assumption",
        ),
        AuditedParameter(
            "slippage_rate",
            str(simulation.slippage_rate),
            ParameterSource.FIXED_DEFAULT,
            "simulation cost assumption",
        ),
        AuditedParameter(
            "cheap_relationship_limit",
            str(CHEAP_RELATIONSHIP_LIMIT),
            ParameterSource.FIXED_DEFAULT,
            "bounded training funnel",
        ),
        AuditedParameter(
            "validation_configuration_limit",
            str(VALIDATION_CONFIGURATION_LIMIT),
            ParameterSource.FIXED_DEFAULT,
            "bounded validation funnel",
        ),
        AuditedParameter(
            "isolated_test_configuration_limit",
            str(ISOLATED_TEST_CONFIGURATION_LIMIT),
            ParameterSource.FIXED_DEFAULT,
            "isolated test budget",
        ),
    ]
    for result in validation:
        configuration = result.configuration
        prefix = configuration.fingerprint
        for name, value, source, rationale in (
            (
                "cadence",
                configuration.cadence,
                ParameterSource.TRAINING_CALIBRATED,
                "selected by training-only screening and frozen before validation",
            ),
            (
                "horizon",
                configuration.horizon,
                ParameterSource.TRAINING_CALIBRATED,
                "selected by training-only screening and frozen before validation",
            ),
            (
                "bins",
                configuration.bins,
                ParameterSource.FIXED_DEFAULT,
                "versioned bounded expansion",
            ),
            (
                "condition",
                configuration.condition.value,
                ParameterSource.FIXED_DEFAULT,
                "existing condition contract",
            ),
            (
                "neutral_policy",
                configuration.neutral_policy.value,
                ParameterSource.FIXED_DEFAULT,
                "versioned neutral policy",
            ),
            (
                "neutral_threshold",
                configuration.neutral_threshold,
                ParameterSource.TRAINING_CALIBRATED
                if configuration.neutral_policy is NeutralBandPolicy.TRAINING_NOISE
                else ParameterSource.FIXED_DEFAULT,
                "frozen before validation",
            ),
            (
                "profile",
                configuration.profile.value,
                ParameterSource.FIXED_DEFAULT,
                "versioned research profile",
            ),
        ):
            values.append(
                AuditedParameter(f"{prefix}:{name}", str(value), source, rationale)
            )
        for calibration in configuration.threshold_calibrations:
            values.append(
                AuditedParameter(
                    f"{prefix}:{calibration.name}",
                    str(calibration.final_value),
                    calibration.source,
                    calibration.method,
                )
            )
    return tuple(values)


def _hypothesis(value: ValidationConfigurationResult) -> StrategyHypothesis:
    config = value.configuration
    identifier = _fingerprint(
        config.fingerprint, "hypothesis", HYPOTHESIS_CONTRACT_VERSION
    )
    return StrategyHypothesis(
        identifier,
        config.predictor.name,
        config.predictor.version,
        HYPOTHESIS_CONTRACT_VERSION,
        config.direction_mapping,
        config.selected_bin_ordinal,
        config.lower_bound,
        config.upper_bound,
        config.suggested_action,
        config.horizon,
        config.condition,
        (config.selected_bin_ordinal,),
        value.gross_expected_movement,
        value.net_expected_movement
        if config.suggested_action is SuggestedAction.LONG_ENTRY
        else None,
        abs(value.net_expected_movement or Decimal(0))
        if config.suggested_action is SuggestedAction.EXIT_OR_AVOID
        else None,
        value.diagnostic.directional_hit_rate,
        value.diagnostic.stable_direction_ratio,
        config.cadence,
        config.horizon,
        config.horizon,
        tuple(range(value.diagnostic.windows_evaluated)),
        EvidenceStatus.PROMISING_POSITIVE,
        "v4 validation evidence was frozen before isolated test and confirmed "
        "without parameter changes",
        "historical evidence does not guarantee future results; no short selling "
        "or automatic activation",
    )


def _validation_rank(value: ValidationConfigurationResult) -> tuple[object, ...]:
    return (
        len(value.diagnostic.failed_requirements),
        -(value.net_expected_movement or Decimal(0)),
        -value.diagnostic.stable_direction_ratio,
        -value.diagnostic.usable_observations,
        value.configuration.fingerprint,
    )


def _count_rejections(
    validation: tuple[ValidationConfigurationResult, ...],
    tests: tuple[IsolatedTestResult, ...],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in validation:
        for reason in item.diagnostic.failed_requirements:
            result[reason.value] = result.get(reason.value, 0) + 1
    for test_result in tests:
        if test_result.rejection_reason is not None:
            result[test_result.rejection_reason.value] = (
                result.get(test_result.rejection_reason.value, 0) + 1
            )
    return result


def _group_rejections(
    validation: tuple[ValidationConfigurationResult, ...],
    key: Callable[[ValidationConfigurationResult], str],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in validation:
        if item.diagnostic.failed_requirements:
            name = key(item)
            result[name] = result.get(name, 0) + 1
    return result


def _bottleneck(
    training_retained: int,
    validation: tuple[ValidationConfigurationResult, ...],
    tests: tuple[IsolatedTestResult, ...],
) -> str:
    if not training_retained:
        return "training"
    if not any(item.selected_for_test for item in validation):
        return "validation"
    if not any(item.confirmed for item in tests):
        return "isolated_test"
    return "confirmed"


def _outcome_explanation(
    conclusion: DiagnosticConclusion, counts: dict[str, int], bottleneck: str
) -> str:
    dominant = max(
        counts.items(), key=lambda item: (item[1], item[0]), default=("none", 0)
    )
    return (
        f"{conclusion.value}; bottleneck={bottleneck}; dominant_rejection="
        f"{dominant[0]} ({dominant[1]}). Failure to detect an edge is not proof "
        "that none exists, and historical evidence does not guarantee future results."
    )


def _bin_ordinal(value: Decimal, bounds: tuple[Decimal, ...]) -> int:
    return sum(value > bound for bound in bounds)


def _fingerprint(*values: object) -> str:
    encoded = json.dumps(
        (PREDICTIVE_EDGE_V4_CONTRACT_VERSION, values),
        default=str,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


_CACHE_DATACLASSES: dict[str, Callable[..., object]] = {
    value.__name__: value
    for value in (
        PredictorDefinition,
        EvidenceThresholds,
        DirectionBenchmarks,
        EdgeMetrics,
        TrainingWindowEvidence,
        PhaseWindowEvidence,
        CheapRelationship,
        ThresholdCalibration,
        FrozenConfiguration,
        RequirementMargin,
        EdgeDiagnostic,
        ValidationConfigurationResult,
        IsolatedTestResult,
    )
}
_CACHE_ENUMS: dict[str, Callable[[str], object]] = {
    value.__name__: value
    for value in (
        PredictorSemanticKind,
        Direction,
        EvidenceStatus,
        DirectionMapping,
        Condition,
        SuggestedAction,
        EvidenceProfile,
        NeutralBandPolicy,
        ParameterSource,
        RejectionReason,
    )
}


def encode_v4_cache_value(value: object) -> object:
    """Encode one completed stage result into a JSON-compatible typed payload."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Decimal):
        return {"$decimal": str(value)}
    if isinstance(value, timedelta):
        return {"$timedelta_seconds": str(value.total_seconds())}
    if isinstance(value, Enum):
        return {"$enum": type(value).__name__, "value": str(value.value)}
    if isinstance(value, tuple):
        return {"$tuple": [encode_v4_cache_value(item) for item in value]}
    if isinstance(value, list):
        return [encode_v4_cache_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode_v4_cache_value(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "$type": type(value).__name__,
            "fields": {
                item.name: encode_v4_cache_value(getattr(value, item.name))
                for item in fields(value)
            },
        }
    raise TypeError(f"unsupported v4 cache value: {type(value).__name__}")


def decode_v4_cache_value(value: object) -> object:
    """Decode only the closed set of version-4 stage evidence types."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, list):
        return [decode_v4_cache_value(item) for item in value]
    if not isinstance(value, dict):
        raise TypeError("invalid v4 cache payload")
    if "$decimal" in value:
        return Decimal(str(value["$decimal"]))
    if "$timedelta_seconds" in value:
        seconds = Decimal(str(value["$timedelta_seconds"]))
        return timedelta(microseconds=int(seconds * Decimal(1_000_000)))
    if "$tuple" in value:
        items = value["$tuple"]
        if not isinstance(items, list):
            raise TypeError("invalid cached tuple")
        return tuple(decode_v4_cache_value(item) for item in items)
    if "$enum" in value:
        name = str(value["$enum"])
        constructor = _CACHE_ENUMS.get(name)
        if constructor is None:
            raise TypeError(f"unknown cached enum: {name}")
        return constructor(str(value["value"]))
    if "$type" in value:
        name = str(value["$type"])
        constructor = _CACHE_DATACLASSES.get(name)
        raw_fields = value.get("fields")
        if constructor is None or not isinstance(raw_fields, dict):
            raise TypeError(f"unknown cached evidence type: {name}")
        decoded = {
            str(key): decode_v4_cache_value(item) for key, item in raw_fields.items()
        }
        return constructor(**decoded)
    return {str(key): decode_v4_cache_value(item) for key, item in value.items()}
