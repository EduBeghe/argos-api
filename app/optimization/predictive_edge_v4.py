"""Bounded, auditable Predictive Edge Discovery v4 contracts.

This module contains only deterministic training/validation policy.  Isolated test
evidence is deliberately absent from screening, ranking and sensitivity inputs.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import timedelta
from decimal import Decimal
from enum import StrEnum

from app.optimization.predictive_edges import (
    Condition,
    EvidenceThresholds,
    PredictorDefinition,
    PredictorSemanticKind,
    StabilityEdge,
)

PREDICTIVE_EDGE_V4_CONTRACT_VERSION = 4
CHEAP_RELATIONSHIP_LIMIT = 120
VALIDATION_CONFIGURATION_LIMIT = 40
ISOLATED_TEST_CONFIGURATION_LIMIT = 12
NEAR_MISS_LIMIT = 50
SENSITIVITY_LIMIT = 20
ECONOMIC_EDGE_MARGIN = Decimal("0.000100000000000000")

DISCOVERY_CADENCES = tuple(
    timedelta(minutes=value) for value in (30, 60, 120, 240, 360)
)
DISCOVERY_HORIZONS = tuple(
    timedelta(minutes=value)
    for value in (30, 60, 120, 240, 360, 480, 720, 1440, 2880, 4320, 10080)
)
DISCOVERY_BIN_COUNTS = (3, 5, 7)


class ParameterSource(StrEnum):
    FIXED_DEFAULT = "fixed_default"
    TRAINING_CALIBRATED = "training_calibrated"
    SELECTED_ON_VALIDATION = "selected_on_validation"


class NeutralBandPolicy(StrEnum):
    ZERO_DIRECTION = "zero_direction"
    TRAINING_NOISE = "training_noise"
    ECONOMIC_FRICTION = "economic_friction"


class EvidenceProfile(StrEnum):
    STRICT_V3_COMPATIBLE = "strict_v3_compatible"
    BALANCED_V4 = "balanced_v4"
    EXPLORATORY_V4 = "exploratory_v4"


class RejectionReason(StrEnum):
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    INSUFFICIENT_WINDOWS = "insufficient_windows"
    UNSTABLE_DIRECTION = "unstable_direction"
    CORRELATION_BELOW_THRESHOLD = "correlation_below_threshold"
    HIT_RATE_BELOW_THRESHOLD = "hit_rate_below_threshold"
    BENCHMARK_IMPROVEMENT_BELOW_THRESHOLD = "benchmark_improvement_below_threshold"
    MOVEMENT_DOES_NOT_COVER_COSTS = "movement_does_not_cover_costs"
    EXCESSIVE_DISPERSION = "excessive_dispersion"
    INCONSISTENT_SIGN = "inconsistent_sign"
    NON_DIRECTIONAL_PREDICTOR = "non_directional_predictor"
    NO_ACTIONABLE_BIN = "no_actionable_bin"
    VALIDATION_FAILED = "validation_failed"
    TEST_NOT_EXECUTED = "test_not_executed"
    TEST_REJECTED = "test_rejected"


class DiagnosticConclusion(StrEnum):
    SIGNAL_FOUND_BUT_NOT_ECONOMIC = "signal_found_but_not_economic"
    ECONOMIC_SIGNAL_BUT_UNSTABLE = "economic_signal_but_unstable"
    STABLE_SIGNAL_BELOW_HIT_RATE = "stable_signal_below_hit_rate"
    INSUFFICIENT_OBSERVATIONS_AFTER_CONDITIONING = (
        "insufficient_observations_after_conditioning"
    )
    THRESHOLDS_PROBABLY_OVERRESTRICTIVE = "thresholds_probably_overrestrictive"
    NO_DETECTABLE_RELATIONSHIP = "no_detectable_relationship"
    CONFIRMED_PREDICTIVE_EDGES_FOUND = "confirmed_predictive_edges_found"


@dataclass(frozen=True, slots=True)
class AuditedParameter:
    name: str
    value: str
    source: ParameterSource
    rationale: str
    contract_version: int = PREDICTIVE_EDGE_V4_CONTRACT_VERSION


@dataclass(frozen=True, slots=True)
class RequirementMargin:
    requirement: RejectionReason
    observed_value: Decimal
    required_value: Decimal
    absolute_margin: Decimal
    relative_margin: Decimal | None
    passed: bool


@dataclass(frozen=True, slots=True)
class EdgeDiagnostic:
    fingerprint: str
    classification: str
    failed_requirements: tuple[RejectionReason, ...]
    passed_requirements: tuple[RejectionReason, ...]
    primary_rejection_reason: RejectionReason | None
    margins: tuple[RequirementMargin, ...]
    usable_observations: int
    windows_evaluated: int
    windows_directionally_consistent: int
    stable_direction_ratio: Decimal
    weighted_correlation: Decimal | None
    median_correlation: Decimal | None
    worst_correlation: Decimal | None
    directional_hit_rate: Decimal | None
    strongest_directional_benchmark: Decimal | None
    hit_rate_improvement: Decimal | None
    gross_expected_movement: Decimal | None
    net_expected_movement: Decimal | None
    standard_error: Decimal | None
    correlation_dispersion: Decimal | None


@dataclass(frozen=True, slots=True)
class RankedNearMiss:
    rank: int
    diagnostic: EdgeDiagnostic
    normalized_missing_distance: Decimal


@dataclass(frozen=True, slots=True)
class ThresholdSensitivity:
    edge_fingerprint: str
    threshold_name: RejectionReason
    current_value: Decimal
    nearest_passing_value: Decimal
    candidate_would_advance: bool
    other_requirements_still_failed: tuple[RejectionReason, ...]


@dataclass(frozen=True, slots=True)
class RedundancyEvidence:
    left_predictor: str
    right_predictor: str
    training_correlation: Decimal
    redundancy_group: str


@dataclass(frozen=True, slots=True)
class DiscoveryConfiguration:
    cadence: timedelta
    horizon: timedelta
    bins: int
    condition: Condition
    neutral_policy: NeutralBandPolicy
    profile: EvidenceProfile
    neutral_threshold: Decimal
    thresholds: EvidenceThresholds

    @property
    def fingerprint(self) -> str:
        payload = asdict(self) | {
            "contract_version": PREDICTIVE_EDGE_V4_CONTRACT_VERSION
        }
        encoded = json.dumps(
            payload, default=str, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode()).hexdigest()


def neutral_threshold_for_training(
    *,
    policy: NeutralBandPolicy,
    training_absolute_returns: tuple[Decimal, ...],
    fee_rate: Decimal,
    slippage_rate: Decimal,
) -> Decimal:
    """Freeze a neutral threshold using training evidence only."""

    if policy is NeutralBandPolicy.ZERO_DIRECTION:
        return Decimal(0)
    if policy is NeutralBandPolicy.ECONOMIC_FRICTION:
        return Decimal(2) * (fee_rate + slippage_rate)
    if not training_absolute_returns:
        return Decimal(0)
    ordered = sorted(abs(value) for value in training_absolute_returns)
    # Versioned lower-quartile nearest-rank policy; validation/test cannot alter it.
    return ordered[max(0, (len(ordered) - 1) // 4)]


def requirement_margin(
    requirement: RejectionReason, observed: Decimal, required: Decimal
) -> RequirementMargin:
    absolute = observed - required
    relative = None if required == 0 else absolute / abs(required)
    return RequirementMargin(
        requirement, observed, required, absolute, relative, observed >= required
    )


def diagnose_stability_edge(
    edge: StabilityEdge,
    *,
    thresholds: EvidenceThresholds,
    round_trip_friction: Decimal,
    profile: EvidenceProfile,
) -> EdgeDiagnostic:
    """Describe every v4 requirement without consulting isolated test evidence."""

    if edge.predictor.semantic_kind in (
        PredictorSemanticKind.MAGNITUDE_OR_ACTIVITY,
        PredictorSemanticKind.CONFIRMATION_ONLY,
    ):
        failed: tuple[RejectionReason, ...] = (
            RejectionReason.NON_DIRECTIONAL_PREDICTOR,
        )
        margins: tuple[RequirementMargin, ...] = ()
    else:
        correlation = abs(edge.weighted_mean_correlation or Decimal(0))
        hit = edge.mean_directional_hit_rate or Decimal(0)
        net = edge.mean_strategy_aligned_net_return
        if net is None:
            net = edge.mean_expected_avoided_downside or Decimal(0)
        margins = (
            requirement_margin(
                RejectionReason.INSUFFICIENT_OBSERVATIONS,
                Decimal(edge.total_usable_observations),
                Decimal(thresholds.minimum_observations),
            ),
            requirement_margin(
                RejectionReason.INSUFFICIENT_WINDOWS,
                Decimal(edge.windows_evaluated),
                Decimal(thresholds.minimum_windows),
            ),
            requirement_margin(
                RejectionReason.UNSTABLE_DIRECTION,
                edge.stable_direction_ratio,
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
                RejectionReason.MOVEMENT_DOES_NOT_COVER_COSTS,
                net,
                round_trip_friction + ECONOMIC_EDGE_MARGIN,
            ),
        )
        failed = tuple(item.requirement for item in margins if not item.passed)
    passed = tuple(item.requirement for item in margins if item.passed)
    fingerprint = hashlib.sha256(
        "|".join(
            (
                edge.predictor.name,
                str(edge.horizon.total_seconds()),
                edge.condition.value,
                str(edge.bin_ordinal),
                edge.suggested_action.value,
                profile.value,
                str(PREDICTIVE_EDGE_V4_CONTRACT_VERSION),
            )
        ).encode()
    ).hexdigest()
    return EdgeDiagnostic(
        fingerprint=fingerprint,
        classification=edge.classification.value,
        failed_requirements=failed,
        passed_requirements=passed,
        primary_rejection_reason=failed[0] if failed else None,
        margins=margins,
        usable_observations=edge.total_usable_observations,
        windows_evaluated=edge.windows_evaluated,
        windows_directionally_consistent=edge.windows_selected_for_test,
        stable_direction_ratio=edge.stable_direction_ratio,
        weighted_correlation=edge.weighted_mean_correlation,
        median_correlation=edge.median_correlation,
        worst_correlation=edge.worst_correlation,
        directional_hit_rate=edge.mean_directional_hit_rate,
        strongest_directional_benchmark=None,
        hit_rate_improvement=None,
        gross_expected_movement=edge.mean_strategy_aligned_gross_return,
        net_expected_movement=(
            edge.mean_strategy_aligned_net_return
            if edge.mean_strategy_aligned_net_return is not None
            else edge.mean_expected_avoided_downside
        ),
        standard_error=None,
        correlation_dispersion=edge.correlation_dispersion,
    )


def rank_near_misses(
    diagnostics: tuple[EdgeDiagnostic, ...], *, limit: int = NEAR_MISS_LIMIT
) -> tuple[RankedNearMiss, ...]:
    """Rank training/validation failures deterministically; test is not an input."""

    def distance(value: EdgeDiagnostic) -> Decimal:
        result = Decimal(0)
        for item in value.margins:
            if not item.passed:
                result += (
                    -item.relative_margin
                    if item.relative_margin is not None
                    else -item.absolute_margin
                )
        return result

    eligible = tuple(item for item in diagnostics if item.failed_requirements)
    ordered = sorted(
        eligible,
        key=lambda item: (
            -len(item.passed_requirements),
            distance(item),
            -(item.net_expected_movement or Decimal(0)),
            -item.stable_direction_ratio,
            -item.usable_observations,
            item.fingerprint,
        ),
    )[:limit]
    return tuple(
        RankedNearMiss(index + 1, item, distance(item))
        for index, item in enumerate(ordered)
    )


def threshold_sensitivity(
    near_misses: tuple[RankedNearMiss, ...],
) -> tuple[ThresholdSensitivity, ...]:
    """Move exactly one threshold to its observed value for each top near miss."""

    results: list[ThresholdSensitivity] = []
    for near_miss in near_misses[:SENSITIVITY_LIMIT]:
        failed = near_miss.diagnostic.failed_requirements
        for margin in near_miss.diagnostic.margins:
            if margin.passed:
                continue
            others = tuple(item for item in failed if item is not margin.requirement)
            results.append(
                ThresholdSensitivity(
                    near_miss.diagnostic.fingerprint,
                    margin.requirement,
                    margin.required_value,
                    margin.observed_value,
                    not others,
                    others,
                )
            )
    return tuple(results)


def diversified_relationships(
    ranked: tuple[tuple[PredictorDefinition, str, Decimal], ...],
    *,
    limit: int = CHEAP_RELATIONSHIP_LIMIT,
) -> tuple[tuple[PredictorDefinition, str, Decimal], ...]:
    """Round-robin redundancy groups so equivalent predictors cannot own top-K."""

    groups: dict[str, list[tuple[PredictorDefinition, str, Decimal]]] = {}
    for item in sorted(ranked, key=lambda value: (-value[2], value[0].name, value[1])):
        groups.setdefault(item[1], []).append(item)
    selected: list[tuple[PredictorDefinition, str, Decimal]] = []
    names = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for name in names:
            if groups[name] and len(selected) < limit:
                selected.append(groups[name].pop(0))
    return tuple(selected)


def diagnostic_conclusion(
    diagnostics: tuple[EdgeDiagnostic, ...], *, confirmed_count: int
) -> DiagnosticConclusion:
    if confirmed_count:
        return DiagnosticConclusion.CONFIRMED_PREDICTIVE_EDGES_FOUND
    counts: dict[RejectionReason, int] = {}
    for item in diagnostics:
        for reason in item.failed_requirements:
            counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return DiagnosticConclusion.NO_DETECTABLE_RELATIONSHIP
    dominant, dominant_count = max(
        counts.items(), key=lambda item: (item[1], item[0].value)
    )
    majority = dominant_count * 2 >= len(diagnostics)
    statistical = {
        RejectionReason.CORRELATION_BELOW_THRESHOLD,
        RejectionReason.HIT_RATE_BELOW_THRESHOLD,
        RejectionReason.BENCHMARK_IMPROVEMENT_BELOW_THRESHOLD,
    }
    economics_only = tuple(
        item
        for item in diagnostics
        if item.failed_requirements
        and set(item.failed_requirements)
        <= {RejectionReason.MOVEMENT_DOES_NOT_COVER_COSTS}
    )
    if economics_only and len(economics_only) * 2 >= len(diagnostics):
        return DiagnosticConclusion.SIGNAL_FOUND_BUT_NOT_ECONOMIC
    if majority and dominant in {
        RejectionReason.UNSTABLE_DIRECTION,
        RejectionReason.INCONSISTENT_SIGN,
    }:
        return DiagnosticConclusion.ECONOMIC_SIGNAL_BUT_UNSTABLE
    if majority and dominant in statistical:
        return DiagnosticConclusion.STABLE_SIGNAL_BELOW_HIT_RATE
    if majority and dominant is RejectionReason.INSUFFICIENT_OBSERVATIONS:
        return DiagnosticConclusion.INSUFFICIENT_OBSERVATIONS_AFTER_CONDITIONING
    if all(len(item.failed_requirements) == 1 for item in diagnostics):
        return DiagnosticConclusion.THRESHOLDS_PROBABLY_OVERRESTRICTIVE
    return DiagnosticConclusion.NO_DETECTABLE_RELATIONSHIP


def audited_parameters(
    configuration: DiscoveryConfiguration,
) -> tuple[AuditedParameter, ...]:
    fixed = {
        "decision_cadence": configuration.cadence,
        "horizon": configuration.horizon,
        "bin_count": configuration.bins,
        "condition": configuration.condition.value,
        "neutral_policy": configuration.neutral_policy.value,
        "profile": configuration.profile.value,
    }
    calibrated = {
        "neutral_threshold": configuration.neutral_threshold,
        "minimum_observations": configuration.thresholds.minimum_observations,
        "minimum_stable_direction_ratio": (
            configuration.thresholds.minimum_stable_direction_ratio
        ),
        "minimum_absolute_correlation": (
            configuration.thresholds.minimum_absolute_correlation
        ),
        "minimum_directional_hit_rate": (
            configuration.thresholds.minimum_directional_hit_rate
        ),
        "minimum_improvement_over_benchmark": (
            configuration.thresholds.minimum_hit_rate_improvement_over_benchmark
        ),
    }
    return tuple(
        AuditedParameter(
            name,
            str(value),
            ParameterSource.FIXED_DEFAULT,
            "versioned bounded discovery surface",
        )
        for name, value in fixed.items()
    ) + tuple(
        AuditedParameter(
            name,
            str(value),
            ParameterSource.TRAINING_CALIBRATED,
            "frozen before validation and isolated test",
        )
        for name, value in calibrated.items()
    )
