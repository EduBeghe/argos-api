import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import timedelta
from decimal import Decimal
from enum import StrEnum
from time import perf_counter

from app.optimization.edge_guided import (
    CandidateFamily,
    EdgeGuidedCandidate,
    EdgeGuidedCandidateEvaluation,
    EdgeGuidedCandidateStatus,
    EdgeGuidedEvaluationResult,
    EdgeGuidedSegmentEvaluation,
    EntryRule,
    ExitRule,
    _aggregate_values,
    evaluate_edge_guided_candidates,
)
from app.optimization.feature_matrix import HistoricalFeatureMatrix
from app.optimization.historical_data import HistoricalDataset
from app.optimization.predictive_edge_engine_v41 import (
    DIRECTIONAL_BIN_COUNT,
    PredictiveEdgeV41Result,
    V41Candidate,
    _sampled_pairs,
    build_derived_feature_rows,
)
from app.optimization.predictive_edges import (
    Condition,
    EvidenceThresholds,
    SuggestedAction,
)
from app.optimization.refinement import MAX_VALIDATION_DRAWDOWN, RefinementSegmentName
from app.optimization.walk_forward import WalkForwardWindow
from app.paper.config import SimulationConfiguration
from app.scores.calculation import MODEL_NAME, MODEL_VERSION
from app.strategies.baseline_long_only import (
    CONFIGURATION_VERSION,
    STRATEGY_NAME,
    STRATEGY_VERSION,
)

TARGETED_REFINEMENT_CONTRACT_VERSION = 1
MAX_TARGETED_SEEDS = 8
MAX_TARGETED_CONFIGURATIONS = 120
MINIMUM_SEED_OBSERVATIONS = 60
MINIMUM_SEED_WINDOWS = 3
TARGETED_CADENCES = tuple(timedelta(hours=value) for value in (4, 6, 8, 12))
TARGETED_HORIZONS = tuple(timedelta(hours=value) for value in (12, 24, 48))


class TargetedRefinementConclusion(StrEnum):
    NO_ELIGIBLE_SEEDS = "no_eligible_seeds"
    REJECTED_TRAINING = "candidates_rejected_training"
    REJECTED_VALIDATION = "candidates_rejected_validation"
    REJECTED_TEST = "candidates_rejected_test"
    CONFIRMED = "confirmed_candidates_found"


class TargetedCandidateStatus(StrEnum):
    REJECTED_TRAINING = "rejected_training"
    REJECTED_VALIDATION = "rejected_validation"
    REJECTED_TEST = "rejected_test"
    CONFIRMED = "confirmed_candidate"


@dataclass(frozen=True, slots=True)
class TargetedSeed:
    candidate: V41Candidate
    normalized_missing_distance: Decimal


@dataclass(frozen=True, slots=True)
class TargetedConfiguration:
    fingerprint: str
    seed_fingerprint: str
    cadence: timedelta
    horizon: timedelta
    bin_ordinal: int
    directional_lower_bound: Decimal | None
    directional_upper_bound: Decimal | None


@dataclass(frozen=True, slots=True)
class TargetedCandidateResult:
    configuration: TargetedConfiguration
    candidate: EdgeGuidedCandidate
    status: TargetedCandidateStatus
    evaluation: EdgeGuidedCandidateEvaluation | None
    training: EdgeGuidedSegmentEvaluation | None
    validation: EdgeGuidedSegmentEvaluation | None
    test: EdgeGuidedSegmentEvaluation | None
    passed_requirements: tuple[str, ...]
    failed_requirements: tuple[str, ...]
    requirement_margins: tuple["TargetedRequirementMargin", ...]
    rejection_reason: str | None
    normalized_distance_to_pass: Decimal


@dataclass(frozen=True, slots=True)
class TargetedRequirementMargin:
    requirement: str
    observed_value: Decimal
    required_value: Decimal
    absolute_margin: Decimal
    relative_margin: Decimal | None
    passed: bool


@dataclass(frozen=True, slots=True)
class TargetedRefinementSummary:
    seeds_selected: int
    configurations_generated: int
    duplicates_removed: int
    training_evaluated: int
    training_retained: int
    validation_evaluated: int
    validation_retained: int
    test_evaluated: int
    confirmed_candidates: int
    evaluations_computed: int
    evaluations_reused: int
    elapsed_seconds: Decimal


@dataclass(frozen=True, slots=True)
class TargetedRefinementResult:
    conclusion: TargetedRefinementConclusion
    seeds: tuple[TargetedSeed, ...]
    configurations: tuple[TargetedConfiguration, ...]
    candidates: tuple[TargetedCandidateResult, ...]
    summary: TargetedRefinementSummary
    edge_guided_result: EdgeGuidedEvaluationResult | None = None


def _missing_distance(candidate: V41Candidate) -> Decimal:
    evidence = candidate.validation
    margins = (
        max(
            Decimal(0),
            Decimal(MINIMUM_SEED_OBSERVATIONS - evidence.observations)
            / Decimal(MINIMUM_SEED_OBSERVATIONS),
        ),
        max(
            Decimal(0),
            Decimal(MINIMUM_SEED_WINDOWS - evidence.usable_window_count)
            / Decimal(MINIMUM_SEED_WINDOWS),
        ),
        Decimal(0) if evidence.stable else Decimal(1),
        Decimal(0)
        if evidence.net_movement is not None and evidence.net_movement > 0
        else Decimal(1),
        Decimal(0)
        if evidence.benchmark_improvement is not None
        and evidence.benchmark_improvement > 0
        else Decimal(1),
    )
    return sum(margins, Decimal(0)) / Decimal(len(margins))


def select_targeted_seeds(
    result: PredictiveEdgeV41Result,
    *,
    maximum_seeds: int = MAX_TARGETED_SEEDS,
) -> tuple[TargetedSeed, ...]:
    """Select bounded actionable evidence and reject apparent tiny-sample returns."""

    eligible: list[TargetedSeed] = []
    for candidate in result.actionable_near_misses:
        relationship = candidate.relationship
        validation = candidate.validation
        if (
            candidate.confirmed
            or validation.observations < MINIMUM_SEED_OBSERVATIONS
            or validation.usable_window_count < MINIMUM_SEED_WINDOWS
            or validation.net_movement is None
            or validation.net_movement <= 0
            or not relationship.directional_predictor
            or not relationship.context_predictor
            or not relationship.regime.value
            or relationship.action is SuggestedAction.DIAGNOSTIC_ONLY
        ):
            continue
        eligible.append(TargetedSeed(candidate, _missing_distance(candidate)))
    eligible.sort(
        key=lambda item: (
            item.normalized_missing_distance,
            -(item.candidate.validation.net_movement or Decimal(0)),
            -item.candidate.validation.observations,
            item.candidate.relationship.fingerprint,
        )
    )
    return tuple(eligible[:maximum_seeds])


def _ordered_unique(values: tuple[timedelta, ...]) -> tuple[timedelta, ...]:
    return tuple(sorted(set(values), key=lambda value: value.total_seconds()))


def _configuration_fingerprint(
    seed: V41Candidate, cadence: timedelta, horizon: timedelta, bin_ordinal: int
) -> str:
    relationship = seed.relationship
    payload = {
        "contract_version": TARGETED_REFINEMENT_CONTRACT_VERSION,
        "seed": relationship.fingerprint,
        "directional": [
            relationship.directional_predictor,
            relationship.directional_version,
        ],
        "context": [relationship.context_predictor, relationship.context_version],
        "regime": relationship.regime.value,
        "regime_bounds": [
            str(relationship.frozen_regime.lower_bound),
            str(relationship.frozen_regime.upper_bound),
        ],
        "mapping": relationship.direction_mapping.value,
        "action": relationship.action.value,
        "cadence_seconds": int(cadence.total_seconds()),
        "horizon_seconds": int(horizon.total_seconds()),
        "bin": bin_ordinal,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def generate_targeted_configurations(
    seeds: tuple[TargetedSeed, ...],
    *,
    maximum_configurations: int = MAX_TARGETED_CONFIGURATIONS,
    training_matrix: HistoricalFeatureMatrix | None = None,
    windows: tuple[WalkForwardWindow, ...] = (),
) -> tuple[tuple[TargetedConfiguration, ...], int]:
    """Build the deterministic local cadence/horizon/adjacent-bin neighborhood."""

    generated: list[TargetedConfiguration] = []
    seen: set[str] = set()
    effective_seen: set[tuple[object, ...]] = set()
    duplicates = 0
    for seed in sorted(
        seeds,
        key=lambda item: (
            item.normalized_missing_distance,
            -(item.candidate.validation.net_movement or Decimal(0)),
            -item.candidate.validation.observations,
            item.candidate.relationship.fingerprint,
        ),
    ):
        relationship = seed.candidate.relationship
        cadences = _ordered_unique((*TARGETED_CADENCES, relationship.cadence))
        horizons = _ordered_unique((*TARGETED_HORIZONS, relationship.horizon))
        bins = tuple(
            value
            for value in range(
                relationship.selected_bin - 1, relationship.selected_bin + 2
            )
            if 1 <= value <= DIRECTIONAL_BIN_COUNT
        )
        for cadence in cadences:
            for horizon in horizons:
                for bin_ordinal in bins:
                    lower = relationship.directional_lower_bound
                    upper = relationship.directional_upper_bound
                    if training_matrix is not None and windows:
                        lower, upper = _training_bin_bounds(
                            training_matrix,
                            windows,
                            relationship.directional_predictor,
                            cadence,
                            horizon,
                            bin_ordinal,
                        )
                    effective_identity = (
                        relationship.fingerprint,
                        cadence,
                        horizon,
                        bin_ordinal if training_matrix is None else None,
                        lower,
                        upper,
                        relationship.direction_mapping,
                        relationship.action,
                    )
                    if effective_identity in effective_seen:
                        duplicates += 1
                        continue
                    effective_seen.add(effective_identity)
                    fingerprint = _configuration_fingerprint(
                        seed.candidate, cadence, horizon, bin_ordinal
                    )
                    if fingerprint in seen:
                        duplicates += 1
                        continue
                    seen.add(fingerprint)
                    generated.append(
                        TargetedConfiguration(
                            fingerprint,
                            relationship.fingerprint,
                            cadence,
                            horizon,
                            bin_ordinal,
                            lower,
                            upper,
                        )
                    )
                    if len(generated) == maximum_configurations:
                        return tuple(generated), duplicates
    return tuple(generated), duplicates


def _training_bin_bounds(
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    predictor: str,
    cadence: timedelta,
    horizon: timedelta,
    bin_ordinal: int,
) -> tuple[Decimal | None, Decimal | None]:
    values = {
        item.candle_open_time: item.values
        for item in build_derived_feature_rows(matrix)
    }
    ordered = sorted(
        value
        for window in windows
        for _, value, _ in _sampled_pairs(
            values,
            matrix,
            window,
            "training",
            predictor,
            cadence,
            horizon,
        )
    )
    if not ordered:
        return None, None
    cuts = (ordered[len(ordered) // 3], ordered[len(ordered) * 2 // 3])
    return (
        None if bin_ordinal == 1 else cuts[min(bin_ordinal - 2, 1)],
        None if bin_ordinal >= 3 else cuts[bin_ordinal - 1],
    )


def _edge_candidate(
    configuration: TargetedConfiguration,
    seed: TargetedSeed,
    simulation: SimulationConfiguration,
) -> EdgeGuidedCandidate:
    relationship = seed.candidate.relationship
    is_entry = relationship.action is SuggestedAction.LONG_ENTRY
    return EdgeGuidedCandidate(
        candidate_fingerprint=configuration.fingerprint,
        source_edge_fingerprint=relationship.fingerprint,
        source_hypothesis_id=f"targeted:{relationship.fingerprint}",
        predictor_name=relationship.directional_predictor,
        predictor_version=relationship.directional_version,
        hypothesis_contract_version=TARGETED_REFINEMENT_CONTRACT_VERSION,
        condition=Condition.ALL,
        lower_bound=configuration.directional_lower_bound,
        upper_bound=configuration.directional_upper_bound,
        suggested_action=relationship.action,
        horizon=configuration.horizon,
        decision_interval=configuration.cadence,
        direction_mapping=relationship.direction_mapping,
        score_model_name=MODEL_NAME,
        score_model_version=MODEL_VERSION,
        strategy_name=STRATEGY_NAME,
        strategy_version=STRATEGY_VERSION,
        strategy_configuration_version=CONFIGURATION_VERSION,
        account_currency=simulation.account_currency,
        fee_rate=simulation.fee_rate,
        slippage_rate=simulation.slippage_rate,
        candidate_family=(
            CandidateFamily.EDGE_ENTRY_BASELINE_EXIT
            if is_entry
            else CandidateFamily.EXIT_OR_AVOID_OVERLAY
        ),
        entry_rule=EntryRule.EDGE_MATCH if is_entry else EntryRule.PRODUCTION_BASELINE,
        exit_rule=ExitRule.PRODUCTION_BASELINE
        if is_entry
        else ExitRule.DEFENSIVE_OVERLAY,
        cadence=configuration.cadence,
        minimum_holding_period=configuration.horizon,
        maximum_holding_period=configuration.horizon,
        cooldown=configuration.cadence,
        context_predictor_name=relationship.context_predictor,
        context_predictor_version=relationship.context_version,
        context_regime=relationship.regime.value,
        context_lower_bound=relationship.frozen_regime.lower_bound,
        context_upper_bound=relationship.frozen_regime.upper_bound,
    )


def _status(
    evaluation: EdgeGuidedCandidateEvaluation | None, has_validation: bool
) -> TargetedCandidateStatus:
    if evaluation is None:
        return (
            TargetedCandidateStatus.REJECTED_VALIDATION
            if has_validation
            else TargetedCandidateStatus.REJECTED_TRAINING
        )
    if evaluation.status is EdgeGuidedCandidateStatus.CONFIRMED:
        return TargetedCandidateStatus.CONFIRMED
    if evaluation.status is EdgeGuidedCandidateStatus.REJECTED_TEST:
        return TargetedCandidateStatus.REJECTED_TEST
    return TargetedCandidateStatus.REJECTED_VALIDATION


def _minimum_requirement(
    name: str, observed: Decimal, required: Decimal, *, strict: bool = False
) -> TargetedRequirementMargin:
    passed = observed > required if strict else observed >= required
    margin = observed - required
    relative = margin / abs(required) if required != 0 else None
    return TargetedRequirementMargin(name, observed, required, margin, relative, passed)


def _maximum_requirement(
    name: str, observed: Decimal, required: Decimal
) -> TargetedRequirementMargin:
    margin = required - observed
    relative = margin / abs(required) if required != 0 else None
    return TargetedRequirementMargin(
        name, observed, required, margin, relative, observed <= required
    )


def _stage_requirements(
    *,
    training: EdgeGuidedSegmentEvaluation | None,
    validation: EdgeGuidedSegmentEvaluation | None,
    test: EdgeGuidedSegmentEvaluation | None,
    validation_window_count: int,
    confirmed_window_count: int,
    confirmation_ratio: Decimal,
    thresholds: EvidenceThresholds,
) -> tuple[TargetedRequirementMargin, ...]:
    requirements: list[TargetedRequirementMargin] = []
    for phase, metrics in (
        ("training", training),
        ("validation", validation),
        ("isolated_test", test),
    ):
        if metrics is None:
            continue
        requirements.extend(
            (
                _minimum_requirement(
                    f"{phase}.sufficient_trades",
                    Decimal(metrics.trade_count),
                    Decimal(1),
                ),
                _minimum_requirement(
                    f"{phase}.positive_net_return",
                    metrics.total_return,
                    Decimal(0),
                    strict=True,
                ),
                _maximum_requirement(
                    f"{phase}.acceptable_drawdown",
                    metrics.max_drawdown,
                    MAX_VALIDATION_DRAWDOWN,
                ),
            )
        )
        if phase != "training":
            requirements.append(
                _minimum_requirement(
                    f"{phase}.beats_production_benchmark",
                    metrics.return_difference_from_production,
                    Decimal(0),
                    strict=True,
                )
            )
    if validation is not None:
        requirements.append(
            _minimum_requirement(
                "validation.minimum_windows",
                Decimal(validation_window_count),
                Decimal(thresholds.minimum_windows),
            )
        )
    if test is not None:
        requirements.extend(
            (
                _minimum_requirement(
                    "isolated_test.minimum_confirmed_windows",
                    Decimal(confirmed_window_count),
                    Decimal(thresholds.minimum_windows),
                ),
                _minimum_requirement(
                    "isolated_test.stable_confirmation_ratio",
                    confirmation_ratio,
                    thresholds.minimum_stable_direction_ratio,
                ),
            )
        )
    return tuple(requirements)


def _normalized_distance(
    requirements: tuple[TargetedRequirementMargin, ...],
) -> Decimal:
    distance = Decimal(0)
    for requirement in requirements:
        if requirement.passed:
            continue
        deficit = max(-requirement.absolute_margin, Decimal("0.000000000000000001"))
        distance += (
            deficit / abs(requirement.required_value)
            if requirement.required_value != 0
            else deficit
        )
    return distance


def _merge_evaluations(
    groups: tuple[EdgeGuidedEvaluationResult, ...],
    generated_count: int,
    windows_available: int,
) -> EdgeGuidedEvaluationResult:
    if not groups:
        raise ValueError("at least one cadence group is required")
    evaluations = tuple(item for group in groups for item in group.evaluations)
    stage_evidence = tuple(item for group in groups for item in group.stage_evidence)
    funnel_keys = {key for group in groups for key in group.funnel}
    funnel = {
        key: sum(group.funnel.get(key, 0) for group in groups) for key in funnel_keys
    }
    return replace(
        groups[0],
        generated_count=generated_count,
        evaluated_count=len(evaluations),
        maximum_candidates=MAX_TARGETED_CONFIGURATIONS,
        windows_available=windows_available,
        evaluations=evaluations,
        funnel=funnel,
        stage_evidence=stage_evidence,
    )


def evaluate_targeted_refinement(
    *,
    regime_result: PredictiveEdgeV41Result,
    dataset: HistoricalDataset,
    feature_matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    initial_cash: Decimal,
    simulation: SimulationConfiguration,
    thresholds: EvidenceThresholds,
) -> TargetedRefinementResult:
    """Evaluate frozen local configurations with the walk-forward simulator."""

    started = perf_counter()
    seeds = select_targeted_seeds(regime_result)
    configurations, duplicates = generate_targeted_configurations(
        seeds, training_matrix=feature_matrix, windows=windows
    )
    if not configurations:
        summary = TargetedRefinementSummary(
            0,
            0,
            duplicates,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            Decimal(str(perf_counter() - started)),
        )
        return TargetedRefinementResult(
            TargetedRefinementConclusion.NO_ELIGIBLE_SEEDS,
            seeds,
            configurations,
            (),
            summary,
        )
    seed_by_fingerprint = {
        item.candidate.relationship.fingerprint: item for item in seeds
    }
    frozen = tuple(
        _edge_candidate(item, seed_by_fingerprint[item.seed_fingerprint], simulation)
        for item in configurations
    )
    cadence_groups: dict[timedelta, list[EdgeGuidedCandidate]] = defaultdict(list)
    for candidate in frozen:
        cadence_groups[candidate.decision_interval].append(candidate)
    grouped_results = tuple(
        evaluate_edge_guided_candidates(
            dataset=dataset,
            feature_matrix=feature_matrix,
            hypotheses=(),
            frozen_candidates=tuple(cadence_groups[cadence]),
            initial_cash=initial_cash,
            simulation=simulation,
            decision_interval=cadence,
            maximum_candidates=len(cadence_groups[cadence]),
            maximum_test_candidates=len(cadence_groups[cadence]),
            windows=windows,
            thresholds=thresholds,
        )
        for cadence in sorted(cadence_groups, key=timedelta.total_seconds)
    )
    evaluated = _merge_evaluations(grouped_results, len(configurations), len(windows))
    complete = {
        item.candidate.candidate_fingerprint: item for item in evaluated.evaluations
    }
    validation_ids = {
        item.candidate_fingerprint
        for item in evaluated.stage_evidence
        if item.stage.value == "validation"
    }
    evidence_by_candidate: dict[
        tuple[str, RefinementSegmentName], list[EdgeGuidedSegmentEvaluation]
    ] = defaultdict(list)
    for evidence in evaluated.stage_evidence:
        evidence_by_candidate[(evidence.candidate_fingerprint, evidence.stage)].append(
            evidence.metrics
        )
    results: list[TargetedCandidateResult] = []
    for configuration, candidate in zip(configurations, frozen, strict=True):
        evaluation = complete.get(candidate.candidate_fingerprint)
        status = _status(evaluation, candidate.candidate_fingerprint in validation_ids)
        stages = {
            stage: _aggregate_values(
                tuple(evidence_by_candidate[(candidate.candidate_fingerprint, stage)]),
                stage,
            )
            for stage in RefinementSegmentName
            if evidence_by_candidate[(candidate.candidate_fingerprint, stage)]
        }
        training = stages.get(RefinementSegmentName.TRAINING)
        validation = stages.get(RefinementSegmentName.VALIDATION)
        test = stages.get(RefinementSegmentName.TEST)
        valid_windows = (
            evaluation.valid_window_count
            if evaluation is not None
            else len(
                evidence_by_candidate[
                    (candidate.candidate_fingerprint, RefinementSegmentName.VALIDATION)
                ]
            )
        )
        confirmed_windows = (
            evaluation.confirmed_window_count if evaluation is not None else 0
        )
        confirmation_ratio = (
            Decimal(confirmed_windows) / Decimal(valid_windows)
            if valid_windows
            else Decimal(0)
        )
        margins = _stage_requirements(
            training=training,
            validation=validation,
            test=test,
            validation_window_count=valid_windows,
            confirmed_window_count=confirmed_windows,
            confirmation_ratio=confirmation_ratio,
            thresholds=thresholds,
        )
        passed = tuple(item.requirement for item in margins if item.passed)
        failed = tuple(item.requirement for item in margins if not item.passed)
        results.append(
            TargetedCandidateResult(
                configuration,
                candidate,
                status,
                evaluation,
                training,
                validation,
                test,
                passed,
                failed,
                margins,
                failed[0] if failed else None,
                _normalized_distance(margins),
            )
        )
    statuses = {item.status for item in results}
    conclusion = (
        TargetedRefinementConclusion.CONFIRMED
        if TargetedCandidateStatus.CONFIRMED in statuses
        else TargetedRefinementConclusion.REJECTED_TEST
        if TargetedCandidateStatus.REJECTED_TEST in statuses
        else TargetedRefinementConclusion.REJECTED_VALIDATION
        if TargetedCandidateStatus.REJECTED_VALIDATION in statuses
        else TargetedRefinementConclusion.REJECTED_TRAINING
    )
    funnel = evaluated.funnel
    summary = TargetedRefinementSummary(
        len(seeds),
        len(configurations),
        duplicates,
        funnel.get("training_evaluated_count", len(configurations)),
        funnel.get("training_retained_count", 0),
        funnel.get("validation_evaluated_count", 0),
        funnel.get("validation_retained_count", 0),
        funnel.get("test_evaluated_count", 0),
        sum(item.status is TargetedCandidateStatus.CONFIRMED for item in results),
        len(evaluated.stage_evidence),
        0,
        Decimal(str(perf_counter() - started)),
    )
    return TargetedRefinementResult(
        conclusion, seeds, configurations, tuple(results), summary, evaluated
    )
