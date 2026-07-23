import hashlib
import json
from dataclasses import dataclass
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
    EntryRule,
    ExitRule,
    evaluate_edge_guided_candidates,
)
from app.optimization.feature_matrix import HistoricalFeatureMatrix
from app.optimization.historical_data import HistoricalDataset
from app.optimization.predictive_edge_engine_v41 import (
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
    passed_requirements: tuple[str, ...]
    failed_requirements: tuple[str, ...]
    rejection_reason: str | None
    normalized_distance_to_pass: Decimal


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
            if value >= 1
        )
        for cadence in cadences:
            for horizon in horizons:
                for bin_ordinal in bins:
                    fingerprint = _configuration_fingerprint(
                        seed.candidate, cadence, horizon, bin_ordinal
                    )
                    if fingerprint in seen:
                        duplicates += 1
                        continue
                    seen.add(fingerprint)
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
    evaluated = evaluate_edge_guided_candidates(
        dataset=dataset,
        feature_matrix=feature_matrix,
        hypotheses=(),
        frozen_candidates=frozen,
        initial_cash=initial_cash,
        simulation=simulation,
        decision_interval=frozen[0].decision_interval,
        maximum_candidates=MAX_TARGETED_CONFIGURATIONS,
        maximum_test_candidates=MAX_TARGETED_CONFIGURATIONS,
        windows=windows,
        thresholds=thresholds,
    )
    complete = {
        item.candidate.candidate_fingerprint: item for item in evaluated.evaluations
    }
    validation_ids = {
        item.candidate_fingerprint
        for item in evaluated.stage_evidence
        if item.stage.value == "validation"
    }
    results: list[TargetedCandidateResult] = []
    for configuration, candidate in zip(configurations, frozen, strict=True):
        evaluation = complete.get(candidate.candidate_fingerprint)
        status = _status(evaluation, candidate.candidate_fingerprint in validation_ids)
        failed = (
            tuple(item.value for item in evaluation.rejection_reasons)
            if evaluation
            else (
                ("training_qualification",)
                if status is TargetedCandidateStatus.REJECTED_TRAINING
                else ("validation_qualification",)
            )
        )
        passed = tuple(
            name
            for name in ("training", "validation", "isolated_test")
            if name not in " ".join(failed)
        )
        results.append(
            TargetedCandidateResult(
                configuration,
                candidate,
                status,
                evaluation,
                passed,
                failed,
                failed[0] if failed else None,
                Decimal(len(failed)),
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
