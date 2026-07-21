import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import timedelta
from decimal import Decimal
from enum import StrEnum

from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.historical_data import HistoricalDataset
from app.optimization.predictive_edges import (
    Condition,
    DirectionMapping,
    EvidenceThresholds,
    StrategyHypothesis,
    SuggestedAction,
    predictive_edge_condition_matches,
    predictive_edge_value,
    predictive_edge_volatility_threshold,
)
from app.optimization.refinement import (
    RefinementSegmentName,
    simulate_buy_and_hold,
)
from app.optimization.simulation import (
    DEFAULT_DECISION_INTERVAL,
    DecisionActionFilter,
    simulate_historical_feature_matrix,
)
from app.optimization.walk_forward import WalkForwardWindow, build_window_matrices
from app.paper.config import SimulationConfiguration
from app.scores.calculation import MODEL_NAME, MODEL_VERSION
from app.strategies.baseline_long_only import (
    CONFIGURATION_VERSION,
    STRATEGY_NAME,
    STRATEGY_VERSION,
    StrategyAction,
)

DEFAULT_MAX_EDGE_GUIDED_CANDIDATES = 12
MAX_EDGE_GUIDED_CANDIDATES = 100


class EdgeGuidedCandidateError(ValueError):
    """Raised when confirmed edge evidence cannot form a bounded candidate."""


class EdgeGuidedCandidateStatus(StrEnum):
    REJECTED_VALIDATION = "rejected_validation"
    REJECTED_TEST = "rejected_test"
    CONFIRMED = "confirmed_candidate"


@dataclass(frozen=True, slots=True)
class EdgeGuidedCandidate:
    candidate_fingerprint: str
    source_edge_fingerprint: str
    source_hypothesis_id: str
    predictor_name: str
    predictor_version: int
    hypothesis_contract_version: int
    condition: Condition
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    suggested_action: SuggestedAction
    horizon: timedelta
    decision_interval: timedelta
    direction_mapping: DirectionMapping
    score_model_name: str
    score_model_version: int
    strategy_name: str
    strategy_version: int
    strategy_configuration_version: int
    account_currency: str
    fee_rate: Decimal
    slippage_rate: Decimal


@dataclass(frozen=True, slots=True)
class EdgeGuidedSegmentEvaluation:
    segment: RefinementSegmentName
    total_return: Decimal
    max_drawdown: Decimal
    trade_count: int
    fees_paid: Decimal
    return_difference_from_production: Decimal
    return_difference_from_buy_and_hold: Decimal


@dataclass(frozen=True, slots=True)
class EdgeGuidedCandidateEvaluation:
    candidate: EdgeGuidedCandidate
    training: EdgeGuidedSegmentEvaluation
    validation: EdgeGuidedSegmentEvaluation
    test: EdgeGuidedSegmentEvaluation
    windows: tuple["EdgeGuidedWindowEvaluation", ...]
    valid_window_count: int
    confirmed_window_count: int
    rejected_validation_window_count: int
    rejected_test_window_count: int
    unconditional_benchmark: str
    status: EdgeGuidedCandidateStatus
    explanation: str


@dataclass(frozen=True, slots=True)
class EdgeGuidedEvaluationResult:
    dataset_fingerprint: str
    generated_count: int
    evaluated_count: int
    maximum_candidates: int
    windows_available: int
    evaluations: tuple[EdgeGuidedCandidateEvaluation, ...]


@dataclass(frozen=True, slots=True)
class EdgeGuidedWindowEvaluation:
    window_ordinal: int
    volatility_threshold: Decimal
    training: EdgeGuidedSegmentEvaluation
    validation: EdgeGuidedSegmentEvaluation
    test: EdgeGuidedSegmentEvaluation | None
    validation_passed: bool
    test_confirmed: bool


def generate_edge_guided_candidates(
    *,
    hypotheses: tuple[StrategyHypothesis, ...],
    simulation: SimulationConfiguration,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
    maximum_candidates: int = DEFAULT_MAX_EDGE_GUIDED_CANDIDATES,
) -> tuple[EdgeGuidedCandidate, ...]:
    """Create at most one deterministic candidate per confirmed hypothesis."""

    if not 1 <= maximum_candidates <= MAX_EDGE_GUIDED_CANDIDATES:
        raise EdgeGuidedCandidateError(
            f"maximum_candidates must be between 1 and {MAX_EDGE_GUIDED_CANDIDATES}"
        )
    candidates: list[EdgeGuidedCandidate] = []
    for hypothesis in sorted(hypotheses, key=lambda item: item.hypothesis_id):
        if hypothesis.suggested_action not in (
            SuggestedAction.LONG_ENTRY,
            SuggestedAction.EXIT_OR_AVOID,
        ):
            continue
        if hypothesis.lower_bound is None and hypothesis.upper_bound is None:
            continue
        if (
            hypothesis.lower_bound is not None
            and hypothesis.upper_bound is not None
            and hypothesis.lower_bound >= hypothesis.upper_bound
        ):
            continue
        payload = {
            "source_hypothesis_id": hypothesis.hypothesis_id,
            "predictor_name": hypothesis.predictor_name,
            "predictor_version": hypothesis.predictor_version,
            "hypothesis_contract_version": hypothesis.hypothesis_contract_version,
            "condition": hypothesis.condition.value,
            "lower_bound": hypothesis.lower_bound,
            "upper_bound": hypothesis.upper_bound,
            "suggested_action": hypothesis.suggested_action.value,
            "horizon_seconds": int(hypothesis.horizon.total_seconds()),
            "decision_interval_seconds": int(decision_interval.total_seconds()),
            "direction_mapping": hypothesis.direction_mapping.value,
            "score_model_name": MODEL_NAME,
            "score_model_version": MODEL_VERSION,
            "strategy_name": STRATEGY_NAME,
            "strategy_version": STRATEGY_VERSION,
            "strategy_configuration_version": CONFIGURATION_VERSION,
            "account_currency": simulation.account_currency,
            "fee_rate": simulation.fee_rate,
            "slippage_rate": simulation.slippage_rate,
        }
        fingerprint = _fingerprint(payload)
        candidates.append(
            EdgeGuidedCandidate(
                fingerprint,
                hypothesis.hypothesis_id,
                hypothesis.hypothesis_id,
                hypothesis.predictor_name,
                hypothesis.predictor_version,
                hypothesis.hypothesis_contract_version,
                hypothesis.condition,
                hypothesis.lower_bound,
                hypothesis.upper_bound,
                hypothesis.suggested_action,
                hypothesis.horizon,
                decision_interval,
                hypothesis.direction_mapping,
                MODEL_NAME,
                MODEL_VERSION,
                STRATEGY_NAME,
                STRATEGY_VERSION,
                CONFIGURATION_VERSION,
                simulation.account_currency,
                simulation.fee_rate,
                simulation.slippage_rate,
            )
        )
    unique = {item.candidate_fingerprint: item for item in candidates}
    return tuple(sorted(unique.values(), key=lambda item: item.candidate_fingerprint))[
        :maximum_candidates
    ]


def apply_edge_guided_action(
    *,
    candidate: EdgeGuidedCandidate,
    row: HistoricalFeatureRow,
    position_is_open: bool,
    baseline_action: StrategyAction,
    volatility_threshold: Decimal,
) -> StrategyAction:
    """Apply one frozen bin rule without introducing short exposure."""

    value = predictive_edge_value(row, candidate.predictor_name)
    inside = (
        (candidate.lower_bound is None or value > candidate.lower_bound)
        and (candidate.upper_bound is None or value <= candidate.upper_bound)
        and predictive_edge_condition_matches(
            row,
            candidate.condition,
            volatility_threshold=volatility_threshold,
        )
    )
    if candidate.suggested_action is SuggestedAction.LONG_ENTRY:
        if position_is_open:
            return baseline_action
        if baseline_action is StrategyAction.OPEN_POSITION:
            return StrategyAction.OPEN_POSITION if inside else StrategyAction.NO_ACTION
        return baseline_action
    if candidate.suggested_action is SuggestedAction.EXIT_OR_AVOID and inside:
        return StrategyAction.SELL_ALL if position_is_open else StrategyAction.NO_ACTION
    return baseline_action


def _candidate_action_filter(
    candidate: EdgeGuidedCandidate,
    volatility_threshold: Decimal,
) -> DecisionActionFilter:
    def apply(
        row: HistoricalFeatureRow,
        position_is_open: bool,
        baseline_action: StrategyAction,
    ) -> StrategyAction:
        return apply_edge_guided_action(
            candidate=candidate,
            row=row,
            position_is_open=position_is_open,
            baseline_action=baseline_action,
            volatility_threshold=volatility_threshold,
        )

    return apply


def evaluate_edge_guided_candidates(
    *,
    dataset: HistoricalDataset,
    feature_matrix: HistoricalFeatureMatrix,
    hypotheses: tuple[StrategyHypothesis, ...],
    initial_cash: Decimal,
    simulation: SimulationConfiguration,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
    maximum_candidates: int = DEFAULT_MAX_EDGE_GUIDED_CANDIDATES,
    windows: tuple[WalkForwardWindow, ...],
    thresholds: EvidenceThresholds,
) -> EdgeGuidedEvaluationResult:
    """Evaluate frozen candidates through the existing isolated walk-forward windows."""

    candidates = generate_edge_guided_candidates(
        hypotheses=hypotheses,
        simulation=simulation,
        decision_interval=decision_interval,
        maximum_candidates=maximum_candidates,
    )
    evaluations: list[EdgeGuidedCandidateEvaluation] = []
    for candidate in candidates:
        window_results = _evaluate_candidate_windows(
            dataset=dataset,
            feature_matrix=feature_matrix,
            windows=windows,
            candidate=candidate,
            initial_cash=initial_cash,
            simulation=simulation,
            decision_interval=decision_interval,
        )
        valid = len(window_results)
        confirmed = sum(item.test_confirmed for item in window_results)
        rejected_validation = sum(not item.validation_passed for item in window_results)
        rejected_test = sum(
            item.validation_passed and not item.test_confirmed
            for item in window_results
        )
        ratio = Decimal(confirmed) / Decimal(valid) if valid else Decimal(0)
        if (
            valid >= thresholds.minimum_windows
            and confirmed >= thresholds.minimum_windows
            and ratio >= thresholds.minimum_stable_direction_ratio
        ):
            status = EdgeGuidedCandidateStatus.CONFIRMED
            explanation = (
                "Frozen candidate confirmed against production across the required "
                "isolated walk-forward test windows."
            )
        elif not any(item.validation_passed for item in window_results):
            status = EdgeGuidedCandidateStatus.REJECTED_VALIDATION
            explanation = (
                "Candidate did not pass validation in the required walk-forward "
                "windows."
            )
        else:
            status = EdgeGuidedCandidateStatus.REJECTED_TEST
            explanation = (
                "Frozen validation candidate did not satisfy the existing stability "
                "criteria across isolated test windows."
            )
        evaluations.append(
            EdgeGuidedCandidateEvaluation(
                candidate,
                _aggregate_segment(window_results, RefinementSegmentName.TRAINING),
                _aggregate_segment(window_results, RefinementSegmentName.VALIDATION),
                _aggregate_segment(window_results, RefinementSegmentName.TEST),
                window_results,
                valid,
                confirmed,
                rejected_validation,
                rejected_test,
                "always_up"
                if candidate.suggested_action is SuggestedAction.LONG_ENTRY
                else "always_down",
                status,
                explanation,
            )
        )
    return EdgeGuidedEvaluationResult(
        dataset.fingerprint,
        len(candidates),
        len(evaluations),
        maximum_candidates,
        len(windows),
        tuple(evaluations),
    )


def _evaluate_candidate_windows(
    *,
    dataset: HistoricalDataset,
    feature_matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    candidate: EdgeGuidedCandidate,
    initial_cash: Decimal,
    simulation: SimulationConfiguration,
    decision_interval: timedelta,
) -> tuple[EdgeGuidedWindowEvaluation, ...]:
    results: list[EdgeGuidedWindowEvaluation] = []
    for window in windows:
        matrices = build_window_matrices(
            feature_matrix, window, decision_interval=decision_interval
        )
        if not all(
            matrix.rows
            for matrix in (matrices.training, matrices.validation, matrices.test)
        ):
            continue
        volatility_threshold = predictive_edge_volatility_threshold(
            matrices.training.rows
        )
        action_filter = _candidate_action_filter(candidate, volatility_threshold)
        training = _evaluate_segment(
            dataset,
            matrices.training,
            RefinementSegmentName.TRAINING,
            candidate,
            initial_cash,
            simulation,
            action_filter,
        )
        validation = _evaluate_segment(
            dataset,
            matrices.validation,
            RefinementSegmentName.VALIDATION,
            candidate,
            initial_cash,
            simulation,
            action_filter,
        )
        validation_passed = validation.return_difference_from_production > 0
        test = (
            _evaluate_segment(
                dataset,
                matrices.test,
                RefinementSegmentName.TEST,
                candidate,
                initial_cash,
                simulation,
                action_filter,
            )
            if validation_passed
            else None
        )
        results.append(
            EdgeGuidedWindowEvaluation(
                window.ordinal,
                volatility_threshold,
                training,
                validation,
                test,
                validation_passed,
                test is not None and test.return_difference_from_production > 0,
            )
        )
    return tuple(results)


def _evaluate_segment(
    dataset: HistoricalDataset,
    matrix: HistoricalFeatureMatrix,
    segment: RefinementSegmentName,
    candidate: EdgeGuidedCandidate,
    initial_cash: Decimal,
    simulation: SimulationConfiguration,
    action_filter: DecisionActionFilter,
) -> EdgeGuidedSegmentEvaluation:
    production = simulate_historical_feature_matrix(
        dataset=dataset,
        feature_matrix=matrix,
        initial_cash=initial_cash,
        fee_rate=simulation.fee_rate,
        slippage_rate=simulation.slippage_rate,
        decision_interval=candidate.decision_interval,
    )
    buy_and_hold = simulate_buy_and_hold(
        matrix,
        segment,
        dataset.identity.instrument,
        initial_cash,
        simulation,
    )
    result = simulate_historical_feature_matrix(
        dataset=dataset,
        feature_matrix=matrix,
        initial_cash=initial_cash,
        fee_rate=candidate.fee_rate,
        slippage_rate=candidate.slippage_rate,
        decision_interval=candidate.decision_interval,
        decision_action_filter=action_filter,
    )
    return EdgeGuidedSegmentEvaluation(
        segment,
        result.total_return,
        result.max_drawdown,
        result.trade_count,
        result.fees_paid,
        result.total_return - production.total_return,
        result.total_return - buy_and_hold.total_return,
    )


def _aggregate_segment(
    windows: tuple[EdgeGuidedWindowEvaluation, ...],
    segment: RefinementSegmentName,
) -> EdgeGuidedSegmentEvaluation:
    values = tuple(
        value
        for item in windows
        for value in (
            item.training
            if segment is RefinementSegmentName.TRAINING
            else item.validation
            if segment is RefinementSegmentName.VALIDATION
            else item.test,
        )
        if value is not None
    )
    if not values:
        return EdgeGuidedSegmentEvaluation(
            segment, Decimal(0), Decimal(0), 0, Decimal(0), Decimal(0), Decimal(0)
        )
    count = Decimal(len(values))
    return EdgeGuidedSegmentEvaluation(
        segment,
        sum((item.total_return for item in values), Decimal(0)) / count,
        sum((item.max_drawdown for item in values), Decimal(0)) / count,
        sum(item.trade_count for item in values),
        sum((item.fees_paid for item in values), Decimal(0)),
        sum((item.return_difference_from_production for item in values), Decimal(0))
        / count,
        sum((item.return_difference_from_buy_and_hold for item in values), Decimal(0))
        / count,
    )


def edge_guided_summary(result: EdgeGuidedEvaluationResult) -> dict[str, object]:
    """Return a structured JSON-safe job summary with Decimal strings."""

    value = json.loads(json.dumps(asdict(result), default=str))
    if not isinstance(value, dict):
        raise EdgeGuidedCandidateError("edge-guided summary must be an object")
    return value


def _fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()
