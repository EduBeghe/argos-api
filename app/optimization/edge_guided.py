import hashlib
import json
from dataclasses import asdict, dataclass, field
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
    MAX_VALIDATION_DRAWDOWN,
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
EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION = 2
MINIMUM_TRADES_PER_SEGMENT = 1


class EdgeGuidedCandidateError(ValueError):
    """Raised when confirmed edge evidence cannot form a bounded candidate."""


class EdgeGuidedCandidateStatus(StrEnum):
    REJECTED_VALIDATION = "rejected_validation"
    REJECTED_TEST = "rejected_test"
    CONFIRMED = "confirmed_candidate"


class EdgeGuidedRejectionReason(StrEnum):
    INSUFFICIENT_TRADES = "insufficient_trades"
    NON_POSITIVE_TRAINING_RETURN = "non_positive_training_return"
    NON_POSITIVE_VALIDATION_RETURN = "non_positive_validation_return"
    NON_POSITIVE_TEST_RETURN = "non_positive_test_return"
    DID_NOT_BEAT_PRODUCTION_VALIDATION = "did_not_beat_production_validation"
    DID_NOT_BEAT_PRODUCTION_TEST = "did_not_beat_production_test"
    EXCESSIVE_DRAWDOWN = "excessive_drawdown"
    INSUFFICIENT_VALID_WINDOWS = "insufficient_valid_windows"
    INSUFFICIENT_CONFIRMED_WINDOWS = "insufficient_confirmed_windows"
    UNSTABLE_ACROSS_WINDOWS = "unstable_across_windows"


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
    qualification_contract_version: int = EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION


@dataclass(frozen=True, slots=True)
class EdgeGuidedSegmentEvaluation:
    segment: RefinementSegmentName
    total_return: Decimal
    max_drawdown: Decimal
    trade_count: int
    fees_paid: Decimal
    return_difference_from_production: Decimal
    return_difference_from_buy_and_hold: Decimal
    sufficient_trades: bool = field(init=False)
    positive_net_return: bool = field(init=False)
    beats_production: bool = field(init=False)
    beats_buy_and_hold: bool = field(init=False)
    risk_acceptable: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sufficient_trades", self.trade_count >= MINIMUM_TRADES_PER_SEGMENT
        )
        object.__setattr__(self, "positive_net_return", self.total_return > 0)
        object.__setattr__(
            self, "beats_production", self.return_difference_from_production > 0
        )
        object.__setattr__(
            self,
            "beats_buy_and_hold",
            self.return_difference_from_buy_and_hold > 0,
        )
        object.__setattr__(
            self, "risk_acceptable", self.max_drawdown <= MAX_VALIDATION_DRAWDOWN
        )


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
    rejection_reasons: tuple[EdgeGuidedRejectionReason, ...] = ()


@dataclass(frozen=True, slots=True)
class EdgeGuidedEvaluationResult:
    dataset_fingerprint: str
    generated_count: int
    evaluated_count: int
    maximum_candidates: int
    windows_available: int
    evaluations: tuple[EdgeGuidedCandidateEvaluation, ...]
    qualification_contract_version: int = EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION


@dataclass(frozen=True, slots=True)
class EdgeGuidedWindowEvaluation:
    window_ordinal: int
    volatility_threshold: Decimal
    training: EdgeGuidedSegmentEvaluation
    validation: EdgeGuidedSegmentEvaluation
    test: EdgeGuidedSegmentEvaluation | None
    validation_passed: bool
    test_confirmed: bool
    rejection_reasons: tuple[EdgeGuidedRejectionReason, ...] = ()


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
            "qualification_contract_version": (
                EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION
            ),
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
        valid = sum(item.validation_passed for item in window_results)
        rejected_validation = len(window_results) - valid
        rejected_test = sum(
            item.validation_passed and not item.test_confirmed
            for item in window_results
        )
        ratio = Decimal(confirmed) / Decimal(valid) if valid else Decimal(0)
        training = _aggregate_segment(window_results, RefinementSegmentName.TRAINING)
        validation = _aggregate_segment(
            window_results, RefinementSegmentName.VALIDATION
        )
        test = _aggregate_segment(window_results, RefinementSegmentName.TEST)
        reasons = _candidate_rejection_reasons(
            training=training,
            validation=validation,
            test=test,
            valid=valid,
            confirmed=confirmed,
            ratio=ratio,
            thresholds=thresholds,
        )
        if not reasons:
            status = EdgeGuidedCandidateStatus.CONFIRMED
            explanation = (
                "The frozen candidate had positive net returns, sufficient trading "
                "activity, acceptable risk, and strictly beat production across the "
                "required isolated walk-forward evidence. Buy-and-hold remains an "
                "independent comparison."
            )
        elif _validation_stage_failed(
            training=training,
            validation=validation,
            valid=valid,
            thresholds=thresholds,
        ):
            status = EdgeGuidedCandidateStatus.REJECTED_VALIDATION
        else:
            status = EdgeGuidedCandidateStatus.REJECTED_TEST
        if reasons:
            explanation = _rejection_explanation(reasons)
        evaluations.append(
            EdgeGuidedCandidateEvaluation(
                candidate,
                training,
                validation,
                test,
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
                reasons,
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
        validation_reasons = _window_segment_rejection_reasons(
            validation, RefinementSegmentName.VALIDATION
        )
        validation_passed = not validation_reasons
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
        test_confirmed = test is not None and not _window_segment_rejection_reasons(
            test, RefinementSegmentName.TEST
        )
        window_reasons = validation_reasons
        if test is not None:
            window_reasons = _stable_reasons(
                (
                    *window_reasons,
                    *_window_segment_rejection_reasons(
                        test, RefinementSegmentName.TEST
                    ),
                )
            )
        results.append(
            EdgeGuidedWindowEvaluation(
                window.ordinal,
                volatility_threshold,
                training,
                validation,
                test,
                validation_passed,
                test_confirmed,
                window_reasons,
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


def _window_segment_rejection_reasons(
    value: EdgeGuidedSegmentEvaluation, segment: RefinementSegmentName
) -> tuple[EdgeGuidedRejectionReason, ...]:
    reasons: list[EdgeGuidedRejectionReason] = []
    if not value.sufficient_trades:
        reasons.append(EdgeGuidedRejectionReason.INSUFFICIENT_TRADES)
    if not value.positive_net_return:
        reasons.append(
            EdgeGuidedRejectionReason.NON_POSITIVE_VALIDATION_RETURN
            if segment is RefinementSegmentName.VALIDATION
            else EdgeGuidedRejectionReason.NON_POSITIVE_TEST_RETURN
        )
    if not value.beats_production:
        reasons.append(
            EdgeGuidedRejectionReason.DID_NOT_BEAT_PRODUCTION_VALIDATION
            if segment is RefinementSegmentName.VALIDATION
            else EdgeGuidedRejectionReason.DID_NOT_BEAT_PRODUCTION_TEST
        )
    if not value.risk_acceptable:
        reasons.append(EdgeGuidedRejectionReason.EXCESSIVE_DRAWDOWN)
    return _stable_reasons(tuple(reasons))


def _candidate_rejection_reasons(
    *,
    training: EdgeGuidedSegmentEvaluation,
    validation: EdgeGuidedSegmentEvaluation,
    test: EdgeGuidedSegmentEvaluation,
    valid: int,
    confirmed: int,
    ratio: Decimal,
    thresholds: EvidenceThresholds,
) -> tuple[EdgeGuidedRejectionReason, ...]:
    reasons: list[EdgeGuidedRejectionReason] = []
    if not all(item.sufficient_trades for item in (training, validation, test)):
        reasons.append(EdgeGuidedRejectionReason.INSUFFICIENT_TRADES)
    for value, reason in (
        (training, EdgeGuidedRejectionReason.NON_POSITIVE_TRAINING_RETURN),
        (validation, EdgeGuidedRejectionReason.NON_POSITIVE_VALIDATION_RETURN),
        (test, EdgeGuidedRejectionReason.NON_POSITIVE_TEST_RETURN),
    ):
        if not value.positive_net_return:
            reasons.append(reason)
    if not validation.beats_production:
        reasons.append(EdgeGuidedRejectionReason.DID_NOT_BEAT_PRODUCTION_VALIDATION)
    if not test.beats_production:
        reasons.append(EdgeGuidedRejectionReason.DID_NOT_BEAT_PRODUCTION_TEST)
    if not all(item.risk_acceptable for item in (training, validation, test)):
        reasons.append(EdgeGuidedRejectionReason.EXCESSIVE_DRAWDOWN)
    if valid < thresholds.minimum_windows:
        reasons.append(EdgeGuidedRejectionReason.INSUFFICIENT_VALID_WINDOWS)
    if confirmed < thresholds.minimum_windows:
        reasons.append(EdgeGuidedRejectionReason.INSUFFICIENT_CONFIRMED_WINDOWS)
    if valid and ratio < thresholds.minimum_stable_direction_ratio:
        reasons.append(EdgeGuidedRejectionReason.UNSTABLE_ACROSS_WINDOWS)
    return _stable_reasons(tuple(reasons))


def _stable_reasons(
    reasons: tuple[EdgeGuidedRejectionReason, ...],
) -> tuple[EdgeGuidedRejectionReason, ...]:
    return tuple(dict.fromkeys(reasons))


def _validation_stage_failed(
    *,
    training: EdgeGuidedSegmentEvaluation,
    validation: EdgeGuidedSegmentEvaluation,
    valid: int,
    thresholds: EvidenceThresholds,
) -> bool:
    return (
        not training.sufficient_trades
        or not training.positive_net_return
        or not training.risk_acceptable
        or not validation.sufficient_trades
        or not validation.positive_net_return
        or not validation.beats_production
        or not validation.risk_acceptable
        or valid < thresholds.minimum_windows
    )


def _rejection_explanation(
    reasons: tuple[EdgeGuidedRejectionReason, ...],
) -> str:
    labels = {
        EdgeGuidedRejectionReason.INSUFFICIENT_TRADES: "insufficient trading activity",
        EdgeGuidedRejectionReason.NON_POSITIVE_TRAINING_RETURN: (
            "non-positive training return"
        ),
        EdgeGuidedRejectionReason.NON_POSITIVE_VALIDATION_RETURN: (
            "non-positive validation return"
        ),
        EdgeGuidedRejectionReason.NON_POSITIVE_TEST_RETURN: "non-positive test return",
        EdgeGuidedRejectionReason.DID_NOT_BEAT_PRODUCTION_VALIDATION: (
            "validation did not strictly beat production"
        ),
        EdgeGuidedRejectionReason.DID_NOT_BEAT_PRODUCTION_TEST: (
            "test did not strictly beat production"
        ),
        EdgeGuidedRejectionReason.EXCESSIVE_DRAWDOWN: (
            "drawdown exceeded the existing risk limit"
        ),
        EdgeGuidedRejectionReason.INSUFFICIENT_VALID_WINDOWS: (
            "insufficient validation windows"
        ),
        EdgeGuidedRejectionReason.INSUFFICIENT_CONFIRMED_WINDOWS: (
            "insufficient confirmed test windows"
        ),
        EdgeGuidedRejectionReason.UNSTABLE_ACROSS_WINDOWS: (
            "insufficient stability across windows"
        ),
    }
    return (
        "Candidate rejected: " + "; ".join(labels[reason] for reason in reasons) + "."
    )


def _fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()
