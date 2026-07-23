import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
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
    PredictiveEdgeError,
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
    SimulationDecisionOverride,
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
MAX_EDGE_GUIDED_CANDIDATES = 120
MAX_GENERATED_STRATEGIES = 120
MAX_TRAINING_RETAINED = 24
MAX_VALIDATION_RETAINED = 12
MAX_TEST_EVALUATED = 5
EDGE_GUIDED_STRATEGY_CONTRACT_VERSION = 4
EDGE_GUIDED_RANKING_CONTRACT_VERSION = 1
EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION = 4
MINIMUM_TRADES_PER_SEGMENT = 1
ECONOMIC_EDGE_MARGIN = Decimal("0.0005")
TARGET_ALLOCATIONS = (Decimal("10"), Decimal("15"), Decimal("20"))
CANDIDATE_CADENCES = tuple(
    timedelta(minutes=value) for value in (30, 60, 120, 240, 360)
)
EDGE_GUIDED_STAGES = (
    RefinementSegmentName.TRAINING,
    RefinementSegmentName.VALIDATION,
    RefinementSegmentName.TEST,
)


class EdgeGuidedCandidateError(ValueError):
    """Raised when confirmed edge evidence cannot form a bounded candidate."""


class EdgeGuidedCandidateStatus(StrEnum):
    REJECTED_VALIDATION = "rejected_validation"
    REJECTED_TEST = "rejected_test"
    CONFIRMED = "confirmed_candidate"


class CandidateFamily(StrEnum):
    EDGE_ENTRY_TIMED_EXIT = "edge_entry_timed_exit"
    EDGE_ENTRY_SIGNAL_EXIT = "edge_entry_signal_exit"
    EDGE_ENTRY_BASELINE_EXIT = "edge_entry_baseline_exit"
    EXIT_OR_AVOID_OVERLAY = "exit_or_avoid_overlay"


class EntryRule(StrEnum):
    EDGE_MATCH = "edge_match"
    PRODUCTION_BASELINE = "production_baseline"


class ExitRule(StrEnum):
    TIMED = "timed"
    EDGE_MATCH = "edge_match"
    PRODUCTION_BASELINE = "production_baseline"
    DEFENSIVE_OVERLAY = "defensive_overlay"


class ResearchOutcome(StrEnum):
    CONFIRMED_CANDIDATES_FOUND = "confirmed_candidates_found"
    CANDIDATES_REJECTED_VALIDATION = "candidates_rejected_validation"
    CANDIDATES_REJECTED_TEST = "candidates_rejected_test"
    NO_PREDICTIVE_EDGE = "no_predictive_edge"
    EDGES_NOT_ECONOMIC = "edges_found_but_not_economic_after_costs"
    INSUFFICIENT_OCCURRENCES = "insufficient_occurrences"
    INSUFFICIENT_DATA = "insufficient_data"
    FAILED = "failed"


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
    candidate_family: CandidateFamily = CandidateFamily.EXIT_OR_AVOID_OVERLAY
    entry_rule: EntryRule = EntryRule.PRODUCTION_BASELINE
    exit_rule: ExitRule = ExitRule.DEFENSIVE_OVERLAY
    cadence: timedelta = DEFAULT_DECISION_INTERVAL
    minimum_holding_period: timedelta = timedelta(0)
    maximum_holding_period: timedelta = timedelta(0)
    cooldown: timedelta = timedelta(0)
    target_allocation_percentage: Decimal = Decimal("15")
    exit_predictor_name: str | None = None
    exit_condition: Condition | None = None
    exit_lower_bound: Decimal | None = None
    exit_upper_bound: Decimal | None = None
    source_hypothesis_ids: tuple[str, ...] = ()
    generation_contract_version: int = EDGE_GUIDED_STRATEGY_CONTRACT_VERSION
    qualification_contract_version: int = EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION
    context_predictor_name: str | None = None
    context_predictor_version: int | None = None
    context_regime: str | None = None
    context_lower_bound: Decimal | None = None
    context_upper_bound: Decimal | None = None


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
class EdgeGuidedRanking:
    median_validation_return: Decimal
    worst_validation_return: Decimal
    maximum_drawdown: Decimal
    return_over_drawdown: Decimal | None
    trade_count: int
    total_fees: Decimal
    cost_efficiency: Decimal | None
    positive_window_ratio: Decimal
    positive_window_count: int
    candidate_fingerprint: str


@dataclass(frozen=True, slots=True)
class EdgeGuidedCandidateEvaluation:
    candidate: EdgeGuidedCandidate
    training: EdgeGuidedSegmentEvaluation
    validation: EdgeGuidedSegmentEvaluation
    test: EdgeGuidedSegmentEvaluation | None
    windows: tuple["EdgeGuidedWindowEvaluation", ...]
    valid_window_count: int
    confirmed_window_count: int
    rejected_validation_window_count: int
    rejected_test_window_count: int
    unconditional_benchmark: str
    status: EdgeGuidedCandidateStatus
    explanation: str
    rejection_reasons: tuple[EdgeGuidedRejectionReason, ...] = ()
    ranking: EdgeGuidedRanking | None = None


@dataclass(frozen=True, slots=True)
class EdgeGuidedEvaluationResult:
    dataset_fingerprint: str
    generated_count: int
    evaluated_count: int
    maximum_candidates: int
    windows_available: int
    evaluations: tuple[EdgeGuidedCandidateEvaluation, ...]
    outcome: ResearchOutcome = ResearchOutcome.NO_PREDICTIVE_EDGE
    outcome_explanation: str = "No complete candidate was generated."
    funnel: dict[str, int] = field(default_factory=dict)
    timings: dict[str, Decimal] = field(default_factory=dict)
    qualification_contract_version: int = EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION
    stage_evidence: tuple["EdgeGuidedStageEvidence", ...] = ()


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
    cache_key: str = ""


@dataclass(frozen=True, slots=True)
class EdgeGuidedStageEvidence:
    cache_key: str
    candidate_fingerprint: str
    stage: RefinementSegmentName
    window_ordinal: int
    segment_from: datetime
    segment_to: datetime
    content_fingerprint: str
    metrics: EdgeGuidedSegmentEvaluation


def generate_edge_guided_candidates(
    *,
    hypotheses: tuple[StrategyHypothesis, ...],
    simulation: SimulationConfiguration,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
    maximum_candidates: int = DEFAULT_MAX_EDGE_GUIDED_CANDIDATES,
) -> tuple[EdgeGuidedCandidate, ...]:
    """Generate a bounded, ranked set of complete long-only Strategies.

    Validation/test never participate here. Entry hypotheses must clear the
    deterministic round-trip cost screen before they can consume simulation budget.
    """

    if not 1 <= maximum_candidates <= MAX_EDGE_GUIDED_CANDIDATES:
        raise EdgeGuidedCandidateError(
            f"maximum_candidates must be between 1 and {MAX_EDGE_GUIDED_CANDIDATES}"
        )
    entries = tuple(
        item
        for item in sorted(hypotheses, key=_hypothesis_rank)
        if item.suggested_action is SuggestedAction.LONG_ENTRY
        and _valid_hypothesis(item)
        and _economically_viable(item, simulation)
    )[:12]
    exits = tuple(
        item
        for item in sorted(hypotheses, key=_hypothesis_rank)
        if item.suggested_action is SuggestedAction.EXIT_OR_AVOID
        and _valid_hypothesis(item)
    )[:6]
    definitions: list[
        tuple[
            StrategyHypothesis,
            CandidateFamily,
            ExitRule,
            timedelta,
            timedelta,
            Decimal,
            StrategyHypothesis | None,
        ]
    ] = []
    for entry in entries:
        cadences = (
            (entry.minimum_evaluation_interval,)
            if entry.context_predictor_name is not None
            else _compatible_cadences(entry.horizon)
        )
        compatible_exits = tuple(
            value
            for value in exits
            if value.condition is entry.condition
            and abs(value.horizon - entry.horizon) <= entry.horizon
            and value.context_predictor_name == entry.context_predictor_name
            and value.context_regime == entry.context_regime
        )
        families = (
            (CandidateFamily.EDGE_ENTRY_TIMED_EXIT, ExitRule.TIMED, None),
            (
                CandidateFamily.EDGE_ENTRY_BASELINE_EXIT,
                ExitRule.PRODUCTION_BASELINE,
                None,
            ),
            *tuple(
                (CandidateFamily.EDGE_ENTRY_SIGNAL_EXIT, ExitRule.EDGE_MATCH, exit_edge)
                for exit_edge in compatible_exits
            ),
        )
        for cadence in cadences:
            # Horizon is frozen evidence. Holding and cooldown are deterministic
            # cadence-aligned derivatives, never ranking-position choices.
            holding = max(cadence, entry.horizon)
            for allocation in TARGET_ALLOCATIONS:
                for family, exit_rule, exit_edge in families:
                    definitions.append(
                        (
                            entry,
                            family,
                            exit_rule,
                            cadence,
                            holding,
                            allocation,
                            exit_edge,
                        )
                    )
    for overlay in exits:
        overlay_cadence = (
            overlay.minimum_evaluation_interval
            if overlay.context_predictor_name is not None
            else decision_interval
        )
        definitions.append(
            (
                overlay,
                CandidateFamily.EXIT_OR_AVOID_OVERLAY,
                ExitRule.DEFENSIVE_OVERLAY,
                overlay_cadence,
                timedelta(0),
                TARGET_ALLOCATIONS[1],
                None,
            )
        )

    candidates: list[EdgeGuidedCandidate] = []
    for (
        hypothesis,
        family,
        exit_rule,
        cadence,
        holding,
        allocation,
        exit_edge,
    ) in definitions:
        source_ids = (hypothesis.hypothesis_id,) + (
            (exit_edge.hypothesis_id,) if exit_edge is not None else ()
        )
        payload: dict[str, object] = {
            "generation_contract_version": EDGE_GUIDED_STRATEGY_CONTRACT_VERSION,
            "qualification_contract_version": (
                EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION
            ),
            "candidate_family": family.value,
            "entry_rule": (
                EntryRule.PRODUCTION_BASELINE.value
                if family is CandidateFamily.EXIT_OR_AVOID_OVERLAY
                else EntryRule.EDGE_MATCH.value
            ),
            "exit_rule": exit_rule.value,
            "source_hypothesis_id": hypothesis.hypothesis_id,
            "source_hypothesis_ids": source_ids,
            "predictor_name": hypothesis.predictor_name,
            "predictor_version": hypothesis.predictor_version,
            "hypothesis_contract_version": hypothesis.hypothesis_contract_version,
            "condition": hypothesis.condition.value,
            "lower_bound": hypothesis.lower_bound,
            "upper_bound": hypothesis.upper_bound,
            "suggested_action": hypothesis.suggested_action.value,
            "horizon_seconds": int(hypothesis.horizon.total_seconds()),
            "decision_interval_seconds": int(cadence.total_seconds()),
            "minimum_holding_seconds": int(holding.total_seconds()),
            "maximum_holding_seconds": int(
                max(holding, hypothesis.minimum_holding_period_to).total_seconds()
            ),
            "cooldown_seconds": int(cadence.total_seconds()),
            "target_allocation_percentage": allocation,
            "exit_predictor_name": exit_edge.predictor_name if exit_edge else None,
            "exit_condition": exit_edge.condition.value if exit_edge else None,
            "exit_lower_bound": exit_edge.lower_bound if exit_edge else None,
            "exit_upper_bound": exit_edge.upper_bound if exit_edge else None,
            "direction_mapping": hypothesis.direction_mapping.value,
            "context_predictor_name": hypothesis.context_predictor_name,
            "context_predictor_version": hypothesis.context_predictor_version,
            "context_regime": hypothesis.context_regime,
            "context_lower_bound": hypothesis.context_lower_bound,
            "context_upper_bound": hypothesis.context_upper_bound,
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
                cadence,
                hypothesis.direction_mapping,
                MODEL_NAME,
                MODEL_VERSION,
                STRATEGY_NAME,
                STRATEGY_VERSION,
                CONFIGURATION_VERSION,
                simulation.account_currency,
                simulation.fee_rate,
                simulation.slippage_rate,
                family,
                EntryRule.PRODUCTION_BASELINE
                if family is CandidateFamily.EXIT_OR_AVOID_OVERLAY
                else EntryRule.EDGE_MATCH,
                exit_rule,
                cadence,
                holding,
                max(holding, hypothesis.minimum_holding_period_to),
                cadence,
                allocation,
                exit_edge.predictor_name if exit_edge else None,
                exit_edge.condition if exit_edge else None,
                exit_edge.lower_bound if exit_edge else None,
                exit_edge.upper_bound if exit_edge else None,
                source_ids,
                context_predictor_name=hypothesis.context_predictor_name,
                context_predictor_version=hypothesis.context_predictor_version,
                context_regime=hypothesis.context_regime,
                context_lower_bound=hypothesis.context_lower_bound,
                context_upper_bound=hypothesis.context_upper_bound,
            )
        )
    unique = {item.candidate_fingerprint: item for item in candidates}
    # Generate and economically pre-screen broadly first. Stratification prevents
    # family ordering or one prolific hypothesis from consuming the whole budget.
    # maximum_candidates is intentionally applied by the staged evaluator.
    return _stratified_generated_candidates(tuple(unique.values()))


def _stratified_generated_candidates(
    candidates: tuple[EdgeGuidedCandidate, ...],
) -> tuple[EdgeGuidedCandidate, ...]:
    family_order = tuple(CandidateFamily)
    family_queues = {
        family: list(
            _round_robin_hypotheses(
                tuple(item for item in candidates if item.candidate_family is family)
            )
        )
        for family in family_order
    }
    selected: list[EdgeGuidedCandidate] = []
    while len(selected) < MAX_GENERATED_STRATEGIES and any(family_queues.values()):
        for family in family_order:
            queue = family_queues[family]
            if queue and len(selected) < MAX_GENERATED_STRATEGIES:
                selected.append(queue.pop(0))
    return tuple(selected)


def _round_robin_hypotheses(
    candidates: tuple[EdgeGuidedCandidate, ...],
) -> tuple[EdgeGuidedCandidate, ...]:
    grouped: dict[str, list[EdgeGuidedCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.source_hypothesis_id, []).append(candidate)
    for hypothesis_index, hypothesis_id in enumerate(sorted(grouped)):
        values = grouped[hypothesis_id]
        values.sort(
            key=lambda item: (
                (
                    TARGET_ALLOCATIONS.index(item.target_allocation_percentage)
                    - CANDIDATE_CADENCES.index(item.cadence)
                )
                % len(TARGET_ALLOCATIONS),
                CANDIDATE_CADENCES.index(item.cadence),
                item.exit_predictor_name or "",
                item.candidate_fingerprint,
            )
        )
        if values:
            offset = hypothesis_index % min(len(values), len(CANDIDATE_CADENCES))
            grouped[hypothesis_id] = values[offset:] + values[:offset]
    ordered: list[EdgeGuidedCandidate] = []
    hypothesis_ids = sorted(grouped)
    while any(grouped.values()):
        for hypothesis_id in hypothesis_ids:
            values = grouped[hypothesis_id]
            if values:
                ordered.append(values.pop(0))
    return tuple(ordered)


def apply_edge_guided_action(
    *,
    candidate: EdgeGuidedCandidate,
    row: HistoricalFeatureRow,
    position_is_open: bool,
    baseline_action: StrategyAction,
    volatility_threshold: Decimal,
) -> StrategyAction:
    """Apply one frozen bin rule without introducing short exposure."""

    value = _candidate_predictor_value(row, candidate.predictor_name)
    if value is None:
        return baseline_action
    inside = (
        (candidate.lower_bound is None or value > candidate.lower_bound)
        and (candidate.upper_bound is None or value <= candidate.upper_bound)
        and predictive_edge_condition_matches(
            row,
            candidate.condition,
            volatility_threshold=volatility_threshold,
        )
        and _candidate_context_matches(row, candidate)
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
    entry_boundary: datetime | None = None
    last_exit_boundary: datetime | None = None

    def apply(
        row: HistoricalFeatureRow,
        position_is_open: bool,
        baseline_action: StrategyAction,
    ) -> StrategyAction | SimulationDecisionOverride:
        nonlocal entry_boundary, last_exit_boundary
        boundary = row.candle_open_time + timedelta(minutes=5)
        if (boundary - datetime(1970, 1, 1, tzinfo=UTC)) % candidate.cadence:
            return baseline_action
        entry_matches = _candidate_matches(
            row,
            predictor_name=candidate.predictor_name,
            condition=candidate.condition,
            lower_bound=candidate.lower_bound,
            upper_bound=candidate.upper_bound,
            volatility_threshold=volatility_threshold,
        ) and _candidate_context_matches(row, candidate)

        if candidate.candidate_family is CandidateFamily.EXIT_OR_AVOID_OVERLAY:
            return apply_edge_guided_action(
                candidate=candidate,
                row=row,
                position_is_open=position_is_open,
                baseline_action=baseline_action,
                volatility_threshold=volatility_threshold,
            )
        if position_is_open:
            if entry_boundary is None:
                entry_boundary = boundary
            held = boundary - entry_boundary
            should_exit = (
                (
                    candidate.exit_rule is ExitRule.TIMED
                    and held >= candidate.maximum_holding_period
                )
                or (
                    candidate.exit_rule is ExitRule.PRODUCTION_BASELINE
                    and baseline_action is StrategyAction.SELL_ALL
                    and held >= candidate.minimum_holding_period
                )
                or (
                    candidate.exit_rule is ExitRule.EDGE_MATCH
                    and held >= candidate.minimum_holding_period
                    and _candidate_matches(
                        row,
                        predictor_name=candidate.exit_predictor_name or "",
                        condition=candidate.exit_condition or Condition.ALL,
                        lower_bound=candidate.exit_lower_bound,
                        upper_bound=candidate.exit_upper_bound,
                        volatility_threshold=volatility_threshold,
                    )
                )
            )
            if should_exit:
                last_exit_boundary = boundary
                entry_boundary = None
                return SimulationDecisionOverride(
                    StrategyAction.SELL_ALL,
                    applied_rule=candidate.exit_rule.value,
                    explanation="The frozen complete Strategy exit rule matched.",
                )
            return StrategyAction.NO_ACTION
        cooldown_complete = (
            last_exit_boundary is None
            or boundary - last_exit_boundary >= candidate.cooldown
        )
        if entry_matches and cooldown_complete:
            entry_boundary = boundary
            return SimulationDecisionOverride(
                StrategyAction.OPEN_POSITION,
                candidate.target_allocation_percentage,
                candidate.entry_rule.value,
                "The frozen edge entry rule matched independently of production.",
            )
        return StrategyAction.NO_ACTION

    return apply


def evaluate_edge_guided_candidates(
    *,
    dataset: HistoricalDataset,
    feature_matrix: HistoricalFeatureMatrix,
    hypotheses: tuple[StrategyHypothesis, ...],
    frozen_candidates: tuple[EdgeGuidedCandidate, ...] | None = None,
    initial_cash: Decimal,
    simulation: SimulationConfiguration,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
    maximum_candidates: int = DEFAULT_MAX_EDGE_GUIDED_CANDIDATES,
    maximum_test_candidates: int = MAX_TEST_EVALUATED,
    windows: tuple[WalkForwardWindow, ...],
    thresholds: EvidenceThresholds,
    reusable_stage_evidence: dict[str, EdgeGuidedStageEvidence] | None = None,
) -> EdgeGuidedEvaluationResult:
    """Evaluate frozen candidates through the existing isolated walk-forward windows."""

    candidates = frozen_candidates or generate_edge_guided_candidates(
        hypotheses=hypotheses,
        simulation=simulation,
        decision_interval=decision_interval,
        maximum_candidates=maximum_candidates,
    )
    cache = dict(reusable_stage_evidence or {})
    evidence: dict[str, EdgeGuidedStageEvidence] = dict(cache)
    reused_keys: set[str] = set()
    computed_keys: set[str] = set()
    training_results = tuple(
        (
            candidate,
            _evaluate_candidate_stage(
                dataset=dataset,
                feature_matrix=feature_matrix,
                windows=windows,
                candidate=candidate,
                initial_cash=initial_cash,
                simulation=simulation,
                decision_interval=decision_interval,
                segment=RefinementSegmentName.TRAINING,
                reusable_stage_evidence=cache,
                stage_evidence=evidence,
                reused_keys=reused_keys,
                computed_keys=computed_keys,
            ),
        )
        for candidate in candidates
    )
    training_retained = tuple(
        sorted(
            (
                item
                for item in training_results
                if _stage_qualifies(item[1], RefinementSegmentName.TRAINING)
            ),
            key=lambda item: _segment_rank(item[1], item[0].candidate_fingerprint),
        )[:MAX_TRAINING_RETAINED]
    )
    validation_results = tuple(
        (
            candidate,
            training,
            _evaluate_candidate_stage(
                dataset=dataset,
                feature_matrix=feature_matrix,
                windows=windows,
                candidate=candidate,
                initial_cash=initial_cash,
                simulation=simulation,
                decision_interval=decision_interval,
                segment=RefinementSegmentName.VALIDATION,
                reusable_stage_evidence=cache,
                stage_evidence=evidence,
                reused_keys=reused_keys,
                computed_keys=computed_keys,
            ),
        )
        for candidate, training in training_retained
    )
    validation_qualified = tuple(
        item
        for item in validation_results
        if _stage_qualifies(item[1], RefinementSegmentName.TRAINING)
        and _stage_qualifies(item[2], RefinementSegmentName.VALIDATION)
    )
    validation_retained = tuple(
        sorted(
            validation_qualified,
            key=lambda item: _segment_rank(item[2], item[0].candidate_fingerprint),
        )[:MAX_VALIDATION_RETAINED]
    )
    test_candidates = validation_retained[
        : min(maximum_test_candidates, maximum_candidates)
    ]
    evaluations: list[EdgeGuidedCandidateEvaluation] = []
    for candidate, _, _ in test_candidates:
        window_results = _evaluate_candidate_windows(
            dataset=dataset,
            feature_matrix=feature_matrix,
            windows=windows,
            candidate=candidate,
            initial_cash=initial_cash,
            simulation=simulation,
            decision_interval=decision_interval,
            thresholds=thresholds,
            reusable_stage_evidence=cache,
            stage_evidence=evidence,
            reused_keys=reused_keys,
            computed_keys=computed_keys,
        )
        confirmed = sum(item.test_confirmed for item in window_results)
        validation_passed_count = sum(item.validation_passed for item in window_results)
        rejected_validation = len(window_results) - validation_passed_count
        rejected_test = sum(
            item.validation_passed and not item.test_confirmed
            for item in window_results
        )
        ratio = (
            Decimal(confirmed) / Decimal(validation_passed_count)
            if validation_passed_count
            else Decimal(0)
        )
        training = _aggregate_segment(window_results, RefinementSegmentName.TRAINING)
        validation = _aggregate_segment(
            window_results, RefinementSegmentName.VALIDATION
        )
        test = _aggregate_optional_test(window_results)
        validation_failed = _validation_stage_failed(
            training=training,
            validation=validation,
            valid=validation_passed_count,
            thresholds=thresholds,
        )
        reasons = _candidate_rejection_reasons(
            training=training,
            validation=validation,
            test=test,
            valid=validation_passed_count,
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
        elif validation_failed:
            status = EdgeGuidedCandidateStatus.REJECTED_VALIDATION
        else:
            status = EdgeGuidedCandidateStatus.REJECTED_TEST
        if reasons:
            explanation = (
                _rejection_explanation(reasons)
                + " Isolated test was not executed because validation failed."
                if validation_failed
                else _rejection_explanation(reasons)
            )
        evaluations.append(
            EdgeGuidedCandidateEvaluation(
                candidate,
                training,
                validation,
                test,
                window_results,
                validation_passed_count,
                confirmed,
                rejected_validation,
                rejected_test,
                "always_up"
                if candidate.suggested_action is SuggestedAction.LONG_ENTRY
                else "always_down",
                status,
                explanation,
                reasons,
                _ranking(window_results, candidate.candidate_fingerprint),
            )
        )
    tested_fingerprints = {item.candidate.candidate_fingerprint for item in evaluations}
    for candidate, training, validation in validation_results:
        if candidate.candidate_fingerprint in tested_fingerprints:
            continue
        if _stage_qualifies(
            training, RefinementSegmentName.TRAINING
        ) and _stage_qualifies(validation, RefinementSegmentName.VALIDATION):
            # Qualified candidates below the isolated-test budget are intentionally
            # represented only by funnel counters; they are not test rejections.
            continue
        reasons = _candidate_rejection_reasons(
            training=training,
            validation=validation,
            test=None,
            valid=0,
            confirmed=0,
            ratio=Decimal(0),
            thresholds=thresholds,
        )
        evaluations.append(
            EdgeGuidedCandidateEvaluation(
                candidate,
                training,
                validation,
                None,
                (),
                0,
                0,
                0,
                0,
                "always_up"
                if candidate.suggested_action is SuggestedAction.LONG_ENTRY
                else "always_down",
                EdgeGuidedCandidateStatus.REJECTED_VALIDATION,
                _rejection_explanation(reasons)
                + " Isolated test was not executed because validation failed.",
                reasons,
                None,
            )
        )
    evaluations = evaluations[:maximum_candidates]
    outcome = _research_outcome(tuple(evaluations), bool(hypotheses), bool(candidates))
    return EdgeGuidedEvaluationResult(
        dataset.fingerprint,
        len(candidates),
        len(evaluations),
        maximum_candidates,
        len(windows),
        tuple(evaluations),
        outcome,
        _outcome_explanation(outcome),
        {
            "hypotheses_generated": len(hypotheses),
            "complete_strategies_generated": len(candidates),
            "generated_count": len(candidates),
            "training_evaluated_count": len(training_results),
            "training_retained_count": len(training_retained),
            "validation_evaluated_count": len(validation_results),
            "validation_retained_count": len(validation_retained),
            "test_evaluated_count": sum(item.test is not None for item in evaluations),
            "training_screened": len(training_results),
            "validation_evaluated": len(validation_results),
            "test_evaluated": sum(item.test is not None for item in evaluations),
            "confirmed_candidates": sum(
                item.status is EdgeGuidedCandidateStatus.CONFIRMED
                for item in evaluations
            ),
            "windows_reused": len(reused_keys),
            "windows_computed": len(computed_keys),
            **{
                f"{stage.value}_windows_reused": sum(
                    item.stage is stage and key in reused_keys
                    for key, item in evidence.items()
                )
                for stage in EDGE_GUIDED_STAGES
            },
            **{
                f"{stage.value}_windows_computed": sum(
                    item.stage is stage and key in computed_keys
                    for key, item in evidence.items()
                )
                for stage in EDGE_GUIDED_STAGES
            },
        },
        stage_evidence=tuple(
            evidence[key]
            for key in sorted(evidence)
            if key in computed_keys or key in reused_keys
        ),
    )


def _evaluate_candidate_stage(
    *,
    dataset: HistoricalDataset,
    feature_matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    candidate: EdgeGuidedCandidate,
    initial_cash: Decimal,
    simulation: SimulationConfiguration,
    decision_interval: timedelta,
    segment: RefinementSegmentName,
    reusable_stage_evidence: dict[str, EdgeGuidedStageEvidence],
    stage_evidence: dict[str, EdgeGuidedStageEvidence],
    reused_keys: set[str],
    computed_keys: set[str],
) -> EdgeGuidedSegmentEvaluation:
    values: list[EdgeGuidedSegmentEvaluation] = []
    for window in windows:
        matrices = build_window_matrices(
            feature_matrix, window, decision_interval=decision_interval
        )
        matrix = (
            matrices.training
            if segment is RefinementSegmentName.TRAINING
            else matrices.validation
        )
        if not matrix.rows or not matrices.training.rows:
            continue
        volatility_threshold = predictive_edge_volatility_threshold(
            matrices.training.rows
        )
        values.append(
            _evaluate_or_reuse_stage(
                dataset=dataset,
                matrix=matrix,
                window=window,
                segment=segment,
                candidate=candidate,
                initial_cash=initial_cash,
                simulation=simulation,
                volatility_threshold=volatility_threshold,
                reusable_stage_evidence=reusable_stage_evidence,
                stage_evidence=stage_evidence,
                reused_keys=reused_keys,
                computed_keys=computed_keys,
            )
        )
    return _aggregate_values(tuple(values), segment)


def _aggregate_values(
    values: tuple[EdgeGuidedSegmentEvaluation, ...],
    segment: RefinementSegmentName,
) -> EdgeGuidedSegmentEvaluation:
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


def _stage_qualifies(
    value: EdgeGuidedSegmentEvaluation, stage: RefinementSegmentName
) -> bool:
    qualifies = (
        value.sufficient_trades and value.positive_net_return and value.risk_acceptable
    )
    return qualifies and (
        stage is not RefinementSegmentName.VALIDATION or value.beats_production
    )


def _segment_rank(
    value: EdgeGuidedSegmentEvaluation, fingerprint: str
) -> tuple[Decimal, Decimal, int, str]:
    return (-value.total_return, value.max_drawdown, -value.trade_count, fingerprint)


def _evaluate_candidate_windows(
    *,
    dataset: HistoricalDataset,
    feature_matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    candidate: EdgeGuidedCandidate,
    initial_cash: Decimal,
    simulation: SimulationConfiguration,
    decision_interval: timedelta,
    thresholds: EvidenceThresholds,
    reusable_stage_evidence: dict[str, EdgeGuidedStageEvidence],
    stage_evidence: dict[str, EdgeGuidedStageEvidence],
    reused_keys: set[str],
    computed_keys: set[str],
) -> tuple[EdgeGuidedWindowEvaluation, ...]:
    prepared: list[
        tuple[
            WalkForwardWindow,
            HistoricalFeatureMatrix | None,
            Decimal,
            EdgeGuidedSegmentEvaluation,
            EdgeGuidedSegmentEvaluation,
            tuple[EdgeGuidedRejectionReason, ...],
        ]
    ] = []
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
        training = _evaluate_or_reuse_stage(
            dataset=dataset,
            matrix=matrices.training,
            window=window,
            segment=RefinementSegmentName.TRAINING,
            candidate=candidate,
            initial_cash=initial_cash,
            simulation=simulation,
            volatility_threshold=volatility_threshold,
            reusable_stage_evidence=reusable_stage_evidence,
            stage_evidence=stage_evidence,
            reused_keys=reused_keys,
            computed_keys=computed_keys,
        )
        validation = _evaluate_or_reuse_stage(
            dataset=dataset,
            matrix=matrices.validation,
            window=window,
            segment=RefinementSegmentName.VALIDATION,
            candidate=candidate,
            initial_cash=initial_cash,
            simulation=simulation,
            volatility_threshold=volatility_threshold,
            reusable_stage_evidence=reusable_stage_evidence,
            stage_evidence=stage_evidence,
            reused_keys=reused_keys,
            computed_keys=computed_keys,
        )
        validation_reasons = _window_segment_rejection_reasons(
            validation, RefinementSegmentName.VALIDATION
        )
        prepared.append(
            (
                window,
                matrices.test,
                volatility_threshold,
                training,
                validation,
                validation_reasons,
            )
        )
    validation_count = sum(not item[5] for item in prepared)
    allow_test = validation_count >= thresholds.minimum_windows
    results: list[EdgeGuidedWindowEvaluation] = []
    for (
        window,
        test_matrix,
        volatility_threshold,
        training,
        validation,
        validation_reasons,
    ) in prepared:
        validation_passed = not validation_reasons
        test = (
            _evaluate_or_reuse_stage(
                dataset=dataset,
                matrix=test_matrix,
                window=window,
                segment=RefinementSegmentName.TEST,
                candidate=candidate,
                initial_cash=initial_cash,
                simulation=simulation,
                volatility_threshold=volatility_threshold,
                reusable_stage_evidence=reusable_stage_evidence,
                stage_evidence=stage_evidence,
                reused_keys=reused_keys,
                computed_keys=computed_keys,
            )
            if validation_passed and allow_test and test_matrix is not None
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
                "",
            )
        )
    return tuple(results)


def edge_guided_stage_cache_key(
    candidate: EdgeGuidedCandidate,
    window: WalkForwardWindow,
    stage: RefinementSegmentName,
    content_fingerprint: str,
) -> str:
    """Identify immutable candidate evidence for one exact stage segment."""

    bounds = {
        RefinementSegmentName.TRAINING: (window.training_from, window.training_to),
        RefinementSegmentName.VALIDATION: (
            window.validation_from,
            window.validation_to,
        ),
        RefinementSegmentName.TEST: (window.test_from, window.test_to),
    }[stage]

    return _fingerprint(
        {
            "candidate_fingerprint": candidate.candidate_fingerprint,
            "stage": stage.value,
            "segment": bounds,
            "content_fingerprint": content_fingerprint,
            "ranking_contract_version": EDGE_GUIDED_RANKING_CONTRACT_VERSION,
            "qualification_contract_version": (
                EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION
            ),
        }
    )


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


def _evaluate_or_reuse_stage(
    *,
    dataset: HistoricalDataset,
    matrix: HistoricalFeatureMatrix,
    window: WalkForwardWindow,
    segment: RefinementSegmentName,
    candidate: EdgeGuidedCandidate,
    initial_cash: Decimal,
    simulation: SimulationConfiguration,
    volatility_threshold: Decimal,
    reusable_stage_evidence: dict[str, EdgeGuidedStageEvidence],
    stage_evidence: dict[str, EdgeGuidedStageEvidence],
    reused_keys: set[str],
    computed_keys: set[str],
) -> EdgeGuidedSegmentEvaluation:
    content_fingerprint = _stage_content_fingerprint(matrix)
    cache_key = edge_guided_stage_cache_key(
        candidate, window, segment, content_fingerprint
    )
    cached = reusable_stage_evidence.get(cache_key)
    if cached is not None:
        reused_keys.add(cache_key)
        stage_evidence[cache_key] = replace(cached, window_ordinal=window.ordinal)
        return cached.metrics
    metrics = _evaluate_segment(
        dataset,
        matrix,
        segment,
        candidate,
        initial_cash,
        simulation,
        _candidate_action_filter(candidate, volatility_threshold),
    )
    bounds = {
        RefinementSegmentName.TRAINING: (window.training_from, window.training_to),
        RefinementSegmentName.VALIDATION: (
            window.validation_from,
            window.validation_to,
        ),
        RefinementSegmentName.TEST: (window.test_from, window.test_to),
    }[segment]
    stage_evidence[cache_key] = EdgeGuidedStageEvidence(
        cache_key,
        candidate.candidate_fingerprint,
        segment,
        window.ordinal,
        bounds[0],
        bounds[1],
        content_fingerprint,
        metrics,
    )
    computed_keys.add(cache_key)
    return metrics


def _stage_content_fingerprint(matrix: HistoricalFeatureMatrix) -> str:
    return _fingerprint({"rows": tuple(asdict(row) for row in matrix.rows)})


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


def _aggregate_optional_test(
    windows: tuple[EdgeGuidedWindowEvaluation, ...],
) -> EdgeGuidedSegmentEvaluation | None:
    if not any(item.test is not None for item in windows):
        return None
    return _aggregate_segment(windows, RefinementSegmentName.TEST)


def _ranking(
    windows: tuple[EdgeGuidedWindowEvaluation, ...], fingerprint: str
) -> EdgeGuidedRanking:
    returns = tuple(sorted(item.validation.total_return for item in windows))
    median = (
        returns[len(returns) // 2]
        if len(returns) % 2
        else (returns[len(returns) // 2 - 1] + returns[len(returns) // 2]) / Decimal(2)
        if returns
        else Decimal(0)
    )
    worst = min(returns, default=Decimal(0))
    drawdown = max(
        (item.validation.max_drawdown for item in windows), default=Decimal(0)
    )
    trades = sum(item.validation.trade_count for item in windows)
    fees = sum((item.validation.fees_paid for item in windows), Decimal(0))
    positive = sum(value > 0 for value in returns)
    ratio = Decimal(positive) / Decimal(len(returns)) if returns else Decimal(0)
    return EdgeGuidedRanking(
        median,
        worst,
        drawdown,
        median / drawdown if drawdown > 0 else None,
        trades,
        fees,
        median / fees if fees > 0 else None,
        ratio,
        positive,
        fingerprint,
    )


def edge_guided_summary(result: EdgeGuidedEvaluationResult) -> dict[str, object]:
    """Return a structured JSON-safe job summary with Decimal strings."""

    value = json.loads(json.dumps(asdict(result), default=str))
    if not isinstance(value, dict):
        raise EdgeGuidedCandidateError("edge-guided summary must be an object")
    value.pop("stage_evidence", None)
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
    test: EdgeGuidedSegmentEvaluation | None,
    valid: int,
    confirmed: int,
    ratio: Decimal,
    thresholds: EvidenceThresholds,
) -> tuple[EdgeGuidedRejectionReason, ...]:
    reasons: list[EdgeGuidedRejectionReason] = []
    if not all(item.sufficient_trades for item in (training, validation)):
        reasons.append(EdgeGuidedRejectionReason.INSUFFICIENT_TRADES)
    for value, reason in (
        (training, EdgeGuidedRejectionReason.NON_POSITIVE_TRAINING_RETURN),
        (validation, EdgeGuidedRejectionReason.NON_POSITIVE_VALIDATION_RETURN),
    ):
        if not value.positive_net_return:
            reasons.append(reason)
    if not validation.beats_production:
        reasons.append(EdgeGuidedRejectionReason.DID_NOT_BEAT_PRODUCTION_VALIDATION)
    if not all(item.risk_acceptable for item in (training, validation)):
        reasons.append(EdgeGuidedRejectionReason.EXCESSIVE_DRAWDOWN)
    if valid < thresholds.minimum_windows:
        reasons.append(EdgeGuidedRejectionReason.INSUFFICIENT_VALID_WINDOWS)
    validation_passed = not _validation_stage_failed(
        training=training,
        validation=validation,
        valid=valid,
        thresholds=thresholds,
    )
    if validation_passed:
        if test is None or not test.sufficient_trades:
            reasons.append(EdgeGuidedRejectionReason.INSUFFICIENT_TRADES)
        if test is not None and not test.positive_net_return:
            reasons.append(EdgeGuidedRejectionReason.NON_POSITIVE_TEST_RETURN)
        if test is not None and not test.beats_production:
            reasons.append(EdgeGuidedRejectionReason.DID_NOT_BEAT_PRODUCTION_TEST)
        if test is not None and not test.risk_acceptable:
            reasons.append(EdgeGuidedRejectionReason.EXCESSIVE_DRAWDOWN)
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


def _valid_hypothesis(value: StrategyHypothesis) -> bool:
    return not (
        (value.lower_bound is None and value.upper_bound is None)
        or (
            value.lower_bound is not None
            and value.upper_bound is not None
            and value.lower_bound >= value.upper_bound
        )
    )


def _economically_viable(
    value: StrategyHypothesis, simulation: SimulationConfiguration
) -> bool:
    gross = value.expected_gross_return
    net = value.expected_net_return
    if gross is None or net is None:
        return False
    entry_multiplier = (Decimal(1) + simulation.slippage_rate) * (
        Decimal(1) + simulation.fee_rate
    )
    exit_multiplier = (Decimal(1) - simulation.slippage_rate) * (
        Decimal(1) - simulation.fee_rate
    )
    round_trip_cost = entry_multiplier / exit_multiplier - Decimal(1)
    return gross > round_trip_cost + ECONOMIC_EDGE_MARGIN and net > ECONOMIC_EDGE_MARGIN


def _hypothesis_rank(value: StrategyHypothesis) -> tuple[Decimal, Decimal, str]:
    economic_value = (
        value.expected_net_return
        if value.suggested_action is SuggestedAction.LONG_ENTRY
        else value.expected_avoided_loss
    ) or Decimal(0)
    return (-economic_value, -value.stability_ratio, value.hypothesis_id)


def _compatible_cadences(horizon: timedelta) -> tuple[timedelta, ...]:
    compatible = tuple(value for value in CANDIDATE_CADENCES if value <= horizon)
    return compatible[-2:] if compatible else (CANDIDATE_CADENCES[0],)


def _candidate_rank(value: EdgeGuidedCandidate) -> tuple[int, str]:
    order = {
        CandidateFamily.EDGE_ENTRY_TIMED_EXIT: 0,
        CandidateFamily.EDGE_ENTRY_SIGNAL_EXIT: 1,
        CandidateFamily.EDGE_ENTRY_BASELINE_EXIT: 2,
        CandidateFamily.EXIT_OR_AVOID_OVERLAY: 3,
    }
    return order[value.candidate_family], value.candidate_fingerprint


def _candidate_matches(
    row: HistoricalFeatureRow,
    *,
    predictor_name: str,
    condition: Condition,
    lower_bound: Decimal | None,
    upper_bound: Decimal | None,
    volatility_threshold: Decimal,
) -> bool:
    if not predictor_name:
        return False
    value = _candidate_predictor_value(row, predictor_name)
    if value is None:
        return False
    return (
        (lower_bound is None or value > lower_bound)
        and (upper_bound is None or value <= upper_bound)
        and predictive_edge_condition_matches(
            row, condition, volatility_threshold=volatility_threshold
        )
    )


def _candidate_context_matches(
    row: HistoricalFeatureRow, candidate: EdgeGuidedCandidate
) -> bool:
    if candidate.context_predictor_name is None:
        return True
    value = _candidate_predictor_value(row, candidate.context_predictor_name)
    if value is None:
        return False
    return (
        candidate.context_lower_bound is None or value >= candidate.context_lower_bound
    ) and (
        candidate.context_upper_bound is None or value <= candidate.context_upper_bound
    )


def _candidate_predictor_value(
    row: HistoricalFeatureRow, predictor_name: str
) -> Decimal | None:
    """Treat unavailable lookback-derived research evidence as a non-match."""

    try:
        return predictive_edge_value(row, predictor_name)
    except PredictiveEdgeError:
        return None


def _research_outcome(
    evaluations: tuple[EdgeGuidedCandidateEvaluation, ...],
    edges_found: bool,
    candidates_generated: bool,
) -> ResearchOutcome:
    if any(item.status is EdgeGuidedCandidateStatus.CONFIRMED for item in evaluations):
        return ResearchOutcome.CONFIRMED_CANDIDATES_FOUND
    if any(
        item.status is EdgeGuidedCandidateStatus.REJECTED_TEST for item in evaluations
    ):
        return ResearchOutcome.CANDIDATES_REJECTED_TEST
    if evaluations:
        return ResearchOutcome.CANDIDATES_REJECTED_VALIDATION
    if edges_found and not candidates_generated:
        return ResearchOutcome.EDGES_NOT_ECONOMIC
    return ResearchOutcome.NO_PREDICTIVE_EDGE


def _outcome_explanation(outcome: ResearchOutcome) -> str:
    return {
        ResearchOutcome.CONFIRMED_CANDIDATES_FOUND: (
            "At least one complete Strategy passed isolated validation and test."
        ),
        ResearchOutcome.CANDIDATES_REJECTED_VALIDATION: (
            "Complete Strategies were generated but failed validation; test was not "
            "used to adjust them."
        ),
        ResearchOutcome.CANDIDATES_REJECTED_TEST: (
            "Candidates were profitable in validation but failed isolated test."
        ),
        ResearchOutcome.NO_PREDICTIVE_EDGE: (
            "No stable predictive edge was detected with the current evidence."
        ),
        ResearchOutcome.EDGES_NOT_ECONOMIC: (
            "Patterns were detected, but expected movement did not cover round-trip "
            "costs and the versioned margin."
        ),
        ResearchOutcome.INSUFFICIENT_OCCURRENCES: (
            "Candidate evidence did not occur often enough for evaluation."
        ),
        ResearchOutcome.INSUFFICIENT_DATA: "The requested history was insufficient.",
        ResearchOutcome.FAILED: "Research failed because of an operational error.",
    }[outcome]


def _fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()
