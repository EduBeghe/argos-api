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
MAX_EDGE_GUIDED_CANDIDATES = 100
MAX_GENERATED_STRATEGIES = 60
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
            "exit_lower_bound": exit_edge.lower_bound if exit_edge el×MôòÚ$z{-®éÜj×Åö66ƒ¢FV6–ÖÂÀ¢6–×VÆF–öã¢6–×VÆF–öä6öæf–wW&F–öâÀ¢föÆF–Æ—G•÷F‡&W6†öÆC¢FV6–ÖÂÀ¢&WW6&ÆU÷7FvUöWf–FVæ6S¢F–7E·7G"ÂVFvTwV–FVE7FvTWf–FVæ6UÒÀ¢7FvUöWf–FVæ6S¢F–7E·7G"ÂVFvTwV–FVE7FvTWf–FVæ6UÒÀ¢&WW6VEö¶W—3¢6WE·7G%ÒÀ¢6ö×WFVEö¶W—3¢6WE·7G%ÒÀ¢’ÓâVFvTwV–FVE6VvÖVçDWfÇVF–öã ¢6öçFVçEöf–ævW'&–çBÒ÷7FvUö6öçFVçEöf–ævW'&–çB†ÖG&—‚¢66†Uö¶W’ÒVFvUöwV–FVE÷7FvUö66†Uö¶W’€¢6æF–FFRÂv–æF÷rÂ6VvÖVçBÂ6öçFVçEöf–ævW'&–ç@¢¢66†VBÒ&WW6&ÆU÷7FvUöWf–FVæ6RævWB†66†Uö¶W’¢–b66†VB—2æ÷BæöæS ¢&WW6VEö¶W—2æFB†66†Uö¶W’¢7FvUöWf–FVæ6U¶66†Uö¶W•ÒÒ&WÆ6R†66†VBÂv–æF÷uö÷&F–æÃ×v–æF÷ræ÷&F–æÂ¢&WGW&â66†VBæÖWG&–70¢ÖWG&–72ÒöWfÇVFU÷6VvÖVçB€¢FF6WBÀ¢ÖG&—‚À¢6VvÖVçBÀ¢6æF–FFRÀ¢–æ—F–Åö66‚À¢6–×VÆF–öâÀ¢ö6æF–FFUö7F–öåöf–ÇFW"†6æF–FFRÂföÆF–Æ—G•÷F‡&W6†öÆB’À¢¢&÷VæG2Ò°¢&Vf–æVÖVçE6VvÖVçDæÖRåE$”ä”äs¢‡v–æF÷rçG&–æ–æuög&öÒÂv–æF÷rçG&–æ–æu÷Fò’À¢&Vf–æVÖVçE6VvÖVçDæÖRådÄ”DD”ôã¢€¢v–æF÷rçfÆ–FF–öåög&öÒÀ¢v–æF÷rçfÆ–FF–öå÷FòÀ¢’À¢&Vf–æVÖVçE6VvÖVçDæÖRåDU5C¢‡v–æF÷rçFW7Eög&öÒÂv–æF÷rçFW7E÷Fò’À¢Õ·6VvÖVçEÐ¢7FvUöWf–FVæ6U¶66†Uö¶W•ÒÒVFvTwV–FVE7FvTWf–FVæ6R€¢66†Uö¶W’À¢6æF–FFRæ6æF–FFUöf–ævW'&–çBÀ¢6VvÖVçBÀ¢v–æF÷ræ÷&F–æÂÀ¢&÷VæG5³ÒÀ¢&÷VæG5³ÒÀ¢6öçFVçEöf–ævW'&–çBÀ¢ÖWG&–72À¢¢6ö×WFVEö¶W—2æFB†66†Uö¶W’¢&WGW&âÖWG&–70  ¦FVb÷7FvUö6öçFVçEöf–ævW'&–çB†ÖG&—ƒ¢†—7F÷&–6ÄfVGW&TÖG&—‚’Óâ7G# ¢&WGW&âöf–ævW'&–çB‡²'&÷w2#¢GWÆR†6F–7B‡&÷r’f÷"&÷r–âÖG&—‚ç&÷w2—Ò  ¦FVbövw&VvFU÷6VvÖVçB€¢v–æF÷w3¢GWÆU´VFvTwV–FVEv–æF÷tWfÇVF–öâÂââåÒÀ¢6VvÖVçC¢&Vf–æVÖVçE6VvÖVçDæÖRÀ¢’ÓâVFvTwV–FVE6VvÖVçDWfÇVF–öã ¢fÇVW2ÒGWÆR€¢fÇVP¢f÷"—FVÒ–âv–æF÷w0¢f÷"fÇVR–â€¢—FVÒçG&–æ–æp¢–b6VvÖVçB—2&Vf–æVÖVçE6VvÖVçDæÖRåE$”ä”äp¢VÇ6R—FVÒçfÆ–FF–öà¢–b6VvÖVçB—2&Vf–æVÖVçE6VvÖVçDæÖRådÄ”DD”ôà¢VÇ6R—FVÒçFW7BÀ¢¢–bfÇVR—2æ÷BæöæP¢¢–bæ÷BfÇVW3 ¢&WGW&âVFvTwV–FVE6VvÖVçDWfÇVF–öâ€¢6VvÖVçBÂFV6–ÖÂƒ’ÂFV6–ÖÂƒ’ÂÂFV6–ÖÂƒ’ÂFV6–ÖÂƒ’ÂFV6–ÖÂƒ¢¢6÷VçBÒFV6–ÖÂ†ÆVâ‡fÇVW2’¢&WGW&âVFvTwV–FVE6VvÖVçDWfÇVF–öâ€¢6VvÖVçBÀ¢7VÒ‚†—FVÒçF÷FÅ÷&WGW&âf÷"—FVÒ–âfÇVW2’ÂFV6–ÖÂƒ’’ò6÷VçBÀ¢7VÒ‚†—FVÒæÖ…öG&vF÷vâf÷"—FVÒ–âfÇVW2’ÂFV6–ÖÂƒ’’ò6÷VçBÀ¢7VÒ†—FVÒçG&FUö6÷VçBf÷"—FVÒ–âfÇVW2’À¢7VÒ‚†—FVÒæfVW5÷–Bf÷"—FVÒ–âfÇVW2’ÂFV6–ÖÂƒ’’À¢7VÒ‚†—FVÒç&WGW&åöF–ffW&Væ6Uög&öÕ÷&öGV7F–öâf÷"—FVÒ–âfÇVW2’ÂFV6–ÖÂƒ’¢ò6÷VçBÀ¢7VÒ‚†—FVÒç&WGW&åöF–ffW&Væ6Uög&öÕö'W•öæEö†öÆBf÷"—FVÒ–âfÇVW2’ÂFV6–ÖÂƒ’¢ò6÷VçBÀ¢  ¦FVbövw&VvFUö÷F–öæÅ÷FW7B€¢v–æF÷w3¢GWÆU´VFvTwV–FVEv–æF÷tWfÇVF–öâÂââåÒÀ¢’ÓâVFvTwV–FVE6VvÖVçDWfÇVF–öâÂæöæS ¢–bæ÷Bç’†—FVÒçFW7B—2æ÷BæöæRf÷"—FVÒ–âv–æF÷w2“ ¢&WGW&âæöæP¢&WGW&âövw&VvFU÷6VvÖVçB‡v–æF÷w2Â&Vf–æVÖVçE6VvÖVçDæÖRåDU5B  ¦FVb÷&æ¶–ær€¢v–æF÷w3¢GWÆU´VFvTwV–FVEv–æF÷tWfÇVF–öâÂââåÒÂf–ævW'&–çC¢7G ¢’ÓâVFvTwV–FVE&æ¶–æs ¢&WGW&ç2ÒGWÆR‡6÷'FVB†—FVÒçfÆ–FF–öâçF÷FÅ÷&WGW&âf÷"—FVÒ–âv–æF÷w2’¢ÖVF–âÒ€¢&WGW&ç5¶ÆVâ‡&WGW&ç2’òò%Ð¢–bÆVâ‡&WGW&ç2’R ¢VÇ6R‡&WGW&ç5¶ÆVâ‡&WGW&ç2’òò"ÒÒ²&WGW&ç5¶ÆVâ‡&WGW&ç2’òò%Ò’òFV6–ÖÂƒ"¢–b&WGW&ç0¢VÇ6RFV6–ÖÂƒ¢¢v÷'7BÒÖ–â‡&WGW&ç2ÂFVfVÇCÔFV6–ÖÂƒ’¢G&vF÷vâÒÖ‚€¢†—FVÒçfÆ–FF–öâæÖ…öG&vF÷vâf÷"—FVÒ–âv–æF÷w2’ÂFVfVÇCÔFV6–ÖÂƒ¢¢G&FW2Ò7VÒ†—FVÒçfÆ–FF–öâçG&FUö6÷VçBf÷"—FVÒ–âv–æF÷w2¢fVW2Ò7VÒ‚†—FVÒçfÆ–FF–öâæfVW5÷–Bf÷"—FVÒ–âv–æF÷w2’ÂFV6–ÖÂƒ’¢÷6—F—fRÒ7VÒ‡fÇVRâf÷"fÇVR–â&WGW&ç2¢&F–òÒFV6–ÖÂ‡÷6—F—fR’òFV6–ÖÂ†ÆVâ‡&WGW&ç2’’–b&WGW&ç2VÇ6RFV6–ÖÂƒ¢&WGW&âVFvTwV–FVE&æ¶–ær€¢ÖVF–âÀ¢v÷'7BÀ¢G&vF÷vâÀ¢ÖVF–âòG&vF÷vâ–bG&vF÷vââVÇ6RæöæRÀ¢G&FW2À¢fVW2À¢ÖVF–âòfVW2–bfVW2âVÇ6RæöæRÀ¢&F–òÀ¢÷6—F—fRÀ¢f–ævW'&–çBÀ¢  ¦FVbVFvUöwV–FVE÷7VÖÖ'’‡&W7VÇC¢VFvTwV–FVDWfÇVF–öå&W7VÇB’ÓâF–7E·7G"Âö&¦V7EÓ ¢""%&WGW&â7G'V7GW&VB¥4ôâ×6fR¦ö"7VÖÖ'’v—F‚FV6–ÖÂ7G&–æw2â""  ¢fÇVRÒ§6öâæÆöG2†§6öâæGV×2†6F–7B‡&W7VÇB’ÂFVfVÇC×7G"’¢–bæ÷B—6–ç7Fæ6R‡fÇVRÂF–7B“ ¢&—6RVFvTwV–FVD6æF–FFTW'&÷"‚&VFvRÖwV–FVB7VÖÖ'’×W7B&Râö&¦V7B"¢fÇVRç÷‚'7FvUöWf–FVæ6R"ÂæöæR¢&WGW&âfÇVP  ¦FVb÷v–æF÷u÷6VvÖVçE÷&V¦V7F–öå÷&V6öç2€¢fÇVS¢VFvTwV–FVE6VvÖVçDWfÇVF–öâÂ6VvÖVçC¢&Vf–æVÖVçE6VvÖVçDæÖP¢’ÓâGWÆU´VFvTwV–FVE&V¦V7F–öå&V6öâÂââåÓ ¢&V6öç3¢Æ—7E´VFvTwV–FVE&V¦V7F–öå&V6öåÒÒµÐ¢–bæ÷BfÇVRç7Vff–6–VçE÷G&FW3 ¢&V6öç2æVæB„VFvTwV–FVE&V¦V7F–öå&V6öâä”å5Tdd”4”TåEõE$DU2¢–bæ÷BfÇVRç÷6—F—fUöæWE÷&WGW&ã ¢&V6öç2æVæB€¢VFvTwV–FVE&V¦V7F–öå&V6öâääôåõõ4•D•dUõdÄ”DD”ôåõ$UEU$à¢–b6VvÖVçB—2&Vf–æVÖVçE6VvÖVçDæÖRådÄ”DD”ôà¢VÇ6RVFvTwV–FVE&V¦V7F–öå&V6öâääôåõõ4•D•dUõDU5Eõ$UEU$à¢¢–bæ÷BfÇVRæ&VG5÷&öGV7F–öã ¢&V6öç2æVæB€¢VFvTwV–FVE&V¦V7F–öå&V6öâäD”EôäõEô$TEõ$ôET5D”ôåõdÄ”DD”ôà¢–b6VvÖVçB—2&Vf–æVÖVçE6VvÖVçDæÖRådÄ”DD”ôà¢VÇ6RVFvTwV–FVE&V¦V7F–öå&V6öâäD”EôäõEô$TEõ$ôET5D”ôåõDU5@¢¢–bæ÷BfÇVRç&—6µö66WF&ÆS ¢&V6öç2æVæB„VFvTwV–FVE&V¦V7F–öå&V6öâäU„4U54•dUôE$tDõtâ¢&WGW&â÷7F&ÆU÷&V6öç2‡GWÆR‡&V6öç2’  ¦FVbö6æF–FFU÷&V¦V7F–öå÷&V6öç2€¢¢À¢G&–æ–æs¢VFvTwV–FVE6VvÖVçDWfÇVF–öâÀ¢fÆ–FF–öã¢VFvTwV–FVE6VvÖVçDWfÇVF–öâÀ¢FW7C¢VFvTwV–FVE6VvÖVçDWfÇVF–öâÂæöæRÀ¢fÆ–C¢–çBÀ¢6öæf—&ÖVC¢–çBÀ¢&F–ó¢FV6–ÖÂÀ¢F‡&W6†öÆG3¢Wf–FVæ6UF‡&W6†öÆG2À¢’ÓâGWÆU´VFvTwV–FVE&V¦V7F–öå&V6öâÂââåÓ ¢&V6öç3¢Æ—7E´VFvTwV–FVE&V¦V7F–öå&V6öåÒÒµÐ¢–bæ÷BÆÂ†—FVÒç7Vff–6–VçE÷G&FW2f÷"—FVÒ–â‡G&–æ–ærÂfÆ–FF–öâ’“ ¢&V6öç2æVæB„VFvTwV–FVE&V¦V7F–öå&V6öâä”å5Tdd”4”TåEõE$DU2¢f÷"fÇVRÂ&V6öâ–â€¢‡G&–æ–ærÂVFvTwV–FVE&V¦V7F–öå&V6öâääôåõõ4•D•dUõE$”ä”äuõ$UEU$â’À¢‡fÆ–FF–öâÂVFvTwV–FVE&V¦V7F–öå&V6öâääôåõõ4•D•dUõdÄ”DD”ôåõ$UEU$â’À¢“ ¢–bæ÷BfÇVRç÷6—F—fUöæWE÷&WGW&ã ¢&V6öç2æVæB‡&V6öâ¢–bæ÷BfÆ–FF–öâæ&VG5÷&öGV7F–öã ¢&V6öç2æVæB„VFvTwV–FVE&V¦V7F–öå&V6öâäD”EôäõEô$TEõ$ôET5D”ôåõdÄ”DD”ôâ¢–bæ÷BÆÂ†—FVÒç&—6µö66WF&ÆRf÷"—FVÒ–â‡G&–æ–ærÂfÆ–FF–öâ’“ ¢&V6öç2æVæB„VFvTwV–FVE&V¦V7F–öå&V6öâäU„4U54•dUôE$tDõtâ¢–bfÆ–BÂF‡&W6†öÆG2æÖ–æ–×VÕ÷v–æF÷w3 ¢&V6öç2æVæB„VFvTwV–FVE&V¦V7F–öå&V6öâä”å5Tdd”4”TåEõdÄ”Eõt”äDõu2¢fÆ–FF–öå÷76VBÒæ÷B÷fÆ–FF–öå÷7FvUöf–ÆVB€¢G&–æ–æs×G&–æ–ærÀ¢fÆ–FF–öã×fÆ–FF–öâÀ¢fÆ–C×fÆ–BÀ¢F‡&W6†öÆG3×F‡&W6†öÆG2À¢¢–bfÆ–FF–öå÷76VC ¢–bFW7B—2æöæR÷"æ÷BFW7Bç7Vff–6–VçE÷G&FW3 ¢&V6öç2æVæB„VFvTwV–FVE&V¦V7F–öå&V6öâä”å5Tdd”4”TåEõE$DU2¢–bFW7B—2æ÷BæöæRæBæ÷BFW7Bç÷6—F—fUöæWE÷&WGW&ã ¢&V6öç2æVæB„VFvTwV–FVE&V¦V7F–öå&V6öâääôåõõ4•D•dUõDU5Eõ$UEU$â¢–bFW7B—2æ÷BæöæRæBæ÷BFW7Bæ&VG5÷&öGV7F–öã ¢&V6öç2æVæB„VFvTwV–FVE&V¦V7F–öå&V6öâäD”EôäõEô$TEõ$ôET5D”ôåõDU5B¢–bFW7B—2æ÷BæöæRæBæ÷BFW7Bç&—6µö66WF&ÆS ¢&V6öç2æVæB„VFvTwV–FVE&V¦V7F–öå&V6öâäU„4U54•dUôE$tDõtâ¢–b6öæf—&ÖVBÂF‡&W6†öÆG2æÖ–æ–×VÕ÷v–æF÷w3 ¢&V6öç2æVæB„VFvTwV–FVE&V¦V7F–öå&V6öâä”å5Tdd”4”TåEô4ôäd•$ÔTEõt”äDõu2¢–bfÆ–BæB&F–òÂF‡&W6†öÆG2æÖ–æ–×VÕ÷7F&ÆUöF—&V7F–öå÷&F–ó ¢&V6öç2æVæB„VFvTwV–FVE&V¦V7F–öå&V6öâåTå5D$ÄUô5$õ55õt”äDõu2¢&WGW&â÷7F&ÆU÷&V6öç2‡GWÆR‡&V6öç2’  ¦FVb÷7F&ÆU÷&V6öç2€¢&V6öç3¢GWÆU´VFvTwV–FVE&V¦V7F–öå&V6öâÂââåÒÀ¢’ÓâGWÆU´VFvTwV–FVE&V¦V7F–öå&V6öâÂââåÓ ¢&WGW&âGWÆR†F–7Bæg&öÖ¶W—2‡&V6öç2’  ¦FVb÷fÆ–FF–öå÷7FvUöf–ÆVB€¢¢À¢G&–æ–æs¢VFvTwV–FVE6VvÖVçDWfÇVF–öâÀ¢fÆ–FF–öã¢VFvTwV–FVE6VvÖVçDWfÇVF–öâÀ¢fÆ–C¢–çBÀ¢F‡&W6†öÆG3¢Wf–FVæ6UF‡&W6†öÆG2À¢’Óâ&ööÃ ¢&WGW&â€¢æ÷BG&–æ–ærç7Vff–6–VçE÷G&FW0¢÷"æ÷BG&–æ–ærç÷6—F—fUöæWE÷&WGW&à¢÷"æ÷BG&–æ–ærç&—6µö66WF&ÆP¢÷"æ÷BfÆ–FF–öâç7Vff–6–VçE÷G&FW0¢÷"æ÷BfÆ–FF–öâç÷6—F—fUöæWE÷&WGW&à¢÷"æ÷BfÆ–FF–öâæ&VG5÷&öGV7F–öà¢÷"æ÷BfÆ–FF–öâç&—6µö66WF&ÆP¢÷"fÆ–BÂF‡&W6†öÆG2æÖ–æ–×VÕ÷v–æF÷w0¢  ¦FVb÷&V¦V7F–öåöW‡ÆæF–öâ€¢&V6öç3¢GWÆU´VFvTwV–FVE&V¦V7F–öå&V6öâÂââåÒÀ¢’Óâ7G# ¢Æ&VÇ2Ò°¢VFvTwV–FVE&V¦V7F–öå&V6öâä”å5Tdd”4”TåEõE$DU3¢&–ç7Vff–6–VçBG&F–ær7F—f—G’"À¢VFvTwV–FVE&V¦V7F–öå&V6öâääôåõõ4•D•dUõE$”ä”äuõ$UEU$ã¢€¢&æöâ×÷6—F—fRG&–æ–ær&WGW&â ¢’À¢VFvTwV–FVE&V¦V7F–öå&V6öâääôåõõ4•D•dUõdÄ”DD”ôåõ$UEU$ã¢€¢&æöâ×÷6—F—fRfÆ–FF–öâ&WGW&â ¢’À¢VFvTwV–FVE&V¦V7F–öå&V6öâääôåõõ4•D•dUõDU5Eõ$UEU$ã¢&æöâ×÷6—F—fRFW7B&WGW&â"À¢VFvTwV–FVE&V¦V7F–öå&V6öâäD”EôäõEô$TEõ$ôET5D”ôåõdÄ”DD”ôã¢€¢'fÆ–FF–öâF–Bæ÷B7G&–7FÇ’&VB&öGV7F–öâ ¢’À¢VFvTwV–FVE&V¦V7F–öå&V6öâäD”EôäõEô$TEõ$ôET5D”ôåõDU5C¢€¢'FW7BF–Bæ÷B7G&–7FÇ’&VB&öGV7F–öâ ¢’À¢VFvTwV–FVE&V¦V7F–öå&V6öâäU„4U54•dUôE$tDõtã¢€¢&G&vF÷vâW†6VVFVBF†RW†—7F–ær&—6²Æ–Ö—B ¢’À¢VFvTwV–FVE&V¦V7F–öå&V6öâä”å5Tdd”4”TåEõdÄ”Eõt”äDõu3¢€¢&–ç7Vff–6–VçBfÆ–FF–öâv–æF÷w2 ¢’À¢VFvTwV–FVE&V¦V7F–öå&V6öâä”å5Tdd”4”TåEô4ôäd•$ÔTEõt”äDõu3¢€¢&–ç7Vff–6–VçB6öæf—&ÖVBFW7Bv–æF÷w2 ¢’À¢VFvTwV–FVE&V¦V7F–öå&V6öâåTå5D$ÄUô5$õ55õt”äDõu3¢€¢&–ç7Vff–6–VçB7F&–Æ—G’7&÷72v–æF÷w2 ¢’À¢Ð¢&WGW&â€¢$6æF–FFR&V¦V7FVC¢"²#²"æ¦ö–â†Æ&VÇ5·&V6öåÒf÷"&V6öâ–â&V6öç2’²"â ¢  ¦FVb÷fÆ–Eö‡—÷F†W6—2‡fÇVS¢7G&FVw”‡—÷F†W6—2’Óâ&ööÃ ¢&WGW&âæ÷B€¢‡fÇVRæÆ÷vW%ö&÷VæB—2æöæRæBfÇVRçWW%ö&÷VæB—2æöæR¢÷"€¢fÇVRæÆ÷vW%ö&÷VæB—2æ÷BæöæP¢æBfÇVRçWW%ö&÷VæB—2æ÷BæöæP¢æBfÇVRæÆ÷vW%ö&÷VæBãÒfÇVRçWW%ö&÷Væ@¢¢  ¦FVböV6öæöÖ–6ÆÇ•÷f–&ÆR€¢fÇVS¢7G&FVw”‡—÷F†W6—2Â6–×VÆF–öã¢6–×VÆF–öä6öæf–wW&F–öà¢’Óâ&ööÃ ¢w&÷72ÒfÇVRæW‡V7FVEöw&÷75÷&WGW&à¢æWBÒfÇVRæW‡V7FVEöæWE÷&WGW&à¢–bw&÷72—2æöæR÷"æWB—2æöæS ¢&WGW&âfÇ6P¢VçG'•ö×VÇF—Æ–W"Ò„FV6–ÖÂƒ’²6–×VÆF–öâç6Æ—vU÷&FR’¢€¢FV6–ÖÂƒ’²6–×VÆF–öâæfVU÷&FP¢¢W†—Eö×VÇF—Æ–W"Ò„FV6–ÖÂƒ’Ò6–×VÆF–öâç6Æ—vU÷&FR’¢€¢FV6–ÖÂƒ’Ò6–×VÆF–öâæfVU÷&FP¢¢&÷VæE÷G&—ö6÷7BÒVçG'•ö×VÇF—Æ–W"òW†—Eö×VÇF—Æ–W"ÒFV6–ÖÂƒ¢&WGW&âw&÷72â&÷VæE÷G&—ö6÷7B²T4ôäôÔ”5ôTDtUôÔ$t”âæBæWBâT4ôäôÔ”5ôTDtUôÔ$t”à  ¦FVbö‡—÷F†W6—5÷&æ²‡fÇVS¢7G&FVw”‡—÷F†W6—2’ÓâGWÆU´FV6–ÖÂÂFV6–ÖÂÂ7G%Ó ¢V6öæöÖ–5÷fÇVRÒ€¢fÇVRæW‡V7FVEöæWE÷&WGW&à¢–bfÇVRç7VvvW7FVEö7F–öâ—27VvvW7FVD7F–öâäÄôäuôTåE%¢VÇ6RfÇVRæW‡V7FVEöfö–FVEöÆ÷70¢’÷"FV6–ÖÂƒ¢&WGW&â‚ÖV6öæöÖ–5÷fÇVRÂ×fÇVRç7F&–Æ—G•÷&F–òÂfÇVRæ‡—÷F†W6—5ö–B  ¦FVbö6ö×F–&ÆUö6FVæ6W2††÷&—¦öã¢F–ÖVFVÇF’ÓâGWÆU·F–ÖVFVÇFÂââåÓ ¢6ö×F–&ÆRÒGWÆR‡fÇVRf÷"fÇVR–â4äD”DDUô4DTä4U2–bfÇVRÃÒ†÷&—¦öâ¢&WGW&â6ö×F–&ÆU²Ó#¥Ò–b6ö×F–&ÆRVÇ6R„4äD”DDUô4DTä4U5³ÒÂ  ¦FVbö6æF–FFU÷&æ²‡fÇVS¢VFvTwV–FVD6æF–FFR’ÓâGWÆU¶–çBÂ7G%Ó ¢÷&FW"Ò°¢6æF–FFTfÖ–Ç’äTDtUôTåE%•õD”ÔTEôU„•C¢À¢6æF–FFTfÖ–Ç’äTDtUôTåE%•õ4”täÅôU„•C¢À¢6æF–FFTfÖ–Ç’äTDtUôTåE%•ô$4TÄ”äUôU„•C¢"À¢6æF–FFTfÖ–Ç’äU„•Eôõ%ôdô”EôõdU$Ä“¢2À¢Ð¢&WGW&â÷&FW%·fÇVRæ6æF–FFUöfÖ–Ç•ÒÂfÇVRæ6æF–FFUöf–ævW'&–ç@  ¦FVbö6æF–FFUöÖF6†W2€¢&÷s¢†—7F÷&–6ÄfVGW&U&÷rÀ¢¢À¢&VF–7F÷%öæÖS¢7G"À¢6öæF—F–öã¢6öæF—F–öâÀ¢Æ÷vW%ö&÷VæC¢FV6–ÖÂÂæöæRÀ¢WW%ö&÷VæC¢FV6–ÖÂÂæöæRÀ¢föÆF–Æ—G•÷F‡&W6†öÆC¢FV6–ÖÂÀ¢’Óâ&ööÃ ¢–bæ÷B&VF–7F÷%öæÖS ¢&WGW&âfÇ6P¢fÇVRÒ&VF–7F—fUöVFvU÷fÇVR‡&÷rÂ&VF–7F÷%öæÖR¢&WGW&â€¢†Æ÷vW%ö&÷VæB—2æöæR÷"fÇVRâÆ÷vW%ö&÷VæB¢æB‡WW%ö&÷VæB—2æöæR÷"fÇVRÃÒWW%ö&÷VæB¢æB&VF–7F—fUöVFvUö6öæF—F–öåöÖF6†W2€¢&÷rÂ6öæF—F–öâÂföÆF–Æ—G•÷F‡&W6†öÆC×föÆF–Æ—G•÷F‡&W6†öÆ@¢¢  ¦FVbö6æF–FFUö6öçFW‡EöÖF6†W2€¢&÷s¢†—7F÷&–6ÄfVGW&U&÷rÂ6æF–FFS¢VFvTwV–FVD6æF–FFP¢’Óâ&ööÃ ¢–b6æF–FFRæ6öçFW‡E÷&VF–7F÷%öæÖR—2æöæS ¢&WGW&âG'VP¢fÇVRÒ&VF–7F—fUöVFvU÷fÇVR‡&÷rÂ6æF–FFRæ6öçFW‡E÷&VF–7F÷%öæÖR¢&WGW&â€¢6æF–FFRæ6öçFW‡EöÆ÷vW%ö&÷VæB—2æöæR÷"fÇVRãÒ6æF–FFRæ6öçFW‡EöÆ÷vW%ö&÷Væ@¢’æB€¢6æF–FFRæ6öçFW‡E÷WW%ö&÷VæB—2æöæR÷"fÇVRÃÒ6æF–FFRæ6öçFW‡E÷WW%ö&÷Væ@¢  ¦FVb÷&W6V&6…ö÷WF6öÖR€¢WfÇVF–öç3¢GWÆU´VFvTwV–FVD6æF–FFTWfÇVF–öâÂââåÒÀ¢VFvW5öf÷VæC¢&ööÂÀ¢6æF–FFW5övVæW&FVC¢&ööÂÀ¢’Óâ&W6V&6„÷WF6öÖS ¢–bç’†—FVÒç7FGW2—2VFvTwV–FVD6æF–FFU7FGW2ä4ôäd•$ÔTBf÷"—FVÒ–âWfÇVF–öç2“ ¢&WGW&â&W6V&6„÷WF6öÖRä4ôäd•$ÔTEô4äD”DDU5ôdõTä@¢–bç’€¢—FVÒç7FGW2—2VFvTwV–FVD6æF–FFU7FGW2å$T¤T5DTEõDU5Bf÷"—FVÒ–âWfÇVF–öç0¢“ ¢&WGW&â&W6V&6„÷WF6öÖRä4äD”DDU5õ$T¤T5DTEõDU5@¢–bWfÇVF–öç3 ¢&WGW&â&W6V&6„÷WF6öÖRä4äD”DDU5õ$T¤T5DTEõdÄ”DD”ôà¢–bVFvW5öf÷VæBæBæ÷B6æF–FFW5övVæW&FVC ¢&WGW&â&W6V&6„÷WF6öÖRäTDtU5ôäõEôT4ôäôÔ”0¢&WGW&â&W6V&6„÷WF6öÖRääõõ$TD”5D•dUôTDtP  ¦FVbö÷WF6öÖUöW‡ÆæF–öâ†÷WF6öÖS¢&W6V&6„÷WF6öÖR’Óâ7G# ¢&WGW&â°¢&W6V&6„÷WF6öÖRä4ôäd•$ÔTEô4äD”DDU5ôdõTäC¢€¢$BÆV7BöæR6ö×ÆWFR7G&FVw’76VB—6öÆFVBfÆ–FF–öâæBFW7Bâ ¢’À¢&W6V&6„÷WF6öÖRä4äD”DDU5õ$T¤T5DTEõdÄ”DD”ôã¢€¢$6ö×ÆWFR7G&FVv–W2vW&RvVæW&FVB'WBf–ÆVBfÆ–FF–öã²FW7Bv2æ÷B ¢'W6VBFòF§W7BF†VÒâ ¢’À¢&W6V&6„÷WF6öÖRä4äD”DDU5õ$T¤T5DTEõDU5C¢€¢$6æF–FFW2vW&R&öf—F&ÆR–âfÆ–FF–öâ'WBf–ÆVB—6öÆFVBFW7Bâ ¢’À¢&W6V&6„÷WF6öÖRääõõ$TD”5D•dUôTDtS¢€¢$æò7F&ÆR&VF–7F—fRVFvRv2FWFV7FVBv—F‚F†R7W'&VçBWf–FVæ6Râ ¢’À¢&W6V&6„÷WF6öÖRäTDtU5ôäõEôT4ôäôÔ”3¢€¢%GFW&ç2vW&RFWFV7FVBÂ'WBW‡V7FVBÖ÷fVÖVçBF–Bæ÷B6÷fW"&÷VæB×G&— ¢&6÷7G2æBF†RfW'6–öæVBÖ&v–ââ ¢’À¢&W6V&6„÷WF6öÖRä”å5Tdd”4”TåEôô45U%$Tä4U3¢€¢$6æF–FFRWf–FVæ6RF–Bæ÷Bö67W"ögFVâVæ÷Vv‚f÷"WfÇVF–öââ ¢’À¢&W6V&6„÷WF6öÖRä”å5Tdd”4”TåEôDD¢%F†R&WVW7FVB†—7F÷'’v2–ç7Vff–6–VçBâ"À¢&W6V&6„÷WF6öÖRäd”ÄTC¢%&W6V&6‚f–ÆVB&V6W6Röbâ÷W&F–öæÂW'&÷"â"À¢Õ¶÷WF6öÖUÐ  ¦FVböf–ævW'&–çB‡–ÆöC¢F–7E·7G"Âö&¦V7EÒ’Óâ7G# ¢Væ6öFVBÒ§6öâæGV×2‡–ÆöBÂ6÷'Eö¶W—3ÕG'VRÂ6W&F÷'3Ò‚"Â"Â#¢"’ÂFVfVÇC×7G"¢&WGW&â†6†Æ–"ç6†#Sb†Væ6öFVBæVæ6öFR‚’’æ†W†F–vW7B‚ 