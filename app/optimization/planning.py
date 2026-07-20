from dataclasses import dataclass
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal

from app.optimization.experience import (
    CandidateHistoricalProfile,
    EvidenceClassification,
)
from app.scores.calculator import BaselineTechnicalParameters
from app.strategies.baseline_long_only import BaselineLongOnlyParameters

REFINEMENT_PLANNER_VERSION = 1
_WEIGHT_QUANTUM = Decimal("0.000000000000000001")
_SCORE_STEP = Decimal("0.05")
_VOLUME_STEP = Decimal("0.125")
_THRESHOLD_STEP = Decimal("0.05")
_TARGET_STEP = Decimal("5")
_ENTRY_RANGE = (Decimal("0.55"), Decimal("0.70"))
_EXIT_RANGE = (Decimal("0.10"), Decimal("0.30"))
_TARGET_RANGE = (Decimal("10"), Decimal("20"))


@dataclass(frozen=True, slots=True)
class PlannerAllocation:
    known_rechecks: Decimal = Decimal("0.15")
    promising_neighborhoods: Decimal = Decimal("0.50")
    uncertain_rechecks: Decimal = Decimal("0.15")
    novel_exploration: Decimal = Decimal("0.20")
    reconsider_rejected_after_runs: int = 4

    def __post_init__(self) -> None:
        values = (
            self.known_rechecks,
            self.promising_neighborhoods,
            self.uncertain_rechecks,
            self.novel_exploration,
        )
        if any(value < 0 or value > 1 for value in values) or sum(values) != 1:
            raise ValueError(
                "planner allocations must be between zero and one and sum to one"
            )
        if self.novel_exploration <= 0:
            raise ValueError("novel exploration allocation must be positive")
        if self.reconsider_rejected_after_runs <= 0:
            raise ValueError("rejected reconsideration runs must be positive")


DEFAULT_PLANNER_ALLOCATION = PlannerAllocation()


@dataclass(frozen=True, slots=True)
class CandidateConfiguration:
    stable_candidate_id: str
    score_parameters: BaselineTechnicalParameters
    strategy_parameters: BaselineLongOnlyParameters


@dataclass(frozen=True, slots=True)
class HistoricalCandidate:
    configuration: CandidateConfiguration
    profile: CandidateHistoricalProfile
    evaluated_on_current_dataset: bool


@dataclass(frozen=True, slots=True)
class SearchPlan:
    planner_version: int
    allocation: PlannerAllocation
    known_rechecks: tuple[CandidateConfiguration, ...]
    promising_neighborhoods: tuple[CandidateConfiguration, ...]
    uncertain_rechecks: tuple[CandidateConfiguration, ...]
    novel_exploration: tuple[CandidateConfiguration, ...]

    @property
    def candidates(self) -> tuple[CandidateConfiguration, ...]:
        return (
            *self.known_rechecks,
            *self.promising_neighborhoods,
            *self.uncertain_rechecks,
            *self.novel_exploration,
        )

    @property
    def counts(self) -> dict[str, int]:
        return {
            "known_rechecks": len(self.known_rechecks),
            "promising_neighborhoods": len(self.promising_neighborhoods),
            "uncertain_rechecks": len(self.uncertain_rechecks),
            "novel_exploration": len(self.novel_exploration),
        }


def build_search_plan(
    *,
    historical: tuple[HistoricalCandidate, ...],
    novel_pool: tuple[CandidateConfiguration, ...],
    max_candidates: int,
    production_baseline: CandidateConfiguration,
    allocation: PlannerAllocation = DEFAULT_PLANNER_ALLOCATION,
) -> SearchPlan | None:
    """Plan a bounded deterministic search; no history preserves legacy fallback."""
    if not historical:
        return None
    # The baseline consumes the first slot. Only the remainder is allocated, so the
    # planner cannot exceed its budget while guaranteeing a production reference.
    quotas = _allocate(max_candidates - 1, allocation)
    used: set[str] = {production_baseline.stable_candidate_id}

    def take(
        values: tuple[CandidateConfiguration, ...], amount: int
    ) -> tuple[CandidateConfiguration, ...]:
        if amount <= 0:
            return ()
        selected = []
        for value in values:
            if value.stable_candidate_id not in used:
                selected.append(value)
                used.add(value.stable_candidate_id)
            if len(selected) == amount:
                break
        return tuple(selected)

    known_profiles = tuple(
        sorted(
            (
                item
                for item in historical
                if not item.evaluated_on_current_dataset
                and item.profile.evidence_classification
                in {
                    EvidenceClassification.PAPER_TRIAL_READY,
                    EvidenceClassification.ROBUST_CANDIDATE,
                    EvidenceClassification.PROMISING,
                }
            ),
            key=_known_key,
        )
    )
    known = (
        production_baseline,
        *take(tuple(item.configuration for item in known_profiles), quotas[0]),
    )
    promising_sources = tuple(
        item.configuration
        for item in known_profiles
        if item.profile.evidence_classification
        in {
            EvidenceClassification.PAPER_TRIAL_READY,
            EvidenceClassification.ROBUST_CANDIDATE,
            EvidenceClassification.PROMISING,
        }
    )
    neighbors = take(
        tuple(
            neighbor for source in promising_sources for neighbor in _neighbors(source)
        ),
        quotas[1],
    )
    uncertain_profiles = tuple(
        sorted(
            (
                item
                for item in historical
                if (
                    item.profile.evidence_classification
                    is EvidenceClassification.INSUFFICIENT_EVIDENCE
                    or (
                        item.profile.evidence_classification
                        is EvidenceClassification.HISTORICALLY_REJECTED
                        and item.profile.compatible_run_count
                        >= allocation.reconsider_rejected_after_runs
                    )
                )
            ),
            key=lambda item: (
                item.profile.last_evaluated_at,
                -item.profile.training_evaluation_count,
                item.configuration.stable_candidate_id,
            ),
        )
    )
    uncertain = take(
        tuple(item.configuration for item in uncertain_profiles), quotas[2]
    )
    evaluation_counts = {
        item.configuration.stable_candidate_id: item.profile.compatible_run_count
        for item in historical
    }
    novel_ordered = tuple(
        sorted(
            novel_pool,
            key=lambda item: (
                evaluation_counts.get(item.stable_candidate_id, 0),
                item.stable_candidate_id,
            ),
        )
    )
    novel = take(novel_ordered, quotas[3])
    categories = [known, neighbors, uncertain, novel]
    missing = max_candidates - sum(len(value) for value in categories)
    if missing:
        categories[3] = (*categories[3], *take(novel_ordered, missing))
    return SearchPlan(
        REFINEMENT_PLANNER_VERSION,
        allocation,
        categories[0],
        categories[1],
        categories[2],
        categories[3],
    )


def _allocate(total: int, allocation: PlannerAllocation) -> tuple[int, int, int, int]:
    if total < 0:
        raise ValueError("planner candidate budget must be positive")
    if total == 0:
        return 0, 0, 0, 0
    weights = (
        allocation.known_rechecks,
        allocation.promising_neighborhoods,
        allocation.uncertain_rechecks,
        allocation.novel_exploration,
    )
    quotas = [
        int((Decimal(total) * value).to_integral_value(rounding=ROUND_FLOOR))
        for value in weights
    ]
    for index in range(total - sum(quotas)):
        quotas[index % len(quotas)] += 1
    if quotas[3] == 0:
        donor = max(range(3), key=quotas.__getitem__)
        quotas[donor] -= 1
        quotas[3] = 1
    return quotas[0], quotas[1], quotas[2], quotas[3]


def _known_key(item: HistoricalCandidate) -> tuple[int, Decimal, str]:
    priority = {
        EvidenceClassification.PAPER_TRIAL_READY: 0,
        EvidenceClassification.ROBUST_CANDIDATE: 1,
        EvidenceClassification.PROMISING: 2,
    }
    return (
        priority[item.profile.evidence_classification],
        -item.profile.validation_win_rate,
        item.configuration.stable_candidate_id,
    )


def _neighbors(source: CandidateConfiguration) -> tuple[CandidateConfiguration, ...]:
    generated: list[CandidateConfiguration] = []
    score = source.score_parameters
    directional = (
        score.trend_weight,
        score.rsi_weight,
        score.return_1_weight,
        score.return_12_weight,
    )
    for from_index in range(4):
        for to_index in range(4):
            if from_index == to_index or directional[from_index] < _SCORE_STEP:
                continue
            values = list(directional)
            values[from_index] -= _SCORE_STEP
            values[to_index] += _SCORE_STEP
            generated.append(
                _configuration(
                    BaselineTechnicalParameters(
                        values[0].quantize(_WEIGHT_QUANTUM, rounding=ROUND_HALF_EVEN),
                        values[1].quantize(_WEIGHT_QUANTUM, rounding=ROUND_HALF_EVEN),
                        values[2].quantize(_WEIGHT_QUANTUM, rounding=ROUND_HALF_EVEN),
                        values[3].quantize(_WEIGHT_QUANTUM, rounding=ROUND_HALF_EVEN),
                        score.volume_confirmation_weight,
                    ),
                    source.strategy_parameters,
                )
            )
    for delta in (-_VOLUME_STEP, _VOLUME_STEP):
        volume = score.volume_confirmation_weight + delta
        if Decimal(0) <= volume <= Decimal("0.50"):
            generated.append(
                _configuration(
                    BaselineTechnicalParameters(*directional, volume),
                    source.strategy_parameters,
                )
            )
    strategy = source.strategy_parameters
    variations = (
        (
            strategy.entry_threshold - _THRESHOLD_STEP,
            strategy.exit_threshold,
            strategy.open_position_target_percentage,
        ),
        (
            strategy.entry_threshold + _THRESHOLD_STEP,
            strategy.exit_threshold,
            strategy.open_position_target_percentage,
        ),
        (
            strategy.entry_threshold,
            strategy.exit_threshold - _THRESHOLD_STEP,
            strategy.open_position_target_percentage,
        ),
        (
            strategy.entry_threshold,
            strategy.exit_threshold + _THRESHOLD_STEP,
            strategy.open_position_target_percentage,
        ),
        (
            strategy.entry_threshold,
            strategy.exit_threshold,
            strategy.open_position_target_percentage - _TARGET_STEP,
        ),
        (
            strategy.entry_threshold,
            strategy.exit_threshold,
            strategy.open_position_target_percentage + _TARGET_STEP,
        ),
    )
    for entry, exit_, target in variations:
        if not (
            _ENTRY_RANGE[0] <= entry <= _ENTRY_RANGE[1]
            and _EXIT_RANGE[0] <= exit_ <= _EXIT_RANGE[1]
            and _TARGET_RANGE[0] <= target <= _TARGET_RANGE[1]
        ):
            continue
        try:
            changed = BaselineLongOnlyParameters(entry, exit_, target)
        except ValueError:
            continue
        generated.append(_configuration(score, changed))
    unique = {item.stable_candidate_id: item for item in generated}
    return tuple(unique[key] for key in sorted(unique))


def _configuration(
    score: BaselineTechnicalParameters, strategy: BaselineLongOnlyParameters
) -> CandidateConfiguration:
    from app.optimization.refinement import candidate_for

    candidate = candidate_for(score, strategy)
    return CandidateConfiguration(candidate.stable_candidate_id, score, strategy)
