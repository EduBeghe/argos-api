from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from app.market.catalog import Instrument

CANDIDATE_PROFILE_VERSION = 1
MAX_PROFILE_DRAWDOWN = Decimal("0.10")


class EvidenceClassification(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PROMISING = "promising"
    ROBUST_CANDIDATE = "robust_candidate"
    PAPER_TRIAL_READY = "paper_trial_ready"
    HISTORICALLY_REJECTED = "historically_rejected"


@dataclass(frozen=True, slots=True)
class HistoricalEvaluation:
    stable_candidate_id: str
    run_id: str
    instrument: Instrument
    segment: str
    total_return: Decimal
    baseline_total_return: Decimal
    max_drawdown: Decimal
    trade_count: int
    fees_paid: Decimal
    eligible: bool | None
    evaluated_at: datetime
    test_confirmed: bool | None = None


@dataclass(frozen=True, slots=True)
class CandidateHistoricalProfile:
    stable_candidate_id: str
    compatible_run_count: int
    training_evaluation_count: int
    validation_evaluation_count: int
    test_evaluation_count: int
    validation_win_count: int
    test_confirmation_count: int
    validation_win_rate: Decimal
    test_confirmation_rate: Decimal
    median_validation_excess_return: Decimal | None
    median_test_excess_return: Decimal | None
    median_validation_return: Decimal | None
    median_test_return: Decimal | None
    worst_validation_drawdown: Decimal | None
    worst_test_drawdown: Decimal | None
    median_trade_count: Decimal | None
    median_fees_paid: Decimal | None
    first_evaluated_at: datetime
    last_evaluated_at: datetime
    consecutive_recent_failures: int
    instruments: tuple[Instrument, ...]
    profile_version: int
    evidence_classification: EvidenceClassification


def build_candidate_profiles(
    evaluations: tuple[HistoricalEvaluation, ...],
) -> tuple[CandidateHistoricalProfile, ...]:
    grouped: dict[str, list[HistoricalEvaluation]] = {}
    for evaluation in evaluations:
        grouped.setdefault(evaluation.stable_candidate_id, []).append(evaluation)
    return tuple(
        _profile(candidate_id, tuple(grouped[candidate_id]))
        for candidate_id in sorted(grouped)
    )


def _profile(
    candidate_id: str, evaluations: tuple[HistoricalEvaluation, ...]
) -> CandidateHistoricalProfile:
    ordered = tuple(
        sorted(evaluations, key=lambda item: (item.evaluated_at, item.run_id))
    )
    training = tuple(item for item in ordered if item.segment == "training")
    validation = tuple(item for item in ordered if item.segment == "validation")
    test = tuple(item for item in ordered if item.segment == "test")
    validation_wins = sum(
        item.total_return > item.baseline_total_return for item in validation
    )
    confirmations = sum(item.test_confirmed is True for item in test)
    recent_failures = 0
    recent = sorted(
        (*validation, *test), key=lambda item: (item.evaluated_at, item.run_id)
    )
    for item in reversed(recent):
        succeeded = (
            item.test_confirmed
            if item.segment == "test"
            else item.total_return > item.baseline_total_return
        )
        if succeeded:
            break
        recent_failures += 1
    instruments = tuple(
        sorted(
            {item.instrument for item in ordered},
            key=lambda value: (
                value.asset.value,
                value.provider.value,
                value.market_type.value,
                value.interval.value,
            ),
        )
    )
    provisional = CandidateHistoricalProfile(
        candidate_id,
        len({item.run_id for item in ordered}),
        len(training),
        len(validation),
        len(test),
        validation_wins,
        confirmations,
        _rate(validation_wins, len(validation)),
        _rate(confirmations, len(test)),
        _median(
            tuple(item.total_return - item.baseline_total_return for item in validation)
        ),
        _median(tuple(item.total_return - item.baseline_total_return for item in test)),
        _median(tuple(item.total_return for item in validation)),
        _median(tuple(item.total_return for item in test)),
        max((item.max_drawdown for item in validation), default=None),
        max((item.max_drawdown for item in test), default=None),
        _median(tuple(Decimal(item.trade_count) for item in ordered)),
        _median(tuple(item.fees_paid for item in ordered)),
        ordered[0].evaluated_at,
        ordered[-1].evaluated_at,
        recent_failures,
        instruments,
        CANDIDATE_PROFILE_VERSION,
        EvidenceClassification.INSUFFICIENT_EVIDENCE,
    )
    return replace(provisional, evidence_classification=classify_evidence(provisional))


def classify_evidence(profile: CandidateHistoricalProfile) -> EvidenceClassification:
    if profile.validation_evaluation_count >= 3 and (
        profile.validation_win_rate == 0
        or _non_positive(profile.median_validation_excess_return)
        or profile.consecutive_recent_failures >= 3
    ):
        return EvidenceClassification.HISTORICALLY_REJECTED
    if (
        profile.validation_evaluation_count >= 4
        and profile.test_evaluation_count >= 3
        and profile.validation_win_rate >= Decimal("0.75")
        and profile.test_confirmation_rate >= Decimal("0.66")
        and _positive(profile.median_test_excess_return)
        and _within(profile.worst_test_drawdown)
        and profile.consecutive_recent_failures < 2
    ):
        return EvidenceClassification.PAPER_TRIAL_READY
    if (
        profile.validation_evaluation_count >= 3
        and profile.test_evaluation_count >= 2
        and profile.validation_win_rate >= Decimal("0.66")
        and profile.test_confirmation_rate >= Decimal("0.50")
        and _positive(profile.median_validation_excess_return)
        and _positive(profile.median_test_excess_return)
        and _within(profile.worst_test_drawdown)
    ):
        return EvidenceClassification.ROBUST_CANDIDATE
    if (
        profile.validation_evaluation_count >= 2
        and profile.validation_win_rate >= Decimal("0.50")
        and _positive(profile.median_validation_excess_return)
        and _within(profile.worst_validation_drawdown)
    ):
        return EvidenceClassification.PROMISING
    return EvidenceClassification.INSUFFICIENT_EVIDENCE


def _median(values: tuple[Decimal, ...]) -> Decimal | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _rate(numerator: int, denominator: int) -> Decimal:
    return Decimal(numerator) / Decimal(denominator) if denominator else Decimal(0)


def _positive(value: Decimal | None) -> bool:
    return value is not None and value > 0


def _non_positive(value: Decimal | None) -> bool:
    return value is not None and value <= 0


def _within(value: Decimal | None) -> bool:
    return value is not None and value <= MAX_PROFILE_DRAWDOWN
