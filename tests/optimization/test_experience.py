from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.experience import (
    CANDIDATE_PROFILE_VERSION,
    CandidateHistoricalProfile,
    EvidenceClassification,
    HistoricalEvaluation,
    build_candidate_profiles,
    classify_evidence,
)

NOW = datetime(2025, 1, 1, tzinfo=UTC)
INSTRUMENT = Instrument(
    Asset.BTC_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)


def profile(**changes: object) -> CandidateHistoricalProfile:
    value = CandidateHistoricalProfile(
        "candidate",
        4,
        4,
        4,
        3,
        3,
        2,
        Decimal("0.75"),
        Decimal("0.666666666666666666"),
        Decimal("0.02"),
        Decimal("0.01"),
        Decimal("0.03"),
        Decimal("0.02"),
        Decimal("0.08"),
        Decimal("0.09"),
        Decimal("2"),
        Decimal("3.5"),
        NOW,
        NOW,
        0,
        (INSTRUMENT,),
        CANDIDATE_PROFILE_VERSION,
        EvidenceClassification.INSUFFICIENT_EVIDENCE,
    )
    return replace(value, **changes)  # type: ignore[arg-type]


def test_evidence_classifications_are_centralized_and_versioned() -> None:
    assert (
        classify_evidence(profile(validation_evaluation_count=1))
        is EvidenceClassification.INSUFFICIENT_EVIDENCE
    )
    assert (
        classify_evidence(
            profile(
                validation_evaluation_count=2,
                test_evaluation_count=0,
                validation_win_rate=Decimal("0.50"),
            )
        )
        is EvidenceClassification.PROMISING
    )
    assert (
        classify_evidence(
            profile(
                validation_evaluation_count=3,
                test_evaluation_count=2,
                validation_win_rate=Decimal("0.67"),
                test_confirmation_rate=Decimal("0.50"),
            )
        )
        is EvidenceClassification.ROBUST_CANDIDATE
    )
    assert classify_evidence(profile()) is EvidenceClassification.PAPER_TRIAL_READY
    assert (
        classify_evidence(
            profile(validation_win_rate=Decimal(0), validation_win_count=0)
        )
        is EvidenceClassification.HISTORICALLY_REJECTED
    )
    assert (
        classify_evidence(profile(consecutive_recent_failures=3))
        is EvidenceClassification.HISTORICALLY_REJECTED
    )


def test_profiles_use_decimal_medians_rates_recent_failures_and_instruments() -> None:
    other = Instrument(
        Asset.ETH_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
    )
    values = (
        HistoricalEvaluation(
            "id",
            "run-1",
            INSTRUMENT,
            "validation",
            Decimal("0.03"),
            Decimal("0.01"),
            Decimal("0.04"),
            1,
            Decimal("2"),
            True,
            NOW,
        ),
        HistoricalEvaluation(
            "id",
            "run-2",
            other,
            "validation",
            Decimal("0.01"),
            Decimal("0.02"),
            Decimal("0.11"),
            3,
            Decimal("4"),
            False,
            NOW + timedelta(days=1),
        ),
        HistoricalEvaluation(
            "id",
            "run-2",
            other,
            "test",
            Decimal("0.04"),
            Decimal("0.02"),
            Decimal("0.08"),
            2,
            Decimal("3"),
            True,
            NOW + timedelta(days=2),
            True,
        ),
        HistoricalEvaluation(
            "id",
            "run-3",
            INSTRUMENT,
            "test",
            Decimal("0.00"),
            Decimal("0.01"),
            Decimal("0.09"),
            4,
            Decimal("5"),
            False,
            NOW + timedelta(days=3),
            False,
        ),
    )
    value = build_candidate_profiles(values)[0]
    assert value.compatible_run_count == 3
    assert value.validation_win_rate == Decimal("0.5")
    assert value.test_confirmation_rate == Decimal("0.5")
    assert value.median_validation_excess_return == Decimal("0.005")
    assert value.median_test_excess_return == Decimal("0.005")
    assert value.median_trade_count == Decimal("2.5")
    assert value.median_fees_paid == Decimal("3.5")
    assert value.consecutive_recent_failures == 1
    assert value.instruments == (INSTRUMENT, other)
    assert value.first_evaluated_at == NOW
    assert value.last_evaluated_at == NOW + timedelta(days=3)
