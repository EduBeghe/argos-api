from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.jobs.analyze_predictive_edges import build_parser
from app.optimization.edge_guided import (
    EdgeGuidedCandidate,
    apply_edge_guided_action,
    generate_edge_guided_candidates,
)
from app.optimization.feature_matrix import HistoricalFeatureRow
from app.optimization.predictive_edges import (
    Condition,
    DirectionMapping,
    EvidenceStatus,
    StrategyHypothesis,
    SuggestedAction,
)
from app.paper.config import SimulationConfiguration
from app.strategies.baseline_long_only import StrategyAction


def hypothesis(
    *,
    identifier: str = "a" * 64,
    action: SuggestedAction = SuggestedAction.LONG_ENTRY,
    lower: Decimal | None = Decimal("-0.02"),
    upper: Decimal | None = Decimal("-0.01"),
    horizon: timedelta = timedelta(minutes=30),
) -> StrategyHypothesis:
    return StrategyHypothesis(
        identifier,
        "return_1",
        3,
        2,
        DirectionMapping.NEGATIVE,
        1,
        lower,
        upper,
        action,
        horizon,
        Condition.ALL,
        (1,),
        Decimal("0.01") if action is SuggestedAction.LONG_ENTRY else None,
        Decimal("0.007") if action is SuggestedAction.LONG_ENTRY else None,
        Decimal("0.007") if action is SuggestedAction.EXIT_OR_AVOID else None,
        Decimal("0.70"),
        Decimal("0.75"),
        timedelta(minutes=30),
        timedelta(minutes=15),
        horizon,
        (1, 2, 3),
        EvidenceStatus.PROMISING_POSITIVE
        if action is SuggestedAction.LONG_ENTRY
        else EvidenceStatus.PROMISING_NEGATIVE,
        "Confirmed frozen bin evidence.",
        "Historical evidence is not a guarantee.",
    )


def row(return_1: str) -> HistoricalFeatureRow:
    return HistoricalFeatureRow(
        datetime(2026, 1, 1, tzinfo=UTC),
        Decimal("100"),
        Decimal("99"),
        Decimal("98"),
        Decimal("50"),
        Decimal("2"),
        Decimal(return_1),
        Decimal("0"),
        Decimal("1"),
    )


def candidates(
    values: tuple[StrategyHypothesis, ...],
    *,
    maximum: int = 12,
    fee_rate: Decimal = Decimal("0.001"),
    slippage_rate: Decimal = Decimal("0.0005"),
) -> tuple[EdgeGuidedCandidate, ...]:
    return generate_edge_guided_candidates(
        hypotheses=values,
        simulation=SimulationConfiguration(
            fee_rate=fee_rate, slippage_rate=slippage_rate
        ),
        maximum_candidates=maximum,
    )


def test_confirmed_long_and_exit_hypotheses_generate_one_candidate_each() -> None:
    long = hypothesis()
    exit_candidate = hypothesis(
        identifier="b" * 64,
        action=SuggestedAction.EXIT_OR_AVOID,
        lower=Decimal("0.01"),
        upper=Decimal("0.02"),
    )
    generated = candidates((long, exit_candidate))
    assert len(generated) == 2
    assert {item.suggested_action for item in generated} == {
        SuggestedAction.LONG_ENTRY,
        SuggestedAction.EXIT_OR_AVOID,
    }


def test_diagnostic_or_unconfirmed_evidence_produces_no_candidate() -> None:
    diagnostic = hypothesis(action=SuggestedAction.DIAGNOSTIC_ONLY)
    assert candidates((diagnostic,)) == ()
    assert candidates(()) == ()


def test_long_entry_only_opens_inside_the_confirmed_bin() -> None:
    candidate = candidates((hypothesis(),))[0]
    assert (
        apply_edge_guided_action(
            candidate=candidate,
            row=row("-0.015"),
            position_is_open=False,
            baseline_action=StrategyAction.OPEN_POSITION,
        )
        is StrategyAction.OPEN_POSITION
    )
    assert (
        apply_edge_guided_action(
            candidate=candidate,
            row=row("0.01"),
            position_is_open=False,
            baseline_action=StrategyAction.OPEN_POSITION,
        )
        is StrategyAction.NO_ACTION
    )


def test_exit_or_avoid_never_creates_a_short_position() -> None:
    source = hypothesis(
        action=SuggestedAction.EXIT_OR_AVOID,
        lower=Decimal("0.01"),
        upper=Decimal("0.02"),
    )
    candidate = candidates((source,))[0]
    assert (
        apply_edge_guided_action(
            candidate=candidate,
            row=row("0.015"),
            position_is_open=False,
            baseline_action=StrategyAction.OPEN_POSITION,
        )
        is StrategyAction.NO_ACTION
    )
    assert (
        apply_edge_guided_action(
            candidate=candidate,
            row=row("0.015"),
            position_is_open=True,
            baseline_action=StrategyAction.NO_ACTION,
        )
        is StrategyAction.SELL_ALL
    )
    assert StrategyAction.INCREASE_POSITION not in {
        apply_edge_guided_action(
            candidate=candidate,
            row=row("0.015"),
            position_is_open=False,
            baseline_action=StrategyAction.OPEN_POSITION,
        )
    }


def test_limit_order_and_fingerprints_are_deterministic_and_complete() -> None:
    values = tuple(hypothesis(identifier=f"{index:064x}") for index in range(1, 5))
    first = candidates(tuple(reversed(values)), maximum=2)
    second = candidates(values, maximum=2)
    assert first == second
    assert len(first) == 2
    base = candidates((hypothesis(),))[0]
    variants = (
        candidates((replace(hypothesis(), lower_bound=Decimal("-0.03")),))[0],
        candidates(
            (
                hypothesis(
                    action=SuggestedAction.EXIT_OR_AVOID,
                    lower=Decimal("0.01"),
                    upper=Decimal("0.02"),
                ),
            )
        )[0],
        candidates((hypothesis(horizon=timedelta(hours=1)),))[0],
        candidates((hypothesis(),), fee_rate=Decimal("0.002"))[0],
    )
    assert all(
        item.candidate_fingerprint != base.candidate_fingerprint for item in variants
    )
    assert base.fee_rate == Decimal("0.001")
    assert base.slippage_rate == Decimal("0.0005")


def test_cli_keeps_edge_guided_evaluation_explicitly_disabled_by_default() -> None:
    required = [
        "--asset",
        "BTC/USDT",
        "--provider",
        "binance",
        "--market-type",
        "spot",
        "--interval",
        "5m",
        "--from",
        "2025-01-01T00:00:00Z",
        "--to",
        "2026-01-01T00:00:00Z",
    ]
    assert build_parser().parse_args(required).edge_guided_candidates is False
    enabled = build_parser().parse_args([*required, "--edge-guided-candidates"])
    assert enabled.edge_guided_candidates is True
    assert enabled.max_edge_guided_candidates == 12
