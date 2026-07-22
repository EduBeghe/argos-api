Exit code: 0
Wall time: 0.6 seconds
Output:
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import app.optimization.edge_guided as edge_guided_module
from app.jobs.analyze_predictive_edges import build_parser
from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.edge_guided import (
    EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION,
    EdgeGuidedCandidate,
    EdgeGuidedCandidateStatus,
    EdgeGuidedEvaluationResult,
    EdgeGuidedRejectionReason,
    EdgeGuidedSegmentEvaluation,
    apply_edge_guided_action,
    edge_guided_summary,
    evaluate_edge_guided_candidates,
    generate_edge_guided_candidates,
)
from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.historical_data import (
    HistoricalDataset,
    HistoricalDatasetIdentity,
    HistoricalDatasetRange,
)
from app.optimization.predictive_edges import (
    Condition,
    DirectionMapping,
    EvidenceStatus,
    EvidenceThresholds,
    StrategyHypothesis,
    SuggestedAction,
)
from app.optimization.refinement import RefinementSegmentName
from app.optimization.walk_forward import WalkForwardWindow
from app.paper.config import SimulationConfiguration
from app.strategies.baseline_long_only import StrategyAction

START = datetime(2026, 1, 1, tzinfo=UTC)
INSTRUMENT = Instrument(
    Asset.BTC_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)


def hypothesis(
    *,
    identifier: str = "a" * 64,
    action: SuggestedAction = SuggestedAction.LONG_ENTRY,
    lower: Decimal | None = Decimal("-0.02"),
    upper: Decimal | None = Decimal("-0.01"),
    horizon: timedelta = timedelta(minutes=30),
    condition: Condition = Condition.ALL,
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
        condition,
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


def row(
    return_1: str,
    *,
    ema_20: str = "99",
    ema_50: str = "98",
    atr_14: str = "2",
) -> HistoricalFeatureRow:
    return HistoricalFeatureRow(
        datetime(2026, 1, 1, tzinfo=UTC),
        Decimal("100"),
        Decimal(ema_20),
        Decimal(ema_50),
        Decimal("50"),
        Decimal(atr_14),
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


def evaluation_evidence(
    *, windows: int = 2
) -> tuple[HistoricalDataset, HistoricalFeatureMatrix, tuple[WalkForwardWindow, ...]]:
    rows: list[HistoricalFeatureRow] = []
    configured_windows: list[WalkForwardWindow] = []
    for ordinal in range(1, windows + 1):
        start = START + timedelta(hours=4 * (ordinal - 1))
        for index in range(36):
            at = start + timedelta(minutes=5 * index)
            rows.append(replace(row("-0.015"), candle_open_time=at))
        configured_windows.append(
            WalkForwardWindow(
                ordinal,
                start,
                start + timedelta(minutes=60),
                start + timedelta(minutes=60),
                start + timedelta(minutes=120),
                start + timedelta(minutes=120),
                start + timedelta(minutes=180),
                1,
                f"dataset-{ordinal}",
                f"selection-{ordinal}",
            )
        )
    end = START + timedelta(hours=4 * windows)
    dataset = HistoricalDataset(
        HistoricalDatasetIdentity(INSTRUMENT, HistoricalDatasetRange(START, end)),
        len(rows),
        len(rows),
        rows[0].candle_open_time,
        rows[-1].candle_open_time,
        0,
        None,
        None,
        (),
        "fingerprint",
    )
    return (
        dataset,
        HistoricalFeatureMatrix("fingerprint", tuple(rows)),
        tuple(configured_windows),
    )


def segment_result(
    segment: RefinementSegmentName, difference: Decimal
) -> EdgeGuidedSegmentEvaluation:
    return EdgeGuidedSegmentEvaluation(
        segment,
        Decimal("0.01") + difference,
        Decimal("0.02"),
        2,
        Decimal("1.25"),
        difference,
        Decimal("0.005"),
    )


def evaluate_with_differences(
    monkeypatch: pytest.MonkeyPatch,
    *,
    validation: tuple[Decimal, ...],
    test: tuple[Decimal, ...],
) -> EdgeGuidedEvaluationResult:
    dataset, matrix, windows = evaluation_evidence(windows=len(validation))
    validation_index = 0
    test_index = 0

    def evaluate_segment(
        *args: object, **kwargs: object
    ) -> EdgeGuidedSegmentEvaluation:
        nonlocal validation_index, test_index
        segment = args[2]
        assert isinstance(segment, RefinementSegmentName)
        if segment is RefinementSegmentName.VALIDATION:
            value = validation[validation_index]
            validation_index += 1
        elif segment is RefinementSegmentName.TEST:
            value = test[test_index]
            test_index += 1
        else:
            value = Decimal(0)
        return segment_result(segment, value)

    monkeypatch.setattr(edge_guided_module, "_evaluate_segment", evaluate_segment)
    return evaluate_edge_guided_candidates(
        dataset=dataset,
        feature_matrix=matrix,
        hypotheses=(hypothesis(),),
        initial_cash=Decimal("10000"),
        simulation=SimulationConfiguration(),
        windows=windows,
        thresholds=EvidenceThresholds(
            minimum_observations=1,
            minimum_windows=len(windows),
            minimum_stable_direction_ratio=Decimal("0.60"),
        ),
    )


def economic_metric(
    segment: RefinementSegmentName,
    *,
    total_return: str,
    production_return: str,
    buy_and_hold_return: str = "0",
    trade_count: int = 1,
    max_drawdown: str = "0.02",
) -> EdgeGuidedSegmentEvaluation:
    total = Decimal(total_return)
    return EdgeGuidedSegmentEvaluation(
        segment,
        total,
        Decimal(max_drawdown),
        trade_count,
        Decimal("1"),
        total - Decimal(production_return),
        total - Decimal(buy_and_hold_return),
    )


def evaluate_with_metrics(
    monkeypatch: pytest.MonkeyPatch,
    *,
    training: EdgeGuidedSegmentEvaluation,
    validation: EdgeGuidedSegmentEvaluation,
    test: EdgeGuidedSegmentEvaluation,
) -> EdgeGuidedEvaluationResult:
    dataset, matrix, windows = evaluation_evidence(windows=1)

    def evaluate_segment(
        *args: object, **kwargs: object
    ) -> EdgeGuidedSegmentEvaluation:
        del kwargs
        segment = args[2]
        assert isinstance(segment, RefinementSegmentName)
        return {
            RefinementSegmentName.TRAINING: training,
            RefinementSegmentName.VALIDATION: validation,
            RefinementSegmentName.TEST: test,
        }[segment]

    monkeypatch.setattr(edge_guided_module, "_evaluate_segment", evaluate_segment)
    return evaluate_edge_guided_candidates(
        dataset=dataset,
        feature_matrix=matrix,
        hypotheses=(hypothesis(),),
        initial_cash=Decimal("10000"),
        simulation=SimulationConfiguration(),
        windows=windows,
        thresholds=EvidenceThresholds(
            minimum_observations=1,
            minimum_windows=1,
            minimum_stable_direction_ratio=Decimal(1),
        ),
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


def test_zero_hypotheses_is_a_valid_empty_evaluation() -> None:
    dataset, matrix, windows = evaluation_evidence(windows=1)
    result = evaluate_edge_guided_candidates(
        dataset=dataset,
        feature_matrix=matrix,
        hypotheses=(),
        initial_cash=Decimal("10000"),
        simulation=SimulationConfiguration(),
        windows=windows,
        thresholds=EvidenceThresholds(minimum_windows=1),
    )
    assert result.generated_count == 0
    assert result.evaluated_count == 0
    assert result.evaluations == ()


def test_long_entry_gates_only_a_baseline_entry() -> None:
    candidate = candidates((hypothesis(),))[0]
    inside = row("-0.015")
    outside = row("0.01")
    assert (
        apply_edge_guided_action(
            candidate=candidate,
            row=inside,
            position_is_open=False,
            baseline_action=StrategyAction.OPEN_POSITION,
            volatility_threshold=Decimal("0.02"),
        )
        is StrategyAction.OPEN_POSITION
    )
    assert (
        apply_edge_guided_action(
            candidate=candidate,
            row=inside,
            position_is_open=False,
            baseline_action=StrategyAction.NO_ACTION,
            volatility_threshold=Decimal("0.02"),
        )
        is StrategyAction.NO_ACTION
    )
    assert (
        apply_edge_guided_action(
            candidate=candidate,
            row=outside,
            position_is_open=False,
            baseline_action=StrategyAction.NO_ACTION,
            volatility_threshold=Decimal("0.02"),
        )
        is StrategyAction.NO_ACTION
    )
    assert (
        apply_edge_guided_action(
            candidate=candidate,
            row=outside,
            position_is_open=False,
            baseline_action=StrategyAction.OPEN_POSITION,
            volatility_threshold=Decimal("0.02"),
        )
        is StrategyAction.NO_ACTION
    )
    assert (
        apply_edge_guided_action(
            candidate=candidate,
            row=inside,
            position_is_open=True,
            baseline_action=StrategyAction.SELL_ALL,
            volatility_threshold=Decimal("0.02"),
        )
        is StrategyAction.SELL_ALL
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
            baseline_action=StrategyAction.NO_ACTION,
            volatility_threshold=Decimal("0.02"),
        )
        is StrategyAction.NO_ACTION
    )
    assert (
        apply_edge_guided_action(
            candidate=candidate,
            row=row("0.015"),
            position_is_open=True,
            baseline_action=StrategyAction.NO_ACTION,
            volatility_threshold=Decimal("0.02"),
        )
        is StrategyAction.SELL_ALL
    )
    assert StrategyAction.INCREASE_POSITION not in {
        apply_edge_guided_action(
            candidate=candidate,
            row=row("0.015"),
            position_is_open=False,
            baseline_action=StrategyAction.NO_ACTION,
            volatility_threshold=Decimal("0.02"),
        )
    }


def test_condition_is_part_of_identity_and_required_for_application() -> None:
    source = hypothesis(condition=Condition.HIGH_VOLATILITY)
    candidate = candidates((source,))[0]
    unconditional = candidates((hypothesis(),))[0]
    assert candidate.condition is Condition.HIGH_VOLATILITY
    assert candidate.candidate_fingerprint != unconditional.candidate_fingerprint
    assert (
        apply_edge_guided_action(
            candidate=candidate,
            row=row("-0.015", atr_14="1"),
            position_is_open=False,
            baseline_action=StrategyAction.OPEN_POSITION,
            volatility_threshold=Decimal("0.02"),
        )
        is StrategyAction.NO_ACTION
    )
    assert (
        apply_edge_guided_action(
            candidate=candidate,
            row=row("-0.015", atr_14="3"),
            position_is_open=False,
            baseline_action=StrategyAction.OPEN_POSITION,
            volatility_threshold=Decimal("0.02"),
        )
        is StrategyAction.OPEN_POSITION
    )


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


def test_walk_forward_candidate_rejected_in_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = evaluate_with_differences(
        monkeypatch,
        validation=(Decimal("-0.01"), Decimal("-0.02")),
        test=(),
    )
    evaluation = result.evaluations[0]
    assert evaluation.status is EdgeGuidedCandidateStatus.REJECTED_VALIDATION
    assert evaluation.valid_window_count == 0
    assert evaluation.rejected_validation_window_count == 2
    assert evaluation.confirmed_window_count == 0
    assert all(item.test is None for item in evaluation.windows)


def test_walk_forward_candidate_passes_validation_but_fails_isolated_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = evaluate_with_differences(
        monkeypatch,
        validation=(Decimal("0.01"), Decimal("0.02")),
        test=(Decimal("-0.01"), Decimal("-0.02")),
    )
    evaluation = result.evaluations[0]
    assert evaluation.status is EdgeGuidedCandidateStatus.REJECTED_TEST
    assert evaluation.rejected_test_window_count == 2
    assert evaluation.confirmed_window_count == 0


def test_candidate_confirms_across_multiple_walk_forward_windows_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = evaluate_with_differences(
        monkeypatch,
        validation=(Decimal("0.01"), Decimal("0.02"), Decimal("0.03")),
        test=(Decimal("0.01"), Decimal("0.02"), Decimal("0.03")),
    )
    evaluation = result.evaluations[0]
    assert result.windows_available == 3
    assert evaluation.status is EdgeGuidedCandidateStatus.CONFIRMED
    assert evaluation.valid_window_count == 3
    assert evaluation.confirmed_window_count == 3
    assert [item.window_ordinal for item in evaluation.windows] == [1, 2, 3]
    assert all(item.test_confirmed for item in evaluation.windows)
    source = candidates((hypothesis(),))[0]
    assert evaluation.candidate == source
    assert evaluation.candidate.candidate_fingerprint == source.candidate_fingerprint
    assert evaluation.candidate.condition is source.condition
    assert evaluation.candidate.lower_bound == source.lower_bound
    assert evaluation.candidate.upper_bound == source.upper_bound
    assert evaluation.candidate.suggested_action is source.suggested_action
    summary = edge_guided_summary(result)
    summary_evaluations = summary["evaluations"]
    assert isinstance(summary_evaluations, list)
    assert summary_evaluations[0]["candidate"]["condition"] == "all"
    assert summary_evaluations[0]["test"]["positive_net_return"] is True
    assert summary_evaluations[0]["test"]["sufficient_trades"] is True
    assert summary_evaluations[0]["rejection_reasons"] == []


def test_zero_trade_candidate_cannot_pass_validation_or_confirm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zero = economic_metric(
        RefinementSegmentName.TRAINING,
        total_return="0",
        production_return="-0.01",
        trade_count=0,
    )
    result = evaluate_with_metrics(
        monkeypatch,
        training=zero,
        validation=replace(zero, segment=RefinementSegmentName.VALIDATION),
        test=replace(zero, segment=RefinementSegmentName.TEST),
    )
    evaluation = result.evaluations[0]
    assert (
        result.qualification_contract_version
        == EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION
    )
    assert evaluation.status is EdgeGuidedCandidateStatus.REJECTED_VALIDATION
    assert evaluation.valid_window_count == 0
    assert evaluation.confirmed_window_count == 0
    assert EdgeGuidedRejectionReason.INSUFFICIENT_TRADES in evaluation.rejection_reasons
    assert evaluation.validation.positive_net_return is False
    assert evaluation.validation.sufficient_trades is False


@pytest.mark.parametrize("candidate_return", ["-0.01", "0"])
def test_non_positive_candidate_is_rejected_even_when_it_beats_losing_production(
    monkeypatch: pytest.MonkeyPatch, candidate_return: str
) -> None:
    result = evaluate_with_metrics(
        monkeypatch,
        training=economic_metric(
            RefinementSegmentName.TRAINING,
            total_return=candidate_return,
            production_return="-0.02",
        ),
        validation=economic_metric(
            RefinementSegmentName.VALIDATION,
            total_return=candidate_return,
            production_return="-0.02",
        ),
        test=economic_metric(
            RefinementSegmentName.TEST,
            total_return=candidate_return,
            production_return="-0.02",
        ),
    )
    evaluation = result.evaluations[0]
    assert evaluation.status is EdgeGuidedCandidateStatus.REJECTED_VALIDATION
    assert (
        EdgeGuidedRejectionReason.NON_POSITIVE_VALIDATION_RETURN
        in evaluation.rejection_reasons
    )


def test_profitable_candidate_must_strictly_beat_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = evaluate_with_metrics(
        monkeypatch,
        training=economic_metric(
            RefinementSegmentName.TRAINING,
            total_return="0.01",
            production_return="0",
        ),
        validation=economic_metric(
            RefinementSegmentName.VALIDATION,
            total_return="0.01",
            production_return="0.02",
        ),
        test=economic_metric(
            RefinementSegmentName.TEST,
            total_return="0.01",
            production_return="0",
        ),
    )
    evaluation = result.evaluations[0]
    assert evaluation.status is EdgeGuidedCandidateStatus.REJECTED_VALIDATION
    assert (
        EdgeGuidedRejectionReason.DID_NOT_BEAT_PRODUCTION_VALIDATION
        in evaluation.rejection_reasons
    )


def test_economically_viable_candidate_can_confirm_without_beating_buy_and_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = evaluate_with_metrics(
        monkeypatch,
        training=economic_metric(
            RefinementSegmentName.TRAINING,
            total_return="0.02",
            production_return="0.01",
            buy_and_hold_return="0.03",
        ),
        validation=economic_metric(
            RefinementSegmentName.VALIDATION,
            total_return="0.02",
            production_return="0.01",
            buy_and_hold_return="0.03",
        ),
        test=economic_metric(
            RefinementSegmentName.TEST,
            total_return="0.02",
            production_return="0.01",
            buy_and_hold_return="0.03",
        ),
    )
    evaluation = result.evaluations[0]
    assert evaluation.status is EdgeGuidedCandidateStatus.CONFIRMED
    assert evaluation.test.beats_production is True
    assert evaluation.test.beats_buy_and_hold is False
    assert "beat production" in evaluation.explanation
    assert "beat both" not in evaluation.explanation


def test_excessive_drawdown_rejects_otherwise_profitable_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = evaluate_with_metrics(
        monkeypatch,
        training=economic_metric(
            RefinementSegmentName.TRAINING,
            total_return="0.02",
            production_return="0.01",
        ),
        validation=economic_metric(
            RefinementSegmentName.VALIDATION,
            total_return="0.02",
            production_return="0.01",
            max_drawdown="0.11",
        ),
        test=economic_metric(
            RefinementSegmentName.TEST,
            total_return="0.02",
            production_return="0.01",
        ),
    )
    evaluation = result.evaluations[0]
    assert evaluation.status is EdgeGuidedCandidateStatus.REJECTED_VALIDATION
    assert EdgeGuidedRejectionReason.EXCESSIVE_DRAWDOWN in evaluation.rejection_reasons
    assert evaluation.validation.risk_acceptable is False


@pytest.mark.parametrize(
    ("test_metric", "expected_reason"),
    [
        (
            economic_metric(
                RefinementSegmentName.TEST,
                total_return="0.02",
                production_return="0.01",
                trade_count=0,
            ),
            EdgeGuidedRejectionReason.INSUFFICIENT_TRADES,
        ),
        (
            economic_metric(
                RefinementSegmentName.TEST,
                total_return="0",
                production_return="-0.01",
            ),
            EdgeGuidedRejectionReason.NON_POSITIVE_TEST_RETURN,
        ),
        (
            economic_metric(
                RefinementSegmentName.TEST,
                total_return="0.02",
                production_return="0.01",
                max_drawdown="0.11",
            ),
            EdgeGuidedRejectionReason.EXCESSIVE_DRAWDOWN,
        ),
    ],
    ids=("no-test-trades", "non-positive-test-return", "excessive-test-drawdown"),
)
def test_test_only_failure_is_classified_as_rejected_test(
    monkeypatch: pytest.MonkeyPatch,
    test_metric: EdgeGuidedSegmentEvaluation,
    expected_reason: EdgeGuidedRejectionReason,
) -> None:
    result = evaluate_with_metrics(
        monkeypatch,
        training=economic_metric(
            RefinementSegmentName.TRAINING,
            total_return="0.02",
            production_return="0.01",
        ),
        validation=economic_metric(
            RefinementSegmentName.VALIDATION,
            total_return="0.02",
            production_return="0.01",
        ),
        test=test_metric,
    )

    evaluation = result.evaluations[0]
    assert evaluation.status is EdgeGuidedCandidateStatus.REJECTED_TEST
    assert evaluation.valid_window_count == 1
    assert evaluation.confirmed_window_count == 0
    assert expected_reason in evaluation.rejection_reasons


@pytest.mark.parametrize("failing_segment", ["training", "validation"])
def test_zero_trades_before_test_are_classified_as_rejected_validation(
    monkeypatch: pytest.MonkeyPatch, failing_segment: str
) -> None:
    training = economic_metric(
        RefinementSegmentName.TRAINING,
        total_return="0.02",
        production_return="0.01",
        trade_count=0 if failing_segment == "training" else 1,
    )
    validation = economic_metric(
        RefinementSegmentName.VALIDATION,
        total_return="0.02",
        production_return="0.01",
        trade_count=0 if failing_segment == "validation" else 1,
    )
    result = evaluate_with_metrics(
        monkeypatch,
        training=training,
        validation=validation,
        test=economic_metric(
            RefinementSegmentName.TEST,
            total_return="0.02",
            production_return="0.01",
        ),
    )

    evaluation = result.evaluations[0]
    assert evaluation.status is EdgeGuidedCandidateStatus.REJECTED_VALIDATION
    assert evaluation.confirmed_window_count == (
        1 if failing_segment == "training" else 0
    )
    assert EdgeGuidedRejectionReason.INSUFFICIENT_TRADES in evaluation.rejection_reasons


def test_rejection_reasons_are_stable_and_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zero = economic_metric(
        RefinementSegmentName.TRAINING,
        total_return="0",
        production_return="0",
        trade_count=0,
    )
    result = evaluate_with_metrics(
        monkeypatch,
        training=zero,
        validation=replace(zero, segment=RefinementSegmentName.VALIDATION),
        test=replace(zero, segment=RefinementSegmentName.TEST),
    )
    reasons = result.evaluations[0].rejection_reasons
    assert reasons == tuple(dict.fromkeys(reasons))
    assert reasons[0] is EdgeGuidedRejectionReason.INSUFFICIENT_TRADES


def test_evaluation_passes_effective_fees_and_slippage_to_real_simulator() -> None:
    dataset, matrix, windows = evaluation_evidence(windows=1)
    simulation = SimulationConfiguration(
        fee_rate=Decimal("0.0023"), slippage_rate=Decimal("0.0007")
    )
    calls: list[tuple[Decimal, Decimal, bool]] = []

    def simulate(**kwargs: object) -> SimpleNamespace:
        fee = kwargs["fee_rate"]
        slippage = kwargs["slippage_rate"]
        filtered = kwargs.get("decision_action_filter") is not None
        assert isinstance(fee, Decimal)
        assert isinstance(slippage, Decimal)
        calls.append((fee, slippage, filtered))
        return SimpleNamespace(
            total_return=Decimal("0.02") if filtered else Decimal("0.01"),
            max_drawdown=Decimal("0.01"),
            trade_count=1,
            fees_paid=Decimal("1.00"),
        )

    benchmark = SimpleNamespace(total_return=Decimal(0))
    with (
        patch.object(
            edge_guided_module, "simulate_historical_feature_matrix", simulate
        ),
        patch.object(
            edge_guided_module, "simulate_buy_and_hold", return_value=benchmark
        ),
    ):
        result = evaluate_edge_guided_candidates(
            dataset=dataset,
            feature_matrix=matrix,
            hypotheses=(hypothesis(),),
            initial_cash=Decimal("10000"),
            simulation=simulation,
            windows=windows,
            thresholds=EvidenceThresholds(
                minimum_observations=1,
                minimum_windows=1,
                minimum_stable_direction_ratio=Decimal(1),
            ),
        )
    assert result.evaluations[0].status is EdgeGuidedCandidateStatus.CONFIRMED
    assert calls
    assert all(
        fee == simulation.fee_rate and slippage == simulation.slippage_rate
        for fee, slippage, _ in calls
    )
    assert any(filtered for _, _, filtered in calls)
    assert any(not filtered for _, _, filtered in calls)

