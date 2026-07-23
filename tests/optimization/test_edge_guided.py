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
    CandidateFamily,
    EdgeGuidedCandidate,
    EdgeGuidedCandidateStatus,
    EdgeGuidedEvaluationResult,
    EdgeGuidedRejectionReason,
    EdgeGuidedSegmentEvaluation,
    EdgeGuidedStageEvidence,
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
from app.optimization.simulation import SimulationDecisionOverride
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


def test_regime_conditioned_hypothesis_is_gated_by_its_frozen_context() -> None:
    base = hypothesis()
    conditioned = replace(
        base,
        context_predictor_name="relative_volume_20",
        context_predictor_version=3,
        context_regime="high",
        context_lower_bound=Decimal("1.5"),
        context_upper_bound=None,
    )
    candidate = candidates((conditioned,))[0]

    outside = apply_edge_guided_action(
        candidate=candidate,
        row=row("-0.015"),
        position_is_open=False,
        baseline_action=StrategyAction.OPEN_POSITION,
        volatility_threshold=Decimal("0.03"),
    )
    inside = apply_edge_guided_action(
        candidate=candidate,
        row=replace(row("-0.015"), relative_volume_20=Decimal(2)),
        position_is_open=False,
        baseline_action=StrategyAction.OPEN_POSITION,
        volatility_threshold=Decimal("0.03"),
    )

    assert outside is StrategyAction.NO_ACTION
    assert inside is StrategyAction.OPEN_POSITION
    assert candidate.context_predictor_name == "relative_volume_20"
    assert candidate.cadence == conditioned.minimum_evaluation_interval
    unconditioned = candidates((base,))[0]
    assert candidate.candidate_fingerprint != unconditioned.candidate_fingerprint


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


def evaluate_cached_evidence(
    dataset: HistoricalDataset,
    matrix: HistoricalFeatureMatrix,
    windows: tuple[WalkForwardWindow, ...],
    reusable: dict[str, EdgeGuidedStageEvidence] | None = None,
) -> EdgeGuidedEvaluationResult:
    return evaluate_edge_guided_candidates(
        dataset=dataset,
        feature_matrix=matrix,
        hypotheses=(hypothesis(),),
        initial_cash=Decimal("10000"),
        simulation=SimulationConfiguration(),
        maximum_candidates=1,
        windows=windows,
        thresholds=EvidenceThresholds(minimum_observations=1, minimum_windows=1),
        reusable_stage_evidence=reusable,
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
            value = validation[validation_index % len(validation)]
            validation_index += 1
        elif segment is RefinementSegmentName.TEST:
            value = test[test_index % len(test)]
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


def test_confirmed_hypotheses_generate_complete_strategy_alternatives() -> None:
    long = hypothesis(horizon=timedelta(hours=6))
    exit_candidate = hypothesis(
        identifier="b" * 64,
        action=SuggestedAction.EXIT_OR_AVOID,
        lower=Decimal("0.01"),
        upper=Decimal("0.02"),
    )
    generated = candidates((long, exit_candidate))
    assert len(generated) > 2
    assert {item.suggested_action for item in generated} == {
        SuggestedAction.LONG_ENTRY,
        SuggestedAction.EXIT_OR_AVOID,
    }
    entry_candidates = tuple(
        item
        for item in generated
        if item.suggested_action is SuggestedAction.LONG_ENTRY
    )
    assert {item.candidate_family for item in entry_candidates} >= {
        CandidateFamily.EDGE_ENTRY_TIMED_EXIT,
        CandidateFamily.EDGE_ENTRY_SIGNAL_EXIT,
        CandidateFamily.EDGE_ENTRY_BASELINE_EXIT,
    }
    assert len({item.cadence for item in entry_candidates}) > 1
    assert {item.target_allocation_percentage for item in entry_candidates} == {
        Decimal("10"),
        Decimal("15"),
        Decimal("20"),
    }


def test_generation_is_broad_before_final_evaluation_limit() -> None:
    generated = candidates((hypothesis(),), maximum=1)
    assert len(generated) > 1
    assert len(generated) <= edge_guided_module.MAX_GENERATED_STRATEGIES


def test_generation_budget_is_stratified_across_available_families() -> None:
    entries = tuple(
        hypothesis(identifier=f"{index:064x}", horizon=timedelta(hours=6))
        for index in range(1, 13)
    )
    exits = tuple(
        hypothesis(
            identifier=f"{index + 100:064x}",
            action=SuggestedAction.EXIT_OR_AVOID,
            lower=Decimal("0.01"),
            upper=Decimal("0.02"),
            horizon=timedelta(hours=6),
        )
        for index in range(6)
    )

    first = candidates(tuple(reversed((*entries, *exits))))
    second = candidates((*entries, *exits))

    assert len(first) == edge_guided_module.MAX_GENERATED_STRATEGIES
    assert tuple(item.candidate_fingerprint for item in first) == tuple(
        item.candidate_fingerprint for item in second
    )
    assert {item.candidate_family for item in first} == set(CandidateFamily)
    assert len({item.source_hypothesis_id for item in first}) > 1
    assert len({item.cadence for item in first}) > 1
    assert {item.target_allocation_percentage for item in first} == {
        Decimal("10"),
        Decimal("15"),
        Decimal("20"),
    }


def test_diagnostic_or_unconfirmed_evidence_produces_no_candidate() -> None:
    diagnostic = hypothesis(action=SuggestedAction.DIAGNOSTIC_ONLY)
    assert candidates((diagnostic,)) == ()
    assert candidates(()) == ()


def test_complete_edge_strategy_opens_without_production_and_uses_target() -> None:
    candidate = candidates((hypothesis(),))[0]
    policy = edge_guided_module._candidate_action_filter(candidate, Decimal("0.10"))
    evidence = replace(row("-0.015"), candle_open_time=START + timedelta(minutes=25))

    action = policy(evidence, False, StrategyAction.NO_ACTION)

    assert isinstance(action, SimulationDecisionOverride)
    assert action.action is StrategyAction.OPEN_POSITION
    assert action.target_allocation_percentage == candidate.target_allocation_percentage


def test_complete_edge_strategy_does_not_buy_while_open_and_timed_exit_closes() -> None:
    candidate = candidates((hypothesis(),))[0]
    policy = edge_guided_module._candidate_action_filter(candidate, Decimal("0.10"))
    entry = replace(row("-0.015"), candle_open_time=START + timedelta(minutes=25))
    before_holding = replace(
        row("-0.015"), candle_open_time=START + timedelta(minutes=25)
    )
    exit_row = replace(row("-0.015"), candle_open_time=START + timedelta(minutes=55))

    assert isinstance(
        policy(entry, False, StrategyAction.NO_ACTION), SimulationDecisionOverride
    )
    assert (
        policy(before_holding, True, StrategyAction.OPEN_POSITION)
        is StrategyAction.NO_ACTION
    )
    exit_action = policy(exit_row, True, StrategyAction.NO_ACTION)
    assert isinstance(exit_action, SimulationDecisionOverride)
    assert exit_action.action is StrategyAction.SELL_ALL


def test_zero_hypotheses_is_a_valid_empty_evaluation() -> None:
    dataset, matrix, windows = evaluation_evidence(windows=1)
    result = evaluate_edge_guided_candidates(
        dataset=dataset,
        feature_maßz¶‰žËkºwµç@€€Ñ•ÍÐõ•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QMP°(€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸õ…¹‘¥‘…Ñ•}É•ÑÕÉ¸°(€€€€€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆ´À¸ÀÈˆ°(€€€€€€€€¤°(€€€€¤(€€€…ÍÍ•ÉÐÉ•ÍÕ±Ð¹•Ù…±Õ…Ñ¥½¹Ì€ôô€ ¤(€€€…ÍÍ•ÉÐÉ•ÍÕ±Ð¹™Õ¹¹•±l‰ÑÉ…¥¹¥¹}É•Ñ…¥¹•‘}½Õ¹Ð‰t€ôô€À(()‘•˜Ñ•ÍÑ}ÁÉ½™¥Ñ…‰±•}…¹‘¥‘…Ñ•}µÕÍÑ}ÍÑÉ¥Ñ±å}‰•…Ñ}ÁÉ½‘ÕÑ¥½¸ (€€€µ½¹­•åÁ…Ñ èÁåÑ•ÍÐ¹5½¹­•åA…Ñ °(¤€´ø9½¹”è(€€€É•ÍÕ±Ð€ô•Ù…±Õ…Ñ•}Ý¥Ñ¡}µ•ÑÉ¥Ì (€€€€€€€µ½¹­•åÁ…Ñ °(€€€€€€€ÑÉ…¥¹¥¹œõ•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QI%9%9°(€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀ¸ÀÄˆ°(€€€€€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀˆ°(€€€€€€€€¤°(€€€€€€€Ù…±¥‘…Ñ¥½¸õ•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹Y1%Q%=8°(€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀ¸ÀÄˆ°(€€€€€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀ¸ÀÈˆ°(€€€€€€€€¤°(€€€€€€€Ñ•ÍÐõ•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QMP°(€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀ¸ÀÄˆ°(€€€€€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀˆ°(€€€€€€€€¤°(€€€€¤(€€€•Ù…±Õ…Ñ¥½¸€ôÉ•ÍÕ±Ð¹•Ù…±Õ…Ñ¥½¹ÍlÁt(€€€…ÍÍ•ÉÐ•Ù…±Õ…Ñ¥½¸¹ÍÑ…ÑÕÌ¥Ì‘•Õ¥‘•‘…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹I)Q}Y1%Q%=8(€€€…ÍÍ•ÉÐ€ (€€€€€€€‘•Õ¥‘•‘I•©•Ñ¥½¹I•…Í½¸¹%}9=Q}	Q}AI=UQ%=9}Y1%Q%=8(€€€€€€€¥¸•Ù…±Õ…Ñ¥½¸¹É•©•Ñ¥½¹}É•…Í½¹Ì(€€€€¤(()‘•˜Ñ•ÍÑ}•½¹½µ¥…±±å}Ù¥…‰±•}…¹‘¥‘…Ñ•}…¹}½¹™¥Éµ}Ý¥Ñ¡½ÕÑ}‰•…Ñ¥¹}‰Õå}…¹‘}¡½± (€€€µ½¹­•åÁ…Ñ èÁåÑ•ÍÐ¹5½¹­•åA…Ñ °(¤€´ø9½¹”è(€€€É•ÍÕ±Ð€ô•Ù…±Õ…Ñ•}Ý¥Ñ¡}µ•ÑÉ¥Ì (€€€€€€€µ½¹­•åÁ…Ñ °(€€€€€€€ÑÉ…¥¹¥¹œõ•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QI%9%9°(€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀ¸ÀÈˆ°(€€€€€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀ¸ÀÄˆ°(€€€€€€€€€€€‰Õå}…¹‘}¡½±‘}É•ÑÕÉ¸ôˆÀ¸ÀÌˆ°(€€€€€€€€¤°(€€€€€€€Ù…±¥‘…Ñ¥½¸õ•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹Y1%Q%=8°(€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀ¸ÀÈˆ°(€€€€€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀ¸ÀÄˆ°(€€€€€€€€€€€‰Õå}…¹‘}¡½±‘}É•ÑÕÉ¸ôˆÀ¸ÀÌˆ°(€€€€€€€€¤°(€€€€€€€Ñ•ÍÐõ•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QMP°(€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀ¸ÀÈˆ°(€€€€€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀ¸ÀÄˆ°(€€€€€€€€€€€‰Õå}…¹‘}¡½±‘}É•ÑÕÉ¸ôˆÀ¸ÀÌˆ°(€€€€€€€€¤°(€€€€¤(€€€•Ù…±Õ…Ñ¥½¸€ôÉ•ÍÕ±Ð¹•Ù…±Õ…Ñ¥½¹ÍlÁt(€€€…ÍÍ•ÉÐ•Ù…±Õ…Ñ¥½¸¹ÍÑ…ÑÕÌ¥Ì‘•Õ¥‘•‘…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹=9%I5(€€€…ÍÍ•ÉÐ•Ù…±Õ…Ñ¥½¸¹Ñ•ÍÐ¥Ì¹½Ð9½¹”(€€€…ÍÍ•ÉÐ•Ù…±Õ…Ñ¥½¸¹Ñ•ÍÐ¹‰•…ÑÍ}ÁÉ½‘ÕÑ¥½¸¥ÌQÉÕ”(€€€…ÍÍ•ÉÐ•Ù…±Õ…Ñ¥½¸¹Ñ•ÍÐ¹‰•…ÑÍ}‰Õå}…¹‘}¡½±¥Ì…±Í”(€€€…ÍÍ•ÉÐ€‰‰•…ÐÁÉ½‘ÕÑ¥½¸ˆ¥¸•Ù…±Õ…Ñ¥½¸¹•áÁ±…¹…Ñ¥½¸(€€€…ÍÍ•ÉÐ€‰‰•…Ð‰½Ñ ˆ¹½Ð¥¸•Ù…±Õ…Ñ¥½¸¹•áÁ±…¹…Ñ¥½¸(()‘•˜Ñ•ÍÑ}½¹±å}Ù…±¥‘…Ñ¥½¹}…¹‘¥‘…Ñ•}‰•…Ñ¥¹}ÁÉ½‘ÕÑ¥½¹}½¹ÍÕµ•Í}Ñ•ÍÑ}‰Õ‘•Ð (€€€µ½¹­•åÁ…Ñ èÁåÑ•ÍÐ¹5½¹­•åA…Ñ °(¤€´ø9½¹”è(€€€‘…Ñ…Í•Ð°µ…ÑÉ¥à°Ý¥¹‘½ÝÌ€ô•Ù…±Õ…Ñ¥½¹}•Ù¥‘•¹”¡Ý¥¹‘½ÝÌôÄ¤(€€€Ý•…­•È€ô¡åÁ½Ñ¡•Í¥Ì¡¥‘•¹Ñ¥™¥•ÈôˆÄˆ€¨€ØÐ¤(€€€ÍÑÉ½¹•È€ô¡åÁ½Ñ¡•Í¥Ì¡¥‘•¹Ñ¥™¥•ÈôˆÈˆ€¨€ØÐ¤(€€€Ñ•ÍÑ}Í½ÕÉ•Ìè±¥ÍÑmÍÑÉt€ômt((€€€‘•˜•Ù…±Õ…Ñ•}Í•µ•¹Ð (€€€€€€€€©…ÉÌè½‰©•Ð°€¨©­Ý…ÉÌè½‰©•Ð(€€€€¤€´ø‘•Õ¥‘•‘M•µ•¹ÑÙ…±Õ…Ñ¥½¸è(€€€€€€€‘•°­Ý…ÉÌ(€€€€€€€Í•µ•¹Ð€ô…ÉÍlÉt(€€€€€€€…¹‘¥‘…Ñ”€ô…ÉÍlÍt(€€€€€€€…ÍÍ•ÉÐ¥Í¥¹ÍÑ…¹”¡Í•µ•¹Ð°I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¤(€€€€€€€…ÍÍ•ÉÐ¥Í¥¹ÍÑ…¹”¡…¹‘¥‘…Ñ”°‘•Õ¥‘•‘…¹‘¥‘…Ñ”¤(€€€€€€€¥˜Í•µ•¹Ð¥ÌI•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QMPè(€€€€€€€€€€€Ñ•ÍÑ}Í½ÕÉ•Ì¹…ÁÁ•¹¡…¹‘¥‘…Ñ”¹Í½ÕÉ•}¡åÁ½Ñ¡•Í¥Í}¥¤(€€€€€€€‘¥™™•É•¹”€ô€ (€€€€€€€€€€€•¥µ…° ˆ´À¸ÀÀÄˆ¤(€€€€€€€€€€€¥˜Í•µ•¹Ð¥ÌI•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹Y1%Q%=8(€€€€€€€€€€€…¹…¹‘¥‘…Ñ”¹Í½ÕÉ•}¡åÁ½Ñ¡•Í¥Í}¥€ôôÝ•…­•È¹¡åÁ½Ñ¡•Í¥Í}¥(€€€€€€€€€€€•±Í”•¥µ…° ˆÀ¸ÀÄˆ¤(€€€€€€€€¤(€€€€€€€É•ÑÕÉ¸Í•µ•¹Ñ}É•ÍÕ±Ð¡Í•µ•¹Ð°‘¥™™•É•¹”¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡•‘•}Õ¥‘•‘}µ½‘Õ±”°€‰}•Ù…±Õ…Ñ•}Í•µ•¹Ðˆ°•Ù…±Õ…Ñ•}Í•µ•¹Ð¤(€€€É•ÍÕ±Ð€ô•Ù…±Õ…Ñ•}•‘•}Õ¥‘•‘}…¹‘¥‘…Ñ•Ì (€€€€€€€‘…Ñ…Í•Ðõ‘…Ñ…Í•Ð°(€€€€€€€™•…ÑÕÉ•}µ…ÑÉ¥àõµ…ÑÉ¥à°(€€€€€€€¡åÁ½Ñ¡•Í•Ìô¡Ý•…­•È°ÍÑÉ½¹•È¤°(€€€€€€€¥¹¥Ñ¥…±}…Í õ•¥µ…° ˆÄÀÀÀÀˆ¤°(€€€€€€€Í¥µÕ±…Ñ¥½¸õM¥µÕ±…Ñ¥½¹½¹™¥ÕÉ…Ñ¥½¸ ¤°(€€€€€€€µ…á¥µÕµ}…¹‘¥‘…Ñ•ÌôÄ°(€€€€€€€Ý¥¹‘½ÝÌõÝ¥¹‘½ÝÌ°(€€€€€€€Ñ¡É•Í¡½±‘ÌõÙ¥‘•¹•Q¡É•Í¡½±‘Ì¡µ¥¹¥µÕµ}½‰Í•ÉÙ…Ñ¥½¹ÌôÄ°µ¥¹¥µÕµ}Ý¥¹‘½ÝÌôÄ¤°(€€€€¤((€€€…ÍÍ•ÉÐÑ•ÍÑ}Í½ÕÉ•Ì(€€€…ÍÍ•ÉÐÍ•Ð¡Ñ•ÍÑ}Í½ÕÉ•Ì¤€ôôíÍÑÉ½¹•È¹¡åÁ½Ñ¡•Í¥Í}¥‘ô(€€€…ÍÍ•ÉÐÉ•ÍÕ±Ð¹™Õ¹¹•±l‰Ñ•ÍÑ}•Ù…±Õ…Ñ•‘}½Õ¹Ð‰t€ôô€Ä(()‘•˜Ñ•ÍÑ}ÍÑ…•}…¡•}É•ÕÍ•Í}ÑÉ…¥¹¥¹}Ù…±¥‘…Ñ¥½¹}…¹‘}Ñ•ÍÑ}Ý½É¬ (€€€µ½¹­•åÁ…Ñ èÁåÑ•ÍÐ¹5½¹­•åA…Ñ °(¤€´ø9½¹”è(€€€‘…Ñ…Í•Ð°µ…ÑÉ¥à°Ý¥¹‘½ÝÌ€ô•Ù…±Õ…Ñ¥½¹}•Ù¥‘•¹”¡Ý¥¹‘½ÝÌôÄ¤(€€€…±±Ìè‘¥ÑmI•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”°¥¹Ñt€ôì(€€€€€€€ÍÑ…”è€À™½ÈÍÑ…”¥¸•‘•}Õ¥‘•‘}µ½‘Õ±”¹}U%}MQL(€€€ô((€€€‘•˜•Ù…±Õ…Ñ•}Í•µ•¹Ð (€€€€€€€€©…ÉÌè½‰©•Ð°€¨©­Ý…ÉÌè½‰©•Ð(€€€€¤€´ø‘•Õ¥‘•‘M•µ•¹ÑÙ…±Õ…Ñ¥½¸è(€€€€€€€‘•°­Ý…ÉÌ(€€€€€€€Í•µ•¹Ð€ô…ÉÍlÉt(€€€€€€€…ÍÍ•ÉÐ¥Í¥¹ÍÑ…¹”¡Í•µ•¹Ð°I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¤(€€€€€€€…±±ÍmÍ•µ•¹Ñt€¬ô€Ä(€€€€€€€É•ÑÕÉ¸Í•µ•¹Ñ}É•ÍÕ±Ð¡Í•µ•¹Ð°•¥µ…° ˆÀ¸ÀÄˆ¤¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡•‘•}Õ¥‘•‘}µ½‘Õ±”°€‰}•Ù…±Õ…Ñ•}Í•µ•¹Ðˆ°•Ù…±Õ…Ñ•}Í•µ•¹Ð¤(€€€™¥ÉÍÐ€ô•Ù…±Õ…Ñ•}…¡•‘}•Ù¥‘•¹”¡‘…Ñ…Í•Ð°µ…ÑÉ¥à°Ý¥¹‘½ÝÌ¤(€€€™¥ÉÍÑ}…±±Ì€ô‘¥Ð¡…±±Ì¤(€€€…ÍÍ•ÉÐ…±°¡Ù…±Õ”€ø€À™½ÈÙ…±Õ”¥¸™¥ÉÍÑ}…±±Ì¹Ù…±Õ•Ì ¤¤((€€€…±±Ì¹ÕÁ‘…Ñ”¡íÍÑ…”è€À™½ÈÍÑ…”¥¸•‘•}Õ¥‘•‘}µ½‘Õ±”¹}U%}MQMô¤(€€€Í•½¹€ô•Ù…±Õ…Ñ•}…¡•‘}•Ù¥‘•¹” (€€€€€€€‘…Ñ…Í•Ð°(€€€€€€€µ…ÑÉ¥à°(€€€€€€€Ý¥¹‘½ÝÌ°(€€€€€€€í¥Ñ•´¹…¡•}­•äè¥Ñ•´™½È¥Ñ•´¥¸™¥ÉÍÐ¹ÍÑ…•}•Ù¥‘•¹•ô°(€€€€¤((€€€…ÍÍ•ÉÐ…±±Ì€ôôíÍÑ…”è€À™½ÈÍÑ…”¥¸•‘•}Õ¥‘•‘}µ½‘Õ±”¹}U%}MQMô(€€€…ÍÍ•ÉÐÍ•½¹¹™Õ¹¹•±l‰Ý¥¹‘½ÝÍ}½µÁÕÑ•‰t€ôô€À(€€€…ÍÍ•ÉÐÍ•½¹¹™Õ¹¹•±l‰Ý¥¹‘½ÝÍ}É•ÕÍ•‰t€ôô±•¸¡Í•½¹¹ÍÑ…•}•Ù¥‘•¹”¤(€€€…ÍÍ•ÉÐÍ•½¹¹™Õ¹¹•±l‰Ñ•ÍÑ}•Ù…±Õ…Ñ•‘}½Õ¹Ð‰t€ôô€Ä((€€€…±±Ì¹ÕÁ‘…Ñ”¡íÍÑ…”è€À™½ÈÍÑ…”¥¸•‘•}Õ¥‘•‘}µ½‘Õ±”¹}U%}MQMô¤(€€€™½É•€ô•Ù…±Õ…Ñ•}…¡•‘}•Ù¥‘•¹”¡‘…Ñ…Í•Ð°µ…ÑÉ¥à°Ý¥¹‘½ÝÌ¤(€€€…ÍÍ•ÉÐ™½É•¹™Õ¹¹•±l‰Ý¥¹‘½ÝÍ}É•ÕÍ•‰t€ôô€À(€€€…ÍÍ•ÉÐ™½É•¹™Õ¹¹•±l‰Ý¥¹‘½ÝÍ}½µÁÕÑ•‰t€ø€À(€€€…ÍÍ•ÉÐ…±±Ì€ôô™¥ÉÍÑ}…±±Ì(()ÁåÑ•ÍÐ¹µ…É¬¹Á…É…µ•ÑÉ¥é” (€€€€ ‰™…¥±•‘}ÍÑ…”ˆ°€‰•áÁ•Ñ•‘}ÍÑ…•Ìˆ¤°(€€€l(€€€€€€€€¡I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QI%9%9°íI•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QI%9%9ô¤°(€€€€€€€€ (€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹Y1%Q%=8°(€€€€€€€€€€€íI•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QI%9%9°I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹Y1%Q%=9ô°(€€€€€€€€¤°(€€€t°(¤)‘•˜Ñ•ÍÑ}É•©•Ñ•‘}ÍÑ…•}•Ù¥‘•¹•}¥Í}É•ÕÍ…‰±” (€€€µ½¹­•åÁ…Ñ èÁåÑ•ÍÐ¹5½¹­•åA…Ñ °(€€€™…¥±•‘}ÍÑ…”èI•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”°(€€€•áÁ•Ñ•‘}ÍÑ…•ÌèÍ•ÑmI•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ•t°(¤€´ø9½¹”è(€€€‘…Ñ…Í•Ð°µ…ÑÉ¥à°Ý¥¹‘½ÝÌ€ô•Ù…±Õ…Ñ¥½¹}•Ù¥‘•¹”¡Ý¥¹‘½ÝÌôÄ¤(€€€…±±Ìè±¥ÍÑmI•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ•t€ômt((€€€‘•˜•Ù…±Õ…Ñ•}Í•µ•¹Ð (€€€€€€€€©…ÉÌè½‰©•Ð°€¨©­Ý…ÉÌè½‰©•Ð(€€€€¤€´ø‘•Õ¥‘•‘M•µ•¹ÑÙ…±Õ…Ñ¥½¸è(€€€€€€€‘•°­Ý…ÉÌ(€€€€€€€ÍÑ…”€ô…ÉÍlÉt(€€€€€€€…ÍÍ•ÉÐ¥Í¥¹ÍÑ…¹”¡ÍÑ…”°I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¤(€€€€€€€…±±Ì¹…ÁÁ•¹¡ÍÑ…”¤(€€€€€€€‘¥™™•É•¹”€ô•¥µ…° ˆ´À¸ÀÈˆ¤¥˜ÍÑ…”¥Ì™…¥±•‘}ÍÑ…”•±Í”•¥µ…° ˆÀ¸ÀÄˆ¤(€€€€€€€É•ÑÕÉ¸Í•µ•¹Ñ}É•ÍÕ±Ð¡ÍÑ…”°‘¥™™•É•¹”¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡•‘•}Õ¥‘•‘}µ½‘Õ±”°€‰}•Ù…±Õ…Ñ•}Í•µ•¹Ðˆ°•Ù…±Õ…Ñ•}Í•µ•¹Ð¤(€€€™¥ÉÍÐ€ô•Ù…±Õ…Ñ•}…¡•‘}•Ù¥‘•¹”¡‘…Ñ…Í•Ð°µ…ÑÉ¥à°Ý¥¹‘½ÝÌ¤(€€€…ÍÍ•ÉÐí¥Ñ•´¹ÍÑ…”™½È¥Ñ•´¥¸™¥ÉÍÐ¹ÍÑ…•}•Ù¥‘•¹•ô€ôô•áÁ•Ñ•‘}ÍÑ…•Ì((€€€…±±Ì¹±•…È ¤(€€€Í•½¹€ô•Ù…±Õ…Ñ•}…¡•‘}•Ù¥‘•¹” (€€€€€€€‘…Ñ…Í•Ð°(€€€€€€€µ…ÑÉ¥à°(€€€€€€€Ý¥¹‘½ÝÌ°(€€€€€€€í¥Ñ•´¹…¡•}­•äè¥Ñ•´™½È¥Ñ•´¥¸™¥ÉÍÐ¹ÍÑ…•}•Ù¥‘•¹•ô°(€€€€¤(€€€…ÍÍ•ÉÐ…±±Ì€ôômt(€€€…ÍÍ•ÉÐÍ•½¹¹™Õ¹¹•±l‰Ý¥¹‘½ÝÍ}½µÁÕÑ•‰t€ôô€À(€€€…ÍÍ•ÉÐÍ•½¹¹™Õ¹¹•±l‰Ý¥¹‘½ÝÍ}É•ÕÍ•‰t€ôô±•¸¡Í•½¹¹ÍÑ…•}•Ù¥‘•¹”¤(()‘•˜Ñ•ÍÑ}¡…¹•‘}Ù…±¥‘…Ñ¥½¹}½¹Ñ•¹Ñ}¥¹Ù…±¥‘…Ñ•Í}½¹±å}Ù…±¥‘…Ñ¥½¹}ÍÑ…” (€€€µ½¹­•åÁ…Ñ èÁåÑ•ÍÐ¹5½¹­•åA…Ñ °(¤€´ø9½¹”è(€€€‘…Ñ…Í•Ð°µ…ÑÉ¥à°Ý¥¹‘½ÝÌ€ô•Ù…±Õ…Ñ¥½¹}•Ù¥‘•¹”¡Ý¥¹‘½ÝÌôÄ¤(€€€…±±Ìè±¥ÍÑmI•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ•t€ômt((€€€‘•˜•Ù…±Õ…Ñ•}Í•µ•¹Ð (€€€€€€€€©…ÉÌè½‰©•Ð°€¨©­Ý…ÉÌè½‰©•Ð(€€€€¤€´ø‘•Õ¥‘•‘M•µ•¹ÑÙ…±Õ…Ñ¥½¸è(€€€€€€€‘•°­Ý…ÉÌ(€€€€€€€ÍÑ…”€ô…ÉÍlÉt(€€€€€€€…ÍÍ•ÉÐ¥Í¥¹ÍÑ…¹”¡ÍÑ…”°I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¤(€€€€€€€…±±Ì¹…ÁÁ•¹¡ÍÑ…”¤(€€€€€€€É•ÑÕÉ¸Í•µ•¹Ñ}É•ÍÕ±Ð¡ÍÑ…”°•¥µ…° ˆÀ¸ÀÄˆ¤¤((€€€µ½¹­•åÁ…Ñ ¹Í•Ñ…ÑÑÈ¡•‘•}Õ¥‘•‘}µ½‘Õ±”°€‰}•Ù…±Õ…Ñ•}Í•µ•¹Ðˆ°•Ù…±Õ…Ñ•}Í•µ•¹Ð¤(€€€™¥ÉÍÐ€ô•Ù…±Õ…Ñ•}…¡•‘}•Ù¥‘•¹”¡‘…Ñ…Í•Ð°µ…ÑÉ¥à°Ý¥¹‘½ÝÌ¤(€€€¡…¹•‘}É½ÝÌ€ô±¥ÍÐ¡µ…ÑÉ¥à¹É½ÝÌ¤(€€€¡…¹•‘}É½ÝÍlÄÅt€ôÉ•Á±…”¡¡…¹•‘}É½ÝÍlÄÅt°É•ÑÕÉ¹|Äõ•¥µ…° ˆÀ¸ÄÈÌˆ¤¤(€€€…±±Ì¹±•…È ¤((€€€Í•½¹€ô•Ù…±Õ…Ñ•}…¡•‘}•Ù¥‘•¹” (€€€€€€€‘…Ñ…Í•Ð°(€€€€€€€É•Á±…”¡µ…ÑÉ¥à°É½ÝÌõÑÕÁ±”¡¡…¹•‘}É½ÝÌ¤¤°(€€€€€€€Ý¥¹‘½ÝÌ°(€€€€€€€í¥Ñ•´¹…¡•}­•äè¥Ñ•´™½È¥Ñ•´¥¸™¥ÉÍÐ¹ÍÑ…•}•Ù¥‘•¹•ô°(€€€€¤((€€€…ÍÍ•ÉÐ…±±Ì(€€€…ÍÍ•ÉÐÍ•Ð¡…±±Ì¤€ôôíI•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹Y1%Q%=9ô(€€€…ÍÍ•ÉÐÍ•½¹¹™Õ¹¹•±l‰ÑÉ…¥¹¥¹}Ý¥¹‘½ÝÍ}½µÁÕÑ•‰t€ôô€À(€€€…ÍÍ•ÉÐÍ•½¹¹™Õ¹¹•±l‰Ñ•ÍÑ}Ý¥¹‘½ÝÍ}½µÁÕÑ•‰t€ôô€À(€€€…ÍÍ•ÉÐÍ•½¹¹™Õ¹¹•±l‰Ù…±¥‘…Ñ¥½¹}Ý¥¹‘½ÝÍ}½µÁÕÑ•‰t€ø€À(()‘•˜Ñ•ÍÑ}•á•ÍÍ¥Ù•}‘É…Ý‘½Ý¹}É•©•ÑÍ}½Ñ¡•ÉÝ¥Í•}ÁÉ½™¥Ñ…‰±•}…¹‘¥‘…Ñ” (€€€µ½¹­•åÁ…Ñ èÁåÑ•ÍÐ¹5½¹­•åA…Ñ °(¤€´ø9½¹”è(€€€É•ÍÕ±Ð€ô•Ù…±Õ…Ñ•}Ý¥Ñ¡}µ•ÑÉ¥Ì (€€€€€€€µ½¹­•åÁ…Ñ °(€€€€€€€ÑÉ…¥¹¥¹œõ•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QI%9%9°(€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀ¸ÀÈˆ°(€€€€€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀ¸ÀÄˆ°(€€€€€€€€¤°(€€€€€€€Ù…±¥‘…Ñ¥½¸õ•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹Y1%Q%=8°(€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀ¸ÀÈˆ°(€€€€€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀ¸ÀÄˆ°(€€€€€€€€€€€µ…á}‘É…Ý‘½Ý¸ôˆÀ¸ÄÄˆ°(€€€€€€€€¤°(€€€€€€€Ñ•ÍÐõ•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QMP°(€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀ¸ÀÈˆ°(€€€€€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀ¸ÀÄˆ°(€€€€€€€€¤°(€€€€¤(€€€•Ù…±Õ…Ñ¥½¸€ôÉ•ÍÕ±Ð¹•Ù…±Õ…Ñ¥½¹ÍlÁt(€€€…ÍÍ•ÉÐ•Ù…±Õ…Ñ¥½¸¹ÍÑ…ÑÕÌ¥Ì‘•Õ¥‘•‘…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹I)Q}Y1%Q%=8(€€€…ÍÍ•ÉÐ‘•Õ¥‘•‘I•©•Ñ¥½¹I•…Í½¸¹aMM%Y}I]=]8¥¸•Ù…±Õ…Ñ¥½¸¹É•©•Ñ¥½¹}É•…Í½¹Ì(€€€…ÍÍ•ÉÐ•Ù…±Õ…Ñ¥½¸¹Ù…±¥‘…Ñ¥½¸¹É¥Í­}…•ÁÑ…‰±”¥Ì…±Í”(()ÁåÑ•ÍÐ¹µ…É¬¹Á…É…µ•ÑÉ¥é” (€€€€ ‰Ñ•ÍÑ}µ•ÑÉ¥Œˆ°€‰•áÁ•Ñ•‘}É•…Í½¸ˆ¤°(€€€l(€€€€€€€€ (€€€€€€€€€€€•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QMP°(€€€€€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀ¸ÀÈˆ°(€€€€€€€€€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀ¸ÀÄˆ°(€€€€€€€€€€€€€€€ÑÉ…‘•}½Õ¹ÐôÀ°(€€€€€€€€€€€€¤°(€€€€€€€€€€€‘•Õ¥‘•‘I•©•Ñ¥½¹I•…Í½¸¹%9MU%%9Q}QIL°(€€€€€€€€¤°(€€€€€€€€ (€€€€€€€€€€€•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QMP°(€€€€€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀˆ°(€€€€€€€€€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆ´À¸ÀÄˆ°(€€€€€€€€€€€€¤°(€€€€€€€€€€€‘•Õ¥‘•‘I•©•Ñ¥½¹I•…Í½¸¹9=9}A=M%Q%Y}QMQ}IQUI8°(€€€€€€€€¤°(€€€€€€€€ (€€€€€€€€€€€•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QMP°(€€€€€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀ¸ÀÈˆ°(€€€€€€€€€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀ¸ÀÄˆ°(€€€€€€€€€€€€€€€µ…á}‘É…Ý‘½Ý¸ôˆÀ¸ÄÄˆ°(€€€€€€€€€€€€¤°(€€€€€€€€€€€‘•Õ¥‘•‘I•©•Ñ¥½¹I•…Í½¸¹aMM%Y}I]=]8°(€€€€€€€€¤°(€€€t°(€€€¥‘Ìô ‰¹¼µÑ•ÍÐµÑÉ…‘•Ìˆ°€‰¹½¸µÁ½Í¥Ñ¥Ù”µÑ•ÍÐµÉ•ÑÕÉ¸ˆ°€‰•á•ÍÍ¥Ù”µÑ•ÍÐµ‘É…Ý‘½Ý¸ˆ¤°(¤)‘•˜Ñ•ÍÑ}Ñ•ÍÑ}½¹±å}™…¥±ÕÉ•}¥Í}±…ÍÍ¥™¥•‘}…Í}É•©•Ñ•‘}Ñ•ÍÐ (€€€µ½¹­•åÁ…Ñ èÁåÑ•ÍÐ¹5½¹­•åA…Ñ °(€€€Ñ•ÍÑ}µ•ÑÉ¥Œè‘•Õ¥‘•‘M•µ•¹ÑÙ…±Õ…Ñ¥½¸°(€€€•áÁ•Ñ•‘}É•…Í½¸è‘•Õ¥‘•‘I•©•Ñ¥½¹I•…Í½¸°(¤€´ø9½¹”è(€€€É•ÍÕ±Ð€ô•Ù…±Õ…Ñ•}Ý¥Ñ¡}µ•ÑÉ¥Ì (€€€€€€€µ½¹­•åÁ…Ñ °(€€€€€€€ÑÉ…¥¹¥¹œõ•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QI%9%9°(€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀ¸ÀÈˆ°(€€€€€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀ¸ÀÄˆ°(€€€€€€€€¤°(€€€€€€€Ù…±¥‘…Ñ¥½¸õ•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹Y1%Q%=8°(€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀ¸ÀÈˆ°(€€€€€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀ¸ÀÄˆ°(€€€€€€€€¤°(€€€€€€€Ñ•ÍÐõÑ•ÍÑ}µ•ÑÉ¥Œ°(€€€€¤((€€€•Ù…±Õ…Ñ¥½¸€ôÉ•ÍÕ±Ð¹•Ù…±Õ…Ñ¥½¹ÍlÁt(€€€…ÍÍ•ÉÐ•Ù…±Õ…Ñ¥½¸¹ÍÑ…ÑÕÌ¥Ì‘•Õ¥‘•‘…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹I)Q}QMP(€€€…ÍÍ•ÉÐ•Ù…±Õ…Ñ¥½¸¹Ù…±¥‘}Ý¥¹‘½Ý}½Õ¹Ð€ôô€Ä(€€€…ÍÍ•ÉÐ•Ù…±Õ…Ñ¥½¸¹½¹™¥Éµ•‘}Ý¥¹‘½Ý}½Õ¹Ð€ôô€À(€€€…ÍÍ•ÉÐ•áÁ•Ñ•‘}É•…Í½¸¥¸•Ù…±Õ…Ñ¥½¸¹É•©•Ñ¥½¹}É•…Í½¹Ì(()ÁåÑ•ÍÐ¹µ…É¬¹Á…É…µ•ÑÉ¥é” ‰™…¥±¥¹}Í•µ•¹Ðˆ°l‰ÑÉ…¥¹¥¹œˆ°€‰Ù…±¥‘…Ñ¥½¸‰t¤)‘•˜Ñ•ÍÑ}é•É½}ÑÉ…‘•Í}‰•™½É•}Ñ•ÍÑ}…É•}±…ÍÍ¥™¥•‘}…Í}É•©•Ñ•‘}Ù…±¥‘…Ñ¥½¸ (€€€µ½¹­•åÁ…Ñ èÁåÑ•ÍÐ¹5½¹­•åA…Ñ °™…¥±¥¹}Í•µ•¹ÐèÍÑÈ(¤€´ø9½¹”è(€€€ÑÉ…¥¹¥¹œ€ô•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QI%9%9°(€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀ¸ÀÈˆ°(€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀ¸ÀÄˆ°(€€€€€€€ÑÉ…‘•}½Õ¹ÐôÀ¥˜™…¥±¥¹}Í•µ•¹Ð€ôô€‰ÑÉ…¥¹¥¹œˆ•±Í”€Ä°(€€€€¤(€€€Ù…±¥‘…Ñ¥½¸€ô•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹Y1%Q%=8°(€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀ¸ÀÈˆ°(€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀ¸ÀÄˆ°(€€€€€€€ÑÉ…‘•}½Õ¹ÐôÀ¥˜™…¥±¥¹}Í•µ•¹Ð€ôô€‰Ù…±¥‘…Ñ¥½¸ˆ•±Í”€Ä°(€€€€¤(€€€É•ÍÕ±Ð€ô•Ù…±Õ…Ñ•}Ý¥Ñ¡}µ•ÑÉ¥Ì (€€€€€€€µ½¹­•åÁ…Ñ °(€€€€€€€ÑÉ…¥¹¥¹œõÑÉ…¥¹¥¹œ°(€€€€€€€Ù…±¥‘…Ñ¥½¸õÙ…±¥‘…Ñ¥½¸°(€€€€€€€Ñ•ÍÐõ•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QMP°(€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀ¸ÀÈˆ°(€€€€€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀ¸ÀÄˆ°(€€€€€€€€¤°(€€€€¤((€€€¥˜™…¥±¥¹}Í•µ•¹Ð€ôô€‰ÑÉ…¥¹¥¹œˆè(€€€€€€€…ÍÍ•ÉÐÉ•ÍÕ±Ð¹•Ù…±Õ…Ñ¥½¹Ì€ôô€ ¤(€€€€€€€…ÍÍ•ÉÐÉ•ÍÕ±Ð¹™Õ¹¹•±l‰ÑÉ…¥¹¥¹}É•Ñ…¥¹•‘}½Õ¹Ð‰t€ôô€À(€€€•±Í”è(€€€€€€€•Ù…±Õ…Ñ¥½¸€ôÉ•ÍÕ±Ð¹•Ù…±Õ…Ñ¥½¹ÍlÁt(€€€€€€€…ÍÍ•ÉÐ•Ù…±Õ…Ñ¥½¸¹ÍÑ…ÑÕÌ¥Ì‘•Õ¥‘•‘…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹I)Q}Y1%Q%=8(€€€€€€€…ÍÍ•ÉÐ•Ù…±Õ…Ñ¥½¸¹½¹™¥Éµ•‘}Ý¥¹‘½Ý}½Õ¹Ð€ôô€À(€€€€€€€…ÍÍ•ÉÐ•Ù…±Õ…Ñ¥½¸¹Ñ•ÍÐ¥Ì9½¹”(€€€€€€€…ÍÍ•ÉÐ€ (€€€€€€€€€€€‘•Õ¥‘•‘I•©•Ñ¥½¹I•…Í½¸¹%9MU%%9Q}QIL(€€€€€€€€€€€¥¸•Ù…±Õ…Ñ¥½¸¹É•©•Ñ¥½¹}É•…Í½¹Ì(€€€€€€€€¤(()‘•˜Ñ•ÍÑ}É•©•Ñ¥½¹}É•…Í½¹Í}…É•}ÍÑ…‰±•}…¹‘}Õ¹¥ÅÕ” (€€€µ½¹­•åÁ…Ñ èÁåÑ•ÍÐ¹5½¹­•åA…Ñ °(¤€´ø9½¹”è(€€€é•É¼€ô•½¹½µ¥}µ•ÑÉ¥Œ (€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QI%9%9°(€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸ôˆÀˆ°(€€€€€€€ÁÉ½‘ÕÑ¥½¹}É•ÑÕÉ¸ôˆÀˆ°(€€€€€€€ÑÉ…‘•}½Õ¹ÐôÀ°(€€€€¤(€€€É•ÍÕ±Ð€ô•Ù…±Õ…Ñ•}Ý¥Ñ¡}µ•ÑÉ¥Ì (€€€€€€€µ½¹­•åÁ…Ñ °(€€€€€€€ÑÉ…¥¹¥¹œõé•É¼°(€€€€€€€Ù…±¥‘…Ñ¥½¸õÉ•Á±…”¡é•É¼°Í•µ•¹ÐõI•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹Y1%Q%=8¤°(€€€€€€€Ñ•ÍÐõÉ•Á±…”¡é•É¼°Í•µ•¹ÐõI•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QMP¤°(€€€€¤(€€€…ÍÍ•ÉÐÉ•ÍÕ±Ð¹•Ù…±Õ…Ñ¥½¹Ì€ôô€ ¤(€€€…ÍÍ•ÉÐÉ•ÍÕ±Ð¹™Õ¹¹•±l‰ÑÉ…¥¹¥¹}É•Ñ…¥¹•‘}½Õ¹Ð‰t€ôô€À(()‘•˜Ñ•ÍÑ}•Ù…±Õ…Ñ¥½¹}Á…ÍÍ•Í}•™™•Ñ¥Ù•}™••Í}…¹‘}Í±¥ÁÁ…•}Ñ½}É•…±}Í¥µÕ±…Ñ½È ¤€´ø9½¹”è(€€€‘…Ñ…Í•Ð°µ…ÑÉ¥à°Ý¥¹‘½ÝÌ€ô•Ù…±Õ…Ñ¥½¹}•Ù¥‘•¹”¡Ý¥¹‘½ÝÌôÄ¤(€€€Í¥µÕ±…Ñ¥½¸€ôM¥µÕ±…Ñ¥½¹½¹™¥ÕÉ…Ñ¥½¸ (€€€€€€€™••}É…Ñ”õ•¥µ…° ˆÀ¸ÀÀÈÌˆ¤°Í±¥ÁÁ…•}É…Ñ”õ•¥µ…° ˆÀ¸ÀÀÀÜˆ¤(€€€€¤(€€€…±±Ìè±¥ÍÑmÑÕÁ±•m•¥µ…°°•¥µ…°°‰½½±ut€ômt((€€€‘•˜Í¥µÕ±…Ñ” ¨©­Ý…ÉÌè½‰©•Ð¤€´øM¥µÁ±•9…µ•ÍÁ…”è(€€€€€€€™•”€ô­Ý…ÉÍl‰™••}É…Ñ”‰t(€€€€€€€Í±¥ÁÁ…”€ô­Ý…ÉÍl‰Í±¥ÁÁ…•}É…Ñ”‰t(€€€€€€€™¥±Ñ•É•€ô­Ý…ÉÌ¹•Ð ‰‘•¥Í¥½¹}…Ñ¥½¹}™¥±Ñ•Èˆ¤¥Ì¹½Ð9½¹”(€€€€€€€…ÍÍ•ÉÐ¥Í¥¹ÍÑ…¹”¡™•”°•¥µ…°¤(€€€€€€€…ÍÍ•ÉÐ¥Í¥¹ÍÑ…¹”¡Í±¥ÁÁ…”°•¥µ…°¤(€€€€€€€…±±Ì¹…ÁÁ•¹ ¡™•”°Í±¥ÁÁ…”°™¥±Ñ•É•¤¤(€€€€€€€É•ÑÕÉ¸M¥µÁ±•9…µ•ÍÁ…” (€€€€€€€€€€€Ñ½Ñ…±}É•ÑÕÉ¸õ•¥µ…° ˆÀ¸ÀÈˆ¤¥˜™¥±Ñ•É••±Í”•¥µ…° ˆÀ¸ÀÄˆ¤°(€€€€€€€€€€€µ…á}‘É…Ý‘½Ý¸õ•¥µ…° ˆÀ¸ÀÄˆ¤°(€€€€€€€€€€€ÑÉ…‘•}½Õ¹ÐôÄ°(€€€€€€€€€€€™••Í}Á…¥õ•¥µ…° ˆÄ¸ÀÀˆ¤°(€€€€€€€€¤((€€€‰•¹¡µ…É¬€ôM¥µÁ±•9…µ•ÍÁ…”¡Ñ½Ñ…±}É•ÑÕÉ¸õ•¥µ…° À¤¤(€€€Ý¥Ñ € (€€€€€€€Á…Ñ ¹½‰©•Ð (€€€€€€€€€€€•‘•}Õ¥‘•‘}µ½‘Õ±”°€‰Í¥µÕ±…Ñ•}¡¥ÍÑ½É¥…±}™•…ÑÕÉ•}µ…ÑÉ¥àˆ°Í¥µÕ±…Ñ”(€€€€€€€€¤°(€€€€€€€Á…Ñ ¹½‰©•Ð (€€€€€€€€€€€•‘•}Õ¥‘•‘}µ½‘Õ±”°€‰Í¥µÕ±…Ñ•}‰Õå}…¹‘}¡½±ˆ°É•ÑÕÉ¹}Ù…±Õ”õ‰•¹¡µ…É¬(€€€€€€€€¤°(€€€€¤è(€€€€€€€É•ÍÕ±Ð€ô•Ù…±Õ…Ñ•}•‘•}Õ¥‘•‘}…¹‘¥‘…Ñ•Ì (€€€€€€€€€€€‘…Ñ…Í•Ðõ‘…Ñ…Í•Ð°(€€€€€€€€€€€™•…ÑÕÉ•}µ…ÑÉ¥àõµ…ÑÉ¥à°(€€€€€€€€€€€¡åÁ½Ñ¡•Í•Ìô¡¡åÁ½Ñ¡•Í¥Ì ¤°¤°(€€€€€€€€€€€¥¹¥Ñ¥…±}…Í õ•¥µ…° ˆÄÀÀÀÀˆ¤°(€€€€€€€€€€€Í¥µÕ±…Ñ¥½¸õÍ¥µÕ±…Ñ¥½¸°(€€€€€€€€€€€Ý¥¹‘½ÝÌõÝ¥¹‘½ÝÌ°(€€€€€€€€€€€Ñ¡É•Í¡½±‘ÌõÙ¥‘•¹•Q¡É•Í¡½±‘Ì (€€€€€€€€€€€€€€€µ¥¹¥µÕµ}½‰Í•ÉÙ…Ñ¥½¹ÌôÄ°(€€€€€€€€€€€€€€€µ¥¹¥µÕµ}Ý¥¹‘½ÝÌôÄ°(€€€€€€€€€€€€€€€µ¥¹¥µÕµ}ÍÑ…‰±•}‘¥É•Ñ¥½¹}É…Ñ¥¼õ•¥µ…° Ä¤°(€€€€€€€€€€€€¤°(€€€€€€€€¤(€€€…ÍÍ•ÉÐÉ•ÍÕ±Ð¹•Ù…±Õ…Ñ¥½¹ÍlÁt¹ÍÑ…ÑÕÌ¥Ì‘•Õ¥‘•‘…¹‘¥‘…Ñ•MÑ…ÑÕÌ¹=9%I5(€€€…ÍÍ•ÉÐ…±±Ì(€€€…ÍÍ•ÉÐ…±° (€€€€€€€™•”€ôôÍ¥µÕ±…Ñ¥½¸¹™••}É…Ñ”…¹Í±¥ÁÁ…”€ôôÍ¥µÕ±…Ñ¥½¸¹Í±¥ÁÁ…•}É…Ñ”(€€€€€€€™½È™•”°Í±¥ÁÁ…”°|¥¸…±±Ì(€€€€¤(€€€…ÍÍ•ÉÐ…¹ä¡™¥±Ñ•É•™½È|°|°™¥±Ñ•É•¥¸…±±Ì¤(€€€…ÍÍ•ÉÐ…¹ä¡¹½Ð™¥±Ñ•É•™½È|°|°™¥±Ñ•É•¥¸…±±Ì¤(