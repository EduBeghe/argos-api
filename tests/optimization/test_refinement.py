from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.historical_data import (
    HistoricalDataset,
    HistoricalDatasetIdentity,
    HistoricalDatasetRange,
)
from app.optimization.refinement import (
    HistoricalRefinementError,
    RecommendationStatus,
    RefinementBudget,
    RefinementSegmentName,
    _buy_and_hold,
    _calculate_buy_and_hold_entry,
    candidate_for,
    generate_ablation_configurations,
    generate_score_configurations,
    generate_strategy_configurations,
    refine_historical_dataset,
    refine_historical_strategy,
    split_feature_matrix,
)
from app.optimization.simulation import HistoricalSimulationResult
from app.paper.config import SimulationConfiguration
from app.scores.calculator import (
    BASELINE_TECHNICAL_V1_PARAMETERS,
    BaselineTechnicalParameters,
)
from app.strategies.baseline_long_only import (
    BASELINE_LONG_ONLY_V1_PARAMETERS,
    BaselineLongOnlyParameters,
)

START = datetime(2024, 1, 1, tzinfo=UTC)
INSTRUMENT = Instrument(
    Asset.ETH_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)


def dataset() -> HistoricalDataset:
    return HistoricalDataset(
        HistoricalDatasetIdentity(
            INSTRUMENT, HistoricalDatasetRange(START, START + timedelta(hours=6))
        ),
        72,
        72,
        START,
        START + timedelta(hours=6) - timedelta(minutes=5),
        0,
        None,
        None,
        (),
        "refinement-fingerprint",
    )


def matrix() -> HistoricalFeatureMatrix:
    return HistoricalFeatureMatrix(
        "refinement-fingerprint",
        tuple(
            HistoricalFeatureRow(
                START + timedelta(minutes=25 + 30 * index),
                Decimal(100 + index),
                Decimal(1),
                Decimal(1),
                Decimal(50),
                Decimal(1),
                Decimal(0),
                Decimal(0),
                Decimal(1),
            )
            for index in range(10)
        ),
    )


def fake_simulator(
    test_return: Decimal = Decimal("0.03"),
) -> tuple[
    list[tuple[int, BaselineTechnicalParameters, BaselineLongOnlyParameters]],
    Callable[..., HistoricalSimulationResult],
]:
    calls: list[
        tuple[int, BaselineTechnicalParameters, BaselineLongOnlyParameters]
    ] = []

    def simulate(**kwargs: object) -> HistoricalSimulationResult:
        selected_matrix = cast(HistoricalFeatureMatrix, kwargs["feature_matrix"])
        score = cast(BaselineTechnicalParameters, kwargs["score_parameters"])
        strategy = cast(BaselineLongOnlyParameters, kwargs["strategy_parameters"])
        calls.append((len(selected_matrix.rows), score, strategy))
        baseline = (
            score == BASELINE_TECHNICAL_V1_PARAMETERS
            and strategy == BASELINE_LONG_ONLY_V1_PARAMETERS
        )
        best = score.trend_weight == 1 and strategy.entry_threshold == Decimal("0.70")
        if len(selected_matrix.rows) == 3:
            total_return = score.trend_weight + score.volume_confirmation_weight / 10
        elif len(selected_matrix.rows) == 6:
            total_return = score.trend_weight + strategy.entry_threshold / 10
        elif selected_matrix.rows[0].candle_open_time.hour == 3:
            total_return = (
                Decimal("0.05")
                if best
                else Decimal("0.01")
                if baseline
                else Decimal("-0.01")
            )
        else:
            total_return = (
                test_return
                if best
                else Decimal("0.01")
                if baseline
                else Decimal("-0.01")
            )
        return cast(
            HistoricalSimulationResult,
            SimpleNamespace(
                total_return=total_return,
                absolute_pnl=total_return * Decimal("10000"),
                max_drawdown=Decimal("0.04"),
                trade_count=2,
                fees_paid=Decimal("3"),
                final_equity=Decimal("10000") * (Decimal(1) + total_return),
            ),
        )

    return calls, simulate


def small_budget() -> RefinementBudget:
    return RefinementBudget(
        score_screening_top_k=1,
        validation_top_k=1,
        max_score_configurations=200,
        max_joint_candidates=60,
    )


def test_simplex_grid_and_production_baseline_are_stable() -> None:
    values = generate_score_configurations()
    assert values == generate_score_configurations()
    assert len(values[:-1]) == 175
    assert (
        len({value for value in values[:-1] if value.volume_confirmation_weight == 0})
        == 35
    )
    assert len(values) == 176
    assert values.count(BASELINE_TECHNICAL_V1_PARAMETERS) == 1
    assert len(generate_strategy_configurations()) == 60


def test_candidate_identity_is_complete_and_reproducible() -> None:
    baseline = candidate_for(
        BASELINE_TECHNICAL_V1_PARAMETERS, BASELINE_LONG_ONLY_V1_PARAMETERS
    )
    repeated = candidate_for(
        BASELINE_TECHNICAL_V1_PARAMETERS, BASELINE_LONG_ONLY_V1_PARAMETERS
    )
    changed_score = candidate_for(
        BaselineTechnicalParameters(
            Decimal("0.5"), Decimal("0.25"), Decimal(0), Decimal("0.25"), Decimal(0)
        ),
        BASELINE_LONG_ONLY_V1_PARAMETERS,
    )
    changed_strategy = candidate_for(
        BASELINE_TECHNICAL_V1_PARAMETERS,
        BaselineLongOnlyParameters(Decimal("0.65"), Decimal("0.20"), Decimal("15")),
    )
    assert baseline == repeated
    assert baseline.is_production_baseline
    assert (
        len(
            {
                baseline.stable_candidate_id,
                changed_score.stable_candidate_id,
                changed_strategy.stable_candidate_id,
            }
        )
        == 3
    )


def test_split_and_screening_are_chronological_and_do_not_use_validation_or_test() -> (
    None
):
    matrices, segments = split_feature_matrix(matrix())
    assert [item.decision_count for item in segments] == [6, 2, 2]
    assert segments[0].last_boundary < segments[1].first_boundary
    assert segments[1].last_boundary < segments[2].first_boundary
    calls, fake = fake_simulator()
    with patch(
        "app.optimization.refinement.simulate_historical_feature_matrix",
        side_effect=fake,
    ):
        result = refine_historical_strategy(
            dataset=dataset(),
            feature_matrix=matrix(),
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            budget=small_budget(),
        )
    assert result.screening_segment.decision_count == 3
    assert all(size == 3 for size, _, _ in calls[:176])
    assert (
        matrices[RefinementSegmentName.VALIDATION].rows[0]
        not in matrices[RefinementSegmentName.TRAINING].rows
    )


def test_budget_limits_fail_closed_without_silent_truncation() -> None:
    with pytest.raises(HistoricalRefinementError, match="score configurations"):
        refine_historical_strategy(
            dataset=dataset(),
            feature_matrix=matrix(),
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            budget=RefinementBudget(1, 1, 175, 60),
        )
    with pytest.raises(HistoricalRefinementError, match="joint candidates"):
        refine_historical_strategy(
            dataset=dataset(),
            feature_matrix=matrix(),
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            budget=RefinementBudget(2, 1, 200, 119),
        )


def test_only_retained_scores_reach_joint_refinement_and_matrix_builds_once() -> None:
    calls, fake = fake_simulator()
    with (
        patch(
            "app.optimization.refinement.build_historical_feature_matrix",
            return_value=matrix(),
        ) as build,
        patch(
            "app.optimization.refinement.simulate_historical_feature_matrix",
            side_effect=fake,
        ),
    ):
        result = refine_historical_dataset(
            dataset=dataset(),
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            budget=small_budget(),
        )
    build.assert_called_once()
    assert result.counts.score_configurations_retained == 1
    assert result.counts.joint_candidates_evaluated == 60
    retained = set(result.retained_score_parameters)
    assert {
        item.candidate.score_parameters for item in result.training_ranking
    } == retained
    assert all(call[1] in retained for call in calls[176:236])


def test_recommendation_beats_baseline_and_test_cannot_change_selection() -> None:
    selected = []
    statuses = []
    for test_return in (Decimal("0.03"), Decimal("-0.03")):
        _, fake = fake_simulator(test_return)
        with patch(
            "app.optimization.refinement.simulate_historical_feature_matrix",
            side_effect=fake,
        ):
            result = refine_historical_strategy(
                dataset=dataset(),
                feature_matrix=matrix(),
                initial_cash=Decimal("10000"),
                fee_rate=Decimal("0.001"),
                slippage_rate=Decimal("0.0005"),
                budget=small_budget(),
            )
        selected.append(
            result.recommendation.candidate if result.recommendation else None
        )
        statuses.append(result.recommendation_status)
        assert (
            result.validation_results[0].total_return
            > result.production_baseline.validation.total_return
        )
        assert "score weights" in result.recommendation_explanation
        assert "production baseline" in result.recommendation_explanation
        assert "buy and hold" in result.recommendation_explanation
    assert selected[0] == selected[1]
    assert statuses == [
        RecommendationStatus.RECOMMENDED_AND_TEST_CONFIRMED,
        RecommendationStatus.RECOMMENDED_BUT_TEST_FAILED,
    ]


def test_worse_than_baseline_is_not_recommended() -> None:
    _, fake = fake_simulator()

    def worse(**kwargs: object) -> HistoricalSimulationResult:
        value = cast(SimpleNamespace, fake(**kwargs))
        matrix_value = cast(HistoricalFeatureMatrix, kwargs["feature_matrix"])
        score = cast(BaselineTechnicalParameters, kwargs["score_parameters"])
        strategy = cast(BaselineLongOnlyParameters, kwargs["strategy_parameters"])
        baseline = (
            score == BASELINE_TECHNICAL_V1_PARAMETERS
            and strategy == BASELINE_LONG_ONLY_V1_PARAMETERS
        )
        if (
            len(matrix_value.rows) == 2
            and matrix_value.rows[0].candle_open_time.hour == 3
        ):
            value.total_return = Decimal("0.02") if baseline else Decimal("0.01")
        return cast(HistoricalSimulationResult, value)

    with patch(
        "app.optimization.refinement.simulate_historical_feature_matrix",
        side_effect=worse,
    ):
        result = refine_historical_strategy(
            dataset=dataset(),
            feature_matrix=matrix(),
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            budget=small_budget(),
        )
    assert result.recommendation is None
    assert result.recommendation_status is RecommendationStatus.NO_VALIDATION_CANDIDATE


def test_ablation_covers_every_signal_and_result_is_deterministic() -> None:
    results = []
    for _ in range(2):
        _, fake = fake_simulator()
        with patch(
            "app.optimization.refinement.simulate_historical_feature_matrix",
            side_effect=fake,
        ):
            results.append(
                refine_historical_strategy(
                    dataset=dataset(),
                    feature_matrix=matrix(),
                    initial_cash=Decimal("10000"),
                    fee_rate=Decimal("0.001"),
                    slippage_rate=Decimal("0.0005"),
                    budget=small_budget(),
                )
            )
    assert results[0] == results[1]
    assert {item.disabled_signal for item in results[0].ablation_summary} == {
        "trend",
        "rsi",
        "return_1",
        "return_12",
        "volume_confirmation",
    }


def test_ablation_configurations_disable_only_one_signal_from_production() -> None:
    configurations = generate_ablation_configurations()
    baseline = BASELINE_TECHNICAL_V1_PARAMETERS
    baseline_directional = (
        baseline.trend_weight,
        baseline.rsi_weight,
        baseline.return_1_weight,
        baseline.return_12_weight,
    )
    assert len(configurations) == 5
    for disabled_index, configuration in enumerate(configurations[:4]):
        directional = (
            configuration.trend_weight,
            configuration.rsi_weight,
            configuration.return_1_weight,
            configuration.return_12_weight,
        )
        assert directional[disabled_index] == 0
        assert all(
            weight > 0
            for index, weight in enumerate(directional)
            if index != disabled_index
        )
        assert sum(directional, start=Decimal(0)) == 1
        assert (
            configuration.volume_confirmation_weight
            == baseline.volume_confirmation_weight
        )
        enabled = tuple(index for index in range(4) if index != disabled_index)
        for left, right in pairwise(enabled):
            proportional_error = abs(
                directional[left] * baseline_directional[right]
                - directional[right] * baseline_directional[left]
            )
            assert proportional_error <= Decimal("0.000000000000000001")

    volume_ablation = configurations[-1]
    assert (
        volume_ablation.trend_weight,
        volume_ablation.rsi_weight,
        volume_ablation.return_1_weight,
        volume_ablation.return_12_weight,
    ) == baseline_directional
    assert volume_ablation.volume_confirmation_weight == 0


def test_ablation_uses_production_strategy_screening_segment_and_counts_five() -> None:
    calls, fake = fake_simulator()
    with patch(
        "app.optimization.refinement.simulate_historical_feature_matrix",
        side_effect=fake,
    ):
        result = refine_historical_strategy(
            dataset=dataset(),
            feature_matrix=matrix(),
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            budget=small_budget(),
        )
    ablation_calls = calls[-5:]
    assert len(ablation_calls) == 5
    assert all(
        size == result.screening_segment.decision_count for size, _, _ in ablation_calls
    )
    assert all(
        strategy == BASELINE_LONG_ONLY_V1_PARAMETERS
        for _, _, strategy in ablation_calls
    )
    assert [score for _, score, _ in ablation_calls] == list(
        generate_ablation_configurations()
    )
    assert result.counts.score_configurations_screened == 176
    assert result.counts.joint_candidates_evaluated == 60
    assert result.counts.simulation_count_total == len(calls)
    assert result.counts.simulation_count_total == 246
    for item in result.ablation_summary:
        assert item.baseline_score_parameters == BASELINE_TECHNICAL_V1_PARAMETERS
        assert item.ablated_score_parameters == item.ablated.candidate.score_parameters
        assert item.baseline.segment is RefinementSegmentName.SCREENING
        assert item.ablated.segment is RefinementSegmentName.SCREENING
        assert (
            item.return_difference_from_baseline
            == item.ablated.total_return - item.baseline.total_return
        )


def test_buy_and_hold_applies_entry_exit_fees_and_slippage() -> None:
    value = _buy_and_hold(
        HistoricalFeatureMatrix("fingerprint", matrix().rows[:2]),
        RefinementSegmentName.TEST,
        INSTRUMENT,
        Decimal("10000.000000000000000000"),
        SimulationConfiguration(
            fee_rate=Decimal("0.001"), slippage_rate=Decimal("0.0005")
        ),
    )
    without_costs = _buy_and_hold(
        HistoricalFeatureMatrix("fingerprint", matrix().rows[:2]),
        RefinementSegmentName.TEST,
        INSTRUMENT,
        Decimal("10000.000000000000000000"),
        SimulationConfiguration(fee_rate=Decimal(0), slippage_rate=Decimal(0)),
    )
    assert value.trade_count == 2
    assert value.fees_paid > 0
    assert value.final_equity < without_costs.final_equity


def realistic_price_matrix(
    entry_price: Decimal, exit_price: Decimal
) -> HistoricalFeatureMatrix:
    rows = matrix().rows[:2]
    return HistoricalFeatureMatrix(
        "fingerprint",
        (
            HistoricalFeatureRow(
                rows[0].candle_open_time,
                entry_price,
                rows[0].ema_20,
                rows[0].ema_50,
                rows[0].rsi_14,
                rows[0].atr_14,
                rows[0].return_1,
                rows[0].return_12,
                rows[0].relative_volume_20,
            ),
            HistoricalFeatureRow(
                rows[1].candle_open_time,
                exit_price,
                rows[1].ema_20,
                rows[1].ema_50,
                rows[1].rsi_14,
                rows[1].atr_14,
                rows[1].return_1,
                rows[1].return_12,
                rows[1].relative_volume_20,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("initial_cash", "entry_price", "fee_rate"),
    (
        (Decimal("10"), Decimal("0.123456789012345678"), Decimal(0)),
        (Decimal("100"), Decimal("12.345678901234567890"), Decimal("0.001")),
        (Decimal("10000"), Decimal("68432.123456789012345678"), Decimal("0.001")),
        (Decimal("1000000"), Decimal("999999.999999999999999999"), Decimal(0)),
    ),
)
def test_buy_and_hold_entry_is_financeable_across_prices_cash_and_fees(
    initial_cash: Decimal, entry_price: Decimal, fee_rate: Decimal
) -> None:
    configuration = SimulationConfiguration(
        fee_rate=fee_rate, slippage_rate=Decimal("0.0005")
    )
    first = _calculate_buy_and_hold_entry(
        cash=initial_cash,
        observed_price=entry_price,
        configuration=configuration,
    )
    second = _calculate_buy_and_hold_entry(
        cash=initial_cash,
        observed_price=entry_price,
        configuration=configuration,
    )
    assert first == second
    assert first.resulting_cash >= 0
    assert first.quantity > 0
    assert first.execution_price > entry_price
    assert first.fee_amount >= 0


def test_realistic_btc_buy_and_hold_regression_is_complete_and_deterministic() -> None:
    history = realistic_price_matrix(
        Decimal("68432.123456789012345678"),
        Decimal("70123.987654321098765432"),
    )
    configuration = SimulationConfiguration(
        fee_rate=Decimal("0.001"), slippage_rate=Decimal("0.0005")
    )
    entry = _calculate_buy_and_hold_entry(
        cash=Decimal("10000"),
        observed_price=history.rows[0].close_price,
        configuration=configuration,
    )
    results = tuple(
        _buy_and_hold(
            history,
            RefinementSegmentName.TEST,
            INSTRUMENT,
            Decimal("10000"),
            configuration,
        )
        for _ in range(2)
    )
    costless = _buy_and_hold(
        history,
        RefinementSegmentName.TEST,
        INSTRUMENT,
        Decimal("10000"),
        SimulationConfiguration(fee_rate=Decimal(0), slippage_rate=Decimal(0)),
    )
    without_slippage = _buy_and_hold(
        history,
        RefinementSegmentName.TEST,
        INSTRUMENT,
        Decimal("10000"),
        SimulationConfiguration(fee_rate=Decimal("0.001"), slippage_rate=Decimal(0)),
    )
    assert results[0] == results[1]
    assert entry.resulting_cash >= 0
    assert entry.quantity > 0
    assert results[0].trade_count == 2
    assert results[0].final_equity > 0
    assert results[0].fees_paid > entry.fee_amount
    assert results[0].final_equity < without_slippage.final_equity
    assert results[0].final_equity < costless.final_equity
