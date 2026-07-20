from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
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
from app.optimization.simulation import (
    HistoricalSimulationError,
    HistoricalSimulationResult,
    simulate_historical_strategy,
)
from app.paper.calculation import calculate_open_position
from app.paper.config import SimulationConfiguration, quantize_financial
from app.scores.calculator import BaselineTechnicalParameters
from app.strategies.baseline_long_only import (
    BaselineLongOnlyParameters,
    evaluate_baseline_long_only,
)

START = datetime(2024, 1, 1, tzinfo=UTC)
INSTRUMENT = Instrument(
    Asset.ETH_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)


def source_dataset() -> HistoricalDataset:
    identity = HistoricalDatasetIdentity(
        INSTRUMENT, HistoricalDatasetRange(START, START + timedelta(hours=8))
    )
    return HistoricalDataset(
        identity,
        96,
        96,
        START,
        START + timedelta(minutes=475),
        0,
        None,
        None,
        (),
        "stable-fingerprint",
    )


def row(boundary: datetime, price: str) -> HistoricalFeatureRow:
    return HistoricalFeatureRow(
        boundary - timedelta(minutes=5),
        Decimal(price),
        Decimal(1),
        Decimal(1),
        Decimal(50),
        Decimal(1),
        Decimal(0),
        Decimal(0),
        Decimal(1),
    )


def matrix() -> HistoricalFeatureMatrix:
    return HistoricalFeatureMatrix(
        "stable-fingerprint",
        tuple(
            row(START + timedelta(hours=5, minutes=30 * index), price)
            for index, price in enumerate(("100", "110", "90", "95"))
        ),
    )


def simulate(
    scores: tuple[str, ...] = ("0.7", "0.7", "0.1", "0.1"),
    *,
    fee_rate: Decimal = Decimal("0.001"),
    slippage_rate: Decimal = Decimal("0.0005"),
    decision_interval: timedelta = timedelta(minutes=30),
) -> HistoricalSimulationResult:
    outputs = [SimpleNamespace(overall_score=Decimal(value)) for value in scores]
    with (
        patch(
            "app.optimization.simulation.build_historical_feature_matrix",
            return_value=matrix(),
        ),
        patch(
            "app.optimization.simulation.calculate_baseline_score", side_effect=outputs
        ),
    ):
        return simulate_historical_strategy(
            dataset=source_dataset(),
            initial_cash=Decimal("10000"),
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            decision_interval=decision_interval,
        )


def test_same_input_is_deterministic_and_boundaries_have_no_lookahead() -> None:
    first = simulate()
    second = simulate()
    assert first == second
    assert [point.boundary.minute for point in first.equity_curve] == [0, 30, 0, 30]
    assert all(
        point.source_candle_open_time + timedelta(minutes=5) == point.boundary
        for point in first.equity_curve
    )
    assert all(trade.source_candle_open_time < trade.boundary for trade in first.trades)


def test_result_preserves_effective_simulation_configuration() -> None:
    fee_rate = Decimal("0.0025")
    slippage_rate = Decimal("0.00075")
    decision_interval = timedelta(minutes=30)

    result = simulate(
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        decision_interval=decision_interval,
    )

    assert result.account_currency == "USDT"
    assert result.fee_rate == fee_rate
    assert result.slippage_rate == slippage_rate
    assert result.decision_interval == decision_interval


def test_result_preserves_effective_strategy_parameters() -> None:
    parameters = BaselineLongOnlyParameters(
        Decimal("0.65"), Decimal("0.15"), Decimal("20")
    )
    outputs = [SimpleNamespace(overall_score=Decimal("0.7"))] * 4
    with (
        patch(
            "app.optimization.simulation.build_historical_feature_matrix",
            return_value=matrix(),
        ),
        patch(
            "app.optimization.simulation.calculate_baseline_score", side_effect=outputs
        ),
    ):
        result = simulate_historical_strategy(
            dataset=source_dataset(),
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            strategy_parameters=parameters,
        )
    assert result.strategy_parameters == parameters


def test_result_preserves_effective_score_parameters() -> None:
    parameters = BaselineTechnicalParameters(
        Decimal("0.5"), Decimal("0.25"), Decimal(0), Decimal("0.25"), Decimal("0.125")
    )
    outputs = [SimpleNamespace(overall_score=Decimal("0.7"))] * 4
    with (
        patch(
            "app.optimization.simulation.build_historical_feature_matrix",
            return_value=matrix(),
        ),
        patch(
            "app.optimization.simulation.calculate_baseline_score", side_effect=outputs
        ) as calculate,
    ):
        result = simulate_historical_strategy(
            dataset=source_dataset(),
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            score_parameters=parameters,
        )
    assert result.score_parameters == parameters
    assert all(
        call.kwargs["parameters"] == parameters for call in calculate.call_args_list
    )


def test_results_retain_distinguishable_rate_configurations_deterministically() -> None:
    first = simulate(fee_rate=Decimal("0.001"), slippage_rate=Decimal("0.0005"))
    repeated = simulate(fee_rate=Decimal("0.001"), slippage_rate=Decimal("0.0005"))
    different = simulate(fee_rate=Decimal("0.002"), slippage_rate=Decimal("0.0008"))

    assert first == repeated
    assert (first.fee_rate, first.slippage_rate) == (
        Decimal("0.001"),
        Decimal("0.0005"),
    )
    assert (different.fee_rate, different.slippage_rate) == (
        Decimal("0.002"),
        Decimal("0.0008"),
    )
    assert (first.fee_rate, first.slippage_rate) != (
        different.fee_rate,
        different.slippage_rate,
    )


def test_buy_no_action_and_sell_all_reuse_paper_execution_math() -> None:
    result = simulate()
    assert (
        result.decision_count,
        result.trade_count,
        result.buy_count,
        result.partial_sell_count,
        result.sell_all_count,
    ) == (4, 2, 1, 0, 1)
    buy, sell = result.trades
    configuration = SimulationConfiguration(
        1, "USDT", Decimal("0.001"), Decimal("0.0005")
    )
    expected_buy = calculate_open_position(
        cash=Decimal("10000.000000000000000000"),
        total_equity=Decimal("10000.000000000000000000"),
        observed_price=Decimal("100"),
        target_percentage=Decimal("15.00"),
        configuration=configuration,
    )
    assert (
        buy.execution_price,
        buy.quantity,
        buy.notional,
        buy.fee,
        buy.cash_after,
    ) == (
        expected_buy.execution_price,
        expected_buy.quantity,
        expected_buy.executed_notional,
        expected_buy.fee_amount,
        expected_buy.resulting_cash,
    )
    assert result.equity_curve[1].cash == result.equity_curve[0].cash
    assert (
        result.equity_curve[1].position_quantity
        == result.equity_curve[0].position_quantity
    )
    assert sell.position_quantity_after == 0
    assert result.final_open_position is None
    assert result.final_cash >= 0
    assert all(point.position_quantity >= 0 for point in result.equity_curve)


def test_open_position_is_marked_at_final_boundary() -> None:
    result = simulate(("0.7", "0.7", "0.7", "0.7"))
    assert result.final_open_position is not None
    position = result.final_open_position
    assert position.market_value == quantize_financial(
        position.quantity * Decimal("95")
    )
    assert result.final_equity == quantize_financial(
        result.final_cash + position.market_value
    )
    assert result.unrealized_pnl == position.unrealized_pnl


def test_metrics_and_drawdown_match_small_equity_curve() -> None:
    result = simulate()
    running_peak = result.initial_cash
    expected_drawdown = Decimal(0)
    for point in result.equity_curve:
        running_peak = max(running_peak, point.equity)
        expected_drawdown = max(
            expected_drawdown,
            quantize_financial((running_peak - point.equity) / running_peak),
        )
    assert result.max_drawdown == expected_drawdown
    assert result.absolute_pnl == quantize_financial(
        result.final_equity - result.initial_cash
    )
    assert result.instrument is INSTRUMENT


def test_sell_without_position_fails_closed() -> None:
    invalid_sell = evaluate_baseline_long_only(
        overall_score=Decimal("0.1"), position_is_open=True
    )
    with (
        patch(
            "app.optimization.simulation.build_historical_feature_matrix",
            return_value=matrix(),
        ),
        patch(
            "app.optimization.simulation.calculate_baseline_score",
            return_value=SimpleNamespace(overall_score=Decimal("0.7")),
        ),
        patch(
            "app.optimization.simulation.evaluate_baseline_long_only",
            return_value=invalid_sell,
        ),
        pytest.raises(HistoricalSimulationError, match="without a position"),
    ):
        simulate_historical_strategy(
            dataset=source_dataset(),
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
        )


def test_insufficient_dataset_has_clear_error() -> None:
    with (
        patch(
            "app.optimization.simulation.build_historical_feature_matrix",
            return_value=HistoricalFeatureMatrix("stable-fingerprint", ()),
        ),
        pytest.raises(HistoricalSimulationError, match="no complete Feature row"),
    ):
        simulate_historical_strategy(
            dataset=source_dataset(),
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
        )
