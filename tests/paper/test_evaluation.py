from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from typing import cast

from app.paper.config import FINANCIAL_QUANTUM, quantize_financial
from app.paper.evaluation import (
    BenchmarkCandles,
    CandleEvidence,
    ClosedPositionEvidence,
    DecisionEvidence,
    MarketIdentity,
    OpenPositionEvidence,
    TradeEvidence,
    calculate_benchmark,
    closed_position_statistics,
    count_values,
    estimated_slippage_cost,
    instrument_evaluations,
    strategy_evaluation,
)
from app.paper.performance import PaperRunPerformance

BTC = MarketIdentity("BTC/USDT", "binance", "spot", "5m")
BTC_OTHER = MarketIdentity("BTC/USDT", "other", "spot", "15m")
NOW = datetime(2026, 7, 19, 12, tzinfo=UTC)


def test_closed_position_statistics_cover_wins_losses_and_breakeven_exactly() -> None:
    positions = tuple(
        ClosedPositionEvidence(BTC, value, value)
        for value in map(Decimal, ("10", "20", "-5", "-15", "0"))
    )
    result = closed_position_statistics(positions)
    assert result == {
        "closed_position_count": 5,
        "winning_position_count": 2,
        "losing_position_count": 2,
        "breakeven_position_count": 1,
        "win_rate": Decimal("0.400000000000000000"),
        "average_win": Decimal("15.000000000000000000"),
        "average_loss": Decimal("-10.000000000000000000"),
        "profit_factor": Decimal("1.500000000000000000"),
    }


def test_nullable_closed_position_metrics_do_not_use_sentinels() -> None:
    empty = closed_position_statistics(())
    assert empty["win_rate"] is None
    assert empty["average_win"] is None
    assert empty["average_loss"] is None
    assert empty["profit_factor"] is None
    only_win = closed_position_statistics(
        (ClosedPositionEvidence(BTC, Decimal("1"), Decimal("1")),)
    )
    assert only_win["average_win"] == Decimal("1.000000000000000000")
    assert only_win["average_loss"] is None
    assert only_win["profit_factor"] is None


def test_buy_and_sell_slippage_uses_persisted_execution_evidence() -> None:
    buy = trade("buy", observed="10", execution="11", quantity="2")
    sell = trade("sell", observed="10", execution="9.5", quantity="3")
    favorable_buy = trade("buy", observed="10", execution="9", quantity="1")
    assert estimated_slippage_cost(buy) == Decimal("2.000000000000000000")
    assert estimated_slippage_cost(sell) == Decimal("1.500000000000000000")
    assert estimated_slippage_cost(favorable_buy) == Decimal(0).quantize(
        FINANCIAL_QUANTUM
    )


def test_strategy_uses_persisted_fees_and_does_not_fabricate_trades() -> None:
    decisions = (
        DecisionEvidence(BTC, "open_position"),
        DecisionEvidence(BTC, "no_action"),
    )
    trades = (trade("buy", fee="0.123456789012345678"),)
    result = strategy_evaluation(
        performance=performance(),
        decisions=decisions,
        trades=trades,
        closed_positions=(),
        open_positions=(OpenPositionEvidence(BTC, Decimal("2.5")),),
    )
    assert result["decision_count"] == 2
    assert result["decision_counts_by_action"] == {
        "no_action": 1,
        "open_position": 1,
        "sell_all": 0,
    }
    assert result["trade_count"] == 1
    assert result["trade_counts_by_side"] == {"buy": 1, "sell": 0}
    assert result["fees_paid"] == Decimal("0.123456789012345678")
    assert result["unrealized_pnl"] == Decimal("2.500000000000000000")


def test_unknown_future_count_values_are_preserved() -> None:
    assert count_values(("no_action", "future_action", "future_action")) == {
        "future_action": 2,
        "no_action": 1,
    }


def test_complete_market_identities_remain_separate_and_ordered() -> None:
    rows = instrument_evaluations(
        decisions=(
            DecisionEvidence(BTC_OTHER, "no_action"),
            DecisionEvidence(BTC, "open_position"),
        ),
        trades=(),
        closed_positions=(),
        open_positions=(),
    )
    assert [(row["provider"], row["interval"]) for row in rows] == [
        ("binance", "5m"),
        ("other", "15m"),
    ]


def test_equal_weight_benchmark_applies_friction_and_retains_cash_remainder() -> None:
    initial = Decimal("100")
    fee_rate = Decimal("0.001")
    slippage_rate = Decimal("0.0005")
    result = calculate_benchmark(
        initial_equity=initial,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        universe=(BTC,),
        candles={
            BTC: BenchmarkCandles(
                CandleEvidence(NOW, Decimal("10")),
                CandleEvidence(NOW, Decimal("12")),
            )
        },
    )
    execution = quantize_financial(Decimal("10") * (Decimal(1) + slippage_rate))
    expected_quantity = (initial / (execution * (Decimal(1) + fee_rate))).quantize(
        FINANCIAL_QUANTUM, rounding=ROUND_DOWN
    )
    instrument = result["instruments"][0]  # type: ignore[index]
    assert instrument["quantity"] == expected_quantity
    assert result["status"] == "available"
    assert result["entry_fees"] == instrument["entry_fee"]
    assert result["entry_slippage_cost"] == instrument["estimated_entry_slippage_cost"]
    assert result["ending_equity"] == quantize_financial(
        result["ending_asset_value"] + result["remaining_cash"]  # type: ignore[operator]
    )
    assert result["remaining_cash"] >= 0  # type: ignore[operator]


def test_incomplete_benchmark_is_unavailable_without_reweighting() -> None:
    result = calculate_benchmark(
        initial_equity=Decimal("100"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        universe=(BTC, BTC_OTHER),
        candles={
            BTC: BenchmarkCandles(
                CandleEvidence(NOW, Decimal("10")),
                CandleEvidence(NOW, Decimal("11")),
            ),
            BTC_OTHER: BenchmarkCandles(None, None),
        },
    )
    assert result["status"] == "unavailable"
    assert result["unavailable_reason"] == "incomplete_candle_evidence"
    assert result["ending_equity"] is None
    instruments = cast(list[dict[str, object]], result["instruments"])
    assert len(instruments) == 2
    assert all(
        item["allocation"] == Decimal("50.000000000000000000") for item in instruments
    )


def trade(
    side: str,
    *,
    observed: str = "10",
    execution: str = "11",
    quantity: str = "1",
    fee: str = "0",
) -> TradeEvidence:
    return TradeEvidence(
        BTC,
        side,
        Decimal(observed),
        Decimal(execution),
        Decimal(quantity),
        Decimal(fee),
    )


def performance() -> PaperRunPerformance:
    return PaperRunPerformance(
        snapshot_count=1,
        initial_equity=Decimal("100"),
        latest_equity=Decimal("102"),
        absolute_pnl=Decimal("2"),
        total_return=Decimal("0.02"),
        peak_equity=Decimal("102"),
        lowest_equity=Decimal("100"),
        current_drawdown=Decimal("0"),
        max_drawdown=Decimal("0"),
        first_snapshot_boundary=NOW,
        latest_snapshot_boundary=NOW,
        evaluated_from=NOW,
        evaluated_until=NOW,
    )
