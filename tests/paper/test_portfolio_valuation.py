from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.market.enums import Asset, Interval, MarketType, Provider
from app.paper.persistence import PaperPositionRecord, PaperTradingRunRecord
from app.paper.portfolio import (
    PaperPortfolioValuationError,
    PositionMark,
    calculate_portfolio_valuation,
    value_paper_run,
)

NOW = datetime(2026, 7, 18, tzinfo=UTC)


def run(*, initial_cash: Decimal = Decimal("1000")) -> PaperTradingRunRecord:
    return PaperTradingRunRecord(
        id=uuid4(),
        status="active",
        account_currency="USDT",
        initial_cash=initial_cash,
        current_cash=Decimal("400"),
        configuration_version=1,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        started_at=NOW,
        stopped_at=None,
        stop_reason=None,
        predecessor_run_id=None,
    )


def mark(
    *,
    asset: Asset = Asset.BTC_USDT,
    quantity: Decimal = Decimal("2"),
    entry_notional: Decimal = Decimal("500"),
    entry_fees: Decimal = Decimal("5"),
    price: Decimal = Decimal("300"),
) -> PositionMark:
    return PositionMark(
        position_id=uuid4(),
        asset=asset,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        quantity=quantity,
        average_entry_price=Decimal("250"),
        total_entry_notional=entry_notional,
        accumulated_entry_fees=entry_fees,
        mark_price=price,
        mark_candle_open_time=NOW,
    )


def test_empty_portfolio_equity_is_current_cash() -> None:
    valuation = calculate_portfolio_valuation(run=run(), marks=[], valued_at=NOW)
    assert valuation.positions_market_value == Decimal(0)
    assert valuation.equity == Decimal("400")


@pytest.mark.parametrize(
    ("price", "expected_pnl"),
    [(Decimal("300"), Decimal("95")), (Decimal("200"), Decimal("-105"))],
)
def test_position_gain_loss_and_entry_fees(
    price: Decimal, expected_pnl: Decimal
) -> None:
    valuation = calculate_portfolio_valuation(
        run=run(), marks=[mark(price=price)], valued_at=NOW
    )
    position = valuation.positions[0]
    assert position.cost_basis == Decimal("505")
    assert position.unrealized_pnl == expected_pnl
    assert position.unrealized_return == expected_pnl / Decimal("505")


def test_multiple_generic_positions_use_decimal_only() -> None:
    valuation = calculate_portfolio_valuation(
        run=run(),
        marks=[
            mark(asset=Asset.BTC_USDT, quantity=Decimal("0.1"), price=Decimal("100")),
            mark(
                asset=Asset.ETH_USDT,
                quantity=Decimal("0.2"),
                entry_notional=Decimal("20"),
                entry_fees=Decimal("0.1"),
                price=Decimal("120"),
            ),
        ],
        valued_at=NOW,
    )
    assert valuation.positions_market_value == Decimal("34.0")
    assert valuation.equity == Decimal("434.0")
    assert all(
        isinstance(position.market_value, Decimal) for position in valuation.positions
    )


def test_zero_initial_cash_omits_total_return() -> None:
    valuation = calculate_portfolio_valuation(
        run=run(initial_cash=Decimal(0)), marks=[], valued_at=NOW
    )
    assert valuation.total_return is None


@pytest.mark.asyncio
async def test_missing_mark_fails_explicitly() -> None:
    position = PaperPositionRecord(
        id=uuid4(),
        run_id=uuid4(),
        asset=Asset.BTC_USDT.value,
        provider=Provider.BINANCE.value,
        market_type=MarketType.SPOT.value,
        interval=Interval.FIVE_MINUTES.value,
        status="open",
        quantity=Decimal("1"),
        average_entry_price=Decimal("100"),
        total_entry_notional=Decimal("100"),
        accumulated_entry_fees=Decimal("0.1"),
        opened_at=NOW,
        closed_at=None,
        exit_price=None,
        gross_realized_pnl=None,
        net_realized_pnl=None,
    )
    result = MagicMock()
    result.all.return_value = [(position, None, None)]
    session = AsyncMock()
    session.execute.return_value = result
    with pytest.raises(PaperPortfolioValuationError, match="mark candle not found"):
        await value_paper_run(session=session, run=run(), valued_at=NOW)
    session.execute.assert_awaited_once()
