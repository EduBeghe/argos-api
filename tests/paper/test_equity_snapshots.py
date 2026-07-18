from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from app.market.enums import Asset, Interval, MarketType, Provider
from app.paper.config import quantize_financial
from app.paper.equity_snapshots import prepare_snapshot_values
from app.paper.portfolio import PaperPortfolioValuation, ValuedPosition

NOW = datetime(2026, 7, 18, tzinfo=UTC)


def position(*, asset: Asset, quantity: Decimal, mark_price: Decimal) -> ValuedPosition:
    cost_basis = Decimal("10.0000000000000000004")
    market_value = quantity * mark_price
    pnl = market_value - cost_basis
    return ValuedPosition(
        position_id=uuid4(),
        asset=asset,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        quantity=quantity,
        average_entry_price=Decimal("9.9999999999999999996"),
        cost_basis=cost_basis,
        mark_price=mark_price,
        market_value=market_value,
        unrealized_pnl=pnl,
        unrealized_return=pnl / cost_basis,
        mark_candle_open_time=NOW,
    )


def test_prepared_parent_is_derived_from_quantized_children() -> None:
    valuation = PaperPortfolioValuation(
        run_id=uuid4(),
        status="active",
        account_currency="USDT",
        initial_cash=Decimal("100.0000000000000000004"),
        current_cash=Decimal("80.0000000000000000006"),
        positions_market_value=Decimal("999"),
        equity=Decimal("999"),
        total_pnl=Decimal("999"),
        total_return=Decimal("999"),
        valued_at=NOW,
        positions=(
            position(
                asset=Asset.BTC_USDT,
                quantity=Decimal("1.0000000000000000004"),
                mark_price=Decimal("10.0000000000000000006"),
            ),
            position(
                asset=Asset.ETH_USDT,
                quantity=Decimal("1.0000000000000000006"),
                mark_price=Decimal("20.0000000000000000004"),
            ),
        ),
    )
    prepared = prepare_snapshot_values(valuation)
    assert prepared.positions_market_value == sum(
        (item.market_value for item in prepared.positions), start=Decimal(0)
    )
    assert prepared.equity == prepared.current_cash + prepared.positions_market_value
    assert prepared.total_pnl == prepared.equity - prepared.initial_cash
    assert prepared.total_return == quantize_financial(
        prepared.total_pnl / prepared.initial_cash
    )
    assert all(
        item.market_value == quantize_financial(item.market_value)
        for item in prepared.positions
    )
    with pytest.raises(FrozenInstanceError):
        prepared.equity = Decimal(0)  # type: ignore[misc]
