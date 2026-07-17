from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from app.market.enums import Asset, Interval, MarketType, Provider
from app.paper.calculation import (
    PaperExecutionError,
    calculate_no_action,
    calculate_open_position,
    calculate_sell_all,
)
from app.paper.config import VERSION_1, quantize_financial
from app.paper.models import NoTrade, PaperPosition, PaperPositionStatus


def test_open_position_uses_adverse_slippage_and_additional_fee() -> None:
    result = calculate_open_position(
        cash=Decimal("10000"),
        total_equity=Decimal("10000"),
        observed_price=Decimal("100"),
        target_percentage=Decimal("15"),
        configuration=VERSION_1,
    )
    assert result.execution_price == Decimal("100.050000000000000000")
    assert result.quantity == quantize_financial(Decimal("1500") / Decimal("100.05"))
    assert result.fee_amount == quantize_financial(
        result.executed_notional * Decimal("0.001")
    )
    assert result.resulting_cash == quantize_financial(
        Decimal("10000") - result.executed_notional - result.fee_amount
    )


def test_open_position_rejects_insufficient_cash() -> None:
    with pytest.raises(PaperExecutionError, match="insufficient"):
        calculate_open_position(
            cash=Decimal("1"),
            total_equity=Decimal("10000"),
            observed_price=Decimal("100"),
            target_percentage=Decimal("15"),
            configuration=VERSION_1,
        )


def test_sell_all_uses_full_quantity_adverse_price_fee_and_net_pnl() -> None:
    position = PaperPosition(
        uuid4(),
        uuid4(),
        Asset.BTC_USDT,
        Provider.BINANCE,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
        PaperPositionStatus.OPEN,
        Decimal("2"),
        Decimal("100"),
        Decimal("200"),
        Decimal("0.2"),
        datetime.now(UTC),
        None,
        None,
        None,
        None,
    )
    result = calculate_sell_all(
        cash=Decimal("800"),
        position=position,
        observed_price=Decimal("120"),
        configuration=VERSION_1,
    )
    assert result.execution_price == Decimal("119.940000000000000000")
    assert result.quantity == Decimal("2")
    assert result.executed_notional == Decimal("239.880000000000000000")
    assert result.fee_amount == Decimal("0.239880000000000000")
    assert result.gross_realized_pnl == Decimal("39.880000000000000000")
    assert result.net_realized_pnl == Decimal("39.440120000000000000")


def test_no_action_is_explicit() -> None:
    assert calculate_no_action() == NoTrade()
