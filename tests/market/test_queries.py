from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.queries import get_market_status_aggregates, list_market_candles
from app.market.schemas import CandleOrder

pytestmark = pytest.mark.asyncio


async def test_candle_query_scopes_results_to_binance_spot() -> None:
    """Other provenance cannot be inserted under current database constraints."""

    session_mock = AsyncMock(spec=AsyncSession)
    scalar_result = MagicMock()
    scalar_result.all.return_value = []
    session_mock.scalars.return_value = scalar_result

    await list_market_candles(
        session=cast(AsyncSession, session_mock),
        asset=Asset.BTC_USDT,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        start_time=None,
        end_time=None,
        limit=100,
        order=CandleOrder.DESC,
    )

    statement = session_mock.scalars.await_args.args[0]
    statement_text = str(statement)
    values = set(statement.compile().params.values())
    assert "market_candles.provider" in statement_text
    assert "market_candles.market_type" in statement_text
    assert Provider.BINANCE.value in values
    assert MarketType.SPOT.value in values


async def test_grouped_status_query_scopes_results_to_binance_spot() -> None:
    """Current constraints permit only Binance Spot persistence fixtures."""

    session_mock = AsyncMock(spec=AsyncSession)
    tuple_result = MagicMock()
    tuple_result.tuples.return_value.all.return_value = []
    session_mock.execute.return_value = tuple_result

    await get_market_status_aggregates(
        session=cast(AsyncSession, session_mock),
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
    )

    statement = session_mock.execute.await_args.args[0]
    statement_text = str(statement)
    values = set(statement.compile().params.values())
    assert "market_candles.provider" in statement_text
    assert "market_candles.market_type" in statement_text
    assert Provider.BINANCE.value in values
    assert MarketType.SPOT.value in values
