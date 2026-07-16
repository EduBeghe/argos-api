from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.calculation import calculate_ema_20_features
from app.market.enums import Asset, Interval, MarketType, Provider

pytestmark = pytest.mark.asyncio


async def test_calculation_reads_once_bulk_inserts_and_never_commits() -> None:
    session_mock = AsyncMock(spec=AsyncSession)
    rows = [
        (
            datetime(2026, 7, 15, tzinfo=UTC) + timedelta(minutes=5 * index),
            Decimal(index + 2),
            Decimal(index),
            Decimal(index + 1),
            Decimal(100 + index),
        )
        for index in range(21)
    ]
    result = MagicMock()
    result.tuples.return_value.all.return_value = rows
    session_mock.execute.return_value = result

    with patch(
        "app.features.calculation.insert_market_features",
        AsyncMock(return_value=["ema_20"]),
    ) as insert:
        calculated = await calculate_ema_20_features(
            session=cast(AsyncSession, session_mock),
            asset=Asset.BTC_USDT,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=Interval.FIVE_MINUTES,
            calculated_at=datetime(2026, 7, 16, tzinfo=UTC),
        )

    session_mock.execute.assert_awaited_once()
    insert.assert_awaited_once()
    session_mock.commit.assert_not_awaited()
    session_mock.rollback.assert_not_awaited()
    assert calculated.candle_count_read == 21
    assert calculated.values_calculated == 47
    assert (calculated.values_inserted, calculated.duplicates) == (1, 46)


async def test_insufficient_history_skips_bulk_insert() -> None:
    session_mock = AsyncMock(spec=AsyncSession)
    result = MagicMock()
    result.tuples.return_value.all.return_value = []
    session_mock.execute.return_value = result

    with patch(
        "app.features.calculation.insert_market_features", AsyncMock(return_value=[])
    ) as insert:
        calculated = await calculate_ema_20_features(
            session=cast(AsyncSession, session_mock),
            asset=Asset.BTC_USDT,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=Interval.FIVE_MINUTES,
            calculated_at=datetime(2026, 7, 16, tzinfo=UTC),
        )

    insert.assert_not_awaited()
    assert calculated.values_calculated == 0
