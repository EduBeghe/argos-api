from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.models import FeatureName
from app.market.enums import Asset, Interval, MarketType, Provider
from app.scores.calculation import calculate_market_scores

NOW = datetime(2026, 7, 16, tzinfo=UTC)
CANDLE_TIME = datetime(2026, 7, 15, tzinfo=UTC)


@pytest.mark.asyncio
async def test_one_ordered_read_one_bulk_insert_and_no_transaction_management() -> None:
    session_mock = AsyncMock(spec=AsyncSession)
    values = {
        FeatureName.EMA_20: Decimal(102),
        FeatureName.EMA_50: Decimal(100),
        FeatureName.RSI_14: Decimal(60),
        FeatureName.ATR_14: Decimal(2),
        FeatureName.RETURN_1: Decimal("0.001"),
        FeatureName.RETURN_12: Decimal("0.005"),
        FeatureName.RELATIVE_VOLUME_20: Decimal(1),
    }
    query_result = MagicMock()
    query_result.tuples.return_value.all.return_value = [
        (CANDLE_TIME, Decimal(100), name.value, value) for name, value in values.items()
    ]
    session_mock.execute.return_value = query_result
    with patch(
        "app.scores.calculation.insert_market_scores", AsyncMock(return_value=1)
    ) as insert:
        result = await calculate_market_scores(
            session=cast(AsyncSession, session_mock),
            asset=Asset.BTC_USDT,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=Interval.FIVE_MINUTES,
            calculated_at=NOW,
        )
    session_mock.execute.assert_awaited_once()
    insert.assert_awaited_once()
    assert insert.await_args is not None
    inserted = insert.await_args.args[1]
    assert len(inserted) == 1 and inserted[0].candle_open_time == CANDLE_TIME
    assert (result.scores_calculated, result.scores_inserted, result.duplicates) == (
        1,
        1,
        0,
    )
    session_mock.commit.assert_not_awaited()
    session_mock.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_incomplete_same_candle_features_are_skipped() -> None:
    session = AsyncMock()
    query_result = MagicMock()
    query_result.tuples.return_value.all.return_value = [
        (CANDLE_TIME, Decimal(100), FeatureName.EMA_20.value, Decimal(100))
    ]
    session.execute.return_value = query_result
    with patch("app.scores.calculation.insert_market_scores", AsyncMock()) as insert:
        result = await calculate_market_scores(
            session=session,
            asset=Asset.ETH_USDT,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=Interval.FIVE_MINUTES,
            calculated_at=NOW,
        )
    insert.assert_not_awaited()
    assert result.candidate_candles_read == 1
    assert result.scores_calculated == 0
