from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.models import FeatureName, MarketFeature
from app.features.persistence import FEATURE_INSERT_BATCH_SIZE, insert_market_features
from app.market.enums import Asset, Interval, MarketType, Provider

pytestmark = pytest.mark.asyncio


def make_features(count: int) -> list[MarketFeature]:
    start = datetime(2026, 7, 15, tzinfo=UTC)
    names = tuple(FeatureName)
    return [
        MarketFeature(
            asset=Asset.BTC_USDT,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=Interval.FIVE_MINUTES,
            candle_open_time=start + timedelta(minutes=5 * index),
            feature_name=names[index % len(names)],
            feature_version=1,
            value=Decimal(index),
            calculated_at=start,
        )
        for index in range(count)
    ]


def scalar_result(names: list[str]) -> MagicMock:
    result = MagicMock()
    result.all.return_value = names
    return result


@pytest.mark.parametrize(
    ("count", "expected_batches"),
    [
        (1, 1),
        (FEATURE_INSERT_BATCH_SIZE - 1, 1),
        (FEATURE_INSERT_BATCH_SIZE, 1),
        (FEATURE_INSERT_BATCH_SIZE + 1, 2),
    ],
)
async def test_insert_uses_bounded_conflict_safe_batches(
    count: int, expected_batches: int
) -> None:
    session_mock = AsyncMock(spec=AsyncSession)
    session_mock.scalars.side_effect = [
        scalar_result([FeatureName.EMA_20.value]) for _ in range(expected_batches)
    ]

    returned = await insert_market_features(
        cast(AsyncSession, session_mock), make_features(count)
    )

    assert returned == [FeatureName.EMA_20] * expected_batches
    assert session_mock.scalars.await_count == expected_batches
    for call in session_mock.scalars.await_args_list:
        statement = call.args[0]
        rendered = str(statement)
        assert "ON CONFLICT (" in rendered
        assert "DO NOTHING" in rendered
        assert len(statement.compile().params) <= FEATURE_INSERT_BATCH_SIZE * 10
    session_mock.commit.assert_not_awaited()
    session_mock.rollback.assert_not_awaited()


async def test_empty_insert_does_not_touch_database() -> None:
    session_mock = AsyncMock(spec=AsyncSession)

    assert await insert_market_features(cast(AsyncSession, session_mock), []) == []

    session_mock.scalars.assert_not_awaited()


async def test_later_batch_failure_propagates() -> None:
    session_mock = AsyncMock(spec=AsyncSession)
    failure = RuntimeError("second batch failed")
    session_mock.scalars.side_effect = [scalar_result([]), failure]

    with pytest.raises(RuntimeError) as captured:
        await insert_market_features(
            cast(AsyncSession, session_mock),
            make_features(FEATURE_INSERT_BATCH_SIZE + 1),
        )

    assert captured.value is failure
    session_mock.commit.assert_not_awaited()
    session_mock.rollback.assert_not_awaited()
