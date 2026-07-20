from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.calculation import (
    FEATURE_VERSION,
    LIVE_FEATURE_CONTEXT_CANDLES,
    calculate_ema_20_features,
)
from app.features.models import FeatureName, MarketFeature
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

    with (
        patch(
            "app.features.calculation.get_existing_feature_identities",
            AsyncMock(return_value=set()),
        ),
        patch(
            "app.features.calculation.insert_market_features",
            AsyncMock(return_value=["ema_20"]),
        ) as insert,
    ):
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


async def test_existing_identities_are_filtered_and_historical_gap_is_rebuilt() -> None:
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
    captured: list[MarketFeature] = []

    async def existing_identities(
        **_: object,
    ) -> set[tuple[datetime, FeatureName, int]]:
        if not captured:
            return set()
        missing = captured[0]
        return {
            (value.candle_open_time, value.feature_name, value.feature_version)
            for value in captured
            if value is not missing
        }

    async def insert_missing(
        _session: AsyncSession, values: list[MarketFeature]
    ) -> list[FeatureName]:
        if not captured:
            captured.extend(values)
        return [value.feature_name for value in values]

    with (
        patch(
            "app.features.calculation.get_existing_feature_identities",
            side_effect=existing_identities,
        ),
        patch(
            "app.features.calculation.insert_market_features",
            side_effect=insert_missing,
        ) as insert,
    ):
        first = await calculate_ema_20_features(
            session=cast(AsyncSession, session_mock),
            asset=Asset.BTC_USDT,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=Interval.FIVE_MINUTES,
            calculated_at=datetime(2026, 7, 16, tzinfo=UTC),
        )
        rebuilt = await calculate_ema_20_features(
            session=cast(AsyncSession, session_mock),
            asset=Asset.BTC_USDT,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=Interval.FIVE_MINUTES,
            calculated_at=datetime(2026, 7, 16, tzinfo=UTC),
        )

    assert first.values_inserted == first.values_calculated
    assert rebuilt.values_inserted == 1
    assert rebuilt.duplicates == rebuilt.values_calculated - 1
    rebuilt_value = insert.await_args_list[1].args[1][0]
    assert (
        rebuilt_value.candle_open_time,
        rebuilt_value.feature_name,
        rebuilt_value.feature_version,
        rebuilt_value.value,
    ) == (
        captured[0].candle_open_time,
        captured[0].feature_name,
        captured[0].feature_version,
        captured[0].value,
    )
    assert captured[0].feature_version == FEATURE_VERSION
    session_mock.commit.assert_not_awaited()
    session_mock.rollback.assert_not_awaited()


async def test_up_to_date_run_performs_no_insert() -> None:
    session_mock = AsyncMock(spec=AsyncSession)
    rows = [
        (
            datetime(2026, 7, 15, tzinfo=UTC) + timedelta(minutes=5 * index),
            Decimal(index + 2),
            Decimal(index),
            Decimal(index + 1),
            Decimal(100 + index),
        )
        for index in range(20)
    ]
    result = MagicMock()
    result.tuples.return_value.all.return_value = rows
    session_mock.execute.return_value = result

    with (
        patch(
            "app.features.calculation.get_existing_feature_identities",
            AsyncMock(
                return_value={
                    (rows[19][0], FeatureName.EMA_20, FEATURE_VERSION),
                    *(
                        (rows[index][0], FeatureName.RSI_14, FEATURE_VERSION)
                        for index in range(14, 20)
                    ),
                    *(
                        (rows[index][0], FeatureName.ATR_14, FEATURE_VERSION)
                        for index in range(13, 20)
                    ),
                    *(
                        (rows[index][0], FeatureName.RETURN_1, FEATURE_VERSION)
                        for index in range(1, 20)
                    ),
                    *(
                        (rows[index][0], FeatureName.RETURN_12, FEATURE_VERSION)
                        for index in range(12, 20)
                    ),
                }
            ),
        ),
        patch(
            "app.features.calculation.insert_market_features",
            AsyncMock(return_value=[]),
        ) as insert,
    ):
        calculated = await calculate_ema_20_features(
            session=cast(AsyncSession, session_mock),
            asset=Asset.BTC_USDT,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=Interval.FIVE_MINUTES,
            calculated_at=datetime(2026, 7, 16, tzinfo=UTC),
        )

    insert.assert_not_awaited()
    assert calculated.values_inserted == 0
    assert calculated.duplicates == calculated.values_calculated


async def test_live_window_bounds_context_and_excludes_historical_targets() -> None:
    session = AsyncMock(spec=AsyncSession)
    end = datetime(2026, 7, 16, tzinfo=UTC)
    start = end - timedelta(minutes=10)
    ascending = [
        (
            end - timedelta(minutes=5 * (LIVE_FEATURE_CONTEXT_CANDLES - index)),
            Decimal(index + 2),
            Decimal(index),
            Decimal(index + 1),
            Decimal(100 + index),
        )
        for index in range(LIVE_FEATURE_CONTEXT_CANDLES)
    ]
    query_result = MagicMock()
    query_result.tuples.return_value.all.return_value = list(reversed(ascending))
    session.execute.return_value = query_result
    captured: list[MarketFeature] = []

    async def insert(_: AsyncSession, values: list[MarketFeature]) -> list[FeatureName]:
        captured.extend(values)
        return [value.feature_name for value in values]

    with (
        patch(
            "app.features.calculation.get_existing_feature_identities",
            AsyncMock(return_value=set()),
        ) as existing,
        patch("app.features.calculation.insert_market_features", side_effect=insert),
    ):
        result = await calculate_ema_20_features(
            session=cast(AsyncSession, session),
            asset=Asset.BTC_USDT,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=Interval.FIVE_MINUTES,
            calculated_at=end,
            target_start_time=start,
            target_end_time=end,
            context_limit=LIVE_FEATURE_CONTEXT_CANDLES,
        )

    statement = session.execute.await_args.args[0]
    assert statement._limit_clause is not None
    assert result.candle_count_read == LIVE_FEATURE_CONTEXT_CANDLES
    assert result.target_candle_count == 2
    assert captured
    assert all(start <= value.candle_open_time < end for value in captured)
    existing.assert_awaited_once_with(
        session=session,
        asset=Asset.BTC_USDT,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        feature_version=FEATURE_VERSION,
        start_time=start,
        end_time=end,
    )
