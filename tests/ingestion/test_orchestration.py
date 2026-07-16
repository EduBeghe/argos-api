import asyncio
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion import orchestration
from app.ingestion.binance.client import BinanceClient
from app.ingestion.binance.ingestion import IngestionResult
from app.ingestion.orchestration import AssetIngestionResult
from app.market.enums import Asset, Interval

pytestmark = pytest.mark.asyncio

NOW = datetime(2026, 7, 15, 12, 7, 30, tzinfo=UTC)
START_TIME = 1_784_116_500_000
END_TIME = 1_784_117_099_999


def client_mock() -> BinanceClient:
    return cast(BinanceClient, Mock(spec=BinanceClient))


async def test_orchestration_ingests_all_assets_atomically_in_order() -> None:
    session_mock = AsyncMock(spec=AsyncSession)
    session = cast(AsyncSession, session_mock)
    client = client_mock()
    btc = IngestionResult(2, 2, 1, 1)
    eth = IngestionResult(2, 2, 2, 0)

    with patch.object(
        orchestration,
        "ingest_klines",
        new_callable=AsyncMock,
        side_effect=[btc, eth],
    ) as ingest:
        result = await orchestration.ingest_latest_closed_klines(
            session=session,
            client=client,
            assets=[Asset.BTC_USDT, Asset.ETH_USDT],
            interval=Interval.FIVE_MINUTES,
            now=NOW,
            limit=2,
        )

    assert ingest.await_args_list == [
        call(
            session=session,
            client=client,
            asset=Asset.BTC_USDT,
            interval=Interval.FIVE_MINUTES,
            start_time=START_TIME,
            end_time=END_TIME,
            limit=2,
            ingested_at=NOW,
        ),
        call(
            session=session,
            client=client,
            asset=Asset.ETH_USDT,
            interval=Interval.FIVE_MINUTES,
            start_time=START_TIME,
            end_time=END_TIME,
            limit=2,
            ingested_at=NOW,
        ),
    ]
    session_mock.commit.assert_awaited_once_with()
    session_mock.rollback.assert_not_awaited()
    assert result.ingested_at is NOW
    assert result.start_time == START_TIME
    assert result.end_time == END_TIME
    assert result.assets == (
        AssetIngestionResult(Asset.BTC_USDT, btc),
        AssetIngestionResult(Asset.ETH_USDT, eth),
    )
    assert (result.received, result.inserted, result.duplicates) == (4, 3, 1)


@pytest.mark.parametrize(
    ("assets", "message"),
    [
        ([], "empty"),
        ([Asset.BTC_USDT, Asset.BTC_USDT], "duplicates"),
        ([cast(Asset, "unsupported")], "unsupported"),
    ],
)
async def test_orchestration_rejects_invalid_assets_before_work(
    assets: list[Asset], message: str
) -> None:
    session_mock = AsyncMock(spec=AsyncSession)
    ingest = AsyncMock()

    with (
        patch.object(orchestration, "ingest_klines", ingest),
        pytest.raises(ValueError, match=message),
    ):
        await orchestration.ingest_latest_closed_klines(
            session=cast(AsyncSession, session_mock),
            client=client_mock(),
            assets=assets,
            interval=Interval.FIVE_MINUTES,
            now=NOW,
            limit=1,
        )

    ingest.assert_not_awaited()
    session_mock.commit.assert_not_awaited()
    session_mock.rollback.assert_not_awaited()


@pytest.mark.parametrize(
    ("interval", "now", "limit"),
    [
        (cast(Interval, "1m"), NOW, 1),
        (Interval.FIVE_MINUTES, datetime(2026, 7, 15, 12, 7, 30), 1),
        (Interval.FIVE_MINUTES, NOW, 0),
        (Interval.FIVE_MINUTES, NOW, 1001),
    ],
)
async def test_orchestration_rejects_invalid_window_before_work(
    interval: Interval, now: datetime, limit: int
) -> None:
    session_mock = AsyncMock(spec=AsyncSession)
    ingest = AsyncMock()

    with (
        patch.object(orchestration, "ingest_klines", ingest),
        pytest.raises(ValueError),
    ):
        await orchestration.ingest_latest_closed_klines(
            session=cast(AsyncSession, session_mock),
            client=client_mock(),
            assets=[Asset.BTC_USDT],
            interval=interval,
            now=now,
            limit=limit,
        )

    ingest.assert_not_awaited()
    session_mock.commit.assert_not_awaited()
    session_mock.rollback.assert_not_awaited()


@pytest.mark.parametrize("failure_index", [0, 1])
async def test_ingestion_failure_rolls_back_and_preserves_original_error(
    failure_index: int,
) -> None:
    session_mock = AsyncMock(spec=AsyncSession)
    original = RuntimeError("ingestion failed")
    successful = IngestionResult(1, 1, 1, 0)
    side_effects: list[IngestionResult | BaseException] = (
        [original] if failure_index == 0 else [successful, original]
    )

    with (
        patch.object(
            orchestration,
            "ingest_klines",
            new_callable=AsyncMock,
            side_effect=side_effects,
        ) as ingest,
        pytest.raises(RuntimeError) as captured,
    ):
        await orchestration.ingest_latest_closed_klines(
            session=cast(AsyncSession, session_mock),
            client=client_mock(),
            assets=[Asset.BTC_USDT, Asset.ETH_USDT],
            interval=Interval.FIVE_MINUTES,
            now=NOW,
            limit=1,
        )

    assert captured.value is original
    assert ingest.await_count == failure_index + 1
    session_mock.rollback.assert_awaited_once_with()
    session_mock.commit.assert_not_awaited()


async def test_commit_failure_rolls_back_and_preserves_original_error() -> None:
    session_mock = AsyncMock(spec=AsyncSession)
    original = RuntimeError("commit failed")
    session_mock.commit.side_effect = original

    with (
        patch.object(
            orchestration,
            "ingest_klines",
            new_callable=AsyncMock,
            return_value=IngestionResult(1, 1, 1, 0),
        ),
        pytest.raises(RuntimeError) as captured,
    ):
        await orchestration.ingest_latest_closed_klines(
            session=cast(AsyncSession, session_mock),
            client=client_mock(),
            assets=[Asset.BTC_USDT],
            interval=Interval.FIVE_MINUTES,
            now=NOW,
            limit=1,
        )

    assert captured.value is original
    session_mock.commit.assert_awaited_once_with()
    session_mock.rollback.assert_awaited_once_with()


async def test_rollback_failure_is_noted_on_original_error() -> None:
    session_mock = AsyncMock(spec=AsyncSession)
    original = RuntimeError("ingestion failed")
    rollback_error = RuntimeError("rollback failed")
    session_mock.rollback.side_effect = rollback_error

    with (
        patch.object(
            orchestration,
            "ingest_klines",
            new_callable=AsyncMock,
            side_effect=original,
        ),
        pytest.raises(RuntimeError) as captured,
    ):
        await orchestration.ingest_latest_closed_klines(
            session=cast(AsyncSession, session_mock),
            client=client_mock(),
            assets=[Asset.BTC_USDT],
            interval=Interval.FIVE_MINUTES,
            now=NOW,
            limit=1,
        )

    assert captured.value is original
    assert original.__notes__ == [f"Rollback also failed: {rollback_error!r}"]


async def test_async_cancellation_is_not_intercepted() -> None:
    session_mock = AsyncMock(spec=AsyncSession)
    cancellation = asyncio.CancelledError()

    with (
        patch.object(
            orchestration,
            "ingest_klines",
            new_callable=AsyncMock,
            side_effect=cancellation,
        ),
        pytest.raises(asyncio.CancelledError) as captured,
    ):
        await orchestration.ingest_latest_closed_klines(
            session=cast(AsyncSession, session_mock),
            client=client_mock(),
            assets=[Asset.BTC_USDT],
            interval=Interval.FIVE_MINUTES,
            now=NOW,
            limit=1,
        )

    assert captured.value is cancellation
    session_mock.commit.assert_not_awaited()
    session_mock.rollback.assert_not_awaited()
