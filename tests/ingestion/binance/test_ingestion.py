from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.binance import ingestion
from app.ingestion.binance.client import BinanceClient
from app.ingestion.binance.errors import BinanceTransportError
from app.ingestion.binance.ingestion import IngestionResult
from app.ingestion.binance.normalization_errors import BinanceKlineStructureError
from app.ingestion.binance.types import BinanceKline
from app.market.enums import Asset, Interval
from tests.market.factories import make_candle

pytestmark = pytest.mark.asyncio

INGESTED_AT = datetime(2026, 7, 15, 18, 0, tzinfo=UTC)


def client_mock_for(rows: list[BinanceKline]) -> tuple[BinanceClient, AsyncMock]:
    get_klines = AsyncMock(return_value=rows)
    client_mock = Mock(spec=BinanceClient)
    client_mock.get_klines = get_klines
    return cast(BinanceClient, client_mock), get_klines


async def test_ingest_klines_runs_complete_pipeline_once() -> None:
    rows: list[BinanceKline] = [["first"], ["second"]]
    client, get_klines = client_mock_for(rows)
    session_mock = AsyncMock(spec=AsyncSession)
    session = cast(AsyncSession, session_mock)
    candles = [make_candle(), make_candle(asset=Asset.ETH_USDT)]

    with (
        patch.object(ingestion, "normalize_kline", side_effect=candles) as normalize,
        patch.object(
            ingestion,
            "insert_market_candles",
            new_callable=AsyncMock,
            return_value=1,
        ) as persist,
    ):
        result = await ingestion.ingest_klines(
            session=session,
            client=client,
            asset=Asset.BTC_USDT,
            interval=Interval.FIVE_MINUTES,
            start_time=1000,
            end_time=2000,
            limit=2,
            ingested_at=INGESTED_AT,
        )

    get_klines.assert_awaited_once_with(
        asset=Asset.BTC_USDT,
        interval=Interval.FIVE_MINUTES,
        start_time=1000,
        end_time=2000,
        limit=2,
    )
    assert normalize.call_count == 2
    normalize.assert_any_call(
        raw_kline=rows[0],
        asset=Asset.BTC_USDT,
        interval=Interval.FIVE_MINUTES,
        ingested_at=INGESTED_AT,
    )
    normalize.assert_any_call(
        raw_kline=rows[1],
        asset=Asset.BTC_USDT,
        interval=Interval.FIVE_MINUTES,
        ingested_at=INGESTED_AT,
    )
    persist.assert_awaited_once_with(session, candles)
    session_mock.commit.assert_not_awaited()
    session_mock.rollback.assert_not_awaited()
    assert result == IngestionResult(received=2, normalized=2, inserted=1, duplicates=1)


async def test_ingest_klines_empty_response_skips_normalization_and_database() -> None:
    client, get_klines = client_mock_for([])
    session_mock = AsyncMock(spec=AsyncSession)

    with (
        patch.object(ingestion, "normalize_kline") as normalize,
        patch.object(
            ingestion, "insert_market_candles", new_callable=AsyncMock
        ) as persist,
    ):
        result = await ingestion.ingest_klines(
            session=cast(AsyncSession, session_mock),
            client=client,
            asset=Asset.BTC_USDT,
            interval=Interval.FIVE_MINUTES,
            ingested_at=INGESTED_AT,
        )

    get_klines.assert_awaited_once()
    normalize.assert_not_called()
    persist.assert_not_awaited()
    session_mock.commit.assert_not_awaited()
    assert result == IngestionResult(0, 0, 0, 0)


async def test_ingest_klines_propagates_client_error() -> None:
    client, get_klines = client_mock_for([])
    get_klines.side_effect = BinanceTransportError("unavailable")

    with pytest.raises(BinanceTransportError):
        await ingestion.ingest_klines(
            session=cast(AsyncSession, AsyncMock(spec=AsyncSession)),
            client=client,
            asset=Asset.BTC_USDT,
            interval=Interval.FIVE_MINUTES,
            ingested_at=INGESTED_AT,
        )


async def test_ingest_klines_propagates_normalization_error() -> None:
    client, _ = client_mock_for([["invalid"]])
    persist = AsyncMock()

    with (
        patch.object(
            ingestion,
            "normalize_kline",
            side_effect=BinanceKlineStructureError("invalid"),
        ),
        patch.object(ingestion, "insert_market_candles", persist),
        pytest.raises(BinanceKlineStructureError),
    ):
        await ingestion.ingest_klines(
            session=cast(AsyncSession, AsyncMock(spec=AsyncSession)),
            client=client,
            asset=Asset.BTC_USDT,
            interval=Interval.FIVE_MINUTES,
            ingested_at=INGESTED_AT,
        )

    persist.assert_not_awaited()


async def test_ingest_klines_propagates_persistence_error() -> None:
    client, _ = client_mock_for([["valid"]])

    with (
        patch.object(ingestion, "normalize_kline", return_value=make_candle()),
        patch.object(
            ingestion,
            "insert_market_candles",
            new_callable=AsyncMock,
            side_effect=RuntimeError("database failed"),
        ),
        pytest.raises(RuntimeError, match="database failed"),
    ):
        await ingestion.ingest_klines(
            session=cast(AsyncSession, AsyncMock(spec=AsyncSession)),
            client=client,
            asset=Asset.BTC_USDT,
            interval=Interval.FIVE_MINUTES,
            ingested_at=INGESTED_AT,
        )


async def test_ingestion_result_is_immutable_and_enforces_invariants() -> None:
    result = IngestionResult(received=1, normalized=1, inserted=1, duplicates=0)

    with pytest.raises(FrozenInstanceError):
        result.received = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="received"):
        IngestionResult(received=2, normalized=1, inserted=1, duplicates=0)
    with pytest.raises(ValueError, match="inserted"):
        IngestionResult(received=1, normalized=1, inserted=0, duplicates=0)
