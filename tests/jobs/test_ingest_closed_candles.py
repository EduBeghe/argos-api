import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.core.config import Settings
from app.ingestion.binance.ingestion import IngestionResult
from app.ingestion.orchestration import (
    AssetIngestionResult,
    ClosedCandleIngestionResult,
)
from app.jobs import ingest_closed_candles as job
from app.market.enums import Asset, Interval

NOW = datetime(2026, 7, 16, 12, 7, tzinfo=UTC)


def settings() -> Settings:
    return Settings.model_validate(
        {
            "DATABASE_URL": "postgresql+asyncpg://user:secret@database/argos",
            "BINANCE_BASE_URL": "https://binance.test",
            "BINANCE_INGESTION_LIMIT": 18,
        }
    )


def result() -> ClosedCandleIngestionResult:
    btc = IngestionResult(received=2, normalized=2, inserted=2, duplicates=0)
    eth = IngestionResult(received=2, normalized=2, inserted=1, duplicates=1)
    return ClosedCandleIngestionResult(
        ingested_at=NOW,
        start_time=1000,
        end_time=2000,
        assets=(
            AssetIngestionResult(asset=Asset.BTC_USDT, result=btc),
            AssetIngestionResult(asset=Asset.ETH_USDT, result=eth),
        ),
        received=4,
        inserted=3,
        duplicates=1,
    )


@pytest.mark.asyncio
async def test_run_composes_one_ingestion_and_closes_resources() -> None:
    database_engine = MagicMock()
    database_engine.dispose = AsyncMock()
    session = AsyncMock()
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=None)
    session_factory = Mock(return_value=session_context)
    client = MagicMock()
    client_context = MagicMock()
    client_context.__aenter__ = AsyncMock(return_value=client)
    client_context.__aexit__ = AsyncMock(return_value=None)
    clock = Mock(return_value=NOW)
    expected = result()

    with (
        patch.object(job, "configure_logging"),
        patch.object(job, "create_session_factory", return_value=session_factory),
        patch.object(job, "BinanceClient", return_value=client_context) as client_type,
        patch.object(
            job, "ingest_latest_closed_klines", AsyncMock(return_value=expected)
        ) as orchestration,
        patch.object(job.logger, "info") as info,
    ):
        actual = await job.run(settings=settings(), clock=clock, engine=database_engine)

    assert actual is expected
    clock.assert_called_once_with()
    client_type.assert_called_once()
    orchestration.assert_awaited_once_with(
        session=session,
        client=client,
        assets=(Asset.BTC_USDT, Asset.ETH_USDT),
        interval=Interval.FIVE_MINUTES,
        now=NOW,
        limit=18,
    )
    info.assert_called_once()
    session_context.__aexit__.assert_awaited_once()
    client_context.__aexit__.assert_awaited_once()
    database_engine.dispose.assert_awaited_once_with()
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_logs_and_propagates_failure_after_cleanup() -> None:
    database_engine = MagicMock()
    database_engine.dispose = AsyncMock()
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=AsyncMock())
    session_context.__aexit__ = AsyncMock(return_value=None)
    client_context = MagicMock()
    client_context.__aenter__ = AsyncMock(return_value=MagicMock())
    client_context.__aexit__ = AsyncMock(return_value=None)
    failure = RuntimeError("provider unavailable")

    with (
        patch.object(job, "configure_logging"),
        patch.object(
            job,
            "create_session_factory",
            return_value=Mock(return_value=session_context),
        ),
        patch.object(job, "BinanceClient", return_value=client_context),
        patch.object(
            job, "ingest_latest_closed_klines", AsyncMock(side_effect=failure)
        ),
        patch.object(job.logger, "exception") as exception_log,
        pytest.raises(RuntimeError, match="provider unavailable"),
    ):
        await job.run(settings=settings(), engine=database_engine)

    exception_log.assert_called_once()
    logged_arguments = repr(exception_log.call_args)
    assert "secret" not in logged_arguments
    assert "postgresql" not in logged_arguments
    session_context.__aexit__.assert_awaited_once()
    client_context.__aexit__.assert_awaited_once()
    database_engine.dispose.assert_awaited_once_with()


def test_completion_log_contains_aggregate_and_per_asset_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=job.__name__):
        job._log_completion(result())

    assert len(caplog.records) == 1
    fields = caplog.records[0].__dict__["structured_fields"]
    assert fields == {
        "job_name": "ingest_closed_candles",
        "ingested_at": NOW.isoformat(),
        "start_time": 1000,
        "end_time": 2000,
        "received": 4,
        "inserted": 3,
        "duplicates": 1,
        "assets": [
            {"asset": "BTC/USDT", "received": 2, "inserted": 2, "duplicates": 0},
            {"asset": "ETH/USDT", "received": 2, "inserted": 1, "duplicates": 1},
        ],
    }
