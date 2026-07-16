import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.core.config import Settings
from app.features.calculation import FeatureCalculationResult
from app.ingestion.binance.ingestion import IngestionResult
from app.ingestion.orchestration import (
    AssetIngestionResult,
    ClosedCandleIngestionResult,
)
from app.jobs import ingest_closed_candles as job
from app.market.enums import Asset, Interval

INGESTED_AT = datetime(2026, 7, 16, 12, 7, tzinfo=UTC)
CALCULATED_AT = datetime(2026, 7, 16, 12, 8, tzinfo=UTC)


def settings() -> Settings:
    return Settings.model_validate(
        {
            "DATABASE_URL": "postgresql+asyncpg://user:secret@database/argos",
            "BINANCE_BASE_URL": "https://binance.test",
            "BINANCE_INGESTION_LIMIT": 18,
        }
    )


def ingestion_result() -> ClosedCandleIngestionResult:
    btc = IngestionResult(2, 2, 2, 0)
    eth = IngestionResult(2, 2, 1, 1)
    return ClosedCandleIngestionResult(
        INGESTED_AT,
        1000,
        2000,
        (
            AssetIngestionResult(Asset.BTC_USDT, btc),
            AssetIngestionResult(Asset.ETH_USDT, eth),
        ),
        4,
        3,
        1,
    )


def feature_result(asset: Asset) -> FeatureCalculationResult:
    return FeatureCalculationResult(asset, 20, 0, 0, 0, None, None, ())


def resources() -> tuple[MagicMock, AsyncMock, MagicMock, MagicMock]:
    engine = MagicMock()
    engine.dispose = AsyncMock()
    session = AsyncMock()
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=None)
    client_context = MagicMock()
    client_context.__aenter__ = AsyncMock(return_value=MagicMock())
    client_context.__aexit__ = AsyncMock(return_value=None)
    return engine, session, session_context, client_context


@pytest.mark.asyncio
async def test_run_commits_ingestion_then_features_and_closes_resources() -> None:
    engine, session, session_context, client_context = resources()
    events: list[str] = []
    ingestion = ingestion_result()
    features = (feature_result(Asset.BTC_USDT), feature_result(Asset.ETH_USDT))

    async def stage_ingestion(**_: object) -> ClosedCandleIngestionResult:
        events.append("ingest")
        return ingestion

    async def stage_features(**_: object) -> tuple[FeatureCalculationResult, ...]:
        events.append("features")
        return features

    ingest = AsyncMock(side_effect=stage_ingestion)
    calculate = AsyncMock(side_effect=stage_features)
    session.commit.side_effect = lambda: events.append("commit")
    clock = Mock(side_effect=[INGESTED_AT, CALCULATED_AT])

    with (
        patch.object(job, "configure_logging"),
        patch.object(
            job,
            "create_session_factory",
            return_value=Mock(return_value=session_context),
        ),
        patch.object(job, "BinanceClient", return_value=client_context),
        patch.object(job, "ingest_latest_closed_klines", ingest),
        patch.object(job, "calculate_all_market_features", calculate),
    ):
        result = await job.run(settings=settings(), clock=clock, engine=engine)

    assert events == ["ingest", "commit", "features", "commit"]
    assert result == job.IngestionPipelineResult(ingestion, CALCULATED_AT, features)
    assert clock.call_count == 2
    ingest.assert_awaited_once_with(
        session=session,
        client=client_context.__aenter__.return_value,
        assets=(Asset.BTC_USDT, Asset.ETH_USDT),
        interval=Interval.FIVE_MINUTES,
        now=INGESTED_AT,
        limit=18,
    )
    calculate.assert_awaited_once_with(session=session, calculated_at=CALCULATED_AT)
    assert session.commit.await_count == 2
    session.rollback.assert_not_awaited()
    engine.dispose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_ingestion_failure_rolls_back_and_never_calculates_features() -> None:
    engine, session, session_context, client_context = resources()
    original = RuntimeError("provider unavailable")
    calculate = AsyncMock()
    with (
        patch.object(job, "configure_logging"),
        patch.object(
            job,
            "create_session_factory",
            return_value=Mock(return_value=session_context),
        ),
        patch.object(job, "BinanceClient", return_value=client_context),
        patch.object(
            job, "ingest_latest_closed_klines", AsyncMock(side_effect=original)
        ),
        patch.object(job, "calculate_all_market_features", calculate),
        pytest.raises(RuntimeError) as captured,
    ):
        await job.run(settings=settings(), engine=engine)

    assert captured.value is original
    session.rollback.assert_awaited_once_with()
    session.commit.assert_not_awaited()
    calculate.assert_not_awaited()
    engine.dispose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_second_feature_failure_rolls_back_only_feature_transaction() -> None:
    engine, session, session_context, client_context = resources()
    original = RuntimeError("ETH calculation failed")
    calculate = AsyncMock(side_effect=original)
    with (
        patch.object(job, "configure_logging"),
        patch.object(
            job,
            "create_session_factory",
            return_value=Mock(return_value=session_context),
        ),
        patch.object(job, "BinanceClient", return_value=client_context),
        patch.object(
            job,
            "ingest_latest_closed_klines",
            AsyncMock(return_value=ingestion_result()),
        ),
        patch.object(job, "calculate_all_market_features", calculate),
        pytest.raises(RuntimeError) as captured,
    ):
        await job.run(
            settings=settings(),
            clock=Mock(side_effect=[INGESTED_AT, CALCULATED_AT]),
            engine=engine,
        )

    assert captured.value is original
    session.commit.assert_awaited_once_with()
    session.rollback.assert_awaited_once_with()
    calculate.assert_awaited_once_with(session=session, calculated_at=CALCULATED_AT)


def test_completion_log_contains_both_stage_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = job.IngestionPipelineResult(
        ingestion_result(),
        CALCULATED_AT,
        (feature_result(Asset.BTC_USDT), feature_result(Asset.ETH_USDT)),
    )
    with caplog.at_level(logging.INFO, logger=job.__name__):
        job._log_completion(result)

    fields = caplog.records[0].__dict__["structured_fields"]
    assert fields["ingestion_completed"] is True
    assert fields["received"] == 4
    assert fields["feature_calculated_at"] == CALCULATED_AT.isoformat()
    assert fields["feature_values_inserted"] == 0
    assert [item["asset"] for item in fields["feature_assets"]] == [
        "BTC/USDT",
        "ETH/USDT",
    ]
