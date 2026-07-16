from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

from app.features.calculation import FeatureCalculationResult
from app.features.models import FeatureName
from app.jobs import calculate_market_features as job
from app.market.enums import Asset, Interval, MarketType, Provider

pytestmark = pytest.mark.asyncio
NOW = datetime(2026, 7, 16, tzinfo=UTC)


def feature_result(asset: Asset) -> FeatureCalculationResult:
    return FeatureCalculationResult(
        asset=asset,
        feature_name=FeatureName.EMA_20,
        feature_version=1,
        candle_count_read=20,
        values_calculated=1,
        values_inserted=1,
        duplicates=0,
        first_feature_time=NOW,
        last_feature_time=NOW,
    )


def resources() -> tuple[MagicMock, AsyncMock, MagicMock]:
    engine = MagicMock()
    engine.dispose = AsyncMock()
    session = AsyncMock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=None)
    return engine, session, context


async def test_job_commits_both_assets_once_with_same_timestamp() -> None:
    engine, session, context = resources()
    calculate = AsyncMock(
        side_effect=[feature_result(Asset.BTC_USDT), feature_result(Asset.ETH_USDT)]
    )
    with (
        patch.object(job, "configure_logging"),
        patch.object(job, "create_database_engine", return_value=engine),
        patch.object(
            job, "create_session_factory", return_value=Mock(return_value=context)
        ),
        patch.object(job, "calculate_ema_20_features", calculate),
    ):
        results = await job.run(clock=Mock(return_value=NOW))

    assert [result.asset for result in results] == [Asset.BTC_USDT, Asset.ETH_USDT]
    assert calculate.await_args_list == [
        call(
            session=session,
            asset=asset,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=Interval.FIVE_MINUTES,
            calculated_at=NOW,
        )
        for asset in (Asset.BTC_USDT, Asset.ETH_USDT)
    ]
    session.commit.assert_awaited_once_with()
    session.rollback.assert_not_awaited()
    engine.dispose.assert_awaited_once_with()


async def test_second_asset_failure_rolls_back_and_propagates() -> None:
    engine, session, context = resources()
    original = RuntimeError("ETH calculation failed")
    calculate = AsyncMock(side_effect=[feature_result(Asset.BTC_USDT), original])
    with (
        patch.object(job, "configure_logging"),
        patch.object(job, "create_database_engine", return_value=engine),
        patch.object(
            job, "create_session_factory", return_value=Mock(return_value=context)
        ),
        patch.object(job, "calculate_ema_20_features", calculate),
        pytest.raises(RuntimeError) as captured,
    ):
        await job.run(clock=Mock(return_value=NOW))

    assert captured.value is original
    session.rollback.assert_awaited_once_with()
    session.commit.assert_not_awaited()
    engine.dispose.assert_awaited_once_with()
