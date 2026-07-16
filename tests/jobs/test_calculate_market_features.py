from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.features.calculation import FeatureCalculationResult
from app.jobs import calculate_market_features as job
from app.market.enums import Asset

pytestmark = pytest.mark.asyncio
NOW = datetime(2026, 7, 16, tzinfo=UTC)


def feature_result(asset: Asset) -> FeatureCalculationResult:
    return FeatureCalculationResult(
        asset=asset,
        candle_count_read=20,
        values_calculated=0,
        values_inserted=0,
        duplicates=0,
        first_feature_time=None,
        last_feature_time=None,
        features=(),
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
    expected = (feature_result(Asset.BTC_USDT), feature_result(Asset.ETH_USDT))
    calculate = AsyncMock(return_value=expected)
    with (
        patch.object(job, "configure_logging"),
        patch.object(job, "create_database_engine", return_value=engine),
        patch.object(
            job, "create_session_factory", return_value=Mock(return_value=context)
        ),
        patch.object(job, "calculate_all_market_features", calculate),
    ):
        results = await job.run(clock=Mock(return_value=NOW))

    assert [result.asset for result in results] == [Asset.BTC_USDT, Asset.ETH_USDT]
    calculate.assert_awaited_once_with(session=session, calculated_at=NOW)
    session.commit.assert_awaited_once_with()
    session.rollback.assert_not_awaited()
    engine.dispose.assert_awaited_once_with()


async def test_second_asset_failure_rolls_back_and_propagates() -> None:
    engine, session, context = resources()
    original = RuntimeError("ETH calculation failed")
    calculate = AsyncMock(side_effect=original)
    with (
        patch.object(job, "configure_logging"),
        patch.object(job, "create_database_engine", return_value=engine),
        patch.object(
            job, "create_session_factory", return_value=Mock(return_value=context)
        ),
        patch.object(job, "calculate_all_market_features", calculate),
        pytest.raises(RuntimeError) as captured,
    ):
        await job.run(clock=Mock(return_value=NOW))

    assert captured.value is original
    session.rollback.assert_awaited_once_with()
    session.commit.assert_not_awaited()
    engine.dispose.assert_awaited_once_with()
