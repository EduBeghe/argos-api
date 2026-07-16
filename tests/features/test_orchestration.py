from datetime import UTC, datetime
from unittest.mock import AsyncMock, call, patch

import pytest

from app.features import orchestration
from app.features.calculation import FeatureCalculationResult
from app.market.enums import Asset, Interval, MarketType, Provider

NOW = datetime(2026, 7, 16, tzinfo=UTC)


def result(asset: Asset) -> FeatureCalculationResult:
    return FeatureCalculationResult(asset, 20, 0, 0, 0, None, None, ())


@pytest.mark.asyncio
async def test_calculates_btc_then_eth_with_same_timestamp() -> None:
    session = AsyncMock()
    calculate = AsyncMock(side_effect=[result(Asset.BTC_USDT), result(Asset.ETH_USDT)])
    with patch.object(orchestration, "calculate_market_features", calculate):
        results = await orchestration.calculate_all_market_features(
            session=session, calculated_at=NOW
        )

    assert tuple(item.asset for item in results) == (Asset.BTC_USDT, Asset.ETH_USDT)
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
