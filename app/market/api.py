from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.session import get_db_session
from app.features.calculation import FEATURE_VERSION
from app.features.persistence import get_latest_feature_snapshot
from app.features.schemas import MarketFeatureResponse
from app.ingestion.time_windows import latest_closed_five_minute_open_time
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.queries import (
    MarketStatusAggregate,
    get_market_status_aggregates,
    list_market_candles,
)
from app.market.schemas import (
    MarketCandleQuery,
    MarketCandleResponse,
    MarketStatusResponse,
)

router = APIRouter(prefix="/market", tags=["market"])
SessionDependency = Annotated[AsyncSession, Depends(get_db_session)]
QueryDependency = Annotated[MarketCandleQuery, Query()]
SUPPORTED_ASSETS = (Asset.BTC_USDT, Asset.ETH_USDT)
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@router.get(
    "/candles",
    response_model=list[MarketCandleResponse],
    summary="List persisted market candles",
)
async def candles(
    params: QueryDependency, session: SessionDependency
) -> list[MarketCandleResponse]:
    """Return persisted candles matching explicit UTC query filters."""

    results = await list_market_candles(
        session=session,
        asset=params.asset,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=params.interval,
        start_time=params.start_time,
        end_time=params.end_time,
        limit=params.limit,
        order=params.order,
    )
    return [MarketCandleResponse.from_domain(candle) for candle in results]


@router.get(
    "/status",
    response_model=list[MarketStatusResponse],
    summary="Get persisted market-data freshness",
)
async def market_status(session: SessionDependency) -> list[MarketStatusResponse]:
    """Report whether each asset includes the latest fully closed UTC candle."""

    now = datetime.now(UTC)
    aggregates = await get_market_status_aggregates(
        session=session,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
    )
    return build_market_status(aggregates=aggregates, now=now)


def build_market_status(
    *, aggregates: list[MarketStatusAggregate], now: datetime
) -> list[MarketStatusResponse]:
    """Build ordered asset status from grouped persistence values and one UTC clock."""

    expected_open_time = _UNIX_EPOCH + timedelta(
        milliseconds=latest_closed_five_minute_open_time(now)
    )
    by_asset = {aggregate.asset: aggregate for aggregate in aggregates}
    statuses: list[MarketStatusResponse] = []
    for asset in SUPPORTED_ASSETS:
        aggregate = by_asset.get(asset)
        if aggregate is None:
            statuses.append(
                MarketStatusResponse(
                    asset=asset,
                    interval=Interval.FIVE_MINUTES,
                    candle_count=0,
                    latest_open_time=None,
                    latest_close_time=None,
                    latest_ingested_at=None,
                    age_seconds=None,
                    is_fresh=False,
                )
            )
            continue
        age = now - aggregate.latest_close_time
        statuses.append(
            MarketStatusResponse(
                asset=asset,
                interval=Interval.FIVE_MINUTES,
                candle_count=aggregate.candle_count,
                latest_open_time=aggregate.latest_open_time,
                latest_close_time=aggregate.latest_close_time,
                latest_ingested_at=aggregate.latest_ingested_at,
                age_seconds=age.days * 86_400 + age.seconds,
                is_fresh=aggregate.latest_open_time >= expected_open_time,
            )
        )
    return statuses


@router.get(
    "/features/latest",
    response_model=MarketFeatureResponse,
    summary="Get the latest persisted feature snapshot",
)
async def latest_feature(
    session: SessionDependency,
    asset: Annotated[Asset, Query(description="Market asset.")],
    interval: Annotated[Interval, Query(description="Candle interval.")],
) -> MarketFeatureResponse:
    """Return same-candle persisted features without triggering calculation."""

    features = await get_latest_feature_snapshot(
        session=session,
        asset=asset,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=interval,
        feature_version=FEATURE_VERSION,
    )
    if not features:
        raise HTTPException(status_code=404, detail="Feature snapshot not found")
    return MarketFeatureResponse.from_features(features)
