from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.session import get_db_session
from app.decisions.persistence import get_latest_decision_snapshot
from app.decisions.schemas import DecisionSnapshotResponse
from app.features.calculation import FEATURE_VERSION
from app.features.persistence import get_latest_feature_snapshot
from app.features.schemas import MarketFeatureResponse
from app.market.catalog import (
    InstrumentCapability,
    MarketCatalogError,
    MarketInstrumentDefinition,
    binance_spot_five_minute_ingestion_instruments,
    market_instruments,
    resolve_instrument,
)
from app.market.enums import Asset, Interval
from app.market.queries import (
    MarketStatusAggregate,
    get_market_status_aggregates_batch,
    list_market_candles,
)
from app.market.schemas import (
    MarketCandleQuery,
    MarketCandleResponse,
    MarketInstrumentResponse,
    MarketStatusResponse,
)
from app.market.status import build_market_statuses
from app.scores.calculation import MODEL_NAME, MODEL_VERSION
from app.scores.persistence import get_latest_market_score
from app.scores.schemas import MarketScoreResponse

router = APIRouter(prefix="/market", tags=["market"])
SessionDependency = Annotated[AsyncSession, Depends(get_db_session)]
QueryDependency = Annotated[MarketCandleQuery, Query()]


@router.get(
    "/instruments",
    response_model=list[MarketInstrumentResponse],
    summary="List configured market instruments",
)
async def instruments() -> list[MarketInstrumentResponse]:
    return [
        MarketInstrumentResponse.from_definition(definition)
        for definition in market_instruments()
    ]


@router.get(
    "/candles",
    response_model=list[MarketCandleResponse],
    summary="List persisted market candles",
)
async def candles(
    params: QueryDependency, session: SessionDependency
) -> list[MarketCandleResponse]:
    """Return persisted candles matching explicit UTC query filters."""

    instrument = _resolve(params.asset, params.interval, InstrumentCapability.INGESTION)
    results = await list_market_candles(
        session=session,
        asset=params.asset,
        provider=instrument.provider,
        market_type=instrument.market_type,
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
    instruments = binance_spot_five_minute_ingestion_instruments()
    aggregates = await get_market_status_aggregates_batch(
        session=session, instruments=instruments
    )
    return build_market_status(aggregates=aggregates, now=now)


def build_market_status(
    *, aggregates: Sequence[MarketStatusAggregate], now: datetime
) -> list[MarketStatusResponse]:
    """Build ordered asset status from grouped persistence values and one UTC clock."""

    instruments = binance_spot_five_minute_ingestion_instruments()
    statuses = build_market_statuses(
        instruments=instruments,
        aggregates=tuple(aggregates),
        now=now,
    )
    return [statuses[instrument] for instrument in instruments]


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

    instrument = _resolve(asset, interval, InstrumentCapability.SCORING)
    features = await get_latest_feature_snapshot(
        session=session,
        asset=asset,
        provider=instrument.provider,
        market_type=instrument.market_type,
        interval=interval,
        feature_version=FEATURE_VERSION,
    )
    if not features:
        raise HTTPException(status_code=404, detail="Feature snapshot not found")
    return MarketFeatureResponse.from_features(features)


@router.get(
    "/scores/latest",
    response_model=MarketScoreResponse,
    summary="Get the latest persisted baseline technical score",
)
async def latest_score(
    session: SessionDependency,
    asset: Annotated[Asset, Query(description="Market asset.")],
    interval: Annotated[Interval, Query(description="Candle interval.")],
) -> MarketScoreResponse:
    instrument = _resolve(asset, interval, InstrumentCapability.SCORING)
    score = await get_latest_market_score(
        session=session,
        asset=asset,
        provider=instrument.provider,
        market_type=instrument.market_type,
        interval=interval,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )
    if score is None:
        raise HTTPException(status_code=404, detail="Market score not found")
    return MarketScoreResponse.from_domain(score)


@router.get(
    "/decisions/latest",
    response_model=DecisionSnapshotResponse,
    summary="Get the latest persisted Decision Snapshot",
)
async def latest_decision(
    session: SessionDependency,
    asset: Annotated[Asset, Query(description="Market asset.")],
    interval: Annotated[Interval, Query(description="Candle interval.")],
) -> DecisionSnapshotResponse:
    instrument = _resolve(asset, interval, InstrumentCapability.DECISION)
    snapshot = await get_latest_decision_snapshot(
        session=session,
        asset=asset,
        provider=instrument.provider,
        market_type=instrument.market_type,
        interval=interval,
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Decision Snapshot not found")
    return DecisionSnapshotResponse.from_domain(snapshot)


def _resolve(
    asset: Asset, interval: Interval, capability: InstrumentCapability
) -> MarketInstrumentDefinition:
    try:
        return resolve_instrument(asset=asset, interval=interval, capability=capability)
    except MarketCatalogError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
