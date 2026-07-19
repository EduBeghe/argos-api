from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.session import get_db_session
from app.decisions.persistence import (
    get_latest_decision_snapshot,
    list_decision_snapshots,
)
from app.decisions.schemas import DecisionSnapshotResponse
from app.features.calculation import FEATURE_VERSION
from app.features.persistence import get_latest_feature_snapshot
from app.features.schemas import MarketFeatureResponse
from app.market.catalog import (
    InstrumentCapability,
    MarketCatalogAmbiguityError,
    MarketCatalogError,
    MarketInstrumentDefinition,
    binance_spot_five_minute_ingestion_instruments,
    market_instruments,
    resolve_instrument,
)
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.queries import (
    MarketStatusAggregate,
    get_market_status_aggregates_batch,
    list_market_candles,
)
from app.market.schemas import (
    DecisionHistoryQuery,
    MarketCandleQuery,
    MarketCandleResponse,
    MarketInstrumentResponse,
    MarketStatusResponse,
    ScoreHistoryQuery,
)
from app.market.status import build_market_statuses
from app.scores.calculation import MODEL_NAME, MODEL_VERSION
from app.scores.persistence import get_latest_market_score, list_market_scores
from app.scores.schemas import MarketScoreResponse

router = APIRouter(prefix="/market", tags=["market"])
SessionDependency = Annotated[AsyncSession, Depends(get_db_session)]
QueryDependency = Annotated[MarketCandleQuery, Query()]
ScoreQueryDependency = Annotated[ScoreHistoryQuery, Query()]
DecisionQueryDependency = Annotated[DecisionHistoryQuery, Query()]


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

    instrument = _resolve(
        params.asset,
        params.interval,
        InstrumentCapability.INGESTION,
        params.provider,
        params.market_type,
    )
    results = await list_market_candles(
        session=session,
        asset=params.asset,
        provider=instrument.provider,
        market_type=instrument.market_type,
        interval=params.interval,
        start_time=params.effective_from,
        end_time=params.effective_to,
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
    provider: Annotated[Provider | None, Query(description="Market provider.")] = None,
    market_type: Annotated[MarketType | None, Query(description="Market type.")] = None,
) -> MarketFeatureResponse:
    """Return same-candle persisted features without triggering calculation."""

    instrument = _resolve(
        asset, interval, InstrumentCapability.SCORING, provider, market_type
    )
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
    provider: Annotated[Provider | None, Query(description="Market provider.")] = None,
    market_type: Annotated[MarketType | None, Query(description="Market type.")] = None,
) -> MarketScoreResponse:
    instrument = _resolve(
        asset, interval, InstrumentCapability.SCORING, provider, market_type
    )
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
    "/scores",
    response_model=list[MarketScoreResponse],
    summary="List persisted technical Score history",
)
async def scores(
    params: ScoreQueryDependency, session: SessionDependency
) -> list[MarketScoreResponse]:
    instrument = _resolve(
        params.asset,
        params.interval,
        InstrumentCapability.SCORING,
        params.provider,
        params.market_type,
    )
    results = await list_market_scores(
        session=session,
        asset=instrument.asset,
        provider=instrument.provider,
        market_type=instrument.market_type,
        interval=instrument.interval,
        from_time=params.from_time,
        to_time=params.to_time,
        order=params.order,
        limit=params.limit,
    )
    return [MarketScoreResponse.from_domain(score) for score in results]


@router.get(
    "/decisions/latest",
    response_model=DecisionSnapshotResponse,
    summary="Get the latest persisted Decision Snapshot",
)
async def latest_decision(
    session: SessionDependency,
    asset: Annotated[Asset, Query(description="Market asset.")],
    interval: Annotated[Interval, Query(description="Candle interval.")],
    provider: Annotated[Provider | None, Query(description="Market provider.")] = None,
    market_type: Annotated[MarketType | None, Query(description="Market type.")] = None,
) -> DecisionSnapshotResponse:
    instrument = _resolve(
        asset, interval, InstrumentCapability.DECISION, provider, market_type
    )
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


@router.get(
    "/decisions",
    response_model=list[DecisionSnapshotResponse],
    summary="List persisted Decision Snapshot history",
)
async def decisions(
    params: DecisionQueryDependency, session: SessionDependency
) -> list[DecisionSnapshotResponse]:
    instrument = _resolve(
        params.asset,
        params.interval,
        InstrumentCapability.DECISION,
        params.provider,
        params.market_type,
    )
    results = await list_decision_snapshots(
        session=session,
        asset=instrument.asset,
        provider=instrument.provider,
        market_type=instrument.market_type,
        interval=instrument.interval,
        run_id=params.run_id,
        from_time=params.from_time,
        to_time=params.to_time,
        order=params.order,
        limit=params.limit,
    )
    return [DecisionSnapshotResponse.from_domain(snapshot) for snapshot in results]


def _resolve(
    asset: Asset,
    interval: Interval,
    capability: InstrumentCapability,
    provider: Provider | None = None,
    market_type: MarketType | None = None,
) -> MarketInstrumentDefinition:
    try:
        return resolve_instrument(
            asset=asset,
            interval=interval,
            capability=capability,
            provider=provider,
            market_type=market_type,
        )
    except MarketCatalogAmbiguityError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except MarketCatalogError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
