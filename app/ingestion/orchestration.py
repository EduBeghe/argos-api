from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.binance.client import BinanceClient
from app.ingestion.binance.ingestion import IngestionResult, ingest_klines
from app.ingestion.time_windows import calculate_closed_candle_window
from app.market.enums import Asset, Interval


@dataclass(frozen=True, slots=True)
class AssetIngestionResult:
    """Ingestion result associated with one requested asset."""

    asset: Asset
    result: IngestionResult


@dataclass(frozen=True, slots=True)
class ClosedCandleIngestionResult:
    """Atomic result for one closed-candle ingestion run."""

    ingested_at: datetime
    start_time: int
    end_time: int
    assets: tuple[AssetIngestionResult, ...]
    received: int
    inserted: int
    duplicates: int

    def __post_init__(self) -> None:
        if self.received != sum(item.result.received for item in self.assets):
            raise ValueError("received must equal the sum of asset results")
        if self.inserted != sum(item.result.inserted for item in self.assets):
            raise ValueError("inserted must equal the sum of asset results")
        if self.duplicates != sum(item.result.duplicates for item in self.assets):
            raise ValueError("duplicates must equal the sum of asset results")


async def ingest_latest_closed_klines(
    *,
    session: AsyncSession,
    client: BinanceClient,
    assets: Sequence[Asset],
    interval: Interval,
    now: datetime,
    limit: int,
) -> ClosedCandleIngestionResult:
    """Atomically ingest the latest fully closed candles for requested assets."""

    validated_assets = _validate_assets(assets)
    window = calculate_closed_candle_window(
        now=now,
        interval=interval,
        limit=limit,
    )

    asset_results: list[AssetIngestionResult] = []
    for asset in validated_assets:
        result = await ingest_klines(
            session=session,
            client=client,
            asset=asset,
            interval=interval,
            start_time=window.start_time,
            end_time=window.end_time,
            limit=limit,
            ingested_at=now,
        )
        asset_results.append(AssetIngestionResult(asset=asset, result=result))

    immutable_results = tuple(asset_results)
    return ClosedCandleIngestionResult(
        ingested_at=now,
        start_time=window.start_time,
        end_time=window.end_time,
        assets=immutable_results,
        received=sum(item.result.received for item in immutable_results),
        inserted=sum(item.result.inserted for item in immutable_results),
        duplicates=sum(item.result.duplicates for item in immutable_results),
    )


def _validate_assets(assets: Sequence[Asset]) -> tuple[Asset, ...]:
    if not assets:
        raise ValueError("assets must not be empty")
    if any(not isinstance(asset, Asset) for asset in assets):
        raise ValueError("assets contains an unsupported asset")

    immutable_assets = tuple(assets)
    if len(set(immutable_assets)) != len(immutable_assets):
        raise ValueError("assets must not contain duplicates")
    return immutable_assets
