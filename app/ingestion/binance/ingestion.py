from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.binance.client import BinanceClient
from app.ingestion.binance.normalizer import normalize_kline
from app.market.enums import Asset, Interval
from app.market.persistence import insert_market_candles


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Counts produced by one complete kline ingestion operation."""

    received: int
    normalized: int
    inserted: int
    duplicates: int

    def __post_init__(self) -> None:
        if min(self.received, self.normalized, self.inserted, self.duplicates) < 0:
            raise ValueError("ingestion counters must be non-negative")
        if self.received != self.normalized:
            raise ValueError("received must equal normalized")
        if self.inserted + self.duplicates != self.normalized:
            raise ValueError("inserted plus duplicates must equal normalized")


async def ingest_klines(
    *,
    session: AsyncSession,
    client: BinanceClient,
    asset: Asset,
    interval: Interval,
    start_time: int | None = None,
    end_time: int | None = None,
    limit: int | None = None,
    ingested_at: datetime,
) -> IngestionResult:
    """Request, normalize, and stage Binance Spot klines for persistence."""

    raw_klines = await client.get_klines(
        asset=asset,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )
    received = len(raw_klines)
    if received == 0:
        return IngestionResult(received=0, normalized=0, inserted=0, duplicates=0)

    candles = [
        normalize_kline(
            raw_kline=raw_kline,
            asset=asset,
            interval=interval,
            ingested_at=ingested_at,
        )
        for raw_kline in raw_klines
    ]
    inserted = await insert_market_candles(session, candles)
    normalized = len(candles)
    return IngestionResult(
        received=received,
        normalized=normalized,
        inserted=inserted,
        duplicates=normalized - inserted,
    )
