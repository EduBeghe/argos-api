from dataclasses import dataclass

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.persistence import MarketCandleRecord


@dataclass(frozen=True, slots=True)
class MarketStorageTelemetry:
    supported: bool
    rows: int | None = None
    table_bytes: int | None = None
    index_bytes: int | None = None
    total_bytes: int | None = None
    database_bytes: int | None = None


async def read_market_storage_telemetry(
    session: AsyncSession,
) -> MarketStorageTelemetry:
    """Read PostgreSQL storage metrics, or explicitly report unsupported dialects."""

    bind = session.get_bind()
    if bind.dialect.name != "postgresql":
        return MarketStorageTelemetry(supported=False)
    rows = await session.scalar(select(func.count()).select_from(MarketCandleRecord))
    sizes = (
        await session.execute(
            text(
                "SELECT pg_relation_size('market_candles'), "
                "pg_indexes_size('market_candles'), "
                "pg_total_relation_size('market_candles'), "
                "pg_database_size(current_database())"
            )
        )
    ).one()
    return MarketStorageTelemetry(
        True, int(rows or 0), *(int(value) for value in sizes)
    )
