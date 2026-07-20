Exit code: 0
Wall time: 0.4 seconds
Output:
import argparse
import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime

from app.core.config import get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.ingestion.backfill import (
    DEFAULT_PAGE_SIZE,
    BackfillRange,
    BackfillResult,
    backfill_instrument,
    validate_closed_range,
)
from app.ingestion.binance.client import BinanceClient
from app.market.catalog import Instrument, ingestion_enabled_instruments
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.storage_telemetry import (
    MarketStorageTelemetry,
    read_market_storage_telemetry,
)

logger = logging.getLogger(__name__)


def parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an ISO-8601 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise argparse.ArgumentTypeError("must use the UTC timezone")
    return parsed.astimezone(UTC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill persisted market candles")
    parser.add_argument("--from", dest="start", required=True, type=parse_utc)
    parser.add_argument("--to", dest="end", required=True, type=parse_utc)
    parser.add_argument("--asset", type=Asset)
    parser.add_argument("--provider", type=Provider)
    parser.add_argument("--market-type", type=MarketType)
    parser.add_argument("--interval", type=Interval)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def select_instruments(args: argparse.Namespace) -> tuple[Instrument, ...]:
    instruments = ingestion_enabled_instruments()
    selected = tuple(
        item
        for item in instruments
        if (args.asset is None or item.asset is args.asset)
        and (args.provider is None or item.provider is args.provider)
        and (args.market_type is None or item.market_type is args.market_type)
        and (args.interval is None or item.interval is args.interval)
        and item.provider is Provider.BINANCE
    )
    if not selected:
        raise ValueError("no enabled Binance instrument matches the requested identity")
    return selected


async def run(
    args: argparse.Namespace, *, now: datetime | None = None
) -> tuple[BackfillResult, ...]:
    settings = get_settings()
    configure_logging(settings.log_level)
    requested_now = now or datetime.now(UTC)
    instruments = select_instruments(args)
    interval = args.interval or instruments[0].interval
    if any(item.interval is not interval for item in instruments):
        raise ValueError("selected instruments must use one interval")
    requested = BackfillRange(args.start, args.end, interval)
    validate_closed_range(requested, now=requested_now)
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    try:
        async with factory() as session, BinanceClient(settings) as client:
            before = await read_market_storage_telemetry(session)
            results: list[BackfillResult] = []
            for instrument in instruments:
                instrument_asset = instrument.asset.value

                def log_page(
                    page: int,
                    received: int,
                    inserted: int,
                    duplicates: int,
                    asset: str = instrument_asset,
                ) -> None:
                    logger.info(
                        "Market candle backfill page completed",
                        extra={
                            "structured_fields": {
                                "asset": asset,
                                "page": page,
                                "received": received,
                                "inserted": inserted,
                                "duplicates": duplicates,
                                "dry_run": args.dry_run,
                            }
                        },
                    )

                result = await backfill_instrument(
                    session=session,
                    client=client,
                    instrument=instrument,
                    requested=requested,
                    ingested_at=requested_now,
                    page_size=args.page_size,
                    max_pages=args.max_pages,
                    dry_run=args.dry_run,
                    page_completed=log_page,
                )
                results.append(result)
                logger.info(
                    "Market candle backfill instrument completed",
                    extra={"structured_fields": _result_fields(result)},
                )
            after = await read_market_storage_telemetry(session)
            logger.info(
                "Market candle backfill storage summary",
                extra={"structured_fields": _storage_fields(before, after)},
            )
            return tuple(results)
    finally:
        await engine.dispose()


def _result_fields(result: BackfillResult) -> dict[str, object]:
    return {
        "asset": result.instrument.asset.value,
        "provider": result.instrument.provider.value,
        "market_type": result.instrument.market_type.value,
        "interval": result.instrument.interval.value,
        "expected_slots": result.expected_slots,
        "received": result.received,
        "inserted": result.inserted,
        "duplicates": result.duplicates,
        "missing": result.missing,
        "first_missing_open_time": result.first_missing_open_time.isoformat()
        if result.first_missing_open_time
        else None,
        "last_missing_open_time": result.last_missing_open_time.isoformat()
        if result.last_missing_open_time
        else None,
        "pages": result.pages,
        "retries": result.retries,
        "first_processed_open_time": result.first_processed_open_time.isoformat()
        if result.first_processed_open_time
        else None,
        "last_processed_open_time": result.last_processed_open_time.isoformat()
        if result.last_processed_open_time
        else None,
    }


def _storage_fields(
    before: MarketStorageTelemetry, after: MarketStorageTelemetry
) -> dict[str, object]:
    if not before.supported or not after.supported:
        return {"postgresql_storage_telemetry": "omitted", "reason": "non_postgresql"}
    row_delta = (after.rows or 0) - (before.rows or 0)
    byte_delta = (after.total_bytes or 0) - (before.total_bytes or 0)
    return {
        "rows_before": before.rows,
        "rows_after": after.rows,
        "row_delta": row_delta,
        "table_bytes_before": before.table_bytes,
        "table_bytes_after": after.table_bytes,
        "index_bytes_before": before.index_bytes,
        "index_bytes_after": after.index_bytes,
        "total_bytes_before": before.total_bytes,
        "total_bytes_after": after.total_bytes,
        "database_bytes_before": before.database_bytes,
        "database_bytes_after": after.database_bytes,
        "byte_delta": byte_delta,
        "approximate_bytes_per_new_candle": byte_delta / row_delta
        if row_delta > 0
        else None,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

