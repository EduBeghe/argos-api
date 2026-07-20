import argparse
import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from time import perf_counter

from app.core.config import get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.market.catalog import (
    MARKET_INSTRUMENTS,
    InstrumentCapability,
    MarketInstrumentDefinition,
    resolve_instrument,
)
from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.feature_matrix import build_historical_feature_matrix
from app.optimization.historical_data import (
    DEFAULT_PAGE_SIZE,
    HistoricalDataset,
    HistoricalDatasetIdentity,
    HistoricalDatasetRange,
    read_historical_dataset,
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
    parser = argparse.ArgumentParser(
        description="Validate one read-only historical optimization dataset"
    )
    parser.add_argument("--from", dest="start", required=True, type=parse_utc)
    parser.add_argument("--to", dest="end", required=True, type=parse_utc)
    parser.add_argument("--asset", required=True, type=Asset)
    parser.add_argument("--provider", required=True, type=Provider)
    parser.add_argument("--market-type", required=True, type=MarketType)
    parser.add_argument("--interval", required=True, type=Interval)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    return parser


def resolve_historical_instrument(
    *,
    asset: Asset,
    provider: Provider,
    market_type: MarketType,
    interval: Interval,
    definitions: tuple[MarketInstrumentDefinition, ...] = MARKET_INSTRUMENTS,
) -> MarketInstrumentDefinition:
    """Resolve an instrument enabled for the Feature matrix capability."""

    return resolve_instrument(
        asset=asset,
        provider=provider,
        market_type=market_type,
        interval=interval,
        capability=InstrumentCapability.SCORING,
        definitions=definitions,
    )


async def run(
    args: argparse.Namespace, *, now: datetime | None = None
) -> HistoricalDataset:
    settings = get_settings()
    configure_logging(settings.log_level)
    definition = resolve_historical_instrument(
        asset=args.asset,
        provider=args.provider,
        market_type=args.market_type,
        interval=args.interval,
    )
    identity = HistoricalDatasetIdentity(
        instrument=definition.instrument,
        range=HistoricalDatasetRange(args.start, args.end),
    )
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    started = perf_counter()
    try:
        async with factory() as session:
            dataset = await read_historical_dataset(
                session=session,
                identity=identity,
                now=now or datetime.now(UTC),
                page_size=args.page_size,
            )
            matrix = build_historical_feature_matrix(dataset)
            if session.new or session.dirty or session.deleted:
                raise RuntimeError(
                    "historical validation attempted to mutate the session"
                )
    finally:
        await engine.dispose()
    logger.info(
        "Historical optimization dataset validated",
        extra={
            "structured_fields": {
                "asset": identity.instrument.asset.value,
                "provider": identity.instrument.provider.value,
                "market_type": identity.instrument.market_type.value,
                "interval": identity.instrument.interval.value,
                "from": identity.range.start.isoformat(),
                "to": identity.range.end.isoformat(),
                "expected_candles": dataset.expected_candle_count,
                "actual_candles": dataset.actual_candle_count,
                "gaps": dataset.gap_count,
                "feature_rows": len(matrix.rows),
                "first_feature_time": matrix.first_feature_time.isoformat()
                if matrix.first_feature_time
                else None,
                "last_feature_time": matrix.last_feature_time.isoformat()
                if matrix.last_feature_time
                else None,
                "fingerprint": dataset.fingerprint,
                "elapsed_seconds": perf_counter() - started,
            }
        },
    )
    return dataset


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
