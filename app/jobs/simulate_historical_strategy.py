import argparse
import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

from app.core.config import get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.jobs.validate_historical_dataset import (
    parse_utc,
    resolve_historical_instrument,
)
from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.historical_data import (
    HistoricalDatasetIdentity,
    HistoricalDatasetRange,
    read_historical_dataset,
)
from app.optimization.simulation import (
    HistoricalSimulationResult,
    simulate_historical_strategy,
)

logger = logging.getLogger(__name__)


def parse_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("must be a decimal number") from error
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError("must be a finite decimal number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate baseline_long_only on historical evidence"
    )
    parser.add_argument("--from", dest="start", required=True, type=parse_utc)
    parser.add_argument("--to", dest="end", required=True, type=parse_utc)
    parser.add_argument("--asset", required=True, type=Asset)
    parser.add_argument("--provider", required=True, type=Provider)
    parser.add_argument("--market-type", required=True, type=MarketType)
    parser.add_argument("--interval", required=True, type=Interval)
    parser.add_argument("--initial-cash", required=True, type=parse_decimal)
    parser.add_argument("--fee-rate", required=True, type=parse_decimal)
    parser.add_argument("--slippage-rate", required=True, type=parse_decimal)
    return parser


async def run(
    args: argparse.Namespace, *, now: datetime | None = None
) -> HistoricalSimulationResult:
    settings = get_settings()
    configure_logging(settings.log_level)
    definition = resolve_historical_instrument(
        asset=args.asset,
        provider=args.provider,
        market_type=args.market_type,
        interval=args.interval,
    )
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            dataset = await read_historical_dataset(
                session=session,
                identity=HistoricalDatasetIdentity(
                    definition.instrument, HistoricalDatasetRange(args.start, args.end)
                ),
                now=now or datetime.now(UTC),
            )
            result = simulate_historical_strategy(
                dataset=dataset,
                initial_cash=args.initial_cash,
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
            )
            if session.new or session.dirty or session.deleted:
                raise RuntimeError(
                    "historical simulation attempted to mutate the session"
                )
    finally:
        await engine.dispose()
    logger.info(
        "Historical strategy simulation completed",
        extra={
            "structured_fields": {
                "dataset_fingerprint": result.dataset_fingerprint,
                "asset": result.instrument.asset.value,
                "provider": result.instrument.provider.value,
                "market_type": result.instrument.market_type.value,
                "interval": result.instrument.interval.value,
                "score_model_name": result.score_model_name,
                "score_model_version": result.score_model_version,
                "strategy_name": result.strategy_name,
                "strategy_version": result.strategy_version,
                "configuration_version": result.configuration_version,
                "account_currency": result.account_currency,
                "fee_rate": str(result.fee_rate),
                "slippage_rate": str(result.slippage_rate),
                "decision_interval_seconds": result.decision_interval.total_seconds(),
                "initial_cash": str(result.initial_cash),
                "final_cash": str(result.final_cash),
                "final_equity": str(result.final_equity),
                "absolute_pnl": str(result.absolute_pnl),
                "total_return": str(result.total_return),
                "max_drawdown": str(result.max_drawdown),
                "fees_paid": str(result.fees_paid),
                "trade_count": result.trade_count,
                "decision_count": result.decision_count,
                "first_boundary": result.first_boundary.isoformat(),
                "last_boundary": result.last_boundary.isoformat(),
            }
        },
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()

