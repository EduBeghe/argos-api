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
from app.jobs.simulate_historical_strategy import parse_decimal
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
from app.optimization.refinement import (
    DEFAULT_VALIDATION_TOP_K,
    CandidateSegmentResult,
    HistoricalRefinementResult,
    refine_historical_dataset,
)

logger = logging.getLogger(__name__)


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine baseline_long_only on historical evidence"
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
    parser.add_argument(
        "--validation-top-k",
        type=positive_integer,
        default=DEFAULT_VALIDATION_TOP_K,
    )
    return parser


async def run(
    args: argparse.Namespace, *, now: datetime | None = None
) -> HistoricalRefinementResult:
    started = perf_counter()
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
            result = refine_historical_dataset(
                dataset=dataset,
                initial_cash=args.initial_cash,
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
                validation_top_k=args.validation_top_k,
            )
            if session.new or session.dirty or session.deleted:
                raise RuntimeError(
                    "historical refinement attempted to mutate the session"
                )
    finally:
        await engine.dispose()
    logger.info(
        "Historical strategy refinement completed",
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
                "initial_cash": str(result.initial_cash),
                "account_currency": result.account_currency,
                "fee_rate": str(result.fee_rate),
                "slippage_rate": str(result.slippage_rate),
                "decision_interval_seconds": result.decision_interval.total_seconds(),
                "candidate_count": result.total_candidate_count,
                "validation_candidate_count": result.validation_candidate_count,
                "recommendation_status": result.recommendation_status.value,
                "recommended_parameters": _parameters(result),
                "segments": [
                    {
                        "name": segment.name.value,
                        "first_boundary": segment.first_boundary.isoformat(),
                        "last_boundary": segment.last_boundary.isoformat(),
                        "decision_count": segment.decision_count,
                    }
                    for segment in result.segments
                ],
                "baseline_segments": {
                    "training": _metric(result.baseline.training),
                    "validation": _metric(result.baseline.validation),
                    "test": _metric(result.baseline.test),
                },
                "recommended_validation": (
                    _metric(result.recommendation.validation)
                    if result.recommendation
                    else None
                ),
                "recommended_test": _metric(result.recommended_test),
                "recommendation_explanation": result.recommendation_explanation,
                "elapsed_seconds": perf_counter() - started,
            }
        },
    )
    return result


def _parameters(result: HistoricalRefinementResult) -> dict[str, str] | None:
    value = result.recommended_parameters
    if value is None:
        return None
    return {
        "entry_threshold": str(value.entry_threshold),
        "exit_threshold": str(value.exit_threshold),
        "open_position_target_percentage": str(value.open_position_target_percentage),
    }


def _metric(value: CandidateSegmentResult | None) -> dict[str, str | int] | None:
    if value is None:
        return None
    return {
        "total_return": str(value.total_return),
        "max_drawdown": str(value.max_drawdown),
        "trade_count": value.trade_count,
        "fees_paid": str(value.fees_paid),
        "final_equity": str(value.final_equity),
    }


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
