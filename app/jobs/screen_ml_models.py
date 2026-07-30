import argparse
import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

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
from app.market.enums import Asset
from app.ml.contract import FUTURE_DURATIONS, temporal_fold_definitions
from app.ml.loading import load_canonical_ml_experiment_dataset
from app.ml.screening.config import (
    ALL_ENGINES,
    ModelScreeningConfiguration,
    ScreeningEngine,
)
from app.ml.screening.engines import library_versions, validate_engine_dependencies
from app.ml.screening.service import ScreeningExecutionResult, screen_models

logger = logging.getLogger(__name__)


def _duration(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("duration must be an integer") from error
    allowed = {int(item.total_seconds()) for item in FUTURE_DURATIONS}
    if seconds not in allowed:
        raise argparse.ArgumentTypeError("duration is outside the ML contract")
    return seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen regression engines on chronological ML folds"
    )
    parser.add_argument("--asset", required=True, type=Asset)
    parser.add_argument("--engine", choices=(*ScreeningEngine, "all"), default="all")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--future-duration-seconds", action="append", type=_duration)
    group.add_argument("--all-future-durations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    return parser


def _resolve_asset(asset: Asset) -> MarketInstrumentDefinition:
    matches = [
        item
        for item in MARKET_INSTRUMENTS
        if item.asset == asset and item.scoring_enabled
    ]
    if len(matches) != 1:
        raise ValueError("asset must resolve to exactly one scoring-enabled instrument")
    item = matches[0]
    return resolve_instrument(
        asset=item.asset,
        provider=item.provider,
        market_type=item.market_type,
        interval=item.interval,
        capability=InstrumentCapability.SCORING,
    )


async def run(args: argparse.Namespace) -> ScreeningExecutionResult | None:
    if args.dry_run and args.force_recompute:
        raise ValueError("--dry-run and --force-recompute cannot be combined")
    settings = get_settings()
    configure_logging(settings.log_level)
    definition = _resolve_asset(args.asset)
    engines = ALL_ENGINES if args.engine == "all" else (ScreeningEngine(args.engine),)
    durations = (
        FUTURE_DURATIONS
        if args.all_future_durations
        else tuple(
            timedelta(seconds=value)
            for value in dict.fromkeys(args.future_duration_seconds)
        )
    )
    validate_engine_dependencies()
    versions = library_versions()
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            dataset = await load_canonical_ml_experiment_dataset(
                session=session, instrument=definition.instrument
            )
            planned = len(engines) * len(durations) * len(temporal_fold_definitions())
            logger.info(
                "ML model screening planned",
                extra={
                    "structured_fields": {
                        "dataset_fingerprint": dataset.metadata.fingerprint,
                        "asset": args.asset.value,
                        "engines": [item.value for item in engines],
                        "duration_count": len(durations),
                        "fold_count": len(dataset.folds),
                        "planned_evaluation_count": planned,
                        "library_versions": versions,
                        "dry_run": args.dry_run,
                    }
                },
            )
            if args.dry_run:
                if session.new or session.dirty or session.deleted:
                    raise RuntimeError("dry-run attempted to mutate the session")
                return None
            return await screen_models(
                session=session,
                dataset=dataset,
                engines=engines,
                durations=durations,
                configuration=ModelScreeningConfiguration(
                    thread_count=settings.ml_screening_thread_count
                ),
                force_recompute=args.force_recompute,
                now=datetime.now(UTC),
            )
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> None:
    result = asyncio.run(run(build_parser().parse_args(argv)))
    if result is not None:
        logger.info(
            "ML screening command finished",
            extra={
                "structured_fields": {
                    "run_id": str(result.run_id),
                    "experiment_key": result.experiment_key,
                    "reused_existing_run": result.reused_existing_run,
                    "evaluations_computed": result.evaluations_computed,
                    "evaluations_reused": result.evaluations_reused,
                    "results_generated": result.results_generated,
                    "pareto_retained_count": result.pareto_retained_count,
                }
            },
        )


if __name__ == "__main__":
    main()
