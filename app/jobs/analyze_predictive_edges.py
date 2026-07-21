import argparse
import asyncio
import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from time import perf_counter

from app.core.config import get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.jobs.validate_historical_dataset import parse_utc
from app.market.catalog import InstrumentCapability, resolve_instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.feature_matrix import build_historical_feature_matrix
from app.optimization.historical_data import (
    HistoricalDatasetIdentity,
    HistoricalDatasetRange,
    read_historical_dataset,
)
from app.optimization.predictive_edge_service import (
    PredictiveEdgeExecution,
    find_reusable_predictive_edge_run,
    persist_predictive_edge_result,
    predictive_edge_experiment_key,
)
from app.optimization.predictive_edges import (
    DEFAULT_HORIZONS,
    EvidenceStatus,
    PredictiveEdgeConfiguration,
    analyze_predictive_edges,
    build_predictive_windows,
)
from app.optimization.walk_forward import WalkForwardConfiguration
from app.paper.config import FEE_RATE, SLIPPAGE_RATE, SimulationConfiguration

logger = logging.getLogger(__name__)
_DURATION = re.compile(r"^(?P<value>[1-9][0-9]*)(?P<unit>m|h|d)$")


def parse_horizon(value: str) -> timedelta:
    match = _DURATION.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "horizon must use a positive value plus m, h, or d"
        )
    amount = int(match.group("value"))
    unit = match.group("unit")
    return (
        timedelta(minutes=amount)
        if unit == "m"
        else timedelta(hours=amount)
        if unit == "h"
        else timedelta(days=amount)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze reproducible historical predictive edges"
    )
    parser.add_argument("--asset", required=True, type=Asset)
    parser.add_argument("--provider", required=True, type=Provider)
    parser.add_argument("--market-type", required=True, type=MarketType)
    parser.add_argument("--interval", required=True, type=Interval)
    parser.add_argument("--from", dest="start", required=True, type=parse_utc)
    parser.add_argument("--to", dest="end", required=True, type=parse_utc)
    parser.add_argument("--horizon", action="append", type=parse_horizon)
    parser.add_argument("--fee-rate", type=Decimal, default=FEE_RATE)
    parser.add_argument("--slippage-rate", type=Decimal, default=SLIPPAGE_RATE)
    parser.add_argument("--training-days", type=int, default=180)
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=60)
    parser.add_argument("--maximum-windows", type=int, default=6)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--persist", action="store_true")
    return parser


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    started_at = datetime.now(UTC)
    clock = perf_counter()
    definition = resolve_instrument(
        asset=args.asset,
        provider=args.provider,
        market_type=args.market_type,
        interval=args.interval,
        capability=InstrumentCapability.SCORING,
    )
    identity = HistoricalDatasetIdentity(
        definition.instrument, HistoricalDatasetRange(args.start, args.end)
    )
    configuration = PredictiveEdgeConfiguration(
        horizons=tuple(args.horizon or DEFAULT_HORIZONS),
        bins=args.bins,
        windows=WalkForwardConfiguration(
            training_days=args.training_days,
            validation_days=args.validation_days,
            test_days=args.test_days,
            step_days=args.step_days,
            maximum_windows=args.maximum_windows,
            maximum_candidates=1,
            minimum_valid_windows=1,
        ),
    )
    simulation = SimulationConfiguration(
        fee_rate=args.fee_rate, slippage_rate=args.slippage_rate
    )
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            dataset = await read_historical_dataset(
                session=session, identity=identity, now=started_at
            )
            matrix = build_historical_feature_matrix(dataset)
            windows = build_predictive_windows(
                dataset=dataset, matrix=matrix, configuration=configuration
            )
            key = predictive_edge_experiment_key(
                dataset_fingerprint=dataset.fingerprint,
                instrument=definition.instrument,
                range_start=args.start,
                range_end=args.end,
                configuration=configuration,
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
            )
            execution = await find_reusable_predictive_edge_run(session, key)
            if execution is None:
                result = analyze_predictive_edges(
                    matrix=matrix,
                    windows=windows,
                    simulation=simulation,
                    configuration=configuration,
                )
                if args.persist:
                    execution = await persist_predictive_edge_result(
                        session=session,
                        experiment_key=key,
                        instrument=definition.instrument,
                        range_start=args.start,
                        range_end=args.end,
                        dataset_fingerprint=dataset.fingerprint,
                        windows=windows,
                        result=result,
                        fee_rate=args.fee_rate,
                        slippage_rate=args.slippage_rate,
                        started_at=started_at,
                    )
                else:
                    execution = PredictiveEdgeExecution(None, False, result, 0, 0)
            if execution is None:
                raise RuntimeError("predictive-edge execution was not initialized")
    finally:
        await engine.dispose()
    completed_result = execution.result
    classifications = (
        tuple(completed_result.stability_edges) if completed_result else ()
    )
    logger.info(
        "Predictive Edge Map completed",
        extra={
            "structured_fields": {
                "predictive_edge_run_id": str(execution.run_id)
                if execution.run_id
                else None,
                "reused_existing_experiment": execution.reused_existing_experiment,
                "instrument": [
                    definition.instrument.asset.value,
                    definition.instrument.provider.value,
                    definition.instrument.market_type.value,
                    definition.instrument.interval.value,
                ],
                "dataset_fingerprint": dataset.fingerprint,
                "windows_generated": len(windows),
                "predictors_evaluated": 14,
                "horizons_evaluated": len(configuration.horizons),
                "conditions_evaluated": len(configuration.conditions),
                "aggregate_metrics_computed": len(completed_result.window_edges)
                if completed_result
                else 0,
                "metrics_reused": execution.metrics_reused,
                "compute_avoided_estimate": execution.compute_avoided_estimate,
                "promising_positive_edges": sum(
                    item.classification is EvidenceStatus.PROMISING_POSITIVE
                    for item in classifications
                ),
                "promising_negative_edges": sum(
                    item.classification is EvidenceStatus.PROMISING_NEGATIVE
                    for item in classifications
                ),
                "unstable_edges": sum(
                    item.classification is EvidenceStatus.UNSTABLE
                    for item in classifications
                ),
                "no_detectable_edges": sum(
                    item.classification is EvidenceStatus.NONE
                    for item in classifications
                ),
                "hypotheses_generated": len(completed_result.hypotheses)
                if completed_result
                else 0,
                "elapsed_seconds": perf_counter() - clock,
            }
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()

