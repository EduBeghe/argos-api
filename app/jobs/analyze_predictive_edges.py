import argparse
import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, replace
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
from app.optimization.edge_guided import (
    DEFAULT_MAX_EDGE_GUIDED_CANDIDATES,
    edge_guided_summary,
    evaluate_edge_guided_candidates,
)
from app.optimization.edge_guided_memory import (
    EdgeGuidedExecution,
    edge_guided_experiment_key,
    find_reusable_edge_guided_run,
    find_reusable_weekly_research_run,
    persist_edge_guided_result,
)
from app.optimization.feature_matrix import build_historical_feature_matrix
from app.optimization.historical_data import (
    HistoricalDatasetIdentity,
    HistoricalDatasetRange,
    read_historical_dataset,
    read_historical_dataset_revision,
)
from app.optimization.predictive_edge_service import (
    PredictiveEdgeExecution,
    find_reusable_predictive_edge_run,
    persist_predictive_edge_result,
    predictive_edge_experiment_key,
)
from app.optimization.predictive_edges import (
    DEFAULT_HORIZONS,
    PREDICTORS,
    EvidenceStatus,
    PredictiveEdgeConfiguration,
    analyze_predictive_edges,
    build_predictive_windows,
)
from app.optimization.walk_forward import WalkForwardConfiguration
from app.optimization.weekly_research import weekly_research_reuse_key
from app.paper.config import (
    DEFAULT_INITIAL_CASH,
    FEE_RATE,
    SLIPPAGE_RATE,
    SimulationConfiguration,
)

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
    parser.add_argument("--edge-guided-candidates", action="store_true")
    parser.add_argument(
        "--max-edge-guided-candidates",
        type=int,
        default=DEFAULT_MAX_EDGE_GUIDED_CANDIDATES,
    )
    parser.add_argument("--initial-cash", type=Decimal, default=DEFAULT_INITIAL_CASH)
    return parser


async def run(
    args: argparse.Namespace, *, now: datetime | None = None
) -> PredictiveEdgeExecution | EdgeGuidedExecution:
    settings = get_settings()
    configure_logging(settings.log_level)
    started_at = now or datetime.now(UTC)
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
    guided_result = None
    guided_execution: EdgeGuidedExecution | None = None
    try:
        async with factory() as session:
            if args.persist and getattr(args, "edge_guided_candidates", False):
                revision = await read_historical_dataset_revision(
                    session=session, identity=identity, now=started_at
                )
                preflight_key = weekly_research_reuse_key(
                    identity=identity,
                    revision=revision,
                    configuration=configuration,
                    simulation=simulation,
                    initial_cash=getattr(args, "initial_cash", DEFAULT_INITIAL_CASH),
                    maximum_candidates=getattr(
                        args,
                        "max_edge_guided_candidates",
                        DEFAULT_MAX_EDGE_GUIDED_CANDIDATES,
                    ),
                    boundary=args.end,
                )
                preflight = await find_reusable_weekly_research_run(
                    session, preflight_key
                )
                if preflight is not None:
                    logger.info(
                        "Predictive Edge Map completed",
                        extra={
                            "structured_fields": {
                                "edge_guided_run_id": str(preflight.run_id),
                                "reused_existing_experiment": True,
                                "metrics_reused": preflight.result.funnel.get(
                                    "edges_evaluated", 0
                                ),
                                "evaluations_reused": preflight.evaluations_reused,
                                "evaluations_computed": 0,
                                "compute_avoided_estimate": (
                                    preflight.compute_avoided_estimate
                                ),
                                "research_outcome": preflight.result.outcome.value,
                                "funnel": preflight.result.funnel,
                                "timings": preflight.result.timings,
                                "result": edge_guided_summary(preflight.result),
                            }
                        },
                    )
                    return preflight
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
            execution = (
                await find_reusable_predictive_edge_run(session, key)
                if args.persist
                else None
            )
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
                    if session.new or session.dirty or session.deleted:
                        raise RuntimeError(
                            "non-persisted predictive-edge analysis mutated the session"
                        )
                    execution = PredictiveEdgeExecution(None, False, result, 0, 0)
            if execution is None:
                raise RuntimeError("predictive-edge execution was not initialized")
            analysis_result = execution.result
            if getattr(args, "edge_guided_candidates", False):
                if analysis_result is None:
                    analysis_result = analyze_predictive_edges(
                        matrix=matrix,
                        windows=windows,
                        simulation=simulation,
                        configuration=configuration,
                    )
                initial_cash = getattr(args, "initial_cash", DEFAULT_INITIAL_CASH)
                maximum_candidates = getattr(
                    args,
                    "max_edge_guided_candidates",
                    DEFAULT_MAX_EDGE_GUIDED_CANDIDATES,
                )
                guided_key = None
                if args.persist:
                    if execution.run_id is None:
                        raise RuntimeError("persisted predictive-edge run is required")
                    guided_key = edge_guided_experiment_key(
                        predictive_edge_run_id=execution.run_id,
                        dataset_fingerprint=dataset.fingerprint,
                        instrument=definition.instrument,
                        range_start=args.start,
                        range_end=args.end,
                        windows=configuration.windows,
                        initial_cash=initial_cash,
                        simulation=simulation,
                        decision_interval=configuration.decision_interval,
                        maximum_candidates=maximum_candidates,
                        hypothesis_fingerprints=tuple(
                            item.hypothesis_id for item in analysis_result.hypotheses
                        ),
                    )
                    guided_execution = await find_reusable_edge_guided_run(
                        session, guided_key
                    )
                if guided_execution is None:
                    guided_result = evaluate_edge_guided_candidates(
                        dataset=dataset,
                        feature_matrix=matrix,
                        hypotheses=analysis_result.hypotheses,
                        initial_cash=initial_cash,
                        simulation=simulation,
                        decision_interval=configuration.decision_interval,
                        maximum_candidates=maximum_candidates,
                        windows=windows,
                        thresholds=configuration.thresholds,
                    )
                    classifications = tuple(analysis_result.stability_edges)
                    guided_result = replace(
                        guided_result,
                        funnel={
                            "predictors_evaluated": len(PREDICTORS),
                            "edges_evaluated": len(analysis_result.window_edges),
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
                            **guided_result.funnel,
                        },
                    )
                    if args.persist:
                        if execution.run_id is None or guided_key is None:
                            raise RuntimeError(
                                "edge-guided persistence identity missing"
                            )
                        guided_execution = await persist_edge_guided_result(
                            session=session,
                            experiment_key=guided_key,
                            predictive_edge_run_id=execution.run_id,
                            instrument=definition.instrument,
                            range_start=args.start,
                            range_end=args.end,
                            initial_cash=initial_cash,
                            simulation=simulation,
                            decision_interval=configuration.decision_interval,
                            started_at=started_at,
                            result=guided_result,
                            reuse_key=preflight_key,
                            dataset_revision=asdict(revision),
                        )
                    else:
                        if session.new or session.dirty or session.deleted:
                            raise RuntimeError(
                                "non-persisted edge-guided evaluation "
                                "mutated the session"
                            )
                        guided_execution = EdgeGuidedExecution(
                            None,
                            "",
                            False,
                            guided_result,
                            0,
                            guided_result.evaluated_count,
                            0,
                        )
                guided_result = guided_execution.result
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
                "predictors_evaluated": len(PREDICTORS),
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
                "edge_guided_candidates": (
                    {
                        **edge_guided_summary(guided_result),
                        "edge_guided_run_id": str(guided_execution.run_id)
                        if guided_execution.run_id
                        else None,
                        "experiment_key": guided_execution.experiment_key,
                        "reused_existing_experiment": (
                            guided_execution.reused_existing_experiment
                        ),
                        "evaluations_reused": guided_execution.evaluations_reused,
                        "evaluations_computed": guided_execution.evaluations_computed,
                        "compute_avoided_estimate": (
                            guided_execution.compute_avoided_estimate
                        ),
                    }
                    if guided_result is not None and guided_execution is not None
                    else None
                ),
            }
        },
    )
    return execution


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
