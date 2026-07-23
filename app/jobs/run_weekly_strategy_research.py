import argparse
import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from app.core.config import get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.market.catalog import enabled_scoring_instruments
from app.market.enums import Asset
from app.optimization.edge_guided import DEFAULT_MAX_EDGE_GUIDED_CANDIDATES
from app.optimization.historical_data import (
    HistoricalDatasetIdentity,
    HistoricalDatasetRange,
)
from app.optimization.predictive_edges import PredictiveEdgeConfiguration
from app.optimization.weekly_research import (
    execute_weekly_instrument_research,
    release_weekly_research_lock,
    try_weekly_research_lock,
    weekly_boundary,
)
from app.paper.config import DEFAULT_INITIAL_CASH, SimulationConfiguration

DEFAULT_LOOKBACK_DAYS = 730
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded weekly Strategy discovery for scoring instruments"
    )
    parser.add_argument("--asset", action="append", type=Asset)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument(
        "--maximum-candidates",
        type=int,
        default=DEFAULT_MAX_EDGE_GUIDED_CANDIDATES,
    )
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


async def run(
    args: argparse.Namespace, *, now: datetime | None = None
) -> dict[str, object]:
    settings = get_settings()
    configure_logging(settings.log_level)
    current = now or datetime.now(UTC)
    boundary = weekly_boundary(current)
    range_end = boundary.replace(
        minute=boundary.minute - boundary.minute % 5, second=0, microsecond=0
    )
    if args.lookback_days <= 0:
        raise ValueError("lookback-days must be positive")
    instruments = tuple(
        item
        for item in enabled_scoring_instruments()
        if not args.asset or item.asset in args.asset
    )
    if not instruments:
        raise ValueError("no scoring-enabled catalog instrument matched --asset")
    configuration = PredictiveEdgeConfiguration()
    simulation = SimulationConfiguration()
    plan = {
        "weekly_boundary": boundary.isoformat(),
        "range_start": (range_end - timedelta(days=args.lookback_days)).isoformat(),
        "range_end": range_end.isoformat(),
        "maximum_candidates": args.maximum_candidates,
        "instruments": [
            [
                item.asset.value,
                item.provider.value,
                item.market_type.value,
                item.interval.value,
            ]
            for item in instruments
        ],
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        logger.info(
            "Weekly Strategy research dry run", extra={"structured_fields": plan}
        )
        return {**plan, "status": "validated", "results": []}

    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    try:
        async with factory() as lock_session:
            if not await try_weekly_research_lock(lock_session):
                return {**plan, "status": "already_running", "results": []}
            try:
                for instrument in instruments:
                    try:
                        async with factory() as session:
                            item = await execute_weekly_instrument_research(
                                session=session,
                                identity=HistoricalDatasetIdentity(
                                    instrument,
                                    HistoricalDatasetRange(
                                        range_end - timedelta(days=args.lookback_days),
                                        range_end,
                                    ),
                                ),
                                configuration=configuration,
                                simulation=simulation,
                                initial_cash=DEFAULT_INITIAL_CASH,
                                maximum_candidates=args.maximum_candidates,
                                boundary=boundary,
                                persist=True,
                                force_recompute=args.force_recompute,
                            )
                            results.append(
                                {
                                    "instrument": [
                                        instrument.asset.value,
                                        instrument.provider.value,
                                        instrument.market_type.value,
                                        instrument.interval.value,
                                    ],
                                    "edge_guided_run_id": str(item.execution.run_id),
                                    "predictive_edge_run_id": (
                                        str(item.predictive_edge_run_id)
                                        if item.predictive_edge_run_id is not None
                                        else None
                                    ),
                                    "reused": item.reused,
                                    "outcome": item.execution.result.outcome.value,
                                    "strategy_research_outcome": (
                                        item.execution.result.outcome.value
                                    ),
                                    "predictive_edge_v4_conclusion": (
                                        item.diagnostics.get(
                                            "predictive_edge_v4_conclusion"
                                        )
                                    ),
                                    "regime_conditioned_v41_conclusion": (
                                        item.diagnostics.get(
                                            "regime_conditioned_v41_conclusion"
                                        )
                                    ),
                                    "v41_stage_counters": item.diagnostics.get(
                                        "regime_conditioned_stage_counters"
                                    ),
                                    "v41_actionable_near_miss_count": (
                                        item.diagnostics.get(
                                            "actionable_near_misses", 0
                                        )
                                    ),
                                    "v41_diagnostic_near_miss_count": (
                                        item.diagnostics.get(
                                            "diagnostic_near_misses", 0
                                        )
                                    ),
                                    "targeted_refinement_conclusion": (
                                        item.diagnostics.get(
                                            "targeted_refinement_conclusion"
                                        )
                                    ),
                                    "targeted_refinement_summary": (
                                        item.diagnostics.get(
                                            "targeted_refinement_summary"
                                        )
                                    ),
                                    "targeted_candidates_url": item.diagnostics.get(
                                        "targeted_candidates_url"
                                    ),
                                    "targeted_confirmed_candidates": (
                                        item.diagnostics.get(
                                            "targeted_confirmed_candidates", 0
                                        )
                                    ),
                                    "best_targeted_candidate": item.diagnostics.get(
                                        "best_targeted_candidate"
                                    ),
                                    "hypotheses_generated": (
                                        item.execution.result.funnel.get(
                                            "hypotheses_generated", 0
                                        )
                                    ),
                                    "complete_strategies_generated": (
                                        item.execution.result.generated_count
                                    ),
                                    "confirmed_candidates": (
                                        item.execution.result.funnel.get(
                                            "confirmed_candidates", 0
                                        )
                                    ),
                                    "compute_avoided_estimate": (
                                        item.work.simulations_avoided
                                    ),
                                    "total_seconds": item.timings["total"],
                                    "diagnostic_url": (
                                        "/optimization/predictive-edge-runs/"
                                        f"{item.predictive_edge_run_id}/diagnostics"
                                        if item.predictive_edge_run_id is not None
                                        else None
                                    ),
                                    "actionable_near_misses_url": (
                                        "/optimization/predictive-edge-runs/"
                                        f"{item.predictive_edge_run_id}/near-misses?kind=actionable"
                                        if item.predictive_edge_run_id is not None
                                        else None
                                    ),
                                    "diagnostic_near_misses_url": (
                                        "/optimization/predictive-edge-runs/"
                                        f"{item.predictive_edge_run_id}/near-misses?kind=diagnostic"
                                        if item.predictive_edge_run_id is not None
                                        else None
                                    ),
                                    "outcome_explanation": (
                                        item.execution.result.outcome_explanation
                                    ),
                                    "funnel": item.execution.result.funnel,
                                    "work": asdict(item.work),
                                    "timings": item.timings,
                                    "predictive_edge_diagnostics": item.diagnostics,
                                }
                            )
                    except Exception as error:
                        failures.append(
                            {
                                "asset": instrument.asset.value,
                                "error": type(error).__name__,
                                "message": str(error),
                            }
                        )
                        logger.exception(
                            "Weekly Strategy research instrument failed",
                            extra={"structured_fields": failures[-1]},
                        )
            finally:
                await release_weekly_research_lock(lock_session)
    finally:
        await engine.dispose()
    status = "completed" if not failures else "failed"
    summary = {**plan, "status": status, "results": results, "failures": failures}
    logger.info(
        "Weekly Strategy research finished",
        extra={"structured_fields": json.loads(json.dumps(summary, default=str))},
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    summary = asyncio.run(run(build_parser().parse_args(argv)))
    if summary["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
