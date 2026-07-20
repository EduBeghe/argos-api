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
from app.optimization.experiment_service import ExperimentExecution, execute_refinement
from app.optimization.historical_data import (
    HistoricalDatasetIdentity,
    HistoricalDatasetRange,
    read_historical_dataset,
)
from app.optimization.refinement import (
    BenchmarkSegmentResult,
    CandidateSegmentResult,
    RefinementBudget,
)
from app.scores.calculator import BaselineTechnicalParameters

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
        default=10,
    )
    parser.add_argument("--score-screening-top-k", type=positive_integer, default=12)
    parser.add_argument(
        "--max-score-configurations", type=positive_integer, default=200
    )
    parser.add_argument("--max-joint-candidates", type=positive_integer, default=1000)
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    return parser


async def run(
    args: argparse.Namespace, *, now: datetime | None = None
) -> ExperimentExecution:
    if args.force_recompute and not args.persist:
        raise ValueError("--force-recompute requires --persist")
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
            execution = await execute_refinement(
                session=session,
                dataset=dataset,
                initial_cash=args.initial_cash,
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
                budget=RefinementBudget(
                    score_screening_top_k=args.score_screening_top_k,
                    validation_top_k=args.validation_top_k,
                    max_score_configurations=args.max_score_configurations,
                    max_joint_candidates=args.max_joint_candidates,
                ),
                persist=args.persist,
                force_recompute=args.force_recompute,
                started_at=now or datetime.now(UTC),
            )
            if not args.persist and (session.new or session.dirty or session.deleted):
                raise RuntimeError(
                    "historical refinement attempted to mutate the session"
                )
    finally:
        await engine.dispose()
    if execution.reused is not None:
        logger.info(
            "Historical strategy refinement reused",
            extra={
                "structured_fields": {
                    "experiment_key": execution.reused.experiment_key,
                    "refinement_run_id": str(execution.reused.run_id),
                    "reused_existing_experiment": True,
                    "evaluations_reused": execution.reused.evaluations_reused,
                    "evaluations_computed": 0,
                    "compute_avoided_estimate": (
                        execution.reused.compute_avoided_estimate
                    ),
                    "result_digest": execution.reused.result_digest,
                }
            },
        )
        return execution
    result = execution.result
    if result is None:
        raise RuntimeError("refinement execution returned no result")
    counts = result.counts
    logger.info(
        "Historical strategy refinement completed",
        extra={
            "structured_fields": {
                "dataset_fingerprint": result.dataset_fingerprint,
                "experiment_key": result.experiment_key,
                "persisted": result.persisted,
                "reused_existing_experiment": result.reused_existing_experiment,
                "history_run_count": result.history_run_count,
                "compatible_profile_count": result.compatible_profile_count,
                "search_plan_counts": result.search_plan.counts
                if result.search_plan
                else {},
                "evaluations_reused": result.evaluations_reused,
                "evaluations_computed": result.evaluations_computed,
                "compute_avoided_estimate": result.compute_avoided_estimate,
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
                "refinement_counts": {
                    "score_configurations_generated": (
                        counts.score_configurations_generated
                    ),
                    "score_configurations_screened": (
                        counts.score_configurations_screened
                    ),
                    "score_configurations_retained": (
                        counts.score_configurations_retained
                    ),
                    "joint_candidates_evaluated": counts.joint_candidates_evaluated,
                    "validation_candidates_evaluated": (
                        counts.validation_candidates_evaluated
                    ),
                    "test_candidates_evaluated": counts.test_candidates_evaluated,
                    "simulation_count_total": counts.simulation_count_total,
                },
                "recommendation_status": result.recommendation_status.value,
                "production_baseline_parameters": _candidate_parameters(
                    result.production_baseline.training.candidate
                ),
                "recommended_parameters": (
                    _candidate_parameters(result.recommendation.candidate)
                    if result.recommendation
                    else None
                ),
                "segments": [
                    {
                        "name": segment.name.value,
                        "first_boundary": segment.first_boundary.isoformat(),
                        "last_boundary": segment.last_boundary.isoformat(),
                        "decision_count": segment.decision_count,
                    }
                    for segment in result.segments
                ],
                "production_baseline_segments": {
                    "training": _metric(result.production_baseline.training),
                    "validation": _metric(result.production_baseline.validation),
                    "test": _metric(result.production_baseline.test),
                },
                "buy_and_hold_segments": {
                    "training": _metric(result.buy_and_hold.training),
                    "validation": _metric(result.buy_and_hold.validation),
                    "test": _metric(result.buy_and_hold.test),
                },
                "recommended_validation": (
                    _metric(result.recommendation.validation)
                    if result.recommendation
                    else None
                ),
                "recommended_test": (
                    _metric(result.recommendation.test)
                    if result.recommendation
                    else None
                ),
                "ablation_summary": [
                    {
                        "disabled_signal": item.disabled_signal,
                        "comparison": "controlled_one_at_a_time",
                        "baseline_score_parameters": _score_parameters(
                            item.baseline_score_parameters
                        ),
                        "ablated_score_parameters": _score_parameters(
                            item.ablated_score_parameters
                        ),
                        "baseline_metrics": _metric(item.baseline),
                        "ablated_metrics": _metric(item.ablated),
                        "return_difference_from_baseline": str(
                            item.return_difference_from_baseline
                        ),
                        "drawdown_difference_from_baseline": str(
                            item.drawdown_difference_from_baseline
                        ),
                        "trade_difference_from_baseline": (
                            item.trade_difference_from_baseline
                        ),
                    }
                    for item in result.ablation_summary
                ],
                "recommendation_explanation": result.recommendation_explanation,
                "history_influence_explanation": result.history_influence_explanation,
                "evidence_classification": (
                    result.evidence_classification.value
                    if result.evidence_classification
                    else None
                ),
                "elapsed_seconds": perf_counter() - started,
            }
        },
    )
    return execution


def _candidate_parameters(value: object) -> dict[str, dict[str, str]]:
    from app.optimization.refinement import RefinementCandidate

    candidate = value
    if not isinstance(candidate, RefinementCandidate):
        raise TypeError("expected RefinementCandidate")
    return {
        "score": _score_parameters(candidate.score_parameters),
        "strategy": {
            "entry_threshold": str(candidate.strategy_parameters.entry_threshold),
            "exit_threshold": str(candidate.strategy_parameters.exit_threshold),
            "open_position_target_percentage": str(
                candidate.strategy_parameters.open_position_target_percentage
            ),
        },
    }


def _score_parameters(value: BaselineTechnicalParameters) -> dict[str, str]:
    return {
        "trend_weight": str(value.trend_weight),
        "rsi_weight": str(value.rsi_weight),
        "return_1_weight": str(value.return_1_weight),
        "return_12_weight": str(value.return_12_weight),
        "volume_confirmation_weight": str(value.volume_confirmation_weight),
    }


def _metric(
    value: CandidateSegmentResult | BenchmarkSegmentResult | None,
) -> dict[str, str | int] | None:
    if value is None:
        return None
    return {
        "total_return": str(value.total_return),
        "absolute_pnl": str(value.absolute_pnl),
        "max_drawdown": str(value.max_drawdown),
        "trade_count": value.trade_count,
        "fees_paid": str(value.fees_paid),
        "final_equity": str(value.final_equity),
        "return_over_drawdown": str(value.return_over_drawdown),
    }


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
