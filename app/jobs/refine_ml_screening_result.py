import argparse
import asyncio
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.market.catalog import InstrumentCapability, resolve_instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.contract import DEVELOPMENT_END, DEVELOPMENT_START
from app.ml.dataset import MLExperimentDataset, build_ml_experiment_dataset
from app.ml.reader import read_ml_experiment_candles
from app.ml.refinement.contracts import MLRefinementRequest
from app.ml.refinement.persistence import resolve_source
from app.ml.refinement.service import execute_refinement, prepare_refinement
from app.ml.screening.engines import validate_engine_dependencies

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine one persisted ML screening result"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_request(path: Path) -> MLRefinementRequest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read refinement configuration: {error}") from error
    return MLRefinementRequest.model_validate(payload)


async def load_refinement_dataset(
    session: AsyncSession, request: MLRefinementRequest
) -> MLExperimentDataset:
    source_run, _ = await resolve_source(session, request)
    definition = resolve_instrument(
        asset=Asset(source_run.asset),
        provider=Provider(source_run.provider),
        market_type=MarketType(source_run.market_type),
        interval=Interval(source_run.interval),
        capability=InstrumentCapability.SCORING,
    )
    candles = await read_ml_experiment_candles(
        session=session,
        instrument=definition.instrument,
        start=DEVELOPMENT_START,
        end=DEVELOPMENT_END,
    )
    return build_ml_experiment_dataset(
        instrument=definition.instrument,
        columns=candles.columns,
        requested_end=DEVELOPMENT_END,
    )


async def run(args: argparse.Namespace) -> None:
    request = load_request(args.config)
    settings = get_settings()
    configure_logging(settings.log_level)
    validate_engine_dependencies()
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            dataset = await load_refinement_dataset(session, request)
            if args.dry_run:
                dry_run = await prepare_refinement(
                    session=session,
                    request=request,
                    dataset_fingerprint=dataset.metadata.fingerprint,
                )
                logger.info(
                    "ML refinement dry run completed",
                    extra={
                        "structured_fields": {
                            "experiment_key": dry_run.experiment_key,
                            "total_combinations": dry_run.plan.total_combinations,
                            "selected_combinations": len(dry_run.plan.evaluations),
                            "omitted_by_trial_budget": (
                                dry_run.plan.omitted_by_trial_budget
                            ),
                            "engines": [item.value for item in request.engines],
                            "objectives": [
                                item.value for item in request.prediction_objectives
                            ],
                            "durations": list(request.future_durations_seconds),
                            "fold_count": dry_run.fold_count,
                            "maximum_runtime_seconds": request.maximum_runtime_seconds,
                            "uncovered_engines": dry_run.plan.uncovered_engines,
                            "uncovered_objectives": dry_run.plan.uncovered_objectives,
                            "uncovered_durations": dry_run.plan.uncovered_durations,
                            "uncovered_feature_policies": (
                                dry_run.plan.uncovered_feature_policies
                            ),
                            "dry_run": True,
                        }
                    },
                )
                if session.new or session.dirty or session.deleted:
                    raise RuntimeError(
                        "dry run attempted to mutate the database session"
                    )
                return
            result = await execute_refinement(
                session=session, dataset=dataset, request=request
            )
            logger.info(
                "ML refinement command finished",
                extra={
                    "structured_fields": {
                        "run_id": str(result.run_id),
                        "experiment_key": result.experiment_key,
                        "reused_existing_run": result.reused_existing_run,
                        "evaluations_computed": result.evaluations_computed,
                        "evaluations_reused": result.evaluations_reused,
                        "evaluations_omitted": result.evaluations_omitted,
                        "budget_exhausted": result.budget_exhausted,
                        "termination_reason": result.termination_reason,
                    }
                },
            )
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
