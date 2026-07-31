import argparse
import asyncio
import logging
import socket
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.ml.campaign.coordinator import coordinate_next_campaign
from app.ml.loading import load_ml_development_dataset
from app.ml.refinement.contracts import MLRefinementRequest
from app.ml.refinement.identity import DatasetIdentityError, validate_dataset_identity
from app.ml.refinement.persistence import fail_run, resolve_source
from app.ml.refinement.queue import claim_next_run, fail_stale_runs
from app.ml.refinement.service import execute_refinement
from app.ml.screening.engines import validate_engine_dependencies

from .refine_ml_screening_result import load_refinement_identity_metadata

logger = logging.getLogger(__name__)


def heartbeat_interval_seconds(stale_after_seconds: int) -> float:
    return float(min(60, max(1, stale_after_seconds // 3)))


def bounded_integer(minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        parsed = int(value)
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"value must be between {minimum} and {maximum}"
            )
        return parsed

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process durable ML refinement work")
    parser.add_argument("--poll-seconds", type=bounded_integer(1, 300), default=5)
    parser.add_argument(
        "--stale-after-seconds", type=bounded_integer(300, 86_400), default=7200
    )
    parser.add_argument("--once", action="store_true")
    return parser


async def process_once(
    factory: async_sessionmaker[AsyncSession],
    *,
    worker_id: str,
    stale_after: timedelta,
) -> bool:
    async with factory() as session:
        await coordinate_next_campaign(session)
        stale_count = await fail_stale_runs(session, stale_after=stale_after)
        if stale_count:
            logger.warning(
                "Expired stale ML refinement leases",
                extra={"structured_fields": {"count": stale_count}},
            )
        run = await claim_next_run(session, worker_id=worker_id)
        if run is None:
            return False
        logger.info(
            "Claimed ML refinement run",
            extra={
                "structured_fields": {"run_id": str(run.id), "worker_id": worker_id}
            },
        )
        try:
            request_payload = dict(run.normalized_configuration)
            request_payload["notes"] = run.notes
            request = MLRefinementRequest.model_validate(request_payload)
            validate_engine_dependencies()
            identity_metadata = await load_refinement_identity_metadata(
                session, request
            )
            source_run, _ = await resolve_source(session, request)
            diagnostics = validate_dataset_identity(
                source=source_run, queued=run, rebuilt=identity_metadata
            )
            logger.info(
                "ML refinement dataset identity verified",
                extra={"structured_fields": diagnostics.structured_fields()},
            )
            development_dataset = await load_ml_development_dataset(
                session=session, instrument=identity_metadata.instrument
            )
            await execute_refinement(
                session=session,
                identity_metadata=identity_metadata,
                development_dataset=development_dataset,
                request=request,
                claimed_run=run,
                worker_id=worker_id,
                heartbeat_interval_seconds=heartbeat_interval_seconds(
                    int(stale_after.total_seconds())
                ),
            )
        except Exception as error:
            await fail_run(
                session, run.id, error, datetime.now(UTC), worker_id=worker_id
            )
            failure_fields: dict[str, object] = {
                "run_id": str(run.id),
                "worker_id": worker_id,
                "failure_type": type(error).__name__,
            }
            if (
                isinstance(error, DatasetIdentityError)
                and error.diagnostics is not None
            ):
                failure_fields.update(error.diagnostics.structured_fields())
            logger.exception(
                "ML refinement queue item failed",
                extra={"structured_fields": failure_fields},
            )
        return True


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    worker_id = f"{socket.gethostname()}-{uuid4()}"
    stale_after = timedelta(seconds=args.stale_after_seconds)
    try:
        while True:
            processed = await process_once(
                factory, worker_id=worker_id, stale_after=stale_after
            )
            if args.once:
                return
            if not processed:
                await asyncio.sleep(args.poll_seconds)
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
