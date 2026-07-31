import argparse
import asyncio
import json
from collections.abc import Sequence
from uuid import UUID

from app.core.config import get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.ml.campaign.contracts import (
    CAMPAIGN_PRESET,
    DEFAULT_MAXIMUM_EVALUATIONS,
    DEFAULT_MAXIMUM_RUNTIME_SECONDS,
    CampaignContract,
    CampaignRequest,
)
from app.ml.campaign.service import enqueue_campaign, preview_campaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enqueue a durable exhaustive OHLCV ML search campaign"
    )
    parser.add_argument(
        "--source-screening-run-id", action="append", type=UUID, required=True
    )
    parser.add_argument("--preset", choices=(CAMPAIGN_PRESET,), default=CAMPAIGN_PRESET)
    parser.add_argument(
        "--maximum-evaluations", type=int, default=DEFAULT_MAXIMUM_EVALUATIONS
    )
    parser.add_argument(
        "--maximum-runtime-seconds", type=int, default=DEFAULT_MAXIMUM_RUNTIME_SECONDS
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--enqueue", action="store_true")
    return parser


async def run(args: argparse.Namespace) -> None:
    request = CampaignRequest(
        source_screening_run_ids=tuple(args.source_screening_run_id),
        preset=args.preset,
        maximum_evaluations=args.maximum_evaluations,
        maximum_runtime_seconds=args.maximum_runtime_seconds,
    )
    engine = create_database_engine(get_settings())
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            sources, family_count, key = await preview_campaign(session, request)
            output: dict[str, object] = {
                "preset": args.preset,
                "experiment_key": key,
                "sources": [
                    {
                        "run_id": item.source_run_id,
                        "asset": item.asset,
                        "provider": item.provider,
                        "market_type": item.market_type,
                        "interval": item.interval,
                        "dataset_fingerprint": item.dataset_fingerprint,
                    }
                    for item in sources
                ],
                "family_count": family_count,
                "minimum_stage_1_evaluations": family_count,
                "contract": CampaignContract().model_dump(mode="json"),
                "limits": {
                    "maximum_evaluations": request.maximum_evaluations,
                    "maximum_runtime_seconds": request.maximum_runtime_seconds,
                },
            }
            if args.enqueue:
                result = await enqueue_campaign(session, request)
                output.update(
                    {
                        "campaign_id": str(result.campaign.id),
                        "status": result.campaign.status,
                        "created": result.created,
                        "location": (
                            f"/optimization/ml-search-campaigns/{result.campaign.id}"
                        ),
                    }
                )
            print(json.dumps(output, sort_keys=True))
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()

