"""Emit a deterministic read-only development economic re-audit report."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from app.core.config import get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.ml.campaign.development_reaudit import (
    DEFAULT_MAXIMUM_CANDIDATES,
    MAXIMUM_CANDIDATES,
    DevelopmentReauditError,
    build_development_reaudit_report,
    report_json,
)

logger = logging.getLogger(__name__)


def maximum_candidates(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "maximum candidates must be an integer"
        ) from error
    if str(parsed) != value.strip() or not 1 <= parsed <= MAXIMUM_CANDIDATES:
        raise argparse.ArgumentTypeError(
            f"maximum candidates must be between 1 and {MAXIMUM_CANDIDATES}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exhaustive ML campaign near misses using development evidence"
        )
    )
    parser.add_argument("--campaign-id", type=UUID, required=True)
    parser.add_argument(
        "--maximum-candidates",
        type=maximum_candidates,
        default=DEFAULT_MAXIMUM_CANDIDATES,
    )
    parser.add_argument("--output", type=Path)
    return parser


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    engine = create_database_engine(settings)
    try:
        async with create_session_factory(engine)() as session:
            logger.info(
                "starting ML campaign development re-audit",
                extra={
                    "structured_fields": {
                        "campaign_id": str(args.campaign_id),
                        "maximum_candidates": args.maximum_candidates,
                    }
                },
            )
            report, has_invalid = await build_development_reaudit_report(
                session=session,
                campaign_id=args.campaign_id,
                maximum_candidates=args.maximum_candidates,
            )
            rendered = report_json(report)
            if args.output is None:
                print(rendered)
            else:
                try:
                    args.output.write_text(rendered + "\n", encoding="utf-8")
                except OSError as error:
                    raise DevelopmentReauditError(
                        "output_write_failed", "could not write the requested report"
                    ) from error
                print(f"Development re-audit report written to {args.output}")
            counts = report["counts"]
            logger.info(
                "completed ML campaign development re-audit",
                extra={
                    "structured_fields": {
                        "campaign_id": str(args.campaign_id),
                        "counts": counts,
                        "selected_count": counts["selected"]
                        if isinstance(counts, dict)
                        else None,
                    }
                },
            )
            return 1 if has_invalid else 0
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        code = asyncio.run(run(args))
    except DevelopmentReauditError as error:
        print(f"{error.code}: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    raise SystemExit(code)


if __name__ == "__main__":
    main()
