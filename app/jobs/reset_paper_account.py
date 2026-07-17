import argparse
import asyncio
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

from app.core.config import get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.paper.reset import reset_paper_account


def decimal_argument(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("initial cash must be a Decimal") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a new Paper Trading experiment")
    parser.add_argument(
        "--initial-cash", type=decimal_argument, default=Decimal("10000.00")
    )
    parser.add_argument("--reason")
    return parser.parse_args()


async def run(*, initial_cash: Decimal, reason: str | None) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            try:
                await reset_paper_account(
                    session=session,
                    started_at=datetime.now(UTC),
                    initial_cash=initial_cash,
                    reason=reason,
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    finally:
        await engine.dispose()


def main() -> None:
    args = parse_args()
    asyncio.run(run(initial_cash=args.initial_cash, reason=args.reason))


if __name__ == "__main__":
    main()
