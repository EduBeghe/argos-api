from datetime import datetime
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.paper.config import DEFAULT_INITIAL_CASH, VERSION_1, validate_initial_cash
from app.paper.models import PaperRunStatus, PaperTradingRun
from app.paper.persistence import create_run, get_active_run, run_domain


async def reset_paper_account(
    *,
    session: AsyncSession,
    started_at: datetime,
    initial_cash: Decimal = DEFAULT_INITIAL_CASH,
    reason: str | None = None,
) -> PaperTradingRun:
    """Stop the active experiment and stage a new run; caller owns transaction."""
    cash = validate_initial_cash(initial_cash)
    normalized_reason = reason.strip() if reason else None
    current = await get_active_run(
        session, currency=VERSION_1.account_currency, lock=True
    )
    predecessor = None
    if current is not None:
        predecessor = current.id
        current.status = PaperRunStatus.STOPPED.value
        current.stopped_at = started_at
        current.stop_reason = normalized_reason
        await session.flush()
    created = await create_run(
        session,
        initial_cash=cash,
        configuration=VERSION_1,
        started_at=started_at,
        predecessor_run_id=predecessor,
    )
    return run_domain(created)
