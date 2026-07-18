from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select, true
from sqlalchemy.ext.asyncio import AsyncSession

from app.decisions.generation import PortfolioState
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord
from app.paper.config import DEFAULT_ACCOUNT_CURRENCY
from app.paper.models import PaperPositionStatus
from app.paper.persistence import (
    PaperPositionRecord,
    PaperTradingRunRecord,
    get_active_run,
    get_open_position,
)


class PaperPortfolioUnavailableError(RuntimeError):
    """Raised when no active persisted paper experiment exists."""


class PaperPortfolioValuationError(RuntimeError):
    """Raised when persisted data cannot safely value a Paper portfolio."""


@dataclass(frozen=True, slots=True)
class PositionMark:
    position_id: UUID
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    quantity: Decimal
    average_entry_price: Decimal
    total_entry_notional: Decimal
    accumulated_entry_fees: Decimal
    mark_price: Decimal
    mark_candle_open_time: datetime


@dataclass(frozen=True, slots=True)
class ValuedPosition:
    position_id: UUID
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    quantity: Decimal
    average_entry_price: Decimal
    cost_basis: Decimal
    mark_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_return: Decimal | None
    mark_candle_open_time: datetime


@dataclass(frozen=True, slots=True)
class PaperPortfolioValuation:
    run_id: UUID
    status: str
    account_currency: str
    initial_cash: Decimal
    current_cash: Decimal
    positions_market_value: Decimal
    equity: Decimal
    total_pnl: Decimal
    total_return: Decimal | None
    valued_at: datetime
    positions: tuple[ValuedPosition, ...]


def calculate_portfolio_valuation(
    *,
    run: PaperTradingRunRecord,
    marks: list[PositionMark],
    valued_at: datetime,
) -> PaperPortfolioValuation:
    if valued_at.tzinfo is None:
        raise ValueError("valued_at must be timezone-aware")
    positions: list[ValuedPosition] = []
    for mark in marks:
        market_value = mark.quantity * mark.mark_price
        cost_basis = mark.total_entry_notional + mark.accumulated_entry_fees
        unrealized_pnl = market_value - cost_basis
        positions.append(
            ValuedPosition(
                position_id=mark.position_id,
                asset=mark.asset,
                provider=mark.provider,
                market_type=mark.market_type,
                interval=mark.interval,
                quantity=mark.quantity,
                average_entry_price=mark.average_entry_price,
                cost_basis=cost_basis,
                mark_price=mark.mark_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_return=(
                    unrealized_pnl / cost_basis if cost_basis > 0 else None
                ),
                mark_candle_open_time=mark.mark_candle_open_time,
            )
        )
    positions_market_value = sum(
        (position.market_value for position in positions), start=Decimal(0)
    )
    equity = run.current_cash + positions_market_value
    total_pnl = equity - run.initial_cash
    return PaperPortfolioValuation(
        run_id=run.id,
        status=run.status,
        account_currency=run.account_currency,
        initial_cash=run.initial_cash,
        current_cash=run.current_cash,
        positions_market_value=positions_market_value,
        equity=equity,
        total_pnl=total_pnl,
        total_return=total_pnl / run.initial_cash if run.initial_cash > 0 else None,
        valued_at=valued_at.astimezone(UTC),
        positions=tuple(positions),
    )


async def value_paper_run(
    *, session: AsyncSession, run: PaperTradingRunRecord, valued_at: datetime
) -> PaperPortfolioValuation:
    position = PaperPositionRecord
    latest_mark = (
        select(MarketCandleRecord.close_price, MarketCandleRecord.open_time)
        .where(
            MarketCandleRecord.asset == position.asset,
            MarketCandleRecord.provider == position.provider,
            MarketCandleRecord.market_type == position.market_type,
            MarketCandleRecord.interval == position.interval,
            MarketCandleRecord.close_time <= valued_at,
        )
        .order_by(MarketCandleRecord.open_time.desc())
        .limit(1)
        .lateral("latest_mark")
    )
    rows = (
        await session.execute(
            select(
                position,
                latest_mark.c.close_price,
                latest_mark.c.open_time,
            )
            .outerjoin(latest_mark, true())
            .where(
                position.run_id == run.id,
                position.status == PaperPositionStatus.OPEN.value,
            )
            .order_by(position.opened_at, position.id)
        )
    ).all()
    marks: list[PositionMark] = []
    for record, mark_price, mark_open_time in rows:
        if mark_price is None or mark_open_time is None:
            raise PaperPortfolioValuationError(
                "persisted mark candle not found for "
                f"{record.asset} {record.provider} {record.market_type} "
                f"{record.interval}"
            )
        marks.append(
            PositionMark(
                position_id=record.id,
                asset=Asset(record.asset),
                provider=Provider(record.provider),
                market_type=MarketType(record.market_type),
                interval=Interval(record.interval),
                quantity=record.quantity,
                average_entry_price=record.average_entry_price,
                total_entry_notional=record.total_entry_notional,
                accumulated_entry_fees=record.accumulated_entry_fees,
                mark_price=mark_price,
                mark_candle_open_time=mark_open_time,
            )
        )
    return calculate_portfolio_valuation(run=run, marks=marks, valued_at=valued_at)


class PersistedPaperPortfolioStateSource:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_portfolio_state(
        self,
        *,
        asset: Asset,
        provider: Provider,
        market_type: MarketType,
        interval: Interval,
    ) -> PortfolioState:
        run = await get_active_run(self._session, currency=DEFAULT_ACCOUNT_CURRENCY)
        if run is None:
            raise PaperPortfolioUnavailableError("active Paper Trading run not found")
        position = await get_open_position(
            self._session,
            run_id=run.id,
            asset=asset,
            provider=provider,
            market_type=market_type,
            interval=interval,
        )
        return PortfolioState(position_is_open=position is not None)
