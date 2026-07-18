from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal, cast
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    asc,
    desc,
    select,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column, relationship, selectinload

from app.core.database.base import Base
from app.market.persistence import DECIMAL_PRECISION, DECIMAL_SCALE
from app.paper.config import quantize_financial
from app.paper.persistence import PaperTradingRunRecord
from app.paper.portfolio import PaperPortfolioValuation, value_paper_run


class PaperPortfolioSnapshotRecord(Base):
    __tablename__ = "paper_portfolio_snapshots"
    __table_args__ = (
        CheckConstraint(
            "length(btrim(account_currency)) > 0",
            name="ck_portfolio_snapshots_currency",
        ),
        CheckConstraint(
            "configuration_version > 0", name="ck_portfolio_snapshots_version"
        ),
        CheckConstraint(
            "recorded_at >= scheduled_boundary",
            name="ck_portfolio_snapshots_timestamps",
        ),
        UniqueConstraint(
            "run_id", "scheduled_boundary", name="uq_portfolio_snapshots_run_boundary"
        ),
        Index(
            "ix_portfolio_snapshots_run_boundary", "run_id", "scheduled_boundary", "id"
        ),
    )

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(PaperTradingRunRecord.id),
        nullable=False,
    )
    scheduled_boundary: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    account_currency: Mapped[str] = mapped_column(String(16), nullable=False)
    initial_cash: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    current_cash: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    positions_market_value: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    equity: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    total_pnl: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    total_return: Mapped[Decimal | None] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE)
    )
    configuration_version: Mapped[int] = mapped_column(Integer, nullable=False)
    positions: Mapped[list["PaperPositionSnapshotRecord"]] = relationship(
        back_populates="portfolio_snapshot",
        lazy="raise",
        order_by="PaperPositionSnapshotRecord.id",
    )


class PaperPositionSnapshotRecord(Base):
    __tablename__ = "paper_position_snapshots"
    __table_args__ = (
        CheckConstraint(
            "length(btrim(asset)) > 0 AND length(btrim(provider)) > 0 AND "
            "length(btrim(market_type)) > 0 AND length(btrim(interval)) > 0",
            name="ck_position_snapshots_identity",
        ),
        CheckConstraint(
            "quantity > 0 AND average_entry_price > 0 AND cost_basis > 0 AND "
            "mark_price > 0 AND market_value > 0",
            name="ck_position_snapshots_amounts",
        ),
        UniqueConstraint(
            "portfolio_snapshot_id",
            "position_id",
            name="uq_position_snapshots_parent_position",
        ),
        Index("ix_position_snapshots_parent", "portfolio_snapshot_id", "id"),
    )

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    portfolio_snapshot_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(PaperPortfolioSnapshotRecord.id, ondelete="CASCADE"),
        nullable=False,
    )
    position_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False
    )
    asset: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(16), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    average_entry_price: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    cost_basis: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    mark_price: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    market_value: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    unrealized_pnl: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    unrealized_return: Mapped[Decimal | None] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE)
    )
    mark_candle_open_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    portfolio_snapshot: Mapped[PaperPortfolioSnapshotRecord] = relationship(
        back_populates="positions"
    )


@dataclass(frozen=True, slots=True)
class SnapshotWriteResult:
    snapshot: PaperPortfolioSnapshotRecord
    created: bool


@dataclass(frozen=True, slots=True)
class PreparedPositionSnapshot:
    position_id: UUID
    asset: str
    provider: str
    market_type: str
    interval: str
    quantity: Decimal
    average_entry_price: Decimal
    cost_basis: Decimal
    mark_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_return: Decimal | None
    mark_candle_open_time: datetime


@dataclass(frozen=True, slots=True)
class PreparedPortfolioSnapshot:
    initial_cash: Decimal
    current_cash: Decimal
    positions_market_value: Decimal
    equity: Decimal
    total_pnl: Decimal
    total_return: Decimal | None
    positions: tuple[PreparedPositionSnapshot, ...]


async def persist_equity_snapshot(
    *,
    session: AsyncSession,
    run: PaperTradingRunRecord,
    scheduled_boundary: datetime,
    recorded_at: datetime,
) -> SnapshotWriteResult:
    existing = await get_equity_snapshot_by_identity(
        session=session, run_id=run.id, scheduled_boundary=scheduled_boundary
    )
    if existing is not None:
        return SnapshotWriteResult(existing, False)
    valuation = await value_paper_run(
        session=session, run=run, valued_at=scheduled_boundary
    )
    prepared = prepare_snapshot_values(valuation)
    snapshot_id = uuid4()
    inserted_id = await session.scalar(
        insert(PaperPortfolioSnapshotRecord)
        .values(
            _parent_values(
                snapshot_id, run, valuation, prepared, scheduled_boundary, recorded_at
            )
        )
        .on_conflict_do_nothing(constraint="uq_portfolio_snapshots_run_boundary")
        .returning(PaperPortfolioSnapshotRecord.id)
    )
    if inserted_id is None:
        concurrent = await get_equity_snapshot_by_identity(
            session=session, run_id=run.id, scheduled_boundary=scheduled_boundary
        )
        if concurrent is None:
            raise RuntimeError("concurrent equity snapshot could not be loaded")
        return SnapshotWriteResult(concurrent, False)
    session.add_all(
        [
            PaperPositionSnapshotRecord(
                id=uuid4(),
                portfolio_snapshot_id=snapshot_id,
                position_id=position.position_id,
                asset=position.asset,
                provider=position.provider,
                market_type=position.market_type,
                interval=position.interval,
                quantity=position.quantity,
                average_entry_price=position.average_entry_price,
                cost_basis=position.cost_basis,
                mark_price=position.mark_price,
                market_value=position.market_value,
                unrealized_pnl=position.unrealized_pnl,
                unrealized_return=position.unrealized_return,
                mark_candle_open_time=position.mark_candle_open_time,
            )
            for position in prepared.positions
        ]
    )
    await session.flush()
    created = await get_equity_snapshot(session, snapshot_id)
    if created is None:
        raise RuntimeError("created equity snapshot could not be loaded")
    return SnapshotWriteResult(created, True)


def _parent_values(
    snapshot_id: UUID,
    run: PaperTradingRunRecord,
    valuation: PaperPortfolioValuation,
    prepared: PreparedPortfolioSnapshot,
    scheduled_boundary: datetime,
    recorded_at: datetime,
) -> dict[str, object]:
    return {
        "id": snapshot_id,
        "run_id": run.id,
        "scheduled_boundary": scheduled_boundary,
        "recorded_at": recorded_at,
        "account_currency": valuation.account_currency,
        "initial_cash": prepared.initial_cash,
        "current_cash": prepared.current_cash,
        "positions_market_value": prepared.positions_market_value,
        "equity": prepared.equity,
        "total_pnl": prepared.total_pnl,
        "total_return": prepared.total_return,
        "configuration_version": run.configuration_version,
    }


def prepare_snapshot_values(
    valuation: PaperPortfolioValuation,
) -> PreparedPortfolioSnapshot:
    positions: list[PreparedPositionSnapshot] = []
    for position in valuation.positions:
        quantity = quantize_financial(position.quantity)
        average_entry_price = quantize_financial(position.average_entry_price)
        cost_basis = quantize_financial(position.cost_basis)
        mark_price = quantize_financial(position.mark_price)
        market_value = quantize_financial(quantity * mark_price)
        unrealized_pnl = quantize_financial(market_value - cost_basis)
        unrealized_return = (
            quantize_financial(unrealized_pnl / cost_basis) if cost_basis > 0 else None
        )
        positions.append(
            PreparedPositionSnapshot(
                position_id=position.position_id,
                asset=position.asset.value,
                provider=position.provider.value,
                market_type=position.market_type.value,
                interval=position.interval.value,
                quantity=quantity,
                average_entry_price=average_entry_price,
                cost_basis=cost_basis,
                mark_price=mark_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_return=unrealized_return,
                mark_candle_open_time=position.mark_candle_open_time,
            )
        )
    initial_cash = quantize_financial(valuation.initial_cash)
    current_cash = quantize_financial(valuation.current_cash)
    positions_market_value = sum(
        (position.market_value for position in positions), start=Decimal(0)
    )
    equity = quantize_financial(current_cash + positions_market_value)
    total_pnl = quantize_financial(equity - initial_cash)
    total_return = (
        quantize_financial(total_pnl / initial_cash) if initial_cash > 0 else None
    )
    return PreparedPortfolioSnapshot(
        initial_cash=initial_cash,
        current_cash=current_cash,
        positions_market_value=positions_market_value,
        equity=equity,
        total_pnl=total_pnl,
        total_return=total_return,
        positions=tuple(positions),
    )


async def get_equity_snapshot(
    session: AsyncSession, snapshot_id: UUID
) -> PaperPortfolioSnapshotRecord | None:
    return cast(
        PaperPortfolioSnapshotRecord | None,
        await session.scalar(
            select(PaperPortfolioSnapshotRecord)
            .options(selectinload(PaperPortfolioSnapshotRecord.positions))
            .where(PaperPortfolioSnapshotRecord.id == snapshot_id)
        ),
    )


async def get_equity_snapshot_by_identity(
    *, session: AsyncSession, run_id: UUID, scheduled_boundary: datetime
) -> PaperPortfolioSnapshotRecord | None:
    return cast(
        PaperPortfolioSnapshotRecord | None,
        await session.scalar(
            select(PaperPortfolioSnapshotRecord)
            .options(selectinload(PaperPortfolioSnapshotRecord.positions))
            .where(
                PaperPortfolioSnapshotRecord.run_id == run_id,
                PaperPortfolioSnapshotRecord.scheduled_boundary == scheduled_boundary,
            )
        ),
    )


async def list_equity_snapshots(
    *,
    session: AsyncSession,
    run_id: UUID,
    limit: int,
    offset: int,
    order: Literal["asc", "desc"],
) -> list[PaperPortfolioSnapshotRecord]:
    direction = asc if order == "asc" else desc
    return list(
        (
            await session.scalars(
                select(PaperPortfolioSnapshotRecord)
                .options(selectinload(PaperPortfolioSnapshotRecord.positions))
                .where(PaperPortfolioSnapshotRecord.run_id == run_id)
                .order_by(
                    direction(PaperPortfolioSnapshotRecord.scheduled_boundary),
                    direction(PaperPortfolioSnapshotRecord.id),
                )
                .limit(limit)
                .offset(offset)
            )
        ).all()
    )
