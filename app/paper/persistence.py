from datetime import datetime
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.base import Base
from app.decisions.persistence import DecisionSnapshotRecord
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.history import HistoryOrder
from app.paper.config import SimulationConfiguration
from app.paper.models import (
    PaperPosition,
    PaperPositionStatus,
    PaperRunStatus,
    PaperTradingRun,
)


class PaperTradingRunRecord(Base):
    __tablename__ = "paper_trading_runs"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'stopped')", name="ck_paper_runs_status"),
        CheckConstraint(
            "length(btrim(account_currency)) > 0", name="ck_paper_runs_currency"
        ),
        CheckConstraint("initial_cash > 0", name="ck_paper_runs_initial_cash"),
        CheckConstraint("current_cash >= 0", name="ck_paper_runs_current_cash"),
        CheckConstraint(
            "fee_rate >= 0 AND fee_rate < 1", name="ck_paper_runs_fee_rate"
        ),
        CheckConstraint(
            "slippage_rate >= 0 AND slippage_rate < 1",
            name="ck_paper_runs_slippage_rate",
        ),
        CheckConstraint(
            "(status = 'active' AND stopped_at IS NULL) OR "
            "(status = 'stopped' AND stopped_at IS NOT NULL)",
            name="ck_paper_runs_lifecycle",
        ),
        CheckConstraint("configuration_version > 0", name="ck_paper_runs_version"),
        CheckConstraint(
            "predecessor_run_id IS NULL OR predecessor_run_id <> id",
            name="ck_paper_runs_predecessor",
        ),
        CheckConstraint(
            "stopped_at IS NULL OR stopped_at >= started_at",
            name="ck_paper_runs_timestamps",
        ),
        Index(
            "uq_paper_runs_active_currency",
            "account_currency",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    account_currency: Mapped[str] = mapped_column(String(16), nullable=False)
    initial_cash: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    current_cash: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    configuration_version: Mapped[int] = mapped_column(Integer, nullable=False)
    fee_rate: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    slippage_rate: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stop_reason: Mapped[str | None] = mapped_column(String(512))
    predecessor_run_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("paper_trading_runs.id")
    )


class PaperPositionRecord(Base):
    __tablename__ = "paper_positions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('open', 'closed')", name="ck_paper_positions_status"
        ),
        CheckConstraint(
            "length(btrim(asset)) > 0 AND length(btrim(provider)) > 0 AND "
            "length(btrim(market_type)) > 0 AND length(btrim(interval)) > 0",
            name="ck_paper_positions_identity",
        ),
        CheckConstraint(
            "quantity >= 0 AND average_entry_price > 0 AND "
            "total_entry_notional > 0 AND accumulated_entry_fees >= 0",
            name="ck_paper_positions_amounts",
        ),
        CheckConstraint(
            "(status = 'open' AND quantity > 0 AND closed_at IS NULL AND "
            "exit_price IS NULL AND gross_realized_pnl IS NULL AND "
            "net_realized_pnl IS NULL) OR (status = 'closed' AND quantity = 0 AND "
            "closed_at IS NOT NULL AND exit_price > 0 AND "
            "gross_realized_pnl IS NOT NULL AND net_realized_pnl IS NOT NULL)",
            name="ck_paper_positions_lifecycle",
        ),
        CheckConstraint(
            "closed_at IS NULL OR closed_at >= opened_at",
            name="ck_paper_positions_timestamps",
        ),
        Index(
            "uq_paper_positions_open_identity",
            "run_id",
            "asset",
            "provider",
            "market_type",
            "interval",
            unique=True,
            postgresql_where=text("status = 'open'"),
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("paper_trading_runs.id"),
        nullable=False,
    )
    asset: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    average_entry_price: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    total_entry_notional: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    accumulated_entry_fees: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    gross_realized_pnl: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    net_realized_pnl: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))


class PaperTradeRecord(Base):
    __tablename__ = "paper_trades"
    __table_args__ = (
        CheckConstraint("side IN ('buy', 'sell')", name="ck_paper_trades_side"),
        CheckConstraint(
            "length(btrim(asset)) > 0 AND length(btrim(provider)) > 0 AND "
            "length(btrim(market_type)) > 0 AND length(btrim(interval)) > 0",
            name="ck_paper_trades_identity",
        ),
        CheckConstraint(
            "observed_market_price > 0 AND execution_price > 0 AND quantity > 0 AND "
            "executed_notional > 0 AND fee_amount >= 0",
            name="ck_paper_trades_amounts",
        ),
        CheckConstraint(
            "fee_rate >= 0 AND fee_rate < 1 AND slippage_rate >= 0 AND "
            "slippage_rate < 1",
            name="ck_paper_trades_rates",
        ),
        CheckConstraint("configuration_version > 0", name="ck_paper_trades_version"),
        CheckConstraint(
            "(side = 'buy' AND cash_impact < 0) OR (side = 'sell' AND cash_impact > 0)",
            name="ck_paper_trades_cash_direction",
        ),
        Index("uq_paper_trades_decision", "decision_snapshot_id", unique=True),
        Index("ix_paper_trades_run_executed", "run_id", "executed_at", "id"),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("paper_trading_runs.id"),
        nullable=False,
    )
    decision_snapshot_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(DecisionSnapshotRecord.id),
        nullable=False,
    )
    position_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("paper_positions.id"), nullable=False
    )
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    asset: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(16), nullable=False)
    observed_market_price: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    slippage_rate: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    execution_price: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    executed_notional: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    fee_rate: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    fee_amount: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    cash_impact: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    configuration_version: Mapped[int] = mapped_column(Integer, nullable=False)
    equity_before: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))


def _run(record: PaperTradingRunRecord) -> PaperTradingRun:
    return PaperTradingRun(
        record.id,
        PaperRunStatus(record.status),
        record.account_currency,
        record.initial_cash,
        record.current_cash,
        record.configuration_version,
        record.fee_rate,
        record.slippage_rate,
        record.started_at,
        record.stopped_at,
        record.stop_reason,
        record.predecessor_run_id,
    )


def _position(record: PaperPositionRecord) -> PaperPosition:
    return PaperPosition(
        record.id,
        record.run_id,
        Asset(record.asset),
        Provider(record.provider),
        MarketType(record.market_type),
        Interval(record.interval),
        PaperPositionStatus(record.status),
        record.quantity,
        record.average_entry_price,
        record.total_entry_notional,
        record.accumulated_entry_fees,
        record.opened_at,
        record.closed_at,
        record.exit_price,
        record.gross_realized_pnl,
        record.net_realized_pnl,
    )


async def get_active_run(
    session: AsyncSession, *, currency: str, lock: bool = False
) -> PaperTradingRunRecord | None:
    statement = select(PaperTradingRunRecord).where(
        PaperTradingRunRecord.account_currency == currency,
        PaperTradingRunRecord.status == PaperRunStatus.ACTIVE.value,
    )
    if lock:
        statement = statement.with_for_update()
    return cast(PaperTradingRunRecord | None, await session.scalar(statement))


async def get_run(
    session: AsyncSession, *, run_id: UUID
) -> PaperTradingRunRecord | None:
    return cast(
        PaperTradingRunRecord | None,
        await session.scalar(
            select(PaperTradingRunRecord).where(PaperTradingRunRecord.id == run_id)
        ),
    )


async def list_runs(
    session: AsyncSession, *, limit: int, offset: int
) -> list[PaperTradingRun]:
    records = (
        await session.scalars(
            select(PaperTradingRunRecord)
            .order_by(
                PaperTradingRunRecord.started_at.desc(),
                PaperTradingRunRecord.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return [_run(record) for record in records]


async def create_run(
    session: AsyncSession,
    *,
    initial_cash: Decimal,
    configuration: SimulationConfiguration,
    started_at: datetime,
    predecessor_run_id: UUID | None = None,
) -> PaperTradingRunRecord:
    record = PaperTradingRunRecord(
        id=uuid4(),
        status=PaperRunStatus.ACTIVE.value,
        account_currency=configuration.account_currency,
        initial_cash=initial_cash,
        current_cash=initial_cash,
        configuration_version=configuration.version,
        fee_rate=configuration.fee_rate,
        slippage_rate=configuration.slippage_rate,
        started_at=started_at,
        stopped_at=None,
        stop_reason=None,
        predecessor_run_id=predecessor_run_id,
    )
    session.add(record)
    await session.flush()
    return record


async def get_open_position(
    session: AsyncSession,
    *,
    run_id: UUID,
    asset: Asset,
    provider: Provider,
    market_type: MarketType,
    interval: Interval,
    lock: bool = False,
) -> PaperPositionRecord | None:
    statement = select(PaperPositionRecord).where(
        PaperPositionRecord.run_id == run_id,
        PaperPositionRecord.asset == asset.value,
        PaperPositionRecord.provider == provider.value,
        PaperPositionRecord.market_type == market_type.value,
        PaperPositionRecord.interval == interval.value,
        PaperPositionRecord.status == PaperPositionStatus.OPEN.value,
    )
    if lock:
        statement = statement.with_for_update()
    return cast(PaperPositionRecord | None, await session.scalar(statement))


async def list_positions(
    session: AsyncSession,
    *,
    run_id: UUID,
    status: PaperPositionStatus | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[PaperPosition]:
    statement = (
        select(PaperPositionRecord)
        .where(PaperPositionRecord.run_id == run_id)
        .order_by(PaperPositionRecord.opened_at.desc(), PaperPositionRecord.id.desc())
        .limit(limit)
        .offset(offset)
    )
    if status is not None:
        statement = statement.where(PaperPositionRecord.status == status.value)
    return [_position(row) for row in (await session.scalars(statement)).all()]


async def list_trades(
    session: AsyncSession,
    *,
    run_id: UUID,
    asset: Asset | None = None,
    provider: Provider | None = None,
    market_type: MarketType | None = None,
    interval: Interval | None = None,
    from_time: datetime | None = None,
    to_time: datetime | None = None,
    order: HistoryOrder = HistoryOrder.DESC,
    limit: int,
    offset: int,
) -> list[PaperTradeRecord]:
    statement = select(PaperTradeRecord).where(PaperTradeRecord.run_id == run_id)
    if asset is not None:
        statement = statement.where(PaperTradeRecord.asset == asset.value)
    if provider is not None:
        statement = statement.where(PaperTradeRecord.provider == provider.value)
    if market_type is not None:
        statement = statement.where(PaperTradeRecord.market_type == market_type.value)
    if interval is not None:
        statement = statement.where(PaperTradeRecord.interval == interval.value)
    if from_time is not None:
        statement = statement.where(PaperTradeRecord.executed_at >= from_time)
    if to_time is not None:
        statement = statement.where(PaperTradeRecord.executed_at <= to_time)
    ordering = (
        (PaperTradeRecord.executed_at.asc(), PaperTradeRecord.id.asc())
        if order is HistoryOrder.ASC
        else (PaperTradeRecord.executed_at.desc(), PaperTradeRecord.id.desc())
    )
    return list(
        (
            await session.scalars(
                statement.order_by(*ordering).limit(limit).offset(offset)
            )
        ).all()
    )


def run_domain(record: PaperTradingRunRecord) -> PaperTradingRun:
    return _run(record)


def position_domain(record: PaperPositionRecord) -> PaperPosition:
    return _position(record)
