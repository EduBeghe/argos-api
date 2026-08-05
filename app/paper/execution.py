from datetime import datetime
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.decisions.models import DecisionSnapshot
from app.decisions.persistence import DecisionSnapshotRecord
from app.market.persistence import MarketCandleRecord
from app.paper.calculation import (
    PaperExecutionError,
    calculate_no_action,
    calculate_open_position,
    calculate_sell_all,
)
from app.paper.config import (
    DEFAULT_ACCOUNT_CURRENCY,
    SimulationConfiguration,
    quantize_financial,
)
from app.paper.decision_evidence import (
    MLDecisionSnapshot,
    MLDecisionSnapshotRecord,
    PaperDecisionIdentityRecord,
    persist_baseline_decision_identity,
)
from app.paper.models import (
    ExecutionCalculation,
    NoTrade,
    PaperPosition,
    PaperPositionStatus,
)
from app.paper.persistence import (
    PaperPositionRecord,
    PaperTradeRecord,
    PaperTradingRunRecord,
    get_active_run,
    get_open_position,
)
from app.strategies.baseline_long_only import StrategyAction


class UnsupportedPaperActionError(PaperExecutionError):
    pass


class PaperValuationError(PaperExecutionError):
    """Raised when persisted evidence cannot value every open position."""


async def execute_decision(
    *,
    session: AsyncSession,
    decision: DecisionSnapshot | MLDecisionSnapshot,
    executed_at: datetime,
    run_id: UUID | None = None,
    decision_identity_id: UUID | None = None,
) -> PaperTradeRecord | NoTrade:
    if decision.action not in {
        StrategyAction.OPEN_POSITION,
        StrategyAction.SELL_ALL,
        StrategyAction.NO_ACTION,
    }:
        raise UnsupportedPaperActionError(
            f"unsupported paper action: {decision.action}"
        )
    run = (
        await session.scalar(
            select(PaperTradingRunRecord)
            .where(PaperTradingRunRecord.id == run_id)
            .with_for_update()
        )
        if run_id is not None
        else await get_active_run(session, currency=DEFAULT_ACCOUNT_CURRENCY, lock=True)
    )
    if run is None:
        raise PaperExecutionError("active Paper Trading run not found")
    if decision.decision_timestamp < run.started_at:
        raise PaperExecutionError(
            "Decision Snapshot predates the active Paper Trading run"
        )
    if executed_at < run.started_at:
        raise PaperExecutionError("execution timestamp predates the active run")
    if executed_at < decision.decision_timestamp:
        raise PaperExecutionError("execution timestamp predates the Decision Snapshot")
    persisted_decision = await _persisted_decision(session, decision)
    if persisted_decision is None:
        raise PaperExecutionError(
            "Decision Snapshot must be persisted before execution"
        )
    if (
        persisted_decision.asset != decision.asset.value
        or persisted_decision.provider != decision.provider.value
        or persisted_decision.market_type != decision.market_type.value
        or persisted_decision.interval != decision.interval.value
        or persisted_decision.action != decision.action.value
        or persisted_decision.source_market_price != decision.source_market_price
    ):
        raise PaperExecutionError("Decision Snapshot does not match persisted intent")
    if decision.action is StrategyAction.NO_ACTION:
        return calculate_no_action()
    if decision_identity_id is None:
        if isinstance(decision, MLDecisionSnapshot):
            raise PaperExecutionError(
                "ML decision identity must be persisted before execution"
            )
        decision_identity_id = (
            await persist_baseline_decision_identity(session, decision.id)
        ).id
    identity = await session.get(PaperDecisionIdentityRecord, decision_identity_id)
    if (
        identity is None
        or (
            isinstance(decision, MLDecisionSnapshot)
            and identity.ml_snapshot_id != decision.id
        )
        or (
            isinstance(decision, DecisionSnapshot)
            and identity.baseline_snapshot_id != decision.id
        )
    ):
        raise PaperExecutionError("decision identity does not match persisted intent")
    existing_trade = await session.scalar(
        select(PaperTradeRecord).where(
            PaperTradeRecord.decision_identity_id == decision_identity_id
        )
    )
    if existing_trade is not None:
        return existing_trade
    config = SimulationConfiguration(
        run.configuration_version, run.account_currency, run.fee_rate, run.slippage_rate
    )
    position = await get_open_position(
        session,
        run_id=run.id,
        asset=decision.asset,
        provider=decision.provider,
        market_type=decision.market_type,
        interval=decision.interval,
        lock=True,
    )
    if decision.action is StrategyAction.OPEN_POSITION:
        if position is not None:
            raise PaperExecutionError("position is already open")
        total_equity = await _total_equity(
            session=session,
            run_id=run.id,
            cash=run.current_cash,
            decision=decision,
        )
        calculation = calculate_open_position(
            cash=run.current_cash,
            total_equity=total_equity,
            observed_price=decision.source_market_price,
            target_percentage=decision.requested_target_portfolio_percentage
            or Decimal("0"),
            configuration=config,
        )
        position = PaperPositionRecord(
            id=uuid4(),
            run_id=run.id,
            asset=decision.asset.value,
            provider=decision.provider.value,
            market_type=decision.market_type.value,
            interval=decision.interval.value,
            status=PaperPositionStatus.OPEN.value,
            quantity=calculation.quantity,
            average_entry_price=calculation.execution_price,
            total_entry_notional=calculation.executed_notional,
            accumulated_entry_fees=calculation.fee_amount,
            opened_at=executed_at,
            closed_at=None,
            exit_price=None,
            gross_realized_pnl=None,
            net_realized_pnl=None,
        )
        session.add(position)
    else:
        if position is None:
            raise PaperExecutionError("SELL_ALL requires an open position")
        calculation = calculate_sell_all(
            cash=run.current_cash,
            position=_domain(position),
            observed_price=decision.source_market_price,
            configuration=config,
        )
        position.status = PaperPositionStatus.CLOSED.value
        position.quantity = Decimal("0")
        position.closed_at = executed_at
        position.exit_price = calculation.execution_price
        position.gross_realized_pnl = calculation.gross_realized_pnl
        position.net_realized_pnl = calculation.net_realized_pnl
    run.current_cash = calculation.resulting_cash
    trade = _trade(
        run.id,
        decision,
        position.id,
        calculation,
        executed_at,
        config,
        decision_identity_id=decision_identity_id,
        equity_before=(
            total_equity if decision.action is StrategyAction.OPEN_POSITION else None
        ),
    )
    session.add(trade)
    await session.flush()
    return trade


async def _total_equity(
    *,
    session: AsyncSession,
    run_id: UUID,
    cash: Decimal,
    decision: DecisionSnapshot | MLDecisionSnapshot,
) -> Decimal:
    positions = (
        await session.scalars(
            select(PaperPositionRecord).where(
                PaperPositionRecord.run_id == run_id,
                PaperPositionRecord.status == PaperPositionStatus.OPEN.value,
            )
        )
    ).all()
    value = cash
    for position in positions:
        price = await session.scalar(
            select(MarketCandleRecord.close_price)
            .where(
                MarketCandleRecord.asset == position.asset,
                MarketCandleRecord.provider == position.provider,
                MarketCandleRecord.market_type == position.market_type,
                MarketCandleRecord.interval == position.interval,
                MarketCandleRecord.close_time <= decision.decision_timestamp,
            )
            .order_by(MarketCandleRecord.close_time.desc())
            .limit(1)
        )
        if price is None:
            raise PaperValuationError(
                f"persisted valuation price not found for {position.asset}"
            )
        value += position.quantity * price
    return quantize_financial(value)


def _domain(record: PaperPositionRecord) -> PaperPosition:
    from app.paper.persistence import position_domain

    return position_domain(record)


def _trade(
    run_id: UUID,
    decision: DecisionSnapshot | MLDecisionSnapshot,
    position_id: UUID,
    result: ExecutionCalculation,
    executed_at: datetime,
    config: SimulationConfiguration,
    decision_identity_id: UUID,
    equity_before: Decimal | None,
) -> PaperTradeRecord:
    return PaperTradeRecord(
        id=uuid4(),
        run_id=run_id,
        decision_snapshot_id=(
            decision.id if isinstance(decision, DecisionSnapshot) else None
        ),
        decision_identity_id=decision_identity_id,
        position_id=position_id,
        side=result.side.value,
        asset=decision.asset.value,
        provider=decision.provider.value,
        market_type=decision.market_type.value,
        interval=decision.interval.value,
        observed_market_price=decision.source_market_price,
        slippage_rate=config.slippage_rate,
        execution_price=result.execution_price,
        quantity=result.quantity,
        executed_notional=result.executed_notional,
        fee_rate=config.fee_rate,
        fee_amount=result.fee_amount,
        cash_impact=result.cash_impact,
        executed_at=executed_at,
        configuration_version=config.version,
        equity_before=equity_before,
    )


async def _persisted_decision(
    session: AsyncSession, decision: DecisionSnapshot | MLDecisionSnapshot
) -> DecisionSnapshotRecord | MLDecisionSnapshotRecord | None:
    record_type = (
        MLDecisionSnapshotRecord
        if isinstance(decision, MLDecisionSnapshot)
        else DecisionSnapshotRecord
    )
    return cast(
        DecisionSnapshotRecord | MLDecisionSnapshotRecord | None,
        await session.scalar(select(record_type).where(record_type.id == decision.id)),
    )
