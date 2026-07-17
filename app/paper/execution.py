from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.decisions.models import DecisionSnapshot
from app.paper.calculation import (
    PaperExecutionError,
    calculate_no_action,
    calculate_open_position,
    calculate_sell_all,
)
from app.paper.config import DEFAULT_ACCOUNT_CURRENCY, SimulationConfiguration
from app.paper.models import (
    ExecutionCalculation,
    NoTrade,
    PaperPosition,
    PaperPositionStatus,
)
from app.paper.persistence import (
    PaperPositionRecord,
    PaperTradeRecord,
    get_active_run,
    get_open_position,
)
from app.strategies.baseline_long_only import StrategyAction


class UnsupportedPaperActionError(PaperExecutionError):
    pass


async def execute_decision(
    *, session: AsyncSession, decision: DecisionSnapshot, executed_at: datetime
) -> PaperTradeRecord | NoTrade:
    if decision.action is StrategyAction.NO_ACTION:
        return calculate_no_action()
    if decision.action not in {StrategyAction.OPEN_POSITION, StrategyAction.SELL_ALL}:
        raise UnsupportedPaperActionError(
            f"unsupported paper action: {decision.action}"
        )
    run = await get_active_run(session, currency=DEFAULT_ACCOUNT_CURRENCY, lock=True)
    if run is None:
        raise PaperExecutionError("active Paper Trading run not found")
    existing_trade = await session.scalar(
        select(PaperTradeRecord).where(
            PaperTradeRecord.decision_snapshot_id == decision.id
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
        calculation = calculate_open_position(
            cash=run.current_cash,
            total_equity=run.current_cash,
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
    trade = _trade(run.id, decision, position.id, calculation, executed_at, config)
    session.add(trade)
    await session.flush()
    return trade


def _domain(record: PaperPositionRecord) -> PaperPosition:
    from app.paper.persistence import position_domain

    return position_domain(record)


def _trade(
    run_id: UUID,
    decision: DecisionSnapshot,
    position_id: UUID,
    result: ExecutionCalculation,
    executed_at: datetime,
    config: SimulationConfiguration,
) -> PaperTradeRecord:
    return PaperTradeRecord(
        id=uuid4(),
        run_id=run_id,
        decision_snapshot_id=decision.id,
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
    )
