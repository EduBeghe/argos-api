from decimal import Decimal

from app.paper.config import SimulationConfiguration, quantize_financial
from app.paper.models import (
    ExecutionCalculation,
    NoTrade,
    PaperPosition,
    PaperTradeSide,
)


class PaperExecutionError(ValueError):
    """Raised when a decision cannot be simulated safely."""


def calculate_open_position(
    *,
    cash: Decimal,
    total_equity: Decimal,
    observed_price: Decimal,
    target_percentage: Decimal,
    configuration: SimulationConfiguration,
) -> ExecutionCalculation:
    execution_price = quantize_financial(
        observed_price * (Decimal("1") + configuration.slippage_rate)
    )
    target_notional = quantize_financial(
        total_equity * target_percentage / Decimal("100")
    )
    quantity = quantize_financial(target_notional / execution_price)
    notional = quantize_financial(quantity * execution_price)
    fee = quantize_financial(notional * configuration.fee_rate)
    cash_impact = quantize_financial(-(notional + fee))
    resulting_cash = quantize_financial(cash + cash_impact)
    if resulting_cash < 0 or quantity <= 0:
        raise PaperExecutionError("insufficient paper cash")
    return ExecutionCalculation(
        PaperTradeSide.BUY,
        execution_price,
        quantity,
        notional,
        fee,
        cash_impact,
        resulting_cash,
    )


def calculate_sell_all(
    *,
    cash: Decimal,
    position: PaperPosition,
    observed_price: Decimal,
    configuration: SimulationConfiguration,
) -> ExecutionCalculation:
    if position.quantity <= 0:
        raise PaperExecutionError("open position has no available quantity")
    execution_price = quantize_financial(
        observed_price * (Decimal("1") - configuration.slippage_rate)
    )
    notional = quantize_financial(position.quantity * execution_price)
    fee = quantize_financial(notional * configuration.fee_rate)
    cash_impact = quantize_financial(notional - fee)
    gross = quantize_financial(notional - position.total_entry_notional)
    net = quantize_financial(gross - position.accumulated_entry_fees - fee)
    return ExecutionCalculation(
        PaperTradeSide.SELL,
        execution_price,
        position.quantity,
        notional,
        fee,
        cash_impact,
        quantize_financial(cash + cash_impact),
        gross,
        net,
    )


def calculate_no_action() -> NoTrade:
    return NoTrade()
