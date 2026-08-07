from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from statistics import median
from uuid import UUID

from app.market.catalog import Instrument
from app.paper.calculation import calculate_open_position, calculate_sell_all
from app.paper.config import SimulationConfiguration, quantize_financial
from app.paper.models import PaperPosition, PaperPositionStatus

from .contracts import INITIAL_CASH, PAPER_CASH_BUFFER_RATE


@dataclass(frozen=True, slots=True)
class ProtectedPrediction:
    boundary: datetime
    signed_score: Decimal
    observed_price: Decimal


@dataclass(frozen=True, slots=True)
class ProtectedTrade:
    entry_at: datetime
    exit_at: datetime
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    fees: Decimal
    slippage: Decimal
    net_return: Decimal
    exit_reason: str


@dataclass(frozen=True, slots=True)
class ProtectedSimulationResult:
    initial_equity: Decimal
    final_equity: Decimal
    net_pnl: Decimal
    total_return: Decimal
    fees: Decimal
    slippage: Decimal
    round_trips: int
    wins: int
    losses: int
    median_trade_return: Decimal
    worst_trade_return: Decimal
    exposure: Decimal
    maximum_drawdown: Decimal
    trades: tuple[ProtectedTrade, ...]
    equity_curve: tuple[tuple[datetime, Decimal], ...]


@dataclass(frozen=True, slots=True)
class ProtectedBenchmarkResult:
    final_equity: Decimal
    net_pnl: Decimal
    total_return: Decimal
    fees: Decimal
    slippage: Decimal


def simulate_fixed_horizon_long_only(
    *,
    predictions: tuple[ProtectedPrediction, ...],
    instrument: Instrument,
    threshold: Decimal,
    hold: timedelta,
    configuration: SimulationConfiguration,
) -> ProtectedSimulationResult:
    if not predictions:
        raise ValueError("protected simulation requires eligible predictions")
    cash = quantize_financial(INITIAL_CASH)
    position: PaperPosition | None = None
    entry_observed = Decimal(0)
    scheduled_exit: datetime | None = None
    trades: list[ProtectedTrade] = []
    curve: list[tuple[datetime, Decimal]] = []
    peak = cash
    maximum_drawdown = Decimal(0)
    exposed = 0
    for index, point in enumerate(predictions):
        if (
            position is None
            and index < len(predictions) - 1
            and point.signed_score >= threshold
        ):
            opened = calculate_open_position(
                cash=cash,
                total_equity=cash,
                observed_price=point.observed_price,
                target_percentage=Decimal("100")
                * (Decimal(1) - PAPER_CASH_BUFFER_RATE)
                / (Decimal(1) + configuration.fee_rate),
                configuration=configuration,
            )
            cash = opened.resulting_cash
            position = PaperPosition(
                UUID(int=1),
                UUID(int=0),
                instrument.asset,
                instrument.provider,
                instrument.market_type,
                instrument.interval,
                PaperPositionStatus.OPEN,
                opened.quantity,
                opened.execution_price,
                opened.executed_notional,
                opened.fee_amount,
                point.boundary,
                None,
                None,
                None,
                None,
            )
            entry_observed = point.observed_price
            scheduled_exit = point.boundary + hold
        elif position is not None:
            exposed += 1
            forced = index == len(predictions) - 1
            assert scheduled_exit is not None
            if point.boundary >= scheduled_exit or forced:
                sold = calculate_sell_all(
                    cash=cash,
                    position=position,
                    observed_price=point.observed_price,
                    configuration=configuration,
                )
                cash = sold.resulting_cash
                entry_total = (
                    position.total_entry_notional + position.accumulated_entry_fees
                )
                net_return = quantize_financial(
                    (sold.cash_impact - entry_total) / entry_total
                )
                slippage = quantize_financial(
                    (position.average_entry_price - entry_observed) * position.quantity
                    + (point.observed_price - sold.execution_price) * position.quantity
                )
                trades.append(
                    ProtectedTrade(
                        position.opened_at,
                        point.boundary,
                        position.average_entry_price,
                        sold.execution_price,
                        position.quantity,
                        position.accumulated_entry_fees + sold.fee_amount,
                        slippage,
                        net_return,
                        "end_of_test_liquidation"
                        if forced and point.boundary < scheduled_exit
                        else "scheduled_horizon",
                    )
                )
                position = None
                scheduled_exit = None
        equity = quantize_financial(
            cash
            + (position.quantity * point.observed_price if position else Decimal(0))
        )
        peak = max(peak, equity)
        if peak > 0:
            maximum_drawdown = max(
                maximum_drawdown, quantize_financial((peak - equity) / peak)
            )
        curve.append((point.boundary, equity))
    if position is not None:
        raise ValueError("protected simulation could not liquidate its final position")
    returns = tuple(item.net_return for item in trades)
    fees = sum((item.fees for item in trades), Decimal(0))
    slippage = sum((item.slippage for item in trades), Decimal(0))
    return ProtectedSimulationResult(
        INITIAL_CASH,
        cash,
        quantize_financial(cash - INITIAL_CASH),
        quantize_financial((cash - INITIAL_CASH) / INITIAL_CASH),
        fees,
        slippage,
        len(trades),
        sum(item > 0 for item in returns),
        sum(item <= 0 for item in returns),
        quantize_financial(median(returns)) if returns else Decimal(0),
        min(returns, default=Decimal(0)),
        quantize_financial(Decimal(exposed) / Decimal(len(predictions))),
        maximum_drawdown,
        tuple(trades),
        tuple(curve),
    )


def simulate_buy_and_hold(
    *,
    predictions: tuple[ProtectedPrediction, ...],
    instrument: Instrument,
    configuration: SimulationConfiguration,
) -> ProtectedBenchmarkResult:
    """Buy at the first eligible mark and liquidate at the last with identical costs."""

    if len(predictions) < 2:
        raise ValueError("buy-and-hold requires at least two eligible predictions")
    opened = calculate_open_position(
        cash=INITIAL_CASH,
        total_equity=INITIAL_CASH,
        observed_price=predictions[0].observed_price,
        target_percentage=Decimal("100")
        * (Decimal(1) - PAPER_CASH_BUFFER_RATE)
        / (Decimal(1) + configuration.fee_rate),
        configuration=configuration,
    )
    if opened.resulting_cash < 0:
        raise ValueError("buy-and-hold technical cash buffer was insufficient")
    position = PaperPosition(
        UUID(int=2),
        UUID(int=0),
        instrument.asset,
        instrument.provider,
        instrument.market_type,
        instrument.interval,
        PaperPositionStatus.OPEN,
        opened.quantity,
        opened.execution_price,
        opened.executed_notional,
        opened.fee_amount,
        predictions[0].boundary,
        None,
        None,
        None,
        None,
    )
    sold = calculate_sell_all(
        cash=opened.resulting_cash,
        position=position,
        observed_price=predictions[-1].observed_price,
        configuration=configuration,
    )
    slippage = quantize_financial(
        (opened.execution_price - predictions[0].observed_price) * opened.quantity
        + (predictions[-1].observed_price - sold.execution_price) * opened.quantity
    )
    return ProtectedBenchmarkResult(
        sold.resulting_cash,
        quantize_financial(sold.resulting_cash - INITIAL_CASH),
        quantize_financial((sold.resulting_cash - INITIAL_CASH) / INITIAL_CASH),
        quantize_financial(opened.fee_amount + sold.fee_amount),
        slippage,
    )
