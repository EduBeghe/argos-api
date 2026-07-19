from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from enum import StrEnum

from app.paper.config import FINANCIAL_QUANTUM, quantize_financial
from app.paper.performance import PaperRunPerformance


class EvaluationStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class EvaluationUnavailableReason(StrEnum):
    NO_EQUITY_SNAPSHOT = "no_equity_snapshot"
    NO_DECISION_UNIVERSE = "no_decision_universe"
    MISSING_COMPATIBLE_CANDLE = "missing_compatible_candle"
    INCOMPLETE_CANDLE_EVIDENCE = "incomplete_candle_evidence"
    STRATEGY_OR_BENCHMARK_UNAVAILABLE = "strategy_or_benchmark_unavailable"


class BenchmarkMethod(StrEnum):
    EQUAL_WEIGHT_BUY_AND_HOLD_V1 = "equal_weight_buy_and_hold_v1"


@dataclass(frozen=True, order=True, slots=True)
class MarketIdentity:
    asset: str
    provider: str
    market_type: str
    interval: str


@dataclass(frozen=True, slots=True)
class DecisionEvidence:
    identity: MarketIdentity
    action: str


@dataclass(frozen=True, slots=True)
class TradeEvidence:
    identity: MarketIdentity
    side: str
    observed_market_price: Decimal
    execution_price: Decimal
    quantity: Decimal
    fee_amount: Decimal


@dataclass(frozen=True, slots=True)
class ClosedPositionEvidence:
    identity: MarketIdentity
    gross_realized_pnl: Decimal
    net_realized_pnl: Decimal


@dataclass(frozen=True, slots=True)
class OpenPositionEvidence:
    identity: MarketIdentity
    unrealized_pnl: Decimal


@dataclass(frozen=True, slots=True)
class CandleEvidence:
    open_time: datetime
    close_price: Decimal


@dataclass(frozen=True, slots=True)
class BenchmarkCandles:
    entry: CandleEvidence | None
    ending: CandleEvidence | None


def count_values(
    values: tuple[str, ...], *, defaults: tuple[str, ...] = ()
) -> dict[str, int]:
    """Count all persisted values without dropping future vocabulary."""
    counts = Counter({value: 0 for value in defaults})
    counts.update(values)
    return dict(sorted(counts.items()))


def estimated_slippage_cost(trade: TradeEvidence) -> Decimal:
    if trade.side == "buy":
        difference = max(
            trade.execution_price - trade.observed_market_price, Decimal(0)
        )
    elif trade.side == "sell":
        difference = max(
            trade.observed_market_price - trade.execution_price, Decimal(0)
        )
    else:
        difference = Decimal(0)
    return quantize_financial(difference * trade.quantity)


def closed_position_statistics(
    positions: tuple[ClosedPositionEvidence, ...],
) -> dict[str, Decimal | int | None]:
    values = tuple(quantize_financial(item.net_realized_pnl) for item in positions)
    wins = tuple(value for value in values if value > 0)
    losses = tuple(value for value in values if value < 0)
    breakeven = len(values) - len(wins) - len(losses)
    win_total = sum(wins, start=Decimal(0))
    loss_total = sum(losses, start=Decimal(0))
    return {
        "closed_position_count": len(values),
        "winning_position_count": len(wins),
        "losing_position_count": len(losses),
        "breakeven_position_count": breakeven,
        "win_rate": (
            quantize_financial(Decimal(len(wins)) / Decimal(len(values)))
            if values
            else None
        ),
        "average_win": (
            quantize_financial(win_total / Decimal(len(wins))) if wins else None
        ),
        "average_loss": (
            quantize_financial(loss_total / Decimal(len(losses))) if losses else None
        ),
        "profit_factor": (
            quantize_financial(win_total / abs(loss_total)) if losses else None
        ),
    }


def strategy_evaluation(
    *,
    performance: PaperRunPerformance,
    decisions: tuple[DecisionEvidence, ...],
    trades: tuple[TradeEvidence, ...],
    closed_positions: tuple[ClosedPositionEvidence, ...],
    open_positions: tuple[OpenPositionEvidence, ...],
) -> dict[str, object]:
    closed = closed_position_statistics(closed_positions)
    return {
        "status": EvaluationStatus.AVAILABLE,
        "unavailable_reason": None,
        "initial_equity": performance.initial_equity,
        "ending_equity": performance.latest_equity,
        "absolute_pnl": performance.absolute_pnl,
        "total_return": performance.total_return,
        "current_drawdown": performance.current_drawdown,
        "max_drawdown": performance.max_drawdown,
        "realized_gross_pnl": quantize_financial(
            sum(
                (item.gross_realized_pnl for item in closed_positions), start=Decimal(0)
            )
        ),
        "realized_net_pnl": quantize_financial(
            sum((item.net_realized_pnl for item in closed_positions), start=Decimal(0))
        ),
        "unrealized_pnl": quantize_financial(
            sum((item.unrealized_pnl for item in open_positions), start=Decimal(0))
        ),
        "fees_paid": quantize_financial(
            sum((item.fee_amount for item in trades), start=Decimal(0))
        ),
        "estimated_slippage_cost": quantize_financial(
            sum((estimated_slippage_cost(item) for item in trades), start=Decimal(0))
        ),
        "decision_count": len(decisions),
        "decision_counts_by_action": count_values(
            tuple(item.action for item in decisions),
            defaults=("open_position", "sell_all", "no_action"),
        ),
        "trade_count": len(trades),
        "trade_counts_by_side": count_values(
            tuple(item.side for item in trades), defaults=("buy", "sell")
        ),
        "open_position_count": len(open_positions),
        **closed,
    }


def unavailable_strategy(initial_equity: Decimal) -> dict[str, object]:
    return {
        "status": EvaluationStatus.UNAVAILABLE,
        "unavailable_reason": EvaluationUnavailableReason.NO_EQUITY_SNAPSHOT,
        "initial_equity": quantize_financial(initial_equity),
        **{
            name: None
            for name in (
                "ending_equity",
                "absolute_pnl",
                "total_return",
                "current_drawdown",
                "max_drawdown",
                "realized_gross_pnl",
                "realized_net_pnl",
                "unrealized_pnl",
                "fees_paid",
                "estimated_slippage_cost",
                "win_rate",
                "average_win",
                "average_loss",
                "profit_factor",
            )
        },
        "decision_count": 0,
        "decision_counts_by_action": count_values(
            (), defaults=("open_position", "sell_all", "no_action")
        ),
        "trade_count": 0,
        "trade_counts_by_side": count_values((), defaults=("buy", "sell")),
        "open_position_count": 0,
        "closed_position_count": 0,
        "winning_position_count": 0,
        "losing_position_count": 0,
        "breakeven_position_count": 0,
    }


def instrument_evaluations(
    *,
    decisions: tuple[DecisionEvidence, ...],
    trades: tuple[TradeEvidence, ...],
    closed_positions: tuple[ClosedPositionEvidence, ...],
    open_positions: tuple[OpenPositionEvidence, ...],
) -> list[dict[str, object]]:
    identities = sorted(
        {item.identity for item in decisions}
        | {item.identity for item in trades}
        | {item.identity for item in closed_positions}
        | {item.identity for item in open_positions}
    )
    result: list[dict[str, object]] = []
    for identity in identities:
        identity_decisions = tuple(
            item for item in decisions if item.identity == identity
        )
        identity_trades = tuple(item for item in trades if item.identity == identity)
        identity_closed = tuple(
            item for item in closed_positions if item.identity == identity
        )
        identity_open = tuple(
            item for item in open_positions if item.identity == identity
        )
        result.append(
            {
                **identity_fields(identity),
                "decision_count": len(identity_decisions),
                "decision_counts_by_action": count_values(
                    tuple(item.action for item in identity_decisions),
                    defaults=("open_position", "sell_all", "no_action"),
                ),
                "trade_count": len(identity_trades),
                "trade_counts_by_side": count_values(
                    tuple(item.side for item in identity_trades),
                    defaults=("buy", "sell"),
                ),
                "fees_paid": quantize_financial(
                    sum((item.fee_amount for item in identity_trades), start=Decimal(0))
                ),
                "estimated_slippage_cost": quantize_financial(
                    sum(
                        (estimated_slippage_cost(item) for item in identity_trades),
                        start=Decimal(0),
                    )
                ),
                "open_position_count": len(identity_open),
                "closed_position_count": len(identity_closed),
                "realized_gross_pnl": quantize_financial(
                    sum(
                        (item.gross_realized_pnl for item in identity_closed),
                        start=Decimal(0),
                    )
                ),
                "realized_net_pnl": quantize_financial(
                    sum(
                        (item.net_realized_pnl for item in identity_closed),
                        start=Decimal(0),
                    )
                ),
                "unrealized_pnl": (
                    quantize_financial(
                        sum(
                            (item.unrealized_pnl for item in identity_open),
                            start=Decimal(0),
                        )
                    )
                    if identity_open
                    else None
                ),
            }
        )
    return result


def calculate_benchmark(
    *,
    initial_equity: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    universe: tuple[MarketIdentity, ...],
    candles: dict[MarketIdentity, BenchmarkCandles],
) -> dict[str, object]:
    initial = quantize_financial(initial_equity)
    if not universe:
        return _unavailable_benchmark(
            EvaluationUnavailableReason.NO_DECISION_UNIVERSE, initial, []
        )
    allocation = (initial / Decimal(len(universe))).quantize(
        FINANCIAL_QUANTUM, rounding=ROUND_DOWN
    )
    instruments: list[dict[str, object]] = []
    spent = Decimal(0)
    ending_assets = Decimal(0)
    total_fees = Decimal(0)
    total_slippage = Decimal(0)
    complete = True
    for identity in sorted(universe):
        evidence = candles.get(identity, BenchmarkCandles(None, None))
        if evidence.entry is None or evidence.ending is None:
            complete = False
            instruments.append(
                {
                    **identity_fields(identity),
                    "status": EvaluationStatus.UNAVAILABLE,
                    "unavailable_reason": (
                        EvaluationUnavailableReason.MISSING_COMPATIBLE_CANDLE
                    ),
                    "allocation": allocation,
                    **{name: None for name in _BENCHMARK_VALUE_FIELDS},
                }
            )
            continue
        observed = quantize_financial(evidence.entry.close_price)
        execution = quantize_financial(observed * (Decimal(1) + slippage_rate))
        unit_cost = execution * (Decimal(1) + fee_rate)
        quantity = (allocation / unit_cost).quantize(
            FINANCIAL_QUANTUM, rounding=ROUND_DOWN
        )
        notional = quantize_financial(quantity * execution)
        fee = quantize_financial(notional * fee_rate)
        while notional + fee > allocation:
            quantity -= FINANCIAL_QUANTUM
            notional = quantize_financial(quantity * execution)
            fee = quantize_financial(notional * fee_rate)
        entry_cost = quantize_financial(notional + fee)
        ending_value = quantize_financial(quantity * evidence.ending.close_price)
        remainder = quantize_financial(allocation - entry_cost)
        pnl = quantize_financial(ending_value + remainder - allocation)
        return_value = quantize_financial(pnl / allocation) if allocation > 0 else None
        slippage = quantize_financial((execution - observed) * quantity)
        spent += entry_cost
        ending_assets += ending_value
        total_fees += fee
        total_slippage += slippage
        instruments.append(
            {
                **identity_fields(identity),
                "status": EvaluationStatus.AVAILABLE,
                "unavailable_reason": None,
                "allocation": allocation,
                "entry_candle_open_time": evidence.entry.open_time,
                "entry_observed_price": observed,
                "entry_execution_price": execution,
                "quantity": quantity,
                "entry_fee": fee,
                "estimated_entry_slippage_cost": slippage,
                "ending_candle_open_time": evidence.ending.open_time,
                "ending_mark_price": quantize_financial(evidence.ending.close_price),
                "ending_market_value": ending_value,
                "absolute_pnl": pnl,
                "total_return": return_value,
            }
        )
    if not complete:
        return _unavailable_benchmark(
            EvaluationUnavailableReason.INCOMPLETE_CANDLE_EVIDENCE,
            initial,
            instruments,
        )
    remaining_cash = quantize_financial(initial - spent)
    ending_equity = quantize_financial(ending_assets + remaining_cash)
    pnl = quantize_financial(ending_equity - initial)
    return {
        "status": EvaluationStatus.AVAILABLE,
        "unavailable_reason": None,
        "method": BenchmarkMethod.EQUAL_WEIGHT_BUY_AND_HOLD_V1,
        "instrument_count": len(universe),
        "initial_equity": initial,
        "ending_asset_value": quantize_financial(ending_assets),
        "remaining_cash": remaining_cash,
        "ending_equity": ending_equity,
        "absolute_pnl": pnl,
        "total_return": quantize_financial(pnl / initial) if initial > 0 else None,
        "entry_fees": quantize_financial(total_fees),
        "entry_slippage_cost": quantize_financial(total_slippage),
        "instruments": instruments,
    }


def comparison(
    strategy: dict[str, object], benchmark: dict[str, object]
) -> dict[str, object]:
    if (
        strategy["status"] != EvaluationStatus.AVAILABLE
        or benchmark["status"] != EvaluationStatus.AVAILABLE
    ):
        return {
            "status": EvaluationStatus.UNAVAILABLE,
            "unavailable_reason": (
                EvaluationUnavailableReason.STRATEGY_OR_BENCHMARK_UNAVAILABLE
            ),
            "strategy_excess_return": None,
            "strategy_minus_benchmark_pnl": None,
        }
    strategy_return = strategy["total_return"]
    benchmark_return = benchmark["total_return"]
    strategy_equity = strategy["ending_equity"]
    benchmark_equity = benchmark["ending_equity"]
    assert isinstance(strategy_return, Decimal)
    assert isinstance(benchmark_return, Decimal)
    assert isinstance(strategy_equity, Decimal)
    assert isinstance(benchmark_equity, Decimal)
    return {
        "status": EvaluationStatus.AVAILABLE,
        "unavailable_reason": None,
        "strategy_excess_return": quantize_financial(
            strategy_return - benchmark_return
        ),
        "strategy_minus_benchmark_pnl": quantize_financial(
            strategy_equity - benchmark_equity
        ),
    }


def identity_fields(identity: MarketIdentity) -> dict[str, str]:
    return {
        "asset": identity.asset,
        "provider": identity.provider,
        "market_type": identity.market_type,
        "interval": identity.interval,
    }


_BENCHMARK_VALUE_FIELDS = (
    "entry_candle_open_time",
    "entry_observed_price",
    "entry_execution_price",
    "quantity",
    "entry_fee",
    "estimated_entry_slippage_cost",
    "ending_candle_open_time",
    "ending_mark_price",
    "ending_market_value",
    "absolute_pnl",
    "total_return",
)


def _unavailable_benchmark(
    reason: EvaluationUnavailableReason,
    initial: Decimal,
    instruments: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "status": EvaluationStatus.UNAVAILABLE,
        "unavailable_reason": reason,
        "method": BenchmarkMethod.EQUAL_WEIGHT_BUY_AND_HOLD_V1,
        "instrument_count": len(instruments),
        "initial_equity": initial,
        "ending_asset_value": None,
        "remaining_cash": None,
        "ending_equity": None,
        "absolute_pnl": None,
        "total_return": None,
        "entry_fees": None,
        "entry_slippage_cost": None,
        "instruments": instruments,
    }
