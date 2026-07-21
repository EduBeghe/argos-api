from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

from app.market.catalog import Instrument
from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
    build_historical_feature_matrix,
)
from app.optimization.historical_data import HistoricalDataset
from app.paper.calculation import calculate_open_position, calculate_sell_all
from app.paper.config import (
    DEFAULT_ACCOUNT_CURRENCY,
    SimulationConfiguration,
    quantize_financial,
    validate_initial_cash,
)
from app.paper.models import PaperPosition, PaperPositionStatus, PaperTradeSide
from app.scores.calculation import MODEL_NAME, MODEL_VERSION
from app.scores.calculator import (
    BASELINE_TECHNICAL_V1_PARAMETERS,
    BaselineScoreInput,
    BaselineTechnicalParameters,
    calculate_baseline_score,
)
from app.strategies.baseline_long_only import (
    BASELINE_LONG_ONLY_V1_PARAMETERS,
    CONFIGURATION_VERSION,
    STRATEGY_NAME,
    STRATEGY_VERSION,
    BaselineLongOnlyParameters,
    StrategyAction,
    evaluate_baseline_long_only,
)

DEFAULT_DECISION_INTERVAL = timedelta(minutes=30)
SIMULATION_ENGINE_VERSION = 1
_ZERO_UUID = UUID(int=0)
DecisionActionFilter = Callable[
    [HistoricalFeatureRow, bool, StrategyAction], StrategyAction
]


class HistoricalSimulationError(ValueError):
    """Raised when historical evidence cannot support a simulation."""


@dataclass(frozen=True, slots=True)
class HistoricalSimulationTrade:
    boundary: datetime
    source_candle_open_time: datetime
    action: StrategyAction
    side: PaperTradeSide
    observed_price: Decimal
    execution_price: Decimal
    quantity: Decimal
    notional: Decimal
    fee: Decimal
    cash_before: Decimal
    cash_after: Decimal
    position_quantity_before: Decimal
    position_quantity_after: Decimal
    equity_before: Decimal
    equity_after: Decimal
    applied_rule: str
    explanation: str


@dataclass(frozen=True, slots=True)
class HistoricalEquityPoint:
    boundary: datetime
    source_candle_open_time: datetime
    observed_price: Decimal
    cash: Decimal
    position_quantity: Decimal
    equity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal


@dataclass(frozen=True, slots=True)
class HistoricalOpenPosition:
    quantity: Decimal
    average_entry_price: Decimal
    total_entry_notional: Decimal
    accumulated_entry_fees: Decimal
    cost_basis: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal


@dataclass(frozen=True, slots=True)
class HistoricalSimulationResult:
    dataset_fingerprint: str
    instrument: Instrument
    score_model_name: str
    score_model_version: int
    strategy_name: str
    strategy_version: int
    configuration_version: int
    account_currency: str
    fee_rate: Decimal
    slippage_rate: Decimal
    decision_interval: timedelta
    strategy_parameters: BaselineLongOnlyParameters
    score_parameters: BaselineTechnicalParameters
    initial_cash: Decimal
    final_cash: Decimal
    final_equity: Decimal
    absolute_pnl: Decimal
    total_return: Decimal
    peak_equity: Decimal
    lowest_equity: Decimal
    max_drawdown: Decimal
    fees_paid: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    trade_count: int
    buy_count: int
    partial_sell_count: int
    sell_all_count: int
    decision_count: int
    first_boundary: datetime
    last_boundary: datetime
    elapsed_market_period: timedelta
    final_open_position: HistoricalOpenPosition | None
    trades: tuple[HistoricalSimulationTrade, ...]
    equity_curve: tuple[HistoricalEquityPoint, ...]


def simulate_historical_strategy(
    *,
    dataset: HistoricalDataset,
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    score_model_name: str = MODEL_NAME,
    score_model_version: int = MODEL_VERSION,
    strategy_name: str = STRATEGY_NAME,
    strategy_version: int = STRATEGY_VERSION,
    configuration_version: int = CONFIGURATION_VERSION,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
    strategy_parameters: BaselineLongOnlyParameters = BASELINE_LONG_ONLY_V1_PARAMETERS,
    score_parameters: BaselineTechnicalParameters = BASELINE_TECHNICAL_V1_PARAMETERS,
) -> HistoricalSimulationResult:
    """Run baseline_long_only v1 deterministically against closed candle evidence."""

    matrix = build_historical_feature_matrix(dataset)
    return simulate_historical_feature_matrix(
        dataset=dataset,
        feature_matrix=matrix,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        score_model_name=score_model_name,
        score_model_version=score_model_version,
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        configuration_version=configuration_version,
        decision_interval=decision_interval,
        strategy_parameters=strategy_parameters,
        score_parameters=score_parameters,
    )


def simulate_historical_feature_matrix(
    *,
    dataset: HistoricalDataset,
    feature_matrix: HistoricalFeatureMatrix,
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    score_model_name: str = MODEL_NAME,
    score_model_version: int = MODEL_VERSION,
    strategy_name: str = STRATEGY_NAME,
    strategy_version: int = STRATEGY_VERSION,
    configuration_version: int = CONFIGURATION_VERSION,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
    strategy_parameters: BaselineLongOnlyParameters = BASELINE_LONG_ONLY_V1_PARAMETERS,
    score_parameters: BaselineTechnicalParameters = BASELINE_TECHNICAL_V1_PARAMETERS,
    decision_action_filter: DecisionActionFilter | None = None,
) -> HistoricalSimulationResult:
    """Simulate a precomputed Feature matrix without rebuilding it."""

    if score_model_name != MODEL_NAME or score_model_version != MODEL_VERSION:
        raise HistoricalSimulationError("unsupported historical score model")
    if strategy_name != STRATEGY_NAME or strategy_version != STRATEGY_VERSION:
        raise HistoricalSimulationError("unsupported historical strategy")
    if configuration_version != CONFIGURATION_VERSION:
        raise HistoricalSimulationError("unsupported strategy configuration")
    if decision_interval != DEFAULT_DECISION_INTERVAL:
        raise HistoricalSimulationError(
            "only the 30-minute decision interval is supported"
        )
    cash = validate_initial_cash(initial_cash)
    starting_cash = cash
    configuration = SimulationConfiguration(
        configuration_version,
        DEFAULT_ACCOUNT_CURRENCY,
        fee_rate,
        slippage_rate,
    )
    boundaries = tuple(
        (row.candle_open_time + timedelta(minutes=5), row)
        for row in feature_matrix.rows
        if _is_boundary(row.candle_open_time + timedelta(minutes=5), decision_interval)
    )
    if not boundaries:
        raise HistoricalSimulationError(
            "historical dataset has no complete Feature row at a decision boundary"
        )

    quantity = Decimal(0)
    average_entry_price = Decimal(0)
    entry_notional = Decimal(0)
    entry_fees = Decimal(0)
    fees_paid = Decimal(0)
    realized_pnl = Decimal(0)
    trades: list[HistoricalSimulationTrade] = []
    equity_curve: list[HistoricalEquityPoint] = []

    for boundary, row in boundaries:
        observed_price = row.close_price
        equity_before = _equity(cash, quantity, observed_price)
        quantity_before = quantity
        cash_before = cash
        score = calculate_baseline_score(_score_input(row), parameters=score_parameters)
        decision = evaluate_baseline_long_only(
            overall_score=score.overall_score,
            position_is_open=quantity > 0,
            parameters=strategy_parameters,
        )
        action = (
            decision_action_filter(row, quantity > 0, decision.action)
            if decision_action_filter is not None
            else decision.action
        )
        applied_rule = (
            "edge_guided_candidate"
            if action is not decision.action
            else decision.applied_rule
        )
        explanation = (
            "The frozen edge-guided bin changed the baseline action."
            if action is not decision.action
            else decision.explanation
        )
        if action is StrategyAction.OPEN_POSITION:
            calculation = calculate_open_position(
                cash=cash,
                total_equity=equity_before,
                observed_price=observed_price,
                target_percentage=decision.requested_target_portfolio_percentage
                or Decimal(0),
                configuration=configuration,
            )
            cash = calculation.resulting_cash
            quantity = calculation.quantity
            average_entry_price = calculation.execution_price
            entry_notional = calculation.executed_notional
            entry_fees = calculation.fee_amount
            fees_paid = quantize_financial(fees_paid + calculation.fee_amount)
            trades.append(
                _trade(
                    boundary=boundary,
                    row=row,
                    action=action,
                    side=calculation.side,
                    observed_price=observed_price,
                    execution_price=calculation.execution_price,
                    quantity=calculation.quantity,
                    notional=calculation.executed_notional,
                    fee=calculation.fee_amount,
                    cash_before=cash_before,
                    cash_after=cash,
                    quantity_before=quantity_before,
                    quantity_after=quantity,
                    equity_before=equity_before,
                    equity_after=_equity(cash, quantity, observed_price),
                    applied_rule=applied_rule,
                    explanation=explanation,
                )
            )
        elif action is StrategyAction.SELL_ALL:
            if quantity <= 0:
                raise HistoricalSimulationError(
                    "strategy attempted to sell without a position"
                )
            calculation = calculate_sell_all(
                cash=cash,
                position=_paper_position(
                    dataset.identity.instrument,
                    quantity,
                    average_entry_price,
                    entry_notional,
                    entry_fees,
                    boundary,
                ),
                observed_price=observed_price,
                configuration=configuration,
            )
            cash = calculation.resulting_cash
            quantity = Decimal(0)
            fees_paid = quantize_financial(fees_paid + calculation.fee_amount)
            realized_pnl = quantize_financial(
                realized_pnl + (calculation.net_realized_pnl or Decimal(0))
            )
            trades.append(
                _trade(
                    boundary=boundary,
                    row=row,
                    action=action,
                    side=calculation.side,
                    observed_price=observed_price,
                    execution_price=calculation.execution_price,
                    quantity=calculation.quantity,
                    notional=calculation.executed_notional,
                    fee=calculation.fee_amount,
                    cash_before=cash_before,
                    cash_after=cash,
                    quantity_before=quantity_before,
                    quantity_after=quantity,
                    equity_before=equity_before,
                    equity_after=_equity(cash, quantity, observed_price),
                    applied_rule=applied_rule,
                    explanation=explanation,
                )
            )
            average_entry_price = entry_notional = entry_fees = Decimal(0)

        market_value = quantize_financial(quantity * observed_price)
        cost_basis = quantize_financial(entry_notional + entry_fees)
        unrealized = (
            quantize_financial(market_value - cost_basis)
            if quantity > 0
            else Decimal(0)
        )
        equity_curve.append(
            HistoricalEquityPoint(
                boundary,
                row.candle_open_time,
                observed_price,
                cash,
                quantity,
                _equity(cash, quantity, observed_price),
                realized_pnl,
                unrealized,
            )
        )

    return _result(
        dataset=dataset,
        configuration=configuration,
        decision_interval=decision_interval,
        strategy_parameters=strategy_parameters,
        score_parameters=score_parameters,
        starting_cash=starting_cash,
        cash=cash,
        quantity=quantity,
        average_entry_price=average_entry_price,
        entry_notional=entry_notional,
        entry_fees=entry_fees,
        fees_paid=fees_paid,
        realized_pnl=realized_pnl,
        trades=tuple(trades),
        curve=tuple(equity_curve),
    )


def _is_boundary(value: datetime, interval: timedelta) -> bool:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    return (value - epoch) % interval == timedelta(0)


def _score_input(row: HistoricalFeatureRow) -> BaselineScoreInput:
    return BaselineScoreInput(
        row.close_price,
        row.ema_20,
        row.ema_50,
        row.rsi_14,
        row.atr_14,
        row.return_1,
        row.return_12,
        row.relative_volume_20,
    )


def _equity(cash: Decimal, quantity: Decimal, price: Decimal) -> Decimal:
    return quantize_financial(cash + quantity * price)


def _paper_position(
    instrument: Instrument,
    quantity: Decimal,
    average_entry_price: Decimal,
    entry_notional: Decimal,
    entry_fees: Decimal,
    boundary: datetime,
) -> PaperPosition:
    return PaperPosition(
        _ZERO_UUID,
        _ZERO_UUID,
        asset=instrument.asset,
        provider=instrument.provider,
        market_type=instrument.market_type,
        interval=instrument.interval,
        status=PaperPositionStatus.OPEN,
        quantity=quantity,
        average_entry_price=average_entry_price,
        total_entry_notional=entry_notional,
        accumulated_entry_fees=entry_fees,
        opened_at=boundary,
        closed_at=None,
        exit_price=None,
        gross_realized_pnl=None,
        net_realized_pnl=None,
    )


def _trade(
    *,
    boundary: datetime,
    row: HistoricalFeatureRow,
    action: StrategyAction,
    side: PaperTradeSide,
    observed_price: Decimal,
    execution_price: Decimal,
    quantity: Decimal,
    notional: Decimal,
    fee: Decimal,
    cash_before: Decimal,
    cash_after: Decimal,
    quantity_before: Decimal,
    quantity_after: Decimal,
    equity_before: Decimal,
    equity_after: Decimal,
    applied_rule: str,
    explanation: str,
) -> HistoricalSimulationTrade:
    return HistoricalSimulationTrade(
        boundary,
        row.candle_open_time,
        action,
        side,
        observed_price,
        execution_price,
        quantity,
        notional,
        fee,
        cash_before,
        cash_after,
        quantity_before,
        quantity_after,
        equity_before,
        equity_after,
        applied_rule,
        explanation,
    )


def _result(
    *,
    dataset: HistoricalDataset,
    configuration: SimulationConfiguration,
    decision_interval: timedelta,
    strategy_parameters: BaselineLongOnlyParameters,
    score_parameters: BaselineTechnicalParameters,
    starting_cash: Decimal,
    cash: Decimal,
    quantity: Decimal,
    average_entry_price: Decimal,
    entry_notional: Decimal,
    entry_fees: Decimal,
    fees_paid: Decimal,
    realized_pnl: Decimal,
    trades: tuple[HistoricalSimulationTrade, ...],
    curve: tuple[HistoricalEquityPoint, ...],
) -> HistoricalSimulationResult:
    final = curve[-1]
    equities = (starting_cash, *(point.equity for point in curve))
    peak = starting_cash
    max_drawdown = Decimal(0)
    for equity in equities[1:]:
        peak = max(peak, equity)
        drawdown = (
            quantize_financial((peak - equity) / peak) if peak > 0 else Decimal(0)
        )
        max_drawdown = max(max_drawdown, drawdown)
    absolute_pnl = quantize_financial(final.equity - starting_cash)
    market_value = quantize_financial(quantity * final.observed_price)
    cost_basis = quantize_financial(entry_notional + entry_fees)
    open_position = (
        HistoricalOpenPosition(
            quantity,
            average_entry_price,
            entry_notional,
            entry_fees,
            cost_basis,
            market_value,
            final.unrealized_pnl,
        )
        if quantity > 0
        else None
    )
    return HistoricalSimulationResult(
        dataset.fingerprint,
        dataset.identity.instrument,
        MODEL_NAME,
        MODEL_VERSION,
        STRATEGY_NAME,
        STRATEGY_VERSION,
        CONFIGURATION_VERSION,
        configuration.account_currency,
        configuration.fee_rate,
        configuration.slippage_rate,
        decision_interval,
        strategy_parameters,
        score_parameters,
        starting_cash,
        cash,
        final.equity,
        absolute_pnl,
        quantize_financial(absolute_pnl / starting_cash),
        max(equities),
        min(equities),
        max_drawdown,
        fees_paid,
        realized_pnl,
        final.unrealized_pnl,
        len(trades),
        sum(item.side is PaperTradeSide.BUY for item in trades),
        0,
        sum(item.action is StrategyAction.SELL_ALL for item in trades),
        len(curve),
        curve[0].boundary,
        curve[-1].boundary,
        curve[-1].boundary - curve[0].boundary,
        open_position,
        trades,
        curve,
    )
