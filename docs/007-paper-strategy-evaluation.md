# Paper Strategy Evaluation v1

`GET /paper/evaluation` evaluates the active Paper Run. Supplying
`run_id=<uuid>` evaluates that active or stopped Run. The endpoint is read-only:
it does not contact a provider, generate analytical evidence, execute Trades,
or persist evaluation records.

## Authoritative boundary

The latest successfully persisted Equity Snapshot defines `evaluated_until`.
Strategy performance uses the complete Equity Snapshot history through that
boundary and therefore agrees with `GET /paper/performance`. Current portfolio
valuation and newer candles are not substituted. Without an Equity Snapshot,
Strategy, benchmark, and comparison results are unavailable with typed reasons;
no performance or benchmark is fabricated.

## Strategy metrics

- **Initial and ending equity:** persisted Run initial cash and latest snapshot
  equity.
- **Absolute P&L and total return:** ending equity minus initial equity, and that
  difference divided by positive initial equity.
- **Current and maximum drawdown:** the latest and worst non-negative declines
  from the preceding running equity peak.
- **Gross and net realized P&L:** sums of the authoritative values on Positions
  closed no later than the evaluation boundary.
- **Unrealized P&L:** sum of Position details stored in the latest Equity
  Snapshot. No newer candle valuation is performed.
- **Fees paid:** sum of persisted Trade `fee_amount` values through the boundary.
  No hypothetical exit fee is added for an open Position.
- **Estimated slippage cost:** for an open-position execution,
  `max(execution_price - observed_market_price, 0) × quantity`; for a sell-all execution,
  `max(observed_market_price - execution_price, 0) × quantity`.
- **Decision and Trade counts:** mappings preserve every persisted action and
  side. Decisions without Trades do not fabricate executions; `no_action` is a
  Decision count only.
- **Position counts:** open Positions come from the latest snapshot; closed
  Positions must have closed by the evaluation boundary.
- **Win rate:** winning closed Positions divided by all closed Positions.
- **Average win and loss:** arithmetic means of positive and negative persisted
  net realized P&L respectively.
- **Profit factor:** total positive net realized P&L divided by the absolute
  total negative net realized P&L.

Win rate is null with no closed Position. Average win or loss is null when its
population is empty. Profit factor is null when there is no loss; Argos does not
return Infinity, NaN, zero, or another sentinel for a zero denominator.

Per-instrument results use the complete `(asset, provider, market_type,
interval)` identity and deterministic ordering. BTC/USDT and ETH/USDT are only
the current configuration, not special evaluation fields.

## Equal-weight buy-and-hold benchmark

`equal_weight_buy_and_hold_v1` derives its universe from persisted Decision
Snapshots attributed to the selected Run through the evaluation boundary. It
does not use the current catalog as historical truth and does not hardcode
assets.

The benchmark divides initial cash equally across the complete universe. Each
allocation is an all-in cash budget. Entry uses the first compatible persisted
candle whose `close_time` is at or after Run start and no later than the
boundary. A candle may therefore open before the Run and still qualify when it
closes after the Run starts. Its close is the observed price; persisted Run
slippage and fee rates determine simulated entry execution, quantity, fee, and
slippage cost. Quantity is rounded down at
18 Decimal places so notional plus fee cannot exceed the allocation. Decimal
remainders remain benchmark cash.

Ending value uses the latest compatible persisted candle ordered by
`close_time` through the same boundary. No hypothetical exit fee or exit
slippage is deducted, matching the mark-to-market convention for open Paper
Positions. If any universe member
lacks required candle evidence, that instrument reports a typed reason and the
aggregate benchmark is unavailable. Argos never silently removes it or
redistributes its weight.

The comparison is available only when both Strategy and aggregate benchmark
are available at the same boundary. It reports Strategy return minus benchmark
return and Strategy ending equity minus benchmark ending equity. These values
are not called alpha.

## Limitations

The benchmark is fully invested and does not share the Strategy's exposure or
risk profile; it compares outcomes, not equal risk. Metrics are not annualized.
The endpoint does not calculate Sharpe ratio or claim alpha, guaranteed returns,
or profitability. All financial arithmetic uses `Decimal` and the project's
18-place half-even financial quantization policy except benchmark quantity,
which deliberately rounds down to respect its cash budget.
