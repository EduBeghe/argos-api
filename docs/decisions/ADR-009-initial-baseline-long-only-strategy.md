# ADR-009: Initial Baseline Long-Only Strategy

## Status

Accepted

## Context

Argos needs one deterministic, explainable Strategy to turn persisted analytical evidence into simulated Decision intent. The initial policy must be narrow enough to evaluate clearly, avoid repeated buying, and separate requested sizing from later paper execution. Its thresholds and allocation are initial hypotheses, not optimized values or profitability claims.

## Decision

Use Strategy `baseline_long_only`, strategy version 1, configuration version 1. It requires persisted `baseline_technical` model-version-1 Scores calculated from fully closed candles. The Strategy logic will be deterministic code with explicit versioned configuration constants; no generic rule engine or database-editable Strategy configuration is introduced.

The Strategy is long-only on Binance Spot, with no shorts or leverage and at most one open position per asset. Version 1 never increases an open position or sells partially. `INCREASE_POSITION` and `SELL_PARTIAL` remain defined future actions but are never emitted by this version.

Evaluate every 30 minutes as established by ADR-006. The eligible five-minute candle is the final closed candle for the corresponding UTC `:00` or `:30` boundary. Intermediate five-minute Scores do not automatically generate Decisions. The external Railway cron continues to run every five minutes; Argos introduces no internal scheduler.

When no position is open:

- `overall_score >= 0.60` produces `OPEN_POSITION`;
- `overall_score < 0.60` produces `NO_ACTION`.

When a position is open:

- `overall_score >= 0.20` produces `NO_ACTION`;
- `overall_score < 0.20` produces `SELL_ALL`.

Exactly `0.60` is eligible for entry. Exactly `0.20` keeps the position unchanged; only a value strictly below `0.20` exits. These different thresholds provide hysteresis: strong evidence is required to enter, while minor movement after entry does not force an exit. Continued qualifying Scores produce `NO_ACTION`, not repeated purchases. `SELL_ALL` is the complete exit operation.

`OPEN_POSITION` requests a target equal to 15% of current total portfolio equity, conceptually:

```text
available cash
+ current market value of all open positions
```

Maximum requested exposure is 15% per asset and 30% combined across BTC/USDT and ETH/USDT. A requested purchase must not exceed available cash. The 15% allocation is a conservative, versioned hypothesis for evaluation, not an optimized value.

A Decision records requested intent: target portfolio percentage for `OPEN_POSITION`, position percentage for `SELL_PARTIAL`, or 100% position liquidation for `SELL_ALL`. Later Paper Trading determines executed asset quantity, execution price, notional, fees, simulated slippage, and resulting cash and position changes after applying price, precision, and available-cash constraints. Requested percentages are not executed quantities.

`signal_strength` remains the absolute magnitude of `overall_score`; it is neither probability nor statistical confidence. Version 1 has no calibrated confidence value, so a future Decision Snapshot may leave confidence absent or null.

## Consequences

The initial Strategy is simple, reproducible, and directly testable across BTC/USDT and ETH/USDT. It cannot short, use leverage, increase positions, or partially sell. Individual trades may lose money. Thresholds and sizing must later be evaluated after fees and simulated slippage and compared with buy-and-hold. Stop-loss and take-profit rules, exact execution behavior, fee and slippage models, initial Paper Account cash, calibrated confidence, future Strategy types, and backtesting methodology remain open. No real-money execution is included.
