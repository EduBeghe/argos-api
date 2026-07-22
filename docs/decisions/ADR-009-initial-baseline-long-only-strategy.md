# ADR-009: Initial Baseline Long-Only Strategy

## Status

Accepted

Implementation note: the formerly planned separate decision job and Paper Position integration described below are now implemented. The decision itself remains current; future-tense implementation notes are retained as historical context and are superseded by `docs/002-architecture.md` and `docs/010-current-system.md`.

## Context

Argos needs one deterministic, explainable Strategy to turn persisted analytical evidence into simulated Decision intent. The initial policy must be narrow enough to evaluate clearly, avoid repeated buying, and separate requested sizing from later paper execution. Its thresholds and allocation are initial hypotheses, not optimized values or profitability claims.

## Decision

Use Strategy `baseline_long_only`, strategy version 1, configuration version 1. It consumes normalized, persisted `baseline_technical` model-version-1 Scores calculated from fully closed candles. It does not require an aggregated Argos Score. The Strategy logic will be deterministic code with explicit versioned configuration constants; no generic rule engine or database-editable Strategy configuration is introduced.

The Strategy is provider-independent and long-only, with no shorts or leverage and at most one open position per asset. The current MVP evaluates it for BTC/USDT and ETH/USDT using normalized Binance Spot observations. It contains no BTC-, ETH-, USDT-, cryptocurrency-, Binance-, or provider-format-specific decision logic. Version 1 never increases an open position or sells partially. `INCREASE_POSITION` and `SELL_PARTIAL` remain defined future actions but are never emitted by this version.

Evaluate every 30 minutes as established by ADR-006. The eligible five-minute candle is the final closed candle for the corresponding UTC `:00` or `:30` boundary. Intermediate five-minute Scores do not automatically generate Decisions. The external Railway cron continues to run every five minutes; Argos introduces no internal scheduler.

Decision generation is a separate future Railway one-shot process, not another ingestion transaction. The planned `python -m app.jobs.generate_market_decisions` command must not be enabled until a persisted Paper Position adapter can supply real `position_is_open` state. Decision history cannot substitute for execution state, so the command fails closed rather than assuming no position is open.

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

Under strategy version 1 and configuration version 1, maximum requested exposure is 15% per asset and 30% combined across the currently configured BTC/USDT and ETH/USDT assets. A requested purchase must not exceed available cash. The 15% allocation is a conservative hypothesis, not an optimized or permanent value; the 30% limit is not a universal limit for future asset counts. Adding assets or changing allocation rules requires an explicit new configuration or Strategy version, as appropriate, so historical Decisions remain reproducible.

A Decision records requested intent: target portfolio percentage for `OPEN_POSITION`, position percentage for `SELL_PARTIAL`, or 100% position liquidation for `SELL_ALL`. Later Paper Trading determines executed asset quantity, execution price, notional, fees, simulated slippage, and resulting cash and position changes after applying price, precision, and available-cash constraints. Requested percentages are not executed quantities.

`signal_strength` remains the absolute magnitude of `overall_score`; it is neither probability nor statistical confidence. Version 1 has no calibrated confidence value, so a future Decision Snapshot may leave confidence absent or null.

## Consequences

The initial Strategy is simple, reproducible, provider-independent, and directly testable across the current BTC/USDT and ETH/USDT configuration. It cannot short, use leverage, increase positions, or partially sell. Future assets or providers require compatible ingestion adapters, market-calendar handling, Features, Scores, and versioned configuration, but do not redefine the core Decision vocabulary. Individual trades may lose money. Thresholds and sizing are now evaluated after versioned fees and simulated slippage and compared with buy-and-hold through Paper Trading and historical simulation. Stop-loss and take-profit rules, calibrated confidence, future Strategy types, and real-money execution remain outside the implemented scope. Current execution and evaluation behavior is documented in `docs/010-current-system.md`.
