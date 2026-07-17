# ADR-010: Versioned Paper Trading Simulation

## Status

Accepted.

## Context

Argos must evaluate decision quality without moving real money. A Decision Snapshot records requested intent; a Paper Trade records deterministic simulated execution. The assumptions must be explicit, reproducible, and replaceable rather than presented as guaranteed exchange costs.

## Decision

Paper simulation configuration version 1 starts each experiment with 10,000 USDT by default, applies a 0.10% fee, and applies 0.05% adverse slippage. A simulated buy executes at `observed price x (1 + 0.0005)` and a simulated sell at `observed price x (1 - 0.0005)`. Slippage means the simulation deliberately assumes a slightly worse price than the observed market price.

All financial calculations use `Decimal`, are quantized to 18 fractional digits, and use round-half-even. Inputs, rates, intermediate execution amounts, and results are persisted with each run or trade.

Resetting starts a new experiment, marks the prior run STOPPED, links its predecessor, and preserves historical Positions and Trades. STOPPED means an administrator ended the experiment; it does not mean open Positions were sold. Reset creates no synthetic Trades and leaves open Positions historically open-at-stop. Histories are queryable by run identity.

Portfolio equity uses the latest eligible persisted closed-candle price for each open Position at or before the Decision timestamp. Missing evidence fails closed; entry price is never a silent valuation fallback. Reset remains CLI-only until authenticated mutation endpoints exist, and a future authenticated frontend action must reuse the same use case. Assets, currencies, providers, and market types remain extension points.

## Consequences

Results are reproducible within configuration version 1 and can be compared between preserved runs. Reset requests are deliberately non-idempotent: each successful request creates a successor; callers needing safe retries will require a future explicit idempotency key. Row locking and the one-ACTIVE-run constraint prevent concurrent requests from leaving multiple active runs. These parameters are simulation hypotheses and may be replaced by later configuration versions. Paper Trading never contacts an exchange or moves real money.
