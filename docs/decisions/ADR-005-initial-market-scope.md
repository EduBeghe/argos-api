# ADR-005: Initial Market Scope

## Status

Accepted

## Context

Argos needs more than one asset to compare behavior and reduce the risk of drawing conclusions from a single market series. BTC/USDT and ETH/USDT provide strong liquidity, abundant historical data, and continuous 24/7 trading.

Restricting the MVP to one asset class and one provider avoids premature complexity from market hours, corporate actions, multiple provider contracts, and heterogeneous data semantics.

## Decision

Use cryptocurrency as the initial asset class, with BTC/USDT and ETH/USDT on the Binance Spot market. Binance Spot is the initial market and data provider.

The MVP excludes futures, leverage, funding rates, open interest, stocks, and ETFs. Argos remains asset-agnostic and provider-independent at the domain level. Binance-specific formats and behavior must remain isolated inside the Ingestion boundary.

## Consequences

The MVP can compare the same Features, Scores, Strategies, and Metrics across two liquid assets while keeping the first implementation operationally simple.

Results cannot yet be generalized to stocks, ETFs, futures, or leveraged products. Supporting another provider must require a new Ingestion adapter rather than changes throughout the analytical pipeline.

