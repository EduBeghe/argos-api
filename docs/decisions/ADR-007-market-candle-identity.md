# ADR-007: Market Candle Identity and Exact Decimal Storage

## Status

Accepted.

## Context

Market Candles need stable application identity, idempotent natural identity, exact decimal storage, and provenance. The Binance contract identifies klines by open time within a symbol and interval, while Argos must also distinguish provider and market type. Current BTC/USDT and ETH/USDT Spot filters require far less range and scale than a general exact-decimal representation.

## Decision

Each Market Candle receives a Python-generated UUIDv4 primary key. The database additionally enforces uniqueness across Asset, Provider, Market Type, Interval, and Open Time.

Prices and volumes use PostgreSQL `NUMERIC(38,18)`. This exact representation provides ample headroom beyond the current Binance Spot filters for the two MVP assets without embedding Binance-specific precision into the domain. Categorical values use strings with check constraints rather than PostgreSQL native enums so future supported values can be migrated without enum-type coupling.

## Consequences

- References can use a stable surrogate identifier while ingestion remains idempotent through the natural uniqueness constraint.
- Decimal arithmetic is exact within the declared precision and scale.
- Supporting values outside the current categorical scope requires a database migration.
- UUIDv4 is used because Python 3.13 does not provide a standard UUIDv7 generator; a future identity change would require a separate decision.
