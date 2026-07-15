# ADR-008: Use httpx Directly for Binance Spot HTTP Access

## Status

Accepted.

## Context

Argos needs one small asynchronous client for Binance Spot public market data. Introducing a Binance SDK, another HTTP library, or a generic provider abstraction would add dependencies and indirection without a current need.

## Decision

The Binance adapter uses `httpx.AsyncClient` directly. `BinanceClient` owns the client and returns provider-specific raw list data; it does not interpret positional fields or construct domain models. Semantic tuple validation and conversion belong to the Binance normalizer. Retries belong to future ingestion orchestration.

The future long-lived ingestion process should reuse one `BinanceClient` instance so the owned `AsyncClient` connection pool is effective.

## Consequences

- No Binance SDK or second HTTP library is introduced.
- Transport failures remain explicit and provider-specific.
- The client stays independent from normalization, domain, and persistence code.
- Ingestion orchestration will manage client lifetime and retry policy.
