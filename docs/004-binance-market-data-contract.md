# Binance Spot Kline Market-Data Contract

## Purpose

Argos designs its normalized market-data contract from the real external provider contract rather than assumptions:

```text
Binance external contract
→ Argos normalized contract
→ domain model
→ persistence model
→ analytical Features
```

This document defines the provider boundary and Argos retention decisions. It does not define Python classes or database tables.

Binance facts in this document come from the current official [Spot REST API market-data documentation](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market). Argos-specific rules are identified separately.

## Initial Endpoint

| Property | Contract |
| --- | --- |
| Method | `GET` |
| Path | `/api/v3/klines` |
| Provider and market | Binance Spot |
| Authentication | None; this is public market data and does not require an API key |
| Request weight | 2 |
| Binance data source | Database |
| Default limit | 500 |
| Maximum limit | 1000 |
| Kline identity | Open time |
| Argos interval | `5m` |
| Argos timezone | UTC |

Binance interprets `startTime` and `endTime` as Unix timestamps in milliseconds and always in UTC, regardless of `timeZone`. If neither is supplied, Binance returns the most recent klines.

### Request parameters

| Parameter | Meaning | MVP use | Argos usage or omission rationale |
| --- | --- | --- | --- |
| `symbol` | Required Binance trading-pair identifier. | Yes | Send `BTCUSDT` or `ETHUSDT`, mapped from the configured Argos Asset. |
| `interval` | Required, case-sensitive candle interval. | Yes | Always send `5m`, the accepted MVP interval. |
| `startTime` | Optional retrieval boundary in Unix milliseconds. | Yes | Supply an explicit UTC boundary for backfill, pagination, and incremental retrieval. Omit only for an intentional latest-klines request. |
| `endTime` | Optional retrieval boundary in Unix milliseconds. | Yes | Supply an explicit UTC upper boundary for bounded backfill and pagination. Omit only when retrieval intentionally continues through the latest available window. |
| `limit` | Maximum number of returned klines; default 500, maximum 1000. | Yes | Set explicitly for deterministic pagination, up to 1000. |
| `timeZone` | Changes how Binance interprets kline interval boundaries; default `0` (UTC). | No | Omit it. Argos uses UTC and must not request non-UTC candle boundaries. |

## Binance Response Contract

Binance returns each kline as a positional tuple in this order:

0. Open time
1. Open price
2. High price
3. Low price
4. Close price
5. Base asset volume
6. Close time
7. Quote asset volume
8. Number of trades
9. Taker buy base asset volume
10. Taker buy quote asset volume
11. Ignore

Prices and volumes are decimal strings. Timestamps are Unix milliseconds. Monetary and quantity values must use exact decimal semantics in the future implementation, not binary floating point.

## Field-by-Field Retention and Usage

| Binance index | Binance field | Argos normalized name | Source type | Meaning | MVP usage | Future analytical usage | Persist? | Persistence rationale |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Open time | `open_time` | Integer, Unix ms | Inclusive beginning of the candle window and Binance's documented kline identity. | Order candles; detect duplicates and gaps; associate Features and Scores with the correct market window; participate in uniqueness. | Replay decision cycles; support time-series partitioning and retention. | Yes | Required for identity, ordering, completeness checks, and reproducibility. |
| 1 | Open price | `open_price` | Decimal string | First traded price in the candle window. | OHLC calculations; candle direction and body size; returns and price-action Features. | Gap analysis and candlestick-pattern research. | Yes | Core measurable price input. |
| 2 | High price | `high_price` | Decimal string | Highest traded price in the candle window. | Candle range; volatility; ATR and true range; intraperiod maximum movement. | Stop-loss and take-profit simulation; breakout and resistance research. | Yes | Required for range and risk analysis. |
| 3 | Low price | `low_price` | Decimal string | Lowest traded price in the candle window. | Candle range; volatility; ATR and true range; intraperiod minimum movement. | Stop-loss and take-profit simulation; breakdown and support research. | Yes | Required for range and risk analysis. |
| 4 | Close price | `close_price` | Decimal string | Last traded price represented by the candle; it is final only after the candle closes. | Returns; moving averages; RSI; MACD; Momentum; closed-candle valuation; buy-and-hold comparison. | Model labels and historical Strategy evaluation. | Yes | Primary input for closed-candle price analysis. |
| 5 | Base asset volume | `base_asset_volume` | Decimal string | Traded quantity in the base asset: BTC for BTC/USDT and ETH for ETH/USDT. | Volume change; relative activity; price-move confirmation; within-Asset liquidity/activity Features. | Volume anomaly and regime analysis. | Yes | Preserves native-quantity activity. |
| 6 | Close time | `close_time` | Integer, Unix ms | End of the candle's temporal window. Binance REST provides no dedicated `is_closed` field. | Determine whether the window completed; prevent partial-candle analysis; validate interval duration; detect malformed timing. | Measure latency and audit ingestion. | Yes | Required for closed-candle eligibility and temporal validation. |
| 7 | Quote asset volume | `quote_asset_volume` | Decimal string | Traded notional in the quote asset, USDT for both MVP Assets. | Compare BTC/USDT and ETH/USDT activity on a common quote basis; measure notional; provide volume/liquidity context. | Approximate price-volume analysis and cross-asset normalization. | Yes | Not redundant with base volume: it expresses quote-currency economic value. |
| 8 | Number of trades | `trade_count` | Integer | Count of trades contributing to the candle. | Measure activity; distinguish many trades from fewer large trades; support average-trade-size Features. | Activity anomaly detection and market-regime classification. | Yes | Adds explainable transaction-count context to volume. |
| 9 | Taker buy base asset volume | `taker_buy_base_asset_volume` | Decimal string | Base-asset volume where the buyer was the taker. | Retained even if absent from the first Feature set so explainable buy-pressure and imbalance Features need no historical re-download. | Aggressive buyer participation, taker-buy ratio, and order-flow imbalance proxies. | Yes | Preserves native-asset aggressive-buy quantity. |
| 10 | Taker buy quote asset volume | `taker_buy_quote_asset_volume` | Decimal string | Quote-currency notional where the buyer was the taker. | Retained to measure aggressive buy-side USDT notional and compare BTC/USDT with ETH/USDT. | Quote-normalized taker-buy ratios and cross-asset pressure comparisons. | Yes | Complements index 9: one is native quantity, the other quote-currency economic value. |
| 11 | Ignore | None | String | Binance explicitly marks this field `Ignore` and gives it no semantic contract. | None. Validate that the tuple has 12 positions, then discard this value. | None. | No | Persisting an undocumented field would invent meaning and couple Argos unnecessarily to it. |

No retained field is assumed to have predictive value. Retention preserves measurable inputs whose value must be evaluated later.

## Context Added by Argos

Some normalized fields come from the request or ingestion context rather than each response tuple.

### Asset identity

The normalized concepts are base asset, quote asset, and canonical Argos Asset identifier:

- Binance `BTCUSDT` → Argos `BTC/USDT`
- Binance `ETHUSDT` → Argos `ETH/USDT`

Asset identity is retained because tuples do not repeat the symbol, every candle must belong to an Asset, and Asset participates in uniqueness and querying.

### Interval

The normalized name is `interval`; its initial value is `5m`. It is retained because the same Asset may later have multiple intervals, uniqueness and Feature interpretation depend on it, and stored history must not rely on an implicit global interval.

### Provider

The normalized name is `provider`; its initial value is Binance Spot. Provider provenance must be persisted or otherwise preserved so analysis is reproducible and provider discrepancies can be investigated. Whether it lives on every candle or a related source/import record remains an open persistence decision.

### Ingestion time

The normalized name is `ingested_at`. It measures ingestion delay, supports debugging and operational audits, and distinguishes market event time from processing time. It is Argos metadata, not Binance market data.

## Normalized Candle Contract

Argos conceptually retains:

- Asset identity;
- provider provenance;
- interval;
- open and close times;
- open, high, low, and close prices;
- base and quote asset volumes;
- trade count;
- taker buy base and quote asset volumes; and
- ingestion time.

This contract defines no IDs, SQL column types, indexes, ORM mappings, or table names. The likely natural uniqueness is Asset, Provider, Interval, and Open time. The final persistence constraint must be confirmed in the following domain/persistence PR.

## Validation and Normalization Rules

The future implementation must:

- require exactly 12 tuple positions and reject malformed tuples rather than shifting fields;
- parse prices and volumes with exact decimal semantics;
- convert Unix millisecond timestamps to timezone-aware UTC datetimes internally;
- reject negative prices, volumes, or trade counts;
- reject a high price below open or close, a low price above open or close, a high below low, or a close time before open time;
- accept only configured Assets and intervals;
- process only completed candles;
- make ingestion idempotent;
- detect duplicates and gaps;
- preserve the provider-field mapping in tests and documentation; and
- avoid persisting the entire raw tuple merely for convenience unless a future audit requirement justifies it.

Invalid provider records are rejected and logged for investigation. This contract does not invent correction behavior.

## Closed Candle Policy

Argos must never calculate authoritative Features, Scores, or Decisions from a partially formed candle. Binance REST klines can include the currently forming candle; merely receiving a REST kline does not guarantee closure.

For the MVP, a candle becomes analytically eligible only after its close time has passed, allowing for a small configurable ingestion-delay margin. The exact margin remains an implementation decision. The system may fetch an open candle but must not persist it as final analytical history or process it as closed.

A future WebSocket implementation may use Binance's explicit kline closure indicator. WebSocket ingestion is outside this contract and the immediate PR.

## Backfill and Incremental Retrieval

Historical backfill uses explicit UTC `startTime`, `endTime`, and `limit`. Requests paginate because Binance limits each response to 1000 candles.

Incremental ingestion starts after the latest successfully persisted closed candle. Re-fetching overlapping boundary candles is acceptable when persistence is idempotent. Exact retry, pagination, and scheduling algorithms remain implementation decisions.

## What This PR Does Not Decide

- Python DTOs or domain classes;
- SQLAlchemy entities;
- database table names;
- column types and precision;
- surrogate UUID versus natural database primary key;
- indexes and partitioning;
- raw-response archival;
- Binance client library versus direct HTTP;
- retry policy;
- rate-limit implementation;
- exact ingestion-delay margin;
- WebSocket ingestion; or
- first Feature formulas.
