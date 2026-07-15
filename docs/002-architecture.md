# Architecture Overview

Argos is an asset-agnostic quantitative research and predictive decision-support platform. Its initial backend is a modular monolith, not a distributed microservices system. Modules are separated by responsibility so they can evolve independently and, if a demonstrated operational need appears, later be extracted into independent services.

The frontend is a responsive Angular application maintained in the separate `argos-web` repository.

## High-Level Data Flow

```text
External data source
→ Ingestion
→ Storage
→ Feature Engine
→ Scoring
→ Decision Engine
→ Decision Snapshot
→ Paper Trading
→ Metrics
→ API
→ Web
```

The analytical pipeline runs independently of its presentation layers. The API exposes system information and actions; the Web application visualizes them and supports user interaction.

## Modules

### Ingestion

Ingestion connects to external data providers and retrieves raw market data. The MVP ingests five-minute Binance Spot candles and retains normalized price, volume, activity, temporal, interval, and provenance fields that have present or justified future analytical value. Later inputs may include trades, order book data, funding rates, open interest, news, and sentiment.

Its input is provider-specific data. Its output is validated, normalized data in Argos internal formats. Binance's positional tuple and all other provider-specific formats must end at this boundary. Ingestion does not calculate trading signals or make decisions.

### Storage

Storage persists and retrieves market history, calculated features and indicators, component scores, decisions, decision snapshots, paper trades, positions, strategy configuration, and performance metrics.

Its inputs are records produced by the application modules; its outputs are consistent, queryable historical records. Storage provides persistence and retrieval boundaries and contains no trading decision logic.

### Feature Engine

The Feature Engine transforms raw market data into measurable analytical variables. A feature is a measurable input used by rules, scoring models, or future machine-learning models.

It may calculate RSI, moving averages, MACD, ATR, volatility, momentum, volume changes, and returns over different windows. Its input is normalized market data; its output is a reusable, persisted feature set. It does not decide whether to buy or sell.

### Scoring

Scoring executes independent analytical models using available features and context. It produces explicit component scores such as technical, momentum, volatility, market context, news, sentiment, and machine-learning scores. Every result must be measurable and persist both its score and explanation. Scoring does not execute trades.

Illustrative output:

```text
technical: 0.65 — price is above the selected moving average
momentum: 0.40 — recent returns are positive but weakening
volatility: 0.25 — current volatility increases downside uncertainty
```

The score ranges, formulas, and explanations above are illustrative rather than selected model definitions.

### Decision Engine

The Decision Engine combines component scores with risk rules, portfolio state, current exposure, open positions, confidence thresholds, volatility, stop-loss rules, and take-profit rules. The final decision is not necessarily a simple average of component scores.

Its inputs are persisted evidence and account context. Its output is an explainable action such as `BUY`, `SELL`, `CLOSE_POSITION`, or `NO_ACTION`. `NO_ACTION` intentionally leaves the current portfolio unchanged without opening, closing, or modifying a position. It does not interact with a real exchange in the MVP.

### Decision Snapshot

A Decision Snapshot is an immutable historical record created whenever Argos generates a decision. It stores the timestamp, asset, price, market inputs, features, component scores, aggregated score, decision, confidence, applied rules, and model and configuration versions.

Snapshots allow the team to reconstruct why a decision was made and support debugging, model-version comparisons, audits, and later model evaluation. This module persists records; it is not an analytical engine.

### Paper Trading

Paper Trading is a virtual brokerage account that uses simulated money. It receives decisions from the Decision Engine, simulates order execution using market prices, applies simulated fees and slippage, and manages virtual cash, holdings, open positions, average entry price, and realized and unrealized P&L. It supports stop-loss, take-profit, and position closing. It never sends orders to a real exchange in the MVP.

Illustrative example: a simulated account starts with USD 1,000 and buys 0.25 ETH at USD 2,000. The notional purchase is USD 500; a USD 1 fee and USD 1 simulated slippage cost leave USD 498 in virtual cash. Selling 0.25 ETH later at USD 2,100 produces USD 525 before costs. After another USD 1 fee and USD 1 slippage cost, the account ends with USD 1,021, a net simulated profit of USD 21. The USD 1 simulated slippage on each side is a simplified illustrative execution cost, not the final slippage model. These costs are illustrative; the fee and slippage models remain open decisions.

### Metrics

Metrics evaluates the quality of Argos decisions and simulated trading performance. It calculates total return, realized P&L, unrealized P&L, maximum drawdown, win rate, profit factor, number of trades, average trade return, exposure, fees, slippage, and comparison against buy-and-hold.

This module consumes persisted decisions and paper-trading outcomes and produces evaluation evidence. It remains separate from decision generation. A high win rate alone does not prove that a strategy is profitable because losses, costs, exposure, and trade sizes also affect results.

### API

The API exposes system information and actions to the frontend. It may provide access to market history, current scores, score explanations, decisions, snapshots, simulated positions, simulated account state, performance metrics, and backtest execution.

The API orchestrates application use cases and maps them to explicit module interfaces. It does not contain core analytical or trading logic.

### Web

The Web module lives in the separate `argos-web` repository and is implemented as a responsive Angular dashboard. It displays price evolution, scores and their explanations, decisions, paper trades, virtual portfolio state, performance metrics, buy-and-hold comparison, strategy status, and process health.

The initial experience is optimized for desktop web use while remaining responsive. The browser does not calculate authoritative trading logic.

## Module Boundaries

- Each module has one primary responsibility.
- Analytical outputs are measurable and persisted where relevant.
- Modules communicate through explicit application interfaces.
- External provider-specific formats do not leak beyond Ingestion.
- Frontend concerns do not leak into analytical modules.
- Real-money trading is outside the MVP.

## Initial MVP Simplifications

The first implementation may use:

- BTC/USDT and ETH/USDT;
- Binance Spot;
- normalized five-minute candle data with retained price, volume, activity, temporal, interval, and provenance fields;
- a small initial feature set;
- a small number of explainable scores;
- one simulated account;
- one decision cycle;
- no real-money execution; and
- no news, sentiment, order book, or machine-learning integration until the base pipeline is measurable.

## Open Decisions

The following remain unresolved:

- first features;
- first component scores;
- score aggregation method;
- decision thresholds;
- simulated fee model;
- simulated slippage model;
- initial capital;
- backtesting approach;
- background job strategy; and
- deployment platform.

## Architectural Principle

> Argos does not attempt to predict the future with certainty. It estimates probabilities using measurable evidence, records why every decision was made, and evaluates those decisions through simulated results.
